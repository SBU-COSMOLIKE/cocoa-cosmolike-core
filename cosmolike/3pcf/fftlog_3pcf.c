#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_gamma.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "simde/x86/sse2.h"
#include "fftlog_3pcf.h"
#include "cosmolike/basics.h"
#include "log.c/src/log.h"

// Two-dimensional FFTLog for the double-Bessel projection integrals of
// the shear three-point transform. The algorithm is the two-Bessel
// 2D-FFTLog of Xiao Fang, in the form Sunao Sugiyama gives it in
// fastnc, MIT revision 7b2336ce37a909bed51f07f32a2a4f096e71b6ce,
// fastnc/twobessel.py. See LICENSE.fastnc-TreeCorr.txt for the
// distributed source notices.
//
// PHYSICAL DERIVATION & LOGIC FLOW
// The target is the projection integral
//
//   F(theta1,theta2) = integral dln(ell1) dln(ell2) f(ell1,ell2)
//                      J_a(ell1*theta1) J_b(ell2*theta2),
//
// where f is a real function on a logarithmic grid of Fourier
// multipoles ell1 and ell2, and J_a and J_b are cylindrical Bessel
// functions. The caller fills f; the three-point transform passes
// ell1^2 ell2^2 H_M, its coupled bispectrum-multipole kernel. Direct
// quadrature would fight two oscillating Bessel factors at every
// output point, so FFTLog evaluates the integral in four steps.
//
// 1. Divide out the bias. The FFTLog bias nu of each axis is a real
//    power-law tilt: dividing f by ell1^(nu1-1/2) ell2^(nu2-1/2)
//    leaves a function that decays toward both ends of the grid, which
//    the FFT can treat as periodic without a large jump at the seam.
//    The two axis rows of the bias table below hold these factors.
// 2. Expand in Fourier modes on the log grid. On a uniform grid in
//    ln(ell) the tilted function equals a finite sum of Fourier modes,
//    and each mode is a pure power law ell^(nu-1/2+i*eta), with eta
//    the frequency of oscillation in ln(ell). One forward FFT finds
//    all coefficients at once.
// 3. Integrate each power law analytically. A power law against one
//    Bessel function is a Mellin transform with a closed
//    Gamma-function form; kernel_fftlog_3pcf below derives it and
//    tabulates it per axis, Bessel order and frequency. The integral
//    of f becomes a double sum of coefficient times kernel times
//    kernel.
// 4. Reassemble with an inverse FFT. The kernel sum at every output
//    pair is itself a Fourier sum, so one inverse FFT evaluates it on
//    the radial grid theta = xy/reversed(ell): output node i carries
//    theta_i = xy/ell_(n-1-i), the reflected Fourier grid rescaled by
//    the constant xy, so that every product ell*theta stays on one
//    fixed logarithmic lattice.
//
// Both axes share one grid of length n; the natural-component
// transform only ever needs this square geometry. The input is real,
// so the FFT runs in FFTW's real-to-complex (r2c) layout: the second
// axis stores only its padded/2+1 nonnegative frequencies, because the
// missing negative-frequency coefficients are complex conjugates of
// the stored ones. The first axis keeps all of its rows, and a row
// index above padded/2 means a negative frequency. The row Nyquist
// frequency - the highest frequency the grid resolves, shared between
// +padded/2 and -padded/2 - appears once, and this module assigns it
// the positive sign exactly as fastnc does, so ported results agree
// with fastnc coefficient by coefficient.
//
// Cache invalidation:
// A workspace owns immutable copies of every transform setting and
// grid; create a new workspace after changing any of them. Each worker
// slice holds its own numerical input and output, and the workspace
// stores no cosmology. FFTW planning and destruction must stay serial.
// Concurrent executions go through FFTW's new-array calls, which reuse
// one plan on separate arrays with identical aligned strides.

// SIMD applies one operation to several numbers held in positions
// called lanes. A v2d holds two doubles. In this file the two lanes
// carry either two adjacent grid cells or the real and imaginary parts
// of one complex coefficient; each loop below says which. SIMDe maps
// the operations to the processor's vector instructions or to its
// portable implementation.
typedef simde__m128d v2d;

struct fftlog_workspace_3pcf {
  int n; // length of the physical grid on each axis
  int padded; // FFT length: n plus the zero padding on both ends
  int padding; // zero cells on one end of one axis
  int max_order; // largest Bessel order with a precomputed kernel row
  int workers; // number of independent worker slices
  size_t input_stride; // doubles per worker slice, rounded to 64 bytes
  size_t real_stride; // padded doubles per worker, rounded to 64 bytes
  size_t complex_stride; // coefficients per worker, rounded to 64 bytes
  double* input; // [worker,n,n] caller-filled f; also the inverse output
  double* real; // [worker,padded,padded], real FFT scratch
  fftw_complex* forward; // [worker,padded,padded/2+1], forward modes
  fftw_complex* scratch; // inverse coefficients, same shape, overwritten
  double* bias; // [2,n] axis-major ell^(1/2-nu): divides both tilts out
  double* output; // [2,n] axis-major theta^(1/2-nu): restores the tilts
  fftw_complex* kernel; // axis 0: [order,padded], all signed row
                        // frequencies; axis 1, following at offset
                        // (max_order+1)*padded: [order,padded/2+1],
                        // nonnegative frequencies
  fftw_plan forward_plan; // one real-to-complex recipe for all workers
  fftw_plan inverse_plan; // one complex-to-real recipe for all workers
  size_t selected_count; // kept output cells; 0 keeps the whole grid
  size_t* selected_offsets; // [selected_count], cells in the inverse FFT
  double* selected_first; // [selected_count], first-axis output factor
  double* selected_second; // [selected_count], second-axis output factor
};


// ---------------------------------------------------------------------------
// Smooth window for one Fourier coefficient of the forward transform.
//
// The FFT treats the tilted input as periodic in ln(ell), so any
// mismatch between the two ends of the finite grid acts like a sharp
// feature. Sharp features live in the highest log-frequencies, and
// keeping those coefficients at full weight spreads oscillatory
// artifacts - ringing - across the output. Tapering suppresses the
// highest-frequency coefficients smoothly instead: coefficients below
// the taper region pass unchanged and the window rolls off to zero at
// the Nyquist frequency. The shape is the FAST-PT smoothing window of
// McEwen, Fang, Hirata & Blazek (arXiv:1603.04826),
//
//   W(x) = x - sin(2*pi*x)/(2*pi),
//
// with x rising from 0 at the Nyquist frequency to 1 at the inner edge
// of the taper region; W joins both ends with zero slope. width sets
// the tapered fraction of the half spectrum. enabled=0 returns one for
// every frequency - the untapered upstream fastnc window - so
// comparison runs reproduce fastnc's coefficients exactly.
// ---------------------------------------------------------------------------
static double taper_fftlog_3pcf(
  const int frequency, // signed or nonnegative discrete frequency
  const int length, // even transform length
  const int enabled, // zero keeps the untapered upstream window
  const double width // fraction of the half-spectrum to taper
)
{
  if (!enabled) {
    return 1.0;
  }

  const int cutoff = (int)floor(width*length*0.5);
  const int absolute = abs(frequency);
  const int edge = length/2;
  if (cutoff == 0 || absolute <= edge-cutoff) {
    return 1.0;
  }
  if (cutoff == 1) {
    // A one-cell taper region has no interior to roll off across, so
    // only the Nyquist coefficient itself turns off.
    return 0.0;
  }

  // x measures how far this frequency sits from the Nyquist end of the
  // taper region, as a fraction of the region's width.
  const double fraction = (double)(edge-absolute)/(cutoff-1);
  const double two_pi = 2.0*M_PI;
  return fraction-sin(two_pi*fraction)/two_pi;
}


// ---------------------------------------------------------------------------
// One kernel table entry: the analytic Bessel integral of one Fourier
// mode.
//
// PHYSICAL DERIVATION & LOGIC FLOW
// After the forward FFT, each Fourier mode of the tilted input is a
// pure power law in one Fourier variable, ell^(nu-1/2+i*eta). Here nu
// is this axis's FFTLog bias - the real power-law tilt divided out of
// the integrand - and eta is the logarithmic Fourier frequency, the
// mode's rate of oscillation in ln(ell). Write the full power as
// s = nu - 1/2 + i*eta. Projecting one such mode on a cylindrical
// Bessel function J_n reduces, after substituting x = ell*theta, to a
// Mellin transform with the closed form
//
//   integral_0^inf dx x^(s-1) J_n(x)
//       = 2^(s-1) Gamma((n+s)/2) / Gamma((n+2-s)/2),
//
// the standard result that the 2D-FFTLog covariance paper of Fang,
// Eifler & Krause (arXiv:2004.04833) applies in this same role. The
// code works with exponent = nu + i*eta = s + 1/2 and multiplies by
// 2^exponent rather than 2^(s-1), so every entry carries an extra
// factor of 2^(3/2); fftlog_inverse_3pcf divides both axes' extra
// factors out again, following fastnc's normalization split. Both
// Gamma factors come from gsl_sf_lngamma_complex_e as log magnitude
// plus phase, so their ratio never overflows at large order. The
// phase -eta*phase_origin aligns the mode with the output grid:
// phase_origin equals log(ell_min*theta_min) on the padded grid, so
// the inverse FFT lands exactly on the radii theta = xy/reversed(ell).
//
// Radial bin averaging. With dlnt > 0 each output radius theta names
// the lower edge of a logarithmic bin of width dlnt, and the caller
// wants the average of the prediction over that bin rather than the
// value at its edge. One Fourier mode varies across the bin as the
// power law theta^(-s). Averaging with the polar area measure
// theta dtheta = theta^2 dln(theta) gives
//
//   <theta^(-s)> = theta^(-s) * S(2-s, dlnt) / S(2, dlnt),
//   S(z, w) = (exp(z*w) - 1) / z,
//
// because the measure adds two powers of theta to the numerator and
// to the normalizing bin area alike. In the code slope = 2 - s =
// 2.5 - exponent, expm1_complex (basics.c) builds the numerator of S
// without cancellation, and denominator holds S(2, dlnt) =
// expm1(2.0*dlnt)/2.0.
//
// Parameters name one Bessel order, one axis bias, one frequency and
// the shared phase origin and bin width; fftlog_create_3pcf calls
// this for every (order, frequency) pair once per workspace. Returns
// the complex kernel value and aborts on a nonfinite result.
// ---------------------------------------------------------------------------
static double complex kernel_fftlog_3pcf(
  const int order, // cylindrical Bessel order, nonnegative
  const double nu, // real bias of this axis
  const double eta, // logarithmic Fourier frequency
  const double phase_origin, // log(ell_min*theta_min)
  const double dlnt // logarithmic bin width, zero for point values
)
{
  const double complex exponent = nu+I*eta;
  gsl_sf_result numerator_log;
  gsl_sf_result numerator_phase;
  gsl_sf_result denominator_log;
  gsl_sf_result denominator_phase;

  const int status1 = gsl_sf_lngamma_complex_e((order-0.5+nu)*0.5,
      eta*0.5, &numerator_log, &numerator_phase);
  const int status2 = gsl_sf_lngamma_complex_e((order+2.5-nu)*0.5,
      -eta*0.5, &denominator_log, &denominator_phase);

  if (!(status1 == GSL_SUCCESS && status2 == GSL_SUCCESS)) {
    log_fatal("fftlog_3pcf: complex gamma evaluation failed");
    exit(1);
  }

  const double complex logarithm = exponent*log(2.0)
      +numerator_log.val-denominator_log.val
      +I*(numerator_phase.val-denominator_phase.val-eta*phase_origin);
  double complex value = cexp(logarithm);

  if (dlnt > 0.0) {
    // Multiply the bin-average factor S(2-s,dlnt)/S(2,dlnt) derived in
    // the header; slope holds 2 - s written through exponent = s + 1/2.
    const double complex slope = 2.5-exponent;
    const double denominator = expm1(2.0*dlnt)*0.5;
    value *= expm1_complex(slope*dlnt)/(slope*denominator);
  }

  if (!(isfinite(creal(value)) && isfinite(cimag(value)))) {
    log_fatal("fftlog_3pcf: nonfinite Mellin kernel");
    exit(1);
  }

  return value;
}


// ---------------------------------------------------------------------------
// Build one immutable FFTLog workspace: grids, kernel tables and FFTW
// plans.
//
// The constructor runs five stages, marked by banners below:
// validation, strides and allocation, bias tables, kernel tables, and
// FFTW plans. After it returns, the numerical entry points read the
// workspace without changing it, so many workers can share one
// workspace through their private slices.
//
// Inputs are described per parameter below; the file head explains the
// role each one plays in the transform. Returns the workspace; every
// failure aborts with a message rather than returning NULL.
// Ownership: the workspace copies all geometry, and
// fftlog_destroy_3pcf frees everything it allocated.
// Thread safety: creation must stay serial, because FFTW planning
// mutates the global planner state.
// ---------------------------------------------------------------------------
fftlog_workspace_3pcf* fftlog_create_3pcf(
  const int n, // original Fourier length
  const double* ell, // positive log-spaced Fourier grid
  const int max_order, // highest needed Bessel order
  const int workers, // private scratch count
  const double nu1, // first bias
  const double nu2, // second bias
  const double xy, // real/Fourier grid phase
  const double dlnt, // radial averaging width
  const int padding, // zero cells at each end
  const int window, // apply coefficient taper
  const double window_width // taper fraction
)
{
  // --- 1. VALIDATION ---
  // Reject any geometry the tables below cannot represent. The biases
  // must lie strictly inside (0.5, 2.5): at nu = 0.5 the kernel's
  // numerator Gamma factor hits its pole for Bessel order zero at zero
  // frequency, and at nu = 2.5 the zero-frequency bin-average divisor
  // slope = 2.5 - nu reaches zero. Inside the open interval every
  // kernel entry stays finite. The grid must be positive and uniformly
  // log-spaced, because the common step enters every kernel phase.
  if (!(n >= 2 && n%2 == 0 && ell != NULL
        && max_order >= 0 && max_order < INT_MAX-1 && workers > 0
        && padding >= 0 && padding <= (INT_MAX-n)/2)) {
    log_fatal("fftlog_3pcf: invalid dimensions or padding");
    exit(1);
  }
  if (!(isfinite(nu1) && nu1 > 0.5 && nu1 < 2.5
        && isfinite(nu2) && nu2 > 0.5 && nu2 < 2.5
        && isfinite(xy) && xy > 0.0 && isfinite(dlnt) && dlnt >= 0.0
        && isfinite(window_width) && window_width >= 0.0
        && window_width <= 1.0 && (window == 0 || window == 1))) {
    log_fatal("fftlog_3pcf: invalid bias, bin width, phase or window");
    exit(1);
  }
  for (int index=0; index<n; index++) {
    if (!(isfinite(ell[index]) && ell[index] > 0.0)) {
      log_fatal("fftlog_3pcf: ell must be positive and finite");
      exit(1);
    }
  }

  const double step = log(ell[1]/ell[0]);
  if (!(isfinite(step) && step > 0.0)) {
    log_fatal("fftlog_3pcf: ell must increase");
    exit(1);
  }

  // The tolerance 1.0e-10*step accepts grids a caller built by
  // exponentiating a uniform array, whose spacing wobbles at roundoff
  // level; a genuinely nonuniform grid still fails by a wide margin.
  for (int index=1; index<n; index++) {
    const double actual = log(ell[index]/ell[index-1]);
    if (!(fabs(actual-step) <= 1.0e-10*step)) {
      log_fatal("fftlog_3pcf: ell must be uniformly spaced in logarithm");
      exit(1);
    }
  }

  // --- 2. STRIDES AND ALLOCATION ---
  // Round each worker's slice up to a whole number of 64-byte cache
  // lines - 8 doubles, or 4 of the 16-byte fftw_complex values - so
  // the slices of different workers never share a cache line. The
  // overflow checks run before any multiplication that could wrap.
  fftlog_workspace_3pcf* work = calloc(1, sizeof(*work));
  if (!(work != NULL)) {
    log_fatal("fftlog_3pcf: workspace allocation failed");
    exit(1);
  }

  work->n = n;
  work->padded = n+2*padding;
  work->padding = padding;
  work->max_order = max_order;
  work->workers = workers;
  const size_t nfft = work->padded;
  const size_t nfrequency = nfft/2+1;
  if (!(nfft <= SIZE_MAX/nfft/sizeof(double))) {
    log_fatal("fftlog_3pcf: FFT array size overflow");
    exit(1);
  }

  work->input_stride = (((size_t)n*n+7)/8)*8;
  work->real_stride = ((nfft*nfft+7)/8)*8;
  work->complex_stride = ((nfft*nfrequency+3)/4)*4;
  if (!((size_t)workers <= SIZE_MAX/sizeof(double)/work->real_stride
        && (size_t)workers <= SIZE_MAX/sizeof(fftw_complex)
            /work->complex_stride
        && (size_t)(max_order+1)
            <= SIZE_MAX/sizeof(fftw_complex)/(nfft+nfrequency))) {
    log_fatal("fftlog_3pcf: worker or Mellin table size overflow");
    exit(1);
  }

  // FFTW's allocator returns memory aligned for its vector code. All
  // numeric blocks are flat, with an explicitly rounded worker stride,
  // so no hidden row-pointer padding exists anywhere in the workspace.
  work->input = fftw_malloc(sizeof(double)*work->input_stride*workers);
  work->real = fftw_malloc(sizeof(double)*work->real_stride*workers);
  work->forward = fftw_malloc(
      sizeof(fftw_complex)*work->complex_stride*workers);
  work->scratch = fftw_malloc(
      sizeof(fftw_complex)*work->complex_stride*workers);
  work->bias = fftw_malloc(sizeof(double)*2*n);
  work->output = fftw_malloc(sizeof(double)*2*n);
  work->kernel = fftw_malloc(
      sizeof(fftw_complex)*((size_t)(max_order+1)*(nfft+nfrequency)));
  if (!(work->input && work->real && work->forward && work->scratch
        && work->bias && work->output && work->kernel)) {
    log_fatal("fftlog_3pcf: allocation failed");
    exit(1);
  }

  // --- 3. BIAS TABLES ---
  // The bias rows divide the power-law tilt out of the input (file
  // head, step 1) and the output rows restore the matching powers
  // theta^(1/2-nu) on the output radii; axis 0 lives at offset 0 and
  // axis 1 at offset n of each table. Output node index points at
  // the reflected Fourier grid: node i carries theta = xy/ell[n-1-i].
  for (int index=0; index<n; index++) {
    work->bias[index] = pow(ell[index], 0.5-nu1);
    work->bias[n+index] = pow(ell[index], 0.5-nu2);
    const double theta = xy/ell[n-1-index];
    work->output[index] = pow(theta, 0.5-nu1);
    work->output[n+index] = pow(theta, 0.5-nu2);
  }

  // --- 4. KERNEL TABLES ---
  // Tabulate the analytic Bessel integral of every Fourier mode, for
  // every order up to max_order. The kernel table's axis 0 block
  // covers the first axis, whose rows carry all signed frequencies;
  // the axis 1 block, at offset axis1 inside the same table, covers
  // the second axis, which the r2c layout keeps at nonnegative
  // frequencies only.
  // frequency_step is the spacing of the discrete logarithmic
  // frequencies, 2*pi over the periodic length nfft*step, and origin
  // is the phase anchor log(ell_min*theta_min) of the padded grid.
  const double origin = log(xy)-(nfft-1)*step;
  const double frequency_step = 2.0*M_PI/(nfft*step);
  const size_t axis1 = (size_t)(max_order+1)*nfft;

  for (int order=0; order<=max_order; order++) {
    for (int index=0; index<work->padded; index++) {
      // Rows above padded/2 hold negative frequencies; the Nyquist row
      // itself takes the positive sign (file head).
      const int frequency = index <= work->padded/2
          ? index : index-work->padded;
      const double taper = taper_fftlog_3pcf(
          frequency, work->padded, window, window_width);
      const double eta = frequency*frequency_step;
      const size_t target = (size_t)order*nfft+index;
      if (order < 2) {
        work->kernel[target] = taper
            *kernel_fftlog_3pcf(order, nu1, eta, origin, dlnt);
      } else {
        // The recurrence Gamma(a+1) = a*Gamma(a), applied once in the
        // numerator and once in the denominator of the Mellin form,
        // advances the Bessel order by two:
        //
        //   K(n)/K(n-2) = (n-2+s)/(n-s), with s = exponent - 1/2.
        //
        // The phase, bin-average and taper factors do not depend on
        // the order and ride along unchanged from the order-2 seed,
        // so each parity needs only one direct GSL evaluation.
        const double complex exponent = nu1+I*eta;
        work->kernel[target] =
            work->kernel[(size_t)(order-2)*nfft+index]
            *(order-2.5+exponent)/(order+0.5-exponent);
      }
    }

    for (size_t index=0; index<nfrequency; index++) {
      const double taper = taper_fftlog_3pcf(
          (int)index, work->padded, window, window_width);
      const double eta = index*frequency_step;
      const size_t target = axis1+(size_t)order*nfrequency+index;
      if (nu1 == nu2) {
        // Equal biases make the two axes' kernels identical, so the
        // nonnegative frequencies copy the first axis's finished
        // values. The two axis blocks stay flat inside one table.
        work->kernel[target] = work->kernel[(size_t)order*nfft+index];
      } else if (order < 2) {
        work->kernel[target] = taper
            *kernel_fftlog_3pcf(order, nu2, eta, origin, dlnt);
      } else {
        // The same order-advancing recurrence as above, with the
        // second axis's bias in the exponent.
        const double complex exponent = nu2+I*eta;
        work->kernel[target] =
            work->kernel[axis1+(size_t)(order-2)*nfrequency+index]
            *(order-2.5+exponent)/(order+0.5-exponent);
      }
    }
  }

  // --- 5. FFTW PLANS ---
  // FFTW_ESTIMATE picks a reasonable recipe without the measurement
  // run that would overwrite the arrays. The execute calls later reuse
  // each recipe on matching aligned worker arrays through the
  // new-array interface.
  work->forward_plan = fftw_plan_dft_r2c_2d(work->padded, work->padded,
      work->real, work->forward, FFTW_ESTIMATE);
  work->inverse_plan = fftw_plan_dft_c2r_2d(work->padded, work->padded,
      work->scratch, work->real, FFTW_ESTIMATE);
  if (!(work->forward_plan != NULL
        && work->inverse_plan != NULL)) {
    log_fatal("fftlog_3pcf: FFTW planning failed");
    exit(1);
  }

  return work;
}


// Free every plan, table and slice of one workspace. NULL is a no-op,
// so error paths can call this unconditionally. Plan destruction must
// stay serial, like creation.
void fftlog_destroy_3pcf(
  fftlog_workspace_3pcf* work // owning workspace
)
{
  if (work == NULL) {
    return;
  }

  fftw_destroy_plan(work->forward_plan);
  fftw_destroy_plan(work->inverse_plan);

  fftw_free(work->input);
  fftw_free(work->real);
  fftw_free(work->forward);
  fftw_free(work->scratch);
  fftw_free(work->bias);
  fftw_free(work->output);
  fftw_free(work->kernel);
  fftw_free(work->selected_offsets);
  fftw_free(work->selected_first);
  fftw_free(work->selected_second);

  free(work);
}


// ---------------------------------------------------------------------------
// Keep only the radial nodes a measurement uses, once per workspace.
//
// The caller lists increasing positions on the output grid
// theta = xy/reversed(ell). For every pair of kept positions this
// stores the flat cell offset inside the padded inverse grid and the
// two tilt-restoring output factors, so each later inverse call can
// finish a compact [ntheta,ntheta] block instead of the full grid.
// Call before any forward or inverse execution; a second call aborts,
// because transforms in flight would read a changing table.
// ---------------------------------------------------------------------------
void fftlog_select_3pcf(
  fftlog_workspace_3pcf* work, // owning workspace, before any execution
  const int ntheta, // number of kept radial samples
  const int* indices // positions on xy/reversed(ell)
)
{
  if (!(work != NULL && ntheta > 0 && ntheta <= work->n
        && indices != NULL && work->selected_count == 0)) {
    log_fatal("fftlog_3pcf: invalid or repeated FFT output selection");
    exit(1);
  }
  for (int index=0; index<ntheta; index++) {
    if (!(indices[index] >= 0 && indices[index] < work->n
          && (index == 0 || indices[index] > indices[index-1]))) {
      log_fatal("fftlog_3pcf: FFT output indices must increase within the "
        "original grid");
      exit(1);
    }
  }

  work->selected_count = (size_t)ntheta*ntheta;
  if (!(work->selected_count <= SIZE_MAX/sizeof(double))) {
    log_fatal("fftlog_3pcf: allocation size overflow");
    exit(1);
  }

  work->selected_offsets =
      fftw_malloc(sizeof(size_t)*work->selected_count);
  work->selected_first = fftw_malloc(sizeof(double)*work->selected_count);
  work->selected_second = fftw_malloc(sizeof(double)*work->selected_count);
  if (!(work->selected_offsets && work->selected_first
        && work->selected_second)) {
    log_fatal("fftlog_3pcf: allocation failed");
    exit(1);
  }

  for (int first=0; first<ntheta; first++) {
    for (int second=0; second<ntheta; second++) {
      const size_t cell = (size_t)first*ntheta+second;
      work->selected_offsets[cell] =
          (size_t)(indices[first]+work->padding)*work->padded
          +indices[second]+work->padding;
      work->selected_first[cell] = work->output[indices[first]];
      work->selected_second[cell] =
          work->output[work->n+indices[second]];
    }
  }
}


// Return the start of one worker's private input slice. The caller
// fills it with f(ell1,ell2) in row-major [n,n] order before calling
// fftlog_forward_3pcf; after an inverse call the same memory holds
// that worker's finished output.
double* fftlog_input_3pcf(
  fftlog_workspace_3pcf* work, // owning workspace
  const int worker // requested private slice
)
{
  if (!(work != NULL && worker >= 0 && worker < work->workers)) {
    log_fatal("fftlog_3pcf: worker exceeds allocated scratch");
    exit(1);
  }

  return work->input+(size_t)worker*work->input_stride;
}


// ---------------------------------------------------------------------------
// Forward step: tilt the worker's grid and save its Fourier
// coefficients.
//
// The worker's input slice holds f(ell1,ell2). This function
// multiplies each cell by the bias factors ell1^(1/2-nu1) and
// ell2^(1/2-nu2) - the tilt division of the file head's step 1 -
// writes the result into the zero-padded real FFT buffer, and runs the
// real-to-complex FFT. The coefficients stay in the worker's forward
// slice, where fftlog_inverse_3pcf reads them once per Bessel-order
// pair, so the forward FFT runs once per filled grid rather than once
// per order pair.
// Thread safety: each worker touches only its own slices.
// ---------------------------------------------------------------------------
void fftlog_forward_3pcf(
  fftlog_workspace_3pcf* work, // workspace with an unweighted input grid
  const int worker // private slice
)
{
  const double* input = fftlog_input_3pcf(work, worker);
  double* real = work->real+(size_t)worker*work->real_stride;
  fftw_complex* forward =
      work->forward+(size_t)worker*work->complex_stride;

  // The fill loop below overwrites every cell inside the physical n*n
  // block of the FFT buffer. Without padding that block is the whole
  // transform, so only the alignment tail after it needs clearing.
  // With padding the border cells must also return to zero, because
  // the previous inverse transform wrote its full padded output here.
  const size_t clear_start = work->padding == 0 ? (size_t)work->n*work->n : 0;
  for (size_t index=clear_start; index<work->real_stride; index++) {
    real[index] = 0.0;
  }

  // Scalar meaning of one iteration, for row first and the column
  // pair second and second+1:
  //   real[first+padding][second+padding] =
  //       input[first][second]*bias[first]*bias[n+second];
  // and the same at second+1. The tilt division multiplies in because
  // the tables already hold the reciprocal powers ell^(1/2-nu). The
  // two column neighbors occupy the two SIMD lanes; the lanes never
  // interact, so vectorizing changes no arithmetic.
  for (int first=0; first<work->n; first++) {
    // Broadcast this row's factor ell1^(1/2-nu1) into both lanes.
    const v2d first_bias = simde_mm_set1_pd(work->bias[first]);
    // n is even - fftlog_create_3pcf enforces it - so column pairs
    // cover every row exactly and no scalar remainder is needed.
    for (int second=0; second<work->n; second+=2) {
      const size_t target =
          (size_t)(first+work->padding)*work->padded+second+work->padding;
      // Load f at columns second and second+1 of this row; loadu needs
      // two valid doubles but no vector-aligned address.
      const v2d values =
          simde_mm_loadu_pd(input+(size_t)first*work->n+second);
      // Multiply both cells by the shared row factor.
      const v2d first_product = simde_mm_mul_pd(values, first_bias);
      // Load the two column factors ell2^(1/2-nu2).
      const v2d second_bias =
          simde_mm_loadu_pd(work->bias+work->n+second);
      // Finish and store, walking the nested call inside out: mul
      // multiplies the row-weighted pair by the column factors, and
      // storeu writes lane 0 at target and lane 1 at target+1 inside
      // the padded FFT buffer.
      simde_mm_storeu_pd(real+target,
          simde_mm_mul_pd(first_product, second_bias));
    }
  }

  fftw_execute_dft_r2c(work->forward_plan, real, forward);
}


// ---------------------------------------------------------------------------
// Inverse step: combine the saved coefficients with two Bessel kernels
// and transform back to the radial grid.
//
// For one pair of Bessel orders the projection integral is the double
// sum, over all Fourier modes, of coefficient times first-axis kernel
// times second-axis kernel - and evaluating that sum on every radial
// pair is itself a Fourier transform, which one complex-to-real FFT
// performs. The loop below builds the combined coefficients, the FFT
// reassembles the grid, and the output stage applies the normalization
// together with the theta^(1/2-nu) factors that restore the tilt
// divided out on input.
//
// The wanted mode sum carries the opposite complex-exponential sign
// from FFTW's backward transform. Its value is real, because every
// negative frequency contributes the complex conjugate of its positive
// partner, so conjugating each combined coefficient flips the sign
// convention while leaving that real value unchanged; the plain c2r
// transform then produces the wanted sum. fastnc builds its inverse
// coefficients the same way.
//
// Returns the worker's private output: [n,n], or [ntheta,ntheta] after
// fftlog_select_3pcf. The next inverse on this worker overwrites it;
// the forward coefficients survive for further order pairs.
// Thread safety: workers share only the immutable kernel tables.
// ---------------------------------------------------------------------------
const double* fftlog_inverse_3pcf(
  fftlog_workspace_3pcf* work, // workspace with saved forward coefficients
  const int worker, // private slice
  const int order1, // first cylindrical Bessel order
  const int order2 // second cylindrical Bessel order
)
{
  double* output = fftlog_input_3pcf(work, worker);

  if (!(order1 >= 0 && order1 <= work->max_order
        && order2 >= 0 && order2 <= work->max_order)) {
    log_fatal("fftlog_3pcf: Bessel order exceeds prepared kernels");
    exit(1);
  }

  const size_t nfrequency = (size_t)work->padded/2+1;
  const size_t axis1 = (size_t)(work->max_order+1)*work->padded;
  const fftw_complex* forward =
      work->forward+(size_t)worker*work->complex_stride;
  fftw_complex* scratch =
      work->scratch+(size_t)worker*work->complex_stride;
  double* real = work->real+(size_t)worker*work->real_stride;

  // Scalar meaning for each coefficient c = forward[index] and the
  // combined kernel k = (axis 0 kernel at the row frequency)*(axis 1
  // kernel at the column frequency):
  //   scratch[index] = conj(c*k);
  // Rows carry the signed first-axis frequencies, so axis 0 reads at
  // the row index directly; columns hold the nonnegative second-axis
  // frequencies of the r2c layout. The SIMD lanes hold the real part
  // (lane 0) and imaginary part (lane 1) of one complex value - the
  // memory order of fftw_complex - and the block computes the complex
  // product from (a+ib)(p+iq) = (ap-bq) + i(bp+aq), then negates the
  // imaginary lane to conjugate.
  for (int first=0; first<work->padded; first++) {
    const double complex row_kernel =
        work->kernel[(size_t)order1*work->padded+first];
    for (size_t second=0; second<nfrequency; second++) {
      const size_t index = (size_t)first*nfrequency+second;
      const double complex factor = row_kernel
          *work->kernel[axis1+(size_t)order2*nfrequency+second];

      // Load c: lane 0 takes its real part, lane 1 its imaginary part.
      const v2d coefficient = simde_mm_loadu_pd((const double*)(forward+index));
      // Exchange the lanes: the shuffle immediate 1 picks element 1
      // for lane 0 and element 0 for lane 1, giving [Im c, Re c].
      const v2d swapped = simde_mm_shuffle_pd(coefficient, coefficient, 1);
      // Broadcast the real part of k to both lanes.
      const v2d real_factor = simde_mm_set1_pd(creal(factor));
      // set_pd lists lanes high first: lane 1 (the imaginary slot)
      // receives +Im k and lane 0 (the real slot) receives -Im k.
      const v2d imag_factor =
          simde_mm_set_pd(cimag(factor), -cimag(factor));
      // Multiply c by Re k in both lanes: [Re c*Re k, Im c*Re k].
      const v2d first_product = simde_mm_mul_pd(coefficient, real_factor);
      // Multiply the swapped lanes by the signed Im k, giving
      // [-Im c*Im k, +Re c*Im k].
      const v2d second_product = simde_mm_mul_pd(swapped, imag_factor);
      // Add the two halves: product = [Re(c*k), Im(c*k)].
      const v2d product = simde_mm_add_pd(first_product, second_product);
      // Conjugate, walking the nested call inside out: set_pd places
      // -1 in lane 1 and +1 in lane 0, and mul then flips the sign of
      // the imaginary lane only.
      const v2d conjugated =
          simde_mm_mul_pd(product, simde_mm_set_pd(-1.0, 1.0));
      // Store [Re, Im] of the finished inverse coefficient.
      simde_mm_storeu_pd((double*)(scratch+index), conjugated);
    }
  }

  fftw_execute_dft_c2r(work->inverse_plan, scratch, real);

  // Undo the two conventions the pipeline carried. FFTW scales neither
  // direction, so the forward and inverse transforms together multiply
  // every value by the cell count padded*padded. Each axis kernel also
  // carries a factor 2^(3/2) beyond the analytic Mellin integral (see
  // kernel_fftlog_3pcf), and the two axes contribute
  // 2^(3/2)*2^(3/2) = 8.
  const double normalization = 1.0/(8.0*work->padded*work->padded);

  if (work->selected_count > 0) {
    // A finite triangle measurement keeps only ntheta of the n radial
    // nodes per axis, so most cells of the padded inverse grid go
    // unread. This loop normalizes and tilt-restores just the kept
    // cells, through the offsets fftlog_select_3pcf stored, instead of
    // finishing a full n*n grid after every inverse.
    //
    // Scalar meaning per kept cell:
    //   output[cell] = normalization*real[selected_offsets[cell]]
    //       *selected_first[cell]*selected_second[cell];
    // Each SIMD lane carries one kept cell. The offsets jump around
    // the padded grid, so the two grid values arrive through two
    // scalar indexed loads rather than an architecture-specific
    // gather; the factor arrays are contiguous and load as pairs. The
    // multiplication order matches the full-grid route below, so both
    // routes round identically.
    const v2d factor = simde_mm_set1_pd(normalization);
    size_t cell = 0;
    for (; cell+1<work->selected_count; cell+=2) {
      // set_pd lists lanes high first: lane 1 takes the value of
      // cell+1 and lane 0 the value of cell.
      const v2d values = simde_mm_set_pd(
          real[work->selected_offsets[cell+1]],
          real[work->selected_offsets[cell]]);
      // Scale both cells by the shared normalization.
      const v2d normalized = simde_mm_mul_pd(factor, values);
      // Multiply in the first-axis factors theta1^(1/2-nu1); the
      // nested loadu reads the contiguous pair at cell and cell+1.
      const v2d first = simde_mm_mul_pd(normalized,
          simde_mm_loadu_pd(work->selected_first+cell));
      // Multiply in the second-axis factors theta2^(1/2-nu2), loaded
      // the same way.
      const v2d second = simde_mm_mul_pd(first,
          simde_mm_loadu_pd(work->selected_second+cell));
      // Store the finished pair into the worker's compact output.
      simde_mm_storeu_pd(output+cell, second);
    }
    // The pair loop stops one short when the kept-cell count is odd -
    // ntheta*ntheta is odd exactly when ntheta is - and this scalar
    // tail finishes the single remaining cell.
    for (; cell<work->selected_count; cell++) {
      output[cell] = normalization*real[work->selected_offsets[cell]]
          *work->selected_first[cell]*work->selected_second[cell];
    }
    return output;
  }

  // Full-grid output: normalize every cell of the physical n*n block
  // and restore the theta^(1/2-nu) powers on both axes, reading from
  // the padded inverse grid and writing the compact worker output.
  // The lanes hold two adjacent second-axis cells; n is even, so the
  // pairs cover each row exactly with no remainder.
  for (int first=0; first<work->n; first++) {
    // Broadcast the normalization into both lanes.
    const v2d factor = simde_mm_set1_pd(normalization);
    // Broadcast this row's factor theta1^(1/2-nu1).
    const v2d first_factor = simde_mm_set1_pd(work->output[first]);
    for (int second=0; second<work->n; second+=2) {
      const size_t source =
          (size_t)(first+work->padding)*work->padded+second+work->padding;
      // Normalize, walking the nested call inside out: loadu reads the
      // two adjacent inverse-FFT values at source and source+1, and
      // mul scales both by the normalization.
      const v2d normalized =
          simde_mm_mul_pd(factor, simde_mm_loadu_pd(real+source));
      // Multiply both cells by the shared row factor.
      const v2d first_product = simde_mm_mul_pd(normalized, first_factor);
      // Load the two column factors theta2^(1/2-nu2).
      const v2d second_factor =
          simde_mm_loadu_pd(work->output+work->n+second);
      // Finish and store, inside out again: mul applies the column
      // factors, and storeu writes the pair into the compact [n,n]
      // output at (first, second) and (first, second+1).
      simde_mm_storeu_pd(output+(size_t)first*work->n+second,
          simde_mm_mul_pd(first_product, second_factor));
    }
  }

  return output;
}
