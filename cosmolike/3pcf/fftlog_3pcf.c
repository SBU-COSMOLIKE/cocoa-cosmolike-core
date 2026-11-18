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

// Two-dimensional cylindrical FFTLog follows Xiao Fang's two-Bessel
// algorithm as modified by Sunao Sugiyama in fastnc, MIT revision
// 7b2336ce37a909bed51f07f32a2a4f096e71b6ce, fastnc/twobessel.py.
// See LICENSE.fastnc-TreeCorr.txt for the distributed source notices.
//
// PHYSICAL DERIVATION & LOGIC FLOW
// We compute integral dln(ell1) dln(ell2) f J_order1(ell1*theta1)
// J_order2(ell2*theta2). Divide f by ell1^(nu1-1/2) ell2^(nu2-1/2),
// Fourier-transform on the logarithmic grid, and integrate each resulting
// complex power law analytically with the Bessel functions. An inverse
// FFT reconstructs values on theta=xy/reversed(ell).
//
// The square-grid restriction is the natural-component path's actual
// geometry. Positive and negative row frequencies share one r2c buffer;
// no expanded conjugate matrix is needed. At the row Nyquist frequency
// use +N/2, matching fastnc's frequency rearrangement exactly.
//
// Cache invalidation:
// A workspace owns immutable copies of every transform setting and grid.
// Create another workspace after changing any of them. Numerical inputs
// and outputs live in private worker slices; no cosmology state is saved.
// FFTW planning and destruction are serial. Concurrent executions use the
// new-array FFTW calls with identical aligned strides and separate arrays.

typedef simde__m128d v2d;

struct fftlog_workspace_3pcf {
  int n; // original grid length
  int padded; // FFT length including both padded ends
  int padding; // zero cells on one end
  int max_order; // largest precomputed Bessel order
  int workers; // allocated independent worker slices
  size_t input_stride; // doubles per worker, rounded to 64 bytes
  size_t real_stride; // padded doubles per worker, rounded to 64 bytes
  size_t complex_stride; // coefficients per worker, rounded to 64 bytes
  double* input; // [worker,n,n], also receives normalized inverse output
  double* real; // [worker,padded,padded], FFT real scratch
  fftw_complex* forward; // [worker,padded,padded/2+1], saved coefficients
  fftw_complex* scratch; // matching inverse coefficients, overwritten
  double* bias1; // [n], ell^(1/2-nu1)
  double* bias2; // [n], ell^(1/2-nu2)
  double* output1; // [n], theta^(1/2-nu1)
  double* output2; // [n], theta^(1/2-nu2)
  fftw_complex* kernel1; // [order,padded], signed row frequencies
  fftw_complex* kernel2; // [order,padded/2+1], nonnegative frequencies
  fftw_plan forward_plan; // shared real-to-complex recipe
  fftw_plan inverse_plan; // shared complex-to-real recipe
  size_t selected_count; // zero means retain the entire original grid
  size_t* selected_offsets; // [selected_count], positions in inverse FFT
  double* selected_first; // [selected_count], first radial output factor
  double* selected_second; // [selected_count], second radial output factor
};


static void require_fftlog_3pcf(
  const int condition, // required geometry or allocation condition
  const char* message // diagnostic
)
{
  if (!condition) {
    fprintf(stderr, "fftlog_3pcf: %s\n", message);
    exit(EXIT_FAILURE);
  }
}


// FFTW supplies its required alignment. Numeric blocks are flat with an
// explicitly rounded worker stride; no hidden row-pointer padding exists.
static void* allocate_fftlog_3pcf(
  const size_t count, // number of scalar or complex elements
  const size_t size // bytes per element
)
{
  require_fftlog_3pcf(count <= SIZE_MAX/size, "allocation size overflow");
  void* result = fftw_malloc(count*size);
  require_fftlog_3pcf(result != NULL, "allocation failed");
  return result;
}


int next_fft_size_3pcf(
  const int minimum // minimum even smooth transform length
)
{
  require_fftlog_3pcf(minimum >= 2 && minimum < INT_MAX-2,
      "invalid minimum FFT length");
  int candidate = minimum+(minimum%2);
  for (; candidate < INT_MAX-2; candidate += 2) {
    int remainder = candidate;
    while (remainder%2 == 0) {
      remainder /= 2;
    }
    while (remainder%3 == 0) {
      remainder /= 3;
    }
    while (remainder%5 == 0) {
      remainder /= 5;
    }
    if (remainder == 1) {
      return candidate;
    }
  }
  require_fftlog_3pcf(0, "no representable FFT length");
  return 0;
}


static double complex expm1_complex_3pcf(
  const double complex value // exponent of a bin-average numerator
)
{
  // Subtracting one from exp(value) loses relative precision in a narrow
  // bin. The local power series retains that numerator without division.
  if (cabs(value) < 1.0e-4) {
    double complex term = value;
    double complex sum = value;
    for (int order=2; order<=8; order++) {
      term *= value/order;
      sum += term;
    }
    return sum;
  }
  return cexp(value)-1.0;
}


static double taper_fftlog_3pcf(
  const int frequency, // signed or nonnegative discrete frequency
  const int length, // even transform length
  const int enabled, // zero reproduces historical untapered coefficients
  const double width // fraction of the half-spectrum to taper
)
{
  if (!enabled) {
    return 1.0;
  }
  const int cutoff = (int)floor(width*length/2.0);
  const int absolute = abs(frequency);
  const int edge = length/2;
  if (cutoff == 0 || absolute <= edge-cutoff) {
    return 1.0;
  }
  if (cutoff == 1) {
    return 0.0;
  }
  // A one-cell cutoff above is the endpoint-only limit. For larger
  // cutoffs this is the original smooth Fourier coefficient window.
  const double fraction = (double)(edge-absolute)/(cutoff-1);
  const double two_pi = 2.0*acos(-1.0);
  return fraction-sin(two_pi*fraction)/two_pi;
}


static double complex kernel_fftlog_3pcf(
  const int order, // cylindrical Bessel order, nonnegative
  const double nu, // real bias
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
  const int status1 = gsl_sf_lngamma_complex_e((order-0.5+nu)/2.0,
      eta/2.0, &numerator_log, &numerator_phase);
  const int status2 = gsl_sf_lngamma_complex_e((order+2.5-nu)/2.0,
      -eta/2.0, &denominator_log, &denominator_phase);
  require_fftlog_3pcf(status1 == GSL_SUCCESS && status2 == GSL_SUCCESS,
      "complex gamma evaluation failed");
  const double complex logarithm = exponent*log(2.0)
      +numerator_log.val-denominator_log.val
      +I*(numerator_phase.val-denominator_phase.val-eta*phase_origin);
  double complex value = cexp(logarithm);
  if (dlnt > 0.0) {
    // Cylindrical radial averaging has measure theta*dtheta. The extra
    // half power comes from converting the spherical Mellin expression.
    const double complex slope = 2.5-exponent;
    const double denominator = expm1(2.0*dlnt)/2.0;
    value *= expm1_complex_3pcf(slope*dlnt)/(slope*denominator);
  }
  require_fftlog_3pcf(isfinite(creal(value)) && isfinite(cimag(value)),
      "nonfinite Mellin kernel");
  return value;
}


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
  require_fftlog_3pcf(n >= 2 && n%2 == 0 && ell != NULL
      && max_order >= 0 && max_order < INT_MAX-1 && workers > 0
      && padding >= 0 && padding <= (INT_MAX-n)/2,
      "invalid dimensions or padding");
  require_fftlog_3pcf(isfinite(nu1) && nu1 > 0.5 && nu1 < 2.5
      && isfinite(nu2) && nu2 > 0.5 && nu2 < 2.5
      && isfinite(xy) && xy > 0.0 && isfinite(dlnt) && dlnt >= 0.0
      && isfinite(window_width) && window_width >= 0.0
      && window_width <= 1.0 && (window == 0 || window == 1),
      "invalid bias, bin width, phase or window");
  for (int index=0; index<n; index++) {
    require_fftlog_3pcf(isfinite(ell[index]) && ell[index] > 0.0,
        "ell must be positive and finite");
  }
  const double step = log(ell[1]/ell[0]);
  require_fftlog_3pcf(isfinite(step) && step > 0.0,
      "ell must increase");
  for (int index=1; index<n; index++) {
    const double actual = log(ell[index]/ell[index-1]);
    require_fftlog_3pcf(fabs(actual-step) <= 1.0e-10*step,
        "ell must be uniformly spaced in logarithm");
  }
  fftlog_workspace_3pcf* work = calloc(1, sizeof(*work));
  require_fftlog_3pcf(work != NULL, "workspace allocation failed");
  work->n = n;
  work->padded = n+2*padding;
  work->padding = padding;
  work->max_order = max_order;
  work->workers = workers;
  const size_t nfft = work->padded;
  const size_t nfrequency = nfft/2+1;
  require_fftlog_3pcf(nfft <= SIZE_MAX/nfft/sizeof(double),
      "FFT array size overflow");
  work->input_stride = (((size_t)n*n+7)/8)*8;
  work->real_stride = ((nfft*nfft+7)/8)*8;
  work->complex_stride = ((nfft*nfrequency+3)/4)*4;
  require_fftlog_3pcf((size_t)workers <= SIZE_MAX/work->real_stride
      && (size_t)workers <= SIZE_MAX/work->complex_stride
      && (size_t)(max_order+1) <= SIZE_MAX/nfft,
      "worker or Mellin table size overflow");
  work->input = allocate_fftlog_3pcf(
      work->input_stride*workers, sizeof(double));
  work->real = allocate_fftlog_3pcf(
      work->real_stride*workers, sizeof(double));
  work->forward = allocate_fftlog_3pcf(
      work->complex_stride*workers, sizeof(fftw_complex));
  work->scratch = allocate_fftlog_3pcf(
      work->complex_stride*workers, sizeof(fftw_complex));
  work->bias1 = allocate_fftlog_3pcf(n, sizeof(double));
  work->bias2 = allocate_fftlog_3pcf(n, sizeof(double));
  work->output1 = allocate_fftlog_3pcf(n, sizeof(double));
  work->output2 = allocate_fftlog_3pcf(n, sizeof(double));
  work->kernel1 = allocate_fftlog_3pcf(
      (size_t)(max_order+1)*nfft, sizeof(fftw_complex));
  work->kernel2 = allocate_fftlog_3pcf(
      (size_t)(max_order+1)*nfrequency, sizeof(fftw_complex));
  for (int index=0; index<n; index++) {
    work->bias1[index] = pow(ell[index], 0.5-nu1);
    work->bias2[index] = pow(ell[index], 0.5-nu2);
    const double theta = xy/ell[n-1-index];
    work->output1[index] = pow(theta, 0.5-nu1);
    work->output2[index] = pow(theta, 0.5-nu2);
  }
  const double origin = log(xy)-(nfft-1)*step;
  const double frequency_step = 2.0*acos(-1.0)/(nfft*step);
  for (int order=0; order<=max_order; order++) {
    for (int index=0; index<work->padded; index++) {
      const int frequency = index <= work->padded/2
          ? index : index-work->padded;
      const double taper = taper_fftlog_3pcf(
          frequency, work->padded, window, window_width);
      const double eta = frequency*frequency_step;
      const size_t target = (size_t)order*nfft+index;
      if (order < 2) {
        work->kernel1[target] = taper
            *kernel_fftlog_3pcf(order, nu1, eta, origin, dlnt);
      } else {
        // Gamma(a+1)=a Gamma(a) advances the cylindrical Bessel order
        // by two. Phase, radial averaging and window factors do not
        // depend on that order, so each parity needs only one GSL seed.
        const double complex exponent = nu1+I*eta;
        work->kernel1[target] =
            work->kernel1[(size_t)(order-2)*nfft+index]
            *(order-2.5+exponent)/(order+0.5-exponent);
      }
    }
    for (size_t index=0; index<nfrequency; index++) {
      const double taper = taper_fftlog_3pcf(
          (int)index, work->padded, window, window_width);
      const double eta = index*frequency_step;
      const size_t target = (size_t)order*nfrequency+index;
      if (nu1 == nu2) {
        // Nonnegative frequencies share the first axis's prepared value
        // when both biases agree. The storage remains separate and flat.
        work->kernel2[target] = work->kernel1[(size_t)order*nfft+index];
      } else if (order < 2) {
        work->kernel2[target] = taper
            *kernel_fftlog_3pcf(order, nu2, eta, origin, dlnt);
      } else {
        const double complex exponent = nu2+I*eta;
        work->kernel2[target] =
            work->kernel2[(size_t)(order-2)*nfrequency+index]
            *(order-2.5+exponent)/(order+0.5-exponent);
      }
    }
  }
  // FFTW_ESTIMATE avoids a tuning run that overwrites prepared data. The
  // recipe is reused on matching aligned worker arrays by new-array calls.
  work->forward_plan = fftw_plan_dft_r2c_2d(work->padded, work->padded,
      work->real, work->forward, FFTW_ESTIMATE);
  work->inverse_plan = fftw_plan_dft_c2r_2d(work->padded, work->padded,
      work->scratch, work->real, FFTW_ESTIMATE);
  require_fftlog_3pcf(work->forward_plan != NULL
      && work->inverse_plan != NULL, "FFTW planning failed");
  return work;
}


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
  fftw_free(work->bias1);
  fftw_free(work->bias2);
  fftw_free(work->output1);
  fftw_free(work->output2);
  fftw_free(work->kernel1);
  fftw_free(work->kernel2);
  fftw_free(work->selected_offsets);
  fftw_free(work->selected_first);
  fftw_free(work->selected_second);
  free(work);
}


void fftlog_select_3pcf(
  fftlog_workspace_3pcf* work, // owning workspace, before any execution
  const int ntheta, // retained radial samples
  const int* indices // positions on xy/reversed(ell)
)
{
  require_fftlog_3pcf(work != NULL && ntheta > 0 && ntheta <= work->n
      && indices != NULL && work->selected_count == 0,
      "invalid or repeated FFT output selection");
  for (int index=0; index<ntheta; index++) {
    require_fftlog_3pcf(indices[index] >= 0 && indices[index] < work->n
        && (index == 0 || indices[index] > indices[index-1]),
        "FFT output indices must increase within the original grid");
  }
  work->selected_count = (size_t)ntheta*ntheta;
  work->selected_offsets = allocate_fftlog_3pcf(
      work->selected_count, sizeof(size_t));
  work->selected_first = allocate_fftlog_3pcf(
      work->selected_count, sizeof(double));
  work->selected_second = allocate_fftlog_3pcf(
      work->selected_count, sizeof(double));
  for (int first=0; first<ntheta; first++) {
    for (int second=0; second<ntheta; second++) {
      const size_t cell = (size_t)first*ntheta+second;
      work->selected_offsets[cell] =
          (size_t)(indices[first]+work->padding)*work->padded
          +indices[second]+work->padding;
      work->selected_first[cell] = work->output1[indices[first]];
      work->selected_second[cell] = work->output2[indices[second]];
    }
  }
}


double* fftlog_input_3pcf(
  fftlog_workspace_3pcf* work, // owning workspace
  const int worker // requested private slice
)
{
  require_fftlog_3pcf(work != NULL && worker >= 0 && worker < work->workers,
      "worker exceeds allocated scratch");
  return work->input+(size_t)worker*work->input_stride;
}


void fftlog_forward_3pcf(
  fftlog_workspace_3pcf* work, // workspace with an unweighted input grid
  const int worker // private slice
)
{
  const double* input = fftlog_input_3pcf(work, worker);
  double* real = work->real+(size_t)worker*work->real_stride;
  fftw_complex* forward =
      work->forward+(size_t)worker*work->complex_stride;
  // Clear every padded cell before copying the physical domain. Padding
  // from a previous inverse must never become part of a later spectrum.
  for (size_t index=0; index<work->real_stride; index++) {
    real[index] = 0.0;
  }
  for (int first=0; first<work->n; first++) {
    const v2d first_bias = simde_mm_set1_pd(work->bias1[first]);
    for (int second=0; second<work->n; second+=2) {
      const size_t target =
          (size_t)(first+work->padding)*work->padded+second+work->padding;
      const v2d values =
          simde_mm_loadu_pd(input+(size_t)first*work->n+second);
      const v2d first_product = simde_mm_mul_pd(values, first_bias);
      const v2d second_bias = simde_mm_loadu_pd(work->bias2+second);
      simde_mm_storeu_pd(real+target,
          simde_mm_mul_pd(first_product, second_bias));
    }
  }
  fftw_execute_dft_r2c(work->forward_plan, real, forward);
}


const double* fftlog_inverse_3pcf(
  fftlog_workspace_3pcf* work, // workspace with saved forward coefficients
  const int worker, // private slice
  const int order1, // first cylindrical Bessel order
  const int order2 // second cylindrical Bessel order
)
{
  double* output = fftlog_input_3pcf(work, worker);
  require_fftlog_3pcf(order1 >= 0 && order1 <= work->max_order
      && order2 >= 0 && order2 <= work->max_order,
      "Bessel order exceeds prepared kernels");
  const size_t nfrequency = (size_t)work->padded/2+1;
  const fftw_complex* forward =
      work->forward+(size_t)worker*work->complex_stride;
  fftw_complex* scratch =
      work->scratch+(size_t)worker*work->complex_stride;
  double* real = work->real+(size_t)worker*work->real_stride;
  for (int first=0; first<work->padded; first++) {
    const double complex kernel1 =
        work->kernel1[(size_t)order1*work->padded+first];
    for (size_t second=0; second<nfrequency; second++) {
      const size_t index = (size_t)first*nfrequency+second;
      const double complex factor = kernel1
          *work->kernel2[(size_t)order2*nfrequency+second];

      // SIMD lanes are real and imaginary parts of one coefficient.
      // (a+ib)(c+id)=(ac-bd)+i(bc+ad); the final sign conjugates it,
      // exactly as fastnc's inverse coefficient construction requires.
      const v2d coefficient = simde_mm_loadu_pd((const double*)(forward+index));
      const v2d swapped = simde_mm_shuffle_pd(coefficient, coefficient, 1);
      const v2d real_factor = simde_mm_set1_pd(creal(factor));
      const v2d imag_factor =
          simde_mm_set_pd(cimag(factor), -cimag(factor));
      const v2d first_product = simde_mm_mul_pd(coefficient, real_factor);
      const v2d second_product = simde_mm_mul_pd(swapped, imag_factor);
      const v2d product = simde_mm_add_pd(first_product, second_product);
      const v2d conjugated =
          simde_mm_mul_pd(product, simde_mm_set_pd(-1.0, 1.0));
      simde_mm_storeu_pd((double*)(scratch+index), conjugated);
    }
  }
  fftw_execute_dft_c2r(work->inverse_plan, scratch, real);
  const double normalization = 1.0/(8.0*work->padded*work->padded);
  if (work->selected_count > 0) {
    // Most FFT nodes are discarded by a finite triangle measurement.
    // Read its prepared offsets directly instead of normalizing/copying
    // a full n*n output after every inverse. Each SIMD lane is a retained
    // output; two scalar indexed loads avoid an architecture-specific
    // gather. The multiplication order matches the full-grid route.
    const v2d factor = simde_mm_set1_pd(normalization);
    size_t cell = 0;
    for (; cell+1<work->selected_count; cell+=2) {
      const v2d values = simde_mm_set_pd(
          real[work->selected_offsets[cell+1]],
          real[work->selected_offsets[cell]]);
      const v2d normalized = simde_mm_mul_pd(factor, values);
      const v2d first = simde_mm_mul_pd(normalized,
          simde_mm_loadu_pd(work->selected_first+cell));
      const v2d second = simde_mm_mul_pd(first,
          simde_mm_loadu_pd(work->selected_second+cell));
      simde_mm_storeu_pd(output+cell, second);
    }
    for (; cell<work->selected_count; cell++) {
      output[cell] = normalization*real[work->selected_offsets[cell]]
          *work->selected_first[cell]*work->selected_second[cell];
    }
    return output;
  }
  for (int first=0; first<work->n; first++) {
    const v2d factor = simde_mm_set1_pd(normalization);
    const v2d first_factor = simde_mm_set1_pd(work->output1[first]);
    for (int second=0; second<work->n; second+=2) {
      const size_t source =
          (size_t)(first+work->padding)*work->padded+second+work->padding;
      const v2d normalized =
          simde_mm_mul_pd(factor, simde_mm_loadu_pd(real+source));
      const v2d first_product = simde_mm_mul_pd(normalized, first_factor);
      const v2d second_factor = simde_mm_loadu_pd(work->output2+second);
      simde_mm_storeu_pd(output+(size_t)first*work->n+second,
          simde_mm_mul_pd(first_product, second_factor));
    }
  }
  return output;
}
