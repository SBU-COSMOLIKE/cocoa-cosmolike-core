#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_gamma.h>
#include "simde/x86/sse2.h"
#include "physics_3pcf.h"
#include "cosmolike/basics.h"
#include "log.c/src/log.h"
#ifdef _OPENMP
#include <omp.h>
#else
static int omp_in_parallel(void) { return 0; }
static int omp_get_max_threads(void) { return 1; }
static int omp_get_thread_num(void) { return 0; }
#endif

// ===========================================================================
// physics_3pcf.c: the physical bispectra of matter and intrinsic
// alignment, and their flat-sky Limber projection onto the observed
// angular bispectra.
// ===========================================================================
// A bispectrum measures how three Fourier amplitudes correlate when
// their wavevectors close into a triangle. Gravity couples Fourier
// modes as structure grows, so even initially Gaussian fluctuations
// develop a matter bispectrum. Intrinsic alignment (IA) adds a second
// effect: the tidal field that assembles a galaxy also aligns its
// shape, so observed shapes correlate with the matter density.
//
// The file works in three stages. The first stage copies the physical
// input tables and prepares fast interpolation readers. The second
// stage evaluates two fitted three-dimensional quantities: the
// BiHalofit matter bispectrum and the TATT (tidal alignment and tidal
// torquing) alignment bispectra, in fastnc's tensor formulation. The
// third stage integrates them along the line of sight (LOS): in the
// flat-sky Limber approximation, the angular bispectrum at multipoles
// (ell1, ell2, ell3) is a single integral over the comoving distance
// chi of three radial kernels against the three-dimensional bispectrum
// at k_i = ell_i/chi, without the +1/2 multipole shift.
//
// Provenance. The formulas and their software expression follow Sunao
// Sugiyama's MIT-licensed fastnc package, tatt commit
// 7b2336ce37a909bed51f07f32a2a4f096e71b6ce (halofit.py,
// ia_bispectra.py and bispectrum.py); Rafael Gomes wrote parts of the
// IA code. The matter fit is BiHalofit (Takahashi et al.,
// arXiv:1911.07886). The TATT and velocity-shear equations follow
// Gomes et al. (arXiv:2601.09133). This port evaluates fastnc's
// active tensor equations only. The package also ships direct-equation
// IA helpers that it never calls; their amplitude normalization
// differs from the active tensor path, and this file does not
// implement them.
//
// Units: the wave number k is in h/Mpc, the comoving distance chi in
// Mpc/h, the power spectrum P in (Mpc/h)^3 and the bispectrum B in
// (Mpc/h)^6. Lensing angles and the Fourier multipoles are
// dimensionless.
//
// Ownership. The model copies every input table at creation and
// normalizes the linear power spectrum to sigma8; after that it reads
// nothing from the caller. The production path resamples each copy
// onto a uniform grid that never changes, so a query finds its cell by
// index arithmetic and interpolates linearly. The one exception is the
// source-comparison mode, which the caller requests explicitly: it
// keeps cubic interpolation on the original tables, to match the
// upstream numbers.
//
// The model holds no file-scope mutable caches and builds nothing
// lazily, so it cannot follow a parameter change. Build a new model
// whenever the cosmology, the power spectra, the distances, the
// growth, the sources or the radial grids change. A change of IA
// parameters is cheaper: it needs only a new, small projection object.
//
// The source efficiency is the flat-space ratio (chi_s - chi)/chi_s,
// with chi_s the comoving distance to the source. The public model
// therefore requires spatial flatness, |Omega_m + Omega_de - 1| <=
// 1e-8. That tolerance absorbs floating-point rounding and nothing
// more: lensing with curved-space distances is not implemented.



// ===========================================================================
// [SECTION] MODEL STORAGE AND SPLINE TABLES
// ===========================================================================

// Two doubles side by side in one 128-bit register. The SIMD code
// below applies one operation to two independent values at a time;
// each value occupies a vector position called a lane.
typedef simde__m128d physics_v2d;

// One interpolation table. make_spline fills x, y and second for
// cubic evaluation on an irregular grid; uniformize_spline later
// replaces the table with a dense uniform grid and sets
// inverse_spacing, after which read_spline switches to linear reads
// with arithmetic cell indices. The power-spectrum tables store log P
// against log k, so their endpoint slopes are the power-law indices
// that read_power extrapolates with.
typedef struct {
  int n;                  // number of nodes
  double* x;              // node coordinates, strictly increasing [n]
  double* y;              // tabulated values at the nodes [n]
  double* second;         // second derivatives of the cubic fit [n]
  double inverse_spacing; // positive only for prepared uniform linear reads
  double slope_low;       // endpoint slope dy/dx of the first interval
  double slope_high;      // endpoint slope dy/dx of the last interval
} physics_spline;

// A redshift row names the one or two P(k) tables a power query reads
// and how to combine them. prepare_power_row builds it once per LOS
// node; every triangle side and source triplet then shares it, and no
// field changes during evaluation.
typedef struct {
  const physics_spline* lower; // P(k) table at or below the query z
  const physics_spline* upper; // next table above; NULL for D^2 scaling
  double fraction;             // interpolation weight toward upper
  double growth;               // growth factor D(z), applied as D^2
} physics_power_row;

// The redshift-dependent coefficients of the BiHalofit matter fit
// (Takahashi et al., arXiv:1911.07886). The names follow the paper's
// subscript-n symbols. With q = k*radius on each triangle leg, the
// one-halo term multiplies 1/[(a q^alpha + b q^beta)(1 + 1/(c q))]
// per leg; the three-halo term builds an enhanced power spectrum from
// f, g, h (its low-k factor) and m, n, mu, nu, p (its high-k
// addition), adds the d*q cross terms and damps each leg by
// 1/(1 + e q). The side ratios r1 = k_min/k_max and
// r2 = (k_mid + k_min - k_max)/k_max describe the triangle shape;
// matter_prepared applies the shape corrections that need them.
typedef struct {
  double radius;      // nonlinear radius 1/k_NL where sigma_G = 1, Mpc/h
  double loga;        // log10 a_n, one-halo amplitude before shape terms
  double b;           // b_n, one-halo coefficient of q^beta_n
  double c;           // c_n, one-halo low-k cutoff coefficient
  double logalpha;    // log10 alpha_n before its shape term and cap
  double alpha_shape; // coefficient of the r2^2 shape term in log alpha
  double logbeta;     // log10 beta_n before its 0.007 r2 shape term
  double gamma;       // gamma_n, exponent of the r1 term in log a_n
  double f;           // f_n, numerator of the low-k enhanced-power factor
  double g;           // g_n, linear-q denominator term of that factor
  double h;           // h_n, quadratic-q denominator term of that factor
  double m;           // m_n, first high-k addition coefficient
  double n;           // n_n, second high-k addition coefficient
  double mu;          // mu_n, exponent of the m_n q term
  double nu;          // nu_n, exponent of the n_n q term
  double p;           // p_n, low-k cutoff of the high-k addition
  double e;           // e_n, per-leg damping of the three-halo term
  double d;           // d_n, amplitude of the q3 cross terms
} physics_bh_coefficients;

// Dimensionless factors of one triangle shape. Rescaling all three
// sides by a common factor changes the physical k values but none of
// these numbers, so a projection prepares them once per angular
// triangle and reuses them at every LOS node. Frame convention: for
// cyclic ordering c the output wavevector is leg (c+2)%3, placed on
// the positive x axis, and triangle closure fixes the unit vectors of
// the two input legs c and (c+1)%3 in that frame.
typedef struct {
  double sides[3];        // side lengths over a common power of two
  double gravity[3];      // F2 density kernel of input legs c, c+1
  double pair_cosine[3];  // cosine between the wavevectors c and c+1
  double first_x[3];      // x component of the unit vector of leg c
  double first_y[3];      // y component of the unit vector of leg c
  double second_x[3];     // x component of the unit vector of leg c+1
  double second_y[3];     // y component of the unit vector of leg c+1
  double dot[3];          // dot product of the two input unit vectors
  double sine_squared[3]; // squared sine between the two input legs
  // The parenthesized IA vertex shapes of the fastnc tensor equations,
  // independent of power spectrum and redshift; the IA coefficients
  // multiply them later (the tensor helpers define the helicities).
  double ia_quadratic_shape[3];    // tidal-torque vertex shape factor
  double ia_density_zero_shape[3]; // density-weighting shape, helicity 0
  double ia_density_two_shape[3];  // density-weighting shape, helicity 2
  double minimum_ratio;   // r1 = k_min/k_max, BiHalofit shape ratio
  double flattened_ratio; // r2 = (k_mid + k_min - k_max)/k_max
} physics_triangle_geometry;

// The TATT vertex weights at one redshift, already combined into the
// factors the tensor equations read; ia_coefficients derives them
// from the public parameters.
typedef struct {
  double c1;        // linear tidal-alignment coefficient C_1(z)
  double linear;    // 2 C_1, the weight of a linear shape on one leg
  double velocity;  // -2 C_t/5, the velocity-shear vertex weight
  double quadratic; // C_2(z), the tidal-torque vertex weight
  double density;   // b_TA C_1 - 2 C_2/3, the density-weighting weight
} physics_ia_coefficients;

// The immutable model. Creation fills every member; afterwards all
// calls only read. Several members below belong to the NLA
// (nonlinear alignment) model, which ties galaxy shapes linearly to
// the tidal field; its source kernel is the normalized source density
// per unit comoving distance. A model without paired feedback tables
// is dark-matter-only (DMO).
struct physics_model_3pcf {
  baseline_3pcf* baseline; // supplied absolute B; NULL uses internal fit
  feedback_3pcf* feedback; // immutable paired matter response, NULL for DMO
  int nwindow;        // integer-ell pixel-window samples; zero disables
  double* window;     // owned integer-ell samples; NULL disables smoothing
  int nk;             // input wave-number nodes
  int nz;             // input background redshift nodes
  int nbin;           // tomographic source bins
  int nlos;           // line-of-sight integration nodes
  int kernel_nz;      // nodes of the uniform lensing-kernel grid
  int source_readers; // 1 keeps cubic reads of the original tables
  int table_boost;    // dense-grid refinement factor of uniform readers
  int fit_nz;         // BiHalofit coefficient anchors in production
  int npower_z;       // stored nonlinear P(k) rows (grows with the boost)
  int nsource;        // source-distribution redshift nodes
  double zmax;        // largest tabulated redshift
  double* power_z;    // redshifts of the nonlinear P(k) rows [npower_z]
  double omega_m;     // matter density fraction today
  double omega_de;    // dark-energy density fraction today
  double w0;          // dark-energy equation of state today
  double wa;          // its evolution, w(a) = w0 + wa (1 - a)
  double ns;          // primordial spectral index
  double sigma8;      // linear fluctuation amplitude at z = 0
  physics_spline linear;     // log P_linear(z=0) against log k
  physics_spline* nonlinear; // log P_nonlinear against log k [npower_z]
  physics_spline distance;   // comoving distance chi(z), Mpc/h
  physics_spline growth;     // linear growth D(z), D(0) = 1
  physics_spline dz_dchi;    // Jacobian dz/dchi (z), h/Mpc
  physics_spline radius;     // BiHalofit nonlinear radius 1/k_NL (z)
  physics_spline neff;       // effective spectral slope n_eff(z)
  physics_spline sigma8z;    // amplitude sigma8(z)
  physics_spline omega_m_z;  // matter fraction Omega_m(z)
  double* sources;         // source densities per unit z [nbin,nsource]
  double* source_z;        // source-grid redshifts [nsource]
  double* source_chi;      // distances at the source nodes [nsource]
  double* source_jacobian; // dz/dchi at the source nodes [nsource]
  double* kernel_z;        // uniform kernel-grid redshifts [kernel_nz]
  double* kernel_chi;      // distances at the kernel nodes [kernel_nz]
  double* kernel_growth;   // growth at the kernel nodes [kernel_nz]
  double* kernel_lens;     // lensing efficiency W_kappa [nbin,kernel_nz]
  double* kernel_shape; // NLA kernel over chi(1+z) [nbin,kernel_nz]
  double* los_z;        // LOS node redshifts [nlos]
  double* los_chi;      // LOS node comoving distances [nlos]
  double* los_dchi;     // trapezoid integration weights [nlos]
  double* los_lens;     // W_kappa resampled on LOS nodes [nbin,nlos]
  double* los_shape;    // NLA kernel resampled on LOS nodes [nbin,nlos]
  physics_bh_coefficients* los_bh; // BiHalofit fit per LOS node [nlos]
  physics_power_row* los_power; // [mode,nlos], fixed redshift brackets
};

struct physics_projection_3pcf {
  const physics_model_3pcf* model; // borrowed; must outlive the projection
  physics_ia_3pcf ia;              // copied IA parameter set
  double* weights; // [8,nlos]: matter baseline and seven residual terms
  physics_ia_coefficients* coefficients; // TATT weights per LOS node [nlos]
};

// A batch stores radial weights, not triangle-dependent predictions.
// The first projector supplies the shared IA coefficients and the
// model, so its owner must outlive the batch; the batch copies every
// projector's weights. Adjacent triplets occupy adjacent doubles, so
// the evaluation loop loads a pair with one unaligned SIMD read.
struct physics_projection_batch_3pcf {
  const physics_projection_3pcf* first; // borrowed first projector
  int nprojection;                      // ordered source triplets
  double* weights; // [nlos,8,nprojection], baseline followed by seven IA terms
};

static physics_power_row prepare_power_row(
  const physics_model_3pcf* model,
  double z,
  int mode
);
static void triangle_physics(double first, double second, double third);


// ---------------------------------------------------------------------------
// increasing_physics: require a finite, strictly increasing grid of
// at least two nodes; positive = 1 also requires every value to be
// positive.
static void increasing_physics(
  const double* values,
  const int count,
  const int positive
)
{
  if (!(values != NULL && count >= 2)) {
    log_fatal("physics_3pcf: missing grid");
    exit(1);
  }

  for (int index=0; index<count; index++) {
    if (!(isfinite(values[index]))) {
      log_fatal("physics_3pcf: nonfinite grid value");
      exit(1);
    }
    if (!(!positive || values[index] > 0.0)) {
      log_fatal("physics_3pcf: grid must be positive");
      exit(1);
    }
    if (!(index == 0 || values[index] > values[index - 1])) {
      log_fatal("physics_3pcf: grid must be strictly increasing");
      exit(1);
    }
  }
}


// ---------------------------------------------------------------------------
// make_spline: build a not-a-knot cubic spline on an irregular grid.
//
// A cubic spline stores the second derivative at every node; between
// nodes the interpolant is the cubic polynomial those endpoint values
// determine. Continuity of the first derivative at each interior node
// gives one equation per interior node,
//
//   h_left M[i-1] + 2 (h_left + h_right) M[i] + h_right M[i+1]
//     = 6 (slope_right - slope_left),
//
// with M the second derivatives, h the interval widths and slope the
// finite-difference slope of each interval. Two boundary conditions
// close the system. The not-a-knot choice of scipy's
// InterpolatedUnivariateSpline makes the first two intervals share
// one cubic polynomial, and likewise the last two; eliminating M[0]
// and M[n-1] folds that condition into the first and last interior
// rows and leaves a tridiagonal system. One forward elimination and
// one back substitution (the Thomas algorithm) solve it here, during
// preparation; evaluation then only reads and needs no mutable GSL
// accelerator object.
static physics_spline make_spline(
  const double* x,
  const double* y,
  const int count
)
{
  physics_spline result = {0};
  increasing_physics(x, count, 0);

  result.n = count;
  result.x = calloc1d(count);
  result.y = calloc1d(count);
  result.second = calloc1d(count);
  memcpy(result.x, x, (size_t)count*sizeof(double));
  memcpy(result.y, y, (size_t)count*sizeof(double));
  result.slope_low = (y[1] - y[0])/(x[1] - x[0]);
  result.slope_high = (y[count - 1] - y[count - 2])
    /(x[count - 1] - x[count - 2]);

  for (int index=0; index<count; index++) {
    if (!(isfinite(y[index]))) {
      log_fatal("physics_3pcf: nonfinite spline data");
      exit(1);
    }
  }

  if (count == 2) {
    return result;
  }
  if (count == 3) {
    const double slope0 = (y[1] - y[0])/(x[1] - x[0]);
    const double slope1 = (y[2] - y[1])/(x[2] - x[1]);
    const double second = 2.0*(slope1 - slope0)/(x[2] - x[0]);
    for (int index=0; index<count; index++) {
      result.second[index] = second;
    }
    return result;
  }

  const int interior = count - 2;
  double* diagonal = calloc1d(interior);
  double* upper = calloc1d(interior);
  double* rhs = calloc1d(interior);

  // Assemble the interior rows and eliminate on the fly: each row
  // folds its subdiagonal entry into the previous pivot, so one pass
  // both builds and forward-eliminates the tridiagonal system.
  for (int row=0; row<interior; row++) {
    const int index = row + 1;
    const double left = x[index] - x[index - 1];
    const double right = x[index + 1] - x[index];
    double lower = left;
    diagonal[row] = 2.0*(left + right);
    upper[row] = right;
    rhs[row] = 6.0*((y[index + 1] - y[index])/right
      - (y[index] - y[index - 1])/left);
    if (row == 0) {
      // First interior row: the not-a-knot condition expresses M[0]
      // through M[1] and M[2]; substituting it modifies this row's
      // diagonal and upper entries and removes its subdiagonal.
      diagonal[row] += left*(left + right)/right;
      upper[row] -= left*left/right;
      lower = 0.0;
    }
    if (row == interior - 1) {
      // Last interior row: the mirrored not-a-knot condition folds
      // M[n-1] into the two previous interior values the same way.
      diagonal[row] += right*(left + right)/left;
      lower -= right*right/left;
      upper[row] = 0.0;
    }
    if (row > 0) {
      // Forward elimination: subtract the multiple of the previous
      // row that zeroes this row's subdiagonal entry.
      const double multiplier = lower/diagonal[row - 1];
      diagonal[row] -= multiplier*upper[row - 1];
      rhs[row] -= multiplier*rhs[row - 1];
    }
  }

  // Back substitution fills the interior second derivatives from the
  // last row upward.
  for (int row=interior - 1; row>=0; row--) {
    const double next = row + 1 < interior ? result.second[row + 2] : 0;
    result.second[row + 1] = (rhs[row] - upper[row]*next)/diagonal[row];
  }

  // Recover the endpoint second derivatives from the not-a-knot
  // relations the elimination removed: each endpoint extrapolates
  // linearly from its two neighboring interior values.
  const double first = x[1] - x[0];
  const double second = x[2] - x[1];
  result.second[0] = ((first + second)*result.second[1]
    - first*result.second[2])/second;
  const double last = x[count - 1] - x[count - 2];
  const double previous = x[count - 2] - x[count - 3];
  result.second[count - 1] = ((last + previous)*result.second[count - 2]
    - last*result.second[count - 3])/previous;

  free(diagonal);
  free(upper);
  free(rhs);

  return result;
}


// ---------------------------------------------------------------------------
// destroy_spline: free the three node arrays and zero the struct.
static void destroy_spline(physics_spline* spline)
{
  free(spline->x);
  free(spline->y);
  free(spline->second);
  memset(spline, 0, sizeof(*spline));
}


// ---------------------------------------------------------------------------
// make_natural_uniform_spline: prepare the second derivatives with
// natural boundaries, M[0] = M[n-1] = 0, for a uniformly spaced grid.
//
// The production coefficient tables live on uniform anchors, and
// fastnc fits them with a natural cubic spline. The shared solver in
// basics.c supplies c = S''/2, whereas read_spline uses S''. Multiplying
// its coefficients by two keeps that reader's convention. There is no
// need to construct an irregular spline before this uniform solve.
static physics_spline make_natural_uniform_spline(
  const double* x, // uniform coordinates [count]
  const double* y, // finite function samples [count]
  const int count // number of nodes, at least two
)
{
  increasing_physics(x, count, 0);
  const double spacing = (x[count - 1] - x[0])/(count - 1);
  for (int index=0; index<count; index++) {
    if (!isfinite(y[index])) { log_fatal(
      "physics_3pcf: nonfinite uniform spline sample"); exit(1); }
    const double expected = x[0] + spacing*index;
    const double tolerance = 64.0*DBL_EPSILON
      *fmax(1.0, fmax(fabs(x[0]), fabs(x[count - 1])));
    if (fabs(x[index] - expected) > tolerance) { log_fatal(
      "physics_3pcf: natural spline coordinates must be uniform"); exit(1); }
  }

  physics_spline result = {0};
  result.n = count;
  result.x = calloc1d(count);
  result.y = calloc1d(count);
  result.second = calloc1d(count);
  memcpy(result.x, x, (size_t)count*sizeof(double));
  memcpy(result.y, y, (size_t)count*sizeof(double));
  result.slope_low = (y[1] - y[0])/(x[1] - x[0]);
  result.slope_high = (y[count - 1] - y[count - 2])
    /(x[count - 1] - x[count - 2]);
  spline_coeffs_uniform(result.y, count, spacing, result.second);
  for (int index=0; index<count; index++) {
    result.second[index] *= 2.0;
  }
  return result;
}


// ---------------------------------------------------------------------------
// interval_physics: binary search for the interval index i with
// grid[i] <= coordinate < grid[i+1], clamped to the valid range.
// Preparation and the source-comparison mode search; the production
// readers find their cell by index arithmetic on uniform grids.
static int interval_physics(
  const double* grid,
  const int count,
  const double coordinate
)
{
  int lower = 0;
  int upper = count - 1;
  while (upper - lower > 1) {
    const int middle = lower + (upper - lower)/2;
    if (grid[middle] <= coordinate) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return lower;
}


// ---------------------------------------------------------------------------
// read_spline: evaluate one spline. A uniform table (positive
// inverse_spacing) interpolates linearly inside the cell its index
// arithmetic selects and follows the edge segments outside the grid.
// An irregular table evaluates the cubic second-derivative form.
// zero_outside = 1 instead returns zero beyond the grid; the radial
// kernels use it so a query outside a source's support contributes
// nothing.
static double read_spline(
  const physics_spline* spline,
  const double coordinate,
  const int zero_outside
)
{
  if (zero_outside && (coordinate < spline->x[0]
      || coordinate > spline->x[spline->n - 1])) {
    return 0.0;
  }

  if (spline->inverse_spacing > 0.0) {
    const double position = (coordinate - spline->x[0])*spline->inverse_spacing;
    int index = (int)floor(fmax(0.0, fmin(spline->n - 2.0, position)));
    const double fraction = position - index;
    return spline->y[index]
      + fraction*(spline->y[index + 1] - spline->y[index]);
  }

  const int index = interval_physics(spline->x, spline->n, coordinate);
  const double width = spline->x[index + 1] - spline->x[index];
  const double right = (coordinate - spline->x[index])/width;
  const double left = 1.0 - right;
  return left*spline->y[index] + right*spline->y[index + 1]
    + width*width/6.0*((left*left*left - left)*spline->second[index]
    + (right*right*right - right)*spline->second[index + 1]);
}


// ---------------------------------------------------------------------------
// read_power: return P(k) from a log P(log k) table. Outside the
// tabulated Fourier support the reader keeps fastnc's convention and
// extends each end as a power law with the stored endpoint slope.
// That is an extrapolation of the physical model, not a statement of
// accuracy: the forecast must test the tabulated support against its
// aperture and scale cuts.
static double read_power(
  const physics_spline* spline,
  const double logk
)
{
  const int last = spline->n - 1;
  if (logk < spline->x[0] || logk > spline->x[last]) {
    const int index = logk < spline->x[0] ? 0 : last - 1;
    const double slope = logk < spline->x[0]
      ? spline->slope_low : spline->slope_high;
    const int endpoint = index == 0 ? 0 : last;
    return exp(spline->y[endpoint] + slope*(logk - spline->x[endpoint]));
  }

  return exp(read_spline(spline, logk, 0));
}


// ---------------------------------------------------------------------------
// uniform_storage: allocate an empty uniform table. CosmoLike's table
// strategy separates expensive preparation from repeated reads: cubic
// interpolation fills a dense uniform grid once, and workers then
// locate a cell by (x - xmin)/dx and linearly interpolate its two
// values. Input tables may be irregular; only the serial preparation
// searches their intervals, and the production reader above never
// searches a grid.
static physics_spline uniform_storage(
  const int count,
  const double lower,
  const double upper
)
{
  physics_spline result = {0};
  if (!(count >= 2 && upper > lower)) {
    log_fatal("physics_3pcf: invalid uniform table");
    exit(1);
  }

  result.n = count;
  result.x = calloc1d(count);
  result.y = calloc1d(count);
  result.inverse_spacing = (count - 1)/(upper - lower);

  for (int index=0; index<count; index++) {
    result.x[index] = lower + (upper - lower)*index/(count - 1);
  }
  result.x[count - 1] = upper;
  return result;
}


// ---------------------------------------------------------------------------
// boosted_count: the node count after refining each original interval
// into boost uniform pieces; the guard keeps the multiplication from
// overflowing int.
static int boosted_count(const int original, const int boost)
{
  if (!(boost >= 1 && original >= 2
        && original - 1 <= (INT_MAX - 1)/boost)) {
    log_fatal("physics_3pcf: dense table count overflows");
    exit(1);
  }
  return (original - 1)*boost + 1;
}


// ---------------------------------------------------------------------------
// uniformize_spline: replace an irregular cubic table with its dense
// uniform resampling. Cubic reads of the original table fill the new
// nodes once; afterwards the table serves linear arithmetic-index
// reads and keeps the original endpoint slopes for extrapolation.
static void uniformize_spline(physics_spline* spline, const int boost)
{
  const int count = boosted_count(spline->n, boost);
  physics_spline dense = uniform_storage(count, spline->x[0],
    spline->x[spline->n - 1]);
  dense.slope_low = spline->slope_low;
  dense.slope_high = spline->slope_high;

  for (int index=0; index<count; index++) {
    dense.y[index] = read_spline(spline, dense.x[index], 0);
  }

  destroy_spline(spline);
  *spline = dense;
}


// ---------------------------------------------------------------------------
// prepare_uniform_readers: convert every production table to the
// dense uniform representation. The background and BiHalofit splines
// refine in place. The nonlinear power needs more care: production
// interpolates between redshift rows, so this routine also builds a
// denser uniform redshift axis and fills each new row by splining
// every fixed-k column, and both endpoint slopes, along redshift.
// The source-comparison mode returns immediately and keeps cubic
// reads of the original tables.
static void prepare_uniform_readers(physics_model_3pcf* model)
{
  if (model->source_readers) {
    return;
  }

  const int boost = model->table_boost;
  uniformize_spline(&model->linear, boost);
  uniformize_spline(&model->growth, boost);
  if (NULL == model->baseline) {
    uniformize_spline(&model->radius, boost);
    uniformize_spline(&model->neff, boost);
    uniformize_spline(&model->sigma8z, boost);
  }
  uniformize_spline(&model->omega_m_z, boost);
  for (int redshift=0; redshift<model->npower_z; redshift++) {
    uniformize_spline(model->nonlinear + redshift, boost);
  }

  const int old_count = model->npower_z;
  const int new_count = boosted_count(old_count, boost);
  const int nk = model->nonlinear[0].n;

  physics_spline* dense = calloc((size_t)new_count,
    sizeof(physics_spline));
  if (dense == NULL) {
    log_fatal("physics_3pcf: allocation failed");
    exit(1);
  }

  double* new_z = calloc1d(new_count);
  for (int redshift=0; redshift<new_count; redshift++) {
    new_z[redshift] = model->zmax*redshift/(new_count - 1);
    dense[redshift] = uniform_storage(nk, model->nonlinear[0].x[0],
      model->nonlinear[0].x[nk - 1]);
  }

  double* column = calloc1d(old_count);
  // Two extra columns carry the endpoint power-law slopes. Extrapolation
  // therefore does not change when the interior log-k grid is refined.
  for (int wave=0; wave<nk + 2; wave++) {
    for (int redshift=0; redshift<old_count; redshift++) {
      if (wave < nk) {
        column[redshift] = model->nonlinear[redshift].y[wave];
      } else {
        column[redshift] = wave == nk ? model->nonlinear[redshift].slope_low
          : model->nonlinear[redshift].slope_high;
      }
    }
    physics_spline along_z = make_spline(model->power_z, column, old_count);
    for (int redshift=0; redshift<new_count; redshift++) {
      const double value = read_spline(&along_z, new_z[redshift], 0);
      if (wave < nk) {
        dense[redshift].y[wave] = value;
      } else if (wave == nk) {
        dense[redshift].slope_low = value;
      } else {
        dense[redshift].slope_high = value;
      }
    }
    destroy_spline(&along_z);
  }

  for (int redshift=0; redshift<old_count; redshift++) {
    destroy_spline(model->nonlinear + redshift);
  }
  free(model->nonlinear);
  free(model->power_z);
  free(column);

  model->nonlinear = dense;
  model->power_z = new_z;
  model->npower_z = new_count;
}


// ---------------------------------------------------------------------------
// linear_read: linear interpolation on an irregular grid, with zero
// outside the tabulated range. The radial preparation reads the
// source densities with it, so a node outside a source's support
// gets zero, and the linear rule matches fastnc's handling of those
// raw tables.
static double linear_read(
  const double* x,
  const double* y,
  const int count,
  const double coordinate
)
{
  if (coordinate < x[0] || coordinate > x[count - 1]) {
    return 0.0;
  }

  const int index = interval_physics(x, count, coordinate);
  const double fraction = (coordinate - x[index])/(x[index + 1] - x[index]);
  return y[index] + fraction*(y[index + 1] - y[index]);
}


// ---------------------------------------------------------------------------
// trapezoid: the trapezoid rule on an irregular grid. Each interval
// contributes its width times the average of its endpoint values,
// which integrates the straight line between the two samples exactly.
static double trapezoid(
  const double* x,
  const double* y,
  const int count
)
{
  double sum = 0.0;
  for (int index=0; index<count - 1; index++) {
    sum += 0.5*(x[index + 1] - x[index])*(y[index + 1] + y[index]);
  }
  return sum;
}


// ---------------------------------------------------------------------------
// simpson_uniform: composite Simpson integration on a uniform grid.
// Simpson's rule needs an odd node count, so for an even count the
// main loop covers the first count - 1 nodes and the final interval
// gets scipy's closing correction, due to Cartwright: fit a parabola
// through the last three nodes and integrate it over that one
// interval, which gives h*(5 f_n/12 + 2 f_(n-1)/3 - f_(n-2)/12).
static double simpson_uniform(
  const double* values,
  const int count,
  const double spacing
)
{
  const int odd_count = count%2 == 0 ? count - 1 : count;
  double sum = values[0] + values[odd_count - 1];
  for (int index=1; index<odd_count - 1; index++) {
    sum += (index%2 == 1 ? 4.0 : 2.0)*values[index];
  }
  sum *= spacing/3.0;

  if (count%2 == 0) {
    sum += spacing*(5.0*values[count - 1]/12.0
      + 2.0*values[count - 2]/3.0 - values[count - 3]/12.0);
  }

  return sum;
}



// ===========================================================================
// [SECTION] BIHALOFIT BACKGROUND QUANTITIES
// ===========================================================================

// variance_physics: the variance of linear density fluctuations
// smoothed on a radius, which fixes BiHalofit's background inputs.
//
// The dimensionless linear power Delta^2(k) = k^3 P_lin(k)/(2 pi^2)
// integrates against a squared smoothing window over log k:
//
//   sigma^2(R) = int dln k  Delta^2(k) W^2(k R) D^2(z),
//
// with D the growth factor. Window 0 is the spherical top hat, used
// for sigma8(z). Window 1 is the Gaussian exp(-x^2/2), whose square
// gives the Gaussian variance sigma_G^2(R) that defines BiHalofit's
// nonlinear radius through sigma_G = 1. Window 2 is x exp(-x^2/2):
// differentiating e^(-x^2) with respect to ln R brings down -2 x^2,
// so the effective slope n_eff = -3 - dln sigma_G^2/dln R equals
// -3 + 2*variance with window 2 at the radius where sigma_G = 1.
//
// A uniform Simpson grid covers log k up to log(100): fastnc samples
// the same 100 h/Mpc upper bound, and the optional analytic tail
// below accounts for everything above it. The 2000-entry cap bounds
// the fixed stack buffer; the callers use 1000 nodes for the
// variances and 2000 for the slope.
static double variance_physics(
  const physics_model_3pcf* model,
  const double radius,
  const double growth_squared,
  const double logk_min,
  const int count,
  const int window,
  const int tail
)
{
  double values[2000];
  if (!(count <= 2000 && count >= 3)) {
    log_fatal("physics_3pcf: invalid variance grid");
    exit(1);
  }

  const double logk_max = log(100.0);
  const double spacing = (logk_max - logk_min)/(count - 1);
  const double inverse_two_pi_squared = 0.05066059182116888572;

  for (int index=0; index<count; index++) {
    const double logk = logk_min + spacing*index;
    const double k = exp(logk);
    const double x = k*radius;
    // The top hat W(x) = 3 (sin x - x cos x)/x^3 loses its leading
    // digits to cancellation at small x; its Taylor series
    // 1 - x^2/10 + x^4/280 is exact to double precision below 1e-3.
    double filter;
    if (window == 0) {
      filter = fabs(x) < 1.0e-3
        ? 1.0 - x*x/10.0 + x*x*x*x/280.0
        : 3.0*(sin(x) - x*cos(x))/(x*x*x);
    } else {
      filter = exp(-0.5*x*x)*(window == 2 ? x : 1.0);
    }
    values[index] = read_power(&model->linear, logk)*k*k*k
      *inverse_two_pi_squared*growth_squared*filter*filter;
  }

  double sum = simpson_uniform(values, count, spacing);

  if (tail) {
    // Gaussian tail: account for the integral above k = 100 exactly
    // enough that the nonlinear-radius solve never feels the cutoff.
    // Approximate Delta^2(k) beyond the last node as the power law
    // A k^s, with s the log slope of the final interval. With the
    // substitution u = (k R)^2 the remaining integral becomes
    //
    //   int_{ln 100}^inf dln k  A k^s e^(-(k R)^2)
    //     = (A/2) R^(-s) Gamma(s/2, (100 R)^2),
    //
    // the upper incomplete gamma function that GSL evaluates below;
    // the growth factor squared multiplies at the end, as in the main
    // sum. A GSL underflow return only means the tail is zero to
    // double precision, so it passes the status check.
    const double logdelta_last = log(read_power(&model->linear, logk_max))
      + 3.0*logk_max;
    const double logdelta_previous = log(read_power(&model->linear,
      logk_max - spacing)) + 3.0*(logk_max - spacing);
    const double slope = (logdelta_last - logdelta_previous)/spacing;
    const double lower = 100.0*radius;
    gsl_sf_result upper_gamma;
    const int status = gsl_sf_gamma_inc_e(0.5*slope,
      lower*lower, &upper_gamma);
    if (!(status == GSL_SUCCESS || status == GSL_EUNDRFLW)) {
      log_fatal("physics_3pcf: Gaussian variance tail failed");
      exit(1);
    }
    sum += 0.5*exp(logdelta_last - slope*logk_max)
      *pow(radius, -slope)*upper_gamma.val
      *inverse_two_pi_squared*growth_squared;
  }

  if (!(isfinite(sum) && sum > 0.0)) {
    log_fatal("physics_3pcf: invalid variance");
    exit(1);
  }

  return sum;
}


// ---------------------------------------------------------------------------
// omega_m_physics: the matter density fraction at redshift z for a
// w0-wa dark-energy background.
//
// Integrating the continuity equation with the equation of state
// w(a) = w0 + wa (1 - a) gives the dark-energy density evolution
//
//   rho_de(a)/rho_de(1) = a^(-3 (1 + w0 + wa)) exp(-3 wa (1 - a)).
//
// Multiplying numerator and denominator of the usual a^-3 form of
// Omega_m(a) by a^2 puts the exact fraction into the shape the code
// uses,
//
//   Omega_m(a) = (Omega_m/a) / (Omega_m/a + Omega_de dark + Omega_k),
//
// where dark = a^2 rho_de(a)/rho_de(1) is the local variable below
// and Omega_k = 1 - Omega_m - Omega_de. The input densities may miss
// flatness by up to the 1e-8 rounding tolerance, so the code keeps
// the Omega_k term, written as the factor total; exactly flat inputs
// make total equal one exactly.
static double omega_m_physics(
  const physics_model_3pcf* model,
  const double z
)
{
  const double scale = 1.0/(1.0 + z);
  const double dark = pow(scale, -1.0 - 3.0*(model->w0 + model->wa))
    *exp(-3.0*(1.0 - scale)*model->wa);
  const double total = 1.0 + (model->omega_m + model->omega_de - 1.0)
    /(1.0 - model->omega_m - model->omega_de
    + model->omega_de*dark + model->omega_m/scale);
  return total*model->omega_m
    /(model->omega_m + model->omega_de*scale*dark);
}


// ---------------------------------------------------------------------------
// bh_coefficients: the redshift-dependent BiHalofit fit coefficients
// of Takahashi et al. (arXiv:1911.07886). The paper expresses each
// coefficient as ten to a polynomial in the effective spectral slope
// n_eff(z) and in log10 sigma8(z), with one Omega_m(z) term in the
// cross-term amplitude d_n; this function copies those polynomials
// digit for digit. The triangle-shape corrections to the one-halo
// coefficients need the side ratios, so matter_prepared applies them
// later; loga, logalpha and logbeta therefore stay in log10 form
// here.
static physics_bh_coefficients bh_coefficients(
  const physics_model_3pcf* model,
  const double z
)
{
  physics_bh_coefficients coefficients;
  const double spectral = read_spline(&model->neff, z, 0);
  const double spectral2 = spectral*spectral;
  const double spectral3 = spectral2*spectral;
  const double amplitude = log10(read_spline(&model->sigma8z, z, 0));
  const double amplitude2 = amplitude*amplitude;
  const double amplitude3 = amplitude2*amplitude;
  const double matter = read_spline(&model->omega_m_z, z, 0);
  coefficients.radius = read_spline(&model->radius, z, 0);

  coefficients.loga = -2.167 - 2.944*amplitude - 1.106*amplitude2
    - 2.865*amplitude3;
  coefficients.b = pow(10.0, -3.428 - 2.681*amplitude
    + 1.624*amplitude2 - 0.095*amplitude3);
  coefficients.c = pow(10.0, 0.159 - 1.107*spectral);
  coefficients.logalpha = -4.348 - 3.006*spectral - 0.5745*spectral2;
  coefficients.alpha_shape = pow(10.0, -0.9 + 0.2*spectral);
  coefficients.logbeta = -1.731 - 2.845*spectral - 1.4995*spectral2
    - 0.2811*spectral3;
  coefficients.gamma = pow(10.0, 0.182 + 0.57*spectral);
  coefficients.f = pow(10.0, -10.533 - 16.838*spectral
    - 9.3048*spectral2 - 1.8263*spectral3);
  coefficients.g = pow(10.0, 2.787 + 2.405*spectral + 0.4577*spectral2);
  coefficients.h = pow(10.0, -1.118 - 0.394*spectral);
  coefficients.m = pow(10.0, -2.605 - 2.434*amplitude + 5.71*amplitude2);
  coefficients.n = pow(10.0, -4.468 - 3.08*amplitude + 1.035*amplitude2);
  coefficients.mu = pow(10.0, 15.312 + 22.977*spectral
    + 10.9579*spectral2 + 1.6586*spectral3);
  coefficients.nu = pow(10.0, 1.347 + 1.246*spectral + 0.4525*spectral2);
  coefficients.p = pow(10.0, 0.071 - 0.433*spectral);
  coefficients.e = pow(10.0, -0.632 + 0.646*spectral);
  coefficients.d = pow(10.0, -0.483 + 0.892*amplitude - 0.086*matter);

  return coefficients;
}


// ---------------------------------------------------------------------------
// prepare_bihalofit: tabulate the BiHalofit background quantities on
// coefficient anchor redshifts. The fit depends on a nonlinear radius
// rather than directly on a guessed k_NL, so at each anchor a
// bisection solves sigma_G(R, z) = 1 on the same 1000-node log-k grid
// as fastnc, and the slope that gives n_eff uses 2000 nodes. In the
// source-comparison mode the anchor spacings copy the source, 0.15 in
// z below 0.5 and 0.3 above it; production places fit_nz uniform
// anchors instead. A small redshift domain still receives at least
// four anchors so the cubic fit stays defined.
static void prepare_bihalofit(physics_model_3pcf* model)
{
  const double zmax = model->growth.x[model->nz - 1];
  if (!(zmax/0.15 < INT_MAX - 8)) {
    log_fatal("physics_3pcf: BiHalofit coefficient grid is too large");
    exit(1);
  }

  const int capacity = model->source_readers
    ? (int)ceil(zmax/0.15) + 8 : model->fit_nz;
  double* redshift = calloc1d(capacity);

  int count = 0;
  if (model->source_readers) {
    for (int index=0; 0.15*index < fmin(0.5, zmax); index++) {
      redshift[count++] = 0.15*index;
    }
    if (zmax > 0.5) {
      for (int index=0; 0.5 + 0.3*index < zmax; index++) {
        redshift[count++] = 0.5 + 0.3*index;
      }
    }
    redshift[count++] = zmax;
  } else {
    count = model->fit_nz;
    for (int index=0; index<count; index++) {
      redshift[index] = zmax*index/(count - 1);
    }
  }
  if (count < 4) {
    count = 4;
    for (int index=0; index<count; index++) {
      redshift[index] = zmax*index/(count - 1);
    }
  }

  double* radius = calloc1d(count);
  double* spectral = calloc1d(count);
  double* sigma8z = calloc1d(count);
  double* matter = calloc1d(count);

  for (int index=0; index<count; index++) {
    const double growth = read_spline(&model->growth, redshift[index], 0);
    const double growth_squared = growth*growth;
    double lower = 0.01;
    double upper = 1.0e4;

    double at_lower = variance_physics(model, lower, growth_squared,
      log(1.0e-4), 1000, 1, 1) - 1.0;
    for (int retry=0; at_lower < 0.0 && retry<10; retry++) {
      lower *= 0.5;
      at_lower = variance_physics(model, lower, growth_squared,
        log(1.0e-4), 1000, 1, 1) - 1.0;
    }

    const double at_upper = variance_physics(model, upper, growth_squared,
      log(1.0e-4), 1000, 1, 1) - 1.0;
    if (!(at_lower >= 0.0 && at_upper <= 0.0)) {
      log_fatal("physics_3pcf: nonlinear radius is not bracketed by the "
        "BiHalofit domain");
      exit(1);
    }

    for (int iteration=0; iteration<100; iteration++) {
      const double middle = 0.5*(lower + upper);
      const double value = variance_physics(model, middle, growth_squared,
        log(1.0e-4), 1000, 1, 1) - 1.0;
      if (value > 0.0) {
        lower = middle;
      } else {
        upper = middle;
      }
      if (upper - lower <= 1.0e-5*middle) {
        break;
      }
    }

    radius[index] = 0.5*(lower + upper);
    spectral[index] = -3.0 + 2.0*variance_physics(model,
      radius[index], growth_squared, log(1.0e-4), 2000, 2, 0);
    sigma8z[index] = sqrt(variance_physics(model, 8.0, growth_squared,
      log(1.0e-4), 1000, 0, 0));
    matter[index] = omega_m_physics(model, redshift[index]);
  }

  if (model->source_readers) {
    model->radius = make_spline(redshift, radius, count);
    model->neff = make_spline(redshift, spectral, count);
    model->sigma8z = make_spline(redshift, sigma8z, count);
    model->omega_m_z = make_spline(redshift, matter, count);
  } else {
    model->radius = make_natural_uniform_spline(redshift, radius, count);
    model->neff = make_natural_uniform_spline(redshift, spectral, count);
    model->sigma8z = make_natural_uniform_spline(redshift, sigma8z, count);
    model->omega_m_z = make_natural_uniform_spline(redshift, matter, count);
  }

  free(redshift);
  free(radius);
  free(spectral);
  free(sigma8z);
  free(matter);
}



// ===========================================================================
// [SECTION] RADIAL KERNELS AND MODEL CONSTRUCTION
// ===========================================================================

// prepare_radial: build every radial table the Limber integrals read.
//
// Four products come out of this pass, all owned by the model:
//
//   1. kernel_lens: the lensing efficiency W_kappa of every source
//      bin on the uniform kernel redshift grid. W_kappa(chi) is the
//      Poisson prefactor (3/2) Omega_m (H0/c)^2 times the mean of the
//      source efficiency (chi_s - chi)/chi_s over the bin's
//      normalized source distribution.
//   2. kernel_shape, plus a source-grid variant: the NLA source
//      kernel, the normalized source density per unit comoving
//      distance n(z) dz/dchi, stored divided by chi (1 + z) (next
//      paragraph).
//   3. los_chi, los_dchi and los_bh: the comoving distance, the
//      trapezoid weight and the BiHalofit coefficients at every LOS
//      node.
//   4. los_lens and los_shape: both kernels resampled by cubic
//      spline onto the LOS nodes, where the projection weights read
//      them.
//
// The combined lensing + alignment kernel of the NLA baseline carries
// a common factor chi (1 + z) that physically belongs to the lensing
// term alone; field_kernel_physics multiplies it onto the combined
// kernel. Dividing the shape kernel by chi (1 + z) here lets both
// terms share that later factor without changing the alignment part.
static void prepare_radial(physics_model_3pcf* model)
{
  const int nz = model->nsource;
  const int nlos = model->nlos;
  const int nkernel = model->kernel_nz;
  const double* source_z = model->source_z;
  const double* source_chi = model->source_chi;
  const double zmin = model->los_z[0];
  // Kernel and LOS support belong to the background, independently of
  // how many samples describe the source distribution within it.
  const double zmax = model->zmax;

  if (!((size_t)model->nbin*nkernel <= INT_MAX
        && (size_t)model->nbin*nlos <= INT_MAX)) {
    log_fatal("physics_3pcf: lensing kernel tables exceed the core "
      "allocator count");
    exit(1);
  }

  model->kernel_z = calloc1d(nkernel);
  model->kernel_chi = calloc1d(nkernel);
  model->kernel_growth = calloc1d(nkernel);
  model->kernel_lens = calloc1d((int)((size_t)model->nbin*nkernel));
  model->kernel_shape = calloc1d((int)((size_t)model->nbin*nkernel));
  model->los_lens = calloc1d((int)((size_t)model->nbin*nlos));
  model->los_shape = calloc1d((int)((size_t)model->nbin*nlos));
  model->los_chi = calloc1d(nlos);
  model->los_dchi = calloc1d(nlos);
  if (NULL == model->baseline) {
    model->los_bh = calloc((size_t)nlos,
      sizeof(physics_bh_coefficients));
    if (model->los_bh == NULL) {
      log_fatal("physics_3pcf: allocation failed");
      exit(1);
    }
  }

  // Sample the background once on the uniform kernel grid: redshift,
  // comoving distance and growth at every node.
  for (int node=0; node<nkernel; node++) {
    const double redshift = zmin + (zmax - zmin)*node/(nkernel - 1);
    model->kernel_z[node] = redshift;
    model->kernel_chi[node] = read_spline(&model->distance, redshift, 0);
    model->kernel_growth[node] = read_spline(&model->growth, redshift, 0);
  }
  // Distances and BiHalofit coefficients at every LOS node;
  // physics_create_3pcf refreshes los_bh after the uniform readers
  // replace the splines.
  for (int node=0; node<nlos; node++) {
    model->los_chi[node] = read_spline(&model->distance,
      model->los_z[node], 0);
    if (NULL == model->baseline) {
      model->los_bh[node] = bh_coefficients(model, model->los_z[node]);
    }
    if (!(model->los_chi[node] > 0.0)) {
      log_fatal("physics_3pcf: line-of-sight distance must be positive");
      exit(1);
    }
  }
  // Trapezoid weights: an interior node owns half the distance
  // between its two neighbors, and each endpoint owns half of its
  // single adjacent interval, so summing f(chi_node)*dchi over the
  // nodes equals the trapezoid rule on the LOS grid.
  for (int node=0; node<nlos; node++) {
    const int lower = node > 0 ? node - 1 : node;
    const int upper = node + 1 < nlos ? node + 1 : node;
    model->los_dchi[node] = 0.5*(model->los_chi[upper]
      - model->los_chi[lower]);
  }

  double* integrand = calloc1d(nz);
  double* source_shape = calloc1d(nz);

  // Lensing prefactor (3/2) Omega_m (H0/c)^2 of the Poisson equation.
  // With H0 = 100 h km/s/Mpc and c = 299792 km/s, the ratio
  // 100/299792 is H0/c in h/Mpc, matching the Mpc/h distances, so the
  // prefactor carries units of (h/Mpc)^2.
  const double prefactor = 1.5*pow(100.0/299792.0, 2)*model->omega_m;

  // One pass per source bin: normalize the bin's source distribution,
  // build its lensing efficiency and NLA kernel on the kernel grid,
  // build the source-grid shape kernel, and resample both kernels
  // onto the LOS nodes.
  for (int bin=0; bin<model->nbin; bin++) {
    // Rows of this bin: density points at the input dn/dz, while lens
    // and nla_shape point at the bin's output rows on the kernel grid.
    const double* density = model->sources + (size_t)bin*nz;
    double* lens = model->kernel_lens + (size_t)bin*nkernel;
    double* nla_shape = model->kernel_shape + (size_t)bin*nkernel;
    // The input n(z) need not integrate to one; this norm divides it
    // out of every average below.
    const double redshift_norm = trapezoid(source_z, density, nz);
    if (!(redshift_norm > 0.0)) {
      log_fatal("physics_3pcf: empty source distribution");
      exit(1);
    }

    for (int node=0; node<nkernel; node++) {
      const double distance = model->kernel_chi[node];
      for (int source=0; source<nz; source++) {
        // Geometric lensing weight of one source at chi_s for a lens
        // at this node's distance: (chi_s - chi)/chi_s = 1 - chi/chi_s.
        // A source in front of the lens cannot be lensed by it and
        // contributes zero, which the ternary enforces.
        integrand[source] = source_chi[source] > distance
          ? density[source]*(1.0 - distance/source_chi[source]) : 0.0;
      }
      // The trapezoid integral over the sources, divided by the n(z)
      // norm, is the density-weighted mean efficiency of the bin.
      lens[node] = prefactor*trapezoid(source_z, integrand, nz)
        /redshift_norm;
      // NLA kernel sample: the source density per unit comoving
      // distance, n(z) dz/dchi, read at this kernel node's redshift.
      nla_shape[node] = linear_read(source_z, density, nz,
        model->kernel_z[node])*read_spline(&model->dz_dchi,
        model->kernel_z[node], 0);
    }

    // Normalize the NLA kernel to unit integral over chi, then remove
    // the common chi (1 + z) factor that field_kernel_physics
    // restores on the combined kernel.
    const double nla_norm = trapezoid(model->kernel_chi,
      nla_shape, nkernel);
    if (!(nla_norm > 0.0)) {
      log_fatal("physics_3pcf: empty sampled NLA source kernel");
      exit(1);
    }
    for (int node=0; node<nkernel; node++) {
      nla_shape[node] /= nla_norm*model->kernel_chi[node]
        *(1.0 + model->kernel_z[node]);
    }

    // The TATT residual reads the shape kernel on the source grid
    // itself, normalized over chi there: fastnc keeps this grid
    // separate from the kernel-grid NLA baseline, and so does this
    // port (see field_kernel_physics).
    for (int source=0; source<nz; source++) {
      integrand[source] = density[source]*model->source_jacobian[source];
    }

    const double shape_norm = trapezoid(source_chi, integrand, nz);
    if (!(shape_norm > 0.0)) {
      log_fatal("physics_3pcf: empty shape source kernel");
      exit(1);
    }

    for (int source=0; source<nz; source++) {
      // Only the actual observer sits at zero distance, where the
      // 1/chi factor would diverge; the observer hosts no sources, so
      // that endpoint stores zero. A source grid may instead begin at
      // a positive midpoint and then keeps its first density. The LOS
      // nodes themselves are strictly positive.
      source_shape[source] = source_chi[source] == 0.0 ? 0.0 : integrand[source]
        /(shape_norm*source_chi[source]*(1.0 + source_z[source]));
    }

    // Resample both kernels onto the LOS nodes by cubic spline; the
    // zero-outside read keeps a node beyond the sources empty.
    physics_spline lens_spline = make_spline(model->kernel_chi,
      lens, nkernel);
    physics_spline shape_spline = make_spline(source_chi, source_shape, nz);

    for (int node=0; node<nlos; node++) {
      model->los_lens[(size_t)bin*nlos + node]
        = read_spline(&lens_spline, model->los_chi[node], 1);
      model->los_shape[(size_t)bin*nlos + node]
        = read_spline(&shape_spline, model->los_chi[node], 1);
    }

    destroy_spline(&lens_spline);
    destroy_spline(&shape_spline);
  }

  free(integrand);
  free(source_shape);
}


// ---------------------------------------------------------------------------
// physics_create_3pcf: validate the input, copy every table and run
// the preparation passes in a fixed order. The constructor normalizes
// the linear power spectrum to sigma8 before anything reads it; the
// BiHalofit background and the radial kernels then build on the
// original cubic splines; prepare_uniform_readers replaces the
// production tables last. Because that replacement changes
// interpolation details, the constructor re-evaluates the per-LOS
// BiHalofit coefficients, power rows and kernel growth samples
// afterwards, so projected means and covariance queries read
// identical model values at identical redshifts.
physics_model_3pcf* physics_create_3pcf(const physics_input_3pcf* input)
{
  if (!(input != NULL)) {
    log_fatal("physics_3pcf: missing model input");
    exit(1);
  }
  if (!(input->nk >= 4 && input->nz >= 4 && input->nbin > 0
        && input->nlos >= 2 && input->kernel_nz >= 4)) {
    log_fatal("physics_3pcf: insufficient model grid dimensions");
    exit(1);
  }

  increasing_physics(input->k, input->nk, 1);
  increasing_physics(input->z, input->nz, 0);
  increasing_physics(input->chi, input->nz, 0);
  increasing_physics(input->los_z, input->nlos, 1);

  if (!(input->z[0] == 0.0 && input->chi[0] == 0.0)) {
    log_fatal("physics_3pcf: background tables must start at z=chi=0");
    exit(1);
  }
  if (!(input->los_z[input->nlos - 1] <= input->z[input->nz - 1])) {
    log_fatal("physics_3pcf: LOS nodes extend beyond the supplied physical "
      "tables");
    exit(1);
  }
  if (!(input->p_linear0 && input->p_nonlinear && input->growth
        && input->dz_dchi && input->sources)) {
    log_fatal("physics_3pcf: missing physical input table");
    exit(1);
  }
  if (!(fabs(input->growth[0] - 1.0) <= 1.0e-10)) {
    log_fatal("physics_3pcf: growth must be normalized to one at z=0");
    exit(1);
  }
  // The supplied route takes its background from arrays. Only Omega_m0
  // (lensing/IA) and sigma8 (the existing power normalization) are scalars
  // it needs. It must never reconstruct a w0-wa background or fit B.
  if (NULL != input->baseline) {
    if (!(isfinite(input->omega_m))) { log_fatal(
      "physics_3pcf: Omega_m0 must be finite"); exit(1); }
    if (!(0.0 < input->omega_m)) { log_fatal(
      "physics_3pcf: Omega_m0 must be positive"); exit(1); }
    if (!(isfinite(input->sigma8))) { log_fatal(
      "physics_3pcf: sigma8 must be finite"); exit(1); }
    if (!(0.0 < input->sigma8)) { log_fatal(
      "physics_3pcf: sigma8 must be positive"); exit(1); }
    if (!(NULL != input->omega_m_z)) { log_fatal(
      "physics_3pcf: supplied B requires tabulated Omega_m(z)"); exit(1); }
    for (int node=0; node<input->nz; node++) {
      if (!(isfinite(input->omega_m_z[node]))) { log_fatal(
        "physics_3pcf: Omega_m(z) must be finite"); exit(1); }
      if (!(0.0 < input->omega_m_z[node])) { log_fatal(
        "physics_3pcf: Omega_m(z) must be positive"); exit(1); }
    }
  } else {
    if (!(isfinite(input->omega_m) && input->omega_m > 0.0
          && isfinite(input->omega_de) && input->omega_de >= 0.0
          && isfinite(input->w0) && isfinite(input->wa)
          && isfinite(input->ns) && isfinite(input->sigma8)
          && input->sigma8 > 0.0)) {
      log_fatal("physics_3pcf: invalid cosmological scalars");
      exit(1);
    }
    const double density_sum = input->omega_m + input->omega_de;
    if (!(isfinite(density_sum) && fabs(density_sum - 1.0) <= 1.0e-8)) {
      log_fatal("physics_3pcf: flat geometry requires abs(omega_m + omega_de "
        "- 1) <= 1e-8");
      exit(1);
    }
  }

  if (!(input->source_readers == 0 || input->source_readers == 1)) {
    log_fatal("physics_3pcf: source_readers must be zero or one");
    exit(1);
  }
  if (!(input->source_z != NULL || input->nsource == 0)) {
    log_fatal("physics_3pcf: nsource requires an independent source_z array");
    exit(1);
  }

  const int nsource = input->source_z ? input->nsource : input->nz;
  const double* source_z = input->source_z ? input->source_z : input->z;
  if (!(nsource >= 4)) {
    log_fatal("physics_3pcf: source grid needs at least four nodes");
    exit(1);
  }
  increasing_physics(source_z, nsource, 0);
  if (!(source_z[0] >= 0.0
        && source_z[nsource - 1] <= input->z[input->nz - 1])) {
    log_fatal("physics_3pcf: source grid lies outside the background range");
    exit(1);
  }

  physics_model_3pcf* model = calloc(1, sizeof(*model));
  if (model == NULL) {
    log_fatal("physics_3pcf: allocation failed");
    exit(1);
  }

  if (!(input->nwindow == 0
        || (input->nwindow >= 2 && input->window != NULL))) {
    log_fatal("physics_3pcf: pixel window needs at least two integer-ell "
      "samples or count zero");
    exit(1);
  }
  model->nwindow = input->nwindow;
  if (model->nwindow > 0) {
    model->window = calloc1d(model->nwindow);
    for (int index=0; index<model->nwindow; index++) {
      if (!(isfinite(input->window[index])
            && input->window[index] >= 0.0)) {
        log_fatal("physics_3pcf: pixel window samples must be finite and "
          "nonnegative");
        exit(1);
      }
      model->window[index] = input->window[index];
    }
  }

  model->baseline = baseline_create_3pcf(input->baseline);
  model->feedback = feedback_create_3pcf(input->feedback);
  model->nk = input->nk;
  model->nz = input->nz;
  model->nbin = input->nbin;
  model->nlos = input->nlos;
  model->kernel_nz = input->kernel_nz;
  model->source_readers = input->source_readers;
  model->table_boost = input->table_boost == 0 ? 8 : input->table_boost;
  model->fit_nz = input->fit_nz == 0 ? 33 : input->fit_nz;
  if (!(model->table_boost >= 1)) {
    log_fatal("physics_3pcf: table_boost must be positive");
    exit(1);
  }
  if (!(model->fit_nz >= 4)) {
    log_fatal("physics_3pcf: fit_nz must be at least four");
    exit(1);
  }
  model->npower_z = input->nz;
  model->nsource = nsource;
  model->zmax = input->z[input->nz - 1];
  model->power_z = calloc1d(input->nz);
  memcpy(model->power_z, input->z, (size_t)input->nz*sizeof(double));
  model->omega_m = input->omega_m;
  if (NULL == model->baseline) {
    model->omega_de = input->omega_de;
    model->w0 = input->w0;
    model->wa = input->wa;
    model->ns = input->ns;
  }
  model->sigma8 = input->sigma8;

  const int nk = model->nk;
  const int nz = model->nz;
  double* logk = calloc1d(nk);
  double* logpower = calloc1d(nk);
  for (int index=0; index<nk; index++) {
    if (!(isfinite(input->p_linear0[index])
          && input->p_linear0[index] > 0.0)) {
      log_fatal("physics_3pcf: linear power must be positive");
      exit(1);
    }
    logk[index] = log(input->k[index]);
    logpower[index] = log(input->p_linear0[index]);
  }

  model->linear = make_spline(logk, logpower, nk);
  const double variance = variance_physics(model, 8.0, 1.0,
    log(1.0e-3), 1000, 0, 0);
  const double log_normalization = log(model->sigma8*model->sigma8/variance);
  for (int index=0; index<nk; index++) {
    model->linear.y[index] += log_normalization;
  }

  model->nonlinear = calloc((size_t)nz, sizeof(physics_spline));
  if (model->nonlinear == NULL) {
    log_fatal("physics_3pcf: allocation failed");
    exit(1);
  }

  for (int redshift=0; redshift<nz; redshift++) {
    if (!(input->growth[redshift] > 0.0
          && input->dz_dchi[redshift] > 0.0)) {
      log_fatal("physics_3pcf: growth and dz/dchi must be positive");
      exit(1);
    }
    for (int index=0; index<nk; index++) {
      const double power = input->p_nonlinear[(size_t)redshift*nk + index];
      if (!(isfinite(power) && power > 0.0)) {
        log_fatal("physics_3pcf: nonlinear power must be positive");
        exit(1);
      }
      logpower[index] = log(power);
    }
    model->nonlinear[redshift] = make_spline(logk, logpower, nk);
  }

  model->distance = make_spline(input->z, input->chi, nz);
  model->growth = make_spline(input->z, input->growth, nz);
  model->dz_dchi = make_spline(input->z, input->dz_dchi, nz);

  model->source_z = calloc1d(nsource);
  model->source_chi = calloc1d(nsource);
  model->source_jacobian = calloc1d(nsource);
  memcpy(model->source_z, source_z, (size_t)nsource*sizeof(double));

  // When the source grid is the background redshift grid itself, copy
  // chi and dz/dchi directly, so this shared source/background grid
  // preparation keeps its original arithmetic bit for bit. An
  // independent source grid instead reads the background splines on
  // its nodes here, once, before any triangle or covariance worker
  // starts.
  const int same_source_grid = nsource == nz
    && memcmp(source_z, input->z, (size_t)nz*sizeof(double)) == 0;
  for (int index=0; index<nsource; index++) {
    model->source_chi[index] = same_source_grid ? input->chi[index]
      : read_spline(&model->distance, source_z[index], 0);
    model->source_jacobian[index] = same_source_grid
      ? input->dz_dchi[index]
      : read_spline(&model->dz_dchi, source_z[index], 0);
    if (!(isfinite(model->source_jacobian[index])
          && model->source_jacobian[index] > 0.0)) {
      log_fatal("physics_3pcf: source background Jacobian is not positive");
      exit(1);
    }
  }

  increasing_physics(model->source_chi, nsource, 0);
  if (!(model->source_chi[0] >= 0.0)) {
    log_fatal("physics_3pcf: source distance is negative");
    exit(1);
  }
  if (!((size_t)input->nbin*nsource <= INT_MAX)) {
    log_fatal("physics_3pcf: source density table exceeds the core "
      "allocator count");
    exit(1);
  }

  model->sources = calloc1d((int)((size_t)input->nbin*nsource));
  for (size_t index=0; index<(size_t)input->nbin*nsource; index++) {
    if (!(isfinite(input->sources[index])
          && input->sources[index] >= 0.0)) {
      log_fatal("physics_3pcf: source densities must be nonnegative");
      exit(1);
    }
    model->sources[index] = input->sources[index];
  }

  for (int bin=0; bin<input->nbin; bin++) {
    if (!(source_z[0] != 0.0
          || model->sources[(size_t)bin*nsource] == 0.0)) {
      log_fatal("physics_3pcf: source density must vanish at z=0");
      exit(1);
    }
  }

  model->los_z = calloc1d(input->nlos);
  memcpy(model->los_z, input->los_z, (size_t)input->nlos*sizeof(double));

  if (NULL == model->baseline) {
    prepare_bihalofit(model);
  } else {
    model->omega_m_z = make_spline(input->z, input->omega_m_z, input->nz);
  }
  prepare_radial(model);
  prepare_uniform_readers(model);

  // The per-LOS fit coefficients and the kernel growth samples must
  // come from the reader that production will use, so projected means
  // and covariance B queries read identical model values at an
  // identical redshift.
  model->los_power = calloc((size_t)3*model->nlos,
    sizeof(physics_power_row));
  if (model->los_power == NULL) {
    log_fatal("physics_3pcf: allocation failed");
    exit(1);
  }

  for (int node=0; node<model->nlos; node++) {
    if (NULL == model->baseline) {
      model->los_bh[node] = bh_coefficients(model, model->los_z[node]);
    }
    for (int mode=0; mode<3; mode++) {
      model->los_power[(size_t)mode*model->nlos + node]
        = prepare_power_row(model, model->los_z[node], mode);
    }
  }

  for (int node=0; node<model->kernel_nz; node++) {
    model->kernel_growth[node] = read_spline(&model->growth,
      model->kernel_z[node], 0);
  }

  free(logk);
  free(logpower);

  return model;
}


// ---------------------------------------------------------------------------
// physics_destroy_3pcf: free every owned table; a NULL model is a
// no-op.
void physics_destroy_3pcf(physics_model_3pcf* model)
{
  if (model == NULL) {
    return;
  }

  destroy_spline(&model->linear);
  for (int index=0; index<model->npower_z; index++) {
    destroy_spline(model->nonlinear + index);
  }
  free(model->nonlinear);
  free(model->power_z);
  destroy_spline(&model->distance);
  destroy_spline(&model->growth);
  destroy_spline(&model->dz_dchi);
  destroy_spline(&model->radius);
  destroy_spline(&model->neff);
  destroy_spline(&model->sigma8z);
  destroy_spline(&model->omega_m_z);
  free(model->sources);
  free(model->source_z);
  free(model->source_chi);
  free(model->source_jacobian);
  free(model->kernel_z);
  free(model->kernel_chi);
  free(model->kernel_growth);
  free(model->kernel_lens);
  free(model->kernel_shape);
  free(model->los_z);
  free(model->los_chi);
  free(model->los_dchi);
  free(model->los_lens);
  free(model->los_shape);
  free(model->los_bh);
  free(model->los_power);
  baseline_destroy_3pcf(model->baseline);
  feedback_destroy_3pcf(model->feedback);
  free(model->window);
  free(model);
}


// ---------------------------------------------------------------------------
// physics_default_ia_3pcf: zero every IA parameter, then set the
// defaults the fastnc comparison uses: the pivot redshift
// zpivot = 0.62; the normalization c1rho = 0.0134, the C_1 rho_crit
// value of the tidal-alignment literature; power_mode = 1, the z = 0
// nonlinear power scaled by the squared growth factor; and
// intrinsic_legs = -1, keeping every field placement. All amplitudes
// stay zero, so the default model has no intrinsic alignment.
void physics_default_ia_3pcf(physics_ia_3pcf* parameters)
{
  if (!(parameters != NULL)) {
    log_fatal("physics_3pcf: missing alignment parameters");
    exit(1);
  }

  memset(parameters, 0, sizeof(*parameters));
  parameters->zpivot = 0.62;
  parameters->c1rho = 0.0134;
  parameters->power_mode = 1;
  parameters->intrinsic_legs = -1;
}


// ---------------------------------------------------------------------------
// physics_window_value_3pcf: read the angular pixel window at a real
// multipole. The samples sit at integer ell, so array index j holds
// ell = j and the read needs no search. Above the final sample the
// window drops to zero on purpose, even when that sample is nonzero.
// Do not interpolate that boundary into a gradual taper.
double physics_window_value_3pcf(
  const int count,
  const double* window,
  const double ell
)
{
  if (count == 0) {
    return 1.0;
  }
  if (ell < 0.0) {
    return 1.0;
  }
  if (ell > count - 1) {
    return 0.0;
  }
  if (ell == count - 1) {
    return window[count - 1];
  }

  const int index = (int)ell;
  const double fraction = ell - index;
  return (1.0 - fraction)*window[index] + fraction*window[index + 1];
}


// ---------------------------------------------------------------------------
// physics_window_samples_3pcf: expose the owned window samples and
// their count; the model keeps ownership.
const double* physics_window_samples_3pcf(
  const physics_model_3pcf* model,
  int* count
)
{
  if (!(model && count)) {
    log_fatal("physics_3pcf: missing window model or count output");
    exit(1);
  }
  *count = model->nwindow;
  return model->window;
}


// ---------------------------------------------------------------------------
// prepare_power_row: choose the P(k) tables for one power convention.
// Mode 0 reads the linear power at z = 0 and scales it by the squared
// growth factor D^2(z); mode 1 does the same with the nonlinear power
// at z = 0; mode 2 interpolates log P between the two nonlinear
// redshift rows that bracket z. Production finds the bracket by index
// arithmetic on the uniform redshift axis; the source-comparison mode
// searches the original irregular axis.
static physics_power_row prepare_power_row(
  const physics_model_3pcf* model,
  const double z,
  const int mode
)
{
  physics_power_row row = {0};
  if (mode < 2) {
    row.lower = mode == 0 ? &model->linear : model->nonlinear;
    row.growth = read_spline(&model->growth, z, 0);
    return row;
  }

  int index;
  if (model->source_readers) {
    index = interval_physics(model->power_z, model->npower_z, z);
  } else {
    const double position = z*(model->npower_z - 1)/model->zmax;
    index = (int)floor(fmax(0.0, fmin(model->npower_z - 2.0, position)));
  }

  row.lower = model->nonlinear + index;
  row.upper = row.lower + 1;
  row.fraction = (z - model->power_z[index])
    /(model->power_z[index + 1] - model->power_z[index]);
  row.growth = 1.0;
  return row;
}


// ---------------------------------------------------------------------------
// read_power_row_log: evaluate one prepared power row at log k. A
// single-table row multiplies its reader by the squared growth; a
// bracketing row interpolates log P linearly between its two redshift
// rows.
static double read_power_row_log(
  const physics_power_row* row,
  const double logk
)
{
  if (row->upper == NULL) {
    return read_power(row->lower, logk)*row->growth*row->growth;
  }

  const physics_spline* lower = row->lower;
  const physics_spline* upper = row->upper;
  if (lower->inverse_spacing > 0.0 && logk >= lower->x[0]
      && logk <= lower->x[lower->n - 1]) {
    // Both redshift rows have the same uniform log-k grid. Locate its
    // cell once and reuse that integer and fraction for the two rows.
    const double position = (logk - lower->x[0])*lower->inverse_spacing;
    const int index = (int)floor(fmax(0.0,
      fmin(lower->n - 2.0, position)));
    const double fraction = position - index;
    const double first = lower->y[index]
      + fraction*(lower->y[index + 1] - lower->y[index]);
    const double second = upper->y[index]
      + fraction*(upper->y[index + 1] - upper->y[index]);
    return exp(first + row->fraction*(second - first));
  }

  // The source-comparison mode and the out-of-range endpoint power
  // laws keep this explicit scalar path; a production in-range query
  // never reaches it and never searches a grid.
  const double first = log(read_power(lower, logk));
  const double second = log(read_power(upper, logk));
  return exp(first + row->fraction*(second - first));
}


// ---------------------------------------------------------------------------
static double read_power_row(const physics_power_row* row, const double k)
{
  return read_power_row_log(row, log(k));
}


// ---------------------------------------------------------------------------
// physics_power_3pcf: public power reader; mode selects the
// convention of prepare_power_row (0 linear times D^2, 1 nonlinear at
// z = 0 times D^2, 2 nonlinear interpolated in redshift).
double physics_power_3pcf(
  const physics_model_3pcf* model,
  const double k,
  const double z,
  const int mode
)
{
  if (!(model && isfinite(k) && k > 0.0 && isfinite(z)
        && z >= 0.0 && z <= model->zmax
        && mode >= 0 && mode <= 2)) {
    log_fatal("physics_3pcf: invalid power query");
    exit(1);
  }

  const physics_power_row row = prepare_power_row(model, z, mode);
  return read_power_row(&row, k);
}


// ---------------------------------------------------------------------------
// physics_feedback_power_3pcf: the paired baryon response ratio for
// the power spectrum; feedback_3pcf.c owns the physics.
double physics_feedback_power_3pcf(
  const physics_model_3pcf* model,
  const double k,
  const double z,
  const double amplitude
)
{
  if (!(model && isfinite(k) && k > 0.0
        && isfinite(z) && z >= 0.0)) {
    log_fatal("physics_3pcf: invalid power-response coordinates");
    exit(1);
  }

  return feedback_power_3pcf(model->feedback, k, z, amplitude);
}


// ---------------------------------------------------------------------------
// physics_feedback_bispectrum_3pcf: the paired baryon response ratio
// for the bispectrum, after validating the triangle.
double physics_feedback_bispectrum_3pcf(
  const physics_model_3pcf* model,
  const double* sides,
  const double z,
  const double amplitude
)
{
  if (!(model && sides && isfinite(z) && z >= 0.0)) {
    log_fatal("physics_3pcf: invalid bispectrum-response coordinates");
    exit(1);
  }

  triangle_physics(sides[0], sides[1], sides[2]);
  return feedback_bispectrum_3pcf(model->feedback, sides, z, amplitude);
}



// ===========================================================================
// [SECTION] TRIANGLE GEOMETRY AND TREE-LEVEL KERNELS
// ===========================================================================

// pair_cosine_physics: the cosine between two wavevectors whose sum
// has the third side's length. The law of cosines subtracts nearly
// equal squares in a squeezed triangle, so the code factors the
// difference of the two large squared lengths before dividing by the
// small side: for sides (epsilon, 1, 1) this keeps mu = -epsilon/2
// even when epsilon squared falls below the precision of a number
// near one.
static double pair_cosine_physics(
  const double first,
  const double second,
  const double opposite
)
{
  const double small = fmin(first, second);
  const double large = fmax(first, second);
  return (((opposite - large)*(opposite + large))/small - small)
    /(2.0*large);
}


// ---------------------------------------------------------------------------
// f2_physics: the second-order density kernel of standard
// perturbation theory,
//
//   F2(k1, k2) = 5/7 + (mu/2) (k1/k2 + k2/k1) + (2/7) mu^2,
//
// with mu the cosine between the two wavevectors (see the large-scale
// structure review of Bernardeau et al., arXiv:astro-ph/0112551). It
// weights the second-order density that gravity generates on the side
// closing the two input modes, and so drives the tree-level
// bispectrum.
static double f2_physics(
  const double first,
  const double second,
  const double opposite
)
{
  const double small = fmin(first, second);
  const double large = fmax(first, second);
  const double cosine = pair_cosine_physics(small, large, opposite);

  if (cosine < -0.5) {
    // For nearly antiparallel equal sides F2 tends to zero, and its
    // usual three terms cancel at order unity. The code forms
    // one_plus = 1 + mu from another factored side difference, then
    // keeps the unequal-side correction with its physical sign; F2
    // can be negative for unequal antiparallel sides.
    const double difference = large - small;
    const double one_plus = 0.5*((opposite - difference)/small)
      *((opposite + difference)/large);
    return one_plus*(3.0 + 2.0*one_plus)/7.0
      + 0.5*cosine*(difference/small)*(difference/large);
  }

  return 5.0/7.0 + 0.5*(cosine*(small/large)
    + (cosine/small)*large) + 2.0/7.0*cosine*cosine;
}


// ---------------------------------------------------------------------------
// closing_cosine_physics: the cosine between a side and the closing
// vector, with the opposite sign convention to pair_cosine_physics.
// Either argument can be the short side, so the code chooses the
// factored subtraction accordingly to keep the squeezed-triangle
// precision.
static double closing_cosine_physics(
  const double first,
  const double closing,
  const double opposite
)
{
  const double small = fmin(first, closing);
  const double large = fmax(first, closing);
  return (((large - opposite)*(large + opposite))/small + small)
    /(2.0*large);
}


// ---------------------------------------------------------------------------
// triangle_geometry: prepare every dimensionless factor of one
// triangle shape (see the struct comment for the frame convention).
// with_ia = 1 also fills the IA vertex shapes; the matter-only
// baseline passes 0 and skips that unused preparation.
static physics_triangle_geometry triangle_geometry(
  const double* sides,
  const int with_ia // prepare IA shapes only when an IA reader will use them
)
{
  physics_triangle_geometry geometry;
  const double minimum = fmin(sides[0], fmin(sides[1], sides[2]));
  const double maximum = fmax(sides[0], fmax(sides[1], sides[2]));

  // Rescale by a power of two to prevent overflow in side products.
  // For normal, non-subnormal doubles the rescaling only shifts the
  // exponent field, so it is exact and cannot round away the small
  // difference between two nearly equal long sides.
  int exponent;
  frexp(maximum, &exponent);
  for (int leg=0; leg<3; leg++) {
    geometry.sides[leg] = scalbn(sides[leg], -exponent);
    if (!(geometry.sides[leg] > 0.0)) {
      log_fatal("physics_3pcf: triangle side ratio underflows double "
        "precision");
      exit(1);
    }
  }

  const double small = fmin(geometry.sides[0],
    fmin(geometry.sides[1], geometry.sides[2]));
  const double large = fmax(geometry.sides[0],
    fmax(geometry.sides[1], geometry.sides[2]));
  // The median uses comparisons only, so (epsilon,1,1) retains 1.
  const double middle = fmax(fmin(geometry.sides[0], geometry.sides[1]),
    fmin(fmax(geometry.sides[0], geometry.sides[1]), geometry.sides[2]));

  const double gap = large - middle;
  const int physical = gap <= small;

  double twice_area = 0.0;
  if (physical) {
    // Kahan's ordered form of Heron's formula gives 16*area squared.
    // Taking each square root separately avoids underflow from first
    // multiplying the two factors proportional to the shortest side.
    twice_area = 0.5*sqrt(large + (middle + small))*sqrt(small - gap)
      *sqrt(small + gap)*sqrt(large + (middle - small));
  }

  geometry.minimum_ratio = minimum/maximum;
  geometry.flattened_ratio = (small - gap)/large;

  for (int cyclic=0; cyclic<3; cyclic++) {
    const double first = geometry.sides[cyclic];
    const double second = geometry.sides[(cyclic + 1)%3];
    const double third = geometry.sides[(cyclic + 2)%3];
    geometry.pair_cosine[cyclic]
      = pair_cosine_physics(first, second, third);
    if (physical) {
      geometry.first_x[cyclic]
        = -closing_cosine_physics(first, third, second);
      geometry.second_x[cyclic]
        = -closing_cosine_physics(second, third, first);
      geometry.first_y[cyclic] = -(twice_area/first)/third;
      geometry.second_y[cyclic] = (twice_area/second)/third;
      geometry.dot[cyclic] = geometry.pair_cosine[cyclic];
      const double sine = (twice_area/first)/second;
      geometry.sine_squared[cyclic] = sine*sine;
    } else {
      // Off-triangle sides appear while filling fastnc's rectangular
      // PUV interpolation grid: the triangle-coordinate table over
      // the perimeter p = ell1 + ell2 + ell3 and the side fractions
      // u = ell1/p, v = ell2/p, whose rectangular corners include
      // side combinations that close no triangle. The source clips
      // the cosine and continues the vertex construction; the same
      // branch absorbs closure error within the direct validator's
      // roundoff tolerance. A valid triangle takes the area branch
      // above instead.
      const double cosine = fmax(-1.0, fmin(1.0,
        closing_cosine_physics(first, third, second)));
      geometry.first_x[cyclic] = -cosine;
      geometry.first_y[cyclic] = -sqrt(fmax(0.0, 1.0 - cosine*cosine));
      geometry.second_x[cyclic]
        = (-third - first*geometry.first_x[cyclic])/second;
      geometry.second_y[cyclic] = -first*geometry.first_y[cyclic]/second;
      geometry.dot[cyclic]
        = geometry.first_x[cyclic]*geometry.second_x[cyclic]
        + geometry.first_y[cyclic]*geometry.second_y[cyclic];
      geometry.sine_squared[cyclic]
        = 1.0 - geometry.dot[cyclic]*geometry.dot[cyclic];
    }

    geometry.gravity[cyclic] = f2_physics(first, second, third);

    geometry.ia_quadratic_shape[cyclic] = 0.0;
    geometry.ia_density_zero_shape[cyclic] = 0.0;
    geometry.ia_density_two_shape[cyclic] = 0.0;

    if (with_ia) {
      // Keep each of fastnc's parenthesized vertex expressions with
      // its original operation order; the IA coefficients multiply
      // them later. A baseline-only geometry skips this preparation
      // because nothing reads it.
      const double first_x = geometry.first_x[cyclic];
      const double first_y = geometry.first_y[cyclic];
      const double second_x = geometry.second_x[cyclic];
      const double second_y = geometry.second_y[cyclic];
      const double dot = geometry.dot[cyclic];
      geometry.ia_quadratic_shape[cyclic]
        = 0.25*(first_x*first_x - first_y*first_y
        + second_x*second_x - second_y*second_y)
        + 0.5*dot*first_x*second_x;
      geometry.ia_density_zero_shape[cyclic]
        = 2.0*first_x*first_x - first_y*first_y
        + 2.0*second_x*second_x - second_y*second_y;
      geometry.ia_density_two_shape[cyclic]
        = first_y*first_y + second_y*second_y;
    }
  }

  return geometry;
}


// ---------------------------------------------------------------------------
// tree_sum_physics: the tree-level bispectrum 2 [F2 P P + cyclic],
// arranged to stay finite in squeezed triangles. The two F2 terms
// containing the short side s can separately grow as 1/s even though
// their weighted sum stays finite, so the code combines their
// opposite large pieces before multiplying. With b, c the other two
// sides,
//
//   F2(s,b,c) = 13/28 + 2 mu_sb^2/7 + mu_sb s/(2 b)
//             + (c^2 - b^2)/(4 s^2),
//
// and the last piece multiplies the difference P(b) - P(c) directly,
// rather than two large products that a later subtraction would
// cancel. The same arrangement serves the enhanced powers in
// BiHalofit's three-halo term.
static double tree_sum_physics(
  const physics_triangle_geometry* geometry,
  const double* power
)
{
  int shortest = 0;
  for (int leg=1; leg<3; leg++) {
    if (geometry->sides[leg] < geometry->sides[shortest]) {
      shortest = leg;
    }
  }

  const int next = (shortest + 1)%3;
  const int last = (shortest + 2)%3;
  const double small = geometry->sides[shortest];
  const double first = geometry->sides[next];
  const double second = geometry->sides[last];
  const double cosine1 = geometry->pair_cosine[shortest];
  const double cosine2 = geometry->pair_cosine[last];
  const double regular1 = 13.0/28.0 + 2.0*cosine1*cosine1/7.0
    + 0.5*cosine1*(small/first);
  const double regular2 = 13.0/28.0 + 2.0*cosine2*cosine2/7.0
    + 0.5*cosine2*(small/second);

  double difference = 0.0;
  if (first != second) {
    difference = 0.25*((second - first)/small)
      *(((second + first)/small)*(power[next] - power[last]));
  }

  return 2.0*(power[shortest]*(regular1*power[next]
    + regular2*power[last] + difference)
    + geometry->gravity[next]*power[next]*power[last]);
}


// ---------------------------------------------------------------------------
// triangle_physics: require three positive finite lengths that close
// into a triangle. The test allows a relative slack of
// 32*DBL_EPSILON on the triangle inequality because callers often
// build the third side in floating point - for example from the law
// of cosines in the multipole integrator - and that construction can
// overshoot a degenerate triangle by a few units in the last place.
// A genuinely impossible triangle misses by far more than the slack.
static void triangle_physics(
  const double first,
  const double second,
  const double third
)
{
  const double largest = fmax(first, fmax(second, third));
  const double sum = first + second + third;

  if (!(isfinite(sum) && first > 0.0 && second > 0.0
        && third > 0.0 && 2.0*largest <= sum*(1.0 + 32.0*DBL_EPSILON))) {
    log_fatal("physics_3pcf: wavevector lengths do not form a finite triangle");
    exit(1);
  }
}



// ===========================================================================
// [SECTION] MATTER BISPECTRUM (BIHALOFIT AND BARYONS)
// ===========================================================================

// baryon_physics: the baryonic correction ratio of the matter
// bispectrum. Takahashi et al. (arXiv:1911.07886) fitted the ratio of
// the bispectrum with baryons to the DMO one in the IllustrisTNG
// simulation TNG300-1; every constant below copies that published
// fit, a function of the scale factor a = 1/(1 + z) and of log10 k on
// each leg. Per leg the ratio has three pieces: a mild enhancement
// bump at intermediate k and low redshift (amplitude0), a Gaussian
// depression near k = 10 h/Mpc (amplitude1), and a strong power-law
// enhancement above the turnover scale. The simulation shows no
// baryon effect in the fitted range above z = 5, so the published
// ratio is one there and the function returns early.
//
// fb interpolates between DMO and the full fitted ratio: the return
// value is 1 + fb (transfer - 1), so fb = 0 keeps DMO and fb = 1
// applies the complete fit. suppress_only = 1 caps the fitted ratio
// at one before scaling, keeping only the suppression.
static double baryon_physics(
  const double* sides,
  const double z,
  const double fb,
  const int suppress_only
)
{
  if (fb == 0.0 || z >= 5.0) {
    return 1.0;
  }

  const double scale = 1.0/(1.0 + z);
  const double amplitude0 = scale > 0.5
    ? 0.068*pow(scale - 0.5, 0.47) : 0.0;
  const double mean0 = 0.018*scale + 0.837*scale*scale;
  const double width0 = 0.881*mean0;
  const double amplitude1 = scale > 0.2
    ? 1.052*pow(scale - 0.2, 1.41) : 0.0;
  const double mean1 = fabs(0.172 + 3.048*scale - 0.675*scale*scale);
  const double width1 = (0.494 - 0.039*scale)*mean1;
  const double turnover = 29.90 - 38.73*scale + 24.30*scale*scale;
  const double exponent = 0.563/(pow(scale/0.06, 0.02) + 1.0)/2.25;

  double transfer = 1.0;
  for (int leg=0; leg<3; leg++) {
    const double logk = log10(sides[leg]);
    const double scaled1 = (logk - mean1)/width1;
    transfer *= amplitude0*exp(-pow(fabs((logk - mean0)/width0), 2.346))
      - amplitude1*exp(-scaled1*scaled1)
      + pow(1.0 + pow(sides[leg]/turnover, 2.25), exponent);
  }

  if (suppress_only) {
    transfer = fmin(transfer, 1.0);
  }

  return 1.0 + fb*(transfer - 1.0);
}


// ---------------------------------------------------------------------------
// matter_prepared: the BiHalofit matter bispectrum of Takahashi et
// al. (arXiv:1911.07886), evaluated with coefficients, a power row
// and a triangle geometry that the caller may have prepared in
// advance (NULL arguments fall back to local preparation). component
// selects the returned piece: 0 the full fit, 1 the tree level, 2 the
// one-halo term, 3 the three-halo term.
//
// PHYSICAL DERIVATION & LOGIC FLOW
//   1. Read the linear power on each leg and scale every side to
//      q = k*radius = k/k_NL, the wave number in units of the
//      nonlinear scale.
//   2. One-halo term, one halo carrying all three legs:
//        B_1h = prod_i 1/[(a q_i^alpha + b q_i^beta)(1 + 1/(c q_i))].
//      The local amplitude is a_n with its squeezed-shape correction,
//      -0.310 r1^gamma_n in log10; alpha is alpha_n with its
//      flattened-shape r2^2 term; beta is beta_n with its 0.007 r2
//      term. r1 and r2 are the side ratios stored in the geometry.
//   3. The fit caps alpha_n at 1 - 2 n_s/3, with n_s the primordial
//      spectral index: the paper imposes this bound so the one-halo
//      term stays below the tree level in the low-k limit.
//   4. Three-halo term, three distinct correlated halos: effective
//      holds the enhanced power spectrum
//        P_E = (1 + f q^2)/(1 + g q + h q^2) P_lin
//            + 1/[(m q^mu + n q^nu)(1 + (p q)^-3)],
//      which replaces P_lin in the tree-level sum; the cross terms
//      2 d q_opposite P_E P_E add the paper's halo-model correction
//      to the F2 weight; and damping multiplies one factor
//      1/(1 + e q) per leg.
//   5. The full bispectrum is the sum of the one-halo and three-halo
//      terms.
static double matter_prepared(
  const physics_model_3pcf* model,
  const physics_bh_coefficients* coefficients,
  const double* sides,
  const double z,
  const int component,
  const physics_power_row* prepared,
  const physics_triangle_geometry* prepared_geometry
)
{
  // A supplied full B has no defined fitted one-/three-halo split. The
  // tree diagnostic still comes directly from P_linear and F2. Keeping
  // these branches before coefficient reads also avoids constructing any
  // unused nonlinear-radius or variance tables for the supplied route.
  if (NULL != model->baseline) {
    if (!(component <= 1)) { log_fatal(
      "physics_3pcf: supplied full B has no fitted one-/three-halo "
      "components"); exit(1); }
    if (0 == component) {
      return baseline_bispectrum_3pcf(model->baseline, sides, z);
    }
    const physics_triangle_geometry geometry = prepared_geometry
      ? *prepared_geometry : triangle_geometry(sides, 0);
    const physics_power_row row = prepared ? *prepared
      : prepare_power_row(model, z, 0);
    double linear[3];
    for (int leg=0; leg<3; leg++) {
      linear[leg] = read_power_row(&row, sides[leg]);
    }
    return tree_sum_physics(&geometry, linear);
  }

  physics_triangle_geometry local_geometry;
  const physics_triangle_geometry* geometry = prepared_geometry;
  if (geometry == NULL) {
    local_geometry = triangle_geometry(sides, 0);
    geometry = &local_geometry;
  }

  const physics_power_row row = prepared ? *prepared
    : prepare_power_row(model, z, 0);

  double linear[3];
  double effective[3];
  double scaled[3];
  for (int leg=0; leg<3; leg++) {
    linear[leg] = read_power_row(&row, sides[leg]);
    scaled[leg] = sides[leg]*coefficients->radius;
  }

  if (component == 1) {
    return tree_sum_physics(geometry, linear);
  }

  const double ratio1 = geometry->minimum_ratio;
  const double ratio2 = geometry->flattened_ratio;
  const double amplitude = pow(10.0, coefficients->loga
    - 0.310*pow(ratio1, coefficients->gamma));
  const double alpha = fmin(pow(10.0, coefficients->logalpha
    + coefficients->alpha_shape*ratio2*ratio2), 1.0 - 2.0*model->ns/3.0);
  const double beta = pow(10.0, coefficients->logbeta + 0.007*ratio2);

  double one_halo = 1.0;
  double damping = 1.0;
  for (int leg=0; leg<3; leg++) {
    const double q = scaled[leg];
    one_halo /= (amplitude*pow(q, alpha) + coefficients->b*pow(q, beta))
      *(1.0 + 1.0/(coefficients->c*q));
    effective[leg] = (1.0 + coefficients->f*q*q)
      /(1.0 + coefficients->g*q + coefficients->h*q*q)*linear[leg]
      + 1.0/((coefficients->m*pow(q, coefficients->mu)
      + coefficients->n*pow(q, coefficients->nu))
      *(1.0 + pow(coefficients->p*q, -3.0)));
    damping /= 1.0 + coefficients->e*q;
  }

  double three_halo = tree_sum_physics(geometry, effective);
  for (int leg=0; leg<3; leg++) {
    const int next = (leg + 1)%3;
    const int opposite = (leg + 2)%3;
    three_halo += 2.0*coefficients->d*scaled[opposite]
      *effective[leg]*effective[next];
  }

  three_halo *= damping;

  if (component == 2) {
    return one_halo;
  }
  if (component == 3) {
    return three_halo;
  }
  return one_halo + three_halo;
}


// ---------------------------------------------------------------------------
// physics_matter_3pcf: public matter-bispectrum query. It validates
// the triangle, evaluates the BiHalofit coefficients at z and applies
// the baryon ratio at the requested fb, with the fit's enhancement
// pieces allowed (suppress_only off).
double physics_matter_3pcf(
  const physics_model_3pcf* model,
  const double k1,
  const double k2,
  const double k3,
  const double z,
  const int component,
  const double fb
)
{
  if (!(model && component >= 0 && component <= 3
        && isfinite(z) && z >= 0.0 && z <= model->zmax
        && isfinite(fb))) {
    log_fatal("physics_3pcf: invalid matter query");
    exit(1);
  }

  triangle_physics(k1, k2, k3);
  const double sides[3] = {k1, k2, k3};
  if (NULL != model->baseline) {
    return matter_prepared(model, NULL, sides, z, component, NULL, NULL)
      *baryon_physics(sides, z, fb, 0);
  }
  const physics_bh_coefficients coefficients = bh_coefficients(model, z);

  return matter_prepared(model, &coefficients, sides, z, component, NULL, NULL)
    *baryon_physics(sides, z, fb, 0);
}



// ===========================================================================
// [SECTION] TATT TENSOR ALGEBRA
// ===========================================================================

// check_ia: require finite IA parameters and known convention
// selectors before any evaluation path reads them.
static void check_ia(const physics_ia_3pcf* parameters)
{
  if (!(parameters != NULL)) {
    log_fatal("physics_3pcf: missing alignment parameters");
    exit(1);
  }
  if (!(isfinite(parameters->a1) && isfinite(parameters->eta1)
        && isfinite(parameters->a2) && isfinite(parameters->eta2)
        && isfinite(parameters->bta) && isfinite(parameters->ct)
        && isfinite(parameters->zpivot) && parameters->zpivot > -1.0
        && isfinite(parameters->c1rho) && parameters->c1rho > 0.0
        && isfinite(parameters->fb)
        && isfinite(parameters->feedback_amplitude))) {
    log_fatal("physics_3pcf: nonfinite alignment or baryon parameter");
    exit(1);
  }
  if (!(parameters->fb == 0.0
        || parameters->feedback_amplitude == 0.0)) {
    log_fatal("physics_3pcf: paired feedback cannot be combined with legacy "
      "bispectrum-only fb");
    exit(1);
  }
  if (!(parameters->power_mode >= 0 && parameters->power_mode <= 2
        && parameters->ct_mode >= 0 && parameters->ct_mode <= 1
        && parameters->suppress_only >= 0 && parameters->suppress_only <= 1)) {
    log_fatal("physics_3pcf: unknown alignment or baryon convention");
    exit(1);
  }
  if (!(parameters->operator_component >= 0
        && parameters->operator_component <= 4
        && parameters->intrinsic_legs >= -1
        && parameters->intrinsic_legs <= 7)) {
    log_fatal("physics_3pcf: operator_component must be 0..4 and "
      "intrinsic_legs must be -1..7");
    exit(1);
  }
}


// ---------------------------------------------------------------------------
// ia_coefficients: the TATT amplitude coefficients at one redshift,
// in the conventions of Gomes et al. (arXiv:2601.09133) that fastnc
// implements. With the pivot ratio r = (1 + z)/(1 + zpivot) and the
// growth factor D(z),
//
//   C_1(z) = -A_1 r^eta1 (C_1 rho_crit) Omega_m / D(z),
//   C_2(z) =  A_2 r^eta2 5 (C_1 rho_crit) Omega_m / D(z)^2,
//   C_t    =  C_1(z) ct  (ct_mode 0, the fastnc convention)
//          or ct alone   (ct_mode 1).
//
// A_1 is the tidal-alignment amplitude (parameters->a1) with its
// redshift power eta1; A_2 is the tidal-torque amplitude (a2) with
// eta2; bta is the density-weighting amplitude b_TA; ct scales the
// velocity-shear term; c1rho is the C_1 rho_crit normalization,
// 0.0134 by default. The combined vertex weights follow: linear is
// 2 C_1 for a linear shape on one leg, velocity is -2 C_t/5,
// quadratic is C_2, and density is b_TA C_1 - 2 C_2/3, whose second
// piece is the density-shaped trace part of the tidal-torque
// operator.
static physics_ia_coefficients ia_coefficients(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const double z,
  const double growth
)
{
  physics_ia_coefficients coefficients;
  const double ratio = (1.0 + z)/(1.0 + parameters->zpivot);
  const double c1 = -parameters->a1*pow(ratio, parameters->eta1)
    *parameters->c1rho*model->omega_m/growth;
  const double c2 = parameters->a2*pow(ratio, parameters->eta2)
    *5.0*parameters->c1rho*model->omega_m/(growth*growth);
  const double ct = parameters->ct_mode == 0
    ? c1*parameters->ct : parameters->ct;
  coefficients.c1 = c1;
  coefficients.linear = 2.0*c1;
  coefficients.velocity = -2.0*ct/5.0;
  coefficients.quadratic = c2;
  coefficients.density = parameters->bta*c1 - 2.0*c2/3.0;

  // Diagnostics select one physical operator. The density-shaped
  // vertex contains the trace part of the tidal torque, so selecting
  // C_2 alone would assign that part to the wrong operator; the
  // tidal-torque diagnostic (3) therefore keeps -2 C_2/3 in the
  // density weight. Every residual keeps the same linear shapes on
  // its remaining intrinsic legs.
  if (parameters->operator_component != 0) {
    coefficients.velocity = parameters->operator_component == 4
      ? -2.0*ct/5.0 : 0.0;
    coefficients.quadratic = parameters->operator_component == 3 ? c2 : 0.0;
    coefficients.density = parameters->operator_component == 2
      ? parameters->bta*c1 : 0.0;
    if (parameters->operator_component == 3) {
      coefficients.density = -2.0*c2/3.0;
    }
  }

  return coefficients;
}


// ---------------------------------------------------------------------------
// ia_vertices: the second-order shape vertices of the three cyclic
// orderings. A shape tensor here has three helicity projections,
// labelled 0, 1 and 2: the tensor components counted around the
// output wavevector as the axis. For each cyclic ordering the output
// wavevector sits on the positive x axis and triangle closure fixes
// the two input vectors; vertices[c][m] is the helicity-m
// second-order vertex of ordering c after removing the
// linear-alignment part. The gravity part joins a common tree sum
// later (ia_prepared adds it once at the end), which avoids
// cancellation between large squeezed cyclic terms; the tree array
// therefore stays zero here. Preparing three explicit vertices keeps
// each leg and orientation visible.
static void ia_vertices(
  const physics_triangle_geometry* geometry,
  const double* power,
  const physics_ia_coefficients* coefficients,
  double vertices[3][3],
  double* tree,
  const int e_only // omit helicity one when only E correlators are requested
)
{
  const double root_two_thirds = sqrt(2.0/3.0);

  for (int cyclic=0; cyclic<3; cyclic++) {
    const int next = (cyclic + 1)%3;
    const double first_x = geometry->first_x[cyclic];
    const double first_y = geometry->first_y[cyclic];
    const double second_x = geometry->second_x[cyclic];
    const double second_y = geometry->second_y[cyclic];
    const double dot = geometry->dot[cyclic];
    const double pair_power = power[cyclic]*power[next];
    const double quadratic = coefficients->quadratic
      *geometry->ia_quadratic_shape[cyclic];
    const double density = 0.25*coefficients->density
      *geometry->ia_density_zero_shape[cyclic];
    const double velocity = coefficients->velocity*5.0/7.0
      *geometry->sine_squared[cyclic];
    vertices[cyclic][0] = 2.0*root_two_thirds*pair_power
      *(quadratic + density + velocity);
    if (!e_only) {
      vertices[cyclic][1] = -pair_power
        *(coefficients->quadratic + coefficients->density)
        *(first_x*first_y + second_x*second_y);
    }
    vertices[cyclic][2] = 2.0*pair_power
      *(0.5*coefficients->quadratic*dot*first_y*second_y
      + 0.25*coefficients->density
      *geometry->ia_density_two_shape[cyclic]);
    tree[cyclic] = 0.0;
  }
}


// ---------------------------------------------------------------------------
// tensor002, tensor022, tensor222: the field contractions. The digit
// labels describe the fields, not their perturbation orders: 0 is the
// scalar density and 2 the rank-two intrinsic-shape tensor. Thus 002
// contains two densities and one shape, 022 one density and two
// shapes, and 222 three shapes. At tree order each contribution
// places one second-order vertex on a leg and linear fields on the
// remaining legs; the cyclic index tracks that leg placement. A
// helicity argument instead selects a tensor component relative to
// its wavevector and does not count perturbative powers. Linear
// shapes occupy helicity zero only, which explains the conditional
// factors in these contractions.
static double tensor002(
  const double vertices[3][3],
  const double* tree,
  const double linear,
  const int cyclic,
  const int helicity
)
{
  return vertices[cyclic][helicity] + (helicity == 0
    ? sqrt(2.0/3.0)*linear*(tree[(cyclic + 1)%3] + tree[(cyclic + 2)%3])
    : 0.0);
}


// ---------------------------------------------------------------------------
static double tensor022(
  const double vertices[3][3],
  const double* tree,
  const double linear,
  const int cyclic,
  const int helicity2,
  const int helicity3
)
{
  const double delta2 = helicity2 == 0 ? sqrt(2.0/3.0) : 0.0;
  const double delta3 = helicity3 == 0 ? sqrt(2.0/3.0) : 0.0;
  return delta2*linear*vertices[cyclic][helicity3]
    + delta2*delta3*linear*linear*tree[(cyclic + 1)%3]
    + delta3*linear*vertices[(cyclic + 2)%3][helicity2];
}


// ---------------------------------------------------------------------------
static double tensor222(
  const double vertices[3][3],
  const double linear,
  const int cyclic,
  const int helicity1,
  const int helicity2,
  const int helicity3
)
{
  const double delta1 = helicity1 == 0 ? sqrt(2.0/3.0) : 0.0;
  const double delta2 = helicity2 == 0 ? sqrt(2.0/3.0) : 0.0;
  const double delta3 = helicity3 == 0 ? sqrt(2.0/3.0) : 0.0;
  return linear*linear*(delta1*delta2*vertices[cyclic][helicity3]
    + delta2*delta3*vertices[(cyclic + 1)%3][helicity1]
    + delta3*delta1*vertices[(cyclic + 2)%3][helicity2]);
}


// ---------------------------------------------------------------------------
// ia_prepared: contract the tensor vertices into the 19 observable
// E/B shape components. These equations keep the source's factors of
// 1/2, sqrt(3/2) and 3/16: changing any of them changes the IA
// amplitude convention, so none of this is a speed optimization. The
// cyclic slot arrays below map output labels, not sorted side
// lengths; sorting only the k values would silently exchange
// source-bin kernels.
static void ia_prepared(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const physics_ia_coefficients* coefficients,
  const double* sides,
  const double z,
  const int remove_alignment,
  double* out,
  const physics_power_row* prepared,
  const physics_triangle_geometry* prepared_geometry,
  const int e_only // output[0..6] only; the public 19-component API uses zero
)
{
  if (coefficients->quadratic == 0.0 && coefficients->density == 0.0
      && coefficients->velocity == 0.0
      && (remove_alignment || coefficients->linear == 0.0)) {
    // With no quadratic shape vertex, subtracting the linear
    // alignment leaves exactly zero. This common NLA case needs no
    // power-spectrum reads.
    const int ncomponent = e_only ? 7 : 19;
    for (int index=0; index<ncomponent; index++) {
      out[index] = 0.0;
    }
    return;
  }

  const physics_power_row row = prepared ? *prepared
    : prepare_power_row(model, z, parameters->power_mode);

  double power[3];
  double vertices[3][3];
  double tree[3];
  for (int leg=0; leg<3; leg++) {
    power[leg] = read_power_row(&row, sides[leg]);
  }

  // A projection owns one immutable triangle geometry for its whole
  // LOS sum and lends it here; a scalar query builds an identical
  // local one. Nothing keeps the geometry after this synchronous call
  // returns.
  physics_triangle_geometry local_geometry;
  const physics_triangle_geometry* geometry = prepared_geometry;
  if (geometry == NULL) {
    local_geometry = triangle_geometry(sides, 1);
    geometry = &local_geometry;
  }

  ia_vertices(geometry, power, coefficients, vertices, tree, e_only);
  const double linear = coefficients->linear;
  const double root_three_halves = sqrt(3.0/2.0);

  // Slot arrays: map cyclic ordering c to the public 19-entry order
  // ddE,dEd,Edd,dEE,EEd,EdE,EEE,ddB,dBd,Bdd,dEB,dBE,EBd,BEd,BdE,EdB,
  // EEB,EBE,BEE, where d is density, E and B are the shape modes, and
  // the three letters name legs 1, 2, 3 in order. In ordering c the
  // helicity arguments sit on legs c, c+1, c+2 (mod 3): the single
  // shape of a 002 term and the lone B of a mixed term sit on leg
  // c+2, and the single density of a 022 term on leg c. For example
  // single_e = {0, 2, 1} sends c = 0, 1, 2 to ddE, Edd, dEd - the
  // entries whose E sits on leg c+2; the other arrays follow the same
  // placement rule.
  const int single_e[3] = {0, 2, 1};
  const int double_e[3] = {3, 5, 4};
  const int single_b[3] = {7, 9, 8};
  const int mixed_eb[3] = {10, 14, 12};
  const int mixed_be[3] = {11, 15, 13};
  const int double_e_b[3] = {16, 18, 17};

  for (int cyclic=0; cyclic<3; cyclic++) {
    out[single_e[cyclic]] = 0.5*(root_three_halves
      *tensor002(vertices, tree, linear, cyclic, 0)
      - tensor002(vertices, tree, linear, cyclic, 2));
    out[double_e[cyclic]] = root_three_halves/4.0
      *(root_three_halves*tensor022(vertices, tree, linear, cyclic, 0, 0)
      - tensor022(vertices, tree, linear, cyclic, 0, 2)
      - tensor022(vertices, tree, linear, cyclic, 2, 0));
    // The projection contracts only the seven all-E entries, so an
    // e_only caller skips the twelve B and mixed entries here;
    // omitting them changes no kept tensor or radial sum.
    if (!e_only) {
      out[single_b[cyclic]] = -tensor002(vertices, tree, linear, cyclic, 1);
      out[mixed_eb[cyclic]] = -sqrt(3.0/8.0)
        *tensor022(vertices, tree, linear, cyclic, 0, 1);
      out[mixed_be[cyclic]] = -sqrt(3.0/8.0)
        *tensor022(vertices, tree, linear, cyclic, 1, 0);
      out[double_e_b[cyclic]] = -3.0/8.0
        *tensor222(vertices, linear, cyclic, 0, 0, 1);
    }
  }

  out[6] = 3.0/16.0*(root_three_halves
    *tensor222(vertices, linear, 0, 0, 0, 0)
    - tensor222(vertices, linear, 0, 0, 0, 2)
    - tensor222(vertices, linear, 0, 2, 0, 0)
    - tensor222(vertices, linear, 0, 0, 2, 0));

  if (!remove_alignment && parameters->operator_component <= 1) {
    // Linear shapes carry helicity zero only, so the linear-alignment
    // contribution to an all-E correlator is C_1 to the number of
    // shape legs times B_tree. Adding it once here equals the sum of
    // its three placements. A TATT residual request
    // (remove_alignment) never subtracts these large terms because it
    // never adds them.
    const double total_tree = tree_sum_physics(geometry, power);
    const double c1 = coefficients->c1;
    for (int index=0; index<3; index++) {
      out[index] += c1*total_tree;
      out[index + 3] += c1*c1*total_tree;
    }
    out[6] += c1*c1*c1*total_tree;
  }
}


// ---------------------------------------------------------------------------
// physics_ia_bispectra_3pcf: public 19-component IA bispectrum query
// at one triangle and redshift (the header file lists the component
// order).
void physics_ia_bispectra_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const double k1,
  const double k2,
  const double k3,
  const double z,
  const int remove_alignment,
  double* out
)
{
  if (!(model && out && isfinite(z) && z >= 0.0
        && z <= model->zmax)) {
    log_fatal("physics_3pcf: invalid alignment query");
    exit(1);
  }
  check_ia(parameters);
  triangle_physics(k1, k2, k3);

  const double sides[3] = {k1, k2, k3};
  const double growth = read_spline(&model->growth, z, 0);
  const physics_ia_coefficients coefficients = ia_coefficients(model,
    parameters, z, growth);

  ia_prepared(model, parameters, &coefficients, sides, z,
    remove_alignment, out, NULL, NULL, 0);
}


// ---------------------------------------------------------------------------
// physics_ia_same_shape_3pcf: evaluate many triangles that share one
// shape. A common shape occurs along every Limber integral and in
// scale scans of one configuration, so the function prepares the
// geometry once and validates the shape association before launching
// workers: changing one leg alone would require a different geometry.
void physics_ia_same_shape_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const int npoint,
  const double* sides,
  const double* z,
  const int remove_alignment,
  double* out
)
{
  if (!(model && sides && z && out && npoint > 0)) {
    log_fatal("physics_3pcf: invalid common-shape IA buffers");
    exit(1);
  }
  check_ia(parameters);

  triangle_physics(sides[0], sides[1], sides[2]);
  const physics_triangle_geometry geometry = triangle_geometry(sides, 1);

  for (int point=0; point<npoint; point++) {
    const double* triangle = sides + (size_t)point*3;
    triangle_physics(triangle[0], triangle[1], triangle[2]);
    if (!(isfinite(z[point]) && z[point] >= 0.0
          && z[point] <= model->zmax)) {
      log_fatal("physics_3pcf: invalid common-shape IA redshift");
      exit(1);
    }
    for (int leg=1; leg<3; leg++) {
      // Common-shape tolerance: every triangle must share one shape
      // for the prepared geometry to apply. Side ratios formed from a
      // common scaling agree to a few units in the last place, and
      // 64*DBL_EPSILON leaves room for a caller that scaled through
      // several rounded operations; a genuinely different shape
      // misses by far more.
      const double ratio = triangle[leg]/triangle[0];
      const double reference = sides[leg]/sides[0];
      if (!(fabs(ratio - reference)
            <= 64.0*DBL_EPSILON*fmax(fabs(reference), fabs(ratio)))) {
        log_fatal("physics_3pcf: IA same-shape inputs change a triangle "
          "side ratio");
        exit(1);
      }
    }
  }

  // Every point evaluates the same triangle shape at its own scale
  // and redshift: the workers share the one prepared geometry
  // read-only and write disjoint 19-entry output rows, so the
  // parallel loop needs no synchronization.
  #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
  for (int point=0; point<npoint; point++) {
    const double growth = read_spline(&model->growth, z[point], 0);
    const physics_ia_coefficients coefficients = ia_coefficients(model,
      parameters, z[point], growth);
    ia_prepared(model, parameters, &coefficients, sides + (size_t)point*3,
      z[point], remove_alignment, out + (size_t)point*19, NULL, &geometry, 0);
  }
}


// ---------------------------------------------------------------------------
// field_kernel_physics: one source bin's radial kernel on the LOS
// nodes, with the common factor chi (1 + z) restored. The field
// argument selects the content: -1 the combined G + C_1 W kernel of
// the NLA baseline, 0 the lensing kernel G alone, 1 the intrinsic
// kernel I = C_1 W alone, all on the same finite kernel grid. G is
// the gravitational lensing efficiency, I the intrinsic-shape field,
// and W the normalized source density per unit distance.
//
// The NLA baseline splines G + C_1 W sampled on the lensing kernel
// grid, while the TATT residual uses the pure G and the source-grid
// W. Both represent the same continuum window, but their finite-grid
// errors differ in fastnc, and this implementation keeps that
// difference on purpose. G and I come out separately only when a
// placement diagnostic asks for one field; the ordinary total keeps
// its original combined-kernel arithmetic.
static void field_kernel_physics(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const int bin,
  const int field, // -1=G+I, 0=G, 1=I, on the same finite kernel grid
  double* out
)
{
  if (!(model && out && bin >= 0 && bin < model->nbin)) {
    log_fatal("physics_3pcf: invalid source bin");
    exit(1);
  }
  check_ia(parameters);

  const int count = model->kernel_nz;
  double* kernel = calloc1d(count);
  for (int node=0; node<count; node++) {
    const physics_ia_coefficients coefficients = ia_coefficients(model,
      parameters, model->kernel_z[node], model->kernel_growth[node]);
    const size_t index = (size_t)bin*count + node;
    kernel[node] = model->kernel_lens[index]
      + coefficients.c1*model->kernel_shape[index];
    if (field == 0) {
      kernel[node] = model->kernel_lens[index];
    } else if (field == 1) {
      kernel[node] = coefficients.c1*model->kernel_shape[index];
    }
  }

  physics_spline spline = make_spline(model->kernel_chi, kernel, count);
  for (int node=0; node<model->nlos; node++) {
    out[node] = read_spline(&spline, model->los_chi[node], 1)
      *model->los_chi[node]*(1.0 + model->los_z[node]);
  }

  destroy_spline(&spline);
  free(kernel);
}


// ---------------------------------------------------------------------------
// physics_nla_kernel_3pcf: the combined NLA kernel G + C_1 W of one
// bin on the LOS nodes, with chi (1 + z) included.
void physics_nla_kernel_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const int bin,
  double* out
)
{
  field_kernel_physics(model, parameters, bin, -1, out);
}



// ===========================================================================
// [SECTION] LIMBER PROJECTION AND BATCHES
// ===========================================================================

// physics_projector_create_3pcf: precompute the radial weights of one
// ordered source-bin triplet.
//
// PHYSICAL DERIVATION & LOGIC FLOW
//   1. Target quantity: the flat-sky Limber projection of a
//      bispectrum,
//
//        b(ell1, ell2, ell3)
//          = int dchi W1(chi) W2(chi) W3(chi) / chi^4
//              * B(ell1/chi, ell2/chi, ell3/chi; z(chi)),
//
//      with W_i the radial kernel of leg i. The trapezoid weights
//      dchi turn the integral into a sum over LOS nodes, so each node
//      needs only the product weight dchi W1 W2 W3 / chi^4.
//   2. Baseline weight (weights[0..nlos-1]): the kernels from
//      field_kernel_physics already carry their chi (1 + z) factor,
//      so dchi K1 K2 K3 / chi^4 is the matter-baseline weight
//      directly.
//   3. Residual weights (seven more per node): the TATT residual
//      needs every placement of the intrinsic kernel on one, two or
//      three legs. The stored lensing and shape rows are each the
//      full kernel divided by chi (1 + z), so a product of three
//      rows is W_i W_j W_k / [chi^3 (1 + z)^3]; multiplying by the
//      common measure dchi (1 + z)^3 / chi cancels the (1 + z)
//      powers and completes the Limber factor 1/chi^4, leaving
//      dchi W_i W_j W_k / chi^4 for every placement. The alignment
//      amplitudes do not enter these kernels: they live in the
//      tensor values that multiply the weights at evaluation time.
//   4. The placement bitmask below gates diagnostics that keep only
//      one field placement.
physics_projection_3pcf* physics_projector_create_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const int* bins
)
{
  if (!(model && bins)) {
    log_fatal("physics_3pcf: missing projection model or bins");
    exit(1);
  }
  check_ia(parameters);
  if (!(model->feedback || parameters->feedback_amplitude == 0.0)) {
    log_fatal("physics_3pcf: nonzero paired feedback needs response tables "
      "in the physical model");
    exit(1);
  }
  for (int leg=0; leg<3; leg++) {
    if (!(bins[leg] >= 0 && bins[leg] < model->nbin)) {
      log_fatal("physics_3pcf: projection source bin is out of range");
      exit(1);
    }
  }

  const int count = model->nlos;
  if (!((size_t)8*count <= INT_MAX)) {
    log_fatal("physics_3pcf: projector weight tables exceed the core "
      "allocator count");
    exit(1);
  }

  physics_projection_3pcf* projection = calloc(1,
    sizeof(*projection));
  if (projection == NULL) {
    log_fatal("physics_3pcf: allocation failed");
    exit(1);
  }
  projection->model = model;
  projection->ia = *parameters;
  projection->weights = calloc1d((int)((size_t)8*count));
  projection->coefficients = calloc((size_t)count,
    sizeof(physics_ia_coefficients));
  if (projection->coefficients == NULL) {
    log_fatal("physics_3pcf: allocation failed");
    exit(1);
  }

  double* kernels = calloc1d((int)((size_t)3*count));
  for (int leg=0; leg<3; leg++) {
    const int field = parameters->intrinsic_legs < 0 ? -1
      : (parameters->intrinsic_legs >> leg)&1;
    // Spline C_1 W as a product: multiplying a splined W by C_1 at
    // the LOS node would change the established finite-grid baseline
    // convention.
    field_kernel_physics(model, parameters, bins[leg], field,
      kernels + (size_t)leg*count);
  }

  // One pass per LOS node: evaluate the IA coefficients at the node's
  // redshift, then form the baseline weight and the seven residual
  // placement weights from the stored kernel rows.
  for (int node=0; node<count; node++) {
    const double distance = model->los_chi[node];
    const double redshift = model->los_z[node];
    const double growth = read_spline(&model->growth, redshift, 0);
    projection->coefficients[node] = ia_coefficients(model,
      parameters, redshift, growth);

    double lens[3];
    double shape[3];
    for (int leg=0; leg<3; leg++) {
      lens[leg] = model->los_lens[(size_t)bins[leg]*count + node];
      shape[leg] = model->los_shape[(size_t)bins[leg]*count + node];
    }

    projection->weights[node] = model->los_dchi[node]
      *kernels[node]*kernels[count + node]*kernels[2*count + node]
      /pow(distance, 4);
    // Diagnostics that isolate a quadratic operator (2 to 4) carry no
    // matter baseline; the NLA diagnostic (1) keeps it.
    if (parameters->operator_component >= 2) {
      projection->weights[node] = 0.0;
    }

    const double measure = model->los_dchi[node]
      *pow(1.0 + redshift, 3)/distance;
    const double residual[7] = {
      lens[0]*lens[1]*shape[2],
      lens[0]*shape[1]*lens[2],
      shape[0]*lens[1]*lens[2],
      lens[0]*shape[1]*shape[2],
      shape[0]*shape[1]*lens[2],
      shape[0]*lens[1]*shape[2],
      shape[0]*shape[1]*shape[2]
    };

    for (int term=0; term<7; term++) {
      // placement[term] encodes which legs of residual[term] carry
      // the intrinsic kernel: bit j (value 2^j) is set when ordered
      // leg j+1 is intrinsic. The residual order above is
      // llI, lIl, Ill, lII, IIl, IlI, III, so the masks read
      // {4, 2, 1, 6, 3, 5, 7}, matching the intrinsic_legs convention
      // of physics_3pcf.h (4 = GGI, 3 = IIG).
      const int placement[7] = {4, 2, 1, 6, 3, 5, 7};
      projection->weights[(size_t)(term + 1)*count + node]
        = measure*residual[term];
      if (parameters->intrinsic_legs >= 0
          && parameters->intrinsic_legs != placement[term]) {
        projection->weights[(size_t)(term + 1)*count + node] = 0.0;
      }
    }
  }

  free(kernels);

  return projection;
}


// ---------------------------------------------------------------------------
// physics_projector_destroy_3pcf: free the owned weights and
// coefficients; a NULL projection is a no-op.
void physics_projector_destroy_3pcf(physics_projection_3pcf* projection)
{
  if (projection != NULL) {
    free(projection->weights);
    free(projection->coefficients);
    free(projection);
  }
}


// ---------------------------------------------------------------------------
// physics_projected_3pcf: the Limber-projected bispectrum of one
// prepared triplet at one angular triangle. component 0 returns the
// total E-mode prediction, 1 the matter baseline with NLA kernels,
// and 2 the TATT residual alone.
double physics_projected_3pcf(
  const physics_projection_3pcf* projection,
  const double ell1,
  const double ell2,
  const double ell3,
  const int component
)
{
  if (!(projection && component >= 0 && component <= 2)) {
    log_fatal("physics_3pcf: invalid projected component");
    exit(1);
  }

  triangle_physics(ell1, ell2, ell3);

  const physics_model_3pcf* model = projection->model;
  const int count = model->nlos;
  const double angular_sides[3] = {ell1, ell2, ell3};
  const physics_triangle_geometry geometry = triangle_geometry(angular_sides,
    component != 1);

  double baseline = 0.0;
  double residual = 0.0;
  // Walk the LOS nodes in fixed order: each node scales the angular
  // sides to k = ell/chi, evaluates the matter bispectrum with its
  // baryon and feedback ratios and the seven all-E residual terms,
  // and contracts them with the node's precomputed weights. The
  // weights carry all tomography, so one matter fit and one tensor
  // evaluation per node suffice.
  for (int node=0; node<count; node++) {
    const double distance = model->los_chi[node];
    const double redshift = model->los_z[node];
    const double sides[3] = {ell1/distance, ell2/distance, ell3/distance};

    if (component != 2) {
      baseline += projection->weights[node]*matter_prepared(model,
        model->baseline ? NULL : model->los_bh + node, sides, redshift, 0,
        model->los_power + node,
        &geometry)
        *baryon_physics(sides, redshift, projection->ia.fb,
        projection->ia.suppress_only)
        *feedback_bispectrum_3pcf(model->feedback, sides, redshift,
        projection->ia.feedback_amplitude);
    }

    if (component != 1) {
      double intrinsic[19];
      ia_prepared(model, &projection->ia, projection->coefficients + node,
        sides, redshift, 1, intrinsic, model->los_power
        + (size_t)projection->ia.power_mode*count + node, &geometry, 1);
      for (int term=0; term<7; term++) {
        residual += projection->weights[(size_t)(term + 1)*count + node]
          *intrinsic[term];
      }
    }
  }

  return baseline + residual;
}


// ---------------------------------------------------------------------------
// physics_projected_puv_extension_3pcf: evaluate the TATT residual
// for a corner of the rectangular PUV interpolation grid (defined at
// triangle_geometry's off-triangle branch). The rectangular grid
// contains corners that close no triangle, so this entry point skips
// the strict validator and follows the source's continuation: clipped
// vertex angles, and zero at a zero closing side.
double physics_projected_puv_extension_3pcf(
  const physics_projection_3pcf* projection,
  const double ell1,
  const double ell2,
  const double ell3
)
{
  if (!(projection && isfinite(ell1) && isfinite(ell2)
        && isfinite(ell3) && ell1 > 0.0 && ell2 > 0.0 && ell3 >= 0.0)) {
    log_fatal("physics_3pcf: invalid source PUV interpolation corner");
    exit(1);
  }

  if (ell3 == 0.0) {
    return 0.0;
  }

  const physics_model_3pcf* model = projection->model;
  const int count = model->nlos;
  const double angular_sides[3] = {ell1, ell2, ell3};
  const physics_triangle_geometry geometry = triangle_geometry(angular_sides,
    1);

  double sum = 0.0;
  for (int node=0; node<count; node++) {
    const double distance = model->los_chi[node];
    const double sides[3] = {ell1/distance, ell2/distance, ell3/distance};
    double intrinsic[19];
    ia_prepared(model, &projection->ia, projection->coefficients + node,
      sides, model->los_z[node], 1, intrinsic, model->los_power
      + (size_t)projection->ia.power_mode*count + node, &geometry, 1);
    for (int term=0; term<7; term++) {
      // fastnc's IA bispectrum class passes each component through
      // numpy's nan_to_num, which replaces every non-finite entry
      // with zero. Keep that replacement for the interpolation fill
      // only: a failed direct physical evaluation must stay visible.
      if (isfinite(intrinsic[term])) {
        sum += projection->weights[(size_t)(term + 1)*count + node]
          *intrinsic[term];
      }
    }
  }

  return sum;
}


// ---------------------------------------------------------------------------
// same_ia_parameters: exact field comparison; a batch requires
// identical IA settings, not merely physically equivalent ones.
static int same_ia_parameters(
  const physics_ia_3pcf* first,
  const physics_ia_3pcf* second
)
{
  return first->a1 == second->a1 && first->eta1 == second->eta1
    && first->a2 == second->a2 && first->eta2 == second->eta2
    && first->bta == second->bta && first->ct == second->ct
    && first->zpivot == second->zpivot && first->c1rho == second->c1rho
    && first->fb == second->fb && first->power_mode == second->power_mode
    && first->ct_mode == second->ct_mode
    && first->suppress_only == second->suppress_only
    && first->operator_component == second->operator_component
    && first->intrinsic_legs == second->intrinsic_legs
    && first->feedback_amplitude == second->feedback_amplitude;
}


// ---------------------------------------------------------------------------
// physics_projection_batch_create_3pcf: validate the projectors and
// repack their weights so that, at one LOS node and term, the values
// of all triplets sit in adjacent doubles. The evaluation loop then
// loads a pair of triplets with one unaligned SIMD read instead of
// gathering across the separate projector arrays.
physics_projection_batch_3pcf* physics_projection_batch_create_3pcf(
  const int nprojection, // number of ordered source-triplet outputs
  physics_projection_3pcf* const* projections // validated before any worker
)
{
  if (!(nprojection > 0 && projections && projections[0]
        && projections[0]->model)) {
    log_fatal("physics_3pcf: invalid prepared projection batch");
    exit(1);
  }

  const physics_projection_3pcf* first = projections[0];
  const int count = first->model->nlos;

  if (!(count > 0 && (size_t)count <= SIZE_MAX/8)) {
    log_fatal("physics_3pcf: prepared projection radial size overflows");
    exit(1);
  }
  const size_t rows = (size_t)count*8;
  if (!((size_t)nprojection <= SIZE_MAX/rows)) {
    log_fatal("physics_3pcf: prepared projection weight count overflows");
    exit(1);
  }
  const size_t nweight = rows*(size_t)nprojection;
  if (!(nweight <= INT_MAX)) {
    log_fatal("physics_3pcf: prepared projection weights exceed the core "
      "allocator count");
    exit(1);
  }

  for (int triplet=0; triplet<nprojection; triplet++) {
    if (!(projections[triplet]
          && projections[triplet]->model == first->model
          && same_ia_parameters(&projections[triplet]->ia, &first->ia))) {
      log_fatal("physics_3pcf: batch projections must share one model and "
        "the same IA parameters");
      exit(1);
    }
  }

  physics_projection_batch_3pcf* batch = calloc(1, sizeof(*batch));
  if (batch == NULL) {
    log_fatal("physics_3pcf: allocation failed");
    exit(1);
  }

  batch->first = first;
  batch->nprojection = nprojection;
  batch->weights = calloc1d((int)nweight);

  // Transpose the per-projector [term,node] storage into the packed
  // [node][term][triplet] order, copying every weight exactly once
  // during this serial construction.
  for (int node=0; node<count; node++) {
    for (int term=0; term<8; term++) {
      const size_t target = ((size_t)node*8 + term)*nprojection;
      for (int triplet=0; triplet<nprojection; triplet++) {
        batch->weights[target + triplet]
          = projections[triplet]->weights[(size_t)term*count + node];
      }
    }
  }

  return batch;
}


// ---------------------------------------------------------------------------
void physics_projection_batch_destroy_3pcf(
  physics_projection_batch_3pcf* batch // borrowed projector/model stay owned
)
{
  if (batch != NULL) {
    free(batch->weights);
    free(batch);
  }
}


// ---------------------------------------------------------------------------
// projected_node_physics: the shared per-node physics of both batch
// layouts. The caller initializes intrinsic[19] and preserves the
// accumulation order; this helper changes no triangle geometry, power
// prescription, baryon convention or PUV treatment between the two
// layouts, so both produce identical per-node physics.
static double projected_node_physics(
  const physics_projection_3pcf* first,
  const double* angular_sides,
  const physics_triangle_geometry* geometry,
  const int node,
  const int component,
  const int puv_extension,
  double* intrinsic
)
{
  const physics_model_3pcf* model = first->model;
  const int count = model->nlos;
  const double distance = model->los_chi[node];
  const double redshift = model->los_z[node];
  const double sides[3] = {angular_sides[0]/distance,
    angular_sides[1]/distance, angular_sides[2]/distance};

  double baseline = 0.0;
  if (component != 2) {
    baseline = matter_prepared(model,
      model->baseline ? NULL : model->los_bh + node,
      sides, redshift, 0, model->los_power + node, geometry)
      *baryon_physics(sides, redshift,
      first->ia.fb, first->ia.suppress_only)
      *feedback_bispectrum_3pcf(model->feedback, sides, redshift,
      first->ia.feedback_amplitude);
  }

  if (component != 1) {
    ia_prepared(model, &first->ia, first->coefficients + node,
      sides, redshift, 1, intrinsic, model->los_power
      + (size_t)first->ia.power_mode*count + node, geometry, 1);
    if (puv_extension) {
      for (int term=0; term<7; term++) {
        if (!isfinite(intrinsic[term])) {
          intrinsic[term] = 0.0;
        }
      }
    }
  }

  return baseline;
}


// ---------------------------------------------------------------------------
// physics_projected_batch_3pcf: project one triangle for many ordered
// source triplets at once. The expensive three-dimensional physics
// depends on the triangle and redshift only, whereas tomography
// enters through the radial weights, so the function evaluates the
// physics once per LOS node and contracts it with every triplet. It
// keeps no array with dimensions [triangle, z, triplet, IA term]: a
// caller supplies one output vector per worker, and each output
// accumulates its LOS nodes in fixed order.
void physics_projected_batch_3pcf(
  const int nprojection,                      // ordered source triplets
  physics_projection_3pcf* const* projections, // shared model and IA
  const double ell1,                          // first Fourier side
  const double ell2,                          // second Fourier side
  const double ell3,                          // closing side
  const int component,                        // 0 total, 1 baseline, 2 IA
  const int puv_extension,                    // nonzero: PUV corner rules
  double* out                                 // [nprojection]
)
{
  if (!(nprojection > 0 && projections && out
        && component >= 0 && component <= 2)) {
    log_fatal("physics_3pcf: invalid batch projection input");
    exit(1);
  }
  const physics_projection_3pcf* first = projections[0];
  if (!(first != NULL)) {
    log_fatal("physics_3pcf: missing batch projection");
    exit(1);

  }
  if (!puv_extension) {
    triangle_physics(ell1, ell2, ell3);
  } else {
    if (!(component == 2 && isfinite(ell1) && isfinite(ell2)
          && isfinite(ell3) && ell1 > 0.0 && ell2 > 0.0 && ell3 >= 0.0)) {
      log_fatal("physics_3pcf: invalid batch PUV corner");
      exit(1);
    }
  }

  for (int triplet=0; triplet<nprojection; triplet++) {
    if (!(projections[triplet]
          && projections[triplet]->model == first->model
          && same_ia_parameters(&projections[triplet]->ia, &first->ia))) {
      log_fatal("physics_3pcf: batch projections must share one model and "
        "the same IA parameters");
      exit(1);
    }
    out[triplet] = 0.0;
  }

  if (puv_extension && ell3 == 0.0) {
    return;
  }

  const physics_model_3pcf* model = first->model;
  const int count = model->nlos;
  const double angular_sides[3] = {ell1, ell2, ell3};
  const physics_triangle_geometry geometry = triangle_geometry(angular_sides,
    component != 1);
  for (int node=0; node<count; node++) {
    double intrinsic[19] = {0.0};
    const double baseline = projected_node_physics(first, angular_sides,
      &geometry, node, component, puv_extension, intrinsic);

    // scalar: at this node every ordered source triplet t adds
    //   out[t] += weights_t[node]*baseline;                  (matter)
    //   out[t] += weights_t[(term+1)*count + node]*b_E[term]; (7 IA)
    // with weights_t the triplet's own radial weights and b_E the
    // seven all-E tensor values, which every triplet shares. The
    // vector loop runs those statements for two triplets at once:
    // lane 0 holds triplet t and lane 1 holds triplet t + 1. The
    // lanes never mix; each continues its own radial sum in node
    // order, and the scalar branch below finishes an odd final
    // triplet.
    int triplet = 0;
    for (; triplet + 1<nprojection; triplet += 2) {
      const double* restrict first_weights = projections[triplet]->weights;
      const double* restrict second_weights
        = projections[triplet + 1]->weights;
      // Adjacent lanes own different ordered source-triplet sums. The
      // output storage does not overlap either immutable weight array;
      // restrict states this promise for reads through the local pointers.
      // Load the two running sums out[triplet] and out[triplet + 1];
      // loadu needs no vector-aligned address.
      physics_v2d sum = simde_mm_loadu_pd(out + triplet);
      if (component != 2) {
        // Pair the two triplets' baseline weights at this node;
        // set_pd places its first argument in the high lane, so lane 1
        // holds triplet t + 1 and lane 0 holds triplet t.
        const physics_v2d weight = simde_mm_set_pd(second_weights[node],
          first_weights[node]);
        // Broadcast the shared matter value of this node to both lanes.
        const physics_v2d value = simde_mm_set1_pd(baseline);
        // Multiply each triplet's weight by the matter value and add
        // the product to that triplet's running sum.
        sum = simde_mm_add_pd(sum, simde_mm_mul_pd(weight, value));
      }
      if (component != 1) {
        for (int term=0; term<7; term++) {
          const size_t index = (size_t)(term + 1)*count + node;
          // Gather one radial weight per triplet; both share this
          // vertex. set_pd again puts triplet t + 1 in the high lane.
          const physics_v2d weight = simde_mm_set_pd(second_weights[index],
            first_weights[index]);
          // Broadcast the shared tensor value of this term.
          const physics_v2d value = simde_mm_set1_pd(intrinsic[term]);
          // Multiply and accumulate, preserving the scalar node/term
          // order independently in each lane.
          sum = simde_mm_add_pd(sum, simde_mm_mul_pd(weight, value));
        }
      }
      // Store both updated sums back; the two output cells belong to
      // these triplets alone.
      simde_mm_storeu_pd(out + triplet, sum);
    }

    if (triplet < nprojection) {
      const double* restrict weights = projections[triplet]->weights;
      if (component != 2) {
        // Match the vector lane's rounded product before accumulation.
        // Adding another triplet must not change this triplet's value.
        volatile double term = weights[node]*baseline;
        out[triplet] += term;
      }
      if (component != 1) {
        for (int term=0; term<7; term++) {
          volatile double contribution =
            weights[(size_t)(term + 1)*count + node]*intrinsic[term];
          out[triplet] += contribution;
        }
      }
    }
  }
}


// ---------------------------------------------------------------------------
// physics_projection_batch_evaluate_3pcf: the packed-weight variant
// of the batch projection. The physics per node is identical to
// physics_projected_batch_3pcf; only the weight layout differs, with
// all triplets of one node and term adjacent in memory.
void physics_projection_batch_evaluate_3pcf(
  const physics_projection_batch_3pcf* batch, // immutable packed preparation
  const double ell1, // ordered first Fourier side
  const double ell2, // ordered second Fourier side
  const double ell3, // closing side
  const int component, // total, baseline or residual
  const int puv_extension, // source continuation for residual grid corners
  double* out // private [nprojection] output, never aliases inputs
)
{
  if (!(batch && out && component >= 0 && component <= 2)) {
    log_fatal("physics_3pcf: invalid prepared batch projection input");
    exit(1);
  }
  if (!puv_extension) {
    triangle_physics(ell1, ell2, ell3);
  } else {
    if (!(component == 2 && isfinite(ell1) && isfinite(ell2)
          && isfinite(ell3) && ell1 > 0.0 && ell2 > 0.0 && ell3 >= 0.0)) {
      log_fatal("physics_3pcf: invalid batch PUV corner");
      exit(1);
    }
  }

  const int nprojection = batch->nprojection;
  for (int triplet=0; triplet<nprojection; triplet++) {
    out[triplet] = 0.0;
  }

  if (puv_extension && ell3 == 0.0) {
    return;
  }

  const physics_projection_3pcf* first = batch->first;
  const int count = first->model->nlos;
  const double angular_sides[3] = {ell1, ell2, ell3};
  const physics_triangle_geometry geometry = triangle_geometry(angular_sides,
    component != 1);
  for (int node=0; node<count; node++) {
    double intrinsic[19] = {0.0};
    const double baseline = projected_node_physics(first, angular_sides,
      &geometry, node, component, puv_extension, intrinsic);

    const double* restrict weights = batch->weights
      + (size_t)node*8*nprojection;

    // scalar: at this node every ordered source triplet t adds
    //   out[t] += weights[t]*baseline;                      (matter)
    //   out[t] += weights[(term+1)*nprojection + t]*b_E[term]; (7 IA)
    // with b_E the shared all-E tensor values. Lane 0 carries triplet
    // t and lane 1 triplet t + 1; the packed [node][term][triplet]
    // layout makes both weights adjacent, so plain loads replace the
    // gathers of the unpacked path, and the scalar branch below
    // finishes an odd final triplet.
    int triplet = 0;
    for (; triplet + 1<nprojection; triplet += 2) {
      // Each lane continues one triplet's existing sum: load
      // out[triplet] and out[triplet + 1] with one unaligned read.
      physics_v2d sum = simde_mm_loadu_pd(out + triplet);
      if (component != 2) {
        // One contiguous load reads the two adjacent baseline weights
        // weights[triplet] and weights[triplet + 1].
        const physics_v2d weight = simde_mm_loadu_pd(weights + triplet);
        // Broadcast the shared matter value of this node to both lanes.
        const physics_v2d value = simde_mm_set1_pd(baseline);
        // Multiply each triplet's weight by the matter value and add
        // the product to that triplet's running sum.
        sum = simde_mm_add_pd(sum, simde_mm_mul_pd(weight, value));
      }
      if (component != 1) {
        for (int term=0; term<7; term++) {
          const size_t index = (size_t)(term + 1)*nprojection + triplet;
          // Term planes keep contiguous triplets, including in odd
          // batches, so one load reads both weights of this term.
          const physics_v2d weight = simde_mm_loadu_pd(weights + index);
          // Broadcast the shared tensor value of this term.
          const physics_v2d value = simde_mm_set1_pd(intrinsic[term]);
          // Multiply and accumulate without exchanging terms or nodes.
          sum = simde_mm_add_pd(sum, simde_mm_mul_pd(weight, value));
        }
      }
      // Store both updated sums back to their private output cells.
      simde_mm_storeu_pd(out + triplet, sum);
    }

    if (triplet < nprojection) {
      if (component != 2) {
        // The odd triplet rounds its product before adding, exactly as
        // the separate multiply/add SIMD instructions do for a full pair.
        volatile double term = weights[triplet]*baseline;
        out[triplet] += term;
      }
      if (component != 1) {
        for (int term=0; term<7; term++) {
          volatile double contribution =
            weights[(size_t)(term + 1)*nprojection + triplet]*intrinsic[term];
          out[triplet] += contribution;
        }
      }
    }
  }
}



// ===========================================================================
// [SECTION] MULTIPOLES, ACCESSORS AND NLA ANGULAR POWER
// ===========================================================================

// gauss_legendre_physics: Gauss-Legendre nodes and weights on the
// open interval (-1, 1). The nodes avoid both endpoints, including
// the degenerate mu = 1 triangle at ell1 = ell2 whose closing side
// would vanish. Each node is a root of the Legendre polynomial of
// degree count, refined by Newton iteration; mirror roots share
// weights by symmetry. This helper runs before worker launch, and
// afterwards nothing reads or writes a scientific table.
static void gauss_legendre_physics(
  const int count,
  double* nodes,
  double* weights
)
{
  for (int root=0; root<(count + 1)/2; root++) {
    // Tricomi's initial guess: the k-th root of the degree-n Legendre
    // polynomial lies near cos(pi (k + 3/4)/(n + 1/2)), and Newton
    // converges from it in a few steps for every order used here.
    double coordinate = cos(3.14159265358979323846
      *(root + 0.75)/(count + 0.5));
    double derivative = 0.0;
    for (int iteration=0; iteration<100; iteration++) {
      // Evaluate P_count(coordinate) with the three-term recurrence
      // order*P_order = (2 order - 1) x P_(order-1)
      // - (order - 1) P_(order-2), keeping the previous polynomial
      // for the derivative identity below.
      double previous = 1.0;
      double value = coordinate;
      for (int order=2; order<=count; order++) {
        const double next = ((2.0*order - 1.0)*coordinate*value
          - (order - 1.0)*previous)/order;
        previous = value;
        value = next;
      }
      // P'_n(x) = n (x P_n - P_(n-1))/(x^2 - 1); one Newton step is
      // P_n/P'_n.
      derivative = count*(coordinate*value - previous)
        /(coordinate*coordinate - 1.0);
      const double step = value/derivative;
      coordinate -= step;
      if (fabs(step) <= 4.0*DBL_EPSILON) {
        break;
      }
      if (!(iteration < 99)) {
        log_fatal("physics_3pcf: Legendre root did not converge");
        exit(1);
      }
    }

    nodes[root] = -coordinate;
    nodes[count - 1 - root] = coordinate;
    // Gauss-Legendre weight w = 2/[(1 - x^2) P'_n(x)^2]; the mirrored
    // root -x carries the same weight.
    const double weight = 2.0/((1.0 - coordinate*coordinate)
      *derivative*derivative);
    weights[root] = weight;
    weights[count - 1 - root] = weight;
  }
}


// ---------------------------------------------------------------------------
// physics_multipoles_3pcf: the Legendre coefficients b_L of the
// projected E-mode bispectrum in mu, the cosine of the angle between
// the two Fourier multipole legs:
//
//   b_L(ell_i, ell_j) = (2L + 1)/2 int_{-1}^{1} dmu P_L(mu)
//                         * b(ell_i, ell_j, ell_3),
//   ell_3^2 = ell_i^2 + ell_j^2 - 2 ell_i ell_j mu,
//
// with no Limber +1/2 shift on any leg. Gauss-Legendre quadrature in
// mu integrates a polynomial of degree 2 nmu - 1 exactly, so nmu must
// exceed lmax. The function imposes no leg-sorting symmetry on out.
void physics_multipoles_3pcf(
  const physics_model_3pcf* model,   // immutable physical tables
  const physics_ia_3pcf* parameters, // IA and baryon settings
  const int* bins,                   // ordered source bins of the legs
  const int nell,                    // multipole nodes per leg
  const double* ell,                 // positive multipole values [nell]
  const int lmax,                    // highest Legendre order L
  const int nmu,                     // Gauss nodes in mu; nmu > lmax
  const int component,               // 0 total, 1 baseline, 2 residual
  double* out                        // b_L [lmax+1, nell, nell]
)
{
  if (!(ell && out && nell > 0 && lmax >= 0 && lmax < INT_MAX
        && nmu >= 2 && nmu > lmax && component >= 0 && component <= 2)) {
    log_fatal("physics_3pcf: invalid multipole grid or component");
    exit(1);
  }
  if (!((size_t)nell <= SIZE_MAX/(size_t)nell
        && (size_t)(lmax + 1) <= SIZE_MAX/((size_t)nell*nell)
        && (size_t)(lmax + 1)*nell*nell <= SIZE_MAX/sizeof(double))) {
    log_fatal("physics_3pcf: multipole output size overflows");
    exit(1);
  }

  for (int index=0; index<nell; index++) {
    if (!(isfinite(ell[index]) && ell[index] > 0.0)) {
      log_fatal("physics_3pcf: multipoles require positive ell");
      exit(1);
    }
  }

  physics_projection_3pcf* projection = physics_projector_create_3pcf(
    model, parameters, bins);

  if (!((size_t)(lmax + 1)*nmu <= INT_MAX)) {
    log_fatal("physics_3pcf: Legendre table exceeds the core allocator "
      "count");
    exit(1);
  }

  double* nodes = calloc1d(nmu);
  double* weights = calloc1d(nmu);
  double* legendre = calloc1d((int)((size_t)(lmax + 1)*nmu));
  gauss_legendre_physics(nmu, nodes, weights);

  // Tabulate the quadrature-ready Legendre factors: entry [L][node]
  // holds (L + 1/2) w_node P_L(mu_node), so each b_L below is a plain
  // dot product of a table row with the bispectrum values. The
  // three-term recurrence advances P_L at a fixed node.
  for (int node=0; node<nmu; node++) {
    double previous = 1.0;
    double value = nodes[node];
    legendre[node] = 0.5*weights[node];
    for (int order=1; order<=lmax; order++) {
      legendre[(size_t)order*nmu + node] = (order + 0.5)*weights[node]*value;
      const double next = ((2.0*order + 1.0)*nodes[node]*value
        - order*previous)/(order + 1.0);
      previous = value;
      value = next;
    }
  }

  // Independent ell pairs own disjoint output cells. Each pair
  // evaluates the costly projected bispectrum only nmu times and
  // shares those values across every order L. SIMD lanes accumulate
  // neighboring orders with the same node sequence,
  // sum_L += LegendreWeight[L,node]*B[node], and no reduction tree or
  // scheduling choice changes that order.
  //
  // Each worker owns one padded slice of scratch. The stride rounds
  // nmu up to a multiple of 8 doubles (one 64-byte cache line) and
  // adds a spare line, so two workers never write the same cache
  // line; without the padding, neighboring slices would share a line
  // and every write would bounce it between cores (false sharing).
  // The allocation happens before parallel work, including for the
  // direct scientific reference path.
  const size_t worker_stride = (((size_t)nmu + 7)/8 + 1)*8;
  const size_t workers = (size_t)omp_get_max_threads();
  if (!(workers > 0 && worker_stride <= (size_t)INT_MAX/workers)) {
    log_fatal("physics_3pcf: multipole worker storage overflows");
    exit(1);
  }
  double* scratch = calloc1d((int)(worker_stride*workers));

  #pragma omp parallel if(!omp_in_parallel())
  {
    double* values = scratch + (size_t)omp_get_thread_num()*worker_stride;
    #pragma omp for collapse(2) schedule(static)
    for (int first=0; first<nell; first++) {
      for (int second=0; second<nell; second++) {
        double* restrict output = out + (size_t)first*nell + second;
        for (int node=0; node<nmu; node++) {
          const double difference = ell[first] - ell[second];
          const double third = sqrt(difference*difference
            + 2.0*ell[first]*ell[second]*(1.0 - nodes[node]));
          values[node] = physics_projected_3pcf(projection,
            ell[first], ell[second], third, component);
          if (model->nwindow > 0) {
            values[node] *= physics_window_value_3pcf(model->nwindow,
              model->window, ell[first])
              *physics_window_value_3pcf(model->nwindow,
              model->window, ell[second])
              *physics_window_value_3pcf(model->nwindow,
              model->window, third);
          }
        }
        // scalar: for each Legendre order L,
        //   b_L = sum_node legendre[L][node]*values[node].
        // The vector loop forms two orders at once - lane 0 carries L
        // and lane 1 carries L + 1, both reading the same node
        // sequence - and the scalar tail below covers an odd lmax.
        int order = 0;
        for (; order + 1<=lmax; order += 2) {
          // Both independent L accumulators start at zero.
          physics_v2d sum = simde_mm_setzero_pd();
          for (int node=0; node<nmu; node++) {
            // set_pd places its first argument in the higher lane.
            const physics_v2d factor = simde_mm_set_pd(
              legendre[(size_t)(order + 1)*nmu + node],
              legendre[(size_t)order*nmu + node]);
            // Broadcast this node's bispectrum to the two L values.
            const physics_v2d value = simde_mm_set1_pd(values[node]);
            // Multiply each Legendre weight by the shared bispectrum.
            const physics_v2d term = simde_mm_mul_pd(factor, value);
            // Add in increasing Gauss-node order for each L.
            sum = simde_mm_add_pd(sum, term);
          }
          double pair[2];
          // Store the two lanes before scattering into separate L planes.
          simde_mm_storeu_pd(pair, sum);
          output[(size_t)order*nell*nell] = pair[0];
          output[(size_t)(order + 1)*nell*nell] = pair[1];
        }

        if (order <= lmax) {
          double sum = 0.0;
          for (int node=0; node<nmu; node++) {
            // The same L keeps its bits when a higher requested order
            // moves this scalar tail into a two-lane accumulation.
            volatile double term =
              legendre[(size_t)order*nmu + node]*values[node];
            sum += term;
          }
          output[(size_t)order*nell*nell] = sum;
        }
      }
    }
  }

  free(scratch);
  free(nodes);
  free(weights);
  free(legendre);
  physics_projector_destroy_3pcf(projection);
}


// ---------------------------------------------------------------------------
// physics_radial_3pcf: expose the LOS grids and kernel rows so the
// covariance shares the same physical tables; the model keeps
// ownership for its lifetime.
void physics_radial_3pcf(
  const physics_model_3pcf* model,
  int* nbin,
  int* nlos,
  const double** z,
  const double** chi,
  const double** dchi,
  const double** lens,
  const double** shape
)
{
  if (!(model && nbin && nlos && z && chi && dchi && lens && shape)) {
    log_fatal("physics_3pcf: missing radial output pointer");
    exit(1);
  }

  *nbin = model->nbin;
  *nlos = model->nlos;
  *z = model->los_z;
  *chi = model->los_chi;
  *dchi = model->los_dchi;
  *lens = model->los_lens;
  *shape = model->los_shape;
}


// ---------------------------------------------------------------------------
// physics_omega_m_3pcf: today's matter density fraction of the model.
double physics_omega_m_3pcf(const physics_model_3pcf* model)
{
  if (!(model != NULL)) {
    log_fatal("physics_3pcf: missing model");
    exit(1);
  }
  return model->omega_m;
}


// ---------------------------------------------------------------------------
// Matter fraction used by halo covariance. A supplied baseline takes this
// from the supplied background array; the legacy route retains its w0-wa
// formula (omega_m_physics). Neither route infers Omega_m(z) from B.
double physics_omega_m_z_3pcf(
  const physics_model_3pcf* model,
  const double z
)
{
  if (!(model && isfinite(z) && z >= 0.0)) {
    log_fatal("physics_3pcf: invalid background matter-fraction query");
    exit(1);
  }

  if (NULL != model->baseline) {
    if (!(z <= model->zmax)) { log_fatal(
      "physics_3pcf: Omega_m(z) query exceeds supplied background"); exit(1); }
    const double value = read_spline(&model->omega_m_z, z, 0);
    if (!(isfinite(value))) { log_fatal(
      "physics_3pcf: interpolated Omega_m(z) is nonfinite"); exit(1); }
    if (!(0.0 < value)) { log_fatal(
      "physics_3pcf: interpolated Omega_m(z) is not positive"); exit(1); }
    return value;
  }
  return omega_m_physics(model, z);
}


// ---------------------------------------------------------------------------
// Read two neighboring ell outputs with their prepared integer cells.
// Both redshift rows share the same uniform log-k nodes. Gathers supply
// the two cell endpoints; SIMD lanes represent independent ell values,
// not partial pieces of a radial sum. The source-comparison and endpoint
// extrapolation cases retain their scalar physical readers.
static physics_v2d nla_power_pair(
  const physics_power_row* row,
  const int* indices,
  const double* fractions
)
{
  if (indices[0] < 0 || indices[1] < 0) {
    double values[2];
    for (int lane=0; lane<2; lane++) {
      if (indices[lane] < 0) {
        values[lane] = read_power_row_log(row, fractions[lane]);
      } else {
        const int index = indices[lane];
        const double fraction = fractions[lane];
        // The other lane's extrapolation must not alter this valid
        // cell's interpolation. Round the same subtraction, product
        // and addition as the vector reader below, without scalar FMA.
        volatile double lower_delta = row->lower->y[index + 1]
          - row->lower->y[index];
        volatile double upper_delta = row->upper->y[index + 1]
          - row->upper->y[index];
        volatile double lower_term = fraction*lower_delta;
        volatile double upper_term = fraction*upper_delta;
        volatile double first = row->lower->y[index] + lower_term;
        volatile double second = row->upper->y[index] + upper_term;
        volatile double redshift_delta = second - first;
        volatile double redshift_term = row->fraction*redshift_delta;
        volatile double logarithm = first + redshift_term;
        values[lane] = exp(logarithm);
      }
    }
    // The first argument goes in the high lane (ell index + 1).
    return simde_mm_set_pd(values[1], values[0]);

  }
  const int index0 = indices[0];
  const int index1 = indices[1];
  const double* lower = row->lower->y;
  const double* upper = row->upper->y;
  // Load the two prepared interpolation fractions contiguously.
  const physics_v2d fraction = simde_mm_loadu_pd(fractions);
  // Gather adjacent table endpoints for the two distinct k cells.
  const physics_v2d lower0 = simde_mm_set_pd(lower[index1], lower[index0]);
  const physics_v2d lower1 = simde_mm_set_pd(lower[index1 + 1],
    lower[index0 + 1]);
  const physics_v2d upper0 = simde_mm_set_pd(upper[index1], upper[index0]);
  const physics_v2d upper1 = simde_mm_set_pd(upper[index1 + 1],
    upper[index0 + 1]);
  // Each lane computes y0 + fraction*(y1-y0) in each redshift row.
  const physics_v2d first = simde_mm_add_pd(lower0,
    simde_mm_mul_pd(fraction, simde_mm_sub_pd(lower1, lower0)));
  const physics_v2d second = simde_mm_add_pd(upper0,
    simde_mm_mul_pd(fraction, simde_mm_sub_pd(upper1, upper0)));
  // Both lanes share the precomputed interpolation fraction in redshift.
  const physics_v2d redshift_fraction = simde_mm_set1_pd(row->fraction);
  const physics_v2d logarithm = simde_mm_add_pd(first,
    simde_mm_mul_pd(redshift_fraction, simde_mm_sub_pd(second, first)));
  double values[2];
  // SSE2 has no exponential intrinsic; apply scalar exp to the two
  // independent log-P values and return them to their original lanes.
  simde_mm_storeu_pd(values, logarithm);
  return simde_mm_set_pd(exp(values[1]), exp(values[0]));
}


// ---------------------------------------------------------------------------
// physics_nla_power_batch_3pcf: the Limber angular power spectrum of
// the combined lensing + NLA intrinsic kernel,
//
//   C(ell) = int dchi K1(chi) K2(chi) / chi^2 P(ell/chi; z(chi)),
//
// with K = G + C_1 W from physics_nla_kernel_3pcf (chi (1 + z)
// included) and P the nonlinear power interpolated in redshift (mode
// 2). The trapezoid LOS sum reuses the model's prepared power rows,
// and a nonzero feedback amplitude multiplies the power by its baryon
// response ratio per ell. The function rejects TATT loop parameters
// and the legacy bispectrum-only fb: this spectrum is strictly NLA.
void physics_nla_power_batch_3pcf(
  const physics_model_3pcf* model,   // immutable physical tables
  const physics_ia_3pcf* parameters, // NLA parameters; TATT rejected
  const int bin1,                    // source bin of the first kernel
  const int bin2,                    // source bin of the second kernel
  const int nell,                    // number of multipoles
  const double* ell,                 // positive multipoles [nell]
  double* out                        // C(ell) [nell]
)
{
  check_ia(parameters);
  if (!(model && ell && out && nell > 0)) {
    log_fatal("physics_3pcf: invalid angular power buffers");
    exit(1);
  }
  if (!(model->feedback || parameters->feedback_amplitude == 0.0)) {
    log_fatal("physics_3pcf: nonzero paired feedback needs response tables "
      "in the physical model");
    exit(1);
  }
  if (!(parameters->a2 == 0.0 && parameters->bta == 0.0
        && parameters->ct == 0.0 && parameters->fb == 0.0)) {
    log_fatal("physics_3pcf: NLA power rejects TATT loops and legacy "
      "bispectrum-only fb");
    exit(1);
  }

  const int count = model->nlos;
  double* first = calloc1d(count);
  double* second = calloc1d(count);
  double* radial_weight = calloc1d(count);
  double* logchi = calloc1d(count);
  double* logell = calloc1d(nell);

  physics_nla_kernel_3pcf(model, parameters, bin1, first);
  physics_nla_kernel_3pcf(model, parameters, bin2, second);

  for (int node=0; node<count; node++) {
    const double distance = model->los_chi[node];
    radial_weight[node] = model->los_dchi[node]*first[node]*second[node]
      /(distance*distance);
    logchi[node] = log(distance);
  }

  for (int index=0; index<nell; index++) {
    if (!(isfinite(ell[index]) && ell[index] > 0.0)) {
      log_fatal("physics_3pcf: angular power requires positive finite ell");
      exit(1);
    }
    logell[index] = log(ell[index]);
  }

  const int capacity = nell < 256 ? nell : 256;
  const size_t cells = (size_t)capacity*count;
  if (!(cells <= INT_MAX)) {
    log_fatal("physics_3pcf: angular power scratch exceeds the core "
      "allocator count");
    exit(1);
  }

  int* indices = malloc1d_int((int)cells);
  memset(indices, 0, sizeof(int)*cells);
  double* fractions = calloc1d((int)cells);

  for (size_t start=0; start<(size_t)nell; start += (size_t)capacity) {
    const size_t remaining = (size_t)nell - start;
    const int length = (int)(remaining < (size_t)capacity
      ? remaining : (size_t)capacity);
    // Prepare log k = log ell - log chi, then the integer cell and
    // fraction of each (ell, node) pair. The bounded tile stores only
    // this geometry; both nonlinear-redshift rows reuse its entries,
    // and the workers only read them.
    for (int node=0; node<count; node++) {
      const physics_spline* spectrum
        = model->los_power[(size_t)2*count + node].lower;
      for (int index=0; index<length; index++) {
        const size_t cell = (size_t)node*capacity + index;
        const double logk = logell[start + index] - logchi[node];
        if (model->source_readers || logk < spectrum->x[0]
            || logk > spectrum->x[spectrum->n - 1]) {
          indices[cell] = -1;
          fractions[cell] = logk;
        } else {
          const double position = (logk - spectrum->x[0])
            *spectrum->inverse_spacing;
          const int left = (int)floor(fmax(0.0,
            fmin(spectrum->n - 2.0, position)));
          indices[cell] = left;
          fractions[cell] = position - left;
        }
      }
    }

    const int pairs = length/2;
    // scalar: for one multipole the radial sum is
    //   C(ell) = sum_node radial_weight[node]*P(ell/chi_node; z_node).
    // Each worker owns one pair of multipoles: lane 0 carries the ell
    // at this pair's even index and lane 1 the next ell, both
    // visiting the radial nodes in increasing order with the same
    // prepared cells, so the thread count never changes any sum's
    // rounding. The block after the parallel loop covers an odd final
    // multipole.
    #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
    for (int pair=0; pair<pairs; pair++) {
      const int index = pair*2;
      // Independent ell accumulators start at zero; each visits radial
      // nodes in increasing order, independent of the worker schedule.
      physics_v2d sum = simde_mm_setzero_pd();
      for (int node=0; node<count; node++) {
        const size_t cell = (size_t)node*capacity + index;
        const physics_power_row* row = model->los_power + (size_t)2*count
          + node;
        physics_v2d power = nla_power_pair(row, indices + cell,
          fractions + cell);
        if (parameters->feedback_amplitude != 0.0) {
          const double first_ratio = feedback_power_3pcf(model->feedback,
            ell[start + index]/model->los_chi[node], model->los_z[node],
            parameters->feedback_amplitude);
          const double second_ratio = feedback_power_3pcf(model->feedback,
            ell[start + index + 1]/model->los_chi[node], model->los_z[node],
            parameters->feedback_amplitude);
          // Each lane is one ell, and its own ratio modifies only the
          // NLA power; the immutable DMO rows also serve unmodified
          // IA loops. set_pd places second_ratio, the ratio of the
          // higher ell, in the high lane to match the power lanes.
          power = simde_mm_mul_pd(power,
            simde_mm_set_pd(second_ratio, first_ratio));
        }
        // Broadcast the node's radial weight: the same weight
        // multiplies both ell values.
        const physics_v2d weight = simde_mm_set1_pd(radial_weight[node]);
        // Multiply weight times power per lane and add each product
        // to that lane's radial sum.
        sum = simde_mm_add_pd(sum, simde_mm_mul_pd(weight, power));
      }
      // Store the two sums; this pair owns the two adjacent output
      // cells exclusively.
      simde_mm_storeu_pd(out + start + index, sum);
    }

    if (length%2 != 0) {
      const int index = length - 1;
      // Duplicate the final coordinate into both lanes. Reusing the
      // prepared reader preserves its interpolation arithmetic as well
      // as its radial sum; a separate scalar spline read would differ.
      physics_v2d sum = simde_mm_setzero_pd();
      for (int node=0; node<count; node++) {
        const size_t cell = (size_t)node*capacity + index;
        const int pair_indices[2] = {indices[cell], indices[cell]};
        const double pair_fractions[2] = {fractions[cell], fractions[cell]};
        const physics_power_row* row = model->los_power + (size_t)2*count
          + node;
        physics_v2d power = nla_power_pair(row, pair_indices,
          pair_fractions);
        if (parameters->feedback_amplitude != 0.0) {
          const double ratio = feedback_power_3pcf(model->feedback,
            ell[start + index]/model->los_chi[node], model->los_z[node],
            parameters->feedback_amplitude);
          // Both lanes duplicate the same final ell coordinate, so one
          // broadcast ratio multiplies them identically.
          power = simde_mm_mul_pd(power, simde_mm_set1_pd(ratio));
        }
        // Broadcast the node's radial weight to both duplicate lanes.
        const physics_v2d weight = simde_mm_set1_pd(radial_weight[node]);
        // Multiply and accumulate exactly as in the paired loop.
        sum = simde_mm_add_pd(sum, simde_mm_mul_pd(weight, power));
      }
      // Store both lanes and keep lane 0; lane 1 repeated the same
      // multipole and its duplicate result is discarded.
      double result[2];
      simde_mm_storeu_pd(result, sum);
      out[start + index] = result[0];
    }
  }

  free(indices);
  free(fractions);
  free(logell);
  free(logchi);
  free(radial_weight);
  free(first);
  free(second);
}


// ---------------------------------------------------------------------------
// physics_nla_power_3pcf: single-multipole convenience wrapper over
// the batch spectrum above.
double physics_nla_power_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const int bin1,
  const int bin2,
  const double ell
)
{
  double result;
  physics_nla_power_batch_3pcf(model, parameters, bin1, bin2, 1, &ell, &result);
  return result;
}



// ===========================================================================
// [SECTION] APERTURE-MASS INTEGRATION
// ===========================================================================

// aperture_filter_physics: the compensated-Gaussian aperture filter
// in Fourier space, U(x) = (x^2/2) exp(-x^2/2) with x = ell R. The
// early return protects the far tail: at (ell R)^2 = 1500 the factor
// exp(-750) is near 1e-326, below the smallest normal double (about
// 2.2e-308), so the product would pass through slow subnormal
// arithmetic on its way to a value that contributes nothing.
// Returning zero above the threshold keeps every operation on normal
// numbers and changes no digit of any integral.
static double aperture_filter_physics(
  const double ell,
  const double radius
)
{
  const double scaled = ell*radius;
  const double square = scaled*scaled;
  return square > 1500.0 ? 0.0 : 0.5*square*exp(-0.5*square);
}


// ---------------------------------------------------------------------------
// physics_aperture_integrate_3pcf: the third-moment aperture-mass
// statistic from a projected bispectrum.
//
// A circular aperture of radius R multiplies each Fourier field by
// the filter U(ell R). The three-point delta function removes the
// third Fourier integral, leaving d^2ell1 d^2ell2/(2 pi)^4, and
// statistical isotropy integrates the overall orientation into 2 pi,
// so the quadrature measure below is
//
//   ell1 dell1 * ell2 dell2 * dphi / (2 pi)^3,
//
// with phi the angle between the two integration legs and the
// closing leg |ell1 + ell2|. This is the full-filter statistic at one
// aperture center; a finite side-angle-side (SAS) triangle selection
// is a different estimator with different weights.
//
// Work organization. The loop stores only a bounded tile of projected
// bispectra: each triangle costs one reader call, and every filter
// triplet reuses it. Workers first own triangle values, then own
// pairs of filter outputs, so no two workers share an accumulator,
// and every output visits the triangles in the same fixed order at
// every thread count. All saved physical model arrays stay read-only.
void physics_aperture_integrate_3pcf(
  const void* context,                   // opaque reader state
  physics_projection_reader_3pcf reader, // total E-mode projected B
  const int nell,                        // radial quadrature nodes
  const double* ell,                     // increasing ell nodes [nell]
  const double* dell,                    // quadrature widths [nell]
  const int nphi,                        // angular quadrature nodes
  const double* phi,                     // angles in [0, 2 pi] [nphi]
  const double* dphi,                    // angular widths, sum 2 pi [nphi]
  const int nfilter,                     // aperture-radius triplets
  const double* filters,                 // radii [nfilter,3], ordered legs
  double* out                            // one moment per triplet [nfilter]
)
{
  if (!(context && reader && ell && dell && phi
        && dphi && filters && out && nell > 0 && nphi > 0 && nfilter > 0)) {
    log_fatal("physics_3pcf: invalid full-aperture buffers or dimensions");
    exit(1);
  }
  if (!((size_t)nell <= SIZE_MAX/(size_t)nell
        && (size_t)nphi <= SIZE_MAX/((size_t)nell*nell))) {
    log_fatal("physics_3pcf: aperture triangle count overflows");
    exit(1);
  }

  const size_t ntriangle = (size_t)nell*nell*nphi;

  for (int node=0; node<nell; node++) {
    if (!(isfinite(ell[node]) && ell[node] > 0.0
          && isfinite(dell[node]) && dell[node] > 0.0
          && (node == 0 || ell[node] > ell[node - 1]))) {
      log_fatal("physics_3pcf: invalid full-aperture ell quadrature");
      exit(1);
    }
  }

  double angle_norm = 0.0;
  for (int node=0; node<nphi; node++) {
    if (!(isfinite(phi[node]) && phi[node] >= 0.0
          && phi[node] <= 6.28318530717958647693
          && isfinite(dphi[node]) && dphi[node] > 0.0
          && (node == 0 || phi[node] > phi[node - 1]))) {
      log_fatal("physics_3pcf: invalid full-aperture angular quadrature");
      exit(1);
    }
    angle_norm += dphi[node];
  }

  if (!(fabs(angle_norm - 6.28318530717958647693) < 1.0e-10)) {
    log_fatal("physics_3pcf: full-aperture angular weights must sum to 2pi");
    exit(1);
  }

  for (size_t index=0; index<(size_t)nfilter*3; index++) {
    if (!(isfinite(filters[index]) && filters[index] > 0.0)) {
      log_fatal("physics_3pcf: full-aperture radii must be positive");
      exit(1);
    }
  }

  const size_t capacity = 4096;
  if (!((size_t)nfilter*nell*2 <= INT_MAX)) {
    log_fatal("physics_3pcf: full-aperture filter table exceeds the core "
      "allocator count");
    exit(1);
  }

  double* values = calloc1d((int)capacity);
  double* closure = calloc1d((int)capacity);
  double* filter_table = calloc1d((int)((size_t)nfilter*nell*2));

  for (int filter=0; filter<nfilter; filter++) {
    out[filter] = 0.0;
    for (int leg=0; leg<2; leg++) {
      for (int node=0; node<nell; node++) {
        filter_table[((size_t)filter*2 + leg)*nell + node]
          = aperture_filter_physics(ell[node], filters[(size_t)filter*3 + leg]);
      }
    }
  }

  // 1/(2 pi)^3 of the quadrature measure derived in the header.
  const double measure_constant = 0.00403144180414993615;
  for (size_t start=0; start<ntriangle; start += capacity) {
    const size_t remaining = ntriangle - start;
    const int count = (int)(remaining < capacity ? remaining : capacity);
    // Fill one tile: each worker evaluates independent triangles,
    // decoding its flat index into (first leg, second leg, angle),
    // and writes disjoint entries of values and closure. The closing
    // side follows from the two legs and their angle, and each stored
    // value is the projected bispectrum already multiplied by its
    // quadrature measure.
    #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
    for (int offset=0; offset<count; offset++) {
      const size_t triangle = start + (size_t)offset;
      const int angle = triangle%(size_t)nphi;
      const int second = (triangle/(size_t)nphi)%(size_t)nell;
      const int first = triangle/((size_t)nphi*nell);
      double third = hypot(ell[first] + ell[second]*cos(phi[angle]),
        ell[second]*sin(phi[angle]));
      // Equal legs at phi = pi close to a zero third side exactly;
      // hypot's rounded result can miss zero, so snap it within four
      // units in the last place of pi.
      if (first == second && fabs(phi[angle] - 3.14159265358979323846)
          <= 4.0*DBL_EPSILON*3.14159265358979323846) {
        third = 0.0;
      }
      closure[offset] = third;
      // The compensation U(0) = 0 removes this isolated degenerate
      // triangle, so the reader never receives a zero closing side;
      // do not call a three-positive-side fit with one.
      values[offset] = third == 0.0 ? 0.0
        : reader(context, ell[first], ell[second], third)
        *ell[first]*dell[first]*ell[second]*dell[second]*dphi[angle]
        *measure_constant;
    }

    const int npair = nfilter/2 + nfilter%2;
    // Contract the tile with every filter triplet: each worker owns
    // two adjacent outputs, reuses the per-leg filter tables and
    // evaluates only the closing-side filter per triangle. The scalar
    // branch finishes an odd final filter with the vector path's
    // rounding.
    #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
    for (int pair_index=0; pair_index<npair; pair_index++) {
      const int filter = pair_index*2;
      const double* restrict table = filter_table;
      double* restrict output = out + filter;
      // scalar: each filter triplet f accumulates, over the tile,
      //   out[f] += values[triangle]*U1_f(ell1)*U2_f(ell2)*U3_f(ell3),
      // with values already carrying the quadrature measure. Lane 0
      // holds filter f and lane 1 filter f + 1; both visit the tile's
      // triangles in the same global order and never mix.
      if (filter + 1 < nfilter) {
        // Continue the two independent filter sums from preceding tiles.
        physics_v2d sum = simde_mm_loadu_pd(output);
        for (int offset=0; offset<count; offset++) {
          const size_t triangle = start + (size_t)offset;
          const int second = (triangle/(size_t)nphi)%(size_t)nell;
          const int first = triangle/((size_t)nphi*nell);
          const double weight0 = table[(size_t)filter*2*nell + first]
            *table[((size_t)filter*2 + 1)*nell + second]
            *aperture_filter_physics(closure[offset],
              filters[(size_t)3*filter + 2]);
          const double weight1 = table[(size_t)(filter + 1)*2*nell + first]
            *table[((size_t)(filter + 1)*2 + 1)*nell + second]
            *aperture_filter_physics(closure[offset],
              filters[(size_t)3*filter + 5]);
          // set_pd puts filter+1 in the high lane, filter in the low lane.
          const physics_v2d weights = simde_mm_set_pd(weight1, weight0);
          // Both filters use the same projected B and integration measure.
          const physics_v2d value = simde_mm_set1_pd(values[offset]);
          // The scalar operation is sum_f += B_measure*U1_f*U2_f*U3_f.
          const physics_v2d term = simde_mm_mul_pd(weights, value);
          // Each lane retains increasing global triangle order.
          sum = simde_mm_add_pd(sum, term);
        }
        // Two valid adjacent output entries belong to this worker alone.
        simde_mm_storeu_pd(output, sum);
      } else {
        double sum = output[0];
        for (int offset=0; offset<count; offset++) {
          const size_t triangle = start + (size_t)offset;
          const int second = (triangle/(size_t)nphi)%(size_t)nell;
          const int first = triangle/((size_t)nphi*nell);
          const double weight = table[(size_t)filter*2*nell + first]
            *table[((size_t)filter*2 + 1)*nell + second]
            *aperture_filter_physics(closure[offset],
              filters[(size_t)3*filter + 2]);
          // Retain the vector path's product/addition rounding when
          // this is the last unpaired filter in the request.
          volatile double term = values[offset]*weight;
          sum += term;
        }
        output[0] = sum;
      }
    }
  }

  free(values);
  free(closure);
  free(filter_table);
}


// ---------------------------------------------------------------------------
// direct_aperture_reader: adapt a projection to the integrator's
// reader callback - the total E-mode projected bispectrum with the
// pixel window applied once per leg.
static double direct_aperture_reader(
  const void* context,
  const double ell1,
  const double ell2,
  const double ell3
)
{
  const physics_projection_3pcf* projection = context;
  const physics_model_3pcf* model = projection->model;
  double value = physics_projected_3pcf(context, ell1, ell2, ell3, 0);

  if (model->nwindow > 0) {
    value *= physics_window_value_3pcf(model->nwindow, model->window, ell1)
      *physics_window_value_3pcf(model->nwindow, model->window, ell2)
      *physics_window_value_3pcf(model->nwindow, model->window, ell3);
  }

  return value;
}


// ---------------------------------------------------------------------------
// physics_aperture_full_3pcf: convenience entry point - build the
// projector for one ordered bin triplet, run the shared integrator
// with the direct reader, and destroy the projector.
void physics_aperture_full_3pcf(
  const physics_model_3pcf* model,   // immutable physical tables
  const physics_ia_3pcf* parameters, // IA and baryon settings
  const int* bins,                   // ordered source bins of the legs
  const int nell,                    // radial quadrature nodes
  const double* ell,                 // increasing ell nodes [nell]
  const double* dell,                // quadrature widths [nell]
  const int nphi,                    // angular quadrature nodes
  const double* phi,                 // angles in [0, 2 pi] [nphi]
  const double* dphi,                // angular widths, sum 2 pi [nphi]
  const int nfilter,                 // aperture-radius triplets
  const double* filters,             // radii [nfilter,3], ordered legs
  double* out                        // one moment per triplet [nfilter]
)
{
  physics_projection_3pcf* projection = physics_projector_create_3pcf(
    model, parameters, bins);

  physics_aperture_integrate_3pcf(projection, direct_aperture_reader,
    nell, ell, dell, nphi, phi, dphi, nfilter, filters, out);

  physics_projector_destroy_3pcf(projection);
}
