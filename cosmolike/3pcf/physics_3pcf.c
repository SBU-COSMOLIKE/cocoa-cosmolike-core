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
#ifdef _OPENMP
#include <omp.h>
#else
static int omp_in_parallel(void) { return 0; }
static int omp_get_max_threads(void) { return 1; }
static int omp_get_thread_num(void) { return 0; }
#endif

// ===========================================================================
// Physical bispectra and their flat-sky, unshifted Limber projection
// ===========================================================================
// A bispectrum measures the correlation of three Fourier amplitudes whose
// wavevectors close into a triangle. Gravitational evolution produces a
// matter bispectrum even from initially Gaussian fluctuations. Intrinsic
// alignment adds correlations between density and galaxy shape. The first
// part below prepares the supplied physical tables. Later parts evaluate
// Takahashi's BiHalofit and the active fastnc TATT tensor equations, then
// integrate these three-dimensional quantities over comoving distance.
//
// Formula and software provenance: Sunao Sugiyama's MIT-licensed fastnc,
// tatt commit 7b2336ce37a909bed51f07f32a2a4f096e71b6ce, halofit.py,
// ia_bispectra.py and bispectrum.py; the IA work includes Rafael Gomes.
// BiHalofit: Takahashi et al., arXiv:1911.07886. TATT and velocity shear:
// Gomes et al., arXiv:2601.09133. The active source equations are preserved;
// the unused alternative direct-equation helpers are not silently selected.
// See the module's credits and the archived study for normalization issues.
//
// Units: k is h/Mpc, chi is Mpc/h, P is (Mpc/h)^3 and B is (Mpc/h)^6.
// Lensing angles and Fourier multipoles are dimensionless. The model copies
// its inputs and normalizes linear P to sigma8. Production converts the
// supplied tables to immutable uniform arrays during preparation; repeated
// queries use arithmetic cell indices and linear interpolation. The named
// source comparison mode retains cubic reads of the supplied tables.
// Recreate it after changing cosmology, power, distances, growth, sources
// or radial grids. IA changes need only a new small projection object.
// There are no file-scope mutable caches and no lazy initialization.
// Source efficiencies use (chi_s-chi)/chi_s, so the public model requires
// spatial flatness: |Omega_m+Omega_de-1| <= 1e-8. This tolerance admits
// scalar rounding only; curved-distance lensing is not implemented.

typedef simde__m128d physics_v2d;

typedef struct {
  int n;
  double* x;
  double* y;
  double* second;
  double inverse_spacing; // positive only for prepared uniform linear reads
  double slope_low;
  double slope_high;
} physics_spline;

// A redshift row identifies the two P(k) arrays and their interpolation
// fraction. It is prepared once per LOS node, shared by all triangle
// sides and source triplets, and contains no mutable interpolation state.
typedef struct {
  const physics_spline* lower;
  const physics_spline* upper;
  double fraction;
  double growth;
} physics_power_row;

typedef struct {
  double radius;
  double loga;
  double b;
  double c;
  double logalpha;
  double alpha_shape;
  double logbeta;
  double gamma;
  double f;
  double g;
  double h;
  double m;
  double n;
  double mu;
  double nu;
  double p;
  double e;
  double d;
} physics_bh_coefficients;

// Common rescaling of all three sides changes their physical k values,
// but leaves these dimensionless triangle factors unchanged. They are
// prepared once per angular triangle before the radial integration.
typedef struct {
  double sides[3];
  double gravity[3];
  double pair_cosine[3];
  double first_x[3];
  double first_y[3];
  double second_x[3];
  double second_y[3];
  double dot[3];
  double sine_squared[3];
  double minimum_ratio;
  double flattened_ratio;
} physics_triangle_geometry;

typedef struct {
  double c1;
  double linear;
  double velocity;
  double quadratic;
  double density;
} physics_ia_coefficients;

struct physics_model_3pcf {
  int nk;
  int nz;
  int nbin;
  int nlos;
  int kernel_nz;
  int source_readers;
  int table_boost;
  int fit_nz;
  int npower_z;
  double zmax;
  double* power_z;
  double omega_m;
  double omega_de;
  double w0;
  double wa;
  double ns;
  double sigma8;
  physics_spline linear;
  physics_spline* nonlinear;
  physics_spline distance;
  physics_spline growth;
  physics_spline dz_dchi;
  physics_spline radius;
  physics_spline neff;
  physics_spline sigma8z;
  physics_spline omega_m_z;
  double* sources;
  double* kernel_z;
  double* kernel_chi;
  double* kernel_growth;
  double* kernel_lens;
  double* kernel_shape;
  double* los_z;
  double* los_chi;
  double* los_dchi;
  double* los_lens;
  double* los_shape;
  physics_bh_coefficients* los_bh;
  physics_power_row* los_power; // [mode,nlos], fixed redshift brackets
};

struct physics_projection_3pcf {
  const physics_model_3pcf* model;
  physics_ia_3pcf ia;
  double* weights; // [8,nlos]: matter baseline and seven residual terms
  physics_ia_coefficients* coefficients;
};

static physics_power_row prepare_power_row(
  const physics_model_3pcf* model,
  double z,
  int mode
);


static void require_physics(
  const int condition,
  const char* message
)
{
  if (!condition) {
    fprintf(stderr, "physics_3pcf: %s\n", message);
    exit(EXIT_FAILURE);
  }
}


// Models are independent owned objects, rather than CosmoLike global
// state. Each allocation is a single flat array, so ordinary calloc/free
// supplies the needed alignment and makes partial construction explicit.
static void* allocate_physics(
  const size_t count,
  const size_t element_size
)
{
  require_physics(count > 0 && count <= SIZE_MAX/element_size,
    "allocation size overflow");
  void* result = calloc(count, element_size);
  require_physics(result != NULL, "allocation failed");
  return result;
}


static void increasing_physics(
  const double* values,
  const int count,
  const int positive
)
{
  require_physics(values != NULL && count >= 2, "missing grid");
  for (int index=0; index<count; index++) {
    require_physics(isfinite(values[index]), "nonfinite grid value");
    require_physics(!positive || values[index] > 0.0,
      "grid must be positive");
    require_physics(index == 0 || values[index] > values[index - 1],
      "grid must be strictly increasing");
  }
}


// A not-a-knot cubic spline matches scipy's InterpolatedUnivariateSpline
// boundary condition: the first two intervals share a cubic polynomial,
// as do the last two. Eliminating the endpoint second derivatives leaves
// a tridiagonal system for interior derivatives. Its solution is prepared
// once; evaluation uses only reads and needs no mutable GSL accelerator.
static physics_spline make_spline(
  const double* x,
  const double* y,
  const int count
)
{
  physics_spline result = {0};
  increasing_physics(x, count, 0);
  result.n = count;
  result.x = allocate_physics((size_t)count, sizeof(double));
  result.y = allocate_physics((size_t)count, sizeof(double));
  result.second = allocate_physics((size_t)count, sizeof(double));
  memcpy(result.x, x, (size_t)count*sizeof(double));
  memcpy(result.y, y, (size_t)count*sizeof(double));
  result.slope_low = (y[1] - y[0])/(x[1] - x[0]);
  result.slope_high = (y[count - 1] - y[count - 2])
    /(x[count - 1] - x[count - 2]);
  for (int index=0; index<count; index++) {
    require_physics(isfinite(y[index]), "nonfinite spline data");
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
  double* diagonal = allocate_physics((size_t)interior, sizeof(double));
  double* upper = allocate_physics((size_t)interior, sizeof(double));
  double* rhs = allocate_physics((size_t)interior, sizeof(double));
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
      diagonal[row] += left*(left + right)/right;
      upper[row] -= left*left/right;
      lower = 0.0;
    }
    if (row == interior - 1) {
      diagonal[row] += right*(left + right)/left;
      lower -= right*right/left;
      upper[row] = 0.0;
    }
    if (row > 0) {
      const double multiplier = lower/diagonal[row - 1];
      diagonal[row] -= multiplier*upper[row - 1];
      rhs[row] -= multiplier*rhs[row - 1];
    }
  }
  for (int row=interior - 1; row>=0; row--) {
    const double next = row + 1 < interior ? result.second[row + 2] : 0;
    result.second[row + 1] = (rhs[row] - upper[row]*next)/diagonal[row];
  }
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


static void destroy_spline(physics_spline* spline)
{
  free(spline->x);
  free(spline->y);
  free(spline->second);
  memset(spline, 0, sizeof(*spline));
}


static physics_spline make_natural_uniform_spline(
  const double* x,
  const double* y,
  const int count
)
{
  physics_spline result = make_spline(x, y, count);
  double* diagonal = allocate_physics((size_t)count, sizeof(double));
  const double inverse_square = pow((count - 1)/(x[count - 1] - x[0]), 2);
  result.second[0] = 0.0;
  result.second[count - 1] = 0.0;
  for (int index=1; index<count - 1; index++) {
    diagonal[index] = 4.0 - (index == 1 ? 0.0 : 1.0/diagonal[index - 1]);
    result.second[index] = 6.0*(y[index + 1] - 2.0*y[index] + y[index - 1])
      *inverse_square - (index == 1 ? 0.0
      : result.second[index - 1]/diagonal[index - 1]);
  }
  for (int index=count - 2; index>=1; index--) {
    result.second[index] = (result.second[index] - result.second[index + 1])
      /diagonal[index];
  }
  free(diagonal);
  return result;
}


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


// Outside supplied Fourier support, retain fastnc's endpoint power laws.
// This is an extrapolation of the physical model, not proof of accuracy.
// The forecast must test the support against its aperture and scale cuts.
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


// CosmoLike's table strategy separates expensive preparation from repeated
// reads: cubic interpolation fills a dense uniform grid once, and workers
// then locate a cell by (x-xmin)/dx and linearly interpolate its two values.
// Input tables may be irregular; only this serial preparation searches
// their intervals. The production reader above never searches a grid.
static physics_spline uniform_storage(
  const int count,
  const double lower,
  const double upper
)
{
  physics_spline result = {0};
  require_physics(count >= 2 && upper > lower, "invalid uniform table");
  result.n = count;
  result.x = allocate_physics((size_t)count, sizeof(double));
  result.y = allocate_physics((size_t)count, sizeof(double));
  result.inverse_spacing = (count - 1)/(upper - lower);
  for (int index=0; index<count; index++) {
    result.x[index] = lower + (upper - lower)*index/(count - 1);
  }
  result.x[count - 1] = upper;
  return result;
}


static int boosted_count(const int original, const int boost)
{
  require_physics(boost >= 1 && original >= 2
    && original - 1 <= (INT_MAX - 1)/boost, "dense table count overflows");
  return (original - 1)*boost + 1;
}


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


static void prepare_uniform_readers(physics_model_3pcf* model)
{
  if (model->source_readers) {
    return;
  }
  const int boost = model->table_boost;
  uniformize_spline(&model->linear, boost);
  uniformize_spline(&model->growth, boost);
  uniformize_spline(&model->radius, boost);
  uniformize_spline(&model->neff, boost);
  uniformize_spline(&model->sigma8z, boost);
  uniformize_spline(&model->omega_m_z, boost);
  for (int redshift=0; redshift<model->npower_z; redshift++) {
    uniformize_spline(model->nonlinear + redshift, boost);
  }
  const int old_count = model->npower_z;
  const int new_count = boosted_count(old_count, boost);
  const int nk = model->nonlinear[0].n;
  physics_spline* dense = allocate_physics((size_t)new_count,
    sizeof(physics_spline));
  double* new_z = allocate_physics((size_t)new_count, sizeof(double));
  for (int redshift=0; redshift<new_count; redshift++) {
    new_z[redshift] = model->zmax*redshift/(new_count - 1);
    dense[redshift] = uniform_storage(nk, model->nonlinear[0].x[0],
      model->nonlinear[0].x[nk - 1]);
  }
  double* column = allocate_physics((size_t)old_count, sizeof(double));
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


// This includes scipy's final Cartwright interval for an even number of
// equally spaced nodes: h*(5*f_n/12+2*f_(n-1)/3-f_(n-2)/12).
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


// Window 0 is a spherical top hat. Windows 1 and 2 are exp(-x^2/2)
// and x*exp(-x^2/2); their squared windows determine sigma_G^2 and
// its logarithmic radius derivative, hence the nonlinear scale and n_eff.
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
  require_physics(count <= 2000 && count >= 3, "invalid variance grid");
  const double logk_max = log(100.0);
  const double spacing = (logk_max - logk_min)/(count - 1);
  const double inverse_two_pi_squared = 0.05066059182116888572;
  for (int index=0; index<count; index++) {
    const double logk = logk_min + spacing*index;
    const double k = exp(logk);
    const double x = k*radius;
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
    const double logdelta_last = log(read_power(&model->linear, logk_max))
      + 3.0*logk_max;
    const double logdelta_previous = log(read_power(&model->linear,
      logk_max - spacing)) + 3.0*(logk_max - spacing);
    const double slope = (logdelta_last - logdelta_previous)/spacing;
    const double lower = 100.0*radius;
    gsl_sf_result upper_gamma;
    const int status = gsl_sf_gamma_inc_e(0.5*slope,
      lower*lower, &upper_gamma);
    require_physics(status == GSL_SUCCESS || status == GSL_EUNDRFLW,
      "Gaussian variance tail failed");
    sum += 0.5*exp(logdelta_last - slope*logk_max)
      *pow(radius, -slope)*upper_gamma.val
      *inverse_two_pi_squared*growth_squared;
  }
  require_physics(isfinite(sum) && sum > 0.0, "invalid variance");
  return sum;
}


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


// BiHalofit coefficients depend on a nonlinear radius, not directly on a
// guessed k_nl. Solve sigma_G(R,z)=1 on the same 1000-node log-k grid as
// fastnc, then evaluate its derivative on 2000 nodes. Coefficient anchor
// spacings are the source's 0.15 below z=.5 and 0.3 above it. Small z
// domains receive at least four anchors so the cubic fit remains defined.
static void prepare_bihalofit(physics_model_3pcf* model)
{
  const double zmax = model->growth.x[model->nz - 1];
  require_physics(zmax/0.15 < INT_MAX - 8,
    "BiHalofit coefficient grid is too large");
  const int capacity = model->source_readers
    ? (int)ceil(zmax/0.15) + 8 : model->fit_nz;
  double* redshift = allocate_physics((size_t)capacity, sizeof(double));
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
  double* radius = allocate_physics((size_t)count, sizeof(double));
  double* spectral = allocate_physics((size_t)count, sizeof(double));
  double* sigma8z = allocate_physics((size_t)count, sizeof(double));
  double* matter = allocate_physics((size_t)count, sizeof(double));
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
    require_physics(at_lower >= 0.0 && at_upper <= 0.0,
      "nonlinear radius is not bracketed by the BiHalofit domain");
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


static void prepare_radial(physics_model_3pcf* model)
{
  const int nz = model->nz;
  const int nlos = model->nlos;
  const int nkernel = model->kernel_nz;
  const double* source_z = model->distance.x;
  const double* source_chi = model->distance.y;
  const double zmin = model->los_z[0];
  const double zmax = source_z[nz - 1];
  model->kernel_z = allocate_physics((size_t)nkernel, sizeof(double));
  model->kernel_chi = allocate_physics((size_t)nkernel, sizeof(double));
  model->kernel_growth = allocate_physics((size_t)nkernel, sizeof(double));
  model->kernel_lens = allocate_physics((size_t)model->nbin*nkernel,
    sizeof(double));
  model->kernel_shape = allocate_physics((size_t)model->nbin*nkernel,
    sizeof(double));
  model->los_lens = allocate_physics((size_t)model->nbin*nlos,
    sizeof(double));
  model->los_shape = allocate_physics((size_t)model->nbin*nlos,
    sizeof(double));
  model->los_chi = allocate_physics((size_t)nlos, sizeof(double));
  model->los_dchi = allocate_physics((size_t)nlos, sizeof(double));
  model->los_bh = allocate_physics((size_t)nlos,
    sizeof(physics_bh_coefficients));
  for (int node=0; node<nkernel; node++) {
    const double redshift = zmin + (zmax - zmin)*node/(nkernel - 1);
    model->kernel_z[node] = redshift;
    model->kernel_chi[node] = read_spline(&model->distance, redshift, 0);
    model->kernel_growth[node] = read_spline(&model->growth, redshift, 0);
  }
  for (int node=0; node<nlos; node++) {
    model->los_chi[node] = read_spline(&model->distance,
      model->los_z[node], 0);
    model->los_bh[node] = bh_coefficients(model, model->los_z[node]);
    require_physics(model->los_chi[node] > 0.0,
      "line-of-sight distance must be positive");
  }
  for (int node=0; node<nlos; node++) {
    const int lower = node > 0 ? node - 1 : node;
    const int upper = node + 1 < nlos ? node + 1 : node;
    model->los_dchi[node] = 0.5*(model->los_chi[upper]
      - model->los_chi[lower]);
  }
  double* integrand = allocate_physics((size_t)nz, sizeof(double));
  double* source_shape = allocate_physics((size_t)nz, sizeof(double));
  const double prefactor = 1.5*pow(100.0/299792.0, 2)*model->omega_m;
  for (int bin=0; bin<model->nbin; bin++) {
    const double* density = model->sources + (size_t)bin*nz;
    double* lens = model->kernel_lens + (size_t)bin*nkernel;
    double* nla_shape = model->kernel_shape + (size_t)bin*nkernel;
    const double redshift_norm = trapezoid(source_z, density, nz);
    require_physics(redshift_norm > 0.0, "empty source distribution");
    for (int node=0; node<nkernel; node++) {
      const double distance = model->kernel_chi[node];
      for (int source=0; source<nz; source++) {
        integrand[source] = source_chi[source] > distance
          ? density[source]*(1.0 - distance/source_chi[source]) : 0.0;
      }
      lens[node] = prefactor*trapezoid(source_z, integrand, nz)
        /redshift_norm;
      nla_shape[node] = linear_read(source_z, density, nz,
        model->kernel_z[node])*read_spline(&model->dz_dchi,
        model->kernel_z[node], 0);
    }
    const double nla_norm = trapezoid(model->kernel_chi,
      nla_shape, nkernel);
    require_physics(nla_norm > 0.0, "empty sampled NLA source kernel");
    for (int node=0; node<nkernel; node++) {
      nla_shape[node] /= nla_norm*model->kernel_chi[node]
        *(1.0 + model->kernel_z[node]);
    }
    for (int source=0; source<nz; source++) {
      integrand[source] = density[source]*model->dz_dchi.y[source];
    }
    const double shape_norm = trapezoid(source_chi, integrand, nz);
    require_physics(shape_norm > 0.0, "empty shape source kernel");
    for (int source=0; source<nz; source++) {
      // Sources are supplied on a background grid containing chi=0.
      // Require n(0)=0 and use zero for that endpoint; LOS nodes are
      // strictly positive. Refining this grid tests the endpoint limit.
      source_shape[source] = source == 0 ? 0.0 : integrand[source]
        /(shape_norm*source_chi[source]*(1.0 + source_z[source]));
    }
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


physics_model_3pcf* physics_create_3pcf(const physics_input_3pcf* input)
{
  require_physics(input != NULL, "missing model input");
  require_physics(input->nk >= 4 && input->nz >= 4 && input->nbin > 0
    && input->nlos >= 2 && input->kernel_nz >= 4,
    "insufficient model grid dimensions");
  increasing_physics(input->k, input->nk, 1);
  increasing_physics(input->z, input->nz, 0);
  increasing_physics(input->chi, input->nz, 0);
  increasing_physics(input->los_z, input->nlos, 1);
  require_physics(input->z[0] == 0.0 && input->chi[0] == 0.0,
    "background tables must start at z=chi=0");
  require_physics(input->los_z[input->nlos - 1] <= input->z[input->nz - 1],
    "LOS nodes extend beyond the supplied physical tables");
  require_physics(input->p_linear0 && input->p_nonlinear && input->growth
    && input->dz_dchi && input->sources, "missing physical input table");
  require_physics(fabs(input->growth[0] - 1.0) <= 1.0e-10,
    "growth must be normalized to one at z=0");
  require_physics(isfinite(input->omega_m) && input->omega_m > 0.0
    && isfinite(input->omega_de) && input->omega_de >= 0.0
    && isfinite(input->w0) && isfinite(input->wa)
    && isfinite(input->ns) && isfinite(input->sigma8)
    && input->sigma8 > 0.0, "invalid cosmological scalars");
  const double density_sum = input->omega_m + input->omega_de;
  require_physics(isfinite(density_sum) && fabs(density_sum - 1.0) <= 1.0e-8,
    "flat geometry requires abs(omega_m + omega_de - 1) <= 1e-8");
  require_physics(input->source_readers == 0 || input->source_readers == 1,
    "source_readers must be zero or one");
  physics_model_3pcf* model = allocate_physics(1, sizeof(*model));
  model->nk = input->nk;
  model->nz = input->nz;
  model->nbin = input->nbin;
  model->nlos = input->nlos;
  model->kernel_nz = input->kernel_nz;
  model->source_readers = input->source_readers;
  model->table_boost = input->table_boost == 0 ? 8 : input->table_boost;
  model->fit_nz = input->fit_nz == 0 ? 33 : input->fit_nz;
  require_physics(model->table_boost >= 1, "table_boost must be positive");
  require_physics(model->fit_nz >= 4, "fit_nz must be at least four");
  model->npower_z = input->nz;
  model->zmax = input->z[input->nz - 1];
  model->power_z = allocate_physics((size_t)input->nz, sizeof(double));
  memcpy(model->power_z, input->z, (size_t)input->nz*sizeof(double));
  model->omega_m = input->omega_m;
  model->omega_de = input->omega_de;
  model->w0 = input->w0;
  model->wa = input->wa;
  model->ns = input->ns;
  model->sigma8 = input->sigma8;
  const int nk = model->nk;
  const int nz = model->nz;
  double* logk = allocate_physics((size_t)nk, sizeof(double));
  double* logpower = allocate_physics((size_t)nk, sizeof(double));
  for (int index=0; index<nk; index++) {
    require_physics(isfinite(input->p_linear0[index])
      && input->p_linear0[index] > 0.0, "linear power must be positive");
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
  model->nonlinear = allocate_physics((size_t)nz, sizeof(physics_spline));
  for (int redshift=0; redshift<nz; redshift++) {
    require_physics(input->growth[redshift] > 0.0
      && input->dz_dchi[redshift] > 0.0, "growth and dz/dchi must be positive");
    for (int index=0; index<nk; index++) {
      const double power = input->p_nonlinear[(size_t)redshift*nk + index];
      require_physics(isfinite(power) && power > 0.0,
        "nonlinear power must be positive");
      logpower[index] = log(power);
    }
    model->nonlinear[redshift] = make_spline(logk, logpower, nk);
  }
  model->distance = make_spline(input->z, input->chi, nz);
  model->growth = make_spline(input->z, input->growth, nz);
  model->dz_dchi = make_spline(input->z, input->dz_dchi, nz);
  model->sources = allocate_physics((size_t)input->nbin*nz, sizeof(double));
  for (size_t index=0; index<(size_t)input->nbin*nz; index++) {
    require_physics(isfinite(input->sources[index])
      && input->sources[index] >= 0.0, "source densities must be nonnegative");
    model->sources[index] = input->sources[index];
  }
  for (int bin=0; bin<input->nbin; bin++) {
    require_physics(model->sources[(size_t)bin*nz] == 0.0,
      "source density must vanish at z=0");
  }
  model->los_z = allocate_physics((size_t)input->nlos, sizeof(double));
  memcpy(model->los_z, input->los_z, (size_t)input->nlos*sizeof(double));
  prepare_bihalofit(model);
  prepare_radial(model);
  prepare_uniform_readers(model);
  // The per-LOS fit coefficients and NLA growth samples must use the
  // selected reader too, so projected means and covariance B queries use
  // identical model values at an identical redshift.
  model->los_power = allocate_physics((size_t)3*model->nlos,
    sizeof(physics_power_row));
  for (int node=0; node<model->nlos; node++) {
    model->los_bh[node] = bh_coefficients(model, model->los_z[node]);
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
  free(model);
}


void physics_default_ia_3pcf(physics_ia_3pcf* parameters)
{
  require_physics(parameters != NULL, "missing alignment parameters");
  memset(parameters, 0, sizeof(*parameters));
  parameters->zpivot = 0.62;
  parameters->c1rho = 0.0134;
  parameters->power_mode = 1;
}


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
  // The original source reader and endpoint power laws retain their
  // explicit comparison path. No binary search occurs in production.
  const double first = log(read_power(lower, logk));
  const double second = log(read_power(upper, logk));
  return exp(first + row->fraction*(second - first));
}


static double read_power_row(const physics_power_row* row, const double k)
{
  return read_power_row_log(row, log(k));
}


double physics_power_3pcf(
  const physics_model_3pcf* model,
  const double k,
  const double z,
  const int mode
)
{
  require_physics(model && isfinite(k) && k > 0.0 && isfinite(z)
    && z >= 0.0 && z <= model->zmax
    && mode >= 0 && mode <= 2, "invalid power query");
  const physics_power_row row = prepare_power_row(model, z, mode);
  return read_power_row(&row, k);
}


// The two wavevectors add to a vector whose length is opposite. Factor
// the difference of the two large squared lengths before subtracting
// the small side. For (epsilon,1,1), this retains mu=-epsilon/2 even
// when epsilon squared is below the precision of a number near one.
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
    // Near antiparallel equal sides F2 tends to zero. Its usual three
    // terms cancel at order unity. Write t=1+mu using another factored
    // side difference, then keep the unequal-side correction with its
    // physical sign. F2 can be negative for unequal antiparallel sides.
    const double difference = large - small;
    const double one_plus = 0.5*((opposite - difference)/small)
      *((opposite + difference)/large);
    return one_plus*(3.0 + 2.0*one_plus)/7.0
      + 0.5*cosine*(difference/small)*(difference/large);
  }
  return 5.0/7.0 + 0.5*(cosine*(small/large)
    + (cosine/small)*large) + 2.0/7.0*cosine*cosine;
}


// Cosine of the angle between a side and the closing vector. This has
// the opposite sign convention to pair_cosine_physics. Either argument
// can be the short side, so choose the factored subtraction accordingly.
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


static physics_triangle_geometry triangle_geometry(const double* sides)
{
  physics_triangle_geometry geometry;
  const double minimum = fmin(sides[0], fmin(sides[1], sides[2]));
  const double maximum = fmax(sides[0], fmax(sides[1], sides[2]));
  // A power-of-two rescaling is exact for ordinary binary numbers. It
  // avoids overflow in side products without rounding away the small
  // difference between two nearly equal long sides.
  int exponent;
  frexp(maximum, &exponent);
  for (int leg=0; leg<3; leg++) {
    geometry.sides[leg] = scalbn(sides[leg], -exponent);
    require_physics(geometry.sides[leg] > 0.0,
      "triangle side ratio underflows double precision");
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
      // Rectangular PUV interpolation includes off-triangle corners.
      // The source's clipped angle defines that continuation. It also
      // handles closure errors within the direct validator's roundoff
      // tolerance. A valid triangle uses its area and both cosines.
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
  }
  return geometry;
}


// The two F2 terms containing a short side can separately grow as 1/s,
// although their weighted sum is finite. Combine their opposite large
// pieces before multiplication. With b,c the other sides,
// F2(s,b,c)=13/28+2*mu_sb^2/7+mu_sb*s/(2*b)+(c^2-b^2)/(4*s^2).
// The last piece therefore multiplies P(b)-P(c), not two large products
// that are subsequently subtracted. This also applies to the effective
// powers in BiHalofit's three-halo term.
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


static void triangle_physics(
  const double first,
  const double second,
  const double third
)
{
  const double largest = fmax(first, fmax(second, third));
  const double sum = first + second + third;
  require_physics(isfinite(sum) && first > 0.0 && second > 0.0
    && third > 0.0 && 2.0*largest <= sum*(1.0 + 32.0*DBL_EPSILON),
    "wavevector lengths do not form a finite triangle");
}


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
  const physics_triangle_geometry geometry = prepared_geometry
    ? *prepared_geometry : triangle_geometry(sides);
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
    return tree_sum_physics(&geometry, linear);
  }
  const double ratio1 = geometry.minimum_ratio;
  const double ratio2 = geometry.flattened_ratio;
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
  double three_halo = tree_sum_physics(&geometry, effective);
  for (int leg=0; leg<3; leg++) {
    const int next = (leg + 1)%3;
    const int opposite = (leg + 2)%3;
    three_halo += 2.0*coefficients->d*scaled[opposite]
      *effective[leg]*effective[next];
  }
  three_halo *= damping;
  return component == 2 ? one_halo
    : (component == 3 ? three_halo : one_halo + three_halo);
}


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
  require_physics(model && component >= 0 && component <= 3
    && isfinite(z) && z >= 0.0 && z <= model->zmax
    && isfinite(fb), "invalid matter query");
  triangle_physics(k1, k2, k3);
  const double sides[3] = {k1, k2, k3};
  const physics_bh_coefficients coefficients = bh_coefficients(model, z);
  return matter_prepared(model, &coefficients, sides, z, component, NULL, NULL)
    *baryon_physics(sides, z, fb, 0);
}


static void check_ia(const physics_ia_3pcf* parameters)
{
  require_physics(parameters != NULL, "missing alignment parameters");
  require_physics(isfinite(parameters->a1) && isfinite(parameters->eta1)
    && isfinite(parameters->a2) && isfinite(parameters->eta2)
    && isfinite(parameters->bta) && isfinite(parameters->ct)
    && isfinite(parameters->zpivot) && parameters->zpivot > -1.0
    && isfinite(parameters->c1rho) && parameters->c1rho > 0.0
    && isfinite(parameters->fb), "nonfinite alignment or baryon parameter");
  require_physics(parameters->power_mode >= 0 && parameters->power_mode <= 2
    && parameters->ct_mode >= 0 && parameters->ct_mode <= 1
    && parameters->suppress_only >= 0 && parameters->suppress_only <= 1,
    "unknown alignment or baryon convention");
}


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
  return coefficients;
}


// A shape tensor has three helicity projections here, labelled 0,1,2.
// For each cyclic ordering put the output wavevector on the positive
// x axis and construct the two input vectors from triangle closure. F[m]
// is the second-order shape vertex for that ordering after removing the
// linear-alignment part. Gravity is collected into a common tree sum
// later, avoiding cancellation between large squeezed cyclic terms.
// Preparing three vertices keeps each leg and orientation explicit.
static void ia_vertices(
  const physics_triangle_geometry* geometry,
  const double* power,
  const physics_ia_coefficients* coefficients,
  double vertices[3][3],
  double* tree
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
      *(0.25*(first_x*first_x - first_y*first_y
      + second_x*second_x - second_y*second_y)
      + 0.5*dot*first_x*second_x);
    const double density = 0.25*coefficients->density
      *(2.0*first_x*first_x - first_y*first_y
      + 2.0*second_x*second_x - second_y*second_y);
    const double velocity = coefficients->velocity*5.0/7.0
      *geometry->sine_squared[cyclic];
    vertices[cyclic][0] = 2.0*root_two_thirds*pair_power
      *(quadratic + density + velocity);
    vertices[cyclic][1] = -pair_power
      *(coefficients->quadratic + coefficients->density)
      *(first_x*first_y + second_x*second_y);
    vertices[cyclic][2] = 2.0*pair_power
      *(0.5*coefficients->quadratic*dot*first_y*second_y
      + 0.25*coefficients->density
      *(first_y*first_y + second_y*second_y));
    tree[cyclic] = 0.0;
  }
}


// These labels describe the fields, not their perturbation orders: 0 is
// scalar density and 2 is the rank-two intrinsic-shape tensor. Thus 002
// contains two densities and one shape, 022 one density and two shapes,
// and 222 three shapes. At tree order each contribution places one
// second-order vertex on a leg and linear fields on the remaining legs.
// The cyclic index tracks that leg placement. A helicity argument instead
// selects a tensor component relative to its wavevector; it does not
// count perturbative powers. Linear shapes occupy helicity zero only,
// which explains the conditional factors in these contractions.
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


// Convert tensor helicities to observable E/B shape components. These
// equations contain the source's factors of 1/2, sqrt(3/2) and 3/16.
// Changing them changes the IA amplitude convention; it is not a speed
// optimization. Cyclic arrays below map output *labels*, not sorted side
// lengths. Sorting only k would silently exchange source-bin kernels.
static void ia_prepared(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const physics_ia_coefficients* coefficients,
  const double* sides,
  const double z,
  const int remove_alignment,
  double* out,
  const physics_power_row* prepared,
  const physics_triangle_geometry* prepared_geometry
)
{
  if (coefficients->quadratic == 0.0 && coefficients->density == 0.0
      && coefficients->velocity == 0.0
      && (remove_alignment || coefficients->linear == 0.0)) {
    // With no quadratic shape vertex, subtracting the linear alignment
    // leaves exactly zero. This common NLA case needs no power reads.
    for (int index=0; index<19; index++) {
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
  const physics_triangle_geometry geometry = prepared_geometry
    ? *prepared_geometry : triangle_geometry(sides);
  ia_vertices(&geometry, power, coefficients, vertices, tree);
  const double linear = coefficients->linear;
  const double root_three_halves = sqrt(3.0/2.0);
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
    out[single_b[cyclic]] = -tensor002(vertices, tree, linear, cyclic, 1);
    out[mixed_eb[cyclic]] = -sqrt(3.0/8.0)
      *tensor022(vertices, tree, linear, cyclic, 0, 1);
    out[mixed_be[cyclic]] = -sqrt(3.0/8.0)
      *tensor022(vertices, tree, linear, cyclic, 1, 0);
    out[double_e_b[cyclic]] = -3.0/8.0
      *tensor222(vertices, linear, cyclic, 0, 0, 1);
  }
  out[6] = 3.0/16.0*(root_three_halves
    *tensor222(vertices, linear, 0, 0, 0, 0)
    - tensor222(vertices, linear, 0, 0, 0, 2)
    - tensor222(vertices, linear, 0, 2, 0, 0)
    - tensor222(vertices, linear, 0, 0, 2, 0));
  if (!remove_alignment) {
    // Linear shapes have only helicity zero. Their contribution to an
    // all-E correlator is C1 to the number of shapes times B_tree.
    // Adding this once is algebraically equal to its three placements.
    // The requested TATT residual never subtracts these large terms.
    const double total_tree = tree_sum_physics(&geometry, power);
    const double c1 = coefficients->c1;
    for (int index=0; index<3; index++) {
      out[index] += c1*total_tree;
      out[index + 3] += c1*c1*total_tree;
    }
    out[6] += c1*c1*c1*total_tree;
  }
}


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
  require_physics(model && out && isfinite(z) && z >= 0.0
    && z <= model->zmax, "invalid alignment query");
  check_ia(parameters);
  triangle_physics(k1, k2, k3);
  const double sides[3] = {k1, k2, k3};
  const double growth = read_spline(&model->growth, z, 0);
  const physics_ia_coefficients coefficients = ia_coefficients(model,
    parameters, z, growth);
  ia_prepared(model, parameters, &coefficients, sides, z,
    remove_alignment, out, NULL, NULL);
}


// A common triangle shape occurs along every Limber integral and in
// scale scans of one configuration. Validate the shape association before
// launching workers: changing one leg alone requires a different geometry.
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
  require_physics(model && sides && z && out && npoint > 0,
    "invalid common-shape IA buffers");
  check_ia(parameters);
  triangle_physics(sides[0], sides[1], sides[2]);
  const physics_triangle_geometry geometry = triangle_geometry(sides);
  for (int point=0; point<npoint; point++) {
    const double* triangle = sides + (size_t)point*3;
    triangle_physics(triangle[0], triangle[1], triangle[2]);
    require_physics(isfinite(z[point]) && z[point] >= 0.0
      && z[point] <= model->zmax, "invalid common-shape IA redshift");
    for (int leg=1; leg<3; leg++) {
      const double ratio = triangle[leg]/triangle[0];
      const double reference = sides[leg]/sides[0];
      require_physics(fabs(ratio - reference)
        <= 64.0*DBL_EPSILON*fmax(fabs(reference), fabs(ratio)),
        "IA same-shape inputs change a triangle side ratio");
    }
  }
  #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
  for (int point=0; point<npoint; point++) {
    const double growth = read_spline(&model->growth, z[point], 0);
    const physics_ia_coefficients coefficients = ia_coefficients(model,
      parameters, z[point], growth);
    ia_prepared(model, parameters, &coefficients, sides + (size_t)point*3,
      z[point], remove_alignment, out + (size_t)point*19, NULL, &geometry);
  }
}


// The nonlinear-alignment baseline uses g+C1*W sampled on the lensing
// kernel grid, then splined in chi. The TATT residual uses the pure g and
// the source-grid W. They represent the same continuum window but have
// different finite-grid errors in fastnc, which this implementation keeps.
// This function returns the *physical* kernel chi*(1+z)*(g+C1*W).
void physics_nla_kernel_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const int bin,
  double* out
)
{
  require_physics(model && out && bin >= 0 && bin < model->nbin,
    "invalid source bin");
  check_ia(parameters);
  const int count = model->kernel_nz;
  double* kernel = allocate_physics((size_t)count, sizeof(double));
  for (int node=0; node<count; node++) {
    const physics_ia_coefficients coefficients = ia_coefficients(model,
      parameters, model->kernel_z[node], model->kernel_growth[node]);
    const size_t index = (size_t)bin*count + node;
    kernel[node] = model->kernel_lens[index]
      + coefficients.c1*model->kernel_shape[index];
  }
  physics_spline spline = make_spline(model->kernel_chi, kernel, count);
  for (int node=0; node<model->nlos; node++) {
    out[node] = read_spline(&spline, model->los_chi[node], 1)
      *model->los_chi[node]*(1.0 + model->los_z[node]);
  }
  destroy_spline(&spline);
  free(kernel);
}


physics_projection_3pcf* physics_projector_create_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const int* bins
)
{
  require_physics(model && bins, "missing projection model or bins");
  check_ia(parameters);
  for (int leg=0; leg<3; leg++) {
    require_physics(bins[leg] >= 0 && bins[leg] < model->nbin,
      "projection source bin is out of range");
  }
  const int count = model->nlos;
  physics_projection_3pcf* projection = allocate_physics(1,
    sizeof(*projection));
  projection->model = model;
  projection->ia = *parameters;
  projection->weights = allocate_physics((size_t)8*count, sizeof(double));
  projection->coefficients = allocate_physics((size_t)count,
    sizeof(physics_ia_coefficients));
  double* kernels = allocate_physics((size_t)3*count, sizeof(double));
  for (int leg=0; leg<3; leg++) {
    physics_nla_kernel_3pcf(model, parameters, bins[leg],
      kernels + (size_t)leg*count);
  }
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
      projection->weights[(size_t)(term + 1)*count + node]
        = measure*residual[term];
    }
  }
  free(kernels);
  return projection;
}


void physics_projector_destroy_3pcf(physics_projection_3pcf* projection)
{
  if (projection != NULL) {
    free(projection->weights);
    free(projection->coefficients);
    free(projection);
  }
}


double physics_projected_3pcf(
  const physics_projection_3pcf* projection,
  const double ell1,
  const double ell2,
  const double ell3,
  const int component
)
{
  require_physics(projection && component >= 0 && component <= 2,
    "invalid projected component");
  triangle_physics(ell1, ell2, ell3);
  const physics_model_3pcf* model = projection->model;
  const int count = model->nlos;
  const double angular_sides[3] = {ell1, ell2, ell3};
  const physics_triangle_geometry geometry = triangle_geometry(angular_sides);
  double baseline = 0.0;
  double residual = 0.0;
  for (int node=0; node<count; node++) {
    const double distance = model->los_chi[node];
    const double redshift = model->los_z[node];
    const double sides[3] = {ell1/distance, ell2/distance, ell3/distance};
    if (component != 2) {
      baseline += projection->weights[node]*matter_prepared(model,
        model->los_bh + node, sides, redshift, 0, model->los_power + node,
        &geometry)
        *baryon_physics(sides, redshift, projection->ia.fb,
        projection->ia.suppress_only);
    }
    if (component != 1) {
      double intrinsic[19];
      ia_prepared(model, &projection->ia, projection->coefficients + node,
        sides, redshift, 1, intrinsic, model->los_power
        + (size_t)projection->ia.power_mode*count + node, &geometry);
      for (int term=0; term<7; term++) {
        residual += projection->weights[(size_t)(term + 1)*count + node]
          *intrinsic[term];
      }
    }
  }
  return baseline + residual;
}


double physics_projected_puv_extension_3pcf(
  const physics_projection_3pcf* projection,
  const double ell1,
  const double ell2,
  const double ell3
)
{
  require_physics(projection && isfinite(ell1) && isfinite(ell2)
    && isfinite(ell3) && ell1 > 0.0 && ell2 > 0.0 && ell3 >= 0.0,
    "invalid source PUV interpolation corner");
  if (ell3 == 0.0) {
    return 0.0;
  }
  const physics_model_3pcf* model = projection->model;
  const int count = model->nlos;
  const double angular_sides[3] = {ell1, ell2, ell3};
  const physics_triangle_geometry geometry = triangle_geometry(angular_sides);
  double sum = 0.0;
  for (int node=0; node<count; node++) {
    const double distance = model->los_chi[node];
    const double sides[3] = {ell1/distance, ell2/distance, ell3/distance};
    double intrinsic[19];
    ia_prepared(model, &projection->ia, projection->coefficients + node,
      sides, model->los_z[node], 1, intrinsic, model->los_power
      + (size_t)projection->ia.power_mode*count + node, &geometry);
    for (int term=0; term<7; term++) {
      // The pinned outer IA class applies nan_to_num to each component.
      // Restrict that historical extension to the interpolation fill;
      // a failed physical direct evaluation must still be visible.
      if (isfinite(intrinsic[term])) {
        sum += projection->weights[(size_t)(term + 1)*count + node]
          *intrinsic[term];
      }
    }
  }
  return sum;
}


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
    && first->suppress_only == second->suppress_only;
}


// The expensive three-dimensional physics depends on triangle and z,
// whereas tomography enters through radial weights. Evaluate the former
// once, then contract it with every ordered source triplet. No array with
// dimensions [triangle,z,triplet,IA term] is retained: a caller supplies
// only one output vector per worker, and each output keeps its z order.
void physics_projected_batch_3pcf(
  const int nprojection,
  physics_projection_3pcf* const* projections,
  const double ell1,
  const double ell2,
  const double ell3,
  const int component,
  const int puv_extension,
  double* out
)
{
  require_physics(nprojection > 0 && projections && out
    && component >= 0 && component <= 2,
    "invalid batch projection input");
  const physics_projection_3pcf* first = projections[0];
  require_physics(first != NULL, "missing batch projection");
  if (!puv_extension) {
    triangle_physics(ell1, ell2, ell3);
  } else {
    require_physics(component == 2 && isfinite(ell1) && isfinite(ell2)
      && isfinite(ell3) && ell1 > 0.0 && ell2 > 0.0 && ell3 >= 0.0,
      "invalid batch PUV corner");
  }
  for (int triplet=0; triplet<nprojection; triplet++) {
    require_physics(projections[triplet]
      && projections[triplet]->model == first->model
      && same_ia_parameters(&projections[triplet]->ia, &first->ia),
      "batch projections must share one model and the same IA parameters");
    out[triplet] = 0.0;
  }
  if (puv_extension && ell3 == 0.0) {
    return;
  }
  const physics_model_3pcf* model = first->model;
  const int count = model->nlos;
  const double angular_sides[3] = {ell1, ell2, ell3};
  const physics_triangle_geometry geometry = triangle_geometry(angular_sides);
  for (int node=0; node<count; node++) {
    const double distance = model->los_chi[node];
    const double redshift = model->los_z[node];
    const double sides[3] = {ell1/distance, ell2/distance, ell3/distance};
    double baseline = 0.0;
    double intrinsic[19] = {0.0};
    if (component != 2) {
      baseline = matter_prepared(model, model->los_bh + node,
        sides, redshift, 0, model->los_power + node, &geometry)
        *baryon_physics(sides, redshift,
        first->ia.fb, first->ia.suppress_only);
    }
    if (component != 1) {
      ia_prepared(model, &first->ia, first->coefficients + node,
        sides, redshift, 1, intrinsic, model->los_power
        + (size_t)first->ia.power_mode*count + node, &geometry);
      if (puv_extension) {
        for (int term=0; term<7; term++) {
          if (!isfinite(intrinsic[term])) {
            intrinsic[term] = 0.0;
          }
        }
      }
    }
    int triplet = 0;
    for (; triplet + 1<nprojection; triplet += 2) {
      const double* restrict first_weights = projections[triplet]->weights;
      const double* restrict second_weights
        = projections[triplet + 1]->weights;
      // Adjacent lanes own different ordered source-triplet sums. The
      // output storage does not overlap either immutable weight array;
      // restrict states this promise for reads through the local pointers.
      physics_v2d sum = simde_mm_loadu_pd(out + triplet);
      if (component != 2) {
        const physics_v2d weight = simde_mm_set_pd(second_weights[node],
          first_weights[node]);
        const physics_v2d value = simde_mm_set1_pd(baseline);
        sum = simde_mm_add_pd(sum, simde_mm_mul_pd(weight, value));
      }
      if (component != 1) {
        for (int term=0; term<7; term++) {
          const size_t index = (size_t)(term + 1)*count + node;
          // Gather one radial weight per triplet; both share this vertex.
          const physics_v2d weight = simde_mm_set_pd(second_weights[index],
            first_weights[index]);
          const physics_v2d value = simde_mm_set1_pd(intrinsic[term]);
          // Preserve the scalar node/term order independently in each lane.
          sum = simde_mm_add_pd(sum, simde_mm_mul_pd(weight, value));
        }
      }
      simde_mm_storeu_pd(out + triplet, sum);
    }
    if (triplet < nprojection) {
      const double* restrict weights = projections[triplet]->weights;
      if (component != 2) {
        out[triplet] += weights[node]*baseline;
      }
      if (component != 1) {
        for (int term=0; term<7; term++) {
          out[triplet] += weights[(size_t)(term + 1)*count + node]
            *intrinsic[term];
        }
      }
    }
  }
}


// Gauss-Legendre nodes avoid both endpoints, including the zero-length
// squeezed triangle at ell1=ell2 and mu=1. Newton iteration solves each
// Legendre root; mirror roots share weights. No scientific table is read
// or written after preparation, and this helper runs before worker launch.
static void gauss_legendre_physics(
  const int count,
  double* nodes,
  double* weights
)
{
  for (int root=0; root<(count + 1)/2; root++) {
    double coordinate = cos(3.14159265358979323846
      *(root + 0.75)/(count + 0.5));
    double derivative = 0.0;
    for (int iteration=0; iteration<100; iteration++) {
      double previous = 1.0;
      double value = coordinate;
      for (int order=2; order<=count; order++) {
        const double next = ((2.0*order - 1.0)*coordinate*value
          - (order - 1.0)*previous)/order;
        previous = value;
        value = next;
      }
      derivative = count*(coordinate*value - previous)
        /(coordinate*coordinate - 1.0);
      const double step = value/derivative;
      coordinate -= step;
      if (fabs(step) <= 4.0*DBL_EPSILON) {
        break;
      }
      require_physics(iteration < 99, "Legendre root did not converge");
    }
    nodes[root] = -coordinate;
    nodes[count - 1 - root] = coordinate;
    const double weight = 2.0/((1.0 - coordinate*coordinate)
      *derivative*derivative);
    weights[root] = weight;
    weights[count - 1 - root] = weight;
  }
}


void physics_multipoles_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const int* bins,
  const int nell,
  const double* ell,
  const int lmax,
  const int nmu,
  const int component,
  double* out
)
{
  require_physics(ell && out && nell > 0 && lmax >= 0 && lmax < INT_MAX
    && nmu >= 2 && nmu > lmax && component >= 0 && component <= 2,
    "invalid multipole grid or component");
  require_physics((size_t)nell <= SIZE_MAX/(size_t)nell
    && (size_t)(lmax + 1) <= SIZE_MAX/((size_t)nell*nell)
    && (size_t)(lmax + 1)*nell*nell <= SIZE_MAX/sizeof(double),
    "multipole output size overflows");
  for (int index=0; index<nell; index++) {
    require_physics(isfinite(ell[index]) && ell[index] > 0.0,
      "multipoles require positive ell");
  }
  physics_projection_3pcf* projection = physics_projector_create_3pcf(
    model, parameters, bins);
  double* nodes = allocate_physics((size_t)nmu, sizeof(double));
  double* weights = allocate_physics((size_t)nmu, sizeof(double));
  double* legendre = allocate_physics((size_t)(lmax + 1)*nmu,
    sizeof(double));
  gauss_legendre_physics(nmu, nodes, weights);
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

  // Independent ell pairs own disjoint output cells. Each pair evaluates
  // the costly projected bispectrum only nmu times and shares it across
  // every L. SIMD lanes accumulate neighboring orders with the same node
  // sequence: sum_L += LegendreWeight[L,node]*B[node]. No reduction tree
  // or scheduling choice changes that order.
  // A padded slice belongs to each possible worker. Allocate before
  // parallel work, including the direct scientific reference path.
  const size_t worker_stride = (((size_t)nmu + 7)/8 + 1)*8;
  const size_t workers = (size_t)omp_get_max_threads();
  require_physics(workers > 0 && worker_stride <= SIZE_MAX/workers,
    "multipole worker storage overflows");
  double* scratch = allocate_physics(worker_stride*workers, sizeof(double));
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
        }
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
            sum += legendre[(size_t)order*nmu + node]*values[node];
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
  require_physics(model && nbin && nlos && z && chi && dchi && lens && shape,
    "missing radial output pointer");
  *nbin = model->nbin;
  *nlos = model->nlos;
  *z = model->los_z;
  *chi = model->los_chi;
  *dchi = model->los_dchi;
  *lens = model->los_lens;
  *shape = model->los_shape;
}


double physics_omega_m_3pcf(const physics_model_3pcf* model)
{
  require_physics(model != NULL, "missing model");
  return model->omega_m;
}


double physics_omega_m_z_3pcf(
  const physics_model_3pcf* model,
  const double z
)
{
  require_physics(model && isfinite(z) && z >= 0.0,
    "invalid background matter-fraction query");
  return omega_m_physics(model, z);
}


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
        const double first = row->lower->y[index]
          + fraction*(row->lower->y[index + 1] - row->lower->y[index]);
        const double second = row->upper->y[index]
          + fraction*(row->upper->y[index + 1] - row->upper->y[index]);
        values[lane] = exp(first + row->fraction*(second - first));
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


void physics_nla_power_batch_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const int bin1,
  const int bin2,
  const int nell,
  const double* ell,
  double* out
)
{
  check_ia(parameters);
  require_physics(model && ell && out && nell > 0,
    "invalid angular power buffers");
  require_physics(parameters->a2 == 0.0 && parameters->bta == 0.0
    && parameters->ct == 0.0 && parameters->fb == 0.0,
    "NLA power does not implement TATT loops or a power baryon transfer");
  const int count = model->nlos;
  double* first = allocate_physics((size_t)count, sizeof(double));
  double* second = allocate_physics((size_t)count, sizeof(double));
  double* radial_weight = allocate_physics((size_t)count, sizeof(double));
  double* logchi = allocate_physics((size_t)count, sizeof(double));
  double* logell = allocate_physics((size_t)nell, sizeof(double));
  physics_nla_kernel_3pcf(model, parameters, bin1, first);
  physics_nla_kernel_3pcf(model, parameters, bin2, second);
  for (int node=0; node<count; node++) {
    const double distance = model->los_chi[node];
    radial_weight[node] = model->los_dchi[node]*first[node]*second[node]
      /(distance*distance);
    logchi[node] = log(distance);
  }
  for (int index=0; index<nell; index++) {
    require_physics(isfinite(ell[index]) && ell[index] > 0.0,
      "angular power requires positive finite ell");
    logell[index] = log(ell[index]);
  }
  const int capacity = nell < 256 ? nell : 256;
  const size_t cells = (size_t)capacity*count;
  int* indices = allocate_physics(cells, sizeof(int));
  double* fractions = allocate_physics(cells, sizeof(double));
  for (size_t start=0; start<(size_t)nell; start += (size_t)capacity) {
    const size_t remaining = (size_t)nell - start;
    const int length = (int)(remaining < (size_t)capacity
      ? remaining : (size_t)capacity);
    // Prepare log(k)=log(ell)-log(chi), then integer cell and fraction.
    // The bounded tile stores only geometry. Its entries are reused by
    // both nonlinear-redshift rows and remain read-only in the workers.
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
        const physics_v2d power = nla_power_pair(row, indices + cell,
          fractions + cell);
        // The same radial weight multiplies both ell values.
        const physics_v2d weight = simde_mm_set1_pd(radial_weight[node]);
        sum = simde_mm_add_pd(sum, simde_mm_mul_pd(weight, power));
      }
      // This pair owns the two adjacent output cells exclusively.
      simde_mm_storeu_pd(out + start + index, sum);
    }
    if (length%2 != 0) {
      const int index = length - 1;
      double sum = 0.0;
      for (int node=0; node<count; node++) {
        sum += radial_weight[node]*read_power_row_log(
          model->los_power + (size_t)2*count + node,
          logell[start + index] - logchi[node]);
      }
      out[start + index] = sum;
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


static double aperture_filter_physics(
  const double ell,
  const double radius
)
{
  const double scaled = ell*radius;
  const double square = scaled*scaled;
  // Underflow represents the Gaussian tail. Handling a very large
  // finite argument before square*exp avoids the indeterminate inf*0.
  return square > 1500.0 ? 0.0 : 0.5*square*exp(-0.5*square);
}


// A circular aperture multiplies each Fourier field by U(ell R). The
// three-point delta function removes the third Fourier integral, leaving
// d^2ell1 d^2ell2/(2pi)^4. Statistical isotropy integrates the common
// orientation into 2pi, so the measure below is
// ell1*dell1 * ell2*dell2 * dphi/(2pi)^3. The closing leg is |ell1+ell2|.
// This is the full-filter statistic at one aperture center. A finite SAS
// triangle selection is a different estimator and has different weights.
//
// Only a bounded tile of projected bispectra is stored. Each triangle is
// evaluated once and reused for every filter triplet. Workers first own
// triangle values, then own pairs of filter outputs; no workers share an
// accumulator. Every output visits triangles in the same fixed order at
// every thread count. All saved physical model arrays remain read-only.
void physics_aperture_integrate_3pcf(
  const void* context,
  physics_projection_reader_3pcf reader,
  const int nell,
  const double* ell,
  const double* dell,
  const int nphi,
  const double* phi,
  const double* dphi,
  const int nfilter,
  const double* filters,
  double* out
)
{
  require_physics(context && reader && ell && dell && phi
    && dphi && filters && out && nell > 0 && nphi > 0 && nfilter > 0,
    "invalid full-aperture buffers or dimensions");
  require_physics((size_t)nell <= SIZE_MAX/(size_t)nell
    && (size_t)nphi <= SIZE_MAX/((size_t)nell*nell),
    "aperture triangle count overflows");
  const size_t ntriangle = (size_t)nell*nell*nphi;
  for (int node=0; node<nell; node++) {
    require_physics(isfinite(ell[node]) && ell[node] > 0.0
      && isfinite(dell[node]) && dell[node] > 0.0
      && (node == 0 || ell[node] > ell[node - 1]),
      "invalid full-aperture ell quadrature");
  }
  double angle_norm = 0.0;
  for (int node=0; node<nphi; node++) {
    require_physics(isfinite(phi[node]) && phi[node] >= 0.0
      && phi[node] <= 6.28318530717958647693
      && isfinite(dphi[node]) && dphi[node] > 0.0
      && (node == 0 || phi[node] > phi[node - 1]),
      "invalid full-aperture angular quadrature");
    angle_norm += dphi[node];
  }
  require_physics(fabs(angle_norm - 6.28318530717958647693) < 1.0e-10,
    "full-aperture angular weights must sum to 2pi");
  for (size_t index=0; index<(size_t)nfilter*3; index++) {
    require_physics(isfinite(filters[index]) && filters[index] > 0.0,
      "full-aperture radii must be positive");
  }
  const size_t capacity = 4096;
  double* values = allocate_physics(capacity, sizeof(double));
  double* closure = allocate_physics(capacity, sizeof(double));
  double* filter_table = allocate_physics((size_t)nfilter*nell*2,
    sizeof(double));
  for (int filter=0; filter<nfilter; filter++) {
    out[filter] = 0.0;
    for (int leg=0; leg<2; leg++) {
      for (int node=0; node<nell; node++) {
        filter_table[((size_t)filter*2 + leg)*nell + node]
          = aperture_filter_physics(ell[node], filters[(size_t)filter*3 + leg]);
      }
    }
  }
  const double measure_constant = 0.00403144180414993615;
  for (size_t start=0; start<ntriangle; start += capacity) {
    const size_t remaining = ntriangle - start;
    const int count = (int)(remaining < capacity ? remaining : capacity);
    #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
    for (int offset=0; offset<count; offset++) {
      const size_t triangle = start + (size_t)offset;
      const int angle = triangle%(size_t)nphi;
      const int second = (triangle/(size_t)nphi)%(size_t)nell;
      const int first = triangle/((size_t)nphi*nell);
      double third = hypot(ell[first] + ell[second]*cos(phi[angle]),
        ell[second]*sin(phi[angle]));
      if (first == second && fabs(phi[angle] - 3.14159265358979323846)
          <= 4.0*DBL_EPSILON*3.14159265358979323846) {
        third = 0.0;
      }
      closure[offset] = third;
      // Compensation U(0)=0 removes this isolated degenerate triangle.
      // Do not call a three-positive-side fit with a zero closing side.
      values[offset] = third == 0.0 ? 0.0
        : reader(context, ell[first], ell[second], third)
        *ell[first]*dell[first]*ell[second]*dell[second]*dphi[angle]
        *measure_constant;
    }
    const int npair = nfilter/2 + nfilter%2;
    #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
    for (int pair_index=0; pair_index<npair; pair_index++) {
      const int filter = pair_index*2;
      const double* restrict table = filter_table;
      double* restrict output = out + filter;
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
          sum += values[offset]*weight;
        }
        output[0] = sum;
      }
    }
  }
  free(values);
  free(closure);
  free(filter_table);
}


static double direct_aperture_reader(
  const void* context,
  const double ell1,
  const double ell2,
  const double ell3
)
{
  return physics_projected_3pcf(context, ell1, ell2, ell3, 0);
}


void physics_aperture_full_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const int* bins,
  const int nell,
  const double* ell,
  const double* dell,
  const int nphi,
  const double* phi,
  const double* dphi,
  const int nfilter,
  const double* filters,
  double* out
)
{
  physics_projection_3pcf* projection = physics_projector_create_3pcf(
    model, parameters, bins);
  physics_aperture_integrate_3pcf(projection, direct_aperture_reader,
    nell, ell, dell, nphi, phi, dphi, nfilter, filters, out);
  physics_projector_destroy_3pcf(projection);
}
