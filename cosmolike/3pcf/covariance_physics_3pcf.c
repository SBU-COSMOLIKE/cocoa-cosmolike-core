#include <math.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_expint.h>
#include "simde/x86/sse2.h"
#include "covariance_physics_3pcf.h"

// Physical covariance of full aperture products and midpoint-centered xi.
// The cumulant partitions follow Linke et al. (2023), arXiv:2212.04485,
// and Wielders et al. (2025), arXiv:2509.20443. The xi spin kernels below
// insert the actual separation-bin weights into the same field estimator.
// This implementation is written from those equations and Fourier rules;
// it does not translate the GPL threepoint implementation.
//
// The field model is pure gravitational lensing with independent Gaussian
// E/B shape noise. The connected four-, five-, and six-field cumulants use
// a Poisson one-halo model. The extra long-mode term is the halo-abundance
// response, not a complete growth, dilation or tidal response. Finite
// center windows retain full filters inside each aperture. A galaxy mask
// inside a filter or finite triangle selection defines another estimator.
//
// No process-global state or lazy initialization occurs. The call owns
// temporary physical tables, prepares them before parallel readers, and
// releases them before returning. Each parallel task owns complete rows
// or upper-triangle cells. Quadrature sums retain their input order.

#define COV_PI 3.141592653589793238462643383279502884
#define COV_I2PI 0.159154943091895335768883763372514362

typedef simde__m128d v2d;

static inline void covariance_require_3pcf(int condition, const char* text)
{
  if (!condition) {
    fprintf(stderr, "physical 3pcf covariance: %s\n", text);
    exit(EXIT_FAILURE);
  }
}

static void* covariance_allocate_3pcf(size_t count, size_t size)
{
  covariance_require_3pcf(size > 0 && count <= (size_t)-1/size,
    "workspace dimensions overflow the address range");
  void* memory = calloc(count, size);
  covariance_require_3pcf(memory != NULL, "cannot allocate covariance tables");
  return memory;
}

// This ordered dot product uses SIMD for two adjacent nodes. Lanes never
// replace the scalar accumulation order: weighted nodes are added first
// from the low lane, then from the high lane. Scalar meaning is sum w*a*b.
static inline double covariance_dot_3pcf(
  int count, const double* weights, const double* first, const double* second
)
{
  double sum = 0.0;
  int node = 0;
  for (; node + 1 < count; node += 2) {
    // Load two valid adjacent quadrature weights; alignment is unrestricted.
    const v2d w = simde_mm_loadu_pd(weights + node);
    // Load the corresponding two values of the first physical function.
    const v2d a = simde_mm_loadu_pd(first + node);
    // Load the second function at those same nodes.
    const v2d b = simde_mm_loadu_pd(second + node);
    // Form the two pointwise products independently.
    const v2d ab = simde_mm_mul_pd(a, b);
    // Apply each node's integration weight.
    const v2d weighted = simde_mm_mul_pd(w, ab);
    double value[2];
    // Copy both lanes into a valid two-element local buffer.
    simde_mm_storeu_pd(value, weighted);
    sum += value[0];
    sum += value[1];
  }
  if (node < count) {
    sum += weights[node]*first[node]*second[node];
  }
  return sum;
}

static inline double covariance_filter_3pcf(double ell, double radius)
{
  if (radius == 0.0) {
    return 1.0;
  }
  const double x = ell*radius;
  const double square = x*x;
  return 0.5*square*exp(-0.5*square);
}

// Spherical top-hat W(kR), and its derivative with respect to ln R.
// The small-x series avoids subtracting two almost equal sine terms.
static inline void covariance_tophat_3pcf(
  double x, double* window, double* derivative
)
{
  const double square = x*x;
  if (fabs(x) < 0.01) {
    *window = 1.0 - square/10.0 + square*square/280.0
      - square*square*square/15120.0;
    *derivative = -square/5.0 + square*square/70.0
      - square*square*square/2520.0;
  } else {
    *window = 3.0*(sin(x) - x*cos(x))/(x*square);
    *derivative = 3.0*sin(x)/x - 3.0*(*window);
  }
}

// Fourier transform of an NFW density truncated at comoving r200m.
// Normalization is integral rho dV=M, so u(0)=1. The Si/Ci expression is
// the integral of x/(1+x)^2 sinc(k*r_s*x), divided by ln(1+c)-c/(1+c).
// At small k*r200 the second radial moment avoids a Ci difference at zero.
static inline double covariance_nfw_3pcf(
  double k, double r200, double concentration
)
{
  const double c = concentration;
  const double t = 1.0 + c;
  const double normalization = log1p(c) - c/t;
  const double x = k*r200/c;
  if (k*r200 < 1.0e-4) {
    const double moment = 0.5*t*t - 3.0*t + 3.0*log(t) + 1.0/t + 1.5;
    return 1.0 - x*x*moment/(6.0*normalization);
  }
  const double y = x*t;
  return (sin(x)*(gsl_sf_Si(y) - gsl_sf_Si(x))
    + cos(x)*(gsl_sf_Ci(y) - gsl_sf_Ci(x)) - sin(x*c)/y)/normalization;
}

// Halo preparation from P_linear(k,z), with no assumed scale-independent
// growth inside this routine. sigma^2 integrates k^2 dk P W^2/(2*pi^2).
// Since R is proportional to M^(1/3), d sigma^2/d ln M has the factor 2/3.
// The Sheth-Tormen multiplicity converts this derivative to dn/dM. The
// separate bias is the response of halo abundance to a long density mode.
// Duffy et al. (2008), Table 1 mean-density/full-sample z=0..2 fit, gives
// c=10.14 (M/2e12 Msun/h)^-0.081 (1+z)^-1.01 for M200m. These coefficients
// are explicit inputs because extrapolation and alternative fits are
// physical assumptions. r200 is comoving and uses today's mean density.
void covariance_halos_3pcf(
  const physics_model_3pcf* model,
  const covariance_input_3pcf* input,
  int nz,
  const double* z,
  int nk_profile,
  const double* k_profile,
  double* sigma2,
  double* dndm,
  double* bias,
  double* concentration,
  double* profile
)
{
  const double rho = 2.77536627e11*physics_omega_m_3pcf(model);
  const int nm = input->nmass;
  double* power = covariance_allocate_3pcf((size_t)nz*input->nk,
    sizeof(double));

  // One worker owns a redshift's linear-power row, shared by every mass.
  #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
  for (int iz=0; iz<nz; iz++) {
    for (int ik=0; ik<input->nk; ik++) {
      power[(size_t)iz*input->nk + ik] =
        physics_power_3pcf(model, input->k[ik], z[iz], 0);
    }
  }

  // Each (redshift,mass) pair owns its variance, abundance and profile row.
  #pragma omp parallel for collapse(2) if(!omp_in_parallel()) schedule(static)
  for (int iz=0; iz<nz; iz++) {
    for (int im=0; im<nm; im++) {
      const size_t index = (size_t)iz*nm + im;
      const double mass = input->mass[im];
      const double radius = cbrt(3.0*mass/(4.0*COV_PI*rho));
      double variance = 0.0;
      double slope = 0.0;
      for (int ik=0; ik<input->nk; ik++) {
        double window;
        double derivative;
        covariance_tophat_3pcf(input->k[ik]*radius, &window, &derivative);
        const double measure = input->dk[ik]*input->k[ik]*input->k[ik]
          *power[(size_t)iz*input->nk + ik]/(2.0*COV_PI*COV_PI);
        variance += measure*window*window;
        slope += measure*window*derivative*(2.0/3.0);
      }
      covariance_require_3pcf(isfinite(variance) && variance > 0.0
        && isfinite(slope) && slope < 0.0,
        "halo variance/slope is invalid; refine the k interval and rule");
      double collapse = input->delta_c;
      if (input->collapse_model == 1) {
        collapse *= 1.0 + 0.0123*log10(
          physics_omega_m_z_3pcf(model, z[iz]));
      }
      const double peak = collapse*collapse/variance;
      const double apeak = input->st_a*peak;
      const double multiplicity = input->st_norm*sqrt(2.0*apeak/COV_PI)
        *(1.0 + pow(apeak, -input->st_p))*exp(-0.5*apeak);
      const double c = input->concentration_a
        *pow(mass/input->concentration_pivot, input->concentration_b)
        *pow(1.0 + z[iz], input->concentration_c);
      sigma2[index] = variance;
      dndm[index] = -rho/(mass*mass)*slope/(2.0*variance)*multiplicity;
      bias[index] = 1.0 + (apeak - 1.0 + 2.0*input->st_p
        /(1.0 + pow(apeak, input->st_p)))/collapse;
      concentration[index] = c;
      const double r200 = radius/cbrt(200.0);
      for (int ik=0; ik<nk_profile; ik++) {
        profile[index*nk_profile + ik] =
          covariance_nfw_3pcf(k_profile[ik], r200, c);
      }
    }
  }
  free(power);
}

typedef struct {
  const physics_model_3pcf* model;
  const covariance_input_3pcf* in;
  int ns;
  int nz;
  int nt;
  int has_cubic;
  int nquadratic;
  int* quadratic_index;
  int* quadratic_representative;
  const double* z;
  const double* chi;
  const double* dchi;
  double* lens;
  int* triples;
  int* triple_index;
  double* filters;
  double* closure_filters;
  double* power;
  double* bispectrum;
  double* bspec;
  double* breal;
  double* pair;
  double* pair_zero;
  double* radial_measure;
  double* fourier_measure;
  double* window_fourier;
  double* geometry_window;
  double* quadratic;
  double* quadratic_mixed;
  double* pair_window;
  double* j0_radial;
  double* j0_fourier;
  double* j0_window;
  double* j2_profile;
  double* quadratic_zero;
  double* radial_volume;
  double* sigma2;
  double* abundance;
  double* bias;
  double* concentration;
  double* responses;
  double* response_zero;
  double* response_bias;
} covariance_work_3pcf;

// Circle/square center weights are normalized by area squared. Fourier G
// is |w_tilde|^2/A^2. The large-area approximation is applied by callers;
// the long-mode variance always retains the declared finite survey shape.
static inline double covariance_window_3pcf(
  const covariance_input_3pcf* in, double value, int fourier
)
{
  if (in->center_window == 1) {
    const double radius = sqrt(in->area/COV_PI);
    if (fourier) {
      const double x = value*radius;
      const double amplitude = x == 0.0 ? 1.0
        : 2.0*gsl_sf_bessel_J1(x)/x;
      return amplitude*amplitude;
    }
    const double x = value/(2.0*radius);
    if (x >= 1.0) {
      return 0.0;
    }
    return (2.0/COV_PI)*(acos(x) - x*sqrt(1.0 - x*x))/in->area;
  }
  const double side = sqrt(in->area);
  double sum = 0.0;
  for (int angle=0; angle<in->nphi; angle++) {
    const double x = value*cos(in->phi[angle]);
    const double y = value*sin(in->phi[angle]);
    double factor;
    if (fourier) {
      const double a = 0.5*side*x;
      const double b = 0.5*side*y;
      const double sx = a == 0.0 ? 1.0 : sin(a)/a;
      const double sy = b == 0.0 ? 1.0 : sin(b)/b;
      factor = sx*sx*sy*sy;
    } else {
      factor = fmax(0.0, 1.0 - fabs(x)/side)
        *fmax(0.0, 1.0 - fabs(y)/side)/in->area;
    }
    sum += in->dphi[angle]*COV_I2PI*factor;
  }
  return sum;
}

static inline size_t covariance_geometry_3pcf(
  const covariance_input_3pcf* in, int q, int p, int angle
)
{
  return ((size_t)q*in->nell + p)*in->nphi + angle;
}

// sin(double(pi)) is not exactly zero. Recognize the exactly specified
// antiparallel equal-length case before constructing its closure vector.
// No neighborhood of nearly degenerate physical triangles is removed.
static inline double covariance_closure_3pcf(double q, double p, double phi)
{
  if (q == p && phi == COV_PI) {
    return 0.0;
  }
  return hypot(q + p*cos(phi), p*sin(phi));
}

static inline size_t covariance_bspec_3pcf(
  const covariance_work_3pcf* work, int a, int b, int source, int ell
)
{
  const covariance_input_3pcf* in = work->in;
  return (((size_t)a*in->nfield + b)*work->ns + source)*in->nell + ell;
}

static inline size_t covariance_breal_3pcf(
  const covariance_input_3pcf* in, int a, int b, int c, int radius
)
{
  return (((size_t)a*in->nfield + b)*in->nfield + c)*in->nr + radius;
}

static inline double covariance_power_value_3pcf(
  const covariance_work_3pcf* work, int a, int b, int ell, int mode
)
{
  const size_t pair = (size_t)a*work->ns + b;
  const double noise = work->in->noise[pair];
  return mode == 0 ? work->power[pair*work->in->nell + ell] + noise : noise;
}

// An xi estimator averages shear products at x-theta/2 and x+theta/2.
// Angular integration gives J0(|p-q|theta/2) for xi+ and J4 for xi-.
// A pure E Fourier shear has phase exp(2i phi_k); a B shear adds i.
// Their real products give the cosine/sine factors below. At q=-p the
// mixed E/B kernels vanish and these reduce to the usual binned J0/J4.
static inline void covariance_quadratic_3pcf(
  const covariance_input_3pcf* in, int row, double p, double q,
  double angle, double* ee, double* eb
)
{
  const int kind = in->row_kind[row];
  const int* fields = in->row_fields + (size_t)row*3;
  if (kind == 0) {
    *ee = covariance_filter_3pcf(p, in->field_radius[fields[0]])
      *covariance_filter_3pcf(q, in->field_radius[fields[1]]);
    *eb = 0.0;
    return;
  }
  const double dx = p - q*cos(angle);
  const double dy = -q*sin(angle);
  const double separation_wave = 0.5*hypot(dx, dy);
  double bessel = 0.0;
  const int order = kind == 2 ? 0 : 4;
  for (int node=0; node<in->ntheta; node++) {
    const size_t index = (size_t)row*in->ntheta + node;
    bessel += in->theta_weight[index]
      *gsl_sf_bessel_Jn(order, separation_wave*in->theta[index]);
  }
  if (kind == 2) {
    *ee = cos(2.0*angle)*bessel;
    *eb = -sin(2.0*angle)*bessel;
  } else {
    const double phase = 2.0*angle - 4.0*atan2(dy, dx);
    *ee = cos(phase)*bessel;
    *eb = -sin(phase)*bessel;
  }
}

// Fourier geometry does not depend on source-bin labels. xi rows share a
// kernel only when their kind, separation nodes and weights agree exactly.
// For Map2 the two radii remain ordered: unequal-radius Q(p,q) changes when
// its Fourier legs are exchanged, even though the full estimator commutes.
static int covariance_quadratic_equal_3pcf(
  const covariance_input_3pcf* in, // row definitions and field radii
  int first,                     // first quadratic row
  int second                     // candidate representative row
)
{
  if (in->row_kind[first] != in->row_kind[second]) {
    return 0;
  }
  if (in->row_kind[first] == 0) {
    for (int leg=0; leg<2; leg++) {
      const int a = in->row_fields[(size_t)first*3 + leg];
      const int b = in->row_fields[(size_t)second*3 + leg];
      if (in->field_radius[a] != in->field_radius[b]) {
        return 0;
      }
    }
  } else {
    for (int t=0; t<in->ntheta; t++) {
      const size_t a = (size_t)first*in->ntheta + t;
      const size_t b = (size_t)second*in->ntheta + t;
      if (in->theta[a] != in->theta[b]
        || in->theta_weight[a] != in->theta_weight[b]) {
        return 0;
      }
    }
  }
  return 1;
}

// Prepare source projection, geometric kernels and the shared B triangle
// table. The physical B is evaluated once per (z,q,p,phi), before source
// weights are attached. Neither BB nor P*T shares independent Limber
// distances: each projected polyspectrum is constructed on its own.
static void covariance_prepare_3pcf(covariance_work_3pcf* work)
{
  const covariance_input_3pcf* in = work->in;
  const double* lens;
  const double* shape;
  physics_radial_3pcf(work->model, &work->ns, &work->nz, &work->z,
    &work->chi, &work->dchi, &lens, &shape);
  const int ns = work->ns;
  const int nz = work->nz;
  const int nf = in->nfield;
  const int nl = in->nell;
  const int nr = in->nr;
  const size_t ng = (size_t)nl*nl*in->nphi;
  for (int row=0; row<in->nrow; row++) {
    work->has_cubic = work->has_cubic || in->row_kind[row] == 1;
  }
  work->nt = ns*(ns + 1)*(ns + 2)/6;
  work->lens = covariance_allocate_3pcf((size_t)ns*nz, sizeof(double));
  work->triples = covariance_allocate_3pcf((size_t)work->nt*3, sizeof(int));
  work->triple_index = covariance_allocate_3pcf((size_t)ns*ns*ns,
    sizeof(int));
  work->filters = covariance_allocate_3pcf((size_t)nf*nl, sizeof(double));
  work->power = covariance_allocate_3pcf((size_t)ns*ns*nl, sizeof(double));
  // A purely quadratic request has no B, S or three-field correlation.
  // Keep its Gaussian and connected halo terms, without constructing
  // unused matter triangles or aperture-partition tables.
  if (work->has_cubic) {
    work->closure_filters = covariance_allocate_3pcf((size_t)nf*ng,
      sizeof(double));
    work->bispectrum = covariance_allocate_3pcf((size_t)work->nt*ng,
      sizeof(double));
    work->bspec = covariance_allocate_3pcf((size_t)nf*nf*ns*nl,
      sizeof(double));
    work->breal = covariance_allocate_3pcf((size_t)nf*nf*nf*nr,
      sizeof(double));
    work->pair = covariance_allocate_3pcf((size_t)nf*nf*nr, sizeof(double));
    work->pair_zero = covariance_allocate_3pcf((size_t)nf*nf, sizeof(double));
    work->pair_window = covariance_allocate_3pcf((size_t)nf*nf*nl,
      sizeof(double));
  }
  work->radial_measure = covariance_allocate_3pcf(nr, sizeof(double));
  work->fourier_measure = covariance_allocate_3pcf(nl, sizeof(double));
  work->window_fourier = covariance_allocate_3pcf(in->nwindow,
    sizeof(double));
  // In the large-area limit the quadratic kernel is needed only at
  // opposite equal Fourier vectors. Its small [row,ell] table below is
  // sufficient; allocating [row,ell,ell,phi] would waste memory and work.
  if (!in->large_area) {
    work->quadratic_index = covariance_allocate_3pcf(in->nrow, sizeof(int));
    work->quadratic_representative = covariance_allocate_3pcf(in->nrow,
      sizeof(int));
    for (int row=0; row<in->nrow; row++) {
      work->quadratic_index[row] = -1;
      if (in->row_kind[row] == 1) {
        continue;
      }
      int shape = 0;
      while (shape < work->nquadratic
        && !covariance_quadratic_equal_3pcf(in, row,
          work->quadratic_representative[shape])) {
        shape++;
      }
      if (shape == work->nquadratic) {
        work->quadratic_representative[work->nquadratic++] = row;
      }
      work->quadratic_index[row] = shape;
    }
    work->geometry_window = covariance_allocate_3pcf(ng, sizeof(double));
    if (work->nquadratic > 0) {
      work->quadratic = covariance_allocate_3pcf((size_t)work->nquadratic*ng,
        sizeof(double));
      work->quadratic_mixed = covariance_allocate_3pcf(
        (size_t)work->nquadratic*ng, sizeof(double));
    }
  }
  work->j0_radial = covariance_allocate_3pcf((size_t)nr*nl, sizeof(double));
  work->j0_fourier = covariance_allocate_3pcf((size_t)nl*nr, sizeof(double));
  work->j0_window = covariance_allocate_3pcf((size_t)in->nwindow*nr,
    sizeof(double));
  work->j2_profile = covariance_allocate_3pcf((size_t)in->nprofile*nl,
    sizeof(double));
  work->quadratic_zero = covariance_allocate_3pcf((size_t)in->nrow*nl,
    sizeof(double));
  work->radial_volume = covariance_allocate_3pcf(nr, sizeof(double));

  // Special functions depend on geometry, not on halo mass or cosmology.
  // Keep both J0 orientations so each transform reads contiguous nodes.
  // The duplicated small matrix avoids a strided read in either direction.
  for (int r=0; r<nr; r++) {
    work->radial_volume[r] = 2.0*COV_PI*in->radius[r]*in->dr[r];
    for (int l=0; l<nl; l++) {
      const double value = gsl_sf_bessel_J0(in->ell[l]*in->radius[r]);
      work->j0_radial[(size_t)r*nl + l] = value;
      work->j0_fourier[(size_t)l*nr + r] = value;
    }
    for (int q=0; q<in->nwindow; q++) {
      work->j0_window[(size_t)q*nr + r] =
        gsl_sf_bessel_J0(in->window_ell[q]*in->radius[r]);
    }
  }
  for (int r=0; r<in->nprofile; r++) {
    for (int l=0; l<nl; l++) {
      work->j2_profile[(size_t)r*nl + l] =
        gsl_sf_bessel_Jn(2, in->ell[l]*in->profile_radius[r]);
    }
  }
  for (int row=0; row<in->nrow; row++) {
    if (in->row_kind[row] != 1) {
      for (int l=0; l<nl; l++) {
        double unused;
        covariance_quadratic_3pcf(in, row, in->ell[l], in->ell[l], COV_PI,
          work->quadratic_zero + (size_t)row*nl + l, &unused);
      }
    }
  }

  // The mean model's g already includes the lensing prefactor. Its
  // physical convergence weight is W=chi*(1+z)*g, in h/Mpc.
  for (int source=0; source<ns; source++) {
    for (int iz=0; iz<nz; iz++) {
      work->lens[(size_t)source*nz + iz] = work->chi[iz]
        *(1.0 + work->z[iz])*lens[(size_t)source*nz + iz];
    }
  }
  int count = 0;
  for (int a=0; a<ns; a++) {
    for (int b=a; b<ns; b++) {
      for (int c=b; c<ns; c++) {
        work->triples[3*count] = a;
        work->triples[3*count + 1] = b;
        work->triples[3*count + 2] = c;
        count++;
      }
    }
  }
  // Scalar matter projection depends on a source-label multiset. Store
  // the six permutations through this map, without changing field radii.
  for (int index=0; index<work->nt; index++) {
    const int* t = work->triples + 3*index;
    const int orders[6][3] = {{0,1,2},{0,2,1},{1,0,2},
      {1,2,0},{2,0,1},{2,1,0}};
    for (int p=0; p<6; p++) {
      work->triple_index[((size_t)t[orders[p][0]]*ns
        + t[orders[p][1]])*ns + t[orders[p][2]]] = index;
    }
  }
  for (int field=0; field<nf; field++) {
    covariance_require_3pcf(in->field_source[field] >= 0
      && in->field_source[field] < ns, "field source index out of range");
    for (int l=0; l<nl; l++) {
      work->filters[(size_t)field*nl + l] =
        covariance_filter_3pcf(in->ell[l], in->field_radius[field]);
    }
  }
  for (int r=0; r<nr; r++) {
    const double window = in->large_area ? 1.0/in->area
      : covariance_window_3pcf(in, in->radius[r], 0);
    work->radial_measure[r] = 2.0*COV_PI*in->radius[r]*in->dr[r]*window;
  }
  for (int l=0; l<nl; l++) {
    work->fourier_measure[l] = in->ell[l]*in->dell[l]*COV_I2PI;
  }
  for (int q=0; q<in->nwindow; q++) {
    work->window_fourier[q] =
      covariance_window_3pcf(in, in->window_ell[q], 1);
  }

  // Each geometry fixes the closure side |q+p|. A zero closure carries
  // zero compensated aperture weight; its isolated angular node is set
  // to zero rather than asking the bispectrum for a zero-length triangle.
  if (work->has_cubic || !in->large_area) {
    #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
    for (size_t g=0; g<ng; g++) {
      const int angle = g%in->nphi;
      const int p = (g/in->nphi)%nl;
      const int q = g/((size_t)in->nphi*nl);
      const double closure = covariance_closure_3pcf(in->ell[q], in->ell[p],
        in->phi[angle]);
      if (work->has_cubic) {
        for (int field=0; field<nf; field++) {
          work->closure_filters[(size_t)field*ng + g] =
            covariance_filter_3pcf(closure, in->field_radius[field]);
        }
      }
      if (!in->large_area) {
        work->geometry_window[g] = covariance_window_3pcf(in, closure, 1);
        for (int shape=0; shape<work->nquadratic; shape++) {
          const int row = work->quadratic_representative[shape];
          covariance_quadratic_3pcf(in, row, in->ell[q], in->ell[p],
            in->phi[angle], work->quadratic + (size_t)shape*ng + g,
            work->quadratic_mixed + (size_t)shape*ng + g);
        }
      }
    }
  }

  double* power_z = covariance_allocate_3pcf((size_t)nz*nl, sizeof(double));
  #pragma omp parallel for collapse(2) if(!omp_in_parallel()) schedule(static)
  for (int iz=0; iz<nz; iz++) {
    for (int l=0; l<nl; l++) {
      if (work->chi[iz] > 0.0) {
        power_z[(size_t)iz*nl + l] = physics_power_3pcf(work->model,
          in->ell[l]/work->chi[iz], work->z[iz], 2);
      }
    }
  }
  #pragma omp parallel for collapse(2) if(!omp_in_parallel()) schedule(static)
  for (int a=0; a<ns; a++) {
    for (int b=0; b<ns; b++) {
      for (int l=0; l<nl; l++) {
        double value = 0.0;
        for (int iz=0; iz<nz; iz++) {
          if (work->chi[iz] > 0.0) {
            value += work->dchi[iz]*work->lens[(size_t)a*nz + iz]
              *work->lens[(size_t)b*nz + iz]
              *power_z[(size_t)iz*nl + l]
              /(work->chi[iz]*work->chi[iz]);
          }
        }
        work->power[((size_t)a*ns + b)*nl + l] = value;
      }
    }
  }
  free(power_z);
  if (!work->has_cubic) {
    return;
  }

  int nworker = 1;
  #ifdef _OPENMP
  nworker = omp_get_max_threads();
  #endif
  // Leave a cache-line-sized gap between mutable worker slices even when
  // calloc's base is not cache-line aligned. Workers never allocate scratch.
  const size_t bstride = ((size_t)nz + 7)/8*8 + 8;
  double* local_b = covariance_allocate_3pcf((size_t)nworker*bstride,
    sizeof(double));
  #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
  for (size_t g=0; g<ng; g++) {
    int worker = 0;
    #ifdef _OPENMP
    worker = omp_get_thread_num();
    #endif
    double* bz = local_b + (size_t)worker*bstride;
    const int angle = g%in->nphi;
    const int p = (g/in->nphi)%nl;
    const int q = g/((size_t)in->nphi*nl);
    const double closure = covariance_closure_3pcf(in->ell[q], in->ell[p],
      in->phi[angle]);
    for (int iz=0; iz<nz; iz++) {
      bz[iz] = 0.0;
      if (work->chi[iz] > 0.0 && closure > 0.0) {
        const double chi = work->chi[iz];
        bz[iz] = work->dchi[iz]/(chi*chi*chi*chi)
          *physics_matter_3pcf(work->model, in->ell[q]/chi,
            in->ell[p]/chi, closure/chi, work->z[iz], 0,
            in->baryon_strength);
      }
    }
    for (int t=0; t<work->nt; t++) {
      const int* sources = work->triples + 3*t;
      double value = 0.0;
      for (int iz=0; iz<nz; iz++) {
        value += bz[iz]*work->lens[(size_t)sources[0]*nz + iz]
          *work->lens[(size_t)sources[1]*nz + iz]
          *work->lens[(size_t)sources[2]*nz + iz];
      }
      work->bispectrum[(size_t)t*ng + g] = value;
    }
  }
  free(local_b);

  // S_ab|c(q) contains the two coincident aperture filters and the
  // bispectrum, but not the separated leg's filter. Its measure is
  // p dp dphi/(2*pi)^2. Symmetrizing a and b preserves the exact
  // commutation of fields at one center at finite quadrature spacing.
  #pragma omp parallel for collapse(2) if(!omp_in_parallel()) schedule(static)
  for (int a=0; a<nf; a++) {
    for (int b=0; b<nf; b++) {
      const int sa = in->field_source[a];
      const int sb = in->field_source[b];
      for (int c=0; c<ns; c++) {
        const int t = work->triple_index[((size_t)sa*ns + sb)*ns + c];
        for (int q=0; q<nl; q++) {
          double value = 0.0;
          for (int p=0; p<nl; p++) {
            const size_t g = covariance_geometry_3pcf(in, q, p, 0);
            const double* bvalues = work->bispectrum + (size_t)t*ng + g;
            const double ab = work->filters[(size_t)a*nl + p]
              *covariance_dot_3pcf(in->nphi, in->dphi, bvalues,
                work->closure_filters + (size_t)b*ng + g);
            const double ba = work->filters[(size_t)b*nl + p]
              *covariance_dot_3pcf(in->nphi, in->dphi, bvalues,
                work->closure_filters + (size_t)a*ng + g);
            value += 0.5*(ab + ba)*in->ell[p]*in->dell[p]
              *COV_I2PI*COV_I2PI;
          }
          work->bspec[covariance_bspec_3pcf(work, a, b, c, q)] = value;
        }
      }
    }
  }

  // K(r), K(0), B_ab|c(r), and the windowed K transform are reusable
  // across positional partitions. Every ordered field pair owns its row.
  #pragma omp parallel for collapse(2) if(!omp_in_parallel()) schedule(static)
  for (int a=0; a<nf; a++) {
    for (int b=0; b<nf; b++) {
      const size_t pair = (size_t)a*nf + b;
      const int sa = in->field_source[a];
      const int sb = in->field_source[b];
      double zero = 0.0;
      for (int l=0; l<nl; l++) {
        zero += work->fourier_measure[l]*work->filters[(size_t)a*nl + l]
          *work->filters[(size_t)b*nl + l]
          *covariance_power_value_3pcf(work, sa, sb, l, 0);
      }
      work->pair_zero[pair] = zero;
      for (int r=0; r<nr; r++) {
        double value = 0.0;
        for (int l=0; l<nl; l++) {
          value += work->fourier_measure[l]
            *work->filters[(size_t)a*nl + l]
            *work->filters[(size_t)b*nl + l]
            *covariance_power_value_3pcf(work, sa, sb, l, 0)
            *work->j0_radial[(size_t)r*nl + l];
        }
        work->pair[pair*nr + r] = value;
      }
      for (int l=0; l<nl; l++) {
        double value = 0.0;
        if (in->large_area) {
          value = work->filters[(size_t)a*nl + l]
            *work->filters[(size_t)b*nl + l]
            *covariance_power_value_3pcf(work, sa, sb, l, 0)/in->area;
        } else {
          for (int r=0; r<nr; r++) {
            value += work->radial_measure[r]*work->pair[pair*nr + r]
              *work->j0_radial[(size_t)r*nl + l];
          }
        }
        work->pair_window[pair*nl + l] = value*work->fourier_measure[l];
      }
      for (int c=0; c<nf; c++) {
        const int sc = in->field_source[c];
        for (int r=0; r<nr; r++) {
          double value = 0.0;
          for (int l=0; l<nl; l++) {
            value += work->fourier_measure[l]
              *work->filters[(size_t)c*nl + l]
              *work->bspec[covariance_bspec_3pcf(work, a, b, sc, l)]
              *work->j0_radial[(size_t)r*nl + l];
          }
          work->breal[covariance_breal_3pcf(in, a, b, c, r)] = value;
        }
      }
    }
  }
}

void covariance_xi_from_power_3pcf(
  int nrow,
  int nell,
  int ntheta,
  const double* ell,
  const double* dell,
  const double* theta,
  const double* theta_weight,
  const double* spectra_e,
  const double* spectra_b,
  const int* sign,
  double* out
)
{
  // Repeated source pairs often use identical separation bins. Prepare
  // their Bessel kernel only once, then read contiguous ell nodes for
  // every spectrum. This entry point accepts spectra on the integration
  // nodes already; it performs no ell interpolation or binary search.
  int* representative = covariance_allocate_3pcf(nrow, sizeof(int));
  int* row_kernel = covariance_allocate_3pcf(nrow, sizeof(int));
  int nkernel = 0;
  for (int row=0; row<nrow; row++) {
    covariance_require_3pcf(sign[row] == 1 || sign[row] == -1,
      "xi sign must be +1 or -1");
    int k = 0;
    for (; k<nkernel; k++) {
      const int previous = representative[k];
      int equal = sign[row] == sign[previous];
      for (int t=0; equal && t<ntheta; t++) {
        const size_t i = (size_t)row*ntheta + t;
        const size_t j = (size_t)previous*ntheta + t;
        equal = theta[i] == theta[j] && theta_weight[i] == theta_weight[j];
      }
      if (equal) {
        break;
      }
    }
    if (k == nkernel) {
      representative[nkernel++] = row;
    }
    row_kernel[row] = k;
  }
  double* kernels = covariance_allocate_3pcf((size_t)nkernel*nell,
    sizeof(double));
  #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
  for (int k=0; k<nkernel; k++) {
    const int row = representative[k];
    const int order = sign[row] == 1 ? 0 : 4;
    for (int l=0; l<nell; l++) {
      double kernel = 0.0;
      for (int t=0; t<ntheta; t++) {
        const size_t index = (size_t)row*ntheta + t;
        kernel += theta_weight[index]
          *gsl_sf_bessel_Jn(order, ell[l]*theta[index]);
      }
      kernels[(size_t)k*nell + l] = ell[l]*dell[l]*COV_I2PI*kernel;
    }
  }
  #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
  for (int row=0; row<nrow; row++) {
    const double* kernel = kernels + (size_t)row_kernel[row]*nell;
    const double* ce = spectra_e + (size_t)row*nell;
    const double* cb = spectra_b + (size_t)row*nell;
    double sum = 0.0;
    int l = 0;
    // Both adjacent ell lanes use the same xi+ or xi- B-spectrum sign.
    const v2d parity = simde_mm_set1_pd(sign[row]);
    for (; l + 1 < nell; l += 2) {
      // Load E power at two valid consecutive quadrature nodes.
      const v2d e = simde_mm_loadu_pd(ce + l);
      // Load B power at those same nodes.
      const v2d b = simde_mm_loadu_pd(cb + l);
      // Apply the xi- minus sign, or retain B for xi+.
      const v2d signed_b = simde_mm_mul_pd(parity, b);
      // Form the two independent E plus/minus B spectra.
      const v2d signal = simde_mm_add_pd(e, signed_b);
      // Load the shared Bessel/bin/Fourier weights for these ell nodes.
      const v2d weight = simde_mm_loadu_pd(kernel + l);
      // Form the two weighted integrand values independently.
      const v2d value = simde_mm_mul_pd(weight, signal);
      double lanes[2];
      // Store the valid lanes and retain the scalar integration order.
      simde_mm_storeu_pd(lanes, value);
      sum += lanes[0];
      sum += lanes[1];
    }
    if (l < nell) {
      sum += kernel[l]*(ce[l] + sign[row]*cb[l]);
    }
    out[row] = sum;
  }
  free(kernels);
  free(representative);
  free(row_kernel);
}

// The halo's projected shape depends on aperture radius, mass and distance;
// its source-bin dependence is only the multiplicative lensing weight W.
// Store unordered radius pairs by their position in an upper triangle.
// This identity applies to the scalar lensing halo model implemented here.
static inline size_t covariance_radius_pair_3pcf(int n, int a, int b)
{
  const int low = a < b ? a : b;
  const int high = a < b ? b : a;
  return (size_t)low*(2*(size_t)n - low + 1)/2 + high - low;
}

// Geometry of the shear lookup, owned by one covariance call. A natural
// cubic spline is prepared on x=log(1+r/r_scale), including r=0. It is then
// sampled onto a uniform x grid. Every displaced xi endpoint stores its
// fine-grid cell and fraction before the halo loop, so repeated reads need
// neither a binary search nor logarithms. This follows the preparation /
// linear-read separation in cosmo2D.c; input quadrature may be nonuniform.
typedef struct {
  int coarse;
  int fine;
  int per_radius;
  double* x;
  double* inverse;
  double* upper;
  double* coefficient;
  double* forward;
  double* values;
  int* cell;
  double* delta;
  int* minus_cell;
  int* plus_cell;
  double* minus_fraction;
  double* plus_fraction;
  double* endpoint_weight;
} covariance_profile_plan_3pcf;

static void covariance_profile_plan_3pcf_prepare(
  const covariance_input_3pcf* in, // fixed profile and xi-bin geometry
  int nshape,                     // distinct estimator geometries
  const int* representative,     // original row for each shape
  const int* xi_shape,            // xi-only index, -1 for aperture products
  int nxi,                        // number of distinct xi geometries
  covariance_profile_plan_3pcf* plan // output, released after this call
)
{
  memset(plan, 0, sizeof(*plan));
  if (nxi == 0) {
    return;
  }
  const int nc = in->nprofile;
  covariance_require_3pcf(nc >= 3 && in->profile_upsample >= 1
    && nc - 1 <= (INT_MAX - 1)/in->profile_upsample,
    "profile preparation needs >=3 nodes and a valid upsample factor");
  covariance_require_3pcf(in->ntheta <= INT_MAX/in->nphi,
    "xi endpoint orientation count exceeds the index range");
  plan->coarse = nc;
  plan->fine = (nc - 1)*in->profile_upsample + 1;
  plan->per_radius = in->ntheta*in->nphi;
  const int nfine = plan->fine;
  const size_t endpoints = (size_t)nxi*in->nr*plan->per_radius;
  plan->x = covariance_allocate_3pcf(nc, sizeof(double));
  plan->inverse = covariance_allocate_3pcf(nc, sizeof(double));
  plan->upper = covariance_allocate_3pcf(nc, sizeof(double));
  plan->coefficient = covariance_allocate_3pcf(nc, sizeof(double));
  plan->forward = covariance_allocate_3pcf(nc, sizeof(double));
  plan->values = covariance_allocate_3pcf(nfine, sizeof(double));
  plan->cell = covariance_allocate_3pcf(nfine, sizeof(int));
  plan->delta = covariance_allocate_3pcf(nfine, sizeof(double));
  plan->minus_cell = covariance_allocate_3pcf(endpoints, sizeof(int));
  plan->plus_cell = covariance_allocate_3pcf(endpoints, sizeof(int));
  plan->minus_fraction = covariance_allocate_3pcf(endpoints, sizeof(double));
  plan->plus_fraction = covariance_allocate_3pcf(endpoints, sizeof(double));
  plan->endpoint_weight = covariance_allocate_3pcf(endpoints, sizeof(double));
  const double scale = in->profile_radius[1];
  covariance_require_3pcf(in->profile_radius[0] == 0.0 && scale > 0.0,
    "profile preparation nodes must start with zero then a positive radius");
  for (int r=0; r<nc; r++) {
    plan->x[r] = log1p(in->profile_radius[r]/scale);
    covariance_require_3pcf(isfinite(plan->x[r])
      && (r == 0 || plan->x[r] > plan->x[r - 1]),
      "profile coordinates must be finite and strictly increasing");
  }
  // The tridiagonal spline matrix depends only on the coarse geometry.
  // Its elimination factors are shared by every mass/redshift node.
  for (int r=1; r<nc - 1; r++) {
    const double previous = plan->x[r] - plan->x[r - 1];
    const double next = plan->x[r + 1] - plan->x[r];
    plan->inverse[r] = 1.0/(2.0*(previous + next)
      - previous*plan->upper[r - 1]);
    plan->upper[r] = next*plan->inverse[r];
  }
  const double step = plan->x[nc - 1]/(nfine - 1);
  int coarse = 0;
  for (int r=0; r<nfine; r++) {
    const double x = r == nfine - 1 ? plan->x[nc - 1] : r*step;
    while (coarse < nc - 2 && x > plan->x[coarse + 1]) {
      coarse++;
    }
    plan->cell[r] = coarse;
    plan->delta[r] = x - plan->x[coarse];
  }
  for (int s=0; s<nshape; s++) {
    if (xi_shape[s] < 0) {
      continue;
    }
    const int row = representative[s];
    for (int r=0; r<in->nr; r++) {
      for (int t=0; t<in->ntheta; t++) {
        const size_t ti = (size_t)row*in->ntheta + t;
        const double half = 0.5*in->theta[ti];
        for (int a=0; a<in->nphi; a++) {
          const size_t index = ((size_t)xi_shape[s]*in->nr + r)
            *plan->per_radius + (size_t)t*in->nphi + a;
          const double dx = half*cos(in->phi[a]);
          const double dy = half*sin(in->phi[a]);
          const double xm = in->radius[r] - dx;
          const double xp = in->radius[r] + dx;
          const double rm = hypot(xm, dy);
          const double rp = hypot(xp, dy);
          covariance_require_3pcf(rm <= in->profile_radius[nc - 1]
            && rp <= in->profile_radius[nc - 1],
            "profile grid does not enclose displaced xi endpoints");
          const double um = log1p(rm/scale)/step;
          const double up = log1p(rp/scale)/step;
          const int im = um >= nfine - 1 ? nfine - 2 : (int)um;
          const int ip = up >= nfine - 1 ? nfine - 2 : (int)up;
          plan->minus_cell[index] = im;
          plan->plus_cell[index] = ip;
          plan->minus_fraction[index] = um - im;
          plan->plus_fraction[index] = up - ip;
          const double phm = atan2(-dy, xm);
          const double php = atan2(dy, xp);
          const double phase = in->row_kind[row] == 2
            ? 2.0*(phm - php) : 2.0*(phm + php) - 4.0*in->phi[a];
          plan->endpoint_weight[index] = in->theta_weight[ti]
            *in->dphi[a]*COV_I2PI*cos(phase);
        }
      }
    }
  }
}

// Update only the spline ordinates for a new halo; every geometric factor
// and destination interval was prepared once above. The coefficients c
// are one half of the spline's second derivative. Natural endpoints c=0
// remain zero throughout the call. Refine both coarse and fine grids.
static void covariance_profile_fill_3pcf(
  covariance_profile_plan_3pcf* plan, // fixed geometry and owned scratch
  const double* shear               // coarse gamma_t [plan->coarse]
)
{
  const int nc = plan->coarse;
  for (int r=1; r<nc - 1; r++) {
    const double previous = plan->x[r] - plan->x[r - 1];
    const double next = plan->x[r + 1] - plan->x[r];
    const double rhs = 3.0*((shear[r + 1] - shear[r])/next
      - (shear[r] - shear[r - 1])/previous);
    plan->forward[r] = (rhs - previous*plan->forward[r - 1])
      *plan->inverse[r];
  }
  for (int r=nc - 2; r>=1; r--) {
    plan->coefficient[r] = plan->forward[r]
      - plan->upper[r]*plan->coefficient[r + 1];
  }
  for (int r=0; r<plan->fine; r++) {
    const int i = plan->cell[r];
    const double width = plan->x[i + 1] - plan->x[i];
    const double c = plan->coefficient[i];
    const double b = (shear[i + 1] - shear[i])/width
      - width*(2.0*c + plan->coefficient[i + 1])/3.0;
    const double d = (plan->coefficient[i + 1] - c)/(3.0*width);
    const double dx = plan->delta[r];
    plan->values[r] = shear[i] + dx*(b + dx*(c + dx*d));
  }
}

// Two orientation nodes occupy SIMD lanes; the halo profile is gathered
// using the already prepared cells. Accumulate low then high lane, keeping
// the same fixed node order for any OpenMP thread count. The final odd node
// is scalar, so no load may extend beyond the endpoint arrays.
static inline double covariance_profile_response_3pcf(
  const covariance_profile_plan_3pcf* plan, // prepared halo and geometry
  size_t start                            // first endpoint for this radius
)
{
  double sum = 0.0;
  int a = 0;
  for (; a + 1 < plan->per_radius; a += 2) {
    const size_t p = start + a;
    const int m0 = plan->minus_cell[p];
    const int m1 = plan->minus_cell[p + 1];
    const int p0 = plan->plus_cell[p];
    const int p1 = plan->plus_cell[p + 1];
    // Gather the lower-minus cell values for two adjacent angular nodes.
    const v2d ml = simde_mm_set_pd(plan->values[m1], plan->values[m0]);
    // Gather the upper-minus values, each from a valid neighboring cell.
    const v2d mh = simde_mm_set_pd(plan->values[m1 + 1], plan->values[m0 + 1]);
    // Load the two prepared fractional positions in those cells.
    const v2d mf = simde_mm_loadu_pd(plan->minus_fraction + p);
    // Subtract the cell endpoints to form the linear-interpolation slopes.
    const v2d md = simde_mm_sub_pd(mh, ml);
    // Multiply each slope by its endpoint's fractional displacement.
    const v2d ms = simde_mm_mul_pd(mf, md);
    // Add the lower values to obtain the two minus-endpoint shear values.
    const v2d minus = simde_mm_add_pd(ml, ms);
    // Gather the lower-plus values for the same two angular nodes.
    const v2d pl = simde_mm_set_pd(plan->values[p1], plan->values[p0]);
    // Gather the corresponding upper-plus values.
    const v2d ph = simde_mm_set_pd(plan->values[p1 + 1], plan->values[p0 + 1]);
    // Load the already prepared plus-endpoint fractional positions.
    const v2d pf = simde_mm_loadu_pd(plan->plus_fraction + p);
    // Form the independent plus-cell slopes.
    const v2d pd = simde_mm_sub_pd(ph, pl);
    // Scale the slopes by the fractional positions.
    const v2d ps = simde_mm_mul_pd(pf, pd);
    // Add the lower values to recover both plus-endpoint shears.
    const v2d plus = simde_mm_add_pd(pl, ps);
    // Multiply shears at the two physical endpoints for each orientation.
    const v2d pair = simde_mm_mul_pd(minus, plus);
    // Load bin, angular and spin-projection weights for the two nodes.
    const v2d weights = simde_mm_loadu_pd(plan->endpoint_weight + p);
    // Apply the prepared scalar-estimator weights to each shear product.
    const v2d weighted = simde_mm_mul_pd(weights, pair);
    double values[2];
    // Store valid lanes locally before adding them in fixed scalar order.
    simde_mm_storeu_pd(values, weighted);
    sum += values[0];
    sum += values[1];
  }
  if (a < plan->per_radius) {
    const size_t p = start + a;
    const int m = plan->minus_cell[p];
    const int u = plan->plus_cell[p];
    const double minus = plan->values[m] + plan->minus_fraction[p]
      *(plan->values[m + 1] - plan->values[m]);
    const double plus = plan->values[u] + plan->plus_fraction[p]
      *(plan->values[u + 1] - plan->values[u]);
    sum += plan->endpoint_weight[p]*minus*plus;
  }
  return sum;
}

static void covariance_profile_release_3pcf(
  covariance_profile_plan_3pcf* plan // all allocations from preparation
)
{
  free(plan->x);
  free(plan->inverse);
  free(plan->upper);
  free(plan->coefficient);
  free(plan->forward);
  free(plan->values);
  free(plan->cell);
  free(plan->delta);
  free(plan->minus_cell);
  free(plan->plus_cell);
  free(plan->minus_fraction);
  free(plan->plus_fraction);
  free(plan->endpoint_weight);
}

// Evaluate one halo's homogeneous estimator response as a function of
// its angular offset u from an estimator center. Aperture products use
// products of radial filtered convergence profiles. xi uses two spin-2
// shear values at the actual separated endpoints, averaged with the
// supplied theta-bin and orientation weights. The shear profile is first
// tabulated on a wider grid so these displaced samples are interpolated,
// not repeatedly transformed. Refine that interpolation independently.
static void covariance_halo_terms_3pcf(
  covariance_work_3pcf* work, double* terms
)
{
  const covariance_input_3pcf* in = work->in;
  const int nf = in->nfield;
  const int nl = in->nell;
  const int nr = in->nr;
  const int no = in->nrow;
  const int nw = in->nwindow;
  const int nm = in->nmass;
  const size_t matrix = (size_t)no*no;
  const size_t nodes = (size_t)work->nz*nm;
  const double rho = 2.77536627e11*physics_omega_m_3pcf(work->model);
  work->sigma2 = covariance_allocate_3pcf(nodes, sizeof(double));
  work->abundance = covariance_allocate_3pcf(nodes, sizeof(double));
  work->bias = covariance_allocate_3pcf(nodes, sizeof(double));
  work->concentration = covariance_allocate_3pcf(nodes, sizeof(double));
  work->response_bias = covariance_allocate_3pcf((size_t)work->nz*no,
    sizeof(double));
  covariance_halos_3pcf(work->model, in, work->nz, work->z, 0, NULL,
    work->sigma2, work->abundance, work->bias, work->concentration, NULL);

  // Equality here means equality of the supplied numerical radii, not a
  // tolerance-based merging of different observables. Source labels are
  // retained in field_lens and row_lens, outside all halo transforms.
  int* radius_index = covariance_allocate_3pcf(nf, sizeof(int));
  double* unique_radius = covariance_allocate_3pcf(nf, sizeof(double));
  int nrad = 0;
  for (int a=0; a<nf; a++) {
    int index = 0;
    while (index < nrad && unique_radius[index] != in->field_radius[a]) {
      index++;
    }
    if (index == nrad) {
      unique_radius[nrad++] = in->field_radius[a];
    }
    radius_index[a] = index;
  }
  const size_t npair = (size_t)nrad*(nrad + 1)/2;
  int* pair_needed = covariance_allocate_3pcf(npair, sizeof(int));
  int* radius_needed = covariance_allocate_3pcf(nrad, sizeof(int));
  int* representative = covariance_allocate_3pcf(no, sizeof(int));
  int* shape_radius = covariance_allocate_3pcf((size_t)no*3, sizeof(int));
  int* row_shape = covariance_allocate_3pcf(no, sizeof(int));
  int* xi_shape = covariance_allocate_3pcf(no, sizeof(int));
  int nshape = 0;
  int nxi = 0;
  for (int row=0; row<no; row++) {
    const int kind = in->row_kind[row];
    const int nleg = kind == 1 ? 3 : 2;
    const int* fields = in->row_fields + (size_t)row*3;
    int radii[3] = {-1, -1, -1};
    if (kind <= 1) {
      for (int a=0; a<nleg; a++) {
        radii[a] = radius_index[fields[a]];
        radius_needed[radii[a]] = 1;
      }
      for (int a=0; a<nleg; a++) {
        for (int b=a + 1; b<nleg; b++) {
          if (radii[b] < radii[a]) {
            const int saved = radii[a];
            radii[a] = radii[b];
            radii[b] = saved;
          }
        }
      }
    }
    if (kind == 1) {
      for (int a=0; a<3; a++) {
        pair_needed[covariance_radius_pair_3pcf(nrad,
          radii[(a + 1)%3], radii[(a + 2)%3])] = 1;
      }
    }
    int s = 0;
    for (; s<nshape; s++) {
      const int original = representative[s];
      if (in->row_kind[original] != kind) {
        continue;
      }
      int equal = 1;
      if (kind <= 1) {
        for (int a=0; a<nleg; a++) {
          equal = equal && radii[a] == shape_radius[3*s + a];
        }
      } else {
        for (int t=0; t<in->ntheta; t++) {
          const size_t here = (size_t)row*in->ntheta + t;
          const size_t there = (size_t)original*in->ntheta + t;
          equal = equal && in->theta[here] == in->theta[there]
            && in->theta_weight[here] == in->theta_weight[there];
        }
      }
      if (equal) {
        break;
      }
    }
    if (s == nshape) {
      representative[s] = row;
      for (int a=0; a<3; a++) {
        shape_radius[3*s + a] = radii[a];
      }
      xi_shape[s] = kind >= 2 ? nxi++ : -1;
      nshape++;
    }
    row_shape[row] = s;
  }
  covariance_profile_plan_3pcf profile_plan;
  covariance_profile_plan_3pcf_prepare(in, nshape, representative, xi_shape,
    nxi, &profile_plan);
  double* unique_filter = covariance_allocate_3pcf((size_t)nrad*nl,
    sizeof(double));
  double* window_filter = covariance_allocate_3pcf((size_t)nrad*nw,
    sizeof(double));
  for (int a=0; a<nrad; a++) {
    for (int l=0; l<nl; l++) {
      unique_filter[(size_t)a*nl + l] =
        covariance_filter_3pcf(in->ell[l], unique_radius[a]);
    }
    for (int q=0; q<nw; q++) {
      window_filter[(size_t)a*nw + q] =
        covariance_filter_3pcf(in->window_ell[q], unique_radius[a]);
    }
  }

  double* profile_fourier = covariance_allocate_3pcf((size_t)nrad*nl,
    sizeof(double));
  double* profile_real = covariance_allocate_3pcf((size_t)nrad*nr,
    sizeof(double));
  double* profile_window = covariance_allocate_3pcf((size_t)nf*nw,
    sizeof(double));
  double* shear = covariance_allocate_3pcf(in->nprofile, sizeof(double));
  double* nfw = covariance_allocate_3pcf(nl, sizeof(double));
  double* nfw_square = covariance_allocate_3pcf(nl, sizeof(double));
  double* nfw_window = covariance_allocate_3pcf(nw, sizeof(double));
  double* field_lens = covariance_allocate_3pcf(nf, sizeof(double));
  double* row_lens = covariance_allocate_3pcf(no, sizeof(double));
  double* product_pair = covariance_allocate_3pcf(npair*nl,
    sizeof(double));
  double* product_real = covariance_allocate_3pcf(npair*nr,
    sizeof(double));
  double* response_real = covariance_allocate_3pcf((size_t)nshape*nr,
    sizeof(double));
  double* response_fourier = covariance_allocate_3pcf((size_t)nshape*nw,
    sizeof(double));
  double* shape_zero = covariance_allocate_3pcf(nshape, sizeof(double));
  double* response_zero = covariance_allocate_3pcf(no, sizeof(double));
  double* window_measure = covariance_allocate_3pcf(nw, sizeof(double));
  for (int q=0; q<nw; q++) {
    window_measure[q] = in->window_ell[q]*in->window_dell[q]*COV_I2PI
      *work->window_fourier[q];
  }

  // Halo nodes are visited in a fixed order. At each node all field
  // profiles are shared across covariance cells; workers own output rows.
  // This keeps memory independent of the number of mass/redshift nodes.
  for (int iz=0; iz<work->nz; iz++) {
    const double chi = work->chi[iz];
    if (chi <= 0.0) {
      continue;
    }
    for (int a=0; a<nf; a++) {
      field_lens[a] = work->lens[(size_t)in->field_source[a]*work->nz + iz];
    }
    for (int row=0; row<no; row++) {
      const int* fields = in->row_fields + (size_t)row*3;
      row_lens[row] = field_lens[fields[0]]*field_lens[fields[1]];
      if (in->row_kind[row] == 1) {
        row_lens[row] *= field_lens[fields[2]];
      }
    }
    for (int im=0; im<nm; im++) {
      const size_t node = (size_t)iz*nm + im;
      const double mass = in->mass[im];
      const double amplitude = mass/(rho*chi*chi);
      const double r200 = cbrt(3.0*mass/(800.0*COV_PI*rho));
      const double number = in->dmass[im]*work->abundance[node];
      const double measure = work->dchi[iz]*chi*chi*number;
      for (int l=0; l<nl; l++) {
        nfw[l] = amplitude*covariance_nfw_3pcf(in->ell[l]/chi,
          r200, work->concentration[node]);
        nfw_square[l] = nfw[l]*nfw[l];
      }
      for (int q=0; q<nw; q++) {
        nfw_window[q] = amplitude*covariance_nfw_3pcf(
          in->window_ell[q]/chi, r200, work->concentration[node]);
      }
      #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
      for (int a=0; a<nrad; a++) {
        if (!radius_needed[a]) {
          continue;
        }
        for (int l=0; l<nl; l++) {
          profile_fourier[(size_t)a*nl + l] = nfw[l]
            *unique_filter[(size_t)a*nl + l];
        }
        for (int r=0; r<nr; r++) {
          profile_real[(size_t)a*nr + r] = covariance_dot_3pcf(nl,
            work->fourier_measure, profile_fourier + (size_t)a*nl,
            work->j0_radial + (size_t)r*nl);
        }
      }
      for (int a=0; a<nf; a++) {
        for (int q=0; q<nw; q++) {
          profile_window[(size_t)a*nw + q] = field_lens[a]*nfw_window[q]
            *window_filter[(size_t)radius_index[a]*nw + q];
        }
      }
      if (nxi > 0) {
        #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
        for (int r=0; r<in->nprofile; r++) {
          shear[r] = covariance_dot_3pcf(nl, work->fourier_measure, nfw,
            work->j2_profile + (size_t)r*nl);
        }
        covariance_profile_fill_3pcf(&profile_plan, shear);
      }

      #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
      for (int s=0; s<nshape; s++) {
        const int row = representative[s];
        const int kind = in->row_kind[row];
        for (int r=0; r<nr; r++) {
          double value;
          if (kind <= 1) {
            value = profile_real[(size_t)shape_radius[3*s]*nr + r]
              *profile_real[(size_t)shape_radius[3*s + 1]*nr + r];
            if (kind == 1) {
              value *= profile_real[(size_t)shape_radius[3*s + 2]*nr + r];
            }
          } else {
            const size_t start = ((size_t)xi_shape[s]*nr + r)
              *profile_plan.per_radius;
            value = covariance_profile_response_3pcf(&profile_plan, start);
          }
          response_real[(size_t)s*nr + r] = value;
        }
        double zero = 0.0;
        for (int r=0; r<nr; r++) {
          zero += work->radial_volume[r]*response_real[(size_t)s*nr + r];
        }
        // The two-point zero-mode response has a direct Fourier check,
        // also avoiding halo-offset truncation for the large-area limit.
        if (kind != 1) {
          zero = covariance_dot_3pcf(nl, work->fourier_measure,
            work->quadratic_zero + (size_t)row*nl, nfw_square);
        }
        shape_zero[s] = zero;
        for (int q=0; q<nw; q++) {
          response_fourier[(size_t)s*nw + q] = covariance_dot_3pcf(nr,
            work->radial_volume, response_real + (size_t)s*nr,
            work->j0_window + (size_t)q*nr);
        }
      }
      for (int row=0; row<no; row++) {
        response_zero[row] = row_lens[row]*shape_zero[row_shape[row]];
        work->response_bias[(size_t)iz*no + row] +=
          number*work->bias[node]*response_zero[row];
      }

      // A product of two radial halo profiles supplies the connected
      // one-halo four-field function. One unordered radius pair supplies
      // every source pair with those radii; W factors are attached below.
      // Skip pairs absent from the requested cubic rows.
      #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
      for (int a=0; a<nrad; a++) {
        for (int b=a; b<nrad; b++) {
          const size_t pair = covariance_radius_pair_3pcf(nrad, a, b);
          if (!pair_needed[pair]) {
            continue;
          }
          for (int r=0; r<nr; r++) {
            product_real[pair*nr + r] = profile_real[(size_t)a*nr + r]
              *profile_real[(size_t)b*nr + r];
          }
          for (int l=0; l<nl; l++) {
            product_pair[pair*nl + l] = covariance_dot_3pcf(nr,
              work->radial_volume, product_real + pair*nr,
              work->j0_fourier + (size_t)l*nr);
          }
        }
      }

      #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
      for (int i=0; i<no; i++) {
        const int* a = in->row_fields + (size_t)i*3;
        for (int j=i; j<no; j++) {
          const int* b = in->row_fields + (size_t)j*3;
          const size_t cell = (size_t)i*no + j;
          const double connected = in->large_area
            ? response_zero[i]*response_zero[j]/in->area
            : row_lens[i]*row_lens[j]*covariance_dot_3pcf(nw, window_measure,
                response_fourier + (size_t)row_shape[i]*nw,
                response_fourier + (size_t)row_shape[j]*nw);
          terms[4*matrix + cell] += measure*connected;
          if (in->row_kind[i] == 1 && in->row_kind[j] == 1) {
            double pt = 0.0;
            for (int p=0; p<3; p++) {
              const int a0 = a[(p + 1)%3];
              const int a1 = a[(p + 2)%3];
              for (int q=0; q<3; q++) {
                const int b0 = b[(q + 1)%3];
                const int b1 = b[(q + 2)%3];
                const size_t pa = covariance_radius_pair_3pcf(nrad,
                  radius_index[a0], radius_index[a1]);
                const size_t pb = covariance_radius_pair_3pcf(nrad,
                  radius_index[b0], radius_index[b1]);
                pt += field_lens[a0]*field_lens[a1]
                  *field_lens[b0]*field_lens[b1]*covariance_dot_3pcf(nl,
                  work->pair_window + ((size_t)a[p]*nf + b[q])*nl,
                  product_pair + pa*nl, product_pair + pb*nl);
              }
              if (!in->large_area) {
                pt += work->pair_zero[(size_t)a0*nf + a1]*row_lens[j]
                  *covariance_dot_3pcf(nw, window_measure,
                    profile_window + (size_t)a[p]*nw,
                    response_fourier + (size_t)row_shape[j]*nw);
                const int b0 = b[(p + 1)%3];
                const int b1 = b[(p + 2)%3];
                pt += work->pair_zero[(size_t)b0*nf + b1]*row_lens[i]
                  *covariance_dot_3pcf(nw, window_measure,
                    profile_window + (size_t)b[p]*nw,
                    response_fourier + (size_t)row_shape[i]*nw);
              }
            }
            terms[2*matrix + cell] += measure*pt;
          }
        }
      }
    }
  }

  // The selected squeezed two-halo term uses the abundance responses
  // R_i(chi)=integral dM dn/dM b H_i. Limber's radial delta gives
  // integral dchi chi^2 R_i R_j integral q dq P_linear(q/chi)G(q)/(2*pi).
  // It retains the finite survey window even when short-mode terms use
  // the large-area approximation. No second copy of this term is SSC.
  for (int iz=0; iz<work->nz; iz++) {
    const double chi = work->chi[iz];
    if (chi <= 0.0) {
      continue;
    }
    double variance = 0.0;
    for (int q=0; q<nw; q++) {
      variance += window_measure[q]*physics_power_3pcf(work->model,
        in->window_ell[q]/chi, work->z[iz], 0);
    }
    const double measure = work->dchi[iz]*chi*chi*variance;
    for (int i=0; i<no; i++) {
      for (int j=i; j<no; j++) {
        terms[5*matrix + (size_t)i*no + j] += measure
          *work->response_bias[(size_t)iz*no + i]
          *work->response_bias[(size_t)iz*no + j];
      }
    }
  }
  free(profile_fourier);
  free(profile_real);
  free(profile_window);
  free(shear);
  free(nfw);
  free(nfw_square);
  free(nfw_window);
  free(field_lens);
  free(row_lens);
  free(product_pair);
  free(product_real);
  free(response_real);
  free(response_fourier);
  free(response_zero);
  free(shape_zero);
  free(window_measure);
  free(radius_index);
  free(unique_radius);
  free(pair_needed);
  free(radius_needed);
  free(representative);
  free(shape_radius);
  free(row_shape);
  free(xi_shape);
  free(unique_filter);
  free(window_filter);
  covariance_profile_release_3pcf(&profile_plan);
}

// Gaussian covariance of two quadratic estimators. For finite windows
// both Fourier vectors are retained. The second Wick pairing reverses
// their assignment to row J; unequal Map2 filters must be swapped too.
// E/B spectra are diagonal in parity space. Shape noise supplies equal
// E and B spectra; treating that noise as E alone biases xi variances.
//
// noise_mode=1 declares continuous, uniformly area-weighted xi annuli in
// the large-area limit. Hankel completeness gives integral ell dell/(2pi)
// Jbar_nu,i Jbar_nu,j = Omega_overlap/(Omega_i Omega_j). Both shear
// components contribute, so NN=2*(N_ac N_bd+N_ad N_bc) times this overlap
// divided by survey area. Opposite xi parities cancel. Form CC/CN directly
// in that mode: subtracting a large finite-band NN would lose small signal
// contributions through rounding. Other Gaussian blocks keep their filters.
static double covariance_gaussian_two_3pcf(
  const covariance_work_3pcf* work, int i, int j
)
{
  const covariance_input_3pcf* in = work->in;
  const int nl = in->nell;
  const size_t ng = (size_t)nl*nl*in->nphi;
  const int* ai = in->row_fields + (size_t)i*3;
  const int* aj = in->row_fields + (size_t)j*3;
  const int a = in->field_source[ai[0]];
  const int b = in->field_source[ai[1]];
  const int c = in->field_source[aj[0]];
  const int d = in->field_source[aj[1]];
  const int both_shear = in->row_kind[i] >= 2 && in->row_kind[j] >= 2;
  const double bb_sign_i = in->row_kind[i] == 3 ? -1.0 : 1.0;
  const double bb_sign_j = in->row_kind[j] == 3 ? -1.0 : 1.0;
  double value = 0.0;
  for (int q=0; q<nl; q++) {
    if (in->large_area) {
      const double qi = work->quadratic_zero[(size_t)i*nl + q];
      const double qj = work->quadratic_zero[(size_t)j*nl + q];
      double spectra;
      if (both_shear && in->noise_mode == 1) {
        const size_t ac = (size_t)a*work->ns + c;
        const size_t bd = (size_t)b*work->ns + d;
        const size_t ad = (size_t)a*work->ns + d;
        const size_t bc = (size_t)b*work->ns + c;
        const double cac = work->power[ac*nl + q];
        const double cbd = work->power[bd*nl + q];
        const double cad = work->power[ad*nl + q];
        const double cbc = work->power[bc*nl + q];
        spectra = cac*cbd + cad*cbc
          + cac*in->noise[bd] + in->noise[ac]*cbd
          + cad*in->noise[bc] + in->noise[ad]*cbc;
      } else {
        spectra = covariance_power_value_3pcf(work, a, c, q, 0)
            *covariance_power_value_3pcf(work, b, d, q, 0)
          + covariance_power_value_3pcf(work, a, d, q, 0)
            *covariance_power_value_3pcf(work, b, c, q, 0);
        if (both_shear) {
          spectra += bb_sign_i*bb_sign_j
            *(in->noise[(size_t)a*work->ns + c]
                *in->noise[(size_t)b*work->ns + d]
              + in->noise[(size_t)a*work->ns + d]
                *in->noise[(size_t)b*work->ns + c]);
        }
      }
      value += work->fourier_measure[q]*qi*qj*spectra/in->area;
      continue;
    }
    for (int p=0; p<nl; p++) {
      const double efirst = covariance_power_value_3pcf(work, a, c, q, 0)
        *covariance_power_value_3pcf(work, b, d, p, 0);
      const double esecond = covariance_power_value_3pcf(work, a, d, q, 0)
        *covariance_power_value_3pcf(work, b, c, p, 0);
      double bbspectra = 0.0;
      double mixedspectra = 0.0;
      if (both_shear) {
        bbspectra = bb_sign_i*bb_sign_j
          *(covariance_power_value_3pcf(work, a, c, q, 1)
              *covariance_power_value_3pcf(work, b, d, p, 1)
            + covariance_power_value_3pcf(work, a, d, q, 1)
              *covariance_power_value_3pcf(work, b, c, p, 1));
        const double mixed_sign_i = in->row_kind[i] == 2 ? -1.0 : 1.0;
        const double mixed_sign_j = in->row_kind[j] == 2 ? -1.0 : 1.0;
        mixedspectra = covariance_power_value_3pcf(work, a, c, q, 0)
            *covariance_power_value_3pcf(work, b, d, p, 1)
          + covariance_power_value_3pcf(work, a, d, q, 0)
            *covariance_power_value_3pcf(work, b, c, p, 1)
          + mixed_sign_i*mixed_sign_j
            *(covariance_power_value_3pcf(work, a, c, q, 1)
                *covariance_power_value_3pcf(work, b, d, p, 0)
              + covariance_power_value_3pcf(work, a, d, q, 1)
                *covariance_power_value_3pcf(work, b, c, p, 0));
      }
      double angle_sum = 0.0;
      for (int angle=0; angle<in->nphi; angle++) {
        const size_t g = covariance_geometry_3pcf(in, q, p, angle);
        const size_t ii = (size_t)work->quadratic_index[i]*ng + g;
        const size_t jj = (size_t)work->quadratic_index[j]*ng + g;
        const double qi = work->quadratic[ii];
        const double qj = work->quadratic[jj];
        const double swapped = in->row_kind[j] == 0
          ? work->filters[(size_t)aj[0]*nl + p]
              *work->filters[(size_t)aj[1]*nl + q] : qj;
        const double mixed = work->quadratic_mixed[ii]
          *work->quadratic_mixed[jj];
        angle_sum += in->dphi[angle]*work->geometry_window[g]
          *(qi*(qj*(efirst + bbspectra) + swapped*esecond)
            + mixed*mixedspectra);
      }
      value += work->fourier_measure[q]*work->fourier_measure[p]
        *COV_I2PI*angle_sum;
    }
  }
  if (both_shear && in->noise_mode == 1
    && in->row_kind[i] == in->row_kind[j]) {
    const double lo_i = in->theta_edges[(size_t)i*2];
    const double hi_i = in->theta_edges[(size_t)i*2 + 1];
    const double lo_j = in->theta_edges[(size_t)j*2];
    const double hi_j = in->theta_edges[(size_t)j*2 + 1];
    const double low = fmax(lo_i, lo_j);
    const double high = fmin(hi_i, hi_j);
    if (high > low) {
      const double area_i = COV_PI*(hi_i - lo_i)*(hi_i + lo_i);
      const double area_j = COV_PI*(hi_j - lo_j)*(hi_j + lo_j);
      const double overlap = COV_PI*(high - low)*(high + low);
      const double noise = in->noise[(size_t)a*work->ns + c]
          *in->noise[(size_t)b*work->ns + d]
        + in->noise[(size_t)a*work->ns + d]
          *in->noise[(size_t)b*work->ns + c];
      value += 2.0*noise*overlap/(in->area*area_i*area_j);
    }
  }
  return value;
}

// PB cross-covariance has nine positional choices after mean subtraction.
// Six choose one leg in each estimator. The two remaining aperture legs
// are already integrated into S, leaving two vectors and their relative
// angle. Three choices pair two aperture legs internally and retain their
// K(0). Compensation removes the latter only in the large-area limit.
static double covariance_pb_3pcf(
  const covariance_work_3pcf* work, int quadratic, int cubic
)
{
  const covariance_input_3pcf* in = work->in;
  const int nf = in->nfield;
  const int nl = in->nell;
  const int ns = work->ns;
  const size_t ng = (size_t)nl*nl*in->nphi;
  const int* a = in->row_fields + (size_t)quadratic*3;
  const int* b = in->row_fields + (size_t)cubic*3;
  double value = 0.0;
  for (int q=0; q<nl; q++) {
    if (in->large_area) {
      const double qe = work->quadratic_zero[(size_t)quadratic*nl + q];
      double partitions = 0.0;
      for (int left=0; left<2; left++) {
        const int source = in->field_source[a[1 - left]];
        for (int right=0; right<3; right++) {
          const int b0 = b[(right + 1)%3];
          const int b1 = b[(right + 2)%3];
          partitions += work->filters[(size_t)b[right]*nl + q]
            *covariance_power_value_3pcf(work,
              in->field_source[a[left]], in->field_source[b[right]], q, 0)
            *work->bspec[covariance_bspec_3pcf(work, b0, b1, source, q)];
        }
      }
      value += work->fourier_measure[q]*qe*partitions/in->area;
      continue;
    }
    for (int p=0; p<nl; p++) {
      double angle_sum = 0.0;
      for (int angle=0; angle<in->nphi; angle++) {
        const size_t g = covariance_geometry_3pcf(in, q, p, angle);
        const double qe = work->quadratic[
          (size_t)work->quadratic_index[quadratic]*ng + g];
        const double swapped = in->row_kind[quadratic] == 0
          ? work->filters[(size_t)a[0]*nl + p]
              *work->filters[(size_t)a[1]*nl + q] : qe;
        double partitions = 0.0;
        for (int left=0; left<2; left++) {
          const int source = in->field_source[a[1 - left]];
          const double kernel = left == 0 ? qe : swapped;
          for (int right=0; right<3; right++) {
            const int b0 = b[(right + 1)%3];
            const int b1 = b[(right + 2)%3];
            partitions += kernel*work->filters[(size_t)b[right]*nl + q]
              *covariance_power_value_3pcf(work,
                in->field_source[a[left]], in->field_source[b[right]], q, 0)
              *work->bspec[covariance_bspec_3pcf(work, b0, b1, source, p)];
          }
        }
        for (int remaining=0; remaining<3; remaining++) {
          const int b0 = b[(remaining + 1)%3];
          const int b1 = b[(remaining + 2)%3];
          const int sa = in->field_source[a[0]];
          const int sb = in->field_source[a[1]];
          const int sc = in->field_source[b[remaining]];
          const int t = work->triple_index[((size_t)sa*ns + sb)*ns + sc];
          partitions += qe*work->pair_zero[(size_t)b0*nf + b1]
            *work->closure_filters[(size_t)b[remaining]*ng + g]
            *work->bispectrum[(size_t)t*ng + g];
        }
        angle_sum += in->dphi[angle]*work->geometry_window[g]*partitions;
      }
      value += work->fourier_measure[q]*work->fourier_measure[p]
        *COV_I2PI*angle_sum;
    }
  }
  return value;
}

// All positional cubic partitions are retained even for repeated labels.
// Gaussian: six all-cross choices plus nine one-cross choices. BB: nine
// splits into two triples after removing the product of estimator means.
static void covariance_cubic_disconnected_3pcf(
  const covariance_work_3pcf* work, int i, int j,
  double* gaussian, double* bb
)
{
  const covariance_input_3pcf* in = work->in;
  const int nf = in->nfield;
  const int nr = in->nr;
  const int nl = in->nell;
  const int* a = in->row_fields + (size_t)i*3;
  const int* b = in->row_fields + (size_t)j*3;
  const int permutations[6][3] = {{0,1,2},{0,2,1},{1,0,2},
    {1,2,0},{2,0,1},{2,1,0}};
  *gaussian = 0.0;
  *bb = 0.0;
  for (int r=0; r<nr; r++) {
    double density = 0.0;
    for (int p=0; p<6; p++) {
      double product = 1.0;
      for (int leg=0; leg<3; leg++) {
        product *= work->pair[((size_t)a[leg]*nf
          + b[permutations[p][leg]])*nr + r];
      }
      density += product;
    }
    if (!in->large_area) {
      for (int p=0; p<3; p++) {
        const double left = work->pair_zero[(size_t)a[(p + 1)%3]*nf
          + a[(p + 2)%3]];
        for (int q=0; q<3; q++) {
          density += left*work->pair[((size_t)a[p]*nf + b[q])*nr + r]
            *work->pair_zero[(size_t)b[(q + 1)%3]*nf + b[(q + 2)%3]];
        }
      }
    }
    *gaussian += work->radial_measure[r]*density;
  }
  for (int p=0; p<3; p++) {
    const int a0 = a[(p + 1)%3];
    const int a1 = a[(p + 2)%3];
    for (int q=0; q<3; q++) {
      const int b0 = b[(q + 1)%3];
      const int b1 = b[(q + 2)%3];
      if (in->large_area) {
        for (int l=0; l<nl; l++) {
          *bb += work->fourier_measure[l]/in->area
            *work->filters[(size_t)a[p]*nl + l]
            *work->filters[(size_t)b[q]*nl + l]
            *work->bspec[covariance_bspec_3pcf(work, a0, a1,
              in->field_source[b[q]], l)]
            *work->bspec[covariance_bspec_3pcf(work, b0, b1,
              in->field_source[a[p]], l)];
        }
      } else {
        *bb += covariance_dot_3pcf(nr, work->radial_measure,
          work->breal + covariance_breal_3pcf(in, a0, a1, b[q], 0),
          work->breal + covariance_breal_3pcf(in, b0, b1, a[p], 0));
      }
    }
  }
}

static void covariance_release_3pcf(covariance_work_3pcf* work)
{
  free(work->lens);
  free(work->triples);
  free(work->triple_index);
  free(work->filters);
  free(work->closure_filters);
  free(work->power);
  free(work->bispectrum);
  free(work->bspec);
  free(work->breal);
  free(work->pair);
  free(work->pair_zero);
  free(work->radial_measure);
  free(work->fourier_measure);
  free(work->window_fourier);
  free(work->geometry_window);
  free(work->quadratic);
  free(work->quadratic_mixed);
  free(work->quadratic_index);
  free(work->quadratic_representative);
  free(work->pair_window);
  free(work->j0_radial);
  free(work->j0_fourier);
  free(work->j0_window);
  free(work->j2_profile);
  free(work->quadratic_zero);
  free(work->radial_volume);
  free(work->sigma2);
  free(work->abundance);
  free(work->bias);
  free(work->concentration);
  free(work->response_bias);
}

void covariance_physical_3pcf(
  const physics_model_3pcf* model,
  const covariance_input_3pcf* input,
  double* mean,
  double* terms
)
{
  covariance_require_3pcf(input->estimator == 0,
    "finite-triangle or catalogue-mask covariance is not this estimator");
  covariance_require_3pcf(input->ia_model == 0,
    "nonlinear IA connected covariance requires additional cumulants");
  covariance_require_3pcf(input->noise_mode == 0 || input->noise_mode == 1,
    "noise_mode must select finite-band (0) or uniform-area xi noise (1)");
  covariance_require_3pcf(input->noise_mode == 0
    || (input->large_area == 1 && input->theta_edges != NULL),
    "analytic xi noise requires large_area=1 and exact theta_edges");
  covariance_require_3pcf(input->center_window == 1
    || input->center_window == 2, "choose a circle or square center window");
  covariance_require_3pcf(input->nell > 0 && input->nwindow > 0
    && input->nphi > 0 && input->nr > 0 && input->nprofile >= 3
    && input->nk > 0 && input->nmass > 0 && input->nfield > 0
    && input->nrow > 0 && input->ntheta > 0 && input->area > 0.0,
    "invalid covariance dimensions or area");
  const int no = input->nrow;
  const int nl = input->nell;
  const size_t matrix = (size_t)no*no;
  memset(terms, 0, 6*matrix*sizeof(double));

  // Observable roles are checked before constructing any field offsets.
  // xi needs unfiltered shear legs; the aperture response needs R>0.
  for (int row=0; row<no; row++) {
    const int kind = input->row_kind[row];
    covariance_require_3pcf(kind >= 0 && kind <= 3, "invalid row kind");
    const int nleg = kind == 1 ? 3 : 2;
    for (int leg=0; leg<nleg; leg++) {
      const int field = input->row_fields[(size_t)row*3 + leg];
      covariance_require_3pcf(field >= 0 && field < input->nfield,
        "row field index out of range");
      covariance_require_3pcf(kind >= 2 ? input->field_radius[field] == 0.0
        : input->field_radius[field] > 0.0,
        "xi legs require R=0; aperture legs require positive R");
    }
    if (kind >= 2) {
      if (input->noise_mode == 1) {
        const double lower = input->theta_edges[(size_t)row*2];
        const double upper = input->theta_edges[(size_t)row*2 + 1];
        covariance_require_3pcf(isfinite(lower) && isfinite(upper)
          && lower > 0.0 && upper > lower,
          "analytic xi noise requires positive ordered bin edges");
        for (int t=0; t<input->ntheta; t++) {
          const double theta = input->theta[(size_t)row*input->ntheta + t];
          covariance_require_3pcf(theta >= lower && theta <= upper,
            "xi quadrature node lies outside declared analytic-noise bin");
        }
      }
      double weight = 0.0;
      for (int t=0; t<input->ntheta; t++) {
        const size_t index = (size_t)row*input->ntheta + t;
        covariance_require_3pcf(input->theta[index] > 0.0
          && input->theta_weight[index] >= 0.0,
          "xi separations must be positive and bin weights nonnegative");
        covariance_require_3pcf(input->profile_radius[input->nprofile - 1]
          >= input->radius[input->nr - 1] + 0.5*input->theta[index],
          "profile_radius must cover radius_max + theta_max/2");
        weight += input->theta_weight[index];
      }
      covariance_require_3pcf(fabs(weight - 1.0) < 1.0e-10,
        "xi bin weights must sum to one");
    }
  }
  covariance_work_3pcf work;
  memset(&work, 0, sizeof(work));
  work.model = model;
  work.in = input;
  covariance_prepare_3pcf(&work);

  // Means omit the known additive Gaussian-noise bias of quadratic rows.
  // Covariance is unchanged by subtracting that deterministic constant.
  #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
  for (int row=0; row<no; row++) {
    const int* fields = input->row_fields + (size_t)row*3;
    double value = 0.0;
    if (input->row_kind[row] == 1) {
      // Averaging the three choices of separated leg preserves a cubic
      // row's permutation symmetry under a finite numerical quadrature.
      for (int p=0; p<3; p++) {
        const int a = fields[(p + 1)%3];
        const int b = fields[(p + 2)%3];
        const int c = fields[p];
        for (int l=0; l<nl; l++) {
          value += work.fourier_measure[l]*work.filters[(size_t)c*nl + l]
            *work.bspec[covariance_bspec_3pcf(&work, a, b,
              input->field_source[c], l)]/3.0;
        }
      }
    } else {
      const int a = input->field_source[fields[0]];
      const int b = input->field_source[fields[1]];
      for (int l=0; l<nl; l++) {
        value += work.fourier_measure[l]
          *work.quadratic_zero[(size_t)row*nl + l]
          *work.power[((size_t)a*work.ns + b)*nl + l];
      }
    }
    mean[row] = value;
  }

  #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
  for (int i=0; i<no; i++) {
    for (int j=i; j<no; j++) {
      const size_t cell = (size_t)i*no + j;
      if (input->row_kind[i] == 1 && input->row_kind[j] == 1) {
        covariance_cubic_disconnected_3pcf(&work, i, j,
          terms + cell, terms + matrix + cell);
      } else if (input->row_kind[i] != 1 && input->row_kind[j] != 1) {
        terms[cell] = covariance_gaussian_two_3pcf(&work, i, j);
      } else {
        const int quadratic = input->row_kind[i] == 1 ? j : i;
        const int cubic = input->row_kind[i] == 1 ? i : j;
        terms[3*matrix + cell] = covariance_pb_3pcf(&work, quadratic, cubic);
      }
    }
  }
  covariance_halo_terms_3pcf(&work, terms);
  for (int term=0; term<6; term++) {
    for (int i=0; i<no; i++) {
      for (int j=i; j<no; j++) {
        const double value = terms[(size_t)term*matrix + (size_t)i*no + j];
        covariance_require_3pcf(isfinite(value),
          "nonfinite covariance term; inspect physical inputs and grids");
        terms[(size_t)term*matrix + (size_t)j*no + i] = value;
      }
    }
  }
  covariance_release_3pcf(&work);
}

