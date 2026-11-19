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
#include "cosmolike/basics.h"
#include "log.c/src/log.h"

// ---------------------------------------------------------------------------
// covariance_physics_3pcf.c: the physical covariance of aperture-mass
// statistics and shear correlation functions measured on the same
// lensing fields.
//
// The requested observables arrive as rows. A quadratic row is second
// order in the shear field: the aperture-mass second moment Map^2 at one
// filter radius, or a binned shear correlation function xi+ / xi-, whose
// estimator averages the product of two shears measured at points
// displaced by -theta/2 and +theta/2 from a common center. A cubic row
// is third order in the shear field: the aperture-mass third moment
// Map^3. For every pair of rows the engine fills six covariance terms -
// Gaussian, BB, P*T, P*B, connected one-halo, and the halo-abundance
// response - following the cumulant partitions of the aperture-statistics
// covariance in Linke et al. (2023), arXiv:2212.04485, with the xi rows
// entering through their actual separation-bin weights as in Wielders et
// al. (2025), arXiv:2509.20443. This implementation is written from
// those equations and the flat-sky Fourier rules; it does not translate
// the GPL threepoint implementation.
//
// Notation shared by the helpers and assemblers below:
//
//   U_R(ell)    the compensated aperture filter of radius R
//               (covariance_filter_3pcf).
//   Q(p,q,phi)  the quadratic kernel of one row: the weight its
//               estimator gives a pair of shear Fourier modes with
//               magnitudes p and q and relative angle phi
//               (covariance_quadratic_3pcf).
//   K_ab(r)     the two-point function of the two filtered fields a and
//               b, shape noise included, transformed to the separation r
//               of two estimator centers; K_ab(0) is its value at
//               coincident centers (work->pair, work->pair_zero).
//   S_ab|c(q)   the projected bispectrum of fields a, b and source bin c
//               with the filters of a and b attached and their angles
//               integrated out, leaving the magnitude q of the third leg
//               open (work->bspec). B_ab|c(r) is its transform to the
//               center separation r once the third filter is attached
//               (work->breal).
//   closure     the third side -(q+p) that closes a Fourier triangle
//               built from two vectors of magnitudes q and p
//               (covariance_closure_3pcf returns its magnitude |q+p|).
//
// The field model is pure gravitational lensing with independent
// Gaussian E/B shape noise. The connected four-, five- and six-field
// cumulants use a Poisson one-halo model. The extra long-mode term is
// the halo-abundance response, not a complete growth, dilation or tidal
// response. Finite center windows keep the full filters inside each
// aperture. A galaxy mask inside a filter, or a finite triangle
// selection, defines a different estimator with its own covariance.
//
// No process-global state or lazy initialization occurs. The call owns
// its temporary physical tables, prepares them before any parallel
// reader starts, and releases them before returning. Each parallel task
// owns complete rows or upper-triangle cells. Quadrature sums keep
// their input order, so results do not depend on the thread count.
// ---------------------------------------------------------------------------

// Pi to double precision.
#define COV_PI 3.141592653589793238462643383279502884
// 1/(2*pi). The planar Fourier measure d^2ell/(2*pi)^2 splits into
// ell dell/(2*pi) times dphi/(2*pi); each split factor below carries one
// COV_I2PI.
#define COV_I2PI 0.159154943091895335768883763372514362

// A v2d packs two doubles into one 128-bit register (SSE2 through SIMDe,
// so the same source also builds on NEON).
typedef simde__m128d v2d;



// ============================================================================
// [SECTION] UTILITIES: CHECKS, ALLOCATION AND THE ORDERED SIMD DOT
// ============================================================================

// Multiply two array extents only after checking that their product
// fits the int element count the core allocators calloc1d and
// malloc1d_int take. Every multidimensional extent below passes through
// this helper first, so no allocation can silently truncate its count,
// and the flattened offsets inside the worker loops stay inside their
// allocations.
static size_t covariance_product_3pcf(
  size_t first, // first array extent
  size_t second // second array extent; must be positive
)
{
  if (!(second > 0
        && first <= (size_t)INT_MAX/second)) {
    log_fatal("physical 3pcf covariance: workspace dimensions overflow the "
      "core allocator count");
    exit(1);
  }

  return first*second;
}


// Allocation convention of this file: every table allocates zero
// filled, through calloc1d, through malloc1d_int with an explicit
// memset, or through calloc, so the += accumulations below need no
// separate clearing pass.


// Weighted dot product sum_i w_i*(a_i*b_i), evaluated with SIMD on two
// adjacent quadrature nodes at a time. The lanes only evaluate; they do
// not reorder the accumulation: after each two-node step the low lane's
// value joins the running sum first and the high lane's value second,
// exactly the order a scalar loop would use. Every node forms w*(a*b)
// with the product a*b rounded before the weight multiplies it - the
// scalar tail included - so moving a node between lanes cannot change a
// single bit of the result.
//
// Parameters:
//   count   - number of quadrature nodes
//   weights - [count] integration weights
//   first   - [count] first integrand factor
//   second  - [count] second integrand factor
static inline double covariance_dot_3pcf(
  int count, // number of quadrature nodes
  const double* weights, // [count] integration weights
  const double* first, // [count] first integrand factor
  const double* second // [count] second integrand factor
)
{
  double sum = 0.0;
  int node = 0;

  // scalar: for every pair of adjacent nodes,
  //   sum += weights[node]*(first[node]*second[node]);
  //   sum += weights[node + 1]*(first[node + 1]*second[node + 1]);
  // The vector step forms both weighted products at once - lane 0 holds
  // node, lane 1 holds node + 1 - and then adds them in that same order.
  for (; node + 1 < count; node += 2) {
    // Load weights[node] and weights[node + 1]; loadu reads two valid
    // array elements and needs no aligned address.
    const v2d w = simde_mm_loadu_pd(weights + node);
    // Load the first function at the same two nodes.
    const v2d a = simde_mm_loadu_pd(first + node);
    // Load the second function at the same two nodes.
    const v2d b = simde_mm_loadu_pd(second + node);
    // Multiply lane by lane: lane 0 holds first[node]*second[node] and
    // lane 1 holds first[node + 1]*second[node + 1], each rounded once.
    const v2d ab = simde_mm_mul_pd(a, b);
    // Apply each node's integration weight to its own lane.
    const v2d weighted = simde_mm_mul_pd(w, ab);
    double value[2];
    // Copy both lanes into a two-element buffer on the stack.
    simde_mm_storeu_pd(value, weighted);
    // Add the low lane (node) before the high lane (node + 1): the
    // scalar accumulation order.
    sum += value[0];
    sum += value[1];
  }

  if (node < count) {
    // The last node of an odd count runs in scalar code. The strict
    // compile flags still allow fused multiply-add contraction, which
    // would skip the separate rounding of first*second that each SIMD
    // lane performs. Writing the product through a volatile forces that
    // rounding before the final addition.
    volatile double weighted = weights[node]*(first[node]*second[node]);
    sum += weighted;
  }

  return sum;
}


// U_R(ell): the compensated aperture filter in Fourier space,
//
//   U_R(ell) = (ell*R)^2/2 * exp(-(ell*R)^2/2),
//
// the Crittenden-type filter (Crittenden et al.,
// arXiv:astro-ph/0012336) that the aperture-mass covariances of
// Linke et al. (2023), arXiv:2212.04485, use. Compensated means the
// real-space filter integrates to zero, so U_R(0) = 0 and a uniform
// mass sheet drops out of every aperture statistic. The convention
// R = 0 returns 1: an xi shear leg carries no aperture filter, and
// evaluating U with R = 0 lets the same kernel tables serve filtered
// and unfiltered legs alike.
//
// Parameters:
//   ell    - angular multipole
//   radius - aperture radius R in radians; 0 marks an unfiltered leg
static inline double covariance_filter_3pcf(
  double ell, // angular multipole
  double radius // aperture radius R in radians; 0: unfiltered leg
)
{
  if (radius == 0.0) {
    return 1.0;
  }
  const double x = ell*radius;
  const double square = x*x;
  return 0.5*square*exp(-0.5*square);
}



// ============================================================================
// [SECTION] HALO INGREDIENTS AND THE SHARED HALO TABLES
// ============================================================================

// Spherical top-hat window W(x) with x = kR, and its logarithmic
// derivative dW/dln R. Because x = kR at fixed wavenumber k, the
// derivative with respect to ln R equals the derivative with respect to
// ln x, which is what the second output evaluates:
//
//   W(x) = 3 (sin x - x cos x)/x^3,   dW/dln x = 3 sin(x)/x - 3 W(x).
//
// Accuracy of the small-x branch: near x = 0 the closed form subtracts
// two almost equal sine terms and loses precision. Below |x| = 0.01 a
// Taylor series takes over; its first omitted terms are of order x^8,
// about 1e-16 at the threshold, so the switch sits below double
// rounding and both outputs stay smooth across it.
//
// Parameters:
//   x          - dimensionless argument kR
//   window     - out: W(x)
//   derivative - out: dW/dln x = dW/dln R at fixed k
static inline void covariance_tophat_3pcf(
  double x, // dimensionless argument kR
  double* window, // out: W(x)
  double* derivative // out: dW/dln x = dW/dln R at fixed k
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


// Fourier transform u(k) of an NFW halo density profile truncated at
// the comoving radius r200. The NFW profile falls as
// 1/[(r/r_s)(1 + r/r_s)^2], where the scale radius r_s = r200/c marks
// the turnover between the inner 1/r and outer 1/r^3 slopes and c is
// the concentration. The transform divides by the enclosed mass, so
// u(0) = 1 exactly. In the scaled variable y = r/r_s the transform is
//
//   u(k) = int_0^c dy y/(1+y)^2 sinc(k r_s y) / [ln(1+c) - c/(1+c)],
//
// and that integral has a closed form in the sine and cosine integrals
// Si(t) = int_0^t sin(s)/s ds and Ci(t) = -int_t^inf cos(s)/s ds, which
// the general branch below evaluates with x = k*r_s.
//
// Accuracy of the small-argument branch: for k*r200 below 1e-4 the Ci
// difference subtracts two nearly equal values close to the common
// logarithmic divergence of Ci at zero. The branch instead expands
// u(k) = 1 - k^2 <r^2>/6, with <r^2> the profile's mass-weighted second
// radial moment; the first omitted term is of order (k*r200)^4, far
// below double rounding at the threshold.
//
// Parameters:
//   k             - comoving wavenumber, h/Mpc
//   r200          - comoving truncation radius, Mpc/h
//   concentration - NFW concentration c = r200/r_s
static inline double covariance_nfw_3pcf(
  double k, // comoving wavenumber, h/Mpc
  double r200, // comoving truncation radius, Mpc/h
  double concentration // NFW concentration c = r200/r_s
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


// ---------------------------------------------------------------------------
// covariance_halos_3pcf: tabulate, for every requested (redshift, mass)
// node, the halo ingredients that the connected covariance terms read -
// the variance sigma^2(M,z) of the smoothed linear density field, the
// halo mass function dn/dM, the linear halo bias b_1, the NFW
// concentration c - and, when nk_profile > 0, the normalized NFW
// transform u(k) at the requested profile wavenumbers.
//
// Power input. The model's linear provider supplies P_linear(k, z=0)
// times the squared growth factor D(z)^2. That is one scale-independent
// growth law; it does not supply the separate cold matter+baryon
// spectrum that a massive-neutrino halo prescription would need.
//
// Variance. The smoothing radius R follows from M = (4 pi/3) rho_m R^3,
// and the top-hat variance is
//
//   sigma^2(M,z) = int k^2 dk P(k,z) W^2(kR) / (2 pi^2).
//
// Because R is proportional to M^(1/3), the ln M derivative of sigma^2
// is the ln R derivative times 1/3, which the quadrature evaluates with
// the factor 2/3 on W dW/dln R (the 2 from differentiating W^2).
//
// Mass function and bias. The Sheth-Tormen multiplicity f(nu) of Sheth
// and Tormen (1999), arXiv:astro-ph/9901122, with peak height
// nu = delta_c/sigma, converts the variance slope into the abundance
//
//   dn/dM = -(rho_m/M^2) f(nu) (d ln sigma^2 / d ln M)/2,
//
// and the peak-background split of the same multiplicity gives the
// linear bias b_1(nu): the response of halo abundance to a long
// background density mode.
//
// Concentration. The default follows the full-sample mean-density fit
// of Duffy et al. (2008), arXiv:0804.2486, Table 1, calibrated at
// z = 0..2 for M200m halos:
//
//   c = 10.14 (M / 2e12 Msun/h)^-0.081 (1+z)^-1.01.
//
// Its coefficients and pivot are explicit inputs, because extrapolating
// the fit or substituting another one is a physical assumption the
// caller must own. r200 is comoving and uses today's mean density.
//
// Parameters and units: see the declaration in covariance_physics_3pcf.h.
// ---------------------------------------------------------------------------
void covariance_halos_3pcf(
  const physics_model_3pcf* model, // linear power and cosmology
  const covariance_input_3pcf* input, // k/dk, M/dM and the halo prescription
  int nz, // redshift count
  const double* z, // redshifts [nz]
  int nk_profile, // profile wavenumber count
  const double* k_profile, // [nk_profile], nonnegative h/Mpc
  double* sigma2, // variance [nz,nmass]
  double* dndm, // abundance [(Mpc/h)^-3 (Msun/h)^-1]
  double* bias, // dimensionless linear halo bias
  double* concentration, // M200m concentration [nz,nmass]
  double* profile // normalized u [nz,nmass,nk_profile]
)
{
  // 2.77536627e11 is the critical density rho_crit/h^2 in units of
  // (Msun/h)/(Mpc/h)^3; multiplied by Omega_m it is today's comoving
  // mean matter density, the rho_m of the M200m halo definition.
  const double rho = 2.77536627e11*physics_omega_m_3pcf(model);
  const int nm = input->nmass;

  if (!(nz > 0
        && nm > 0
        && input->nk > 0
        && nk_profile >= 0)) {
    log_fatal("physical 3pcf covariance: invalid halo-table dimensions");
    exit(1);

  }
  const size_t halo_count = covariance_product_3pcf(nz, nm);
  if (nk_profile > 0) {
    covariance_product_3pcf(halo_count, nk_profile);
  }
  const size_t power_count = covariance_product_3pcf(nz, input->nk);
  double* power = calloc1d((int)power_count);

  // Tabulate the linear power once per (redshift, wavenumber) node. The
  // variance integral reads the same power row for every mass at one
  // redshift, so one worker fills one redshift's row and all masses
  // share it. Mode 0 asks the provider for the linear power
  // P_linear(k) D^2(z): halo formation statistics are set by the linear
  // density field, not by the nonlinear lensing spectrum.
  #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
  for (int iz=0; iz<nz; iz++) {
    for (int ik=0; ik<input->nk; ik++) {
      power[(size_t)iz*input->nk + ik] =
        physics_power_3pcf(model, input->k[ik], z[iz], 0);
    }
  }

  // Each (redshift, mass) pair owns its variance, abundance and profile
  // row, so the collapsed loop parallelizes with no shared writes.
  /* PHYSICAL DERIVATION & LOGIC FLOW
     1. Smoothing radius from the mass: R = (3M/(4 pi rho_m))^(1/3).
     2. Top-hat variance and its logarithmic slope by quadrature:
        sigma^2 = int k^2 dk P W^2(kR)/(2 pi^2) and
        d sigma^2/d ln M = (2/3) int k^2 dk P W (dW/dln R)/(2 pi^2).
     3. Squared peak height nu^2 = delta_c^2/sigma^2, with delta_c
        optionally rescaled by the Omega_m(z) collapse correction.
     4. Sheth-Tormen multiplicity f(nu) and the mass function
        dn/dM = -(rho_m/M^2) f(nu) (d ln sigma^2/d ln M)/2.
     5. Peak-background-split bias b_1(nu) of the same multiplicity.
     6. Duffy concentration c(M,z) and, if requested, the truncated-NFW
        transform u(k) at the profile wavenumbers. */
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

      if (!(isfinite(variance)
            && variance > 0.0
            && isfinite(slope)
            && slope < 0.0)) {
        log_fatal("physical 3pcf covariance: halo variance/slope is "
          "invalid; refine the k interval and rule");
        exit(1);
      }

      // collapse_model = 1 adds the weak Omega_m(z) dependence of the
      // spherical-collapse threshold delta_c; model 0 keeps it constant.
      double collapse = input->delta_c;
      if (input->collapse_model == 1) {
        collapse *= 1.0 + 0.0123*log10(
          physics_omega_m_z_3pcf(model, z[iz]));
      }
      // peak holds the squared peak height nu^2 = delta_c^2/sigma^2, and
      // apeak holds the Sheth-Tormen rescaled a*nu^2; both variables
      // keep their short names because every formula below uses the
      // squared combination.
      const double peak = collapse*collapse/variance;
      const double apeak = input->st_a*peak;
      // Sheth-Tormen multiplicity (arXiv:astro-ph/9901122):
      //   f(nu) = A sqrt(2 a nu^2/pi) [1 + (a nu^2)^-p] exp(-a nu^2/2).
      const double multiplicity = input->st_norm*sqrt(2.0*apeak/COV_PI)
        *(1.0 + pow(apeak, -input->st_p))*exp(-0.5*apeak);
      // Duffy concentration c = A (M/M_pivot)^B (1+z)^C with the
      // explicit input coefficients (see the function header).
      const double c = input->concentration_a
        *pow(mass/input->concentration_pivot, input->concentration_b)
        *pow(1.0 + z[iz], input->concentration_c);

      sigma2[index] = variance;
      dndm[index] = -rho/(mass*mass)*slope/(2.0*variance)*multiplicity;
      // Peak-background-split bias of the same multiplicity
      // (arXiv:astro-ph/9901122, with a nu^2 written as apeak):
      //   b_1 = 1 + [a nu^2 - 1 + 2p/(1 + (a nu^2)^p)] / delta_c.
      bias[index] = 1.0 + (apeak - 1.0 + 2.0*input->st_p
        /(1.0 + pow(apeak, input->st_p)))/collapse;
      concentration[index] = c;

      // radius encloses the mean density once, r200 encloses it 200
      // times: M = (4 pi/3) 200 rho_m r200^3, so r200 = radius/200^(1/3).
      const double r200 = radius/cbrt(200.0);
      for (int ik=0; ik<nk_profile; ik++) {
        profile[index*nk_profile + ik] =
          covariance_nfw_3pcf(k_profile[ik], r200, c);
      }
    }
  }

  free(power);
}



// ============================================================================
// [SECTION] WORKSPACE, GEOMETRY KERNELS AND SHARED PREPARATION
// ============================================================================

// Workspace of one covariance call: the tables covariance_prepare_3pcf
// and covariance_halo_terms_3pcf build before the assemblers read them.
// Pointers an option does not need stay NULL from the initial memset,
// and free(NULL) is harmless in covariance_release_3pcf. Dimension
// shorthand in the member comments: ns source bins, nz redshift nodes,
// nf = in->nfield field labels, nl = in->nell Fourier nodes, nr =
// in->nr center-separation nodes, ng = nl*nl*nphi Fourier triangles.
// The physical symbols U_R, Q, K_ab, S_ab|c and B_ab|c are the ones the
// file head defines.
typedef struct {
  const physics_model_3pcf* model;    // borrowed matter and source model
  const covariance_input_3pcf* in;    // borrowed grids and row definitions
  int ns;                             // source bins in the model
  int nz;                             // redshift nodes of the radial grid
  int nt;                             // source triples ns(ns+1)(ns+2)/6
  size_t ngeometry;                   // Fourier triangles ng = nl*nl*nphi
  int has_cubic;                      // 1 when any requested row is Map^3
  int nquadratic;                     // distinct quadratic kernel shapes
  int* quadratic_index;               // [nrow] shape of each row; -1 cubic
  int* quadratic_representative;      // [nrow] row defining each shape
  const double* z;                    // [nz] borrowed redshifts
  const double* chi;                  // [nz] borrowed distances, Mpc/h
  const double* dchi;                 // [nz] borrowed dchi weights, Mpc/h
  double* lens;                       // [ns,nz] weight W = chi*(1+z)*g
  int* triples;                       // [nt,3] nondecreasing source triples
  int* triple_index;                  // [ns,ns,ns] ordered -> triple index
  double* filters;                    // [nf,nl] U_R(ell) of every field
  double* closure_filters;            // [nf,ng] U_R(|q+p|) per triangle
  double* power;                      // [ns,ns,nl] Limber C(ell), no noise
  double* bispectrum;                 // [nt,ng] projected B(q,p,phi)
  double* bspec;                      // [nf,nf,ns,nl] S_ab|c(q)
  double* breal;                      // [nf,nf,nf,nr] B_ab|c(r), finite win.
  double* pair;                       // [nf,nf,nr] K_ab(r), noise included
  double* pair_zero;                  // [nf,nf] K_ab(0), coincident centers
  double* radial_measure;             // [nr] 2*pi*r*dr times pair density
  double* fourier_measure;            // [nl] ell*dell/(2*pi)
  double* window_fourier;             // [nwindow] G(q) = |w~(q)|^2/A^2
  double* geometry_window;            // [ng] G(|q+p|), finite window only
  double* quadratic;                  // [nquadratic,ng] EE part of Q
  double* quadratic_mixed;            // [nquadratic,ng] EB part of Q
  double* pair_window;                // [nf,nf,nl] windowed K + ell measure
  double* j0_radial;                  // [nr,nl] J0(ell*r), ell contiguous
  double* j0_fourier;                 // [nl,nr] J0(ell*r), r contiguous
  double* j0_window;                  // [nwindow,nr] J0(q*r), survey nodes
  double* j2_profile;                 // [nprofile,nl] J2(ell*u) shear kernel
  double* quadratic_zero;             // [nrow,nl] Q at p = q, phi = pi
  double* radial_volume;              // [nr] bare 2*pi*r*dr, no window
  double* sigma2;                     // [nz,nmass] linear variance sigma^2
  double* abundance;                  // [nz,nmass] mass function dn/dM
  double* bias;                       // [nz,nmass] linear halo bias b_1
  double* concentration;              // [nz,nmass] NFW concentration
  double* responses;                  // never allocated here; stays NULL
  double* response_zero;              // never allocated here; stays NULL
  double* response_bias;              // [nz,nrow] R_I = int dM dn/dM b_1 H_I
} covariance_work_3pcf;


// Survey-window weight for a pair of estimator centers, normalized by
// the survey area A. The argument value is a center separation r in
// real space (fourier = 0) or a Fourier magnitude q (fourier = 1).
//
// Real space. The covariance averages over positions of two centers a
// distance r apart, both inside the survey, so the weight is the
// density of such center pairs. For a circular survey of radius
// S = sqrt(A/pi) that density is the overlap area of two circles of
// radius S whose centers sit r apart, divided by A^2. With x = r/(2S)
// the overlap area is 2 S^2 (acos x - x sqrt(1 - x^2)); dividing by
// A^2 = (pi S^2) A gives the returned
// (2/pi)(acos x - x sqrt(1 - x^2))/A, which vanishes for r >= 2S.
// For a square of side L = sqrt(A) the pair density factorizes into
// one triangle overlap per axis, (1 - |dx|/L)(1 - |dy|/L)/A, and it
// depends on the direction of the separation; the square branch
// therefore averages the offsets (r cos phi, r sin phi) over the same
// relative-angle quadrature phi/dphi that the Fourier triangles use.
//
// Fourier space. G(q) = |w~(q)|^2/A^2, with w~ the window transform.
// The circle gives the squared Airy amplitude [2 J1(qS)/(qS)]^2; the
// square gives one squared sinc factor per axis, again averaged over
// the orientation quadrature.
//
// Callers apply the large-area approximation themselves when they want
// it; the long-mode variance always keeps the finite survey shape the
// input declares, through this function.
//
// Parameters:
//   in      - window shape (circle/square), area and phi quadrature
//   value   - center separation r, or Fourier magnitude q
//   fourier - 0 returns the pair density; 1 returns G(q)
static inline double covariance_window_3pcf(
  const covariance_input_3pcf* in, // window shape, area and phi quadrature
  double value, // center separation r, or Fourier magnitude q
  int fourier // 0 returns the pair density; 1 returns G(q)
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


// Flattened index of the Fourier triangle (q, p, phi) in the [nl,nl,nphi]
// geometry tables: the angle runs fastest, then the second magnitude p.
static inline size_t covariance_geometry_3pcf(
  const covariance_input_3pcf* in, // supplies the nell and nphi extents
  int q, // first Fourier-magnitude node, in [0,nell)
  int p, // second Fourier-magnitude node, in [0,nell)
  int angle // relative-angle node, in [0,nphi)
)
{
  return ((size_t)q*in->nell + p)*in->nphi + angle;
}


// |q+p|: the magnitude of the closure side that completes the Fourier
// triangle built from magnitudes q and p with relative angle phi. In
// floating point sin(pi) is not exactly zero, so the exactly specified
// antiparallel equal-length case (q == p with phi == pi) is recognized
// first and returns an exact zero. Only that one exact input maps to
// zero; no neighborhood of nearly degenerate physical triangles is
// removed.
static inline double covariance_closure_3pcf(
  double q, // first Fourier magnitude
  double p, // second Fourier magnitude
  double phi // relative angle between the two vectors, radians
)
{
  if (q == p && phi == COV_PI) {
    return 0.0;
  }
  return hypot(q + p*cos(phi), p*sin(phi));
}


// Flattened index of S_ab|c(q) in the [nf,nf,ns,nl] bspec table: fields
// a and b carry the attached filters, source is the separated leg c.
static inline size_t covariance_bspec_3pcf(
  const covariance_work_3pcf* work, // supplies the nfield, ns and nell extents
  int a, // field label of the first attached filter
  int b, // field label of the second attached filter
  int source, // source bin of the separated leg c
  int ell // Fourier-magnitude node q, in [0,nell)
)
{
  const covariance_input_3pcf* in = work->in;
  return (((size_t)a*in->nfield + b)*work->ns + source)*in->nell + ell;
}


// Flattened index of B_ab|c(r) in the [nf,nf,nf,nr] breal table; here
// the separated leg c is a field, because its filter is attached too.
static inline size_t covariance_breal_3pcf(
  const covariance_input_3pcf* in, // supplies the nfield and nr extents
  int a, // field label of the first attached filter
  int b, // field label of the second attached filter
  int c, // field label of the separated leg
  int radius // center-separation node r, in [0,nr)
)
{
  return (((size_t)a*in->nfield + b)*in->nfield + c)*in->nr + radius;
}


// One source-pair spectrum at Fourier node ell, white shape noise
// included. mode = 0 returns the E-mode spectrum C^ab(ell) + N^ab: the
// lensing signal is pure E, so the E spectrum carries signal plus
// noise. mode = 1 returns the B-mode spectrum, which is the noise N^ab
// alone, because the pure-lensing field model produces no B-mode
// signal.
static inline double covariance_power_value_3pcf(
  const covariance_work_3pcf* work, // prepared Limber spectra and shape noise
  int a, // first source bin
  int b, // second source bin
  int ell, // Fourier node, in [0,nell)
  int mode // 0 E-mode spectrum plus noise; 1 B-mode (noise)
)
{
  const size_t pair = (size_t)a*work->ns + b;
  const double noise = work->in->noise[pair];
  return mode == 0 ? work->power[pair*work->in->nell + ell] + noise : noise;
}


// Q(p,q,phi) of one quadratic row: the weight its estimator gives a
// pair of shear Fourier modes with magnitudes p and q and relative
// angle phi (the code calls it angle below).
//
// A Map^2 row multiplies its two aperture filters, U_R1(p) U_R2(q),
// and mixes no E and B. An xi row averages the product of two shears
// measured at x - theta/2 and x + theta/2 around the center x; in
// Fourier space the two modes then meet at half their difference
// vector, and integrating the estimator over orientations turns that
// offset into J0(|p-q| theta/2) for xi+ and J4 for xi-, averaged over
// the row's separation bin with the row's own weights.
//
// E/B outputs. A pure E-mode Fourier shear carries the phase
// exp(2i phi_k) of its wavevector, and a B-mode shear carries an extra
// factor i. The real part of the product of the two shear phases gives
// a cosine factor when both modes have the same parity and a sine
// factor when one is E and one is B: *ee multiplies two E spectra (or
// two B spectra) in the Gaussian assembler, and *eb multiplies one E
// with one B spectrum. At antiparallel vectors q = -p the sine factor
// vanishes, the mixed kernel is zero, and *ee reduces to the usual
// binned J0/J4 kernel.
//
// Parameters:
//   in    - row definitions, field radii and separation bins
//   row   - the quadratic row (Map^2 or xi)
//   p, q  - Fourier magnitudes of the row's first and second leg
//   angle - relative angle phi between the two Fourier vectors
//   ee    - out: kernel for equal-parity (EE or BB) spectra
//   eb    - out: kernel for mixed E*B spectra
static inline void covariance_quadratic_3pcf(
  const covariance_input_3pcf* in, // rows, field radii and separation bins
  int row, // the quadratic row (Map^2 or xi)
  double p, // Fourier magnitude of the row's first leg
  double q, // Fourier magnitude of the row's second leg
  double angle, // relative angle phi between the two vectors
  double* ee, // out: kernel for equal-parity (EE or BB) spectra
  double* eb // out: kernel for mixed E*B spectra
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


// Decide whether two quadratic rows share one kernel shape. The kernel
// is pure Fourier geometry, so source-bin labels play no part. Two xi
// rows share a kernel only when their kind, separation nodes and
// weights agree exactly. Two Map^2 rows compare their radii in leg
// order, because an unequal-radius Q(p,q) changes when its Fourier
// legs are exchanged, even though the full estimator commutes.
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


// Build every geometric and spectral table the assemblers read: the
// source projection weights, the aperture filters and quadratic
// kernels, the Limber spectra, and - when a cubic row is requested -
// the shared bispectrum tables. The physical matter bispectrum is
// evaluated once per (z, q, p, phi) node, before any source weight is
// attached. Each projected spectrum and bispectrum gets its own full
// Limber integral: the product terms never share a line-of-sight
// distance between their two factors.
static void covariance_prepare_3pcf(
  covariance_work_3pcf* work // holds the input; receives every table
)
{
  const covariance_input_3pcf* in = work->in;
  const double* lens;
  const double* shape;
  physics_radial_3pcf(work->model, &work->ns, &work->nz, &work->z,
    &work->chi, &work->dchi, &lens, &shape);

  const int ns = work->ns;   // source bins
  const int nz = work->nz;   // redshift nodes
  const int nf = in->nfield; // field labels
  const int nl = in->nell;   // Fourier nodes
  const int nr = in->nr;     // center-separation nodes

  const size_t source_pairs = covariance_product_3pcf(ns, ns);
  const size_t field_pairs = covariance_product_3pcf(nf, nf);
  const size_t source_triples = covariance_product_3pcf(
    covariance_product_3pcf(ns, (size_t)ns + 1), (size_t)ns + 2)/6;

  if (!(source_triples <= INT_MAX)) {
    log_fatal("physical 3pcf covariance: source-triplet count exceeds the "
      "integer index range");
    exit(1);
  }

  work->nt = (int)source_triples;
  work->ngeometry = covariance_product_3pcf(
    covariance_product_3pcf(nl, nl), in->nphi);
  const size_t ng = work->ngeometry; // Fourier triangles (q, p, phi)

  for (int row=0; row<in->nrow; row++) {
    work->has_cubic = work->has_cubic || in->row_kind[row] == 1;
  }

  work->lens = calloc1d((int)covariance_product_3pcf(ns, nz));
  work->triples = malloc1d_int((int)covariance_product_3pcf(work->nt, 3));
  memset(work->triples, 0, sizeof(int)*(size_t)work->nt*3);
  work->triple_index = malloc1d_int(
    (int)covariance_product_3pcf(source_pairs, ns));
  memset(work->triple_index, 0, sizeof(int)*source_pairs*ns);
  work->filters = calloc1d((int)covariance_product_3pcf(nf, nl));
  work->power = calloc1d((int)covariance_product_3pcf(source_pairs, nl));

  // A purely quadratic request involves no bispectrum, no S table and
  // no three-field correlation. Such a request keeps its Gaussian and
  // connected halo terms but skips every matter-triangle and
  // aperture-partition table, which would otherwise dominate memory.
  if (work->has_cubic) {
    work->closure_filters = calloc1d((int)covariance_product_3pcf(nf, ng));
    work->bispectrum = calloc1d((int)covariance_product_3pcf(work->nt, ng));
    work->bspec = calloc1d((int)covariance_product_3pcf(field_pairs,
      covariance_product_3pcf(ns, nl)));
    if (!in->large_area) {
      work->breal = calloc1d((int)covariance_product_3pcf(field_pairs,
        covariance_product_3pcf(nf, nr)));
    }
    work->pair = calloc1d((int)covariance_product_3pcf(field_pairs, nr));
    work->pair_zero = calloc1d((int)field_pairs);
    work->pair_window = calloc1d((int)covariance_product_3pcf(field_pairs, nl));
  }
  work->radial_measure = calloc1d(nr);
  work->fourier_measure = calloc1d(nl);
  work->window_fourier = calloc1d(in->nwindow);

  // The large-area limit needs the quadratic kernel only at opposite
  // equal Fourier vectors, so the small [nrow, nl] quadratic_zero table
  // below suffices for it. The full [shape, ng] kernel tables here are
  // built only for a finite window, where every triangle contributes.
  if (!in->large_area) {
    work->quadratic_index = malloc1d_int(in->nrow);
    memset(work->quadratic_index, 0, sizeof(int)*(size_t)in->nrow);
    work->quadratic_representative = malloc1d_int(in->nrow);
    memset(work->quadratic_representative, 0, sizeof(int)*(size_t)in->nrow);
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
    work->geometry_window = calloc1d((int)ng);
    if (work->nquadratic > 0) {
      const size_t quadratic_count = covariance_product_3pcf(
        work->nquadratic, ng);
      work->quadratic = calloc1d((int)quadratic_count);
      work->quadratic_mixed = calloc1d((int)quadratic_count);
    }
  }

  const size_t radial_fourier = covariance_product_3pcf(nr, nl);
  if (work->has_cubic || !in->large_area) {
    work->j0_radial = calloc1d((int)radial_fourier);
    work->j0_fourier = calloc1d((int)radial_fourier);
  }
  if (!in->large_area) {
    work->j0_window = calloc1d((int)covariance_product_3pcf(in->nwindow, nr));
    work->j2_profile = calloc1d((int)covariance_product_3pcf(in->nprofile, nl));
  }
  work->quadratic_zero = calloc1d((int)covariance_product_3pcf(in->nrow, nl));
  work->radial_volume = calloc1d(nr);

  // The Bessel factors depend on geometry alone, not on halo mass or
  // cosmology, so one pass tabulates them for the whole call. J0 is
  // stored in both index orders so every later transform reads
  // contiguous nodes; duplicating this small matrix is cheaper than a
  // strided read in either direction.
  for (int r=0; r<nr; r++) {
    work->radial_volume[r] = 2.0*COV_PI*in->radius[r]*in->dr[r];
    if (work->has_cubic || !in->large_area) {
      for (int l=0; l<nl; l++) {
        const double value = gsl_sf_bessel_J0(in->ell[l]*in->radius[r]);
        work->j0_radial[(size_t)r*nl + l] = value;
        work->j0_fourier[(size_t)l*nr + r] = value;
      }
    }
    if (!in->large_area) {
      for (int q=0; q<in->nwindow; q++) {
        work->j0_window[(size_t)q*nr + r] =
          gsl_sf_bessel_J0(in->window_ell[q]*in->radius[r]);
      }
    }
  }
  if (!in->large_area) {
    for (int r=0; r<in->nprofile; r++) {
      for (int l=0; l<nl; l++) {
        work->j2_profile[(size_t)r*nl + l] =
          gsl_sf_bessel_Jn(2, in->ell[l]*in->profile_radius[r]);
      }
    }
  }

  // Quadratic kernels at antiparallel equal vectors: the large-area
  // limit and the zero-mode halo response only ever need Q(l, l, pi),
  // where the mixed E/B kernel vanishes identically.
  for (int row=0; row<in->nrow; row++) {
    if (in->row_kind[row] != 1) {
      for (int l=0; l<nl; l++) {
        double unused;
        covariance_quadratic_3pcf(in, row, in->ell[l], in->ell[l], COV_PI,
          work->quadratic_zero + (size_t)row*nl + l, &unused);
      }
    }
  }

  // The model's g already contains the lensing prefactor, so the full
  // physical convergence weight is W = chi*(1 + z)*g, in h/Mpc.
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
        work->triples[(size_t)3*count] = a;
        work->triples[(size_t)3*count + 1] = b;
        work->triples[(size_t)3*count + 2] = c;
        count++;
      }
    }
  }

  // The scalar matter projection depends only on the multiset of source
  // labels, not on their order. This map stores all six orderings of
  // each triple, so an ordered lookup lands on the shared unordered
  // entry; field radii are not involved here.
  for (int index=0; index<work->nt; index++) {
    const int* t = work->triples + (size_t)3*index;
    const int orders[6][3] = {{0,1,2},{0,2,1},{1,0,2},
      {1,2,0},{2,0,1},{2,1,0}};
    for (int p=0; p<6; p++) {
      work->triple_index[((size_t)t[orders[p][0]]*ns
        + t[orders[p][1]])*ns + t[orders[p][2]]] = index;
    }
  }

  for (int field=0; field<nf; field++) {
    if (!(in->field_source[field] >= 0
          && in->field_source[field] < ns)) {
      log_fatal("physical 3pcf covariance: field source index out of range");
      exit(1);
    }
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

  // Fill the per-triangle tables: each geometry node fixes the closure
  // side |q+p|, and one worker owns one node, filling every field's
  // closure filter and - for a finite window - the window value and the
  // kernel of every quadratic shape there. A zero closure carries zero
  // compensated aperture weight, so its isolated angular node simply
  // stores zero instead of asking the bispectrum provider for a
  // zero-length triangle.
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

  // Limber projection of the convergence spectra: at multipole ell the
  // line-of-sight integral collapses to
  //   C^ab(ell) = int dchi W_a(chi) W_b(chi) P(ell/chi, z) / chi^2.
  // The power factor does not depend on the source bins, so the first
  // loop tabulates it once per (z, ell) node; mode 2 asks the provider
  // for the nonlinear matter power at redshift z, the spectrum the
  // lensing two-point signal physically measures.
  double* power_z = calloc1d((int)covariance_product_3pcf(nz, nl));

  #pragma omp parallel for collapse(2) if(!omp_in_parallel()) schedule(static)
  for (int iz=0; iz<nz; iz++) {
    for (int l=0; l<nl; l++) {
      if (work->chi[iz] > 0.0) {
        power_z[(size_t)iz*nl + l] = physics_power_3pcf(work->model,
          in->ell[l]/work->chi[iz], work->z[iz], 2);
      }
    }
  }

  // Contract the shared power column with each ordered pair of lensing
  // kernels. One worker owns one (a, b) pair at one multipole, and the
  // inner redshift sum keeps the quadrature's input order.
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

  // Mark the ordered field pairs (a, b) that the S_ab|c table must
  // hold. S_ab|c is read only when a and b are the two unpaired legs of
  // a requested cubic row: the row's mean, its BB covariance and its
  // P*B covariance all draw on these same three leg choices, in their
  // exact ordered form. The separated source c still spans every source
  // bin, because the quadratic rows of a joint covariance can bring in
  // bins that no cubic row carries.
  int* bspec_pair_needed = malloc1d_int((int)field_pairs);
  memset(bspec_pair_needed, 0, sizeof(int)*field_pairs);

  for (int row=0; row<in->nrow; row++) {
    if (in->row_kind[row] != 1) {
      continue;
    }
    const int* fields = in->row_fields + (size_t)row*3;
    for (int p=0; p<3; p++) {
      const int a = fields[(p + 1)%3];
      const int b = fields[(p + 2)%3];
      bspec_pair_needed[(size_t)a*nf + b] = 1;
    }
  }

  int nworker = 1;
  #ifdef _OPENMP
  nworker = omp_get_max_threads();
  #endif

  // Give each worker its own slice of scratch for the dchi column, with
  // a cache-line-sized gap between slices: two workers then never write
  // the same cache line, even when calloc's base address is not
  // cache-line aligned. Workers never allocate scratch of their own.
  const size_t bstride = ((size_t)nz + 7)/8*8 + 8;
  double* local_b = calloc1d((int)covariance_product_3pcf(nworker, bstride));

  // Project the matter bispectrum to the convergence bispectrum
  //   B^abc(q,p,phi) = int dchi W_a W_b W_c B_m(q/chi, p/chi, |q+p|/chi)
  //                    / chi^4
  // for every source triple. The matter bispectrum is the expensive
  // factor and does not depend on the sources, so each worker owns one
  // triangle geometry, evaluates the dchi column of B_m once, and then
  // contracts that cached column with every triple's three lensing
  // kernels.
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
      const int* sources = work->triples + (size_t)3*t;
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

  // Contract the bispectrum into S_ab|c(q). S carries the two
  // coincident aperture filters and the bispectrum, but not the
  // separated leg's filter; its measure is p dp dphi/(2*pi)^2.
  // Averaging the (a, b) and (b, a) orders restores, at finite
  // quadrature spacing, the exact commutation of two fields measured at
  // the same center.
  #pragma omp parallel for collapse(2) if(!omp_in_parallel()) schedule(static)
  for (int a=0; a<nf; a++) {
    for (int b=0; b<nf; b++) {
      if (!bspec_pair_needed[(size_t)a*nf + b]) {
        continue;
      }
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

  // Build the real-space tables K_ab(r), K_ab(0), the windowed K
  // transform, and B_ab|c(r): the building blocks that the positional
  // partitions of the assemblers reuse. One worker owns one ordered
  // field pair, so the collapsed loop has no shared writes.
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

      // Windowed K transform for the P*T partitions: a finite window
      // transforms K_ab(r) back to Fourier space with the center-pair
      // density attached, while the large-area limit reduces the same
      // object to the filtered spectrum divided by the area. The ell
      // measure is folded in here so the P*T contraction later is a
      // plain dot product.
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

      // Only the finite-window BB term reads B_ab|c(r); the large-area
      // BB expression contracts S_ab|c(ell) directly, so no real-space
      // B table enters that branch. The first two fields of B use the
      // same ordered cubic-pair list as S. The K tables above stay
      // available to every Gaussian and P*T pair.
      if (!in->large_area && bspec_pair_needed[pair]) {
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

  free(bspec_pair_needed);
}



// ============================================================================
// [SECTION] XI FROM CALLER SPECTRA: THE PUBLIC BIN-AVERAGE ENTRY POINT
// ============================================================================

// ---------------------------------------------------------------------------
// covariance_xi_from_power_3pcf: the bin-averaged two-point function
//
//   xi[row] = int ell dell/(2*pi) Jbar(ell) [C_E(ell) + sign*C_B(ell)],
//
// from spectra the caller already evaluated on the quadrature nodes.
// Jbar is the average of J0 (xi+, sign = +1) or J4 (xi-, sign = -1)
// over the row's separation nodes with the row's own weights; the E
// and B spectra add for xi+ and subtract for xi-. The function performs no
// ell interpolation and no binary search, so the caller controls the
// quadrature completely. The additive noise bias of the estimator is
// not removed. Argument shapes and units: see the declaration in
// covariance_physics_3pcf.h.
// ---------------------------------------------------------------------------
void covariance_xi_from_power_3pcf(
  int nrow, // number of xi rows
  int nell, // Fourier quadrature nodes
  int ntheta, // separation samples per row
  const double* ell, // multipoles [nell]
  const double* dell, // dell weights [nell]
  const double* theta, // radians [nrow,ntheta]
  const double* theta_weight, // normalized weights [nrow,ntheta]
  const double* spectra_e, // E angular spectra [nrow,nell]
  const double* spectra_b, // B angular spectra [nrow,nell]
  const int* sign, // +1 or -1 [nrow]
  double* out // bin-averaged xi [nrow]
)
{
  // Rows for different source pairs often share identical separation
  // bins. Find one representative row for each distinct (bins, sign)
  // combination, so the Bessel average is prepared only once and every
  // spectrum then reads its kernel at contiguous ell nodes.
  int* representative = malloc1d_int(nrow);
  memset(representative, 0, sizeof(int)*(size_t)nrow);
  int* row_kernel = malloc1d_int(nrow);
  memset(row_kernel, 0, sizeof(int)*(size_t)nrow);
  int nkernel = 0;
  for (int row=0; row<nrow; row++) {
    if (!(sign[row] == 1
          || sign[row] == -1)) {
      log_fatal("physical 3pcf covariance: xi sign must be +1 or -1");
      exit(1);
    }
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

  double* kernels = calloc1d((int)covariance_product_3pcf(nkernel, nell));

  // Build each distinct kernel once: the Bessel function averaged over
  // the separation bin, times the integration measure ell*dell/(2*pi).
  // One worker owns one kernel, so no two workers share a write.
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

  // Contract each row's spectra with its kernel; one worker owns a row.
  // scalar: for every ell node l,
  //   sum += kernel[l]*(ce[l] + sign*cb[l]);
  // The vector loop runs two consecutive nodes per pass - lane 0 holds
  // node l, lane 1 holds node l + 1 - and adds lane 0 before lane 1,
  // the scalar accumulation order.
  #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
  for (int row=0; row<nrow; row++) {
    const double* kernel = kernels + (size_t)row_kernel[row]*nell;
    const double* ce = spectra_e + (size_t)row*nell;
    const double* cb = spectra_b + (size_t)row*nell;
    double sum = 0.0;
    int l = 0;

    // Broadcast the row's parity (+1 for xi+, -1 for xi-) to both
    // lanes: adjacent ell nodes use the same B-spectrum sign.
    const v2d parity = simde_mm_set1_pd(sign[row]);
    for (; l + 1 < nell; l += 2) {
      // Load the E spectrum at nodes l and l + 1; loadu needs no
      // aligned address.
      const v2d e = simde_mm_loadu_pd(ce + l);
      // Load the B spectrum at the same two nodes.
      const v2d b = simde_mm_loadu_pd(cb + l);
      // Apply the parity to B: xi- subtracts it, xi+ keeps it.
      const v2d signed_b = simde_mm_mul_pd(parity, b);
      // Form the two signal values C_E + sign*C_B, one per lane.
      const v2d signal = simde_mm_add_pd(e, signed_b);
      // Load the shared Bessel/bin/Fourier weights at these ell nodes.
      const v2d weight = simde_mm_loadu_pd(kernel + l);
      // Weight each lane's signal value independently.
      const v2d value = simde_mm_mul_pd(weight, signal);
      double lanes[2];
      // Store both lanes to the stack, then add low lane before high
      // lane: the scalar integration order.
      simde_mm_storeu_pd(lanes, value);
      sum += lanes[0];
      sum += lanes[1];
    }

    if (l < nell) {
      // Round the product before the final addition, as every SIMD
      // lane does; without the volatile the strict build may fuse this
      // scalar tail into an FMA with a different rounding.
      volatile double weighted = kernel[l]*(ce[l] + sign[row]*cb[l]);
      sum += weighted;
    }

    out[row] = sum;
  }

  free(kernels);
  free(representative);
  free(row_kernel);
}



// ============================================================================
// [SECTION] HALO SHEAR-PROFILE INTERPOLATION PLAN
// ============================================================================

// Index of an unordered pair of aperture radii in an upper triangle.
// A halo's projected shape depends on the aperture radius, the halo
// mass and the distance; the source bin enters only through the
// multiplicative lensing weight W, which stays outside the halo
// transforms. Profile products therefore depend on radius pairs alone,
// and the unordered pair (a, b) is stored once. This identity holds for
// the scalar lensing halo model implemented here.
static inline size_t covariance_radius_pair_3pcf(
  int n, // number of aperture radii
  int a, // first radius index, in [0,n)
  int b // second radius index, in [0,n); order is irrelevant
)
{
  const int low = a < b ? a : b;
  const int high = a < b ? b : a;
  // Rows 0..low-1 of the triangle hold n, n-1, ... entries, which sum
  // to low*(2n - low + 1)/2; high - low is the offset inside row `low`.
  return (size_t)low*(2*(size_t)n - low + 1)/2 + high - low;
}


// Geometry of the halo shear lookup, owned by one covariance call. A
// natural cubic spline of the shear profile is prepared on the
// coordinate x = log(1 + r/r_scale), which includes r = 0, and then
// sampled onto a uniform grid in x. Every displaced xi endpoint stores
// its fine-grid cell and fractional position before the halo loop
// starts, so the repeated per-halo reads need neither a binary search
// nor a logarithm. This follows the preparation / linear-read
// separation used in cosmo2D.c; the input quadrature may be nonuniform.
typedef struct {
  int coarse;              // input profile nodes nc = in->nprofile
  int fine;                // uniform fine nodes (nc-1)*upsample + 1
  int per_radius;          // xi endpoint samples per radius: ntheta*nphi
  double* x;               // [coarse] spline coordinate log(1+r/r_scale)
  double* inverse;         // [coarse] Thomas solver: 1/pivot per row
  double* upper;           // [coarse] Thomas solver: scaled superdiagonal
  double* coefficient;     // [coarse] spline c = (second derivative)/2
  double* forward;         // [coarse] Thomas solver: substituted RHS
  double* values;          // [fine] current halo's shear on the fine grid
  int* cell;               // [fine] coarse interval holding each fine node
  double* delta;           // [fine] x offset inside that coarse interval
  int* minus_cell;         // [nxi,nr,per_radius] fine cell, minus endpoint
  int* plus_cell;          // same shape: fine cell, plus endpoint
  double* minus_fraction;  // same shape: position inside minus_cell
  double* plus_fraction;   // same shape: position inside plus_cell
  double* endpoint_weight; // same shape: bin*angle weight * cos(phase)
} covariance_profile_plan_3pcf;


// Build the plan: the shared spline factorization, the fine-grid cell
// map, and - for every xi shape - each displaced endpoint's cell,
// fractional position and estimator weight. A request without xi rows
// leaves the plan empty.
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
  if (!(nc >= 3
        && in->profile_upsample >= 1
        && nc - 1 <= (INT_MAX - 1)/in->profile_upsample)) {
    log_fatal("physical 3pcf covariance: profile preparation needs >=3 "
      "nodes and a valid upsample factor");
    exit(1);
  }
  if (!(in->ntheta <= INT_MAX/in->nphi)) {
    log_fatal("physical 3pcf covariance: xi endpoint orientation count "
      "exceeds the index range");
    exit(1);
  }

  plan->coarse = nc;
  plan->fine = (nc - 1)*in->profile_upsample + 1;
  plan->per_radius = in->ntheta*in->nphi;
  const int nfine = plan->fine;
  const size_t endpoints = covariance_product_3pcf(
    covariance_product_3pcf(nxi, in->nr), plan->per_radius);

  plan->x = calloc1d(nc);
  plan->inverse = calloc1d(nc);
  plan->upper = calloc1d(nc);
  plan->coefficient = calloc1d(nc);
  plan->forward = calloc1d(nc);
  plan->values = calloc1d(nfine);
  plan->cell = malloc1d_int(nfine);
  memset(plan->cell, 0, sizeof(int)*(size_t)nfine);
  plan->delta = calloc1d(nfine);
  plan->minus_cell = malloc1d_int((int)endpoints);
  memset(plan->minus_cell, 0, sizeof(int)*endpoints);
  plan->plus_cell = malloc1d_int((int)endpoints);
  memset(plan->plus_cell, 0, sizeof(int)*endpoints);
  plan->minus_fraction = calloc1d((int)endpoints);
  plan->plus_fraction = calloc1d((int)endpoints);
  plan->endpoint_weight = calloc1d((int)endpoints);

  const double scale = in->profile_radius[1];
  if (!(in->profile_radius[0] == 0.0
        && scale > 0.0)) {
    log_fatal("physical 3pcf covariance: profile preparation nodes must "
      "start with zero then a positive radius");
    exit(1);
  }

  for (int r=0; r<nc; r++) {
    plan->x[r] = log1p(in->profile_radius[r]/scale);
    if (!(isfinite(plan->x[r])
          && (r == 0 || plan->x[r] > plan->x[r - 1]))) {
      log_fatal("physical 3pcf covariance: profile coordinates must be "
        "finite and strictly increasing");
      exit(1);
    }
  }

  // The spline's second derivatives solve a tridiagonal linear system
  // whose matrix depends only on the node geometry, not on the shear
  // values. The Thomas algorithm (Gaussian elimination specialized to
  // tridiagonal systems) factors that matrix once here: inverse holds
  // the reciprocal pivots and upper the scaled superdiagonal. Every
  // mass/redshift node later reuses them; only the right-hand side
  // changes per halo.
  for (int r=1; r<nc - 1; r++) {
    const double previous = plan->x[r] - plan->x[r - 1];
    const double next = plan->x[r + 1] - plan->x[r];
    plan->inverse[r] = 1.0/(2.0*(previous + next)
      - previous*plan->upper[r - 1]);
    plan->upper[r] = next*plan->inverse[r];
  }

  // For every uniform fine node, record which coarse interval contains
  // it and the x offset inside, so the per-halo fill evaluates the
  // cubic polynomial directly, without searching. Both grids increase,
  // so one sweep with a running coarse index suffices.
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

  // For every xi shape, halo-offset node and endpoint orientation,
  // store where the two displaced shear samples land on the fine grid
  // and the weight that multiplies their product. An xi estimator whose
  // center sits at offset u from the halo reads the shear at the two
  // points u -/+ (theta/2)(cos phi, sin phi); precomputing their cells
  // and fractions makes the per-halo read two linear interpolations.
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
          // Half-separation displacement along orientation phi. With
          // the estimator center at (radius, 0) relative to the halo,
          // the minus endpoint sits at (radius - dx, -dy) and the plus
          // endpoint at (radius + dx, +dy); rm and rp are their
          // distances from the halo center.
          const double dx = half*cos(in->phi[a]);
          const double dy = half*sin(in->phi[a]);
          const double xm = in->radius[r] - dx;
          const double xp = in->radius[r] + dx;
          const double rm = hypot(xm, dy);
          const double rp = hypot(xp, dy);
          if (!(rm <= in->profile_radius[nc - 1]
                && rp <= in->profile_radius[nc - 1])) {
            log_fatal("physical 3pcf covariance: profile grid does not "
              "enclose displaced xi endpoints");
            exit(1);
          }
          // Fine-grid coordinates of the two endpoints: um and up are
          // continuous positions in units of the fine step; im and ip
          // are the containing cells, clamped so the last node still
          // interpolates inside the final cell; the stored fractions
          // are the positions inside those cells.
          const double um = log1p(rm/scale)/step;
          const double up = log1p(rp/scale)/step;
          const int im = um >= nfine - 1 ? nfine - 2 : (int)um;
          const int ip = up >= nfine - 1 ? nfine - 2 : (int)up;
          plan->minus_cell[index] = im;
          plan->plus_cell[index] = ip;
          plan->minus_fraction[index] = um - im;
          plan->plus_fraction[index] = up - ip;
          // Spin-2 projection phase. The halo's shear at an endpoint is
          // tangential about the halo center, so it carries the phase
          // 2*ph of that endpoint's polar angle ph (phm for the minus
          // endpoint, php for the plus endpoint). xi+ multiplies one
          // shear by the conjugate of the other after rotating both to
          // the line joining the endpoints, so the line direction
          // cancels and the product keeps cos(2(phm - php)). xi-
          // multiplies the two shears without conjugation, so the line
          // direction phi survives: the phase is 2(phm + php) - 4 phi.
          // The imaginary parts average to zero over the full
          // orientation quadrature, so only the cosine is stored.
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


// Spline one halo's shear profile onto the fine grid. Only the spline
// ordinates change from halo to halo; every geometric factor and
// destination interval was prepared once above. The coefficients c are
// one half of the spline's second derivative at each node, and the
// natural boundary condition keeps c = 0 at both ends throughout the
// call. Accuracy is controlled by refining both the coarse input nodes
// and the fine sampling grid.
static void covariance_profile_fill_3pcf(
  covariance_profile_plan_3pcf* plan, // fixed geometry and owned scratch
  const double* shear               // coarse gamma_t [plan->coarse]
)
{
  const int nc = plan->coarse;

  // Forward substitution of the Thomas solve: fold this halo's
  // right-hand side through the prepared elimination factors.
  for (int r=1; r<nc - 1; r++) {
    const double previous = plan->x[r] - plan->x[r - 1];
    const double next = plan->x[r + 1] - plan->x[r];
    const double rhs = 3.0*((shear[r + 1] - shear[r])/next
      - (shear[r] - shear[r - 1])/previous);
    plan->forward[r] = (rhs - previous*plan->forward[r - 1])
      *plan->inverse[r];
  }

  // Back substitution yields the interior c coefficients.
  for (int r=nc - 2; r>=1; r--) {
    plan->coefficient[r] = plan->forward[r]
      - plan->upper[r]*plan->coefficient[r + 1];
  }

  // Evaluate the cubic on every fine node with Horner's scheme. The
  // linear coefficient b and cubic coefficient d follow from the
  // standard cubic-spline interval formulas; dx is the prepared offset
  // inside the interval.
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


// One halo-offset node's xi response: the weighted sum, over separation
// nodes and orientations, of the product of the two endpoint shear
// values. The endpoint samples were planned above, so each read is a
// linear interpolation on the fine grid. Two consecutive endpoint
// samples share the SIMD lanes; the sum adds the low lane before the
// high lane, keeping one fixed node order for any OpenMP thread count.
// The final odd sample runs in scalar code, so no vector load ever
// extends past the endpoint arrays.
static inline double covariance_profile_response_3pcf(
  const covariance_profile_plan_3pcf* plan, // prepared halo and geometry
  size_t start                            // first endpoint for this radius
)
{
  double sum = 0.0;
  int a = 0;

  // scalar: for every endpoint sample p = start + a,
  //   minus = values[mc] + minus_fraction[p]*(values[mc+1] - values[mc]);
  //   plus  = values[pc] + plus_fraction[p] *(values[pc+1] - values[pc]);
  //   sum  += endpoint_weight[p]*(minus*plus);
  // with mc = minus_cell[p] and pc = plus_cell[p] the prepared fine
  // cells. Each term interpolates the halo's shear at the two displaced
  // xi endpoints and weights their product with the prepared
  // bin/orientation/spin weight. The vector loop runs two consecutive
  // samples at once - lane 0 holds sample a, lane 1 holds sample a + 1 -
  // and adds lane 0 before lane 1, the scalar order.
  for (; a + 1 < plan->per_radius; a += 2) {
    const size_t p = start + a;
    const int m0 = plan->minus_cell[p];
    const int m1 = plan->minus_cell[p + 1];
    const int p0 = plan->plus_cell[p];
    const int p1 = plan->plus_cell[p + 1];
    // Gather the shear at the lower edge of each minus cell. set_pd
    // lists the high lane first: lane 1 receives values[m1] (sample
    // a + 1) and lane 0 receives values[m0] (sample a), matching the
    // lane order of the contiguous loadu reads below.
    const v2d ml = simde_mm_set_pd(plan->values[m1], plan->values[m0]);
    // Gather the shear at the upper edge of each minus cell, in the
    // same set_pd lane order: lane 1 from m1 + 1, lane 0 from m0 + 1.
    const v2d mh = simde_mm_set_pd(plan->values[m1 + 1], plan->values[m0 + 1]);
    // Load the two prepared fractional positions inside those cells.
    const v2d mf = simde_mm_loadu_pd(plan->minus_fraction + p);
    // Subtract the cell edges to form each lane's interpolation slope.
    const v2d md = simde_mm_sub_pd(mh, ml);
    // Multiply each slope by its endpoint's fractional position.
    const v2d ms = simde_mm_mul_pd(mf, md);
    // Add the lower edge back: the two minus-endpoint shear values.
    const v2d minus = simde_mm_add_pd(ml, ms);
    // Gather the lower edge of each plus cell; set_pd lane order as
    // above, lane 1 from p1 and lane 0 from p0.
    const v2d pl = simde_mm_set_pd(plan->values[p1], plan->values[p0]);
    // Gather the upper edge of each plus cell: lane 1 from p1 + 1,
    // lane 0 from p0 + 1.
    const v2d ph = simde_mm_set_pd(plan->values[p1 + 1], plan->values[p0 + 1]);
    // Load the prepared plus-endpoint fractional positions.
    const v2d pf = simde_mm_loadu_pd(plan->plus_fraction + p);
    // Form the two plus-cell interpolation slopes.
    const v2d pd = simde_mm_sub_pd(ph, pl);
    // Scale the slopes by the fractional positions.
    const v2d ps = simde_mm_mul_pd(pf, pd);
    // Add the lower edge back: the two plus-endpoint shear values.
    const v2d plus = simde_mm_add_pd(pl, ps);
    // Multiply the shears at the two physical endpoints, one
    // orientation sample per lane.
    const v2d pair = simde_mm_mul_pd(minus, plus);
    // Load the prepared bin/orientation/spin weights of the two samples.
    const v2d weights = simde_mm_loadu_pd(plan->endpoint_weight + p);
    // Weight each lane's shear product.
    const v2d weighted = simde_mm_mul_pd(weights, pair);
    double values[2];
    // Store both lanes to the stack, then add them in the fixed scalar
    // order: low lane (sample a) first, high lane (sample a + 1) second.
    simde_mm_storeu_pd(values, weighted);
    sum += values[0];
    sum += values[1];
  }

  if (a < plan->per_radius) {
    const size_t p = start + a;
    const int m = plan->minus_cell[p];
    const int u = plan->plus_cell[p];
    // Each SIMD interpolation above rounds its multiply and its add
    // separately. The volatiles force the same rounding boundaries on
    // this single scalar endpoint, where compiler contraction would
    // otherwise fuse scalar-only FMAs and change the result bits.
    volatile double minus_step = plan->minus_fraction[p]
      *(plan->values[m + 1] - plan->values[m]);
    volatile double plus_step = plan->plus_fraction[p]
      *(plan->values[u + 1] - plan->values[u]);
    const double minus = plan->values[m] + minus_step;
    const double plus = plan->values[u] + plus_step;
    volatile double weighted = plan->endpoint_weight[p]*(minus*plus);
    sum += weighted;
  }

  return sum;
}


// Free every allocation of the plan preparation. A large-area call
// never fills the plan, so its pointers are NULL and free does nothing.
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



// ============================================================================
// [SECTION] HALO-MODEL TERMS: P*T, CONNECTED ONE-HALO AND THE RESPONSE
// ============================================================================

// Fill the three halo-model slabs of the covariance: terms[2] receives
// the P*T cross term between cubic rows, terms[4] the connected
// one-halo term of every row pair, and terms[5] the halo-abundance
// response. The mean is untouched here.
//
// The common object is one halo's estimator response as a function of
// its angular offset u from the estimator center. An aperture row
// multiplies radial filtered convergence profiles of the halo; an xi
// row multiplies two spin-2 shear values at the actual displaced
// endpoints, averaged with the separation-bin and orientation weights
// the input carries. The halo's shear profile is first tabulated on a wider
// radial grid, so those displaced samples are interpolated instead of
// transformed again and again; that interpolation grid is refined
// independently of the quadratures.
//
// Parameters:
//   work  - prepared workspace; the halo tables are allocated here
//   terms - output [6, nrow, nrow]; slabs 2, 4 and 5 are accumulated
static void covariance_halo_terms_3pcf(
  covariance_work_3pcf* work, // prepared workspace; halo tables allocated here
  double* terms // out [6,nrow,nrow]; adds to slabs 2, 4 and 5
)
{
  const covariance_input_3pcf* in = work->in;
  const int nf = in->nfield;  // field labels
  const int nl = in->nell;    // Fourier nodes
  const int nr = in->nr;      // halo-offset nodes
  const int no = in->nrow;    // covariance rows
  const int nw = in->nwindow; // survey-window nodes
  const int nm = in->nmass;   // halo mass nodes
  const int need_profiles = work->has_cubic || !in->large_area;
  const size_t matrix = covariance_product_3pcf(no, no);
  const size_t nodes = covariance_product_3pcf(work->nz, nm);
  // 2.77536627e11 is rho_crit/h^2 in (Msun/h)/(Mpc/h)^3; times Omega_m
  // it is today's comoving mean matter density, as in the halo tables.
  const double rho = 2.77536627e11*physics_omega_m_3pcf(work->model);

  work->sigma2 = calloc1d((int)nodes);
  work->abundance = calloc1d((int)nodes);
  work->bias = calloc1d((int)nodes);
  work->concentration = calloc1d((int)nodes);
  work->response_bias = calloc1d((int)covariance_product_3pcf(work->nz, no));

  covariance_halos_3pcf(work->model, in, work->nz, work->z, 0, NULL,
    work->sigma2, work->abundance, work->bias, work->concentration, NULL);

  // Deduplicate the aperture radii. Equality means equality of the
  // numerical radii the input carries; no tolerance ever merges two
  // different observables. The source labels stay outside every halo transform,
  // carried by field_lens and row_lens below.
  int* radius_index = malloc1d_int(nf);
  memset(radius_index, 0, sizeof(int)*(size_t)nf);
  double* unique_radius = calloc1d(nf);
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
  const size_t npair = covariance_product_3pcf(nrad, (size_t)nrad + 1)/2;
  int* pair_needed = malloc1d_int((int)npair);
  memset(pair_needed, 0, sizeof(int)*npair);
  int* radius_needed = malloc1d_int(nrad);
  memset(radius_needed, 0, sizeof(int)*(size_t)nrad);
  int* representative = malloc1d_int(no);
  memset(representative, 0, sizeof(int)*(size_t)no);
  int* shape_radius = malloc1d_int((int)covariance_product_3pcf(no, 3));
  memset(shape_radius, 0, sizeof(int)*(size_t)no*3);
  int* row_shape = malloc1d_int(no);
  memset(row_shape, 0, sizeof(int)*(size_t)no);
  int* xi_shape = malloc1d_int(no);
  memset(xi_shape, 0, sizeof(int)*(size_t)no);

  int nshape = 0;
  int nxi = 0;

  // Group the rows into shapes. A shape is exactly the geometry a halo
  // response depends on: an aperture row reduces to its unordered
  // multiset of radii, an xi row to its separation nodes and weights.
  // Rows that share a shape share the response tables below and differ
  // only through their lensing weights.
  for (int row=0; row<no; row++) {
    const int kind = in->row_kind[row];
    const int nleg = kind == 1 ? 3 : 2;
    const int* fields = in->row_fields + (size_t)row*3;
    int radii[3] = {-1, -1, -1};
    if (kind <= 1) {
      for (int a=0; a<nleg; a++) {
        radii[a] = radius_index[fields[a]];
        radius_needed[radii[a]] = radius_needed[radii[a]]
          || kind == 1 || !in->large_area;
      }
      // Sort the radius indices in place: the halo response depends on
      // the unordered multiset of radii, so putting them in canonical
      // order lets two rows with permuted legs share one shape.
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
          equal = equal && radii[a] == shape_radius[(size_t)3*s + a];
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
        shape_radius[(size_t)3*s + a] = radii[a];
      }
      xi_shape[s] = kind >= 2 ? nxi++ : -1;
      nshape++;
    }
    row_shape[row] = s;
  }

  // Index maps for the P*T contractions. A P*T partition chooses one
  // field from each cubic estimator for the two-point function P; the
  // other two fields of each estimator supply a pair of halo profiles
  // inside the connected four-point function T. The dot product the
  // partition needs depends only on the two chosen single fields and
  // the two complementary radius pairs, so one "entity" is a
  // (field, radius pair) choice and one contraction is an ordered pair
  // of entities. The source weights of the four halo fields stay in the
  // covariance-cell sum, in their original order. Only choices present
  // in the requested cubic rows are gathered. These integer maps belong
  // to this call: their values are fixed before any halo is visited,
  // and only the contraction values are refilled per halo.
  size_t* row_entity = NULL;
  int* entity_field = NULL;
  size_t* entity_radius_pair = NULL;
  size_t* contraction_index = NULL;
  size_t* contraction_left = NULL;
  size_t* contraction_right = NULL;
  double* pt_contraction = NULL;
  size_t nentity = 0;
  size_t ncontraction = 0;

  if (work->has_cubic) {
    size_t ncubic = 0;
    for (int row=0; row<no; row++) {
      ncubic += in->row_kind[row] == 1;
    }

    const size_t capacity = covariance_product_3pcf(ncubic, 3);
    row_entity = calloc1d_generic(
      covariance_product_3pcf(no, 3), sizeof(size_t));
    if (row_entity == NULL) {
      log_fatal("physical 3pcf covariance: cannot allocate covariance tables");
      exit(1);
    }
    entity_field = malloc1d_int((int)capacity);
    memset(entity_field, 0, sizeof(int)*capacity);
    entity_radius_pair = calloc1d_generic(capacity, sizeof(size_t));
    if (entity_radius_pair == NULL) {
      log_fatal("physical 3pcf covariance: cannot allocate covariance tables");
      exit(1);
    }

    for (int row=0; row<no; row++) {
      if (in->row_kind[row] != 1) {
        continue;
      }
      const int* fields = in->row_fields + (size_t)row*3;
      for (int p=0; p<3; p++) {
        const size_t pair = covariance_radius_pair_3pcf(nrad,
          radius_index[fields[(p + 1)%3]],
          radius_index[fields[(p + 2)%3]]);
        size_t e = 0;
        while (e < nentity && (entity_field[e] != fields[p]
          || entity_radius_pair[e] != pair)) {
          e++;
        }
        if (e == nentity) {
          entity_field[e] = fields[p];
          entity_radius_pair[e] = pair;
          nentity++;
        }
        row_entity[(size_t)row*3 + p] = e;
      }
    }

    contraction_index = calloc1d_generic(
      covariance_product_3pcf(nentity, nentity), sizeof(size_t));
    if (contraction_index == NULL) {
      log_fatal("physical 3pcf covariance: cannot allocate covariance tables");
      exit(1);
    }
    // Keep entity pairs ordered: the finite-window P table multiplies
    // its fields in a fixed order, and exchanging them could round
    // differently. In contraction_index a zero marks an unused pair,
    // and a used pair stores its compact contraction index plus one.
    for (int i=0; i<no; i++) {
      if (in->row_kind[i] != 1) {
        continue;
      }
      for (int j=i; j<no; j++) {
        if (in->row_kind[j] != 1) {
          continue;
        }
        for (int p=0; p<3; p++) {
          const size_t left = row_entity[(size_t)i*3 + p];
          for (int q=0; q<3; q++) {
            const size_t right = row_entity[(size_t)j*3 + q];
            const size_t pair = left*nentity + right;
            if (contraction_index[pair] == 0) {
              contraction_index[pair] = 1;
              ncontraction++;
            }
          }
        }
      }
    }

    contraction_left = calloc1d_generic(ncontraction, sizeof(size_t));
    if (contraction_left == NULL) {
      log_fatal("physical 3pcf covariance: cannot allocate covariance tables");
      exit(1);
    }
    contraction_right = calloc1d_generic(ncontraction, sizeof(size_t));
    if (contraction_right == NULL) {
      log_fatal("physical 3pcf covariance: cannot allocate covariance tables");
      exit(1);
    }
    pt_contraction = calloc1d((int)ncontraction);

    size_t next = 0;
    for (size_t left=0; left<nentity; left++) {
      for (size_t right=0; right<nentity; right++) {
        const size_t pair = left*nentity + right;
        if (contraction_index[pair] != 0) {
          contraction_left[next] = left;
          contraction_right[next] = right;
          contraction_index[pair] = ++next;
        }
      }
    }
  }

  covariance_profile_plan_3pcf profile_plan;
  memset(&profile_plan, 0, sizeof(profile_plan));
  if (!in->large_area) {
    covariance_profile_plan_3pcf_prepare(in, nshape, representative,
      xi_shape, nxi, &profile_plan);
  }

  // In the large-area limit the connected terms use only each
  // estimator's zero-mode response H = int d^2u f(u). Quadratic H has a
  // direct Fourier expression below, so that branch never touches
  // displaced xi profiles or profile interpolation. Cubic H and P*T
  // still need the aperture profiles and their pair products. A finite
  // center window additionally needs every response transformed to the
  // survey multipoles q.
  const size_t radius_fourier = covariance_product_3pcf(nrad, nl);
  double* unique_filter = need_profiles
    ? calloc1d((int)radius_fourier) : NULL;
  double* window_filter = !in->large_area
    ? calloc1d((int)covariance_product_3pcf(nrad, nw)) : NULL;

  for (int a=0; a<nrad; a++) {
    if (need_profiles) {
      for (int l=0; l<nl; l++) {
        unique_filter[(size_t)a*nl + l] =
          covariance_filter_3pcf(in->ell[l], unique_radius[a]);
      }
    }
    if (!in->large_area) {
      for (int q=0; q<nw; q++) {
        window_filter[(size_t)a*nw + q] =
          covariance_filter_3pcf(in->window_ell[q], unique_radius[a]);
      }
    }
  }

  double* profile_fourier = need_profiles
    ? calloc1d((int)radius_fourier) : NULL;
  double* profile_real = need_profiles
    ? calloc1d((int)covariance_product_3pcf(nrad, nr)) : NULL;
  double* profile_window = !in->large_area
    ? calloc1d((int)covariance_product_3pcf(nf, nw)) : NULL;
  double* shear = !in->large_area && nxi > 0
    ? calloc1d(in->nprofile) : NULL;
  double* nfw = calloc1d(nl);
  double* nfw_square = calloc1d(nl);
  double* nfw_window = !in->large_area
    ? calloc1d(nw) : NULL;
  double* field_lens = calloc1d(nf);
  double* row_lens = calloc1d(no);
  double* product_pair = work->has_cubic
    ? calloc1d((int)covariance_product_3pcf(npair, nl)) : NULL;
  double* product_real = work->has_cubic
    ? calloc1d((int)covariance_product_3pcf(npair, nr)) : NULL;
  double* response_real = need_profiles
    ? calloc1d((int)covariance_product_3pcf(nshape, nr)) : NULL;
  double* response_fourier = !in->large_area
    ? calloc1d((int)covariance_product_3pcf(nshape, nw)) : NULL;
  double* shape_zero = calloc1d(nshape);
  double* response_zero = calloc1d(no);
  double* window_measure = calloc1d(nw);

  for (int q=0; q<nw; q++) {
    window_measure[q] = in->window_ell[q]*in->window_dell[q]*COV_I2PI
      *work->window_fourier[q];
  }

  // Main halo traversal. The (redshift, mass) nodes are visited in one
  // fixed serial order; parallel workers appear only inside a node,
  // where each owns an output row or a table slice. One node's field
  // profiles are shared by every covariance cell before the next node
  // overwrites them, so memory stays independent of the number of
  // mass/redshift nodes.
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
      // Amplitude of the halo's filtered angular profile: in the Phi
      // equation of COVARIANCE.md, Phi = (W/chi^2)(M/rho_m) u U, this
      // factor is (M/rho_m)/chi^2. The lensing weight W is attached per
      // field below, and u and U enter through the nfw/filter tables.
      const double amplitude = mass/(rho*chi*chi);
      // M200m radius: M = (4 pi/3)*200*rho_m*r200^3, so the 800 in the
      // denominator is 4*200.
      const double r200 = cbrt(3.0*mass/(800.0*COV_PI*rho));
      // Expected halo count per steradian: dnu = chi^2 dchi dM dn/dM.
      // number carries dM dn/dM; measure supplies the chi^2 dchi part.
      const double number = in->dmass[im]*work->abundance[node];
      const double measure = work->dchi[iz]*chi*chi*number;

      for (int l=0; l<nl; l++) {
        nfw[l] = amplitude*covariance_nfw_3pcf(in->ell[l]/chi,
          r200, work->concentration[node]);
        nfw_square[l] = nfw[l]*nfw[l];
      }

      if (!in->large_area) {
        for (int q=0; q<nw; q++) {
          nfw_window[q] = amplitude*covariance_nfw_3pcf(
            in->window_ell[q]/chi, r200, work->concentration[node]);
        }
      }

      // Transform this halo's filtered profile to real space once per
      // unique aperture radius:
      //   profile_real(u) = int ell dell/(2*pi) J0(ell u) u_NFW U_R,
      // the filtered convergence the aperture at offset u sees. Every
      // row with that radius reads the same table, and one worker owns
      // one radius.
      #pragma omp parallel for \
        if(need_profiles && !omp_in_parallel()) schedule(static)
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

      if (!in->large_area) {
        for (int a=0; a<nf; a++) {
          for (int q=0; q<nw; q++) {
            profile_window[(size_t)a*nw + q] = field_lens[a]*nfw_window[q]
              *window_filter[(size_t)radius_index[a]*nw + q];
          }
        }
      }

      // Tangential shear profile of this halo on the wide grid:
      //   shear(u) = int ell dell/(2*pi) J2(ell u) u_NFW.
      // J2 appears because shear is the spin-2 distortion sourced by
      // the convergence profile. The spline fill then prepares the
      // fine-grid samples that the displaced xi endpoints read.
      if (nxi > 0 && !in->large_area) {
        #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
        for (int r=0; r<in->nprofile; r++) {
          shear[r] = covariance_dot_3pcf(nl, work->fourier_measure, nfw,
            work->j2_profile + (size_t)r*nl);
        }
        covariance_profile_fill_3pcf(&profile_plan, shear);
      }

      // Response of every shape to this one halo. An aperture shape
      // multiplies its radial profiles at each halo offset; an xi shape
      // sums its displaced endpoint products. The zero mode
      // H = int d^2u f(u) follows from the radial quadrature, and a
      // finite window also stores the response transformed to the
      // survey multipoles. One worker owns one shape.
      #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
      for (int s=0; s<nshape; s++) {
        const int row = representative[s];
        const int kind = in->row_kind[row];
        if (in->large_area && kind != 1) {
          shape_zero[s] = covariance_dot_3pcf(nl, work->fourier_measure,
            work->quadratic_zero + (size_t)row*nl, nfw_square);
          continue;
        }
        for (int r=0; r<nr; r++) {
          double value;
          if (kind <= 1) {
            const int* radii = shape_radius + (size_t)3*s;
            value = profile_real[(size_t)radii[0]*nr + r]
              *profile_real[(size_t)radii[1]*nr + r];
            if (kind == 1) {
              value *= profile_real[(size_t)radii[2]*nr + r];
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
        // A quadratic shape's zero mode has the direct Fourier
        // expression sum_l (ell dell/2pi) Q(l,l,pi) u^2(l), which
        // replaces the radial sum here: it avoids truncating the
        // halo-offset quadrature and is the same expression the
        // large-area branch above uses.
        if (kind != 1) {
          zero = covariance_dot_3pcf(nl, work->fourier_measure,
            work->quadratic_zero + (size_t)row*nl, nfw_square);
        }
        shape_zero[s] = zero;
        if (!in->large_area) {
          // Transform of the response over the halo offset, read by the
          // finite-window connected term at the survey multipoles.
          for (int q=0; q<nw; q++) {
            response_fourier[(size_t)s*nw + q] = covariance_dot_3pcf(nr,
              work->radial_volume, response_real + (size_t)s*nr,
              work->j0_window + (size_t)q*nr);
          }
        }
      }

      for (int row=0; row<no; row++) {
        response_zero[row] = row_lens[row]*shape_zero[row_shape[row]];
        work->response_bias[(size_t)iz*no + row] +=
          number*work->bias[node]*response_zero[row];
      }

      // Products of two radial halo profiles: the building block of the
      // connected one-halo four-field function in P*T. One unordered
      // radius pair serves every source pair with those radii, because
      // the lensing weights W are attached later in the cell sum. Pairs
      // absent from the requested cubic rows are skipped.
      #pragma omp parallel for \
        if(work->has_cubic && !omp_in_parallel()) schedule(static)
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

      // The pair-profile transforms above change with halo mass and
      // distance, so every requested P*T contraction is refilled here
      // before the covariance workers read it. Each entry runs the same
      // ordered quadrature its uncached expression would run, unequal
      // aperture radii included.
      if (work->has_cubic) {
        #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
        for (size_t c=0; c<ncontraction; c++) {
          const size_t left = contraction_left[c];
          const size_t right = contraction_right[c];
          const int a = entity_field[left];
          const int b = entity_field[right];
          pt_contraction[c] = covariance_dot_3pcf(nl,
            work->pair_window + ((size_t)a*nf + b)*nl,
            product_pair + entity_radius_pair[left]*nl,
            product_pair + entity_radius_pair[right]*nl);
        }
      }

      // Accumulate this halo's connected and P*T contributions into the
      // upper-triangle cells. Upper-triangle rows shorten as i grows,
      // and the input may group all cubic rows together, so the
      // round-robin schedule(static,1) hands successive rows to
      // successive workers and spreads both costs across the team. Each
      // worker still owns complete rows and visits them in order.
      #pragma omp parallel for if(!omp_in_parallel()) schedule(static,1)
      for (int i=0; i<no; i++) {
        const int* a = in->row_fields + (size_t)i*3;
        for (int j=i; j<no; j++) {
          const int* b = in->row_fields + (size_t)j*3;
          const size_t cell = (size_t)i*no + j;
          // Connected one-halo term: H_i H_j / A in the large-area
          // limit, or the window-weighted overlap of the two response
          // transforms for a finite survey.
          const double connected = in->large_area
            ? response_zero[i]*response_zero[j]/in->area
            : row_lens[i]*row_lens[j]*covariance_dot_3pcf(nw, window_measure,
                response_fourier + (size_t)row_shape[i]*nw,
                response_fourier + (size_t)row_shape[j]*nw);
          terms[4*matrix + cell] += measure*connected;

          if (in->row_kind[i] == 1 && in->row_kind[j] == 1) {
            // P*T partition sum. Nine choices pair the two P fields
            // across the estimators: p picks row i's separated field
            // and q picks row j's; the prepared contraction holds the
            // profile-pair transforms, and the four lensing weights are
            // attached here in their original order.
            double pt = 0.0;
            for (int p=0; p<3; p++) {
              const int a0 = a[(p + 1)%3];
              const int a1 = a[(p + 2)%3];
              for (int q=0; q<3; q++) {
                const int b0 = b[(q + 1)%3];
                const int b1 = b[(q + 2)%3];
                const size_t left = row_entity[(size_t)i*3 + p];
                const size_t right = row_entity[(size_t)j*3 + q];
                const size_t c = contraction_index[left*nentity + right] - 1;
                pt += field_lens[a0]*field_lens[a1]
                  *field_lens[b0]*field_lens[b1]*pt_contraction[c];
              }
              // Six more choices close P inside one estimator; the
              // compensated filters remove them only in the large-area
              // limit, so a finite window keeps them. The internal pair
              // contributes its coincident K(0), while the estimator's
              // separated field couples to the other row's full
              // response through the survey window: three placements
              // with row i's pair internal (first addition) and three
              // with row j's pair internal (second).
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

  // The squeezed two-halo term uses the abundance responses that the
  // main traversal accumulated, R_i(chi) = int dM (dn/dM) b_1 H_i: how
  // strongly each row responds to a long background density mode
  // through the halo abundance. Limber's radial delta reduces the long-mode
  // coupling to
  //   int dchi chi^2 R_i R_j int q dq P_linear(q/chi) G(q)/(2*pi).
  // The finite survey window G stays in this term even when the
  // short-mode terms use the large-area approximation, because the
  // long-mode variance is set by the survey shape. No second copy of
  // this term is an SSC: another SSC prescription must first check this
  // overlap. Mode 0 asks for linear power - the survey-scale modes that
  // modulate halo abundance are linear.
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
  free(row_entity);
  free(entity_field);
  free(entity_radius_pair);
  free(contraction_index);
  free(contraction_left);
  free(contraction_right);
  free(pt_contraction);
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



// ============================================================================
// [SECTION] GAUSSIAN, BB AND P*B ASSEMBLERS
// ============================================================================

// Gaussian covariance of two quadratic rows i and j. Wick's theorem
// splits the four-field cumulant into two pairings, and each pairing
// joins one field of row i to one field of row j, so the covariance
// integrates the two kernels Q_i and Q_j against products of two
// spectra. A finite window keeps both Fourier magnitudes independent
// and ties them together with the window G at the closure vector. The
// second Wick pairing reverses which of row j's legs receives which
// vector; an unequal-radius Map^2 kernel changes under that exchange,
// so the loop forms its swapped filter product explicitly.
//
// Parity. E and B spectra are diagonal in parity space. Shape noise
// supplies equal E and B spectra, and treating that noise as E-mode
// alone would bias every xi variance, so the B rows of each pairing are
// kept throughout.
//
// noise_mode = 1 declares continuous xi annuli weighted uniformly by
// area, in the large-area limit. The purpose of this mode is accuracy
// of the dominant noise block: the completeness of the Hankel transform
// gives int ell dell/(2*pi) Jbar_i Jbar_j = overlap/(Omega_i Omega_j),
// with Omega the annulus areas, so the pure-noise block has the closed
// form NN = 2 (N_ac N_bd + N_ad N_bc) overlap/(A Omega_i Omega_j),
// added after the spectral loop. Inside the loop only the CC and CN
// pieces are formed: evaluating a large finite-band NN there and
// subtracting it later would lose the small signal pieces to rounding.
// Between opposite xi parities the noise cancels, so only equal-kind
// cells receive the closed form. Every other Gaussian block keeps its
// finite-band filters.
//
// Parameters:
//   work - prepared kernels, spectra and window tables
//   i, j - the two quadratic rows (j >= i in the caller's triangle)
static double covariance_gaussian_two_3pcf(
  const covariance_work_3pcf* work, // prepared kernels, spectra and windows
  int i, // first quadratic row
  int j // second quadratic row, j >= i
)
{
  const covariance_input_3pcf* in = work->in;
  const int nl = in->nell;           // Fourier nodes
  const size_t ng = work->ngeometry; // Fourier triangles
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
    // Large-area limit: the window collapses the pairing onto opposite
    // equal Fourier vectors, both Wick pairings coincide, and a single
    // ell integral with the kernels at (q, q, pi) remains; dividing by
    // the area replaces the window G.
    if (in->large_area) {
      const double qi = work->quadratic_zero[(size_t)i*nl + q];
      const double qj = work->quadratic_zero[(size_t)j*nl + q];
      double spectra;
      // noise_mode = 1 forms CC and CN only; the closed-form NN of the
      // header is added once after this loop.
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
    // Finite window: both Wick pairings keep their own spectra.
    // efirst joins (a,c) at vector q with (b,d) at vector p; esecond
    // joins (a,d) at q with (b,c) at p. The E spectra carry signal
    // plus noise (mode 0).
    for (int p=0; p<nl; p++) {
      const double efirst = covariance_power_value_3pcf(work, a, c, q, 0)
        *covariance_power_value_3pcf(work, b, d, p, 0);
      const double esecond = covariance_power_value_3pcf(work, a, d, q, 0)
        *covariance_power_value_3pcf(work, b, c, p, 0);

      double bbspectra = 0.0;
      double mixedspectra = 0.0;
      if (both_shear) {
        // Pure-B rows of the two pairings: B spectra are noise alone
        // (mode 1). bb_sign flips xi- rows, whose kernel enters the B
        // sector with the opposite sign to E.
        bbspectra = bb_sign_i*bb_sign_j
          *(covariance_power_value_3pcf(work, a, c, q, 1)
              *covariance_power_value_3pcf(work, b, d, p, 1)
            + covariance_power_value_3pcf(work, a, d, q, 1)
              *covariance_power_value_3pcf(work, b, c, p, 1));
        // Mixed rows pair one E spectrum with one B spectrum, in both
        // orders. mixed_sign is the parity each row's EB kernel
        // acquires when the E and B legs swap between its two Fourier
        // arguments: odd for xi+ (its sine is of the relative angle
        // alone), even for xi-.
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

      // Angular sum at fixed (q, p): the window G at the closure side
      // ties the two magnitudes together.
      double angle_sum = 0.0;
      for (int angle=0; angle<in->nphi; angle++) {
        const size_t g = covariance_geometry_3pcf(in, q, p, angle);
        const size_t ii = (size_t)work->quadratic_index[i]*ng + g;
        const size_t jj = (size_t)work->quadratic_index[j]*ng + g;
        const double qi = work->quadratic[ii];
        const double qj = work->quadratic[jj];
        // The second pairing hands row j's legs the vectors in the
        // opposite order. An xi kernel and an equal-radius Map^2 kernel
        // are symmetric under that exchange - the estimator weighs its
        // two legs identically, and the kernel depends on the two
        // vectors only through exchange-symmetric combinations - so qj
        // already serves both pairings. Only an unequal-radius Map^2
        // kernel changes: its swapped filter product is formed here.
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

  // Closed-form NN block of noise_mode = 1 (see the header).
  // Omega = pi (hi^2 - lo^2) is each row's annulus area and the overlap
  // is the area of the annulus intersection. Only equal xi kinds
  // receive it: between xi+ and xi- the E and B noise cancel.
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


// P*B cross-covariance of one quadratic row with one cubic row. After
// mean subtraction the five-field cumulant leaves nine positional
// choices. Six of them connect one leg of the quadratic estimator to
// one leg of the cubic estimator through the two-point function P; the
// cubic estimator's remaining two legs are already integrated into S,
// which leaves two Fourier vectors and their relative angle. The other
// three choices close one pair of the cubic estimator's legs into its
// coincident K(0), while the two quadratic fields and the remaining
// cubic field span the full bispectrum triangle. The compensated
// filters remove those internal choices only in the large-area limit,
// so a finite window keeps them.
//
// Parameters:
//   work      - prepared kernels, spectra, S and bispectrum tables
//   quadratic - the Map^2 or xi row
//   cubic     - the Map^3 row
static double covariance_pb_3pcf(
  const covariance_work_3pcf* work, // prepared kernels, spectra, S and B tables
  int quadratic, // the Map^2 or xi row
  int cubic // the Map^3 row
)
{
  const covariance_input_3pcf* in = work->in;
  const int nf = in->nfield;         // field labels
  const int nl = in->nell;           // Fourier nodes
  const int ns = work->ns;           // source bins
  const size_t ng = work->ngeometry; // Fourier triangles
  const int* a = in->row_fields + (size_t)quadratic*3;
  const int* b = in->row_fields + (size_t)cubic*3;

  double value = 0.0;
  for (int q=0; q<nl; q++) {
    // Large-area limit: compensation leaves only the six cross
    // choices, the kernel collapses to (q, q, pi), and every factor
    // sits at the same magnitude q. left picks the quadratic leg that
    // P joins (the other leg's source becomes S's separated leg), and
    // right picks the cubic leg, which keeps its filter at q.
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
        // The choice that joins the quadratic row's second leg uses the
        // kernel with its legs exchanged. As in the Gaussian assembler,
        // an xi kernel and an equal-radius Map^2 kernel are symmetric
        // under that exchange, so qe serves both; only an
        // unequal-radius Map^2 forms a swapped filter product.
        const double swapped = in->row_kind[quadratic] == 0
          ? work->filters[(size_t)a[0]*nl + p]
              *work->filters[(size_t)a[1]*nl + q] : qe;
        // Six cross choices: P joins quadratic field a[left] to cubic
        // field b[right] at the first vector q, where the cubic leg
        // keeps its filter; the cubic row's other two fields enter S at
        // the second vector p, with the remaining quadratic source as
        // the separated leg.
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

        // Three internal choices: one pair of the cubic row closes into
        // its coincident K(0); the two quadratic fields and the
        // remaining cubic field form the bispectrum triangle, with that
        // field's filter on the closure side.
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


// Gaussian and BB covariance of two cubic rows, evaluated in real
// space.
//
// Derivation. Both estimators average over their center positions
// inside the survey. For statistically stationary fields, the
// correlation of a field at one center with a field at the other
// depends only on the offset r between the centers, and is exactly the
// transformed two-point function K(r) of the file head. The survey
// window supplies the density of center pairs at each offset
// (covariance_window_3pcf), so every disconnected term becomes one
// radial integral of products of K factors against that pair density.
//
// Gaussian: Wick pairing of the six fields with no pair inside a
// single estimator gives six all-cross choices, a product of three
// cross K(r) factors for each way of matching the legs. A finite
// window keeps nine more one-cross choices, where each estimator
// closes one internal pair - contributing its coincident K(0) - and a
// single cross K(r) connects the remaining legs. BB: after the product
// of the two estimator means is removed, nine splits into two field
// triples remain, each the product of two B_ab|c factors. All
// positional choices are kept even when field labels repeat.
//
// Parameters:
//   work     - prepared K and B tables
//   i, j     - the two cubic rows (j >= i in the caller's triangle)
//   gaussian - out: Gaussian term of the cell
//   bb       - out: BB term of the cell
static void covariance_cubic_disconnected_3pcf(
  const covariance_work_3pcf* work, // prepared K and B tables
  int i, // first cubic row
  int j, // second cubic row, j >= i
  double* gaussian, // out: Gaussian term of the cell
  double* bb // out: BB term of the cell
)
{
  const covariance_input_3pcf* in = work->in;
  const int nf = in->nfield; // field labels
  const int nr = in->nr;     // center-separation nodes
  const int nl = in->nell;   // Fourier nodes
  const int* a = in->row_fields + (size_t)i*3;
  const int* b = in->row_fields + (size_t)j*3;
  const int permutations[6][3] = {{0,1,2},{0,2,1},{1,0,2},
    {1,2,0},{2,0,1},{2,1,0}};

  *gaussian = 0.0;
  *bb = 0.0;

  // Gaussian term: quadrature over the center offset r. radial_measure
  // already contains 2*pi*r*dr times the center-pair density.
  for (int r=0; r<nr; r++) {
    double density = 0.0;
    // Six all-cross Wick pairings: permutation p matches leg `leg` of
    // row i with leg permutations[p][leg] of row j, and the product of
    // the three cross K(r) factors follows.
    for (int p=0; p<6; p++) {
      double product = 1.0;
      for (int leg=0; leg<3; leg++) {
        product *= work->pair[((size_t)a[leg]*nf
          + b[permutations[p][leg]])*nr + r];
      }
      density += product;
    }

    // Nine one-cross choices survive only under a finite window; the
    // compensated filters remove them in the large-area limit. Each
    // estimator closes one internal pair at K(0), and one cross K(r)
    // joins the two remaining legs.
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

  // BB term: pair the two B factors over the nine (p, q) splits into
  // field triples. The large-area branch contracts the S tables
  // directly in Fourier space; the finite window contracts the
  // real-space B tables over the center offset.
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



// ============================================================================
// [SECTION] DRIVER: VALIDATION, DISPATCH AND SYMMETRIZATION
// ============================================================================

// Free every workspace table. Pointers an option never allocated are
// still NULL from the initial memset, and free(NULL) does nothing.
static void covariance_release_3pcf(
  covariance_work_3pcf* work // workspace whose tables are freed
)
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


// Public driver: validate the request, prepare the workspace, fill the
// mean and the six covariance term slabs, and symmetrize. The refusal
// checks below name the estimators this implementation does not cover,
// so a caller learns which physical assumption failed rather than
// receiving a silently wrong matrix.
void covariance_physical_3pcf(
  const physics_model_3pcf* model, // immutable matter/source model
  const covariance_input_3pcf* input, // physical and numerical definitions
  double* mean, // output [nrow]
  double* terms // output [6,nrow,nrow]
)
{
  int pixel_count;
  physics_window_samples_3pcf(model, &pixel_count);
  if (!(pixel_count == 0)) {
    log_fatal("physical 3pcf covariance: pixel-smoothed means require a "
      "separately matched covariance");
    exit(1);
  }
  if (!(input->estimator == 0)) {
    log_fatal("physical 3pcf covariance: finite-triangle or catalogue-mask "
      "covariance is not this estimator");
    exit(1);
  }
  if (!(input->ia_model == 0)) {
    log_fatal("physical 3pcf covariance: nonlinear IA connected covariance "
      "requires additional cumulants");
    exit(1);
  }
  if (!(input->noise_mode == 0
        || input->noise_mode == 1)) {
    log_fatal("physical 3pcf covariance: noise_mode must select finite-band "
      "(0) or uniform-area xi noise (1)");
    exit(1);
  }
  if (!(input->noise_mode == 0
        || (input->large_area == 1 && input->theta_edges != NULL))) {
    log_fatal("physical 3pcf covariance: analytic xi noise requires "
      "large_area=1 and exact theta_edges");
    exit(1);
  }
  if (!(input->center_window == 1
        || input->center_window == 2)) {
    log_fatal("physical 3pcf covariance: choose a circle or square center "
      "window");
    exit(1);
  }
  if (!(input->nell > 0
        && input->nwindow > 0
        && input->nphi > 0
        && input->nr > 0
        && input->nprofile >= 3
        && input->nk > 0
        && input->nmass > 0
        && input->nfield > 0
        && input->nrow > 0
        && input->ntheta > 0
        && input->area > 0.0)) {
    log_fatal("physical 3pcf covariance: invalid covariance dimensions or "
      "area");
    exit(1);
  }

  const int no = input->nrow; // covariance rows
  const int nl = input->nell; // Fourier nodes
  const size_t matrix = covariance_product_3pcf(no, no);
  const size_t term_count = covariance_product_3pcf(6, matrix);
  memset(terms, 0, sizeof(double)*term_count);

  // Check every row's field roles before any field offset is built: an
  // xi leg must be unfiltered (R = 0), and an aperture leg needs a
  // positive filter radius.
  for (int row=0; row<no; row++) {
    const int kind = input->row_kind[row];
    if (!(kind >= 0
          && kind <= 3)) {
      log_fatal("physical 3pcf covariance: invalid row kind");
      exit(1);
    }
    const int nleg = kind == 1 ? 3 : 2;
    for (int leg=0; leg<nleg; leg++) {
      const int field = input->row_fields[(size_t)row*3 + leg];
      if (!(field >= 0
            && field < input->nfield)) {
        log_fatal("physical 3pcf covariance: row field index out of range");
        exit(1);
      }
      if (!(kind >= 2 ? input->field_radius[field] == 0.0
            : input->field_radius[field] > 0.0)) {
        log_fatal("physical 3pcf covariance: xi legs require R=0; aperture "
          "legs require positive R");
        exit(1);
      }
    }
    if (kind >= 2) {
      if (input->noise_mode == 1) {
        const double lower = input->theta_edges[(size_t)row*2];
        const double upper = input->theta_edges[(size_t)row*2 + 1];
        if (!(isfinite(lower)
              && isfinite(upper)
              && lower > 0.0
              && upper > lower)) {
          log_fatal("physical 3pcf covariance: analytic xi noise requires "
            "positive ordered bin edges");
          exit(1);
        }
        for (int t=0; t<input->ntheta; t++) {
          const double theta = input->theta[(size_t)row*input->ntheta + t];
          if (!(theta >= lower
                && theta <= upper)) {
            log_fatal("physical 3pcf covariance: xi quadrature node lies "
              "outside declared analytic-noise bin");
            exit(1);
          }
        }
      }
      double weight = 0.0;
      for (int t=0; t<input->ntheta; t++) {
        const size_t index = (size_t)row*input->ntheta + t;
        if (!(input->theta[index] > 0.0
              && input->theta_weight[index] >= 0.0)) {
          log_fatal("physical 3pcf covariance: xi separations must be "
            "positive and bin weights nonnegative");
          exit(1);
        }
        if (!(input->profile_radius[input->nprofile - 1]
              >= input->radius[input->nr - 1] + 0.5*input->theta[index])) {
          log_fatal("physical 3pcf covariance: profile_radius must cover "
            "radius_max + theta_max/2");
          exit(1);
        }
        weight += input->theta_weight[index];
      }
      if (!(fabs(weight - 1.0) < 1.0e-10)) {
        log_fatal("physical 3pcf covariance: xi bin weights must sum to one");
        exit(1);
      }
    }
  }

  covariance_work_3pcf work;
  memset(&work, 0, sizeof(work));
  work.model = model;
  work.in = input;
  covariance_prepare_3pcf(&work);

  // Fill the mean of every row; one worker owns a row. The means omit
  // the known additive Gaussian-noise bias of the quadratic rows:
  // subtracting a deterministic constant leaves every covariance term
  // unchanged.
  #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
  for (int row=0; row<no; row++) {
    const int* fields = input->row_fields + (size_t)row*3;
    double value = 0.0;
    if (input->row_kind[row] == 1) {
      // A cubic estimator is symmetric under permuting its three legs.
      // Averaging the three choices of separated leg keeps that
      // symmetry exact under a finite numerical quadrature.
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

  // Dispatch the disconnected terms; one worker owns an upper-triangle
  // row. Two cubic rows fill the Gaussian and BB slabs; two quadratic
  // rows fill the Gaussian slab; a mixed cell fills only P*B, because
  // the three-field cumulant of zero-mean Gaussian fields vanishes
  // identically, so a quadratic and a cubic estimator have no Gaussian
  // covariance.
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

  // Every term was filled on the upper triangle only. Check each cell
  // is finite - one invalid physical table would otherwise propagate
  // silently - and mirror it, so each slab returns as a full symmetric
  // matrix.
  for (int term=0; term<6; term++) {
    for (int i=0; i<no; i++) {
      for (int j=i; j<no; j++) {
        const double value = terms[(size_t)term*matrix + (size_t)i*no + j];
        if (!(isfinite(value))) {
          log_fatal("physical 3pcf covariance: nonfinite covariance term; "
            "inspect physical inputs and grids");
          exit(1);
        }
        terms[(size_t)term*matrix + (size_t)j*no + i] = value;
      }
    }
  }

  covariance_release_3pcf(&work);
}
