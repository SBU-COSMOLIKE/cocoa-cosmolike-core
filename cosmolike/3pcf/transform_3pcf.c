#include <complex.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "simde/x86/sse2.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "aperture_3pcf.h"
#include "coupling_3pcf.h"
#include "fftlog_3pcf.h"
#include "transform_3pcf.h"

// Natural shear components follow Sunao Sugiyama's fastnc, MIT revision
// 7b2336ce37a909bed51f07f32a2a4f096e71b6ce, fastnc.py:319-409.
// See LICENSE.fastnc-TreeCorr.txt. The physical multipoles are supplied;
// this file owns only angular coupling, transforms and projection phases.
//
// PHYSICAL DERIVATION & LOGIC FLOW
// H_M(ell1,ell2)=sum_L (-1)^L G_LM(psi) b_L(ell1,ell2), where
// tan(psi)=ell2/ell1. Four pairs of Bessel orders turn H_M into four
// natural shear components. One forward FFT of H_M ell1^2 ell2^2 serves
// all four inverse transforms. A finite sum exp(i M phi)/(2 pi) then
// reconstructs the opening-angle dependence. Ordered tomographic TATT
// legs need independent positive and negative M transforms: exchanging
// ell1 and ell2 without exchanging source labels changes their bispectrum.
// A historical transpose shortcut is retained only as an explicit option.
//
// Cache invalidation and ownership:
// All geometric arrays belong to this immutable plan. A different grid,
// multipole cutoff, worker capacity or numerical setting needs a new plan.
// Only multipoles change between calls. No global cosmology state or
// initial_setup hook is involved. The same plan is not reentrant: the
// binding retains Python's GIL and one call owns its scratch until return.
// FFTW plans are prepared before OpenMP; workers execute one-thread FFTs.

struct transform_plan_3pcf {
  int nell;
  int ntheta;
  int nphi;
  int lmax;
  int mmax;
  int workers;
  int fft_workers; // no more scratch slices than independent signed M jobs
  int projection;
  int convention;
  int ordered_legs;
  int* indices; // radial selection copied from the caller
  double* ell2; // [nell], squared Fourier radii
  double* coupling; // [M,L,2*nell-1], ratio depends on j-i only
  double* modes; // [signed M,4,ntheta,ntheta], or historical positive M
  double* phases; // [4,triangle,2], real/imag centroid rotations
  double* angles; // [M,nphi,2], cos/sin(M phi)
  double* gamma; // [2,4,triangle], aperture-call temporary
  fftlog_workspace_3pcf* fft;
};


static void require_transform_3pcf(
  const int condition, // required domain or size condition
  const char* message // diagnostic
)
{
  if (!condition) {
    fprintf(stderr, "transform_3pcf: %s\n", message);
    exit(EXIT_FAILURE);
  }
}


static void* allocate_transform_3pcf(
  const size_t count, // number of elements
  const size_t size // element size
)
{
  require_transform_3pcf(count <= SIZE_MAX/size,
      "allocation size overflow");
  void* result = calloc(count, size);
  require_transform_3pcf(result != NULL, "allocation failed");
  return result;
}


static double complex spin_phase_3pcf(
  const double complex side // centroid direction, arbitrarily scaled
)
{
  const double length = cabs(side);
  require_transform_3pcf(isfinite(length) && length > 0.0,
      "centroid direction is not representable");
  const double complex unit = side/length;
  return unit*unit;
}


transform_plan_3pcf* transform_create_3pcf(
  const int nell, // even Fourier grid length
  const double* ell, // [nell], logarithmic grid
  const int ntheta, // selected radial bins
  const int* indices, // [ntheta], radial selection
  const int nphi, // opening-angle samples
  const double* phi, // [nphi], radians
  const int lmax, // maximum Legendre degree
  const int mmax, // maximum Fourier mode
  const int workers, // private worker capacity
  const double nu1, // first bias
  const double nu2, // second bias
  const double xy, // real/Fourier grid alignment
  const double dlnt, // logarithmic radial-bin width
  const int padding, // zero padding per end
  const int window, // apply high-frequency taper
  const double window_width, // taper fraction
  const int npsi, // angular interpolation grid size
  const double dphi, // opening-angle width; zero is historical point rule
  const int projection, // X, TATT centroid, or main centroid
  const int convention, // fastnc or TreeCorr component order/sign
  const int ordered_legs // general ordered bins or historical shortcut
)
{
  require_transform_3pcf(nell >= 2 && nell <= (INT_MAX-1)/2
      && ntheta > 0 && nphi > 0 && lmax >= 0 && lmax < INT_MAX-1
      && mmax >= 0 && lmax <= INT_MAX-1-mmax
      && mmax < INT_MAX-3 && workers > 0 && ell != NULL
      && indices != NULL && phi != NULL, "invalid dimensions");
  require_transform_3pcf(ntheta <= INT_MAX/ntheta
      && ntheta*ntheta <= INT_MAX/nphi, "too many triangles");
  require_transform_3pcf(projection >= 0 && projection <= 2
      && convention >= 0 && convention <= 1
      && ordered_legs >= 0 && ordered_legs <= 1
      && mmax < (INT_MAX-1)/2,
      "unknown projection or component convention");
  require_transform_3pcf(isfinite(dphi) && dphi >= 0.0
      && dphi <= acos(-1.0), "invalid opening-angle bin width");
  require_transform_3pcf(npsi >= 0 && npsi != 1,
      "coupling grid needs zero or at least two points");
  transform_plan_3pcf* plan = allocate_transform_3pcf(1, sizeof(*plan));
  plan->nell = nell;
  plan->ntheta = ntheta;
  plan->nphi = nphi;
  plan->lmax = lmax;
  plan->mmax = mmax;
  plan->workers = workers;
  plan->projection = projection;
  plan->convention = convention;
  plan->ordered_legs = ordered_legs;
  const int nmodes = ordered_legs ? 2*mmax+1 : mmax+1;
  plan->fft_workers = workers < nmodes ? workers : nmodes;
  const size_t nratio = 2*(size_t)nell-1;
  const size_t radial_pairs = (size_t)ntheta*ntheta;
  const size_t triangles = radial_pairs*nphi;
  plan->workers = workers < (int)triangles ? workers : (int)triangles;
  require_transform_3pcf((size_t)(mmax+1) <= SIZE_MAX/(lmax+1)/nratio,
      "coupling size overflow");
  require_transform_3pcf((size_t)nmodes <= SIZE_MAX/4/radial_pairs,
      "mode size overflow");
  plan->indices = allocate_transform_3pcf(ntheta, sizeof(int));
  plan->ell2 = allocate_transform_3pcf(nell, sizeof(double));
  plan->coupling = allocate_transform_3pcf(
      (size_t)(mmax+1)*(lmax+1)*nratio, sizeof(double));
  plan->modes = allocate_transform_3pcf(
      (size_t)nmodes*4*radial_pairs, sizeof(double));
  plan->phases = allocate_transform_3pcf(8*triangles, sizeof(double));
  plan->angles = allocate_transform_3pcf(
      (size_t)(mmax+1)*nphi*2, sizeof(double));
  plan->gamma = allocate_transform_3pcf(8*triangles, sizeof(double));
  for (int index=0; index<ntheta; index++) {
    require_transform_3pcf(indices[index] >= 0 && indices[index] < nell,
        "radial index outside Fourier grid");
    if (index > 0) {
      require_transform_3pcf(indices[index] > indices[index-1],
          "radial indices must strictly increase");
    }
    plan->indices[index] = indices[index];
  }
  // The FFT workspace checks positivity and uniform log spacing before
  // any grid-dependent calculation below. It also creates every plan.
  plan->fft = fftlog_create_3pcf(nell, ell, mmax+3, plan->fft_workers,
      nu1, nu2, xy, dlnt, padding, window, window_width);
  fftlog_select_3pcf(plan->fft, ntheta, plan->indices);
  for (int index=0; index<nell; index++) {
    plan->ell2[index] = ell[index]*ell[index];
    require_transform_3pcf(isfinite(plan->ell2[index]),
        "squared Fourier radius overflow");
  }
  double* psi = allocate_transform_3pcf(nratio, sizeof(double));
  for (int difference=1-nell; difference<nell; difference++) {
    const int first = difference < 0 ? -difference : 0;
    const int second = first+difference;
    psi[difference+nell-1] = atan2(ell[second], ell[first]);
  }
  legendre_coupling_3pcf(lmax, mmax, (int)nratio, psi, npsi,
      plan->coupling);
  free(psi);
  for (int angle=0; angle<nphi; angle++) {
    require_transform_3pcf(isfinite(phi[angle]) && phi[angle] > 0.0
        && phi[angle] < acos(-1.0), "phi must lie in (0,pi)");
    for (int mode=0; mode<=mmax; mode++) {
      const size_t offset = ((size_t)mode*nphi+angle)*2;
      const double half_width = 0.5*mode*dphi;
      const double average = half_width == 0.0
          ? 1.0 : sin(half_width)/half_width;
      // This averages the X-projected Fourier series. A later centroid
      // phase still uses the bin center, as specified by this API; it is
      // not an exact average of the varying centroid-projected shear.
      plan->angles[offset] = average*cos(mode*phi[angle]);
      plan->angles[offset+1] = average*sin(mode*phi[angle]);
    }
  }
  // Projection is geometric, so store it once. Scaling both sides by
  // their maximum avoids squaring large or tiny angular units. TATT's
  // historical phase and main's conjugate are explicit different options.
  for (int first=0; first<ntheta; first++) {
    const double theta1 = xy/ell[nell-1-indices[first]];
    for (int second=0; second<ntheta; second++) {
      const double theta2 = xy/ell[nell-1-indices[second]];
      const double scale = fmax(theta1, theta2);
      for (int angle=0; angle<nphi; angle++) {
        const double complex side1 = theta1/scale;
        const double complex side2 = theta2/scale
            *(cos(phi[angle])-I*sin(phi[angle]));
        const double complex phase1 = spin_phase_3pcf(side1+side2);
        const double complex phase2 = spin_phase_3pcf(-2*side1+side2);
        const double complex phase3 = spin_phase_3pcf(side1-2*side2);
        const double complex rotation = cexp(I*phi[angle]);
        const double complex rotation3 = rotation*rotation*rotation;
        double complex phases[4] = {
          phase1*phase2*phase3*rotation3,
          conj(phase1)*phase2*phase3*rotation,
          phase1*conj(phase2)*phase3*rotation3,
          phase1*phase2*conj(phase3)/rotation
        };
        const size_t triangle =
            ((size_t)first*ntheta+second)*nphi+angle;
        for (int component=0; component<4; component++) {
          if (projection == 0) {
            phases[component] = 1.0;
          } else if (projection == 2) {
            phases[component] = conj(phases[component]);
          }
          const size_t offset = ((size_t)component*triangles+triangle)*2;
          plan->phases[offset] = creal(phases[component]);
          plan->phases[offset+1] = cimag(phases[component]);
        }
      }
    }
  }
  return plan;
}


void transform_destroy_3pcf(
  transform_plan_3pcf* plan // owning plan
)
{
  if (plan == NULL) {
    return;
  }
  fftlog_destroy_3pcf(plan->fft);
  free(plan->indices);
  free(plan->ell2);
  free(plan->coupling);
  free(plan->modes);
  free(plan->phases);
  free(plan->angles);
  free(plan->gamma);
  free(plan);
}


void transform_natural_3pcf(
  transform_plan_3pcf* plan, // non-reentrant owning geometry plan
  const double* multipoles, // [L,nell,nell], minus-cosine convention
  double* gamma_real, // [component,theta1,theta2,phi]
  double* gamma_imag // same shape
)
{
  require_transform_3pcf(plan != NULL && multipoles != NULL
      && gamma_real != NULL && gamma_imag != NULL, "null array");
  const int nell = plan->nell;
  const int ntheta = plan->ntheta;
  const int nphi = plan->nphi;
  const size_t nratio = 2*(size_t)nell-1;
  const size_t radial_pairs = (size_t)ntheta*ntheta;
  const size_t triangles = radial_pairs*nphi;
  const double two_pi = 2.0*acos(-1.0);
  const double normalization = 1.0/pow(two_pi, 4);
  const int nmodes = plan->ordered_legs ? 2*plan->mmax+1 : plan->mmax+1;

  // Each M owns its scratch and completed four-component output block.
  // L sums keep their prescribed order. No OpenMP reduction changes a
  // cancellation-sensitive multipole sum when the worker count changes.
  #pragma omp parallel for num_threads(plan->fft_workers) \
      if(nmodes > 1) schedule(static)
  for (int task=0; task<nmodes; task++) {
    const int mode = plan->ordered_legs ? task-plan->mmax : task;
    int worker = 0;
    #ifdef _OPENMP
    worker = omp_get_thread_num();
    #endif
    double* input = fftlog_input_3pcf(plan->fft, worker);
    for (int first=0; first<nell; first++) {
      // Adjacent second-leg samples occupy the two SIMD lanes. Their L
      // sums advance together without horizontal reductions or reassoci-
      // ation. Even nell means every row ends on a full vector. This uses
      // contiguous coupling and multipole loads, not vector gathers.
      for (int second=0; second<nell; second+=2) {
        const size_t cell = (size_t)first*nell+second;
        const size_t ratio = second-first+nell-1;
        simde__m128d sum = simde_mm_setzero_pd();
        for (int degree=0; degree<=plan->lmax; degree++) {
          const size_t offset =
              ((size_t)abs(mode)*(plan->lmax+1)+degree)*nratio;
          const double sign = degree%2 == 0 ? 1.0 : -1.0;
          simde__m128d coupling;
          if (mode >= 0) {
            coupling = simde_mm_loadu_pd(plan->coupling+offset+ratio);
          } else {
            // G_L,-M(psi)=G_L,M(pi/2-psi). Adjacent ell2 values then
            // traverse the saved ratio table backwards; load the pair
            // contiguously and exchange lanes. b_L is not transposed.
            coupling = simde_mm_loadu_pd(
                plan->coupling+offset+nratio-2-ratio);
            coupling = simde_mm_shuffle_pd(coupling, coupling, 1);
          }
          const simde__m128d values = simde_mm_loadu_pd(
              multipoles+(size_t)degree*nell*nell+cell);
          const simde__m128d signed_coupling =
              simde_mm_mul_pd(simde_mm_set1_pd(sign), coupling);
          sum = simde_mm_add_pd(sum,
              simde_mm_mul_pd(signed_coupling, values));
        }
        const simde__m128d weighted_first =
            simde_mm_mul_pd(sum, simde_mm_set1_pd(plan->ell2[first]));
        const simde__m128d weighted_second = simde_mm_mul_pd(
            weighted_first, simde_mm_loadu_pd(plan->ell2+second));
        simde_mm_storeu_pd(input+cell, weighted_second);
      }
    }
    fftlog_forward_3pcf(plan->fft, worker);
    const int order1[4] = {mode-3, -mode-1, mode+1, mode-3};
    const int order2[4] = {-mode-3, mode-1, -mode-3, -mode+1};
    const double* completed[4] = {NULL, NULL, NULL, NULL};
    for (int component=0; component<4; component++) {
      const int first_order = order1[component];
      const int second_order = order2[component];
      const double sign1 = first_order < 0 && abs(first_order)%2
          ? -1.0 : 1.0;
      const double sign2 = second_order < 0 && abs(second_order)%2
          ? -1.0 : 1.0;
      // Conjugating the shear at vertex 1 reverses the angular coupling
      // phase. Its radial transform of H_M belongs to Gamma1,-M; the
      // other three transformed components belong to Gamma_mu,M.
      int destination = task;
      if (plan->ordered_legs && component == 1) {
        destination = plan->mmax-mode;
      }
      double* target =
          plan->modes+((size_t)destination*4+component)*radial_pairs;
      int reuse = -1;
      for (int previous=0; previous<component; previous++) {
        if (order1[previous] == first_order
            && order2[previous] == second_order) {
          reuse = previous;
        }
      }
      // At M=+/-1 two component pairs have identical Bessel orders up
      // to even-order signs. Reuse their retained output values. This is
      // an identity of the kernels and does not exchange physical legs.
      if (abs(mode) == 1) {
        for (int previous=0; previous<component; previous++) {
          if (abs(order1[previous]) == abs(first_order)
              && abs(order2[previous]) == abs(second_order)) {
            reuse = previous;
          }
        }
      }
      if (reuse >= 0) {
        for (size_t cell=0; cell<radial_pairs; cell++) {
          target[cell] = completed[reuse][cell];
        }
      } else {
        const double* transformed = fftlog_inverse_3pcf(
            plan->fft, worker, abs(first_order), abs(second_order));
        for (size_t cell=0; cell<radial_pairs; cell++) {
          target[cell] = normalization*sign1*sign2*transformed[cell];
        }
      }
      completed[component] = target;
    }
  }

  #pragma omp parallel for num_threads(plan->workers) if(triangles > 1) \
      schedule(static)
  for (size_t triangle=0; triangle<triangles; triangle++) {
    const size_t pair = triangle/nphi;
    const int angle = triangle%nphi;
    const size_t transpose = (pair%ntheta)*ntheta+pair/ntheta;
    double real[4] = {0.0, 0.0, 0.0, 0.0};
    double imag[4] = {0.0, 0.0, 0.0, 0.0};
    if (plan->ordered_legs) {
      for (int mode=-plan->mmax; mode<=plan->mmax; mode++) {
        const size_t task = mode+plan->mmax;
        const double* values = plan->modes+task*4*radial_pairs;
        const size_t offset = ((size_t)abs(mode)*nphi+angle)*2;
        const double sign = mode < 0 ? -1.0 : 1.0;
        for (int component=0; component<4; component++) {
          real[component] +=
              values[(size_t)component*radial_pairs+pair]
              *plan->angles[offset];
          imag[component] +=
              values[(size_t)component*radial_pairs+pair]
              *sign*plan->angles[offset+1];
        }
      }
    } else {
      real[0] = plan->modes[transpose];
      real[1] = plan->modes[radial_pairs+pair];
      real[2] = plan->modes[3*radial_pairs+transpose];
      real[3] = plan->modes[3*radial_pairs+pair];
      for (int mode=1; mode<=plan->mmax; mode++) {
        const double* values = plan->modes+(size_t)mode*4*radial_pairs;
        const double positive[4] = {values[pair],
          values[radial_pairs+transpose], values[2*radial_pairs+pair],
          values[3*radial_pairs+pair]};
        const double negative[4] = {values[transpose],
          values[radial_pairs+pair], values[3*radial_pairs+transpose],
          values[2*radial_pairs+transpose]};
        const size_t offset = ((size_t)mode*nphi+angle)*2;
        for (int component=0; component<4; component++) {
          real[component] += (positive[component]+negative[component])
              *plan->angles[offset];
          imag[component] += (positive[component]-negative[component])
              *plan->angles[offset+1];
        }
      }
    }
    const int permutation[4] = {0, 1, 3, 2};
    for (int output=0; output<4; output++) {
      const int component = plan->convention ? permutation[output] : output;
      const double sign = plan->convention ? -1.0 : 1.0;
      const size_t phase = ((size_t)component*triangles+triangle)*2;
      const double cosine = plan->phases[phase];
      const double sine = plan->phases[phase+1];
      gamma_real[(size_t)output*triangles+triangle] = sign
          *(real[component]*cosine-imag[component]*sine);
      gamma_imag[(size_t)output*triangles+triangle] = sign
          *(real[component]*sine+imag[component]*cosine);
    }
  }
}


void transform_aperture_3pcf(
  transform_plan_3pcf* plan, // owning centroid-projection plan
  const double* multipoles, // supplied bispectrum multipoles
  const int nfilter, // ordered aperture triplets
  const double* weights_real, // TreeCorr real weights
  const double* weights_imag, // TreeCorr imaginary weights
  double* output // real aperture moments
)
{
  require_transform_3pcf(plan != NULL && plan->projection != 0,
      "aperture contraction requires centroid projection");
  const int triangles = plan->ntheta*plan->ntheta*plan->nphi;
  double* real = plan->gamma;
  double* imag = plan->gamma+4*(size_t)triangles;
  transform_natural_3pcf(plan, multipoles, real, imag);
  if (plan->convention == 0) {
    // TreeCorr uses -[Gamma0,Gamma1,Gamma3,Gamma2]. Convert in place;
    // both members of the exchanged pair are read before either is set.
    for (int index=0; index<triangles; index++) {
      for (int part=0; part<2; part++) {
        double* values = part == 0 ? real : imag;
        values[index] = -values[index];
        values[(size_t)triangles+index] =
            -values[(size_t)triangles+index];
        const double third = values[2*(size_t)triangles+index];
        values[2*(size_t)triangles+index] =
            -values[3*(size_t)triangles+index];
        values[3*(size_t)triangles+index] = -third;
      }
    }
  }
  aperture_apply_3pcf(nfilter, triangles, 1, weights_real, weights_imag,
      real, imag, output);
}
