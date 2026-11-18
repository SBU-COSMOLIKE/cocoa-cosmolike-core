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
#include "cosmolike/basics.h"
#include "log.c/src/log.h"

// Natural components of the shear three-point correlation function.
// The code follows Sunao Sugiyama's fastnc, MIT revision
// 7b2336ce37a909bed51f07f32a2a4f096e71b6ce, fastnc.py:319-409; see
// LICENSE.fastnc-TreeCorr.txt for the distributed source notices. The
// method is the multipole expansion of the fastnc paper by Sugiyama,
// Gomes & Jarvis (arXiv:2407.01798). The caller computes the physical
// bispectrum multipoles; this file owns only the angular coupling, the
// radial transforms and the projection phases.
//
// PHYSICAL DERIVATION & LOGIC FLOW
// The natural components Gamma_0..Gamma_3 are the four complex
// three-shear correlators of Schneider & Lombardi
// (arXiv:astro-ph/0207454): a rotation only multiplies each one by a
// phase, so they carry the full three-point information without
// mixing. Gamma_0 conjugates no shear; Gamma_i conjugates the shear
// at vertex i. A triangle is labeled by two side lengths theta1 and
// theta2 and the opening angle phi between them. The pipeline from
// bispectrum to Gamma runs in four steps.
//
// 1. Legendre multipoles of the bispectrum. The projected E-mode
//    bispectrum depends on its two Fourier legs ell1 and ell2 and on
//    mu, the cosine of the angle between them in the triangle
//    convention ell3^2 = ell1^2 + ell2^2 - 2*ell1*ell2*mu. The caller
//    expands it in Legendre polynomials P_L(mu); the coefficients
//    b_L(ell1,ell2) arrive as this file's input.
// 2. Angular coupling. Integrating out the Fourier-plane angles
//    couples each real-space multipole index M to every Legendre
//    degree L:
//
//      H_M(ell1,ell2) = sum_L (-1)^L G_LM(psi) b_L(ell1,ell2),
//
//    where psi = arctan(ell2/ell1) is the opening angle in [0,pi/2]
//    that encodes the ratio of the two multipole legs, and G_LM(psi)
//    is the geometric coupling function of coupling_3pcf.c. The
//    coupling integral runs over the angle between the two Fourier
//    wave vectors, which is pi minus the angle that mu describes, and
//    the Legendre parity P_L(-x) = (-1)^L P_L(x) supplies the factor
//    (-1)^L of that conversion.
// 3. Radial double-Bessel transforms. Four pairs of cylindrical
//    Bessel orders turn H_M into the four component multipoles
//    Gamma_mu^(M)(theta1,theta2); one forward FFT of the weighted
//    kernel ell1^2 ell2^2 H_M serves all four inverse transforms
//    (fftlog_3pcf.c). The order table inside transform_natural_3pcf
//    lists the pairs.
// 4. Opening-angle resummation. The finite Fourier sum
//
//      Gamma_mu = (1/(2*pi)) sum_M Gamma_mu^(M) exp(i*M*phi)
//
//    restores the phi dependence on every triangle.
//
// Ordered tomographic legs need independent positive and negative M
// transforms: exchanging ell1 and ell2 without exchanging the source
// labels changes a tomographic TATT bispectrum, so its b_L lacks the
// leg symmetry that ties -M to +M. The fastnc main-branch transpose
// shortcut, which rebuilds the negative-M terms from transposed
// positive-M values, stays available as an explicit option for b_L
// symmetric under leg exchange.
//
// Cache invalidation and ownership:
// All geometric arrays belong to this immutable plan; a different
// grid, multipole cutoff, worker capacity or numerical setting needs
// a new plan. Only the multipoles change between calls, and the plan
// stores no global cosmology state and no initial_setup hook. The
// plan is not reentrant: the Python binding keeps the interpreter
// lock, and one call owns the plan's scratch until it returns. FFTW
// plans exist before any OpenMP region, and each worker executes
// one-thread FFTs.

struct transform_plan_3pcf {
  int nell; // Fourier grid length per axis, even
  int ntheta; // number of radial bins the plan keeps per axis
  int nphi; // number of opening-angle samples
  int lmax; // highest Legendre degree L of the input multipoles
  int mmax; // highest real-space multipole index M in the resummation
  int workers; // worker capacity of the triangle loop
  int fft_workers; // no more scratch slices than independent signed M jobs
  int projection; // 0 x projection, 1 or 2 centroid (see create)
  int convention; // 0 fastnc order, 1 TreeCorr order and sign
  int ordered_legs; // 1 general b_L; 0 fastnc main-branch shortcut
  int* indices; // [ntheta], the plan's copy of the radial selection
  double* ell; // [nell], exact Fourier nodes for prepared-grid queries
  double* ell2; // [nell], squared Fourier radii
  double* coupling; // [M,L,2*nell-1], (-1)^L G_LM per leg-index difference
  double* modes; // [signed M,4,ntheta,ntheta]; the shortcut stores M>=0
  double* phases; // [4,triangle,2], real/imag centroid rotations
  double* angles; // [M,nphi,2], bin-averaged cos/sin(M phi)
  double* gamma; // [2,4,triangle], scratch for the aperture entry point
  fftlog_workspace_3pcf* fft; // radial double-Bessel engine, plan-owned
  physics_grid_query_3pcf* query; // owned axis-keyed geometry, allocated on use
  double* grid_multipoles; // [L,nell,nell], refilled for every physical grid
};


// ---------------------------------------------------------------------------
// Squared unit phase of one centroid-to-vertex direction.
//
// A shear is a spin-2 quantity: projecting it along a direction at
// angle beta from the x axis multiplies it by exp(-2i*beta), so
// converting an x-projected shear to this direction costs the phase
// exp(2i*beta). Dividing the complex side by its length gives the
// unit direction exp(i*beta); squaring that unit gives exactly the
// spin-2 phase. The length drops out, which is why callers may scale
// the direction arbitrarily, and an overall sign of the direction
// drops out with it. A zero or nonfinite length leaves the direction
// undefined, so the helper stops instead of guessing one.
// ---------------------------------------------------------------------------
static double complex spin_phase_3pcf(
  const double complex side // centroid direction, arbitrarily scaled
)
{
  const double length = cabs(side);
  if (!(isfinite(length) && length > 0.0)) {
    log_fatal("transform_3pcf: centroid direction is not representable");
    exit(1);
  }

  const double complex unit = side/length;
  return unit*unit;
}


// ---------------------------------------------------------------------------
// Build one immutable geometry plan for the natural-component
// transform.
//
// The constructor runs six stages, marked by banners below:
// validation; sizes and allocation; the radial selection and the FFT
// workspace; the angular coupling in psi; the opening-angle Fourier
// table; and the projection phases. Everything here depends only on
// geometry and numerical settings - no cosmology enters - so one plan
// serves every likelihood evaluation on this triangle grid.
//
// Inputs are described per parameter below; transform_3pcf.h states
// the input multipole convention. Returns the plan; every failure
// aborts with a message.
// Ownership: the plan copies all geometry, and transform_destroy_3pcf
// frees it.
// Thread safety: creation must stay serial because FFTW planning
// mutates global planner state; afterward each transform call owns
// the plan's scratch until it returns.
// ---------------------------------------------------------------------------
transform_plan_3pcf* transform_create_3pcf(
  const int nell, // even Fourier grid length
  const double* ell, // [nell], logarithmic grid
  const int ntheta, // number of kept radial bins
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
  const double dphi, // opening-angle width; zero keeps point values
  const int projection, // x, TATT-comparison centroid, or its conjugate
  const int convention, // fastnc or TreeCorr component order/sign
  const int ordered_legs // general ordered bins or symmetric shortcut
)
{
  // --- 1. VALIDATION ---
  // Reject dimensions whose products could overflow the index
  // arithmetic below, angles outside (0,pi), and settings without a
  // defined meaning. The checks run before the first allocation so a
  // bad call never leaks memory.
  if (!(nell >= 2 && nell <= (INT_MAX-1)/2
        && ntheta > 0 && nphi > 0 && lmax >= 0 && lmax < INT_MAX-1
        && mmax >= 0 && lmax <= INT_MAX-1-mmax
        && mmax < INT_MAX-3 && workers > 0 && ell != NULL
        && indices != NULL && phi != NULL)) {
    log_fatal("transform_3pcf: invalid dimensions");
    exit(1);
  }
  if (!(ntheta <= INT_MAX/ntheta
        && ntheta*ntheta <= INT_MAX/nphi)) {
    log_fatal("transform_3pcf: too many triangles");
    exit(1);
  }
  if (!(projection >= 0 && projection <= 2
        && convention >= 0 && convention <= 1
        && ordered_legs >= 0 && ordered_legs <= 1
        && mmax < (INT_MAX-1)/2)) {
    log_fatal("transform_3pcf: unknown projection or component convention");
    exit(1);
  }
  if (!(isfinite(dphi) && dphi >= 0.0
        && dphi <= M_PI)) {
    log_fatal("transform_3pcf: invalid opening-angle bin width");
    exit(1);
  }

  // One grid point cannot define an interpolation segment, so the
  // coupling accepts zero (exact evaluation) or at least two points.
  if (!(npsi >= 0 && npsi != 1)) {
    log_fatal("transform_3pcf: coupling grid needs zero or at least two "
      "points");
    exit(1);
  }

  // --- 2. SIZES AND ALLOCATION ---
  // Worker caps: the multipole stage runs one job per signed mode, so
  // extra FFT scratch slices beyond nmodes would sit idle, and the
  // triangle stage never needs more workers than triangles. The
  // coupling table spans the 2*nell-1 leg-index differences explained
  // at stage 4. Every product is overflow-checked before use.
  transform_plan_3pcf* plan = calloc(1, sizeof(*plan));
  if (plan == NULL) {
    log_fatal("transform_3pcf: allocation failed");
    exit(1);
  }

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

  if (!(triangles <= INT_MAX/8)) {
    log_fatal("transform_3pcf: too many triangles");
    exit(1);
  }
  if (!((size_t)(mmax+1) <= (size_t)INT_MAX/(lmax+1)/nratio)) {
    log_fatal("transform_3pcf: coupling size overflow");
    exit(1);
  }
  if (!((size_t)nmodes <= (size_t)INT_MAX/4/radial_pairs)) {
    log_fatal("transform_3pcf: mode size overflow");
    exit(1);
  }
  if (!((size_t)(mmax+1)*nphi <= (size_t)INT_MAX/2)) {
    log_fatal("transform_3pcf: angular phase table is too large");
    exit(1);
  }

  plan->indices = malloc1d_int(ntheta);
  plan->ell = calloc1d(nell);
  plan->ell2 = calloc1d(nell);
  plan->coupling = calloc1d((int)((size_t)(mmax+1)*(lmax+1)*nratio));
  plan->modes = calloc1d((int)((size_t)nmodes*4*radial_pairs));
  plan->phases = calloc1d((int)(8*triangles));
  plan->angles = calloc1d((int)((size_t)(mmax+1)*nphi*2));
  plan->gamma = calloc1d((int)(8*triangles));

  // --- 3. RADIAL SELECTION AND FFT WORKSPACE ---
  // Copy the caller's radial selection, then build the double-Bessel
  // engine. The Bessel orders of the four natural components are
  // M-3, -M-1, M+1 and their partners (transform_natural_3pcf), so
  // over all |M| <= mmax the largest magnitude is mmax+3 and the
  // workspace prepares kernels up to that order. The workspace also
  // checks grid positivity and uniform log spacing before any
  // grid-dependent work below, and it creates every FFTW plan.
  for (int index=0; index<ntheta; index++) {
    if (!(indices[index] >= 0 && indices[index] < nell)) {
      log_fatal("transform_3pcf: radial index outside Fourier grid");
      exit(1);
    }
    if (index > 0) {
      if (!(indices[index] > indices[index-1])) {
        log_fatal("transform_3pcf: radial indices must strictly increase");
        exit(1);
      }
    }
    plan->indices[index] = indices[index];
  }

  plan->fft = fftlog_create_3pcf(nell, ell, mmax+3, plan->fft_workers,
      nu1, nu2, xy, dlnt, padding, window, window_width);
  fftlog_select_3pcf(plan->fft, ntheta, plan->indices);

  // The squared radii ell1^2 ell2^2 weight the transform input: each
  // Fourier plane contributes the polar measure
  // d^2(ell) = ell^2 dln(ell) d(angle) in logarithmic variables.
  for (int index=0; index<nell; index++) {
    plan->ell[index] = ell[index];
    plan->ell2[index] = ell[index]*ell[index];
    if (!(isfinite(plan->ell2[index]))) {
      log_fatal("transform_3pcf: squared Fourier radius overflow");
      exit(1);
    }
  }

  // --- 4. ANGULAR COUPLING IN PSI ---
  // The coupling G_LM depends on the two legs only through the
  // opening angle psi = arctan(ell2/ell1). On a logarithmic grid the
  // ratio ell[j]/ell[i] = exp((j-i)*step) depends only on the index
  // difference j-i, so psi takes just 2*nell-1 distinct values - one
  // per difference from 1-nell to nell-1 - instead of nell*nell. The
  // table stores them at index (j-i)+nell-1, and the transform reads
  // the coupling with that same offset.
  double* psi = calloc1d((int)nratio);

  for (int difference=1-nell; difference<nell; difference++) {
    // atan2(ell2, ell1) equals arctan(ell2/ell1) for positive legs,
    // without an explicit ratio that could overflow. first and second
    // name one grid pair with this difference; every pair with the
    // same difference shares this psi.
    const int first = difference < 0 ? -difference : 0;
    const int second = first+difference;
    psi[difference+nell-1] = atan2(ell[second], ell[first]);
  }

  legendre_coupling_3pcf(lmax, mmax, (int)nratio, psi, npsi,
      plan->coupling);

  // Fold the Legendre parity factor (-1)^L into the stored coupling.
  // The factor converts the Legendre expansion in mu to the coupling
  // integral's own angle variable (file head, step 2); it is pure
  // geometry, so applying it once here spares every prediction from
  // repeating the sign at every cell.
  for (int mode=0; mode<=mmax; mode++) {
    for (int degree=0; degree<=lmax; degree++) {
      const size_t offset = ((size_t)mode*(lmax + 1) + degree)*nratio;
      const double sign = degree%2 == 0 ? 1.0 : -1.0;
      for (size_t ratio=0; ratio<nratio; ratio++) {
        plan->coupling[offset + ratio] = sign*plan->coupling[offset + ratio];
      }
    }
  }

  free(psi);

  // --- 5. OPENING-ANGLE FOURIER TABLE ---
  // Store cos(M*phi) and sin(M*phi) for the resummation (file head,
  // step 4). With dphi > 0 the table also averages each x-projected
  // Fourier mode over its angle bin: averaging exp(i*M*phi) over a
  // bin of width dphi centered on phi multiplies it by
  // sin(M*dphi/2)/(M*dphi/2). That average is exact for the
  // x-projected series only; a later centroid phase still uses the
  // bin center, as this API specifies, and is not an exact average of
  // the varying centroid-projected shear.
  for (int angle=0; angle<nphi; angle++) {
    if (!(isfinite(phi[angle]) && phi[angle] > 0.0
          && phi[angle] < M_PI)) {
      log_fatal("transform_3pcf: phi must lie in (0,pi)");
      exit(1);
    }
    for (int mode=0; mode<=mmax; mode++) {
      const size_t offset = ((size_t)mode*nphi+angle)*2;
      const double half_width = 0.5*mode*dphi;
      const double average = half_width == 0.0
          ? 1.0 : sin(half_width)/half_width;
      plan->angles[offset] = average*cos(mode*phi[angle]);
      plan->angles[offset+1] = average*sin(mode*phi[angle]);
    }
  }

  // --- 6. PROJECTION PHASES ---
  // PHYSICAL DERIVATION & LOGIC FLOW
  // The transform works in the x projection, where every shear is
  // measured against one fixed axis. Published correlators instead
  // project each shear onto the direction from the triangle's
  // centroid to its own vertex. The conversion is one phase per
  // component, fixed by geometry alone, so the constructor stores it
  // once per triangle.
  //
  // Build the triangle in the complex plane. The first side lies on
  // the real axis, side1 = theta1/scale, and this file places the
  // second side at side2 = (theta2/scale)*exp(-i*phi), so the opening
  // angle turns clockwise from side 1 to side 2. The aperture kernels
  // of aperture_3pcf.c draw the same opening angle counterclockwise
  // because they follow TreeCorr, whose angle is minus the fastnc one
  // (arXiv:2407.01798); at equal (theta1,theta2,phi) arguments the
  // two conventions differ only by the component order and sign of
  // the TreeCorr conversion, so each drawing is right in its own
  // file. Dividing both sides by scale = max(theta1,theta2) keeps the
  // squares below well inside double range for any angular unit; the
  // scale drops out because spin_phase_3pcf normalizes its argument.
  // For the triangle whose vertex separations obey x_1 - x_2 = 3*side1
  // and x_1 - x_3 = 3*side2, the centroid-to-vertex offsets are
  //
  //   q_1 = side1 + side2,
  //   q_2 = -2*side1 + side2,
  //   q_3 = side1 - 2*side2,
  //
  // and they sum to zero, as offsets from a centroid must. The common
  // factor 3 and any overall sign drop out in the squared unit phase.
  // spin_phase_3pcf turns each offset into the spin-2 phase
  // exp(2i*beta) of its direction; component mu conjugates the phase
  // of its conjugated vertex. rotation = exp(i*phi) is the unit phase
  // of the opening angle itself and rotation3 = exp(3i*phi) its cube;
  // each component carries its own power of this rotation to complete
  // the x-to-centroid bookkeeping of fastnc.
  //
  // projection selects the stored value: 0 keeps the x projection
  // (phase one); 1 applies this centroid phase in the convention of
  // the upstream TATT comparison code; 2 applies its complex
  // conjugate, which the fastnc main branch uses.
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


// Free the plan, its FFT workspace and any prepared-grid geometry.
// NULL is a no-op, so error paths can call this unconditionally.
void transform_destroy_3pcf(
  transform_plan_3pcf* plan // owning plan
)
{
  if (plan == NULL) {
    return;
  }

  fftlog_destroy_3pcf(plan->fft);
  free(plan->indices);
  free(plan->ell);
  free(plan->ell2);
  physics_grid_query_destroy_3pcf(plan->query);
  free(plan->grid_multipoles);
  free(plan->coupling);
  free(plan->modes);
  free(plan->phases);
  free(plan->angles);
  free(plan->gamma);
  free(plan);
}


// ---------------------------------------------------------------------------
// Turn bispectrum multipoles into the four natural components on
// every triangle of the plan's grid.
//
// Stage one computes, for each real-space multipole M, the coupling
// sum H_M = sum_L (-1)^L G_LM(psi) b_L of the file head, weights it
// by ell1^2 ell2^2, and runs the radial double-Bessel transforms that
// produce the component multipoles Gamma_mu^(M)(theta1,theta2). Stage
// two resums the finite Fourier series in the opening angle and
// applies the stored projection phase and component convention.
//
// Inputs: multipoles holds b_L on [lmax+1,nell,nell] in the
// convention ell3^2 = ell1^2 + ell2^2 - 2*ell1*ell2*mu stated in
// transform_3pcf.h. The outputs receive the [4,ntheta,ntheta,nphi]
// real and imaginary parts.
// Thread safety: one call owns the plan's scratch until it returns;
// the plan is not reentrant.
// ---------------------------------------------------------------------------
void transform_natural_3pcf(
  transform_plan_3pcf* plan, // non-reentrant owning geometry plan
  const double* multipoles, // [L,nell,nell], minus-cosine convention
  double* gamma_real, // [component,theta1,theta2,phi]
  double* gamma_imag // same shape
)
{
  if (!(plan != NULL && multipoles != NULL
        && gamma_real != NULL && gamma_imag != NULL)) {
    log_fatal("transform_3pcf: null array");
    exit(1);
  }

  const int nell = plan->nell;
  const int ntheta = plan->ntheta;
  const int nphi = plan->nphi;
  const size_t nratio = 2*(size_t)nell-1;
  const size_t radial_pairs = (size_t)ntheta*ntheta;
  const size_t triangles = radial_pairs*nphi;

  // Each Fourier leg contributes a plane measure d^2(ell)/(2*pi)^2,
  // giving (2*pi)^-4 in total. The fastnc paper books three of those
  // powers on each multipole and the fourth on the resummation
  // prefactor 1/(2*pi); this code multiplies the whole product into
  // the stored modes, so the resummation loop below is a plain sum.
  const double normalization = 1.0/pow(2.0*M_PI, 4);
  const int nmodes = plan->ordered_legs ? 2*plan->mmax+1 : plan->mmax+1;

  // Each job computes one signed multipole M: it fills the FFT input
  // with ell1^2 ell2^2 H_M, runs one forward FFT, and stores the four
  // transformed components in that M's block of plan->modes. Jobs are
  // independent - each owns one worker slice and one output block -
  // and every L sum keeps its fixed order, so no OpenMP reduction can
  // change a cancellation-sensitive multipole sum when the worker
  // count changes.
  //
  // The job index maps to M in the order -mmax..-2, 0, +2..+mmax, -1,
  // +1 (without ordered legs: 0, 2..mmax, 1). Modes +1 and -1 need
  // only two inverse transforms each, because at |M| = 1 two
  // component pairs share Bessel kernels; every other mode needs
  // four. Queueing the shorter jobs last lets the final worker take
  // both while its peers finish longer jobs, and schedule(dynamic, 1)
  // also accommodates teams smaller than the plan's worker capacity.
  #pragma omp parallel for num_threads(plan->fft_workers) \
      if(nmodes > 1) schedule(dynamic, 1)
  for (int job=0; job<nmodes; job++) {
    int mode = job;
    // These branches spell out the job order above. With ordered legs
    // the first branch covers -mmax..-2, the second inserts 0, the
    // third covers +2..+mmax, and the last two append -1 and +1. The
    // symmetric shortcut keeps nonnegative modes only, again with
    // mode 1 last.
    if (plan->mmax > 0 && plan->ordered_legs) {
      if (job < plan->mmax - 1) {
        mode = job - plan->mmax;
      } else if (job == plan->mmax - 1) {
        mode = 0;
      } else if (job < nmodes - 2) {
        mode = job - plan->mmax + 2;
      } else {
        mode = job == nmodes - 2 ? -1 : 1;
      }
    } else if (plan->mmax > 0) {
      mode = job == nmodes - 1 ? 1 : (job == 0 ? 0 : job + 1);
    }

    // The storage slot always follows physical M, whatever the job
    // order: slot mode+mmax with ordered legs, slot mode without.
    const int task = plan->ordered_legs ? mode + plan->mmax : mode;

    int worker = 0;
    #ifdef _OPENMP
    worker = omp_get_thread_num();
    #endif
    double* input = fftlog_input_3pcf(plan->fft, worker);

    // Scalar meaning for one Fourier cell (first, second):
    //   sum = 0;
    //   for (degree = 0; degree <= lmax; degree++) {
    //     sum += coupling[|M|][degree][second-first+nell-1]
    //            *multipoles[degree][first][second];
    //   }
    //   input[first][second] = sum*ell2[first]*ell2[second];
    // This is the coupling sum H_M of the file head weighted by the
    // squared leg radii of the logarithmic plane measure. For M < 0
    // the coupling comes from the mirrored table entry explained at
    // the loads below. The two SIMD lanes hold the cells
    // (first, second) and (first, second+1): adjacent second-leg
    // nodes whose L sums advance together, lane by lane, with no
    // horizontal reduction and no reassociation. Coupling and
    // multipole values load contiguously; no gathers are needed.
    for (int first=0; first<nell; first++) {
      int second = 0;
      // The four-wide loop advances two independent lane pairs per
      // iteration - ordinary unrolling: while one pair's addition
      // completes, the processor starts the other pair's, hiding the
      // add latency. Each cell still visits L in its original order.
      for (; second + 3<nell; second+=4) {
        const size_t cell = (size_t)first*nell+second;
        const size_t ratio = second-first+nell-1;
        // Two accumulator pairs, all lanes starting at zero: sum0
        // holds cells (second, second+1), sum1 (second+2, second+3).
        simde__m128d sum0 = simde_mm_setzero_pd();
        simde__m128d sum1 = simde_mm_setzero_pd();
        for (int degree=0; degree<=plan->lmax; degree++) {
          const size_t offset =
              ((size_t)abs(mode)*(plan->lmax+1)+degree)*nratio;
          simde__m128d coupling0;
          simde__m128d coupling1;
          if (mode >= 0) {
            // Load the coupling at the four consecutive ratio indices
            // of these cells: coupling0 holds (ratio, ratio+1) and
            // coupling1 holds (ratio+2, ratio+3).
            coupling0 = simde_mm_loadu_pd(plan->coupling+offset+ratio);
            coupling1 = simde_mm_loadu_pd(plan->coupling+offset+ratio+2);
          } else {
            // G_L,-M(psi) = G_L,M(pi/2-psi), and pi/2-psi sits at the
            // mirrored ratio index nratio-1-r of the M >= 0 table.
            // Walking second upward therefore walks the mirrored
            // entries downward: load each needed pair contiguously
            // from below its position - coupling0 takes the entries
            // at (nratio-2-ratio, nratio-1-ratio), coupling1 the two
            // below them...
            coupling0 = simde_mm_loadu_pd(
                plan->coupling+offset+nratio-2-ratio);
            coupling1 = simde_mm_loadu_pd(
                plan->coupling+offset+nratio-4-ratio);
            // ...and the shuffles (immediate 1: element 1 to lane 0,
            // element 0 to lane 1) reverse each pair, so lane 0
            // carries the mirrored entry of second and lane 1 that of
            // second+1, then second+2 and second+3. b_L itself is not
            // mirrored: only the coupling depends on the sign of M.
            coupling0 = simde_mm_shuffle_pd(coupling0, coupling0, 1);
            coupling1 = simde_mm_shuffle_pd(coupling1, coupling1, 1);
          }
          // One pointer to b_L at (degree, first, second).
          const double* values = multipoles+(size_t)degree*nell*nell+cell;
          // Walking the nested calls inside out: loadu reads b_L at
          // cells (second, second+1); mul multiplies each lane by its
          // coupling entry; add folds the products into sum0.
          // Multiplication and addition round separately here - no
          // fused step - matching the scalar example.
          sum0 = simde_mm_add_pd(sum0,
              simde_mm_mul_pd(coupling0, simde_mm_loadu_pd(values)));
          // The same walkthrough for the second pair: loadu reads b_L
          // at (second+2, second+3), mul applies coupling1, and add
          // folds into sum1.
          sum1 = simde_mm_add_pd(sum1,
              simde_mm_mul_pd(coupling1, simde_mm_loadu_pd(values+2)));
        }
        // Broadcast the shared first-leg weight ell1^2 of this row.
        const simde__m128d first_factor = simde_mm_set1_pd(plan->ell2[first]);
        // Multiply both lane pairs by it.
        const simde__m128d first0 = simde_mm_mul_pd(sum0, first_factor);
        const simde__m128d first1 = simde_mm_mul_pd(sum1, first_factor);
        // Inside out: loadu reads the second-leg weights ell2^2 at
        // (second, second+1); mul completes sum*ell1^2*ell2^2; storeu
        // writes the finished pair into the FFT input at cell and
        // cell+1 (two valid doubles, no aligned address needed).
        simde_mm_storeu_pd(input+cell, simde_mm_mul_pd(first0,
            simde_mm_loadu_pd(plan->ell2+second)));
        // The same three steps for lanes (second+2, second+3), stored
        // at cell+2 and cell+3.
        simde_mm_storeu_pd(input+cell+2, simde_mm_mul_pd(first1,
            simde_mm_loadu_pd(plan->ell2+second+2)));
      }
      // nell is even, so after the four-wide loop either zero or two
      // cells remain in this row: this two-wide loop runs at most
      // once and repeats the same arithmetic on a single lane pair.
      for (; second<nell; second+=2) {
        const size_t cell = (size_t)first*nell+second;
        const size_t ratio = second-first+nell-1;
        simde__m128d sum = simde_mm_setzero_pd();
        for (int degree=0; degree<=plan->lmax; degree++) {
          const size_t offset =
              ((size_t)abs(mode)*(plan->lmax+1)+degree)*nratio;
          simde__m128d coupling;
          if (mode >= 0) {
            coupling = simde_mm_loadu_pd(plan->coupling+offset+ratio);
          } else {
            // The same mirrored read as in the four-wide loop:
            // G_L,-M(psi) = G_L,M(pi/2-psi) sends ratio r to
            // nratio-1-r, so the pair loads contiguously from below
            // its position and the shuffle (element 1 to lane 0,
            // element 0 to lane 1) restores ascending second order.
            // b_L is not mirrored.
            coupling = simde_mm_loadu_pd(
                plan->coupling+offset+nratio-2-ratio);
            coupling = simde_mm_shuffle_pd(coupling, coupling, 1);
          }
          // Load b_L at cells (second, second+1) for this degree.
          const simde__m128d values = simde_mm_loadu_pd(
              multipoles+(size_t)degree*nell*nell+cell);
          // Inside out: mul forms coupling times b_L per lane; add
          // folds the pair into the running L sums (separate
          // roundings, fixed order).
          sum = simde_mm_add_pd(sum,
              simde_mm_mul_pd(coupling, values));
        }
        // Inside out: set1 broadcasts ell1^2 of this row to both
        // lanes; mul scales both lane sums by it.
        const simde__m128d weighted_first =
            simde_mm_mul_pd(sum, simde_mm_set1_pd(plan->ell2[first]));
        // Inside out: loadu reads ell2^2 at (second, second+1); mul
        // completes the integrand weight.
        const simde__m128d weighted_second = simde_mm_mul_pd(
            weighted_first, simde_mm_loadu_pd(plan->ell2+second));
        // Store the finished pair into the FFT input grid.
        simde_mm_storeu_pd(input+cell, weighted_second);
      }
    }

    fftlog_forward_3pcf(plan->fft, worker);

    // Bessel-order pairs of the four natural components, from the
    // fastnc paper's multipole formulas (Sugiyama, Gomes & Jarvis,
    // arXiv:2407.01798), with J the cylindrical Bessel function:
    //
    //   Gamma_0^(M): J_(M-3)(ell1*theta1) J_(-M-3)(ell2*theta2)
    //   Gamma_1^(M): J_(M-1)(ell1*theta1) J_(-M-1)(ell2*theta2),
    //                paired with H_M(ell2,ell1) = H_(-M)(ell1,ell2)
    //   Gamma_2^(M): J_(M+1)(ell1*theta1) J_(-M-3)(ell2*theta2)
    //   Gamma_3^(M): J_(M-3)(ell1*theta1) J_(-M+1)(ell2*theta2)
    //
    // Conjugating the shear at vertex 2 or 3 flips that vertex's
    // spin-2 projection phase, which raises the Bessel order of its
    // leg by four: Gamma_2 raises the first-leg order and Gamma_3 the
    // second-leg order relative to Gamma_0. Conjugating vertex 1
    // instead exchanges the roles of the two legs, so Gamma_1 couples
    // to H of the exchanged arguments. This loop transforms H_M for
    // one signed M at a time, so it uses that identity in reverse:
    // transforming H_M with Gamma_1's orders written at -M (the table
    // rows below) produces Gamma_1^(-M), which lands in the -M slot
    // (destination below).
    const int order1[4] = {mode-3, -mode-1, mode+1, mode-3};
    const int order2[4] = {-mode-3, mode-1, -mode-3, -mode+1};
    const double* completed[4] = {NULL, NULL, NULL, NULL};

    for (int component=0; component<4; component++) {
      const int first_order = order1[component];
      const int second_order = order2[component];
      // fftlog_inverse_3pcf accepts nonnegative orders only, so the
      // reflection identity of cylindrical Bessel functions,
      // J_(-k)(x) = (-1)^k J_k(x), moves each sign into a prefactor:
      // -1 exactly when the order is negative and odd.
      const double sign1 = first_order < 0 && abs(first_order)%2
          ? -1.0 : 1.0;
      const double sign2 = second_order < 0 && abs(second_order)%2
          ? -1.0 : 1.0;
      // Component 1's transform of H_M belongs to Gamma_1 at -M (see
      // the order table above); every other component stores at M.
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

      // At M = +1 and M = -1 two component pairs carry the same
      // Bessel orders up to even-order sign flips, so their
      // transforms coincide; copying the finished values spares a
      // second inverse FFT. This is an identity of the kernels alone
      // and does not exchange physical legs.
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
        // Apply the (2*pi)^-4 and the reflection signs while copying
        // the worker's transient output into this mode's block.
        for (size_t cell=0; cell<radial_pairs; cell++) {
          target[cell] = normalization*sign1*sign2*transformed[cell];
        }
      }

      completed[component] = target;
    }
  }

  // Opening-angle resummation and projection, one triangle per
  // iteration. A triangle is one (theta1-bin, theta2-bin, phi-bin)
  // cell of the flattened output; pair indexes its radial block and
  // transpose exchanges theta1 and theta2 inside that block, which
  // the symmetric shortcut below needs. Each iteration reads the
  // finished mode blocks, sums the finite Fourier series in phi - the
  // angles table already carries cos/sin(M*phi) with the phi
  // bin-average factor - and applies the projection phase and the
  // component convention. Iterations write disjoint outputs, so a
  // static schedule splits them evenly.
  #pragma omp parallel for num_threads(plan->workers) if(triangles > 1) \
      schedule(static)
  for (size_t triangle=0; triangle<triangles; triangle++) {
    const size_t pair = triangle/nphi;
    const int angle = triangle%nphi;
    const size_t transpose = (pair%ntheta)*ntheta+pair/ntheta;
    double real[4] = {0.0, 0.0, 0.0, 0.0};
    double imag[4] = {0.0, 0.0, 0.0, 0.0};

    if (plan->ordered_legs) {
      // General route: every signed M has its own stored block. The
      // real part of Gamma accumulates value*cos(M*phi) and the
      // imaginary part value*sin(M*phi); the table holds the angles
      // of |M|, so the sign flips the sine term only.
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
      // The fastnc main-branch transpose shortcut, valid when b_L is
      // symmetric under leg exchange. The fastnc paper's reflection
      // identities then tie negative to positive multipoles:
      //
      //   Gamma_0^(-M)(t1,t2) = Gamma_0^(M)(t2,t1)
      //   Gamma_1^(-M)(t1,t2) = Gamma_1^(M)(t2,t1)
      //   Gamma_2^(-M)(t1,t2) = Gamma_3^(M)(t2,t1)
      //   Gamma_3^(-M)(t1,t2) = Gamma_2^(M)(t2,t1),
      //
      // so the mode loop rebuilds every negative-M value from a
      // transposed positive-M block, with components 2 and 3 trading
      // places there. One storage detail carries over from the
      // transform stage: each block keeps component 1 at -M, so for
      // component 1 the roles of pair and transpose are exchanged
      // relative to the other components. The M = 0 terms below read
      // components 0 and 2 through that same transposed pattern and
      // components 1 and 3 directly; at M = 0 the identities make the
      // transposed and direct values agree for a leg-symmetric b_L,
      // and this form mirrors the fastnc main branch exactly.
      real[0] = plan->modes[transpose];
      real[1] = plan->modes[radial_pairs+pair];
      real[2] = plan->modes[3*radial_pairs+transpose];
      real[3] = plan->modes[3*radial_pairs+pair];

      for (int mode=1; mode<=plan->mmax; mode++) {
        const double* values = plan->modes+(size_t)mode*4*radial_pairs;
        // Gamma_mu^(+M) at this radial pair: components 0, 2 and 3
        // read their own entries directly, while component 1 reads
        // its block at the transposed pair because that block stores
        // the -M multipole.
        const double positive[4] = {values[pair],
          values[radial_pairs+transpose], values[2*radial_pairs+pair],
          values[3*radial_pairs+pair]};

        // Gamma_mu^(-M) through the identities above: component 0
        // transposes its own block, component 1 reads directly, and
        // components 2 and 3 read each other's transposed entries.
        const double negative[4] = {values[transpose],
          values[radial_pairs+pair], values[3*radial_pairs+transpose],
          values[2*radial_pairs+transpose]};
        const size_t offset = ((size_t)mode*nphi+angle)*2;

        // Pair the +M and -M terms of the Fourier series. The stored
        // multipoles are real, so the cosine collects their sum in
        // the real part and the sine their difference in the
        // imaginary part.
        for (int component=0; component<4; component++) {
          real[component] += (positive[component]+negative[component])
              *plan->angles[offset];
          imag[component] += (positive[component]-negative[component])
              *plan->angles[offset+1];
        }
      }
    }

    // Component order and sign of the output. fastnc reports
    // [Gamma_0, Gamma_1, Gamma_2, Gamma_3]; TreeCorr's natural
    // components equal -[Gamma_0, Gamma_1, Gamma_3, Gamma_2] in that
    // order, so convention 1 swaps the last two components through
    // the permutation and negates all four. The stored projection
    // phase then rotates each component as one complex
    // multiplication.
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


// ---------------------------------------------------------------------------
// Natural components contracted into real aperture-mass third
// moments.
//
// The aperture statistic weights the four components over every
// triangle (aperture_3pcf.c). This entry point computes the
// components into the plan's scratch, converts them to the TreeCorr
// convention the aperture weights expect, and runs the contraction
// for every filter triplet. Centroid projection is required because
// the aperture kernels describe centroid-projected shears.
// ---------------------------------------------------------------------------
void transform_aperture_3pcf(
  transform_plan_3pcf* plan, // owning centroid-projection plan
  const double* multipoles, // bispectrum multipoles from the caller
  const int nfilter, // ordered aperture triplets
  const double* weights_real, // TreeCorr real weights
  const double* weights_imag, // TreeCorr imaginary weights
  double* output // real aperture moments
)
{
  if (!(plan != NULL && plan->projection != 0)) {
    log_fatal("transform_3pcf: aperture contraction requires centroid "
      "projection");
    exit(1);
  }

  const int triangles = plan->ntheta*plan->ntheta*plan->nphi;
  double* real = plan->gamma;
  double* imag = plan->gamma+4*(size_t)triangles;

  transform_natural_3pcf(plan, multipoles, real, imag);

  if (plan->convention == 0) {
    // The aperture weights follow TreeCorr, whose natural components
    // read -[Gamma_0, Gamma_1, Gamma_3, Gamma_2]. A fastnc-convention
    // plan therefore converts in place before the contraction; both
    // members of the exchanged pair are read before either is set.
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


// ---------------------------------------------------------------------------
// Aperture moments straight from an axis-keyed physical grid.
//
// The physics grid holds bispectrum multipoles on its own axes; a
// query object interpolates them onto this plan's Fourier grid. The
// plan keeps the query and the b_L buffer so repeated calls reuse the
// geometry, then validates finiteness and hands the interpolated
// multipoles to the aperture contraction above.
// ---------------------------------------------------------------------------
void transform_aperture_grid_3pcf(
  transform_plan_3pcf* plan, // one caller owns this plan's mutable scratch
  const physics_grid_3pcf* grid, // immutable physical table for this call
  const int nfilter, // number of aperture filters in the weights
  const double* weights_real, // [filter,4,triangle], TreeCorr order
  const double* weights_imag, // matching imaginary weights
  double* output // [filter]
)
{
  if (!(plan && grid && weights_real && weights_imag
        && output && nfilter > 0)) {
    log_fatal("transform_3pcf: invalid prepared-grid aperture input");
    exit(1);
  }

  if (!physics_grid_query_matches_3pcf(plan->query, grid)) {
    physics_grid_query_destroy_3pcf(plan->query);
    plan->query = physics_grid_query_create_3pcf(grid, plan->nell, plan->ell);
  }

  if (!(physics_grid_query_lmax_3pcf(plan->query) == plan->lmax
        && physics_grid_query_nell_3pcf(plan->query) == plan->nell)) {
    log_fatal("transform_3pcf: projected-grid multipoles mismatch the "
      "transform plan");
    exit(1);
  }
  if (!((size_t)plan->nell <= (size_t)INT_MAX/(size_t)plan->nell)) {
    log_fatal("transform_3pcf: prepared-grid Fourier size overflow");
    exit(1);
  }
  const size_t pairs = (size_t)plan->nell*plan->nell;
  if (!((size_t)(plan->lmax + 1) <= (size_t)INT_MAX/pairs)) {
    log_fatal("transform_3pcf: prepared-grid multipole size overflow");
    exit(1);
  }

  const size_t count = (size_t)(plan->lmax + 1)*pairs;
  if (plan->grid_multipoles == NULL) {
    plan->grid_multipoles = calloc1d((int)count);
  }

  // Keeping this buffer between calls reuses the allocation only: the
  // query overwrites the entire prediction even when geometry, IA
  // settings or source bins agree with the previous call.
  physics_grid_query_evaluate_3pcf(plan->query, grid, 0, plan->grid_multipoles);

  int finite = 1;
  #pragma omp parallel for num_threads(plan->workers) if(count > 4096) \
      reduction(&:finite) schedule(static)
  for (size_t index=0; index<count; index++) {
    finite &= isfinite(plan->grid_multipoles[index]) ? 1 : 0;
  }

  // The reduction combines integer validity flags only. No
  // floating-point sum is reordered; the interpolation and the
  // transforms behind it are unchanged by the worker count.
  if (!(finite)) {
    log_fatal("transform_3pcf: nonfinite prepared-grid multipoles");
    exit(1);
  }

  transform_aperture_3pcf(plan, plan->grid_multipoles, nfilter,
      weights_real, weights_imag, output);
}
