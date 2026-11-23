#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <gsl/gsl_sf_bessel.h>
#include "simde/x86/sse2.h"
#include "covariance_3pcf.h"
#include "cosmolike/basics.h"
#include "log.c/src/log.h"

// ===========================================================================
// Covariance contractions for filtered scalar fields
// ===========================================================================
// A circular aperture filter turns a map into a scalar value at each
// aperture center. Correlations of these values are easier to assemble
// than a six-wavevector integral when the fields are Gaussian: Wick's
// theorem expresses a six-field expectation as products of two-field
// correlations. A model for the connected part of a covariance - the
// part that no product of two-field correlations accounts for -
// allows a different reduction: when its covariance is an integral
// dnu H_i(nu) H_j(nu), with nu the integration label and H_i the
// response of observable i, the covariance is a weighted Gram matrix
// of those responses, the matrix whose entry (i,j) is the weighted
// inner product of response i with response j. The functions below
// provide these two contractions and the radial pair transform.
//
// Linke et al. (aperture-statistics covariances, arXiv 2212.04485)
// discuss the Gaussian aperture covariance and the survey-window
// distinction; the sums below implement their stated equations
// directly. The functions import no survey, no cosmology, no halo
// profiles and no galaxy-catalogue estimator. In particular, the
// connected response contraction does not supply the
// bispectrum-squared or power-times-trispectrum covariance
// contributions.
//
// This file is integrated with the core utilities: it reports errors
// through log_fatal and allocates its one temporary Bessel table
// through the core allocator malloc1d, like the rest of the module.
//
// All arrays use contiguous C order, with the last index changing
// fastest. size_t is an unsigned type large enough for an array
// offset; casting each dimension to size_t before multiplying keeps
// the intermediate index products from overflowing int. The caller
// checks array lengths, finite values and physical domains; the guards
// below catch invalid dimensions, pointers and field labels. The pair
// transform fills one temporary Bessel table (an OpenMP team computes
// the table itself) before the field-pair contraction workers start;
// the other contractions need no heap storage. The functions keep no
// global state, leave every input unchanged, and require that the
// output storage overlap no input.

// SIMD performs the same arithmetic on several values in positions
// called lanes. A v2d holds two doubles side by side. The loops below
// use lanes for separate outputs or radial samples, never to reorder a
// sum over integration nodes. SIMDe supplies the same intrinsic
// interface across supported processors. Unaligned loads and stores
// accept ordinary caller buffers, which make no alignment promise.
typedef simde__m128d v2d;


// ---------------------------------------------------------------------------
// Radial pair correlation from an angular power spectrum.
//
// Derivation and logic:
// The Fourier convention is X(theta) = integral d^2ell/(2*pi)^2 X(ell)
// exp(i ell.theta), with theta the angular position and ell the
// angular wavevector. Isotropy makes the power depend on ell's length
// alone. Integrating the polar angle of ell gives 2*pi*J0(ell*r), with
// J0 the Bessel function of the first kind of order zero and r the
// separation, so the remaining measure is dell*ell/(2*pi). Each of the
// two fields contributes one real circular filter U. The quadrature
// the caller supplies approximates precisely that integral, and the
// result is the pair correlation K_ij(r) of the two filtered fields.
// A finite ell interval returns a finite-interval correlation; the
// caller must test that the interval and spacing resolve the
// separations it requests.
//
// A field label can combine a source bin and an aperture radius. power
// may repeat the same source-bin spectrum for several radii. Noise,
// when present, already lives inside power; adding it again here would
// double count it. With angles in radians, ell*r is dimensionless, and
// K carries the units of the product of the two filtered fields under
// the Fourier convention above.
//
// Cache invalidation:
// The temporary J0 table belongs to this call's ell and separation grids.
// Every source/filter pair shares it, and the call frees it on return.
// Thread safety:
// Each worker owns one ordered field-pair output row. The ell sum is
// serial within each radial sample, so scheduling does not change it.
// Calling within an existing OpenMP region does not create another team.
// ---------------------------------------------------------------------------
void filtered_pair_3pcf(
  const int nfield,            // number of field/filter labels
  const int nell,              // number of multipole quadrature nodes
  const int nr,                // number of separation samples
  const double* ell,           // nonnegative multipoles [nell]
  const double* ell_weights,   // quadrature weights for dell [nell]
  const double* r,             // nonnegative separations, radians [nr]
  const double* filters,       // real Fourier filters [nfield,nell]
  const double* power,         // total angular power [nfield,nfield,nell]
  double* out                  // pair functions [nfield,nfield,nr]
)
{
  if (nfield <= 0
      || nell <= 0
      || nr <= 0
      || NULL == ell
      || NULL == ell_weights
      || NULL == r
      || NULL == filters
      || NULL == power
      || NULL == out) {
    log_fatal("filtered_pair_3pcf: invalid dimensions or pointer");
    exit(1);
  }

  // The flat-sky measure factor 1/(2*pi) in dell*ell/(2*pi). Halving
  // M_1_PI, the nearest double to 1/pi, is exact and gives the nearest
  // double to 1/(2*pi).
  const double inverse_two_pi = 0.5*M_1_PI;

  if ((size_t)nell > (size_t)INT_MAX/(size_t)nr) {
    log_fatal("filtered_pair_3pcf: Bessel table size overflow");
    exit(1);
  }

  double* bessel_table = malloc1d((int)((size_t)nell*nr));

  // J0 depends only on the two geometric coordinates. Calculate it once
  // per (ell,r), before any field-pair contraction. A row stores adjacent
  // radii contiguously so that two SIMD lanes need one ordinary load.
  #pragma omp parallel for collapse(2) if(!omp_in_parallel()) schedule(static)
  for (int multipole=0; multipole<nell; multipole++) {
    for (int radial=0; radial<nr; radial++) {
      bessel_table[(size_t)multipole*nr + radial] =
        gsl_sf_bessel_J0(ell[multipole]*r[radial]);
    }
  }

  // collapse(2) distributes ordered pairs (i,j), each with its own row of
  // nr outputs. No worker writes another pair's row. Both orders are
  // evaluated so that the caller's power rows determine their own values.
  #pragma omp parallel for collapse(2) if(!omp_in_parallel()) schedule(static)
  for (int i=0; i<nfield; i++) {
    for (int j=0; j<nfield; j++) {
      const size_t field_pair = (size_t)i*nfield + j;
      const double* restrict filter_i = filters + (size_t)i*nell;
      const double* restrict filter_j = filters + (size_t)j*nell;
      const double* restrict spectrum = power + field_pair*nell;
      double* restrict correlation = out + field_pair*nr;

      // restrict tells the compiler that writes through correlation do
      // not change the input rows. Read-only rows may coincide, as happens
      // for i=j. Accesses in this loop use these qualified row pointers.
      // Each lane accumulates K_ij at its own r in increasing ell order:
      // scalar_sum += dell_weight*ell/(2*pi)*U_i*U_j*P_ij*J0(ell*r).
      for (int radial=0; radial<nr; radial++) {
        correlation[radial] = 0.0;
      }

      for (int multipole=0; multipole<nell; multipole++) {
        double coefficient = ell_weights[multipole]*ell[multipole]
          *inverse_two_pi;
        coefficient *= filter_i[multipole]*filter_j[multipole];
        coefficient *= spectrum[multipole];
        const double* restrict bessel_row =
          bessel_table + (size_t)multipole*nr;

        // Each radius shares the same filter/power coefficient. Hoisting
        // it outside the radius loop avoids recomputing it for every pair
        // of lanes. Node order for each output remains increasing ell.
        // Broadcast that one coefficient into both lanes of a vector.
        const v2d amplitude = simde_mm_set1_pd(coefficient);
        int radial = 0;

        // Lanes hold neighboring radii: the low lane accumulates the
        // output at r[radial], the high lane at r[radial + 1]. Each sum
        // stays in its own lane; nothing crosses between them.
        for (; radial + 1 < nr; radial += 2) {
          // Load the two running sums correlation[radial] and
          // correlation[radial + 1]; the unaligned load needs no
          // special address.
          const v2d previous = simde_mm_loadu_pd(correlation + radial);
          // Load J0(ell*r) at the same two radii from this node's row.
          const v2d bessel = simde_mm_loadu_pd(bessel_row + radial);
          // Multiply coefficient times J0 separately in each lane.
          const v2d contribution = simde_mm_mul_pd(amplitude, bessel);
          // Add this ell node's contribution to each lane's running sum.
          const v2d updated = simde_mm_add_pd(previous, contribution);
          // Store both updated sums back over the two slots just read.
          simde_mm_storeu_pd(correlation + radial, updated);
        }

        // An odd radial count leaves one value with the identical scalar
        // expression. It is updated at this same multipole node.
        if (radial < nr) {
          correlation[radial] += coefficient*bessel_row[radial];
        }
      }
    }
  }

  free(bessel_table);
}


// ---------------------------------------------------------------------------
// Symmetric weighted Gram matrix of connected responses.
//
// Derivation and logic:
// Suppose a connected covariance has the form C_ij = integral dnu
// H_i(nu) H_j(nu), with nu the integration label and H_i the response
// of observable i. Quadrature replaces dnu by weights[node], and H_i
// by one row of responses. For a Poisson one-halo model the node
// labels halo mass and distance; H_i is then that halo's estimator
// response integrated over its center, and dnu carries the
// number-density measure and the inverse survey area. That
// interpretation requires deriving H_i for the actual estimator. This
// routine only performs the contraction the caller supplies, and any
// other response model with the same mathematical form can use it.
//
// For nonnegative weights, the quadratic form x_i C_ij x_j of any test
// vector x equals a sum of nonnegative squares,
// sum_node weights[node]*(sum_i x_i H_i[node])^2, so the matrix is
// positive semidefinite in exact arithmetic. Roundoff may still
// produce small negative eigenvalues; this code does not repair or
// invert the matrix, and it inserts no factor of two and no survey
// area.
//
// Cache invalidation:
// No cache. Responses and all measure factors come from the caller.
// Thread safety:
// Worker i computes columns j>=i and writes their symmetric partners.
// No other worker owns either member of those pairs. Each node sum keeps
// its original order; there is no OpenMP reduction across nodes.
// ---------------------------------------------------------------------------
void connected_gram_3pcf(
  const int nrow,              // number of observable responses
  const int nnode,             // number of shared quadrature nodes
  const double* responses,     // observable responses [nrow,nnode]
  const double* weights,       // nonnegative complete measure [nnode]
  double* out                  // symmetric covariance [nrow,nrow]
)
{
  if (nrow <= 0
      || nnode <= 0
      || NULL == responses
      || NULL == weights
      || NULL == out) {
    log_fatal("connected_gram_3pcf: invalid dimensions or pointer");
    exit(1);
  }

  // All output writes use this restrict-qualified pointer. No input
  // aliases its storage; the binding checks the weight domain and sizes.
  double* restrict covariance = out;

  // Row i owns every upper-triangle entry starting there and its mirror.
  // The two SIMD lanes below represent columns j and j+1 in that row.
  #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
  for (int i=0; i<nrow; i++) {
    const double* restrict response_i = responses + (size_t)i*nnode;
    int j = i;

    // The loop bound keeps both columns j and j+1 inside the matrix.
    // Scalar equivalent for each column c:
    // covariance[i,c] += weights[node]*response_i[node]*response_c[node].
    for (; j + 1 < nrow; j += 2) {
      const double* restrict response_j = responses + (size_t)j*nnode;
      const double* restrict response_next =
        responses + (size_t)(j + 1)*nnode;

      // Zero the sums for columns j (low lane) and j+1 (high lane).
      v2d sum = simde_mm_setzero_pd();

      // Advance through the same quadrature nodes for both columns. Only
      // the right response changes between lanes; the left is shared.
      for (int node=0; node<nnode; node++) {
        const double weighted_left = weights[node]*response_i[node];
        // Copy the weighted left response into both lanes.
        const v2d left = simde_mm_set1_pd(weighted_left);
        // High lane is column j+1; low lane is column j.
        const v2d right = simde_mm_set_pd(
          response_next[node], response_j[node]);
        // Form the two response products at this node.
        const v2d contribution = simde_mm_mul_pd(left, right);
        // Add each product to its own ordered node sum.
        sum = simde_mm_add_pd(sum, contribution);
      }

      double values[2];
      // Extract the two ordered node sums into a local array:
      // values[0] is column j (low lane), values[1] is column j+1.
      simde_mm_storeu_pd(values, sum);
      covariance[(size_t)i*nrow + j] = values[0];
      covariance[(size_t)j*nrow + i] = values[0];
      covariance[(size_t)i*nrow + j + 1] = values[1];
      covariance[(size_t)(j + 1)*nrow + i] = values[1];
    }

    // A single remaining column gets the same ordered contraction.
    // When that column is the diagonal (j = i), both mirror writes
    // land on the same entry, which this row owns.
    if (j < nrow) {
      const double* restrict response_j = responses + (size_t)j*nnode;
      double sum = 0.0;
      for (int node=0; node<nnode; node++) {
        const double weighted_left = weights[node]*response_i[node];
        sum += weighted_left*response_j[node];
      }
      covariance[(size_t)i*nrow + j] = sum;
      covariance[(size_t)j*nrow + i] = sum;
    }
  }
}


// ---------------------------------------------------------------------------
// Gaussian covariance of raw cubic products of filtered maps.
//
// Derivation and logic:
// Let row I describe M_a(x) M_b(x) M_c(x), averaged over centers x,
// and row J describe M_d(y) M_e(y) M_f(y), where M_a is the filtered
// map of field label a and every M has zero mean. For a Gaussian field
// the cubic mean vanishes, and Wick's theorem gives the six-field
// expectation as 15 pairings. A pairing divides the six labeled
// positions into three pairs. It has either three cross-pairs between
// x and y (3! = 6 choices), or one cross-pair (3*3 = 9 choices). With
// one cross-pair, the two fields left behind at each center pair with
// each other at that center. Those within-center pairs contribute the
// zero-separation correlation K(0), while each cross-pair contributes
// K(r) at the center separation r = |x-y|. Repeated field labels do
// not remove these choices.
//
// The resulting integrand for triplets a[3], b[3] is
//   sum_permutation product_p K_{a[p],b[permutation[p]]}(r)
//   + sum_pq K_{a[p],b[q]}(r)
//       *K_{a[remaining p,0],a[remaining p,1]}(0)
//       *K_{b[remaining q,0],b[remaining q,1]}(0).
// For six identical labels this becomes 6*K(r)^3 + 9*K(r)*K(0)^2.
// The caller's radial_weights integrate this expression together with
// the center window; for a uniform large area A they approximate
// 2*pi*r*dr/A. A finite common center window can instead supply its
// angular-averaged overlap measure, provided the filtered map
// correlations remain stationary. Masking the field before filtering
// generally breaks that stationarity; changing radial_weights alone
// does not account for such a mask.
//
// This is the covariance of raw map products, including any noise in
// K. A catalogue statistic that excludes repeated galaxies removes
// particular internal noise contractions and requires its own overlap
// counting. Likewise, a finite-triangle shear estimator is not
// automatically the complete aperture map product. This routine makes
// neither change. Connected non-Gaussian cumulants and spin-2
// xi-plus/minus rows require additional kernels. The result carries
// the units of the product of the two cubic observables when the
// caller's radial measure is dimensionless.
//
// Cache invalidation:
// No cache. pair_zero is explicit, independent of the radial grid.
// Thread safety:
// One worker owns each upper-triangle entry and its mirror. SIMD forms
// integrands at two r values, then adds them to the scalar integral in
// radial order. No radial sum is shared or reduced across workers.
// ---------------------------------------------------------------------------
void gaussian_aperture_covariance_3pcf(
  const int nfield,            // number of field/filter labels
  const int nr,                // number of radial quadrature nodes
  const int nrow,              // number of cubic observable rows
  const int* triplets,         // zero-based field labels [nrow,3]
  const double* pair,          // K_ij(r) [nfield,nfield,nr]
  const double* pair_zero,     // K_ij(0) [nfield,nfield]
  const double* radial_weights, // complete radial/window measure [nr]
  double* out                  // symmetric covariance [nrow,nrow]
)
{
  if (nfield <= 0
      || nr <= 0
      || nrow <= 0
      || NULL == triplets
      || NULL == pair
      || NULL == pair_zero
      || NULL == radial_weights
      || NULL == out) {
    log_fatal("gaussian_aperture_covariance_3pcf: invalid dimensions or "
      "pointer");
    exit(1);
  }

  // Check positional field labels before any worker turns them into pair
  // table offsets. Repeated valid labels are expected and remain intact.
  for (int row=0; row<nrow; row++) {
    for (int position=0; position<3; position++) {
      const int field = triplets[(size_t)row*3 + position];
      if (field < 0
          || field >= nfield) {
        log_fatal("gaussian_aperture_covariance_3pcf: field index out of "
          "range");
        exit(1);
      }
    }
  }

  // Every permutation matches the three left positions to three right
  // positions once. remaining[p] names the two positions other than p.
  const int permutations[6][3] = {
    {0, 1, 2}, {0, 2, 1}, {1, 0, 2},
    {1, 2, 0}, {2, 0, 1}, {2, 1, 0}
  };
  const int remaining[3][2] = {{1, 2}, {0, 2}, {0, 1}};

  double* restrict covariance = out;

  // Row i owns each upper-triangle pair (i,j) and the reflected (j,i).
  // Symmetric input correlations make that reflection the same covariance.
  #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
  for (int i=0; i<nrow; i++) {
    const int* a = triplets + (size_t)i*3;

    // Each j supplies a second cubic estimator. The pointer tables below
    // name its nine possible cross-correlations without copying their nr
    // values; repeated labels may correctly point to the same input row.
    for (int j=i; j<nrow; j++) {
      const int* b = triplets + (size_t)j*3;
      const double* cross[9];
      double zero_product[9];

      // Choosing p and q selects the one cross-pair. The two remaining
      // positions in each triplet supply its within-center correlation.
      for (int p=0; p<3; p++) {
        const int left0 = a[remaining[p][0]];
        const int left1 = a[remaining[p][1]];
        const double left_zero = pair_zero[(size_t)left0*nfield + left1];
        for (int q=0; q<3; q++) {
          const int right0 = b[remaining[q][0]];
          const int right1 = b[remaining[q][1]];
          const double right_zero =
            pair_zero[(size_t)right0*nfield + right1];
          const size_t field_pair = (size_t)a[p]*nfield + b[q];
          cross[p*3 + q] = pair + field_pair*nr;
          zero_product[p*3 + q] = left_zero*right_zero;
        }
      }

      double sum = 0.0;
      int ir = 0;

      // The two lanes form the full Wick integrand at adjacent radial
      // nodes. Scalar equivalent: density = sum of six K*K*K terms plus
      // nine K*K(0)*K(0) terms; sum += radial_weight*density.
      for (; ir + 1 < nr; ir += 2) {
        // Start both radial-node integrands at zero.
        v2d density = simde_mm_setzero_pd();

        // Three-cross terms use all six permutations of the right labels.
        // Each load reads two valid adjacent r entries of its pair row.
        for (int permutation=0; permutation<6; permutation++) {
          const int* order = permutations[permutation];
          // Load K_{a[0],b[order[0]]} at r[ir] and r[ir+1].
          const v2d first = simde_mm_loadu_pd(cross[order[0]] + ir);
          // Load the second cross-pair at the same radial nodes.
          const v2d second = simde_mm_loadu_pd(cross[3 + order[1]] + ir);
          // Load the third cross-pair at the same radial nodes.
          const v2d third = simde_mm_loadu_pd(cross[6 + order[2]] + ir);
          // Multiply the first two pair values separately at each r.
          const v2d first_two = simde_mm_mul_pd(first, second);
          // Complete the product of all three cross-pairs at each r.
          const v2d contribution = simde_mm_mul_pd(first_two, third);
          // Add this positional pairing to each radial integrand.
          density = simde_mm_add_pd(density, contribution);
        }

        // One-cross terms use each of the 3*3 choices once, with their
        // corresponding two within-center correlations fixed at r=0.
        for (int choice=0; choice<9; choice++) {
          // Load the chosen cross-correlation at two adjacent radii.
          const v2d between = simde_mm_loadu_pd(cross[choice] + ir);
          // Both radial nodes share the two within-center correlations.
          const v2d within = simde_mm_set1_pd(zero_product[choice]);
          // Form K(r)*K_left(0)*K_right(0) separately in each lane.
          const v2d contribution = simde_mm_mul_pd(between, within);
          // Add this one-cross pairing to both radial integrands.
          density = simde_mm_add_pd(density, contribution);
        }

        // Load the two radial window/quadrature weights; the measure
        // already carries any area factor.
        const v2d measure = simde_mm_loadu_pd(radial_weights + ir);
        // Weight each radial integrand with its own integration measure.
        const v2d weighted = simde_mm_mul_pd(measure, density);

        double values[2];
        // Store both weighted samples in their radial order.
        simde_mm_storeu_pd(values, weighted);
        sum += values[0];
        sum += values[1];
      }

      // An odd nr has one last sample. Evaluate its pairings in the same
      // order as the two-lane loop, then add its weighted contribution.
      if (ir < nr) {
        double density = 0.0;
        for (int permutation=0; permutation<6; permutation++) {
          const int* order = permutations[permutation];
          const double first_two =
            cross[order[0]][ir]*cross[3 + order[1]][ir];
          density += first_two*cross[6 + order[2]][ir];
        }
        for (int choice=0; choice<9; choice++) {
          density += cross[choice][ir]*zero_product[choice];
        }
        sum += radial_weights[ir]*density;
      }

      covariance[(size_t)i*nrow + j] = sum;
      covariance[(size_t)j*nrow + i] = sum;
    }
  }
}
