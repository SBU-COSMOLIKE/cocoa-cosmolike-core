#include <complex.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "simde/x86/sse2.h"
#include "aperture_3pcf.h"

// ===========================================================================
// FINITE TRIANGLE INTEGRATION FOR APERTURE MASS
// ===========================================================================
// Aperture mass averages the projected mass density with a compensated
// circular filter: a constant density gives zero. Its third moment can
// also be recovered from four complex shear three-point correlations.
// The kernels below describe the Gaussian aperture filter, whose Fourier
// form is U(ell*R) = (ell*R)^2 exp[-(ell*R)^2/2]/2.
//
// The physical equations are from Schneider, Kilbinger & Lombardi (2005),
// A&A 431, 9, Eqs. (62)-(69), https://arxiv.org/abs/astro-ph/0308328.
// Their equal-radius limit is the aperture statistic of Jarvis, Bernstein
// & Jain (2004), MNRAS 352, 338. TreeCorr's _calculateT/calculateMap3
// implementation by Mike Jarvis provides the centroid/SAS convention.
// The application to fastnc natural components follows the work of
// Sunao Sugiyama and Rafael Gomes; repository credits and license notices
// accompany this module. These routines implement the published math in C.
//
// A finite SAS grid labels a triangle by two lengths and the angle between
// them. This file integrates only the supplied triangle bins. It neither
// extrapolates missing triangles nor asserts that finite coverage equals
// the infinite-domain aperture statistic. Finite coverage and quadrature
// can mix E and B modes even for a physical field with only E modes.
//
// Arrays are flat, caller-owned C buffers; no core cosmology state is read.
// The caller checks units, bin definitions, sizes, finite inputs and lack
// of overlap. In particular, phi lies strictly between zero and pi. This
// excludes collinear triangles whose centroid can coincide with a vertex,
// where the direction used to project its spin-2 shear is undefined.
// Near-collinear triangles still require numerical accuracy checks.

// SIMD applies one operation to several numbers held in positions called
// lanes. v2d holds two doubles. Each lane below carries a separate aperture
// output, with its own ordered triangle sum; the lanes are never added.
// SIMDe maps these operations to the processor's vector instructions, or
// its portable implementation. No optional scalar implementation is used.
typedef simde__m128d v2d;


// A positive physical length can still be too large or too small for
// the dimensionless products below. Stop before using a zero or infinite
// denominator; replacing it by a clipped value would change the kernel.
// This helper keeps the diagnostic consistent at each arithmetic stage.
static inline void aperture_geometry_error_3pcf(const char* stage)
{
  fprintf(stderr,
    "aperture_weights_3pcf: geometry is not representable at %s; "
    "check nearly collinear triangles and length/radius ratios\n", stage);
  exit(EXIT_FAILURE);
}


// ---------------------------------------------------------------------------
// Four complex kernels for one ordered set of aperture radii.
//
// PHYSICAL DERIVATION & LOGIC FLOW
// Put the triangle vertices at 0, side and other, in units of R1. The
// three centroid offsets q_i sum to zero. Their squared lengths give the
// rotationally invariant part of the Gaussian convolution; their complex
// phases describe the rotation of a spin-2 shear at each vertex.
//
// For radii (R1,R2,R3), set a_i = (R_i/R1)^2 / Theta2, where
// Theta2 = sqrt[(R1^2 R2^2 + R1^2 R3^2 + R2^2 R3^2)/(3 R1^4)].
// The dimensionless functions f_i and g_i in Eqs. (65) and (69) describe
// the unequal filter sizes. Kernel zero has no conjugated shear. Kernels
// 1, 2 and 3 conjugate the shear at the corresponding centroid offset.
// The cyclic loop evaluates the same published expression at each vertex.
//
// Cache invalidation:
// No cache. Inputs describe one triangle and one ordered radius triplet.
// Thread safety:
// Only local arrays and the caller's four output values are written.
// static limits this helper to this file; inline permits call expansion
// while retaining a callable definition in an unoptimized debug build.
// ---------------------------------------------------------------------------
static inline void aperture_kernel_3pcf(
  const double side,              // d2/R1, positive
  const double complex other,     // (d3/R1)*exp(i*phi)
  const double ratio2,            // R2/R1, positive
  const double ratio3,            // R3/R1, positive
  double complex* kernel         // output [4], before bin measure
)
{
  // Squared radius ratios enter the Gaussian normalization. Reject an
  // unrepresentable square before it can enter a square root or divisor.
  // This bound comes from double precision, not from an angular cut.
  const double largest_ratio = sqrt(DBL_MAX);
  if (!isfinite(side)
      || side <= 0.0
      || !isfinite(creal(other))
      || !isfinite(cimag(other))
      || !isfinite(ratio2)
      || !isfinite(ratio3)
      || ratio2 <= 0.0
      || ratio3 <= 0.0
      || ratio2 > largest_ratio
      || ratio3 > largest_ratio) {
    aperture_geometry_error_3pcf("dimensionless lengths");
  }
  const double radius2_sq = ratio2*ratio2;
  const double radius3_sq = ratio3*ratio3;
  if (!isfinite(radius2_sq)
      || radius2_sq <= 0.0
      || !isfinite(radius3_sq)
      || radius3_sq <= 0.0) {
    aperture_geometry_error_3pcf("squared radius ratios");
  }
  const double theta_argument = (radius2_sq + radius3_sq
                                + radius2_sq*radius3_sq)/3.0;
  if (!isfinite(theta_argument)
      || theta_argument <= 0.0) {
    aperture_geometry_error_3pcf("Gaussian radius normalization");
  }
  const double theta2 = sqrt(theta_argument);
  const double theta4 = theta2*theta2;
  const double theta6 = theta4*theta2;
  if (!isfinite(theta4)
      || theta4 <= 0.0
      || !isfinite(theta6)
      || theta6 <= 0.0) {
    aperture_geometry_error_3pcf("Gaussian radius powers");
  }
  const double aperture_sq[3] = {
    1.0/theta2, radius2_sq/theta2, radius3_sq/theta2
  };

  double complex centroid[3];
  double distance_sq[3];
  double complex filter_f[3];
  double complex filter_g_conj[3];

  centroid[0] = (side + other)/3.0;
  centroid[1] = centroid[0] - other;
  centroid[2] = centroid[0] - side;

  // One iteration supplies the radial distance and filter deformation at
  // one vertex. next and last keep the cyclic order of the three shears.
  // Conjugation changes i to -i and thus reverses a complex phase.
  for (int vertex=0; vertex<3; vertex++) {
    const int next = (vertex + 1)%3;
    const int last = (vertex + 2)%3;
    const double real_part = creal(centroid[vertex]);
    const double imag_part = cimag(centroid[vertex]);

    // A centroid can approach a vertex even with 0<phi<pi. Its squared
    // distance may then underflow to zero in double precision. Check it
    // before forming the projected direction; no angle is clipped.
    distance_sq[vertex] = real_part*real_part + imag_part*imag_part;
    if (!isfinite(distance_sq[vertex])
        || distance_sq[vertex] <= 0.0) {
      aperture_geometry_error_3pcf("centroid squared distance");
    }
    const double complex direction =
      (centroid[next] - centroid[last])/centroid[vertex];

    filter_f[vertex] = (aperture_sq[next] + aperture_sq[last])/2.0
      + (aperture_sq[next] - aperture_sq[last])*direction/6.0;
    filter_g_conj[vertex] = conj(aperture_sq[next]*aperture_sq[last]
      + (aperture_sq[last] - aperture_sq[next])
        *aperture_sq[vertex]*direction/3.0);
  }

  // Z is the Gaussian exponent after integrating the common aperture
  // center. theta4 restores its area normalization in units of R1.
  const double exponent =
    ((2.0*aperture_sq[1] + 2.0*aperture_sq[2] - aperture_sq[0])
       *distance_sq[0]
     + (2.0*aperture_sq[2] + 2.0*aperture_sq[0] - aperture_sq[1])
       *distance_sq[1]
     + (2.0*aperture_sq[0] + 2.0*aperture_sq[1] - aperture_sq[2])
       *distance_sq[2])/(6.0*theta2);
  const double radial_product =
    distance_sq[0]*distance_sq[1]*distance_sq[2];
  // The conjugated-shear kernel divides by this product. Individually
  // representable distances need not have a representable product.
  if (!isfinite(radial_product)
      || radial_product <= 0.0) {
    aperture_geometry_error_3pcf("centroid distance product");
  }
  const double prefactor = -aperture_sq[0]*aperture_sq[1]
    *aperture_sq[2]*exp(-exponent)/theta4;
  const double complex all_f_conj =
    conj(filter_f[0])*conj(filter_f[1])*conj(filter_f[2]);

  kernel[0] = prefactor*radial_product*all_f_conj*all_f_conj
    /(24.0*theta6);

  // Conjugating one shear adds the lower-degree Gaussian moments in
  // Eq. (68). The three terms have degrees six, four and two in centroid
  // lengths. b carries the phase difference for the conjugated vertex.
  for (int vertex=0; vertex<3; vertex++) {
    const int next = (vertex + 1)%3;
    const int last = (vertex + 2)%3;
    const double complex centroid_conj = conj(centroid[vertex]);
    const double complex phase_b = centroid_conj*centroid_conj
      *centroid[next]*centroid[last];
    const double complex other_f_conj =
      conj(filter_f[next])*conj(filter_f[last]);
    const double complex sixth = radial_product
      *filter_f[vertex]*filter_f[vertex]*other_f_conj*other_f_conj
      /(24.0*theta6);
    const double complex fourth = phase_b*filter_f[vertex]
      *other_f_conj*filter_g_conj[vertex]/(9.0*theta4);
    const double complex second =
      (phase_b*phase_b*filter_g_conj[vertex]*filter_g_conj[vertex]
       + 2.0*aperture_sq[next]*aperture_sq[last]
         *distance_sq[next]*distance_sq[last]*phase_b*other_f_conj)
      /(27.0*theta2*radial_product);

    kernel[vertex + 1] = prefactor*(sixth - fourth + second);
  }
}


// ---------------------------------------------------------------------------
// Construct reusable weights on the measured finite triangle grid.
//
// PHYSICAL DERIVATION & LOGIC FLOW
// The polar area measure is s ds d^2t/(2 pi). With logarithmic side
// bins, ds = s dlnd and d|t| = |t| dlnd, giving
//   measure = s^2 |t|^2 dlnd^2 dphi/(2 pi).
// s=d2/R1 and |t|=d3/R1 make this measure dimensionless. The SAS grid
// stores one orientation, 0<phi<pi. Adding the kernel with R2 and R3
// exchanged supplies the mirrored-radius contribution in this estimator.
// This is an estimator convention; it does not exchange source-bin labels
// or establish an arbitrary tomographic permutation identity.
//
// Re(sum_mu T_mu Gamma_mu)/4 selects the real aperture third moment.
// Include the measure and 1/4 here so applying the weights is only a
// complex dot product. The input Gamma must already have the projection
// and component ordering declared in aperture_3pcf.h.
//
// Cache invalidation:
// This function always overwrites all weights. A caller retaining them
// must rebuild after any coordinate, bin width, radius or convention
// changes, even when dimensions are unchanged. Cosmology alone does not
// change these geometric weights.
// Thread safety:
// Each worker owns one (filter, first-side-bin) slice. Other workers read
// the same inputs but write disjoint slices. An active enclosing OpenMP
// team suppresses a nested team; its caller must own separate outputs.
// ---------------------------------------------------------------------------
void aperture_weights_3pcf(
  const int nd2,                 // first-side count
  const int nd3,                 // second-side count
  const int nphi,                // opening-angle count
  const int nfilter,             // aperture-triplet count
  const double* d2,             // radians [nd2]
  const double* d3,             // radians [nd3]
  const double* phi,            // radians [nphi]
  const double dlnd,            // logarithmic side width
  const double dphi,            // angle width, radians
  const double* filters,        // radians [nfilter,3]
  double* weights_real,         // output [nfilter,4,ntriangle]
  double* weights_imag          // output [nfilter,4,ntriangle]
)
{
  const double two_pi = 6.283185307179586476925286766559;
  const size_t ntriangle = (size_t)nd2*(size_t)nd3*(size_t)nphi;
  const double bin_factor = 0.25*dlnd*dlnd*dphi/two_pi;

  // A slice uses one R1 and d2 for all second sides and opening angles.
  // collapse(2) distributes both outer indices, providing more tasks than
  // the number of aperture triplets alone. schedule(static) assigns fixed
  // slices to workers. No worker contributes to another worker's output.
#ifdef _OPENMP
  // Cap the team by the number of independent geometric slices.
  const size_t jobs = (size_t)nfilter*nd2;
  int workers = omp_in_parallel() ? 1 : omp_get_max_threads();
  if (jobs > 0 && jobs < (size_t)workers) {
    workers = (int)jobs;
  }
#pragma omp parallel for collapse(2) schedule(static) \
  num_threads(workers) if(!omp_in_parallel())
#endif
  for (int filter=0; filter<nfilter; filter++) {
    for (int first=0; first<nd2; first++) {
      const double radius1 = filters[3*(size_t)filter];
      const double ratio2 = filters[3*(size_t)filter + 1]/radius1;
      const double ratio3 = filters[3*(size_t)filter + 2]/radius1;
      const double side = d2[first]/radius1;

      // Each second-side bin fixes the remaining radial area factor.
      // No equality of d2 and d3 grids is assumed; both use width dlnd.
      for (int second=0; second<nd3; second++) {
        const double other_length = d3[second]/radius1;
        // Check both side ratios before forming the complex triangle and
        // its area measure. Physical inputs may be finite while a ratio
        // overflows or underflows; either case needs a clear refusal.
        if (!isfinite(side)
            || side <= 0.0
            || !isfinite(other_length)
            || other_length <= 0.0) {
          aperture_geometry_error_3pcf("side-to-radius ratios");
        }
        const double measure = bin_factor*side*side
          *other_length*other_length;

        // Each angle specifies a complex triangle with the first side
        // along the real axis. Keep phi as the fastest-changing index.
        for (int angle=0; angle<nphi; angle++) {
          const double complex other = other_length
            *(cos(phi[angle]) + I*sin(phi[angle]));
          double complex direct[4];
          double complex mirrored[4];
          const size_t triangle =
            ((size_t)first*(size_t)nd3 + (size_t)second)*(size_t)nphi
            + (size_t)angle;

          aperture_kernel_3pcf(side, other, ratio2, ratio3, direct);
          aperture_kernel_3pcf(side, other, ratio3, ratio2, mirrored);

          // Store one row per natural component. Both orientations use
          // the same physical bin measure and contribute before taking
          // a real part with the supplied complex correlation function.
          for (int component=0; component<4; component++) {
            const size_t offset =
              (4*(size_t)filter + (size_t)component)*ntriangle + triangle;
            const double complex weight =
              measure*(direct[component] + mirrored[component]);

            weights_real[offset] = creal(weight);
            weights_imag[offset] = cimag(weight);
          }
        }
      }
    }
  }
}


// ---------------------------------------------------------------------------
// Contract fixed geometric weights with one or more source combinations.
//
// PHYSICAL DERIVATION & LOGIC FLOW
// For weight w=wr+i wi and correlation gamma=gr+i gi,
//   Re(w gamma) = wr*gr - wi*gi.
// Sum this expression over the four natural components and every triangle.
// Each source combination is independent, and all can share the same
// geometric weights. Any multiplicative shear calibration belongs in the
// caller's supplied correlations or in a later calibration of the result.
//
// Cache invalidation:
// No cache or allocation. Weights and correlations are borrowed read-only.
// Thread safety:
// One worker owns a source combination and a pair of aperture outputs.
// SIMD lanes hold those two outputs and accumulate components and then
// triangles in a fixed order. No horizontal sum or OpenMP reduction mixes
// the lanes, so thread count cannot change the order for an output.
// ---------------------------------------------------------------------------
void aperture_apply_3pcf(
  const int nfilter,             // aperture-triplet count
  const int ntriangle,           // flattened triangle count
  const int nsample,             // source-combination count
  const double* wr,             // [nfilter,4,ntriangle]
  const double* wi,             // [nfilter,4,ntriangle]
  const double* gamma_real,     // [nsample,4,ntriangle]
  const double* gamma_imag,     // [nsample,4,ntriangle]
  double* out                    // [nsample,nfilter]
)
{
  const size_t row_size = 4*(size_t)ntriangle;
  const int filter_pairs = nfilter/2 + nfilter%2;

  // Scalar meaning for each independent output lane:
  //   sum += wr[row]*gamma_real[row] - wi[row]*gamma_imag[row];
  // Both lanes read the same correlation and use different aperture
  // weights. A final unpaired aperture is duplicated into the other lane;
  // that duplicate is discarded. This avoids reading beyond the buffers.
#ifdef _OPENMP
  // A single source triplet often asks for only a few aperture radii.
  // More workers than independent output pairs would remain idle. This
  // cap changes only scheduling, never the component/triangle sum order.
  const size_t jobs = (size_t)nsample*filter_pairs;
  int workers = omp_in_parallel() ? 1 : omp_get_max_threads();
  if (jobs > 0 && jobs < (size_t)workers) {
    workers = (int)jobs;
  }
#pragma omp parallel for collapse(2) schedule(static) \
  num_threads(workers) if(!omp_in_parallel())
#endif
  for (int sample=0; sample<nsample; sample++) {
    for (int pair=0; pair<filter_pairs; pair++) {
      const int first_filter = 2*pair;
      int second_filter = first_filter + 1;
      if (second_filter == nfilter) {
        second_filter = first_filter;
      }

      // The caller guarantees separate input and output storage. Access
      // through these local restrict pointers lets the compiler rely on
      // writes to out never changing a weight or correlation being read.
      const double* restrict first_wr = wr + first_filter*row_size;
      const double* restrict first_wi = wi + first_filter*row_size;
      const double* restrict second_wr = wr + second_filter*row_size;
      const double* restrict second_wi = wi + second_filter*row_size;
      const double* restrict real_row = gamma_real + sample*row_size;
      const double* restrict imag_row = gamma_imag + sample*row_size;

      // Initialize both independent aperture sums to zero.
      v2d vSum = simde_mm_setzero_pd();

      // Flat row order visits component 0 through 3 and, within each,
      // every supplied triangle. The two lanes retain this same order.
      for (size_t row=0; row<row_size; row++) {
        // Lane 0 receives the first aperture; lane 1 receives the second.
        const v2d vWeightReal =
          simde_mm_set_pd(second_wr[row], first_wr[row]);
        // Use the same lane order for the imaginary weight coefficients.
        const v2d vWeightImag =
          simde_mm_set_pd(second_wi[row], first_wi[row]);
        // Broadcast one real correlation value to both aperture lanes.
        const v2d vGammaReal = simde_mm_set1_pd(real_row[row]);
        // Broadcast its imaginary part to those same two lanes.
        const v2d vGammaImag = simde_mm_set1_pd(imag_row[row]);

        // Multiply wr*gr separately for each aperture output.
        const v2d vRealProduct = simde_mm_mul_pd(vWeightReal, vGammaReal);
        // Multiply wi*gi separately for each aperture output.
        const v2d vImagProduct = simde_mm_mul_pd(vWeightImag, vGammaImag);
        // The real part of w*gamma is wr*gr-wi*gi, without conjugation.
        const v2d vContribution =
          simde_mm_sub_pd(vRealProduct, vImagProduct);
        // Add this triangle/component to each output's existing sum.
        vSum = simde_mm_add_pd(vSum, vContribution);
      }

      double values[2];
      // Store lane 0 at values[0] and lane 1 at values[1]. storeu needs
      // two valid doubles but does not require a vector-aligned address.
      simde_mm_storeu_pd(values, vSum);

      out[(size_t)sample*(size_t)nfilter + (size_t)first_filter] = values[0];
      if (second_filter != first_filter) {
        out[(size_t)sample*(size_t)nfilter + (size_t)second_filter] =
          values[1];
      }
    }
  }
}
