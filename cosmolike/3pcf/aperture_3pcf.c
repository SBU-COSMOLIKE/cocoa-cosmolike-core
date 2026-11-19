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
#include "log.c/src/log.h"

// ===========================================================================
// FINITE TRIANGLE INTEGRATION FOR APERTURE MASS
// ===========================================================================
// Aperture mass averages the projected mass density inside a circular
// window with a compensated filter: the filter integrates to zero, so
// a constant density sheet contributes nothing. The third moment of
// the aperture mass measures the skewness of the projected density
// field, and the four complex three-point correlations of the shear
// contain the information to rebuild that moment. The kernels in this
// file belong to the Gaussian aperture filter, whose Fourier transform
// is U(ell*R) = (ell*R)^2 exp[-(ell*R)^2/2]/2, with ell the angular
// multipole and R the aperture radius.
//
// The physical equations come from Schneider, Kilbinger & Lombardi
// (2005), A&A 431, 9, Eqs. (62)-(69),
// https://arxiv.org/abs/astro-ph/0308328. Their equal-radius limit is
// the aperture statistic of Jarvis, Bernstein & Jain (2004), MNRAS
// 352, 338. TreeCorr's _calculateT/calculateMap3 implementation by
// Mike Jarvis provides the centroid/SAS convention. The application to
// fastnc natural components follows the work of Sunao Sugiyama and
// Rafael Gomes; repository credits and license notices accompany this
// module. These routines implement the published math in C.
//
// A finite side-angle-side (SAS) grid labels a triangle by two side
// lengths and the angle between them. This file integrates the
// triangle bins the caller passes in, and nothing else. A finite set
// of bins is not the infinite-domain aperture statistic. Finite
// coverage and quadrature can therefore mix E modes (the curl-free
// shear pattern that gravitational lensing produces) with B modes
// (the divergence-free pattern), even when the physical field carries
// E modes only.
//
// Arrays are flat, caller-owned C buffers, and this file reads no core
// cosmology state. The caller checks units, bin definitions, sizes,
// finite inputs and the absence of overlap between buffers. In
// particular, the opening angle phi lies strictly between zero and pi.
// That bound excludes collinear triangles, whose centroid can coincide
// with a vertex; at such a vertex the direction that projects the
// spin-2 shear is undefined. Nearly collinear triangles stay inside
// the domain, so the caller still needs to check their numerical
// accuracy.

// SIMD (single instruction, multiple data) applies one operation to
// several numbers at once; each number occupies a vector position
// called a lane. The type v2d holds two doubles side by side. In this
// file each lane carries a separate aperture output and accumulates
// its own triangle sum in a fixed order; the code never adds the two
// lanes together. SIMDe translates each operation into the processor's
// vector instructions when they exist, and into portable C when they
// do not. This file keeps no optional scalar implementation.
typedef simde__m128d v2d;


// A positive physical length can still overflow or underflow the
// dimensionless products below. The checks in this file stop the
// program before a zero or infinite denominator enters a kernel,
// because replacing such a value by a clipped one would change the
// kernel silently. Each check names the arithmetic stage that failed
// through log_fatal and then exits.


// ---------------------------------------------------------------------------
// Four complex kernels for one ordered set of aperture radii.
//
// PHYSICAL DERIVATION & LOGIC FLOW
// Work in units of the first aperture radius R1. The triangle then has
// its vertices at 0, at the real number side = d2/R1, and at the
// complex number other = (d3/R1) exp(i phi). Each vertex sits at a
// complex offset q_i from the triangle centroid, and the three offsets
// add to zero because the centroid is their mean.
//
// The offsets carry the two pieces of the kernel. Their squared
// lengths |q_i|^2 feed the Gaussian aperture convolution, and these
// are invariant when the whole triangle rotates. Their phases record
// how each vertex is oriented around the centroid. A shear is a
// spin-2 quantity: rotating the reference direction shifts its
// complex phase by twice the rotation angle, and the phases of the
// q_i supply exactly that rotation at each vertex.
//
// Unequal radii (R1, R2, R3) enter through the dimensionless weights
//
//   a_i = (R_i/R1)^2 / Theta2,
//   Theta2 = sqrt[(R1^2 R2^2 + R1^2 R3^2 + R2^2 R3^2)/(3 R1^4)],
//
// where Theta2 averages the three pairwise products of squared radii
// and takes a square root: an effective squared radius in units of
// R1^2, so for equal radii every a_i is 1. The weights build the
// dimensionless filter functions f_i and g_i of Schneider, Kilbinger
// & Lombardi 2005 (astro-ph/0308328, Eqs. 65 and 69), which carry the
// whole effect of the unequal filter sizes.
//
// The four kernels differ only in which shear is conjugated. Kernel 0
// conjugates none. Kernel i (i = 1, 2, 3) conjugates the shear at the
// vertex with centroid offset q_i. The loop below cycles through the
// vertices because each kernel evaluates the same published
// expression, read from a different starting vertex.
//
// Cache invalidation: no cache. The inputs describe one triangle and
// one ordered radius triplet, so there is nothing to reuse.
// Thread safety: the helper writes only local arrays and the caller's
// four output values. static keeps it private to this file; inline
// lets the compiler expand calls while an unoptimized debug build
// still keeps a callable definition.
// ---------------------------------------------------------------------------
static inline void aperture_kernel_3pcf(
  const double side,              // d2/R1, positive
  const double complex other,     // (d3/R1)*exp(i*phi)
  const double ratio2,            // R2/R1, positive
  const double ratio3,            // R3/R1, positive
  double complex* kernel         // output [4]; no bin measure yet
)
{
  // This guard protects the Gaussian normalization: the squared radius
  // ratios feed it, and a ratio near sqrt(DBL_MAX) squares to
  // infinity. The guard rejects such an input before its square can
  // reach a square root or a divisor. The bound comes from double
  // precision alone, not from an angular or physical cut.
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
    log_fatal("aperture_weights_3pcf: geometry is not representable at "
      "dimensionless lengths; check nearly collinear triangles and "
      "length/radius ratios");
    exit(1);
  }

  const double radius2_sq = ratio2*ratio2;
  const double radius3_sq = ratio3*ratio3;
  if (!isfinite(radius2_sq)
      || radius2_sq <= 0.0
      || !isfinite(radius3_sq)
      || radius3_sq <= 0.0) {
    log_fatal("aperture_weights_3pcf: geometry is not representable at "
      "squared radius ratios; check nearly collinear triangles and "
      "length/radius ratios");
    exit(1);
  }

  const double theta_argument = (radius2_sq + radius3_sq
                                + radius2_sq*radius3_sq)/3.0;
  if (!isfinite(theta_argument)
      || theta_argument <= 0.0) {
    log_fatal("aperture_weights_3pcf: geometry is not representable at "
      "Gaussian radius normalization; check nearly collinear "
      "triangles and length/radius ratios");
    exit(1);
  }

  const double theta2 = sqrt(theta_argument);
  const double theta4 = theta2*theta2;
  const double theta6 = theta4*theta2;
  if (!isfinite(theta4)
      || theta4 <= 0.0
      || !isfinite(theta6)
      || theta6 <= 0.0) {
    log_fatal("aperture_weights_3pcf: geometry is not representable at "
      "Gaussian radius powers; check nearly collinear triangles and "
      "length/radius ratios");
    exit(1);
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

  // Loop overview: one iteration computes, for one vertex, its squared
  // distance from the centroid and the two filter values f_i and g_i
  // that encode the unequal aperture radii. The indices next and last
  // walk the other two vertices in cyclic order, because the published
  // filter at a vertex mixes the aperture weights a_i of its two
  // neighbors. conj changes i to -i and therefore reverses a complex
  // phase.
  for (int vertex=0; vertex<3; vertex++) {
    const int next = (vertex + 1)%3;
    const int last = (vertex + 2)%3;
    const double real_part = creal(centroid[vertex]);
    const double imag_part = cimag(centroid[vertex]);

    // This guard protects the projected direction: a centroid offset
    // can approach zero even with 0 < phi < pi, and its squared length
    // can then underflow to exactly zero in double precision. The
    // check runs before the division that forms the direction; the
    // code clips no angle and moves no vertex.
    distance_sq[vertex] = real_part*real_part + imag_part*imag_part;
    if (!isfinite(distance_sq[vertex])
        || distance_sq[vertex] <= 0.0) {
      log_fatal("aperture_weights_3pcf: geometry is not representable at "
        "centroid squared distance; check nearly collinear "
        "triangles and length/radius ratios");
      exit(1);
    }

    const double complex direction =
      (centroid[next] - centroid[last])/centroid[vertex];

    filter_f[vertex] = (aperture_sq[next] + aperture_sq[last])*0.5
      + (aperture_sq[next] - aperture_sq[last])*direction/6.0;
    filter_g_conj[vertex] = conj(aperture_sq[next]*aperture_sq[last]
      + (aperture_sq[last] - aperture_sq[next])
        *aperture_sq[vertex]*direction/3.0);
  }

  // Integrating the triangle over the common aperture center leaves a
  // Gaussian in the centroid offsets; exponent is its argument, the Z
  // of Schneider, Kilbinger & Lombardi 2005 (astro-ph/0308328). The
  // division by theta4 in the prefactor restores the area
  // normalization in units of the first aperture radius R1.
  const double exponent =
    ((2.0*aperture_sq[1] + 2.0*aperture_sq[2] - aperture_sq[0])
       *distance_sq[0]
     + (2.0*aperture_sq[2] + 2.0*aperture_sq[0] - aperture_sq[1])
       *distance_sq[1]
     + (2.0*aperture_sq[0] + 2.0*aperture_sq[1] - aperture_sq[2])
       *distance_sq[2])/(6.0*theta2);
  const double radial_product =
    distance_sq[0]*distance_sq[1]*distance_sq[2];

  // This guard protects the conjugated-shear kernels, which divide by
  // this product: three individually representable distances can
  // still multiply to an overflowing or underflowing product.
  if (!isfinite(radial_product)
      || radial_product <= 0.0) {
    log_fatal("aperture_weights_3pcf: geometry is not representable at "
      "centroid distance product; check nearly collinear triangles "
      "and length/radius ratios");
    exit(1);
  }

  const double prefactor = -aperture_sq[0]*aperture_sq[1]
    *aperture_sq[2]*exp(-exponent)/theta4;
  const double complex all_f_conj =
    conj(filter_f[0])*conj(filter_f[1])*conj(filter_f[2]);

  kernel[0] = prefactor*radial_product*all_f_conj*all_f_conj
    /(24.0*theta6);

  // Loop overview: one iteration builds the kernel that conjugates the
  // shear at one vertex. Conjugating one shear brings in the
  // lower-degree Gaussian moments of Schneider, Kilbinger & Lombardi
  // 2005 (astro-ph/0308328, Eq. 68): the three terms below have
  // degrees six, four and two in the centroid offset lengths. phase_b
  // is the paper's quantity b, the conjugated-squared centroid offset
  // of this vertex times the offsets of the other two; it carries the
  // phase difference that the conjugated vertex introduces.
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
// The aperture third moment integrates the shear correlations over all
// triangle shapes. Write the first side as the real length s and the
// second side as the complex number t; the integral then runs over s
// and over the two-dimensional t with the polar area measure
// s ds d^2t/(2 pi), where d^2t = |t| d|t| dphi. Both side grids bin
// logarithmically, so ds = s dlnd and d|t| = |t| dlnd, and one
// triangle bin carries the measure
//
//   measure = s^2 |t|^2 dlnd^2 dphi/(2 pi).
//
// Measuring the sides in units of the first aperture radius R1, as
// s = d2/R1 and |t| = d3/R1, makes this measure dimensionless.
//
// The SAS grid stores each triangle in one orientation, 0 < phi < pi.
// The loop therefore adds a second kernel with the radii R2 and R3
// exchanged; this mirrored-radius term represents the reflected
// triangle in this estimator. The exchange is an estimator convention:
// it does not exchange source-bin labels, and it does not create a
// tomographic permutation identity.
//
// The four kernels T_mu pair with the four natural components Gamma_mu
// of the shear three-point correlation, and the real part of
// (1/4) sum_mu T_mu Gamma_mu is the real aperture third moment. This
// function folds the bin measure and the factor 1/4 into the weights,
// so applying them later is one complex dot product. The input Gamma
// must already carry the projection and the component ordering that
// aperture_3pcf.h documents.
//
// Cache invalidation: this function always overwrites every weight. A
// caller who keeps the weights must rebuild them after any change of
// coordinates, bin widths, radii or conventions, even when the array
// dimensions stay the same. Cosmology alone never changes these
// purely geometric weights.
// Thread safety: each OpenMP worker owns one (filter, first-side-bin)
// slice of the output. All workers read the same inputs, and no two
// workers write the same slice. When an enclosing OpenMP team is
// already active this function runs inside it without opening a
// nested team, so concurrent callers must pass separate outputs.
// ---------------------------------------------------------------------------
void aperture_weights_3pcf(
  const int nd2,                 // first-side count
  const int nd3,                 // second-side count
  const int nphi,                // opening-angle count
  const int nfilter,             // aperture-triplet count
  const double* d2,             // radians [nd2]
  const double* d3,             // radians [nd3]
  const double* phi,            // radians [nphi]
  const double dlnd,            // logarithmic side-bin width
  const double dphi,            // angle width, radians
  const double* filters,        // radians [nfilter,3]
  double* weights_real,         // output [nfilter,4,ntriangle]
  double* weights_imag          // output [nfilter,4,ntriangle]
)
{
  const double two_pi = 2.0*M_PI;
  const size_t ntriangle = (size_t)nd2*(size_t)nd3*(size_t)nphi;

  // bin_factor carries the constant part of the bin measure and the
  // final factor 1/4 of the aperture estimator; the loop multiplies
  // in the s^2 |t|^2 part separately for each triangle.
  const double bin_factor = 0.25*dlnd*dlnd*dphi/two_pi;

  // Loop overview: one slice fixes the aperture triplet and the first
  // side, then sweeps every second side and opening angle. collapse(2)
  // spreads both outer indices across the team, which offers many
  // more independent tasks than the aperture triplets alone.
  // schedule(static) gives each worker a fixed set of slices, and no
  // worker writes into another worker's output.
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
      // This code never assumes the d2 and d3 grids are equal; it only
      // uses that both share the logarithmic width dlnd.
      for (int second=0; second<nd3; second++) {
        const double other_length = d3[second]/radius1;

        // This guard protects the triangle construction: a physical
        // length can be finite while its ratio to the aperture radius
        // overflows or underflows. The check refuses both cases before
        // the code builds the complex triangle and its area measure
        // from these ratios.
        if (!isfinite(side)
            || side <= 0.0
            || !isfinite(other_length)
            || other_length <= 0.0) {
          log_fatal("aperture_weights_3pcf: geometry is not representable "
            "at side-to-radius ratios; check nearly collinear "
            "triangles and length/radius ratios");
          exit(1);
        }

        const double measure = bin_factor*side*side
          *other_length*other_length;

        // Loop overview: each angle fixes one triangle. The first side
        // lies along the positive real axis, and the second side sits
        // at other_length*(cos(phi) + i sin(phi)) = other_length*
        // exp(i*phi): in this file the opening angle rotates the
        // second side counterclockwise from the first. transform_3pcf.c
        // draws the same angle clockwise because it follows fastnc,
        // whose angle is minus TreeCorr's (arXiv:2407.01798); the
        // correlations that meet these weights arrive in the TreeCorr
        // component order and sign, so each file draws its own
        // convention. phi stays the fastest-changing index, so the
        // flat triangle index below matches the layout the header
        // documents.
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
          // the same physical bin measure, and both enter the weight
          // here, before the caller takes a real part against its
          // complex correlation function.
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
// Write one weight as w = wr + i*wi and one correlation value as
// gamma = gr + i*gi. The real part of their product is
//   Re(w gamma) = wr*gr - wi*gi,
// and the aperture moment sums this expression over the four natural
// components and over every triangle. Each source combination is an
// independent sum, and all of them share the same geometric weights.
// Any multiplicative shear calibration belongs in the correlations the
// caller passes in, or in a calibration the caller applies to the
// result afterward.
//
// Cache invalidation: no cache and no allocation. This function
// borrows the weights and correlations read-only.
// Thread safety: one worker owns one source combination and one pair
// of aperture outputs. The two SIMD lanes hold those two outputs and
// accumulate components, then triangles, in a fixed order. No
// horizontal sum and no OpenMP reduction mixes the lanes, so the
// thread count cannot change the summation order of any output.
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

  // The number of lane pairs is the integer ceiling of nfilter/2, so
  // an odd aperture count gets one final pair that duplicates its
  // single aperture (see the lane comment below).
  const int filter_pairs = nfilter/2 + nfilter%2;

  // Scalar meaning of each independent output lane:
  //   sum += wr[row]*gamma_real[row] - wi[row]*gamma_imag[row];
  // Both lanes read the same correlation values and apply different
  // aperture weights. When nfilter is odd, the last pair loads its
  // single aperture into both lanes, and the guarded store at the end
  // discards the duplicate lane; the duplication keeps every weight
  // read inside the arrays.
#ifdef _OPENMP
  // A single source combination often asks for only a few aperture
  // radii, and workers beyond the number of independent output pairs
  // would sit idle. This cap changes only the scheduling; it never
  // changes the component/triangle summation order.
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

      // The caller guarantees separate input and output storage.
      // Reading through these local restrict pointers lets the
      // compiler assume that a write to out never changes a weight or
      // a correlation that this loop still reads.
      const double* restrict first_wr = wr + first_filter*row_size;
      const double* restrict first_wi = wi + first_filter*row_size;
      const double* restrict second_wr = wr + second_filter*row_size;
      const double* restrict second_wi = wi + second_filter*row_size;
      const double* restrict real_row = gamma_real + sample*row_size;
      const double* restrict imag_row = gamma_imag + sample*row_size;

      // Initialize both independent aperture sums to zero.
      v2d vSum = simde_mm_setzero_pd();

      // Loop overview: the flat row order visits natural component 0
      // through 3 and, inside each component, every triangle the
      // caller passed. Both lanes keep this same order.
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
