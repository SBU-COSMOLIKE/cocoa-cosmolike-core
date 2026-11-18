#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <spdlog/spdlog.h>
#include "threepcf_wrapper.hpp"
#include "aperture_3pcf.h"
#include "covariance_3pcf.h"

namespace cosmolike_interface {

// ---------------------------------------------------------------------------
// threepcf_wrapper.cpp: notebook adapters for the aperture and covariance
// kernels. Each wrapper exists to bridge two storage orders: the C
// kernels read flat row-major buffers, where the last index changes
// fastest, while Armadillo stores its first index fastest. An explicit
// copy changes only the memory order - entry (field, field, ell) keeps
// that meaning on both sides. Every integral, SIMD loop and OpenMP
// region stays in C; these functions validate, repack and call.
//
// Cache invalidation:
// These wrappers keep no stored result and no mutable model state.
// Every call owns its converted inputs and its output, so a later call
// cannot replace values an earlier caller still holds.
// ---------------------------------------------------------------------------

// Abort with a diagnostic when a notebook input breaks its contract.
// The notebook route favors a hard stop over an exception: a partial
// covariance must never reach an interactive session silently.
static void require_3pcf(
    const bool condition, // required input condition
    const char* message // diagnostic naming the violated contract
  )
{
  if (!condition) {
    spdlog::critical("3pcf notebook: {}", message);
    std::exit(1);
  }
}


// Convert an Armadillo axis length to the C int the kernels take,
// aborting when the length is zero or exceeds the int range.
static int count_3pcf(
    const arma::uword count, // axis length passed to a C int argument
    const char* name // axis description for the error message
  )
{
  if (count < 1
      || count > INT_MAX) {
    spdlog::critical("3pcf notebook: invalid {} count", name);
    std::exit(1);
  }
  return static_cast<int>(count);
}


// The Gaussian kernel fills one covariance triangle and mirrors it, so
// it requires symmetric field correlations. This check accepts only
// rounding-scale differences relative to the entries themselves; an
// absolute tolerance of order one would hide real asymmetry in small
// correlations.
//
// Why 64 machine epsilons: the caller typically builds K_ij and K_ji by
// two separate floating-point sums (filtered_pair_3pcf evaluates every
// ordered pair independently), so the two orderings can disagree by a
// few dozen roundings even for an exactly symmetric model. The factor
// 64 leaves that accumulation headroom while still rejecting genuine
// asymmetry, which enters at a far larger scale.
static bool symmetric_values_3pcf(
    const double first, // K_ij or its zero-separation value
    const double second // K_ji at the same separation
  )
{
  const double scale = std::max(std::abs(first), std::abs(second));
  const double tolerance = 64.0*std::numeric_limits<double>::epsilon();
  return std::abs(first-second) <= tolerance*scale;
}


// Pack an Armadillo matrix into the row-major order C reads: C stores a
// matrix row together, so a [2,3] matrix maps entry (1,2) to flat
// offset 1*3+2 = 5. The returned column owns a contiguous copy; the C
// kernel never sees the caller's array or Armadillo's own layout.
template <typename Element>
static arma::Col<Element> pack_matrix_3pcf(
    const arma::Mat<Element>& input // matrix with physical axes unchanged
  )
{
  arma::Col<Element> packed(input.n_elem);
  for (arma::uword row=0; row<input.n_rows; row++) {
    for (arma::uword column=0; column<input.n_cols; column++) {
      packed[row*input.n_cols+column] = input(row, column);
    }
  }
  return packed;
}


// Extend the same row-major rule to three axes: cell (first, second,
// third) lives at flat offset (first*ncols + second)*nslices + third.
static arma::Col<double> pack_cube_3pcf(
    const arma::Cube<double>& input // [first,second,third] physical axes
  )
{
  arma::Col<double> packed(input.n_elem);
  for (arma::uword first=0; first<input.n_rows; first++) {
    for (arma::uword second=0; second<input.n_cols; second++) {
      for (arma::uword third=0; third<input.n_slices; third++) {
        const arma::uword offset =
            (first*input.n_cols+second)*input.n_slices+third;
        packed[offset] = input(first, second, third);
      }
    }
  }
  return packed;
}


// Reverse the storage-order conversion on a C result. The returned
// Armadillo object owns its values, including when CARMA, the
// NumPy-Armadillo bridge library, later exports it to Python.
static arma::Mat<double> unpack_matrix_3pcf(
    const arma::Col<double>& packed, // row-major C output
    const int nrow, // first output axis
    const int ncolumn // second output axis
  )
{
  arma::Mat<double> output(nrow, ncolumn);
  for (int row=0; row<nrow; row++) {
    for (int column=0; column<ncolumn; column++) {
      const arma::uword offset = arma::uword(row)*ncolumn+column;
      output(row, column) = packed[offset];
    }
  }
  return output;
}


// Reverse pack_cube_3pcf's rule on a last-index-fastest C result.
static arma::Cube<double> unpack_cube_3pcf(
    const arma::Col<double>& packed, // last-index-fastest C output
    const int nfirst, // first physical axis
    const int nsecond, // second physical axis
    const int nthird // third physical axis
  )
{
  arma::Cube<double> output(nfirst, nsecond, nthird);
  for (int first=0; first<nfirst; first++) {
    for (int second=0; second<nsecond; second++) {
      for (int third=0; third<nthird; third++) {
        const arma::uword offset =
            (arma::uword(first)*nsecond+second)*nthird+third;
        output(first, second, third) = packed[offset];
      }
    }
  }
  return output;
}


// Prepare the finite triangle-bin aperture weights. The shared C kernel
// includes both mirrored triangle configurations and the final factor
// 1/4, so the two returned cubes contract directly with natural
// components; the caller applies no further factor.
aperture_weights_result_3pcf aperture_weights_cpp_3pcf(
    const arma::Col<double>& d2, // [nd2], side-bin centers in radians
    const arma::Col<double>& d3, // [nd3], side-bin centers in radians
    const arma::Col<double>& phi, // [nphi], opening-angle centers
    const double dlnd, // logarithmic side-bin width
    const double dphi, // opening-angle width in radians
    const arma::Mat<double>& filters // [nfilter,3], radii in radians
  )
{
  const int nd2 = count_3pcf(d2.n_elem, "d2");
  const int nd3 = count_3pcf(d3.n_elem, "d3");
  const int nphi = count_3pcf(phi.n_elem, "phi");
  const int nfilter = count_3pcf(filters.n_rows, "filter");
  require_3pcf(filters.n_cols == 3, "filters must have three columns");
  require_3pcf(d2.is_finite()
      && d3.is_finite()
      && phi.is_finite()
      && filters.is_finite(), "aperture inputs must be finite");
  require_3pcf(std::isfinite(dlnd)
      && dlnd > 0.0
      && std::isfinite(dphi)
      && dphi > 0.0, "bin widths must be positive");
  require_3pcf(d2.min() > 0.0
      && d3.min() > 0.0
      && filters.min() > 0.0, "side lengths and radii must be positive");
  require_3pcf(phi.min() > 0.0
      && phi.max() < std::acos(-1.0),
      "phi must lie strictly between zero and pi");
  require_3pcf(nd2 <= INT_MAX/nd3, "triangle count exceeds C int range");
  require_3pcf(nd2*nd3 <= INT_MAX/nphi,
      "triangle count exceeds C int range");
  const int ntriangle = nd2*nd3*nphi;

  const arma::Col<double> packed_filters = pack_matrix_3pcf(filters);
  const arma::uword nweight = arma::uword(nfilter)*4*ntriangle;
  arma::Col<double> weights_real(nweight);
  arma::Col<double> weights_imag(nweight);
  aperture_weights_3pcf(nd2, nd3, nphi, nfilter, d2.memptr(), d3.memptr(),
      phi.memptr(), dlnd, dphi, packed_filters.memptr(),
      weights_real.memptr(), weights_imag.memptr());
  require_3pcf(weights_real.is_finite()
      && weights_imag.is_finite(), "aperture weights are not finite");

  aperture_weights_result_3pcf result;
  result.real = unpack_cube_3pcf(weights_real, nfilter, 4, ntriangle);
  result.imag = unpack_cube_3pcf(weights_imag, nfilter, 4, ntriangle);
  return result;
}


// Contract prepared weights with natural components for several source
// triplets or model samples at once. The C routine takes the real part
// of the product weight*gamma without conjugating either factor, so the
// caller must pass both imaginary arrays: the real part of a complex
// product mixes them with the real parts. The output [sample,filter]
// holds one real aperture moment per sample and radius triplet.
arma::Mat<double> aperture_apply_cpp_3pcf(
    const arma::Cube<double>& weights_real, // [filter,4,triangle]
    const arma::Cube<double>& weights_imag, // matching imaginary weights
    const arma::Cube<double>& gamma_real, // [sample,4,triangle]
    const arma::Cube<double>& gamma_imag // matching imaginary components
  )
{
  const int nfilter = count_3pcf(weights_real.n_rows, "filter");
  const int ntriangle = count_3pcf(weights_real.n_slices, "triangle");
  const int nsample = count_3pcf(gamma_real.n_rows, "sample");
  require_3pcf(weights_real.n_cols == 4
      && gamma_real.n_cols == 4,
      "aperture cubes must have four natural components");
  require_3pcf(arma::size(weights_real) == arma::size(weights_imag)
      && arma::size(gamma_real) == arma::size(gamma_imag)
      && gamma_real.n_slices == weights_real.n_slices,
      "real/imaginary cube shapes and triangle counts must agree");
  require_3pcf(weights_real.is_finite()
      && weights_imag.is_finite()
      && gamma_real.is_finite()
      && gamma_imag.is_finite(),
      "aperture cubes must be finite");

  const arma::Col<double> packed_wr = pack_cube_3pcf(weights_real);
  const arma::Col<double> packed_wi = pack_cube_3pcf(weights_imag);
  const arma::Col<double> packed_gr = pack_cube_3pcf(gamma_real);
  const arma::Col<double> packed_gi = pack_cube_3pcf(gamma_imag);
  arma::Col<double> output(arma::uword(nsample)*nfilter);
  aperture_apply_3pcf(nfilter, ntriangle, nsample, packed_wr.memptr(),
      packed_wi.memptr(), packed_gr.memptr(), packed_gi.memptr(),
      output.memptr());
  require_3pcf(output.is_finite(), "aperture result is not finite");
  return unpack_matrix_3pcf(output, nsample, nfilter);
}


// Pair correlation of filtered fields: the C kernel transforms the
// caller's filtered angular power with J0, the cylindrical Bessel
// function of the first kind and order zero. The caller supplies the
// d ell quadrature weights; C inserts the factor ell/(2 pi).
arma::Cube<double> filtered_pair_cpp_3pcf(
    const arma::Col<double>& ell, // [nell], angular multipoles
    const arma::Col<double>& ell_weights, // [nell], d ell weights
    const arma::Col<double>& radius, // [nr], separation in radians
    const arma::Mat<double>& filters, // [field,nell], Fourier filters
    const arma::Cube<double>& power // [field,field,nell], angular power
  )
{
  const int nell = count_3pcf(ell.n_elem, "multipole");
  const int nr = count_3pcf(radius.n_elem, "separation");
  const int nfield = count_3pcf(filters.n_rows, "field");
  require_3pcf(ell_weights.n_elem == ell.n_elem
      && filters.n_cols == ell.n_elem
      && power.n_rows == filters.n_rows
      && power.n_cols == filters.n_rows
      && power.n_slices == ell.n_elem, "filtered-pair axes disagree");
  require_3pcf(ell.is_finite()
      && ell_weights.is_finite()
      && radius.is_finite()
      && filters.is_finite()
      && power.is_finite(),
      "filtered-pair inputs must be finite");
  require_3pcf(ell.min() >= 0.0
      && radius.min() >= 0.0, "multipoles and separations must be >=0");

  const arma::Col<double> packed_filters = pack_matrix_3pcf(filters);
  const arma::Col<double> packed_power = pack_cube_3pcf(power);
  arma::Col<double> output(arma::uword(nfield)*nfield*nr);
  filtered_pair_3pcf(nfield, nell, nr, ell.memptr(), ell_weights.memptr(),
      radius.memptr(), packed_filters.memptr(), packed_power.memptr(),
      output.memptr());
  require_3pcf(output.is_finite(), "filtered-pair result is not finite");
  return unpack_cube_3pcf(output, nfield, nfield, nr);
}


// Covariance from responses the caller evaluates on common nodes: the C
// kernel forms sum_node w H_row H_column. The caller places the entire
// physical measure in the weights; this wrapper adds no area, abundance
// or distance factor and chooses no halo model.
arma::Mat<double> connected_gram_cpp_3pcf(
    const arma::Mat<double>& responses, // [row,node], halo responses
    const arma::Col<double>& weights // [node], nonnegative weights
  )
{
  const int nrow = count_3pcf(responses.n_rows, "response row");
  const int nnode = count_3pcf(responses.n_cols, "response node");
  require_3pcf(weights.n_elem == responses.n_cols,
      "response-node and weight counts must agree");
  require_3pcf(responses.is_finite()
      && weights.is_finite(),
      "response inputs must be finite");
  require_3pcf(weights.min() >= 0.0, "Gram weights must be nonnegative");

  const arma::Col<double> packed_responses = pack_matrix_3pcf(responses);
  arma::Col<double> output(arma::uword(nrow)*nrow);
  connected_gram_3pcf(nrow, nnode, packed_responses.memptr(),
      weights.memptr(), output.memptr());
  require_3pcf(output.is_finite(), "Gram result is not finite");
  return unpack_matrix_3pcf(output, nrow, nrow);
}


// Gaussian covariance of raw cubic aperture fields. For Gaussian
// fields, Wick's theorem splits the expectation of a product of six
// fields into products of pair expectations; these splittings are the
// Wick pairings. The C kernel keeps all fifteen position pairings of
// two field triplets, including the pairs that join two fields at the
// same aperture center. A catalogue estimator over distinct galaxies
// drops its self-pair noise terms, which this kernel does not subtract.
arma::Mat<double> gaussian_aperture_covariance_cpp_3pcf(
    const arma::Mat<int>& triplets, // [row,3], field labels
    const arma::Cube<double>& pair, // [field,field,r], correlations
    const arma::Mat<double>& pair_zero, // [field,field], r=0 values
    const arma::Col<double>& radial_weights // [nr], radial window measure
  )
{
  const int nrow = count_3pcf(triplets.n_rows, "triplet");
  const int nfield = count_3pcf(pair.n_rows, "field");
  const int nr = count_3pcf(pair.n_slices, "separation");
  require_3pcf(triplets.n_cols == 3
      && pair.n_cols == pair.n_rows
      && pair_zero.n_rows == pair.n_rows
      && pair_zero.n_cols == pair.n_rows
      && radial_weights.n_elem == pair.n_slices,
      "Gaussian covariance axes disagree");
  require_3pcf(pair.is_finite()
      && pair_zero.is_finite()
      && radial_weights.is_finite(), "Gaussian inputs must be finite");
  require_3pcf(radial_weights.min() >= 0.0,
      "radial window weights must be nonnegative");

  // The C loop fills one covariance triangle and reflects it. That is
  // the same Gaussian covariance only for symmetric field correlations.
  for (int first=0; first<nfield; first++) {
    for (int second=0; second<first; second++) {
      require_3pcf(symmetric_values_3pcf(pair_zero(first, second),
          pair_zero(second, first)), "pair_zero must be symmetric");
      for (int radial=0; radial<nr; radial++) {
        require_3pcf(symmetric_values_3pcf(pair(first, second, radial),
            pair(second, first, radial)), "pair must be symmetric");
      }
    }
  }

  const arma::Col<int> packed_triplets = pack_matrix_3pcf(triplets);
  const arma::Col<double> packed_pair = pack_cube_3pcf(pair);
  const arma::Col<double> packed_zero = pack_matrix_3pcf(pair_zero);
  arma::Col<double> output(arma::uword(nrow)*nrow);
  gaussian_aperture_covariance_3pcf(nfield, nr, nrow,
      packed_triplets.memptr(), packed_pair.memptr(), packed_zero.memptr(),
      radial_weights.memptr(), output.memptr());
  require_3pcf(output.is_finite(), "Gaussian result is not finite");
  return unpack_matrix_3pcf(output, nrow, nrow);
}

}
