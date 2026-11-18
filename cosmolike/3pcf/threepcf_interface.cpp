#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <utility>
#include <carma.h>
#include <spdlog/spdlog.h>
#include "threepcf_interface.hpp"
#include "threepcf_wrapper.hpp"
#include "aperture_3pcf.h"
#include "covariance_3pcf.h"
#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" void openblas_set_num_threads(int nthreads);

namespace py = pybind11;
namespace cosmolike_interface {

// ---------------------------------------------------------------------------
// threepcf_interface.cpp: the pybind entry points of the aperture and
// covariance stage of the shear three-point pipeline. Python supplies
// natural components - the four complex three-shear correlators
// Gamma_0..Gamma_3, distinguished by which of the three shears is
// complex-conjugated (Gamma_0 conjugates none, Gamma_i the shear at
// vertex i; Schneider & Lombardi, arXiv:astro-ph/0207454) - together
// with aperture weights and pair correlations, and receives aperture
// moments - third moments of circularly filtered aperture fields -
// and their Gaussian covariances. Complex quantities travel
// split-complex: the real and imaginary parts go as two separate real
// arrays; the bindings name them wr/wi for the integration weights
// and gr/gi for the natural-component grid.
//
// Two routes reach the same C kernels; C++ owns no numerical integral.
// The production route borrows C-contiguous float64 arrays and
// allocates its own NumPy results; noconvert() on every argument stops
// pybind from copying a different dtype or layout silently. The
// notebook route copies every input into Armadillo instead, so sliced
// and Fortran-order arrays work unchanged.
//
// Cache invalidation:
// These bindings keep no array or model state. Prepared aperture
// weights belong to the caller, who must rebuild them after changing
// bins or filters.
// ---------------------------------------------------------------------------
using array_3pcf = py::array_t<double, py::array::c_style>;
using int_array_3pcf = py::array_t<int, py::array::c_style>;



// ============================================================================
// [SECTION] VALIDATION HELPERS
// ============================================================================


// Check the rank and axis lengths of a borrowed NumPy argument, naming
// the offending argument in the diagnostic.
template <typename Array>
static void rank_3pcf(
    const Array& values, // borrowed NumPy argument
    const int rank, // required number of physical axes
    const char* name // argument name for the diagnostic
  )
{
  if (values.ndim() != rank) {
    spdlog::critical("3pcf interface: {} has the wrong rank", name);
    std::exit(1);
  }

  for (int axis=0; axis<rank; axis++) {
    if (values.shape(axis) < 1
        || values.shape(axis) > INT_MAX) {
      spdlog::critical("3pcf interface: {} has an invalid axis", name);
      std::exit(1);
    }
  }
}


// C-contiguity describes strides, not pointer alignment. A NumPy
// buffer built with a one-byte offset can be contiguous, yet a typed
// read of a double through a misaligned pointer is undefined behavior
// in C. Reject such a buffer before the first typed dereference.
static void aligned_3pcf(
    const py::array& values, // borrowed array before typed data access
    const std::size_t alignment, // alignof its C element type
    const char* name // argument name for the diagnostic
  )
{
  const auto address = reinterpret_cast<std::uintptr_t>(values.data());
  if (address%alignment != 0) {
    spdlog::critical("3pcf interface: {} has an unaligned buffer", name);
    std::exit(1);
  }
}


// Abort when any element of a real array is NaN or infinite. The
// alignment check runs first, so the typed read below is defined.
static void finite_3pcf(
    const array_3pcf& values, // real array in any physical units
    const char* name // argument name for the diagnostic
  )
{
  aligned_3pcf(values, alignof(double), name);

  for (py::ssize_t index=0; index<values.size(); index++) {
    if (!std::isfinite(values.data()[index])) {
      spdlog::critical("3pcf interface: {} must be finite", name);
      std::exit(1);
    }
  }
}


// Compare the shapes of the two halves of a split-complex pair. The
// pair is consistent only when the real and imaginary arrays agree
// axis by axis.
static bool same_shape_3pcf(
    const array_3pcf& first, // first split-complex component
    const array_3pcf& second // matching split-complex component
  )
{
  if (first.ndim() != second.ndim()) {
    return false;
  }

  for (py::ssize_t axis=0; axis<first.ndim(); axis++) {
    if (first.shape(axis) != second.shape(axis)) {
      return false;
    }
  }

  return true;
}


// Abort when a value falls below its physical lower bound; strict
// excludes zero as well, for quantities such as side lengths.
static void lower_bound_3pcf(
    const array_3pcf& values, // finite array already checked by the caller
    const bool strict, // true: >0, false: >=0
    const char* name // argument name for the diagnostic
  )
{
  for (py::ssize_t index=0; index<values.size(); index++) {
    if (values.data()[index] < 0.0
        || (strict && values.data()[index] == 0.0)) {
      spdlog::critical("3pcf interface: {} violates its lower bound", name);
      std::exit(1);
    }
  }
}


// The Gaussian kernel reflects one covariance triangle across the
// diagonal, which assumes K_ij = K_ji. This check accepts only
// rounding-scale differences relative to the entries themselves; an
// absolute tolerance of order one would hide real asymmetry in small
// correlations.
//
// Why 64 machine epsilons: the caller typically builds K_ij and K_ji
// by two separate floating-point sums (filtered_pair evaluates every
// ordered pair independently), so the two orderings can disagree by a
// few dozen roundings even for an exactly symmetric model. The factor
// 64 leaves that accumulation headroom while still rejecting genuine
// asymmetry, which enters at a far larger scale.
static bool symmetric_values_3pcf(
    const double first, // K_ij at a fixed separation
    const double second // K_ji at that separation
  )
{
  const double scale = std::max(std::abs(first), std::abs(second));
  const double tolerance = 64.0*std::numeric_limits<double>::epsilon();
  return std::abs(first-second) <= tolerance*scale;
}



// ============================================================================
// [SECTION] PRODUCTION ENTRY POINTS
// ============================================================================

// ---------------------------------------------------------------------------
// Finite triangle-bin aperture weights. The C kernel returns the real
// and imaginary parts of the integration weights [filter, 4, triangle]
// that contract natural components into aperture moments. A triangle's
// flat index orders (d2 index, d3 index, phi index), with phi adjacent
// in memory. The kernel already includes both mirrored triangle
// configurations and the final factor 1/4, so callers apply no extra
// factor.
// ---------------------------------------------------------------------------
static py::tuple aperture_weights_production_3pcf(
    const array_3pcf& d2, // [nd2], positive centers in radians
    const array_3pcf& d3, // [nd3], positive centers in radians
    const array_3pcf& phi, // [nphi], opening angles in (0,pi)
    const double dlnd, // positive logarithmic side-bin width
    const double dphi, // positive opening-angle-bin width
    const array_3pcf& filters // [nfilter,3], ordered radii in radians
  )
{
  rank_3pcf(d2, 1, "d2");
  rank_3pcf(d3, 1, "d3");
  rank_3pcf(phi, 1, "phi");
  rank_3pcf(filters, 2, "filters");

  if (!(filters.shape(1) == 3)) {
    spdlog::critical("3pcf interface: filters need three columns");
    std::exit(1);
  }

  finite_3pcf(d2, "d2");
  finite_3pcf(d3, "d3");
  finite_3pcf(phi, "phi");
  finite_3pcf(filters, "filters");

  lower_bound_3pcf(d2, true, "d2");
  lower_bound_3pcf(d3, true, "d3");
  lower_bound_3pcf(filters, true, "filters");

  if (!(std::isfinite(dlnd)
        && dlnd > 0.0
        && std::isfinite(dphi)
        && dphi > 0.0)) {
    spdlog::critical("3pcf interface: bin widths must be finite and positive");
    std::exit(1);
  }

  for (py::ssize_t index=0; index<phi.size(); index++) {
    if (!(phi.data()[index] > 0.0
          && phi.data()[index] < std::acos(-1.0))) {
      spdlog::critical("3pcf interface: phi must lie strictly between zero "
        "and pi");
      std::exit(1);
    }
  }

  // C receives an int triangle count. Check before multiplying, so a
  // large request cannot wrap into a smaller allocation or loop bound.
  const int nd2 = static_cast<int>(d2.size());
  const int nd3 = static_cast<int>(d3.size());
  const int nphi = static_cast<int>(phi.size());
  const int nfilter = static_cast<int>(filters.shape(0));

  if (!(nd2 <= INT_MAX/nd3)) {
    spdlog::critical("3pcf interface: triangle count exceeds C int range");
    std::exit(1);
  }
  if (!(nd2*nd3 <= INT_MAX/nphi)) {
    spdlog::critical("3pcf interface: triangle count exceeds C int range");
    std::exit(1);
  }

  const int ntriangle = nd2*nd3*nphi;
  array_3pcf weights_real({nfilter, 4, ntriangle});
  array_3pcf weights_imag({nfilter, 4, ntriangle});

  aperture_weights_3pcf(nd2, nd3, nphi, nfilter, d2.data(), d3.data(),
      phi.data(), dlnd, dphi, filters.data(), weights_real.mutable_data(),
      weights_imag.mutable_data());

  finite_3pcf(weights_real, "result wr");
  finite_3pcf(weights_imag, "result wi");

  return py::make_tuple(weights_real, weights_imag);
}


// ---------------------------------------------------------------------------
// Contract prepared aperture weights with natural components for
// several source triplets or model samples at once. The C routine
// takes the real part of the product weight*gamma without conjugating
// either factor, so the caller passes both imaginary halves as well:
// the real part of a complex product mixes them with the real parts.
// The output [sample, filter] holds one real aperture moment per
// sample and radius triplet.
// ---------------------------------------------------------------------------
static array_3pcf aperture_apply_production_3pcf(
    const array_3pcf& weights_real, // [filter,4,triangle], real weights
    const array_3pcf& weights_imag, // matching imaginary weights
    const array_3pcf& gamma_real, // [sample,4,triangle], real components
    const array_3pcf& gamma_imag // matching imaginary components
  )
{
  rank_3pcf(weights_real, 3, "wr");
  rank_3pcf(weights_imag, 3, "wi");
  rank_3pcf(gamma_real, 3, "gr");
  rank_3pcf(gamma_imag, 3, "gi");

  if (!(same_shape_3pcf(weights_real, weights_imag)
        && same_shape_3pcf(gamma_real, gamma_imag)
        && weights_real.shape(1) == 4
        && gamma_real.shape(1) == 4
        && weights_real.shape(2) == gamma_real.shape(2))) {
    spdlog::critical("3pcf interface: aperture component and triangle axes "
      "disagree");
    std::exit(1);
  }

  finite_3pcf(weights_real, "wr");
  finite_3pcf(weights_imag, "wi");
  finite_3pcf(gamma_real, "gr");
  finite_3pcf(gamma_imag, "gi");

  const int nfilter = static_cast<int>(weights_real.shape(0));
  const int ntriangle = static_cast<int>(weights_real.shape(2));
  const int nsample = static_cast<int>(gamma_real.shape(0));

  array_3pcf output({nsample, nfilter});
  aperture_apply_3pcf(nfilter, ntriangle, nsample, weights_real.data(),
      weights_imag.data(), gamma_real.data(), gamma_imag.data(),
      output.mutable_data());

  finite_3pcf(output, "aperture result");

  return output;
}


// ---------------------------------------------------------------------------
// Filtered pair correlation of two fields. The C kernel evaluates
//
//   K_ij(r) = integral dell ell/(2 pi) U_i(ell) U_j(ell) P_ij(ell)
//             J0(ell r),
//
// where ell is the angular multipole, U_i is the Fourier filter of
// field i, P_ij is the angular power spectrum of fields i and j, r is
// the separation in radians and J0 is the cylindrical Bessel function
// of the first kind and order zero. The caller supplies the d ell
// quadrature weights; the kernel inserts the factor ell J0(ell r) and
// the 1/(2 pi).
// ---------------------------------------------------------------------------
static array_3pcf filtered_pair_production_3pcf(
    const array_3pcf& ell, // [nell], nonnegative multipoles
    const array_3pcf& ell_weights, // [nell], d ell quadrature weights
    const array_3pcf& radius, // [nr], nonnegative radians
    const array_3pcf& filters, // [field,nell], Fourier filter values
    const array_3pcf& power // [field,field,nell], angular power
  )
{
  rank_3pcf(ell, 1, "ell");
  rank_3pcf(ell_weights, 1, "ell_weights");
  rank_3pcf(radius, 1, "r");
  rank_3pcf(filters, 2, "filters");
  rank_3pcf(power, 3, "power");

  if (!(ell_weights.size() == ell.size()
        && filters.shape(1) == ell.size()
        && power.shape(0) == filters.shape(0)
        && power.shape(1) == filters.shape(0)
        && power.shape(2) == ell.size())) {
    spdlog::critical("3pcf interface: filtered-pair axes disagree");
    std::exit(1);
  }

  finite_3pcf(ell, "ell");
  finite_3pcf(ell_weights, "ell_weights");
  finite_3pcf(radius, "r");
  finite_3pcf(filters, "filters");
  finite_3pcf(power, "power");

  lower_bound_3pcf(ell, false, "ell");
  lower_bound_3pcf(radius, false, "r");

  const int nfield = static_cast<int>(filters.shape(0));
  const int nell = static_cast<int>(ell.size());
  const int nr = static_cast<int>(radius.size());

  array_3pcf output({nfield, nfield, nr});
  filtered_pair_3pcf(nfield, nell, nr, ell.data(), ell_weights.data(),
      radius.data(), filters.data(), power.data(), output.mutable_data());

  finite_3pcf(output, "filtered-pair result");

  return output;
}


// ---------------------------------------------------------------------------
// Weighted contraction of connected responses that the caller evaluates
// on common quadrature nodes:
//
//   out[row, column] = sum_node weights[node] responses[row, node]
//                      responses[column, node].
//
// The caller places the entire physical measure, any survey-area factor
// included, in the weights; the kernel adds nothing of its own.
// ---------------------------------------------------------------------------
static array_3pcf connected_gram_production_3pcf(
    const array_3pcf& responses, // [row,node], caller's halo responses
    const array_3pcf& weights // [node], nonnegative integration measure
  )
{
  rank_3pcf(responses, 2, "responses");
  rank_3pcf(weights, 1, "weights");

  if (!(responses.shape(1) == weights.size())) {
    spdlog::critical("3pcf interface: response-node and weight counts must "
      "agree");
    std::exit(1);
  }

  finite_3pcf(responses, "responses");
  finite_3pcf(weights, "weights");

  lower_bound_3pcf(weights, false, "weights");

  const int nrow = static_cast<int>(responses.shape(0));
  const int nnode = static_cast<int>(weights.size());

  array_3pcf output({nrow, nrow});
  connected_gram_3pcf(nrow, nnode, responses.data(), weights.data(),
      output.mutable_data());

  finite_3pcf(output, "Gram result");

  return output;
}


// ---------------------------------------------------------------------------
// Gaussian covariance of raw cubic aperture fields. For Gaussian
// fields, Wick's theorem splits the expectation of a product of six
// fields into products of pair expectations; these splittings are the
// Wick pairings, and the C kernel keeps all fifteen of them, the pairs
// joining two fields at the same aperture center included. A catalogue
// estimator over distinct galaxies drops its self-pair noise terms,
// which this kernel does not subtract.
// ---------------------------------------------------------------------------
static array_3pcf gaussian_covariance_production_3pcf(
    const int_array_3pcf& triplets, // [row,3], zero-based field labels
    const array_3pcf& pair, // [field,field,r], symmetric correlations
    const array_3pcf& pair_zero, // [field,field], zero-separation values
    const array_3pcf& radial_weights // [nr], nonnegative window measure
  )
{
  rank_3pcf(triplets, 2, "triplets");
  rank_3pcf(pair, 3, "pair");
  rank_3pcf(pair_zero, 2, "pair_zero");
  rank_3pcf(radial_weights, 1, "radial_weights");
  aligned_3pcf(triplets, alignof(int), "triplets");

  if (!(triplets.shape(1) == 3
        && pair.shape(0) == pair.shape(1)
        && pair_zero.shape(0) == pair.shape(0)
        && pair_zero.shape(1) == pair.shape(0)
        && radial_weights.size() == pair.shape(2))) {
    spdlog::critical("3pcf interface: Gaussian covariance axes disagree");
    std::exit(1);
  }

  finite_3pcf(pair, "pair");
  finite_3pcf(pair_zero, "pair_zero");
  finite_3pcf(radial_weights, "radial_weights");

  lower_bound_3pcf(radial_weights, false, "radial_weights");

  const int nfield = static_cast<int>(pair.shape(0));
  const int nr = static_cast<int>(pair.shape(2));
  const int nrow = static_cast<int>(triplets.shape(0));

  // The C loop fills one covariance triangle and reflects it. That is
  // the same Gaussian covariance only for symmetric field correlations.
  for (int first=0; first<nfield; first++) {
    for (int second=0; second<first; second++) {
      if (!(symmetric_values_3pcf(*pair_zero.data(first, second),
            *pair_zero.data(second, first)))) {
        spdlog::critical("3pcf interface: pair_zero must be symmetric");
        std::exit(1);
      }
      for (int radial=0; radial<nr; radial++) {
        if (!(symmetric_values_3pcf(*pair.data(first, second, radial),
              *pair.data(second, first, radial)))) {
          spdlog::critical("3pcf interface: pair must be symmetric");
          std::exit(1);
        }
      }
    }
  }

  array_3pcf output({nrow, nrow});
  gaussian_aperture_covariance_3pcf(nfield, nr, nrow, triplets.data(),
      pair.data(), pair_zero.data(), radial_weights.data(),
      output.mutable_data());

  finite_3pcf(output, "Gaussian result");

  return output;
}



// ============================================================================
// [SECTION] NOTEBOOK INPUT CONVERSION
// ============================================================================

// Convert a notebook input into an owning Armadillo array. This
// converter insists on a NumPy array and rejects everything else,
// because a long Python list that enters through the binding-level
// caster can arrive zero-filled; a caller converts a list safely with
// numpy.asarray first. A private Fortran-order copy matches
// Armadillo's first-index-fastest storage, and a second explicit copy
// gives Armadillo ownership, including for small cubes whose elements
// would otherwise live inside the C++ temporary. No input caster of
// CARMA, the NumPy-Armadillo bridge library, runs on this route.
template <typename Array>
static Array notebook_input_3pcf(
    const py::object& input, // NumPy array; may be strided or read-only
    const int rank // physical rank: vector, matrix or cube
  )
{
  if (!(py::isinstance<py::array>(input))) {
    spdlog::critical("3pcf interface: notebook inputs must be NumPy arrays");
    std::exit(1);
  }

  const py::array values = py::reinterpret_borrow<py::array>(input);
  rank_3pcf(values, rank, "notebook input");

  using Element = typename Array::elem_type;
  if constexpr (std::is_integral<Element>::value) {
    if (!(values.dtype().kind() == 'i'
          || values.dtype().kind() == 'u')) {
      spdlog::critical("3pcf interface: triplets must have integer dtype");
      std::exit(1);
    }
    const double minimum = py::cast<double>(input.attr("min")());
    const double maximum = py::cast<double>(input.attr("max")());
    if (!(minimum >= INT_MIN
          && maximum <= INT_MAX)) {
      spdlog::critical("3pcf interface: triplets exceed int32 range");
      std::exit(1);
    }
  } else {
    if (!(values.dtype().kind() == 'f'
          || values.dtype().kind() == 'i'
          || values.dtype().kind() == 'u')) {
      spdlog::critical("3pcf interface: physical inputs must have a real "
        "numeric dtype");
      std::exit(1);
    }
  }

  const py::object owned = input.attr("astype")(
      py::dtype::of<Element>(), py::arg("order") = "F",
      py::arg("copy") = true);
  const py::buffer_info buffer =
      py::reinterpret_borrow<py::buffer>(owned).request();

  Array output;
  if constexpr (arma::is_Cube<Array>::value) {
    output.set_size(buffer.shape[0], buffer.shape[1], buffer.shape[2]);
  } else if constexpr (arma::is_Col<Array>::value) {
    output.set_size(buffer.shape[0]);
  } else {
    output.set_size(buffer.shape[0], buffer.shape[1]);
  }

  std::copy_n(static_cast<const Element*>(buffer.ptr), buffer.size,
      output.memptr());

  return output;
}



// ============================================================================
// [SECTION] THREAD CONTROL
// ============================================================================

// Fix the OpenMP team size for later C calls. openblas_set_num_threads
// pins BLAS to one thread first: the C kernels parallelize their own
// loops with OpenMP, and a BLAS call that spawned its own team inside
// an OpenMP region would oversubscribe the cores - more threads than
// cores, all competing for them.
static void set_threads_3pcf(
    const int nthreads // positive requested OpenMP team size
  )
{
  if (!(nthreads > 0)) {
    spdlog::critical("3pcf interface: nthreads must be positive");
    std::exit(1);
  }

  openblas_set_num_threads(1);

#ifdef _OPENMP
  omp_set_dynamic(0);
  omp_set_num_threads(nthreads);
#else
  if (!(nthreads == 1)) {
    spdlog::critical("3pcf interface: this module was built without OpenMP");
    std::exit(1);
  }
#endif
}



// ============================================================================
// [SECTION] MODULE REGISTRATION
// ============================================================================

// Register both submodules. Python names repeat across the two routes;
// only ownership and storage conversion differ between them. Results
// keep the documented physical axes, and the notebook route exports
// matrices and cubes in Fortran storage order.
void bind_threepcf(
    py::module_& module // project extension receiving both interfaces
  )
{
  // Triplet labels cross the Python/C boundary as 32-bit C int.
  static_assert(sizeof(int) == 4, "triplets require 32-bit C int");

  // Pin BLAS to one thread at import, before any OpenMP region runs
  // (set_threads_3pcf explains why).
  openblas_set_num_threads(1);

  module.def("set_omp_threads", &set_threads_3pcf,
      "Set the positive OpenMP worker count; BLAS remains serial.",
      py::arg("nthreads"));

  module.def("get_omp_threads", []() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
  }, "Return the maximum OpenMP team size for subsequent C calls.");

  py::module_ production = module.def_submodule("production",
      "C-contiguous float64/int32 NumPy interface; the bindings borrow "
      "the caller's arrays without copying them.");

  production.def("aperture_weights", &aperture_weights_production_3pcf,
      "Finite triangle-bin aperture weights. The pair (wr, wi) holds "
      "the real and imaginary parts of the integration weights "
      "[filter, 4, triangle], including both mirrored triangle "
      "configurations and the factor 1/4. d2, d3, phi and the "
      "[filter, 3] aperture radii are in radians.",
      py::arg("d2").noconvert(), py::arg("d3").noconvert(),
      py::arg("phi").noconvert(), py::arg("dlnd"), py::arg("dphi"),
      py::arg("filters").noconvert());

  production.def("aperture_apply", &aperture_apply_production_3pcf,
      "Contract prepared aperture weights with natural components. wr "
      "and wi hold the real and imaginary parts of the integration "
      "weights [filter, 4, triangle]; gr and gi hold the real and "
      "imaginary parts of the natural-component grid "
      "[sample, 4, triangle]. The result [sample, filter] is the real "
      "part of the product, with neither factor conjugated.",
      py::arg("wr").noconvert(), py::arg("wi").noconvert(),
      py::arg("gr").noconvert(), py::arg("gi").noconvert());

  production.def("filtered_pair", &filtered_pair_production_3pcf,
      "Filtered pair correlations K[field, field, r] from the "
      "caller's angular spectra: K_ij(r) = integral dell ell/(2 pi) "
      "U_i U_j P_ij J0(ell r), with J0 the cylindrical Bessel function "
      "of the first kind and order zero. ell_weights holds the d ell "
      "quadrature; C inserts ell J0(ell r)/(2 pi).",
      py::arg("ell").noconvert(), py::arg("ell_weights").noconvert(),
      py::arg("r").noconvert(), py::arg("filters").noconvert(),
      py::arg("power").noconvert());

  production.def("connected_gram", &connected_gram_production_3pcf,
      "Connected covariance from responses on common nodes: "
      "out[row, column] = sum_node weights * responses[row] * "
      "responses[column]. The nonnegative weights carry the entire "
      "physical measure; the kernel adds no factor of its own.",
      py::arg("responses").noconvert(), py::arg("weights").noconvert());

  production.def("gaussian_aperture_covariance",
      &gaussian_covariance_production_3pcf,
      "Gaussian covariance of raw cubic aperture fields, keeping all "
      "fifteen Wick pairings: the splittings of a Gaussian six-field "
      "expectation into products of pairs. radial_weights carries the "
      "radial area and window measure. pair and pair_zero must be "
      "symmetric to 64 machine epsilons. The kernel subtracts no "
      "distinct-galaxy catalogue exclusions.",
      py::arg("triplets").noconvert(), py::arg("pair").noconvert(),
      py::arg("pair_zero").noconvert(),
      py::arg("radial_weights").noconvert());

  py::module_ notebook = module.def_submodule("notebook",
      "Armadillo-owned copies of every input, over the same C kernels.");

  notebook.def("aperture_weights", [](
      const py::object& d2, // [nd2], radians
      const py::object& d3, // [nd3], radians
      const py::object& phi, // [nphi], radians
      const double dlnd, // logarithmic side-bin width
      const double dphi, // opening-angle-bin width
      const py::object& filters // [filter,3], radii in radians
    ) {
    const auto d2_input = notebook_input_3pcf<arma::Col<double>>(d2, 1);
    const auto d3_input = notebook_input_3pcf<arma::Col<double>>(d3, 1);
    const auto phi_input = notebook_input_3pcf<arma::Col<double>>(phi, 1);
    const auto filters_input =
        notebook_input_3pcf<arma::Mat<double>>(filters, 2);

    auto result = aperture_weights_cpp_3pcf(d2_input, d3_input, phi_input,
        dlnd, dphi, filters_input);
    return py::make_tuple(carma::cube_to_arr(std::move(result.real)),
        carma::cube_to_arr(std::move(result.imag)));
  }, "Notebook twin of production.aperture_weights: copy the inputs "
      "through Armadillo, then return the same weights on the "
      "production axes.",
      py::arg("d2"), py::arg("d3"), py::arg("phi"), py::arg("dlnd"),
      py::arg("dphi"), py::arg("filters"));

  notebook.def("aperture_apply", [](
      const py::object& wr, // [filter,4,triangle], real weights
      const py::object& wi, // matching imaginary weights
      const py::object& gr, // [sample,4,triangle], real components
      const py::object& gi // matching imaginary components
    ) {
    const auto weights_real = notebook_input_3pcf<arma::Cube<double>>(wr, 3);
    const auto weights_imag = notebook_input_3pcf<arma::Cube<double>>(wi, 3);
    const auto gamma_real = notebook_input_3pcf<arma::Cube<double>>(gr, 3);
    const auto gamma_imag = notebook_input_3pcf<arma::Cube<double>>(gi, 3);

    return carma::mat_to_arr(aperture_apply_cpp_3pcf(weights_real,
        weights_imag, gamma_real, gamma_imag));
  }, "Notebook twin of production.aperture_apply: copy the "
      "split-complex pairs - wr/wi, the real and imaginary parts of "
      "the integration weights, and gr/gi, the real and imaginary "
      "parts of the natural-component grid - through Armadillo, then "
      "return the real moments [sample, filter].",
      py::arg("wr"), py::arg("wi"), py::arg("gr"), py::arg("gi"));

  notebook.def("filtered_pair", [](
      const py::object& ell, // [nell], angular multipoles
      const py::object& ell_weights, // [nell], d ell quadrature
      const py::object& radius, // [nr], radians
      const py::object& filters, // [field,nell], Fourier filters
      const py::object& power // [field,field,nell], angular power
    ) {
    const auto ell_input = notebook_input_3pcf<arma::Col<double>>(ell, 1);
    const auto weight_input =
        notebook_input_3pcf<arma::Col<double>>(ell_weights, 1);
    const auto radius_input =
        notebook_input_3pcf<arma::Col<double>>(radius, 1);
    const auto filters_input =
        notebook_input_3pcf<arma::Mat<double>>(filters, 2);
    const auto power_input =
        notebook_input_3pcf<arma::Cube<double>>(power, 3);

    return carma::cube_to_arr(filtered_pair_cpp_3pcf(ell_input,
        weight_input, radius_input, filters_input, power_input));
  }, "Notebook twin of production.filtered_pair: copy the inputs "
      "through Armadillo, then return the filtered pair correlations "
      "[field, field, r].",
      py::arg("ell"), py::arg("ell_weights"), py::arg("r"),
      py::arg("filters"), py::arg("power"));

  notebook.def("connected_gram", [](
      const py::object& responses, // [row,node], caller's responses
      const py::object& weights // [node], nonnegative measure
    ) {
    const auto response_input =
        notebook_input_3pcf<arma::Mat<double>>(responses, 2);
    const auto weight_input =
        notebook_input_3pcf<arma::Col<double>>(weights, 1);

    return carma::mat_to_arr(
        connected_gram_cpp_3pcf(response_input, weight_input));
  }, "Notebook twin of production.connected_gram: copy the responses "
      "and weights through Armadillo, then return the covariance "
      "[row, row].",
      py::arg("responses"), py::arg("weights"));

  notebook.def("gaussian_aperture_covariance", [](
      const py::object& triplets, // [row,3], integer field labels
      const py::object& pair, // [field,field,r], correlations
      const py::object& pair_zero, // [field,field], r=0 values
      const py::object& radial_weights // [nr], radial window measure
    ) {
    const auto triplet_input =
        notebook_input_3pcf<arma::Mat<int>>(triplets, 2);
    const auto pair_input =
        notebook_input_3pcf<arma::Cube<double>>(pair, 3);
    const auto zero_input =
        notebook_input_3pcf<arma::Mat<double>>(pair_zero, 2);
    const auto weight_input =
        notebook_input_3pcf<arma::Col<double>>(radial_weights, 1);

    return carma::mat_to_arr(gaussian_aperture_covariance_cpp_3pcf(
        triplet_input, pair_input, zero_input, weight_input));
  }, "Notebook twin of production.gaussian_aperture_covariance: copy "
      "the inputs through Armadillo and keep all fifteen Wick "
      "pairings. Pair correlations must be symmetric to 64 machine "
      "epsilons.",
      py::arg("triplets"), py::arg("pair"), py::arg("pair_zero"),
      py::arg("radial_weights"));
}

}
