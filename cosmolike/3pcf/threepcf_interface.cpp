#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>
#include <utility>
#include <carma.h>
#include <spdlog/spdlog.h>
#include "threepcf_interface.hpp"
#include "threepcf_wrapper.hpp"
#include "physics_wrapper.hpp"
#include "aperture_3pcf.h"
#include "covariance_3pcf.h"
#include "threepcf_core_bridge.h"
#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" void openblas_set_num_threads(
  int nthreads // BLAS worker count; this module always requests one
);

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
    if (1 > values.shape(axis)
        || INT_MAX < values.shape(axis)) {
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
  if (0 != address%alignment) {
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
    if (0.0 > values.data()[index]
        || (strict && 0.0 == values.data()[index])) {
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

  if (!(3 == filters.shape(1))) {
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
        && 0.0 < dlnd
        && std::isfinite(dphi)
        && 0.0 < dphi)) {
    spdlog::critical("3pcf interface: bin widths must be finite and positive");
    std::exit(1);
  }

  for (py::ssize_t index=0; index<phi.size(); index++) {
    if (!(0.0 < phi.data()[index]
          && M_PI > phi.data()[index])) {
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
        && 4 == weights_real.shape(1)
        && 4 == gamma_real.shape(1)
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
#ifndef COSMOLIKE_NO_COVARIANCE
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

  if (!(3 == triplets.shape(1)
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
#endif



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
    if (!('i' == values.dtype().kind()
          || 'u' == values.dtype().kind())) {
      spdlog::critical("3pcf interface: triplets must have integer dtype");
      std::exit(1);
    }
    const double minimum = py::cast<double>(input.attr("min")());
    const double maximum = py::cast<double>(input.attr("max")());
    if (!(INT_MIN <= minimum
          && INT_MAX >= maximum)) {
      spdlog::critical("3pcf interface: triplets exceed int32 range");
      std::exit(1);
    }
  } else {
    if (!('f' == values.dtype().kind()
          || 'i' == values.dtype().kind()
          || 'u' == values.dtype().kind())) {
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
  if (!(0 < nthreads)) {
    spdlog::critical("3pcf interface: nthreads must be positive");
    std::exit(1);
  }

  openblas_set_num_threads(1);

#ifdef _OPENMP
  omp_set_dynamic(0);
  omp_set_num_threads(nthreads);
#else
  if (!(1 == nthreads)) {
    spdlog::critical("3pcf interface: this module was built without OpenMP");
    std::exit(1);
  }
#endif
}



// ============================================================================
// [SECTION] CORE-STRUCT BRIDGE ADAPTERS
// ============================================================================
// Python adapters of threepcf_core_bridge.h: the supplied-tier
// setters, the standalone-module equivalents of the core struct
// installers, the IA translation and the cached model factory. The
// adapters validate NumPy arrays with the helpers above and forward
// plain C pointers; every conversion and refusal lives in the C
// bridge.

// Borrow a one-dimensional contiguous float64 argument and abort on a
// wrong rank, a misaligned buffer or a nonfinite entry.
static const double* bridge_vector_3pcf(
    const array_3pcf& values, // borrowed NumPy argument
    const char* name // argument name for the diagnostic
  )
{
  rank_3pcf(values, 1, name);
  finite_3pcf(values, name);
  return values.data();
}

// ---------------------------------------------------------------------------
// The cached model owner. The C bridge owns the two-tier guard, the
// key snapshot and the assembled input; this adapter owns the Python-
// facing PhysicsModel. One shared owner is cached at file scope and
// handed out unchanged while the guard reports no change, so repeated
// factory calls during a chain never rebuild and downstream grids
// keep borrowing one model. When the guard reports a change the
// existing physics_owner_3pcf constructor rebuilds through
// physics_create_3pcf - the same factory path the direct Python
// constructor uses - and the previous owner lives on until its last
// Python reference drops, which keeps borrowed projections valid.
//
// Cache invalidation:
// threepcf_bridge_refresh decides; reset_threepcf_core_bridge clears
// its snapshot, so the first factory call after a reset rebuilds.
// Keep the Python GIL through refresh and copying: setters use that same
// lock, so no Python thread can replace the shared power or scratch arrays
// halfway through preparation. C OpenMP workers do not need the GIL.
// ---------------------------------------------------------------------------
static std::shared_ptr<physics_owner_3pcf> model_from_core_3pcf()
{
  static std::shared_ptr<physics_owner_3pcf> cached;
  const physics_input_3pcf* input = threepcf_bridge_refresh();
  if (NULL == input && !cached) {
    // the guard snapshot exists but this adapter never built: some
    // other caller consumed a refresh without the owner this factory
    // hands out; refuse rather than return an empty owner
    spdlog::critical("3pcf interface: the bridge guard is committed "
      "but no cached model exists; model_from_core must be the only "
      "threepcf_bridge_refresh consumer in this process");
    std::exit(1);
  }
  if (NULL != input) {
    cached = std::make_shared<physics_owner_3pcf>(*input);
  }
  return cached;
}

// Translate the stored nuisance IA state into the dictionary the
// model queries accept; every physics_ia_3pcf field appears, so the
// dictionary determines the parameter struct completely.
static py::dict ia_parameters_from_core_3pcf()
{
  physics_ia_3pcf parameters;
  threepcf_bridge_ia_from_core(&parameters);
  py::dict result;
  result["a1"] = parameters.a1;
  result["eta1"] = parameters.eta1;
  result["a2"] = parameters.a2;
  result["eta2"] = parameters.eta2;
  result["bta"] = parameters.bta;
  result["ct"] = parameters.ct;
  result["zpivot"] = parameters.zpivot;
  result["c1rho"] = parameters.c1rho;
  result["fb"] = parameters.fb;
  result["power_mode"] = parameters.power_mode;
  result["ct_mode"] = parameters.ct_mode;
  result["suppress_only"] = parameters.suppress_only;
  result["operator_component"] = parameters.operator_component;
  result["intrinsic_legs"] = parameters.intrinsic_legs;
  result["feedback_amplitude"] = parameters.feedback_amplitude;
  return result;
}

// Supplied sources: both None clears the override, both arrays set
// it; mixing is refused because a half-cleared override has no
// meaning.
static void bridge_sources_3pcf(
    const py::object& source_z, // [nsource] redshifts, or None
    const py::object& sources // [nbin,nsource] densities, or None
  )
{
  if (source_z.is_none() && sources.is_none()) {
    threepcf_bridge_set_sources(NULL, NULL, 0, 0);
    return;
  }
  if (source_z.is_none() || sources.is_none()) {
    spdlog::critical("3pcf interface: supplied sources need both "
      "source_z and sources, or None twice to clear");
    std::exit(1);
  }
  if (!(array_3pcf::check_(source_z) && array_3pcf::check_(sources))) {
    spdlog::critical("3pcf interface: supplied sources must be "
      "contiguous float64 arrays");
    std::exit(1);
  }
  const auto axis = py::reinterpret_borrow<array_3pcf>(source_z);
  const auto density = py::reinterpret_borrow<array_3pcf>(sources);
  rank_3pcf(axis, 1, "source_z");
  rank_3pcf(density, 2, "sources");
  finite_3pcf(axis, "source_z");
  finite_3pcf(density, "sources");
  if (!(density.shape(1) == axis.size())) {
    spdlog::critical("3pcf interface: sources must hold one column "
      "per source_z node");
    std::exit(1);
  }
  threepcf_bridge_set_sources(axis.data(), density.data(),
      static_cast<int>(density.shape(0)),
      static_cast<int>(axis.size()));
}

// Paired feedback response: the dictionary uses the same eight keys
// as the PhysicsModel factory (feedback_3pcf.h documents each), with
// the shared log-response amplitude alongside. An empty dictionary
// with amplitude zero clears the stored response.
static void bridge_feedback_3pcf(
    const py::dict& feedback, // six arrays plus boost and clamp
    const double amplitude // shared log-response amplitude
  )
{
  if (feedback.empty()) {
    if (0.0 != amplitude) {
      spdlog::critical("3pcf interface: a feedback amplitude without "
        "response tables has no meaning");
      std::exit(1);
    }
    threepcf_bridge_set_feedback(NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        NULL, NULL, 0, 0, 0.0);
    return;
  }
  const char* names[6] = {"z", "logk", "logu", "v", "log_power",
    "log_bispectrum"};
  if (!(8 == feedback.size()
        && feedback.contains("boost") && feedback.contains("clamp"))) {
    spdlog::critical("3pcf interface: feedback requires six arrays "
      "plus boost and clamp");
    std::exit(1);
  }
  std::vector<array_3pcf> arrays;
  for (int index=0; 6 > index; index++) {
    if (!(feedback.contains(names[index])
          && array_3pcf::check_(feedback[names[index]]))) {
      spdlog::critical("3pcf interface: feedback arrays must be "
        "contiguous float64 vectors");
      std::exit(1);
    }
    arrays.push_back(py::reinterpret_borrow<array_3pcf>(
      feedback[names[index]]));
    rank_3pcf(arrays.back(), 1, names[index]);
    finite_3pcf(arrays.back(), names[index]);
  }
  const auto nz = static_cast<size_t>(arrays[0].size());
  const auto nk = static_cast<size_t>(arrays[1].size());
  const auto nu = static_cast<size_t>(arrays[2].size());
  const auto nv = static_cast<size_t>(arrays[3].size());
  // one chained check, as in the model factory: every axis fits an
  // int, the products cannot wrap, and the flattened tables match
  if (!(INT_MAX >= nz && INT_MAX >= nk && INT_MAX >= nu && INT_MAX >= nv
        && nz <= SIZE_MAX/nk && nz*nk <= SIZE_MAX/nu
        && nz*nk*nu <= SIZE_MAX/nv
        && static_cast<size_t>(arrays[4].size()) == nz*nk
        && static_cast<size_t>(arrays[5].size()) == nz*nk*nu*nv)) {
    spdlog::critical("3pcf interface: flattened feedback arrays "
      "disagree with their axes");
    std::exit(1);
  }
  threepcf_bridge_set_feedback(
      arrays[0].data(), static_cast<int>(nz),
      arrays[1].data(), static_cast<int>(nk),
      arrays[2].data(), static_cast<int>(nu),
      arrays[3].data(), static_cast<int>(nv),
      arrays[4].data(), arrays[5].data(),
      py::cast<int>(feedback["boost"]),
      py::cast<int>(feedback["clamp"]), amplitude);
}

// The core bridge owns copies just as the direct model factories do.
// Clearing this override never changes the stored legacy fit scalars.
static void bridge_baseline_3pcf(
  const py::object& baseline, // five arrays plus boost and clamp; None clears
  const py::object& omega_m_z, // matter fraction on core background z
  const py::object& sigma8 // required positive normalization when installing
)
{
  if (baseline.is_none()) {
    threepcf_bridge_set_baseline_bispectrum(NULL, NULL, 0, 0.0);
    return;
  }
  if (!(py::isinstance<py::dict>(baseline))) { throw py::value_error(
    "baseline must be a dictionary or None"); }
  if (omega_m_z.is_none()
      || sigma8.is_none()) { throw py::value_error(
    "baseline requires omega_m_z and sigma8"); }
  const auto dictionary = py::cast<py::dict>(baseline);
  const char* names[5] = {"z", "logk", "logu", "v", "log_bispectrum"};
  if (!(7 == dictionary.size())) { throw py::value_error(
    "baseline requires five arrays, boost and clamp"); }
  if (!(dictionary.contains("boost"))) { throw py::value_error(
    "baseline boost is missing"); }
  if (!(dictionary.contains("clamp"))) { throw py::value_error(
    "baseline clamp is missing"); }
  std::vector<array_3pcf> arrays;
  size_t size = 1;
  for (int axis=0; 5 > axis; axis++) {
    if (!(dictionary.contains(names[axis]))) { throw py::value_error(
      "baseline array is missing"); }
    if (!(array_3pcf::check_(dictionary[names[axis]]))) { throw py::value_error(
      "baseline arrays must be contiguous float64 vectors"); }
    arrays.push_back(py::reinterpret_borrow<array_3pcf>(
      dictionary[names[axis]]));
    rank_3pcf(arrays.back(), 1, names[axis]);
    finite_3pcf(arrays.back(), names[axis]);
    if (4 > axis) {
      const size_t count = arrays.back().size();
      if (!(4 <= count)) { throw py::value_error(
        "baseline axes need four nodes"); }
      if (!(count <= INT_MAX/size)) { throw py::value_error(
        "baseline dimensions exceed C storage"); }
      size *= count;
    }
  }
  if (!(size == static_cast<size_t>(arrays[4].size()))) { throw py::value_error(
    "baseline values disagree with axis sizes"); }
  if (!(array_3pcf::check_(omega_m_z))) { throw py::value_error(
    "omega_m_z must be a contiguous float64 vector"); }
  const auto matter_fraction = py::reinterpret_borrow<array_3pcf>(omega_m_z);
  const double* fractions = bridge_vector_3pcf(matter_fraction, "omega_m_z");
  baseline_input_3pcf input = {};
  input.nz = static_cast<int>(arrays[0].size());
  input.nk = static_cast<int>(arrays[1].size());
  input.nu = static_cast<int>(arrays[2].size());
  input.nv = static_cast<int>(arrays[3].size());
  input.z = arrays[0].data();
  input.logk = arrays[1].data();
  input.logu = arrays[2].data();
  input.v = arrays[3].data();
  input.log_bispectrum = arrays[4].data();
  input.boost = py::cast<int>(dictionary["boost"]);
  input.clamp = py::cast<int>(dictionary["clamp"]);
  threepcf_bridge_set_baseline_bispectrum(&input, fractions,
    static_cast<int>(matter_fraction.size()), py::cast<double>(sigma8));
}

// Prepare owned variance arrays from the current core state. Keep the GIL
// during this serial entry: another Python thread must not replace the
// global power tables while the lazy shared builders read them. The
// builders themselves may use their usual OpenMP workers.
static py::dict bridge_variance_snapshot_3pcf(
  const array_3pcf& z, // requested increasing redshifts [nz]
  const py::object& mass_values, // mass nodes [nmass], or None
  const double reference_rho, // reference matter density; zero selects core
  const bool include_fit // include Gaussian radius, slope and sigma8
)
{
  bridge_vector_3pcf(z, "variance redshifts");
  array_3pcf mass(0);
  if (!mass_values.is_none()) {
    if (!array_3pcf::check_(mass_values)) {
      spdlog::critical("3pcf variance snapshot: masses must be contiguous "
        "float64 values");
      std::exit(1);
    }
    mass = py::reinterpret_borrow<array_3pcf>(mass_values);
    if (1 != mass.ndim()
        || INT_MAX < mass.size()) {
      spdlog::critical("3pcf variance snapshot: masses must be one vector "
        "with at most INT_MAX entries");
      std::exit(1);
    }
    finite_3pcf(mass, "variance masses");
  }
  if (0 < mass.size()
      && INT_MAX/mass.size() < z.size()) {
    spdlog::critical("3pcf variance snapshot: redshift times mass count "
      "exceeds INT_MAX");
    std::exit(1);
  }

  const py::ssize_t fit_count = include_fit ? z.size() : 0;
  array_3pcf rnl(fit_count);
  array_3pcf neff(fit_count);
  array_3pcf sigma8(fit_count);
  array_3pcf halo_sigma2({z.size(), mass.size()});
  array_3pcf halo_slope({z.size(), mass.size()});
  const double core_rho = threepcf_core_variance_snapshot(
    static_cast<int>(z.size()), z.data(), static_cast<int>(mass.size()),
    mass.data(), reference_rho, include_fit ? 1 : 0, rnl.mutable_data(),
    neff.mutable_data(), sigma8.mutable_data(), halo_sigma2.mutable_data(),
    halo_slope.mutable_data());
  finite_3pcf(rnl, "variance nonlinear radii");
  finite_3pcf(neff, "variance effective slopes");
  finite_3pcf(sigma8, "variance sigma8");
  finite_3pcf(halo_sigma2, "halo variances");
  finite_3pcf(halo_slope, "halo logarithmic slopes");

  array_3pcf owned_z(z.size());
  array_3pcf owned_mass(mass.size());
  std::copy_n(z.data(), z.size(), owned_z.mutable_data());
  std::copy_n(mass.data(), mass.size(), owned_mass.mutable_data());
  py::dict result;
  result["z"] = owned_z;
  result["mass"] = owned_mass;
  result["rnl"] = rnl;
  result["neff"] = neff;
  result["sigma8"] = sigma8;
  result["halo_sigma2"] = halo_sigma2;
  result["halo_dlnsigma_dlnm"] = halo_slope;
  result["core_rho"] = core_rho;
  result["reference_rho"] = 0.0 == reference_rho ? core_rho : reference_rho;
  return result;
}

// Only fit fields belong to the model constructor. A full variance_snapshot
// also contains halo rows; select these four named arrays explicitly.
static void bind_threepcf_core_bridge(
  py::module_& module // Python module receiving the bindings
)
{
  py::module_ core = module.def_submodule("core",
      "Bridge between Cosmolike's global structs and the three-point "
      "model: struct installers, supplied-tier setters and the cached "
      "model_from_core factory (threepcf_core_bridge.h).");

  core.def("variance_snapshot", &bridge_variance_snapshot_3pcf,
      py::arg("z").noconvert(), py::arg("mass") = py::none(),
      py::arg("reference_rho") = 0.0, py::arg("include_fit") = true,
      "Owned total-matter variance arrays from the installed linear power. "
      "No cosmology is installed or changed. reference_rho fixes the "
      "mass-to-radius convention; zero uses core rho_m. include_fit=False "
      "skips Gaussian preparation. Halo slopes are d ln sigma/d ln M.");

  core.def("variance_sampling", []() {
    const auto sampling = threepcf_core_variance_sampling();
    py::dict result;
    result["mass_nodes"] = sampling.mass_nodes;
    result["scale_factor_nodes"] = sampling.scale_factor_nodes;
    result["gaussian_radius_nodes"] = sampling.gaussian_radius_nodes;
    result["gaussian_scale_factor_nodes"] =
        sampling.gaussian_scale_factor_nodes;
    result["gaussian_inverse_nodes"] = sampling.gaussian_inverse_nodes;
    return result;
  }, "Copy the five current shared variance table counts after initial_setup.");

  core.def("set_variance_sampling", [](
    const int mass_nodes, // top-hat ln(M) sampling
    const int scale_factor_nodes, // top-hat a sampling
    const int gaussian_radius_nodes, // Gaussian ln(R) sampling
    const int gaussian_scale_factor_nodes, // Gaussian a sampling
    const int gaussian_inverse_nodes // Gaussian inverse-table sampling
  ) {
    const threepcf_variance_sampling sampling = {
      mass_nodes, scale_factor_nodes, gaussian_radius_nodes,
      gaussian_scale_factor_nodes, gaussian_inverse_nodes
    };
    threepcf_core_set_variance_sampling(&sampling);
  }, "Install complete shared variance counts; unchanged counts retain caches. "
     "Derive refinements from a saved baseline, not the previous result.",
      py::arg("mass_nodes").noconvert(),
      py::arg("scale_factor_nodes").noconvert(),
      py::arg("gaussian_radius_nodes").noconvert(),
      py::arg("gaussian_scale_factor_nodes").noconvert(),
      py::arg("gaussian_inverse_nodes").noconvert());

  // --- supplied tier -----------------------------------------------------
  core.def("set_scalars", &threepcf_bridge_set_scalars,
      "Legacy cosmological scalars; changing them invalidates supplied B. "
      "Resupply the theory bispectrum after a change.",
      py::arg("ns"), py::arg("sigma8"), py::arg("w0"), py::arg("wa"));

  core.def("set_los", [](
    const array_3pcf& z // redshift coordinates of the supplied values
  ) {
    threepcf_bridge_set_los(bridge_vector_3pcf(z, "los_z"),
        static_cast<int>(z.size()));
  }, "Supply the line-of-sight node redshifts.",
      py::arg("z").noconvert());

  core.def("set_resolution", &threepcf_bridge_set_resolution,
      "Supply the model preparation resolution (physics_3pcf.h); the "
      "defaults equal the PhysicsModel constructor's.",
      py::arg("kernel_nz") = 101, py::arg("source_readers") = 0,
      py::arg("table_boost") = 8, py::arg("fit_nz") = 33);

  core.def("set_sources", &bridge_sources_3pcf,
      "Supply shifted source densities on their own redshift axis "
      "(None, None clears; the struct-read path then requires every "
      "photo-z shift to be exactly zero).",
      py::arg("source_z") = py::none(), py::arg("sources") = py::none());

  core.def("set_pixel_window", [](
    // optional W(ell) at integer multipoles; None clears it
    const py::object& window
  ) {
    if (window.is_none()) {
      threepcf_bridge_set_pixel_window(NULL, 0);
      return;
    }
    if (!(array_3pcf::check_(window))) {
      spdlog::critical("3pcf interface: the pixel window must be a "
        "contiguous float64 vector");
      std::exit(1);
    }
    const auto values = py::reinterpret_borrow<array_3pcf>(window);
    threepcf_bridge_set_pixel_window(
        bridge_vector_3pcf(values, "pixel window"),
        static_cast<int>(values.size()));
  }, "Supply W(ell) at integer ell (None clears).",
      py::arg("window") = py::none());

  core.def("set_feedback", &bridge_feedback_3pcf,
      "Supply the paired hydro/DMO response tables and the shared "
      "log-response amplitude (empty dict with amplitude 0 clears).",
      py::arg("feedback") = py::dict(), py::arg("amplitude") = 0.0);

  core.def("set_baseline_bispectrum", &bridge_baseline_3pcf,
      "Supply absolute B, Omega_m(z) on the core background grid, and sigma8. "
      "None clears only the B override; existing model owners stay valid.",
      py::arg("baseline") = py::none(),
      py::arg("omega_m_z") = py::none(),
      py::arg("sigma8") = py::none());

  core.def("set_prescriptions", &threepcf_bridge_set_prescriptions,
      "Supply the IA prescription selectors power_mode, ct_mode and "
      "suppress_only (physics_3pcf.h).",
      py::arg("power_mode") = 1, py::arg("ct_mode") = 0,
      py::arg("suppress_only") = 0);

  // --- struct installers (standalone-module equivalents) -----------------
  core.def("set_cosmological_parameters",
      &threepcf_core_set_cosmological_parameters,
      "Install Omega_m (massive neutrinos included), Omega_b, H0 in "
      "km/s/Mpc and omega_nu h^2 into the cosmology struct.",
      py::arg("omega_matter"), py::arg("omega_baryon"),
      py::arg("hubble"), py::arg("omega_nu_h2") = 0.0);

  core.def("set_distances", [](
    const array_3pcf& z, // redshift coordinates of the supplied values
    const array_3pcf& chi // comoving distances at each z, in Mpc/h
  ) {
    rank_3pcf(z, 1, "z");
    rank_3pcf(chi, 1, "chi");
    finite_3pcf(z, "z");
    finite_3pcf(chi, "chi");
    if (!(z.size() == chi.size())) {
      spdlog::critical("3pcf interface: z and chi sizes disagree");
      std::exit(1);
    }
    threepcf_core_set_distances(z.data(), chi.data(),
        static_cast<int>(z.size()));
  }, "Install chi(z) in Mpc/h into the cosmology struct.",
      py::arg("z").noconvert(), py::arg("chi").noconvert());

  core.def("set_growth", [](
    const array_3pcf& z, // redshift coordinates of the supplied values
    const array_3pcf& G // growth table G(z)=(1+z)D(z)
  ) {
    rank_3pcf(z, 1, "z");
    rank_3pcf(G, 1, "G");
    finite_3pcf(z, "z");
    finite_3pcf(G, "G");
    if (!(z.size() == G.size())) {
      spdlog::critical("3pcf interface: z and G sizes disagree");
      std::exit(1);
    }
    threepcf_core_set_growth(z.data(), G.data(),
        static_cast<int>(z.size()));
  }, "Install G(z) with D = G * a into the cosmology struct.",
      py::arg("z").noconvert(), py::arg("G").noconvert());

  // one lambda serves both power installers: same layout, same units
  const auto power_installer = [](
    const array_3pcf& log10k, // uniform log10 k coordinates, k in h/Mpc
    const array_3pcf& z, // redshift coordinates of the supplied values
    const array_3pcf& lnP, // flattened ln P [k,z], P in (Mpc/h)^3
    // true installs linear power; false installs nonlinear power
    const bool linear
  ) {
    rank_3pcf(log10k, 1, "log10k");
    rank_3pcf(z, 1, "z");
    rank_3pcf(lnP, 1, "lnP");
    finite_3pcf(log10k, "log10k");
    finite_3pcf(z, "z");
    finite_3pcf(lnP, "lnP");
    if (!(lnP.size() == log10k.size()*z.size())) {
      spdlog::critical("3pcf interface: lnP must hold nk*nz entries "
        "flattened as lnP[i*nz + j]");
      std::exit(1);
    }
    if (linear) {
      threepcf_core_set_linear_power_spectrum(log10k.data(),
          static_cast<int>(log10k.size()), z.data(),
          static_cast<int>(z.size()), lnP.data());
    }
    else {
      threepcf_core_set_non_linear_power_spectrum(log10k.data(),
          static_cast<int>(log10k.size()), z.data(),
          static_cast<int>(z.size()), lnP.data());
    }
  };

  core.def("set_linear_power_spectrum", [power_installer](
      const array_3pcf& log10k, // uniform log10 k coordinates, k in h/Mpc
      const array_3pcf& z, // redshift coordinates of the supplied values
      const array_3pcf& lnP // flattened ln P [k,z], P in (Mpc/h)^3
  ) {
    power_installer(log10k, z, lnP, true);
  }, "Install ln P_lin(log10 k, z), flattened as lnP[i*nz + j], into "
      "the cosmology struct (k in h/Mpc, P in (Mpc/h)^3).",
      py::arg("log10k").noconvert(), py::arg("z").noconvert(),
      py::arg("lnP").noconvert());

  core.def("set_non_linear_power_spectrum", [power_installer](
      const array_3pcf& log10k, // uniform log10 k coordinates, k in h/Mpc
      const array_3pcf& z, // redshift coordinates of the supplied values
      const array_3pcf& lnP // flattened ln P [k,z], P in (Mpc/h)^3
  ) {
    power_installer(log10k, z, lnP, false);
  }, "Install ln P_nonlin(log10 k, z), same layout and units as the "
      "linear installer.",
      py::arg("log10k").noconvert(), py::arg("z").noconvert(),
      py::arg("lnP").noconvert());

  core.def("set_source_sample_size",
      &threepcf_core_set_source_sample_size,
      "Declare the number of source tomographic bins.",
      py::arg("ntomo"));

  core.def("set_source_sample", [](
    const array_3pcf& nofz // redshift column followed by source-density columns
  ) {
    rank_3pcf(nofz, 2, "nofz");
    finite_3pcf(nofz, "nofz");
    if (!(2 <= nofz.shape(1))) {
      spdlog::critical("3pcf interface: the n(z) table needs a z "
        "column plus at least one bin column");
      std::exit(1);
    }
    threepcf_core_set_source_sample(nofz.data(),
        static_cast<int>(nofz.shape(0)),
        static_cast<int>(nofz.shape(1)) - 1);
  }, "Install the source n(z) table [nzbins, ntomo+1] (column 0 is "
      "the redshift axis) into the redshift struct.",
      py::arg("nofz").noconvert());

  core.def("set_nuisance_shear_photoz", [](
    const array_3pcf& bias // one source photo-z displacement per bin
  ) {
    threepcf_core_set_nuisance_shear_photoz(
        bridge_vector_3pcf(bias, "photo-z shifts"),
        static_cast<int>(bias.size()));
  }, "Install the per-bin source photo-z shifts.",
      py::arg("bias").noconvert());

  core.def("init_IA", &threepcf_core_init_ia,
      "Install the IA registry flags (IA.h: model and redshift "
      "dependence).",
      py::arg("ia_model"), py::arg("ia_redshift_evolution"));

  core.def("set_nuisance_ia", [](
    const array_3pcf& A1, // linear tidal-alignment parameters
    const array_3pcf& A2, // quadratic tidal-torquing parameters
    // density-weighting parameters of the alignment model
    const array_3pcf& B_TA
  ) {
    rank_3pcf(A1, 1, "A1");
    rank_3pcf(A2, 1, "A2");
    rank_3pcf(B_TA, 1, "B_TA");
    finite_3pcf(A1, "A1");
    finite_3pcf(A2, "A2");
    finite_3pcf(B_TA, "B_TA");
    if (!(A1.size() == A2.size() && A1.size() == B_TA.size())) {
      spdlog::critical("3pcf interface: IA amplitude sizes disagree");
      std::exit(1);
    }
    threepcf_core_set_nuisance_ia(A1.data(), A2.data(), B_TA.data(),
        static_cast<int>(A1.size()));
  }, "Install the IA amplitudes for the active nuisance.IA mode.",
      py::arg("A1").noconvert(), py::arg("A2").noconvert(),
      py::arg("B_TA").noconvert());

  core.def("set_nuisance_ia_ct", &threepcf_core_set_nuisance_ia_ct,
      "Install the velocity-shear amplitude C_t (nuisance.ia[1][2]).",
      py::arg("ct"));

  core.def("initial_setup", &threepcf_core_initial_setup,
      "Reset the global structs and the bridge, as initial_setup does "
      "in full-core builds.");

  // --- translations and the factory --------------------------------------
  core.def("ia_parameters_from_core", &ia_parameters_from_core_3pcf,
      "Translate the stored nuisance IA state into the model's "
      "parameter dictionary (the single conversion site in "
      "threepcf_core_bridge.c).");

  core.def("model_from_core", &model_from_core_3pcf,
      "Return the cached PhysicsModel built from the global structs "
      "and the supplied tier; rebuilds only when the two-tier guard "
      "reports a change.");

  core.def("model_generation", []() {
    return threepcf_bridge_generation();
  }, "Build generation of the cached model; advances on every "
      "rebuild.");
}



// Measurement state follows the ordinary IP singleton: one active dataset
// per process. The full vector is [two-point rows, three-point rows] in the
// supplied order. Select BOTH covariance axes before inversion; selecting
// an inverse would condition on the discarded measurements instead.
class IP3PCF
{
public:
  static IP3PCF& get_instance()
  {
    static IP3PCF instance;
    return instance;
  }

  void reset()
  {
    ready_ = false;
    data_.reset();
    covariance_.reset();
    cholesky_.reset();
    compression_.reset();
    selected_.reset();
    mask_.reset();
    full_size_ = 0;
  }

  void set_data(
    const array_3pcf& data, // measured joint vector before selection
    const array_3pcf& covariance, // full C22, C23, C32 and C33 blocks
    const int_array_3pcf& indices, // retained row indices, in requested order
    const array_3pcf& compression, // optional [Ncompressed,Nselected] matrix
    const int mock_count, // zero: Gaussian; otherwise raw sample covariance
    const int parameter_count // varied mean parameters for Percival exponent
  )
  {
    if (data.ndim() != 1 || covariance.ndim() != 2 || indices.ndim() != 1
        || covariance.shape(0) != data.size()
        || covariance.shape(1) != data.size() || indices.size() == 0) {
      throw py::value_error("Joint data, covariance and selected rows disagree.");
    }
    const auto values = data.unchecked<1>();
    const auto matrix = covariance.unchecked<2>();
    const auto rows = indices.unchecked<1>();
    arma::uvec selected(indices.size());
    arma::ivec mask(data.size(), arma::fill::zeros);
    arma::vec observed(indices.size());
    arma::mat reduced(indices.size(), indices.size());
    for (py::ssize_t i=0; i<indices.size(); i++) {
      if (rows(i) < 0 || rows(i) >= data.size() || mask(rows(i)) != 0) {
        throw py::value_error("Joint indices must be distinct valid rows.");
      }
      mask(rows(i)) = 1;
      selected(i) = rows(i);
      observed(i) = values(rows(i));
    }
    for (arma::uword i=0; i<selected.n_elem; i++) {
      for (arma::uword j=0; j<selected.n_elem; j++) {
        reduced(i,j) = matrix(selected(i),selected(j));
      }
    }
    if (!observed.is_finite() || !reduced.is_finite()
        || !arma::approx_equal(reduced, reduced.t(), "both", 1.e-13, 1.e-10)) {
      throw py::value_error("Selected joint data need finite symmetric covariance.");
    }
    arma::mat transform;
    if (compression.size() != 0) {
      if (compression.ndim() != 2 || compression.shape(0) < 1
          || compression.shape(1) != indices.size()) {
        throw py::value_error("Compression columns must match selected data.");
      }
      const auto input = compression.unchecked<2>();
      transform.set_size(compression.shape(0), compression.shape(1));
      for (arma::uword i=0; i<transform.n_rows; i++) {
        for (arma::uword j=0; j<transform.n_cols; j++) {
          transform(i,j) = input(i,j);
        }
      }
      if (!transform.is_finite()) {
        throw py::value_error("Compression matrix must be finite.");
      }
      observed = transform*observed;
      reduced = transform*reduced*transform.t();
    }
    arma::mat lower;
    if (!arma::chol(lower, reduced, "lower")) {
      throw py::value_error("Selected covariance must be positive definite.");
    }
    const double dimension = observed.n_elem;
    const double logdet = 2.0*arma::sum(arma::log(lower.diag()));
    const double pi = std::acos(-1.0);
    double exponent = 0.0;
    double normalization = -0.5*(dimension*std::log(2.0*pi) + logdet);
    if (mock_count != 0) {
      if (mock_count <= dimension+4 || parameter_count < 1
          || parameter_count > dimension) {
        throw py::value_error("Percival needs Ns>Nd+4 and 1<=Np<=Nd.");
      }
      const double difference = mock_count-dimension;
      const double correction = (difference-2.0)/
        ((difference-1.0)*(difference-4.0));
      const double product = correction*(dimension-parameter_count);
      exponent = parameter_count+2.0+(mock_count-1.0+product)/(1.0+product);
      if (exponent <= dimension) {
        throw py::value_error("Percival requires m>Nd for a normalized density.");
      }
      normalization = std::lgamma(0.5*exponent) -
        std::lgamma(0.5*(exponent-dimension)) -
        0.5*dimension*std::log(pi*(mock_count-1.0)) - 0.5*logdet;
    }
    // Publish only a fully checked replacement. A failed setup leaves the
    // previous data usable; changing any mask or compression refactors C.
    full_size_ = data.size();
    selected_ = std::move(selected);
    mask_ = std::move(mask);
    data_ = std::move(observed);
    covariance_ = std::move(reduced);
    cholesky_ = std::move(lower);
    compression_ = std::move(transform);
    mock_count_ = mock_count;
    exponent_ = exponent;
    log_normalization_ = normalization;
    ready_ = true;
  }

  arma::vec select(
    const array_3pcf& prediction // full joint theory vector
  ) const
  {
    require_data();
    if (prediction.ndim() != 1 || prediction.size() != full_size_) {
      throw py::value_error("Joint theory vector has the wrong full length.");
    }
    const auto values = prediction.unchecked<1>();
    arma::vec selected(selected_.n_elem);
    for (arma::uword i=0; i<selected_.n_elem; i++) {
      selected(i) = values(selected_(i));
    }
    if (!selected.is_finite()) {
      throw py::value_error("Selected joint theory must be finite.");
    }
    if (compression_.n_elem != 0) { selected = compression_*selected; }
    return selected;
  }

  double get_chi2(
    const arma::vec& prediction // selected and optionally compressed mean
  ) const
  {
    require_data();
    if (prediction.n_elem != data_.n_elem || !prediction.is_finite()) {
      throw py::value_error("Selected mean must have the finite shape of data.");
    }
    const arma::vec residual = prediction-data_;
    const arma::vec whitened = arma::solve(arma::trimatl(cholesky_), residual,
      arma::solve_opts::fast);
    return arma::dot(whitened, whitened);
  }

  double get_loglike(
    const double chi2 // residual contraction, without parameter priors
  ) const
  {
    require_data();
    if (mock_count_ == 0) { return log_normalization_ - 0.5*chi2; }
    return log_normalization_ - 0.5*exponent_*std::log1p(chi2/(mock_count_-1.0));
  }

  void require_data() const
  {
    if (!ready_) { throw py::value_error("Initialize joint data before chi2."); }
  }

  arma::vec data() const { require_data(); return data_; }
  arma::mat covariance() const { require_data(); return covariance_; }
  double exponent() const { require_data(); return exponent_; }
  double normalization() const { require_data(); return log_normalization_; }

private:
  IP3PCF() = default;
  IP3PCF(const IP3PCF&) = delete;
  IP3PCF& operator=(const IP3PCF&) = delete;
  bool ready_ = false;
  py::ssize_t full_size_ = 0;
  int mock_count_ = 0;
  double exponent_ = 0.0;
  double log_normalization_ = 0.0;
  arma::uvec selected_;
  arma::ivec mask_;
  arma::vec data_;
  arma::mat covariance_;
  arma::mat cholesky_;
  arma::mat compression_;
};

// Numerical controls are resolved in the interface, as in the two-point
// library. Python keeps ordinary dictionaries. Every count below is derived
// from its original baseline, so repeating initialization cannot square a
// boost. These helpers do not change physical ranges or model parameters.
py::dict init_accuracy_3pcf(
  const py::object& value // optional dictionary of global and stage boosts
)
{
  py::dict result;
  const char* names[] = {"accuracy_boost",
    "integration_accuracy_boost",
    "chi_boost",
    "kernel_boost",
    "source_boost",
    "power_table_boost",
    "bispectrum_boost",
    "feedback_table_boost",
    "feedback_z_boost",
    "feedback_k_boost",
    "feedback_u_boost",
    "feedback_v_boost",
    "fit_z_boost",
    "variance_mass_boost",
    "variance_a_boost",
    "variance_radius_boost",
    "variance_inverse_boost",
    "triangle_boost",
    "multipole_boost",
    "angular_boost",
    "fft_boost",
    "profile_boost",
    "profile_upsample_boost",
    "aperture_ell_boost",
    "aperture_phi_boost",
    "xi_ell_boost",
    "xi_theta_boost",
    "covariance_ell_boost",
    "covariance_window_boost",
    "covariance_phi_boost",
    "covariance_radius_boost",
    "covariance_k_boost",
    "covariance_mass_boost"};
  for (const char* name : names) { result[name] = 1.0; }
  if (value.is_none()) { return result; }
  if (!py::isinstance<py::dict>(value)) {
    throw py::value_error("Accuracy controls must be a dictionary.");
  }
  const auto supplied = py::cast<py::dict>(value);
  for (auto item : supplied) {
    if (!result.contains(item.first)) {
      throw py::value_error("Unknown accuracy control: " +
        py::cast<std::string>(py::str(item.first)));
    }
    const double factor = py::cast<double>(item.second);
    if (py::isinstance<py::bool_>(item.second) || !std::isfinite(factor)
        || factor < 1.0) {
      throw py::value_error(py::cast<std::string>(item.first) +
        " must be finite and >=1.");
    }
    result[item.first] = factor;
  }
  return result;
}

int accuracy_count_3pcf(
  const py::dict& accuracy, // global and stage multipliers
  const double baseline, // original count, before any refinement
  const std::string& stage, // individual multiplier name
  const bool integration, // choose integration rather than sampling boost
  const bool endpoints // refine N-1 intervals while retaining both endpoints
)
{
  const auto controls = init_accuracy_3pcf(accuracy);
  if (!controls.contains(stage.c_str())) {
    throw py::value_error("Unknown accuracy stage: " + stage);
  }
  if (!std::isfinite(baseline) || baseline < 1.0
      || std::floor(baseline) != baseline) {
    throw py::value_error("Baseline count must be a positive integer.");
  }
  const char* global = integration ? "integration_accuracy_boost" :
    "accuracy_boost";
  const double factor = py::cast<double>(controls[global]) *
    py::cast<double>(controls[stage.c_str()]);
  const double offset = endpoints ? 1.0 : 0.0;
  const double count = std::ceil((baseline-offset)*factor + offset);
  if (!std::isfinite(count) || count > INT_MAX) {
    throw py::value_error("Resolved count exceeds the C int range.");
  }
  return static_cast<int>(count);
}

static array_3pcf accuracy_los_3pcf(
  const py::dict& accuracy, // global and chi multipliers
  const array_3pcf& baseline // supplied positive increasing LOS redshifts
)
{
  const auto controls = init_accuracy_3pcf(accuracy);
  if (baseline.ndim() != 1 || baseline.size() < 2) {
    throw py::value_error("LOS baseline needs two increasing nodes.");
  }
  const double factor = py::cast<double>(controls["accuracy_boost"]) *
    py::cast<double>(controls["integration_accuracy_boost"]) *
    py::cast<double>(controls["chi_boost"]);
  if (!std::isfinite(factor) || factor > 1024.0) {
    throw py::value_error("LOS subdivision exceeds 1024.");
  }
  int subdivision = 1;
  while (subdivision < factor) { subdivision *= 2; }
  const auto nodes = baseline.unchecked<1>();
  for (py::ssize_t i=0; i<baseline.size(); i++) {
    if (!std::isfinite(nodes(i)) || nodes(i) <= 0.0
        || (i > 0 && nodes(i) <= nodes(i-1))) {
      throw py::value_error("LOS baseline needs increasing positive nodes.");
    }
  }
  if ((baseline.size()-1) > (INT_MAX-1)/subdivision) {
    throw py::value_error("LOS node count exceeds the C int range.");
  }
  array_3pcf result((baseline.size()-1)*subdivision + 1);
  auto output = result.mutable_unchecked<1>();
  for (py::ssize_t i=0; i<baseline.size()-1; i++) {
    for (int j=0; j<subdivision; j++) {
      output(i*subdivision+j) = nodes(i) +
        (static_cast<double>(j)/subdivision)*(nodes(i+1)-nodes(i));
    }
  }
  output(result.size()-1) = nodes(baseline.size()-1);
  return result;
}

static py::dict accuracy_model_3pcf(
  const py::dict& accuracy, // multipliers applied to the supplied baselines
  const int kernel_nz, // source-window nodes
  const int table_boost, // dense-reader interpolation ratio
  const int fit_nz // requested external fit anchors
)
{
  py::dict result;
  result["kernel_nz"] = accuracy_count_3pcf(accuracy, kernel_nz,
    "kernel_boost", false, true);
  result["table_boost"] = accuracy_count_3pcf(accuracy, table_boost,
    "power_table_boost", false, false);
  result["fit_nz"] = accuracy_count_3pcf(accuracy, fit_nz,
    "fit_z_boost", false, true);
  return result;
}

static py::dict accuracy_variance_3pcf(
  const py::dict& accuracy, // shared variance sampling multipliers
  const py::dict& baseline // original five variance counts
)
{
  const auto controls = init_accuracy_3pcf(accuracy);
  const char* names[] = {"mass_nodes", "scale_factor_nodes",
    "gaussian_radius_nodes", "gaussian_scale_factor_nodes",
    "gaussian_inverse_nodes"};
  const char* stages[] = {"variance_mass_boost", "variance_a_boost",
    "variance_radius_boost", "variance_a_boost", "variance_inverse_boost"};
  if (baseline.size() != 5) {
    throw py::value_error("Variance baseline requires five sampling counts.");
  }
  py::dict result;
  for (int i=0; i<5; i++) {
    const double count = py::cast<double>(baseline[names[i]]);
    const double factor = py::cast<double>(controls["accuracy_boost"]) *
      py::cast<double>(controls[stages[i]]);
    if (!std::isfinite(count) || count < 2 || count != std::floor(count)
        || !std::isfinite(factor) || factor > 16.0) {
      throw py::value_error("Variance counts need N>=2 and boost<=16.");
    }
    int multiplier = 1;
    while (multiplier < factor) { multiplier *= 2; }
    const double resolved = (count-1)*multiplier + 1;
    if (resolved > INT_MAX) {
      throw py::value_error("Resolved variance count exceeds the C range.");
    }
    result[names[i]] = static_cast<int>(resolved);
  }
  return result;
}

py::dict accuracy_projection_3pcf(
  const py::dict& accuracy, // interpolation and angular-integration multipliers
  const py::dict& baseline // original projection settings and physical ranges
)
{
  py::dict result(baseline.attr("copy")());
  const char* names[] = {"nr", "nu", "nv", "np", "npu_log",
    "npu_linear", "nshape", "nell", "npsi_log", "npsi_linear",
    "npsi", "nmu_log", "nmu_linear"};
  const int defaults[] = {35, 35, 25, 35, 8, 22, 80, 100, 50, 80, 80, 30, 50};
  for (int i=0; i<13; i++) {
    const double count = result.contains(names[i]) ?
      py::cast<double>(result[names[i]]) : defaults[i];
    if (i == 8 && count == 0.0) { continue; }
    const char* stage = i < 7 ? "triangle_boost" :
      (i < 11 ? "multipole_boost" : "angular_boost");
    result[names[i]] = accuracy_count_3pcf(accuracy, count, stage,
      i >= 11, i < 11);
  }
  return result;
}

py::dict accuracy_quadrature_3pcf(
  const py::dict& accuracy, // integration multiplier and individual stage
  const py::dict& baseline, // original count, bounds and quadrature convention
  const std::string& stage // selected integration control
)
{
  py::dict result(baseline.attr("copy")());
  result["count"] = accuracy_count_3pcf(accuracy,
    py::cast<double>(baseline["count"]), stage, true, false);
  return result;
}

// ============================================================================
// [SECTION] MODULE REGISTRATION
// ============================================================================

// Register both submodules. Python names repeat across the two routes;
// only ownership and storage conversion differ between them. Results
// keep the documented physical axes, and the notebook route exports
// matrices and cubes in Fortran storage order.
void reset_threepcf_joint_data()
{
  IP3PCF::get_instance().reset();
}

void bind_threepcf(
    py::module_& module // project extension receiving both interfaces
  )
{
  module.def("init_joint_data", [](const array_3pcf& data,
      const array_3pcf& covariance, const int_array_3pcf& indices,
      const array_3pcf& compression, const int mock_count,
      const int parameter_count) {
    IP3PCF::get_instance().set_data(data, covariance, indices, compression,
      mock_count, parameter_count);
  }, py::arg("data"), py::arg("covariance"), py::arg("indices"),
    py::arg("compression"), py::arg("mock_count") = 0,
    py::arg("parameter_count") = 0);
  module.def("reset_joint_data", &reset_threepcf_joint_data);
  module.def("select_joint_data_vector", [](const array_3pcf& prediction) {
    auto result = carma::col_to_arr(IP3PCF::get_instance().select(prediction));
    return result.reshape({result.size()});
  }, py::arg("prediction"));
  module.def("compute_joint_chi2", [](const array_3pcf& prediction) {
    const auto& ip = IP3PCF::get_instance();
    return ip.get_chi2(ip.select(prediction));
  }, py::arg("prediction"));
  module.def("compute_joint_loglike", [](const array_3pcf& prediction) {
    const auto& ip = IP3PCF::get_instance();
    return ip.get_loglike(ip.get_chi2(ip.select(prediction)));
  }, py::arg("prediction"));
  module.def("score_joint_mean", [](const array_3pcf& mean) {
    if (mean.ndim() != 1) {
      throw py::value_error("Selected mean must be one-dimensional.");
    }
    const arma::vec prediction(mean.data(), mean.size());
    const auto& ip = IP3PCF::get_instance();
    const double chi2 = ip.get_chi2(prediction);
    return py::make_tuple(ip.get_loglike(chi2), chi2);
  }, py::arg("mean"));
  module.def("get_joint_data", []() {
    auto result = carma::col_to_arr(IP3PCF::get_instance().data());
    return result.reshape({result.size()});
  });
  module.def("get_joint_covariance", []() {
    return carma::mat_to_arr(IP3PCF::get_instance().covariance());
  });
  module.def("get_joint_exponent", []() {
    return IP3PCF::get_instance().exponent();
  });
  module.def("get_joint_log_normalization", []() {
    return IP3PCF::get_instance().normalization();
  });

  module.def("init_3pcf_accuracy", &init_accuracy_3pcf,
    py::arg("value") = py::none());
  module.def("accuracy_count", &accuracy_count_3pcf,
    py::arg("accuracy"), py::arg("baseline"), py::arg("stage"),
    py::arg("integration") = false, py::arg("endpoints") = false);
  module.def("accuracy_los_nodes", &accuracy_los_3pcf,
    py::arg("accuracy"), py::arg("baseline"));
  module.def("accuracy_model_settings", &accuracy_model_3pcf,
    py::arg("accuracy"), py::arg("kernel_nz") = 101,
    py::arg("table_boost") = 8, py::arg("fit_nz") = 33);
  module.def("accuracy_variance_settings", &accuracy_variance_3pcf,
    py::arg("accuracy"), py::arg("baseline"));
  module.def("accuracy_projection_settings", &accuracy_projection_3pcf,
    py::arg("accuracy"), py::arg("baseline"));
  module.def("accuracy_quadrature_settings", &accuracy_quadrature_3pcf,
    py::arg("accuracy"), py::arg("baseline"), py::arg("stage"));

  // Triplet labels cross the Python/C boundary as 32-bit C int.
  static_assert(4 == sizeof(int), "triplets require 32-bit C int");

  // Pin BLAS to one thread at import, before any OpenMP region runs
  // (set_threads_3pcf explains why).
  openblas_set_num_threads(1);

  module.def("set_omp_threads", &set_threads_3pcf, py::arg("n"));
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

#ifndef COSMOLIKE_NO_COVARIANCE
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

#endif

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

#ifndef COSMOLIKE_NO_COVARIANCE
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

#endif

  // The core-struct bridge registers last: its factory returns the
  // PhysicsModel owner type that bind_threepcf_physics registers at
  // import, and pybind only needs that class registered before the
  // first call, not before this def.
  bind_threepcf_core_bridge(module);
}

}
