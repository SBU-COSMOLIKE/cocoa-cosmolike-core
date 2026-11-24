#include <algorithm>
#include <array>
#include <vector>
#include "aperture_3pcf.h"
#include "threepcf_interface.hpp"
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <carma.h>
#include <spdlog/spdlog.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "coupling_3pcf.h"
#include "fftlog_3pcf.h"
#include "transform_3pcf.h"
#include "transform_interface.hpp"
#include "transform_wrapper.hpp"
#include "physics_interface.hpp"
#include "cosmolike/basics.h"

namespace py = pybind11;
namespace cosmolike_interface {

using transform_array = py::array_t<double, py::array::c_style>;
using transform_indices = py::array_t<int, py::array::c_style>;

// ---------------------------------------------------------------------------
// transform_interface.cpp: the pybind entry points of the transform stage
// of the shear three-point pipeline. Python supplies bispectrum Legendre
// coefficients - or a prepared physical grid that evaluates them - and
// receives the natural components of the shear three-point function on
// a fixed triangle grid, plus their aperture-moment contractions.
//
// Two routes reach the same C engines. The production route borrows
// C-contiguous float64 NumPy arrays after dtype, layout, alignment and
// finiteness checks, and allocates NumPy results directly. The notebook
// route copies arbitrary NumPy-compatible input through Armadillo, so
// sliced, Fortran-order and read-only arrays also work. The C plan
// copies all geometry at construction, so mutating the original NumPy
// arrays afterwards cannot invalidate a running or later transform.
//
// Every call keeps the GIL - Python's global interpreter lock, which
// serializes Python threads - because each plan owns one set of mutable
// FFT scratch for the duration of a call; releasing the lock would let
// a second Python thread into that scratch.
//
// Terms this file uses:
//   natural components - the four complex three-shear correlators
//       Gamma_0..Gamma_3, distinguished by which of the three shears is
//       complex-conjugated: Gamma_0 conjugates none, Gamma_i the shear
//       at vertex i (Schneider & Lombardi, arXiv:astro-ph/0207454).
//   b_L - the Legendre coefficients, in mu, of the projected E-mode
//       bispectrum; mu is the cosine in the triangle closure
//       ell3^2 = ell1^2 + ell2^2 - 2 ell1 ell2 mu.
//   psi - the opening angle arctan(ell2/ell1) in [0, pi/2]; it encodes
//       the ratio of the two multipole legs but is itself an angle.
//   FFTLog bias nu - the power-law tilt divided out of the integrand so
//       the integral on the logarithmic grid becomes a convolution that
//       the FFT evaluates.
//   xy - the constant in theta = xy/ell that fixes where the FFTLog
//       output grid lands.
//   taper - suppression of the highest log-frequency coefficients to
//       damp the ringing that the FFT's periodic assumption causes.
//   TreeCorr - the external catalogue-correlation code whose component
//       convention the aperture outputs adopt.
//   SAS binning - side-angle-side: a triangle is recorded by two side
//       lengths and the opening angle between them, each on a finite
//       bin. The finite-SAS aperture statistics integrate the natural
//       components over those finite bins instead of sampling ideal
//       point triangles.
//   aperture mass - the convergence field smoothed by a compensated
//       filter of scale R; the third moment of three such filters,
//       measured from shear, is the statistic the aperture methods
//       below predict.
//   prepared projection - a physical grid, built once per ordered
//       source triplet, that tabulates the projected bispectrum; the
//       plan rereads the table instead of repeating the line-of-sight
//       projection quadrature for every triangle.
// ---------------------------------------------------------------------------



// ============================================================================
// [SECTION] VALIDATION HELPERS
// ============================================================================


// Check the rank, axis lengths and pointer alignment of a borrowed
// NumPy array before the first typed read. C-contiguity describes
// strides, not alignment: a buffer built with a one-byte offset can be
// contiguous, yet a typed read through it is undefined behavior in C.
template <typename Array>
static void shape_transform(
    const Array& values, // borrowed NumPy input
    const int rank // required number of axes
  )
{
  if (!(values.ndim() == rank)) {
    spdlog::critical("3pcf transform: array has wrong rank");
    std::exit(1);
  }

  for (int axis=0; axis<rank; axis++) {
    if (!(0 < values.shape(axis)
          && INT_MAX >= values.shape(axis))) {
      spdlog::critical("3pcf transform: invalid array axis");
      std::exit(1);
    }
  }

  using scalar = typename Array::value_type;
  const auto address = reinterpret_cast<std::uintptr_t>(values.data());
  if (!(0 == address%alignof(scalar))) {
    spdlog::critical("3pcf transform: unaligned array");
    std::exit(1);
  }
}


// Abort when any element of a real array is NaN or infinite.
static void finite_transform(
    const transform_array& values // aligned real buffer
  )
{
  for (py::ssize_t index=0; index<values.size(); index++) {
    if (!(std::isfinite(values.data()[index]))) {
      spdlog::critical("3pcf transform: array must contain finite values");
      std::exit(1);
    }
  }
}


// Report the OpenMP team size the C plan allocates scratch for; a
// build without OpenMP runs a single worker.
static int worker_count_transform()
{
  #ifdef _OPENMP
  return omp_get_max_threads();
  #else
  return 1;
  #endif
}


// Map a phase-convention name onto the C plan's integer code: "X"
// selects the X-mode projection (0), "centroid" and "centroid_tatt"
// both select the TATT reference centroid phase (1), and
// "centroid_main" selects its explicit conjugate (2). Other names abort.
static int projection_transform(
    const std::string& projection // phase-convention name
  )
{
  if (projection == "X") {
    return 0;
  }
  if (projection == "centroid" || projection == "centroid_tatt") {
    return 1;
  }
  if (!(projection == "centroid_main")) {
    spdlog::critical("3pcf transform: projection must be X, centroid_tatt, "
      "or centroid_main");
    std::exit(1);
  }
  return 2;
}



// ============================================================================
// [SECTION] NOTEBOOK INPUT CONVERSION
// ============================================================================

// Convert any notebook input into an owning Armadillo array of the
// requested rank. numpy.asarray performs the conversion inside Python
// itself, so this helper accepts whatever asarray accepts: lists,
// sliced views, read-only and Fortran-order arrays. The threepcf
// notebook converter instead insists on a NumPy array, because a long
// Python list that enters through the binding-level caster can arrive
// zero-filled; asarray runs before any caster and does not share that
// hazard. astype(copy=True, order="F") then yields an aligned owning
// temporary whose storage order matches Armadillo, and one further
// copy gives Armadillo ownership, so no borrowed lifetime escapes.
template <typename Array>
static Array notebook_transform_input(
    const py::object& values, // any NumPy-compatible notebook input
    const int rank // expected physical rank
  )
{
  py::module_ numpy = py::module_::import("numpy");
  py::object converted = numpy.attr("asarray")(values).attr("astype")(
      numpy.attr("float64"), py::arg("order")="F", py::arg("copy")=true);
  py::array_t<double, py::array::f_style> input(converted);

  if (!(input.ndim() == rank)) {
    spdlog::critical("3pcf transform: wrong notebook array rank");
    std::exit(1);
  }
  for (int axis=0; axis<rank; axis++) {
    if (!(0 < input.shape(axis)
          && INT_MAX >= input.shape(axis))) {
      spdlog::critical("3pcf transform: invalid notebook axis");
      std::exit(1);
    }
  }

  Array output;
  if constexpr (std::is_same_v<Array, arma::Col<double>>) {
    output.set_size(input.shape(0));
  } else if constexpr (std::is_same_v<Array, arma::Mat<double>>) {
    output.set_size(input.shape(0), input.shape(1));
  } else {
    output.set_size(input.shape(0), input.shape(1), input.shape(2));
  }

  std::copy_n(input.data(), input.size(), output.memptr());

  if (!(output.is_finite())) {
    spdlog::critical("3pcf transform: nonfinite notebook input");
    std::exit(1);
  }

  return output;
}



// ---------------------------------------------------------------------------
// Retained double-Bessel preparation for repeated integrands on one grid.
// A workspace contains two FFTW recipes and the Mellin and bias tables.
// Those depend on the fixed grid and settings, while the Fourier
// coefficients depend on the latest input. Keeping the workspace therefore
// avoids repeated preparation without treating old coefficients as new data.
//
// forward() copies the complete input and replaces its Fourier coefficients.
// inverse() can then reuse those coefficients for several order pairs;
// evaluate() performs both stages for a new input. The readiness flag prevents
// reading uninitialized coefficients before the first forward transform.
// Each returned array is an independent copy: another inverse may overwrite
// workspace scratch but cannot overwrite an earlier Python result.
//
// One serial workspace slice is sufficient for a standalone double-Bessel
// transform. Angular-mode OpenMP parallelism belongs to TransformPlan below.
// Every Python entry retains the GIL, so two Python threads cannot enter this
// owner's mutable scratch together. Different owners have separate storage.
// The unique owner also releases the workspace if output allocation throws.
//
// Cache invalidation:
// Geometry and order capacity stay fixed until this owner is destroyed.
// forward() replaces the coefficients for every supplied input. Core resets
// do not affect this independent workspace; no process-global cache is used.
// ---------------------------------------------------------------------------
class bessel_owner_3pcf {
 public:
  bessel_owner_3pcf(
      const transform_array& ell, // positive logarithmic Fourier grid
      const int max_order, // largest prepared cylindrical Bessel order
      const double nu1, // first FFTLog bias
      const double nu2, // second FFTLog bias
      const double xy, // theta times reversed ell
      const double dlnt, // logarithmic radial-bin width
      const int padding, // zero cells at each end
      const bool window, // coefficient taper enabled
      const double window_width // tapered frequency fraction
    )
  {
    shape_transform(ell, 1);
    finite_transform(ell);
    initialize(static_cast<int>(ell.size()), ell.data(), max_order,
        nu1, nu2, xy, dlnt, padding, window, window_width);
  }

  bessel_owner_3pcf(const bessel_owner_3pcf&) = delete;
  bessel_owner_3pcf& operator=(const bessel_owner_3pcf&) = delete;

  // The notebook constructor copies arbitrary NumPy-compatible coordinates
  // through Armadillo before the C workspace copies its immutable geometry.
  static std::unique_ptr<bessel_owner_3pcf> notebook(
      const py::object& ell, // copied notebook Fourier coordinates
      const int max_order, // largest prepared Bessel order
      const double nu1, // first bias
      const double nu2, // second bias
      const double xy, // output-grid alignment
      const double dlnt, // radial-bin width
      const int padding, // zero cells at each end
      const bool window, // coefficient taper enabled
      const double window_width // tapered frequency fraction
    )
  {
    const auto copied =
        notebook_transform_input<arma::Col<double>>(ell, 1);
    auto owner = std::unique_ptr<bessel_owner_3pcf>(new bessel_owner_3pcf());
    owner->initialize(static_cast<int>(copied.n_elem), copied.memptr(),
        max_order, nu1, nu2, xy, dlnt, padding, window, window_width);
    return owner;
  }

  // Copy a complete borrowed [nell, nell] integrand into the C input
  // slice and recompute the forward FFT coefficients. Refuses a shape
  // that differs from this plan's grid and any nonfinite value.
  void forward(
      const transform_array& integrand // [nell,nell], dln ell integrand
    )
  {
    shape_transform(integrand, 2);
    if (!(integrand.shape(0) == nell_
          && integrand.shape(1) == nell_)) {
      spdlog::critical("3pcf two-Bessel plan: integrand shape mismatch");
      std::exit(1);
    }
    finite_transform(integrand);
    std::copy_n(integrand.data(), integrand.size(),
        fftlog_input_3pcf(work_.get(), 0));
    fftlog_forward_3pcf(work_.get(), 0);
    ready_ = true;
  }

  // Contract the latest forward coefficients with the Bessel kernels
  // of one order pair and return an independent [nell, nell] NumPy
  // copy. Refuses orders outside the prepared capacity and refuses to
  // run before the first forward.
  transform_array inverse(
      const int order1, // first cylindrical Bessel order
      const int order2 // second cylindrical Bessel order
    )
  {
    validate_inverse(order1, order2);
    const double* values =
        fftlog_inverse_3pcf(work_.get(), 0, order1, order2);
    transform_array output({nell_, nell_});
    std::copy_n(values, output.size(), output.mutable_data());
    finite_transform(output);
    return output;
  }

  // Forward and inverse in one call: replace the stored input, then
  // return its transform for a single order pair.
  transform_array evaluate(
      const transform_array& integrand, // complete replacement input
      const int order1, // first cylindrical Bessel order
      const int order2 // second cylindrical Bessel order
    )
  {
    validate_orders(order1, order2);
    forward(integrand);
    return inverse(order1, order2);
  }

  // Notebook twin of forward: copy arbitrary NumPy-compatible input
  // through Armadillo before the same C forward transform runs.
  void forward_notebook(
      const py::object& integrand // copied input with arbitrary strides
    )
  {
    const auto copied =
        notebook_transform_input<arma::Mat<double>>(integrand, 2);
    bessel_forward_cpp_3pcf(work_.get(), nell_, copied);
    ready_ = true;
  }

  // Notebook twin of inverse: the Armadillo result returns through
  // carma as an independently owned NumPy array.
  py::object inverse_notebook(
      const int order1, // first cylindrical Bessel order
      const int order2 // second cylindrical Bessel order
    )
  {
    validate_inverse(order1, order2);
    return carma::mat_to_arr(
        bessel_inverse_cpp_3pcf(work_.get(), nell_, order1, order2));
  }

  // Notebook twin of evaluate: one copied input, one owned output.
  py::object evaluate_notebook(
      const py::object& integrand, // complete copied replacement input
      const int order1, // first cylindrical Bessel order
      const int order2 // second cylindrical Bessel order
    )
  {
    validate_orders(order1, order2);
    forward_notebook(integrand);
    return inverse_notebook(order1, order2);
  }

  // Independent copy of the output grid theta = xy/reversed(ell); a
  // Python-side write cannot reach the plan's own table.
  transform_array theta() const
  {
    transform_array result(nell_);
    std::copy_n(theta_.data(), nell_, result.mutable_data());
    return result;
  }

 private:
  bessel_owner_3pcf() = default;

  // Shared constructor body: build the C workspace, which copies the
  // grid, then tabulate the reversed reciprocal output grid
  // theta[i] = xy/ell[nell-1-i].
  void initialize(
      const int nell, // physical grid length
      const double* ell, // coordinates copied by the C workspace
      const int max_order, // largest prepared Bessel order
      const double nu1, // first bias
      const double nu2, // second bias
      const double xy, // output-grid alignment
      const double dlnt, // radial-bin width
      const int padding, // zero cells at each end
      const bool window, // coefficient taper enabled
      const double window_width // tapered frequency fraction
    )
  {
    work_.reset(fftlog_create_3pcf(nell, ell, max_order, 1,
        nu1, nu2, xy, dlnt, padding, window, window_width));
    nell_ = nell;
    max_order_ = max_order;
    theta_ = transform_array(nell_);
    for (int index=0; index<nell_; index++) {
      theta_.mutable_data()[index] = xy/ell[nell_-1-index];
    }
  }

  // Refuse any Bessel order outside the prepared capacity
  // [0, max_order]: no kernel table exists beyond it.
  void validate_orders(
      const int order1, // first requested Bessel order
      const int order2 // second requested Bessel order
    ) const
  {
    if (!(0 <= order1
          && order1 <= max_order_
          && 0 <= order2
          && order2 <= max_order_)) {
      spdlog::critical("3pcf two-Bessel plan: orders must be in [0, {}]",
          max_order_);
      std::exit(1);
    }
  }

  // Refuse an inverse before the first forward: the coefficient
  // buffer holds no data yet.
  void validate_inverse(
      const int order1, // first requested Bessel order
      const int order2 // second requested Bessel order
    ) const
  {
    validate_orders(order1, order2);
    if (!ready_) {
      spdlog::critical("3pcf two-Bessel plan: call forward before inverse");
      std::exit(1);
    }
  }

  std::unique_ptr<fftlog_workspace_3pcf,
      decltype(&fftlog_destroy_3pcf)> work_{nullptr, fftlog_destroy_3pcf};
  int nell_ = 0; // immutable physical-grid length
  int max_order_ = 0; // immutable prepared-order capacity
  bool ready_ = false; // coefficients belong to the last completed forward
  transform_array theta_; // copied reversed reciprocal output grid
};



// ============================================================================
// [SECTION] PRODUCTION TRANSFORM PLAN
// ============================================================================

// ---------------------------------------------------------------------------
// transform_owner_3pcf: the Python-visible transform plan. A plan is the
// precomputed geometry of one triangle configuration - the Fourier
// grid, the radial bins the caller picks, the opening angles and the
// angular-coupling table - together with the FFTW scratch that turns
// b_L into natural components on that geometry.
//
// Ownership and lifetime. The constructor builds the C plan, which
// copies every geometry array; this object keeps that plan, its FFTW
// scratch and two theta tables alive until Python destroys the object,
// and the destructor then releases the C plan. The class deletes its
// copy and assignment operators: two owners of one C plan pointer
// would destroy it twice and would share mutable FFT scratch.
//
// Geometry is immutable after construction. A new Fourier grid, radial
// selection, angle grid, cutoff or convention needs a new object; no
// method changes the stored geometry, so nothing a plan computed can go
// out of date behind the caller's back.
//
// theta convention: theta_full_ holds the full reversed reciprocal grid
// theta = xy/ell, and theta_ holds the entries the caller's radial
// indices pick from it. Each theta in that selection is the lower edge of its
// radial bin; a positive dlnt averages over [theta, theta e^dlnt].
// ---------------------------------------------------------------------------
class transform_owner_3pcf {
 public:
  transform_owner_3pcf(
      const transform_array& ell, // [nell], positive logarithmic grid
      const transform_array& phi, // [nphi], opening angles in radians
      const transform_indices& indices, // [ntheta], picks theta nodes
      const int lmax, // Legendre cutoff of b_L
      const int mmax, // angular Fourier cutoff
      const double nu1, // first FFTLog bias (tilt)
      const double nu2, // second FFTLog bias (tilt)
      const double xy, // constant in theta = xy/ell
      const double dlnt, // log radial-bin width; 0 = point values
      const int padding, // zero cells added per grid end
      const bool window, // enable the high-frequency coefficient taper
      const double window_width, // tapered coefficient fraction
      const int npsi, // coupling interpolation length; 0 = exact
      const double dphi, // X-mode opening-angle averaging width
      const std::string& projection, // phase-convention name
      const std::string& convention, // component order: fastnc/treecorr
      const bool ordered_legs // general ordered-bin angular transform
    )
  {
    shape_transform(ell, 1);
    shape_transform(phi, 1);
    shape_transform(indices, 1);
    finite_transform(ell);
    finite_transform(phi);

    if (!(0 <= lmax && INT_MAX-1 > lmax)) {
      spdlog::critical("3pcf transform: invalid Legendre cutoff");
      std::exit(1);
    }
    if (!(convention == "fastnc" || convention == "treecorr")) {
      spdlog::critical("3pcf transform: convention must be fastnc or treecorr");
      std::exit(1);
    }
    // The triangle count ntheta*ntheta*nphi is later passed as an int
    // and used in array shapes; refuse a plan whose count does not fit,
    // so every product of these sizes below is safe. shape_transform
    // keeps each axis between 1 and INT_MAX, so ntheta*ntheta fits in
    // size_t. Comparing it with INT_MAX/nphi never forms the full
    // product, which could wrap around in size_t and pass the check.
    const size_t pair_count = static_cast<size_t>(indices.size())
      *static_cast<size_t>(indices.size());
    if (!(static_cast<size_t>(INT_MAX)/static_cast<size_t>(phi.size())
        >= pair_count)) {
      spdlog::critical("3pcf transform: ntheta = {} and nphi = {} give "
        "more than INT_MAX triangles", indices.size(), phi.size());
      std::exit(1);
    }

    ell_.assign(ell.data(), ell.data()+ell.size());
    nell_ = static_cast<int>(ell.size());
    ntheta_ = static_cast<int>(indices.size());
    nphi_ = static_cast<int>(phi.size());
    lmax_ = lmax;

    plan_ = transform_create_3pcf(nell_, ell.data(), ntheta_,
        indices.data(), nphi_, phi.data(), lmax, mmax,
        worker_count_transform(), nu1, nu2, xy, dlnt, padding, window,
        window_width, npsi, dphi, projection_transform(projection),
        convention == "treecorr", ordered_legs);

    // Build the two theta tables. FFTLog maps the rising logarithmic
    // ell grid onto the reversed reciprocal grid theta = xy/ell, so
    // theta_full_[i] = xy/ell[nell-1-i] rises with i. The second loop
    // keeps the entries the caller's indices pick; transform_create
    // above already rejected any index outside [0, nell) or out of
    // increasing order, so these reads stay inside the table.
    theta_ = transform_array(ntheta_);
    theta_full_ = transform_array(nell_);

    for (int index=0; index<nell_; index++) {
      theta_full_.mutable_data()[index] = xy/ell.data()[nell_-1-index];
    }

    for (int index=0; index<ntheta_; index++) {
      theta_.mutable_data()[index] =
          theta_full_.data()[indices.data()[index]];
    }
  }

  ~transform_owner_3pcf()
  {
    transform_destroy_3pcf(plan_);
  }

  transform_owner_3pcf(const transform_owner_3pcf&) = delete;
  transform_owner_3pcf& operator=(const transform_owner_3pcf&) = delete;

  // Natural components from b_L on this plan's triangle grid. The two
  // returned arrays hold the real and imaginary parts, with the four
  // natural components on the leading axis.
  py::tuple natural(
      const transform_array& multipoles // b_L, [L,nell,nell]
    )
  {
    validate(multipoles);

    transform_array real({4, ntheta_, ntheta_, nphi_});
    transform_array imag({4, ntheta_, ntheta_, nphi_});

    transform_natural_3pcf(plan_, multipoles.data(), real.mutable_data(),
        imag.mutable_data());

    finite_transform(real);
    finite_transform(imag);

    return py::make_tuple(real, imag);
  }

  // Notebook twin of natural: copy arbitrary NumPy input through
  // Armadillo, run the same prepared C plan, and reshape the exported
  // result to the production axes.
  py::tuple natural_notebook(
      const py::object& multipoles // copied b_L, [L,nell,nell]
    )
  {
    const auto input =
        notebook_transform_input<arma::Cube<double>>(multipoles, 3);

    if (!(input.n_rows == static_cast<arma::uword>(lmax_+1)
          && input.n_cols == static_cast<arma::uword>(nell_)
          && input.n_slices == static_cast<arma::uword>(nell_))) {
      spdlog::critical("3pcf transform: notebook multipoles mismatch plan");
      std::exit(1);
    }

    auto output = natural_cpp_3pcf(plan_, input, ntheta_, nphi_);
    py::object real = carma::cube_to_arr(std::move(output.real));
    py::object imag = carma::cube_to_arr(std::move(output.imag));

    // natural_cpp_3pcf returns cubes [4, ntheta*ntheta, nphi] whose
    // pair axis packs (theta1, theta2) as theta1*ntheta + theta2. A
    // NumPy reshape reads logical indices in C order whatever the
    // export's internal storage, so splitting the pair axis into
    // (ntheta, ntheta) reproduces exactly that packing: both adapters
    // expose the same [4, theta, theta, phi] layout to Python callers.
    const auto shape = py::make_tuple(4, ntheta_, ntheta_, nphi_);
    return py::make_tuple(real.attr("reshape")(shape),
        imag.attr("reshape")(shape));
  }

  // Aperture moments from b_L: transform to natural components, convert
  // them to the TreeCorr component convention -[Gamma_0, Gamma_1,
  // Gamma_3, Gamma_2], and contract with the caller's finite-bin
  // aperture weights, all inside the C engine.
  transform_array aperture(
      const transform_array& multipoles, // b_L, [L,nell,nell]
      const transform_array& real, // [filter,4,triangle], real weights
      const transform_array& imag // matching imaginary weights
    )
  {
    validate(multipoles);
    shape_transform(real, 3);
    shape_transform(imag, 3);

    const size_t triangles = static_cast<size_t>(ntheta_)*ntheta_*nphi_;
    if (!(4 == real.shape(1)
          && static_cast<size_t>(real.shape(2)) == triangles)) {
      spdlog::critical("3pcf transform: weights do not match this plan's "
        "triangle grid");
      std::exit(1);
    }
    for (int axis=0; 3 > axis; axis++) {
      if (!(real.shape(axis) == imag.shape(axis))) {
        spdlog::critical("3pcf transform: real and imaginary weight shapes "
          "differ");
        std::exit(1);
      }
    }

    finite_transform(real);
    finite_transform(imag);

    const int nfilter = static_cast<int>(real.shape(0));
    transform_array output(nfilter);
    transform_aperture_3pcf(plan_, multipoles.data(), nfilter, real.data(),
        imag.data(), output.mutable_data());

    finite_transform(output);

    return output;
  }

  transform_array theta() const
  {
    // Return an independent copy, so a Python-side write cannot reach
    // the plan's own table.
    transform_array result(ntheta_);
    std::copy_n(theta_.data(), ntheta_, result.mutable_data());
    return result;
  }

  // Aperture moments refreshed from a physical grid: the C engine reads
  // fresh b_L from the grid on every call and reuses only this plan's
  // query geometry and temporary storage.
  transform_array aperture_grid(
      const py::object& grid, // physical grid, kept alive by Python
      const transform_array& real, // [filter,4,triangle], real weights
      const transform_array& imag // same shape
    )
  {
    shape_transform(real, 3);
    shape_transform(imag, 3);

    const size_t triangles = static_cast<size_t>(ntheta_)*ntheta_*nphi_;
    if (!(4 == real.shape(1)
          && static_cast<size_t>(real.shape(2)) == triangles)) {
      spdlog::critical("3pcf transform: weights do not match this plan's "
        "triangle grid");
      std::exit(1);
    }
    for (int axis=0; 3 > axis; axis++) {
      if (!(real.shape(axis) == imag.shape(axis))) {
        spdlog::critical("3pcf transform: real and imaginary weight shapes "
          "differ");
        std::exit(1);
      }
    }

    finite_transform(real);
    finite_transform(imag);

    const auto* physical = borrow_physics_grid_3pcf(grid);
    const int nfilter = static_cast<int>(real.shape(0));
    transform_array result(nfilter);

    // This call keeps the GIL, like every method here: the plan owns
    // one set of mutable FFT and interpolation scratch, and the lock
    // keeps a second Python thread out of it until the method returns.
    transform_aperture_grid_3pcf(plan_, physical, nfilter, real.data(),
        imag.data(), result.mutable_data());

    finite_transform(result);

    return result;
  }

  // Notebook twin of aperture_grid: copy the weights through Armadillo
  // and reuse the plan's prepared C interpolation.
  py::object aperture_grid_notebook(
      const py::object& grid, // prepared physical grid
      const py::object& real, // real weights [filter,4,triangle]
      const py::object& imag // matching imaginary weights
    )
  {
    const auto real_copy =
        notebook_transform_input<arma::Cube<double>>(real, 3);
    const auto imag_copy =
        notebook_transform_input<arma::Cube<double>>(imag, 3);

    const auto* physical = borrow_physics_grid_3pcf(grid);
    py::object output = carma::col_to_arr(aperture_grid_cpp_3pcf(plan_,
        physical, real_copy, imag_copy, ntheta_*ntheta_*nphi_));

    // Armadillo columns arrive as [nfilter,1]. Both adapters expose the
    // same one-dimensional [nfilter] aperture vector to Python callers.
    return output.attr("reshape")(py::make_tuple(real_copy.n_rows));
  }

  transform_array theta_full() const
  {
    transform_array result(nell_);
    std::copy_n(theta_full_.data(), nell_, result.mutable_data());
    return result;
  }

  // Prepare the finite-SAS aperture measurement on this plan's fixed
  // triangle grid. Each requested data-vector row names an ordered
  // source triplet and one triplet of aperture radii, and this method
  // stores everything aperture_vector later needs: the deduplicated
  // triplet and radius lists, the finite-bin integration weights and
  // the resolved projection settings. Repeated triplets then share
  // one projection and repeated radius triplets share one weight
  // table, however many rows request them.
  //
  // Returns nothing; it only fills this plan's aperture_* members.
  // Refuses row tables whose bin and radius counts disagree, a side
  // mask that drops every radial bin, a representative side outside
  // its bin, nonpositive bin widths and an angular quadrature count
  // nmu <= lmax, which could not resolve every Legendre degree.
  void init_aperture_rows(
    const transform_indices& bins, // [row,3], source-bin indices
    const transform_array& filters, // [row,3], aperture radii in radians
    const transform_array& lower, // original side-bin lower edges
    const transform_array& phi, // [nphi], opening-angle representatives
    const double dlnt, // logarithmic side-bin width
    const double dphi, // opening-angle integration width
    const py::object& side_keep, // optional Boolean mask on side bins
    const std::string& representative_rule, // geometric or area_weighted
    const double representative_offset, // fraction of the lower side edge
    const py::dict& settings, // resolved projection controls
    const int batch_size, // maximum simultaneous source projection grids
    const bool prepared, // false selects direct reference quadrature
    const int nmu // angular quadrature count for the direct reference
  )
  {
    shape_transform(bins, 2);
    shape_transform(filters, 2);
    finite_transform(filters);

    // One guard covers the row tables and the quadrature controls:
    // both tables are [nrow, 3] and equally long, a batch holds at
    // least one grid, the direct reference quadrature resolves every
    // Legendre degree, and the bin widths and the representative
    // offset are finite, the widths also positive.
    if (bins.shape(1) != 3 || filters.shape(1) != 3
        || bins.shape(0) != filters.shape(0) || batch_size < 1
        || nmu <= lmax_ || !std::isfinite(dlnt) || dlnt <= 0.0
        || !std::isfinite(dphi) || dphi <= 0.0
        || !std::isfinite(representative_offset)) {
      throw py::value_error("Invalid aperture rows or angular quadrature.");
    }

    // Optional side mask. side_keep marks the radial bins that enter
    // the aperture integration; a dropped bin keeps its place in the
    // triangle grid, but its weights are zeroed below, so the final
    // contraction skips it. The default keeps every bin.
    std::vector<int> keep(ntheta_, 1);
    if (!side_keep.is_none()) {
      auto raw = py::array::ensure(side_keep);
      if (!raw || !raw.dtype().is(py::dtype::of<bool>())) {
        throw py::value_error("side_keep must be a Boolean array.");
      }
      auto supplied = py::array_t<bool, py::array::c_style>::ensure(raw);
      if (!supplied || supplied.ndim() != 1 || supplied.size() != ntheta_) {
        throw py::value_error("side_keep must match the radial-bin count.");
      }
      for (int i=0; i<ntheta_; i++) {
        keep[i] = supplied.data()[i];
      }
      if (std::count(keep.begin(), keep.end(), 1) == 0) {
        throw py::value_error("side_keep must retain at least one radial bin.");
      }
    }

    // Representative side of each logarithmic bin [lower, lower
    // e^dlnt], as a factor on the lower edge. The geometric rule
    // takes the log midpoint, lower e^(dlnt/2). The area-weighted
    // rule averages the side theta over the bin with the pair-count
    // measure theta dtheta,
    //
    //   <theta> = lower (2/3) (e^(3 dlnt) - 1)/(e^(2 dlnt) - 1),
    //
    // the mean side of uniformly distributed pairs in that bin. The
    // offset then displaces the representative by a fraction of the
    // lower edge, and the guard keeps the result inside its bin up
    // to rounding slack (the 1e-14 terms).
    double factor = std::exp(0.5*dlnt);
    if (representative_rule == "area_weighted") {
      factor = (2.0/3.0)*std::expm1(3.0*dlnt)/std::expm1(2.0*dlnt);
    } else if (representative_rule != "geometric") {
      throw py::value_error("Unknown side representative rule.");
    }
    factor += representative_offset;
    if (factor < 1.0-1.e-14 || factor > std::exp(dlnt)*(1.0+1.e-14)) {
      throw py::value_error("Representative side lies outside its bin.");
    }

    std::vector<double> sides(ntheta_);
    for (int i=0; i<ntheta_; i++) {
      sides[i] = lower.data()[i]*factor;
    }
    // Deduplicate the requested rows. For each row, aperture_triplet_
    // points at its unique source triplet and aperture_filter_ at its
    // unique radius triplet, so aperture_vector later projects each
    // distinct triplet once and contracts each distinct radius set
    // once, whatever the number of rows that reuse them.
    // The row tables are built in locals and installed only after every
    // row passed its checks, so a refused request leaves the plan as it
    // was. The settings dictionary is converted first for the same
    // reason: a malformed one throws here, before anything changes.
    const auto resolved_settings = settings_physics_grid(settings);
    const py::dict settings_record = py::dict(settings.attr("copy")());
    std::vector<std::array<int,3>> triplets;
    std::vector<std::array<double,3>> radii;
    std::vector<int> row_triplet;
    std::vector<int> row_filter;
    for (int row=0; row<bins.shape(0); row++) {
      std::array<int,3> triplet;
      std::array<double,3> radius;
      for (int leg=0; leg<3; leg++) {
        triplet[leg] = bins.data()[3*row+leg];
        radius[leg] = filters.data()[3*row+leg];
        if (triplet[leg] < 0 || radius[leg] <= 0.0) {
          throw py::value_error("Aperture bins/radii must be nonnegative/positive.");
        }
      }
      auto t = std::find(triplets.begin(), triplets.end(), triplet);
      row_triplet.push_back(t-triplets.begin());
      if (t == triplets.end()) {
        triplets.push_back(triplet);
      }
      auto r = std::find(radii.begin(), radii.end(), radius);
      row_filter.push_back(r-radii.begin());
      if (r == radii.end()) {
        radii.push_back(radius);
      }
    }
    aperture_bins_.assign(bins.data(), bins.data()+bins.size());
    aperture_triplet_ = row_triplet;
    aperture_filter_ = row_filter;
    // Flatten the unique lists into the [count, 3] buffers the C
    // engines read.
    aperture_triplets_.clear();
    for (const auto& triplet : triplets) {
      aperture_triplets_.insert(aperture_triplets_.end(),
        triplet.begin(), triplet.end());
    }
    std::vector<double> flat_filters;
    for (const auto& radius : radii) {
      flat_filters.insert(flat_filters.end(), radius.begin(), radius.end());
    }

    // Build one finite-bin weight table per unique radius triplet.
    // aperture_weights_3pcf integrates the compensated aperture
    // filters over every SAS bin, so the later step that turns
    // natural components into aperture moments is a plain weighted
    // sum over triangles.
    const size_t triangles = static_cast<size_t>(ntheta_)*ntheta_*nphi_;
    aperture_nfilter_ = radii.size();
    aperture_real_.resize(radii.size()*4*triangles);
    aperture_imag_.resize(aperture_real_.size());
    aperture_weights_3pcf(ntheta_, ntheta_, nphi_, aperture_nfilter_,
      sides.data(), sides.data(), phi.data(), dlnt, dphi,
      flat_filters.data(), aperture_real_.data(), aperture_imag_.data());

    // Zero the weights of every triangle that touches a dropped side
    // bin. The weight table carries the mask, not the triangle grid,
    // so the transform geometry stays untouched.
    for (size_t f=0; f<radii.size()*4; f++) {
      for (int i=0; i<ntheta_; i++) {
        for (int j=0; j<ntheta_; j++) {
          if (!keep[i] || !keep[j]) {
            const size_t offset = f*triangles+(i*ntheta_+j)*nphi_;
            std::fill_n(aperture_real_.data()+offset, nphi_, 0.0);
            std::fill_n(aperture_imag_.data()+offset, nphi_, 0.0);
          }
        }
      }
    }

    // Record the resolved controls for aperture_vector and for the
    // read-only diagnostic properties.
    aperture_keep_ = keep;
    aperture_sides_ = sides;
    aperture_settings_ = resolved_settings;
    aperture_settings_record_ = settings_record;
    aperture_batch_ = batch_size;
    aperture_prepared_ = prepared;
    aperture_nmu_ = nmu;
  }

  // Evaluate the aperture data vector for one physical model:
  // project each unique ordered source triplet once, transform it to
  // natural components, contract it with each unique weight table
  // and assign the values back to their original rows. Physical
  // grids are rebuilt on every call, so no cosmology-dependent table
  // survives between likelihood evaluations; only geometry and FFT
  // scratch persist in this plan.
  //
  // component selects the physical content: 0 the total prediction,
  // 1 the NLA (nonlinear linear alignment) baseline alone, 2 the
  // intrinsic-alignment residual beyond that baseline.
  //
  // Returns one aperture moment per initialized row, scaled by the
  // optional shear-calibration factor of that row's sources. Refuses
  // to run before init_aperture_rows and refuses a calibration
  // vector that omits a requested source bin.
  transform_array aperture_vector(
    const py::object& model, // immutable physical model for this evaluation
    const py::dict& ia, // IA and feedback parameters in CosmoLike convention
    const py::object& calibration, // optional source shear offsets
    const int component // total=0, NLA baseline=1, IA residual=2
  )
  {
    if (aperture_filter_.empty()) {
      throw py::value_error("Initialize aperture rows before evaluating them.");
    }
    const auto* physical = borrow_physics_model_3pcf(model);
    const auto parameters = parameters_physics(ia);
    const int nrow = aperture_filter_.size();

    // Optional multiplicative shear calibration. Each leg of a row's
    // source triplet contributes a factor 1 + m_i, with m_i the
    // offset of source bin i, so the row's aperture moment scales by
    // the product over its three legs.
    std::vector<double> calibration_factors(nrow, 1.0);
    if (!calibration.is_none()) {
      auto shear = transform_array::ensure(calibration);
      if (!shear || shear.ndim() != 1) {
        throw py::value_error("Calibration must contain one offset per source.");
      }
      finite_transform(shear);
      for (int row=0; row<nrow; row++) {
        double factor = 1.0;
        for (int leg=0; leg<3; leg++) {
          const int source = aperture_bins_[3*row+leg];
          if (source >= shear.size()) {
            throw py::value_error("Calibration omits an aperture source bin.");
          }
          factor *= 1.0+shear.data()[source];
        }
        calibration_factors[row] = factor;
      }
    }
    transform_array result(nrow);
    const int ntriplet = aperture_triplets_.size()/3;
    std::vector<double> values(aperture_nfilter_);

    // The explicit b_L buffer serves only the routes that read
    // multipoles - the direct reference and the component
    // selections; the prepared total contracts straight from the
    // grid and never fills it.
    std::vector<double> multipoles;
    if (!aperture_prepared_ || component != 0) {
      multipoles.resize(static_cast<size_t>(lmax_+1)*nell_*nell_);
    }

    // Walk the unique triplets in batches. One batch prepares at
    // most aperture_batch_ projection grids at a time, which bounds
    // the memory the prepared route holds while the grid
    // construction still amortizes its setup across sources.
    for (int start=0; start<ntriplet; start+=aperture_batch_) {
      const int count = std::min(aperture_batch_, ntriplet-start);
      std::vector<physics_grid_3pcf*> grids(count, nullptr);
      if (aperture_prepared_) {
        physics_grid_create_many_3pcf(physical, &parameters, count,
          aperture_triplets_.data()+3*start, &aperture_settings_, grids.data());
      }
      // Three routes lead to the aperture values of one triplet: the
      // prepared total contracts its grid directly; a prepared
      // component first exports the selected multipoles from that
      // grid; the direct reference evaluates the multipoles by
      // quadrature with aperture_nmu_ angular nodes. All three end
      // in the same contraction against the finite-bin weights.
      for (int i=0; i<count; i++) {
        const int triplet = start+i;
        if (aperture_prepared_ && component == 0) {
          transform_aperture_grid_3pcf(plan_, grids[i], aperture_nfilter_,
            aperture_real_.data(), aperture_imag_.data(), values.data());
        } else {
          if (aperture_prepared_) {
            physics_grid_multipoles_3pcf(grids[i], nell_, ell_.data(),
              component, multipoles.data());
          } else {
            physics_multipoles_3pcf(physical, &parameters,
              aperture_triplets_.data()+3*triplet, nell_, ell_.data(),
              lmax_, aperture_nmu_, component, multipoles.data());
          }
          transform_aperture_3pcf(plan_, multipoles.data(), aperture_nfilter_,
            aperture_real_.data(), aperture_imag_.data(), values.data());
        }
        // Copy this triplet's values into every row that requested
        // it, each row picking its own radius triplet.
        for (int row=0; row<nrow; row++) {
          if (aperture_triplet_[row] == triplet) {
            result.mutable_data()[row] = values[aperture_filter_[row]];
          }
        }
        physics_grid_destroy_3pcf(grids[i]);
      }
    }
    for (int row=0; row<nrow; row++) {
      result.mutable_data()[row] *= calibration_factors[row];
    }
    finite_transform(result);
    return result;
  }

  // Decompose the aperture prediction along two axes at once: the
  // physical operator and the placement of the intrinsic shapes. A
  // placement label such as GGI reads leg by leg - G marks a
  // gravitationally lensed shear, I an intrinsic shape - so GGI puts
  // the intrinsic shape on the third leg and IIG on the first two.
  // Each (operator, placement) term is evaluated directly, by
  // selecting legs inside the C engine, because subtracting two
  // near-equal totals would lose the small intrinsic-alignment terms
  // to cancellation.
  //
  // Returns a dict: "operators" and "placements" name the first two
  // axes of "values" [5, 8, nrow]; "total" is the full prediction
  // and "closure" the sum of all terms minus that total, a
  // cross-check that must vanish to rounding. Refuses a preselected
  // operator or leg placement in ia and, on the prepared route, an
  // unsigned baseline, because the decomposition must control both
  // choices itself.
  py::dict aperture_components(
    const py::object& model, // same physical input as the total prediction
    const py::dict& ia, // full IA choice, without a preselected operator
    const py::object& calibration // optional source shear offsets
  )
  {
    const auto parameters = parameters_physics(ia);
    if (parameters.operator_component != 0 || parameters.intrinsic_legs != -1
        || (aperture_prepared_ && aperture_settings_.signed_baseline != 1)) {
      throw py::value_error("Components require full IA and signed_baseline=1.");
    }
    const auto total = aperture_vector(model, ia, calibration, 0);
    const int nrow = total.size();
    transform_array values({5, 8, nrow});
    std::fill_n(values.mutable_data(), values.size(), 0.0);

    // intrinsic_legs is a bitmask: bit 0 marks an intrinsic shape on
    // leg 1, bit 1 on leg 2 and bit 2 on leg 3. masks lists the
    // eight placements in the display order GGG, GGI, GIG, IGG,
    // GII, IIG, IGI, III - GGI, intrinsic only on leg 3, is bit 2
    // alone, the mask value 4.
    const int masks[8] = {0, 4, 2, 1, 6, 3, 5, 7};

    // Evaluate each nonzero term. The placement GGG carries no
    // intrinsic leg, so only the gravity row op = 0 fills it; the
    // operator code max(1, op) = 1 it passes is inert when the leg
    // mask is zero. Every alignment operator op >= 1 pairs with the
    // seven placements that hold at least one intrinsic leg, and
    // the skipped slots of values stay zero.
    for (int op=0; op<5; op++) {
      for (int place=0; place<8; place++) {
        if ((op == 0 && place != 0) || (op != 0 && place == 0)) {
          continue;
        }
        py::dict selected(ia.attr("copy")());
        selected["operator_component"] = std::max(1, op);
        selected["intrinsic_legs"] = masks[place];
        const auto part = aperture_vector(model, selected, calibration, 0);
        std::copy_n(part.data(), nrow,
          values.mutable_data()+(op*8+place)*nrow);
      }
    }

    // Closure check: the 5 x 8 = 40 slots, zeros included, must
    // reassemble the total prediction, so their sum minus the total
    // exposes any term the decomposition lost or double-counted.
    transform_array closure(nrow);
    for (int row=0; row<nrow; row++) {
      double sum = 0.0;
      for (int term=0; term<40; term++) {
        sum += values.data()[term*nrow+row];
      }
      closure.mutable_data()[row] = sum-total.data()[row];
    }

    // Name the axes, so a notebook reads the decomposition without
    // this file at hand.
    py::dict result;
    result["operators"] = py::make_tuple("gravity", "linear_alignment",
      "density_weighting", "tidal_torque", "velocity_shear");
    result["placements"] = py::make_tuple("GGG", "GGI", "GIG", "IGG",
      "GII", "IIG", "IGI", "III");
    result["values"] = values;
    result["total"] = total;
    result["closure"] = closure;
    return result;
  }

  // Independent copy of the plan's logarithmic Fourier grid.
  transform_array ell() const
  {
    transform_array result(nell_);
    std::copy(ell_.begin(), ell_.end(), result.mutable_data());
    return result;
  }

  // Diagnostic arrays are copies: notebook edits cannot change retained FFT
  // geometry or the triangle selection used by later likelihood calls.
  transform_array weights_real() const
  {
    transform_array result({aperture_nfilter_, 4, ntheta_*ntheta_*nphi_});
    std::copy(aperture_real_.begin(), aperture_real_.end(), result.mutable_data());
    return result;
  }

  transform_array weights_imag() const
  {
    transform_array result({aperture_nfilter_, 4, ntheta_*ntheta_*nphi_});
    std::copy(aperture_imag_.begin(), aperture_imag_.end(), result.mutable_data());
    return result;
  }

  // Mask of radial side bins the aperture weights retain (true =
  // the bin enters the integration).
  py::array_t<bool> side_keep() const
  {
    py::array_t<bool> result(aperture_keep_.size());
    for (size_t i=0; i<aperture_keep_.size(); i++) {
      result.mutable_data()[i] = aperture_keep_[i];
    }
    return result;
  }

  // Representative side length of each radial bin, in radians, after
  // the rule and offset init_aperture_rows applied.
  transform_array representative_sides() const
  {
    transform_array result(aperture_sides_.size());
    std::copy(aperture_sides_.begin(), aperture_sides_.end(), result.mutable_data());
    return result;
  }

  // Copy of the resolved projection settings, so a notebook can
  // record the provenance of a data vector.
  py::dict projection_settings() const
  {
    return py::dict(aperture_settings_record_.attr("copy")());
  }

 private:
  // Shared input contract of the production methods: b_L must match
  // this plan's [lmax+1, nell, nell] shape, aligned and finite.
  void validate(
      const transform_array& multipoles // borrowed real coefficients
    ) const
  {
    shape_transform(multipoles, 3);
    if (!(multipoles.shape(0) == lmax_+1
          && multipoles.shape(1) == nell_
          && multipoles.shape(2) == nell_)) {
      spdlog::critical("3pcf transform: multipoles mismatch plan shape");
      std::exit(1);
    }
    finite_transform(multipoles);
  }

  std::vector<double> ell_; // owned physical Fourier coordinates
  std::vector<int> aperture_keep_; // 1 = side bin enters the weights
  std::vector<double> aperture_sides_; // representative sides, radians
  // aperture_bins_: source triplet of every row, flattened [row, 3];
  // aperture_triplets_: the unique ordered triplets, flattened
  std::vector<int> aperture_bins_, aperture_triplets_;
  // per row: index of its unique triplet and of its unique radius set
  std::vector<int> aperture_triplet_, aperture_filter_;
  // finite-bin weights [nfilter, 4, triangle], real and imaginary
  std::vector<double> aperture_real_, aperture_imag_;
  physics_grid_settings_3pcf aperture_settings_; // resolved C controls
  py::dict aperture_settings_record_; // Python copy for diagnostics
  // unique radius triplets; grids per batch; direct quadrature count
  int aperture_nfilter_=0, aperture_batch_=1, aperture_nmu_=0;
  bool aperture_prepared_=true; // false = direct reference quadrature
  transform_plan_3pcf* plan_ = nullptr; // owned C plan and FFT scratch
  int nell_ = 0; // Fourier grid length
  int ntheta_ = 0; // radial bins the indices pick
  int nphi_ = 0; // opening-angle samples
  int lmax_ = 0; // highest Legendre degree in b_L
  transform_array theta_; // lower edges of the kept radial bins, radians
  transform_array theta_full_; // full reversed grid xy/ell, radians
};



// ---------------------------------------------------------------------------
// Build a logarithmic Fourier grid whose reversed reciprocal grid
// theta = 1/ell contains every requested lower side edge exactly.
// FFTLog measures angles only on that reciprocal grid, so the FFT
// step must divide the side-bin width dlnt: the step is dlnt over an
// integer subdivision count, and refining the grid or extending the
// Fourier range then moves no requested measurement angle.
//
// Returns the pair (ell, indices): the grid [length] and, for each
// side edge, the index of its node in the reversed grid. Refuses
// nonpositive or disordered bounds, side edges that are not equally
// log-spaced at width dlnt, and a grid beyond 65536 nodes - a cap
// that keeps the quadratic [nell, nell] transform stages affordable.
// ---------------------------------------------------------------------------
static py::tuple aligned_fft_grid_3pcf(
  const transform_array& theta, // lower side edges in radians
  const double dlnt, // common logarithmic bin width
  const double ell_min, // required smallest Fourier multipole
  const double ell_max, // required largest Fourier multipole
  const int minimum_size // minimum even FFT length
)
{
  shape_transform(theta, 1);
  finite_transform(theta);
  if (!std::isfinite(dlnt) || dlnt <= 0.0
      || !std::isfinite(ell_min) || ell_min <= 0.0
      || !std::isfinite(ell_max) || ell_max <= ell_min || minimum_size < 4) {
    throw py::value_error("Invalid angular/Fourier alignment bounds.");
  }

  // The side edges must march in steps of exactly dlnt in the log;
  // the tolerance only absorbs the rounding of the caller's own
  // exp/log arithmetic.
  for (int i=0; i<theta.size(); i++) {
    if (theta.data()[i] <= 0.0 || (i > 0 &&
        std::abs(std::log(theta.data()[i])-std::log(theta.data()[i-1])-dlnt)
          > 1.e-13+1.e-11*std::abs(dlnt))) {
      throw py::value_error("Theta edges must be equally log-spaced.");
    }
  }

  // Choose the FFT step as dlnt/subdivisions, with enough integer
  // subdivisions that at least minimum_size nodes span the requested
  // Fourier range; the step both divides the side-bin width and
  // meets the resolution floor.
  const double subdivisions = std::max(1.0,
    std::ceil(minimum_size*dlnt/std::log(ell_max/ell_min)));
  const double step = dlnt/subdivisions;

  // Pin the grid to the first side edge: "first" is the
  // reversed-grid index where theta[0] = 1/ell sits, chosen as the
  // smallest index for which the grid top exp(log_maximum) still
  // reaches ell_max, so theta[0] lands exactly on a node without
  // wasted range.
  const double first = std::max(0.0,
    std::ceil(std::log(theta.data()[0]*ell_max)/step));
  const double log_maximum = -std::log(theta.data()[0])+first*step;

  // The grid must hold at least minimum_size nodes, reach down to
  // ell_min, and still contain the last side edge, which sits
  // "subdivisions" nodes after each previous one.
  const double required = std::max({double(minimum_size),
    std::ceil((log_maximum-std::log(ell_min))/step)+1.0,
    first+(theta.size()-1)*subdivisions+1.0});
  if (!std::isfinite(required) || required > 65536.0) {
    throw py::value_error("FFT length exceeds 65536; reduce its density.");
  }
  const int length = next_even_fft_size(static_cast<int>(required));
  if (length > 65536) {
    throw py::value_error("FFT length exceeds 65536; reduce its density.");
  }

  // Fill the grid downward from its pinned top and record where each
  // side edge sits; the final comparison catches any accumulated
  // rounding beyond two parts in 1e12.
  transform_array ell(length);
  transform_indices indices(theta.size());
  for (int i=0; i<length; i++) {
    ell.mutable_data()[i] = std::exp(log_maximum-step*(length-1-i));
  }
  for (int i=0; i<theta.size(); i++) {
    const int index = first+subdivisions*i;
    indices.mutable_data()[i] = index;
    if (std::abs(1.0/ell.data()[length-1-index]-theta.data()[i])
        > 2.e-12*theta.data()[i]) {
      throw py::value_error("FFT alignment lost angular precision.");
    }
  }
  finite_transform(ell);
  return py::make_tuple(ell, indices);
}

// ---------------------------------------------------------------------------
// Build the complete finite-SAS aperture plan behind
// TransformPlan.for_aperture: resolve the accuracy multipliers,
// align the FFT grid with the side bins, fill and cross-check the
// projection settings, parse the transform options, and return a
// plan whose aperture rows are initialized, ready for
// aperture_vector.
//
// Returns the owning plan object. Refuses projection settings that
// fail to cover the FFT multipole range or disagree with lmax, an
// opening-angle grid that is not uniform inside (0, pi), an unknown
// projection mode, and any transform option this route does not
// support.
// ---------------------------------------------------------------------------
static std::unique_ptr<transform_owner_3pcf> aperture_plan_3pcf(
  const transform_indices& bins, // [row,3], ordered source bins
  const transform_array& filters, // [row,3], aperture radii in radians
  const transform_array& theta_lower, // lower side edges in radians
  const transform_array& phi, // opening-angle bin representatives
  double dlnt, // logarithmic side-bin width
  double dphi, // angular integration width
  double ell_min, // required Fourier lower endpoint
  double ell_max, // required Fourier upper endpoint
  int minimum_size, // baseline minimum FFT length
  int lmax, // largest Legendre degree
  int mmax, // largest Fourier mode in the opening angle
  int nmu, // baseline angular quadrature count
  const py::dict& transform_options, // existing transform constructor controls
  const std::string& projection_mode, // prepared or direct reference
  const py::dict& projection_settings, // baseline physical interpolation controls
  int projection_batch_size, // maximum concurrent source grids
  const py::object& accuracy, // global and individual refinement multipliers
  const py::object& side_keep, // optional Boolean mask on side bins
  const std::string& representative_rule, // geometric or area_weighted
  double representative_offset // displacement as fraction of a lower bin edge
)
{
  // Resolve the accuracy multipliers first - fft_boost refines the
  // minimum FFT length and angular_boost the reference quadrature
  // count - so every later grid derives from the boosted values.
  const auto controls = init_accuracy_3pcf(accuracy);
  minimum_size = accuracy_count_3pcf(controls, minimum_size,
    "fft_boost", false, false);
  nmu = accuracy_count_3pcf(controls, nmu, "angular_boost", true, false);

  const auto aligned = aligned_fft_grid_3pcf(theta_lower, dlnt,
    ell_min, ell_max, minimum_size);
  const auto ell = aligned[0].cast<transform_array>();
  const auto indices = aligned[1].cast<transform_indices>();

  // Fill the projection defaults from the aligned grid itself, then
  // insist on consistency: the projection must tabulate at least the
  // FFT's multipole range, at the same Legendre cutoff and with
  // multipole preparation on, or the prepared route would read
  // outside its table.
  auto settings = accuracy_projection_3pcf(controls, projection_settings);
  if (!settings.contains("ell_min")) {
    settings["ell_min"] = ell.data()[0];
  }
  if (!settings.contains("ell_max")) {
    settings["ell_max"] = ell.data()[ell.size()-1];
  }
  if (!settings.contains("lmax")) {
    settings["lmax"] = lmax;
  }
  if (!settings.contains("prepare_multipoles")) {
    settings["prepare_multipoles"] = 1;
  }
  if (py::cast<int>(settings["lmax"]) != lmax
      || py::cast<int>(settings["prepare_multipoles"]) != 1
      || py::cast<double>(settings["ell_min"]) > ell.data()[0]
      || py::cast<double>(settings["ell_max"]) < ell.data()[ell.size()-1]
      || (projection_mode != "prepared" && projection_mode != "direct")) {
    throw py::value_error("Finite-SAS projection must cover its FFT multipoles.");
  }

  // The opening-angle representatives must march uniformly at width
  // dphi, strictly inside (0, pi), the open interval where a SAS
  // triangle stays nondegenerate.
  shape_transform(phi, 1);
  finite_transform(phi);
  for (int i=0; i<phi.size(); i++) {
    if (phi.data()[i] <= 0.0 || phi.data()[i] >= M_PI
        || (i > 0 && std::abs(phi.data()[i]-phi.data()[i-1]-dphi)
          > 1.e-12*std::abs(dphi))) {
      throw py::value_error("Require uniformly spaced 0<phi<pi.");
    }
  }
  // Match the existing TransformPlan defaults. These FFT biases and
  // taper controls are numerical choices, independent of the
  // physical model. The two silently accepted options are the values
  // this route fixes anyway: the projection "centroid" and xy = 1,
  // the reciprocal grid the alignment above assumed.
  double nu1=1.01, nu2=1.01, window_width=0.25, averaging=0.0;
  int padding=0, npsi=200;
  bool window=false, ordered=true;
  std::string convention="fastnc";
  for (const auto& item : transform_options) {
    const std::string key = py::cast<std::string>(item.first);
    if (key == "nu1") {
      nu1 = py::cast<double>(item.second);
    }
    else if (key == "nu2") {
      nu2 = py::cast<double>(item.second);
    }
    else if (key == "window_width") {
      window_width = py::cast<double>(item.second);
    }
    else if (key == "dphi") {
      averaging = py::cast<double>(item.second);
    }
    else if (key == "padding") {
      padding = py::cast<int>(item.second);
    }
    else if (key == "npsi") {
      npsi = py::cast<int>(item.second);
    }
    else if (key == "window") {
      window = py::cast<bool>(item.second);
    }
    else if (key == "ordered_legs") {
      ordered = py::cast<bool>(item.second);
    }
    else if (key == "convention") {
      convention = py::cast<std::string>(item.second);
    }
    else if (key == "projection" && py::cast<std::string>(item.second) == "centroid") {}
    else if (key == "xy" && py::cast<double>(item.second) == 1.0) {}
    else {
      throw py::value_error("Unsupported aperture transform option: "+key);
    }
  }
  // Construct the plan on the aligned geometry, then prepare its
  // aperture rows: the result is ready for aperture_vector.
  auto plan = std::make_unique<transform_owner_3pcf>(ell, phi, indices,
    lmax, mmax, nu1, nu2, 1.0, dlnt, padding, window, window_width,
    npsi, averaging, "centroid", convention, ordered);
  plan->init_aperture_rows(bins, filters, theta_lower, phi, dlnt, dphi, side_keep,
    representative_rule, representative_offset, settings,
    projection_batch_size, projection_mode == "prepared", nmu);
  return plan;
}

// ============================================================================
// [SECTION] NOTEBOOK REGISTRATION
// ============================================================================

// Register the notebook twins on the notebook submodule that
// bind_threepcf created. Each def wraps its production sibling behind
// an Armadillo copy, so arbitrary NumPy-compatible input works there.
static void bind_transform_notebook_3pcf(
    py::module_& module // owning extension
  )
{
  py::module_ notebook = module.attr("notebook").cast<py::module_>();

  notebook.def("legendre_coupling", [](
      const py::object& psi, // opening angles in radians
      const int lmax, // Legendre cutoff
      const int mmax, // angular Fourier cutoff
      const int npsi // interpolation length
    ) {
    const auto input =
        notebook_transform_input<arma::Col<double>>(psi, 1);
    return carma::cube_to_arr(coupling_cpp_3pcf(input, lmax, mmax, npsi));
  }, "Notebook twin of production.legendre_coupling: copy the opening "
      "angles psi through Armadillo, then evaluate the same C coupling "
      "table G[M, L, psi].",
      py::arg("psi"), py::arg("lmax"), py::arg("mmax"),
      py::arg("npsi")=200);

  notebook.def("two_bessel", [](
      const py::object& ell, // logarithmic Fourier grid
      const py::object& integrand, // dln ell integrand
      const int order1, // first Bessel order
      const int order2, // second Bessel order
      const double nu1, // first FFTLog bias
      const double nu2, // second FFTLog bias
      const double xy, // constant in theta = xy/ell
      const double dlnt, // log radial-bin width
      const int padding, // zero padding per end
      const bool window, // enable the coefficient taper
      const double window_width // tapered fraction
    ) {
    const auto ell_copy =
        notebook_transform_input<arma::Col<double>>(ell, 1);
    const auto values =
        notebook_transform_input<arma::Mat<double>>(integrand, 2);
    return carma::mat_to_arr(bessel_cpp_3pcf(ell_copy, values, order1,
        order2, nu1, nu2, xy, dlnt, padding, window, window_width));
  }, "Notebook twin of production.two_bessel: copy the grid and the "
      "integrand through Armadillo, then run the same C double-Bessel "
      "transform.",
      py::arg("ell"), py::arg("integrand"), py::arg("order1"),
      py::arg("order2"), py::arg("nu1")=1.01, py::arg("nu2")=1.01,
      py::arg("xy")=1.0, py::arg("dlnt")=0.0, py::arg("padding")=0,
      py::arg("window")=false, py::arg("window_width")=0.25);

  notebook.def("two_bessel_plan", &bessel_owner_3pcf::notebook,
      "Create a retained two-Bessel plan after copying ell through "
      "Armadillo. Use the notebook.two_bessel_forward, "
      "two_bessel_inverse and two_bessel_evaluate adapters for copied "
      "integrands and independently owned outputs.",
      py::arg("ell"), py::arg("max_order"), py::arg("nu1")=1.01,
      py::arg("nu2")=1.01, py::arg("xy")=1.0, py::arg("dlnt")=0.0,
      py::arg("padding")=0, py::arg("window")=false,
      py::arg("window_width")=0.25);

  notebook.def("two_bessel_forward", [](
      bessel_owner_3pcf& plan, // retained geometry and coefficients
      const py::object& integrand // copied notebook input
    ) {
    plan.forward_notebook(integrand);
  },
      "Copy a new [nell,nell] integrand and replace the retained "
      "forward coefficients. Later inverse calls use this input.",
      py::arg("plan"), py::arg("integrand"));

  notebook.def("two_bessel_inverse", [](
      bessel_owner_3pcf& plan, // owner with prepared coefficients
      const int order1, // first Bessel order
      const int order2 // second Bessel order
    ) {
    return plan.inverse_notebook(order1, order2);
  },
      "Transform the latest forward coefficients for an order pair. "
      "Return an independent [nell,nell] array through Armadillo; "
      "forward must have run first.",
      py::arg("plan"), py::arg("order1"), py::arg("order2"));

  notebook.def("two_bessel_evaluate", [](
      bessel_owner_3pcf& plan, // retained geometry and scratch
      const py::object& integrand, // complete replacement input
      const int order1, // first Bessel order
      const int order2 // second Bessel order
    ) {
    return plan.evaluate_notebook(integrand, order1, order2);
  },
      "Copy a new integrand, recompute its forward coefficients and "
      "return one independent double-Bessel transform.",
      py::arg("plan"), py::arg("integrand"), py::arg("order1"),
      py::arg("order2"));

  notebook.def("natural_components", [](
      transform_owner_3pcf& plan, // already prepared geometry
      const py::object& multipoles // copied, arbitrary-stride b_L
    ) {
    return plan.natural_notebook(multipoles);
  }, "Notebook twin of TransformPlan.natural_components: copy the "
      "bispectrum Legendre coefficients b_l through Armadillo, then "
      "run the same prepared C plan.",
      py::arg("plan"), py::arg("b_l"));

  notebook.def("aperture_from_grid", [](
      transform_owner_3pcf& plan, // already prepared geometry
      const py::object& grid, // prepared physical grid
      const py::object& real, // real weights [filter,4,triangle]
      const py::object& imag // matching imaginary weights
    ) {
    return plan.aperture_grid_notebook(grid, real, imag);
  }, "Notebook twin of TransformPlan.aperture_from_grid: copy the "
      "weights through Armadillo, then reuse the plan's prepared C "
      "interpolation map.",
      py::arg("plan"), py::arg("grid"), py::arg("weights_real"),
      py::arg("weights_imag"));
}



// ============================================================================
// [SECTION] PRODUCTION ENTRY POINTS
// ============================================================================

// ---------------------------------------------------------------------------
// Angular coupling table G_ML(psi). The C routine evaluates
//
//   G_ML(psi) = 4 pi integral_0^pi P_L(cos delta)
//               cos(2 beta_bar + M delta) d delta,
//
// where P_L is the Legendre polynomial of degree L, M is the angular
// Fourier mode, delta is the integration angle and beta_bar is the
// triangle phase that the coupling geometry fixes (coupling_3pcf.c
// derives it). The query angle psi lies in [0, pi/2]. npsi = 0
// evaluates the exact finite Fourier sum at every query; npsi >= 2
// interpolates a uniform table of that length, matching fastnc's
// tabulation convention.
//
// The cutoff guard also keeps lmax + mmax within int range, because
// the coupling's Fourier sum reaches the combined harmonic M + L.
//
// Returns:
//   NumPy array [mmax+1, lmax+1, nquery] of coupling values.
// ---------------------------------------------------------------------------
static transform_array coupling_production(
    const transform_array& psi, // [nquery], opening angles in radians
    const int lmax, // largest Legendre degree
    const int mmax, // largest angular Fourier mode
    const int npsi // optional interpolation grid
  )
{
  shape_transform(psi, 1);
  finite_transform(psi);

  if (!(0 <= lmax && INT_MAX-1 > lmax
        && 0 <= mmax && INT_MAX-1 > mmax
        && lmax <= INT_MAX-1-mmax)) {
    spdlog::critical("3pcf transform: invalid coupling cutoffs");
    std::exit(1);
  }

  const int nquery = static_cast<int>(psi.size());
  transform_array output({mmax+1, lmax+1, nquery});

  legendre_coupling_3pcf(lmax, mmax, nquery, psi.data(), npsi,
      output.mutable_data());

  finite_transform(output);

  return output;
}


// ---------------------------------------------------------------------------
// Double-Bessel transform of a tabulated integrand:
//
//   g(theta1, theta2) = integral dln(ell1) dln(ell2) f(ell1, ell2)
//                       J_order1(ell1 theta1) J_order2(ell2 theta2),
//
// with J_n the cylindrical Bessel function of the first kind. FFTLog
// evaluates it: the bias nu divides a power-law tilt out of the
// integrand so the logarithmic-grid integral becomes a convolution,
// which the FFT turns into a product of coefficients; xy places the
// output grid through theta = xy/ell on the reversed grid; the
// optional taper suppresses the highest log-frequency coefficients to
// damp the ringing the FFT's periodic assumption causes. A positive
// dlnt averages each output over the radial bin [theta, theta e^dlnt]
// with measure theta dtheta instead of returning point values.
//
// Returns:
//   NumPy array [nell, nell]: g on the theta = xy/reversed(ell) grid.
// ---------------------------------------------------------------------------
static transform_array bessel_production(
    const transform_array& ell, // [nell], logarithmic Fourier grid
    const transform_array& integrand, // [nell,nell], dln ell integrand
    const int order1, // first nonnegative cylindrical Bessel order
    const int order2, // second nonnegative cylindrical Bessel order
    const double nu1, // first FFTLog bias (tilt)
    const double nu2, // second FFTLog bias (tilt)
    const double xy, // constant in theta = xy/ell
    const double dlnt, // log radial-bin width; 0 = point values
    const int padding, // zero cells added per grid end
    const bool window, // enable the high-frequency taper
    const double window_width // tapered coefficient fraction
  )
{
  shape_transform(ell, 1);
  shape_transform(integrand, 2);
  finite_transform(ell);
  finite_transform(integrand);

  if (!(integrand.shape(0) == ell.size()
        && integrand.shape(1) == ell.size())) {
    spdlog::critical("3pcf transform: integrand shape mismatch");
    std::exit(1);
  }
  if (!(0 <= order1 && 0 <= order2)) {
    spdlog::critical("3pcf transform: negative Bessel order");
    std::exit(1);
  }

  const int nell = static_cast<int>(ell.size());

  // Create a one-shot workspace. The literal 1 is the worker count:
  // one scratch slice suffices because this helper fills, transforms
  // and frees the workspace within a single serial call.
  auto* work = fftlog_create_3pcf(nell, ell.data(),
      std::max(order1, order2), 1, nu1, nu2, xy, dlnt, padding, window,
      window_width);

  // Copy the integrand into the workspace's input slice.
  std::copy_n(integrand.data(), integrand.size(),
      fftlog_input_3pcf(work, 0));

  // Forward pass: compute the FFT coefficients of the biased integrand.
  fftlog_forward_3pcf(work, 0);

  // Inverse pass: contract those coefficients with the Bessel kernels
  // of the requested orders and land on the theta grid.
  const double* result = fftlog_inverse_3pcf(work, 0, order1, order2);

  // Copy the [nell,nell] result into an output the caller owns, then
  // release the workspace before the final finiteness check.
  transform_array output({nell, nell});
  std::copy_n(result, integrand.size(), output.mutable_data());
  fftlog_destroy_3pcf(work);

  finite_transform(output);

  return output;
}



// ============================================================================
// [SECTION] MODULE REGISTRATION
// ============================================================================

// Register the production transform entries and the TransformPlan class
// on the production submodule that bind_threepcf created, then add the
// notebook twins. bind_threepcf must therefore run first.
void bind_threepcf_transform(
    py::module_& module // owning project extension
  )
{
  py::module_ production = module.attr("production").cast<py::module_>();

  production.def("legendre_coupling", &coupling_production,
      "Angular coupling table G[M, L, psi]: the coefficients that "
      "couple Legendre degree L and angular Fourier mode M at opening "
      "angle psi = arctan(ell2/ell1) in [0, pi/2]. npsi=0 evaluates the exact "
      "finite Fourier sum at every query angle; npsi=200 reproduces "
      "fastnc's 200-node linear interpolation table.",
      py::arg("psi").noconvert(), py::arg("lmax"), py::arg("mmax"),
      py::arg("npsi")=200);

  production.def("next_fft_size", &next_even_fft_size,
      "Smallest even integer >= minimum whose only prime factors are "
      "2, 3 and 5. FFTW runs its fast small-radix kernels on such "
      "lengths, while a length with a large prime factor falls back to "
      "a much slower general algorithm, so padding to this size buys "
      "speed for little memory.",
      py::arg("minimum"));

  production.def("two_bessel", &bessel_production,
      "Double-Bessel transform: integral dln(ell1) dln(ell2) "
      "f(ell1, ell2) J_order1(ell1 theta1) J_order2(ell2 theta2), with "
      "J the cylindrical Bessel function of the first kind, evaluated "
      "by FFTLog on the grid theta = xy/reversed(ell). dlnt > 0 "
      "averages each radial bin starting at theta with measure "
      "theta dtheta. The transform never extrapolates beyond the grid "
      "endpoints.",
      py::arg("ell").noconvert(), py::arg("integrand").noconvert(),
      py::arg("order1"), py::arg("order2"), py::arg("nu1")=1.01,
      py::arg("nu2")=1.01, py::arg("xy")=1.0, py::arg("dlnt")=0.0,
      py::arg("padding")=0, py::arg("window")=false,
      py::arg("window_width")=0.25);

  py::class_<bessel_owner_3pcf>(production, "TwoBesselPlan")
      .def(py::init<const transform_array&, int, double, double, double,
          double, int, bool, double>(),
          "Immutable double-Bessel FFTLog geometry, kernels and FFTW "
          "plans. max_order prepares every cylindrical Bessel order "
          "from zero through that value. Each object owns separate "
          "scratch; a new grid or numerical setting needs a new object.",
          py::arg("ell").noconvert(), py::arg("max_order"),
          py::arg("nu1")=1.01, py::arg("nu2")=1.01, py::arg("xy")=1.0,
          py::arg("dlnt")=0.0, py::arg("padding")=0,
          py::arg("window")=false, py::arg("window_width")=0.25)
      .def("forward", &bessel_owner_3pcf::forward,
          "Copy the complete [nell,nell] float64 integrand and replace "
          "its forward coefficients. Reuses preparation, never old input.",
          py::arg("integrand").noconvert())
      .def("inverse", &bessel_owner_3pcf::inverse,
          "Return the double-Bessel transform of the latest input for "
          "orders in [0,max_order]. Requires a completed forward call. "
          "The returned [nell,nell] array owns its values independently.",
          py::arg("order1"), py::arg("order2"))
      .def("evaluate", &bessel_owner_3pcf::evaluate,
          "Replace the input, run its forward FFT and return one "
          "double-Bessel transform on the retained geometry.",
          py::arg("integrand").noconvert(), py::arg("order1"),
          py::arg("order2"))
      .def_property_readonly("theta", &bessel_owner_3pcf::theta,
          "Independent copy of theta = xy/reversed(ell); with dlnt>0 "
          "these are lower radial-bin edges.");

  module.def("aligned_fft_grid", &aligned_fft_grid_3pcf,
    py::arg("theta_lower"), py::arg("dlnt"), py::arg("ell_min"),
    py::arg("ell_max"), py::arg("minimum_size"));

  py::class_<transform_owner_3pcf>(production, "TransformPlan")
      .def(py::init<const transform_array&, const transform_array&,
          const transform_indices&, int, int, double, double, double,
          double, int, bool, double, int, double, const std::string&,
          const std::string&, bool>(),
          "A transform plan: immutable triangle geometry with private "
          "FFT scratch. theta holds the lower edges of the radial bins "
          "that indices pick from the grid theta = xy/reversed(ell); "
          "the plan never interpolates to a nearest node. projection "
          "names the phase convention: centroid and centroid_tatt give "
          "the TATT reference centroid phase, centroid_main its "
          "explicit conjugate, and X the X-mode projection. dphi > 0 "
          "averages the X modes over the opening-angle bin before the "
          "plan applies the centroid phase at the bin center. "
          "ordered_legs=True keeps a general b_L with ordered multipole "
          "legs; False assumes b_L symmetric in its two legs and uses "
          "the finite-FFT transpose shortcut of the fastnc reference "
          "code.",
          py::arg("ell").noconvert(), py::arg("phi").noconvert(),
          py::arg("indices").noconvert(), py::arg("lmax"), py::arg("mmax"),
          py::arg("nu1")=1.01, py::arg("nu2")=1.01, py::arg("xy")=1.0,
          py::arg("dlnt")=0.0, py::arg("padding")=0,
          py::arg("window")=false, py::arg("window_width")=0.25,
          py::arg("npsi")=200, py::arg("dphi")=0.0,
          py::arg("projection")="centroid", py::arg("convention")="fastnc",
          py::arg("ordered_legs")=true)
      .def_static("for_aperture", &aperture_plan_3pcf,
          py::arg("bins"), py::arg("filters"), py::arg("theta_lower"),
          py::arg("phi"), py::arg("dlnt"), py::arg("dphi"),
          py::arg("ell_min"), py::arg("ell_max"), py::arg("minimum_size"),
          py::arg("lmax"), py::arg("mmax"), py::arg("nmu"),
          py::arg("transform_options")=py::dict(),
          py::arg("projection_mode")="prepared",
          py::arg("projection_settings")=py::dict(),
          py::arg("projection_batch_size")=4,
          py::arg("accuracy")=py::none(), py::arg("side_keep")=py::none(),
          py::arg("representative_rule")="geometric",
          py::arg("representative_offset")=0.0)
      .def("aperture_vector", &transform_owner_3pcf::aperture_vector,
          py::arg("model"), py::arg("ia"),
          py::arg("calibration")=py::none(), py::arg("component")=0)
      .def("aperture_components", &transform_owner_3pcf::aperture_components,
          py::arg("model"), py::arg("ia"),
          py::arg("calibration")=py::none())
      .def_property_readonly("weights_real", &transform_owner_3pcf::weights_real)
      .def_property_readonly("weights_imag", &transform_owner_3pcf::weights_imag)
      .def_property_readonly("side_keep", &transform_owner_3pcf::side_keep)
      .def_property_readonly("representative_sides",
          &transform_owner_3pcf::representative_sides)
      .def_property_readonly("ell", &transform_owner_3pcf::ell)
      .def_property_readonly("projection_settings",
          &transform_owner_3pcf::projection_settings)
      .def("natural_components", &transform_owner_3pcf::natural,
          "Natural components from bispectrum multipoles: return the "
          "pair (real, imag) of arrays [4, theta, theta, phi] holding "
          "the four natural components Gamma_0..Gamma_3 split into "
          "real and imaginary parts. b_l holds the Legendre "
          "coefficients of the projected E-mode bispectrum in mu, the "
          "cosine in the triangle closure ell3^2 = ell1^2 + ell2^2 - "
          "2 ell1 ell2 mu, on axes [L, ell1, ell2].",
          py::arg("b_l").noconvert())
      .def("aperture_from_multipoles", &transform_owner_3pcf::aperture,
          "Aperture moments from bispectrum multipoles: transform b_l "
          "to the four natural components, convert them to the "
          "TreeCorr component convention -[Gamma_0, Gamma_1, Gamma_3, "
          "Gamma_2], and contract with the caller's finite-bin "
          "aperture weights. weights_real and weights_imag hold the "
          "real and imaginary parts of the integration weights, on "
          "axes [filter, 4, triangle].",
          py::arg("b_l").noconvert(),
          py::arg("weights_real").noconvert(),
          py::arg("weights_imag").noconvert())
      .def("aperture_from_grid", &transform_owner_3pcf::aperture_grid,
          "Aperture moments refreshed from a physical grid: read fresh "
          "bispectrum multipoles b_L from the grid on every call, "
          "reusing only this plan's query geometry and temporary "
          "storage, and return the same aperture moments as "
          "aperture_from_multipoles.",
          py::arg("grid"), py::arg("weights_real").noconvert(),
          py::arg("weights_imag").noconvert())
      .def_property_readonly("theta", &transform_owner_3pcf::theta)
      .def_property_readonly("theta_full", &transform_owner_3pcf::theta_full);

  bind_transform_notebook_3pcf(module);
}

}
