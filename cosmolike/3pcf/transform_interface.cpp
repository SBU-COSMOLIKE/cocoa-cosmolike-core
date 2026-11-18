#include <algorithm>
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
    if (!(values.shape(axis) > 0
          && values.shape(axis) <= INT_MAX)) {
      spdlog::critical("3pcf transform: invalid array axis");
      std::exit(1);
    }
  }

  using scalar = typename Array::value_type;
  const auto address = reinterpret_cast<std::uintptr_t>(values.data());
  if (!(address%alignof(scalar) == 0)) {
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
    if (!(input.shape(axis) > 0
          && input.shape(axis) <= INT_MAX)) {
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

    if (!(lmax >= 0 && lmax < INT_MAX-1)) {
      spdlog::critical("3pcf transform: invalid Legendre cutoff");
      std::exit(1);
    }
    if (!(convention == "fastnc" || convention == "treecorr")) {
      spdlog::critical("3pcf transform: convention must be fastnc or treecorr");
      std::exit(1);
    }

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

    const int triangles = ntheta_*ntheta_*nphi_;
    if (!(real.shape(1) == 4 && real.shape(2) == triangles)) {
      spdlog::critical("3pcf transform: weights do not match this plan's "
        "triangle grid");
      std::exit(1);
    }
    for (int axis=0; axis<3; axis++) {
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

    const int triangles = ntheta_*ntheta_*nphi_;
    if (!(real.shape(1) == 4 && real.shape(2) == triangles)) {
      spdlog::critical("3pcf transform: weights do not match this plan's "
        "triangle grid");
      std::exit(1);
    }
    for (int axis=0; axis<3; axis++) {
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

  transform_plan_3pcf* plan_ = nullptr; // owned C plan and FFT scratch
  int nell_ = 0; // Fourier grid length
  int ntheta_ = 0; // radial bins the indices pick
  int nphi_ = 0; // opening-angle samples
  int lmax_ = 0; // highest Legendre degree in b_L
  transform_array theta_; // lower edges of the kept radial bins, radians
  transform_array theta_full_; // full reversed grid xy/ell, radians
};



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

  if (!(lmax >= 0 && lmax < INT_MAX-1
        && mmax >= 0 && mmax < INT_MAX-1
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
  if (!(order1 >= 0 && order2 >= 0)) {
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
