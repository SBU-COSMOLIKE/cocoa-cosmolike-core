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

namespace py = pybind11;
namespace cosmolike_interface {
using transform_array = py::array_t<double, py::array::c_style>;
using transform_indices = py::array_t<int, py::array::c_style>;

// Production arrays are borrowed after dtype, layout, alignment and
// finiteness checks. Geometry is copied by the C plan, so modifying its
// original NumPy array cannot invalidate a running or later transform.
// The GIL is retained: one plan owns its mutable scratch during a call.
static void require_transform(
    const bool condition, // input or output requirement
    const char* message // failure explanation
  )
{
  if (!condition) {
    spdlog::critical("3pcf transform: {}", message);
    std::exit(1);
  }
}


template <typename Array>
static void shape_transform(
    const Array& values, // borrowed NumPy input
    const int rank // required number of axes
  )
{
  require_transform(values.ndim() == rank, "array has wrong rank");
  for (int axis=0; axis<rank; axis++) {
    require_transform(values.shape(axis) > 0
        && values.shape(axis) <= INT_MAX, "invalid array axis");
  }
  using scalar = typename Array::value_type;
  const auto address = reinterpret_cast<std::uintptr_t>(values.data());
  require_transform(address%alignof(scalar) == 0, "unaligned array");
}


static void finite_transform(
    const transform_array& values // aligned real buffer
  )
{
  for (py::ssize_t index=0; index<values.size(); index++) {
    require_transform(std::isfinite(values.data()[index]),
        "array must contain finite values");
  }
}


static int worker_count_transform()
{
  #ifdef _OPENMP
  return omp_get_max_threads();
  #else
  return 1;
  #endif
}


static int projection_transform(
    const std::string& projection // named phase convention
  )
{
  if (projection == "X") {
    return 0;
  }
  if (projection == "centroid" || projection == "centroid_tatt") {
    return 1;
  }
  require_transform(projection == "centroid_main",
      "projection must be X, centroid_tatt, or centroid_main");
  return 2;
}


template <typename Array>
static Array notebook_transform_input(
    const py::object& values, // any NumPy-compatible notebook input
    const int rank // expected physical rank
  )
{
  // astype(copy=True, order=F) gives an aligned owning temporary. Copy it
  // once more into Armadillo-owned storage; no borrowed lifetime escapes.
  py::module_ numpy = py::module_::import("numpy");
  py::object converted = numpy.attr("asarray")(values).attr("astype")(
      numpy.attr("float64"), py::arg("order")="F", py::arg("copy")=true);
  py::array_t<double, py::array::f_style> input(converted);
  require_transform(input.ndim() == rank, "wrong notebook array rank");
  for (int axis=0; axis<rank; axis++) {
    require_transform(input.shape(axis) > 0
        && input.shape(axis) <= INT_MAX, "invalid notebook axis");
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
  require_transform(output.is_finite(), "nonfinite notebook input");
  return output;
}


class transform_owner_3pcf {
 public:
  transform_owner_3pcf(
      const transform_array& ell, // [nell], Fourier grid
      const transform_array& phi, // [nphi], radians
      const transform_indices& indices, // selected radial grid indices
      const int lmax, // Legendre cutoff
      const int mmax, // angular Fourier cutoff
      const double nu1, // first FFTLog bias
      const double nu2, // second FFTLog bias
      const double xy, // grid alignment
      const double dlnt, // logarithmic radial-bin width
      const int padding, // zero padding on both ends
      const bool window, // corrected coefficient taper
      const double window_width, // taper fraction
      const int npsi, // coupling interpolation length
      const double dphi, // X-mode opening-angle averaging width
      const std::string& projection, // phase convention
      const std::string& convention, // component sign/order
      const bool ordered_legs // general ordered-bin angular transform
    )
  {
    shape_transform(ell, 1);
    shape_transform(phi, 1);
    shape_transform(indices, 1);
    finite_transform(ell);
    finite_transform(phi);
    require_transform(lmax >= 0 && lmax < INT_MAX-1,
        "invalid Legendre cutoff");
    require_transform(convention == "fastnc" || convention == "treecorr",
        "convention must be fastnc or treecorr");
    nell_ = static_cast<int>(ell.size());
    ntheta_ = static_cast<int>(indices.size());
    nphi_ = static_cast<int>(phi.size());
    lmax_ = lmax;
    plan_ = transform_create_3pcf(nell_, ell.data(), ntheta_,
        indices.data(), nphi_, phi.data(), lmax, mmax,
        worker_count_transform(), nu1, nu2, xy, dlnt, padding, window,
        window_width, npsi, dphi, projection_transform(projection),
        convention == "treecorr", ordered_legs);
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

  py::tuple natural(
      const transform_array& multipoles // [L,nell,nell]
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

  py::tuple natural_notebook(
      const py::object& multipoles // copied [L,nell,nell] input
    )
  {
    const auto input =
        notebook_transform_input<arma::Cube<double>>(multipoles, 3);
    require_transform(input.n_rows == static_cast<arma::uword>(lmax_+1)
        && input.n_cols == static_cast<arma::uword>(nell_)
        && input.n_slices == static_cast<arma::uword>(nell_),
        "notebook multipoles mismatch plan");
    auto output = natural_cpp_3pcf(plan_, input, ntheta_, nphi_);
    py::object real = carma::cube_to_arr(std::move(output.real));
    py::object imag = carma::cube_to_arr(std::move(output.imag));
    const auto shape = py::make_tuple(4, ntheta_, ntheta_, nphi_);
    return py::make_tuple(real.attr("reshape")(shape),
        imag.attr("reshape")(shape));
  }

  transform_array aperture(
      const transform_array& multipoles, // [L,nell,nell]
      const transform_array& real, // [filter,4,triangle], real weights
      const transform_array& imag // matching imaginary weights
    )
  {
    validate(multipoles);
    shape_transform(real, 3);
    shape_transform(imag, 3);
    const int triangles = ntheta_*ntheta_*nphi_;
    require_transform(real.shape(1) == 4 && real.shape(2) == triangles,
        "weights do not match this plan's triangle grid");
    for (int axis=0; axis<3; axis++) {
      require_transform(real.shape(axis) == imag.shape(axis),
          "real and imaginary weight shapes differ");
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
    // Return an independent copy: public metadata cannot mutate the plan.
    transform_array result(ntheta_);
    std::copy_n(theta_.data(), ntheta_, result.mutable_data());
    return result;
  }

  transform_array aperture_grid(
      const py::object& grid, // immutable physical grid, kept alive by Python
      const transform_array& real, // borrowed [filter,4,triangle] weights
      const transform_array& imag // same shape
    )
  {
    shape_transform(real, 3);
    shape_transform(imag, 3);
    const int triangles = ntheta_*ntheta_*nphi_;
    require_transform(real.shape(1) == 4 && real.shape(2) == triangles,
        "weights do not match this plan's triangle grid");
    for (int axis=0; axis<3; axis++) {
      require_transform(real.shape(axis) == imag.shape(axis),
          "real and imaginary weight shapes differ");
    }
    finite_transform(real);
    finite_transform(imag);
    const auto* physical = borrow_physics_grid_3pcf(grid);
    const int nfilter = static_cast<int>(real.shape(0));
    transform_array result(nfilter);
    // The GIL remains held, as for the existing transform call: one plan
    // owns mutable FFT and interpolation scratch until this method returns.
    transform_aperture_grid_3pcf(plan_, physical, nfilter, real.data(),
        imag.data(), result.mutable_data());
    finite_transform(result);
    return result;
  }

  py::object aperture_grid_notebook(
      const py::object& grid,
      const py::object& real,
      const py::object& imag
    )
  {
    const auto real_copy = notebook_transform_input<arma::Cube<double>>(real, 3);
    const auto imag_copy = notebook_transform_input<arma::Cube<double>>(imag, 3);
    const auto* physical = borrow_physics_grid_3pcf(grid);
    py::object output = carma::col_to_arr(aperture_grid_cpp_3pcf(plan_, physical,
        real_copy, imag_copy, ntheta_*ntheta_*nphi_));
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
  void validate(
      const transform_array& multipoles // borrowed real coefficients
    ) const
  {
    shape_transform(multipoles, 3);
    require_transform(multipoles.shape(0) == lmax_+1
        && multipoles.shape(1) == nell_
        && multipoles.shape(2) == nell_, "multipoles mismatch plan shape");
    finite_transform(multipoles);
  }

  transform_plan_3pcf* plan_ = nullptr;
  int nell_ = 0;
  int ntheta_ = 0;
  int nphi_ = 0;
  int lmax_ = 0;
  transform_array theta_;
  transform_array theta_full_;
};


static void bind_transform_notebook_3pcf(
    py::module_& module // owning extension
  )
{
  py::module_ notebook = module.attr("notebook").cast<py::module_>();
  notebook.def("legendre_coupling", [](
      const py::object& psi, // query angles in radians
      const int lmax, // Legendre cutoff
      const int mmax, // Fourier cutoff
      const int npsi // interpolation length
    ) {
    const auto input =
        notebook_transform_input<arma::Col<double>>(psi, 1);
    return carma::cube_to_arr(coupling_cpp_3pcf(input, lmax, mmax, npsi));
  }, py::arg("psi"), py::arg("lmax"), py::arg("mmax"),
      py::arg("npsi")=200);
  notebook.def("two_bessel", [](
      const py::object& ell, // Fourier grid
      const py::object& integrand, // dln ell integrand
      const int order1, // first order
      const int order2, // second order
      const double nu1, // first bias
      const double nu2, // second bias
      const double xy, // alignment
      const double dlnt, // radial-bin width
      const int padding, // zero padding
      const bool window, // coefficient taper
      const double window_width // taper fraction
    ) {
    const auto ell_copy =
        notebook_transform_input<arma::Col<double>>(ell, 1);
    const auto values =
        notebook_transform_input<arma::Mat<double>>(integrand, 2);
    return carma::mat_to_arr(bessel_cpp_3pcf(ell_copy, values, order1,
        order2, nu1, nu2, xy, dlnt, padding, window, window_width));
  }, py::arg("ell"), py::arg("integrand"), py::arg("order1"),
      py::arg("order2"), py::arg("nu1")=1.01, py::arg("nu2")=1.01,
      py::arg("xy")=1.0, py::arg("dlnt")=0.0, py::arg("padding")=0,
      py::arg("window")=false, py::arg("window_width")=0.25);
  notebook.def("natural_components", [](
      transform_owner_3pcf& plan, // already prepared geometry
      const py::object& multipoles // copied, arbitrary-stride values
    ) {
    return plan.natural_notebook(multipoles);
  }, "Copy b_L through Armadillo; execute the same prepared C plan.",
      py::arg("plan"), py::arg("b_l"));
  notebook.def("aperture_from_grid", [](
      transform_owner_3pcf& plan,
      const py::object& grid,
      const py::object& real,
      const py::object& imag
    ) {
    return plan.aperture_grid_notebook(grid, real, imag);
  }, "Copy weights through Armadillo and use the prepared C interpolation map.",
      py::arg("plan"), py::arg("grid"), py::arg("weights_real"),
      py::arg("weights_imag"));
}


static transform_array coupling_production(
    const transform_array& psi, // [query], radians
    const int lmax, // largest Legendre degree
    const int mmax, // largest angular mode
    const int npsi // optional interpolation grid
  )
{
  shape_transform(psi, 1);
  finite_transform(psi);
  require_transform(lmax >= 0 && lmax < INT_MAX-1
      && mmax >= 0 && mmax < INT_MAX-1
      && lmax <= INT_MAX-1-mmax, "invalid coupling cutoffs");
  const int nquery = static_cast<int>(psi.size());
  transform_array output({mmax+1, lmax+1, nquery});
  legendre_coupling_3pcf(lmax, mmax, nquery, psi.data(), npsi,
      output.mutable_data());
  finite_transform(output);
  return output;
}


static transform_array bessel_production(
    const transform_array& ell, // [nell], logarithmic Fourier grid
    const transform_array& integrand, // [nell,nell], dln ell integrand
    const int order1, // first nonnegative cylindrical Bessel order
    const int order2, // second nonnegative cylindrical Bessel order
    const double nu1, // first bias
    const double nu2, // second bias
    const double xy, // alignment
    const double dlnt, // logarithmic radial-bin width
    const int padding, // zero padding per end
    const bool window, // corrected taper
    const double window_width // tapered frequency fraction
  )
{
  shape_transform(ell, 1);
  shape_transform(integrand, 2);
  finite_transform(ell);
  finite_transform(integrand);
  require_transform(integrand.shape(0) == ell.size()
      && integrand.shape(1) == ell.size(), "integrand shape mismatch");
  require_transform(order1 >= 0 && order2 >= 0, "negative Bessel order");
  const int nell = static_cast<int>(ell.size());
  auto* work = fftlog_create_3pcf(nell, ell.data(),
      std::max(order1, order2), 1, nu1, nu2, xy, dlnt, padding, window,
      window_width);
  std::copy_n(integrand.data(), integrand.size(),
      fftlog_input_3pcf(work, 0));
  fftlog_forward_3pcf(work, 0);
  const double* result = fftlog_inverse_3pcf(work, 0, order1, order2);
  transform_array output({nell, nell});
  std::copy_n(result, integrand.size(), output.mutable_data());
  fftlog_destroy_3pcf(work);
  finite_transform(output);
  return output;
}


void bind_threepcf_transform(
    py::module_& module // owning project extension
  )
{
  py::module_ production = module.attr("production").cast<py::module_>();
  production.def("legendre_coupling", &coupling_production,
      "G[M,L,psi]. npsi=0 evaluates exact finite Fourier sums; "
      "npsi=200 reproduces the historical angular interpolation grid.",
      py::arg("psi").noconvert(), py::arg("lmax"), py::arg("mmax"),
      py::arg("npsi")=200);
  production.def("next_fft_size", &next_fft_size_3pcf,
      "Smallest even integer >= minimum with only factors 2,3,5.",
      py::arg("minimum"));
  production.def("two_bessel", &bessel_production,
      "Integral dln(ell1)dln(ell2) f J_order1 J_order2 on the "
      "theta=xy/reversed(ell) grid. dlnt averages radial bins beginning "
      "at theta, with measure theta*dtheta. No endpoint extrapolation.",
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
          "Immutable geometry with private FFT scratch. theta contains "
          "selected radial lower edges; no nearest-node interpolation. "
          "centroid is the historical TATT phase; centroid_main is its "
          "explicit conjugate. dphi averages X modes before applying "
          "the centroid phase at the bin center. ordered_legs=True "
          "retains nonsymmetric b_L; False assumes symmetric b_L and "
          "retains the historical finite-FFT transpose shortcut.",
          py::arg("ell").noconvert(), py::arg("phi").noconvert(),
          py::arg("indices").noconvert(), py::arg("lmax"), py::arg("mmax"),
          py::arg("nu1")=1.01, py::arg("nu2")=1.01, py::arg("xy")=1.0,
          py::arg("dlnt")=0.0, py::arg("padding")=0,
          py::arg("window")=false, py::arg("window_width")=0.25,
          py::arg("npsi")=200, py::arg("dphi")=0.0,
          py::arg("projection")="centroid", py::arg("convention")="fastnc",
          py::arg("ordered_legs")=true)
      .def("natural_components", &transform_owner_3pcf::natural,
          "Return (real,imag)[4,theta,theta,phi] from b_L[L,ell,ell]. "
          "b_L uses ell3^2=ell1^2+ell2^2-2 ell1 ell2 mu.",
          py::arg("b_l").noconvert())
      .def("aperture_from_multipoles", &transform_owner_3pcf::aperture,
          "Transform, convert to TreeCorr -[G0,G1,G3,G2], and apply "
          "supplied finite aperture weights [filter,4,triangle].",
          py::arg("b_l").noconvert(),
          py::arg("weights_real").noconvert(),
          py::arg("weights_imag").noconvert())
      .def("aperture_from_grid", &transform_owner_3pcf::aperture_grid,
          "Read fresh b_L from a physical grid, reusing only query geometry "
          "and temporary storage; return the same aperture moments.",
          py::arg("grid"), py::arg("weights_real").noconvert(),
          py::arg("weights_imag").noconvert())
      .def_property_readonly("theta", &transform_owner_3pcf::theta)
      .def_property_readonly("theta_full", &transform_owner_3pcf::theta_full);
  bind_transform_notebook_3pcf(module);
}
}
