#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <pybind11/numpy.h>
#include <spdlog/spdlog.h>
#include "physics_interface.hpp"
#include "covariance_wrapper.hpp"

namespace py = pybind11;
namespace cosmolike_interface {

// Covariance inputs describe both observables and numerical integration
// rules. The dictionary is checked before any expensive C calculation.
// Production retains references to its contiguous NumPy buffers for the
// duration of the call. Notebook inputs may be sliced; private copies
// then pass through the Armadillo wrapper. Neither adapter changes a
// physical equation, adds a halo term, or performs a numerical integral.
//
// Field labels name (source bin, aperture radius) pairs. Row labels name
// observables built from two or three such fields. Keeping these two
// indices separate prevents a source-noise matrix from being mistaken
// for a covariance matrix between measurement rows.
static void covariance_check(
  const bool condition,
  const char* message
)
{
  if (!condition) {
    spdlog::critical("3pcf covariance interface: {}", message);
    std::exit(1);
  }
}


static size_t covariance_count(const std::vector<py::ssize_t>& shape)
{
  size_t product = 1;
  for (const auto length : shape) {
    covariance_check(length > 0 && length <= INT_MAX,
      "invalid array dimension");
    covariance_check(static_cast<size_t>(length) <= SIZE_MAX/product,
      "array dimension product overflows");
    product *= static_cast<size_t>(length);
  }
  covariance_check(product <= static_cast<size_t>(PY_SSIZE_T_MAX)
    /sizeof(double), "array byte count overflows");
  return product;
}


template <typename Number>
static py::array_t<Number, py::array::c_style> covariance_array(
  const py::handle object,
  const int rank,
  const bool notebook
)
{
  covariance_check(py::isinstance<py::array>(object),
    "numerical inputs must be NumPy arrays");
  const py::array input = py::reinterpret_borrow<py::array>(object);
  covariance_check(input.dtype().is(py::dtype::of<Number>()),
    "array has the wrong dtype; require float64 or int32 as documented");
  covariance_check(input.ndim() == rank, "array has the wrong rank");
  std::vector<py::ssize_t> shape;
  for (int axis=0; axis<rank; axis++) {
    shape.push_back(input.shape(axis));
  }
  const size_t count = covariance_count(shape);
  if (!notebook) {
    covariance_check((input.flags() & py::array::c_style) != 0,
      "production requires contiguous C-order inputs");
    const auto address = reinterpret_cast<std::uintptr_t>(input.data());
    covariance_check(address%alignof(Number) == 0,
      "production input is not aligned for its C element type");
    return py::reinterpret_borrow<py::array_t<Number, py::array::c_style>>(
      input);
  }
  py::array_t<Number, py::array::c_style> result(shape);
  const char* source = static_cast<const char*>(input.data());
  Number* destination = result.mutable_data();
  for (size_t index=0; index<count; index++) {
    size_t remainder = index;
    py::ssize_t offset = 0;
    for (int axis=rank - 1; axis>=0; axis--) {
      const size_t coordinate = remainder%static_cast<size_t>(shape[axis]);
      remainder /= static_cast<size_t>(shape[axis]);
      offset += static_cast<py::ssize_t>(coordinate)*input.strides(axis);
    }
    // A copy through bytes also accepts a valid but unaligned notebook
    // view. No typed dereference is made until the owned buffer exists.
    std::memcpy(destination + index, source + offset, sizeof(Number));
  }
  return result;
}


static void covariance_finite(const py::array_t<double>& array)
{
  for (py::ssize_t index=0; index<array.size(); index++) {
    covariance_check(std::isfinite(array.data()[index]),
      "nonfinite numerical input or output");
  }
}


class covariance_arguments {
 public:
  covariance_input_3pcf input = {};
  int nsource = 0;
  std::map<std::string, py::array_t<double, py::array::c_style>> real;
  std::map<std::string, py::array_t<int, py::array::c_style>> integer;

  covariance_arguments(
    const physics_model_3pcf* model,
    const py::dict& values,
    const bool notebook
  )
  {
    const std::set<std::string> real_vectors = {
      "ell", "dell", "window_ell", "window_dell", "phi", "dphi",
      "radius", "dr", "profile_radius", "k", "dk", "mass", "dmass",
      "field_radius"
    };
    const std::set<std::string> real_matrices = {
      "theta", "theta_weight", "noise"
    };
    const std::set<std::string> integer_vectors = {
      "field_source", "row_kind"
    };
    const std::set<std::string> scalars = {
      "area", "center_window", "large_area", "estimator", "ia_model",
      "noise_mode",
      "st_a", "st_p", "st_norm", "delta_c", "collapse_model",
      "profile_upsample",
      "concentration_a",
      "concentration_b", "concentration_c", "concentration_pivot",
      "baryon_strength"
    };
    for (const auto& item : values) {
      covariance_check(py::isinstance<py::str>(item.first),
        "input dictionary keys must be strings");
      const std::string name = py::cast<std::string>(item.first);
      if (real_vectors.count(name) || real_matrices.count(name)
        || name == "theta_edges") {
        const int rank = real_vectors.count(name) ? 1 : 2;
        auto array = covariance_array<double>(item.second, rank, notebook);
        covariance_finite(array);
        real.emplace(name, std::move(array));
      } else if (integer_vectors.count(name) || name == "row_fields") {
        const int rank = name == "row_fields" ? 2 : 1;
        auto array = covariance_array<int>(item.second, rank, notebook);
        integer.emplace(name, std::move(array));
      } else {
        covariance_check(scalars.count(name) != 0,
          "unknown covariance dictionary key");
      }
    }
    for (const auto& name : real_vectors) {
      covariance_check(real.count(name) != 0, "missing real input vector");
    }
    for (const auto& name : real_matrices) {
      covariance_check(real.count(name) != 0, "missing real input matrix");
    }
    for (const auto& name : integer_vectors) {
      covariance_check(integer.count(name) != 0, "missing integer vector");
    }
    covariance_check(integer.count("row_fields") != 0,
      "missing row_fields matrix");
    covariance_check(values.contains("area"), "survey area is required");
    int nlos;
    const double* z;
    const double* chi;
    const double* dchi;
    const double* lens;
    const double* shape;
    physics_radial_3pcf(model, &nsource, &nlos, &z, &chi, &dchi, &lens, &shape);

    input.nell = vector_length("ell");
    input.nwindow = vector_length("window_ell");
    input.nphi = vector_length("phi");
    input.nr = vector_length("radius");
    input.nprofile = vector_length("profile_radius");
    input.nk = vector_length("k");
    input.nmass = vector_length("mass");
    input.nfield = vector_length("field_radius");
    input.nrow = static_cast<int>(integer.at("row_kind").size());
    input.ntheta = static_cast<int>(real.at("theta").shape(1));
    equal_length("dell", input.nell);
    equal_length("window_dell", input.nwindow);
    equal_length("dphi", input.nphi);
    equal_length("dr", input.nr);
    equal_length("dk", input.nk);
    equal_length("dmass", input.nmass);
    covariance_check(integer.at("field_source").size() == input.nfield,
      "field_source and field_radius lengths disagree");
    covariance_check(integer.at("row_fields").shape(0) == input.nrow
      && integer.at("row_fields").shape(1) == 3,
      "row_fields must have shape [nrow,3]");
    matrix_shape("theta", input.nrow, input.ntheta);
    matrix_shape("theta_weight", input.nrow, input.ntheta);
    matrix_shape("noise", nsource, nsource);
    if (real.count("theta_edges")) {
      matrix_shape("theta_edges", input.nrow, 2);
      input.theta_edges = real.at("theta_edges").data();
    }

    input.ell = real.at("ell").data();
    input.dell = real.at("dell").data();
    input.window_ell = real.at("window_ell").data();
    input.window_dell = real.at("window_dell").data();
    input.phi = real.at("phi").data();
    input.dphi = real.at("dphi").data();
    input.radius = real.at("radius").data();
    input.dr = real.at("dr").data();
    input.profile_radius = real.at("profile_radius").data();
    input.k = real.at("k").data();
    input.dk = real.at("dk").data();
    input.mass = real.at("mass").data();
    input.dmass = real.at("dmass").data();
    input.field_source = integer.at("field_source").data();
    input.field_radius = real.at("field_radius").data();
    input.row_kind = integer.at("row_kind").data();
    input.row_fields = integer.at("row_fields").data();
    input.theta = real.at("theta").data();
    input.theta_weight = real.at("theta_weight").data();
    input.noise = real.at("noise").data();
    input.area = scalar(values, "area", 0.0);
    input.center_window = integer_scalar(values, "center_window", 1);
    input.large_area = integer_scalar(values, "large_area", 0);
    input.estimator = integer_scalar(values, "estimator", 0);
    input.ia_model = integer_scalar(values, "ia_model", 0);
    input.noise_mode = integer_scalar(values, "noise_mode", 0);
    input.st_a = scalar(values, "st_a", 0.707);
    input.st_p = scalar(values, "st_p", 0.3);
    input.st_norm = scalar(values, "st_norm", 0.3222);
    input.delta_c = scalar(values, "delta_c", 1.686);
    input.collapse_model = integer_scalar(values, "collapse_model", 0);
    input.profile_upsample = integer_scalar(values, "profile_upsample", 8);
    input.concentration_a = scalar(values, "concentration_a", 10.14);
    input.concentration_b = scalar(values, "concentration_b", -0.081);
    input.concentration_c = scalar(values, "concentration_c", -1.01);
    input.concentration_pivot = scalar(values, "concentration_pivot", 2.0e12);
    input.baryon_strength = scalar(values, "baryon_strength", 0.0);
    validate();
  }

 private:
  int vector_length(const char* name) const
  {
    return static_cast<int>(real.at(name).size());
  }

  void equal_length(const char* name, const int expected) const
  {
    covariance_check(vector_length(name) == expected,
      "integration nodes and weights have different lengths");
  }

  void matrix_shape(const char* name, const int rows, const int columns) const
  {
    const auto& values = real.at(name);
    covariance_check(values.shape(0) == rows && values.shape(1) == columns,
      "matrix shape disagrees with physical labels");
  }

  static double scalar(
    const py::dict& values,
    const char* name,
    const double fallback
  )
  {
    const double result = values.contains(name)
      ? py::cast<double>(values[name]) : fallback;
    covariance_check(std::isfinite(result), "scalar input is nonfinite");
    return result;
  }

  static int integer_scalar(
    const py::dict& values,
    const char* name,
    const int fallback
  )
  {
    if (!values.contains(name)) {
      return fallback;
    }
    covariance_check(py::isinstance<py::int_>(values[name]),
      "discrete model settings must be Python integers");
    return py::cast<int>(values[name]);
  }

  static void increasing(
    const double* values,
    const int count,
    const bool strictly_positive
  )
  {
    for (int index=0; index<count; index++) {
      covariance_check(strictly_positive ? values[index] > 0.0
        : values[index] >= 0.0, "grid has an invalid lower bound");
      covariance_check(index == 0 || values[index] > values[index - 1],
        "integration grid must be strictly increasing");
    }
  }

  static void positive(const double* values, const int count)
  {
    for (int index=0; index<count; index++) {
      covariance_check(values[index] > 0.0,
        "integration weights must be positive");
    }
  }

  void validate() const
  {
    increasing(input.ell, input.nell, true);
    increasing(input.window_ell, input.nwindow, true);
    increasing(input.phi, input.nphi, false);
    increasing(input.radius, input.nr, false);
    increasing(input.profile_radius, input.nprofile, false);
    increasing(input.k, input.nk, true);
    increasing(input.mass, input.nmass, true);
    positive(input.dell, input.nell);
    positive(input.window_dell, input.nwindow);
    positive(input.dphi, input.nphi);
    positive(input.dr, input.nr);
    positive(input.dk, input.nk);
    positive(input.dmass, input.nmass);
    covariance_check(input.nprofile >= 3 && input.profile_radius[0] == 0.0,
      "profile_radius needs at least three nodes and must start at zero");
    covariance_check(input.profile_upsample >= 1
      && input.nprofile - 1 <= (INT_MAX - 1)/input.profile_upsample,
      "profile upsampling count is invalid or overflows");
    covariance_check(input.phi[input.nphi - 1] <= 2.0*arma::datum::pi,
      "relative angle lies outside [0,2pi]");
    double angle_measure = 0.0;
    for (int index=0; index<input.nphi; index++) {
      angle_measure += input.dphi[index];
    }
    covariance_check(std::abs(angle_measure - 2.0*arma::datum::pi) < 1.0e-10,
      "relative-angle quadrature weights must sum to 2pi");
    covariance_check(input.area > 0.0 && input.area <= 4.0*arma::datum::pi,
      "survey solid angle must lie in (0,4pi]");
    covariance_check(input.center_window == 1 || input.center_window == 2,
      "center_window must be 1 (circle) or 2 (square)");
    covariance_check(input.large_area == 0 || input.large_area == 1,
      "large_area must be zero or one");
    covariance_check(input.estimator == 0 && input.ia_model == 0,
      "only full-filter map products with gravitational covariance exist");
    covariance_check(input.noise_mode == 0 || input.noise_mode == 1,
      "noise_mode must be finite-band (0) or uniform-area xi noise (1)");
    covariance_check(input.noise_mode == 0
      || (input.large_area == 1 && input.theta_edges != nullptr),
      "analytic xi noise requires large_area=1 and theta_edges[nrow,2]");
    covariance_check(input.st_a > 0.0 && input.st_p >= 0.0
      && input.st_norm > 0.0 && input.delta_c > 0.0
      && input.concentration_a > 0.0 && input.concentration_pivot > 0.0,
      "invalid halo prescription");
    covariance_check(input.collapse_model == 0 || input.collapse_model == 1,
      "collapse_model must select constant (0) or evolving (1) threshold");
    for (int field=0; field<input.nfield; field++) {
      covariance_check(input.field_source[field] >= 0
        && input.field_source[field] < nsource,
        "field source index is out of range");
      covariance_check(input.field_radius[field] >= 0.0,
        "field aperture radius must be nonnegative");
    }
    double largest_theta = 0.0;
    for (int row=0; row<input.nrow; row++) {
      const int kind = input.row_kind[row];
      covariance_check(kind >= 0 && kind <= 3, "unknown observable row kind");
      const int legs = kind == 1 ? 3 : 2;
      for (int leg=0; leg<legs; leg++) {
        const int field = input.row_fields[static_cast<size_t>(row)*3 + leg];
        covariance_check(field >= 0 && field < input.nfield,
          "observable field index is out of range");
        covariance_check(kind < 2 ? input.field_radius[field] > 0.0
          : input.field_radius[field] == 0.0,
          "Map rows need positive filters and xi rows need bare shear legs");
      }
      if (kind >= 2) {
        if (input.noise_mode == 1) {
          const double lower = input.theta_edges[static_cast<size_t>(row)*2];
          const double upper = input.theta_edges[
            static_cast<size_t>(row)*2 + 1];
          covariance_check(lower > 0.0 && upper > lower,
            "analytic xi noise needs positive ordered bin edges");
          for (int node=0; node<input.ntheta; node++) {
            const double theta = input.theta[
              static_cast<size_t>(row)*input.ntheta + node];
            covariance_check(theta >= lower && theta <= upper,
              "xi node lies outside its declared analytic-noise bin");
          }
        }
        double normalization = 0.0;
        for (int node=0; node<input.ntheta; node++) {
          const size_t index = static_cast<size_t>(row)*input.ntheta + node;
          covariance_check(input.theta[index] > 0.0
            && input.theta_weight[index] >= 0.0,
            "xi nodes must be positive and their weights nonnegative");
          largest_theta = std::max(largest_theta, input.theta[index]);
          normalization += input.theta_weight[index];
        }
        covariance_check(std::abs(normalization - 1.0) <= 1.0e-10,
          "xi-bin weights must sum to one");
      }
    }
    covariance_check(input.profile_radius[input.nprofile - 1]
      >= input.radius[input.nr - 1] + 0.5*largest_theta,
      "profile grid does not cover shifted xi/shear evaluations");

    // A white-noise power matrix is a covariance between source maps.
    // Its eigenvalues must be nonnegative. A scale-relative roundoff
    // tolerance permits exact zero-noise directions without admitting
    // a negative physical variance merely because its units are small.
    arma::Mat<double> noise(nsource, nsource);
    double magnitude = 0.0;
    for (int row=0; row<nsource; row++) {
      for (int column=0; column<nsource; column++) {
        const double value = input.noise[static_cast<size_t>(row)*nsource
          + column];
        const double reverse = input.noise[static_cast<size_t>(column)*nsource
          + row];
        const double scale = std::max(std::abs(value), std::abs(reverse));
        covariance_check(std::abs(value - reverse)
          <= 64.0*std::numeric_limits<double>::epsilon()*scale,
          "noise matrix is not symmetric");
        noise(row, column) = value;
        magnitude = std::max(magnitude, std::abs(value));
      }
    }
    arma::Col<double> eigenvalues;
    covariance_check(arma::eig_sym(eigenvalues, noise),
      "noise eigenvalue check failed");
    covariance_check(eigenvalues.min() >= -64.0*nsource
      *std::numeric_limits<double>::epsilon()*magnitude,
      "noise matrix is not positive semidefinite");
  }
};


static py::tuple physical_covariance_binding(
  const py::handle owner,
  const py::dict& values,
  const bool notebook
)
{
  const physics_model_3pcf* model = borrow_physics_model_3pcf(owner);
  covariance_arguments arguments(model, values, notebook);
  const int count = arguments.input.nrow;
  covariance_count({6, count, count});
  py::array_t<double> mean(count);
  py::array_t<double> terms({static_cast<py::ssize_t>(6),
    static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(count)});
  if (notebook) {
    const covariance_request_cpp_3pcf request(arguments.input,
      arguments.nsource);
    std::pair<arma::Col<double>, arma::Cube<double>> result;
    {
      py::gil_scoped_release release;
      result = covariance_physical_cpp_3pcf(model, request);
    }
    for (int row=0; row<count; row++) {
      mean.mutable_data()[row] = result.first(row);
      for (int column=0; column<count; column++) {
        for (int term=0; term<6; term++) {
          terms.mutable_data()[(static_cast<size_t>(term)*count + row)*count
            + column] = result.second(row, column, term);
        }
      }
    }
  } else {
    py::gil_scoped_release release;
    covariance_physical_3pcf(model, &arguments.input, mean.mutable_data(),
      terms.mutable_data());
  }
  covariance_finite(mean);
  covariance_finite(terms);
  return py::make_tuple(mean, terms);
}


static py::tuple halos_binding(
  const py::handle owner,
  const py::dict& values,
  const py::handle redshifts,
  const py::handle wavenumbers,
  const bool notebook
)
{
  const physics_model_3pcf* model = borrow_physics_model_3pcf(owner);
  covariance_arguments arguments(model, values, notebook);
  const auto z = covariance_array<double>(redshifts, 1, notebook);
  const auto k = covariance_array<double>(wavenumbers, 1, notebook);
  covariance_finite(z);
  covariance_finite(k);
  for (py::ssize_t index=0; index<z.size(); index++) {
    covariance_check(z.data()[index] >= 0.0
      && (index == 0 || z.data()[index] > z.data()[index - 1]),
      "halo redshifts must be nonnegative and strictly increasing");
  }
  for (py::ssize_t index=0; index<k.size(); index++) {
    covariance_check(k.data()[index] >= 0.0
      && (index == 0 || k.data()[index] > k.data()[index - 1]),
      "profile wavenumbers must be nonnegative and strictly increasing");
  }
  const int mass_count = arguments.input.nmass;
  covariance_count({z.size(), mass_count, k.size()});
  std::vector<py::array_t<double>> result;
  for (int index=0; index<4; index++) {
    result.emplace_back(std::vector<py::ssize_t>{z.size(), mass_count});
  }
  result.emplace_back(std::vector<py::ssize_t>{z.size(), mass_count, k.size()});
  if (notebook) {
    const covariance_request_cpp_3pcf request(arguments.input,
      arguments.nsource);
    const arma::Col<double> redshift(z.data(), z.size());
    const arma::Col<double> wave(k.data(), k.size());
    std::vector<arma::Col<double>> computed;
    {
      py::gil_scoped_release release;
      computed = covariance_halos_cpp_3pcf(model, request, redshift, wave);
    }
    for (int index=0; index<5; index++) {
      std::memcpy(result[index].mutable_data(), computed[index].memptr(),
        static_cast<size_t>(result[index].size())*sizeof(double));
    }
  } else {
    py::gil_scoped_release release;
    covariance_halos_3pcf(model, &arguments.input, static_cast<int>(z.size()),
      z.data(), static_cast<int>(k.size()), k.data(), result[0].mutable_data(),
      result[1].mutable_data(), result[2].mutable_data(),
      result[3].mutable_data(),
      result[4].mutable_data());
  }
  for (const auto& array : result) {
    covariance_finite(array);
  }
  return py::make_tuple(result[0], result[1], result[2], result[3], result[4]);
}


static py::array_t<double> xi_binding(
  const py::handle ell_object,
  const py::handle dell_object,
  const py::handle theta_object,
  const py::handle weight_object,
  const py::handle e_object,
  const py::handle b_object,
  const py::handle sign_object,
  const bool notebook
)
{
  const auto ell = covariance_array<double>(ell_object, 1, notebook);
  const auto dell = covariance_array<double>(dell_object, 1, notebook);
  const auto theta = covariance_array<double>(theta_object, 2, notebook);
  const auto weight = covariance_array<double>(weight_object, 2, notebook);
  const auto spectra_e = covariance_array<double>(e_object, 2, notebook);
  const auto spectra_b = covariance_array<double>(b_object, 2, notebook);
  const auto sign = covariance_array<int>(sign_object, 1, notebook);
  covariance_finite(ell);
  covariance_finite(dell);
  covariance_finite(theta);
  covariance_finite(weight);
  covariance_finite(spectra_e);
  covariance_finite(spectra_b);
  const int nrow = static_cast<int>(theta.shape(0));
  const int ntheta = static_cast<int>(theta.shape(1));
  const int nell = static_cast<int>(ell.size());
  covariance_check(dell.size() == nell && weight.shape(0) == nrow
    && weight.shape(1) == ntheta && spectra_e.shape(0) == nrow
    && spectra_e.shape(1) == nell && spectra_b.shape(0) == nrow
    && spectra_b.shape(1) == nell && sign.size() == nrow,
    "xi input shapes disagree");
  for (int node=0; node<nell; node++) {
    covariance_check(ell.data()[node] > 0.0 && dell.data()[node] > 0.0
      && (node == 0 || ell.data()[node] > ell.data()[node - 1]),
      "xi Fourier nodes must increase and have positive weights");
  }
  for (int row=0; row<nrow; row++) {
    covariance_check(sign.data()[row] == 1 || sign.data()[row] == -1,
      "xi sign must be +1 or -1");
    double normalization = 0.0;
    for (int node=0; node<ntheta; node++) {
      const size_t index = static_cast<size_t>(row)*ntheta + node;
      covariance_check(theta.data()[index] > 0.0 && weight.data()[index] >= 0.0,
        "xi separation nodes must be positive and weights nonnegative");
      normalization += weight.data()[index];
    }
    covariance_check(std::abs(normalization - 1.0) <= 1.0e-10,
      "xi-bin weights must sum to one");
  }
  py::array_t<double> result(nrow);
  if (notebook) {
    // Armadillo keeps the physical [row,node] axes here. The notebook
    // wrapper packs its columns into the C kernel's row-major layout.
    const arma::Col<double> ell_copy(ell.data(), ell.size());
    const arma::Col<double> dell_copy(dell.data(), dell.size());
    arma::Mat<double> theta_copy(nrow, ntheta);
    arma::Mat<double> weight_copy(nrow, ntheta);
    arma::Mat<double> e_copy(nrow, nell);
    arma::Mat<double> b_copy(nrow, nell);
    for (int row=0; row<nrow; row++) {
      for (int node=0; node<ntheta; node++) {
        const size_t index = static_cast<size_t>(row)*ntheta + node;
        theta_copy(row, node) = theta.data()[index];
        weight_copy(row, node) = weight.data()[index];
      }
      for (int node=0; node<nell; node++) {
        const size_t index = static_cast<size_t>(row)*nell + node;
        e_copy(row, node) = spectra_e.data()[index];
        b_copy(row, node) = spectra_b.data()[index];
      }
    }
    const arma::Col<int> sign_copy(sign.data(), sign.size());
    const arma::Col<double> output = covariance_xi_cpp_3pcf(ell_copy,
      dell_copy, theta_copy, weight_copy, e_copy, b_copy, sign_copy);
    std::memcpy(result.mutable_data(), output.memptr(),
      static_cast<size_t>(nrow)*sizeof(double));
  } else {
    py::gil_scoped_release release;
    covariance_xi_from_power_3pcf(nrow, nell, ntheta, ell.data(), dell.data(),
      theta.data(), weight.data(), spectra_e.data(), spectra_b.data(),
      sign.data(), result.mutable_data());
  }
  covariance_finite(result);
  return result;
}


void bind_threepcf_covariance(py::module_& module)
{
  py::module_ production = module.attr("production").cast<py::module_>();
  py::module_ notebook = module.attr("notebook").cast<py::module_>();
  for (const bool copying : {false, true}) {
    py::module_ target = copying ? notebook : production;
    target.def("physical_covariance",
      [copying](py::handle model, const py::dict& inputs) {
        return physical_covariance_binding(model, inputs, copying);
      }, py::arg("model"), py::arg("inputs"),
      "Return mean[nrow], terms[6,nrow,nrow] for gravitational map products.");
    target.def("halo_tables",
      [copying](py::handle model, const py::dict& inputs, py::handle z,
          py::handle k_profile) {
        return halos_binding(model, inputs, z, k_profile, copying);
      }, py::arg("model"), py::arg("inputs"), py::arg("z"),
      py::arg("k_profile"),
      "Return sigma2,dndm,bias,concentration[z,M], then u[z,M,k].");
    target.def("xi_from_power",
      [copying](py::handle ell, py::handle dell, py::handle theta,
          py::handle theta_weight, py::handle spectra_e,
          py::handle spectra_b, py::handle sign) {
        return xi_binding(ell, dell, theta, theta_weight, spectra_e,
          spectra_b, sign, copying);
      }, py::arg("ell"), py::arg("dell"), py::arg("theta"),
      py::arg("theta_weight"), py::arg("spectra_e"), py::arg("spectra_b"),
      py::arg("sign"),
      "Bin-averaged xi: sign +1 uses J0(CE+CB), sign -1 uses J4(CE-CB).");
  }
}

}
