#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>
#include <pybind11/numpy.h>
#include <spdlog/spdlog.h>
#include "physics_interface.hpp"
#include "physics_wrapper.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
namespace cosmolike_interface {

// The production adapter borrows contiguous float64 buffers only during
// each call. Model creation copies their physical values into immutable C
// tables. The notebook factory and methods accept sliced arrays, copy to
// Armadillo, and call the separate wrapper. Both reach the same C model.
// No interpolation, normalization, projection or cosmology is computed
// here; these routines check shapes, arrange storage and manage lifetime.
using physics_array = py::array_t<double, py::array::c_style>;
using physics_int_array = py::array_t<int, py::array::c_style>;
using physics_notebook_array = py::array_t<double, py::array::forcecast>;
using physics_notebook_int = py::array;

static void require_physics_interface(
  const bool condition,
  const char* message
)
{
  if (!condition) {
    spdlog::critical("3pcf physics interface: {}", message);
    std::exit(1);
  }
}


template <typename Array>
static void shape_physics(
  const Array& values,
  const int rank
)
{
  require_physics_interface(values.ndim() == rank, "wrong input rank");
  size_t product = 1;
  for (int axis=0; axis<rank; axis++) {
    require_physics_interface(values.shape(axis) > 0
      && values.shape(axis) <= INT_MAX, "invalid input dimension");
    const size_t length = static_cast<size_t>(values.shape(axis));
    require_physics_interface(length <= SIZE_MAX/product,
      "input shape product overflows");
    product *= length;
  }
  require_physics_interface(product <= SIZE_MAX/values.itemsize(),
    "input byte count overflows");
}


static void aligned_physics(
  const py::array& values,
  const size_t alignment
)
{
  const auto address = reinterpret_cast<std::uintptr_t>(values.data());
  require_physics_interface(address%alignment == 0,
    "production input buffer is not aligned for its element type");
}


static void finite_physics(const physics_array& values)
{
  aligned_physics(values, alignof(double));
  for (py::ssize_t index=0; index<values.size(); index++) {
    require_physics_interface(std::isfinite(values.data()[index]),
      "input or result contains a nonfinite value");
  }
}


template <typename Number>
static arma::Col<Number> copy_column_physics(
  const py::array_t<Number, py::array::forcecast>& values
)
{
  shape_physics(values, 1);
  arma::Col<Number> result(values.shape(0));
  const char* base = static_cast<const char*>(values.request().ptr);
  for (py::ssize_t index=0; index<values.shape(0); index++) {
    // memcpy reads potentially unaligned NumPy buffers without creating
    // a misaligned C pointer; negative slice strides remain valid.
    Number value;
    std::memcpy(&value, base + index*values.strides(0), sizeof(Number));
    require_physics_interface(std::isfinite(static_cast<double>(value)),
      "notebook input contains a nonfinite value");
    result(index) = value;
  }
  return result;
}


template <typename Number>
static arma::Mat<Number> copy_matrix_physics(
  const py::array_t<Number, py::array::forcecast>& values
)
{
  shape_physics(values, 2);
  arma::Mat<Number> result(values.shape(0), values.shape(1));
  const char* base = static_cast<const char*>(values.request().ptr);
  for (py::ssize_t row=0; row<values.shape(0); row++) {
    for (py::ssize_t column=0; column<values.shape(1); column++) {
      Number value;
      std::memcpy(&value, base + row*values.strides(0)
        + column*values.strides(1), sizeof(Number));
      require_physics_interface(std::isfinite(value),
        "notebook matrix contains a nonfinite value");
      result(row, column) = value;
    }
  }
  return result;
}


// Inspect source labels before asking NumPy to convert their dtype.
// A force-cast parameter would already have truncated fractional values
// or wrapped wide integers by the time this function received them.
// Match the aperture adapter: labels have integer dtype, may be sliced,
// and must fit in int32 before being copied into Armadillo storage.
static py::array_t<int, py::array::forcecast> integer_notebook_physics(
  const py::array& values,
  const int rank
)
{
  shape_physics(values, rank);
  require_physics_interface(values.dtype().kind() == 'i'
    || values.dtype().kind() == 'u',
    "notebook source bins must have integer dtype");
  const double minimum = py::cast<double>(values.attr("min")());
  const double maximum = py::cast<double>(values.attr("max")());
  require_physics_interface(minimum >= INT_MIN && maximum <= INT_MAX,
    "notebook source bins exceed int32 range");
  return py::array_t<int, py::array::forcecast>(values);
}


static arma::Col<int> copy_column_physics(const py::array& values)
{
  const auto integers = integer_notebook_physics(values, 1);
  return copy_column_physics<int>(integers);
}


static arma::Mat<int> copy_matrix_physics(const py::array& values)
{
  const auto integers = integer_notebook_physics(values, 2);
  return copy_matrix_physics<int>(integers);
}


static physics_ia_3pcf parameters_physics(const py::dict& values)
{
  physics_ia_3pcf parameters;
  physics_default_ia_3pcf(&parameters);
  for (const auto& item : values) {
    const std::string key = py::cast<std::string>(item.first);
    if (key == "a1") {
      parameters.a1 = py::cast<double>(item.second);
    } else if (key == "eta1") {
      parameters.eta1 = py::cast<double>(item.second);
    } else if (key == "a2") {
      parameters.a2 = py::cast<double>(item.second);
    } else if (key == "eta2") {
      parameters.eta2 = py::cast<double>(item.second);
    } else if (key == "bta") {
      parameters.bta = py::cast<double>(item.second);
    } else if (key == "ct") {
      parameters.ct = py::cast<double>(item.second);
    } else if (key == "zpivot") {
      parameters.zpivot = py::cast<double>(item.second);
    } else if (key == "c1rho") {
      parameters.c1rho = py::cast<double>(item.second);
    } else if (key == "fb") {
      parameters.fb = py::cast<double>(item.second);
    } else if (key == "power_mode") {
      parameters.power_mode = py::cast<int>(item.second);
    } else if (key == "ct_mode") {
      parameters.ct_mode = py::cast<int>(item.second);
    } else if (key == "suppress_only") {
      parameters.suppress_only = py::cast<int>(item.second);
    } else if (key == "operator_component") {
      parameters.operator_component = py::cast<int>(item.second);
    } else if (key == "intrinsic_legs") {
      parameters.intrinsic_legs = py::cast<int>(item.second);
    } else {
      require_physics_interface(false, "unknown IA dictionary key");
    }
  }
  return parameters;
}


static physics_grid_settings_3pcf settings_physics_grid(const py::dict& values)
{
  physics_grid_settings_3pcf settings;
  physics_grid_defaults_3pcf(&settings);
  const std::map<std::string, double*> real = {
    {"ell_min", &settings.ell_min},
    {"ell_max", &settings.ell_max},
    {"epmu", &settings.epmu},
    {"shape_stretch", &settings.shape_stretch},
    {"psi_scale", &settings.psi_scale}
  };
  const std::map<std::string, int*> integer = {
    {"nr", &settings.nr},
    {"nu", &settings.nu},
    {"nv", &settings.nv},
    {"np", &settings.np},
    {"npu_log", &settings.npu_log},
    {"npu_linear", &settings.npu_linear},
    {"nell", &settings.nell},
    {"npsi_log", &settings.npsi_log},
    {"npsi_linear", &settings.npsi_linear},
    {"nmu_log", &settings.nmu_log},
    {"nmu_linear", &settings.nmu_linear},
    {"lmax", &settings.lmax},
    {"reference_mode", &settings.reference_mode},
    {"table_boost", &settings.table_boost},
    {"multipole_boost", &settings.multipole_boost},
    {"npsi", &settings.npsi},
    {"prepare_multipoles", &settings.prepare_multipoles},
    {"nshape", &settings.nshape},
    {"signed_baseline", &settings.signed_baseline},
    {"residual_scaling", &settings.residual_scaling}
  };
  for (const auto& item : values) {
    const std::string name = py::cast<std::string>(item.first);
    if (real.count(name)) {
      *real.at(name) = py::cast<double>(item.second);
      require_physics_interface(std::isfinite(*real.at(name)),
        "nonfinite interpolation setting");
    } else {
      require_physics_interface(integer.count(name) != 0,
        "unknown interpolation setting");
      require_physics_interface(py::isinstance<py::int_>(item.second),
        "interpolation counts must be integers");
      *integer.at(name) = py::cast<int>(item.second);
    }
  }
  return settings;
}


static std::shared_ptr<physics_owner_3pcf> model_physics(
  const physics_array& k,
  const physics_array& p_linear0,
  const physics_array& z,
  const physics_array& p_nonlinear,
  const physics_array& chi,
  const physics_array& dz_dchi,
  const physics_array& growth,
  const physics_array& sources,
  const double omega_m,
  const double omega_de,
  const double w0,
  const double wa,
  const double ns,
  const double sigma8,
  const physics_array& los_z,
  const int kernel_nz,
  const bool source_readers,
  const int table_boost,
  const int fit_nz,
  const py::object& source_z
)
{
  shape_physics(k, 1);
  shape_physics(p_linear0, 1);
  shape_physics(z, 1);
  shape_physics(p_nonlinear, 2);
  shape_physics(chi, 1);
  shape_physics(dz_dchi, 1);
  shape_physics(growth, 1);
  shape_physics(sources, 2);
  shape_physics(los_z, 1);
  // The optional axis is borrowed under the same float64/contiguous rule
  // as other production arrays. The C constructor copies it before return.
  physics_array source_axis;
  py::ssize_t nsource = z.size();
  if (!source_z.is_none()) {
    require_physics_interface(physics_array::check_(source_z),
      "source_z must be a contiguous float64 NumPy array");
    source_axis = py::reinterpret_borrow<physics_array>(source_z);
    shape_physics(source_axis, 1);
    finite_physics(source_axis);
    nsource = source_axis.size();
  }
  require_physics_interface(p_linear0.size() == k.size()
    && p_nonlinear.shape(0) == z.size()
    && p_nonlinear.shape(1) == k.size()
    && chi.size() == z.size() && dz_dchi.size() == z.size()
    && growth.size() == z.size() && sources.shape(1) == nsource,
    "model table shapes disagree");
  finite_physics(k);
  finite_physics(p_linear0);
  finite_physics(z);
  finite_physics(p_nonlinear);
  finite_physics(chi);
  finite_physics(dz_dchi);
  finite_physics(growth);
  finite_physics(sources);
  finite_physics(los_z);
  physics_input_3pcf input = {};
  input.nk = static_cast<int>(k.size());
  input.nz = static_cast<int>(z.size());
  input.nbin = static_cast<int>(sources.shape(0));
  input.nlos = static_cast<int>(los_z.size());
  input.kernel_nz = kernel_nz;
  input.source_readers = source_readers ? 1 : 0;
  input.table_boost = table_boost;
  input.fit_nz = fit_nz;
  input.k = k.data();
  input.p_linear0 = p_linear0.data();
  input.z = z.data();
  input.p_nonlinear = p_nonlinear.data();
  input.chi = chi.data();
  input.dz_dchi = dz_dchi.data();
  input.growth = growth.data();
  input.sources = sources.data();
  if (!source_z.is_none()) {
    input.nsource = static_cast<int>(nsource);
    input.source_z = source_axis.data();
  }
  input.los_z = los_z.data();
  input.omega_m = omega_m;
  input.omega_de = omega_de;
  input.w0 = w0;
  input.wa = wa;
  input.ns = ns;
  input.sigma8 = sigma8;
  return std::make_shared<physics_owner_3pcf>(input);
}


static std::shared_ptr<physics_owner_3pcf> model_notebook_physics(
  const physics_notebook_array& k,
  const physics_notebook_array& p_linear0,
  const physics_notebook_array& z,
  const physics_notebook_array& p_nonlinear,
  const physics_notebook_array& chi,
  const physics_notebook_array& dz_dchi,
  const physics_notebook_array& growth,
  const physics_notebook_array& sources,
  const double omega_m,
  const double omega_de,
  const double w0,
  const double wa,
  const double ns,
  const double sigma8,
  const physics_notebook_array& los_z,
  const int kernel_nz,
  const bool source_readers,
  const int table_boost,
  const int fit_nz,
  const py::object& source_z
)
{
  const arma::Col<double> wave = copy_column_physics(k);
  const arma::Col<double> linear = copy_column_physics(p_linear0);
  const arma::Col<double> redshift = copy_column_physics(z);
  const arma::Mat<double> nonlinear = copy_matrix_physics(p_nonlinear);
  const arma::Col<double> distance = copy_column_physics(chi);
  const arma::Col<double> jacobian = copy_column_physics(dz_dchi);
  const arma::Col<double> normalized_growth = copy_column_physics(growth);
  const arma::Mat<double> density = copy_matrix_physics(sources);
  const arma::Col<double> los = copy_column_physics(los_z);
  arma::Col<double> source_axis;
  if (!source_z.is_none()) {
    const auto supplied = py::cast<physics_notebook_array>(source_z);
    source_axis = copy_column_physics(supplied);
  }
  return physics_model_cpp_3pcf(wave, linear, redshift, nonlinear, distance,
    jacobian, normalized_growth, density, omega_m, omega_de, w0, wa, ns,
    sigma8, los, kernel_nz, source_readers, table_boost, fit_nz, source_axis);
}


static physics_array points_physics(
  const physics_owner_3pcf& owner,
  const physics_array& sides,
  const physics_array& redshifts,
  const physics_int_array& bins,
  const physics_ia_3pcf& parameters,
  const int operation,
  const int component,
  const bool remove_alignment
)
{
  shape_physics(sides, 2);
  finite_physics(sides);
  require_physics_interface(sides.shape(1) == 3,
    "sides must have shape [point,3]");
  if (operation == 2) {
    shape_physics(bins, 1);
    aligned_physics(bins, alignof(int));
    require_physics_interface(bins.size() == 3, "three bins are required");
  } else {
    shape_physics(redshifts, 1);
    finite_physics(redshifts);
    require_physics_interface(redshifts.size() == sides.shape(0),
      "redshift and triangle counts disagree");
  }
  const int count = static_cast<int>(sides.shape(0));
  const int ncomponent = operation == 1 || operation == 3 ? 19 : 1;
  physics_array output = operation == 1 || operation == 3
    ? physics_array({static_cast<py::ssize_t>(count),
      static_cast<py::ssize_t>(ncomponent)})
    : physics_array(count);
  const double* triangles = sides.data();
  double* result = output.mutable_data();
  if (operation == 3) {
    {
      py::gil_scoped_release release;
      physics_ia_same_shape_3pcf(owner.get(), &parameters, count, triangles,
        redshifts.data(), remove_alignment ? 1 : 0, result);
    }
    finite_physics(output);
    return output;
  }
  physics_projection_3pcf* projection = operation == 2
    ? physics_projector_create_3pcf(owner.get(), &parameters, bins.data())
    : nullptr;
  {
    py::gil_scoped_release release;
    #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
    for (int point=0; point<count; point++) {
      const double* triangle = triangles + static_cast<size_t>(point)*3;
      if (operation == 0) {
        result[point] = physics_matter_3pcf(owner.get(), triangle[0],
          triangle[1], triangle[2], redshifts.data()[point], component,
          parameters.fb);
      } else if (operation == 1) {
        physics_ia_bispectra_3pcf(owner.get(), &parameters, triangle[0],
          triangle[1], triangle[2], redshifts.data()[point],
          remove_alignment ? 1 : 0,
          result + static_cast<size_t>(point)*ncomponent);
      } else {
        result[point] = physics_projected_3pcf(projection, triangle[0],
          triangle[1], triangle[2], component);
      }
    }
  }
  physics_projector_destroy_3pcf(projection);
  finite_physics(output);
  return output;
}


static physics_array points_notebook_physics(
  const physics_owner_3pcf& owner,
  const physics_notebook_array& sides,
  const physics_notebook_array& redshifts,
  const physics_notebook_int& bins,
  const physics_ia_3pcf& parameters,
  const int operation,
  const int component,
  const bool remove_alignment
)
{
  const arma::Mat<double> triangles = copy_matrix_physics(sides);
  arma::Col<double> redshift;
  arma::Col<int> bin;
  if (operation == 2) {
    bin = copy_column_physics(bins);
  } else {
    redshift = copy_column_physics(redshifts);
  }
  const arma::Mat<double> result = physics_points_cpp_3pcf(owner, triangles,
    redshift, bin, parameters, operation, component, remove_alignment);
  physics_array output = operation == 1 || operation == 3
    ? physics_array({static_cast<py::ssize_t>(result.n_rows),
      static_cast<py::ssize_t>(result.n_cols)})
    : physics_array(result.n_rows);
  for (arma::uword row=0; row<result.n_rows; row++) {
    for (arma::uword column=0; column<result.n_cols; column++) {
      output.mutable_data()[row*result.n_cols + column] = result(row, column);
    }
  }
  finite_physics(output);
  return output;
}


static physics_array multipoles_physics(
  const physics_owner_3pcf& owner,
  const physics_array& ell,
  const physics_int_array& bins,
  const py::dict& parameters,
  const int lmax,
  const int nmu,
  const int component
)
{
  shape_physics(ell, 1);
  shape_physics(bins, 1);
  finite_physics(ell);
  aligned_physics(bins, alignof(int));
  require_physics_interface(bins.size() == 3 && lmax >= 0 && lmax < INT_MAX,
    "invalid multipole dimensions");
  const size_t nell = static_cast<size_t>(ell.size());
  require_physics_interface(nell <= SIZE_MAX/nell
    && static_cast<size_t>(lmax + 1) <= SIZE_MAX/(nell*nell)
    && static_cast<size_t>(lmax + 1)*nell*nell
    <= static_cast<size_t>(PY_SSIZE_T_MAX)/sizeof(double),
    "multipole allocation size overflows");
  physics_array result({static_cast<py::ssize_t>(lmax + 1),
    ell.size(), ell.size()});
  const physics_ia_3pcf ia = parameters_physics(parameters);
  {
    py::gil_scoped_release release;
    physics_multipoles_3pcf(owner.get(), &ia, bins.data(),
      static_cast<int>(ell.size()), ell.data(), lmax, nmu, component,
      result.mutable_data());
  }
  finite_physics(result);
  return result;
}


static physics_array multipoles_notebook_physics(
  const physics_owner_3pcf& owner,
  const physics_notebook_array& ell,
  const physics_notebook_int& bins,
  const py::dict& parameters,
  const int lmax,
  const int nmu,
  const int component
)
{
  const arma::Col<double> multipole = copy_column_physics(ell);
  const arma::Col<int> bin = copy_column_physics(bins);
  const physics_ia_3pcf ia = parameters_physics(parameters);
  const arma::Cube<double> result = physics_multipoles_cpp_3pcf(owner,
    multipole, bin, ia, lmax, nmu, component);
  physics_array output({static_cast<py::ssize_t>(result.n_rows),
    static_cast<py::ssize_t>(result.n_cols),
    static_cast<py::ssize_t>(result.n_slices)});
  for (arma::uword order=0; order<result.n_rows; order++) {
    for (arma::uword first=0; first<result.n_cols; first++) {
      for (arma::uword second=0; second<result.n_slices; second++) {
        output.mutable_data()[(order*result.n_cols + first)*result.n_slices
          + second] = result(order, first, second);
      }
    }
  }
  finite_physics(output);
  return output;
}


static physics_array aperture_full_physics(
  const physics_owner_3pcf& owner,
  const physics_array& ell,
  const physics_array& dell,
  const physics_array& phi,
  const physics_array& dphi,
  const physics_array& filters,
  const physics_int_array& bins,
  const py::dict& parameters
)
{
  shape_physics(ell, 1);
  shape_physics(dell, 1);
  shape_physics(phi, 1);
  shape_physics(dphi, 1);
  shape_physics(filters, 2);
  shape_physics(bins, 1);
  finite_physics(ell);
  finite_physics(dell);
  finite_physics(phi);
  finite_physics(dphi);
  finite_physics(filters);
  aligned_physics(bins, alignof(int));
  require_physics_interface(ell.size() == dell.size()
    && phi.size() == dphi.size() && filters.shape(1) == 3 && bins.size() == 3,
    "full-aperture input shapes disagree");
  const physics_ia_3pcf ia = parameters_physics(parameters);
  physics_array output(filters.shape(0));
  {
    py::gil_scoped_release release;
    physics_aperture_full_3pcf(owner.get(), &ia, bins.data(),
      static_cast<int>(ell.size()), ell.data(), dell.data(),
      static_cast<int>(phi.size()), phi.data(), dphi.data(),
      static_cast<int>(filters.shape(0)), filters.data(),
      output.mutable_data());
  }
  finite_physics(output);
  return output;
}


static physics_array aperture_full_notebook_physics(
  const physics_owner_3pcf& owner,
  const physics_notebook_array& ell,
  const physics_notebook_array& dell,
  const physics_notebook_array& phi,
  const physics_notebook_array& dphi,
  const physics_notebook_array& filters,
  const physics_notebook_int& bins,
  const py::dict& parameters
)
{
  const arma::Col<double> multipole = copy_column_physics(ell);
  const arma::Col<double> multipole_weights = copy_column_physics(dell);
  const arma::Col<double> angles = copy_column_physics(phi);
  const arma::Col<double> angle_weights = copy_column_physics(dphi);
  const arma::Mat<double> radii = copy_matrix_physics(filters);
  const arma::Col<int> source_bins = copy_column_physics(bins);
  const physics_ia_3pcf ia = parameters_physics(parameters);
  const arma::Col<double> result = physics_aperture_full_cpp_3pcf(owner,
    multipole, multipole_weights, angles, angle_weights, radii,
    source_bins, ia);
  physics_array output(result.n_elem);
  std::memcpy(output.mutable_data(), result.memptr(),
    static_cast<size_t>(result.n_elem)*sizeof(double));
  finite_physics(output);
  return output;
}


static physics_array aperture_grid_physics(
  const physics_grid_owner_3pcf& owner,
  const physics_array& ell,
  const physics_array& dell,
  const physics_array& phi,
  const physics_array& dphi,
  const physics_array& filters
)
{
  shape_physics(ell, 1);
  shape_physics(dell, 1);
  shape_physics(phi, 1);
  shape_physics(dphi, 1);
  shape_physics(filters, 2);
  finite_physics(ell);
  finite_physics(dell);
  finite_physics(phi);
  finite_physics(dphi);
  finite_physics(filters);
  require_physics_interface(ell.size() == dell.size()
    && phi.size() == dphi.size() && filters.shape(1) == 3,
    "grid aperture input shapes disagree");
  physics_array output(filters.shape(0));
  {
    py::gil_scoped_release release;
    physics_grid_aperture_full_3pcf(owner.get(), static_cast<int>(ell.size()),
      ell.data(), dell.data(), static_cast<int>(phi.size()), phi.data(),
      dphi.data(), static_cast<int>(filters.shape(0)), filters.data(),
      output.mutable_data());
  }
  finite_physics(output);
  return output;
}


static physics_array projected_grid_physics(
  const physics_grid_owner_3pcf& owner,
  const physics_array& sides,
  const int component
)
{
  shape_physics(sides, 2);
  finite_physics(sides);
  require_physics_interface(sides.shape(1) == 3,
    "projected grid sides must have shape [point,3]");
  const int count = static_cast<int>(sides.shape(0));
  physics_array output(count);
  const double* triangles = sides.data();
  double* values = output.mutable_data();
  {
    py::gil_scoped_release release;
    #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
    for (int row=0; row<count; row++) {
      const double* triangle = triangles + static_cast<size_t>(row)*3;
      values[row] = physics_grid_projected_3pcf(owner.get(), triangle[0],
        triangle[1], triangle[2], component);
    }
  }
  finite_physics(output);
  return output;
}


static physics_array multipoles_grid_physics(
  const physics_grid_owner_3pcf& owner,
  const physics_array& ell,
  const int component
)
{
  shape_physics(ell, 1);
  finite_physics(ell);
  const size_t nell = static_cast<size_t>(ell.size());
  const size_t norder = static_cast<size_t>(owner.lmax() + 1);
  require_physics_interface(nell <= SIZE_MAX/nell
    && norder <= SIZE_MAX/(nell*nell)
    && norder*nell*nell <= static_cast<size_t>(PY_SSIZE_T_MAX)/sizeof(double),
    "interpolated-multipole output size overflows");
  physics_array output({static_cast<py::ssize_t>(norder),
    ell.size(), ell.size()});
  {
    py::gil_scoped_release release;
    physics_grid_multipoles_3pcf(owner.get(), static_cast<int>(nell),
      ell.data(), component, output.mutable_data());
  }
  finite_physics(output);
  return output;
}


const physics_model_3pcf* borrow_physics_model_3pcf(py::handle owner)
{
  return py::cast<const physics_owner_3pcf&>(owner).get();
}


const physics_grid_3pcf* borrow_physics_grid_3pcf(py::handle owner)
{
  return py::cast<const physics_grid_owner_3pcf&>(owner).get();
}


void bind_threepcf_physics(py::module_& module)
{
  py::class_<physics_grid_owner_3pcf,
    std::shared_ptr<physics_grid_owner_3pcf>> grid(module, "ProjectionGrid",
    "Immutable RUV baseline, ordered PUV residual and two-chart b_L tables.");
  grid.def("projected", &projected_grid_physics,
    py::arg("sides").noconvert(), py::arg("component") = 0);
  grid.def("multipoles", &multipoles_grid_physics,
    py::arg("ell").noconvert(), py::arg("component") = 0);
  grid.def("projected_notebook", [](const physics_grid_owner_3pcf& owner,
      const physics_notebook_array& sides, int component) {
    const arma::Mat<double> triangles = copy_matrix_physics(sides);
    const arma::Col<double> values = physics_grid_projected_cpp_3pcf(owner,
      triangles, component);
    physics_array output(values.n_elem);
    std::memcpy(output.mutable_data(), values.memptr(),
      static_cast<size_t>(values.n_elem)*sizeof(double));
    finite_physics(output);
    return output;
  }, py::arg("sides"), py::arg("component") = 0);
  grid.def("multipoles_notebook", [](const physics_grid_owner_3pcf& owner,
      const physics_notebook_array& ell, int component) {
    const arma::Col<double> nodes = copy_column_physics(ell);
    const arma::Cube<double> values = physics_grid_multipoles_cpp_3pcf(owner,
      nodes, component);
    physics_array output({static_cast<py::ssize_t>(values.n_rows),
      static_cast<py::ssize_t>(values.n_cols),
      static_cast<py::ssize_t>(values.n_slices)});
    for (arma::uword order=0; order<values.n_rows; order++) {
      for (arma::uword first=0; first<values.n_cols; first++) {
        for (arma::uword second=0; second<values.n_slices; second++) {
          output.mutable_data()[(order*values.n_cols + first)*values.n_slices
            + second] = values(order, first, second);
        }
      }
    }
    finite_physics(output);
    return output;
  }, py::arg("ell"), py::arg("component") = 0);
  grid.def("aperture_full", &aperture_grid_physics,
    py::arg("ell").noconvert(), py::arg("dell").noconvert(),
    py::arg("phi").noconvert(), py::arg("dphi").noconvert(),
    py::arg("filters").noconvert(),
    "Full-filter Fourier mean from prepared ordered B; no b_L needed.");
  grid.def("aperture_full_notebook", [](const physics_grid_owner_3pcf& owner,
      const physics_notebook_array& ell, const physics_notebook_array& dell,
      const physics_notebook_array& phi, const physics_notebook_array& dphi,
      const physics_notebook_array& filters) {
    const arma::Col<double> values = physics_grid_aperture_cpp_3pcf(owner,
      copy_column_physics(ell), copy_column_physics(dell),
      copy_column_physics(phi), copy_column_physics(dphi),
      copy_matrix_physics(filters));
    physics_array output(values.n_elem);
    std::memcpy(output.mutable_data(), values.memptr(),
      static_cast<size_t>(values.n_elem)*sizeof(double));
    finite_physics(output);
    return output;
  }, py::arg("ell"), py::arg("dell"), py::arg("phi"), py::arg("dphi"),
    py::arg("filters"));
  grid.def("axes", [](const physics_grid_owner_3pcf& owner) {
    py::dict result;
    const char* names[8] = {"log_r", "log_u", "v", "log_p", "pu",
      "log_ell", "log_psi", "mu"};
    for (int axis=0; axis<8; axis++) {
      int count;
      const double* values;
      physics_grid_axis_3pcf(owner.get(), axis, &count, &values);
      physics_array output(count);
      std::memcpy(output.mutable_data(), values,
        static_cast<size_t>(count)*sizeof(double));
      result[names[axis]] = output;
    }
    return result;
  }, "Copies of actual interpolation coordinates, including endpoint mu.");
  py::class_<physics_owner_3pcf, std::shared_ptr<physics_owner_3pcf>> model(
    module, "PhysicsModel",
    "Owned BiHalofit/TATT model from k[h/Mpc], P[(Mpc/h)^3], chi[Mpc/h]. "
    "All physical arrays are copied. Recreate after changing their values.");
  model.def(py::init(&model_physics),
    py::arg("k").noconvert(),
    py::arg("p_linear0").noconvert(),
    py::arg("z").noconvert(),
    py::arg("p_nonlinear").noconvert(),
    py::arg("chi").noconvert(),
    py::arg("dz_dchi").noconvert(),
    py::arg("growth").noconvert(),
    py::arg("sources").noconvert(),
    py::arg("omega_m"),
    py::arg("omega_de"),
    py::arg("w0"),
    py::arg("wa"),
    py::arg("ns"),
    py::arg("sigma8"),
    py::arg("los_z").noconvert(),
    py::arg("kernel_nz") = 101,
    py::arg("source_readers") = false,
    py::arg("table_boost") = 8,
    py::arg("fit_nz") = 33,
    py::arg("source_z") = py::none());
  module.def("physics_model_notebook", &model_notebook_physics,
    py::arg("k"),
    py::arg("p_linear0"),
    py::arg("z"),
    py::arg("p_nonlinear"),
    py::arg("chi"),
    py::arg("dz_dchi"),
    py::arg("growth"),
    py::arg("sources"),
    py::arg("omega_m"),
    py::arg("omega_de"),
    py::arg("w0"),
    py::arg("wa"),
    py::arg("ns"),
    py::arg("sigma8"),
    py::arg("los_z"),
    py::arg("kernel_nz") = 101,
    py::arg("source_readers") = false,
    py::arg("table_boost") = 8,
    py::arg("fit_nz") = 33,
    py::arg("source_z") = py::none(),
    "Copy sliced or Fortran-order arrays through the Armadillo adapter.");

  model.def("matter", [](const physics_owner_3pcf& owner,
      const physics_array& sides, const physics_array& redshifts,
      int component, double fb) {
    physics_ia_3pcf parameters;
    physics_default_ia_3pcf(&parameters);
    parameters.fb = fb;
    return points_physics(owner, sides, redshifts, physics_int_array(),
      parameters, 0, component, false);
  }, py::arg("sides").noconvert(), py::arg("redshifts").noconvert(),
    py::arg("component") = 0, py::arg("fb") = 0.0,
    "B_matter[point]: component 0=BiHalofit,1=tree,2=Bh1,3=Bh3.");
  model.def("ia", [](const physics_owner_3pcf& owner,
      const physics_array& sides, const physics_array& redshifts,
      const py::dict& parameters, bool remove_alignment) {
    return points_physics(owner, sides, redshifts, physics_int_array(),
      parameters_physics(parameters), 1, 0, remove_alignment);
  }, py::arg("sides").noconvert(), py::arg("redshifts").noconvert(),
    py::arg("parameters") = py::dict(), py::arg("remove_alignment") = false,
    "Return [point,19] active-source IA terms in the documented leg order.");
  model.def("ia_same_shape", [](const physics_owner_3pcf& owner,
      const physics_array& sides, const physics_array& redshifts,
      const py::dict& parameters, bool remove_alignment) {
    return points_physics(owner, sides, redshifts, physics_int_array(),
      parameters_physics(parameters), 3, 0, remove_alignment);
  }, py::arg("sides").noconvert(), py::arg("redshifts").noconvert(),
    py::arg("parameters") = py::dict(), py::arg("remove_alignment") = false,
    "All 19 IA terms with shared geometry; side ratios must be constant.");
  model.def("ia_same_shape_notebook", [](const physics_owner_3pcf& owner,
      const physics_notebook_array& sides,
      const physics_notebook_array& redshifts, const py::dict& parameters,
      bool remove_alignment) {
    return points_notebook_physics(owner, sides, redshifts,
      physics_notebook_int(), parameters_physics(parameters), 3, 0,
      remove_alignment);
  }, py::arg("sides"), py::arg("redshifts"),
    py::arg("parameters") = py::dict(), py::arg("remove_alignment") = false);
  model.def("projected", [](const physics_owner_3pcf& owner,
      const physics_array& sides, const physics_int_array& bins,
      const py::dict& parameters, int component) {
    return points_physics(owner, sides, physics_array(), bins,
      parameters_physics(parameters), 2, component, false);
  }, py::arg("sides").noconvert(), py::arg("bins").noconvert(),
    py::arg("parameters") = py::dict(), py::arg("component") = 0,
    "Unshifted flat-Limber E bispectrum; 0=total,1=NLA baseline,2=residual.");
  model.def("multipoles", &multipoles_physics,
    py::arg("ell").noconvert(), py::arg("bins").noconvert(),
    py::arg("parameters"), py::arg("lmax"), py::arg("nmu"),
    py::arg("component") = 0,
    "Direct LOS/Gauss-Legendre b_L[L,ell1,ell2]; refine both node grids.");
  model.def("multipoles_notebook", &multipoles_notebook_physics,
    py::arg("ell"), py::arg("bins"), py::arg("parameters"),
    py::arg("lmax"), py::arg("nmu"), py::arg("component") = 0);
  model.def("aperture_full", &aperture_full_physics,
    py::arg("ell").noconvert(), py::arg("dell").noconvert(),
    py::arg("phi").noconvert(), py::arg("dphi").noconvert(),
    py::arg("filters").noconvert(), py::arg("bins").noconvert(),
    py::arg("parameters") = py::dict(),
    "Full-filter Map3 from explicit Fourier quadratures, angles in radians.");
  model.def("aperture_full_notebook", &aperture_full_notebook_physics,
    py::arg("ell"), py::arg("dell"), py::arg("phi"), py::arg("dphi"),
    py::arg("filters"), py::arg("bins"), py::arg("parameters") = py::dict());
  model.def("projection_grid", [](
      const std::shared_ptr<physics_owner_3pcf>& owner,
      const physics_int_array& bins, const py::dict& parameters,
      const py::dict& settings) {
    shape_physics(bins, 1);
    aligned_physics(bins, alignof(int));
    require_physics_interface(bins.size() == 3,
      "three source bins are required");
    const physics_ia_3pcf ia = parameters_physics(parameters);
    const physics_grid_settings_3pcf configuration
      = settings_physics_grid(settings);
    py::gil_scoped_release release;
    return std::make_shared<physics_grid_owner_3pcf>(owner, ia, bins.data(),
      configuration);
  }, py::arg("bins").noconvert(), py::arg("parameters") = py::dict(),
    py::arg("settings") = py::dict(),
    "Prepare projected-B and b_L tables with explicit resolution settings.");
  model.def("projection_grids", [](const physics_owner_3pcf& owner,
      const physics_int_array& bins, const py::dict& parameters,
      const py::dict& settings) {
    shape_physics(bins, 2);
    aligned_physics(bins, alignof(int));
    require_physics_interface(bins.shape(1) == 3,
      "batch bins must have shape [ntriplet,3]");
    const physics_ia_3pcf ia = parameters_physics(parameters);
    const physics_grid_settings_3pcf configuration
      = settings_physics_grid(settings);
    std::vector<physics_grid_3pcf*> prepared(bins.shape(0));
    {
      py::gil_scoped_release release;
      physics_grid_create_many_3pcf(owner.get(), &ia,
        static_cast<int>(bins.shape(0)), bins.data(), &configuration,
        prepared.data());
    }
    py::list result;
    for (physics_grid_3pcf* table : prepared) {
      result.append(std::make_shared<physics_grid_owner_3pcf>(table,
        configuration.lmax));
    }
    return result;
  }, py::arg("bins").noconvert(), py::arg("parameters") = py::dict(),
    py::arg("settings") = py::dict(),
    "Prepare ordered triplets while sharing all triangle/redshift physics.");
  model.def("projection_grids_notebook", [](const physics_owner_3pcf& owner,
      const physics_notebook_int& bins, const py::dict& parameters,
      const py::dict& settings) {
    const arma::Mat<int> sources = copy_matrix_physics(bins);
    const auto prepared = physics_grids_cpp_3pcf(owner, sources,
      parameters_physics(parameters), settings_physics_grid(settings));
    py::list result;
    for (const auto& table : prepared) {
      result.append(table);
    }
    return result;
  }, py::arg("bins"), py::arg("parameters") = py::dict(),
    py::arg("settings") = py::dict());
  model.def("projection_grid_notebook", [](
      const std::shared_ptr<physics_owner_3pcf>& owner,
      const physics_notebook_int& bins, const py::dict& parameters,
      const py::dict& settings) {
    const arma::Col<int> source = copy_column_physics(bins);
    require_physics_interface(source.n_elem == 3, "three bins are required");
    const physics_ia_3pcf ia = parameters_physics(parameters);
    const physics_grid_settings_3pcf configuration
      = settings_physics_grid(settings);
    return std::make_shared<physics_grid_owner_3pcf>(owner, ia, source.memptr(),
      configuration);
  }, py::arg("bins"), py::arg("parameters") = py::dict(),
    py::arg("settings") = py::dict());
  model.def("matter_notebook", [](const physics_owner_3pcf& owner,
      const physics_notebook_array& sides,
      const physics_notebook_array& redshifts, int component, double fb) {
    physics_ia_3pcf parameters;
    physics_default_ia_3pcf(&parameters);
    parameters.fb = fb;
    return points_notebook_physics(owner, sides, redshifts,
      physics_notebook_int(), parameters, 0, component, false);
  }, py::arg("sides"), py::arg("redshifts"), py::arg("component") = 0,
    py::arg("fb") = 0.0);
  model.def("ia_notebook", [](const physics_owner_3pcf& owner,
      const physics_notebook_array& sides,
      const physics_notebook_array& redshifts,
      const py::dict& parameters, bool remove_alignment) {
    return points_notebook_physics(owner, sides, redshifts,
      physics_notebook_int(), parameters_physics(parameters), 1, 0,
      remove_alignment);
  }, py::arg("sides"), py::arg("redshifts"),
    py::arg("parameters") = py::dict(), py::arg("remove_alignment") = false);
  model.def("projected_notebook", [](const physics_owner_3pcf& owner,
      const physics_notebook_array& sides, const physics_notebook_int& bins,
      const py::dict& parameters, int component) {
    return points_notebook_physics(owner, sides, physics_notebook_array(),
      bins, parameters_physics(parameters), 2, component, false);
  }, py::arg("sides"), py::arg("bins"),
    py::arg("parameters") = py::dict(), py::arg("component") = 0);

  model.def("power", [](const physics_owner_3pcf& owner,
      const physics_array& k, const physics_array& z, int mode) {
    shape_physics(k, 1);
    shape_physics(z, 1);
    finite_physics(k);
    finite_physics(z);
    require_physics_interface(k.size() == z.size(),
      "power k and z arrays must have equal length");
    physics_array result(k.size());
    for (py::ssize_t index=0; index<k.size(); index++) {
      result.mutable_data()[index] = physics_power_3pcf(owner.get(),
        k.data()[index], z.data()[index], mode);
    }
    finite_physics(result);
    return result;
  }, py::arg("k").noconvert(), py::arg("z").noconvert(),
    py::arg("mode") = 2,
    "P modes: 0=linear D^2,1=Pnl(k,0)D^2,2=supplied nonlinear evolution.");
  model.def("nla_power", [](const physics_owner_3pcf& owner,
      const physics_array& ell, const physics_int_array& bins,
      const py::dict& parameters) {
    shape_physics(ell, 1);
    shape_physics(bins, 1);
    finite_physics(ell);
    aligned_physics(bins, alignof(int));
    require_physics_interface(bins.size() == 2, "two power bins are required");
    const physics_ia_3pcf ia = parameters_physics(parameters);
    physics_array result(ell.size());
    {
      py::gil_scoped_release release;
      physics_nla_power_batch_3pcf(owner.get(), &ia, bins.data()[0],
        bins.data()[1], static_cast<int>(ell.size()), ell.data(),
        result.mutable_data());
    }
    finite_physics(result);
    return result;
  }, py::arg("ell").noconvert(), py::arg("bins").noconvert(),
    py::arg("parameters") = py::dict(),
    "Shear C_ell with deterministic NLA; rejects TATT loops and baryons.");
  model.def("nla_power_notebook", [](const physics_owner_3pcf& owner,
      const physics_notebook_array& ell, const physics_notebook_int& bins,
      const py::dict& parameters) {
    const arma::Col<double> values = physics_nla_power_cpp_3pcf(owner,
      copy_column_physics(ell), copy_column_physics(bins),
      parameters_physics(parameters));
    physics_array result(values.n_elem);
    std::memcpy(result.mutable_data(), values.memptr(),
      static_cast<size_t>(values.n_elem)*sizeof(double));
    finite_physics(result);
    return result;
  }, py::arg("ell"), py::arg("bins"), py::arg("parameters") = py::dict());
  model.def("radial", [](const physics_owner_3pcf& owner) {
    int nbin;
    int nlos;
    const double* z;
    const double* chi;
    const double* dchi;
    const double* lens;
    const double* shape;
    physics_radial_3pcf(owner.get(), &nbin, &nlos, &z, &chi, &dchi,
      &lens, &shape);
    py::dict result;
    const char* names[5] = {"z", "chi", "dchi", "lens", "shape"};
    const double* arrays[5] = {z, chi, dchi, lens, shape};
    for (int index=0; index<5; index++) {
      physics_array values = index < 3 ? physics_array(nlos)
        : physics_array({static_cast<py::ssize_t>(nbin),
          static_cast<py::ssize_t>(nlos)});
      std::memcpy(values.mutable_data(), arrays[index],
        static_cast<size_t>(values.size())*sizeof(double));
      finite_physics(values);
      result[names[index]] = values;
    }
    return result;
  }, "Copies of LOS nodes, dchi weights, g and source Wg; no writable cache.");
  module.attr("production").attr("PhysicsModel") = module.attr("PhysicsModel");
  module.attr("notebook").attr("PhysicsModel")
    = module.attr("physics_model_notebook");
}

}
