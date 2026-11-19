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

// ---------------------------------------------------------------------------
// physics_interface.cpp: the Python adapters of the 3PCF physics model.
// pybind11 is the C++ library that lets Python call C++ functions and
// hold C++ objects; this file registers the PhysicsModel and
// ProjectionGrid classes and their methods, while physics_3pcf.h and
// physics_grid_3pcf.h hold every piece of physics they reach.
//
// Two routes lead to the same C model:
//
//   1. Production methods borrow the caller's NumPy buffers: they read
//      the array's memory in place (a view - the same bytes, no copy),
//      so the array must already be contiguous float64. Every
//      production argument carries .noconvert(), which tells pybind11
//      to refuse an array of another dtype or layout instead of
//      converting it silently; a silent conversion would hide a copy
//      and could truncate values. The borrow ends with the call:
//      model creation copies the physical values into immutable C
//      tables before it returns.
//
//   2. Notebook factories and *_notebook methods accept sliced or
//      Fortran-order arrays, copy them element by element into
//      Armadillo storage, and call the twin routines in
//      physics_wrapper.cpp. This route trades the cost of a copy for
//      convenience in interactive work.
//
// The layout conversions here are hand-written as owned copies with
// explicit loops, by design: the student can see exactly which values
// cross the Python/C boundary and who owns each buffer. CARMA remains
// the NumPy/Armadillo conversion library elsewhere in the core.
//
// Storage orders. NumPy arrays and the C tables are row-major: the
// last index runs fastest in memory. Armadillo is column-major: the
// first index runs fastest. Each copy loop below says which order its
// two sides use.
//
// This file computes no interpolation, normalization, projection or
// cosmology; its routines check shapes, arrange storage and manage
// lifetime.
// ---------------------------------------------------------------------------

using physics_array = py::array_t<double, py::array::c_style>;
using physics_int_array = py::array_t<int, py::array::c_style>;
using physics_notebook_array = py::array_t<double, py::array::forcecast>;
using physics_notebook_int = py::array;


// ---------------------------------------------------------------------------
// Check an array's rank and axis sizes before any allocation uses them.
// Each axis must be positive and fit a C int. The division guards
// protect against size_t wrap-around: unsigned arithmetic wraps at
// SIZE_MAX instead of failing, so a huge shape could silently produce
// a small product. Checking length <= SIZE_MAX/product before each
// multiplication proves the running product stays representable, and
// the last guard bounds the byte count (product times the element
// size) the same way.
//
// Parameters:
//   values - borrowed NumPy array of any dtype
//   rank   - required number of axes
// ---------------------------------------------------------------------------
template <typename Array>
static void shape_physics(
  const Array& values, // input array whose rank and dimension sizes are checked
  const int rank // required number of array axes
)
{
  if (!(values.ndim() == rank)) {
    spdlog::critical("3pcf physics interface: wrong input rank");
    std::exit(1);
  }

  size_t product = 1;
  for (int axis=0; axis<rank; axis++) {
    if (!(0 < values.shape(axis)
          && INT_MAX >= values.shape(axis))) {
      spdlog::critical("3pcf physics interface: invalid input dimension");
      std::exit(1);
    }
    const size_t length = static_cast<size_t>(values.shape(axis));
    if (!(length <= SIZE_MAX/product)) {
      spdlog::critical("3pcf physics interface: input shape product overflows");
      std::exit(1);
    }
    product *= length;
  }

  if (!(product <= SIZE_MAX/values.itemsize())) {
    spdlog::critical("3pcf physics interface: input byte count overflows");
    std::exit(1);
  }
}


// ---------------------------------------------------------------------------
// C-contiguous describes strides, not pointer alignment. A NumPy buffer
// that starts at a one-byte offset into a larger allocation can be
// contiguous and still impossible to read through a C double*, which
// requires an address that is a multiple of the element alignment.
// Reject such a buffer before the first typed dereference.
//
// Parameters:
//   values    - borrowed array, before any typed data access
//   alignment - alignof the C element type
// ---------------------------------------------------------------------------
static void aligned_physics(
  // input buffer whose starting address must satisfy alignment
  const py::array& values,
  const size_t alignment // required byte alignment for typed C reads
)
{
  const auto address = reinterpret_cast<std::uintptr_t>(values.data());
  if (!(0 == address%alignment)) {
    spdlog::critical("3pcf physics interface: production input buffer is "
      "not aligned for its element type");
    std::exit(1);
  }
}


// ---------------------------------------------------------------------------
// Purpose of this check: keep NaN and Inf out of the immutable C
// tables, and out of Python. The adapter runs it on every borrowed
// input before the C side reads it, and again on every output after
// the GIL scope closes, so a NaN produced inside C cannot reach Python
// silently. The alignment check comes first because this scan itself
// reads the data through a double pointer.
// ---------------------------------------------------------------------------
static void finite_physics(
  // double array scanned for NaN or infinite entries
  const physics_array& values
)
{
  aligned_physics(values, alignof(double));

  for (py::ssize_t index=0; index<values.size(); index++) {
    if (!(std::isfinite(values.data()[index]))) {
      spdlog::critical("3pcf physics interface: input or result contains a "
        "nonfinite value");
      std::exit(1);
    }
  }
}


// ---------------------------------------------------------------------------
// Copy a one-dimensional NumPy array into a fresh Armadillo column the
// notebook adapter owns. The stride walk accepts sliced inputs, and
// every element must be finite.
//
// Returns:
//   owned arma::Col of length shape(0)
// ---------------------------------------------------------------------------
template <typename Number>
static arma::Col<Number> copy_column_physics(
  // NumPy column copied into an independently owned Armadillo vector
  const py::array_t<Number, py::array::forcecast>& values
)
{
  shape_physics(values, 1);

  arma::Col<Number> result(values.shape(0));
  const char* base = static_cast<const char*>(values.request().ptr);

  for (py::ssize_t index=0; index<values.shape(0); index++) {
    // std::memcpy copies the bytes of one element without forming a
    // misaligned typed pointer, so unaligned buffers and negative
    // slice strides both read correctly.
    Number value;
    std::memcpy(&value, base + index*values.strides(0), sizeof(Number));
    if (!(std::isfinite(static_cast<double>(value)))) {
      spdlog::critical("3pcf physics interface: notebook input contains a "
        "nonfinite value");
      std::exit(1);
    }
    result(index) = value;
  }

  return result;
}


// ---------------------------------------------------------------------------
// Copy a two-dimensional NumPy array into a fresh Armadillo matrix,
// with the same stride walk and finiteness rule as copy_column_physics.
// ---------------------------------------------------------------------------
template <typename Number>
static arma::Mat<Number> copy_matrix_physics(
  // NumPy matrix copied into independently owned Armadillo storage
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
      if (!(std::isfinite(value))) {
        spdlog::critical("3pcf physics interface: notebook matrix contains "
          "a nonfinite value");
        std::exit(1);
      }
      result(row, column) = value;
    }
  }

  return result;
}


// ---------------------------------------------------------------------------
// Inspect source-bin labels before NumPy converts their dtype. A
// force-cast parameter type would already have truncated fractional
// values or wrapped wide integers by the time this function received
// them, so the function takes the raw array and checks it first: the
// labels must carry an integer dtype, may arrive sliced, and must fit
// in int32 before the copy into Armadillo storage. The aperture
// adapter in threepcf_interface.cpp applies the same rule.
//
// Returns:
//   the same data as an int32 array, ready for copy_column/matrix
// ---------------------------------------------------------------------------
static py::array_t<int, py::array::forcecast> integer_notebook_physics(
  // integer labels checked before conversion to C int
  const py::array& values,
  const int rank // required number of array axes
)
{
  shape_physics(values, rank);

  if (!('i' == values.dtype().kind()
        || 'u' == values.dtype().kind())) {
    spdlog::critical("3pcf physics interface: notebook source bins must "
      "have integer dtype");
    std::exit(1);
  }

  const double minimum = py::cast<double>(values.attr("min")());
  const double maximum = py::cast<double>(values.attr("max")());
  if (!(INT_MIN <= minimum && INT_MAX >= maximum)) {
    spdlog::critical("3pcf physics interface: notebook source bins exceed "
      "int32 range");
    std::exit(1);
  }

  return py::array_t<int, py::array::forcecast>(values);
}


// Integer overload: check the dtype first, then copy the column.
static arma::Col<int> copy_column_physics(
  const py::array& values // integer labels copied into an owned column
)
{
  const auto integers = integer_notebook_physics(values, 1);
  return copy_column_physics<int>(integers);
}


// Integer overload: check the dtype first, then copy the matrix.
static arma::Mat<int> copy_matrix_physics(
  const py::array& values // integer labels copied into an owned matrix
)
{
  const auto integers = integer_notebook_physics(values, 2);
  return copy_matrix_physics<int>(integers);
}


// ---------------------------------------------------------------------------
// Read the IA (intrinsic alignment) parameter dictionary into the C
// struct physics_ia_3pcf, starting from its defaults. The keys follow
// the TATT model - tidal alignment plus tidal torquing, the model
// introduced by Blazek et al. (arXiv:1708.09247) - extended by the
// velocity-shear operator of Gomes et al. (arXiv:2601.09133):
//
//   a1, eta1 - the tidal-alignment amplitude A_1 and the exponent of
//              its (1+z)/(1+z_pivot) redshift power law
//   a2, eta2 - the tidal-torquing amplitude A_2 and its exponent
//   bta      - the density-weighting bias b_TA
//   ct       - the velocity-shear amplitude C_t of Gomes et al.
//              (arXiv:2601.09133); ct_mode in physics_3pcf.h says how
//              it scales
//   zpivot   - the pivot redshift of both power laws
//   c1rho    - the normalization product C_1 rho_crit that converts an
//              alignment amplitude into a power-spectrum weight
//   fb       - the baryon-feedback amplitude of the BiHalofit matter
//              bispectrum (Takahashi et al., arXiv:1911.07886)
//   feedback_amplitude - the shared log-response amplitude of the
//              paired hydro/DMO tables; zero keeps dark matter only
//
// physics_3pcf.h documents the integer modes (power_mode, ct_mode,
// suppress_only, operator_component, intrinsic_legs) field by field.
// An unknown key stops the program: a silently ignored typo would run
// the wrong physics.
//
// Parameters:
//   values - Python dictionary; absent keys keep their defaults
//
// Returns:
//   the complete physics_ia_3pcf parameter struct, by value
// ---------------------------------------------------------------------------
static physics_ia_3pcf parameters_physics(
  const py::dict& values // intrinsic-alignment and feedback settings
)
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
    } else if (key == "feedback_amplitude") {
      parameters.feedback_amplitude = py::cast<double>(item.second);
    } else {
      spdlog::critical("3pcf physics interface: unknown IA dictionary key");
      std::exit(1);
    }
  }

  return parameters;
}


// ---------------------------------------------------------------------------
// Read the projection-grid resolution dictionary into the C struct
// physics_grid_settings_3pcf, starting from its defaults. Every key
// names one field of that struct - the angular range ell_min/ell_max,
// the node counts of each table axis, the interpolation-chart choices
// and the Legendre cutoff lmax - and physics_grid_3pcf.h documents
// them field by field. Real-valued settings must be finite. Counts
// must arrive as Python integers, because accepting a float such as
// 64.5 would silently drop its fraction. An unknown key stops the
// program.
//
// Parameters:
//   values - Python dictionary; absent keys keep their defaults
//
// Returns:
//   the complete physics_grid_settings_3pcf struct, by value
// ---------------------------------------------------------------------------
static physics_grid_settings_3pcf settings_physics_grid(
  const py::dict& values // grid bounds, node counts and angular cutoffs
)
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
      if (!(std::isfinite(*real.at(name)))) {
        spdlog::critical("3pcf physics interface: nonfinite interpolation "
          "setting");
        std::exit(1);
      }
    } else {
      if (!(0 != integer.count(name))) {
        spdlog::critical("3pcf physics interface: unknown interpolation "
          "setting");
        std::exit(1);
      }
      if (!(py::isinstance<py::int_>(item.second))) {
        spdlog::critical("3pcf physics interface: interpolation counts must "
          "be integers");
        std::exit(1);
      }
      *integer.at(name) = py::cast<int>(item.second);
    }
  }

  return settings;
}


// ---------------------------------------------------------------------------
// Production model factory. It borrows the caller's NumPy tables,
// validates them, and hands their pointers to physics_create_3pcf,
// which copies every physical value into an immutable C model.
// physics_3pcf.h defines the table contract and the units: k in h/Mpc,
// chi in Mpc/h, P in (Mpc/h)^3, a redshift grid that includes zero.
//
// Parameters:
//   k, p_linear0      - wavenumbers and the z=0 linear power on them
//   z, p_nonlinear    - redshift grid and the [nz,nk] nonlinear power
//   chi, dz_dchi, growth - comoving distance, dz/dchi and the linear
//                       growth factor, all on the z grid
//   sources           - [nbin,nsource] source densities per unit z
//   omega_m..sigma8   - background and amplitude scalars
//   los_z             - line-of-sight integration nodes
//   kernel_nz, source_readers, table_boost, fit_nz - preparation
//                       controls (physics_3pcf.h)
//   source_z          - optional separate source redshift axis
//   feedback          - optional paired hydro/DMO response dictionary
//   pixel_window      - optional W(ell) samples at integer ell
//
// Returns:
//   shared owner of the C model; Python keeps it alive
// ---------------------------------------------------------------------------
static std::shared_ptr<physics_owner_3pcf> model_physics_common(
  const physics_array& k, // wavenumbers in h/Mpc
  const physics_array& p_linear0, // linear P(k,0), in (Mpc/h)^3
  const physics_array& z, // redshift coordinates of the supplied values
  const physics_array& p_nonlinear, // nonlinear P [z,k], in (Mpc/h)^3
  const physics_array& chi, // comoving distances at each z, in Mpc/h
  const physics_array& dz_dchi, // background Jacobian at each z, in h/Mpc
  const physics_array& growth, // linear growth D(z), normalized to D(0)=1
  // source densities per unit z, [bin,source node]
  const physics_array& sources,
  const double omega_m, // matter density parameter today
  const double omega_de, // dark-energy density parameter today
  const double w0, // dark-energy equation of state today
  const double wa, // coefficient in w(a)=w0+wa*(1-a)
  const double ns, // primordial scalar spectral index
  const double sigma8, // linear rms amplitude at R=8 Mpc/h today
  const physics_array& los_z, // line-of-sight integration redshifts
  const int kernel_nz, // number of radial-kernel preparation nodes
  // retain cubic input readers for source comparisons
  const bool source_readers,
  // dense intervals per coarse interval; zero uses default
  const int table_boost,
  const int fit_nz, // BiHalofit anchor count; zero uses the default
  const py::object& source_z, // optional redshift axis of the source densities
  // optional paired hydro/DMO power and bispectrum responses
  const py::dict& feedback,
  // optional W(ell) samples at integer multipoles
  const py::object& pixel_window,
  const py::dict& baseline, // optional absolute B input
  const py::object& omega_m_z, // required background when baseline is active
  const py::dict& variance, // optional shared fit quantities
  const py::object& omega_m_redshift = py::none() // optional fraction axis
)
{
  if (baseline.empty()) { throw py::value_error(
    "A theory-supplied matter bispectrum is required; internal C "
    "BiHalofit has been removed. Supply baseline and omega_m_z."); }
  shape_physics(k, 1);
  shape_physics(p_linear0, 1);
  shape_physics(z, 1);
  shape_physics(p_nonlinear, 2);
  shape_physics(chi, 1);
  shape_physics(dz_dchi, 1);
  shape_physics(growth, 1);
  shape_physics(sources, 2);
  shape_physics(los_z, 1);

  // The optional source axis obeys the same float64/contiguous rule as
  // the other production arrays, and the C constructor copies it before
  // returning. check_ asks whether the Python object already satisfies
  // the array type without converting it; reinterpret_borrow then wraps
  // that same object - a borrowed view, not a copy - so the adapter can
  // read it through the typed interface.
  physics_array source_axis;
  py::ssize_t nsource = z.size();
  if (!source_z.is_none()) {
    if (!(physics_array::check_(source_z))) {
      spdlog::critical("3pcf physics interface: source_z must be a "
        "contiguous float64 NumPy array");
      std::exit(1);
    }
    source_axis = py::reinterpret_borrow<physics_array>(source_z);
    shape_physics(source_axis, 1);
    finite_physics(source_axis);
    nsource = source_axis.size();
  }

  // One chained cross-table check: the linear power lives on the k
  // axis; the nonlinear power is [nz,nk]; chi, dz/dchi and the growth
  // factor live on the z axis; each source row holds one density per
  // source-coordinate node.
  if (!(p_linear0.size() == k.size()
        && p_nonlinear.shape(0) == z.size()
        && p_nonlinear.shape(1) == k.size()
        && chi.size() == z.size() && dz_dchi.size() == z.size()
        && growth.size() == z.size() && sources.shape(1) == nsource)) {
    spdlog::critical("3pcf physics interface: model table shapes disagree");
    std::exit(1);
  }

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

  // The optional paired hydro/DMO response arrives as a dictionary of
  // exactly eight entries: six arrays plus the two integers boost and
  // clamp (feedback_3pcf.h documents each field). The borrowed arrays
  // stay alive in response_arrays until physics_create_3pcf has copied
  // them.
  feedback_input_3pcf response = {};
  std::vector<physics_array> response_arrays;
  if (!feedback.empty()) {
    const char* names[6] = {"z", "logk", "logu", "v",
      "log_power", "log_bispectrum"};
    if (!(8 == feedback.size()
          && feedback.contains("boost") && feedback.contains("clamp"))) {
      spdlog::critical("3pcf physics interface: feedback requires six "
        "arrays plus boost and clamp");
      std::exit(1);
    }

    for (int index=0; 6 > index; index++) {
      if (!(feedback.contains(names[index])
            && physics_array::check_(feedback[names[index]]))) {
        spdlog::critical("3pcf physics interface: production feedback "
          "arrays must be contiguous float64 vectors");
        std::exit(1);
      }
      response_arrays.push_back(py::reinterpret_borrow<physics_array>(
        feedback[names[index]]));
      shape_physics(response_arrays.back(), 1);
      finite_physics(response_arrays.back());
    }

    const size_t nz = response_arrays[0].size();
    const size_t nk = response_arrays[1].size();
    const size_t nu = response_arrays[2].size();
    const size_t nv = response_arrays[3].size();

    // One chained check: each axis needs at least four nodes for cubic
    // preparation and must fit a C int; each division guard proves the
    // next multiplication cannot wrap around SIZE_MAX; the flattened
    // tables must then match their axis products exactly.
    if (!(4 <= nz && 4 <= nk && 4 <= nu && 4 <= nv
          && INT_MAX >= nz && INT_MAX >= nk && INT_MAX >= nu && INT_MAX >= nv
          && nz <= SIZE_MAX/nk && nz*nk <= SIZE_MAX/nu
          && nz*nk*nu <= SIZE_MAX/nv
          && static_cast<size_t>(response_arrays[4].size()) == nz*nk
          && static_cast<size_t>(response_arrays[5].size()) == nz*nk*nu*nv)) {
      spdlog::critical("3pcf physics interface: flattened paired feedback "
        "arrays disagree with their axes");
      std::exit(1);
    }

    response.nz = static_cast<int>(nz);
    response.nk = static_cast<int>(nk);
    response.nu = static_cast<int>(nu);
    response.nv = static_cast<int>(nv);
    response.z = response_arrays[0].data();
    response.logk = response_arrays[1].data();
    response.logu = response_arrays[2].data();
    response.v = response_arrays[3].data();
    response.log_power = response_arrays[4].data();
    response.log_bispectrum = response_arrays[5].data();
    response.boost = py::cast<int>(feedback["boost"]);
    response.clamp = py::cast<int>(feedback["clamp"]);
    input.feedback = &response;
  }

  // The optional pixel window borrows like source_z above. Its reader
  // interpolates linearly between integer multipoles, so it needs at
  // least two samples.
  physics_array window;
  if (!pixel_window.is_none()) {
    if (!(physics_array::check_(pixel_window))) {
      spdlog::critical("3pcf physics interface: production pixel window "
        "must be a contiguous float64 vector");
      std::exit(1);
    }
    window = py::reinterpret_borrow<physics_array>(pixel_window);
    shape_physics(window, 1);
    if (!(2 <= window.size())) {
      spdlog::critical("3pcf physics interface: pixel window needs at least "
        "two integer-ell samples");
      std::exit(1);
    }
    finite_physics(window);
    input.nwindow = static_cast<int>(window.size());
    input.window = window.data();
  }

  baseline_input_3pcf supplied = {};
  std::vector<physics_array> baseline_arrays;
  physics_array matter_fraction;
  physics_array matter_axis;
  if (!omega_m_redshift.is_none() && omega_m_z.is_none()) {
    throw py::value_error("omega_m_redshift requires omega_m_z");
  }
  if (!omega_m_z.is_none()) {
    if (!physics_array::check_(omega_m_z)) { throw py::value_error(
      "Omega_m(z) must be contiguous float64"); }
    matter_fraction = py::reinterpret_borrow<physics_array>(omega_m_z);
    shape_physics(matter_fraction, 1);
    finite_physics(matter_fraction);
    py::ssize_t count = z.size();
    if (!omega_m_redshift.is_none()) {
      if (!physics_array::check_(omega_m_redshift)) { throw py::value_error(
        "omega_m_redshift must be contiguous float64"); }
      matter_axis = py::reinterpret_borrow<physics_array>(omega_m_redshift);
      shape_physics(matter_axis, 1);
      finite_physics(matter_axis);
      count = matter_axis.size();
      if (INT_MAX < count) { throw py::value_error(
        "omega_m_redshift exceeds C storage"); }
      input.nomega = static_cast<int>(count);
      input.omega_m_redshift = matter_axis.data();
    }
    if (count != matter_fraction.size()) { throw py::value_error(
      "Omega_m(z) must match its redshift axis"); }
    input.omega_m_z = matter_fraction.data();
  }
  if (!baseline.empty()) {
    const char* names[5] = {"z", "logk", "logu", "v", "log_bispectrum"};
    if (!(7 == baseline.size())) { spdlog::critical(
      "3pcf: baseline requires five arrays, boost and clamp"); std::exit(1); }
    if (!(baseline.contains("boost"))) { spdlog::critical(
      "3pcf: baseline boost is missing"); std::exit(1); }
    if (!(baseline.contains("clamp"))) { spdlog::critical(
      "3pcf: baseline clamp is missing"); std::exit(1); }
    size_t size = 1;
    for (int axis=0; 5 > axis; axis++) {
      if (!(baseline.contains(names[axis]))) { spdlog::critical(
        "3pcf: baseline array is missing"); std::exit(1); }
      if (!(physics_array::check_(baseline[names[axis]]))) { spdlog::critical(
        "3pcf: production baseline arrays must be contiguous float64");
        std::exit(1); }
      baseline_arrays.push_back(py::reinterpret_borrow<physics_array>(
        baseline[names[axis]]));
      shape_physics(baseline_arrays.back(), 1);
      finite_physics(baseline_arrays.back());
      if (4 > axis) {
        const size_t count = baseline_arrays.back().size();
        if (!(4 <= count)) { spdlog::critical(
          "3pcf: B axes need four nodes"); std::exit(1); }
        if (!(count <= INT_MAX/size)) { spdlog::critical(
          "3pcf: B dimensions exceed C storage"); std::exit(1); }
        size *= count;
      }
    }
    if (!(size == static_cast<size_t>(baseline_arrays[4].size()))) {
      spdlog::critical(
      "3pcf: B table size disagrees with axes"); std::exit(1); }
    supplied.nz = static_cast<int>(baseline_arrays[0].size());
    supplied.nk = static_cast<int>(baseline_arrays[1].size());
    supplied.nu = static_cast<int>(baseline_arrays[2].size());
    supplied.nv = static_cast<int>(baseline_arrays[3].size());
    supplied.z = baseline_arrays[0].data();
    supplied.logk = baseline_arrays[1].data();
    supplied.logu = baseline_arrays[2].data();
    supplied.v = baseline_arrays[3].data();
    supplied.log_bispectrum = baseline_arrays[4].data();
    supplied.boost = py::cast<int>(baseline["boost"]);
    supplied.clamp = py::cast<int>(baseline["clamp"]);
    input.baseline = &supplied;
  }

  // Retain borrowed production arrays until the immutable C owner has
  // copied them. The dictionary is deliberately small and exact so a
  // misspelled field cannot silently omit a required fit quantity.
  physics_variance_input_3pcf fit = {};
  std::vector<physics_array> fit_arrays;
  if (!variance.empty()) {
    const char* names[4] = {"z", "rnl", "neff", "sigma8"};
    if (4 != variance.size()) {
      throw py::value_error("variance requires exactly z, rnl, neff, sigma8");
    }
    for (const char* name : names) {
      if (!variance.contains(name)
          || !physics_array::check_(variance[name])) {
        throw py::value_error("variance requires contiguous float64 arrays");
      }
      fit_arrays.push_back(py::reinterpret_borrow<physics_array>(
        variance[name]));
      shape_physics(fit_arrays.back(), 1);
      finite_physics(fit_arrays.back());
      if (fit_arrays[0].size() != fit_arrays.back().size()) {
        throw py::value_error("variance array lengths disagree");
      }
    }
    if (4 > fit_arrays[0].size()
        || INT_MAX < fit_arrays[0].size()) {
      throw py::value_error("variance requires at least four anchors");
    }
    fit.nz = static_cast<int>(fit_arrays[0].size());
    fit.z = fit_arrays[0].data();
    fit.rnl = fit_arrays[1].data();
    fit.neff = fit_arrays[2].data();
    fit.sigma8 = fit_arrays[3].data();
    input.variance = &fit;
  }

  return std::make_shared<physics_owner_3pcf>(input);
}

static std::shared_ptr<physics_owner_3pcf> model_physics(
  const physics_array& k, // wave numbers, h/Mpc
  const physics_array& p_linear0, // linear P(z=0), (Mpc/h)^3
  const physics_array& z, // background z, includes zero
  const physics_array& p_nonlinear, // P [nz,nk], (Mpc/h)^3
  const physics_array& chi, // distance, Mpc/h
  const physics_array& dz_dchi, // Jacobian, h/Mpc
  const physics_array& growth, // D(z), one today
  const physics_array& sources, // density [nbin,nsource] per z
  const double omega_m, // matter fraction today
  const double omega_de, // dark-energy fraction today
  const double w0, // equation of state today
  const double wa, // evolution in w(a)=w0+wa(1-a)
  const double ns, // primordial spectral index
  const double sigma8, // legacy scalar; does not rescale power
  const physics_array& los_z, // line-of-sight nodes
  const int kernel_nz, // radial-kernel sampling
  const bool source_readers, // cubic comparison mode
  const int table_boost, // dense-grid refinement
  const int fit_nz, // internal-fit anchors
  const py::object& source_z, // optional source z axis
  const py::dict& feedback, // optional paired responses
  const py::object& pixel_window, // optional integer-ell window
  const py::dict& variance, // optional shared fit quantities
  const py::object& omega_m_z, // optional exact matter fraction
  const py::object& omega_m_redshift, // optional fraction axis
  const py::dict& baseline // required theory-supplied matter bispectrum
)
{
  return model_physics_common(k, p_linear0, z, p_nonlinear, chi, dz_dchi,
    growth,
    sources, omega_m, omega_de, w0, wa, ns, sigma8, los_z, kernel_nz,
    source_readers,
    table_boost, fit_nz, source_z, feedback, pixel_window,
    baseline, omega_m_z, variance, omega_m_redshift);
}


static std::shared_ptr<physics_owner_3pcf> model_physics_with_bispectrum(
  const physics_array& k, // wavenumbers, h/Mpc
  const physics_array& p_linear0, // linear P at z=0, (Mpc/h)^3
  const physics_array& z, // background redshift grid, includes zero
  const physics_array& p_nonlinear, // nonlinear P [nz,nk], (Mpc/h)^3
  const physics_array& chi, // comoving distance, Mpc/h
  const physics_array& dz_dchi, // background Jacobian, h/Mpc
  const physics_array& growth, // linear D(z), normalized to one today
  const physics_array& sources, // density [nbin,nsource] per z
  const double omega_m, // matter fraction today for lensing and IA
  const double sigma8, // legacy scalar; does not rescale power
  const py::dict& baseline, // absolute ln B and its four uniform axes
  const py::object& omega_m_z, // matter fraction on background z [nz]
  const physics_array& los_z, // fixed line-of-sight integration nodes
  const int kernel_nz, // radial-kernel sampling
  const bool source_readers, // original cubic-reader comparison mode
  const int table_boost, // dense power/background intervals per input interval
  const int fit_nz, // legacy fit setting; unused for supplied B
  const py::object& source_z, // optional independent source redshift axis
  const py::dict& feedback, // optional independent hydro/DMO ratios
  const py::object& pixel_window, // optional integer-ell angular window
  const py::dict& variance // optional shared fit quantities
)
{
  // Empty is never a request to reconstruct an internal fit here. The four
  // unused legacy scalars are zero-initialized fields, not a background.
  if (baseline.empty()) { throw py::value_error(
    "supplied B requires a nonempty baseline dictionary"); }
  if (omega_m_z.is_none()) { throw py::value_error(
    "supplied B requires Omega_m(z) on the background grid"); }
  return model_physics_common(k, p_linear0, z, p_nonlinear, chi, dz_dchi,
    growth,
    sources, omega_m, 0.0, 0.0, 0.0, 0.0, sigma8, los_z, kernel_nz,
    source_readers,
    table_boost, fit_nz, source_z, feedback, pixel_window,
    baseline, omega_m_z, variance);
}



// ---------------------------------------------------------------------------
// Notebook model factory: the copying twin of model_physics. It copies
// every array - sliced and Fortran-order inputs included - into
// Armadillo storage and defers validation and repacking to
// physics_model_cpp_3pcf in physics_wrapper.cpp. The arguments match
// model_physics.
// ---------------------------------------------------------------------------
static std::shared_ptr<physics_owner_3pcf> model_notebook_physics_common(
  const physics_notebook_array& k, // wavenumbers in h/Mpc
  const physics_notebook_array& p_linear0, // linear P(k,0), in (Mpc/h)^3
  // redshift coordinates of the supplied values
  const physics_notebook_array& z,
  const physics_notebook_array& p_nonlinear, // nonlinear P [z,k], in (Mpc/h)^3
  const physics_notebook_array& chi, // comoving distances at each z, in Mpc/h
  // background Jacobian at each z, in h/Mpc
  const physics_notebook_array& dz_dchi,
  // linear growth D(z), normalized to D(0)=1
  const physics_notebook_array& growth,
  // source densities per unit z, [bin,source node]
  const physics_notebook_array& sources,
  const double omega_m, // matter density parameter today
  const double omega_de, // dark-energy density parameter today
  const double w0, // dark-energy equation of state today
  const double wa, // coefficient in w(a)=w0+wa*(1-a)
  const double ns, // primordial scalar spectral index
  const double sigma8, // linear rms amplitude at R=8 Mpc/h today
  const physics_notebook_array& los_z, // line-of-sight integration redshifts
  const int kernel_nz, // number of radial-kernel preparation nodes
  // retain cubic input readers for source comparisons
  const bool source_readers,
  // dense intervals per coarse interval; zero uses default
  const int table_boost,
  const int fit_nz, // BiHalofit anchor count; zero uses the default
  const py::object& source_z, // optional redshift axis of the source densities
  // optional paired hydro/DMO power and bispectrum responses
  const py::dict& feedback,
  // optional W(ell) samples at integer multipoles
  const py::object& pixel_window,
  const py::dict& baseline, // optional absolute B input
  const py::object& omega_m_z, // required background when baseline is active
  const py::dict& variance, // optional shared fit quantities
  const py::object& omega_m_redshift = py::none() // optional fraction axis
)
{
  if (baseline.empty()) { throw py::value_error(
    "A theory-supplied matter bispectrum is required; internal C "
    "BiHalofit has been removed. Supply baseline and omega_m_z."); }
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

  physics_feedback_tables_cpp_3pcf response;
  if (!feedback.empty()) {
    const char* names[8] = {"z", "logk", "logu", "v", "log_power",
      "log_bispectrum", "boost", "clamp"};
    if (!(8 == feedback.size())) {
      spdlog::critical("3pcf physics interface: feedback requires six "
        "arrays plus boost and clamp");
      std::exit(1);
    }

    for (const char* name : names) {
      if (!(feedback.contains(name))) {
        spdlog::critical("3pcf physics interface: missing notebook feedback "
          "field");
        std::exit(1);
      }
    }

    response.z = copy_column_physics(py::cast<physics_notebook_array>(
      feedback["z"]));
    response.logk = copy_column_physics(py::cast<physics_notebook_array>(
      feedback["logk"]));
    response.logu = copy_column_physics(py::cast<physics_notebook_array>(
      feedback["logu"]));
    response.v = copy_column_physics(py::cast<physics_notebook_array>(
      feedback["v"]));
    response.log_power = copy_column_physics(py::cast<physics_notebook_array>(
      feedback["log_power"]));
    response.log_bispectrum = copy_column_physics(
      py::cast<physics_notebook_array>(feedback["log_bispectrum"]));
    response.boost = py::cast<int>(feedback["boost"]);
    response.clamp = py::cast<int>(feedback["clamp"]);
  }

  arma::Col<double> window;
  if (!pixel_window.is_none()) {
    window = copy_column_physics(
      py::cast<physics_notebook_array>(pixel_window));
  }

  physics_baseline_table_cpp_3pcf supplied;
  arma::Col<double> matter_fraction;
  arma::Col<double> matter_axis;
  if (!omega_m_redshift.is_none() && omega_m_z.is_none()) {
    throw py::value_error("omega_m_redshift requires omega_m_z");
  }
  if (!omega_m_z.is_none()) {
    matter_fraction = copy_column_physics(
      py::cast<physics_notebook_array>(omega_m_z));
  }
  if (!omega_m_redshift.is_none()) {
    matter_axis = copy_column_physics(
      py::cast<physics_notebook_array>(omega_m_redshift));
  }
  if (!baseline.empty()) {
    const char* names[7] = {"z", "logk", "logu", "v", "log_bispectrum",
      "boost", "clamp"};
    if (!(7 == baseline.size())) { spdlog::critical(
      "3pcf: baseline requires five arrays, boost and clamp"); std::exit(1); }
    for (const char* name : names) {
      if (!(baseline.contains(name))) { spdlog::critical(
        "3pcf: missing notebook baseline field"); std::exit(1); }
    }
    supplied.z = copy_column_physics(
      py::cast<physics_notebook_array>(baseline["z"]));
    supplied.logk = copy_column_physics(
      py::cast<physics_notebook_array>(baseline["logk"]));
    supplied.logu = copy_column_physics(
      py::cast<physics_notebook_array>(baseline["logu"]));
    supplied.v = copy_column_physics(
      py::cast<physics_notebook_array>(baseline["v"]));
    supplied.log_bispectrum = copy_column_physics(
      py::cast<physics_notebook_array>(baseline["log_bispectrum"]));
    supplied.boost = py::cast<int>(baseline["boost"]);
    supplied.clamp = py::cast<int>(baseline["clamp"]);
  }

  physics_variance_tables_cpp_3pcf fit;
  if (!variance.empty()) {
    const char* names[4] = {"z", "rnl", "neff", "sigma8"};
    if (4 != variance.size()) {
      throw py::value_error("variance requires exactly z, rnl, neff, sigma8");
    }
    for (const char* name : names) {
      if (!variance.contains(name)) {
        throw py::value_error("missing notebook variance field");
      }
    }
    fit.z = copy_column_physics(
      py::cast<physics_notebook_array>(variance["z"]));
    fit.rnl = copy_column_physics(
      py::cast<physics_notebook_array>(variance["rnl"]));
    fit.neff = copy_column_physics(
      py::cast<physics_notebook_array>(variance["neff"]));
    fit.sigma8 = copy_column_physics(
      py::cast<physics_notebook_array>(variance["sigma8"]));
  }

  return physics_model_cpp_3pcf(wave, linear, redshift, nonlinear, distance,
    jacobian, normalized_growth, density, omega_m, omega_de, w0, wa, ns,
    sigma8, los, kernel_nz, source_readers, table_boost, fit_nz, source_axis,
    feedback.empty() ? nullptr : &response,
    pixel_window.is_none() ? nullptr : &window,
    baseline.empty() ? nullptr : &supplied,
    omega_m_z.is_none() ? nullptr : &matter_fraction,
    variance.empty() ? nullptr : &fit,
    omega_m_redshift.is_none() ? nullptr : &matter_axis);
}

static std::shared_ptr<physics_owner_3pcf> model_notebook_physics(
  const physics_notebook_array& k, // wave numbers, h/Mpc
  const physics_notebook_array& p_linear0, // linear P(z=0), (Mpc/h)^3
  const physics_notebook_array& z, // background z, includes zero
  const physics_notebook_array& p_nonlinear, // P [nz,nk], (Mpc/h)^3
  const physics_notebook_array& chi, // distance, Mpc/h
  const physics_notebook_array& dz_dchi, // Jacobian, h/Mpc
  const physics_notebook_array& growth, // D(z), one today
  const physics_notebook_array& sources, // density [nbin,nsource] per z
  const double omega_m, // matter fraction today
  const double omega_de, // dark-energy fraction today
  const double w0, // equation of state today
  const double wa, // evolution in w(a)=w0+wa(1-a)
  const double ns, // primordial spectral index
  const double sigma8, // legacy scalar; does not rescale power
  const physics_notebook_array& los_z, // line-of-sight nodes
  const int kernel_nz, // radial-kernel sampling
  const bool source_readers, // cubic comparison mode
  const int table_boost, // dense-grid refinement
  const int fit_nz, // internal-fit anchors
  const py::object& source_z, // optional source z axis
  const py::dict& feedback, // optional paired responses
  const py::object& pixel_window, // optional integer-ell window
  const py::dict& variance, // optional shared fit quantities
  const py::object& omega_m_z, // optional exact matter fraction
  const py::object& omega_m_redshift, // optional fraction axis
  const py::dict& baseline // required theory-supplied matter bispectrum
)
{
  return model_notebook_physics_common(k, p_linear0, z, p_nonlinear, chi,
    dz_dchi,
    growth, sources, omega_m, omega_de, w0, wa, ns, sigma8, los_z, kernel_nz,
    source_readers, table_boost, fit_nz, source_z, feedback, pixel_window,
    baseline, omega_m_z, variance, omega_m_redshift);
}


static std::shared_ptr<physics_owner_3pcf>
model_notebook_physics_with_bispectrum(
  const physics_notebook_array& k, // wavenumbers, h/Mpc
  const physics_notebook_array& p_linear0, // linear P at z=0, (Mpc/h)^3
  const physics_notebook_array& z, // background redshift grid, includes zero
  const physics_notebook_array& p_nonlinear, // nonlinear P [nz,nk], (Mpc/h)^3
  const physics_notebook_array& chi, // comoving distance, Mpc/h
  const physics_notebook_array& dz_dchi, // background Jacobian, h/Mpc
  const physics_notebook_array& growth, // linear D(z), normalized to one today
  const physics_notebook_array& sources, // density [nbin,nsource] per z
  const double omega_m, // matter fraction today for lensing and IA
  const double sigma8, // legacy scalar; does not rescale power
  const py::dict& baseline, // absolute ln B and its four uniform axes
  const py::object& omega_m_z, // matter fraction on background z [nz]
  const physics_notebook_array& los_z, // fixed line-of-sight integration nodes
  const int kernel_nz, // radial-kernel sampling
  const bool source_readers, // original cubic-reader comparison mode
  const int table_boost, // dense power/background intervals per input interval
  const int fit_nz, // legacy fit setting; unused for supplied B
  const py::object& source_z, // optional independent source redshift axis
  const py::dict& feedback, // optional independent hydro/DMO ratios
  const py::object& pixel_window, // optional integer-ell angular window
  const py::dict& variance // optional shared fit quantities
)
{
  // Empty is never a request to reconstruct an internal fit here. The four
  // unused legacy scalars are zero-initialized fields, not a background.
  if (baseline.empty()) { throw py::value_error(
    "supplied B requires a nonempty baseline dictionary"); }
  if (omega_m_z.is_none()) { throw py::value_error(
    "supplied B requires Omega_m(z) on the background grid"); }
  return model_notebook_physics_common(k, p_linear0, z, p_nonlinear, chi,
    dz_dchi,
    growth, sources, omega_m, 0.0, 0.0, 0.0, 0.0, sigma8, los_z, kernel_nz,
    source_readers, table_boost, fit_nz, source_z, feedback, pixel_window,
    baseline, omega_m_z, variance);
}



// ---------------------------------------------------------------------------
// Production point evaluator shared by several methods. One integer
// selects the physical operation, and physics_3pcf.h holds each C
// function it reaches:
//
//   0 - matter bispectrum (physics_matter_3pcf)
//   1 - the nineteen IA bispectra (physics_ia_bispectra_3pcf)
//   2 - projected bispectrum through a shared projector; takes the
//       three source bins instead of redshifts
//   3 - the nineteen IA bispectra for triangles of one common shape
//       (physics_ia_same_shape_3pcf), evaluated in a single C call
//   4 - baryon-feedback bispectrum response
//       (physics_feedback_bispectrum_3pcf)
//
// Parameters:
//   owner      - the immutable model
//   sides      - [point,3] triangle sides: k in h/Mpc, or multipoles
//                ell for operation 2
//   redshifts  - [point] evaluation redshifts (operations 0, 1, 3, 4)
//   bins       - [3] ordered source bins (operation 2 only)
//   parameters - IA parameter choice (physics_ia_3pcf)
//   operation, component, remove_alignment - see above and
//                physics_3pcf.h
//
// Returns:
//   NumPy array [point] or [point,19], checked finite
// ---------------------------------------------------------------------------
static physics_array points_physics(
  const physics_owner_3pcf& owner, // owner of the immutable physical model
  // triangles [point,3], in h/Mpc or angular multipoles
  const physics_array& sides,
  // redshift for each physical query; empty after projection
  const physics_array& redshifts,
  // ordered source-bin labels for the requested correlation
  const physics_int_array& bins,
  // intrinsic-alignment and feedback parameters
  const physics_ia_3pcf& parameters,
  // 0=matter, 1=IA, 2=projected, 3=shared-shape IA, 4=response
  const int operation,
  const int component, // matter-fit or projected split selected by operation
  // omit the linear-alignment baseline from IA outputs
  const bool remove_alignment
)
{
  shape_physics(sides, 2);
  finite_physics(sides);
  if (!(3 == sides.shape(1))) {
    spdlog::critical("3pcf physics interface: sides must have shape [point,3]");
    std::exit(1);
  }

  if (2 == operation) {
    shape_physics(bins, 1);
    aligned_physics(bins, alignof(int));
    if (!(3 == bins.size())) {
      spdlog::critical("3pcf physics interface: three bins are required");
      std::exit(1);
    }
  } else {
    shape_physics(redshifts, 1);
    finite_physics(redshifts);
    if (!(redshifts.size() == sides.shape(0))) {
      spdlog::critical("3pcf physics interface: redshift and triangle "
        "counts disagree");
      std::exit(1);
    }
  }

  const int count = static_cast<int>(sides.shape(0));

  // Operations 1 and 3 return the nineteen density/E/B leg combinations
  // that physics_ia_bispectra_3pcf lists in physics_3pcf.h; every other
  // operation returns one value per triangle.
  const int ncomponent = 1 == operation || 3 == operation ? 19 : 1;
  physics_array output = 1 == operation || 3 == operation
    ? physics_array({static_cast<py::ssize_t>(count),
      static_cast<py::ssize_t>(ncomponent)})
    : physics_array(count);

  const double* triangles = sides.data();
  double* result = output.mutable_data();

  if (3 == operation) {
    {
      // Python's global interpreter lock (the GIL) lets only one thread
      // run Python code at a time. py::gil_scoped_release hands the
      // lock back for the duration of this scope, so other Python
      // threads can run while the long C evaluation works, and so an
      // OpenMP region never starts while the adapter still holds the
      // lock. While the lock is released no Python object may be
      // touched - that is why the finite check on the output waits
      // until the scope closes and the lock returns.
      py::gil_scoped_release release;
      physics_ia_same_shape_3pcf(owner.get(), &parameters, count, triangles,
        redshifts.data(), remove_alignment ? 1 : 0, result);
    }
    finite_physics(output);
    return output;
  }

  // One projector serves the whole loop below: the adapter creates it
  // before the parallel region and destroys it after, and every worker
  // reads it without writing. physics_3pcf.h states this thread
  // contract: a projection and its model support simultaneous
  // read-only calls.
  physics_projection_3pcf* projection = 2 == operation
    ? physics_projector_create_3pcf(owner.get(), &parameters, bins.data())
    : nullptr;
  {
    // Release the GIL for the C work (see points_physics); no Python
    // object may be touched until the scope closes.
    py::gil_scoped_release release;

    // Each iteration evaluates one physical triangle: three sides plus
    // its redshift, or the shared projector for operation 2. The
    // triangles are independent because the model and the projector
    // are read-only here, so the loop parallelizes over points.
    // if(!omp_in_parallel()) keeps this loop serial when the caller
    // already runs inside an OpenMP region: nested parallel regions
    // would oversubscribe the cores.
    #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
    for (int point=0; point<count; point++) {
      const double* triangle = triangles + static_cast<size_t>(point)*3;
      if (0 == operation) { // matter bispectrum
        result[point] = physics_matter_3pcf(owner.get(), triangle[0],
          triangle[1], triangle[2], redshifts.data()[point], component,
          parameters.fb);
      } else if (1 == operation) { // nineteen IA bispectra
        physics_ia_bispectra_3pcf(owner.get(), &parameters, triangle[0],
          triangle[1], triangle[2], redshifts.data()[point],
          remove_alignment ? 1 : 0,
          result + static_cast<size_t>(point)*ncomponent);
      } else if (4 == operation) { // feedback bispectrum response
        result[point] = physics_feedback_bispectrum_3pcf(owner.get(),
          triangle, redshifts.data()[point], parameters.feedback_amplitude);
      } else { // operation 2: projection through the shared projector
        result[point] = physics_projected_3pcf(projection, triangle[0],
          triangle[1], triangle[2], component);
      }
    }
  }

  physics_projector_destroy_3pcf(projection);

  finite_physics(output);

  return output;
}


// ---------------------------------------------------------------------------
// Notebook point evaluator: copy the inputs into Armadillo, run the
// serial twin physics_points_cpp_3pcf (physics_wrapper.cpp), and copy
// its matrix back into a NumPy array. Operations and arguments match
// points_physics.
// ---------------------------------------------------------------------------
static physics_array points_notebook_physics(
  const physics_owner_3pcf& owner, // owner of the immutable physical model
  // triangles [point,3], in h/Mpc or angular multipoles
  const physics_notebook_array& sides,
  // redshift for each physical query; empty after projection
  const physics_notebook_array& redshifts,
  // ordered source-bin labels for the requested correlation
  const physics_notebook_int& bins,
  // intrinsic-alignment and feedback parameters
  const physics_ia_3pcf& parameters,
  // 0=matter, 1=IA, 2=projected, 3=shared-shape IA, 4=response
  const int operation,
  const int component, // matter-fit or projected split selected by operation
  // omit the linear-alignment baseline from IA outputs
  const bool remove_alignment
)
{
  const arma::Mat<double> triangles = copy_matrix_physics(sides);
  arma::Col<double> redshift;
  arma::Col<int> bin;

  if (2 == operation) {
    bin = copy_column_physics(bins);
  } else {
    redshift = copy_column_physics(redshifts);
  }

  const arma::Mat<double> result = physics_points_cpp_3pcf(owner, triangles,
    redshift, bin, parameters, operation, component, remove_alignment);

  physics_array output = 1 == operation || 3 == operation
    ? physics_array({static_cast<py::ssize_t>(result.n_rows),
      static_cast<py::ssize_t>(result.n_cols)})
    : physics_array(result.n_rows);

  // copy out: Armadillo column-major (row index fastest) to the NumPy
  // row-major output, where the column index runs fastest
  for (arma::uword row=0; row<result.n_rows; row++) {
    for (arma::uword column=0; column<result.n_cols; column++) {
      output.mutable_data()[row*result.n_cols + column] = result(row, column);
    }
  }

  finite_physics(output);

  return output;
}


// ---------------------------------------------------------------------------
// Direct Legendre multipoles b_L(ell1,ell2) of the projected
// bispectrum, through the mu quadrature of physics_multipoles_3pcf;
// physics_3pcf.h displays the defining integral.
//
// Parameters:
//   ell        - Fourier sides of the b_L output grid
//   bins       - [3] ordered source bins
//   parameters - IA parameter dictionary (see parameters_physics)
//   lmax, nmu  - Legendre cutoff and mu quadrature resolution
//   component  - total=0, NLA baseline=1, residual IA=2
//
// Returns:
//   NumPy array [lmax+1,nell,nell], checked finite
// ---------------------------------------------------------------------------
static physics_array multipoles_physics(
  const physics_owner_3pcf& owner, // owner of the immutable physical model
  const physics_array& ell, // angular-multipole nodes
  // ordered source-bin labels for the requested correlation
  const physics_int_array& bins,
  const py::dict& parameters, // intrinsic-alignment and feedback parameters
  const int lmax, // largest retained Legendre degree L
  const int nmu, // angular quadrature nodes per Legendre integral
  const int component // 0=total, 1=NLA baseline, 2=residual IA
)
{
  shape_physics(ell, 1);
  shape_physics(bins, 1);
  finite_physics(ell);
  aligned_physics(bins, alignof(int));

  if (!(3 == bins.size() && 0 <= lmax && INT_MAX > lmax)) {
    spdlog::critical("3pcf physics interface: invalid multipole dimensions");
    std::exit(1);
  }

  const size_t nell = static_cast<size_t>(ell.size());

  // Overflow guards before allocating [lmax+1,nell,nell]: first
  // nell*nell, then the full element count with lmax+1. The byte bound
  // uses PY_SSIZE_T_MAX because NumPy measures its allocations with
  // the signed ssize_t; the notebook twin in physics_wrapper.cpp
  // bounds the same count by std::vector's unsigned size_t instead.
  if (!(nell <= SIZE_MAX/nell
        && static_cast<size_t>(lmax + 1) <= SIZE_MAX/(nell*nell)
        && static_cast<size_t>(lmax + 1)*nell*nell
        <= static_cast<size_t>(PY_SSIZE_T_MAX)/sizeof(double))) {
    spdlog::critical("3pcf physics interface: multipole allocation size "
      "overflows");
    std::exit(1);
  }

  physics_array result({static_cast<py::ssize_t>(lmax + 1),
    ell.size(), ell.size()});
  const physics_ia_3pcf ia = parameters_physics(parameters);
  {
    // Release the GIL for the C work (see points_physics); no Python
    // object may be touched until the scope closes.
    py::gil_scoped_release release;
    physics_multipoles_3pcf(owner.get(), &ia, bins.data(),
      static_cast<int>(ell.size()), ell.data(), lmax, nmu, component,
      result.mutable_data());
  }

  finite_physics(result);

  return result;
}


// ---------------------------------------------------------------------------
// Notebook multipoles: copy the inputs into Armadillo, run the twin
// physics_multipoles_cpp_3pcf, and copy its cube back into a NumPy
// array. Arguments match multipoles_physics.
// ---------------------------------------------------------------------------
static physics_array multipoles_notebook_physics(
  const physics_owner_3pcf& owner, // owner of the immutable physical model
  const physics_notebook_array& ell, // angular-multipole nodes
  // ordered source-bin labels for the requested correlation
  const physics_notebook_int& bins,
  const py::dict& parameters, // intrinsic-alignment and feedback parameters
  const int lmax, // largest retained Legendre degree L
  const int nmu, // angular quadrature nodes per Legendre integral
  const int component // 0=total, 1=NLA baseline, 2=residual IA
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

  // copy out: the Armadillo cube is column-major inside each slice;
  // the NumPy output is row-major [L,ell1,ell2], ell2 running fastest
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


// ---------------------------------------------------------------------------
// Full third-order aperture statistic by the direct (ell, phi)
// quadrature of physics_aperture_full_3pcf; physics_3pcf.h gives the
// compensated-Gaussian filter U. One value comes back per
// filter-radius triple.
//
// Parameters:
//   ell, dell  - Fourier quadrature nodes and weights
//   phi, dphi  - angular quadrature nodes and weights, radians
//   filters    - [nfilter,3] aperture radii in radians, one per
//                ordered leg
//   bins       - [3] ordered source bins
//   parameters - IA parameter dictionary
//
// Returns:
//   NumPy array [nfilter], checked finite
// ---------------------------------------------------------------------------
static physics_array aperture_full_physics(
  const physics_owner_3pcf& owner, // owner of the immutable physical model
  const physics_array& ell, // angular-multipole nodes
  const physics_array& dell, // weights for integration over ell
  const physics_array& phi, // triangle opening-angle nodes, in radians
  const physics_array& dphi, // opening-angle integration weights, in radians
  const physics_array& filters, // aperture radii [filter,3], in radians
  // ordered source-bin labels for the requested correlation
  const physics_int_array& bins,
  const py::dict& parameters // intrinsic-alignment and feedback parameters
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

  // One chained check: each weight array matches its node array; each
  // filter row holds three radii; exactly three source bins arrive.
  if (!(ell.size() == dell.size()
        && phi.size() == dphi.size() && 3 == filters.shape(1)
        && 3 == bins.size())) {
    spdlog::critical("3pcf physics interface: full-aperture input shapes "
      "disagree");
    std::exit(1);
  }

  const physics_ia_3pcf ia = parameters_physics(parameters);
  physics_array output(filters.shape(0));

  {
    // Release the GIL for the C work (see points_physics); no Python
    // object may be touched until the scope closes.
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


// ---------------------------------------------------------------------------
// Notebook aperture statistic: copy the inputs into Armadillo, run the
// twin physics_aperture_full_cpp_3pcf, and copy the result vector into
// a NumPy array. Arguments match aperture_full_physics.
// ---------------------------------------------------------------------------
static physics_array aperture_full_notebook_physics(
  const physics_owner_3pcf& owner, // owner of the immutable physical model
  const physics_notebook_array& ell, // angular-multipole nodes
  const physics_notebook_array& dell, // weights for integration over ell
  const physics_notebook_array& phi, // triangle opening-angle nodes, in radians
  // opening-angle integration weights, in radians
  const physics_notebook_array& dphi,
  // aperture radii [filter,3], in radians
  const physics_notebook_array& filters,
  // ordered source-bin labels for the requested correlation
  const physics_notebook_int& bins,
  const py::dict& parameters // intrinsic-alignment and feedback parameters
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


// ---------------------------------------------------------------------------
// Aperture statistic from a prepared projection-grid table: the same
// (ell, phi) quadrature as aperture_full_physics, reading interpolated
// values instead of line-of-sight projections. The table already fixed
// the source bins and IA parameters at its construction.
//
// Parameters:
//   ell, dell - Fourier quadrature nodes and weights
//   phi, dphi - angular quadrature nodes and weights, radians
//   filters   - [nfilter,3] aperture radii in radians
//
// Returns:
//   NumPy array [nfilter], checked finite
// ---------------------------------------------------------------------------
static physics_array aperture_grid_physics(
  // owner of the immutable projected-bispectrum grid
  const physics_grid_owner_3pcf& owner,
  const physics_array& ell, // angular-multipole nodes
  const physics_array& dell, // weights for integration over ell
  const physics_array& phi, // triangle opening-angle nodes, in radians
  const physics_array& dphi, // opening-angle integration weights, in radians
  const physics_array& filters // aperture radii [filter,3], in radians
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

  // One chained check: each weight array matches its node array and
  // each filter row holds three radii.
  if (!(ell.size() == dell.size()
        && phi.size() == dphi.size() && 3 == filters.shape(1))) {
    spdlog::critical("3pcf physics interface: grid aperture input shapes "
      "disagree");
    std::exit(1);
  }

  physics_array output(filters.shape(0));
  {
    // Release the GIL for the C work (see points_physics); no Python
    // object may be touched until the scope closes.
    py::gil_scoped_release release;
    physics_grid_aperture_full_3pcf(owner.get(), static_cast<int>(ell.size()),
      ell.data(), dell.data(), static_cast<int>(phi.size()), phi.data(),
      dphi.data(), static_cast<int>(filters.shape(0)), filters.data(),
      output.mutable_data());
  }

  finite_physics(output);

  return output;
}


// ---------------------------------------------------------------------------
// Interpolated projected bispectrum: one read of the prepared grid
// table (physics_grid_3pcf.h) per row of multipole triangles.
//
// Parameters:
//   sides     - [point,3] angular Fourier triangles
//   component - total=0, NLA baseline=1, residual IA=2
//
// Returns:
//   NumPy array [point], checked finite
// ---------------------------------------------------------------------------
static physics_array projected_grid_physics(
  // owner of the immutable projected-bispectrum grid
  const physics_grid_owner_3pcf& owner,
  const physics_array& sides, // multipole triangles [point,3]
  const int component // 0=total, 1=NLA baseline, 2=residual IA
)
{
  shape_physics(sides, 2);
  finite_physics(sides);
  if (!(3 == sides.shape(1))) {
    spdlog::critical("3pcf physics interface: projected grid sides must "
      "have shape [point,3]");
    std::exit(1);
  }

  const int count = static_cast<int>(sides.shape(0));
  physics_array output(count);
  const double* triangles = sides.data();
  double* values = output.mutable_data();

  {
    // Release the GIL for the C work (see points_physics); no Python
    // object may be touched until the scope closes.
    py::gil_scoped_release release;

    // Each iteration interpolates the prepared table at one triangle
    // of Fourier sides. The reads are independent because the table is
    // immutable, so the loop parallelizes over rows, and
    // if(!omp_in_parallel()) keeps it serial inside an enclosing
    // OpenMP region.
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


// ---------------------------------------------------------------------------
// Interpolated Legendre multipoles b_L from a prepared grid table; the
// owner carries the Legendre cutoff the table was built with.
//
// Parameters:
//   ell       - Fourier sides of the b_L output grid
//   component - total=0, NLA baseline=1, residual IA=2
//
// Returns:
//   NumPy array [lmax+1,nell,nell], checked finite
// ---------------------------------------------------------------------------
static physics_array multipoles_grid_physics(
  // owner of the immutable projected-bispectrum grid
  const physics_grid_owner_3pcf& owner,
  const physics_array& ell, // angular-multipole nodes
  const int component // 0=total, 1=NLA baseline, 2=residual IA
)
{
  shape_physics(ell, 1);
  finite_physics(ell);
  const size_t nell = static_cast<size_t>(ell.size());
  const size_t norder = static_cast<size_t>(owner.lmax() + 1);

  // Overflow guards as in multipoles_physics; the byte bound is
  // NumPy's signed ssize_t.
  if (!(nell <= SIZE_MAX/nell
        && norder <= SIZE_MAX/(nell*nell)
        && norder*nell*nell
        <= static_cast<size_t>(PY_SSIZE_T_MAX)/sizeof(double))) {
    spdlog::critical("3pcf physics interface: interpolated-multipole output "
      "size overflows");
    std::exit(1);
  }

  physics_array output({static_cast<py::ssize_t>(norder),
    ell.size(), ell.size()});
  {
    // Release the GIL for the C work (see points_physics); no Python
    // object may be touched until the scope closes.
    py::gil_scoped_release release;
    physics_grid_multipoles_3pcf(owner.get(), static_cast<int>(nell),
      ell.data(), component, output.mutable_data());
  }

  finite_physics(output);

  return output;
}


// ---------------------------------------------------------------------------
// Borrow readers for the covariance and transform interfaces: they
// reach the C pointers that the Python-side owners hold. The Python
// owner must stay alive until the consumer finishes, because the
// pointer borrows tables the owner will release. py::cast throws when
// the Python object has the wrong type; reading the pointer changes no
// state and builds no second model.
// ---------------------------------------------------------------------------
const physics_model_3pcf* borrow_physics_model_3pcf(
  py::handle owner // Python owner of the immutable physical model
)
{
  return py::cast<const physics_owner_3pcf&>(owner).get();
}


// Same contract for a prepared projection-grid table.
const physics_grid_3pcf* borrow_physics_grid_3pcf(
  py::handle owner // Python owner of the immutable projected grid
)
{
  return py::cast<const physics_grid_owner_3pcf&>(owner).get();
}


// ---------------------------------------------------------------------------
// Register the 3PCF physics classes on the Python module. pybind11
// builds the Python classes from the functions and lambdas above; each
// def(...) call below attaches one method together with its argument
// names, defaults and docstring. The function proceeds in bannered
// stages: projection-grid table bindings, model construction, point
// and angular queries, projection-grid factories, notebook point
// queries, power diagnostics, and the submodule aliases.
//
// Parameters:
//   module - the extension module under construction
// ---------------------------------------------------------------------------
void bind_threepcf_physics(
  py::module_& module // Python module receiving the bindings
)
{
  // --- 1. PROJECTION-GRID TABLE BINDINGS ---------------------------------
  // ProjectionGrid wraps one prepared interpolation table: the stored
  // projected bispectrum and its Legendre multipoles for one ordered
  // source-bin triplet and one IA parameter choice, fixed at
  // construction. Python queries the table through a borrowing
  // production method or its copying notebook twin; the physics lives
  // in physics_grid_3pcf.h.
  py::class_<physics_grid_owner_3pcf,
    std::shared_ptr<physics_grid_owner_3pcf>> grid(module, "ProjectionGrid",
    "Immutable projected-bispectrum tables for one ordered source-bin "
    "triplet. The NLA baseline lives on the sorted (r,u,v) triangle "
    "chart: r = middle side, u = smallest/middle, "
    "v = (largest-middle)/smallest. The TATT residual lives on the "
    "unsorted (p,u,v) chart: p = side sum, u = ell1/p, v = ell2/p. "
    "The Legendre multipoles b_L of the projected bispectrum come in "
    "two angular charts, one per ordering of the two free sides, so "
    "unequal source bins keep their ordered legs.");

  grid.def("projected", &projected_grid_physics,
    py::arg("sides").noconvert(), py::arg("component") = 0,
    "Interpolated projected bispectrum B[point] without pixel "
    "smoothing; the model's pixel window enters the angular "
    "observables only.");

  grid.def("multipoles", &multipoles_grid_physics,
    py::arg("ell").noconvert(), py::arg("component") = 0,
    "Legendre multipoles b_L[L,ell1,ell2]; the model's pixel window "
    "multiplies all three legs.");

  grid.def("projected_notebook", [](
    // owner of the immutable projected-bispectrum grid
    const physics_grid_owner_3pcf& owner,
    const physics_notebook_array& sides, // multipole triangles [point,3]
    int component // 0=total, 1=NLA baseline, 2=residual IA
  ) {
    const arma::Mat<double> triangles = copy_matrix_physics(sides);
    const arma::Col<double> values = physics_grid_projected_cpp_3pcf(owner,
      triangles, component);
    physics_array output(values.n_elem);
    // one memcpy suffices: both vectors are one-dimensional
    std::memcpy(output.mutable_data(), values.memptr(),
      static_cast<size_t>(values.n_elem)*sizeof(double));
    finite_physics(output);
    return output;
  }, py::arg("sides"), py::arg("component") = 0,
    "Interpolated projected bispectrum B[point] without pixel "
    "smoothing, copied through the notebook adapter.");

  grid.def("multipoles_notebook", [](
    // owner of the immutable projected-bispectrum grid
    const physics_grid_owner_3pcf& owner,
    const physics_notebook_array& ell, // angular-multipole nodes
    int component // 0=total, 1=NLA baseline, 2=residual IA
  ) {
    const arma::Col<double> nodes = copy_column_physics(ell);
    const arma::Cube<double> values = physics_grid_multipoles_cpp_3pcf(owner,
      nodes, component);
    physics_array output({static_cast<py::ssize_t>(values.n_rows),
      static_cast<py::ssize_t>(values.n_cols),
      static_cast<py::ssize_t>(values.n_slices)});
    // copy out: the Armadillo cube is column-major inside each slice;
    // the NumPy output is row-major [L,ell1,ell2], ell2 running fastest
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
  }, py::arg("ell"), py::arg("component") = 0,
    "Legendre multipoles b_L[L,ell1,ell2] through the copying notebook "
    "adapter; the model's pixel window multiplies all three legs.");

  grid.def("aperture_full", &aperture_grid_physics,
    py::arg("ell").noconvert(), py::arg("dell").noconvert(),
    py::arg("phi").noconvert(), py::arg("dphi").noconvert(),
    py::arg("filters").noconvert(),
    "Full-filter aperture mean from the prepared table; the model's "
    "pixel window multiplies all three legs.");

  grid.def("aperture_full_notebook", [](
    // owner of the immutable projected-bispectrum grid
    const physics_grid_owner_3pcf& owner,
    const physics_notebook_array& ell, // angular-multipole nodes
    const physics_notebook_array& dell, // weights for integration over ell
    // triangle opening-angle nodes, in radians
    const physics_notebook_array& phi,
    // opening-angle integration weights, in radians
    const physics_notebook_array& dphi,
    // aperture radii [filter,3], in radians
    const physics_notebook_array& filters
  ) {
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
    py::arg("filters"),
    "Full-filter aperture mean through the copying notebook adapter; "
    "the model's pixel window multiplies all three legs.");

  grid.def("axes", [](
    // owner of the immutable projected-bispectrum grid
    const physics_grid_owner_3pcf& owner
  ) {
    py::dict result;
    // The eight axis codes and their names follow physics_grid_axis_3pcf
    // in physics_grid_3pcf.h; the dictionary keys repeat those names.
    const char* names[8] = {"log_r", "log_u", "v", "log_p", "pu",
      "log_ell", "log_psi", "mu"};
    for (int axis=0; 8 > axis; axis++) {
      int count;
      const double* values;
      physics_grid_axis_3pcf(owner.get(), axis, &count, &values);
      physics_array output(count);
      // each axis is one contiguous vector, so one memcpy copies it;
      // Python owns the copy and cannot dangle into the grid
      std::memcpy(output.mutable_data(), values,
        static_cast<size_t>(count)*sizeof(double));
      result[names[axis]] = output;
    }
    return result;
  }, "Owned copies of the actual interpolation coordinates, keyed by "
    "the axis names of physics_grid_3pcf.h; the mu axis includes its "
    "endpoint nodes.");

  // --- 2. MODEL CONSTRUCTION ---------------------------------------------
  // PhysicsModel owns one immutable C model: the matter and IA
  // bispectrum fits, radial kernels and power tables that every later
  // query reads. The production constructor borrows NumPy arrays and
  // the notebook factory copies them; both end in physics_create_3pcf,
  // which owns its data, so Python may discard the inputs afterwards.
  py::class_<physics_owner_3pcf, std::shared_ptr<physics_owner_3pcf>> model(
    module, "PhysicsModel",
    "Owned matter/TATT model from k[h/Mpc], P[(Mpc/h)^3], chi[Mpc/h]. "
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
    py::arg("source_z") = py::none(), py::arg("feedback") = py::dict(),
    py::arg("pixel_window") = py::none(),
    py::arg("variance") = py::dict(),
    py::arg("omega_m_z") = py::none(),
    py::arg("omega_m_redshift") = py::none(),
    py::arg("baseline") = py::dict());

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
    py::arg("source_z") = py::none(), py::arg("feedback") = py::dict(),
    py::arg("pixel_window") = py::none(),
    py::arg("variance") = py::dict(),
    py::arg("omega_m_z") = py::none(),
    py::arg("omega_m_redshift") = py::none(),
    py::arg("baseline") = py::dict(),
    "Copy sliced or Fortran-order arrays through the Armadillo adapter.");

  module.def("physics_model_production_with_bispectrum",
    &model_physics_with_bispectrum,
    py::arg("k").noconvert(),
    py::arg("p_linear0").noconvert(),
    py::arg("z").noconvert(),
    py::arg("p_nonlinear").noconvert(),
    py::arg("chi").noconvert(),
    py::arg("dz_dchi").noconvert(),
    py::arg("growth").noconvert(),
    py::arg("sources").noconvert(),
    py::arg("omega_m"),
    py::arg("sigma8"),
    py::arg("baseline"),
    py::arg("omega_m_z"),
    py::arg("los_z").noconvert(),
    py::arg("kernel_nz") = 101,
    py::arg("source_readers") = false,
    py::arg("table_boost") = 8,
    py::arg("fit_nz") = 33,
    py::arg("source_z") = py::none(),
    py::arg("feedback") = py::dict(),
    py::arg("pixel_window") = py::none(),
    py::arg("variance") = py::dict(),
    "Construct from supplied absolute B and Omega_m(z); no internal B fit. "
    "The caller supplies a flat-geometry background consistent with P and B.");

  module.def("physics_model_notebook_with_bispectrum",
    &model_notebook_physics_with_bispectrum,
    py::arg("k"),
    py::arg("p_linear0"),
    py::arg("z"),
    py::arg("p_nonlinear"),
    py::arg("chi"),
    py::arg("dz_dchi"),
    py::arg("growth"),
    py::arg("sources"),
    py::arg("omega_m"),
    py::arg("sigma8"),
    py::arg("baseline"),
    py::arg("omega_m_z"),
    py::arg("los_z"),
    py::arg("kernel_nz") = 101,
    py::arg("source_readers") = false,
    py::arg("table_boost") = 8,
    py::arg("fit_nz") = 33,
    py::arg("source_z") = py::none(),
    py::arg("feedback") = py::dict(),
    py::arg("pixel_window") = py::none(),
    py::arg("variance") = py::dict(),
    "Construct from supplied absolute B and Omega_m(z); no internal B fit. "
    "The caller supplies a flat-geometry background consistent with P and B.");

  // --- 3. POINT AND ANGULAR QUERIES --------------------------------------
  // Direct evaluations of the model: bispectra at physical or angular
  // triangles, Legendre multipoles and full aperture means. The
  // production methods funnel into points_physics or the dedicated
  // evaluators above; physics_3pcf.h holds every integrand.
  model.def("matter", [](
    const physics_owner_3pcf& owner, // owner of the immutable physical model
    const physics_array& sides, // Fourier triangles [point,3], in h/Mpc
    // redshift at which each physical query is evaluated
    const physics_array& redshifts,
    int component, // 0=full matter B, 1=tree, 2=one-halo, 3=three-halo fit
    double fb // bispectrum-only feedback amplitude
  ) {
    physics_ia_3pcf parameters;
    physics_default_ia_3pcf(&parameters);
    parameters.fb = fb;
    return points_physics(owner, sides, redshifts, physics_int_array(),
      parameters, 0, component, false);
  }, py::arg("sides").noconvert(), py::arg("redshifts").noconvert(),
    py::arg("component") = 0, py::arg("fb") = 0.0,
    "B_matter[point]: component 0=selected full B, 1=tree level, 2=one-halo "
    "fit, 3=three-halo fit. Supplied full B rejects components 2 and 3. "
    "These are fitted matter terms, not the "
    "halo integrals of the covariance module.");

  model.def("ia", [](
    const physics_owner_3pcf& owner, // owner of the immutable physical model
    const physics_array& sides, // Fourier triangles [point,3], in h/Mpc
    // redshift at which each physical query is evaluated
    const physics_array& redshifts,
    const py::dict& parameters, // intrinsic-alignment and feedback parameters
    bool remove_alignment // omit the linear-alignment baseline from IA outputs
  ) {
    return points_physics(owner, sides, redshifts, physics_int_array(),
      parameters_physics(parameters), 1, 0, remove_alignment);
  }, py::arg("sides").noconvert(), py::arg("redshifts").noconvert(),
    py::arg("parameters") = py::dict(), py::arg("remove_alignment") = false,
    "[point,19] IA bispectra in the nineteen-entry leg order that "
    "physics_3pcf.h defines.");

  model.def("ia_same_shape", [](
    const physics_owner_3pcf& owner, // owner of the immutable physical model
    const physics_array& sides, // Fourier triangles [point,3], in h/Mpc
    // redshift at which each physical query is evaluated
    const physics_array& redshifts,
    const py::dict& parameters, // intrinsic-alignment and feedback parameters
    bool remove_alignment // omit the linear-alignment baseline from IA outputs
  ) {
    return points_physics(owner, sides, redshifts, physics_int_array(),
      parameters_physics(parameters), 3, 0, remove_alignment);
  }, py::arg("sides").noconvert(), py::arg("redshifts").noconvert(),
    py::arg("parameters") = py::dict(), py::arg("remove_alignment") = false,
    "All nineteen IA terms sharing one geometry preparation: the "
    "triangles may differ only by a positive common scaling of all "
    "three sides.");

  model.def("ia_same_shape_notebook", [](
    const physics_owner_3pcf& owner, // owner of the immutable physical model
    // Fourier triangles [point,3], in h/Mpc
    const physics_notebook_array& sides,
    // redshift at which each physical query is evaluated
    const physics_notebook_array& redshifts,
    const py::dict& parameters, // intrinsic-alignment and feedback parameters
    bool remove_alignment // omit the linear-alignment baseline from IA outputs
  ) {
    return points_notebook_physics(owner, sides, redshifts,
      physics_notebook_int(), parameters_physics(parameters), 3, 0,
      remove_alignment);
  }, py::arg("sides"), py::arg("redshifts"),
    py::arg("parameters") = py::dict(), py::arg("remove_alignment") = false);

  model.def("projected", [](
    const physics_owner_3pcf& owner, // owner of the immutable physical model
    const physics_array& sides, // multipole triangles [point,3]
    // ordered source-bin labels for the requested correlation
    const physics_int_array& bins,
    const py::dict& parameters, // intrinsic-alignment and feedback parameters
    int component // 0=total, 1=NLA baseline, 2=residual IA
  ) {
    return points_physics(owner, sides, physics_array(), bins,
      parameters_physics(parameters), 2, component, false);
  }, py::arg("sides").noconvert(), py::arg("bins").noconvert(),
    py::arg("parameters") = py::dict(), py::arg("component") = 0,
    "Projected E-mode bispectrum without pixel smoothing: 0=total, "
    "1=matter with NLA kernels, 2=TATT residual.");

  model.def("multipoles", &multipoles_physics,
    py::arg("ell").noconvert(), py::arg("bins").noconvert(),
    py::arg("parameters"), py::arg("lmax"), py::arg("nmu"),
    py::arg("component") = 0,
    "Direct b_L[L,ell1,ell2]; the model's pixel window multiplies all "
    "three legs. Convergence is the caller's duty: rerun with denser "
    "ell and mu nodes until the result stabilizes.");

  model.def("multipoles_notebook", &multipoles_notebook_physics,
    py::arg("ell"), py::arg("bins"), py::arg("parameters"),
    py::arg("lmax"), py::arg("nmu"), py::arg("component") = 0,
    "Direct b_L through the copying notebook adapter; the model's "
    "pixel window multiplies all three legs.");

  model.def("aperture_full", &aperture_full_physics,
    py::arg("ell").noconvert(), py::arg("dell").noconvert(),
    py::arg("phi").noconvert(), py::arg("dphi").noconvert(),
    py::arg("filters").noconvert(), py::arg("bins").noconvert(),
    py::arg("parameters") = py::dict(),
    "Full-filter third-order aperture mean with pixel smoothing on all "
    "three legs; phi and the aperture radii are in radians.");

  model.def("aperture_full_notebook", &aperture_full_notebook_physics,
    py::arg("ell"), py::arg("dell"), py::arg("phi"), py::arg("dphi"),
    py::arg("filters"), py::arg("bins"), py::arg("parameters") = py::dict(),
    "Full-filter third-order aperture mean through the copying "
    "notebook adapter; phi and the aperture radii are in radians.");

  // --- 4. PROJECTION-GRID FACTORIES --------------------------------------
  // These methods build ProjectionGrid tables from the model: one table
  // per ordered source triplet, with the IA parameters and resolution
  // settings fixed at construction. The batch factory shares the
  // triangle and redshift preparation across all triplets.
  model.def("projection_grid", [](
      // owner of the immutable physical model
      const std::shared_ptr<physics_owner_3pcf>& owner,
      // ordered source-bin labels for the requested correlation
      const physics_int_array& bins,
      const py::dict& parameters, // intrinsic-alignment and feedback parameters
      // physical-grid bounds, node counts and angular cutoffs
      const py::dict& settings
  ) {
    shape_physics(bins, 1);
    aligned_physics(bins, alignof(int));
    if (!(3 == bins.size())) {
      spdlog::critical("3pcf physics interface: three source bins are "
        "required");
      std::exit(1);
    }
    const physics_ia_3pcf ia = parameters_physics(parameters);
    const physics_grid_settings_3pcf configuration
      = settings_physics_grid(settings);
    // The lambda releases the GIL while the C side fills the table
    // (see points_physics); no Python object until it returns.
    py::gil_scoped_release release;
    return std::make_shared<physics_grid_owner_3pcf>(owner, ia, bins.data(),
      configuration);
  }, py::arg("bins").noconvert(), py::arg("parameters") = py::dict(),
    py::arg("settings") = py::dict(),
    "Prepare projected-B and b_L tables with explicit resolution settings.");

  model.def("projection_grids", [](
    const physics_owner_3pcf& owner, // owner of the immutable physical model
    // ordered source-bin labels for the requested correlation
    const physics_int_array& bins,
    const py::dict& parameters, // intrinsic-alignment and feedback parameters
    // physical-grid bounds, node counts and angular cutoffs
    const py::dict& settings
  ) {
    shape_physics(bins, 2);
    aligned_physics(bins, alignof(int));
    if (!(3 == bins.shape(1))) {
      spdlog::critical("3pcf physics interface: batch bins must have shape "
        "[ntriplet,3]");
      std::exit(1);
    }
    const physics_ia_3pcf ia = parameters_physics(parameters);
    const physics_grid_settings_3pcf configuration
      = settings_physics_grid(settings);
    std::vector<physics_grid_3pcf*> prepared(bins.shape(0));
    {
      // Release the GIL while the C side fills every table in the
      // batch (see points_physics).
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

  model.def("projection_grids_notebook", [](
    const physics_owner_3pcf& owner, // owner of the immutable physical model
    // ordered source-bin labels for the requested correlation
    const physics_notebook_int& bins,
    const py::dict& parameters, // intrinsic-alignment and feedback parameters
    // physical-grid bounds, node counts and angular cutoffs
    const py::dict& settings
  ) {
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
      // owner of the immutable physical model
      const std::shared_ptr<physics_owner_3pcf>& owner,
      // ordered source-bin labels for the requested correlation
      const physics_notebook_int& bins,
      const py::dict& parameters, // intrinsic-alignment and feedback parameters
      // physical-grid bounds, node counts and angular cutoffs
      const py::dict& settings
  ) {
    const arma::Col<int> source = copy_column_physics(bins);
    if (!(3 == source.n_elem)) {
      spdlog::critical("3pcf physics interface: three bins are required");
      std::exit(1);
    }
    const physics_ia_3pcf ia = parameters_physics(parameters);
    const physics_grid_settings_3pcf configuration
      = settings_physics_grid(settings);
    return std::make_shared<physics_grid_owner_3pcf>(owner, ia, source.memptr(),
      configuration);
  }, py::arg("bins"), py::arg("parameters") = py::dict(),
    py::arg("settings") = py::dict());

  // --- 5. NOTEBOOK POINT QUERIES -----------------------------------------
  // The copying twins of the stage 3 point queries: same physics, same
  // dispatch, routed through points_notebook_physics so sliced and
  // Fortran-order inputs work.
  model.def("matter_notebook", [](
    const physics_owner_3pcf& owner, // owner of the immutable physical model
    // Fourier triangles [point,3], in h/Mpc
    const physics_notebook_array& sides,
    // redshift at which each physical query is evaluated
    const physics_notebook_array& redshifts,
    int component, // 0=full matter B, 1=tree, 2=one-halo, 3=three-halo fit
    double fb // bispectrum-only feedback amplitude
  ) {
    physics_ia_3pcf parameters;
    physics_default_ia_3pcf(&parameters);
    parameters.fb = fb;
    return points_notebook_physics(owner, sides, redshifts,
      physics_notebook_int(), parameters, 0, component, false);
  }, py::arg("sides"), py::arg("redshifts"), py::arg("component") = 0,
    py::arg("fb") = 0.0);

  model.def("ia_notebook", [](
    const physics_owner_3pcf& owner, // owner of the immutable physical model
    // Fourier triangles [point,3], in h/Mpc
    const physics_notebook_array& sides,
    // redshift at which each physical query is evaluated
    const physics_notebook_array& redshifts,
    const py::dict& parameters, // intrinsic-alignment and feedback parameters
    bool remove_alignment // omit the linear-alignment baseline from IA outputs
  ) {
    return points_notebook_physics(owner, sides, redshifts,
      physics_notebook_int(), parameters_physics(parameters), 1, 0,
      remove_alignment);
  }, py::arg("sides"), py::arg("redshifts"),
    py::arg("parameters") = py::dict(), py::arg("remove_alignment") = false);

  model.def("projected_notebook", [](
    const physics_owner_3pcf& owner, // owner of the immutable physical model
    const physics_notebook_array& sides, // multipole triangles [point,3]
    // ordered source-bin labels for the requested correlation
    const physics_notebook_int& bins,
    const py::dict& parameters, // intrinsic-alignment and feedback parameters
    int component // 0=total, 1=NLA baseline, 2=residual IA
  ) {
    return points_notebook_physics(owner, sides, physics_notebook_array(),
      bins, parameters_physics(parameters), 2, component, false);
  }, py::arg("sides"), py::arg("bins"),
    py::arg("parameters") = py::dict(), py::arg("component") = 0,
    "Projected E-mode bispectrum without pixel smoothing, copied "
    "through the notebook adapter.");

  // --- 6. POWER DIAGNOSTICS ----------------------------------------------
  // Scalar power-spectrum readers and response diagnostics: cheap
  // cross-checks of the tables the model interpolates, plus the NLA
  // shear C_ell and the radial-kernel dump that the covariance route
  // reuses.
  model.def("power", [](
    const physics_owner_3pcf& owner, // owner of the immutable physical model
    const physics_array& k, // wavenumbers in h/Mpc
    const physics_array& z, // redshift coordinates of the supplied values
    int mode // 0=linear(0)*D^2, 1=nonlinear(0)*D^2, 2=nonlinear(z)
  ) {
    shape_physics(k, 1);
    shape_physics(z, 1);
    finite_physics(k);
    finite_physics(z);
    if (!(k.size() == z.size())) {
      spdlog::critical("3pcf physics interface: power k and z arrays must "
        "have equal length");
      std::exit(1);
    }
    physics_array result(k.size());
    for (py::ssize_t index=0; index<k.size(); index++) {
      result.mutable_data()[index] = physics_power_3pcf(owner.get(),
        k.data()[index], z.data()[index], mode);
    }
    finite_physics(result);
    return result;
  }, py::arg("k").noconvert(), py::arg("z").noconvert(),
    py::arg("mode") = 2,
    "P(k,z) modes: 0=linear times D^2, 1=nonlinear(z=0) times D^2, "
    "2=the tabulated nonlinear evolution. D is the linear growth "
    "factor, normalized to one today.");

  model.def("power_response", [](
    const physics_owner_3pcf& owner, // owner of the immutable physical model
    const physics_array& k, // wavenumbers in h/Mpc
    const physics_array& z, // redshift coordinates of the supplied values
    double amplitude // shared logarithmic feedback response; zero is DMO
  ) {
    shape_physics(k, 1);
    shape_physics(z, 1);
    finite_physics(k);
    finite_physics(z);
    if (!(k.size() == z.size())) {
      spdlog::critical("3pcf physics interface: power-response k and z "
        "arrays must have equal length");
      std::exit(1);
    }
    physics_array result(k.size());
    for (py::ssize_t index=0; index<k.size(); index++) {
      result.mutable_data()[index] = physics_feedback_power_3pcf(owner.get(),
        k.data()[index], z.data()[index], amplitude);
    }
    return result;
  }, py::arg("k").noconvert(), py::arg("z").noconvert(),
    py::arg("amplitude"));

  model.def("power_response_notebook", [](
    const physics_owner_3pcf& owner, // owner of the immutable physical model
    const physics_notebook_array& k, // wavenumbers in h/Mpc
    // redshift coordinates of the supplied values
    const physics_notebook_array& z,
    double amplitude // shared logarithmic feedback response; zero is DMO
  ) {
    const arma::Col<double> values = physics_feedback_power_cpp_3pcf(owner,
      copy_column_physics(k), copy_column_physics(z), amplitude);
    physics_array result(values.n_elem);
    std::memcpy(result.mutable_data(), values.memptr(),
      static_cast<size_t>(values.n_elem)*sizeof(double));
    return result;
  }, py::arg("k"), py::arg("z"), py::arg("amplitude"));

  model.def("bispectrum_response", [](
    const physics_owner_3pcf& owner, // owner of the immutable physical model
    const physics_array& sides, // Fourier triangles [point,3], in h/Mpc
    // redshift at which each physical query is evaluated
    const physics_array& redshifts,
    double amplitude // shared logarithmic feedback response; zero is DMO
  ) {
    physics_ia_3pcf parameters;
    physics_default_ia_3pcf(&parameters);
    parameters.feedback_amplitude = amplitude;
    return points_physics(owner, sides, redshifts, physics_int_array(),
      parameters, 4, 0, false);
  }, py::arg("sides").noconvert(), py::arg("redshifts").noconvert(),
    py::arg("amplitude"));

  model.def("bispectrum_response_notebook", [](
      const physics_owner_3pcf& owner, // owner of the immutable physical model
      // Fourier triangles [point,3], in h/Mpc
      const physics_notebook_array& sides,
      // redshift at which each physical query is evaluated
      const physics_notebook_array& redshifts,
      double amplitude // shared logarithmic feedback response; zero is DMO
  ) {
    physics_ia_3pcf parameters;
    physics_default_ia_3pcf(&parameters);
    parameters.feedback_amplitude = amplitude;
    return points_notebook_physics(owner, sides, redshifts,
      physics_notebook_int(), parameters, 4, 0, false);
  }, py::arg("sides"), py::arg("redshifts"), py::arg("amplitude"));

  model.def("nla_power", [](
    const physics_owner_3pcf& owner, // owner of the immutable physical model
    const physics_array& ell, // angular-multipole nodes
    // ordered source-bin labels for the requested correlation
    const physics_int_array& bins,
    const py::dict& parameters // intrinsic-alignment and feedback parameters
  ) {
    shape_physics(ell, 1);
    shape_physics(bins, 1);
    finite_physics(ell);
    aligned_physics(bins, alignof(int));
    if (!(2 == bins.size())) {
      spdlog::critical("3pcf physics interface: two power bins are required");
      std::exit(1);
    }
    const physics_ia_3pcf ia = parameters_physics(parameters);
    physics_array result(ell.size());
    {
      // Release the GIL for the C work (see points_physics); no
      // Python object may be touched until the scope closes.
      py::gil_scoped_release release;
      physics_nla_power_batch_3pcf(owner.get(), &ia, bins.data()[0],
        bins.data()[1], static_cast<int>(ell.size()), ell.data(),
        result.mutable_data());
    }
    finite_physics(result);
    return result;
  }, py::arg("ell").noconvert(), py::arg("bins").noconvert(),
    py::arg("parameters") = py::dict(),
    "Shear C_ell with NLA and optional paired feedback; no TATT loops.");

  model.def("nla_power_notebook", [](
    const physics_owner_3pcf& owner, // owner of the immutable physical model
    const physics_notebook_array& ell, // angular-multipole nodes
    // ordered source-bin labels for the requested correlation
    const physics_notebook_int& bins,
    const py::dict& parameters // intrinsic-alignment and feedback parameters
  ) {
    const arma::Col<double> values = physics_nla_power_cpp_3pcf(owner,
      copy_column_physics(ell), copy_column_physics(bins),
      parameters_physics(parameters));
    physics_array result(values.n_elem);
    std::memcpy(result.mutable_data(), values.memptr(),
      static_cast<size_t>(values.n_elem)*sizeof(double));
    finite_physics(result);
    return result;
  }, py::arg("ell"), py::arg("bins"), py::arg("parameters") = py::dict());

  model.def("background_matter_fraction", [](
    const physics_owner_3pcf& owner, // owner of the immutable physical model
    // redshift at which each physical query is evaluated
    const physics_array& redshifts
  ) {
    shape_physics(redshifts, 1);
    finite_physics(redshifts);
    physics_array output(redshifts.size());
    for (py::ssize_t node=0; node<redshifts.size(); node++) {
      output.mutable_data()[node] = physics_omega_m_z_3pcf(owner.get(),
        redshifts.data()[node]);
    }
    return output;
  }, py::arg("redshifts").noconvert(),
    "Omega_m(z) used by halo covariance: supplied table when selected, "
    "otherwise the pressureless fraction from H(z). Dimensionless.");

  model.def("radial", [](
    const physics_owner_3pcf& owner // owner of the immutable physical model
  ) {
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
    // The C side lends its row-major [nbin,nlos] kernel tables for the
    // model's lifetime; the loop copies each one into an owned NumPy
    // array of the same layout, so Python cannot dangle into the model.
    for (int index=0; 5 > index; index++) {
      physics_array values = 3 > index ? physics_array(nlos)
        : physics_array({static_cast<py::ssize_t>(nbin),
          static_cast<py::ssize_t>(nlos)});
      std::memcpy(values.mutable_data(), arrays[index],
        static_cast<size_t>(values.size())*sizeof(double));
      finite_physics(values);
      result[names[index]] = values;
    }
    return result;
  }, "Owned copies of the radial tables, keyed z, chi, dchi, lens and "
    "shape: z and chi are the line-of-sight nodes and their comoving "
    "distances, dchi the trapezoidal weights of the distance integral, "
    "lens[nbin,nlos] the lensing efficiency kernel (fastnc's g), and "
    "shape[nbin,nlos] the radial weight of the intrinsic shapes "
    "(fastnc's W_g).");

  // --- 7. SUBMODULE ALIASES ----------------------------------------------
  // threepcf_interface.cpp creates the production and notebook
  // submodules during its registration, which must run before this
  // function. These two lines alias names it already defined:
  // production.PhysicsModel is the same class as module.PhysicsModel,
  // and notebook.PhysicsModel is the copying factory
  // physics_model_notebook.
  module.attr("production").attr("PhysicsModelWithBispectrum")
    = module.attr("physics_model_production_with_bispectrum");
  module.attr("notebook").attr("PhysicsModelWithBispectrum")
    = module.attr("physics_model_notebook_with_bispectrum");
  module.attr("production").attr("PhysicsModel") = module.attr("PhysicsModel");
  module.attr("notebook").attr("PhysicsModel")
    = module.attr("physics_model_notebook");
}

}
