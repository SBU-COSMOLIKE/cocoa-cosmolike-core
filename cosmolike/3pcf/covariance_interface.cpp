#include <climits>
#include <memory>
#include <algorithm>
#include <pybind11/stl.h>
#include "cosmolike/basics.h"
#include "threepcf_interface.hpp"
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
#include "cosmolike/cosmo2D_wrapper.hpp"
#include "physics_interface.hpp"
#include "covariance_wrapper.hpp"

namespace py = pybind11;
namespace cosmolike_interface {

// ---------------------------------------------------------------------------
// Pybind adapters for the 3PCF covariance: Python dictionaries in,
// validated C-kernel calls out. pybind11 is the library that binds
// these C++ functions into Python.
//
//   Python dict of NumPy arrays and scalars
//     -> covariance_arguments (type, shape and domain checks)
//     -> covariance_input_3pcf (documented in covariance_physics_3pcf.h)
//     -> covariance_physical_3pcf / covariance_halos_3pcf /
//        covariance_xi_from_power_3pcf (the C kernels)
//
// Every function registers twice, with one input convention per
// submodule. The production submodule borrows: it takes a view, an
// array object that shares the caller's memory, so the caller must
// pass contiguous C-order float64/int32 arrays, and the adapter holds
// a Python reference that keeps each buffer alive for the length of
// the call. The notebook submodule copies: it accepts sliced or
// strided arrays and writes each one into a private buffer the adapter
// owns, and the copies then travel through the Armadillo wrapper
// (covariance_wrapper.cpp). Neither adapter changes a physical
// equation, adds a halo term, or performs a numerical integral.
//
// Field labels name (source bin, aperture radius) pairs. Row labels
// name observables built from two or three such fields. Keeping these
// two indices separate prevents a reader from mistaking a source-noise
// matrix for a covariance matrix between measurement rows.
// ---------------------------------------------------------------------------



// ============================================================================
// [SECTION] VALIDATION HELPERS
// ============================================================================


// ---------------------------------------------------------------------------
// Element count of an array shape, with overflow guards.
//
// Each dimension must be positive and fit an int, and the running
// product must fit size_t. The final bound divides PY_SSIZE_T_MAX by
// sizeof(double) because Python measures an array in bytes with the
// signed type ssize_t: an element count beyond that bound could never
// become a valid double array on the Python side, even when size_t
// itself could hold it.
//
// Parameters:
//   shape - the array dimensions, as NumPy reports them
//
// Returns:
//   the product of the dimensions: the number of array elements
// ---------------------------------------------------------------------------
static size_t covariance_count(
  // axis lengths of the requested C-order storage
  const std::vector<py::ssize_t>& shape
)
{
  size_t product = 1;

  for (const auto length : shape) {
    if (!(0 < length && INT_MAX >= length)) {
      spdlog::critical("3pcf covariance interface: invalid array dimension");
      std::exit(1);
    }
    if (!(static_cast<size_t>(length) <= SIZE_MAX/product)) {
      spdlog::critical("3pcf covariance interface: array dimension product "
        "overflows");
      std::exit(1);
    }
    product *= static_cast<size_t>(length);
  }

  if (!(product <= static_cast<size_t>(PY_SSIZE_T_MAX)
        /sizeof(double))) {
    spdlog::critical("3pcf covariance interface: array byte count overflows");
    std::exit(1);
  }

  return product;
}


// ---------------------------------------------------------------------------
// One validated input array, as a borrowed view or a private copy.
//
// The object must be a NumPy array of the exact dtype Number and the
// requested rank. In production (notebook = false) the function
// returns a view, an array object that shares the caller's memory: it
// checks that the buffer is contiguous C order (row-major, the last
// index changing fastest) and aligned for the element type, and the
// returned object holds a Python reference that keeps the caller's
// buffer alive while the adapter reads it. In notebook mode the
// function returns a copy that owns its storage, so slices and
// transposed views are welcome.
//
// The copy loop walks the flat output index and decodes it into one
// coordinate per axis, like reading a mixed-radix number: the
// remainder modulo the last dimension is the last coordinate, and the
// quotient continues with the next axis. Each coordinate then
// multiplies that axis's stride - NumPy strides count bytes, not
// elements - to find the source element wherever the view scattered
// it.
//
// Parameters:
//   object   - the Python object the dictionary supplied
//   rank     - the required number of dimensions
//   notebook - false borrows a contiguous buffer; true copies any view
//
// Returns:
//   a C-order array of the requested type: a view of the caller's
//   buffer in production, a private copy in notebook mode
// ---------------------------------------------------------------------------
template <typename Number>
static py::array_t<Number, py::array::c_style> covariance_array(
  const py::handle object, // Python buffer to validate or copy
  const int rank, // required number of array axes
  // true copies notebook inputs; false borrows production arrays
  const bool notebook
)
{
  if (!(py::isinstance<py::array>(object))) {
    spdlog::critical("3pcf covariance interface: numerical inputs must be "
      "NumPy arrays");
    std::exit(1);
  }
  const py::array input = py::reinterpret_borrow<py::array>(object);
  if (!(input.dtype().is(py::dtype::of<Number>()))) {
    spdlog::critical("3pcf covariance interface: array has the wrong dtype; "
      "require float64 or int32 as documented");
    std::exit(1);
  }
  if (!(input.ndim() == rank)) {
    spdlog::critical("3pcf covariance interface: array has the wrong rank");
    std::exit(1);
  }

  std::vector<py::ssize_t> shape;
  for (int axis=0; axis<rank; axis++) {
    shape.push_back(input.shape(axis));
  }
  const size_t count = covariance_count(shape);

  if (!notebook) {
    if (!(0 != (input.flags() & py::array::c_style))) {
      spdlog::critical("3pcf covariance interface: production requires "
        "contiguous C-order inputs");
      std::exit(1);
    }
    const auto address = reinterpret_cast<std::uintptr_t>(input.data());
    if (!(0 == address%alignof(Number))) {
      spdlog::critical("3pcf covariance interface: production input is not "
        "aligned for its C element type");
      std::exit(1);
    }
    return py::reinterpret_borrow<py::array_t<Number, py::array::c_style>>(
      input);
  }

  py::array_t<Number, py::array::c_style> result(shape);
  const char* source = static_cast<const char*>(input.data());
  Number* destination = result.mutable_data();

  for (size_t index=0; index<count; index++) {
    size_t remainder = index;
    py::ssize_t offset = 0;
    for (int axis=rank - 1; 0 <= axis; axis--) {
      const size_t coordinate = remainder%static_cast<size_t>(shape[axis]);
      remainder /= static_cast<size_t>(shape[axis]);
      offset += static_cast<py::ssize_t>(coordinate)*input.strides(axis);
    }
    // A copy through bytes also accepts a valid but unaligned notebook
    // view: no typed dereference happens until the owned buffer exists.
    std::memcpy(destination + index, source + offset, sizeof(Number));
  }

  return result;
}


// Stop when any element is not finite: a NaN or infinity in an input
// or output marks a broken calculation, never valid physics.
static void covariance_finite(
  const py::array_t<double>& array // real array whose entries must be finite
)
{
  for (py::ssize_t index=0; index<array.size(); index++) {
    if (!(std::isfinite(array.data()[index]))) {
      spdlog::critical("3pcf covariance interface: nonfinite numerical "
        "input or output");
      std::exit(1);
    }
  }
}



// ============================================================================
// [SECTION] ARGUMENT PARSING
// ============================================================================

// ---------------------------------------------------------------------------
// covariance_arguments: the validated inputs of one covariance call.
//
// The constructor reads the Python dictionary once, and the object
// then owns every validated view or copy. The raw pointers inside its
// public covariance_input_3pcf member point into those owned arrays,
// so the struct stays valid exactly as long as this object lives; the
// struct's fields are documented in covariance_physics_3pcf.h.
// validate() checks every physical and numerical domain before any
// expensive C calculation starts.
// ---------------------------------------------------------------------------
class covariance_arguments {
 public:
  covariance_input_3pcf input = {}; // pointers into real/integer below
  int nsource = 0;                  // source-bin count of the model
  // owned real-valued views/copies, keyed by dictionary name
  std::map<std::string, py::array_t<double, py::array::c_style>> real;
  // owned integer views/copies, keyed by dictionary name
  std::map<std::string, py::array_t<int, py::array::c_style>> integer;

  covariance_arguments(
    // immutable physical model retained for this call
    const physics_model_3pcf* model,
    const py::dict& values, // named input arrays and scalar settings
    // true copies notebook inputs; false borrows production arrays
    const bool notebook
  )
  {
    // --- 1. KEY SETS ---
    // The sets below name every key the dictionary may carry, by type
    // and rank. real_vectors and real_matrices hold float64 arrays;
    // integer_vectors hold int32; scalars are plain Python numbers.
    // The loop handles theta_edges (optional [nrow,2] real matrix) and
    // row_fields ([nrow,3] integer matrix) by name.
    const std::set<std::string> real_vectors = {
      "ell", "dell", "window_ell", "window_dell", "phi", "dphi",
      "radius", "dr", "profile_radius", "k", "dk", "mass", "dmass",
      "field_radius"
    };
    const std::set<std::string> real_matrices = {
      "theta", "theta_weight", "noise"
    };
    const std::set<std::string> variance_vectors = {
      "halo_z", "halo_mass"
    };
    const std::set<std::string> variance_matrices = {
      "halo_sigma2", "halo_dlnsigma_dlnm"
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
      "baryon_strength", "halo_reference_rho"
    };

    // --- 2. PARSING LOOP ---
    // Walk the dictionary once. Every key must belong to one of the
    // sets above; an unknown key stops the run, so a typo cannot
    // silently leave a default in place. Array values become owned
    // views or copies, and every real array must be finite.
    for (const auto& item : values) {
      if (!(py::isinstance<py::str>(item.first))) {
        spdlog::critical("3pcf covariance interface: input dictionary keys "
          "must be strings");
        std::exit(1);
      }
      const std::string name = py::cast<std::string>(item.first);
      if (real_vectors.count(name)
          || real_matrices.count(name)
          || variance_vectors.count(name)
          || variance_matrices.count(name)
          || name == "theta_edges") {
        const int rank = real_vectors.count(name)
          || variance_vectors.count(name) ? 1 : 2;
        auto array = covariance_array<double>(item.second, rank, notebook);
        covariance_finite(array);
        real.emplace(name, std::move(array));
      } else if (integer_vectors.count(name) || name == "row_fields") {
        const int rank = name == "row_fields" ? 2 : 1;
        auto array = covariance_array<int>(item.second, rank, notebook);
        integer.emplace(name, std::move(array));
      } else {
        if (!(0 != scalars.count(name))) {
          spdlog::critical("3pcf covariance interface: unknown covariance "
            "dictionary key");
          std::exit(1);
        }
      }
    }

    // --- 3. REQUIRED KEYS ---
    // Every array key is required (only theta_edges is optional), and
    // the survey area has no sensible default.
    for (const auto& name : real_vectors) {
      if (!(0 != real.count(name))) {
        spdlog::critical("3pcf covariance interface: missing real input "
          "vector");
        std::exit(1);
      }
    }
    for (const auto& name : real_matrices) {
      if (!(0 != real.count(name))) {
        spdlog::critical("3pcf covariance interface: missing real input "
          "matrix");
        std::exit(1);
      }
    }
    for (const auto& name : integer_vectors) {
      if (!(0 != integer.count(name))) {
        spdlog::critical("3pcf covariance interface: missing integer vector");
        std::exit(1);
      }
    }
    if (!(0 != integer.count("row_fields"))) {
      spdlog::critical("3pcf covariance interface: missing row_fields matrix");
      std::exit(1);
    }
    if (!(values.contains("area"))) {
      spdlog::critical("3pcf covariance interface: survey area is required");
      std::exit(1);
    }

    // --- 4. SOURCE COUNT FROM THE MODEL ---
    // physics_radial_3pcf returns every radial table of the prepared
    // model, but this adapter keeps only the source-bin count nsource,
    // which sizes the noise matrix below. The C engine reads the same
    // tables directly from the model during the calculation.
    int nlos;
    const double* z;
    const double* chi;
    const double* dchi;
    const double* lens;
    const double* shape;
    physics_radial_3pcf(model, &nsource, &nlos, &z, &chi, &dchi, &lens, &shape);

    // --- 5. SIZES AND SHAPES ---
    // Each grid size comes from its node array. The weight array of
    // every quadrature must match its nodes, and the labeled matrices
    // must match the field, row and source counts.
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

    if (!(integer.at("field_source").size() == input.nfield)) {
      spdlog::critical("3pcf covariance interface: field_source and "
        "field_radius lengths disagree");
      std::exit(1);
    }
    if (!(integer.at("row_fields").shape(0) == input.nrow
          && 3 == integer.at("row_fields").shape(1))) {
      spdlog::critical("3pcf covariance interface: row_fields must have "
        "shape [nrow,3]");
      std::exit(1);
    }

    matrix_shape("theta", input.nrow, input.ntheta);
    matrix_shape("theta_weight", input.nrow, input.ntheta);
    matrix_shape("noise", nsource, nsource);

    if (real.count("theta_edges")) {
      matrix_shape("theta_edges", input.nrow, 2);
      input.theta_edges = real.at("theta_edges").data();
    }

    // A variance snapshot is optional, but partial snapshots are not.
    // Its mass axis is retained even though mass already appears above:
    // the C consumer compares them before using these prepared values.
    const char* variance_names[5] = {"halo_z", "halo_mass",
      "halo_sigma2", "halo_dlnsigma_dlnm", "halo_reference_rho"};
    int variance_keys = 0;
    for (const char* name : variance_names) {
      variance_keys += values.contains(name) ? 1 : 0;
    }
    if (0 != variance_keys
        && 5 != variance_keys) {
      spdlog::critical("3pcf covariance interface: supply halo_z, "
        "halo_mass, halo_sigma2, halo_dlnsigma_dlnm and "
        "halo_reference_rho together");
      std::exit(1);
    }
    if (5 == variance_keys) {
      input.nhalo_z = vector_length("halo_z");
      equal_length("halo_mass", input.nmass);
      matrix_shape("halo_sigma2", input.nhalo_z, input.nmass);
      matrix_shape("halo_dlnsigma_dlnm", input.nhalo_z, input.nmass);
      input.halo_z = real.at("halo_z").data();
      input.halo_mass = real.at("halo_mass").data();
      input.halo_sigma2 = real.at("halo_sigma2").data();
      input.halo_dlnsigma_dlnm = real.at("halo_dlnsigma_dlnm").data();
      input.halo_reference_rho = py::cast<double>(values["halo_reference_rho"]);
      if (!std::isfinite(input.halo_reference_rho)
          || 0.0 >= input.halo_reference_rho) {
        spdlog::critical("3pcf covariance interface: halo_reference_rho "
          "must be finite and positive");
        std::exit(1);
      }
    }

    // --- 6. POINTER WIRING ---
    // Point the C struct at the owned buffers. From here on the C
    // kernels read these arrays through covariance_input_3pcf; the
    // maps above keep them alive.
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

    // --- 7. SCALARS AND MODEL SWITCHES ---
    // area must come from the caller (checked above); every other
    // scalar falls back to its documented default when the key is
    // absent.
    input.area = scalar(values, "area", 0.0);
    input.center_window = integer_scalar(values, "center_window", 1);
    input.large_area = integer_scalar(values, "large_area", 0);
    input.estimator = integer_scalar(values, "estimator", 0);
    input.ia_model = integer_scalar(values, "ia_model", 0);
    input.noise_mode = integer_scalar(values, "noise_mode", 0);

    // Sheth-Tormen halo mass function parameters (ellipsoidal collapse,
    // arXiv astro-ph/9901122): a = 0.707 rescales the peak height,
    // p = 0.3 shapes the low-mass tail, and the normalization 0.3222
    // makes the multiplicity function integrate to one, so that all
    // mass sits in halos. delta_c = 1.686 is the spherical-collapse
    // overdensity threshold.
    input.st_a = scalar(values, "st_a", 0.707);
    input.st_p = scalar(values, "st_p", 0.3);
    input.st_norm = scalar(values, "st_norm", 0.3222);
    input.delta_c = scalar(values, "delta_c", 1.686);
    input.collapse_model = integer_scalar(values, "collapse_model", 0);
    input.profile_upsample = integer_scalar(values, "profile_upsample", 8);

    // Duffy et al. concentration-mass relation for M200m halos
    // (arXiv 0804.2486, full-sample fit):
    // c = 10.14 (M/M_pivot)^-0.081 (1+z)^-1.01, pivot 2.0e12 Msun/h.
    input.concentration_a = scalar(values, "concentration_a", 10.14);
    input.concentration_b = scalar(values, "concentration_b", -0.081);
    input.concentration_c = scalar(values, "concentration_c", -1.01);
    input.concentration_pivot = scalar(values, "concentration_pivot", 2.0e12);
    input.baryon_strength = scalar(values, "baryon_strength", 0.0);

    validate();
  }

 private:
  // Length of an owned real vector, as the C struct's int size.
  int vector_length(
    const char* name // dictionary key or array label checked here
  ) const
  {
    return static_cast<int>(real.at(name).size());
  }

  // A quadrature's weight array must have exactly as many entries as
  // its node array.
  void equal_length(
    const char* name, // dictionary key or array label checked here
    const int expected // required array length
  ) const
  {
    if (!(vector_length(name) == expected)) {
      spdlog::critical("3pcf covariance interface: integration nodes and "
        "weights have different lengths");
      std::exit(1);
    }
  }

  // A labeled matrix must match the row and column counts its
  // physical labels promise.
  void matrix_shape(
    const char* name, // dictionary key or array label checked here
    const int rows, // required first-axis length
    const int columns // required second-axis length
  ) const
  {
    const auto& values = real.at(name);
    if (!(values.shape(0) == rows && values.shape(1) == columns)) {
      spdlog::critical("3pcf covariance interface: matrix shape disagrees "
        "with physical labels");
      std::exit(1);
    }
  }

  // Real scalar with a default: an absent key falls back, a present
  // key must be a finite number.
  static double scalar(
    const py::dict& values, // named input arrays and scalar settings
    const char* name, // dictionary key or array label checked here
    const double fallback // default value when the dictionary omits this key
  )
  {
    const double result = values.contains(name)
      ? py::cast<double>(values[name]) : fallback;
    if (!(std::isfinite(result))) {
      spdlog::critical("3pcf covariance interface: scalar input is nonfinite");
      std::exit(1);
    }
    return result;
  }

  // Integer model switch with a default. Only Python integers pass:
  // a float here would hide a mis-set discrete switch.
  static int integer_scalar(
    const py::dict& values, // named input arrays and scalar settings
    const char* name, // dictionary key or array label checked here
    const int fallback // default value when the dictionary omits this key
  )
  {
    if (!values.contains(name)) {
      return fallback;
    }
    if (!(py::isinstance<py::int_>(values[name]))) {
      spdlog::critical("3pcf covariance interface: discrete model settings "
        "must be Python integers");
      std::exit(1);
    }
    return py::cast<int>(values[name]);
  }

  // Every node grid must increase strictly, from a nonnegative (or,
  // when strictly_positive, a positive) lower bound.
  static void increasing(
    const double* values, // integration nodes checked in their stored order
    const int count, // number of entries in the supplied array
    // require positive nodes instead of allowing zero
    const bool strictly_positive
  )
  {
    for (int index=0; index<count; index++) {
      if (!(strictly_positive ? 0.0 < values[index]
            : 0.0 <= values[index])) {
        spdlog::critical("3pcf covariance interface: grid has an invalid "
          "lower bound");
        std::exit(1);
      }
      if (!(0 == index || values[index] > values[index - 1])) {
        spdlog::critical("3pcf covariance interface: integration grid must "
          "be strictly increasing");
        std::exit(1);
      }
    }
  }

  // Quadrature weights must be strictly positive.
  static void positive(
    const double* values, // integration weights that must be positive
    const int count // number of entries in the supplied array
  )
  {
    for (int index=0; index<count; index++) {
      if (!(0.0 < values[index])) {
        spdlog::critical("3pcf covariance interface: integration weights "
          "must be positive");
        std::exit(1);
      }
    }
  }

  // Physical and numerical domain checks, run once at the end of the
  // constructor. The order follows the input struct: grids, weights,
  // angular closure, model switches, field labels, observable rows,
  // and the noise matrix last.
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

    if (!(3 <= input.nprofile && 0.0 == input.profile_radius[0])) {
      spdlog::critical("3pcf covariance interface: profile_radius needs at "
        "least three nodes and must start at zero");
      std::exit(1);
    }
    if (!(1 <= input.profile_upsample
          && input.nprofile - 1 <= (INT_MAX - 1)/input.profile_upsample)) {
      spdlog::critical("3pcf covariance interface: profile upsampling count "
        "is invalid or overflows");
      std::exit(1);
    }
    if (!(2.0*arma::datum::pi >= input.phi[input.nphi - 1])) {
      spdlog::critical("3pcf covariance interface: relative angle lies "
        "outside [0,2pi]");
      std::exit(1);
    }

    // The relative-angle weights must close the full circle: their sum
    // is the measure of [0,2pi]. The 1.0e-10 tolerance absorbs the
    // roundoff of summing many weights while rejecting a genuinely
    // truncated angular range.
    double angle_measure = 0.0;
    for (int index=0; index<input.nphi; index++) {
      angle_measure += input.dphi[index];
    }
    if (!(1.0e-10 > std::abs(angle_measure - 2.0*arma::datum::pi))) {
      spdlog::critical("3pcf covariance interface: relative-angle "
        "quadrature weights must sum to 2pi");
      std::exit(1);
    }

    if (!(0.0 < input.area && 4.0*arma::datum::pi >= input.area)) {
      spdlog::critical("3pcf covariance interface: survey solid angle must "
        "lie in (0,4pi]");
      std::exit(1);
    }
    if (!(1 == input.center_window || 2 == input.center_window)) {
      spdlog::critical("3pcf covariance interface: center_window must be 1 "
        "(circle) or 2 (square)");
      std::exit(1);
    }
    if (!(0 == input.large_area || 1 == input.large_area)) {
      spdlog::critical("3pcf covariance interface: large_area must be zero "
        "or one");
      std::exit(1);
    }
    if (!(0 == input.estimator && 0 == input.ia_model)) {
      spdlog::critical("3pcf covariance interface: only full-filter map "
        "products with gravitational covariance exist");
      std::exit(1);
    }
    if (!(0 == input.noise_mode || 1 == input.noise_mode)) {
      spdlog::critical("3pcf covariance interface: noise_mode must be "
        "finite-band (0) or uniform-area xi noise (1)");
      std::exit(1);
    }
    if (!(0 == input.noise_mode
          || (1 == input.large_area && nullptr != input.theta_edges))) {
      spdlog::critical("3pcf covariance interface: analytic xi noise "
        "requires large_area=1 and theta_edges[nrow,2]");
      std::exit(1);
    }
    if (!(0.0 < input.st_a && 0.0 <= input.st_p
          && 0.0 < input.st_norm && 0.0 < input.delta_c
          && 0.0 < input.concentration_a && 0.0 < input.concentration_pivot)) {
      spdlog::critical("3pcf covariance interface: invalid halo prescription");
      std::exit(1);
    }
    if (!(0 == input.collapse_model || 1 == input.collapse_model)) {
      spdlog::critical("3pcf covariance interface: collapse_model must "
        "select constant (0) or evolving (1) threshold");
      std::exit(1);
    }

    for (int field=0; field<input.nfield; field++) {
      if (!(0 <= input.field_source[field]
            && input.field_source[field] < nsource)) {
        spdlog::critical("3pcf covariance interface: field source index is "
          "out of range");
        std::exit(1);
      }
      if (!(0.0 <= input.field_radius[field])) {
        spdlog::critical("3pcf covariance interface: field aperture radius "
          "must be nonnegative");
        std::exit(1);
      }
    }

    double largest_theta = 0.0;
    for (int row=0; row<input.nrow; row++) {
      const int kind = input.row_kind[row];
      if (!(0 <= kind && 3 >= kind)) {
        spdlog::critical("3pcf covariance interface: unknown observable row "
          "kind");
        std::exit(1);
      }

      const int legs = 1 == kind ? 3 : 2;
      for (int leg=0; leg<legs; leg++) {
        const int field = input.row_fields[static_cast<size_t>(row)*3 + leg];
        if (!(0 <= field && field < input.nfield)) {
          spdlog::critical("3pcf covariance interface: observable field "
            "index is out of range");
          std::exit(1);
        }
        if (!(2 > kind ? 0.0 < input.field_radius[field]
              : 0.0 == input.field_radius[field])) {
          spdlog::critical("3pcf covariance interface: Map rows need "
            "positive filters and xi rows need bare shear "
            "legs");
          std::exit(1);
        }
      }

      if (2 <= kind) {
        if (1 == input.noise_mode) {
          const double lower = input.theta_edges[static_cast<size_t>(row)*2];
          const double upper = input.theta_edges[
            static_cast<size_t>(row)*2 + 1];
          if (!(0.0 < lower && upper > lower)) {
            spdlog::critical("3pcf covariance interface: analytic xi noise "
              "needs positive ordered bin edges");
            std::exit(1);
          }
          for (int node=0; node<input.ntheta; node++) {
            const double theta = input.theta[
              static_cast<size_t>(row)*input.ntheta + node];
            if (!(theta >= lower && theta <= upper)) {
              spdlog::critical("3pcf covariance interface: xi node lies "
                "outside its declared analytic-noise bin");
              std::exit(1);
            }
          }
        }

        double normalization = 0.0;
        for (int node=0; node<input.ntheta; node++) {
          const size_t index = static_cast<size_t>(row)*input.ntheta + node;
          if (!(0.0 < input.theta[index]
                && 0.0 <= input.theta_weight[index])) {
            spdlog::critical("3pcf covariance interface: xi nodes must be "
              "positive and their weights nonnegative");
            std::exit(1);
          }
          largest_theta = std::max(largest_theta, input.theta[index]);
          normalization += input.theta_weight[index];
        }
        // A bin average needs weights that sum to one, or the xi value
        // would carry a spurious overall scale. 1.0e-10 forgives the
        // roundoff of accumulating ntheta weights, nothing larger.
        if (!(1.0e-10 >= std::abs(normalization - 1.0))) {
          spdlog::critical("3pcf covariance interface: xi-bin weights must "
            "sum to one");
          std::exit(1);
        }
      }
    }

    if (!(input.profile_radius[input.nprofile - 1]
          >= input.radius[input.nr - 1] + 0.5*largest_theta)) {
      spdlog::critical("3pcf covariance interface: profile grid does not "
        "cover shifted xi/shear evaluations");
      std::exit(1);
    }

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
        // The two orderings of a source pair must state the same
        // covariance. Comparing against 64 machine epsilons of the
        // larger entry (machine epsilon is the gap between 1 and the
        // next double) forgives the roundoff of building the matrix in
        // Python while rejecting a transposed or corrupted input.
        if (!(std::abs(value - reverse)
              <= 64.0*std::numeric_limits<double>::epsilon()*scale)) {
          spdlog::critical("3pcf covariance interface: noise matrix is not "
            "symmetric");
          std::exit(1);
        }
        noise(row, column) = value;
        magnitude = std::max(magnitude, std::abs(value));
      }
    }

    arma::Col<double> eigenvalues;
    if (!(arma::eig_sym(eigenvalues, noise))) {
      spdlog::critical("3pcf covariance interface: noise eigenvalue check "
        "failed");
      std::exit(1);
    }
    if (!(eigenvalues.min() >= -64.0*nsource
          *std::numeric_limits<double>::epsilon()*magnitude)) {
      spdlog::critical("3pcf covariance interface: noise matrix is not "
        "positive semidefinite");
      std::exit(1);
    }
  }
};



// ============================================================================
// [SECTION] PYTHON BINDINGS
// ============================================================================

// ---------------------------------------------------------------------------
// Mean and covariance terms of the gravitational map products.
//
// The dictionary becomes a covariance_arguments object, the C engine
// runs without the GIL, and the six covariance families come back in
// one [6,nrow,nrow] array: Gaussian, bispectrum-squared (BB), power
// times one-halo trispectrum, power times bispectrum, connected
// one-halo, and the chosen halo-abundance response, in the order
// covariance_physics_3pcf.h documents.
//
// Parameters:
//   owner    - the Python object that owns the prepared physics model
//   values   - the input dictionary (arrays, scalars, model switches)
//   notebook - false borrows production buffers; true copies views
//
// Returns:
//   (mean[nrow], terms[6,nrow,nrow]) as a tuple of NumPy arrays
// ---------------------------------------------------------------------------
static py::tuple physical_covariance_binding(
  const py::handle owner, // owner of the immutable model or projected grid
  const py::dict& values, // named input arrays and scalar settings
  // true copies notebook inputs; false borrows production arrays
  const bool notebook
)
{
  const physics_model_3pcf* model = borrow_physics_model_3pcf(owner);
  covariance_arguments arguments(model, values, notebook);

  const int count = arguments.input.nrow;
  // The 6 counts the covariance families listed above. The call's
  // return value is discarded: it runs only to prove, before any
  // allocation, that the [6,nrow,nrow] element and byte counts fit
  // their integer types.
  covariance_count({6, count, count});

  py::array_t<double> mean(count);
  py::array_t<double> terms({static_cast<py::ssize_t>(6),
    static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(count)});

  if (notebook) {
    const covariance_request_cpp_3pcf request(arguments.input,
      arguments.nsource);
    std::pair<arma::Col<double>, arma::Cube<double>> result;

    // Python's global interpreter lock (the GIL) lets only one thread
    // run Python code at a time. Releasing it here lets other Python
    // threads run during the long C calculation. The price is a hard
    // rule: while the lock is released, this code must not touch any
    // Python object. The braces keep only pure C++ inside, and the
    // lock returns before the loop below fills the NumPy outputs.
    {
      py::gil_scoped_release release;
      result = covariance_physical_cpp_3pcf(model, request);
    }

    // Repack the Armadillo cube, indexed (row, column, term), into the
    // C layout [term,row,column] of the NumPy output, mirroring the
    // layout note in covariance_wrapper.cpp. Naming all three indices
    // keeps the six families in their documented order.
    for (int row=0; row<count; row++) {
      mean.mutable_data()[row] = result.first(row);
      for (int column=0; column<count; column++) {
        for (int term=0; 6 > term; term++) {
          terms.mutable_data()[(static_cast<size_t>(term)*count + row)*count
            + column] = result.second(row, column, term);
        }
      }
    }
  } else {
    // Release the GIL while the C kernel runs (the note above states
    // the rule this scope obeys).
    py::gil_scoped_release release;
    covariance_physical_3pcf(model, &arguments.input, mean.mutable_data(),
      terms.mutable_data());
  }

  covariance_finite(mean);
  covariance_finite(terms);

  return py::make_tuple(mean, terms);
}


// ---------------------------------------------------------------------------
// Halo ingredient tables on a (redshift, mass) grid.
//
// The same validated dictionary sets the halo prescription; z and
// k_profile set the output grids. The engine fills five tables: the
// mass variance sigma^2(M,z), the halo abundance dn/dM, the linear
// halo bias and the Duffy et al. concentration, each with shape
// [z,M], and the normalized NFW Fourier profile u(k|M,z) with shape
// [z,M,k].
//
// Parameters:
//   owner       - the Python object that owns the physics model
//   values      - the covariance input dictionary
//   redshifts   - nonnegative, strictly increasing z grid
//   wavenumbers - nonnegative, strictly increasing k grid, h/Mpc
//   notebook    - false borrows production buffers; true copies views
//
// Returns:
//   (sigma2, dndm, bias, concentration, u) as NumPy arrays with the
//   shapes above
// ---------------------------------------------------------------------------
static py::tuple halos_binding(
  const py::handle owner, // owner of the immutable model or projected grid
  const py::dict& values, // named input arrays and scalar settings
  const py::handle redshifts, // redshift nodes for the halo tables
  const py::handle wavenumbers, // profile wavenumbers in h/Mpc
  // true copies notebook inputs; false borrows production arrays
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
    if (!(0.0 <= z.data()[index]
          && (0 == index || z.data()[index] > z.data()[index - 1]))) {
      spdlog::critical("3pcf covariance interface: halo redshifts must be "
        "nonnegative and strictly increasing");
      std::exit(1);
    }
  }
  for (py::ssize_t index=0; index<k.size(); index++) {
    if (!(0.0 <= k.data()[index]
          && (0 == index || k.data()[index] > k.data()[index - 1]))) {
      spdlog::critical("3pcf covariance interface: profile wavenumbers must "
        "be nonnegative and strictly increasing");
      std::exit(1);
    }
  }

  const int mass_count = arguments.input.nmass;

  // Discard the returned count: the call runs only to prove the
  // largest output, [z,M,k], fits its integer types before any
  // allocation.
  covariance_count({z.size(), mass_count, k.size()});

  // The first four outputs share the [z,M] grid: sigma2 (the mass
  // variance), dndm (the halo abundance), bias and concentration.
  // The fifth is the NFW Fourier profile u on [z,M,k].
  std::vector<py::array_t<double>> result;
  for (int index=0; 4 > index; index++) {
    result.emplace_back(std::vector<py::ssize_t>{z.size(), mass_count});
  }
  result.emplace_back(std::vector<py::ssize_t>{z.size(), mass_count, k.size()});

  if (notebook) {
    const covariance_request_cpp_3pcf request(arguments.input,
      arguments.nsource);
    const arma::Col<double> redshift(z.data(), z.size());
    const arma::Col<double> wave(k.data(), k.size());
    std::vector<arma::Col<double>> computed;

    // Release the GIL while the C engine runs; the paragraph in
    // physical_covariance_binding states the rule this scope obeys.
    {
      py::gil_scoped_release release;
      computed = covariance_halos_cpp_3pcf(model, request, redshift, wave);
    }

    // The wrapper returns each table as one flat Armadillo column
    // already in C row-major order (covariance_wrapper.hpp), so a
    // plain byte copy lands every element in its [z,M] or [z,M,k]
    // slot. No two-dimensional Armadillo matrix enters, so no
    // column-major transpose can occur.
    for (int index=0; 5 > index; index++) {
      std::memcpy(result[index].mutable_data(), computed[index].memptr(),
        static_cast<size_t>(result[index].size())*sizeof(double));
    }
  } else {
    // Release the GIL while the C engine runs (rule stated in
    // physical_covariance_binding).
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


// ---------------------------------------------------------------------------
// Bin-averaged shear correlation functions from E/B angular spectra.
//
// For each row the C kernel integrates its spectra against a Bessel
// kernel at every separation node and averages over the bin: sign +1
// forms xi+ from J0 and C_ell^E + C_ell^B, and sign -1 forms xi- from
// J4 and C_ell^E - C_ell^B, with C_ell^E and C_ell^B the E/B-mode
// angular power spectra and J0, J4 Bessel functions of the first
// kind. This binding validates shapes and domains; the physics lives
// in covariance_xi_from_power_3pcf.
//
// Parameters:
//   ell_object    - multipole nodes [nell], increasing and positive
//   dell_object   - dell quadrature weights [nell], positive
//   theta_object  - separation nodes [nrow,ntheta], radians
//   weight_object - bin weights [nrow,ntheta], each row sums to one
//   e_object      - E-mode spectra [nrow,nell]
//   b_object      - B-mode spectra [nrow,nell]
//   sign_object   - +1 (xi+) or -1 (xi-) per row [nrow]
//   notebook      - false borrows production buffers; true copies
//
// Returns:
//   NumPy array [nrow]: the bin-averaged xi of each row
// ---------------------------------------------------------------------------
static py::array_t<double> xi_binding(
  const py::handle ell_object, // angular-multipole quadrature nodes
  const py::handle dell_object, // weights for integration over ell
  const py::handle theta_object, // separations [row,node], in radians
  // normalized separation-bin weights [row,node]
  const py::handle weight_object,
  const py::handle e_object, // E-mode angular power [row,ell]
  const py::handle b_object, // B-mode angular power [row,ell]
  const py::handle sign_object, // +1 for xi+, -1 for xi-
  // true copies notebook inputs; false borrows production arrays
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

  // One chained check compares five arrays against the reference
  // shapes of ell and theta. When it fails, compare in this order:
  // dell length vs nell; theta_weight shape vs [nrow,ntheta];
  // spectra_e shape vs [nrow,nell]; spectra_b shape vs [nrow,nell];
  // sign length vs nrow.
  if (!(dell.size() == nell && weight.shape(0) == nrow
        && weight.shape(1) == ntheta && spectra_e.shape(0) == nrow
        && spectra_e.shape(1) == nell && spectra_b.shape(0) == nrow
        && spectra_b.shape(1) == nell && sign.size() == nrow)) {
    spdlog::critical("3pcf covariance interface: xi input shapes disagree");
    std::exit(1);
  }
  for (int node=0; node<nell; node++) {
    if (!(0.0 < ell.data()[node] && 0.0 < dell.data()[node]
          && (0 == node || ell.data()[node] > ell.data()[node - 1]))) {
      spdlog::critical("3pcf covariance interface: xi Fourier nodes must "
        "increase and have positive weights");
      std::exit(1);
    }
  }

  for (int row=0; row<nrow; row++) {
    if (!(1 == sign.data()[row] || -1 == sign.data()[row])) {
      spdlog::critical("3pcf covariance interface: xi sign must be +1 or -1");
      std::exit(1);
    }

    double normalization = 0.0;
    for (int node=0; node<ntheta; node++) {
      const size_t index = static_cast<size_t>(row)*ntheta + node;
      if (!(0.0 < theta.data()[index] && 0.0 <= weight.data()[index])) {
        spdlog::critical("3pcf covariance interface: xi separation nodes "
          "must be positive and weights nonnegative");
        std::exit(1);
      }
      normalization += weight.data()[index];
    }
    // A bin average needs weights that sum to one (the same rule and
    // 1.0e-10 roundoff allowance as in covariance_arguments).
    if (!(1.0e-10 >= std::abs(normalization - 1.0))) {
      spdlog::critical("3pcf covariance interface: xi-bin weights must sum "
        "to one");
      std::exit(1);
    }
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
    // Release the GIL while the C kernel runs (rule stated in
    // physical_covariance_binding).
    py::gil_scoped_release release;
    covariance_xi_from_power_3pcf(nrow, nell, ntheta, ell.data(), dell.data(),
      theta.data(), weight.data(), spectra_e.data(), spectra_b.data(),
      sign.data(), result.mutable_data());
  }

  covariance_finite(result);

  return result;
}



// ============================================================================
// [SECTION] MODULE REGISTRATION
// ============================================================================

// ---------------------------------------------------------------------------
// Register the covariance bindings on the production and notebook
// submodules.
//
// The two-pass loop runs the same registrations twice: copying = false
// defines the production versions, which borrow contiguous buffers,
// and copying = true defines the notebook versions, which copy any
// view. Each lambda captures only that flag and forwards to its
// binding above, so the two submodules expose identical signatures
// with different input conventions.
//
// Parameters:
//   module - the extension module; its production and notebook
//            submodules already exist
//
// Returns:
//   nothing; the three functions appear in both submodules
// ---------------------------------------------------------------------------
// Fixed Gauss-Legendre rules use the same GSL allocator as the rest of
// CosmoLike. Weights include dx: for t=ln(x), dx=x dt exactly once.
static py::tuple quadrature_3pcf(
  double lower, // integration lower endpoint
  double upper, // integration upper endpoint
  int count, // quadrature order
  bool logarithmic // integrate uniformly in ln(x)
)
{
  if (!std::isfinite(lower) || !std::isfinite(upper) || lower >= upper
      || count < 2 || (logarithmic && lower <= 0.0)) {
    throw py::value_error("Quadrature requires ordered finite bounds and count>=2.");
  }
  const double left = logarithmic ? std::log(lower) : lower;
  const double right = logarithmic ? std::log(upper) : upper;
  std::unique_ptr<gsl_integration_glfixed_table,
    decltype(&gsl_integration_glfixed_table_free)> rule(
      malloc_gslint_glfixed(count), &gsl_integration_glfixed_table_free);
  py::array_t<double> nodes(count), weights(count);
  for (int i=0; i<count; i++) {
    double x, w;
    gsl_integration_glfixed_point(left, right, i, &x, &w, rule.get());
    if (logarithmic) {
      x = std::exp(x);
      w *= x;
    }
    nodes.mutable_data()[i] = x;
    weights.mutable_data()[i] = w;
  }
  return py::make_tuple(nodes, weights);
}

static py::array_t<double> shape_noise_3pcf(
  const py::array_t<double>& density, // effective sources per square arcminute
  const py::array_t<double>& dispersion // rms of ONE ellipticity component
)
{
  const auto n = covariance_array<double>(density, 1, true);
  const auto sigma = covariance_array<double>(dispersion, 1, true);
  if (n.size() < 1 || sigma.size() != n.size()) {
    throw py::value_error("Shape-noise arrays must have matching source bins.");
  }
  const int bins = n.size();
  py::array_t<double> noise({bins, bins});
  std::fill_n(noise.mutable_data(), noise.size(), 0.0);
  const double arcmin = M_PI/(180.0*60.0);
  for (int i=0; i<bins; i++) {
    if (!(n.data()[i] > 0.0) || !(sigma.data()[i] > 0.0)) {
      throw py::value_error("Density and component dispersion must be positive.");
    }
    noise.mutable_data()[i*bins+i] =
      sigma.data()[i]*sigma.data()[i]/(n.data()[i]/(arcmin*arcmin));
  }
  return noise;
}

// Each xi row averages over its actual annulus. Uniform-area weights are
// theta dtheta; measured weights replace the whole discrete rule. Map3
// rows carry unused dummy theta entries for the combined covariance ABI.
static py::tuple xi_bin_weights_3pcf(
  const std::vector<std::string>& kind, // xip, xim, map3 per row
  const py::array_t<double>& angles, // [row,3], xi lower/upper edges in radians
  int count, // separation quadrature order
  const py::object& pair_weight // uniform_area or [row,count] weights
)
{
  const auto edges = covariance_array<double>(angles, 2, true);
  const int nrow = kind.size();
  if (count < 2 || edges.shape(0) != nrow || edges.shape(1) != 3) {
    throw py::value_error("Xi bins require [row,3] edges and count>=2.");
  }
  const bool uniform = py::isinstance<py::str>(pair_weight);
  py::array_t<double, py::array::c_style|py::array::forcecast> supplied;
  if (uniform) {
    if (py::cast<std::string>(pair_weight) != "uniform_area") {
      throw py::value_error("Pair-weight name must be uniform_area.");
    }
  } else {
    supplied = decltype(supplied)::ensure(pair_weight);
    if (!supplied || supplied.ndim() != 2 || supplied.shape(0) != nrow
        || supplied.shape(1) != count) {
      throw py::value_error("Explicit pair weights must have shape [row,count].");
    }
    for (py::ssize_t i=0; i<supplied.size(); i++) {
      const double value = supplied.data()[i];
      if (!std::isfinite(value) || value < 0.0) {
        throw py::value_error("Explicit pair weights must be finite and nonnegative.");
      }
    }
  }
  py::array_t<double> theta({nrow, count}), weights({nrow, count});
  for (int row=0; row<nrow; row++) {
    if (kind[row] == "map3") {
      std::fill_n(theta.mutable_data()+row*count, count, 1.0);
      std::fill_n(weights.mutable_data()+row*count, count, 1.0/count);
      continue;
    }
    if (kind[row] != "xip" && kind[row] != "xim") {
      throw py::value_error("Unknown measurement row kind.");
    }
    const auto rule = quadrature_3pcf(edges.data()[3*row],
      edges.data()[3*row+1], count, false);
    const auto x = rule[0].cast<py::array_t<double>>();
    const auto w = rule[1].cast<py::array_t<double>>();
    double norm = 0.0;
    for (int i=0; i<count; i++) {
      theta.mutable_data()[row*count+i] = x.data()[i];
      const double weight = uniform ? w.data()[i]*x.data()[i] :
        supplied.data()[row*count+i];
      weights.mutable_data()[row*count+i] = weight;
      norm += weight;
    }
    if (!(norm > 0.0) || !std::isfinite(norm)) {
      throw py::value_error("Xi row has zero or nonfinite pair-weight normalization.");
    }
    for (int i=0; i<count; i++) {
      weights.mutable_data()[row*count+i] /= norm;
    }
  }
  return py::make_tuple(theta, weights);
}

// Prepare the existing C covariance input dictionary from explicit survey
// arrays. Unique (source,radius) pairs identify filtered fields shared by
// different rows. This changes no estimator and assumes no survey mask.
static py::dict covariance_inputs_3pcf(
  const std::vector<std::string>& kind, // measurement type per row
  const py::array_t<int, py::array::c_style|py::array::forcecast>& bins, // source indices
  const py::array_t<double>& angles, // [row,3], radians
  const py::array_t<double>& noise, // [source,source], angular power
  const py::dict& grids, // baseline quadratures and profile radii
  const py::dict& physical, // physical covariance settings
  int theta_count, // baseline xi separation quadrature order
  const py::object& pair_weight, // uniform_area or measured discrete weights
  const py::object& accuracy // global and individual refinement controls
)
{
  const int nrow = kind.size();
  const auto radii = covariance_array<double>(angles, 2, true);
  if (bins.ndim() != 2 || bins.shape(0) != nrow || bins.shape(1) != 3
      || radii.shape(0) != nrow || radii.shape(1) != 3 || nrow == 0) {
    throw py::value_error("Covariance rows need [row,3] bins and angles.");
  }
  auto result = py::dict(physical.attr("copy")());
  const auto controls = init_accuracy_3pcf(accuracy);
  const char* names[] = {"ell", "phi", "radius", "k", "mass", "window_ell"};
  const char* weights[] = {"dell", "dphi", "dr", "dk", "dmass", "window_dell"};
  for (int i=0; i<6; i++) {
    const std::string stage = i == 5 ? "covariance_window_boost" :
      std::string("covariance_")+names[i]+"_boost";
    if (!grids.contains(names[i])) {
      throw py::value_error(std::string("Missing covariance quadrature: ")+names[i]);
    }
    const auto settings = accuracy_quadrature_3pcf(controls,
      py::cast<py::dict>(grids[names[i]]), stage);
    const auto rule = quadrature_3pcf(py::cast<double>(settings["lower"]),
      py::cast<double>(settings["upper"]), py::cast<int>(settings["count"]),
      settings.contains("logarithmic") && py::cast<bool>(settings["logarithmic"]));
    result[names[i]] = rule[0];
    result[weights[i]] = rule[1];
  }
  const auto profile = covariance_array<double>(grids["profile_radius"], 1, true);
  if (profile.size() < 2 || profile.data()[0] != 0.0) {
    throw py::value_error("Profile radius needs increasing nodes from zero.");
  }
  const double factor = py::cast<double>(controls["accuracy_boost"])*
    py::cast<double>(controls["profile_boost"]);
  if (!std::isfinite(factor) || factor > 1024.0) {
    throw py::value_error("Profile subdivision exceeds 1024.");
  }
  const int subdivision = std::pow(2.0, std::ceil(std::log2(factor)));
  py::array_t<double> refined((profile.size()-1)*subdivision+1);
  for (int i=0; i<profile.size()-1; i++) {
    const double width = profile.data()[i+1]-profile.data()[i];
    if (!(width > 0.0)) {
      throw py::value_error("Profile radius must be increasing.");
    }
    for (int j=0; j<subdivision; j++) {
      refined.mutable_data()[i*subdivision+j] =
        profile.data()[i]+width*(double(j)/subdivision);
    }
  }
  refined.mutable_data()[refined.size()-1] = profile.data()[profile.size()-1];
  result["profile_radius"] = refined;
  result["profile_upsample"] = accuracy_count_3pcf(controls,
    physical.contains("profile_upsample") ? py::cast<int>(physical["profile_upsample"]) : 8,
    "profile_upsample_boost", false, false);
  result["noise"] = py::array(noise.attr("copy")());
  std::vector<std::pair<int,double>> fields;
  py::array_t<int> row_fields({nrow, 3}), row_kind(nrow);
  std::fill_n(row_fields.mutable_data(), row_fields.size(), 0);
  for (int row=0; row<nrow; row++) {
    const bool aperture = kind[row] == "map3";
    if (!aperture && kind[row] != "xip" && kind[row] != "xim") {
      throw py::value_error("Unsupported covariance row kind.");
    }
    row_kind.mutable_data()[row] = aperture ? 1 : (kind[row] == "xip" ? 2 : 3);
    for (int leg=0; leg<(aperture ? 3 : 2); leg++) {
      const int source = bins.data()[row*3+leg];
      const double radius = aperture ? radii.data()[row*3+leg] : 0.0;
      if (source < 0 || radius < 0.0) {
        throw py::value_error("Covariance fields require nonnegative bins/radii.");
      }
      const std::pair<int,double> field(source, radius);
      const auto found = std::find(fields.begin(), fields.end(), field);
      row_fields.mutable_data()[row*3+leg] = found-fields.begin();
      if (found == fields.end()) {
        fields.push_back(field);
      }
    }
  }
  py::array_t<int> field_source(fields.size());
  py::array_t<double> field_radius(fields.size());
  for (size_t i=0; i<fields.size(); i++) {
    field_source.mutable_data()[i] = fields[i].first;
    field_radius.mutable_data()[i] = fields[i].second;
  }
  result["field_source"] = field_source;
  result["field_radius"] = field_radius;
  result["row_fields"] = row_fields;
  result["row_kind"] = row_kind;
  const int resolved = accuracy_count_3pcf(controls, theta_count,
    "xi_theta_boost", true, false);
  if (resolved != theta_count && !py::isinstance<py::str>(pair_weight)) {
    throw py::value_error("Refining theta nodes requires new measured pair weights.");
  }
  const auto theta = xi_bin_weights_3pcf(kind, angles, resolved, pair_weight);
  result["theta"] = theta[0];
  result["theta_weight"] = theta[1];
  if (physical.contains("noise_mode") && py::cast<int>(physical["noise_mode"]) == 1) {
    if (!py::isinstance<py::str>(pair_weight)
        || py::cast<std::string>(pair_weight) != "uniform_area"
        || !physical.contains("large_area") || py::cast<int>(physical["large_area"]) != 1) {
      throw py::value_error("Analytic xi NN requires uniform_area and large_area.");
    }
    py::array_t<double> edges({nrow,2});
    std::fill_n(edges.mutable_data(), edges.size(), 0.0);
    for (int row=0; row<nrow; row++) {
      if (kind[row] != "map3") {
        edges.mutable_data()[2*row] = radii.data()[3*row];
        edges.mutable_data()[2*row+1] = radii.data()[3*row+1];
      }
    }
    result["theta_edges"] = edges;
  }
  return result;
}
// Select the requested source pairs from the ordinary batch calculation.
// The existing two-point wrapper prepares its tables serially before the
// OpenMP read loops. No second TATT spectrum or IA convention is introduced.
static py::tuple shear_power_rows_3pcf(
  const py::array_t<double>& ell, // angular Fourier multipoles
  const py::array_t<int, py::array::c_style|py::array::forcecast>& bins
)
{
  const auto nodes = covariance_array<double>(ell, 1, true);
  if (bins.ndim() != 2 || bins.shape(1) != 2 || bins.shape(0) == 0) {
    throw py::value_error("Shear pairs require a nonempty [row,2] array.");
  }
  const arma::Col<double> copied(nodes.data(), nodes.size());
  const auto cubes = C_ss_tomo_limber_cpp(copied);
  const auto even = covariance_array<double>(cubes[0], 3, true);
  const auto odd = covariance_array<double>(cubes[1], 3, true);
  const int nrow = bins.shape(0), nell = nodes.size(), nbin = even.shape(1);
  py::array_t<double> e({nrow,nell}), b({nrow,nell});
  for (int row=0; row<nrow; row++) {
    const int first = std::min(bins.data()[2*row], bins.data()[2*row+1]);
    const int second = std::max(bins.data()[2*row], bins.data()[2*row+1]);
    if (first < 0 || second >= nbin) {
      throw py::value_error("Requested shear bin is outside the current sample.");
    }
    for (int l=0; l<nell; l++) {
      e.mutable_data()[row*nell+l] = even.data()[(l*nbin+first)*nbin+second];
      b.mutable_data()[row*nell+l] = odd.data()[(l*nbin+first)*nbin+second];
    }
  }
  return py::make_tuple(e,b);
}

// The isolated NLA comparison uses exactly the same native source kernels
// as Map3. Production joint TATT uses the ordinary-core batch above.
static py::array_t<double> nla_power_rows_3pcf(
  const py::object& model, // immutable physical input owner
  const py::array_t<double>& ell, // Fourier multipoles
  const py::array_t<int, py::array::c_style|py::array::forcecast>& bins,
  const py::dict& ia // NLA parameters; nonlinear IA requires CFASTPT
)
{
  const auto parameters = parameters_physics(ia);
  if (parameters.a2 != 0.0 || parameters.bta != 0.0
      || parameters.ct != 0.0 || parameters.fb != 0.0) {
    throw py::value_error("Isolated NLA xi cannot include quadratic IA or fb.");
  }
  const auto nodes = covariance_array<double>(ell,1,true);
  if (bins.ndim() != 2 || bins.shape(1) != 2 || bins.shape(0) == 0) {
    throw py::value_error("NLA source pairs require [row,2].");
  }
  const auto* physical = borrow_physics_model_3pcf(model);
  const int nrow=bins.shape(0), nell=nodes.size();
  py::array_t<double> result({nrow,nell});
  std::map<std::pair<int,int>, int> previous;
  for (int row=0; row<nrow; row++) {
    const int first = std::min(bins.data()[2*row], bins.data()[2*row+1]);
    const int second = std::max(bins.data()[2*row], bins.data()[2*row+1]);
    const std::pair<int,int> pair(first,second);
    const auto found = previous.find(pair);
    if (found == previous.end()) {
      physics_nla_power_batch_3pcf(physical,&parameters,first,second,nell,
        nodes.data(),result.mutable_data()+row*nell);
      previous[pair]=row;
    } else {
      std::copy_n(result.data()+found->second*nell,nell,
        result.mutable_data()+row*nell);
    }
  }
  return result;
}

void bind_threepcf_covariance(
  py::module_& module // Python module receiving the bindings
)
{
  module.def("quadrature", &quadrature_3pcf,
    py::arg("lower"), py::arg("upper"), py::arg("count"),
    py::arg("logarithmic")=false);
  module.def("shape_noise", &shape_noise_3pcf,
    py::arg("number_density"), py::arg("sigma_component"));
  module.def("xi_bin_weights", &xi_bin_weights_3pcf,
    py::arg("kind"), py::arg("angles"), py::arg("count"),
    py::arg("pair_weight"));
  module.def("covariance_inputs", &covariance_inputs_3pcf,
    py::arg("kind"), py::arg("bins"), py::arg("angles"),
    py::arg("noise"), py::arg("grids"), py::arg("physical"),
    py::arg("theta_count"), py::arg("pair_weight"),
    py::arg("accuracy")=py::none());
  module.def("shear_power_rows", &shear_power_rows_3pcf,
    py::arg("ell"), py::arg("bins"));
  module.def("nla_power_rows", &nla_power_rows_3pcf,
    py::arg("model"), py::arg("ell"), py::arg("bins"), py::arg("ia"));

  py::module_ production = module.attr("production").cast<py::module_>();
  py::module_ notebook = module.attr("notebook").cast<py::module_>();

  // One pass per input convention: false registers into production
  // (borrowing), true into notebook (copying).
  for (const bool copying : {false, true}) {
    py::module_ target = copying ? notebook : production;
    target.def("physical_covariance",
      [copying](
        py::handle model, // immutable physical model retained for this call
        // covariance grids, row definitions and physical settings
        const py::dict& inputs
      ) {
        return physical_covariance_binding(model, inputs, copying);
      }, py::arg("model"), py::arg("inputs"),
      "Mean data vector mean[nrow] and covariance terms[6,nrow,nrow] of "
      "the gravitational map products. The six terms are, in order: "
      "Gaussian, bispectrum-squared (BB), power times one-halo "
      "trispectrum, power times bispectrum, connected one-halo, and the "
      "selected halo-abundance response.");

    target.def("halo_tables",
      [copying](
        py::handle model, // immutable physical model retained for this call
        // covariance grids, row definitions and physical settings
        const py::dict& inputs,
        py::handle z, // redshift coordinates of the supplied values
        py::handle k_profile // profile wavenumbers in h/Mpc
      ) {
        return halos_binding(model, inputs, z, k_profile, copying);
      }, py::arg("model"), py::arg("inputs"), py::arg("z"),
      py::arg("k_profile"),
      "Halo tables on the (z, M) grid: sigma2 is the mass variance "
      "sigma^2(M,z), dndm the halo abundance dn/dM, bias the linear "
      "halo bias, and concentration the Duffy et al. c(M,z), each with "
      "shape [z,M]; the fifth output is the normalized NFW Fourier "
      "profile u(k|M,z) with shape [z,M,k].");

    target.def("xi_from_power",
      [copying](
        py::handle ell, // angular-multipole nodes
        py::handle dell, // weights for integration over ell
        py::handle theta, // separations [row,node], in radians
        py::handle theta_weight, // normalized separation-bin weights [row,node]
        py::handle spectra_e, // E-mode angular power [row,ell]
        py::handle spectra_b, // B-mode angular power [row,ell]
        py::handle sign, // +1 for xi+, -1 for xi-
        const py::object& bins, // [row,2], needed with calibration
        const py::object& calibration // optional shear offsets
      ) {
        auto result = xi_binding(ell, dell, theta, theta_weight, spectra_e,
          spectra_b, sign, copying);
        if (!calibration.is_none()) {
          const auto sources = covariance_array<int>(bins, 2, true);
          const auto shear = covariance_array<double>(calibration, 1, true);
          if (sources.shape(0) != result.size() || sources.shape(1) != 2) {
            throw py::value_error("Xi calibration needs [row,2] source bins.");
          }
          for (int row=0; row<result.size(); row++) {
            const int first=sources.data()[2*row], second=sources.data()[2*row+1];
            if (first < 0 || second < 0 || first >= shear.size() || second >= shear.size()) {
              throw py::value_error("Calibration omits a xi source bin.");
            }
            result.mutable_data()[row] *= (1.0+shear.data()[first])*(1.0+shear.data()[second]);
          }
        }
        return result;
      }, py::arg("ell"), py::arg("dell"), py::arg("theta"),
      py::arg("theta_weight"), py::arg("spectra_e"), py::arg("spectra_b"),
      py::arg("sign"), py::arg("bins")=py::none(),
      py::arg("calibration")=py::none(),
      "Bin-averaged shear correlations from the E/B-mode angular power "
      "spectra C_ell^E and C_ell^B: sign +1 returns xi+, the J0 "
      "transform of C_ell^E + C_ell^B; sign -1 returns xi-, the J4 "
      "transform of C_ell^E - C_ell^B. J0 and J4 are Bessel functions "
      "of the first kind.");
  }
}

}
