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
        auto array = input_array_3pcf<double>(item.second, rank, notebook);
        covariance_finite(array);
        real.emplace(name, std::move(array));
      } else if (integer_vectors.count(name) || name == "row_fields") {
        const int rank = name == "row_fields" ? 2 : 1;
        auto array = input_array_3pcf<int>(item.second, rank, notebook);
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
  input_count_3pcf({6, count, count});

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

  const auto z = input_array_3pcf<double>(redshifts, 1, notebook);
  const auto k = input_array_3pcf<double>(wavenumbers, 1, notebook);
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
  input_count_3pcf({z.size(), mass_count, k.size()});

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


// ============================================================================
// [SECTION] MODULE REGISTRATION
// ============================================================================

// Diagonal shape-noise power from the source densities and dispersions.
static py::array_t<double> shape_noise_3pcf(
  const py::array_t<double>& density, // effective sources per square arcminute
  const py::array_t<double>& dispersion // rms of ONE ellipticity component
)
{
  const auto n = input_array_3pcf<double>(density, 1, true);
  const auto sigma = input_array_3pcf<double>(dispersion, 1, true);
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
  const auto radii = input_array_3pcf<double>(angles, 2, true);
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
  const auto profile = input_array_3pcf<double>(grids["profile_radius"], 1, true);
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
void bind_threepcf_covariance(
  py::module_& module // Python module receiving the bindings
)
{
  module.def("shape_noise", &shape_noise_3pcf,
    py::arg("number_density"), py::arg("sigma_component"));
  module.def("covariance_inputs", &covariance_inputs_3pcf,
    py::arg("kind"), py::arg("bins"), py::arg("angles"),
    py::arg("noise"), py::arg("grids"), py::arg("physical"),
    py::arg("theta_count"), py::arg("pair_weight"),
    py::arg("accuracy")=py::none());
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


  }
}

}
