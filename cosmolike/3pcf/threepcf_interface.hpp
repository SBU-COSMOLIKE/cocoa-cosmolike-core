#ifndef COSMOLIKE_THREEPCF_INTERFACE_HPP
#define COSMOLIKE_THREEPCF_INTERFACE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <spdlog/spdlog.h>

namespace cosmolike_interface {

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
inline size_t input_count_3pcf(
  // axis lengths of the requested C-order storage
  const std::vector<pybind11::ssize_t>& shape
)
{
  size_t product = 1;

  for (const auto length : shape) {
    if (!(0 < length && INT_MAX >= length)) {
      spdlog::critical("3pcf interface: invalid array dimension");
      std::exit(1);
    }
    if (!(static_cast<size_t>(length) <= SIZE_MAX/product)) {
      spdlog::critical("3pcf interface: array dimension product "
        "overflows");
      std::exit(1);
    }
    product *= static_cast<size_t>(length);
  }

  if (!(product <= static_cast<size_t>(PY_SSIZE_T_MAX)
        /sizeof(double))) {
    spdlog::critical("3pcf interface: array byte count overflows");
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
pybind11::array_t<Number, pybind11::array::c_style> input_array_3pcf(
  const pybind11::handle object, // Python buffer to validate or copy
  const int rank, // required number of array axes
  // true copies notebook inputs; false borrows production arrays
  const bool notebook
)
{
  if (!(pybind11::isinstance<pybind11::array>(object))) {
    spdlog::critical("3pcf interface: numerical inputs must be "
      "NumPy arrays");
    std::exit(1);
  }
  const pybind11::array input = pybind11::reinterpret_borrow<pybind11::array>(object);
  if (!(input.dtype().is(pybind11::dtype::of<Number>()))) {
    spdlog::critical("3pcf interface: array has the wrong dtype; "
      "require float64 or int32 as documented");
    std::exit(1);
  }
  if (!(input.ndim() == rank)) {
    spdlog::critical("3pcf interface: array has the wrong rank");
    std::exit(1);
  }

  std::vector<pybind11::ssize_t> shape;
  for (int axis=0; axis<rank; axis++) {
    shape.push_back(input.shape(axis));
  }
  const size_t count = input_count_3pcf(shape);

  if (!notebook) {
    if (!(0 != (input.flags() & pybind11::array::c_style))) {
      spdlog::critical("3pcf interface: production requires "
        "contiguous C-order inputs");
      std::exit(1);
    }
    const auto address = reinterpret_cast<std::uintptr_t>(input.data());
    if (!(0 == address%alignof(Number))) {
      spdlog::critical("3pcf interface: production input is not "
        "aligned for its C element type");
      std::exit(1);
    }
    return pybind11::reinterpret_borrow<pybind11::array_t<Number, pybind11::array::c_style>>(
      input);
  }

  pybind11::array_t<Number, pybind11::array::c_style> result(shape);
  const char* source = static_cast<const char*>(input.data());
  Number* destination = result.mutable_data();

  for (size_t index=0; index<count; index++) {
    size_t remainder = index;
    pybind11::ssize_t offset = 0;
    for (int axis=rank - 1; 0 <= axis; axis--) {
      const size_t coordinate = remainder%static_cast<size_t>(shape[axis]);
      remainder /= static_cast<size_t>(shape[axis]);
      offset += static_cast<pybind11::ssize_t>(coordinate)*input.strides(axis);
    }
    // A copy through bytes also accepts a valid but unaligned notebook
    // view: no typed dereference happens until the owned buffer exists.
    std::memcpy(destination + index, source + offset, sizeof(Number));
  }

  return result;
}




// bind_threepcf registers two Python submodules whose entry points run
// the same C kernels: production borrows NumPy arrays without copying
// them, while notebook copies every input through Armadillo first. The
// project extension calls this function once, during module import.
// Clear the joint measurement when the project resets native state.
void reset_threepcf_joint_data();

// Native preparation stages share the same refinement arithmetic.
pybind11::dict init_accuracy_3pcf(
  const pybind11::object& value // None or explicit global/stage multipliers
);
int accuracy_count_3pcf(
  const pybind11::dict& accuracy, // validated refinement multipliers
  double baseline, // unboosted count; never reuse a previously refined count
  const std::string& stage, // individual stage name
  bool integration, // select integration rather than sampling global factor
  bool endpoints // refine N-1 intervals while retaining both endpoints
);
pybind11::dict accuracy_projection_3pcf(
  const pybind11::dict& accuracy, // global and individual multipliers
  const pybind11::dict& baseline // unboosted projection-grid controls
);
pybind11::dict accuracy_quadrature_3pcf(
  const pybind11::dict& accuracy, // global and individual multipliers
  const pybind11::dict& baseline, // bounds and unboosted quadrature count
  const std::string& stage // individual quadrature name
);

// Mean and covariance share these fixed integration rules. Their
// bindings remain available when covariance generation is disabled.
pybind11::tuple quadrature_3pcf(
  double lower, // integration lower endpoint
  double upper, // integration upper endpoint
  int count, // quadrature order
  bool logarithmic // integrate uniformly in ln(x)
);
pybind11::tuple xi_bin_weights_3pcf(
  const std::vector<std::string>& kind, // observable name per row
  const pybind11::array_t<double>& angles, // [row,3], angles in radians
  int count, // separation quadrature order
  const pybind11::object& pair_weight // uniform_area or explicit weights
);

void bind_threepcf(
    pybind11::module_& module // owning Python extension module
  );

}
#endif
