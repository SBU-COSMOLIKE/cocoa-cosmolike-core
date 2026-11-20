#ifndef COSMOLIKE_THREEPCF_INTERFACE_HPP
#define COSMOLIKE_THREEPCF_INTERFACE_HPP

#include <pybind11/pybind11.h>

namespace cosmolike_interface {

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

void bind_threepcf(
    pybind11::module_& module // owning Python extension module
  );

}
#endif
