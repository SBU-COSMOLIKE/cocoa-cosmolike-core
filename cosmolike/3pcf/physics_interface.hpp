#ifndef COSMOLIKE_PHYSICS_INTERFACE_HPP
#define COSMOLIKE_PHYSICS_INTERFACE_HPP

#include <pybind11/pybind11.h>
#include "physics_3pcf.h"
#include "physics_grid_3pcf.h"

namespace cosmolike_interface {

void bind_threepcf_physics(pybind11::module_& module);

// The covariance and transform interfaces call these readers to reach
// the C model and grid pointers that a Python-side owner object holds.
// The owner must stay alive until that consumer finishes, because the
// pointer borrows tables the owner will release. Reading the pointer
// changes no state and builds no second model. py::cast throws when the
// Python object has the wrong type.
const physics_model_3pcf* borrow_physics_model_3pcf(pybind11::handle owner);
const physics_grid_3pcf* borrow_physics_grid_3pcf(pybind11::handle owner);

}
#endif
