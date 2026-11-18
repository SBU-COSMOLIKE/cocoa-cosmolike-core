#ifndef COSMOLIKE_PHYSICS_INTERFACE_HPP
#define COSMOLIKE_PHYSICS_INTERFACE_HPP

#include <pybind11/pybind11.h>
#include "physics_3pcf.h"
#include "physics_grid_3pcf.h"

namespace cosmolike_interface {

void bind_threepcf_physics(pybind11::module_& module);

// The Python argument must remain alive until the consumer finishes.
// Reading this pointer changes no state and needs no second model.
const physics_model_3pcf* borrow_physics_model_3pcf(pybind11::handle owner);
const physics_grid_3pcf* borrow_physics_grid_3pcf(pybind11::handle owner);

}
#endif
