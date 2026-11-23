#ifndef COSMOLIKE_COVARIANCE_INTERFACE_3PCF_HPP
#define COSMOLIKE_COVARIANCE_INTERFACE_3PCF_HPP

#include <pybind11/pybind11.h>

namespace cosmolike_interface {

// Registers the covariance bindings (covariance_interface.cpp): the
// input builders shape_noise and covariance_inputs on the module
// itself, and physical_covariance and halo_tables on its production
// and notebook submodules.
void bind_threepcf_covariance(
  pybind11::module_& module // Python module receiving the bindings
);

}
#endif
