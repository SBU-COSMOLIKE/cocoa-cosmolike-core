#ifndef COSMOLIKE_COVARIANCE_INTERFACE_3PCF_HPP
#define COSMOLIKE_COVARIANCE_INTERFACE_3PCF_HPP

#include <pybind11/pybind11.h>

namespace cosmolike_interface {

// Registers physical_covariance and halo_tables on the
// module's production and notebook submodules (covariance_interface.cpp).
void bind_threepcf_covariance(
  pybind11::module_& module // Python module receiving the bindings
);

}
#endif
