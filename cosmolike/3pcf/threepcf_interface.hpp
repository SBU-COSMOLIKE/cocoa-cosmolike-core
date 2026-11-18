#ifndef COSMOLIKE_THREEPCF_INTERFACE_HPP
#define COSMOLIKE_THREEPCF_INTERFACE_HPP

#include <pybind11/pybind11.h>

namespace cosmolike_interface {

// Register production NumPy and notebook Armadillo entry points over the
// same C kernels. The project module only calls this registration function.
void bind_threepcf(
    pybind11::module_& module // owning Python extension module
  );

}
#endif
