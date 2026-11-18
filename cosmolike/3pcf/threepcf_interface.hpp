#ifndef COSMOLIKE_THREEPCF_INTERFACE_HPP
#define COSMOLIKE_THREEPCF_INTERFACE_HPP

#include <pybind11/pybind11.h>

namespace cosmolike_interface {

// bind_threepcf registers two Python submodules whose entry points run
// the same C kernels: production borrows NumPy arrays without copying
// them, while notebook copies every input through Armadillo first. The
// project extension calls this function once, during module import.
void bind_threepcf(
    pybind11::module_& module // owning Python extension module
  );

}
#endif
