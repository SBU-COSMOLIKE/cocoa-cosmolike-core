#ifndef COSMOLIKE_TRANSFORM_INTERFACE_HPP
#define COSMOLIKE_TRANSFORM_INTERFACE_HPP
#include <pybind11/pybind11.h>

namespace cosmolike_interface {
void bind_threepcf_transform(
    pybind11::module_& module // owning Python extension
  );
}
#endif
