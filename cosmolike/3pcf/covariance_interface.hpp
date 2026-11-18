#ifndef COSMOLIKE_COVARIANCE_INTERFACE_3PCF_HPP
#define COSMOLIKE_COVARIANCE_INTERFACE_3PCF_HPP
#include <pybind11/pybind11.h>
namespace cosmolike_interface {
void bind_threepcf_covariance(pybind11::module_& module);
}
#endif
