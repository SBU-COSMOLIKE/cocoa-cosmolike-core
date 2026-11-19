#ifndef THREEPCF_TRACE_FFTW_HPP
#define THREEPCF_TRACE_FFTW_HPP
#include <pybind11/pybind11.h>

// Only the private validation extension registers these counters. The
// installed library has no test bindings or lifecycle instrumentation.
void bind_fftw_trace(
    pybind11::module_& module // private validation extension
  );
#endif
