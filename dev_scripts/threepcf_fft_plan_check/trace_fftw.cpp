#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <fftw3.h>
#include "trace_fftw.hpp"

// This separate validation translation unit sees the original FFTW names.
// Only fftlog_3pcf.c is compiled with the trace_* name substitutions. The
// counters therefore describe the three-point workspace, without counting
// unrelated plans or changing production code. Preparation is serialized.
// During OpenMP execution the recipe map is read-only; no shared counter
// or lock is added to an FFT execution.
namespace {
struct recipe {
  int rows; // first padded FFT dimension
  int columns; // second padded FFT dimension
  int input_alignment; // alignment class of the planning input
  int output_alignment; // alignment class of the planning output
  bool forward; // true for real-to-complex, false for complex-to-real
};
std::mutex trace_mutex;
std::unordered_map<fftw_plan, recipe> recipes;
size_t created = 0;
size_t destroyed = 0;

void require(
    bool condition, // property required before calling FFTW
    const char* message // diagnostic naming the violated property
  )
{
  if (!condition) {
    std::fprintf(stderr, "FFTW trace: %s\n", message);
    std::abort();
  }
}

recipe inspect(
    fftw_plan plan, // live recipe registered during serial preparation
    double* input, // current execution input address
    double* output, // current execution output address
    bool forward // expected transform direction
  )
{
  const auto found = recipes.find(plan);
  require(found != recipes.end(), "execution uses an unknown plan");
  const recipe value = found->second;
  require(value.forward == forward, "plan direction differs");
  require(fftw_alignment_of(input) == value.input_alignment,
      "new-array input alignment differs");
  require(fftw_alignment_of(output) == value.output_alignment,
      "new-array output alignment differs");
  return value;
}
}

extern "C" {
void* trace_malloc(
    size_t count // requested bytes, including rounded worker strides
  )
{
  void* storage = fftw_malloc(count);
  if (storage != nullptr) {
    // Every work buffer starts poisoned. Execution checks the entire FFT
    // input, including padding, before FFTW can consume uninitialized cells.
    auto* values = static_cast<double*>(storage);
    std::fill_n(values, count/sizeof(double),
        std::numeric_limits<double>::quiet_NaN());
  }
  return storage;
}

fftw_plan trace_plan_r2c(
    int rows, // first padded dimension
    int columns, // second padded dimension
    double* input, // planning real buffer
    fftw_complex* output, // planning half-spectrum buffer
    unsigned flags // must remain FFTW_ESTIMATE for this ownership change
  )
{
  require(flags == FFTW_ESTIMATE, "planner default changed");
  fftw_plan plan = fftw_plan_dft_r2c_2d(rows, columns, input, output, flags);
  require(plan != nullptr, "forward plan allocation failed");
  std::lock_guard<std::mutex> lock(trace_mutex);
  recipes.emplace(plan, recipe{rows, columns, fftw_alignment_of(input),
      fftw_alignment_of(reinterpret_cast<double*>(output)), true});
  created++;
  return plan;
}

fftw_plan trace_plan_c2r(
    int rows, // first padded dimension
    int columns, // second padded dimension
    fftw_complex* input, // planning half-spectrum buffer
    double* output, // planning real buffer
    unsigned flags // same planner policy as the forward recipe
  )
{
  require(flags == FFTW_ESTIMATE, "planner default changed");
  fftw_plan plan = fftw_plan_dft_c2r_2d(rows, columns, input, output, flags);
  require(plan != nullptr, "inverse plan allocation failed");
  std::lock_guard<std::mutex> lock(trace_mutex);
  recipes.emplace(plan, recipe{rows, columns,
      fftw_alignment_of(reinterpret_cast<double*>(input)),
      fftw_alignment_of(output), false});
  created++;
  return plan;
}

void trace_destroy(
    fftw_plan plan // recipe whose sole workspace owner is being released
  )
{
  {
    std::lock_guard<std::mutex> lock(trace_mutex);
    require(recipes.erase(plan) == 1, "unknown or double destruction");
    destroyed++;
  }
  fftw_destroy_plan(plan);
}

void trace_execute_r2c(
    fftw_plan plan, // retained real-to-complex recipe
    double* input, // one worker's full padded input
    fftw_complex* output // that worker's forward coefficients
  )
{
  const recipe value = inspect(plan, input,
      reinterpret_cast<double*>(output), true);
  const size_t size = static_cast<size_t>(value.rows)*value.columns;
  for (size_t index=0; index<size; index++) {
    require(std::isfinite(input[index]), "nonfinite forward FFT input");
  }
  fftw_execute_dft_r2c(plan, input, output);
}

void trace_execute_c2r(
    fftw_plan plan, // retained complex-to-real recipe
    fftw_complex* input, // that worker's disposable inverse coefficients
    double* output // that worker's real FFT scratch
  )
{
  const recipe value = inspect(plan, reinterpret_cast<double*>(input),
      output, false);
  const size_t size = static_cast<size_t>(value.rows)*(value.columns/2+1);
  for (size_t index=0; index<size; index++) {
    require(std::isfinite(input[index][0]), "nonfinite inverse real part");
    require(std::isfinite(input[index][1]), "nonfinite inverse imag part");
  }
  fftw_execute_dft_c2r(plan, input, output);
}
}

void bind_fftw_trace(
    pybind11::module_& module // private validation module, never installed
  )
{
  module.def("_fftw_trace_counts", []() {
    std::lock_guard<std::mutex> lock(trace_mutex);
    pybind11::dict values;
    values["created"] = created;
    values["destroyed"] = destroyed;
    values["live"] = recipes.size();
    return values;
  });
}
