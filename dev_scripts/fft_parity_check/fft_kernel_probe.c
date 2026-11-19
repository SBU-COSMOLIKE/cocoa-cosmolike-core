// fft_kernel_probe.c: run the production scalar FAST-PT kernel twice.
//
// fft_parity_probe.c (same folder) checks the even-length SELECTION;
// this probe exercises the production code that CONSUMES the selected
// length. check_fft_parity.py compiles it together with the
// repository's own cfastpt/cfastpt.c (plus its utils files),
// cosmolike/basics.c, cosmolike/structs.c and log.c/src/log.c, so the
// J_abl that runs here is the working tree's J_abl, padding rules,
// length selection and all - nothing is replicated on the parity-sensitive
// path. The python driver asserts bitwise repeatability of the
// printed output across the two in-process calls, across fresh
// processes and across OpenMP thread counts, and compares the peak
// against the reference value of the repaired kernel: with an odd
// transform length, J_abl read an unwritten work-buffer entry and
// those comparisons failed (2026-11-18 defect report).
//
// Usage: fft_kernel_probe <size>
//   size = number of k points of the input spectrum. 1306 reproduces
//   the public accuracy setting (internal boost 1.187) whose padded
//   raw request 6056 formerly selected the odd length 6125; 1100 is
//   the unboosted default, whose selection was always even.
//
// Output, all on stdout:
//   sizes <size> raw <raw> selected <sel> ntotal <ntot>
//   call <i> finite <0|1> peak <max absolute value>
//   ... 3*size output values in C99 %a hexadecimal-float form, one
//   per line, exact to the last bit, for each of the two calls ...
//
// Exit codes: 0 after both calls, 1 on allocation failure, 2 on a
// usage or parse error.

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfastpt/cfastpt.h"
#include "cosmolike/basics.h"

// Number of FAST-PT terms the probe requests: three exponent triples
// drawn from the galaxy-bias calculation, the same configuration the
// defect report used to demonstrate the odd-length failure.
#define NTERMS 3

// ---------------------------------------------------------------------------
// Integer ceiling of numerator/denominator for positive operands,
// used for the padding counts below; (a + b - 1)/b rounds the exact
// quotient up because any nonzero remainder pushes the sum past the
// next multiple of b.
//
// Parameters:
//   numerator   = positive dividend
//   denominator = positive divisor
// Returns:
//   the smallest integer at or above numerator/denominator
// ---------------------------------------------------------------------------
static long ceil_div(const long numerator, const long denominator)
{
  return (numerator + denominator - 1) / denominator;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    fprintf(stderr, "usage: fft_kernel_probe <size>\n");
    return 2;
  }
  char* end = NULL;
  errno = 0;
  const long size = strtol(argv[1], &end, 10);
  if (end == argv[1] || *end != '\0' || errno == ERANGE || size < 4) {
    fprintf(stderr, "fft_kernel_probe: size must be an integer >= 4\n");
    return 2;
  }

  // The padding rules of the scalar bias tables: pad and
  // extrapolation counts scale with the point count so the padded
  // ln-k spans stay fixed (1500 and 500 points at the default 1100).
  // At size 1306 this gives N_pad 1781 and N_extrap 594 on each side,
  // the raw request 6056 of the defect report.
  fastpt_config config;
  config.nu = -2.0;
  config.c_window_width = 0.65;
  config.N_pad = ceil_div(1500 * size, 1100);
  config.N_extrap_low = ceil_div(500 * size, 1100);
  config.N_extrap_high = config.N_extrap_low;

  const long raw = size + 2 * config.N_pad
                        + config.N_extrap_low + config.N_extrap_high;
  // The length J_abl is about to use, printed for the driver: J_abl
  // keeps its selected length in function-local static variables with
  // no read-out function, so the probe asks the same
  // next_even_fft_size_radix7 the production call site asks. The
  // production call inside J_abl itself is guarded by the
  // repeatability and reference-peak assertions on this probe's
  // output.
  const long selected = next_even_fft_size_radix7(raw);
  // The convolution transform length J_abl derives from the selected
  // length; it has no evenness requirement of its own.
  const long ntotal = next_fft_size(2 * selected + 1);
  printf("sizes %ld raw %ld selected %ld ntotal %ld\n",
         size, raw, selected, ntotal);

  // A smooth positive spectrum on a log-spaced grid from 1e-4 to 10:
  // the same synthetic input the defect report used, so the repaired
  // kernel's reference peak applies.
  double* wavenumber = (double*) malloc(size * sizeof(double));
  double* power = (double*) malloc(size * sizeof(double));
  if (wavenumber == NULL || power == NULL) {
    fprintf(stderr, "fft_kernel_probe: input allocation failed\n");
    return 1;
  }
  const double log_low = log(1.0e-4);
  const double log_step = (log(10.0) - log_low) / (double)(size - 1);
  for (long i = 0; i < size; i++) {
    wavenumber[i] = exp(log_low + (double)i * log_step);
    const double ratio = wavenumber[i] / 0.15;
    const double suppression = 1.0 + ratio * ratio;
    power[i] = 1.0e4 * pow(wavenumber[i], 0.96)
                     / (suppression * suppression);
  }

  // Three galaxy-bias exponent triples; isP13type is unused by J_abl
  // and stays zero.
  int alpha[NTERMS] = {0, 0, 1};
  int beta[NTERMS] = {0, 0, -1};
  int ell[NTERMS] = {0, 2, 1};
  int is_p13[NTERMS] = {0, 0, 0};

  double* rows[NTERMS];
  for (int term = 0; term < NTERMS; term++) {
    rows[term] = (double*) malloc(size * sizeof(double));
    if (rows[term] == NULL) {
      fprintf(stderr, "fft_kernel_probe: output allocation failed\n");
      return 1;
    }
  }

  for (int call = 0; call < 2; call++) {
    // A fixed zero baseline before every call, so the printed output
    // reflects exactly what the kernel writes; the second call
    // exercises J_abl's cached plans and grids (the first call built
    // them).
    for (int term = 0; term < NTERMS; term++) {
      memset(rows[term], 0, size * sizeof(double));
    }
    J_abl(wavenumber, power, size, alpha, beta, ell, is_p13,
          NTERMS, &config, rows);
    int all_finite = 1;
    double peak = 0.0;
    for (int term = 0; term < NTERMS; term++) {
      for (long i = 0; i < size; i++) {
        const double value = rows[term][i];
        if (!isfinite(value)) {
          all_finite = 0;
        }
        if (fabs(value) > peak) {
          peak = fabs(value);
        }
      }
    }
    printf("call %d finite %d peak %.17g\n", call, all_finite, peak);
    // %a prints the exact bits of each double (hexadecimal floating
    // point), so the driver's string comparison of these lines is a
    // bitwise comparison of the kernel output.
    for (int term = 0; term < NTERMS; term++) {
      for (long i = 0; i < size; i++) {
        printf("%a\n", rows[term][i]);
      }
    }
  }
  return 0;
}
