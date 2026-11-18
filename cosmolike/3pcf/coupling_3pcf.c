#include <math.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "coupling_3pcf.h"

// Angular-coupling convention: fastnc by Sunao Sugiyama, MIT, revision
// 7b2336ce37a909bed51f07f32a2a4f096e71b6ce, fastnc/coupling.py.
// The accompanying LICENSE.fastnc-TreeCorr.txt preserves its notices.
// This implementation evaluates the same integral by a finite Fourier
// sum instead of repeatedly integrating the opening angle numerically.
//
// PHYSICAL DERIVATION & LOGIC FLOW
// Let z=exp(i delta), t=tan(psi), and c_Ln be the complex Fourier
// coefficients of P_L(cos delta), with c_L,-n=c_Ln. Triangle geometry gives
//   exp(2i beta_bar)=(t+z)/(1+t*z).
// For t<1 its series is t+(1-t*t) sum_n>=1 (-t)^(n-1) z^n.
// For t>1 use q=1/t and replace z^n by z^-n in the same series.
// Multiplication by exp(i M delta) and P_L selects its constant Fourier
// coefficient. The real integral on [0,pi] is pi times that coefficient.
// Only |n|<=L occur in P_L, so the required sum is finite even when the
// rational phase has infinitely many harmonics. At t=1 the phase is one
// except at a measure-zero endpoint, and G_ML=4*pi*pi*c_LM.
//
// Cache invalidation:
// None here. The transform plan stores the resulting geometry table.
// Every temporary allocation is released before this function returns.

static double coefficient_3pcf(
  const double* coefficients, // c_Ln for nonnegative n, fixed L
  const int degree, // largest harmonic with a nonzero coefficient
  const int harmonic // requested signed harmonic
)
{
  const int absolute = abs(harmonic);
  if (absolute > degree) {
    return 0.0;
  }
  return coefficients[absolute];
}


static double coupling_value_3pcf(
  const double* coefficients, // fixed Legendre row c_Ln
  const int degree, // Legendre degree L
  const int mode, // nonnegative Fourier order M
  const double psi // angular ratio in [0,pi/2]
)
{
  const double pi = acos(-1.0);
  const int upper = psi > pi/4.0;
  const double ratio = tan(upper ? pi/2.0-psi : psi);
  double sum = ratio*coefficient_3pcf(coefficients, degree, mode);
  const double amplitude = 1.0-ratio*ratio;

  // Below pi/4 only harmonics M+n<=L contribute. Above it the surviving
  // harmonics satisfy |M-n|<=L. Advancing powers retains their sign.
  const int last = upper ? mode+degree : degree-mode;
  double power = 1.0;
  for (int harmonic=1; harmonic<=last; harmonic++) {
    const int selected = upper ? mode-harmonic : mode+harmonic;
    sum += amplitude*power
        *coefficient_3pcf(coefficients, degree, selected);
    power *= -ratio;
  }
  return 4.0*pi*pi*sum;
}


void legendre_coupling_3pcf(
  const int lmax, // largest Legendre degree
  const int mmax, // largest Fourier order
  const int nquery, // number of psi values
  const double* psi, // [nquery], radians
  const int npsi, // optional uniform interpolation grid length
  double* output // [M,L,query], C order
)
{
  if (lmax < 0 || mmax < 0 || lmax > INT_MAX-1-mmax
      || nquery < 1 || npsi == 1 || npsi < 0
      || psi == NULL || output == NULL) {
    fprintf(stderr, "legendre_coupling_3pcf: invalid input\n");
    exit(EXIT_FAILURE);
  }
  const double pi = acos(-1.0);
  for (int query=0; query<nquery; query++) {
    if (!isfinite(psi[query]) || psi[query] < 0.0 || psi[query] > pi/2.0) {
      fprintf(stderr, "legendre_coupling_3pcf: psi outside [0,pi/2]\n");
      exit(EXIT_FAILURE);
    }
  }
  const size_t stride = (size_t)lmax+1;
  if (stride > SIZE_MAX/stride/sizeof(double)) {
    fprintf(stderr, "legendre_coupling_3pcf: size overflow\n");
    exit(EXIT_FAILURE);
  }
  double* coefficients = calloc(stride*stride, sizeof(double));
  if (coefficients == NULL) {
    fprintf(stderr, "legendre_coupling_3pcf: allocation failed\n");
    exit(EXIT_FAILURE);
  }
  coefficients[0] = 1.0;

  // The Legendre recurrence multiplies by cos(delta), which shifts a
  // Fourier coefficient one place left and right with a factor 1/2.
  // Each completed degree is used by the next; this short stage is serial.
  for (int degree=1; degree<=lmax; degree++) {
    const double* previous = coefficients+(size_t)(degree-1)*stride;
    double* current = coefficients+(size_t)degree*stride;
    for (int harmonic=0; harmonic<=degree; harmonic++) {
      const double neighbors =
          coefficient_3pcf(previous, degree-1, harmonic-1)
          +coefficient_3pcf(previous, degree-1, harmonic+1);
      current[harmonic] = (2.0*degree-1.0)/(2.0*degree)*neighbors;
      if (degree >= 2) {
        const double* older = coefficients+(size_t)(degree-2)*stride;
        current[harmonic] -= (degree-1.0)/degree
            *coefficient_3pcf(older, degree-2, harmonic);
      }
    }
  }

  // Each worker owns an entire (M,L) output row. Tabulation brackets are
  // geometric and do not depend on the physical bispectrum.
  #pragma omp parallel for collapse(2) if(lmax > 0 || mmax > 0) \
      schedule(static)
  for (int mode=0; mode<=mmax; mode++) {
    for (int degree=0; degree<=lmax; degree++) {
      const double* row = coefficients+(size_t)degree*stride;
      double* result = output+((size_t)mode*stride+degree)*nquery;
      for (int query=0; query<nquery; query++) {
        if (npsi == 0) {
          result[query] = coupling_value_3pcf(row, degree, mode, psi[query]);
        } else {
          const double step = pi/(2.0*(npsi-1));
          const double coordinate = psi[query]/step;
          int lower = (int)floor(coordinate);
          if (lower >= npsi-1) {
            lower = npsi-2;
          }
          const double fraction = coordinate-lower;
          const double left = coupling_value_3pcf(
              row, degree, mode, lower*step);
          const double right = coupling_value_3pcf(
              row, degree, mode, (lower+1)*step);
          result[query] = left+fraction*(right-left);
        }
      }
    }
  }
  free(coefficients);
}
