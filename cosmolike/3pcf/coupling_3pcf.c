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
#include "log.c/src/log.h"

// Angular-coupling convention: fastnc by Sunao Sugiyama, MIT, revision
// 7b2336ce37a909bed51f07f32a2a4f096e71b6ce, fastnc/coupling.py.
// The accompanying LICENSE.fastnc-TreeCorr.txt preserves its notices.
// The fastnc method paper (Sugiyama, Gomes and Jarvis, arXiv:2407.01798)
// introduces this coupling function for the multipole evaluation of the
// shear three-point correlation. fastnc integrates the opening angle
// numerically for every table entry; this file evaluates the same
// integral through a finite Fourier sum.
//
// PHYSICAL DERIVATION & LOGIC FLOW
//
// The target is the angular coupling function of the multipole
// transform,
//
//   G_ML(psi) = 4*pi integral_0^pi P_L(cos delta)
//                        cos(2 beta_bar + M delta) d delta,
//
// where L is the Legendre degree that pairs with the bispectrum
// multipoles b_L, M is the nonnegative Fourier order of the real-space
// natural shear components, psi is the opening angle arctan(ell2/ell1)
// in [0, pi/2] - an angle that encodes the ratio of the two multipole
// legs, not itself a ratio - and delta, the integration variable, is
// the angle between the two legs.
//
// beta_bar is the triangle's shear-rotation phase. Write t = tan(psi)
// and z = exp(i delta). Drawing leg one along the x axis and leg two at
// angle delta gives the resultant ell1 + ell2*z; swapping the two leg
// lengths gives ell2 + ell1*z. Both vectors have the same length, so
// their ratio is a pure phase, and that phase defines beta_bar:
//
//   exp(2i beta_bar) = (ell2 + ell1*z)/(ell1 + ell2*z)
//                    = (t + z)/(1 + t*z).
//
// Geometrically beta_bar is the angle between the resultant of the two
// legs and the bisector of the legs; it vanishes for equal legs (t=1),
// where the ratio above is exactly one. The pinned fastnc source
// computes the same normalized sine/cosine pair in utils.sincos2angbar
// and calls it only "the triangle phase factor"; the geometric reading
// here follows from the formula itself.
//
// The finite sum replaces the numerical integral in three steps.
//
//   1. Expand the phase in powers of z. For t < 1,
//        (t + z)/(1 + t*z) = t + (1 - t*t) sum_n>=1 (-t)^(n-1) z^n.
//      For t > 1 the same series holds with q = 1/t in place of t and
//      z^-n in place of z^n.
//   2. Expand the Legendre polynomial as a finite Fourier series,
//      P_L(cos delta) = sum_|n|<=L c_Ln z^n, with real coefficients
//      that satisfy c_L,-n = c_Ln.
//   3. Multiply the two series by z^M and keep the constant Fourier
//      coefficient: the integrand is even in delta, so the real
//      integral on [0,pi] equals pi times that coefficient. P_L
//      contributes only |n| <= L, so finitely many terms survive even
//      though the phase alone has infinitely many harmonics. At t = 1
//      the phase is one away from the single endpoint delta = pi, and
//      G_ML = 4*pi*pi*c_LM.
//
// Cache invalidation:
// None here. The transform plan stores the resulting geometry table,
// and this file releases every temporary allocation before returning.



// ---------------------------------------------------------------------------
// Read one Fourier coefficient c_Ln of the Legendre polynomial
// P_L(cos delta) = sum_n c_Ln exp(i n delta). The coefficients are real
// and even in the harmonic, c_L,-n = c_Ln, so the table stores only
// n >= 0 and this reader folds a negative request onto its mirror. A
// harmonic beyond the degree carries no power in P_L and reads as zero,
// which lets the series code request any index without bounds checks.
//
// Parameters:
//   coefficients - c_Ln for n = 0..degree at one fixed degree L
//   degree - the Legendre degree L, the largest nonzero harmonic
//   harmonic - the requested harmonic n, of either sign
//
// Returns:
//   c_L|n|, or zero when |n| exceeds the degree.
// ---------------------------------------------------------------------------
static double coefficient_3pcf(
  const double* coefficients, // c_Ln for n = 0..degree, one fixed L
  const int degree, // Legendre degree L, the largest nonzero harmonic
  const int harmonic // requested harmonic n, of either sign
)
{
  const int absolute = abs(harmonic);
  if (absolute > degree) {
    return 0.0;
  }
  return coefficients[absolute];
}


// ---------------------------------------------------------------------------
// Evaluate G_ML(psi) for one Legendre degree and Fourier order through
// the finite sum of the file head. Both series branches share one code
// path. Below the fold psi = pi/4 the expansion variable is
// t = tan(psi), and the series term n couples the coefficient c_L,M+n,
// so harmonics stop at n = L - M. Above the fold the variable is
// 1/t = tan(pi/2 - psi), the series runs over z^-n, a term couples
// c_L,M-n, and harmonics stop at n = L + M. In both branches the n = 0
// term contributes t*c_LM, and the n >= 1 terms carry the common
// amplitude (1 - t*t) and the alternating power (-t)^(n-1).
//
// Parameters:
//   coefficients - Fourier row c_Ln of P_L at this degree
//   degree - Legendre degree L
//   mode - nonnegative Fourier order M
//   psi - the opening angle arctan(ell2/ell1), in [0, pi/2]
//
// Returns:
//   G_ML(psi), i.e. 4*pi*pi times the constant Fourier coefficient
//   assembled above.
// ---------------------------------------------------------------------------
static double coupling_value_3pcf(
  const double* coefficients, // Fourier row c_Ln of P_L at this degree
  const int degree, // Legendre degree L
  const int mode, // nonnegative Fourier order M
  const double psi // opening angle arctan(ell2/ell1), in [0,pi/2]
)
{
  const double pi = acos(-1.0);
  const int upper = psi > pi/4.0;
  const double ratio = tan(upper ? pi/2.0-psi : psi);
  double sum = ratio*coefficient_3pcf(coefficients, degree, mode);
  const double amplitude = 1.0-ratio*ratio;

  // One loop term adds amplitude*(-ratio)^(n-1) times the coupled
  // coefficient: c_L,M+n below the fold, c_L,M-n above it, with the
  // reader folding a negative index. The bound last stops the loop
  // where every further coefficient is zero. Multiplying power by
  // -ratio each round advances the alternating sign exactly.
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


// ---------------------------------------------------------------------------
// Tabulate G_ML(psi) for every Fourier order M = 0..mmax and Legendre
// degree L = 0..lmax at nquery opening angles. The routine first builds
// the Fourier coefficients of every Legendre polynomial once, then
// evaluates the finite sum of the file head - exactly at each query
// when npsi is zero, or through linear interpolation of a uniform
// [0,pi/2] table with npsi nodes, which reproduces fastnc's tabulation
// convention.
//
// Parameters:
//   lmax - largest Legendre degree L
//   mmax - largest nonnegative Fourier order M
//   nquery - number of opening-angle queries
//   psi - [nquery], opening angles arctan(ell2/ell1) in [0, pi/2]
//   npsi - zero for exact evaluation, or the uniform table length
//   output - [mmax+1,lmax+1,nquery] in C order, written completely
//
// Returns:
//   Nothing; invalid input aborts with a message instead.
// ---------------------------------------------------------------------------
void legendre_coupling_3pcf(
  const int lmax, // largest Legendre degree L
  const int mmax, // largest nonnegative Fourier order M
  const int nquery, // number of opening-angle queries
  const double* psi, // [nquery], opening angles in radians
  const int npsi, // zero for exact values, or uniform table length
  double* output // [mmax+1,lmax+1,nquery], C order
)
{
  if (lmax < 0 || mmax < 0 || lmax > INT_MAX-1-mmax
      || nquery < 1 || npsi == 1 || npsi < 0
      || psi == NULL || output == NULL) {
    log_fatal("legendre_coupling_3pcf: invalid input");
    exit(1);
  }

  const double pi = acos(-1.0);
  for (int query=0; query<nquery; query++) {
    if (!isfinite(psi[query]) || psi[query] < 0.0 || psi[query] > pi/2.0) {
      log_fatal("legendre_coupling_3pcf: psi outside [0,pi/2]");
      exit(1);
    }
  }

  const size_t stride = (size_t)lmax+1;
  if (stride > SIZE_MAX/stride/sizeof(double)) {
    log_fatal("legendre_coupling_3pcf: size overflow");
    exit(1);
  }

  double* coefficients = calloc(stride*stride, sizeof(double));
  if (coefficients == NULL) {
    log_fatal("legendre_coupling_3pcf: allocation failed");
    exit(1);
  }

  coefficients[0] = 1.0;

  // Build the Fourier rows of every Legendre polynomial from the
  // three-term recurrence
  //   L P_L = (2L-1) cos(delta) P_{L-1} - (L-1) P_{L-2}.
  // In Fourier space cos(delta) = (z + 1/z)/2, so multiplying by it
  // shifts a coefficient one harmonic down and one up, each with a
  // factor 1/2; the neighbors sum below collects both shifts, and the
  // folded reader supplies the mirrored negative harmonics. Each
  // completed degree feeds the next one, so this short stage stays
  // serial.
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

  // Fill the output table. One iteration owns one (M,L) row: it reads
  // the prepared Fourier row of P_L and writes G_ML at every query
  // angle, so no two workers ever touch the same output element. Every
  // value is pure triangle geometry, with no dependence on the physical
  // bispectrum, which is why the transform can reuse this table across
  // models. The if clause skips the OpenMP team when lmax and mmax are
  // both zero: the collapsed loop then has a single row and a team
  // would only add startup overhead.
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
          // Reproduce fastnc's tabulation: npsi uniform nodes span
          // [0,pi/2], step is their spacing, and coordinate counts how
          // many steps the query sits above zero.
          const double step = pi/(2.0*(npsi-1));
          const double coordinate = psi[query]/step;
          int lower = (int)floor(coordinate);

          // A query at the upper endpoint psi = pi/2 lands on the last
          // node, where the right neighbor would read past the table;
          // clamping the left node to npsi-2 turns that endpoint into
          // an exact read of the final segment.
          if (lower >= npsi-1) {
            lower = npsi-2;
          }
          const double fraction = coordinate-lower;

          // Linear interpolation between the two bracketing nodes, each
          // evaluated by the same finite sum as the exact branch.
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
