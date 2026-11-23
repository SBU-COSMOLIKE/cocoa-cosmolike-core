#include <stdlib.h>
#include "cosmolike/basics.h"
#include "physics_reference_3pcf.h"

// Upstream-comparison interpolation for the physical model. The production
// preparation uses uniform segments and the house natural-cubic solver.
// These routines retain fastnc's not-a-knot boundary condition and searches
// on distance coordinates solely when source_readers is explicitly selected.
// They own no static state and require no mutable GSL accelerator.

// reference_spline_coeffs_3pcf: build a not-a-knot cubic spline on an
// irregular grid.
//
// A cubic spline stores the second derivative at every node; between
// nodes the interpolant is the cubic polynomial those endpoint values
// determine. Continuity of the first derivative at each interior node
// gives one equation per interior node,
//
//   h_left M[i-1] + 2 (h_left + h_right) M[i] + h_right M[i+1]
//     = 6 (slope_right - slope_left),
//
// with M the second derivatives, h the interval widths and slope the
// finite-difference slope of each interval. Two boundary conditions
// close the system. The not-a-knot choice of scipy's
// InterpolatedUnivariateSpline makes the first two intervals share
// one cubic polynomial, and likewise the last two; eliminating M[0]
// and M[n-1] folds that condition into the first and last interior
// rows and leaves a tridiagonal system. One forward elimination and
// one back substitution (the Thomas algorithm) solve it here, during
// preparation; evaluation then only reads and needs no mutable GSL
// accelerator object.
void reference_spline_coeffs_3pcf(
  const double* x, // validated increasing source coordinates [count]
  const double* y, // validated finite source samples [count]
  const int count, // at least two nodes
  double* curvature // caller-owned zero-initialized second derivatives
)
{
  if (count == 2) {
    return;
  }
  if (count == 3) {
    const double slope0 = (y[1] - y[0])/(x[1] - x[0]);
    const double slope1 = (y[2] - y[1])/(x[2] - x[1]);
    const double second = 2.0*(slope1 - slope0)/(x[2] - x[0]);
    for (int index=0; index<count; index++) {
      curvature[index] = second;
    }
    return;
  }

  const int interior = count - 2;
  double* diagonal = calloc1d(interior);
  double* upper = calloc1d(interior);
  double* rhs = calloc1d(interior);

  // Assemble the interior rows and eliminate on the fly: each row
  // folds its subdiagonal entry into the previous pivot, so one pass
  // both builds and forward-eliminates the tridiagonal system.
  for (int row=0; row<interior; row++) {
    const int index = row + 1;
    const double left = x[index] - x[index - 1];
    const double right = x[index + 1] - x[index];
    double lower = left;
    diagonal[row] = 2.0*(left + right);
    upper[row] = right;
    rhs[row] = 6.0*((y[index + 1] - y[index])/right
      - (y[index] - y[index - 1])/left);
    if (row == 0) {
      // First interior row: the not-a-knot condition expresses M[0]
      // through M[1] and M[2]; substituting it modifies this row's
      // diagonal and upper entries and removes its subdiagonal.
      diagonal[row] += left*(left + right)/right;
      upper[row] -= left*left/right;
      lower = 0.0;
    }
    if (row == interior - 1) {
      // Last interior row: the mirrored not-a-knot condition folds
      // M[n-1] into the two previous interior values the same way.
      diagonal[row] += right*(left + right)/left;
      lower -= right*right/left;
      upper[row] = 0.0;
    }
    if (row > 0) {
      // Forward elimination: subtract the multiple of the previous
      // row that zeroes this row's subdiagonal entry.
      const double multiplier = lower/diagonal[row - 1];
      diagonal[row] -= multiplier*upper[row - 1];
      rhs[row] -= multiplier*rhs[row - 1];
    }
  }

  // Back substitution fills the interior second derivatives from the
  // last row upward.
  for (int row=interior - 1; row>=0; row--) {
    const double next = row + 1 < interior ? curvature[row + 2] : 0;
    curvature[row + 1] = (rhs[row] - upper[row]*next)/diagonal[row];
  }

  // Recover the endpoint second derivatives from the not-a-knot
  // relations the elimination removed: each endpoint extrapolates
  // linearly from its two neighboring interior values.
  const double first = x[1] - x[0];
  const double second = x[2] - x[1];
  curvature[0] = ((first + second)*curvature[1]
    - first*curvature[2])/second;
  const double last = x[count - 1] - x[count - 2];
  const double previous = x[count - 2] - x[count - 3];
  curvature[count - 1] = ((last + previous)*curvature[count - 2]
    - last*curvature[count - 3])/previous;

  free(diagonal);
  free(upper);
  free(rhs);

}

// reference_interval_3pcf: binary search for the interval index i with
// grid[i] <= coordinate < grid[i+1], clamped to the valid range.
// Only source-comparison readers call this function. Production preparation
// and evaluation find cells by arithmetic on declared uniform segments.
int reference_interval_3pcf(
  const double* grid, // increasing node coordinates [count]
  const int count, // number of nodes
  const double coordinate // query position
)
{
  int lower = 0;
  int upper = count - 1;
  while (upper - lower > 1) {
    const int middle = lower + (upper - lower)/2;
    if (grid[middle] <= coordinate) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return lower;
}


