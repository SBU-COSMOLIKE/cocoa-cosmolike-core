#ifndef COSMOLIKE_PHYSICS_REFERENCE_3PCF_H
#define COSMOLIKE_PHYSICS_REFERENCE_3PCF_H

// Internal helpers for the explicit upstream-comparison reader. Callers
// validate increasing finite coordinates, finite samples and count>=2.
// Production uses basics.c uniform coefficients and arithmetic indexing.
// No helper retains state; independent models and readers may run in parallel.

void reference_spline_coeffs_3pcf(
  const double* x, // increasing coordinates [count]
  const double* y, // finite samples [count]
  int count, // at least two nodes
  double* curvature // zero-initialized output [count], S'' convention
);

int reference_interval_3pcf(
  const double* grid, // increasing coordinates [count]
  int count, // at least two nodes
  double coordinate // finite query; returned cell clamps to an edge
);

#endif
