#ifndef COSMOLIKE_BASELINE_3PCF_H
#define COSMOLIKE_BASELINE_3PCF_H

#ifdef __cplusplus
extern "C" {
#endif

// A supplied matter bispectrum B. The caller tabulates ln B, where B is
// the positive numerical value in (Mpc/h)^6, and the table must describe
// the same matter definition and cosmology as the power and background
// tables of the physical model. This is an absolute baseline rather than
// a feedback ratio: it stores B itself, not B relative to another model.
// The coordinates follow the sorted-triangle convention: with the sides
// sorted as a <= b <= c, the table is indexed by k = b, u = a/b and
// v = (c - b)/a. Every axis is uniform and carries at least four nodes.
// Preparation copies all arrays, upsamples them once with natural cubics
// and afterwards reads them by linear arithmetic-index lookups, so every
// query returns interpolated input and nothing else: no fitting formula
// enters the table, and the code infers neither a background nor an
// extrapolation from it. The one explicit boundary choice is clamp = 1,
// which holds each boundary value constant outside the declared domain.
typedef struct {
  int nz; // redshift count
  int nk; // log(k/[h/Mpc]) count
  int nu; // log(u) count
  int nv; // v count
  const double* z; // redshift axis [nz], z>=0
  const double* logk; // natural log wavenumber axis [nk]
  const double* logu; // natural log side-ratio axis [nu], <=0
  const double* v; // shape axis [nv], between zero and one
  const double* log_bispectrum; // ln B [nz,nk,nu,nv], last axis contiguous
  int boost; // dense intervals per input interval, >=1
  int clamp; // 0 rejects queries outside the table; 1 holds each boundary
} baseline_input_3pcf;

typedef struct baseline_3pcf baseline_3pcf;

// Validate axes and finite logarithms without allocating or upsampling.
void baseline_validate_3pcf(
  const baseline_input_3pcf* input // required input table
);
baseline_3pcf* baseline_create_3pcf(
  const baseline_input_3pcf* input // borrowed arrays; NULL returns NULL
);
void baseline_destroy_3pcf(
  baseline_3pcf* baseline // owned object; NULL does nothing
);
double baseline_bispectrum_3pcf(
  const baseline_3pcf* baseline, // prepared physical B table
  const double* sides, // validated triangle sides [3], h/Mpc, any order
  double z // redshift inside the declared domain
);

#ifdef __cplusplus
}
#endif
#endif
