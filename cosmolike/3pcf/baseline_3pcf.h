#ifndef COSMOLIKE_BASELINE_3PCF_H
#define COSMOLIKE_BASELINE_3PCF_H

#ifdef __cplusplus
extern "C" {
#endif

// Supplied positive matter bispectrum B, in (Mpc/h)^6. The logarithm is
// of its numerical value in these units. This is an absolute baseline,
// not a feedback ratio: the caller must supply the same matter definition
// and cosmology as the power and background tables of the physical model.
// Sort triangle sides a<=b<=c and use k=b, u=a/b, v=(c-b)/a. Every axis
// is uniform with at least four nodes. Preparation copies all arrays,
// cubic-upsamples once and uses linear arithmetic-index reads afterwards.
// No fitting formula, inferred background, or inferred extrapolation is
// part of this table. clamp=1 explicitly chooses constant boundary values.
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
