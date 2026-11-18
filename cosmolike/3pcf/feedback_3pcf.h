#ifndef COSMOLIKE_FEEDBACK_3PCF_H
#define COSMOLIKE_FEEDBACK_3PCF_H

#ifdef __cplusplus
extern "C" {
#endif

// A paired response specifies P_hydro/P_DMO and B_hydro/B_DMO independently.
// The same z and log(k) axes describe P and the middle B triangle side.
// For sorted sides a<=b<=c: r=b, u=a/b, v=(c-b)/a. Symmetry belongs to
// matter, not to the ordered intrinsic-shape legs in a projected signal.
// Input arrays are borrowed only during creation. They must be uniform;
// natural cubic preparation makes dense uniform log-response tables.
typedef struct {
  int nz; // redshift nodes, at least four
  int nk; // log(k/[h/Mpc]) nodes, at least four
  int nu; // log(u) nodes, at least four
  int nv; // v nodes, at least four
  const double* z;
  const double* logk;
  const double* logu;
  const double* v;
  const double* log_power; // [nz,nk], natural logarithm of positive R_P
  const double* log_bispectrum; // [nz,nk,nu,nv], last axis contiguous
  int boost; // dense interval multiplier; one retains the supplied grid
  int clamp; // 0 refuses out-of-domain queries; 1 holds each boundary value
} feedback_input_3pcf;

typedef struct feedback_3pcf feedback_3pcf;
feedback_3pcf* feedback_create_3pcf(const feedback_input_3pcf* input);
void feedback_destroy_3pcf(feedback_3pcf* response);
double feedback_power_3pcf(
  const feedback_3pcf* response,
  double k,
  double z,
  double amplitude
);
double feedback_bispectrum_3pcf(
  const feedback_3pcf* response,
  const double* sides,
  double z,
  double amplitude
);

#ifdef __cplusplus
}
#endif
#endif
