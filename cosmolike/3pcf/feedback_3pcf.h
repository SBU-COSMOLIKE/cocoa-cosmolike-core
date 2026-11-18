#ifndef COSMOLIKE_FEEDBACK_3PCF_H
#define COSMOLIKE_FEEDBACK_3PCF_H

#ifdef __cplusplus
extern "C" {
#endif

// Baryonic feedback changes matter clustering relative to a
// dark-matter-only (DMO) simulation. A paired response describes that
// change for two statistics independently: the power spectrum ratio
// R_P = P_hydro/P_DMO and the bispectrum ratio R_B = B_hydro/B_DMO. A
// single amplitude A scales both as R_X(A) = exp(A ln R_X), so A = 0
// is the DMO limit and A = 1 reproduces the loaded pair.
//
// The same z and log(k) axes serve the power spectrum and the middle
// bispectrum triangle side. With the triangle sides sorted
// a <= b <= c, the shape coordinates are r = b, u = a/b and
// v = (c - b)/a. Symmetry under side exchange belongs to the matter
// bispectrum, not to the ordered intrinsic-shape legs of a projected
// signal.
//
// The creation call borrows the input arrays and never keeps them.
// Each axis must be uniform. Creation resamples both tables onto a
// denser uniform grid with a natural cubic spline, so the linear
// interpolation of later queries follows the cubic shape.
typedef struct {
  int nz; // redshift nodes, at least four
  int nk; // log(k/[h/Mpc]) nodes, at least four
  int nu; // log(u) nodes, at least four
  int nv; // v nodes, at least four
  const double* z; // redshift coordinates [nz], uniform, z >= 0
  const double* logk; // log(k/[h/Mpc]) coordinates [nk], uniform
  const double* logu; // log(u) coordinates [nu], uniform, log(u) <= 0
  const double* v; // v coordinates [nv], uniform, 0 <= v <= 1
  const double* log_power; // [nz,nk], natural logarithm of positive R_P
  const double* log_bispectrum; // [nz,nk,nu,nv], last axis contiguous
  int boost; // dense intervals per loaded interval; 1 keeps the loaded grid
  int clamp; // 0 refuses out-of-domain queries; 1 holds each boundary value
} feedback_input_3pcf;

typedef struct feedback_3pcf feedback_3pcf;

// Build the immutable response object from one loaded input pair. A
// NULL input returns NULL, which the query functions accept whenever
// the amplitude is zero.
feedback_3pcf* feedback_create_3pcf(const feedback_input_3pcf* input);

// Release a response object; a NULL response does nothing.
void feedback_destroy_3pcf(feedback_3pcf* response);

// Multiplicative hydro-to-DMO power ratio exp(A ln R_P), which the
// caller applies to its DMO power spectrum.
double feedback_power_3pcf(
  const feedback_3pcf* response, // prepared tables; NULL only when A = 0
  double k,                      // comoving wavenumber in h/Mpc
  double z,                      // redshift
  double amplitude               // feedback amplitude A; 0 returns 1
);

// Multiplicative hydro-to-DMO bispectrum ratio exp(A ln R_B), which
// the caller applies to its DMO bispectrum.
double feedback_bispectrum_3pcf(
  const feedback_3pcf* response, // prepared tables; NULL only when A = 0
  const double* sides,           // triangle wavenumbers in h/Mpc [3]
  double z,                      // redshift
  double amplitude               // feedback amplitude A; 0 returns 1
);

#ifdef __cplusplus
}
#endif
#endif
