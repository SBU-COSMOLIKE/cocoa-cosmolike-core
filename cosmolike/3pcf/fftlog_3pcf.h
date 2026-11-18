#ifndef COSMOLIKE_FFTLOG_3PCF_H
#define COSMOLIKE_FFTLOG_3PCF_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct fftlog_workspace_3pcf fftlog_workspace_3pcf;

// This workspace evaluates double-Bessel projection integrals on a
// logarithmic grid:
//
//   F(theta1,theta2) = integral dln(ell1) dln(ell2) f(ell1,ell2)
//                      J_a(ell1*theta1) J_b(ell2*theta2),
//
// where J_a and J_b are cylindrical Bessel functions. The caller fills
// the real grid f once; one forward FFT then serves inverse transforms
// for many order pairs (a,b). The shear three-point transform uses
// these integrals to turn bispectrum multipoles into the multipoles of
// the natural components.

// Build one workspace. The constructor copies every grid and setting,
// so the caller may free its own arrays afterward. Creation and
// destruction must stay serial because FFTW planning is not thread
// safe; the numerical calls below may then run concurrently on
// distinct worker slices. The grid ell must be positive, increasing
// and uniformly spaced in the logarithm, with an even length. The
// biases nu1 and nu2 are the power-law tilts that FFTLog divides out
// of f, one per axis, and must lie strictly inside (0.5,2.5) so that
// every kernel table entry stays finite.
fftlog_workspace_3pcf* fftlog_create_3pcf(
  int n, // unpadded length on each axis, even
  const double* ell, // [n], logarithmic Fourier grid
  int max_order, // largest absolute cylindrical Bessel order
  int workers, // number of independent scratch slices
  double nu1, // first-axis FFTLog bias (power-law tilt)
  double nu2, // second-axis FFTLog bias
  double xy, // output radii are theta=xy/reversed(ell)
  double dlnt, // logarithmic radial-bin width; zero for point values
  int padding, // zero cells added at each end of both axes
  int window, // 1 tapers the highest Fourier coefficients to damp
              // ringing; 0 keeps the untapered upstream fastnc window
  double window_width // tapered fraction of the half spectrum, [0,1]
);

void fftlog_destroy_3pcf(
  fftlog_workspace_3pcf* work // owning workspace, NULL is allowed
);

// Optional: keep only the radial nodes a measurement uses. Call once,
// before any worker runs a transform; inverse results then hold the
// compact [ntheta,ntheta] samples instead of the full grid.
void fftlog_select_3pcf(
  fftlog_workspace_3pcf* work, // workspace, before any execution
  int ntheta, // number of radial nodes the caller keeps
  const int* indices // increasing positions on the unpadded output grid
);

double* fftlog_input_3pcf(
  fftlog_workspace_3pcf* work, // owning workspace
  int worker // private slice, 0<=worker<workers
);

void fftlog_forward_3pcf(
  fftlog_workspace_3pcf* work, // workspace whose input the caller filled
  int worker // private slice holding f(ell1,ell2)
);

// Returns a private [n,n] matrix, or [ntheta,ntheta] after selection.
// The next inverse on this worker overwrites it. The forward
// coefficients survive and can serve further Bessel orders.
const double* fftlog_inverse_3pcf(
  fftlog_workspace_3pcf* work, // workspace holding forward coefficients
  int worker, // private slice
  int order1, // nonnegative first Bessel order
  int order2 // nonnegative second Bessel order
);

int next_fft_size_3pcf(
  int minimum // smallest acceptable length; the result is the next
              // even length whose only prime factors are 2, 3 and 5
);

#ifdef __cplusplus
}
#endif
#endif
