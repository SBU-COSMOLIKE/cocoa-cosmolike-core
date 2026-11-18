#ifndef COSMOLIKE_FFTLOG_3PCF_H
#define COSMOLIKE_FFTLOG_3PCF_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct fftlog_workspace_3pcf fftlog_workspace_3pcf;

// All geometry is copied. Plan creation/destruction must be serial.
// n is even and ell is positive, increasing, uniformly spaced in log.
// Both biases lie in (0.5,2.5); dlnt=0 selects point values. padding is
// the number of zero cells added on each side of both logarithmic axes.
// window=0 reproduces the historical untapered TATT transform; window=1
// applies the coefficient taper whose result the historical code discarded.
fftlog_workspace_3pcf* fftlog_create_3pcf(
  int n, // unpadded length on each axis
  const double* ell, // [n], logarithmic Fourier grid
  int max_order, // largest absolute cylindrical Bessel order
  int workers, // number of independent scratch slices
  double nu1, // first FFTLog bias
  double nu2, // second FFTLog bias
  double xy, // theta=xy/reversed(ell)
  double dlnt, // logarithmic radial-bin width, zero for point values
  int padding, // zero padding per end
  int window, // whether to taper coefficients
  double window_width // tapered fraction, in [0,1]
);

void fftlog_destroy_3pcf(
  fftlog_workspace_3pcf* work // owning workspace, NULL is allowed
);

// Optional immutable output selection. Call once, before any worker uses
// this workspace. Inverse results then contain [ntheta,ntheta] samples.
void fftlog_select_3pcf(
  fftlog_workspace_3pcf* work, // prepared FFT geometry
  int ntheta, // selected radial count
  const int* indices // increasing indices on the unpadded output grid
);

double* fftlog_input_3pcf(
  fftlog_workspace_3pcf* work, // owning workspace
  int worker // private slice, 0<=worker<workers
);

void fftlog_forward_3pcf(
  fftlog_workspace_3pcf* work, // workspace with input filled
  int worker // private slice containing f(ell1,ell2)
);

// Returns a private [n,n] matrix, or [ntheta,ntheta] after selection. The
// next inverse on this worker overwrites it. The forward coefficients
// survive and can serve further Bessel orders.
const double* fftlog_inverse_3pcf(
  fftlog_workspace_3pcf* work, // prepared forward coefficients
  int worker, // private slice
  int order1, // nonnegative first Bessel order
  int order2 // nonnegative second Bessel order
);

int next_fft_size_3pcf(
  int minimum // smallest requested length; returns an even 2/3/5 size
);

#ifdef __cplusplus
}
#endif
#endif
