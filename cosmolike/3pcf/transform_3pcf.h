#ifndef COSMOLIKE_TRANSFORM_3PCF_H
#define COSMOLIKE_TRANSFORM_3PCF_H

#include "physics_grid_3pcf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct transform_plan_3pcf transform_plan_3pcf;

// The plan turns bispectrum multipoles into the four natural
// components of the shear three-point correlation function - the
// complex three-shear correlators Gamma_0..Gamma_3 of Schneider &
// Lombardi (arXiv:astro-ph/0207454) - on a fixed triangle grid, with
// the multipole method of the fastnc paper by Sugiyama, Gomes & Jarvis
// (arXiv:2407.01798). The input values b_L(ell1,ell2) are the Legendre
// coefficients of the projected E-mode bispectrum B in mu, the cosine
// of the angle between the two Fourier multipole legs in the
// convention ell3^2 = ell1^2 + ell2^2 - 2*ell1*ell2*mu:
//
//   b_L = (2L+1)/2 integral[-1,1] B(mu) P_L(mu) dmu.
//
// The plan stores geometry only; no model or cosmology value survives
// inside it between calls.
transform_plan_3pcf* transform_create_3pcf(
  int nell, // even Fourier grid length
  const double* ell, // [nell], positive logarithmic grid
  int ntheta, // number of radial bins the transform keeps
  const int* indices, // [ntheta], positions on theta=xy/reversed(ell)
  int nphi, // opening-angle samples
  const double* phi, // [nphi], radians in (0,pi)
  int lmax, // highest Legendre degree
  int mmax, // highest angular Fourier mode
  int workers, // private scratch slices, fixed for the plan's lifetime
  double nu1, // first FFTLog bias (power-law tilt; fftlog_3pcf.h)
  double nu2, // second FFTLog bias
  double xy, // Fourier/radial grid alignment constant
  double dlnt, // radial logarithmic bin width; zero means point samples
  int padding, // zero cells at both ends of the Fourier grid
  int window, // 1 tapers the highest Fourier coefficients;
              // 0 keeps the untapered upstream fastnc window
  double window_width, // tapered fraction of the half spectrum
  int npsi, // angular-coupling grid: any size >= 2 interpolates a
            // uniform psi table (200 matches the upstream fastnc
            // table); 0 evaluates the coupling exactly
  double dphi, // opening-angle bin width: positive averages the
               // x-projected modes over each bin; zero keeps the
               // point values at the angle samples
  int projection, // 0 x projection; 1 the upstream TATT-comparison
                  // centroid phase; 2 its conjugate, which the fastnc
                  // main branch uses
  int convention, // 0 fastnc component order; 1 TreeCorr order/sign
  int ordered_legs // 1 general b_L; 0 the fastnc main-branch transpose
                   // shortcut, for b_L symmetric under leg exchange
);

void transform_destroy_3pcf(
  transform_plan_3pcf* plan // owned plan; NULL is accepted
);

void transform_natural_3pcf(
  transform_plan_3pcf* plan, // geometry and private worker scratch
  const double* multipoles, // [lmax+1,nell,nell], real b_L
  double* gamma_real, // [4,ntheta,ntheta,nphi], caller-owned
  double* gamma_imag // same axes as the real part
);

void transform_aperture_3pcf(
  transform_plan_3pcf* plan, // must have centroid projection
  const double* multipoles, // [lmax+1,nell,nell]
  int nfilter, // number of ordered aperture-radius triplets
  const double* weights_real, // [nfilter,4,ntriangle], TreeCorr order
  const double* weights_imag, // matching imaginary weights
  double* output // [nfilter], real aperture third moments
);

// The plan keeps the interpolation geometry and the intermediate b_L
// buffer. Every call refills b_L from the caller's physical grid; a
// grid with different interpolation axes rebuilds the geometry. No
// cosmological value survives inside the plan between calls.
void transform_aperture_grid_3pcf(
  transform_plan_3pcf* plan,
  const physics_grid_3pcf* grid,
  int nfilter,
  const double* weights_real,
  const double* weights_imag,
  double* output
);

#ifdef __cplusplus
}
#endif
#endif
