#ifndef COSMOLIKE_TRANSFORM_3PCF_H
#define COSMOLIKE_TRANSFORM_3PCF_H

#include "physics_grid_3pcf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct transform_plan_3pcf transform_plan_3pcf;

// Input b_L uses ell3^2=ell1^2+ell2^2-2 ell1 ell2 mu and the
// Legendre coefficient (2L+1)/2 integral[-1,1] B(mu) P_L(mu) dmu.
// No model or cosmology globals are retained by this geometry-only plan.
transform_plan_3pcf* transform_create_3pcf(
  int nell, // even Fourier grid length
  const double* ell, // [nell], positive logarithmic grid
  int ntheta, // selected radial bins
  const int* indices, // [ntheta], indices of xy/reversed(ell)
  int nphi, // angular samples
  const double* phi, // [nphi], radians in (0,pi)
  int lmax, // highest Legendre degree
  int mmax, // highest angular Fourier mode
  int workers, // private scratch slices, fixed for plan lifetime
  double nu1, // first FFTLog bias
  double nu2, // second FFTLog bias
  double xy, // Fourier/radial grid alignment
  double dlnt, // radial logarithmic width, zero means point samples
  int padding, // zero cells at both ends of the Fourier grid
  int window, // 0: historical untapered; 1: corrected taper
  double window_width, // high-frequency taper fraction
  int npsi, // 200: historical linear table; 0: exact angular coupling
  double dphi, // zero: point angle; positive: average X modes over bin
  int projection, // 0: X; 1: historical TATT centroid; 2: main centroid
  int convention, // 0: fastnc components; 1: TreeCorr sign/permutation
  int ordered_legs // 1: general b_L; 0: historical symmetric-b_L shortcut
);

void transform_destroy_3pcf(
  transform_plan_3pcf* plan // owned plan; NULL is accepted
);

void transform_natural_3pcf(
  transform_plan_3pcf* plan, // geometry and private worker scratch
  const double* multipoles, // [lmax+1,nell,nell], real b_L
  double* gamma_real, // [4,ntheta,ntheta,nphi], caller-owned
  double* gamma_imag // same axes as real part
);

void transform_aperture_3pcf(
  transform_plan_3pcf* plan, // must have centroid projection
  const double* multipoles, // [lmax+1,nell,nell]
  int nfilter, // number of ordered aperture-radius triplets
  const double* weights_real, // [nfilter,4,ntriangle], TreeCorr convention
  const double* weights_imag, // matching imaginary weights
  double* output // [nfilter], real aperture third moments
);

// Keep interpolation geometry and the intermediate b_L buffer inside this
// plan. Every call refills b_L from the supplied physical grid. Changed
// interpolation axes rebuild the geometry; no cosmological value is cached.
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
