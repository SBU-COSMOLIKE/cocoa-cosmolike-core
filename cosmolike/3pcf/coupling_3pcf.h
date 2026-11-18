#ifndef COSMOLIKE_COUPLING_3PCF_H
#define COSMOLIKE_COUPLING_3PCF_H
#ifdef __cplusplus
extern "C" {
#endif

// G_ML(psi) is the angular coupling function of the multipole transform,
//
//   G_ML(psi) = 4*pi integral_0^pi P_L(cos delta)
//                        cos(2 beta_bar + M delta) d delta.
//
// The transform contracts the bispectrum multipoles b_L with this table:
// G_ML carries every geometric factor that couples the Legendre degree L
// of the bispectrum expansion to the Fourier order M of the real-space
// natural shear components, so the physical bispectrum never re-enters
// the angular integral. The fastnc method paper (Sugiyama, Gomes and
// Jarvis, arXiv:2407.01798) introduces this coupling.
//
// psi is the opening angle arctan(ell2/ell1) in [0, pi/2] - an angle
// that encodes the ratio of the two multipole legs, not itself a ratio.
// delta is the integration angle between the two legs, whose cosine the
// Legendre polynomial takes, and beta_bar is the phase angle between the
// resultant of the two legs and their bisector; coupling_3pcf.c derives
// both the phase and the finite sum that evaluates the integral.
//
// The output array has C shape [mmax+1,lmax+1,nquery]. npsi=0 evaluates
// the finite sum exactly at every query. npsi>=2 instead interpolates a
// uniform psi table on [0,pi/2] with npsi nodes linearly, reproducing
// fastnc's tabulation convention; the fastnc default table holds 200
// nodes.
void legendre_coupling_3pcf(
  int lmax, // largest Legendre degree L
  int mmax, // largest nonnegative Fourier order M
  int nquery, // number of opening-angle queries
  const double* psi, // [nquery], opening angles in radians
  int npsi, // zero for exact values, or uniform tabulation length
  double* output // [mmax+1,lmax+1,nquery], caller-owned, nonoverlapping
);

#ifdef __cplusplus
}
#endif
#endif
