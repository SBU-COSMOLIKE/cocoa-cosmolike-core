#ifndef COSMOLIKE_COUPLING_3PCF_H
#define COSMOLIKE_COUPLING_3PCF_H
#ifdef __cplusplus
extern "C" {
#endif

// G_ML(psi)=4*pi integral_0^pi P_L(cos delta)
//                       cos(2 beta_bar+M delta) d delta.
// Output has C shape [mmax+1,lmax+1,nquery]. psi lies in [0,pi/2].
// npsi=0 evaluates the integral's finite Fourier sum at every query.
// npsi>=2 linearly interpolates a uniform psi table, matching fastnc's
// tabulation convention. The historical grid contains 200 actual nodes.
void legendre_coupling_3pcf(
  int lmax, // largest Legendre order
  int mmax, // largest nonnegative angular Fourier order
  int nquery, // number of angular-ratio queries
  const double* psi, // [nquery], angular ratios in radians
  int npsi, // zero for direct values, or uniform tabulation length
  double* output // [M,L,query], caller-owned nonoverlapping storage
);

#ifdef __cplusplus
}
#endif
#endif
