#ifndef __COSMOLIKE_COSMO2D_SCUTS_H
#define __COSMOLIKE_COSMO2D_SCUTS_H
#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// Naming convention:
// ----------------------------------------------------------------------------
// c = cluster position ("c" as in "cluster")
// g = galaxy positions ("g" as in "galaxy")
// k = kappa CMB ("k" as in "kappa")
// s = kappa from source galaxies ("s" as in "shear")

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// DERIVATIVE: dlnX/dlnk: important to determine scale cuts (2011.06469 eq 17)
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double dlnxi_dlnk_pm_tomo(
    const double k,
    const int pm, 
    const int nt, 
    const int ni, 
    const int nj
  );

double dlnC_ss_dlnk_tomo_limber_nointerp(
    const double k, 
    const double l,
    const int ni, 
    const int nj, 
    const int EE, 
    const int init
  );

double dlnC_ss_dlnk_tomo_limber(
    const double k,
    const double l, 
    const int ni, 
    const int nj, 
    const int EE
  );

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD