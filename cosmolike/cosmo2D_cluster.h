#ifndef __COSMOLIKE_COSMO2D_CLUSTER_HPP
#define __COSMOLIKE_COSMO2D_CLUSTER_HPP
#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// Naming convention: (same as cosmo2D.h)
// ----------------------------------------------------------------------------
// c = cluster position ("c" as in "cluster")
// g = galaxy positions ("g" as in "galaxy")
// k = kappa CMB ("k" as in "kappa")
// s = kappa from source galaxies ("s" as in "shear")

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Correlation Functions (real Space) - Full Sky - bin average
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double w_cs_tomo(const int nt, const int nl, const int ni, const int ns, const int limber);

double w_cc_tomo(const int nt, const int nl1, const int nl2, const int ni, const int limber);

double w_cg_tomo(const int nt, const int nl, const int ni, const int nj, const int limber);

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Limber Approximation (Angular Power Spectrum)
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double C_cs_tomo_limber(const double l, const int nl, const int ni, const int ns);

double C_cc_tomo_limber(const double l, const int nl1, const int nl2, const int ni);

double C_cg_tomo_limber(const double l, const int nl, const int ni, const int nj);

// ----------------------------------------------------------------------------
// Non-Interpolated Version (Will compute the Integral at every call)
// ----------------------------------------------------------------------------

double C_cs_tomo_limber_nointerp(const double l, const int nl, const int ni, 
  const int ns, const int init);

double C_cc_tomo_limber_nointerp(const double l, const int nl1, const int nl2, 
  const int ni, const int init);

double C_cg_tomo_limber_nointerp(const double l, const int nl,const int ni, 
  const int nj, const int init);

// ----------------------------------------------------------------------------
// Integrands 
// ----------------------------------------------------------------------------

double int_for_C_cs_tomo_limber(double a, void* params);

double int_for_C_cg_tomo_limber(double a, void* params);

double int_for_C_cc_tomo_limber(double a, void* params);

/*
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// cluster number counts
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// nl = lambda_obs bin

double binned_N_nointerp(const int nl, const int nz, const int interpolate_survey_area, 
const int init_static_vars_only);

double binned_N(const int nl, const int nz);
*/

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD