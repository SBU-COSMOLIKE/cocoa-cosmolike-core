#ifndef __COSMOLIKE_COSMO3D_CLUSTER_HPP
#define __COSMOLIKE_COSMO3D_CLUSTER_HPP
#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double prob_lambda_obs_given_m_given_ztrue(double l, void* params);

double prob_lambda_obs_in_nl_given_m_given_ztrue_nointerp(
    const int nl, 
    const double m, 
    const double a,
    const int init,
  );

double prob_lambda_obs_in_nl_given_m_given_ztrue(
    const int nl, 
    const double m, 
    const double a
  );

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double int_ncl_given_lambda_obs_within_nl_given_ztrue(double lnM, void* params);

double ncl_given_lambda_obs_within_nl_given_ztrue_nointerp(
    const int nl, 
    const double a,
    const int init
  );

double ncl_given_lambda_obs_within_nl_given_ztrue(
    const int nl, 
    const double a
  );

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double int_cluster_b1_given_lambda_in_nl_obs_given_ztrue(double lnM, void* params);

double cluster_b1_given_lambda_obs_in_nl_given_ztrue_nointerp(
    const int nl, 
    const double a, 
    const int init
  ); // lambda weighted linear bias

double cluster_b1_given_lambda_obs_in_nl_given_ztrue(
    const int nl, 
    const double z
  ); // lambda weighted linear bias

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double pcc_linpsopt(
    const double k, 
    const double a, 
    const int nl1, 
    const int nl2, 
    const int linear
  );

double pcc_nointerp(
    const double k, 
    const double a, 
    const int nl1, 
    const int nl2
  );

double pcc(
    const double k, 
    const double a, 
    const int nl1, // observable richness bin
    const int nl2 // observable richness bin
  );

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double pcc_with_excl(
    const double k, 
    const double a, 
    const int nl1, 
    const int nl2
  );

double pcc_with_excl(
    const double k, 
    const double a, 
    const int nl1, // observable richness bin
    const int nl2 // observable richness bin
  );

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double int_pcm_1halo(double lnM, void* params);

double pcm_1halo(
    const double k, 
    const double a, 
    const int nl,     // observable richness bin
    const int init
  );

double pcm_1halo(
    const double k, 
    const double a, 
    const int nl,  // observable richness bin
    const int ni   // cluster redshift bin
  );

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
