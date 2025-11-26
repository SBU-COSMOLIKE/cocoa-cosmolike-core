#ifndef __COSMOLIKE_COSMO3D_CLUSTER_HPP
#define __COSMOLIKE_COSMO3D_CLUSTER_HPP
#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// CLUSTER POWER SPECTRUM
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------

double pcc_given_lambda_obs(const double k, 
                            const double a, 
                            const int nl1, 
                            const int nl2, 
                            const int use_linear_ps);

// ---------------------------------------------------------------------------------------
// binned_p_cc_incl_halo_exclusion_nointerp has an usual interface (no doubles for k and a vars) 
// given the need for threading FFTW calls; nk = k bin, na = a bin and the {k, a} grid must be

// N_k = Ntable.N_k_halo_exclusion;
// ln_k_min = log(1E-2); 
// ln_k_max = log(3E6);
// dlnk = (ln_k_max - ln_k_min)/((double) N_k - 1.0);  

// N_a = Ntable.N_a_halo_exclusion;
// amin = 1.0/(1.0 + tomo.cluster_zmax[tomo.cluster_Nbin - 1]);
// amax = 1.0/(1.0 + tomo.cluster_zmin[0]) + 0.01;
// da = (amax - amin)/((double) N_a - 1.0);

double binned_p_cc_incl_halo_exclusion_nointerp(const int nl1, const int nl2, const int na, 
  const int nk, const int init_static_vars_only);
// ---------------------------------------------------------------------------------------

double binned_p_cc_incl_halo_exclusion(const double k, const double a, const int nl1, const int nl2);

double binned_p_cc_incl_halo_exclusion_with_constant_lambd(const double k, const double a, 
  const int nl1, const int nl2);

// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// CLUSTER CROSS SPECTRUM WITH GALAXIES 
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------

// nl = lambda_obs bin, nj = galaxy redshift bin
double binned_p_cg(const double k, const double a, const int nl, const int nj, 
  const int use_linear_ps);
