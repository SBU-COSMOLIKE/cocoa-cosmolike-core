#ifndef __COSMOLIKE_CLUSTER_UTIL_HPP
#define __COSMOLIKE_CLUSTER_UTIL_HPP
#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// CLUSTER PROBABILITIES
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------
// BUZZARD 
// ---------------------------------------------------------------------------------------

double buzzard_P_lambda_obs_given_M(const double obs_lambda, const double M, const double z);

double buzzard_binned_P_lambda_obs_given_M(const int nl, const double M, const double z, 
  const int init_static_vars_only);

// ---------------------------------------------------------------------------------------
// SDSS
// ---------------------------------------------------------------------------------------

// Cocoa: we try to avoid reading of files directly in the cosmolike_core code 
void setup_SDSS_P_true_lambda_given_mass(int* io_nintrinsic_sigma, double** io_intrinsic_sigma, 
  int* io_natsrgm, double** io_atsrgm, double** io_alpha, double** io_sigma, int io);

// Cocoa: we try to avoid reading of files directly in the cosmolike_core code 
void setup_SDSS_P_lambda_obs_given_true_lambda(int* io_nz, double** io_z, int* io_nlambda, 
  double** io_lambda, double** io_tau, double** io_mu, double** io_sigma, double** io_fmask, 
  double** io_fprj, int io);

double SDSS_P_true_lambda_given_mass(const double true_lambda, const double mass, const double z);

double SDSS_P_lambda_obs_given_true_lambda(const double observed_lambda, const double true_lambda, 
  const double z);

double SDSS_P_lambda_obs_given_M(const double observed_lambda, const double M, const double z, 
  const int init_static_vars_only);

double SDSS_binned_P_lambda_obs_given_M(const int nl, const double M, const double z, 
  const int init_static_vars_only);

// ---------------------------------------------------------------------------------------
// Interface
// ---------------------------------------------------------------------------------------

double binned_P_lambda_obs_given_M_nointerp(const int nl, const double M, const double z, 
  const int init_static_vars_only);

double binned_P_lambda_obs_given_M(const int nl, const double M, const double z); 

// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// BINNED CLUSTER MASS FUNCTION 
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------

double dndlnM_times_binned_P_lambda_obs_given_M(double lnM, void* params);

// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// CLUSTER BIAS
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------

double BSF(const double M); // BSF = selection bias

double B1_x_BSF(const double M, const double a); 

double B2_x_BSF(const double M, const double a);

double B1M1_x_BSF(const double M, const double a); // (B1 - 1) x Selection Bias

double weighted_B1_nointerp(const int nl, const double z, const int init_static_vars_only);
double weighted_B1(const int nl, const double z);

double weighted_B2_nointerp(const int nl, const double z, const int init_static_vars_only);
double weighted_B2(const int nl, const double z);

double weighted_B1M1_nointerp(const int nl, const double z, const int init_static_vars_only);
double weighted_B1M1(const int nl, const double z);


// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// cluster number counts
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
double binned_Ndensity_nointerp(const int nl, const double z, const int init_static_vars_only);

double binned_Ndensity(const int nl, const double z);

// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// CLUSTER CROSS SPECTRUM WITH MATTER)
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------

double binned_p_cm_nointerp(const double k, const double a, const int nl, 
  const int include_1h_term, const int use_linear_ps, int init_static_vars_only);

// nl = lambda_obs bin, ni = cluster redshift bin
double binned_p_cm(const double k, const double a, const int nl, const int ni, 
const int include_1h_term, const int use_linear_ps);

// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// Area
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------

// Cocoa: we try to avoid reading of files in the cosmolike_core code 
void setup_get_area(int* io_nz, double** io_z, double** io_A, int io);

double get_area(const double zz, const int interpolate_survey_area);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD