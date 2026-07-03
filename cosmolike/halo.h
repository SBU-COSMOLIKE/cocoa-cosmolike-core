#ifndef __COSMOLIKE_HALO_H
#define __COSMOLIKE_HALO_H
#ifdef __cplusplus
extern "C" {
#endif

// HALO BIAS OPTIONS ---------------------------
#define HALO_BIAS_TINKER_2010 0

// HMF OPTIONS ---------------------------------
#define HMF_TINKER_2010 0

// CONCENTRATION OPTIONS -----------------------
#define CONCENTRATION_BHATTACHARYA_2013 0

// HALO PROFILE OPTIONS OPTIONS -----------------------
#define HALO_PROFILE_NFW 0

// In halo.h, alongside the other existing function declarations


double hm_funcs_nointerp(const int ni, const double a, const int func, const int init);
double test_u_ia_sat(const double k, const double m, const double a);
double I_for_IA_nointerp(const double k, const double a, const int ni,
                         const int ia_func, const int init);
double n_red_sat_bar(const int ni, const double a);
double p_II_1h_nointerp(const double k, const double a, const int ni);
double int_for_bred_cen(double lnM, void* params);
double I_bred_cen_nointerp(const double a, const int ni, const int func, const int init);
double b_red_cen(const int ni, const double a);
double A_nla_cen(const int ni, const double a);
double p_II_2h_cen_nointerp(const double k, const double a, const int ni);
double p_dI_2h_cen_nointerp(const double k, const double a, const int ni);
double p_dI_1h_nointerp(const double k, const double a, const int ni);
double p_mm(const double k, const double a);

double p_gm(const double k, const double a, const int ni);

double p_gg(const double k, const double a, const int ni, const int nj);

double p_my(const double k, const double a);

double p_yy(const double k, const double a);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
