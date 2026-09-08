#include <assert.h>
#include <gsl/gsl_deriv.h>
#include <gsl/gsl_sf.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_integration.h>

#include "halo.h"
#include "basics.h"
#include "cosmo3D.h"
#include "IA.h"
#include "redshift_spline.h"
#include "structs.h"

#include "log.c/src/log.h"

#define DEFAULT_INT_PREC 1000
#define delta_c 1.686
#define Delta 200

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// BASIC PEAK BACKGROUND SPLIT ROUTINES
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
static const double F2_ANGULAR = 3 * M_PI  / 2.0;   // ~= 4.712
double hb1nu(const double nu, const double a)
{ // Halo bias based on peak-background split

  double ans;
  switch(like.halo_model[1])
  {
    case HALO_BIAS_TINKER_2010:
    {
      const double y = log10(200.0);
      
      const double ALPHA    = 1.0 + 0.24 * y * exp(-pow(4.0 / y, 4.0));
      const double nu_alpha = pow(nu, 0.44 * y - 0.88);
      
      const double BETA = 0.183;
      const double nu_beta = pow(nu, 1.5);
      
      const double GAMMA = 0.019 + 0.107 * y + 0.19 * exp(-pow(4.0 / y, 4.0));
      const double nu_gamma = pow(nu, 2.4);
      
      ans = 1.0 - ALPHA * nu_alpha / (nu_alpha + pow(delta_c, 0.44 * y - 0.88)) 
               + BETA * nu_beta + GAMMA * nu_gamma;
      break;
    }
    default:
    {
      log_fatal("like.halo_model[1] = %d not supported", like.halo_model[1]);
      exit(1);  
    }
  }
  return ans;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double fnu(const double nu, const double a)
{ // Halo bias based on peak-background split 
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  double ans;
  switch(like.halo_model[0])
  {
    case HMF_TINKER_2010:
    { // Eqs. (8-12) + Table 4 from Tinker et al. 2010
      const double aa = fmax(0.25, a); // limit fit range of mass function evolution to
                                       // z <= 3 (discussed after Eq. 12 of 1001.3162)
      const double alpha = 0.368;
      const double beta = 0.589 * pow(aa, -0.2);
      const double gamma = 0.864 * pow(aa, 0.01);
      const double phi = -0.729 * pow(aa, .08);
      const double eta = -0.243 * pow(aa, -0.27);

      ans = alpha*(1. + pow(beta*nu,-2*phi))*pow(nu,2*eta)*exp(-gamma*nu*nu/2.);
      break;
    }
    default:
    {
      log_fatal("like.halo_model[0] = %d not supported", like.halo_model[0]);
      exit(1);  
    }
  }
  return ans;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double conc(const double m, const double growfac_a) 
{
  double ans;
  switch(like.halo_model[2])
  {
    case CONCENTRATION_BHATTACHARYA_2013:
    { // Bhattacharya et al. 2013, Delta = 200 rho_{mean} (Table 2)
      const double nu = delta_c/(sqrt(sigma2(m))*growfac_a);
      ans =  9.0*pow(nu, -0.29)*pow(growfac_a, 1.15); 
      break;
    }
    default:
    {
      log_fatal("like.halo_model[2] = %d not supported", like.halo_model[2]);
      exit(1);  
    }
  }
  return ans;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_bias_norm(double nu, void* params) 
{ // correction for halo mass cuts so large-scale 2h matches PT at all redshifts 
  double* ar = (double*) params;
  const double a = ar[0];
  return hb1nu(nu, a) * fnu(nu, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double bias_norm_nointerp(const double a, const int init)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = DEFAULT_INT_PREC + 500*Ntable.high_def_integration;
    if (w != NULL)  gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  const double growfac_a = growfac(a);
  const double nu_min = delta_c/(sqrt(sigma2(limits.halo_m_min))*growfac_a);
  const double nu_max = delta_c/(sqrt(sigma2(limits.halo_m_max))*growfac_a);

  double ar[2] = {a, growfac_a};

  double res;
  if (init == 1) {
    res = int_for_bias_norm(0.5*(nu_min+nu_max), (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_bias_norm;
    res = gsl_integration_glfixed(&F, nu_min, nu_max, w);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double bias_norm(const double a) 
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double* table = NULL;
  static double lim[3];
  
  if (NULL == table  || fdiff2(cache[1], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double*) malloc(sizeof(double)*Ntable.N_a);
    lim[0] = limits.a_min; 
    lim[1] = 0.9999999;
    lim[2] = (lim[1] - lim[0]) / ((double) Ntable.N_a - 1.0);
  }
  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random)) {
    (void) bias_norm_nointerp(lim[0], 1); // init static vars
    #pragma omp parallel for schedule(static,1)
    for (int i=0; i<Ntable.N_a; i++) {
      table[i] = bias_norm_nointerp(lim[0] + i*lim[2], 0);
    }
    table[Ntable.N_a-1] = 1.0;
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
  return interpol1d(table, Ntable.N_a, lim[0], lim[1], lim[2], a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double lognu0_gsl(double lnM, void* params __attribute__((unused))) 
{ 
  return log(delta_c/sqrt(sigma2(exp(lnM)))); 
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double dlognudlogm(const double M) 
{ // if sigma(z) \propto to D(z), then d\ln \nu/dlnM independent of z
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double* table = NULL;
  static double lim[3];

  if (NULL == table || fdiff2(cache[1], Ntable.random))
  {
    if (table != NULL) free(table);
    table = (double*) malloc(sizeof(double) * Ntable.N_M);
    lim[0] = log(limits.halo_m_min);
    lim[1] = log(limits.halo_m_max);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_M - 1.0);
  }

  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random))
  {
    {
      const int i = 0;
      double result, abserr;
      double ar[1] = {1.0};

      gsl_function F;
      F.function = &lognu0_gsl;
      F.params = (void*) ar;

      int status = gsl_deriv_central(&F, lim[0]+i*lim[2], 
                                    0.1*(lim[0]+i*lim[2]), &result, &abserr);
      if (status) { log_fatal(gsl_strerror(status)); exit(1); }
      table[i] = result;
    }
    #pragma omp parallel for schedule(static,1)
    for (int i=1; i <Ntable.N_M; i++) 
    {
      double result, abserr;
      double ar[1] = {1.0};

      gsl_function F;
      F.function = &lognu0_gsl;
      F.params = (void*) ar;

      int status = gsl_deriv_central(&F, lim[0]+i*lim[2], 
                                    0.1*(lim[0]+i*lim[2]), &result, &abserr);
      if (status) { log_fatal(gsl_strerror(status)); exit(1); }
      table[i] = result;
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }  
  return interpol1d(table, Ntable.N_M, lim[0], lim[1], lim[2], log(M));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HALO PROFILES
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double u_nfw_c(
    const double c, 
    const double k, 
    const double m, 
    const double a
  ) 
{ // analytic FT of NFW profile, from Cooray & Sheth 01 
  const double rho_delta = Delta * cosmology.rho_crit * cosmology.Omega_m;
  const double r_delta = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);
  const double x = k * r_delta / c;
  const double xu = (1. + c) * x;

  gsl_sf_result SI_XU;
  int status = gsl_sf_Si_e(xu, &SI_XU);
  if (status) {
    log_fatal(gsl_strerror(status)); exit(1);
  }

  gsl_sf_result SI_X;
  {
    int status = gsl_sf_Si_e(x, &SI_X);
    if (status) {
      log_fatal(gsl_strerror(status)); exit(1);
    }
  }

  gsl_sf_result CI_XU;
  {
    int status = gsl_sf_Ci_e(xu, &CI_XU);
    if (status) {
      log_fatal(gsl_strerror(status)); exit(1);
    }
  }

  gsl_sf_result CI_X;
  {
    int status = gsl_sf_Ci_e(x, &CI_X);
    if (status) {
      log_fatal(gsl_strerror(status)); exit(1);
    }
  }
  return (sin(x)*(SI_XU.val - SI_X.val) 
          - sinl(c*x)/xu 
          + cos(x)*(CI_XU.val - CI_X.val))/(log(1. + c) - c/(1. + c));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double u_c(
    const double c, 
    const double k, 
    const double m, 
    const double a
  ) 
{
  double ans;
  switch(like.halo_model[3])
  {
    case HALO_PROFILE_NFW:
    {
      ans = u_nfw_c(c, k, m, a);
      break;
    }
    default:
    {
      log_fatal("like.halo_model[3] = %d not supported", like.halo_model[3]);
      exit(1);  
    }
  }
  return ans;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GALAXY PROFILES
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------


double u_g(
    const double c, 
    const double k, 
    const double m, 
    const double a,
    const int ni
  ) 
{
  double gc = nuisance.gc[ni];
  if (!(gc > 0)) gc = 1.0;      // default: galaxies trace DM
  return u_nfw_c(c*gc, k, m, a);

}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double HOD_nc(const double m, const double a, const int ni)
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  if (ni < 0 || ni > redshift.clustering_nbin - 1) { 
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  if (nuisance.hod[ni][0] < 10 || nuisance.hod[ni][0] > 16) {
    log_fatal("HOD parameters in redshift bin %d not set", ni); exit(1);
  }

  const double x = (log10(m) - nuisance.hod[ni][0])/nuisance.hod[ni][1];
  
  gsl_sf_result ERF;
  {
    int status = gsl_sf_erf_e(x, &ERF);
    if (status) {
      log_fatal(gsl_strerror(status)); exit(1);
    }
  }
  return 0.5*(1.0 + ERF.val);
}

double HOD_ns(
    const double m, 
    const double a, 
    const int ni
  )
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1) { 
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  const double x = (m - pow(10., nuisance.hod[ni][3]))/pow(10., nuisance.hod[ni][2]);
  // Below M_0 the satellite term is undefined (negative base in pow).
  // Physically there are no satellites below the cutoff mass, so ns = 0.
  if (x <= 0.0) {
    return 1.e-15;
  }
  const double ns = HOD_nc(m, a, ni)*pow(x, nuisance.hod[ni][4]);
  return (ns > 0) ? ns : 1.e-15;
}

double HOD_fc(const int ni)
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  return (nuisance.hod[ni][5]) ? nuisance.hod[ni][5] : 1.0;
}


double f_red_cen(const double m, const int ni)
{
    const double w = nuisance.hod[ni][7];
    if (!(w > 0)) return 0.0;   // unset/zero width -> no red split (avoid div-by-0)
    const double x = (log10(m) - nuisance.hod[ni][6]) / w;
    return 0.5 * (1.0 + tanh(x));
}


double f_red_sat(const double m, const int ni)
{
    // Satellites are generally less red than centrals
    const double x = (log10(m) - nuisance.hod[ni][8]) / nuisance.hod[ni][9];
    return 0.5 * (1.0 + tanh(x));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GAS PROFILES
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_F0_KS(double x, void* params __attribute__((unused)))
{
  return x*x*pow(log(1.0 + x)/x, 1.0/(nuisance.gas[0] - 1.0));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double F0_KS_nointerp(double c, const int init)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = DEFAULT_INT_PREC + 500*Ntable.high_def_integration;
    if (w != NULL)  gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[1] = {0.0};
  const double xmin = 0.0;
  const double xmax = c;
  
  double res = 0.0;
  if (1 == init) {
    res = int_F0_KS((xmin + xmax)/2.0, (void*) ar);
  }
  else
  {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_F0_KS;
    res = gsl_integration_glfixed(&F, xmin, xmax, w);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_F_KS(double x, void* params)
{
  double* ar = (double*) params;
  const double y = ar[0];  
  return (x*sinl(y*x)/y)*
         pow(log(1.0 + x)/x, nuisance.gas[0]/(nuisance.gas[0] - 1.0));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double F_KS_nointerp(double c, double krs, const int init) 
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = DEFAULT_INT_PREC + 500*Ntable.high_def_integration;
    if (w != NULL)  gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[1] = {krs};
  const double cmin = 0.0;
  const double cmax = c;

  double res = 0.0;
  if (1 == init) {
    res = int_F_KS((cmin + cmax)/2.0, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_F_KS;
    res = gsl_integration_glfixed(&F, cmin, cmax, w);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double u_KS(double c, double k, const double rv)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = 0;
  static double* norm = 0;
  static double  lim[2][3]; // lim[0][0] = cmin;  lim[1][0] = lnxmin;
                            // lim[0][1] = cmax;  lim[1][1] = lnxmax;
                            // lim[0][2] = dc;    lim[1][2] = dlnx; 
 
  if (NULL == table || fdiff2(cache[1], Ntable.random)) {   
    if (table != NULL) free(table); 
    table = (double**) malloc2d(Ntable.halo_uks_nc, Ntable.halo_uks_nx);
    if (norm != NULL) free(norm); 
    norm = (double*) malloc1d(Ntable.halo_uks_nc);

    lim[0][0] = limits.halo_uks_cmin; 
    lim[0][1] = limits.halo_uks_cmax;
    lim[0][2] = (lim[0][1] - lim[0][0])/((double) Ntable.halo_uks_nc - 1.);
    lim[1][0] = log(limits.halo_uks_xmin); // full range of possible k*R_200/c in the code
    lim[1][1] = log(limits.halo_uks_xmax); 
    lim[1][2] = (lim[1][1] - lim[1][0])/((double) Ntable.halo_uks_nx - 1.); 
  }

  if (fdiff2(cache[0], nuisance.random_gas) || fdiff2(cache[1], Ntable.random)) 
  { 
    (void) F0_KS_nointerp(lim[0][0], 1);                 // init static vars
    (void) F_KS_nointerp(lim[0][0],exp(lim[1][0]), 1);   // init static vars
    #pragma omp parallel for schedule(static,1)
    for (int i=0; i<Ntable.halo_uks_nc; i++) {
      norm[i] = F0_KS_nointerp(lim[0][0] + i*lim[0][2], 0);
    }
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Ntable.halo_uks_nc; i++) {
      for (int j=0; j<Ntable.halo_uks_nx; j++) {
        table[i][j] = F_KS_nointerp(lim[0][0]+i*lim[0][2], 
                                    exp(lim[1][0]+j*lim[1][2]), 0)/norm[i];
      }
    }
    cache[0] = nuisance.random_gas; 
    cache[1] = Ntable.random;
  }
  return interpol2d(table, 
    Ntable.halo_uks_nc, lim[0][0], lim[0][1], lim[0][2], c, 
    Ntable.halo_uks_nx, lim[1][0], lim[1][1], lim[1][2], log(k * rv/c));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double frac_bnd(double M)
{
  const double M0 = pow(10.0, nuisance.gas[2]);
  return cosmology.Omega_b/(cosmology.Omega_m*(1.0+ pow(M0/M, nuisance.gas[1])));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double frac_ejc(double M)
{
  const double logM = log10(M);
  const double delta = (logM - nuisance.gas[7])/nuisance.gas[8];
  
  const double tmp = nuisance.gas[6] * exp(-0.5*delta*delta);  
  const double frac_star = ((logM > nuisance.gas[2]) && 
                           (tmp < nuisance.gas[6]/3.0)) ? nuisance.gas[6]/3.0 : tmp; 
  
  return frac_bnd(M) - frac_star;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double u_y_bnd(double c, double k, double m, double a)
{ //unit: [G(M_solar/h)^2 / (c/H0)]
  
  const double rho_delta = Delta * cosmology.rho_crit * cosmology.Omega_m;
  const double r_delta = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);
  const double rv = r_delta;

  const double mu_p = 4.0/(3.0 + 5*nuisance.gas[10]);
  const double mu_e = 2.0/(1.0 + nuisance.gas[10]);
  
  return (2.0*nuisance.gas[5]/(3.0*a))*(mu_p/mu_e)*frac_bnd(m)*m*(m/rv)*u_KS(c, k, rv);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double u_y_ejc(double m)
{ // [m] = [Msun/h]
  const double num_p = 1.1892e57; // proton number in 1 solar mass, unit [1/Msun]
  
  // convert ejected gas T (K) to E (eV) then to [G (Msun/h)^2 / (c/H0) * h]
  const double E_w = pow(10,nuisance.gas[9]) * 8.6173e-5 * 5.616e-44;
  const double mu_e = 2./(1.+nuisance.gas[10]);
  
  return (num_p * m * frac_ejc(m) / mu_e) * E_w; // final unit in [G(Msun/h)^2 / (c/H0)]
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
/*
double n_s_cmv(double a) 
{ 
  double dV_dz = pow(f_K(chi(a)), 2.0) / hoverh0(a); // comoving dV/dz per radian^2
  return nz_source_photoz(1.0/a - 1., -1) * survey.n_gal * 
    survey.n_gal_conversion_factor / dV_dz; // dN/dz/radian^2/(dV/dz/radian^2)
}
*/
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HALO MODEL ROUTINES
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Integrand for the satellite IA profile
// x = r/r_s (dimensionless radius)
// params[0] = k * r_s (the dimensionless wavenumber)
// ---------------------------------------------------------------------------
// Satellite IA Fourier profile gamma_hat(k|M), following Fortuna et al. 2021
// (arXiv:2003.02700), Sect. 4.1 + Appendix C, and Schneider & Bridle 2010.
//
// The satellite intrinsic shape has angular structure sin(theta) e^{2 i phi}
// (Eq. 9), so its Fourier transform is NOT the monopole (j_0). The angular
// integral is dominated by the l=2 multipole (paper truncates at l_max=6,
// evaluated at theta_k = pi/2). We keep the leading l=2 term here.
//
// Radial integrand:  gamma_bar(r) * u_NFW(r|M) * j_2(k r) * r^2 dr
//   - gamma_bar(r) = (r/r_vir)^b   radial alignment strength (G19 fit, b~-2)
//   - u_NFW(r|M)   ~ 1 / [ (r/rs)(1+r/rs)^2 ]   NFW number-density profile
//   - j_2(kr)      l=2 spherical Bessel function
// In terms of x = r/rs (so r = rs x, r_vir = rs c):
//   gamma_bar = (x/c)^b
//   u_NFW    ∝ 1/[x (1+x)^2]
//   r^2 dr   = rs^3 x^2 dx
//   integrand ∝ (x/c)^b * 1/[x(1+x)^2] * j_2(krs x) * x^2 dx
//            = c^{-b} * x^{b+1}/(1+x)^2 * j_2(krs x) dx
// ---------------------------------------------------------------------------

// l=2 spherical Bessel function j_2(u) = (3/u^3 - 1/u) sin u - (3/u^2) cos u
static inline double sph_bessel_j2(double u)
{
    if (u < 1.0e-4) {
        // small-u series: j_2(u) = u^2/15 - u^4/210 + ...
        const double u2 = u*u;
        return u2/15.0 * (1.0 - u2/14.0);
    }
    const double u2 = u*u;
    const double u3 = u2*u;
    return (3.0/u3 - 1.0/u)*sin(u) - (3.0/u2)*cos(u);
}


double int_u_ia_sat(double x, void* params)
{
    const double* ar = (double*) params;
    const double krs     = ar[0];
    const double c       = ar[1];
    const double b       = ar[2];
    const double x_floor = ar[3];

    const double j2 = sph_bessel_j2(krs * x);

    // gamma_bar(r) = (r/r_vir)^b for r > r_floor, held CONSTANT below r_floor
    // (Fortuna 2021: radial power law floored at small radius to avoid the
    //  unphysical central divergence for b<0).
    const double x_eff = (x > x_floor) ? x : x_floor;
    double gamma_bar = pow(x_eff / c, b);
    if (gamma_bar > 0.3) gamma_bar = 0.3;   // Eq. 20: perfect-alignment ceiling

    // NFW number-density profile * r^2 measure:  x/(1+x)^2
    const double nfw_r2 = x / ((1.0 + x)*(1.0 + x));

    return gamma_bar * nfw_r2 * j2;
}
// Satellite IA Fourier profile.
// c = concentration, k = wavenumber, m = halo mass, a = scale factor.
// x = r/rs (dimensionless radius)
// params[0] = k*rs   (dimensionless wavenumber)
// params[1] = c      (concentration, needed for the gamma_bar normalization)
// params[2] = b      (radial power-law slope of gamma_bar; ~ -2 from G19)
// x = r/rs;  params[0]=krs, params[1]=c, params[2]=b, params[3]=x_floor
// x_floor = r_floor/rs, where r_floor ~ 0.06 Mpc/h (Fortuna 2021 / G19)
// ---------------------------------------------------------------------------
// Direct (uncached) evaluation of the satellite IA Fourier profile.
// This is the original implementation, kept as the ground truth that the
// tabulated u_ia_sat() below must reproduce.
//
// PERFORMANCE NOTE (why the table exists):
//   This routine runs a GLFIXED quadrature with DEFAULT_INT_PREC (=1000)
//   nodes. It is called from int_for_IA(), which is ITSELF the integrand of a
//   1000-node mass quadrature -> 1e6 evaluations of int_u_ia_sat (each with a
//   sin, cos and pow) for a SINGLE (k,a,ni). Multiplied over the
//   nbin x (N_a/5) x N_k_nlin table grid this is ~1e10 transcendental calls
//   per table build. That is the "takes forever" the user reported.
//   u_ia_sat_nointerp is therefore wrapped by a 2D interpolation table,
//   exactly the way u_KS() is handled elsewhere in this file.
// ---------------------------------------------------------------------------
double u_ia_sat_nointerp(const double c, const double k, const double m,
                         const double a __attribute__((unused)),
                         const int init)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;
  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    // The integrand is a smooth j2-weighted NFW profile on x in [0,c]; the
    // full DEFAULT_INT_PREC is overkill, but keep it here so this routine
    // remains a strict reference for validating the table.
    const size_t szint = DEFAULT_INT_PREC + 500*Ntable.high_def_integration;
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  const double rho_delta = Delta * cosmology.rho_crit * cosmology.Omega_m;
  const double r_delta   = pow(3.0/(4.0*M_PI) * (m/rho_delta), 1.0/3.0);
  const double rs        = r_delta / c;
  const double krs       = k * rs;

  const double b_slope = -2.0;
  const double r_floor_code = 0.06 / cosmology.coverH0;
  const double x_floor = r_floor_code / rs;

  const double norm = log(1.0 + c) - c/(1.0 + c);   // NFW mass norm

  double ar[4] = {krs, c, b_slope, x_floor};
  gsl_function F;
  F.function = int_u_ia_sat;
  F.params   = (void*) ar;

  if (1 == init) {
    return int_u_ia_sat(c/2.0, (void*) ar);   // touch static vars only
  }

  const double result = gsl_integration_glfixed(&F, 0.0, c, w);

  const double f2_angular = F2_ANGULAR;
  return f2_angular * result / norm;
}

// ---------------------------------------------------------------------------
// Tabulated satellite IA Fourier profile.
//
// CHOICE OF TABLE VARIABLES (this is the subtle part -- read before editing):
//   The integrand int_u_ia_sat depends on (krs, c, b_slope, x_floor), and the
//   integration range is [0, c]. Of these:
//     * b_slope is a hard-coded constant (-2.0)
//     * c       = conc(m, growfac(a))
//     * x_floor = r_floor_code / rs,  rs = r_delta(m)/c
//   so x_floor is NOT a function of c and krs alone -- it carries an extra,
//   independent dependence on the halo mass m through r_delta(m).
//
//   => A naive 2D table in (c, krs) would be WRONG: two different masses can
//      share a (c, krs) pair while having different x_floor.
//
//   We tabulate in (a, ln m, ln k). Fixing (a, m) fixes c(m,a) and x_floor(m)
//   exactly, so there is no hidden dependence left.
//
//   NOTE: a MUST be a table axis, not a cache key. The mass quadrature is
//   evaluated at many different a within a single table build upstream, so
//   keying a 2D (ln m, ln k) table on "current a" would rebuild on nearly
//   every call and be SLOWER than no cache at all.
//
// The 'c' argument is now redundant (it is recomputed internally from (m,a))
// but is retained so the call sites in int_for_IA do not have to change.
// ---------------------------------------------------------------------------
// Table state for u_ia_sat. File-scope (not function-static) so that the
// serial initializer and the read-only accessor can share it.
static double*** uias_table = NULL;
static double uias_lim[3][3];   // [0] = a axis, [1] = ln m axis, [2] = ln k axis
static int uias_na = 0, uias_nm = 0, uias_nk = 0;
static uint64_t uias_cache[MAX_SIZE_ARRAYS];

// Serial initializer for the u_ia_sat table.
//
// THREAD SAFETY -- WHY THIS IS SEPARATE FROM u_ia_sat():
//   u_ia_sat() is called from int_for_IA(), i.e. from inside the mass
//   quadrature, which itself runs inside the "#pragma omp parallel for" of
//   P_II_halo / P_dI_halo. If the table were built lazily on first use:
//     * many threads would simultaneously see table == NULL and each call
//       malloc3d -> leaked buffers and threads writing through a pointer
//       another thread just replaced  => SEGFAULT;
//     * the build itself contains an omp parallel for, so it would be a
//       NESTED parallel region inside an already-parallel loop.
//   Both are exactly the failure mode p_gg/p_gm guard against with their
//   "force interpolation tables to build SERIALLY before the parallel region"
//   comment. So the build lives here and must be called before any omp region.
void u_ia_sat_init(void)
{
  const int na = (int) Ntable.N_a/5.0;
  // GRID SIZES -- do NOT use Ntable.halo_uks_nc / halo_uks_nx here.
  // Those two fields are DECLARED in structs.h but never initialized:
  // reset_Ntable_struct() does not set them and no interface setter exists
  // (u_KS is only reached via the gas/Compton-y path, so nobody noticed).
  // Sizing this table from them means malloc3d() gets uninitialized garbage
  // -> an absurd allocation -> the OOM killer takes the process ("Killed").
  // Ntable.N_M is the halo-model mass-grid size and IS initialized (=1000),
  // so derive from it and from N_k_nlin, both of which have real defaults.
  const int nm = (Ntable.N_M > 0) ? (int)(Ntable.N_M/10) : 100;   // ~100 mass nodes
  const int nk = (Ntable.N_k_nlin > 0) ? (int)(Ntable.N_k_nlin/5) : 100; // ~100 k nodes

  if (nm < 4 || nk < 4 || na < 4) {
    log_fatal("u_ia_sat_init: degenerate grid (na=%d, nm=%d, nk=%d). "
              "Check Ntable.N_a / N_M / N_k_nlin are initialized.", na, nm, nk);
    exit(1);
  }

  const int need_alloc = (NULL == uias_table) || (na != uias_na) ||
                         (nm != uias_nm) || (nk != uias_nk);

  if (need_alloc) {
    if (uias_table != NULL) free(uias_table);
    uias_table = (double***) malloc3d(na, nm, nk);
    uias_na = na; uias_nm = nm; uias_nk = nk;
    uias_lim[0][0] = limits.a_min;
    uias_lim[0][1] = 1.0;
    uias_lim[0][2] = (uias_lim[0][1] - uias_lim[0][0])/((double) na - 1.0);
    uias_lim[1][0] = log(limits.halo_m_min);
    uias_lim[1][1] = log(limits.halo_m_max);
    uias_lim[1][2] = (uias_lim[1][1] - uias_lim[1][0])/((double) nm - 1.0);
    uias_lim[2][0] = log(limits.k_min_cH0);
    uias_lim[2][1] = log(limits.k_max_cH0);
    uias_lim[2][2] = (uias_lim[2][1] - uias_lim[2][0])/((double) nk - 1.0);
  }

  if (!need_alloc &&
      !fdiff2(uias_cache[0], cosmology.random) &&
      !fdiff2(uias_cache[1], Ntable.random)) {
    return;   // table already current
  }

  // Build static/GSL state serially before this function's own parallel region.
  (void) u_ia_sat_nointerp(1.0, 1.0, exp(uias_lim[1][0]), uias_lim[0][0], 1);
  (void) sigma2(exp(uias_lim[1][0]));
  (void) growfac(uias_lim[0][0]);

  #pragma omp parallel for collapse(3) schedule(static,1)
  for (int p=0; p<na; p++) {
    for (int i=0; i<nm; i++) {
      for (int j=0; j<nk; j++) {
        const double ap = uias_lim[0][0] + p*uias_lim[0][2];
        const double mi = exp(uias_lim[1][0] + i*uias_lim[1][2]);
        const double kj = exp(uias_lim[2][0] + j*uias_lim[2][2]);
        const double ci = conc(mi, growfac(ap));
        uias_table[p][i][j] = u_ia_sat_nointerp(ci, kj, mi, ap, 0);
      }
    }
  }
  uias_cache[0] = cosmology.random;
  uias_cache[1] = Ntable.random;
}

// Read-only accessor. Safe to call from inside an omp parallel region PROVIDED
// u_ia_sat_init() has already run. If the table is absent we fall back to the
// direct quadrature rather than building it here -- building lazily inside a
// parallel region is the bug this split exists to prevent.
double u_ia_sat(const double c, const double k, const double m, const double a)
{
  if (NULL == uias_table) {
    return u_ia_sat_nointerp(c, k, m, a, 0);
  }

  const int na = uias_na, nm = uias_nm, nk = uias_nk;
  const double lnm = log(m);
  const double lnk = log(k);

  // Outside the tabulated range fall back to direct evaluation rather than
  // extrapolating: u_ia_sat oscillates (j2) and extrapolation is unsafe.
  if (a   < uias_lim[0][0] || a   > uias_lim[0][1] ||
      lnm < uias_lim[1][0] || lnm > uias_lim[1][1] ||
      lnk < uias_lim[2][0] || lnk > uias_lim[2][1]) {
    return u_ia_sat_nointerp(c, k, m, a, 0);
  }

  // interpol2d works on the (ln m, ln k) plane; interpolate linearly in a
  // between the two bracketing planes.
  const double ra = (a - uias_lim[0][0])/uias_lim[0][2];
  int ia0 = (int) floor(ra);
  if (ia0 < 0) ia0 = 0;
  if (ia0 > na - 2) ia0 = na - 2;
  const double wa = ra - ia0;

  const double u0 = interpol2d(uias_table[ia0],
      nm, uias_lim[1][0], uias_lim[1][1], uias_lim[1][2], lnm,
      nk, uias_lim[2][0], uias_lim[2][1], uias_lim[2][2], lnk);
  const double u1 = interpol2d(uias_table[ia0+1],
      nm, uias_lim[1][0], uias_lim[1][1], uias_lim[1][2], lnm,
      nk, uias_lim[2][0], uias_lim[2][1], uias_lim[2][2], lnk);

  return u0 + wa*(u1 - u0);
}

//debug

// Standalone test accessor for the satellite IA Fourier profile gamma_hat(k|M).
// Computes concentration internally so it can be called with just (k, m, a).
// Use to validate the single-halo profile shape against Fortuna 2021 Fig. C1.
double test_u_ia_sat(const double k, const double m, const double a)
{
  const double growfac_a = growfac(a);
  const double c = conc(m, growfac_a);
  return u_ia_sat(c, k, m, a);
}

double int_for_IA(double lnM, void* params)
{
    // ... unpack params: a, k, ni, growfac_a, ia_func
    // ... compute nu, dNdlnM, c exactly as in int_for_I02_XY
    double* ar      = (double*) params;
    const double a          = ar[0];
    const double k          = ar[1];
    const int    ni         = (int) ar[2];
    const int    ia_func    = (int) ar[3];
    const double growfac_a  = ar[4];
    const double m          = exp(lnM);

    const double nu     = delta_c / (sqrt(sigma2(m)) * growfac_a);
    const double gnu    = fnu(nu, a) * nu;
    const double rhom   = cosmology.rho_crit * cosmology.Omega_m;
    const double dNdlnM = gnu * (rhom / m) * dlognudlogm(m);
    const double c      = conc(m, growfac_a);
    
    const double nc      = HOD_nc(m, a, ni);
    const double ns      = HOD_ns(m, a, ni);
    const double fc      = HOD_fc(ni);
    const double fred_c  = f_red_cen(m, ni);
    const double fred_s  = f_red_sat(m, ni);
    
    // Red central number density contribution
    const double nc_red  = fc * nc * fred_c;
    // Red satellite number density contribution  
    const double ns_red  = ns * fred_s;
    
    // u_IA: the radial alignment profile for satellites
    // This is the NFW profile derivative (Schneider & Bridle 2010, Eq. 13)
    //const double u_ia = u_ia_sat(c, k, m, a); // new function to implement
    
    switch(ia_func) {
        case 0: // n_red_cen
            return dNdlnM * nc_red;
        case 1: // n_red_sat
            return dNdlnM * ns_red;

        case 2: // 1-halo II: sat-sat
            {
              const double u_ia = u_ia_sat(c, k, m, a);
            
              return dNdlnM * ns_red * ns_red * u_ia * u_ia;
            }
          
        case 3: // 1-halo dI: matter-sat
        {
          const double u_ia = u_ia_sat(c, k, m, a);
          return dNdlnM * (m/rhom) * u_c(c,k,m,a) * ns_red * fabs(u_ia);
        }
        
        default:
        {
          log_fatal("ia_func = %d not supported", ia_func);
          exit(1);
        }
    }
}

double int_hm_funcs(double lnM, void* params)
{ // 0 = ngal, 1 = m_mean, 2 = fsat, 3 = bgal 
  double* ar = (double*) params;
  
  const double a = ar[0];
  const int ni = (int) ar[1];
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  const int func = (int) ar[2];
  const double growfac_a = (double) ar[3];
  const double m = exp(lnM);
  
  const double nu = delta_c/(sqrt(sigma2(m))*growfac_a);
  const double gnu  = fnu(nu, a) * nu; 
  const double rhom = cosmology.rho_crit * cosmology.Omega_m;
  const double dNdlnM = gnu * (rhom/m) * dlognudlogm(m);
  
  const double nc = HOD_fc(ni)*HOD_nc(m, a, ni);
  const double ns = HOD_ns(m, a, ni);
  const double frc = f_red_cen(m, ni);
  const double frs = f_red_sat(m, ni);
  



  double res;
  switch(func)
  {
    case 0:
    { // N_gal = \int dM n(M)*(nc + ns) = \int dlnM M n(M)*(nc + ns) 
      res = dNdlnM*(nc + ns);
      break;
    }
    case 1:
    { // <M> = \int dM M*n(M)*(nc + ns) = \int dlnM M^2 n(M)*(nc + ns) 
      res = m*(dNdlnM*(nc + ns));
      break;
    }
    case 2:
    {
      res = dNdlnM*ns;
      break;
    }
    case 3:
    {
      res = hb1nu(nu, a)*(dNdlnM*(nc + ns));
      break;
    }
    case 4:
    {
      res = dNdlnM * nc * frc;
      break;
    }
    case 5:
    {
      res = dNdlnM * ns * frs;
      break;
    }
    case 6:
    {
      res = dNdlnM * nc * (1 - frc);
      break;
    }
    case 7:
    {
      res = dNdlnM * ns * (1 - frs);
      break;
    }

    default:
    {
      log_fatal("option not supported");
      exit(1);
    }
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double ngal_nointerp(
    const int ni, 
    const double a, 
    const int init
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = DEFAULT_INT_PREC + 500*Ntable.high_def_integration;
    if (w != NULL)  gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[4] = {a, (double) ni, (double) 0, growfac(a)};
  const double lnMmin = log(10.0)*(nuisance.hod[ni][0] - 2.);
  const double lnMmax = log(limits.halo_m_max);

  double res = 0.0;
  if (1 == init) {
    res = int_hm_funcs((lnMmin + lnMmax)/2.0, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_hm_funcs;
    res = gsl_integration_glfixed(&F, lnMmin, lnMmax, w);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double ngal(const int ni, const double a)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double lim[3]; // [0] = amin; [1] = amax; [2] = da

  if (table == NULL || 
      fdiff2(cache[1], Ntable.random) ||
      fdiff2(cache[3], redshift.random_clustering)) 
  { 
    if (table != NULL) free(table);
    table = (double**) malloc2d(redshift.clustering_nbin, Ntable.N_a);

    lim[0] = 1.0/(redshift.clustering_zdist_zmax_all + 1.0);
    lim[1] = 1.0/(redshift.clustering_zdist_zmin_all + 1.0);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_a - 1.0);
  }
  
  if (fdiff2(cache[0], cosmology.random) || 
      fdiff2(cache[1], Ntable.random)    ||
      fdiff2(cache[2], nuisance.random_galaxy_bias) ||
      fdiff2(cache[3], redshift.random_clustering))
  {
    (void) ngal_nointerp(0, lim[0], 1);    
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<redshift.clustering_nbin; i++) {
      for (int j=0; j<Ntable.N_a; j++) {
        table[i][j] = ngal_nointerp(i, lim[0] + j*lim[2], 0);
      }
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_galaxy_bias;
    cache[3] = redshift.random_clustering;
  }
  return ((a < lim[0]) || (a > lim[1]))? 0.0 :
    interpol1d(table[ni], Ntable.N_a, lim[0], lim[1], lim[2], a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double hm_funcs_nointerp(
    const int ni, 
    const double a, 
    const int func,
    const int init
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  if (w == NULL || fdiff2(cache[0], Ntable.random))
  {
    const size_t szint = DEFAULT_INT_PREC + 500*Ntable.high_def_integration;
    if (w != NULL)  gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }
  double ar[4] = {a, (double) ni, (double) func, growfac(a)}; 
  const double lnMmin = log(10.0)*(nuisance.hod[ni][0] - 2.);
  const double lnMmax = log(limits.halo_m_max);
  double res = 0.0;
  if (init == 1)
    res = int_hm_funcs((lnMmin + lnMmax)/2.0, (void*) ar);
  else
  {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_hm_funcs;
    res = gsl_integration_glfixed(&F, lnMmin, lnMmax, w);
  }
  // func=1 (<M>/n_gal) and func=2 (f_sat = n_sat/n_gal) are normalized by n_gal.
  // func=0 (n_gal), func=3 (bias-weighted density), and func=4-7 (red/blue 
  // sub-population densities) are raw integrals and must NOT be divided by n_gal.
  switch(func)
  {
    case 1: // mean halo mass per galaxy
    case 2: // satellite fraction
    case 3: // effective galaxy bias = <b1 * n_g> / n_gal
      return res / ngal(ni, a);
    default: // all other cases: raw number densities
      return res;
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double mmean_nointerp(
    const int ni, 
    const double a, 
    const int init
  )
{
  return hm_funcs_nointerp(ni, a, 1, init);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double fsat_nointerp(
    const int ni, 
    const double a, 
    const int init
  )
{
  return hm_funcs_nointerp(ni, a, 2, init);
} 

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double bgal_nointerp(
    const int ni, 
    const double a, 
    const int init
  )
{
  return hm_funcs_nointerp(ni, a, 3, init);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double bgal(const int ni, const double a)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double lim[3]; // [0] = amin; [1] = amax; [2] = da

  if (NULL == table || 
      fdiff2(cache[1], Ntable.random) ||
      fdiff2(cache[3], redshift.random_clustering)) 
  {  
    if (table != NULL) free(table); 
    table = (double**) malloc2d(redshift.clustering_nbin, Ntable.N_a);
    lim[0] = 1.0/(redshift.clustering_zdist_zmax_all + 1.0);
    lim[1] = 1.0/(redshift.clustering_zdist_zmin_all + 1.0);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_a - 1.0);
  }
  if (fdiff2(cache[0], cosmology.random) || 
      fdiff2(cache[1], Ntable.random)    ||
      fdiff2(cache[2], nuisance.random_galaxy_bias) ||
      fdiff2(cache[3], redshift.random_clustering)) 
  {
    (void) bgal_nointerp(0, lim[0], 1); // init static vars  
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<redshift.clustering_nbin; i++) {
      for (int j=0; j<Ntable.N_a; j++) {
        table[i][j] = bgal_nointerp(i, lim[0] + j*lim[2], 0);
      }
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_galaxy_bias;
    cache[3] = redshift.random_clustering;
  }  
  return (a < lim[0]) || (a > lim[1]) ? 0.0 : 
    interpol1d(table[ni], Ntable.N_a, lim[0], lim[1], lim[2], a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_I02_XY(double lnM, void* params) 
{
  double* ar = (double*) params;
  const double a = ar[0];
  const double k1 = ar[1];
  const double k2 = ar[2];
  const int XY = (int) ar[3];
  const double growfac_a = ar[4];
  const double m = exp(lnM);
  
  const double nu = delta_c/(sqrt(sigma2(m))*growfac_a);
  const double gnu  = fnu(nu, a) * nu; 
  const double rhom = cosmology.rho_crit * cosmology.Omega_m;
  const double dNdlnM = gnu * (rhom/m) * dlognudlogm(m); // mass function

  const double c = conc(m, growfac_a);

  double u;
  switch(XY)
  {
    case 0:
    { // matter-matter
      u = u_c(c, k1, m, a) * u_c(c, k2, m, a);
      break;
    }
    case 1:
    { // matter-y
      u = u_y_bnd(c, k1, m, a) * u_c(c, k2, m, a);
      break;
    }
    case 2:
    { // y-y 
      u = u_y_bnd(c, k1, m, a) * u_y_bnd(c, k2, m, a);
      break;
    }
    default:
    {
      log_fatal("option not supported"); exit(1);
    }
  }
  return dNdlnM * u * (m/rhom) * (m/rhom);
}  

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double I02_XY_nointerp(
    const double k1, 
    const double k2, 
    const double a,
    const int func, // 1 = MM, 2 = MY, 3 = YY
    const int init
  ) 
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = DEFAULT_INT_PREC + 500*Ntable.high_def_integration;
    if (w != NULL)  gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[5] = {a, k1, k2, func, growfac(a)};
  const double lnMmin = log(limits.halo_m_min);
  const double lnMmax = log(limits.halo_m_max);

  double res;
  if (1 == init) {
    res = int_for_I02_XY((lnMmin + lnMmax)/2.0, (void*) ar);
  }
  else
  {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_I02_XY;
    res = gsl_integration_glfixed(&F, lnMmin, lnMmax, w);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_I11_X(double lnM, void* params) 
{
  const double* ar = (double*) params;  
  const double a = ar[0];
  const double k = ar[1];
  const int func = (int) ar[2];
  const double growfac_a = ar[3];
  const double m = exp(lnM);
  
  const double nu = delta_c/(sqrt(sigma2(m))*growfac_a);
  const double gnu = fnu(nu, a) * nu; 
  const double rhom = cosmology.rho_crit * cosmology.Omega_m;
  const double dNdlnM = gnu * (rhom/m) * dlognudlogm(m);

  const double c = conc(m, growfac_a);

  double u;
  switch(func)
  {
    case 0:
    { // matter
      u = u_c(c, k, m, a);
      break;
    }
    case 1:
    { // y
      u = u_y_bnd(c, k, m, a) + u_y_ejc(m);
      break;
    }
    default:
    {
      log_fatal("option not supported"); exit(1);
    }
  }
  return dNdlnM * u * (m/rhom) * hb1nu(nu, a)/bias_norm(a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double I11_X_nointerp(
    const double k, 
    const double a,
    const int func, 
    const int init
  ) 
{ 
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = DEFAULT_INT_PREC + 500*Ntable.high_def_integration;
    if (w != NULL)  gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[4] = {a, k, func, growfac(a)};
  const double lnMmin = log(limits.halo_m_min);
  const double lnMmax = log(limits.halo_m_max);
  
  double res;
  if (1 == init) {
    res = int_for_I11_X((lnMmin + lnMmax)/2.0, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_I11_X;
    res = gsl_integration_glfixed(&F, lnMmin, lnMmax, w);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_G02(double lnM, void* param)
{
  double* ar = (double*) param;
  
  const double k = ar[0];
  const double a = ar[1];
  const int ni = (int) ar[2];
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  const double growfac_a = ar[3];
  const double m  = exp(lnM);

  const double nu = delta_c/(sqrt(sigma2(m))*growfac_a);
  const double gnu = fnu(nu, a) * nu; 
  const double rhom = cosmology.rho_crit * cosmology.Omega_m;
  const double dNdlnM = gnu * (rhom/m) * dlognudlogm(m);

  const double c  = conc(m, growfac_a);
  const double u  = u_g(c, k, m, a, ni);
  const double ns = HOD_ns(m, a, ni);
  const double nc = HOD_nc(m, a, ni);
  const double fc = HOD_fc(ni);
  

  return dNdlnM*(u*u*ns*ns + 2.0*u*ns*nc*fc);
}

// ---------------------------------------------------------------------------
// IA POWER SPECTRUM ASSEMBLY (1-halo II, satellite-satellite)
// ---------------------------------------------------------------------------

// Mass integral of int_for_IA for a given ia_func case.
// Returns the raw integral (no n_bar normalization, no amplitude).
double I_for_IA_nointerp(
    const double k,
    const double a,
    const int ni,
    const int ia_func,
    const int init
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = DEFAULT_INT_PREC + 500*Ntable.high_def_integration;
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[5] = {a, k, (double) ni, (double) ia_func, growfac(a)};
  const double lnMmin = log(10.0)*(nuisance.hod[ni][0] - 2.);
  const double lnMmax = log(limits.halo_m_max);

  double res;
  if (1 == init) {
    res = int_for_IA((lnMmin + lnMmax)/2.0, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_IA;
    res = gsl_integration_glfixed(&F, lnMmin, lnMmax, w);
  }
  return res;
}

// n_bar for red satellites: ∫ dlnM (dN/dlnM) n_sat_red(M)
// This is exactly int_for_IA case 1 integrated over mass.
//
// PERFORMANCE: this quantity is k-INDEPENDENT, but p_II_1h/p_dI_1h need it at
// every k node of the P_II/P_dI table build (~N_k_nlin times per (a,ni)).
// Recomputing a 1000-node mass quadrature each time is pure waste, so it is
// precomputed into a table indexed by (ni, a-node).
//
// THREAD SAFETY (this is why the code looks the way it does):
//   This routine is called from P_II_halo_nointerp / P_dI_halo_nointerp, which
//   run INSIDE "#pragma omp parallel for" in P_II_halo / P_dI_halo. A lazily
//   populated cache would be a data race: several threads would simultaneously
//   see an empty slot, all run the quadrature, and race on the write -- and a
//   lazy malloc would be worse still (one thread can free the buffer another
//   is writing through). That is a segfault, not a wrong number.
//
//   So: the table is allocated AND fully populated by n_red_sat_bar_init(),
//   which the P_II_halo / P_dI_halo builders call SERIALLY before entering
//   their parallel regions -- the same discipline p_gg/p_gm already use when
//   they pre-touch bgal/ngal/Pdelta. Inside the parallel region this function
//   is then strictly READ-ONLY, which is safe.
// ---------------------------------------------------------------------------

static double* nrsb_val  = NULL;   // [nbin*na]
static int     nrsb_na   = 0;
static int     nrsb_nbin = 0;
static uint64_t nrsb_cache[MAX_SIZE_ARRAYS];

// Serial initializer: allocate + fill. MUST be called outside any omp region.
void n_red_sat_bar_init(void)
{
  const int na   = (int) Ntable.N_a/5.0;
  const int nbin = redshift.clustering_nbin;

  const int need_alloc = (NULL == nrsb_val) || (na != nrsb_na) ||
                         (nbin != nrsb_nbin);
  const int need_fill = need_alloc ||
      fdiff2(nrsb_cache[0], cosmology.random) ||
      fdiff2(nrsb_cache[1], Ntable.random)    ||
      fdiff2(nrsb_cache[2], nuisance.random_ia) ||
      fdiff2(nrsb_cache[3], redshift.random_clustering);

  if (!need_fill) return;

  if (need_alloc) {
    if (nrsb_val != NULL) free(nrsb_val);
    nrsb_val  = (double*) malloc1d(nbin*na);
    nrsb_na   = na;
    nrsb_nbin = nbin;
  }

  (void) I_for_IA_nointerp(0.0, amin_lens(0), 0, 1, 1); // init static vars

  for (int l=0; l<nbin; l++) {
    const double amin = amin_lens(l);
    const double amax = amax_lens(l);
    const double da = (amax - amin)/((double) na - 1.0);
    for (int i=0; i<na; i++) {
      const double a = amin + i*da;
      nrsb_val[l*na + i] = I_for_IA_nointerp(0.0, a, l, 1, 0);
    }
  }

  nrsb_cache[0] = cosmology.random;
  nrsb_cache[1] = Ntable.random;
  nrsb_cache[2] = nuisance.random_ia;
  nrsb_cache[3] = redshift.random_clustering;
}

double n_red_sat_bar(const int ni, const double a)
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  // If the table is not ready we are being called from outside the intended
  // build path (e.g. a direct _nointerp call from a test). Fall back to the
  // direct quadrature rather than lazily filling a shared buffer, which would
  // race if this happened inside an omp region.
  if (NULL == nrsb_val || nrsb_na <= 0) {
    return I_for_IA_nointerp(0.0, a, ni, 1, 0);
  }

  const int na = nrsb_na;
  const double amin = amin_lens(ni);
  const double amax = amax_lens(ni);
  const double da = (amax - amin)/((double) na - 1.0);
  if (!(da > 0)) {
    return I_for_IA_nointerp(0.0, a, ni, 1, 0);
  }
  // Linear interpolation in a between the bracketing nodes.
  double r = (a - amin)/da;
  if (r < 0.0 || r > (double)(na - 1)) {           // outside tabulated range
    return I_for_IA_nointerp(0.0, a, ni, 1, 0);
  }
  int i0 = (int) floor(r);
  if (i0 > na - 2) i0 = na - 2;
  if (i0 < 0) i0 = 0;
  const double w = r - i0;
  const double v0 = nrsb_val[ni*na + i0];
  const double v1 = nrsb_val[ni*na + i0 + 1];
  return v0 + w*(v1 - v0);
}

// 1-halo II satellite power spectrum, normalized and amplitude-weighted.
// P_II^1h(k,a,ni) = A^2 * [ ∫dlnM dN/dlnM n_sat_red^2 u_ia^2 ] / n_bar^2
double p_II_1h_nointerp(const double k, const double a, const int ni)
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  const double nbar = n_red_sat_bar(ni, a);
  if (!(nbar > 0)) {
    return 0.0;
  }
  // Constant alignment amplitude stored by set_nuisance_ia_halo in ia[3][ni].
  // NOTE: ia[3] is indexed per *source* bin in set_nuisance_ia_halo, but here
  // ni is a *lens* bin. See the amplitude caveat in the notes below.
  const double A = nuisance.ia[5][ni];;//nuisance.ia[3][ni];

  const double I_II = I_for_IA_nointerp(k, a, ni, 2, 0);

  return (A*A) * I_II / (nbar*nbar);
}
// ---------------------------------------------------------------------------
// 2-HALO CENTRAL IA (NLA limit)
// ---------------------------------------------------------------------------

// Integrand for red-central bias-weighted density.
// func: 0 = bias-weighted numerator  ∫ dN/dlnM b1(M) n_cen_red(M)
//       1 = density normalization     ∫ dN/dlnM n_cen_red(M)
double int_for_bred_cen(double lnM, void* params)
{
  double* ar = (double*) params;
  const double a         = ar[0];
  const int    ni        = (int) ar[1];
  const int    func      = (int) ar[2];
  const double growfac_a = ar[3];
  const double m         = exp(lnM);

  const double nu     = delta_c/(sqrt(sigma2(m))*growfac_a);
  const double gnu    = fnu(nu, a)*nu;
  const double rhom   = cosmology.rho_crit*cosmology.Omega_m;
  const double dNdlnM = gnu*(rhom/m)*dlognudlogm(m);

  const double nc      = HOD_nc(m, a, ni);
  const double fc      = HOD_fc(ni);
  const double fred_c  = f_red_cen(m, ni);
  const double nc_red  = fc * nc * fred_c;

  if (func == 0) {
    return dNdlnM * hb1nu(nu, a) * nc_red;   // bias-weighted
  } else {
    return dNdlnM * nc_red;                    // density (normalization)
  }
}

double I_bred_cen_nointerp(const double a, const int ni, const int func,
                           const int init)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = DEFAULT_INT_PREC + 500*Ntable.high_def_integration;
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }
  double ar[4] = {a, (double) ni, (double) func, growfac(a)};
  const double lnMmin = log(10.0)*(nuisance.hod[ni][0] - 2.);
  const double lnMmax = log(limits.halo_m_max);
  double res;
  if (1 == init) {
    res = int_for_bred_cen((lnMmin+lnMmax)/2.0, (void*) ar);
  } else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_bred_cen;
    res = gsl_integration_glfixed(&F, lnMmin, lnMmax, w);
  }
  return res;
}

// Effective linear bias of the red central population in lens bin ni.
//
// PERFORMANCE + THREAD SAFETY: identical situation to n_red_sat_bar above --
// k-independent, called from inside an omp parallel region, each call would
// otherwise run TWO 1000-node mass quadratures. Precomputed serially by
// b_red_cen_init(); read-only thereafter. See the long comment on
// n_red_sat_bar for why a lazily-filled cache here is a segfault, not just a
// performance detail.
// ---------------------------------------------------------------------------

static double* brc_val  = NULL;   // [nbin*na]
static int     brc_na   = 0;
static int     brc_nbin = 0;
static uint64_t brc_cache[MAX_SIZE_ARRAYS];

static double b_red_cen_direct(const int ni, const double a)
{
  const double num = I_bred_cen_nointerp(a, ni, 0, 0);
  const double den = I_bred_cen_nointerp(a, ni, 1, 0);
  return (den > 0) ? num/den : 0.0;
}

// Serial initializer: allocate + fill. MUST be called outside any omp region.
void b_red_cen_init(void)
{
  const int na   = (int) Ntable.N_a/5.0;
  const int nbin = redshift.clustering_nbin;

  const int need_alloc = (NULL == brc_val) || (na != brc_na) || (nbin != brc_nbin);
  const int need_fill = need_alloc ||
      fdiff2(brc_cache[0], cosmology.random) ||
      fdiff2(brc_cache[1], Ntable.random)    ||
      fdiff2(brc_cache[2], nuisance.random_ia) ||
      fdiff2(brc_cache[3], redshift.random_clustering);

  if (!need_fill) return;

  if (need_alloc) {
    if (brc_val != NULL) free(brc_val);
    brc_val  = (double*) malloc1d(nbin*na);
    brc_na   = na;
    brc_nbin = nbin;
  }

  (void) I_bred_cen_nointerp(amin_lens(0), 0, 0, 1); // init static vars
  (void) I_bred_cen_nointerp(amin_lens(0), 0, 1, 1);

  for (int l=0; l<nbin; l++) {
    const double amin = amin_lens(l);
    const double amax = amax_lens(l);
    const double da = (amax - amin)/((double) na - 1.0);
    for (int i=0; i<na; i++) {
      brc_val[l*na + i] = b_red_cen_direct(l, amin + i*da);
    }
  }

  brc_cache[0] = cosmology.random;
  brc_cache[1] = Ntable.random;
  brc_cache[2] = nuisance.random_ia;
  brc_cache[3] = redshift.random_clustering;
}

double b_red_cen(const int ni, const double a)
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  if (NULL == brc_val || brc_na <= 0) {
    return b_red_cen_direct(ni, a);
  }

  const int na = brc_na;
  const double amin = amin_lens(ni);
  const double amax = amax_lens(ni);
  const double da = (amax - amin)/((double) na - 1.0);
  if (!(da > 0)) {
    return b_red_cen_direct(ni, a);
  }
  double r = (a - amin)/da;
  if (r < 0.0 || r > (double)(na - 1)) {
    return b_red_cen_direct(ni, a);
  }
  int i0 = (int) floor(r);
  if (i0 > na - 2) i0 = na - 2;
  if (i0 < 0) i0 = 0;
  const double w = r - i0;
  const double v0 = brc_val[ni*na + i0];
  const double v1 = brc_val[ni*na + i0 + 1];
  return v0 + w*(v1 - v0);
}

// NLA amplitude for centrals (standard cosmolike form):
//   A_NLA(a) = -A_IA * C1 * rho_crit * Omega_m / D(a) * ((1+z)/(1+z0))^eta
// A_IA and eta read from the lens-bin slots ia[6], ia[7]
// (set by set_nuisance_halo_model).
double A_nla_cen(const int ni, const double a)
{
  const double z    = 1.0/a - 1.0;
  const double D     = growfac(a);
  const double A_IA  = nuisance.ia[6][ni];
  const double eta   = nuisance.ia[7][ni];
  const double z0    = (nuisance.oneplusz0_ia > 0) ? nuisance.oneplusz0_ia - 1.0 : 0.62;

  const double amp = -A_IA * nuisance.c1rhocrit_ia * cosmology.Omega_m / D;
  const double zev = pow((1.0 + z)/(1.0 + z0), eta);
  return amp * zev;
}

// ---------------------------------------------------------------------------
// MATTER SPECTRUM USED IN THE 2-HALO CENTRAL IA TERM
// ---------------------------------------------------------------------------
// The NLA 2-halo central term is  (A*b)^n * P_mm(k,a). Two choices for the
// matter spectrum P_mm:
//
//   HM_IA_2H_PMM == 0 : Pdelta(k,a)  -- the pipeline's nonlinear matter power
//                       (CAMB/Halofit or EE2, whatever set_cosmology loaded).
//                       This is the standard NLA choice used everywhere else
//                       in cosmolike and matches Fortuna et al. (2020) Eq. 1
//                       (NLA replaces P_lin with the nonlinear P).
//
//   HM_IA_2H_PMM == 1 : p_mm(k,a)    -- the halo model's OWN matter power
//                       (1-halo + 2-halo built from I02_XY / I11_X). This makes
//                       the IA 2-halo term consistent with the same halo-model
//                       matter field the rest of the IA halo terms live in,
//                       rather than mixing a halofit/EE2 P with halo-model IA.
//
// UNITS: p_mm and Pdelta share the exact same convention -- k in c/H0 units,
// P in (c/H0)^3 -- so this is a drop-in swap with no rescaling.
//
// CAVEAT worth knowing before trusting results: the halo-model p_mm is NOT
// identical to halofit/EE2. It is typically accurate at the ~10-20% level and
// tends to under-predict power in the mildly nonlinear "trough" around
// k ~ 0.1-1 h/Mpc where neither the 1h nor 2h term is complete. So switching
// to p_mm makes the IA term self-consistent with the halo model, but it does
// NOT make it more accurate in an absolute sense against N-body. Which one you
// want depends on the goal: self-consistency (p_mm) vs. best matter power
// (Pdelta). Validate by comparing the two data vectors on a fixed cosmology.
// ---------------------------------------------------------------------------
#ifndef HM_IA_2H_PMM
#define HM_IA_2H_PMM 1   // 0 = Pdelta (pipeline nonlinear P), 1 = p_mm (halo model)
#endif

static inline double p_mm_2h_ia(const double k, const double a)
{
#if HM_IA_2H_PMM == 1
  return p_mm(k, a);
#else
  return Pdelta(k, a);
#endif
}

// 2-halo central II (intrinsic-intrinsic) power spectrum, NLA limit.
double p_II_2h_cen_nointerp(const double k, const double a, const int ni)
{
  const double A = A_nla_cen(ni, a);
  const double b = b_red_cen(ni, a);
  return (A*b)*(A*b) * p_mm_2h_ia(k, a);
}

// 2-halo central dI (density-intrinsic / matter-IA) power spectrum, NLA limit.
double p_dI_2h_cen_nointerp(const double k, const double a, const int ni)
{
  const double A = A_nla_cen(ni, a);
  const double b = b_red_cen(ni, a);
  return (A*b) * p_mm_2h_ia(k, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double G02_nointerp(
    double k, 
    double a, 
    int ni, 
    const int init
  )
{ //needs to be divided by ngal^2
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = DEFAULT_INT_PREC + 500*Ntable.high_def_integration;
    if (w != NULL)  gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[4] = {k, a, (double) ni, growfac(a)};
  const double lnMmin = log(limits.halo_m_min);
  const double lnMmax = log(limits.halo_m_max);

  double res;
  if (1 == init) {
    res = int_for_G02((lnMmin + lnMmax)/2.0, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_G02;
    res = gsl_integration_glfixed(&F, lnMmin, lnMmax, w);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_GM02(double lnM, void* params)
{ // 1-halo galaxy-matter spectrum
  double* ar = (double*) params;
  
  const double k = ar[0];
  const double a = ar[1];
  const int ni = (int) ar[2];
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  const double growfac_a = ar[3];
  const double m = exp(lnM);

  const double nu = delta_c/(sqrt(sigma2(m))*growfac_a);
  const double gnu  = fnu(nu, a) * nu; 
  const double rhom = cosmology.rho_crit * cosmology.Omega_m;
  const double dNdlnM = gnu * (rhom/m) * dlognudlogm(m);

  const double c = conc(m, growfac_a);
  const double ns = HOD_ns(m, a, ni);
  const double nc = HOD_nc(m, a, ni);
  const double fc = HOD_fc(ni);

  return dNdlnM*(m/rhom)*u_c(c,k,m,a)*(u_g(c,k,m,a,ni)*ns + nc*fc);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double GM02_nointerp(
    double k, 
    double a, 
    int ni, 
    const int init
  )
{ // needs to be divided by ngal
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }

  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = DEFAULT_INT_PREC + 500*Ntable.high_def_integration;
    if (w != NULL)  gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[4] = {k, a, (double) ni, growfac(a)};
  const double lnMmin = log(10.)*(nuisance.hod[ni][0] - 1.0);
  const double lnMmax = log(limits.halo_m_max);

  double res;
  if (1 == init) {
    res = int_GM02((lnMmin + lnMmax)/2.0, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_GM02;
    res = gsl_integration_glfixed(&F, lnMmin, lnMmax, w);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HALO MODEL POWER SPECTRA
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double p_xy_nointerp(
    const double k, 
    const double a,
    const int func,
    const int init
  ) 
{
  const double I02 = I02_XY_nointerp(k, k, a, func, init);

  double P1H, I11X, I11Y;

  switch(func)
  {
    case 0:
    { // PMM
      P1H  = I02;
      I11X = I11_X_nointerp(k, a, func, init);
      I11Y = I11X;
      break;
    }
    case 1:
    { // PMY
      // convert to code unit, Table 2, 2009.01858
      const double ks = 0.05618/pow(cosmology.sigma_8*a,1.013)*cosmology.coverH0; 
      const double x = ks*ks*ks*ks;
      P1H  = I02*(1.0/(x + 1.0)); // suppress lowk (Eq17;2009.01858)
      I11X = I11_X_nointerp(k, a, 0, init);
      I11Y = I11_X_nointerp(k, a, 2, init);
      break;
    }
    case 2:
    { // PYY
      // convert to code unit, Table 2, 2009.01858
      const double ks = 0.05618/pow(cosmology.sigma_8*a,1.013)*cosmology.coverH0; 
      const double x = ks*ks*ks*ks;
      P1H  = I02*(1.0/(x + 1.0)); // suppress lowk (Eq17;2009.01858)
      I11X = I11_X_nointerp(k, a, func, init);
      I11Y = I11X;
      break;
    }
    default:
    {
      log_fatal("option not supported");
      exit(1);
    }
  }

  return P1H + (I11X * I11Y * p_lin(k, a));;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double p_mm(
    const double k, 
    const double a
  )
{ 
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double lim[2][3]; // lim[0][0] = amin, lim[0][1] = amax, lim[0][2] = da 
                           // lim[1][0] = lnkmin, lim[1][1] = lnkmax, lim[1][2] = dlnk

  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double**) malloc2d(Ntable.N_a, Ntable.N_k_nlin);   
    lim[0][0] = limits.a_min;
    lim[0][1] = 0.9999999;
    lim[0][2] = (lim[0][1] - lim[0][0]) / ((double) Ntable.N_a - 1.0);
    lim[1][0] = log(limits.k_min_cH0);
    lim[1][1] = log(limits.k_max_cH0);
    lim[1][2] = (lim[1][1] - lim[1][0]) / ((double) Ntable.N_k_nlin - 1.0);
  }
  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random)) {
    (void) p_xy_nointerp(exp(lim[1][0]), lim[0][0], 0, 1); 
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=1; i<Ntable.N_a; i++) {
      for (int j=0; j<Ntable.N_k_nlin; j++) { 
        table[i][j] = log(p_xy_nointerp(exp(lim[1][0] + j*lim[1][2]), 
                                        lim[0][0] + i*lim[0][2], 0, 0));
      }
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
  return exp(interpol2d(table, 
                        Ntable.N_a, lim[0][0], lim[0][1], lim[0][2], a, 
                        Ntable.N_k_nlin, lim[1][0], lim[1][1], lim[1][2], log(k)));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double p_my(
    const double k, 
    const double a
  )
{ 
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = 0;
  static double lim[2][3]; // lim[0][0] = amin, lim[0][1] = amax, lim[0][2] = da 
                           // lim[1][0] = lnkmin, lim[1][1] = lnkmax, lim[1][2] = dlnk

  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double**) malloc2d(Ntable.N_a, Ntable.N_k_nlin); 
    lim[0][0] = limits.a_min;
    lim[0][1] = 0.9999999;
    lim[0][2] = (lim[0][1] - lim[0][0]) / ((double) Ntable.N_a - 1.0);
    lim[1][0] = log(limits.k_min_cH0);
    lim[1][1] = log(limits.k_max_cH0);
    lim[1][2] = (lim[1][1] - lim[1][0]) / ((double) Ntable.N_k_nlin - 1.0);
  }
  if (fdiff2(cache[0], cosmology.random) || 
      fdiff2(cache[1], Ntable.random) ||
      fdiff2(cache[2], nuisance.random_gas))
  {
    (void) p_xy_nointerp(exp(lim[1][0]), lim[0][0], 1, 1); // init static vars
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=1; i<Ntable.N_a; i++) {
      for (int j=0; j<Ntable.N_k_nlin; j++) {
        table[i][j] = log(p_xy_nointerp(exp(lim[1][0] + j*lim[1][2]), 
                                            lim[0][0] + i*lim[0][2], 1, 0));
      }
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_gas;
  }
  return exp(interpol2d(table, 
                        Ntable.N_a, lim[0][0], lim[0][1], lim[0][2], a, 
                        Ntable.N_k_nlin, lim[1][0], lim[1][1], lim[1][2], log(k)));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double p_yy(
    const double k, 
    const double a
  )
{ 
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = 0;
  static double lim[2][3]; // lim[0][0]=amin, lim[0][1]=amax, lim[0][2]=da 
                           // lim[1][0]=lnkmin, lim[1][1]=lnkmax, lim[1][2]=dlnk

  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double**) malloc2d(Ntable.N_a, Ntable.N_k_nlin);
    lim[0][0] = limits.a_min;
    lim[0][1] = 0.9999999;
    lim[0][2] = (lim[0][1] - lim[0][0]) / ((double) Ntable.N_a - 1.0);
    lim[1][0] = log(limits.k_min_cH0);
    lim[1][1] = log(limits.k_max_cH0);
    lim[1][2] = (lim[1][1] - lim[1][0]) / ((double) Ntable.N_k_nlin - 1.0);
  }
  if (fdiff2(cache[0], cosmology.random) || 
      fdiff2(cache[1], Ntable.random) ||
      fdiff2(cache[2], nuisance.random_gas))
  { 
    (void) p_xy_nointerp(exp(lim[1][0]), lim[0][0], 2, 1); // init static vars
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=1; i<Ntable.N_a; i++) {
      for (int j=0; j<Ntable.N_k_nlin; j++) {
        table[i][j] = log(p_xy_nointerp(exp(lim[1][0] + j*lim[1][2]), 
                                            lim[0][0] + i*lim[0][2], 2, 0));
      }
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_gas;
  }
  return exp(interpol2d(table, 
                        Ntable.N_a, lim[0][0], lim[0][1], lim[0][2], a, 
                        Ntable.N_k_nlin, lim[1][0], lim[1][1], lim[1][2], log(k)));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double p_gm_nointerp(
    const double k, 
    const double a, 
    const int ni,
    const int init
  )
{
  const double bg = bgal(ni, a);
  const double ng = ngal(ni, a);
  if (!(ng > 0)) return 1.0e-30;   // avoid div-by-zero
  return Pdelta(k, a)*bg + GM02_nointerp(k, a, ni, init)/ng;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double p_gm(
    const double k, 
    const double a, 
    const int ni
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double*** table = NULL;
  static double** lim = NULL; //lim[:,0] = amin; lim[:,1] = amax; lim[:,2] = da; 
                              //lim[redshift.clustering_nbin][0] = lnkmin; 
                              //lim[redshift.clustering_nbin][1] = lnkmax; 
                              //lim[redshift.clustering_nbin][2] = dlnk; 

  const int nbin = redshift.clustering_nbin;
  const int na = (int) Ntable.N_a/5.0; // range is the (\delta a) of a single bin
  
  if (NULL == table || fdiff2(cache[1], Ntable.random))
  {
    if (table != NULL) free(table);
    table = (double***) malloc3d(nbin, na, Ntable.N_k_nlin);
    if (lim != NULL) free(lim);
    lim = (double**) malloc2d(nbin+1, 3);
    for (int l=0; l<redshift.clustering_nbin; l++) {
      lim[l][0] = amin_lens(l);
      lim[l][1] = amax_lens(l);
      lim[l][2] = (lim[l][1] - lim[l][0])/((double) na - 1.0);
    }
    lim[nbin][0] = log(limits.k_min_cH0);
    lim[nbin][1] = log(limits.k_max_cH0);
    lim[nbin][2] = (lim[nbin][1]-lim[nbin][0])/((double) Ntable.N_k_nlin - 1.0);
  }

  if (fdiff2(cache[0], cosmology.random) || 
      fdiff2(cache[1], Ntable.random)    ||
      fdiff2(cache[2], nuisance.random_galaxy_bias) ||
      fdiff2(cache[3], redshift.random_clustering))
  { 
    (void) p_gm_nointerp(exp(lim[nbin][0]), lim[0][0], 0, 1); // init static vars
    (void) bgal(0, lim[0][0]);
    (void) ngal(0, lim[0][0]);
    (void) Pdelta(exp(lim[nbin][0]), lim[0][0]);
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int l=0; l<redshift.clustering_nbin; l++) {
      for (int i=0; i<na; i++) {
        for (int j=0; j<Ntable.N_k_nlin; j++) {
          table[l][i][j] = log(p_gm_nointerp(exp(lim[nbin][0] + j*lim[nbin][2]), 
                                             lim[l][0] + i*lim[l][2], l, 0));
        }
      }
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_galaxy_bias;
    cache[3] = redshift.random_clustering;
  }
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  return (a < lim[ni][0] || a > lim[ni][1]) ? 0.0 : exp(interpol2d(table[ni], 
    na, lim[ni][0], lim[ni][1], lim[ni][2], a, 
    Ntable.N_k_nlin, lim[nbin][0], lim[nbin][1], lim[nbin][2], log(k)));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double p_gg_nointerp(const double k, const double a, const int ni, const int nj, const int init)
{
  if (ni != nj) { log_fatal("..."); exit(1); }
  const double bg  = bgal(ni, a);
  const double ng  = ngal(ni, a);
  if (!(ng > 0)) return 1.0e-30;
  const double two_halo = Pdelta(k, a) * bg * bg;
  const double one_halo = G02_nointerp(k, a, ni, init) / (ng*ng);

  

  return two_halo + one_halo;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double p_gg(
    const double k, 
    const double a, 
    const int ni, 
    const int nj
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double*** table = NULL;
  static double** lim = NULL; //lim[0,:] = amin; lim[:,1] = amax; lim[:,2] = da; 
                              //lim[redshift.clustering_nbin] = lnkmin; 
                              //lim[redshift.clustering_nbin] = lnkmax; 
                              //lim[redshift.clustering_nbin] = dlnk; 
  const int nbin = redshift.clustering_nbin;
  const int na = (int) Ntable.N_a/5.0;

  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double***) malloc3d(nbin, na, Ntable.N_k_nlin);
    if (lim != NULL) free(lim);
    lim = (double**) malloc2d(nbin+1, 3);
    for (int l=0; l<redshift.clustering_nbin; l++) {
      lim[l][0] = amin_lens(l);
      lim[l][1] = amax_lens(l);
      lim[l][2] = (lim[l][1] - lim[l][0])/((double) na - 1.);
    }
    lim[nbin][0] = log(limits.k_min_cH0);
    lim[nbin][1] = log(limits.k_max_cH0);
    lim[nbin][2] = (lim[nbin][1]-lim[nbin][0])/((double) Ntable.N_k_nlin - 1.);
  }
  if (fdiff2(cache[0], cosmology.random) || 
      fdiff2(cache[1], Ntable.random)    ||
      fdiff2(cache[2], nuisance.random_galaxy_bias) ||
      fdiff2(cache[3], redshift.random_clustering))
  { 
    (void) p_gg_nointerp(exp(lim[nbin][0]), lim[0][0], 0, 0, 1); // init GSL static vars
    // Force interpolation tables to build SERIALLY before the parallel region,
    // otherwise nested omp parallel-for inside bgal/ngal/Pdelta races -> segfault.
    (void) bgal(0, lim[0][0]);
    (void) ngal(0, lim[0][0]);
    (void) Pdelta(exp(lim[nbin][0]), lim[0][0]);
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int l=0; l<nbin; l++) {
      for (int i=0; i<na; i++) {
        for (int j=0; j<Ntable.N_k_nlin; j++) {
          table[l][i][j] = log(p_gg_nointerp(exp(lim[nbin][0]+j*lim[nbin][2]),
                                             lim[l][0]+i*lim[l][2], l, l, 0));
        }
      }
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_galaxy_bias;
    cache[3] = redshift.random_clustering;
  }
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  if (ni != nj) {
    log_fatal("cross-tomography (ni,nj) = (%d,%d) bins not supported", ni, nj);
    exit(1);
  }  
  return (a < lim[ni][0] || a > lim[ni][1]) ? 0.0 : exp(
    interpol2d(table[ni], 
               na, lim[ni][0], lim[ni][1], lim[ni][2], a, 
               Ntable.N_k_nlin, lim[nbin][0], lim[nbin][1], lim[nbin][2], log(k)));
}

// 1-halo dI (matter-satellite) power spectrum.
// P_dI^1h(k,a,ni) = A * [ ∫dlnM dN/dlnM (M/rhom) u_c n_sat_red u_ia ] / n_bar
double p_dI_1h_nointerp(const double k, const double a, const int ni)
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  const double nbar = n_red_sat_bar(ni, a);   // ∫ dN/dlnM n_sat_red  (case 1)
  if (!(nbar > 0)) {
    return 0.0;
  }
  const double A = nuisance.ia[5][ni];          // satellite amplitude, lens-bin slot
  const double I_dI = I_for_IA_nointerp(k, a, ni, 3, 0);  // case 3 = matter-sat

  return A * I_dI / nbar;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// MISCELLANEOUS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
/*
void set_HOD(const int ni)
{ 
  const double z = zmean(ni);
  const double a = 1.0/(z + 1.0);
  
  // Parameterization of Zehavi et al. 
  // hod[zi][] = {lg(M_min), sigma_{lg M}, lg M_1, lg M_0, alpha, f_c}
  // gbias.gc[] = {f_g} (shift of concentration parameter: c_g(M) = f_g c(M))
  
  // Values from Coupon etal. (2012) for red gals with M_r < -21.8 (Table B.2)
  switch (ni)
  {
    case 0:
    {
      nuisance.hod[0][0] = 13.17;
      nuisance.hod[0][1] = 0.39;
      nuisance.hod[0][2] = 14.53;
      nuisance.hod[0][3] = 11.09;
      nuisance.hod[0][4] = 1.27;
      nuisance.hod[0][5] = 1.00;
      nuisance.hod[0][6] = 13.0;   // log10(M_transition) for red centrals
      nuisance.hod[0][7] = 0.5;    // width of sigmoid for red centrals
      nuisance.hod[0][8] = 12.5;   // log10(M_transition) for red satellites
      nuisance.hod[0][9] = 0.5;    // width of sigmoid for red satellites
      nuisance.gb[0][ni] = bgal(ni, a);      
      break;
    }
    case 1:
    {
      nuisance.hod[1][0] = 13.18;
      nuisance.hod[1][1] = 0.30;
      nuisance.hod[1][2] = 14.47;
      nuisance.hod[1][3] = 10.93;
      nuisance.hod[1][4] = 1.36;
      nuisance.hod[1][5] = 1.00;
      nuisance.hod[1][6] = 13.0;   // log10(M_transition) for red centrals
      nuisance.hod[1][7] = 0.5;    // width of sigmoid for red centrals
      nuisance.hod[1][8] = 12.5;   // log10(M_transition) for red satellites
      nuisance.hod[1][9] = 0.5;    // width of sigmoid for red satellites
      nuisance.gb[0][ni] = hm_funcs_nointerp(ni, a, 3, 0);
      break;
    }
    case 2:
    {
      nuisance.hod[2][0] = 12.96;
      nuisance.hod[2][1] = 0.38;
      nuisance.hod[2][2] = 14.10;
      nuisance.hod[2][3] = 12.47;
      nuisance.hod[2][4] = 1.28;
      nuisance.hod[2][5] = 1.00;
      nuisance.hod[2][6] = 13.0;   // log10(M_transition) for red centrals
      nuisance.hod[2][7] = 0.5;    // width of sigmoid for red centrals
      nuisance.hod[2][8] = 12.5;   // log10(M_transition) for red satellites
      nuisance.hod[2][9] = 0.5;    // width of sigmoid for red satellites
      nuisance.gb[0][ni] = hm_funcs_nointerp(ni, a, 3, 0);
      break;
    }
    case 3:
    {
      nuisance.hod[3][0] = 12.80;
      nuisance.hod[3][1] = 0.35;
      nuisance.hod[3][2] = 13.94;
      nuisance.hod[3][3] = 12.15;
      nuisance.hod[3][4] = 1.52;
      nuisance.hod[3][5] = 1.00;
      nuisance.hod[3][6] = 13.0;   // log10(M_transition) for red centrals
      nuisance.hod[3][7] = 0.5;    // width of sigmoid for red centrals
      nuisance.hod[3][8] = 12.5;   // log10(M_transition) for red satellites
      nuisance.hod[3][9] = 0.5;    // width of sigmoid for red satellites
      nuisance.gb[0][ni] = hm_funcs_nointerp(ni, a, 3, 0);
      break;
    }
    case 4:
    { // no information for higher redshift populations - copy 1<z<1.2 values
      nuisance.hod[4][0] = 12.80;
      nuisance.hod[4][1] = 0.35;
      nuisance.hod[4][2] = 13.94;
      nuisance.hod[4][3] = 12.15;
      nuisance.hod[4][4] = 1.52;
      nuisance.hod[4][5] = 1.00;
      nuisance.hod[4][6] = 13.0;   // log10(M_transition) for red centrals
      nuisance.hod[4][7] = 0.5;    // width of sigmoid for red centrals
      nuisance.hod[4][8] = 12.5;   // log10(M_transition) for red satellites
      nuisance.hod[4][9] = 0.5;    // width of sigmoid for red satellites
      nuisance.gb[0][ni] = hm_funcs_nointerp(ni, a, 3, 0);
      break;
    }
    default:
    {
      log_fatal("no HOD parameters specified to initialize bin %d\n", ni);
      exit(1);
    }
  }

  log_debug("HOD: bin %d; <z> %.2f; <n_g> %e(h/Mpc)^3", ni, z, 
    ngal_nointerp(ni, a, 0)*pow(cosmology.coverH0, -3.0));
  
  log_debug("HOD: bin %d; <z> %.2f; <M> h/Msun %.4e", ni, z, mmean_nointerp(ni,a,0));
  
  log_debug("HOD: bin %d; <z> %.2f; f_sat %.3f", ni, z, fsat_nointerp(ni,a,0));
  
  log_debug("HOD: bin %d; <z> %.2f; <b_g> %.2f", ni, z, nuisance.gb[0][ni]);
}
*/
// ===========================================================================
// ===========================================================================
// HALO-MODEL IA POWER SPECTRA -- PIPELINE-FACING INTERFACE
// ===========================================================================
// ===========================================================================
//
// These routines are what cosmo2D.c calls when
//     nuisance.IA_code == IA_CODE_HALO_MODEL.
//
// ---------------------------------------------------------------------------
// UNITS (read this before touching anything)
// ---------------------------------------------------------------------------
// cosmolike works internally in "code units" set by cosmology.coverH0:
//     coverH0 = c/H0 = 2997.92458 Mpc/h
//     k_code  = k_phys [h/Mpc]  * coverH0        (dimensionless)
//     P_code  = P_phys [(Mpc/h)^3] / coverH0^3   (dimensionless)
//     rho_crit is ALREADY the comoving critical density in code units
//     (cosmology.rho_crit = 7.4775e21, i.e. M_sun/h per (c/H0)^3).
//
// Consequences for this file:
//   * The mass integrals use dNdlnM = gnu*(rhom/m)*dlognudlogm(m) with
//     rhom = rho_crit*Omega_m in code units -> dNdlnM is a number density in
//     (c/H0)^-3. Dividing by nbar^2 (also (c/H0)^-3) and multiplying by the
//     dimensionless u profiles therefore yields (c/H0)^3 == P_code. Correct.
//   * u_ia_sat() already converts the 0.06 Mpc/h alignment floor via
//     "0.06/cosmology.coverH0" -> code units. Correct.
//   * p_II_2h_cen / p_dI_2h_cen are built on the matter spectrum selected by
//     HM_IA_2H_PMM (Pdelta or the halo-model p_mm), which is already
//     P_code, times dimensionless (A*b) factors. Correct.
//   * A_nla_cen uses nuisance.c1rhocrit_ia (= C1*rho_crit, dimensionless by
//     construction in cosmolike) -> the NLA amplitude is dimensionless.
//
//  => Every term below is already in P_code. DO NOT apply any additional
//     coverH0 power at the call site in cosmo2D.c. The only thing the caller
//     must guarantee is that it passes k in code units (which it does:
//     k = ell/fK where fK is in c/H0 units).
//
// ---------------------------------------------------------------------------
// SIGN / AMPLITUDE CONVENTION (this is the subtle part)
// ---------------------------------------------------------------------------
// The FAST-PT path in cosmo2D.c writes, schematically, for shear-shear EE:
//     ans = WK1*WK2*PK - WS1*WK2*C11*PK - WS2*WK1*C12*PK + WS1*WS2*C11*C12*PK
// i.e. the IA amplitude C1 and the matter power spectrum PK are supplied
// SEPARATELY and multiplied at the call site.
//
// The halo model does NOT factorize this way: P_II_halo and P_dI_halo return
// the FULL spectra with the alignment amplitude already inside. The pipeline
// must therefore substitute:
//     C11*PK          ->  P_dI_halo(k,a,n1)
//     C11*C12*PK      ->  P_II_halo(k,a,n1)   [auto-spectrum]
// and must NOT multiply by C1/C2/b_ta/PK again.
//
// SIGN: A_nla_cen() carries the standard cosmolike minus sign
//     A_NLA = -A_IA * C1*rho_crit * Omega_m / D(a) * ((1+z)/(1+z0))^eta
// exactly like IA_A1_Z1Z2 in IA.c. So P_dI_halo has the SAME sign convention
// as (C1*PK) in the FAST-PT branch, and the pipeline keeps its existing
// "-WS*..." structure unchanged. P_II_halo is quadratic in the amplitude and
// is positive-definite, matching "+WS1*WS2*C11*C12*PK".
//
// ---------------------------------------------------------------------------
// LENS-BIN vs SOURCE-BIN CAVEAT
// ---------------------------------------------------------------------------
// The halo-model IA routines (p_II_1h_nointerp, p_dI_1h_nointerp,
// b_red_cen, A_nla_cen, ...) are all indexed by a CLUSTERING (lens) bin,
// because they need HOD parameters (nuisance.hod[ni][...]) and the red
// central/satellite fractions, which only exist for lens bins.
//
// The shear kernels, however, need IA in SOURCE bins. We bridge this with
// halo_IA_lensbin_of_sourcebin(). The default is an identity-with-clamp map.
// If the lens and source samples are not the same galaxies, this is a
// modelling assumption the user must own -- see the log_warn below.
// ---------------------------------------------------------------------------

int halo_IA_lensbin_of_sourcebin(const int ns)
{
  const int nlbin = redshift.clustering_nbin;
  if (nlbin < 1) {
    log_fatal("halo-model IA requires redshift.clustering_nbin >= 1 "
              "(HOD parameters are defined per lens bin)");
    exit(1);
  }
  if (ns < 0) {
    log_fatal("invalid source bin ns = %d", ns);
    exit(1);
  }
  // Identity map, clamped to the available lens bins.
  return (ns > nlbin - 1) ? nlbin - 1 : ns;
}

// ---------------------------------------------------------------------------
// HALO-EXCLUSION / 1h-2h TRANSITION WINDOWS
// ---------------------------------------------------------------------------
// The naive total P = P_1h + P_2h double-counts in the transition region: the
// 2-halo term uses the (non)linear matter power, which already contains 1-halo
// power at those scales. Following Fortuna et al. (2020), Appendix B, we
// suppress each term where it should not contribute:
//
//   P_total(k) = f_1h(k) * P_1h(k)  +  f_2h(k) * P_2h(k)
//   f_2h(k) = exp[ -(k/k_2h)^2 ]        -- kills 2-halo at HIGH k
//   f_1h(k) = 1 - exp[ -(k/k_1h)^2 ]    -- kills 1-halo at LOW k
//
// with the paper's defaults k_2h = 6 h/Mpc, k_1h = 4 h/Mpc. The offset leaves
// a small overlap so the transition is gradual, not a hard switch.
//
// UNITS (critical): k here is in CODE units (k_phys * coverH0), while the
// thresholds are quoted in h/Mpc. So the code-unit threshold is
//   k_thr_code = k_thr_hMpc * cosmology.coverH0.
// Getting this wrong puts the transition at the wrong scale by a factor
// coverH0 ~ 3000 and is silent -- do not use a bare 4.0 / 6.0 here.
//
// CAVEATS worth knowing (see Fortuna App. B, and Sect. 6.2.1):
//   * This is a fudge to avoid double counting, NOT a simulation-calibrated
//     halo exclusion. The paper found Stage-IV cosmological bias is sensitive
//     to the exact recipe (a smoother/wider transition trades Omega_m bias
//     against S8/w bias). k_1h, k_2h and the Gaussian form are all knobs.
//   * The "right" k_2h depends on WHICH matter power the 2-halo term uses
//     (Pdelta vs the halo-model p_mm, via HM_IA_2H_PMM): p_mm already carries
//     its own 1h+2h split, so the double-counting structure differs. Fix the
//     matter-power choice first, then tune the window against it.
//
// Toggle with HM_IA_TRUNC (0 = plain sum, as before; 1 = apply windows).
// ---------------------------------------------------------------------------
#ifndef HM_IA_TRUNC
#define HM_IA_TRUNC 1        // 0 = plain 1h+2h sum, 1 = Fortuna App. B windows
#endif
#ifndef HM_IA_K1H_HMPC
#define HM_IA_K1H_HMPC 4.0   // 1-halo low-k cutoff [h/Mpc]
#endif
#ifndef HM_IA_K2H_HMPC
#define HM_IA_K2H_HMPC 6.0   // 2-halo high-k cutoff [h/Mpc]
#endif

static inline double f_1h_trunc(const double k)
{
#if HM_IA_TRUNC == 1
  const double k1h = HM_IA_K1H_HMPC * cosmology.coverH0;  // -> code units
  const double r = k / k1h;
  return 1.0 - exp(-r*r);
#else
  (void) k;
  return 1.0;
#endif
}

static inline double f_2h_trunc(const double k)
{
#if HM_IA_TRUNC == 1
  const double k2h = HM_IA_K2H_HMPC * cosmology.coverH0;  // -> code units
  const double r = k / k2h;
  return exp(-r*r);
#else
  (void) k;
  return 1.0;
#endif
}

// ---------------------------------------------------------------------------
// Full II spectrum = 1-halo (satellite-satellite) + 2-halo (central NLA).
// ni is a CLUSTERING bin index.
// ---------------------------------------------------------------------------
double P_II_halo_nointerp(const double k, const double a, const int ni,
                          const int init)
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }
  if (1 == init) {
    // Touch the underlying static GSL tables serially before any omp region.
    (void) I_for_IA_nointerp(k, a, ni, 1, 1);
    (void) I_for_IA_nointerp(k, a, ni, 2, 1);
    (void) I_bred_cen_nointerp(a, ni, 0, 1);
    (void) I_bred_cen_nointerp(a, ni, 1, 1);
    (void) Pdelta(k, a);
    return 0.0;
  }
  const double p1h = p_II_1h_nointerp(k, a, ni);
  const double p2h = p_II_2h_cen_nointerp(k, a, ni);
  const double res = f_1h_trunc(k)*p1h + f_2h_trunc(k)*p2h;
  return (res > 0.0) ? res : 0.0;   // II is positive-definite
}

// ---------------------------------------------------------------------------
// Full dI spectrum = 1-halo (matter-satellite) + 2-halo (central NLA).
// NOTE the sign: p_dI_2h_cen_nointerp is NEGATIVE for A_IA > 0 (via
// A_nla_cen), while p_dI_1h_nointerp is built with fabs(u_ia) and a bare
// nuisance.ia[5][ni] amplitude. To keep the two halo terms in a CONSISTENT
// sign convention we force the 1-halo term to follow the sign of the 2-halo
// (NLA) amplitude. Without this, the 1h and 2h terms can spuriously cancel.
// ---------------------------------------------------------------------------
double P_dI_halo_nointerp(const double k, const double a, const int ni,
                          const int init)
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }
  if (1 == init) {
    (void) I_for_IA_nointerp(k, a, ni, 1, 1);
    (void) I_for_IA_nointerp(k, a, ni, 3, 1);
    (void) I_bred_cen_nointerp(a, ni, 0, 1);
    (void) I_bred_cen_nointerp(a, ni, 1, 1);
    (void) Pdelta(k, a);
    return 0.0;
  }
  const double p2h = p_dI_2h_cen_nointerp(k, a, ni);
  const double p1h_mag = p_dI_1h_nointerp(k, a, ni);   // magnitude-like
  const double sgn = (A_nla_cen(ni, a) < 0.0) ? -1.0 : 1.0;
  return f_2h_trunc(k)*p2h + f_1h_trunc(k)*sgn*fabs(p1h_mag);
}

// ---------------------------------------------------------------------------
// Tabulated / interpolated wrappers.
//
// Caching follows the p_gg / p_gm pattern already used in this file:
//   cache[0] cosmology, cache[1] Ntable, cache[2] IA nuisance,
//   cache[3] clustering redshift.
//
// We tabulate in (a, ln k) per lens bin. II is stored as log(P) since it is
// positive-definite. dI CHANGES SIGN in general (A_IA can be negative), so it
// is stored LINEARLY -- storing log(dI) would silently produce NaNs.
// ---------------------------------------------------------------------------

static void halo_IA_setup_limits(double** lim, const int nbin, const int na)
{
  for (int l=0; l<nbin; l++) {
    lim[l][0] = amin_lens(l);
    lim[l][1] = amax_lens(l);
    lim[l][2] = (lim[l][1] - lim[l][0])/((double) na - 1.0);
  }
  lim[nbin][0] = log(limits.k_min_cH0);
  lim[nbin][1] = log(limits.k_max_cH0);
  lim[nbin][2] = (lim[nbin][1] - lim[nbin][0])/((double) Ntable.N_k_nlin - 1.0);
}

double P_II_halo(const double k, const double a, const int ni)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double*** table = NULL;
  static double** lim = NULL;

  const int nbin = redshift.clustering_nbin;
  const int na = (int) Ntable.N_a/5.0;

  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double***) malloc3d(nbin, na, Ntable.N_k_nlin);
    if (lim != NULL) free(lim);
    lim = (double**) malloc2d(nbin+1, 3);
    halo_IA_setup_limits(lim, nbin, na);
  }

  if (fdiff2(cache[0], cosmology.random)   ||
      fdiff2(cache[1], Ntable.random)      ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_clustering))
  {
    halo_IA_setup_limits(lim, nbin, na);
    // Build all static/GSL tables SERIALLY first -- nested omp inside
    // Pdelta/growfac/sigma2 otherwise races (same reason as in p_gg).
    (void) P_II_halo_nointerp(exp(lim[nbin][0]), lim[0][0], 0, 1);
    (void) Pdelta(exp(lim[nbin][0]), lim[0][0]);
    (void) growfac(lim[0][0]);
#if HM_IA_2H_PMM == 1
    // p_mm runs its own omp table build; force it serially here (same race
    // class as Pdelta above) since the 2-halo central term now calls it.
    (void) p_mm(exp(lim[nbin][0]), lim[0][0]);
#endif
    // MANDATORY: these three build shared lookup tables and MUST run serially.
    // The parallel region below calls them read-only; building them lazily
    // inside it races (concurrent malloc + nested omp) and segfaults.
    u_ia_sat_init();
    n_red_sat_bar_init();
    b_red_cen_init();

    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int l=0; l<nbin; l++) {
      for (int i=0; i<na; i++) {
        for (int j=0; j<Ntable.N_k_nlin; j++) {
          const double p = P_II_halo_nointerp(exp(lim[nbin][0] + j*lim[nbin][2]),
                                              lim[l][0] + i*lim[l][2], l, 0);
          table[l][i][j] = log(p > 1.0e-30 ? p : 1.0e-30);
        }
      }
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_clustering;
  }

  if (ni < 0 || ni > nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }
  if (a < lim[ni][0] || a > lim[ni][1]) return 0.0;

  const double lnk = log(k);
  if (lnk < lim[nbin][0] || lnk > lim[nbin][1]) return 0.0;

  return exp(interpol2d(table[ni],
      na, lim[ni][0], lim[ni][1], lim[ni][2], a,
      Ntable.N_k_nlin, lim[nbin][0], lim[nbin][1], lim[nbin][2], lnk));
}

double P_dI_halo(const double k, const double a, const int ni)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double*** table = NULL;
  static double** lim = NULL;

  const int nbin = redshift.clustering_nbin;
  const int na = (int) Ntable.N_a/5.0;

  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double***) malloc3d(nbin, na, Ntable.N_k_nlin);
    if (lim != NULL) free(lim);
    lim = (double**) malloc2d(nbin+1, 3);
    halo_IA_setup_limits(lim, nbin, na);
  }

  if (fdiff2(cache[0], cosmology.random)   ||
      fdiff2(cache[1], Ntable.random)      ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_clustering))
  {
    halo_IA_setup_limits(lim, nbin, na);
    (void) P_dI_halo_nointerp(exp(lim[nbin][0]), lim[0][0], 0, 1);
    (void) Pdelta(exp(lim[nbin][0]), lim[0][0]);
    (void) growfac(lim[0][0]);
#if HM_IA_2H_PMM == 1
    (void) p_mm(exp(lim[nbin][0]), lim[0][0]);   // serial pre-build; see P_II_halo
#endif
    // MANDATORY serial table builds -- see the note in P_II_halo.
    u_ia_sat_init();
    n_red_sat_bar_init();
    b_red_cen_init();

    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int l=0; l<nbin; l++) {
      for (int i=0; i<na; i++) {
        for (int j=0; j<Ntable.N_k_nlin; j++) {
          // stored LINEARLY: dI is sign-indefinite
          table[l][i][j] = P_dI_halo_nointerp(exp(lim[nbin][0] + j*lim[nbin][2]),
                                              lim[l][0] + i*lim[l][2], l, 0);
        }
      }
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_clustering;
  }

  if (ni < 0 || ni > nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }
  if (a < lim[ni][0] || a > lim[ni][1]) return 0.0;

  const double lnk = log(k);
  if (lnk < lim[nbin][0] || lnk > lim[nbin][1]) return 0.0;

  return interpol2d(table[ni],
      na, lim[ni][0], lim[ni][1], lim[ni][2], a,
      Ntable.N_k_nlin, lim[nbin][0], lim[nbin][1], lim[nbin][2], lnk);
}