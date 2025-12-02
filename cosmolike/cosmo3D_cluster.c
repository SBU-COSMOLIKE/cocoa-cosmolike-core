#include <assert.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_spline.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../cfftlog/cfftlog.h"

#include "bias.h"
#include "basics.h"
#include "cosmo3D.h"
#include "cosmo3D_cluster.h"
#include "cluster_util.h"
#include "cosmo2D.h"
#include "cosmo2D_cluster.h"
#include "radial_weights.h"
#include "recompute.h"
#include "redshift_spline.h"
#include "structs.h"

#include "log.c/src/log.h"

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double prob_lambda_obs_given_m_given_ztrue(double l, void* params)
{
  double* ar = (double*) params; //mass, a
  const double M = ar[0];
  const double a = ar[1];
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  const double lnl = log(l);

  const double lnl0 = nuisance.cluster_mor[0]; 
  const double Al = nuisance.cluster_mor[1];
  const double Bl = nuisance.cluster_mor[3];
  // below: \sigma_ln(l)_intrinsic^2
  const double slnli2 = nuisance.cluster_MOR[2]*nuisance.cluster_MOR[2]; 

  const double alnl = lnl0 + Al*log(M/5.E14) + Bl*log(1./(1.45*a)); // average ln(l)
  const double slnl2 = (alnl > 0) ? slnli2 + (exp(alnl) - 1.)/exp(2*alnl) : slnli2;
  const double slnl = sqrt(slnl2)

  // below: normalized log-normal 1/sqrt(2pi) \int dlnx e(-(lnx-a)^2/b^2) = 1
  return exp(-0.5*(lnl-alnl)*(lnl-alnl)/slnl2)/(M_SQRTPI*M_SQRT2*slnl*l); 
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double prob_lambda_obs_in_nl_given_m_given_ztrue_nointerp(
    const int nl, 
    const double m, 
    const double a,
    const int init,
  ) 
{
  static double cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;
  if (nl < 0 || nl > Cluster.n200_nbin - 1) {
    log_fatal("error in bin number (nl) = %d", nl); exit(1); 
  }
  if (NULL == w || fdiff(cache[0], Ntable.random)) {
    const size_t szint = 50 + 40 * abs(Ntable.high_def_integration);
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  const double nlmin = like.cluster_lambda_lims[0][nl];
  const double nlmax = like.cluster_lambda_lims[1][nl];
  double params[2] = {m, a};

  double res = 0.0; 
  if (1 == init) { 
    (void) prob_lambda_obs_given_m_given_ztrue(nlmin, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = prob_lambda_obs_given_m_given_ztrue;
    res = gsl_integration_glfixed(&F, nlmin, nlmax, w);
  }
  return res;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double prob_lambda_obs_in_nl_given_m_given_ztrue(
    const int nl, 
    const double m, 
    const double a
  ) 
{
  static double cache[MAX_SIZE_ARRAYS];
  static double*** table = NULL;
  static double lim[6];
  const int nlnM = Ntable.nlnm_cosmo3D_cluster;
  const int na   = Ntable.na_cosmo3D_cluster;
  // ---------------------------------------------------------------------------
  if (NULL == table || fdiff(cache[3], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double***) malloc3d(Cluster.n200_nbin, na, nlnM);  
    const double zmin = fmax(redshift.clusters_zdist_zmin_all - 0.05, 0.01);
    const double zmax = redshift.clusters_zdist_zmax_all + 0.05;
    lim[0] = 1.0/(1.0 + zmax);                          // amin
    lim[1] = 1.0/(1.0 + zmin);                          // amax
    lim[2] = (lim[1] - lim[0])/((double) na - 1.);      // da 
    lim[3] = log(limits.halo_m_min);                    // lnMin
    lim[4] = log(limits.halo_m_max);                    // lnMax
    lim[5] = (lim[1] - lim[0])/((double) nlnM - 1.);    // dlnM
  }
  if (fdiff(cache[1], nuisance.random_clusters) ||
      fdiff(cache[2], redshift.random_clusters) ||
      fdiff(cache[3], Ntable.random))
  {
    (void) prob_lambda_obs_in_nl_given_m_given_ztrue_nointerp(0,lim[0],lim[4],1);
    #pragma omp parallel for collapse(3) schedule(static,1)
    for (int i=0; i<Cluster.n200_nbin; i++) {
      for (int j=0; j<nlnM; j++) {
        for (int k=0; k<na; k++) {
          const double ain = lim[0] + k*lim[2];
          const double Min = exp(lim[3] + j*lim[5]);
          table[i][k][j] = 
                prob_lambda_obs_in_nl_given_m_given_ztrue_nointerp(i,Min,ain,0);
        }
      } 
    }
    // -------------------------------------------------------------------------
    cache[1] = nuisance.random_clusters;
    cache[2] = redshift.random_clusters;
    cache[3] = Ntable.random;
  }
  if (nl < 0 || nl > Cluster.n200_nbin - 1) {
    log_fatal("error in bin number (nl) = %d", nl); exit(1); 
  }
  const double lnM = log(m);
  return (a<lim[0] || a>lim[1]) ? 0.0 : (lnM<lim[3] || lnM>lim[4]) ? 0.0 :
    interpol2d(table[i],na,lim[0],lim[1],lim[2],a,nlnM,lim[3],lim[4],lim[5],lnM);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double int_ncl_given_lambda_obs_within_nl_given_ztrue(double lnM, void* params)
{
  double* ar = (double *) params;
  const int nl = (int) ar[0];
  const double a = ar[1];
  const double growfac_a = ar[2];
  const double M = exp(lnM); 
  
  const double nu = delta_c/(sqrt(sigma2(M))*growfac_a);
  const double gnu = fnu(nu, a) * nu; 
  const double rhom = cosmology.rho_crit * cosmology.Omega_m;
  const double dndlnM = gnu * (rhom/M) * dlognudlogm(M);

  return dndlnM*prob_lambda_obs_within_nl_given_m_given_ztrue(nl, M, a);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double ncl_given_lambda_obs_within_nl_given_ztrue_nointerp(
    const int nl, 
    const double a,
    const int init
  )
{
  static double cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;
  if (nl < 0 || nl > Cluster.n200_nbin - 1) {
    log_fatal("error in bin number (nl) = %d", nl); exit(1); 
  }
  if (NULL == w || fdiff(cache[0], Ntable.random)) {
    const size_t szint = 50 + 40 * abs(Ntable.high_def_integration);
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  const double lnMmin = log(limits.cluster_halo_m_min);
  const double lnMmax = log(limits.cluster_halo_m_max);
  double ar[3] = {(double) nl, a, growfac(a)};
  
  double res = 0.0; 
  if (1 == init) { 
    (void) int_ncl_given_lambda_obs_within_nl_given_ztrue(lnMmin, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_ncl_given_lambda_obs_within_nl_given_ztrue;
    res = gsl_integration_glfixed(&F, lnMmin, lnMmax, w);
  }
  return res;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double ncl_given_lambda_obs_within_nl_given_ztrue(
    const int nl, 
    const double a
  )
{
  static double cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double lim[3];
  const int na = Ntable.na_cosmo3D_cluster;
  // ---------------------------------------------------------------------------
  if (NULL == table || fdiff(cache[3], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double**) malloc2d(Cluster.n200_nbin, na);  
    const double zmin = fmax(redshift.clusters_zdist_zmin_all - 0.05, 0.01);
    const double zmax = redshift.clusters_zdist_zmax_all + 0.05;
    lim[0] = 1.0/(1.0 + zmax);                      // amin
    lim[1] = 1.0/(1.0 + zmin);                      // amax
    lim[2] = (lim[1] - lim[0])/((double) na - 1.);  // da 
  }
  if (fdiff(cache[0], cosmology.random) || 
      fdiff(cache[1], nuisance.random_clusters) ||
      fdiff(cache[2], redshift.random_clusters) ||
      fdiff(cache[3], Ntable.random))
  {
    (void) ncl_given_lambda_obs_within_nl_given_ztrue_nointerp(0, lim[0], 1);
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Cluster.n200_nbin; i++) {
      for (int k=0; k<na; k++) {
        const double ain = lim[0] + k*lim[2];
        table[i][k] = 
                   ncl_given_lambda_obs_within_nl_given_ztrue_nointerp(i,ain,0);
      }
    }
    // -------------------------------------------------------------------------
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_clusters;
    cache[2] = redshift.random_clusters;
    cache[3] = Ntable.random;
  }
  if (nl < 0 || nl > Cluster.n200_nbin - 1) {
    log_fatal("error in bin number (nl) = %d", nl); exit(1); 
  }
  return (a<lim[0] || a>lim[1]) ? 0.0 : 
                                interpol1d(table[i],na,lim[0],lim[1],lim[2],a); 
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double csb1(const double M, const double a) // cluster selection bias
{
  double ans;
  switch(like.cluster_bias_model)
  {
    case 1:
    {
      ans = nuisance.clusters_sb[0];
      break;
    }
    case 2:
    {
      ans = nuisance.clusters_sb[0]**pow(M/5E14, nuisance.clusters_sb[1]); 
      break;
    }
    case 3:
    {
      ans = 1;
      break;
    }
    default:
    {
      log_fatal("like.cluster_bias_model[0] = %d not supported", 
        like.cluster_bias_model[0]); exit(1);  
    }
  }
  return ans;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double int_cluster_b1_given_lambda_in_nl_obs_given_ztrue(double lnM, void* params)
{ // first integral in 
  double* ar = (double *) params;
  const int nl = (int) ar[0];
  const double a = ar[1];
  const double growfac_a = ar[2];
  const double M = exp(lnM); 

  const double nu = delta_c/(sqrt(sigma2(M))*growfac_a);
  const double gnu = fnu(nu, a) * nu; 
  const double rhom = cosmology.rho_crit * cosmology.Omega_m;
  const double dndlnM = gnu * (rhom/M) * dlognudlogm(M);

  const double hb1 = hb1nu(nu,a);
  const double sb1 = csb1(M,a);

  return hb1*sb1*dndlnM*prob_lambda_obs_in_nl_given_m_given_ztrue(nl, M, a);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double cluster_b1_given_lambda_obs_in_nl_given_ztrue_nointerp(
    const int nl, 
    const double a, 
    const int init
  )
{
  static double cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;
  if (nl < 0 || nl > Cluster.n200_nbin - 1) {
    log_fatal("error in bin number (nl) = %d", nl); exit(1); 
  }
  if (NULL == w || fdiff(cache[0], Ntable.random)) {
    const size_t szint = 50 + 40 * abs(Ntable.high_def_integration);
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }
  const double lnMmin = log(limits.cluster_halo_m_min);
  const double lnMmax = log(limits.cluster_halo_m_max);
  double ar[3] = {(double) nl, a, growfac(a)};
  double res = 0.0; 
  if (1 == init) { 
    (void) int_cluster_b1_given_lambda_in_nl_obs_given_ztrue(lnMmin, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_cluster_b1_given_lambda_in_nl_obs;
    const double num = gsl_integration_glfixed(&F, lnMmin, lnMmax, w);
    const double den = ncl_given_lambda_obs_in_nl_given_ztrue(nl, a);
    res = (den > 0) ? ((num/den > 0) ? num/den : 0.0) : 0.0;
  }
  return res;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double cluster_b1_given_lambda_obs_in_nl_given_ztrue(
    const int nl, 
    const double a
  )
{ 
  static double cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double lim[3];
  const int na = Ntable.na_cosmo3D_cluster;
  if (NULL == table || fdiff(cache[3], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double**) malloc2d(Cluster.n200_nbin, na);
  }
  if (fdiff(cache[2], redshift.random_clusters) ||
      fdiff(cache[3], Ntable.random)) {
    lim[0] = 1./(1. + redshift.clusters_zdist_zmax_all);                  // amin
    lim[1] = 1./(1. + redshift.clusters_zdist_zmin_all);                  // amax
    lim[2] = (lim[1] - lim[0])/((double) Ntable.na_cosmo3D_cluster - 1.); // da
  }
  if (fdiff(cache[0], cosmology.random) || 
      fdiff(cache[1], nuisance.random_clusters) ||
      fdiff(cache[2], redshift.random_clusters) ||
      fdiff(cache[3], Ntable.random))
  {
    (void) cluster_b1_given_lambda_obs_in_nl_given_ztrue_nointerp(0,lim[0],1); // init static vars
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int j=0; j<Cluster.N200_Nbin; j++){
      for (int i=0; i<na; i++){
        const double ain = lim[0] + i*lim[2];
        table[j][i] = 
              cluster_b1_given_lambda_obs_in_nl_given_ztrue_nointerp(j, ain, 0);
      }
    }
    // -------------------------------------------------------------------------
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_clusters;
    cache[2] = redshift.random_clusters;
    cache[3] = Ntable.random;
  }
  if (nl < 0 || nl > Cluster.n200_nbin - 1) {
    log_fatal("error in bin number nl = %d", nl); exit(1); 
  }
  return (a<lim[0] || a>lim[1]) ? 0. : interpol(table[nl],na,lim[0],lim[1],lim[2],a);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double pcc_linpsopt_nointerp(
    const double k, 
    const double a, 
    const int nl1, 
    const int nl2, 
    const int linear
  )
{
  const double cb1 = cluster_b1_given_lambda_obs_in_nl_given_ztrue(nl1, a); 
  const double cb2 = (nl1 == nl2) ? cb1 : 
                     cluster_b1_given_lambda_obs_in_nl_given_ztrue(nl2, a);
  const double pk = (1 == linear) ? p_lin(k,a) : Pdelta(k,a);
  return cb1*cb2*pk;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double pcc_nointerp(
    const double k, 
    const double a, 
    const int nl1, 
    const int nl2
  )
{
  return pcc_linpsopt_nointerp(k, a, nl1, nl2, 0);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double pcc(
    const double k, 
    const double a, 
    const int nl1, // observable richness bin
    const int nl2 // observable richness bin
  )
{
  static double cache[MAX_SIZE_ARRAYS];
  static double*** table = NULL;
  static double lim[6];
  // ---------------------------------------------------------------------------
  const int na = Ntable.na_cosmo3D_cluster;
  const int nlnk = Ntable.nlnk_cosmo3D_cluster;
  const int NSIZE = Cluster.n200_nbin*Cluster.n200_nbin;
  const int shift = 1E8;
  if (NULL == table || fdiff(cache[3], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double***) malloc3d(NSIZE, na, nlnk);
  }
  if (fdiff(cache[2], redshift.random_clusters) ||
      fdiff(cache[3], Ntable.random)) {
    lim[0] = 1./(1. + redshift.clusters_zdist_zmax_all);                    // amin
    lim[1] = 1./(1. + redshift.clusters_zdist_zmin_all);                    // amax
    lim[2] = (lim[1] - lim[0])/((double) Ntable.na_cosmo3D_cluster - 1.);   // da
    lim[3] = log(limits.linkmin_cosmo3D_cluster);                           // logkmin
    lim[4] = log(limits.linkmax_cosmo3D_cluster);                           // logkmax
    lim[5] = (lim[4] - lim[3])/((double) Ntable.nlnk_cosmo3D_cluster - 1.); //dlnk 
  }
  if (fdiff(cache[0], cosmology.random) || 
      fdiff(cache[1], nuisance.random_clusters) ||
      fdiff(cache[2], redshift.random_clusters) ||
      fdiff(cache[3], Ntable.random))
  {
    (void) pcc_nointerp(exp(lim[3]),lim[0],0,0); // init static vars 
    #pragma omp parallel for collapse(4) schedule(static,1)
    for (int p=0; p<Cluster.n200_nbin; p++) { 
      for (int l=0; l<Cluster.n200_nbin; l++) { 
        for (int i=0; i<na; i++) { 
          for (int j=0; j<nlnk; j++) {   
            const double ain = lim[0] + i*lim[2];
            const double kin = exp(lim[3] + j*lim[5]);
            const int q = k*Cluster.n200_nbin + l;
            table[q][i][j] = log(kin*kin*kin*sqrt(ain)*
                                          pcc_nointerp(kin, ain, p, l) + shift);
          }
        }
      }
    }
    // -------------------------------------------------------------------------
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_clusters;
    cache[2] = redshift.random_clusters;
    cache[3] = Ntable.random;
  }
  if (nl1 < 0 || nl1 > Cluster.n200_nbin - 1 ||
      nl2 < 0 || nl2 > Cluster.n200_nbin - 1) {
    log_fatal("error in bin number (nl1,nl2) = [%d,%d]", nl1, nl2); exit(1); 
  }
  const int q = nl1*Cluster.n200_nbin + nl2;
  if (q < 0 || q > NSIZE - 1) {
    log_fatal("internal logic error in selecting bin number"); exit(1);
  }
  const double lnk = log(k);
  const double res = interpol2d(table[q], na, lim[0], lim[1], lim[2], a, 
                                          nlnk, lim[3], lim[4], lim[5], lnk);
  return (a<lim[0] || a>lim[1]) ? 0.0 : 
         (lnk<lim[3] || lnk>lim[4]) ? 0.0 : (exp(res)-shift)/(k*k*k*sqrt(a));
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double pcc_with_excl_nointerp(
    const double k, 
    const double a, 
    const int nl1, 
    const int nl2
  )
{
  const double cb1 = cluster_b1_given_lambda_obs_in_nl_given_ztrue(nl1,a); 
  const double cb2 = (nl1 == nl2) ? cb1 : 
                     cluster_b1_given_lambda_obs_in_nl_given_ztrue(nl2,a);
  const double R = 1.5*pow(0.25*(Cluster.N_min[N_lambda1]+
                                 Cluster.N_min[N_lambda2] +
                                 Cluster.N_max[N_lambda1] +
                                 Cluster.N_max[N_lambda2])/100., 0.2)/cosmology.coverH0/a; 
  double pcc = 0.0;
  if(0 == R) {
    pcc = Pdelta(k,a)*cb1*cb2;
  }
  else {
    const double VexclWR = 4*M_PI*(sin(k*R) - k*R*cos(k*R))/(k*k*k);
    const double cff = 1.; // cff = cut off
    const double kcff = cff/R;
    if (k > kcoff) {
      const double VexclWRcff = 4*M_PI*(sin(cff) -cff*cos(cff))/(kcff*kcff*kcff);
      const double pcccff = (pk_halo_with_excl(kcff,R,a) + VexclWRcff)*cb1*cb2 - VexclWRcff;
      pcc  = Pdelta(k,a)*cb1*cb2 - VexclWR;
      pcc -= (-pcccff + (Pdelta(kcff,a)*cb1*cb2 - VexclWRcoff))*VexclWR/VexclWRcoff;
      pcc *= pow((k/kcoff),-0.7); 
    }
    else {
      // original cosmolike: Check it out!! This is my cool trick!! 
      pcc = (pk_halo_with_excl(k,R,a) + VexclWR)*cb1*cb2 - VexclWR; 
    }
  }
  return pcc;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double pcc_with_excl(
    const double k, 
    const double a, 
    const int nl1, // observable richness bin
    const int nl2 // observable richness bin
  )
{
  static double cache[MAX_SIZE_ARRAYS];
  static double*** table = NULL;
  static double lim[6];
  // ---------------------------------------------------------------------------
  const int na = Ntable.na_cosmo3D_cluster;
  const int nlnk = Ntable.nlnk_cosmo3D_cluster;
  const int NSIZE = Cluster.n200_nbin*Cluster.n200_nbin;
  const int shift = 1E8;
  if (NULL == table || fdiff(cache[3], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double***) malloc3d(NSIZE, na, nlnk);
  }
  if (fdiff(cache[2], redshift.random_clusters) ||
      fdiff(cache[3], Ntable.random)) {
    lim[0] = 1./(1. + redshift.clusters_zdist_zmax_all);    // amin
    lim[1] = 1./(1. + redshift.clusters_zdist_zmin_all);    // amax
    lim[2] = (lim[1] - lim[0])/((double) na - 1.);          // da
    lim[3] = log(limits.linkmin_cosmo3D_cluster);           // logkmin
    lim[4] = log(limits.linkmax_cosmo3D_cluster);           // logkmax
    lim[5] = (lim[4] - lim[3])/((double) nlnk - 1.);        // dlnk 
  }
  if (fdiff(cache[0], cosmology.random) || 
      fdiff(cache[1], nuisance.random_clusters) ||
      fdiff(cache[2], redshift.random_clusters) ||
      fdiff(cache[3], Ntable.random))
  {
    (void) pcc_with_excl(exp(lim[3]), lim[0], 0, 0); // init static vars
    #pragma omp parallel for collapse(4) schedule(static,1)
    for (int p=0; p<Cluster.n200_nbin; p++) { 
      for (int l=0; l<Cluster.n200_nbin; l++) { 
        for (int i=0; i<na; i++) { 
          for (int j=0; j<nlnk; j++) { 
            const double ain = lim[0] + i*lim[2];
            const double kin = exp(lim[3] + j*lim[5]);
            const int q = k*Cluster.n200_nbin + l;
            table[q][i][j] = log(kin*kin*kin*sqrt(ain)*
                                pcc_with_excl_nointerp(kin, ain, p, l) + shift);
          }
        }
      }
    }
    // -------------------------------------------------------------------------
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_clusters;
    cache[2] = redshift.random_clusters;
    cache[3] = Ntable.random;
  }
  if (nl1 < 0 || nl1 > Cluster.n200_nbin - 1 ||
      nl2 < 0 || nl2 > Cluster.n200_nbin - 1) {
    log_fatal("error in bin number (nl1,nnl2) = [%d,%d]", nl1, nl2); exit(1); 
  }
  const int q = nl1*Cluster.n200_nbin + nl2;
  if (q < 0 || q > NSIZE - 1) {
    log_fatal("internal logic error in selecting bin number"); exit(1);
  }
  const double lnk = log(k);
  const double res = interpol2d(table[q], na, lim[0], lim[1], lim[2], a, 
                                          nlnk, lim[3], lim[4], lim[5], lnk);
  return (a<lim[0] || a>lim[1]) ? 0.0 : 
         (lnk<lim[3] || lnk>lim[4]) ? 0.0 : (exp(res)-shift)/(k*k*k*sqrt(a));
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double int_pcm_1halo(double lnM, void* params) {
  double* ar = (double *) params;
  const int nl = (int) ar[0];
  const double a = ar[1];
  const double k = ar[2];
  const double M = exp(lnM); 
  const double growfac_a = ar[3];
  const double c = conc(m, growfac_a);  // defined in halo.c
  const double rhom  = cosmology.rho_crit * cosmology.Omega_m;
  
  const double nu = delta_c/(sqrt(sigma2(M))*growfac_a);
  const double gnu = fnu(nu, a) * nu; 
  const double rhom = cosmology.rho_crit * cosmology.Omega_m;
  const double dndlnM = gnu * (rhom/M) * dlognudlogm(M);

  return (M/rhom)*u_c(c,k,M,a)*dndlnM*prob_lambda_obs_given_m(nl, M, a);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double pcm_1halo_nointerp(
    const double k, 
    const double a, 
    const int nl,     // observable richness bin
    const int init
  )
{
  static double cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;
  if (nl < 0 || nl > Cluster.n200_nbin - 1) {
    log_fatal("error in bin number (nl) = %d", nl); exit(1); 
  }
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  if (NULL == w || fdiff(cache[0], Ntable.random)) {
    const size_t szint = 50 + 40 * abs(Ntable.high_def_integration);
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }
  const double lnMmin = log(limits.cluster_halo_m_min);
  const double lnMmax = log(limits.cluster_halo_m_max);
  const double growfac_a = growfac(a);
  const double norm = n_lambda_obs_z(nl, 1./a-1.0);
  double ar[4] = {(double) nl, a, k, growfac_a};
  double res = 0.0; 
  if (1 == init) { 
    (void) int_pcm_1halo(lnMmin, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_pcm_1halo;
    res = gsl_integration_glfixed(&F, lnMmin, lnMmax, w);
  }
  return (1 == init) ? 0.0 : (norm<1.E-14) ? 0.0 : (res/norm>1.E5) ? 0.0 : res/norm;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double pcm_1halo(
    const double k, 
    const double a, 
    const int nl,  // observable richness bin
    const int ni   // cluster redshift bin
  )
{
  static double cache[MAX_SIZE_ARRAYS];
  static double*** table = NULL;
  static double lim[6*MAX_SIZE_ARRAYS];
  // ---------------------------------------------------------------------------
  const int NSIZE = Cluster.n200_nbin*redshift.clusters_nbin;
  const int nlnk = Ntable.nlnk_cosmo3D_cluster;
  const int na = Ntable.na_cosmo3D_cluster;
  if (NULL == table || fdiff(cache[3], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double***) malloc3d(NSIZE, na, nlnk);
  }
  if (fdiff(cache[2], redshift.random_clusters) ||
      fdiff(cache[3], Ntable.random)) {
    for (int i=0; i<redshift.clusters_nbin; i++) {
      lim[6*i+0] = 1./(1.+tomo.cluster_zmax[i]);                 // amin
      lim[6*i+1] = 1./(1.+tomo.cluster_zmin[i]);                 // amax
      lim[6*i+2] = (lim[6*i+1]-lim[6*i+0])/((double) na - 1.);   // da
      lim[6*i+3] = log(limits.k_min_cH0);                        // logkmin
      lim[6*i+4] = log(limits.k_max_cH0);                        // logkmax
      lim[6*i+5] = (lim[6*i+4]-lim[6*i+3])/((double) nlnk - 1.); // dlnk 
    }
  }
  if (fdiff(cache[0], cosmology.random) || 
      fdiff(cache[1], nuisance.random_clusters) ||
      fdiff(cache[2], redshift.random_clusters) ||
      fdiff(cache[3], Ntable.random))
  {
    (void) pcm_1halo(lim[3], lim[0], 0, 1); // init static vars
    #pragma omp parallel for collapse(4) schedule(static,1)
    for (int p=0; p<Cluster.n200_nbin; p++) { 
      for (int l=0; l<redshift.clusters_nbin; l++) { 
        for (int i=0; i<na; i++) { 
          for (int j=0; j<nlnk; j++) { 
            const double ain = lim[6*l+0] + i*lim[6*l+2];
            const double kin = exp(lim[6*l+3] + j*lim[6*l+5]);
            const double tmp = pcm_1halo_nointerp(kin, ain, p, l, 0); 
            table[k*redshift.clusters_nbin+l][i][j] = (tmp<0) ? 0.0 : tmp; 
          }
        }
      }
    }
    // -------------------------------------------------------------------------
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_clusters;
    cache[2] = redshift.random_clusters;
    cache[3] = Ntable.random;
  }
  // -------------------------------------------------------------------------
  if (nl < 0 || nl > Cluster.n200_nbin - 1 ||
      ni < 0 || ni > redshift.clusters_nbin - 1) {
    log_fatal("error in bin number (nl,ni) = [%d,%d]", nl, ni); exit(1); 
  }
  const int q = nl*redshift.clusters_nbin + ni;
  if (q < 0 || q > NSIZE - 1) {
    log_fatal("internal logic error in selecting bin number"); exit(1);
  }
  double res = 0.0;
  const double amin   = lim[6*ni+0];
  const double amax   = lim[6*ni+1];
  const double da     = lim[6*ni+2];
  const double lnkmin = lim[6*ni+3];
  const double lnkmax = lim[6*ni+4];
  const double dlnk   = lim[6*ni+5];
  const double lnk    = log(k)
  return ((a < amin) || (a > amax)) ? 0.0 :
         ((lnk < lnkmin) || (lnk > lnkmax)) ? 0.0 :
  interpol2d(table[q], na, amin, amax, da, a, nlnk, lnkmin, lnkmax, dlnk, lnk);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
