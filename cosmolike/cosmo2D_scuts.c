#include <assert.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gsl/gsl_sum.h>
#include <gsl/gsl_integration.h>
#include "../cfftlog/cfftlog.h"
#include <fftw3.h>

#include "bias.h"
#include "basics.h"
#include "cfastpt/cfastpt.h"
#include "cosmo3D.h"
#include "cosmo2D.h"
#include "cosmo2D_scuts.h"
#include "halo.h"
#include "IA.h"
#include "pt_cfastpt.h"
#include "radial_weights.h"
#include "redshift_spline.h"
#include "structs.h"
#include "log.c/src/log.h"

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double dC_ss_dlnk_tomo_limber_nointerp(
    const double k, 
    const double l,
    const int ni, 
    const int nj, 
    const int EE
  )
{
  if (!(k>0)) {
    log_fatal("k>0 not true"); exit(1);
  }
  if (ni < -1 || ni > redshift.shear_nbin -1 || 
      nj < -1 || nj > redshift.shear_nbin -1) {
    log_fatal("invalid bin input (ni, nj) = (%d, %d)", ni, nj); exit(1);
  }
  // First: determine the scale factor such as chi(a) = (l + 1/2)/k
  const double ell = l + 0.5;
  // convert from (Mpc/h)^{-1} to ((Mpc/h)/(c/H0=100)^3)^{-1}
  const double a = a_chi(f_K(ell/(k*cosmology.coverH0)));
  const double amin = 1./(redshift.shear_zdist_zmax_all+1.);
  const double amax = 1./(1.+fmax(redshift.shear_zdist_zmin_all,1e-6));
  // Second: compute dCXY/dlnk
  double ans = 0.0;
  if (a > amin || a < amax) {
    double ar[5] = {(double) ni, 
                    (double) nj, 
                    l, 
                    (double) EE, 
                    (double) 1}; // last argument: get derivative
    ans = int_for_C_ss_tomo_limber(a, (void*) ar);
  }
  return ans;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double dlnC_ss_dlnk_tomo_limber_nointerp(
    const double k, 
    const double l,
    const int ni, 
    const int nj, 
    const int EE, 
    const int init
  )
{
  const double dC_ss_dlnk = dC_ss_dlnk_tomo_limber_nointerp(k, l, ni, nj, EE);
  const double C_ss = (fabs(dC_ss_dlnk) > 1.e-50) ? 
                           C_ss_tomo_limber_nointerp(l, ni, nj, EE, init) : 1.0;
  return (fabs(C_ss) > 1.e-50) ? dC_ss_dlnk/C_ss : 0.0;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double dC_ss_dlnk_tomo_limber(
    const double k,
    const double l, 
    const int ni, 
    const int nj, 
    const int EE
  )
{
  static double cache[MAX_SIZE_ARRAYS];
  static double**** table;
  static double lim[6];
  static int nell;
  static int nlnk;
  
  if (NULL == table || fdiff(cache[4], Ntable.random)) {
    nell = Ntable.N_ell;
    lim[0] = log(fmax(limits.LMIN_tab - 1., 1.0));
    lim[1] = log(Ntable.LMAX + 1.);
    lim[2] = (lim[1] - lim[0]) / ((double) nell - 1.);

    nlnk = Ntable.dCX_dlnk_nlnk;
    lim[3] = log(Ntable.dCX_dlnk_kmin);
    lim[4] = log(Ntable.dCX_dlnk_kmax);
    lim[5] = (lim[4] - lim[3]) / ((double) nlnk - 1.);

    if (table != NULL) free(table);
    table = (double****) malloc4d(2, tomo.shear_Npowerspectra, nlnk, nell);
  }
  if (fdiff(cache[0], cosmology.random) ||
      fdiff(cache[1], nuisance.random_photoz_shear) ||
      fdiff(cache[2], nuisance.random_ia) ||
      fdiff(cache[3], redshift.random_shear) ||
      fdiff(cache[4], Ntable.random))
  {
    // init static vars
    (void) dC_ss_dlnk_tomo_limber_nointerp(exp(lim[3]), exp(lim[0]),
                                                               Z1(0), Z2(0), 1);  
    #pragma omp parallel for collapse(3) schedule(static,1)
    for (int f=0; f<nlnk; f++) {  
      for (int p=0; p<tomo.shear_Npowerspectra; p++) {  
        for (int i=0; i<nell; i++) { 
          const double kin = exp(lim[3] + f*lim[5]);
          const double l = exp(lim[0] + i*lim[2]);
          const double Z1NZ = Z1(p);
          const double Z2NZ = Z2(p);
          table[0][p][f][i] = dC_ss_dlnk_tomo_limber_nointerp(kin, l, Z1NZ, Z2NZ, 1);
          table[1][p][f][i] = dC_ss_dlnk_tomo_limber_nointerp(kin,l,Z1NZ,Z2NZ,0);
        }
      }
    }
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_shear;
    cache[4] = Ntable.random;
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1 || 
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number (ni,nj) = [%d,%d]",ni,nj); exit(1);
  }
  const int q = N_shear(ni, nj);
  if (q < 0 || q > tomo.shear_Npowerspectra - 1) {
    log_fatal("internal logic error in selecting bin number"); exit(1);
  }
  const double lnl = log(l);
  const double lnk = log(k);
  return (lnk < lim[3] || lnk > lim[4]) ? 0.0 :
         (lnl < lim[0] || lnl > lim[1]) ? 0.0 :
         interpol2d((1==EE) ? table[0][q] : table[1][q],
                    nlnk, lim[3], lim[4], lim[5], lnk,
                    nell, lim[0], lim[1], lim[2], lnl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double dlnC_ss_dlnk_tomo_limber(
    const double k,
    const double l, 
    const int ni, 
    const int nj, 
    const int EE
  )
{
  static double cache[MAX_SIZE_ARRAYS];
  static double**** table;
  static double lim[6];
  static int nell;
  static int nlnk;
  
  if (NULL == table || fdiff(cache[4], Ntable.random)) {
    nell = Ntable.N_ell;
    lim[0] = log(fmax(limits.LMIN_tab - 1., 1.0));
    lim[1] = log(Ntable.LMAX + 1.);
    lim[2] = (lim[1] - lim[0]) / ((double) nell - 1.);

    nlnk = Ntable.dCX_dlnk_nlnk;
    lim[3] = log(Ntable.dCX_dlnk_kmin);
    lim[4] = log(Ntable.dCX_dlnk_kmax);
    lim[5] = (lim[4] - lim[3]) / ((double) nlnk - 1.);

    if (table != NULL) free(table);
    table = (double****) malloc4d(2, tomo.shear_Npowerspectra, nlnk, nell);
  }
  if (fdiff(cache[0], cosmology.random) ||
      fdiff(cache[1], nuisance.random_photoz_shear) ||
      fdiff(cache[2], nuisance.random_ia) ||
      fdiff(cache[3], redshift.random_shear) ||
      fdiff(cache[4], Ntable.random))
  {
    // init static vars
    (void) dlnC_ss_dlnk_tomo_limber_nointerp(exp(lim[3]), 
                                             exp(lim[0]), Z1(0), Z2(0), 1, 1);
    #pragma omp parallel for collapse(3) schedule(static,1)
    for (int f=0; f<nlnk; f++) {  
      for (int p=0; p<tomo.shear_Npowerspectra; p++) {  
        for (int i=0; i<nell; i++) { 
          const double kin = exp(lim[3] + f*lim[5]);
          const double l   = exp(lim[0] + i*lim[2]);
          const double Z1NZ = Z1(p);
          const double Z2NZ = Z2(p);
          table[0][p][f][i] = dlnC_ss_dlnk_tomo_limber_nointerp(kin, l, Z1NZ, Z2NZ, 1, 0);
          table[1][p][f][i] = dlnC_ss_dlnk_tomo_limber_nointerp(kin, l, Z1NZ, Z2NZ, 0, 0);
        }
      }
    }
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_shear;
    cache[4] = Ntable.random;
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1 || 
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number (ni, nj) = [%d,%d]", ni, nj);
    exit(1);
  }
  const int q = N_shear(ni, nj);
  if (q < 0 || q > tomo.shear_Npowerspectra - 1) {
    log_fatal("internal logic error in selecting bin number"); exit(1);
  }
  const double lnl = log(l);
  const double lnk = log(k);
  return (lnk < lim[3] || lnk > lim[4] || lnl < lim[0] || lnl > lim[1]) ? 0.0 : 
    interpol2d((1==EE) ? table[0][q] : table[1][q],
          nlnk, lim[3], lim[4], lim[5], lnk, nell, lim[0], lim[1], lim[2], lnl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// physical mode cut-off (R = response function): 
// find kk such that RF \equiv \int_{-\infty}^{ln(kk)} dlnk |dlnXdlnk| = alpha
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_RF_C_ss(double t, void* params) 
{ // \int_{-infty}^{b} dlnk f(lnk) = \int_0^1 f(b-(1-t)/t)/t^2 
  double* ar = (double*) params;
  const double l = ar[0];
  const int ni = (int) ar[1];
  const int nj = (int) ar[2];
  if (ni < 0 || ni > redshift.shear_nbin - 1 || 
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number (ni,nj) = [%d,%d]", ni, nj); 
    exit(1);
  }
  const int EE = (int) ar[3];
  const double lnkmax = ar[4];
  const double kin = exp(lnkmax - (1-t)/t);
  double f1;
  if (l > limits.LMIN_tab) {
    f1 = fabs(dlnC_ss_dlnk_tomo_limber(kin, l, ni, nj, EE));
  }
  else {
    f1 = fabs(dlnC_ss_dlnk_tomo_limber_nointerp(kin, l, ni, nj, EE, 0));
  }
  return f1/(t*t);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_norm_RF_C_ss(double t, void* params) 
{ // \int_{-infty}^{infty} dlnk f(lnk) = \int_0^1 (f((1-t)/t)+f(-(1-t)/t))/t^2 
  double* ar = (double*) params;
  const double l = ar[0];
  const int ni = (int) ar[1];
  const int nj = (int) ar[2];
  if (ni < 0 || ni > redshift.shear_nbin - 1 || 
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number (ni,nj) = [%d,%d]", ni, nj); exit(1);
  }
  const int EE = (int) ar[3];
  const double lnkmax = ar[4];
  double f1 = 0.0;
  double f2 = 0.0;
  double kin = 0.0;
  if (l > limits.LMIN_tab) {
    kin = exp((1.-t)/t);
    f1 = fabs(dlnC_ss_dlnk_tomo_limber(kin, l, ni, nj, EE));
    kin = exp(-(1.-t)/t);
    f2 = fabs(dlnC_ss_dlnk_tomo_limber(kin, l, ni, nj, EE));
  }
  else {
    kin = exp((1.-t)/t);
    f1 = fabs(dlnC_ss_dlnk_tomo_limber_nointerp(kin, l, ni, nj, EE, 0));
    kin = exp(-(1.-t)/t);
    f2 = fabs(dlnC_ss_dlnk_tomo_limber_nointerp(kin, l, ni, nj, EE, 0));
  }
  return (f1 + f2)/(t*t);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double RF_C_ss_nointerp(
    const double kmax,
    const double l, 
    const int ni, 
    const int nj, 
    const int EE, 
    const int init
  ) // compute R_X = \int_{-infty}^{kmax} dlnk |dlnX_dlnk|
{
  static double cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL; 
  if (NULL == w || fdiff(cache[0], Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 256 : 
                         (1 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }
  double ar[5] = {l, (double) ni, (double) nj, (double) EE, log(kmax)};
  double res = 0.0;
  if (1 == init) {
    (void) int_RF_C_ss(1e-1, (void*) ar);
    (void) int_norm_RF_C_ss(1e-1, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_RF_C_ss;
    const double num = gsl_integration_glfixed(&F, 1e-6, 1.0, w);
    F.function = int_norm_RF_C_ss;
    const double den = gsl_integration_glfixed(&F, 1e-6, 1.0, w);
    res = num/den;
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double RF_C_ss(
    const double kmax, 
    const double l, 
    const int ni, 
    const int nj,
    const int EE
  ) 
{
  static double cache[MAX_SIZE_ARRAYS];
  static double**** table;
  static double lim[6];
  static int nell;
  static int nlnk;
  
  if (NULL == table || fdiff(cache[4], Ntable.random)) {
    nell = Ntable.N_ell;
    lim[0] = log(fmax(limits.LMIN_tab - 1., 1.0));
    lim[1] = log(Ntable.LMAX + 1.);
    lim[2] = (lim[1] - lim[0]) / ((double) nell - 1.);

    nlnk = Ntable.dCX_dlnk_nlnk;
    lim[3] = log(1e-4);
    lim[4] = log(100.0);
    lim[5] = (lim[4] - lim[3]) / ((double) nlnk - 1.);

    if (table != NULL) free(table);
    table = (double****) malloc4d(2, tomo.shear_Npowerspectra, nlnk, nell);
  }
  if (fdiff(cache[0], cosmology.random) ||
      fdiff(cache[1], nuisance.random_photoz_shear) ||
      fdiff(cache[2], nuisance.random_ia) ||
      fdiff(cache[3], redshift.random_shear) ||
      fdiff(cache[4], Ntable.random))
  {
    // init static vars
    (void) RF_C_ss_nointerp(exp(lim[3]), exp(lim[0]), Z1(0), Z2(0), 1, 1); 
    #pragma omp parallel for collapse(3) schedule(static,1)
    for (int f=0; f<nlnk; f++) {  
      for (int p=0; p<tomo.shear_Npowerspectra; p++) {  
        for (int i=0; i<nell; i++) { 
          const double kin = exp(lim[3] + f*lim[5]);
          const double l = exp(lim[0] + i*lim[2]);
          const double Z1NZ = Z1(p);
          const double Z2NZ = Z2(p);
          table[0][p][f][i] = RF_C_ss_nointerp(kin, l, Z1NZ, Z2NZ, 1, 0);
          table[1][p][f][i] = 0.0; // for not I am ignoring TATT
                           //RF_C_ss_nointerp(kin,(double) l, Z1NZ, Z2NZ, 0, 0);
        }
      }
    } 
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_shear;
    cache[4] = Ntable.random;
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1 || 
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number (ni, nj) = [%d,%d]", ni, nj); exit(1);
  }
  const int q = N_shear(ni, nj);
  if (q < 0 || q > tomo.shear_Npowerspectra - 1) {
    log_fatal("internal logic error in selecting bin number"); exit(1);
  }
  const double lnl = log(l);
  const double lnk = log(kmax);
  return (lnk < lim[3] || lnk > lim[4]) ? 0.0 : 
         (lnl < lim[0] || lnl > lim[1]) ? 0.0 : 
         interpol2d((1==EE) ? table[0][q] : table[1][q],
                    nlnk, lim[3], lim[4], lim[5], lnk, 
                    nell, lim[0], lim[1], lim[2], lnl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double** dlnxi_dlnk_pm_tomo(const double k)
{  
  static double*** Glpm = NULL; //Glpm[0] = Gl+, Glpm[1] = Gl-
  static double cache[MAX_SIZE_ARRAYS];
  static double**** dCldlnk = NULL;
  static double lim[3];
  static int nlnk;
  const int lmin = 1;
  //Ntable.LMAX = 10000;
  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized"); exit(1);
  }
  const int NSIZE = tomo.shear_Npowerspectra;
  if (NULL == Glpm ||  NULL == dCldlnk || fdiff(cache[4], Ntable.random))
  {
    printf("what\n\n\n");
    if (Glpm != NULL) free(Glpm);
    Glpm = (double***) malloc3d(2, Ntable.Ntheta, Ntable.LMAX);
    
    double*** P = (double***) malloc3d(4, Ntable.Ntheta, Ntable.LMAX + 1);
    double** Pmin  = P[0]; double** Pmax  = P[1];
    double** dPmin = P[2]; double** dPmax = P[3];

    double xmin[Ntable.Ntheta];
    double xmax[Ntable.Ntheta];
    for (int i=0; i<Ntable.Ntheta; i++)
    { // Cocoa: dont thread (init of static variables inside set_bin_average)
      bin_avg r = set_bin_average(i, 0);
      xmin[i] = r.xmin;
      xmax[i] = r.xmax;
    }
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<(Ntable.LMAX+1); l++) {
        bin_avg r   = set_bin_average(i, l);
        Pmin[i][l]  = r.Pmin;
        Pmax[i][l]  = r.Pmax;
        dPmin[i][l] = r.dPmin;
        dPmax[i][l] = r.dPmax;
      }
    }
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<lmin; l++) {
        Glpm[0][i][l] = 0.0;
        Glpm[1][i][l] = 0.0;
      }
    }
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=lmin; l<Ntable.LMAX; l++) {
        Glpm[0][i][l] = (2.*l+1)/(2.*M_PI*l*l*(l+1)*(l+1))*(
          -l*(l-1.)/2*(l+2./(2*l+1)) * (Pmin[i][l-1]-Pmax[i][l-1])
          -l*(l-1.)*(2.-l)/2 * (xmin[i]*Pmin[i][l]-xmax[i]*Pmax[i][l])
          +l*(l-1.)/(2.*l+1) * (Pmin[i][l+1]-Pmax[i][l+1])
          +(4-l)*(dPmin[i][l]-dPmax[i][l])
          +(l+2)*(xmin[i]*dPmin[i][l-1] - xmax[i]*dPmax[i][l-1] - Pmin[i][l-1] + Pmax[i][l-1])
          +2*(l-1)*(xmin[i]*dPmin[i][l] - xmax[i]*dPmax[i][l] - Pmin[i][l] + Pmax[i][l])
          -2*(l+2)*(dPmin[i][l-1]-dPmax[i][l-1])
        )/(xmin[i]-xmax[i]);

        Glpm[1][i][l] = (2.*l+1)/(2.*M_PI*l*l*(l+1)*(l+1))*(
          -l*(l-1.)/2*(l+2./(2*l+1)) * (Pmin[i][l-1]-Pmax[i][l-1])
          -l*(l-1.)*(2.-l)/2 * (xmin[i]*Pmin[i][l]-xmax[i]*Pmax[i][l])
          +l*(l-1.)/(2.*l+1)* (Pmin[i][l+1]-Pmax[i][l+1])
          +(4-l)*(dPmin[i][l]-dPmax[i][l])
          +(l+2)*(xmin[i]*dPmin[i][l-1] - xmax[i]*dPmax[i][l-1] - Pmin[i][l-1] + Pmax[i][l-1])
          -2*(l-1)*(xmin[i]*dPmin[i][l] - xmax[i]*dPmax[i][l] - Pmin[i][l] + Pmax[i][l])
          +2*(l+2)*(dPmin[i][l-1]-dPmax[i][l-1])
          )/(xmin[i]-xmax[i]);
      }
    }
    free(P);
    
    nlnk = Ntable.dCX_dlnk_nlnk;
    lim[0] = log(Ntable.dCX_dlnk_kmin);
    lim[1] = log(Ntable.dCX_dlnk_kmax);
    lim[2] = (lim[1] - lim[0]) / ((double) nlnk - 1.);
    if (dCldlnk != NULL) free(dCldlnk);
    dCldlnk = (double****) malloc4d(2, NSIZE, Ntable.LMAX, nlnk);
  }
  if (fdiff(cache[0], cosmology.random) ||
      fdiff(cache[1], nuisance.random_photoz_shear) ||
      fdiff(cache[2], nuisance.random_ia) ||
      fdiff(cache[3], redshift.random_shear) ||
      fdiff(cache[4], Ntable.random))
  {
    for (int q=0; q<nlnk; q++) {
      for (int i=0; i<NSIZE; i++) {
        for (int l=0; l<lmin; l++) {
          dCldlnk[0][i][l][q] = 0.0;
          dCldlnk[1][i][l][q] = 0.0;
        }
      }
    }
    // init static vars
    (void) dC_ss_dlnk_tomo_limber(Ntable.dCX_dlnk_kmin, 
                                 (double) limits.LMIN_tab + 1, 
                                 Z1(0), Z2(0), 1);
    #pragma omp parallel for collapse(4) schedule(static,1)
    for (int p=0; p<2; p++) {
      for (int q=0; q<nlnk; q++)  {
        for (int nz=0; nz<NSIZE; nz++)  {
          for (int l=lmin; l<limits.LMIN_tab; l++) {
            const double kin = exp(lim[0] + q * lim[2]);
            dCldlnk[p][nz][l][q] = 
            dC_ss_dlnk_tomo_limber_nointerp(kin, (double) l, Z1(nz), Z2(nz), 1-p);
          }
        }
      }
    }
    #pragma omp parallel for collapse(4) schedule(static,1)
    for (int p=0; p<2; p++) {
      for (int q=0; q<nlnk; q++)  {
        for (int nz=0; nz<NSIZE; nz++) {
          for (int l=limits.LMIN_tab; l<Ntable.LMAX; l++) {
            const double kin = exp(lim[0] + q * lim[2]);
            dCldlnk[p][nz][l][q] = 
                   dC_ss_dlnk_tomo_limber(kin, (double) l, Z1(nz), Z2(nz), 1-p);
          }
        }
      }
    }
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_shear;
    cache[4] = Ntable.random;
  }
  double** ans = (double**) malloc2d(2, NSIZE*Ntable.Ntheta);
  #pragma omp parallel for collapse(3) schedule(static,1)
  for (int p=0; p<2; p++) {
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        const int q = nz * Ntable.Ntheta + i;
        ans[p][q] = 0.0;
      }
    }
  }
  const double lnk = log(k);
  if (lnk > lim[0] || lnk < lim[1]) {
    double*** cx = (double***) malloc3d(2, NSIZE, Ntable.LMAX);
    for (int nz=0; nz<NSIZE; nz++) {
      for (int l=0; l<lmin; l++) {
        cx[0][nz][l] = 0.0;
        cx[1][nz][l] = 0.0;
      }
    }
    #pragma omp parallel for collapse(3) schedule(static,1)
    for (int p=0; p<2; p++) {
      for (int nz=0; nz<NSIZE; nz++) {
        for (int l=lmin; l<Ntable.LMAX; l++) {
          cx[p][nz][l] = interpol1d(dCldlnk[p][nz][l], nlnk, lim[0], lim[1], lim[2], lnk);
        } 
      } 
    }
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        const int q = nz * Ntable.Ntheta + i;
        double sum0 = 0.0;
        double sum1 = 0.0;   
        for (int l=lmin; l<Ntable.LMAX; l++) {
          const double c0 = cx[0][nz][l];
          const double c1 = cx[1][nz][l];
          sum0 += Glpm[0][i][l] * (c0 + c1); 
          sum1 += Glpm[1][i][l] * (c0 - c1);
        }
        ans[0][q] = sum0;
        ans[1][q] = sum1;
      } 
    }
    for (int p=0; p<2; p++) {
      for (int nz=0; nz<NSIZE; nz++) {
        for (int i=0; i<Ntable.Ntheta; i++) {
          const int q = nz * Ntable.Ntheta + i;
          const double dxipmdlnk = ans[p][q]; 
          if (fabs(ans[p][q])>1.e-50) {
            const double xipm = xi_pm_tomo(p, i, Z1(nz), Z2(nz), 1);
            ans[p][q] = (fabs(xipm) > 1.e-50) ? dxipmdlnk/xipm : 0.0;
          }
        }
      }
    }
    free(cx);
  }
  return ans;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
