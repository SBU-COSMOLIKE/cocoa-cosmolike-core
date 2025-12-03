#include <assert.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_spline.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "basics.h"
#include "bias.h"
#include "cosmo3D.h"
#include "redshift_spline.h"
#include "redshift_spline_cluster.h"
#include "structs.h"

#include "log.c/src/log.h"

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double get_effective_redmapper_area(const double a) {
  if (redshift.clusters_survey_area_table == NULL) {
    log_fatal("redshift redmapper survey not loaded"); exit(1);
  }
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  const double z = 1./a - 1.;
  const int nz = redshift.clusters_survey_area_nzbins;
  const double* const zv = redshift.clusters_survey_area_table[0];
  const double* const av = redshift.clusters_survey_area_table[1];
  double lim[3];
  lim[0] = zv[0];                                // zmin
  lim[1] = zv[nz-1];                             // zmax
  lim[2] = (lim[1] - lim[0])/((double) nz - 1.); // dz  
  return (z < lim[0] || z > lim[1]) ? 0.0 : 
    interpol1d(av, nz, lim[0], lim[1], lim[2], z);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double pf_cluster_histo_n(double z, const int ni)
{
  if (redshift.clusters_zdist_table == NULL) {
    log_fatal("redshift n(z) not loaded");
    exit(1);
  } 
  double res = 0.0;
  if ((z >= redshift.clusters_zdist_zmin_all) && 
      (z <  redshift.clusters_zdist_zmax_all)) 
  {
    // -------------------------------------------------------------------------
    const int ntomo  = redshift.clusters_nbin;
    const int nzbins = redshift.clusters_nzbins;
    double** tab = redshift.clusters_zdist_table;
    double* z_v  = redshift.clusters_zdist_table[ntomo];
    // -------------------------------------------------------------------------
    const double dz_histo = (z_v[nzbins - 1] - z_v[0]) / ((double) nzbins - 1.);
    const double zhisto_min = z_v[0];
    const double zhisto_max = z_v[nzbins - 1] + dz_histo;
    // -------------------------------------------------------------------------
    const int nj = (int) floor((z - zhisto_min) / dz_histo);
    if (ni < 0 || ni > ntomo-1 || nj < 0 || nj > nzbins-1) {
      log_fatal("invalid bin input (zbin = ni, bin = nj) = (%d, %d)", ni, nj);
      exit(1);
    } 
    res = table[ni][nj];
  }
  return res;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double pz_cluster(double zz, const int nj) 
{
  static double cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static gsl_interp* photoz_splines[MAX_SIZE_ARRAYS+1];

  if (table == NULL || fdiff(cache[0], redshift.random_shear)) {
    // -------------------------------------------------------------------------
    if (table == NULL) {
      for (int i=0; i<MAX_SIZE_ARRAYS+1; i++) {
        photoz_splines[i] = NULL;
      }
    }
    const int ntomo  = redshift.clusters_nbin;
    const int nzbins = redshift.clusters_nzbins;
    if (table != NULL) free(table);
    table = (double**) malloc2d(ntomo+2, nzbins);
    // -------------------------------------------------------------------------
    const double zmin = redshift.clusters_zdist_zmin_all;
    const double zmax = redshift.clusters_zdist_zmax_all;
    const double dz_histo = (zmax - zmin) / ((double) nzbins);  
    for (int k=0; k<nzbins; k++) { // redshift stored at zv = table[ntomo+1]
      table[ntomo+1][k] = zmin + (k + 0.5) * dz_histo;
    }
    // -------------------------------------------------------------------------
    double NORM[MAX_SIZE_ARRAYS];    
    double norm = 0; 
    #pragma omp parallel for reduction( + : norm )
    for (int i=0; i<ntomo; i++) {
      NORM[i] = 0.0;
      for (int k=0; k<nzbins; k++) {    
        const double z = table[ntomo+1][k];  
        NORM[i] += pf_cluster_histo_n(z, i) * dz_histo;
      }
      norm += NORM[i];
    }
    #pragma omp parallel for
    for (int k=0; k<nzbins; k++) { 
      table[0][k] = 0; // store normalization in table[0][:]
      for (int i=0; i<ntomo; i++) {
        const double z = table[ntomo+1][k];
        table[i + 1][k] = pf_cluster_histo_n(z, i)/NORM[i];
        table[0][k] += table[i+1][k] * NORM[i] / norm;
      }
    }
    // -------------------------------------------------------------------------
    for (int i=0; i<ntomo+1; i++) {
      if (photoz_splines[i] != NULL) {
        gsl_interp_free(photoz_splines[i]);
      }
      photoz_splines[i] = malloc_gsl_interp(nzbins);
    }
    #pragma omp parallel for
    for (int i=0; i<ntomo+1; i++) {
      int status = gsl_interp_init(photoz_splines[i], 
                                   table[ntomo+1], // z_v = table[ntomo+1]
                                   table[i], 
                                   nzbins);
      if (status) {
        log_fatal(gsl_strerror(status));
        exit(1);
      }
    }
    cache[0] = redshift.random_clusters;
  }
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  const int ntomo  = redshift.clusters_nbin;
  const int nzbins = redshift.clusters_nzbins;

  if (nj < 0 || nj > ntomo-1) {
    log_fatal("nj = %d bin outside range (max = %d)", nj, ntomo);
    exit(1);
  }

  double res; 
  if (zz < table[ntomo+1][0] || zz > table[ntomo+1][nzbins-1]) { // z_v = table[ntomo+1]
    res = 0.0;
  }
  else {
    int status = gsl_interp_eval_e(photoz_splines[nj+1], 
                                   table[ntomo+1],
                                   table[nj+1],
                                   zz, 
                                   NULL, 
                                   &res);
    if (status) {
      log_fatal(gsl_strerror(status));
      exit(1);
    }
  }
  return res;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double int_norm_zdistr_cluster(const double z, void* params) { 
  double* ar = (double*) params;
  const int ni = (int) ar[0];
  if (!(z>-1)) {
    log_fatal("z>-1 not true"); exit(1);
  }
  const double a = 1./(1. + z);

  // below: int_zmin^zmax dz_obs p(zobs | ztrue)
  const double prob_zobs_in_zbin_given_ztrue = pz_cluster(z, ni);

  struct chis chidchi = chi_all(a);
  const double hoverh0 = hoverh0v2(a, chidchi.dchida);
  const double fK = f_K(chidchi.chi);
  const double omega_mask = get_effective_redmapper_area(a);
  const double dVdz = omega_mask*fK*fK/hoverh0; 

  return dVdz*prob_zobs_in_zbin_given_ztrue;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double norm_zdistr_cluster(const int ni)
{
  static double cache[MAX_SIZE_ARRAYS];
  static double* table;
  static gsl_integration_glfixed_table* w = NULL;

  if (NULL == table || NULL == w || fdiff(cache[0], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double**) malloc1d(redshift.clusters_nbin);
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    const size_t szint = 200 + 50 * (Ntable.high_def_integration);
    w = malloc_gslint_glfixed(szint);
  }
  if (fdiff(cache[0], Ntable.random) || 
      fdiff(cache[1], cosmology.random) ||
      fdiff(cache[2], redshift.random_clusters) ||
      fdiff(cache[3], nuisance.random_photoz_clusters)) 
  {
    { // init static vars only 
      double params[1] = {(double) 0};
      (void) int_norm_zdistr_cluster(tomo.cluster_zmin[0], (void*) params);
    }
    #pragma omp parallel for
    for (int i=0; i<redshift.clusters_nbin; i++) {
      double params[1] = {(double) i};
      gsl_function F;
      F.params = (void*) params;
      F.function = dV_cluster;
      table[i] = gsl_integration_glfixed(&F, tomo.cluster_zmin[ni], 
                                             tomo.cluster_zmax[ni], w);
    }
    cache[0] = Ntable.random;
    cache[1] = cosmology.random;
    cache[2] = redshift.random_clusters;
    cache[3] = nuisance.random_photoz_clusters;
  }
  return table[nz];
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double zdistr_cluster(const double z, const int ni)
{ // func below disregards evolution of N-M relation+mass function within z bin
  double params[1] = {(double) ni};
  return int_norm_zdistr_cluster(z, (void*) params)/norm_zdistr_cluster(ni);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double int_for_g_lens_cluster(double aprime, void* params)
{
  double* ar = (double*) params;
  const int ni = (int) ar[0];
  const double a = ar[1];
  // ---------------------------------------------------------------------------
  const double zprime = 1.0/aprime - 1.0;
  const double chi1 = chi(a);
  const double chiprime = chi(aprime);
  // ---------------------------------------------------------------------------
  const double res = zdistr_cluster(zprime, ni)*f_K(chiprime-chi1)/f_K(chiprime);
  return res/(aprime*aprime);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double g_lens_cluster(const double a, const int ni)
{ 
  static double cache[MAX_SIZE_ARRAYS];
  static double** table = 0;
  static double lim[3];
  static gsl_integration_glfixed_table* w = NULL;
  const int na = Ntable.N_a;
  // ---------------------------------------------------------------------------
  if (table == NULL || fdiff(cache_table_params, Ntable.random)) {
    if (table != NULL) free(table);
    table = (double**) malloc2d(redshift.clusters_nbin+1, na); 
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    const size_t szint = 200 + 50 * (Ntable.high_def_integration);
    w = malloc_gslint_glfixed(szint);  
  }
  if (fdiff(cache[0], Ntable.random) || 
      fdiff(cache[1], cosmology.random) ||
      fdiff(cache[2], redshift.random_clusters) ||
      fdiff(cache[3], redshift.random_photoz_clusters))
  {
    lim[0] = 1.0/(redshift.clusters_zdist_zmax_all + 1.);
    lim[1] = 0.999999;
    lim[2] = (lim[1] - lim[0])/((double) na - 1.);
    // -------------------------------------------------------------------------
    for (int j=-1; j<redshift.clusters_nbin; j++) { // init static vars
      double params[2] = {(double) j, lim[0]} // j = -1: no tomography
      (void) int_for_g_lens_cluster(lim[0], (void*) params);
    }
    // -------------------------------------------------------------------------
    #pragma omp parallel for collapse(2)
    for (int j=-1; j<redshift.clusters_nbin; j++) { 
      for (int i=0; i<na; i++) {
        const double aa = lim[0] + i*lim[2];
        double ar[2] = {(double) j, aa}
        gsl_function F;
        F.params = ar;
        F.function = int_for_g_lens_cluster;
        table[j+1][i] = gsl_integration_glfixed(&F,lim[0], aa, w);
      }      
    } 
    // -------------------------------------------------------------------------
    cache[0] = Ntable.random;
    cache[1] = cosmology.random;
    cache[2] = redshift.random_clusters;
    cache[3] = nuisance.random_photoz_clusters;
  }
  // ---------------------------------------------------------------------------
  if (ni < -1 || ni > redshift.clusters_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  }
  return (a < lim[0] || a > lim[1]) ? 0.0 :
    interpol1d(table[ni+1], na, lim[0], lim[1], lim[2], a);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

int test_zoverlap_cg(const int ni, const int nj)
{
  if (ni < 0 || ni > redshift.clusters_nbin - 1 || 
      nj < 0 || nj > redshift.clustering_nbin - 1) {
    log_fatal("error in bin number (ni, nj) = [%d, %d]", ni, nj);
    exit(1);
  }
  if (tomo.cg_exclude != NULL) {
    static int N[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS] = {{-42}};
    if (N[0][0] < -1) {
      for (int i=0; i<redshift.clusters_nbin; i++) {
        for (int j=0; j<redshift.clustering_nbin; j++) {
          N[i][j] = 1;
          for (int k=0; k<tomo.n_cg_exclude; k++) {
            const int p = k*2+0;
            const int q = k*2+1;
            if ((i == tomo.cg_exclude[p]) && 
                (j == tomo.cg_exclude[q])) {
              N[i][j] = 0;
              break;
            }
          }
        }
      } 
    }
    return  N[ni][nj];
  }
  else {
    return 1;
  }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

int ZCG1(const int ni)
{
  static int N[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS] = {-42};
  if (N[0] < -1) {
    int n = 0;
    for (int i=0; i<redshift.clusters_nbin; i++) {
      for (int j=0; j<redshift.clustering_nbin; j++) {
        if (test_zoverlap_cg(i, j)) {
          N[n] = i;
          n++;
        }
      }
    }
  }
  if (ni < 0 || ni > tomo.cg_clustering_Npowerspectra - 1) {
    log_fatal("error in cg bin number nc = %d", ni);
    exit(1);
  }
  return N[ni];
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

int ZCG2(const int nj)
{
  static int N[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS] = {-42};
  if (N[0] < -1) {
    int n = 0;
    for (int i=0; i<redshift.clusters_nbin; i++) {
      for (int j=0; j<redshift.clustering_nbin; j++) {
        if (test_zoverlap_cg(i, j)) {
          N[n] = j;
          n++;
        }
      }
    }
  }
  if (nj < 0 || nj > tomo.cg_clustering_Npowerspectra - 1) {
    log_fatal("error in cg bin number ng = %d", nj);
    exit(1);
  }
  return N[nj];
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

int NCG(const int ni, const int nj)
{ 
  static int N[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS] = {{-42}};
  if (N[0][0] < 0) {
    int n = 0;
    for (int i=0; i<redshift.clusters_nbin; i++) {
      for (int j=0; j<redshift.clustering_nbin; j++) {
        if (test_zoverlap_cg(i,j)) {
          N[i][j] = n;
          n++;
        } 
        else {
          N[i][j] = -1;
        }
      }
    }
  }
  if (ni < 0 || ni > redshift.clusters_nbin - 1 || 
      nj < 0 || nj > redshift.clustering_nbin - 1)
  {
    log_fatal("error in cg bin number (ni,nj) = [%d,%d]",ni,nj);
    exit(1);
  }
  return N[ni][nj];
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

int test_zoverlap_cggl(const int ni, const int nj)
{
  if (ni < 0 || ni > redshift.clusters_nbin - 1 || 
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("invalid bin cs input (ni, nj) = [%d, %d]", ni, nj);
    exit(1);
  }
  if (tomo.cggl_exclude != NULL) {
    static int N[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS] = {{-42}};
    if (N[0][0] < -1) {
      for (int i=0; i<redshift.clustering_nbin; i++) {
        for (int j=0; j<redshift.shear_nbin; j++) {
          N[i][j] = 1;
          for (int k=0; k<tomo.n_cggl_exclude; k++) {
            const int p = k*2+0;
            const int q = k*2+1;
            if ((i == tomo.cggl_exclude[p]) && 
                (j == tomo.cggl_exclude[q])) {
              N[i][j] = 0;
              break;
            }
          }
        }
      } 
    }
    return  N[ni][nj];
  }
  else {
    return 1;
  }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

int ZCL(const int ni) 
{
  static int N[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS] = {-42};
  if (N[0] < -1) {
    int n = 0;
    for (int i=0; i<redshift.clusters_nbin; i++) {
      for (int j=0; j<redshift.shear_nbin; j++) {
        if (test_zoverlap_cggl(i, j)) {
          N[n] = i;
          n++;
        }
      }
    }
  }
  if (ni < 0 || ni > tomo.cgl_Npowerspectra - 1) {
    log_fatal("error in cs bin number ni = %d", ni);
    exit(1);
  }
  return N[ni];
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

int ZCS(const int ni) 
{
  static int N[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS] = {-42};
  if (N[0] < -1) 
  {
    int n = 0;
    for (int i=0; i<redshift.clusters_nbin; i++) {
      for (int j=0; j<redshift.shear_nbin; j++) {
        if (test_zoverlap_cggl(i,j)) {
          N[n] = j;
          n++;
        }
      }
    }
  }
  if (ni < 0 || ni > tomo.cgl_Npowerspectra - 1) {
    log_fatal("error in cs bin number ni = %d", ni);
    exit(1);
  }
  return N[ni];
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

int NCGL(const int ni, const int nj) 
{
  static int N[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS] = {{-42}};
  if (N[0][0] < 0) {
    int n = 0;
    for (int i=0; i<redshift.clusters_nbin; i++) {
      for (int j=0; j<redshift.shear_nbin; j++) {
        if (test_zoverlap_cggl(i,j)) {
          N[i][j] = n;
          n++;
        } 
        else {
          N[i][j] = -1;
        }
      }
    }
  }
  if (ni < 0 || ni > redshift.clusters_nbin - 1 || 
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input (ni, nj) = [%d, %d]", ni, nj);
    exit(1);
  }
  return N[ni][nj];
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
