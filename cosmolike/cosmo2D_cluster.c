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
#include "cluster_util.h"
#include "cosmo2D.h"
#include "cosmo2D_cluster.h"
#include "radial_weights.h"
#include "recompute.h"
#include "redshift_spline.h"
#include "structs.h"

#include "log.c/src/log.h"

static int include_exclusion = 0; // 0 or 1
static int adopt_dark_emulator = 0; // 0 or 1

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Cluster number counts
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double int_Ncl(double a, void* params)
{
  double *ar = (double*) params;
  const int nl = (int) ar[0];
  const int ni = (int) ar[1];
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  const double z = 1./a - 1.;
  const double dzda = 1./(a*a);
  
  // int_zmin^zmax dz_obs p(zobs | ztrue)
  const double prob_zobs_in_zbin_given_ztrue = pz_cluster(z, ni);

  struct chis chidchi = chi_all(a);
  const double hoverh0 = hoverh0v2(a, chidchi.dchida);
  const double fK = f_K(chidchi.chi);
  const double omega_mask = get_effective_redmapper_area(a);
  const double dVdztrue = fK*fK*omega_mask/hoverh0;

  return dVdztrue*prob_zobs_in_zbin_given_ztrue*dzda*
                              ncl_given_lambda_obs_within_nl_given_ztrue(nl, a);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double Ncl_nointerp(
    const int nl, 
    const int ni, 
    const int init
  )
{
  static double cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;
  if (nl < 0 || nl > Cluster.n200_nbin - 1) {
    log_fatal("error in bin number (nl) = %d", nl); exit(1); 
  }
  if (ni < 0 || ni > redshift.clusters_nbin - 1) {
    log_fatal("error in bin number (ni) = %d", nl); exit(1); 
  }
  if (NULL == w || fdiff(cache[0], Ntable.random)) {
    const size_t szint = 50 + 40 * abs(Ntable.high_def_integration);
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  const int amin = 1./(1 + tomo.cluster_zmax[ni]); // integration on zobs
  const int amax = 1./(1 + tomo.cluster_zmin[ni]); // integration on zobs
  double ar[1] = {(double) nl, (double) ni};
  
  double res = 0.0; 
  if (1 == init) { 
    (void) int_Ncl(aobsmin, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_Ncl;
    res = gsl_integration_glfixed(&F, amin, amax, w);
  }
  return res;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double Ncl(const int nl, const int ni)
{
  static double cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  // ---------------------------------------------------------------------------
  if (NULL == table || fdiff(cache[3], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double**) malloc2d(Cluster.n200_nbin,redshift.clusters_nbin);  
  }
  if (fdiff(cache[0], cosmology.random) || 
      fdiff(cache[1], nuisance.random_clusters) ||
      fdiff(cache[2], redshift.random_clusters) ||
      fdiff(cache[3], Ntable.random) ||
      fdiff(cache[4], nuisance.random_photoz_clusters))
  {
    (void) Ncl_nointerp(0, 0, 1); // init static vars
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Cluster.n200_nbin; i++) {
      for (int j=0; j<redshift.clusters_nbin; j++) {
        table[i][j] = Ncl_nointerp(i, j, 0);
      }
    }
    // -------------------------------------------------------------------------
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_clusters;
    cache[2] = redshift.random_clusters;
    cache[3] = Ntable.random;
    cache[4] = nuisance.random_photoz_clusters;
  }
  return table[nl][ni]; 
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// 2pt correlation function
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double w_cs_tomo(
    const int nt, 
    const int nl, 
    const int ni, 
    const int nj, 
    const int limber
  )
{
  static double** Pl = NULL;
  static double* w_vec = NULL;
  static double** Cl = NULL;
  static double cache[MAX_SIZE_ARRAYS];
  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized");
    exit(1);
  }
  const int NSIZE = Cluster.n200_nbin*tomo.cs_npowerspectra;
  if (NULL == Pl || NULL == w_vec || 
      NULL == Cl || fdiff(cache[6], Ntable.random))
  {
    const int lmin = 1;    
    if (Pl != NULL) free(Pl);
    Pl = (double**) malloc2d(like.Ntheta, limits.LMAX);
    if (w_vec != NULL) free(w_vec);
    w_vec = (double*) calloc1d(NSIZE*like.Ntheta);
    double*** P = (double***) malloc3d(2, like.Ntheta, limits.LMAX + 1);
    double** Pmin  = P[0]; double** Pmax  = P[1];
    // -------------------------------------------------------------------------
    double xmin[like.Ntheta];
    double xmax[like.Ntheta];
    for (int i=0; i<like.Ntheta; i++)
    { // Cocoa: dont thread (init of static variables inside set_bin_average)
      bin_avg r = set_bin_average(i,0);
      xmin[i] = r.xmin;
      xmax[i] = r.xmax;
    }
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<like.Ntheta; i ++) {
      for (int l=0; l<(limits.LMAX+1); l++) {
        bin_avg r = set_bin_average(i, l);
        Pmin[i][l] = r.Pmin;
        Pmax[i][l] = r.Pmax;
      }
    }
    // -------------------------------------------------------------------------
    for (int i=0; i<like.Ntheta; i++) {
      for (int l=0; l<lmin; l++) {
        Pl[i][l] = 0.0;
      }
    }
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<like.Ntheta; i++) {
      for (int l=lmin; l<limits.LMAX; l++) {
        Pl[i][l] = (2.*l+1)/(4.*M_PI*l*(l+1)*(xmin[i]-xmax[i]))
          *((l+2./(2*l+1.))*(Pmin[l-1]-Pmax[l-1])
          +(2-l)*(xmin[i]*Pmin[l]-xmax[i]*Pmax[l])
          -2./(2*l+1.)*(Pmin[l+1]-Pmax[l+1]));
      }
    }
    // -------------------------------------------------------------------------
    free(P);
    if (Cl != NULL) free(Cl);
    Cl = (double**) malloc2d(NSIZE, limits.LMAX);
  }
  if (fdiff(cache[0], cosmology.random) ||
      fdiff(cache[1], nuisance.random_photoz_shear) ||
      fdiff(cache[2], nuisance.random_photoz_clusters) ||
      fdiff(cache[3], nuisance.random_ia) ||
      fdiff(cache[4], redshift.random_shear) ||
      fdiff(cache[5], redshift.random_clusters) ||
      fdiff(cache[6], Ntable.random) ||
      fdiff(cache[7], nuisance.random_clusters))
  {
    const int lmin = 1;
    for (int i=0; i<NSIZE; i++) {
      for (int l=0; l<lmin; l++) {
        Cl[i][l] = 0.0;
      }
    } 
    (void) C_cs_tomo_limber(limits.LMIN_tab+1,0,ZCL(0),ZCS(0)); // init static vars
    // -------------------------------------------------------------------------
    if (1 == limber) { 
      #pragma omp parallel for collapse(3) 
      for (int i=0; i<Cluster.n200_nbin; i++) {  
        for (int j=0; j<tomo.cs_npowerspectra; j++) { 
          for (int l=lmin; l<limits.LMIN_tab; l++) {
            const int q = i*tomo.cs_npowerspectra + j;
            Cl[q][l] = C_cs_tomo_limber_nointerp(l, i, ZCL(j), ZCS(j), 0);
          }
        }    
      } 
      #pragma omp parallel for collapse(3) 
      for (int i=0; i<Cluster.n200_nbin; i++) {  
        for (int j=0; j<tomo.cs_npowerspectra; j++) { 
          for (int l=lmin; l<limits.LMIN_tab; l++) {
            const int q = i*tomo.cs_npowerspectra + j;
            Cl[q][l] = C_cs_tomo_limber(l, i, ZCL(j), ZCS(j));
          }
        }    
      } 
    }
    else { 
      log_fatal("NonLimber not implemented"); exit(1);     
    }
    // -------------------------------------------------------------------------
    #pragma omp parallel for collapse(3)
    for (int i=0; i<Cluster.n200_nbin; i++) {
      for (int j=0; j<tomo.cs_npowerspectra; j++) { 
        for (int p=0; p<like.Ntheta; p++) {
          const int nz = i*tomo.cs_npowerspectra + j;
          double sum = 0.0;
          for (int l=lmin; l<limits.LMAX; l++) {
            sum += Pl[p][l]*Cl[nz][l];
          }
          w_vec[nz*like.Ntheta + p] = sum;
        }
      }
    }
    // -------------------------------------------------------------------------
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_photoz_clusters;
    cache[3] = nuisance.random_ia;
    cache[4] = redshift.random_shear;
    cache[5] = redshift.random_clusters;
    cache[6] = Ntable.random;
    cache[7] = nuisance.random_clusters;
  }
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  if (nl < 0 || nl > Cluster.n200_nbin - 1 ||
      ni < 0 || ni > redshift.clusters_nbin - 1 ||
      nj < 0 || nj > redshift.shear_nbin - 1 ||
      nt < 0 || nt > Ntable.Ntheta - 1) {
    log_fatal("error in bin number (nl,ni,nj,nt) = [%d,%d,%d,%d]",nl,ni,nj,nt);
    exit(1); 
  } 
  if (test_zoverlap_cggl(ni, nj)) {
    const int q = nl*NCGL(ni,nj)*Ntable.Ntheta + nt;
    if (q > NSIZE*Ntable.Ntheta - 1) {
      log_fatal("internal logic error in selecting bin number"); exit(1);
    }
    return w_vec[q];
  }
  else {
    return 0.0;
  }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double w_cc_tomo(
    const int nt, 
    const int nl1, 
    const int nl2, 
    const int ni, 
    const int limber
  )
{
  static double** Pl = NULL;
  static double* w_vec = NULL;
  static double** Cl = NULL; 
  static double cache[MAX_SIZE_ARRAYS];
  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized");
    exit(1);
  }
  const int NSIZE = Cluster.n200_nbin*Cluster.n200_nbin*redshift.clusters_nbin;

  if (NULL == Pl || NULL == w_vec || 
      NULL == Cl || fdiff(cache[3],Ntable.random))
  {
    const int lmin = 1;
    if (Pl != NULL) free(Pl);
    Pl = (double**) malloc2d(Ntable.Ntheta, Ntable.LMAX);
    if (w_vec != NULL) free(w_vec);
    w_vec = (double*) calloc1d(NSIZE*Ntable.Ntheta);
    double*** P = (double***) malloc3d(2, Ntable.Ntheta, Ntable.LMAX+1);
    double** Pmin  = P[0]; double** Pmax  = P[1];
    // -------------------------------------------------------------------------
    double xmin[Ntable.Ntheta];
    double xmax[Ntable.Ntheta];
    for (int i=0; i<Ntable.Ntheta; i ++)
    { // Cocoa: dont thread (init of static variables inside set_bin_average)
      bin_avg r = set_bin_average(i,0);
      xmin[i] = r.xmin;
      xmax[i] = r.xmax;
    }
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<(Ntable.LMAX+1); l++) {
        bin_avg r = set_bin_average(i,l);
        Pmin[i][l] = r.Pmin;
        Pmax[i][l] = r.Pmax;
      }
    }
    // -------------------------------------------------------------------------
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<lmin; l++) {
        Pl[i][l] = 0.0;
      }
    }
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=lmin; l<Ntable.LMAX; l++) { 
        const double tmp = (1.0/(xmin[i] - xmax[i]))*(1. / (4.0 * M_PI));
        Pl[i][l] = tmp*(Pmin[i][l + 1] - Pmax[i][l + 1] 
                        - Pmin[i][l - 1] + Pmax[i][l - 1]);
      }
    }
    // -------------------------------------------------------------------------
    free(P);
    if (Cl != NULL) free(Cl);
    Cl = (double**) malloc2d(NSIZE, Ntable.LMAX);
  }

  if (fdiff(cache[0], cosmology.random) || 
      fdiff(cache[1], nuisance.random_photoz_clusters) ||
      fdiff(cache[2], redshift.random_clusters) ||
      fdiff(cache[3], Ntable.random) ||
      fdiff(cache[4], nuisance.random_clusters))
  {
    const int lmin = 1;
    for (int i=0; i<NSIZE; i++) {
      for (int l=0; l<lmin; l++) {
        Cl[i][l] = 0.0;
      }
    } 
    (void) C_cc_tomo_limber(limits.LMIN_tab+1, 0, 0, 0, 0); // init static vars
    // -------------------------------------------------------------------------
    if (1 == limber) {
      #pragma omp parallel for collapse(3)
      for (int i=0; i<Cluster.n200_nbin; i++) {
        for (int k=0; k<redshift.clusters_nbin; k++) {
          for (int l=lmin; l<limits.LMIN_tab; l++) {
            for (int j=i; j<Cluster.n200_nbin; j++) {
              const int q  = i*Cluster.n200_nbin*redshift.clusters_nbin + 
                             j*redshift.clusters_nbin + 
                             k;
              Cl[q][l] = C_cc_tomo_limber_nointerp(l, i, j, k, k, 0);
              
              const int q2  = j*Cluster.n200_nbin*redshift.clusters_nbin + 
                              i*redshift.clusters_nbin + 
                              k;
              Cl[q2][l] = Cl[q][l];
            }
          } 
        }
      }
      #pragma omp parallel for collapse(3)
      for (int i=0; i<Cluster.n200_nbin; i++) {
        for (int k=0; k<redshift.clusters_nbin; k++) {
          for (int l=limits.LMIN_tab; l<Ntable.LMAX; l++) {
            for (int j=i; j<Cluster.n200_nbin; j++) {
              const int q  = i*Cluster.n200_nbin*redshift.clusters_nbin + 
                             j*redshift.clusters_nbin + 
                             k;
              Cl[q][l] = C_cc_tomo_limber(l, i, j, k, k);

              const int q2  = j*Cluster.n200_nbin*redshift.clusters_nbin + 
                              i*redshift.clusters_nbin + 
                              k;
              Cl[q2][l] = Cl[q][l];
            }
          } 
        }
      }
    }
    else { // TODO: implement nonlimber
      log_fatal("NonLimber not implemented"); exit(1);
    }
    // -------------------------------------------------------------------------
    #pragma omp parallel for collapse(4) schedule(static,1)
    for (int i=0; i<Cluster.n200_nbin; i++) { 
      for (int j=0; j<Cluster.n200_nbin; j++) {
        for (int k=0; k<redshift.clusters_nbin; k++) {
          for (int p=0; p<Ntable.Ntheta; p++) {
            const int nz = i*Cluster.n200_nbin*redshift.clusters_nbin + 
                           j*redshift.clusters_nbin + 
                           k;            
            double sum = 0.0;
            for (int l=lmin; l<Ntable.LMAX; l++) {
              sum += Pl[p][l]*Cl[nz][l];
            }
            w_vec[nz*Ntable.Ntheta + p] = sum;
          }
        }
      }
    }
    // -------------------------------------------------------------------------
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_clusters;
    cache[2] = redshift.random_clusters;
    cache[3] = Ntable.random;
    cache[4] = nuisance.random_clusters;
  }
  // ---------------------------------------------------------------------------
  if (nl1 < 0 || nl1 > Cluster.n200_nbin - 1 ||
      nl2 < 0 || nl2 > Cluster.n200_nbin - 1 ||
      ni < 0  || ni > redshift.clusters_nbin - 1 ||
      nt < 0  || nt > Ntable.Ntheta - 1) {
    log_fatal("error in bin number (nl1,nl2,ni,nt) = [%d,%d,%d,%d]",nl1,nl2,ni,nt);
    exit(1); 
  } 
  const int q = (nl1*Cluster.n200_nbin*redshift.clusters_nbin + 
                 nl2*redshift.clusters_nbin + 
                 ni)*Ntable.Ntheta + nt;
  if (q  < 0 || q > NSIZE*Ntable.Ntheta - 1) {
    log_fatal("internal logic error in selecting bin number"); exit(1);
  }
  return w_vec[q];
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double w_cg_tomo(const int nt, const int nl, const int ni, const int nj, const int limber)
{
  static double** Pl = NULL;
  static double* w_vec = NULL;
  static double** Cl = NULL; 
  static double cache[MAX_SIZE_ARRAYS];

  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized");
    exit(1);
  }

  const int NSIZE = Cluster.n200_nbin*tomo.cg_npowerspectra;

  if (NULL == Pl || NULL == w_vec || NULL == Cl || fdiff(cache[3], Ntable.random))
  {
    const int lmin = 1;
    if (Pl != NULL) free(Pl);
    Pl = (double**) malloc2d(like.Ntheta, limits.LMAX);
    if (w_vec != NULL) free(w_vec);
    w_vec = (double*) calloc1d(NSIZE*like.Ntheta);
    double*** P = (double***) malloc3d(2, like.Ntheta, limits.LMAX + 1);
    double** Pmin  = P[0]; double** Pmax  = P[1];
    // -------------------------------------------------------------------------
    double xmin[like.Ntheta];
    double xmax[like.Ntheta];
    for (int i=0; i<like.Ntheta; i ++)
    { // Cocoa: dont thread (init of static variables inside set_bin_average)
      bin_avg r = set_bin_average(i,0);
      xmin[i] = r.xmin;
      xmax[i] = r.xmax;
    }
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<like.Ntheta; i++) {
      for (int l=0; l<(limits.LMAX+1); l++) {
        bin_avg r = set_bin_average(i,l);
        Pmin[i][l] = r.Pmin;
        Pmax[i][l] = r.Pmax;
      }
    }
    // -------------------------------------------------------------------------
    for (int i=0; i<like.Ntheta; i++) {
      for (int l=0; l<lmin; l++) {
        Pl[i][l] = 0.0;
      }
    }
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<like.Ntheta; i++) {
      for (int l=lmin; l<limits.LMAX; l++) { 
        const double tmp = (1.0/(xmin[i] - xmax[i]))*(1. / (4.0 * M_PI));
        Pl[i][l] = tmp*(Pmin[i][l + 1] - Pmax[i][l + 1] 
                        - Pmin[i][l - 1] + Pmax[i][l - 1]);
      }
    }
    // -------------------------------------------------------------------------
    free(P);
    if (Cl != NULL) free(Cl);
    Cl = (double**) malloc2d(NSIZE, limits.LMAX);
  }

  if (fdiff(cache[0], cosmology.random) || 
      fdiff(cache[1], nuisance.random_photoz_clustering) ||
      fdiff(cache[2], redshift.random_clustering) ||
      fdiff(cache[3], Ntable.random) ||
      fdiff(cache[4], nuisance.random_galaxy_bias) ||
      fdiff(cache[5], nuisance.random_photoz_clusters) ||
      fdiff(cache[6], nuisance.random_clusters) ||
      fdiff(cache[7], redshift.random_clusters))
  {
    const int lmin = 1;
    for (int i=0; i<NSIZE; i++) {
      for (int l=0; l<lmin; l++) {
        Cl[i][l] = 0.0;
      }
    } 
    (void) C_cg_tomo_limber(limits.LMIN_tab+1, 0, ZCG1(0), ZCG2(0)); // init static vars
    // -------------------------------------------------------------------------
    if (1==limber) {
      #pragma omp parallel for collapse(3) schedule(static,1)
      for (int i=0; i<Cluster.n200_nbin; i++) {
        for (int j=0; j<tomo.cg_npowerspectra; j++) {
          for (int l=lmin; l<limits.LMIN_tab; l++) {
            const int q = i*tomo.cg_npowerspectra + j;
            Cl[q][l] = C_cg_tomo_limber_nointerp(l, i, ZCG1(j), ZCG2(j), 0);
          }
        }
      }
      #pragma omp parallel for collapse(3) schedule(static,1)
      for (int i=0; i<Cluster.n200_nbin; i++) {
        for (int j=0; j<tomo.cg_npowerspectra; j++) {
          for (int l=limits.LMIN_tab; l<limits.LMAX; l++) {
            const int q = i*tomo.cg_npowerspectra + j;
            Cl[q][l] = C_cg_tomo_limber(l, i, ZCG1(j), ZCG2(j));
          }
        }
      }
    }
    else {
      log_fatal("NonLimber not implemented");
      exit(1);
    }
    // -------------------------------------------------------------------------
    #pragma omp parallel for collapse(3) schedule(static,1)
    for (int i=0; i<Cluster.n200_nbin; i++) {
      for (int j=0; j<tomo.cg_npowerspectra; j++) {
        for (int p=0; p<like.Ntheta; p++) {
          const int nz = i*tomo.cg_npowerspectra + j;
          double sum = 0.0;
          for (int l=lmin; l<limits.LMAX; l++) {
            sum += Pl[p][l]*Cl[nz][l];
          }
          w_vec[nz*like.Ntheta + p] = sum;
        }
      }
    }
    // -------------------------------------------------------------------------
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_clustering;
    cache[2] = redshift.random_clustering;
    cache[3] = Ntable.random;
    cache[4] = nuisance.random_galaxy_bias;
    cache[5] = nuisance.random_photoz_clusters;
    cache[6] = nuisance.random_clusters;
    cache[7] = redshift.random_clusters;
  } 
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  if (nl < 0 || nl > Cluster.n200_nbin - 1 ||
      ni < 0 || ni > redshift.clusters_nbin - 1 ||
      nj < 0 || nj > redshift.clustering_nbin - 1
      nt < 0 || nt > like.Ntheta - 1) {
    log_fatal("error in bin number (nl,ni,nj,nt) = [%d,%d,%d,%d]",nl,ni,nj,nt);
    exit(1); 
  }
  if (test_zoverlap_cg(ni, nj)) {
    const int q = nl*NCG(ni, nj)*like.Ntheta + nt;
    if (q < 0 || q > NSIZE*like.Ntheta - 1) {
      log_fatal("internal logic error in selecting bin number");
      exit(1);
    }
    return w_vec[q];
  }
  else {
    return 0.0;
  }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Limber Approximation (Angular Power Spectrum)
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double int_for_C_cs_tomo_limber(double a, void* params)
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true");
    exit(1);
  }
  double* ar = (double *) params;

  const int nl = (int) ar[0];
  const int ni = (int) ar[1];
  const int nj = (int) ar[2];
  const double ell = ar[3] + 0.5;

  struct chis chidchi = chi_all(a);
  const double hoverh0 = hoverh0v2(a, chidchi.dchida);
  const double fK  = f_K(chidchi.chi);
  const double k   = ell/fK;  
  
  const double bc     = cluster_b1_given_lambda_obs_in_nl_given_ztrue(nl, a);
  const double WCL    = W_cluster(nl, ni, a, hoverh0);
  const double WMAGCL = W_mag_cluster(ni, a, fK);
  const double WK     = W_kappa(a, fK, nj);
  const double WS     = W_source(a, nj, hoverh0);

  double res = 1.0;
  switch(nuisance.IA_MODEL)
  {
    case IA_MODEL_NLA:
    {
      const double C1ZS = IA_A1_Z1(a, growfac_a, nj);
      res  = (bc*WCL - 2*WMAGCL)*chidchi.dchida/(fK*fK);
      res *= (WK - WS*C1ZS)
    }
    default:
    {
      log_fatal("nuisance.IA_MODEL = %d not supported", nuisance.IA_MODEL);
      exit(1);
    }
  }
  if (1 == adopt_dark_emulator) {
    res *= pcm_darkemu(k, a, nl, ni, nj)
  }
  else {
    res *= Pdelta(k,a);
    double one_halo = pcm_1halo(k, a, nl, ni, nj);
    one_halo *= WK*WCL*chidchi.dchida/(fK*fK);
    res += one_halo;
  }
  return (res > 1E10) ? 0.0 : res; // COSMOLIKE (in the original code)
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double C_cs_tomo_limber_nointerp(const double l, const int nl, const int ni, 
  const int nj, const int init)
{
  static double cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (nl < 0 || nl > Cluster.n200_nbin - 1 ||
      ni < 0 || ni > redshift.clusters_nbin - 1 ||
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("error in bin number (nl,ni,nj) = [%d,%d,%d]", nl,ni,nj);
    exit(1); 
  }
  if (NULL == w || fdiff(cache[0], Ntable.random)) {
    const size_t szint = 90 + 40 * abs(Ntable.high_def_integration);
    if (w != NULL)  {
      gsl_integration_glfixed_table_free(w);
    }
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }
  // ---------------------------------------------------------------------------
  const double zmin = tomo.cluster_zmin[ni];
  const double zmax = tomo.cluster_zmax[ni];
  if ((zmin > zmax) || !(zmin>0) || !(zmax>0)) {
    log_fatal("error in redshift range (min,max) = [%e,%e]",zmin,zmax);
    exit(1); 
  } 
  const double amin = 1./(1. + zmax);
  const double amax = 1./(1. + zmin);
  double ar[4] = {(double) nl, (double) ni, (double) nj, l};
  // ---------------------------------------------------------------------------
  double res = 0.0;
  if (1 == init) {
    (void) int_for_C_cs_tomo_limber(amin, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_C_cs_tomo_limber;
    res = gsl_integration_glfixed(&F, amin, amax, w);
  }
  return res;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double C_cs_tomo_limber_nointerp(const double l, const int nl, const int ni, 
  const int ns, const int init) 
{
  return C_cs_tomo_limber_linpsopt_nointerp(l, nl, ni, ns, 0, init);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double C_cs_tomo_limber(const double l, const int nl, const int ni, const int nj)
{
  static double cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double lim[3];
  // ---------------------------------------------------------------------------
  if (NULL == table || fdiff(cache[6], Ntable.random)) {
    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2] = (lim[1] - lim[0]) / ((double) Ntable.N_ell - 1.0);
    if (table != NULL) free(table);
    table = (double**) malloc2d(Cluster.n200_nbin*tomo.cs_npowerspectra, Ntable.N_ell);
  }
  // ---------------------------------------------------------------------------
  if (fdiff(cache[0], cosmology.random) ||
      fdiff(cache[1], nuisance.random_photoz_shear) ||
      fdiff(cache[2], nuisance.random_photoz_clusters) ||
      fdiff(cache[3], nuisance.random_ia) ||
      fdiff(cache[4], redshift.random_shear) ||
      fdiff(cache[5], redshift.random_clusters) ||
      fdiff(cache[6], Ntable.random) ||
      fdiff(cache[7], nuisance.random_clusters))
  {
    (void) C_cs_tomo_limber_nointerp(exp(lnlmin), 0, ZCL(0), ZCS(0), 1); // init static vars
    #pragma omp parallel for collapse(3) schedule(static,1)
    for (int i=0; i<Cluster.n200_nbin; i++) { 
      for (int j=0; j<tomo.cs_npowerspectra; j++) {
        for (int p=0; p<Ntable.N_ell; p++) {
          const int q = i*tomo.cs_npowerspectra + j;
          table[q][p] = log(C_cs_tomo_limber_nointerp(exp(lnlmin+p*dlnl),i,ZCL(j),ZCS(j),0));
        }
      }
    }
    // -------------------------------------------------------------------------
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_photoz_clusters;
    cache[3] = nuisance.random_ia;
    cache[4] = redshift.random_shear;
    cache[5] = redshift.random_clusters;
    cache[6] = Ntable.random;
    cache[7] = nuisance.random_clusters;
  }
  // ---------------------------------------------------------------------------
  if (nl < 0 || nl > Cluster.n200_nbin - 1 ||
      ni < 0 || ni > redshift.clusters_nbin - 1 ||
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("error in bin number (nl,ni,nj) = [%d,%d,%d]", nl, ni, nj);
    exit(1); 
  } 
  double res = 0.0;
  if (test_zoverlap_c(ni, nj)) {
    const double lnl = log(l);
    if (lnl < lim[0]) {
      log_warn("l = %e < l_min = %e. Extrapolation adopted", l, exp(lim[0]));
    }
    if (lnl > lim[1]) {
      log_warn("l = %e > l_max = %e. Extrapolation adopted", l, exp(lim[1]));
    }
    const int q = nl*tomo.cs_npowerspectra + NCGL(ni, nj);
    if (q < 0 || q > Cluster.n200_nbin*tomo.cs_npowerspectra - 1) {
      log_fatal("internal logic error in selecting bin number");
      exit(1);
    }
    res = exp(interpol1d(table[q], Ntable.N_ell, lim[0], lim[1], lim[2], lnl));
  }
  return res;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double int_for_C_cc_tomo_limber(double a, void* params)
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true");
    exit(1);
  }
  double* ar = (double*) params;  
  const int nl1 = (int) ar[0];
  const int nl2 = (int) ar[1];
  const int ni  = (int) ar[2];
  const double ell = ar[3] + 0.5;
  const int use_linear_ps = (int) ar[4];
  
  struct chis chidchi = chi_all(a);
  const double hoverh0 = hoverh0v2(a, chidchi.dchida);
  const double fK = f_K(chidchi.chi);
  const double k  = ell/fK;
  const double WCNL1 = W_cluster(nl1, ni, a, hoverh0);
  const double WCNL2 = W_cluster(nl2, ni, a, hoverh0);
  
  double res = 1.0;
  if (1 == include_exclusion) {
    res  = WCNL1*WCNL2*chidchi.dchida/(fK*fK);
    res *= pcc_with_excl(k, a, nl1, nl2, use_linear_ps);
  }
  else {
    const double bc1 = cluster_b1_given_lambda_obs_in_nl_given_ztrue(nl1, a);
    const double bc2 = (nl1 == nl2) ? bc1 : 
                                      cluster_b1_given_lambda_obs_in_nl(nl2, a);
    const double WMAGCL = W_mag_cluster(ni, a, fK);
    res  = (bc1*WCNL1 - 2*WMAGCL)*(bc2*WCNL2 - 2*WMAGCL)*chidchi.dchida/(fK*fK);
    res *= (1 == use_linear_ps) ? p_lin(k,a) : Pdelta(k,a);
  }
  return res;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double C_cc_tomo_limber_linpsopt_nointerp(
    const double l, 
    const int nl1, 
    const int nl2, 
    const int ni,
    const int use_linear_ps, 
    const int init
  ) 
{
  static double cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;  

  if (nl1 < 0 || nl1 > Cluster.n200_nbin - 1 ||
      nl2 < 0 || nl2 > Cluster.n200_nbin - 1 ||
      ni < 0  || ni > redshift.clusters_nbin - 1) {
    log_fatal("error in bin number (nl1,nl2,ni) = [%d,%d,%d]", nl1,nl2,ni);
    exit(1); 
  }
  if (NULL == w || fdiff(cache[0], Ntable.random)) {
    const size_t szint = 90 + 40 * abs(Ntable.high_def_integration);
    if (w != NULL)  {
      gsl_integration_glfixed_table_free(w);
    }
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }
  // ---------------------------------------------------------------------------
  double ar[5] = {(double) nl1, 
                  (double) nl2, 
                  (double) ni, 
                  l, 
                  (double) use_linear_ps};  
  const double zmin = tomo.cluster_zmin[ni];
  const double zmax = tomo.cluster_zmax[ni];
  if ((zmin > zmax) || !(zmin>0) || !(zmax>0)) {
    log_fatal("error in redshift range (min,max) = [%e,%e]", zmin, zmax);
    exit(1); 
  } 
  const double amin = 1.0/(1.0 + zmax);
  const double amax = 1.0/(1.0 + zmin);
  // ---------------------------------------------------------------------------
  double res = 0.0;
  if (1 == init) {
    (void) int_for_C_cc_tomo_limber(amin, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_C_cc_tomo_limber;
    res = gsl_integration_glfixed(&F, amin, amax, w);
  }
  return res;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double C_cc_tomo_limber_nointerp(
    const double l, 
    const int nl1, 
    const int nl2, 
    const int ni,
    const int init) {
  return C_cc_tomo_limber_linpsopt_nointerp(l, nl1, ni, nl2, ni, 0, init);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double C_cc_tomo_limber(const double l, const int nl1, const int nl2, const int ni) 
{
  static double cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double lim[3];
  // ---------------------------------------------------------------------------
  if (NULL == table || fdiff(cache[3], Ntable.random)) {
    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2] = (lim[1] - lim[0]) / ((double) Ntable.N_ell - 1.0);
    const int NSIZE = Cluster.n200_nbin*Cluster.n200_nbin*redshift.clusters_nbin;
    if (table != NULL) free(table);
    table = (double**) malloc2d(NSIZE, Ntable.N_ell);
  }
  // ---------------------------------------------------------------------------
  if (fdiff(cache[0], cosmology.random) || 
      fdiff(cache[1], nuisance.random_photoz_clusters) ||
      fdiff(cache[2], redshift.random_clusters) ||
      fdiff(cache[3], Ntable.random) ||
      fdiff(cache[4], nuisance.random_clusters))
  {
    (void) C_cc_tomo_limber_nointerp(exp(lnlmin), 0, 0, 0, 1); // init static vars
    #pragma omp parallel for collapse(3) schedule(static,1)
    for (int i=0; i<Cluster.n200_nbin; i++) {
      for (int k=0; k<redshift.clusters_nbin; k++) {
        for (int p=0; p<Ntable.N_ell; ++p) {
          for (int j=i; j<Cluster.n200_nbin; j++) {
            const int q  = i*Cluster.n200_nbin*redshift.clusters_nbin + 
                           j*redshift.clusters_nbin + 
                           k;
            const int q2 = j*Cluster.n200_nbin*redshift.clusters_nbin + 
                           i*redshift.clusters_nbin + 
                           k;
            table[q][p] = log(C_cc_tomo_limber_nointerp(exp(lnlmin+p*dlnl),i,j,k,0));
            table[q2][p] = table[q][p];
          }
        }
      }
    }
    // -------------------------------------------------------------------------
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_clusters;
    cache[2] = redshift.random_clusters;
    cache[3] = Ntable.random;
    cache[4] = nuisance.random_clusters;
  }
  // ---------------------------------------------------------------------------
  if (nl1 < 0 || nl1 > Cluster.n200_nbin - 1 ||
      nl2 < 0 || nl2 > Cluster.n200_nbin - 1 ||
      ni < 0  || ni > redshift.clusters_nbin - 1) {
    log_fatal("error in bin number (nl1,nl2,ni,nt) = [%d,%d,%d]",nl1,nl2,ni);
    exit(1); 
  } 
  const double lnl = log(l);
  if (lnl < lim[0]) {
    log_warn("l = %e < l_min = %e. Extrapolation adopted", l, exp(lim[0]));
  }
  if (lnl > lim[1]) {
    log_warn("l = %e > l_max = %e. Extrapolation adopted", l, exp(lim[1]));
  }
  const int q = nl1*Cluster.n200_nbin*redshift.clusters_nbin + 
                nl2*redshift.clusters_nbin + 
                ni;
  if (q > Cluster.n200_nbin*Cluster.n200_nbin*redshift.clusters_nbin-1) {
    log_fatal("internal logic error in selecting bin number");
    exit(1);
  }
  return exp(interpol1d(table[q], Ntable.N_ell, lim[0], lim[1], lim[2], lnl));
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double int_for_C_cg_tomo_limber(double a, void* params)
{ 
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true");
    exit(1);
  }
  double* ar = (double*) params;

  const int nl = (int) ar[0];
  const int ni = (int) ar[1]; // Cluster redshift bin
  const int nj = (int) ar[2]; // Galaxy  redshift bin
  const double ell = ar[3] + 0.5;
  const int use_linear_ps = (int) ar[4];

  struct chis chidchi = chi_all(a);
  const double hoverh0 = hoverh0v2(a, chidchi.dchida);
  const  double fK  = f_K(chidchi.chi);
  const double k = ell/fK;

  const double WC   = W_cluster(nl, ni, a, hoverh0);
  const double WGAL = W_gal(a, nj);
  
  double res = WC*WGAL*chidchi.dchida/(fK*fK);
  res *= (1 == use_linear_ps) ? p_lin(k, a) : Pdelta(k, a);
  return (res < 0) 0. : res; // COSMOLIKE: this makes the code much faster
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double C_cg_tomo_limber_linpsopt_nointerp(
    const double l, 
    const int nl, // lambda_obs bin
    const int ni, // cluster redshift bin
    const int nj, // galaxy redshift bin
    const int use_linear_ps, 
    const int init
  )
{
  static double cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL; 

  if (nl < 0 || nl > Cluster.n200_nbin - 1 ||
      ni < 0 || ni > redshift.clusters_nbin - 1 ||
      nj < 0 || nj > redshift.clustering_nbin - 1) {
    log_fatal("error in bin number (nl,ni,nj) = [%d,%d,%d]",nl,ni,nj);
    exit(1); 
  }
  if (NULL == w || fdiff(cache[0], Ntable.random)) {
    const size_t szint = 90 + 40 * abs(Ntable.high_def_integration);
    if (w != NULL)  {
      gsl_integration_glfixed_table_free(w);
    }
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }
  // ---------------------------------------------------------------------------
  double ar[5] = {(double) nl, 
                  (double) ni, 
                  (double) nj, 
                  l, 
                  (double) use_linear_ps};
  const double zmin = fmax(tomo.cluster_zmin[ni], tomo.clustering_zmin[nj]);
  const double zmax = fmin(tomo.cluster_zmax[ni], tomo.clustering_zmax[nj]);
  if ((zmin > zmax) || !(zmin>0) || !(zmax>0)) {
    log_fatal("error in redshift range (min,max) = [%e,%e]", zmin, zmax);
    exit(1); 
  } 
  const double amin = 1.0/(1.0 + zmax);
  const double amax = 1.0/(1.0 + zmin);
  // ---------------------------------------------------------------------------
  double res = 0.0;
  if (1 == init) {
    (void) int_for_C_cg_tomo_limber(amin, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_C_cg_tomo_limber;
    res = gsl_integration_glfixed(&F, amin, amax, w);
  }
  return res;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double C_cg_tomo_limber_nointerp(
    const double l, 
    const int nl,
    const int ni, 
    const int nj, 
    const int init
  ) {
  return C_cg_tomo_limber_linpsopt_nointerp(l, nl, ni, nj, 0, init);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double C_cg_tomo_limber(const double l, const int nl, const int ni, const int nj)
{
  static double cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double lim[3];
  // ---------------------------------------------------------------------------
  if (NULL == table || fdiff(cache[3], Ntable.random)) {
    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2] = (lim[1] - lim[0]) / ((double) Ntable.N_ell - 1.0);
    const int NSIZE = Cluster.n200_nbin*tomo.cg_clustering_Npowerspectra;
    if (table != NULL) free(table);
    table = (double**) malloc2d(NSIZE, Ntable.N_ell);
  }
  // ---------------------------------------------------------------------------
  if (fdiff(cache[0], cosmology.random) || 
      fdiff(cache[1], nuisance.random_photoz_clustering) ||
      fdiff(cache[2], redshift.random_clustering) ||
      fdiff(cache[3], Ntable.random) ||
      fdiff(cache[4], nuisance.random_galaxy_bias) ||
      fdiff(cache[5], nuisance.random_photoz_clusters) ||
      fdiff(cache[6], nuisance.random_clusters) ||
      fdiff(cache[7], redshift.random_clusters))
  {
    (void) C_cg_tomo_limber_nointerp(exp(lnlmin), 0, ZCG1(0), ZCG2(0), 1);
    #pragma omp parallel for collapse(3) schedule(static,1)
    for (int i=0; i<Cluster.n200_nbin; i++) {
      for (int j=0; j<tomo.cg_clustering_Npowerspectra; j++) {
        for (int p=0; p<Ntable.N_ell; ++p) {
          const int q = i*tomo.cg_clustering_Npowerspectra + j;
          table[q][p] = log(C_cg_tomo_limber_nointerp(exp(lnlmin + p*dlnl), i, 
                                                      ZCG1(j), ZCG2(j), 0));
        }
      }
    }
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_clustering;
    cache[2] = redshift.random_clustering;
    cache[3] = Ntable.random;
    cache[4] = nuisance.random_galaxy_bias;
    cache[5] = nuisance.random_photoz_clusters;
    cache[6] = nuisance.random_clusters;
    cache[7] = redshift.random_clusters;
  }
  // ---------------------------------------------------------------------------
  if (nl < 0 || nl > Cluster.n200_nbin - 1 ||
      ni < 0 || ni > redshift.clusters_nbin - 1 ||
      nj < 0 || nj > redshift.clustering_nbin - 1) {
    log_fatal("error in bin number (nl,ni,nj) = [%d,%d,%d]",nl,ni,nj);
    exit(1); 
  }
  double res = 0.0;
  const int ntomo = NCG(ni,nj);
  if (ntomo>0) {
    const double lnl = log(l);
    if (lnl < lim[0]) {
      log_warn("l = %e < l_min = %e. Extrapolation adopted", l, exp(lim[0]));
    }
    if (lnl > lim[1]) {
      log_warn("l = %e > l_max = %e. Extrapolation adopted", l, exp(lim[1]));
    }
    const int q = nl*tomo.cg_clustering_Npowerspectra + ntomo;
    if (q > Cluster.n200_nbin*tomo.cg_clustering_Npowerspectra-1) {
      log_fatal("internal logic error in selecting bin number");
      exit(1);
    }
    res = exp(interpol1d(table[q], Ntable.N_ell, lim[0], lim[1], lim[2], lnl));
  }
  return res;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Cluster number counts
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

/*
// nl = lambda_obs bin, ni = cluster redshift bin

double int_for_binned_N(double a, void* params)
{
  if (!(a>0))
  {
    log_fatal("a > 0 not true");
    exit(1);
  }
  double* ar = (double*) params;   
  const int nl = (int) ar[0];
  const int nz = (int) ar[1];
  const int interpolate_survey_area = (int) ar[2];
  const double z = 1.0/a - 1.0 ;
  const double dzda = 1.0/(a*a); 
  const double norm = get_area(z, interpolate_survey_area);  

  double tmp_param[1] = {(double) nz};
  return dV_cluster(z, (void*) tmp_param)*dzda*binned_Ndensity(nl, z)*norm;
}

double binned_N_nointerp(const int nl, const int nz, const int interpolate_survey_area, 
const int init_static_vars_only)
{
  double params[3] = {(double) nl, (double) nz, interpolate_survey_area};
  const double tmp = 4.0*M_PI/41253.0;
  const double amin = 1.0/(1.0 + tomo.cluster_zmax[nz]);
  const double amax = 1.0/(1.0 + tomo.cluster_zmin[nz]);
  return (init_static_vars_only == 1) ? int_for_binned_N(amin, (void*) params) :
    tmp*int_gsl_integrate_low_precision(int_for_binned_N, (void*) params, amin, amax, NULL, 
      GSL_WORKSPACE_SIZE);
}

double binned_N(const int nl, const int nz)
{
  static cosmopara C;
  static nuisancepara N;
  static double** table;

  const int N_l = Cluster.n200_nbin;
  const int N_z = redshift.clustering_nbin;

  if (table == 0)
  {
    table = (double**) malloc(sizeof(double*)*N_l);
    for (int i=0; i<N_l; i++)
    {
      table[i] = (double*) malloc(sizeof(double)*N_z);
    }
  }
  if (recompute_clusters(C, N))
  {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-variable"
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-but-set-variable"
    {
      double init_static_vars_only = binned_N_nointerp(0, 0, Cluster.interpolate_survey_area, 1);
    }
    #pragma GCC diagnostic pop
    #pragma GCC diagnostic pop
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<N_l; i++)
    {
      for (int j=0; j<N_z; j++)
      {
        table[i][j] = binned_N_nointerp(i, j, Cluster.interpolate_survey_area, 0);
      }
    }
    update_cosmopara(&C);
    update_nuisance(&N);
  }
  if (nl < 0 || nl > N_l - 1)
  {
    log_fatal("error in selecting bin number");
    exit(1);
  }
  if (nz < 0 || nz > N_z - 1)
  {
    log_fatal("error in selecting bin number");
    exit(1);
  }
  return table[nl][nz];
}
*/