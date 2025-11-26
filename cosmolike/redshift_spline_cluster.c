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

double pf_cluster_histo_n(double z,  void* params) 
{
  static int njmax;
  static double** table;
  static double zhisto_max, zhisto_min, dz;

  if (table == 0)
  {
    const int zbins = line_count(redshift.clusters_REDSHIFT_FILE);
    njmax = zbins;
    table = (double**) malloc(sizeof(double*)*redshift.cluster_nbin);
    for (int i=0; i<redshift.cluster_nbin; i++)
    {
      table[i] = (double*) malloc(sizeof(double)*zbins);
    }
    double* z_v = (double*) malloc(sizeof(double)*zbins);
    
    FILE* ein = fopen(redshift.clusters_REDSHIFT_FILE, "r");
    if (ein == NULL)
    {
      log_fatal("file not opened");
      exit(1);
    }
    int p = 0;
    for (int i=0; i<zbins; i++)
    {
      fscanf(ein, "%le", &z_v[i]);
      p++;
      if (i > 0 && z_v[i] < z_v[i-1]) 
      {
        break;
      }
      for (int k=0; k<redshift.cluster_nbin; k++)
      {
        fscanf(ein," %le", &table[k][i]);
      }
    }
    fclose(ein);

    dz = (z_v[p - 1] - z_v[0]) / ((double) p - 1.0);
    zhisto_max = z_v[p - 1] + dz;
    zhisto_min = z_v[0];
    
    for (int k=0; k<redshift.cluster_nbin; k++)
    { // now, set tomography bin boundaries
      double max = table[k][0];
      for (int i=1; i<zbins; i++)
      {
        if (table[k][i]> max)
        {
          max = table[k][i];
        }
      }
      {
        int i = 0;
        while (table[k][i] <1.e-8*max && i < zbins-2)
        {
          i++;
        }
        tomo.cluster_zmin[k] = z_v[i];
      }
      {
        int i = zbins-1;
        while (table[k][i] <1.e-8*max && i > 0)
        {
          i--;
        }
        tomo.cluster_zmax[k] = z_v[i];
      }
      log_info("tomo.cluster_zmin[%d] = %.3f,tomo.cluster_zmax[%d] = %.3f",
        k, tomo.cluster_zmin[k], k, tomo.cluster_zmax[k]);
    }
    free(z_v);
    
    if (zhisto_max < tomo.cluster_zmax[redshift.cluster_nbin - 1] || zhisto_min > tomo.cluster_zmin[0])
    {
      log_fatal("%e %e %e %e", zhisto_min,tomo.cluster_zmin[0], 
        zhisto_max,tomo.cluster_zmax[redshift.cluster_nbin-1]);
      log_fatal("pf_cluster_histo_n: redshift file = %s is incompatible with bin choice", 
        redshift.clusters_REDSHIFT_FILE);
      exit(1);
    }
  }

  double res = 0.0;
  if ((z >= zhisto_min) && (z < zhisto_max)) 
  {
    double *ar = (double*) params;
    const int ni = (int) ar[0];
    const int nj = (int) floor((z - zhisto_min)/dz);

    if (ni < 0 || ni > redshift.cluster_nbin - 1 || nj < 0 || nj > njmax - 1)
    {
      log_fatal("invalid bin input (ni, nj) = (%d, %d)", ni, nj);
      exit(1);
    } 
    res = table[ni][nj];
  }
  return res;
}

double pz_cluster(const double zz, const int nz)
{
  static double** table = 0;
  static double* z_v = 0;
  static int zbins = -1;
  static gsl_spline* photoz_splines[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = 0;

  if (nz < -1 || nz > redshift.cluster_nbin - 1)
  {
    log_fatal("invalid bin input nz = %d (max %d)", nz, redshift.cluster_nbin);
    exit(1);
  }

  if (redshift.clusters_photoz == 0)
  {
    if ((zz >= tomo.cluster_zmin[nz]) & (zz <= tomo.cluster_zmax[nz])) 
    {
      return 1;
    }
    else
    { 
      return 0;
    }
  }

  if (table == 0)
  {
    const size_t nsize_integration = 2250 + 500 * (Ntable.high_def_integration);
    w = gsl_integration_glfixed_table_alloc(nsize_integration);

    zbins = line_count(redshift.clusters_REDSHIFT_FILE);
 
    table = (double**) malloc(sizeof(double*)*(redshift.cluster_nbin+1));
    for (int i=0; i<(redshift.cluster_nbin+1); i++)
    {
      table[i] = (double*) malloc(sizeof(double)*zbins);
    }
    z_v = (double*) malloc(sizeof(double)*zbins);

    for (int i=0; i<redshift.clustering_nbin+1; i++) 
    {
      if (Ntable.photoz_interpolation_type == 0)
      {
        photoz_splines[i] = gsl_spline_alloc(gsl_interp_cspline, zbins);
      }
      else if (Ntable.photoz_interpolation_type == 1)
      {
        photoz_splines[i] = gsl_spline_alloc(gsl_interp_linear, zbins);
      }
      else
      {
        photoz_splines[i] = gsl_spline_alloc(gsl_interp_steffen, zbins);
      }
      if (photoz_splines[i] == NULL)
      {
        log_fatal("fail allocation");
        exit(1);
      }
    }

    FILE* ein = fopen(redshift.clusters_REDSHIFT_FILE,"r");
    int p = 0;
    for (int i=0; i<zbins; i++)
    {
      fscanf(ein, "%le", &z_v[i]);
      p++;
      if (i > 0 && z_v[i] < z_v[i-1]) 
      {
        break;
      }
      for (int k=0; k<redshift.cluster_nbin; k++)
      {
        double space;
        fscanf(ein,"%le",&space);
      }
    }
    fclose(ein);

    {
      const double zhisto_min = fmax(z_v[0], 1.e-5);
      const double zhisto_max = z_v[p - 1] + (z_v[p - 1] - z_v[0])/((double) p - 1.0);  
      const double da = (zhisto_max - zhisto_min)/((double) zbins);
      for (int i=0; i<zbins; i++)
      { 
        z_v[i] = zhisto_min + (i + 0.5)*da;
      }
    }
    
    { // init the function pf_cluster_histo_n
      double ar[1] = {(double) 0};
      pf_cluster_histo_n(0., (void*) ar);
    }

    double NORM[MAX_SIZE_ARRAYS]; 
    
    #pragma omp parallel for
    for (int i=0; i<redshift.cluster_nbin; i++)
    {
      double ar[1] = {(double) i};

      gsl_function F;
      F.params = (void*) ar;
      F.function = pf_cluster_histo_n;
      const double norm = gsl_integration_glfixed(&F, 1E-5, tomo.cluster_zmax[i] + 1.0, w) / 
        (tomo.cluster_zmax[i] - tomo.cluster_zmin[i]);
      
      if (norm == 0) 
      {
        log_fatal("pz_cluster: norm(nz = %d) = 0", i);
        exit(1);
      }

      for (int k=0; k<zbins; k++)
      { 
        table[i + 1][k] = pf_cluster_histo_n(z_v[k], (void*) ar)/norm;
      }
      NORM[i] = norm;
    }
    
    double norm = 0;
    for (int i=0; i<redshift.cluster_nbin; i++)
    { // calculate normalized overall redshift distribution (without bins), store in table[0][:]
      norm += NORM[i];
    }

    for (int k=0; k<zbins; k++)
    {
      table[0][k] = 0; 
      for (int i=0; i<redshift.cluster_nbin; i++)
      {
        table[0][k] += table[i+1][k]*NORM[i]/norm;
      }
    }

    #pragma omp parallel for
    for (int i=-1; i<redshift.cluster_nbin; i++) 
    {
      int status = gsl_spline_init(photoz_splines[i + 1], z_v, table[i + 1], zbins);
      if (status) 
      {
        log_fatal(gsl_strerror(status));
        exit(1);
      }
    }
  }

  if (nz > redshift.cluster_nbin - 1 || nz < -1)
  {
    log_fatal("pz_cluster(z, %d) outside redshift.cluster_nbin range", nz);
    exit(1);
  }

  double res;
  if (zz <= z_v[0] || zz >= z_v[zbins - 1]) 
  {
    res = 0.0;
  }
  else
  {
    double result = 0.0;
    int status = gsl_spline_eval_e(photoz_splines[nz + 1], zz, NULL, &result);
    if (status) 
    {
      log_fatal(gsl_strerror(status));
      exit(1);
    }
    res = result;
  }
  return res;
}

double dV_cluster(double z, void* params)
{
  double* ar = (double*) params;
  const int nz = ar[0];
  const double a = 1.0/(1.0 + z);
  struct chis chidchi = chi_all(a);
  const double hoverh0 = hoverh0v2(a, chidchi.dchida);
  const double fK = f_K(chidchi.chi);
  return (fK*fK/hoverh0)*pz_cluster(z, nz);
}

double norm_z_cluster(const int nz)
{
  static double cache_cosmo_params;
  static double* table;
  static gsl_integration_glfixed_table* w = 0;

  const int N_z = redshift.clustering_nbin;
  const double zmin = tomo.cluster_zmin[nz];
  const double zmax = tomo.cluster_zmax[nz];
  
  if (table == 0) 
  {
    table = (double*) malloc(sizeof(double)*N_z);
    const size_t nsize_integration = 2500 + 50 * (Ntable.high_def_integration);
    w = gsl_integration_glfixed_table_alloc(nsize_integration);
  }
  if (recompute_cosmo3D(C))
  {
    { // init static vars only 
      const int i = 0;
      double params[1] = {0.0};
      dV_cluster(tomo.cluster_zmin[i], (void*) params);
    }
    #pragma omp parallel for
    for (int i=0; i<N_z; i++)
    {
      double ar[1] = {(double) i};

      gsl_function F;
      F.params = (void*) ar;
      F.function = dV_cluster;
      table[i] = gsl_integration_glfixed(&F, zmin, zmax, w);
    }
    update_cosmopara(&C);
  }
  return table[nz];
}

double zdistr_cluster(const int nz, const double z)
{ //simplfied selection function, disregards evolution of N-M relation+mass function within z bin
  double params[1] = {nz};
  return dV_cluster(z, (void*) params)/norm_z_cluster(nz);
}


double int_for_g_lens_cl(double aprime, void* params)
{
  double* ar = (double*) params;
  const int ni = (int) ar[0];
  const double a = ar[1];
  const int nl = (int) ar[2];
  
  if (ni < -1 || ni > redshift.cluster_nbin - 1)
  {
    log_fatal("invalid bin input ni = %d (max %d)", ni, redshift.cluster_nbin);
    exit(1);
  }
  if (nl < 0 || nl > Cluster.N200_Nbin - 1)
  {
    log_fatal("invalid bin input ni = %d (max %d)", nl, Cluster.N200_Nbin);
    exit(1);
  } 
  
  const double zprime = 1.0/aprime - 1.0;
  const double chi1 = chi(a);
  const double chiprime = chi(aprime);
  return zdistr_cluster(zprime, ni)*f_K(chiprime - chi1)/f_K(chiprime)/(aprime*aprime);
}

double g_lens_cluster(const double a, const int nz, const int nl)
{ 
  static double cache_cosmo_params;
  static double** table = 0;
  static gsl_integration_glfixed_table* w = 0;

  const int N_z = redshift.cluster_nbin;
  const int N_l = Cluster.N200_Nbin;
  if (nl < 0 || nl > N_l - 1)
  {
    log_fatal("invalid bin input ni = %d (max %d)", nl, N_l);
    exit(1);
  } 
  
  const double amin = 1.0/(tomo.cluster_zmax[redshift.cluster_nbin - 1] + 1.);
  const double amax = 0.999999;
  const double da = (amax - amin)/((double) Ntable.N_a - 1.0);
  
  if (table == 0) 
  {
    table = (double**) malloc(sizeof(double*)*(redshift.cluster_nbin + 1));
    for (int i=0; i<redshift.cluster_nbin+1; i++)
    {
      table[i] = (double*) malloc(sizeof(double)*Ntable.N_a);
    }

    const size_t nsize_integration = 300 + 50 * (Ntable.high_def_integration);
    w = gsl_integration_glfixed_table_alloc(nsize_integration);
  }

  if (recompute_cosmo3D(C)) // there is no nuisance bias/sigma parameters yet
  {
    { // init static vars only, if j=-1, no tomography is being done
      double ar[3];
      ar[0] = (double) -1; // j = -1 no tomography is being done
      ar[1] = amin;
      ar[2] = (double) 0;
      int_for_g_lens_cl(amin, (void*) ar);
      if (N_z > 0)
      {
        ar[0] = (double) 0;
        int_for_g_lens_cl(amin, (void*) ar);
      }
    }

    #pragma omp parallel for collapse(2)
    for (int j=-1; j<N_z; j++) 
    { 
      for (int i=0; i<Ntable.N_a; i++) 
      {
        const double aa = amin + i*da;
        double ar[3];
        ar[0] = (double) j; 
        ar[1] = aa;
        ar[2] = nl;

        gsl_function F;
        F.params = ar;
        F.function = int_for_g_lens_cl;

        table[j + 1][i] = 
          gsl_integration_glfixed(&F, 1.0/(redshift.shear_zdist_zmax_all + 1.0), aa, w);
      }      
    } 

    update_cosmopara(&C);
  }

  if (nz < -1 || nz > N_z - 1)
  {
    log_fatal("invalid bin input ni = %d (max %d)", nz, N_z);
    exit(1);
  }
  if (a < amin || a > amax)
  {
    log_fatal("a = %e outside look-up table range [%e,%e]", a, amin, amax);
    exit(1);
  }
  return interpol(table[nz + 1], Ntable.N_a, amin, amax, da, a, 1.0, 1.0); 
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Cluster-Galaxy Cross Clustering
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

int ZCGCL1(const int ni)
{
  if (ni < 0 || ni > tomo.cg_clustering_Npowerspectra - 1)
  {
    log_fatal("invalid bin input ni = %d (max %d)", ni, tomo.cg_clustering_Npowerspectra);
    exit(1);
  }
  // we assume cluster bin = galaxy bin (no cross)
  return tomo.external_selection_cg_clustering[ni];
}

int ZCGCL2(const int nj)
{
  if (nj < 0 || nj > tomo.cg_clustering_Npowerspectra - 1)
  {
    log_fatal("invalid bin input ni = %d (max %d)", nj, tomo.cg_clustering_Npowerspectra);
    exit(1);
  }
  // we assume cluster bin = galaxy bin (no cross)
  return tomo.external_selection_cg_clustering[nj];
}

int N_CGCL(const int ni, const int nj)
{ // ni = Cluster Nbin, nj = Galaxy Nbin
  static int N[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS] = {{-42}};
  if (N[0][0] < 0)
  {
    int n = 0;
    for (int i=0; i<redshift.cluster_nbin; i++)
    {
      for (int j=0; j<redshift.clustering_nbin; j++)
      {
        if (i == j) // we are not considering cross spectrum
        {
          for (int k=0; k<tomo.cg_clustering_Npowerspectra; k++)
          {
            if (i == tomo.external_selection_cg_clustering[k])
            {
              N[i][j] = n;
              n++;
            }
            else
            {
              N[i][j] = -1;
            }
          }
        }
        else
        {
          N[i][j] = -1;
        }
      }
    }
  }
  if (ni < 0 || ni > redshift.cluster_nbin - 1 || nj < 0 || nj > redshift.clustering_nbin - 1)
  {
    log_fatal("invalid bin input (ni (cluster nbin), nj (galaxy nbin)) = (%d, %d)", ni, nj);
    exit(1);
  }
  return N[ni][nj];
}


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Cluster-Galaxy Lensing bins (redshift overlap tests)
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

int test_zoverlap_c(int zc, int zs) // test whether source bin zs is behind lens bin zl
{
  if (redshift.shear_photoz < 4 && tomo.cluster_zmax[zc] <= tomo.shear_zmin[zs]) 
  {
    return 1;
  }
  if (redshift.shear_photoz == 4 && tomo.cluster_zmax[zc] < zmean_source(zs)) 
  {
    return 1;
  }
  return 0;
}

int ZCL(int ni) 
{
  static int N[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS] = {-42};
  if (N[0] < -1) 
  {
    int n = 0;
    for (int i=0; i<redshift.cluster_nbin; i++) 
    {
      for (int j=0; j<redshift.shear_nbin; j++) 
      {
        if (test_zoverlap_c(i, j)) 
        {
          N[n] = i;
          n++;
        }
      }
    }
  }
  if (ni < 0 || ni > tomo.cgl_Npowerspectra - 1)
  {
    log_fatal("invalid bin input ni = %d (max %d)", ni, tomo.cgl_Npowerspectra);
    exit(1);
  }
  return N[ni];
}

int ZCS(int nj) 
{
  static int N[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS] = {-42};
  if (N[0] < -1) 
  {
    int n = 0;
    for (int i=0; i<redshift.cluster_nbin; i++) 
    {
      for (int j=0; j<redshift.shear_nbin; j++) 
      {
        if (test_zoverlap_c(i, j)) 
        {
          N[n] = j;
          n++;
        }
      }
    }
  }
  if (nj < 0 || nj > tomo.cgl_Npowerspectra - 1)
  {
    log_fatal("invalid bin input nj = %d (max %d)", nj, tomo.cgl_Npowerspectra);
    exit(1);
  }
  return N[nj];
}

int N_cgl(int ni, int nj) 
{
  static int N[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS] = {{-42}};
  if (N[0][0] < 0) 
  {
    int n = 0;
    for (int i=0; i<redshift.cluster_nbin; i++) 
    {
      for (int j=0; j<redshift.shear_nbin; j++) 
      {
        if (test_zoverlap_c(i, j)) 
        {
          N[i][j] = n;
          n++;
        } 
        else 
        {
          N[i][j] = -1;
        }
      }
    }
  }
  if (ni < 0 || ni > redshift.cluster_nbin - 1 || nj < 0 || nj > redshift.shear_nbin - 1)
  {
    log_fatal("invalid bin input (ni, nj) = (%d, %d)", ni, nj);
    exit(1);
  }
  return N[ni][nj];
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Cluster Clustering (redshift overlap tests)
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

int ZCCL1(const int ni)
{ // find ZCCL1 of tomography combination (zccl1, zccl2) constituting c-clustering tomo bin Nbin
  static int N[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS] = {-42};
  if (N[0] < -1) 
  {
    int n = 0;
    for (int i=0; i<redshift.cluster_nbin; i++) 
    {
      for (int j=i; j<redshift.cluster_nbin; j++) 
      {
        N[n] = i;
        n++;
      }
    }
  }
  if (ni < 0 || ni > tomo.cc_clustering_Npowerspectra - 1)
  {
    log_fatal("invalid bin input ni = %d (max %d)", ni, tomo.cc_clustering_Npowerspectra);
    exit(1);
  }
  return N[ni];
}

int ZCCL2(const int nj)
{ // find ZCCL2 of tomography combination (zcl1, zcl2) constituting c-clustering tomo bin Nbin
  static int N[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS] = {-42};
  if (N[0] < -1) 
  {
    int n = 0;
    for (int i=0; i<redshift.cluster_nbin; i++) 
    {
      for (int j=i; j<redshift.cluster_nbin; j++) 
      {
        N[n] = j;
        n++;
      }
    }
  }
  if (nj < 0 || nj > tomo.cc_clustering_Npowerspectra - 1)
  {
    log_fatal("invalid bin input nj = %d (max %d)", nj, tomo.cc_clustering_Npowerspectra);
    exit(1);
  }
  return N[nj];
}

int N_CCL(const int ni, const int nj)
{
  static int N[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS] = {{-42}};
  if (N[0][0] < -1) 
  {
    int n = 0;
    for (int i=0; i<redshift.cluster_nbin; i++) 
    {
      for (int j=i; j<redshift.cluster_nbin; j++) 
      {
        N[i][j] = n;
        N[j][i] = n;
        n++;
      }
    }
  }
  if (ni < 0 || ni > redshift.cluster_nbin - 1 || nj < 0 || nj > redshift.cluster_nbin - 1)
  {
    log_fatal("invalid bin input (ni, nj) = (%d, %d)", ni, nj);
    exit(1);
  }
  return N[ni][nj];
}
