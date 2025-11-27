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

//Halo exclusion routine//
 

double P_cluster_x_cluster_clustering_exclusion_constant_lambd_exact(double k, double a, int N_lambda1, int N_lambda2){
       double z = 1./a-1;
       double cluster_bias1, cluster_bias2;
       double pk, R, VexclWR;
       //static int count =1;
       cluster_bias1 = weighted_bias(N_lambda1, z); 
       if (N_lambda1==N_lambda2){
            cluster_bias2 = cluster_bias1;
       }
       else{
          cluster_bias2 = weighted_bias(N_lambda2, z); 
       }
       R=1.5*pow(0.25*(Cluster.N_min[N_lambda1]+Cluster.N_min[N_lambda2]+Cluster.N_max[N_lambda1]+Cluster.N_max[N_lambda2])/100., 0.2)/cosmology.coverH0/a; // RedMaPPer exclusion radius (Rykoff et al. 2014  eq4) in units coverH0 [change to comoving]
       //R = pow((3*mass_mean(N_lambda1, z)/(M_PI*4*(200*cosmology.rho_crit*cosmology.Omega_m))), (1./3.));;
       //R = pow((3*mass_mean(N_lambda1, z)/(M_PI*4*(30*cosmology.rho_crit*cosmology.Omega_m))), (1./3.))/a;
       //printf("R_compare: R_200 %e, Rperco %e \n", R2, R);
       VexclWR = 4*M_PI*(sin(k*R) - k*R*cos(k*R))/pow(k, 3.);
       double cutoff =1.;
       //if (k*R>cutoff){ // to avoid high k oscillation. 1/x**2 damping term is motivated by the envelop of 3(sinkx-kx*coskx)/kx**3
       //     pk = pow(cutoff/R,2)/pow(k,2); 
       //     k = cutoff/R;
       //}
       if (k*R> cutoff){
            pk = Pdelta(k,a)*cluster_bias1*cluster_bias2;
            double Pdeltacutoff = Pdelta(cutoff/R,a)*cluster_bias1*cluster_bias2;
            double VexclWRcutoff = (4*M_PI*(sin(cutoff) - cutoff*cos(cutoff))/pow(cutoff/R, 3.));
            double Pexclusioncutoff = (pk_halo_with_exclusion(cutoff/R, R, a, 1, 1, 1)+VexclWRcutoff)*cluster_bias1*cluster_bias2-VexclWRcutoff;
            double kcutoff = cutoff/R;
            //pk = Pexclusioncutoff*(pk-VexclWR)/(Pdeltacutoff-VexclWRcutoff)*(VexclWR)/VexclWRcutoff;
            pk = (pk-VexclWR-VexclWR*(-1*Pexclusioncutoff+(Pdeltacutoff-VexclWRcutoff))/VexclWRcutoff)*pow((k/kcutoff),-0.7);
            return pk;
       }

       if(R==0){
            pk = Pdelta(k,a)*cluster_bias1*cluster_bias2;
       }else{
            pk = (pk_halo_with_exclusion(k, R, a, 1, 1, 1)+VexclWR)*cluster_bias1*cluster_bias2-VexclWR; // Check it out!! This is my cool trick!! 
       }
       return pk;
}

double P_cluster_x_cluster_clustering_exclusion_constant_lambd_tab(double k, double a, int N_lambda1, int N_lambda2, double linear){
          if(linear>0) return  P_cluster_x_cluster_clustering_mass_given_Dlambda_obs(k, a, N_lambda1, N_lambda2, linear);
          static cosmopara C;
          static nuisancepara N;
          static int N_lambda1_in=-1;
          static int N_lambda2_in=-1;
          static double logkmin = 0., logkmax = 0., dk = 0., da = 0.;
          static double **table_P_NL=0;
          const double amin = 1./(1+tomo.cluster_zmax[tomo.cluster_Nbin-1]);
          const double amax = 1./(1+tomo.cluster_zmin[0]-1E-6);
          double klog,val;
          int i,j;
          double kin;
          logkmin = log(1E-2);
          logkmax = log(1E8);
          dk = (logkmax - logkmin)/(Ntable_cluster.N_k_exclusion_pk_for_cell-1.);
          da = (amax - amin)/(Ntable_cluster.N_a-1.);

          if (recompute_DESclusters(C, N)|| (N_lambda1_in != N_lambda1)|| (N_lambda2_in != N_lambda2)){
            update_cosmopara(&C);
            update_nuisance(&N);
            N_lambda1_in = N_lambda1;
            N_lambda2_in = N_lambda2;
            if (table_P_NL!=0) free_double_matrix(table_P_NL,0, Ntable_cluster.N_a-1, 0, Ntable_cluster.N_k_exclusion_pk_for_cell-1);
            table_P_NL = create_double_matrix(0, Ntable_cluster.N_a-1, 0, Ntable_cluster.N_k_exclusion_pk_for_cell-1);     
            double aa = amin;
            for (i=0; i<Ntable_cluster.N_a; i++, aa +=da) { 
                for (j=0; j<Ntable_cluster.N_k_exclusion_pk_for_cell; ++j) { 
                    if(aa>1.0) aa=1.0;
                    kin   = exp(logkmin+j*dk);
                    table_P_NL[i][j] = log(pow(kin,3)*pow(aa, 0.5)*P_cluster_x_cluster_clustering_exclusion_constant_lambd_exact(kin, aa, N_lambda1, N_lambda2)+1E8);
                }
            }

          }
          klog = log(k);
          val = interpol2d(table_P_NL, Ntable_cluster.N_a, amin, amax, da, a, Ntable_cluster.N_k_exclusion_pk_for_cell, logkmin, logkmax, dk, klog, 1.0, 1.0);
          return (exp(val)-1E8)/k/k/k*pow(a, -0.5);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double pcc_given_lambda_nointerp(
    const double k, 
    const double a, 
    const int nl1, 
    const int nl2, 
    const int linear
  )
{
  const double z   = 1./a - 1.;
  const double cb1 = weighted_bias(nl1, z); 
  const double cb2 = (nl1 == nl2) ? cb1 : weighted_bias(nl2, z);
  const double pk = (linear == 1) ? p_lin(k,a) : Pdelta(k,a);
  return cb1*cb2*pk;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double pcc_with_excl_given_lambda_nointerp(
    const double k, 
    const double a, 
    const int nl1, 
    const int nl2
  )
{
  const double z   = 1./a - 1.;
  const double cb1 = weighted_bias(nl1, z); 
  const double cb2 = (nl1 == nl2) ? cb1 : weighted_bias(nl2, z);
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
    const double coff = 1.; // cff = cut off
    const double kcff = coff/R;
    if (k > kcoff) {
      const double VexclWRcff = 4*M_PI*(sin(coff) -coff*cos(coff))/(kcff*kcff*kcff);
      const double pcccff = (phh_with_excl(kcff,R,a,1,1,1) + VexclWRcff)*cb1*cb2 - VexclWRcff;
      pcc  = Pdelta(k,a)*cb1*cb2 - VexclWR;
      pcc -= (-pcccff + (Pdelta(kcff,a)*cb1*cb2 - VexclWRcoff))*VexclWR/VexclWRcoff;
      pcc *= pow((k/kcoff),-0.7); 
    }
    else {
      // original cosmolike: Check it out!! This is my cool trick!! 
      pcc = (phh_with_excl(k,R,a,1,1,1) + VexclWR)*cb1*cb2 - VexclWR; 
    }
  }
  return pcc;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double pcc_with_excl_given_lambda(
    const double k, 
    const double a, 
    const int nl1, 
    const int nl2, 
    const int linear
  )
{



  static cosmopara C;
  static nuisancepara N;
  static int N_lambda1_in=-1;
  static int N_lambda2_in=-1;
  static double logkmin = 0., logkmax = 0., dk = 0., da = 0.;
  static double **table_P_NL=0;
  const double amin = 1./(1+tomo.cluster_zmax[tomo.cluster_Nbin-1]);
  const double amax = 1./(1+tomo.cluster_zmin[0]-1E-6);
  double klog,val;
  int i,j;
  double kin;
  logkmin = log(1E-2);
  logkmax = log(1E8);
  dk = (logkmax - logkmin)/(Ntable_cluster.N_k_exclusion_pk_for_cell-1.);
  da = (amax - amin)/(Ntable_cluster.N_a-1.);

  if (recompute_DESclusters(C, N)|| (N_lambda1_in != N_lambda1)|| (N_lambda2_in != N_lambda2)){
    update_cosmopara(&C);
    update_nuisance(&N);
    N_lambda1_in = N_lambda1;
    N_lambda2_in = N_lambda2;
    if (table_P_NL!=0) free_double_matrix(table_P_NL,0, Ntable_cluster.N_a-1, 0, Ntable_cluster.N_k_exclusion_pk_for_cell-1);
    table_P_NL = create_double_matrix(0, Ntable_cluster.N_a-1, 0, Ntable_cluster.N_k_exclusion_pk_for_cell-1);     
    double aa = amin;
    for (i=0; i<Ntable_cluster.N_a; i++, aa +=da) { 
        for (j=0; j<Ntable_cluster.N_k_exclusion_pk_for_cell; ++j) { 
            if(aa>1.0) aa=1.0;
            kin   = exp(logkmin+j*dk);
            table_P_NL[i][j] = log(pow(kin,3)*pow(aa, 0.5)*P_cluster_x_cluster_clustering_exclusion_constant_lambd_exact(kin, aa, N_lambda1, N_lambda2)+1E8);
        }
    }

  }
 if(linear>0) return  pcc_given_lambda_nointerp(k, a, N_lambda1, N_lambda2, linear);

  klog = log(k);
  val = interpol2d(table_P_NL, Ntable_cluster.N_a, amin, amax, da, a, Ntable_cluster.N_k_exclusion_pk_for_cell, logkmin, logkmax, dk, klog, 1.0, 1.0);
  return (exp(val)-1E8)/k/k/k*pow(a, -0.5);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double int_pcm_given_lambda_obs_1halo(double logM, void* params) {
  double* ar = (double *) params;
  const int nl   = (int) ar[0];
  const double a = ar[1];
  const double k = ar[2];
  const double M = exp(logM); 
  const double z = 1.0/a-1.0; 
  const double cmr   = c_m_relation_tab(M, a);
  const double unfw  = u_nfw_c(cmr, k, M, a);
  const double PCM1H = M/(cosmology.rho_crit*cosmology.Omega_m)*unfw;
  return M*PCM1H*p_lambda_obs_given_m(nl, M, 1.0/a-1.0);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double pcm_given_lambda_obs_1halo_nointerp(
    const double k, 
    const double a, 
    const int nl, // cluster richness bin
    const int ni, // cluster redshift bin
    const int init
  )
{
  static double cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;
  if (nl < 0 || nl > Cluster.n200_nbin - 1 ||
      ni < 0 || ni > redshift.clusters_nbin - 1) {
    log_fatal("error in bin number (nl,ni) = [%d,%d]", nl,ni);
    exit(1); 
  }
  if (NULL == w || fdiff(cache[0], Ntable.random)) {
    const size_t szint = 50 + 40 * abs(Ntable.high_def_integration);
    if (w != NULL)  {
      gsl_integration_glfixed_table_free(w);
    }
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  const double norm = n_lambda_obs_z(nl, 1./a-1.0);
  double ar[3] = {(double) nl, a, k};
  
  double res = 0.0; 
  if (1 == init) {
    (void) int_pcm_given_lambda_obs_1halo(amin, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_pcm_given_lambda_obs_1halo;
    res = gsl_integration_glfixed(&F, log(pow(10.,12)), log(pow(10.,15.9)), w);
  }
  return (1 == init) ? 0.0 : 
          (norm<1.E-14) ? 0.0 : 
            (res/norm>1.E5) ? 0.0 : res/norm;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double pcm_given_lambda_obs_1halo(
    const double k, 
    const double a, 
    const int nl,   // cluster richness bin
    const int ni   // cluster redshift bin
  )
{
  static double cache[MAX_SIZE_ARRAYS];
  static double*** table = NULL;
  static double lim[6*MAX_SIZE_ARRAYS];
  // ---------------------------------------------------------------------------
  const int NSIZE = Cluster.n200_nbin*redshift.clusters_nbin;
  const int nlnk = Ntable.nlnk_pcm_given_lambda_obs_1halo;
  const int na = Ntable.na_pcm_given_lambda_obs_1halo;
  if (NULL == table || fdiff(cache[3], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double***) malloc3d(NSIZE, nlnk, na);
  }
  // ---------------------------------------------------------------------------
  if (fdiff(cache[0], cosmology.random) || 
      fdiff(cache[1], nuisance.random_clusters) ||
      fdiff(cache[2], redshift.random_clusters) ||
      fdiff(cache[3], Ntable.random))
  {
    for (int i=0; i<redshift.clusters_nbin; i++) {
      lim[6*i+0] = 1./(1.+tomo.cluster_zmax[i]);                 // amin
      lim[6*i+1] = 1./(1.+tomo.cluster_zmin[i]);                 // amax
      lim[6*i+2] = (lim[6*i+1]-lim[6*i+0])/((double) na - 1.);   // da
      lim[6*i+3] = log(limits.k_min_cH0);                        // logkmin
      lim[6*i+4] = log(limits.k_max_cH0);                        // logkmax
      lim[6*i+5] = (lim[6*i+4]-lim[6*i+3])/((double) nlnk - 1.); // dlnk 
    }
    // -------------------------------------------------------------------------
    for (int k=0; k<Cluster.n200_nbin; k++) { // init static vars
      for (int l=0; l<redshift.clusters_nbin; l++) { 
        const double kin = exp(lim[6*l+3]);
        const double ain = lim[6*l+0];
        (void) pcm_given_lambda_obs_1halo_nointerp(kin, ain, k, l, 1); 
      }
    }
    #pragma omp parallel for collapse(4) schedule(static,1)
    for (int k=0; k<Cluster.n200_nbin; k++) { 
      for (int l=0; l<redshift.clusters_nbin; l++) { 
        for (int i=0; i<na; i++) { 
          for (int j=0; j<nlnk; j++) { 
            const double kin = exp(lim[6*l+3] + j*lim[6*l+5]);
            const double ain = lim[6*l+0] + i*lim[6*l+2];
            double tmp = pcm_given_lambda_obs_1halo_nointerp(kin, ain, k, l, 0); 
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
    log_fatal("error in bin number (nl,ni) = [%d,%d]", nl, ni);
    exit(1); 
  }
  const int q = nl*redshift.clusters_nbin + ni;
  if (q < 0 || q > NSIZE - 1) {
    log_fatal("internal logic error in selecting bin number");
    exit(1);
  }

  double res = 0.0;
  const double amin   = lim[6*ni+0];
  const double amax   = lim[6*ni+1];
  const double da     = lim[6*ni+2];
  const double lnkmin = lim[6*ni+3];
  const double lnkmax = lim[6*ni+4];
  const double dlnk   = lim[6*ni+5];
  return ((lnk > lnkmin) & (lnk < lnkmax)) ?
    interpol2d(table[q],nlnk,lnkmin,lnkmax,dlnk,log(k),na,amin,amax,da,a) : 0.0;
   
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#ifdef darkemu
double calculate_p_darkemu(double mass, double k, double z){
    static cosmopara C;
    static PyObject *Call_darkemu_set;
    static PyObject *pModule;
    static PyObject *darkemu_pk;
    PyObject *pArgs2;
    PyObject *darkemu_set, *call_darkemu_set, *call_darkemu_pk;
    PyObject *pArgs;

    /* Error checking of pName left out */
    if (recompute_cosmo3D(C)){
        printf("Initialize\n\n\n\n");
        Py_Initialize();
        PyObject *sys_path = PySys_GetObject("path");
        PyList_Append(sys_path, PyUnicode_FromString("/home/users/chto/code/lighthouse/cpp"));
        pModule = PyImport_ImportModule("call_darkemu");
        if (pModule == NULL) {
            printf("ERROR importing module");
            assert(0);
        }
        update_cosmopara(&C);
        darkemu_set= PyObject_GetAttrString(pModule, "darkemu_set");
        pArgs = PyTuple_New(6);
        PyTuple_SetItem(pArgs, 0, PyFloat_FromDouble(cosmology.omb*cosmology.h0*cosmology.h0));
        PyTuple_SetItem(pArgs, 1, PyFloat_FromDouble((cosmology.Omega_m-cosmology.Omega_nu-cosmology.omb)*cosmology.h0*cosmology.h0));
        PyTuple_SetItem(pArgs, 2, PyFloat_FromDouble(1-cosmology.Omega_m));
        PyTuple_SetItem(pArgs, 3, PyFloat_FromDouble(log(pow(10,10)*cosmology.A_s)));
        PyTuple_SetItem(pArgs, 4, PyFloat_FromDouble(cosmology.n_spec));
        PyTuple_SetItem(pArgs, 5, PyFloat_FromDouble(cosmology.w0));
        call_darkemu_set=PyObject_CallObject(darkemu_set,pArgs);
        if (call_darkemu_set == NULL) {
            printf("ERROR bad call_darkemu_set");
            assert(0);
        }
        Call_darkemu_set = call_darkemu_set;
        darkemu_pk= PyObject_GetAttrString(pModule, "darkemu_pk");
    } 
    pArgs2 = PyTuple_New(4);
    PyTuple_SetItem(pArgs2, 0, PyFloat_FromDouble(k/cosmology.coverH0));
    PyTuple_SetItem(pArgs2, 1, PyFloat_FromDouble(mass));
    PyTuple_SetItem(pArgs2, 2, PyFloat_FromDouble(z));
    PyTuple_SetItem(pArgs2, 3, Call_darkemu_set);
    call_darkemu_pk=PyObject_CallObject(darkemu_pk,pArgs2);
    double f1output = PyFloat_AsDouble(call_darkemu_pk)/pow(cosmology.coverH0,3);
    //Py_Finalize();
    return f1output;
}

double calculate_p_darkemu_tab(double mass, double k, double z){
   static cosmopara C;
   static nuisancepara N;
   static double **table;
   static double da, amin, amax;
   static double kin=-1000;
    static double dm =0., dz =0., logmmin = 12.0,logmmax = 15.9;
    static int NM = 20, N_a=30;
   if (table==0) {
      table = create_double_matrix(0, NM-1, 0, N_a-1);
      double zmin, zmax;
      zmin = fmax(tomo.cluster_zmin[0]-0.05,0.01); zmax = tomo.cluster_zmax[tomo.cluster_Nbin-1]+0.05;
      amin = 1./(1.+zmax); amax = 1./(1.+zmin);
      da = (amax-amin)/(N_a-1.);
      dm = (logmmax-logmmin)/(NM-1.);
      //printf("zmin = %.2f  zmax = %.2f\n",zmin, zmax);
   }
   int n=0;
   if (recompute_DESclusters(C, N) || (kin!=k)){
      for (double lgM = logmmin; lgM < logmmax; lgM+= dm){
         double aa = amin;
         for (int i = 0; i < N_a; i++, aa+=da){
            table[n][i] = calculate_p_darkemu(pow(10.,lgM), k, 1./aa-1.);
         }
         n+=1;
      }
      update_cosmopara(&C);
      update_nuisance(&N);
      kin =k;
   }

    if (z>1/amax-1 && z < 1/amin-1 && mass > pow(10.,logmmin) && mass < pow(10.,logmmax)){
      return interpol2d(table, NM, logmmin, logmmax, dm, log10(mass), N_a, amin, amax, da, 1.0/(z+1), 1.0, 1.0);
    }
    else{
        return 0;
    }

}



double int_darkemu(double logM, void *params){
   double mass = exp(logM);
   double *array = (double*) params; //n_lambda, z,k
   double z=array[1];
   double k = array[2];
   double P_darkemu=calculate_p_darkemu_tab(mass, k, z);
   return mass*(P_darkemu)*massfunc_probability_observed_richness_given_mass_tab((int) array[0], mass , array[1]);
}


double P_cluster_mass_given_Dlambda_obs_darkemu(double k, double a, int N_lambda, int nz_cluster, int nz_galaxy){
   double z = 1./a-1;
   double params[3] = {1.0*N_lambda, z, k};
   double result; 
   double norm =  n_lambda_obs_z_tab(N_lambda, z);
   result = int_gsl_integrate_low_precision(int_darkemu,params,log(pow(10.,12.)),log(pow(10.,15.9)),NULL,1000)/norm; 
   if(result>1E5) return 0;

   if(isnan(result)) return 0;
   else return result;
}
double P_cluster_mass_given_Dlambda_obs_darkemu_tab(double k, double a, int N_lambda, int nz_cluster, int nz_galaxy){
    static cosmopara C;
    static nuisancepara N;
    static double dk = 0., da = 0.;
    static int N_lambda_in, nz_cluster_in, nz_galaxy_in;
    static double logkmin, logkmax;
    static double **table_P_k_a=0;
    double klog,val, result; 
    double zmin, zmax, aa, kin;
    static double amin, amax;
    int N_a = 30;
    int i, j;
    if (recompute_DESclusters(C, N) || (N_lambda_in != N_lambda) || (nz_cluster_in != nz_cluster) || (nz_galaxy_in != nz_galaxy)){
        update_cosmopara(&C);
        update_nuisance(&N);
        N_lambda_in = N_lambda; 
        nz_cluster_in = nz_cluster; 
        nz_galaxy_in = nz_galaxy;
        zmin = tomo.cluster_zmin[nz_cluster];
        zmax = tomo.cluster_zmax[nz_cluster];
        amin = 1./(1.+zmax); amax = 1./(1.+zmin);
        da = (amax-amin)/(N_a-1.);

        if (table_P_k_a!=0) free_double_matrix(table_P_k_a,0, Ntable.N_ell-1, 0, N_a-1);
        table_P_k_a = create_double_matrix(0, Ntable.N_ell-1, 0, N_a-1);     
        logkmin = log(limits.k_min_cH0);
        logkmax = log(limits.k_max_cH0);
        dk = (logkmax - logkmin)/(Ntable.N_ell-1);     
        for (i=0; i<Ntable.N_ell; i++) { 
                aa = amin;
                printf("%d, %d\n", i, Ntable.N_ell);
                for (j=0; j<N_a; ++j, aa+=da) { 
                    kin   = exp(logkmin+i*dk);
                    result = P_cluster_mass_given_Dlambda_obs_darkemu(kin, aa, N_lambda, nz_cluster, nz_galaxy); 
                    if(result<0) result=0;
                //if (result ==0) table_P_k_a[i][j] = -1E10; 
                    table_P_k_a[i][j] = result; 
                }
        }
        
    }
    klog = log(k);
    if ((klog > logkmin)&(klog < logkmax)) val = interpol2d(table_P_k_a, Ntable.N_ell, logkmin, logkmax, dk, klog, N_a, amin, amax, da, a, 1.0, 1.0);
    else val=0;
    return val;
}
double P_cluster_x_galaxy_clustering_mass_given_Dlambda_obs(double k, double a, int N_lambda, int nz_cluster, int nz_galaxy, double linear){
   double z = 1./a-1.;
   double cluster_bias = weighted_bias(N_lambda, z); 
   if(linear>0)
        return cluster_bias * gbias.b1_function(z, nz_galaxy)* p_lin(k,a);
   else return cluster_bias * gbias.b1_function(z, nz_galaxy)* Pdelta(k,a);
}

#endif
