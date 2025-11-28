#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_integration.h>

#include "basics.h"
#include "bias.h"
#include "cosmo3D.h"
#include "cluster_util.h"
#include "halo.h"
#include "pt_cfastpt.h"
#include "recompute.h"
#include "radial_weights.h"
#include "redshift_spline.h"
#include "structs.h"
#include "tinker_emulator.h"

#include "log.c/src/log.h"

static int GSL_WORKSPACE_SIZE = 250;
static double M_PIVOT = 5E14; // Msun/h

static int has_b2_galaxies()
{
  int res = 0;
  for(int i=0; i<tomo.clustering_Nbin; i++) 
  {
    if (gbias.b2[i])
    {
      res = 1;
    }
  }
  return res;
}

// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// BUZZARD binned_P_lambda_obs_given_M
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------

double buzzard_P_lambda_obs_given_M(const double obs_lambda, const double M, const double z)
{        
  const double lnlm0 = nuisance.cluster_MOR[0]; 
  const double Alm = nuisance.cluster_MOR[1];
  const double sigma_lnlm_intrinsic = nuisance.cluster_MOR[2];
  const double Blm = nuisance.cluster_MOR[3];
  
  const double Mpiv = M_PIVOT; //Msun/h
  
  const double lnlm = lnlm0 + (Alm)*log(M/Mpiv) + Blm*log((1 + z)/1.45);
  
  const double sigma_total = (lnlm>0) ? 
    sqrt(sigma_lnlm_intrinsic*sigma_lnlm_intrinsic+(exp(lnlm)-1.)/exp(2*lnlm)) : 
    sqrt(sigma_lnlm_intrinsic*sigma_lnlm_intrinsic);
  
  const double x = 1.0/2.0*(log(obs_lambda)-lnlm)*(log(obs_lambda)-lnlm)/pow(sigma_total,2.0);
  
  return exp(-x)/M_SQRTPI/M_SQRT2/sigma_total/obs_lambda;
}

static double buzzard_P_lambda_obs_given_M_wrapper(double obs_lambda, void* params)
{      
  double* ar = (double*) params;
  const double M = ar[0];
  const double z = ar[1];
  return buzzard_P_lambda_obs_given_M(obs_lambda, M, z);
}

// \int_(bin_lambda_obs_min)^(bin_lambda_obs_max) \dlambda_obs P(\lambda_obs|M)
// (see for example https://arxiv.org/pdf/1810.09456.pdf - eq3 qnd 6) 
double buzzard_binned_P_lambda_obs_given_M(const int nl, const double M, const double z, 
  const int init_static_vars_only)
{
  double params[2] = {M, z};
  const double bin_lambda_obs_min = Cluster.N_min[nl];
  const double bin_lambda_obs_max = Cluster.N_max[nl];
  return (init_static_vars_only == 1) ? 
    buzzard_P_lambda_obs_given_M_wrapper(bin_lambda_obs_min, (void*) params) :
    int_gsl_integrate_medium_precision(buzzard_P_lambda_obs_given_M_wrapper,
      (void*) params, bin_lambda_obs_min, bin_lambda_obs_max, NULL, GSL_WORKSPACE_SIZE);
}

// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// SDSS binned_P_lambda_obs_given_M
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------

// Cocoa: we try to avoid reading of files in the cosmolike_core code 
// Cocoa: (reading is done in the C++/python interface)
void setup_SDSS_P_true_lambda_given_mass(int* io_nintrinsic_sigma, double** io_intrinsic_sigma, 
int* io_natsrgm, double** io_atsrgm, double** io_alpha, double** io_sigma, int io)
{
  static int nintrinsic_sigma;
  static int natsrgm;  
  static double* intrinsic_sigma = NULL;
  static double* atsrgm = NULL;
  static double* alpha = NULL;
  static double* sigma = NULL;

  if (io == 1) // IO == 1 IMPLES THAT IO_XXX WILL COPIED TO LOCAL XXX
  {
    if (intrinsic_sigma != NULL)
    {
      free(intrinsic_sigma);
    }
    else if (atsrgm != NULL)
    {
      free(atsrgm);
    }
    else if (alpha != NULL)
    {
      free(alpha);
    }
    else if (sigma != NULL)
    {
      free(sigma);
    }

    nintrinsic_sigma = (*io_nintrinsic_sigma);
    natsrgm = (*io_natsrgm);

    if (!(nintrinsic_sigma > 5) || !(natsrgm > 5))
    {
      log_fatal("array to small for 2D interpolation");
      exit(1);
    }

    intrinsic_sigma = (double*) malloc(nintrinsic_sigma*sizeof(double));
    atsrgm = (double*) malloc(natsrgm*sizeof(double));
    alpha = (double*) malloc(natsrgm*nintrinsic_sigma*sizeof(double));
    sigma = (double*) malloc(natsrgm*nintrinsic_sigma*sizeof(double));
    if (intrinsic_sigma == NULL || atsrgm == NULL || alpha == NULL || sigma == NULL)
    {
      log_fatal("fail allocation");
      exit(1);
    }
    
    for (int i=0; i<nintrinsic_sigma; i++)
    {
      intrinsic_sigma[i] = (*io_intrinsic_sigma)[i];
      for (int j=0; j<natsrgm; j++)
      {
        if (i == 0) 
        {
          atsrgm[j] = (*io_atsrgm)[j];
        }
        alpha[i*nintrinsic_sigma + j] = (*io_alpha)[i*nintrinsic_sigma + j];
        sigma[i*nintrinsic_sigma + j] = (*io_sigma)[i*nintrinsic_sigma + j];
      }
    }
  }
  else  // IO != 1 IMPLES THAT LOCAL XXX WILL BE COPIED TO IO_XXX
  {
    if (intrinsic_sigma == NULL || atsrgm == NULL || alpha == NULL || sigma == NULL ||
        io_intrinsic_sigma == NULL || io_atsrgm == NULL || io_alpha == NULL || io_sigma == NULL)
    {
      log_fatal("array/pointers not allocated\n");
      exit(1);
    }
    if ((*io_intrinsic_sigma) != NULL)
    {
      free((*io_intrinsic_sigma));
      (*io_intrinsic_sigma) = NULL;
    }
    else if ((*io_atsrgm) != NULL)
    {
      free((*io_atsrgm));
      (*io_atsrgm) = NULL;
    }
    else if ((*io_alpha) != NULL)
    {
      free((*io_alpha));
      (*io_alpha) = NULL;
    }
    else if ((*io_sigma) != NULL)
    {
      free((*io_sigma));
      (*io_sigma) = NULL;
    }

    (*io_intrinsic_sigma) = (double*) malloc(nintrinsic_sigma*sizeof(double));
    (*io_atsrgm) = (double*) malloc(natsrgm*sizeof(double));
    (*io_alpha) = (double*) malloc(nintrinsic_sigma*natsrgm*sizeof(double));
    (*io_sigma) = (double*) malloc(nintrinsic_sigma*natsrgm*sizeof(double));
    if (*io_intrinsic_sigma == NULL || *io_atsrgm == NULL || *io_alpha == NULL || *io_sigma == NULL)
    {
      log_fatal("fail allocation");
      exit(1);
    }

    for (int i=0; i<nintrinsic_sigma; i++)
    {
      (*io_intrinsic_sigma)[i] = intrinsic_sigma[i];
      for (int j=0; j<natsrgm; j++)
      {
        if (i == 0) 
        {
          (*io_atsrgm)[j] = atsrgm[j];
        }
        (*io_alpha)[i*nintrinsic_sigma + j] = alpha[i*nintrinsic_sigma + j];
        (*io_sigma)[i*nintrinsic_sigma + j] = sigma[i*nintrinsic_sigma + j];
      }
    }
  }
}

double SDSS_P_true_lambda_given_mass(const double true_lambda, const double mass, const double z)
{ // SKEW-NORMAL APPROXIMATION eq B1 of https://arxiv.org/pdf/1810.09456.pdf
  static int first = 0;
  static gsl_spline2d* falpha = NULL; // skewness of the skew-normal distribution
  static gsl_spline2d* fsigma = NULL; // variance of the skew-normal distribution
  if (first == 0)
  {
    first = 1;
    int nintrinsic_sigma;
    int natsrgm;
    double** intrinsic_sigma;
    double** atsrgm;
    double** tmp_alpha;
    double** tmp_sigma;
    double* alpha;
    double* sigma;

    intrinsic_sigma = (double**) malloc(1*sizeof(double*));
    atsrgm = (double**) malloc(1*sizeof(double*));
    tmp_alpha = (double**) malloc(1*sizeof(double*));
    tmp_sigma = (double**) malloc(1*sizeof(double*));
    if (intrinsic_sigma == NULL || atsrgm == NULL || tmp_alpha == NULL || tmp_sigma == NULL)
    {
      log_fatal("fail allocation");
      exit(1);
    }
    else 
    {
      (*intrinsic_sigma) = NULL;
      (*atsrgm) = NULL;
      (*tmp_alpha) = NULL;
      (*tmp_sigma) = NULL; 
    }
   
    setup_SDSS_P_true_lambda_given_mass(&nintrinsic_sigma, intrinsic_sigma, &natsrgm, atsrgm, 
      tmp_alpha, tmp_sigma, 0);

    const gsl_interp2d_type* T = gsl_interp2d_bilinear;
    falpha = gsl_spline2d_alloc(T, nintrinsic_sigma, natsrgm);
    fsigma = gsl_spline2d_alloc(T, nintrinsic_sigma, natsrgm);
    if (falpha == NULL || fsigma == NULL)
    {
      log_fatal("fail allocation");
      exit(1);
    }
    { // we don't want to guess the appropriate GSL z array ordering in z = f(x, y) BEGINS
      alpha = (double*) malloc(nintrinsic_sigma*natsrgm*sizeof(double));
      sigma = (double*) malloc(nintrinsic_sigma*natsrgm*sizeof(double));
      if (alpha == NULL || sigma == NULL)
      {
        log_fatal("fail allocation");
        exit(1);
      }
      for (int i=0; i<nintrinsic_sigma; i++) 
      {
        for (int j=0; j<natsrgm; j++) 
        {
          int status = 0;
          status = gsl_spline2d_set(falpha, alpha, i, j, (*tmp_alpha)[i*nintrinsic_sigma+j]);
          if (status) 
          {
            log_fatal(gsl_strerror(status));
            exit(1);
          }
          status = gsl_spline2d_set(fsigma, sigma, i, j, (*tmp_sigma)[i*nintrinsic_sigma+j]);
          if (status) 
          {
            log_fatal(gsl_strerror(status));
            exit(1);
          }
        }
      }
    } // we don't want to guess the appropriate GSL z array ordering in z = f(x, y) ENDS

    int status = 0;
    status = gsl_spline2d_init(falpha, intrinsic_sigma[0], atsrgm[0], alpha, 
      nintrinsic_sigma, natsrgm);
    if (status) 
    {
      log_fatal(gsl_strerror(status));
      exit(1);
    }
    
    status = gsl_spline2d_init(fsigma, intrinsic_sigma[0], atsrgm[0], sigma, 
      nintrinsic_sigma, natsrgm);
    if (status) 
    {
      log_fatal(gsl_strerror(status));
      exit(1);
    }

    free(intrinsic_sigma[0]);
    free(atsrgm[0]);
    free(tmp_alpha[0]);
    free(tmp_sigma[0]);
    free(intrinsic_sigma);
    free(atsrgm);
    free(tmp_alpha);
    free(tmp_sigma);
    free(alpha); // GSL SPLINE 2D copies the array
    free(sigma); // GSL SPLINE 2D copies the array
  }

  const double mass_min = pow(10.0, nuisance.cluster_MOR[0]);
  const double mass_M1 = pow(10.0, nuisance.cluster_MOR[1]);
  const double intrinsic_alpha = nuisance.cluster_MOR[2];
  const double intrinsic_sigma = nuisance.cluster_MOR[3]; //intrisic scatter, mass-richness relation
  
  // atsrgm = average true satellite richness given mass
  const double tmp = (mass - mass_min)/(mass_M1 - mass_min);
  double atsrgm = pow(tmp, intrinsic_alpha)*pow(((1 + z)/1.45), nuisance.cluster_MOR[4]); 
  if (atsrgm > 160) 
  {
    atsrgm = 160;
  }
  else if (atsrgm < 1) 
  {
    atsrgm = 1;
  }
  
  int status = 0;

  double alpha = 0.0;
  status = gsl_spline2d_eval_e(falpha, intrinsic_sigma, atsrgm, NULL, NULL, &alpha);
  if (status) 
  {
    log_fatal(gsl_strerror(status));
    exit(1);
  }
  double sigma = 0.0;
  status = gsl_spline2d_eval_e(fsigma, intrinsic_sigma, atsrgm, NULL, NULL, &sigma);
  if (status) 
  {
    log_fatal(gsl_strerror(status));
    exit(1);
  }

  const double y = 1.0/(M_SQRT2*abs(sigma));
  const double x = (true_lambda - atsrgm)*y;
  const double result1 = exp(-x*x)*y/M_SQRTPI;
  gsl_sf_result result2;
  status = gsl_sf_erfc_e(-alpha*x, &result2);
  if (status) 
  {
    log_fatal(gsl_strerror(status));
    exit(1);
  }  
  return result1*result2.val;
}

// Cocoa: we try to avoid reading of files in the cosmolike_core code 
// Cocoa: (reading is done in the C++/python interface)
void setup_SDSS_P_lambda_obs_given_true_lambda(int* io_nz, double** io_z, int* io_nlambda, 
double** io_lambda, double** io_tau, double** io_mu, double** io_sigma, double** io_fmask, 
double** io_fprj, int io)
{
  static int nz;
  static int nlambda; 
  static double* z = NULL; 
  static double* lambda = NULL;
  static double* tau = NULL;
  static double* mu = NULL;
  static double* sigma = NULL;
  static double* fmask = NULL;
  static double* fprj = NULL;

  if (io == 1) // IO == 1 IMPLES THAT IO_XXX WILL COPIED TO LOCAL XXX
  {
    if (z != NULL)
    {
      free(z);
    }
    else if (lambda != NULL)
    {
      free(lambda);
    }
    else if (tau != NULL)
    {
      free(tau);
    }
    else if (mu != NULL)
    {
      free(mu);
    }
    else if (sigma != NULL)
    {
      free(sigma);
    }
    else if (fmask != NULL)
    {
      free(fmask);
    }
    else if (fprj != NULL)
    {
      free(fprj);
    }

    nz = (*io_nz);
    nlambda = (*io_nlambda);
    if (!(nz > 5) || !(nlambda > 5))
    {
      log_fatal("array to small for 2D interpolation");
      exit(1);
    }

    z =       (double*) malloc(nz*sizeof(double));
    lambda =  (double*) malloc(nlambda*sizeof(double));
    tau =     (double*) malloc(nz*nlambda*sizeof(double));
    mu =      (double*) malloc(nz*nlambda*sizeof(double));
    sigma =   (double*) malloc(nz*nlambda*sizeof(double));
    fmask =   (double*) malloc(nz*nlambda*sizeof(double));
    fprj =    (double*) malloc(nz*nlambda*sizeof(double));
    if (z == NULL || lambda == NULL || tau == NULL || mu == NULL || sigma == NULL || 
      fmask == NULL || fprj == NULL) 
    {
      log_fatal("fail allocation");
      exit(1);
    }
    for (int i=0; i<nz; i++)
    {
      z[i] = (*io_z)[i];
      for (int j=0; j<nlambda; j++)
      {
        if (i == 0) 
        {
          lambda[j] = (*io_lambda)[j];
        }
        tau[i*nz + j] = (*io_tau)[i*nz + j];
        mu[i*nz + j] = (*io_mu)[i*nz + j];
        sigma[i*nz + j] = (*io_sigma)[i*nz + j];
        fmask[i*nz + j] = (*io_fmask)[i*nz + j];
        fprj[i*nz + j] = (*io_fprj)[i*nz + j];
      }
    }
  }
  else
  {
    // IO != 1 IMPLES THAT LOCAL H(Z) WILL BE COPIED TO IO_chi(Z)
    if (z == NULL || lambda == NULL || tau == NULL || mu == NULL || sigma == NULL || fmask == NULL 
        || fprj == NULL || io_lambda == NULL || io_tau == NULL || io_mu == NULL || io_sigma == NULL
        || io_fmask == NULL || io_fprj == NULL)
    {
      log_fatal("array/pointer not allocated");
      exit(1);
    }
    if ((*io_z) != NULL)
    {
      free((*io_z));
      (*io_z) = NULL;
    }
    else if ((*io_lambda) != NULL)
    {
      free((*io_lambda));
      (*io_lambda) = NULL;
    }
    else if ((*io_tau) != NULL)
    {
      free((*io_tau));
      (*io_tau) = NULL;
    }
    else if ((*io_mu) != NULL)
    {
      free((*io_mu));
      (*io_mu) = NULL;
    }
    else if ((*io_sigma) != NULL)
    {
      free((*io_sigma));
      (*io_sigma) = NULL;
    }
    else if ((*io_fmask) != NULL)
    {
      free((*io_fmask));
      (*io_fmask) = NULL;
    }   
    else if ((*io_fprj) != NULL)
    {
      free((*io_fprj));
      (*io_fprj) = NULL;
    }
   
    (*io_z)       = (double*) malloc(nz*sizeof(double));
    (*io_lambda)  = (double*) malloc(nlambda*sizeof(double));
    (*io_tau)     = (double*) malloc(nz*nlambda*sizeof(double));
    (*io_mu)      = (double*) malloc(nz*nlambda*sizeof(double));
    (*io_sigma)   = (double*) malloc(nz*nlambda*sizeof(double));
    (*io_fmask)   = (double*) malloc(nz*nlambda*sizeof(double));
    (*io_fprj)    = (double*) malloc(nz*nlambda*sizeof(double));

    if ((*io_z) == NULL || (*io_lambda) == NULL || (*io_tau) == NULL || (*io_mu) == NULL ||
       (*io_sigma) == NULL || (*io_fmask) == NULL || (*io_fprj) == NULL) 
    {
      log_fatal("fail allocation");
      exit(1);
    }
    for (int i=0; i<nz; i++)
    {
      (*io_z)[i] = z[i];
      for (int j=0; j<nlambda; j++)
      {
        if (i == 0) 
        {
          (*io_lambda)[j] = lambda[j];
        }
        (*io_tau)[i*nz + j] = tau[i*nz + j];
        (*io_mu)[i*nz + j]  = mu[i*nz + j];
        (*io_sigma)[i*nz + j] = sigma[i*nz + j];
        (*io_fmask)[i*nz + j] = fmask[i*nz + j];
        (*io_fprj)[i*nz + j] = fprj[i*nz + j];
      }
    }
  }
}

double SDSS_P_lambda_obs_given_true_lambda(const double observed_lambda, const double true_lambda, 
const double zz) 
{
  static int first = 0;
  static gsl_spline2d* ftau;
  static gsl_spline2d* fmu;
  static gsl_spline2d* fsigma;
  static gsl_spline2d* ffmask;
  static gsl_spline2d* ffprj;

  if (first == 0)
  {
    first = 1;
    int nz;
    int nlambda;
    double** tmp_tau;
    double** tmp_mu;
    double** tmp_sigma;
    double** tmp_fmask;
    double** tmp_fprj;
    double** z;
    double** lambda;
    double* tau;
    double* mu;
    double* sigma;
    double* fmask;
    double* fprj;

    z           = (double**) malloc(1*sizeof(double*));
    lambda      = (double**) malloc(1*sizeof(double*));
    tmp_tau     = (double**) malloc(1*sizeof(double*));
    tmp_mu      = (double**) malloc(1*sizeof(double*));
    tmp_sigma   = (double**) malloc(1*sizeof(double*));
    tmp_fmask   = (double**) malloc(1*sizeof(double*));
    tmp_fprj    = (double**) malloc(1*sizeof(double*));
    if (z == NULL || lambda == NULL || tmp_tau == NULL || tmp_mu == NULL || tmp_sigma == NULL
        || tmp_fmask == NULL || tmp_fprj == NULL)
    {
      log_fatal("fail allocation");
      exit(1);
    }
    else 
    {
      (*z) = NULL;
      (*lambda) = NULL;
      (*tmp_tau) = NULL;
      (*tmp_mu) = NULL; 
      (*tmp_sigma) = NULL; 
      (*tmp_fmask) = NULL; 
      (*tmp_fprj) = NULL; 
    }
   
    setup_SDSS_P_lambda_obs_given_true_lambda(&nz, z, &nlambda, lambda, tmp_tau, tmp_mu, tmp_sigma, 
      tmp_fmask, tmp_fprj, 0);

    const gsl_interp2d_type *T = gsl_interp2d_bilinear;
    ftau = gsl_spline2d_alloc(T, nz, nlambda);
    fmu = gsl_spline2d_alloc(T, nz, nlambda);
    fsigma = gsl_spline2d_alloc(T, nz, nlambda);
    ffmask = gsl_spline2d_alloc(T, nz, nlambda);    
    ffprj = gsl_spline2d_alloc(T, nz, nlambda);
    if (ftau == NULL || fmu == NULL || fsigma == NULL || ffmask == NULL || ffprj == NULL)
    {
      log_fatal("fail allocation");
      exit(1);
    }

    // we don't want to guess the appropriate GSL z array ordering in z = f(x, y)
    tau     = (double*) malloc(nz*nlambda*sizeof(double));
    mu      = (double*) malloc(nz*nlambda*sizeof(double));
    sigma   = (double*) malloc(nz*nlambda*sizeof(double));
    fmask   = (double*) malloc(nz*nlambda*sizeof(double));
    fprj    = (double*) malloc(nz*nlambda*sizeof(double));
    if (tau == NULL || mu == NULL || sigma == NULL || fmask == NULL || fprj == NULL)
    {
      log_fatal("fail allocation");
      exit(1);
    }

    for (int i=0; i<nz; i++) 
    {
      for (int j=0; j<nlambda; j++) 
      {
        int status = 0;
        status = gsl_spline2d_set(ftau, tau, i, j, (*tmp_tau)[i*nz + j]);
        if (status) 
        {
          log_fatal(gsl_strerror(status));
          exit(1);
        }
        status = gsl_spline2d_set(fmu, mu, i, j, (*tmp_mu)[i*nz + j]);
        if (status) 
        {
          log_fatal(gsl_strerror(status));
          exit(1);
        }
        status = gsl_spline2d_set(fsigma, sigma, i, j, (*tmp_sigma)[i*nz + j]);
        if (status) 
        {
          log_fatal(gsl_strerror(status));
          exit(1);
        }
        status = gsl_spline2d_set(ffmask, fmask, i, j, (*tmp_fmask)[i*nz + j]);
        if (status) 
        {
          log_fatal(gsl_strerror(status));
          exit(1);
        }
        status = gsl_spline2d_set(ffprj, fprj, i, j, (*tmp_fprj)[i*nz + j]);
        if (status) 
        {
          log_fatal(gsl_strerror(status));
          exit(1);
        }
      }
    }
    int status = 0;
    status = gsl_spline2d_init(ftau, (*z), (*lambda), tau, nz, nlambda);
    if (status) 
    {
      log_fatal(gsl_strerror(status));
      exit(1);
    }
    status = gsl_spline2d_init(fmu, (*z), (*lambda), mu, nz, nlambda);
    if (status) 
    {
      log_fatal(gsl_strerror(status));
      exit(1);
    }
    status = gsl_spline2d_init(fsigma, (*z), (*lambda), sigma, nz, nlambda);
    if (status) 
    {
      log_fatal(gsl_strerror(status));
      exit(1);
    }
    status = gsl_spline2d_init(ffmask, (*z), (*lambda), fmask, nz, nlambda);
    if (status) 
    {
      log_fatal(gsl_strerror(status));
      exit(1);
    }
    status = gsl_spline2d_init(ffprj, (*z), (*lambda), fprj, nz, nlambda);
    if (status) 
    {
      log_fatal(gsl_strerror(status));
      exit(1);
    }
    free(z[0]);
    free(lambda[0]);
    free(tmp_tau[0]);
    free(tmp_mu[0]);
    free(tmp_sigma[0]);
    free(tmp_fmask[0]);
    free(tmp_fprj[0]);
    free(z);
    free(lambda);
    free(tmp_tau);
    free(tmp_mu);
    free(tmp_sigma);
    free(tmp_fmask);
    free(tmp_fprj);
    free(tau);   // GSL SPLINE 2D copies the array
    free(mu);    // GSL SPLINE 2D copies the array
    free(sigma); // GSL SPLINE 2D copies the array
    free(fmask); // GSL SPLINE 2D copies the array
    free(fprj);  // GSL SPLINE 2D copies the array
  }
  
  double tau, mu, sigma, fmask, fprj;
  {
    int status = 0;
    status = gsl_spline2d_eval_e(ftau, zz, true_lambda, NULL, NULL, &tau);
    if (status) 
    {
      log_fatal(gsl_strerror(status));
      exit(1);
    }  
    status = gsl_spline2d_eval_e(fmu, zz, true_lambda, NULL, NULL, &mu);
    if (status) 
    {
      log_fatal(gsl_strerror(status));
      exit(1);
    }  
    status = gsl_spline2d_eval_e(fsigma, zz, true_lambda, NULL, NULL, &sigma);
    if (status) 
    {
      log_fatal(gsl_strerror(status));
      exit(1);
    }
    status = gsl_spline2d_eval_e(ffmask, zz, true_lambda, NULL, NULL, &fmask);
    if (status) 
    {
      log_fatal(gsl_strerror(status));
      exit(1);
    }
    status = gsl_spline2d_eval_e(ffprj, zz, true_lambda, NULL, NULL, &fprj);
    if (status) 
    {
      log_fatal(gsl_strerror(status));
      exit(1);
    }
  }

  const double x = 1.0/(M_SQRT2*abs(sigma));
  const double y = 1.0/true_lambda;
  const double j = exp(0.5*tau*(2*mu+tau*sigma*sigma-2*observed_lambda));

  double r0 = exp(-1.*(observed_lambda-mu)*(observed_lambda-mu)*x*x);
  r0 *= (1-fmask)*(1-fprj)*x/M_SQRTPI;

  double r1 = 0.5*((1-fmask)*fprj*tau + fmask*fprj*y)*j;
  r1 *= gsl_sf_erfc((mu+tau*sigma*sigma-observed_lambda)*x);

  double r2 = 0.5*fmask*y;
  r2 *= gsl_sf_erfc((mu-observed_lambda-true_lambda)*x) - gsl_sf_erfc((mu-observed_lambda)*x);

  double r3 = 0.5*fmask*fprj*y*exp(-tau*true_lambda)*j;
  r3 *= gsl_sf_erfc((mu+tau*sigma*sigma-observed_lambda-true_lambda)*x);
  
  return r0 + r1 + r2 - r3;
}

static double SDSS_P_lambda_obs_lambda_true_given_M(double true_lambda, 
const double observed_lambda, const double M, const double z)
{
  const double r1 = SDSS_P_lambda_obs_given_true_lambda(observed_lambda, true_lambda, z);
  const double r2 = SDSS_P_true_lambda_given_mass(true_lambda, M, z);
  return r1*r2;
}

double SDSS_P_lambda_obs_lambda_true_given_M_wrapper(double true_lambda, void* params)
{
  double* ar = (double*) params;
  const double M = ar[0];
  const double z = ar[1];
  const double observed_lambda = ar[2];
  return SDSS_P_lambda_obs_lambda_true_given_M(true_lambda, observed_lambda, M, z);
}


double SDSS_P_lambda_obs_given_M(const double observed_lambda, const double M, const double z, 
const int init_static_vars_only)
{  
  double params_in[3] = {M, z, observed_lambda}; 
  
  const double true_lambda_min = limits.SDSS_P_lambda_obs_given_M_true_lambda_min;
  const double true_lambda_max = limits.SDSS_P_lambda_obs_given_M_true_lambda_max;
  
  return (init_static_vars_only == 1) ? 
    SDSS_P_lambda_obs_lambda_true_given_M_wrapper(true_lambda_min, (void*) params_in) :
    int_gsl_integrate_medium_precision(SDSS_P_lambda_obs_lambda_true_given_M_wrapper, 
      (void*) params_in, true_lambda_min, true_lambda_max, NULL, GSL_WORKSPACE_SIZE);
}

static double SDSS_P_lambda_obs_given_M_wrapper(double observed_lambda, void* params)
{
  double* ar = (double*) params; 
  const double M = ar[0];
  const double z = ar[1];
  const int init_static_vars_only = (int) ar[2];
  return SDSS_P_lambda_obs_given_M(observed_lambda, M, z, init_static_vars_only);
}

double SDSS_binned_P_lambda_obs_given_M(const int nl, const double M, const double z, 
const int init_static_vars_only)
{
  double params[3] = {M, z, (double) init_static_vars_only};
  
  const int nl_min = Cluster.N_min[nl];
  const int nl_max = Cluster.N_max[nl];
  
  return (init_static_vars_only == 1) ?
    SDSS_P_lambda_obs_given_M_wrapper((double) nl_min, (void*) params) :
    int_gsl_integrate_medium_precision(SDSS_P_lambda_obs_given_M_wrapper, (void*) params, 
      (double) nl_min, (double) nl_max, NULL, GSL_WORKSPACE_SIZE);
}

// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// INTERFACE - binned_P_lambda_obs_given_M
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------

// \int_(bin_lambda_obs_min)^(bin_lambda_obs_max) \dlambda_obs P(\lambda_obs|M)
// (see for example https://arxiv.org/pdf/1810.09456.pdf - eq 3 and eq 6) 
double binned_P_lambda_obs_given_M_nointerp(const int nl, const double M, const double z, 
const int init_static_vars_only)
{
  if (strcmp(Cluster.model, "SDSS") == 0) 
  {
    return SDSS_binned_P_lambda_obs_given_M(nl, M, z, init_static_vars_only);
  }
  else if (strcmp(Cluster.model, "BUZZARD") == 0)
  {
    return buzzard_binned_P_lambda_obs_given_M(nl, M, z, init_static_vars_only);
  }
  else
  {
    log_fatal("Cluster.model not implemented");
    exit(1);
  }
}

double p_lambda_obs_given_m(const int nl, const double M, const double z) 
{
  static cosmopara C;
  static nuisancepara N;
  static double*** table = 0;

  const int N_l = Cluster.N200_Nbin;
  
  const int N_M  = Ntable.binned_P_lambda_obs_given_M_size_M_table;
  const double log_M_min = limits.cluster_util_log_M_min;
  const double log_M_max = limits.cluster_util_log_M_max;
  const double dlogM = (log_M_max - log_M_min)/((double) N_M - 1.0);
  
  const int N_z = Ntable.binned_P_lambda_obs_given_M_size_z_table;
  const double zmin = limits.binned_P_lambda_obs_given_M_zmin_table; 
  const double zmax = limits.binned_P_lambda_obs_given_M_zmax_table;
  const double dz = (zmax - zmin)/((double) N_z - 1.0);
  
  if (table == 0)
  {
    table = (double***) malloc(sizeof(double**)*N_l);
    for(int i=0; i<N_l; i++)
    {
      table[i] = (double**) malloc(sizeof(double*)*N_z);
      for(int j=0; j<N_z; i++)
      {
        table[i][j] = (double*) malloc(sizeof(double)*N_M);
      }
    }
  }
  if (recompute_clusters(C, N))
  { 
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-variable"
    {
      const double MM = pow(10.0, limits.cluster_util_log_M_min);
      double init_static_vars_only = binned_P_lambda_obs_given_M_nointerp(0, MM, zmin, 1);
    }
    #pragma GCC diagnostic pop
    #pragma omp parallel for collapse(3)
    for (int i=0; i<N_l; i++) 
    {
      for (int j=0; j<N_z; j++) 
      {
        for (int k=0; k<N_M; k++) 
        {
          const double zz = zmin + j*dz;
          const double MM = pow(10.0, limits.cluster_util_log_M_min + k*dlogM);
          table[i][j][k] = binned_P_lambda_obs_given_M_nointerp(i, MM, zz, 0);
        }
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
  if (z < zmin || z > zmax)
  {
    log_fatal("z = %e outside look-up table range [%e,%e]", z, zmin, zmax);
    exit(1);
  } 
  const double logM = log10(M);
  if (logM < limits.cluster_util_log_M_min || logM > limits.cluster_util_log_M_max)
  {
    log_fatal("logM = %e outside look-up table range [%e,%e]", logM, 
      limits.cluster_util_log_M_min, limits.cluster_util_log_M_max);
    exit(1);
  } 
  return interpol2d(table[nl], N_z, zmin, zmax, dz, z, N_M, log_M_min, log_M_max, dlogM,
    logM, 1.0, 1.0);
}

// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// BINNED CLUSTER MASS FUNCTION 
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------

double dndlnM_times_binned_P_lambda_obs_given_M(double lnM, void* params)
{
  double* ar = (double*) params; 
  const int nl = (int) ar[0];
  if (nl < 0 || nl > Cluster.N200_Nbin - 1)
  {
    log_fatal("invalid bin input nl = %d", nl);
    exit(1);
  }
  const double z = ar[1];
  if (!(z>0))
  {
    log_fatal("invalid redshift input z = %d", z);
    exit(1);
  } 
  const double a = 1.0/(1.0 + z);
  const double M = exp(lnM);

  double mfunc; 
  if (Cluster.hmf_model == 0)
  {
    mfunc = massfunc(M, a);
  }
  else if (Cluster.hmf_model == 1)
  {
    mfunc = tinker_emulator_massfunc(M, a);
  }
  else
  {
    log_fatal("massfunc model %i not implemented", Cluster.hmf_model);
    exit(1); 
  }
  return mfunc*M*binned_P_lambda_obs_given_M(nl, M, z);
}

// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// CLUSTER BIAS
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------

double BSF(const double M)
{
  double BS;
  if (Cluster.N_SF == 1) 
  {
    BS = nuisance.cluster_selection[0]; 
  }
  else if (Cluster.N_SF == 2)
  {
    BS = 
    nuisance.cluster_selection[0]*pow((M/M_PIVOT), nuisance.cluster_selection[1]);
  }
  else
  {
    log_fatal("Cluster selection bias model %i not implemented", Cluster.bias_model);
    exit(0); 
  }
  return BS;
}

double B1_x_BSF(const double M, const double a)
{ // cluster bias including selection bias
  double tmp_B1;
  if (Cluster.bias_model == 0) 
  {
    tmp_B1 = B1(M, a);
  } 
  else if (Cluster.bias_model == 1) 
  {
    tmp_B1 = tinker_emulator_B1(M, a);
  }
  else 
  {
    log_fatal("Cluster bias model %i not implemented", Cluster.bias_model);
    exit(0); 
  }
  return tmp_B1 * BSF(M);
}

double B2_x_BSF(const double M, const double a)
{ // cluster bias (b2) including selection bias
  double tmp_B1;
  if (Cluster.bias_model == 0) 
  {
    tmp_B1 = B1(M, a);
  } 
  else if (Cluster.bias_model == 1) 
  {
    tmp_B1 = tinker_emulator_B1(M, a);
  }
  else 
  {
    log_fatal("Cluster bias model %i not implemented", Cluster.bias_model);
    exit(0); 
  }
  return b2_from_b1(tmp_B1) * BSF(M);
}

double B1M1_x_BSF(const double M, const double a)
{
  double B1M1;
  if (Cluster.bias_model == 0) 
  {
    B1M1 = B1(M, a) - 1;
  } 
  else if (Cluster.bias_model == 1) 
  {
    B1M1 = tinker_emulator_B1(M, a) - 1;
  }
  else 
  {
    log_fatal("Cluster bias model %i not implemented", Cluster.bias_model);
    exit(0); 
  }
  return B1M1 * BSF(M);
}

double int_for_weighted_B1(double lnM, void* params)
{
  double* ar = (double*) params; // {nl, z}
  const double z = ar[1];
  if (!(z>0))
  {
    log_fatal("invalid redshift input z = %d", z);
    exit(1);
  } 
  const double M = exp(lnM);  
  const double a = 1.0/(1.0 + z);
  return B1_x_BSF(M, a)*dndlnM_times_binned_P_lambda_obs_given_M(lnM, params); 
}

double weighted_B1_nointerp(const int nl, const double z, const int init_static_vars_only)
{  
  const double ln_M_min = limits.cluster_util_log_M_min/M_LOG10E;
  const double ln_M_max = limits.cluster_util_log_M_max/M_LOG10E;
 
  double param[2] = {(double) nl, z};

  if (init_static_vars_only == 1)
  {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-variable"
    {
      const double r1 = int_for_weighted_B1(ln_M_min, (void*) param);
      const double r2 = dndlnM_times_binned_P_lambda_obs_given_M(ln_M_min, (void*) param);
      return 0.0;
    }
    #pragma GCC diagnostic pop
  }
  else
  {
    const double r1 = int_gsl_integrate_low_precision(int_for_weighted_B1, 
      (void*) param, ln_M_min, ln_M_max, NULL, GSL_WORKSPACE_SIZE);
  
    const double r2 = int_gsl_integrate_low_precision(dndlnM_times_binned_P_lambda_obs_given_M, 
      (void*) param, ln_M_min, ln_M_max, NULL, GSL_WORKSPACE_SIZE);

    return (r2 <= 0) ? 0.0 : r1/r2;
  }
}

double weighted_B1(const int nl, const double z)
{
  static cosmopara C;
  static nuisancepara N;
  static double** table;

  const int N_l = Cluster.N200_Nbin;
  const int N_a = Ntable.N_a;
  const double zmin = fmax(tomo.cluster_zmin[0] - 0.05, 0.01); 
  const double zmax = tomo.cluster_zmax[tomo.cluster_Nbin - 1] + 0.05;
  const double amin = 1.0/(1.0 + zmax); 
  const double amax = 1.0/(1.0 + zmin);
  const double da = (amax - amin)/((double) N_a - 1.);

  if (table == 0)
  {
    table = (double**) malloc(sizeof(double*)*N_l);
    for(int i=0; i<N_l; i++)
    {
      table[i] = (double*) malloc(sizeof(double)*N_a);
    }
  }
  if (recompute_clusters(C, N))
  {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-variable"    
    {
      double init_static_vars_only = weighted_B1_nointerp(0, 1.0/amin - 1.0, 1); 
    }
    #pragma GCC diagnostic pop
    #pragma omp parallel for collapse(2)
    for (int n=0; n<N_l; n++)
    {
      for (int i=0; i<N_a; i++)
      {
        const double a = amin + i*da;
        table[n][i] = weighted_B1_nointerp(n, 1.0/a - 1.0, 0);
      }
    }
    update_cosmopara(&C);
    update_nuisance(&N);
  }
  const double a = 1.0/(z + 1.0);
  if (a < amin || a > amax)
  {
    log_fatal("a = %e outside look-up table range [%e,%e]", a, amin, amax);
    exit(1);
  } 
  if (nl < 0 || nl > N_l - 1)
  {
    log_fatal("invalid bin input nl = %d", nl);
    exit(1);
  }
  return interpol(table[nl], N_a, amin, amax, da, a, 0., 0.);
}

double int_for_weighted_B2(double lnM, void* params)
{
  double* ar = (double*) params; //nl, z
  const double z = ar[1];
  if (!(z>0))
  {
    log_fatal("invalid redshift input z = %d", z);
    exit(1);
  }
  const double M = exp(lnM);  
  const double a = 1./(1. + z);
  return B2_x_BSF(M, a)*dndlnM_times_binned_P_lambda_obs_given_M(lnM, params); 
}

double weighted_B2_nointerp(const int nl, const double z, const int init_static_vars_only)
{  
  const double ln_M_min = limits.cluster_util_log_M_min/M_LOG10E;
  const double ln_M_max = limits.cluster_util_log_M_max/M_LOG10E;
 
  double param[2] = {(double) nl, z};

  if (init_static_vars_only == 1)
  {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-variable"
    {
      const double r1 = int_for_weighted_B2(ln_M_min, (void*) param);
      const double r2 = dndlnM_times_binned_P_lambda_obs_given_M(ln_M_min, (void*) param);
      return 0.0;
    }
    #pragma GCC diagnostic pop
  }
  else
  {
    const double r1 = int_gsl_integrate_low_precision(int_for_weighted_B2, 
      (void*) param, ln_M_min, ln_M_max, NULL, GSL_WORKSPACE_SIZE);

    const double r2 = int_gsl_integrate_low_precision(dndlnM_times_binned_P_lambda_obs_given_M, 
      (void*) param, ln_M_min, ln_M_max, NULL, GSL_WORKSPACE_SIZE);

    return (r2 <= 0) ? 0.0 : r1/r2; 
  }
}

double weighted_B2(const int nl, const double z)
{
  static cosmopara C;
  static nuisancepara N;
  static double** table;

  const int N_l = Cluster.N200_Nbin;
  const int N_a = Ntable.N_a;
  const double zmin = fmax(tomo.cluster_zmin[0] - 0.05, 0.01); 
  const double zmax = tomo.cluster_zmax[tomo.cluster_Nbin - 1] + 0.05;
  const double amin = 1.0/(1.0 + zmax); 
  const double amax = 1.0/(1.0 + zmin);
  const double da = (amax-amin)/((double) N_a - 1.0);

  if (table == 0) 
  {
    table = (double**) malloc(sizeof(double*)*N_l);
    for(int n=0; n<N_l; n++)
    {
      table[n] = (double*) malloc(sizeof(double)*N_a);
    }
  }
  if (recompute_clusters(C, N))
  {  
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-variable"    
    {
      double init_static_vars_only =  weighted_B2_nointerp(0, 1.0/amin - 1.0, 1);
    }
    #pragma GCC diagnostic pop
    #pragma omp parallel for collapse(2)
    for (int n=0; n<N_l; n++)
    {
      for (int i=0; i<N_a; i++)
      {
        const double a = amin + i*da;
        table[n][i] = weighted_B2_nointerp(n, 1.0/a - 1.0, 0);
      }
    }
    update_cosmopara(&C);
    update_nuisance(&N);
  }

  const double a = 1.0/(z + 1.0);
  if (a < amin || a > amax)
  {
    log_fatal("a = %e outside look-up table range [%e,%e]", a, amin, amax);
    exit(1);
  } 
  if (nl < 0 || nl > N_l - 1)
  {
    log_fatal("invalid bin input nl1 = %d", nl);
    exit(1);
  }
  return interpol(table[nl], N_a, amin, amax, da, a, 0., 0.);
}

double int_for_weighted_B1M1(double lnM, void* params)
{
  double* ar = (double*) params;
  const double z = ar[1];
  if (!(z>0))
  {
    log_fatal("invalid redshift input z = %d", z);
    exit(1);
  }
  const double M = exp(lnM);  
  const double a = 1.0/(1.0 + z);
  return B1M1_x_BSF(M, a)*dndlnM_times_binned_P_lambda_obs_given_M(lnM, params);
}

double weighted_B1M1_nointerp(const int nl, const double z, const int init_static_vars_only)
{  
  const double ln_M_min = limits.cluster_util_log_M_min/M_LOG10E;
  const double ln_M_max = limits.cluster_util_log_M_max/M_LOG10E;
  
  double params[2] = {(double) nl, z};

  if (init_static_vars_only == 1)
  {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-variable"
    {
      const double r1 = int_for_weighted_B1M1(ln_M_min, (void*) params);
      const double r2 = dndlnM_times_binned_P_lambda_obs_given_M(ln_M_min, (void*) params);
      return 0.0;
    }
    #pragma GCC diagnostic pop
  }
  else
  {
    const double r1 = int_gsl_integrate_low_precision(int_for_weighted_B1M1, 
      (void*) params, ln_M_min, ln_M_max, NULL, GSL_WORKSPACE_SIZE);

    const double r2 = int_gsl_integrate_low_precision(dndlnM_times_binned_P_lambda_obs_given_M, 
      (void*) params, ln_M_min, ln_M_max, NULL, GSL_WORKSPACE_SIZE);
    
    return (r2 <= 0) ? 0.0 : r1/r2; 
  }
}

double weighted_B1M1(const int nl, const double z)
{
  static cosmopara C;
  static nuisancepara N;
  static double** table;

  const int N_l = Cluster.N200_Nbin; 
  const int N_a = Ntable.N_a;
  const double zmin = fmax(tomo.cluster_zmin[0] - 0.05, 0.01); 
  const double zmax = tomo.cluster_zmax[tomo.cluster_Nbin - 1] + 0.05;
  const double amin = 1./(1.0 + zmax); 
  const double amax = 1./(1.0 + zmin);
  const double da = (amax - amin)/((double) N_a - 1.0);

  if (table == 0) 
  {
    table = (double**) malloc(sizeof(double*)*N_l);
    for(int i=0; i<N_l; i++)
    {
      table[i] = (double*) malloc(sizeof(double)*N_a);
    }
  }
  if (recompute_clusters(C, N))
  { 
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-variable"    
    {
      double init_static_vars_only =  weighted_B1M1_nointerp(0, 1.0/amin - 1.0, 1);
    }
    #pragma GCC diagnostic pop   
    #pragma omp parallel for collapse(2)
    for (int n=0; n<N_l; n++)
    {
      for (int i=0; i<N_a; i++)
      {
        const double a = amin + i*da;
        table[n][i] = weighted_B1M1_nointerp(n, 1.0/a - 1.0, 0);
      }
    }
    update_cosmopara(&C);
    update_nuisance(&N);
  }
  const double a = 1.0/(z + 1.0);
  if (a < amin || a > amax)
  {
    log_fatal("a = %e outside look-up table range [%e,%e]", a, amin, amax);
    exit(1);
  } 
  if (nl < 0 || nl > N_l - 1)
  {
    log_fatal("invalid bin input nl = %d", nl);
    exit(1);
  }
  return interpol(table[nl], N_a, amin, amax, da, a, 0., 0.);
}

// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// cluster number counts
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// nl = lambda_obs bin, ni = cluster redshift bin

double binned_Ndensity_nointerp(const int nl, const double z, const int init_static_vars_only)
{
  const double ln_M_min = limits.cluster_util_log_M_min/M_LOG10E;
  const double ln_M_max = limits.cluster_util_log_M_max/M_LOG10E;
  double params[2] = {(double) nl, z};
  return (init_static_vars_only == 1) ?
    dndlnM_times_binned_P_lambda_obs_given_M(ln_M_min, (void*) params) :
    int_gsl_integrate_low_precision(dndlnM_times_binned_P_lambda_obs_given_M, (void*) params,
      ln_M_min, ln_M_max, NULL, GSL_WORKSPACE_SIZE);
}

double binned_Ndensity(const int nl, const double z)
{
  static cosmopara C;
  static nuisancepara N;
  static double** table;

  const int N_l = Cluster.N200_Nbin;
  const int N_a = Ntable.N_a;
  const double zmin = fmax(tomo.cluster_zmin[0] - 0.05, 0.01);
  const double zmax = tomo.cluster_zmax[tomo.cluster_Nbin - 1] + 0.05;
  const double amin = 1.0/(1.0 + zmax);
  const double amax = 1.0/(1.0 + zmin);
  const double da = (amax - amin)/((double) N_a - 1.0);

  if (table == 0)
  {
    table = (double**) malloc(sizeof(double*)*N_l);
    for (int i=0; i<N_l; i++)
    {
      table[i] = (double*) malloc(sizeof(double)*N_a);
    }
  }
  if (recompute_clusters(C, N))
  {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-variable"
    {
      double init_static_vars_only = binned_Ndensity_nointerp(0, 1.0/amin - 1.0, 1);
    }
    #pragma GCC diagnostic pop
    #pragma omp parallel for collapse(2)
    for (int i=0; i<N_l; i++)
    {
      for (int j=0; j<N_a; j++)
      {
        const double aa = amin + j*da;
        table[i][j] = binned_Ndensity_nointerp(i, 1.0/aa - 1.0, 0);
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
  if (z < zmin || z > zmax)
  {
    log_fatal("z = %e outside look-up table range [%e,%e]", z, zmin, zmax);
    exit(1);
  }
  return interpol(table[nl], N_a, amin, amax, da, 1.0/(z + 1.0), 0., 0.);
}

// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// Area
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------

// Cocoa: we try to avoid reading of files in the cosmolike_core code 
// Cocoa: (reading is done in the C++/python interface)
void setup_get_area(int* io_nz, double** io_z, double** io_A, int io) 
{
  static int nz;
  static double* z = NULL;
  static double* A = NULL;

  if (io == 1)
  { // IO == 1 IMPLES THAT IO_A(Z) WILL COPIED TO LOCAL A(Z)
    if (z != NULL)
    {
      free(z);
    }
    if (A != NULL)
    {
      free(A);
    }
    nz = (*io_nz);
    if (!(nz > 5))
    {
      log_fatal("array to small");
      exit(1);
    }
    z = (double*) malloc(nz*sizeof(double));
    A = (double*) malloc(nz*sizeof(double));
    if (z == NULL || A == NULL)
    {
      log_fatal("fail allocation");
      exit(1);
    }
    for (int i=0; i<nz; i++)
    {
      z[i] = (*io_z)[i];
      A[i] = (*io_A)[i];
    }
  }
  else
  { // IO != 1 IMPLES THAT LOCAL A(Z) WILL BE COPIED TO IO_A(Z)
    if (z == NULL || A == NULL)
    {
      log_fatal("array/pointer not allocated");
      exit(1);
    }
    if ((*io_z) != NULL)
    {
      free((*io_z));
      (*io_z) = NULL;
    }
    else if ((*io_A) != NULL)
    {
      free((*io_A));
      (*io_A) = NULL;
    }
    (*io_z) = (double*) malloc(nz*sizeof(double));
    (*io_A) = (double*) malloc(nz*sizeof(double));
    if ((*io_z) == NULL || (*io_A) == NULL)
    {
      log_fatal("fail allocation");
      exit(1);
    }
    for (int i=0; i<nz; i++)
    {
      (*io_z)[i] = z[i];
      (*io_A)[i] = A[i];
    }
  }
}

double get_area(const double zz, const int interpolate_survey_area)
{
  static gsl_spline* fA = NULL;

  if (interpolate_survey_area == 1)
  {
    if (fA == NULL)
    {
      int nz;
      double** z = (double**) malloc(1*sizeof(double*));
      double** A = (double**) malloc(1*sizeof(double*));
         
      if (z == NULL || A == NULL)
      {
        log_fatal("fail allocation");
        exit(1);
      }
      else 
      {
        (*z) = NULL;
        (*A) = NULL;
      }

      setup_get_area(&nz, z, A, 0);

      const gsl_interp_type* T = gsl_interp_linear;
      fA = gsl_spline_alloc(T, nz);
      if (fA == NULL)
      {
        log_fatal("fail allocation");
        exit(1);
      }

      int status = 0;
      status = gsl_spline_init(fA, (*z), (*A), nz);
      if (status) 
      {
        log_fatal(gsl_strerror(status));
        exit(1);
      }
      free(z[0]); // spline makes a copy of the data
      free(A[0]); // spline makes a copy of the data
      free(z);    // spline makes a copy of the data
      free(A);    // spline makes a copy of the data
    }
    double res;
    int status = gsl_spline_eval_e(fA, zz, NULL, &res);
    if (status) 
    {
      log_fatal(gsl_strerror(status));
      exit(1);
    }
    return res;
  }
  else
  {
    return survey.area;
  }
}