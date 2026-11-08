#include <assert.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "basics.h"
#include "bias.h"
#include "cosmo3D.h"
#include "redshift_spline.h"
#include "structs.h"

#include "log.c/src/log.h"

double gb1(const double z, const int ni)
{ // g = galaxy, b1 = linear galaxy bias
  if (ni < -1 || ni > redshift.clustering_nbin - 1)
  {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }

  double ans;

  switch(like.galaxy_bias_model[0])
  {
    case B1_PER_BIN:
    {
      ans = nuisance.gb[0][ni];
      break;
    }
    case B1_PER_BIN_EVOLV:
    {
      ans = nuisance.gb[0][ni] * pow((1.0 + z) / (1. + zmean(ni)), 1.0);
      break;
    }
    case B1_PER_BIN_PASS_EVOLV:
    {
      const double z_evolv_passiv = growfac(1. / (z + 1.)) /
                growfac(1. / (1. + 0.5 * (redshift.clustering_zdist_z[RANGE_MIN][ni] +
                                          redshift.clustering_zdist_z[RANGE_MAX][ni])));
      
      ans = (nuisance.gb[0][ni] - 1.) / z_evolv_passiv + 1.;
      break;
    }
    case B1_GROWTH_SCALING:
    {
      ans = nuisance.gb[0][0] / growfac(1. / (z + 1.));
      break;
    }
    case B1_POWER_LAW:
    {
      ans = nuisance.gb[0][0] * pow(1 + z, nuisance.gb[0][1]);
      break;
    }
    default:
    {
      log_fatal("like.galaxy_bias_model[0] = %d not supported", 
        like.galaxy_bias_model[0]);
      exit(1);  
    }
  }
  return ans;
}

double gb2(const double z, const int ni)
{ // g = galaxy, b2 = non-linear galaxy bias
  if (ni < -1 || ni > redshift.clustering_nbin - 1)
  {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }

  double ans;

  switch(like.galaxy_bias_model[1])
  {
    case B2_PER_BIN:
    {
      ans = nuisance.gb[1][ni];
      break;
    }
    case B2_FROM_B1:
    {
      // fitting formula from Lazeyras et al. 2016 (Eq. 5.2) - 1511.01096
      const double b1 = gb1(z, ni);
      ans = 0.412 - 2.143 * b1 + 0.929 * b1 * b1 + 0.008 * b1 * b1 * b1;
      break;
    }
    default:
    {
      log_fatal("like.galaxy_bias_model[1] = %d not supported", 
        like.galaxy_bias_model[1]);
      exit(1);  
    }
  }
  return ans;
}

double gbs2(const double z, const int ni)
{
  if (ni < -1 || ni > redshift.clustering_nbin - 1)
  {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }

  double ans;

  switch(like.galaxy_bias_model[2])
  {
    case BS2_PER_BIN:
    {
      ans = nuisance.gb[2][ni];
      break;
    }
    case BS2_FROM_B1:
    {
      ans = -4. / 7. * (gb1(z, ni) - 1.0); // 1405.1447v4.pdf
      break;
    }
    default:
    {
      log_fatal("like.galaxy_bias_model[2] = %d not supported", 
        like.galaxy_bias_model[2]);
      exit(1);  
    }
  }
  return ans;
}

double gb3(const double z, const int ni)
{
  if (ni < -1 || ni > redshift.clustering_nbin - 1)
  {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }

  double ans;

  switch(like.galaxy_bias_model[3])
  {
    case B3_PER_BIN:
    {
      ans = nuisance.gb[3][ni];
      break;
    }
    case B3_FROM_B1:
    {
      ans = (gb1(z, ni) - 1.0);
      break;
    }
    default:
    {
      log_fatal("like.galaxy_bias_model[3] = %d not supported", 
        like.galaxy_bias_model[3]);
      exit(1);  
    }
  }
  return ans;
}

double gbmag(const double z, const int ni)
{
  if (ni < -1 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  double ans;
  switch(like.galaxy_bias_model[4])
  {
    case BMAG_PER_BIN:
    {
      ans = nuisance.gb[4][ni];
      break;
    }
    default:
    {
      log_fatal("like.galaxy_bias_model[4] = %d not supported", 
        like.galaxy_bias_model[4]);
      exit(1);  
    }
  }
  return ans;
}

double gbK(const double z, const int ni)
{
  if (ni < -1 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  double ans;
  switch(like.galaxy_bias_model[5])
  {
    case BK_PER_BIN:
    {
      ans = nuisance.gb[5][ni];
      break;
    }
    default:
    {
      log_fatal("like.galaxy_bias_model[5] = %d not supported", 
        like.galaxy_bias_model[5]);
      exit(1);
    }
  }
  return ans;
}

// ---------------------------------------------------------------------------
// The redshift-space counterterm alpha_2 of clustering bin ni, in the
// convention of the DESI DR1 full-shape analysis (Maus et al. 2024,
// arXiv:2404.07312).
//
// DESI adds a counterterm field to the linear galaxy density in
// redshift space. Its alpha_2 part is
//
//   delta_CT = (1/2) f mu^2 alpha_2 k^2 delta_L,
//
// with mu the cosine between the wavevector and the line of sight, f
// the linear growth rate and delta_L the linear matter density. Twice
// its cross with the linear field (b1 + f mu^2) delta_L adds the power
//
//   P_CT = (b1 + f mu^2) f alpha_2 mu^2 k^2 P_lin(k, z).
//
// This is the convention in which DESI's prior width on alpha_2 is
// quoted. cosmo2D.c projects the term (has_alpha2_galaxies there).
//
// alpha_2 is one number per bin, with no redshift dependence inside the
// bin. It is stored in units of (c/H0)^2, the unit of 1/k^2 in this code;
// set_nuisance_bias_alpha2 (generic_interface.cpp) converts it from
// h^-2 Mpc^2.
// ---------------------------------------------------------------------------
double galpha2(const int ni)
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }
  return nuisance.gb[6][ni];
}
