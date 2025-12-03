#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log.c/src/log.h"

#include "basics.h"
#include "cosmo3D.h"
#include "redshift_spline_cluster.h"
#include "radial_weights_cluster.h"
#include "structs.h"

double W_cluster(const int nl, const int nz, const double a, const double hoverh0)
{
  if(!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  if (nl < -1 || nl > Cluster.n200_nbin - 1)  {
    log_fatal("invalid bin input nl = %d", nl); exit(1);
  }
  return zdistr_cluster(nl, nz, 1. / a - 1.)*hoverh0;
}

double W_mag_cluster(const int nz, const double a, const double fK)
{
  if (!(a>0) || !(a<1))  {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  if (nz < -1 || nz > redshift.cluster_nbin - 1)  {
    log_fatal("invalid bin input ni = %d", nz); exit(1);
  }
  return (1.5 * cosmology.Omega_m * fK / a)  * g_lens_cluster(a, nz, -1);
}
