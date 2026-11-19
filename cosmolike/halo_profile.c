#include <gsl/gsl_sf.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>

#include "halo_profile.h"
#include "basics.h"

// ---------------------------------------------------------------------------
// Builders of the shared NFW profile table (halo_profile.h): the f, G
// rows, the Hermite slope rows of the IA reader, and the release of a
// descriptor. The readers live in the header, inlined into their
// callers; only the builders have a home here. Ownership, cache
// invalidation and thread safety: the file banner of halo_profile.h.
// ---------------------------------------------------------------------------


void nfw_profile_prepare(
    nfw_profile* prof,   // the descriptor to (re)build
    const int n_nodes,   // ln t nodes of the table
    const uint64_t key   // the caller's cache tag, stored in prof
  )
{
  // f(t_i), G(t_i) from GSL Si, Ci (formulas in the table banner); the
  // caller owns the rebuild condition. The slope rows derive from the
  // table, so they go with it; nfw_profile_slopes rebuilds them on
  // demand.
  if (prof->tab != NULL) {
    free(prof->tab);
  }
  if (prof->fg_slope != NULL) {
    free(prof->fg_slope);
    prof->fg_slope = NULL;
  }

  // --- 1. AXIS SETUP ---
  double** tab = (double**) malloc2d(2, n_nodes);
  double* lim  = prof->lim;
  lim[0] = log(NFW_TMIN);
  lim[1] = log(NFW_TASY);
  lim[2] = (lim[1] - lim[0])/((double) n_nodes - 1.0);

  // --- 2. TABLE FILL ---
  /* PHYSICAL DERIVATION & LOGIC FLOW
     1. node: ln t_i = lim[0] + i lim[2],   t_i = exp(ln t_i)
     2. f(t) =  Ci(t) sin t + (pi/2 - Si(t)) cos t
     3. G(t) = -Ci(t) cos t + (pi/2 - Si(t)) sin t + ln t = g(t) + ln t */
  #pragma omp parallel for schedule(static)
  for (int i=0; i<n_nodes; i++) {
    const double lnt = lim[0] + i*lim[2];
    const double t   = exp(lnt);
    const double si  = gsl_sf_Si(t);
    const double ci  = gsl_sf_Ci(t);
    tab[0][i] = ci*sin(t) + (M_PI_2 - si)*cos(t);        // f(t)
    tab[1][i] = -ci*cos(t) + (M_PI_2 - si)*sin(t) + lnt; // G(t)
  }

  // --- 3. PUBLISH ---
  prof->n_nodes     = n_nodes;
  prof->inv_spacing = 1.0/lim[2];
  prof->tab         = tab;
  prof->cache       = key;
}


void nfw_profile_slopes(
    nfw_profile* prof    // the descriptor to add slope rows to
  )
{
  // Hermite slopes of the f, G table (nfw_fg_read's header): at node
  // ln t_i, h df/dln t = -h t g(t) and h dG/dln t = h t f(t), with
  // g = G - ln t and h the node spacing. nfw_profile_prepare frees the
  // rows with the table, so NULL here means "not built for this table".
  if (prof->fg_slope != NULL) {
    return;
  }

  double** fg_slope = (double**) malloc2d(2, prof->n_nodes);
  for (int i=0; i<prof->n_nodes; i++) {
    const double lnt = prof->lim[0] + i*prof->lim[2];
    const double t   = exp(lnt);
    const double f_i = prof->tab[0][i];
    const double g_i = prof->tab[1][i] - lnt;
    fg_slope[0][i] = -t*g_i*prof->lim[2];
    fg_slope[1][i] = t*f_i*prof->lim[2];
  }
  prof->fg_slope = fg_slope;
}


void nfw_profile_reset(
    nfw_profile* prof    // the descriptor to release
  )
{
  if (prof->tab != NULL) {
    free(prof->tab);
  }
  if (prof->fg_slope != NULL) {
    free(prof->fg_slope);
  }
  *prof = (nfw_profile) {0};
}
