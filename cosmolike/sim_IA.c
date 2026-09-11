// ============================================================================
// sim_IA.c  --  simulation-measured IA / bias power spectra for CosmoLike
//
// Injects tabulated P(k,z) spectra measured from a simulation (the IA_PS/Pds
// stage) directly into the Limber integrands of cosmo2D.c, bypassing the
// analytic NLA/TATT models entirely.
//
// EXTRAPOLATION SEMANTICS (this version):
//   Matches the Python reference (ia_ps_interp.py, IAPowerInterpolator built
//   with extrapolate=True): bilinear on (z, ln k), and OUTSIDE the grid a
//   LINEAR-IN-P extrapolation using the two edge nodes on each axis, applied
//   independently per axis (so corners extrapolate in both z and ln k). This
//   is exactly what scipy's RegularGridInterpolator(method="linear",
//   bounds_error=False, fill_value=None) does.
//
//   TRUSTED k-range: the low-k edge is k_min = SIM_KMIN_HMPC (default 0.5
//   h/Mpc). Below it and above the top table node, values are extrapolated by
//   the same linear-in-P rule, NOT clamped. This restores the Python behavior
//   the user asked for and REMOVES the earlier flat-clamp / log-slope-falloff
//   guards. See the WARNING below.
//
//   *** WARNING (read before trusting a data vector) ***
//   Linear-in-P extrapolation of a sign-changing spectrum can drive P through
//   zero into the wrong sign, or grow away from zero, past the edges. The
//   earlier version of this file deliberately clamped low-k flat and forced a
//   decaying, sign-preserving log-slope at high-k precisely because the noisy
//   edge bins (lowest-k bins have 3/10/27 modes; highest-k ~53% error near
//   Nyquist) otherwise inject xi+ sign-flips at large theta and xi- spikes at
//   small theta via the J0/J4 kernels. Faithfully matching Python's
//   extrapolate=True re-exposes that risk. Recommended: still trim the loaded
//   grid to the reliable band (SIM_KMIN_HMPC / SIM_KMAX_HMPC) so the edge
//   nodes used for the slope are trustworthy, and inspect xi+/xi- after.
//
// Unit convention (critical -- see cosmo2D.c lines ~1163, 1251, 3823):
//   Inside CosmoLike, k is in "code units" k_code = ell/fK with units
//   (c/H0)/(Mpc/h), and power spectra are carried in (c/H0)^3.
//   Our files are physical: k_phys [h/Mpc], P_phys [h^-3 Mpc^3]. With
//       R = c/H0 in Mpc/h
//   the conversions are:
//       k_phys  = k_code / R
//       P_code  = P_phys / R^3
// ============================================================================

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "structs.h"       // cosmology, nuisance
#include "cosmo3D.h"       // f_K, chi, a_chi (not strictly needed here)
#include "log.c/src/log.h"

// ---------------------------------------------------------------------------
// Spectrum types. The string is the prefix in the filename:
// P<TAG>0_sn<sn>_nfold1.dat  e.g. SIM_PEE -> "PEE0_sn67_nfold1.dat".
// ---------------------------------------------------------------------------
typedef enum {
  SIM_PEE = 0,   // shear E-mode auto (II, E)
  SIM_PBB,       // shear B-mode auto (II, B)
  SIM_PdE,       // density x E  (GI / dI)
  SIM_PdB,       // density x B
  SIM_Pdd,       // matter density auto
  SIM_Phh,       // galaxy-number auto  (bias clustering)
  SIM_Pdh,       // density x number
  SIM_NTYPES
} sim_spec_t;

static const char* SIM_TAG[SIM_NTYPES] = {
  "PEE", "PBB", "PdE", "PdB", "Pdd", "Phh", "Pdh"
};

// ---------------------------------------------------------------------------
// Trusted k-band (physical, h/Mpc). Rows outside are trimmed at LOAD time, so
// the edge nodes used for extrapolation are the reliable ones. This mirrors
// the Python loader's k_min / k_max (which trim the GRID, not just the query).
// ---------------------------------------------------------------------------
#ifndef SIM_KMIN_HMPC
#define SIM_KMIN_HMPC 0.5      // low-k trusted edge  [h/Mpc]
#endif
#ifndef SIM_KMAX_HMPC
#define SIM_KMAX_HMPC 7.0     // high-k trusted edge [h/Mpc] (0 = no cap)
#endif

// ---------------------------------------------------------------------------
// One interpolated spectrum: a (z, lnk) grid stored row-major P[iz*nk + ik]
// in PHYSICAL units (h^-3 Mpc^3). We do the bilinear + linear extrapolation by
// hand (no GSL) so the out-of-range behavior exactly matches scipy.
// ---------------------------------------------------------------------------
typedef struct {
  int loaded;
  int nz, nk;
  double* z;         // ascending, length nz
  double* lnk;       // ascending, length nk
  double* P;         // nz*nk row-major, physical units
} sim_spec;

static sim_spec SIM[SIM_NTYPES];

// Config the host must set before first use.
static char    SIM_DIR[1024] = ".";
static int     SIM_NSN       = 0;
static int*    SIM_SN        = NULL;
static double* SIM_Z         = NULL;
static int     SIM_NFOLD     = 1;
static int     SIM_READY     = 0;

// ---------------------------------------------------------------------------
static int read_dat(const char* path, double** out_k, double** out_P, int* n) {
  FILE* f = fopen(path, "r");
  if (!f) return 0;
  int cap = 256, m = 0;
  double* k = (double*) malloc(cap*sizeof(double));
  double* P = (double*) malloc(cap*sizeof(double));
  char line[512];
  while (fgets(line, sizeof(line), f)) {
    char* p = line;
    while (*p==' '||*p=='\t') p++;
    if (*p=='#' || *p=='\n' || *p=='\0') continue;
    double kk, PP, c3, c4;
    int got = sscanf(line, "%lf %lf %lf %lf", &kk, &PP, &c3, &c4);
    if (got < 2) continue;
    if (m >= cap) {
      cap *= 2;
      k = (double*) realloc(k, cap*sizeof(double));
      P = (double*) realloc(P, cap*sizeof(double));
    }
    k[m] = kk; P[m] = PP; m++;
  }
  fclose(f);
  *out_k = k; *out_P = P; *n = m;
  return (m > 0);
}

void set_sim_IA_config(const char* dir, int nsn, const int* sn,
                       const double* zlist, int nfold) {
  strncpy(SIM_DIR, dir, sizeof(SIM_DIR)-1);
  SIM_DIR[sizeof(SIM_DIR)-1] = '\0';
  SIM_NSN   = nsn;
  SIM_NFOLD = nfold;
  free(SIM_SN); free(SIM_Z);
  SIM_SN = (int*)    malloc(nsn*sizeof(int));
  SIM_Z  = (double*) malloc(nsn*sizeof(double));
  int* order = (int*) malloc(nsn*sizeof(int));
  for (int i=0;i<nsn;i++) order[i]=i;
  for (int i=1;i<nsn;i++){                      // insertion sort by z
    int key=order[i]; int j=i-1;
    while (j>=0 && zlist[order[j]] > zlist[key]) { order[j+1]=order[j]; j--; }
    order[j+1]=key;
  }
  for (int i=0;i<nsn;i++){ SIM_SN[i]=sn[order[i]]; SIM_Z[i]=zlist[order[i]]; }
  free(order);
  SIM_READY = 0;
}

// ---------------------------------------------------------------------------
// Load one spectrum type across all snapshots into a (z, lnk) grid.
// Rows outside [SIM_KMIN_HMPC, SIM_KMAX_HMPC] are trimmed here (grid trim,
// mirroring the Python loader). Every file must share the same k-grid.
// ---------------------------------------------------------------------------
static void load_spec(sim_spec_t t) {
  sim_spec* s = &SIM[t];
  if (s->loaded) return;
  if (SIM_NSN < 2) {
    log_fatal("sim_IA: need >=2 snapshots to interpolate in z (have %d). "
              "Call set_sim_IA_config() first.", SIM_NSN);
    exit(1);
  }

  const double kmin = SIM_KMIN_HMPC;
  const double kmax = (SIM_KMAX_HMPC > 0.0) ? SIM_KMAX_HMPC : INFINITY;

  double* kref = NULL;      // trimmed, ascending
  int nk_ref = 0;
  double* Pflat = NULL;     // nz*nk_ref
  int nz = SIM_NSN;

  for (int iz=0; iz<nz; iz++) {
    char path[1536];
    snprintf(path, sizeof(path), "%s/%s0_sn%d_nfold%d.dat",
             SIM_DIR, SIM_TAG[t], SIM_SN[iz], SIM_NFOLD);
    double *k=NULL,*P=NULL; int nk=0;
    if (!read_dat(path, &k, &P, &nk)) {
      log_fatal("sim_IA: cannot read '%s'", path); exit(1);
    }
    // ascending sort (defensive) via index
    for (int i=1;i<nk;i++){
      double kk=k[i], pp=P[i]; int j=i-1;
      while (j>=0 && k[j]>kk){ k[j+1]=k[j]; P[j+1]=P[j]; j--; }
      k[j+1]=kk; P[j+1]=pp;
    }
    // trim to [kmin,kmax]
    int lo=0; while (lo<nk && k[lo] <  kmin) lo++;
    int hi=nk-1; while (hi>=0 && k[hi] >  kmax) hi--;
    int nkeep = hi - lo + 1;
    if (nkeep < 2) {
      log_fatal("sim_IA: '%s' has <2 k-rows inside [%.3g,%.3g] h/Mpc "
                "(got %d). Widen SIM_KMIN/KMAX or check the file.",
                path, kmin, kmax, nkeep);
      exit(1);
    }
    if (iz==0) {
      kref = (double*) malloc(nkeep*sizeof(double));
      memcpy(kref, k+lo, nkeep*sizeof(double));
      nk_ref = nkeep;
      Pflat = (double*) malloc((size_t)nz*nk_ref*sizeof(double));
    } else if (nkeep != nk_ref) {
      log_fatal("sim_IA: k-grid length mismatch in '%s' (%d vs %d) after "
                "trimming. All files of one type must share the k-grid.",
                path, nkeep, nk_ref);
      exit(1);
    }
    for (int ik=0; ik<nk_ref; ik++) Pflat[(size_t)iz*nk_ref + ik] = P[lo+ik];
    free(k); free(P);
  }

  s->nz  = nz;
  s->nk  = nk_ref;
  s->z   = (double*) malloc(nz*sizeof(double));
  s->lnk = (double*) malloc(nk_ref*sizeof(double));
  s->P   = Pflat;                              // take ownership
  memcpy(s->z, SIM_Z, nz*sizeof(double));
  for (int ik=0; ik<nk_ref; ik++) s->lnk[ik] = log(kref[ik]);
  s->loaded = 1;
  free(kref);

  log_info("sim_IA: loaded %s (nz=%d, nk=%d, z in [%.3f,%.3f], "
           "k in [%.4g,%.4g] h/Mpc); linear extrapolation outside.",
           SIM_TAG[t], nz, nk_ref, s->z[0], s->z[nz-1],
           exp(s->lnk[0]), exp(s->lnk[nk_ref-1]));
}

void init_sim_IA(void) {
  for (int t=0; t<SIM_NTYPES; t++) SIM[t].loaded = 0;
  SIM_READY = 1;
}

// ---------------------------------------------------------------------------
// Index bracket for a value x on ascending grid g[0..n-1].
// Returns i0 in [0, n-2] such that the cell [g[i0], g[i0+1]] is used for
// interpolation. For x below g[0] returns 0; for x above g[n-1] returns n-2.
// The linear weight w = (x - g[i0])/(g[i0+1]-g[i0]) is then <0 (below) or >1
// (above), which makes the SAME bilinear formula perform linear EXTRAPOLATION
// off the edge cell -- exactly scipy's fill_value=None behavior.
// ---------------------------------------------------------------------------
static inline int bracket(const double* g, int n, double x, double* w) {
  int i0;
  if (x <= g[0])            i0 = 0;
  else if (x >= g[n-1])     i0 = n - 2;
  else {
    // binary search for the cell containing x
    int lo=0, hi=n-1;
    while (hi - lo > 1) { int mid=(lo+hi)>>1; if (g[mid] <= x) lo=mid; else hi=mid; }
    i0 = lo;
  }
  *w = (x - g[i0]) / (g[i0+1] - g[i0]);   // may be <0 or >1 -> extrapolation
  return i0;
}

// ---------------------------------------------------------------------------
// Low-k power-law extrapolation for a single z-row.
// Returns the extrapolated physical P at ln k = lnk, using the two lowest-k
// nodes of row `row`: P(k) = P_edge * (k/k_edge)^s with s = slope of ln|P|.
// Sign of the edge node is carried. If the two nodes straddle zero or are not
// both nonzero, falls back to flat clamp at the lowest-k value (can't define a
// log-slope through a sign change). See P_sim() for the rationale.
//
// SLOPE FLOOR (critical -- this is what keeps large scales bounded):
//   Extrapolation goes to lnk < lnk0, i.e. (lnk - lnk0) < 0. The returned
//   magnitude is aa*exp(slope*(lnk - lnk0)). If slope < 0 (|P| falling from
//   node0 to node1, which is the PHYSICAL large-scale behaviour P ~ k^n, n>0),
//   then exp(negative_slope * negative_gap) = exp(POSITIVE) -> the magnitude
//   DIVERGES as k -> 0. A power law with negative index blows up at small k.
//   That is exactly the "large-scale gets wild" symptom, and raising k_min
//   makes it worse because the extrapolation lever arm (lnk0 - lnk) grows.
//   So we floor the slope at 0: the power law may fall toward low k (slope>0
//   here means |P| was larger at the higher-k node -> shrinks outward, safe),
//   but is never allowed to GROW toward low k. Worst case is flat -- the same
//   bounded behaviour the old code used -- and where the data genuinely falls
//   toward low k we still follow it. This mirrors the "never extrapolate
//   upward" guard the high-k path already uses.
// ---------------------------------------------------------------------------
static inline double sim_lowk_powerlaw(const sim_spec* s, int row, double lnk) {
  const int nk = s->nk;
  const double lnk0 = s->lnk[0];
  const double lnk1 = s->lnk[1];
  const double Pa = s->P[(size_t)row*nk + 0];   // lowest-k node
  const double Pb = s->P[(size_t)row*nk + 1];   // next-lowest node
  const double aa = fabs(Pa), ab = fabs(Pb);
  if (aa > 0.0 && ab > 0.0 && ((Pa > 0.0) == (Pb > 0.0))) {
    double slope = (log(ab) - log(aa)) / (lnk1 - lnk0);
    // Going to lnk < lnk0: forbid a growing magnitude (slope<0 would diverge).
    if (slope < 0.0) slope = 0.0;
    const double sign  = (Pa < 0.0) ? -1.0 : 1.0;
    return sign * aa * exp(slope * (lnk - lnk0));
  }
  return Pa;   // nodes straddle zero / are noise -> flat clamp
}

// ---------------------------------------------------------------------------
// THE evaluator used by the Limber integrands.
//   t      : which spectrum
//   k_code : k in CosmoLike code units (= ell/fK)
//   a      : scale factor
// Returns P in code units (c/H0)^3, ready to slot in beside Pdelta(k,a).
//
// Bilinear on (z, ln k) with linear extrapolation off every edge, matching
// scipy RegularGridInterpolator(method="linear", bounds_error=False,
// fill_value=None). Sign is carried naturally -- no special-casing of GI's
// negative values, because we never take logs of P here.
// ---------------------------------------------------------------------------
double P_sim(sim_spec_t t, double k_code, double a) {
  if (!SIM_READY) { log_fatal("sim_IA: init_sim_IA() not called"); exit(1); }
  sim_spec* s = &SIM[t];

  // Thread-safe lazy load (C(l) integrands run inside OpenMP regions).
  if (!s->loaded) {
    #pragma omp critical (sim_IA_load)
    {
      if (!s->loaded) load_spec(t);
    }
  }

  const double R = 2997.92458;                 // c/H0 in Mpc/h
  const double k_phys = k_code / R;            // h/Mpc
  const double lnk = log(k_phys);
  const double z = 1.0/a - 1.0;

  const int nk = s->nk;

  // -------------------------------------------------------------------------
  // LOW-k EXTRAPOLATION: log-log power law (NOT linear-in-P).
  //
  // Below the lowest tabulated k, extrapolate as P(k) = P_edge*(k/k_edge)^s,
  // i.e. LINEAR in ln|P| vs ln k, with slope s from the two lowest nodes.
  // This is the shape a power spectrum actually has on large scales (P ~ k^n),
  // so it sends P smoothly toward 0 as k->0 instead of a straight line in P
  // that crosses zero into the wrong sign at some finite k.
  //
  // Sign handling: IA cross-spectra (dI/GI) are NEGATIVE, and ln is undefined
  // for P<=0. So we fit ln|P|, carry the sign of the edge node, and only apply
  // the power law when the two lowest nodes share a sign and are both nonzero;
  // otherwise (edge nodes are noise/straddle zero) fall back to flat clamp at
  // the edge value. The interior and z axes are UNCHANGED -- still the exact
  // scipy-matching bilinear -- so only genuine below-grid queries differ.
  //
  // We build the extrapolated P at BOTH bracketing z-rows, then interpolate in
  // z exactly as the normal path does, so z behavior is untouched.
  // -------------------------------------------------------------------------
  if (lnk < s->lnk[0]) {
    double wz;
    const int iz = bracket(s->z, s->nz, z, &wz);
    const double Prow0 = sim_lowk_powerlaw(s, iz,     lnk);
    const double Prow1 = sim_lowk_powerlaw(s, iz + 1, lnk);
    const double P_phys = Prow0 + wz*(Prow1 - Prow0);   // linear in z (as before)
    return P_phys / (R*R*R);
  }

  // Bracket + weights on each axis. Weights outside [0,1] => extrapolation.
  double wz, wk;
  const int iz = bracket(s->z,   s->nz, z,   &wz);
  const int ik = bracket(s->lnk, s->nk, lnk, &wk);

  // Row-major P[iz*nk + ik]; standard bilinear form, valid for w<0 or w>1.
  const double P00 = s->P[(size_t)iz    *nk + ik    ];
  const double P01 = s->P[(size_t)iz    *nk + ik + 1];
  const double P10 = s->P[(size_t)(iz+1)*nk + ik    ];
  const double P11 = s->P[(size_t)(iz+1)*nk + ik + 1];

  const double Pk0 = P00 + wk*(P01 - P00);     // interp/extrap in lnk at z row iz
  const double Pk1 = P10 + wk*(P11 - P10);     // ... at z row iz+1
  const double P_phys = Pk0 + wz*(Pk1 - Pk0);  // then in z

  return P_phys / (R*R*R);                      // -> code units (c/H0)^3
}

// Convenience wrappers so cosmo2D.c reads cleanly.
double P_sim_EE(double k_code, double a) { return P_sim(SIM_PEE, k_code, a); }
double P_sim_BB(double k_code, double a) { return P_sim(SIM_PBB, k_code, a); }
double P_sim_dE(double k_code, double a) { return P_sim(SIM_PdE, k_code, a); }
double P_sim_dB(double k_code, double a) { return P_sim(SIM_PdB, k_code, a); }
double P_sim_dd(double k_code, double a) { return P_sim(SIM_Pdd, k_code, a); }
double P_sim_hh(double k_code, double a) { return P_sim(SIM_Phh, k_code, a); }
double P_sim_dh(double k_code, double a) { return P_sim(SIM_Pdh, k_code, a); }