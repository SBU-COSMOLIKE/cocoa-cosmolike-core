// ============================================================================
// sim_IA.c  --  simulation-measured IA / bias power spectra for CosmoLike
//
// Injects tabulated P(k,z) spectra measured from a simulation (the IA_PS/Pds
// stage) directly into the Limber integrands of cosmo2D.c, bypassing the
// analytic NLA/TATT models entirely.
//
// Design (mirrors what we built on the Python side):
//   * one .dat file per snapshot/redshift, per spectrum type;
//   * each file: col0 = k [h/Mpc], col1 = P [h^-3 Mpc^3] (cols 2,3 ignored);
//   * spectra are stored on a (z, ln k) rectangle and bilinearly interpolated;
//   * NO nuisance parameters, NO growth factor, NO redshift-evolution model --
//     all evolution is already baked into the per-redshift files.
//
// Unit convention (critical -- see cosmo2D.c lines ~1163, 1251, 3823):
//   Inside CosmoLike, k is in "code units" k_code = ell/fK with units
//   (c/H0)/(Mpc/h), and power spectra are carried in (c/H0)^3. Pdelta(k,a),
//   P_dI_halo, etc. all use this convention.
//   Our files are physical: k_phys [h/Mpc], P_phys [h^-3 Mpc^3]. With
//       R = real_coverH0 = cosmology.coverH0/cosmology.h0   ([Mpc/h], = c/H0)
//   the conversions are:
//       k_phys  = k_code / R
//       P_code  = P_phys / R^3
//   So P_sim() below takes k_code, converts to k_phys to look up the table,
//   and divides the tabulated P_phys by R^3 before returning. This is exactly
//   the rescaling the halo-model branch says it does NOT need (because halo.c
//   already returns code units) -- we DO need it, because our files are in h-
//   units.
//
// Author: (generated to spec) -- see PATCH_cosmo2D.md for the wiring.
// ============================================================================

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_interp2d.h>

#include "structs.h"       // cosmology, nuisance
#include "cosmo3D.h"       // f_K, chi, a_chi (not strictly needed here)
#include "log.c/src/log.h"

// ---------------------------------------------------------------------------
// Spectrum types. Extend this list to match the files you actually produce.
// The string is the prefix in the filename: P<TAG>0_sn<sn>_nfold1.dat
// e.g. SIM_PEE -> "PEE0_sn67_nfold1.dat".
// ---------------------------------------------------------------------------
typedef enum {
  SIM_PEE = 0,   // shear E-mode auto (II, E)
  SIM_PBB,       // shear B-mode auto (II, B)
  SIM_PdE,       // density x E  (GI / dI)
  SIM_PdB,       // density x B
  SIM_Pdd,       // matter density auto (for gg clustering, if desired)
  SIM_Phh,       // galaxy-number auto  (bias clustering)
  SIM_Pdh,       // density x number
  SIM_NTYPES
} sim_spec_t;

static const char* SIM_TAG[SIM_NTYPES] = {
  "PEE", "PBB", "PdE", "PdB", "Pdd", "Phh", "Pdh"
};

// ---------------------------------------------------------------------------
// One interpolated spectrum: a (z, lnk) grid + GSL bilinear spline.
// ---------------------------------------------------------------------------
typedef struct {
  int loaded;
  int nz, nk;
  double* z;         // ascending, length nz
  double* lnk;       // ascending, length nk
  double* P;         // nz*nk, row-major P[iz*nk + ik]  (physical, h^-3 Mpc^3)
  gsl_spline2d* sp;
  gsl_interp_accel* za;
  gsl_interp_accel* ka;
  double lnk_min, lnk_max, z_min, z_max;
} sim_spec;

static sim_spec SIM[SIM_NTYPES];

// Config the host must set before first use (see init_sim_IA()).
static char   SIM_DIR[1024]   = "/xdisk/timeifler/yijiezhu/cocoa_demo/Cocoa/PS_2re_unred_new";   // directory holding the .dat files
static int    SIM_NSN         = 0;     // number of snapshots
static int*   SIM_SN          = NULL;  // snapshot indices, length NSN
static double* SIM_Z          = NULL;  // matching redshifts, length NSN
static int    SIM_NFOLD       = 1;
static int    SIM_READY       = 0;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
// Count rows and read (k,P) from one .dat file. Returns malloc'd arrays via
// out_k/out_P and the count via *n. Caller frees. Ignores cols 2,3.
static int read_dat(const char* path, double** out_k, double** out_P, int* n) {
  FILE* f = fopen(path, "r");
  if (!f) return 0;
  int cap = 256, m = 0;
  double* k = (double*) malloc(cap*sizeof(double));
  double* P = (double*) malloc(cap*sizeof(double));
  char line[512];
  while (fgets(line, sizeof(line), f)) {
    // skip blank / comment
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

// ---------------------------------------------------------------------------
// Public config entry point. Call ONCE at startup (before any C_*_limber).
//   dir     : folder with the .dat files
//   nsn     : number of snapshots
//   sn      : array[nsn] of snapshot indices (used to build filenames)
//   zlist   : array[nsn] of redshifts matching sn
//   nfold   : the nfold value in the filename (usually 1)
// The (sn,z) pairs need not be sorted; we sort by z internally.
// ---------------------------------------------------------------------------
void set_sim_IA_config(const char* dir, int nsn, const int* sn,
                       const double* zlist, int nfold) {
  strncpy(SIM_DIR, dir, sizeof(SIM_DIR)-1);
  SIM_DIR[sizeof(SIM_DIR)-1] = '\0';
  SIM_NSN   = nsn;
  SIM_NFOLD = nfold;
  free(SIM_SN); free(SIM_Z);
  SIM_SN = (int*)    malloc(nsn*sizeof(int));
  SIM_Z  = (double*) malloc(nsn*sizeof(double));

  // sort snapshot indices by ascending z
  int* order = (int*) malloc(nsn*sizeof(int));
  for (int i=0;i<nsn;i++) order[i]=i;
  // simple insertion sort by zlist (nsn is tiny)
  for (int i=1;i<nsn;i++){
    int key=order[i]; int j=i-1;
    while (j>=0 && zlist[order[j]] > zlist[key]) { order[j+1]=order[j]; j--; }
    order[j+1]=key;
  }
  for (int i=0;i<nsn;i++){ SIM_SN[i]=sn[order[i]]; SIM_Z[i]=zlist[order[i]]; }
  free(order);
  SIM_READY = 0; // force (re)load on next access
}

// ---------------------------------------------------------------------------
// Load one spectrum type across all snapshots into a (z, lnk) grid.
// Requires every file to share the same k-grid (as Pds produces).
// ---------------------------------------------------------------------------
static void load_spec(sim_spec_t t) {
  sim_spec* s = &SIM[t];
  if (s->loaded) return;
  if (SIM_NSN < 2) {
    log_fatal("sim_IA: need >=2 snapshots to interpolate in z (have %d). "
              "Call set_sim_IA_config() first.", SIM_NSN);
    exit(1);
  }

  double* kref = NULL;
  int nk_ref = 0;
  double* Pflat = NULL;   // nz*nk
  int nz = SIM_NSN;

  for (int iz=0; iz<nz; iz++) {
    char path[1536];
    snprintf(path, sizeof(path), "%s/%s0_sn%d_nfold%d.dat",
             SIM_DIR, SIM_TAG[t], SIM_SN[iz], SIM_NFOLD);
    double *k=NULL,*P=NULL; int nk=0;
    if (!read_dat(path, &k, &P, &nk)) {
      log_fatal("sim_IA: cannot read '%s'", path);
      exit(1);
    }
    // NOTE: shot-noise (shape-noise) subtraction is now done UPSTREAM in the
    // Pds/power.hpp stage (BinnedData::subtract_const), so the .dat files are
    // already floor-subtracted. Do NOT subtract anything here -- the old
    // FLOOR_SANITY=300 stub drove the (already small) EE/BB below zero and the
    // clamp zeroed the whole signal. It has been removed.

    if (iz==0) {
      kref = (double*) malloc(nk*sizeof(double));
      memcpy(kref, k, nk*sizeof(double));
      nk_ref = nk;
      Pflat = (double*) malloc((size_t)nz*nk*sizeof(double));
    } else if (nk != nk_ref) {
      log_fatal("sim_IA: k-grid length mismatch in '%s' (%d vs %d). "
                "All files of one type must share the k-grid.", path, nk, nk_ref);
      exit(1);
    }
    for (int ik=0; ik<nk_ref; ik++) Pflat[(size_t)iz*nk_ref + ik] = P[ik];
    free(k); free(P);
  }

  // Build ascending lnk (files are ascending in k already, but be safe).
  s->nz  = nz;
  s->nk  = nk_ref;
  s->z   = (double*) malloc(nz*sizeof(double));
  s->lnk = (double*) malloc(nk_ref*sizeof(double));
  memcpy(s->z, SIM_Z, nz*sizeof(double));
  for (int ik=0; ik<nk_ref; ik++) s->lnk[ik] = log(kref[ik]);

  // GSL spline2d wants z[iz] as x-grid, lnk[ik] as y-grid, and a flat array
  // laid out with gsl_spline2d_set(sp, za, ix, iy, val).
  s->sp = gsl_spline2d_alloc(gsl_interp2d_bilinear, nz, nk_ref);
  s->za = gsl_interp_accel_alloc();
  s->ka = gsl_interp_accel_alloc();
  s->P  = (double*) malloc((size_t)nz*nk_ref*sizeof(double));

  double* za_grid = (double*) malloc((size_t)nz*nk_ref*sizeof(double));
  for (int iz=0; iz<nz; iz++)
    for (int ik=0; ik<nk_ref; ik++)
      gsl_spline2d_set(s->sp, za_grid, iz, ik, Pflat[(size_t)iz*nk_ref + ik]);
  gsl_spline2d_init(s->sp, s->z, s->lnk, za_grid, nz, nk_ref);
  memcpy(s->P, za_grid, (size_t)nz*nk_ref*sizeof(double));
  free(za_grid);

  s->z_min = s->z[0];       s->z_max = s->z[nz-1];
  // Trustworthy k-range for the interpolator, set from the data-quality
  // profile of the 18-snapshot set (fractional error vs k):
  //   * below K_FLOOR the low-k bins are few-mode and noisy (err >~ 20%, and
  //     the first 3 bins have only 3/10/27 modes) -> they inject large-theta
  //     xi+ sign-flips. Clamp flat below K_FLOOR.
  //   * above K_CAP the top bins are aliasing/window-suppressed near Nyquist
  //     (k=13 bin ~53% error) -> flat power there manufactures small-theta
  //     xi- spikes via J4. Handled by the falloff policy in P_sim().
  {
    const double K_FLOOR = 0.2;                // h/Mpc  (low-k reliability edge)
    const double K_CAP   = 7.0;                // h/Mpc  (high-k reliability edge)
    const double lnk_floor = log(K_FLOOR);
    const double lnk_cap   = log(K_CAP);
    const double lnk_bot = s->lnk[0];
    const double lnk_top = s->lnk[nk_ref-1];
    s->lnk_min = (lnk_bot > lnk_floor) ? lnk_bot : lnk_floor;
    s->lnk_max = (lnk_top < lnk_cap)   ? lnk_top : lnk_cap;
  }
  s->loaded = 1;
  free(kref); free(Pflat);

  log_info("sim_IA: loaded %s  (nz=%d, nk=%d, z in [%.3f,%.3f], "
           "k in [%.4g,%.4g] h/Mpc)", SIM_TAG[t], nz, nk_ref,
           s->z_min, s->z_max, exp(s->lnk_min), exp(s->lnk_max));
}

void init_sim_IA(void) {
  for (int t=0; t<SIM_NTYPES; t++) SIM[t].loaded = 0;
  // Lazy: individual specs load on first P_sim() call for that type, so you
  // only pay for the spectra your data vector actually uses.
  SIM_READY = 1;
}

// ---------------------------------------------------------------------------
// THE evaluator used by the Limber integrands.
//   t      : which spectrum
//   k_code : k in CosmoLike code units (= ell/fK)
//   a      : scale factor
// Returns P in code units (c/H0)^3, ready to slot in beside Pdelta(k,a).
//
// Out-of-range policy:
//   * k below/above the tabulated range -> clamp lnk to the edge (flat
//     extrapolation). IA spectra fall smoothly; clamping avoids NaNs at the
//     ends of the ell range. Change to "return 0.0" if you prefer hard cuts.
//   * z below/above -> clamp to nearest snapshot. Your snapshots should span
//     the source/lens n(z); if a query lands outside, the geometry weight
//     there is usually tiny, but clamping is the safe default.
// ---------------------------------------------------------------------------
double P_sim(sim_spec_t t, double k_code, double a) {
  if (!SIM_READY) { log_fatal("sim_IA: init_sim_IA() not called"); exit(1); }
  sim_spec* s = &SIM[t];
  // Thread-safe lazy load: the C(l) integrands run inside OpenMP parallel
  // regions, so the first touch of a given spectrum could race two threads
  // into load_spec at once. Double-checked lock: the cheap unsynchronized
  // read short-circuits once loaded; the critical section guards the build.
  // (CosmoLike solves the analogous halo-model race by forcing a serial
  //  warm-up call in each driver; guarding here keeps the module correct
  //  regardless of where the first call lands.)
  if (!s->loaded) {
    #pragma omp critical (sim_IA_load)
    {
      if (!s->loaded) load_spec(t);
    }
  }

  const double R = cosmology.coverH0 / cosmology.h0;//cosmology.coverH0 / cosmology.h0;   // c/H0 in Mpc/h
  const double k_phys = k_code / R;                    // h/Mpc
  double lnk = log(k_phys);
  double z = 1.0/a - 1.0;

  // z clamp: snapshots should span the n(z); outside, geometry weight is tiny.
  if (z < s->z_min) z = s->z_min;
  if (z > s->z_max) z = s->z_max;

  // low-k clamp: flat below K_FLOOR (set in load_spec). The lowest table bins
  // are few-mode and noisy; xi via J0/J4 has little sensitivity to the very
  // largest scales, so holding P flat there is harmless and avoids the noisy
  // bottom bins driving large-theta xi+ through zero.
  if (lnk < s->lnk_min) lnk = s->lnk_min;

  // --- high-k out-of-range policy ---------------------------------------
  // Above the (capped) table top we must NOT hold P flat: a non-decaying
  // plateau fed through the ell*J4 kernel manufactures large xi- at small
  // theta. Instead extrapolate the MAGNITUDE with the local log-slope from
  // the top two nodes and restore the sign. Handling the sign explicitly is
  // essential: cross-spectra like PdE (GI) are NEGATIVE, and a naive
  // "P1>0 && P2>0" guard would fail for them and zero the entire GI term
  // above the cap -- which silently killed the dominant IA contribution.
  if (lnk > s->lnk_max) {
    const double lnk1 = s->lnk_max - (s->lnk[1] - s->lnk[0]); // one step below cap
    const double P2 = gsl_spline2d_eval(s->sp, z, s->lnk_max, s->za, s->ka);
    const double P1 = gsl_spline2d_eval(s->sp, z, lnk1,       s->za, s->ka);
    const double a2 = fabs(P2), a1 = fabs(P1);
    // Need both magnitudes positive and a consistent (non-flipping) sign to
    // define a log-slope; otherwise the top nodes are noise -> no power.
    if (a1 > 0.0 && a2 > 0.0 && (P1 > 0.0) == (P2 > 0.0)) {
      double slope = (log(a2) - log(a1)) / (s->lnk_max - lnk1);
      if (slope > 0.0) slope = 0.0;               // never extrapolate upward
      const double sign  = (P2 < 0.0) ? -1.0 : 1.0;
      const double a_ex  = a2 * exp(slope * (lnk - s->lnk_max));
      return sign * a_ex / (R*R*R);               // restore GI's negative sign
    }
    return 0.0;                                    // top nodes are noise -> no power
  }

  const double P_phys = gsl_spline2d_eval(s->sp, z, lnk, s->za, s->ka);
  return P_phys / (R*R*R);   // -> code units (c/H0)^3
}

// Convenience wrappers so cosmo2D.c reads cleanly.
double P_sim_EE(double k_code, double a) { return P_sim(SIM_PEE, k_code, a); }
double P_sim_BB(double k_code, double a) { return P_sim(SIM_PBB, k_code, a); }
double P_sim_dE(double k_code, double a) { return P_sim(SIM_PdE, k_code, a); }
double P_sim_dB(double k_code, double a) { return P_sim(SIM_PdB, k_code, a); }
double P_sim_dd(double k_code, double a) { return P_sim(SIM_Pdd, k_code, a); }
double P_sim_hh(double k_code, double a) { return P_sim(SIM_Phh, k_code, a); }
double P_sim_dh(double k_code, double a) { return P_sim(SIM_Pdh, k_code, a); }