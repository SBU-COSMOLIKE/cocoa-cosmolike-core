// ============================================================================
// sim_IA.c  --  simulation-measured IA / bias power spectra for CosmoLike
//
// Injects tabulated P(k,z) spectra measured from a simulation (the IA_PS/Pds
// stage) directly into the Limber integrands of cosmo2D.c, bypassing the
// analytic NLA/TATT/halo models entirely.
//
// ----------------------------------------------------------------------------
// WHAT THE TABLES ARE, AND WHAT COSMOLIKE NEEDS
// ----------------------------------------------------------------------------
// The files P<TAG><l>_sn<sn>_nfold<n>.dat hold, for the 3D E/B decomposition
// of the projected galaxy shapes,
//     M_l(k) = < P(k, mu) L_l(mu) >_modes      (no 2l+1 factor),
// with shapes defined as the DISTORTION
//     e1 = (I11 - I22)/(I11 + I22),   e2 = 2 I12/(I11 + I22)    (= chi).
// The Limber integrands in cosmo2D.c need something else in two ways:
//
// (1) SHEAR, not distortion. The IA term adds to the lensing shear, so it
//     must be the intrinsic shear gamma^I. A population with per-component
//     rms e_rms responds to shear as <chi> = 2 R gamma, R = 1 - e_rms^2
//     (Bernstein & Jarvis 2002; Fortuna et al. 2021, footnote 8). Every IA
//     shape leg is divided by 2R:
//         dE, dB : x 1/(2R)        EE, BB : x 1/(2R)^2
//     e_rms is per snapshot (set_sim_IA_erms / set_sim_IA_erms_from_hinfo).
//
// (2) mu = 0, not the monopole. Limber only uses k perpendicular to the
//     line of sight. The 3D E-mode carries (1 - mu^2) per shape leg, so the
//     monopole is a mu-average of a mu-dependent spectrum. Two treatments:
//       SIM_IA_LOS_LA        index-0 files only; P(mu=0) = M0 x 3/2 (dE),
//                            x 15/8 (EE). Exact for the linear-alignment
//                            structure (1-mu^2)^n; no factor for BB, dB.
//                            Measured: holds for dE at all k (TNG300,
//                            P(mu=0)/M0 = 1.5 +/- 0.1 for k = 0.1-5 h/Mpc);
//                            for EE it OVERCORRECTS at k > 0.3 h/Mpc, where
//                            the measured ratio is ~1.0-1.3.
//       SIM_IA_LOS_MULTIPOLE uses the index-0, 2, 4 files:
//                            P(mu=0) = sum_l (2l+1) M_l L_l(0)
//                                    = M0 - (5/2) M2 + (27/8) M4.
//                            No angular model assumed; exact if the spectrum
//                            has no l > 4 structure ((1-mu^2)^2 has none).
//       SIM_IA_LOS_NONE      monopole as is (the previous behavior).
//
// Both conversions are applied ONCE, per snapshot row, when a table is
// loaded, so interpolation and extrapolation act on the converted spectra.
// Pdd, Phh, Pdh are not shape spectra and are never converted.
//
// Default: apply_shear = 1, los_mode = SIM_IA_LOS_LA. With apply_shear = 1 an
// IA table refuses to load until e_rms has been supplied (fail loud rather
// than silently inject distortion-unit spectra).
//
// ----------------------------------------------------------------------------
// INTERPOLATION / EXTRAPOLATION
// ----------------------------------------------------------------------------
//   * Interior: bilinear in (z, ln k), linear in P (IA spectra change sign).
//   * Above the top k node and outside the z range: linear-in-P
//     extrapolation from the two edge nodes (scipy RegularGridInterpolator
//     with fill_value=None).
//   * Below the lowest k node: sign-preserving power law in |P| from the two
//     lowest nodes, with the log-slope floored at 0 so |P| never grows toward
//     k -> 0; flat clamp if the two nodes differ in sign.
//   * Rows outside [SIM_KMIN_HMPC, SIM_KMAX_HMPC] are trimmed at load time,
//     so the edge nodes used above are the reliable ones.
//   WARNING: linear-in-P extrapolation of a noisy, sign-changing spectrum can
//   flip sign past the edges; inspect xi+/xi- after changing the k band.
//
// ----------------------------------------------------------------------------
// UNITS (see cosmo2D.c)
// ----------------------------------------------------------------------------
//   k_code = ell/fK in (c/H0)^-1 units; P carried in (c/H0)^3. Files are
//   physical (h/Mpc, (Mpc/h)^3). With R = c/H0 = 2997.92458 Mpc/h:
//       k_phys = k_code / R,      P_code = P_phys / R^3.
// ============================================================================

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "structs.h"       // cosmology, nuisance
#include "cosmo3D.h"       // f_K, chi, a_chi (not strictly needed here)
#include "log.c/src/log.h"
#include "sim_IA.h"

// ---------------------------------------------------------------------------
// Spectrum types. The string is the prefix in the filename:
// P<TAG><l>_sn<sn>_nfold<n>.dat  e.g. SIM_PEE -> "PEE0_sn67_nfold1.dat".
// Pdd/Phh/Pdh carry no multipole index in their filenames.
// ---------------------------------------------------------------------------
typedef enum {
  SIM_PEE = 0,   // shape E-mode auto (II, E)
  SIM_PBB,       // shape B-mode auto (II, B)
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

// Number of shape (IA) legs in each spectrum: the power of 1/(2R).
static const int SIM_NSHAPE[SIM_NTYPES] = { 2, 2, 1, 1, 0, 0, 0 };

// Does the filename carry a multipole index ("PEE0_...")?
static const int SIM_HAS_L[SIM_NTYPES]  = { 1, 1, 1, 1, 0, 0, 0 };

// SIM_IA_LOS_LA: P(mu=0)/M0 for the linear-alignment angular structure.
//   EE ~ (1-mu^2)^2 -> 1/(8/15) = 15/8;  dE ~ (1-mu^2) -> 1/(2/3) = 3/2.
//   LA predicts no B modes, so BB and dB get no factor.
static const double SIM_LA_LOS[SIM_NTYPES] = {
  15.0/8.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0
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
#define SIM_KMAX_HMPC 7.0      // high-k trusted edge [h/Mpc] (0 = no cap)
#endif

// ---------------------------------------------------------------------------
// One interpolated spectrum: a (z, lnk) grid stored row-major P[iz*nk + ik]
// in PHYSICAL units (h^-3 Mpc^3), already converted (shear, mu = 0).
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
static int*    SIM_SN        = NULL;     // sorted by ascending z
static double* SIM_Z         = NULL;
static int     SIM_NFOLD     = 1;
static int     SIM_READY     = 0;

// Shape convention.
static int     SIM_SHEAR     = 1;
static int     SIM_LOS       = SIM_IA_LOS_LA;

// e_rms inputs: sparse (sn, e_rms) list as supplied by the host.
static int     SIM_NE        = 0;
static int*    SIM_E_SN      = NULL;
static double* SIM_E_VAL     = NULL;
static char    SIM_E_SRC[64] = "none";

// e_rms resolved onto the configured snapshot list (same order as SIM_Z).
static int     SIM_E_RESOLVED = 0;
static double* SIM_ERMS       = NULL;

// ---------------------------------------------------------------------------
// File reading
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
  if (m == 0) { free(k); free(P); return 0; }
  *out_k = k; *out_P = P; *n = m;
  return 1;
}

// read_dat + ascending sort in k (defensive); fatal if missing.
static void read_sorted_or_die(const char* path, double** k, double** P,
                               int* n) {
  if (!read_dat(path, k, P, n)) {
    log_fatal("sim_IA: cannot read '%s'", path); exit(1);
  }
  double* kk = *k; double* pp = *P;
  for (int i=1;i<*n;i++){
    double kv=kk[i], pv=pp[i]; int j=i-1;
    while (j>=0 && kk[j]>kv){ kk[j+1]=kk[j]; pp[j+1]=pp[j]; j--; }
    kk[j+1]=kv; pp[j+1]=pv;
  }
}

static void spec_path(char* buf, size_t sz, sim_spec_t t, int ell, int sn) {
  if (SIM_HAS_L[t])
    snprintf(buf, sz, "%s/%s%d_sn%d_nfold%d.dat",
             SIM_DIR, SIM_TAG[t], ell, sn, SIM_NFOLD);
  else
    snprintf(buf, sz, "%s/%s_sn%d_nfold%d.dat",
             SIM_DIR, SIM_TAG[t], sn, SIM_NFOLD);
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
static void free_spec(sim_spec* s) {
  free(s->z); free(s->lnk); free(s->P);
  s->z = NULL; s->lnk = NULL; s->P = NULL;
  s->loaded = 0; s->nz = 0; s->nk = 0;
}

static void drop_tables(void) {
  for (int t=0; t<SIM_NTYPES; t++) free_spec(&SIM[t]);
  free(SIM_ERMS); SIM_ERMS = NULL;
  SIM_E_RESOLVED = 0;
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
  drop_tables();
  SIM_READY = 0;
}

void set_sim_IA_shape_convention(int apply_shear, int los_mode) {
  if (los_mode != SIM_IA_LOS_NONE && los_mode != SIM_IA_LOS_LA &&
      los_mode != SIM_IA_LOS_MULTIPOLE) {
    log_fatal("sim_IA: unknown los_mode %d (use %d none, %d LA, %d multipole)",
              los_mode, SIM_IA_LOS_NONE, SIM_IA_LOS_LA, SIM_IA_LOS_MULTIPOLE);
    exit(1);
  }
  SIM_SHEAR = apply_shear ? 1 : 0;
  SIM_LOS   = los_mode;
  drop_tables();
}

void set_sim_IA_erms(int n, const int* sn, const double* erms) {
  free(SIM_E_SN); free(SIM_E_VAL);
  SIM_NE = 0; SIM_E_SN = NULL; SIM_E_VAL = NULL;
  if (n > 0) {
    SIM_E_SN  = (int*)    malloc(n*sizeof(int));
    SIM_E_VAL = (double*) malloc(n*sizeof(double));
    for (int i=0;i<n;i++) {
      if (erms[i] >= 1.0) {
        log_fatal("sim_IA: e_rms = %g for sn %d is >= 1 (R = 1 - e_rms^2 "
                  "would be <= 0)", erms[i], sn[i]);
        exit(1);
      }
      SIM_E_SN[i] = sn[i]; SIM_E_VAL[i] = erms[i];
    }
    SIM_NE = n;
  }
  snprintf(SIM_E_SRC, sizeof(SIM_E_SRC), "user list");
  drop_tables();
}

// Read e_rms = (e1z_rms + e2z_rms)/2 and ngal_used from one hinfo file.
// Returns 1 on success. Format: header
//   "#ngal ngal_nan ngal_used e1z_rms e2z_rms ..." then one data row.
static int read_hinfo(const char* path, double* e_rms, long* ngal_used) {
  FILE* f = fopen(path, "r");
  if (!f) return 0;
  char line[1024];
  int ok = 0;
  while (fgets(line, sizeof(line), f)) {
    char* p = line;
    while (*p==' '||*p=='\t') p++;
    if (*p=='#' || *p=='\n' || *p=='\0') continue;
    double ngal, nnan, nused, e1z, e2z;
    if (sscanf(p, "%lf %lf %lf %lf %lf", &ngal, &nnan, &nused, &e1z, &e2z)
        == 5 && e1z > 0 && e2z > 0) {
      *e_rms = 0.5*(e1z + e2z);
      *ngal_used = (long) nused;
      ok = 1;                                   // keep the LAST data row
    }
  }
  fclose(f);
  return ok;
}

// Reads ONLY  <dir>/hinfo_<stem>_sn<sn>_nfold<nfold>.dat  -- never *.dat.bak.
// Backups are left over from earlier galaxy selections (e.g. a different mass
// cut), so falling back to them would silently use the wrong sample. The
// files are read when this is called, so after the hinfo files change, call
// it again (or restart) to pick up the new values. ngal_used is logged per
// snapshot so the sample can be checked against the run that made the spectra.
int set_sim_IA_erms_from_hinfo(const char* dir, const char* stem,
                               int nskip, const int* skip) {
  if (SIM_NSN < 1) {
    log_fatal("sim_IA: call set_sim_IA_config() before "
              "set_sim_IA_erms_from_hinfo()");
    exit(1);
  }
  int*    sn = (int*)    malloc(SIM_NSN*sizeof(int));
  double* e  = (double*) malloc(SIM_NSN*sizeof(double));
  int n = 0;
  for (int i=0;i<SIM_NSN;i++) {
    int skipped = 0;
    for (int j=0;j<nskip;j++) if (skip[j] == SIM_SN[i]) skipped = 1;
    if (skipped) {
      log_info("sim_IA: hinfo sn %d skipped on request (e_rms interpolated)",
               SIM_SN[i]);
      continue;
    }
    char path[1536];
    double val; long nused = 0;
    snprintf(path, sizeof(path), "%s/hinfo_%s_sn%d_nfold%d.dat",
             dir, stem, SIM_SN[i], SIM_NFOLD);
    if (read_hinfo(path, &val, &nused)) {
      sn[n] = SIM_SN[i]; e[n] = val; n++;
      log_info("sim_IA: hinfo sn %3d  e_rms %.4f  ngal_used %ld  (%s)",
               SIM_SN[i], val, nused, path);
    } else {
      log_warn("sim_IA: no readable '%s'; e_rms for sn %d will be "
               "interpolated in z", path, SIM_SN[i]);
    }
  }
  if (n == 0) {
    log_fatal("sim_IA: no hinfo_%s_sn*_nfold%d.dat files read in '%s'",
              stem, SIM_NFOLD, dir);
    exit(1);
  }
  set_sim_IA_erms(n, sn, e);
  snprintf(SIM_E_SRC, sizeof(SIM_E_SRC), "hinfo (%d files)", n);
  free(sn); free(e);
  return n;
}

void init_sim_IA(void) {
  drop_tables();
  SIM_READY = 1;
}

// ---------------------------------------------------------------------------
// e_rms on the configured snapshot list: supplied values where available,
// linear interpolation in z elsewhere (clamped at the ends, like np.interp).
// ---------------------------------------------------------------------------
static void resolve_erms(void) {
  if (SIM_E_RESOLVED) return;
  // collect usable (z, e) pairs: supplied, > 0, snapshot in the config
  double* zs = (double*) malloc(SIM_NSN*sizeof(double));
  double* es = (double*) malloc(SIM_NSN*sizeof(double));
  int*    given = (int*) calloc(SIM_NSN, sizeof(int));
  int m = 0;
  for (int i=0;i<SIM_NSN;i++) {                  // SIM_Z ascending
    for (int j=0;j<SIM_NE;j++) {
      if (SIM_E_SN[j] == SIM_SN[i] && SIM_E_VAL[j] > 0.0) {
        zs[m] = SIM_Z[i]; es[m] = SIM_E_VAL[j]; m++; given[i] = 1; break;
      }
    }
  }
  for (int j=0;j<SIM_NE;j++) {
    int found = 0;
    for (int i=0;i<SIM_NSN;i++) if (SIM_E_SN[j] == SIM_SN[i]) found = 1;
    if (!found)
      log_warn("sim_IA: e_rms given for sn %d, which is not in the "
               "configured snapshot list; ignored", SIM_E_SN[j]);
  }
  if (m == 0) {
    log_fatal("sim_IA: shear conversion is on (apply_shear = 1) but no e_rms "
              "was supplied for the configured snapshots. Call "
              "set_sim_IA_erms() or set_sim_IA_erms_from_hinfo(), or "
              "set_sim_IA_shape_convention(0, ...) to inject the raw "
              "distortion spectra.");
    exit(1);
  }
  free(SIM_ERMS);
  SIM_ERMS = (double*) malloc(SIM_NSN*sizeof(double));
  for (int i=0;i<SIM_NSN;i++) {
    const double z = SIM_Z[i];
    if (given[i]) {
      for (int j=0;j<m;j++) if (zs[j] == z) { SIM_ERMS[i] = es[j]; break; }
    } else if (m == 1 || z <= zs[0]) {
      SIM_ERMS[i] = es[0];
    } else if (z >= zs[m-1]) {
      SIM_ERMS[i] = es[m-1];
    } else {
      int j = 0; while (zs[j+1] < z) j++;
      const double w = (z - zs[j])/(zs[j+1] - zs[j]);
      SIM_ERMS[i] = es[j] + w*(es[j+1] - es[j]);
    }
  }
  log_info("sim_IA: shear conversion, e_rms from %s "
           "(P_dE/(2R), P_EE/(2R)^2, R = 1 - e_rms^2):", SIM_E_SRC);
  for (int i=0;i<SIM_NSN;i++) {
    const double R = 1.0 - SIM_ERMS[i]*SIM_ERMS[i];
    log_info("sim_IA:   sn %3d  z %5.2f  e_rms %.4f  1/(2R) %.4f  %s",
             SIM_SN[i], SIM_Z[i], SIM_ERMS[i], 0.5/R,
             given[i] ? "given" : "interpolated");
  }
  free(zs); free(es); free(given);
  SIM_E_RESOLVED = 1;
}

// 1/(2R) at configured row iz.
static double shear_factor_row(int iz) {
  if (!SIM_SHEAR) return 1.0;
  resolve_erms();
  return 0.5/(1.0 - SIM_ERMS[iz]*SIM_ERMS[iz]);
}

double sim_IA_shear_factor(double z) {
  if (!SIM_SHEAR) return 1.0;
  if (SIM_NSN < 1) {
    log_fatal("sim_IA: set_sim_IA_config() not called"); exit(1);
  }
  if (!SIM_E_RESOLVED) {
    #pragma omp critical (sim_IA_load)
    {
      resolve_erms();
    }
  }
  double e;
  if (SIM_NSN == 1 || z <= SIM_Z[0]) e = SIM_ERMS[0];
  else if (z >= SIM_Z[SIM_NSN-1])    e = SIM_ERMS[SIM_NSN-1];
  else {
    int j = 0; while (SIM_Z[j+1] < z) j++;
    const double w = (z - SIM_Z[j])/(SIM_Z[j+1] - SIM_Z[j]);
    e = SIM_ERMS[j] + w*(SIM_ERMS[j+1] - SIM_ERMS[j]);
  }
  return 0.5/(1.0 - e*e);
}

// ---------------------------------------------------------------------------
// Load one spectrum type across all snapshots into a (z, lnk) grid, applying
// the mu = 0 and shear conversions row by row. Rows outside
// [SIM_KMIN_HMPC, SIM_KMAX_HMPC] are trimmed. Every file of one type (and
// every multipole of one snapshot) must share the same k-grid.
// ---------------------------------------------------------------------------
static void load_spec(sim_spec_t t) {
  sim_spec* s = &SIM[t];
  if (s->loaded) return;
  if (SIM_NSN < 2) {
    log_fatal("sim_IA: need >=2 snapshots to interpolate in z (have %d). "
              "Call set_sim_IA_config() first.", SIM_NSN);
    exit(1);
  }

  const int is_shape = (SIM_NSHAPE[t] > 0);
  const int los = is_shape ? SIM_LOS : SIM_IA_LOS_NONE;

  const double kmin = SIM_KMIN_HMPC;
  const double kmax = (SIM_KMAX_HMPC > 0.0) ? SIM_KMAX_HMPC : INFINITY;

  double* kref = NULL;      // trimmed, ascending
  int nk_ref = 0;
  double* Pflat = NULL;     // nz*nk_ref
  const int nz = SIM_NSN;

  for (int iz=0; iz<nz; iz++) {
    char path[1536];
    spec_path(path, sizeof(path), t, 0, SIM_SN[iz]);
    double *k=NULL, *P=NULL; int nk=0;
    read_sorted_or_die(path, &k, &P, &nk);

    // ---- line of sight: monopole -> mu = 0 ----
    if (los == SIM_IA_LOS_LA) {
      for (int i=0;i<nk;i++) P[i] *= SIM_LA_LOS[t];
    }
    else if (los == SIM_IA_LOS_MULTIPOLE) {
      // P(mu=0) = sum_l (2l+1) M_l L_l(0),  L_2(0) = -1/2,  L_4(0) = 3/8
      static const int    ells[2] = { 2, 4 };
      static const double coef[2] = { 5.0*(-0.5), 9.0*(3.0/8.0) };
      for (int il=0; il<2; il++) {
        char pl[1536];
        spec_path(pl, sizeof(pl), t, ells[il], SIM_SN[iz]);
        double *kl=NULL, *Pl=NULL; int nkl=0;
        read_sorted_or_die(pl, &kl, &Pl, &nkl);
        if (nkl != nk) {
          log_fatal("sim_IA: '%s' has %d k-rows, monopole has %d", pl, nkl, nk);
          exit(1);
        }
        for (int i=0;i<nk;i++) {
          if (fabs(kl[i] - k[i]) > 1e-6*fabs(k[i])) {
            log_fatal("sim_IA: k-grid of '%s' differs from the monopole "
                      "(row %d: %g vs %g)", pl, i, kl[i], k[i]);
            exit(1);
          }
          P[i] += coef[il]*Pl[i];
        }
        free(kl); free(Pl);
      }
    }

    // ---- shape convention: distortion -> shear, 1/(2R) per shape leg ----
    if (is_shape && SIM_SHEAR) {
      const double f  = shear_factor_row(iz);
      const double fn = pow(f, SIM_NSHAPE[t]);
      for (int i=0;i<nk;i++) P[i] *= fn;
    }

    // ---- trim to [kmin,kmax] ----
    int lo=0; while (lo<nk && k[lo] <  kmin) lo++;
    int hi=nk-1; while (hi>=0 && k[hi] >  kmax) hi--;
    const int nkeep = hi - lo + 1;
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

  static const char* LOSNAME[3] = { "monopole", "LA mu=0 factor",
                                    "mu=0 from l=0,2,4" };
  log_info("sim_IA: loaded %s (nz=%d, nk=%d, z in [%.3f,%.3f], "
           "k in [%.4g,%.4g] h/Mpc); %s; %s.",
           SIM_TAG[t], nz, nk_ref, s->z[0], s->z[nz-1],
           exp(s->lnk[0]), exp(s->lnk[nk_ref-1]),
           is_shape ? LOSNAME[los] : "no conversion (not a shape spectrum)",
           (is_shape && SIM_SHEAR) ? "shear units" :
           (is_shape ? "DISTORTION units" : "-"));
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
    int lo=0, hi=n-1;
    while (hi - lo > 1) { int mid=(lo+hi)>>1; if (g[mid] <= x) lo=mid; else hi=mid; }
    i0 = lo;
  }
  *w = (x - g[i0]) / (g[i0+1] - g[i0]);   // may be <0 or >1 -> extrapolation
  return i0;
}

// ---------------------------------------------------------------------------
// Low-k power-law extrapolation for a single z-row.
// P(k) = P_edge * (k/k_edge)^s, s = slope of ln|P| between the two lowest
// nodes, sign of the edge node carried. The slope is floored at 0: going to
// k < k_edge, a negative slope would make |P| GROW without bound as k -> 0
// (and raising k_min lengthens that lever arm). Worst case is flat. If the
// two nodes differ in sign (or one is zero), flat clamp at the edge value.
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
// ---------------------------------------------------------------------------
static double P_sim(sim_spec_t t, double k_code, double a) {
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

  // Below the grid in k: per-row power law, then linear in z.
  if (lnk < s->lnk[0]) {
    double wz;
    const int iz = bracket(s->z, s->nz, z, &wz);
    const double Prow0 = sim_lowk_powerlaw(s, iz,     lnk);
    const double Prow1 = sim_lowk_powerlaw(s, iz + 1, lnk);
    const double P_phys = Prow0 + wz*(Prow1 - Prow0);
    return P_phys / (R*R*R);
  }

  // Bilinear in (z, ln k); weights outside [0,1] => linear extrapolation.
  double wz, wk;
  const int iz = bracket(s->z,   s->nz, z,   &wz);
  const int ik = bracket(s->lnk, s->nk, lnk, &wk);

  const double P00 = s->P[(size_t)iz    *nk + ik    ];
  const double P01 = s->P[(size_t)iz    *nk + ik + 1];
  const double P10 = s->P[(size_t)(iz+1)*nk + ik    ];
  const double P11 = s->P[(size_t)(iz+1)*nk + ik + 1];

  const double Pk0 = P00 + wk*(P01 - P00);
  const double Pk1 = P10 + wk*(P11 - P10);
  const double P_phys = Pk0 + wz*(Pk1 - Pk0);

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