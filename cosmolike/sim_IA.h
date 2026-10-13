// sim_IA.h -- see sim_IA.c for documentation.
#ifndef SIM_IA_H
#define SIM_IA_H

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Line-of-sight treatment (how the measured 3D spectra are turned into the
// k-perp, mu = 0 spectra that the Limber integrals need).
// ---------------------------------------------------------------------------
#define SIM_IA_LOS_NONE      0  // inject the monopole as is (old behavior)
#define SIM_IA_LOS_LA        1  // index-0 files only: dE x 3/2, EE x 15/8
                                //   (exact for the linear-alignment angular
                                //   structure (1-mu^2)^n; default)
#define SIM_IA_LOS_MULTIPOLE 2  // P(mu=0) = M0 - (5/2) M2 + (27/8) M4 from the
                                //   index-0, 2 and 4 files (model independent)
#define SIM_IA_LOS_HYBRID    3  // cross spectra (dE, dB): LA factor from the
                                //   index-0 file; auto spectra (EE, BB): mu=0
                                //   from the index-0, 2, 4 files (default)

// Configure once at startup: directory, snapshot list, matching redshifts,
// and the nfold value used in the filenames. (sn,z) pairs may be unsorted.
void set_sim_IA_config(const char* dir, int nsn, const int* sn,
                       const double* zlist, int nfold);

// Shape convention of the tables.
//   apply_shear : 1 -> convert the distortion chi to shear, dividing every
//                 IA shape leg by 2R, R = 1 - e_rms^2 (needs e_rms, below);
//                 0 -> use the tables as they are.
//   los_mode    : SIM_IA_LOS_NONE / _LA / _MULTIPOLE / _HYBRID.
// Default: apply_shear = 1, los_mode = SIM_IA_LOS_HYBRID.
void set_sim_IA_shape_convention(int apply_shear, int los_mode);

// Mesh-window correction for the shape AUTO spectra (EE, BB), which were
// painted with NGP on an ng^3 mesh of side boxsize/nfold. Each table row is
// divided by the angle-averaged NGP window <W^2>(k). The cross spectra (dE,
// dB) are left alone: their two windows (NGP shapes x the matter scheme)
// cancel at O(k^2 Delta^2). Default: apply = 1, boxsize = 205 Mpc/h (TNG300),
// ng = 1024. Call before the tables load (i.e. at setup).
void set_sim_IA_window(int apply, double boxsize_hMpc, int ng);

// Per-snapshot e_rms (per-component rms of the distortion, z projection).
// Replaces any previous list. Snapshots of the configured list that are not
// given here (or are given a value <= 0) are interpolated linearly in z.
void set_sim_IA_erms(int n, const int* sn, const double* erms);

// Same, read from the Galaxy_DS_Field files
//   <dir>/hinfo_<stem>_sn<sn>_nfold<nfold>.dat   (never *.dat.bak)
// using e_rms = (e1z_rms + e2z_rms)/2. Snapshots listed in `skip` are not
// read (they get interpolated). Call after set_sim_IA_config().
// Returns the number of files read.
int set_sim_IA_erms_from_hinfo(const char* dir, const char* stem,
                               int nskip, const int* skip);

// Initialize (call after set_sim_IA_config; spectra load lazily on first use).
// Also drops any tables already loaded, so call it again after changing the
// shape convention or e_rms.
void init_sim_IA(void);

// Shear conversion factor 1/(2R) at redshift z (1 if apply_shear = 0).
double sim_IA_shear_factor(double z);

// Evaluators: k_code is CosmoLike code-unit k (= ell/fK); a is scale factor.
// Return P in code units (c/H0)^3, matching Pdelta(k,a). The IA spectra
// (EE, BB, dE, dB) are returned in SHEAR units at mu = 0 per the convention
// set above; dd, hh, dh are returned unchanged (monopoles).
double P_sim_EE(double k_code, double a);
double P_sim_BB(double k_code, double a);
double P_sim_dE(double k_code, double a);
double P_sim_dB(double k_code, double a);
double P_sim_dd(double k_code, double a);
double P_sim_hh(double k_code, double a);
double P_sim_dh(double k_code, double a);

#ifdef __cplusplus
}
#endif

#endif