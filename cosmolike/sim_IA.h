// sim_IA.h -- see sim_IA.c for documentation.
#ifndef SIM_IA_H
#define SIM_IA_H

#ifdef __cplusplus
extern "C" {
#endif

// Configure once at startup: directory, snapshot list, matching redshifts,
// and the nfold value used in the filenames. (sn,z) pairs may be unsorted.
void set_sim_IA_config(const char* dir, int nsn, const int* sn,
                       const double* zlist, int nfold);

// Initialize (call after set_sim_IA_config; spectra load lazily on first use).
void init_sim_IA(void);

// Evaluators: k_code is CosmoLike code-unit k (= ell/fK); a is scale factor.
// Return P in code units (c/H0)^3, matching Pdelta(k,a).
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
