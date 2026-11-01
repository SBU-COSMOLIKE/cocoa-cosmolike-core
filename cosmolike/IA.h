#ifndef __COSMOLIKE_IA_H
#define __COSMOLIKE_IA_H
#ifdef __cplusplus
extern "C" {
#endif

#define IA_MODEL_NLA 0
#define IA_MODEL_TATT 1

// nuisance.IA_code: which engine supplies the IA spectra.
//   0 = C FAST-PT, 1 = Python FAST-PT (one-loop kernels; init_IA)
//   3 = IA_CODE_SIM: full P_dE, P_EE tabulated from a simulation
//       (sim_IA.c; set by init_IA_sim, NLA only, Limber ss/gs only)
#define IA_CODE_SIM 3

#define NO_IA 0
#define IA_NLA_LF 1
#define IA_REDSHIFT_BINNING 2
#define IA_REDSHIFT_EVOLUTION 3

void IA_A1_Z1Z2(
    const double a, 
    const double growfac_a, 
    const int n1, 
    const int n2, 
    double res[2]
  );

double IA_A1_Z1(
    const double a, 
    const double growfac_a, 
    const int n1
  );

void IA_A2_Z1Z2(
    const double a, 
    const double growfac_a, 
    const int n1, 
    const int n2, 
    double res[2]
  );

double IA_A2_Z1(
    const double a, 
    const double growfac_a, 
    const int n1
  );

void IA_BTA_Z1Z2(
    const double a, 
    const double growfac_a, 
    const int n1, 
    const int n2, 
    double res[2]
  );

double IA_BTA_Z1(
    const double a, 
    const double growfac_a, 
    const int n1
  );

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
