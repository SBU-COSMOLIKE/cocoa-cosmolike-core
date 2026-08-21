#ifndef __COSMOLIKE_IA_H
#define __COSMOLIKE_IA_H
#ifdef __cplusplus
extern "C" {
#endif

#define IA_MODEL_NLA 0
#define IA_MODEL_TATT 1

// ---------------------------------------------------------------------------
// nuisance.IA_code -- selects WHICH ENGINE supplies the IA (and nonlinear
// galaxy bias) power spectra that get injected into the C(l) integrands.
//   IA_CODE_CFASTPT    : C implementation of FAST-PT (get_FPT_IA/get_FPT_bias)
//   IA_CODE_PYFASTPT   : Python FAST-PT, tables pushed in from the interface
//   IA_CODE_HALO_MODEL : halo.c -- returns the FULL P(k), not PT kernels
//   IA_CODE_SIM        : sim_IA.c -- FULL P(k,z) tabulated from a simulation
//                        (no nuisance params, no growth, evolution baked in)
// ---------------------------------------------------------------------------
#define IA_CODE_CFASTPT     0
#define IA_CODE_PYFASTPT    1
#define IA_CODE_HALO_MODEL  2
#define IA_CODE_SIM         3

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
