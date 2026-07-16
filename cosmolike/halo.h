#ifndef __COSMOLIKE_HALO_H
#define __COSMOLIKE_HALO_H
#ifdef __cplusplus
extern "C" {
#endif

// HALO BIAS OPTIONS ---------------------------
#define HALO_BIAS_TINKER_2010 0

// HMF OPTIONS ---------------------------------
#define HMF_TINKER_2010 0

// CONCENTRATION OPTIONS -----------------------
#define CONCENTRATION_BHATTACHARYA_2013 0

// HALO PROFILE OPTIONS OPTIONS -----------------------
#define HALO_PROFILE_NFW 0

// ---------------------------------------------------------------------------
// HALO-MODEL IA / GALAXY-BIAS PUBLIC INTERFACE
// ---------------------------------------------------------------------------
// These are the entry points the main cosmolike pipeline (cosmo2D.c) calls
// when nuisance.IA_code == IA_CODE_HALO_MODEL (== 2).
//
// UNITS CONTRACT (identical to Pdelta / p_lin in cosmo3D.h):
//   k   -- input wavenumber in inverse c/H0 units, i.e. k_code = k_phys*coverH0
//          with k_phys in h/Mpc and coverH0 = 2997.92458 Mpc/h.
//   a   -- scale factor.
//   ret -- power spectrum in (c/H0)^3 units, i.e. P_code = P_phys/coverH0^3
//          with P_phys in (Mpc/h)^3.
// Every routine below is built on Pdelta(), cosmology.rho_crit and
// cosmology.coverH0, so the code-unit convention is already internally
// consistent -- do NOT apply any extra coverH0 rescaling at the call site.
//
// IMPORTANT SEMANTIC DIFFERENCE vs FAST-PT:
//   FAST-PT returns *perturbative kernels* that the pipeline multiplies by
//   amplitudes (C1, C2, b_ta) and by P_delta. The halo model returns the
//   *FULL* power spectrum with amplitudes already folded in. Therefore the
//   pipeline must inject these terms directly and must NOT multiply them by
//   C1/C2/b_ta or by PK again.
// ---------------------------------------------------------------------------

// Full IA power spectra (1-halo + 2-halo), tabulated & interpolated.
// P_II  = intrinsic-intrinsic   (enters the WS1*WS2 term)
// P_dI  = density-intrinsic     (enters the WK*WS cross term)
double P_II_halo(const double k, const double a, const int ni);
double P_dI_halo(const double k, const double a, const int ni);

// Non-interpolated (direct) versions -- mainly for testing/validation.
double P_II_halo_nointerp(const double k, const double a, const int ni, const int init);
double P_dI_halo_nointerp(const double k, const double a, const int ni, const int init);

// Maps a source (shear) tomography bin to the clustering (lens) bin whose
// HOD/red-fraction parameters are used by the halo-model IA routines.
// See the caveat block in halo.c for why this indirection is required.
int halo_IA_lensbin_of_sourcebin(const int ns);

// In halo.h, alongside the other existing function declarations


double hm_funcs_nointerp(const int ni, const double a, const int func, const int init);
double test_u_ia_sat(const double k, const double m, const double a);
// Direct (uncached) satellite IA profile -- reference implementation used to
// validate the tabulated u_ia_sat(). Exposed for testing.
double u_ia_sat_nointerp(const double c, const double k, const double m,
                         const double a, const int init);

// ---------------------------------------------------------------------------
// SERIAL TABLE INITIALIZERS -- MUST be called outside any OpenMP region.
// u_ia_sat(), n_red_sat_bar() and b_red_cen() are called from inside the
// "#pragma omp parallel for" of P_II_halo/P_dI_halo. Their lookup tables are
// therefore built here, serially, and are strictly read-only thereafter.
// Building them lazily on first use inside the parallel region causes
// concurrent malloc + nested omp regions => segfault.
// P_II_halo/P_dI_halo already call these; anything else that drives the
// halo-model IA routines in parallel must call them too.
// ---------------------------------------------------------------------------
void u_ia_sat_init(void);
void n_red_sat_bar_init(void);
void b_red_cen_init(void);
double I_for_IA_nointerp(const double k, const double a, const int ni,
                         const int ia_func, const int init);
double n_red_sat_bar(const int ni, const double a);
double p_II_1h_nointerp(const double k, const double a, const int ni);
double int_for_bred_cen(double lnM, void* params);
double I_bred_cen_nointerp(const double a, const int ni, const int func, const int init);
double b_red_cen(const int ni, const double a);
double A_nla_cen(const int ni, const double a);
double p_II_2h_cen_nointerp(const double k, const double a, const int ni);
double p_dI_2h_cen_nointerp(const double k, const double a, const int ni);
double p_dI_1h_nointerp(const double k, const double a, const int ni);
double p_mm(const double k, const double a);

double p_gm(const double k, const double a, const int ni);

double p_gg(const double k, const double a, const int ni, const int nj);

double p_my(const double k, const double a);

double p_yy(const double k, const double a);
double ngal(const int ni, const double a);
double bgal(const int ni, const double a);
double G02_nointerp(double k, double a, int ni, const int init);
double GM02_nointerp(double k, double a, int ni, const int init);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
