#ifndef __COSMOLIKE_COSMO2D_SCUTS_H
#define __COSMOLIKE_COSMO2D_SCUTS_H
#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// Naming convention:
// ----------------------------------------------------------------------------
// c = cluster position ("c" as in "cluster")
// g = galaxy positions ("g" as in "galaxy")
// k = kappa CMB ("k" as in "kappa")
// s = kappa from source galaxies ("s" as in "shear")

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// DERIVATIVE: dlnX/dlnk: important to determine scale cuts (2011.06469 eq 17)
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

// dlnxi_pm/dlnk at one wavenumber for every tomographic pair and angular
// bin: Legendre sums of the batched dC_ss/dlnk over every integer
// multipole, normalized by xi_pm(theta).
double** dlnxi_dlnk_pm_tomo_nointerp(
    const double k    // wavenumber in (Mpc/h)^-1
  ); // returns [2][NSIZE*Ntheta] (caller frees): [0] = xi+, [1] = xi-

// Cached (in ln k) version of the function above; what the RF_xi
// integrals read. A k outside the Ntable.dCX_dlnk grid returns 0.
double dlnxi_dlnk_pm_tomo(
    const double k,   // wavenumber in (Mpc/h)^-1
    const int pm,     // 1 = xi_+, 0 = xi_-
    const int nt,     // angular bin index (0..Ntheta-1)
    const int ni,     // first source redshift bin
    const int nj      // second source redshift bin
  );

// dlnw_ks/dlnk at one wavenumber for every source bin and angular bin:
// Legendre sums of the batched dC_ks/dlnk over every integer multipole,
// weighted by the CMB filter (beam + optional HEALPix window) and
// normalized by w_ks(theta).
double* dlnw_ks_dlnk_tomo_nointerp(
    const double k    // wavenumber in (Mpc/h)^-1
  ); // returns [NSIZE*Ntheta] (caller frees), NSIZE = shear_nbin

// Cached (in ln k) version of the function above; what the RF_w_ks
// integrals read. A k outside the Ntable.dCX_dlnk grid returns 0.
double dlnw_ks_dlnk_tomo(
    const double k,   // wavenumber in (Mpc/h)^-1
    const int nt,     // angular bin index (0..Ntheta-1)
    const int ni      // source redshift bin
  );

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// Cached dC_ss/dlnk (2011.06469 eq 17) on a (ln k, ln l) table filled by
// dC_ss_dlnk_tomo_limber_work; a (k, l) outside the table returns 0.
double dC_ss_dlnk_tomo_limber(
    const double k,   // wavenumber in (Mpc/h)^-1
    const double l,   // multipole
    const int ni,     // first source redshift bin
    const int nj,     // second source redshift bin
    const int EE      // 1 = E-mode, 0 = B-mode
  );

// Cached dlnC_ss/dlnk = (dC_ss/dlnk)/C_ss on the same grid, filled by the
// normalized mode of dC_ss_dlnk_tomo_limber_work.
double dlnC_ss_dlnk_tomo_limber(
    const double k,   // wavenumber in (Mpc/h)^-1
    const double l,   // multipole
    const int ni,     // first source redshift bin
    const int nj,     // second source redshift bin
    const int EE      // 1 = E-mode, 0 = B-mode
  );

// Cached dC_ks/dlnk (2011.06469 eq 17) on a (ln k, ln l) table filled by
// dC_ks_dlnk_tomo_limber_work; a (k, l) outside the table returns 0.
double dC_ks_dlnk_tomo_limber(
    const double k,   // wavenumber in (Mpc/h)^-1
    const double l,   // multipole
    const int ni      // source redshift bin
  );

// Cached dlnC_ks/dlnk = (dC_ks/dlnk)/C_ks on the same grid, filled by the
// normalized mode of dC_ks_dlnk_tomo_limber_work.
double dlnC_ks_dlnk_tomo_limber(
    const double k,   // wavenumber in (Mpc/h)^-1
    const double l,   // multipole
    const int ni      // source redshift bin
  );

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// Batch real-space response function RF(kmax, theta) =
// int_{-infty}^{ln kmax} |dlnxi_pm/dlnk| normalized by the full-line
// integral (2011.06469 eq 17), for every tomographic pair and angular
// bin on a ln kmax grid.
void RF_xi_tomo_limber_work(
    const double* lnkmaxx, // ln kmax values (length nkmax), k in (Mpc/h)^-1
    const int nkmax,       // number of ln kmax values
    const int NSIZE,       // number of tomo shear power spectra
    double**** table       // output [2][NSIZE][nkmax][Ntheta]: xi+ and xi-
  );

// Batch real-space response function RF(kmax, theta) =
// int_{-infty}^{ln kmax} |dlnw_ks/dlnk| normalized by the full-line
// integral (2011.06469 eq 17), for every source bin and angular bin on
// a ln kmax grid.
void RF_w_ks_tomo_limber_work(
    const double* lnkmaxx, // ln kmax values (length nkmax), k in (Mpc/h)^-1
    const int nkmax,       // number of ln kmax values
    const int NSIZE,       // number of source tomographic bins (= shear_nbin)
    double*** table        // output [NSIZE][nkmax][Ntheta]
  );

// Batch response function RF(kmax, l) = int_{-infty}^{ln kmax} |dlnC/dlnk|
// normalized by the full-line integral (2011.06469 eq 17), for every
// tomographic pair on a (ln kmax, ell) grid.
void RF_C_ss_tomo_limber_work(
    const double* lnkmaxx, // ln kmax values (length nkmax), k in (Mpc/h)^-1
    const int nkmax,       // number of ln kmax values
    const double* lx,      // multipole values (length nl)
    const int nl,          // number of multipole values
    const int NSIZE,       // number of tomo shear power spectra
    double**** table       // output [2][NSIZE][nkmax][nl]: EE and BB
  );

// Batch response function RF(kmax, l) = int_{-infty}^{ln kmax} |dlnC/dlnk|
// normalized by the full-line integral (2011.06469 eq 17), for every
// source bin on a (ln kmax, ell) grid.
void RF_C_ks_tomo_limber_work(
    const double* lnkmaxx, // ln kmax values (length nkmax), k in (Mpc/h)^-1
    const int nkmax,       // number of ln kmax values
    const double* lx,      // multipole values (length nl)
    const int nl,          // number of multipole values
    const int NSIZE,       // number of source tomographic bins (= shear_nbin)
    double*** table        // output [NSIZE][nkmax][nl]
  );

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// UV-FRACTION SCALE CUT: FOURIER C_gg (arXiv:2603.13031)
// The split C_l = C_l^IR + C_l^UV, with C_l^IR computed from the 3D
// power spectrum times a step keeping k <= kmax/D(z) (kmax in h/Mpc at
// z = 0, D the linear growth factor with D(0) = 1); a band is kept
// while |C_l^UV|/(|C_l^UV| + |C_l^IR|) < alpha
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// The Fourier gg spectra of one evaluation (autos, then the cross pairs
// when installed) along the data-vector path, with the k window on for
// kmax > 0 (the IR part for keep_ir = 1, the UV part for keep_ir = 0)
// or off for kmax <= 0 (the full spectra).
void C_gg_kwindow_tomo_ells(
    const double kmax,    // cut at z = 0 in h/Mpc; <= 0 = no window
    const int keep_ir,    // 1 = IR part, 0 = UV part
    const double* ells,   // band centers (length nell)
    const int nell,       // number of band centers
    double** out,         // output autos [clustering_Npowerspectra][nell]
    double** outx         // output pairs [gg_cross_npairs][nell] (or NULL)
  );

// The UV-fraction scale cuts of every gg spectrum (s = autos, then
// pairs: the data-vector order) on an ascending kmax grid: the full
// spectra, their IR parts, the UV fractions and the number of bands
// kept (a prefix in l, shortened where needed so the cut tightens as
// kmax decreases).
void gg_scale_cuts_tomo_ells_work(
    const double* kmax,   // cuts at z = 0 in h/Mpc, ascending (nkmax)
    const int nkmax,      // number of cuts
    const double alpha,   // allowed UV fraction
    const double* ells,   // band centers, ascending (nell)
    const int nell,       // number of band centers
    double** full,        // output [nspec][nell]
    double*** ir,         // output [nkmax][nspec][nell]
    double*** uvfrac,     // output [nkmax][nspec][nell]
    int** nkeep           // output [nkmax][nspec]
  );

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD