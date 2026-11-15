#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "cfastpt/utils_cfastpt.h"
#include "cfastpt/utils_complex_cfastpt.h"
#include "cfastpt/cfastpt.h"
#include "basics.h"
#include "cosmo3D.h"
#include "pt_cfastpt.h"
#include "structs.h"

#include "log.c/src/log.h"

// ---------------------------------------------------------------------------
// fpt_regrid: move FFTLog outputs from the internal convolution grid onto
// the output table.
//
// Ntable.FPT_internal_accuracy_boost sets the internal (convolution) grid
// as a fraction of the output grid. The FFTLog convolutions are smooth in
// ln k, so their cost can shrink while the output table - whose density is
// what the likelihood's linear interpolation actually resolves - keeps the
// validated point count (same split as the python FAST-PT theory block and
// bfmt: coarse internal grid, dense output table).
//
// Both grids are uniform in ln k with the same lower endpoint
// (k[i] = kmin * exp(i * dlnk)), so the source interval of an output point
// is one divide - no binary search: r = j*ddst/dsrc. Natural cubic
// coefficients come from spline_coeffs_uniform and are evaluated in Horner
// form, the same scheme as the n(z) fine-grid upsampling in
// redshift_spline.c. The VALUE is splined, not its log: several FPT
// spectra cross zero. The internal grid stops one internal spacing below
// k_max, so the few output points past its last node evaluate clamped at
// that node (nothing physical is read that close to k_max).
// ---------------------------------------------------------------------------
static void fpt_regrid(
    double** restrict src,   // spectra on the internal grid [.][Nsrc]
    const long Nsrc,         // internal (convolution) grid points
    double** restrict dst,   // output tables [.][Ndst]
    const long Ndst,         // output grid points
    const int* restrict rows,   // which table rows to move
    const int nrows,            // number of entries in rows[] (5 or 10)
    double** restrict coeff,    // caller-owned spline scratch [nrows][Nsrc]
    const double lnk_span       // ln(k_max / k_min), shared by both grids
  )
{
  const double dsrc = lnk_span / Nsrc;
  const double ddst = lnk_span / Ndst;

  // phase 1, serial: the tridiagonal solves (nrows is 5 or 10; the work
  // is microseconds and each row's solve is a sequential recursion)
  for (int m = 0; m < nrows; m++) {
    spline_coeffs_uniform(src[rows[m]], (int) Nsrc, dsrc, coeff[m]);
  }

  // phase 2: Horner evaluation. nrows alone (5 or 10) underfills the
  // thread team, so collapse(2) spreads nrows * Ndst iterations.
  // Local restrict pointers: without them the compiler cannot prove
  // that the rows reached through the double indirection never overlap,
  // so it re-reads y[] and c[] from memory after every write to out[]
  // instead of keeping them in registers.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int m = 0; m < nrows; m++) {
    for (long j = 0; j < Ndst; j++) {
      const double* restrict y = src[rows[m]];
      const double* restrict c = coeff[m];
      double* restrict out = dst[rows[m]];
      double r = j * ddst / dsrc;
      if (r > Nsrc - 1) r = Nsrc - 1;             // top clamp (see above)
      const long i = (r < Nsrc - 2) ? (long) r : Nsrc - 2;
      const double delx = (r - i) * dsrc;
      const double dy = y[i+1] - y[i];
      const double b = dy/dsrc - dsrc*(c[i+1] + 2.0*c[i])/3.0;
      const double d = (c[i+1] - c[i])/(3.0*dsrc);
      out[j] = y[i] + delx*(b + delx*(c[i] + delx*d));
    }
  }
}

// ---------------------------------------------------------------------------
// get_FPT_bias: compute all 5 nonlinear galaxy bias spectra
// ---------------------------------------------------------------------------
//
// Computes the one-loop bias power spectra needed for nonlinear galaxy
// bias modeling: Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2. These are the
// correlators between the density field (d), its square (d2), the tidal
// field squared (s2), and the third-order bias operator (p3).
//
// One J_abl call evaluates all 13 FFTLog terms, and the coefficients of
// the table at the top of the function combine them into the spectra:
//   Pd1d2(k) = sum_{terms 0,1,2} coeff[i] * Fy[i](k)
//   Pd2d2(k) = coeff[3] * Fy[3](k)
//   Pd1s2(k) = sum_{terms 4..8} coeff[i] * Fy[i](k)
//   ...etc
//
// FPTbias.tab layout (8 rows x FPTbias.N columns; k in H0/c, spectra in
// (c/H0)^3 at a = 1, which the Limber integrands of cosmo2D.c scale by
// D(a)^4):
//   [0] Pd1d2   - density * density-squared correlator
//   [1] Pd2d2   - density-squared auto-correlator
//   [2] Pd1s2   - density * tidal-squared correlator
//   [3] Pd2s2   - density-squared * tidal-squared correlator
//   [4] Ps2s2   - tidal-squared auto-correlator
//   [5] Pd1p3   - density * third-order (from precomputed table, not FAST-PT)
//   [6] k       - wavenumber grid (log-spaced from k_min to k_max)
//   [7] P_lin   - linear power spectrum at z=0 (a=1)
//
// Caching: rows 0-4 depend only on P_lin(k, a = 1), but they are
// recomputed whenever cosmology.random changes (each cosmology setter of
// generic_interface.cpp draws a new value when its input changes), when
// Ntable.random changes (grid and accuracy settings), and when
// FPTbias.tab is not the table this function allocated. Row 5 comes from
// the fixed table tab_d1d3 of pt_cfastpt.h: it does not change with
// cosmology.
//
// Two producers share FPTbias.tab. With nuisance.IA_code = 1, set_bias_PS
// (generic_interface.cpp) frees this function's table and installs the
// Python FAST-PT one (its own k grid and size), and cosmo2D.c stops
// calling this function. When it runs again (IA_code = 0), it finds the
// foreign table, frees it and rebuilds its own grid and spectra.
// ---------------------------------------------------------------------------

// 1 while FPTIA.tab / FPTbias.tab hold the Python tables set_IA_PS /
// set_bias_PS installed; a C-FAST-PT rebuild below clears them
static int FPTIA_tab_from_python = 0;
static int FPTbias_tab_from_python = 0;

void set_FPT_IA_from_python(void)
{
  FPTIA_tab_from_python = 1;
}

void set_FPT_bias_from_python(void)
{
  FPTbias_tab_from_python = 1;
}

void FPT_IA_tables(void)
{
  if (0 == nuisance.IA_code) {
    get_FPT_IA();
  }
  else if (0 == FPTIA_tab_from_python) {
    log_fatal("the Python FAST-PT IA table is not installed or was replaced "
              "(a covariance call rebuilds C-FAST-PT): call set_IA_PS again");
    exit(1);
  }
}

void FPT_bias_tables(void)
{
  if (0 == nuisance.IA_code) {
    get_FPT_bias();
  }
  else if (0 == FPTbias_tab_from_python) {
    log_fatal("the Python FAST-PT bias table is not installed or was "
              "replaced (a covariance call rebuilds C-FAST-PT): call "
              "set_bias_PS again");
    exit(1);
  }
}

void get_FPT_bias(void)
{
  const int NTAB = 8; // FPTbias.tab rows: 5 spectra, Pd1p3, k, P_lin
  static double** Fy = NULL;           // per-term FFTLog outputs [NTERMS][N_int]
  static double** regrid_coeff = NULL;  // fpt_regrid spline scratch [5][N_int]
  static fastpt_config config;          // FFTLog padding/window setup
  // 13 terms -> 5 output spectra
  // ---------------------------------------------------------------
  // idx | alpha | beta | ell | output     | coeff
  // ----|-------|------|-----|------------|------------------
  //  0  |   0   |   0  |  0  | Pd1d2 [0]  | 2*(17/21)
  //  1  |   0   |   0  |  2  | Pd1d2 [0]  | 2*(4/21)
  //  2  |   1   |  -1  |  1  | Pd1d2 [0]  | 2
  //  3  |   0   |   0  |  0  | Pd2d2 [1]  | 2
  //  4  |   0   |   0  |  0  | Pd1s2 [2]  | 2*(8/315)
  //  5  |   0   |   0  |  2  | Pd1s2 [2]  | 2*(254/441)
  //  6  |   0   |   0  |  4  | Pd1s2 [2]  | 2*(16/245)
  //  7  |   1   |  -1  |  1  | Pd1s2 [2]  | 2*(4/15)
  //  8  |   1   |  -1  |  3  | Pd1s2 [2]  | 2*(2/5)
  //  9  |   0   |   0  |  2  | Pd2s2 [3]  | 2*(2/3)
  // 10  |   0   |   0  |  0  | Ps2s2 [4]  | 2*(4/45)
  // 11  |   0   |   0  |  2  | Ps2s2 [4]  | 2*(8/63)
  // 12  |   0   |   0  |  4  | Ps2s2 [4]  | 2*(8/35)
  // ---------------------------------------------------------------
  const int NTERMS = 13;
  const int NOUT   = 5;
  const int OUT_D2D2 = 1; // used for sigma4 below
 
  const int alpha_tab[13] = {0,0,1, 0, 0,0,0,1,1, 0, 0,0,0};
  const int beta_tab[13] = {0,0,-1, 0, 0,0,0,-1,-1, 0, 0,0,0};
  const int ell_tab[13] = {0,2,1, 0, 0,2,4,1,3, 2, 0,2,4};
  const int out_idx[13] = {0,0,0, 1, 2,2,2,2,2, 3, 4,4,4};
 
  const double coeff[13] = {
    2.*(17./21.), 2.*(4./21.), 2.,
    2.,
    2.*(8./315.), 2.*(254./441.), 2.*(16./245.), 2.*(4./15.), 2.*(2./5.),
    2.*(2./3.),
    2.*(4./45.), 2.*(8./63.), 2.*(8./35.)
  };
  static uint64_t cache[MAX_SIZE_ARRAYS];
  // FPTbias.tab as allocated here. set_bias_PS (python FAST-PT, IA_code = 1)
  // replaces the table with its own grid; the next call from this function
  // must then rebuild the grid and recompute, even if Ntable.random and
  // cosmology.random did not change.
  static double** owned_tab = NULL;
  const int new_tables = (fdiff2(cache[1], Ntable.random) ||
                          FPTbias.tab != owned_tab);
 
  if (new_tables)
  {
    FPTbias_tab_from_python = 0; // the table below replaces any Python one
    FPTbias.krange[RANGE_MIN]    = 0.05;
    FPTbias.krange[RANGE_MAX]    = 1.0e+6;
    FPTbias.k_cutoff = 1.0e+4;
    FPTbias.N        = 1100 + 200 * Ntable.FPTboost;
    // internal (convolution) grid, rounded up to even (the FFTLog engine
    // requires it); == N makes tab_int alias tab (exact legacy path)
    FPTbias.N_int = ((int) ceil(FPTbias.N
                                * Ntable.FPT_internal_accuracy_boost) + 1) & ~1;
    FPTbias.sigma4   = 0.0;
    if (FPTbias.tab_int != NULL && FPTbias.tab_int != FPTbias.tab)
    {
      free(FPTbias.tab_int);
    }
    if (FPTbias.tab != NULL)
    {
      free(FPTbias.tab);
    }
    FPTbias.tab = (double**) malloc2d(NTAB, FPTbias.N);
    FPTbias.tab_int = (FPTbias.N_int == FPTbias.N) ? FPTbias.tab
        : (double**) malloc2d(NTAB, FPTbias.N_int);
    owned_tab = FPTbias.tab;
    if (regrid_coeff != NULL)
    {
      free(regrid_coeff);
    }
    regrid_coeff = (FPTbias.N_int == FPTbias.N) ? NULL
        : (double**) malloc2d(5, FPTbias.N_int); // one row per regridded spectrum
  }
 
  if (new_tables ||
      fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], Ntable.random))
  {
    // Internal (convolution) grid: FPT_internal_accuracy_boost = 1 keeps
    // it equal to the output grid, tab_int aliases the output tables and this is
    // the exact legacy path; any other value runs the convolutions on
    // ceil(boost * N) points and fpt_regrid moves the results onto the
    // output grid, whose density is what the likelihood's interpolation
    // resolves (see fpt_regrid above).
    const long Nout = FPTbias.N;
    const long Nk = FPTbias.N_int;
    double** tab_int = FPTbias.tab_int; // aliases FPTbias.tab when Nk == Nout
    const double lnkmin = log(FPTbias.krange[RANGE_MIN]);
    const double lnspan = log(FPTbias.krange[RANGE_MAX]) - lnkmin;
    const double dlogk = lnspan / Nk;
 
    // --- build k grid and linear P(k) ---
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Nk; i++) {
      tab_int[6][i] = exp(lnkmin + i * dlogk);
      tab_int[7][i] = p_lin(tab_int[6][i], 1.0);
    }
 
    // single J_abl call with all 13 bias terms; the work array and the
    // FFTLog configuration depend only on the grids, so they persist
    // across cosmologies and rebuild only when Ntable changes (the
    // cache[1] stamp below still holds the pre-rebuild value here)
    if (NULL == Fy || fdiff2(cache[1], Ntable.random)) {
      if (Fy != NULL) {
        free(Fy);
      }
      Fy = (double**) malloc2d(NTERMS, Nk);
      config.nu             = -2.;
      config.c_window_width = 0.65;
      // N_pad and N_extrap are point counts, but what they protect is a
      // LENGTH in ln k (span = count * dlnk). The FFTLog convolution is
      // circular: N_pad zero-pads so the slowly decaying kernel tails do
      // not wrap around and alias into the physical k range, and N_extrap
      // extends P(k) as power laws so the array does not end in a sharp
      // step whose Gibbs ringing (only partly damped by c_window_width)
      // would contaminate the spectra near the k-range edges. Scaling the
      // counts with Nk/Nout keeps all three spans exactly as validated on
      // the legacy single grid, for any internal-grid density.
      config.N_pad          = (int) ceil(1500. * Nk / (double) Nout);
      config.N_extrap_low   = (int) ceil(500. * Nk / (double) Nout);
      config.N_extrap_high  = (int) ceil(500. * Nk / (double) Nout);
    }
 
    int alpha_ar[13];
    int beta_ar[13];
    int ell_ar[13];
    int isP13type_ar[13];
    for (int i = 0; i < NTERMS; i++) {
      alpha_ar[i]     = alpha_tab[i];
      beta_ar[i]      = beta_tab[i];
      ell_ar[i]       = ell_tab[i];
      isP13type_ar[i] = 0;
    }
 
    J_abl(tab_int[6], tab_int[7], Nk, alpha_ar, beta_ar, ell_ar, isP13type_ar,
          NTERMS, &config, Fy);
 
    // --- accumulate 13 Fy terms into 5 bias spectra ---
    // parallel over the outputs, like get_FPT_IA's group accumulation:
    // each thread owns its output row (no reduction needed), and the
    // per-term parallel regions of the legacy code collapse into one
    #pragma omp parallel for schedule(static)
    for (int out = 0; out < NOUT; out++) {
      memset(tab_int[out], 0, sizeof(double) * Nk);
      for (int i = 0; i < NTERMS; i++) {
        if (out_idx[i] != out) {
          continue;
        }
        const double c = coeff[i];
        for (long j = 0; j < Nk; j++) {
          tab_int[out][j] += c * Fy[i][j];
        }
      }
    }
 
    FPTbias.sigma4 = tab_int[OUT_D2D2][0] / 2.; // both grids share k[0] = k_min

    if (tab_int != FPTbias.tab) {
      const double dl = lnspan / Nout;
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < Nout; i++) {
        FPTbias.tab[6][i] = exp(lnkmin + i * dl);
        FPTbias.tab[7][i] = p_lin(FPTbias.tab[6][i], 1.0);
      }
      const int rows[5] = {0, 1, 2, 3, 4};
      fpt_regrid(tab_int, Nk, FPTbias.tab, Nout, rows, 5, regrid_coeff, lnspan);
    }

    // Pd1p3: interpolated from a precomputed (grid-independent) table, so
    // it is filled directly on the output grid
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Nout; i++)
    {
      const double lnk = log(FPTbias.tab[6][i]);
      FPTbias.tab[5][i] =
        (lnk < tab_d1d3_lnkmin || lnk > tab_d1d3_lnkmax) ? 0.0 :
        interpol1d(tab_d1d3, tab_d1d3_Nk, tab_d1d3_lnkmin,
                   tab_d1d3_lnkmax, tab_d1d3_dlnk, lnk);
    }
 
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// NONLINEAR KAISER: THE ONE-LOOP VELOCITY TEMPLATES
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The angular bracket B(r) of a one-loop propagator (P13-type) term,
// r = q/k: which = 0 for the density, P13_dd, which = 1 for the
// velocity divergence, P13_tt (the forms of Carlson, White and
// Padmanabhan 2009, arXiv:0905.0479):
//
//   B(r) = c_0/r^2 + c_1 + c_2 r^2 + c_3 r^4
//          + 3 E(r)/r^3 ln|(1 + r)/(1 - r)|,
//   E(r) = sum_p e_p r^(2p), p = 0..4 (the polynomial (r^2 - 1)^3 times
//          7 r^2 + 2 for the density, r^2 + 2 for the velocity).
//
// Far from r = 1 the polynomial and the logarithm cancel to many digits,
// so below R_LO and above R_HI the series of the logarithm,
// ln|(1+r)/(1-r)| = 2 sum_n x^(2n+1)/(2n+1) with x = r or 1/r, is summed
// term by term instead; the cancelling powers are dropped exactly.
// ---------------------------------------------------------------------------
static double fpt_kaiser_bracket(const double r, const int which)
{
  static const double cpoly[2][4] = {{12.0, -158.0, 100.0, -42.0},
                                     {12.0, -82.0, 4.0, -6.0}};
  static const double epoly[2][5] = {{-2.0, -1.0, 15.0, -19.0, 7.0},
                                     {-2.0, 5.0, -3.0, -1.0, 1.0}};
  const double R_LO = 0.1;  // series in r^2 below
  const double R_HI = 10.0; // series in 1/r^2 above
  const int NSERIES = 16;   // terms: the last one is below 1e-30 of B
  const double* c = cpoly[which];
  const double* e = epoly[which];

  if (r < R_LO) {
    // coefficient of r^(2m): c_(m+1) + 6 sum_p e_p/(2(m - p) + 3)
    double sum = 0.0;
    double r2m = 1.0;
    for (int m=0; m<NSERIES; m++) {
      double s = (m <= 2) ? c[m + 1] : 0.0;
      for (int p=0; p<=4 && p<=m+1; p++) {
        s += 6.0*e[p]/(2.0*(m - p) + 3.0);
      }
      sum += s*r2m;
      r2m *= r*r;
    }
    return sum;
  }
  if (r > R_HI) {
    // coefficient of r^(-2m): c_(1-m) + 6 sum_p e_p/(2(m + p) - 3)
    const double u2 = 1.0/(r*r);
    double sum = 0.0;
    double u2m = 1.0;
    for (int m=0; m<NSERIES; m++) {
      double t = (m <= 1) ? c[1 - m] : 0.0;
      const int p0 = (m < 2) ? 2 - m : 0;
      for (int p=p0; p<=4; p++) {
        t += 6.0*e[p]/(2.0*(m + p) - 3.0);
      }
      sum += t*u2m;
      u2m *= u2;
    }
    return sum;
  }
  const double r2 = r*r;
  double val = c[0]/r2 + c[1] + c[2]*r2 + c[3]*r2*r2;
  // at r = 1 the logarithm diverges, but (r^2 - 1)^3 times it vanishes
  if (r != 1.0) {
    const double E = e[0] + r2*(e[1] + r2*(e[2] + r2*(e[3] + r2*e[4])));
    val += 3.0*E*log(fabs((1.0 + r)/(1.0 - r)))/(r2*r);
  }
  return val;
}

// ---------------------------------------------------------------------------
// The four one-loop velocity templates on a uniform ln k grid, by direct
// quadrature (the k unit is the caller's: P in that unit^-3).
//
//   out[0] = P22_dt(k) = 2 int_q F2 G2 P(q) P(|k - q|)
//   out[1] = P22_tt(k) = 2 int_q G2^2 P(q) P(|k - q|)
//   out[2] = A_d(k)    = P13_dd/2 = <delta_1 delta_3>
//   out[3] = A_t(k)    = P13_tt/2 = <theta_1 theta_3>
//
// with F2, G2 the second-order density and velocity kernels (Einstein-de
// Sitter, Bernardeau et al. 2002, arXiv:astro-ph/0112551) and theta the
// velocity divergence in units of f.
//
// Both integrals run over q = k_j, the table's own nodes (r = q/k on the
// same ln grid), with the trapezoid rule in ln q.
//
// P22: with x the cosine between k and q and y = |k - q|/k,
//
//   P22_XY = k^3/(392 pi^2) int dr int_-1^1 dx P(kr) P(ky) a_X a_Y/y^4,
//   a_d = 3r + 7x - 10 r x^2 = 14 r y^2 F2,  a_t = 7x - r - 6 r x^2,
//
// symmetric under q <-> k - q: the half domain y >= r (x <= 1/(2r)),
// doubled, keeps y >= 1/2, away from the kernels' pole at q = k. The x
// integral uses NX Gauss-Legendre nodes; P(ky) is a cubic spline of ln P
// in ln k (power laws past the table ends).
//
// P13: P13_X = pref_X k^3 P(k)/(4 pi^2) int dr P(kr) B_X(r), pref 1/252
// (density) and 1/84 (velocity), B_X of fpt_kaiser_bracket.
//
// Arguments:
//   N, lnk0, dlnk - the grid, ln k_i = lnk0 + i dlnk
//   P             - P_lin at the nodes [N]
//   NX, xg, wg    - Gauss-Legendre nodes and weights on [-1, 1]
//   out           - output [4][N]
// ---------------------------------------------------------------------------
static void fpt_kaiser_compute(
    const int N,
    const double lnk0,
    const double dlnk,
    const double* P,
    const int NX,
    const double* xg,
    const double* wg,
    double** out
  )
{
  // ln P and its spline coefficients, for the off-node reads P(k y)
  double* lnP = (double*) malloc1d(N);
  double* cP  = (double*) malloc1d(N);
  for (int i=0; i<N; i++) {
    lnP[i] = log(P[i]);
  }
  spline_coeffs_uniform(lnP, N, dlnk, cP);
  const double slope_lo = (lnP[1] - lnP[0])/dlnk;
  const double slope_hi = (lnP[N-1] - lnP[N-2])/dlnk;

  // the brackets at every node ratio r = k_j/k_i = exp(m dlnk),
  // m = j - i in [-(N-1), N-1], stored at index m + N - 1
  const int NR = 2*N - 1;
  double* B13 = (double*) malloc1d(2*NR);
  for (int m=0; m<NR; m++) {
    const double r = exp((m - (N - 1))*dlnk);
    B13[m]      = fpt_kaiser_bracket(r, 0);
    B13[NR + m] = fpt_kaiser_bracket(r, 1);
  }

  const double PI2 = M_PI*M_PI;
  #pragma omp parallel for schedule(dynamic, 4)
  for (int i=0; i<N; i++) {
    const double lnk = lnk0 + i*dlnk;
    double s13d  = 0.0;
    double s13t  = 0.0;
    double s22dt = 0.0;
    double s22tt = 0.0;
    for (int j=0; j<N; j++) {
      const double w   = (0 == j || N - 1 == j) ? 0.5*dlnk : dlnk;
      const double r   = exp((j - i)*dlnk);
      const double wrP = w*r*P[j]; // dr P(kr) = dln r r P(k_j)
      s13d += wrP*B13[j - i + N - 1];
      s13t += wrP*B13[NR + j - i + N - 1];

      // the x integral on the half domain x <= 1/(2r)
      const double xmax = (r <= 0.5) ? 1.0 : 0.5/r;
      const double half = 0.5*(xmax + 1.0);
      double idt = 0.0;
      double itt = 0.0;
      for (int g=0; g<NX; g++) {
        const double x  = -1.0 + half*(xg[g] + 1.0);
        const double y2 = 1.0 + r*r - 2.0*r*x;
        // P(k y) from the spline of ln P
        const double lnky = lnk + 0.5*log(y2);
        const double t = (lnky - lnk0)/dlnk;
        double lnPky;
        if (t < 0.0) {
          lnPky = lnP[0] + slope_lo*(lnky - lnk0);
        }
        else if (t > N - 1) {
          lnPky = lnP[N-1] + slope_hi*(lnky - (lnk0 + (N - 1)*dlnk));
        }
        else {
          const int n = (t < N - 2) ? (int) t : N - 2;
          const double dx = (t - n)*dlnk;
          const double b  = (lnP[n+1] - lnP[n])/dlnk
                            - dlnk*(cP[n+1] + 2.0*cP[n])/3.0;
          const double d  = (cP[n+1] - cP[n])/(3.0*dlnk);
          lnPky = lnP[n] + dx*(b + dx*(cP[n] + dx*d));
        }
        const double ad = 3.0*r + 7.0*x - 10.0*r*x*x;
        const double at = 7.0*x - r - 6.0*r*x*x;
        const double base = half*wg[g]*exp(lnPky)/(y2*y2);
        idt += base*ad*at;
        itt += base*at*at;
      }
      s22dt += wrP*idt;
      s22tt += wrP*itt;
    }
    const double k3 = exp(3.0*lnk);
    // the factor 2 doubles the half domain
    out[0][i] = 2.0*k3/(392.0*PI2)*s22dt;
    out[1][i] = 2.0*k3/(392.0*PI2)*s22tt;
    out[2][i] = 0.5*k3*P[i]/(252.0*4.0*PI2)*s13d;
    out[3][i] = 0.5*k3*P[i]/(84.0*4.0*PI2)*s13t;
  }
  free(lnP);
  free(cP);
  free(B13);
}

// ---------------------------------------------------------------------------
// get_FPT_kaiser: the one-loop velocity templates of the nonlinear Kaiser
// terms (cosmo2D.c use_nonlinear_kaiser_GG)
// ---------------------------------------------------------------------------
//
// Theta = -div v/(a H) is the velocity divergence, with linear part
// f delta_L, and theta = Theta/f. Standard perturbation theory with the
// Einstein-de Sitter kernels and growth D^n at order n gives, for a
// density leg at z1 and a velocity leg at z2 (D_a = D(z_a), f_a = f(z_a)),
//
//   P_dTheta(k; z1, z2) = f2 [D1 D2 P_L + D1^2 D2^2 P22_dt
//                             + D1 D2^3 A_t + D1^3 D2 A_d],
//   P_ThetaTheta        = f1 f2 [D1 D2 P_L + D1^2 D2^2 P22_tt
//                                + (D1^3 D2 + D1 D2^3) A_t],
//
// each term a product of one function of z1 and one of z2, so the gg
// engines project it with extra FFTLog rows. The templates are those of
// fpt_kaiser_compute (above), at a = 1.
//
// FPTkaiser.tab layout (6 rows x FPTkaiser.N; k in H0/c, spectra in
// (c/H0)^3 at a = 1):
//   [0] P22_dt, [1] P22_tt, [2] A_d, [3] A_t, [4] k, [5] P_lin
//
// The quadrature runs on an internal grid of N_int nodes, 100 per decade
// (doubled per Ntable.FPTboost level), and fpt_regrid moves the four
// templates onto the output table, four times denser, which the engines
// read linearly. The k range covers every wavenumber the gg engines read
// (the templates are zero outside it).
//
// Caching: rebuilt when cosmology.random or Ntable.random changes.
// Called serially, before any parallel read (the engines' warm-ups).
// ---------------------------------------------------------------------------
void get_FPT_kaiser(void)
{
  const int NTAB = 6;     // FPTkaiser.tab rows: 4 templates, k, P_lin
  const int NTEMPL = 4;   // the templates, rows 0..3
  const int NX = 32;      // Gauss-Legendre nodes of the P22 x integral
  const int NPD = 680;    // internal nodes over the 6.8 decades: 100/decade
  const int UPSAMPLE = 4; // output nodes per internal node
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** regrid_coeff = NULL; // fpt_regrid scratch [4][N_int]
  static double* xg = NULL;            // Gauss-Legendre nodes on [-1, 1]
  static double* wg = NULL;            // and their weights

  const int new_tables = (NULL == FPTkaiser.tab ||
                          fdiff2(cache[1], Ntable.random));
  if (new_tables) {
    FPTkaiser.krange[RANGE_MIN] = 0.05;   // H0/c: 1.7e-5 h/Mpc
    FPTkaiser.krange[RANGE_MAX] = 3.0e+5; // H0/c: 100 h/Mpc
    FPTkaiser.k_cutoff = 0.0;
    FPTkaiser.sigma4   = 0.0;
    FPTkaiser.N_int = NPD*(1 + Ntable.FPTboost);
    FPTkaiser.N     = UPSAMPLE*FPTkaiser.N_int;
    if (FPTkaiser.tab != NULL) {
      free(FPTkaiser.tab);
    }
    if (FPTkaiser.tab_int != NULL) {
      free(FPTkaiser.tab_int);
    }
    if (regrid_coeff != NULL) {
      free(regrid_coeff);
    }
    FPTkaiser.tab     = (double**) malloc2d(NTAB, FPTkaiser.N);
    FPTkaiser.tab_int = (double**) malloc2d(NTAB, FPTkaiser.N_int);
    regrid_coeff      = (double**) malloc2d(NTEMPL, FPTkaiser.N_int);
    if (NULL == xg) {
      xg = (double*) malloc1d(NX);
      wg = (double*) malloc1d(NX);
      gsl_integration_glfixed_table* w = malloc_gslint_glfixed(NX);
      for (int g=0; g<NX; g++) {
        gsl_integration_glfixed_point(-1.0, 1.0, g, &xg[g], &wg[g], w);
      }
      gsl_integration_glfixed_table_free(w);
    }
  }

  if (new_tables ||
      fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], Ntable.random))
  {
    const long Nk   = FPTkaiser.N_int;
    const long Nout = FPTkaiser.N;
    double** tab_int = FPTkaiser.tab_int;
    const double lnkmin = log(FPTkaiser.krange[RANGE_MIN]);
    const double lnspan = log(FPTkaiser.krange[RANGE_MAX]) - lnkmin;
    const double dlnk   = lnspan/Nk; // fpt_regrid's grid convention

    (void) p_lin(FPTkaiser.krange[RANGE_MIN], 1.0); // serial warm-up
    #pragma omp parallel for schedule(static)
    for (long i=0; i<Nk; i++) {
      tab_int[4][i] = exp(lnkmin + i*dlnk);
      tab_int[5][i] = p_lin(tab_int[4][i], 1.0);
    }

    fpt_kaiser_compute((int) Nk, lnkmin, dlnk, tab_int[5], NX, xg, wg,
                       tab_int);

    #pragma omp parallel for schedule(static)
    for (long i=0; i<Nout; i++) {
      FPTkaiser.tab[4][i] = exp(lnkmin + i*lnspan/Nout);
      FPTkaiser.tab[5][i] = p_lin(FPTkaiser.tab[4][i], 1.0);
    }
    const int rows[4] = {0, 1, 2, 3};
    fpt_regrid(tab_int, Nk, FPTkaiser.tab, Nout, rows, NTEMPL, regrid_coeff,
               lnspan);

    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// INTRINSIC ALIGNMENT
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

enum {
  COL_ID_ALPHA = 0,
  COL_ID_BETA  = 1,
  COL_ID_L1    = 2,
  COL_ID_L2    = 3,
  COL_ID_L     = 4,
  NCOLS        = 5
};

// ---------------------------------------------------------------------------
// Nmax_from_terms: Upper bound on J_table output rows.
//
// Each input term (l1, l2, l) expands into a triple loop over (J1, J2, Jk),
// where each index is constrained by the triangle inequality for angular
// momentum coupling:
//   J1 ranges over |l  - l2| .. l  + l2  (couples l  and l2)
//   J2 ranges over |l1 - l|  .. l1 + l   (couples l1 and l)
//   Jk ranges over |l1 - l2| .. l1 + l2  (couples l1 and l2)
//
// The number of allowed values for each is (max - min + 1). Their product
// bounds how many (J1, J2, Jk) combinations each term can produce.
// Summing over all input terms gives the total upper bound.
// ---------------------------------------------------------------------------
static int Nmax_from_terms(int N, int (*terms)[NCOLS]) {
  int Nmax = 0;
  for (int i = 0; i < N; i++) {
    // assumed terms layout: {alpha, beta, l1, l2, l} → columns 0..4
    const int l1 = terms[i][COL_ID_L1];
    const int l2 = terms[i][COL_ID_L2];
    const int l  = terms[i][COL_ID_L];

    const int nJ1 = l  + l2 - abs(l  - l2) + 1;  // count of allowed J1
    const int nJ2 = l1 + l  - abs(l1 - l)  + 1;  // count of allowed J2
    const int nJk = l1 + l2 - abs(l1 - l2) + 1;  // count of allowed Jk

    Nmax += nJ1 * nJ2 * nJk;
  }
  return Nmax;
}

// ---------------------------------------------------------------------------
// get_FPT_IA: compute the ten one-loop TATT spectra of FPTIA.tab
// ---------------------------------------------------------------------------
//
// The one-loop intrinsic-alignment power spectra of the TATT model
// (Blazek et al. 2019, arXiv:1708.09247), evaluated with FAST-PT from the
// linear power spectrum at a = 1. The 184 FFTLog terms of eight J_table
// expansions go through one J_abJ1J2Jk call; two P_13-like terms are
// direct convolutions.
//
// FPTIA.tab layout (12 rows x FPTIA.N columns; k in H0/c, spectra in
// (c/H0)^3 at a = 1, which cosmo2D.c and covariances/ia_cov.c scale by
// D(a)^4, each term being quadratic in P_lin):
//   [0] tt_E,    [1] tt_B     - (s s) auto power, E and B
//   [2] ta_dE1,  [3] ta_dE2   - delta with (delta s): A_{0|0E}, C_{0|0E}
//   [4] ta_0E0E, [5] ta_0B0B  - (delta s) auto power, E and B
//   [6] mix_A,   [7] mix_B    - delta with (s s): A_{0|E2}, B_{0|E2}
//   [8] mix_DEE, [9] mix_DBB  - (delta s) with (s s), E and B
//   [10] k                    - wavenumber grid, uniform in ln k
//   [11] P_lin                - linear power spectrum at a = 1
// with s the tidal field, (s s) its trace-free square and (delta s) the
// density-weighted tidal field (term names of Blazek et al. 2019).
//
// Cache invalidation: the spectra depend only on P_lin(k, a = 1), but
// they are recomputed whenever cosmology.random changes (each cosmology
// setter of generic_interface.cpp draws a new value when its input
// changes: parameters, distances, growth, linear or nonlinear power),
// when Ntable.random changes (grids, FAST-PT accuracy boosts), and when
// FPTIA.tab is not the table this function allocated.
//
// Two producers share FPTIA.tab. With nuisance.IA_code = 1, set_IA_PS
// (generic_interface.cpp) frees the table and installs the Python
// FAST-PT one, with its own k grid and size; cosmo2D.c then stops
// calling this function. When it runs again (cosmo2D.c with IA_code = 0,
// or covariances/ia_cov.c, which calls it whatever IA_code says), it
// finds the foreign table, frees it and rebuilds its own grid and
// spectra.
// ---------------------------------------------------------------------------
void get_FPT_IA(void)
{
  const int NTAB = 12; // FPTIA.tab rows: 10 spectra, k, P_lin
  static double *Fy_flat = NULL;        // contiguous per-term FFTLog outputs
  static double **Fy_ptrs = NULL;       // row pointers into Fy_flat
  static double** regrid_coeff = NULL;  // fpt_regrid spline scratch [10][N_int]
  static fastpt_config fpt_config;      // FFTLog padding/window setup (nu = 0)
  static uint64_t cache[MAX_SIZE_ARRAYS];
  // FPTIA.tab as allocated here. set_IA_PS (python FAST-PT, IA_code = 1)
  // replaces the table with its own grid; the next call from this function
  // must then rebuild the grid and recompute (see get_FPT_bias).
  static double** owned_tab = NULL;
  const int new_tables = (fdiff2(cache[1], Ntable.random) ||
                          FPTIA.tab != owned_tab);
  if (new_tables)
  {
    FPTIA_tab_from_python = 0; // the table below replaces any Python one
    FPTIA.krange[RANGE_MIN]    = 0.05;
    FPTIA.krange[RANGE_MAX]    = 1.0e+6;
    FPTIA.k_cutoff = 1.0e+4;
    FPTIA.sigma4   = 0.0;
    FPTIA.N        = 1100 + 200 * Ntable.FPTboost;
    // internal (convolution) grid, rounded up to even (the FFTLog engine
    // requires it); == N makes tab_int alias tab (exact legacy path)
    FPTIA.N_int = ((int) ceil(FPTIA.N
                              * Ntable.FPT_internal_accuracy_boost) + 1) & ~1;
    if (FPTIA.tab_int != NULL && FPTIA.tab_int != FPTIA.tab) {
      free(FPTIA.tab_int);
    }
    if (FPTIA.tab != NULL) {
      free(FPTIA.tab);
    }
    FPTIA.tab = (double**) malloc2d(NTAB, FPTIA.N);
    FPTIA.tab_int = (FPTIA.N_int == FPTIA.N) ? FPTIA.tab
        : (double**) malloc2d(NTAB, FPTIA.N_int);
    owned_tab = FPTIA.tab;
    if (regrid_coeff != NULL) {
      free(regrid_coeff);
    }
    regrid_coeff = (FPTIA.N_int == FPTIA.N) ? NULL
        : (double**) malloc2d(10, FPTIA.N_int); // one row per regridded spectrum
  }
  if (new_tables || 
      fdiff2(cache[0], cosmology.random) || 
      fdiff2(cache[1], Ntable.random))
  {
    // Internal (convolution) grid - same scheme as get_FPT_bias: boost = 1
    // aliases the output tables (exact legacy path); otherwise the ~184
    // FFTLog terms and the two direct convolutions run on ceil(boost * N)
    // points and fpt_regrid moves the ten spectra onto the output grid.
    const long Nout = FPTIA.N;
    const long Nk = FPTIA.N_int;
    double** tab_int = FPTIA.tab_int; // aliases FPTIA.tab when Nk == Nout
    double *k   = tab_int[10];
    double *Pin = tab_int[11];

    double lim[3];
    lim[0] = log(FPTIA.krange[RANGE_MIN]);
    lim[1] = log(FPTIA.krange[RANGE_MAX]);
    lim[2] = (lim[1] - lim[0]) / Nk;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Nk; i++) {
      k[i]   = exp(lim[0] + i*lim[2]);
      Pin[i] = p_lin(k[i], 1.0);
    }

    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    // Instead of calling IA_tt, IA_ta, IA_mix separately (3 calls to
    // J_abJ1J2Jk with 3 separate extrapolation setups, 3 separate
    // alpha/beta FFT precomputations, and 3 separate OpenMP regions),
    // this function expands all J_tables here, concatenates them into one
    // big array, and makes a single call. OpenMP then gets ~184 terms in
    // one parallel region for much better load balancing.
    //
    // Output mapping (8 groups → 8 output arrays):
    //   Group 0: IA_tt  E-mode   → FPTIA.tab[0]
    //   Group 1: IA_tt  B-mode   → FPTIA.tab[1]
    //   Group 2: IA_ta  deltaE1  → FPTIA.tab[2]
    //   Group 3: IA_ta  0E0E     → FPTIA.tab[4]
    //   Group 4: IA_ta  0B0B     → FPTIA.tab[5]
    //   Group 5: IA_mix A        → FPTIA.tab[6]
    //   Group 6: IA_mix D_EE     → FPTIA.tab[8]
    //   Group 7: IA_mix D_BB     → FPTIA.tab[9]
    //
    // The two direct convolution terms (IA_ta deltaE2, IA_mix B) don't
    // use J_abJ1J2Jk and are computed separately below.
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // Group 0: IA_tt E-mode, remind: {alpha, beta, l1, l2, l}
    // -------------------------------------------------------------------------
    int terms_tt_E[][NCOLS] = {
      {0, 0, 0, 0, 0},
      {0, 0, 2, 0, 0},
      {0, 0, 4, 0, 0},
      {0, 0, 2, 2, 0},
      {0, 0, 1, 1, 1},
      {0, 0, 3, 1, 1},
      {0, 0, 0, 0, 2},
      {0, 0, 2, 0, 2},
      {0, 0, 2, 2, 2},
      {0, 0, 1, 1, 3},
      {0, 0, 0, 0, 4},
    };

    double coeff_tt_E[] = {
      2*(16./81.),   2*(713./1134.), 2*(38./315.),
      2*(95./162.),  2*(-107./60.),  2*(-19./15.),
      2*(239./756.), 2*(11./9.),     2*(19./27.),
      2*(-7./10.),   2*(3./35.)
    };

    // -------------------------------------------------------------------------
    // Group 1: IA_tt B-mode, remind:  {alpha, beta, l1, l2, l}
    // -------------------------------------------------------------------------
    int terms_tt_B[][NCOLS] = {
      {0, 0, 0, 0, 0},
      {0, 0, 2, 0, 0},
      {0, 0, 4, 0, 0},
      {0, 0, 2, 2, 0},
      {0, 0, 1, 1, 1},
      {0, 0, 3, 1, 1},
      {0, 0, 0, 0, 2},
      {0, 0, 2, 0, 2},
      {0, 0, 2, 2, 2},
      {0, 0, 1, 1, 3},
    };

    double coeff_tt_B[] = {
      2*(-41./405.), 2*(-298./567.), 2*(-32./315.),
      2*(-40./81.),  2*(59./45.),    2*(16./15.),
      2*(-2./9.),    2*(-20./27.),   2*(-16./27.),
      2*(2./5.)
    };

    // -------------------------------------------------------------------------
    // Group 2: IA_ta deltaE1, remind:  {alpha, beta, l1, l2, l}
    // -------------------------------------------------------------------------
    int terms_ta_dE1[][NCOLS] = {
      { 0,  0, 0, 2, 0},
      { 0,  0, 0, 2, 2},
      { 1, -1, 0, 2, 1},
      {-1,  1, 0, 2, 1},
    };

    double coeff_ta_dE1[] = {2*(17./21.), 2*(4./21.), 1., 1.};

    // -------------------------------------------------------------------------
    // Group 3: IA_ta 0E0E, remind:  {alpha, beta, l1, l2, l}
    // -------------------------------------------------------------------------
    int terms_ta_0E0E[][NCOLS] = {
      {0, 0, 0, 0, 0},
      {0, 0, 2, 0, 0},
      {0, 0, 2, 2, 0},
      {0, 0, 0, 4, 0},
    };

    double coeff_ta_0E0E[] = {29./90., 5./63., 19./18., 19./35.};

    // -------------------------------------------------------------------------
    // Group 4: IA_ta 0B0B, remind:  {alpha, beta, l1, l2, l}
    // -------------------------------------------------------------------------
    int terms_ta_0B0B[][NCOLS] = {
      {0, 0, 0, 0, 0},
      {0, 0, 2, 0, 0},
      {0, 0, 2, 2, 0},
      {0, 0, 0, 4, 0},
      {0, 0, 1, 1, 1},
    };

    double coeff_ta_0B0B[] = {2./45., -44./63., -8./9., -16./35., 2.};

    // -------------------------------------------------------------------------
    // Group 5: IA_mix A, remind:  {alpha, beta, l1, l2, l}
    // -------------------------------------------------------------------------
    int terms_mix_A[][NCOLS] = {
      { 0,  0, 0, 0, 0},
      { 0,  0, 2, 0, 0},
      { 0,  0, 0, 0, 2},
      { 0,  0, 2, 0, 2},
      { 0,  0, 1, 1, 1},
      { 0,  0, 1, 1, 3},
      { 0,  0, 0, 0, 4},
      { 1, -1, 0, 0, 1},
      { 1, -1, 2, 0, 1},
      { 1, -1, 1, 1, 0},
      { 1, -1, 1, 1, 2},
      { 1, -1, 0, 2, 1},
      { 1, -1, 0, 0, 3},
    };

    double coeff_mix_A[] = {
      2*(-31./210.), 2*(-34./63.), 2*(-47./147.), 2*(-8./63.),
      2*(93./70.),   2*(6./35.),   2*(-8./245.),
      2*(-3./10.),   2*(-1./3.),   2*(1./2.),
      2*(1.),        2*(-1./3.),   2*(-1./5.)
    };

    // -------------------------------------------------------------------------
    // Group 6: IA_mix D_EE, remind:  {alpha, beta, l1, l2, l}
    // -------------------------------------------------------------------------
    int terms_mix_DEE[][NCOLS] = {
      {0, 0, 0, 0, 0},
      {0, 0, 2, 0, 0},
      {0, 0, 4, 0, 0},
      {0, 0, 0, 0, 2},
      {0, 0, 2, 0, 2},
      {0, 0, 1, 1, 1},
      {0, 0, 3, 1, 1},
      {0, 0, 2, 2, 0},
    };

    double coeff_mix_DEE[] = {
      2*(-43./540.), 2*(-167./756.), 2*(-19./105.), 2*(1./18.),
      2*(-7./18.),   2*(11./20.),    2*(19./20.),   2*(-19./54.)
    };

    // -------------------------------------------------------------------------
    // Group 7: IA_mix D_BB, remind:  {alpha, beta, l1, l2, l}
    // -------------------------------------------------------------------------
    int terms_mix_DBB[][NCOLS] = {
      {0, 0, 0, 0, 0},
      {0, 0, 2, 0, 0},
      {0, 0, 4, 0, 0},
      {0, 0, 0, 0, 2},
      {0, 0, 2, 0, 2},
      {0, 0, 1, 1, 1},
      {0, 0, 3, 1, 1},
      {0, 0, 2, 2, 0},
    };

    double coeff_mix_DBB[] = {
      2*(13./135.), 2*(86./189.), 2*(16./105.), 2*(2./9.),
      2*(4./9.),    2*(-13./15.), 2*(-4./5.),   2*(8./27.)
    };

    // -------------------------------------------------------------------------
    // Number of input terms (rows) in each group's terms table
    // -------------------------------------------------------------------------
    enum {
      ID_TT_E    = 0,
      ID_TT_B    = 1,
      ID_TA_DE1  = 2,
      ID_TA_0E0E = 3,
      ID_TA_0B0B = 4,
      ID_MIX_A   = 5,
      ID_MIX_DEE = 6,
      ID_MIX_DBB = 7,
      NGROUPS    = 8
    };

    int Nrows[NGROUPS];
    Nrows[ID_TT_E]    = (int)(sizeof(terms_tt_E)    / sizeof(terms_tt_E[0]));
    Nrows[ID_TT_B]    = (int)(sizeof(terms_tt_B)    / sizeof(terms_tt_B[0]));
    Nrows[ID_TA_DE1]  = (int)(sizeof(terms_ta_dE1)  / sizeof(terms_ta_dE1[0]));
    Nrows[ID_TA_0E0E] = (int)(sizeof(terms_ta_0E0E) / sizeof(terms_ta_0E0E[0]));
    Nrows[ID_TA_0B0B] = (int)(sizeof(terms_ta_0B0B) / sizeof(terms_ta_0B0B[0]));
    Nrows[ID_MIX_A]   = (int)(sizeof(terms_mix_A)   / sizeof(terms_mix_A[0]));
    Nrows[ID_MIX_DEE] = (int)(sizeof(terms_mix_DEE) / sizeof(terms_mix_DEE[0]));
    Nrows[ID_MIX_DBB] = (int)(sizeof(terms_mix_DBB) / sizeof(terms_mix_DBB[0]));

    // -------------------------------------------------------------------------
    // Upper bound on J_table output rows per input term. Each term has three
    // angular momentum couplings (J1, J2, Jk) constrained by the triangle
    // inequality: e.g., J1 ranges from |l-l2| to l+l2, giving l+l2-|l-l2|+1
    // allowed values. The product of the three ranges bounds the output count.
    // -------------------------------------------------------------------------
    int Nmax[NGROUPS];
    Nmax[ID_TT_E]    = Nmax_from_terms(Nrows[ID_TT_E],    terms_tt_E);
    Nmax[ID_TT_B]    = Nmax_from_terms(Nrows[ID_TT_B],    terms_tt_B);
    Nmax[ID_TA_DE1]  = Nmax_from_terms(Nrows[ID_TA_DE1],  terms_ta_dE1);
    Nmax[ID_TA_0E0E] = Nmax_from_terms(Nrows[ID_TA_0E0E], terms_ta_0E0E);
    Nmax[ID_TA_0B0B] = Nmax_from_terms(Nrows[ID_TA_0B0B], terms_ta_0B0B);
    Nmax[ID_MIX_A]   = Nmax_from_terms(Nrows[ID_MIX_A],   terms_mix_A);
    Nmax[ID_MIX_DEE] = Nmax_from_terms(Nrows[ID_MIX_DEE], terms_mix_DEE);
    Nmax[ID_MIX_DBB] = Nmax_from_terms(Nrows[ID_MIX_DBB], terms_mix_DBB);

    // -------------------------------------------------------------------------
    // VLAs for J_table outputs
    // -------------------------------------------------------------------------
    int jterms_tt_E[Nmax[ID_TT_E]][NCOLS];
    int jterms_tt_B[Nmax[ID_TT_B]][NCOLS];
    int jterms_ta_dE1[Nmax[ID_TA_DE1]][NCOLS];
    int jterms_ta_0E0E[Nmax[ID_TA_0E0E]][NCOLS];
    int jterms_ta_0B0B[Nmax[ID_TA_0B0B]][NCOLS];
    int jterms_mix_A[Nmax[ID_MIX_A]][NCOLS];
    int jterms_mix_DEE[Nmax[ID_MIX_DEE]][NCOLS];
    int jterms_mix_DBB[Nmax[ID_MIX_DBB]][NCOLS];

    double jcoeff_tt_E[Nmax[ID_TT_E]];
    double jcoeff_tt_B[Nmax[ID_TT_B]];
    double jcoeff_ta_dE1[Nmax[ID_TA_DE1]];
    double jcoeff_ta_0E0E[Nmax[ID_TA_0E0E]];
    double jcoeff_ta_0B0B[Nmax[ID_TA_0B0B]];
    double jcoeff_mix_A[Nmax[ID_MIX_A]];
    double jcoeff_mix_DEE[Nmax[ID_MIX_DEE]];
    double jcoeff_mix_DBB[Nmax[ID_MIX_DBB]];
    
    // -------------------------------------------------------------------------
    // Expand each group's input terms via J_table into (alpha, beta, J1, J2, Jk) rows.
    // Njterms[g] is the actual number of output rows for group g (≤ Nmax[g]).
    // -------------------------------------------------------------------------
    int Njterms[NGROUPS];
    Njterms[ID_TT_E] = J_table(NCOLS, Nrows[ID_TT_E], terms_tt_E, coeff_tt_E, jterms_tt_E, jcoeff_tt_E);
    Njterms[ID_TT_B] = J_table(NCOLS, Nrows[ID_TT_B], terms_tt_B, coeff_tt_B, jterms_tt_B, jcoeff_tt_B);
    Njterms[ID_TA_DE1] = J_table(NCOLS, Nrows[ID_TA_DE1], terms_ta_dE1, coeff_ta_dE1, jterms_ta_dE1, jcoeff_ta_dE1);
    Njterms[ID_TA_0E0E] = J_table(NCOLS, Nrows[ID_TA_0E0E], terms_ta_0E0E, coeff_ta_0E0E, jterms_ta_0E0E, jcoeff_ta_0E0E);
    Njterms[ID_TA_0B0B] = J_table(NCOLS, Nrows[ID_TA_0B0B], terms_ta_0B0B, coeff_ta_0B0B, jterms_ta_0B0B, jcoeff_ta_0B0B);
    Njterms[ID_MIX_A] = J_table(NCOLS, Nrows[ID_MIX_A], terms_mix_A, coeff_mix_A, jterms_mix_A, jcoeff_mix_A);
    Njterms[ID_MIX_DEE] = J_table(NCOLS, Nrows[ID_MIX_DEE], terms_mix_DEE, coeff_mix_DEE, jterms_mix_DEE, jcoeff_mix_DEE);
    Njterms[ID_MIX_DBB] = J_table(NCOLS, Nrows[ID_MIX_DBB], terms_mix_DBB, coeff_mix_DBB, jterms_mix_DBB, jcoeff_mix_DBB);

    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    // Concatenate all groups into single arrays
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------

    // Pointers to the per-group J_table output arrays
    int (*jterms_ptrs[])[NCOLS] = {
      jterms_tt_E,
      jterms_tt_B,
      jterms_ta_dE1,
      jterms_ta_0E0E,
      jterms_ta_0B0B,
      jterms_mix_A,
      jterms_mix_DEE,
      jterms_mix_DBB
    };
    double *jcoeff_ptrs[] = {
      jcoeff_tt_E,
      jcoeff_tt_B,
      jcoeff_ta_dE1,
      jcoeff_ta_0E0E,
      jcoeff_ta_0B0B,
      jcoeff_mix_A,
      jcoeff_mix_DEE,
      jcoeff_mix_DBB
    };

    // -----------------------------------------------------------------------
    // Concatenate all NGROUPS J_table outputs into flat arrays for a single
    // J_abJ1J2Jk call. Each group g contributes Njterms[g] rows starting
    // at offset starts[g]. The flat arrays (alpha_all, beta_all, J1_all,
    // J2_all, Jk_all, coeff_all) are indexed from 0 to Ntotal-1.
    //
    // Note: after J_table expansion, the column meaning changed from the
    // input layout {alpha, beta, l1, l2, l} to {alpha, beta, J1, J2, Jk}.
    // Columns 0-1 (alpha, beta) are copied through unchanged by J_table.
    // Columns 2-4 were overwritten: l1->J1, l2->J2, l->Jk.
    // -----------------------------------------------------------------------
    int Ntotal = 0;
    int starts[NGROUPS];
    for (int g = 0; g < NGROUPS; g++) {
      starts[g] = Ntotal;
      Ntotal += Njterms[g];
    }

    enum {
      JCOL_ID_ALPHA = 0,
      JCOL_ID_BETA  = 1,
      JCOL_ID_J1    = 2,
      JCOL_ID_J2    = 3,
      JCOL_ID_JK    = 4
    };

    int alpha_all[Ntotal];
    int beta_all[Ntotal];
    int J1_all[Ntotal];
    int J2_all[Ntotal];
    int Jk_all[Ntotal];
    double coeff_all[Ntotal];
    for (int g=0; g<NGROUPS; g++) {
      int off = starts[g];
      for (int i = 0; i < Njterms[g]; i++) {
        alpha_all[off+i] = jterms_ptrs[g][i][JCOL_ID_ALPHA];
        beta_all[off+i]  = jterms_ptrs[g][i][JCOL_ID_BETA];
        J1_all[off+i]    = jterms_ptrs[g][i][JCOL_ID_J1];
        J2_all[off+i]    = jterms_ptrs[g][i][JCOL_ID_J2];
        Jk_all[off+i]    = jterms_ptrs[g][i][JCOL_ID_JK];
        coeff_all[off+i] = jcoeff_ptrs[g][i];
      }
    }

    // -----------------------------------------------------------------------
    // Single J_abJ1J2Jk call with all ~184 terms.
    //
    // J_abJ1J2Jk expects double **Fy where Fy[i] points to an array of
    // Nk doubles for term i's output. We allocate one contiguous block
    // (Fy_flat) and set up an array of pointers (Fy_ptrs) into it:
    //
    //   Fy_flat:  [--- term 0 ---][--- term 1 ---]...[--- term Ntotal-1 ---]
    //              ^               ^                   ^
    //   Fy_ptrs:  [0]             [1]                 [Ntotal-1]
    //
    // After the J_abJ1J2Jk call, Fy_ptrs[i][j] = result for term i at
    // k-point j.
    // -----------------------------------------------------------------------
    // the work arrays and the FFTLog configuration depend only on the
    // grids (Ntotal is fixed by the hardcoded term tables), so they
    // persist across cosmologies and rebuild only when Ntable changes
    // (the cache[1] stamp below still holds the pre-rebuild value here)
    if (NULL == Fy_flat || fdiff2(cache[1], Ntable.random)) {
      if (Fy_flat != NULL) {
        free(Fy_ptrs);
        free(Fy_flat);
      }
      Fy_flat = (double*) malloc(sizeof(double) * Ntotal * Nk);
      if (NULL == Fy_flat) {
        log_fatal("malloc failed");
        exit(1);
      }
      Fy_ptrs = (double**) malloc(sizeof(double*) * Ntotal);
      if (NULL == Fy_ptrs) {
        log_fatal("malloc failed");
        exit(1);
      }
      for (int i = 0; i < Ntotal; i++) {
        Fy_ptrs[i] = Fy_flat + i * Nk;
      }
      fpt_config.c_window_width = 0.65;
      // N_pad and N_extrap are point counts, but what they protect is a
      // LENGTH in ln k (span = count * dlnk). The FFTLog convolution is
      // circular: N_pad zero-pads so the slowly decaying kernel tails do
      // not wrap around and alias into the physical k range, and N_extrap
      // extends P(k) as power laws so the array does not end in a sharp
      // step whose Gibbs ringing (only partly damped by c_window_width)
      // would contaminate the spectra near the k-range edges. Scaling the
      // counts with Nk/Nout keeps all three spans exactly as validated on
      // the legacy single grid, for any internal-grid density.
      fpt_config.N_pad          = (int) ceil(1500. * Nk / (double) Nout);
      fpt_config.N_extrap_low   = (int) ceil(500. * Nk / (double) Nout);
      fpt_config.N_extrap_high  = (int) ceil(500. * Nk / (double) Nout);
    }

    J_abJ1J2Jk(k,         // input k grid, length N
               Pin,       // input power spectrum P(k)
               Nk,        // number of input k points (before padding)
               alpha_all, // biasing exponent 1 per term: nu1 = -2 - alpha[i]
               beta_all,  // biasing exponent 2 per term: nu2 = -2 - beta[i]
               J1_all,    // angular momentum coupling 1 per term (indexes g_m cache)
               J2_all,    // angular momentum coupling 2 per term (indexes g_m cache)
               Jk_all,    // angular momentum coupling 3 per term (indexes g_m cache)
               Ntotal,    // number of terms to compute
               &fpt_config, // padding/windowing config (N_pad, c_window_width, etc.)
               Fy_ptrs);  // output: Fy[i][j] = result for term i at k-point j


    // -----------------------------------------------------------------------
    // Accumulate J_abJ1J2Jk results into the 8 FPTIA output arrays.
    //
    // Each group's result is a weighted sum over its terms:
    //   output[j] = sum_i  coeff_all[i] * Fy[i][j]
    //
    // The mapping from group ID to FPTIA.tab index:
    //   ID_TT_E    → FPTIA.tab[0]   (IA_tt E-mode)
    //   ID_TT_B    → FPTIA.tab[1]   (IA_tt B-mode)
    //   ID_TA_DE1  → FPTIA.tab[2]   (IA_ta deltaE1)
    //   ID_TA_0E0E → FPTIA.tab[4]   (IA_ta 0E0E)
    //   ID_TA_0B0B → FPTIA.tab[5]   (IA_ta 0B0B)
    //   ID_MIX_A   → FPTIA.tab[6]   (IA_mix A)
    //   ID_MIX_DEE → FPTIA.tab[8]   (IA_mix D_EE)
    //   ID_MIX_DBB → FPTIA.tab[9]   (IA_mix D_BB)
    //
    // Note: FPTIA.tab[3] (deltaE2) and FPTIA.tab[7] (mix B) are computed
    // separately below via direct convolution, not J_abJ1J2Jk.
    // -----------------------------------------------------------------------
    double *outputs[NGROUPS];
    outputs[ID_TT_E]    = tab_int[0];
    outputs[ID_TT_B]    = tab_int[1];
    outputs[ID_TA_DE1]  = tab_int[2];
    outputs[ID_TA_0E0E] = tab_int[4];
    outputs[ID_TA_0B0B] = tab_int[5];
    outputs[ID_MIX_A]   = tab_int[6];
    outputs[ID_MIX_DEE] = tab_int[8];
    outputs[ID_MIX_DBB] = tab_int[9];

    #pragma omp parallel for schedule(static)
    for (int g = 0; g < NGROUPS; g++) {
      // Zero the output array, then accumulate the weighted sum of all
      // terms belonging to this group (indices starts[g] .. starts[g]+Njterms[g]-1)
      memset(outputs[g], 0, sizeof(double) * Nk);
      for (int i = starts[g]; i < starts[g] + Njterms[g]; i++) {
        const double c = coeff_all[i];              // combined coefficient A*B
        const double *row = Fy_flat + i * Nk;       // term i's result array
        for (int j = 0; j < Nk; j++)
          outputs[g][j] += c * row[j];
      }
    }

    
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // Next section is gonna need the following definitions
    const double dL = log(k[1] / k[0]);  // log-spacing of k grid
    const long Ncut = floor(3. / dL);     // transition index from exact to asymptotic
  
    double* exps = malloc(sizeof(double) * (size_t)(2*Nk - 1));
    if (NULL == exps) {
      log_fatal("malloc failed"); exit(1);
    }
    for (int i = 0; i < 2*Nk-1; i++) {
      // Precompute r = k'/k ratios for all convolution offsets
      exps[i] = exp(-dL * (i - Nk + 1));
    }

    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // IA_ta deltaE2 term: direct convolution (not via J_abJ1J2Jk).
    //
    // Computes P_deltaE2(k) = 2 * k^3 / (896 * pi^2) * Pin(k) * [Pin ⊛ f](k) * dL
    //
    // This is C_{0|0E} of Blazek et al. 2019 (arXiv:1708.09247), the
    // density-weighted tidal-alignment part of the density-shape (GI)
    // correlator: a P_13-like integral, Pin(k) times a convolution of Pin
    // with an angular kernel. The leading 2 is its prefactor, which
    // FAST-PT's IA_ta also applies; the 1/(896 pi^2) comes from the
    // angular integration of the kernel.
    //
    // Array index i of f stands for s = dL*(i - Nk + 1) and for the ratio
    // r = exp(-s) = k'/k, so low indices hold r >> 1 and high indices
    // r << 1. The convolution kernel f(r) has three regimes:
    //   r >> 1 (index far below the midpoint): asymptotic expansion in
    //          negative powers of r
    //   r ~ 1  (near the midpoint): exact closed form with the
    //          log(|r-1|/(r+1)) term
    //   r << 1 (index far above the midpoint): asymptotic expansion in
    //          positive powers of r
    //
    // The cutoff Ncut = floor(3/dL) switches between the exact formula and
    // the asymptotic expansions near |s| = 3, as FAST-PT does. The exact
    // formula has a log singularity at r=1, so the midpoint f[Nk-1] is set
    // analytically.
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    {
      double* f = malloc(sizeof(double) * (size_t)(2*Nk - 1));
      if (NULL == f) {
        log_fatal("malloc failed"); exit(1);
      }

      int i;

      // Region 1: r >> 1 (asymptotic expansion for large r)
      for (i = 0; i < Nk-1-Ncut; i++) {
        double r = exps[i];
        double r2 = r*r, r4 = r2*r2, r6 = r4*r2, r8 = r4*r4, r10 = r8*r2;
        f[i] = r * (768./7 - 256/(7293.*r10) - 256/(3003.*r8)
                - 256/(1001.*r6) - 256/(231.*r4) - 256/(21.*r2));
      }

       // Region 2: r ~ 1, below midpoint (exact closed-form with log term)
      for ( ; i < Nk-1; i++) {
        double r = exps[i];
        double r2 = r*r, r3 = r2*r, r4 = r2*r2, r5 = r4*r, r6 = r4*r2, r7 = r6*r;
        f[i] = r * (30. + 146*r2 - 110*r4 + 30*r6
                + log(fabs(r-1.)/(r+1.)) * (15./r - 60.*r + 90*r3 - 60*r5 + 15*r7));
      }

      // Region 3: r ~ 1, above midpoint (same exact formula, mirrored)
      for (i = Nk; i < Nk-1+Ncut; i++) {
        double r = exps[i];
        double r2 = r*r, r3 = r2*r, r4 = r2*r2, r5 = r4*r, r6 = r4*r2, r7 = r6*r;
        f[i] = r * (30. + 146*r2 - 110*r4 + 30*r6
                + log(fabs(r-1.)/(r+1.)) * (15./r - 60.*r + 90*r3 - 60*r5 + 15*r7));
      }

       // Region 4: r << 1 (asymptotic expansion for small r)
      for ( ; i < 2*Nk-1; i++) {
        double r = exps[i];
        double r2 = r*r, r4 = r2*r2, r6 = r4*r2, r8 = r4*r4, r10 = r8*r2, r12 = r6*r6, r14 = r8*r6;
        f[i] = r * (256*r2 - 256*r4 + (768*r6)/7.
                - (256*r8)/21. - (256*r10)/231.
                - (256*r12)/1001. - (256*r14)/3003.);
      }

      // Midpoint: r = 1 exactly (analytic limit of the closed-form expression)
      f[Nk-1] = 96.;

      // Convolve Pin with the kernel f, then extract and normalize
      double* g = malloc(sizeof(double) * (size_t)(3*Nk - 2));
      if (NULL == g) {
        log_fatal("malloc failed"); exit(1);
      }

      fftconvolve_real(Pin, f, Nk, 2*Nk-1, g);
      
      // P_deltaE2(k) = 2 * k^3 / (896 * pi^2) * Pin(k) * [Pin ⊛ f](k) * dL
      for (i = 0; i < Nk; i++) {
        double ki3 = k[i] * k[i] * k[i];
        tab_int[3][i] = 2. * ki3 / (896.*M_PI*M_PI) * Pin[i] * g[Nk-1+i] * dL;
      }

      free(g);
      free(f);
    }

    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // IA_mix B term: direct convolution (not via J_abJ1J2Jk).
    //
    // Computes P_B(k) = 4 * k^3 / (2 * pi^2) * Pin(k) * [Pin ⊛ f](k) * dL
    //
    // This is B_{0|E2} of Blazek et al. 2019 (arXiv:1708.09247), the
    // tidal-torquing part of the density-shape (GI) correlator. "B" names
    // the term: it is an E-mode contribution, not a B mode. Like deltaE2
    // it is a P_13-like integral, Pin(k) times a convolution of Pin with
    // an angular kernel; the 4 is its prefactor, which FAST-PT's IA_mix
    // also applies to this output (P_Btype2).
    //
    // Same structure as the deltaE2 kernel but with a different convolution
    // kernel f(r), the same index-to-r map (low indices hold r >> 1, high
    // indices r << 1) and three regimes:
    //   r >> 1 (index far below the midpoint): asymptotic expansion in
    //          negative powers of r
    //   r ~ 1  (near the midpoint): exact closed form with the
    //          log(|r-1|/(r+1)) term and the (r^2-1)^4 factor from the
    //          angular integral
    //   r << 1 (index far above the midpoint): asymptotic expansion in
    //          positive powers of r
    //
    // The overall /2 factor in each region is part of the kernel normalization.
    //
    // r = exp(-dL*(i - Nk + 1)) maps array index i to the ratio k'/k.
    // Ncut = floor(3/dL) sets the transition between exact and asymptotic forms.
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    {
      double* f = malloc(sizeof(double) * (size_t)(2*Nk - 1));
      if (NULL == f) {
        log_fatal("malloc failed"); exit(1);
      }
      int i;

      // Region 1: r >> 1 (asymptotic expansion for large r)
      for (i = 0; i < Nk-1-Ncut; i++) {
        double r = exps[i];
        double r2 = r*r, r4 = r2*r2, r6 = r4*r2, r8 = r4*r4, r10 = r8*r2, r12 = r6*r6;
        f[i] = r * (-16./147 - 16/(415701.*r12) - 32/(357357.*r10) - 16/(63063.*r8)
                - 64/(63063.*r6) - 16/(1617.*r4) + 32/(441.*r2)) / 2.;
      }

      // Region 2: r ~ 1, below midpoint (exact closed-form with log term)
      // The (r^2-1)^4 factor arises from the angular integration of the
      // IA_mix kernel; rm1_4 = (r^2 - 1)^4 is precomputed to avoid pow().
      for ( ; i < Nk-1; i++) {
        double r = exps[i];
        double r2 = r*r, r3 = r2*r, r4 = r2*r2, r6 = r4*r2, r8 = r4*r4;
        double rm1 = r2 - 1.;
        double rm1_2 = rm1*rm1, rm1_4 = rm1_2*rm1_2;
        f[i] = r * ((2. * r * (225. - 600.*r2 + 1198.*r4 - 600.*r6 + 225.*r8)
                + 225. * rm1_4 * (r2 + 1.) * log(fabs(r-1.)/(r+1.))) / (20160.*r3)
                - 29./315.*r2) / 2.;
      }

       // Region 3: r ~ 1, above midpoint (same exact formula, mirrored)
      for (i = Nk; i < Nk-1+Ncut; i++) {
        double r = exps[i];
        double r2 = r*r, r3 = r2*r, r4 = r2*r2, r6 = r4*r2, r8 = r4*r4;
        double rm1 = r2 - 1.;
        double rm1_2 = rm1*rm1, rm1_4 = rm1_2*rm1_2;
        f[i] = r * ((2. * r * (225. - 600.*r2 + 1198.*r4 - 600.*r6 + 225.*r8)
                + 225. * rm1_4 * (r2 + 1.) * log(fabs(r-1.)/(r+1.))) / (20160.*r3)
                - 29./315.*r2) / 2.;
      }

      // Region 4: r << 1 (asymptotic expansion for small r)
      for ( ; i < 2*Nk-1; i++) {
        double r = exps[i];
        double r2 = r*r, r4 = r2*r2, r6 = r4*r2, r8 = r4*r4, r10 = r8*r2, r12 = r6*r6, r14 = r8*r6, r16 = r8*r8;
        f[i] = r * ((-16*r4)/147. + (32*r6)/441. - (16*r8)/1617.
                - (64*r10)/63063. - (16*r12)/63063. - (32*r14)/357357.
                - (16*r16)/415701.) / 2.;
      }

      // Midpoint: r = 1 exactly (analytic limit of the closed-form expression)
      f[Nk-1] = -1./42.;

      // Convolve Pin with the kernel f, then extract and normalize
      double* g = malloc(sizeof(double) * (size_t)(3*Nk - 2));
      if (NULL == g) {
        log_fatal("malloc failed"); exit(1);
      }
      
      fftconvolve_real(Pin, f, Nk, 2*Nk-1, g);
      
      // P_B(k) = 4 * k^3 / (2 * pi^2) * Pin(k) * [Pin ⊛ f](k) * dL
      // The 4 is the prefactor of B_{0|E2} (see the block header above).
      for (i = 0; i < Nk; i++) {
        double ki3 = k[i] * k[i] * k[i];
        tab_int[7][i] = 4. * ki3 / (2.*M_PI*M_PI) * Pin[i] * g[Nk-1+i] * dL;
      }

      free(g);
      free(f);
    }

    free(exps);

    if (tab_int != FPTIA.tab) {
      const double dl = (lim[1] - lim[0]) / Nout;
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < Nout; i++) {
        FPTIA.tab[10][i] = exp(lim[0] + i*dl);
        FPTIA.tab[11][i] = p_lin(FPTIA.tab[10][i], 1.0);
      }
      const int rows[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
      fpt_regrid(tab_int, Nk, FPTIA.tab, Nout, rows, 10, regrid_coeff,
                 lim[1] - lim[0]);
    }

    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
}

