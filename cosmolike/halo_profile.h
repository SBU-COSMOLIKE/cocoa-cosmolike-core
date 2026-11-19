#ifndef __COSMOLIKE_HALO_PROFILE_H
#define __COSMOLIKE_HALO_PROFILE_H

#include <gsl/gsl_sf.h>
#include <stdint.h>
#include <stdlib.h>

#include "basics.h"

#include "log.c/src/log.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// The shared NFW profile table: one owner of the numerical machinery
// behind the truncated-NFW Fourier transform (halo.c's u_nfw_c header
// derives the formulas). The table lives in a caller-owned descriptor
// (nfw_profile below), so every consumer of these readers states which
// table it reads:
//
//   nfw_profile_prepare   builds the f, G rows into a descriptor
//   nfw_profile_slopes    adds the Hermite slope rows (the IA reader)
//   nfw_profile_reset     frees a descriptor's rows
//   nfw_pos, nfw_um,
//   nfw_um4 and helpers   the scalar and SIMD table readers
//   nfw_fg_read           f(t), g(t) by cubic Hermite (the IA kernel)
//
// Ownership: the descriptor belongs to the caller (halo.c holds the
// instance of the ordinary halo and IA code; the three-point
// covariance holds its own). A new descriptor must start
// zero-initialized - nfw_profile prof = {0} - so the first
// preparation sees NULL rows: the builders free whatever row pointers
// they find. Preparation and the slope build take a mutable
// descriptor; every reader takes a const one. The caller decides when
// to rebuild: compare its own cache key AND the node count itself
// against the built descriptor, then call nfw_profile_prepare
// (halo.c's nfw_table shows the idiom). The table depends only on the
// node count, never on cosmology or nuisance parameters.
//
// The builders are defined inline in this header, so every target
// that compiles a consumer links without a dedicated halo_profile
// object (the standalone three-point covariance target among them);
// halo_profile.c remains the one-line home that project build files
// name.
//
// Thread safety: the builders are not thread-safe and must finish on
// one thread before any parallel region reads the descriptor (prepare,
// publish, then workers read). Every reader is thread-safe on a
// published descriptor.
//
// Units: the table axis is ln t for the dimensionless t = k r_s (and
// t = (1 + c) k r_s); f, g and u m(c) are dimensionless.
// ---------------------------------------------------------------------------

// SIMDe vectors (simde/x86/avx2.h and fma.h, included by basics.h).
//
// A v4d holds four doubles side by side, its "lanes" 0, 1, 2, 3, and one
// simde_mm256_* call applies the same operation to all four lanes at
// once (one AVX2 register on x86-64, two NEON registers on arm64). A v2d
// holds two doubles: one half of a v4d, lanes 0,1 (the low half) or
// lanes 2,3 (the high half). Vector variables carry a v prefix. In the
// halo spectra the four lanes are four consecutive mass nodes q, q+1,
// q+2, q+3 of one quadrature sum, so a v4d line does what the scalar
// path's line does for one node, four nodes at a time.
typedef simde__m256d v4d;
typedef simde__m128d v2d;


// ---------------------------------------------------------------------------
// The NFW transform table, shared by u_nfw_c, the spectrum table
// builders and the IA readers: the two smooth functions f, g of
// Abramowitz & Stegun
// 5.2.6-5.2.7 that carry the non-oscillating part of the sine and
// cosine integrals (u_nfw_c's header in halo.c derives them),
//
//   Si(t) = pi/2 - f(t) cos t - g(t) sin t,   Ci(t) = f(t) sin t - g(t) cos t
//   ->  f = Ci sin t + (pi/2 - Si) cos t,   g = -Ci cos t + (pi/2 - Si) sin t
//
// stored as f(t) and G(t) = g(t) + ln t (g ~ -ln t at t -> 0; G stays
// finite, -gamma_E there) at n_nodes nodes uniform in ln t over
// [NFW_TMIN, NFW_TASY]:
//
//   tab[0][i] = f(t_i),   tab[1][i] = G(t_i),   ln t_i = lim[0] + i lim[2]
//
// Reads (nfw_um) interpolate linearly in ln t between nodes, clamp
// below NFW_TMIN and switch to the asymptotic series above NFW_TASY.
// ---------------------------------------------------------------------------
static const double NFW_TMIN = 1e-10; // reads clamp below NFW_TMIN
static const double NFW_TASY = 50.0;  // asymptotic series above NFW_TASY

typedef struct {
  uint64_t cache;      // the caller's key of the table (Ntable.random
                       // in halo.c; the caller also compares n_nodes)
  int n_nodes;         // number of ln t nodes
  double lim[3];       // ln t axis: first, last, spacing
  double inv_spacing;  // 1/spacing (nfw_pos multiplies by it)
  double** tab;        // [2][n_nodes] f(t), G(t) = g(t) + ln t
  double** fg_slope;   // [2][n_nodes] Hermite rows h df/dln t and
                       // h dG/dln t of nfw_fg_read (nfw_profile_slopes;
                       // NULL until built, freed on every rebuild)
} nfw_profile;


// Build the f, G rows of prof at n_nodes ln t nodes and store key as
// its cache tag: frees any previous rows (the Hermite slope rows too,
// which derive from the table), fills the new table (a parallel loop:
// call outside any parallel region) and publishes it in prof. The
// caller owns the rebuild condition (the file banner), hands in a
// zero-initialized descriptor on the first call, and asks for at
// least two nodes: the axis spacing divides by n_nodes - 1 and the
// reads clamp onto the cell i = n_nodes - 2, so fewer than two nodes
// defines no table (checked; a short request aborts).
static inline void nfw_profile_prepare(
    nfw_profile* prof,   // the descriptor to (re)build
    const int n_nodes,   // ln t nodes of the table, at least 2
    const uint64_t key   // the caller's cache tag, stored in prof
  )
{
  if (2 > n_nodes) {
    log_fatal("nfw_profile_prepare: n_nodes = %d < 2", n_nodes);
    exit(1);
  }

  // f(t_i), G(t_i) from GSL Si, Ci (formulas in the table banner); the
  // caller owns the rebuild condition. The slope rows derive from the
  // table, so they go with it; nfw_profile_slopes rebuilds them on
  // demand.
  if (prof->tab != NULL) {
    free(prof->tab);
  }
  if (prof->fg_slope != NULL) {
    free(prof->fg_slope);
    prof->fg_slope = NULL;
  }

  // --- 1. AXIS SETUP ---
  double** tab = (double**) malloc2d(2, n_nodes);
  double* lim  = prof->lim;
  lim[0] = log(NFW_TMIN);
  lim[1] = log(NFW_TASY);
  lim[2] = (lim[1] - lim[0])/((double) n_nodes - 1.0);

  // --- 2. TABLE FILL ---
  /* PHYSICAL DERIVATION & LOGIC FLOW
     1. node: ln t_i = lim[0] + i lim[2],   t_i = exp(ln t_i)
     2. f(t) =  Ci(t) sin t + (pi/2 - Si(t)) cos t
     3. G(t) = -Ci(t) cos t + (pi/2 - Si(t)) sin t + ln t = g(t) + ln t */
  #pragma omp parallel for schedule(static)
  for (int i=0; i<n_nodes; i++) {
    const double lnt = lim[0] + i*lim[2];
    const double t   = exp(lnt);
    const double si  = gsl_sf_Si(t);
    const double ci  = gsl_sf_Ci(t);
    tab[0][i] = ci*sin(t) + (M_PI_2 - si)*cos(t);        // f(t)
    tab[1][i] = -ci*cos(t) + (M_PI_2 - si)*sin(t) + lnt; // G(t)
  }

  // --- 3. PUBLISH ---
  prof->n_nodes     = n_nodes;
  prof->inv_spacing = 1.0/lim[2];
  prof->tab         = tab;
  prof->cache       = key;
}


// Build the cubic Hermite slope rows of nfw_fg_read from the table of
// prof, if they are not built: at node ln t_i, h df/dln t = -h t g(t)
// and h dG/dln t = h t f(t), with g = G - ln t and h the node spacing
// (the slopes are exact: nfw_fg_read's header). nfw_profile_prepare
// frees the rows with the table, so a rebuilt table never keeps stale
// slopes. nfw_profile_prepare must have run; single-threaded, as the
// builders above.
static inline void nfw_profile_slopes(
    nfw_profile* prof    // the descriptor to add slope rows to
  )
{
  // NULL means "not built for this table" (the header above)
  if (prof->fg_slope != NULL) {
    return;
  }

  double** fg_slope = (double**) malloc2d(2, prof->n_nodes);
  for (int i=0; i<prof->n_nodes; i++) {
    const double lnt = prof->lim[0] + i*prof->lim[2];
    const double t   = exp(lnt);
    const double f_i = prof->tab[0][i];
    const double g_i = prof->tab[1][i] - lnt;
    fg_slope[0][i] = -t*g_i*prof->lim[2];
    fg_slope[1][i] = t*f_i*prof->lim[2];
  }
  prof->fg_slope = fg_slope;
}


// Free every row of prof and zero the descriptor, so a later
// nfw_profile_prepare starts from a clean state.
static inline void nfw_profile_reset(
    nfw_profile* prof    // the descriptor to release
  )
{
  if (prof->tab != NULL) {
    free(prof->tab);
  }
  if (prof->fg_slope != NULL) {
    free(prof->fg_slope);
  }
  *prof = (nfw_profile) {0};
}


// Position of ln t on the table grid of prof: node index i and the
// fraction frac of the interval [i, i + 1], so a read is
// tab[i] + frac*(tab[i + 1] - tab[i]); f and G share the grid. Below
// NFW_TMIN the read returns tab[0]; i is clamped onto the last
// interval. nfw_profile_prepare must have run.
static inline int nfw_pos(
    const nfw_profile* prof, // the table descriptor
    const double lnt,        // ln t
    double* frac             // output: fraction of the interval [i, i + 1]
  )
{
  const double pos = (fmax(lnt, prof->lim[0]) - prof->lim[0])
                     *prof->inv_spacing;

  int i = (int) pos;
  if (i > prof->n_nodes - 2) {
    i = prof->n_nodes - 2;
  }

  *frac = pos - i;
  return i;
}


// u m(c) of the NFW transform for one halo at one k (u_nfw_c header),
//
//   u m(c) = [g(x) - g(xu)] + 2 g(xu) sin^2(c x/2) + [f(xu) - 1/xu] sin(c x)
//   x = k r_s,   xu = (1 + c) x
//
// with f, g read from the table of prof as f and G = g + ln t:
//
//   fu = f(xu),   Gx = G(x),   Gu = G(xu),   gu = g(xu) = Gu - ln xu,
//   g(x) - g(xu) = Gx - Gu + ln(1 + c)
//
// The caller passes lnx = ln x and ln1c = ln(1 + c) (the table axis is
// ln t) and divides by m(c). nfw_profile_prepare must have run (the
// build is not thread-safe; this read is).
static inline double nfw_um(
    const nfw_profile* prof, // the table descriptor
    const double c,          // concentration r_Delta/r_s
    const double x,          // k r_s
    const double lnx,        // ln x
    const double ln1c        // ln(1 + c)
  )
{
  const double* restrict tab_f = prof->tab[0];  // f(t) on the ln t grid
  const double* restrict tab_G = prof->tab[1];  // G(t) = g(t) + ln t
  const double lnxu = lnx + ln1c;       // ln xu, xu = (1 + c) x
  const double xu = (1.0 + c)*x;

  // f(xu), G(x), G(xu): table reads up to NFW_TASY, the asymptotic
  // series (A&S 5.2.34-35) above it, nested,
  //   f(t) ~ (1 - 2!/t^2 + 4!/t^4 - 6!/t^6 + 8!/t^8)/t
  //   g(t) ~ (1 - 3!/t^2 + 5!/t^4 - 7!/t^6 + 9!/t^8)/t^2
  // (2, 12, 30, 56 and 6, 20, 42, 72 are the ratios of consecutive
  // coefficients). x < xu, so x may sit in the table when xu does not.
  double fu, Gx, Gu;
  if (xu <= NFW_TASY) {
    double frac_u, frac_x;
    // f and G share the grid
    const int iu = nfw_pos(prof, lnxu, &frac_u);
    const int ix = nfw_pos(prof, lnx, &frac_x);
    Gu = frac_u*(tab_G[iu + 1] - tab_G[iu]) + tab_G[iu];
    fu = frac_u*(tab_f[iu + 1] - tab_f[iu]) + tab_f[iu];
    Gx = frac_x*(tab_G[ix + 1] - tab_G[ix]) + tab_G[ix];
  }
  else {
    const double v = 1.0/(xu*xu); // series variable v = 1/xu^2
    fu = (1.0 - 2.0*v*(1.0 - 12.0*v*(1.0 - 30.0*v*(1.0 - 56.0*v))))/xu;
    Gu = v*(1.0 - 6.0*v*(1.0 - 20.0*v*(1.0 - 42.0*v*(1.0 - 72.0*v)))) + lnxu;
    if (x <= NFW_TASY) {
      double frac_x;
      const int ix = nfw_pos(prof, lnx, &frac_x);
      Gx = frac_x*(tab_G[ix + 1] - tab_G[ix]) + tab_G[ix];
    }
    else {
      const double w = 1.0/(x*x); // series variable w = 1/x^2
      Gx = w*(1.0 - 6.0*w*(1.0 - 20.0*w*(1.0 - 42.0*w*(1.0 - 72.0*w)))) + lnx;
    }
  }

  // u m(c) = [g(x) - g(xu)] + 2 g(xu) sin^2(c x/2) + [f(xu) - 1/xu] sin(c x),
  // with g(x) - g(xu) = Gx - Gu + ln1c
  const double gu = Gu - lnxu;          // g(xu)
  const double sin_half = sin(0.5*c*x); // sin(c x/2)
  return (Gx - Gu + ln1c) + 2.0*gu*sin_half*sin_half + (fu - 1.0/xu)*sin(c*x);
}


// ---------------------------------------------------------------------------
// SIMD path of the halo spectra: nfw_um on four mass nodes at once.
//
// The mass-node sums of p_gm and p_gg call nfw_um4, which
// is nfw_um with each of its four arguments carrying four nodes (one
// per lane of a v4d) and each lane of its result bitwise the scalar
// nfw_um of that node. The helpers below build it, in reading order:
//
//   nfw_fmadd4, nfw_fnmadd4 - a*b + c and c - a*b, rounded once where
//                             the scalar path fuses them
//   nfw_pos4                - position (node index, fraction) of ln t
//                             on the table grid
//   nfw_read4               - the linear table read at that position
//   nfw_sin4                - libm sin on each lane
//   nfw_series_step4        - one bracket of the asymptotic series
//   nfw_G_asym4             - the asymptotic series of G(t)
//   nfw_um4                 - the kernel itself
//
// Each header says what its function computes, how the lanes map to
// nodes and why its result is bitwise the scalar path's, under the
// strict IEEE flags (-frounding-math -ftrapping-math): fused
// multiply-adds exactly where the scalar path fuses, sign-bit masks
// instead of FP compares, table indices truncated and clamped in
// double, and every helper always inlined (on arm64 a v4d is a union of
// two NEON registers and a real call would pass it through memory).
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// nfw_fmadd4: a*b + c on four lanes with one rounding, the vector form of
// the scalar path's fused multiply-add.
//
// A fused multiply-add computes a*b + c as one operation: the product
// a*b is kept exact and only the final sum is rounded to double (one
// rounding). A separate multiply and add rounds twice, and the two
// results can differ in the last bit. The compiler fuses the scalar
// path's a*b + c (nfw_um, the mass-node sums), so the vector path must
// fuse the same products at the same places to stay bitwise equal to
// it; nfw_fmadd4 is the one place where it does.
//
// With native x86 FMA, simde_mm256_fmadd_pd is one 256-bit FMA3
// instruction (vfmadd...pd, one rounding).
// Without it (arm64), that same call is a multiply and then an add, two
// roundings: SIMDe has no NEON branch at 256 bits. The two-lane
// simde_mm_fmadd_pd is a real fused NEON instruction (vfmaq_f64), so the
// v4d is split into its two v2d halves (lanes 0,1 = the low half, lanes
// 2,3 = the high half), each half is fused, and the halves are joined
// again into a v4d. Lane l of the result is a[l]*b[l] + c[l] either
// way. An x86 build without FMA also takes this split branch; there the
// halves round twice, as the scalar path does, so the two still match.
//
// Parameters:
//   va - the four multiplicands a
//   vb - the four multipliers b
//   vc - the four addends c
//
// Returns:
//   a*b + c on each lane, one rounding (two on x86 built without FMA)
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d nfw_fmadd4(
    const v4d va,   // a on four lanes
    const v4d vb,   // b on four lanes
    const v4d vc    // c on four lanes
  )
{
#ifdef SIMDE_X86_FMA_NATIVE
  // a*b + c on all four lanes, one fused instruction
  return simde_mm256_fmadd_pd(va, vb, vc);
#else
  // the low half of each input (castpd256_pd128 keeps the lower two
  // doubles; it moves no data)
  const v2d va_low = simde_mm256_castpd256_pd128(va);  // a, lanes 0,1
  const v2d vb_low = simde_mm256_castpd256_pd128(vb);  // b, lanes 0,1
  const v2d vc_low = simde_mm256_castpd256_pd128(vc);  // c, lanes 0,1

  // the high half of each input (extractf128_pd(v, 1) takes the upper
  // two doubles)
  const v2d va_high = simde_mm256_extractf128_pd(va, 1);  // a, lanes 2,3
  const v2d vb_high = simde_mm256_extractf128_pd(vb, 1);  // b, lanes 2,3
  const v2d vc_high = simde_mm256_extractf128_pd(vc, 1);  // c, lanes 2,3

  // a*b + c fused on lanes 0,1
  const v2d vlow  = simde_mm_fmadd_pd(va_low, vb_low, vc_low);

  // a*b + c fused on lanes 2,3
  const v2d vhigh = simde_mm_fmadd_pd(va_high, vb_high, vc_high);

  // join the halves: set_m128d(high, low) puts vlow in lanes 0,1 and
  // vhigh in lanes 2,3
  return simde_mm256_set_m128d(vhigh, vlow);
#endif
}


// ---------------------------------------------------------------------------
// nfw_fnmadd4: c - a*b on four lanes with one rounding, the vector form
// of the scalar path's fused negative multiply-add.
//
// As nfw_fmadd4 with the product negated: the exact a*b is subtracted
// from c and only the difference is rounded (one rounding). The
// asymptotic series of nfw_um is a chain of 1 - k v (...) steps that
// the compiler fuses this way in the scalar path, so nfw_series_step4
// and nfw_G_asym4 fuse them through this function to stay bitwise
// equal.
//
// One 256-bit FMA3 instruction with native x86 FMA; otherwise the
// 128-bit split of nfw_fmadd4 (lanes 0,1 low, lanes 2,3 high), each half
// fused on arm64 by vfmsq_f64, joined again (on x86 without FMA the
// halves round twice, as the scalar path does). Lane l of the result is
// c[l] - a[l]*b[l] either way.
//
// Parameters:
//   va - the four multiplicands a
//   vb - the four multipliers b
//   vc - the four minuends c
//
// Returns:
//   c - a*b on each lane, one rounding (two on x86 built without FMA)
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d nfw_fnmadd4(
    const v4d va,   // a on four lanes
    const v4d vb,   // b on four lanes
    const v4d vc    // c on four lanes
  )
{
#ifdef SIMDE_X86_FMA_NATIVE
  // c - a*b on all four lanes, one fused instruction
  return simde_mm256_fnmadd_pd(va, vb, vc);
#else
  // the low half of each input, as in nfw_fmadd4
  const v2d va_low = simde_mm256_castpd256_pd128(va);  // a, lanes 0,1
  const v2d vb_low = simde_mm256_castpd256_pd128(vb);  // b, lanes 0,1
  const v2d vc_low = simde_mm256_castpd256_pd128(vc);  // c, lanes 0,1

  // the high half of each input
  const v2d va_high = simde_mm256_extractf128_pd(va, 1);  // a, lanes 2,3
  const v2d vb_high = simde_mm256_extractf128_pd(vb, 1);  // b, lanes 2,3
  const v2d vc_high = simde_mm256_extractf128_pd(vc, 1);  // c, lanes 2,3

  // c - a*b fused on lanes 0,1
  const v2d vlow  = simde_mm_fnmadd_pd(va_low, vb_low, vc_low);

  // c - a*b fused on lanes 2,3
  const v2d vhigh = simde_mm_fnmadd_pd(va_high, vb_high, vc_high);

  // join the halves: vlow in lanes 0,1, vhigh in lanes 2,3
  return simde_mm256_set_m128d(vhigh, vlow);
#endif
}


// ---------------------------------------------------------------------------
// nfw_pos4: nfw_pos on four lanes, the position of ln t on the table
// grid of prof for four values of t at once.
//
// The table stores f and G at uniform nodes ln t_i = ln t_min +
// i spacing. A "position" is the pair (i, frac) that places ln t in the
// grid: i is the index of the node at or below ln t, and frac in [0, 1)
// is how far ln t sits into the cell [ln t_i, ln t_{i+1}], in cells:
//
//   pos  = (max(ln t, ln t_min) - ln t_min)/spacing
//   i    = min(trunc(pos), n_nodes - 2)      (clamped onto the last cell)
//   frac = pos - i
//
// Lane l of vlnt is one ln t; index[l] and lane l of the result are its
// i and frac. Below ln t_min the clamp puts the read at the first node
// (f and G are flat there, see NFW_TMIN).
//
// Why it is bitwise nfw_pos: the scalar path computes pos in double,
// converts it to int (which truncates) and clamps. Here trunc and min
// are taken in double, and both are exact for the non-negative
// positions of the grid (an integer-valued double rounds to nothing),
// so pos - i is the same double and the int conversion, done lane by
// lane, yields the same i. Compares are avoided on purpose: the strict
// IEEE flags split a vector FP compare into scalar compares per lane.
//
// Parameters:
//   prof  - the table descriptor
//   vlnt  - ln t on four lanes
//   index - output: the node index i of each lane
//
// Returns:
//   frac, the fraction of the cell [i, i + 1], on each lane
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d nfw_pos4(
    const nfw_profile* prof, // the table descriptor
    const v4d vlnt,          // ln t on four lanes
    int index[4]             // output: node index i of each lane
  )
{
  // ln t_min, the first grid node, in all four lanes (set1 copies one
  // scalar into every lane)
  const v4d vlnt_min = simde_mm256_set1_pd(prof->lim[0]);

  // 1/spacing of the grid in all four lanes
  const v4d vinv_spacing = simde_mm256_set1_pd(prof->inv_spacing);

  // n_nodes - 2, the index of the last cell, in all four lanes
  const v4d vlast_cell = simde_mm256_set1_pd((double) (prof->n_nodes - 2));

  // scalar (nfw_pos): pos = (max(ln t, ln t_min) - ln t_min)/spacing,
  // i = min(trunc(pos), n - 2), each step below on all four lanes

  // max(ln t, ln t_min): the table clamp below ln t_min
  const v4d vlnt_clamped = simde_mm256_max_pd(vlnt, vlnt_min);

  // ln t - ln t_min, the distance from the first grid node
  const v4d vlnt_offset = simde_mm256_sub_pd(vlnt_clamped, vlnt_min);

  // pos = (ln t - ln t_min)/spacing, the position in grid cells
  const v4d vpos = simde_mm256_mul_pd(vlnt_offset, vinv_spacing);

  // trunc(pos): round toward zero, the cell number as a double
  const v4d vpos_trunc = simde_mm256_round_pd(vpos, SIMDE_MM_FROUND_TO_ZERO);

  // i = min(trunc(pos), n - 2): the last-cell clamp
  const v4d vnode = simde_mm256_min_pd(vpos_trunc, vlast_cell);

  double node[4];

  // the four cell numbers to the plain double[4] (storeu writes the four
  // lanes to memory), then to int lane by lane
  simde_mm256_storeu_pd(node, vnode);
  for (int lane=0; lane<4; lane++) {
    index[lane] = (int) node[lane];
  }

  // frac = pos - i on each lane
  return simde_mm256_sub_pd(vpos, vnode);
}


// ---------------------------------------------------------------------------
// nfw_read4: the linear table read of nfw_um on four lanes.
//
// A read of the f, G table at position (i, frac) is the straight line
// between the two nodes that bracket ln t,
//
//   tab[i] + frac*(tab[i + 1] - tab[i]),
//
// frac = 0 at node i, frac = 1 at node i + 1 (nfw_pos4 gives i and
// frac). Lane l reads tab at index[l] with lane l of vfrac.
//
// Memory access: each lane needs the two neighbours tab[i], tab[i + 1],
// which sit side by side, so one 16-byte load per lane fetches both
// (a v2d pair). The four pairs are then regrouped into a v4d of left
// nodes and a v4d of right nodes. A gather instruction (four scattered
// loads in one call) would do the same, but it is slow on several x86
// cores and is lane-by-lane loads on NEON anyway.
//
// Why it is bitwise nfw_um: the scalar read is
// frac*(tab[i + 1] - tab[i]) + tab[i], which the compiler fuses into
// one multiply-add; the vector read fuses the same product through
// nfw_fmadd4.
//
// Parameters:
//   tab   - the table row to read, f (prof->tab[0]) or G (prof->tab[1])
//   index - the node index i of each lane (from nfw_pos4)
//   vfrac - the fraction of the cell [i, i + 1] of each lane
//
// Returns:
//   the interpolated table value on each lane
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d nfw_read4(
    const double* restrict tab,  // table row on the ln t grid: f or G
    const int index[4],          // node index i of each lane
    const v4d vfrac              // fraction of the cell of each lane
  )
{
  // the pair (tab[i], tab[i + 1]) of each lane, one two-double load each
  // (loadu reads two consecutive doubles from memory into a v2d)
  const v2d vpair0 = simde_mm_loadu_pd(tab + index[0]);  // lane 0's pair
  const v2d vpair1 = simde_mm_loadu_pd(tab + index[1]);  // lane 1's pair
  const v2d vpair2 = simde_mm_loadu_pd(tab + index[2]);  // lane 2's pair
  const v2d vpair3 = simde_mm_loadu_pd(tab + index[3]);  // lane 3's pair

  // regroup the pairs into left nodes tab[i] and right nodes tab[i + 1]:
  // unpacklo takes the first double of each pair, unpackhi the second

  // (tab[i0], tab[i1]): the left nodes of lanes 0,1
  const v2d vleft_low = simde_mm_unpacklo_pd(vpair0, vpair1);

  // (tab[i2], tab[i3]): the left nodes of lanes 2,3
  const v2d vleft_high = simde_mm_unpacklo_pd(vpair2, vpair3);

  // (tab[i0 + 1], tab[i1 + 1]): the right nodes of lanes 0,1
  const v2d vright_low = simde_mm_unpackhi_pd(vpair0, vpair1);

  // (tab[i2 + 1], tab[i3 + 1]): the right nodes of lanes 2,3
  const v2d vright_high = simde_mm_unpackhi_pd(vpair2, vpair3);

  // tab[i] on all four lanes (set_m128d joins the halves, low first)
  const v4d vleft = simde_mm256_set_m128d(vleft_high, vleft_low);

  // tab[i + 1] on all four lanes
  const v4d vright = simde_mm256_set_m128d(vright_high, vright_low);

  // tab[i + 1] - tab[i], the rise across the cell
  const v4d vrise = simde_mm256_sub_pd(vright, vleft);

  // frac*(tab[i + 1] - tab[i]) + tab[i], fused as the scalar read
  return nfw_fmadd4(vfrac, vrise, vleft);
}


// ---------------------------------------------------------------------------
// nfw_sin4: sin on each of four lanes, with the libm sin of the scalar
// path.
//
// nfw_um needs sin(c x/2) and sin(c x) per node, and they are about
// half of the kernel's cost. There is no vector sine here by maintainer
// decision: a vector math library (SLEEF and the like) would give a
// different last bit from libm, and the SIMD path must be bitwise
// nfw_um. So the
// four angles are written out of the v4d into a plain double[4]
// (storeu), sin is called on each one exactly as the scalar path does,
// and the four sines are read back into a v4d (loadu). Lane l of the
// result is sin of lane l of vangle.
//
// Parameters:
//   vangle - the four angles, in radians
//
// Returns:
//   sin(angle) on each lane
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d nfw_sin4(
    const v4d vangle   // the four angles
  )
{
  double angle[4];
  double sine[4];

  // the four angles to a plain double[4]
  simde_mm256_storeu_pd(angle, vangle);

  // sin of each angle, the same libm call as the scalar path
  for (int lane=0; lane<4; lane++) {
    sine[lane] = sin(angle[lane]);
  }

  // the four sines back into one v4d (loadu reads four doubles from memory)
  return simde_mm256_loadu_pd(sine);
}


// ---------------------------------------------------------------------------
// nfw_series_step4: one step of the nested asymptotic series of nfw_um,
//
//   poly -> 1 - k v poly,
//
// on four lanes, fused as the scalar 1.0 - k*v*(...) (nfw_fnmadd4).
//
// Above the table's top (t > NFW_TASY) nfw_um evaluates f and g by their
// asymptotic series in v = 1/t^2 (A&S 5.2.34-35),
//
//   f(t) ~ (1 - 2!/t^2 + 4!/t^4 - 6!/t^6 + 8!/t^8)/t
//   g(t) ~ (1 - 3!/t^2 + 5!/t^4 - 7!/t^6 + 9!/t^8)/t^2
//
// written in nested (Horner) form, innermost bracket first:
//
//   f(t) = (1 - 2v(1 - 12v(1 - 30v(1 - 56v))))/t
//   g(t) = v(1 - 6v(1 - 20v(1 - 42v(1 - 72v))))
//
// where 2, 12, 30, 56 and 6, 20, 42, 72 are the ratios of consecutive
// coefficients (2!/0!, 4!/2!, ...). Starting from the innermost bracket
// (1 - 56v or 1 - 72v), each call of this function wraps the series in
// one more bracket: poly = 1 - 56v becomes 1 - 30v(1 - 56v), and so on
// outward, k taking the next ratio each time. nfw_um4 chains the f
// steps (30, 12, 2) and nfw_G_asym4 the g steps (42, 20, 6).
//
// Parameters:
//   k     - the coefficient ratio of this bracket
//   vv    - the series variable v = 1/t^2 on four lanes
//   vpoly - the bracket built so far on four lanes
//
// Returns:
//   1 - k v poly on each lane, rounded as nfw_fnmadd4 rounds it
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d nfw_series_step4(
    const double k,   // coefficient ratio of this bracket
    const v4d vv,     // series variable v = 1/t^2 on four lanes
    const v4d vpoly   // the nested bracket built so far
  )
{
  // the coefficient ratio k in all four lanes
  const v4d vratio = simde_mm256_set1_pd(k);

  // k v on each lane
  const v4d vkv = simde_mm256_mul_pd(vratio, vv);

  // 1 in all four lanes
  const v4d vone = simde_mm256_set1_pd(1.0);

  // 1 - (k v) poly, one rounding, as the scalar 1.0 - k*v*(...)
  return nfw_fnmadd4(vkv, vpoly, vone);
}


// ---------------------------------------------------------------------------
// nfw_G_asym4: the asymptotic series of G(t) = g(t) + ln t on four lanes,
//
//   G(t) = v(1 - 6v(1 - 20v(1 - 42v(1 - 72v)))) + ln t,   v = 1/t^2,
//
// the nested form of g(t) ~ (1 - 3!/t^2 + 5!/t^4 - 7!/t^6 + 9!/t^8)/t^2
// (nfw_series_step4 explains the nesting and the ratios 6, 20, 42, 72).
//
// The f, G table stores G = g + ln t up to t = NFW_TASY; above it the
// series replaces the table, since there the dropped terms (11!/t^10
// and beyond) are below the table's accuracy while the table would need
// ever more nodes for g's slow 1/t^2 fall-off. nfw_um4 calls this for
// G(xu) and G(x) on the lanes past NFW_TASY (the other lanes get a
// finite stand-in that the caller discards).
//
// Bitwise nfw_um: the same brackets in the same order, each 1 - k v (..)
// fused (nfw_fnmadd4), and the final v poly + ln t fused (nfw_fmadd4),
// as the compiler fuses the scalar expression.
//
// Parameters:
//   vv   - the series variable v = 1/t^2 on four lanes
//   vlnt - ln t on four lanes
//
// Returns:
//   G(t) on each lane
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d nfw_G_asym4(
    const v4d vv,    // series variable v = 1/t^2 on four lanes
    const v4d vlnt   // ln t on four lanes
  )
{
  // 1 in all four lanes
  const v4d vone = simde_mm256_set1_pd(1.0);

  // the innermost coefficient ratio 72 in all four lanes
  const v4d vratio72 = simde_mm256_set1_pd(72.0);

  // scalar: v*(1 - 6v(1 - 20v(1 - 42v(1 - 72v)))) + ln t, innermost first

  // 1 - 72v
  v4d vpoly = nfw_fnmadd4(vratio72, vv, vone);

  // the outer brackets, one per step
  vpoly = nfw_series_step4(42.0, vv, vpoly);  // 1 - 42v(1 - 72v)
  vpoly = nfw_series_step4(20.0, vv, vpoly);  // 1 - 20v(...)
  vpoly = nfw_series_step4(6.0, vv, vpoly);   // 1 - 6v(...)

  // v poly + ln t, fused
  return nfw_fmadd4(vv, vpoly, vlnt);
}


// ---------------------------------------------------------------------------
// nfw_um4: nfw_um on four mass nodes at once, the NFW transform u m(c)
// of four halos at one k (u_nfw_c header),
//
//   u m(c) = [g(x) - g(xu)] + 2 g(xu) sin^2(c x/2) + [f(xu) - 1/xu] sin(c x)
//   x = k r_s,   xu = (1 + c) x,
//
// with f and g read from the table of prof as f and G = g + ln t:
//
//   fu = f(xu),   Gx = G(x),   Gu = G(xu),   g(xu) = Gu - ln xu,
//   g(x) - g(xu) = Gx - Gu + ln(1 + c).
//
// Lanes: lane l of every argument belongs to one mass node (the callers
// pass nodes q, q+1, q+2, q+3 of a quadrature sum), and lane l of the
// result is u m(c) of that node. The callers divide by m(c) (folded into
// their weights).
//
// Algorithm, in order (the section banners in the body):
//   1. branch masks: per lane, table (t <= NFW_TASY) or asymptotic
//      series (t > NFW_TASY) for t = xu and for t = x, from the sign
//      bit of NFW_TASY - t (movemask), no FP compare;
//   2. table reads: the position (i, frac) of ln xu and of ln x on the
//      ln t grid (nfw_pos4), then f(xu), G(xu), G(x) by linear
//      interpolation (nfw_read4), skipped when every lane is past
//      NFW_TASY;
//   3. asymptotic series: f(xu), G(xu), G(x) in v = 1/t^2
//      (nfw_series_step4, nfw_G_asym4), skipped when no lane needs it;
//      per lane, blendv keeps the table value or takes the series;
//   4. the two sines sin(c x/2) and sin(c x) (nfw_sin4), then the
//      combination above.
//
// Why it is bitwise nfw_um: the same operations in the same order on
// every lane, a fused multiply-add exactly where the compiler fuses the
// scalar a*b + c (nfw_fmadd4 / nfw_fnmadd4), and libm sin on every lane
// (nfw_sin4). The branch is taken per lane by masks, so a lane past
// NFW_TASY gets the series value and a lane below it the table value,
// as the scalar if/else would give. Every function here is always
// inlined: on arm64 a v4d is a union of two NEON registers and a real
// call would pass it through memory.
//
// nfw_profile_prepare must have run (the build is not thread-safe;
// this read is).
//
// Parameters:
//   prof  - the table descriptor
//   vc    - concentration c = r_Delta/r_s of the four nodes
//   vx    - x = k r_s of the four nodes
//   vlnx  - ln x of the four nodes
//   vln1c - ln(1 + c) of the four nodes
//
// Returns:
//   u m(c) of each node on its lane, dimensionless
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d nfw_um4(
    const nfw_profile* prof, // the table descriptor
    const v4d vc,            // concentration r_Delta/r_s on four lanes
    const v4d vx,            // k r_s on four lanes
    const v4d vlnx,          // ln x on four lanes
    const v4d vln1c          // ln(1 + c) on four lanes
  )
{
  const double* restrict tab_f = prof->tab[0];  // f(t) on the ln t grid
  const double* restrict tab_G = prof->tab[1];  // G(t) = g(t) + ln t

  // 1 in all four lanes
  const v4d vone = simde_mm256_set1_pd(1.0);

  // NFW_TASY, the top of the table, in all four lanes
  const v4d vtasy = simde_mm256_set1_pd(NFW_TASY);

  // scalar: lnxu = lnx + ln1c;  xu = (1.0 + c)*x

  // ln xu = ln x + ln(1 + c)
  const v4d vlnxu = simde_mm256_add_pd(vlnx, vln1c);

  // 1 + c
  const v4d vone_plus_c = simde_mm256_add_pd(vone, vc);

  // xu = (1 + c) x
  const v4d vxu = simde_mm256_mul_pd(vone_plus_c, vx);

  // --- 1. BRANCH MASKS ---
  // scalar: if (xu <= NFW_TASY) read the table, else the series; the
  // same for x. NFW_TASY - t has its sign bit set exactly where
  // t > NFW_TASY (the asymptotic series) and clear where the table is
  // read (+0 at t = NFW_TASY). The masks must see the rounded xu of the
  // scalar compare: xu = (1 + c) x stays unfused because it has other
  // uses (1/xu, the blend), which GCC's -ffp-contract=fast needs to
  // leave the product alone

  // NFW_TASY - xu: negative (sign bit set) on the series lanes
  const v4d vasym_u = simde_mm256_sub_pd(vtasy, vxu);

  // NFW_TASY - x
  const v4d vasym_x = simde_mm256_sub_pd(vtasy, vx);

  // movemask collects the sign bit of each lane into bit l of an int:
  // 0 = every lane reads the table, 0xF = every lane takes the series
  const int asym_u = simde_mm256_movemask_pd(vasym_u);  // for t = xu
  const int asym_x = simde_mm256_movemask_pd(vasym_x);  // for t = x

  // --- 2. TABLE READS: f(xu), G(xu), G(x) ---
  // lanes past NFW_TASY read the clamped last interval; the blend below
  // replaces them

  // (0, 0, 0, 0) until a branch fills them
  v4d vfu = simde_mm256_setzero_pd();  // f(xu)
  v4d vGu = simde_mm256_setzero_pd();  // G(xu)
  v4d vGx = simde_mm256_setzero_pd();  // G(x)

  if (asym_u != 0xF) {
    int index_u[4];

    // scalar: iu = nfw_pos(prof, lnxu, &frac_u), f and G share the grid
    const v4d vfrac_u = nfw_pos4(prof, vlnxu, index_u);

    // scalar: Gu = frac_u*(tab_G[iu + 1] - tab_G[iu]) + tab_G[iu]
    vGu = nfw_read4(tab_G, index_u, vfrac_u);

    // scalar: fu = frac_u*(tab_f[iu + 1] - tab_f[iu]) + tab_f[iu]
    vfu = nfw_read4(tab_f, index_u, vfrac_u);
  }
  if (asym_x != 0xF) {
    int index_x[4];

    // scalar: ix = nfw_pos(prof, lnx, &frac_x)
    const v4d vfrac_x = nfw_pos4(prof, vlnx, index_x);

    // scalar: Gx = frac_x*(tab_G[ix + 1] - tab_G[ix]) + tab_G[ix]
    vGx = nfw_read4(tab_G, index_x, vfrac_x);
  }

  // --- 3. ASYMPTOTIC SERIES (A&S 5.2.34-35, nfw_um) ---
  // lanes that read the table evaluate the series at t = NFW_TASY, a
  // finite stand-in (no 1/t^2 overflow at tiny t) that the blend discards
  if (asym_u != 0) {
    // t = xu on the series lanes, NFW_TASY on the table lanes (blendv
    // takes lane l from its second argument where bit l of the mask is
    // set, from its first argument otherwise)
    const v4d vt = simde_mm256_blendv_pd(vtasy, vxu, vasym_u);

    // t^2
    const v4d vt2 = simde_mm256_mul_pd(vt, vt);

    // scalar: v = 1.0/(xu*xu), the series variable
    const v4d vv = simde_mm256_div_pd(vone, vt2);

    // scalar: fu = (1 - 2v(1 - 12v(1 - 30v(1 - 56v))))/xu, innermost first

    // the innermost coefficient ratio 56 in all four lanes
    const v4d vratio56 = simde_mm256_set1_pd(56.0);

    // 1 - 56v
    v4d vpoly = nfw_fnmadd4(vratio56, vv, vone);

    // the outer brackets, one per step
    vpoly = nfw_series_step4(30.0, vv, vpoly);  // 1 - 30v(1 - 56v)
    vpoly = nfw_series_step4(12.0, vv, vpoly);  // 1 - 12v(...)
    vpoly = nfw_series_step4(2.0, vv, vpoly);   // 1 - 2v(...)

    // poly/t
    const v4d vfu_asym = simde_mm256_div_pd(vpoly, vt);

    // f(xu): the series on the series lanes, the table value elsewhere
    vfu = simde_mm256_blendv_pd(vfu, vfu_asym, vasym_u);

    // scalar: Gu = v*(1 - 6v(1 - 20v(1 - 42v(1 - 72v)))) + lnxu
    const v4d vGu_asym = nfw_G_asym4(vv, vlnxu);

    // G(xu): the series on the series lanes, the table value elsewhere
    vGu = simde_mm256_blendv_pd(vGu, vGu_asym, vasym_u);
  }
  if (asym_x != 0) {
    // t = x on the series lanes, NFW_TASY on the table lanes
    const v4d vt = simde_mm256_blendv_pd(vtasy, vx, vasym_x);

    // t^2
    const v4d vt2 = simde_mm256_mul_pd(vt, vt);

    // scalar: w = 1.0/(x*x), the series variable
    const v4d vw = simde_mm256_div_pd(vone, vt2);

    // scalar: Gx = w*(1 - 6w(1 - 20w(1 - 42w(1 - 72w)))) + lnx
    const v4d vGx_asym = nfw_G_asym4(vw, vlnx);

    // G(x): the series on the series lanes, the table value elsewhere
    vGx = simde_mm256_blendv_pd(vGx, vGx_asym, vasym_x);
  }

  // --- 4. u m(c) ---
  // u m(c) = [Gx - Gu + ln(1+c)] + 2 g(xu) sin^2(c x/2)
  //          + [f(xu) - 1/xu] sin(c x),
  // g(xu) = Gu - ln xu; the scalar sum fuses both products:
  //   gu = Gu - lnxu;  sin_half = sin(0.5*c*x);
  //   (Gx - Gu + ln1c) + 2.0*gu*sin_half*sin_half + (fu - 1.0/xu)*sin(c*x)

  // 2 in all four lanes
  const v4d vtwo = simde_mm256_set1_pd(2.0);

  // gu = Gu - ln xu, that is g(xu)
  const v4d vgu = simde_mm256_sub_pd(vGu, vlnxu);

  // 2 gu
  const v4d vtwo_gu = simde_mm256_mul_pd(vtwo, vgu);

  // 0.5 in all four lanes
  const v4d vhalf = simde_mm256_set1_pd(0.5);

  // 0.5 c
  const v4d vhalf_c = simde_mm256_mul_pd(vhalf, vc);

  // 0.5 c x, the half angle
  const v4d vhalf_cx = simde_mm256_mul_pd(vhalf_c, vx);

  // sin(c x/2)
  const v4d vsin_half = nfw_sin4(vhalf_cx);

  // c x, the full angle
  const v4d vcx = simde_mm256_mul_pd(vc, vx);

  // sin(c x)
  const v4d vsin_full = nfw_sin4(vcx);

  // Gx - Gu
  const v4d vGx_minus_Gu = simde_mm256_sub_pd(vGx, vGu);

  // Gx - Gu + ln(1 + c), that is g(x) - g(xu)
  const v4d vg_diff = simde_mm256_add_pd(vGx_minus_Gu, vln1c);

  // 1/xu
  const v4d vinv_xu = simde_mm256_div_pd(vone, vxu);

  // fu - 1/xu
  const v4d vf_term = simde_mm256_sub_pd(vfu, vinv_xu);

  // 2 gu sin(c x/2)
  const v4d vtwo_gu_sin = simde_mm256_mul_pd(vtwo_gu, vsin_half);

  // (2 gu sin_half) sin_half + g_diff, fused as the scalar sum
  const v4d vsum = nfw_fmadd4(vtwo_gu_sin, vsin_half, vg_diff);

  // (fu - 1/xu) sin(c x) + the rest, fused: u m(c) on the four lanes
  return nfw_fmadd4(vf_term, vsin_full, vsum);
}


// ---------------------------------------------------------------------------
// nfw_fg_read: f(t) and g(t), the auxiliary functions of the sine and
// cosine integrals (the table banner above), read from the table of
// prof by cubic Hermite interpolation in ln t. The IA kernel only;
// nfw_um keeps its linear read.
//
// What cubic Hermite interpolation is: between two neighbouring nodes
// it uses the cubic that matches the tabulated value and the tabulated
// slope at both nodes. With s in [0, 1] the fraction of the interval,
//
//   y(s) = h00(s) y_i + h10(s) h y'_i + h01(s) y_(i+1) + h11(s) h y'_(i+1)
//   h00 = 2s^3 - 3s^2 + 1,   h10 = s^3 - 2s^2 + s,
//   h01 = -2s^3 + 3s^2,      h11 = s^3 - s^2
//
// (h the node spacing, ' = d/dln t). Its error falls as h^4 (about
// h^4 y''''/384), where the linear read's falls as h^2 (h^2 y''/8), on
// the same table: at the table's spacing it is several orders of
// magnitude more accurate.
//
// Why the slopes are exact and free: f and g obey f' = -g and
// g' = f - 1/t (A&S 5.2.6-7), so in the table variables
//
//   df/dln t = -t g(t),   dG/dln t = t f(t)     (G = g + ln t)
//
// follow from the stored values themselves: fg_slope (built from the
// table by nfw_profile_slopes) holds h df/dln t and h dG/dln t at
// every node, no derivative is approximated.
//
// Why the IA kernel needs it: the closed form of the kernel (the
// section's kernel header) subtracts terms that can be about a million
// times larger than their sum (near the series switch, large c, l = 6).
// The linear read's relative error of f, g is amplified by that factor
// into percent-level errors of gamma_hat; the Hermite read keeps the
// kernel near its floating-point limit. nfw_um has no such cancellation
// and keeps the cheaper linear read.
//
// Above NFW_TASY: the asymptotic series (A&S 5.2.34-35), nested,
//
//   f(t) ~ (1 - 2!/t^2 + 4!/t^4 - ... - 14!/t^14)/t
//   g(t) ~ (1 - 3!/t^2 + 5!/t^4 - ... - 15!/t^14)/t^2
//
// three terms longer than nfw_um's, which this accuracy needs (2, 12,
// ..., 182 and 6, 20, ..., 210 are the ratios of consecutive
// coefficients). Below NFW_TMIN the read returns the first node
// (nfw_pos).
//
// Cache invalidation:
//   prof->tab: the caller's rebuild condition (the file banner);
//   prof->fg_slope: freed with the table by nfw_profile_prepare,
//   rebuilt by nfw_profile_slopes. Both builds must have run.
//
// Parameters:
//   prof   - the table descriptor, slope rows built
//   t, lnt - the argument and its log (the caller has ln t)
//   f, g   - output: f(t), g(t)
// ---------------------------------------------------------------------------
static inline void nfw_fg_read(
    const nfw_profile* prof, // the table descriptor, slope rows built
    const double t,          // argument t > 0
    const double lnt,        // ln t
    double* f,               // output: f(t)
    double* g                // output: g(t)
  )
{
  if (t <= NFW_TASY) {
    const double* restrict tab_f   = prof->tab[0];       // f(t_i)
    const double* restrict tab_G   = prof->tab[1];       // G(t_i)
    const double* restrict slope_f = prof->fg_slope[0];  // h df/dln t
    const double* restrict slope_G = prof->fg_slope[1];  // h dG/dln t

    // node i, fraction s of [i, i + 1]
    double s;
    const int i = nfw_pos(prof, lnt, &s);

    // the four Hermite basis polynomials at s
    const double s2  = s*s;
    const double s3  = s2*s;
    const double h00 = 2.0*s3 - 3.0*s2 + 1.0;
    const double h10 = s3 - 2.0*s2 + s;
    const double h01 = -2.0*s3 + 3.0*s2;
    const double h11 = s3 - s2;

    *f = h00*tab_f[i] + h10*slope_f[i] + h01*tab_f[i + 1] + h11*slope_f[i + 1];

    const double G = h00*tab_G[i] + h10*slope_G[i]
                     + h01*tab_G[i + 1] + h11*slope_G[i + 1];
    *g = G - lnt;  // g = G - ln t
  }
  else {
    const double v = 1.0/(t*t);  // series variable v = 1/t^2

    *f = (1.0 - 2.0*v*(1.0 - 12.0*v*(1.0 - 30.0*v*(1.0 - 56.0*v
         *(1.0 - 90.0*v*(1.0 - 132.0*v*(1.0 - 182.0*v)))))))/t;

    *g = v*(1.0 - 6.0*v*(1.0 - 20.0*v*(1.0 - 42.0*v*(1.0 - 72.0*v
         *(1.0 - 110.0*v*(1.0 - 156.0*v*(1.0 - 210.0*v)))))));
  }
}

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
