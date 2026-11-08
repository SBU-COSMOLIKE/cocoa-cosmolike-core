// ============================================================================
// Radial inputs and all-pairs Limber spectra for the covariance
// ============================================================================
//
// A covariance needs the angular power spectrum of every pair of catalog
// fields, including pairs that the data vector never measures. This file
// prepares the radial ingredients of those spectra and the Limber spectra
// themselves. Reading order:
//
//   1. power_rows_cov, linear_power_logk_rows_cov
//                         - matter power P(k,a) for independent rows of
//                           wavenumbers at one scale factor (physical
//                           values, or base-10 logs plus a shared shift)
//   2. lensing_efficiency_cov, fill_radial_cov
//                         - lensing efficiency g(chi) and the density,
//                           lensing/magnification and signed NLA windows
//   3. radial_inputs_cov, radial_logchi_cov
//                         - the radial snapshot on Gauss-Legendre nodes
//                           (Limber) or on uniform ln(chi) samples (FFTLog)
//   4. limber_spectra_cov - every field-pair Limber spectrum on one common
//                           radial rule
//   5. oneloop_bias_spectra_cov
//                         - the one-loop galaxy-bias terms (b2, bs2) added
//                           to every spectrum with a lens field
//
// Units follow the core: distances in c/H0, wavenumbers in (c/H0)^-1,
// power in (c/H0)^3 and radial windows in (c/H0)^-1, so the spectra are
// dimensionless. Higher-order intrinsic-alignment (TATT) spectra are added
// by ia_cov.c on top of these.
// ============================================================================

#include <math.h>
#include <stdlib.h>

#include "spectra_cov.h"
#include "perturbation_cov.h"
#include "cosmolike/basics.h"
#include "cosmolike/bias.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/IA.h"
#include "cosmolike/pt_cfastpt.h"
#include "cosmolike/radial_weights.h"
#include "cosmolike/redshift_spline.h"
#include "cosmolike/structs.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

// ---------------------------------------------------------------------------
// SIMD vocabulary for the vector loops below.
//
// SIMD (single instruction, multiple data) applies one operation to several
// numbers at once. A v2d holds two doubles side by side; each position is a
// lane, numbered 0 and 1. Every call below acts on lane 0 and on lane 1
// separately: no call in this file moves a value from one lane to the
// other. The efficiency loop puts the two cumulative integrals A and B in
// the lanes; the spectrum loop puts two independent field-pair spectra
// there. Vector variables carry a v prefix.
//
// simde_mm_setzero_pd() sets both lanes to 0.0; simde_mm_set1_pd(x) copies
// x into both lanes; simde_mm_set_pd(high, low) puts its last argument in
// lane 0 and its first in lane 1. simde_mm_add_pd and simde_mm_mul_pd add
// or multiply lane by lane, rounding each result once.
// simde_mm_storeu_pd(p, v) writes lane 0 to p[0] and lane 1 to p[1]; the
// u (unaligned) form accepts any address, not only multiples of 16 bytes,
// but p[0] and p[1] must both exist.
//
// simde_mm_fmadd_pd(a, b, c) returns a*b + c in each lane. On arm64 (NEON)
// and on x86 builds with FMA enabled it is one fused instruction: the exact
// product is added to c and the sum is rounded once. Without such an
// instruction SIMDe multiplies and then adds, rounding twice. The scalar
// examples below write this step as fma(a, b, c).
// ---------------------------------------------------------------------------
typedef simde__m128d v2d;



// ============================================================================
// [SECTION] MATTER POWER FOR INDEPENDENT WAVENUMBER ROWS
// ============================================================================


// ---------------------------------------------------------------------------
// Read matter power for independent rows of wavenumbers at one scale factor.
//
// In a trispectrum integral, each row represents a pair K,Q. Its columns
// contain |K+Q| at the sampled relative angles. Every sample has the same
// redshift, so the core reader (p_lin_at_a or Pdelta_at_a) locates the
// redshift interpolation bracket once per row and repeats only the
// wavenumber half of its bilinear interpolation for each sample. Different
// rows only read the initialized cosmology tables and write separate
// outputs; they can therefore be assigned to different workers.
//
// The existing reader still performs every interpolation. This changes
// neither the power model nor its arithmetic, and introduces no cache or
// reduction.
//
// Parameters:
//   a      - scale factor shared by every sample
//   nrow   - number of independent rows, at least 1
//   ncol   - samples per row, at least 1
//   k      - [nrow][ncol] positive wavenumbers in (c/H0)^-1
//   linear - nonzero: linear p_lin(k,a); zero: the core's run-mode
//            Pdelta(k,a), nonlinear unless the run mode is "linear"
//   power  - [nrow][ncol] caller-owned output in (c/H0)^3, not overlapping k
//
// Call outside an OpenMP region, after initializing the core power tables.
// ---------------------------------------------------------------------------
void power_rows_cov(
    const double a,                  // shared scale factor
    const int nrow,                  // independent wavenumber rows
    const int ncol,                  // samples per row
    const double* const* k,          // positive physical wavenumbers
    const int linear,               // linear or configured nonlinear power
    double* const* power            // caller-owned output rows
  )
{
  if (nrow < 1
      || ncol < 1) {
    log_fatal("power_rows_cov needs positive row and column counts");
    exit(1);
  }

  // The nonlinear reader shares the core's run-mode latch (cosmo3D.c,
  // pdelta_dispatch): a static that the first Pdelta-family call writes
  // when the run mode is "linear". On the production path this function
  // can be that first call, so touch the dispatch serially here; the
  // workers below then only read it. The linear reader is stateless, and
  // one extra sample costs nothing.
  if (linear) {
    (void) p_lin(k[0][0], a);
  } else {
    (void) Pdelta(k[0][0], a);
  }

  // One iteration fills one complete output row at the common redshift.
  // A row depends on no other row, so no synchronization is needed and no
  // value depends on the number of workers; the static schedule gives each
  // worker a contiguous block of rows.
  #pragma omp parallel for schedule(static)
  for (int row=0; row<nrow; row++) {
    if (linear) {
      p_lin_at_a(a, k[row], ncol, power[row]);
    } else {
      Pdelta_at_a(a, k[row], ncol, power[row]);
    }
  }
}


// ---------------------------------------------------------------------------
// Locate the z bracket on the core's piecewise-uniform redshift grid.
//
// Verbatim copy of the core's piecewise_index (cosmo3D.c): that helper is
// file-local (static) there, so it cannot be linked from this module.
// Keep the two copies identical.
//
// Returns:
//   bracket index j with grid[j] <= q < grid[j+1], clamped to
//   [0, n_total-2]
// ---------------------------------------------------------------------------
static inline int piecewise_index(double q,
                                  int nseg,
                                  const int *start,
                                  const int *len,
                                  const double *xmin,
                                  const double *inv_dx,
                                  int n_total)
{
  // Pick the segment: q is in segment s if q is in [xmin[s], xmin[s+1]).
  // Linear scan is fine for nseg <= ~10 (branch-predicted, all in L1).
  int s = 0;
  while (s < nseg - 1 && q >= xmin[s+1]) s++;

  // Direct index within segment s.
  int j = start[s] + (int)((q - xmin[s]) * inv_dx[s]);

  // Clamp to valid bilinear bracket range.
  if (j < 0)             j = 0;
  if (j > n_total - 2)   j = n_total - 2;
  return j;
}


// ---------------------------------------------------------------------------
// Read linear matter power for rows of base-10 LOG wavenumbers plus a shift.
//
// The connected-covariance angle integrals read P_lin at the internal
// momentum |K+Q| for every (pair, angle) sample. Under Limber that grid is
// the same at every radial shell up to one overall factor: the wavenumber
// of sample m at a shell with transverse distance f_K is
// k[m] = magnitude[m]/f_K, with magnitude fixed in multipole units. The
// standard reader (p_lin_at_a, cosmo3D.c) computes log10(k/coverH0) for
// every sample, and that logarithm heads each sample's dependency chain:
// its result is the table index, so the four table loads and everything
// after them wait on it. On the shared grid the logarithm is
// shift-invariant,
//
//   log10(magnitude/f_K/coverH0) = log10(magnitude)
//                                  - log10(f_K) - log10(coverH0),
//
// so the caller takes log10(magnitude) once for the whole run and this
// reader adds one scalar per call. Removing the per-sample log10 roughly
// halves the read on the production |K+Q| grid. The exp that restores P
// from the stored lnP stays: its latency hides under the next sample's
// table loads, so removing it buys about one percent.
//
// The z half of the bilinear read is also a call constant: one bracket j
// and one weight dy serve every sample. Collapsing it once per call into
// a slice, lnP_a[i] = (1-dy) lnPL[i][j] + dy lnPL[i][j+1], turns each
// sample's read from six loads spread over two rows of the large lnPL
// table (four lnP corners plus the two stored axis values) into four
// loads of 32 contiguous bytes of a small array: the slice interleaves
// each column's axis value and z-interpolated lnP, holds
// 2 * lnPL_nk doubles (about 190 KB at the production refinement), and
// stays cache resident across the millions of samples of one call.
//
// Two deviations from p_lin_at_a's arithmetic, neither bitwise:
// 1. the sum log10k + shift is not the single rounded log10(k/coverH0),
//    so the weight dx (and, within one ulp of a cell edge, the index i)
//    can differ in the last bits;
// 2. the slice regroups the bilinear combination,
//    (1-dx)[(1-dy) t00 + dy t01] + dx[(1-dy) t10 + dy t11] instead of
//    the four-term sum, which rounds differently in the last bits.
// A caller that requires bitwise agreement with p_lin_at_a keeps the
// standard reader. The clamp, the exp and the unit factor are
// p_lin_at_a's. The linear table is stateless (no lazy first-call
// build), so the parallel rows only read initialized memory.
//
// Parameters:
//   a      - scale factor shared by every sample
//   nrow   - number of independent rows, at least 1
//   ncol   - samples per row, at least 1
//   log10k - [nrow][ncol] base-10 log wavenumbers before the shift
//   shift  - common addend: the physical wavenumber of sample m is
//            10^(log10k[m]+shift) in (c/H0)^-1, so a caller holding
//            log10(magnitude) passes -log10(f_K)
//   power  - [nrow][ncol] caller-owned output in (c/H0)^3, not
//            overlapping log10k
//
// Call outside an OpenMP region, after initializing the linear power
// table. Rows are divided among OpenMP workers; samples within a row are
// read in order by one worker.
// ---------------------------------------------------------------------------
void linear_power_logk_rows_cov(
    const double a,                  // shared scale factor
    const int nrow,                  // independent log-wavenumber rows
    const int ncol,                  // samples per row
    const double* const* log10k,     // base-10 logs before the shift
    const double shift,              // common addend to every sample
    double* const* power            // caller-owned output rows
  )
{
  if (nrow < 1
      || ncol < 1) {
    log_fatal("linear_power_logk_rows_cov needs positive row and column "
              "counts");
    exit(1);
  }

  // One scalar completes the shift: the standard reader's logarithm is
  // log10(k/coverH0) = log10k + shift - log10(coverH0).
  const double total_shift = shift - log10(cosmology.coverH0);

  // The z half of the bilinear read depends only on a: one bracket and
  // one weight serve every sample, exactly as in p_lin_at_a.
  const double z = 1.0 / a - 1.0;
  const int j = piecewise_index(z, cosmology.lnPL_z_nseg,
                                cosmology.lnPL_z_seg_start,
                                cosmology.lnPL_z_seg_len,
                                cosmology.lnPL_z_seg_xmin,
                                cosmology.lnPL_z_seg_inv_dx,
                                cosmology.lnPL_nz);
  const double zj  = cosmology.lnPL[cosmology.lnPL_nk][j  ];
  const double zj1 = cosmology.lnPL[cosmology.lnPL_nk][j+1];
  const double dy = (z - zj) / (zj1 - zj);

  // Collapse the z half once per call (the header explains the layout):
  // slice[2i] is column i's stored log10 k axis value and slice[2i+1] its
  // z-interpolated lnP, so one sample's bracket occupies one cache line.
  // The slice is built serially before the team starts and freed after
  // it ends; the workers only read it.
  double* slice = malloc(sizeof(double) * 2 * (size_t) cosmology.lnPL_nk);
  if (slice == NULL) {
    log_fatal("linear_power_logk_rows_cov: cannot allocate the z slice");
    exit(1);
  }
  for (int i=0; i<cosmology.lnPL_nk; i++) {
    slice[2*i]   = cosmology.lnPL[i][cosmology.lnPL_nz];
    slice[2*i+1] = (1.0-dy) * cosmology.lnPL[i][j]
                 +      dy  * cosmology.lnPL[i][j+1];
  }

  // One iteration fills one complete output row. A row depends on no
  // other row and the slice is read-only here, so no value depends on the
  // number of workers; the static schedule gives each worker a contiguous
  // block of rows.
  #pragma omp parallel for schedule(static)
  for (int row=0; row<nrow; row++) {
    for (int m=0; m<ncol; m++) {
      const double lg = log10k[row][m] + total_shift;
      int i = (int)((lg - cosmology.lnPL_log10k_min)
                    * cosmology.lnPL_log10k_inv_dx);
      if (i < 0)                       i = 0;
      if (i > cosmology.lnPL_nk - 2)   i = cosmology.lnPL_nk - 2;
      // node[0], node[2]: the bracketing axis values; node[1], node[3]:
      // their z-interpolated lnP. The 1D form of the bilinear read.
      const double* node = slice + 2*i;
      const double dx = (lg - node[0]) / (node[2] - node[0]);
      const double out_lnP = (1.0-dx) * node[1]
                           +      dx  * node[3];
      power[row][m] = exp(out_lnP)
                      / (cosmology.coverH0
                         * cosmology.coverH0
                         * cosmology.coverH0);
    }
  }
  free(slice);
}


// ---------------------------------------------------------------------------
// Tree-level averages fed by the log-domain reader, one block at a time.
//
// WHY THIS FUNCTION EXISTS - THE MEMORY-TRAFFIC ARGUMENT
// The cNG angle integrals need P_lin(|K+Q|) at every (pair, angle)
// sample: npair*nangle values, about 127 MB per radial shell at the
// production grids (8256 pairs x 1920 angles x 8 bytes). Computed as two
// separate stages, that table is written once by the power reader and
// read once by tree_averages_cov, and the reader also reads the equally
// large log-wavenumber table: three full passes over main memory per
// shell, roughly 380 MB, repeated for more than a thousand shells.
// tree_averages_cov itself is limited by that stream, not by arithmetic:
// adding workers speeds it little, because every worker waits on the
// same memory bus.
//
// The two stages do not need the whole table at once. Each pair's angle
// integral uses only its own row. So this driver walks the pairs in
// blocks: it evaluates the power for one block of rows into a small
// buffer, hands that block straight to the kernel, and reuses the buffer
// for the next block. A block of 128 pairs is 128 x nangle doubles,
// about 2 MB at the production angle rule - small enough to still sit in
// the processor's cache when the kernel reads back what the reader just
// wrote. The buffer is "hot": its second pass costs almost nothing. The
// only full pass over main memory that remains is the one unavoidable
// read of the log-wavenumber table. Three passes become one, and the
// 127 MB intermediate never exists.
//
// WHY THE RESULTS ARE BIT-FOR-BIT UNCHANGED
// Nothing here computes: both stages run unmodified.
//
// 1. The power values are produced by the same linear_power_logk_rows_cov
//    call as before, just for count rows at a time instead of npair. That
//    function treats every row independently, so splitting the rows into
//    calls cannot change any value. (Its small z slice is rebuilt per
//    block - about 12,000 multiply-adds against ten million per block -
//    and is identical every time, because it depends only on a.)
//
// 2. tree_averages_cov is lane safe under any split of the pair list,
//    because of how its SIMD lanes work. A v2d register holds two
//    doubles, called lanes 0 and 1, and the kernel gives each lane one
//    COMPLETE (K,Q) pair: at every angle node it broadcasts the shared
//    node values to both lanes (set1_pd), packs the two pairs' own
//    values side by side (set_pd), and updates two separate running
//    sums with one fused multiply-add,
//
//        lane 0:  sum_A = fma(P_A(node) * weight(node), G_A, sum_A)
//        lane 1:  sum_B = fma(P_B(node) * weight(node), G_B, sum_B)
//
//    No instruction ever adds lane 0 to lane 1, so a pair's angle sum
//    depends only on its own row - never on which pair happens to ride
//    in the neighboring lane. Worked example with six pairs, 0..5:
//
//        one whole call:        registers carry (0,1) (2,3) (4,5)
//        blocks of four:        call A carries (0,1) (2,3)
//                               call B carries (4,5)
//        blocks of three (odd): call A carries (0,1) (2,2*)
//                               call B carries (3,4) (5,5*)
//
//    The starred lane duplicates the block's last pair so the register
//    reads stay valid; the kernel computes it and throws it away. In
//    all three layouts pair 2's sum adds the same numbers in the same
//    order, so every layout returns identical doubles. The EVEN block
//    size below is therefore not a correctness requirement but an
//    efficiency one: with an odd size every block would pay one wasted
//    duplicate lane, with an even size only an odd npair's final block
//    can.
//
// The kernel also keeps its charter: tree_averages_cov still reads no
// core table and allocates nothing; every table read stays in this file.
//
// Parameters (shapes as in the two functions this driver calls):
//   npair   - number of (K,Q) pairs, at least 1
//   nangle  - number of angular nodes, at least 1
//   k, pk   - [2][npair] magnitudes and their linear power
//   corner, weight - [nangle] stable 1+cos(theta) and dtheta/pi weights
//   a       - scale factor of the shell
//   log10s  - [npair][nangle] base-10 logs of the internal momenta
//             before the shift (the run-constant magnitude table)
//   shift   - common addend, -log10(f_K) for this shell
//   average - [3][npair] caller-owned output rows: AvgP, AvgB, AvgT
//
// The block buffer is allocated once per call, outside both stages'
// OpenMP regions, and freed before returning. Call serially; the two
// stages parallelize themselves inside each block.
// ---------------------------------------------------------------------------
void tree_averages_logk_cov(
    const int npair,                 // number of K,Q pairs
    const int nangle,                // number of angular nodes
    const double* const* k,         // [2][npair] positive K and Q
    const double* const* pk,        // [2][npair] matching linear power
    const double* corner,           // stable 1+cos(theta)
    const double* weight,           // normalized dtheta/pi weights
    const double a,                  // scale factor of the shell
    const double* const* log10s,    // base-10 logs before the shift
    const double shift,              // common addend to every sample
    double* const* average           // three output averages
  )
{
  if (npair < 1
      || nangle < 1) {
    log_fatal("tree_averages_logk_cov needs positive pair and angle counts");
    exit(1);
  }

  // 128 pairs x nangle doubles is about 2 MB at the production angle
  // rule: large enough to occupy the OpenMP team in both stages, small
  // enough to stay cache resident between them. EVEN, so only an odd
  // npair's final block pays the kernel's duplicated tail lane (the
  // header's worked example shows the lanes; results are identical for
  // any block size).
  const int pair_block = 128;
  double** ps_block = (double**) malloc2d(pair_block, nangle);

  for (int start=0; start<npair; start+=pair_block) {
    const int count = npair-start < pair_block ? npair-start : pair_block;

    // Stage 1: the log-domain reader fills this block's power rows.
    // log10s+start passes the block's row pointers; values and rounding
    // are those of a whole-table call, row for row.
    linear_power_logk_rows_cov(a, count, nangle, log10s+start, shift,
                               ps_block);

    // Stage 2: the unmodified kernel consumes the block while it is
    // still cache resident. Column views select the block's pairs; the
    // angle rule is the full one, revalidated by the kernel per block.
    const double* k_block[2] = {k[0]+start, k[1]+start};
    const double* pk_block[2] = {pk[0]+start, pk[1]+start};
    double* average_block[3] = {average[0]+start, average[1]+start,
                                average[2]+start};
    tree_averages_cov(count, nangle, k_block, pk_block, corner, weight,
                      (const double* const*) ps_block, average_block);
  }
  free(ps_block);
}



// ============================================================================
// [SECTION] RADIAL SNAPSHOT: LENSING EFFICIENCY AND FIELD WINDOWS
// ============================================================================


// ---------------------------------------------------------------------------
// Integrate each catalog's lensing efficiency on a covariance-owned grid.
//
// In a flat universe a foreground shell at chi lenses a source at chi'
// with efficiency (1-chi/chi'). Average over the normalized source density:
//
//   g(chi) = integral_chi^infinity dchi' n_chi(chi') (1-chi/chi')
//          = A(chi) - chi B(chi),
//   A = integral_amin^a da' n_z(z(a'))/a'^2,
//   B = integral_amin^a da' n_z(z(a'))/[a'^2 chi(a')].
//
// Here n_chi dchi' = n_z dz' and |dz'/da'| = 1/a'^2. A source behind the
// shell (chi' > chi) has a' < a, so "behind" becomes the interval from the
// far end amin up to the shell's own a. The two cumulative integrals
// avoid a separate nested integral for every foreground shell. This is the
// same factorization as g_tomo/g_lens in redshift_spline.c, with a
// covariance-owned grid and no foreground cut. Sources beyond amin are not
// counted, so amin must enclose every catalog's redshift support.
//
// The upper endpoint is the observer, a=1, where chi=0 and the B integrand
// n_z/(a^2 chi) is undefined. This implementation assumes n_z/chi tends to
// zero at the observer, as for a catalog with a positive lower-redshift
// cutoff. It assigns that limit explicitly. Checking n_z(0)=0 below is a
// necessary endpoint check; it does not establish the limiting behavior.
// Assigning a finite value avoids the NaN that 0/0 would produce. That
// value enters only B at the final node, where B is multiplied by chi=0,
// so g at the observer is A(a=1) whatever finite value is assigned.
//
// Parameters:
//   amin    - far end of the integration volume, 0 < amin < 1
//   nwindow - number of uniform-a samples from amin to 1, at least 2
//   nlens   - number of lens catalogs; fields 0..nlens-1 use the lens n(z)
//   nfield  - total number of catalogs, lenses followed by sources
//
// Returns [field][nwindow] g (dimensionless) on the uniform grid
// a = amin + node*da, allocated by malloc2d; the caller frees it. Linear
// interpolation of g uses direct arithmetic on this uniform grid.
// The two SIMD lanes are A and B, not parts of one sum: cumulative order
// is preserved. Each worker owns one field. No core table is changed.
// ---------------------------------------------------------------------------
static double** lensing_efficiency_cov(
    const double amin, // far endpoint of the supplied radial volume
    const int nwindow, // covariance grid size
    const int nlens,   // number of lens catalogs
    const int nfield   // total number of catalogs
  )
{
  const double da = (1.0-amin)/(nwindow-1);

  // geometry[0][node] and geometry[1][node] hold a and chi(a) on the
  // uniform grid; they are shared across fields. Each efficiency row
  // belongs to one field.
  double** geometry = (double**) malloc2d(2, nwindow);
  double** efficiency = (double**) malloc2d(nfield, nwindow);

  // Sample a uniformly from the far end to the observer. The last sample is
  // set to a=1 exactly, because amin+(nwindow-1)*da can miss 1 by rounding.
  // chi(a) is read from the core distance table, in c/H0.
  for (int node=0; node<nwindow; node++) {
    const double a = node == nwindow-1 ? 1.0 : amin+node*da;
    geometry[0][node] = a;
    geometry[1][node] = chi(a);
  }

  // The observer's distance is set to exactly 0, rather than the table's
  // interpolated value. It is never used as a divisor below.
  geometry[1][nwindow-1] = 0.0;

  // Matter at distance chi lenses only galaxies farther away, at chi'>chi.
  // Each such galaxy contributes the geometric factor 1-chi/chi'. Averaging
  // this factor over a catalog gives two simpler integrals: A is the
  // fraction of galaxies behind chi; B is that same fraction weighted by
  // 1/chi'. Their combination A-chi*B is the lensing efficiency g(chi).
  //
  // The grid begins at the far boundary and moves toward the observer as a
  // increases. Each step includes another slice of galaxies in A and B.
  // Keeping the accumulated values avoids reintegrating all the farther
  // slices at every distance. One loop iteration constructs this whole g
  // row for one catalog, so different OpenMP workers can own different
  // catalogs independently. SIMD lane 0 accumulates A and lane 1
  // accumulates B: both use the same integration steps, but their
  // integrands differ by the factor 1/chi'.
  #pragma omp parallel for schedule(static)
  for (int field=0; field<nfield; field++) {
    // row is this catalog's output. restrict promises the compiler that no
    // other pointer used in this loop writes the same memory.
    double* restrict row = efficiency[field];

    // scalar: the same row with ordinary doubles, starting from
    // A = B = previous_A = previous_B = 0,
    //
    //   for (int node=0; node<nwindow; node++) {
    //     current_A = density;                   // n_z/a^2
    //     current_B = density*inverse_distance;  // n_z/(a^2 chi)
    //     if (node > 0) {
    //       A = fma(da/2, previous_A+current_A, A);
    //       B = fma(da/2, previous_B+current_B, B);
    //     }
    //     row[node] = A-distance*B;
    //     previous_A = current_A;
    //     previous_B = current_B;
    //   }
    //
    // density, inverse_distance and distance are the per-node values formed
    // at the top of the loop below. A counts the sources behind the shell;
    // B weights the same sources by 1/chi'. In the vector code lane 0 holds
    // every A quantity and lane 1 the matching B quantity: the lanes share
    // da/2 and the node order but are never added together.

    // vprevious = [previous_A, previous_B] = [0, 0] (setzero sets both lanes
    // to 0.0). These are the integrands n_z/a^2 and n_z/(a^2 chi) at the
    // preceding node; node 0 has no preceding node and does not use them.
    v2d vprevious = simde_mm_setzero_pd();

    // vintegral = [A, B] = [0, 0]: both cumulative integrals start at zero
    // at the far boundary a = amin, where no source has been counted yet.
    v2d vintegral = simde_mm_setzero_pd();

    // vhalf_step = [da/2, da/2]: set1 copies the scalar da/2 into both
    // lanes. A trapezoid is half the step size times the sum of the
    // integrand at its two endpoints.
    const v2d vhalf_step = simde_mm_set1_pd(da/2.0);

    // Moving to the next a sample adds the galaxies in one new radial slice.
    // Its contribution to A is the integral of n_z/a^2 across that interval;
    // for B the integrand is n_z/(a^2*chi). The 1/a^2 converts the catalog's
    // density per unit redshift into a density per unit scale factor.
    //
    // Each integrand is approximated as a straight line between its two
    // endpoint samples. The area below that line is the interval width da
    // times the mean endpoint value: da*(previous+current)/2. This is the
    // trapezoidal rule. The first node has no preceding interval, so both
    // integrals remain zero there. Later nodes add one such area per lane.
    // SIMD applies this same rule to A and B together without mixing them.
    // At each node, A-chi*B then gives g for matter at that node's distance.
    for (int node=0; node<nwindow; node++) {
      const double a = geometry[0][node];
      const double distance = geometry[1][node];
      const double z = 1.0/a-1.0;
      double density;

      // n_z is the catalog's normalized redshift distribution, read through
      // the core photo-z reader. Change variables from redshift to scale
      // factor: |dz/da| = 1/a^2.
      if (field < nlens) {
        density = nz_lens_photoz(z, field)/(a*a);
      } else {
        density = nz_source_photoz(z, field-nlens)/(a*a);
      }

      // At the observer chi=0, direct division would give 0/0 (or x/0 for a
      // catalog with sources at z=0). Use the assumed zero limit of n_z/chi
      // there, and reject a nonzero endpoint density (see the header).
      double inverse_distance = 0.0;
      if (node < nwindow-1) {
        inverse_distance = 1.0/distance;
      } else if (density != 0.0) {
        log_fatal("lensing_efficiency_cov needs n(z=0)=0; field %d "
                  "has density %g", field, density);
        exit(1);
      }

      // vcurrent = [current_A, current_B] = [n_z/a^2, n_z/(a^2 chi)], i.e.
      // [density, density*inverse_distance]. set_pd takes lane 1 first and
      // lane 0 last, so density lands in lane 0.
      const v2d vcurrent = simde_mm_set_pd(density*inverse_distance,
                                         density);

      if (node > 0) {
        // scalar: previous_A+current_A and previous_B+current_B.
        // vendpoints = vprevious + vcurrent lane by lane (add): the
        // trapezoid's endpoint sum in each integral, not an addition of A
        // to B.
        const v2d vendpoints = simde_mm_add_pd(vprevious, vcurrent);

        // scalar: A = fma(da/2, previous_A+current_A, A), and likewise B.
        // fmadd computes vhalf_step*vendpoints + vintegral in each lane,
        // appending one trapezoid area to each cumulative integral, with
        // one rounding per lane where the processor has a fused
        // instruction.
        vintegral = simde_mm_fmadd_pd(vhalf_step, vendpoints, vintegral);
      }

      double integral[2];

      // integral[0..1] = [A, B]: storeu writes lane 0 to integral[0] and
      // lane 1 to integral[1]. This ordinary stack array needs no special
      // vector alignment.
      simde_mm_storeu_pd(integral, vintegral);

      // scalar: row[node] = A-distance*B, the lensing efficiency g for
      // matter at this node's distance.
      row[node] = integral[0]-distance*integral[1];

      // The current integrands become the next trapezoid's old endpoints.
      vprevious = vcurrent;
    }
  }

  // geometry was scratch for this function; efficiency belongs to the
  // caller.
  free(geometry);
  return efficiency;
}


// ---------------------------------------------------------------------------
// Fill the distances and field windows of a snapshot at its radial samples.
//
// Both radial builders call this helper: radial_inputs_cov with
// Gauss-Legendre nodes for Limber quadrature, radial_logchi_cov with
// uniform ln(chi) samples for FFTLog. Their sample positions differ, but
// the efficiency interpolation and catalog conventions must not differ, so
// one function forms every window.
//
// On entry geometry[0] holds each sample's scale factor a and geometry[3]
// its quadrature weight in a (zero for the FFTLog samples, which carry no
// weight). The helper then writes, in c/H0 units,
//
//   geometry[1] = chi(a),  geometry[2] = f_K(chi),  geometry[3] *= |dchi/da|,
//
// so the last row becomes a positive distance weight. For each catalog it
// writes the windows, all in (c/H0)^-1:
//
//   lens:    window[0] = b1(z) n_l(z) H/H0                  galaxy density
//            window[1] = b_mag (3/2) Omega_m (f_K/a) g_l    magnification
//            window[3] = b2(z) n_l(z) H/H0                  quadratic bias
//            window[4] = bs2(z) n_l(z) H/H0                 tidal bias
//   source:  window[0] = n_s(z) H/H0                        source density
//            window[1] = (3/2) Omega_m (f_K/a) g_s          lensing W_kappa
//            window[2] = -C1(z) n_s(z) H/H0                 signed NLA
//
// H/H0 = dz/dchi in these units converts a density per unit redshift into
// a density per unit distance, and g is the efficiency of
// lensing_efficiency_cov. C1 = IA_A1_Z1 = A1(z) Omega_m c1rhocrit_ia/D(a)
// is positive for a positive alignment amplitude; with the minus sign in
// the window, a positive amplitude gives a negative GI correlation.
// Source window[2] stays zero unless include_ia is 1; lens window[2] is
// always zero.
//
// b2 and bs2 are the core's quadratic and tidal galaxy biases (gb2, gbs2
// of bias.c), in the normalization of the galaxy overdensity
// delta_g = b1 delta + (b2/2) delta^2 + (bs2/2) s^2, with s^2 the square
// of the tidal field. Windows 3 and 4 are zero when those biases are zero,
// and only oneloop_bias_spectra_cov reads them: every other consumer of
// the snapshot keeps the linear-bias model of window[0].
//
// Parameters:
//   radial     - snapshot with nnode, nlens, nsource, geometry[0] and
//                geometry[3] set; its other rows are overwritten
//   amin       - far end of the efficiency grid; amin <= a < 1 at every
//                sample
//   nwindow    - uniform-a samples of the cumulative efficiencies
//   include_ia - 1 adds the signed NLA window; 0 leaves window[2] zero
//
// The efficiency table is a temporary owned and freed here. Lazy core
// readers are warmed serially before the OpenMP loops.
// ---------------------------------------------------------------------------
static void fill_radial_cov(
    struct radial_cov* radial, // owner of geometry and window arrays
    const double amin,         // far edge for cumulative efficiencies
    const int nwindow,         // uniform-a efficiency samples
    const int include_ia       // include the signed linear alignment field
  )
{
  const int nfield = radial->nlens+radial->nsource;

  // --- 1. WARM THE CORE READERS BEFORE PARALLEL SAMPLING ---

  // Some core readers build lazy tables on their first call (the photo-z
  // n(z) readers, for example). Call each one once here, serially, before
  // OpenMP workers read them. This is setup work, so an explicit serial
  // pass is preferable to locks around individual table reads in the
  // integration loops. The first sample's distance, H/H0 and growth only
  // supply arguments for those calls: chi_all, hoverh0v2 and growfac keep
  // no static state.
  const double a_first = radial->geometry[0][0];
  const struct chis distance = chi_all(a_first);
  const double hubble_first = hoverh0v2(a_first, distance.dchida);
  const double growth_first = growfac(a_first);

  // Visit each catalog once to trigger any lazy core table construction.
  // This serial pass prepares the density, bias and IA readers for the
  // parallel loops: the n(z) readers behind W_gal and W_source are also the
  // ones that lensing_efficiency_cov calls from its workers. It does not
  // compute the covariance windows yet.
  for (int field=0; field<nfield; field++) {
    if (field < radial->nlens) {
      (void) W_gal(a_first, field, hubble_first);
      (void) gb1(1.0/a_first-1.0, field);
      (void) gb2(1.0/a_first-1.0, field);
      (void) gbs2(1.0/a_first-1.0, field);
      (void) gbmag(0.0, field);
    } else {
      const int source = field-radial->nlens;
      (void) W_source(a_first, source, hubble_first);
      if (include_ia) {
        (void) IA_A1_Z1(a_first, growth_first, source);
      }
    }
  }

  // --- 2. TABULATE THE LENSING EFFICIENCY ON A UNIFORM a GRID ---

  // g(chi) is needed at every radial sample, but its cumulative integrals
  // are cheapest on their own uniform grid in a: one ordered pass builds a
  // whole row, and the two grid points that bracket any a follow from one
  // multiplication by inv_da, with no search. The samples themselves are
  // the caller's (Gauss-Legendre nodes or uniform ln(chi) points).
  double** efficiency = lensing_efficiency_cov(amin, nwindow,
                                               radial->nlens, nfield);
  const double inv_da = (nwindow-1)/(1.0-amin);

  // --- 3. CONVERT THE MEASURE AND SAMPLE EACH FIELD WINDOW ---

  // Projected correlations add contributions along the line of sight,
  // weighted by how strongly each catalog responds at that distance.
  // These windows include local galaxy density, lensing by foreground
  // matter, and intrinsic alignment. They must refer to the same shells
  // so their products later describe correlations of the same matter.
  // One iteration is one radial sample: its worker fills the geometry and
  // every catalog's windows there, so no two workers write the same entry.
  // Convert the sample's da weight with |dchi/da|: increasing a moves
  // toward the observer, but the physical integration measure must remain
  // a positive distance interval.
  #pragma omp parallel for schedule(static)
  for (int node=0; node<radial->nnode; node++) {
    const double a = radial->geometry[0][node];
    const double z = 1.0/a-1.0;
    const struct chis distance = chi_all(a);
    const double fk = f_K(distance.chi);

    // f_K must be positive (a sample at the observer would give 0), and so
    // must |dchi/da| (a nonpositive value means a broken distance table).
    // The negated comparisons !(x > 0) also reject NaN, for which every
    // comparison is false.
    if (!(fk > 0.0)
        || !(distance.dchida > 0.0)) {
      log_fatal("radial_inputs_cov: nonpositive distance or measure at "
                "a=%g; check the distance table and panel endpoints", a);
      exit(1);
    }

    // H/H0 = 1/(a^2 |dchi/da|) from the same table lookup, and the growth
    // factor D(a), normalized to D(1)=1, for the NLA amplitude.
    const double hubble = hoverh0v2(a, distance.dchida);
    const double growth = growfac(a);

    // chi_all returns the positive magnitude |dchi/da|. Thus multiplying
    // the positive Gaussian da weight gives a positive dchi measure (an
    // FFTLog sample's zero weight stays zero).
    radial->geometry[1][node] = distance.chi;
    radial->geometry[2][node] = fk;
    radial->geometry[3][node] *= distance.dchida;

    // Direct indexing on the uniform efficiency grid: position counts grid
    // steps from amin, its integer part selects the left grid point and the
    // fraction blends the two neighbours. (int) truncates toward zero; for
    // amin <= a < 1, left and left+1 stay inside the table.
    // prefactor = 1.5 Omega_m f_K/a is the lensing-kernel factor
    // (3/2) Omega_m (H0/c)^2 f_K/a in c/H0 units; times g it gives a
    // lensing window in (c/H0)^-1.
    const double position = (a-amin)*inv_da;
    const int left = (int) position;
    const double fraction = position-left;
    const double prefactor = 1.5*cosmology.Omega_m*fk/a;

    // At this fixed distance, interpolate each catalog's efficiency and
    // form its physical windows (table in the header). Store contributions
    // separately because their ell-dependent factors are supplied later, by
    // limber_spectra_cov or by the FFTLog kernels.
    for (int field=0; field<nfield; field++) {
      // Interpolate g between its two bracketing uniform-a samples.
      const double g = (1.0-fraction)*efficiency[field][left]
                       +fraction*efficiency[field][left+1];

      if (field < radial->nlens) {
        // Galaxy density b1 n_l H/H0 and magnification b_mag W_mag have
        // different ell factors; keep their radial windows separate until
        // the spectrum stage. The quadratic and tidal bias densities
        // b2 n_l H/H0 and bs2 n_l H/H0 share the same n_l H/H0; they
        // multiply one-loop templates instead of the matter power.
        const double density = W_gal(a, field, hubble);
        radial->window[0][field][node] = gb1(z, field)*density;
        radial->window[1][field][node] = gbmag(z, field)
                                       *prefactor*g;
        radial->window[3][field][node] = gb2(z, field)*density;
        radial->window[4][field][node] = gbs2(z, field)*density;
      } else {
        // Store the source density n_s H/H0 (the radial weight of the TATT
        // terms in ia_cov.c, also exported for audits), the lensing window
        // W_kappa, and the signed NLA term -C1 n_s H/H0 in its own role.
        const int source = field-radial->nlens;
        radial->window[0][field][node] = W_source(a, source, hubble);
        radial->window[1][field][node] = prefactor*g;
        if (include_ia) {
          radial->window[2][field][node] = -W_source(a, source, hubble)
                                         *IA_A1_Z1(a, growth, source);
        }
      }
    }
  }

  free(efficiency);
}


// ---------------------------------------------------------------------------
// Sample the radial windows on a common, positive quadrature rule.
//
// A projected field is A(n) = integral dchi W_A(chi) delta(chi*n).
// In the Limber approximation its cross spectrum with B is
//
//   C_AB(ell) = integral dchi W_A W_B P((ell+1/2)/f_K, a)/f_K^2.
//
// See Krause & Eifler, arXiv:1601.05779, Eqs. 4-7. Their flat-sky
// expression uses k = ell/chi; the core and this file use the extended
// Limber wavenumber (ell+1/2)/f_K (LoVerde & Afshordi, arXiv:0809.5112),
// and limber_spectra_cov below adds the core's spin and magnification
// transfer factors.
//
// Use the same chi nodes for every pair, including pairs absent from the
// measured data vector. At each node W_A W_B is an outer product. With
// positive integration weights and P >= 0, their sum is a positive
// semidefinite field matrix. A negative cross spectrum is allowed: two
// windows can have opposite signs because of magnification or alignment.
//
// The input intervals are in scale factor, which runs in the opposite
// direction to distance. geometry[3] stores da * |dchi/da|, a positive
// distance measure. It does not contain f_K^-2: SSC and cNG need different
// distance powers and reuse this same geometry.
//
// Separate window contributions have distinct multipole dependence:
//   lens:   window[0] = b1 n_l(z) H/H0; window[1] = b_mag W_mag;
//   source: window[1] = W_kappa; window[2] = -C1 n_s(z) H/H0.
// Source window[0] stores the unbiased source density n_s(z) H/H0. It does
// not enter the Limber shear spectrum, whose NLA part already carries it in
// window[2]; ia_cov.c uses it as the radial weight of the higher-order TATT
// terms, and the interfaces export it for audits. All windows have units
// of inverse distance. This builder supplies the linear/NLA part of IA;
// higher-order TATT correlations are added separately by ia_cov.c. They
// are quadratic in the tidal and density fields, so a single deterministic
// field window cannot represent them. The galaxy windows here are linear
// in the matter density; the one-loop galaxy-bias terms are added
// separately by oneloop_bias_spectra_cov, from the bias windows 3 and 4.
//
// Parameters and ownership:
//   npanel, a_edges - common integration intervals, strictly inside (0,1)
//   nquad - Gauss-Legendre nodes per interval, one of the tabulated sizes
//           64, 96, 128, 256, 512, 1024, 2048, 4096, 8192
//   nwindow - number of uniform-a samples for cumulative efficiencies
//   include_ia - include the core's NLA amplitude when 1; otherwise omit IA
// Returns a heap-owned snapshot; release it with free_radial_cov.
//
// There is no persistent covariance cache. Recreate the snapshot after a
// cosmology, photo-z, bias, IA or redshift-distribution change. Some core
// readers build lazy tables on first use; fill_radial_cov calls them
// serially before its parallel loops. Call this entry outside any OpenMP
// region.
// ---------------------------------------------------------------------------
struct radial_cov* radial_inputs_cov(
    const int npanel,       // number of common integration panels
    const double* a_edges,  // increasing scale-factor edges
    const int nquad,        // tabulated nodes per panel
    const int nwindow,      // cumulative lensing-efficiency grid size
    const int include_ia    // whether to include NLA
  )
{
  // Reject unsupported inputs before allocating anything. The efficiency
  // factor 1-chi/chi' holds only in flat space; Gauss-Legendre rules exist
  // only at the tabulated sizes; the IA window here is the linear (NLA)
  // part shared by NLA and TATT; and the panel edges must increase
  // strictly inside (0,1), so that every node lies between the far end and
  // the observer.
  if (npanel < 1
      || nwindow < 2
      || redshift.clustering_nbin < 1
      || redshift.shear_nbin < 1
      || (include_ia != 0
          && include_ia != 1)) {
    log_fatal("radial_inputs_cov needs panels, nwindow >= 2, bins, "
              "and include_ia = 0 or 1");
    exit(1);
  }
  if (fabs(cosmology.Omega_m+cosmology.Omega_v-1.0) > 1.e-10) {
    log_fatal("radial_inputs_cov lensing efficiency requires flat geometry");
    exit(1);
  }
  if (nquad != 64
      && nquad != 96
      && nquad != 128
      && nquad != 256
      && nquad != 512
      && nquad != 1024
      && nquad != 2048
      && nquad != 4096
      && nquad != 8192) {
    log_fatal("radial_inputs_cov: nquad=%d is not a supported "
              "fixed rule (64,96,128,256,512,1024,2048,4096,8192)", nquad);
    exit(1);
  }
  if (include_ia
      && nuisance.IA_MODEL != IA_MODEL_NLA
      && nuisance.IA_MODEL != IA_MODEL_TATT) {
    log_fatal("radial_inputs_cov supports NLA/TATT linear windows; IA_MODEL=%d",
              nuisance.IA_MODEL);
    exit(1);
  }
  for (int edge=0; edge<=npanel; edge++) {
    if (!isfinite(a_edges[edge])
        || a_edges[edge] <= 0.0
        || a_edges[edge] >= 1.0
        || (edge > 0
            && a_edges[edge] <= a_edges[edge-1])) {
      log_fatal("radial_inputs_cov: edge %d = %g; supply strictly "
                "increasing scale factors inside (0,1)",
                edge, a_edges[edge]);
      exit(1);
    }
  }

  // --- 1. ALLOCATE THE SHARED RADIAL SNAPSHOT ---

  // Allocate by physical role, rather than one allocation per field.
  // malloc2d/malloc3d (basics.c) return one block whose rows are padded to
  // 64-byte boundaries: free each parent once, and never treat the block
  // as one flat array for memset or memcpy (zero3d skips the padding).
  struct radial_cov* radial = malloc(sizeof(*radial));
  if (radial == NULL) {
    log_fatal("radial_inputs_cov: cannot allocate the snapshot");
    exit(1);
  }

  radial->nnode = npanel*nquad;
  radial->nlens = redshift.clustering_nbin;
  radial->nsource = redshift.shear_nbin;
  const int nfield = radial->nlens + radial->nsource;

  // geometry[4][node]: a, chi, f_K and dchi weight. window[5][field][node]:
  // density, lensing/magnification, signed intrinsic alignment, and the
  // quadratic- and tidal-bias densities (spectra_cov.h). Roles a field does
  // not use (lens window[2], source window[2] without IA, source windows 3
  // and 4) stay zero.
  radial->geometry = (double**) malloc2d(4, radial->nnode);
  radial->window = (double***) malloc3d(5, nfield, radial->nnode);
  zero3d(radial->window, 5, nfield, radial->nnode);

  // --- 2. MAP THE GAUSS-LEGENDRE RULE ONTO EVERY PANEL ---

  // An n-node Gauss-Legendre rule integrates polynomials of degree 2n-1
  // exactly on one interval. Mapping the tabulated rule onto each panel
  // [a_edges[p], a_edges[p+1]] gives nodes strictly inside the panel and
  // positive weights that already include the panel half-width.
  gsl_integration_glfixed_table* rule = malloc_gslint_glfixed(nquad);

  // Map the Gaussian rule onto each scale-factor panel, then concatenate
  // its samples into geometry's common a and da-weight rows.
  for (int panel=0; panel<npanel; panel++) {
    // One node supplies an a value and its positive integration weight;
    // its global index identifies the same shell in every field window.
    for (int node=0; node<nquad; node++) {
      const int index = panel*nquad+node;
      double a;       // quadrature abscissa in this scale-factor interval
      double weight;  // positive quadrature weight, including interval width

      gsl_integration_glfixed_point(a_edges[panel], a_edges[panel+1],
                                    node, &a, &weight, rule);

      // Retain the da weight here; the next stage multiplies by |dchi/da|.
      radial->geometry[0][index] = a;
      radial->geometry[3][index] = weight;
    }
  }

  gsl_integration_glfixed_table_free(rule);

  // --- 3. CONVERT THE WEIGHTS AND FILL THE FIELD WINDOWS ---

  // a_edges[0] is the far end of the volume, so it also starts the
  // cumulative efficiency grid.
  fill_radial_cov(radial, a_edges[0], nwindow, include_ia);
  return radial;
}


// ---------------------------------------------------------------------------
// Sample the same field windows uniformly in ln(chi) for FFTLog.
//
// FFTLog transforms a function sampled at equal steps in ln(chi):
//
//   chi_n = chi_min exp(n*step),  step = ln(chi_max/chi_min)/(nchi-1),
//
// with chi_max = chi(amin) and n = 0..nchi-1. The lower distance is
// positive because ln(0) is undefined. Its omitted foreground must be
// tested by lowering chi_min; it is not the zero guard of the FFT. The far
// distance corresponds to amin, which must enclose every catalog.
// Multiplying the number of intervals nchi-1 by an integer keeps every
// old node. The shared fill keeps the Limber and non-Limber window models
// identical.
//
// These samples carry no quadrature weight: FFTLog supplies its own
// dln(chi) measure, so geometry[3] is zero.
//
// Parameters:
//   amin       - scale factor at the far boundary
//   chi_min    - near distance in c/H0, 0 < chi_min < chi(amin)
//   nchi       - samples including both endpoints, at least 3
//   nwindow    - uniform-a samples of the cumulative efficiencies, >= 2
//   include_ia - 1 adds the signed NLA source window
// Returns a heap-owned snapshot; release it with free_radial_cov. Call
// outside OpenMP, after radial_inputs_cov has validated the cosmology and
// catalog setup; this builder checks only its own grid.
// ---------------------------------------------------------------------------
struct radial_cov* radial_logchi_cov(
    const double amin,      // far boundary in scale factor
    const double chi_min,   // positive near distance, in c/H0
    const int nchi,         // physical samples, including both endpoints
    const int nwindow,      // uniform-a efficiency grid
    const int include_ia    // signed NLA source contribution
  )
{
  // The far boundary sets the largest distance. Reject fewer than 3
  // samples, a nonpositive near distance (ln(0) is undefined), a near
  // distance at or beyond the far one, and fewer than 2 efficiency samples.
  const double chi_max = chi(amin);
  if (nchi < 3
      || chi_min <= 0.0
      || chi_min >= chi_max
      || nwindow < 2) {
    log_fatal("radial_logchi_cov: invalid distance or window grid");
    exit(1);
  }

  // This discarded call only evaluates a(chi_min); a_chi keeps no static
  // state (cosmo3D.c), so it prepares nothing for the serial loop below.
  (void) a_chi(chi_min);

  // Same grouped layout as radial_inputs_cov: geometry[4][node] and
  // window[5][field][node], with unused roles left at zero.
  struct radial_cov* radial = malloc(sizeof(*radial));
  if (radial == NULL) {
    log_fatal("radial_logchi_cov: workspace allocation failed");
    exit(1);
  }
  radial->nnode = nchi;
  radial->nlens = redshift.clustering_nbin;
  radial->nsource = redshift.shear_nbin;
  const int nfield = radial->nlens+radial->nsource;
  radial->geometry = (double**) malloc2d(4, nchi);
  radial->window = (double***) malloc3d(5, nfield, nchi);
  zero3d(radial->window, 5, nfield, nchi);

  // Logarithmic step between neighbouring samples.
  const double step = log(chi_max/chi_min)/(nchi-1);

  // FFTLog integrates in log distance itself, so these samples carry no
  // Gaussian da measure. Store zero in that unused row explicitly.
  // The far node takes amin exactly, rather than the a_chi inversion of
  // chi_min*exp((nchi-1)*step), which can differ from amin by rounding;
  // the far sample then coincides with the first efficiency grid point.
  for (int node=0; node<nchi; node++) {
    const double distance = chi_min*exp(node*step);
    radial->geometry[0][node] = node == nchi-1 ? amin : a_chi(distance);
    radial->geometry[3][node] = 0.0;
  }

  // Distances, measure conversion and windows, as for the Limber nodes.
  fill_radial_cov(radial, amin, nwindow, include_ia);
  return radial;
}


// Release the grouped arrays and their owner. Each grouped array is one
// malloc2d/malloc3d block, so one free releases all its rows; no other
// object owns a row.
void free_radial_cov(struct radial_cov* radial)
{
  free(radial->window);
  free(radial->geometry);
  free(radial);
}



// ============================================================================
// [SECTION] ALL-PAIRS LIMBER SPECTRA
// ============================================================================


// ---------------------------------------------------------------------------
// Build all Limber field spectra, with common windows and radial nodes.
//
// For fields A and B with complete windows W_A, W_B at multipole ell,
//
//   C_AB(ell) = integral dchi W_A W_B P(k,a)/f_K^2,  k = (ell+1/2)/f_K,
//
// which on the snapshot's radial rule becomes, for each output,
//
//   C_AB = sum_p [dchi_p P_p/f_K,p^2] W_A,p W_B,p,
//
// with p the radial node and dchi_p = geometry[3][p].
//
// First precompute the spectrum and complete field windows. This hoists
// power-spectrum reads and RSD evaluations out of the field-pair loop.
// Then integrate pairs in groups of two. The two SIMD lanes accumulate
// two different spectra, each in increasing radial-node order. Threads
// own different (ell, pair-group) outputs; there is no cross-thread sum.
//
// The complete windows multiply the base windows by angular factors:
//
//   lens:   W = b1 n_l H/H0 + [ell(ell+1)/(ell+1/2)^2] b_mag W_mag
//               (+ the RSD window when requested)
//   source: W = [sqrt((ell-1) ell (ell+1) (ell+2))/(ell+1/2)^2]
//               (W_kappa - C1 n_s H/H0)
//
// Magnification and shear come from angular derivatives of the lensing
// potential. W_kappa assumes the flat-sky value (k f_K)^2 = (ell+1/2)^2
// for those derivatives, which cancels the 1/k^2 of the Poisson equation.
// On the sphere the convergence that magnifies lens counts carries the
// Laplacian eigenvalue ell(ell+1), and the shear carries the spin-2 factor
// sqrt((ell+2)!/(ell-2)!); the ratios above restore them and tend to 1 at
// high ell. The core applies the shear factor to the NLA term too, since
// the alignment is also a spin-2 field.
//
// If requested, the same RSD window is added to each lens wherever that
// field appears, including lens-source pairs. Using RSD for gg but not
// gs would describe two different random fields with the same name and
// would lose the positive-semidefinite construction.
// These are the core C_ell conventions. Converting to observed shear
// spectra for unit-normalized spin kernels is a separate per-field
// rescaling; never apply that signal rescaling to white shape noise.
//
// Parameters:
//   radial - current snapshot from radial_inputs_cov
//   nell, ell - caller's multipole grid, finite and >= 1 (does not alter
//               Ntable)
//   linear - power model: 0 = the core's run-mode Pdelta(k,a);
//            1 = p_lin(k,a); 2 = D(a)^2 p_lin(k,1), the separable linear
//            field that the non-Limber correction subtracts
//   include_rsd - one common lens-field choice, 0 or 1
//   spectra - caller-owned triangular pair rows, each of length nell
//
// Output and inputs must not overlap. Array entries are overwritten.
// Scratch is grouped by role and released before return. This function
// reads the current core P tables, so the snapshot and core must describe
// the same cosmology. There is no covariance cache or hidden boost.
// ---------------------------------------------------------------------------
void limber_spectra_cov(
    const struct radial_cov* radial, // shared radial rule and base windows
    const int nell,                 // number of multipole nodes
    const double* ell,              // supplied multipoles
    const int linear,               // power mode 0, 1 or 2 (see above)
    const int include_rsd,          // common lens RSD choice
    double* const* spectra          // triangular pair output
  )
{
  // Reject an empty grid, an unknown power or RSD mode, and multipoles
  // below 1, where the spin-2 factor would take the square root of a
  // negative number.
  if (nell < 1
      || (linear != 0
          && linear != 1
          && linear != 2)
      || (include_rsd != 0
          && include_rsd != 1)) {
    log_fatal("limber_spectra_cov needs nell > 0, power mode 0..2 and RSD 0/1");
    exit(1);
  }
  for (int index=0; index<nell; index++) {
    if (!isfinite(ell[index])
        || ell[index] < 1.0) {
      log_fatal("limber_spectra_cov: ell[%d]=%g; need finite ell >= 1",
                index, ell[index]);
      exit(1);
    }
  }

  // nnode radial nodes; nfield fields, lenses first; npair unordered pairs
  // (A,B) with A <= B, the triangle of the symmetric field matrix.
  const int nnode = radial->nnode;
  const int nfield = radial->nlens+radial->nsource;
  const int npair = nfield*(nfield+1)/2;

  // --- 1. ENUMERATE ALL FIELD PAIRS AND PREPARE SHARED READERS ---

  // Include pairs excluded from the data vector: the Wick contractions
  // of retained observables can still need those cross spectra. pairs[0]
  // and pairs[1] hold the two field indices of each output row, in the
  // i-major order (0,0),(0,1),...,(0,n-1),(1,1),... of the header.
  int** pairs = (int**) malloc2d_int(2, npair);
  int pair = 0;

  for (int first=0; first<nfield; first++) {
    for (int second=first; second<nfield; second++) {
      pairs[0][pair] = first;
      pairs[1][pair] = second;
      pair++;
    }
  }

  // The two power roles share an allocation, power[2][nnode][nell]. Role 0
  // holds k during the batched P(k,a) read, then becomes dchi*P/f_K^2;
  // role 1 stores P. Node-major rows let Pdelta_at_a locate the redshift
  // bracket once per row. Windows, window[nell][nfield][nnode], are
  // ell-major for contiguous integration over distance.
  double*** power = (double***) malloc3d(2, nnode, nell);
  double*** window = (double***) malloc3d(nell, nfield, nnode);

  // Touch shared core state serially before the workers read it. The
  // Pdelta call writes the run-mode latch when the run mode is "linear"
  // (see power_rows_cov); p_lin keeps no state, so its call is only a
  // harmless read. W_RSD reads the lens n(z), whose reader can build lazy
  // tables on first use; a_chi keeps no static state.
  if (linear) {
    (void) p_lin(1.0, radial->geometry[0][0]);
  } else {
    (void) Pdelta(1.0, radial->geometry[0][0]);
  }
  if (include_rsd) {
    const double a = radial->geometry[0][0];
    (void) a_chi(radial->geometry[1][0]);
    (void) W_RSD(ell[0]+0.5, a, a, 0);
  }

  // Largest tabulated distance, converted from Mpc/h to c/H0. The shifted
  // RSD shell below must stay inside the table.
  const double chi_max = cosmology.chi[1][cosmology.chi_nz-1]
                         /cosmology.coverH0;

  // --- 2. READ POWER ONCE PER NODE AND FORM COMPLETE FIELD WINDOWS ---

  // Under Limber, an angular mode ell samples the matter spectrum near
  // k=(ell+1/2)/f_K. At a fixed distance this P(k,a) is common to every
  // catalog pair; only their windows differ. Read all these powers once
  // per distance, and attach the projection weight dchi/f_K^2.
  // The spin, magnification and RSD factors then turn the base windows
  // into complete fields at each ell. One iteration is one distance
  // (node): it writes only that node's power and window entries, and the
  // next loop reuses its results across every catalog pairing.
  #pragma omp parallel for schedule(static)
  for (int node=0; node<nnode; node++) {
    const double a = radial->geometry[0][node];
    const double fk = radial->geometry[2][node];

    // This node's rows of the two power roles. restrict promises that they
    // do not overlap, so a store to one never forces a reload of the other.
    double* restrict k = power[0][node];
    double* restrict pk = power[1][node];

    // Limber wavenumbers k = (ell+1/2)/f_K of every multipole at this
    // node, in (c/H0)^-1. All share one redshift bracket in the power
    // reader.
    for (int index=0; index<nell; index++) {
      k[index] = (ell[index]+0.5)/fk;
    }

    if (linear == 2) {
      // Subtract precisely the separable field used by FFTLog: its
      // anchor spectrum at a=1 times the same supplied growth squared,
      // with growfac normalized to D(1)=1. Using p_lin(k,a) instead would
      // leave a scale-dependent residual.
      p_lin_at_a(1.0, k, nell, pk);
      const double growth = growfac(a);
      for (int index=0; index<nell; index++) {
        pk[index] *= growth*growth;
      }
    } else if (linear) {
      p_lin_at_a(a, k, nell, pk);
    } else {
      Pdelta_at_a(a, k, nell, pk);
    }

    // For each ell at this distance, combine its angular factors (see the
    // header) with the base windows and optional RSD. Retain dchi*P/f_K^2
    // once per ell, instead of recomputing it for every pair of catalogs.
    for (int index=0; index<nell; index++) {
      // l is the multipole and ell_shift = l+1/2. The angular factors are
      // magnification = l(l+1)/(l+1/2)^2 and
      // shear = sqrt((l-1) l (l+1) (l+2))/(l+1/2)^2; l >= 1 keeps the
      // square root real.
      const double l = ell[index];
      const double ell_shift = l+0.5;
      const double magnification = l*(l+1.0)/(ell_shift*ell_shift);
      const double shear = sqrt((l-1.0)*l*(l+1.0)*(l+2.0))
                           /(ell_shift*ell_shift);

      // The k row has served the power read. Reuse it for the common
      // positive integration factor dchi*P/f_K^2 at each multipole.
      k[index] = radial->geometry[3][node]*pk[index]/(fk*fk);

      // The Limber form of the RSD window (W_RSD, radial_weights.c)
      // combines the lens density at chi and at a second shell
      // chi*(ell+3/2)/(ell+1/2). If that shell is beyond the supplied
      // distance table, stop rather than silently discarding its
      // contribution or extrapolating a_chi.
      double a_shift = a;
      if (include_rsd) {
        const double chi_shift = radial->geometry[1][node]
                                 *(ell_shift+1.0)/ell_shift;
        if (chi_shift > chi_max) {
          log_fatal("limber_spectra_cov: RSD distance %g exceeds %g; "
                    "extend the distance table", chi_shift, chi_max);
          exit(1);
        }
        a_shift = a_chi(chi_shift);
      }

      // Combine contributions to one observed field before pairing fields.
      // This retains density-magnification and lensing-IA cross terms.
      for (int field=0; field<nfield; field++) {
        if (field < radial->nlens) {
          double value = radial->window[0][field][node]
                         +magnification*radial->window[1][field][node];
          if (include_rsd) {
            value += W_RSD(ell_shift, a, a_shift, field);
          }
          window[index][field][node] = value;
        } else {
          window[index][field][node] = shear
              *(radial->window[1][field][node]
                +radial->window[2][field][node]);
        }
      }
    }
  }

  // --- 3. INTEGRATE EACH FIELD PAIR ON THE SAME RADIAL RULE ---

  // In Limber, a shared shell contributes W_A*W_B*P*dchi/f_K^2 to C_AB.
  // Multiplying the two windows selects matter to which both fields
  // respond; summing over shells gives their projected cross spectrum.
  // Use the same positive integration rule for every pair so this remains
  // a consistent matrix of field correlations, including unmeasured pairs
  // needed by the covariance. collapse(2) lets OpenMP divide the combined
  // (ell, pair-group) iterations among workers; one iteration is one ell
  // and two pairs. SIMD shares the matter power and radial weight, but
  // each lane keeps its own pair of windows and its own complete spectrum
  // through storage.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int index=0; index<nell; index++) {
    // At this ell, process two catalog pairs together using the prepared
    // windows. Lane 0 and lane 1 each produce one independent spectrum.
    for (int first_pair=0; first_pair<npair; first_pair+=2) {
      // Repeat the last valid pair in an unused lane of an odd-sized
      // block. Only valid outputs are stored, so no padding is required.
      const int next_pair = first_pair+1 < npair ? first_pair+1 : npair-1;

      // Windows of the two fields of each pair at this ell: left0/right0
      // for first_pair, left1/right1 for next_pair. restrict promises the
      // compiler that no store in this loop changes these rows; two of
      // them may be the same row, which is allowed because none is written.
      const double* restrict left0 = window[index][pairs[0][first_pair]];
      const double* restrict right0 = window[index][pairs[1][first_pair]];
      const double* restrict left1 = window[index][pairs[0][next_pair]];
      const double* restrict right1 = window[index][pairs[1][next_pair]];

      // scalar: for each catalog pair p = first_pair (lane 0) and
      // p = next_pair (lane 1),
      //   sum = 0;
      //   for (int node=0; node<nnode; node++) {
      //     product = window[index][pairs[0][p]][node]
      //               *window[index][pairs[1][p]][node];
      //     sum = fma(product, power[0][node][index], sum);
      //   }
      //   spectra[p][index] = sum;
      // power[0][node][index] already holds dchi P/f_K^2, so the sum is the
      // radial quadrature of C_AB. SIMD carries the two pairs' sums in
      // separate lanes; it never adds one pair's sum to the other's.

      // vtotal = [C(first_pair), C(next_pair)] = [0, 0] (setzero sets both
      // lanes to 0.0). Neither lane is part of the other's sum.
      v2d vtotal = simde_mm_setzero_pd();

      // Walk all radial nodes in increasing order. SIMD updates the two
      // spectra together using the same dchi*P/f_K^2, but their own W_A W_B.
      for (int node=0; node<nnode; node++) {
        // vleft = [left0[node], left1[node]]: the first field's window of
        // first_pair (lane 0) and of next_pair (lane 1) at this node.
        // set_pd takes lane 1 first and lane 0 last.
        const v2d vleft = simde_mm_set_pd(left1[node], left0[node]);

        // vright = [right0[node], right1[node]]: the second field's window
        // of each pair, in the same lane order as vleft.
        const v2d vright = simde_mm_set_pd(right1[node], right0[node]);

        // vmeasure = [w, w] with w = power[0][node][index] = dchi P/f_K^2
        // at this node and ell: set1 copies the shared weight into both
        // lanes.
        const v2d vmeasure = simde_mm_set1_pd(power[0][node][index]);

        // scalar: product = window of field A times window of field B.
        // mul multiplies lane by lane: vproduct = [W_A W_B of first_pair,
        // W_A W_B of next_pair], each rounded once, with no cross-lane
        // products.
        const v2d vproduct = simde_mm_mul_pd(vleft, vright);

        // scalar: sum = fma(product, power[0][node][index], sum) per pair.
        // fmadd computes vproduct*vmeasure + vtotal in each lane, adding
        // this node's W_A W_B dchi P/f_K^2 to that pair's own radial sum,
        // with one rounding per lane where the processor has a fused
        // instruction.
        vtotal = simde_mm_fmadd_pd(vproduct, vmeasure, vtotal);
      }

      double result[2];

      // result[0..1] = [C(first_pair), C(next_pair)]: storeu writes lane 0
      // to result[0] and lane 1 to result[1]. This ordinary two-double
      // array needs no special vector alignment.
      simde_mm_storeu_pd(result, vtotal);

      // scalar: spectra[p][index] = sum. Store only real pairs: for an odd
      // pair count the last group's lane 1 repeats first_pair and is
      // discarded, so each output row is written once.
      spectra[first_pair][index] = result[0];
      if (first_pair+1 < npair) {
        spectra[first_pair+1][index] = result[1];
      }
    }
  }

  // Release the scratch: one free per grouped array.
  free(window);
  free(power);
  free(pairs);
}



// ============================================================================
// [SECTION] ONE-LOOP GALAXY BIAS IN THE LENS SPECTRA
// ============================================================================


// ---------------------------------------------------------------------------
// Add the one-loop galaxy-bias terms to every spectrum with a lens field.
//
// PHYSICAL MODEL
// Beyond linear order a galaxy sample responds to the matter density
// delta through more than its linear bias b1. The real-space expansion of
// the data vector (cosmo2D.c), in the normalization of Saito et al.,
// arXiv:1405.1447, is
//
//   delta_g = b1 delta + (b2/2) delta^2 + (bs2/2) s^2.
//
// b2 is the quadratic bias, bs2 the tidal bias, and s^2 = s_ij s_ij is the
// square of the trace-free tidal field, s_ij = (k_i k_j/k^2 - delta_ij/3)
// delta in Fourier space. The third-order bias b3 and the k^2 counterterm
// bK are not included here; the function stops if either is nonzero.
//
// Two such fields correlate through b1 b1 P, already in the spectra of
// limber_spectra_cov, and through one-loop correlators of the operators.
// FAST-PT computes those at a = 1 from the linear power (McEwen et al.,
// arXiv:1603.04826):
//
//   P_d1d2 = <delta delta^2>,      P_d2d2 = <delta^2 delta^2>,
//   P_d1s2 = <delta s^2>,          P_d2s2 = <delta^2 s^2>,
//   P_s2s2 = <s^2 s^2>.
//
// Each correlator contains two linear spectra, so at scale factor a it is
// D(a)^4 times the a = 1 table; D is the linear growth with D(1) = 1. The
// three quadratic correlators tend to constants as k -> 0. The core
// subtracts those limits, 2 sigma4, (4/3) sigma4 and (8/9) sigma4 with
// sigma4 = P_d2d2(k_min)/2, so that the bias operators add no constant,
// shot-noise-like power on large scales (cosmo2D.c explains the limits).
//
// Collecting the coefficients of delta_g, the one-loop part of the cross
// power of galaxy bins i and j is
//
//   P_ij^loop = D^4 [ (1/2)(b1_i b2_j + b2_i b1_j) P_d1d2
//                     + (1/4) b2_i b2_j P_d2d2
//                     + (1/2)(b1_i bs2_j + bs2_i b1_j) P_d1s2
//                     + (1/4)(b2_i bs2_j + bs2_i b2_j) P_d2s2
//                     + (1/4) bs2_i bs2_j P_s2s2 ],
//
// the spectrum of the data vector's gg cross pairs
// (C_gg_cross_tomo_limber_work); i = j gives its auto spectrum. A galaxy
// bin and the matter density correlate through
//
//   P_im^loop = D^4 [ (1/2) b2_i P_d1d2 + (1/2) bs2_i P_d1s2 ],
//
// the one-loop galaxy term of the data vector's galaxy-shear integrand.
//
// LIMBER PROJECTION
// Write the lens windows of the snapshot as u1 = b1 n_l H/H0 (window 0),
// u2 = b2 n_l H/H0 (window 3) and u3 = bs2 n_l H/H0 (window 4), with n_l
// the lens redshift distribution and H/H0 = dz/dchi. At one radial node
// the gg one-loop integrand is then the quadratic form u_i^T Q u_j, with
// the symmetric matrix
//
//   Q = w |    0        P_d1d2/2   P_d1s2/2 |
//         | P_d1d2/2    P_d2d2/4   P_d2s2/4 |,    w = D^4 dchi/f_K^2,
//         | P_d1s2/2    P_d2s2/4   P_s2s2/4 |
//
// whose rows and columns are the three operators delta, delta^2 and s^2,
// all read at the Limber wavenumber k = (ell+1/2)/f_K. The (1,1) entry is
// zero because b1 b1 P is already in the input spectra. Summed over the
// radial nodes p,
//
//   lens-lens:    dC_ij = sum_p u_i^T Q u_j,
//   lens-source:  dC_ij = shear(ell) sum_p (Q_12 u2_i + Q_13 u3_i) S_j,
//
// where S_j = W_kappa - C1 n_s H/H0 (windows 1 + 2) is the complete source
// window and shear(ell) = sqrt((ell-1) ell (ell+1) (ell+2))/(ell+1/2)^2
// the spin-2 factor of limber_spectra_cov. The loop terms are correlators
// of the density, so they never multiply the magnification or RSD parts
// of a lens field, as in the data vector. Source-source spectra are
// unchanged.
//
// NUMERICAL CHOICES
// The five templates are the core C-FAST-PT tables of get_FPT_bias
// (pt_cfastpt.c), which the data vector reads: the same table, the same
// linear reader in ln k (interpol1d), the same sigma4 subtractions and
// zero outside the table's k range. get_FPT_bias builds the tables for
// the current cosmology, or reuses them while the cosmology and settings
// are unchanged. Like the TATT reader of ia_cov.c, this function does not
// test nuisance.IA_code: a table installed from Python is replaced by the
// C-FAST-PT one.
//
// Step 2 below tabulates the five independent entries of Q, including w,
// at every (ell, node). Step 4 integrates each pair; one OpenMP iteration
// is one complete (ell, pair) output, and its two SIMD lanes hold two
// adjacent radial nodes of that same sum.
//
// Parameters:
//   radial  - snapshot from radial_inputs_cov (Gauss-Legendre nodes)
//   nell    - number of multipoles, at least 1
//   ell     - multipoles >= 1, the grid limber_spectra_cov used
//   spectra - [npair][nell] rows of limber_spectra_cov (linear = 0) on the
//             same snapshot, i-major order; the terms are added in place
//
// Call serially: the FAST-PT tables are prepared before the OpenMP loops,
// and every output belongs to one worker, with no cross-thread sum. SSC
// and cNG never read these terms.
// ---------------------------------------------------------------------------
void oneloop_bias_spectra_cov(
    const struct radial_cov* radial, // common rule and bias windows
    const int nell,                 // multipole samples
    const double* ell,              // multipoles >= 1
    double* const* spectra          // Limber input, one-loop output
  )
{
  // Reject an empty multipole grid before any allocation.
  if (nell < 1) {
    log_fatal("oneloop_bias_spectra_cov needs at least one multipole");
    exit(1);
  }

  // nnode radial nodes; nfield fields, lenses first; npair unordered pairs
  // (A,B) with A <= B, the rows of spectra.
  const int nlens = radial->nlens;
  const int nsource = radial->nsource;
  const int nfield = nlens+nsource;
  const int npair = nfield*(nfield+1)/2;
  const int nnode = radial->nnode;

  // --- 1. CHECK THE BIAS MODEL AND PREPARE THE C-FAST-PT TABLES ---

  // Only b1, b2 and bs2 have terms here. A nonzero third-order bias b3 or
  // k^2 counterterm bK at any node of any lens bin would otherwise be
  // dropped without notice, so it stops the calculation. gb3 and gbK
  // (bias.c) keep no lazy state.
  for (int field=0; field<nlens; field++) {
    for (int node=0; node<nnode; node++) {
      const double z = 1.0/radial->geometry[0][node]-1.0;
      const double b3 = gb3(z, field);
      const double bk = gbK(z, field);
      if (b3 != 0.0
          || bk != 0.0) {
        log_fatal("oneloop_bias_spectra_cov models b2 and bs2 only; lens "
                  "bin %d has b3 = %g and bK = %g", field, b3, bk);
        exit(1);
      }
    }
  }

  // get_FPT_bias computes the one-loop tables from p_lin(k, a=1) with the
  // core C-FAST-PT, or reuses them while the cosmology and settings are
  // unchanged. It owns static FFTLog state and runs its own OpenMP loops,
  // so it is called here, on the calling thread, before workers read
  // FPTbias.tab.
  get_FPT_bias();

  // The tables sample ln k uniformly: FPTbias.N samples from ln k_min,
  // with lnk_max one step beyond the last sample. lnk_step is the spacing
  // interpol1d needs, defined exactly as in cosmo2D.c.
  const double lnk_min = log(FPTbias.krange[RANGE_MIN]);
  const double lnk_max = log(FPTbias.krange[RANGE_MAX]);
  const double lnk_step = (lnk_max-lnk_min)/FPTbias.N;
  const int ntable = FPTbias.N;
  const double sigma4 = FPTbias.sigma4;

  // weight[node] = w = D^4 dchi/f_K^2, the growth of a one-loop template
  // times the Limber measure; source_window[source][node] = S_j, the
  // complete source window W_kappa - C1 n_s H/H0 before its ell factor.
  // Neither depends on the multipole.
  double* weight = (double*) malloc1d(nnode);
  double** source_window = (double**) malloc2d(nsource, nnode);

  // One iteration is one radial node; it writes only that node's entries.
  // growfac(a) is D(a) with D(1) = 1, the growth the data vector applies
  // to the same tables, and it keeps no static state.
  #pragma omp parallel for schedule(static)
  for (int node=0; node<nnode; node++) {
    const double growth = growfac(radial->geometry[0][node]);
    const double growth4 = growth*growth*growth*growth;
    const double distance = radial->geometry[2][node];
    weight[node] = growth4*radial->geometry[3][node]/(distance*distance);

    for (int source=0; source<nsource; source++) {
      source_window[source][node] = radial->window[1][nlens+source][node]
                                  +radial->window[2][nlens+source][node];
    }
  }

  // --- 2. THE OPERATOR MATRIX Q AT EVERY MULTIPOLE AND NODE ---

  // operator_power[index][role][node] holds the five independent entries
  // of Q, w included, in (c/H0)^2: (c/H0)^3 from the power times
  // (c/H0)^-1 from w. Two windows in (c/H0)^-1 then make each spectrum
  // term dimensionless.
  //   role 0 = Q_12 = w P_d1d2/2      role 1 = Q_22 = w P_d2d2/4
  //   role 2 = Q_13 = w P_d1s2/2      role 3 = Q_23 = w P_d2s2/4
  //   role 4 = Q_33 = w P_s2s2/4
  // Radial nodes are innermost, so step 4 reads two adjacent nodes with
  // one vector load.
  double*** operator_power = (double***) malloc3d(nell, 5, nnode);

  // One iteration is one multipole: it interpolates the five tables at
  // the Limber wavenumber of every node and writes only that multipole's
  // rows. interpol1d (basics.c) is a pure function of the shared table.
  #pragma omp parallel for schedule(static)
  for (int index=0; index<nell; index++) {
    for (int node=0; node<nnode; node++) {
      const double lnk = log((ell[index]+0.5)/radial->geometry[2][node]);

      // Outside the table's k range every one-loop template is zero, as
      // in cosmo2D.c; nothing is extrapolated.
      double d1d2 = 0.0;
      double d2d2 = 0.0;
      double d1s2 = 0.0;
      double d2s2 = 0.0;
      double s2s2 = 0.0;
      if (lnk >= lnk_min
          && lnk <= lnk_max) {
        // Rows 0..4 of FPTbias.tab: P_d1d2, P_d2d2, P_d1s2, P_d2s2 and
        // P_s2s2 at a = 1, with the core's low-k subtractions of the three
        // quadratic correlators (the same expressions as cosmo2D.c).
        d1d2 = interpol1d(FPTbias.tab[0], ntable, lnk_min, lnk_max,
                          lnk_step, lnk);
        d2d2 = interpol1d(FPTbias.tab[1], ntable, lnk_min, lnk_max,
                          lnk_step, lnk) - 2.*sigma4;
        d1s2 = interpol1d(FPTbias.tab[2], ntable, lnk_min, lnk_max,
                          lnk_step, lnk);
        d2s2 = interpol1d(FPTbias.tab[3], ntable, lnk_min, lnk_max,
                          lnk_step, lnk) - 4./3.*sigma4;
        s2s2 = interpol1d(FPTbias.tab[4], ntable, lnk_min, lnk_max,
                          lnk_step, lnk) - 8./9.*sigma4;
      }

      // The factors 1/2 and 1/4 are the operator normalizations of
      // delta_g: one quadratic operator meets b1 delta with weight 1/2,
      // two quadratic operators meet with weight (1/2)(1/2) = 1/4.
      const double w = weight[node];
      operator_power[index][0][node] = 0.5*w*d1d2;
      operator_power[index][1][node] = 0.25*w*d2d2;
      operator_power[index][2][node] = 0.5*w*d1s2;
      operator_power[index][3][node] = 0.25*w*d2s2;
      operator_power[index][4][node] = 0.25*w*s2s2;
    }
  }

  // --- 3. LIST THE OUTPUT ROWS THAT CONTAIN A LENS FIELD ---

  // rows[0][n], rows[1][n] and rows[2][n] are the output row and the two
  // field indices (first <= second) of the n-th such pair, in the i-major
  // order of limber_spectra_cov: (0,0),(0,1),...,(0,nfield-1),(1,1),...
  // The first field of every listed pair is a lens; source-source rows
  // receive no one-loop term and are not listed.
  const int nloop = nlens*(nlens+1)/2+nlens*nsource;
  int** rows = (int**) malloc2d_int(3, nloop);
  int pair = 0;
  int listed = 0;

  for (int first=0; first<nfield; first++) {
    for (int second=first; second<nfield; second++) {
      if (first < nlens) {
        rows[0][listed] = pair;
        rows[1][listed] = first;
        rows[2][listed] = second;
        listed++;
      }
      pair++;
    }
  }

  if (pair != npair
      || listed != nloop) {
    log_fatal("oneloop_bias_spectra_cov: pair bookkeeping failed");
    exit(1);
  }

  // --- 4. INTEGRATE EVERY LENS PAIR ON THE COMMON RADIAL RULE ---

  // A Limber spectrum adds the integrand of each shared radial shell, so
  // the one-loop increment of a pair is a sum over nodes of the bilinear
  // form of the header. collapse(2) lets OpenMP divide the combined
  // (ell, pair) iterations among workers; one iteration is one multipole
  // and one pair, a complete output that only this worker writes. The two
  // SIMD lanes hold two adjacent radial nodes of that one sum: lane 0 adds
  // the even nodes 0, 2, 4, ... and lane 1 the odd nodes 1, 3, 5, ...; the
  // two partial sums are added at the end. That order is fixed by nnode,
  // so the result does not depend on the thread count.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int index=0; index<nell; index++) {
    for (int loop=0; loop<nloop; loop++) {
      const int row = rows[0][loop];
      const int first = rows[1][loop];
      const int second = rows[2][loop];

      // The five entries of Q at this multipole, one value per node.
      // restrict promises that no store in this loop changes these rows.
      const double* restrict q12 = operator_power[index][0];
      const double* restrict q22 = operator_power[index][1];
      const double* restrict q13 = operator_power[index][2];
      const double* restrict q23 = operator_power[index][3];
      const double* restrict q33 = operator_power[index][4];

      // The first field is a lens: its operator windows u1, u2, u3 are
      // the density windows of b1 delta, b2 delta^2 and bs2 s^2.
      const double* restrict u1_i = radial->window[0][first];
      const double* restrict u2_i = radial->window[3][first];
      const double* restrict u3_i = radial->window[4][first];

      // vtotal = [sum over even nodes, sum over odd nodes] = [0, 0]
      // (setzero sets both lanes to 0.0).
      v2d vtotal = simde_mm_setzero_pd();
      double total = 0.0;
      int node = 0;

      if (second < nlens) {
        // --- lens-lens: dC_ij = sum_p u_i^T Q u_j ---
        const double* restrict u1_j = radial->window[0][second];
        const double* restrict u2_j = radial->window[3][second];
        const double* restrict u3_j = radial->window[4][second];

        // scalar: at each node p, first the vector v = Q u_j, the response
        // of the three operators of bin i to bin j, then u_i . v:
        //   v1 = fma(q12[p], u2_j[p], q13[p]*u3_j[p]);
        //   v2 = fma(q12[p], u1_j[p], fma(q22[p], u2_j[p], q23[p]*u3_j[p]));
        //   v3 = fma(q13[p], u1_j[p], fma(q23[p], u2_j[p], q33[p]*u3_j[p]));
        //   total = fma(u3_i[p], v3, fma(u2_i[p], v2,
        //               fma(u1_i[p], v1, total)));
        // v1 has no Q_11 term: b1 b1 P is already in the input spectrum.
        // Lanes 0 and 1 below are the nodes p and p+1 of this same pair.
        for (; node+1<nnode; node+=2) {
          // loadu reads entries [node] and [node+1] of a row into lanes 0
          // and 1; the address needs no vector alignment. These six loads
          // give the operator windows of bin j and then of bin i.
          const v2d vu1j = simde_mm_loadu_pd(u1_j+node);
          const v2d vu2j = simde_mm_loadu_pd(u2_j+node);
          const v2d vu3j = simde_mm_loadu_pd(u3_j+node);
          const v2d vu1i = simde_mm_loadu_pd(u1_i+node);
          const v2d vu2i = simde_mm_loadu_pd(u2_i+node);
          const v2d vu3i = simde_mm_loadu_pd(u3_i+node);

          // loadu again: the five entries of Q at the same two nodes.
          const v2d vq12 = simde_mm_loadu_pd(q12+node);
          const v2d vq22 = simde_mm_loadu_pd(q22+node);
          const v2d vq13 = simde_mm_loadu_pd(q13+node);
          const v2d vq23 = simde_mm_loadu_pd(q23+node);
          const v2d vq33 = simde_mm_loadu_pd(q33+node);

          // scalar: v1 = fma(q12, u2_j, q13*u3_j). mul gives q13*u3_j in
          // each lane, rounded once; fmadd then adds q12*u2_j with one
          // rounding where the processor has a fused instruction.
          v2d vv1 = simde_mm_mul_pd(vq13, vu3j);
          vv1 = simde_mm_fmadd_pd(vq12, vu2j, vv1);

          // scalar: v2 = fma(q12, u1_j, fma(q22, u2_j, q23*u3_j)). mul
          // starts from q23*u3_j; the first fmadd adds q22*u2_j and the
          // second q12*u1_j, lane by lane.
          v2d vv2 = simde_mm_mul_pd(vq23, vu3j);
          vv2 = simde_mm_fmadd_pd(vq22, vu2j, vv2);
          vv2 = simde_mm_fmadd_pd(vq12, vu1j, vv2);

          // scalar: v3 = fma(q13, u1_j, fma(q23, u2_j, q33*u3_j)), in the
          // same three steps as v2.
          v2d vv3 = simde_mm_mul_pd(vq33, vu3j);
          vv3 = simde_mm_fmadd_pd(vq23, vu2j, vv3);
          vv3 = simde_mm_fmadd_pd(vq13, vu1j, vv3);

          // scalar: total = fma(u1_i, v1, total), then the same for u2_i v2
          // and u3_i v3. Each fmadd adds one operator's contribution to
          // the lane's own partial sum; lanes never mix.
          vtotal = simde_mm_fmadd_pd(vu1i, vv1, vtotal);
          vtotal = simde_mm_fmadd_pd(vu2i, vv2, vtotal);
          vtotal = simde_mm_fmadd_pd(vu3i, vv3, vtotal);
        }

        // An odd node count leaves one node: the scalar form of the same
        // steps, added after the two lane sums below.
        for (; node<nnode; node++) {
          const double v1 = fma(q12[node], u2_j[node], q13[node]*u3_j[node]);
          const double v2 = fma(q12[node], u1_j[node],
                                fma(q22[node], u2_j[node],
                                    q23[node]*u3_j[node]));
          const double v3 = fma(q13[node], u1_j[node],
                                fma(q23[node], u2_j[node],
                                    q33[node]*u3_j[node]));
          total = fma(u3_i[node], v3,
                      fma(u2_i[node], v2, fma(u1_i[node], v1, total)));
        }
      } else {
        // --- lens-source: dC_ij = shear(ell) sum_p (Q_12 u2_i
        //                                  + Q_13 u3_i) S_j ---
        const double* restrict window_j = source_window[second-nlens];

        // scalar: at each node p,
        //   total = fma(fma(q12[p], u2_i[p], q13[p]*u3_i[p]),
        //               window_j[p], total);
        // the one-loop galaxy-matter power of bin i times the source
        // window. Lanes 0 and 1 are the nodes p and p+1 of this pair.
        for (; node+1<nnode; node+=2) {
          // loadu reads entries [node] and [node+1] into lanes 0 and 1:
          // the two bias windows of the lens, the two entries of Q that
          // couple delta^2 and s^2 to the matter, and the source window.
          const v2d vu2i = simde_mm_loadu_pd(u2_i+node);
          const v2d vu3i = simde_mm_loadu_pd(u3_i+node);
          const v2d vq12 = simde_mm_loadu_pd(q12+node);
          const v2d vq13 = simde_mm_loadu_pd(q13+node);
          const v2d vsource = simde_mm_loadu_pd(window_j+node);

          // scalar: m = fma(q12, u2_i, q13*u3_i), the lens one-loop
          // matter cross power with its weight w. mul rounds q13*u3_i
          // once per lane; fmadd adds q12*u2_i.
          v2d vmatter = simde_mm_mul_pd(vq13, vu3i);
          vmatter = simde_mm_fmadd_pd(vq12, vu2i, vmatter);

          // scalar: total = fma(m, window_j, total): fmadd adds this
          // node's product to the lane's own partial sum.
          vtotal = simde_mm_fmadd_pd(vmatter, vsource, vtotal);
        }

        // The scalar remainder of an odd node count, as above.
        for (; node<nnode; node++) {
          const double matter = fma(q12[node], u2_i[node],
                                    q13[node]*u3_i[node]);
          total = fma(matter, window_j[node], total);
        }
      }

      double lanes[2];

      // lanes[0..1] = [even-node sum, odd-node sum]: storeu writes lane 0
      // to lanes[0] and lane 1 to lanes[1]. This ordinary stack array
      // needs no vector alignment.
      simde_mm_storeu_pd(lanes, vtotal);

      // The complete radial sum: the even-node lane, then the odd-node
      // lane, then the scalar remainder (zero for an even node count).
      double increment = (lanes[0]+lanes[1])+total;

      // A source leg carries the spin-2 factor of limber_spectra_cov;
      // l >= 1 keeps the square root real.
      if (second >= nlens) {
        const double l = ell[index];
        const double ell_shift = l+0.5;
        increment *= sqrt((l-1.0)*l*(l+1.0)*(l+2.0))
                     /(ell_shift*ell_shift);
      }
      spectra[row][index] += increment;
    }
  }

  // Release the scratch: one free per grouped array.
  free(rows);
  free(operator_power);
  free(source_window);
  free(weight);
}
