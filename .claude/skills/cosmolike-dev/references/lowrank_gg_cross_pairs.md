# Thin-bin galaxy clustering: the low-rank non-Limber gg engine and the gg cross pairs

Design record of the two thin-bin clustering extensions of `cosmo2D.c`:
the low-rank density-density non-Limber method (the `[SECTION] LOW-RANK`
block) and the cross-bin gg pair machinery (the `[SECTION] CROSS-BIN`
block). Read this before changing either, their setters
(`set_lowrank_DD_GG`, `set_gg_cross_pairs`, `limits.LMAX_NOLIMBER`) or
their bindings (`init_lowrank_nonlimber_gg`, `init_gg_cross_pairs`,
`init_lmax_nonlimber` in `generic_interface.cpp`). The k window that
both engines also apply is a separate feature with its own record,
`scale_cuts_uv_fraction.md`.

## Problem

A spectroscopic lens sample split into thin redshift bins breaks two
assumptions of the long-standing gg machinery:

- The data vector keeps only auto spectra (ni = nj everywhere). Thin
  adjacent bins overlap through their smeared n(z) tails, and bins of
  two samples can sit at the same redshift, so cross-bin spectra
  C_l^{g_i g_j} are measured.
- FKEM (`C_cl_tomo`) projects a separable linear spectrum exactly and
  hands everything nonlinear to a Limber term. Limber's error shrinks
  with the kernel width, so for a very thin bin the hand-off stays
  wrong far past the default switch multipole of 150 - exactly where
  non-Limber accuracy was the goal.

Both extensions are off by default, and a project that never enables
them runs bit for bit the code it ran before they existed (no pair
list installed, `use_lowrank_DD_GG = 0`).

## The low-rank engine

### Method

FFTLog can only project a factorized z-dependence: one k-function
times one z-function per transform row. Instead of demanding one
separable term (FKEM), an SVD writes the square root of the full
one-loop galaxy spectrum as a short sum of separable terms,

    sqrt(P_g(k, z)) ~ sum_r T_r(k) Z_r(z),    r = 1..nrank,

(the tables sqrtPg_kfactor and sqrtPg_zfactor; Z_r is written alpha_r
in some code comments, renamed here to keep alpha_2 for the
counterterm),

and the unequal-time spectrum is the geometric mean,
P_g(k; z1, z2) ~ sqrt(P_g(k, z1)) sqrt(P_g(k, z2)) - exact at equal
times, fully-correlated tracers assumed at unequal times. That
assumption is the method's physical approximation and the reason the
flag defaults to off.

`C_gg_lowrank_decomp_work` factorizes, per lens bin, the weighted
matrix M[ik][iz] = sqrt(nz_weight(z)) sqrt(P_g(k, z)) / sqrtPg_zref(k)
with `gsl_linalg_SV_decomp`; truncation at nrank is the least-squares
optimum (Eckart-Young). The n(z) weight steers the truncation error to
redshifts without galaxies and cancels in the reconstruction; the
k-only reference sqrtPg_zref = |b1(zmean)| sqrt(Pdelta(k, a_ref))
makes rank 1 the FKEM separable baseline, so every higher rank
measures a genuine departure from separability (scale-dependent
growth, nonlinear evolution, one-loop bias). z grid: GG_LOWRANK_NZ =
128 nodes on the bin's n(z) support; the k grid comes from the
caller's FFTLog chi grid and cap, so every query is interior. NK >=
128 is required (GSL's SVD needs rows >= columns).

P_g is the Limber gg integrand's one-loop spectrum minus its constant
tail: the sigma4 subtractions of the quadratic templates leave
P_inf = -(sigma4/2) D^4 (b2 + 2 bs2/3)^2 <= 0 at high k, which can
drive P_g negative. The decomposition therefore factorizes
P_g - P_inf (positive), and the constant needs no non-Limber term: a
constant projects exactly in Limber, and the DD subtraction uses the
same unsubtracted templates. A floor max(P_g, 1e-15 b1^2 Pdelta)
guards the remaining negative crossings (count log_debug'd).

### Assembly

`C_cl_tomo_lowrank_core` is the structural twin of `C_cl_tomo_core`
(same variable names where roles coincide, so diffs align). It
replaces the Limber hand-off for the DD term only:

    Cl[i][l] = FFT(F^2 - F0^2)                       FKEM rows minus DD
             + (2/pi) sum_q dlnk k^3 G^2             low-rank DD
             + (CLnl - CLdd_nl) - (CLlin - CLdd_lin) Limber corr.,
                                                     non-DD rows only

with F0 the density slot alone and G the rank sum of the transformed
rank kernels. RSD and magnification keep FKEM (velocities do not grow
like the one-loop density; the magnification kernel is broad, where
Limber is accurate), and so does the alpha_2 counterterm (a
linear-theory term of the density and RSD rows). At high l the terms
cancel pairwise, so Cl -> CLnl up to the rank truncation and the
Limber splice works unchanged. The rank kernels ride the shared
transforms as extra rows; SIZE2 stays 3 so the FFTLog configs remain
byte-identical to the FKEM ones and their static caches never rebuild.
With `use_nonlinear_kaiser_GG = 1` the RSD rows additionally carry the
one-loop velocity spectra through two more FFTLog rows per bin,
outside the SVD (the loop spectra are exactly separable, so they need
no decomposition); the switch, the templates, the Limber match and the
validation record are in `nonlinear_kaiser.md`.

The DD Limber tables come from the auto batch's DD outputs
(`C_gg_tomo_limber_work` table_dd/table_dd_lin;
`C_gg_tomo_limber_nl_lin_dd_nointerp_ells` at band centers): density
rows W_gal b1 only, same nodes, same mask, same summation order as the
full model, or the high-l cancellation leaves a residual. With
one-loop bias the pair DD output is the geometric mean
W_gal_i W_gal_j sqrt(P_g,i P_g,j) - exactly what the rank product
tends to - not the bilinear spectrum; the Limber splice adds back the
bilinear-minus-geometric-mean residual and the white-noise constant,
which Limber projects exactly.

### Memory design

The twin stores transforms for every multipole up to the cap; with
many rows (nbins FKEM rows + nbins*nrank rank rows) and a high cap
that product explodes. The core instead keeps pointer *views* with
absolute-l indexing over one-block *pools* (BLOCK = 16 multipoles):
`Fy`/`Fy_pool`, `vres`/`vres_pool`, the shared wavenumber table
`y_rows` aliased by every row (detected by pointer identity, filled
once), and `malloc3d_ptr` (basics.c) for the pointer-only levels. Only
addresses change, never values or order, so outputs are bitwise those
of the plain layout at BLOCK/cap of the footprint. The FKEM pair twin
(`C_cl_tomo_fkem_pairs_core`) uses the same scheme, so its footprint
does not grow with `lmax_nonlimber`; the classic gg/gs cores still
hold every multipole below the cap (rows x cap x NL_Nchi doubles).

The rebuild guards of both cores' pair storage compare every size
their allocations use — npairs, nbins, nchi, the cap and
`Ntable.random` — in dedicated cache slots. `set_lowrank_DD_GG` does
not bump `Ntable.random`, so without the size keys a cap raised in
one process (a notebook rerun at a larger lmax) kept rows of the old
length and the pair pass wrote past them (pitfall 14 of
`pitfalls.md`; restored for the low-rank core in commit 4f9739d).

### Knobs and validation

`init_lowrank_nonlimber_gg(flag, nrank, lmax)` -> `set_lowrank_DD_GG`:
flag in {0, 1}; 1 <= nrank <= 16; 32 <= lmax <= Ntable.LMAX; flag = 1
refuses `include_HOD_GX = 1`. Likelihood yaml keys:
`nonlimber_lowrank_gg`, `nonlimber_lowrank_rank`,
`nonlimber_lowrank_lmax`. The low-rank cap is its own; it ignores
`limits.LMAX_NOLIMBER`. No nuisance random is redrawn on a flip:
`w_gg_tomo` keys its cache on the three values directly, so a round
trip restores the FKEM data vector bitwise. `C_cl_tomo_lowrank`
re-checks the cap against `Ntable.LMAX` at call time (Ntable may have
been re-initialized since the setter ran).

Real space (`C_cl_tomo_lowrank`) keeps the per-bin early exit and the
Limber continuation; the Fourier entry (`C_gg_tomo_lowrank_ells`) runs
tol = 0 - the early exit is a real-space optimization whose truncation
the band centers would see directly in chi2 - and evaluates its Limber
base exactly at the band centers (nointerp batch), never through the
interpolated table.

## The gg cross pairs

### Pair list

`set_gg_cross_pairs(bi, bj, npairs)` installs the measured pairs, in
the data vector's gg-cross block order; the project reads them from
its dataset files (the pair list is data, not a rule). Validation:
0 <= bi < bj < clustering_nbin, npairs <= GG_CROSS_MAX_PAIRS = 512;
violations fatal (a wrong list silently scrambles the layout).
npairs = 0 restores auto-only behavior. `initial_setup` clears the
list (`set_gg_cross_pairs(NULL, NULL, 0)`): the list is file-scope
state outside the reset structs, pitfall 13 of `pitfalls.md`.

### Pair quadrature

`create_cosmo_nodes_lens_pair` builds the pair's Gauss-Legendre nodes
on the union of the two Limber ranges, split into panels at every
interior support edge, so the narrow density-density overlap sliver
gets full node density. Panels never depend on b_mag; only when both
bins have b_mag != 0 does one extra panel cover the
magnification-times-magnification foreground. With b_mag = 0
everywhere the nodes are bitwise those of a build without
magnification.

All three pair integrals (the pair Limber batch, its FKEM twin and
the low-rank core) draw one cached Gauss-Legendre rule per panel,
`gg_pair_glrule`: 128 nodes plus 64 per level of
`integration_accuracy` (magnitude only, capped at 8192 like every
other ladder), rebuilt when `Ntable.random` changes. The ladder grows
more slowly than the auto batches' doubling because each panel is one
short, smooth stretch; `pair_quadrature_suite`
(`cocoa_testing_suites.py`) sweeps it against a converged reference
per project.

### Pair Limber batch

`C_gg_cross_tomo_limber_work` fills up to four outputs per pair:
full model, separable linear term (one pivot per pair, needed for the
high-l cancellation), and the DD part of each (the low-rank
subtraction contract above). One-loop bias enters the full output with
each bias product shared between the bins ((b1_i b2_j + b2_i b1_j)/2
P_d1d2, ...; a bin paired with itself reproduces its auto). The
alpha_2 counterterm adds
(k^2/2)[alpha_2,j K_i W_RSD,j + alpha_2,i W_RSD,i K_j] P_lin to the
full and linear outputs, never the DD ones; it requires the RSD row
(`has_alpha2_galaxies` is fatal with `include_RSD_GG = 0` or HOD
galaxies). Magnification rows are formed only when some bin of the
list has b_mag != 0. HOD galaxies are refused.

### Engines and the stash

`C_gg_tomo_ells` dispatches on the low-rank flag: with it on, the pair
corrections come out of `C_cl_tomo_lowrank_core`'s pair pass; with it
off, out of the FKEM twin `C_cl_tomo_fkem_pairs_core` (same block
pools, same FFTLog configs as `C_cl_tomo_core`, cap
`limits.LMAX_NOLIMBER`). Either way the pair corrections share every
transform of the auto run, so the auto Fourier entry stashes the
band-center corrections (`ggx_stash`) and `C_gg_cross_tomo_ells`
consumes them in the same evaluation. The stash is keyed on
(cosmology.random, Ntable.random, npairs, nell, gg_kwin.nonce,
use_nonlinear_kaiser_GG) plus a fresh flag; any other call order is
fatal, never silently recomputed,
and one stash serves one evaluation. With Limber autos
(`adopt_limber_gg = 1`) the pairs are Limber at every multipole,
consistent with the autos. The cross pass requires tol = 0 (no
per-pair freeze logic exists; the cores refuse npairs > 0 with
tol > 0).

## Thread safety and determinism

The cores are called serially; every lazily built table is warmed
single-threaded before the parallel regions (the auto work's prelude
pattern). OpenMP loops write disjoint outputs with no cross-thread
reductions, so results are independent of the thread count. The pair
node builder runs outside parallel regions (lazy chi/growth init).

## Numerical settings

The non-Limber chi grid is shared with FKEM: `Ntable.NL_Nchi`,
baseline 512, scaled by `nonlimber_accuracyboost` times the global
boost. The grid spans fixed endpoints (dlnchi = log(chi_max/chi_min) /
(NL_Nchi - 1)), so only interval doubling nests its nodes: for a
combined factor above 1, `set_nl_nchi` rounds it up to a power of two
m and multiplies NL_Nchi - 1 by it (511 m at the shipped baseline).
Thin radial kernels alias on the
baseline grid - a thin-bin configuration measured delta chi2 of order
1e5 between boost 1 and boost 16 (2026-11) and settled on
`nonlimber_accuracyboost: 8` - so convergence in this knob must be
tested per project, with the nested rule keeping lower boosts
comparable node-by-node.

**The ln k grid of the decomposition depends on the cap, by design.**
The SVD's k nodes span the FFTLog range up to the low-rank cap with a
fixed node count NK (independent of the cap), so raising the cap
stretches the same nodes over a wider ln k range and coarsens the
grid. Pair values below the cap then shift - up to 5.1e-3 relative on
the measured configuration (cap 600 -> 1000, 2026-11-14) - through
the factorization and the linear reads of `sqrtPg_kfactor`. This is a
discretization property, not out-of-date state: an in-process cap
change and a fresh process at the final cap agree bitwise. Judge the
tail against the covariance's per-band error bars rather than as a
defect; the global accuracy boost scales NK at fixed cap and is the
clean convergence knob (the non-Limber boost deliberately divides
out). A grid with fixed node spacing, or anchored at `Ntable.LMAX`,
would make sub-cap values exactly cap-independent at SVD cost growing
with the cap; it remains a possible study, not the design.

## Remaining limits

- The geometric-mean unequal-time approximation is unvalidated against
  an exact unequal-time P_g; the flag stays off by default and per
  project until a dedicated comparison exists.
- No HOD support in either extension (fatal guards).
- The pair pass has no per-pair early exit; real-space pair
  projections (w(theta) crosses) do not exist - the pairs are
  Fourier-only.
- Rank truncation is capped at 16 and unmeasured beyond the
  configurations the owning project tested; a new survey layout needs
  its own rank-convergence check (FoM-level, not chi2-only).
