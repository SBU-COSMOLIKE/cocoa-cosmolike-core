# UV-fraction scale cuts of the Fourier C_gg: the IR/UV split

Design and validation record of the physical scale cuts of
arXiv:2603.13031 (following DeRose et al. 2025) for the Fourier
galaxy-clustering spectra: the autos and the gg cross pairs. Read this
before changing the k window of the gg engines (`gg_kwin` in
`cosmo2D.c`) or the scale-cut driver and wrappers (`cosmo2D_scuts.c`,
`cosmo2D_scuts_wrapper.cpp`).

## Criterion

Each spectrum is split as C_l = C_l^IR + C_l^UV. The IR part is computed
with the 3D power spectrum multiplied by a step that keeps
k <= k_max/D(z), with k_max in h/Mpc quoted at z = 0 and the linear
growth factor normalized to D(0) = 1; the UV part is the remainder.
A band power is kept while

    |C_l^UV| / (|C_l^UV| + |C_l^IR|) < alpha

(alpha = 0.05 in the paper's baseline). This is a second criterion next
to the shear response function already in `cosmo2D_scuts.c` (Doux et
al., arXiv:2011.06469, eq. 17), and differs from it in three ways: the
absolute values are taken of the integrated parts, not of the integrand;
the cut scales as 1/D(z) instead of sitting at one fixed k_max; and it
is a step on P(k) rather than a construction on the Limber relation, so
it holds for Limber and non-Limber projections alike. The last point is
the reason the criterion exists here: a thin lens bin keeps clustering
power in radial modes that only the non-Limber engines carry, and the
split counts their small-scale power toward C_l^UV.

## Window state (`cosmo2D.c`)

The split is implemented as a file-scope "k window" read by every
Fourier gg engine:

- `gg_kwin`: `active`, `keep_ir`, `kmax`, `per_bin`,
  `Kbin[MAX_SIZE_ARRAYS]` (per-lens-bin thresholds in h/Mpc),
  `Kpair` (heap, one threshold per gg cross pair), `npairs`, `nonce`.
- `set_gg_kwindow(kmax, keep_ir)`: validates kmax > 0, keep_ir in
  {0, 1} and clustering_nbin <= MAX_SIZE_ARRAYS; fills
  `Kbin[i] = kmax/D` at bin i's mean redshift and
  `Kpair[n] = kmax/D` at the midpoint of pair n's two mean redshifts
  (the redshifts the FKEM pivots use); increments `nonce`. It reads
  `growfac` of the current cosmology, so it belongs immediately before
  the spectra it windows, outside parallel regions.
- `unset_gg_kwindow()`: frees `Kpair`, deactivates, increments `nonce`.
  `initial_setup` (generic_interface.cpp) calls it, so a rebuilt
  likelihood in the same process never inherits a window.
- `gg_kwin_pass(k, K)`: returns 1.0 or 0.0. The engines measure k in
  units of H0/c; the thresholds are in h/Mpc and `cosmology.coverH0`
  converts them.

The window is off by default, and every data-vector evaluation runs
without it (measured bitwise identical; see Validation).

## Where each engine applies the step

- `C_gg_tomo_limber_work` (Limber autos): multiplies the node mask
  `KG[2]` per line-of-sight node. Threshold: the node's own
  `kmax/D(a)` (`per_bin = 0`), or the bin's `Kbin` inside a non-Limber
  core (`per_bin = 1`).
- `C_gg_cross_tomo_limber_work` (Limber crosses): same, on `maskp`,
  with `Kpair[n]` in per-bin mode.
- `C_cl_tomo_core` (classic FKEM): steps the final sum over the FFTLog
  wavenumbers (`vres`) at `Kbin[i]`.
- `C_cl_tomo_lowrank_core` (low-rank non-Limber): steps the final sums
  `vr` (autos, in a separate loop so the unwindowed arithmetic loop is
  untouched) and `vrx` (pairs, at `Kpair[n]`).
- `C_cl_tomo_fkem_pairs_core` (FKEM with cross pairs): steps
  `vres_pool` (autos) and `vrx` (pairs).

Each non-Limber core wraps its internal Limber helper calls
(`C_gg_tomo_limber_nl_lin_nointerp_ells`,
`C_gg_tomo_limber_nl_lin_dd_nointerp_ells`,
`C_gg_cross_tomo_limber_work`) in `per_bin = 1`, restored to 0 after.

## The per-bin rule and why

The non-Limber engines apply the step after the radial FFTLog
transforms, in the final sum over wavenumbers. At that point the radial
integral has already run: the integrand no longer knows the redshift of
its modes, so an exact per-node threshold k_max/D(a) is unavailable.
Each lens bin therefore gets one threshold, D at its mean redshift (the
pair midpoint for crosses). Across a thin bin D barely changes; across
a wide bin the Limber sums, which dominate there, keep the exact
per-node step.

Inside a non-Limber core, the Limber correction terms that cancel the
FFTLog sum at high l must see the same per-bin threshold as that sum,
or the cancellation (FFT term minus Limber term -> 0) fails. That is
what the `per_bin = 1` bracket enforces. The Limber values at the band
centers, computed outside the cores, keep the per-node threshold.

Applying the step after the transforms is exact: the final sum is
linear in the per-wavenumber contributions, so partitioning the
wavenumbers partitions C_l with no numerical residue beyond rounding.

## The white-noise constant of the one-loop spectrum

The one-loop galaxy templates are renormalized by subtracting the
k -> 0 limits of the quadratic-operator spectra (the sigma4
subtractions of `C_gg_tomo_limber_work`). The subtracted spectrum is
therefore the raw loop sum plus one constant power per bin pair,

    P_wn = -(sigma4/2) c_i c_j D^2(z_i) D^2(z_j),  c = b2 + (2/3) bs2,

and the raw sum carries a white-noise plateau of exactly -P_wn at low
k, so in 3D the two cancel and the subtracted spectrum vanishes as
k -> 0. Under the window, each engine must split the plateau and the
constant with the same step, or the cancellation leaks into the IR/UV
split:

- Limber (`adopt_limber_gg = 1`): the subtracted templates are
  evaluated per node, so plateau and constant pass the same node step.
  Consistent as is.
- Classic FKEM: the FFTLog sum carries only the separable linear
  spectrum; every one-loop term, plateau and constant alike, rides the
  Limber difference under the node step. Consistent as is.
- Low-rank: the factorization needs `sqrt(P_gg)`, which exists only
  for the positive raw sum (P_wn < 0 drives the subtracted spectrum
  through zero). The raw sum therefore passes the finite-k step of the
  FFTLog sum while the spliced constant passes the Limber node step.
  On a kept band the node step keeps (nearly) all of the constant's
  local projection, the finite-k kernel keeps only a fraction of the
  raw plateau, and the IR part comes out too negative - the UV part
  too positive - by the difference. The spurious UV excess grows as
  c^2 and moved kept bands at the paper settings, where many bands sit
  within 1e-3 of alpha.

`C_gg_kwindow_white_noise_ells` (cosmo2D.c) repairs the low-rank IR
spectra in place: it replaces the constant's Limber-step IR projection
L_K by the finite-k projection N_K at the same threshold the FFTLog
sum uses (`Kbin` per auto, `Kpair` per pair). The k integral of N_K
has a closed form per radial node pair,

    int_0^K dk k^2 j_l(ka) j_l(kb)
      = K^2 [b j_l(Ka) j_{l-1}(Kb) - a j_{l-1}(Ka) j_l(Kb)]/(a^2 - b^2)

and (K^3/2) [j_l(Ka)^2 - j_{l-1}(Ka) j_{l+1}(Ka)] at a = b, so N_K is
a double sum over a radial grid with no k integration. The grid is
uniform in chi, 0.5 Mpc/h halved per `integration_accuracy` level up
to level 2, and shared by every bin: overlapping supports meet on
common nodes (where the a = b limit applies), and L_K is discretized
on the same nodes, so the part the two projections share cancels node
by node and only the small difference survives discretization error.
The spherical Bessel tables come from two exact GSL evaluations per
node at an order in the Airy tail (l - x > 20 (x/2)^(1/3) + 20, where
j_l is below ~1e-20 of its peak) and stable downward recurrence.

The correction applies exactly where the splice put the constant: at
band centers 1 <= ell < `lowrank_DD_GG_lmax` - 1, with N_K
interpolated linearly between the two neighboring integers as the
splice interpolates, and L_K mirroring the Limber workers' integrand
(per-node step kmax/D(a), FPT k range, RSD reach mask). Above the cap
the spectrum is Limber throughout and already consistent. Without a
window, or with linear bias only, nothing changes.

## Rejected prototype: stepping P(k) before the transforms

A prototype multiplied P(k) by the step before the FFTLog transforms.
A discontinuous input rings under FFTLog, and IR + UV missed the full
spectrum by up to 3.7% in the transition band. The post-transform step
replaced it; IR + UV = full now holds to 1e-14 (see Validation). Do not
reintroduce a pre-transform step without solving the ringing.

## Guards, cache and stash

- `C_gg_tomo_limber` (the cached real-space Limber table), `C_cl_tomo`
  and `C_cl_tomo_lowrank` (the real-space projections) abort when the
  window is active: the window serves the Fourier scale cuts only and
  must never leak into cached or real-space results.
- The HOD path aborts (`include_HOD_GX`): its sums carry no node mask.
- The gg cross-pair stash key carries `gg_kwin.nonce`
  (`ggx_stash_key[4]`): corrections stashed under one window setting
  cannot be consumed under another, and set/unset both bump the nonce.

## Thread safety

`gg_kwin` is file-scope mutable state. `set_gg_kwindow`,
`unset_gg_kwindow` and the `per_bin` flips run in serial code between
parallel regions; the OpenMP loops only read the window. One window
setting exists per process at a time, and the driver brackets each
evaluation with set/unset. Do not move the flips inside a parallel
region or hold a window across unrelated evaluations.

## Driver and wrappers

- `C_gg_kwindow_tomo_ells` (`cosmo2D_scuts.c`): the data-vector path
  with or without the window — Limber when `adopt_limber[LIMBER_GG]`
  is set, the configured non-Limber engine otherwise; autos before the
  crosses, which consume the stash. `kmax <= 0` means no window; the
  window is always unset on return. With the low-rank engine a
  windowed IR run also receives the white-noise correction above, and
  a windowed UV run is computed as full - IR, so IR + UV = full stays
  exact.
- `gg_scale_cuts_tomo_ells_work`: over an ascending k_max grid, the
  full spectra, the IR parts and the UV fractions (UV = full - IR,
  exact because the window partitions P(k)). The bands kept are a
  prefix in l, ended by the first band whose UV fraction reaches
  alpha; prefixes are then shortened downward in k_max, because the
  UV ratio can be non-monotonic where a spectrum changes sign (cross
  pairs, RSD terms). Cost: one engine evaluation for the full spectra
  plus one per k_max.
- `C_gg_kwindow_tomo_cpp`: (autos, pairs) tuple of the IR or UV
  spectra at the data-vector band centers.
- `C_gg_kwindow_white_noise_cpp`: the white-noise correction alone at
  one kmax, as an (autos, pairs) tuple - a diagnostic for notebooks
  and the scale-cut tests (`C_gg_kwindow_tomo_cpp` already applies
  it). Zero unless the low-rank engine runs with one-loop galaxies.
- `gg_scale_cuts_tomo_cpp`: dict of kmax, alpha, ell, nautos, lmax,
  mask, uv_fraction, full, ir; rows are the autos, then the pairs, in
  data-vector order. Projects bind these in their `interface.cpp`;
  lssty1xdesi binds them as `gg_scale_cuts_tomo`, `C_gg_kwindow_tomo`
  and `C_gg_kwindow_white_noise`.

## Validation (lssty1xdesi, dz = 0.05 layout)

- Window off: the data vector is bitwise identical to the pre-feature
  code.
- IR + UV = full to 1e-14 (low-rank, FKEM with pairs) and 1e-15
  (Limber).
- k_max = 1e3 h/Mpc gives IR = full exactly; k_max = 1e-4 gives
  IR = 0.
- The white-noise correction: an asserted study test (lssty1xdesi
  `tests/studies/test_scale_cut_white_noise.py`) checks the exact zero
  with linear bias, IR + UV = full under the corrected split, the
  correction's shape against an independent spherical-Bessel quadrature
  on deep-IR bands (one fitted prefactor per spectrum, bilinear across
  bins) and closure at a large cut, on the wide and thin paper layouts.
  The shape comparison stays in the deep IR because near the transition
  l = K chi the finite-k projection rises steeply with the threshold K,
  so percent-level input differences between two independent pipelines
  are amplified to tens of percent of the correction.
- The white-noise correction on dz = 0.2 at the final accuracy tier:
  the fiducial data vector and the full spectra are bit for bit the
  pre-correction cached ones (the correction only moves the IR/UV
  split), and the corrected cuts differ from an independent numpy
  finite-k re-cut of the cached IR parts in 10 of about 11,000 entries,
  all sitting at the threshold, within the agreement of the two
  implementations.
- Chi-grid convergence of the correction, step 0.5 against 0.125 Mpc/h
  (dz = 0.2): at most 8.6e-5 of C_l on the 243 entries within 0.01 of
  the alpha = 0.05 threshold. The largest differences, up to 1.3e-3 of
  C_l, sit on bands with UV fractions 0.16-0.87, which are cut anyway.
- Cost: the correction adds 0.7 s per kmax on the dz = 0.2 layout.
- Auto band powers kept at alpha = 0.05 (summed over the auto spectra):

| Method | 0.05 | 0.075 | 0.1 | 0.125 | 0.15 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Limber hook | 518 | 808 | 1067 | 1171 | 1198 |
| Independent numpy Limber integral | 519 | 808 | 1068 | 1171 | 1198 |
| Low-rank non-Limber | 367 | 663 | 945 | 1148 | 1195 |
| Independent flat-sky non-Limber estimate | 376 | 683 | 973 | 1154 | 1198 |
| FKEM with pairs | 370 | 674 | 975 | 1154 | 1196 |

Columns are k_max in h/Mpc at z = 0. The Limber columns agree with the
independent integral to one band; the non-Limber engines keep fewer
bands at low k_max, as expected from the radial modes counted into
C_l^UV, and agree with the independent flat-sky estimate at the
10-30 band level where the cut bites.

## Remaining limits

- The non-Limber final sums use one threshold per bin (D at the mean
  redshift), not the exact per-node D(a); only the Limber sums keep the
  per-node step.
- The classic FKEM path without cross pairs applies the window but has
  not been exercised end to end: projects that use it (such as lsst_y1)
  need the wrapper binding first.
- No HOD support and no real-space support; both abort by design.
