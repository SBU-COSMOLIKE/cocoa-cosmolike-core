# Out-of-date state: the invalidation rule and the setting-sequence tests

Why the compiled code can serve out-of-date numbers, or overflow, when
a notebook changes settings in one process without restarting, and the
rule that prevents it. Read this before adding a cached static, a
setter, a C++ singleton, or any file-scope state, and before reviewing
a patch that touches one. The audit behind it is the notebook-safety
review of 2026-11-14 (six code audits plus a synthesis; the fix plan
lives with the review records, not here).

Vocabulary. A **nonce** is a `double` field such as `cosmology.random`
or `Ntable.random` holding a fresh random draw; a setter redraws it
when an input actually changed, and a cached table stores a copy
(`cache[k]`) and compares it with `fdiff2` (a bitwise `uint64`
compare) to decide whether its contents are out of date. The
**allocation tier** of a cache is the block that mallocs its arrays;
the **refill tier** is the block that rewrites their contents.

## The problem

The notebook workflow changes settings between evaluations - bin
counts, pair lists, accuracy knobs, the IA or bias model, the run
mode, a baryon scenario, non-Limber caps - with no `initial_setup`
and no restart. Every failure the audits found has one of three
shapes:

- an allocation tier that does not see a size change: the refill tier
  writes the new extent into the old allocation (heap overflow);
- a refill tier whose key misses an input: the cache serves the
  previous setting's numbers, silently;
- file-scope state that `initial_setup` does not reset: a second
  model built in the same process inherits the first model's choices.

A fresh process at the final settings never shows any of these, which
is why they survive ordinary testing.

## The rule

One rule, three clauses, plus one safety net:

1. **Sizes.** The rebuild guard of every lazily allocated static
   compares, by value, every size that appears in its allocation
   calls: bin counts, pair counts, rank, multipole caps, thread
   counts, node counts. A size whose every setter bumps
   `Ntable.random` (the grid knobs) may be covered by `Ntable.random`
   alone; any other size is compared directly, as a plain integer,
   even when a nonce "should" also have changed.
2. **Values.** Every runtime setting a cached build reads is either
   tagged by a nonce that its setter redraws on change, or sits in
   the cache key as a value (bit-packed where several share a slot).
   Derived state (a measurement mask, a correction stash) also stamps
   the settings that shaped it, and its consumer compares the stamp
   and stops with a clear message on mismatch.
3. **Resets.** Every file-scope static outside the reset structs -
   model flags, latches, singletons, loaded scenarios - has a reset
   that `initial_setup` invokes (or, for optionally compiled modules,
   that the project binding invokes right after `initial_setup`).

**Safety net.** `init_ntomo_powerspectra` - the one choke point every
tomography change passes through - additionally bumps
`Ntable.random`, so a future cache that forgets a tomography size in
clause 1 still rebuilds. The bump never substitutes for clause 1.

The cost is an integer compare per call or one nonce draw per setup;
nothing rebuilds per MCMC step. The two failure directions are
covered from both sides: a caller that resizes the samples but skips
`init_ntomo_powerspectra` is caught by the per-cache size compare,
and a future cache whose author forgets a size key is caught by the
safety-net bump. Either alone leaves one direction open, which is
exactly how the audited gaps arose.

## The radial endpoint

The non-Limber upper redshift follows the same rule. The serial setter
`init_nonlimber_zmax(zmax)` validates a finite value at least 4.0 before
writing `Ntable.NL_zmax`. Only a changed value redraws `Ntable.random`.
The existing grid and projected-vector cache keys then rebuild the FKEM,
low-rank, cross-bin, galaxy-lensing and cluster calculations. The endpoint
does not change `NL_Nchi`; it changes the spacing of those nodes.
`initial_setup` restores 4.0, while sampling boosts leave it unchanged.
Notebook wrappers apply their configured endpoint after `initial_setup`
and in `_set_state`. `configure(nonlimber_zmax=...)` also calls the setter
directly, covering helpers that prepare CAMB without `_set_state`. The
next prediction then rebuilds. The shared `nonlimber_endpoint_suite` checks
the 4 to 5 to 4.5 to 4 sequence against fresh processes for NLA and TATT,
using both gg engines; cluster projects test their joint vector.

The bindings cover LSST Y1, DES Y3, DES Y1 with Planck, DES cluster,
Roman real, Roman Fourier, Roman KL and LSST Y1 with DESI. The DESI
likelihood omits galaxy-galaxy lensing: its endpoint suite explicitly
sets `galaxy_lensing=False`, so it does not insert an unsupported
`adopt_limber_gs` option. Its vector getter must include Fourier
cross-bin clustering. The generic real-space getter is not a valid
substitute even though that symbol exists in its dual-space interface.

## Thread count for numerical validation

The shared test harness uses four OpenMP threads by default. Set
`COCOA_TEST_OMP_THREADS` before importing the harness to select another
integer count of at least two. Each project's data-vector test entry
point and reference generator uses that value to set `OMP_NUM_THREADS`
before native imports; worker processes inherit the same selected count.
Set `OMP_THREAD_LIMIT` consistently. BLAS stays at one thread.

The lower bound matters because a single-thread run cannot exercise a
race between OpenMP workers. This override does not establish numerical
determinism by itself: cross-thread comparisons remain separate tests.
Frozen reference metadata records the count that actually generated it.
The import/validation checks are in `dev_scripts/testing_threads/`.

## The failure classes, with one example each

Each example is written as the audit found it; the Status section
lists the commit that closed it.

- **An allocation keyed on `Ntable.random` but sized by a bin
  count.** The lensing-efficiency tables `g_tomo`, `g2_tomo` and
  `g_lens` (redshift_spline.c) are sized by `redshift.shear_nbin` or
  `redshift.clustering_nbin`, yet their guards compare only NULL and
  `Ntable.random`; no sample setter bumps `Ntable.random`, so growing
  a sample mid-process overflows the old rows. The same shape:
  `C_cl_tomo_core` (keyed on `Ntable.random` alone, sized by the lens
  bin count), `C_ks_tomo_limber`, and the `xi_pm_tomo` /
  `w_gammat_tomo` / `w_gg_tomo` family (sized by pair counts).
  Pitfall 14 of `pitfalls.md`.
- **A model switch that draws no nonce.** `init_IA` and `init_bias`
  (generic_interface.cpp) rewrite the model selectors and bump
  nothing; a switch at identical amplitudes leaves every cached
  spectrum serving the previous model.
- **A latch that never resets.** The `Pdelta` run-mode dispatch
  (`pdelta_type`, cosmo3D.c) latches "linear" one-way at first use;
  `init_cosmo_runmode` rewrites the string without resetting the
  latch or bumping `cosmology.random`, so a run-mode change after the
  first evaluation does nothing.
- **A singleton that survives `initial_setup`.** `PointMass::pm_` and
  the `IPCMB` flags (generic_interface.cpp) are C++ state outside the
  reset structs: a second model in one process inherits the first
  model's point masses or bandpower layout. Pitfall 13's rule,
  extended to C++ singletons.
- **A measured row whose meaning changes.** A change to an angular
  limit or Fourier band center moves the prediction to another bin.
  If the vector length stays the same, the old mask and covariance
  still fit the array shape and can be applied to the wrong bins.

## The idioms to copy

- **Size keys next to the NULL check:** `static int NSIZE_alloc = -1;`
  and `NSIZE != NSIZE_alloc` in the rebuild condition, set inside the
  block - cosmo2D_scuts.c (the derivative tables) and basics.c are
  in-file models.
- **Every size in one guard:** `C_gs_tomo_core` keys
  `Ntable.random + SIZE1 + NSIZE`; the FKEM pairs twin and the
  low-rank cross-pair storage (`C_cl_tomo_lowrank_core`) key npairs,
  nbins, nchi and the cap in dedicated cache slots.
- **Table producer.** `set_IA_PS` and `set_bias_PS` mark tables
  supplied by Python. The CFASTPT builders use those marks alongside
  the grid and allocation checks. A Python table can be allocated at
  the address of the former C table, so a matching address alone
  cannot establish which backend filled it.
- **Model reset.** `initial_setup` invokes
  `IP::reset` to clear measurements and `BaryonScenario::reset` to
  clear loaded baryon scenarios and principal-component templates.
  Cluster projects clear their joint measurement state through
  `reset_cluster`. Add the reset for any model-specific static where
  its initialization sequence calls it.
- **Measured bins.** `IP::set_mask` records the real-space angular bin
  count and limits or the Fourier band centers. `IPCluster::set_mask`
  records the angular count and limits for the joint vector. The
  masked calculations and likelihood score check those values before
  using the loaded measurements. If the bins differ, the error names
  `init_data` or `init_data_cluster`. Integration resolution is a
  separate setting.

## How to test it

The test design is the **setting-sequence suite**: a shared factory
(`setting_sequence_suite` in `cocoa_testing_suites.py`, next to
`accuracy_suite` and `pair_quadrature_suite`) that each project binds
in `tests/data_vector/test_setting_sequences.py` with the shared
catalog (`common_sequences`) plus its own sequences. For each
sequence, one worker process builds the base configuration and
applies the setting changes step by step, evaluating after each; one
fresh worker per compared step applies the same operations and
evaluates once, so its tables never saw an earlier setting. Each
compared in-process vector must equal its fresh twin bit for bit
(float64 equality, no tolerance) - or, for sequences whose contract
is refusal, the in-process worker must abort with the named message.
The operations, evaluators and reach checks are the `SEQUENCE_OPS`,
`SEQUENCE_EVALUATORS` and `SEQUENCE_CHECKS` tables of
`cocoa_testing.py`, next to `_sequence_worker`, which documents the
two legs.

**A sequence must reach the faulty path.** A sequence that does not
reach the broken code passes on broken code: the low-rank cap-raise
overflow fires only when a band center sits at or above the old cap,
so a run whose bands all sit below it is bitwise clean on broken
code. Every sequence therefore declares a reach check - a predicate
on the configuration (a band center in [old cap, new cap), a bin
count that actually grows) that the suite asserts before trusting a
pass, so a later yaml edit cannot silently defuse a reproducer.

The catalog covers both directions of every switch: sample grow and
shrink (with and without `init_ntomo_powerspectra`), ggl-exclusion
swaps at fixed list length, cap raises with a band center in the
raised range, IA/bias model round trips, run-mode and baryon round
trips, the two-model `initial_setup` round trip, and the
layout-change refusals after `init_data` (and, in the cluster
projects, after `init_data_cluster`). Heavier variants (the full
probe matrix, an AddressSanitizer leg for overflows that do not
change the numbers, a repeated-`initial_setup` leak loop) remain
future studies behind environment keys, in the style of the existing
deep suites.

## Status

Every restoration of the audit is in the tree. Landed
(cosmolike_core, branch `lowrank_fftlog`), by clause:

Earlier fixes:

- `4f9739d` - the low-rank cross-pair storage of
  `C_cl_tomo_lowrank_core` keys every size its allocations use
  (npairs, nbins, nchi, the cap, `Ntable.random`); raising the cap in
  one process with pairs installed previously wrote past the old
  rows.
- `c8c1840` - `init_accuracy_boost`'s baselines are forgotten in
  `initial_setup` and re-captured per setup (pitfall 13's second
  instance).
- `82e0e3b` - `initial_setup` clears the gg cross-pair list
  (pitfall 13's first instance).

Sizes (clause 1) and memory safety:

- `d27a502` - the masked branch of the pair Limber batch
  (`C_gg_cross_tomo_limber_work`) zero-fills the RSD pair nodes it
  skips, removing an uninitialized read.
- `99758e9` - `C_cl_tomo_core` keys the lens-bin count; `w_gg_tomo`
  refuses a pair count that disagrees with the bin count, naming
  `init_ntomo_powerspectra`.
- `0f94f4e` - `C_ks_tomo_limber` keys the source-bin count; its fill
  stops on a NULL table.
- `e4d3665` - `g_tomo`, `g2_tomo` and `g_lens` key their bin count as
  a cache slot compared in both the allocation and the refill.
- `ed895ff` - the pair-count size slot in `xi_pm_tomo`,
  `w_gammat_tomo`, `w_gg_tomo`, `w_gk_tomo`, `w_ks_tomo`,
  `C_ss_tomo_limber`, `C_gs_tomo_limber`, `C_gk_tomo_limber`,
  `C_gg_tomo_limber` and the six derivative tables of
  `cosmo2D_scuts.c`.
- `9dfbf7e` - the kernel-pointer maps (`Z1`, `Z2`, `N_shear`, `ZCL1`,
  `ZCL2`, `N_CL`) record their bin count and rebuild when it changes;
  `init_ntomo_powerspectra` warms them serially.
- `0b66516` - `set_source_sample`/`set_lens_sample` store the bin
  count of the saved n(z) table, so the change-detection scan never
  reads past it; a reload with a changed count stops with a message.
- `18fa2e5` - `p_gm`/`p_gg` compare the OpenMP thread count in their
  rebuild condition.
- `5e4cd6d` - `reset_cosmology_struct` frees `chi`, `G`, `lnPL` and
  `lnP` before NULLing them (no leak across `initial_setup` loops).

Values (clause 2):

- `73aab04` - `init_cosmo_runmode` resets the `Pdelta` dispatch latch
  (`reset_pdelta_dispatch`) and bumps `cosmology.random` on a
  run-mode change.
- `0a09c24` - `init_IA_fastpt` compares the stored IA selectors and
  bumps `nuisance.random_ia` on a model switch.
- `c5b6c3c` - `init_bias` compares the stored bias codes and bumps
  `nuisance.random_galaxy_bias` on a change.
- `7d9d56b` - the baryon scenario loaders draw a fresh key and
  `reset_bary_struct` steps `cosmology.random` when a scenario was
  loaded.
- `5ca65ca` - `w_gammat_tomo` and the cached ggl path key
  `tomo.random_ggl`, so an exclusion-list swap at equal length
  rebuilds.
- `421ffa5` - `init_binning_fourier_ells` bumps `Ntable.random` when
  any stored multipole changes.
- `1540cd7` - the Python FAST-PT tables carry a producer flag; after
  a covariance call rebuilds C-FAST-PT, the `IA_code = 1` read paths
  stop with "call set_IA_PS again".
- `56b21df` - the safety net: `init_ntomo_powerspectra` bumps
  `Ntable.random`.
- `518eab7` - `init_ggl_exclude` recounts the ggl spectra itself once
  both samples are loaded.

Resets (clause 3) and layout checks:

- `9c4581c` - the `initial_setup` reset batch: the low-rank trio,
  `limits.LMAX_NOLIMBER`, the baryon scenario, the `Pdelta` latch,
  `PointMass` cleared, `IPCMB::reset`, and the nuisance/like struct
  resets of the bias codes and nonces.
- `89553d3` - `IP::set_mask` records the probe blocks' total;
  the masked compute refuses a layout that no longer matches,
  naming `init_data`.
- `30c874a` - `IP::set_mask` stores the probe string; a probe change
  after `init_data` is refused the same way.
- `dc9bd17` - `like.Ndata` is assigned, not accumulated, so
  `init_data` runs twice in one process (project half: lssty1xdesi
  `cd3285c`).
- `60868da` - `IPCluster::set_mask` stores the enabled-probe flags
  and the Y transform; the cluster compute refuses a change, naming
  `init_data_cluster`.
- des_cluster `5cf3492` - the project binds `initial_setup` to a
  lambda that also calls `reset_cluster`.

Named stops and hardenings:

- `36f615a` - `PointMass::get_pm` refuses a lens bin without a set
  point mass.
- `495e404` - the non-Limber chi-grid boost cap is a named stop, not
  a silent clamp.
- `e1f710e` - the covariance entry points refuse RSD with a nonzero
  alpha_2 (the counterterm is not implemented there).
- `094920d` - cluster lensing refuses TATT second-order amplitudes
  (it implements the NLA A_1 term only).
- `8e031b7` - the run-validation guard sits above the `lx` fill in
  the non-Limber cores (mechanical move).
- `c155815` - `get_binning_fourier_space` returns the stored
  `like.ell` verbatim and stops when none is loaded.
- `45df237` - the white-noise chi grid clamps its lowest node to
  qa >= 1 (the chi = 0 node carries no volume).
- `7926262` - `IP::set_inv_cov` checks every variance of the
  covariance file before the squeeze.
- `43779ed` - three latent index hazards became named stops
  (`set_gg_cross_pairs` under an active k window, the IP/IPCMB index
  getters, `set_nuisance_IA`'s power-law branch).

The suite and the Python side:

- `d3d5b80` - the setting-sequence factory (`setting_sequence_suite`,
  `common_sequences`, the worker and its op/evaluator/check tables),
  plus `518eab7` (the "sizes" evaluator) and `34e7178` (the lens-bins
  op passes the photo-z stretch where the binding requires it).
- `eec2c60` - `lowrank_gg_suite` and `pair_quadrature_suite` validate
  their arguments with named errors.
- `cc8a4c0` - `freeze_main --overwrite` keeps only the N-models files
  of the current `NMODELS_EXAMPLES` keys and prints each dropped
  file.
- `daee920` - `covariance/survey.py` refuses a stored Gaussian model
  without the `rsd` choice.
- `24ba5c9` - the Fisher triangles use the Okabe-Ito palette with
  per-forecast line widths.
- Every project binds the suite in
  `tests/data_vector/test_setting_sequences.py`: lsst_y1 `3f74146`,
  des_y3 `54b8db1`, desy1xplanck `505d0c7`, roman_real `a8f00ec`,
  roman_fourier `5ab23d8`, roman_kl `aba76ce`, des_cluster `1a80f0e`,
  lssty1xdesi `c32b877`.

Open:

- P-33 (a user decision): hashing the frozen data files into
  `nmodels_config_hash` would change every stored N-models
  fingerprint and force the `--nmodels` refreeze of every project.
- The group-boundary full data-vector runs (lsst_y1 and lssty1xdesi,
  behind the heavy lock) are queued and running; every bound setting
  sequence, fast regression and quick frozen-reference test has
  passed on the final tree.
