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
(The bump is part of the restoration sweep below; until it lands,
clause 1 is the only protection.)

The cost is an integer compare per call or one nonce draw per setup;
nothing rebuilds per MCMC step. The two failure directions are
covered from both sides: a caller that resizes the samples but skips
`init_ntomo_powerspectra` is caught by the per-cache size compare,
and a future cache whose author forgets a size key is caught by the
safety-net bump. Either alone leaves one direction open, which is
exactly how the audited gaps arose.

## The failure classes, with one example each

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
- **A derived mask that is not stamped.** `IPCluster::set_mask`
  (generic_interface_cluster.cpp) does not record the probe flags and
  the Y-transform switch that shaped the mask, and `IP`'s evaluation
  never re-checks the data-vector layout against the current binning
  and probe flags, so an `init_binning_*` or `init_probes` call after
  `init_data` silently scores the wrong layout.

## The idioms to copy

- **Size keys next to the NULL check:** `static int NSIZE_alloc = -1;`
  and `NSIZE != NSIZE_alloc` in the rebuild condition, set inside the
  block - cosmo2D_scuts.c (the derivative tables) and basics.c are
  in-file models.
- **Every size in one guard:** `C_gs_tomo_core` keys
  `Ntable.random + SIZE1 + NSIZE`; the FKEM pairs twin and the
  low-rank cross-pair storage (`C_cl_tomo_lowrank_core`) key npairs,
  nbins, nchi and the cap in dedicated cache slots.
- **A reset the interface calls:** `IPCluster::reset`, and
  `initial_setup`'s existing `set_gg_cross_pairs(NULL, NULL, 0)`,
  `unset_gg_kwindow()` and `reset_boost_baselines` calls. A new
  static setter adds its reset there in the same patch.
- **A layout check at evaluation time:**
  `compute_data_vector_cluster_masked` recomputes the expected data
  length and stops with a named message when it disagrees with the
  stored mask - the pattern every masked-evaluation path should run.

## How to test it

The test design is the **setting-sequence suite**: a shared factory
(`setting_sequence_suite` in `cocoa_testing_suites.py`, planned next
to `accuracy_suite` and `pair_quadrature_suite`) that each project
binds with its catalog of sequences. For each sequence, one fresh
worker process builds the base configuration and applies the setting
changes step by step, evaluating after each; a second worker builds
the base plus the cumulative final settings in one pass. The final
in-process vector must equal the fresh one bit for bit (float64
equality, no tolerance) - or, for sequences whose contract is
refusal, the in-process worker must abort with the named message.

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
raised range, accuracy-boost round trips, IA/bias model round trips
at identical amplitudes, run-mode and baryon round trips, the
two-model `initial_setup` round trip, and the layout-change refusals
after `init_data`. Heavier variants (the full probe matrix, an
AddressSanitizer leg for overflows that do not change the numbers,
a repeated-`initial_setup` leak loop) are studies behind environment
keys, in the style of the existing deep suites.

## Status

Landed (cosmolike_core, branch `lowrank_fftlog`):

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

Open: the remaining restorations of the audit - the per-cache size
keys of the classic projection and interpolation caches, the
`init_IA`/`init_bias`/baryon/run-mode nonces, the `initial_setup`
reset batch (low-rank trio, `limits.LMAX_NOLIMBER`, the baryon
scenario, `PointMass`/`IPCMB`), the evaluation-time layout checks,
the `init_ntomo_powerspectra` safety-net bump, and the
setting-sequence suite itself - are being implemented from the
review's fix plan; treat any path above as unprotected until its fix
is in the tree. The n(z) change-detection scan's out-of-bounds read
(`set_lens_sample`/`set_source_sample`) and the RSD-masked pair-node
initialization are in the same open list.
