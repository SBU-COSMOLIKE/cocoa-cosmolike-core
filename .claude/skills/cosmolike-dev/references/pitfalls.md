# CosmoLike pitfalls: observed numerical and performance failures

Every entry here happened. When reviewing a patch, scan this list against the
diff. When debugging, match the symptom column first — most "mystery" failures
in this codebase are reruns of one of these.

## 1. ~2x slowdown after "simplifying" a SIMD helper into an inline loop

- **Symptom:** wall time roughly doubles on a Legendre-summation loop after
  replacing a `restrict`-qualified helper with an inline
  `#pragma omp simd reduction` loop indexing `Pl[i][l]`, `Cl[nz][l]` directly.
- **Cause:** pointer aliasing. Inside a `collapse(2)` region GCC cannot prove
  the rows of two pointer-to-pointer arrays don't overlap and emits
  reload-checking code.
- **Fix:** hoist `const double* restrict pl = Pl[i];` etc. and index through
  the locals. Declaring the locals but still indexing `Pl[i][l]` in the body
  does nothing (also a real mistake that was made — the compiler only honors
  `restrict` on the accesses that go through the qualified pointer).

## 2. Non-deterministic chi2 (run-to-run, or thread-count dependent)

Two independent root causes have produced this exact symptom:

- **2a. Uninitialized FFTW work-buffer gap.** `get_FPT_IA`'s `fb` buffer had
  an unwritten region between the data and the padded end; FFTW read it.
  chi2 differed across runs depending on heap garbage. **Fix:** `memset` /
  explicit zeroing of the full allocated buffer (here a flat buffer, so
  memset is fine) before the transform. **Rule:** any buffer handed to FFTW
  is fully initialized, including padding.
- **2b. Double-checked locking race on a `static int N[]` sentinel** in
  `ZL()`/`ZS()` (`redshift_spline.c`). Plain reads/writes of a static
  sentinel under OpenMP are a data race: threads observed the sentinel set
  before the table writes were visible. **Fix:** proper synchronization
  (single-threaded init before the parallel region, or `omp critical` with
  re-check, or atomics with correct ordering). **Rule:** no new
  lazily-initialized `static` state in code reachable from parallel regions
  without real synchronization. "It's just an int flag" is exactly the bug.

Before classifying a changing chi2, pin the input files as well as the
binary. Record hashes of the data vector, covariance, mask, redshift
distributions and resolved settings. A refreeze in another process can
replace the data vector between runs: the returned theory remains
byte-identical while `(theory - data)^T C^-1 (theory - data)` takes two
different values. This occurred in the unmasked LSST Y1 TATT check; changing
only the reference data reproduced both reported values bitwise in one
process. A binary two-valued result alone does not establish a memory bug.

Triage when chi2 wobbles:

1. Recompute chi2 from the saved theory and the exact data, covariance and
   mask used by that run. Copy inputs before the sequence; a manifest that
   is itself regenerated during the sequence does not pin one version.
2. Compare the likelihood chi2 with the chi2 of the returned theory vector
   using the same installed likelihood state. The development check in
   `dev_scripts/likelihood_state_check/` records these comparisons across
   fresh processes, model orders, and one/four OpenMP threads.
3. With inputs fixed, rerun at `OMP_NUM_THREADS=1`. Thread dependence helps
   locate a race, but agreement at one thread does not prove that one
   exists. A one-thread failure can have several causes, including an
   uninitialized buffer, stale state, or changed inputs.
4. Run the DEBUG sanitizers and inspect lazy initialization and buffer
   padding. Test the suspected mechanism directly before assigning the
   discrepancy to a numerical commit.

## 3. Flat `memset`/`memcpy` over padded multi-dim allocations

- **Symptom:** corruption, sanitizer reports, or silent wrong values near row
  boundaries after zeroing a `malloc2d/3d/4d` array with one
  `memset(a[0], 0, n1*n2*sizeof(double))`.
- **Cause:** rows are 64-byte padded; the logical array is not contiguous and
  the flat size arithmetic is wrong. A related over-allocation bug existed in
  `malloc2d` itself (padding arithmetic).
- **Fix:** `zero1d/2d/3d/4d` from `basics.c` — they exist precisely for this.
  Review rule: any `memset`/`memcpy` whose target came from `malloc2d+` is
  wrong until proven otherwise.

## 4. Bare `inline` breaks DEBUG builds

- **Symptom:** undefined-reference link errors only in `COSMOLIKE_DEBUG_MODE`
  (`-O0`).
- **Cause:** C99 `inline` without `static` provides no external definition
  when the compiler doesn't inline (which it doesn't at -O0).
- **Fix:** `static inline`, always.

## 5. Pointer comparison where content comparison was meant

- **Symptom:** FAST-PT self-convolution path taken (or not taken) incorrectly;
  subtly wrong P(k) terms.
- **Cause:** `if (a == b)` on array pointers to decide "same input spectrum".
- **Fix:** compare configuration/contents. Pointers tell you about storage,
  not physics.

## 6. Double free after changing an init/allocator signature

- **Symptom:** crash in teardown (or under sanitizers) after removing an
  `init` parameter / changing ownership in a refactor.
- **Cause:** two sites both believed they owned the buffer.
- **Fix/rule:** any patch that changes an allocation signature must audit
  every `free` of that object, and run the DEBUG build to completion —
  teardown is part of the test.

## 7. "Round" FFT sizes that are terrible FFT sizes

- **Symptom:** FAST-PT mysteriously slow; profile dominated by FFTW.
- **Cause:** transform length like 10201 (= 101^2) has large prime-ish
  factors.
- **Fix:** pad through `next_fft_size` (10201 → 10240). Rule: never hand a
  raw data length to FFTW; always pass it through `next_fft_size` and zero
  the padding (see #2a).

## 8. FFTW plan churn and plan-creation races

- **Symptom:** per-call planning overhead dominating; or crashes/garbage when
  multiple threads create plans.
- **Cause:** creating plans per (bin, config, ell) call; FFTW plan creation
  is not thread-safe.
- **Fix:** static plan cache keyed by size, creation serialized, plans
  reused. Validate with two calls in one process (first call exercises
  creation, second exercises the cache).

## 9. Grid alignment off-by-one in fine/coarse grids

- **Symptom:** small but real chi2 shift after a resampling optimization.
- **Cause:** fine grid sized `x*N_a` instead of `x*(N_a - 1) + 1`, so coarse
  nodes don't coincide with fine nodes and the interpolant differs at the
  reference points.
- **Fix:** `Na = x*(N_a - 1) + 1`. Review any grid-refinement factor for this
  pattern.

## 10. Uniform-grid fast paths applied to non-uniform grids

- **Symptom:** wrong interpolation values after copying the n(z) direct-index
  trick to another table.
- **Cause:** the `idx = (x - x0)*inv_dx` lookup assumes a uniform grid.
  The inverse-distance chi column is nonuniform, so one such index does
  not identify its interpolation bracket.
- **Fix/rule:** use the current setter-validated segment metadata and
  `piecewise_index` on piecewise-uniform redshift axes. For `a_chi`, reuse
  its bucket index and short local walk. The binary-search alternatives
  have been retired; do not restore them in a covariance copy.

## 11. Trusting single-evaluation timings

- **Symptom:** optimization "wins" that vanish, or regressions that were
  never real.
- **Cause:** shared HPC nodes; one evaluation is noise.
- **Fix:** `perf stat -r 3` on the 1000-eval benchmark is the only accepted
  evidence. Instruction counts and FP-op counters are far more stable than
  wall time — quote both.

## 12. Sweeping renames/restructures mixed into functional patches

- **Symptom:** unreviewable diffs; regressions hidden inside style noise.
- **Cause:** combining a rename (e.g. `pf_photoz` → `nz_lens_photoz`) with an
  algorithmic change.
- **Fix/rule:** renames are separate, mechanical commits. Functional patches
  preserve surrounding naming and structure exactly.

## 13. Model state outside the reset structs survives `initial_setup`

- **Symptom:** a process that builds a second likelihood (a Fisher
  notebook combining probes, a test sweep over examples) dies in a
  bin-index check, e.g. `gbmag: error in selecting bin number ni = 6`
  with five lens bins, or silently computes spectra of the previous
  model's layout.
- **Cause:** `initial_setup` resets the global structs (`reset_*_struct`),
  but a file-scope static in a module (here the gg cross-pair list of
  `cosmo2D.c`) is not part of any struct. A likelihood that never calls
  that module's setter (a project without cross pairs) inherits the
  previous model's value.
- **Fix/rule:** every model-defining static gets cleared in
  `initial_setup` (`set_gg_cross_pairs(NULL, NULL, 0)` since 2026-11-04).
  A new static setter must add its reset there in the same patch.
  Validate by building a model with the feature, then one without it,
  in one process.
- **Second instance (2026-11-14):** the accuracy-boost baselines of
  `init_accuracy_boost` (generic_interface.cpp) were captured once per
  process, so a second likelihood kept the first one's internal FAST-PT
  boost. They are now forgotten in `initial_setup`
  (`reset_boost_baselines`) and re-captured per setup.
- **The rule, extended (the 2026-11-14 notebook-safety review):** every
  file-scope static outside the reset structs - model flags, switches,
  latches, loaded scenarios, and C++ singletons (`IP`, `IPCMB`,
  `PointMass`) - is reset by `initial_setup`; a cluster project's
  binding wraps `initial_setup` to call `reset_cluster`. The review
  found five more survivors: the low-rank trio of `set_lowrank_DD_GG`,
  `limits.LMAX_NOLIMBER`, the loaded baryon scenario, the `Pdelta`
  run-mode latch (cosmo3D.c), and the `PointMass`/`IPCMB` singletons.
  Their resets are part of the restoration sweep; the state of that
  sweep is in `references/state_invalidation.md`.

## 14. An allocation guard that forgets one of its sizes

- **Symptom:** heap overflow (ASAN report, corruption, or a crash) -
  or silently reused rows - after changing a bin count, pair count,
  rank, or multipole cap in one process; the same sequence is clean
  in a fresh process. A shrink shows nothing (rows are wasted, not
  overrun), which hides the bug in tests that only shrink.
- **Cause:** a lazily allocated static sized its arrays with a count
  (`redshift.shear_nbin`, `tomo.ggl_Npowerspectra`, a cap such as
  `lowrank_DD_GG_lmax`) inside a guard keyed on `Ntable.random`
  alone, and no setter of that count bumps `Ntable.random`. The
  refill tier, keyed on the redshift or nuisance nonces, then writes
  the new extent into the old allocation.
- **Fix/rule:** the rebuild guard compares, by value, every size that
  appears in the block's allocation calls - `static int NSIZE_alloc =
  -1;` and `NSIZE != NSIZE_alloc` next to the pointer-NULL check -
  even when a nonce "should" also change. An integer compare is
  free; a missed one is a heap overflow. The models in-tree:
  `C_gs_tomo_core`, the FKEM pairs core and the low-rank cross-pair
  storage (cosmo2D.c), the cluster caches, `nz_histogram_akima`
  (redshift_spline.c), `sigma_fields_` (cosmo3D.c, thread count
  included). The full invalidation rule, the audited gaps and the
  state of their restoration: `references/state_invalidation.md`.
- **Validate:** grow the size mid-process without `initial_setup` and
  compare bitwise against a fresh process at the final settings,
  under the DEBUG/ASAN build; then shrink and compare again.
