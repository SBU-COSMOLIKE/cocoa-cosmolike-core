# Copied variance inputs for three-point covariance

This stage adds `core.variance_snapshot` and optional variance inputs to
the covariance adapters. It does not select them by default. The snapshot
reads the installed evolving linear total-matter spectrum; it does not
install a cosmology, renormalize that spectrum, or reproduce the legacy
factorized-growth approximation automatically.

The snapshot owns its returned arrays. Covariance consumption requires
matching redshift nodes, mass nodes and reference density, positive
variance and negative logarithmic mass slope. The mass-to-radius density
conversion preserves the covariance's existing critical-density constant.
The Sheth–Tormen and Duffy prescriptions remain unchanged. Both standalone
and ordinary `initial_setup` explicitly release the shared variance caches.

The initial private build included `cosmo3D.c`, `baryons.c` and HDF5 through
[`variance_build.mk`](variance_build.mk). The project Makefile also lists
these dependencies explicitly, alongside `halo_profile.c`, so the core
variance, NFW and baryon symbols resolve inside the standalone extension.
Its pre-existing bridge additions remain separately owned in Git.

For the subprocess checks, `PYTHONPATH` must include the selected extension,
the project's `tests/` and `tests/data_vector/`, and this directory.
The public `core.variance_sampling` and `core.set_variance_sampling`
bindings exercise accuracy changes directly; current tests no longer
compile the external C helper used in the earlier build records.
Each refusal test prints a marker immediately before the intended request;
an import or setup error cannot establish a successful validation refusal.

Strict and debug/UBSan builds each passed all 19 checks in
[`test_snapshot.py`](test_snapshot.py). They cover copied ownership,
amplitude replacement and restoration, reset, one/six-worker equality,
fit-only and halo-only requests, both covariance adapters, mean and
non-halo term invariance, and rejection of incomplete or mismatched inputs.

For a curved spectrum, direct smoothing quadrature gave maximum relative
differences of `2.55e-4` in the nonlinear radius and `1.45e-4` in the
mass variance; the effective-index absolute difference was `4.01e-5`.
These are component checks at controlled inputs, not survey acceptance.

The power-law accuracy finding is addressed by the analytic endpoint
correction documented in [FFTLOG_TAILS.md](FFTLOG_TAILS.md). Its expanded
54-test strict and UBSan checks include smaller derivative steps and
independent quadrature. The earlier failed checks remain part of the
record; ordinary turnover and DES fiducial snapshots are byte-identical
before and after the correction.

The full joint mean, derivative and Fisher acceptance checks remain
required before replacing the legacy variance calculation. The generic
full-project interface's new reset call also needs a full-project rebuild;
these private builds exercise the standalone reset entry point.


## Optional fit inputs in the model constructors

Both constructors now accept `variance={"z": ..., "rnl": ..., "neff": ...,
"sigma8": ...}`. These four copied arrays replace local fit-variance work
and retain the amplitude of the supplied power spectrum. The old constructor
behavior remains the default. Fit redshifts may use up to ten uniform
segments, following the same grid contract as the background.

`core.set_variance` keeps its own copies, stamps `cosmology.random` and
`Ntable.random`, and refuses a stale snapshot after spectrum, cosmology or
accuracy changes. Callers must explicitly resupply or clear it. Model owners
already returned remain valid after replacements and resets. A full snapshot
also has halo arrays; callers select the four named fit arrays explicitly.

The combined private strict and debug/UBSan gates each passed 42 tests:
the 19 snapshot/covariance checks above and 23 constructor checks in
[`test_fit_input.py`](test_fit_input.py). The latter exercise both adapters,
supplied-B construction, power-amplitude preservation, each fit quantity,
sliced notebook input, array ownership, A/B/A replacement, reset, stale-state
refusal for power/cosmology/accuracy, resupply, valid two-piece grids and
excess-segment refusal. One check feeds actual shared FFTLog outputs through
both direct and bridge constructors. The build record is
[`fit_validation.json`](fit_validation.json); private sources and full logs
are in `/private/tmp/threepcf_fit_variance_20261119/`.

These checks do not adopt the new engine for forecasts. Shared variances
read evolving linear power, whereas the model's tree and IA power legs still
follow their existing selected growth prescriptions. The full mean,
derivative and Fisher acceptance criteria remain applicable. The separate FFTLog correction does not establish
forecast acceptance or select the shared route by default.

## Combined validation

The later standalone builds used the actual updated project Makefile,
with pinned core sources. The strict command passed all 875 project and
variance checks without skips. Under UBSan, the 833 project checks and
42 variance checks passed across separate commands. The first combined
debug command lacked source/compiler paths for the external harness;
those environment errors were corrected and the affected checks rerun.
The final strict command includes the strengthened refusal marker check.
Logs are in `/private/tmp/threepcf_consolidation_20261119/`, named
`final_strict_combined.log`, `full_debug_combined.log`,
`final_debug_variance.log` and `final_debug_live_original.log`.

## Shared sampling controls

The standalone interface exposes exact counts for top-hat mass and scale
factor, plus Gaussian radius, scale factor and inverse tables. A dictionary
copied after `initial_setup` supplies the baseline. Reinstalling unchanged
counts preserves caches. Changing any count invalidates shared tables and
previously supplied bridge fit arrays, while returned immutable models and
snapshots remain usable. The setter validates all counts before mutation.

All five counts passed individual A/B/A restoration and stale-fit refusal
checks. The resulting suite passed 63 tests in strict mode and 63 under
UBSan. Logs are `sampling_strict_tests.log` and `sampling_debug_tests.log`
under the same private validation directory. These controls expose existing
Ntable sampling; they add no smoothing prescription or integration rule.
