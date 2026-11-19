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
`test_fit_input.py` also requires `VARIANCE_CORE_SOURCE` pointing to that
build's core headers, plus matching `CONDA_PREFIX` and `C_COMPILER` values.
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

An unresolved shared-engine accuracy finding remains: an unbroken
`P(k) proportional to k^-2` spectrum gives a nonlinear-radius discrepancy
of 9.72% and a sigma8 discrepancy of 11.69% against its analytic infinite
integral. No documented input-domain exclusion was established for this
case. The bias-matched `k^-2.2` normalization test passes, but it does not
resolve that finding. The original failed checks and slope probes remain
in `/private/tmp/threepcf_variance_20261119/` alongside source hashes,
binary identities and both successful logs.

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
derivative and Fisher acceptance criteria and the unresolved power-law
accuracy finding above remain applicable.

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
