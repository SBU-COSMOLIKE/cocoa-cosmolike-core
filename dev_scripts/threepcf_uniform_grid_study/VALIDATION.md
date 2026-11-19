# Uniform-grid consolidation: numerical evidence

The production physical model uses the shared natural-cubic solver and
direct indexing of uniform log-k and piecewise-uniform redshift inputs.
Each redshift segment keeps its original endpoints and receives the same
interval refinement. The explicit upstream-comparison mode still contains
the original not-a-knot spline and distance-coordinate reader. The local
variance calculation also remains the default. These retained paths must
not be described as deleted or scientifically replaced.

## What was compared

The interpolation comparison holds the physical prescription fixed:
internal BiHalofit, the existing scalar power normalization and factorized
linear growth. It changes cubic boundary conditions and radial lookup
coordinates. The Python BiHalofit producer was changed to the same natural
conditions in production, while its upstream-comparison convention remains
unchanged. Its equation tests retain the original 2e-11 relative tolerance.

The first shared-solver change, commit `174dae9`, passed 90 component tests
in strict mode and 90 under UBSan. The later piecewise implementation passed
90 under UBSan before adding the optional shared-variance interface. The
combined private strict library and corrected project producer passed all
875 project and variance checks in one run, without skips. The debug/UBSan
library passed all 833 project checks and 42 variance checks across separate
commands. The live upstream comparison used the existing local checkouts.
Targeted equation/CLI/CAMB checks passed 95
tests; five checks cover segment joins, coarse-grid refinement, Python/C
agreement, grid rejection and the frozen small joint-vector comparison.

The small joint-vector test retains its original input, data and covariance
archives. Its original mean came from not-a-knot power/background
preparation. Exact equality to that vector is therefore replaced by the
prescribed mean-error test in its full covariance. No frozen input or
reference value was regenerated. The supplied-route refinement bound of
5e-5 remains unchanged.

## Full DES Y3 NLA comparison

Two independent processes evaluated the reference and candidate extensions
at the same 103 distinct physical points. These form fourth-order central
derivatives at spacings h and h/2 for all 17 parameters. Each result records
the source files, covariance, row definitions, binary identity, every
parameter dictionary and hashes of the generated physical tables. The
comparison requires identical table hashes and bitwise-identical two-point
entries across the entire set of points.

The 480-row vector contains all 400 two-point and 80 aperture entries from
the project's stated demonstration configuration. Its full fixed covariance
includes the two-/three-point cross block. If C = L L^T, the mean score is
the squared norm of L^-1(new - reference). Each derivative score divides
the norm of its whitened change by the norm of the reference whitened
column. Fisher matrices retain the same parameter basis and priors.

| Quantity | h | h/2 | Required bound |
|---|---:|---:|---:|
| Mean squared displacement | 8.53e-9 | Same fiducial | 0.01 |
| Largest relative whitened derivative change | 9.89e-6 | 9.88e-6 | 1e-3 |
| Largest fractional marginalized-error change | 5.31e-5 | 5.35e-5 | 0.005 |
| Largest fractional pair-FoM change | 5.81e-5 | 6.25e-5 | 0.005 |

All two-point values are bitwise unchanged at all 103 points. Every pair
of forecast parameters was checked. These values establish the NLA
comparison for this configuration; TATT and further coarse-grid refinement
remain separate checks. They do not establish a reproduction of either
published analysis or justify changing the physical approximations.

## Shared variance is a separate decision

Optional copied fit and halo-variance inputs are described in
[the variance validation record](../threepcf_variance_check/VALIDATION.md).
They allow a consumer to reuse shared core work while retaining immutable
ownership and explicit stale-state refusals. They have not replaced the
local variance or its scalar normalization in the project provider.

A fixed NLA mean with factorized input power and the shared fit snapshot
gave a squared displacement of 2.39e-4 from the original mean. That single
point is not a derivative or Fisher acceptance test. The separate
[FFTLog endpoint correction](../threepcf_variance_check/FFTLOG_TAILS.md)
addresses the earlier unbroken-power-law error with analytic and derivative
checks. Those component checks do not replace the forecast comparison.

## Evidence locations

The Cocoa workspace record is
`to_reproduce/threepcf_consolidation_20261119/`. It contains the producer-grid
audit, frozen-mean driver, full-stencil driver, paired analysis and
`nla_forecast_comparison.json`. The private build and result directory is
`/private/tmp/threepcf_consolidation_20261119/`; strict and debug binaries are
kept apart. Installed project extensions are unchanged by these checks.
Numerical jobs are serial, BLAS uses one thread, and OpenMP is capped at six.
The final strict log is `final_strict_combined.log`. Debug evidence is in
`full_debug_combined.log`, `final_debug_variance.log` and
`final_debug_live_original.log`: the first command exposed incomplete test
environment paths for the separate variance tests; the corrected command
passed those tests. Both final libraries were built with the actual project
Makefile and pinned sources, without the earlier private build fragment.
