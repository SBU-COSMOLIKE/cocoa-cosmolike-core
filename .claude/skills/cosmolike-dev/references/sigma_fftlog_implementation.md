# Evolving variance and cb halo statistics: implementation record

The evolving matter and cb variance tables supply halo peak heights, mass
slopes and concentration growth. The measurements below identify the
2026-10-03 implementation configuration; the covariance-tail configuration
is stated separately. Raw scripts and logs reside in
`test/neutrino_growth_study/implementation/`.

## Calculation and public interface

The covariance-tail adoption on 2026-10-06 supersedes the original grid
choices below: bias 0.8, high-k continuation through 1e25 h/Mpc, and mass
coverage through 1e-40 Msun/h. Extra low-mass nodes retain the existing
lnM spacing instead of stretching the old grid. The HOD integration range
remains 1e4--1e17. See `covariance_mass_cutoff.md` for the adoption tests;
the older measurements below describe their recorded configurations.
The initial bias-0.5 trial failed Roman's independent variance check.
Bias 0.8 reduces its maximum tested error from 1.8265e-4 to 6.9887e-6
without doubling the FFT workspace. A separately doubled FFT interval
at bias 0.5 agrees within 1.32e-8 relatively down to 1e-40 Msun/h.
See `wynn_validation_20261006.md` for the tail-stability validation.

The shared smoothing engines subtract growing weighted endpoint power
laws and restore their analytic Mellin integrals. This fixes periodic-image
errors for spectra such as P proportional to k^-2 while retaining the
fixed bias, cached kernels and retained plans. A bias-matched constant
also needs this treatment for stable numerical derivatives. Ordinary
turnover and DES fiducial snapshots are byte-identical across the change.
The [endpoint correction](#endpoint-correction-and-validation) below gives
the identity, convergence conditions and measured limits. It applies to the
shared top-hat and Gaussian engines in `cosmo3D.c`, including halo consumers
outside 3PCF; it does not change the angular transforms in `cosmo2D.c` or
the separate two-dimensional 3PCF FFTLog transform.

`cosmo3D.c` builds sigma_m^2(M,a), sigma_cb^2(M,a), and their logarithmic
mass slopes from the evolving linear spectra. The measured 2026-10-03
configuration used FFTLog bias 1.5,
the supplied uniform logarithmic k spacing, edge-power continuation to
1e-7 through 1e5 h/Mpc, a smooth taper over the upper quarter of Fourier
frequencies, and the analytic Mellin transform of the spherical top-hat
window squared. A second inverse transform differentiates that kernel.
Sixteen extra radial nodes at each mass boundary isolate the natural
spline's end condition. The final tables use the existing dense mass
and scale-factor grids (1024 by 256 at the default settings), storing
ln(sigma^2/a^2) to reduce early-time interpolation curvature.

The Python diagnostic is `sigma2(M, a=1, field=0)` (0: total; 1: cb).
Production C `sigma2(M,a)` always selects cb. `dlognudlogm(M,a)` reads
the evolving cb slope. `conc(M,a)` takes scale factor, not a growth
factor, and uses D_cb(M,a) = sigma_cb(M,a)/sigma_cb(M,1).
The halo-field initializer and YAML option have been retired.

All halo peak heights and mass-function densities use cb. The M200m
NFW radius still uses the total mean density defining that mass, and
lensing weights and nonlinear matter spectra retain total matter.
The Bhattacharya growth prescription is an adopted extension of a fit
without a massive-neutrino calibration. EMUL2 retains its pre-existing
approximation P_cb = P_lin/(1-f_nu)^2. CAMB supplies the actual cb table.
The external DES reference keeps its non-halo growth convention and
integrates the evolving cb spectrum independently for halo statistics.

## Endpoint correction and validation

The smoothing variance is the integral of the supplied dimensionless power
`Delta^2(k) = k^3 P(k)/(2 pi^2)` against a window squared. FFTLog represents
`k^-b Delta^2(k)` as periodic in `ln k`, with fixed bias `b = 0.8`.
For an endpoint continuation `Delta^2 = A k^s`, the weighted function grows
toward high k when `s > b`, or toward low k when `s < b`. Its periodic
repetition can contaminate radii inside the requested range; extending
the same growing continuation does not cure that error. An unbroken
`P proportional to k^-2` stress test gave a 9.72% nonlinear-radius error
and an 11.69% sigma8 error before the correction. Those are analytic test
errors, not measured errors in a DES forecast.

### Analytic subtraction

For each problematic continuation, subtract `A k^s` at every wavenumber,
transform the residual, and add `A R^-s M_W(s)` to the variance. Here
`M_W(s) = integral dx x^(s-1) W(x)^2` is the analytic Mellin integral of
the selected window. For the Gaussian window squared it is
`Gamma(s/2)/2`. For the spherical top-hat window squared it is

```math
M_{\rm TH}(s)=
\frac{9\pi\,\Gamma(4-s)\Gamma(s/2)}
{2^{5-s}\Gamma^2((5-s)/2)\Gamma(4-s/2)}.
```

The top-hat expression requires `0 < s < 4`. The Gaussian expression
requires `s > 0`. These are convergence conditions of the analytic
counterterm. A nonpositive low-k exponent makes the original infinite
variance diverge; a top-hat high-k exponent at least four also diverges.
Those cases are refused rather than assigned a finite truncated answer.
The radial derivative of each added term is `-s` times that term.

The input spectrum and its endpoint continuation remain the same. The
subtraction changes the numerical representation of their integral. Each
row owns two small endpoint descriptors; no new persistent cache exists.
The fixed-bias FFT kernels, retained plans and threading remain unchanged.
Outside an endpoint, the matching residual is zero by algebra. Evaluating
that zero by subtracting two large rounded exponentials manufactured false
power in the initial independent experiment, so the C implementation avoids
that subtraction. Interior near-cancellation uses `expm1`.

A bias-matched high-k tail is included in the correction. Leaving that
single spectrum on the zero-padded path gave a small mean offset but a
0.366% error in a spectral-slope derivative at step 0.00025. The corrected
path passes the unchanged 0.1% derivative bound at steps 0.001, 0.0005 and
0.00025. The shared threshold on endpoint slopes prevents roundoff from
selecting different treatments for an exactly matched spectrum.

### Checks and limits

The expanded component suite passes 54 tests in strict mode and 54 under
UBSan. It includes five analytic spectral slopes, curved spectra activating
neither, either or both corrections, derivative-step checks, one/six-worker
equality, reset, copied ownership and stale-state refusals. The power-law
and curved-spectrum accuracy bounds remain 0.1%.

The independent low-tail quadrature must extend below k = 1e-7 h/Mpc.
For `Delta^2 proportional to sqrt(k)`, truncating there omits approximately
0.001 of variance. Its reference now extends to 1e-16 h/Mpc. The low-tail
curved case has a z = 1 unit crossing near R = 0.005 Mpc/h. At the default
Gaussian-table resolution its radius error is 0.1088%; refining the existing
scale-factor and inverse tables reduces it. The test requires a decrease
by more than a factor of three and the original 0.1% bound after two nested
refinements. This is a recorded resolution requirement, not evidence that
default sampling suffices for every spectrum.

Separate processes captured an ordinary turnover spectrum and the saved
DES fiducial spectrum, with identical input arrays. All 18 snapshot fields
are byte-identical before and after the correction. The mass queries span
401 samples inside the density-adjusted supported mass range, at three
redshifts. These checks cover total matter with factorized input growth;
they do not establish a cb-field forecast or the final shared-variance
adoption for the three-point provider.

## Threading experiment

Hardware: Apple M2 Pro. Strict IEEE default build. Each timing warms the
FFTW plans first, then changes Omega_m between refills so the table is
rebuilt. Timers surround the C table construction only: no CAMB or input
copying is included. FFTW plans are created serially on allocation and
reused across cosmologies with separate, aligned arrays per OpenMP worker.
The Gamma-function factors depend only on the grid and are cached too.

The production layout completes a (field,a) row in one worker: one forward
FFT, two inverse FFTs, normalization, and splining. The alternative saves
all forward coefficients, splits inverse transforms with collapse(3) over
(field,a,derivative), then normalizes and splines after a barrier. It
performs the same number of FFTs but needs additional arrays and copies.

Mean milliseconds per refill, three repeats of 100 refills:

| Threads | Complete rows | Split inverse transforms |
|---:|---:|---:|
| 1 | 20.5542 | 20.7780 |
| 4 | 5.4216 | 5.5512 |
| 8 | 2.9569 | 3.1638 |

A longer run reversed the variant order to check warming/order effects:
three repeats of 1000 refills at eight threads gave 2.9993 ms for complete
rows (3.0100, 3.0255, 2.9623), versus 3.1809 ms for the staged layout
(3.1743, 3.1757, 3.1926). The staged layout was about 6% slower there.
Both layouts and one/eight-thread results were bit-identical across both
fields, two neutrino masses, four scale factors and 1024 mass nodes,
including the mass slopes. Keep complete rows: 2 x 256 = 512 independent
rows already provide 64 rows per worker at eight threads.

These are variance-table timings, not whole-likelihood speedups. Linux
`perf` hardware counters are unavailable on this macOS host.

## Documentation and numerical review

The comments derive smoothing, biasing, the Mellin integral, and its
mass derivative before discussing implementation. The spline helper
explains its tridiagonal equations and boundary conditions. The FFT-size
helper explains radix decomposition. The reader gives a fractional-node
example, the upper-edge bracket rule, table-index meanings, and the two
steps of bilinear interpolation. Like-shaped work arrays share a role
dimension; struct members have one declaration and explanation per line.
Cache allocation conditions follow cosmo2D.c directly in the `if`.
Guards validate inputs at their boundary; speculative recovery must not
conceal invalid numerical results.

Independent validation of the measured configuration:

- C against the independently implemented Python FFTLog study: maximum
  relative discrepancy 1.83e-13 in variance and 5.5e-13 in slope.
- Independent direct Simpson quadrature at five cluster masses, four
  scale factors and two neutrino densities: maximum variance discrepancy
  1.81e-5 (acceptance 1e-4).
- Roman halo invariants, including the optional slow integrals and
  spectra: 25 passed (31.69 seconds).
- Saved-reference chi2 drift measured before regeneration, with only
  the retired configuration key removed in memory:

| Project | Maximum absolute chi2 drift |
|---|---:|
| lsst_y1 | 0.033724 |
| roman_real | 0.076432 |
| roman_fourier | 0.003509 |
| roman_kl | 0.006486 |
| des_y3 | 0.052772 |
| desy1xplanck | 0.054075 |
| des_cluster | 0.151836 |

All are below the existing 0.2 threshold. Refreezes also remove retired
options and update the halo probe interface. Only the documented
reference generator writes frozen data and manifests; survey observations
and shipped synthetic vectors are not regenerated for this change.

## Direct indexing and grid assumptions

The FFTLog redshift lookup reuses `piecewise_index` and the existing
lnPL segment metadata. A before/after comparison at every one of the 256
scale-factor rows, 33 masses from 1e6 to 1e17 M_sun/h, both fields and both
saved neutrino masses was bit-identical for variance and slope (67,584
values). This check precedes the retirement of the build flag below.

`cosmo3D.c` uses direct indexing for its piecewise-uniform grids.
`COSMO3D_ASSUME_PIECEWISE_UNIFORM` is not a selectable build option. The segment
metadata, detector and setter calls are unconditional, including in debug
builds; all seven Makefiles remove the flag. The retained C/C++ tokens in
cosmo3D.c, generic_interface.cpp, basics.c, basics.h and structs.h were
checked against the former enabled branch and are unchanged. The existing
bucket lookup of a_chi is retained because its chi grid is not uniform.
No binary-search implementation remains in cosmo3D.c.

The full unmasked Roman 3x2pt vectors also agree bit for bit with the
pre-retirement library: NLA and TATT, 2115 entries each, at the fiducial
cosmology, a joint Omega_m/H0 detour, and the return to the fiducial point.
All six vectors include points excluded by the survey mask. A temporary
all-ones mask and unit covariance are installed before initialization;
no frozen or survey data is edited. The current side includes permanent
OpenBLAS pinning too. Raw result: full_vector_comparison.log.

The drift table measures all changes since each frozen snapshot, not the
isolated effect of the evolving variance tables. In particular, the non-halo configurations
have HOD and halo IA disabled: their drift includes the previously committed
growth-k change to 0.05/Mpc. Their September frozen configurations
do not pin growth_k and therefore resolve that value from the live code.
The cluster configurations additionally change through the evolving cb halo
statistics. Reference-update commits must state both contributions.

## Independent cluster comparison

The independent Python reference self-tests passed (30 tests). With
matched settings, counts agree to 9.27e-6 relative, richness-bin number
densities to 1.39e-5, and richness-bin biases to 3.12e-6. The joint cluster
data-vector difference has delta chi2 = 7.512e-6 with the Y6 cuts and
3.031e-5 with the extended positive-definite mask. With nonzero NLA and
shear calibration, the corresponding cluster-lensing differences are
6.984e-6 and 2.815e-5. Production settings give joint differences of
5.037e-5 and 0.001865, respectively, below the 0.2 budget.

The stricter 1e-4 row-level target is not met everywhere: three of the
33 matched-setting rows (cluster-lensing C_ell, gamma_t and its Y-space
transform) miss it. These numerical limitations remain visible in the
comparison report; a passing data-vector budget does not imply that every
individual row passes. The historical unshifted source-tail diagnostic
intentionally compares different source-support conventions. The nuisance
check instead matches the core's existing photo-z-shifted support.

The full joint vector, its cosmology detour, and its return to the original
point are bit-identical at one and eight OpenMP threads. The separate
massless Lighthouse comparison still shows the documented coarse-CQUAD
precision differences (about 2.29% in counts and 0.759% in bias); it is
reported separately from the independent matched-settings oracle.

## Permanent OpenBLAS thread limit

OpenBLAS stays at one thread; parallelism belongs to
the explicit CosmoLike OpenMP loops. A shared runtime lookup calls
openblas_set_num_threads(1) at module import, initial setup, OpenMP team
configuration, and before both covariance inversion paths. There is no
restoration of a larger BLAS team. Other BLAS backends need not export this
entry point. Linux builds link libdl for the runtime lookup.

The ordinary covariance path also checks the inverse in correlation units:
R_ij = C_ij/(sigma_i sigma_j), (R^-1)_ij = (C^-1)_ij sigma_i sigma_j,
with max |R R^-1 - I| < 1e-8, before applying the final inverse mask.
The cluster path retains its corresponding check. No retry or alternative
factorization is added.

The controlled cluster check deliberately sets OpenBLAS to eight threads
before each of three inversions of the real survey covariance. Each leaves
OpenBLAS at one and the CosmoLike OpenMP team at four; inverses are
bit-identical and their correlation residual is 3.93e-13 or smaller.
Three fresh LSST model initializations with the same deliberately raised
BLAS limit likewise leave BLAS at one and OpenMP at four, produce
bit-identical inverses, and have residual 1.19e-13 or smaller.
The same controlled three-inversion check passes in all seven projects,
including Roman-Real, Roman-Fourier, Roman-KL, DES-Y3 and DESxPlanck:
OpenBLAS remains at one, the configured OpenMP team remains at
four, and repeated inverses agree bitwise. Calls requesting one, four
and eight OpenMP threads likewise preserve that team while pinning BLAS.

The aggressive build still fails its covariance residual check despite
the enforced BLAS limit. An isolated build of the pre-change core
(4adf735) and cluster interface (aa32535) reproduces this failure, with
OpenBLAS explicitly at one thread and OMP_NUM_THREADS=1: max |R R^-1 - I| = 4 during dummy-data
initialization, before any cosmology or variance calculation. This is a
pre-existing aggressive-mode limitation, not a passing validation; the
default build passes and the residual guard correctly stops the bad
aggressive result. Raw reproduction: baseline_aggressive_cluster_init.log.

The supported Makefiles reject the aggressive-mode environment flag
because its fast-math configuration fails covariance-inverse checks. The
ordinary Roman covariance guard also catches an aggressive-build residual
of 83919.9135. Its selected tests had 34 passes and five failures (four
inverse guards and one 1.47e-12 halo rounding mismatch against a 1e-12
frozen tolerance). These failures are recorded, not counted as passing
validation. The supported configurations are the optimized strict-IEEE
default and debug. Installation guidance and the development skill list
the unsupported floating-point flags; no individual flag was isolated
as the cause. All seven retired-mode rejection checks passed.

## Project validation of the measured configuration

Every suite runs in a started cocoapy311 Cocoa environment, with
OMP_NUM_THREADS=4 and COCOA_HALO_SLOW=1. BLAS is held at one by the
compiled interface as well as the test shell. Command from Cocoa/:
`python -m pytest projects/PROJECT/tests -q`.

The 2026-10-03 default-build validation, including permanent BLAS pinning:

| Project | Result | Elapsed seconds |
|---|---:|---:|
| des_cluster | 29 passed | 577.39 |
| lsst_y1 | 57 passed | 891.94 |
| roman_real | 104 passed | 1367.35 |
| roman_fourier | 45 passed | 1093.93 |
| roman_kl | 49 passed | 1564.64 |
| des_y3 | 63 passed | 1191.31 |
| desy1xplanck | 45 passed | 948.81 |

All seven suites passed: 392 tests total. The independent Python
cluster reference suite adds 30 passing tests. Frozen references were
regenerated only with the documented project generators and are committed
separately from implementation changes.

The final debug builds also pass the cluster neutrino check (1 test,
45.24 seconds) and Roman halo plus NLA/TATT example checks (39 tests,
459.72 seconds), with the configured undefined-behavior and floating-point
division sanitizers enabled.
