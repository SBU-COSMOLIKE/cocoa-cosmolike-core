# Three-point predictions and covariance

The numerical port lives in `cosmolike/3pcf/`. It now predicts the physical
three-point mean and a specified joint two-/three-point covariance. The
companion Cocoa project owns survey input, optional CAMB preparation and
Fisher orchestration. Existing two-point C files are unchanged. This is an
implemented numerical model, not a demonstrated reproduction of DES Y3 or
Y6 or a measured performance result. Read the estimator and physical limits
before combining its outputs.

For cache changes, accuracy controls or notebook reuse, read
[state and resolution](threepcf_state_and_accuracy.md). It maps the ordinary
core's allocation/refill/reset rules onto the isolated owners and identifies
the fresh-process setting-sequence tests, including the outer Fisher cache.

The [archived study](threepcf_study/index.md) contains 24 chapters and an
index, with mathematical derivations, pinned source comparisons and proposed
acceptance tests. It is a design reference, not a feature list. Earlier
proposals placing this work in `cosmolike/fastnc/` or the existing two-point
`cosmolike/covariances/` directory are superseded by `cosmolike/3pcf/`.
The archive omits downloaded source trees and machine-specific paths.

## Implemented model and entry points

A matter bispectrum describes correlations among three Fourier modes whose
wavevectors close into a triangle. Intrinsic alignment adds correlations
between galaxy shapes and the tidal environment. The C physical model
accepts power spectra, distances, growth and source-redshift distributions;
it calculates BiHalofit and the selected TATT/velocity-shear terms, then
projects them into angular statistics. TATT means tidal alignment and tidal
torquing. Its quadratic contributions must retain their assigned Fourier
legs and source labels.

| Layer | Source | Implemented responsibility |
|---|---|---|
| Physical model | [physics_3pcf.c](../../../../cosmolike/3pcf/physics_3pcf.c) | Owned input tables, BiHalofit, 19 active intrinsic-shape components, source kernels, fixed radial projection, NLA angular power and direct full-filter aperture integration |
| Prepared angular tables | [physics_grid_3pcf.c](../../../../cosmolike/3pcf/physics_grid_3pcf.c) | Baseline and signed-residual triangle charts, shared source-triplet preparation, Legendre coefficients and dense production readers |
| Angular transform | [coupling_3pcf.c](../../../../cosmolike/3pcf/coupling_3pcf.c), [fftlog_3pcf.c](../../../../cosmolike/3pcf/fftlog_3pcf.c), [transform_3pcf.c](../../../../cosmolike/3pcf/transform_3pcf.c) | Legendre coupling, two-dimensional FFTLog, signed angular modes, radial averaging and the four complex natural shear components |
| Finite aperture estimator | [aperture_3pcf.c](../../../../cosmolike/3pcf/aperture_3pcf.c) | Reusable weights for measured side-angle-side bins and contraction with the declared natural-component convention |
| Physical joint covariance | [covariance_physics_3pcf.c](../../../../cosmolike/3pcf/covariance_physics_3pcf.c) | Gravitational projections, halo abundances/profiles, binned xi spin kernels, six separate covariance families and their two-/three-point cross blocks |
| Supplied-array contractions | [covariance_3pcf.c](../../../../cosmolike/3pcf/covariance_3pcf.c) | Filtered pairs, fifteen Gaussian cubic-field pairings and a weighted response Gram matrix; these lower-level functions do not construct a halo model |

Use the [module guide](../../../../cosmolike/3pcf/README.md) for conventions,
[NUMERICS.md](../../../../cosmolike/3pcf/NUMERICS.md) for preparation and
reuse, and [COVARIANCE.md](../../../../cosmolike/3pcf/COVARIANCE.md) for the
actual covariance equations. A catalogue estimator, a full-filter map
estimator and a finite triangle integral can have equal array dimensions
while measuring different quantities. Matching their labels is insufficient.

The production C++ adapters validate and borrow aligned, C-contiguous
NumPy arrays without conversion. The notebook adapters use owned Armadillo
copies and explicit layout conversion. Both call the same C numerical
functions; neither may acquire a separate scientific implementation. The
headers define axes, units and ownership. Wave numbers use h/Mpc, distances
use Mpc/h, power uses (Mpc/h)^3, and angles use radians. Growth is D(z),
normalized to one today. The ordinary core's D/a table requires the explicit
conversion supplied by the project bridge.

## Numerical strategy required for changes

The production sequence is **coarse physical calculation, natural-cubic
preparation, dense uniform table, arithmetic-index linear read**. Uniform
spacing refers to the stored coordinate, such as log wave number, redshift,
or a bounded triangle-shape coordinate. Input files need not already use
those grids: conversion belongs in preparation. Retain named upstream
reference readers for cross-code comparisons, rather than putting their
binary searches or repeated splines into the production calculation.

For a uniform coordinate x, the cell is floor((x-x0)/dx), and its fractional
position specifies the two linear weights. Prepare those indices and
fractions once when repeated outputs request the same coordinates. The
xi path in `cosmo2D.c` provides the template: E/B or multiple Legendre
outputs share the map instead of repeating index arithmetic. Explicit
SIMD lanes evaluate neighboring outputs or independent coefficients, with
contiguous storage wherever possible. Scalar tails and boundaries must
remain inside the supplied buffers. Preserve strict arithmetic; a compiler
fast-math flag is not a substitute for identifying independent work.

In a Limber read, log(k)=log(ell)-log(chi). Prepare the two logarithms and
fixed redshift brackets before the repeated power reads. The physical
model's NLA power batch uses that shifted coordinate and shared indices
for both nonlinear-power redshift rows. A fixed physical-k table and a
fixed-ell table are not interchangeable: changing distance moves k at a
fixed ell. Include that dependence in any proposed reuse.

Coarse and dense resolutions control different errors. More dense nodes
approximate the chosen cubic curve more closely; they cannot recover a
feature missed by coarse physical sampling. Likewise, integration nodes
and lookup nodes have different roles. Fixed Gauss-Legendre or trapezoidal
quadrature may use nonuniform nodes while reading a uniform table. Keep
node counts, ranges and Jacobians explicit. Logarithmic integration weights
already contain the change of variable; do not multiply that factor again
inside C. No adaptive refinement loop runs during a Fisher displacement.

Positive baseline tables may use logarithmic values where the model
supports that representation. Signed residual IA tables may cross zero
and must preserve their sign. Preparation omits an identically absent
residual only for the exact coefficient condition established in C; a
small nonzero residual must not be dropped through a tolerance. Source
triplet batches share physical matter/IA work before different radial
weights are applied. `projection_batch_size` bounds the number of prepared
source tables held together; consume and release a batch before constructing
the next one. Full-filter means do not require Legendre tables.

FFTW plan creation, memory allocation and initialization belong before
OpenMP workers start. One parallel layer owns a phase. Transform workers
execute serial FFTW plans through the new-array API on separate aligned
buffers; this is outer OpenMP parallelism, not internally threaded FFTW.
Retain Mellin kernels, geometric phases and selected output offsets in the
plan. Reuse forward coefficients across different Bessel orders and equal
inverse kernels where the identity holds. The two-order Gamma recurrence
needs high-order checks against direct complex Gamma evaluations, not only
low-order tests. Cap scratch capacity and worker teams by available tasks.

## Scientific conventions that must survive optimization

The physical projection uses flat-sky, unshifted Limber wave numbers
k=ell/chi. The deterministic NLA baseline and residual TATT source kernels
retain their separately defined normalization and interpolation sequences.
`power_mode` and `ct_mode` select explicit power and velocity-shear
conventions. A continuous algebraic identity is insufficient to equate two
different finite interpolation/projection operators.

For ordered unequal source bins, swapping ell1 and ell2 alone need not
leave the bispectrum unchanged. `ordered_legs=True` therefore calculates
both signs of angular mode M; the historical transpose shortcut is an
explicit comparison option. The Legendre definition uses
ell3^2=ell1^2+ell2^2-2 ell1 ell2 mu. Preserve its sign together with the
coupling factor (-1)^L. Independent Cartesian Gaussian-moment tests check
all four components and an asymmetric ordered example. The TATT centroid
rotation passes those tests; the upstream main conjugate rotation remains
a named comparison convention, not an interchangeable choice.

Radial FFTLog averages refer to lower bin edges and a declared logarithmic
width. Finite aperture weights use the matching representative side lengths.
A positive transform `dphi` averages X-projected angular modes before the
centroid rotation at the bin center. That is not the exact average of the
varying centroid rotation. The historical untapered FFT and corrected
coefficient taper are explicit choices. On coarse untapered grids, Nyquist
handling can distinguish historical and general signed-mode finite
operators even for a symmetric input; do not erase that difference by
silently changing a window or claiming a universal transpose identity.

The analytic covariance describes the specified gravitational full-filter
map estimator, with circular/square center windows or the declared
large-area approximation. It returns Gaussian pairings, BB, power times
one-halo trispectrum, power times bispectrum, connected one-halo, and the
selected halo-abundance response. The last family is not every SSC
contribution: growth, dilation, tidal and other missing responses remain
separate physical work. Nonlinear-IA connected covariance, catalogue masks,
distinct-galaxy exclusions and finite triangle selection are not supplied
by this map calculation.

The halo prescription is one pressureless matter fluid, using linear
P(k,0)D(z)^2 and total mean density. It does not implement the ordinary
core's cold-plus-baryon halo-variance treatment for massive neutrinos.
The project refuses a nonzero mass unless the user explicitly selects the
single-fluid effective-growth approximation. Do not describe that choice
as a validated massive-neutrino halo model. The analytic covariance also
requires zero fiducial multiplicative shear calibration; a Fisher run can
vary calibration in its mean while holding that covariance fixed.

Xi white noise needs its angular-bin definition. For `noise_mode=1`, the
large-area uniform-annulus route replaces only the pure NN contribution
with the analytic continuous-bin overlap, using actual lower/upper edges.
The signal and mixed signal-noise terms retain their fixed quadrature.
`noise_mode=0` is the explicit finite-band reference. Custom pair weights
or finite center windows cannot borrow the annular correction without a
matching derivation. Never infer true bin boundaries from arbitrary
quadrature nodes or interpret an unresolved high-ell noise sum as a
continuum covariance.

## Ownership, cache keys and the Cocoa driver

A `PhysicsModel` owns immutable copies of its physical arrays. A transform
plan owns fixed geometry and worker capacity. A new physical input or
changed geometry requires the corresponding new object. For any cache,
the allocation key must cover all sizes, and the refill key must cover
every value used. Exact parameter keys preserve small Fisher displacements;
rounding a cosmology before comparison changes the derivative.

The ordinary two-point module has one mutable calculation per process.
`CoreTwoPoint` construction stores its choices without changing that state.
Every `spectra` call performs `initial_setup()` and reinstalls its declared
numerical resolution, worker count, nonlinear runmode, TATT selectors,
source/lens counts, photo-z convention, zero internal nuisance shifts and
inactive baryon contamination before installing the shared physical arrays.
The reset is required even when the previous caller used the same adapter:
a notebook cell may have changed a selector or left another model only
partly configured. The existing API cannot read back all settings, and its
common accuracy control retains hidden baseline node counts. Relying on a
Python ownership flag or a manual reclaim call would miss those edits.

The adapter already calls `init_ntomo_powerspectra()` on each request; that
function changes `Ntable.random` unconditionally. The reset therefore does
not introduce an additional numerical table evaluation before the requested
spectra, although it reconstructs the input splines. No cost or speed claim
follows from this correctness choice. The independent 3pcf objects retain
their own immutable state. Calls into the same ordinary module must remain
sequential: neither another adapter nor a raw notebook call may change its
state between restoration and evaluation. This is a process-state ownership
contract, not a promise of simultaneous Python-call safety.

`tests/test_core_two_point_sequences.py` exercises the actual existing DES Y3
extension against separate clean interpreters. It retains old adapters
through two-/three-bin changes, different declared numerical resolutions,
raw selector and nuisance changes, partial source/accuracy configurations,
and loading then removing a baryon suppression table. Require bitwise E/B
agreement after restoration and demonstrate that the raw IA/baryon edits
changed a spectrum before restoration; merely invoking a setter does not
prove that a test reached the affected calculation. These small checks test
state restoration, not survey convergence or physical feedback validation.

The covariance instead shares preparation within one offline call. It
allocates Bessel/filter/window geometry before mass/redshift loops, then
reuses it across the requested terms and row pairs. For the selected
lensing model a halo profile separates into source efficiency times a
radius-dependent shape. Transform each distinct aperture radius and needed
unordered radius pair once, retaining source factors explicitly. Identical
xi bins or aperture radius multisets may share geometric responses only
under exact equality of their defining inputs. The halo offset profile
uses natural-cubic preparation and dense uniform transformed-radius reads;
its coarse radial grid must resolve inner structure as well as outer
oscillations. Releasing these arrays when the offline call ends is valid;
a persistent global cache is not required without subsequent reuse.

The companion project is `Cocoa/projects/des_3pcf/`, outside a standalone
core checkout. Its main files are:

| Project path | Responsibility |
|---|---|
| `interface/` | Isolated build and registration of the core adapters |
| `python/des3pcf/inputs.py`, `survey.py` | Physical NPZ/CAMB tables, actual source curves, row identities and bin edges |
| `python/des3pcf/mean.py`, `pipeline.py` | Geometry plans, bounded projection batches, exact cosmology/source/IA keys and final calibration factors |
| `python/des3pcf/core_two_point.py` | Existing CFASTPT TATT E/B spectra with explicit normalization/pivot and growth conversion |
| `python/des3pcf/covariance.py` | Fixed quadrature and physical covariance input arrays |
| `python/des3pcf/forecast.py`, `precision.py` | Full-vector finite differences, fixed covariance, Gaussian priors, marginalized FoM and optional precision-matrix expansion |
| `scripts/run_3pcf.py`, `configs/` | Versioned prepare/evaluate/covariance/fisher commands and separate Y3/Y6 examples |

Fisher output retains every declared xi and aperture row and their cross
covariance; no compression is implicit. Derivatives at h and h/2 diagnose
step sensitivity. One fiducial covariance is held fixed for all displaced
means. Optional precision-matrix expansion uses the full same joint sample
covariance and an independently justified realization count. It does not
turn correlated patches into independent Wishart samples. Missing survey
area, noise, priors or Y6 source files must remain missing, rather than being
filled with undocumented choices. Y3 simulated public inputs and example
settings alone do not establish publication reproduction.

## Read the study by task

The table gives entry points; chapter references inside each file retain
the longer chain of derivation. Intrinsic alignment means a correlation of
galaxy shapes with their environment. TATT denotes tidal alignment and
tidal torquing; the velocity-shear extension is a distinct physical term.
Read the convention checks before combining coefficients across codes.

| Task | Study chapters |
|---|---|
| Mean physics, interpolation and integral reductions | [12: physical identities](threepcf_study/12_second_pass_physics.md), [13: numerical boundaries](threepcf_study/13_second_pass_numerics.md), [20: full-filter reductions](threepcf_study/20_further_data_vector_reductions.md), [21: finite-estimator adjoint](threepcf_study/21_further_execution_and_cache_design.md) |
| Covariance physics and simulation corrections | [15: CPU covariance port](threepcf_study/15_analytic_covariance_cpu_port.md), [17: core mechanisms](threepcf_study/17_cosmolike_covariance_cpu_transfer.md), [18: precision matrix expansion](threepcf_study/18_precision_matrix_expansion.md), [19: integral reductions](threepcf_study/19_further_covariance_reductions.md), [22: stress tests](threepcf_study/22_stress_test_of_reductions.md) |
| Ownership, persistent arrays, OpenMP and SIMD | [8: allocation/refill keys](threepcf_study/08_static_cache_design.md), [14: core integration](threepcf_study/14_second_pass_integration.md), [23: execution and memory](threepcf_study/23_unified_execution_and_memory_plan.md), [24: priorities and acceptance](threepcf_study/24_priorities_and_acceptance.md) |
| Upstream versions, conventions and credit | [9: main/TATT reconciliation](threepcf_study/09_main_branch_reconciliation.md), [10: Y6 workflow](threepcf_study/10_desy6_workflow.md), [11: pinned sources and notices](threepcf_study/11_sources_and_credits.md) |
| Survey reproduction and Fisher interpretation | [1: scientific targets](threepcf_study/01_physics_and_reproduction.md), [5: joint forecasts](threepcf_study/05_covariance_fisher_des_y6.md), [6: acceptance sequence](threepcf_study/06_work_plan_and_acceptance.md), [16: Fisher execution](threepcf_study/16_fisher_execution_checks.md) |

## Verification and remaining scientific work

Preserve the independent checks of aperture kernels, defining coupling
integrals, FFTLog against NumPy/SciPy, Gaussian moment phases, halo profiles,
halo abundance derivatives, xi E/B kernels and covariance partitions.
Check full vectors and parameter derivatives after changing interpolation,
finite bounds or a contraction order. Thread comparisons must vary the
worker count without changing model settings. A fixed-operator regression,
a numerical refinement and an independent physical derivation answer
three different questions; retain all that are relevant to a change.

Record the exact build and executed tests in the project's
`tests/VALIDATION.md`. Early foundation-only test counts are not evidence
for subsequently added physical modules. The implementation still needs
survey-scale integration and model comparisons, matched catalogue
estimators where required, separate Y3/Y6 inputs and priors, and validation
of absolute Fisher information. CPU timing and scaling must be measured
separately on an available machine, separating preparation, repeated means
and offline covariance. No CUDA dependency or speedup follows from an
operation count or from choosing C/OpenMP/SIMD.

## Maintenance rules

Keep this page synchronized with implemented entry points and recorded
verification. Preserve the archived distinction between inspected code,
derived identities, approximations and proposed tests. Update a scientific
claim when a new result changes it; retain the pinned upstream revision
needed to understand the original source observation.

The [development skill](../SKILL.md) and
[explanation guidance](../../explaining-code/SKILL.md) apply to code and
documentation. Define physical quantities, dimensions, approximations and
array ownership before describing optimizations. Maintain the prohibited
vocabulary and plain-language rules used by the core. In a Cocoa checkout,
its maintenance skill also governs shared project and installation changes.
Preserve applicable source notices; citations do not replace licenses.
The archive's GPL covariance source is a comparison reference, not an
authorization to transplant its code into the MIT-licensed core.
