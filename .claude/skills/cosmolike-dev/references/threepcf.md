# Three-point predictions and covariance

The numerical port lives in `cosmolike/3pcf/`. It predicts the physical
three-point mean and a specified joint two-/three-point covariance. The
companion Cocoa project owns survey input, optional CAMB preparation and
Fisher orchestration. Shared grid indexing and variance readers live in the
common core; the two-point physical prescriptions retain their definitions.
This is an implemented numerical model, not a demonstrated reproduction of DES Y3 or
Y6 or a measured performance result. Read the estimator and physical limits
before combining its outputs.

For cache changes, accuracy controls or notebook reuse, read
[state and resolution](threepcf_state_and_accuracy.md). It maps the ordinary
core's allocation/refill/reset rules onto the isolated owners and identifies
the fresh-process setting-sequence tests, including the outer Fisher cache.
The state reference also defines `residual_scaling`, `shape_stretch`, and
`psi_scale`: separate controls for signed radial amplitudes, physical triangle
fractions, and multipole angular sampling. Read their diagnostic separation
before changing table bounds or interpreting an aperture/Fisher result. The
[core numerical guide](../../../../cosmolike/3pcf/NUMERICS.md) explains why the
physical angle exported to users differs from its uniform interpolation
coordinate and which coordinate belongs in the retained-query key.

For plotting, original-code comparisons and YAML-driven forecasts, read
[notebook workflows](threepcf_notebooks.md). Shared plotters belong in
`cosmolike_notebook_utils/3pcf/`; physical and survey choices stay with the
core calculation and companion project, respectively.

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
it selects the internal BiHalofit calculation or an explicitly supplied
full matter bispectrum, calculates the selected TATT/velocity-shear terms, then
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

The joint bridge also converts the fastnc alignment amplitudes to the
ordinary core's fixed normalization and redshift pivot. It multiplies each
amplitude by `(c1_rho/0.01389)*(1.62/(1+z_pivot))**eta`, preserving the
physical C1 and C2. The ordinary core stores -C1 and C2/5 and applies the
remaining sign/factors in its correlators; do not compare those intermediate
names directly with the three-point coefficients. The tensor-basis density
coefficient is C1delta-2*C2/3, while the physical bta remains C1delta/C1.

Matching supplied growth nodes does not match intermediate interpolation:
the ordinary core linearly reads G=(1+z)D, whereas three-point preparation
uses a cubic interpolant of D followed by dense linear reads. Refine the
supplied growth axis separately from the three-point LOS nodes. Ordinary
CFASTPT IA loops use linear power at z=0 and a D^4 factor; the full linear
power table supplied by the bridge is its interface contract. This differs
from the deliberate nonlinear-power prescription of three-point mode2.
The project's `tests/studies/joint_ia_conventions.md` derives these mappings
and distinguishes source/growth quadrature from physical conventions.

The low-level nineteen-component IA API contains seven density/E terms
and twelve terms with one B leg. Only the first seven enter the projected
mean. That matches the paper's all-E numerical approximation and suffices
for ideal E-aperture moments. Finite-SAS truncation need not preserve exact
E/B separation. The project's `tests/studies/finite_window_eb.md` derives
why a reflection-odd single-B bispectrum and an odd response kernel can
give an even integrand; parity alone is not a zero test. No nonzero
correction for the particular weights has been measured. Require an
oriented-template transformation, ideal-aperture null, identical finite
weights and full-joint-covariance scoring before claiming that omission
is negligible. Two-B bispectra begin beyond the retained tree order;
two-point TATT BB power is a different order of correlation.

## Supplied absolute matter bispectrum

`baseline_3pcf.h` defines an optional absolute B table, distinct from the
paired hydro/DMO ratios in `feedback_3pcf.h`. Both owners share the existing
uniform natural-cubic preparation and multilinear logarithmic reader in
`feedback_3pcf.c`. Keep that shared interpolation machinery; do not create a
dummy power response or an amplitude parameter to represent absolute B.
For sorted sides a<=b<=c, its axes are z, ln(b/[h/Mpc]), ln(a/b), (c-b)/a;
flatten ln B in `[nz,nk,nu,nv]` order. B uses `(Mpc/h)^6`, with no coverH0
conversion. Zero or negative physical B cannot be repaired by absolute
values or clipping before taking its logarithm.

The additive `production.PhysicsModelWithBispectrum` and
`notebook.PhysicsModelWithBispectrum` factories require `baseline` and
`omega_m_z[nz]` on the physical background z axis. They retain `omega_m`
for lensing/IA and `sigma8` for the existing P normalization, and omit
`omega_de`, `w0`, `wa`, `ns`. The caller establishes flat geometry and a
consistent cosmology/matter definition across P, B, growth and distances.
The active route skips every internal-fit coefficient/radius preparation;
`physics_omega_m_z_3pcf` reads the supplied background for halo covariance.
Do not introduce a w0-wa reconstruction in that route.

`matter_prepared` selects supplied full B before any fit-coefficient read.
Public matter queries, projected NLA and covariance B terms follow that
selection. Component 1 remains tree level from linear P; components 2/3
are explicitly refused because a full B table does not specify the fit's
separate one-/three-halo pieces. Raw perturbative IA and signed residuals
retain their P prescription. Halo integrals in the covariance are separate
physical terms and never request diagnostic components 2/3.

The [module guide](../../../../cosmolike/3pcf/README.md#supplied-bispectrum)
and [project producer guide](https://github.com/SBU-COSMOLIKE/cocoa_des_3pcf/blob/main/BIHALOFIT_PYTHON.md)
explain the caller contract. `dev_scripts/supplied_baseline_check/` contains
default-route captures; the project `tests/test_supplied_baseline.py` checks
analytic values, ownership, bridge invalidation, refusals and independent
BB/PB scaling. Preserve the pre-edit binary and every reference input before
source changes. The existing full project suite and original stage-0
snapshot quantities remain separate gates; a small direct-B comparison is
not a replacement for final-observable convergence.

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

The retained standalone `production.TwoBesselPlan` and its copied notebook
adapters are documented in [FFT plan reuse](threepcf_fft_plan_reuse.md).
That record distinguishes preparation from refreshed input coefficients,
explains ownership and alignment, and points to private lifecycle tests.

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

The normalized NFW reader is `nfw_u` in `halo_profile.h`. It takes a
caller-owned f/G table, concentration and the dimensionless product
q=k*r200. It owns the exact zero limit and small-q expansion before the
general `nfw_um` table expression. The three-point adapter supplies its
radius and retains its independent 524288-node descriptor, prepared before
parallel reads. Keep mass definitions, density constants and concentration
laws in the callers. In particular, preserve each caller's radius
arithmetic when sharing the dimensionless reader; equivalent formulas can
round differently. This table is separate from the per-halo angular shear
spline and its `profile_upsample` setting.

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
Its public settings are immutable, so an outer mean cache cannot silently
retain a vector after a numerical-control edit. Construct a new adapter for
new controls; no kernel restart is needed. Its supported integration levels
are 0 through 7, the eight distinct shear quadrature orders from 96 to 8192
nodes. Larger raw-core values are not accepted by this adapter.
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


### Projected IA operator diagnostics

`physics_ia_3pcf` has immutable `operator_component` (0=total, 1=NLA,
2=density weighting, 3=torque, 4=VS) and `intrinsic_legs` (-1=all;
bit mask on ordered legs). Defaults preserve the ordinary prediction.
These fields are checked by batch compatibility. Never omit the -2 C2/3
piece when isolating torque, or the linear C1 factors on remaining I legs.
`field_kernel_physics` splits G/I on the original baseline kernel grid;
residual weights retain their different source-grid definition. Component
grids require signed_baseline=1 and reset both selectors on the shared
positive gravity denominator. Sum closure extends through interpolation,
FFTLog and aperture, and is tested in `test_ia_components.py`.

Project `ia_components.map3_components` returns [5,8,nrow], with gravity
separate from linear IA; the full total is evaluated independently. The
notebook is EXAMPLE_IA_COMPONENTS. The new shared array-only plotter is
`plot_ia_operator_placement`; the earlier `plot_ia_components` remains
unchanged. Read the core README's operator section before interpreting
Figures 5–6. The active VS vertex differs from literal manuscript Eqs.62/67;
retain this qualification until the equations and source convention agree.

## Paired feedback response ownership

`feedback_3pcf.c/.h` owns uniform log R_P and log R_B tables. Both are
required and independently supplied for total matter. The shared
`feedback_amplitude` uses exp(A log R), with fixed ratios across cosmology.
`physics_3pcf` applies it only to the nonlinear-alignment baseline; raw
building blocks and quadratic IA residuals remain DMO. Projection-grid
normalization also remains DMO so a changed amplitude cannot alter the
interpolated residual. Batch keys include the amplitude. Both C++ adapters
copy borrowed constructor arrays into the native owner.

Project `FeedbackPair` is immutable; `feedback_power_tables` changes only
nonlinear P for the ordinary two-point adapter. Linear P and CFASTPT loops
stay DMO. Response-file hashes enter saved-result identities. The controls
`feedback.boost` and `feedback_table_boost` refine cubic preparation followed
by linear hot reads. Explicit clamp/error domains describe extrapolation.
The project FEEDBACK.md and EXAMPLE_JOINT_FEEDBACK.ipynb teach the file
format and the absence of a hydro calibration in the example.
