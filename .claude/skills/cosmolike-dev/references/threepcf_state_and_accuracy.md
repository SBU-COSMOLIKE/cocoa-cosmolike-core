# Three-point state, resolution and notebook use

Read this when changing a 3pcf table, constructor, Python prediction cache,
accuracy control, or notebook wrapper. The general
[state-invalidation rules](state_invalidation.md) still apply. The distinction
is that the isolated 3pcf C module owns prepared arrays through model/plan
objects, while the ordinary data vector has mutable global tables.

## Map the ordinary-core rules to the isolated objects

An allocation key identifies the dimensions for which storage was allocated.
A refill key identifies the input values used to calculate its contents.
Increasing the dimensions without reallocating can overrun storage; changing
only values without refilling can silently return an old calculation.
Neither failure is tested by constructing every case in a fresh process.

| State | Allocation dimensions | Values defining contents | Reset/lifetime |
|---|---|---|---|
| Physical model | k/z/LOS/kernel counts, source count, coarse/dense fit counts | All physical axes and arrays, cosmology, source-reader convention | Constructor copies input; destructor releases owned arrays |
| Supplied B baseline | Four coarse/dense axis counts and background Omega_m(z) count | z/logk/logu/v axes, ln B, boost/clamp, supplied Omega_m(z), power normalization | Separate immutable owner; constructor copies and destructor releases arrays |
| Projection grid | Triangle/multipole axes, L cap, chart choice and dense ratios | Model, ordered bins, every IA/baryon selector, bounds and coordinates | New owner for a changed definition; a kept old grid stays valid |
| Transform plan | FFT size/padding, radial selection, L/M caps, scratch capacity | ell/phi/selection values, biases, window, phase/projection convention | Constructor prepares serial FFTW plans; destructor releases plans/storage |
| Standalone double-Bessel plan | Physical and padded grid lengths, maximum Bessel order; one workspace slice | Copied ell, both biases, xy, bin width, padding and taper; latest complete forward input | Explicit owner retains FFTW/Mellin preparation; core reset leaves live owners valid |
| Multipole query map | Requested ell count, projected-grid axis counts and L cap | Exact copied log-ell, physical log-psi and internal PSI-coordinate axes; psi_scale and reference-mode selector; requested ell fixed by plan | Plan owns the map; recreate when the complete axis key differs |
| Projection batch | LOS count, eight terms, number of triplets | Copied radial weights from projectors sharing one immutable model and exactly equal IA parameters | Call-local owner, destroyed before borrowed first projector/model |
| Aperture weights | Side/opening-angle/filter counts | Bin widths, coordinates and ordered filter radii | Caller owns the returned numerical weights |
| Physical covariance | Source/field/row and all quadrature dimensions | All caller arrays, windows/noise, halo prescription and estimator choices | Call-local preparation, released on return |
| Python physical provider | Depends on the new C model | Exact cosmology, photo-z shifts, producer prescription and numerical controls | Replaces latest model; previously returned models remain owned and usable |
| Python prediction/derivative cache | Full row count of the selection | Model identity, complete IA dictionary, and numerical state token | Changed input invalidates cached results before lookup |

Do not add a shared `static` merely to imitate its spelling in `cosmo2D.c`.
Choose persistence where a later call actually reuses the work. If adding a
mutable static, implement the complete size and value comparisons and its
reset in the same change. A model-owned immutable table has those values
fixed for its lifetime, so construction/destruction performs the corresponding
allocation/refill/reset work. The C++ owner must preserve dependencies: a grid
that borrows a physical model cannot outlive that model's storage.

For the standalone FFTLog readiness flag, copied adapters and lifecycle
checks, read [FFT plan reuse](threepcf_fft_plan_reuse.md).

Production arrays are borrowed only for the duration stated by the C/C++
entry point. The notebook adapter copies through Armadillo and converts
layout explicitly. A returned diagnostic axis or output must be an independent
copy; editing it cannot change the table C keeps. In the Cocoa Python
project, survey rows and prepared engine geometry are independent immutable
copies. Construct a new engine in the same interpreter to change its geometry.
Mutating one public array while retaining old weights is refused with an
explicit diagnostic.

The physical model's optional `source_z[nsource]` is independent of its
background `z[nz]` and power rows. C copies the source axis and densities,
then prepares chi and dz/dchi on that axis before parallel readers run.
Omission, or an explicit byte-identical background axis, retains the old
radial preparation arithmetic. An independent axis must remain within
background support. Keep tests of its analytic lensing measure, owned
inputs/outputs, both adapters, and source-count A/B/A restoration.

In the project, `source_nz` and `source_boost` refine an explicitly chosen
source sampling grid while retaining the provider's cosmology tables.
Every photo-z shift is reconstructed from immutable native samples;
successive notebook calls must never shift the previous shifted result.
Reconstruction and endpoint policy are fixed physical-model choices for
one provider. Native samples refer to the chosen FITS, which may already
be an explicitly processed demonstration file. Preserve its hash and
processing record. A continuous interior cubic need not join a zero
extension continuously in its derivative, so endpoint and derivative
checks remain necessary even when every sampled density is nonnegative.


## Supplied baseline and bridge replacement

The physical input contract is one uniform log-k segment and at most ten
uniform pieces in background/source redshift, with spacing tolerance 1e-9.
Keep the DES producer's `np.linspace` construction consistent with the other
Cocoa projects. Do not restore arbitrary irregular-k input or merge fine
low-z and coarse high-z segments into one grid. Production shares the
natural-cubic solver and direct piece lookup; the explicit source comparison
retains its upstream interpolation. The project test
`data_vector/test_piecewise_physics.py` checks joins, refinement and matching
Python/C BiHalofit preparation. The numerical guide explains its boundary
conditions; changing only C leaves the Python producer inconsistent.

### Copied shared variance inputs

`core.variance_snapshot` reads the installed total-matter linear power and
returns owned z/rnl/neff/sigma8 arrays plus optional halo mass/variance/slope
arrays. Its serial entry retains the Python GIL while shared lazy tables
are prepared. It installs no spectrum and changes no physical convention.
In particular, evolving P(k,z) and P(k,0)D(z)^2 are different inputs with
massive neutrinos. The halo snapshot's `reference_rho` preserves the
consumer's mass-to-radius convention; the shared query mass is multiplied
by core_rho/reference_rho. The returned slope is d ln sigma/d ln M, so the
covariance multiplies it by 2 sigma² where d sigma²/d ln M is required.

The optional model `variance` dictionary contains exactly z/rnl/neff/sigma8.
Its presence also disables the local rescaling of linear P to the separate
sigma8 scalar. Supplied fit anchors must cover the model's full redshift
domain. Covariance snapshots require exact requested mass/redshift axes and
the prescribed density. Their caller must establish that they belong to
the same physical power state; matching dimensions alone is insufficient.

`core.set_variance` copies its arrays and stamps both `cosmology.random`
and `Ntable.random`. A later bridge read refuses stale supplied variances
before considering a model cache hit. Resupply or explicitly clear after
power, cosmology or accuracy changes. The anchor count belongs to the
allocation key. Retained immutable model owners survive replacement/reset.
Both ordinary and standalone `initial_setup` call `reset_variance_tables`,
which frees both shared FFTLog workspaces and their plans serially.

These inputs are an optional integration step, not an accepted replacement
of the local variance prescription. The shared FFTLog endpoint correction
passes five power-law slopes, curved spectra and smaller derivative steps;
ordinary turnover and DES fiducial snapshots retain their exact values.
The nearly saturated low-tail stress case requires Gaussian-table
refinement. These component checks do not replace full joint mean,
derivative and Fisher acceptance before selecting shared variances or
removing the local path. Read
[`dev_scripts/threepcf_variance_check/VALIDATION.md`](../../../../dev_scripts/threepcf_variance_check/VALIDATION.md)
for the measured domain, state checks and outstanding scientific tests.

### Supplied baseline state

`physics_input_3pcf.baseline == NULL` selects the ordinary internal fit.
An active baseline owns a positive absolute B table and requires the
background `omega_m_z` array. Replacing the physical model replaces all
these arrays; a prepared projection grid owns its projected values and
remains valid after the source model is released. The baseline's four
uniform axes and `boost` define dense allocation sizes. `clamp` is part
of the model definition, not a numerical fallback after a refused query.

`core.set_baseline_bispectrum` copies input before releasing old storage.
This order also permits a native caller to resupply the bridge's current
input view. The bridge compares every table/axis byte, including signed
zero, every dimension, `boost`, `clamp`, `omega_m_z` and `sigma8`. A
changed input increments its change counter, which joins the core's
cosmology/source counters in the model-reuse decision. The allocation key
also includes all four baseline axis counts and the background count.
Clearing an absent override changes nothing; clearing an active override
returns to the independently stored internal-fit scalars. Core reset
clears the override but cannot invalidate existing immutable model owners.

The bridge's model key excludes `nuisance.random_ia`, IA prescription
selectors and feedback amplitude. These values enter the freshly copied
`ia_parameters_from_core()` dictionary, not `physics_input_3pcf`. Changing
them must preserve both the model owner and its build generation while
changing the relevant projected result. Never key a projected-result cache
on model identity alone: include the complete query dictionary, as the
project's `JointPrediction` does. Old dictionaries and projection grids
retain their own values. The model already prepares all three power modes.

The source, LOS, pixel-window and feedback setters copy replacement arrays
before freeing their old storage. A native caller may resubmit a borrowed
bridge input, including a shorter valid LOS/window prefix or an unchanged
source axis alongside new densities. Such a call must not read freed input.
`tests/test_bridge_cache.py` in the project checks these public C calls
through a compiler-checked helper, alongside IA/amplitude A/B/A reuse and
continued physical-input invalidation. No hand-written Python struct layout
is part of that test.

In the companion project, `physics.bispectrum_source` chooses `internal` or
`supplied`. `accuracy_boost * bispectrum_boost` refines the four coarse
interval counts from immutable original axes; the configured dense `boost`
remains separate. Reapplying accuracy controls must not refine an already
refined grid. The producer caches its physical-array/control identity
alongside the immutable result in one tuple, so concurrent Python readers
cannot associate one input key with another input's table.

Keep A/B/A, repeated-identical-input, clear/reset, resized-array and retained
owner checks. Baseline replacement must leave tree-level P diagnostics and
raw IA unchanged when their inputs are unchanged. For covariance routing,
holding P and the background fixed while multiplying B by two multiplies
BB by four and PB by two; the other covariance families retain their own
physical inputs. These identities test dispatch, not a calibration of a
supplied B model. Refine its coarse axes separately from dense `boost`,
then assess the final mean using the same fixed survey covariance.


## Signed amplitudes and angular chart controls

The triangle and multipole arrays distinguish an interpolation's
coordinate from the physical quantity stored at a node. The optional
controls below are part of the immutable projection-grid definition.
Their default values preserve the preceding numerical representation;
choosing a new value needs a new grid. No new value is a claim of survey
accuracy or a substitute for a resolution study.

The explicit shape-endpoint branches can change last bits relative to an
older executable's tanh/inverse-tanh round trip. Comparing an omitted
`shape_stretch` with an explicit value of three in the current executable
checks default selection, not historical binary identity.

| Control | Default | What changes | What does not follow from it |
|---|---:|---|---|
| `signed_baseline` | 0 | With 1, interpolate log positive GGG and signed NLA/GGG separately | NLA remains the same physical effective-kernel projection |
| `residual_scaling` | 0 | With 1, interpolate log GGG at equal sides of the same perimeter and signed residual/GGG separately | No absolute value, sign clipping or change of IA coefficients is allowed |
| `shape_stretch` | 3 | Redistribute fixed PUV shape nodes toward their endpoints | Stated extreme support is not proof of resolved squeezed triangles |
| `psi_scale` | 0 | With a positive value, use uniform log(psi)+psi/psi_scale for production b_L tables | The exported physical log_psi axis need not be uniform |

The residual scale removes a broad radial amplitude before cubic
preparation. It uses the same physical model, source triplet and matter
prescription with zero IA amplitudes. Its logarithm and the signed ratio
have independent owned arrays with a common perimeter axis. Coarse values
are complete before their separate cubic preparations start. Pure NLA
prepares neither array. The default source comparison keeps raw residuals;
explicitly enabling scaling in reference mode is a different interpolation
choice and must be named as such.

For PUV shape, increasing `shape_stretch` resolves smaller edge fractions
but coarsens the middle of the interval at fixed `nshape`. Refining the
number of dense samples alone cannot supply the missing physical samples.
At fractions close to machine precision, constructing a short closing side
and then recovering a fraction can itself lose relative accuracy; the inverse
shape mapping amplifies that loss. Keep interior knot checks distinct from
extreme side-geometry tests and from near-boundary interpolation convergence.
Do not turn a failed extreme test into a loose tolerance for ordinary shapes.

For multipole angle, the mixed coordinate is monotone, so its inverse has a
unique root within the stated physical endpoints. Root finding occurs only
while preparing physical node arrays. Cubic upsampling uses the uniform
internal coordinate; the repeated reader uses arithmetic indices and linear
fractions in that same coordinate. A fixed iteration count during axis
preparation is not adaptive physical integration. Reference mode preserves
its original irregular log-PSI axis and reader. No extra integration Jacobian
belongs in b_L merely because the interpolation coordinate changed.

### The geometric query key must name the coordinate

The map the plan keeps checks grid `nell`, `npsi`, `lmax`, `reference_mode`,
`psi_scale`, and exact copied values of `log_ell`, physical `log_psi`, and
internal `psi_coordinate`. Its requested Fourier array is owned and fixed
for the map's lifetime. The two ordered source-leg charts remain distinct.
With only two angular nodes and no dense upsampling, zero and positive
`psi_scale` share the same physical endpoints and sampled table values,
but their interior interpolation weights differ. This gives a direct test
of a key that checks physical endpoints or dimensions without the chart.

Changing cosmology, source curves, IA amplitudes, `signed_baseline`,
`residual_scaling`, or `shape_stretch` can change b_L values while leaving
the query axes unchanged. Those values must be read from the current grid;
they are not stored in the geometric map. Reusing its cell indices is then
correct. A blanket rule that all physics settings invalidate every geometry
object would confuse its role with the projection table's owned values.

### Separate the three numerical questions

1. Compare projected B with direct physical line-of-sight values, first at
   interior physical knots and then between them. Isolate radial range from
   shape-edge resolution; an off-node perimeter at an original shape knot
   provides a test of the radial interpolation alone. Retain both signs and
   actual residual crossings. Use an explicit positive error scale there.
2. Hold projected B and its angular nodes fixed. Integrate each linear B(mu)
   segment against P_L with an independent exact polynomial quadrature.
   Compare b_L at off-node scale/angle pairs and both source-leg orders,
   changing only its interpolation controls. Check the projected samples
   remain bitwise identical. Scale small coefficients by an angular norm,
   rather than divide by a coefficient close to zero.
3. Refine the transform, aperture estimator and parameter derivatives on the
   chosen survey rows and covariance only after the earlier stages are
   understood. Source reconstruction and photo-z shifts are another numerical
   stage: changing angular nodes cannot remove a kink introduced while
   shifting a piecewise-linear source curve at its own knots.

For every control, combine the independent-value comparison with unchanged
legacy/reference behavior and setting sequences through both adapters. Use
grids kept and rebuilt, a transform plan kept across calls, A/B/A
restoration, and worker counts within the stated resource cap. A test of cold-state
reconstruction alone does not exercise reuse of a geometric map. Record
which checks actually ran in the project validation record; source review
and these proposed comparisons are not execution evidence.

## Setting sequences, with independent references

BLAS is always serial: set `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1` and `BLIS_NUM_THREADS=1` before Python imports.
Parallel cores belong to OpenMP. On the shared development laptop, run one
test process at a time and cap OpenMP at four workers with
`OMP_THREAD_LIMIT=4`; ordinary cases request one. Child processes inherit
these settings. FFTW uses serial plans executed by outer OpenMP workers,
so it must not start another thread team inside those loops.

The tests live in `Cocoa/projects/des_3pcf/tests/`. They follow the ordinary
data-vector setting-sequence design, using their own workers because these
objects do not use the ordinary global initialization API:

- `data_vector/test_physics_setting_sequences.py`: physical sizes, values,
  reader choices, IA conventions, source ordering and projected grid settings.
- `data_vector/test_transform_setting_sequences.py`: FFT geometry, angular
  modes, chosen outputs, aperture weights, caller/output mutations and workers.
- `data_vector/test_grid_transform_reuse.py`: kept query geometry versus
  refilled physical values, exact interior-axis keys, both adapters and a
  ten-node Fourier axis exercising the two-cell SIMD tail.
- `data_vector/test_transform_mode_schedule.py`: small signed-mode sets,
  both phase conventions and worker counts with changed/restored multipoles.
- `covariance/test_setting_sequences.py`: every allocation axis, individual
  coordinate/weight/model changes, estimator/noise switches and old models.
- `test_pipeline_setting_sequences.py`: provider keys, IA/photo-z/calibration,
  global/stage boosts, returned ownership and the outer Fisher mean cache.

For each compared step, a long-lived process first evaluates other settings,
applies the requested changes, then evaluates again. Its reference is a
separate interpreter configured directly at the final settings. Compare
float64 bit patterns, including signed zero, rather than a tolerance that
could conceal out-of-date values. Same-process reconstruction alone cannot
expose a shared out-of-date table. Exercise both adapters, increasing and
decreasing sizes, same-size value changes, A/B/A object interleaving,
destruction/recreation,
strided/read-only notebook inputs, and workers changed after plan creation.

A reach assertion must establish that a changed input enters the calculation.
For example, large-area xi zero modes bypass the radius-profile interpolation;
varying profile density there does not test its refill rule. Use finite-window
xi rows for that test. An IA toggle must reach a nonzero IA component; pure NLA
must return exact zero residuals when expected. A larger L/M cap needs input
with power in the added modes. Objects kept from earlier calls must still
reproduce their old outputs after a newer object is evaluated.

## Global and individual controls

`python/des3pcf/accuracy.py` in the Cocoa project resolves counts through an
immutable `Accuracy` object. Defaults are multiplicative one, including
`integration_accuracy_boost`. This differs from the ordinary core's integer
integration-level selector; do not pass one to the other as if they had the
same meaning. The C constructors continue receiving explicit resolved counts
and arrays, so the two C++ adapters share one numerical implementation.

Sampling counts use `accuracy_boost`; fixed quadrature counts use
`integration_accuracy_boost`. Counts that keep both endpoints refine their
N-1 intervals. LOS nodes use a dyadic subdivision of each original redshift
interval, with factor `accuracy_boost * integration_accuracy_boost * chi_boost`.
Old nodes remain exactly present; the C model reevaluates chi and trapezoidal
distance weights. Bounds remain fixed and a subdivision above 1024 is refused.
All controls resolve from unboosted baselines, never from their previous result.

Individual controls separate kernel sampling, fit-z anchors, physical dense
readers, projected triangle/multipole grids, angular quadrature, FFT density,
xi/aperture quadratures, covariance ell/window/angle/radius/k/mass rules and
coarse/dense halo profiles. The physical model's reader ratio also affects its
dense fit table. A global sampling increase can refine both that ratio and the
fit anchors. One-stage tests distinguish these errors before a global run.

Input Boltzmann k/z samples, external source measurements, integration bounds,
L/M truncations, FFT padding/window choices and survey bins remain separate
explicit inputs. Fine interpolation cannot repair an unresolved input table.
Measured pair weights cannot be silently resampled when a theta count changes.
The project refuses that combination unless weights are supplied at the new
baseline count; the uniform-area approximation can generate its own new weights.

`PhysicalProvider` includes controls in its exact model key. `JointPrediction`
provides a `cache_token` for those mutable controls, and `ExactMeanCache` checks
it before a parameter lookup. Otherwise a notebook can change chi sampling
but receive an old Fisher prediction for the same cosmology. Generic custom
callables without such a token must have fixed hidden inputs or explicitly
clear their cache. Owned returned tables and cached arrays must not allow
writeability to be restored: immutable backing storage protects that boundary.

## Numerical revisions that require independent checks

Squeezed triangles require stable triangle algebra. Subtracting squared side
lengths can discard a physically relevant small term even when all inputs and
outputs are finite. For sides (epsilon,1,1), the F2 pair on the short and long
leg tends to 13/28, not 5/7. Use normalized factored differences with a stated
valid-triangle domain. Check orientations and cyclic bispectrum sums with
high-precision arithmetic; agreement with an upstream implementation that
uses the same unstable subtraction is insufficient evidence.

SIMD lanes and scalar tails must preserve the same arithmetic grouping under
strict floating-point compilation. `w*(a*b)` and `(w*a)*b` can differ in rounding,
underflow or overflow. An odd/even padding comparison can isolate a tail
without changing the mathematical sum. Dimension products must enter checked
size_t arithmetic before multiplication. A guard after an overflowing int
product does not protect an allocation.

Strict flags do not universally disable scalar fused multiply-add. A compiler
can contract a tail's multiplication and addition while explicit SIMD calls
retain two rounded operations. The covariance tails use local `volatile`
temporaries at the same rounding boundaries as their vector lanes. The public
checks prepend a zero-measure radius or add zero-weight angular/power samples:
the physical sum stays fixed while the same output moves between a vector
lane and a scalar tail. Preserve these tests and local boundaries instead of
changing unrelated compiler flags.

The Legendre preparation and bilinear multipole reader need the same
boundary. `data_vector/test_multipole_lane_invariance.py` adds an unused
query or an
extra kept order, then requires the existing coefficients to keep
their float64 bits. The test holds the physical grid and weights fixed;
changing a quadrature resolution would not justify that exact comparison.

The physical projection has the same requirement. Its paired source triplets,
Legendre orders, NLA power reads and aperture filters must agree with their
unpaired tails. `data_vector/test_physics_lane_invariance.py` moves existing
outputs between
those positions with the physical nodes and weights held fixed. A mixed pair
with one extrapolated coordinate must preserve the valid coordinate's prepared
linear interpolation. For a final unpaired NLA coordinate, duplicate its
prepared cell and fraction into both SIMD lanes and retain the first result;
this reuses both the interpolation and accumulation operations. These tests
exercise both adapters and both power-reader conventions.

Skip a prepared covariance table only after tracing every consumer. In the
large-area branch, quadratic halo responses use their direct Fourier dot
product; displaced shear profiles and finite-window response transforms have
no reader. The cubic halo response and PT term still need radial profiles.
The real-space bispectrum table is used only by finite-window BB. Preserve
those distinctions when reducing work, and compare all six terms separately
across both window branches, mixed rows and quadratic-only rows.

Read [NUMERICS.md](../../../../cosmolike/3pcf/NUMERICS.md) for the production
cubic-preparation/linear-read strategy and
[COVARIANCE.md](../../../../cosmolike/3pcf/COVARIANCE.md) for the physical scope.
Record executed strict/debug checks and their limits in the project's
`tests/VALIDATION.md`; source-level reasoning is not a replacement for them.

## 2026-11 state-audit record (moved from COVARIANCE.md)

During this audit, correcting parentheses alone did not pass the
radial-padding check: the optimized ARM64 object still contained a scalar
`fmadd` where the vector path had two multiplications followed by
addition. The local rounding boundary in the covariance scalar tails
addresses that measured compiler behavior.

The covariance setting-sequence tests in
`Cocoa/projects/des_3pcf/tests/covariance/test_setting_sequences.py`
passed eight parametrized tests through both adapters, using 41 final
states and 82 independent fresh-process references; each reference
process constructed only its requested model and made one covariance
call. The 36 same-size changes covered each integration axis and weight
array, source/field/row labels, aperture radii, shape noise, exact xi
edges, the finite/large-area switch, both NN modes, profile upsampling,
every exposed ST/Duffy scalar, the evolving collapse threshold, the
BiHalofit baryon parameter, survey area and window shape. All tested
transitions matched their fresh references bitwise. The audit found no
native covariance state defect requiring a code change.

### Sequential calls into the core bridge

The core bridge uses process-global mutable input, scratch arrays and a
cached model owner. Calls to `core.set_*`, reset, native refresh and
`core.model_from_core()` must complete sequentially within the process.
The model factory releases the Python GIL during preparation; it does not
protect these shared inputs against another Python thread. Internal OpenMP
and retained immutable owners do not establish concurrent bridge-entry
safety. Cache tests cover sequences of completed calls, not overlapping
setters and model creation.

Keep comparison contracts distinct: the supplied-baseline setter compares
array bytes, including signed zero. Older setters using `bridge_differs`
compare exact numerical values without a tolerance, for which signed zeros
are equal. Do not describe that older helper as a byte comparison.
