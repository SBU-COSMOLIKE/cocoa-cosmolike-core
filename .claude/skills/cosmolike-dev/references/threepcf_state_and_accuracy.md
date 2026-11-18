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
| Projection grid | Triangle/multipole axes, L cap, chart choice and dense ratios | Model, ordered bins, every IA/baryon selector, bounds and coordinates | New owner for a changed definition; retained old grid stays valid |
| Transform plan | FFT size/padding, selected cells, L/M caps, scratch capacity | ell/phi/selection values, biases, window, phase/projection convention | Constructor prepares serial FFTW plans; destructor releases plans/storage |
| Aperture weights | Side/opening-angle/filter counts | Bin widths, coordinates and ordered filter radii | Caller owns the returned numerical weights |
| Physical covariance | Source/field/row and all quadrature dimensions | All supplied arrays, windows/noise, halo prescription and estimator choices | Call-local preparation, released on return |
| Python physical provider | Depends on the new C model | Exact cosmology, photo-z shifts, producer prescription and numerical controls | Replaces latest model; previously returned models remain owned and usable |
| Python prediction/derivative cache | Full selected row count | Model identity, complete IA dictionary, and numerical state token | Changed input invalidates retained results before lookup |

Do not add a shared `static` merely to imitate its spelling in `cosmo2D.c`.
Choose persistence where a later call actually reuses the work. If adding a
mutable static, implement the complete size and value comparisons and its
reset in the same change. A model-owned immutable table has those values
fixed for its lifetime, so construction/destruction performs the corresponding
allocation/refill/reset work. The C++ owner must preserve dependencies: a grid
that borrows a physical model cannot outlive that model's storage.

Production arrays are borrowed only for the duration stated by the C/C++
entry point. The notebook adapter copies through Armadillo and converts
layout explicitly. A returned diagnostic axis or output must be an independent
copy; editing it cannot change the retained C table. In the Cocoa Python
project, survey rows and prepared engine geometry are independent immutable
copies. Construct a new engine in the same interpreter to change its geometry.
Mutating one public array while retaining old weights is a named refusal.

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
  modes, selected outputs, aperture weights, caller/output mutations and workers.
- `covariance/test_setting_sequences.py`: every allocation axis, individual
  coordinate/weight/model changes, estimator/noise switches and old models.
- `test_pipeline_setting_sequences.py`: provider keys, IA/photo-z/calibration,
  global/stage boosts, returned ownership and the outer Fisher mean cache.

For each compared step, a retained process first evaluates other settings,
applies the requested changes, then evaluates again. Its reference is a
separate interpreter configured directly at the final settings. Compare
float64 bit patterns, including signed zero, rather than a tolerance that
could conceal stale values. Same-process reconstruction alone cannot expose
a shared stale table. Exercise both adapters, increasing and decreasing sizes,
same-size value changes, A/B/A object interleaving, destruction/recreation,
strided/read-only notebook inputs, and workers changed after plan creation.

A reach assertion must establish that a changed input enters the calculation.
For example, large-area xi zero modes bypass the radius-profile interpolation;
varying profile density there does not test its refill rule. Use finite-window
xi rows for that test. An IA toggle must reach a nonzero IA component; pure NLA
must return exact zero residuals when expected. A larger L/M cap needs input
with power in the added modes. Old retained objects must still reproduce their
old outputs after a newer object is evaluated.

## Global and individual controls

`python/des3pcf/accuracy.py` in the Cocoa project resolves counts through an
immutable `Accuracy` object. Defaults are multiplicative one, including
`integration_accuracy_boost`. This differs from the ordinary core's integer
integration-level selector; do not pass one to the other as if they had the
same meaning. The C constructors continue receiving explicit resolved counts
and arrays, so the two C++ adapters share one numerical implementation.

Sampling counts use `accuracy_boost`; fixed quadrature counts use
`integration_accuracy_boost`. Counts with two retained endpoints refine their
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
boundary. `test_multipole_lane_invariance.py` adds an unused query or an
extra retained order, then requires the existing coefficients to keep
their float64 bits. The test holds the physical grid and weights fixed;
changing a quadrature resolution would not justify that exact comparison.

Read [NUMERICS.md](../../../../cosmolike/3pcf/NUMERICS.md) for the production
cubic-preparation/linear-read strategy and
[COVARIANCE.md](../../../../cosmolike/3pcf/COVARIANCE.md) for the physical scope.
Record executed strict/debug checks and their limits in the project's
`tests/VALIDATION.md`; source-level reasoning is not a replacement for them.
