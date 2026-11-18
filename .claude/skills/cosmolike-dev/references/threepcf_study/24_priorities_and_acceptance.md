# Implementation priorities and acceptance conditions

> **Archive status.** This source study records design proposals and validation requirements, not completed features. Read the [maintained implementation status](../threepcf.md) first. Both the mean and its covariance belong in `cosmolike/3pcf/`; earlier destination proposals using `fastnc/` or `covariances/` are superseded.

The study supports starting a CPU port. It does not establish that every
useful optimization has been found, or that ten workers will reduce the
reported latency by a factor of ten. The next useful evidence comes from
an implemented, independently checked calculation. Further broad source
reading should be driven by a specific missing physical term, failed
comparison or measured bottleneck.

## 1. Location and first implementation boundary

The user selected `cosmolike/3pcf/`, beside `cosmolike/covariances/`, for
all new three-point numerical code, including its analytic covariance.
Earlier suggestions of `cosmolike/fastnc/` or putting this covariance in
the existing covariance directory are superseded by that choice.
Duplicating helpers is permitted to preserve the established data-vector
C files. A new project under `projects/` will own its build, bindings,
test scripts and survey configurations, using `lsst_y1` as the layout
example. Scientific routines belong to the core, not to test scripts.

The core README distinguishes two C++ adapters. Production `_interface`
functions borrow suitable contiguous NumPy inputs and call shared C
calculations. Notebook `_wrapper` functions expose named Armadillo arrays
for inspection. They must call the same C routines rather than implement
the physics twice. A portable core skill reference must record the
supported equations, units, conventions, ownership, validation status and
remaining tickets. A later user should not need this conversation or the
original LSST project to understand the module.

The first component should have independently checkable physical outputs,
both adapters, a project test command, and explanations beside the code.
An empty directory or a new binding with no verified numerical result is
not an adequate implementation milestone. Conversely, passing component
tests does not establish either requested paper reproduction or a usable
Y3/Y6 forecast. Those remain explicit later acceptance stages.

## 2. Rank by eliminated work, not an invented speed factor

| Priority | Opportunity | Work eliminated or shared | Effort and principal validation |
|---|---|---|---|
| First | Correct source conventions and complete cache keys | Prevents inconsistent inputs and false reuse | Small to moderate; odd/even grids, same-shape coordinate changes, unit and phase tests |
| First | Prepared geometry, source windows and power readers | Repeated setup with unchanged inputs | Moderate; fresh/reused arrays and changed-state sequences |
| First | Shared aperture weights and batched contractions | Rebuilding the same measurement weights; repeated reads across rows | Moderate; independent complex-weight calculation and unequal-filter cases |
| Next | Three-vertex residual TATT assembly | Repeated quadratic shapes and seven weighted integrands | Moderate; compare seven labels with three insertions at identical radial nodes |
| Next | Common-side $`BB`$ factorization | Cartesian product of two independent triangle integrations | Moderate; identical domains and independent direct quadrature |
| Next | One-halo row responses and pair/triple transforms | Repeated halo-profile work for many covariance entries | Larger; cut-response definition, direct selected integrals and full-block checks |
| Next | Batched remaining covariance outputs | Repeated physical samples across output entries | Moderate; per-output integration error, including small sign-changing entries |
| Conditional | Finite-grid aperture dual weights | Prediction-time FFT, phase and angular reconstruction after a fixed boundary | Larger; exact transpose, memory budget, nonlinear-stage boundaries |
| Conditional | Full-filter Hankel or analytic-angle evaluation | Angular or triangle integrations for eligible terms | Larger; estimator comparison and term-specific separability |
| After structure | OpenMP and SIMDe in remaining kernels | Executes independent work concurrently or in vector lanes | Moderate; ownership, repeated runs, thread-count comparisons and generated instructions |
| Fisher stage | Analytic nuisance derivatives and exact request reuse | Redundant displaced predictions | Moderate; derivative checks, null directions and exact state identity |

The ordering is based on visible repeated work and the difficulty of
establishing equivalence. It is not a measured performance ranking.
The first implementation can include simple explicit SIMD arithmetic
where the independent lanes are already defined; its performance must
remain unclaimed until measured on a suitable machine.

## 3. Savings that must not be multiplied together

The full aperture dual-weight route removes the downstream transform
chain from repeated predictions. Eliminating a duplicate inverse FFT or
reusing its plan then helps dual-weight construction and diagnostic
forward calls; it is not another independent factor on work already
removed. Prepared physical inputs and three-vertex IA collection remain
upstream, so they can combine with either route.

For a given eligible full-filter term, Hankel products and analytic
Gaussian-angle integration are alternative evaluators. Their agreement
is useful as a numerical check. Counting both as successive reductions
of that same integral would count the same work twice. Different terms
can choose different evaluators while preserving one stated observable.

After common-side factorization of a bispectrum-product covariance term,
batch and parallelize the remaining triangle tables and final
contraction. Do not also credit the cost of batching the original nested
integrand, which no longer exists. Similarly, a shared halo profile bank
is prepared once for connected, power–trispectrum and response terms.
Its reuse is one saving across those consumers, not three separate
multiplicative gains.

Analytic IA or calibration derivatives remove some finite-difference
requests. Apply the $`4p+1`$ request-reuse count only to the $`p`$
parameters still differentiated with that four-displacement stencil.
Changes of step size or provider configuration create new requests.

## 4. Setup cost and memory determine when reuse is useful

Let $`S`$ be the cost of constructing a fixed operator, $`F`$ its original
forward cost per use, and $`D`$ the cost of a contraction with retained
dual weights. For $`N`$ uses of that exact operator, it reduces total
work only if

```math
N(F-D)>S,\qquad F>D.
```

These symbols are costs to measure later, not supplied timings. Count
tomographic uses that genuinely share the operator as well as repeated
cosmologies. If source-dependent cuts change the operator, those uses
belong to different setup costs.

At the illustrative $`150^2`$ grid with modes zero through 100, five real
dual rows occupy 90.9 MB. Thirty-five distinct radius triplets occupy
636.3 MB. These numbers count stored doubles, not total resident memory
or the actual tuned FFT size. A dot product must still read the weights.
Batching compatible tomographic outputs can reuse those reads, but
storing every possible operator can defeat the memory budget.

Chapter 23 counts persistent inputs, active workspaces, interface copies
and worker scratch together. Its newly identified Slepian source issue
illustrates why compressed mathematical dimensions are insufficient:
rank-truncated slices can retain full singular-vector allocations, and
reconstruction checks create additional dense arrays. Keep that optional
route outside the initial TATT implementation. Use owned compact factors
and bounded construction only after its approximation is accepted.

## 5. Acceptance stages for the port

**Physical components.** Establish units and signs with direct small
examples: known Gaussian filter integrals, independent tensor IA
contractions, unequal triangles, unequal filters, and analytic noise
limits. Preserve the distinction between a paper equation, a particular
source convention, and an independently selected correction.

**Numerical composition.** Compare intermediate arrays before comparing
final predictions. Refine radial, angular and transform settings
separately and then together. A fixed-grid identity is checked at that
grid; a changed discretization needs convergence. The old phase/bin
averaging and logarithmic-interpolation questions remain explicit tests.

**State and parallel ownership.** Change dimensions and coordinates,
source counts, physical parameters and worker counts in sequence;
restore the original inputs and compare with a fresh process. Run both
strict and debug builds. A repeated-request cache must be bypassed in a
test whose purpose is to find nondeterministic recomputation.

**Joint observable.** Supply the actual pair/triplet weights and noise
rules, then validate the full selected two-point, three-point and cross
covariance. Test symmetry, positivity, degenerate limits and parameter
constraints under numerical refinement. Passing the unrestricted
aperture covariance tests does not close this stage for finite catalogue
estimators. PME can be assessed only with the joint baseline and mock
assumptions described in Chapter 18.

**Reproduction and forecasts.** Freeze the Y3 and Y6 survey definitions,
priors, masks and data ordering separately. Reproduce the requested
physical component results and compare direct derivatives with the
original pipelines before interpreting Fisher differences. Keep the
full selected two-point vector primary and compression optional.

**Performance.** After numerical acceptance, measure setup, changed-model
prediction, nuisance-only updates and complete fiducial covariance
separately. Use a quiet machine and a fixed core/memory budget. No
benchmark or speedup claim is part of this study. The user has authorized
beginning implementation after this final pass; that authorization does
not remove the busy-laptop timing restriction.

At each substantial component, read its complete code and documentation
as an advanced undergraduate. The reader must be able to explain what
is being integrated, what each array index means, why a cached value
remains valid, and why workers cannot overwrite one another. Record that
review with the tests before proceeding to the next component.
