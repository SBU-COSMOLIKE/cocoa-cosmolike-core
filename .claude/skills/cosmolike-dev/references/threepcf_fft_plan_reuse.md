# Retained double-Bessel FFTLog preparation

Read this when changing `TwoBesselPlan`, the standalone `two_bessel`
adapters, or the FFTW workspace lifetime. The mathematical transform stays
in `cosmolike/3pcf/fftlog_3pcf.c`; the owner and Python registration are in
`transform_interface.cpp`, and the copied notebook stages are in
`transform_wrapper.cpp`. The existing finite-triangle `TransformPlan`
already retains the same C workspace across input bispectrum arrays.

## What an owner retains

For a real input f sampled on a uniform logarithmic ell grid, the operator
is the double integral of f times J_order1(ell1 theta1) times
J_order2(ell2 theta2), with measure dln(ell1) dln(ell2). The FFTLog workspace
prepares power-law bias factors, analytic Mellin kernels for both Bessel
orders, output factors, two serial FFTW plans, and flat aligned scratch.
The output coordinates are `theta = xy / ell[::-1]`. With `dlnt > 0`, these
are lower radial-bin edges, and the operator averages using theta dtheta.

Those preparations depend on the grid and numerical settings. They do not
depend on the latest f. The owner therefore retains preparation while
requiring a forward transform whenever f changes. A forward call copies the
complete input, initializes the padded real array, and overwrites all forward
coefficients. An inverse call uses a separate complex scratch array, leaving
those coefficients available for another Bessel-order pair. Its output uses
the input workspace as scratch; the next forward must copy every input cell.

`max_order` prepares every nonnegative cylindrical Bessel order through that
value. It is immutable, as are ell, the two biases, xy, bin width, padding,
and taper settings. Construct another owner after changing any of them.
Increasing `max_order` uses more Mellin-table memory but does not change the
arithmetic of an already prepared lower order. Preparing many unnecessary
orders is not free.

## Production and copied notebook routes

The production constructor and forward/evaluate methods accept aligned,
C-contiguous float64 NumPy arrays. They reject implicit dtype/layout
conversion. The input is borrowed during that call and copied into the C
workspace; it is safe to change the caller's array after forward returns.
The `theta` property and every inverse/evaluate result are independent
arrays. Changing or retaining an old output cannot change a later result.

```python
plan = compiled.production.TwoBesselPlan(
    ell=ell,
    max_order=6,
    nu1=1.01,
    nu2=1.01,
    xy=1.0,
    dlnt=0.0,
    padding=0,
    window=False,
    window_width=0.25,
)
```

For a new integrand and one order pair, call
`plan.evaluate(integrand=values, order1=2, order2=3)`. To reuse one input for
several order pairs, call `plan.forward(integrand=values)` once, followed by
`plan.inverse(order1=..., order2=...)` for each pair. Inverse before the first
forward and orders outside `[0, max_order]` stop with a named error. Evaluate
checks the orders before replacing the input. A rejected Python dtype/layout
conversion leaves the previous valid coefficients intact.

The notebook factory is `compiled.notebook.two_bessel_plan`, with the same
constructor parameters. It accepts NumPy-compatible arrays and lists. Use
`compiled.notebook.two_bessel_forward(plan=plan, integrand=values)`,
`two_bessel_inverse(plan=plan, order1=..., order2=...)`, and
`two_bessel_evaluate(plan=plan, integrand=values, order1=..., order2=...)`.
These adapters make the documented Armadillo copies and pack row-major C
input explicitly. The returned object is the same production owner type;
calling its production methods still requires the production array contract.
The original one-shot functions remain available and retain their defaults.

## Threading and reset contract

A standalone double-Bessel owner allocates one workspace slice and uses
serial FFTW. Parallel angular modes belong to the finite-triangle
`TransformPlan`, which retains independent worker slices under outer OpenMP.
Keep BLAS at one thread and do not add internal FFTW threads.

All Python entry points retain the GIL. This serializes independent Python
calls while one object uses its mutable scratch. It does not establish that
the raw C workspace is safe for two host invocations at once. FFTW planning
and destruction remain outside OpenMP execution. The workspace compares
each worker's real, forward-complex and inverse-complex buffer alignment with
its corresponding planning buffer using `fftw_alignment_of`. A mismatch
stops before use; it does not select `FFTW_UNALIGNED` and lose aligned FFTW
execution.

Each owner releases its own workspace through a `unique_ptr` deleter. The
same cleanup also applies if constructing a Python output axis throws. There
is no process-global FFTW cache and no borrowed plan owned by another object.
A global core reset can therefore leave explicit retained owners alive;
their geometry has no dependency on the mutable cosmology or Ntable structs.
Do not call `fftw_cleanup` to release one owner: other modules may still own
live plans.

## Evidence and limits

The focused validation lives in `dev_scripts/threepcf_fft_plan_check/`.
Its build script copies the sources and the project's transform tests into
a private directory before compiling. Only that copy substitutes external
FFTW wrappers and registers lifecycle counters. Production source has no
test-only C API, lifecycle counters, or per-execution lock. In the private
extension, plan creation/destruction updates serialized counters; OpenMP
execution only reads the completed recipe map.

The private allocator fills workspace allocations with NaNs. Immediately
before each FFT, the wrapper checks the entire input, including padded
regions, for finite values and verifies the actual new-array alignment
classes. This catches incomplete initialization that a zeroed allocator
could hide. It is validation instrumentation and must never be timed as a
performance build.

The focused checks cover both adapters, changed inputs A/B/A, low and high
Bessel orders, equal padded sizes with different physical sizes, changed
ell values/spacing, both biases, xy, averaging, padding, taper width, caller
mutations, independent owners, destruction order, core reset, dtype/layout
refusal, and fresh-process agreement. Existing NumPy FFT/SciPy log-Gamma
references check the finite operator independently. Existing angular,
projection and aperture tests still supply their own references.

The lifecycle assertions distinguish ten one-shot calls (twenty FFTW plan
creations and destructions) from ten calls through one standalone owner
(two creations, no intervening destruction). Retained finite-triangle owners
are also checked at one, four and six OpenMP workers. Six is the explicit
user authorization for this validation run, not a new project-wide default.
Full-array bitwise comparison with the one-shot path checks ownership-only
changes; independent references use their declared numerical tolerances.
Consult the validation run record for completed commands and results.

These counts do not establish a Fisher-loop speedup. The finite-triangle
science route already retains `TransformPlan`, and standalone calls are
primarily component/reference uses. The retained API provides reusable
preparation to callers that choose the one-shot function. No
project caller is silently rewritten.

The planner remains `FFTW_ESTIMATE`. Sharing a pair between distinct geometry
owners, or changing to `FFTW_MEASURE`, remains conditional on measured cost
and numerical evidence from a quiet machine. Equal padded dimensions alone
are not a sufficient future ownership rule: layouts, alignment, precision,
planner flags, thread policy and live users must also be accounted for.
The current explicit owners avoid adding that lifetime complexity without
evidence that repeated geometry construction is a material cost.
