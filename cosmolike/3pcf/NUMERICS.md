# Numerical preparation and repeated calculations

1. [Why there are two table resolutions](#tables)
2. [Uniform coordinates and interpolation](#interpolation)
3. [Integration rules are a separate choice](#integration)
4. [Reuse, ownership and changing inputs](#reuse)
5. [OpenMP and SIMD](#parallel)
6. [What a numerical comparison establishes](#validation)

## Why there are two table resolutions <a name="tables"></a>

A projected bispectrum requires a line-of-sight integral for each Fourier
triangle. Repeating that integral every time a shear transform requests a
nearby triangle would duplicate most of the physical work. The calculation
therefore separates the nodes where it evaluates the physical model from
the denser table used by the repeated interpolation. This is the same
distinction used for angular-power tables in `cosmo2D.c` and the prepared
power tables in `covariances/`.

The preparation sequence has three parts. First, calculate the expensive
function on a coarse grid. Second, solve for cubic coefficients and fill
a denser uniform grid. Third, let the integration or transform read that
dense grid by linear interpolation. The cubic solve and table allocation
belong before the repeated calculation. The reader performs only index
arithmetic and a weighted combination of neighboring values.

These two resolutions control different errors. More coarse nodes can
resolve curvature that was absent from the original samples. More dense
nodes reduce the error of reading the already constructed cubic curve
with straight segments. Increasing the dense refinement alone cannot
recover an unresolved feature in the coarse calculation. Both choices
need separate refinement checks.

| Table | Physical calculation or input | Repeated read |
|---|---|---|
| Matter power and background | Supplied physical arrays, resampled during model construction | Uniform log-wave-number/redshift cells |
| Matter-fit coefficients | Fixed uniform redshift anchors | Dense uniform redshift table |
| Projected bispectrum | Fixed Fourier-triangle charts and radial nodes | Dense triangle-coordinate tables |
| Angular Legendre coefficients | Fixed angular projection of the prepared bispectrum | Uniform logarithmic scale/shape cells |
| Halo shear profiles | Fixed Fourier quadrature at radial preparation nodes | Uniform transformed-radius cells |

The external input tables need not already be uniform. Their conversion
can use a general interpolator during serial preparation. That does not
put a spline or a search inside the production reader. Named upstream
reference modes preserve their original interpolation sequence for
cross-code comparisons; they are distinct numerical choices.

## Uniform coordinates and interpolation <a name="interpolation"></a>

Uniform means evenly spaced in the coordinate actually stored. For a
logarithmic multipole table the coordinate is $`x=\ln\ell`$, rather than
$`\ell`$. With spacing $`\Delta x`$, the lower cell and its fractional
position are

```math
j=\left\lfloor\frac{x-x_0}{\Delta x}\right\rfloor,
\qquad
t=\frac{x-x_0}{\Delta x}-j,
\qquad
f(x)=(1-t)f_j+t f_{j+1}.
```

There is no binary search because the coordinate determines the cell
directly. Endpoint handling must keep $`j+1`$ inside the table. The
declared extrapolation rule is a separate scientific choice; clamping an
out-of-range request is not an interpolation accuracy test.

For repeated multipole requests, calculate all indices and fractions
once. Every Legendre order then reads the same prepared map. Two-dimensional
interpolation uses four neighboring values; a three-dimensional chart
uses eight. Keeping the final coordinate contiguous lets neighboring
outputs share the same memory layout as the angular-power reads used by
the existing $`\xi_\pm`$ calculation.

The positive matter/NLA baseline can be stored logarithmically only where
its sign is known. Residual intrinsic-alignment terms can cross zero and
must retain their signed values. Source labels remain attached to their
Fourier sides when a triangle is sorted or mapped to another shape chart.
Equal array dimensions do not make those permutations interchangeable.

## Integration rules are a separate choice <a name="integration"></a>

Fixed integration nodes are prepared before the physical calculation.
The Fourier, angular, mass and radial rules have explicit nodes and
weights. The same nodes are used at every displaced cosmology in a
derivative. There is no data-dependent subdivision of an integral while
an OpenMP worker evaluates it.

A Gauss-Legendre rule has nonuniform nodes. This is consistent with a
uniform interpolation table: the integration nodes ask for function
values, and arithmetic indexing locates those values on the table.
Replacing a fixed quadrature merely to make its nodes uniform would
change its numerical error without removing a lookup search that the
prepared reader already avoids.

For a logarithmic integration coordinate, the Jacobian belongs in the
weights. For example, $`d\ell=\ell\,d\ln\ell`$. The project quadrature
builder returns weights in $`d\ell`$ even when it places the nodes in
$`\ln\ell`$. The C integrand then adds only the physical factors required
by its equation. Multiplying by that Jacobian again changes the units
and the answer.

The source-comparison angular projection integrates its piecewise-linear
bispectrum against Legendre polynomials with prepared weights. It should
not be described as a direct Gauss-Legendre evaluation of the physical
bispectrum. Both methods use fixed nodes, but they approximate different
intermediate functions.

## Reuse, ownership and changing inputs <a name="reuse"></a>

An immutable `PhysicsModel` owns copies of its power, distance, growth,
source and radial arrays. Constructing another model is the explicit
operation that changes those inputs. A transform plan owns its angular
geometry, FFTW plans and worker buffers. A new cosmology supplies new
bispectrum values while retaining a compatible transform plan.

| Retained object | Inputs that define its contents |
|---|---|
| Physical model | All physical arrays, cosmological scalars, source kernels and preparation controls |
| Projection tables | Physical model, ordered source labels, all IA/baryon coefficients, triangle ranges and resolution controls |
| Transform plan | Fourier and angular coordinates, selected bins, angular orders, FFT bias/padding/window, projection convention and worker capacity |
| Aperture weights | Side/angle bins, widths, three ordered filter radii and the shear convention |
| Project mean | Immutable model, exact IA values and owned observable geometry; shear calibration is applied afterward |

A cache key is the list of inputs compared before reuse. Sizes alone are
not enough: two arrays with the same shape can contain different source
curves or Fourier nodes. The allocation decision and the physical refill
decision must both include every input they use. Python's exact parameter
keys retain floating-point values without a rounding tolerance, so a small
Fisher displacement is not mistaken for the reference cosmology.

The covariance has a different lifetime from the repeatedly evaluated
mean. It is an offline calculation with explicit shared inputs. Within
one calculation it prepares the halo, filter, Bessel and window tables
once, then shares them across covariance terms and row pairs. A persistent
cosmology cache is unnecessary when there is no subsequent call that can
reuse it. This distinction follows the existing covariance implementation;
the presence of a file-scope `static` is not itself an optimization.

Source-bin factors can often be separated exactly. In the pure-lensing
halo model, a filtered profile is $`\Phi_{i,R}=W_i\phi_R`$. All sources
with the same radius therefore share $`\phi_R`$. A pair transform depends
on the two radii, while its source efficiencies multiply the result.
For five distinct radii there are fifteen unordered radius pairs. Creating
separate transforms for every source/radius label would repeat the same
radial integral. This factorization requires the declared physical model;
it must be rederived before adding a source-dependent halo prescription.

## OpenMP and SIMD <a name="parallel"></a>

OpenMP assigns independent outputs to workers. Each worker writes its
own rows or triangle cells and owns its scratch storage. FFTW planning
and allocations occur before those workers start. Inner numerical sums
retain their declared node order; changing the worker count does not
change which worker contributes to a shared floating-point reduction.

SIMD evaluates several independent arithmetic operations in one vector
instruction. Here a lane can hold one neighboring multipole output, one
Legendre coefficient or one aperture radius. For example, two lanes can
evaluate the linear-read formula above at two prepared cells while both
use the same physical table. The code comments identify the quantity in
each lane and the scalar expression it implements.

SIMDe intrinsics express these operations explicitly under the strict
floating-point build. An OpenMP SIMD annotation alone does not establish
that a compiler vectorized a loop. Scalar special-function preparation,
array gathering and horizontal summation remain distinct costs. Adjacent
outputs avoid changing an integration sum into an unordered reduction;
where lane products must be summed, the code restores their scalar order.
Odd output counts use an explicit scalar tail.

FFTW uses the plans prepared for the outer calculation. Independent
angular modes can run on separate OpenMP workers, each with private
Fourier scratch. Starting an additional FFT thread team inside each such
worker would multiply the active workers and compete for the same cores.
Thread-count changes must respect the capacity owned by the plan.

## What a numerical comparison establishes <a name="validation"></a>

Three checks answer different questions. A pinned-source comparison
tests whether the stated source convention was translated. An independent
analytic or quadrature calculation tests a physical normalization or
operator. A refinement test asks whether the chosen discretization has
resolved the observable. Agreement with the first does not replace the
other two, particularly when a source shortcut assumes exchange symmetry
that an ordered tomographic TATT term does not have.

The [executed validation record](../../../../../projects/des_3pcf/tests/VALIDATION.md)
identifies the checks and build modes actually run. Survey-size accuracy
requires the selected Y3 or Y6 rows, a matched covariance and derivative
step checks. Runtime claims require a separate controlled measurement on
an available machine. The implementation choices described here identify
reused work and arithmetic structure; they are not measured speedups.
