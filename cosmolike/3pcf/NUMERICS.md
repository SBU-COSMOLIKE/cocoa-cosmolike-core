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
| Angular Legendre coefficients | Fixed angular projection of the prepared bispectrum | Uniform log-scale and selected angular-coordinate cells |
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

The prepared grid's `signed_baseline: 1` setting uses two RUV tables:
the logarithm of the positive gravity-only projection $`B_{GGG}`$, and
the signed ratio $`B_{\rm NLA}/B_{GGG}`$. Cubic preparation and dense
linear reads apply to each table. Their reconstructed product can be
negative or zero without taking the logarithm of an intrinsic-alignment
signal. The default `signed_baseline: 0` retains the positive-logarithm
representation and rejects nonpositive samples.

Broad alignment priors can cross that sign boundary. Choose the signed
representation explicitly for those calculations; a numerical domain
restriction must not become an undeclared prior. Both representations
still need grid refinement, especially near cancellations. At a zero,
scale an absolute interpolation error by $`B_{GGG}`$ rather than dividing
by the vanishing baseline. This setting does not change the separately
computed TATT residual or its accuracy requirements.


### Signed residuals across a large Fourier range

The residual is the TATT or velocity-shear contribution left after removing
the separately calculated NLA baseline. It is a signed function, so storing
its logarithm would discard negative values and fail at a physical zero.
Storing the raw value also has a numerical cost: a cubic fit along
$`\ln p`$, with $`p=\ell_1+\ell_2+\ell_3`$, can carry an error from a large
sample into a region where the true residual is many orders smaller.
Densely reading that incorrect cubic does not remove the error.

The explicit setting `residual_scaling: 1` prepares

```math
E(p)=B_{GGG}(p/3,p/3,p/3)>0,
\qquad
Q(p,u,v)=\frac{B_{\rm residual}(p,u,v)}{E(p)}.
```

Here $`u=\ell_1/p`$ and $`v=\ell_2/p`$. The denominator uses the same
cosmology, ordered source triplet and matter/baryon prescription, with all
intrinsic-shape amplitudes set to zero. The code stores $`\ln E`$ and
signed $`Q`$, prepares their cubic curves separately, and reconstructs
$`\exp(\operatorname{interp}\ln E)\operatorname{interp}Q`$. The ratio can
pass through zero without any change of sign convention. At the original
sample locations this is the same physical residual up to floating-point
rounding. Between samples it is a different interpolation approximation,
whose accuracy must be checked against direct projection.

This setting removes a broad radial amplitude from the interpolation; it
does not resolve an angular feature or modify the physical IA coefficients.
The default `residual_scaling: 0` retains the raw residual. Pure NLA has
exactly zero residual and prepares neither the scale nor the ratio. When
`signed_baseline: 1` also needs a gravity-only projector, both preparations
share that projector. The temporary positive scales and worker buffers
are allocated before the worker loops and released after table preparation.
A nonpositive denominator or an unrepresentable ratio produces a named
refusal; no absolute value or positive floor replaces a physical result.

### Resolving the physical triangle shape

The rectangular PUV table samples $`u,v\in[m,1/2]`$, where production
uses $`m=\min(10^{-6},\ell_{\min}/(4\ell_{\max}))`$. A physical triangle
also requires $`u+v\geq1/2`$ and a positive closing side
$`1-u-v>0`$. The remaining rectangular cells use the named upstream
continuation outside the triangle domain. The zero-side corner at
$`u=v=1/2`$ is assigned a continuation value; it is not a derivation of a
physical squeezed limit. Keeping these cells is an interpolation convention.
Source labels stay with their original sides throughout. Check boundary
influence with direct physical triangles when changing the shape resolution
or the angular endpoint cutoff.

The `shape_stretch` setting chooses how uniformly spaced coordinates
$`t\in[-1,1]`$ map into physical fractions:

```math
u(t)=m+\frac{1/2-m}{2}
\left[1+\frac{\tanh(s t)}{\tanh s}\right],
\qquad s=\mathtt{shape\_stretch}.
```

The default $`s=3`$ preserves the previous grid. Increasing $`s`$ puts more
nodes near small fractions and near $`1/2`$, where squeezed triangles occur.
It simultaneously spreads the central nodes farther apart. The support
limit $`m`$ and the first interior node are different quantities: an axis
can include $`m\ll10^{-6}`$ while its next node is near $`10^{-4}`$.
Declaring that support alone does not resolve the intervening shapes.
Changing `shape_stretch` therefore needs a separate `nshape` refinement,
including ordinary, squeezed and nearly flattened physical triangles.
The scale suggested by $`s\simeq\tfrac12\ln[(1/2)/m]`$ is a sampling
choice to investigate, not an accuracy guarantee.

Close to a machine-precision boundary, even recovering a stored knot can
be ill-conditioned. The short side $`p(1-u-v)`$ subtracts order-one fractions,
and converting rounded sides back into $`u,v`$ changes their final bits.
The inverse hyperbolic tangent can amplify those small changes near its
endpoints. Exact endpoint branches protect the declared endpoints, but
they cannot restore information lost in a nonzero side ratio. Interior
knot tests, high-precision side-geometry checks, and convergence on near-edge
physical triangles answer different questions. A broad tolerance applied to
all three would conceal which calculation is failing.

### Physical angles and their interpolation coordinate

The angular multipoles use
$`\mathrm{ELL}=\sqrt{\ell_1^2+\ell_2^2}`$ and
$`\mathrm{PSI}=\operatorname{atan2}(\ell_2,\ell_1)`$. Two ordered charts
retain $`\psi=\min(\mathrm{PSI},\pi/2-\mathrm{PSI})`$ and remember which
leg was larger. A very wide Fourier range makes $`\psi_{\min}`$ tiny.
Uniform spacing in $`\ln\psi`$ then devotes many samples to small angles
and can leave large gaps near $`\pi/4`$, where equal-side configurations
can vary rapidly. Raising only the dense interpolation factor leaves those
missing coarse samples unresolved.

With `psi_scale: s_psi` positive, production instead samples uniformly in

```math
x(\psi)=\ln\psi+\frac{\psi}{s_\psi},
\qquad
\frac{d\psi}{dx}=\frac{\psi s_\psi}{\psi+s_\psi}.
```

For $`\psi\ll s_\psi`$ the spacing is approximately logarithmic; for
$`\psi\gg s_\psi`$ it is approximately linear in the angle. The scale is
in radians. The default `psi_scale: 0` retains the original logarithmic
coordinate. A positive scale changes where a fixed number `npsi` of coarse
samples is spent; it does not add samples or establish a target error.

Only preparation inverts this relation. A fixed bracketed root calculation
produces the physical log-angles at each coarse or dense coordinate, with
exactly retained physical endpoints. The exported `log_psi` axis continues
to contain $`\ln\psi`$, which is generally nonuniform in this mode. A separate
internal `psi_coordinate` array is uniform and supplies arithmetic cell
indices, interpolation fractions and cubic preparation. There is no new
binary search or inverse root in the repeated production reader. There is
also no new integration Jacobian: this is an interpolation coordinate,
while the Legendre projection still integrates over its declared $`\mu`$
nodes. The source `reference_mode: 1` keeps its original angular axis and
reader regardless of `psi_scale`.

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
| Multipole query map inside the plan | Exact requested Fourier nodes, complete projected-grid axes, their dimensions, the Legendre cutoff and reader convention |
| Aperture weights | Side/angle bins, widths, three ordered filter radii and the shear convention |
| Project mean | Immutable model, exact IA values and owned observable geometry; shear calibration is applied afterward |

A cache key is the list of inputs compared before reuse. Sizes alone are
not enough: two arrays with the same shape can contain different source
curves or Fourier nodes. The allocation decision and the physical refill
decision must both include every input they use. Python's exact parameter
keys retain floating-point values without a rounding tolerance, so a small
Fisher displacement is not mistaken for the reference cosmology.

The query map makes this distinction concrete. A transform repeatedly asks
for the bispectrum multipoles at the same pairs of Fourier nodes. For each
pair, the map retains the interpolation cell and its four bilinear weights.
Changing the cosmology or alignment amplitude changes the values at the
cell's corners, so every call reads the current physical grid and overwrites
the entire multipole buffer. The geometry alone can survive that change.
Moving an interior interpolation node requires rebuilding the map even when
the array dimensions and endpoints stay fixed. The key therefore compares
every copied axis value, not just its size or address.

Changing the angular coordinate can require new weights even when the
physical nodes coincide. With two PSI nodes and no dense upsampling,
`psi_scale: 0` and a positive scale expose the same physical endpoints,
but give different interpolation fractions between them. The query key
therefore includes `psi_scale`, the complete internal coordinate array,
and the physical `log_psi` array, as well as the Fourier axis, dimensions,
Legendre cutoff and reference-mode choice. By contrast, `residual_scaling`
and `shape_stretch` change the current physical table values; if the b_L
query geometry is unchanged, the map may still read those new values.
Neither the normalized residual nor its positive scale is retained as a
physical-value cache inside the geometric map.

`TransformPlan.aperture_from_grid` performs this read and the aperture
calculation in C. Its notebook counterpart copies the supplied weights
through Armadillo and returns the same one-dimensional filter vector.
The two-step `grid.multipoles` followed by `aperture_from_multipoles` remains
available when the multipoles themselves are needed. The plan owns its
temporary buffer; editing a returned vector cannot modify a later result.

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

The same separation reduces the $`P\,T`$ covariance term. Here $`P`$ joins
one field from each cubic estimator, while the one-halo connected four-point
function $`T`$ joins the other four. There are nine choices of the two fields
assigned to $`P`$. Several covariance entries ask for the same radial dot
product, specified by the two single-field labels and the two complementary
radius pairs. Each needed ordered choice is calculated once per halo mass
and distance. The remaining source weights and the nine-term sum retain
their order in each matrix cell. Reversing the field pair is not used to
merge entries: finite-window products can have a different rounding order.
All contraction values are refilled at the next halo and released when the
covariance call returns. The six additional finite-window terms remain
separate from these nine contractions.

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

Projection batches store radial weights with the source triplet as the
contiguous index. Two SIMD lanes can therefore load neighboring triplets
from one array while sharing the same matter and intrinsic-shape calculation.
The weights depend on the immutable model and alignment parameters, and are
prepared before triangle workers start. The packed owner is destroyed before
the projectors it borrows. Increasing the batch size shares more physical
work but keeps more projected grids in memory; it is a memory choice, not an
accuracy control.

A scalar tail must round at the same points as a vector lane. Strict
floating-point flags can still permit a scalar fused multiply-add, which
rounds once, while the explicit SIMD product and sum round separately.
Local rounding boundaries keep an existing multipole unchanged when an
extra query or Legendre order moves it between those paths. The covariance
checks similarly add zero-weight nodes that change lane placement without
changing the mathematical integral.

The same condition applies to physical triplet batches, direct angular
multipoles, NLA power reads and aperture-filter batches. An additional output
can move the previous final scalar output into a paired lane. Its physical
weights have not changed, so its arithmetic should not change either. The
unpaired NLA read uses a duplicated prepared coordinate in the two-lane
reader, retaining the first result. The mixed interpolation/extrapolation
branch also keeps the valid lane's original rounding boundaries.

FFTW uses the plans prepared for the outer calculation. Independent
angular modes can run on separate OpenMP workers, each with private
Fourier scratch. Starting an additional FFT thread team inside each such
worker would multiply the active workers and compete for the same cores.
Thread-count changes must respect the capacity owned by the plan.
Modes with two inverse transforms are scheduled after those with four.
Workers take the next available mode, while each completed result is stored
under its physical angular order. A worker still evaluates each Legendre
sum in increasing order. The coupling table already includes its fixed
$`(-1)^L`$ factor, and two independent SIMD pairs advance four output cells
without regrouping any cell's sum.
BLAS remains at one thread. Parallel CPU work belongs to OpenMP; a BLAS
thread team inside an OpenMP worker would oversubscribe that allocation.

## What a numerical comparison establishes <a name="validation"></a>

Three checks answer different questions. A pinned-source comparison
tests whether the stated source convention was translated. An independent
analytic or quadrature calculation tests a physical normalization or
operator. A refinement test asks whether the chosen discretization has
resolved the observable. Agreement with the first does not replace the
other two, particularly when a source shortcut assumes exchange symmetry
that an ordered tomographic TATT term does not have.

Diagnose the stages separately. First compare projected-table values with
the direct line-of-sight calculation at physical triangles. Use original
interior knots to check storage and signs, then off-node ordinary and edge
shapes to check interpolation. An error divided by a nearly vanishing signed
residual is misleading; retain an absolute error with a stated nonzero
physical scale, such as $`B_{GGG}`$.

Next hold that projected table and its $`\mu`$ nodes fixed. Independently
integrate each piecewise-linear angular segment against $`P_L(\mu)`$, and
compare with `grid.multipoles` at off-node ELL/PSI pairs in both ordered
charts. A Gauss rule of sufficient degree integrates this polynomial
product exactly. Vary `nell`, `npsi`, `psi_scale` and `multipole_boost`
separately, checking that the projected samples did not change. For a small
or vanishing $`b_L`$, the scale
$`\frac12\int|B_{\rm linear}(\mu)|\,d\mu`$ avoids a division by that
coefficient. This comparison isolates multipole interpolation; it does not
validate the physical projected table or the omitted angular endpoint cap.

Only after those checks should transform, finite-aperture and derivative
refinement be interpreted together. In particular, a plausible-looking
aperture amplitude or a stable Fisher ratio cannot certify the earlier
stages. The optional representations above have separate controls because
radial amplitude, triangle shape and multipole angular sampling are separate
sources of numerical error. Their presence does not establish convergence
for a selected survey data vector.

The [executed validation record](../../../../../projects/des_3pcf/tests/VALIDATION.md)
identifies the checks and build modes actually run. Survey-size accuracy
requires the selected Y3 or Y6 rows, a matched covariance and derivative
step checks. Runtime claims require a separate controlled measurement on
an available machine. The implementation choices described here identify
reused work and arithmetic structure; they are not measured speedups.
