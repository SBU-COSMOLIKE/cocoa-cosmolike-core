# Three-point shear and aperture statistics

1. [What can be calculated?](#scope)
2. [From matter and galaxy shapes to angular statistics](#physics)
3. [From galaxy shapes to aperture mass](#aperture)
4. [What the covariance components mean](#covariance)
5. [Files and the two C++ adapters](#files)
6. [Arrays, units and reuse](#arrays)
7. [Parallel work and numerical checks](#checks)
8. [Credits and scientific validation](#credits)

## What can be calculated? <a name="scope"></a>

This directory owns the new three-point numerical calculations, including
their joint covariance. It sits beside `covariances/`; existing data-vector
C files remain unchanged. The physical input is a cosmology together with
linear/nonlinear matter power, distances, growth and source-redshift
distributions. An optional CAMB table producer belongs to the companion
project. The C calculation then predicts the matter and intrinsic-shape
bispectra, angular shear correlations and aperture moments.

| Implemented calculation | Input | Result |
|---|---|---|
| BiHalofit and intrinsic-shape bispectra | Physical tables, triangle sides and IA coefficients | Matter fit and 19 active TATT/velocity-shear components |
| Radial projection and angular multipoles | Physical model and an ordered source-bin triplet | Projected E bispectrum and its Legendre coefficients |
| Natural-component transform | Legendre coefficients and fixed geometry | Four complex shear correlations, using two-dimensional FFTLog |
| Full-filter aperture mean | Projected bispectrum, three filters and fixed Fourier quadrature | Third moment for the unrestricted circular-filter estimator |
| Finite-triangle aperture weights | Two side grids, opening angles and three radii per output | Reusable complex integration weights |
| Aperture contraction | Weights and four complex natural components | One aperture moment per sample and radius triplet |
| Physical joint covariance | Matter model, ST/NFW halo prescription, noise, rows and quadratures | Separate Gaussian, BB, PT, PB, connected and selected long-mode terms |
| Filtered pair correlations | Angular power, two Fourier filters and an explicit quadrature | Correlation of two filtered fields at each separation |
| Gaussian cubic-map covariance | Filtered pair functions and centre-separation weights | All fifteen Gaussian six-field contractions |
| Connected response contraction | Observable responses and nonnegative integration weights | A symmetric Gram matrix |

The physical covariance constructs its own halo profiles and responses.
It describes full-filter map estimators with specified circular or square
center windows. Catalogue masks, distinct-galaxy exclusions and finite
triangle selection require different estimator calculations. Nonlinear
intrinsic-alignment connected covariance is also outside the selected
matter/halo model. These limits are explained with the equations in the
[physical covariance guide](COVARIANCE.md).

The companion [DES project](../../../../../projects/des_3pcf/README.md)
owns builds, survey-file readers, model preparation and Fisher calculations.
It follows the `lsst_y1` separation of core numerics, project registration
and tests. Its full selected two-point vector is retained. Predicted means
and analytic covariances suffice to define a forecast; a reproduction of
published posterior results additionally needs their data, cuts, priors
and likelihood choices. No survey accuracy or measured speedup follows
from the component tests alone.

## From matter and galaxy shapes to angular statistics <a name="physics"></a>

A matter bispectrum describes correlations among three Fourier modes
whose wavevectors close into a triangle. Its side lengths are positive
wave numbers $`k_1,k_2,k_3`$. BiHalofit predicts this quantity using a
calibrated nonlinear fit, while the separate tree-level result supplies
a perturbative reference. TATT means tidal alignment and tidal torquing:
the local gravitational tidal field can align a galaxy, and quadratic
tidal terms can create further correlations. The velocity-shear extension
adds another specified coefficient. These intrinsic shapes contribute
to observed shear correlations even before gravitational lensing is added.

The model accepts $`k`$ in $`h/\mathrm{Mpc}`$, distances in
$`\mathrm{Mpc}/h`$, power in $`(\mathrm{Mpc}/h)^3`$ and bispectra in
$`(\mathrm{Mpc}/h)^6`$. Growth is $`D(z)`$ with $`D(0)=1`$.
The ordinary core's growth table stores a quantity proportional to
$`D/a`$ instead; the project bridge makes that conversion explicitly.
Source arrays are probability densities per unit redshift. Their
normalization is calculated from the supplied grids.

The model uses spatially flat lensing geometry. It requires
$`|\Omega_m+\Omega_{\rm de}-1|\leq10^{-8}`$ as a scalar-rounding
tolerance; it does not change either supplied value. Supplying a curved
distance table would not replace the flat distance ratios in the lensing
kernel. The project CAMB producer explicitly requests flat geometry.

For gravitational lensing, the flat-sky Limber projection is

```math
B_\kappa^{ijk}(\ell_1,\ell_2,\ell_3)
=\int d\chi\,\frac{W_i(\chi)W_j(\chi)W_k(\chi)}{\chi^4}
B_m(\ell_1/\chi,\ell_2/\chi,\ell_3/\chi;z(\chi)).
```

Here $`W_i`$ is the usual lensing efficiency, with units of inverse
distance. A factor of $`\chi^{-4}`$ converts the three-dimensional
bispectrum to the angular one. The implementation preserves the source
convention with no $`\ell+1/2`$ substitution. Its internal `radial()`
arrays named `lens` and `shape` use fastnc's rescaled kernel convention;
they are not bare $`W_i`$. The covariance code converts those arrays before
using the physical powers of distance.

The deterministic NLA part replaces each lensing efficiency by its
lensing-plus-linear-alignment value. Residual TATT terms are then added
without counting that baseline again. The upstream baseline and residual
source kernels have different normalization/interpolation sequences.
They remain separate here: combining their continuous formulas does not
prove their finite numerical grids are equivalent. The IA power choice is
also explicit: `power_mode=0` uses linear $`P(k,0)D^2`$, `1` uses the
upstream nonlinear $`P(k,0)D^2`$ prescription, and `2` uses the supplied
evolving nonlinear table. `ct_mode=0` defines the physical velocity-shear
coefficient as $`C_t=C_1\,ct`$; `1` takes `ct` itself as that coefficient.

Angular Legendre coefficients reorganize the remaining Fourier triangle
dependence. A two-dimensional Fourier transform in logarithmic side
coordinates evaluates the pair of Bessel integrals; this is FFTLog.
An immutable transform plan retains its angular coupling, Mellin factors,
FFTW plans and worker scratch. It receives new bispectrum coefficients
when cosmology or IA parameters change.

Source labels and triangle sides must move together. With unequal source
bins, ordered TATT coefficients need not be symmetric under exchanging
the two Fourier sides. `ordered_legs=True` therefore evaluates both signs
of angular mode $`M`$ explicitly. `ordered_legs=False` preserves the
historical transpose shortcut for source comparisons. Independent
Cartesian Gaussian-moment tests check the general case as well as the
shear rotation toward the triangle centroid. The TATT centroid phase
passes that test; the conjugate phase found on upstream main is retained
only as a named comparison convention.

Two aperture routes answer different questions. `aperture_full` computes

```math
\langle M_1M_2M_3\rangle
=\int\frac{\ell_1d\ell_1\,\ell_2d\ell_2\,d\varphi}{(2\pi)^3}
B_\kappa^{ijk}(\ell_1,\ell_2,|\boldsymbol\ell_1+\boldsymbol\ell_2|)
U_{R_1}(\ell_1)U_{R_2}(\ell_2)U_{R_3}(\ell_3).
```

Its Fourier bounds are numerical truncations to refine. The finite SAS
route instead integrates shear correlations over the specified measured
triangle bins. Its missing triangles are part of the observable
definition. The following section explains those weights and conventions.

## From galaxy shapes to aperture mass <a name="aperture"></a>

Weak lensing describes a small coherent distortion of galaxy images by a
complex shear $`\gamma=\gamma_1+i\gamma_2`$. Rotating the coordinate axes
changes its phase by twice the rotation angle; this is what a spin-two
field means. A three-point shear correlation therefore needs a specified
direction at each triangle vertex. The four natural components contain
the product of three rotated shears, with either no conjugation or one
conjugated shear at each of the three vertices.

Aperture mass averages projected matter fluctuations with a circular
filter whose integral is zero. A constant foreground sheet then gives
zero signal. For the Gaussian compensated filter used here, its Fourier
weight at angular multipole $`\ell`$ and radius $`R`$ is

```math
\widetilde U_R(\ell)=\frac{(\ell R)^2}{2}
\exp\left[-\frac{(\ell R)^2}{2}\right].
```

The observable returned is the E-mode third moment
$`\langle M_{\rm ap}(R_1)M_{\rm ap}(R_2)M_{\rm ap}(R_3)\rangle`$.
It can be calculated by integrating the natural
components with four complex kernels. The implementation uses the
centroid convention of the published aperture kernels: each shear is
rotated relative to the direction between its vertex and the triangle
centroid. Supplying components in another projection changes the answer
even when every array dimension matches.

The finite grid describes a triangle by two side lengths $`d_2,d_3`$ and
the angle $`\phi`$ between them. SAS means side–angle–side. In units of
$`R_1`$, write $`s=d_2/R_1`$ and $`t=(d_3/R_1)e^{i\phi}`$. For logarithmic
side bins of common width $`\Delta\ln d`$ and linear angle bins of width
$`\Delta\phi`$, the integration measure is

```math
\frac{s^2\lvert t\rvert^2}{2\pi}
(\Delta\ln d)^2\Delta\phi.
```

The powers of the side lengths arise because a radial area element is
$`s\,ds=s^2d\ln s`$. The supplied grids contain bin representatives;
this is a finite quadrature, not an exact bin average of the correlation.
The two side grids can have different numbers of bins but use the same
logarithmic width. Angles must lie strictly between zero and $`\pi`$;
collinear centroid projections are excluded.

For each output radius triplet, the prepared weights include this measure,
the sum with radii two and three exchanged, and the final factor one
quarter. The radius exchange implements the mirrored-orientation
convention used by the inspected Y6 aperture calculation. If $`W`$ denotes
those complete weights, the C contraction is simply

```math
\langle M_1M_2M_3\rangle_{\rm grid}
=\operatorname{Re}\sum_{\alpha=0}^3\sum_b W_{\alpha b}\Gamma_{\alpha b}
=\sum_{\alpha,b}
\left[\operatorname{Re}W_{\alpha b}\operatorname{Re}\Gamma_{\alpha b}
-\operatorname{Im}W_{\alpha b}\operatorname{Im}\Gamma_{\alpha b}\right].
```

Here $`b`$ visits measured triangle bins and $`\alpha`$ selects the natural
component. There is no complex conjugation in this multiplication. Do not
apply another factor one quarter after calling the contraction.

The adapter expects components already in this centroid and component-order
convention. In Rafael Gomes's inspected Y6 backend, the corresponding
conversion from the returned fastnc arrays is
`-[Gamma0, Gamma1, Gamma3, Gamma2]`. The C routine does not perform that
conversion or infer the convention from the data. The higher-level
`TransformPlan.aperture_from_multipoles` method performs the declared
conversion before calling that same C contraction. Do not apply it again.

Finite triangle coverage can differ from the unrestricted aperture
integral and can cause E/B leakage. The routine returns the stated real
combination, not the full set of eight E/B aperture combinations.
Agreement with the upstream
finite weights verifies this numerical operation; it does not remove
those physical distinctions or establish a matching measured covariance.

## What the covariance components mean <a name="covariance"></a>

The full physical engine and its six output families are described in
[COVARIANCE.md](COVARIANCE.md). The lower-level routines below remain
available for independent tests and user-supplied physical models. They
describe the mathematical operations used within covariance calculations;
their inputs must still refer to the intended estimator.

A filtered field $`X_a`$ is labeled by its source sample and filter.
Its pair correlation at centre separation $`r`$ is

```math
K_{ab}(r)=\int_0^\infty\frac{\ell\,d\ell}{2\pi}
\widetilde U_a(\ell)\widetilde U_b(\ell)
C_{ab}(\ell)J_0(\ell r).
```

$`C_{ab}`$ is the angular cross-power, including whichever noise model is
intended, and $`J_0`$ is the order-zero Bessel function from the angular
Fourier integral. The caller supplies the sampled filters and a quadrature
weight for $`d\ell`$. The C function inserts $`\ell/(2\pi)`$ once. A
logarithmic integration rule must first convert its weights using
$`d\ell=\ell\,d\ln\ell`$.

For zero-mean Gaussian fields, the mean of a product of three fields is
zero. The covariance of two such products is their six-field moment.
Wick's theorem expresses that moment as fifteen pairings. Six pairings
join every field at one centre to a field at the other centre. Nine
pairings contain one connection between centres and one local pair at
each centre. The local pairs use $`K(0)`$, which is supplied separately
even if the sampled radial grid does not contain zero.

For three copies of one field, this becomes

```math
C^{\rm G}_{333}=\int d\mu(r)
\left[6K(r)^3+9K(0)^2K(r)\right].
```

The weights $`d\mu`$ encode the averaging of aperture centres. In a
homogeneous large area $`A`$, use $`2\pi r\,dr/A`$; a specified finite
centre window changes those weights. The routine receives the complete
weights and does not divide by area again. Both pair-correlation inputs
must be symmetric in their field labels; the adapters allow only relative
roundoff differences up to 64 machine epsilons, with no absolute floor. An internal galaxy mask or
distinct-object estimator requires its own contractions. Do not remove
the nine local-pair terms merely because another estimator excludes
self-pairs; establish the estimator's noise rules first.

The connected response component has a different mathematical input. If
$`H_I(\lambda)`$ is observable $`I`$'s response to one halo or one declared
long-mode source, and $`w_\lambda`$ includes its nonnegative integration
measure and normalization, the returned matrix is

```math
C_{IJ}=\sum_\lambda w_\lambda H_I(\lambda)H_J(\lambda).
```

For a single-halo interpretation, both observables must refer to the same
halo properties at each node. This is a Gram matrix: every variance
$`v^TCv`$ is a sum of squares with nonnegative weights. Individual cross
entries can be negative and the matrix can be singular. Neither case is
repaired by clipping eigenvalues or adding artificial diagonal terms.
The routine does not determine the responses or prove that the sum is
the complete connected covariance.

## Files and the two C++ adapters <a name="files"></a>

| File | Responsibility |
|---|---|
| [physics_3pcf.c](physics_3pcf.c), [header](physics_3pcf.h) | Owned physical tables, BiHalofit, TATT/velocity shear, projection and full-filter aperture mean |
| [physics_grid_3pcf.c](physics_grid_3pcf.c), [header](physics_grid_3pcf.h) | Shared source-triplet preparation, uniform triangle/multipole tables and SIMD linear reads |
| [coupling_3pcf.c](coupling_3pcf.c), [fftlog_3pcf.c](fftlog_3pcf.c), [transform_3pcf.c](transform_3pcf.c) | Angular coupling, two-Bessel transforms and natural-component reconstruction |
| [aperture_3pcf.c](aperture_3pcf.c), [header](aperture_3pcf.h) | Finite aperture geometry and contractions |
| [covariance_3pcf.c](covariance_3pcf.c), [header](covariance_3pcf.h) | Filtered pairs, Gaussian map contractions and response matrices |
| [covariance_physics_3pcf.c](covariance_physics_3pcf.c), [header](covariance_physics_3pcf.h) | Halo preparation and physical joint covariance, including actual xi-bin spin weights |
| [threepcf_interface.cpp](threepcf_interface.cpp), [header](threepcf_interface.hpp) | Production NumPy adapter with borrowed contiguous inputs |
| [threepcf_wrapper.cpp](threepcf_wrapper.cpp), [header](threepcf_wrapper.hpp) | Notebook adapter with owned Armadillo arrays |
| `physics_*`, `transform_*`, `covariance_*` C++ adapter pairs | The same production/notebook ownership distinction for the physical engines |
| [CREDITS.md](CREDITS.md) | Scientific attribution, exact source revisions and license scope |

The project registers these adapters under `production` and `notebook`.
Both call the same C functions. The production path requires C-contiguous
NumPy `float64` arrays and `int32` index arrays with naturally aligned
addresses; it refuses implicit
conversion and allocates NumPy-owned outputs. It is intended for repeated
prediction and covariance work. The notebook path copies values into
Armadillo storage for inspection and convenience, then calls the same C
calculation. These roles follow the [core interface guide](../README.md#interfaces).

## Arrays, units and reuse <a name="arrays"></a>

Every angle and aperture radius is in radians. Angular spectra use the
normalization in the displayed pair integral. The C code reads packed
row-major arrays: the last index changes fastest. Real and imaginary
parts are separate arrays, each with the same dimensions.

| Quantity | Shape | Meaning of indices |
|---|---|---|
| Radius triplets | `[nfilter, 3]` | Aperture output, then ordered radius |
| Aperture weights | `[nfilter, 4, ntriangle]` | Output, component, triangle |
| Natural components | `[nsample, 4, ntriangle]` | Model/source-triplet sample, component, triangle |
| Aperture output | `[nsample, nfilter]` | Input sample and aperture output |
| Fourier filters | `[nfield, nell]` | Field/filter label and multipole node |
| Angular power | `[nfield, nfield, nell]` | Ordered pair of fields and multipole |
| Pair correlation | `[nfield, nfield, nr]` | Ordered fields and centre separation |
| Cubic field labels | `[nrow, 3]` | Observable row and its three field indices |
| Responses | `[nrow, nnode]` | Observable row and shared integration node |

The triangle count is `len(d2)*len(d3)*len(phi)`. Its flat index visits
`phi` first, then `d3`, then `d2`. Reshaping an array with another triangle
order gives the wrong integral without necessarily producing an error.

Prepare aperture weights once and retain the returned arrays while changing
the cosmology or source triplet. The C routines have no hidden cosmology
state. Their caller owns the persistent
weights and all outputs. Physical models and transform plans own their
prepared arrays explicitly. Changing the coordinates, radii, widths, triangle
selection or component convention requires new weights even when the
array dimensions are unchanged. Rebuilding the owning model or plan makes
the invalidation decision visible rather than leaving an incomplete
process-global cache key. A transform plan is not reentrant and its
worker capacity is fixed when it is constructed.

The lower-level contraction accepts supplied weights. Their geometry and
convention cannot be recovered from their shape alone, so the driver must
keep those definitions with the weights. Workers read inputs only; the
caller must not mutate them during a calculation.

## Parallel work and numerical checks <a name="checks"></a>

OpenMP assigns complete output entries to workers. SIMD applies the same
arithmetic to independent output entries in vector lanes; a lane is one
position in a small vector of numbers. Each output retains an ordered
sum over its integration nodes. The code comments identify lane meanings,
tail entries and ownership beside the loops. No cosmological setters or
FFTW planners are called from these workers.

The [numerical guide](NUMERICS.md) explains the preparation and repeated
read stages. Expensive physical calculations use coarse nodes; natural
cubic interpolation fills dense uniform tables; repeated reads use linear
interpolation with prepared indices and fractions. Fourier, angular and
radial quadratures use fixed nodes. Their refinement is separate from the
table-resolution controls. Source-reference modes preserve the pinned
interpolation sequence for comparisons and are explicitly selected.

`PhysicsModel.projection_grids` shares matter and intrinsic-shape work
across a batch of ordered source triplets. The project limits the number
of dense tables held simultaneously. Pure NLA omits the identically zero
residual table. NLA angular-power arrays likewise prepare source kernels
once and read neighboring multipoles with SIMD using the logarithmic
wave-number shift. These are reductions in repeated work, with no timing
claim attached.

The component tests compare aperture kernels and physical bispectra with
pinned external Python calculations; angular coupling with its defining
integral; two-Bessel transforms and shear phases with independent
references; NFW profiles with a direct radial integral; and Gaussian,
PB, BB and halo covariance factors with analytic or separately assembled
results. They also check the
two adapters and repeated calls with different worker counts. The
project's [test guide](../../../../../projects/des_3pcf/tests/README.md)
contains the commands and scope; its validation record reports executed
checks. Component agreement does not establish survey-scale numerical
convergence or a performance gain.

## Credits and scientific validation <a name="credits"></a>

This work builds on Sunao Sugiyama's fastnc, Rafael Gomes's Y3 and Y6
three-point pipelines, Mike Jarvis's TreeCorr aperture implementation,
and the aperture formulas of Jarvis, Bernstein and Jain and of Schneider,
Kilbinger and Lombardi. The two-Bessel transform follows Xiao Fang's
FFTLog work and Sunao Sugiyama's modifications. The analytic covariance draws on the work
of Linke, Heydenreich, Burger, Schneider, Wielders and Porth. Full source
links, papers and retained notices are in [CREDITS.md](CREDITS.md).

The project implements fixed-covariance Fisher calculations and
precision-matrix expansion using the full joint covariance. Their physical
interpretation still requires numerical refinement, a matched estimator
and explicit priors. The downloaded Y3 inputs are pinned candidates for
published-analysis comparisons. The inspected Y6 repository provides its
pipeline but keeps measured vectors, joint covariance and trained networks
external. A forecast configuration must therefore identify its source
curves and noise inputs rather than silently borrowing Y3 values.
