# Physical covariance of aperture moments and shear correlations

The physical covariance entry point predicts fluctuations of specified map
estimators from a matter power spectrum, matter bispectrum, source
distributions and a halo model. It constructs the halo responses itself.
The earlier `connected_gram_3pcf` routine only contracts responses supplied
by its caller; these are different levels of calculation.

The returned rows can contain aperture variance, aperture third moments,
and the full selected set of binned $`\xi_+`$ and $`\xi_-`$ measurements.
The two-point rows are not replaced by aperture variances. Their separation
nodes and normalized bin weights enter the shear-spin kernels directly.
Both `production.physical_covariance` and `notebook.physical_covariance`
call [the same C implementation](covariance_physics_3pcf.c).

## Which field and estimator are calculated?

An aperture field $`M_{i,R}(\boldsymbol x)`$ is the convergence for source
sample $`i`$, smoothed around center $`\boldsymbol x`$ with a compensated
circular filter. Compensation means that a constant convergence gives zero.
The Fourier filter is

```math
U_R(\ell)=\frac{\ell^2R^2}{2}\exp(-\ell^2R^2/2).
```

An aperture row averages two or three of these fields at the same center.
For a shear-correlation row the center is instead the midpoint of two
positions separated by $`\theta`$. Their shear product is averaged over
orientation and over the supplied separation-bin weights. A common survey
window weights the estimator centers. The underlying fields remain
stationary: their correlations depend on relative positions.

This is the full-filter map estimator. A finite center window is supported
for a circle or square. A mask that removes galaxies *inside* an aperture,
finite triangle selection, distinct-galaxy exclusions, and a random
catalogue-dependent denominator change the estimator. They are not obtained
by changing its area or by supplying arrays with matching dimensions.
`estimator=0` selects the supported map definition; other values are refused.
In particular, a finite-triangle aperture mean must not be paired with this
covariance without a separate estimator comparison.

The signal field is pure gravitational lensing. Gaussian shape noise is
specified by a source cross-power matrix $`N^{ij}`$, with the same noise
power in each of its independent E and B components. For disjoint samples,
$`N^{ii}=\sigma_{e,\mathrm{component}}^2/n_i`$, with $`n_i`$ per steradian.
If dispersion is supplied for both ellipticity components together, divide
its squared value by two first. Noise enters pair correlations once. It
does not enter the matter bispectrum or connected halo cumulants.

`ia_model=0` selects this physical model. Nonlinear TATT or velocity-shear
means do not determine their connected four-, five-, or six-field
correlations. Other IA model codes are refused. A forecast that adopts
this covariance while varying an IA mean therefore makes an explicit
fixed, lensing-only covariance approximation.

## The six returned covariance families

The result is `(mean, terms)`, with shapes `[nrow]` and `[6,nrow,nrow]`.
The first axis of `terms` has the following order. A family contributes
only to the blocks listed; its other entries are zero.

| Index | Family | Two-point block | Two-/three-point cross block | Three-point block |
|---|---|---|---|---|
| 0 | Gaussian pairings | Two Wick pairings, with xi E/B spin factors | Zero for zero-mean Gaussian fields | Six all-cross and nine one-cross pairings |
| 1 | $`BB`$ | Zero | Zero | Nine products of connected three-field functions |
| 2 | $`P\,T_{1h}`$ | Zero | Zero | Nine cross-pair and six internal-pair placements |
| 3 | $`P\,B`$ | Zero | Six cross-pair and three internal-triple-pair placements | Zero |
| 4 | Connected one-halo | Four-field cumulant | Five-field cumulant | Six-field cumulant |
| 5 | Selected long-mode response | Abundance-response product | Abundance-response product | Selected squeezed two-halo contribution |

A *connected* correlation is what remains after products of lower-order
correlations have been subtracted. This distinction prevents, for example,
counting Gaussian pair products again inside a trispectrum. The partition
counts refer to field positions. They remain valid when several fields have
the same source bin and radius.

The aperture covariance construction follows
[Linke et al.](https://arxiv.org/abs/2212.04485); the aperture second-/third-order
cross-covariance follows [Wielders et al.](https://arxiv.org/abs/2509.20443).
The xi extension inserts the midpoint pair estimator's spin and separation
weights into those same cumulant partitions. The selected response product
is also applied to the two-point and cross blocks, consistently with the
declared biased-halo response model. It is a named extension of that response
construction, not a claim that every term was present in the upstream
aperture-cross calculation.

The `large_area` switch replaces short-mode center-window integrals by their
large-area limit. In that limit a compensated single aperture leg removes
the nine Gaussian one-cross terms, the six internal $`PT`$ terms, and the
three internal $`PB`$ terms. The finite-window route retains them. The
selected long-mode response always uses the declared finite survey window;
it is not set to zero when the short-mode approximation is selected.

The mean excludes the known additive noise bias of a quadratic row.
Subtracting a fixed noise mean leaves its covariance unchanged. No inverse
covariance, eigenvalue clipping or diagonal regularization is performed.
The one-halo and selected-response matrices are positive semidefinite in
exact arithmetic. An individual $`BB`$ or $`PT`$ family need not be.

## Projection, halo physics and normalization

The borrowed `PhysicsModel` supplies the power and BiHalofit bispectrum.
The physical lensing weight is $`W_i(\chi)=\chi(1+z)g_i(\chi)`$ in that
model's convention; $`g_i`$ already contains its lensing prefactor. The
unshifted flat-sky Limber projections are

```math
C^{ij}(\ell)=\int d\chi\,\frac{W_iW_j}{\chi^2}
P_{\rm nl}(\ell/\chi,z),\qquad
B^{ijk}(\ell_1,\ell_2,\ell_3)=
\int d\chi\,\frac{W_iW_jW_k}{\chi^4}
B_m(\ell_1/\chi,\ell_2/\chi,\ell_3/\chi,z).
```

Two projected bispectra in $`BB`$ have independent line-of-sight integrals.
Likewise, the projected power and trispectrum in $`PT`$ do not share one
forced redshift. The code prepares each projection before contracting them.

Halo abundances and bias use the Sheth–Tormen multiplicity and response.
The smoothing radius is $`R=(3M/4\pi\bar\rho)^{1/3}`$, with
$`\bar\rho=2.77536627\times10^{11}\Omega_m`$ in the units below. The code
integrates the linear power with a spherical top-hat to obtain both
$`\sigma^2(M,z)`$ and its mass derivative. It rejects an invalid variance
or nondecreasing variance slope instead of replacing the slope by its
absolute value. The parameters `st_a`, `st_p`, `st_norm` and `delta_c`
are explicit. `collapse_model=1` additionally uses
$`\delta_c[1+0.0123\log_{10}\Omega_m(z)]`$; zero retains the constant
threshold. These choices permit controlled comparisons of conventions.

Halos have mass $`M_{200m}`$: their enclosing mean density is 200 times the
cosmological mean matter density. The normalized NFW profile is truncated
at that radius. The named default concentration is the full-sample,
mean-density, redshift-dependent fit in
[Duffy et al., Table 1](https://articles.adsabs.harvard.edu/pdf/2008MNRAS.390L..64D):
$`c=10.14(M/(2\times10^{12}))^{-0.081}(1+z)^{-1.01}`$.
Its coefficients and pivot are explicit inputs. Applying this fit outside
its fitted mass/redshift range remains an extrapolation. A supplied
nonlinear power spectrum or BiHalofit baryon transfer does not also modify
the NFW halo prescription.

For one halo, the filtered angular profile is

```math
\widetilde\Phi_{i,R}(\ell\mid M,\chi)
=\frac{W_i(\chi)}{\chi^2}\frac{M}{\bar\rho}
u_{\rm NFW}(\ell/\chi\mid M,z)U_R(\ell).
```

The expected halo count per solid angle is
$`d\nu=\chi^2d\chi\,dM\,dn/dM`$. These two equations fix all distance
and mass powers. For a row $`I`$, evaluate its estimator on this one halo
to obtain a radial function $`f_I(u)`$. Aperture rows use products of
$`\Phi`$. Xi rows use the two displaced spin-two shear profiles, with the
actual separation weights. Write $`H_I=\int d^2u\,f_I(u)`$. The connected
large-area term is

```math
C^{1h}_{IJ}=\frac1A\int d\nu\,H_IH_J.
```

For a finite center window $`w`$, set
$`G(q)=|\widetilde w(q)|^2/A^2`$, angularly averaged for a square. The
corresponding expression is
$`\int d\nu\int q\,dq\,G(q)\widetilde f_I(q)
\widetilde f_J(q)/(2\pi)`$. The code calculates the profile products and
their transforms; it does not ask the caller for $`H`$.

The selected long-mode term uses
$`R_I(\chi)=\int dM\,(dn/dM)b_1H_I`$. Its remaining measure is
$`d\chi\,\chi^2R_IR_J\int q\,dq\,P_L(q/\chi,z)G(q)/(2\pi)`$.
This is the halo-abundance response. It omits growth, dilation, tidal,
halo-exclusion and nonlinear-IA responses. Adding another SSC prescription
requires checking which response terms overlap.

The halo prescription assumes one pressureless clustering matter field.
The shared matter provider's linear mode currently supplies
$`P_{L,0}(k)D^2(z)`$; the halo routine integrates that supplied function.
Its $`M_{200m}`$ radius and normalization use the total mean matter density.
This is distinct from the existing core's massive-neutrino prescription,
which uses cold-plus-baryon power for halo variance. A massive-neutrino
extension must distinguish that formation power from total-matter lensing
and supply a consistent profile prescription. Feeding total-matter power
into this one-field halo fit does not implement that extension. The raw
power-table API cannot infer neutrino content from array values; callers
must enforce the stated physical model when constructing those arrays.

## Xi kernels and shape noise

The public `xi_from_power` function calculates the bin-averaged mean
directly from supplied E/B spectra:

```math
\xi_{\pm,b}^{ij}=\int\frac{\ell d\ell}{2\pi}
\overline J_{0/4,b}(\ell)\,[C_E^{ij}(\ell)\pm C_B^{ij}(\ell)].
```

$`\overline J`$ is the supplied normalized average over separation nodes,
not an assumed bin center. The covariance includes B noise even when the
physical lensing signal has no B mode. In the large-area single-source
limit, xi auto-covariance contains both $`(C_E+N)^2`$ and $`N^2`$.
The B-noise contribution changes sign between a xi-plus and xi-minus row.
Finite-window kernels also retain the mixed E/B spin factors. Omitting
those factors would calculate a scalar-convergence pair statistic instead.

There is a separate white-noise resolution issue for xi. An ideal noise
field has a delta-function correlation: an exactly fixed separation has
no finite continuum variance until a separation bin or angular resolution
is specified. A finite sum of separation nodes approximates a bin only
over a finite multipole range. Keeping those nodes fixed while taking
`ell_max` arbitrarily high therefore does not converge the pure $`NN`$
variance. `noise_mode=0`, the default, evaluates the given finite rules.
With that mode, refine the bin quadrature together with the Fourier rule
and cutoff, or specify an actual band limit. The API does not infer exact
bin edges or measured pair areas from quadrature nodes.

`noise_mode=1` selects a continuous, uniformly area-weighted xi annulus.
It requires `large_area=1` and explicit `theta_edges[nrow,2]` in radians.
For a xi row the lower edge is positive, the upper edge exceeds it, and
every separation quadrature node lies inside these edges. Non-xi edge
entries are unused. This mode asserts that the continuous bin weight is
proportional to $`\theta\,d\theta`$; the project driver refuses it for custom
pair weights. The low-level API permits different valid quadrature rules
for that same continuous weight and cannot infer a survey's weighting from
arrays. Supplying edges and selecting this mode is the caller's physical
declaration, not a reconstruction of unknown catalogue weights.

Let $`\Omega_i=\pi(\theta_{i,\max}^2-\theta_{i,\min}^2)`$ be a flat-sky
annulus area, and $`\Omega_{ij}`$ the area of its overlap with bin $`j`$.
For two xi rows of the same kind, with source legs $`ab`$ and $`cd`$,

```math
C^{NN}_{ij}=\frac{2\Omega_{ij}}{A\Omega_i\Omega_j}
\left(N^{ac}N^{bd}+N^{ad}N^{bc}\right).
```

It vanishes for disjoint annuli and for xi-plus versus xi-minus. Two shear
components account for the leading factor of two; both catalogue pairings
remain in parentheses. An auto-correlation in one bin therefore gives
$`4N^2/(A\Omega_i)`$. The implementation forms the $`CC`$ and $`CN+NC`$
integrands directly, then adds this $`NN`$ once. It avoids cancellation from
subtracting a large numerical noise integral. The signal still uses the
supplied separation quadrature, which must resolve its finite ell range.
This is the flat-sky annulus counterpart of the existing core's
[pair-count noise calculation](../covariances/gaussian_cov.c). It does not
use measured galaxy-pair areas or claim a finite-mask extension. The mode
is refused for a finite center-window calculation; the other covariance
families retain their stated noise and filter definitions.

## Arrays, units and numerical controls

| Quantity | Unit or representation |
|---|---|
| $`\chi`$, halo radii | Mpc/$`h`$ |
| $`k`$, physical lensing weight $`W`$ | $`h`$/Mpc |
| Matter power / bispectrum | (Mpc/$`h`$)$`^3`$ / (Mpc/$`h`$)$`^6`$ |
| Halo mass | $`M_\odot/h`$ |
| $`dn/dM`$ | (Mpc/$`h`$)$`^{-3}`$ ($`M_\odot/h`$)$`^{-1}`$ |
| Aperture radii, separations, opening angles | Radians |
| Survey area | Steradians |
| `field_source`, `row_fields` | Zero-based `int32` labels |

All arrays use C order and `float64`, except integer labels. The production
adapter borrows contiguous arrays without conversion. The notebook adapter
copies them through its owned Armadillo workspace. Both return owned arrays.

| Input grid | What it controls |
|---|---|
| `ell`, `dell` | Short-mode Fourier integrals; weights contain $`d\ell`$ |
| `window_ell`, `window_dell` | Independent survey-mode integrals in the connected finite-window and selected-response terms |
| `phi`, `dphi` | Relative Fourier angles, square-window orientation average, and xi endpoint orientation |
| `radius`, `dr` | Halo-offset product integrals and radial center-separation contractions |
| `profile_radius`, `profile_upsample` | Coarse shear preparation nodes and fine uniform lookup resolution for displaced xi endpoints |
| `k`, `dk` | Linear-power integral for halo variance and its slope |
| `mass`, `dmass` | Halo mass integral; weights contain $`dM`$, not $`d\ln M`$ |
| `theta`, `theta_weight` | Actual xi separation-bin quadrature, one normalized row per xi measurement |
| `theta_edges` | Optional exact `[nrow,2]` annulus edges, required by analytic xi noise mode |

The low-$`q`$ window grid must reach the survey scale. Using an ordinary
short-mode lower bound such as $`\ell=10`$ for that integral can omit the
long modes that produce the selected response. `profile_radius` starts at
zero, contain at least three increasing nodes, and must extend beyond
`max(radius)+max(theta)/2`. These are preparation samples rather than
integration weights. At each halo the code forms a natural cubic spline in
$`x=\ln(1+r/r_s)`$, where $`r_s=\texttt{profile_radius[1]}`$. This choice
includes zero and resolves a wide angular range. It samples that spline on
a uniform $`x`$ grid with
`(nprofile - 1) * profile_upsample + 1` nodes. The default upsample factor is
8; values must be positive integers. Each displaced endpoint's cell and
fraction are prepared once, then each halo uses linear interpolation with
those fixed indices. The spline sampling error and subsequent linear-read
error are distinct: increasing only `profile_upsample` cannot repair an
under-resolved coarse profile. Refine both.

There is another numerical constraint: the finite-window Gaussian two-point
and $`PB`$ integrals use independent Fourier vectors. For a large footprint,
$`G(|\boldsymbol p+\boldsymbol q|)`$ concentrates near opposite vectors.
Their radial and relative-angle rules must resolve this narrow region.
Refining the separate low-$`q`$ window grid does not resolve that region.
Likewise, the real-space Gaussian cubic term requires adequate center
separation coverage and oscillatory-transform accuracy. The small tests
are not evidence of convergence for a DES footprint. Refine these controls
and inspect the final covariance modes and marginalized errors before using
the result for an absolute Fisher forecast.

## A small dictionary request

The following is the request used in the component checks, reduced here to
its array construction. It assumes a `PhysicsModel` named `model` with two
source bins. These small node counts demonstrate the API; they are not
forecast accuracy settings. The complete controlled-model constructor is
in the project's tracked
[physical covariance tests](../../../../../projects/des_3pcf/tests/covariance/test_physical_covariance.py).

```python
import numpy as np
from numpy.polynomial.legendre import leggauss
import cosmolike_des_3pcf_interface as compiled

def log_rule(lower, upper, count):
    nodes, weights = leggauss(deg=count)
    half = 0.5 * np.log(upper / lower)
    values = np.exp(0.5 * np.log(upper * lower) + half * nodes)
    return values, weights * half * values

ell, dell = log_rule(10.0, 8000.0, 8)
window, dwindow = log_rule(0.02, 300.0, 28)
radius, dr = log_rule(1.0e-6, 0.03, 32)
wave, dwave = log_rule(1.0e-5, 500.0, 192)
mass, dmass = log_rule(1.0e11, 2.0e14, 3)
phi = (np.arange(8) + 0.5) * (2.0 * np.pi / 8)
inputs = {
    "ell": ell,
    "dell": dell,
    "window_ell": window,
    "window_dell": dwindow,
    "radius": radius,
    "dr": dr,
    "profile_radius": np.concatenate(([0.0], np.geomspace(1e-7, 0.04, 96))),
    "profile_upsample": 8,
    "k": wave,
    "dk": dwave,
    "mass": mass,
    "dmass": dmass,
    "phi": phi,
    "dphi": np.full(8, 2.0 * np.pi / 8),
    "field_source": np.asarray([0, 1, 0, 1], dtype=np.int32),
    "field_radius": np.asarray([0.002, 0.003, 0.0, 0.0]),
    "row_kind": np.asarray([1, 2, 3, 0], dtype=np.int32),
    "row_fields": np.asarray(
        [[0, 0, 1], [2, 3, 0], [2, 3, 0], [0, 1, 0]], dtype=np.int32),
    "theta": np.asarray([[0, 0], [.002, .003], [.002, .003], [0, 0.]]),
    "theta_weight": np.asarray([[0, 0], [.25, .75], [.25, .75], [0, 0.]]),
    "noise": np.diag([2.0e-9, 3.0e-9]),
    "area": 0.2,
    "center_window": 1,
    "large_area": 1,
    "estimator": 0,
    "ia_model": 0,
    "noise_mode": 0,
    "st_a": .707,
    "st_p": .3,
    "st_norm": .3222,
    "delta_c": 1.686,
    "collapse_model": 0,
    "concentration_a": 10.14,
    "concentration_b": -.081,
    "concentration_c": -1.01,
    "concentration_pivot": 2.0e12,
    "baryon_strength": 0.0,
}
mean, terms = compiled.production.physical_covariance(model=model, inputs=inputs)
covariance = terms.sum(axis=0)
```

`compiled` is the project's imported `cosmolike_des_3pcf_interface` module.
`halo_tables(model, inputs, z, k_profile)` exposes the physical halo tables
for inspection; its profile is normalized to one at zero wavenumber.
`xi_from_power` accepts `ell`, `dell`, `theta`, `theta_weight`,
`spectra_e`, `spectra_b`, and `sign`; signs `+1` and `-1` select xi-plus and
xi-minus. Its E/B spectra have shape `[nrow,nell]`.
The example keeps the finite-band noise reference because its arbitrary
two-node weights do not declare a continuous uniform annulus. The project
survey-row constructor retains actual bin edges and selects analytic xi
noise only with the declared uniform-area bin weighting.

## Reuse and validation

Within one call, the engine shares physical matter triangles across source
projections and filtered bispectra across $`BB`$ and $`PB`$. For the declared
scalar lensing model a halo profile factors as
$`\Phi_{i,R}=W_i\phi_R`$. The code therefore transforms each distinct
aperture radius once per halo, then multiplies its source weights. It also
transforms each needed unordered radius pair once. With four source bins
and five aperture radii, the pair-transform stage can require 15 distinct
radius pairs, compared with 400 ordered pairs of 20 labeled fields. This
counts transforms at that stage; it is not a measured total speed factor.
Pairs absent from the requested cubic rows are skipped. Aperture responses
with identical radius multisets, and xi responses with identical kind,
separation nodes and weights, share their geometric transform. Equality is
exact numerical equality of those inputs, without merging nearby radii or
different bin definitions. All source products are retained explicitly.

Bessel tables, window kernels, quadratic zero-mode kernels and xi endpoint
geometry are prepared before mass/redshift loops. The $`J_0(\ell r)`$
matrix is stored in both orientations so either Hankel transform reads
contiguous integration nodes. Xi bin kernels are also shared by identical
bins in `xi_from_power`; that API accepts power already sampled on the
integration nodes and performs no power interpolation. The covariance
assembly's $`P`$, $`B`$, $`S`$ and $`K`$ arrays are likewise read at prepared
quadrature nodes, without interpolation during covariance contractions.
The source model owns its own power/bispectrum interpolation preparation.
In the large-area route only the quadratic zero-mode kernel is allocated;
the finite two-vector geometry is absent. In the finite-window route its
storage is per distinct xi bin definition, not per source pair. A bin's
kind, every separation and every weight must agree exactly to share it.
Unequal-radius aperture pairs retain their ordered Fourier-leg assignment.
A request with no cubic rows skips bispectrum and aperture-partition tables
while retaining the full selected Gaussian and halo two-point terms.

Halo nodes are visited in a fixed order. Output rows, distinct profile
shapes and upper-triangle cells have exclusive worker ownership. The
ordered dot products, xi spectrum contraction and displaced-endpoint reads
use explicit two-lane SIMD. Lanes evaluate adjacent nodes; their values are
added in low-then-high order, and an odd final node is scalar. Allocations
occur before worker loops. No adaptive integration, lazy initialization or
shared reduction appears in this covariance calculation. The C routine
releases its working tables on return; it has no persistent cosmology cache
across separate calls. This lifetime matches an offline covariance call.

The tests compare NFW profiles to an independent radial mass integral,
halo variance to a different quadrature and Bessel representation, halo
bias to an abundance derivative, and xi transforms to analytic Gaussian
spectra. The radius-sharing test constructs every labeled field and pair
independently and compares the connected and $`PT`$ terms for finite and
large-area windows. A separate xi test evaluates $`J_2`$ directly at every
displaced endpoint and measures the error as the profile preparation grids
are refined. They also check the physical covariance's noise dependence, area
scaling, matrix symmetry, response-matrix positivity, adapters and worker
ownership. The project's validation record identifies the executed build
and tests. These are implementation checks. Agreement with an external
survey covariance, Fourier/radial convergence, and the accuracy of the
chosen halo model are separate scientific checks.

For the direct-endpoint xi test, the measured relative Frobenius-norm error
in the finite-window connected two-row xi block was as follows. The positive
preparation nodes span $`10^{-7}`$ to $`0.04`$ radians, with an additional
node at zero. The short-mode quadrature is fixed at nine nodes between
$`\ell=10`$ and 1500; other integrals are held fixed. The reference evaluates
$`J_2`$ at each displaced endpoint and has no shear-profile interpolation.

| Positive preparation nodes | Upsample factor | Relative block error |
|---:|---:|---:|
| 96 | 4 | 0.419795 |
| 192 | 8 | 0.0155093 |
| 384 | 16 | 0.000385945 |

These results measure this lookup approximation on a controlled, fixed
quadrature. They do not establish convergence of those other integrals or
choose a production profile grid. In particular, a large upsample factor
cannot recover oscillations missed by the coarse preparation nodes.

## Changing settings without restarting Python

The covariance audit follows the existing core's
[state-invalidation rules](../../.claude/skills/cosmolike-dev/references/state_invalidation.md):
allocation sizes, value changes, reset behavior and a check that each test
actually reaches the affected calculation. The ownership differs from a
lazy global table. The covariance C functions have no persistent mutable
static arrays. Each physical call constructs its workspace and profile
lookup plan from the current request, uses them, and releases them. The
borrowed matter model owns immutable physical tables. Several models may
remain alive simultaneously, and a covariance call names its model
explicitly. There is no covariance nonce or hidden model-selection latch
to reset with `initial_setup`.

The production adapter retains references to the caller's contiguous
arrays for the duration of one call. The notebook adapter copies arrays
into its own request, including optional exact xi edges. Neither request
is retained after the call. Both adapters return newly owned NumPy arrays.
These guarantees concern mutations between completed calls; they do not
authorize modifying borrowed production arrays concurrently with a running
native calculation.

| Existing core failure class | Covariance ownership rule | Sequence check |
|---|---|---|
| An allocation misses a changed size | Workspace extents come from the current request on every call | Joint requests grow and shrink fields, rows, Fourier/real/window/profile/mass/angle/variance grids and xi-bin node counts; newly added fields occur in added rows |
| A refill misses a changed value | Every call reads current values; buffer identity is not a cache key | One setting at a time changes in place while dimensions and NumPy identities stay fixed, followed by restoration and reversed sequence order |
| A model flag or singleton survives reset | The caller names an immutable model owner; covariance has no selected-model singleton | Two models with different source counts and preparation sizes are interleaved A/B/A without restarting Python |
| A test never reaches the relevant calculation | The test must prove a setting changes an output on its intended route | Profile controls are changed in finite-window mode, window shape is checked in the selected-response term, and exact bin edges are checked with analytic NN enabled |

The project's
[covariance setting-sequence tests](../../../../../projects/des_3pcf/tests/covariance/test_setting_sequences.py)
passed eight parametrized tests through both adapters. They use 41 final
states and 82 independent fresh-process references: each reference process
constructs only its requested model and makes one covariance call. The
reused interpreter then evaluates the sequences and compares complete
`mean` and `terms` arrays by raw float64 bytes, including the sign of zero.
No numerical tolerance is used for these state comparisons.

The 36 same-size changes include each integration axis and weight array,
source/field/row labels, aperture radii, shape noise, exact xi edges, the
finite/large-area switch, both NN modes, profile upsampling, every exposed
ST/Duffy scalar, the evolving collapse threshold, the BiHalofit baryon
parameter, survey area and window shape. Each change is compared with a
reference differing in exactly one setting, so another changed key cannot
hide an incomplete invalidation rule. Additional transitions remove and
restore cubic rows, edit constructor arrays after model creation, retain
old output arrays across later calls, mutate returned arrays, use notebook
strided/column-major inputs, and switch between one and two OpenMP workers.
All tested transitions matched their fresh references bitwise. The audit
found no native covariance state defect requiring a code change.

These checks address state ownership and output reproducibility on small
rules. They do not measure performance, establish survey convergence, or
replace a separate sanitizer or repeated-construction leak study.
