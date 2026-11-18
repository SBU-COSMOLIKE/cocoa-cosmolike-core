# Three-point shear and aperture statistics

1. [What can be calculated?](#scope)
2. [From matter and galaxy shapes to angular statistics](#physics)
3. [Angular multipoles and the four shear components](#transform)
4. [From galaxy shapes to aperture mass](#aperture)
5. [What the covariance components mean](#covariance)
6. [A joint two- and three-point Fisher calculation](#fisher)
7. [Files and the two C++ adapters](#files)
8. [Arrays, units and reuse](#arrays)
9. [Preparation, interpolation and parallel work](#preparation)
10. [Changing a calculation without restarting Python](#state)
11. [Accuracy and independent checks](#checks)
12. [Acknowledgements](#credits)

## What can be calculated? <a name="scope"></a>

Weak gravitational lensing slightly distorts the images of distant
galaxies. Two galaxies can have correlated distortions because light from
both passes through some of the same foreground matter. A two-point
statistic measures that correlation as a function of their separation.
A three-point statistic asks how three distortions correlate as a function
of the triangle formed by the galaxies. Triangle shape matters: three
nearly aligned positions and an equilateral arrangement sample structure
in different ways.

A Gaussian random field is completely specified by its mean and two-point
correlation, and its zero-mean three-point
correlation vanishes. Gravitational evolution makes the matter field
non-Gaussian: dense structures grow and couple Fourier modes. The
three-point function measures part of that extra structure. Its parameter
dependence can differ from the two-point function, which can help separate
cosmology from uncertain galaxy-shape physics. The gain depends on their
shared covariance and on the model assumptions; additional array entries
alone do not establish additional independent information.

This directory predicts those means and a specified joint covariance. It
sits beside `covariances/`; existing data-vector C files remain unchanged.
Its physical inputs are matter power spectra, distances, growth and the
redshift distribution of each source-galaxy sample, together with the
cosmological and intrinsic-shape parameters. An optional CAMB producer
in the companion project calculates the power and background tables.
The C functions read those tables; they do not run a Boltzmann solver.

The following table maps the calculation. The next sections define its
physical quantities before introducing their storage and interfaces.

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

Readers interested in the observable can follow Sections 2–6 without
reading the C++ interfaces. To change the implementation, continue through
the ownership and numerical sections, then read the named source header
and its function comments. [NUMERICS.md](NUMERICS.md) gives the table and
integration rules. [COVARIANCE.md](COVARIANCE.md) gives the full physical
covariance construction and its estimator restrictions.

## From matter and galaxy shapes to angular statistics <a name="physics"></a>

### The matter field and its bispectrum

Write the fractional matter-density fluctuation as
$`\delta=(\rho-\bar\rho)/\bar\rho`$. Its power spectrum $`P(k,z)`$
describes pairs of Fourier modes. Its bispectrum $`B_m`$ describes triples:

```math
\left\langle\delta(\boldsymbol k_1,z)
\delta(\boldsymbol k_2,z)\delta(\boldsymbol k_3,z)\right\rangle
=(2\pi)^3\delta_D(\boldsymbol k_1+\boldsymbol k_2+\boldsymbol k_3)
B_m(k_1,k_2,k_3;z).
```

The Dirac delta enforces translation invariance: the three wavevectors
must close into a triangle. Their lengths $`k_1,k_2,k_3`$ therefore obey
the triangle inequalities. A larger $`k`$ means a smaller spatial scale.
BiHalofit supplies a calibrated nonlinear fit to this bispectrum. A
separate tree-level result supplies the lowest-order gravitational
perturbation-theory reference. Increasing a numerical resolution does not
make either physical approximation exact.

The model accepts $`k`$ in $`h/\mathrm{Mpc}`$, distances in
$`\mathrm{Mpc}/h`$, power in $`(\mathrm{Mpc}/h)^3`$ and bispectra in
$`(\mathrm{Mpc}/h)^6`$. Here $`h=H_0/(100\,\mathrm{km\,s^{-1}Mpc^{-1}})`$
specifies the present expansion rate. Growth $`D(z)`$ describes the linear
fluctuation amplitude, normalized by $`D(0)=1`$.
The ordinary core's growth table stores a quantity proportional to
$`D/a`$ instead, where $`a=1/(1+z)`$ is scale factor; the project bridge
makes that conversion explicitly.
Source arrays are probability densities per unit redshift. Their
normalization is calculated from the supplied grids. A normalized source
curve describes where galaxies lie, not how many there are. The source
number density needed for shape noise is a separate survey input.

### Projecting along the line of sight

The model uses spatially flat lensing geometry. It requires
$`|\Omega_m+\Omega_{\rm de}-1|\leq10^{-8}`$ as a scalar-rounding
tolerance; it does not change either supplied value. These parameters
are the present matter and dark-energy density fractions. Supplying a curved
distance table would not replace the flat distance ratios in the lensing
kernel. The project CAMB producer explicitly requests flat geometry.

Convergence $`\kappa`$ is the scalar lensing field obtained by weighting
matter fluctuations along the line of sight. Its angular variations also
produce shear. For gravitational lensing, the flat-sky Limber projection is

```math
B_\kappa^{ijk}(\ell_1,\ell_2,\ell_3)
=\int d\chi\,\frac{W_i(\chi)W_j(\chi)W_k(\chi)}{\chi^4}
B_m(\ell_1/\chi,\ell_2/\chi,\ell_3/\chi;z(\chi)).
```

Here $`W_i`$ is the usual lensing efficiency, with units of inverse
distance, and $`\chi`$ is comoving radial distance. A source sample has a
broad redshift distribution, so a single angular triangle receives
contributions from many foreground distances. $`W_i`$ weights each
distance by the amount of sample $`i`$ behind it and by the lens–source
geometry. The three efficiencies appear because the same foreground
fluctuation lenses all three source samples. The source labels
$`i,j,k`$ specify a tomographic triplet: tomography means dividing the
galaxies into samples selected by estimated redshift.

Limber assigns a transverse spatial wave number $`k=\ell/\chi`$ to an
angular multipole $`\ell`$; larger multipoles describe finer angular
structure. The same angular mode probes smaller
physical structures at a smaller distance. A units check gives
$`d\chi\,W_iW_jW_k\chi^{-4}B_m`$ units
$`L\,L^{-3}L^{-4}L^6=1`$, as required for a correlation of
dimensionless lensing fields. The implementation preserves the source
convention with no $`\ell+1/2`$ substitution. Its internal `radial()`
arrays named `lens` and `shape` use fastnc's rescaled kernel convention;
they are not bare $`W_i`$. The covariance code converts those arrays before
using the physical powers of distance.

### Intrinsic shapes and ordered source labels

The observed shape contains both the lensing distortion $`G`$ and an
intrinsic contribution $`I`$. The latter can correlate with the tidal
field where the galaxy formed. Expanding a three-shape product gives
$`GGG`$, three placements of $`GGI`$, three placements of $`GII`$ and
$`III`$. A lensing leg weights foreground matter through $`W_i`$; an
intrinsic leg instead selects the source population at the local
distance. Exchanging these legs changes their radial weights.

TATT means tidal alignment and tidal torquing. It includes a response
linear in the tidal field and responses quadratic in that field, with
a density-weighting term. The latter accounts for sampling shapes at
galaxy positions rather than uniformly through space. The optional
velocity-shear contribution is
another declared intrinsic-shape response. The nineteen active kernels
returned by the physical engine are the chosen perturbative components
used to assemble these correlations; they are not nineteen independently
measured data vectors. Their amplitudes and redshift laws must be kept
with their Fourier legs and source labels.

The nonlinear-alignment (NLA) baseline keeps a shape response linear in
the tidal field while using the selected nonlinear matter correlations.
Its linear response amplitude is denoted $`C_1`$. This deterministic part
replaces each lensing efficiency by its lensing-plus-linear-alignment
value. Residual TATT terms are then added
without counting that baseline again. The upstream baseline and residual
source kernels have different normalization/interpolation sequences.
They remain separate here: combining their continuous formulas does not
prove their finite numerical grids are equivalent. The IA power choice is
also explicit: `power_mode=0` uses linear $`P(k,0)D^2`$, `1` uses the
upstream nonlinear $`P(k,0)D^2`$ prescription, and `2` uses the supplied
evolving nonlinear table. `ct_mode=0` defines the physical velocity-shear
coefficient as $`C_t=C_1\,ct`$; `1` takes `ct` itself as that coefficient.

The mean's selected three-point IA model predicts E-mode correlations.
It does not supply every possible intrinsic-shape B-mode bispectrum.
The companion joint mean obtains two-point TATT E/B spectra from the
ordinary CFASTPT routines, which evaluate the perturbative convolution
integrals for that two-point model. Those two statements concern different
orders of correlation and must not be interchanged. Likewise, the
BiHalofit baryon-response option changes a specified matter bispectrum;
it does not by itself define a two-point feedback spectrum or modify the
halo profiles used in the covariance.

## Angular multipoles and the four shear components <a name="transform"></a>

The angular bispectrum still depends on the size and shape of a Fourier
triangle. For fixed $`\ell_1,\ell_2`$, the source convention defines
$`\mu`$ by $`\ell_3^2=\ell_1^2+\ell_2^2-2\ell_1\ell_2\mu`$.
Legendre polynomials $`P_L`$ are an orthogonal basis on $`[-1,1]`$.
Expanding in that basis separates the shape dependence:

```math
B^{ijk}(\ell_1,\ell_2,\mu)
\simeq\sum_{L=0}^{L_{\max}}b_L^{ijk}(\ell_1,\ell_2)P_L(\mu),
\qquad
b_L^{ijk}=\frac{2L+1}{2}\int_{-1}^{1}d\mu\,B^{ijk}P_L(\mu).
```

$`L`$ is a basis order for triangle shape, whereas $`\ell_1,\ell_2`$
are Fourier side lengths. Truncating at $`L_{\max}`$ is a numerical
approximation to refine. The minus sign in the definition of $`\mu`$
also fixes the $`(-1)^L`$ factor in the angular coupling. It cannot be
changed independently of that coupling.

Shear has two components, conveniently written
$`\gamma=\gamma_1+i\gamma_2`$. Rotating the axes by an angle changes
its phase by twice that angle; this is the meaning of a spin-two field.
Let $`g_a`$ denote the shear at vertex $`a`$ after rotating it into a
specified triangle direction. The four natural components are

```math
\Gamma_0=\langle g_1g_2g_3\rangle,\qquad
\Gamma_1=\langle g_1^*g_2g_3\rangle,\qquad
\Gamma_2=\langle g_1g_2^*g_3\rangle,\qquad
\Gamma_3=\langle g_1g_2g_3^*\rangle.
```

They retain the orientation information needed to convert a shear
correlation into an aperture statistic. Both the complex phase and the
position of the conjugated leg matter. The same four arrays in another
rotation convention are different inputs to an aperture kernel.

The transform first couples the Legendre coefficients to angular Fourier
modes $`M`$. Two radial Bessel integrals then turn Fourier-side lengths
into angular separations. FFTLog evaluates those integrals using Fourier
transforms on logarithmic grids and analytic transforms of power laws.
Finally, the modes are summed with $`e^{iM\phi}`$ to recover the opening
angle $`\phi`$, and the shears are rotated into the requested projection.
The retained transform plan contains geometry, transform factors and
worker storage. Its input $`b_L`$ changes when the physical model changes.

Source labels and triangle sides must move together. With unequal source
bins, ordered TATT coefficients need not be symmetric under exchanging
the two Fourier sides. `ordered_legs=True` therefore evaluates both signs
of angular mode $`M`$ explicitly. `ordered_legs=False` preserves the
historical transpose shortcut for source comparisons. Independent
Cartesian Gaussian-moment tests check the general case as well as the
shear rotation toward the triangle centroid. The TATT centroid phase
passes that test; the conjugate phase found on upstream main is retained
only as a named comparison convention.

Logarithmic FFT grids have finite bounds and a periodic representation.
Padding and coefficient tapering control parts of the resulting numerical
error. The historical untapered operator and the corrected taper remain
explicit choices for source comparisons. A radial-bin average uses the
declared lower edges and logarithmic width. An angular-bin average with
positive `dphi` averages the angular Fourier modes before applying the
centroid phase at the bin centre; it is not the exact bin average of that
varying phase. These choices must remain visible when comparing codes.

## From galaxy shapes to aperture mass <a name="aperture"></a>

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
E modes have the shear pattern produced by a scalar lensing potential;
B modes describe the independent rotated pattern. The leading scalar
lensing model used here has no B-mode signal, although random shape noise
has both components. These statements do not remove the complex phases
of a three-point shear correlation.

### Full filters and measured triangle coverage

There are two routes to this moment. The full-filter route works directly
with the projected bispectrum:

```math
\langle M_1M_2M_3\rangle
=\int\frac{\ell_1d\ell_1\,\ell_2d\ell_2\,d\varphi}{(2\pi)^3}
B^{ijk}(\ell_1,\ell_2,|\boldsymbol\ell_1+\boldsymbol\ell_2|)
\widetilde U_{R_1}(\ell_1)\widetilde U_{R_2}(\ell_2)
\widetilde U_{R_3}(\ell_3).
```

Here $`M_a`$ is the aperture field of the source sample and radius on leg
$`a`$, and $`\varphi`$ is the angle between its two independent Fourier
vectors. The third vector closes the triangle. This formula uses the
usual vector sum; its $`\mu`$ in the Legendre convention above is
$`-\cos\varphi`$. The C `aperture_full` route needs no intermediate
$`b_L`$ or natural-component arrays. Its finite Fourier bounds are
numerical truncations of the unrestricted filter integral and must be
refined.

The finite-triangle route instead starts from shear correlations available
only for selected side and opening-angle bins. It integrates those natural
components with four complex kernels. The implementation uses the
centroid convention of the published aperture kernels: each shear is
rotated relative to the direction between its vertex and the triangle
centroid. Supplying components in another projection changes the answer
even when every array dimension matches.

For example, a measurement that excludes all triangles with a side below
one arcminute cannot be turned into the unrestricted aperture moment by
increasing the density of its remaining bins. The missing region changes
the estimator. This is why the full-filter analytic covariance and a
finite-triangle mean need an estimator comparison before they are used
together. An aperture radius is a smoothing scale; it is not an angular
cut on each side of a galaxy triangle.

The two paths share the physical bispectrum and separate at the observable:

```mermaid
flowchart TD
  A["Matter and intrinsic-shape bispectra"] --> B["Radial projection with source weights"]
  B --> C["Projected angular bispectrum"]
  C --> D["Integrate three circular Fourier filters"]
  D --> E["Full-filter aperture third moment"]
  C --> F["Legendre coefficients b_L"]
  F --> G["Angular coupling, FFTLog and shear rotation"]
  G --> H["Four natural components on side–angle–side bins"]
  H --> I["Integrate only the declared measured triangles"]
  I --> J["Finite-triangle aperture statistic"]
```

### The finite side–angle–side integral

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

A model mean predicts the average measurement over possible realizations
of the Universe. Its covariance measures the scatter around that mean:

```math
\mathcal C_{IJ}=\left\langle
(\hat d_I-\langle\hat d_I\rangle)
(\hat d_J-\langle\hat d_J\rangle)\right\rangle.
```

A hat marks a measurement. A row label $`I`$ includes the statistic, source
samples and angular bin or filter radii. A diagonal entry is a variance;
an off-diagonal entry records fluctuations shared by two measurements.
The same massive halo can affect several source samples and aperture
radii, and both a two-point and a three-point measurement. This physical
overlap is why their cross covariance is required.

### From products of fields to covariance families

The covariance of two quadratic measurements contains four fields; that
of a quadratic and a cubic measurement contains five; that of two cubic
measurements contains six. A connected correlation, or cumulant, is what
remains after subtracting every product of lower-order correlations. Thus
the four-field expectation contains Gaussian pair products and a connected
four-field correlation, called a trispectrum in Fourier space. The
six-field expectation also contains products of two bispectra and products
of a power spectrum with a trispectrum. This is a bookkeeping of distinct
physical correlations, not six alternative estimates of the same matrix.

The physical engine returns each of the following six families separately.
Here $`P`$, $`B`$ and $`T`$ denote power, bispectrum and connected
trispectrum ingredients after their required projections.

| Family | Physical content | Contributes to |
|---|---|---|
| Gaussian | Pair correlations of the signal and random shape noise | Two-point and three-point variances; the five-field cross block vanishes for zero-mean Gaussian fields |
| $`BB`$ | Products of two connected three-field correlations, after subtracting the product of the two means | Three-point covariance |
| $`PT_{1h}`$ | A pair correlation times the connected four-field correlation produced by one halo | Three-point covariance |
| $`PB`$ | A pair correlation times a connected three-field correlation | Two-/three-point cross covariance |
| Connected one-halo | Four, five or six fields produced by the same halo | Two-point, cross and three-point blocks |
| Selected long-mode response | A survey-scale density fluctuation changes the abundance of halos contributing to each observable | Two-point, cross and three-point blocks |

The nonlinear projected power and BiHalofit bispectrum are supplied by the
matter model. The connected four-, five- and six-field parts use the
declared halo prescription: Sheth–Tormen abundance and bias, truncated
Navarro–Frenk–White density profiles, and an explicit concentration–mass
relation. One-halo means that all fields in that connected contribution
come from one halo. It is an approximation to the full connected
correlation. For instance, a complete gravitational trispectrum also
contains contributions involving several halos.

The long-mode family models the change in halo abundance. It is a
specified part of super-sample covariance, the modulation of measurements
by fluctuations larger than the observed patch. Growth, dilation, tidal
and intrinsic-shape responses are not all supplied by that abundance
term. Calling it the complete SSC would overstate the model. Two projected
bispectra in $`BB`$ retain independent radial integrals; their source
weights cannot be forced to the same redshift merely to reduce the work.

The covariance signal is gravitational lensing, with Gaussian shape
noise. A TATT mean does not provide the missing intrinsic-shape four-,
five- and six-field correlations. Adopting this covariance with a TATT
mean is therefore an explicit fixed, lensing-only covariance approximation.
The gravitational `mean` returned alongside the covariance terms is useful
for inspecting that covariance model. It is not the joint TATT prediction
assembled by the separate mean engine. The current project covariance
also requires zero fiducial multiplicative shear calibration; calibration
can still vary in the Fisher mean while that covariance is held fixed.
The selected halo model also uses one pressureless matter fluid. It does
not implement the ordinary core's cold-plus-baryon halo-formation
prescription for massive neutrinos. The project rejects a nonzero mass
unless the single-fluid effective-growth approximation is selected
explicitly. [COVARIANCE.md](COVARIANCE.md) states these assumptions and
the exact normalization of each term.

### Survey windows and random shape noise

The supported estimator averages full filtered maps over aperture centres,
or shear pairs over their midpoints. A circle or square weights those
centres. This window is distinct from holes in a galaxy catalogue inside
an aperture, and from the cuts that remove data-vector rows. Changing its
area does not reproduce either kind of selection. The `large_area` option
uses the large-footprint limit for short-mode contractions; the selected
long-mode contribution still uses the specified finite survey window.

For disjoint source samples, the white-noise angular power is
$`N^{ii}=\sigma_{e,\mathrm{component}}^2/n_i`$, where $`n_i`$ is the
number per steradian and the dispersion describes one ellipticity
component. Noise enters the pair correlations once. It is not inserted
into the physical matter bispectrum or the connected halo cumulants.

An ideal white-noise field has a delta-function correlation. Its
two-point variance therefore needs the actual angular-bin area; a finite
list of displayed bin centres is insufficient. For large-area,
uniformly area-weighted $`\xi_\pm`$ bins, `noise_mode=1` calculates
the pure noise–noise term from the overlap of their exact annular edges.
It retains numerical quadrature for signal and mixed signal–noise terms.
`noise_mode=0` instead describes the supplied finite-band rule. The
annular correction is not assumed for custom pair weights or finite
centre windows. Both E and B shape-noise components enter the shear
correlation covariance even when the lensing signal is pure E.

### Pair functions and the one-halo response

The lower-level routines also accept arrays from an independently supplied
physical model. They expose useful intermediate operations and tests;
they do not infer an estimator or construct a halo population from the
array dimensions.

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
roundoff differences up to 64 machine epsilons, with no absolute floor.
For compensated apertures in the ideal large-area limit, the complete
separation integral of a single $`K(r)`$ vanishes because
$`\widetilde U(0)=0`$. The nine local-pair terms then cancel. A finite
centre window generally prevents that cancellation; the physical
`large_area` route and the finite-window route keep this distinction.
An internal galaxy mask or distinct-object estimator requires its own
contractions. Do not remove
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

For the physical one-halo calculation, $`\lambda`$ labels halo mass and
distance, and $`H_I`$ is the observable produced by that halo after
integrating over its angular position. The large-area result is

```math
\mathcal C^{1h}_{IJ}=\frac{1}{A}\int d\chi\,\chi^2
\int dM\,\frac{dn}{dM}(M,z)H_I(M,\chi)H_J(M,\chi).
```

$`A`$ is survey solid angle, and $`dn/dM`$ is the number of halos per
comoving volume per unit mass. The factor $`\chi^2d\chi`$ converts
volume into a count per solid angle. The physical engine calculates
these halo responses itself, including the displaced spin-two profiles
needed for a binned shear pair. This reduces the calculation to shared
profiles and their products under the stated one-halo model; it does not
omit a term from that model in exchange for a lower numerical cost.

## A joint two- and three-point Fisher calculation <a name="fisher"></a>

The companion project forms a vector containing the selected
$`\xi_+`$, $`\xi_-`$ and aperture third-moment rows. It keeps the
selected two-point measurements themselves; it does not replace them
with aperture variances or compress them implicitly. In block form,

```math
d=\begin{pmatrix}d_2\\d_3\end{pmatrix},\qquad
\mathcal C=\begin{pmatrix}
\mathcal C_{22}&\mathcal C_{23}\\
\mathcal C_{23}^{T}&\mathcal C_{33}
\end{pmatrix}.
```

Each covariance row must refer to exactly the same observable and order
as the corresponding mean entry. The off-diagonal block measures their
common fluctuations. Setting it to zero asserts independence and can
change the apparent improvement from adding three-point measurements.
Selecting a subset removes its rows and columns before inverting the
covariance. A sub-block of the inverse full matrix describes a different,
conditional problem.

Let $`m(p)`$ be the predicted mean and $`p_\alpha`$ a cosmological or
nuisance parameter. Near a fiducial model $`p_*`$, the fixed-covariance
Fisher calculation is

```math
D_{I\alpha}=\left.\frac{\partial m_I}{\partial p_\alpha}\right|_{p_*},
\qquad
F_{\alpha\beta}=D_{I\alpha}
(\mathcal C_*^{-1})_{IJ}D_{J\beta}+\Pi_{\alpha\beta}.
```

Repeated indices sum over the retained measurements. $`\Pi`$ is prior
precision: an independent Gaussian prior of width $`\sigma_\alpha`$
adds $`1/\sigma_\alpha^2`$ to its diagonal. A flat interval does not
become that Gaussian term. The approximation holds one fiducial
covariance $`\mathcal C_*`$ fixed while evaluating displaced means;
it does not include parameter derivatives of the covariance.

This division of work matters computationally. The covariance can be an
offline calculation performed once for the declared fiducial model. The
mean must be evaluated repeatedly for the finite-difference derivatives.
The project compares derivatives at steps $`h`$ and $`h/2`$, where
each parameter's step has its declared units. A step that is too large
mixes curvature into a local derivative. A step that is too small can
amplify interpolation or floating-point errors. More precise covariance
arithmetic cannot repair an unresolved derivative.

If the Fisher matrix is identifiable, $`F^{-1}`$ is the local parameter
covariance. Its diagonal gives marginalized variances: other parameters
are allowed to adjust. A two-parameter figure of merit can be defined as
$`1/\sqrt{\det[(F^{-1})_{ab}]}`$, with the parameter pair stated.
This is different from fixing every nuisance parameter. An unconstrained
direction signals a missing independent constraint or a locally inactive
parameter; it should not be hidden by adding an arbitrary diagonal.

To test an IA or baryon claim, use the same survey rows, covariance,
parameter basis and declared priors in the compared forecasts. Vary the
specified nuisance model, refine the numerical derivatives, and report
which physical parts of the covariance were held approximate. An increase
in a conditional figure of merit is not a demonstration that the analysis
can distinguish an omitted physical effect. Y3 and Y6 require their own
source curves, noise and measurement definitions; a repository name is
not a substitute for those inputs.

Optional precision-matrix expansion belongs to the project layer. It
combines an analytic covariance model with a supplied joint sample
covariance and a justified independent-realization count, or independent
sample groups. It does not create simulated information or make correlated
patches independent. Its statistical assumptions and finite-sample tests
are separate from the C covariance integral.

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

Command-line production calculations use the borrowing interface. This
avoids the notebook adapter's extra input conversion and temporary
Armadillo arrays; the immutable physical model still copies the inputs
it must retain after a call. Notebook inspection can return intermediate
halo tables, angular spectra or natural components, which also require
their own arrays. Request those diagnostics when they answer a question
about the model; they are not required merely to obtain a final vector.
Neither interface changes the underlying scientific formulas. The cost
of conversion or diagnostics must be measured separately before assigning
a runtime advantage to the command-line route.

| Reading task | Follow these stages |
|---|---|
| One aperture mean from matter | `physics_3pcf.h` model and projection contracts; `physics_grid_3pcf.c` preparation; direct full-filter contraction or the transform/aperture chain |
| Natural-component signs and source order | `coupling_3pcf.c`; `fftlog_3pcf.c`; `transform_3pcf.c`; the independent phase tests in the project |
| One physical covariance block | The estimator and family definitions in `COVARIANCE.md`; `covariance_physics_3pcf.h`; the corresponding assembly in its C file |
| A notebook input or retained output | The matching `*_wrapper.cpp` and `*_interface.cpp`; the C header's ownership contract; setting-sequence tests |

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

## Preparation, interpolation and parallel work <a name="preparation"></a>

A projected bispectrum requires a radial integral for each Fourier
triangle. Repeating that physical integral whenever a transform requests
a nearby triangle would repeat much of the same work. The production
reader instead uses three stages: evaluate the expensive function on a
coarse grid, prepare a natural cubic spline and sample it densely, then
read the dense uniform table with linear interpolation. Natural cubic
means that the spline's second derivative is zero at its endpoints; its
boundary rule is part of the numerical approximation.

Uniform refers to the stored coordinate, such as $`x=\ln\ell`$.
For spacing $`\Delta x`$, the lower cell index and linear fraction are

```math
j=\left\lfloor\frac{x-x_0}{\Delta x}\right\rfloor,\qquad
t=\frac{x-x_0}{\Delta x}-j,\qquad
f(x)=(1-t)f_j+t f_{j+1}.
```

The read requires index arithmetic rather than a binary search. If many
outputs request the same coordinates, their indices and fractions are
prepared once and reused. This is the strategy used by the ordinary
$`\xi_\pm`$ readers in `cosmo2D.c`. The Limber relation gives
$`\ln k=\ln\ell-\ln\chi`$, so prepared multipole and distance
logarithms also remove repeated coordinate work. Changing distance still
changes the physical wave number: sharing a lookup map must respect that
dependence.

The coarse and dense controls address different errors. Increasing the
coarse node count can resolve structure missed by the physical samples.
Increasing the dense count better approximates the already chosen cubic
curve with short straight segments. The latter cannot recover missing
coarse-grid structure. Signed IA residuals must remain signed across
zero; logarithmic storage is used only where the selected baseline admits
it. Source-reference readers retain named upstream interpolation choices
for controlled comparisons.

Integration is a separate operation. Fourier, angular, mass and radial
integrals use fixed nodes and weights. A Gauss–Legendre quadrature may
have nonuniform nodes while the function it reads lives on a uniform
table. Refining the table does not refine that quadrature. The weights
also carry the coordinate Jacobian: if they already represent
$`d\ell=\ell\,d\ln\ell`$, the C integrand must not insert that
factor a second time. No adaptive subdivision occurs during a Fisher
displacement.

OpenMP distributes work among CPU threads that share the input memory.
It assigns complete output entries to workers. SIMD, single instruction
multiple data, applies the same
arithmetic to independent output entries in vector lanes; a lane is one
position in a small vector of numbers. Each output retains an ordered
sum over its integration nodes. The code comments identify lane meanings,
tail entries and ownership beside the loops. No cosmological setters or
FFTW planners are called from these workers.

The FFTW library prepares a plan for each required Fourier-transform
shape. Plan creation is serial preparation. During a transform, OpenMP
workers execute those plans on private aligned arrays using serial FFTs.
Starting a second FFT thread team within every worker would multiply the
number of active workers; it is not the parallel layout used here. Scratch
allocations precede the worker loops. SIMD tails, including an odd last
entry, preserve the declared scalar arithmetic. The strict build avoids
fast-math reassociation, and local rounding boundaries are used where a
fused scalar operation would otherwise disagree with its vector counterpart.

`PhysicsModel.projection_grids` shares matter and intrinsic-shape work
across a batch of ordered source triplets. The project limits the number
of dense tables held simultaneously. Pure NLA omits the identically zero
residual table. NLA angular-power arrays likewise prepare source kernels
once and read neighboring multipoles with SIMD using the logarithmic
wave-number shift. These are reductions in repeated work, with no timing
claim attached.

Covariance preparation shares Bessel/filter/window geometry across terms.
Under its pure-lensing halo model, a filtered profile separates into a
source efficiency and a radius-dependent shape. Different source samples
with the same radius can share that shape; needed equal radius pairs can
share their transforms. Source weights are retained explicitly. These
are identities of the selected model. Adding a source-dependent halo
prescription would require revisiting that reuse. All covariance work
arrays are released when the offline call returns, while the mean retains
its immutable model and transform geometry for repeated use.

The [numerical guide](NUMERICS.md) gives the individual grids and their
lifetime. OpenMP, SIMD and fewer repeated integrals identify how work is
organized. They do not establish a speedup or an acceptable survey
resolution without a separate measurement and refinement study.

## Changing a calculation without restarting Python <a name="state"></a>

Changing settings in a notebook must either create the requested new
calculation or stop with a clear instruction. It must not leave an array
from the previous model behind a new label. The physical model owns
immutable copies of its input arrays. Editing a caller's original power
table therefore does not partially alter a live model. Construct a new
model to use the new table, and retain the old model if it is still needed
for an A/B comparison.

| Owner | Retained definition | Operation that changes it |
|---|---|---|
| `PhysicsModel` | Physical arrays, cosmological scalars and preparation controls | Construct a new model |
| Projection grid | Model, ordered source triplet, IA/baryon coefficients and every grid setting | Prepare a new grid or source batch |
| `TransformPlan` | Fourier/angle coordinates, selected indices, orders, padding, taper, biases, rotation and worker capacity | Construct a new plan |
| Finite aperture weights | Side/angle bins, widths and three ordered filter radii | Recalculate the weights |
| Project mean engine | Owned observable geometry and exact parameter keys | Construct an engine for new geometry; supply new parameters for new means |
| Physical covariance call | Current model, rows, quadratures, survey and halo settings | Make a new call; no hidden covariance model persists |

Array dimensions alone are insufficient for reuse: moving every
separation by one percent keeps its shape while changing its observable.
Keys must retain the actual values used, including numerical settings.
The Python mean caches use exact parameter values, so a small nonzero
Fisher displacement cannot be rounded into the fiducial cache entry.
Prepared project settings are immutable; create a new owner to change
them in the same Python kernel. Previously returned outputs are owned
arrays and survive later calls. Borrowed production inputs must remain
unchanged for the duration of the native calculation.

The ordinary two-point bridge has a different ownership problem. Its
compiled module has one mutable process-wide state, which a raw notebook
cell can change without telling the project. `CoreTwoPoint.spectra()`
therefore resets and reinstalls its complete declared shear calculation
before loading the new arrays: numerical controls, source counts, TATT,
nonlinear runmode, photo-z conventions and inactive internal baryon
contamination. It also clears internal shifts and calibrations because
the project already applies those effects in its source preparation and
final mean. A reset protects against partly configured raw notebook
models as well as complete ones. Its declared adapter controls are
immutable, with integration levels 0–7 supported explicitly.

This restoration reconstructs input splines. The bridge already invalidates
the ordinary numerical tables on every spectra request; no performance
claim is made for the extra reset work. Calls that mutate the same ordinary
module must remain sequential. A transform plan likewise owns reusable
scratch and is not reentrant. Neither borrowing input arrays nor using
OpenMP inside C permits simultaneous Python callers to modify those owners.

## Accuracy and independent checks <a name="checks"></a>

Three comparisons answer different questions. Agreement with a pinned
original implementation checks the translation of a declared source
convention. Agreement with an independent equation or quadrature checks
normalization and physical identities. Refinement checks whether finite
grids, bounds and angular orders resolve the intended observable. A
source implementation can share a numerical shortcut or bug, so agreement
with it cannot replace the other checks.

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

Setting-sequence checks compare an object reused after many changes with
the same requested calculation in a clean process. They grow and shrink
arrays, change values without changing shapes, interleave old models,
change worker counts and mutate returned outputs. A setting must first
be shown to affect the intended calculation; invoking a setter that never
reaches a tested output proves little. Bitwise equality is appropriate
for these identical-operator state checks. It is generally not the right
requirement for independent quadratures or different source conventions.

For survey predictions, inspect the full vector rather than a few plotted
points. With a matched covariance, a useful measure of a numerical change
is $`\Delta\chi^2=\Delta m^T\mathcal C^{-1}\Delta m`$. This
weights correlated errors by the measurement's precision. Also compare
the derivatives and marginalized Fisher errors: cancellation in a
derivative can reveal an error that was inconspicuous in its mean. Refine
coarse sampling, dense lookup, quadrature and physical integration bounds
separately, holding the estimator fixed. A positive covariance matrix is
not by itself evidence that those integrals have converged.

Live original-code comparisons must record source revisions, hashes,
input arrays and conventions. Historical transpose reuse, the FFT taper,
centroid phases, unequal-radius aperture selection and IA parameter
mapping are separate checks. A comparison of a shared kernel is not an
execution of the complete original survey pipeline. The project validation
record states what was actually run; planned notebooks or unpublished
survey inputs must not be counted as completed evidence.

## Acknowledgements <a name="credits"></a>

**We thank Sunao Sugiyama and Rafael C. H. Gomes for their scientific work
and for making their three-point methods, implementations and analysis
pipelines available to the community. Their work is the foundation of this
CosmoLike port.**

**Special thanks to:**

- **Sunao Sugiyama**, for [fastnc](https://github.com/git-sunao/fastnc), its
  shear three-point multipole method, and the numerical organization used
  by the transforms in this directory. The
  [TATT branch](https://github.com/git-sunao/fastnc/tree/tatt) and main branch
  provide separately identified source comparisons.
- **Rafael C. H. Gomes**, for the intrinsic-alignment modeling and DES
  two-/three-point analysis, and for
  [3pcf_integrator](https://github.com/rchgomes/3pcf_integrator) and
  [desy6-3pcf](https://github.com/rchgomes/desy6-3pcf). Their aperture
  integration and survey workflows guide the port and its companion project.
- **Mike Jarvis**, for [TreeCorr](https://github.com/rmjarvis/TreeCorr),
  its aperture kernels and shear-projection conventions, and **Xiao Fang**
  for the two-dimensional FFTLog work credited by the original transforms.

**Users of this implementation must give appropriate credit to the original
software and scientific papers.** Cite the relevant works below alongside
Cocoa/CosmoLike and the other methods used in the calculation.

| Original scientific work | Contribution used here |
|---|---|
| Sunao Sugiyama, Rafael C. H. Gomes and Mike Jarvis, [Fast modeling of the shear three-point correlation function](https://arxiv.org/abs/2407.01798) | Bispectrum multipoles, natural shear components and the fastnc method |
| Rafael C. H. Gomes and collaborators, [Cosmology with second and third-order shear statistics for the Dark Energy Survey: Methods and simulated analysis](https://arxiv.org/abs/2503.03964) | Joint two-/three-point analysis, aperture statistics and DES Y3 reference inputs |
| Rafael C. H. Gomes, Kyle Miller, Sunao Sugiyama, Jonathan Blazek, Thomas Bakx and Bhuvnesh Jain, [Tidal alignment and tidal torquing modeling for the cosmic shear three-point correlation function and mass aperture skewness](https://arxiv.org/abs/2601.09133) | Three-point TATT and velocity-shear intrinsic-alignment modeling |

The aperture formulas are due to Jarvis, Bernstein and Jain and to
Schneider, Kilbinger and Lombardi. The analytic covariance draws on the
work of Linke, Heydenreich, Burger, Schneider, Wielders and Porth. We also
thank **Tim Eifler, Elisabeth Krause and the CosmoLike contributors** for
the numerical framework, and the Cocoa contributors for the project and
interface conventions. [CREDITS.md](CREDITS.md) gives the corresponding
paper links, source revisions and retained license notices.

The primary code references are
[fastnc TATT](https://github.com/git-sunao/fastnc/tree/7b2336ce37a909bed51f07f32a2a4f096e71b6ce),
[fastnc main](https://github.com/git-sunao/fastnc/tree/e59eab5a07401d2f484bfe00dd48827f2d2162c7),
[Rafael's Y3 integrator](https://github.com/rchgomes/3pcf_integrator/tree/d6098d326dcb03bc750adcc9627119f52d928944)
and [his Y6 pipeline](https://github.com/rchgomes/desy6-3pcf/tree/fd59fe4fee694d359453d5f573d6a4a55a47a812).
The scientific targets include the
[fastnc method](https://arxiv.org/abs/2407.01798),
[joint Y3 analysis](https://arxiv.org/abs/2503.03964) and
[TATT/velocity-shear analysis](https://arxiv.org/abs/2601.09133).
Their source conventions, selected estimators and publication assumptions
must remain identifiable in every comparison. Source attribution does
not replace the retained license notices.

The project implements fixed-covariance Fisher calculations and
precision-matrix expansion using the full joint covariance. Their physical
interpretation still requires numerical refinement, a matched estimator
and explicit priors. The downloaded Y3 inputs are pinned candidates for
published-analysis comparisons. The inspected Y6 repository provides its
pipeline but keeps measured vectors, joint covariance and trained networks
external. A forecast configuration must therefore identify its source
curves and noise inputs rather than silently borrowing Y3 values.
