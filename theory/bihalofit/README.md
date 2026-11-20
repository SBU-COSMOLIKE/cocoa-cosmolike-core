# BiHalofit theory block

BiHalofit is one producer for the supplied-bispectrum interface. Its fit
and calibration belong to this theory component; they are not requirements
on an alternative matter-bispectrum model.

The Python theory producer evaluates the dark-matter-only BiHalofit fit and
always supplies its positive, full matter bispectrum as a table. C performs
the line-of-sight projection, intrinsic alignment, aperture integration and
covariance calculation. The internal C matter fit has been removed.
`physics.bispectrum_source` therefore accepts only `supplied`. Existing
configurations without a selector use that value; those without explicit
axes use the supplied example's uniform sampling, with the background z
endpoint belonging to their own project.

The provider owns the background and matter prescription. It supplies the
matter-density fraction `omega_m_z` from the same cosmology as P, B, growth
and distances. The C model requires B and this fraction; missing inputs are
errors. Tree-level F2 and power calculations remain in C because TATT needs
them. One-/three-halo fit diagnostics are available from Python BiHalofit,
not from a full-B native projection model.

The required table introduces a separate interpolation error and preparation
cost. Earlier comparisons below record the retired C fit as a historical
reference; their dates and numerical results are retained. They do not
certify current full-forecast accuracy or timing. The regression tests now
compare the Python equation with frozen pre-removal C outputs, and test C
projection/state behavior with explicitly supplied tables.

## Acknowledgements and origin

**We thank Sunao Sugiyama for fastnc, including its TATT branch, and Rafael
Gomes for 3pcf_integrator, the DES Y6 pipeline, and the analyses that motivate
this joint two/three-point implementation. Their original work and code are
the basis for this port.**

- [Sunao Sugiyama's fastnc TATT branch](https://github.com/git-sunao/fastnc/tree/tatt),
  with comparison revision `7b2336ce37a909bed51f07f32a2a4f096e71b6ce`.
- [Rafael Gomes's 3pcf_integrator](https://github.com/rchgomes/3pcf_integrator)
  and [DES Y6 pipeline](https://github.com/rchgomes/desy6-3pcf).
- The [methods analysis](https://arxiv.org/abs/2503.03964) and
  [subsequent analysis](https://arxiv.org/abs/2601.09133) motivate the
  observable and inference work. The BiHalofit fitting formula itself is
  from [Takahashi et al., *Fitting the nonlinear matter bispectrum by the
  Halofit approach*](https://arxiv.org/abs/1911.07886).

The Python module includes Sunao's MIT notice. The equation comparison pins
the pre-change `physics_3pcf.c` SHA-256 to
`af3c533d7a85ddf493ab5840823134206fcd154ac2d5cd648875185de41ac351`.
That identifies the implementation whose numerical conventions were
ported; it is not a replacement for citing the fit or the original codes.

## What the fit receives and computes

The inputs are the linear power spectrum at redshift zero, a growth factor,
the primordial spectral index, the present-day fluctuation amplitude and
the matter-density fraction as a function of redshift. All are supplied
explicitly. The units are

| Quantity | Convention |
| --- | --- |
| `k` | $`h/\mathrm{Mpc}`$ |
| `p_linear0` | $`(\mathrm{Mpc}/h)^3`$ |
| `growth` | dimensionless $`D(z)`$, with $`D(0)=1`$ |
| `omega_m_z` | dimensionless $`\rho_m(z)/\rho_{\mathrm{crit}}(z)`$ |
| returned $`B`$ | $`(\mathrm{Mpc}/h)^6`$ |

The fit begins with the dimensionless spectrum

```math
\Delta^2(k,z)=\frac{k^3 P_{\mathrm{L}}(k,0)D(z)^2}{2\pi^2}.
```

It finds a nonlinear radius $`R_\sigma(z)`$ from

```math
\sigma_G^2(R,z)=\int d\ln k\,\Delta^2(k,z)e^{-k^2R^2}=1.
```

At this radius, the logarithmic slope of the variance gives an effective
spectral slope. The fitted coefficients depend on that slope, the
redshift-dependent top-hat fluctuation amplitude and, for one coefficient,
$`\Omega_m(z)`$. The final result is the sum of the fit's one-halo and
three-halo pieces. These names describe the terms in the fitting formula;
they do not imply that the implementation performs a separate halo-mass
integral for each queried triangle.

The producer uses the same input-grid convention as the Cocoa likelihoods:
`log10k = np.linspace(...)`, followed by `k = 10.0**log10k`, and redshift
segments constructed with `np.linspace`. Production power and growth
preparation uses a natural cubic within each uniform segment, followed by
dense linear interpolation with arithmetic cell indices. Refinement keeps
every segment endpoint, including a join between fine low-redshift sampling
and coarser high-redshift sampling. Each join needs a coarse-grid refinement
check because natural end conditions need not match the physical curvature.
The explicit `source_readers=True` comparison retains fastnc's original
not-a-knot preparation. Python and C must select the same convention when
comparing their fitted equations; changing a dense sampling factor cannot
remove a discrepancy between different cubic boundary conditions.

The old C normalization is deliberately retained. First, a top-hat integral
over $`10^{-3}\leq k\leq100\ h/\mathrm{Mpc}`$ rescales the input power to the
supplied `sigma8`. The later fit variances use a lower limit of
$`10^{-4}\ h/\mathrm{Mpc}`$. The Gaussian variance includes the same analytic
power-law tail beyond $`100\ h/\mathrm{Mpc}`$ as C. Its incomplete-gamma
integral is evaluated for positive, zero and negative shape parameters.
There is no adaptive integration in the production producer: the numerical
rules use explicit uniform logarithmic nodes and Simpson integration, with
the same even-node closing correction as C.

A special-function check found that naive downward recurrence loses
precision when its gamma shape approaches zero: a subtraction of nearly
equal numbers is divided by that small shape. At a shape of minus one part
in a trillion, its relative error reached 0.00137 in the small probe.
The corrected evaluation removes that cancellation analytically for
absolute shape below 0.001. It uses a convergent series when the
dimensionless lower bound is at most one, and a continued fraction above
one. Negative shapes with larger lower bounds also use that fraction to
avoid a second recurrence cancellation. These are evaluations of the same
integral, with explicit iteration limits; the physical Simpson grids and
root tolerance are unchanged.

The [special-function tests](../../../../../projects/des_3pcf/tests/data_vector/test_bihalofit_python.py)
compare with 80-digit mpmath values. The saved validation covers 24 shapes
on both sides of zero and the algorithm boundaries, a negative-integer
recurrence crossing, and 12 lower bounds from $`10^{-300}`$ to 1000.
The largest relative error was $`1.03\times10^{-12}`$, at the shape-branch
boundary. The values near zero agreed within $`4.93\times10^{-14}`$; underflow
to zero was checked separately. mpmath is a validation dependency already
present in Cocoa, and does not enter the production calculation.

These conventions matter for an implementation comparison. Removing the
power renormalization, changing the integration bounds or supplying a
different density history can change a prediction even if the equation
port is correct. The producer records its normalization factor so a user
can inspect it. A fit that trusts the raw power normalization would be a
separate, tested change.

## Uniform triangle coordinates and the C reader

Sort the three wavenumber magnitudes as $`a\leq b\leq c`$. The stored
coordinates are

```math
z,\qquad \ln r=\ln b,\qquad \ln u=\ln(a/b),\qquad v=(c-b)/a.
```

The triangle inequality makes $`0\leq v\leq1`$, and $`u\leq1`$. At $`v=0`$
the two longer sides coincide; at $`v=1`$ the triangle is flattened. The
native key `logk` denotes $`\ln r`$, the middle side. This spelling matches
the existing feedback-table interface. The shortest side can be much
smaller than the middle side, so `logu` is logarithmic. Every axis must have
at least four finite, increasing, uniformly spaced nodes in its stored
coordinate.

The array is `log_bispectrum[nz, nk, nu, nv]` in C order, with `v` varying
fastest. A positive full $`B`$ is required to take its logarithm. A signed
diagnostic piece cannot be substituted for the full table or clipped to
make the logarithm possible. The supplied C route retains the power-derived
tree diagnostic and explicitly refuses the unavailable one-halo/three-halo
diagnostics. The Python equation object can still evaluate all four pieces
for reference checks.

Preparation uses cubic interpolation to construct the dense uniform C
table. Repeated C reads use an arithmetic cell index and linear weights;
they need no binary search or mutable interpolation accelerator. Python's
power/background readers use the same cubic-preparation/uniform-linear-read
strategy in production mode. `source_readers=True` is an explicit comparison
mode that retains the upstream cubic conventions. It is not the default.

The default table policy refuses a query outside its domain. The standalone
producer also accepts `domain="clamp"`, which holds boundary values. Clamping
changes the prediction and requires its own accuracy study; it must not be
used to conceal missing table coverage. The project configuration exposes
the refusing policy. The chosen domain must cover all triangles reached by
the mean and covariance, including squeezed triangles and the lowest
line-of-sight distance used in a projection.

## Producer and ownership

The reusable implementation is
[bihalofit.py](model.py). `BiHalofit` evaluates the
equation, `BiHalofitProducer` prepares a uniform table, and
the result is an ordinary dictionary of arrays, background and metadata. A minimal preparation is:

```python
from theory.bihalofit.model import BiHalofitProducer

producer = BiHalofitProducer(
    z=baseline_z,
    logk=baseline_logk,
    logu=baseline_logu,
    v=baseline_v,
    boost=2,
    domain="error",
)
baseline = producer.prepare(
    tables=physical_tables,
    fit_nz=33,
    table_boost=8,
)
```

`physical_tables` contains `k`, `p_linear0`, `z`, `growth`, `omega_m_z`,
`ns` and `sigma8`. An optional `omega_m_redshift` gives the coordinate axis
of the density-fraction samples when it differs from `z`. For equal axes,
the returned background preserves the supplied `omega_m_z` bytes. For
different axes, `baseline["background_omega_m_z"]` is the explicitly resampled
array on `baseline["background_z"]`. A direct native caller must use those
returned background arrays, so the baseline and its C model share one
background. The project helper checks this agreement.

At the native boundary pass `baseline["native"]`. This dictionary contains
exactly `z`, `logk`, `logu`, `v`, `log_bispectrum`, `boost` and `clamp`.
The likelihood base checks the shapes, finite values, uniform axes and
background agreement. The C constructor copies the arrays and prepares its
interpolation tables. There is no required Python table class; another
theory can supply the same arrays.

The producer keeps its latest result. It hashes every consumed physical
array, its dtype and shape, the output axes, and every numerical control.
An unchanged call returns that owner. A changed call constructs a new owner
before replacing the cache. The input key and its owner are published
together, so independent Python preparations cannot pair one cosmology
with another cosmology's table. An invalid replacement leaves the last valid
owner usable. The A-to-B-to-A sequence therefore recovers the same A values
without requiring a notebook restart. Source distributions and IA
amplitudes do not enter the dark-matter-only fit; they remain dependencies
of the projected C model.

`baseline["metadata"]` contains the calculation record.
It includes source credits, the pinned C source hash, physical-input
hashes, numerical controls, the normalization factor, and hashes of every
actual native table/background array. Data-vector, covariance and mask
files are outside this producer; a likelihood comparison must hash them
separately.

## Refinement controls

There are several different approximations. Increasing only a dense-table
boost does not repair an inadequately sampled coarse function.

| Control | Approximation it changes |
| --- | --- |
| output `z`, `logk`, `logu`, `v` axes | Sampling of the full bispectrum before four-dimensional interpolation |
| producer `boost` | C uniform upsampling of that full table |
| `fit_nz` | Redshift anchors of the fitted radius and coefficients |
| `table_boost` | Python dense readers for input power, growth and fit coefficients |
| `variance_nodes` | Normalization and Gaussian/top-hat integration; default 1000 |
| `slope_nodes` | Effective-slope integration; default 2000 |
| `radius_rtol` | Nonlinear-radius bisection; default $`10^{-5}`$ |
| `block_size` | Number of vectorized triangle queries at once; it controls temporary memory, not numerical resolution |

The project `Accuracy` integration multiplies the full-table interval counts
by `accuracy_boost * bispectrum_boost`. The individual `bispectrum_boost`
changes this table alone. The global `accuracy_boost` also refines other
prepared sampling grids, including the fit readers. Use the individual
controls to isolate an error source before applying a global refinement.
The direct producer exposes quadrature/root controls for additional studies.
The final four-dimensional table is always stored explicitly; block
evaluation bounds temporary arrays, not the size of that output or its
dense C copy. Count the dense nodes before choosing a large boost on four
axes.

## Cobaya Theory adapter

[bihalofit_theory.py](bihalofit.py) is an actual
`cobaya.theory.Theory` subclass. It declares requirements for linear
total-matter `Pk_interpolator`, `sigma8_z`, `Hubble`, `Omega_b`, `Omega_cdm`,
`Omega_nu_massive` and `ns`. It provides `bihalofit_baseline` and
`bihalofit_background`. A consuming component requests these products
through the normal Cobaya provider. Merely importing the module enables
nothing.

The required configuration values are `baseline_z`, `baseline_logk`,
`baseline_logu` and `baseline_v`. `background_z` can specify the C model's
different background grid. The provider is asked for matter fractions on
both that grid and the exact fit anchors. Their sum is used directly,
including the provider's massive-neutrino and expansion treatment. The
adapter uses $`D(z)=\sigma_8(z)/\sigma_8(0)`$.

The adapter converts physical units exactly once:

```math
k_{\mathrm{physical}}=h\,k_{h},\qquad
P_{h}=h^3 P_{\mathrm{physical}}.
```

Its default Boltzmann computation limit is `kmax_boltzmann=5` in
$`\mathrm{Mpc}^{-1}`$. The wider output `power_k` grid is sampled through the
provider interpolator with explicit power-law extrapolation. Asking for
large output wavenumbers therefore does not silently request a Boltzmann
calculation up to $`1000\ h/\mathrm{Mpc}`$. The extrapolation remains a physical
assumption whose effect requires testing; a fast interpolation is not a
calibration of high-$`k`$ power.

The direct project pipeline uses the reusable producer without Cobaya's
component manager. The focused unit test instantiates Cobaya's real base
class with an explicit array provider. The separate
[actual provider-graph check](../../../../../projects/des_3pcf/tests/studies/bihalofit_cobaya_check.py) creates
a real Cobaya Model, loads the installed CAMB 1.6.7, and uses a validating
likelihood consumer to request both baseline and background products.
Its A, unchanged A, B, restored A sequence passed: unchanged inputs retain
the same owner, a changed primordial amplitude replaces it, and restoring
the amplitude restores identical table values.

The [saved provider result](../../../../../projects/des_3pcf/tests/studies/results/bihalofit_cobaya_provider.json)
records all component settings, requirements and input hashes. CAMB was
asked for a physical wavenumber limit of 5 per Mpc; its internally extended
raw grid ended at 5.69845 per Mpc. The much wider output grid was obtained
through the explicitly requested interpolator extrapolation. This is a
small component and cache check with a zero-valued validating likelihood.
It does not test a survey likelihood or replace a full inference run.

## Validation and physical limits

Component regression, supplied-table projection and refinement are
different checks. The [recorded comparisons](../../../../../projects/des_3pcf/tests/studies/implementation_record.md#bihalofit_python)
retain their input choices and numerical results. They do not establish
convergence of every survey Fisher derivative. The fit's use of
P(k,0)D(z)^2 assumes scale-independent growth; agreement with its
implementation does not extend its dark-matter-only calibration to
arbitrary neutrino, dark-energy or feedback models.

## Shared variance inputs

Both the direct project producer and the optional Cobaya theory block can
consume Gaussian and top-hat variances from the ordinary CosmoLike FFTLog
engine. Select `variance_source: shared` in the theory block and set
`variance_module` to the calling project's compiled interface import name
(for DES, `cosmolike_des_3pcf_interface`), or select
`physics.variance_source: shared` in the CLI configuration. The default
remains `local` until full joint mean and Fisher acceptance is established.
The shared route supplies owned z/rnl/neff/sigma8 arrays to the fit and
retains the amplitude of the supplied power spectrum. A separate sigma8
scalar no longer renormalizes that power.

The default `variance_growth: factorized` uses P(k,0)D(z)^2. The optional
`evolving` choice uses the supplied full P(k,z). Those prescriptions can
differ with massive neutrinos; neither changes the existing growth choice
on the perturbative tree and IA power legs. For the Cobaya adapter, D is
still the provider's sigma8(z)/sigma8(0), and Omega_m(z) remains the sum of
its baryon, cold-dark-matter and massive-neutrino fractions. All h-unit
conversions occur at the Python boundary.

The theory's `variance_accuracy` mapping accepts the shared controls
described in [Accuracy](../../cosmolike/3pcf/README.md#shared-smoothing). Table counts derive
from an unchanged baseline. The cached bispectrum includes every supplied
variance array in its input identity, and old outputs retain their own
copies after replacement. The physical-model and CLI tests compare C and
Python equations with identical anchors, exercise both covariance adapters,
and check unchanged power when only the redundant sigma8 scalar changes.
These tests establish the interface contract; they do not establish survey
forecast convergence or a new neutrino halo prescription.
