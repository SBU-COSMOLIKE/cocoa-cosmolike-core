# Uniform interpolation tables for the three-point physical model

The proposed production design is one interpolation sequence shared with
the rest of CosmoLike: evaluate a physical quantity on a declared uniform
coordinate, prepare its cubic interpolant using `basics.c`, fill a denser
uniform table, and use arithmetic-index linear interpolation in repeated
calculations. A modest increase in node counts is acceptable when it removes
a separate interpolation implementation and passes the accuracy checks below.

The objective is less numerical code to explain, maintain and test at a
measured accuracy. A runtime reduction is a separate result to establish.
Moving the existing irregular spline into `basics.c` changes its ownership;
it does not establish whether production still needs that interpolation.

**Status, 2026-11-18:** this is a plan, supported by read-only code inspection
and in-memory numerical pilots. No C implementation was changed for those
pilots. Their Python calculations use the natural-cubic equations implemented
by the shared helper; they do not validate a compiled replacement. No full
data-vector, derivative, Fisher or timing comparison has been run for this
proposal. Nothing under `Cocoa/projects/des_3pcf/` is to be edited until the
maintainer explicitly releases that restriction.

1. [Scope and ownership](#scope)
2. [What the current calculation actually does](#current)
3. [What changes when a spline is replaced](#mathematics)
4. [Measured pilots and their limits](#pilots)
5. [Proposed production representation](#representation)
6. [Implementation sequence](#sequence)
7. [Source kernels and integration measures](#sources)
8. [Shared Gaussian variance and physical conventions](#variance)
9. [Accuracy controls and memory](#controls)
10. [Numerical acceptance tests](#accuracy)
11. [State changes, threads and ownership](#state)
12. [Performance measurements](#performance)
13. [Deliverables and adoption decision](#deliverables)
14. [Pilot inputs and reproduction record](#record)

## Scope and ownership <a name="scope"></a>

This study concerns interpolation and preparation in
[`physics_3pcf.c`](../../cosmolike/3pcf/physics_3pcf.c), reuse of the
[`basics.c`](../../cosmolike/basics.c) uniform spline functions, and the
relationship to the newly available background quantities in
[`cosmo3D.c`](../../cosmolike/cosmo3D.c). Its first implementation experiment
should change one numerical choice at a time. It should not simultaneously
change the bispectrum prescription, intrinsic-alignment model, estimator,
covariance, parameter priors or Fourier extrapolation.

The current inspection refers to core commit
`ba8b490a3da20203989114e9017046e833d1d88a` on `lowrank_fftlog`. Another team
is integrating the three-point library with the common core. Before an
implementation starts, identify which functions that work has moved or
replaced. Locate them by responsibility and name, rather than applying the
line numbers from this inspection to a later tree. The plan does not require
reversing an intervening relocation of the irregular spline.

This directory owns the proposed study document and can later hold its
independent comparison drivers and result records. It must not become a
second production implementation. C experiments should use isolated build
outputs, leaving installed project libraries and the other team's working
files unchanged. The project freeze also covers notebooks, YAML files,
Makefiles, tests and saved outputs inside `des_3pcf`; it is not limited to C.

The current request creates this document only. Proposed C edits, native
experiments and project integration remain subsequent work. No README or
shared skill file needs to change to deliver this plan. When an implementation
is accepted, its documented behavior and validation limits must be reconciled
with the [three-point numerical guide](../../cosmolike/3pcf/NUMERICS.md) and
the [state and accuracy reference](../../.claude/skills/cosmolike-dev/references/threepcf_state_and_accuracy.md).

## What the current calculation actually does <a name="current"></a>

The matter power spectrum describes correlations between two Fourier
density fluctuations; the bispectrum describes three fluctuations whose
wavevectors close into a triangle. BiHalofit is the fitting formula used
here for the nonlinear matter bispectrum. The calculation integrates these
matter correlations along the line of sight to predict angular lensing
statistics, then transforms them into shear correlations and aperture moments.

Intrinsic alignment describes correlations of galaxy shapes with the tidal
environment before light propagation produces gravitational shear. The
nonlinear-alignment model, NLA, supplies the baseline shape contribution.
The tidal-alignment and tidal-torquing model, TATT, adds quadratic shape
contributions. Their different source kernels explain why one successful
matter-power interpolation check cannot validate the entire prediction.

An irregular input axis and an irregular repeated reader are different
things. The physical model accepts increasing arrays, constructs splines,
and then replaces several of them with dense uniform tables. The ordinary
production power queries already find their cells by arithmetic. The
presence of `interval_physics` in the file therefore does not mean every
bispectrum evaluation performs a binary search.

| Quantity or operation | Current representation or calculation | Proposed first action |
|---|---|---|
| Linear power | A general spline in log k, later sampled uniformly | Test the shared natural spline on the already-uniform log-k input |
| Nonlinear power within each redshift row | A general log-k spline, later sampled uniformly | Test the same replacement at fixed redshift |
| Nonlinear power between redshifts | General cubic preparation of log-power columns, followed by uniform redshift reads | Test the redshift endpoint change separately from the log-k change |
| Growth | General cubic input followed by dense uniform reads | Compare natural interpolation in the chosen uniform time coordinate |
| Distance | General cubic input used during radial preparation | Test independently sampled uniform background grids |
| Distance Jacobian | General cubic input used during radial preparation | Test accuracy and consistency with the selected distance representation |
| BiHalofit coefficient tables | Natural cubic on uniform production anchors | Replace the duplicate solve with the shared uniform coefficient function |
| Lensing kernel resampling | Cubic interpolation in irregular comoving distance | Compare a uniform-coordinate representation with the existing kernel |
| Alignment kernel resampling | Separate baseline and residual grids in comoving distance | Preserve both definitions while changing one interpolant at a time |
| Reference mode | Original spline reads and upstream-style anchors | Retain an explicit comparison reference during the experiment |

Construction currently calls `prepare_bihalofit`, then `prepare_radial`,
then `prepare_uniform_readers`. In particular, `variance_physics` can read
the original linear-power spline repeatedly during the nonlinear-radius
solve, before the uniform production reader exists. Calling a calculation
"preparation" does not make all work inside it negligible: this preparation
is repeated for changed cosmologies.

Conversely, the final power reader already shares a log-k cell and fraction
between its two redshift rows. Its production redshift bracket is also
found arithmetically. Any speed claim must distinguish savings in model
construction from savings in the final triangle and projection calculations.

The input archive used in the pilot has 256 uniformly spaced log-k nodes,
125 uniformly spaced redshift nodes and 9601 uniformly spaced source-redshift
nodes. Its comoving-distance array is not uniform. The measured source
samples from which the source array was reconstructed are also not a
uniform grid over their entire padded support. These facts describe this
input; production must validate its own axes rather than infer uniformity
from a familiar node count.

## What changes when a spline is replaced <a name="mathematics"></a>

For a uniform stored coordinate $`x`$, write
$`x_i=x_0+i\Delta x`$. A query uses

```math
j=\left\lfloor\frac{x-x_0}{\Delta x}\right\rfloor,
\qquad t=\frac{x-x_0}{\Delta x}-j,
\qquad f(x)=(1-t)f_j+t f_{j+1}.
```

The coordinate might be $`\ln k`$, redshift, scale factor or a declared
triangle coordinate. Uniformity is a property of that stored coordinate.
Uniform redshift does not make $`\chi(z)`$ uniform. The index must be found
from the coordinate used to construct the table, with explicit endpoint
handling so that $`j+1`$ remains inside its allocation.

Distance affects both geometry and the physical scales being sampled. For
example, in the flat-sky Limber approximation a lensing bispectrum contains

```math
B_\kappa^{ijk}(\ell_1,\ell_2,\ell_3)=
\int d\chi\,\frac{W_i(\chi)W_j(\chi)W_k(\chi)}{\chi^4}
B_\delta\left(\frac{\ell_1}{\chi},\frac{\ell_2}{\chi},
\frac{\ell_3}{\chi};z(\chi)\right).
```

Here $`W_i`$ is the physical convergence window per unit distance for source
bin $`i`$, and $`\ell`$ is an angular Fourier wavenumber. An interpolation
error in distance changes the geometric weight and the three matter
wavenumbers. The source code factors some of these terms into prepared
kernel arrays; those factorizations must preserve this physical dependence.

The current general spline uses a not-a-knot endpoint condition: its first
two intervals share one cubic polynomial, as do its last two intervals.
The shared `spline_coeffs_uniform` function instead uses natural boundaries,
setting the second derivative to zero at both ends. Both interpolate the
supplied nodes, but their values between nodes differ near a curved boundary.
The distinction persists even when all input nodes are already uniform.

This explains the distance pilot. The exact distance vanishes at the
observer, while its curvature there is generally nonzero. Forcing zero
curvature produces an error close to the endpoint; dividing by the small
distance makes its fractional error larger. More uniform samples can reduce
that error. A denser *read* table alone cannot correct the endpoint condition
of a fixed coarse cubic curve.

The shared coefficient convention also matters. `basics.c` returns
$`c_i=S''(x_i)/2`$, whereas the local three-point spline stores
$`M_i=S''(x_i)`$. For $`\delta=x-x_i`$, the shared coefficients give

```math
S_i(x)=y_i+b_i\delta+c_i\delta^2+d_i\delta^3,
\qquad
b_i=\frac{y_{i+1}-y_i}{\Delta x}
     -\frac{\Delta x}{3}(2c_i+c_{i+1}),
\qquad
d_i=\frac{c_{i+1}-c_i}{3\Delta x}.
```

Passing $`c_i`$ into an expression expecting $`M_i`$ would halve the
curvature correction. The first shared-helper test should compare node
values, off-grid values and endpoint behavior with an independent natural
cubic reference. Bitwise equality is not presumed when the order of the
linear-system arithmetic changes.

The extrapolation rule is another separate choice. Power currently retains
endpoint power-law slopes. Replacing an interior interpolator must not
silently replace those slopes with dense-grid finite differences, or turn
an extrapolation into endpoint clamping. Tests must cover both sides of
every boundary as well as the interior.

## Measured pilots and their limits <a name="pilots"></a>

The pilots used one saved cosmology, serial BLAS and one OpenMP worker.
They neither timed an implementation nor rebuilt a native library. Their
purpose was to determine whether the uniform natural-cubic alternative
merits a compiled experiment and to identify the most sensitive endpoint.

**Changing the spline with the input nodes fixed.** The power comparisons
used the existing arrays and compared natural-cubic and not-a-knot
interpolation of log power. The redshift comparison kept the existing
log-k preparation fixed, so it isolated the second interpolation axis.

| Quantity changed | Largest sampled relative difference |
|---|---:|
| Linear power: log-k spline condition | $`5.64\times10^{-7}`$ |
| Nonlinear power: log-k spline condition | $`3.19\times10^{-6}`$ |
| Nonlinear power: redshift spline condition | $`1.56\times10^{-4}`$ |
| Growth: redshift spline condition | $`4.60\times10^{-6}`$ |
| Distance Jacobian: redshift spline condition | $`2.08\times10^{-5}`$ |

These are differences between interpolants of the same supplied samples,
not errors against independently calculated power or growth. The nonlinear
power redshift difference is largest near the low-redshift endpoint in this
sample. The power tests do not determine the accuracy of a projected shear
statistic or of its parameter derivatives.

**Increasing independently evaluated background samples.** For distance,
CAMB evaluated new background values directly, without a transfer-function
or nonlinear-power solve. Each candidate used a natural cubic, an eightfold
dense-grid refinement, and arithmetic-index linear reads. The reference
was CAMB distance at 16001 query redshifts from 0.005 to 3.1. Zero redshift
was excluded from the fractional error because its exact distance is zero.

| Uniform coordinate | Coarse nodes | Maximum relative distance error |
|---|---:|---:|
| Redshift | 125 | $`2.16466\times10^{-3}`$ |
| Redshift | 249 | $`6.73834\times10^{-4}`$ |
| Redshift | 497 | $`7.14414\times10^{-5}`$ |
| Redshift | 993 | $`7.68528\times10^{-6}`$ |
| Redshift | 1985 | $`3.47694\times10^{-7}`$ |
| Scale factor | 125 | $`2.27240\times10^{-4}`$ |
| Scale factor | 249 | $`2.69531\times10^{-5}`$ |
| Scale factor | 497 | $`6.52189\times10^{-7}`$ |
| Scale factor | 993 | $`2.48745\times10^{-7}`$ |
| Scale factor | 1985 | $`8.84520\times10^{-8}`$ |

The scale-factor axis runs from $`a=1/4.1`$ to $`a=1`$. Its values map to
redshift through $`z=1/a-1`$. The last interval therefore places more samples
near the observer than a uniform-redshift grid of the same size. The
497-node scale-factor case is a candidate to test, not a selected default.
The pilot did not vary cosmology or test lower integration endpoints.

For comparison, the existing 125-node not-a-knot distance interpolant had
a maximum relative difference of approximately $`2.38\times10^{-6}`$
against direct CAMB background values in the companion query check.
The background values at the saved nonzero nodes matched exactly. This
comparison concerns the existing cubic distance reader; it is not a
comparison with a hypothetical dense-linear version of that reader.

The pilot results were printed by in-memory calculations. No reusable
comparison script or full result archive was created at that stage. Before
using these numbers to choose a production default, reproduce them in the
study driver described below and save the actual query arrays and errors.

## Proposed production representation <a name="representation"></a>

The target sequence is

```text
declared physical samples on a uniform coordinate
    -> shared natural-cubic preparation in basics.c
    -> dense uniform samples
    -> direct cell index and linear interpolation
```

Use the existing one-dimensional coefficient routine and, where its layout
matches the data, `spline2d_upsample_uniform`. The two-dimensional function
uses successive one-dimensional natural-spline preparations. Do not add a
second tensor interpolation implementation merely because the original
three-point file used a different array layout. Conversion at construction
is preferable to maintaining two numerical algorithms, provided allocation
and conversion costs are measured.

| Quantity | Candidate coordinate | Quantity stored | Boundary behavior to retain or test |
|---|---|---|---|
| Linear power | $`\ln k`$ | $`\ln P_L`$ | Declared power-law continuation |
| Nonlinear power | $`\ln k`$ and a tested uniform time coordinate | $`\ln P_{NL}`$ | Independent Fourier and time-domain policies |
| Distance | Scale factor as the first candidate | $`\chi`$ | Exact observer value and tested near-observer accuracy |
| Distance Jacobian | Same background coordinate | $`dz/d\chi`$ | Positive finite values and consistency with distance |
| Growth | Tested uniform background coordinate | The currently declared growth quantity | Preserve its normalization and physical definition |
| BiHalofit coefficients | Existing uniform redshift anchors initially | Existing coefficient values | Existing natural-cubic convention |
| Radial kernels | Uniform redshift, scale factor or distance, tested separately | Signed physical kernel values | Source support, normalization and observer behavior |

The first implementation need not force every table to use scale factor.
Uniform log k plus uniform redshift is already a usable power representation.
Different *coordinates* can share the same coefficient routine and reader;
they do not require different spline solvers. Record the coordinate in the
table definition so a caller cannot interpret a scale-factor array as a
redshift array.

A production table needs its count, lower coordinate, spacing or inverse
spacing, values and explicit boundary policy. It should not retain copied
coordinate and second-derivative arrays solely for reads that use neither.
Reuse existing core storage conventions and allocators. If an exported
diagnostic needs the coordinates, derive them from the stored definition or
retain an explicitly owned diagnostic array; do not change a public return
shape unnoticed.

Arbitrary measured source samples remain legitimate inputs. They can be
converted once using their declared reconstruction model. Removing a search
from repeated calculations does not require discarding measured edges,
changing histogram meaning or interpreting irregular samples as uniform.
The intended endpoint is one production interpolation family, with any
upstream finite-operator reference kept outside repeated production work.

## Implementation sequence <a name="sequence"></a>

### A. Establish an executable comparison

Record the exact core revision, relevant source hashes, native-library hashes,
input hashes and resolved settings. Reconcile the plan with any intervening
shared-spline relocation. Create an isolated study driver that reads the
existing physical archives without writing into the project. Its outputs
must distinguish a mathematical Python pilot from a compiled production
comparison.

Keep the existing implementation available as a recorded comparison. Also
prepare an independently refined reference: agreement with an old finite
grid is a regression check, not proof that either result is accurate.
Before changing a default, reproduce the baseline mean and its declared
numerical settings in the isolated study environment.

**Deliverable:** repeatable component comparisons, input identities and a
baseline record. No default changes.

### B. Share the already-uniform natural-spline calculation

Replace the duplicate natural solve used for production BiHalofit
coefficients with `spline_coeffs_uniform`, retaining its existing anchors,
values and boundary conditions. The current local builder first constructs
a general not-a-knot spline and then overwrites its second derivatives.
The replacement should construct only the values and coefficients it uses.

Keep the dense-reader resolution unchanged for this first comparison. Check
the factor-of-two coefficient convention, endpoint values, small legal node
counts and both interface layers. Differences beyond floating-point
reordering require investigation before another interpolation changes.

**Deliverable:** one shared natural-cubic solver for this path, with no
intentional change to the mathematical interpolant.

### C. Replace the power preparation one axis at a time

First test natural-cubic preparation of log power on uniform log k while
holding redshift interpolation fixed. Then test the redshift interpolation
while holding the accepted log-k choice fixed. Compare each with both the
current calculation and an independently refined physical reference.

Validate uniformity at input. A producer that declares a uniform grid
should generate it from its endpoints and interval count. Validation must
allow floating-point construction error but reject a physically different
irregular axis; silently snapping measured nodes is not acceptable.
For genuinely irregular supplied power arrays, specify the one-time input
conversion or explicit rejection before changing the public contract.

After each axis passes, test the shared two-dimensional preparation. Keep
the existing Fourier slopes and extrapolation rules fixed. Include queries
just inside, exactly at and just outside each endpoint. Do not optimize
only the in-range branch and accidentally change the high-k model.

**Deliverable:** a uniform power table produced by shared spline machinery,
with an explicit input contract and unchanged extrapolation assumptions.

### D. Separate background resolution from power redshifts

Test uniform scale factor for distance using independently evaluated
background samples. Start with the pilot counts, then vary cosmology,
redshift range and the lower line-of-sight endpoint. Test growth and the
distance Jacobian independently; the distance result does not choose their
resolution automatically.

The current project producer limits its requested transfer redshifts to
150, and the physical input structure shares its background redshift axis
with power. Increasing that common axis to 497 would therefore not be an
appropriate implementation of the pilot. Background evaluation and power
transfer sampling must have separate counts and, if needed, separate axes.
Choose that interface after checking the other team's integration work.

Possible implementations are independently owned background arrays, or
snapshots prepared from the ordinary core's initialized background functions.
In either case, retained three-point model objects must continue to represent
their original cosmology. A live callback into mutable global background
state would violate that requirement.

**Deliverable:** tested background resolution without a forced increase in
Boltzmann transfer redshifts or in the nonlinear-power table size. Any
required project binding or producer change stays pending during the freeze.

### E. Replace radial-kernel interpolation

Use the kernel tests in the next section to choose uniform redshift, scale
factor or distance. Test the pure lensing kernel first, then the source
kernel, then the combined lensing and alignment kernel. Preserve the
separate NLA baseline and TATT residual definitions during each experiment.

Increase only the kernel or source resolution implicated by a failing
comparison. A candidate that needs more one-dimensional samples may still
remove substantially more code than it adds. Do not multiply all table
dimensions to compensate for one unresolved boundary.

**Deliverable:** projected kernels that meet the declared mean and derivative
criteria using shared uniform interpolation.

### F. Remove machinery with no remaining production consumer

After the replacements pass, inspect every remaining use of the irregular
builder, reader and destructor. Keep any required upstream reference in an
external comparison driver where possible. Do not add C entry points whose
only purpose is exposing discarded production internals to tests.

If the general spline has already moved to `basics.c`, check all its callers
before removing it. Other consumers may require it. The desired result is
that the three-point production path no longer needs that machinery; deleting
a genuinely shared function is not an independent objective.

**Deliverable:** an account of which duplicate code was removed, what remains
and why, followed by an undergraduate-level review of the resulting flow.

## Source kernels and integration measures <a name="sources"></a>

The source probability per unit redshift, $`n_z(z)`$, differs from probability
per unit distance:

```math
n_\chi(\chi)=n_z(z(\chi))\frac{dz}{d\chi},
\qquad
\int n_z(z)\,dz=\int n_\chi(\chi)\,d\chi.
```

Changing the interpolation coordinate does not change which probability
density is stored. A kernel sampled uniformly in scale factor does not
acquire a factor $`dz/da`$ unless its integration variable is also changed.
Keep units and the integral measure explicit at every stage.

The current baseline forms a combined kernel containing lensing and the
redshift-dependent linear-alignment coefficient before splining it. Splining
a product and multiplying separately interpolated factors give different
finite-grid approximations. Preserve this order for the coordinate trial.
Changing the order would need a separately labeled comparison.

The residual TATT terms use a source-grid kernel distinct from the baseline
kernel grid. These terms can change sign and cancel the baseline. Do not
interpolate their logarithms or divide their errors by individual entries
near zero. Compare signed differences using a declared component scale and
then the survey covariance.

| Kernel test | Required comparison |
|---|---|
| Source normalization | Area in the declared measure before and after conversion |
| Measured support | No invented sources outside the adopted reconstruction support |
| Observer boundary | Correct zero-source condition without a division by zero |
| Pure lensing | Same geometry, source normalization and physical distance |
| Pure alignment | Same density Jacobian and redshift-dependent amplitude |
| Combined baseline | Same product-before-interpolation convention |
| Residual TATT kernel | Same independent source sampling and signed values |
| Photo-z displacement | Positive and negative shifts reconstructed from original source data |
| Source-bin permutation | Correct ordered source labels on Fourier legs |

Uniform lookup tables do not require uniform quadrature nodes. The current
distance-weighted trapezoid visits its prescribed nodes in order; that loop
does not perform a binary search merely because its distances are irregular.
Keep those nodes and weights fixed while testing interpolation. If a later
experiment changes the integration variable, derive and test its Jacobian
and weights in a separate change.

The existing source-efficiency preparation also admits a distinct study of
cumulative integrals, as in the ordinary core. That could remove repeated
source integrations, but it must not be bundled with the spline comparison:
otherwise a changed projected kernel would have two possible numerical causes.

## Shared Gaussian variance and physical conventions <a name="variance"></a>

The core now exposes `sigma2_gauss`, `dlnsigma2_gauss_dlnr`, `rnl_gauss`,
`neff_gauss` and the total-matter top-hat quantity `sigma2_matter`. Their
existence changes the justification for retaining a separate three-point
variance implementation. Review them before optimizing that duplicate.

The Gaussian variance and nonlinear radius obey

```math
\sigma_G^2(R,z)=\int d\ln k\,
  \frac{k^3P_L(k,z)}{2\pi^2}\exp[-(kR)^2],
\qquad \sigma_G^2(R_{\rm NL},z)=1,
\qquad n_{\rm eff}=-3-
  \left.\frac{d\ln\sigma_G^2}{d\ln R}\right|_{R_{\rm NL}}.
```

Matching these equations is necessary but insufficient. The present
three-point preparation uses $`P_L(k,0)D^2(z)`$ with its declared effective
growth. The shared core can use a supplied redshift-dependent total-matter
power table. With massive neutrinos, those can differ physically because
growth depends on scale. A replacement must either supply the same
factorized spectrum for an algorithm comparison, or explicitly classify
the use of the full spectrum as a separate model change.

Also compare total matter versus cold matter, radius and power units,
growth normalization, Fourier limits, continuation tails, root accuracy
and the top-hat window used for $`\sigma_8`$. Use identical input spectra
for the first numerical comparison. A native regression that happens to
pass at one cosmology cannot establish equivalence of these conventions.

Do not adopt a shared global reader without a state-lifetime decision.
Warm its tables serially, record the inputs and numerical settings, and
either copy the required prepared quantities into the immutable three-point
model or enforce an explicit current-generation contract. The existing
promise that old model objects remain usable favors owned snapshots.

## Accuracy controls and memory <a name="controls"></a>

Coarse physical samples, dense interpolation samples and integration nodes
control different errors. For a coarse axis with $`N`$ points and dense
subdivision factor $`b`$, preserve endpoints and old nodes through

```math
N_{\rm dense}=b(N-1)+1.
```

For noninteger requested boosts, resolve an explicit integer interval count
and record it. Reapplying a boost must start from the saved baseline counts,
not from the previously boosted counts. Restore the previous counts when
the user restores the previous settings.

| Independent control | What it should change | What it should leave fixed in the isolated test |
|---|---|---|
| Power log-k sampling | Physical power samples along log k | Fourier range and extrapolation law |
| Power time sampling | Physical power samples along the selected time coordinate | Background grid and source reconstruction |
| Background sampling | Distance, Jacobian or growth preparation axis | Number of transfer redshifts |
| BiHalofit anchors | Physical coefficient calculations | Dense interpolation factor |
| Kernel sampling | The kernel's physical preparation nodes | Measured source reconstruction model |
| Source sampling | The numerical representation of the declared source curve | Original measured samples and support policy |
| Dense subdivision | Reads of an already prepared cubic | Coarse physical samples |
| Integration accuracy | Fixed quadrature nodes and weights | Interpolation ranges and physical model |

These are responsibilities, not a prescription to add eight new public
settings. Reuse existing controls where their meaning matches. Add a new
control only where the background/power separation or another actual
independent experiment requires one. Both the global sampling boost and
the integration boost must retain their documented meanings.

A 497-node background curve with subdivision eight has 3969 dense values,
or 31,752 bytes for one double-precision value array. This makes extra
one-dimensional samples a plausible trade for removing a second spline
implementation. The number excludes scratch, metadata and other fields.

By contrast, the present 125-by-256 power input with subdivision eight on
both axes gives 993-by-2041 dense values: 16,213,704 bytes for the values
alone. Increasing both dimensions multiplies this storage. Avoid coupling
a distance accuracy requirement to that two-dimensional allocation. Measure
total allocations and peak temporary storage, including any duplicated axes,
before selecting a default.

## Numerical acceptance tests <a name="accuracy"></a>

Run component, projected-mean and derivative checks as separate tests. A
component result can reject a candidate cheaply; it cannot accept the whole
forecast. Numerical equality at input nodes alone does not test interpolation.

**Component comparisons.** Use direct physical evaluations where available,
an independently implemented natural cubic for the shared coefficient check,
and refined physical grids for quantities without a direct cheap reference.
Check node values, midpoints, deterministic off-grid points and boundary
neighborhoods. Include curved functions so a wrong endpoint condition is
visible; a linear polynomial alone cannot distinguish spline choices.

Vary cosmology within the declared model domain, including the neutrino
settings used for the joint forecast. Vary the redshift interval and the
lower line-of-sight limit. Check positive power and monotonic distance, but
do not clip a failing interpolant into compliance. Validate signed kernels
through signed comparisons rather than positivity tests.

**Projected means.** Use the same observable rows, finite-triangle weights,
pixel window, source reconstruction and covariance for both implementations.
Compare NLA and TATT separately, retaining unequal source combinations.
Define the mean displacement and its survey-weighted size as

```math
\delta\mu=\mu_{\rm candidate}-\mu_{\rm reference},
\qquad
\Delta\chi^2_{\rm mean}=\delta\mu^{\mathsf T}C^{-1}\delta\mu.
```

Here $`C`$ is the full joint covariance, including the two-/three-point
cross block. Solve using its factorization rather than explicitly forming
an inverse just to evaluate this quadratic form. Inspect individual
tomographic combinations as well, so cancellations in a summary cannot
hide a misplaced source label.

**Derivatives and Fisher information.** For each parameter $`p_i`$, compare
$`D_i=\partial\mu/\partial p_i`$ at common finite-difference steps and
then refine the steps. Include photo-z shifts and TATT amplitudes because
source boundaries and cancellations can expose errors absent in the mean.
Compare all columns in the intended parameter set before accepting a
marginalized FoM; a small selected-column test is only a screening test.

```math
\epsilon_i=
\frac{\sqrt{(D_i^{\rm candidate}-D_i^{\rm reference})^{\mathsf T}
C^{-1}(D_i^{\rm candidate}-D_i^{\rm reference})}}
{\sqrt{(D_i^{\rm reference})^{\mathsf T}C^{-1}D_i^{\rm reference}}},
\qquad F=D^{\mathsf T}C^{-1}D+F_{\rm prior}.
```

For an exactly zero or nearly unresolved derivative, report an absolute
covariance-weighted difference with a declared parameter displacement;
do not manufacture a large relative error by dividing by zero. Keep
physical parameter units, priors and rank checks fixed. Report unresolved
directions instead of using a pseudoinverse to produce a finite FoM.

For a chosen parameter pair, define the figure of merit as
$`{\rm FoM}=1/\sqrt{\det\Sigma_{\rm pair}}`$, where
$`\Sigma_{\rm pair}`$ is the two-by-two submatrix selected from the inverse
of the full Fisher matrix. Inverting only the pair's Fisher block would
instead hold all other parameters fixed and would test a different result.
State the pair and its units beside every reported FoM.

The following are proposed initial engineering targets, not achieved
results or universal survey requirements. They can be tightened for the
intended science, but should not be loosened after seeing a failure without
an explicit reason and a new comparison.

| Quantity | Proposed initial target against an independently refined reference |
|---|---|
| Full-vector displacement | $`\Delta\chi^2_{\rm mean}\le0.01`$ |
| Resolved derivative change | $`\epsilon_i\le10^{-3}`$ for each tested column |
| Marginalized parameter errors | Relative change at most 0.5% |
| Reported parameter-pair FoMs | Relative change at most 0.5% |
| Numerical refinement | A further targeted refinement remains within the same limits |

These targets supplement existing core regression requirements. A change
in the fitted data $`\chi^2`$ is not the same quantity as
$`\Delta\chi^2_{\rm mean}`$: the former also depends on the data residual.
Record both when a data likelihood is part of the test. If the accepted
candidate differs from the old implementation because the old finite grid
was inaccurate, retain that evidence rather than relabeling the old result
as an exact physical reference.

**Other consumers of the physical model.** The analytic covariance also
reads physical power, bispectra or kernels. Holding a supplied covariance
fixed in the mean/Fisher test isolates changes in the mean; it does not
validate a newly computed covariance. Compare affected covariance terms
separately with their integration settings held fixed. Retain the declared
Gaussian and non-Gaussian physical approximations.

Use diagonal relative differences where the reference is nonzero and
inspect changes after weighting by the reference covariance for the full
matrix. A Cholesky factorization $`C=LL^{\mathsf T}`$ gives the dimensionless
matrix $`L^{-1}\delta C L^{-\mathsf T}`$. Report its matrix norm with the
norm definition, rather than dividing individual cross-covariance entries
by values that may cross zero. Symmetry and positive definiteness are
necessary algebraic checks, not evidence of physical or numerical accuracy.

The project freeze prevents integrating new tests or saving new results
there. Full comparisons may use an isolated study driver and read-only
copies of existing inputs once a native experiment is authorized. Project
notebook and YAML adoption must remain explicitly pending until the freeze
is lifted. Do not mark those steps passed based on the component pilot.

## State changes, threads and ownership <a name="state"></a>

Every persistent allocation must know the dimensions it was allocated for.
Every prepared value must represent all physical inputs and numerical
choices that determine it. Coordinate kind, endpoint policy, bounds,
coarse counts and dense counts are part of that definition; changing
"redshift" to "scale factor" with the same array length still changes it.

The existing model-owned tables can remain immutable. A changed definition
constructs a new model, while an old model keeps its own arrays and results.
If common core static tables are used, their allocation and refill checks
must cover the same information, and `initial_setup` must reset their
generation state. Reusing a pointer or matching a count is not evidence
that its values describe the current cosmology.

| Sequence | Required result |
|---|---|
| Cosmology A, then B, then A | Final A agrees with independently constructed A |
| Baseline resolution, refined resolution, baseline | Restored values and sizes match the baseline definition |
| Coordinate choice changed at fixed count | Preparation changes; an old cached result is not returned |
| Table range changed at fixed count | Reader coordinates and boundary behavior are rebuilt |
| Source or photo-z change | Radial kernels are rebuilt from the corresponding source model |
| IA change | Combined kernels and projection weights use the new amplitudes |
| One worker, four workers, one worker | Numerical differences satisfy the declared determinism policy |
| Two retained model objects | Evaluating one does not mutate the other's prediction |
| Failed model construction | Existing models remain valid and allocated scratch is released |

Prepare allocations, spline coefficients and any lazy shared tables before
parallel reads. One OpenMP layer owns a calculation. FFT planning remains
serial where relevant, and BLAS remains at one worker. Run strict and DEBUG
builds in separate study outputs, with the existing sanitizer checks.
Do not rebuild or overwrite the frozen project's installed extension.

Use ordinary C and the existing allocator, interpolation and OpenMP patterns.
Avoid pointer aliasing tricks, unchecked casts or relaxed floating-point
flags. SIMD work should reuse cell indices and contiguous data where it
fits the existing implementation; a new vectorized interpolation algorithm
is not required to establish the uniform-grid design.

## Performance measurements <a name="performance"></a>

No speedup is inferred from the pilot. Measure the accepted numerical
candidate at equal accuracy, and include all preparation it performs when
a model changes. Report the stage costs as well as the total so that a
cheaper reader cannot conceal a larger preparation or memory cost.

| Stage | Work to include |
|---|---|
| Physical-model preparation | Input conversion, power/background interpolation and allocations |
| Matter-fit preparation | Variance, nonlinear radius and coefficient tables |
| Radial preparation | Source normalization, kernel integrals and resampling |
| Angular prediction | Bispectrum projection and transforms |
| Observable assembly | Two-point rows and aperture contraction |

Use the established CLI benchmark design: ten recorded, distinct physical
parameter points, identical between reference and candidate. An unchanged
notebook call can return a cached vector and is not a recomputation timing.
Use a separate warm-up point, report initial construction separately, and
include the changed-point preparation in the measured evaluations. Prepare
identical CAMB inputs outside the timed interval and record that exclusion.

NLA and TATT are separate tests. Start with one and four OpenMP workers,
with BLAS fixed at one. Eight-worker measurements require a later quiet
machine window and the corresponding authorization; the current ordinary
test limit remains four. Do not run a timing study on the busy laptop or
publish a single evaluation as a scaling result.

Record the distribution across the same parameter points and repeated
passes, the compiler/build mode, thread environment and peak memory.
Interleave candidate and reference passes when appropriate to reduce a
systematic order effect. The two-/three-point comparison must exercise
the same retained rows and numerical acceptance targets.

A candidate may be adopted for removing duplicate code even if runtime is
unchanged. Any runtime or memory increase must be measured and stated. A
large regression should trigger another coordinate/count comparison, not
an assumption that fewer binary searches necessarily made the code faster.

## Deliverables and adoption decision <a name="deliverables"></a>

| Deliverable | Evidence required before calling it finished |
|---|---|
| Reproduced pilots | Saved inputs, query arrays, errors, versions and hashes |
| Shared natural-spline use | Independent coefficient and interpolation comparisons |
| Uniform power preparation | Separate axis tests and unchanged declared continuations |
| Background representation | Direct-background checks over the chosen cosmologies and ranges |
| Radial representation | Support, normalization, signed-kernel and photo-z checks |
| Joint mean comparison | Identical rows, physical assumptions and full covariance |
| Fisher comparison | All intended derivative columns, rank checks and targeted refinement |
| Notebook-state behavior | Parameter/setting sequences and retained-model tests |
| Performance record | Equal-accuracy changed-point timings and peak memory |
| Code reduction | Remaining general-spline consumers identified and justified |
| Didactic review | A reader can follow units, coordinates, boundaries, preparation and reads |
| Project adoption | Performed only after the project restriction is lifted |

Keep the experiments in separate reviewable commits: shared-helper reuse,
power interpolation, background representation, source kernels, and final
removal of unused code. A failed candidate should not replace the accepted
default. Preserve its measurements so that a later change does not repeat
the same unsuccessful configuration without new evidence.

The preferred outcome is one shared uniform interpolation family with
enough samples to pass the science tests. If one quantity needs a different
coordinate or more nodes, that is compatible with the objective. If a
natural endpoint condition demands excessive cost for a particular quantity,
first test another physically appropriate uniform coordinate and range.
Only then consider a documented shared uniform-grid boundary option; do
not silently restore a second module-local spline implementation.

The upstream numerical sequence remains a comparison target, while the
continuum equations and independent refinement establish numerical accuracy.
Retain the credits to Sunao Sugiyama, Rafael Gomes and the scientific sources
listed in the [three-point source record](../../cosmolike/3pcf/CREDITS.md).
Changing interpolation does not change the origin of the physical model.

## Pilot inputs and reproduction record <a name="record"></a>

The pilot archive is the existing project file
`Cocoa/projects/des_3pcf/runs/y3_bacco_physics.npz`. Its SHA256 is
`e95b7dcb32d7af5b2f4807eb8a944a37be1e48a1ccf0952c16edc6d7ce660527`.
It remains read-only. No archived result was relabeled to match a later
implementation.

| Input | Value |
|---|---:|
| $`\Omega_m`$ | 0.289 |
| $`\Omega_b`$ | 0.0463 |
| Reduced Hubble parameter $`h`$ | 0.722 |
| Scalar spectral index $`n_s`$ | 0.959 |
| Total-matter $`\sigma_8`$ | 0.7865548373195137 |
| $`w_0`$ | -1 |
| $`w_a`$ | 0 |
| Neutrino mass sum | 0.06 eV |
| Massive neutrino species | 3 |
| Effective neutrino number | 3.046 |
| Saved background redshifts | 125 over [0, 3.1] |
| Saved power wavenumbers | 256 over $`[10^{-5},10^3]\,h\,{\rm Mpc}^{-1}`$ |
| NumPy version | 1.26.3 |
| SciPy version | 1.12.0 |
| CAMB version | 1.6.7 |

For the independent background calculation, set the same cosmology and
neutrino species as the archive. Subtract CAMB's exact neutrino density
from the initially assigned cold-dark-matter density so that the stated
$`\Omega_m`$ counts neutrinos once. Use the same dark-energy prescription.
The pilot called `get_background` with `no_thermo=True`; it did not request
matter transfer functions or nonlinear power. Convert CAMB distances from
Mpc to Mpc/h by multiplying by the same $`h`$.

For each distance candidate, evaluate the background directly at its new
coarse nodes. Construct `CubicSpline` with `bc_type="natural"`, evaluate
that cubic at $`8(N-1)+1`$ dense nodes, and read those dense values by the
arithmetic-index linear formula. Compare with direct CAMB distances on the
query grid. Reusing a not-a-knot curve to generate new coarse distances
would be a different experiment and would retain its original error.

The power log-k pilot used 8193 uniformly spaced query coordinates over
the saved log-k interval and all supplied nonlinear redshift rows. The
redshift pilot first made 2041 dense log-k values with the existing
not-a-knot interpolation, then compared the two redshift spline conditions
at 993 redshifts. The growth/Jacobian comparison used 16385 redshift
queries over [0, 3.1], retaining queries at or above 0.005 for the reported
relative differences. Save these query definitions explicitly in the
reproduced study; they are finite samples rather than global error bounds.

Source SHA256 values for the inspected revision:

| Source | SHA256 |
|---|---|
| `cosmolike/3pcf/physics_3pcf.c` | `af3c533d7a85ddf493ab5840823134206fcd154ac2d5cd648875185de41ac351` |
| `cosmolike/basics.c` | `09db9a04e41d3f1f57aaa3e27d84286329bdd1548754f8b8e6e0401026bcb225` |
| `cosmolike/cosmo3D.c` | `39d1d69b1b0a028589a9f43126063081e48367eb5d459c4feae55eb06bf1e3c0` |
