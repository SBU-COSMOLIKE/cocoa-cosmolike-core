# Three-point notebooks, plotting and source comparisons

Read this for the shared plotting folder, DES notebook workflows or live
original-code comparisons. Use [threepcf.md](threepcf.md) for the physical
implementation and [threepcf_state_and_accuracy.md](threepcf_state_and_accuracy.md)
for ownership and numerical controls. The companion project belongs at
`Cocoa/projects/des_3pcf/`; it is a separate Git repository. Preserve unrelated
core changes and commit only the paths owned by this work.

## Put calculation and presentation in their owning layers

| Location | Responsibility |
|---|---|
| `cosmolike/3pcf/` | Physical C calculations and production/notebook C++ adapters |
| `cosmolike_notebook_utils/3pcf/` | Pure-array figures, shared between projects |
| Project `python/des3pcf/` | Physical input preparation, survey rows, comparison workers and Fisher algebra |
| Project `interface/*_notebook_wrappers.py` | Small callable interfaces used by notebook cells |
| Project `interface/cosmolike_des_3pcf_fisher_wrappers.py` | YAML workflow and bounded IA teaching calculation |
| Project `EXAMPLE_COMPARE_CODES.ipynb` | Live original-source comparisons and separate stored physical references |
| Project `EXAMPLE_FISHER_IA.ipynb` | IA response explanation, public-input audit and explicit survey forecast commands |
| Project `EXAMPLE_FISHER_ACCURACY.ipynb` | Default-accuracy Fisher weights, selected noise stencils and full derivative checks |
| Project `EXAMPLE_COVARIANCE_PME.ipynb` | Precision expansion with independent controlled simulation vectors |
| Project `configs/des_y3_*_fisher.yaml` | Declared NLA or TATT forecast design, rows, covariance, priors and numerical controls |
| Project `EXAMPLE_NAUTILUS*.yaml` and `.py` | Joint fixed-Gaussian inference with explicit synthetic or observed data |

Notebook cells call wrappers and shared plotters. They must not contain a
second implementation of a transform, covariance contraction or Fisher
matrix. Production commands use the borrowing `_interface` path; notebook
inspection can use the owned Armadillo `_wrapper` path. Both call the same C
functions. Neither the word notebook nor the word CLI establishes a speedup.
Explain which conversions and diagnostic arrays a particular call requests.

The optional covariance `save_components` setting controls output persistence.
The current C call still computes and returns six component matrices. Omitting
those diagnostic files does not remove their integral work or peak C storage.

## Thread ownership survives a reused notebook kernel

Set every BLAS vendor to one thread before importing numerical libraries.
OpenMP owns the available parallel cores. Tests on the shared laptop use at
most four OpenMP workers, and ordinary examples use one. FFTW plans execute
serially within those outer workers. Never start another FFT or BLAS team in
an OpenMP loop.

A notebook may already have imported NumPy. Environment variables alone do
not reconfigure a loaded BLAS pool. Use the installed `threadpoolctl` public
API to set or verify one BLAS thread, or stop with a clear restart/setup
instruction. This changes a declared numerical-library setting; do not replace
Python methods or entries in `sys.modules`. Child comparison processes receive
the full thread environment before any imports.

Keep fixed model selectors as integers when a likelihood merges a sampled
point with its reference dictionary. Converting every field to float can
leave the first cached prediction working while a later IA displacement
fails at the native integer read. Test full parameter dictionaries, changed
continuous amplitudes and restored starting values; a repeated reference
call alone does not exercise the native input conversion.

## Plot supplied quantities and name their meanings

The folder name `3pcf` begins with a digit. Import it explicitly with
`importlib.import_module("cosmolike_notebook_utils.3pcf")`; do not rename the
folder or modify the package's root `__init__.py` to hide the distinction.
The [plotting guide](../../../../cosmolike_notebook_utils/3pcf/README.md)
contains input shapes and teaching examples.

The shared functions receive arrays. They never import a project's compiled
extension, run CAMB, interpolate a reference onto another grid, change source
order or infer a missing shear convention. Project wrappers perform physical
calculations and return the declared units and axes. The plotter preserves
signs: a negative IA contribution or zero imaginary Gamma is a valid value,
not automatically an excluded measurement.

Use the existing panel, marker, colorbar and `show=None` conventions. Fonts
and rcParams belong in the notebook. The current Cocoa figure rule displays
code/setting differences in percent or supplied error units; label the factor
100 explicitly. A pointwise percent is undefined at a zero reference. Mask
that point visibly, or use an absolute difference or a stated fixed physical
scale. Never insert a denominator floor and present the result as a percent.
A plotted diagonal error does not account for cross correlations.

Paper-related figures need their own scientific explanation:

- A two-side Gamma map fixes the opening angle, component, projection and
  source triplet; changing any of them changes the observable.
- An equal-radius Map3 curve varies a filter radius, not a minimum side cut.
- Operator components and ordered G/I placements are different decompositions;
  summing overlapping component arrays can double count a contribution.
- Correlation matrices use the full declared row order and retain cross blocks.
- Parameter-error comparisons require identical parameters and priors, and
  marginalized errors from the parameter covariance.

Render and inspect every offered mode. Use actual supplied arrays or a named
analytic reference; an illustrative array must not be labeled a DES result.
Record which figures were inspected. Keep saved notebook outputs small enough
that the committed notebook remains usable.

## Compare original code without changing it

The project comparison manifest records source commit labels and SHA256 hashes
of the files executed. YAML locates the checkouts. A missing or changed source
is reported as unavailable; do not replace it silently with a stored result.
Run original packages in separate processes so class-level dictionaries and
import names cannot contaminate a later case. Disable bytecode writes and
source caches. No dependency upgrade, package installation or source patch is
part of the comparison.

Separate these numerical questions:

1. The pinned TATT two-Bessel operation uses its historical taper convention.
2. The pinned main two-Bessel operation contains the corrected taper.
3. Natural components require matching finite Fourier grids, angular orders,
   coupling interpolation, projection and source-leg assumptions.
4. The original NumPy aperture integrator has distinct equal- and unequal-filter
   branches. A demonstrated unequal-filter branch error remains a discrepancy,
   not a reason to alter the physical C formula.
5. Stored physical reference arrays check regression against a recorded source;
   they are not a live run of the original survey pipeline.

Matching only an FFT size is insufficient. The original coupling can contain
finite-quadrature error even when interpolation nodes agree. An independent
analytic coupling check answers correctness; a source comparison measures the
source operator's discretization. Record both rather than increasing a global
tolerance until every difference disappears. COMPARISON.md in the project
owns the executed cases and their measured residuals.

At the pinned TATT revision, `fastnc/integration.py` uses left rectangles
and its default `max_itern=10` stops at 4096 samples. The coupling caller in
`fastnc/coupling.py` discards the convergence flag. For L=M=0 and psi=0,
the integral of `4*pi*cos(delta)` over [0,pi] is zero, but that rectangle
rule gives `4*pi*pi/N`. The live comparison varies the public `max_itern`
argument to expose this bias without editing upstream code. Its coupling
table actually contains 200 nodes despite metadata naming 201. Keep these
source-discretization facts separate from C's analytic coupling checks.

## Interpret the Fisher notebook as a declared experiment

The public Y3 files contain 400 xi rows and 80 aperture rows with a labeled
480-row covariance. Retaining them is an explicit full-vector forecast choice;
it does not reproduce publication scale cuts or compression. The measured
finite-triangle aperture operator must also match the predicted one. In the
Y3 paper, the 8 arcminute cut applies to both triangle-side parameters; it
is not an aperture-radius cut. Exact retained edges, bin averaging and the
emulator definition require their own comparison.

The model's xi transform is flat-sky, while the inspected publication setup
uses a full-sky bin-averaged transform. The available covariance and source
curves alone do not resolve that distinction. PAPER_TARGETS.md records known
publication assumptions, available inputs and unresolved matches. Read it
before labeling a notebook output a reproduction.

The Fisher derivative matrix has rows for measurements and columns for
parameters. To compare xi alone with the joint vector, select its covariance
rows and columns before inversion. An upper block of the inverse joint matrix
would condition on the three-point measurements and answer another question.
For covariance blocks C22, C23 and C33, the added information uses

    S = C33 - C23.T @ inverse(C22) @ C23
    E = D3 - C23.T @ inverse(C22) @ D2
    F_joint = F_two_point + E.T @ inverse(S) @ E

These formulas explain the calculation; the implementation uses Cholesky
solves rather than explicit inverses. Add the same Gaussian prior once to
each posterior comparison. Uniform parameter bounds are not Gaussian prior
precisions. Report rank failures separately; do not use a pseudoinverse to
manufacture finite uncertainties for an unconstrained direction.

A declared fixed-dark-energy comparison selects the retained posterior
Fisher submatrix before inversion, fixing w0=-1 and wa=0 at the saved
fiducial. Invert all retained parameters before selecting Omega_m and S8;
otherwise other nuisance parameters would also be fixed. Count the saved
prior once. This array calculation changes the local parameter experiment,
not the physical fiducial, estimator, scale cuts or posterior likelihood.
The supplied joint Nautilus examples do not by themselves provide a
two-point versus joint posterior comparison; that needs two separately
recorded selections and sampling checks with common priors.

A controlled two-source IA example teaches sign and power-prescription
responses without claiming DES amplitudes. The full survey forecast is a
separate, explicit calculation. A small teaching calculation does not validate
its integration grids or finite-difference steps. Likewise, the existing
bispectrum-only feedback parameter does not define a joint two-/three-point
feedback model. No new physical parameter should be varied in only one probe
while its forecast is described as a consistent joint feedback constraint.

## Reuse the shared Fisher-noise estimator

The expensive full derivative base and its marginalized parameter covariance
are computed once at default accuracy. Use the existing shared functions
`get_stencil_noise`, `get_fom_noise_weights`, and `predict_fom_noise` in
`cosmolike_notebook_utils/fisher.py`. Do not implement a second version of the
formula in this project or in a notebook. The project adapter
`python/des3pcf/fisher_accuracy.py` supplies saved vectors and absolute
parameter offsets only; `scripts/run_fisher_accuracy.py` owns candidate inputs
and output provenance.

`python/des3pcf/provenance.py` shares file and implementation fingerprints
with inference. A new Fisher archive records physical input bytes, project
Python files, both used native extensions and the CAMB implementation when
cosmology is varied. The CLI checks these identities before derivatives
and immediately before saving. Screening refuses a baseline with missing
or mismatched identity; never assign today's hash to an older calculation.
Disk hashes do not inspect already-loaded executable memory or make
concurrent writes atomic. Freeze calculation inputs and implementation
while a scientific run is active.

The mean CLI uses the same beginning/end check. Each accuracy candidate
also rechecks the baseline archive, candidate configuration, plan, shared
diagnostic and physical dependencies immediately before saving. In the
notebook, `FisherWorkflow.ensure_result` means execute or reuse under the
current files, so it rejects a missing or different saved identity.
`load_result` is the explicitly historical reader; it must not imply that
old results were calculated with today's source tables or binaries.

Full accuracy comparisons also require identical archived physical-input
and implementation identities across the baseline, selected-column scan
and full candidate. Matching only row labels and covariance can mix two
implementations and incorrectly attribute their difference to resolution.
Historical readers compare the saved identities with one another; an
executable `run_full_check` additionally checks current files before the
child calculation starts. Keep those reading and execution contracts distinct.

For a partial `--only` scan, reconstruct the summary from every existing
current-plan case whose saved identity still matches. Compute means only
for requested missing cases. Never discard earlier validated results merely
because the latest invocation requested a smaller subset, and never merge
old summary values without checking their numerical archives.

The project's fourth-order steps are absolute YAML values. Supply zero-valued
offset coordinates to the shared estimator, and add those offsets to the
physical fiducial inside the adapter. This avoids treating an absolute step
as a fraction of a negative or nonzero fiducial. The core estimator's
`AccuracyBoost` argument remains one when a candidate predictor already owns
its numerical controls; construct that predictor with the desired settings.

The full Fisher CLI saves `stencil_values_h` and `stencil_values_h2`, each
ordered [parameter, five offsets, data row]. Offsets are [-2,-1,0,1,2]. All
these vectors already exist in its exact mean cache, so retaining them adds
no physical evaluations. Score every baseline column from those saved arrays,
rank its measured contribution, and evaluate a few selected columns at each
candidate setting. A shared center plus four displaced vectors per selected
column gives the reduced evaluation count.

For step studies, reuse those saved vectors as explicit arrays. The h and
h/2 stencils already contain zero and +/-h/2, +/-h, +/-2h. New +/-h/4
points complete h/4; optional new +/-4h points complete 2h. Two selected
columns need four or eight new means. Do not fabricate entries in a model's
internal cache. Check identical shared-node bits and input identities before
work, then retain the new stencils and verify dependencies before saving.
Normalize a plotted noise ratio to the same selected columns and reference
spacing, rather than to a different all-parameter score.

The project's `tests/studies/cosmology_step_study.py` performs this targeted
Omega_m/Hubble-h test for the w0-wa FoM. Its notebook reader checks retained
array shapes, shared-node bits and the complete newly evaluated parameter
coordinates; another FoM pair is refused. The independent array studies
`analyze_des3pcf_fisher_amplification.py` and
`analyze_des3pcf_fisher_step_blocks.py` inspect nuisance projection and
two-point/three-point sensitivity without evaluating another physical mean.
`full_triangle_report.py` compares completed all-column refinements with
selected-column and complementary substitutions, including fixed/free
dark-energy IA and S8 summaries. It first compares the shared screening
stencils with the full run bit for bit. A partial replacement mixes two
numerical grids and can miss compensating changes in nuisance directions;
its FoM shift is not a prediction of the all-column shift. Establish the
shared-stencil agreement before assigning that difference to omitted
columns rather than model state or evaluation order.
The completed NLA and TATT triangle comparisons demonstrate this effect:
full h/2 FoM changes are +0.170% and -0.055%, whereas the four-column
substitutions give -6.72% and -5.30%. The original step disagreements
remain. These values belong to the project's stated demonstration and
must not be turned into general accuracy tolerances.

The estimate assumes approximately independent evaluation jitter. Smooth
fourth derivatives can contaminate it; odd coherent errors and correlated
stencil errors can escape it. Report the omitted columns and measured
baseline contribution fraction, rather than treating a small correlation
weight as a bound on unknown noise. Replacing selected derivative columns
and recomputing the FoM is an additional partial test. Final acceptance still
requires full derivatives at selected settings and both step sizes, together
with integration-range, angular-truncation and external-table checks.

In the current plan, `multipole_1p5` refines the multipole interpolation
tables. It leaves Lmax and Mmax fixed. A historical xi quadrature plot
using linear-midpoint sources motivates the Fourier-node increase but
does not certify the later measured-edge PCHIP baseline. Each numerical
evidence record must identify its source reconstruction and actual controls.

The companion project's `tests/studies/angular_order_study.py` separates
Legendre truncation, Fourier truncation and prepared angular integration.
Its `angular_order_validation.md` protocol starts from an explicit completed
triangle-refined Fisher archive, reuses its mean and launches no models
without `--execute`. The prepared angular counts are `nmu_log/nmu_linear`,
not the direct route's `nmu`. Both lmax fields must agree. Coupling `npsi=0`
removes the historical interpolation of the analytic coupling; it does
not refine the physical multipole table's separate `npsi` coordinate.
The protocol compares L50/M20 and L50/M50, then angular-only, coupling-only
and combined refinements at L50/M50. The 63-angle point estimator aliases signed Fourier modes separated
by 126, so an M100 comparison on those points is not a resolved continuous
angular integral. A successful fiducial comparison is not derivative
convergence. Recalculate allocation estimates before raising several
three-dimensional interpolation counts together.
The completed ten fiducial comparisons are recorded separately in
`tests/studies/angular_order_results.md`. Angular-sampling and coupling
changes partly cancel in the covariance metric. Compare each with their
common L50/M50 reference; do not add squared distances as independent
error variances or infer derivative accuracy from the small combined
mean distance. The protocol remains unchanged because its bytes are
part of the saved execution identity.

Compare each derivative column in the full covariance metric. A maximum
raw-component difference can hide aperture errors because xi and Map3
have different numerical amplitudes. Whitening by the Cholesky factor of
the joint covariance preserves cross correlations and makes the ratio
`norm(D_h-D_h2)/norm(D_h2)` invariant under a change of observable units.
Do not combine parameters with different units into one unweighted norm.

The two-point Fourier transform also needs refinement. Its oscillating
Bessel functions are not resolved merely because the input power spectrum
is smooth. `tests/studies/xi_refinement.py` changes ell and within-bin angular
quadratures and support separately. The Python fixed Gauss-Legendre rule
uses SciPy's tridiagonal root preparation, avoiding a dense count-by-count
companion matrix at large orders. This is a fixed quadrature, not adaptive
integration. Numerical rule checks and physical integral convergence are
different tests.

## Source support before a new Fisher baseline

The configured source FITS is part of the physical target. The Y3 final
midpoint feature can be missed by the old coarse background grid. An
independent `source_nz` separates source resolution from CAMB redshifts.
`source_boost`, `kernel_boost` and `chi_boost` test source sampling,
prepared windows and radial quadrature separately.

The explicit `pchip_midpoint` choice has two endpoint policies. The
`zero_at_measured_edges` policy preserves supplied `Z_LOW/Z_HIGH`, takes
their outer support intersected with nonnegative redshift, and adds a zero
plateau to the larger background limit. The
`zero_at_observer_and_background_edge` alternative changes the curve all
the way to that limit. Neither is an implicit histogram-cell probability
reconstruction. Record original edges, the selected FITS hash, the SciPy
version, and unshifted support before comparing photo-z derivatives.

The linear midpoint curve can jump to zero at a nonzero endpoint. Products
of differently shifted source curves can then have overlap boundaries
that create a deterministic cusp. A nonzero fourth difference is not
automatically random numerical noise. The project's
`tests/studies/source_policy_convergence.py` compares saved full joint
means, one-sided slopes and three derivative step sizes without new CAMB
solves. Its sampled IA-kernel minima do not prove continuous positivity.
Use that evidence to distinguish a reconstruction assumption from refinement
of one fixed physical target before recomputing a full Fisher base.

The project's `tests/studies/source_policy_validation.md` records the
completed source, kernel and radial comparison. Its selected demonstration
counts do not become core defaults. Step agreement within one source grid
does not bound the change under another grid: the tested fourth difference
can decrease on a coarser grid even when its derivative is farther from
the refined result. Preserve the original source support and bin-probability
qualification when choosing a smoother reconstruction.

## Observed inputs and inference examples

The pinned integrator repository contains real unblinded Y3 measurements in
addition to its simulated analysis inputs. The project's separate
`data/y3_observed_manifest.json` records the matched observed two-point and
aperture files, joint covariance, upstream INI files and Git blob hashes.
`scripts/fetch_y3_observed.py` verifies those identities. Do not describe the
twelve-file simulation selection as proof that observed data are unavailable.
The observational companion, arXiv:2508.14018, is distinct from the simulation
methods paper arXiv:2503.03964.

The matched row definitions contain 400 xi and 80 Map3 entries. Their source
columns match the selected simulated reference, including twelve negative
samples in the first bin. The explicitly authorized demonstration processing
clips and normalizes a derived file with a separate record; it is not an
official recalibration. Use the separately paired joint covariance. Extra
headers in the observed two-point file's own covariance do not define a
joint matrix with the aperture vector.

`python/des3pcf/likelihood.py` selects matching data/mean rows and both
covariance axes before a Cholesky factorization. It includes the Gaussian
normalization and no prior penalties. The inference YAML supplies normalized
priors to Nautilus once. A synthetic example generates its data with the exact
declared fiducial model; it does not relabel a different public simulated
mean as its own prediction. An observed example remains a declared Gaussian
demonstration until estimator, cuts, compression and finite-mock likelihood
choices match the publication. A retained checkpoint requires verified target
identity before resuming.

## Match the estimator before claiming a paper reproduction

The project's `tests/studies/estimator_conventions.md` distinguishes the
published side-bin bank, the pinned theory quadrature and unknown catalogue
weights. The Y3 paper starts with 20 logarithmic side bins over 0.5–80
arcmin before its low-side cut. Ten replacement bins over 8–80 are a
different quadrature. The pinned Y3 code also uses an empirical correction
to the area-weighted representative side; a geometric midpoint does not
reproduce that convention. The 63-angle description in the paper and the
100-angle active pinned configuration remain different choices. State
each one rather than infer it from the 80 packed aperture labels.

The project's `ia_figure9_workflows` holds configured inputs fixed and supplies
a common no-IA mean plus opposite-sign NLA and TATT means. Its finite
sign-pair distance uses every selected row and the conditional Map3
covariance through `finite_sign_pair_separation`. It is not a
nuisance-profiled significance. The full Fisher
experiment provides a separate local marginalization test.

An overall Map3 sign transformation must act on both the mean and the
covariance cross block. The inspected upstream likelihood applies no
hidden sign conversion. Its public packing notebook copies the selected
joint covariance from an earlier NPY; the realization-level producer of
that NPY has not been traced. This is a provenance limit, not evidence
that the provided covariance is wrong. Preserve the paired-file convention
until its generating inputs establish a reason to change it. Neither
positive definiteness nor agreement of the C33 block settles the C23 sign.

## Distinguish the precision-expansion simulation designs

The project `precision.py` exposes three statistical constructions. The
full-covariance Wishart estimator requires independent Gaussian summary
vectors with an unbiased sample covariance. The known-component estimator
uses C=A+B and M=A+Bm, where simulations measure B with the A-generating
random contribution absent. Its quadratic correction must act on Bhat, not
on A+Bhat: adding a deterministic A does not produce a Wishart full sample.
The independent-group construction uses two independent unbiased covariance
estimates to remove the Gaussian assumption from the quadratic product's
expectation. None of these removes Taylor truncation error.

For Map3, shape-noise covariance includes mixed signal/noise terms. A diagonal
white-noise matrix or a two-point block is not automatically a valid known
component A. Check the same estimator, mask, row order and cross covariance,
and establish the simulation decomposition. Independent parent skies must
not be counted as independent patches merely because many patch vectors were
stored. The PME notebook uses controlled Gaussian vectors with a known true
covariance to teach these requirements; it does not manufacture DES mock
catalogues or claim a calibrated survey correction. Report nonpositive
precision estimates without eigenvalue repair or a pseudoinverse.

An estimator-family string such as `finite_sas`, even with exact row labels,
does not identify its triangle boundaries, weights, masks or calibration.
The simulation-file loader checks only its declared fields. Require an
analyst's matching measurement/input manifest before interpreting a real
survey correction. Do not imply that the controlled notebook supplies a
converged finite-SAS TATT covariance or independent survey realizations.

State the outcome of the saved draw even when the noisy correction is
worse than the analytic model. Removing product bias in expectation does
not guarantee improvement for one small simulation sample. Keep held-out
validation independent of both covariance estimation and model tuning;
do not replace inconvenient draws or repair their eigenvalues.
