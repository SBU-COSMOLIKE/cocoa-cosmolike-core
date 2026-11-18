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
| Project `configs/des_y3_*_fisher.yaml` | Declared NLA or TATT forecast design, rows, covariance, priors and numerical controls |

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

A controlled two-source IA example teaches sign and power-prescription
responses without claiming DES amplitudes. The full survey forecast is a
separate, explicit calculation. A small teaching calculation does not validate
its integration grids or finite-difference steps. Likewise, the existing
bispectrum-only feedback parameter does not define a joint two-/three-point
feedback model. No new physical parameter should be varied in only one probe
while its forecast is described as a consistent joint feedback constraint.
