# Covariance and Fisher forecasts for DES Y3 and Y6

> **Archive status.** This source study records design proposals and validation requirements, not completed features. Read the [maintained implementation status](../threepcf.md) first. Both the mean and its covariance belong in `cosmolike/3pcf/`; earlier destination proposals using `fastnc/` or `covariances/` are superseded.

A checked three-point mean prediction and a CPU analytic covariance are the two numerical deliverables. Credible DES Y3 and Y6 forecasts also need a controlled calculation of parameter derivatives and a stated treatment of covariance-model uncertainty. A successful C translation of the mean does not finish those tasks. The proposed sequence is to reproduce the supplied papers at their stated scope, establish a Y3 Fisher reference, and evaluate a separate Y6 forecast with its own survey specification. The Y6 workflow is examined in [Chapter 10](10_desy6_workflow.md); the covariance port, actual core mechanisms, and simulation corrections are developed in [Chapter 15](15_analytic_covariance_cpu_port.md), [Chapter 17](17_cosmolike_covariance_cpu_transfer.md), and [Chapter 18](18_precision_matrix_expansion.md).

## 1. Specify what is being forecast

The initial joint vector should contain cosmic-shear two-point correlations and aperture skewness. Write it as

```math
\boldsymbol{d}=\begin{pmatrix}\boldsymbol{\xi}\cr\boldsymbol{m}_3\end{pmatrix},
\qquad
\boldsymbol{C}=\begin{pmatrix}
C_{22}&C_{23}\\
C_{23}^{\mathsf T}&C_{33}
\end{pmatrix}.
```

Here $`\boldsymbol{\xi}`$ includes the selected $`\xi_+`$ and $`\xi_-`$ values, and $`\boldsymbol{m}_3`$ contains the selected third moments of aperture mass. The subscripts on the covariance label the order of the statistics. $`C_{23}`$ describes how fluctuations in a two-point measurement accompany fluctuations in a three-point measurement of the same survey. A different cosmological response of the means does not imply statistical independence of their fluctuations.

For four source bins, an equal-radius aperture statistic has 20 unordered source triplets, because the number of combinations with repetition is $`4\times5\times6/6`$. Four aperture radii then give the Y3 reference's 80 entries. The inspected Y6 repository instead specifies five radii and therefore 100 entries before cuts. These counts assume the statistic and filter assignments actually permit that permutation symmetry. A full natural-component vector at labeled vertices, or unequal filters attached to particular source bins, needs a more explicit ordering. The index table must specify the statistic, source labels, filter radii or triangle bins, complex component, and scale selection for every row.

Adding galaxy clustering and galaxy–shear correlations later creates further cross-covariances with aperture skewness. That is a separate extension beyond the shear-only reproduction target. Starting from an existing `3x2pt` covariance and appending a three-point diagonal block would omit those correlations.

## 2. Why the existing covariance engine is a foundation, rather than the finished covariance

CosmoLike's covariance module already contains radial projections, survey windows, shape noise, Gaussian power-spectrum contractions, connected four-point terms, and super-sample responses for its current observables. Those numerical and survey ingredients are useful. The statistical order changes when the observable contains three fields.

This follows directly from the definition of covariance. If $`x`$ represents a field value, the covariance of a quadratic and cubic estimator contains an average of five field factors; the covariance of two cubic estimators contains six. Schematically,

```math
\begin{aligned}
C_{23}&\sim \langle x^2x^3\rangle
             -\langle x^2\rangle\langle x^3\rangle,\\
C_{33}&\sim \langle x^3x^3\rangle
             -\langle x^3\rangle\langle x^3\rangle.
\end{aligned}
```

The notation suppresses positions, bins, angular filters, and survey weights. It does not mean that one can compute the covariance from a single scalar fifth or sixth moment. Those omitted indices are what make the calculation expensive.

For a zero-mean Gaussian field, odd moments vanish. Its ideal quadratic–cubic cross-covariance is therefore zero. The same Gaussian field has a nonzero variance of a cubic estimator through products of three two-point functions, even though its mean cubic statistic vanishes. Nonlinear structure adds products involving power spectra and bispectra, connected four-point correlations, and higher connected correlations. Shape noise contributes through estimator-specific contractions and shared galaxy indices.

| Covariance block | Correlation content to account for | Consequence for the proposed module |
|---|---|---|
| $`C_{22}`$ | Gaussian pairings and connected four-point correlations | Existing shear covariance is the starting point, with its documented IA and mask scope |
| $`C_{23}`$ | Power-spectrum–bispectrum products and connected five-point correlations after mean subtraction | Extend the published aperture cross-covariance to tomography and the selected two-point estimator |
| $`C_{33}`$ | Gaussian three-pair contractions, power-spectrum–trispectrum products, bispectrum products, and connected six-point correlations | Port the published analytic terms, then match tomography and the measured aperture operator |
| Super-sample terms | Responses of each observable to modes larger than the survey | Requires the bispectrum or aperture response as well as the power response |

Mean subtraction removes the pairing corresponding to the product of the two estimator means. It does not remove every product of two bispectra from $`C_{33}`$. Likewise, simply labeling the extra block “non-Gaussian” does not specify which contractions, noise terms, or long-wavelength contributions it contains.

There is relevant work to use before deriving everything from the beginning. Linke and collaborators develop an analytic covariance for third-order aperture statistics and examine the relation between convergence-map and shear-correlation estimators. Wielders and collaborators study the cross-covariance of second- and third-order aperture masses in a non-tomographic setting. The latter is not a ready-made tomographic $`\xi_\pm`$–aperture covariance: the statistic, bin labels, and survey estimator must be translated and validated. [Third-order covariance](https://arxiv.org/abs/2212.04485), [aperture cross-covariance](https://arxiv.org/abs/2509.20443)

Their public [threepoint code](https://github.com/sheydenreich/threepoint) supplies the concrete source for the CPU port. Its CUDA implementation is not a hardware requirement: [Chapter 15](15_analytic_covariance_cpu_port.md) maps individual terms, records source corrections, and derives proposed filtered-correlation and one-halo reductions. [Chapter 17](17_cosmolike_covariance_cpu_transfer.md) connects those reductions to existing CosmoLike covariance preparation and integration.

## 3. Choose a covariance route appropriate to each milestone

For reproduction of the first paper, use its selected simulation covariance, compression, mask, and likelihood prescription. The integrator repository contains several covariance generations; a filename containing a date or an approximate number of simulations is not enough to select the publication input. Match the configuration and obtain Rafael's confirmation of the exact run manifest. The inspected publication-like configuration selects 796 realizations and the Percival likelihood, while its direct prediction modules are replaced by an emulator. [Pinned configuration](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/blind-metacal-796sims-february-Percival/blind-metacal-shear-2pt-map3-NLA-all.ini)

For an initial DES Y6 exploratory forecast, a Gaussian covariance can diagnose whether the mean and derivatives contain useful information and expose indexing errors. Label that result by its covariance approximation. It cannot establish the final information gain if non-Gaussian covariance or cross-covariance has not been bounded. Rescaling a Y3 covariance by survey area alone is also an exploratory calculation: changes in source distributions, galaxy density, shape noise, angular selection, and survey geometry affect its blocks differently.

For the full-vector forecasts, compare an analytic covariance with an analytic model corrected using simulations. Precision-matrix expansion supplies a specific version of the latter approach; it estimates corrections to the precision around an analytic model and avoids directly inverting a noisy sample covariance. Its application to the joint statistic is developed in [Chapter 18](18_precision_matrix_expansion.md), following [Friedrich and Eifler](https://arxiv.org/pdf/1703.07786). A conventional matched-mock covariance remains a comparison where its simulation count permits it. Simulations must measure the two- and three-point vectors together to estimate their correlations. All routes need the same estimator windows and a record of long-wavelength contributions, so that an added super-sample term is not counted twice.

A useful diagnostic is the conditional covariance

```math
C_{3|2}=C_{33}-C_{23}^{\mathsf T}C_{22}^{-1}C_{23}.
```

This is the variance of the three-point vector remaining after its linear correlation with the two-point vector is accounted for. It must be positive definite on the retained independent data subspace. The corresponding derivative also changes:

```math
J_{3|2}=J_3-C_{23}^{\mathsf T}C_{22}^{-1}J_2,
\qquad
F_{\rm data}=J_2^{\mathsf T}C_{22}^{-1}J_2
+J_{3|2}^{\mathsf T}C_{3|2}^{-1}J_{3|2}.
```

Here $`J_2`$ and $`J_3`$ contain the parameter derivatives of the two mean vectors. For a fixed Gaussian covariance, this identity separates the two-point information from the extra information in the conditional three-point response. A smaller conditional variance alone does not establish a larger gain, because the conditional response can also shrink. These quantities diagnose a supplied covariance; they do not estimate the missing cross-covariance.

## 4. Compression changes all covariance blocks

The initial Y3 and Y6 Fisher forecasts should use the full two-point vector after the stated scale cuts. Compression is not needed to calculate a Fisher matrix of this size and is not part of the proposed C acceleration. It enters the study because the published Y3 analysis compresses 227 retained two-point entries to 16 combinations while retaining 80 aperture entries, reducing the joint dimension from 307 to 96. With a simulation-estimated covariance, a smaller dimension can reduce noise in its inverse and the resulting uncertainty in parameter constraints. With a sufficiently accurate covariance, that statistical motivation may disappear. Compression can still require the full theoretical vector before the final matrix multiplication, so it does not automatically save bispectrum or two-point prediction time.

Retain the paper's compression for its reproduction. For the separate Fisher forecasts, use the full vector as the starting reference and compare an explicitly designed compression under the same covariance assumptions and simulation budget. Compression and simulation correction can also be combined. The remaining discussion specifies how to measure the effect on joint information rather than assume it is preserved.

Suppose a fixed matrix $`W_2`$ compresses the two-point data and a matrix $`W_3`$ compresses the three-point data. The full transformation is block diagonal, and the compressed cross-covariance is

```math
C'_{23}=W_2 C_{23} W_3^{\mathsf T}.
```

Compressing only the diagonal covariance blocks is inconsistent. The same matrices must transform the mean, covariance, and derivatives, and their row order must be recorded. For a local Fisher forecast, keep the compression weights fixed at the chosen fiducial model. Rebuilding the weights at each derivative displacement changes the summary statistic while differentiating it.

MOPED constructs a small set of weighted data combinations from parameter derivatives. Its local information-preservation statement assumes the relevant model and covariance conditions; it is not a guarantee over an arbitrarily broad posterior or across a changed IA parameter set. Reusing an NLA compression for a TATT forecast requires checking that the extra TATT derivative directions are retained. Recomputing a TATT compression is an analysis choice to validate, not a harmless relabeling of the old vector.

Even a compression that preserves the two-point Fisher matrix can lose information in the joint analysis. A discarded two-point combination may have zero derivative of its own mean but fluctuate in correlation with the aperture data. Observing that fluctuation helps predict part of the aperture noise, reducing its conditional variance. Dropping the combination removes that information. Compare the full joint Fisher matrix with the block-compressed result, including the cross-covariance, before claiming that two-point-only MOPED preserves the joint constraints.

## 5. Finite simulation covariance and a Fisher approximation are different statistical objects

If an estimated covariance enters a Student-type likelihood, the likelihood is not a Gaussian with that matrix's inverse precision. In the inspected integrator branch, the Percival option evaluates a function of the form

```math
\ln L=-\frac{\nu_L}{2}\ln\left(1+
\frac{Q}{N_s-1}\right)+\text{constant},
\qquad
Q=(d-\mu)^{\mathsf T}\widehat C^{-1}(d-\mu).
```

$`N_s`$ is the simulation count, $`\widehat C`$ is the estimated covariance, and $`\nu_L`$ is the likelihood exponent determined by the chosen prescription and dimensions. This notation deliberately distinguishes the exponent from a shear-calibration parameter. The covariance-estimation treatment and its assumptions should be taken from the statistical model, rather than stacked as several unrelated correction factors. [Percival and collaborators](https://arxiv.org/abs/2108.10402), [implemented likelihood](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/joint_likelihood.py#L211)

At a noiseless fiducial point, expanding the logarithm gives a local curvature proportional to $`\nu_L/(N_s-1)`$ times $`J^{\mathsf T}\widehat C^{-1}J`$, where $`J`$ is the derivative of the mean vector. That is a curvature calculation at zero residual. It is not automatically the expected Fisher information over data realizations, nor the covariance of the full posterior with finite parameter ranges. The 2025 paper's posterior results should therefore be reproduced with the appropriate likelihood when that milestone is reached. A Fisher study can reproduce its local response and explain its complementarity without claiming to reproduce its posterior FoM exactly.

The covariance-estimation model also needs independent realizations and the statistical assumptions underlying the chosen Wishart-based treatment. Repeated or rotated views of one simulated volume are not automatically independent samples. Record whether compression weights were fixed independently of the same realizations used to estimate the compressed covariance; learning those weights from the same noisy covariance changes the statistical problem. The number of files alone does not establish the effective independent simulation count.

## 6. Derivative accuracy determines whether an acceleration is useful for forecasts

For a fixed covariance and Gaussian local forecast, use

```math
F_{\alpha\beta}
=J_\alpha^{\mathsf T} C^{-1}J_\beta
+F^{\rm prior}_{\alpha\beta},
\qquad
J_\alpha=\frac{\partial\mu}{\partial p_\alpha}.
```

The parameter labels $`\alpha,\beta`$ cover cosmological and nuisance parameters. A derivative is a small difference between predictions, so an error that is harmless for a single mean can dominate the derivative when the step becomes small. The acceptance tests must therefore include a step-size study and convergence of derivatives under finer quadrature and transform settings. Agreement of a few plotted mean curves is insufficient.

A Cholesky factorization writes $`C=L L^{\mathsf T}`$. Solving $`L y=\delta\mu`$ expresses a numerical mean error in units that account for correlated observational uncertainty. Then $`y^{\mathsf T}y=\delta\mu^{\mathsf T}C^{-1}\delta\mu`$ is a covariance-weighted error, denoted $`\Delta\chi^2_{\rm num}`$. Use the same solves for derivative columns; do not form a dense inverse merely for convenience.

For a proposed numerical tolerance, examine both this full-vector error and the induced local parameter shift

```math
\delta p=F^{-1}J^{\mathsf T}C^{-1}\delta\mu.
```

For the positive sign displayed here, define $`\delta\mu=\mu_{\rm reference}-\mu_{\rm numerical}`$: the data follow the reference and are fitted with the numerical model. Reversing the residual definition reverses the shift. This expression assumes a locally linear mean and the specified fixed covariance. It identifies errors aligned with informative parameter directions. A small shift does not alone bound errors in the Fisher curvature, so also compare marginalized uncertainties, parameter correlations, and the chosen FoM across refinement levels.

Working engineering targets could be $`\Delta\chi^2_{\rm num}<0.01`$, induced shifts below `0.05` of a marginalized standard deviation, and changes below 0.5% in the headline FoM under a further numerical refinement. These are proposed targets to agree upon after the survey covariance is selected, not measured results or universal physical thresholds. They supplement full unmasked component comparisons and do not replace the repository's existing regression tolerances. The tolerance for physical model error must be assessed separately.

Use one fixed set of radial nodes, angular bins, and masks across a finite-difference stencil whenever possible. An adaptive algorithm that changes its accepted grid as a parameter moves can create derivative noise unrelated to physics. If adaptation is needed, establish that it converges smoothly and record its settings. Scale cuts used in a forecast should ordinarily be fixed at the fiducial analysis definition; differentiating a moving cut differentiates a changed data selection.

## 7. Some nuisance derivatives can be computed without repeating the expensive calculation

The integrator applies multiplicative calibration to a three-bin statistic through

```math
\mu_{ijk}^{\rm obs}=(1+m_i)(1+m_j)(1+m_k)\,\mu_{ijk}.
```

If a bin appears $`n_a`$ times in the triplet, the derivative with respect to its calibration is $`n_a\mu^{\rm obs}/(1+m_a)`$. For the triplet $`(2,2,4)`$, the factors are squared for bin 2 and linear for bin 4. This derivative can be checked analytically without rerunning the bispectrum or FFT stages. [Calibration implementation](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/map3_m_bias.py#L9)

An amplitude-basis decomposition of TATT offers related savings for $`A_1`$, $`A_2`$, density weighting, and the chosen velocity-shear ratio. It must preserve every mixed amplitude product and the nonlinear-alignment assembly. Redshift-exponent and photo-z derivatives change radial weights; retaining the expensive physical basis can still help. Cosmological derivatives generally require new spectra, distances, and projections. Report these costs separately rather than quoting a single time for every parameter.

The fiducial parameterization also matters. At $`A_2=0`$, a redshift exponent multiplying only that amplitude can be locally unconstrained. At $`A_1=0`$, a velocity-shear coefficient defined as $`C_1`$ times a ratio loses its independent effect. A rank-deficient Fisher matrix in these limits may reflect the model rather than bad numerical derivatives. Diagnose the null direction explicitly; do not hide it with an unexplained eigenvalue cutoff. A different independent velocity-shear amplitude is a scientific extension to be documented.

## 8. DES Y6 inputs and physical modeling need their own specification

The official release links the Y6 cosmology papers and data products. The cosmic-shear paper uses four source bins and examines NLA and TATT. This supports the proposed survey application, but it does not supply a validated Y6 three-point covariance or make Y3 scale cuts transferable. The forecast must identify the actual sample, mask, redshift distributions, calibration priors, and retained angular measurements. [DES release](https://www.darkenergysurvey.org/des-y6-cosmology-results-papers/), [Y6 cosmic shear](https://arxiv.org/abs/2602.10065)

| Input or modeling choice | What the forecast record must state | Why it affects the result |
|---|---|---|
| Source distributions | File version, normalization, tomographic definition, and allowed shifts or shape changes | Three lensing/source kernels multiply in the projection |
| Shape noise and density | Effective density and ellipticity convention for each selected bin | The cubic estimator has several signal/noise contractions |
| Angular footprint | Actual sample mask and effective area | Covariance and long-mode responses depend on geometry |
| Shear calibration | Priors, correlations, and response convention | Three-point means contain three multiplicative factors |
| Angular estimator | Triangle binning, aperture filter, cuts, pair/triplet weights | Theory must predict the measured compressed quantity |
| Matter spectra | Nonlinear model, units, neutrino prescription, allowed cosmological domain | Bispectrum derivatives are sensitive to assumptions outside the fit's calibration |
| Alignment model | NLA, TATT, or the specified velocity-shear extension, with shared parameters | Added three-point information can constrain or be degraded by these directions |
| Baryonic treatment | Bispectrum correction or excluded scales, with validation cases | A power-spectrum suppression is not a general bispectrum prescription |
| Covariance | Source, approximations, cross-blocks, and simulation-count treatment | Absolute errors and relative gains depend on all of them |

A useful first cosmological target is the $`\Omega_m`$–$`S_8`$ plane used in the reproduction study, where $`S_8=\sigma_8\sqrt{\Omega_m/0.3}`$ combines matter density and the present linear fluctuation amplitude on $`8\,h^{-1}{\rm Mpc}`$ spheres. Constant-$`w`$ forecasts require matching the chosen bispectrum model's calibrated domain. A $`w_0`$–$`w_a`$ or freely varying neutrino-mass extension needs additional validation of the bispectrum response; successful calls to a background or power-spectrum provider do not establish that response. BiHalofit was calibrated over specified cosmological models, and its accuracy is finite even when its numerical evaluation is exact. [BiHalofit calibration](https://arxiv.org/abs/1911.07886)

Baryonic corrections deserve tests on squeezed and unequal-side triangles, not just equilateral configurations. Reduced shear, lensing selection, source clustering, and beyond-Born effects should be evaluated as separate contributions where the retained scales make them relevant. Limber and flat-sky approximations should be checked on the selected angular range. CosmoLike's non-Limber two-point engine supplies numerical techniques, but not an exact three-point projection by substitution.

For each survey, the desired scientific result is an information-gain table that states both absolute marginalized errors and the ratio of joint to two-point FoM, under several specified IA, baryon, covariance, and scale choices. Define $`{\rm FoM}=1/\sqrt{\det{\rm Cov}(p_a,p_b)}`$ using the marginalized two-parameter covariance. Keep a distinct row group for Y3 and Y6; compare their gains only after stating any differences in priors, scale cuts, or physical models. A large ratio accompanied by an unstable absolute covariance, a missing derivative direction, or a changed prior is not evidence that the three-point measurement has delivered that much information.
