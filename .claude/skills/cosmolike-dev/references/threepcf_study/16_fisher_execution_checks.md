# 16. Checking what the Fisher calculation actually computes

> **Archive status.** This source study records design proposals and validation requirements, not completed features. Read the [maintained implementation status](../threepcf.md) first. Both the mean and its covariance belong in `cosmolike/3pcf/`; earlier destination proposals using `fastnc/` or `covariances/` are superseded.

## 1. Inspect the sampler as well as the likelihood

The proposed Y3 and Y6 forecasts should use the full selected two-point vector, a matched three-point vector, and their joint covariance or precision estimate. Whether that matrix comes from an analytic calculation, simulations, or precision-matrix expansion is a separate specification. The code that turns derivatives into a Fisher matrix must consume the intended statistical object. The presence of a likelihood expression elsewhere in the pipeline does not establish that the Fisher sampler differentiates that expression.

This chapter inspects CosmoSIS commit [`1a6c123095e45895d44f85cff93211192b62b2b0`](https://github.com/joezuntz/cosmosis/tree/1a6c123095e45895d44f85cff93211192b62b2b0), specifically its [Fisher sampler](https://github.com/joezuntz/cosmosis/blob/1a6c123095e45895d44f85cff93211192b62b2b0/cosmosis/samplers/fisher/fisher_sampler.py) and [derivative calculation](https://github.com/joezuntz/cosmosis/blob/1a6c123095e45895d44f85cff93211192b62b2b0/cosmosis/samplers/fisher/fisher.py). This source pin is not evidence of the CosmoSIS version used for either paper. A reproduction needs that environment separately. No sampler was executed here.

The inspected sampler concatenates the theory arrays for the listed likelihoods and builds a block-diagonal precision from their exported inverse-covariance arrays. It differentiates the theory and forms $`J^{\mathsf T}\Psi J`$, then adds Gaussian prior precision. Rafael's joint configuration correctly lists only `joint`, allowing its exported matrix to retain the two-point–three-point cross-block. Its Fisher wrapper includes that configuration and selects the Fisher sampler. Listing `2pt`, `map3`, and `joint` as three likelihoods would count the same data repeatedly; listing only the first two would discard their correlation. Preserve the single joint entry when installing a different mean or covariance backend. [Joint configuration](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/config/inference/joint-2pt-map3.ini), [Fisher wrapper](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/config/fisher/joint-emulator.ini)

## 2. The optional finite-simulation likelihood needs a distinct Fisher choice

The supplied defaults use a Gaussian likelihood with covariance treated as known. Its optional Percival path instead evaluates

```math
\ln L=-\frac{m}{2}\ln\left(1+\frac{Q}{N_s-1}\right),
\qquad Q=(d-\mu)^{\mathsf T}\widehat C^{-1}(d-\mu).
```

At zero residual, this expression has local curvature $`mJ^{\mathsf T}\widehat C^{-1}J/(N_s-1)`$. The Y6 module exports the unscaled $`\widehat C^{-1}`$ on that path. The inspected CosmoSIS sampler consumes that exported matrix directly. Thus these two inspected code paths, if combined with `percival = T` and valid simulation and free-parameter counts, produce the unscaled mean-response Fisher matrix, not the zero-residual curvature of the configured Percival likelihood. Missing counts instead cause the supplied likelihood to raise an error. The source chain establishes the conditional mismatch; it does not show that a published run or the Gaussian default is affected. [Likelihood implementation](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/desy6_3pcf/likelihood.py), [default options](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/config/common/map3.ini)

Choose explicitly between a fixed-Gaussian Fisher forecast, zero-residual likelihood curvature, and an expected information or posterior calculation. These are different statistical targets. A scalar rescaling can match that particular local curvature, but it does not turn it into the full finite-simulation posterior. Precision-matrix expansion likewise needs an interface that accepts its defined precision estimate rather than automatically applying a sample-covariance correction intended for another estimator.

## 3. The derivative step is a fraction of the parameter range

The inspected sampler normalizes varied parameters to their allowed ranges before using its finite-difference stencil. Its default step of 0.01 therefore means one percent of that range, not one percent of the fiducial value. Four displaced calculations at $`\pm h`$ and $`\pm2h`$ give

```math
\frac{\partial\mu}{\partial p}\simeq
\frac{-\mu(p+2h)+8\mu(p+h)-8\mu(p-h)+\mu(p-2h)}{12h}.
```

For the shipped $`\Omega_m`$ range 0.12–0.50, the physical step is 0.0038. For an IA exponent spanning −5 to 5, it is 0.1. Those choices may be reasonable starting points, but only a step/refinement study establishes a stable derivative. Changing a prior range changes the numerical step even if the interior physical model is unchanged. Record physical steps, not only the shared configuration value. [Parameter ranges](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/config/values/lcdm-tatt.ini)

There is a second implementation detail when introducing an analytic covariance provider. The default stencil code obtains the covariance from its first displaced point, not the central point. This has no effect when every point loads the same fixed matrix. It matters if a replacement provider recomputes covariance from the displaced cosmology. A fixed-covariance forecast should prepare the fiducial matrix explicitly and reuse it. A parameter-dependent covariance forecast requires its own likelihood and derivative prescription; it is not obtained by accidentally changing the matrix used by a mean-only Fisher contraction.

## 4. The shipped TATT fiducial has an exact null direction

The TATT values file varies both $`A_2`$ and its redshift exponent $`\alpha_2`$, with fiducial $`A_2=0`$. If the coefficient has the specified form $`C_2(z)=A_2 f(z,\alpha_2)`$, then

```math
\left.\frac{\partial C_2}{\partial\alpha_2}\right|_{A_2=0}=0.
```

Neither the two-point nor three-point mean can constrain that exponent locally through this coefficient when its amplitude vanishes. The supplied prior file contains placeholder calibration and redshift priors, but no prior restoring this exponent's local information. The physical TATT model therefore has a null derivative direction at these inputs. This is a concrete instance of the rank issue described in Chapter 5, not a reason to insert an unexplained numerical floor. [TATT fiducial](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/config/values/lcdm-tatt.ini), [prior file](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/config/priors/des-y6-priors.ini)

The generic neural network accepts $`A_2`$ and $`\alpha_2`$ as separate coordinates and does not enforce this identity algebraically. Its returned derivative can therefore be nonzero through approximation error. Require that derivative to vanish within the calibrated emulator error, then check its effect on the Fisher eigensystem. No trained network was evaluated here, so this finding does not establish that a saved numerical Fisher matrix is exactly singular. It identifies information that such a matrix must not obtain from emulator error. [Training coordinates](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/config/emulator/designs.yaml), [emulator prediction](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/desy6_3pcf/emulator.py)

Use a stated nonzero-amplitude fiducial if the question is local sensitivity to that exponent, or explicitly fix or otherwise constrain the unidentifiable direction. A posterior calculation can describe nonlinear amplitude–exponent behavior that a Fisher matrix at zero amplitude cannot. The chosen treatment must be identical when comparing two-point-only and joint results. Chapter 12 adds a second rank diagnostic: at fixed redshift, the transverse tree-level kernels depend on the quadratic combination $`C_{1\delta}+C_2/3`$. Their separate survey constraints must be traced to the assumed redshift evolution and additional two-point information.

## 5. Forecast uncertainty must include the matrix-estimation decision

For a deterministic covariance model at the fiducial point, a repeated Fisher calculation with smaller derivative steps can isolate derivative error. For a simulation-corrected precision, changing the simulation subset adds another source of variation. Keep those exercises separate: first fix the precision estimate while refining the numerical derivatives; then hold the derivatives fixed while studying precision-estimation scatter and convergence. This avoids mistaking a noisy covariance correction for a noisy C mean prediction.

Compare two-point, three-point, and joint results with common parameter coordinates and priors. For each, record the retained data rows, matrix construction, derivative steps, null directions, and numerical tolerances. Report absolute marginalized errors as well as FoM ratios. If an analytic-plus-simulation correction changes the gain, inspect whether the cause is the diagonal blocks, cross-correlations, or a poorly modeled covariance direction. A hybrid method does not require compression by definition; compression can be a separately tested way of reducing the number of directions that need accurate matrix estimation.

The proposed numerical acceptance is an end-to-end statement: refining the mean and covariance calculations should leave the forecast stable to the agreed tolerance. The proposed physical acceptance is separate: changing allowed IA, halo, baryon, window, and covariance assumptions should reveal whether the additional information remains credible. A well-conditioned matrix computed quickly cannot substitute for either check.
