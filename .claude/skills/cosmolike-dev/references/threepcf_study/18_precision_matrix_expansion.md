# 18. Analytic covariance with simulation corrections: precision matrix expansion

> **Archive status.** This source study records design proposals and validation requirements, not completed features. Read the [maintained implementation status](../threepcf.md) first. Both the mean and its covariance belong in `cosmolike/3pcf/`; earlier destination proposals using `fastnc/` or `covariances/` are superseded.

## 1. Keep the full two-point vector as a scientific option

The proposed Y3 and Y6 analyses should include an uncompressed joint vector of shear correlations, $`\xi_\pm`$, and third-order aperture statistics, $`\langle M_{\rm ap}^3\rangle`$. An analytic covariance supplies a calculable reference for every retained component. Simulations can then estimate its errors. Compression remains an optional comparison; it is not a prerequisite for combining an analytic model with simulations.

[Friedrich and Eifler, *Precision matrix expansion*](https://arxiv.org/pdf/1703.07786), provide a method for estimating corrections directly to the **precision**, meaning inverse covariance. Their second-order estimator corrects the noise bias introduced by multiplying sample covariance matrices. It avoids inverting the sample covariance and assumes Wishart sampling statistics. Finite-order truncation error remains. Their calculation treats covariance as fixed during parameter inference.

For historical reproduction, retain the covariance and likelihood choices actually reported in the Y3 paper. Its Sections III and IV describe a simulation covariance, analytic two-point covariance for compression, and a Percival likelihood. I found no reference to the Friedrich–Eifler PME paper in the inspected bibliography and no use of PME in those methods sections. This identifies the published procedure, not what the authors may have considered privately. [Y3 methods and references](https://arxiv.org/html/2503.03964v2#S3)

The remainder of this chapter develops a proposed application to our joint vector. No covariance correction or forecast has been computed here.

## 2. Measure covariance error relative to the analytic prediction

Let $`C`$ be the unknown true covariance and $`M`$ a positive-definite analytic model. Factor $`M=LL^{\mathsf T}`$ by Cholesky decomposition. Applying $`L^{-1}`$ rescales and mixes the data so that the model covariance becomes the identity. This operation, called whitening, gives a dimensionless measure of the model error:

```math
E=L^{-1}(C-M)L^{-\mathsf T},\qquad
C=L(I+E)L^{\mathsf T},\qquad
\Psi=C^{-1}=L^{-\mathsf T}(I+E)^{-1}L^{-1}.
```

The matrix $`E`$ asks a physical question: how much does the variance of each combination of observables differ from its modeled variance? Comparing diagonal entries alone misses combinations whose individual variances are correct but whose correlations are wrong. An error in the two-point–three-point correlation can affect precisely such a combination.

The geometric series for the middle factor gives

```math
\Psi_{(p)}=L^{-\mathsf T}
\left[\sum_{k=0}^{p}(-E)^k\right]L^{-1}.
```

Because $`E`$ is symmetric, the series converges when every eigenvalue has absolute value below one. Positive definiteness of $`C`$ only guarantees eigenvalues greater than minus one; it does not prevent an eigenvalue above one. For example, if the true variance along a whitened direction is three times the modeled variance, that direction has eigenvalue two and this series fails to converge. Increasing the number of simulations does not repair that modeling error.

Writing $`r=\lVert E\rVert_2<1`$, the truncation error in the whitened inverse obeys

```math
\left\lVert (I+E)^{-1}-\sum_{k=0}^{p}(-E)^k\right\rVert_2
\leq\frac{r^{p+1}}{1-r}.
```

This is a sufficient bound, not a precision requirement inferred from actual DES data. Estimated eigenvalues themselves fluctuate with the simulations. Use repeated independent subsets to distinguish a persistent poorly modeled direction from sample noise, and improve $`M`$ where needed. A positive second-order result is not a convergence test: the deterministic polynomial $`1-\lambda+\lambda^2`$ is positive even for arbitrarily large $`\lambda`$.

## 3. Why squaring a noisy residual gives the wrong correction

Suppose $`C=A+B`$, where $`A`$ is treated as known and $`\widehat B`$ is an unbiased sample covariance from independent realizations of a vector with covariance $`B`$. Let $`M=A+B_m`$. Define $`S=L^{-1}\widehat B L^{-\mathsf T}`$ and $`S_0=L^{-1}B_mL^{-\mathsf T}`$. The estimated residual is $`S-S_0`$, but its square contains the variance of its own estimation error. That extra contribution survives averaging over mock ensembles.

For Wishart sampling with $`\nu=N_s-1`$, the corrected quadratic term and corresponding precision estimate are

```math
\widehat Q=
\frac{\nu^2 S^2-\nu S\,\operatorname{tr}S}{\nu^2+\nu-2},
\qquad
\widehat\Psi_{(2)}=L^{-\mathsf T}
\left[I-(S-S_0)+\widehat Q-SS_0-S_0S+S_0^2\right]L^{-1}.
```

Here $`\widehat Q`$ estimates $`(\mathbb E S)^2`$. This is the whitened form of Equation 12 and Appendix B of the [PME paper](https://arxiv.org/pdf/1703.07786). Its expectation equals the truncated second-order precision, not the exact precision.

The formula requires $`\nu>1`$, but imposes no requirement that the number of simulations exceed the data-vector dimension. Only the analytic $`M`$ is factorized. A singular sample covariance therefore need not prevent the calculation. This removes an inversion barrier; it does not establish that three, thirty, or any specified number of simulations gives adequate precision. Trace terms, correlations, and accumulated uncertainty across many data directions still affect the answer.

Unlike the deterministic polynomial, this noise-corrected estimate is not guaranteed positive definite for every realization. Check its eigenvalues. Silently replacing negative eigenvalues by a floor changes the estimator and its bias. Likewise, applying a Hartlap factor or the original simulation-covariance likelihood prescription on top of this estimator requires a separate derivation. Those prescriptions do not automatically describe its sampling distribution.

An independent diagnostic can avoid the Wishart assumption for this quadratic expectation. Split independent mocks into two disjoint groups, compute unbiased whitened sample covariances $`S_1`$ and $`S_2`$, and define

```math
\widehat Q_{\rm split}=\frac{S_1S_2+S_2S_1}{2},
\qquad
\mathbb E\widehat Q_{\rm split}=(\mathbb E S_1)^2.
```

The identity follows from independence and a common underlying covariance. Substituting this quadratic estimate and $`S=(S_1+S_2)/2`$ into the second-order expression gives another unbiased estimate of its truncation, provided the analytic model is fixed independently. This is an algebraic proposal for comparison, not a result measured here or attributed to the paper's Wishart estimator. Splitting the ensemble changes variance and generally sacrifices some sampling efficiency; compare that cost explicitly. It does not remove truncation error or establish likelihood coverage.

## 4. A joint baseline must retain the cross-covariance

Order the vector as $`d=(d_2,d_3)`$, where $`d_2`$ contains $`\xi_\pm`$ and $`d_3`$ contains aperture moments. The proposed baseline is

```math
M=\begin{pmatrix}M_{22}&M_{23}\\M_{23}^{\mathsf T}&M_{33}\end{pmatrix}.
```

CosmoLike's two-point calculation supplies $`M_{22}`$. The estimator-matched extensions in [Chapter 15](15_analytic_covariance_cpu_port.md) supply $`M_{23}`$ and $`M_{33}`$. Both diagonal blocks and their cross-block enter the whitening and correction. A baseline with zero $`M_{23}`$ is possible if the resulting residual converges; it is not evidence that the true cross-covariance vanishes.

There is a tempting but invalid shortcut: identify $`A`$ with the analytic two-point block alone and identify $`B`$ with everything else. If $`B_{22}=0`$ while $`B_{23}\ne0`$, the resulting $`B`$ has a negative eigenvalue. A random variable with exactly zero variance cannot correlate with another variable. Such a matrix is not a covariance and cannot be assigned the Wishart estimator above. Similarly, subtracting a deterministic analytic block from a full sample covariance does not leave a Wishart-distributed residual.

The initial proposal should therefore use **the full joint sample covariance**, take $`A=0`$, and set $`B_m=M`$. The full analytic calculation still supplies the reference and the invertible matrix. This does not discard its two-point accuracy. A more selective physical signal/noise split can follow if it produces a genuine sample covariance for $`B`$ and a justified calculation of all terms in $`A`$.

The unbiased-estimator argument also treats $`M`$ as fixed. Fitting its halo amplitudes or cross-correlation parameters to the same mocks subsequently used for the correction introduces statistical dependence. Select or fit the baseline using separate information, or validate the entire fitting-and-correction procedure with independent ensembles.

## 5. Shape noise is more than a noise-only covariance

A physical split can be defined through conditional expectation. Let $`s`$ specify a cosmic realization and $`n`$ its random shape noise. For the actual catalog estimator,

```math
C=\underbrace{\operatorname{Cov}_s\!\left[\mathbb E_n(d\mid s)\right]}_{B}
+\underbrace{\mathbb E_s\!\left[\operatorname{Cov}_n(d\mid s)\right]}_{A}.
```

Both terms are covariances. If the estimator's conditional mean equals its noiseless value, noiseless maps estimate $`B`$. This equality must be established for the finite estimator, including repeated-galaxy exclusions and noise subtraction. It cannot be assumed from the name “aperture statistic.”

Field counting explains why $`A`$ requires more than randomized-shape maps. Assume zero-mean signal and independent Gaussian shape noise. Write $`P`$ for the signal spectrum, $`N`$ for noise power, $`B_s`$ for the signal bispectrum, and $`P_4`$ for the connected signal trispectrum. Noise contributions can contain:

| Covariance block | Schematic noise-containing terms | Reason |
|---|---|---|
| Two-point with two-point | $`PN`$, $`N^2`$ | Four fields allow one or two noise pairs |
| Two-point with three-point | $`NB_s`$ | A noise pair leaves three signal fields |
| Three-point with three-point | $`NP_4`$, $`NP^2`$, $`N^2P`$, $`N^3`$ | Six fields permit several signal/noise partitions |

These symbols suppress geometry, source labels, and multiplicities. Some contractions vanish for a specified estimator; retained contractions must follow its distinct-galaxy rules and mask. Non-Gaussian shape-noise moments can require further terms. Randomized shapes with signal removed estimate pure-noise contributions, not the terms that contain signal. Repeated noise seeds on one cosmic map help estimate conditional noise scatter but do not constitute independent cosmic realizations for the Wishart degrees of freedom.

## 6. The proposed Y3 and Y6 validation

The Gaussian assumption concerns the **measured summary vector**. A non-Gaussian density field does not itself disprove it. Conversely, averaging over a large survey does not establish it for every aperture combination: rare massive structures can affect cubic statistics. Inspect held-out mock distributions of whitened components and physically relevant combinations before assuming Wishart covariance fluctuations. A covariance correction cannot repair a wrong mean IA model, missing baryon response, or an estimator mismatch.

For each survey and scale selection, compare the analytic precision, first-order PME, and corrected second-order PME with identical mean derivatives and priors. Vary the number of independent correction mocks; keep additional mocks for validation. Track absolute marginalized errors, changes in the joint/two-point FoM ratio, and scatter between correction ensembles. A stable ratio can conceal a common bias in both absolute forecasts. Resampling an existing ensemble diagnoses sensitivity but does not create additional independent realizations.

The Y3 covariance uses four footprints from each of 199 full-sky maps, yielding 796 measured vectors. Shared large-scale modes can correlate footprints from one parent map; the published construction alone does not establish the size of that correlation. For correction-versus-validation splits, keep each parent map's footprints together and inspect within-parent dependence. Check the underlying simulation provenance for any further shared realizations. Do not assume either 796 or 199 is automatically the correct Wishart count for every covariance direction. [Y3 simulation construction](https://arxiv.org/html/2503.03964v2#S3)

Grouping prevents a parent realization from entering both correction and validation; it does not make the usual covariance estimated from correlated footprints unbiased or Wishart. The split-product diagnostic also needs unbiased covariance estimates in each group. Using one footprint per independent parent provides a comparison with a defined sampling interpretation. An estimator that retains correlated footprints needs its own expectation and sampling analysis.

Mock resolution also sets a physical limit on what the correction learns. The Y3 analysis reports a covariance sensitivity to map pixelization and applies an approximately eight-arcminute lower triangle-side cut. An analytic prediction for finer scales should not be corrected toward an unresolved simulation merely because the correction is statistically well defined. Match the measured estimator and resolution, or establish convergence with adequate simulations before adding those scales. [Y3 estimator and resolution choice](https://arxiv.org/html/2503.03964v2#S3)

The connection to parameter information is explicit. If $`J`$ contains mean derivatives and $`Z=L^{-1}J`$, the fixed-covariance Gaussian Fisher matrix is

```math
F=Z^{\mathsf T}(I+E)^{-1}Z+F_{\rm prior}.
```

This identifies covariance directions affecting cosmology and nuisance parameters. Validate those directions and the remaining residual directions used for goodness of fit. For subsequent posterior work, held-out mocks should test whether intervals cover their input parameters at the stated frequency and whether best-fit scatter matches reported uncertainties. An unbiased estimate of a truncated precision does not guarantee either property. Use the explicit precision interface described in [Chapter 16](16_fisher_execution_checks.md); a cosmology-dependent covariance and its likelihood normalization require a separate calculation.

## 7. Compression can be tested alongside PME

For fixed compression weights $`W`$, the compressed vector and covariances are $`d_c=Wd`$, $`M_c=WMW^{\mathsf T}`$, and $`C_c=WCW^{\mathsf T}`$. Apply the same weights to every mock, retaining the transformed cross-block, then construct PME in that space. In general, $`C_c^{-1}\ne WC^{-1}W^{\mathsf T}`$: projecting a precision is not equivalent to compressing the data and inverting its covariance.

Use weights selected independently of the mocks that validate the correction. Compare the full vector with a jointly designed compression and with the historical partial two-point compression. The latter can preserve two-point information while losing joint information carried through the cross-block. PME addresses covariance estimation; compression removes data directions. Their combination is a proposal to test against the uncompressed calculation, with losses and simulation requirements measured separately for Y3 and Y6.
