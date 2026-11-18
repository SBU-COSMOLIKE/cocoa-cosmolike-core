# Stress tests of the proposed reductions

> **Archive status.** This source study records design proposals and validation requirements, not completed features. Read the [maintained implementation status](../threepcf.md) first. Both the mean and its covariance belong in `cosmolike/3pcf/`; earlier destination proposals using `fastnc/` or `covariances/` are superseded.

The last pass asks whether the reductions in Chapters 19–21 still calculate
the intended observable when assumptions are changed. An algebraic identity
can be exact for a numerical prediction and still leave the covariance of
the measured statistic unspecified. This distinction determines which
parts of the port can be implemented independently and which need a joint
definition of the measurement and its fluctuations.

## 1. Three objects that must remain distinct

The unrestricted aperture moment integrates a filtered shear field over
all triangle sizes. The numerical fastnc calculation evaluates natural
components on a finite grid and contracts them with aperture weights.
The catalogue estimator averages noisy galaxy products with actual pair
or triplet weights, angular bins and a survey mask. These three objects
approach one another only under stated approximations and numerical
limits. In particular, the Y6 configuration's comment about having no
scale cuts refers to selecting aperture-radius rows. Its triangle grid
still extends only from one to 150 arcminutes. [Configured domain](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/config/common/map3.ini#L9-L30),
[row selection](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/modules/map3_binning.py#L18-L25).

Let $`h`$ contain the coupled Fourier multipoles, $`T`$ the fixed numerical
transform to natural components, and $`A_3`$ their fixed aperture
contraction. Chapter 21's dual weights preserve

```math
\mu_3=A_3Th=(T^T A_3^T)^T h.
```

All complex arrays are understood here as stacked real and imaginary
parts. This is a statement about a deterministic mean prediction. It does
not establish a random catalogue quantity $`\widehat h`$ satisfying
$`\widehat\Gamma=T\widehat h`$. For example, a measurement could obey
$`\widehat\Gamma=T\widehat h+\eta`$, where $`\eta`$ has zero mean and
independent covariance $`Q`$. Its predicted mean is unchanged, but its
aperture covariance acquires $`A_3QA_3^T`$. An exact dot-product test of
the numerical transpose cannot detect that missing term.

The measurement-level identities remain

```math
C_{33}=A_3\operatorname{Cov}(\widehat\Gamma)A_3^T,
\qquad
C_{\xi3}=\operatorname{Cov}(\widehat\xi,\widehat\Gamma)A_3^T.
```

They require the natural-component covariance and its cross-covariance
with the selected two-point estimator. They are not supplied by the mean
transform. The Y6 aperture code establishes the linear contraction;
the earlier integrator loads a covariance product. Neither operation
derives the missing fluctuations. [Aperture contraction](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/desy6_3pcf/map3.py#L98-L126).

Even agreement of unrestricted and truncated means does not bound their
covariance difference. If a truncated statistic differs by
$`\epsilon q`$ with $`\langle q\rangle=0`$, its variance changes by
$`2\epsilon\operatorname{Cov}(d,q)+\epsilon^2\operatorname{Var}(q)`$.
Consequently, the finite-domain covariance needs its own comparison.
This is consistent with the estimator dependence discussed in
[Linke et al., Section 4](https://arxiv.org/abs/2212.04485) and the
map/window definitions in
[Wielders et al., Sections 2–3](https://arxiv.org/abs/2509.20443).

## 2. Fixed triangle cuts preserve a more general one-halo factorization

There is a constructive route beyond rejecting every cut estimator. Write
a fixed, homogeneous degree-$`p`$ signal estimator as a kernel acting on
$`p`$ shear fields, with $`p=2`$ or $`3`$. Its kernel contains spin
phases, triangle cuts, weights and a deterministic normalization.
Homogeneous means that every term contains exactly $`p`$ fields. In an
additive halo model, evaluate this estimator on the shear field of one
halo with centre $`X`$ and properties $`\lambda`$ such as mass and
distance. Call the result $`R_I(X,\lambda)`$ for observable row $`I`$.

The Poisson one-halo connected cumulant then gives

```math
C^{1h}_{IJ}=\int d\nu(\lambda)\int d^2X\,
R_I(X,\lambda)R_J(X,\lambda).
```

Here $`d\nu`$ is the expected halo count per solid angle in the selected
mass and distance interval. The equation follows because all legs of
this connected contribution belong to the same halo. For any real vector
$`v`$, contraction with $`v_Iv_J`$ gives an integral of
$`(\sum_I v_I R_I)^2`$ with nonnegative measure. Thus this particular
connected contribution is positive semidefinite. This is our extension
of the response construction, not a claim that the inspected source
already implements catalogue-cut responses.

For a translation-invariant estimator with a common large survey area
$`A`$, define its single-halo response shape $`F_I(u;\lambda)`$ and
$`H_I=\int d^2u\,F_I(u;\lambda)`$. Neglecting boundary effects yields

```math
C^{1h}_{IJ}\simeq A^{-1}\int d\nu(\lambda)H_I(\lambda)H_J(\lambda).
```

Fixed triangle cuts change $`F_I`$ and $`H_I`$; they need not destroy this
outer-product structure. They do generally prevent replacing $`F_I`$
by the simple pointwise product of three radial filtered halo profiles.
The expensive step becomes constructing the correct cut response.
With a finite centre window, retain its Fourier weighting and the
possibly nonradial transforms of $`F_I`$. With an internal catalogue mask,
the general $`R_I(X,\lambda)`$ expression remains available but is not
automatically inexpensive.

Data-dependent denominators require a separate derivation or conditioning
on the observed catalogue. Noise subtractions can add terms of lower
polynomial degree. Applying the entire corrected statistic to one halo
and taking its Gram matrix can then mix distinct cumulant orders.
This test must therefore specify the homogeneous signal kernel first.

## 3. A starting point for the full two-point cross-block

For a spherical pure-E halo on the flat sky, let
$`\widetilde\kappa_i(\ell|\lambda)`$ be its convergence profile for
source bin $`i`$. Define $`\overline J_{mb}(\ell)`$ by averaging
$`J_m(\ell\theta)`$ with the actual normalized separation weights in
two-point bin $`b`$. Uniform pair sampling gives a $`\theta d\theta`$
measure; real catalogue weights need not do so. The halo responses are

```math
H_{\xi_+,b}^{ij}=\int_0^\infty\frac{\ell d\ell}{2\pi}
\overline J_{0b}(\ell)\widetilde\kappa_i(\ell)
\widetilde\kappa_j^*(\ell),
\qquad
H_{\xi_-,b}^{ij}=\int_0^\infty\frac{\ell d\ell}{2\pi}
\overline J_{4b}(\ell)\widetilde\kappa_i(\ell)
\widetilde\kappa_j^*(\ell).
```

These responses and $`H_3^{\rm cut}`$ produce the connected one-halo
cross-block through the previous outer product. If the two-point
calculation uses curved-sky spin kernels, use matching kernels or
quantify the flat-sky approximation. Replacing an aperture-variance
radius in upstream Map2–Map3 code does not produce $`\xi_\pm`$.

The power-times-bispectrum and noise contractions additionally depend
on how a pair and a triplet overlap, including shared galaxies and their
weights. Separate pair-count and triplet-count histograms do not fix
that joint overlap. A useful independent limit is a zero-mean Gaussian
field: the two-point–three-point covariance vanishes because it contains
five fields, while the three-point variance contains six fields and
remains nonzero. This limit tests different blocks for different reasons.

## 4. Which source and aperture symmetries survive?

If each halo shear for source $`i`$ is a common profile times a scalar
weight $`W_i(\chi)`$, a common source-independent cut operator gives
$`H_{ijk,R}^{\rm cut}=W_iW_jW_k h_R^{\rm cut}`$. Chapter 19's count of
84 source multisets times 15 equal-aperture radius pairs can therefore
survive fixed common cuts for the isolated connected one-halo block.
The count is 1260 reusable classes, not a prediction of total runtime
or covariance rank.

Compare entries with source triplets $`(111,222)`$ and $`(112,122)`$.
They have the same six-label multiset. They agree under this factorization,
but need not agree if the two measurements use different cuts, masks,
or angular weights. Unequal filter triplets increase the number of
radius-kernel classes beyond 15. Constant per-bin shear calibration can
factor into six multiplicative coefficients; spatial calibration cannot
generally be pulled out. Mean calibration does not by itself prescribe
how measured shape noise changes.

For four identical source kernels and five equal-aperture radii, the
isolated one-halo matrix has at most five distinct responses and rank at
most five, despite having 100 rows. Do not invert that contribution or
add diagonal noise to manufacture information. Inversion concerns the
complete selected covariance after physically specified terms are added.

The additive-field assumption also limits the IA extension. A quadratic
tidal operator acts on the sum of halo tidal fields:
$`(\sum_h s_h)^2=\sum_h s_h^2+\sum_{h\ne h'}s_hs_{h'}`$.
Applying the operator to isolated halos drops the cross-halo pieces.
Pure lensing and a specified deterministic linear NLA prescription can
retain the additive construction. A full TATT covariance needs its mixed
four-, five-, and six-field cumulants and long-mode responses; its mean
coefficients do not determine those quantities.

## 5. Two concrete cache failures to exclude in the port

The pinned TATT coupling constructor records an odd value through
`self.Npsi=(Npsi//2)*2+1`, but constructs the nodes using the original
argument. Requests for 200 and 201 nodes therefore share the recorded
value 201 while producing different arrays. Because that recorded value
enters the cache key, one request can load arrays of the other size.
The usual wrapper always requests 200, so this is a demonstrated source
counterexample, not evidence of a failed published run.
[Constructor and key](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/coupling.py#L113-L151).

Y6's aperture cache is keyed only by filter bytes, although its weights
also depend on the triangle coordinates and spacings. Changing the
coordinates while retaining their dimensions can silently reuse old
weights. Shape validation cannot detect that change.
[Cache key](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/modules/nc_to_map3.py#L20-L28),
[weight dependence](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/desy6_3pcf/map3.py#L98-L126).

Allocation keys must use actual dimensions, including worker scratch
counts. Refill keys must also represent actual coordinate values,
conventions, cut operators and physical inputs. The tests should change
coordinates without changing shape, change odd/even requested sizes,
restore the initial configuration, and compare with a fresh calculation.
Exact Fisher requests must not be merged using a broad numerical
closeness test: doing so can erase the derivative being measured.

## 6. Continuum factorization and discrete reference reproduction

Chapter 20's continuum linear weight $`L=W+C_1n`$ needs a discrete
qualification. The pinned matter-plus-NLA path resamples the source
distribution on a 101-node lens grid, renormalizes it there, and splines
the product with $`C_1`$. The residual TATT path normalizes its shape
window on the original source nodes and applies $`C_1`$ later at
projection nodes. These operations need not yield identical arrays.
[Matter preparation](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/bispectrum.py#L478-L490),
[separate shape window](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/bispectrum.py#L526-L538).

Preserve the actual matter baseline weight separately. The residual
three-vertex identity uses its own actual node weights. Making both
paths use a common continuum discretization is a numerical-method
change and needs a radial-grid convergence comparison. Likewise,
moving a sum through logarithmic interpolation is not a linear
rearrangement. Fixed transforms may be transposed only after these
nonlinear preparation steps.

## 7. Acceptance checks that challenge the assumptions

| Proposed reduction | Independent check | Failure this reveals |
|---|---|---|
| Mean dual weights | Forward/transpose dot products, then full mean and derivative comparisons | Incorrect adjoint, endpoint or component ordering; does not validate covariance |
| Cut one-halo Gram | Direct kernel contraction of connected five-/six-field terms versus single-halo responses | Missing cut, phase or normalization in the response |
| Full $`\xi_\pm`$ cross-block | Actual pair-bin responses and joint pair–triplet contractions | Substitution of aperture variance for a different estimator |
| 1260-class reuse | Shared cuts, then deliberately different cuts at the same dimensions | A cache or symmetry key that ignores the operator |
| Finite windows | General position response versus centre-window simplification | Confusion between a centre mask and masks inside the aperture |
| Noise contractions | Zero noise, analytic white-noise pair function, distinct-object catalogue case | Removing legitimate shared objects or retaining forbidden self-pairs |
| TATT assembly | Seven source labels versus three vertices using identical node weights | Wrong cyclic labels, normalization or radial discretization |
| Joint covariance | Symmetry, full-block positivity, known degenerate limits and refinement | Information manufactured by inversion or inconsistent blocks |

These checks finish the source-and-algebra study. They provide a bounded
implementation plan in Chapters 23 and 24. No new numerical result or
speed measurement is asserted by this pass.
