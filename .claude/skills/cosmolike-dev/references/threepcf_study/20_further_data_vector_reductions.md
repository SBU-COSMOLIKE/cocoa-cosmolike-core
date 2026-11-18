# 20. Further reductions of the three-point mean

> **Archive status.** This source study records design proposals and validation requirements, not completed features. Read the [maintained implementation status](../threepcf.md) first. Both the mean and its covariance belong in `cosmolike/3pcf/`; earlier destination proposals using `fastnc/` or `covariances/` are superseded.

The requested data vector contains only a few aperture values for each source-bin triplet, whereas the natural-component calculation visits many triangle shapes and angular modes. That difference permits reductions beyond translating the existing loops into C. Two routes should remain distinct. A fixed linear operator can combine the stages of the **measured, binned statistic**, retaining its triangle exclusions; Chapter 21 develops that route. The analytic reductions below apply to **full circular aperture filters**. They provide independent reference calculations and a possible production method only after the relation to the measured estimator has been established.

This chapter derives an exact radial-product representation of the BiHalofit three-halo contribution, reduces the inspected E-mode TATT assembly to three quadratic insertions, and gives an analytic opening-angle integral for selected Gaussian-filter terms. “Exact” describes an algebraic identity under stated assumptions. Finite quadrature, interpolation, FFTLog boundaries, and floating-point summation still introduce numerical error. No scientific program, benchmark, or forecast was executed for this study.

## 1. Define the integral before changing its organization

Let $`a=(i,R_a)`$ denote a source bin and aperture radius. For three projected scalar E fields, the full aperture third moment is

```math
\mu_{abc}=
\int\frac{d^2\ell_1}{(2\pi)^2}
\frac{d^2\ell_2}{(2\pi)^2}
\widetilde U_{R_a}(\ell_1)
\widetilde U_{R_b}(\ell_2)
\widetilde U_{R_c}(|\boldsymbol\ell_1+\boldsymbol\ell_2|)
B^{ijk}_E(\ell_1,\ell_2,\ell_3),
\qquad \boldsymbol\ell_3=-\boldsymbol\ell_1-\boldsymbol\ell_2.
```

The two-dimensional vectors close a triangle. Statistical isotropy removes its absolute orientation, leaving two lengths and their opening angle. The filter used in the inspected covariance repository is $`\widetilde U_R(\ell)=(\ell R)^2\exp[-(\ell R)^2/2]/2`$; $`R`$ must be in radians when $`\ell`$ is an angular multipole. Its zero value at $`\ell=0`$ expresses compensation: a constant projected density contributes no aperture mass. [Filter implementation](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/apertureStatistics.cu#L18-L26)

The displayed physical kernels use the flat-sky conversion $`k_i=\ell_i/\chi`$, so angular and comoving triangles have the same shape. Main also permits $`k_i=(\ell_i+s)/\chi`$ with a user-selected shift $`s`$. A nonzero shift changes the angles inferred from the three comoving lengths. It cannot be inserted into the radial functions while keeping the unshifted harmonic identities below without a new derivation. Treat the shift as part of the model and comparison configuration. [Conversion option](https://github.com/git-sunao/fastnc/blob/e59eab5a07401d2f484bfe00dd48827f2d2162c7/fastnc/projection/geometry.py#L22-L28)

For the Limber model, the projected bispectrum itself contains a radial integral. One can evaluate the aperture response of each three-dimensional physical term at a radial node, then contract those responses with source weights. This interchange is justified for convergent integrals and also for finite, fixed linear quadrature rules. It permits source-bin reuse without storing a bispectrum for every triangle and triplet. The single-source slice convention needs its stated distance prefactor: Rafael's backend explicitly removes a matter-slice factor so that matter and TATT slices share one convention. [Slice normalization](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/desy6_3pcf/fastnc_backend.py#L245-L281)

This identity must not be extended through a nonlinear numerical operation. The TATT branch interpolates the logarithm of the projected matter bispectrum and exponentiates the result. In general, interpolation of the logarithm of a sum differs from summing separately interpolated terms. An aperture response projected after a different interpolation sequence therefore need not reproduce the existing discretized calculation exactly, even when both approximate the same physical integral. Establish agreement under refinement, or keep the original interpolation stage before applying a fixed downstream operator. The FFTLog option that extrapolates logarithms of endpoint values creates a similar boundary for a fixed adjoint. [Matter interpolation](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/bispectrum.py#L949-L994), [optional endpoint extrapolation](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/twobessel.py#L37-L55)

## 2. Assemble the E-mode alignment signal by its quadratic vertex

At the tree order retained by the inspected alignment source, a bispectrum contains two linear fields and one second-order insertion. An insertion means that one vertex supplies a quadratic combination of two density fields. There are three choices of this vertex. The seven names `ddE`, `dEd`, `Edd`, `dEE`, `EEd`, `EdE`, and `EEE` instead classify which observed legs are intrinsic shapes. Both descriptions can encode the same calculation, but the insertion description collects repeated products before forming seven arrays.

Write the canonical lensing weight for source bin $`i`$ as $`W_i(\chi)`$ and its normalized comoving source density as $`n_i(\chi)`$. These definitions use the usual common projection measure $`d\chi/\chi^4`$. Fastnc's internal kernels move factors of distance and scale factor outside the three leg weights, so the conversion must precede a numerical comparison. With a common linear-alignment coefficient $`C_1(z)`$, define

```math
L_i(\chi)=W_i(\chi)+n_i(\chi)C_1(z),
\qquad Q(z)=C_{1\delta}(z)+\frac{C_2(z)}{3}.
```

Here $`L_i`$ is the linear E-field weight, including the signed alignment coefficient. The second definition uses the transverse tensor identity derived in Chapter 12. It collects the density-weighted and quadratic tidal kernels at this perturbative order; it does not identify their distinct redshift laws or two-point responses. [Internal kernel convention](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/bispectrum.py#L451-L490), [alignment coefficients](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L600-L613)

For the active source normalization, after subtracting its linear-alignment tree contribution, the weighted E-only integrand can be organized as

```math
\mathcal B^{ijk}=
L_iL_jL_k B_{\rm matter}
+\sum_{\mathrm{cyc}(i,j,k)}
n_i L_jL_k\,
P_jP_k\,[Q\,S_{jk\to i}+V_{jk\to i}].
```

In each term, the ordered source weights and Fourier legs move together. $`P_j=P(k_j,z)`$ uses the alignment source's specified power prescription. It need not equal the power used internally by BiHalofit. $`S`$ is the transverse quadratic E shape defined below. For the inspected velocity-shear implementation, $`V_{jk\to i}=-(2/7)C_t(1-\mu_{jk}^2)`$, where $`\mu_{jk}=\widehat{\boldsymbol k}_j\cdot\widehat{\boldsymbol k}_k`$ and $`C_t`$ means the internal coefficient after the source multiplies its input ratio by $`C_1`$. Chapter 12's physical normalization question remains applicable.

This expression follows directly from the active assembly. The quadratic part of `ddE` is $`P_1P_2[QS+V]`$. The corresponding insertion in `dEE` acquires one factor of $`C_1`$, and in `EEE` it acquires two. Expanding $`n_i(W_j+n_jC_1)(W_k+n_kC_1)`$ produces precisely those label choices. The matter-plus-NLA baseline supplies the removed alignment-only contribution with its chosen nonlinear matter model. These statements concern the source normalization; no extra Wick factor should be inserted while implementing the reduction. [Density–density–shape assembly](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L299-L317), [two-shape assembly](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L322-L351), [three-shape assembly](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L353-L394), [E projection and subtraction](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L625-L748)

The upstream projected calculation already sums the seven E components before its natural-component transform when no component is selected. Thus seven eliminated FFT chains are **not** a saving attributable to this proposal. The additional saving is earlier: evaluate three insertion shapes, share their power products, and contract their collected weights without retaining seven weighted radial integrands. The separate matter and TATT calculations in the Y6 direct module can also share the downstream linear transform after each has reached an identical multipole grid. Preserve their separate interpolation choices and subtraction bookkeeping before that point. [Existing component sum](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/bispectrum.py#L838-L869), [two calculator paths](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/modules/fastnc_interface.py#L119-L127)

For exact reproduction of the pinned discretization, keep the matter baseline weight separate from the residual IA weight. The baseline resamples and normalizes on its lens grid, whereas the residual uses source-grid normalization and applies the IA coefficient later. A common continuum weight does not make these discrete arrays identical; [Chapter 22](22_stress_test_of_reductions.md) gives the source evidence and required radial-convergence check. The three-vertex residual identity applies to its actual node weights.

The identity assumes the stated shared coefficients, transverse tree kernels, and E-only observable. Bin-dependent IA parameters must stay with their vertices. Single-B components and higher-order contractions require their own assembly. Compare the collected formula against each active tensor component before projection, using unequal leg lengths and unequal source distributions. A comparison only at equal bins would conceal misplaced weights.

## 3. Replace separable triangle integrals by radial products

Consider one bispectrum contribution whose filtered angular dependence is

```math
f(\ell_1)g(\ell_2)h(\ell_3)
\cos[m(\varphi_1-\varphi_2)].
```

“Separable” means that each radial factor depends on only one side length. The opening-angle harmonic is kept explicitly. Define three one-dimensional Hankel transforms. For order $`m`$, these are the radial parts of Fourier transforms of angular harmonics; order zero describes a rotationally symmetric field:

```math
F_m(r)=\int_0^\infty\frac{\ell\,d\ell}{2\pi}f(\ell)J_m(\ell r),
\quad
G_m(r)=\int_0^\infty\frac{\ell\,d\ell}{2\pi}g(\ell)J_m(\ell r),
\quad
H_0(r)=\int_0^\infty\frac{\ell\,d\ell}{2\pi}h(\ell)J_0(\ell r).
```

The exact product identity is

```math
\int\frac{d^2\ell_1d^2\ell_2}{(2\pi)^4}
f(\ell_1)g(\ell_2)h(|\boldsymbol\ell_1+\boldsymbol\ell_2|)
\cos[m(\varphi_1-\varphi_2)]
=2\pi(-1)^m\int_0^\infty r\,dr\,F_m(r)G_m(r)H_0(r).
```

To see the mechanism, express $`h(|\boldsymbol\ell_1+\boldsymbol\ell_2|)`$ as the Fourier transform of $`H_0(r)`$. Its exponential becomes a product of one factor involving $`\boldsymbol\ell_1`$ and another involving $`\boldsymbol\ell_2`$. The angular integral of each factor produces $`J_m`$. Their phases multiply to $`(-1)^m`$, and integration over the direction of $`\boldsymbol r`$ supplies $`2\pi`$. The sign is positive for $`m=0,2`$ and negative for $`m=1`$; dropping it would reverse the dipole contribution.

This identity applies to arbitrary circular filters when the radial integrals exist. It does not require a Gaussian filter or a fit to a finite collection of triangle shapes. Multiplying each leg by its own radial Fourier cutoff preserves the identity. A window depending jointly on side ratios or an exclusion imposed on real-space shear triangles generally destroys the factorization. The same distinction applies to baryonic modifications: a product of three one-leg factors can be absorbed into $`f,g,h`$; a general triangle-dependent ratio cannot.

### 3.1 The BiHalofit three-halo contribution meets this condition

At fixed redshift, write the source's damping as $`D(k)=[1+e_n q]^{-1}`$, where $`q=k r_\sigma`$, and its effective radial power as $`P_E(k)`$. Its three-halo term is

```math
B_{h3}=\sum_{\rm cyc}
2\,[F_2(k_1,k_2,k_3)+d_nq_3]
[D(k_1)P_E(k_1)]
[D(k_2)P_E(k_2)]D(k_3).
```

The fitted $`d_nq_3`$ addition and damping factors remain in this formula. For $`\phi=\varphi_1-\varphi_2`$, the tree kernel satisfies

```math
2F_2=\frac{12}{7}
+\left(\frac{k_1}{k_2}+\frac{k_2}{k_1}\right)\cos\phi
+\frac{2}{7}\cos2\phi.
```

Each cyclic contribution therefore contains five real separable products: a monopole, two dipole terms carrying the two radial ratios, a quadrupole, and the fitted monopole proportional to $`q_3D(k_3)`$. Every product has the form required by the Hankel identity. Filters belong to their observed legs throughout the cyclic sum. Equal radii permit further reuse; unequal radii require the correctly permuted filter factors.

This decomposition is already visible in main's exact BiHalofit primitive list: it represents the real dipole and quadrupole through pairs of positive and negative complex harmonics. Its radial helpers expose the damping and effective powers separately. The proposed aperture calculation can use those physical factors directly and does not need the natural-component coupling matrices or a two-dimensional FFTLog for this term. [TATT three-halo formula](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/halofit.py#L563-L575), [main harmonic primitives](https://github.com/git-sunao/fastnc/blob/e59eab5a07401d2f484bfe00dd48827f2d2162c7/fastnc/bispectrum/models/bihalofit.py#L109-L120), [one-leg radial factors](https://github.com/git-sunao/fastnc/blob/e59eab5a07401d2f484bfe00dd48827f2d2162c7/fastnc/bispectrum/models/bihalofit.py#L424-L486)

The reduction moves the numerical burden to several one-dimensional transforms and a radial product integral. The number of distinct transforms is set by radial function, filter radius, Bessel order, and any power of $`\ell`$, not by the number of covariance or tomographic entries that later use them. FFTLog, checked direct quadrature, or another converged Hankel method can supply these functions. Their radial tails, zero crossings, and cancellation between cyclic terms need validation; reducing dimensions alone does not certify accuracy.

### 3.2 The exact one-halo fit does not have the same separation

BiHalofit's one-halo coefficients depend on the sorted triangle ratios $`r_1,r_2`$. That couples all three lengths even though the code multiplies three apparently one-leg factors. Main's trained representation approximates a one-leg profile as $`\sum_a A_a(r_1,r_2)V_a(k,z)`$. The same shape-dependent amplitudes enter each leg. Expanding the product creates terms containing $`A_aA_bA_c`$, which still depend on the entire triangle; they cannot be pulled into three independent radial transforms. The current semi-analytic implementation retains angular sampling for exactly this reason. [Exact shape dependence](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/halofit.py#L509-L559), [trained radial reconstruction](https://github.com/git-sunao/fastnc/blob/e59eab5a07401d2f484bfe00dd48827f2d2162c7/fastnc/bispectrum/models/bihalofit.py#L368-L422), [remaining angular integration](https://github.com/git-sunao/fastnc/blob/e59eab5a07401d2f484bfe00dd48827f2d2162c7/fastnc/threepcf/semi_analytic.py#L251-L304)

A physical one-halo bispectrum written as a halo-mass integral of three profile factors does admit the radial-product reduction at each mass and redshift. The resulting integrated three-filter halo response is also the object needed by the connected one-halo covariance in Chapter 15. This is an opportunity to share physical profile arrays between the mean and covariance when they use the same halo prescription. It is not permission to replace the published BiHalofit mean by a different halo model. Main's fixed-shape BiHalofit option explicitly changes the fit, and its `NFWOneHaloBispectrum3D` default is described as an NFW-like debug profile rather than a truncated NFW transform. [Fixed-shape qualification](https://github.com/git-sunao/fastnc/blob/e59eab5a07401d2f484bfe00dd48827f2d2162c7/fastnc/bispectrum/models/bihalofit.py#L574-L580), [debug profile qualification](https://github.com/git-sunao/fastnc/blob/e59eab5a07401d2f484bfe00dd48827f2d2162c7/fastnc/bispectrum/models/onehalo.py#L64-L70)

## 4. Gaussian filters allow an independent analytic angular integral

There is another reduction for the tree and active TATT terms. Set $`x=\ell_1`$, $`y=\ell_2`$, $`c=\cos\phi`$, and $`t=\ell_3^2=x^2+y^2+2xyc`$. The product of three Gaussian aperture filters is

```math
\widetilde U_{R_1}(x)\widetilde U_{R_2}(y)\widetilde U_{R_3}(\sqrt t)
=\frac{R_1^2R_2^2R_3^2}{8}\,x^2y^2t\,e^{-A-bc},
```

where $`A=[(R_1^2+R_3^2)x^2+(R_2^2+R_3^2)y^2]/2`$ and $`b=R_3^2xy`$. For one tree insertion, $`P(x/\chi)P(y/\chi)`$ has no opening-angle dependence. Since $`F_2`$ is quadratic in $`c`$ and $`t`$ is linear in $`c`$, the remaining polynomial has degree at most three. Each cyclic insertion can be expressed using its own two input sides, with the filters relabeled at the same time.

Define the angular moments $`\mathcal I_n(b)=(2\pi)^{-1}\int_0^{2\pi}d\phi\,e^{-b\cos\phi}\cos^n\phi`$. In terms of modified Bessel functions $`I_m`$,

```math
\mathcal I_0=I_0(b),\qquad
\mathcal I_1=-I_1(b),\qquad
\mathcal I_2=\frac{I_0(b)+I_2(b)}{2},\qquad
\mathcal I_3=-\frac{3I_1(b)+I_3(b)}{4}.
```

Consequently, the opening-angle quadrature can be replaced by four known functions. Two length integrals and the line-of-sight integral remain. This is a useful independent reference for the Hankel product route because it removes the angle by a different identity. It also preserves the nonlinear power substitution in the active IA source when that substitution is only a radial function on each input leg.

The quadratic E shape from Section 2 is particularly useful. In a frame whose x axis follows the output wavevector, it is $`S=[\cos(2\alpha_1)+\cos(2\alpha_2)]/2`$, where the two angles specify the input directions. Expressing those directions through $`x,y,\phi`$ gives

```math
S=\frac{(x^2+y^2)c^2+2xyc}{t}.
```

The factor $`t`$ in the third aperture filter cancels this denominator. The filtered quadratic kernel therefore needs only $`\mathcal I_0,\mathcal I_1,\mathcal I_2`$. The active velocity-shear shape $`1-c^2`$, multiplied by the filters' $`t`$, needs moments through $`\mathcal I_3`$. These statements can be checked directly against `get_F_20`, `get_F_22`, and the final E projection, without invoking the inactive alternative helper functions. [Active quadratic and velocity kernels](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L189-L216), [second tensor component](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L258-L281), [E combination](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L625-L628)

The same $`S`$ also fits the separable harmonic method: write $`c^2=(1+\cos2\phi)/2`$ and absorb $`1/\ell_3^2`$ into the third radial factor. Its product with the compensated Gaussian filter has a finite zero-wavenumber limit. Form that product analytically before evaluating it; computing an infinity and subsequently multiplying it by a zero filter is unnecessary and can generate a nonfinite value.

Modified Bessel functions grow exponentially for large positive argument. Evaluate $`e^{-A}I_m(b)`$ using a scaled function $`e^{-b}I_m(b)`$ and the remaining factor $`e^{-(A-b)}`$. Here $`A-b=[R_1^2x^2+R_2^2y^2+R_3^2(x-y)^2]/2`$ is nonnegative, so the remaining exponential cannot overflow. Near cancellations, combinations of these moments need their own precision checks. A stable exponential representation does not resolve cancellation between large polynomial contributions.

The full BiHalofit three-halo term has the additional function $`D(\sqrt t)`$ and its fitted $`q_3`$ contribution. They prevent a finite polynomial-in-$`c`$ treatment by these four angular moments. Section 3's radial-product identity retains them exactly. The two reductions serve different parts of the calculation; using the tree angular formula for the full BiHalofit term would omit fitted physics.

## 5. Store aperture responses for nuisance derivatives

Once an aperture response is available at each radial node, repeated nuisance-parameter evaluations can contract small arrays. A schematic residual is

```math
\mu_A(p)=\sum_{r,q}w_{Arq}(p)\,\mathcal R_{Arq},
\qquad
\frac{\partial\mu_A}{\partial p}
=\sum_{r,q}\frac{\partial w_{Arq}}{\partial p}\,\mathcal R_{Arq}.
```

The output index $`A`$ labels an aperture and ordered source-bin choice; $`q`$ labels a physical insertion or basis term. The weights therefore retain $`A`$: different source triplets have different radial weights. When their angular estimator is identical, the response can be stored once per filter and physical insertion, with the source labels carried only by the weights. Bin-dependent triangle cuts require the corresponding additional operator label. The derivative expression holds when $`p`$ changes only the weights. The expensive response $`\mathcal R`$ contains the spectra, geometric factors, and angular integration at the current cosmology. Source-distribution shifts, common IA amplitudes, and IA redshift exponents can use this organization after their coefficient dependence has been removed from $`\mathcal R`$. Cosmological displacements generally change both responses and weights.

For a coefficient $`C(z)=A f(z)[(1+z)/(1+z_p)]^\eta`$, the exponent derivative is $`\partial_\eta C=C\ln[(1+z)/(1+z_p)]`$. The amplitude derivative should be evaluated as $`f(z)[(1+z)/(1+z_p)]^\eta`$, including at $`A=0`$, rather than as $`C/A`$. This preserves the physical null exponent direction at zero amplitude and avoids an artificial division by zero. The finite polynomial amplitude dependence of the tree source permits analytic weight derivatives or a small exact amplitude basis at fixed exponents. Redshift exponents must remain node-dependent; a single constant amplitude multiplier cannot replace them.

Shear calibration is even later in the calculation. For an entry with source bins $`i,j,k`$, the Y6 module multiplies by $`(1+m_i)(1+m_j)(1+m_k)`$. If a bin appears twice, differentiating its calibration gives two contributions. At zero calibration, the derivative is the uncalibrated entry multiplied by that bin's multiplicity. This derivative needs no bispectrum or transform call. [Calibration implementation](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/modules/map3_m_bias.py#L12-L15)

Photo-z derivatives still require the correct shifted source distribution, normalization, lensing integral, and any support changes. Reusing the angular response does not mean holding those kernels fixed. Likewise, replacing a full physical calculation by an emulator does not establish exact amplitude identities unless its architecture enforces them. Compare analytic weight derivatives with finite differences of the accepted direct calculation at several step sizes. For an exact match to historical log-interpolated matter outputs, retain the nonlinear interpolation derivative or separately quantify the refined linear-operator approximation.

Rafael's slice emulator already separates source weights from the expensive slice prediction. The proposed response arrays apply that separation to a deterministic calculation and expose the physical basis explicitly. Their benefit relative to the emulator is controlled direct evaluation and derivative checks, not an assumed shorter evaluation time. Main's structured projected route also prepares radial coefficient arrays before contracting line-of-sight weights; it supplies an existing design to compare against. [Y6 slice contraction](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/modules/map3_emulator.py#L94-L119), [main radial coefficients](https://github.com/git-sunao/fastnc/blob/e59eab5a07401d2f484bfe00dd48827f2d2162c7/fastnc/threepcf/slepian.py#L1801-L1838)

## 6. Shared tables, OpenMP ownership, and SIMD after the reductions

The reduced algorithms change which loops deserve attention. For the radial-product calculation, workers can own a redshift block and a set of filter responses. Within that work item, several filters or independent radial outputs can occupy SIMD lanes. A fixed scalar summation order over radial nodes avoids combining partial sums across lanes. Batch one-dimensional FFTs with the same grid and layout; prepare their plans before workers start. Outer OpenMP and internal FFT threading remain alternative allocations of the same processors, not two independent speed factors.

| Persistent object | What can share it | Values must change when |
|---|---|---|
| One-leg spectra, damping, and effective powers | All filters, cyclic insertions, and source triplets at that cosmology and radial node | Cosmology, spectrum prescription, baryonic model, or physical radial grid changes |
| Filtered Hankel functions indexed by radial family, Bessel order, and filter radius | All separable terms that use the same leg function | A physical leg function, filter, transform domain, order, bias, or boundary rule changes |
| Gaussian angular moments on a fixed $`x,y`$ grid | Tree and quadratic shapes with identical aperture radii | Fourier nodes or filter radii change; cosmology does not change them on fixed angular nodes |
| Aperture response per radial node and physical insertion | Nuisance derivatives and all source triplets with the same angular estimator | Cosmology, power prescription, angular operator, or radial nodes change |
| Source and coefficient weights, including derivative weights | All physical response contractions | Photo-z parameters, source arrays, IA coefficients, cosmology, or normalization convention changes |
| FFT plans and worker scratch | Repeated refills with identical dimensions and layout | Array dimensions, alignment, transform kind, or chosen thread organization changes |

These tables should be bounded. Stream a radial block through the required filter transforms, accumulate or retain its small aperture responses, then reuse the workspace. Keeping all intermediate Hankel functions for every redshift, component, filter, and derivative is unnecessary when the final retained object is a few aperture numbers per radial node. Static allocation concerns storage lifetime; it does not make cosmology-dependent values permanent. Initial setup and changes of numerical grids must reset the appropriate values even when the allocated pointer remains unchanged.

For SIMD, the most direct additional targets are the products $`F_mG_mH_0`$ for several independent filter combinations, the short polynomial combinations of scaled $`I_0,\ldots,I_3`$, and the simultaneous contractions of a response with several tomographic or derivative weights. Special functions can be prepared once where their arguments are geometric. A SIMD loop should not repeatedly evaluate those functions for every source triplet. Whether matrix multiplication or explicit lane arithmetic is preferable depends on the resulting dimensions and must be measured after numerical acceptance.

Mean and covariance work can share read-only filtered functions when their physical inputs and conventions match. [Chapter 19's collapsed-bispectrum response](19_further_covariance_reductions.md) is another such shared object. Its cache must identify the ordered pair of filters and fields, the external Fourier variable, and the actual bispectrum model. A covariance using a halo prescription and a mean using BiHalofit cannot share a numerical value merely because both arrays are described as a bispectrum response.

## 7. Tests that distinguish an identity from an approximation

The first comparison should isolate a single separable harmonic. Compare direct length-and-angle quadrature with the radial-product formula for $`m=0,1,2`$, including unequal filters. This tests the Fourier normalization and $`(-1)^m`$ sign independently of BiHalofit. Then compare every three-halo primitive before their cyclic sum, retaining the fitted $`d_n`$ contribution and damping. Refine radial support as well as node counts; agreement on a truncated interval does not establish a converged full-filter integral.

For TATT, compare the active seven-component assembly with the collected insertion expression before the line-of-sight integral. Use nonzero and zero amplitudes, unequal source weights, independent redshift exponents, and both velocity-shear settings. Compare the quadratic shape's vector definition with its rational expression and its analytically filtered limit near $`\ell_3=0`$. The active-source coefficient is the initial numerical reference; physical normalization discrepancies identified elsewhere need their own resolution.

Compare the Gaussian angular-moment calculation with explicit angular quadrature and the Hankel-product calculation. This gives three organizations of the same full-filter terms. Their agreement should then survive radial projection and nuisance differentiation. Test the final derivatives in the covariance metric used by Fisher inference, rather than accepting small unweighted differences in one fiducial vector.

Finally, compare the full-filter result with the binned, cut estimator as the measurement domain expands and bins narrow. The remaining finite-domain difference is part of the observable definition. It must be modeled through the measured linear operator, shown negligible on retained rows, or retained as an explicit limitation. The exact discrete route in [Chapter 21](21_further_execution_and_cache_design.md) is the production candidate when that difference matters; the reductions here remain independent physics and numerical checks.
