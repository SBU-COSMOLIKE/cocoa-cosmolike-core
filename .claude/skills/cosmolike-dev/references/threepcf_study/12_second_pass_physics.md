# 12. Second physics audit: identities and counterexamples the implementation must respect

> **Archive status.** This source study records design proposals and validation requirements, not completed features. Read the [maintained implementation status](../threepcf.md) first. Both the mean and its covariance belong in `cosmolike/3pcf/`; earlier destination proposals using `fastnc/` or `covariances/` are superseded.

The second reading changes the assessment in three places. Some TATT coefficients share the same triangle dependence at the perturbative order being implemented; the velocity-shear prescription needs a more explicit operator and normalization comparison; and the Y6 emulator configuration varies a neutrino parameter absent from its training coordinates. These findings affect both the proposed numerical decomposition and the interpretation of a Fisher gain. They do not invalidate the case for a C implementation, but agreement with one existing output cannot settle them.

This chapter contains source inspection and analytic reasoning. No numerical predictions, tests, forecasts, or timings were run. Statements about what a later calculation should return are proposed acceptance tests. The source pins are those in Chapters 1, 9, and 10; the publications reopened were [the Y3 methods paper, v2](https://arxiv.org/abs/2503.03964v2) and [the TATT paper, v1](https://arxiv.org/abs/2601.09133v1).

## 1. Two quadratic shape operators have the same transverse tree-level kernel

**The distinction between operators in three dimensions need not survive the projection being used.** Let the line of sight be the third coordinate direction and let the two wavevectors entering a quadratic operator lie in the transverse plane. Write their directions as $`\boldsymbol n_r=(\cos\varphi_r,\sin\varphi_r,0)`$. The tidal tensor per unit density is $`T_{ab}(\boldsymbol n)=n_an_b-\delta_{ab}/3`$. For the transverse components define the complex spin-two projection

```math
\mathcal P[A]=A_{xx}-A_{yy}+2iA_{xy}.
```

This removes the transverse isotropic part and records the two shape components in one number. Direct multiplication gives

```math
\begin{aligned}
\mathcal P\!\left[\frac{T_1+T_2}{2}\right]
 &=\frac{e^{2i\varphi_1}+e^{2i\varphi_2}}{2},\\
\mathcal P\!\left[\frac{T_1T_2+T_2T_1}{2}\right]
 &=\frac{e^{2i\varphi_1}+e^{2i\varphi_2}}{6}.
\end{aligned}
```

The trace subtraction in the quadratic tidal operator does not change the second line because $`\mathcal P[\delta_{ab}]=0`$. The first line is the symmetrized kernel of density weighting, $`\delta s_{ab}`$; the second is the kernel of quadratic tidal torquing. Their projected kernels differ only by a factor of three. Consequently, at fixed redshift and this tree-level transverse limit, the quadratic shape response depends on

```math
Q(z)=C_{1\delta}(z)+\frac{C_2(z)}{3}.
```

This identity also follows by reducing the displayed density–density–shape expression with $`\cos(\varphi_1-\varphi_2)\cos(\varphi_1+\varphi_2)=[\cos(2\varphi_1)+\cos(2\varphi_2)]/2`$. In the active source, combining `get_F_20` and `get_F_22` gives the same sum: `cg2_2+cg2_3` becomes $`C_{1\delta}+C_2/3`$. This is an analytic inference from the implemented kernels, not a measured degeneracy. [Pinned kernel definitions](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L167), [coefficient mapping](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L608).

**The scope matters.** This does not prove that the projected survey vector has an exact degeneracy between the sampled parameters $`A_2`$ and $`b_{\rm TA}`$. Their prescribed coefficients have different growth and redshift dependence; tomography can distinguish those prescribed functions. Two-point loop integrals also involve internal three-dimensional momenta and need not obey this transverse identity. Beyond-Limber contributions and higher perturbative orders require a separate derivation. What is excluded here is independent discrimination of these two fixed-redshift tree-level operators solely through triangle shape.

The first proposed test holds $`C_1`$, $`C_t`$, redshift, and spectra fixed while changing $`C_2\rightarrow C_2+3\epsilon`$ and $`C_{1\delta}\rightarrow C_{1\delta}-\epsilon`$. Every relevant tree-level projected shape kernel should remain unchanged. Follow this with a derivative-rank comparison using the standard redshift laws, more freely varying redshift coefficients, and the joint two-point vector. That separates information supplied by triangle dependence from information supplied by assumptions about redshift evolution. For performance, derive one fixed-redshift $`Q`$ template wherever this identity applies, but retain the distinct parameter-dependent radial weights when projecting it.

## 2. Velocity shear requires an operator test, not only a parameter-transfer test

**First specify the dimensionless velocity variable.** For the convention $`\theta=-\nabla\!\cdot\boldsymbol v/(aHf)`$, linear growing-mode density and velocity divergence agree. If the intended operator is

```math
t_{ab}(\boldsymbol k)=
\left(\hat k_a\hat k_b-\frac{\delta_{ab}}{3}\right)
[\theta(\boldsymbol k)-\delta(\boldsymbol k)],
```

its transverse spin-two part is proportional to $`e^{2i\varphi_k}`$ times a scalar. Rotating into the E/B frame therefore leaves a pure E contribution. This is a geometric statement about that operator, stronger than a general parity argument. A parity-symmetric theory can have an oriented single-B bispectrum that reverses sign under reflection; parity alone does not force that bispectrum to vanish point by point.

The active `get_F_21` contains no dependence on `cg2_1`, the coefficient through which `Ct` enters. The returned single-B bispectra consequently have no `Ct` response. The paper's displayed VS single-B expressions, Eqs. 67–68, instead contain that response. The source behavior is compatible with the longitudinal operator above, while those displayed expressions require an explanation of the operator mapping. The E-only reproduction should not hide the question: the same mapping determines its E amplitudes. [Active B kernel](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L228), [paper operator and VS expressions](https://arxiv.org/html/2601.09133v1#S2.SS4).

There is also a normalization diagnostic. In the Einstein–de Sitter second-order kernels, $`G_2-F_2=-2(1-\mu^2)/7`$, with $`\mu=\hat{\boldsymbol k}_1\cdot\hat{\boldsymbol k}_2`$. Wick contracting two linear densities with the quadratic velocity-difference field gives a factor of two. With the same observable normalization as the linear tidal coefficient, that definition predicts

```math
B_{\delta\delta E}^{t}
=-\frac{4}{7}C_t(1-\mu^2)P_1P_2.
```

In the active source, its `cg2_1=-2*Ct/5`, the `get_F_20` VS term, and the final E projection instead combine to $`-2C_t(1-\mu^2)P_1P_2/7`$. These expressions differ under the stated common normalization. This is not yet a justified code correction: first establish whether the source coefficient denotes half or all of the physical operator coefficient, whether its tensor convention introduces another factor, and which normalization the plotted $`C_t/C_1`$ values use. An independent Cartesian contraction with one non-collinear triangle can settle the mapping before any FFT. The reference $`F_2,G_2`$ expressions are given in [Euclid's perturbation-theory appendix, B.9–B.10](https://acris.aalto.fi/ws/portalfiles/portal/170662540/Euclid_preparation_XLI._Galaxy_power_spectrum_modelling_in_real_space.pdf).

## 3. Nonlinear power spectra do not restore missing perturbative orders

The active calculation contains one quadratic insertion and two linear insertions in a bispectrum. Replacing the power spectra inside that expression by nonlinear spectra changes its scale dependence, but does not generate all contractions with two or three quadratic shape operators. This distinction can be exposed without running the code. Set $`C_1=0`$, keep $`C_2\ne0`$, and use the source's linked density-weighting and VS coefficients. Its tree-level density–density–shape term survives, while its density–shape–shape and three-shape terms vanish. Yet three quadratic shapes contain six linear density factors; their Gaussian contractions need not vanish and contribute at order $`P_L^3`$, beyond the retained tree bispectrum. [Active tensor assembly](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L299).

Thus a vanishing tree-level three-shape prediction near small $`C_1`$ is not evidence of a physically vanishing three-shape signal. This matters when broad IA priors admit small alignment and appreciable torquing. It also matters for emulation: the implemented prediction is at most linear in $`A_2`$ at fixed other inputs, whereas omitted contractions can contain higher powers. An emulator may reproduce that linearity accurately while the physical truncation remains uncertain.

Proposed tests should therefore include amplitude limits as well as grid convergence. Verify the implemented polynomial dependence; compare representative low-$`C_1`$, high-$`C_2`$ cases with an independent higher-order or simulation-based estimate on the retained scales; and test the stability of the joint gain under excluding scales or parameter regions where the omitted terms are appreciable. Completing a second-order operator list and computing a controlled bispectrum to a specified perturbative order are different claims.

The single-row nonlinear-spectrum evolution in Chapter 1 has a useful further consequence. With zero redshift exponents and the source's $`P(k,z)=D^2(z)P(k,0)`$, a $`C_2P^2`$ contribution scales as $`D^2`$, while a $`C_{1\delta}P^2`$ contribution scales as $`D^3`$. Those distinct assumed redshift laws can help separate the coefficients in Section 1. Changing to fully evolved nonlinear spectra can consequently change inferred identifiability, not only the total amplitude. Compare both prescriptions component by component before treating that separation as measured physical information.

## 4. The Y6 configurations contain an explicit neutrino-response mismatch

The inference configuration varies the summed neutrino mass from 0.06 to 0.6 eV, with fiducial value 0.0773 eV. The training configuration fixes it to 0.06 eV, and the emulator design has no neutrino-mass coordinate. These are source facts, rather than a hypothetical future extension. [Inference values](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/config/values/lcdm-tatt.ini#L7-L18), [training values](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/config/emulator/values-training.ini#L7-L17), [training coordinates](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/config/emulator/designs.yaml).

At fixed emulator coordinates, its matter-slice predictions cannot acquire an independent neutrino response. The final projected prediction can still change through quantities supplied outside the network, such as the growth entering NLA weights; this does not restore the missing response of the slices. A check that parameters lie inside the training box cannot detect a missing coordinate. The direct calculation can respond to a changed input power-spectrum shape, so agreement with the emulator at the training mass would not establish agreement along the sampled mass direction.

The physical reference also has limited scope: BiHalofit used a fixed 0.06 eV mass and an approximation to neutrino growth, rather than a calibration over the sampled mass interval. For the first reproduction, preserve the paper's treatment and label it. For a Y3/Y6 Fisher forecast, either fix the mass consistently or supply and validate the missing response. The discriminating comparison varies mass while fixing $`\Omega_m,S_8,h,\Omega_b,n_s,w`$, then records changes in the input spectra, direct slices, emulated slices, projection weights, and final derivatives separately. [BiHalofit calibration, Section III.1](https://arxiv.org/html/1911.07886#S3.SS1).

The training box also spans $`-2.1\le w\le-0.45`$, whereas the original BiHalofit simulations sampled constant $`w`$ within approximately 20% of $`-1`$. A network trained on extrapolated fitting-formula outputs can interpolate its training labels accurately without extending their physical calibration. Keep an explicit intersection of survey priors, emulator coordinates, fitting-formula calibration, and validated derivative domain.

## 5. The equation sheet needs dimensional and algebraic corrections before reuse

Additional printed expressions cannot be used literally as the C specification. The following checks refer to the [TATT PDF, v1](https://arxiv.org/pdf/2601.09133v1); its extracted PDF expressions agree with the corresponding HTML on the discrepancies below. They establish problems in the written account, not that published numerical arrays used those expressions.

| Printed location | Independent check and required specification |
|---|---|
| Eq. 2 | With dimensionless lensing efficiency and $`H_0^2/c^2`$ carrying inverse-length squared, the convergence integral needs the missing factor of $`\chi`$. |
| Eq. 23 | The stated $`p(\chi)`$ is already a density per distance. The NLA replacement must reconcile both that definition and the prefactor removed from the lensing weight; an additional Jacobian alone does not do so. |
| Eq. 36 | The final angular term needs $`\mu^2`$, as in the executable `F2_tree`. The zero-output-wavevector limit $`\boldsymbol k_2=-\boldsymbol k_1`$ gives $`F_2=0`$ with the squared term and $`-4/7`$ with the printed term. |
| Eq. 69 | The lensing weight needs $`c^2`$ in the denominator to have inverse-length units. |
| Eq. 82 | The displayed sum omits the coupling function described immediately before it. Specify its normalization and angular indices explicitly. |

For clarity, a consistent convention is

```math
W_G=\frac{3\Omega_mH_0^2}{2c^2}\frac{\chi q}{a},
\qquad W_I=C_1p_\chi,
\qquad
B_{\rm angular}=\int d\chi\,\chi^{-4}W_1W_2W_3B_{3D}.
```

Each weight has inverse-length units. The pinned code instead factors out $`(1+z)^3/\chi`$ and uses an IA weight proportional to $`p_\chi a/\chi`$, along with its lensing prefactor. These conventions can agree; the correspondence must be demonstrated, including distance units and each power of $`h`$. Test pure lensing, one intrinsic leg, two intrinsic legs, and three intrinsic legs independently. A common error that cancels in one ratio need not cancel between those terms. [Implemented radial kernels](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/bispectrum.py#L419).

## 6. Baseline subtraction and scale cuts need tests that expose cancellations

The description following Eq. 85 says its lensing weights include NLA. That cannot be applied indiscriminately to every residual term after the linear-alignment contribution has been subtracted from each ordered bispectrum. A density–shape–shape residual already contains a linear intrinsic field on one shape leg; placing another NLA replacement on density legs of the density–density–shape residual generates contributions of that type again. The Y6 execution instead uses NLA in the reference term when required and pure lensing weights at `d` positions in the residuals. Preserve that distinction in the written model. [Residual projection assembly](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/modules/map3_emulator.py#L100-L121).

A discriminating test expands the prediction in powers of $`C_1`$ while holding one quadratic coefficient nonzero, then compares each coefficient with a direct expansion of three observed fields. The usual test that all residuals vanish when quadratic coefficients vanish does not catch this error: the potentially double-counted terms vanish in that limit too.

Scale selection also needs physical support information. An aperture radius is not a sharp maximum wavenumber: the prediction integrates over several Fourier magnitudes and distances, with $`k=\ell/\chi`$. Determine how much of each retained component and derivative comes from regions beyond the chosen spectral or fitting-formula domain. The Y6 backend interpolates nonlinear power onto the linear grid using `np.interp`, whose endpoint values are used outside the supplied range. If the grids have unequal coverage, the result can acquire flat spectral tails before fastnc's own extrapolation. That conditional risk should be tested with explicit grid-coverage records. [Spectrum transfer](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/desy6_3pcf/fastnc_backend.py#L215-L219).

Finally, the first paper's baryon validation uses an OWLS-like rescaling of a TNG-derived bispectrum correction. It is a specified stress case, not coverage of all feedback shapes or of quadratic IA responses. The executable correction rescales $`R_b-1`$, preserving the large-scale limit $`R_b\rightarrow1`$; multiplying the entire ratio by that amplitude would not. Compare enhancements and suppressions separately before relying on cancellation in a joint parameter shift, and repeat the accepted cuts for each survey definition. [Paper validation prescription](https://arxiv.org/html/2503.03964v2#S4.SS3), [implemented baryon factor](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/baryon.py#L32).

The immediate scientific requirements are therefore a checked operator-to-coefficient map, the fixed-redshift quadratic identity, a separate test of the VS normalization, and a forecast specification that states which physical responses are actually varied. Numerical acceleration should preserve these identities and reveal approximation errors rather than make their assumptions harder to inspect.
