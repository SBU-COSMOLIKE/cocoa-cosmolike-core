# Physics and reproduction of the two papers

> **Archive status.** This source study records design proposals and validation requirements, not completed features. Read the [maintained implementation status](../threepcf.md) first. Both the mean and its covariance belong in `cosmolike/3pcf/`; earlier destination proposals using `fastnc/` or `covariances/` are superseded.

## What the C implementation must reproduce

The two requested papers set different targets. [Gomes et al., arXiv:2503.03964v2](https://arxiv.org/abs/2503.03964v2) tests a joint inference pipeline using NLA intrinsic alignments, simulated DES Y3 measurements, and a covariance estimated from 796 realizations. Its fiducial combination has 96 entries after compressing the two-point contribution, compared with 307 before that compression. The reported 83% improvement concerns the marginalized matter-density–$`S_8`$ figure of merit. Matching that number requires matching the inference assumptions, not merely the three-point prediction.

[Gomes et al., arXiv:2601.09133v1](https://arxiv.org/abs/2601.09133v1) supplies the second target: TATT and its velocity-shear extension. Its numerical natural-component predictions retain the E-mode contribution, and its aperture predictions include the seven ordered intrinsic-alignment contributions. Reproducing those calculations does not require claiming a full E/B three-point implementation. The equation and figure comparisons below are proposed acceptance tests; none has been run for this study.

Keep four named scientific configurations throughout the project: the first paper's NLA reproduction, the second paper's TATT reproduction, the DES Y3 Fisher forecast, and the DES Y6 Fisher forecast. Each configuration must have its own parameter values, physical approximations, survey inputs, scale cuts, and numerical settings. The Y3 Fisher forecast can share inputs with the first reproduction, but a local curvature calculation does not reproduce a marginalized posterior by definition. A common numerical library can evaluate all four configurations; changing a survey or model still requires a separately identified comparison.

## Why intrinsic alignments change the calculation

A galaxy's measured shape contains a response to gravitational lensing and a correlated intrinsic shape. Intrinsic alignment means that galaxies in related tidal environments tend to have related shapes. A tidal field describes the direction-dependent part of gravitational stretching: a spherical overdensity contributes a density contrast, but its surroundings can stretch a galaxy differently along different axes. The intrinsic shape therefore has tensor structure even when the density field is a scalar.

Write the deterministic shape model schematically as

```math
I_{ab}=C_1 s_{ab}+C_{1\delta}\delta s_{ab}
 +C_2\left(s_{ac}s_{cb}-\frac{\delta_{ab}}{3}s_{cd}s_{cd}\right)
 +C_t t_{ab}.
```

Here $`a,b,c,d`$ label spatial directions, repeated direction indices are summed, $`\delta`$ is the fractional matter overdensity, and $`\delta_{ab}`$ is the identity tensor. The tensor $`s_{ab}`$ is the trace-free tidal field. Removing its trace removes the isotropic part, because a scalar change in size is not a preferred shape direction. The coefficients describe how the galaxy population responds to these fields. The last term is retained only for the velocity-shear model.

The first term is linear tidal alignment. The density-weighting term allows the sampled shapes to respond to the fact that galaxies occupy a nonuniform set of positions. The quadratic tidal term allows the shape response to depend on products of tidal-field components. The linear and quadratic contributions have different triangle dependence, so the full effect cannot generally be represented by changing one overall amplitude of the lensing bispectrum. The two quadratic operators themselves share a transverse tree-level shape kernel: at fixed redshift they enter through $`C_{1\delta}+C_2/3`$. Their different assumed redshift laws and the three-dimensional loops in the two-point calculation can distinguish them. [Chapter 12](12_second_pass_physics.md) derives this restriction; [the pinned IA implementation](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L167) supplies the tensor kernels and ordered components.

The source maps the phenomenological parameters to redshift-dependent coefficients as

```math
C_1(z)=-A_1\,c_*\,\Omega_m\,D(z)^{-1}
       \left(\frac{1+z}{1+z_0}\right)^{\alpha_1},
\qquad C_{1\delta}(z)=b_{\rm TA}C_1(z),
```

```math
C_2(z)=5A_2\,c_*\,\Omega_m\,D(z)^{-2}
       \left(\frac{1+z}{1+z_0}\right)^{\alpha_2}.
```

In this implementation $`c_*=0.0134`$ collects the normalization convention. The growth factor $`D(z)`$ measures the linear density amplitude relative to its normalization epoch; it is not the logarithmic growth rate $`f=d\ln D/d\ln a`$. The pivot redshift $`z_0`$ makes the redshift power law equal to one at a defined epoch. The signs, factor of five, growth normalization, and pivot must agree between the two-point and three-point predictions. Merely giving parameters the same names does not establish that agreement. These expressions are read directly from [the coefficient construction](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L597).

A useful acceptance test starts with a small table of coefficients evaluated independently at several redshifts. Vary $`A_2`$ while holding $`\alpha_2`$ fixed, then vary $`\alpha_2`$ while holding $`A_2`$ fixed. The first variation rescales the quadratic term at every redshift. The second changes its relative redshift dependence while leaving its value at the pivot unchanged. This test will expose a parameter-transfer mistake before it propagates into a projected data vector. The source audit identifies an actual transfer requiring this check.

## What the velocity-shear parameter means

The velocity-shear operator measures the trace-free part of the difference between an appropriately normalized velocity divergence and the density. Its intended perturbative role is easiest to understand by requiring that the two scalar fields agree at linear order. Their difference then begins beyond linear order and supplies a distinct shape response. A bare velocity divergence has units and sign conventions that must be resolved before it can be subtracted from a dimensionless density contrast. The implementation note should state the normalization explicitly, including any factors of scale factor, Hubble rate, and linear growth rate.

The supplied source takes an input named `Ct` and multiplies it by $`C_1(z)`$ before evaluating the bispectra. Its external meaning is therefore a ratio, which this study denotes $`r_t`$:

```math
C_t(z)=r_t C_1(z).
```

This is an inference from [the assignment at the beginning of the IA calculation](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L606), not a proposal to rename the upstream parameter. The C interface should document that meaning. A parameter called $`C_t`$ that accepts a physical coefficient directly would define a different model.

This distinction matters for a Fisher calculation. With the ratio convention, setting $`A_1=0`$ also removes the velocity-shear term at finite $`r_t`$. A free physical velocity-shear amplitude could remain nonzero there. Those models have different derivatives and different parameter degeneracies. The initial reproduction should use the source convention; a later independent-amplitude study must specify its redshift law and priors separately.

CosmoLike's nonlinear velocity terms used in galaxy redshift-space clustering do not supply this intrinsic-shape operator. Both involve velocity fields, but the observables, tensor contractions, perturbative combinations, and nuisance coefficients differ. Likewise, adding $`r_t`$ only to the three-point prediction while retaining an ordinary TATT two-point prediction defines a restricted joint model. It cannot be described as a fully shared TATT-plus-velocity-shear model without checking the corresponding two-point contributions. This is a physical specification needed before discussing the cost of an additional coefficient.

## Why CFASTPT alone does not provide the missing three-point physics

An ordinary power spectrum describes correlations between two Fourier modes. Translational invariance forces those modes to be opposite, leaving one independent wavenumber magnitude for an isotropic two-point calculation. A bispectrum instead describes three modes whose vector sum vanishes. It depends on a triangle, so two magnitudes and an opening angle remain independent. This extra geometry survives even when one reuses the same matter power spectrum and the same IA parameters.

The CFASTPT IA tables in CosmoLike evaluate the convolution structures needed by its two-point TATT model. Their FFT organization, numerical padding, and reusable spectral inputs are relevant examples. Their numerical arrays cannot simply be inserted as the IA bispectra of the new module. The new calculation needs the density–density–shape, density–shape–shape, and shape–shape–shape contractions for each relevant triangle. The source computes these through tensor components in [the IA kernel and bispectrum methods](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L299).

This gives a clean division of responsibility. Reuse cosmological inputs and their conventions. Preserve independently testable implementations of the three-point tensor algebra. Share only numerical helpers whose mathematical inputs and outputs agree. Replacing a Python loop by a C loop does not change this requirement, and using an existing FFT library does not establish the validity of a new physical kernel.

## Ordered legs, rotations, and the projection integral

For three observed shears, expanding $`(G+I)_1(G+I)_2(G+I)_3`$ gives one purely gravitational term and seven terms containing intrinsic shapes. Each index identifies a triangle vertex and its source population. For example, two lensing fields followed by an intrinsic shape use two lensing kernels and the local source distribution of the third population. Moving the intrinsic shape to another vertex changes both the bispectrum arguments and the radial weights.

Use a dimensionally explicit convention in the C documentation. Let $`p_i(\chi)`$ be the source probability density per comoving distance, normalized by $`\int d\chi\,p_i(\chi)=1`$. Let $`W_i^G(\chi)`$ be the lensing weight with units of inverse length, and let $`W_i^I(\chi)=p_i(\chi)`$ when the IA coefficients are kept inside the intrinsic bispectrum. For a flat geometry, each ordered contribution has the form

```math
B^{XYZ}_{ijk}(\ell_1,\ell_2,\ell_3)
=\int d\chi\,
 \frac{W_i^X(\chi)W_j^Y(\chi)W_k^Z(\chi)}{\chi^4}
 B_{XYZ}\left(\frac{\ell_1}{\chi},\frac{\ell_2}{\chi},
              \frac{\ell_3}{\chi};z(\chi)\right).
```

Here $`X,Y,Z`$ specify gravitational density or intrinsic-shape fields. The 3D bispectrum has units of length to the sixth power when its fields are dimensionless. The integral is dimensionless: the powers of length from the measure, three weights, projection denominator, and bispectrum cancel. An implementation using $`n(z)`$, distances in Mpc/$`h`$, or a lensing-efficiency function with different prefactors must demonstrate the equivalent cancellation. The upstream code factors these weights differently; the proposed convention is a documentation and test convention, not a claim that its variable named `W` is literally this $`W_i^I`$. See [the actual ordered integrands](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/bispectrum.py#L758).

The triangle axes need similar care. The IA source constructs a frame with the third wavevector on a coordinate axis, then reconstructs that frame for permuted arguments. Sorting three side lengths without also transporting the field identities loses physical information. A test with three equal source bins can conceal this mistake. Require unequal-bin cases, a scalene triangle, every cyclic permutation, and a reflected triangle before enabling symmetry-based reductions. The reflection test must distinguish an E-mode scalar contribution from a B-mode pseudoscalar contribution; one must not erase a sign change by replacing all triangle orientations with positive angles.

## The nonlinear prescription and the danger of counting alignment twice

The source provides two ingredients with different roles: a nonlinear matter-bispectrum calculation that can include NLA through its radial kernel, and perturbative IA bispectra containing linear-alignment pieces as well as higher-order shape terms. Adding their full outputs would repeat the linear-alignment contribution.

Define $`B_{\rm base}[P]`$ to be the cyclic second-order matter-bispectrum expression formed with the chosen input power spectrum. An IA component containing $`r`$ intrinsic-shape legs includes a term $`C_1^r B_{\rm base}[P]`$. The `remove_alignment` option removes this term for the seven ordered components. A physically explicit assembly is then

```math
d_{\rm total}
=d_{\rm BiHalofit+NLA}
 +\sum_{XYZ\ne GGG}d_{XYZ,\,\rm residual}.
```

The residual contains the density-weighting, tidal-torquing, and selected velocity-shear terms. Its definition must identify the power spectrum used in the subtraction. The relevant behavior is visible in [the residual return expressions](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L742). Test this assembly algebra before running the FFTs. Setting $`C_2=C_{1\delta}=C_t=0`$ should remove every residual and recover the separately evaluated NLA prediction. Setting all IA amplitudes to zero should recover the purely gravitational prediction.

There is also a redshift convention to resolve. The direct interface supplies a single nonlinear power-spectrum row. The IA reader interpolates that row in wavenumber and multiplies it by $`D(z)^2`$ when asked for a redshift. This makes the assumed nonlinear shape evolve with linear growth, rather than reading an independently evolved nonlinear spectrum at each redshift. That statement follows from [the interface](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/fastnc_interface.py#L168) and [the IA nonlinear reader](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L106). It has not been quantified here.

The reproduction branch must first establish whether this is the prescription used for the saved reference figures. A comparison against fully evolving $`P_{\rm NL}(k,z)`$ then measures a model change, not a floating-point discrepancy. Report the effect separately on the gravitational term, each IA residual, and the projected observables. A total-vector comparison can hide a cancellation between components.

## E modes, B modes, and what the aperture statistic retains

An E-mode shear field has the orientation pattern generated by a scalar lensing potential at leading order. Rotating the pattern by 45 degrees gives the corresponding B-mode construction. Intrinsic shapes can generate both kinds of patterns. The imaginary part of one natural component is not, by itself, a pure B-mode observable: the spin rotations and triangle geometry already make the natural components complex.

The source's low-level IA calculation returns E-mode and single-B-mode components, but the `BispectrumTATT` projection wrapper keeps only seven E-mode combinations. This is visible in [the wrapper's return statement](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/bispectrum.py#L1313). Preserve that scope for the requested reproduction. A future full E/B natural-component prediction requires the appropriate parity-dependent angular basis and tests of its phases, rather than simply passing the unused arrays to the existing Legendre transform.

For an ideal continuous aperture, the compensated filter separates E and B contributions. A finite set of measured triangle bins with excluded separations is a different numerical operator. Validate that operator by projecting a known E-only input through the same binning and cuts as the data, then measuring any leakage into nominal B combinations. Apply the corresponding check to a controlled B input. This tests the statistic actually delivered to the likelihood and avoids assuming that a continuous-filter identity holds after truncation.

## Reproduction comparisons to preserve before optimization

The following comparisons are proposed records for the implementation effort. Keep the underlying arrays, input parameters, and component ordering, rather than storing only plot images. Figure references identify comparison targets in [the second paper](https://arxiv.org/pdf/2601.09133v1).

| Target | Proposed comparison and what it diagnoses |
|---|---|
| Equations 49–59 | Compare tensor and separately derived scalar contractions before projection; isolate each coefficient. |
| Equations 60–68 | Isolate velocity shear and verify the input-ratio convention and signs. |
| Equations 70–72 | Compare each ordered radial projection with direct quadrature using identical units. |
| Equations 78–83 | Compare direct angular integration with multipoles and their reconstruction. |
| Equations 84–85 | Compare aperture integration before and after radial projection at matched discretization. |
| Figures 1–2 | Compare natural-component arrays across source triples and triangle shapes; inspect zero crossings separately from ratios. |
| Figure 3 | Recover coefficient-dependent aperture residuals and their NLA limit. |
| Figure 4 | Recover the response to the velocity-shear ratio. |
| Figure 5 | Compare separate operator contributions and their signed sum. |
| Figure 6 | Compare separate ordered field contributions and their signed sum. |
| Figure 7 | Compare the total NLA, TATT, and velocity-shear aperture predictions. |
| Figures 8–9 | Compare the two-point and aperture responses using one parameter convention. |

For a tensor-versus-scalar comparison, the reference must be independently executable. The disabled `direct_methods` branch is not yet such a reference: it calls helpers whose definitions were not found in the pinned module and does not forward `Ct` to those calls. Its density–shape–shape expression also needs reconciliation with the displayed signs in Eq. 52. The source audit gives the file-level evidence. Agreement between two copies of the same unresolved formula would not settle the physical question.

The velocity-shear equations require a similar written reconciliation. The displayed Eq. 62 and the direct source differ in the placement of a square involving the angle cosine. The displayed Eq. 64 contains a sum with unlike powers of wavenumber, whereas the direct source uses matching fourth powers. These are reasons to obtain an agreed equation sheet from the authors and check dimensions, permutation symmetry, and an independent tensor derivation. They are not evidence that a published numerical result used the literal displayed expression. The distinction matters because the scientific target is the intended physical calculation, with any transcription corrections recorded openly.

## Limits to carry into DES Y6

Numerical agreement and physical accuracy need separate records. Grid refinement measures the error from quadrature, interpolation, angular truncation, and transforms. Agreement with the original Python prediction measures implementation consistency. Neither establishes that a phenomenological nonlinear IA continuation describes the survey at every retained scale. Forecasts should vary the physical prescription and angular cuts after the reference implementation agrees, keeping these variations separate from numerical-error tests.

The highest-value model comparisons are specific: NLA versus TATT with the same two-point assumptions; ordinary TATT versus the documented velocity-shear prescription; alternative nonlinear redshift evolution of the IA residuals; and changes to baryonic effects in the matter bispectrum. Small-scale shear corrections, including reduced shear and source selection effects, should be assessed at the accuracy demanded by the retained data, with any neglected terms stated. A power-spectrum baryon correction does not uniquely determine a bispectrum correction because triangle configurations carry additional information.

Finally, model the cosmological parameter actually varied. A fitting formula or emulator validated for a fixed neutrino mass does not acquire a tested neutrino-mass response merely because CosmoLike can vary that mass. The same applies to a time-dependent dark-energy equation of state beyond the source's model family. Initially restrict the forecast to the domain with a defensible bispectrum prescription, or report the additional response model as an assumption and test its effect. The study's covariance and Fisher chapter specifies how these physical choices must enter the final parameter constraints.
