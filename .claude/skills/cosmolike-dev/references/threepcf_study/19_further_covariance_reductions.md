# 19. Further reductions of the covariance integrals

> **Archive status.** This source study records design proposals and validation requirements, not completed features. Read the [maintained implementation status](../threepcf.md) first. Both the mean and its covariance belong in `cosmolike/3pcf/`; earlier destination proposals using `fastnc/` or `covariances/` are superseded.

## 1. Which additional work can be removed

The covariance can share more than spectra and halo profiles. Several integrals in the inspected source integrate two sets of variables that communicate through only one common wavenumber. Compute each set once at that common wavenumber, store the result, and assemble the covariance by a remaining weighted sum. This is a change in the organization of the same integral. It can remove repeated high-dimensional integrations before OpenMP distributes the remaining work.

The strongest additional candidates are listed below. These are derived proposals supported by source inspection, not implemented methods or measured speedups. The source revision remains `08de9645a7294092995e5aeaf7b36e48e8fc10a3`. [Chapter 15](15_analytic_covariance_cpu_port.md) defines the covariance terms and the initial Gaussian and connected one-halo reductions; this chapter extends that analysis.

| Contribution | Additional reduction | Condition |
|---|---|---|
| Bispectrum product, `T4` | Two reusable two-dimensional triangle integrals at each common side, followed by a one-dimensional contraction | Exact rearrangement of the inspected large-area integrand with identical bounds |
| Power times one-halo trispectrum, `T5` | Hankel transforms of products of two filtered halo profiles | Same one-halo trispectrum and full circular filters |
| Finite-field power–trispectrum, `T6` | A single-profile/triple-profile correlation, multiplied by a cached zero-separation pair correlation | Center-window estimator; preserve the finite separation window |
| Connected one-halo five-/six-field terms | Fourier transforms of pair/triple halo products, including a finite center window | Extends the large-area factorization, not an arbitrary masked catalogue estimator |
| Selected long-mode term, `T7_2h` | One bias-weighted aperture response per row and redshift, then a response product | Retains the specific source response model |
| Aperture second-/third-order cross-block | Reuse the same filtered bispectrum functions used by `T4` and the mean | Reference observable is aperture variance; the full shear-correlation vector needs its own kernel |
| Gaussian noise | Closed expression for its filtered pair correlation | White, independent Gaussian noise and full Gaussian aperture filters |
| Tomographic halo terms | Attach source-window products after source-independent halo transforms | Deterministic common matter profiles; nonlinear IA needs additional fields |

The real-space aperture-correlation interpretation is already part of the literature: [Linke et al., Section 3.3](https://arxiv.org/html/2212.04485v2#S3.SS3) and [Wielders et al., Section 3.2](https://arxiv.org/html/2509.20443v1#S3.SS2). The proposed implementation contribution is to calculate and cache those connected functions analytically, exploit the one-halo products in the actual source, and reuse the resulting arrays across covariance entries and mean calculations.

## 2. Organize the covariance by connected correlations

Let $`X_a(\boldsymbol x)`$ be a zero-mean scalar E-mode aperture field centered at $`\boldsymbol x`$. The label $`a=(i,R)`$ specifies a source population and filter radius. Assume statistical homogeneity, isotropy, circular full filters, and a common weight $`w`$ for aperture centers. The measured second and third moments are spatial averages of products of two or three such fields. Write

```math
K_{a|b}(r)=\langle X_a(\boldsymbol x)X_b(\boldsymbol x+\boldsymbol r)\rangle,
\qquad
\mathcal B_{ab|c}(r)=
\langle X_a(\boldsymbol x)X_b(\boldsymbol x)X_c(\boldsymbol x+\boldsymbol r)\rangle_c,
```

```math
\mathcal T_{ab|cd}(r)=
\langle X_a(\boldsymbol x)X_b(\boldsymbol x)
X_c(\boldsymbol x+\boldsymbol r)X_d(\boldsymbol x+\boldsymbol r)\rangle_c.
```

The vertical bar separates fields at the two positions. The subscript $`c`$ denotes a connected correlation: lower-order partitions have been subtracted. For example, the covariance of the composite fields $`X_aX_b`$ and $`X_cX_d`$ is not just $`\mathcal T_{ab|cd}`$. It also contains $`K_{a|c}K_{b|d}+K_{a|d}K_{b|c}`$. Using that full composite covariance as a connected trispectrum input would count Gaussian contributions twice.

Define the normalized separation weight

```math
\mathcal W(\boldsymbol r)=
\frac{1}{A_w^2}\int d^2x\,w(\boldsymbol x)w(\boldsymbol x+\boldsymbol r),
\qquad A_w=\int d^2x\,w(\boldsymbol x).
```

The covariance is the integral of this weight times the connected correlation between the two local estimator products. An isotropic correlation depends on $`r=|\boldsymbol r|`$ alone, so only the angular average of $`\mathcal W`$ enters. Its radial measure remains $`2\pi r\,dr`$. A square or irregular center window does not require keeping the common absolute orientation inside every physical integral. This statement concerns the window on centers; masking the individual galaxies inside each aperture changes the estimator itself.

There are exact combinatorial counts behind the source's permutation sums:

| Covariance | Partition type after subtracting the product of means | Number of positional partitions |
|---|---|---:|
| Third moment with third moment | Three pair correlations | 15: six with three cross-pairs, nine with one cross-pair |
| Third moment with third moment | Pair correlation times connected four-field correlation | 15: nine cross-pairs, six internal pairs |
| Third moment with third moment | Product of connected three-field correlations | 9 |
| Third moment with third moment | Connected six-field correlation | 1 |
| Second moment with third moment | Pair correlation times connected three-field correlation | 9: six cross-pairs, three pairs internal to the triple |
| Second moment with third moment | Connected five-field correlation | 1 |

For six fields, there are ten ways to split them into two unordered triples. Mean subtraction removes the split containing all three fields at the first center and all three at the second, leaving nine. For five fields there are ten choices of the pair; subtracting the mean product removes the pair containing the entire second-order estimator. These are counts of positions, even if several source labels or filter radii coincide. Equal radii justify multiplying identical terms only after checking the source labels and noise contractions too.

This organization identifies a reusable library of radial functions: $`K`$, $`\mathcal B`$, $`\mathcal T_{2|2}`$, $`\mathcal T_{1|3}`$, and the selected connected five-/six-field functions. The covariance assembly becomes products of those functions followed by window-weighted sums. A generic connected four-field function can still be expensive to calculate; the one-halo model below supplies an additional reduction of that input.

## 3. Factor the bispectrum-product integral at its common side

The inspected `T4` integrand has independent triangle variables $`(p,\phi)`$ and $`(k,\psi)`$ at fixed common side $`q`$. Its other sides are $`s=\sqrt{q^2+p^2+2qp\cos\phi}`$ and $`t=\sqrt{q^2+k^2+2qk\cos\psi}`$. For one source population, define

```math
J_{bc}(q)=\int p\,dp\int_0^{2\pi}d\phi\,
\widetilde U_b(p)\widetilde U_c(s)B_\kappa(q,p,s).
```

The filters $`\widetilde U_b`$ and $`\widetilde U_c`$ carry their radii. The source's lower and upper wavenumber limits and closure-side exclusions must appear inside this definition. One ordered contribution is then

```math
C^{BB}_{abc,def}\big|_{\rm ordered}
=\frac{1}{A(2\pi)^5}\int q\,dq\,
\widetilde U_a(q)\widetilde U_d(q)J_{bc}(q)J_{ef}(q).
```

This follows directly by exchanging the order of integration in [the five-dimensional source integrand](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/apertureStatisticsCovariance.cu#L2118). The factor $`(2\pi)^{-5}/A`$ agrees with [its driver](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/apertureStatisticsCovariance.cu#L1166). Tomography adds the three source labels of each bispectrum to the key; it does not destroy separation of the two triangles at fixed $`q`$.

For tomography, identify the residual partition explicitly as $`(bc|d)(ef|a)`$. The first triangle integral uses $`B_{dbc}`$, with source $`d`$ on the shared side, and the second uses $`B_{aef}`$, with source $`a`$ there. In labeled notation the product is $`J_{bc|d}J_{ef|a}`$. The filters on the shared side commute, so the one-population expression hides this exchange. Keeping each original triple together would reconstruct the mean-product partition that covariance subtraction removes.

The original integral evaluates the second triangle repeatedly for many values of the first. A table of $`J`$ removes that repeated dependence. Use common $`q`$ nodes for compatible outputs, compute all required filter pairs at those nodes, then contract the tables for each allowed partition. This also changes the parallel work: a task builds a row of $`J`$ with one owned quadrature sum; SIMD can process independent $`q`$ rows or filter outputs sharing the same bispectrum values.

For a fixed-rule illustration, let $`N_q`$ count common-side nodes and $`N_t=N_pN_\phi`$ count nodes of each remaining triangle. Directly summing every triangle pair uses $`N_qN_t^2`$ point combinations. Preparing the two triangle sums separately uses $`2N_qN_t`$ input evaluations and $`N_q`$ products. Sharing those sums across covariance entries gives further reuse. These counts compare two organizations of the same product rule, not the unknown point count of adaptive cubature or a measured speed factor.

The source chooses its upper limit from both aperture triplets. A cached $`J`$ therefore cannot be keyed only by its own two radii if exact agreement with that finite integration domain is required. Include the common cutoff, or adopt a single converged domain for the calculation and measure the change from the historical domain. Equality of infinite integrals does not establish equality of differently truncated numerical sums. Fixed quadrature and interpolation of $`J`$ are further numerical choices that require refinement checks.

There is an equivalent form that reveals reuse with the data vector. Define the spectrum of a two-at-one-center/one-at-the-other correlation:

```math
\mathcal S_{ab|c}(q)=\widetilde U_c(q)
\int\frac{d^2p}{(2\pi)^2}\,
\widetilde U_a(p)\widetilde U_b(|\boldsymbol q-\boldsymbol p|)
B_{ijk}(\boldsymbol p,\boldsymbol q-\boldsymbol p,-\boldsymbol q),
```

```math
\mathcal B_{ab|c}(r)
=\int_0^\infty\frac{q\,dq}{2\pi}J_0(qr)\mathcal S_{ab|c}(q).
```

Here $`B_{ijk}`$ is the projected bispectrum with the source labels attached to the displayed legs. A full-aperture mean is $`\langle X_aX_bX_c\rangle=\mathcal B_{ab|c}(0)`$. The same functions supply the nine $`BB`$ products in the cubic covariance and the $`PB`$ cross-covariance. They should have one physical definition and one normalization record across those uses.

For example, a cross-pair partition in the aperture second-/third-order covariance is $`K_{a|c}(r)\mathcal B_{b|de}(r)`$. Its large-area integral is a one-dimensional product of their spectra. An internal pair in the triple gives $`K_{c|d}(0)\mathcal B_{ab|e}(r)`$. Its full-plane integral can vanish for compensated filters and regular zero-mode cumulants; a finite window generally retains it. Preserve all nine partitions before applying a justified large-area limit.

This is a route for the aperture-variance reference and its cross-block. The requested uncompressed $`\xi_\pm`$ vector has separation-bin and shear-spin kernels. Insert those kernels into the two-point side of the derivation rather than replacing the measured two-point statistic by aperture variance. For finite triangle-cut third moments, use the measurement-operator route in [Chapter 21](21_further_execution_and_cache_design.md); the full-filter formulas here are independent references, not replacements for that estimator.

## 4. Reduce the power–trispectrum term with pair-profile transforms

Let $`\Phi_a(\boldsymbol u|M,\chi)`$ be the filtered contribution of one halo of mass $`M`$ at comoving distance $`\chi`$, with all mass, lensing, and distance factors included as in Chapter 15. Define products of profiles at the same offset,

```math
f_{ab}(\boldsymbol u)=\Phi_a(\boldsymbol u)\Phi_b(\boldsymbol u),
\qquad
f_{abc}(\boldsymbol u)=\Phi_a(\boldsymbol u)\Phi_b(\boldsymbol u)\Phi_c(\boldsymbol u).
```

These are pointwise products, not new Fourier convolutions to integrate separately. For spherical halos and circular filters they are radial functions. Their Fourier transforms are one-dimensional Hankel integrals. The one-halo connected four-field correlation is

```math
\mathcal T^{1h}_{ab|cd}(r)
=\int d\nu\int d^2u\,f_{ab}(\boldsymbol u)f_{cd}(\boldsymbol u+\boldsymbol r),
\qquad
d\nu=\chi^2d\chi\,dM\,\frac{dn}{dM}.
```

The convolution theorem converts its offset integral into a product of the two profile transforms. Its radial spectrum is

```math
\widetilde{\mathcal T}^{1h}_{ab|cd}(q)
=\int d\nu\,\widetilde f_{ab}(q)\widetilde f_{cd}(q)^*.
```

For a cross-pair partition of the cubic covariance in the large-area limit,

```math
C^{P P_4,1h}_{abc,def}\big|_{a|d}
=\frac1A\int_0^\infty\frac{q\,dq}{2\pi}
\left[\widetilde U_a(q)\widetilde U_d(q)P_{ad}(q)\right]
\int d\nu\,\widetilde f_{bc}(q)\widetilde f_{ef}(q)^*.
```

Here $`P_{ad}`$ is the projected signal-plus-noise spectrum with the specified noise contractions. This expression replaces the source's two internal triangle integrations by pair-profile transforms. It retains the one-halo approximation already present in [the four-profile physical kernel](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/halomodel.cu#L401), whose radial integral is nested inside [the six-dimensional `T5` integrand](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/apertureStatisticsCovariance.cu#L2151).

The line-of-sight projection of $`P_{ad}`$ remains independent of the line-of-sight projection inside the trispectrum. Combining both into a single common redshift integral would change the physical term. The same restriction applies to the two projected bispectra in $`BB`$. Sharing inputs does not mean forcing independent projection variables to coincide.

For a finite center window, construct $`\mathcal T^{1h}_{2|2}(r)`$ and integrate its product with $`K(r)\mathcal W(r)`$. The six internal-pair partitions instead use $`\mathcal T^{1h}_{1|3}`$, obtained from $`\widetilde\Phi_a\widetilde f_{bcd}^*`$, multiplied by $`K(0)`$. These are the profile products needed by the source's finite-field `T6` structure. The single compensated leg makes their full-plane integral vanish under a regular zero-mode limit; retain the window before making that reduction.

The physical input matters. Replacing a full connected trispectrum by its one-halo component would be a modeling approximation. Reorganizing the source's existing one-halo trispectrum by these identities is an algebraic reduction. Adding perturbative or multi-halo trispectra requires their own connected functions, or additional separable representations with measured truncation error.

## 5. Extend the connected halo factorization to a finite center window

For an estimator label $`I`$, define $`f_I`$ to contain its two or three filtered halo factors. Its integrated halo response in Chapter 15 is $`H_I=\widetilde f_I(0)`$. With a finite common center window, define

```math
G(\boldsymbol q)=\frac{|\widetilde w(\boldsymbol q)|^2}{A_w^2}.
```

The connected one-halo contribution for a four-, five-, or six-field covariance is

```math
C^{1h}_{IJ}
=\int d\nu\int\frac{d^2q}{(2\pi)^2}
G(\boldsymbol q)\widetilde f_I(q)\widetilde f_J(q)^*
=\int d\nu\int_0^\infty\frac{q\,dq}{2\pi}
\overline G(q)\widetilde f_I(q)\widetilde f_J(q)^*.
```

The second equality uses isotropic halo-product transforms and the angular average $`\overline G`$. In the large-area limit, $`G`$ approaches $`(2\pi)^2\delta_D^{(2)}(\boldsymbol q)/A`$, giving $`\int d\nu H_IH_J/A`$. Thus one set of profile-product transforms can supply the finite-window expression and its large-area check. A finite window does not force a return to the source's seven-dimensional triangle integral.

Positive mass and quadrature weights make this isolated contribution a weighted product of a matrix and its transpose. Stack the two- and three-point response columns before forming the three covariance blocks, so the connected cross-block and diagonal blocks use identical physical ingredients. This preserves the positive-semidefinite structure of that contribution; it does not imply that each separate $`BB`$ or $`P P_4`$ partition must be positive semidefinite.

The window in this formula acts on aperture centers while each filtered field retains its full spatial support. A real catalogue mask can remove galaxies inside the filters, and pair/triplet exclusions can depend on center and orientation. Those effects need an estimator-specific kernel. Different windows for different source combinations also invalidate a single shared $`\overline G`$. The proposed calculation should identify which window it uses and compare with the same estimator in simulations.

## 6. Factor the selected long-mode response before integrating triangles

The source's `T7_2h` calls a radial integrand containing $`I_{31}`$ for each of two independent triangles. Each $`I_{31}`$ is a mass integral of three halo profiles, weighted by halo abundance and linear halo bias. The remaining radial factor contains the source window and its long-mode variance. [Product of responses](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/halomodel.cu#L467), [mass moment](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/halomodel.cu#L893).

Exchange each triangle integral with its mass integral. At fixed redshift, prepare one filtered triple-profile response weighted by halo bias for every requested aperture row. Then the source expression has the form

```math
C^{\rm selected\ long}_{IJ}=\int dz\,D_W(z)\,R_I(z)R_J(z),
```

where $`R_I`$ contains the integrated triangle and halo-bias response, and $`D_W`$ retains the exact remaining source normalization and long-mode variance. The allocation of redshift prefactors between these functions is a convention to fix once; verify their product against the source rather than guessing the normalization of its tabulated variance.

This removes the repeated nested mass integrations at every pair of triangle configurations. It also permits sharing halo profiles with the connected one-halo term and the mean, while keeping their different mass weights. The two mass integrals in this response product are independent; the connected one-halo covariance instead uses one shared halo. Interchanging those two constructions would change the halo model.

This is the particular halo-abundance response supplied by the source. Growth, scale dilation, tidal long modes, and IA responses are separate contributions to a general super-sample covariance. Combining this term with a more complete response calculation requires checking overlap. It is not an additional independent covariance to add merely because its function name differs.

## 7. Remove source labels from halo transforms when they are only weights

For pure lensing with common matter profiles,

```math
\Phi_{i,R}(u|M,\chi)=W_i(\chi)\,\phi_R(u|M,\chi).
```

The source population changes the lensing efficiency $`W_i`$, but it does not change the halo or the aperture filter. Compute $`\phi_R`$ once per radius and physical halo node. Products such as $`\phi_R^2`$ and $`\phi_R^3`$, and their transforms, are shared by every source combination. Attach $`W_iW_j`$ or $`W_iW_jW_k`$ only during the radial projection. When the requested apertures are equal within each third moment, this avoids constructing separate radial profiles for all twenty source triplets at each of five radii.

The isolated connected six-field term offers a further exact counting reduction. With four source bins its radial weights depend on the multiset of six labels, giving $`\binom{9}{6}=84`$ possible products. Five radii give fifteen unordered radius pairs. Thus fifteen mass-integrated radius kernels and eighty-four source-window products can describe the relevant one-halo work, before mapping it into the 5,050 upper-triangle cells of the 100-entry covariance. The resulting 1,260 combinations are not symmetry classes of the entire covariance: Gaussian noise, cross-pair placement, IA, and source-dependent masks retain other distinctions.

For a deterministic NLA extension, an explicitly derived effective radial kernel can preserve some of this source factoring. A nonlinear TATT shape is not simply the same matter profile multiplied by a source weight. Keep additional field roles where needed. Halo-profile scatter must also average products of responses from the same halo, rather than multiply independently averaged responses.

At the same cosmology and physical grids, Y3 and Y6 can share the source-independent halo arrays. Their source distributions, noise amplitudes, center windows, and measurement operators remain separate inputs. A photo-z displacement can then rebuild radial weights and projected results without rebuilding halo profiles. Changing cosmology, the halo prescription, or the mass and radial grids invalidates the corresponding physical arrays.

## 8. Gaussian filters permit further pair and noise reuse

The pinned [filter definition](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/apertureStatistics.cu#L18) is

```math
\widetilde U_R(\ell)=\frac{R^2\ell^2}{2}e^{-R^2\ell^2/2},
\qquad
\widetilde U_R\widetilde U_S
=\frac{R^2S^2\ell^4}{4}e^{-(R^2+S^2)\ell^2/2}.
```

A filtered pair correlation therefore depends on the unordered source pair and unordered radius pair separately. With four bins and five common filter radii, there are at most $`10\times15=150`$ such functions. The earlier count of 210 unordered pairs of twenty combined source/filter labels is a valid upper bound, but misses this additional equivalence. It holds when filters and transfer functions are source independent; otherwise retain the full labels.

The Gaussian filter also gives a closed white-noise contribution. Let $`N^{ij}`$ be the constant angular noise spectrum, $`a=(R^2+S^2)/2`$, and $`x=r^2/(4a)`$. Then

```math
K^{N}_{iR|jS}(r)=
\frac{N^{ij}R^2S^2}{8\pi a^3}
e^{-x}\left(1-2x+\frac{x^2}{2}\right).
```

To derive it, start from $`\int_0^\infty \ell\,d\ell\,e^{-a\ell^2}J_0(\ell r)=e^{-r^2/(4a)}/(2a)`$. Differentiating twice with respect to $`a`$ supplies the extra $`\ell^4`$ in the filter product. At equal radii and zero separation the result is $`N^{ij}/(8\pi R^2)`$. This provides both an arithmetic saving and an independent normalization check for the noise table.

The expression integrates the full continuum spectrum. Finite Fourier bounds, a pixel transfer function, spatially varying noise, or a different filter require the corresponding integral. Distinct-galaxy estimator rules can remove internal noise pairings; the formula does not override those rules. Independent Gaussian noise has no connected cumulants above order two. Changing only its amplitude therefore leaves the expensive connected three-, four-, five-, and six-field tables intact, while changing Gaussian products and the $`K\mathcal T`$ and $`K\mathcal B`$ terms. Rebuild those contractions, not the halo and bispectrum calculations.

## 9. Integration domains, caches, and parallel ownership

The reductions suggest a staged calculation with explicit cached objects:

| Object | Reused by | Changes requiring a new value |
|---|---|---|
| Filter arrays and pair products | Mean, Gaussian covariance, $`BB`$, $`P P_4`$ | Radii, wavenumber grid, filter or pixel convention |
| Filtered bispectrum tables $`\mathcal S`$ | Full-aperture mean, $`BB`$, aperture cross-$`PB`$ | Physical bispectrum, source weights, filter labels, integration bounds |
| Radial halo profiles and their pointwise products | Connected terms, one-halo $`P P_4`$, selected long response | Cosmology, halo prescription, mass/radial grids, filters |
| Pair/triple profile transforms | Finite-window connected terms and one-halo four-field correlations | Profile values, transform grid, padding and normalization |
| Center-window spectrum or separation weight | Every compatible covariance contraction | Survey center window and normalization |
| Source-window products | Tomographic projection | Source distributions, distance conversion, supported IA kernel |
| Filtered pair correlations | Gaussian, $`P P_4`$, cross-$`PB`$ | Projected spectra, noise, filters, bounds |
| Positional partition maps | All covariance entries | Observable definitions, field roles, source/filter labels, estimator exclusions |

Prepare shared immutable arrays before workers read them. A task can own a physical halo block, a row of a filtered bispectrum table, or a final covariance tile. Those are different stages; activating all three parallel layers at once would oversubscribe the machine. SIMD should evaluate independent radii, wavenumbers, or output entries with shared inputs, with the summation order of each output kept explicit. [Chapter 21](21_further_execution_and_cache_design.md) gives the source-level layout and ownership choices.

Changing to common integration nodes permits sharing expensive input values across many outputs, but it changes the adaptive source algorithm. An adaptive vector-valued integral also needs per-output error scales: one large covariance entry must not determine the refinement for all smaller entries. Group outputs with compatible domains and scales, or normalize their error targets explicitly. For fixed rules, keep every singular boundary and narrow feature inside the convergence study.

The common absolute orientation can be integrated out for isotropic fields and a radialized center window. The source's Cartesian box cutoffs are not rotationally invariant, however. A polar-domain reformulation will not reproduce such a box at finite bounds unless the transformed domain is retained. Increase the bounds or explicitly compare the two truncations before assigning any difference to physics. Likewise, reorganizing halo integrals must preserve the distinction between $`dM`$ and $`d\ln M`$, whose Jacobian is $`M`$.

## 10. Additional reference discrepancies exposed by the reductions

The independent normalization provides a new source check. Wielders et al.'s [Equation 27](https://arxiv.org/html/2509.20443v1#S3.SS1) has a cross-$`PB`$ term proportional to

```math
\frac{2}{A}\int\frac{d^2\ell_1}{(2\pi)^2}
\frac{d^2\ell_2}{(2\pi)^2}\,B_\kappa P_\kappa\,\text{filters}.
```

Removing one common rotation leaves $`2/[A(2\pi)^3]`$ multiplying the two radial integrals and the relative angle. The inspected `Map2Map3_T3` integrates that angle over $`[0,2\pi]`$ but uses $`2/[A(2\pi)^2]`$. Its [filter](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/apertureStatistics.cu#L18) and [bispectrum reader](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/bispectrum.cu#L293) contain no compensating angular normalization. The source's own aperture mean uses $`(2\pi)^{-3}`$ for the same radial/relative-angle measure. This is a factor-$`2\pi`$ source/equation discrepancy to resolve with an independent integral, in addition to the closure-side cutoff issue identified in Chapter 15. [Cross driver](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/apertureStatisticsCovariance.cu#L2797), [mean normalization](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/apertureStatistics.cu#L229).

A second source finding concerns `T5_total`. Its nested branch repeats the same `th0 == th1` condition. When the first two radii coincide but the third differs, execution enters the all-equal branch and replaces the third radius by the first. [Permutation branch](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/apertureStatisticsCovariance.cu#L701). The equal-aperture forecast configuration conceals this defect. Enumerate positional partitions independently and compare an unequal-radius reference before reusing the source's multiplicities.

Neither finding establishes the effect on a published analysis: its exact source revision, settings, and executed paths are not identified here. Both matter when selecting a numerical reference for the proposed CPU port. Matching an unchecked normalization or permutation would preserve an error while making it cheaper to evaluate.

The additionally inspected `apertureStatistics.cu` also contains unresolved merge markers near its includes and a bispectrum call interface different from the older covariance reader. Its local filter and normalization expressions can be inspected, but the selected checkout must not be described as a runnable reference without resolving that source state. No compilation was attempted. [Inspected file](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/apertureStatistics.cu#L6).

## 11. Acceptance checks for the reductions

Start with one source distribution and unequal radii. Compare the direct `T4` integrand against the factorized fixed-side calculation with identical bounds and Fourier conventions. Then compare one-halo four-, five-, and six-field contributions through both Fourier and profile representations, first in the large-area limit and then for a specified center window. Retain each partition family separately so disagreement can be localized.

Test zero noise, analytic white noise, coincident filters, source-label exchanges, and changes in every closure-side cutoff. Check $`\mathcal B(0)`$ against the independent full-aperture mean. Refine profile sampling, Hankel grids, mass/redshift rules, and outer covariance contraction separately; using the same interpolation on both sides of a comparison can hide a shared error. Verify the joint covariance and Fisher quantities after these component checks.

The proposed priority is fixed-side $`BB`$ tables, shared filtered halo products, the selected response contraction, and then estimator-specific extensions. These offer reductions in the number of expensive physical evaluations. Threading and SIMD apply to the work remaining after those reductions. The roughly 50-minute covariance target remains a future acceptance test; this pass neither executed a covariance nor measured its cost.
