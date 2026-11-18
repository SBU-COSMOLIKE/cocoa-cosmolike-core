# 15. Porting the analytic covariance to CPU C and OpenMP

> **Archive status.** This source study records design proposals and validation requirements, not completed features. Read the [maintained implementation status](../threepcf.md) first. Both the mean and its covariance belong in `cosmolike/3pcf/`; earlier destination proposals using `fastnc/` or `covariances/` are superseded.

## 1. The covariance is a second numerical deliverable

The proposed project includes both the fastnc mean prediction and an analytic covariance implementation. Use CPU C, OpenMP, reusable tables, and SIMD as the primary architecture for both. CUDA is not a mathematical requirement. The covariance work should live under CosmoLike's covariance module and share physical inputs with the mean, while retaining its own numerical controls and storage. For a fixed-covariance Fisher forecast, compute this matrix at the fiducial model and reuse its factorization throughout the derivative calculation. Its cost is therefore different from a mean prediction that must be repeated at displaced parameter values.

The user's reported current covariance time is about 30 seconds. A three-point extension costing one hundred times as much would take 3,000 seconds, or 50 minutes. That is an acceptable planning target for a matrix prepared once per scientific configuration. It is not a prediction obtained by multiplying the dimension of the data vector. [Chapter 17](17_cosmolike_covariance_cpu_transfer.md) examines the actual core preparation and integration patterns against this roughly one-hour budget. Multiple scale selections can reuse submatrices of one correctly computed parent covariance when only the selection changes; a changed estimator or physical model may require new work.

The relevant public covariance repository is **[sheydenreich/threepoint](https://github.com/sheydenreich/threepoint)**. Its [releases page](https://github.com/sheydenreich/threepoint/releases) is the location linked by the third-order covariance paper. This study inspects commit [`08de9645a7294092995e5aeaf7b36e48e8fc10a3`](https://github.com/sheydenreich/threepoint/tree/08de9645a7294092995e5aeaf7b36e48e8fc10a3). It contains both CPU C++ and CUDA source, including second-/third-order aperture cross-covariance routines. The latest listed release is `v2.0.0`, published in 2022; that release name should not be assumed to identify the later cross-covariance calculation. Retain the inspected commit and obtain the publication configuration separately.

The original scientific sources have distinct roles. The [Y3 paper, Section III](https://arxiv.org/html/2503.03964v2#S3), uses a simulation-estimated joint covariance and an analytic two-point covariance for comparison and compression. The TATT paper develops the mean physical model. [Linke et al.](https://arxiv.org/abs/2212.04485) supply the third-order aperture covariance, including finite-survey terms. [Wielders et al.](https://arxiv.org/abs/2509.20443) supply the non-tomographic second-/third-order aperture cross-covariance. Their existence makes analytic covariance a concrete implementation route. The required extensions concern the requested statistic, tomography, physical ingredients, and survey estimator.

No covariance program was compiled or executed for this study. The reductions below are proposed mathematical organization, and the source discrepancies are identified by inspection. They are not measured errors in published results.

[Chapter 19](19_further_covariance_reductions.md) extends the integral study to fixed-side $`BB`$ factors, one-halo $`P P_4`$ products, finite center windows, the selected long-mode response, and analytic filtered noise. It also records additional normalization and permutation findings. [Chapter 21](21_further_execution_and_cache_design.md) specifies how the remaining integrals can share input batches and owned worker scratch.

## 2. Files and terms that should enter the port

The [repository README](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/README.md) states that its CUDA version is the fully tested release. That statement describes the implementation the authors tested; it does not imply that the integrals require a GPU. The CPU source already has batched integration callbacks with OpenMP. It is nevertheless incomplete as a reference for the entire covariance: its function named `Cov_NonGaussian` returns only `T4_total`.

| Source file | Actual role | Use in the CPU study |
|---|---|---|
| [CPU covariance source](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/classed_version/apertureStatisticsCovariance.cpp) | Gaussian `T1`, `T2`, and a non-Gaussian `T4` implementation with batched OpenMP callbacks | Numerical organization and limited independent comparisons; not the whole covariance |
| [CUDA covariance source](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/apertureStatisticsCovariance.cu) | Higher-order covariance integrands, geometry factors, permutations, and integration drivers | Principal source inventory for the additional terms |
| [Third-order driver](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/calculateApertureStatisticsCovariance.cu) | Selects and writes individual third-order covariance contributions | Reference configuration and term-by-term output specification |
| [Cross-covariance driver](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/calculateMap2Map3Covariance.cu) | Covariance of aperture variance and aperture skewness for one source distribution | Cross-block reference before adding tomography or changing the two-point statistic |
| [Halo-model source](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/halomodel.cu) | Mass function, halo profiles, and higher connected polyspectra | Physical ingredient comparison and profile-table design |
| [Cross-covariance example](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/examples/exampleMap2Map3Covariance.sh) | Separate finite-window and large-area term calls | Explains why one geometry switch does not produce every term in one run |

A covariance involves more fields than the mean it describes. Let $`P`$ denote a two-field spectrum, $`B`$ a three-field bispectrum, and $`P_n`$ a connected $`n`$-field polyspectrum. Using field count avoids ambiguity: the code's tetraspectrum is $`P_5`$, whereas its trispectrum is $`P_4`$. The covariance of two cubic statistics contains six fields. Subtracting their mean product removes one disconnected partition; it leaves Gaussian pairings, mixed connected partitions, and the connected six-field term.

| Covariance contribution | Source names | Mathematical content | Inspected outer integral structure |
|---|---|---|---|
| Third-order Gaussian, all pairs across the two estimators | `T1` | Products of three $`P`$ factors | Three dimensions in the large-area limit, six for the finite-window route |
| Third-order Gaussian, one pair across the estimators | `T2` | Three $`P`$ factors with different pair placement | Products of one-dimensional integrals and a one- or two-dimensional window integral |
| Third-order bispectrum product | `T4` | $`BB`$ partitions remaining after mean subtraction | Five dimensions in the implemented large-area route |
| Third-order power–trispectrum | `T5`, `T6` | $`P P_4`$, with distinct window dependence | Six dimensions, or a one-dimensional factor times a seven-dimensional integral |
| Third-order connected six-field | `T7` | One-halo $`P_6`$ model | Seven outer dimensions, with additional radial work in the physical kernel |
| Additional long-mode contribution | `T7_2h` | Selected two-halo/long-mode part of the six-field model | Six outer dimensions plus prepared halo/window responses |
| Aperture second-/third-order cross-block | `Map2Map3_T2`, `Map2Map3_T3` | $`PB`$ partitions | Factorized finite-window term and three-dimensional large-area term |
| Connected cross-block | `Map2Map3_T4` | One-halo $`P_5`$ model | Five outer dimensions plus physical radial work |

These dimensions describe the inspected algorithms, not lower bounds on the work. The source uses different geometry approximations for different terms: for example, several non-Gaussian terms accept only the large-area option, while the added long-mode term has its own finite-window path. Preserve a term-specific geometry record. “Finite survey covariance” cannot mean that every contribution uses an exact DES mask merely because one Gaussian term does.

## 3. Why the CPU approach should reorganize the integrals

The CUDA wrapper passes quadrature points from a CPU integrator to device arrays, evaluates one integrand value per point, copies the results back, and frees the device arrays. In a CPU port, the same callback can read the supplied points directly. OpenMP workers own disjoint result entries; SIMD lanes process adjacent points. Shared spectra, profile tables, and window tables are prepared before entering the callback. This is the same division between serial preparation and independent repeated arithmetic used elsewhere in CosmoLike.

The existing CPU callback in [`integrand_T1`](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/classed_version/apertureStatisticsCovariance.cpp#L407) already demonstrates this pattern. The CUDA source's [host callback](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/apertureStatisticsCovariance.cu#L1358) supplies the corresponding transfer-based implementation. A C port should retain neither device transfers nor large variable-length stack arrays used to stage those transfers.

CosmoLike's existing Gauss–Legendre and transform machinery is suitable for many reduced integrals. It is not efficient to replace every seven-dimensional integral by an unexamined tensor product: $`n`$ nodes per coordinate produce $`n^7`$ points. First reduce dimensions analytically, then compare fixed quadrature, transformed-coordinate quadrature, or adaptive multidimensional cubature for the remaining integral. The inspected core has no linked cubature interface; retaining a small CPU cubature reference is an explicit dependency choice if needed, not a reason to require CUDA.

There are two particularly useful reductions to derive before implementing the most expensive multidimensional loops. They also provide independent checks of the existing formulas.

## 4. Gaussian covariance from reusable filtered correlations

For this derivation, assume statistically homogeneous scalar E-mode fields, fixed source populations, uniform noise within each population, and the full aperture filter. Label a filtered field by $`a=(i,R)`$, its source bin and filter radius. Define its correlation with field $`b=(j,S)`$ at separation $`r`$ by

```math
K_{ab}(r)=\int_0^\infty\frac{\ell\,d\ell}{2\pi}
J_0(\ell r)\,\widetilde U_R(\ell)\widetilde U_S(\ell)
\left[C_\ell^{ij}+N^{ij}\right].
```

Here $`J_0`$ is a cylindrical Bessel function, $`\widetilde U`$ is the Fourier aperture filter, $`C_\ell^{ij}`$ is the projected signal spectrum, and $`N^{ij}`$ is the noise spectrum. For disjoint source samples with uncorrelated intrinsic ellipticities, the noise matrix is diagonal. Its amplitude must use the same one- versus two-component ellipticity convention as the survey input. Overlapping weighted samples need their actual cross-noise matrix.

Let $`w(\boldsymbol x)`$ weight allowed aperture centers and let $`A_w=\int d^2x\,w(\boldsymbol x)`$. The normalized separation weight is

```math
\mathcal W(\boldsymbol r)=\frac{1}{A_w^2}
\int d^2x\,w(\boldsymbol x)w(\boldsymbol x+\boldsymbol r).
```

The Gaussian covariance of spatial averages of three filtered fields is obtained by Wick's rule: pair the six fields in every possible way. Six pairings connect all three fields in the first estimator to fields in the second. The other nine contain one cross-pair and one internal pair in each estimator. In compact notation,

```math
C^{\rm G}_{\boldsymbol a,\boldsymbol b}=\int d^2r\,\mathcal W(\boldsymbol r)
\left[
\sum_{\pi\in S_3}\prod_{i=1}^3 K_{a_i b_{\pi(i)}}(r)
+\sum_{i=1}^3\sum_{j=1}^3 K_{a_i b_j}(r)
K_{a_{i'}a_{i''}}(0)K_{b_{j'}b_{j''}}(0)
\right].
```

Here $`\boldsymbol a=(a_1,a_2,a_3)`$ and $`\boldsymbol b=(b_1,b_2,b_3)`$ label the three positions in each estimator. The primed indices are the two positions remaining after selecting $`i`$ or $`j`$. For example, $`i=j=1`$ gives $`K_{a_1b_1}(r)K_{a_2a_3}(0)K_{b_2b_3}(0)`$. Sum over positions even when their source/filter labels coincide. If all six labels are identical, the bracket is $`6K(r)^3+9K(r)K(0)^2`$. For isotropic $`K`$, average the separation weight over orientation and perform a one-dimensional radial contraction.

Including noise in every $`K`$ gives the covariance of raw noisy filtered-map products. A catalogue estimator that excludes repeated galaxy indices, or an estimator with explicit noise subtraction, can remove within-estimator noise pairings. Its allowed contractions must be derived from that definition. The formula above supplies a transparent Gaussian reference; it is not a claim that every noise contraction can be transferred unchanged to TreeCorr's estimator.

This gives a concrete CPU plan: compute filtered pair correlations once using Hankel transforms or checked quadrature, retain their values on a common separation grid, and assemble covariance rows from products and weighted sums. Four source bins and five radii define twenty filtered fields, requiring at most 210 distinct symmetric pair functions. Those functions feed all 5,050 independent entries of the 100-element Y6 equal-aperture covariance. The corresponding Y3 covariance has 3,240 independent entries for eighty observables. These counts describe storage and reuse, not measured runtime.

The one-cross-pair terms explain why a finite survey cannot always be replaced by an inverse-area prefactor. A compensated aperture has zero Fourier response at zero wavenumber. In the ideal large-area limit, an unweighted full-plane integral of the cross-pair correlation can then vanish. A finite separation window changes that integral. The model should retain the window before taking the limit, and compare the reduced correlation calculation with the direct Gaussian integrals term by term.

This derivation is for the specified aperture-field estimator. A masked catalogue statistic constructed from truncated, binned shear three-point functions has a different measurement operator. The covariance port must establish the relation to that operator before using this formula as the survey covariance.

## 5. A proposed one-halo reduction for the connected terms

The connected one-halo term has a useful factorization. Define $`\Phi_a(\boldsymbol\theta\mid M,\chi)`$ as one halo's contribution to the filtered field labeled $`a`$, including all lensing amplitudes, distances, mass normalization, and filter factors. In a flat comoving description, the Poisson intensity of halos per solid angle is

```math
d\nu=\chi^2\,d\chi\,dM\,\frac{dn(M,z)}{dM}.
```

The mass function $`dn/dM`$ counts halos per comoving volume per mass. Define the integrated responses of one halo to an aperture-second-moment and aperture-third-moment measurement,

```math
H_{ab}(M,\chi)=\int d^2\theta\,\Phi_a(\boldsymbol\theta)\Phi_b(\boldsymbol\theta),
\qquad
H_{abc}(M,\chi)=\int d^2\theta\,\Phi_a(\boldsymbol\theta)
\Phi_b(\boldsymbol\theta)\Phi_c(\boldsymbol\theta).
```

For a large survey of solid angle $`A`$, the fully connected one-halo covariance takes the form

```math
C^{1h}_{33;abc,def}=\frac1A\int d\nu\,H_{abc}H_{def},
\qquad
C^{1h}_{23;ab,cde}=\frac1A\int d\nu\,H_{ab}H_{cde}.
```

The reason is that the same halo supplies every field in this connected contribution. Integrating its center over a large survey supplies one factor of area; the two spatial averages supply two inverse factors. The integrations over the offsets of the two apertures then separate. Mean subtraction removes contributions disconnected between estimators, not the single halo connecting all five or six fields.

This factorization is a proposed alternative to the inspected high-dimensional Fourier calculation. For circular halo profiles, computing $`\Phi`$ requires radial transforms, and computing each $`H`$ requires a radial product integral. The remaining covariance is a mass/redshift contraction. If $`X_{qI}=\sqrt{w_q/A}\,H_I(M_q,\chi_q)`$ includes positive quadrature and halo-count weights, the joint connected one-halo block is $`X^{\mathsf T}X`$, with columns for both second- and third-order responses. Its rectangular cross-block is $`X_2^{\mathsf T}X_3`$; its two-point block must consistently contain $`X_2^{\mathsf T}X_2`$. This form shares profile calculations across covariance entries and makes the joint contribution positive semidefinite. A scalar loop, SIMD across nodes, and a controlled matrix contraction are all CPU implementations of the same expression. If CosmoLike already supplies the corresponding two-point one-halo term, compare the prescriptions and use that contribution once rather than adding a duplicate.

Several restrictions prevent overstating this reduction. It covers the connected one-halo contribution, not the Gaussian, $`BB`$, $`P P_4`$, or additional long-mode terms. Halo-profile scatter requires averaging the same-halo product $`\langle H_IH_J\rangle`$, not multiplying separately averaged responses. Correlated halo counts or halo exclusion require their own terms. The formula for $`H_{ab}`$ is for aperture variance; the requested $`\xi_\pm`$ cross-block requires its pair-separation and spin weights inside the two-point halo response. IA fields, a finite survey mask, and triangle cuts likewise require a corresponding estimator derivation.

The inspected Fourier implementation also imposes cutoffs on selected independent Fourier legs, while some closure legs have no identical explicit cutoff. Filtering every halo profile at one common maximum wavenumber therefore changes that finite-domain approximation. Establish the untruncated identity and convergence as the Fourier bounds are enlarged; do not expect bitwise agreement between differently truncated representations. Before adding a separate super-sample covariance, map its response contribution against `T7_2h`. A selected two-halo term is neither automatically the whole super-sample covariance nor automatically disjoint from it.

As a normalization check, use the Fourier convention $`\Phi(\theta)=\int d^2\ell\,\widetilde\Phi(\ell)e^{i\ell\cdot\theta}/(2\pi)^2`$. The product integrals then contain their Fourier factors internally; none should be added outside the real-space covariance above. Write each thin-lens profile as $`\widetilde\Phi_a=W_i(\chi)(M/\bar\rho)u(\ell/\chi)\widetilde U_R(\ell)/\chi^2`$. Six such factors, combined with the volume factor $`\chi^2`$, give the connected-six-field measure $`\chi^{-10}\prod_iW_i`$; five give $`\chi^{-8}\prod_iW_i`$. Since $`W_i=[3\Omega_mH_0^2/(2c^2)]\chi(1+z)g_i`$, this agrees with the source's explicit $`g^6/\chi^4`$ and $`g^5/\chi^3`$ times their lensing prefactors. Here $`g_i`$ is the dimensionless lensing efficiency. Check these profile conventions against the [source halo model](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/halomodel.cu) before judging speed or agreement.

## 6. Extending to the actual Y3 and Y6 vectors

Tomography requires carrying a source label on every field, through every contraction. A covariance entry between two equal-aperture triplets has six source labels; a two-point–three-point entry has five. These labels determine which projected spectra, noise contractions, and radial products occur. Multiplying a one-bin covariance by an effective lensing amplitude generally loses that structure. A useful first check makes all source distributions identical: the signal should reduce to the one-population result, while the noise still follows whether the catalogues are shared or disjoint.

The target two-point observable remains the full selected $`\xi_\pm`$ vector. Availability of an aperture-variance cross-covariance does not justify replacing it by a few aperture variances. Write the two-point estimator kernel explicitly and insert it into the same five-field cumulant calculation. For the Gaussian field that cross-covariance vanishes; the $`PB`$ and connected-five-field terms do not generally vanish for the evolved density field. Keep the aperture-variance result as a comparison observable for the extension.

The finite three-point measurement matters equally. The Y3 reproduction removes triangles below its stated side cut before aperture integration. Let $`\Gamma`$ stack the real and imaginary natural-component entries as a real vector. If the real matrix $`A_3`$ maps those measured bins to the aperture vector, then

```math
C_{33}=A_3 C_{\Gamma\Gamma} A_3^{\mathsf T},
\qquad C_{23}=C_{\xi\Gamma}A_3^{\mathsf T}.
```

These identities are exact for a fixed linear measurement operator and matching covariance. They do not require allocating the enormous uncompressed three-point covariance: contract the weights inside its integrals or over blocks. They also do not permit replacing $`C_{\Gamma\Gamma}`$ by an unrelated infinite-domain bispectrum covariance without an estimator derivation. The same bin widths, triangle-side exclusions, phases, and weights must appear in the mean and covariance.

For a first physical covariance, separate a pure-lensing reference, an NLA extension, and any TATT treatment. In a deterministic NLA prescription, radial kernels can supply some extensions directly. A general TATT covariance introduces mixed higher-order shape cumulants that are not supplied merely by its mean bispectrum. A fiducial lensing-plus-noise covariance can still define a forecast, provided that approximation is stated and its effect is tested. Do not label it a full TATT covariance by virtue of the mean model's parameter names.

## 7. Source discrepancies to resolve before reference matching

The following findings were checked in source without executing the calculations. They motivate independent references rather than blind reproduction of every returned number.

| Source behavior | Consequence and affected scope | Required check |
|---|---|---|
| Cross-covariance `Map2Map3_T3` computes $`\ell_3^2`$, compares it with `dev_lMax`, and only then takes the square root | The compared cutoff is inconsistent with the radius passed to `dev_lMax`; this affects the implemented large-area $`PB`$ cross term | Evaluate support at known triangles, then compare corrected and historical terms with an independent integral |
| `T2` changes a local upper-limit variable but reuses the old `vals_max1` array in later factors | Unequal apertures need not receive the intended integration bounds; the behavior occurs in both CPU and CUDA sources | Change each factor's cutoff independently and exchange covariance legs |
| CPU `T4_total` contains a purported permutation with labels `(2,1,1)` rather than three distinct second-triplet labels | Equal apertures hide the repeated/omitted argument | Enumerate all nine residual bispectrum partitions and check unequal-radius permutation symmetry |
| Rectangular `T2` and `T6` can call an internal callback that accepts only circle or square | A listed geometry reaches an exit path | Test the supported geometry for each term, rather than only parsing the top-level option |
| Global bounds and geometry are mutated, with host and device lower bounds updated at different points | Concurrent entries and evaluation order require explicit ownership | Use immutable per-integral bounds and compare fresh versus sequential calculations |
| Integration status and reported error are not consistently checked; some target relative errors are 0.1 | A returned number is not an accuracy certificate | Propagate status and errors; refine the matrix and its Fisher consequences |
| Circular window factor evaluates a Bessel ratio at zero without its analytic limit | Zero window wavenumber needs a defined value | Check the limit and nearby small arguments |

The relevant locations are [cross-term support](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/apertureStatisticsCovariance.cu#L2867), [factor bounds](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/apertureStatisticsCovariance.cu#L1108), [CPU permutation](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/classed_version/apertureStatisticsCovariance.cpp#L253), and [geometry callback](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/apertureStatisticsCovariance.cu#L1420). The paper-validation run may have used different revisions or settings. No numerical impact on that run is asserted here.

## 8. Port sequence and numerical acceptance

Begin with a written term list and input units. Reproduce a one-source Gaussian reference using both direct quadrature and the filtered-correlation reduction. Add the connected one-halo terms with a separate Fourier/profile comparison. Then implement the remaining $`BB`$, $`P P_4`$, and specified long-mode contributions, preserving individual outputs. Only after those checks should the same engine acquire tomographic labels, the $`\xi_\pm`$ cross-block, and the measured finite-bin operator. Survey geometry and IA extensions must remain named assumptions at every stage.

The CPU implementation should precompute spectrum and profile grids, source kernels, filter values, mask-separation weights, and halo mass/redshift factors. Their cache keys include the physical model and the estimator. Assign either covariance blocks or integrand batches to OpenMP, with one owner of each output; do not allow several workers to mutate a shared integration context. Batch several aperture combinations when the expensive spectrum or halo evaluation is identical. Keep fixed reduction rules and test scalar versus SIMD arithmetic before measuring throughput.

Numerical covariance accuracy should be assessed in the metric used by inference. If $`C=LL^{\mathsf T}`$ is a trusted positive-definite covariance and $`\delta C`$ is a numerical change, define

```math
E=L^{-1}\delta C L^{-\mathsf T}.
```

For fixed mean derivatives, a small covariance error changes the data Fisher matrix by

```math
\delta F=-J^{\mathsf T}C^{-1}\delta C C^{-1}J+O(\delta C^2).
```

The eigenvalues of $`E`$ measure errors relative to covariance directions, including directions with small variance. Small diagonal-entry errors alone can miss a damaged correlated direction. Report the largest absolute eigenvalue, symmetry, positive definiteness on the retained data subspace, conditional cross-block covariance, and changes in marginalized errors and FoM. A tolerance such as a sub-percent change in the final forecast is a proposed numerical acceptance target, not a claim of sub-percent accuracy of the halo model.

The output should distinguish a historical reference, a corrected numerical reference, and a survey forecast with stated physical approximations. Retain simulation comparisons for validation; an analytic model removes the need to estimate every production matrix element from mocks, not the need to test the model. No Hartlap correction should be attached merely because a deterministic analytic covariance is being used; any finite-simulation component needs its own justified statistical treatment.

[Chapter 18](18_precision_matrix_expansion.md) develops the proposed simulation correction through precision matrix expansion. It uses this joint analytic covariance as a fixed reference and estimates corrections without inverting the sample covariance. That method provides a concrete way to test the full two-point vector before choosing compression. Its convergence, simulation scatter, and summary-vector sampling assumptions require independent checks in addition to the numerical quadrature checks above.

Credit the covariance work to its authors, including Laila Linke, Sven Heydenreich, Pierre Burger, Peter Schneider, Niek Wielders, and Lucas Porth as appropriate to the cited papers and reused source. The threepoint README states GPL version 3, while the inspected CosmoLike core declares MIT. Record the distribution decision and inherited notices explicitly before copying source; do not relabel translated source simply because its destination has a different license. This study makes no source-distribution change.
