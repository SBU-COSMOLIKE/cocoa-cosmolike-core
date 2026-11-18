# 17. What the CosmoLike covariance implementation contributes to a CPU port

> **Archive status.** This source study records design proposals and validation requirements, not completed features. Read the [maintained implementation status](../threepcf.md) first. Both the mean and its covariance belong in `cosmolike/3pcf/`; earlier destination proposals using `fastnc/` or `covariances/` are superseded.

The existing covariance code gives a concrete basis for a CPU implementation of aperture-statistic covariance. Its main savings come from computing physical quantities before entering the catalog loops: one matter calculation serves many tomographic combinations, one halo profile serves many moments, and one transform serves many outputs. OpenMP then distributes independent results, while SIMD evaluates several results with shared inputs. The proposed three-point module should reproduce this arrangement of work, extending the statistical expressions where necessary.

The user's approximately 30-second covariance calculation establishes a practical reference, and the stated acceptance of a calculation taking 100 times longer means that about 50 minutes can be acceptable. That allowance changes the design decision: the first target can be an accurate CPU covariance calculated once at the fiducial model, rather than a covariance recomputed in seconds at every parameter point. It does not establish how long the larger calculation will take. This chapter inspects source only; it performs no timing, imports, compilation, or scientific calculations. The inspected core revision is `0c0cf3838852396933c7f49550e35917dad0537c`, with the existing working-tree modifications described in Chapter 14 left untouched.

## 1. Follow the calculation from the survey to the numerical kernels

Let $`x_2`$ denote a vector of second-order observables and $`x_3`$ the aperture third moments. Their joint covariance has blocks $`C_{22}={\rm Cov}(x_2,x_2)`$, $`C_{33}={\rm Cov}(x_3,x_3)`$, and $`C_{23}={\rm Cov}(x_2,x_3)`$. The inspected core assembles $`C_{22}`$. It does not implement the additional blocks merely by accepting more rows: its observable rows contain two catalog indices, and its connected matter input is a trispectrum. The reusable work is below and around those particular statistical expressions.

For example, a third moment on each side supplies six shear factors. In the Gaussian approximation those factors form products of three two-point correlations, with their allowed pairings and measurement filters. A second moment against a third supplies five factors; its expectation vanishes for a zero-mean Gaussian shear field. The physical cross-covariance therefore needs non-Gaussian contributions, including products involving the bispectrum and a connected five-point term. Reusing the existing two-spectrum Wick formula for that block would give the wrong statistical object. Reusing its input preparation, task allocation, and matrix contraction is a different operation and does not make that mistake.

The actual survey driver calls the following stages. First, `covariance_spectra` constructs common radial geometry, catalog windows, and all crossed field spectra. Optional non-Limber, intrinsic-alignment, and one-loop galaxy-bias terms enter the selected Gaussian spectra. Second, angular-bin or Fourier-band operators transform those spectra and their Wick products. Third, `_matter_covariance_tables` builds halo moments, tree-level angular averages, the connected matter trispectrum, and the power response to a long density fluctuation. It projects the matter functions before attaching catalog windows. Finally, separate contractions assemble the connected and super-sample matrices, and the driver returns their sum with the Gaussian matrix. [Driver and return values](../../../../../cosmolike_notebook_utils/covariance/survey.py); [shared matter stage](../../../../../cosmolike_notebook_utils/covariance/survey.py).

This separation matters physically. The Gaussian spectra may contain TATT and non-Limber corrections, but the inspected survey's connected and super-sample terms use zero intrinsic alignment, linear galaxy bias, and Limber projection. The driver requires massless neutrinos and rejects nonzero magnification. Its adopted footprint is a spherical cap. These conditions define what the existing calculation contains; a short runtime under these choices is not a measurement for a full TATT six-point covariance with a DES mask. The three-point study should report its own term inventory just as explicitly. [Survey model and guards](../../../../../cosmolike_notebook_utils/covariance/survey.py).

## 2. Mechanisms that transfer, and the expressions they do not supply

| Existing source mechanism | Work it avoids | Application to $`C_{33}`$ and $`C_{23}`$ |
|---|---|---|
| Cumulative lensing integrals, `spectra_cov.c:504` | A separate source-distance integral for every foreground shell | Prepare each source-bin efficiency once; retain its redshift and normalization conventions. |
| Shared power rows and windows, `spectra_cov.c:1283` | Repeated matter-power reads for every catalog pair | Share power, bispectrum ingredients, and radial factors across all triplets. The three-point physics needs its own products. |
| Halo abundance and concentration tables, `halo_cov.c:407` | Repeated halo statistics at the same mass and redshift | Prepare the halo inputs for four-, five-, and six-leg terms once per mass/redshift sample. |
| Shared NFW profiles, `halo_cov.c:495` | Repeated profile evaluation for every moment or pair | Reuse profiles in filtered-halo constructions and products of higher order. |
| Five moment accumulators in one mass loop, `halo_cov.c:684` | Five independent passes over the same profiles | Group covariance contributions that use the same profiles and mass weights. |
| Streamed tree averages, `spectra_cov.c:407` | A full power table for every pair and angle at every shell | Evaluate bounded blocks of higher-order geometries, consume their powers immediately, then reuse scratch. |
| Interpolation compressed into measurement operators, `survey.py:92` | A dense interpolated covariance before projection | Precontract fixed aperture/bin weights with interpolation where the operation is linear. This is not compression of the observed data vector. |
| Triangular tasks and tiled projection, `assembly_cov.c:57`, `gaussian_cov.c:235` | Duplicate matrix entries and repeated input loads | Own one covariance block and its transpose; accumulate separate entries in SIMD lanes. |
| Shared response contraction, `survey.py:849` | Separate radial integrations for every SSC entry | Use joint second- and third-order response rows, after deriving the third-order response. |
| FFT workspace and Mellin recurrences, `fftlog_cov.c:147` | Repeated forward transforms, plans, and special functions | Transfer plan/scratch ownership and recurrence methods. The spherical-Bessel non-Limber kernel itself is not an aperture or six-point kernel. |

The table names execution mechanisms rather than promising speed factors. Their effect depends on how much repeated work the proposed calculation contains. The source already shows each mechanism operating on shared arrays; the remaining task is to arrange the additional covariance formulas into similarly reusable pieces.

## 3. Halo preparation is more useful than repeatedly calling halo integrals

For a halo of mass $`M`$, let $`u(k|M,a)`$ be its normalized Fourier density profile, $`n(M,a)`$ its abundance, and $`b(M,a)`$ its linear bias. A halo moment integrates products of profiles with powers of $`M/\bar\rho`$. The core prepares mass quadrature, abundance, bias, and concentration, then computes $`u`$ once at each requested $`(a,k,M)`$. Its later mass loops read these profiles to accumulate $`I_{11}`$ and five pair moments. Increasing the number of moment roles therefore need not multiply the cost of profile evaluation by the same factor. [Preparation and moment ownership](../../../../../cosmolike/covariances/halo_cov.c).

The underlying NFW reader supplies another saving. `halo.c` tabulates combinations of the sine and cosine integrals on a logarithmic argument grid. Its profile reader uses those tables and stable algebra instead of evaluating two pairs of special functions from scratch for every halo and wavenumber. The table is built before parallel profile reads; changing the numerical-table state rebuilds it. An upstream-compatible calculation could use this numerical representation with the agreed mass definition and concentration. Calling the core's entire halo prescription would do more: its abundance, bias, cold-matter convention, and low-mass completion differ from the upstream prescription and must be compared separately. [NFW table](../../../../../cosmolike/halo.c); [profile reader](../../../../../cosmolike/halo.c).

Define $`N_a`$, $`N_k`$, and $`N_M`$ as the counts of scale factors, wavenumbers per scale factor, and mass nodes. There are $`N_p=N_k(N_k+1)/2`$ unordered wavenumber pairs. The source requires approximately $`N_aN_M`$ halo-statistics evaluations, $`N_aN_kN_M`$ profile evaluations, and $`5N_aN_pN_M`$ moment-accumulator updates, plus the separate $`I_{11}`$ sums. A scalar update is not a unit of runtime: interpolation, transcendental functions, and multiply-adds have different costs. The counts identify which operations should be reused and which grow quadratically with $`N_k`$.

The driver holds eight radial shells at a time. At the baseline 96-node rule, the default mass panels contain eleven tail panels with 32 nodes and ten ordinary panels with 96 nodes, giving $`N_M=1312`$. With $`N_k=128`$, the profile block occupies approximately $`8\times128\times1313\times8=10.8`$ MB before padding; the five moment arrays occupy about 2.6 MB. These are analytic allocation estimates, not measured peak memory. The temporary halo arrays are freed on return. Keeping them across related calculations would require an explicit workspace and invalidation rules; the current implementation does not provide a persistent covariance cache. [Mass rule](../../../../../cosmolike/covariances/halo_cov.c); [eight-shell batching](../../../../../cosmolike_notebook_utils/covariance/survey.py).

For the one-halo $`C_{33}`$ and $`C_{23}`$ terms, this preparation can feed the filtered halo profiles and integrated products discussed in the covariance chapters. That route avoids enumerating every six-leg Fourier configuration when the one-halo expression factorizes. It must retain the intended profile, abundance, and mass limits. The core's Wynn extrapolation and completion enforce a particular large-scale normalization of $`I_{11}`$; they are not a reason to extrapolate every higher moment or replace the upstream mass model silently.

## 4. Geometry, interpolation, and projections should precede catalog loops

The perturbative angular calculation illustrates how to remove work without dropping a physical term. At a radial shell, $`K=(\ell_1+1/2)/f_K`$ and $`Q=(\ell_2+1/2)/f_K`$, where $`f_K`$ is the transverse comoving distance. Hence

```math
\log_{10}|\boldsymbol K+\boldsymbol Q|
=\log_{10}\sqrt{(\ell_1-\ell_2)^2+
2(\ell_1+1/2)(\ell_2+1/2)(1+\cos\varphi)}
-\log_{10}f_K.
```

The driver calculates the angular expression once. Each shell supplies only the additive distance shift. The power reader also interpolates the redshift part once to make a one-dimensional slice, then performs arithmetic indexing along the log-wavenumber axis. The tree calculation combines the two potentially singular quadratic-kernel terms algebraically and uses an analytic planar average for the relevant cubic kernel. These operations preserve the specified four-point model while reducing repeated logarithms and cancellations. Their particular formulas apply to the covariance parallelogram; higher-point configurations require a fresh derivation before adopting an analogous reduction. [Shared geometry](../../../../../cosmolike_notebook_utils/covariance/survey.py); [power slice](../../../../../cosmolike/covariances/spectra_cov.c); [perturbative algebra](../../../../../cosmolike/covariances/perturbation_cov.c).

At $`N_k=128`$ and the baseline 20 panels of 96 angular nodes, $`N_p=8256`$ and $`N_\varphi=1920`$. A full $`P(|K+Q|)`$ table would have 15,851,520 entries, about 126.8 MB. Instead, `tree_averages_logk_cov` fills 128 pairs at a time, requiring about 1.97 MB of power scratch, and immediately integrates those rows. The geometry table remains shared. The source therefore offers a specific design for higher-dimensional covariance: retain reusable geometry, stream model-dependent values in blocks, and reduce each block before moving on. At baseline settings the existing reader rebuilds its redshift slice for each 128-pair block; a longer-lived shell slice is a candidate for further reuse, subject to unchanged interpolation and ownership. [Block driver](../../../../../cosmolike/covariances/spectra_cov.c).

The angular projection has a separate algebraic saving. If $`H`$ linearly interpolates a coarse trispectrum $`T`$ onto dense multipoles and $`K`$ projects dense multipoles into measured bins, then

```math
K(H T H^{\mathsf T})K^{\mathsf T}
=(KH)T(KH)^{\mathsf T}.
```

The code forms $`KH`$ once, summing every integer multipole with its bin kernel. It avoids storing the dense interpolated $`T`$ and preserves the chosen interpolation exactly. The approximation is the coarse representation of $`T`$, whose grid needs convergence tests. This is not MOPED or a reduction of the number of measured bins. An aperture-covariance module should similarly absorb fixed filter and bin weights into interpolation or transform operators wherever linearity permits. [Operator construction](../../../../../cosmolike_notebook_utils/covariance/survey.py).

## 5. Shared responses and deterministic matrix work

The core's super-sample covariance has the form

```math
C^{\rm SSC}_{ij}=\sum_s w_s\,R_i(s)R_j(s),
\qquad w_s=\Delta\chi_s\,\sigma_b^2(\chi_s),
```

where $`R_i`$ is the response of measured row $`i`$ to a background density perturbation in shell $`s`$. Its mask variance uses the supplied raw mask spectrum and validates $`C_0^W=\Omega_W^2/(4\pi)`$. The radial-local long-mode approximation produces positive weights, so this contraction is a weighted Gram matrix. It keeps all cross-bin entries and does not need a separate integral for every pair. A joint calculation can stack second- and third-order response rows and obtain the three covariance blocks from the same contraction. The missing input is the third-order response itself: the existing power response does not determine the bispectrum response, its tidal dependence, or its intrinsic-alignment terms. [Mask variance](../../../../../cosmolike/covariances/ssc_cov.c); [SSC assembly](../../../../../cosmolike_notebook_utils/covariance/survey.py).

`gaussian_project_cov` is already a general weighted contraction. Its four-by-four output tiles reuse input loads; each SIMD lane accumulates a different covariance entry in increasing node order. The outer matrix assemblers enumerate triangular tasks and let one worker own a block and its transpose. Helpers test whether an OpenMP team is active, avoiding a nested team for each block. This pattern applies to both $`C_{33}`$ and $`C_{23}`$: the latter is rectangular and requires its full set of entries, while its transpose supplies $`C_{32}`$. Thread count does not split the summation of an individual entry. Cross-platform last bits can still differ because available fused instructions differ. [Projection](../../../../../cosmolike/covariances/gaussian_cov.c); [task ownership](../../../../../cosmolike/covariances/assembly_cov.c).

For the studied aperture layouts, an 80-entry Y3 vector has 3,240 distinct $`C_{33}`$ entries and a 100-entry Y6 vector has 5,050. Their dense double-precision matrices occupy only about 51 and 80 kB. The large storage belongs to intermediate functions, not these final matrices. Shared profiles and bounded geometry blocks should therefore receive more design attention than compressing a 100-by-100 result.

## 6. Non-Limber FFTs and angular windows offer execution patterns

The covariance FFTLog path prepares each radial field's forward transform once, saves its coefficients, and processes multipoles in blocks of 16. One serial setup creates FFTW plans; outer OpenMP tasks execute transforms on private real and complex scratch rows. Mellin kernels use two parity seeds and recurrence within each block, and complex phases advance by multiplication with a periodic reset. The workspace checks that a later thread count does not exceed the scratch allocation. These are source examples for transform reuse, bounded memory, and worker ownership. They do not require FFTW's internal thread pool or CUDA. [Workspace and plans](../../../../../cosmolike/covariances/fftlog_cov.c); [blocked execution](../../../../../cosmolike/covariances/fftlog_cov.c).

The physical non-Limber construction uses a separable linear field and subtracts its matched Limber contribution before adding the correction. It is not a non-Limber six-point covariance prescription. Likewise, the real-space two-point operators average spherical spin kernels over angular bins using precomputed Jacobi recurrences. Their panel count is set by the highest angular oscillation, then the quadrature order is refined independently. Those methods support prepared aperture filters and integration rules, but the spherical spin kernels and the pair-noise correction cannot be inserted unchanged into a third-order estimator. A mask pair area describes two positions; three-point noise and finite-footprint covariance require the relevant higher-order survey geometry. [Non-Limber execution](../../../../../cosmolike/covariances/nonlimber_cov.c); [angular panels](../../../../../cosmolike/covariances/operators_cov.c); [pair mask](../../../../../cosmolike/covariances/mask_cov.c).

## 7. An execution plan that matches the source and the time allowance

Serial entry should establish one cosmology, the power provider, catalog definitions, and numerical controls. It must finish lazy power, redshift, FAST-PT, halo, and NFW table construction before workers read those tables. The core already performs this warm-up in `fill_radial_cov` and `halo_moments_cov`. Shared immutable arrays can then feed parallel profile, transform, or covariance-block phases. Mass and angular sums remain owned by one worker, with SIMD across distinct outputs where possible. Calling global setters from concurrent Fisher tasks would violate this ownership model; independent cosmologies need separate processes or an implemented context design. [Radial warm-up](../../../../../cosmolike/covariances/spectra_cov.c); [halo warm-up](../../../../../cosmolike/covariances/halo_cov.c).

The CPU port should retain the production array boundary. Its NumPy binding borrows contiguous input rows, whereas the Armadillo notebook wrapper copies matrices into row-contiguous C storage. Both reach the same projection kernel, but their allocation and copying costs differ. A calculation with large shared geometry should not pass that geometry repeatedly through the copying wrapper. Internally retained geometry must be invalidated when bins, integration nodes, or filter definitions change; model-dependent profiles require cosmology and halo-model invalidation as well. [Production borrowing](../../../../../cosmolike/covariances/components_interface_cov.cpp); [notebook copying](../../../../../cosmolike/covariances/components_wrapper_cov.cpp).

The actual quadrature controls are part of the cost specification. The baseline resolver chooses 96 nodes per radial, ordinary mass, and angular panel, and 20 tree-angle panels. Its 128-point non-Gaussian multipole grid reaches the configured real-space cutoff; Fourier calculations retain only the nodes needed to bracket their measured bands. The project has seven radial panels, giving 672 nodes at that level. These numbers explain the existing loop sizes but do not prescribe the Y3/Y6 settings or identify the settings behind the user's 30 seconds. Increasing interpolation density, increasing quadrature order, and adding a physical covariance term are three different changes and should be recorded separately. [Accuracy resolver](../../../../../cosmolike_notebook_utils/covariance/accuracy.py); project radial panels (inspected host file `covariance/<survey>_covariance.py`, line 555).

For a fixed-covariance Fisher analysis, the work budget is approximately

```math
T_{\rm study}=T_{\rm preparation}+T_{C_{22},C_{23},C_{33}}
+N_{\rm mean}\,T_{\rm mean}+T_{\rm checks}.
```

Here $`N_{\rm mean}`$ counts all fiducial and displaced mean evaluations, including derivative-step checks. A 50-minute covariance is paid once per selected fiducial covariance model, whereas mean cost is multiplied by $`N_{\rm mean}`$. This favors retaining all requested covariance contributions and spending effort on reuse before considering physical truncations. On a quiet machine, a future measurement should record preparation, each covariance term, assembly, memory, and convergence separately. The existing driver already returns stage durations, but those exclude CAMB/initialization, diagnostic eigensystems, and file output. The full study budget must include those stages. Source inspection supports this CPU-first program; only that later measurement can determine whether it meets the stated time allowance.
