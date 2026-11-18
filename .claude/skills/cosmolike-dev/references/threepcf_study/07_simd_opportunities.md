# Where SIMD belongs in the three-point calculation

> **Archive status.** This source study records design proposals and validation requirements, not completed features. Read the [maintained implementation status](../threepcf.md) first. Both the mean and its covariance belong in `cosmolike/3pcf/`; earlier destination proposals using `fastnc/` or `covariances/` are superseded.

A C implementation has several concrete opportunities for SIMD: triangle geometry, TATT kernel arithmetic, interpolation weights, radial projection, angular-mode coupling, Fourier coefficient products, and aperture contractions. The strongest candidates combine several operations in one pass over contiguous data, keeping intermediate values in processor registers instead of allocating another array for every expression. `cosmo2D.c` supplies examples of that design. The proposed work should first remove repeated calculations, then distribute independent tasks with OpenMP, and then apply SIMD within those tasks.

This chapter is a source-only design study. It uses fastnc's `tatt` revision `7b2336ce37a909bed51f07f32a2a4f096e71b6ce` and CosmoLike core revision `0c0cf3838852396933c7f49550e35917dad0537c`. No code was implemented and no timing exercise was run. Statements about a possible limiting resource identify what a later profile should examine; they are not measured bottlenecks on the user's laptop.

The integral reductions in [Chapter 19](19_further_covariance_reductions.md) and [Chapter 20](20_further_data_vector_reductions.md) change the production loops. They add products of filtered halo transforms, separable radial mean functions, and short combinations of precomputed angular moments. SIMD can evaluate several filter or tomographic outputs sharing those inputs. [Chapter 21](21_further_execution_and_cache_design.md) also moves fixed FFT work into aperture-weight preparation and batches remaining covariance outputs; vectorizing a loop that those reductions remove would be a lower priority.

## 1. Choose what one lane represents

SIMD means that one instruction applies an operation to several numerical values. A **lane** is one position in that group. Four lanes can hold four triangles, four aperture radii, or four terms of a sum. Those choices lead to different memory access and numerical behavior. OpenMP is a separate layer: ten workers can each use SIMD while calculating different output blocks.

Consider the tree-level kernel in [`ia_bispectra.py:142`](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L142):

```math
F_2(k_1,k_2,k_3)=\frac57+
\frac12\mu_{12}\left(\frac{k_1}{k_2}+\frac{k_2}{k_1}\right)
+\frac27\mu_{12}^2,
\qquad
\mu_{12}=\frac{k_3^2-k_1^2-k_2^2}{2k_1k_2}.
```

The three positive side lengths describe a closed Fourier triangle; $`\mu_{12}`$ is the cosine between its first two wavevectors. For four independent triangles, the vector of first side lengths is $`[k_{1,0},k_{1,1},k_{1,2},k_{1,3}]`$, with corresponding vectors for the other sides. Every arithmetic operation applies to matching positions. Lane 2 computes triangle 2's cosine and kernel; it never combines triangle 2 with triangle 0. The output is four separate kernel values, so this operation needs no addition across lanes and no change in an integral's summation order.

This example also exposes an earlier saving. Under the common Limber conversion $`k_a=\ell_a/f_K(\chi)`$, the ratios in $`F_2`$ and the triangle's normalized directions do not depend on radial distance. Prepare them once per angular triangle, then reuse them for every radial node. Vectorizing a geometry calculation that is unnecessarily repeated at every node leaves the larger saving unused. Any multipole shift or geometric convention must remain identical to the chosen reference.

## 2. A source map of candidate loops

The table distinguishes the mathematical work from the machine layout. “Contiguous triangles” means that neighboring memory values hold the same quantity for neighboring triangles. “Independent outputs” means that a lane retains its own running sum.

| Candidate and upstream source | CosmoLike template | Suggested lane and layout | Numerical issue to preserve | Resource to inspect later |
|---|---|---|---|---|
| Triangle vectors and dot products, `ia_bispectra.py:146` | FFTLog normalization, `cosmo2D.c:9680` | Four triangles; separate arrays for side lengths and direction components | Degenerate triangles, square-root domain, orientation signs | Division/square-root throughput and avoided temporary arrays |
| TATT `F_00`, `F_20`, `F_21`, `F_22`, `ia_bispectra.py:167` | Scalar physical cores and prepared inputs, `cosmo2D.c:2898` | Four triangles for one named physical term | Signs, coefficients, field permutations, cancellation | Arithmetic throughput and register capacity |
| Power and multipole interpolation, `halofit.py:104`, `bispectrum.py:1053` | `limber_fill_interp`, `cosmo2D.c:745` | Four queries after preparing brackets and weights | Grid spacing, spline order, signed values, extrapolation | Scattered loads and scalar transcendental calls |
| Seven radial projections, `bispectrum.py:762` | Batched shear projection, `cosmo2D.c:3081` | Four output triangles or triplets; fixed node order | Quadrature rule and physical field placement | Bispectrum reuse and memory traffic |
| Angular decomposition, `multipole.py:93` | Grouped Legendre sums, `cosmo2D.c:974` | Four triangle outputs or expansion coefficients | Integrated interpolation basis and normalization | Repeated reads and contraction throughput |
| Coupling sum `HM`, `fastnc.py:231` | Grouped output sums, `cosmo2D.c:974` | Four grid locations or modes, with expansion index summed in order | Diagonal correction and mode signs | Temporary arrays and bandwidth |
| Complex FFT coefficient products, `twobessel.py:164` | Per-worker FFT preparation, `cosmo2D.c:9650` | Several frequency entries; explicit real/imaginary layout | Conjugation, negative orders, Hermitian layout | Packing, shuffles, complex multiplication |
| Aperture projection, `fastnc.py:559` | Covariance output tiles, `gaussian_cov.c:237` | Several aperture outputs; common correlation input | Projection convention, integration measure, permutations | Matrix reads and contraction throughput |
| Bin-averaged Gamma kernels, `twobessel.py:186` | Gamma recurrences, `cosmo2D.c:9172` | Frequency entries when special-function evaluation permits it | Pole treatment, asymptotic branch, small bin-width limit | Setup cost versus reuse count |
| Bihalofit and high-wavenumber branches, `halofit.py:482` | Prepared inputs and model choice outside inner loops, `cosmo2D.c:3144` | Homogeneous groups of valid triangle queries | Physical-domain masks and extrapolation formula | Powers, divisions, branch packing |

The principal local templates are [interpolation](../../../../../cosmolike/cosmo2D.c), [grouped Legendre sums](../../../../../cosmolike/cosmo2D.c), [TATT projection](../../../../../cosmolike/cosmo2D.c), [FFTLog execution and normalization](../../../../../cosmolike/cosmo2D.c), and [covariance projection tiles](../../../../../cosmolike/covariances/gaussian_cov.c). Their ownership and loop structures transfer; their physical kernels do not replace the corresponding three-point equations.

## 3. TATT algebra: share the triangle, then evaluate its physical terms

The upstream [TATT primitives](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L167) repeatedly construct normalized vectors, their dot products, squared components, and products of the side power spectra. `F_20`, for example, combines the tree kernel with products of the form $`\widehat{k}_{1x}^2-\widehat{k}_{1y}^2`$ and $`(\widehat{\boldsymbol{k}}_1\cdot\widehat{\boldsymbol{k}}_2)\widehat{k}_{1x}\widehat{k}_{2x}`$. Once the geometric factors are available, these are additions and multiplications on independent triangles. They are direct candidates for explicit SIMD.

A suitable array layout stores one field across a block of triangles: all $`\widehat{k}_{1x}`$ values together, all $`\widehat{k}_{1y}`$ values together, and separate rows for the powers $`P_1`$, $`P_2`$, and $`P_3`$. This is often called a structure-of-arrays layout. Four adjacent values can then be loaded together. Storing an entire record for triangle 0, followed by the record for triangle 1, would separate successive values of the same field by the record length and require extra loading or rearrangement.

The proposed loop should keep the meaning of each intermediate visible: normalized directions, common products, one physical kernel, then its output. Reusing a common product can avoid both arithmetic and an array write. Combining every TATT expression into one huge vector function would create many simultaneously live values and can force register contents into memory. A small block of related outputs, with explicitly named intermediates, is the better starting design. The measured best block size remains to be established.

The seven-versus-nineteen component issue must be settled before this layout becomes an interface. [`get_ia_bispectra`](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L597) returns nineteen named components in the inspected source, including parity components with a B-mode field. The [radial projection](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/bispectrum.py#L762) describes and unpacks seven E-mode components. This is a scientific and interface question, not a reason to discard twelve arrays during optimization. A future implementation should carry a named component list and evaluate the components required by its agreed observable.

## 4. Interpolation: prepare the lookup before vectorizing it

CosmoLike's interpolation example computes four table positions and interpolated values together. That approach is useful when queries share table structure and many physical rows use the same positions. In [fastnc's multipole interpolation](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/bispectrum.py#L1053), the requested angular grid is reused across expansion orders. Brackets and interpolation weights can be prepared once for that grid. Each expansion coefficient then supplies only its table values.

A uniform grid permits a direct arithmetic index. A nonuniform grid needs a search or a prepared mapping. Replacing a spline with linear interpolation changes the approximation and needs a separate convergence assessment. The port should first retain the reference interpolation order, coordinate transformation, boundary correction, and extrapolation. Signed TATT components must retain a signed interpolation representation; taking their logarithm is not a harmless speed optimization.

A **gather** loads vector lanes from different memory addresses. It can express four interpolation queries, but it does not make the four table values adjacent in memory. CosmoLike's [rejected four-wavenumber gather experiment](../covariance_optimization_review.md) passed its accuracy comparisons yet did not reduce runtime on the tested machines. The source study should therefore prioritize shared brackets, fewer lookups, and contiguous table layout before recommending gathers.

Boundary branches need explicit care. [High-wavenumber power extrapolation](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/halofit.py#L104) uses different expressions inside and outside the tabulated range. Evaluating every expression on every lane and masking the answer afterward can still evaluate an invalid logarithm, square root, or division in an inactive lane. Under strict floating-point rules, the mask does not erase that operation. Separate homogeneous query groups or substitute valid operands before evaluating a masked branch, then restore the prescribed result. Preserve the distinction between clipping tiny roundoff at a physical boundary and accepting an invalid triangle.

## 5. Four radial outputs without a horizontal reduction

The [upstream radial calculation](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/bispectrum.py#L831) forms separate weighted integrand arrays and applies the trapezoidal rule. A four-lane alternative assigns one output triangle to each lane and walks radial panels in increasing order. For triangle $`t`$, one panel contributes

```math
\Delta b_t^{(p)}=
\frac{\chi_{p+1}-\chi_p}{2}
\left[I_t(\chi_p)+I_t(\chi_{p+1})\right],
\qquad
\boldsymbol{b}\leftarrow\boldsymbol{b}+
[\Delta b_0^{(p)},\Delta b_1^{(p)},\Delta b_2^{(p)},\Delta b_3^{(p)}].
```

$`I_t`$ is the full integrand, including the appropriate radial weights and bispectrum component. Each lane accumulates its own panels in the same sequence. There is no addition across lanes. Four source triplets can be grouped instead when they share a bispectrum value and differ only in radial weights. The data layout should follow whichever grouping provides the required reuse with contiguous reads.

This preserves the chosen scalar panel order, but does not promise bitwise agreement with NumPy's reduction implementation. Algebraically combining panel weights, introducing fused multiply-add, or changing how several physical components are summed can change rounding. Establish the scalar C reference order explicitly and compare it with upstream at a declared tolerance. SIMD across independent outputs can then preserve that chosen order. SIMD across four radial nodes is another candidate, but its final horizontal reduction combines lanes and changes the order of summation.

## 6. Angular coupling and FFT arithmetic

The [coupling sum](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/fastnc.py#L231) has the form

```math
H_M(q)=\sum_L(-1)^L G_{LM}(q)b_L(q).
```

$`q`$ labels a two-dimensional angular grid location, $`L`$ labels a bispectrum expansion coefficient, and $`M`$ labels the output angular mode. Four neighboring $`q`$ values can occupy four lanes while $`L`$ is summed in a fixed order. A small group of $`M`$ outputs can share each loaded $`b_L(q)`$. This follows CosmoLike's grouped Legendre contractions and avoids the full temporary product array in the NumPy expression. The separately computed diagonal correction must remain in the output with its original normalization and index range.

The [two-Bessel transform](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/twobessel.py#L164) multiplies saved Fourier coefficients by a kernel for each axis and then conjugates the result. Several frequency entries can be processed together. For two complex values,

```math
(a_r+i a_i)(b_r+i b_i)
=(a_rb_r-a_ib_i)+i(a_rb_i+a_ib_r).
```

Separate real and imaginary vectors make this arithmetic visible. FFTW commonly receives interleaved complex storage, with real and imaginary parts adjacent. Converting the entire array to another layout on every transform can cost more than the arithmetic saved. Candidate designs should compare a fused loop on interleaved storage with a prepared split layout; neither should allocate and repack repeatedly without accounting for that traffic. Conjugation changes the imaginary sign, and negative Bessel orders require the existing parity factors.

The FFT itself already runs in a compiled numerical library. Rewriting surrounding Python does not create a further fourfold gain inside that library. The relevant SIMD opportunities are the biasing, coefficient products, layout preparation, normalization, and contractions around the FFT, together with better reuse of its inputs. There is also a reference detail to resolve: `get_c_mn` constructs `c_mn_filter` but returns `c_mn` in the inspected source. A port must not silently activate that taper while claiming a purely arithmetic rewrite.

## 7. Aperture contractions and setup kernels

The [aperture calculation](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/fastnc.py#L559) forms geometric kernels and contracts them with natural components. For fixed bins and aperture radii, much of the kernel preparation can be performed once. At each subsequent cosmology, a group of aperture outputs can share the loaded correlation values. CosmoLike's covariance projection provides the model: one worker owns an output tile; each lane is an independent aperture sum; the angular sample order is fixed.

This design must retain the integration measure, the minus signs and component interchange in `calculateMap3`, and the extra permutations for unequal aperture radii. The existing matrix products already call compiled numerical routines. The proposed gain is from reusing fixed kernels, reducing temporary storage, and choosing a contraction layout suited to the actual number of outputs. Replacing a well-sized matrix multiplication with hand-written SIMD is not automatically preferable.

Gamma ratios and bin-averaging factors in [`twobessel.py:186`](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/twobessel.py#L186) are lower-priority SIMD targets when cached setup removes them from repeated cosmology evaluations. Recurrences across transform orders and reuse across identical grids should be studied first. Any vector special-function replacement must preserve the high-frequency asymptotic branch, pole treatment, and the small-width limit of the bin average. A vectorized polynomial loop and a vectorized complex Gamma evaluation have different validation burdens.

## 8. Portability and acceptance

SIMDe expresses vector operations with a common interface. A logical four-double vector matches a 256-bit AVX2 register on suitable x86 processors; on Apple silicon it can require multiple 128-bit NEON operations. The same source notation therefore does not imply the same instruction count or a fourfold speedup. NumPy's elementwise loops, SciPy functions, BLAS, and FFT libraries are already compiled, and some already use SIMD. The C opportunity is to improve the organization of the remaining arithmetic and its data movement.

Retain strict floating-point settings and avoid fast-math flags. Keep scalar tails for blocks whose sizes are not multiples of the lane width, and verify all loads remain within valid rows. Tests must include signed and nearly cancelling components, squeezed and near-degenerate triangles, grid boundaries, extrapolation transitions, zero alignment amplitudes, both transform-axis sizes, and unequal aperture radii. Compare the full requested vector, including entries later removed by a science mask.

Fisher derivatives need their own comparison. If the finite-difference step is $`h`$, uncorrelated evaluation errors can enter a derivative at a scale proportional to $`1/h`$. Agreement of a single fiducial vector is insufficient. Compare derivative columns at more than one step and assess their effect on the covariance-weighted Fisher matrix and marginalized errors. Repeated evaluations at fixed settings and different worker counts should also expose races or uninitialized values.

The first SIMD candidates should be fused geometry and TATT algebra, fixed-order radial and coupling contractions across independent outputs, FFT coefficient products, and output normalization. Interpolation gathers and special-function replacements should follow only when their remaining cost warrants them. These are specific implementation targets for the later coding task; their accuracy and runtime must be established on a quiet machine after the reference physics and interfaces are fixed.
