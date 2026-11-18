# What the CosmoLike implementation teaches us

> **Archive status.** This source study records design proposals and validation requirements, not completed features. Read the [maintained implementation status](../threepcf.md) first. Both the mean and its covariance belong in `cosmolike/3pcf/`; earlier destination proposals using `fastnc/` or `covariances/` are superseded.

CosmoLike supplies a useful numerical design for a shear three-point module: prepare quantities at the level where their inputs change, give each worker an independent output, and keep the expensive transforms separate from the algebra that combines them. The source provides concrete examples of each choice. It does **not** establish a runtime for a proposed fastnc port. The work described here is source inspection; no implementation or timing experiment was performed.

The inspected core revision is `0c0cf3838852396933c7f49550e35917dad0537c`. The working tree contains unrelated edits, including scale-cut and Python files. A path-restricted comparison against that revision found no changes in `cosmolike/cosmo2D.c`, `cosmolike/pt_cfastpt.c`, `cfastpt/`, or `cosmolike/covariances/`, which supply the examples below. References point to the local files at the inspected lines. Some older development notes refer to a `cfftlog/` directory; the active projection functions examined here are in `cosmo2D.c` itself.

## 1. The first saving comes from asking what depends on what

An angular two-point spectrum combines a three-dimensional spectrum with radial selection functions. Schematically, its Limber projection is

```math
C_\ell^{ij}=\int d\chi\,
\frac{W_i(\chi)W_j(\chi)}{f_K(\chi)^2}
P\!\left(\frac{\ell+1/2}{f_K(\chi)},z(\chi)\right).
```

Here $`\chi`$ is comoving radial distance, $`f_K`$ is transverse comoving distance, $`W_i`$ is the projection weight for source bin $`i`$, and $`P`$ is the three-dimensional power spectrum. The integration is evaluated at a finite set of radial nodes. At one cosmology, distance and growth at a node are identical for every multipole and source-bin pair. Calling those functions inside every spectrum integral repeats the same calculation.

The [`cosmo_nodes` structure and its construction](../../../../../cosmolike/cosmo2D.c) place scale factor, quadrature weight, distance, growth, Hubble rate, and the radial Jacobian into arrays. [`C_ss_tomo_limber_work`](../../../../../cosmolike/cosmo2D.c) then separates three tasks. It prepares radial weights once per bin and node; it prepares power and alignment templates once per multipole and node; and it combines those tables for all source-bin pairs. The innermost integration loop performs arithmetic on prepared arrays. It does not repeatedly discover distances, build a power table, or choose an alignment model.

This arrangement transfers directly to a Limber bispectrum projection. For the lensing-only part,

```math
B_\kappa^{ijk}(\ell_1,\ell_2,\ell_3)
=\int d\chi\,
\frac{W_i(\chi)W_j(\chi)W_k(\chi)}{f_K(\chi)^4}
B_\delta\!\left(\frac{\ell_1}{f_K},
\frac{\ell_2}{f_K},\frac{\ell_3}{f_K};z(\chi)\right).
```

The angular wavevectors close to form a triangle. The notation above uses the flat-sky projection and does not insert a two-point curved-sky correction into the three-point expression. At fixed triangle and radial node, the matter bispectrum is common to all source-bin triplets. At fixed node and source bin, the lensing and intrinsic-shape weights are common to all triangles. A C design should expose these two kinds of reuse separately. Intrinsic-alignment terms require the appropriate matter–shape bispectra and the correct placement of each bin on each field; replacing every weight by one effective weight is valid only for a model whose algebra actually permits that factorization.

Suppose the calculation has $`N_t`$ triangles, $`N_z`$ radial nodes, and $`N_b`$ source bins. Preparing one radial quantity independently for every ordered triplet would require up to $`N_tN_zN_b^3`$ evaluations of that quantity. A shared node table requires $`N_z`$ evaluations, and a per-bin table requires $`N_zN_b`$. This is a count of avoidable calls, not a prediction for total runtime: the bispectrum evaluations, angular transforms, and final contractions still have their own costs. Upstream fastnc already uses arrays and precomputed couplings, so the port must identify remaining duplication rather than assume that Python repeats every scalar operation.

The useful source detail is the batched power read at [`cosmo2D.c:3155`](../../../../../cosmolike/cosmo2D.c). For one radial node, the code prepares all required wavenumbers and calls `Pdelta_at_a` once. The interpolation work associated with redshift is shared across that row. A three-point implementation can apply the same idea to the three triangle sides, but it must also account for repeated side lengths and preserve signed cross-field quantities. A logarithmic interpolation rule suitable for a positive matter spectrum is not automatically suitable for a signed alignment bispectrum.

## 2. Non-Limber projection shows how to share transforms

The non-Limber functions illustrate a broader principle: the input transform and the kernel applied to it often depend on different variables. [`cfftlog_ells_p1`](../../../../../cosmolike/cosmo2D.c) prepares a radial function on a logarithmic distance grid, applies a power-law bias, inserts guard zeros, and computes its forward Fourier transform. That forward transform does not depend on the angular multipole $`\ell`$. The code stores it once per active radial component.

[`cfftlog_ells_p2`](../../../../../cosmolike/cosmo2D.c) applies the multipole-dependent spherical-Bessel kernel and performs the inverse transform. Its work is divided into blocks of at most 16 multipoles. A block shares the forward coefficients and kernel preparation, and each independent output uses its own working arrays. The phase factor also has regular structure: its argument changes by a constant increment along the frequency grid. The implementation advances a complex phase by multiplication and recomputes it from trigonometric functions every 1024 steps to limit accumulated rounding error. Such a recurrence needs its own accuracy test when transferred to another frequency grid.

The Gamma-function ratios in the Bessel kernel contain a second reuse opportunity. The code evaluates two starting multipoles directly, then advances the even and odd sequences using $`\Gamma(z+1)=z\Gamma(z)`$. This replaces many special-function evaluations by complex arithmetic. In a two-dimensional FFTLog calculation, a related recurrence may exist across transform orders. It must be derived for the actual cylindrical-Bessel kernel and bias convention used by fastnc. The spherical-Bessel normalization in CosmoLike is not an interchangeable formula.

The distinction matters physically. The two-point FKEM construction described near [`cosmo2D.c:8880`](../../../../../cosmolike/cosmo2D.c) combines a non-Limber projection of a separable reference power spectrum with a Limber residual. Its schematic form is

```math
C_\ell=C_\ell^{\mathrm{nonLimber,ref}}
+C_\ell^{\mathrm{Limber,full}}
-C_\ell^{\mathrm{Limber,ref}}.
```

This identity defines a particular approximation to a two-point projection. A shear bispectrum would require three radial legs, their angular couplings, and an unequal-time three-point model before an analogous split could be justified. Reusing the scheduling, planning, and memory ideas is justified by their numerical structure. Importing the FKEM formula as a non-Limber three-point prescription is not justified by this source study. Reproducing the paper should retain its projection assumptions first.

## 3. The low-rank path teaches memory management and its limits

The thin-bin density calculation provides a useful example of separating a numerical approximation from a physical approximation. [`C_gg_lowrank_decomp_work`](../../../../../cosmolike/cosmo2D.c) approximates a weighted grid of $`\sqrt{P_g(k,z)}`$ by a finite sum of products of a wavenumber function and a redshift function. The retained number of products is its rank. Increasing rank can reduce the numerical error of this representation. It does not remove the geometric-mean assumption used to construct the unequal-time spectrum from the equal-time spectra. The source explains both choices separately in its [physical description](../../../../../cosmolike/cosmo2D.c).

A bispectrum has additional shape dependence and can change sign when alignment contributions are included. A square-root decomposition of an auto power spectrum is therefore not a proposed representation for the full bispectrum. A later compression study could factor sampled bispectrum templates over redshift and triangle coordinates, but acceptance would depend on projected observables and their derivatives, not only on discarded singular values. Near a cancellation, a small average grid error can still cause a substantial relative error in a particular observable.

The immediately useful example is the [block storage in the low-rank projection](../../../../../cosmolike/cosmo2D.c). A naive array stores every radial row, component, multipole, and radial node. The active calculation only needs one 16-multipole block at a time. The implementation keeps a smaller pool of numerical rows and points the current block's indices into that pool. It also stores a common wavenumber grid once because that grid does not depend on the radial row. Storage decreases without dropping any multipole from the calculation.

For fastnc, the analogous question is which arrays must coexist. If a transformed angular mode can be contracted into every requested aperture statistic and then discarded, retaining all modes, all triplets, all natural components, and every angular grid point is unnecessary. If the user requests the full natural-component surface, those outputs must remain available or be written in blocks. The allocation design must follow the requested observable. A single complex double-precision array with $`N\times N`$ entries occupies approximately $`16N^2`$ bytes before metadata and padding. At $`N=1024`$, that is 16 MiB; 20 triplets, four components, and 20 modes would multiply this illustrative storage to 25 GiB. These dimensions are an example of why lifetimes matter, not a statement of the upstream default allocation.

The [specialized low-rank transform scheduler](../../../../../cosmolike/cosmo2D.c) also distinguishes work with unequal costs. It creates a list of active row/component pairs and distributes the pair–multipole tasks across workers. Each task owns a whole output row, so changing which worker runs it does not change its summation order. Dynamic scheduling is useful there because different components have different transform lengths. For a three-point calculation, static scheduling should remain a candidate when tasks have equal cost; choosing a dynamic scheduler solely because it exists in another function would add overhead without evidence of a benefit.

## 4. FFT threading means two different things

The inspected core uses OpenMP to run **different transforms concurrently**. It creates FFTW plans serially and executes them through the new-array interface on independent arrays. A repository-wide search found no calls to `fftw_init_threads` or `fftw_plan_with_nthreads`. This establishes what the source explicitly requests; it is not a measurement of every linked library. The design should not be described as an existing implementation of internally multithreaded FFTW transforms.

An FFTW plan is a prepared sequence of transform operations for specified dimensions and memory layout. It can be reused when the numerical input changes. The [FFTW thread-safety documentation](https://www.fftw.org/fftw3_doc/Thread-safety.html) permits concurrent execution of a shared plan on separate arrays, while plan creation and destruction must be serialized. CosmoLike uses `fftw_execute_dft_c2r(plan, input, output)` to supply a worker's arrays explicitly. Calling `fftw_execute(plan)` would instead use the arrays attached when the plan was created, causing several workers to overwrite the same storage.

Reuse also has a memory contract. Transform dimensions, strides, in-place versus out-of-place layout, and alignment must agree with the plan. The [FFTW new-array documentation](https://www.fftw.org/fftw3_doc/New_002darray-Execute-Functions.html) gives those conditions. A two-dimensional port must include both axes and their strides in this reasoning. A plan for a contiguous square grid cannot be applied to a differently padded array merely because the logical number of elements matches.

The proposed module should first compare two arrangements: one FFT thread per independently scheduled transform, and a single outer task using several FFT threads for a sufficiently large transform. A mixed arrangement belongs in the comparison only if resources are divided explicitly. For example, eight outer workers each starting an eight-thread FFT request up to 64 runnable threads before any other library participates. That is not eight-core parallelism. FFTW itself advises that the best count depends on transform size and synchronization cost in [its thread-count guidance](https://www.fftw.org/fftw3_doc/How-Many-Threads-to-Use_003f.html).

For Fisher work, concurrency across parameter perturbations introduces another boundary. [`C_cl_tomo_lowrank_core` documents process-global state](../../../../../cosmolike/cosmo2D.c): internal workers can read prepared tables concurrently, but simultaneous outer cosmology evaluations cannot independently change those same tables. Separate processes isolate that state. A threaded Python driver evaluating two cosmologies at once would require a broader ownership redesign, even if the proposed fastnc module had private arrays.

### A concrete arrangement for ten workers

There is enough independent work to justify investigating a C implementation with extensive OpenMP use. The restriction is not that only a few outer loops may be threaded. It is that each threaded task must read prepared shared inputs and write storage that no other task writes. The following dependency map identifies candidate tasks before any timing exercise. A boundary between stages means that the next stage cannot read a table until its producers have finished.

| Stage | Work shared before the stage starts | Candidate work for ten workers | Ownership and completion rule |
|---|---|---|---|
| Model setup | Current cosmology, source tables, numerical settings | Keep global setters, lazy initialization, and FFTW planning on the calling thread | Finish mutation before shared reads |
| Radial and triangle preparation | Initialized power and distance readers | Different radial nodes or disjoint triangle blocks | Each task fills its own table rows |
| Projected bispectrum preparation | Radial weights and three-dimensional bispectrum templates | Different output triplets, angular-expansion coefficients, or triangle blocks | Each output integral has one owner and a fixed radial summation order |
| Natural-component transforms | Projected bispectrum coefficients, coupling tables, transform plans | Different triplet–angular-mode pairs | Each task retains its common forward transform and owns its inverse scratch |
| Aperture contractions | Natural components or the required transformed block | Different aperture outputs, with mode partials kept separately if necessary | Final combination uses an explicitly specified order |
| Output assembly | Finished independent outputs | Usually a small copy/reordering step | Preserve the documented data-vector order |

For the transform stage, [`FastNaturalComponents.compute` at the pinned upstream revision](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/fastnc.py#L319) already constructs one two-Bessel object per nonnegative angular mode and uses it for several natural components. A C task should preserve this sharing. Giving each natural component a separate task without coordinating the common forward coefficients can duplicate work. Keeping the component loop inside one triplet–mode task also makes ownership easier to follow at the zero mode, where positive- and negative-mode output slots coincide. A further split needs an explicit treatment of that boundary and of component-transpose assignments.

With 20 selected triplets and 21 nonnegative modes, this candidate arrangement offers 420 tasks before grouping them for locality. Those counts are illustrative, not the repository's measured task count. Twenty is the number of unordered triplets with repetition for four source bins when the observable's permutation symmetry permits that representation. Ordered natural components require their own field and angular permutation rules. Even when only a subset of components is needed, the available task count can exceed ten without splitting a single transform across ten internal FFT threads.

The cost is worker scratch. If a design retained six square complex work arrays per worker, ten workers at $`N=1024`$ would need about 960 MiB for those arrays alone; at $`N=512`$ the corresponding estimate is 240 MiB. Six is an assumed workspace count for planning, not an inspected requirement. Real arrays, shared kernels, final outputs, alignment padding, and any saved forward coefficients add to that amount. This arithmetic should be done before selecting a mode block size or a hybrid process/thread layout. A design that exhausts physical memory can lose its arithmetic savings to memory traffic or paging.

This map supports a substantial optimization effort because it exposes repeated preparation, many independent transforms, and output contractions that can use SIMD. The thread allocation behind the approximately 40-second starting estimate has not been verified, so it cannot establish a one-core baseline or an existing ten-core result. The implementation target is to distribute independent work across the chosen cores while eliminating duplicated calls and avoiding oversized temporaries. Timing comparisons belong on a quiet machine after this dependency design and the scientific reference are agreed.

## 5. What CFASTPT already shares, and what it does not compute

[`get_FPT_IA`](../../../../../cosmolike/pt_cfastpt.c) computes ten one-loop TATT **power-spectrum** templates. Eight groups are assembled from 184 FFTLog terms in one `J_abJ1J2Jk` call; two additional terms use separate convolutions. The physical input is the linear matter power spectrum at scale factor one. The resulting loop templates have power-spectrum units and acquire the stated growth factors when projected. Their table entries are not the three-point TATT kernels needed for $`B_{\delta\delta I}`$, $`B_{\delta II}`$, or $`B_{III}`$.

The transferable operation is batching common mathematical terms. [`J_abJ1J2Jk` caches transform plans and Gamma kernels](../../../../../cfastpt/cfastpt.c). It computes the biased forward spectrum once per distinct bias exponent, then lets multiple angular-coupling terms read those coefficients. The [unique-exponent preparation](../../../../../cfastpt/cfastpt.c) makes the reason explicit: the exponent controls the forward input, whereas the angular index enters a later Gamma factor. Sharing an exponent permits sharing that transform even when the final terms differ.

The three-point analogue is a table of mathematical dependencies. Two TATT terms should share a triangle kernel only when their wavenumber, redshift, and spin dependencies coincide. Different nuisance amplitudes can multiply the same stored basis function, but a different physical kernel cannot be relabeled as an amplitude. This basis organization can also help Fisher derivatives: an amplitude derivative can sometimes be formed by changing only the coefficients of prepared templates. Redshift-evolution parameters, source-distribution changes, and cosmological changes generally require more preparation and need separate dependency rules.

CFASTPT also separates the grid used inside a convolution from the grid stored for later interpolation. [`pt_cfastpt.c:849`](../../../../../cosmolike/pt_cfastpt.c) chooses an internal resolution and then regrids the result if it differs from the output resolution. This allows independent tests of convolution accuracy and interpolation accuracy. Padding and extrapolation counts scale with the internal sampling to preserve their physical span in log wavenumber, as explained at [`pt_cfastpt.c:1260`](../../../../../cosmolike/pt_cfastpt.c). Increasing sample density while leaving a guard interval's sample count fixed would shrink the guard interval; that is a different numerical experiment.

Rounding a convolution length to a product of small primes can reduce FFT cost. For a linear convolution, zero-padding beyond the required convolution length preserves the required output samples. FFTLog also assigns physical coordinates, frequencies, bias factors, and reciprocal output points to its arrays. Changing an FFTLog length without updating those definitions can change the numerical transform. The port must test the full operation, not infer accuracy from the presence of extra zeros.

## 6. SIMD should have an explicit physical meaning per lane

SIMD applies one machine instruction to several values, called lanes. It differs from OpenMP: SIMD operates within one worker; OpenMP distributes tasks across workers. The source uses both. [`limber_fill_interp`](../../../../../cosmolike/cosmo2D.c) reads several table entries together, and the FFTLog output normalization multiplies four independent radial samples together. Neither operation needs to add values across lanes.

For reductions, the choice of lane matters. A vector containing four terms of one integral requires a final addition across lanes, which can change floating-point summation order. A vector containing four independent outputs can preserve each output's integration order. [`gaussian_project_cov`](../../../../../cosmolike/covariances/gaussian_cov.c) uses the second design: each lane accumulates a different covariance column, and a worker owns an entire output tile. The source groups four left and four right bins so that one loaded kernel contributes to several outputs.

A three-point contraction can test the same arrangement across source triplets, output apertures, or independent angular modes. The best choice depends on which data are contiguous and shared. Loading four scattered table values can cost more than four scalar loads, so the presence of vector instructions is not proof of a shorter runtime. The development skill records a rejected gather-based power-reader optimization for exactly this reason.

The `restrict` keyword is a promise about permitted memory overlap, not an acceleration switch. Its useful effect depends on accesses through the qualified pointer and on a correct ownership contract. Read-only inputs can sometimes coincide, but writable output and scratch must satisfy the promises made by the function. A port must not add `restrict` to arrays that are intentionally reused as both input and output.

The study should preserve the strict floating-point build required by the [development skill](../../SKILL.md). An OpenMP SIMD pragma does not establish that a particular compiler emitted vector arithmetic under those settings. Compiler reports and generated instructions are later evidence. Fused multiply-add also has one rounding instead of the two roundings of separate multiplication and addition; cross-platform agreement needs an error criterion even when every output keeps the same node order.

## 7. Covariance code supplies ownership patterns, not a three-point covariance

[`gaussian_wick_cov`](../../../../../cosmolike/covariances/gaussian_cov.c) computes the Gaussian covariance of two angular power spectra from two products of signal-plus-noise spectra. Its required expectation has four field factors. A covariance between two three-point estimators has six, and the covariance between a two-point and a three-point estimator has five. The existing four-field expression cannot supply those blocks by increasing an array dimension.

What can transfer is the division of responsibilities. Covariance primitives receive prepared spectra and caller-owned scratch. Their output blocks do not depend on hidden mutable cosmology state. They suppress nested worker teams when an outer parallel region already distributes blocks. [`fftlog_workspace_cov`](../../../../../cosmolike/covariances/fftlog_cov.c) owns its plans, saved forward transforms, and per-worker arrays explicitly. These examples are suitable starting points for the ownership of a `fastnc/` workspace.

A practical module would distinguish survey geometry, numerical transform setup, current cosmological templates, and current outputs. Geometry includes fixed angular bins and projection conventions. Numerical setup includes grid dimensions, transform orders, bias exponents, taper and padding. Cosmological templates include distances, power, bispectrum, and alignment coefficients. Outputs depend on all relevant inputs, including nuisance parameters and the requested ordering. Reallocating arrays and refilling their contents are different operations: changing cosmology usually needs new values without new storage; changing grid dimensions needs both.

The [core allocation rules](../../../../../cosmolike/basics.c) require particular care at this boundary. Each multidimensional allocation has contiguous values within a row, but rows are padded to 64-byte boundaries. A logical $`n_x\times n_y`$ array is not necessarily a flat block of exactly $`n_xn_y`$ doubles. Padding is uninitialized, and the ordinary zeroing helpers clear logical values only. Every element that an FFT actually reads, including its explicit guard region, must be initialized. A batched FFT plan must use the physical stride rather than infer it from a logical dimension.

## 8. Tests that distinguish successful reuse from accidental agreement

The implementation sequence should retain a scalar mathematical reference outside the production paths, then test one change at a time. The first comparison is the full natural-component or aperture vector at identical inputs and conventions. Numerical refinement asks a separate question: whether increasing radial resolution, angular expansion order, Fourier extent, transform sampling, and integration resolution changes that vector within the chosen error budget. Agreement with a reference at one setting does not answer it.

State tests must include an A–B–A sequence: evaluate cosmology A, change to B, and return to A. The returned A should agree with an independently prepared A. Repeat that sequence for source distributions, alignment coefficients, triangle grids, transform lengths, aperture choices, and worker counts. This exposes a cache that remembered a numerical shape but forgot the physical input that filled it. Test zero amplitudes and signs crossing zero because exact-zero special cases can hide derivative problems.

Thread tests should repeat the same input within one process and in separate processes, at one worker and several worker counts. Keeping the operation order fixed should permit repeatable results for that path. Where a justified transformation changes summation order, compare a declared per-entry error and the covariance-weighted effect, and still require repeatability at fixed settings. Race conditions are not an acceptable source of numerical scatter.

Only after these checks should a profile on a quiet machine establish the next optimization. No timing exercise is part of this source study or suitable for the user's busy laptop. The future comparison should measure first-call preparation, a changed cosmology, nuisance-only changes, and a repeated identical call separately. A cache hit does not measure the cost of a Fisher derivative. Report the CosmoLike three-point calculation separately from the external matter-spectrum provider and also report the end-to-end forecast cost. This preserves both the local optimization question and the scientist's actual waiting time.

The strongest transfer from this source study is a sequence of decisions: factor the physics into reusable tables, share transforms only across identical inputs, store only live blocks, assign independent outputs to workers, and then test SIMD on the remaining arithmetic. The source gives working examples of that sequence. The gain for a DES-Y6 three-point calculation remains a quantity to measure after a scientifically matched reference exists.
