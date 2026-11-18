# Work plan and evidence required before a forecast

> **Archive status.** This source study records design proposals and validation requirements, not completed features. Read the [maintained implementation status](../threepcf.md) first. Both the mean and its covariance belong in `cosmolike/3pcf/`; earlier destination proposals using `fastnc/` or `covariances/` are superseded.

The recommended project has two scientific outputs: a checked implementation of the two requested papers, and a common engine for DES Y3 and Y6 Fisher forecasts. Its numerical deliverables are the three-point mean and the joint analytic covariance, both using CPU C and OpenMP. The mean implementation should begin from the TATT physics, incorporate applicable corrections from current `fastnc/main`, and use the Y6 repository's separation between analysis configuration and prediction backend. The covariance should extend the existing core with the estimator-matched expressions mapped in [Chapter 15](15_analytic_covariance_cpu_port.md). A direct translation alone would preserve repeated work and unresolved conventions.

This chapter specifies future work. No implementation, forecast, scientific test, or timing exercise was run for this study. The laptop is busy, so performance work should begin with the source dependency map and proceed to timings only on a quiet machine.

## 1. Fix the reference calculation before distributing its work

The first deliverable should be a small written specification agreed with Sunao and Rafael. It identifies the intended equations, exact source revisions, estimator conventions, and numerical arrays behind the paper figures. The revisions inspected here are recorded in the [source list](11_sources_and_credits.md); they must not be assumed to be the revisions used for either publication. The newer Y6 repository is also an analysis scaffold with provisional inputs, as described in [Chapter 10](10_desy6_workflow.md).

The specification should answer these concrete questions:

1. Which configurations and saved arrays define the NLA reproduction?
2. Which configurations define the TATT and velocity-shear figures?
3. How are the nonlinear-alignment baseline and IA residuals combined?
4. Which nonlinear power-spectrum redshift prescription is intended?
5. What velocity-shear coefficient does the external parameter represent?
6. Which angular phase, side order, and bin-averaging rules define the observable?
7. Which applicable `main` corrections belong in the checked reference?
8. Which Y3 and Y6 data, priors, masks, and covariance versions are approved inputs?

These are questions for the implementation specification, not requests that the user answer before receiving this study. The source inspection already supports the design work below. No message has been sent to either author.

Keep a record of an unmodified historical prediction and a separately identified corrected reference whenever a numerical correction changes the result. For example, returning the FFTLog taper changes the transform coefficients. Its accuracy must be established against an independent integral or analytic function; preserving the untapered output merely because it is old would not establish correctness. Equally, silently changing the reference and reporting a speedup at a different accuracy would make the comparison uninterpretable.

## 2. The implementation sequence

| Stage | Deliverable | Evidence needed to proceed |
|---|---|---|
| Reference specification | Publication settings, source pins, intended equations, data ordering | Every requested paper result has a named comparison target; unresolved conventions are explicit |
| Branch reconciliation | A list of applicable corrections and new algorithms | Each proposed import is tied to an actual formula or behavior and a focused check |
| Physical input adapter | Distances, spectra, growth, source distributions, IA coefficients | Units and values agree at selected nodes before projection |
| Matter and IA kernels | Scalar C predictions for ordered triangles | Independent equations, symmetry tests, zero-amplitude limits, and component comparisons agree |
| Radial projection | Batched projected bispectra | Direct quadrature and refinement agree for unequal source bins and multiple parameter points |
| Multipoles and transforms | Natural components with retained intermediate arrays | Analytic-transform checks, direct-angle comparisons, phase checks, and mode convergence agree |
| Aperture prediction | The measured finite-bin statistic | Same filter, side ordering, cuts, and orientation multiplicities as the reference estimator |
| Finite-estimator dual weights | Transpose of the fixed linear transform applied to aperture weights | Forward/transpose dot-product identity, complete-vector agreement, and derivative checks; nonlinear interpolation and extrapolation remain outside the operator |
| Full-aperture analytic references | Separable BiHalofit three-halo terms and analytic TATT opening-angle integrals | Primitive-by-primitive Fourier normalization, unequal-filter checks, and convergence toward the full-filter limit |
| Persistent arrays and OpenMP | Reused geometry, plans, spectra, and owned output blocks | Reset, invalidation, repeated-call, and thread-count tests agree |
| SIMD kernels | Selected contiguous arithmetic loops | Scalar reference, vector tails, exceptional inputs, full vectors, and derivatives agree |
| Covariance term specification | Gaussian, bispectrum-product, power–trispectrum, connected five-/six-field, and long-mode contributions | Each term names its estimator, tomography, noise contractions, mask approximation, and IA scope; omitted terms are explicit |
| Covariance reference and reductions | Fixed-side bispectrum products, filtered halo products, connected correlations, and prepared long-mode responses | Positional partitions, independent projection variables, Fourier normalization, finite windows, and cutoff limits agree term by term |
| CPU covariance execution | Shared physical tables, bounded scratch, OpenMP tasks, and SIMD contractions | Refinement, thread-count checks, matrix symmetry, eigenvalues, and covariance-weighted numerical errors pass before timing |
| Joint analytic baseline | Matched two-point, three-point, and cross-covariance blocks | Positive definiteness and agreement with independent survey mocks are checked; long-mode terms are not counted twice |
| Simulation correction | First- and second-order precision matrix expansion for the full joint vector | Convergence, finite-simulation bias, sampling assumptions, held-out validation, and variation between mock subsets are assessed |
| Fisher adapter | Y3 and Y6 parameter derivatives and priors using the full selected two-point vector | Step-size plateaus, numerical refinement, parameter rank, and covariance treatment are checked; compression is a separate comparison |
| Survey forecasts | Two-point, three-point, and joint constraints for each survey | Survey-specific covariance, physical-model variations, cuts, and absolute errors are reported |

Several stages can be prepared concurrently once their contracts are fixed. One researcher can derive the IA reference while another specifies the estimator and data ordering. The numerical implementation should still introduce one change at a time: sharing triangle geometry, changing the quadrature, adding OpenMP, and vectorizing arithmetic are distinct changes with distinct tests. This makes a discrepancy traceable to a physical or numerical operation.

The initial port should expose a scalar triangle evaluator and a batched array evaluator. The first provides a readable connection to the equations; the second is the production entry point. Scalar references used to compare optimized kernels can remain in external validation rather than becoming a second selectable production library. The public Python call should move arrays across the interface once per stage or evaluation, rather than once per triangle.

## 3. Prioritize savings by the dependency map

Start by eliminating work that is repeated without changing its inputs. The strongest candidates are triangle geometry reconstructed for several IA components, spectra reconstructed for several source combinations, aperture matrices rebuilt for unchanged bins in the historical integrator, transform factors rebuilt for unchanged grids, and shared physical inputs refilled for calibration-only parameter changes. The Y6 package already retains aperture matrices; that reuse is part of its baseline, not an additional C saving. These opportunities are visible without executing the calculation. Their exact scope is documented in the [source audit](02_upstream_source_audit.md) and [persistent-array design](08_static_cache_design.md).

Next assign independent outputs to OpenMP workers. A worker should receive already initialized read-only tables and its own scratch. The main source-level choice is the outer work index: triangle block, source triplet, or angular mode. The best choice can vary by stage. Preserve a single owner for every output element, especially the zero angular mode and natural-component transposes. Serial plan construction and serial initialization of lazy readers should precede the worker region.

Then vectorize arithmetic whose access pattern is known. The [SIMD chapter](07_simd_opportunities.md) maps actual operations to `cosmo2D.c` patterns. Begin with geometry, spectrum interpolation where locality is favorable, independent-output radial sums, coupling products, and aperture contractions. Do not count FFTW's existing vector instructions as a gain created by the new C loops. A possible semi-analytic or low-rank reduction from current `main` belongs in a separate experiment with its own derivative and truncation checks.

For covariance, first prepare the shared radial windows, matter inputs, halo profiles, and filter operators. Assemble Gaussian terms from reusable filtered pair correlations where the estimator permits it. Check the proposed one-halo matrix-product reduction against selected direct connected five-/six-field integrals before extending it to all tomographic entries. Retain independent adaptive CPU quadrature where no justified reduction has been established. The existing core provides specific patterns for profile reuse, eight-shell batches, bounded angular scratch, and output-owned SIMD; [Chapter 17](17_cosmolike_covariance_cpu_transfer.md) identifies their source locations and physical limits.

The additional reductions have a specific order. Resolve the reference normalization and permutation findings in [Chapter 19](19_further_covariance_reductions.md), then factor the $`BB`$ term at its common side and reuse halo pair/triple transforms for the one-halo four-, five-, and six-field terms. Prepare the selected long-mode response once per row and redshift. For the data vector, first preserve the finite estimator through the fixed operator in [Chapter 21](21_further_execution_and_cache_design.md); use [Chapter 20's analytic full-filter routes](20_further_data_vector_reductions.md) as independent checks and as production candidates where their estimator assumptions apply. These reductions precede assigning the remaining tasks to workers.

## 4. Numerical comparisons should reveal where disagreement begins

Save intermediate results at a few selected parameter points: IA coefficients, matter spectra, each ordered bispectrum component, projected multipoles, coupled transform inputs, natural components, and aperture outputs. If the final vector disagrees, find the earliest intermediate that disagrees. A phase error and a radial normalization error can both change the aperture amplitude, but they require different fixes.

The physical triangle set should include equilateral, scalene, squeezed, nearly collinear, and side-exchanged configurations, with valid-domain boundaries treated explicitly. Use unequal redshift bins and unequal apertures where the implementation claims to support them. Equal-bin and equal-filter tests are necessary but can conceal field-order mistakes. Inspect signed values and absolute differences near zeros; a relative error alone becomes undefined or misleading there.

Vary one numerical setting at a time: radial quadrature density, angular-multipole cutoff, natural-component mode cutoff, Fourier limits, FFT sampling and padding, interpolation density, and aperture discretization. Each setting controls a different approximation. Once each is understood, repeat a joint refinement to check that apparent convergence was not caused by cancellation between two errors. Keep an independent direct calculation for a small number of triangles even if it is unsuitable for a full forecast.

A documentation pass is part of every major stage. An advanced undergraduate should be able to identify the physical input, the output units, the meaning of every array axis, the approximation being made, and the reason a loop may run in parallel. For vector loops, explain what one lane computes. For a static array, explain its allocation and refill conditions. These explanations should appear beside the relevant function and in a concise module guide; an external study cannot compensate for opaque production code.

## 5. Test cache and thread behavior before measuring it

Test a fiducial model in a fresh process, after a different cosmology, after a photo-z change, and after restoring the initial parameter values. The final fiducial output should agree in all cases. Repeat after changing bin counts, angular grids, FFT dimensions, projection convention, and worker count. These sequences exercise different failure modes from repeatedly requesting the same cached answer.

The critical checks include two likelihood objects initialized sequentially in one process and a second call after FFT plan construction. Changing the allocated number of workers must resize worker scratch where needed. Changing only a shear-calibration coefficient must not trigger a physical recomputation, but it must change the final prediction by the exact multiplicative factor. Changing an IA exponent must change the radial dependence even if the coefficient at the pivot stays fixed.

Thread tests should cover one worker and several higher counts, repeated runs, and the supported processor architectures. Compare full arrays, rather than only a likelihood number after masking. Deterministic assignment of output blocks removes one source of variability; reductions and fused arithmetic may still need a specified comparison tolerance. Any nondeterminism outside that defined arithmetic behavior must be explained before performance work proceeds.

## 6. A later benchmark must answer separate questions

A future benchmark should report cold setup, a repeated identical request, a changed cosmology with reused geometric resources, an IA-only change, a photo-z change, and a calibration-only change. These times answer different questions. The identical request measures cache retrieval; it is not the cost of a Fisher displacement. The cosmology-changing sequence is the relevant comparison for expensive derivatives and training data.

Use current `main` in its exact numerical route as a comparison wherever the physical scope matches. Compare the TATT C path to the checked TATT reference, not to a matter-only `main` calculation. Compare an approximate structured route only after reporting its approximation and matched accuracy. Record the upstream emulator separately with its training domain and derivative accuracy.

| Future measurement | Required context | Question answered |
|---|---|---|
| One-core changed-state time | Same physics, grids, precision, compiler, and inputs | What did reuse, fused loops, and compiled arithmetic change? |
| 1, 2, 4, 8, and 10 workers | Same total workload and physical core allocation | How well does OpenMP scale? |
| Time by calculation stage | Physical preparation, projection, transforms, aperture, assembly | Which operation still limits the result? |
| Peak resident memory | Grid sizes, worker count, cached arrays, and scratch layout | Can the selected process/thread arrangement fit on the target machine? |
| Instructions and memory counters | Available processor-specific counters | Is a loop limited by arithmetic, allocation, or data movement? |
| Compiler vectorization evidence | Build flags, reports, and generated instructions | Which proposed SIMD loops actually vectorize? |
| Full derivative campaign | Parameter list, stencil, steps, and shared core budget | How long does the forecast take? |
| Complete fiducial covariance | Every retained term, estimator, source count, numerical controls, and input preparation | Does the CPU calculation meet the roughly 50-minute target at accepted accuracy? |

Run repeated, serialized jobs on a quiet machine connected to power; record the processor, available physical cores, compiler, library versions, thread affinity, and BLAS/FFT thread settings. Do not infer a speedup by comparing an old published timing to a busy-laptop run. Report spread across repeats and do not run competing benchmarks at the same time.

For Fisher throughput, compare process/thread combinations at the same total core count. Ten one-thread processes can evaluate different cosmologies independently; one ten-thread process can reduce single-vector latency. Intermediate arrangements may be better once memory and serial work are included. CosmoLike's shared global state means different cosmologies should initially live in separate processes, not in concurrent threads that change the same globals.

As an illustration, a four-displacement derivative stencil for 20 parameters plus one fiducial evaluation uses 81 predictions if none can be shared or computed analytically. At 40 seconds each that is 54 minutes of serial prediction time; at four seconds each it is 5.4 minutes. These figures are arithmetic examples, not measured costs. Analytic calibration derivatives and reused IA basis terms can lower the number of expensive evaluations, independently of OpenMP speedup.

Charge the fixed fiducial covariance once per configuration. The user accepts a cost near one hundred times the reported 30-second core calculation, or about 50 minutes. This budget makes a CPU implementation useful even if its latency is much longer than a mean evaluation. Report which expensive inputs survive a scale-mask change and which require recalculation after a physical-model change. Do not hide a missing covariance term behind a timing comparison to a calculation that includes it.

## 7. Make Y3 and Y6 forecast results comparable without conflating them

The first Y3 forecast should use the agreed Y3 apertures, covariance, parameter set, priors, and cuts. Produce two-point-only, aperture-only, and joint results. Compare local derivatives with the checked original pipeline, then establish why any Fisher contour differs from the published sampled posterior. Record the likelihood approximation explicitly.

The first Y6 forecast should use the separately selected Y6 files and replace the provisional priors in the new repository. The five default apertures, including the smallest ones, are candidates rather than validated retained scales. Determine cuts from physical contamination and parameter-bias tests. Keep the resulting selection fixed while evaluating numerical derivatives.

For each survey, report marginalized and conditional parameter uncertainties, the marginalized two-parameter FoM, the joint-to-two-point ratio, and the full nuisance treatment. Repeat the relevant comparison for NLA and TATT, and for a velocity-shear model only when its joint two- and three-point meaning is defined. Changing priors or fixing nuisance parameters can increase a FoM without improving the data; the table should make such changes explicit.

The main covariance comparison should retain the full selected two-point vector and distinguish an analytic baseline, simulation-corrected precision, and any historical simulation-covariance likelihood. Start the PME experiment with a full joint sample covariance and a fixed joint analytic baseline, as specified in [Chapter 18](18_precision_matrix_expansion.md). Treating the analytic two-point block as known and assigning the remaining cross-block to a Wishart covariance is generally invalid. Compare optional compression using the same mock budget and independently chosen weights; assess absolute errors and joint information, not only matrix dimension.

The eventual implementation decision should depend on both runtime and scientific reach. A C predictor that allows checked TATT derivatives on both survey definitions is useful even if a trained network is faster. A further order-of-magnitude performance claim requires a measured reference, and a tighter Y6 forecast requires a validated covariance and physical model. The project should preserve separate records for those two achievements.

## Final implementation decision

The user has authorized the port in `cosmolike/3pcf/`, including its covariance, with a separate test project based on `lsst_y1`. Earlier directory proposals are superseded. [Chapter 24](24_priorities_and_acceptance.md) gives the current sequence and prevents double-counting savings; [Chapter 23](23_unified_execution_and_memory_plan.md) is the combined execution and memory plan.
