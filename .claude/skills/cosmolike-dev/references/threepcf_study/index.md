# Three-point predictions and joint covariance for DES Y3 and Y6

> **Archive status.** This source study records design proposals and validation requirements, not completed features. Read the [maintained implementation status](../threepcf.md) first. Both the mean and its covariance belong in `cosmolike/3pcf/`; earlier destination proposals using `fastnc/` or `covariances/` are superseded.

1. [Recommendation and scope](#scope)
2. [Repositories and scientific targets](#sources)
3. [What the source review establishes](#findings)
4. [Reading guide](#reading)

## Recommendation and scope <a name="scope"></a>

Proceed with a C implementation centered on reducing integrals, sharing physical calculations, and then applying OpenMP, SIMD, and persistent arrays. The inspected sources permit more than translating their current loops: covariance triangle products can be factored at a common side, halo terms can share transforms of profile products, and an aperture-only data vector can use precomputed weights through its fixed transform stages. These opportunities extend the reuse already present in fastnc. [Covariance reductions](19_further_covariance_reductions.md), [mean reductions](20_further_data_vector_reductions.md), [execution and cache design](21_further_execution_and_cache_design.md).

The user-selected destination is `cosmolike/3pcf/`, beside `cosmolike/covariances/`, for both predictions and the **CPU implementation of the joint analytic covariance**, with separate survey configurations for **DES Y3 and DES Y6**. Rafael's Y6 package provides an existing interface and analysis organization worth preserving. C can supply the expensive numerical calculation while Python continues to describe input files, priors, masks, and Fisher jobs. This archive contains the documentation study without the source snapshots. Its source inspection and algebra did not execute scientific calculations or timings. The maintained implementation status records the C components, adapters, and remaining verification work.

The prediction target is the cost of a changed physical model, including Fisher displacements. The source review identifies ways to reduce repeated work and distribute independent calculations across ten cores. The thread allocation and physical settings behind the approximately 40-second starting estimate have not been verified, so it cannot be divided by ten to predict an observed result. The [performance chapter](04_design_and_performance.md) separates operation counts from hypothetical timing scenarios.

The covariance has a different budget. The user's approximately 30-second existing calculation and willingness to spend 100 times longer allow a target near **50 minutes per fiducial joint covariance**. That is an acceptance target, not a runtime prediction. CUDA is not required for the proposed CPU route: upstream already contains CPU cubature, and the proposed reductions share profiles, filters, and projection work before applying OpenMP and SIMD. [Analytic covariance port](15_analytic_covariance_cpu_port.md), [actual CosmoLike covariance mechanisms](17_cosmolike_covariance_cpu_transfer.md).

## Repositories and scientific targets <a name="sources"></a>

These are the requested URLs, together with the separate role of each source. Commit-pinned links and attribution details appear in the [source record](11_sources_and_credits.md).

| Source | URL | Role |
|---|---|---|
| Sunao Sugiyama's fastnc, TATT branch | [github.com/git-sunao/fastnc/tree/tatt](https://github.com/git-sunao/fastnc/tree/tatt) | Explicit TATT and velocity-shear kernels; historical direct calculation |
| fastnc main | [github.com/git-sunao/fastnc](https://github.com/git-sunao/fastnc) | Relevant numerical fixes, independent calculations, and alternative transform methods |
| Rafael Gomes's 3pcf_integrator | [github.com/rchgomes/3pcf_integrator](https://github.com/rchgomes/3pcf_integrator) | Y3 prediction, aperture integration, emulation, covariance, and inference reference |
| Rafael Gomes's DES Y6 package | [github.com/rchgomes/desy6-3pcf](https://github.com/rchgomes/desy6-3pcf) | Survey interface, Y6 binning, aperture integration, emulation, and Fisher configuration |
| Joint two-point and three-point analysis | [arXiv:2503.03964](https://arxiv.org/abs/2503.03964) | Reproduce the simulated DES Y3 NLA analysis at its stated inference scope |
| TATT and velocity-shear analysis | [arXiv:2601.09133](https://arxiv.org/abs/2601.09133) | Reproduce component predictions and physical responses, including triangle-dependent results |
| Original fastnc method | [arXiv:2407.01798](https://arxiv.org/abs/2407.01798) | Credit and understand the multipole-based natural-component calculation |
| Analytic covariance implementation | [github.com/sheydenreich/threepoint](https://github.com/sheydenreich/threepoint), [releases](https://github.com/sheydenreich/threepoint/releases) | CPU and CUDA aperture covariance; the selected terms and limitations are mapped in Chapter 15 |
| Third-order aperture covariance | [arXiv:2212.04485](https://arxiv.org/abs/2212.04485) | Gaussian, connected, and finite-field covariance terms |
| Second-order–third-order aperture covariance | [arXiv:2509.20443](https://arxiv.org/abs/2509.20443) | Cross-covariance expressions and the public code extension |
| Precision matrix expansion | [arXiv:1703.07786](https://arxiv.org/pdf/1703.07786) | Correct an analytic covariance's inverse using simulations without inverting their sample covariance |

Four configurations should remain identifiable: the first paper's NLA reproduction, the second paper's TATT reproduction, a Y3 Fisher forecast, and a Y6 Fisher forecast. A Fisher matrix describes local parameter sensitivity and observational uncertainty. It does not by itself reproduce the marginalized posterior of the first paper. The Y3 reference has four equal aperture radii and 80 three-point entries; the inspected Y6 defaults have five radii and 100 entries before cuts. Their covariance, source distributions, calibration priors, and scale selection must describe their respective measurements.

Use the **full selected two-point vector** as the primary Y3/Y6 forecast option. Precision matrix expansion supplies a concrete route for testing an analytic joint covariance with simulation corrections. Its convergence and sampling assumptions must be checked for the cubic statistic; a singular sample covariance alone does not prevent the method. Compression remains a comparison at the same simulation budget and belongs in the historical Y3 reproduction. Compare absolute errors and joint information, including the cross-covariance, before choosing between the full and compressed vectors. [Joint PME proposal](18_precision_matrix_expansion.md).

## What the source review establishes <a name="findings"></a>

**Main should be considered.** Its FFTLog correction returns a windowed Fourier array that the TATT branch computes and discards. Other commits repair refactoring-specific normalization or convention choices, so applying all of them to TATT would be wrong. In particular, the centroid-conversion factors differ by complex conjugation for the same supplied geometry. The complete phase conventions need a derivation and independent comparison before either branch becomes the numerical reference. Main also supplies structured transform methods worth assessing for compatible physical terms; it does not supply the explicit TATT and velocity-shear bispectra required here. [Branch reconciliation](09_main_branch_reconciliation.md)

**Rafael's Y6 package contributes more than survey configuration.** It corrects the older transfer of the quadratic IA redshift exponent, separates IA settings between instances, removes the problematic unequal-filter branch, and caches aperture weights. The inherited interpolation settings still have a separate shared-state issue. Its covariance products and trained networks are external, its prior widths are placeholders, and its emulator design omits velocity shear and fixes neutrino mass despite a varying inference parameter. These distinctions define the required direct-versus-emulated checks. [Y6 assessment](10_desy6_workflow.md), [physical limits](12_second_pass_physics.md), [numerical checks](13_second_pass_numerics.md).

**CosmoLike provides concrete numerical patterns.** Its non-Limber calculations reuse transforms and execute independent FFTs through OpenMP with compatible plans and separate buffers. Its CFASTPT machinery batches related spectral terms. Its vector loops show how to process contiguous geometry and interpolation work while retaining a scalar mathematical explanation. These patterns can guide a three-point implementation, but its two-point IA tables and covariance formulas do not already constitute the required three-point physics. [Core patterns](03_cosmolike_patterns.md), [SIMD opportunities](07_simd_opportunities.md), [persistent arrays](08_static_cache_design.md)

**Shared IA parameters require an explicit conversion.** The core and upstream code use different normalization constants, pivot redshifts, growth inputs, and coefficient conventions. The transverse tree-level quadratic kernels depend on one combination, $`C_{1\delta}+C_2/3`$, at fixed redshift. This can reduce repeated arithmetic and supplies a derivative check, but does not erase their different survey redshift dependence or two-point loop information. The velocity-shear equations also require a coefficient and E/B convention reconciliation. [Physics derivations](12_second_pass_physics.md), [core interface checks](14_second_pass_integration.md).

**The covariance has further integral reductions.** A bispectrum-product term factors into two cached triangle integrals and one common-side contraction. Pair/triple halo-profile transforms also reduce the one-halo power–trispectrum and connected terms, including a defined finite center window. The source's selected long-mode term becomes a product of prepared responses. Full-filter noise has an analytic reference, and source-window products can be attached after halo transforms. Finite triangle cuts, catalogue masks, IA, and the $`\xi_\pm`$ cross-block still need their specific estimator. These are derived proposals with validation requirements, not measured speedups. [Full derivations](19_further_covariance_reductions.md).

**The data vector has two complementary routes.** For full aperture filters, BiHalofit's three-halo term admits one-dimensional Hankel-product integrals, and selected TATT terms admit analytic opening-angle integration. For the existing finite-bin estimator, fixed weights can be moved backward through the linear FFT and aperture stages and retained between model evaluations. Logarithmic interpolation and value-dependent endpoint extrapolation mark boundaries to that linear rearrangement. The seven E-only IA residual labels can also be collected into three quadratic insertions before projection. [Mean algebra](20_further_data_vector_reductions.md), [finite-estimator operator](21_further_execution_and_cache_design.md).

**The final stress tests distinguish mean and covariance.** A numerical adjoint preserves the forward prediction but does not derive the measured covariance. Fixed cuts can retain a generalized one-halo response factorization; full TATT covariance needs additional physics. The audit also identifies two cache collisions and distinct discrete matter/IA radial weights. [Stress tests and source evidence](22_stress_test_of_reductions.md).

**The reference needs independent normalization checks.** The covariance source has a cross-$`PB`$ angular-normalization discrepancy relative to the paper, an unequal-radius permutation defect, and an unresolved merge conflict in an additionally inspected mean-source file. These findings do not establish an error in a published run. They require selecting and checking the reference before treating agreement with a source return value as correctness. [Specific evidence and checks](19_further_covariance_reductions.md).

**Fisher and simulation corrections must consume the intended matrix.** The inspected sampler uses exported inverse covariance, so enabling a finite-simulation likelihood alone does not change its Fisher curvature. The zero quadratic-alignment fiducial has an unidentifiable redshift exponent; an emulator must not create information in that direction. PME offers a full-vector route but requires a joint analytic baseline, an appropriate sampling model, and independent validation. [Fisher execution](16_fisher_execution_checks.md), [precision correction](18_precision_matrix_expansion.md).

## Reading guide <a name="reading"></a>

The intended reader knows undergraduate calculus, linear algebra, and Fourier transforms, but need not know fastnc or CosmoLike. Intrinsic alignment is the correlation of galaxy shapes with their environment. NLA means nonlinear alignment; TATT means tidal alignment and tidal torquing. A natural component is a three-shear correlation with its complex phase defined relative to the triangle. Aperture skewness integrates such correlations with specified filters. These physical definitions are developed before the implementation recommendations.

| Chapter | Question answered |
|---|---|
| [1. Physics and reproduction](01_physics_and_reproduction.md) | Which equations, conventions, and results must agree? |
| [2. Upstream source audit](02_upstream_source_audit.md) | What does the historical pipeline already reuse, and where can errors or repeated work enter? |
| [3. CosmoLike patterns](03_cosmolike_patterns.md) | Which actual C mechanisms transfer to this calculation? |
| [4. Design and performance](04_design_and_performance.md) | How should work be shared, distributed, and stored? |
| [5. Covariance and Fisher forecasts](05_covariance_fisher_des_y6.md) | What makes Y3 and Y6 absolute errors and information gains interpretable? |
| [6. Work plan and acceptance](06_work_plan_and_acceptance.md) | What should be implemented and checked, in what order? |
| 7. [SIMD opportunities](07_simd_opportunities.md) | Which loops can evaluate several independent values per instruction? |
| [8. Persistent-array design](08_static_cache_design.md) | Which arrays can survive a call, and what changes require refilling them? |
| [9. Main-branch reconciliation](09_main_branch_reconciliation.md) | Which fixes and algorithms should enter the proposed implementation? |
| [10. Rafael's Y6 workflow](10_desy6_workflow.md) | How can one numerical interface serve two separate survey definitions? |
| [11. Sources and credits](11_sources_and_credits.md) | Which exact versions and original contributions support the study? |
| [12. Physical derivations and limits](12_second_pass_physics.md) | Which kernel identities, coefficient ambiguities, and model domains need independent tests? |
| 13. Further [numerical checks](13_second_pass_numerics.md) | What do bin averaging, grid boundaries, interpolation, and shared state change? |
| [14. Core integration checks](14_second_pass_integration.md) | How do units, IA normalization, growth, setters, and ownership map between codes? |
| [15. Analytic covariance CPU port](15_analytic_covariance_cpu_port.md) | Where is the public code, which terms does it supply, and how can some integrals be reduced? |
| [16. Fisher execution checks](16_fisher_execution_checks.md) | Which precision, parameter steps, priors, and null directions does the sampler use? |
| [17. CosmoLike covariance mechanisms](17_cosmolike_covariance_cpu_transfer.md) | Which existing halo, projection, batching, FFT, OpenMP, and SIMD patterns transfer? |
| [18. Precision matrix expansion](18_precision_matrix_expansion.md) | How can simulations correct the analytic joint model while retaining the full two-point vector? |
| [19. Further covariance reductions](19_further_covariance_reductions.md) | Which shared-side integrals, connected correlations, halo products, and noise expressions remove repeated work? |
| [20. Further data-vector reductions](20_further_data_vector_reductions.md) | Which full-aperture integrals and TATT terms admit a different analytic organization? |
| 21. Further [execution and cache design](21_further_execution_and_cache_design.md) | How can finite-estimator weights, interpolation stencils, covariance batches, and Fisher predictions be reused? |
| [22. Stress tests](22_stress_test_of_reductions.md) | Which identities survive finite cuts, noise, IA and a changed numerical grid? |
| [23. Unified execution and memory](23_unified_execution_and_memory_plan.md) | What coexists in memory, and which worker owns each stage? |
| [24. Priorities and acceptance](24_priorities_and_acceptance.md) | Which reductions combine, what should be implemented first, and what establishes completion? |

Begin with Chapters 22–24 for the final stress tests, combined execution and memory plan, and implementation priorities. Chapters 19–21 derive the additional integral, cache, OpenMP, and SIMD opportunities. Chapters 1, 4–6, 15, 17, and 18 explain the broader scientific and implementation decision. The future module must credit Sunao Sugiyama, Rafael Gomes, and the contributors to the inherited transforms and aperture methods, alongside the covariance authors and Friedrich–Eifler precision method. [Chapter 11](11_sources_and_credits.md) records sources and notices; the GPL-3.0 declaration of the covariance repository needs an explicit distribution decision before translating it into the MIT-licensed core.
