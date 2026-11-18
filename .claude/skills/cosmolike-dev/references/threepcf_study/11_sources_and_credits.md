# Sources, versions, and attribution

> **Archive status.** This source study records design proposals and validation requirements, not completed features. Read the [maintained implementation status](../threepcf.md) first. Both the mean and its covariance belong in `cosmolike/3pcf/`; earlier destination proposals using `fastnc/` or `covariances/` are superseded.

## Reproducible source references

This study is based on source inspection, equations, configurations, and primary publications. A commit identifies the source being discussed even when a floating branch changes. It does not identify the authors' publication environment, saved emulator weights, or private survey products unless those are separately supplied. The listed revisions were inspected without installing or executing them. This archive carries links to those upstream revisions instead of copying their source files.

| Source | Inspected version | Pinned evidence |
|---|---|---|
| [fastnc TATT](https://github.com/git-sunao/fastnc/tree/7b2336ce37a909bed51f07f32a2a4f096e71b6ce) | `7b2336ce37a909bed51f07f32a2a4f096e71b6ce` | [Pinned source tree](https://github.com/git-sunao/fastnc/tree/7b2336ce37a909bed51f07f32a2a4f096e71b6ce) |
| [fastnc main](https://github.com/git-sunao/fastnc/tree/e59eab5a07401d2f484bfe00dd48827f2d2162c7) | `e59eab5a07401d2f484bfe00dd48827f2d2162c7`, package version 2.0.84 | [Pinned source tree](https://github.com/git-sunao/fastnc/tree/e59eab5a07401d2f484bfe00dd48827f2d2162c7) |
| [3pcf_integrator](https://github.com/rchgomes/3pcf_integrator/tree/d6098d326dcb03bc750adcc9627119f52d928944) | `d6098d326dcb03bc750adcc9627119f52d928944` | [Selected source files](https://github.com/rchgomes/3pcf_integrator/tree/d6098d326dcb03bc750adcc9627119f52d928944); this is a partial source selection |
| [desy6-3pcf](https://github.com/rchgomes/desy6-3pcf/tree/fd59fe4fee694d359453d5f573d6a4a55a47a812) | `fd59fe4fee694d359453d5f573d6a4a55a47a812` | [Pinned source tree](https://github.com/rchgomes/desy6-3pcf/tree/fd59fe4fee694d359453d5f573d6a4a55a47a812) |
| [threepoint analytic covariance](https://github.com/sheydenreich/threepoint/tree/08de9645a7294092995e5aeaf7b36e48e8fc10a3) | `08de9645a7294092995e5aeaf7b36e48e8fc10a3` | [Selected CPU/CUDA covariance files](https://github.com/sheydenreich/threepoint/tree/08de9645a7294092995e5aeaf7b36e48e8fc10a3), [commit record](https://github.com/sheydenreich/threepoint/commit/08de9645a7294092995e5aeaf7b36e48e8fc10a3) |
| [CosmoSIS Fisher sampler](https://github.com/joezuntz/cosmosis/tree/1a6c123095e45895d44f85cff93211192b62b2b0) | `1a6c123095e45895d44f85cff93211192b62b2b0` | [Sampler](https://github.com/joezuntz/cosmosis/blob/1a6c123095e45895d44f85cff93211192b62b2b0/cosmosis/samplers/fisher/fisher_sampler.py), [derivative code](https://github.com/joezuntz/cosmosis/blob/1a6c123095e45895d44f85cff93211192b62b2b0/cosmosis/samplers/fisher/fisher.py), [commit record](https://github.com/joezuntz/cosmosis/commit/1a6c123095e45895d44f85cff93211192b62b2b0) |
| Local cosmolike_core | HEAD `0c0cf3838852396933c7f49550e35917dad0537c` | [Core directory](../../../../..) |

The inspected core included pre-existing working-tree changes. The archived study did not modify them. Core links resolve against the reader's checkout and may have changed since the source audit; its recorded HEAD is not a claim that every inspected local file matched that commit. The principal numerical mechanisms discussed are in `cosmo2D.c`, CFASTPT, and the covariance code; each chapter names the relevant functions and source locations. A future benchmark should record the actual working-tree state and compiler configuration as well as HEAD.

The fastnc branches have common ancestor `b384353c001d479c2e144717225f81ee968d047a`. The inspected comparison contains 138 commits unique to main and 15 unique to TATT. The [pinned branch comparison](https://github.com/git-sunao/fastnc/compare/7b2336ce37a909bed51f07f32a2a4f096e71b6ce...e59eab5a07401d2f484bfe00dd48827f2d2162c7) and [correction discussion](09_main_branch_reconciliation.md) support the classifications in Chapter 9. The Y6 package pins the same TATT commit inspected here. Fixes in its wrapper therefore do not automatically correct the fastnc dependency.

The covariance paper links the [threepoint releases](https://github.com/sheydenreich/threepoint/releases). The inspected release metadata lists `v2.0.0` from 2022 as the latest release; the later cross-covariance functions were inspected at the commit in the table. [Inspected release](https://github.com/sheydenreich/threepoint/releases/tag/v2.0.0) and [pinned repository tree](https://github.com/sheydenreich/threepoint/tree/08de9645a7294092995e5aeaf7b36e48e8fc10a3) preserve that distinction. Neither this commit nor the CosmoSIS pin establishes a publication's exact runtime environment.

The covariance selection also includes [apertureStatistics.cu](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/apertureStatistics.cu) and [bispectrum.cu](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/cuda_version/bispectrum.cu) at that same commit, used to check the filter and Fourier normalization. The aperture source contains unresolved merge markers and mixed interface history. These pinned sources are reading evidence, not a compiled reference. Chapter 19 separates equations and source discrepancies from any unmeasured effect on published calculations.

## Credits to preserve in a future module

The `cosmolike/3pcf/README.md` should identify **Sunao Sugiyama's fastnc method and implementation** and **Rafael Gomes's intrinsic-alignment and survey-analysis contributions**, and link all three upstream repositories. Credit should describe what is inherited: the multipole construction of natural components, the TATT and velocity-shear kernels, the conversion to aperture statistics, and the Y3/Y6 analysis organization. New C organization, OpenMP work assignment, SIMD arithmetic, and cache management should be identified as implementation contributions to those methods.

The original method reference is [Sugiyama, Gomes, and Jarvis, arXiv:2407.01798](https://arxiv.org/abs/2407.01798). The two requested science references are [Gomes et al., arXiv:2503.03964v2](https://arxiv.org/abs/2503.03964v2) and [Gomes et al., arXiv:2601.09133v1](https://arxiv.org/abs/2601.09133v1). The future module should cite these papers beside the physical methods they supply, so that readers can connect an equation to its scientific source.

The [TATT snapshot's license](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/LICENSE) includes the MIT notice for Sunao Sugiyama and separate notice and redistribution terms for portions derived from Mike Jarvis's software. The two-Bessel source also credits Xiao Fang, with modifications by Sunao Sugiyama. The aperture integration contains TreeCorr-derived expressions associated with Mike Jarvis and the [three-point shear/aperture work](https://arxiv.org/abs/astro-ph/0307393). Preserve the applicable notices in translated files; a citation in prose is not a substitute for carrying the source notices.

The generalized unequal-aperture expressions also require the scientific credit given in the Y6 source to [Schneider, Kilbinger, and Lombardi](https://arxiv.org/abs/astro-ph/0308328). Their expressions connect the three-point natural components to aperture statistics with different filter radii. That contribution is distinct from the software attribution to TreeCorr and from the fastnc transformation method.

No repository-wide license file was found in the inspected integrator tree or Y6 snapshot. Before redistributing translated portions from those repositories, establish the intended license with Rafael and retain any notices already attached to individual components. This is a concrete distribution step for a future implementation, not a missing prerequisite for reading the code or completing this study. No messages have been sent to the authors.

The covariance extension should credit the methods and source contributions of Laila Linke, Sven Heydenreich, Pierre Burger, Peter Schneider, Niek Wielders, and Lucas Porth as applicable to the cited papers and copied components. Its [README](https://github.com/sheydenreich/threepoint/blob/08de9645a7294092995e5aeaf7b36e48e8fc10a3/README.md) explicitly declares GNU GPL version 3 and identifies cubature and Cuba as integration sources. GitHub's missing license metadata must not be read as absence of that declaration. The core declares MIT; a future source translation needs an explicit distribution decision and preserved notices. This study only reads the source and derives proposed calculations.

Credit Friedrich and Eifler for [precision matrix expansion](https://arxiv.org/pdf/1703.07786). Chapter 18 applies that method to the proposed full joint vector and derives the constraints on a signal/noise split. The filtered-correlation and one-halo matrix-product organization in Chapter 15 is a proposed derivation for this port, requiring numerical comparison with the published expressions. Reading source and checking algebra do not establish a validated covariance implementation.

The two-position aperture-correlation formulation used in Chapter 19 is already described by Linke et al. and Wielders et al.; the proposed CPU tables and factorized contractions retain that scientific attribution. Main's exact BiHalofit harmonic primitives also precede Chapter 20's proposed aperture Hankel-product implementation. Its mathematical reorganization does not replace credit for BiHalofit or fastnc. Chapter 21's finite-estimator dual weights are a proposed transpose of the actual selected numerical operations, with their approximation boundaries stated explicitly.

## Primary scientific and numerical references

| Reference | Use in the study |
|---|---|
| [fastnc method, 2407.01798](https://arxiv.org/abs/2407.01798) | Multipole organization, natural components, reported reference performance |
| [Y3 joint inference, 2503.03964v2](https://arxiv.org/abs/2503.03964v2) | Published NLA inference target, data compression, covariance, and posterior information gain |
| [TATT and velocity shear, 2601.09133v1](https://arxiv.org/abs/2601.09133v1) | Physical operators, ordered bispectra, and component-level reproduction targets |
| [BiHalofit, 1911.07886](https://arxiv.org/abs/1911.07886) | Matter-bispectrum prescription and limits of its calibration |
| [Beyond-Limber calculation, 1911.11947](https://arxiv.org/abs/1911.11947) | Numerical context for reusable radial transforms; not a three-point projection supplied by substitution |
| [Third-order aperture covariance, 2212.04485](https://arxiv.org/abs/2212.04485) | Higher-order covariance and estimator considerations |
| [Aperture cross-covariance, 2509.20443](https://arxiv.org/abs/2509.20443) | Second-order–third-order aperture correlations and limits of transfer to the requested statistic |
| [Precision matrix expansion, 1703.07786](https://arxiv.org/pdf/1703.07786) | Simulation corrections to an analytic precision model; full-vector and compressed Y3/Y6 comparisons |
| [Finite simulation covariance, 2108.10402](https://arxiv.org/abs/2108.10402) | Covariance uncertainty and likelihood interpretation |
| [Official DES Y6 cosmology release](https://www.darkenergysurvey.org/des-y6-cosmology-results-papers/) | Survey publication and release context |
| [DES Y6 cosmic shear, 2602.10065](https://arxiv.org/abs/2602.10065) | Source tomography and the two-point modeling comparison |
| [DES Y6 data portal](https://des.ncsa.illinois.edu/releases/y6a2) | Discovery point for survey products; no numerical data products were loaded here |
| [FFTW thread safety](https://www.fftw.org/fftw3_doc/Thread-safety.html) | Plan construction versus execution and shared-library state |
| [FFTW new-array execution](https://www.fftw.org/fftw3_doc/New_002darray-Execute-Functions.html) | Compatible layouts and alignment when reusing a plan on separate buffers |
| [FFTW threading guidance](https://www.fftw.org/fftw3_doc/How-Many-Threads-to-Use_003f.html) | Conditions affecting internal FFT threading and competing outer parallelism |

All performance numbers in the chapters are attributed source or user reports, proposed acceptance targets, or explicitly labeled arithmetic scenarios. The roughly 50-minute covariance target follows from the user's allowance of one hundred times the reported 30-second calculation. No wall-clock time, thread scaling, SIMD gain, or peak-memory measurement is claimed from this study. Source-reading findings establish operations and dependencies; proposed numerical checks establish what must be demonstrated later. This distinction also applies to supplied tests: reading an assertion documents intended behavior, not a passed result.

## Writing and scope rules

The [CosmoLike development guidance](../../SKILL.md) and [didactic explanation guidance](../../../explaining-code/SKILL.md) govern work in this repository. Cocoa's maintenance skill additionally governs changes made in a Cocoa checkout; it is not a prerequisite for reading this portable archive. Define physical quantities before discussing storage, explain parallel ownership and invalidation with concrete examples, and distinguish algebraic identities from numerical approximations.

Only the 25 Markdown study files are included here. Upstream source trees, downloaded metadata, survey products, and trained models are not redistributed in this archive. Links identify pinned public sources wherever available. References to an inspected host project's provider, interface or build file are historical evidence descriptions rather than dependencies on that project. The maintained [implementation status](../threepcf.md) records the actual C module and validation limits; this archived study makes no implementation or benchmark claim.
