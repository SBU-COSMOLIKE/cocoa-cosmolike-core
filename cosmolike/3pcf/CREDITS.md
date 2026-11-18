# Sources, attribution and scope

The numerical implementation covers the matter/IA mean, angular transforms,
aperture estimators and a declared analytic covariance model. Its targets
are [the DES Y3 joint two-/three-point analysis](https://arxiv.org/abs/2503.03964)
and [the TATT and velocity-shear analysis](https://arxiv.org/abs/2601.09133).
Component comparisons are recorded separately from survey reproduction.
Neither a working transform nor a working Fisher driver establishes that
the published cuts, priors, estimator and covariance have been reproduced.

| Source and contributors | Role | Inspected revision |
|---|---|---|
| [fastnc, Sunao Sugiyama](https://github.com/git-sunao/fastnc/tree/tatt) | Natural-component method, IA kernels, external aperture reference | `7b2336ce37a909bed51f07f32a2a4f096e71b6ce` |
| [fastnc main](https://github.com/git-sunao/fastnc) | Numerical corrections and alternative transform methods assessed in the study | `e59eab5a07401d2f484bfe00dd48827f2d2162c7` |
| [3pcf_integrator, Rafael Gomes](https://github.com/rchgomes/3pcf_integrator) | Y3 analysis and aperture-integration workflow | `d6098d326dcb03bc750adcc9627119f52d928944` |
| [desy6-3pcf, Rafael Gomes](https://github.com/rchgomes/desy6-3pcf) | Y6 organization, finite aperture convention and survey-driver reference | `fd59fe4fee694d359453d5f573d6a4a55a47a812` |
| [TreeCorr, Mike Jarvis](https://github.com/rmjarvis/TreeCorr) | Aperture kernels and centroid shear convention retained in the upstream fastnc reference | Notice carried by the pinned fastnc source |
| [threepoint](https://github.com/sheydenreich/threepoint) | Analytic covariance comparison source and CPU/CUDA term inventory | `08de9645a7294092995e5aeaf7b36e48e8fc10a3` |

Credit for the numerical fastnc method belongs to its authors;
[Sugiyama et al.](https://arxiv.org/abs/2407.01798) describe its multipole
organization. The two-Bessel transform source credits Xiao Fang and
modifications by Sunao Sugiyama. Their Mellin-transform and two-dimensional
FFTLog method is implemented in `fftlog_3pcf.c`; `coupling_3pcf.c` and
`transform_3pcf.c` organize angular modes and shear rotations. The serial
FFTW plan preparation, retained worker storage and explicit SIMD
arithmetic follow the local CosmoLike implementation patterns.

The aperture equations come from
[Jarvis, Bernstein and Jain](https://arxiv.org/abs/astro-ph/0307393) and
[Schneider, Kilbinger and Lombardi](https://arxiv.org/abs/astro-ph/0308328),
especially the latter's unequal-radius expressions. The C implementation
uses a cyclic organization of those complex kernels. The external test
reference executes the pinned fastnc `_calculateT` method, whose equal
and unequal branches also provide a check of the equal-radius limit.
The finite-bin measure, mirrored radius sum and output convention follow
the inspected Y6 workflow; this is scientific and workflow attribution,
not a claim to its authorship.

The covariance identities are implemented from the equations and the
Gaussian moment theorem. Scientific credit goes to
[Linke et al.](https://arxiv.org/abs/2212.04485) for third-order aperture
covariance and [Wielders et al.](https://arxiv.org/abs/2509.20443) for the
second-/third-order cross-covariance treatment. The relevant contributors
include Laila Linke, Sven Heydenreich, Pierre Burger, Peter Schneider,
Niek Wielders and Lucas Porth. The response Gram construction is a
calculation pattern within a specified connected model; it is not a
replacement for those physical covariance derivations.

The matter fit in `physics_3pcf.c` is
[BiHalofit, Takahashi et al.](https://arxiv.org/abs/1911.07886).
Its coefficients, nonlinear-scale construction and baryon-response
convention follow the pinned fastnc source. The TATT and velocity-shear
coefficients follow the analysis linked above. All nineteen active
intrinsic-shape components have pinned-source comparisons, including
the distinction between the NLA baseline and the residual terms.

The project precision-expansion implementation follows
[Friedrich and Eifler](https://arxiv.org/abs/1703.07786).
It uses a full joint sample covariance and an explicitly declared number
of independent simulations, or two independent sample groups. It does
not apply a Wishart correction to an arbitrary difference between noisy
covariances. The project tests check its finite-sample second-order
expectation separately from any survey realization.

## License notices

[LICENSE.fastnc-TreeCorr.txt](LICENSE.fastnc-TreeCorr.txt) retains the
complete MIT notice for Sunao Sugiyama and the Mike Jarvis redistribution
terms carried by the inspected fastnc source. Preserve these notices with
fastnc-derived code and its external references. The core's license
continues to govern its original contributions.

The inspected `threepoint` README declares GPL version 3. Its CPU/CUDA
implementation has not been copied into these components. The covariance
calculation is independently written from the published equations and
their estimator definitions. A direct source translation would require
a distribution choice compatible with those terms. The inspected Rafael
Gomes repositories did not contain a repository-wide license file; the
survey adapters are independently written and keep their scientific and
workflow credits. The Y3 downloader retrieves only the explicitly listed
public data/configuration files, verifies their pinned Git blob hashes,
and retains their repository paths in its provenance manifest.
