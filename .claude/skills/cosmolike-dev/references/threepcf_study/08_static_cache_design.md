# Persistent arrays: what can remain in memory between evaluations

> **Archive status.** This source study records design proposals and validation requirements, not completed features. Read the [maintained implementation status](../threepcf.md) first. Both the mean and its covariance belong in `cosmolike/3pcf/`; earlier destination proposals using `fastnc/` or `covariances/` are superseded.

Persistent arrays are a central part of the proposed calculation. The useful distinction is between quantities that remain physically unchanged, memory that can be reused for new values, and results whose inputs have changed. CosmoLike already implements all three cases. A C rewrite can carry these distinctions into the three-point calculation, provided every stored result records the inputs that made it valid.

This chapter proposes a design, based on source inspection. It contains no implementation or measured timing. Local source references refer to the inspected CosmoLike checkout at commit `0c0cf3838852396933c7f49550e35917dad0537c`; the surrounding working tree has unrelated modifications. The upstream revisions are recorded in [the source audit](02_upstream_source_audit.md).

## 1. What `static` does, and what it does not do

A C variable declared `static` retains its storage for the lifetime of the process. A static pointer can retain the address of an allocated array after the function that prepared it returns. This avoids allocating that array on every evaluation. It does not make the array's contents constant, prove that its values belong to the current cosmology, or protect simultaneous reads and writes. Those properties require explicit rules.

There are two separate questions whenever a cached array is used. **Is its allocation large enough and correctly laid out?** This depends on dimensions, padding, alignment, and the number of worker buffers. **Do its values describe the requested calculation?** This depends on cosmology, nuisance parameters, source distributions, numerical grids, model choices, and estimator settings. An array may answer yes to the first question and no to the second: it needs new values in its existing memory. Conversely, a new bin count can require a new allocation even when the cosmology is unchanged.

`xi_pm_tomo` demonstrates this separation. Its static `Glpm` angular kernels, `Cl` spectra, and `xipm` predictions have an allocation/geometry condition followed by a physics-refill condition. The refill condition includes cosmology, source photo-z, IA, source-distribution, and numerical-setting tags. The pair count is compared directly because it determines the array dimensions. The tags are `uint64_t` values in the inspected source; a changed tag means that its parameter group changed. These tags are often called nonces. They identify changes, rather than encoding the numerical parameter values. [Cache explanation and allocation guard](../../../../../cosmolike/cosmo2D.c), [physics-refill guard](../../../../../cosmolike/cosmo2D.c), [tag types](../../../../../cosmolike/structs.h).

The proposed three-point module should use the same discipline. Names such as `triangle_geometry`, `radial_weights`, `bispectrum_basis`, `fft_workspace`, and `aperture_weights` would describe distinct stored objects. Each needs an allocation key and a value key beside its arrays. A single flag saying that initialization happened would be insufficient.

## 2. What the current CosmoLike arrays teach us

The non-Limber FFTLog code separates the forward transform from the order-dependent inverse transforms. Its first phase retains FFTW plans and taper tables. Its second phase retains inverse plans and per-thread work buffers; the allocation guard checks thread count, maximum transform length, block capacity, component count, and individual transform lengths. Yet its radial-grid phase factor is recalculated because the comoving-distance grid changes with cosmology. This is a direct example of reusable transform machinery surrounding cosmology-dependent values. A reusable plan is not a reusable transformed spectrum. [Forward-phase storage](../../../../../cosmolike/cosmo2D.c), [inverse-phase storage and guards](../../../../../cosmolike/cosmo2D.c).

The FKEM cross-pair calculation uses a fixed block of 16 multipoles for its large transformed arrays. Pointer views expose the current multipoles, while one bounded pool supplies their memory. The wavenumber output grid is shared among bins because its values are identical there. These choices prevent storage from growing as the product of every bin, multipole, component, and radial node. The same idea applies to triangle tiles in `fastnc`: persist a bounded pool, refill it, project its contents into the accumulated outputs, and reuse it for the next tile. [Blocked non-Limber storage](../../../../../cosmolike/cosmo2D.c).

CFASTPT retains frequency grids, gamma-function factors, padded input coordinates, FFTW plans, and work arrays. Its inspected `J_abJ1J2Jk_ar` checks the padded size and term count, and also notices changes in grid origin or spacing. The term calculation uses prepared Fourier transforms and gamma factors, then writes new convolution results. The higher-level IA table is refilled when cosmology changes. IA nuisance amplitudes are applied elsewhere, so the amplitude-independent loop spectra can be reused when only those amplitudes change. [CFASTPT persistent state](../../../../../cfastpt/cfastpt.c), [term calculation](../../../../../cfastpt/cfastpt.c), [IA table refill](../../../../../cosmolike/pt_cfastpt.c).

Precomputation does not always imply persistent storage. In `C_ss_tomo_limber_work`, the source-kernel array `WC`, spectrum array `KIA`, and per-thread power-reader scratch `KPN` are allocated for that work call. They remove repeated function evaluations inside the integration loops, even though they are not static arrays. A future port can first copy that dependency separation, then retain the allocations if profiling shows that repeated allocation matters. It should not retain large old values merely because retaining their memory is useful. [Per-call precomputed arrays](../../../../../cosmolike/cosmo2D.c).

## 3. Proposed inventory of persistent objects

The table separates allocation requirements from the inputs that determine values. A value change can also change dimensions, in which case both operations are needed. “Geometry” means numerical or estimator geometry, not necessarily cosmological distances.

| Stored object | Allocation key | Inputs requiring new values | What can reuse it |
|---|---|---|---|
| Triangle shape invariants and rotations | Number of triangle nodes; component convention | Triangle coordinates, orientation convention, physical-domain mask | All cosmologies on the same angular grid |
| Interpolation brackets and weights | Input/output grid sizes; stencil size | Actual grid coordinates, interpolation rule, boundary prescription | New bispectrum values on the same grids |
| Angular multipole coefficients | Angular-node count; maximum order | Angular nodes, basis normalization, integration rule | All cosmologies and nuisance amplitudes |
| Geometric coupling factors | Orders; unique angular-ratio count | Ratio grid, projection convention, integration tolerance | All source bins and cosmologies |
| Bessel/FFTLog factors | FFT lengths; retained orders | Log-grid spacing and origin, biases, bin widths, padding, derivative order, taper choice | New spectra on an identical transform grid |
| FFTW plans | Transform type, dimensions, strides, layout, alignment, planning policy, FFT thread count | The same plan-defining settings | Arbitrarily many new input arrays satisfying those settings |
| Worker scratch | Worker count; tile capacity; padded lengths; active-component count | No stored physical result to validate | Every evaluation after required entries are overwritten |
| Power-spectrum row data | Wavenumber and radial-node counts | Cosmology, power prescription, baryon inputs, queried coordinates | Several IA terms and bin triples within an evaluation |
| Amplitude-independent bispectrum terms | Tile size; basis-term count | Cosmology, power prescription, baryons where applicable, physical wavenumbers | IA-amplitude changes at fixed cosmology |
| Radial source/lensing weights | Source-bin count; radial-node count | Source distributions, photo-z parameters, cosmological distances, radial nodes | All triangle shapes for those inputs |
| Aperture and estimator weights | Aperture count; triangle/output-grid sizes | Aperture radii, angular units, bin averaging, triangle cuts, convention | All cosmologies sharing the estimator |
| Aperture weights after applying the transpose of the fixed transform | Filter count; actual tuned grid and mode counts | Every included linear transform, phase, taper, endpoint rule, mask, and filter | Aperture-only calls through the same finite estimator; nonlinear preprocessing stays outside |
| Filtered bispectrum and halo-product transforms | Physical-node and filter counts; streamed tile sizes | Model, source-weight convention, actual bounds, profiles, filters, transform rules | Compatible mean, $`BB`$, one-halo $`P P_4`$, and connected covariance terms |
| Fixed covariance factor or inverse | Kept data length; matrix layout | Covariance contents, mask, ordering, compression, correction prescription | Every mean evaluation under a fixed-covariance forecast |

The additional shared objects are derived in [Chapter 19](19_further_covariance_reductions.md) and [Chapter 20](20_further_data_vector_reductions.md). [Chapter 21](21_further_execution_and_cache_design.md) specifies the finite-estimator dual weights, interpolation stencils, and exact Fisher prediction matching. A cache of an integral must include its domain: the upstream covariance cutoff depends on more than the two local filter radii in some factors.

### Shape geometry and physical wavenumbers have different lifetimes

Triangle dot products normalized by side lengths depend on shape. An angular triangle's ratios and orientation can remain fixed while cosmology changes. Its physical wavenumbers do not: the Limber mapping divides each multipole by a cosmology-dependent distance. Thus a stored angular shape table can survive a cosmology update, whereas cached $`P(k,z)`$ values and interpolation brackets for the physical $`k`$ queries generally cannot.

The upstream TATT calculator repeatedly reconstructs vectors for the same triangles. Storing the shape invariants once would avoid that repetition, but it must preserve the association between Fourier legs, source bins, and intrinsic-shape fields. A side permutation is not just an exchange of array indices when different fields occupy the legs. [Upstream triangle construction](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L146).

### Cache angular coupling compactly

The coupling $`G_{LM}`$ depends on angular ratio, rather than separately on every pair of Fourier-grid coordinates. `fastnc` already identifies duplicate ratios within a call and has an option to tabulate this coupling. A C module should retain the unique-ratio representation and a map from grid entries to ratios when that reduces memory. On an equal logarithmic grid, the ratio depends on the difference between grid indices, giving far fewer distinct ratios than grid pairs. [Upstream unique-ratio preparation](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/coupling.py#L195).

A dense coupling array can otherwise become large. For illustration, 51 bispectrum orders, 101 nonnegative natural-component orders, and a $`256\times256`$ grid require $`51\times101\times256^2\times8`$ bytes, or about 2.70 GB in double precision. This is arithmetic from an illustrative layout, not a memory measurement or a recommended grid. Keeping angular ratios compact, generating one order block at a time, or retaining only the contraction needed next can avoid that allocation.

### Retain power-reader organization without changing the physics

CosmoLike's `p_lin_at_a` and `p_nonlin_at_a` read many wavenumbers at one scale factor. They find the redshift bracket once and use direct indexing on the validated logarithmic wavenumber grid. A three-point evaluation can prepare the three legs' power values for a tile and reuse them across IA terms. The brackets depend on the actual queries and table coordinates, so equal array lengths are not enough to establish validity. The conversion between CosmoLike's internal units and the upstream Mpc/$`h`$ convention must precede this lookup. [Batched power readers](../../../../../cosmolike/cosmo3D.c).

The upstream nonlinear IA reader scales a supplied zero-redshift nonlinear spectrum by the square of the growth factor. Using CosmoLike's full redshift-dependent nonlinear table changes that prescription. A cache must include a power-prescription identifier, and reproduction must retain the specified prescription until a separate physics comparison is made. [Upstream nonlinear reader](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L106).

### Factor IA amplitudes before deciding what to retain

An amplitude-independent basis separates triangle and power-spectrum terms from $`A_1`$, $`A_2`$, their redshift slopes, density weighting, and velocity shear. Terms can then be combined with coefficients at each radial node. Changing an amplitude requires new coefficients and final contractions, but can preserve the underlying basis. Changing a redshift slope also changes the radial weighting of that basis. It cannot generally be represented by one constant multiplication of the final projected spectrum.

This separation must be derived from the selected equations. The upstream `Ct` input is multiplied internally by $`C_1(z)`$, so an $`A_1`$ update also changes the effective velocity-shear coefficient. A future cache must record whether the user specified that ratio or an independent physical velocity-shear coefficient. The chosen NLA or TATT model and the velocity-shear option are explicit keys; a zero amplitude does not erase the model choice. The working NLA baseline and a velocity-shear-off calculation retain their stated scope, while reproducing the second paper requires its nonzero velocity-shear cases. These choices must not be inherited from CosmoLike's separate galaxy-clustering nonlinear-Kaiser switch. [Upstream coefficient construction](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L597).

## 4. Which parameter changes invalidate which results?

The following table assumes fixed angular transform grids and a fixed forecast covariance unless the corresponding row changes them. “Refill” means recalculate values in existing memory when dimensions permit. Every change still requires a final prediction to be assembled from consistent inputs.

| Changed input | Geometry/FFT machinery | Cosmology and bispectrum basis | Radial or IA coefficients | Final statistic and covariance |
|---|---|---|---|---|
| Cosmological parameter | Keep fixed angular geometry; update any distance-dependent transform grid | Refill power, distances, growth, and basis | Refill lensing/source-coordinate weights and IA growth factors | Recompute statistic; keep declared fixed covariance |
| $`A_1`$ | Keep | Keep amplitude-independent terms | Refill $`C_1`$, density weighting, and any $`C_1`$-scaled velocity term | Recombine and project |
| $`A_2`$ | Keep | Keep amplitude-independent terms | Refill $`C_2`$ | Recombine and project |
| $`\eta_1,\eta_2`$ or upstream $`\alpha_1,\alpha_2`$ | Keep | Keep amplitude-independent terms | Refill their redshift-dependent coefficients | Reintegrate radial contractions |
| $`b_{\rm TA}`$ | Keep | Keep amplitude-independent terms | Refill density-weighted alignment coefficient | Recombine and project |
| Velocity-shear amplitude | Keep | Keep a basis that already includes the requested terms | Refill its coefficient using the declared normalization | Recombine and project |
| Shear calibration $`m_i`$ | Keep | Keep | Keep source weights under multiplicative-only calibration | Rescale uncalibrated outputs; keep declared fixed covariance |
| Photo-z shift/stretch | Keep fixed angular geometry | Keep only if the radial queries remain fixed | Refill source distributions and lensing weights | Reproject; changed support can require new radial queries |
| Source-distribution values or radial grid | Keep fixed angular geometry | Refill queried rows if radial nodes change | Refill weights and normalization | Reproject; reallocate if node counts change |
| Aperture radii, angular binning, or triangle cuts | Rebuild affected estimator weights and possibly transform grids | Keep only values sampled on unchanged grids | Keep unchanged radial weights | Reassemble statistic; rebuild masked/compressed covariance |
| Source-bin count or ordered-bin list | Resize bin-dependent arrays; rebuild index maps | Keep shared physical terms | Rebuild bin-dependent weights | Rebuild output ordering, mask, and covariance layout |
| Worker or FFT thread count | Resize worker scratch; recreate plans if their thread setting changes | Keep physics values | Keep physics values | Require reproducibility checks |

For multiplicative calibration, a three-bin aperture moment receives the product $`(1+m_i)(1+m_j)(1+m_k)`$. Repeated bin indices repeat their factor. An auto-bin third moment therefore has a cubic calibration factor. Storing the uncalibrated result makes these updates and their derivatives inexpensive without touching the FFT calculation. This conclusion assumes calibration enters only through those multiplicative factors; any selection-dependent calibration model would need additional keys.

The first implementation should invalidate whole named objects when one of their dependencies changes. For example, a photo-z change can refill the entire radial-weight object. Selective updates of only affected bin rows are possible later, but require a dependency map from bins to every affected ordered triple. A partially updated table must never carry the same validity stamp as a fully updated table. Such mistakes can produce smooth but incorrect Fisher derivatives, because successive displaced parameter points reuse different mixtures of old and new rows.

## 5. Reuse bounded scratch; do not retain every intermediate value

At the upstream default TATT interpolation dimensions, a full triangle-by-redshift component contains $`35\times40\times40\times170`$ doubles. Seven components require about 533 MB before geometry and temporary arrays. Retaining a separate copy for each bin triple would multiply storage without adding physical information: the same unprojected spectrum is reused with different weights. [Upstream grid construction](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/bispectrum.py#L211).

A proposed tile of 256 triangles, 170 radial nodes, and seven components occupies approximately 2.44 MB in doubles. That illustrative figure excludes powers, geometric arrays, transform buffers, and alignment padding. It shows why a bounded tile can be retained per worker while the calculation streams through the full triangle grid. Tile size must ultimately be chosen by measurement on the intended CPUs. Retaining the allocation does not mean that the previous tile's spectrum remains valid: the next tile must overwrite every value that will be read, including any FFT padding.

A separate decision is whether to retain a full amplitude-independent basis between Fisher derivatives. This can help when many nuisance derivatives share one cosmology, but the basis can have more terms than the seven combined components. Measure that memory cost explicitly. A reasonable first design can retain prepared geometry and bounded workspaces while recomputing the physical tile for each cosmological displacement, then reuse it across all bin triples and relevant amplitude contractions within that evaluation.

## 6. Initialization, threads, and the covariance

All allocation, plan creation, and lazy initialization should happen before worker threads read the prepared objects. `C_ss_tomo_limber_work` explicitly warms cosmology, source-kernel, and IA readers before entering its OpenMP loops. The same boundary should be visible in the three-point module: prepare inputs serially, publish the completed tables, then assign independent outputs or scratch rows to workers. A single worker must own each writable scratch region. [Serial preparation before parallel reads](../../../../../cosmolike/cosmo2D.c).

FFTW plan reuse and FFT threading are separate choices. The current CFASTPT example executes reused plans on distinct term arrays inside an outer OpenMP loop. A proposed internally threaded FFT plan must record its configured FFT thread count and must be benchmarked with the outer worker count controlled. Increasing both levels independently can create more active threads than allocated cores. Plan layout, strides, alignment, and input/output arrangement must remain compatible with every array used for execution. [Reused plans on term arrays](../../../../../cfastpt/cfastpt.c).

Different cosmologies should initially run in separate processes, each with its own CosmoLike state and caches. Two threads must not call global cosmology setters while other threads are reading old tables. Explicit independent contexts would be another design, but retaining global static arrays does not provide that independence. Upstream shared configuration dictionaries also need attention: object construction can mutate settings shared by another object. The proposed C ownership rules should prevent the equivalent effect, with each model's keys and settings belonging to one declared owner. [Upstream configuration mutation](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/utils.py#L117).

Every new model-defining static must have a reset invoked by `initial_setup` or the documented module initialization immediately after it. CosmoLike already resets file-scope pair lists, model switches, external-table-origin flags, and other state outside its main structs. Reusable storage can remain allocated only if its validity markers force the next model to establish its own values. Equality of pointer addresses does not establish equality of physics; an input buffer may retain its address while its contents change. [Model reset sequence](../../../../../cosmolike/generic_interface.cpp).

The covariance belongs to a different lifetime. Under a fixed-covariance Fisher forecast, its matrix factorization or inverse is prepared once for the selected data ordering and mask. The inspected CosmoLike interface performs a Cholesky positivity check on the normalized correlation matrix, releases that factor, then forms and stores an inverse with a residual check. Retaining a Cholesky factor for repeated solves would be a proposed alternative, not a description of that implementation. A new covariance, mask, compression matrix, ordering, or finite-simulation correction invalidates the stored result. A changed mean cosmology does not when the forecast explicitly holds the covariance fixed. [Current covariance preparation](../../../../../cosmolike/generic_interface.cpp).

## 7. Tests that demonstrate the cache is correct

Cache validation should compare each changed-setting evaluation with a fresh process using the same final settings. Required sequences include cosmology A to B to A; independent changes of $`A_1,A_2,\eta_1,\eta_2,b_{\rm TA}`$ and velocity shear; photo-z changes; new source values at unchanged array addresses; larger and smaller bin counts; altered apertures and cuts; different worker counts; and two successive `initial_setup` calls with different models. Tests must reach every affected output, including mixed-bin permutations.

The recorded evidence should state which objects were reallocated, which were refilled, and which were reused, alongside the full prediction comparison. Equal predictions alone can hide a cache that recomputes everything. Lower allocation counts alone can hide stale physics. These two checks establish different properties and are both needed before attributing a runtime gain to persistent arrays.
