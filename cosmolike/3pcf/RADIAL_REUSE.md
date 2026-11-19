# Reusing radial kernels without changing their scientific meaning

The ordinary data vector and the three-point model describe the same flat-space
lensing geometry. Their stored arrays are not interchangeable. A direct call to
`W_kappa` or `W_source` would change units, source normalization, sampling and
state ownership. This review identifies the conversions and the checks needed
before replacing three-point preparation. It does not claim numerical agreement
between those two preparations.

## What each kernel represents

Let `L = 2997.92458 Mpc/h`, `chi_h` be distance in Mpc/h, and
`chi_core = chi_h/L`. For a normalized source density per redshift, define

    g(z) = integral_z^infinity dz_s n(z_s) [1 - chi(z)/chi(z_s)].

The physical lensing kernel is

    W_lens(z) = (3/2) Omega_m0 chi_h (1+z) g(z) / L^2.

The ordinary `radial_weights.c::W_kappa` returns the dimensionless version,
`L*W_lens`. The three-point `kernel_lens` instead stores the factor before
`chi_h*(1+z)`. Ignoring the constant convention discussed below, the conversion
would therefore be

    kernel_lens = W_kappa / [L chi_h (1+z)].

This expression is for positive distance. It is not a prescription for dividing
by zero at the observer. The three-point integration nodes are positive.

The physical source kernel is `n(z) dz/dchi_h`. The ordinary `W_source` returns
`n(z) E(z)`, so divide by `L` to obtain physical inverse-distance units. The
three-point shape arrays additionally divide by `chi_h*(1+z)`, which the
combined field kernel restores. Their finite-grid normalization factors must
also be accounted for; equal units alone do not make the arrays equal.

The constants differ slightly: three-point lensing follows the upstream
`c=299792 km/s`; ordinary `coverH0` uses `c=299792.458 km/s`. The relative
lensing-prefactor difference is about 3.06e-6, and three such factors give about
9.17e-6 for a pure lensing bispectrum. This is a deterministic convention
change, not interpolation error. It should have its own recorded comparison
rather than disappear inside a kernel replacement.

## Source normalization and photo-z shifts

Three-point preparation receives copied source samples. It normalizes lensing
with a trapezoidal integral over the source-redshift grid. The NLA baseline
normalizes `n(z) dz/dchi` on its kernel-distance grid, whereas the quadratic TATT
residual normalizes it on the source-distance grid. These two normalizations
coincide in the continuum but need not agree at finite resolution. They preserve
the upstream estimator's numerical convention.

The ordinary reader `nz_source_photoz` applies the selected source reconstruction
and nuisance shifts inside the shared source machinery. A bridge cannot pass
already-shifted three-point sources to that machinery and apply the same shift
again. Conversely, copying an unshifted source table misses the shift. The
three-point bridge's struct-source route currently refuses nonzero shifts;
its explicit-source route receives already-prepared densities.

A future shared preparation must record whether each density is raw or shifted,
which reconstruction produced it, its support and its normalization measure.
The same shifted function must be supplied to both algorithms before comparing
lensing or intrinsic-alignment kernels.

## Different integration coordinates and endpoint rules

`redshift_spline.c::g_tomo` integrates uniform scale-factor samples using
cumulative integrals of `n(z)/a^2` and `n(z)/(a^2 chi)`. Its fine count is
`60*(1+abs(high_def_integration))*(N_a-1)+1`; it stores coarse scale-factor
samples and reads them linearly. Its upper preparation limit is `a=0.999999`,
and its reader returns zero in the last coarse interval near the observer.
The lower limit follows the shifted source support.

Three-point lensing integrates the explicitly supplied redshift samples. Its
kernel grid is uniform in z, from the first LOS node to the background endpoint.
Production prepares natural cubic splines in z; upstream comparisons retain
not-a-knot interpolation in distance. Both eventually provide copied weights
for fixed LOS quadrature. A replacement must compare the low-redshift cutoff,
shifted upper support and every piecewise-grid join, not just interior values.

## State ownership and threading

Ordinary kernel tables are mutable global caches. Their identities include
cosmology, source distributions, source-bin count, photo-z nuisance parameters
and sampling settings. Three-point models are immutable, and a notebook can
retain an old model while preparing a new cosmology.

Any reuse must warm shared tables serially, copy their samples before releasing
shared-state protection, and retain only private immutable arrays for triangle
workers. Calling a lazy global kernel directly from those workers would change
the present ownership contract. Linking the standalone three-point extension to
`redshift_spline.c` and `radial_weights.c` also introduces dependencies that are
absent from its current build; that is a separate integration change.

## Recommended next implementation and acceptance checks

The first useful shared operation is the lensing-efficiency integral over
explicit arrays, rather than a call through global survey state. The existing
three-point nodal trapezoid can be expressed as two suffix sums:

    P_j = sum_(i>=j) w_i n_i,
    Q_j = sum_(i>=j) w_i n_i/chi_i,
    g(chi) = P_j - chi Q_j,

where `w_i` are the original full-grid trapezoid weights and `j` is the first
source node behind the lens. Omit the observer from `Q`: its source density
vanishes, but explicitly forming `0/chi(0)` would divide by zero. This preserves the existing nodal cutoff by
algebra; it does not replace the crossed cell with a different partial-cell
integral. Monotonic queries permit a running index. Summation order and
cancellation near the source endpoint still require numerical checks. A pure
helper could serve more than one module without importing mutable caches.

Before selecting any shared radial implementation:

1. Compare normalized source functions, their support and the actual Jacobian
   after zero, positive and negative photo-z shifts. Include independent source
   grids, midpoint grids and piecewise-uniform grids.
2. Convert units and expose the speed-of-light convention explicitly. Compare
   lensing, NLA and quadratic-TATT shape kernels separately, including their
   normalization integrals.
3. Vary source, kernel and LOS counts independently. Test the observer cutoff,
   the source endpoint and grid joins. Retain the existing accuracy controls.
4. Propagate differences to GGG, ordered GGI/GII/III terms, NLA and quadratic
   TATT projections, aperture means and all six covariance families. Do not
   infer projection accuracy from a pointwise relative error near a kernel zero.
5. Check A/B/A updates, identical inputs, reset, retained old owners and one/six
   OpenMP workers with BLAS fixed to one. Assess survey-weighted mean and Fisher
   differences before changing the default.

The current conclusion is that radial reuse is possible, but replacing the
existing calls is not a mechanical deduplication. Radial code remains unchanged
until these convention and convergence checks are completed.
