#ifndef COSMOLIKE_THREEPCF_CORE_BRIDGE_H
#define COSMOLIKE_THREEPCF_CORE_BRIDGE_H

#include <stdint.h>
#include "physics_3pcf.h"

#ifdef __cplusplus
extern "C" {
#endif

// Bridge between Cosmolike's global structs and the three-point (3PCF)
// physics model. The model itself (physics_3pcf.h) owns immutable
// copies of plain physical tables and knows nothing about the global
// structs; this bridge reads those structs, adapts units and layout at
// the boundary, and assembles the physics_input_3pcf that
// physics_create_3pcf consumes. Nothing here changes the model's
// internals.
//
// Two input tiers feed the assembled model:
//
//   struct-read tier - quantities an ordinary Cosmolike likelihood
//     already installs into the global structs: the nonlinear power
//     and the z = 0 linear power row (cosmology.lnP / lnPL), the
//     comoving distance and its redshift derivative (cosmology.chi),
//     the growth factor (cosmology.G), the matter density
//     (cosmology.Omega_m), the source densities
//     (redshift.shear_zdist_table) and the intrinsic-alignment (IA)
//     amplitudes (nuisance.ia).
//
//   supplied tier - quantities no global struct holds yet: the
//     line-of-sight nodes, the model resolution controls, the scalars
//     ns / sigma8 / w0 / wa (these stay supplied until a later stage
//     retires them into the structs), the pixel window, the paired
//     baryonic-feedback response, the shifted source densities when
//     photo-z shifts are active, and the IA prescription selectors.
//     Each setter below stores an owned copy, compares the incoming
//     values against the stored ones exactly (bitwise, never through a
//     rounded tolerance) and redraws the single bridge nonce when
//     anything changed.
//
// Cache invalidation:
// threepcf_bridge_refresh guards the assembled input with two tiers:
// an allocation tier comparing every table count as integers, and a
// refill tier comparing cosmology.random, redshift.random_shear,
// nuisance.random_photoz_shear, nuisance.random_ia, Ntable.random and
// the bridge nonce against the values snapshotted at the previous
// assembly. reset_threepcf_core_bridge (called by initial_setup in
// full-core builds, and by the standalone module's initial_setup
// binding) clears the supplied tier, the snapshot and the nonce, so a
// fresh setup can never read a previous model's state.
//
// Units at the boundary: the structs store log10 k with k in h/Mpc,
// ln P with P in (Mpc/h)^3, and chi in Mpc/h, which are already the
// model's units, so the only numeric conversions are pow(10, log10 k),
// exp(ln P), the dz/dchi finite difference, the D = G * a growth
// conversion and the IA amplitude translation. No c/H0 (coverH0)
// rescaling enters: the power and distance structs never store those
// units.
//
// Threading: every function here mutates file-scope state and must run
// outside parallel regions, from one thread, exactly like the set_
// installers of generic_interface.cpp.

// ---------------------------------------------------------------------------
// Supplied tier. Each setter validates its input (NaN and size checks
// abort through log_fatal), stores an owned copy, and redraws the
// bridge nonce only when a stored value actually changed, so an MCMC
// that resends identical arrays never invalidates the cached model.
// ---------------------------------------------------------------------------

// Background scalars the structs do not hold: the spectral index ns,
// the z = 0 amplitude sigma8, and the dark-energy equation of state
// w(a) = w0 + wa (1 - a). The model refuses to build before this
// setter has run at least once: these numbers have no safe default.
void threepcf_bridge_set_scalars(
  double ns,     // primordial spectral index
  double sigma8, // linear fluctuation amplitude at z = 0
  double w0,     // dark-energy equation of state today
  double wa      // its evolution in w(a) = w0 + wa (1 - a)
);

// Line-of-sight projection nodes (physics_3pcf.h documents their
// contract: positive, strictly increasing, at least two). The model
// refuses to build before this setter has run.
void threepcf_bridge_set_los(
  const double* z, // node redshifts [n]
  int n            // node count
);

// Model preparation resolution (physics_3pcf.h documents each knob).
// The defaults match the Python constructor's: kernel_nz = 101,
// source_readers = 0, table_boost = 8, fit_nz = 33.
void threepcf_bridge_set_resolution(
  int kernel_nz,      // internal lensing-kernel redshift resolution
  int source_readers, // 1 keeps cubic source reads for comparisons
  int table_boost,    // dense-grid refinement factor; zero means eight
  int fit_nz          // uniform BiHalofit anchor count; zero means 33
);

// Shifted source densities on their own redshift axis. The project's
// photo-z convention (des3pcf.inputs.shifted_sources) interpolates
// n(z - delta z), pins the observer sample and renormalizes with
// NumPy's pairwise-summed trapezoid; that reduction order is a NumPy
// implementation detail, so the bridge never reproduces it in C.
// Python applies the shifts and supplies the result here instead.
// While a supplied set is stored, the model uses it and ignores
// redshift.shear_zdist_table; passing (NULL, NULL, 0, 0) clears the
// override, and the struct-read path then requires every stored
// photo-z shift nuisance.photoz[0][0][i] to be exactly zero.
void threepcf_bridge_set_sources(
  const double* source_z, // source redshifts [nsource], increasing
  const double* sources,  // densities per unit z [nbin, nsource]
  int nbin,               // tomographic source bins
  int nsource             // source redshift nodes
);

// Angular pixel window W(ell) at integer ell = 0 .. n - 1
// (physics_window_value_3pcf documents the read convention). Passing
// (NULL, 0) clears it; no window is a valid state.
void threepcf_bridge_set_pixel_window(
  const double* window, // W(ell) samples [n], or NULL
  int n                 // sample count, or zero
);

// Paired hydro/DMO feedback response tables (feedback_3pcf.h documents
// every axis) plus the shared log-response amplitude the downstream IA
// dictionary carries. Passing NULL axes with zero counts clears the
// response. amplitude only travels into
// threepcf_bridge_ia_from_core's output; the model tables themselves
// stay dark-matter-only, as the module requires.
void threepcf_bridge_set_feedback(
  const double* z,              // redshift nodes [nz]
  int nz,                       // redshift count
  const double* logk,           // log(k/[h/Mpc]) nodes [nk]
  int nk,                       // wave-number count
  const double* logu,           // log(u) nodes [nu]
  int nu,                       // side-ratio count
  const double* v,              // v nodes [nv]
  int nv,                       // shape-coordinate count
  const double* log_power,      // ln R_P [nz, nk]
  const double* log_bispectrum, // ln R_B [nz, nk, nu, nv]
  int boost,                    // dense intervals per loaded interval
  int clamp,                    // 0 refuses out-of-domain; 1 clamps
  double amplitude              // shared log-response amplitude
);

// Install an absolute matter B table and the matter-fraction background
// on the core nonlinear-power redshift axis. Copies every input. sigma8
// retains the existing linear-power normalization; ns/w0/wa are unused
// while this table is active. A NULL baseline clears only this override.
void threepcf_bridge_set_baseline_bispectrum(
  const baseline_input_3pcf* baseline, // supplied absolute B, or NULL
  const double* omega_m_z, // Omega_m(z) on the core background z [nz]
  int nz, // background count; zero when clearing
  double sigma8 // positive linear-power normalization; unused when clearing
);

// IA prescription selectors of physics_ia_3pcf (physics_3pcf.h
// documents each integer). Defaults match physics_default_ia_3pcf:
// power_mode = 1, ct_mode = 0, suppress_only = 0.
void threepcf_bridge_set_prescriptions(
  int power_mode,   // 0 linear D^2; 1 nonlinear(z=0) D^2; 2 full table
  int ct_mode,      // 0 physical Ct = C1*ct (fastnc); 1 Ct = ct
  int suppress_only // cap the baryon transfer at one before scaling
);

// ---------------------------------------------------------------------------
// Struct-read conversions and the cached-input factory.
// ---------------------------------------------------------------------------

// Translate the IA amplitudes the ordinary core stores into the
// module's fastnc convention. This is the single conversion site: the
// amplitude mapping, the pivot mapping and every guard live here and
// nowhere else. threepcf_core_bridge.c carries the didactic
// derivation; in brief, both conventions use
// C_1 = -A_1 c1rho Omega_m / D times a redshift power law, the core
// fixes c1rho = nuisance.c1rhocrit_ia = 0.01389 with pivot
// 1 + z_pivot = nuisance.oneplusz0_ia = 1.62, the module defaults to
// c1rho = 0.0134 with its own pivot, and matching the physical C_1
// divides each incoming amplitude by
// (c1rho / 0.01389) * ((1 + 0.62)/(1 + z_pivot))^eta.
//
// Guards: requires nuisance.IA == IA_REDSHIFT_EVOLUTION and
// nuisance.IA_MODEL == IA_MODEL_TATT. Under IA_REDSHIFT_BINNING the
// slot nuisance.ia[1][2] holds the per-bin tidal-torque amplitude A_2
// of shear bin 2, not the velocity-shear amplitude C_t, so that mode
// is refused rather than silently misread. A leftover per-bin b_TA
// table whose entries differ is refused as well: one scalar b_TA
// cannot represent it.
void threepcf_bridge_ia_from_core(
  physics_ia_3pcf* parameters // receives the translated parameters
);

// Assemble the model input from the two tiers when the two-tier guard
// says the previous assembly is out of date. Returns NULL when the
// stored snapshot still matches every count and cache key: the caller
// keeps its current model. Returns the assembled input otherwise; the
// bridge then owns the pointed-to scratch arrays until the next
// refresh or reset, has snapshotted the new keys and has advanced the
// build generation. The caller must create the new model from the
// returned input before calling any bridge function again, and must do
// so outside parallel regions (physics_create_3pcf's contract).
const physics_input_3pcf* threepcf_bridge_refresh(void);

// Build generation: starts at zero, advances on every non-NULL
// threepcf_bridge_refresh return. Lets callers and tests observe
// rebuilds without comparing pointers.
uint64_t threepcf_bridge_generation(void);

// Clear every piece of bridge state: the supplied-tier copies and
// scalars (back to their defaults), the assembled scratch, the guard
// snapshot and the nonce. initial_setup calls this in full-core
// builds through a weak reference, so a setup that never loads the
// bridge is unaffected; the standalone 3PCF module calls it from its
// own initial_setup binding. Without this call a second setup in the
// same process would reuse the previous model through a stale
// snapshot - the classic missed-reset trap.
void reset_threepcf_core_bridge(void);

// ---------------------------------------------------------------------------
// Core struct installers. The standalone 3PCF module compiles neither
// generic_interface.cpp nor the 2D/3D kernels, so the set_ functions
// an ordinary likelihood uses to fill the global structs do not exist
// there. These installers are their equivalents for exactly the
// fields the bridge reads, with the same units, the same fdiff change
// detection and the same cache-key bumps (a process-unique counter
// stands in for the C++ RandomNumber singleton). In a full-core build
// the original installers run instead and these are simply unused.
// Divergences from the originals are deliberate and documented in the
// .c file: no chi bucket index, no growth-rate row, no n(z) spline
// warm-up - each belongs to kernels this module does not compile and
// the bridge never reads.
// ---------------------------------------------------------------------------

// set_cosmological_parameters equivalent: Omega_m (massive neutrinos
// included), Omega_b, H0 in km/s/Mpc and omega_nu h^2.
void threepcf_core_set_cosmological_parameters(
  double omega_matter, // Omega_m today
  double omega_baryon, // Omega_b today
  double hubble,       // H0 (km/s/Mpc)
  double omega_nu_h2   // omega_nu h^2 of the massive neutrinos
);

// set_distances equivalent: chi(z) in Mpc/h on its redshift grid.
void threepcf_core_set_distances(
  const double* z,   // redshift grid [n], at least five nodes
  const double* chi, // comoving distance at z in Mpc/h [n]
  int n              // node count
);

// set_growth equivalent: G(z) with D = G * a on its redshift grid.
// The growth-rate row the full core derives (set_growth's five-point
// stencil) is left at zero: nothing in this module reads it.
void threepcf_core_set_growth(
  const double* z, // redshift grid [n], at least five nodes
  const double* G, // growth G at z, D = G * a [n]
  int n            // node count
);

// set_linear_power_spectrum equivalent: ln P_lin on (log10 k, z),
// flattened as lnP[i*nz + j] = ln P(k_i, z_j), P in (Mpc/h)^3.
void threepcf_core_set_linear_power_spectrum(
  const double* log10k, // log10 k grid [nk], k in h/Mpc, uniform
  int nk,               // wave-number count
  const double* z,      // redshift grid [nz]
  int nz,               // redshift count
  const double* lnP     // flattened ln P [nk*nz]
);

// set_non_linear_power_spectrum equivalent, same layout and units.
void threepcf_core_set_non_linear_power_spectrum(
  const double* log10k, // log10 k grid [nk], k in h/Mpc, uniform
  int nk,               // wave-number count
  const double* z,      // redshift grid [nz]
  int nz,               // redshift count
  const double* lnP     // flattened ln P [nk*nz]
);

// set_source_sample_size equivalent: declare the source tomography.
void threepcf_core_set_source_sample_size(
  int ntomo // number of source tomographic bins
);

// set_source_sample equivalent. table is row-major [nzbins][ntomo+1]:
// column 0 holds the redshift axis, column k+1 holds n(z) of bin k,
// exactly the file layout read_nz_sample loads.
//
// This installer owns redshift.shear_zdist_table: it allocates the
// table with ntomo + 1 rows and records the bin count it installed.
// The model refresh refuses to read the table while that count
// differs from redshift.shear_nbin, so a set_source_sample_size call
// must be followed by a matching set_source_sample before the next
// model. The recorded count describes only tables this installer
// wrote; a future full-core route that fills the table elsewhere must
// bring its own allocation metadata.
void threepcf_core_set_source_sample(
  const double* table, // row-major n(z) table [nzbins][ntomo+1]
  int nzbins,          // redshift rows
  int ntomo            // source bins; must equal redshift.shear_nbin
);

// set_nuisance_shear_photoz equivalent: per-bin source photo-z shifts.
void threepcf_core_set_nuisance_shear_photoz(
  const double* shifts, // per-bin shifts [n]
  int n                 // must equal redshift.shear_nbin
);

// init_IA equivalent (IA_code fixed at 0): the IA model registry
// flags nuisance.IA_MODEL and nuisance.IA. The ia[][] rows change
// meaning with nuisance.IA, so a change of the redshift mode resets
// every IA amplitude to zero; install them again afterwards. A change
// of IA_MODEL alone keeps them.
void threepcf_core_init_ia(
  int ia_model,             // 0 = IA_MODEL_NLA, 1 = IA_MODEL_TATT
  int ia_redshift_evolution // NO_IA .. IA_REDSHIFT_EVOLUTION (IA.h)
);

// set_nuisance_IA equivalent: amplitudes for the active nuisance.IA
// mode (structs.h documents the ia[][] registry rows).
void threepcf_core_set_nuisance_ia(
  const double* A1,  // tidal alignment [n]: binned, one per source
                     // bin; power law, [0] amplitude, [1] z exponent
  const double* A2,  // tidal torque [n]: binned, one per source bin;
                     // power law, [0] amplitude, [1] z exponent
  const double* BTA, // density weighting [n]: binned, one per source
                     // bin; power law, entry [0] only is read
  int n              // length of every array: at least shear_nbin in
                     // both modes, and at least 2 under the power law
);

// set_nuisance_IA_ct equivalent: the velocity-shear amplitude C_t in
// nuisance.ia[1][2]. Refused (log_fatal, exit) unless nuisance.IA =
// IA_REDSHIFT_EVOLUTION: in binned mode that slot is the A2 amplitude
// of source bin 2.
void threepcf_core_set_nuisance_ia_ct(
  double ct // velocity-shear alignment amplitude
);

// initial_setup equivalent for the standalone module: runs the
// reset_*_struct family of structs.c and then
// reset_threepcf_core_bridge. State the full core resets beyond the
// structs (baryon scenarios, Pdelta dispatch, FAST-PT origins, point
// masses, CMB products) does not exist in this module.
void threepcf_core_initial_setup(void);

#ifdef __cplusplus
}
#endif
#endif
