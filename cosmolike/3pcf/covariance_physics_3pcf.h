#ifndef COSMOLIKE_COVARIANCE_PHYSICS_3PCF_H
#define COSMOLIKE_COVARIANCE_PHYSICS_3PCF_H

#include "physics_3pcf.h"

#ifdef __cplusplus
extern "C" {
#endif

// Fixed quadrature and observable definitions for the physical
// covariance. Every array is contiguous C order, and the engine borrows
// it for one call without copying it. Distances are in Mpc/h, masses in
// Msun/h, wavenumbers in h/Mpc and angles in radians. The matter model
// supplies the redshift nodes, the distances and the source windows.
// Each integration-weight array already contains its differential -
// d(ell), dr, dk, dM or dphi, as its name says - and none hides an
// extra Fourier factor. All node arrays increase.
typedef struct {
  int nell;                       // Fourier radial nodes
  int nwindow;                    // independent low-q survey-window nodes
  int nphi;                       // relative-angle nodes over [0,2*pi]
  int nr;                         // halo-offset and center-separation nodes
  int nprofile;                   // wider halo shear interpolation grid
  int nk;                         // linear-power variance quadrature nodes
  int nmass;                      // halo mass quadrature nodes
  int nfield;                     // ordered (source, aperture radius) labels
  int nrow;                       // requested covariance rows
  int ntheta;                     // separation quadrature nodes per xi row
  const double* ell;              // [nell], positive
  const double* dell;             // [nell], positive dell weights
  const double* window_ell;       // [nwindow], positive survey multipoles
  const double* window_dell;      // [nwindow], positive dq weights
  const double* phi;              // [nphi], relative angles
  const double* dphi;             // [nphi], positive dphi weights
  const double* radius;           // [nr], nonnegative angular radius
  const double* dr;               // [nr], positive dr weights
  const double* profile_radius;   // [nprofile], starts at zero
  const double* k;                // [nk], positive variance quadrature nodes
  const double* dk;               // [nk], positive dk weights
  const double* mass;             // [nmass], positive M200m
  const double* dmass;            // [nmass], positive dM weights
  const int* field_source;        // [nfield], zero-based source bin
  const double* field_radius;     // [nfield], aperture R; 0 for xi legs
  // Row kinds: Map2 and Map3 are the second and third moments of the
  // aperture mass; xi+ and xi- are the binned shear correlations.
  const int* row_kind;            // [nrow]: 0=Map2, 1=Map3, 2=xi+, 3=xi-
  const int* row_fields;          // [nrow,3], field labels; third unused 2pt
  const double* theta;            // [nrow,ntheta], xi separation nodes
  const double* theta_weight;     // [nrow,ntheta], normalized xi-bin weights
  const double* theta_edges;      // optional [nrow,2], exact xi annulus edges
  // nsource is the source-bin count of the physics model the call
  // receives.
  const double* noise;            // [nsource,nsource], white E and B noise
  double area;                    // solid angle, steradians
  int center_window;              // 1=circle, 2=square, with that area
  int large_area;                 // 1 uses large-area short-mode contractions
  int estimator;                  // only 0: full filters and map products
  int ia_model;                   // only 0: pure gravitational lensing
  int noise_mode;                 // 0=finite-band NN; 1=uniform-area xi NN
  double st_a;                    // Sheth-Tormen a, normally 0.707
  double st_p;                    // Sheth-Tormen p, normally 0.3
  double st_norm;                 // Sheth-Tormen norm, normally 0.3222
  double delta_c;                 // collapse threshold, normally 1.686
  int collapse_model;             // 0=const; 1=delta_c(1+.0123 log10 Om(z))
  int profile_upsample;           // >=1, fine profile intervals per input
  double concentration_a;         // Duffy M200m normalization, normally 10.14
  double concentration_b;         // mass exponent, normally -0.081
  double concentration_c;         // (1+z) exponent, normally -1.01
  double concentration_pivot;     // Msun/h, normally 2e12
  double baryon_strength;         // BiHalofit baryon-transfer parameter
  // Optional copied shared-core variance on these exact halo nodes.
  // With nhalo_z = 0 all pointers stay NULL and the k quadrature remains
  // active. Otherwise every array is required; no interpolation occurs.
  int nhalo_z;                    // redshifts represented by the snapshot
  const double* halo_z;            // [nhalo_z], must match queried redshifts
  const double* halo_mass;         // [nmass], must match mass above
  const double* halo_sigma2;       // [nhalo_z,nmass], total-matter variance
  const double* halo_dlnsigma_dlnm; // [nhalo_z,nmass], d ln sigma/d ln M
  double halo_reference_rho;       // density defining snapshot mass radii
} covariance_input_3pcf;

// Full physical covariance. terms has shape [6,nrow,nrow] and holds,
// in order: the Gaussian term, the BB term, the P*T one-halo cross
// term, the P*B term, the connected one-halo term, and the
// halo-abundance response. That response is the squeezed two-halo model
// alone; it is not a complete growth/dilation/tidal SSC. mean has shape
// [nrow] and excludes the additive noise mean of the quadratic rows.
// The same Fourier and radial rules define every term, so accuracy is
// established by refining each grid the caller passes in. The call
// allocates its workspaces serially and frees them before returning.
void covariance_physical_3pcf(
  const physics_model_3pcf* model, // immutable matter/source model
  const covariance_input_3pcf* input, // physical and numerical definitions
  double* mean,                // output [nrow]
  double* terms                // output [6,nrow,nrow]
);

// Independent halo-table entry point; the covariance engine calls it
// too. sigma2, dndm, bias and concentration have shape [nz,nmass].
// profile has shape [nz,nmass,nk_profile] and holds the normalized
// truncated-NFW transform, with u(k=0) = 1.
void covariance_halos_3pcf(
  const physics_model_3pcf* model, // linear power and cosmology
  const covariance_input_3pcf* input, // k/dk, M/dM and the halo prescription
  int nz,                      // redshift count
  const double* z,             // redshifts [nz]
  int nk_profile,              // profile wavenumber count
  const double* k_profile,     // [nk_profile], nonnegative h/Mpc
  double* sigma2,              // variance [nz,nmass]
  double* dndm,                // abundance [(Mpc/h)^-3 (Msun/h)^-1]
  double* bias,                // dimensionless linear halo bias
  double* concentration,       // M200m concentration [nz,nmass]
  double* profile              // normalized u [nz,nmass,nk_profile]
);

// Bin-averaged xi mean from the E/B angular spectra the caller
// evaluates on the quadrature nodes. sign = +1
// selects xi+, which uses the J0 kernel and the sum C_E + C_B;
// sign = -1 selects xi-, which uses J4 and the difference C_E - C_B.
// The weights on each separation bin sum to one. The additive noise
// bias is not removed.
void covariance_xi_from_power_3pcf(
  int nrow,                    // number of xi rows
  int nell,                    // Fourier quadrature nodes
  int ntheta,                  // separation samples per row
  const double* ell,           // multipoles [nell]
  const double* dell,          // dell weights [nell]
  const double* theta,         // radians [nrow,ntheta]
  const double* theta_weight,  // normalized weights [nrow,ntheta]
  const double* spectra_e,     // E angular spectra [nrow,nell]
  const double* spectra_b,     // B angular spectra [nrow,nell]
  const int* sign,             // +1 or -1 [nrow]
  double* out                  // bin-averaged xi [nrow]
);

#ifdef __cplusplus
}
#endif
#endif
