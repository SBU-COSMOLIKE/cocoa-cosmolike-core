#ifndef COSMOLIKE_COVARIANCE_PHYSICS_3PCF_H
#define COSMOLIKE_COVARIANCE_PHYSICS_3PCF_H

#include "physics_3pcf.h"

#ifdef __cplusplus
extern "C" {
#endif

// Fixed quadrature and observable definitions for the physical covariance.
// Every array is contiguous C order and borrowed for one call. Distances
// use Mpc/h, masses Msun/h, wavenumbers h/Mpc and angles radians. The
// matter model supplies the redshift nodes, distances and source windows.
// Integration weights contain d(ell), dr, dk, dM or dphi, as named; none
// contains an unmentioned Fourier factor. All node arrays are increasing.
typedef struct {
  int nell;                    // Fourier radial nodes
  int nwindow;                 // independent low-q survey-window nodes
  int nphi;                    // relative-angle nodes over [0,2*pi]
  int nr;                      // halo-offset and center-separation nodes
  int nprofile;                // wider halo shear interpolation grid
  int nk;                      // linear-power variance quadrature nodes
  int nmass;                   // halo mass quadrature nodes
  int nfield;                  // ordered (source bin, aperture radius) labels
  int nrow;                    // requested covariance rows
  int ntheta;                  // separation quadrature nodes per xi row
  const double* ell;           // [nell], positive
  const double* dell;          // [nell], positive dell weights
  const double* window_ell;     // [nwindow], positive survey multipoles
  const double* window_dell;    // [nwindow], positive dq weights
  const double* phi;           // [nphi], relative angles
  const double* dphi;          // [nphi], positive dphi weights
  const double* radius;        // [nr], nonnegative angular radius
  const double* dr;            // [nr], positive dr weights
  const double* profile_radius; // [nprofile], starts at zero
  const double* k;             // [nk], positive variance quadrature nodes
  const double* dk;            // [nk], positive dk weights
  const double* mass;          // [nmass], positive M200m
  const double* dmass;         // [nmass], positive dM weights
  const int* field_source;     // [nfield], zero-based source bin
  const double* field_radius; // [nfield], aperture R; zero for xi shear legs
  const int* row_kind;         // [nrow]: 0=Map2, 1=Map3, 2=xi+, 3=xi-
  const int* row_fields;       // [nrow,3], field labels; third unused for 2pt
  const double* theta;         // [nrow,ntheta], xi separation nodes
  const double* theta_weight;  // [nrow,ntheta], normalized xi-bin weights
  const double* theta_edges;   // optional [nrow,2], exact xi annulus edges
  const double* noise;         // [nsource,nsource], white E and B noise power
  double area;                // solid angle, steradians
  int center_window;          // 1=circle, 2=square, with the specified area
  int large_area;             // 1 uses large-area short-mode contractions
  int estimator;              // only 0: full filters and map products
  int ia_model;               // only 0: pure gravitational lensing
  int noise_mode;             // 0=finite-band NN; 1=uniform-area xi NN
  double st_a;                // Sheth-Tormen a, normally 0.707
  double st_p;                // Sheth-Tormen p, normally 0.3
  double st_norm;             // Sheth-Tormen normalization, normally 0.3222
  double delta_c;             // collapse threshold, normally 1.686
  int collapse_model;         // 0=constant; 1=delta_c*(1+.0123 log10 Om(z))
  int profile_upsample;       // >=1, fine uniform profile intervals per input
  double concentration_a;     // Duffy M200m normalization, normally 10.14
  double concentration_b;     // mass exponent, normally -0.081
  double concentration_c;     // (1+z) exponent, normally -1.01
  double concentration_pivot; // Msun/h, normally 2e12
  double baryon_strength;     // BiHalofit baryon-transfer parameter
} covariance_input_3pcf;

// Terms [6,nrow,nrow]: Gaussian, BB, P*T_1h, P*B, connected one-halo,
// selected halo-abundance response. The response term is the selected
// squeezed two-halo model, not a complete growth/dilation/tidal SSC.
// Mean [nrow] excludes the additive noise mean. The same Fourier/radial
// rules define all terms; refine each supplied grid to establish accuracy.
// Workspaces are allocated serially within the call and freed on return.
void covariance_physical_3pcf(
  const physics_model_3pcf* model, // immutable matter/source model
  const covariance_input_3pcf* input, // physical and numerical definitions
  double* mean,                // output [nrow]
  double* terms                // output [6,nrow,nrow]
);

// Independent halo-table entry point, also used by the covariance engine.
// sigma2, dndm, bias, concentration have shape [nz,nmass]. profile has
// shape [nz,nmass,nk_profile], with normalized truncated-NFW u(k=0)=1.
void covariance_halos_3pcf(
  const physics_model_3pcf* model, // linear power and cosmology
  const covariance_input_3pcf* input, // k/dk and M/dM, named halo prescription
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

// Bin-averaged xi mean from supplied E/B spectra. sign=+1 selects xi+
// with J0 and CE+CB; sign=-1 selects xi- with J4 and CE-CB. The weights
// on each separation bin sum to one. Additive noise bias is not removed.
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
