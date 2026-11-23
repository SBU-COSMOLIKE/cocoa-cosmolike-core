#ifndef COSMOLIKE_PHYSICS_WRAPPER_HPP
#define COSMOLIKE_PHYSICS_WRAPPER_HPP

#include <armadillo>
#include <memory>
#include <vector>
#include "physics_3pcf.h"
#include "physics_grid_3pcf.h"

namespace cosmolike_interface {

// Owned copies of the paired hydro/DMO response tables for the notebook
// route. The axes follow feedback_input_3pcf in feedback_3pcf.h: for
// sorted triangle sides a <= b <= c, r = b, u = a/b and v = (c - b)/a.
struct physics_feedback_tables_cpp_3pcf {
  arma::Col<double> z; // redshift nodes of both response tables
  arma::Col<double> logk; // log wavenumber nodes, k in h/Mpc
  arma::Col<double> logu; // log of the side ratio u = smallest/middle
  arma::Col<double> v; // triangle-shape coordinate (largest-middle)/smallest
  arma::Col<double> log_power; // flattened [nz,nk], k varies fastest
  arma::Col<double> log_bispectrum; // flattened [nz,nk,nu,nv], v fastest
  int boost = 1; // dense-interval multiplier; one keeps the input grid
  int clamp = 0; // 0 refuses out-of-domain queries; 1 holds edge values
};

// A copied absolute matter B table, distinct from a feedback response.
// Units, coordinate order and positivity are defined in baseline_3pcf.h.
struct physics_baseline_table_cpp_3pcf {
  arma::Col<double> z; // uniform redshifts
  arma::Col<double> logk; // uniform natural log middle side, h/Mpc
  arma::Col<double> logu; // uniform natural log smallest/middle
  arma::Col<double> v; // uniform (largest-middle)/smallest
  arma::Col<double> log_bispectrum; // ln B [nz,nk,nu,nv], v fastest
  int boost = 1; // dense intervals per input interval
  int clamp = 0; // reject out-of-domain queries by default
};

// Copied legacy smoothing snapshots for the notebook constructor: at
// each anchor redshift, rnl holds the radius R_NL at which the linear
// density variance, smoothed with a Gaussian filter, reaches one, and
// neff holds n_eff, the effective slope of the linear power spectrum
// at that radius. The anchor redshifts are piecewise uniform - a few
// consecutive uniform stretches, so a reader finds a segment by index
// arithmetic instead of a search. The adapters validate these arrays,
// but the C constructor ignores the field (physics_3pcf.h marks
// variance a retained ABI field).
struct physics_variance_tables_cpp_3pcf {
  arma::Col<double> z; // piecewise-uniform fit redshifts
  arma::Col<double> rnl; // nonlinear radii R_NL, Mpc/h
  arma::Col<double> neff; // effective spectral indices n_eff
  arma::Col<double> sigma8; // top-hat rms at R = 8 Mpc/h
};

// Python keeps this owner alive; its deleter releases every immutable C
// table of the model. A projection or covariance may borrow get() only
// while the owner lives. Constructing a second model changes no global
// cosmology state.
class physics_owner_3pcf {
 public:
  explicit physics_owner_3pcf(
    // validated arrays and scalars copied into this owner
    const physics_input_3pcf& input
  );
  const physics_model_3pcf* get() const;

 private:
  // the owned C model; the deleter physics_destroy_3pcf releases it
  std::unique_ptr<physics_model_3pcf, decltype(&physics_destroy_3pcf)> model;
};

// Owner of one prepared projection-grid table (physics_grid_3pcf.h).
class physics_grid_owner_3pcf {
 public:
  physics_grid_owner_3pcf(
    // physical model retained while the grid is prepared
    const std::shared_ptr<physics_owner_3pcf>& model,
    // intrinsic-alignment and feedback parameters
    const physics_ia_3pcf& parameters,
    const int* bins, // ordered source-bin labels for the requested correlation
    // physical-grid bounds, node counts and angular cutoffs
    const physics_grid_settings_3pcf& settings
  );

  // This constructor adopts ownership of a C table that the batch
  // factory physics_grid_create_many_3pcf already built, and the owner
  // will destroy it. The public C interface exposes no reader for the
  // table's Legendre cutoff, so the caller passes lmax alongside.
  physics_grid_owner_3pcf(
    // prepared C grid whose ownership is transferred here
    physics_grid_3pcf* prepared,
    int lmax // largest retained Legendre degree L
  );
  const physics_grid_3pcf* get() const;
  int lmax() const;

 private:
  // the owned C table; the deleter physics_grid_destroy_3pcf releases it
  std::unique_ptr<physics_grid_3pcf, decltype(&physics_grid_destroy_3pcf)> grid;
  int maximum_order; // largest Legendre degree L stored in the table
};

std::shared_ptr<physics_owner_3pcf> physics_model_cpp_3pcf(
  const arma::Col<double>& k, // wavenumbers, h/Mpc
  const arma::Col<double>& p_linear0, // linear P(k, z=0), (Mpc/h)^3
  const arma::Col<double>& z, // redshift grid, includes z = 0
  const arma::Mat<double>& p_nonlinear, // [nz,nk] nonlinear P, (Mpc/h)^3
  const arma::Col<double>& chi, // comoving distance at each z, Mpc/h
  const arma::Col<double>& dz_dchi, // dz/dchi at each z, h/Mpc
  const arma::Col<double>& growth, // linear growth factor D, one at z = 0
  const arma::Mat<double>& sources, // [nbin,nsource] densities per unit z
  double omega_m, // matter density parameter today
  double omega_de, // dark energy density parameter today
  double w0, // dark energy equation of state today
  double wa, // its evolution in w(a) = w0 + wa (1 - a)
  double ns, // scalar spectral index
  double sigma8, // amplitude of matter fluctuations at z = 0
  const arma::Col<double>& los_z, // line-of-sight nodes, increasing
  int kernel_nz, // radial-kernel resolution (physics_3pcf.h)
  bool source_readers, // keep cubic source reads for comparisons
  int table_boost, // dense-grid factor; zero selects the default
  int fit_nz, // coefficient anchors; zero selects the default
  const arma::Col<double>& source_z = arma::Col<double>(), // optional axis
  const physics_feedback_tables_cpp_3pcf* feedback = nullptr, // optional
  const arma::Col<double>* pixel_window = nullptr, // W(ell) at integer ell
  const physics_baseline_table_cpp_3pcf* baseline = nullptr, // absolute B
  const arma::Col<double>* omega_m_z = nullptr, // required with baseline
  const physics_variance_tables_cpp_3pcf* variance = nullptr, // fit anchors
  const arma::Col<double>* omega_m_redshift = nullptr // optional fraction axis
);

arma::Mat<double> physics_points_cpp_3pcf(
  const physics_owner_3pcf& owner, // owner of the immutable physical model
  const arma::Mat<double>& sides, // [point,3], h/Mpc or angular ell
  const arma::Col<double>& redshifts, // empty for projected queries
  const arma::Col<int>& bins, // [3] only for projected queries
  // intrinsic-alignment and feedback parameters
  const physics_ia_3pcf& parameters,
  // operation 4 is the baryon-feedback bispectrum response; B names the
  // bispectrum there, not the B shape mode of the IA entries
  int operation, // 0=matter, 1=IA, 2=projected, 3=shared-shape IA, 4=response
  int component, // matter-fit or projected split selected by operation
  bool remove_alignment // omit the linear-alignment baseline from IA outputs
);

arma::Col<double> physics_feedback_power_cpp_3pcf(
  const physics_owner_3pcf& owner, // owner of the immutable physical model
  const arma::Col<double>& k, // wavenumbers, h/Mpc
  const arma::Col<double>& z, // one redshift per wavenumber
  double amplitude // shared log-response amplitude; zero is DMO
);

arma::Cube<double> physics_multipoles_cpp_3pcf(
  const physics_owner_3pcf& owner, // owner of the immutable physical model
  const arma::Col<double>& ell, // Fourier sides of the b_L output grid
  const arma::Col<int>& bins, // [3] ordered source bins
  // intrinsic-alignment and feedback parameters
  const physics_ia_3pcf& parameters,
  int lmax, // largest Legendre degree L
  int nmu, // mu quadrature nodes per Legendre integral
  int component // total=0, NLA baseline=1, residual IA=2
);

arma::Col<double> physics_aperture_full_cpp_3pcf(
  const physics_owner_3pcf& owner, // owner of the immutable physical model
  const arma::Col<double>& ell, // Fourier quadrature nodes
  const arma::Col<double>& dell, // their quadrature weights
  const arma::Col<double>& phi, // angular quadrature nodes, radians
  const arma::Col<double>& dphi, // their quadrature weights, radians
  const arma::Mat<double>& filters, // [nfilter,3] aperture radii, radians
  const arma::Col<int>& bins, // [3] ordered source bins
  // intrinsic-alignment and feedback parameters
  const physics_ia_3pcf& parameters
);

arma::Col<double> physics_grid_projected_cpp_3pcf(
  // owner of the immutable projected-bispectrum grid
  const physics_grid_owner_3pcf& owner,
  const arma::Mat<double>& sides, // [point,3] multipole triangles
  int component // total=0, NLA baseline=1, residual IA=2
);

std::vector<std::shared_ptr<physics_grid_owner_3pcf>>
physics_grids_cpp_3pcf(
  const physics_owner_3pcf& owner, // owner of the immutable physical model
  const arma::Mat<int>& bins, // [ntriplet,3] ordered source triplets
  // intrinsic-alignment and feedback parameters
  const physics_ia_3pcf& parameters,
  // physical-grid bounds, node counts and angular cutoffs
  const physics_grid_settings_3pcf& settings
);

arma::Col<double> physics_grid_aperture_cpp_3pcf(
  // owner of the immutable projected-bispectrum grid
  const physics_grid_owner_3pcf& owner,
  const arma::Col<double>& ell, // Fourier quadrature nodes
  const arma::Col<double>& dell, // their quadrature weights
  const arma::Col<double>& phi, // angular quadrature nodes, radians
  const arma::Col<double>& dphi, // their quadrature weights, radians
  const arma::Mat<double>& filters // [nfilter,3] aperture radii, radians
);

arma::Col<double> physics_nla_power_cpp_3pcf(
  const physics_owner_3pcf& owner, // owner of the immutable physical model
  const arma::Col<double>& ell, // output multipoles of the shear C_ell
  const arma::Col<int>& bins, // [2] source bins of the power spectrum
  // intrinsic-alignment and feedback parameters
  const physics_ia_3pcf& parameters
);

arma::Cube<double> physics_grid_multipoles_cpp_3pcf(
  // owner of the immutable projected-bispectrum grid
  const physics_grid_owner_3pcf& owner,
  const arma::Col<double>& ell, // Fourier sides of the b_L output grid
  int component // total=0, NLA baseline=1, residual IA=2
);

}
#endif
