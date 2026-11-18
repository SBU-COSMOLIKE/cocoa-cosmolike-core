#ifndef COSMOLIKE_PHYSICS_WRAPPER_HPP
#define COSMOLIKE_PHYSICS_WRAPPER_HPP

#include <armadillo>
#include <memory>
#include <vector>
#include "physics_3pcf.h"
#include "physics_grid_3pcf.h"

namespace cosmolike_interface {

// Python retains this owner, whose deleter releases every immutable C
// table. A projection or covariance may borrow get() only while its owner
// is alive. No global cosmology is changed by constructing a second model.
class physics_owner_3pcf {
 public:
  explicit physics_owner_3pcf(const physics_input_3pcf& input);
  const physics_model_3pcf* get() const;

 private:
  std::unique_ptr<physics_model_3pcf, decltype(&physics_destroy_3pcf)> model;
};

class physics_grid_owner_3pcf {
 public:
  physics_grid_owner_3pcf(
    const std::shared_ptr<physics_owner_3pcf>& model,
    const physics_ia_3pcf& parameters,
    const int* bins,
    const physics_grid_settings_3pcf& settings
  );
  physics_grid_owner_3pcf(physics_grid_3pcf* prepared, int lmax);
  const physics_grid_3pcf* get() const;
  int lmax() const;

 private:
  std::unique_ptr<physics_grid_3pcf, decltype(&physics_grid_destroy_3pcf)> grid;
  int maximum_order;
};

std::shared_ptr<physics_owner_3pcf> physics_model_cpp_3pcf(
  const arma::Col<double>& k,
  const arma::Col<double>& p_linear0,
  const arma::Col<double>& z,
  const arma::Mat<double>& p_nonlinear,
  const arma::Col<double>& chi,
  const arma::Col<double>& dz_dchi,
  const arma::Col<double>& growth,
  const arma::Mat<double>& sources,
  double omega_m,
  double omega_de,
  double w0,
  double wa,
  double ns,
  double sigma8,
  const arma::Col<double>& los_z,
  int kernel_nz,
  bool source_readers,
  int table_boost,
  int fit_nz
);

arma::Mat<double> physics_points_cpp_3pcf(
  const physics_owner_3pcf& owner,
  const arma::Mat<double>& sides, // [point,3], h/Mpc or angular ell
  const arma::Col<double>& redshifts, // empty for projected queries
  const arma::Col<int>& bins, // [3] only for projected queries
  const physics_ia_3pcf& parameters,
  int operation, // 0=matter, 1=19 IA, 2=projected, 3=19 IA with shared shape
  int component,
  bool remove_alignment
);

arma::Cube<double> physics_multipoles_cpp_3pcf(
  const physics_owner_3pcf& owner,
  const arma::Col<double>& ell,
  const arma::Col<int>& bins,
  const physics_ia_3pcf& parameters,
  int lmax,
  int nmu,
  int component
);

arma::Col<double> physics_aperture_full_cpp_3pcf(
  const physics_owner_3pcf& owner,
  const arma::Col<double>& ell,
  const arma::Col<double>& dell,
  const arma::Col<double>& phi,
  const arma::Col<double>& dphi,
  const arma::Mat<double>& filters,
  const arma::Col<int>& bins,
  const physics_ia_3pcf& parameters
);

arma::Col<double> physics_grid_projected_cpp_3pcf(
  const physics_grid_owner_3pcf& owner,
  const arma::Mat<double>& sides,
  int component
);
std::vector<std::shared_ptr<physics_grid_owner_3pcf>>
physics_grids_cpp_3pcf(
  const physics_owner_3pcf& owner,
  const arma::Mat<int>& bins,
  const physics_ia_3pcf& parameters,
  const physics_grid_settings_3pcf& settings
);
arma::Col<double> physics_grid_aperture_cpp_3pcf(
  const physics_grid_owner_3pcf& owner,
  const arma::Col<double>& ell,
  const arma::Col<double>& dell,
  const arma::Col<double>& phi,
  const arma::Col<double>& dphi,
  const arma::Mat<double>& filters
);
arma::Col<double> physics_nla_power_cpp_3pcf(
  const physics_owner_3pcf& owner,
  const arma::Col<double>& ell,
  const arma::Col<int>& bins,
  const physics_ia_3pcf& parameters
);
arma::Cube<double> physics_grid_multipoles_cpp_3pcf(
  const physics_grid_owner_3pcf& owner,
  const arma::Col<double>& ell,
  int component
);

}
#endif
