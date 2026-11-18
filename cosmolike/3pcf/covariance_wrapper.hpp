#ifndef COSMOLIKE_COVARIANCE_WRAPPER_3PCF_HPP
#define COSMOLIKE_COVARIANCE_WRAPPER_3PCF_HPP

#include <armadillo>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "covariance_physics_3pcf.h"

namespace cosmolike_interface {

// Notebook calculations own all numerical input buffers. Each Armadillo
// column below stores a flattened C-order array, so no implicit matrix
// transpose can swap source, mass, or Fourier axes. The prepared physics
// model is immutable and remains owned by its separate Python object.
class covariance_request_cpp_3pcf {
 public:
  covariance_request_cpp_3pcf(
    const covariance_input_3pcf& input, int nsource
  );
  covariance_request_cpp_3pcf(const covariance_request_cpp_3pcf&) = delete;
  covariance_request_cpp_3pcf& operator=(
    const covariance_request_cpp_3pcf&
  ) = delete;
  const covariance_input_3pcf& get() const;

 private:
  covariance_input_3pcf request;
  std::map<std::string, arma::Col<double>> real;
  std::map<std::string, arma::Col<int>> integer;
};

std::pair<arma::Col<double>, arma::Cube<double>> covariance_physical_cpp_3pcf(
  const physics_model_3pcf* model,
  const covariance_request_cpp_3pcf& input
);

// Halo arrays retain C axis order: each returned column is flattened as
// [redshift,mass], or [redshift,mass,k] for the fifth (profile) result.
std::vector<arma::Col<double>> covariance_halos_cpp_3pcf(
  const physics_model_3pcf* model,
  const covariance_request_cpp_3pcf& input,
  const arma::Col<double>& z,
  const arma::Col<double>& k_profile
);

arma::Col<double> covariance_xi_cpp_3pcf(
  const arma::Col<double>& ell,
  const arma::Col<double>& dell,
  const arma::Mat<double>& theta,
  const arma::Mat<double>& theta_weight,
  const arma::Mat<double>& spectra_e,
  const arma::Mat<double>& spectra_b,
  const arma::Col<int>& sign
);

}
#endif
