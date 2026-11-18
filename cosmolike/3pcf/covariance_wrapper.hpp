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
// transpose can swap source, mass, or Fourier axes. The physics model,
// which Python prepares and owns through a separate object, stays
// immutable here.
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
  covariance_input_3pcf request;  // pointers into real/integer below
  std::map<std::string, arma::Col<double>> real; // owned real copies
  std::map<std::string, arma::Col<int>> integer; // owned integer copies
};

// Returns (mean, terms): mean[nrow], and a cube indexed
// (row, column, term) holding the six covariance families in the
// order covariance_physics_3pcf.h documents.
std::pair<arma::Col<double>, arma::Cube<double>> covariance_physical_cpp_3pcf(
  const physics_model_3pcf* model,          // immutable physical tables
  const covariance_request_cpp_3pcf& input  // owned covariance inputs
);

// Halo arrays keep C axis order: each returned column is flattened as
// [redshift,mass], or [redshift,mass,k] for the fifth (profile) result.
// The five columns are, in order: sigma2, dndm, bias, concentration
// and the NFW profile u.
std::vector<arma::Col<double>> covariance_halos_cpp_3pcf(
  const physics_model_3pcf* model,          // read-only cosmology
  const covariance_request_cpp_3pcf& input, // owned halo quadratures
  const arma::Col<double>& z,               // redshift grid
  const arma::Col<double>& k_profile        // profile wavenumbers, h/Mpc
);

// Returns the bin-averaged xi of each row, as a column of length nrow.
arma::Col<double> covariance_xi_cpp_3pcf(
  const arma::Col<double>& ell,          // multipole nodes
  const arma::Col<double>& dell,         // dell quadrature weights
  const arma::Mat<double>& theta,        // separations [row,node], radians
  const arma::Mat<double>& theta_weight, // normalized bin weights
  const arma::Mat<double>& spectra_e,    // E-mode power [row,ell]
  const arma::Mat<double>& spectra_b,    // B-mode power [row,ell]
  const arma::Col<int>& sign             // +1 for xi+, -1 for xi-
);

}
#endif
