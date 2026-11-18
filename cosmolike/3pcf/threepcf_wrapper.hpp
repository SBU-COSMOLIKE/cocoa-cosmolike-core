#ifndef COSMOLIKE_THREEPCF_WRAPPER_HPP
#define COSMOLIKE_THREEPCF_WRAPPER_HPP

#include <armadillo>

namespace cosmolike_interface {

// Notebook quantities keep their physical axis order in Armadillo. The
// binding copies NumPy inputs before these functions are called; the
// wrappers only validate, change storage order, and call the shared C.
// Angles and aperture radii are in radians. No cosmology is installed.
struct aperture_weights_result_3pcf {
  arma::Cube<double> real; // [filter,component,triangle], real weights
  arma::Cube<double> imag; // same axes, imaginary weights
};

aperture_weights_result_3pcf aperture_weights_cpp_3pcf(
    const arma::Col<double>& d2, // [nd2], positive side-bin centers
    const arma::Col<double>& d3, // [nd3], positive side-bin centers
    const arma::Col<double>& phi, // [nphi], opening angles in (0,pi)
    double dlnd, // positive logarithmic side-bin width
    double dphi, // positive opening-angle-bin width
    const arma::Mat<double>& filters // [nfilter,3], aperture radii
  );

arma::Mat<double> aperture_apply_cpp_3pcf(
    const arma::Cube<double>& weights_real, // [filter,4,triangle]
    const arma::Cube<double>& weights_imag, // matching imaginary part
    const arma::Cube<double>& gamma_real, // [sample,4,triangle]
    const arma::Cube<double>& gamma_imag // matching imaginary part
  );

arma::Cube<double> filtered_pair_cpp_3pcf(
    const arma::Col<double>& ell, // [nell], angular multipoles
    const arma::Col<double>& ell_weights, // [nell], integration d ell
    const arma::Col<double>& radius, // [nr], separation in radians
    const arma::Mat<double>& filters, // [field,nell], Fourier filters
    const arma::Cube<double>& power // [field,field,nell], angular power
  );

arma::Mat<double> connected_gram_cpp_3pcf(
    const arma::Mat<double>& responses, // [row,node], halo responses
    const arma::Col<double>& weights // [node], nonnegative measure
  );

arma::Mat<double> gaussian_aperture_covariance_cpp_3pcf(
    const arma::Mat<int>& triplets, // [row,3], field labels
    const arma::Cube<double>& pair, // [field,field,r], pair correlations
    const arma::Mat<double>& pair_zero, // [field,field], at zero distance
    const arma::Col<double>& radial_weights // [nr], window-weighted area
  );

}
#endif
