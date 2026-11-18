#ifndef COSMOLIKE_TRANSFORM_WRAPPER_HPP
#define COSMOLIKE_TRANSFORM_WRAPPER_HPP
#include <armadillo>
#include "transform_3pcf.h"

namespace cosmolike_interface {
// Notebook arrays own their storage. Cube axes remain [L,ell1,ell2],
// even though Armadillo stores its first axis contiguously. These wrappers
// pack C-order buffers before calling exactly the production C routines.
arma::Cube<double> coupling_cpp_3pcf(
    const arma::Col<double>& psi, // angular ratio in radians
    int lmax, // maximum Legendre degree
    int mmax, // maximum angular mode
    int npsi // optional interpolation length
  );

arma::Mat<double> bessel_cpp_3pcf(
    const arma::Col<double>& ell, // positive logarithmic grid
    const arma::Mat<double>& integrand, // dln ell integrand
    int order1, // first Bessel order
    int order2, // second Bessel order
    double nu1, // first bias
    double nu2, // second bias
    double xy, // grid alignment
    double dlnt, // radial-bin width
    int padding, // zero padding on both ends
    bool window, // enable taper
    double window_width // taper fraction
  );

struct natural_result_cpp_3pcf {
  arma::Cube<double> real; // [4,ntheta*ntheta,nphi]
  arma::Cube<double> imag; // identical axes
};

natural_result_cpp_3pcf natural_cpp_3pcf(
    transform_plan_3pcf* plan, // owned by the notebook prepared-plan object
    const arma::Cube<double>& multipoles, // [L,nell,nell]
    int ntheta, // output radial count
    int nphi // output angle count
  );
}
#endif
