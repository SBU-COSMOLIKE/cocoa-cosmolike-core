#ifndef COSMOLIKE_TRANSFORM_WRAPPER_HPP
#define COSMOLIKE_TRANSFORM_WRAPPER_HPP
#include <armadillo>
#include "transform_3pcf.h"
#include "fftlog_3pcf.h"

namespace cosmolike_interface {

// Notebook adapters for the angular coupling, the double-Bessel FFTLog
// transform and the natural-component reconstruction. Every Armadillo
// argument owns its storage: the pybind layer copies the caller's NumPy
// input before these wrappers run. A cube keeps its physical axis order,
// [L,ell1,ell2] for the Legendre coefficients b_L, even though Armadillo
// stores the first index fastest while C reads the last index fastest;
// each wrapper therefore packs an explicit row-major buffer and hands
// that buffer to the same C routine the production path calls.
//
// Terms the comments below use:
//   psi   - the opening angle arctan(ell2/ell1) in [0,pi/2]. It encodes
//           the ratio of the two multipole legs but is itself an angle.
//   b_L   - the Legendre coefficients, in mu, of the projected E-mode
//           bispectrum; mu is the cosine in the triangle closure
//           ell3^2 = ell1^2 + ell2^2 - 2 ell1 ell2 mu.
//   nu    - the FFTLog bias: the power-law tilt divided out of the
//           integrand so the integral on the logarithmic grid becomes a
//           convolution that the FFT evaluates.
//   xy    - the constant in theta = xy/ell that fixes where the FFTLog
//           output grid lands.
//   taper - suppression of the highest log-frequency coefficients, which
//           damps the ringing that the FFT's periodic assumption causes;
//           window enables it, window_width sets the tapered fraction.
arma::Cube<double> coupling_cpp_3pcf(
    const arma::Col<double>& psi, // [nquery], opening angles in [0,pi/2]
    int lmax, // maximum Legendre degree
    int mmax, // maximum angular Fourier mode
    int npsi // optional interpolation length
  );

arma::Mat<double> bessel_cpp_3pcf(
    const arma::Col<double>& ell, // positive logarithmic grid
    const arma::Mat<double>& integrand, // dln ell integrand
    int order1, // first Bessel order
    int order2, // second Bessel order
    double nu1, // first FFTLog bias (tilt)
    double nu2, // second FFTLog bias (tilt)
    double xy, // constant in theta = xy/ell
    double dlnt, // log radial-bin width; 0 = point values
    int padding, // zero cells added at both grid ends
    bool window, // enable the high-frequency taper
    double window_width // tapered coefficient fraction
  );

// Prepared notebook stages. The owner keeps this workspace alive; these
// functions copy between Armadillo and its flat C buffers. A forward
// replaces the input and coefficients. Repeated inverses reuse those
// coefficients and return independent matrices, not borrowed scratch.
void bessel_forward_cpp_3pcf(
    fftlog_workspace_3pcf* work, // borrowed retained workspace
    int nell, // owner's physical grid length
    const arma::Mat<double>& integrand // [nell,nell], copied notebook input
  );

arma::Mat<double> bessel_inverse_cpp_3pcf(
    fftlog_workspace_3pcf* work, // workspace with prepared coefficients
    int nell, // owner's physical grid length
    int order1, // first Bessel order within the prepared capacity
    int order2 // second Bessel order within the prepared capacity
  );

struct natural_result_cpp_3pcf {
  arma::Cube<double> real; // [4,ntheta*ntheta,nphi]
  arma::Cube<double> imag; // identical axes
};

natural_result_cpp_3pcf natural_cpp_3pcf(
    transform_plan_3pcf* plan, // owned by the notebook prepared-plan object
    const arma::Cube<double>& multipoles, // b_L, [L,nell,nell]
    int ntheta, // output radial count
    int nphi // output angle count
  );

arma::Col<double> aperture_grid_cpp_3pcf(
    transform_plan_3pcf* plan, // owns geometric query and temporary b_L
    const physics_grid_3pcf* grid, // borrowed while this call runs
    const arma::Cube<double>& real, // [filter,4,triangle]
    const arma::Cube<double>& imag, // same shape, copied notebook values
    int ntriangle // expected triangle count from the owning plan
  );

}
#endif
