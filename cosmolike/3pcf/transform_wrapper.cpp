#include <algorithm>
#include <climits>
#include <cstdlib>
#include <spdlog/spdlog.h>
#include "coupling_3pcf.h"
#include "fftlog_3pcf.h"
#include "transform_wrapper.hpp"

namespace cosmolike_interface {
static void require_wrapper_transform(
    const bool condition, // shape or finite-input requirement
    const char* message // diagnostic
  )
{
  if (!condition) {
    spdlog::critical("3pcf transform wrapper: {}", message);
    std::exit(1);
  }
}


arma::Cube<double> coupling_cpp_3pcf(
    const arma::Col<double>& psi, // query angles
    const int lmax, // Legendre cutoff
    const int mmax, // Fourier cutoff
    const int npsi // interpolation grid length
  )
{
  require_wrapper_transform(psi.n_elem > 0 && psi.n_elem <= INT_MAX
      && psi.is_finite() && lmax >= 0 && lmax < INT_MAX-1
      && mmax >= 0 && mmax < INT_MAX-1
      && lmax <= INT_MAX-1-mmax, "invalid coupling input");
  const arma::uword count = (arma::uword)(mmax+1)*(lmax+1)*psi.n_elem;
  arma::Col<double> packed(count);
  legendre_coupling_3pcf(lmax, mmax, static_cast<int>(psi.n_elem),
      psi.memptr(), npsi, packed.memptr());
  arma::Cube<double> result(mmax+1, lmax+1, psi.n_elem);
  for (int mode=0; mode<=mmax; mode++) {
    for (int degree=0; degree<=lmax; degree++) {
      for (arma::uword query=0; query<psi.n_elem; query++) {
        result(mode, degree, query) =
            packed(((arma::uword)mode*(lmax+1)+degree)*psi.n_elem+query);
      }
    }
  }
  require_wrapper_transform(result.is_finite(), "nonfinite coupling");
  return result;
}


arma::Mat<double> bessel_cpp_3pcf(
    const arma::Col<double>& ell, // logarithmic Fourier grid
    const arma::Mat<double>& integrand, // dln ell integrand
    const int order1, // first Bessel order
    const int order2, // second Bessel order
    const double nu1, // first bias
    const double nu2, // second bias
    const double xy, // grid alignment
    const double dlnt, // radial-bin width
    const int padding, // zero padding
    const bool window, // taper enabled
    const double window_width // taper fraction
  )
{
  require_wrapper_transform(ell.n_elem > 0 && ell.n_elem <= INT_MAX
      && integrand.n_rows == ell.n_elem
      && integrand.n_cols == ell.n_elem && ell.is_finite()
      && integrand.is_finite() && order1 >= 0 && order2 >= 0,
      "invalid two-Bessel input");
  const int nell = static_cast<int>(ell.n_elem);
  auto* work = fftlog_create_3pcf(nell, ell.memptr(),
      std::max(order1, order2), 1, nu1, nu2, xy, dlnt, padding, window,
      window_width);
  double* packed = fftlog_input_3pcf(work, 0);
  for (int first=0; first<nell; first++) {
    for (int second=0; second<nell; second++) {
      packed[(size_t)first*nell+second] = integrand(first, second);
    }
  }
  fftlog_forward_3pcf(work, 0);
  const double* output = fftlog_inverse_3pcf(work, 0, order1, order2);
  arma::Mat<double> result(nell, nell);
  for (int first=0; first<nell; first++) {
    for (int second=0; second<nell; second++) {
      result(first, second) = output[(size_t)first*nell+second];
    }
  }
  fftlog_destroy_3pcf(work);
  require_wrapper_transform(result.is_finite(), "nonfinite transform");
  return result;
}


natural_result_cpp_3pcf natural_cpp_3pcf(
    transform_plan_3pcf* plan, // prepared geometry
    const arma::Cube<double>& multipoles, // [L,nell,nell]
    const int ntheta, // selected radial count
    const int nphi // output angular count
  )
{
  require_wrapper_transform(multipoles.is_finite(),
      "nonfinite bispectrum multipoles");
  arma::Col<double> packed(multipoles.n_elem);
  for (arma::uword degree=0; degree<multipoles.n_rows; degree++) {
    for (arma::uword first=0; first<multipoles.n_cols; first++) {
      for (arma::uword second=0; second<multipoles.n_slices; second++) {
        const arma::uword cell =
            (degree*multipoles.n_cols+first)*multipoles.n_slices+second;
        packed(cell) = multipoles(degree, first, second);
      }
    }
  }
  const arma::uword pairs = (arma::uword)ntheta*ntheta;
  const arma::uword triangles = pairs*nphi;
  arma::Col<double> real(4*triangles);
  arma::Col<double> imag(4*triangles);
  transform_natural_3pcf(plan, packed.memptr(), real.memptr(),
      imag.memptr());
  natural_result_cpp_3pcf result = {
    arma::Cube<double>(4, pairs, nphi),
    arma::Cube<double>(4, pairs, nphi)
  };
  for (int component=0; component<4; component++) {
    for (arma::uword pair=0; pair<pairs; pair++) {
      for (int angle=0; angle<nphi; angle++) {
        const arma::uword cell = (component*pairs+pair)*nphi+angle;
        result.real(component, pair, angle) = real(cell);
        result.imag(component, pair, angle) = imag(cell);
      }
    }
  }
  require_wrapper_transform(result.real.is_finite()
      && result.imag.is_finite(), "nonfinite natural components");
  return result;
}
}
