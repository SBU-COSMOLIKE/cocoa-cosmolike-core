#include <carma.h>
#include <armadillo>
#include <map>

// Python Binding
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#ifndef __COSMOLIKE_COSMO2D_SCUTS_WRAPPER_HPP
#define __COSMOLIKE_COSMO2D_SCUTS_WRAPPER_HPP

namespace cosmolike_interface
{

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Scale-cut diagnostics (2011.06469 eq 17), implemented in
// cosmo2D_scuts_wrapper.cpp. Per observable X in {xi_pm, w_ks (real
// space); C_ss, C_ks (fourier space)}:
//
//   dlnX/dlnk  = (dX/dlnk)/X, the log-response of X to the power at
//                wavenumber k - where the signal comes from
//   RF(kmax)   = int_{-inf}^{ln kmax} |dlnX/dlnk| dlnk over the full
//                integral (RF = response function): the fraction of
//                the response below kmax - the scale-cut statistic
//
// Scalar overloads = point diagnostics (full batch cost per call);
// array overloads = the batch tools.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// dlnxi_pm/dlnk at one k / at many k (real space, all pairs and bins)
py::tuple dlnxi_dlnk_pm_tomo_limber_cpp(const double k);

py::tuple dlnxi_dlnk_pm_tomo_limber_cpp(const arma::Col<double> k);

// -----------------------------------------------------------------------------

// dlnw_ks/dlnk at one k / at many k (real space, all source bins)
arma::Mat<double> dlnw_ks_dlnk_tomo_cpp(const double k);

py::array_t<double,py::array::f_style> dlnw_ks_dlnk_tomo_cpp(
    const arma::Col<double> k
  );

// -----------------------------------------------------------------------------

// RF of xi_pm at one (kmax, theta, pair) / on a kmax grid
py::tuple RF_xi_tomo_limber_cpp(
    const double k,
    const int nt,
    const int ni,
    const int nj
  );

py::tuple RF_xi_tomo_limber_cpp(const arma::Col<double> k);

// -----------------------------------------------------------------------------

// RF of w_ks at one (kmax, theta, source bin) / on a kmax grid
double RF_w_ks_tomo_cpp(
    const double k,
    const int nt,
    const int ni
  );

py::array_t<double,py::array::f_style> RF_w_ks_tomo_cpp(
    const arma::Col<double> k
  );

// -----------------------------------------------------------------------------

// dlnC_ss/dlnk at one (k, l, pair) / on a (k, l) grid (EE, BB tuple)
py::tuple dlnC_ss_dlnk_tomo_limber_cpp(
    const double k,
    const double l,
    const int ni,
    const int nj
  );

py::tuple dlnC_ss_dlnk_tomo_limber_cpp(
    const arma::Col<double> k,
    const arma::Col<double> l
  );

// -----------------------------------------------------------------------------

// dlnC_ks/dlnk at one (k, l, source bin) / on a (k, l) grid
double dlnC_ks_dlnk_tomo_limber_cpp(
    const double k,
    const double l,
    const int ni
  );

py::array_t<double,py::array::f_style> dlnC_ks_dlnk_tomo_limber_cpp(
    const arma::Col<double> k,
    const arma::Col<double> l
  );

// -----------------------------------------------------------------------------

// RF of C_ss at one (kmax, l, pair) / on a (kmax, l) grid (EE, BB)
py::tuple RF_C_ss_tomo_limber_cpp(
    const double k,
    const double l,
    const int ni,
    const int nj
  );

py::tuple RF_C_ss_tomo_limber_cpp(const arma::Col<double> k,
                                  const arma::Col<double> l);

// -----------------------------------------------------------------------------

// RF of C_ks at one (kmax, l, source bin) / on a (kmax, l) grid
double RF_C_ks_tomo_limber_cpp(
    const double k,
    const double l,
    const int ni
  );

py::array_t<double,py::array::f_style> RF_C_ks_tomo_limber_cpp(
    const arma::Col<double> k,
    const arma::Col<double> l
  );

// -----------------------------------------------------------------------------

// the Fourier gg spectra (autos, pairs) at the data-vector band centers
// with the k window of the UV-fraction scale cuts (arXiv:2603.13031):
// IR (keep_ir = 1) or UV part at kmax (h/Mpc at z = 0), the full
// spectra for kmax <= 0
py::tuple C_gg_kwindow_tomo_cpp(
    const double kmax,
    const int keep_ir
  );

// UV-fraction scale cuts (arXiv:2603.13031) of the Fourier gg spectra
// (autos, then cross pairs) at the data-vector band centers, on an
// ascending kmax grid (h/Mpc at z = 0): dict of lmax, mask, UV
// fractions and spectra
py::dict gg_scale_cuts_tomo_cpp(
    const arma::Col<double> kmax,
    const double alpha
  );

// -----------------------------------------------------------------------------

}  // namespace cosmolike_interface
#endif // HEADER GUARD
