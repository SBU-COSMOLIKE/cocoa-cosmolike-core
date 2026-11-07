#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <stdexcept>
#include <array>
#include <random>
#include <variant>
#include <cmath> 

// SPDLOG
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/cfg/env.h>

// ARMADILLO LIB AND PYBIND WRAPPER (CARMA)
#include <carma.h>
#include <armadillo>

// Python Binding
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
namespace py = pybind11;

// cosmolike
#include "cosmolike/basics.h"
#include "cosmolike/bias.h"
#include "cosmolike/IA.h"
#include "cosmolike/cosmo2D_wrapper.hpp"
#include "cosmolike/cosmo2D_scuts.h"
#include "cosmolike/cosmo2D.h"
#include "cosmolike/redshift_spline.h"
#include "cosmolike/structs.h"

using vector = arma::Col<double>;
using matrix = arma::Mat<double>;
using cube = arma::Cube<double>;

// ---------------------------------------------------------------------------
// Pybind wrappers of the scale-cut ("scuts") diagnostics: two tool
// families over each observable X in {xi_pm, w_ks (real space); C_ss,
// C_ks (fourier space)},
//
//   dlnX/dlnk  - where in wavenumber the signal comes from
//   RF_X(kmax) - what fraction of it comes from below kmax
//
// (definitions restated at the family banners below; engines in
// cosmo2D_scuts.c). Same batching and layout conventions as
// cosmo2D_wrapper.cpp: array overloads are the batch tools, scalar
// overloads are point diagnostics that pay the full batch cost.
// ---------------------------------------------------------------------------
namespace cosmolike_interface
{

// ---------------------------------------------------------------------------
// Stack a field of equally shaped matrices into a (n, rows, cols) numpy
// array: the 3d analogue of cosmo2D_wrapper.hpp's to_np4d, used by the
// ks diagnostics, whose single component (no EE/BB or xi+/xi- split, one
// source bin instead of a pair) drops one dimension from the ss outputs.
//
// Parameters:
//   f - field of equally shaped (rows, cols) matrices; a shape mismatch
//       aborts (spdlog::critical + exit)
//
// Returns:
//   Fortran-ordered numpy array of shape (n_elem, rows, cols); an empty
//   field gives a (0, 0, 0) array
// ---------------------------------------------------------------------------
static py::array_t<double,py::array::f_style> to_np3d(
    const arma::field<arma::Mat<double>>& f
  )
{
  if (0 == f.n_elem) {
    return py::array_t<double,py::array::f_style>(std::vector<int>{0,0,0});
  }
  const auto& m0 = f(0);
  const int n1 = static_cast<int>(m0.n_rows);
  const int n2 = static_cast<int>(m0.n_cols);
  const int n3 = static_cast<int>(f.n_elem);

  for (int k=1; k<n3; ++k) { // we need all matrices to have the same shape
    const auto& mk = f(k);
    const int m1 = static_cast<int>(mk.n_rows);
    const int m2 = static_cast<int>(mk.n_cols);
    if (m1 != n1 || m2 != n2) {
      spdlog::critical("{}: incompatible array structure", "to_np3d"); exit(1);
    }
  }

  // first: do a list of matrices
  py::list t;
  for (int k=0; k<n3; ++k) t.append(carma::mat_to_arr(f(k)));

  // second: stack the list of matrices into 3d np tensor
  py::module_ np = py::module_::import("numpy");
  py::array tmp = np.attr("stack")(t, py::arg("axis") = 0).cast<py::array>();

  // last: ensure Fortran-contiguous output to match the return type (f_style).
  py::array fA = np.attr("asfortranarray")(tmp).cast<py::array>();
  return fA.cast<py::array_t<double,py::array::f_style>>();
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// DERIVATIVE dlnX/dlnk (2011.06469 eq 17) - REAL SPACE
//
// The Limber projections are integrals over the matter power spectrum;
// rewritten per log-wavenumber, X = int dlnk (dX/dlnk). The quantity
//
//   dlnX/dlnk = (dX/dlnk) / X
//
// is the observable's log-response: the fractional contribution to X
// per unit ln k. It says which wavenumbers a given data point actually
// measures - the raw material of the scale cuts. The RESPONSE FUNCTION
// banners below integrate |dlnX/dlnk| in ln k to decide how much of a
// point rides on untrusted small-scale power.
//
// This family: X = xi_pm(theta) and w_ks(theta), Legendre sums of the
// dC/dlnk rows (engines in cosmo2D_scuts.c).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared engine of the two dlnxi_dlnk_pm_tomo_limber_cpp overloads: one
// dlnxi_dlnk_pm_tomo_nointerp call at wavenumber k (which computes every
// tomographic pair and angular bin at once), scattered into the
// (theta, ni, nj) cubes with both bin orderings filled (xi is symmetric).
//
// Parameters:
//   k  - wavenumber in (Mpc/h)^-1
//   XP - output xi+ cube (Ntheta, shear_nbin, shear_nbin)
//   XM - output xi- cube (Ntheta, shear_nbin, shear_nbin)
//
// Returns:
//   nothing; the result is written into XP and XM
// ---------------------------------------------------------------------------
static void dlnxi_dlnk_cubes(
    const double k,           // wavenumber in (Mpc/h)^-1
    arma::Cube<double>& XP,   // output xi+ (Ntheta, shear_nbin, shear_nbin)
    arma::Cube<double>& XM    // output xi- (Ntheta, shear_nbin, shear_nbin)
  )
{
  const int NSIZE = tomo.shear_Npowerspectra;
  double** tmp = dlnxi_dlnk_pm_tomo_nointerp(k);
  for (int nz=0; nz<NSIZE; nz++) {
    const int z1 = Z1(nz);
    const int z2 = Z2(nz);
    for (int i=0; i<Ntable.Ntheta; i++) {
      const int q = nz * Ntable.Ntheta + i;
      XP(i,z1,z2) = tmp[0][q];
      XP(i,z2,z1) = tmp[0][q];
      XM(i,z1,z2) = tmp[1][q];
      XM(i,z2,z1) = tmp[1][q];
    }
  }
  free(tmp);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dlnxi_pm/dlnk at one wavenumber, every tomographic pair and angular bin
// (one wavenumber is already a full batch: dlnxi_dlnk_cubes computes all
// pairs and bins in one call).
//
// Parameters:
//   k - wavenumber in (Mpc/h)^-1
//
// Returns:
//   (xi+, xi-) tuple of numpy arrays of shape
//   (Ntheta, shear_nbin, shear_nbin): rows = angular bin, both bin
//   orderings filled (xi is symmetric)
// ---------------------------------------------------------------------------
py::tuple dlnxi_dlnk_pm_tomo_limber_cpp(
    const double k    // wavenumber in (Mpc/h)^-1
  )
{
  arma::Cube<double> dlnxp_dlnk(Ntable.Ntheta,
                                redshift.shear_nbin,
                                redshift.shear_nbin,
                                arma::fill::zeros);
  arma::Cube<double> dlnxm_dlnk(Ntable.Ntheta,
                                redshift.shear_nbin,
                                redshift.shear_nbin,
                                arma::fill::zeros);
  dlnxi_dlnk_cubes(k, dlnxp_dlnk, dlnxm_dlnk);
  return py::make_tuple(carma::cube_to_arr(dlnxp_dlnk),
                        carma::cube_to_arr(dlnxm_dlnk));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dlnxi_pm/dlnk at many wavenumbers, stacked by to_np4d from one
// dlnxi_dlnk_cubes call per k (serial loop; each call is already a full
// batch over pairs and bins).
//
// Parameters:
//   k - wavenumbers in (Mpc/h)^-1; an empty array aborts
//       (spdlog::critical + exit)
//
// Returns:
//   (xi+, xi-) tuple of numpy arrays of shape
//   (nk, Ntheta, shear_nbin, shear_nbin): leading axis = wavenumber,
//   both bin orderings filled (xi is symmetric)
// ---------------------------------------------------------------------------
py::tuple dlnxi_dlnk_pm_tomo_limber_cpp(
    const arma::Col<double> k    // wavenumbers in (Mpc/h)^-1
  )
{
  const int nk = static_cast<int>(k.n_elem);
  if (!(nk > 0)) {
    spdlog::critical("{}: k array size = {}",
                     "dlnxi_dlnk_pm_tomo_limber_cpp", nk);
    exit(1);
  }
  arma::field<arma::Cube<double>> dlnxp_dlnk(nk);
  arma::field<arma::Cube<double>> dlnxm_dlnk(nk);
  for (int m=0; m<nk; m++) {
    arma::Cube<double> tdlnxp_dlnk(Ntable.Ntheta,
                                   redshift.shear_nbin,
                                   redshift.shear_nbin,
                                   arma::fill::zeros);
    arma::Cube<double> tdlnxm_dlnk(Ntable.Ntheta,
                                   redshift.shear_nbin,
                                   redshift.shear_nbin,
                                   arma::fill::zeros);
    dlnxi_dlnk_cubes(k(m), tdlnxp_dlnk, tdlnxm_dlnk);
    dlnxp_dlnk(m) = tdlnxp_dlnk;
    dlnxm_dlnk(m) = tdlnxm_dlnk;
  }
  return py::make_tuple(to_np4d(dlnxp_dlnk), to_np4d(dlnxm_dlnk));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared engine of the two dlnw_ks_dlnk_tomo_cpp overloads: one
// dlnw_ks_dlnk_tomo_nointerp call at wavenumber k (which computes every
// source bin and angular bin at once), scattered into the (theta, ni)
// matrix (one source bin per column: CMB lensing has a single source
// plane, the last-scattering surface, so kappa carries no bin index).
//
// Parameters:
//   k  - wavenumber in (Mpc/h)^-1
//   WK - output matrix (Ntheta, shear_nbin)
//
// Returns:
//   nothing; the result is written into WK
// ---------------------------------------------------------------------------
static void dlnw_ks_dlnk_mat(
    const double k,        // wavenumber in (Mpc/h)^-1
    arma::Mat<double>& WK  // output (Ntheta, shear_nbin)
  )
{
  const int NSIZE = redshift.shear_nbin;
  double* tmp = dlnw_ks_dlnk_tomo_nointerp(k);
  for (int nz=0; nz<NSIZE; nz++) {
    for (int i=0; i<Ntable.Ntheta; i++) {
      WK(i, nz) = tmp[nz * Ntable.Ntheta + i];
    }
  }
  free(tmp);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dlnw_ks/dlnk at one wavenumber, every source bin and angular bin (one
// wavenumber is already a full batch: dlnw_ks_dlnk_mat computes all bins
// in one call).
//
// Parameters:
//   k - wavenumber in (Mpc/h)^-1
//
// Returns:
//   arma::Mat (Ntheta, shear_nbin): rows = angular bin, columns =
//   source bin
// ---------------------------------------------------------------------------
arma::Mat<double> dlnw_ks_dlnk_tomo_cpp(
    const double k    // wavenumber in (Mpc/h)^-1
  )
{
  arma::Mat<double> dlnwks_dlnk(Ntable.Ntheta,
                                redshift.shear_nbin,
                                arma::fill::zeros);
  dlnw_ks_dlnk_mat(k, dlnwks_dlnk);
  return dlnwks_dlnk;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dlnw_ks/dlnk at many wavenumbers, stacked by to_np3d from one
// dlnw_ks_dlnk_mat call per k (serial loop; each call is already a full
// batch over bins).
//
// Parameters:
//   k - wavenumbers in (Mpc/h)^-1; an empty array aborts
//       (spdlog::critical + exit)
//
// Returns:
//   numpy array of shape (nk, Ntheta, shear_nbin): leading axis =
//   wavenumber, then angular bin, then source bin
// ---------------------------------------------------------------------------
py::array_t<double,py::array::f_style> dlnw_ks_dlnk_tomo_cpp(
    const arma::Col<double> k    // wavenumbers in (Mpc/h)^-1
  )
{
  const int nk = static_cast<int>(k.n_elem);
  if (!(nk > 0)) {
    spdlog::critical("{}: k array size = {}",
                     "dlnw_ks_dlnk_tomo_cpp", nk);
    exit(1);
  }
  arma::field<arma::Mat<double>> dlnwks_dlnk(nk);
  for (int m=0; m<nk; m++) {
    arma::Mat<double> tdlnwks_dlnk(Ntable.Ntheta,
                                   redshift.shear_nbin,
                                   arma::fill::zeros);
    dlnw_ks_dlnk_mat(k(m), tdlnwks_dlnk);
    dlnwks_dlnk(m) = tdlnwks_dlnk;
  }
  return to_np3d(dlnwks_dlnk);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// DERIVATIVE dlnX/dlnk (2011.06469 eq 17) - FOURIER SPACE
//
// The same log-response dlnX/dlnk = (dX/dlnk)/X as the REAL SPACE
// banner above, now for X = C_l^ss and C_l^ks. In Limber each (k, l)
// pair maps onto one line-of-sight node chi = (l + 1/2)/k, so dC/dlnk
// is a single evaluation of the C_l integrand at that node (see
// dC_ss_dlnk_tomo_limber in cosmo2D_scuts.c); the value is 0 when the
// node falls outside the source support.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared batch engine of the two dlnC_ss_dlnk_tomo_limber_cpp overloads:
// dlnC = dC/C computed exactly on the (ln k, ell) grid by the normalized
// mode of dC_ss_dlnk_tomo_limber_work, indexed [2][NSIZE][nk][nl] with
// row nz the pair (Z1(nz), Z2(nz)). The caller owns (and frees) the
// returned array.
//
// Parameters:
//   lnkx  - ln k grid values (length nk), k in (Mpc/h)^-1
//   nk    - number of k values
//   lx    - multipole values (length nl)
//   nl    - number of multipole values
//   NSIZE - number of tomo shear power spectra
//
// Returns:
//   malloc4d array [2][NSIZE][nk][nl] ([0] = EE, [1] = BB)
// ---------------------------------------------------------------------------
static double**** dlnC_ss_dlnk_grid(
    const double* lnkx, // ln k grid values (length nk), k in (Mpc/h)^-1
    const int nk,       // number of k values
    const double* lx,   // multipole values (length nl)
    const int nl,       // number of multipole values
    const int NSIZE     // number of tomo shear power spectra
  )
{
  double**** dlnC = (double****) malloc4d(2, NSIZE, nk, nl);
  dC_ss_dlnk_tomo_limber_work(lnkx, nk, lx, nl, NSIZE, 1, dlnC);
  return dlnC;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dlnC_ss/dlnk at one (k, l) for the (ni, nj) source pair in either
// ordering; 0 when the Limber node chi = (l + 1/2)/k falls outside the
// source support.
//
// Point diagnostic: runs the full batch of dlnC_ss_dlnk_grid at a
// single (k, l) and reads one entry, so it pays the whole-tomography
// batch cost per call. Loops over (k, l, ni, nj) should call the array
// overload once and index the returned arrays instead.
//
// Parameters:
//   k  - wavenumber in (Mpc/h)^-1; k <= 0 aborts (spdlog::critical +
//        exit)
//   l  - multipole
//   ni - first source redshift bin; outside [0, shear_nbin) aborts
//   nj - second source redshift bin; same validation as ni
//
// Returns:
//   (EE, BB) tuple of doubles
// ---------------------------------------------------------------------------
py::tuple dlnC_ss_dlnk_tomo_limber_cpp(
    const double k,   // wavenumber in (Mpc/h)^-1
    const double l,   // multipole
    const int ni,     // first source redshift bin
    const int nj      // second source redshift bin
  )
{
  if (!(k > 0)) {
    spdlog::critical("{}: k = {} not positive",
                     "dlnC_ss_dlnk_tomo_limber_cpp", k);
    exit(1);
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1 ||
      nj < 0 || nj > redshift.shear_nbin - 1) {
    spdlog::critical("{}: invalid bin input (ni, nj) = ({}, {})",
                     "dlnC_ss_dlnk_tomo_limber_cpp", ni, nj);
    exit(1);
  }
  const int NSIZE = tomo.shear_Npowerspectra;
  const double lnkx = std::log(k);
  const double lx = l;
  double**** dlnC = dlnC_ss_dlnk_grid(&lnkx, 1, &lx, 1, NSIZE);
  const int q = N_shear(ni, nj); // symmetric: any (ni, nj) ordering works
  const double CEE = dlnC[0][q][0][0];
  const double CBB = dlnC[1][q][0][0];
  free(dlnC);
  return py::make_tuple(CEE, CBB);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dlnC_ss/dlnk on a (k, l) grid, scattered from one dlnC_ss_dlnk_grid
// call; only the enumerated Z1 <= Z2 ordering is filled, the reversed
// entries stay zero.
//
// Parameters:
//   k - wavenumbers in (Mpc/h)^-1; an empty array or any k(m) <= 0
//       aborts (spdlog::critical + exit)
//   l - multipoles; an empty array aborts
//
// Returns:
//   (EE, BB) tuple of numpy arrays of shape
//   (nk, nl, shear_nbin, shear_nbin): leading axis = wavenumber, then
//   multipole, then the source bin pair
// ---------------------------------------------------------------------------
py::tuple dlnC_ss_dlnk_tomo_limber_cpp(
    const arma::Col<double> k,   // wavenumbers in (Mpc/h)^-1
    const arma::Col<double> l    // multipoles
  )
{
  const int nl = static_cast<int>(l.n_elem);
  const int nk = static_cast<int>(k.n_elem);
  if (!(nl > 0)) {
    spdlog::critical("{}: l array size = {}", "dlnw_ks_dlnk_tomo_cpp", nl);
    exit(1);
  }
  if (!(nk > 0)) {
    spdlog::critical("{}: k array size = {}", "dlnw_ks_dlnk_tomo_cpp", nk);
    exit(1);
  } 
  const int NSIZE = tomo.shear_Npowerspectra;
  double* lnkx = (double*) malloc1d(nk);
  for (int m=0; m<nk; m++) {
    if (!(k(m) > 0)) {
      spdlog::critical("{}: k({}) = {} not positive",
                       "dlnC_ss_dlnk_tomo_limber_cpp", m, k(m));
      exit(1);
    }
    lnkx[m] = std::log(k(m));
  }
  double* lx = (double*) malloc1d(nl);
  for (int i=0; i<nl; i++) {
    lx[i] = l(i);
  }
  double**** dlnC = dlnC_ss_dlnk_grid(lnkx, nk, lx, nl, NSIZE);
  arma::field<arma::Cube<double>> dlnCEEdlnk(nk);
  arma::field<arma::Cube<double>> dlnCBBdlnk(nk);
  for (int m=0; m<nk; m++) {
    arma::Cube<double> EE(l.n_elem,
                          redshift.shear_nbin,
                          redshift.shear_nbin,
                          arma::fill::zeros);
    arma::Cube<double> BB(l.n_elem,
                          redshift.shear_nbin,
                          redshift.shear_nbin,
                          arma::fill::zeros);
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<nl; i++) {
        EE(i, Z1(nz), Z2(nz)) = dlnC[0][nz][m][i];
        BB(i, Z1(nz), Z2(nz)) = dlnC[1][nz][m][i];
      }
    }
    dlnCEEdlnk(m) = EE;
    dlnCBBdlnk(m) = BB;
  }
  free(dlnC);
  free(lnkx);
  free(lx);
  return py::make_tuple(to_np4d(dlnCEEdlnk), to_np4d(dlnCBBdlnk));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared batch engine of the two dlnC_ks_dlnk_tomo_limber_cpp overloads:
// dlnC = dC/C computed exactly on the (ln k, ell) grid by the normalized
// mode of dC_ks_dlnk_tomo_limber_work, indexed [NSIZE][nk][nl] with row
// nz the source bin (one component per bin: CMB lensing has a single
// source plane, so kappa carries no bin index). The caller owns (and
// frees) the returned array.
//
// Parameters:
//   lnkx  - ln k grid values (length nk), k in (Mpc/h)^-1
//   nk    - number of k values
//   lx    - multipole values (length nl)
//   nl    - number of multipole values
//   NSIZE - number of source tomographic bins (= shear_nbin)
//
// Returns:
//   malloc3d array [NSIZE][nk][nl]
// ---------------------------------------------------------------------------
static double*** dlnC_ks_dlnk_grid(
    const double* lnkx, // ln k grid values (length nk), k in (Mpc/h)^-1
    const int nk,       // number of k values
    const double* lx,   // multipole values (length nl)
    const int nl,       // number of multipole values
    const int NSIZE     // number of source tomographic bins (= shear_nbin)
  )
{
  double*** dlnC = (double***) malloc3d(NSIZE, nk, nl);
  dC_ks_dlnk_tomo_limber_work(lnkx, nk, lx, nl, NSIZE, 1, dlnC);
  return dlnC;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dlnC_ks/dlnk at one (k, l) for source bin ni; 0 when the Limber node
// chi = (l + 1/2)/k falls outside the bin's source support.
//
// Point diagnostic: runs the full batch of dlnC_ks_dlnk_grid at a
// single (k, l) and reads one entry, so it pays the whole-tomography
// batch cost per call. Loops over (k, l, ni) should call the array
// overload once and index the returned array instead.
//
// Parameters:
//   k  - wavenumber in (Mpc/h)^-1; k <= 0 aborts (spdlog::critical +
//        exit)
//   l  - multipole
//   ni - source redshift bin; outside [0, shear_nbin) aborts
//
// Returns:
//   dlnC_ks/dlnk as a double
// ---------------------------------------------------------------------------
double dlnC_ks_dlnk_tomo_limber_cpp(
    const double k,   // wavenumber in (Mpc/h)^-1
    const double l,   // multipole
    const int ni      // source redshift bin
  )
{
  if (!(k > 0)) {
    spdlog::critical("{}: k = {} not positive",
                     "dlnC_ks_dlnk_tomo_limber_cpp", k);
    exit(1);
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    spdlog::critical("{}: invalid bin input ni = {}",
                     "dlnC_ks_dlnk_tomo_limber_cpp", ni);
    exit(1);
  }
  const int NSIZE = redshift.shear_nbin;
  const double lnkx = std::log(k);
  const double lx = l;
  double*** dlnC = dlnC_ks_dlnk_grid(&lnkx, 1, &lx, 1, NSIZE);
  const double CKS = dlnC[ni][0][0];
  free(dlnC);
  return CKS;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dlnC_ks/dlnk on a (k, l) grid, scattered from one dlnC_ks_dlnk_grid
// call (one column per source bin).
//
// Parameters:
//   k - wavenumbers in (Mpc/h)^-1; an empty array or any k(m) <= 0
//       aborts (spdlog::critical + exit)
//   l - multipoles; an empty array aborts
//
// Returns:
//   numpy array of shape (nk, nl, shear_nbin): leading axis =
//   wavenumber, then multipole, then source bin
// ---------------------------------------------------------------------------
py::array_t<double,py::array::f_style> dlnC_ks_dlnk_tomo_limber_cpp(
    const arma::Col<double> k,   // wavenumbers in (Mpc/h)^-1
    const arma::Col<double> l    // multipoles
  )
{
  const int nl = static_cast<int>(l.n_elem);
  const int nk = static_cast<int>(k.n_elem);
  if (!(nl > 0)) {
    spdlog::critical("{}: l array size = {}", "dlnC_ks_dlnk_tomo_limber_cpp", nl);
    exit(1);
  }
  if (!(nk > 0)) {
    spdlog::critical("{}: k array size = {}", "dlnC_ks_dlnk_tomo_limber_cpp", nk);
    exit(1);
  }
  const int NSIZE = redshift.shear_nbin;
  double* lnkx = (double*) malloc1d(nk);
  for (int m=0; m<nk; m++) {
    if (!(k(m) > 0)) {
      spdlog::critical("{}: k({}) = {} not positive",
                       "dlnC_ks_dlnk_tomo_limber_cpp", m, k(m));
      exit(1);
    }
    lnkx[m] = std::log(k(m));
  }
  double* lx = (double*) malloc1d(nl);
  for (int i=0; i<nl; i++) {
    lx[i] = l(i);
  }
  double*** dlnC = dlnC_ks_dlnk_grid(lnkx, nk, lx, nl, NSIZE);
  arma::field<arma::Mat<double>> dlnCKSdlnk(nk);
  for (int m=0; m<nk; m++) {
    arma::Mat<double> KS(l.n_elem, redshift.shear_nbin, arma::fill::zeros);
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<nl; i++) {
        KS(i, nz) = dlnC[nz][m][i];
      }
    }
    dlnCKSdlnk(m) = KS;
  }
  free(dlnC);
  free(lnkx);
  free(lx);
  return to_np3d(dlnCKSdlnk);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// RESPONSE FUNCTION RF (2011.06469 eq 17) - REAL SPACE
//
// The scale-cut statistic. For a data point of observable X,
//
//   RF(kmax) = int_{-inf}^{ln kmax} dlnk |dlnX/dlnk|
//              / int_{-inf}^{+inf} dlnk |dlnX/dlnk|
//
// is the fraction of the point's total log-k response accumulated
// below kmax; 1 - RF(kmax) is the share riding on scales above kmax,
// where the power-spectrum modeling (nonlinearity, baryons) is not
// trusted. Per-point cuts come from solving RF(kk) = alpha downstream
// of these exports (neither this file nor cosmo2D_scuts.c root-finds;
// see the cosmo2D_scuts.c preamble): kk is the wavenumber below which
// the point has accumulated fraction alpha of its response, and
// 2011.06469 keeps the point when kk <= kmax.
//
// This family: X = xi_pm(theta) and w_ks(theta).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared batch engine of the two RF_xi_tomo_limber_cpp overloads:
// RF computed by RF_xi_tomo_limber_work on the ln kmax grid, indexed
// [2][NSIZE][nk][Ntheta] with row nz the pair (Z1(nz), Z2(nz)). The
// caller owns (and frees) the returned array.
//
// Parameters:
//   lnkmaxx - ln kmax values (length nk), k in (Mpc/h)^-1
//   nk      - number of kmax values
//   NSIZE   - number of tomo shear power spectra
//
// Returns:
//   malloc4d array [2][NSIZE][nk][Ntheta] ([0] = xi+, [1] = xi-)
// ---------------------------------------------------------------------------
static double**** RF_xi_grid(
    const double* lnkmaxx, // ln kmax values (length nk), k in (Mpc/h)^-1
    const int nk,          // number of kmax values
    const int NSIZE        // number of tomo shear power spectra
  )
{
  double**** RF = (double****) malloc4d(2, NSIZE, nk, Ntable.Ntheta);
  RF_xi_tomo_limber_work(lnkmaxx, nk, NSIZE, RF);
  return RF;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Real-space response function RF(kmax, theta_nt) (2011.06469 eq 17)
// for the (ni, nj) source pair in either ordering.
//
// Point diagnostic: runs the full batch of RF_xi_grid at a single kmax
// and reads one entry, so it pays the whole-tomography batch cost per
// call. Loops over (kmax, nt, ni, nj) should call the array overload
// once and index the returned arrays instead.
//
// Parameters:
//   k  - cutoff wavenumber kmax in (Mpc/h)^-1; k <= 0 aborts
//        (spdlog::critical + exit)
//   nt - angular bin index; outside [0, Ntheta) aborts
//   ni - first source redshift bin; outside [0, shear_nbin) aborts
//   nj - second source redshift bin; same validation as ni
//
// Returns:
//   (RF for xi+, RF for xi-) tuple of doubles
// ---------------------------------------------------------------------------
py::tuple RF_xi_tomo_limber_cpp(
    const double k,   // cutoff wavenumber kmax in (Mpc/h)^-1
    const int nt,     // angular bin index (0..Ntheta-1)
    const int ni,     // first source redshift bin
    const int nj      // second source redshift bin
  )
{
  if (!(k > 0)) {
    spdlog::critical("{}: k = {} not positive",
                     "RF_xi_tomo_limber_cpp", k);
    exit(1);
  }
  if (nt < 0 || nt > Ntable.Ntheta - 1) {
    spdlog::critical("{}: invalid angular bin input nt = {}",
                     "RF_xi_tomo_limber_cpp", nt);
    exit(1);
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1 ||
      nj < 0 || nj > redshift.shear_nbin - 1) {
    spdlog::critical("{}: invalid bin input (ni, nj) = ({}, {})",
                     "RF_xi_tomo_limber_cpp", ni, nj);
    exit(1);
  }
  const int NSIZE = tomo.shear_Npowerspectra;
  const double lnkmax = std::log(k);
  double**** RF = RF_xi_grid(&lnkmax, 1, NSIZE);
  const int q = N_shear(ni, nj); // symmetric: any (ni, nj) ordering works
  const double RFXIP = RF[0][q][0][nt];
  const double RFXIM = RF[1][q][0][nt];
  free(RF);
  return py::make_tuple(RFXIP, RFXIM);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Real-space response function RF(kmax, theta) (2011.06469 eq 17) on a
// kmax grid, scattered from one RF_xi_grid call with both bin orderings
// filled (xi is symmetric).
//
// Parameters:
//   k - cutoff wavenumbers kmax in (Mpc/h)^-1; an empty array or any
//       k(m) <= 0 aborts (spdlog::critical + exit)
//
// Returns:
//   (xi+, xi-) tuple of numpy arrays of shape
//   (nk, Ntheta, shear_nbin, shear_nbin): leading axis = kmax, then
//   angular bin, then the source bin pair
// ---------------------------------------------------------------------------
py::tuple RF_xi_tomo_limber_cpp(
    const arma::Col<double> k    // cutoff wavenumbers kmax in (Mpc/h)^-1
  )
{
  const int nk = static_cast<int>(k.n_elem);
  if (!(nk > 0)) {
    spdlog::critical("{}: k array size = {}", "RF_xi_tomo_limber_cpp", nk);
    exit(1);
  }
  const int NSIZE = tomo.shear_Npowerspectra;
  double* lnkmaxx = (double*) malloc1d(nk);
  for (int m=0; m<nk; m++) {
    if (!(k(m) > 0)) {
      spdlog::critical("{}: k({}) = {} not positive",
                       "RF_xi_tomo_limber_cpp", m, k(m));
      exit(1);
    }
    lnkmaxx[m] = std::log(k(m));
  }
  double**** RF = RF_xi_grid(lnkmaxx, nk, NSIZE);
  arma::field<arma::Cube<double>> RFXIP(nk);
  arma::field<arma::Cube<double>> RFXIM(nk);
  for (int m=0; m<nk; m++) {
    arma::Cube<double> XP(Ntable.Ntheta,
                          redshift.shear_nbin,
                          redshift.shear_nbin,
                          arma::fill::zeros);
    arma::Cube<double> XM(Ntable.Ntheta,
                          redshift.shear_nbin,
                          redshift.shear_nbin,
                          arma::fill::zeros);
    for (int nz=0; nz<NSIZE; nz++) {
      const int z1 = Z1(nz);
      const int z2 = Z2(nz);
      for (int i=0; i<Ntable.Ntheta; i++) {
        XP(i,z1,z2) = RF[0][nz][m][i];
        XP(i,z2,z1) = XP(i,z1,z2);
        XM(i,z1,z2) = RF[1][nz][m][i];
        XM(i,z2,z1) = XM(i,z1,z2);
      }
    }
    RFXIP(m) = XP;
    RFXIM(m) = XM;
  }
  free(RF);
  free(lnkmaxx);
  return py::make_tuple(to_np4d(RFXIP), to_np4d(RFXIM));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared batch engine of the two RF_w_ks_tomo_cpp overloads: RF computed
// by RF_w_ks_tomo_limber_work on the ln kmax grid, indexed
// [NSIZE][nk][Ntheta] with row nz the source bin. The caller owns (and
// frees) the returned array.
//
// Parameters:
//   lnkmaxx - ln kmax values (length nk), k in (Mpc/h)^-1
//   nk      - number of kmax values
//   NSIZE   - number of source tomographic bins (= shear_nbin)
//
// Returns:
//   malloc3d array [NSIZE][nk][Ntheta]
// ---------------------------------------------------------------------------
static double*** RF_w_ks_grid(
    const double* lnkmaxx, // ln kmax values (length nk), k in (Mpc/h)^-1
    const int nk,          // number of kmax values
    const int NSIZE        // number of source tomographic bins (= shear_nbin)
  )
{
  double*** RF = (double***) malloc3d(NSIZE, nk, Ntable.Ntheta);
  RF_w_ks_tomo_limber_work(lnkmaxx, nk, NSIZE, RF);
  return RF;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CMB-lensing x shear response function RF(kmax, theta_nt) (2011.06469
// eq 17) for source bin ni.
//
// Point diagnostic: runs the full batch of RF_w_ks_grid at a single
// kmax and reads one entry, so it pays the whole-tomography batch cost
// per call. Loops over (kmax, nt, ni) should call the array overload
// once and index the returned array instead.
//
// Parameters:
//   k  - cutoff wavenumber kmax in (Mpc/h)^-1; k <= 0 aborts
//        (spdlog::critical + exit)
//   nt - angular bin index; outside [0, Ntheta) aborts
//   ni - source redshift bin; outside [0, shear_nbin) aborts
//
// Returns:
//   RF as a double
// ---------------------------------------------------------------------------
double RF_w_ks_tomo_cpp(
    const double k,   // cutoff wavenumber kmax in (Mpc/h)^-1
    const int nt,     // angular bin index (0..Ntheta-1)
    const int ni      // source redshift bin
  )
{
  if (!(k > 0)) {
    spdlog::critical("{}: k = {} not positive",
                     "RF_w_ks_tomo_cpp", k);
    exit(1);
  }
  if (nt < 0 || nt > Ntable.Ntheta - 1) {
    spdlog::critical("{}: invalid angular bin input nt = {}",
                     "RF_w_ks_tomo_cpp", nt);
    exit(1);
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    spdlog::critical("{}: invalid bin input ni = {}",
                     "RF_w_ks_tomo_cpp", ni);
    exit(1);
  }
  const int NSIZE = redshift.shear_nbin;
  const double lnkmax = std::log(k);
  double*** RF = RF_w_ks_grid(&lnkmax, 1, NSIZE);
  const double RFWKS = RF[ni][0][nt];
  free(RF);
  return RFWKS;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CMB-lensing x shear response function RF(kmax, theta) (2011.06469
// eq 17) on a kmax grid, scattered from one RF_w_ks_grid call (one
// column per source bin).
//
// Parameters:
//   k - cutoff wavenumbers kmax in (Mpc/h)^-1; an empty array or any
//       k(m) <= 0 aborts (spdlog::critical + exit)
//
// Returns:
//   numpy array of shape (nk, Ntheta, shear_nbin): leading axis =
//   kmax, then angular bin, then source bin
// ---------------------------------------------------------------------------
py::array_t<double,py::array::f_style> RF_w_ks_tomo_cpp(
    const arma::Col<double> k    // cutoff wavenumbers kmax in (Mpc/h)^-1
  )
{
  const int nk = static_cast<int>(k.n_elem);
  if (!(nk > 0)) {
    spdlog::critical("{}: k array size = {}", "RF_w_ks_tomo_cpp", nk);
    exit(1);
  }
  const int NSIZE = redshift.shear_nbin;
  double* lnkmaxx = (double*) malloc1d(nk);
  for (int m=0; m<nk; m++) {
    if (!(k(m) > 0)) {
      spdlog::critical("{}: k({}) = {} not positive",
                       "RF_w_ks_tomo_cpp", m, k(m));
      exit(1);
    }
    lnkmaxx[m] = std::log(k(m));
  }
  double*** RF = RF_w_ks_grid(lnkmaxx, nk, NSIZE);
  arma::field<arma::Mat<double>> RFWKS(nk);
  for (int m=0; m<nk; m++) {
    arma::Mat<double> WK(Ntable.Ntheta, redshift.shear_nbin,
                         arma::fill::zeros);
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        WK(i, nz) = RF[nz][m][i];
      }
    }
    RFWKS(m) = WK;
  }
  free(RF);
  free(lnkmaxx);
  return to_np3d(RFWKS);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// RESPONSE FUNCTION RF (2011.06469 eq 17) - FOURIER SPACE
//
// The same statistic as the REAL SPACE banner above - RF(kmax) =
// (response below kmax) / (total response), from integrating
// |dlnX/dlnk| in ln k - now for X = C_l^ss and C_l^ks per multipole.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared batch engine of the two RF_C_ss_tomo_limber_cpp overloads:
// RF computed by RF_C_ss_tomo_limber_work on the (ln kmax, ell) grid,
// indexed [2][NSIZE][nk][nl] with row nz the pair (Z1(nz), Z2(nz)).
// The caller owns (and frees) the returned array.
//
// Parameters:
//   lnkmaxx - ln kmax values (length nk), k in (Mpc/h)^-1
//   nk      - number of kmax values
//   lx      - multipole values (length nl)
//   nl      - number of multipole values
//   NSIZE   - number of tomo shear power spectra
//
// Returns:
//   malloc4d array [2][NSIZE][nk][nl] ([0] = EE, [1] = BB)
// ---------------------------------------------------------------------------
static double**** RF_C_ss_grid(
    const double* lnkmaxx, // ln kmax values (length nk), k in (Mpc/h)^-1
    const int nk,          // number of kmax values
    const double* lx,      // multipole values (length nl)
    const int nl,          // number of multipole values
    const int NSIZE        // number of tomo shear power spectra
  )
{
  double**** RF = (double****) malloc4d(2, NSIZE, nk, nl);
  RF_C_ss_tomo_limber_work(lnkmaxx, nk, lx, nl, NSIZE, RF);
  return RF;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fourier-space response function RF(kmax, l) (2011.06469 eq 17) for
// the (ni, nj) source pair in either ordering.
//
// Point diagnostic: runs the full batch of RF_C_ss_grid at a single
// (kmax, l) and reads one entry, so it pays the whole-tomography batch
// cost per call. Loops over (kmax, l, ni, nj) should call the array
// overload once and index the returned arrays instead.
//
// Parameters:
//   k  - cutoff wavenumber kmax in (Mpc/h)^-1; k <= 0 aborts
//        (spdlog::critical + exit)
//   l  - multipole
//   ni - first source redshift bin; outside [0, shear_nbin) aborts
//   nj - second source redshift bin; same validation as ni
//
// Returns:
//   (RF for EE, RF for BB) tuple of doubles
// ---------------------------------------------------------------------------
py::tuple RF_C_ss_tomo_limber_cpp(
    const double k,   // cutoff wavenumber kmax in (Mpc/h)^-1
    const double l,   // multipole
    const int ni,     // first source redshift bin
    const int nj      // second source redshift bin
  )
{
  if (!(k > 0)) {
    spdlog::critical("{}: k = {} not positive",
                     "RF_C_ss_tomo_limber_cpp", k);
    exit(1);
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1 ||
      nj < 0 || nj > redshift.shear_nbin - 1) {
    spdlog::critical("{}: invalid bin input (ni, nj) = ({}, {})",
                     "RF_C_ss_tomo_limber_cpp", ni, nj);
    exit(1);
  }
  const int NSIZE = tomo.shear_Npowerspectra;
  const double lnkmax = std::log(k);
  const double lx = l;
  double**** RF = RF_C_ss_grid(&lnkmax, 1, &lx, 1, NSIZE);
  const int q = N_shear(ni, nj); // symmetric: any (ni, nj) ordering works
  const double RFEE = RF[0][q][0][0];
  const double RFBB = RF[1][q][0][0];
  free(RF);
  return py::make_tuple(RFEE, RFBB);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fourier-space response function RF(kmax, l) (2011.06469 eq 17) on a
// (kmax, l) grid, scattered from one RF_C_ss_grid call; only the
// enumerated Z1 <= Z2 ordering is filled, the reversed entries stay
// zero.
//
// Parameters:
//   k - cutoff wavenumbers kmax in (Mpc/h)^-1; an empty array or any
//       k(m) <= 0 aborts (spdlog::critical + exit)
//   l - multipoles; an empty array aborts
//
// Returns:
//   (EE, BB) tuple of numpy arrays of shape
//   (nk, nl, shear_nbin, shear_nbin): leading axis = kmax, then
//   multipole, then the source bin pair
// ---------------------------------------------------------------------------
py::tuple RF_C_ss_tomo_limber_cpp(
    const arma::Col<double> k,   // cutoff wavenumbers kmax in (Mpc/h)^-1
    const arma::Col<double> l    // multipoles
  )
{
  const int nl = static_cast<int>(l.n_elem);
  const int nk = static_cast<int>(k.n_elem);
  if (!(nl > 0)) {
    spdlog::critical("{}: l array size = {}", "RF_w_ks_tomo_cpp", nl);
    exit(1);
  }
  if (!(nk > 0)) {
    spdlog::critical("{}: k array size = {}", "RF_w_ks_tomo_cpp", nk);
    exit(1);
  } 
  const int NSIZE = tomo.shear_Npowerspectra;
  double* lnkmaxx = (double*) malloc1d(nk);
  for (int m=0; m<nk; m++) {
    if (!(k(m) > 0)) {
      spdlog::critical("{}: k({}) = {} not positive",
                       "RF_C_ss_tomo_limber_cpp", m, k(m));
      exit(1);
    }
    lnkmaxx[m] = std::log(k(m));
  }
  double* lx = (double*) malloc1d(nl);
  for (int i=0; i<nl; i++) {
    lx[i] = l(i);
  }
  double**** RF = RF_C_ss_grid(lnkmaxx, nk, lx, nl, NSIZE);
  arma::field<arma::Cube<double>> RFEE(nk);
  arma::field<arma::Cube<double>> RFBB(nk);
  for (int m=0; m<nk; m++) {
    arma::Cube<double> EE(l.n_elem,
                          redshift.shear_nbin,
                          redshift.shear_nbin,
                          arma::fill::zeros);
    arma::Cube<double> BB(l.n_elem,
                          redshift.shear_nbin,
                          redshift.shear_nbin,
                          arma::fill::zeros);
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<nl; i++) {
        EE(i, Z1(nz), Z2(nz)) = RF[0][nz][m][i];
        BB(i, Z1(nz), Z2(nz)) = RF[1][nz][m][i];
      }
    }
    RFEE(m) = EE;
    RFBB(m) = BB;
  }
  free(RF);
  free(lnkmaxx);
  free(lx);
  return py::make_tuple(to_np4d(RFEE), to_np4d(RFBB));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared batch engine of the two RF_C_ks_tomo_limber_cpp overloads:
// RF computed by RF_C_ks_tomo_limber_work on the (ln kmax, ell) grid,
// indexed [NSIZE][nk][nl] with row nz the source bin. The caller owns
// (and frees) the returned array.
//
// Parameters:
//   lnkmaxx - ln kmax values (length nk), k in (Mpc/h)^-1
//   nk      - number of kmax values
//   lx      - multipole values (length nl)
//   nl      - number of multipole values
//   NSIZE   - number of source tomographic bins (= shear_nbin)
//
// Returns:
//   malloc3d array [NSIZE][nk][nl]
// ---------------------------------------------------------------------------
static double*** RF_C_ks_grid(
    const double* lnkmaxx, // ln kmax values (length nk), k in (Mpc/h)^-1
    const int nk,          // number of kmax values
    const double* lx,      // multipole values (length nl)
    const int nl,          // number of multipole values
    const int NSIZE        // number of source tomographic bins (= shear_nbin)
  )
{
  double*** RF = (double***) malloc3d(NSIZE, nk, nl);
  RF_C_ks_tomo_limber_work(lnkmaxx, nk, lx, nl, NSIZE, RF);
  return RF;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CMB-lensing x shear fourier-space response function RF(kmax, l)
// (2011.06469 eq 17) for source bin ni.
//
// Point diagnostic: runs the full batch of RF_C_ks_grid at a single
// (kmax, l) and reads one entry, so it pays the whole-tomography batch
// cost per call. Loops over (kmax, l, ni) should call the array
// overload once and index the returned array instead.
//
// Parameters:
//   k  - cutoff wavenumber kmax in (Mpc/h)^-1; k <= 0 aborts
//        (spdlog::critical + exit)
//   l  - multipole
//   ni - source redshift bin; outside [0, shear_nbin) aborts
//
// Returns:
//   RF as a double
// ---------------------------------------------------------------------------
double RF_C_ks_tomo_limber_cpp(
    const double k,   // cutoff wavenumber kmax in (Mpc/h)^-1
    const double l,   // multipole
    const int ni      // source redshift bin
  )
{
  if (!(k > 0)) {
    spdlog::critical("{}: k = {} not positive",
                     "RF_C_ks_tomo_limber_cpp", k);
    exit(1);
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    spdlog::critical("{}: invalid bin input ni = {}",
                     "RF_C_ks_tomo_limber_cpp", ni);
    exit(1);
  }
  const int NSIZE = redshift.shear_nbin;
  const double lnkmax = std::log(k);
  const double lx = l;
  double*** RF = RF_C_ks_grid(&lnkmax, 1, &lx, 1, NSIZE);
  const double RFKS = RF[ni][0][0];
  free(RF);
  return RFKS;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CMB-lensing x shear fourier-space response function RF(kmax, l)
// (2011.06469 eq 17) on a (kmax, l) grid, scattered from one
// RF_C_ks_grid call (one column per source bin).
//
// Parameters:
//   k - cutoff wavenumbers kmax in (Mpc/h)^-1; an empty array or any
//       k(m) <= 0 aborts (spdlog::critical + exit)
//   l - multipoles; an empty array aborts
//
// Returns:
//   numpy array of shape (nk, nl, shear_nbin): leading axis = kmax,
//   then multipole, then source bin
// ---------------------------------------------------------------------------
py::array_t<double,py::array::f_style> RF_C_ks_tomo_limber_cpp(
    const arma::Col<double> k,   // cutoff wavenumbers kmax in (Mpc/h)^-1
    const arma::Col<double> l    // multipoles
  )
{
  const int nl = static_cast<int>(l.n_elem);
  const int nk = static_cast<int>(k.n_elem);
  if (!(nl > 0)) {
    spdlog::critical("{}: l array size = {}", "RF_C_ks_tomo_limber_cpp", nl);
    exit(1);
  }
  if (!(nk > 0)) {
    spdlog::critical("{}: k array size = {}", "RF_C_ks_tomo_limber_cpp", nk);
    exit(1);
  }
  const int NSIZE = redshift.shear_nbin;
  double* lnkmaxx = (double*) malloc1d(nk);
  for (int m=0; m<nk; m++) {
    if (!(k(m) > 0)) {
      spdlog::critical("{}: k({}) = {} not positive",
                       "RF_C_ks_tomo_limber_cpp", m, k(m));
      exit(1);
    }
    lnkmaxx[m] = std::log(k(m));
  }
  double* lx = (double*) malloc1d(nl);
  for (int i=0; i<nl; i++) {
    lx[i] = l(i);
  }
  double*** RF = RF_C_ks_grid(lnkmaxx, nk, lx, nl, NSIZE);
  arma::field<arma::Mat<double>> RFKS(nk);
  for (int m=0; m<nk; m++) {
    arma::Mat<double> KS(l.n_elem, redshift.shear_nbin, arma::fill::zeros);
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<nl; i++) {
        KS(i, nz) = RF[nz][m][i];
      }
    }
    RFKS(m) = KS;
  }
  free(RF);
  free(lnkmaxx);
  free(lx);
  return to_np3d(RFKS);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// UV-FRACTION SCALE CUT: FOURIER C_gg (arXiv:2603.13031)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The Fourier gg spectra at the data-vector band centers (like.ell)
// with the k window of the UV-fraction scale cuts (arXiv:2603.13031):
// the IR part (keep_ir = 1, modes k <= kmax/D(z)) or the UV part
// (keep_ir = 0), or the full spectra for kmax <= 0. Engine:
// C_gg_kwindow_tomo_ells (cosmo2D_scuts.c). The two parts add up to
// the full spectra; this is the diagnostic that checks it.
//
// Parameters:
//   kmax    - cut at z = 0 in h/Mpc; kmax <= 0 = no window
//   keep_ir - 1 = IR part, 0 = UV part; anything else aborts
//
// Returns:
//   (autos, pairs) tuple of numpy arrays, shapes
//   (clustering_Npowerspectra, nell) and (gg_cross_npairs, nell)
// ---------------------------------------------------------------------------
py::tuple C_gg_kwindow_tomo_cpp(
    const double kmax,   // cut at z = 0 in h/Mpc (<= 0: no window)
    const int keep_ir    // 1 = IR part, 0 = UV part
  )
{
  if (keep_ir != 0 && keep_ir != 1) {
    spdlog::critical("{}: keep_ir = {} (0 or 1)",
                     "C_gg_kwindow_tomo_cpp", keep_ir);
    exit(1);
  }
  const int nell = like.Ncl;
  if (nell < 1 || NULL == like.ell) {
    spdlog::critical("{}: no Fourier band centers (init_binning_fourier"
                     "_ells)", "C_gg_kwindow_tomo_cpp");
    exit(1);
  }
  const int nautos = tomo.clustering_Npowerspectra;
  const int npairs = get_gg_cross_npairs();
  double** ca = (double**) malloc2d(nautos, nell);
  double** cx = (npairs > 0) ? (double**) malloc2d(npairs, nell) : NULL;
  C_gg_kwindow_tomo_ells(kmax, keep_ir, like.ell, nell, ca, cx);
  matrix A(nautos, nell);
  matrix X(npairs, nell);
  for (int i=0; i<nell; i++) {
    for (int s=0; s<nautos; s++) {
      A(s, i) = ca[s][i];
    }
    for (int n=0; n<npairs; n++) {
      X(n, i) = cx[n][i];
    }
  }
  free(ca);
  if (cx != NULL) {
    free(cx);
  }
  return py::make_tuple(carma::mat_to_arr(A), carma::mat_to_arr(X));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The UV-fraction scale cuts (arXiv:2603.13031) of the Fourier gg
// spectra at the data-vector band centers (like.ell): per band power
// the fraction |C_l^UV|/(|C_l^UV| + |C_l^IR|) of signal above the cut
// kmax/D(z), and the bands kept below alpha, for the cosmology and
// nuisances of the model's last evaluation. The engine is
// gg_scale_cuts_tomo_ells_work (cosmo2D_scuts.c, where the criterion
// is explained); this wrapper validates, allocates and converts.
//
// Spectrum rows s: the autos 0 .. clustering_Npowerspectra - 1, then
// the cross pairs in the order of the gg-cross block. Row s, column i
// is therefore entry nell*s + i counted from the first entry of the gg
// block of the data vector (the gg-cross block follows the gg block).
//
// Parameters:
//   kmax  - cuts at z = 0 in h/Mpc, strictly ascending; an empty array
//           aborts (spdlog::critical + exit), the engine checks the rest
//   alpha - allowed UV fraction, in (0, 1)
//
// Returns:
//   dict of numpy arrays:
//     kmax        (nk,)                the cuts, as passed
//     alpha       float
//     ell         (nell,)              the band centers
//     nautos      int                  rows 0 .. nautos - 1 are autos
//     lmax        (nspec, nk)          center of the last band kept
//                                      (0 when no band survives)
//     mask        (nspec, nell, nk)    1 = band kept, 0 = cut
//     uv_fraction (nspec, nell, nk)    |C^UV|/(|C^UV| + |C^IR|)
//     full        (nspec, nell)        the full spectra
//     ir          (nspec, nell, nk)    the IR parts
// ---------------------------------------------------------------------------
py::dict gg_scale_cuts_tomo_cpp(
    const arma::Col<double> kmax,   // cuts at z = 0 in h/Mpc
    const double alpha              // allowed UV fraction
  )
{
  const int nk = static_cast<int>(kmax.n_elem);
  if (nk < 1) {
    spdlog::critical("{}: empty kmax array", "gg_scale_cuts_tomo_cpp");
    exit(1);
  }
  const int nell = like.Ncl;
  if (nell < 1 || NULL == like.ell) {
    spdlog::critical("{}: no Fourier band centers (init_binning_fourier"
                     "_ells)", "gg_scale_cuts_tomo_cpp");
    exit(1);
  }
  const int nautos = tomo.clustering_Npowerspectra;
  const int nspec  = nautos + get_gg_cross_npairs();

  double* kv = (double*) malloc1d(nk);
  for (int m=0; m<nk; m++) {
    kv[m] = kmax(m);
  }
  double** full    = (double**) malloc2d(nspec, nell);
  double*** ir     = (double***) malloc3d(nk, nspec, nell);
  double*** uvfrac = (double***) malloc3d(nk, nspec, nell);
  int** nkeep      = (int**) malloc2d_int(nk, nspec);

  gg_scale_cuts_tomo_ells_work(kv, nk, alpha, like.ell, nell,
                               full, ir, uvfrac, nkeep);

  vector ell(nell);
  for (int i=0; i<nell; i++) {
    ell(i) = like.ell[i];
  }
  matrix F(nspec, nell);
  matrix L(nspec, nk);
  cube IR(nspec, nell, nk);
  cube UV(nspec, nell, nk);
  cube M(nspec, nell, nk);
  for (int s=0; s<nspec; s++) {
    for (int i=0; i<nell; i++) {
      F(s, i) = full[s][i];
    }
  }
  for (int m=0; m<nk; m++) {
    for (int s=0; s<nspec; s++) {
      L(s, m) = (nkeep[m][s] > 0) ? like.ell[nkeep[m][s] - 1] : 0.0;
      for (int i=0; i<nell; i++) {
        IR(s, i, m) = ir[m][s][i];
        UV(s, i, m) = uvfrac[m][s][i];
        M(s, i, m)  = (i < nkeep[m][s]) ? 1.0 : 0.0;
      }
    }
  }
  free(kv);
  free(full);
  free(ir);
  free(uvfrac);
  free(nkeep);

  vector K = kmax;
  py::dict out;
  out["kmax"]        = carma::col_to_arr(K);
  out["alpha"]       = alpha;
  out["ell"]         = carma::col_to_arr(ell);
  out["nautos"]      = nautos;
  out["lmax"]        = carma::mat_to_arr(L);
  out["mask"]        = carma::cube_to_arr(M);
  out["uv_fraction"] = carma::cube_to_arr(UV);
  out["full"]        = carma::mat_to_arr(F);
  out["ir"]          = carma::cube_to_arr(IR);
  return out;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

} // end namespace cosmolike_interface

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
