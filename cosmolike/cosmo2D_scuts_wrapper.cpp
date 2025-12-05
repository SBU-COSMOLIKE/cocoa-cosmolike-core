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
#include "cosmolike/cosmo2D_scuts.h"
#include "cosmolike/cosmo2D.h"
#include "cosmolike/redshift_spline.h"
#include "cosmolike/structs.h"

using vector = arma::Col<double>;
using matrix = arma::Mat<double>;
using cube = arma::Cube<double>;

namespace cosmolike_interface
{

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// DERIVATIVE: dlnX/dlnk: important to determine scale cuts (2011.06469 eq 17)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

py::tuple dlnxi_dlnk_pm_tomo_cpp(const double k)
{ 
  arma::Cube<double> dlnxp_dlnk(Ntable.Ntheta,
                                redshift.shear_nbin,
                                redshift.shear_nbin,
                                arma::fill::zeros);
  arma::Cube<double> dlnxm_dlnk(Ntable.Ntheta,
                                redshift.shear_nbin,
                                redshift.shear_nbin,
                                arma::fill::zeros);
  for (int nz=0; nz<tomo.shear_Npowerspectra; nz++) {    
    for (int i=0; i<Ntable.Ntheta; i++) {
      const int z1 = Z1(nz);
      const int z2 = Z2(nz);
      dlnxp_dlnk(i,z1,z2) = dlnxi_dlnk_pm_tomo(k, 1, i, z1, z2);
      dlnxm_dlnk(i,z1,z2) = dlnxi_dlnk_pm_tomo(k, -1, i, z1, z2);
      dlnxp_dlnk(i,z2,z1) = dlnxp_dlnk(i,z1,z2);
      dlnxm_dlnk(i,z2,z1) = dlnxm_dlnk(i,z1,z2);
    }
  }
  return py::make_tuple(carma::cube_to_arr(dlnxp_dlnk), 
                        carma::cube_to_arr(dlnxm_dlnk));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

py::tuple dlnC_ss_dlnk_tomo_limber_cpp(
    const double k, 
    const double l, 
    const int ni, 
    const int nj
  )
{
  return py::make_tuple(
    dlnC_ss_dlnk_tomo_limber_nointerp(k, l, ni, nj, 1, 0),
    dlnC_ss_dlnk_tomo_limber_nointerp(k, l, ni, nj, 0, 0) 
  );
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

py::tuple dlnC_ss_dlnk_tomo_limber_cpp(const double k, const arma::Col<double> l)
{
  if (!(l.n_elem > 0)) {
    spdlog::critical("{}: l array size = {}", "dC_ss_dlnk_tomo_limber_cpp", l.n_elem);
    exit(1);
  }
  arma::Cube<double> EE(l.n_elem,
                        redshift.shear_nbin,
                        redshift.shear_nbin,
                        arma::fill::zeros);
  arma::Cube<double> BB(l.n_elem,
                        redshift.shear_nbin,
                        redshift.shear_nbin,
                        arma::fill::zeros); 
  // init static vars
  (void) dlnC_ss_dlnk_tomo_limber_nointerp(k, l(0), Z1(0), Z2(0), 1, 1); // EE
  (void) dlnC_ss_dlnk_tomo_limber_nointerp(k, l(0), Z1(0), Z2(0), 0, 1); // BB
  #pragma omp parallel for collapse(2)
  for (int nz=0; nz<tomo.shear_Npowerspectra; nz++) {
    for (int i=0; i<static_cast<int>(l.n_elem); i++) {
      const int ni = Z1(nz);
      const int nj = Z2(nz);
      const double lx = l(i);
      EE(i, ni, nj) = dlnC_ss_dlnk_tomo_limber_nointerp(k, lx, ni, nj, 1, 0);
      BB(i, ni, nj) = dlnC_ss_dlnk_tomo_limber_nointerp(k, lx, ni, nj, 0, 0);
    }
  }
  return py::make_tuple(carma::cube_to_arr(EE), carma::cube_to_arr(BB));
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
