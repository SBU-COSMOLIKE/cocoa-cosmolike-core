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
//#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
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

// boost library
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>

// cosmolike
#include "cosmolike/basics.h"
#include "cosmolike/bias.h"
#include "cosmolike/baryons.h"
#include "cosmolike/cosmo2D.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/IA.h"
#include "cosmolike/halo.h"
#include "cosmolike/radial_weights.h"
#include "cosmolike/pt_cfastpt.h"
#include "cosmolike/redshift_spline.h"
#include "cosmolike/structs.h"

#include "cosmolike/generic_interface.hpp"

static const int force_cache_update_test = 0;

using vector = arma::Col<double>;
using matrix = arma::Mat<double>;
using cube = arma::Cube<double>;

// Why the cpp functions accept and return STL vectors (instead of arma:Col)?
// Answer: the conversion between STL vector and python np array is cleaner
// Answer: arma:Col is cast to 2D np array with 1 column (not as nice!)

namespace cosmolike_interface
{

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// AUX FUNCTIONS (PRIVATE)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

arma::Mat<double> read_table(const std::string file_name)
{
  std::ifstream input_file(file_name);
  if (!input_file.is_open()) {
    spdlog::critical("{}: file {} cannot be opened", "read_table", file_name);
    exit(1);
  }

  // --------------------------------------------------------
  // Read the entire file into memory
  // --------------------------------------------------------

  std::string tmp;
  
  input_file.seekg(0,std::ios::end);
  
  tmp.resize(static_cast<size_t>(input_file.tellg()));
  
  input_file.seekg(0,std::ios::beg);
  
  input_file.read(&tmp[0],tmp.size());
  
  input_file.close();
  
  if (tmp.empty())
  {
    spdlog::critical("{}: file {} is empty", "read_table", file_name);
    exit(1);
  }
  
  // --------------------------------------------------------
  // Second: Split file into lines
  // --------------------------------------------------------
  
  std::vector<std::string> lines;
  lines.reserve(50000);

  boost::trim_if(tmp, boost::is_any_of("\t "));
  
  boost::trim_if(tmp, boost::is_any_of("\n"));
  
  boost::split(lines, tmp,boost::is_any_of("\n"), boost::token_compress_on);
  
  // Erase comment/blank lines
  auto check = [](std::string mystr) -> bool
  {
    return boost::starts_with(mystr, "#");
  };
  lines.erase(std::remove_if(lines.begin(), lines.end(), check), lines.end());
  
  // --------------------------------------------------------
  // Third: Split line into words
  // --------------------------------------------------------

  arma::Mat<double> result;
  size_t ncols = 0;
  
  { // first line
    std::vector<std::string> words;
    words.reserve(100);
    
    boost::trim_left(lines[0]);
    boost::trim_right(lines[0]);

    boost::split(
      words,lines[0], 
      boost::is_any_of(" \t"),
      boost::token_compress_on
    );
    
    ncols = words.size();

    result.set_size(lines.size(), ncols);
    
    for (size_t j=0; j<ncols; j++)
      result(0,j) = std::stod(words[j]);
  }

  #pragma omp parallel for
  for (size_t i=1; i<lines.size(); i++)
  {
    std::vector<std::string> words;
    
    boost::trim_left(lines[i]);
    boost::trim_right(lines[i]);

    boost::split(
      words, 
      lines[i], 
      boost::is_any_of(" \t"),
      boost::token_compress_on
    );
    
    if (words.size() != ncols)
    {
      spdlog::critical("{}: file {} is not well formatted"
                       " (regular table required)", 
                       "read_table", 
                       file_name
                      );
      exit(1);
    }
    
    for (size_t j=0; j<ncols; j++)
      result(i,j) = std::stod(words[j]);
  };
  
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

std::tuple<std::string,int> get_baryon_sim_name_and_tag(std::string sim)
{
  // Desired Convention:
  // (1) Python input: not be case sensitive
  // (2) simulation names only have "_" as deliminator, e.g., owls_AGN.
  // (3) simulation IDs are indicated by "-", e.g., antilles-1.
 
  boost::trim_if(sim, boost::is_any_of("\t "));
  sim = boost::algorithm::to_lower_copy(sim);
  
  { // Count occurrences of - (dashes)
    size_t pos = 0; 
    size_t count = 0; 
    std::string tmp = sim;

    while ((pos = tmp.rfind("-")) != std::string::npos) 
    {
      tmp = tmp.substr(pos+1);
      ++count;
    }

    if (count > 1)
    {
      spdlog::critical(
        "{}: Scenario {} not supported (too many dashes)", 
        "get_baryon_sim_name_and_tag", sim);
      exit(1);
    }
  }

  if (sim.rfind("owls_agn") != std::string::npos)
  {
    boost::replace_all(sim, "owls_agn", "owls_AGN");
    boost::replace_all(sim, "_t80", "-1");
    boost::replace_all(sim, "_t85", "-2");
    boost::replace_all(sim, "_t87", "-3");
  } 
  else if (sim.rfind("bahamas") != std::string::npos)
  {
    boost::replace_all(sim, "bahamas", "BAHAMAS");
    boost::replace_all(sim, "_t78", "-1");
    boost::replace_all(sim, "_t76", "-2");
    boost::replace_all(sim, "_t80", "-3");
  } 
  else if (sim.rfind("hzagn") != std::string::npos)
  {
    boost::replace_all(sim, "hzagn", "HzAGN");
  }
  else if (sim.rfind("tng") != std::string::npos)
  {
    boost::replace_all(sim, "tng", "TNG");
  }
  
  std::string name;
  int tag;

  if (sim.rfind('-') != std::string::npos)
  {
    const size_t pos = sim.rfind('-');
    name = sim.substr(0, pos);
    tag = std::stoi(sim.substr(pos + 1));
  } 
  else
  { 
    name = sim;
    tag = 1; 
  }

  return std::make_tuple(name, tag);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// INIT FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void initial_setup()
{
  spdlog::cfg::load_env_levels();
  
  spdlog::debug("{}: Begins", "initial_setup");

  like.shear_shear = 0;
  like.shear_pos = 0;
  like.pos_pos = 0;

  like.Ncl = 0;
  like.lmin = 0;
  like.lmax = 0;

  like.gk = 0;
  like.kk = 0;
  like.ks = 0;
  
    // no priors
  like.clusterN = 0;
  like.clusterWL = 0;
  like.clusterCG = 0;
  like.clusterCC = 0;

  // reset bias - pretty important to setup variables to zero or 1 via reset
  reset_redshift_struct();
  reset_nuisance_struct();
  reset_cosmology_struct();
  reset_tomo_struct();
  reset_Ntable_struct();
  reset_like_struct();
  reset_cmb_struct();

  like.adopt_limber_gg = 0;

  std::string mode = "Halofit";
  memcpy(pdeltaparams.runmode, mode.c_str(), mode.size() + 1);

  spdlog::debug("{}: Ends", "initial_setup");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_ntable_lmax(const int lmax) {
  Ntable.LMAX = lmax;
  Ntable.random = RandomNumber::get_instance().get(); // update cache
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_accuracy_boost(
    const double accuracy_boost, 
    const int integration_accuracy
  )
{
  static int N_a = 0;
  static int N_ell = 0;

  if (0 == N_a) N_a = Ntable.N_a;
  Ntable.N_a = static_cast<int>(ceil(N_a*accuracy_boost));
  
  if (0 == N_ell) N_ell = Ntable.N_ell;
  Ntable.N_ell = static_cast<int>(ceil(N_ell*accuracy_boost));

  if (accuracy_boost>1) {
    Ntable.FPTboost = static_cast<int>(accuracy_boost-1.0);
  }
  else {
    Ntable.FPTboost = 0.0;
  }
  /*  
  Ntable.N_k_lin = 
    static_cast<int>(ceil(Ntable.N_k_lin*sampling_boost));
  
  Ntable.N_k_nlin = 
    static_cast<int>(ceil(Ntable.N_k_nlin*sampling_boost));

  Ntable.N_M = 
    static_cast<int>(ceil(Ntable.N_M*sampling_boost));
  */

  Ntable.high_def_integration = int(integration_accuracy);
  Ntable.random = RandomNumber::get_instance().get();
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_baryons_contamination(std::string sim)
{ // OLD API
  spdlog::debug("{}: Begins", "init_baryons_contamination");

  auto [name, tag] = get_baryon_sim_name_and_tag(sim);
  
  spdlog::debug(
      "{}: Baryon simulation w/ Name = {} & Tag = {} selected",
      "init_baryons_contamination", 
      name, 
      tag
    );

  std::string tmp = name + "-" + std::to_string(tag);

  init_baryons(tmp.c_str());
  
  spdlog::debug("{}: Ends", "init_baryons_contamination");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

#ifdef HDF5LIB
void init_baryons_contamination(
    std::string sim, std::string all_sims_hdf5_file
  )
{ // NEW API
  spdlog::debug("{}: Begins", "init_baryons_contamination");

  auto [name, tag] = get_baryon_sim_name_and_tag(sim);
       
  spdlog::debug(
      "{}: Baryon simulation w/ Name = {} & Tag = {} selected",
      "init_baryons_contamination",
      name,
      tag
    );

  init_baryons_from_hdf5_file(name.c_str(), tag, all_sims_hdf5_file.c_str());

  spdlog::debug("{}: Ends", "init_baryons_contamination");
}
#endif

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_bias(arma::Col<double> bias_z_evol_model)
{
  spdlog::debug("{}: Begins", "init_bias");
  
  if (MAX_SIZE_ARRAYS < static_cast<int>(bias_z_evol_model.n_elem)) [[unlikely]] {
    spdlog::critical(
        "{}: incompatible input {} size = {:d} (>{:d})", 
        "init_bias", 
        "bias_z_evol_model", 
        bias_z_evol_model.n_elem, 
        MAX_SIZE_ARRAYS
      );
    exit(1);
  }

  /*
  int galaxy_bias_model[MAX_SIZE_ARRAYS]; // [0] = b1, 
                                          // [1] = b2, 
                                          // [2] = bs2, 
                                          // [3] = b3, 
                                          // [4] = bmag 
  */
  for(int i=0; i<static_cast<int>(bias_z_evol_model.n_elem); i++) {
    if (std::isnan(bias_z_evol_model(i))) [[unlikely]] {
      // can't compile cosmolike with -O3 or -fast-math
      // see: https://stackoverflow.com/a/47703550/2472169
      spdlog::critical("{}: NaN found on index {}.", "init_bias", i);
      exit(1);
    }

    like.galaxy_bias_model[i] = bias_z_evol_model(i);
    
    spdlog::debug(
        "{}: {}[{}] = {} selected.", "init_bias", 
        "like.galaxy_bias_model",
        i,
        bias_z_evol_model(i)
      );
  }
  spdlog::debug("{}: Ends", "init_bias");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_binning_fourier(
    const int Nells, 
    const int lmin, 
    const int lmax,
    const int lmax_shear)
{
  spdlog::debug("{}: Begins", "init_binning_fourier");

  if (!(Nells > 0)) [[unlikely]] {
    spdlog::critical(
        "{}: {} = {:d} not supported",
        "init_binning_fourier", 
        "Number of l modes (Nells)", 
        Nells
      );
    exit(1);
  }
  spdlog::debug(
      "{}: {} = {:d} selected.",
      "init_binning_fourier",
      "Nells", 
      Nells
    );
  spdlog::debug(
      "{}: {} = {:d} selected.",
      "init_binning_fourier",
      "l_min", 
      lmin
    );
  spdlog::debug(
      "{}: {} = {:d} selected.",
      "init_binning_fourier",
      "l_max", 
      lmax
    );
  spdlog::debug(
      "{}: {} = {:d} selected.",
      "init_binning_fourier",
      "l_max_shear", 
      lmax_shear
    );

  like.Ncl = Nells;
  
  like.lmin = lmin;
  
  like.lmax = lmax;

  like.lmax_shear = lmax_shear;
  
  const double logdl = (std::log(lmax) - std::log(lmin))/ (double) like.Ncl;
  
  if (like.ell != NULL) {
    free(like.ell);
  }
  like.ell = (double*) malloc(sizeof(double)*like.Ncl);
  
  for (int i=0; i<like.Ncl; i++) {
    like.ell[i] = std::exp(std::log(like.lmin) + (i + 0.5)*logdl);
    /*spdlog::debug(
        "{}: Bin {:d}, {} = {:d}, {} = {:d} and {} = {:d}",
        "init_binning_fourier",
        i,
        "lmin",
        lmin,
        "ell",
        like.ell[i],
        "lmax",
        lmax
      );*/
  }
  spdlog::debug("{}: Ends", "init_binning_fourier");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_binning_real_space(
    const int Ntheta, 
    const double theta_min_arcmin, 
    const double theta_max_arcmin
  )
{
  spdlog::debug("{}: Begins", "init_binning_real_space");
  if (!(Ntheta > 0)) [[unlikely]] {
    spdlog::critical(
        "{}: {} = {:d} not supported", "init_binning_real_space",
        "Ntheta", 
        Ntheta
      );
    exit(1);
  }
  spdlog::debug(
      "{}: {} = {:d} selected.", "init_binning_real_space", 
      "Ntheta", 
      Ntheta
    );
  spdlog::debug(
      "{}: {} = {} selected.", "init_binning_real_space", 
      "theta_min_arcmin", 
      theta_min_arcmin
    );
  spdlog::debug(
      "{}: {} = {} selected.", 
      "init_binning_real_space", 
      "theta_max_arcmin", 
      theta_max_arcmin
    );
  Ntable.Ntheta = Ntheta;
  Ntable.vtmin  = theta_min_arcmin * 2.90888208665721580e-4; // arcmin to rad conv
  Ntable.vtmax  = theta_max_arcmin * 2.90888208665721580e-4; // arcmin to rad conv  
  spdlog::debug("{}: Ends", "init_binning_real_space");
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_cmb_cross_correlation (
    const double lmin, 
    const double lmax, 
    const double fwhm, // fwhm = beam size in arcmin
    std::string healpixwin_filename
  ) 
{
  spdlog::debug("{}: Begins", "init_cmb_cross_correlation");
  IPCMB& cmb = IPCMB::get_instance();
  // fwhm = beam size in arcmin - cmb.fwhm = beam size in rad
  cmb.set_wxk_beam_size(fwhm*2.90888208665721580e-4);
  cmb.set_wxk_lminmax(lmin, lmax);
  cmb.set_wxk_healpix_window(healpixwin_filename);
  cmb.update_chache(RandomNumber::get_instance().get());
  spdlog::debug("{}: Ends", "init_cmb_cross_correlation");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_cmb_auto_bandpower (
    const int nbp, 
    const int lmin, 
    const int lmax,
    std::string binning_matrix, 
    std::string theory_offset,
    const double alpha
  )
{
  spdlog::debug("{}: Begins", "init_cmb_auto_bandpower");
  IPCMB& cmb = IPCMB::get_instance();
  cmb.set_kk_binning_bandpower(nbp, lmin, lmax);
  cmb.set_kk_binning_mat(binning_matrix);
  cmb.set_kk_theory_offset(theory_offset);
  cmb.set_alpha_Hartlap_cov_kkkk(alpha);
  cmb.update_chache(RandomNumber::get_instance().get());
  spdlog::debug("{}: Ends", "init_cmb_auto_bandpower");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_cosmo_runmode(const bool is_linear)
{
  spdlog::debug("{}: Begins", "init_cosmo_runmode");

  std::string mode = is_linear ? "linear" : "Halofit";
  
  const size_t size = mode.size();
  
  memcpy(pdeltaparams.runmode, mode.c_str(), size + 1);

  spdlog::debug("{}: {} = {} selected", 
    "init_cosmo_runmode", "runmode", mode);
  
  spdlog::debug("{}: Ends", "init_cosmo_runmode");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_data_vector_size(
    arma::Col<int>::fixed<6> exclude,
    arma::Col<int>::fixed<6> ndv
  )
{
  spdlog::debug("{}: Begins", "init_data_vector_size");
  like.Ndata = 0.0;
  for(int i=0; i<static_cast<int>(exclude.n_elem); i++) 
  {
    if (exclude(i) > 0) {
      if (0 == i && 0 == tomo.shear_Npowerspectra) [[unlikely]] {
        spdlog::critical(
            "{}: {} not set prior to this function call",
            "init_data_vector_size", 
            "tomo.shear_Npowerspectra"
          );
        exit(1);
      }
      if (1 == i && 0 == tomo.ggl_Npowerspectra) [[unlikely]] {
        spdlog::critical(
            "{}: {} not set prior to this function call",
            "init_data_vector_size", 
            "tomo.ggl_Npowerspectra"
          );
        exit(1);
      }
      if (2 == i && 0 == tomo.clustering_Npowerspectra) [[unlikely]] {
        spdlog::critical(
            "{}: {} not set prior to this function call",
            "init_data_vector_size", 
            "tomo.clustering_Npowerspectra"
          );
        exit(1);
      }
      if (3 == i && 0 == redshift.clustering_nbin) [[unlikely]] {
        spdlog::critical(
            "{}: {} not set prior to this function call",
            "init_data_vector_size", 
            "redshift.clustering_nbin"
          );
        exit(1);
      }
      if (4 == i && 0 == redshift.shear_nbin) [[unlikely]] {
        spdlog::critical(
            "{}: {} not set prior to this function call",
            "init_data_vector_size", 
            "redshift.shear_nbin"
          );
        exit(1);
      }
      if (5 == i) {
        if (0 == IPCMB::get_instance().is_kk_bandpower()) {
          if (0 == like.Ncl) [[unlikely]] {
            spdlog::critical(
                "{}: {} not set prior to this function call",
                "init_data_vector_size", 
                "like.Ncl"
              );
            exit(1);
          }
        }
      }
      like.Ndata += ndv(i);
    }
  }
  spdlog::debug("{}: Ends", "init_data_vector_size");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_IA(const int IA_MODEL, const int IA_REDSHIFT_EVOL)
{
  spdlog::debug("{}: Begins", "init_IA");
  spdlog::debug("{}: {} = {} selected.", 
      "init_IA", "IA MODEL", IA_MODEL, 
      "IA REDSHIFT EVOLUTION", IA_REDSHIFT_EVOL
    );
  if (IA_MODEL == 0 || IA_MODEL == 1) {
    nuisance.IA_MODEL = IA_MODEL;
  }
  else [[unlikely]] {
    spdlog::critical(
        "{}: {} = {} not supported", 
        "init_IA", 
        "nuisance.IA_MODEL", 
        IA_MODEL
      );
    exit(1);
  }
  if (IA_REDSHIFT_EVOL == NO_IA                   || 
      IA_REDSHIFT_EVOL == IA_NLA_LF               ||
      IA_REDSHIFT_EVOL == IA_REDSHIFT_BINNING     || 
      IA_REDSHIFT_EVOL == IA_REDSHIFT_EVOLUTION)
  {
    nuisance.IA = IA_REDSHIFT_EVOL;
  }
  else [[unlikely]] {
    spdlog::critical(
        "{}: {} = {} not supported", 
        "init_IA", 
        "nuisance.IA", 
        IA_REDSHIFT_EVOL
      );
    exit(1);
  }
  spdlog::debug("{}: Ends", "init_IA");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_probes(std::string possible_probes)
{
  spdlog::debug("{}: Begins", "init_probes");
  
  static const std::unordered_map<std::string, arma::Col<int>::fixed<6>> 
    probe_map = {
        { "xi",     arma::Col<int>::fixed<6>{{1,0,0,0,0,0}} },
        { "gammat", arma::Col<int>::fixed<6>{{0,1,0,0,0,0}} },
        { "wtheta", arma::Col<int>::fixed<6>{{0,0,1,0,0,0}} },
        { "2x2pt",  arma::Col<int>::fixed<6>{{0,1,1,0,0,0}} },
        { "3x2pt",  arma::Col<int>::fixed<6>{{1,1,1,0,0,0}} },
        { "xi_ggl", arma::Col<int>::fixed<6>{{1,1,0,0,0,0}} },
        { "xi_gg",  arma::Col<int>::fixed<6>{{1,0,1,0,0,0}} },
        { "5x2pt",  arma::Col<int>::fixed<6>{{1,1,1,1,1,0}} },
        { "6x2pt",  arma::Col<int>::fixed<6>{{1,1,1,1,1,1}} },
        { "c3x2pt", arma::Col<int>::fixed<6>{{0,0,0,1,1,1}} }
    };
  static const std::unordered_map<std::string,std::string> 
    names = {
       {"xi", "cosmic shear"},
       {"gammat", "gammat"},
       {"wtheta", "wtheta"},
       {"2x2pt", "2x2pt"},
       {"3x2pt", "3x2pt"},
       {"xi_ggl", "xi + ggl (2x2pt)"},
       {"xi_gg",  "xi + ggl (2x2pt)"},
       {"5x2pt",  "5x2pt"},
       {"c3x2pt", "c3x2pt (gk + sk + kk)"},
       {"6x2pt",  "6x2pt"},
    };

  boost::trim_if(possible_probes, boost::is_any_of("\t "));
  possible_probes = boost::algorithm::to_lower_copy(possible_probes);
  auto it = probe_map.find(possible_probes);
  if (it == probe_map.end()) {
    spdlog::critical(
        "{}: {} = {} probe not supported","init_probes",
        "possible_probes",
        possible_probes
    );
    std::exit(1);
  }
  const auto& flags = it->second;

  like.shear_shear = flags(0);
  like.shear_pos = flags(1);
  like.pos_pos = flags(2);
  like.gk = flags(3);
  like.ks = flags(4);
  like.kk = flags(5);

  spdlog::debug(
      "{}: {} = {} selected", 
      "init_probes", 
      "possible_probes", 
      names.at(possible_probes)
    );
  spdlog::debug("{}: Ends", "init_probes");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

arma::Mat<double> read_nz_sample(std::string multihisto_file, const int Ntomo)
{
  spdlog::debug("{}: Begins", "read_nz_sample");

  if (!(multihisto_file.size() > 0)) [[unlikely]] {
    spdlog::critical(
        "{}: empty {} string not supported", 
        "read_nz_sample", 
        "multihisto_file"
      );
    exit(1);
  }
  if (!(Ntomo > 0) || Ntomo > MAX_SIZE_ARRAYS) [[unlikely]] {
    spdlog::critical(
        "{}: {} = {} not supported (max = {})", 
        "read_nz_sample", 
        "Ntomo", 
        Ntomo, 
        MAX_SIZE_ARRAYS
      );
    exit(1);
  }  
  spdlog::debug(
      "{}: {} = {} selected.", 
      "read_nz_sample",
      "redshift file:", 
      multihisto_file
    );
  spdlog::debug(
      "{}: {} = {} selected.", 
      "redshift",
      "nbin", 
      Ntomo
    );

  // READ THE N(Z) FILE BEGINS ------------
  arma::Mat<double> input_table = read_table(multihisto_file);
  if (!input_table.col(0).eval().is_sorted("ascend")) {
    spdlog::critical("bad n(z) file (z vector not monotonic)");
    exit(1);
  }
  
  spdlog::debug("{}: Ends", "read_nz_sample");
  return input_table;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_lens_sample(std::string multihisto_file, const int Ntomo)
{
  spdlog::debug("{}: Begins", "init_lens_sample v2.0");

  arma::Mat<double> input_table = read_nz_sample(multihisto_file, Ntomo);

  set_lens_sample_size(Ntomo);

  set_lens_sample(input_table);

  spdlog::debug("{}: Ends", "init_lens_sample v2.0");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_source_sample(std::string multihisto_file, const int Ntomo)
{
  spdlog::debug("{}: Begins", "init_source_sample");

  arma::Mat<double> input_table = read_nz_sample(multihisto_file, Ntomo);

  set_source_sample_size(Ntomo);

  set_source_sample(input_table);

  spdlog::debug("{}: Ends", "init_source_sample");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_ntomo_powerspectra()
{
  if (0 == redshift.shear_nbin) [[unlikely]] {
    spdlog::critical(
        "{}: {} not set prior to this function call", 
        "init_ntomo_powerspectra", 
        "redshift.shear_nbin"
      );
    exit(1);
  }
  if (0 == redshift.clustering_nbin) [[unlikely]] {
    spdlog::critical(
        "{}: {} not set prior to this function call", 
        "init_ntomo_powerspectra", 
        "redshift.clustering_nbin"
      );
    exit(1);
  }

  tomo.shear_Npowerspectra = redshift.shear_nbin * (redshift.shear_nbin + 1) / 2;

  int n = 0;
  for (int i=0; i<redshift.clustering_nbin; i++) {
    for (int j=0; j<redshift.shear_nbin; j++) {
      n += test_zoverlap(i, j);
      if(test_zoverlap(i, j) == 0) {
        spdlog::info(
            "{}: GGL pair L{:d}-S{:d} is excluded",
            "init_ntomo_powerspectra", 
            i, 
            j
          );
      }
    }
  }
  tomo.ggl_Npowerspectra = n;

  tomo.clustering_Npowerspectra = redshift.clustering_nbin;

  spdlog::debug(
      "{}: tomo.shear_Npowerspectra = {}",
      "init_ntomo_powerspectra", 
      tomo.shear_Npowerspectra
    );
  spdlog::debug(
      "{}: tomo.ggl_Npowerspectra = {}",
      "init_ntomo_powerspectra", 
      tomo.ggl_Npowerspectra
    );
  spdlog::debug(
      "{}: tomo.clustering_Npowerspectra = {}",
      "init_ntomo_powerspectra", 
      tomo.clustering_Npowerspectra
    );
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

py::tuple read_redshift_distributions_from_files(
  std::string lens_multihisto_file, const int lens_ntomo,
  std::string source_multihisto_file, const int source_ntomo)
{
  arma::Mat<double> input_lens_table = 
    read_nz_sample(lens_multihisto_file, lens_ntomo);

  arma::Mat<double> input_source_table = 
    read_nz_sample(source_multihisto_file, source_ntomo);

  return py::make_tuple(carma::mat_to_arr(input_lens_table), 
                        carma::mat_to_arr(input_source_table));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_redshift_distributions_from_files(
  std::string lens_multihisto_file, const int lens_ntomo,
  std::string source_multihisto_file, const int source_ntomo)
{
  init_lens_sample(lens_multihisto_file, lens_ntomo);
  
  init_source_sample(source_multihisto_file, source_ntomo);

  init_ntomo_powerspectra();
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_survey(
    std::string surveyname, 
    double area, 
    double sigma_e)
{
  spdlog::debug("{}: Begins", "init_survey");

  boost::trim_if(surveyname, boost::is_any_of("\t "));
  
  surveyname = boost::algorithm::to_lower_copy(surveyname);
    
  if (surveyname.size() > CHAR_MAX_SIZE - 1) {
    spdlog::critical(
        "{}: survey name too large for Cosmolike "
        "(C char memory overflow)", "init_survey"
      );
    exit(1);
  }
  if (!(surveyname.size()>0)) {
    spdlog::critical("{}: incompatible input", "init_survey");
    exit(1);
  }

  memcpy(survey.name, surveyname.c_str(), surveyname.size() + 1);
  
  survey.area = area;
  
  survey.sigma_e = sigma_e;

  spdlog::debug("{}: Ends", "init_survey");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_ggl_exclude(arma::Col<int> ggl_exclude)
{
  spdlog::debug("{}: Begins", "init_ggl_exclude");

  arma::Col<int> _ggl_excl_ = arma::conv_to<arma::Col<int>>::from(ggl_exclude);
  
  if (tomo.ggl_exclude != NULL) {
    free(tomo.ggl_exclude);
  }
  tomo.ggl_exclude = (int*) malloc(sizeof(int)*ggl_exclude.n_elem);
  if (NULL == tomo.ggl_exclude) {
    spdlog::critical("array allocation failed");
    exit(1);
  }

  tomo.N_ggl_exclude = int(ggl_exclude.n_elem/2);
  
  spdlog::debug("init_ggl_exclude: {} ggl pairs excluded", tomo.N_ggl_exclude);
  
  #pragma omp parallel for
  for(int i=0; i<static_cast<int>(ggl_exclude.n_elem); i++) {
    if (std::isnan(ggl_exclude(i))) {
      // can't compile cosmolike with -O3 or -fast-math
      // see: https://stackoverflow.com/a/47703550/2472169
      spdlog::critical("{}: NaN found on index {}.", "init_ggl_exclude", i);
      exit(1);
    }
    tomo.ggl_exclude[i] = ggl_exclude(i);
  }
  spdlog::debug("{}: Ends", "init_ggl_exclude");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// SET FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_cosmological_parameters(
    const double omega_matter,
    const double hubble
  )
{
  spdlog::debug("{}: Begins", "set_cosmological_parameters");

  // Cosmolike should not need parameters from inflation or dark energy.
  // Cobaya provides P(k,z), H(z), D(z), Chi(z)...
  // It may require H0 to set scales and \Omega_M to set the halo model

  int cache_update = 0;
  if (fdiff(cosmology.Omega_m, omega_matter) ||
      fdiff(cosmology.h0, hubble/100.0)) // assuming H0 in km/s/Mpc 
  {
    cache_update = 1;
  }
  if (1 == cache_update || 1 == force_cache_update_test) {
    cosmology.Omega_m = omega_matter;
    cosmology.Omega_v = 1.0-omega_matter;
    // Cosmolike only needs to know that there are massive neutrinos (>0)
    cosmology.Omega_nu = 0.1;
    cosmology.h0 = hubble/100.0; 
    cosmology.MGSigma = 0.0;
    cosmology.MGmu = 0.0;
    cosmology.random = cosmolike_interface::RandomNumber::get_instance().get();
  }

  spdlog::debug("{}: Ends", "set_cosmological_parameters");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_distances(arma::Col<double> io_z, arma::Col<double> io_chi)
{
  spdlog::debug("{}: Begins", "set_distances");
  bool debug_fail = false;
  if (io_z.n_elem != io_chi.n_elem) [[unlikely]] {
    debug_fail = true;
  }
  else {
    if (io_z.n_elem == 0) [[unlikely]] {
      debug_fail = true;
    }
  }
  if (debug_fail) [[unlikely]] {
    spdlog::critical(
        "{}: incompatible input w/ z.size = {:d} and G.size = {:d}",
        "set_distances", 
        io_z.n_elem, 
        io_chi.n_elem
      );
    exit(1);
  }
  if(io_z.n_elem < 5) [[unlikely]] {
    spdlog::critical("{}: bad input with z.size = {:d} and chi.size = {:d}",
                     "set_distances", 
                     io_z.n_elem, 
                     io_chi.n_elem);
    exit(1);
  }

  int cache_update = 0;
  if (cosmology.chi_nz != static_cast<int>(io_z.n_elem) || 
      NULL == cosmology.chi) {
    cache_update = 1;
  }
  else {
    for (int i=0; i<cosmology.chi_nz; i++) {
      if (fdiff(cosmology.chi[0][i], io_z(i)) ||
          fdiff(cosmology.chi[1][i], io_chi(i))) {
        cache_update = 1; 
        break; 
      }    
    }
  }
  if (1 == cache_update || 1 == force_cache_update_test) {
    cosmology.chi_nz = static_cast<int>(io_z.n_elem);
    if (cosmology.chi != NULL) {
      free(cosmology.chi);
    }
    cosmology.chi = (double**) malloc2d(2, cosmology.chi_nz);

    #pragma omp parallel for
    for (int i=0; i<cosmology.chi_nz; i++) {
      if (std::isnan(io_z(i)) || std::isnan(io_chi(i))) [[unlikely]] {
        // can't compile cosmolike with -O3 or -fast-math
        // see: https://stackoverflow.com/a/47703550/2472169
        spdlog::critical("{}: NaN found on interpolation table.", "set_distances");
        exit(1);
      }
      cosmology.chi[0][i] = io_z(i);
      cosmology.chi[1][i] = io_chi(i);
    }
    cosmology.random = RandomNumber::get_instance().get();
  }
  spdlog::debug("{}: Ends", "set_distances");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// Growth: D = G * a
void set_growth(arma::Col<double> io_z, arma::Col<double> io_G)
{
  spdlog::debug("{}: Begins", "set_growth");
  bool debug_fail = false;
  if (io_z.n_elem != io_G.n_elem) [[unlikely]] {
    debug_fail = true;
  }
  else {
    if (io_z.n_elem == 0) [[unlikely]] {
      debug_fail = true;
    }
  }
  if (debug_fail) [[unlikely]] {
    spdlog::critical(
        "{}: incompatible input w/ z.size = {} and G.size = {}",
        "set_growth", 
        io_z.n_elem, io_G.n_elem
      );
    exit(1);
  }
  if(io_z.n_elem < 5) [[unlikely]] {
    spdlog::critical(
        "{}: bad input w/ z.size = {} and G.size = {}",
        "set_growth", 
        io_z.n_elem, io_G.n_elem
      );
    exit(1);
  }

  int cache_update = 0;
  if (cosmology.G_nz != static_cast<int>(io_z.n_elem) || 
      NULL == cosmology.G) {
    cache_update = 1;
  }
  else
  {
    for (int i=0; i<cosmology.G_nz; i++) {
      if (fdiff(cosmology.G[0][i], io_z(i)) ||
          fdiff(cosmology.G[1][i], io_G(i))) {
        cache_update = 1; 
        break;
      }    
    }
  }
  if (1 == cache_update || 1 == force_cache_update_test)
  {
    cosmology.G_nz = static_cast<int>(io_z.n_elem);
    if (cosmology.G != NULL) {
      free(cosmology.G);
    }
    cosmology.G = (double**) malloc2d(2, cosmology.G_nz);

    #pragma omp parallel for
    for (int i=0; i<cosmology.G_nz; i++) {
      if (std::isnan(io_z(i)) || std::isnan(io_G(i))) [[unlikely]] {
        // can't compile cosmolike with -O3 or -fast-math
        // see: https://stackoverflow.com/a/47703550/2472169
        spdlog::critical("{}: NaN found on interpolation table.", "set_growth");
        exit(1);
      }
      cosmology.G[0][i] = io_z(i);
      cosmology.G[1][i] = io_G(i);
    }
    cosmology.random = RandomNumber::get_instance().get();
  }
  spdlog::debug("{}: Ends", "set_growth");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_linear_power_spectrum(
    arma::Col<double> io_log10k, 
    arma::Col<double> io_z, 
    arma::Col<double> io_lnP
  )
{
  spdlog::debug("{}: Begins", "set_linear_power_spectrum");

  bool debug_fail = false;
  if (io_z.n_elem*io_log10k.n_elem != io_lnP.n_elem) [[unlikely]] {
    debug_fail = true;
  }
  else {
    if (io_z.n_elem == 0 || io_log10k.n_elem == 0) [[unlikely]] {
      debug_fail = true;
    }
  }
  if (debug_fail) [[unlikely]] {
    spdlog::critical(
        "{}: incompatible input w/ k.size = {}, z.size = {}, "
        "and lnP.size = {}", "set_linear_power_spectrum", 
        io_log10k.n_elem, io_z.n_elem, io_lnP.n_elem
      );
    exit(1);
  }
  if(io_z.n_elem < 5 || io_log10k.n_elem < 5) [[unlikely]] {
    spdlog::critical(
        "{}: bad input w/ k.size = {}, z.size = {}, "
        "and lnP.size = {}", "set_linear_power_spectrum", 
        io_log10k.n_elem, io_z.n_elem, io_lnP.n_elem
      );
    exit(1);
  }

  int cache_update = 0;
  if (cosmology.lnPL_nk != static_cast<int>(io_log10k.n_elem) ||
      cosmology.lnPL_nz != static_cast<int>(io_z.n_elem) || 
      NULL == cosmology.lnPL) {
    cache_update = 1;
  }
  else {
    for (int i=0; i<cosmology.lnPL_nk; i++) {
      for (int j=0; j<cosmology.lnPL_nz; j++) {
        if (fdiff(cosmology.lnPL[i][j], io_lnP(i*cosmology.lnPL_nz+j))) {
          cache_update = 1; 
          goto jump;
        }
      }
    }
    for (int i=0; i<cosmology.lnPL_nk; i++) {
      if (fdiff(cosmology.lnPL[i][cosmology.lnPL_nz], io_log10k(i))) {
        cache_update = 1; 
        break;
      }
    }
    for (int j=0; j<cosmology.lnPL_nz; j++) {
      if (fdiff(cosmology.lnPL[cosmology.lnPL_nk][j], io_z(j))) {
        cache_update = 1; 
        break;
      }
    }
  }

  jump:
  if (1 == cache_update || 1 == force_cache_update_test) {
    cosmology.lnPL_nk = static_cast<int>(io_log10k.n_elem);
    cosmology.lnPL_nz = static_cast<int>(io_z.n_elem);

    if (cosmology.lnPL != NULL) {
      free(cosmology.lnPL);
    }
    cosmology.lnPL = (double**) malloc2d(cosmology.lnPL_nk+1,cosmology.lnPL_nz+1);

    #pragma omp parallel for
    for (int i=0; i<cosmology.lnPL_nk; i++) {
      if (std::isnan(io_log10k(i))) [[unlikely]] {
        // can't compile cosmolike with -O3 or -fast-math
        // see: https://stackoverflow.com/a/47703550/2472169
        spdlog::critical("{}: NaN found on interpolation table.", "set_linear_power_spectrum");
        exit(1);
      }
      cosmology.lnPL[i][cosmology.lnPL_nz] = io_log10k(i);
    }
    #pragma omp parallel for
    for (int j=0; j<cosmology.lnPL_nz; j++) {
      if (std::isnan(io_z(j))) [[unlikely]] {
        // can't compile cosmolike with -O3 or -fast-math
        // see: https://stackoverflow.com/a/47703550/2472169
        spdlog::critical("{}: NaN found on interpolation table.", "set_linear_power_spectrum");
        exit(1);
      }
      cosmology.lnPL[cosmology.lnPL_nk][j] = io_z(j);
    }
    #pragma omp parallel for collapse(2)
    for (int i=0; i<cosmology.lnPL_nk; i++) {
      for (int j=0; j<cosmology.lnPL_nz; j++) {
        if (std::isnan(io_lnP(i*cosmology.lnP_nz+j))) [[unlikely]] {
          // can't compile cosmolike with -O3 or -fast-math
          // see: https://stackoverflow.com/a/47703550/2472169
          spdlog::critical("{}: NaN found on interpolation table.", "set_linear_power_spectrum");
          exit(1);
        }
        cosmology.lnPL[i][j] = io_lnP(i*cosmology.lnPL_nz+j);
      }
    }
    cosmology.random = RandomNumber::get_instance().get();
  }

  spdlog::debug("{}: Ends", "set_linear_power_spectrum");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_non_linear_power_spectrum(
    arma::Col<double> io_log10k, 
    arma::Col<double> io_z, 
    arma::Col<double> io_lnP
  )
{
  spdlog::debug("{}: Begins", "set_non_linear_power_spectrum");

  bool debug_fail = false;
  if (io_z.n_elem*io_log10k.n_elem != io_lnP.n_elem) [[unlikely]] {
    debug_fail = true;
  }
  else {
    if (io_z.n_elem == 0) [[unlikely]] {
      debug_fail = true;
    }
  }
  if (debug_fail) [[unlikely]] {
    spdlog::critical(
        "{}: incompatible input w/ k.size = {}, z.size = {}, "
        "and lnP.size = {}", "set_non_linear_power_spectrum", 
        io_log10k.n_elem, io_z.n_elem, io_lnP.n_elem
      );
    exit(1);
  }
  if (io_z.n_elem < 5 || io_log10k.n_elem < 5) [[unlikely]] {
    spdlog::critical(
        "{}: bad input w/ k.size = {}, z.size = {}, "
        "and lnP.size = {}", "set_non_linear_power_spectrum", 
        io_log10k.n_elem, io_z.n_elem, io_lnP.n_elem
      );
    exit(1);
  }

  int cache_update = 0;
  if (cosmology.lnP_nk != static_cast<int>(io_log10k.n_elem) ||
      cosmology.lnP_nz != static_cast<int>(io_z.n_elem) || 
      NULL == cosmology.lnP) {
    cache_update = 1;
  }
  else
  {
    for (int i=0; i<cosmology.lnP_nk; i++) {
      for (int j=0; j<cosmology.lnP_nz; j++) {
        if (fdiff(cosmology.lnP[i][j], io_lnP(i*cosmology.lnP_nz+j))) {
          cache_update = 1; 
          goto jump;
        }
      }
    }
    for (int i=0; i<cosmology.lnP_nk; i++) {
      if (fdiff(cosmology.lnP[i][cosmology.lnP_nz], io_log10k(i))) {
        cache_update = 1; 
        break;
      }
    }
    for (int j=0; j<cosmology.lnP_nz; j++) {
      if (fdiff(cosmology.lnP[cosmology.lnP_nk][j], io_z(j))) {
        cache_update = 1; 
        break;
      }
    }
  }

  jump:

  if (1 == cache_update || 1 == force_cache_update_test) {
    cosmology.lnP_nk = static_cast<int>(io_log10k.n_elem);
    cosmology.lnP_nz = static_cast<int>(io_z.n_elem);
    if (cosmology.lnP != NULL) {
      free(cosmology.lnP);
    }
    cosmology.lnP = (double**) malloc2d(cosmology.lnP_nk+1,cosmology.lnP_nz+1);

    #pragma omp parallel for
    for (int i=0; i<cosmology.lnP_nk; i++) {
      if (std::isnan(io_log10k(i))) [[unlikely]] {
        // can't compile cosmolike with -O3 or -fast-math
        // see: https://stackoverflow.com/a/47703550/2472169
        spdlog::critical(
          "{}: NaN found on interpolation table.", 
          "set_non_linear_power_spectrum"
        );
        exit(1);
      }
      cosmology.lnP[i][cosmology.lnP_nz] = io_log10k(i);
    }
    #pragma omp parallel for
    for (int j=0; j<cosmology.lnP_nz; j++) {
      if (std::isnan(io_z(j))) [[unlikely]] {
        // can't compile cosmolike with -O3 or -fast-math
        // see: https://stackoverflow.com/a/47703550/2472169
        spdlog::critical(
          "{}: NaN found on interpolation table.", 
          "set_non_linear_power_spectrum"
        );
        exit(1);
      }
      cosmology.lnP[cosmology.lnP_nk][j] = io_z(j);
    }
    #pragma omp parallel for collapse(2)
    for (int i=0; i<cosmology.lnP_nk; i++) {
      for (int j=0; j<cosmology.lnP_nz; j++) {
        if (std::isnan(io_lnP(i*cosmology.lnP_nz+j))) [[unlikely]] {
          // can't compile cosmolike with -O3 or -fast-math
          // see: https://stackoverflow.com/a/47703550/2472169
          spdlog::critical(
            "{}: NaN found on interpolation table.", 
            "set_non_linear_power_spectrum"
          );
          exit(1);
        }
        cosmology.lnP[i][j] = io_lnP(i*cosmology.lnP_nz+j);
      }
    }
    cosmology.random = RandomNumber::get_instance().get();
  }
  spdlog::debug("{}: Ends", "set_non_linear_power_spectrum");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_nuisance_shear_calib(arma::Col<double> M)
{
  spdlog::debug("{}: Begins", "set_nuisance_shear_calib");
  if (0 == redshift.shear_nbin) {
    spdlog::critical(
        "{}: {} = 0 is invalid", 
        "set_nuisance_shear_calib", 
        "shear_Nbin"
      );
    exit(1);
  }
  if (redshift.shear_nbin != static_cast<int>(M.n_elem)) {
    spdlog::critical(
        "{}: incompatible input w/ size = {} (!= {})",
        "set_nuisance_shear_calib", 
        M.n_elem, 
        redshift.shear_nbin
      );
    exit(1);
  }
  for (int i=0; i<redshift.shear_nbin; i++) {
    if (std::isnan(M(i))) [[unlikely]] {
      // can't compile cosmolike with -O3 or -fast-math
      // see: https://stackoverflow.com/a/47703550/2472169
      spdlog::critical(
        "{}: NaN found on index {} ({}).", 
        "set_nuisance_shear_calib", 
        i,
        "common error if `params_values.get(p, None)` return None"
      );
      exit(1);
    }
    nuisance.shear_calibration_m[i] = M(i);
  }
  spdlog::debug("{}: Ends", "set_nuisance_shear_calib");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_nuisance_shear_photoz(arma::Col<double> SP)
{
  spdlog::debug("{}: Begins", "set_nuisance_shear_photoz");

  if (0 == redshift.shear_nbin) [[unlikely]] {
    spdlog::critical(
        "{}: {} = 0 is invalid",
        "set_nuisance_shear_photoz", 
        "shear_Nbin"
      );
    exit(1);
  }
  if (redshift.shear_nbin != static_cast<int>(SP.n_elem)) [[unlikely]] {
    spdlog::critical(
        "{}: incompatible input w/ size = {} (!= {})",
        "set_nuisance_shear_photoz", 
        SP.n_elem, 
        redshift.shear_nbin
      );
    exit(1);
  }

  int cache_update = 0;
  for (int i=0; i<redshift.shear_nbin; i++) {
    if (std::isnan(SP(i))) [[unlikely]] {
      // can't compile cosmolike with -O3 or -fast-math
      // see: https://stackoverflow.com/a/47703550/2472169
      spdlog::critical(
        "{}: NaN found on index {} ({}).", 
        "set_nuisance_shear_photoz", 
        i,
        "common error if `params_values.get(p, None)` return None"
      );
      exit(1);
    }
    if (fdiff(nuisance.photoz[0][0][i], SP(i))) {
      cache_update = 1;
      nuisance.photoz[0][0][i] = SP(i);
    } 
  }
  if (1 == cache_update || 1 == force_cache_update_test) {
    nuisance.random_photoz_shear = RandomNumber::get_instance().get();
  }
  spdlog::debug("{}: Ends", "set_nuisance_shear_photoz");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_nuisance_clustering_photoz(arma::Col<double> CP)
{
  spdlog::debug("{}: Begins", "set_nuisance_clustering_photoz");

  if (0 == redshift.clustering_nbin) [[unlikely]] {
    spdlog::critical(
        "{}: {} = 0 is invalid",
        "set_nuisance_clustering_photoz", 
        "clustering_Nbin"
      );
    exit(1);
  }
  if (redshift.clustering_nbin != static_cast<int>(CP.n_elem)) [[unlikely]] {
    spdlog::critical(
        "{}: incompatible input w/ size = {} (!= {})",
        "set_nuisance_clustering_photoz", 
        CP.n_elem, 
        redshift.clustering_nbin
      );
    exit(1);
  }

  int cache_update = 0;
  for (int i=0; i<redshift.clustering_nbin; i++)
  {
    if (std::isnan(CP(i))) [[unlikely]] {
      // can't compile cosmolike with -O3 or -fast-math
      // see: https://stackoverflow.com/a/47703550/2472169
      spdlog::critical(
        "{}: NaN found on index {} ({}).", 
        "set_nuisance_clustering_photoz", 
        i,
        "common error if `params_values.get(p, None)` return None"
      );
      exit(1);
    }
    if (fdiff(nuisance.photoz[1][0][i], CP(i))) { 
      cache_update = 1;
      nuisance.photoz[1][0][i] = CP(i);
    }
  }
  if (1 == cache_update || 1 == force_cache_update_test) {
    nuisance.random_photoz_clustering = RandomNumber::get_instance().get();
  }
  spdlog::debug("{}: Ends", "set_nuisance_clustering_photoz");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_nuisance_clustering_photoz_stretch(arma::Col<double> CPS)
{
  spdlog::debug("{}: Begins", "set_nuisance_clustering_photoz_stretch");

  if (0 == redshift.clustering_nbin) [[unlikely]] {
    spdlog::critical(
        "{}: {} = 0 is invalid",
        "set_nuisance_clustering_photoz_stretch",
        "clustering_Nbin"
      );
    exit(1);
  }
  if (redshift.clustering_nbin != static_cast<int>(CPS.n_elem)) [[unlikely]] {
    spdlog::critical(
        "{}: incompatible input w/ size = {} (!= {})",
        "set_nuisance_clustering_photoz_stretch",
        CPS.n_elem,
        redshift.clustering_nbin
      );
    exit(1);
  }

  int cache_update = 0;
  for (int i=0; i<redshift.clustering_nbin; i++) {
    if (std::isnan(CPS(i))) [[unlikely]] {
      // can't compile cosmolike with -O3 or -fast-math
      // see: https://stackoverflow.com/a/47703550/2472169
      spdlog::critical(
        "{}: NaN found on index {} ({}).", 
        "set_nuisance_clustering_photoz_stretch", 
        i,
        "common error if `params_values.get(p, None)` return None"
      );
      exit(1);
    }
    if (fdiff(nuisance.photoz[1][1][i], CPS(i))) {
      cache_update = 1;
      nuisance.photoz[1][1][i] = CPS(i);
    }
  }
  if (1 == cache_update || 1 == force_cache_update_test) {
    nuisance.random_photoz_clustering = RandomNumber::get_instance().get();
  }
  spdlog::debug("{}: Ends", "set_nuisance_clustering_photoz_stretch");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_nuisance_linear_bias(arma::Col<double> B1)
{
  spdlog::debug("{}: Begins", "set_nuisance_linear_bias");

  if (0 == redshift.clustering_nbin) [[unlikely]] {
    spdlog::critical(
        "{}: {} = 0 is invalid",
        "set_nuisance_linear_bias", "clustering_Nbin"
      );
    exit(1);
  }
  if (redshift.clustering_nbin != static_cast<int>(B1.n_elem)) [[unlikely]] {
    spdlog::critical(
        "{}: incompatible input w/ size = {} (!= {})",
        "set_nuisance_linear_bias", 
        B1.n_elem, 
        redshift.clustering_nbin
      );
    exit(1);
  }

  // GALAXY BIAS ------------------------------------------
  // 1st index: b[0][i] = linear galaxy bias in clustering bin i (b1)
  //            b[1][i] = linear galaxy bias in clustering bin i (b2)
  //            b[2][i] = leading order tidal bias in clustering bin i (b3)
  //            b[3][i] = leading order tidal bias in clustering bin i
  int cache_update = 0;
  for (int i=0; i<redshift.clustering_nbin; i++) {
    if (std::isnan(B1(i))) [[unlikely]] {
      // can't compile cosmolike with -O3 or -fast-math
      // see: https://stackoverflow.com/a/47703550/2472169
      spdlog::critical(
        "{}: NaN found on index {} ({}).", 
        "set_nuisance_linear_bias", 
        i,
        "common error if `params_values.get(p, None)` return None"
      );
      exit(1);
    }
    if(fdiff(nuisance.gb[0][i], B1(i))) {
      cache_update = 1;
      nuisance.gb[0][i] = B1(i);
    } 
  }
  if (1 == cache_update || 1 == force_cache_update_test) {
    nuisance.random_galaxy_bias = RandomNumber::get_instance().get();
  }
  spdlog::debug("{}: Ends", "set_nuisance_linear_bias");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_nuisance_nonlinear_bias(arma::Col<double> B1, arma::Col<double> B2)
{
  spdlog::debug("{}: Begins", "set_nuisance_nonlinear_bias");

  if (0 == redshift.clustering_nbin) [[unlikely]]{
    spdlog::critical(
        "{}: {} = 0 is invalid",
        "set_nuisance_nonlinear_bias", 
        "clustering_Nbin"
      );
    exit(1);
  }
  if (redshift.clustering_nbin != static_cast<int>(B1.n_elem) ||
      redshift.clustering_nbin != static_cast<int>(B2.n_elem)) [[unlikely]] {
    spdlog::critical(
      "{}: incompatible input w/ sizes = {} and {} (!= {})",
      "set_nuisance_nonlinear_bias", 
      B1.n_elem, B2.n_elem, redshift.clustering_nbin);
    exit(1);
  }

  // GALAXY BIAS ------------------------------------------
  // 1st index: b[0][i]: linear galaxy bias in clustering bin i
  //            b[1][i]: nonlinear b2 galaxy bias in clustering bin i
  //            b[2][i]: leading order tidal bs2 galaxy bias in clustering bin i
  //            b[3][i]: nonlinear b3 galaxy bias  in clustering bin i 
  //            b[4][i]: amplitude of magnification bias in clustering bin i 
  int cache_update = 0;
  for (int i=0; i<redshift.clustering_nbin; i++) {
    if (std::isnan(B1(i)) || std::isnan(B2(i))) [[unlikely]] {
      // can't compile cosmolike with -O3 or -fast-math
      // see: https://stackoverflow.com/a/47703550/2472169
      spdlog::critical(
          "{}: NaN found on index {} ({}).", 
          "set_nuisance_nonlinear_bias", 
          i,
          "common error if `params_values.get(p, None)` return None"
        );
      exit(1);
    }
    if(fdiff(nuisance.gb[1][i], B2(i))) {
      cache_update = 1;
      nuisance.gb[1][i] = B2(i);
      nuisance.gb[2][i] = almost_equal(B2(i), 0.) ? 0 : (-4./7.)*(B1(i)-1.0);
    }
  }
  if (1 == cache_update || 1 == force_cache_update_test) {
    nuisance.random_galaxy_bias = RandomNumber::get_instance().get();
  }
  spdlog::debug("{}: Ends", "set_nuisance_nonlinear_bias");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_nuisance_magnification_bias(arma::Col<double> B_MAG)
{
  spdlog::debug("{}: Begins", "set_nuisance_magnification_bias");

  if (0 == redshift.clustering_nbin) [[unlikely]] {
    spdlog::critical(
        "{}: {} = 0 is invalid",
        "set_nuisance_magnification_bias", 
        "clustering_Nbin"
      );
    exit(1);
  }
  if (redshift.clustering_nbin != static_cast<int>(B_MAG.n_elem)) [[unlikely]] {
    spdlog::critical(
        "{}: incompatible input w/ size = {} (!= {})",
        "set_nuisance_magnification_bias", 
        B_MAG.n_elem, redshift.clustering_nbin
      );
    exit(1);
  }

  // GALAXY BIAS ------------------------------------------
  // 1st index: b[0][i]: linear galaxy bias in clustering bin i
  //            b[1][i]: nonlinear b2 galaxy bias in clustering bin i
  //            b[2][i]: leading order tidal bs2 galaxy bias in clustering bin i
  //            b[3][i]: nonlinear b3 galaxy bias  in clustering bin i 
  //            b[4][i]: amplitude of magnification bias in clustering bin i
  int cache_update = 0;
  for (int i=0; i<redshift.clustering_nbin; i++) {
    if (std::isnan(B_MAG(i))) [[unlikely]] {
      // can't compile cosmolike with -O3 or -fast-math
      // see: https://stackoverflow.com/a/47703550/2472169
      spdlog::critical(
        "{}: NaN found on index {} ({}).", 
        "set_nuisance_magnification_bias", 
        i,
        "common error if `params_values.get(p, None)` return None"
      );
      exit(1);
    }
    if(fdiff(nuisance.gb[4][i], B_MAG(i))) {
      cache_update = 1;
      nuisance.gb[4][i] = B_MAG(i);
    }
  }
  if(1 == cache_update || 1 == force_cache_update_test) {
    nuisance.random_galaxy_bias = RandomNumber::get_instance().get();
  }
  spdlog::debug("{}: Ends", "set_nuisance_magnification_bias");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_nuisance_bias(
    arma::Col<double> B1, 
    arma::Col<double> B2, 
    arma::Col<double> B_MAG
  )
{
  set_nuisance_linear_bias(B1);
  
  set_nuisance_nonlinear_bias(B1, B2);
  
  set_nuisance_magnification_bias(B_MAG);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_nuisance_IA(
    arma::Col<double> A1, 
    arma::Col<double> A2,
    arma::Col<double> BTA
  )
{
  spdlog::debug("{}: Begins", "set_nuisance_IA");

  if (0 == redshift.shear_nbin) [[unlikely]] {
    spdlog::critical(
        "{}: {} = 0 is invalid",
        "set_nuisance_IA", 
        "shear_Nbin"
      );
    exit(1);
  }
  if (redshift.shear_nbin > static_cast<int>(A1.n_elem) ||
      redshift.shear_nbin > static_cast<int>(A2.n_elem) ||
      redshift.shear_nbin > static_cast<int>(BTA.n_elem)) [[unlikely]] {
    spdlog::critical(
        "{}: incompatible input w/ sizes = {:d}, {:d} and {:d} (!= {:d})",
        "set_nuisance_IA", 
        A1.n_elem, 
        A2.n_elem, 
        BTA.n_elem, 
        redshift.shear_nbin
      );
    exit(1);
  }

  // INTRINSIC ALIGMENT ------------------------------------------  
  // ia[0][0] = A_ia          if(IA_NLA_LF || IA_REDSHIFT_EVOLUTION)
  // ia[0][1] = eta_ia        if(IA_NLA_LF || IA_REDSHIFT_EVOLUTION)
  // ia[0][2] = eta_ia_highz  if(IA_NLA_LF, Joachimi2012)
  // ia[0][3] = beta_ia       if(IA_NLA_LF, Joachimi2012)
  // ia[0][4] = LF_alpha      if(IA_NLA_LF, Joachimi2012)
  // ia[0][5] = LF_P          if(IA_NLA_LF, Joachimi2012)
  // ia[0][6] = LF_Q          if(IA_NLA_LF, Joachimi2012)
  // ia[0][7] = LF_red_alpha  if(IA_NLA_LF, Joachimi2012)
  // ia[0][8] = LF_red_P      if(IA_NLA_LF, Joachimi2012)
  // ia[0][9] = LF_red_Q      if(IA_NLA_LF, Joachimi2012)
  // ------------------
  // ia[1][0] = A2_ia        if IA_REDSHIFT_EVOLUTION
  // ia[1][1] = eta_ia_tt    if IA_REDSHIFT_EVOLUTION
  // ------------------
  // ia[2][MAX_SIZE_ARRAYS] = b_ta_z[MAX_SIZE_ARRAYS]

  int cache_update = 0;
  nuisance.c1rhocrit_ia = 0.01389;
  
  if (nuisance.IA == IA_REDSHIFT_BINNING)
  {
    for (int i=0; i<redshift.shear_nbin; i++) {
      if (std::isnan(A1(i)) || 
          std::isnan(A2(i)) || 
          std::isnan(BTA(i))) [[unlikely]] 
      {
        // can't compile cosmolike with -O3 or -fast-math
        // see: https://stackoverflow.com/a/47703550/2472169
        spdlog::critical(
            "{}: NaN found on index {} ({}).", 
            "set_nuisance_ia", 
            i,
            "common error if `params_values.get(p, None)` return None"
          );
        exit(1);
      }
      if (fdiff(nuisance.ia[0][i],A1(i)) ||
          fdiff(nuisance.ia[1][i],A2(i)) ||
          fdiff(nuisance.ia[2][i],A2(i)))
      {
        nuisance.ia[0][i] = A1(i);
        nuisance.ia[1][i] = A2(i);
        nuisance.ia[2][i] = BTA(i);
        cache_update = 1;
      }
    }
  }
  else if (nuisance.IA == IA_REDSHIFT_EVOLUTION)
  {
    nuisance.oneplusz0_ia = 1.62;
    if (fdiff(nuisance.ia[0][0],A1(0)) ||
        fdiff(nuisance.ia[0][1],A1(1)) ||
        fdiff(nuisance.ia[1][0],A2(0)) ||
        fdiff(nuisance.ia[1][1],A2(1)) ||
        fdiff(nuisance.ia[2][0],BTA(0)))
    {
      nuisance.ia[0][0] = A1(0);
      nuisance.ia[0][1] = A1(1);
      nuisance.ia[1][0] = A2(0);
      nuisance.ia[1][1] = A2(1);
      nuisance.ia[2][0] = BTA(0);
      cache_update = 1;
    }
  }
  if(1 == cache_update || 1 == force_cache_update_test) {
    nuisance.random_ia = RandomNumber::get_instance().get();
  }

  spdlog::debug("{}: Ends", "set_nuisance_ia");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_lens_sample_size(const int Ntomo)
{
  if (std::isnan(Ntomo) || 
      !(Ntomo > 0) || 
      Ntomo > MAX_SIZE_ARRAYS) [[unlikely]] {
    spdlog::critical(
        "{}: {} = {} not supported (max = {})", 
        "set_lens_sample_size", 
        "Ntomo", 
        Ntomo, 
        MAX_SIZE_ARRAYS
      );
    exit(1);
  }
  redshift.clustering_photoz = 4;
  redshift.clustering_nbin = Ntomo;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_lens_sample(arma::Mat<double> input_table)
{
  spdlog::debug("{}: Begins", "set_lens_sample");

  const int Ntomo = redshift.clustering_nbin;
  if (std::isnan(Ntomo) || 
      !(Ntomo > 0) || 
      Ntomo > MAX_SIZE_ARRAYS) [[unlikely]] {
    spdlog::critical(
        "{}: {} = {} not supported (max = {})", 
        "set_lens_sample_size", 
        "Ntomo", 
        Ntomo, 
        MAX_SIZE_ARRAYS
      );
    exit(1);
  }

  int cache_update = 0;
  if (redshift.clustering_nzbins != static_cast<int>(input_table.n_rows) ||
      NULL == redshift.clustering_zdist_table) {
    cache_update = 1;
  }
  else
  {
    for (int i=0; i<redshift.clustering_nzbins; i++) {
      double** tab = redshift.clustering_zdist_table;        // alias
      double* z_v = redshift.clustering_zdist_table[Ntomo];  // alias

      if (fdiff(z_v[i], input_table(i,0))) {
        cache_update = 1;
        break;
      }
      for (int k=0; k<Ntomo; k++) {  
        if (fdiff(tab[k][i], input_table(i,k+1))) {
          cache_update = 1;
          goto jump;
        }
      }
    }
  }

  jump:

  if (1 == cache_update || 1 == force_cache_update_test)
  {
    redshift.clustering_nzbins = input_table.n_rows;
    const int nzbins = redshift.clustering_nzbins;    // alias

    if (redshift.clustering_zdist_table != NULL) {
      free(redshift.clustering_zdist_table);
    }
    redshift.clustering_zdist_table = (double**) malloc2d(Ntomo + 1, nzbins);
    
    double** tab = redshift.clustering_zdist_table;        // alias
    double* z_v = redshift.clustering_zdist_table[Ntomo];  // alias
    
    for (int i=0; i<nzbins; i++) {
      z_v[i] = input_table(i,0);
      for (int k=0; k<Ntomo; k++) {
        tab[k][i] = input_table(i,k+1);
      }
    }
    
    redshift.clustering_zdist_zmin_all = fmax(z_v[0], 1.e-5);
    
    redshift.clustering_zdist_zmax_all = z_v[nzbins-1] + 
      (z_v[nzbins-1] - z_v[0]) / ((double) nzbins - 1.);

    for (int k=0; k<Ntomo; k++) { // Set tomography bin boundaries
      auto nofz = input_table.col(k+1).eval();
      
      arma::uvec idx = arma::find(nofz > 0.999e-8*nofz.max());
      
      redshift.clustering_zdist_zmin[k] = z_v[idx(0)];
      
      redshift.clustering_zdist_zmax[k] = z_v[idx(idx.n_elem-1)];
    }
    // READ THE N(Z) FILE ENDS ------------
    redshift.random_clustering = RandomNumber::get_instance().get();

    pf_photoz(0.1, 0); // init static variables

    for (int k=0; k<Ntomo; k++) {
      redshift.clustering_zdist_zmean[k] = zmean(k);
      spdlog::debug(
          "{}: bin {} - {} = {}.",
          "set_lens_sample",
          k,
          "<z_s>",
          redshift.clustering_zdist_zmean[k]
        );
    }
  }

  spdlog::debug("{}: Ends", "set_lens_sample");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_source_sample_size(const int Ntomo)
{
  if (std::isnan(Ntomo) || 
      !(Ntomo > 0) || 
      Ntomo > MAX_SIZE_ARRAYS) [[unlikely]] {
    spdlog::critical(
        "{}: {} = {} not supported (max = {})", 
        "set_source_sample_size", 
        "Ntomo", 
        Ntomo, 
        MAX_SIZE_ARRAYS
      );
    exit(1);
  } 
  redshift.shear_photoz = 4;
  redshift.shear_nbin = Ntomo;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_source_sample(arma::Mat<double> input_table)
{
  spdlog::debug("{}: Begins", "set_source_sample");

  const int Ntomo = redshift.shear_nbin;
  if (std::isnan(Ntomo) ||  
      !(Ntomo > 0) || 
      Ntomo > MAX_SIZE_ARRAYS) [[unlikely]] {
    spdlog::critical(
        "{}: {} = {} not supported (max = {})", 
        "set_source_sample", 
        "Ntomo", 
        Ntomo, 
        MAX_SIZE_ARRAYS
      );
    exit(1);
  } 

  int cache_update = 0;
  if (redshift.shear_nzbins != static_cast<int>(input_table.n_rows) ||
      NULL == redshift.shear_zdist_table) {
    cache_update = 1;
  }
  else
  {
    double** tab = redshift.shear_zdist_table;         // alias  
    double* z_v  = redshift.shear_zdist_table[Ntomo];  // alias
    for (int i=0; i<redshift.shear_nzbins; i++)  {
      if (fdiff(z_v[i], input_table(i,0))) {
        cache_update = 1;
        goto jump;
      }
      for (int k=0; k<Ntomo; k++) {
        if (fdiff(tab[k][i], input_table(i,k+1))) {
          cache_update = 1;
          goto jump;
        }
      }
    }
  }

  jump:

  if (1 == cache_update || 1 == force_cache_update_test)
  {
    redshift.shear_nzbins = input_table.n_rows;
    const int nzbins = redshift.shear_nzbins; // alias

    if (redshift.shear_zdist_table != NULL) {
      free(redshift.shear_zdist_table);
    }
    redshift.shear_zdist_table = (double**) malloc2d(Ntomo + 1, nzbins);

    double** tab = redshift.shear_zdist_table;        // alias  
    double* z_v = redshift.shear_zdist_table[Ntomo];  // alias
    for (int i=0; i<nzbins; i++) {
      z_v[i] = input_table(i,0);
      for (int k=0; k<Ntomo; k++) {
        tab[k][i] = input_table(i,k+1);
      }
    }
  
    redshift.shear_zdist_zmin_all = fmax(z_v[0], 1.e-5);
    redshift.shear_zdist_zmax_all = z_v[nzbins-1] + 
      (z_v[nzbins-1] - z_v[0]) / ((double) nzbins - 1.);

    for (int k=0; k<Ntomo; k++) 
    { // Set tomography bin boundaries
      auto nofz = input_table.col(k+1).eval();
      
      arma::uvec idx = arma::find(nofz > 0.999e-8*nofz.max());
      redshift.shear_zdist_zmin[k] = fmax(z_v[idx(0)], 1.001e-5);
      redshift.shear_zdist_zmax[k] = z_v[idx(idx.n_elem-1)];
    }
  
    // READ THE N(Z) FILE ENDS ------------
    if (redshift.shear_zdist_zmax_all < redshift.shear_zdist_zmax[Ntomo-1] || 
        redshift.shear_zdist_zmin_all > redshift.shear_zdist_zmin[0]) [[unlikely]] {
      spdlog::critical(
          "zhisto_min = {},zhisto_max = {}", 
          redshift.shear_zdist_zmin_all, 
          redshift.shear_zdist_zmax_all
        );
      spdlog::critical(
          "shear_zdist_zmin[0] = {},"
          " shear_zdist_zmax[redshift.shear_nbin-1] = {}", 
          redshift.shear_zdist_zmin[0], 
          redshift.shear_zdist_zmax[Ntomo-1]
        );
      exit(1);
    } 

    zdistr_photoz(0.1, 0); // init static variables

    for (int k=0; k<Ntomo; k++) {
      spdlog::debug(
          "{}: bin {} - {} = {}.",
          "set_source_sample",
          k,
          "<z_s>",
          zmean_source(k)
        );
    }
    redshift.random_shear = RandomNumber::get_instance().get();
  }

  spdlog::debug("{}: Ends", "set_source_sample");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GET FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double get_baryon_power_spectrum_ratio(const double log10k, const double a)
{
  const double KNL = pow(10.0, log10k)*cosmology.coverH0;
  return PkRatio_baryons(KNL, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// COMPUTE FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double compute_pm(const int zl, const int zs, const double theta)
{
  return PointMass::get_instance().get_pm(zl, zs, theta);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

vector compute_binning_real_space()
{
  spdlog::debug("{}: Begins", "get_binning_real_space");
  if (0 == Ntable.Ntheta)  [[unlikely]] {
    spdlog::critical(
        "{}: {} not set (or ill-defined) prior to this function call",
        "get_binning_real_space", 
        "Ntable.Ntheta"
      );
    exit(1);
  }
  if (!(Ntable.vtmax > Ntable.vtmin))  [[unlikely]] {
    spdlog::critical(
        "{}: {} not set (or ill-defined) prior to this function call",
        "get_binning_real_space", 
        "Ntable.vtmax and Ntable.vtmin"
      );
    exit(1);
  }
  const double logvtmin = std::log(Ntable.vtmin);
  const double logvtmax = std::log(Ntable.vtmax);
  const double logdt=(logvtmax - logvtmin)/Ntable.Ntheta;
  constexpr double fac = (2./3.);

  arma::Col<double> theta(Ntable.Ntheta, arma::fill::zeros);
  for (int i=0; i<Ntable.Ntheta; i++) {
    const double thetamin = std::exp(logvtmin + (i + 0.)*logdt);
    const double thetamax = std::exp(logvtmin + (i + 1.)*logdt);
    theta(i) = fac * (std::pow(thetamax,3) - std::pow(thetamin,3)) /
                     (thetamax*thetamax    - thetamin*thetamin);
    spdlog::debug(
        "{}: Bin {:d} - {} = {:.4e}, {} = {:.4e} and {} = {:.4e}",
        "init_binning_real_space", 
        i, 
        "theta_min [rad]", 
        thetamin, 
        "theta [rad]", 
        theta(i), 
        "theta_max [rad]", 
        thetamax
      );
  }
  return theta;
  spdlog::debug("{}: Ends", "get_binning_real_space");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

vector compute_add_baryons_pcs(vector Q, vector dv)
{
  spdlog::debug("{}: Begins", "compute_add_baryons_pcs");
  BaryonScenario& bs = BaryonScenario::get_instance();
  if (!bs.is_pcs_set()) [[unlikely]] {
    spdlog::critical(
        "{}: {} not set prior to this function call",
        "compute_add_baryons_pcs", "baryon PCs"
      );
    exit(1);
  }
  if (bs.get_pcs().row(0).n_elem < Q.n_elem) [[unlikely]] {
    spdlog::critical(
        "{}: invalid PC amplitude vector or PC eigenvectors",
        "compute_add_baryons_pcs"
      );
    exit(1);
  }
  if (bs.get_pcs().col(0).n_elem != dv.n_elem) [[unlikely]] {
    spdlog::critical(
        "{}: invalid datavector or PC eigenvectors",
        "compute_add_baryons_pcs"
      );
    exit(1);
  }
  for (int j=0; j<static_cast<int>(dv.n_elem); j++) {
    for (int i=0; i<static_cast<int>(Q.n_elem); i++) {
      if (IP::get_instance().get_mask(j)) {
        dv(j) += Q(i) * bs.get_pcs(j, i);
      }
    }
  }
  spdlog::debug("{}: Ends", "compute_add_baryons_pcs");
  return dv;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Class IP MEMBER FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void IP::set_data(std::string datavector_filename)
{
  if (!(this->is_mask_set_)) {
    spdlog::critical(
        "{}: {} not set prior to this function call", 
        "set_data",
        "mask"
      );
    exit(1);
  }

  this->data_masked_.set_size(this->ndata_);
    
  this->data_masked_sqzd_.set_size(this->ndata_sqzd_);

  this->data_filename_ = datavector_filename;

  matrix table = read_table(datavector_filename);
  if (static_cast<int>(table.n_rows) != this->ndata_) {
    spdlog::critical("{}: inconsistent data vector", "IP::set_data");
    exit(1);
  }
  for(int i=0; i<like.Ndata; i++) {
    this->data_masked_(i) = table(i,1);
    this->data_masked_(i) *= this->get_mask(i);
    if(this->get_mask(i) == 1) {
      if(this->get_index_sqzd(i) < 0) {
        spdlog::critical(
            "{}: logical error, internal"
            " inconsistent mask operation", 
            "IP::set_data"
          );
        exit(1);
      }
      this->data_masked_sqzd_(this->get_index_sqzd(i)) = this->data_masked_(i);
    }
  }
  this->is_data_set_ = true;
}

void IP::set_mask(
    std::string mask_filename, 
    arma::Col<int>::fixed<3> order, 
    const int real_space
  )
{
  if (!(like.Ndata>0)) {
    spdlog::critical(
        "{}: {} not set prior to this function call",
        "IP::set_mask",
        "like.Ndata"
      );
    exit(1);
  }

  this->ndata_ = like.Ndata;  
  
  this->mask_.set_size(this->ndata_);
  
  this->mask_filename_ = mask_filename;

  matrix table = read_table(mask_filename);
  if (static_cast<int>(table.n_rows) != this->ndata_) {
    spdlog::critical("{}: inconsistent mask", "IP::set_mask");
    exit(1);
  }
  
  for (int i=0; i<this->ndata_; i++) {
    this->mask_(i) = static_cast<int>(table(i,1) + 1e-13);
    if (!(0 == this->mask_(i) || 1 == this->mask_(i))) {
      spdlog::critical("{}: inconsistent mask", "IP::set_mask");
      exit(1);
    }
  }

  arma::Col<int>::fixed<3> sizes = (1 == real_space) ? 
    compute_data_vector_3x2pt_real_sizes() :
    compute_data_vector_3x2pt_fourier_sizes();

  arma::Col<int>::fixed<3> start = (1 == real_space) ? 
    compute_data_vector_3x2pt_real_starts(order) :
    compute_data_vector_3x2pt_fourier_starts(order);

  if (0 == like.shear_shear) {
    const int N = start(0);
    const int M = N + sizes(0);
    for (int i=N; i<M; i++) {
      this->mask_(i) = 0;
    }
  }
  if (0 == like.shear_pos) {
    const int N = start(1);
    const int M = N + sizes(1);
    for (int i=N; i<M; i++) {
      this->mask_(i) = 0;
    }
  }
  if (0 == like.pos_pos) {
    const int N = start(2);
    const int M = N + sizes(2);
    for (int i=N; i<M; i++) {
      this->mask_(i) = 0;
    }
  }

  this->ndata_sqzd_ = arma::accu(this->mask_);
  
  if (!(this->ndata_sqzd_>0)) {
    spdlog::critical(
        "{}: mask file {} left no data points after masking",
        "IP::set_mask", mask_filename
      );
    exit(1);
  }
  spdlog::debug(
      "{}: mask file {} left {} non-masked elements after masking",
      "IP::set_mask", mask_filename, this->ndata_sqzd_
    );

  this->index_sqzd_.set_size(this->ndata_);
  {
    double j=0;
    for (int i=0; i<this->ndata_; i++) {
      if(this->get_mask(i) > 0) {
        this->index_sqzd_(i) = j;
        j++;
      }
      else {
        this->index_sqzd_(i) = -1;
      }
    }
    if(j != this->ndata_sqzd_) {
      spdlog::critical(
          "{}: logical error, internal inconsistent mask operation",
          "IP::set_mask"
        );
      exit(1);
    }
  }
  this->is_mask_set_ = true;
}

void IP::set_mask(
    std::string mask_filename,
    arma::Col<int>::fixed<6> order,
    const int real_space
  )
{
  if (!(like.Ndata>0)) [[unlikely]] {
    spdlog::critical(
        "{}: {} not set prior to this function call",
        "IP::set_mask", 
        "like.Ndata"
      );
    exit(1);
  }

  this->ndata_ = like.Ndata;

  this->mask_.set_size(this->ndata_);

  this->mask_filename_ = mask_filename;

  matrix table = read_table(mask_filename);
  if (static_cast<int>(table.n_rows) != this->ndata_) [[unlikely]] {
    spdlog::critical("{}: inconsistent mask", "IP::set_mask");
    exit(1);
  }

  for (int i=0; i<this->ndata_; i++) {
    this->mask_(i) = static_cast<int>(table(i,1)+1e-13);
    if (!(0 == this->mask_(i) || 1 == this->mask_(i))) [[unlikely]] {
      spdlog::critical("{}: inconsistent mask", "IP::set_mask");
      exit(1);
    }
  }

  arma::Col<int>::fixed<6> sizes = (1 == real_space) ?
    compute_data_vector_6x2pt_real_sizes() :
    compute_data_vector_6x2pt_fourier_sizes();

  arma::Col<int>::fixed<6> start = (1 == real_space) ?
    compute_data_vector_6x2pt_real_starts(order) :
    compute_data_vector_6x2pt_fourier_starts(order);

  if (0 == like.shear_shear) {
    const int N = start(0);
    const int M = N + sizes(0);
    for (int i=N; i<M; i++) {
      this->mask_(i) = 0;
    }
  }
  if (0 == like.shear_pos) {
    const int N = start(1);
    const int M = N + sizes(1);
    for (int i=N; i<M; i++) {
      this->mask_(i) = 0;
    }
  }
  if (0 == like.pos_pos) {
    const int N = start(2);
    const int M = N + sizes(2);
    for (int i=N; i<M; i++) {
      this->mask_(i) = 0;
    }
  }
  if (0 == like.gk) {
    const int N = start(3);
    const int M = N + sizes(3);;
    for (int i=N; i<M; i++) {
      this->mask_(i) = 0.0;
    }
  }
  if (0 == like.ks)  {
    const int N = start(4);
    const int M = N + sizes(4);
    for (int i=N; i<M; i++) {
      this->mask_(i) = 0.0;
    }
  }
  if (0 == like.kk) {
    const int N = start(5);
    const int M = N + sizes(5);
    for (int i=N; i<M; i++) {
      this->mask_(i) = 0.0;
    }
  }
  
  this->ndata_sqzd_ = arma::accu(this->mask_);
  
  if(!(this->ndata_sqzd_>0)) [[unlikely]] {
    spdlog::critical(
        "{}: mask file {} left no data points after masking",
        "IP::set_mask", 
        mask_filename
      );
    exit(1);
  }
  spdlog::debug(
      "{}: mask file {} left {} non-masked elements "
      "after masking",
      "IP::set_mask", 
      mask_filename, 
      this->ndata_sqzd_
    );

  this->index_sqzd_.set_size(this->ndata_);
  {
    double j=0;
    for(int i=0; i<this->ndata_; i++) {
      if(this->get_mask(i) > 0) {
        this->index_sqzd_(i) = j;
        j++;
      }
      else {
        this->index_sqzd_(i) = -1;
      }
    }
    if(j != this->ndata_sqzd_) [[unlikely]] {
      spdlog::critical(
          "{}: logical error, internal "
          "inconsistent mask operation", "IP::set_mask"
        );
      exit(1);
    }
  }
  this->is_mask_set_ = true;
}

void IP::set_inv_cov(std::string covariance_filename)
{
  if (!(this->is_mask_set_)) [[unlikely]] {
    spdlog::critical(
        "{}: {} not set prior to this function call",
        "IP::set_inv_cov",
        "mask"
      );
    exit(1);
  }

  this->cov_filename_ = covariance_filename;
  matrix table = read_table(covariance_filename); 
  
  this->cov_masked_.set_size(this->ndata_, this->ndata_);
  this->cov_masked_.zeros();
  this->cov_masked_sqzd_.set_size(this->ndata_sqzd_, this->ndata_sqzd_);
  this->inv_cov_masked_sqzd_.set_size(this->ndata_sqzd_, this->ndata_sqzd_);

  switch (table.n_cols)
  {
    case 3:
    {
      #pragma omp parallel for
      for (int i=0; i<static_cast<int>(table.n_rows); i++) {
        const int j = static_cast<int>(table(i,0));
        const int k = static_cast<int>(table(i,1));
        this->cov_masked_(j,k) = table(i,2);
        if (j!=k) {
          // apply mask to off-diagonal covariance elements
          this->cov_masked_(j,k) *= this->get_mask(j);
          this->cov_masked_(j,k) *= this->get_mask(k);
          // m(i,j) = m(j,i)
          this->cov_masked_(k,j) = this->cov_masked_(j,k);
        }
      };
      break;
    }
    case 4:
    {
      #pragma omp parallel for
      for (int i=0; i<static_cast<int>(table.n_rows); i++) {
        const int j = static_cast<int>(table(i,0));
        const int k = static_cast<int>(table(i,1));
        this->cov_masked_(j,k) = table(i,2) + table(i,3);
        if (j!=k) {
          // apply mask to off-diagonal covariance elements
          this->cov_masked_(j,k) *= this->get_mask(j);
          this->cov_masked_(j,k) *= this->get_mask(k);
          // m(i,j) = m(j,i)
          this->cov_masked_(k,j) = this->cov_masked_(j,k);
        }
      };
      break;
    }
    case 10:
    {
      #pragma omp parallel for
      for (int i=0; i<static_cast<int>(table.n_rows); i++) {
        const int j = static_cast<int>(table(i,0));
        const int k = static_cast<int>(table(i,1));
        this->cov_masked_(j,k) = table(i,8) + table(i,9);
        if (j!=k) {
          // apply mask to off-diagonal covariance elements
          this->cov_masked_(j,k) *= this->get_mask(j);
          this->cov_masked_(j,k) *= this->get_mask(k);
          // m(i,j) = m(j,i)
          this->cov_masked_(k,j) = this->cov_masked_(j,k);
        }
      }
      break;
    }
    default:
    {
      spdlog::critical(
          "{}: data format for covariance file = {} is invalid",
          "IP::set_inv_cov", 
          covariance_filename
        );
      exit(1);
    }
  }

  if (1 == IPCMB::get_instance().is_kk_bandpower())
  {
    IPCMB& cmb = IPCMB::get_instance();
    const int N5x2pt = this->ndata_ - cmb.get_nbins_kk_bandpower();
    if (!(N5x2pt>0)) [[unlikely]] {
      spdlog::critical(
          "{}, {}: inconsistent dv size and number of binning in (kappa-kappa)",
          "IP::set_inv_cov", this->ndata_, cmb.get_nbins_kk_bandpower()
        );
      exit(1);
    }
    const double hartlap_factor = cmb.get_alpha_Hartlap_cov_kkkk();
    #pragma omp parallel for collapse(2)
    for (int i=N5x2pt; i<this->ndata_; i++) {
      for (int j=N5x2pt; j<this->ndata_; j++) {
        this->cov_masked_(i,j) /= hartlap_factor;
      }
    }
  }

  vector eigvals = arma::eig_sym(this->cov_masked_);
  for(int i=0; i<this->ndata_; i++) {
    if(eigvals(i) < 0) [[unlikely]] {
      spdlog::critical(
          "{}: masked cov not positive definite", 
          "IP::set_inv_cov"
        );
      exit(-1);
    }
  }

  this->inv_cov_masked_ = arma::inv(this->cov_masked_);

  // apply mask again to make sure numerical errors in matrix inversion don't 
  // cause problems. Also, set diagonal elements corresponding to datavector
  // elements outside mask to 0, so that they don't contribute to chi2
  #pragma omp parallel for
  for (int i=0; i<this->ndata_; i++) {
    this->inv_cov_masked_(i,i) *= this->get_mask(i)*this->get_mask(i);
    for (int j=0; j<i; j++) {
      this->inv_cov_masked_(i,j) *= this->get_mask(i)*this->get_mask(j);
      this->inv_cov_masked_(j,i) = this->inv_cov_masked_(i,j);
    }
  };
  
  #pragma omp parallel for collapse(2)
  for(int i=0; i<this->ndata_; i++)
  {
    for(int j=0; j<this->ndata_; j++)
    {
      if((this->mask_(i)>0.99) && (this->mask_(j)>0.99)) {
        if(this->get_index_sqzd(i) < 0) [[unlikely]] {
          spdlog::critical(
              "{}: logical error, internal inconsistent mask operation", 
              "IP::set_inv_cov"
            );
          exit(1);
        }
        if(this->get_index_sqzd(j) < 0) [[unlikely]] {
          spdlog::critical(
              "{}: logical error, internal inconsistent mask operation", 
              "IP::set_inv_cov"
            );
          exit(1);
        }
        const int idxa = this->get_index_sqzd(i);
        const int idxb = this->get_index_sqzd(j);
        this->cov_masked_sqzd_(idxa,idxb) = this->cov_masked_(i,j);
        this->inv_cov_masked_sqzd_(idxa,idxb) = this->inv_cov_masked_(i,j);
      }
    }
  }
  this->is_inv_cov_set_ = true;
}

int IP::get_mask(const int ci) const
{
  if (ci > like.Ndata || ci < 0) [[unlikely]] {
    spdlog::critical(
        "{}: index i = {} is not valid (min = {}, max = {})",
        "IP::get_mask", ci, 0, like.Ndata
      );
    exit(1);
  }
  return this->mask_(ci);
}

int IP::get_index_sqzd(const int ci) const
{
  if (ci > like.Ndata || ci < 0) [[unlikely]] {
    spdlog::critical(
        "{}: index i = {} is not valid (min = {}, max = {})", 
        "IP::get_index_sqzd", ci, 0, like.Ndata
      );
    exit(1);
  }
  return this->index_sqzd_(ci);
}

double IP::get_dv_masked(const int ci) const
{
  if (ci > like.Ndata || ci < 0) [[unlikely]] {
    spdlog::critical(
        "{}: index i = {} is not valid (min = {}, max = {})",
        "IP::get_dv_masked", ci, 0, like.Ndata
      );
    exit(1);
  }
  return this->data_masked_(ci);
}

double IP::get_dv_masked_sqzd(const int ci) const
{
  if (ci > like.Ndata || ci < 0) [[unlikely]] {
    spdlog::critical(
        "{}: index i = {} is not valid (min = {}, max = {})",
        "IP::get_dv_masked_sqzd", ci, 0, like.Ndata
      );
    exit(1);
  }
  return this->data_masked_sqzd_(ci);
}

double IP::get_inv_cov_masked(const int ci, const int cj) const
{
  if (ci > like.Ndata || ci < 0) [[unlikely]] {
    spdlog::critical(
        "{}: index i = {} is not valid (min = {}, max = {})",
        "IP::get_inv_cov_masked", ci, 0, like.Ndata);
    exit(1);
  }
  if (cj > like.Ndata || cj < 0) [[unlikely]] {
    spdlog::critical(
        "{}: index j = {} is not valid (min = {}, max = {})",
        "IP::get_inv_cov_masked",  cj,  0,  like.Ndata
      );
    exit(1);
  }
  return this->inv_cov_masked_(ci, cj);
}

double IP::get_inv_cov_masked_sqzd(const int ci, const int cj) const
{
  if (ci > like.Ndata || ci < 0) [[unlikely]] {
    spdlog::critical(
        "{}: index i = {} is not valid (min = {}, max = {})",
        "IP::get_inv_cov_masked_sqzd", ci,  0, like.Ndata);
    exit(1);
  }
  if (cj > like.Ndata || cj < 0) [[unlikely]] {
    spdlog::critical(
        "{}: index j = {} is not valid (min = {}, max = {})",
        "IP::get_inv_cov_masked_sqzd", cj, 0, like.Ndata );
    exit(1);
  }
  return this->inv_cov_masked_sqzd_(ci, cj);
}

double IP::get_chi2(arma::Col<double> datavector) const
{
  if (!(this->is_data_set_)) [[unlikely]] {
    spdlog::critical(
        "{}: {} not set prior to this function call",
        "IP::get_chi2", 
        "data_vector"
      );
    exit(1);
  }
  if (!(this->is_mask_set_)) [[unlikely]] {
    spdlog::critical(
        "{}: {} not set prior to this function call",
        "IP::get_chi2", 
        "mask"
      );
    exit(1);
  }
  if (!(this->is_inv_cov_set_)) [[unlikely]] {
    spdlog::critical(
        "{}: {} not set prior to this function call",
        "IP::get_chi2", 
        "inv_cov"
      );
    exit(1);
  }
  if (static_cast<int>(datavector.n_elem) != like.Ndata) [[unlikely]] {
    spdlog::critical("{}: incompatible data vector (theory size = {}, data size = {})",
        "IP::get_chi2", 
        datavector.n_elem, 
        like.Ndata
      );
    exit(1);
  }
  double chi2 = 0.0;
  #pragma omp parallel for collapse (2) reduction(+:chi2) schedule(static)
  for (int i=0; i<like.Ndata; i++) {
    for (int j=0; j<like.Ndata; j++) {
      if (this->get_mask(i) && this->get_mask(j)) {
        const double x = datavector(i) - this->get_dv_masked(i);
        const double y = datavector(j) - this->get_dv_masked(j);
        chi2 += x*this->get_inv_cov_masked(i,j)*y;
      }
    }
  }
  if (chi2 < 0.0) [[unlikely]] {
    spdlog::critical("{}: chi2 = {} (invalid)", "IP::get_chi2", chi2);
    exit(1);
  }
  return chi2;
}

arma::Col<double> 
IP::expand_theory_data_vector_from_sqzd(arma::Col<double> input) const
{
  if (this->ndata_sqzd_ != static_cast<int>(input.n_elem)) [[unlikely]] {
    spdlog::critical(
        "{}: invalid input data vector",
        "IP::expand_theory_data_vector_from_sqzd"
      );
    exit(1);
  }
  arma::Col<double> result(this->ndata_, arma::fill::zeros);
  for(int i=0; i<this->ndata_; i++) {
    if(this->mask_(i) > 0.99) {
      if(this->get_index_sqzd(i) < 0) [[unlikely]] {
        spdlog::critical(
            "{}: logical error, inconsistent mask operation",
            "IP::expand_theory_data_vector_from_sqzd"
          );
        exit(1);
      }
      result(i) = input(this->get_index_sqzd(i));
    }
  }
  return result;
}

arma::Col<double> IP::sqzd_theory_data_vector(arma::Col<double> input) const
{
  if (this->ndata_ != static_cast<int>(input.n_elem)) [[unlikely]] {
    spdlog::critical(
        "{}: invalid input data vector",
        "IP::sqzd_theory_data_vector"
      );
    exit(1);
  }
  arma::Col<double> result(this->ndata_sqzd_, arma::fill::zeros);
  for (int i=0; i<this->ndata_; i++) {
    if (this->get_mask(i) > 0.99) {
      result(this->get_index_sqzd(i)) = input(i);
    }
  }
  return result;
}

/*
void ima::RealData::set_PMmarg(std::string U_PMmarg_file)
{
  if (!(this->is_mask_set_))
  {
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: {} not set prior to this function call",
      "set_PMmarg", "mask"
    );
    exit(1);
  }

  arma::Mat<double> table = ima::read_table(U_PMmarg_file);
  if (table.n_cols!=3){
    spdlog::critical(
      "\x1b[90m{}\x1b[0m: U_PMmarg_file should has three columns, but has {}!"
      "set_PMmarg", table.n_cols);
    exit(1);
  }
  // U has shape of Ndata x Nlens
  arma::Mat<double> U;
  U.set_size(this->ndata_, tomo.clustering_Nbin);
  U.zeros();
  for (int i=0; i<static_cast<int>(table.n_rows); i++)
  {
    const int j = static_cast<int>(table(i,0));
    const int k = static_cast<int>(table(i,1));
    U(j,k) = static_cast<double>(table(i,2)) * this->get_mask(j);
  };
  // Calculate precision matrix correction
  // invC * U * (I+UT*invC*U)^-1 * UT * invC
  arma::Mat<double> iden = arma::eye<arma::Mat<double>>(tomo.clustering_Nbin, tomo.clustering_Nbin);
  arma::Mat<double> central_block = iden + U.t() * this->inv_cov_masked_ * U;
  // test positive-definite
  arma::Col<double> eigvals = arma::eig_sym(central_block);
  for(int i=0; i<tomo.clustering_Nbin; i++)
  {
    if(eigvals(i)<=0.0){
      spdlog::critical("{}: central block not positive definite!", "set_PMmarg");
      exit(-1);
    }
  }
  arma::Mat<double> invcov_PMmarg = this->inv_cov_masked_ * U * arma::inv_sympd(central_block) * U.t() * this->inv_cov_masked_; 
  //invcov_PMmarg.save("PMmarg_invcov_corr.h5", arma::hdf5_binary);
  // add the PM correction to inverse covariance
  for (int i=0; i<this->ndata_; i++)
  {
    invcov_PMmarg(i,i) *= this->get_mask(i);
    this->inv_cov_masked_(i,i) -= invcov_PMmarg(i,i);
    for (int j=0; j<i; j++)
    {
      double corr = this->get_mask(i)*this->get_mask(j)*(invcov_PMmarg(i,j)+invcov_PMmarg(j,i))/2.0;
      this->inv_cov_masked_(i,j) -= corr;
      this->inv_cov_masked_(j,i) -= corr;
    }
  }
  // examine again the positive-definite-ness
  arma::Col<double> eigvals_corr = arma::eig_sym(this->inv_cov_masked_);
  for(int i=0; i<tomo.clustering_Nbin; i++)
  {
    if(eigvals(i)<0){
      spdlog::critical("{}: PM-marged invcov not positive definite!", "set_PMmarg");
      exit(-1);
    }
  }

  // Update the reduced covariance and precision matrix
  for(int i=0; i<this->ndata_; i++)
  {
    for(int j=0; j<this->ndata_; j++)
    {
      if((this->mask_(i)>0.99) && (this->mask_(j)>0.99))
      {
        if(this->get_index_reduced_dim(i) < 0)
        {
          spdlog::critical("\x1b[90m{}\x1b[0m: logical error, internal"
            " inconsistent mask operation", "set_PMmarg");
          exit(1);
        }
        if(this->get_index_reduced_dim(j) < 0)
        {
          spdlog::critical("\x1b[90m{}\x1b[0m: logical error, internal"
            " inconsistent mask operation", "set_PMmarg");
          exit(1);
        }

        this->cov_masked_reduced_dim_(this->get_index_reduced_dim(i),
          this->get_index_reduced_dim(j)) = this->cov_masked_(i,j);

        this->inv_cov_masked_reduced_dim_(this->get_index_reduced_dim(i),
          this->get_index_reduced_dim(j)) = this->inv_cov_masked_(i,j);
      }
    }
  }
  //this->inv_cov_masked_.save("cocoa_invcov_PMmarg_masked.h5",arma::hdf5_binary);
}
*/

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Class IPCMB MEMBER FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void IPCMB::set_wxk_healpix_window(std::string healpixwin_filename) {
  matrix table = read_table(healpixwin_filename);
  this->params_->healpixwin_ncls = static_cast<int>(table.n_rows);
  if (this->params_->healpixwin != NULL) {
    free(this->params_->healpixwin);
  }
  this->params_->healpixwin = (double*) malloc1d(this->params_->healpixwin_ncls);
  for (int i=0; i<this->params_->healpixwin_ncls; i++) {
    this->params_->healpixwin[i] = static_cast<double>(table(i,0));
  }
  this->is_wxk_healpix_window_set_ = true;
}

void IPCMB::set_kk_binning_mat(std::string binned_matrix_filename)
{
  if(!this->is_kk_bandpower_) [[unlikely]] {
    spdlog::critical(
        "{}: {} == 0, incompatible choice", 
        "IPCMB::set_kk_binning_mat", "is_kk_bandpower"
      );
    exit(1);
  }
  matrix table = read_table(binned_matrix_filename);

  const int nbp  = this->get_nbins_kk_bandpower();
  const int lmax = this->get_lmax_kk_bandpower();
  const int lmin = this->get_lmin_kk_bandpower();
  const int ncl  = lmax - lmin + 1;
  
  if (this->params_->binning_matrix_kk != NULL) {
    free(this->params_->binning_matrix_kk);
  }
  this->params_->binning_matrix_kk = (double**) malloc2d(nbp, ncl);
    
  #pragma omp parallel for
  for (int i=0; i<nbp; i++) {
    for (int j=0; j<ncl; j++) {
      this->params_->binning_matrix_kk[i][j] = table(i,j);
    }
  }
  
  spdlog::debug(
      "{}: CMB kk binning matrix from file {} has {} x {} elements",
      "IPCMB::set_kk_binning_mat", 
      binned_matrix_filename, 
      nbp, 
      ncl
    );
  this->is_kk_binning_matrix_set_ = true;
}

void IPCMB::set_kk_theory_offset(std::string theory_offset_filename)
{
  if(!this->is_kk_bandpower_) [[unlikely]] {
    spdlog::critical(
        "{}: {} == 0, incompatible choice", 
        "IPCMB::set_kk_theory_offset", "is_kk_bandpower"
      );
    exit(1);
  }
  const int nbp = this->get_nbins_kk_bandpower();
  if (this->params_->theory_offset_kk != NULL) {
    free(this->params_->theory_offset_kk);
  }
  this->params_->theory_offset_kk = (double*) malloc1d(nbp);

  if (!theory_offset_filename.empty()) {
    matrix table = read_table(theory_offset_filename);
    for (int i=0; i<nbp; i++) {
      this->params_->theory_offset_kk[i] = static_cast<double>(table(i,0));
    }
  }
  else {
    for (int i=0; i<nbp; i++) {
      this->params_->theory_offset_kk[i] = 0.0;
    }
  }
  spdlog::debug(
      "{}: CMB theory offset from file {} has {} elements", 
      "IPCMB::set_kk_theory_offset", 
      theory_offset_filename, 
      nbp
    );
  this->is_kk_offset_set_ = true;
}

void IPCMB::set_kk_binning_bandpower(
    const int nbp, 
    const int lmin, 
    const int lmax
  )
{
  if (!(nbp > 0)) [[unlikely]] {
    spdlog::critical(
      "{}: {} = {} not supported", 
      "set_kk_bandpower_binning",
      "Number of Bins (nbp)", 
      nbp
    );
    exit(1);
  }
  if (!(lmin > 0)) [[unlikely]] {
    spdlog::critical(
      "{}: {} = {} not supported", 
      "set_kk_bandpower_binning",
      "lmin", 
      lmin
    );
    exit(1);
  }
  if (!(lmax > 0)) [[unlikely]] {
    spdlog::critical(
      "{}: {} = {} not supported", 
      "set_kk_bandpower_binning",
      "lmax", 
      lmax
    );
    exit(1);
  }
  spdlog::debug(
      "{}: {} = {} selected.", "init_binning_cmb_kk_bandpower", "NBins", nbp
    );
  spdlog::debug(
      "{}: {} = {} selected.", "init_binning_cmb_kk_bandpower", "lmin", lmin
    );
  spdlog::debug(
      "{}: {} = {} selected.",  "init_binning_cmb_kk_bandpower", "lmax", lmax
    );
  this->params_->nbp_kk = nbp;
  this->params_->lminbp_kk = lmin;
  this->params_->lmaxbp_kk = lmax;
  this->is_kk_bandpower_ = true;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Class PointMass MEMBER FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double PointMass::get_pm(
    const int zl, 
    const int zs, 
    const double theta
  ) const
{ // JX: add alens^2 in the den to be consistent with y3_production
  constexpr double Goverc2 = 1.6e-23;
  const double a_lens = 1.0/(1.0 + zmean(zl));
  const double chi_lens = chi(a_lens);
  return 4*M_PI*Goverc2*this->pm_[zl]*1.e+13*
    g_tomo(a_lens, zs)/(theta*theta)/(chi_lens*a_lens*a_lens*a_lens);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// BaryonScenario MEMBER FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void BaryonScenario::set_scenarios(std::string scenarios)
{
  std::vector<std::string> lines;
  lines.reserve(50);

  boost::trim_if(scenarios, boost::is_any_of("\t "));
  boost::trim_if(scenarios, boost::is_any_of("\n"));

  if (scenarios.empty()) [[unlikely]] {
    spdlog::critical(
        "{}: invalid string input (empty)",
        "BaryonScenario::set_scenarios"
      );
    exit(1);
  }
  
  spdlog::debug(
      "{}: Selecting baryon scenarios for PCA", 
      "BaryonScenario::set_scenarios"
    );

  boost::split(
      lines, 
      scenarios, 
      boost::is_any_of("/ \t"), 
      boost::token_compress_on
    );
  
  int nscenarios = 0;
  
  for (auto it=lines.begin(); it != lines.end(); ++it)
  {
    auto [name, tag] = get_baryon_sim_name_and_tag(*it);

    this->scenarios_[nscenarios++] = name + "-" + std::to_string(tag);
  }

  this->nscenarios_ = nscenarios;

  spdlog::debug(
      "{}: {} scenarios are registered", 
      "BaryonScenario::set_scenarios", this->nscenarios_
    );
  spdlog::debug(
      "{}: Registering baryon scenarios for PCA done!", 
      "BaryonScenario::set_scenarios"
    );
  
  this->is_scenarios_set_ = true;
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

} // end namespace cosmolike_interface

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
