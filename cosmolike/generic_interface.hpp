#include <carma.h>
#include <armadillo>
#include <map>
#include "structs.h"

// Python Binding
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#ifndef __COSMOLIKE_GENERIC_INTERFACE_HPP
#define __COSMOLIKE_GENERIC_INTERFACE_HPP

namespace cosmolike_interface
{

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Class RandomNumber
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

class RandomNumber
{ // Singleton Class that holds a random number generator
  public:
    static RandomNumber& get_instance()
    {
      static RandomNumber instance;
      return instance;
    }
    ~RandomNumber() = default;

    double get()
    {
      return dist_(mt_);
    }

  protected:
    std::random_device rd_;
    std::mt19937 mt_;
    std::uniform_real_distribution<double> dist_;
  
  private:
    RandomNumber() :
      rd_(),
      mt_(rd_()),
      dist_(0.0, 1.0) {
      };
    RandomNumber(RandomNumber const&) = delete;
};

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Class IP (InterfaceProducts)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

class IP
{ // InterfaceProducts: Singleton Class that holds data vector, covariance...
  public:
    static IP& get_instance()
    {
      static IP instance;
      return instance;
    }
    ~IP() = default;

    // ----------------------------------------------

    bool is_mask_set() const {
      return this->is_mask_set_;
    }

    bool is_data_set() const {
      return this->is_data_set_;
    }

    bool is_inv_cov_set() const {
      return this->is_inv_cov_set_;
    }

    // ----------------------------------------------

    void set_data(std::string datavector_filename);

    // 3x2pt
    void set_mask(std::string mask_filename, 
                  arma::Col<int>::fixed<3> order, 
                  const int real_space);

    // 6x2pt
    void set_mask(std::string mask_filename, 
                  arma::Col<int>::fixed<6> order, 
                  const int real_space);

    void set_inv_cov(std::string covariance_filename);

    //void set_PMmarg(std::string U_PMmarg_file);

    // ----------------------------------------------

    int get_mask(const int ci) const;

    double get_dv_masked(const int ci) const;

    double get_inv_cov_masked(const int ci, const int cj) const;

    int get_index_sqzd(const int ci) const;

    double get_dv_masked_sqzd(const int ci) const;

    double get_inv_cov_masked_sqzd(const int ci, const int cj) const;

    arma::Col<double> expand_theory_data_vector_from_sqzd(arma::Col<double>) const;

    arma::Col<double> sqzd_theory_data_vector(arma::Col<double>) const;

    double get_chi2(arma::Col<double> datavector) const;

    // ----------------------------------------------

    int get_ndata() const;

    arma::Col<int> get_mask() const;

    arma::Col<double> get_dv_masked() const;

    arma::Mat<double> get_cov_masked() const;

    arma::Mat<double> get_inv_cov_masked() const;

    int get_ndata_sqzd() const;

    arma::Col<double> get_dv_masked_sqzd() const;

    arma::Mat<double> get_cov_masked_sqzd() const;

    arma::Mat<double> get_inv_cov_masked_sqzd() const;

  private:

    bool is_mask_set_ = false;
    
    bool is_data_set_ = false;
    
    bool is_inv_cov_set_ = false;
    
    int ndata_ = 0;
    
    int ndata_sqzd_ = 0;
    
    std::string mask_filename_;
    
    std::string cov_filename_;
    
    std::string data_filename_;
    
    arma::Col<int> mask_;

    arma::Col<double> data_masked_;
    
    arma::Mat<double> cov_masked_;

    arma::Col<int> index_sqzd_;
    
    arma::Mat<double> inv_cov_masked_;
    
    arma::Col<double> data_masked_sqzd_;

    arma::Mat<double> cov_masked_sqzd_; 

    arma::Mat<double> inv_cov_masked_sqzd_;
    
    IP() = default;
    IP(IP const&) = delete;
};

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Class IPCMB (Interface to Cosmolike C glocal struct CMBParams cmb)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

class IPCMB
{
  public:
    static IPCMB& get_instance()
    {
      static IPCMB instance;
      if (instance.params_ == NULL) {
        instance.params_ = &cmb;
      }
      return instance;
    }
    ~IPCMB() = default;

    // ----------------------------------------------
    
    bool is_kk_bandpower() const {
      return this->is_kk_bandpower_;
    }

    void update_chache(const double random) {
      this->params_->random = random;
    }

    void set_wxk_beam_size(const double fwhm) {
      this->params_->fwhm = fwhm;
      this->is_wxk_fwhm_set_ = true;
    }

    void set_wxk_lminmax(const int lmin, const int lmax) {
      this->params_->lmink_wxk = lmin;
      this->params_->lmaxk_wxk = lmax;
      this->is_wxk_lminmax_set_ = true;
    }

    void set_wxk_healpix_window(std::string healpixwin_filename);

    void set_kk_binning_mat(std::string binned_matrix_filename);

    void set_kk_theory_offset(std::string theory_offset_filename);

    void set_kk_binning_bandpower(const int, const int, const int);
    
    void set_alpha_Hartlap_cov_kkkk(const double alpha) {
      this->params_->alpha_Hartlap_cov_kkkk = alpha;
      this->is_alpha_Hartlap_cov_kkkk_set_ = true;
    }

    double get_kk_binning_matrix(const int, const int) const;

    double get_kk_theory_offset(const int) const;

    double get_alpha_Hartlap_cov_kkkk() const;

    int get_nbins_kk_bandpower() const;

    int get_lmin_kk_bandpower() const;

    int get_lmax_kk_bandpower() const;
    
  private: 
    CMBparams* params_ = NULL;
    bool is_wxk_fwhm_set_ = false;
    bool is_wxk_lminmax_set_ = false;
    bool is_wxk_healpix_window_set_ = false;
    bool is_kk_bandpower_ = false;
    bool is_kk_binning_matrix_set_ = false; 
    bool is_kk_offset_set_ = false; 
    bool is_alpha_Hartlap_cov_kkkk_set_ = false; 
    IPCMB() = default;
    IPCMB(IP const&) = delete; 
};

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Class PointMass
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

class PointMass
{// Singleton Class that Evaluate Point Mass Marginalization
  public:
    static PointMass& get_instance()
    {
      static PointMass instance;
      return instance;
    }
    ~PointMass() = default;

    void set_pm_vector(arma::Col<double> pm);

    arma::Col<double> get_pm_vector() const;

    double get_pm(const int zl, const int zs, const double theta) const;

  private:
    arma::Col<double> pm_;

    PointMass() = default;
    
    PointMass(PointMass const&) = delete;
};

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Class BaryonScenario
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

class BaryonScenario
{ // Singleton Class that map Baryon Scenario (integer to name)
  public:
    static BaryonScenario& get_instance()
    {
      static BaryonScenario instance;
      return instance;
    }
    ~BaryonScenario() = default;

    int nscenarios() const;

    bool is_pcs_set() const;

    bool is_scenarios_set() const;

    void set_scenarios(std::string data_sims, std::string scenarios);

    void set_scenarios(std::string scenarios);

    void set_pcs(arma::Mat<double> eigenvectors);

    std::string get_scenario(const int i) const;

    std::tuple<std::string,int> select_baryons_sim(const std::string scenario);

    arma::Mat<double> get_pcs() const;

    double get_pcs(const int ci, const int cj) const;

  private:
    bool is_pcs_set_;

    bool is_scenarios_set_;

    int nscenarios_;
    
    std::map<int, std::string> scenarios_;
    
    arma::Mat<double> eigenvectors_;

    BaryonScenario() = default;
    BaryonScenario(BaryonScenario const&) = delete;
};

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// AUX FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

arma::Mat<double> read_table(const std::string file_name);

// https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type 
almost_equal(T x, T y, int ulp = 100)
{
  // the machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs (units in the last place)
  return std::fabs(x-y) <= std::numeric_limits<T>::epsilon() * std::fabs(x+y) * ulp
      // unless the result is subnormal
      || std::fabs(x-y) < std::numeric_limits<T>::min();
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GLOBAL INIT FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void init_ntable_lmax(
    const int lmax
  );

void init_accuracy_boost(
    const double accuracy_boost, 
    const int integration_accuracy
  );

#ifdef HDF5LIB
void init_baryons_contamination(
    std::string sim, std::string all_sims_hdf5_file
  ); // NEW API
#endif

void init_baryons_contamination(std::string sim); // OLD API

void init_bias(arma::Col<double> bias_z_evol_model);

void init_binning_fourier(
    const int Ncl, 
    const int lmin, 
    const int lmax, 
    const int lmax_shear
  );

void init_binning_real_space(
    const int Ntheta, 
    const double theta_min_arcmin, 
    const double theta_max_arcmin
  );

void init_binning_cmb_bandpower(
    const int Nbandpower, 
    const int lmin, 
    const int lmax
  );

void init_cosmo_runmode(
    const bool is_linear
  );

void init_cmb(
    const double lmin_kappa_cmb, 
    const double lmax_kappa_cmb, 
    const double fwhm, 
    std::string pixwin_file
  );

void init_cmb_bandpower(
    const int is_cmb_bandpower, 
    const int is_cmb_kkkk_covariance_from_simulation, 
    const double alpha
  );

void init_data_3x2pt_real_space(
    std::string cov, 
    std::string mask, 
    std::string data,
    arma::Col<int>::fixed<3> order
  );

void init_data_6x2pt_real_space(
    std::string cov, 
    std::string mask, 
    std::string data,
    arma::Col<int>::fixed<6> order
  );

void init_data_3x2pt_fourier_space(
    std::string cov, 
    std::string mask, 
    std::string data,
    arma::Col<int>::fixed<3> order
  );

void init_data_6x2pt_fourier_space(
    std::string cov, 
    std::string mask, 
    std::string data,
    arma::Col<int>::fixed<6> order
  );

void init_data_vector_size(arma::Col<int>::fixed<6> exclude);

void init_data_vector_size_real_space(arma::Col<int>::fixed<6> exclude);

void init_data_vector_size_3x2pt_real_space();

void init_data_vector_size_6x2pt_real_space();

void init_data_vector_size_3x2pt_fourier_space();

void init_data_vector_size_6x2pt_fourier_space();

void init_IA(
    const int IA_MODEL, 
    const int IA_REDSHIFT_EVOL
  );

void init_probes(
    std::string possible_probes
  );

void initial_setup();

py::tuple read_redshift_distributions_from_files(
    std::string lens_multihisto_file, 
    const int lens_ntomo,
    std::string source_multihisto_file, 
    const int source_ntomo
  );

void init_redshift_distributions_from_files(
    std::string lens_multihisto_file, 
    const int lens_ntomo,
    std::string source_multihisto_file, 
    const int source_ntomo
  );

void init_survey(
    std::string surveyname, 
    double area, 
    double sigma_e
  );

void init_ggl_exclude(
	arma::Col<int> ggl_exclude
  );

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GLOBAL SET FUNCTIONS
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
  );

void set_distances(
    arma::Col<double> io_z, 
    arma::Col<double> io_chi
  );

void set_growth(
    arma::Col<double> io_z, 
    arma::Col<double> io_G
  );

void set_linear_power_spectrum(
    arma::Col<double> io_log10k,
    arma::Col<double> io_z, 
    arma::Col<double> io_lnP
  );

void set_non_linear_power_spectrum(
    arma::Col<double> io_log10k,
    arma::Col<double> io_z, 
    arma::Col<double> io_lnP
  );

void set_nuisance_bias(
    arma::Col<double> B1, 
    arma::Col<double> B2, 
    arma::Col<double> B_MAG
  );

void set_nuisance_clustering_photoz(
    arma::Col<double> CP
  );

void set_nuisance_clustering_photoz_stretch(
    arma::Col<double> CPS
  );

void set_nuisance_IA(
    arma::Col<double> A1, 
    arma::Col<double> A2,
    arma::Col<double> BTA
  );

void set_nuisance_magnification_bias(
    arma::Col<double> B_MAG
  );

void set_nuisance_nonlinear_bias(
    arma::Col<double> B1,
    arma::Col<double> B2
  );

void set_nuisance_shear_calib(
    arma::Col<double> M
  );

void set_nuisance_shear_photoz(
    arma::Col<double> SP
  );

void set_lens_sample_size(const int Ntomo);

void set_lens_sample(arma::Mat<double> input_table);

void set_source_sample_size(const int Ntomo);

void set_source_sample(arma::Mat<double> input_table);

void init_ntomo_powerspectra();

// ------------------------------------------------------------------
// Functions relevant for machine learning emulators
// ------------------------------------------------------------------

template <int N, int M, int P> 
arma::Col<double> compute_add_fpm_Mx2pt_N_masked(
    vector data_vector, 
    arma::Col<int>::fixed<M> ord
  )
{
  static_assert(0 == N || 1 == N, "N must be 0 (real) or 1 (fourier)");
  static_assert(3 == M || 6 == M, "M must be 3 (3x2pt) or 6 (6x2pt)");
  static_assert(P == 0 || P == 1, "P must be 0/1 (exclude PM))");
  arma::Col<int>::fixed<M> start = compute_data_vector_Mx2pt_N_starts<N,M>(ord);
  if constexpr (0 == N) {
    compute_ss_real_add_shear_calib_and_mask(data_vector, start(0));
    if constexpr (P == 1) {
      compute_gs_real_add_pm(data_vector, start(1));
    }
    compute_gs_real_add_shear_calib_and_mask(data_vector, start(1));
    compute_gg_real_add_mask(data_vector, start(2));
    if constexpr (6 == M) {
      compute_gk_real_add_mask(data_vector, start(3));
      compute_ks_real_add_shear_calib_and_mask(data_vector, start(4));
      compute_kk_real_add_mask(data_vector, start(5));
    }
  }
  return data_vector;
}

template <int N, int M>
arma::Col<int>::fixed<M> compute_data_vector_Mx2pt_N_starts() 
{
  static_assert(0 == N || 1 == N, "N must be 0 (real) or 1 (fourier)");
  static_assert(3 == M || 6 == M, "M must be 3 (3x2pt) or 6 (6x2pt)");
  using namespace arma;
  Col<int>::fixed<M> sizes = compute_data_vector_Mx2pt_N_sizes<N,M>();
  auto indices = conv_to<Col<int>>::from(stable_sort_index(order, "ascend"));
  Col<int>::fixed<M> start = {0,0,0};
  for(int i=0; i<M; i++) {
    for(int j=0; j<indices(i); j++) {
      start(i) += sizes(indices(j));
    }
  } 
}

template <int N, int M>
arma::Col<int>::fixed<M> compute_data_vector_Mx2pt_N_sizes() 
{
  static_assert(0 == N || 1 == N, "N must be 0 (real) or 1 (fourier)");
  static_assert(3 == M || 6 == M, "M must be 3 (3x2pt) or 6 (6x2pt)");
  arma::Col<int>::fixed<M> sizes;
  if constexpr (N == 0) {
    sizes(0) = 2*Ntable.Ntheta*tomo.shear_Npowerspectra;
    sizes(1) = Ntable.Ntheta*tomo.ggl_Npowerspectra;
    sizes(2) = Ntable.Ntheta*tomo.clustering_Npowerspectra;
    if constexpr (6 == M) {
      IPCMB& cmb = IPCMB::get_instance();
      sizes(3) = Ntable.Ntheta*redshift.clustering_nbin;
      sizes(4) = Ntable.Ntheta*redshift.shear_nbin;
      sizes(5) = cmb.is_kk_bandpower() == 1 ? cmb.get_nbins_kk_bandpower() : like.Ncl;
    }
  } 
  else {
    sizes(0) = 2*like.Ncl*tomo.shear_Npowerspectra;
    sizes(1) = like.Ncl*tomo.ggl_Npowerspectra;
    sizes(2) = like.Ncl*tomo.clustering_Npowerspectra;
    if constexpr (6 == M) {
      IPCMB& cmb = IPCMB::get_instance();
      sizes(3) = like.Ncl*redshift.clustering_nbin;
      sizes(4) = like.Ncl*redshift.shear_nbin;
      sizes(5) = cmb.is_kk_bandpower() == 1 ? cmb.get_nbins_kk_bandpower() : like.Ncl;
    }
  }
  return sizes;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GLOBAL COMPUTE FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

arma::Col<double> compute_binning_real_space();

arma::Col<double> compute_add_baryons_pcs(arma::Col<double> Q, arma::Col<double> dv);

template <int N, int M> 
matrix compute_baryon_pcas_Mx2pt_N(arma::Col<int>::fixed<M> ord)
{
  using vector = arma::Col<double>;
  static_assert(0 == N || 1 == N, "N must be 0 (real) or 1 (fourier)");
  static_assert(3 == M || 6 == M, "M must be 3 (3x2pt) or 6 (6x2pt)");
  const string name = "compute_baryon_pcas_Mx2pt_N";
  IP& ip = IP::get_instance();
  const int ndata = ip.get_ndata();
  const int ndata_sqzd = ip.get_ndata_sqzd();
  const int nscenarios = BaryonScenario::get_instance().nscenarios();

  // Compute Cholesky Decomposition of the Covariance Matrix --------------
  spdlog::debug("{}: Cholesky Decomposition of the Cov Matrix begins", name);
  matrix L = arma::chol(ip.get_cov_masked_sqzd(), "lower");
  matrix inv_L = arma::inv(L);
  spdlog::debug("{}: Cholesky Decomposition of the Cov Matrix ends", name);

  // Compute Dark Matter data vector --------------------------------------
  spdlog::debug("{}: Comp. DM only data vector begins", name);
  cosmology.random = RandomNumber::get_instance().get();
  reset_bary_struct(); // make sure there is no baryon contamination
  vector dv_dm = ip.sqzd_theory_data_vector(compute_data_vector_Mx2pt_N_masked<N,M>(ord));
  spdlog::debug("{}: Comp. DM only data vector ends", name);
  
  // Compute data vector for all Baryon scenarios -------------------------
  matrix D = matrix(ndata_sqzd, nscenarios);
  for (int i=0; i<nscenarios; i++) {
    spdlog::debug("{}: Comp. data vector w/ baryon scenario {} begins", name, bs);
    const string bs = BaryonScenario::get_instance().get_scenario(i);
    cosmology.random = RandomNumber::get_instance().get(); // clear cosmolike cache
    init_baryons_contamination(bs);
    vector dv = ip.sqzd_theory_data_vector(compute_data_vector_Mx2pt_N_masked<N,M>(ord));
    D.col(i) = dv - dv_dm;
    spdlog::debug("{}: Comp. data vector w/ baryon scenario {} ends", name, bs);
  }
  reset_bary_struct();
  cosmology.random = RandomNumber::get_instance().get();  // clear cosmolike cache

  // weight the diff matrix by inv_L; then SVD ----------------------------  
  matrix U, V;
  vector s;
  arma::svd(U, s, V, inv_L * D);

  // compute PCs ----------------------------------------------------------
  matrix PC = L * U; 

  // Expand the number of dims --------------------------------------------
  matrix R = matrix(ndata, nscenarios); 
  for (int i=0; i<nscenarios; i++) {
    R.col(i) = ip.expand_theory_data_vector_from_sqzd(PC.col(i));
  }

  return R;
}

template <int N, int M> 
arma::Col<double> compute_data_vector_Mx2pt_N_masked(arma::Col<int>::fixed<M> ord)
{
  using vector = arma::Col<double>;
  static_assert(0 == N || 1 == N, "N must be 0 (real) or 1 (fourier)");
  static_assert(3 == M || 6 == M, "M must be 3 (3x2pt) or 6 (6x2pt)");
  arma::Col<int>::fixed<M> start = compute_data_vector_Mx2pt_N_starts<N,M>(ord);
  vector data_vector(like.Ndata, arma::fill::zeros); 
  if constexpr (0 == N) {
    compute_ss_real_masked(data_vector, start(0));
    compute_gs_real_masked(data_vector, start(1));
    compute_gg_real_masked(data_vector, start(2));
    if constexpr (6 == M) {
      compute_gk_real_masked(data_vector, start(3));
      compute_ks_real_masked(data_vector, start(4));
      compute_kk_fourier_masked(data_vector, start(5));
    }
  } 
  else {
    compute_ss_fourier_masked(data_vector, start(0));
    compute_gs_fourier_masked(data_vector, start(1));
    compute_gg_fourier_masked(data_vector, start(2));
    if constexpr (6 == M) {
      compute_gk_real_add_mask(data_vector, start(3));
      compute_ks_real_add_shear_calib_and_mask(data_vector, start(4));
      compute_kk_real_add_mask(data_vector, start(5));
    }
  }
  return data_vector;
}

}  // namespace cosmolike_interface
#endif // HEADER GUARD
