#include <algorithm>
#include "covariance_wrapper.hpp"

namespace cosmolike_interface {

// ---------------------------------------------------------------------------
// Notebook wrappers of the 3PCF covariance kernels. A notebook wrapper
// takes Armadillo-typed inputs (Armadillo is the C++ linear-algebra
// library the interactive bindings use), copies them into buffers it
// owns, calls the C kernel, and returns Armadillo containers. The
// wrappers keep no state between calls, so an interactive session can
// slice, move or overwrite its arrays freely between calls.
// ---------------------------------------------------------------------------

// The constructor reads each source pointer only long enough to make
// the Armadillo-owned copy. Once it returns, moving or changing the
// caller's NumPy array cannot change this notebook calculation.
covariance_request_cpp_3pcf::covariance_request_cpp_3pcf(
  const covariance_input_3pcf& input, // validated arrays and scalars
  const int nsource // full physical source count, including unselected bins
) : request(input)
{
  // Each arrays[] entry is {name, {source pointer, element count}}:
  // the name keys the owned copy, and the count sizes it.
  const std::pair<const char*, std::pair<const double*, size_t>> arrays[] = {
    {"ell", {input.ell, (size_t)input.nell}},
    {"dell", {input.dell, (size_t)input.nell}},
    {"window_ell", {input.window_ell, (size_t)input.nwindow}},
    {"window_dell", {input.window_dell, (size_t)input.nwindow}},
    {"phi", {input.phi, (size_t)input.nphi}},
    {"dphi", {input.dphi, (size_t)input.nphi}},
    {"radius", {input.radius, (size_t)input.nr}},
    {"dr", {input.dr, (size_t)input.nr}},
    {"profile_radius", {input.profile_radius, (size_t)input.nprofile}},
    {"k", {input.k, (size_t)input.nk}},
    {"dk", {input.dk, (size_t)input.nk}},
    {"mass", {input.mass, (size_t)input.nmass}},
    {"dmass", {input.dmass, (size_t)input.nmass}},
    {"field_radius", {input.field_radius, (size_t)input.nfield}},
    {"theta", {input.theta, (size_t)input.nrow*input.ntheta}},
    {"theta_weight", {input.theta_weight,
        (size_t)input.nrow*input.ntheta}},
    {"noise", {input.noise, (size_t)nsource*nsource}}
  };

  for (const auto& item : arrays) {
    const auto count = item.second.second;
    arma::Col<double> values(count);
    std::copy_n(item.second.first, count, values.memptr());
    real.emplace(item.first, std::move(values));
  }

  // Exact bin edges are optional in finite-band mode. Preserve absence as
  // a null pointer; do not manufacture an annulus from quadrature nodes.
  if (nullptr != input.theta_edges) {
    arma::Col<double> edges((size_t)input.nrow*2);
    std::copy_n(input.theta_edges, edges.n_elem, edges.memptr());
    real.emplace("theta_edges", std::move(edges));
    request.theta_edges = real.at("theta_edges").memptr();
  }

  // Retain independent copies of every optional variance coordinate and
  // value. The notebook caller may change or discard its arrays as soon
  // as this request has been constructed.
  if (0 < input.nhalo_z) {
    const size_t count = (size_t)input.nhalo_z*input.nmass;
    real.emplace("halo_z", arma::Col<double>(input.halo_z, input.nhalo_z));
    real.emplace("halo_mass", arma::Col<double>(input.halo_mass, input.nmass));
    real.emplace("halo_sigma2", arma::Col<double>(input.halo_sigma2, count));
    real.emplace("halo_dlnsigma_dlnm",
      arma::Col<double>(input.halo_dlnsigma_dlnm, count));
    request.halo_z = real.at("halo_z").memptr();
    request.halo_mass = real.at("halo_mass").memptr();
    request.halo_sigma2 = real.at("halo_sigma2").memptr();
    request.halo_dlnsigma_dlnm = real.at("halo_dlnsigma_dlnm").memptr();
  }

  // The source count comes from the model, not from the field subset
  // the request selects, so the noise copy keeps every off-diagonal
  // source pair.
  const std::pair<const char*, std::pair<const int*, size_t>> indices[] = {
    {"field_source", {input.field_source, (size_t)input.nfield}},
    {"row_kind", {input.row_kind, (size_t)input.nrow}},
    {"row_fields", {input.row_fields, (size_t)input.nrow*3}}
  };

  for (const auto& item : indices) {
    arma::Col<int> values(item.second.second);
    std::copy_n(item.second.first, item.second.second, values.memptr());
    integer.emplace(item.first, std::move(values));
  }

  request.ell = real.at("ell").memptr();
  request.dell = real.at("dell").memptr();
  request.window_ell = real.at("window_ell").memptr();
  request.window_dell = real.at("window_dell").memptr();
  request.phi = real.at("phi").memptr();
  request.dphi = real.at("dphi").memptr();
  request.radius = real.at("radius").memptr();
  request.dr = real.at("dr").memptr();
  request.profile_radius = real.at("profile_radius").memptr();
  request.k = real.at("k").memptr();
  request.dk = real.at("dk").memptr();
  request.mass = real.at("mass").memptr();
  request.dmass = real.at("dmass").memptr();
  request.field_radius = real.at("field_radius").memptr();
  request.theta = real.at("theta").memptr();
  request.theta_weight = real.at("theta_weight").memptr();
  request.noise = real.at("noise").memptr();
  request.field_source = integer.at("field_source").memptr();
  request.row_kind = integer.at("row_kind").memptr();
  request.row_fields = integer.at("row_fields").memptr();
}


const covariance_input_3pcf& covariance_request_cpp_3pcf::get() const
{
  return request;
}


// ---------------------------------------------------------------------------
// Notebook covariance: the mean and the six covariance families.
//
// The C engine writes the terms as one flat [term,row,column] array;
// the loop below unpacks it into a cube indexed (row, column, term).
//
// Parameters:
//   model - borrowed immutable physical tables
//   input - privately owned quadrature arrays
//
// Returns:
//   (mean, terms): mean[nrow], and an arma::Cube indexed
//   (row, column, term) whose six slices are, in order: Gaussian,
//   bispectrum-squared (BB), power times one-halo trispectrum, power
//   times bispectrum, connected one-halo, and the chosen
//   halo-abundance response (covariance_physics_3pcf.h).
// ---------------------------------------------------------------------------
std::pair<arma::Col<double>, arma::Cube<double>> covariance_physical_cpp_3pcf(
  const physics_model_3pcf* model, // borrowed immutable physical tables
  const covariance_request_cpp_3pcf& input // privately owned quadrature arrays
)
{
  const auto& request = input.get();
  const size_t n = request.nrow;
  arma::Col<double> mean(n);
  arma::Col<double> packed(6*n*n);

  covariance_physical_3pcf(model, &request, mean.memptr(), packed.memptr());

  arma::Cube<double> terms(n, n, 6);
  // The C output is [term,row,column], with columns contiguous. An
  // Armadillo matrix instead stores rows contiguously within a column.
  // Naming all three indices makes the physical term order visible.
  for (size_t term=0; 6 > term; term++) {
    for (size_t row=0; row<n; row++) {
      for (size_t column=0; column<n; column++) {
        terms(row, column, term) = packed[(term*n+row)*n+column];
      }
    }
  }

  return {std::move(mean), std::move(terms)};
}


// ---------------------------------------------------------------------------
// Notebook halo tables on the (z, M) grid.
//
// Parameters:
//   model     - read-only cosmology
//   input     - owned halo quadratures
//   z         - redshift grid
//   k_profile - nonnegative h/Mpc profile wavenumbers
//
// Returns:
//   five flat C-order columns: sigma2 (the mass variance), dndm (the
//   halo abundance), bias and concentration, each flattened [z,M],
//   and then the normalized NFW profile u, flattened [z,M,k].
// ---------------------------------------------------------------------------
std::vector<arma::Col<double>> covariance_halos_cpp_3pcf(
  const physics_model_3pcf* model, // read-only cosmology
  const covariance_request_cpp_3pcf& input, // owned halo quadratures
  const arma::Col<double>& z, // redshifts
  const arma::Col<double>& k_profile // nonnegative h/Mpc profile nodes
)
{
  const auto& request = input.get();
  const size_t count = z.n_elem*request.nmass;

  std::vector<arma::Col<double>> result;
  for (int index=0; 4 > index; index++) {
    result.emplace_back(count);
  }
  result.emplace_back(count*k_profile.n_elem);

  covariance_halos_3pcf(model, &request, (int)z.n_elem, z.memptr(),
      (int)k_profile.n_elem, k_profile.memptr(), result[0].memptr(),
      result[1].memptr(), result[2].memptr(), result[3].memptr(),
      result[4].memptr());

  return result;
}


// ---------------------------------------------------------------------------
// Notebook bin-averaged xi from E/B angular power spectra.
//
// Parameters:
//   ell, dell            - multipole nodes and their dell weights
//   theta                - separation nodes [row,separation], radians
//   theta_weight         - normalized bin weights, same shape
//   spectra_e, spectra_b - E- and B-mode signal power [row,ell]
//   sign                 - +1 selects xi+ (J0), -1 selects xi- (J4)
//
// Returns:
//   arma::Col of length nrow: the bin-averaged xi of each row
// ---------------------------------------------------------------------------
arma::Col<double> covariance_xi_cpp_3pcf(
  const arma::Col<double>& ell, // multipoles
  const arma::Col<double>& dell, // integration weights in dell
  const arma::Mat<double>& theta, // [row,separation] in radians
  const arma::Mat<double>& theta_weight, // normalized bin weights
  const arma::Mat<double>& spectra_e, // [row,ell] signal E power
  const arma::Mat<double>& spectra_b, // [row,ell] signal B power
  const arma::Col<int>& sign // +1 for xi+, -1 for xi-
)
{
  const size_t nrow = theta.n_rows;
  const size_t ntheta = theta.n_cols;
  const size_t nell = ell.n_elem;

  arma::Col<double> angles(nrow*ntheta);
  arma::Col<double> weights(nrow*ntheta);
  arma::Col<double> even(nrow*nell);
  arma::Col<double> odd(nrow*nell);

  // Packing changes memory order only. Each physical row keeps its
  // separation weights and E/B spectra together before entering C.
  for (size_t row=0; row<nrow; row++) {
    for (size_t point=0; point<ntheta; point++) {
      angles[row*ntheta+point] = theta(row, point);
      weights[row*ntheta+point] = theta_weight(row, point);
    }
    for (size_t point=0; point<nell; point++) {
      even[row*nell+point] = spectra_e(row, point);
      odd[row*nell+point] = spectra_b(row, point);
    }
  }

  arma::Col<double> result(nrow);
  covariance_xi_from_power_3pcf((int)nrow, (int)nell, (int)ntheta,
      ell.memptr(), dell.memptr(), angles.memptr(), weights.memptr(),
      even.memptr(), odd.memptr(), sign.memptr(), result.memptr());

  return result;
}

}
