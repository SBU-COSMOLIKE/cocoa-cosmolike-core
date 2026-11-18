#ifndef COSMOLIKE_COVARIANCE_3PCF_H
#define COSMOLIKE_COVARIANCE_3PCF_H

#ifdef __cplusplus
extern "C" {
#endif

// These routines operate on flat, contiguous, row-major arrays. The last
// index changes fastest: pair[i,j,k] lives at (i*nfield + j)*nr + k.
// The caller owns every array and supplies finite inputs, valid lengths,
// positive dimensions, and an output buffer that overlaps no input.
// Read-only inputs may share storage. The pair transform allocates one
// temporary geometric table for the length of the call and frees it
// before returning. The contractions keep no state between calls.

// Filtered correlation of two real, statistically isotropic scalar fields:
//   K_ij(r) = integral dell ell/(2*pi) U_i(ell) U_j(ell) P_ij(ell) J0(ell*r),
// with U_i the real circular filter of field i, P_ij the total angular
// power of fields i and j, and J0 the Bessel function of the first
// kind of order zero. ell [nell] is nonnegative angular wavenumber;
// r [nr] is nonnegative separation in radians. Their product must be
// dimensionless. ell_weights [nell] holds the quadrature weights for
// dell alone; the code supplies the 1/(2*pi). For an integral sampled
// in log(ell), include the extra ell in these weights. filters
// [nfield,nell] holds the real dimensionless U_i. power
// [nfield,nfield,nell] holds the angular power, including any noise
// once. Each field label may specify a source bin and a filter. The
// function overwrites out [nfield,nfield,nr] and evaluates every
// ordered pair. The sum imposes no symmetry, no infinite ell range and
// no convergence.
void filtered_pair_3pcf(
  int nfield,                  // number of field/filter labels
  int nell,                    // number of multipole quadrature nodes
  int nr,                      // number of separation samples
  const double* ell,           // nonnegative multipoles [nell]
  const double* ell_weights,   // quadrature weights for dell [nell]
  const double* r,             // nonnegative separations, radians [nr]
  const double* filters,       // real Fourier filters [nfield,nell]
  const double* power,         // total angular power [nfield,nfield,nell]
  double* out                  // pair functions [nfield,nfield,nr]
);

// Weighted contraction of the caller's connected responses:
//   out[i,j] = sum_node weights[node] responses[i,node] responses[j,node].
// responses [nrow,nnode] holds the response of each row at each node;
// weights [nnode] is finite and nonnegative. The function overwrites
// out [nrow,nrow] with the symmetric result. The caller places the
// entire physical measure, including any survey area factor, in
// weights. This function does not infer halo profiles, a response
// model, or the other contributions a complete covariance needs.
void connected_gram_3pcf(
  int nrow,                    // number of observable responses
  int nnode,                   // number of shared quadrature nodes
  const double* responses,     // observable responses [nrow,nnode]
  const double* weights,       // nonnegative complete measure [nnode]
  double* out                  // symmetric covariance [nrow,nrow]
);

// Gaussian covariance of spatial averages of raw cubic scalar-map products.
// triplets [nrow,3] lists zero-based field indices for each product.
// pair [nfield,nfield,nr] is K_ij(r); pair_zero [nfield,nfield] is K_ij(0).
// Both describe symmetric correlations of real, zero-mean, isotropic
// Gaussian fields. The caller supplies pair_zero explicitly, so r need
// not start at zero. radial_weights [nr] is finite and nonnegative and
// includes the full radial quadrature and estimator window measure,
// e.g. 2*pi*r*dr/area in the large-area approximation; the function
// adds no extra area or angular factor. The function overwrites out
// [nrow,nrow] with all 15 positional Wick pairings: six with three
// cross-pairs and nine with one cross-pair. Repeated labels keep their
// multiplicity. This is not a distinct-galaxy catalogue estimator (the
// sum keeps the internal noise self-pairs), and it adds no connected
// non-Gaussian term and no xi-plus/minus cross-covariance.
void gaussian_aperture_covariance_3pcf(
  int nfield,                  // number of field/filter labels
  int nr,                      // number of radial quadrature nodes
  int nrow,                    // number of cubic observable rows
  const int* triplets,         // zero-based field labels [nrow,3]
  const double* pair,          // K_ij(r) [nfield,nfield,nr]
  const double* pair_zero,     // K_ij(0) [nfield,nfield]
  const double* radial_weights, // complete radial/window measure [nr]
  double* out                  // symmetric covariance [nrow,nrow]
);

#ifdef __cplusplus
}
#endif

#endif
