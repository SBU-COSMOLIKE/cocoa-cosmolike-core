#ifndef COSMOLIKE_APERTURE_3PCF_H
#define COSMOLIKE_APERTURE_3PCF_H

// Finite side-angle-side (SAS) integration of centroid-projected shear
// three-point correlations. A SAS grid labels a triangle by two side
// lengths and the opening angle between them. All angles and all
// aperture radii are in radians.
//
// The caller validates the inputs: dimensions, finite values and
// buffer sizes. d2, d3, the radii, dlnd and dphi are positive, and
// phi lies strictly between zero and pi. Every array is contiguous in
// C order, with the last index adjacent in memory. Inputs and outputs
// must not overlap. No function here keeps a pointer after it
// returns; the aperture functions allocate no memory, and the
// two-point helper frees its own scratch before returning.
//
// Weights have shape [nfilter, 4, nd2*nd3*nphi]. A triangle's flat
// index is (index_d2*nd3 + index_d3)*nphi + index_phi. Filters have
// shape [nfilter, 3], and each row holds one ordered radius triplet
// (R1, R2, R3). The weights already contain the mirrored-radius
// contribution and the final factor 1/4, so the contraction is only
// Re(weight*gamma), and it conjugates neither factor.
//
// gamma uses the centroid/SAS convention of the aperture kernels. A
// caller who starts from fastnc centroid outputs converts them
// externally to
//   gamma = -[Gamma0, Gamma1, Gamma3, Gamma2].
// This interface neither infers nor changes the projection the caller
// passes in.

#ifdef __cplusplus
extern "C" {
#endif

void aperture_weights_3pcf(
  int nd2,                       // number of first-side bins
  int nd3,                       // number of second-side bins
  int nphi,                      // number of opening-angle bins
  int nfilter,                   // number of ordered aperture triplets
  const double* d2,              // first-side representatives [nd2]
  const double* d3,              // second-side representatives [nd3]
  const double* phi,             // angle representatives [nphi]
  double dlnd,                   // common logarithmic side-bin width
  double dphi,                   // common angle-bin width, radians
  const double* filters,         // radii [nfilter, 3], radians
  double* weights_real,          // real weights [nfilter, 4, ntriangle]
  double* weights_imag           // imaginary weights, same shape
);

void aperture_apply_3pcf(
  int nfilter,                   // number of ordered aperture triplets
  int ntriangle,                 // nd2*nd3*nphi from weight construction
  int nsample,                   // number of source-bin combinations
  const double* wr,              // real weights [nfilter, 4, ntriangle]
  const double* wi,              // imaginary weights, same shape
  const double* gamma_real,      // real gamma [nsample, 4, ntriangle]
  const double* gamma_imag,      // imaginary gamma, same shape
  double* out                    // real aperture moments [nsample,nfilter]
);

// Bin-averaged two-point functions xi+ and xi- from the E-mode and
// B-mode angular spectra the caller evaluates on the quadrature
// nodes. sign = +1 selects xi+, which uses the Bessel kernel J_0 and
// the sum C_E + C_B; sign = -1 selects xi-, which uses J_4 and the
// difference C_E - C_B. The weights on each separation bin sum to
// one. The additive noise bias is not removed.
void covariance_xi_from_power_3pcf(
  int nrow,                    // number of xi rows
  int nell,                    // Fourier quadrature nodes
  int ntheta,                  // separation samples per row
  const double* ell,           // multipoles [nell]
  const double* dell,          // dell weights [nell]
  const double* theta,         // radians [nrow,ntheta]
  const double* theta_weight,  // normalized weights [nrow,ntheta]
  const double* spectra_e,     // E angular spectra [nrow,nell]
  const double* spectra_b,     // B angular spectra [nrow,nell]
  const int* sign,             // +1 or -1 [nrow]
  double* out                  // bin-averaged xi [nrow]
);

#ifdef __cplusplus
}
#endif

#endif
