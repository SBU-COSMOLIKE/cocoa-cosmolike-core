#ifndef COSMOLIKE_APERTURE_3PCF_H
#define COSMOLIKE_APERTURE_3PCF_H

// Finite side-angle-side (SAS) integration of centroid-projected shear
// three-point correlations. Angles and aperture radii are in radians.
// The caller validates dimensions, finite values and buffer sizes.
// d2, d3, radii, dlnd and dphi are positive; 0 < phi < pi.
// All arrays are contiguous in C order, with the last index adjacent.
// Inputs and outputs must not overlap; no buffer is retained or allocated.
//
// Weights have shape [nfilter, 4, nd2*nd3*nphi]. A triangle's flat index
// is (index_d2*nd3 + index_d3)*nphi + index_phi. Filters have shape
// [nfilter, 3] and keep their ordered radii (R1, R2, R3).
// Both the mirrored-radius contribution and the final factor 1/4 are
// included in the weights. The contraction takes Re(weight*gamma),
// without conjugating either factor.
//
// gamma uses the centroid/SAS convention of the aperture kernels. For
// fastnc centroid outputs, convert externally to
//   gamma = -[Gamma0, Gamma1, Gamma3, Gamma2].
// This interface does not infer or change a supplied projection.

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

#ifdef __cplusplus
}
#endif

#endif
