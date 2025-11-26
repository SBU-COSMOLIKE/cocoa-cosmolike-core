#ifndef __COSMOLIKE_RADIAL_CLUSTER_WEIGHTS_H
#define __COSMOLIKE_RADIAL_CLUSTER_WEIGHTS_H
#ifdef __cplusplus
extern "C" {
#endif

double W_cluster(const int nl, const int nz, const double a, const double hoverh0);

double W_mag_cluster(const int nz, const double a, const double fK);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD