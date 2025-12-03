#ifndef __COSMOLIKE_REDSHIFT_SPLINE_CLUSTER_HPP
#define __COSMOLIKE_REDSHIFT_SPLINE_CLUSTER_HPP
#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// clusters routines
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

double get_effective_redmapper_area(const double a);

double pf_cluster_histo_n(double z, const int ni); 

double pz_cluster(double zz, const int nj); 

double zdistr_cluster(const int nz, const double z);

double g_lens_cluster(const double a, const int nz);

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// cluster-galaxy lensing routines
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

int test_zoverlap_cggl(const int nl, const int ns);

int ZCL(const int ni);

int ZCS(const int nj);

int NCGL(const int nl, const int ns);

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// cluster-galaxy clustering routines
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

int test_zoverlap_cg(const int nc, const int ng);

int ZCG1(const int ni); // find z1 bin of tomo combination

int ZCG2(const int nj); // find z2 bin of tomo combination

int NCG(const int nc, const int ng);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD