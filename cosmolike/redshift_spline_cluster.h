#ifndef __COSMOLIKE_REDSHIFT_SPLINE_CLUSTER_HPP
#define __COSMOLIKE_REDSHIFT_SPLINE_CLUSTER_HPP
#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// cluster routines
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

int test_zoverlap_c(int zc, int zs); // test whether source bin zs is behind lens bin zl (clusters)

int ZCL(int Nbin); // find zlens bin of tomo combination (cluster-galaxy lensing)

int ZCS(int Nbin); // find zsource bin of tomo combination (cluster-galaxy lensing)

int N_cgl(int zl, int zs); // find tomo bin number tomography combination

int ZCCL1(const int Nbin); // find z1 bin of tomo combination (cluster clustering)

int ZCCL2(const int Nbin); // find z2 bin of tomo combination (cluster clustering)

// find tomo bin number tomography combination (cluster clustering)
int N_CCL(const int z1, const int z2);


int ZCGCL1(const int ni); // find z1 bin of tomo combination (cluster-galaxy cross clustering)

int ZCGCL2(const int nj); // find z2 bin of tomo combination (cluster-galaxy cross clustering)

// find tomo bin number tomography combination (cluster-galaxy cross clustering)
int N_CGCL(const int ni, const int nj); // ni = Cluster Nbin, nj = Galaxy Nbin


//return pf(z,j) based on redshift file with structure z[i] nz[0][i]..nz[tomo.clustering_Nbin-1][i]
double pf_cluster_histo_n(double z, void* params); 

double pz_cluster(const double zz, const int nz); // int_zmin^zmax dz_obs p(z_obs|z_true)

double dV_cluster(double z, void* params);

// simplfied selection function, disregards evolution of N-M relation+mass function within z bin
double zdistr_cluster(const int nz, const double z);

// lens efficiency of lens cluster in tomo bin nz, lambda bin nl used in magnification calculations
double g_lens_cluster(const double a, const int nz, const int nl);


#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD