#ifndef COSMOLIKE_PHYSICS_3PCF_H
#define COSMOLIKE_PHYSICS_3PCF_H

#ifdef __cplusplus
extern "C" {
#endif

// Tables use k in h/Mpc, chi in Mpc/h and P in (Mpc/h)^3. The redshift
// grid includes zero, where growth equals one. All buffers are borrowed
// only during creation: the resulting model owns its interpolation data.
// Spatial flatness requires |omega_m+omega_de-1| <= 1e-8; this allows
// rounding of supplied density scalars, not a curved-lensing model.
// LOS nodes are positive, strictly increasing, at least two in number,
// and contained in the supplied redshift range. They define a fixed
// trapezoidal distance integral; kernel_nz must be at least four.
typedef struct {
  int nk;
  int nz;
  int nbin;
  int nlos;
  int kernel_nz;
  const double* k;
  const double* p_linear0;
  const double* z;
  const double* p_nonlinear; // [nz,nk], row-major
  const double* chi;
  const double* dz_dchi;
  const double* growth;
  const double* sources; // [nbin,nz], densities per unit redshift
  const double* los_z;
  double omega_m;
  double omega_de;
  double w0;
  double wa;
  double ns;
  double sigma8;
  int source_readers; // 1 retains source cubic reads for explicit comparisons
  int table_boost; // production dense-grid factor; zero selects eight
  int fit_nz; // production uniform coefficient anchors; zero selects 33
} physics_input_3pcf;

typedef struct {
  double a1;
  double eta1;
  double a2;
  double eta2;
  double bta;
  double ct;
  double zpivot;
  double c1rho;
  double fb;
  int power_mode; // 0: linear D^2; 1: nonlinear(z=0) D^2; 2: nonlinear(z)
  int ct_mode; // 0: physical Ct=C1*ct (fastnc); 1: physical Ct=ct
  int suppress_only; // cap the baryon transfer at one before scaling fb
} physics_ia_3pcf;

typedef struct physics_model_3pcf physics_model_3pcf;
typedef struct physics_projection_3pcf physics_projection_3pcf;

physics_model_3pcf* physics_create_3pcf(
  const physics_input_3pcf* input
);
void physics_destroy_3pcf(physics_model_3pcf* model);
void physics_default_ia_3pcf(physics_ia_3pcf* parameters);

double physics_power_3pcf(
  const physics_model_3pcf* model,
  double k,
  double z,
  int mode
);

// component: 0=BiHalofit, 1=tree, 2=one-halo fit, 3=three-halo fit.
// These are fitted matter terms, not the halo integrals of covariance.
double physics_matter_3pcf(
  const physics_model_3pcf* model,
  double k1,
  double k2,
  double k3,
  double z,
  int component,
  double fb
);

// The 19 entries follow the active fastnc tensor calculation, in order:
// ddE,dEd,Edd,dEE,EEd,EdE,EEE,ddB,dBd,Bdd,dEB,dBE,EBd,BEd,BdE,EdB,
// EEB,EBE,BEE. d denotes matter density; E/B are intrinsic shape modes.
// remove_alignment removes C1^p B_tree only from the seven E entries.
void physics_ia_bispectra_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  double k1,
  double k2,
  double k3,
  double z,
  int remove_alignment,
  double* out
);

// Evaluate physical triangles that differ only by a positive common side
// scaling. All 19 IA components share one orientation/F2 preparation.
// sides is [npoint,3]; z is [npoint]; out is [npoint,19].
void physics_ia_same_shape_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  int npoint,
  const double* sides,
  const double* z,
  int remove_alignment,
  double* out
);

// A projection owns weights for one ordered source-bin triplet and IA
// parameter choice. It borrows its immutable model, which must outlive it.
// Both objects support simultaneous read-only calls from OpenMP workers.
physics_projection_3pcf* physics_projector_create_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const int* bins
);
void physics_projector_destroy_3pcf(physics_projection_3pcf* projection);
double physics_projected_3pcf(
  const physics_projection_3pcf* projection,
  double ell1,
  double ell2,
  double ell3,
  int component // 0=total E, 1=matter with NLA kernels, 2=TATT residual
);

// Only a rectangular PUV interpolation table uses this source extension.
// Its off-triangle corners follow the active IA vector clipping; the zero
// closing-side corner is zero. Physical direct queries use the guard above.
double physics_projected_puv_extension_3pcf(
  const physics_projection_3pcf* projection,
  double ell1,
  double ell2,
  double ell3
);

// Batch preparation requires the same immutable model and IA parameters
// for every projector. Only the ordered source triplets differ. One set
// of physical vertices is contracted with all radial weights at each z.
void physics_projected_batch_3pcf(
  int nprojection,
  physics_projection_3pcf* const* projections,
  double ell1,
  double ell2,
  double ell3,
  int component,
  int puv_extension,
  double* out
);

// The shared Fourier integrator accepts either a direct physical
// projection or an immutable prepared table. The callback returns total
// ordered E-mode B; it must support concurrent read-only evaluations.
typedef double (*physics_projection_reader_3pcf)(
  const void* context,
  double ell1,
  double ell2,
  double ell3
);
void physics_aperture_integrate_3pcf(
  const void* context,
  physics_projection_reader_3pcf reader,
  int nell,
  const double* ell,
  const double* dell,
  int nphi,
  const double* phi,
  const double* dphi,
  int nfilter,
  const double* filters,
  double* out
);

// b_L=(2L+1)/2 integral[-1,1] dmu P_L(mu) B(ell_i,ell_j,ell_3),
// ell_3^2=ell_i^2+ell_j^2-2 ell_i ell_j mu. No Limber +1/2 shift.
// out has shape [lmax+1,nell,nell]; no leg-sorting symmetry is imposed.
void physics_multipoles_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const int* bins,
  int nell,
  const double* ell,
  int lmax,
  int nmu,
  int component,
  double* out
);

// Full compensated-Gaussian aperture mean, with U(ell R)=(ell R)^2
// exp[-(ell R)^2/2]/2. filters [nfilter,3] retains the ordered source
// legs. Explicit positive quadratures integrate dell1,dell2,dphi.
void physics_aperture_full_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const int* bins,
  int nell,
  const double* ell,
  const double* dell,
  int nphi,
  const double* phi,
  const double* dphi,
  int nfilter,
  const double* filters,
  double* out
);

// Covariance can share the same physical tables. The returned arrays are
// borrowed for the model's lifetime. dchi weights are trapezoidal nodes;
// lens and shape are g and W_g in the fastnc convention [nbin,nlos].
void physics_radial_3pcf(
  const physics_model_3pcf* model,
  int* nbin,
  int* nlos,
  const double** z,
  const double** chi,
  const double** dchi,
  const double** lens,
  const double** shape
);

double physics_omega_m_3pcf(const physics_model_3pcf* model);
double physics_omega_m_z_3pcf(
  const physics_model_3pcf* model,
  double z
);
void physics_nla_kernel_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  int bin,
  double* out // physical W(chi), [nlos], includes chi*(1+z)
);
double physics_nla_power_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  int bin1,
  int bin2,
  double ell
);
void physics_nla_power_batch_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  int bin1,
  int bin2,
  int nell,
  const double* ell,
  double* out
);

#ifdef __cplusplus
}
#endif
#endif
