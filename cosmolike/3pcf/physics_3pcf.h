#ifndef COSMOLIKE_PHYSICS_3PCF_H
#define COSMOLIKE_PHYSICS_3PCF_H

#include "feedback_3pcf.h"
#include "baseline_3pcf.h"

#ifdef __cplusplus
extern "C" {
#endif

// Interface of the three-point (3PCF) physics model: supplied matter
// and intrinsic alignment (IA) bispectra and their flat-sky Limber
// projection along the line of sight (LOS).
//
// Units and grids. Tables use the wave number k in h/Mpc, the
// comoving distance chi in Mpc/h and the power spectrum P in
// (Mpc/h)^3. The background redshift grid includes zero, where the
// growth factor equals one. k is uniform in ln k; z and source_z are
// piecewise uniform, with at most MAX_GRID_SEGMENTS = 10 segments.
// Generate each segment with the np.linspace convention; the shared grid
// detector uses relative spacing tolerance 1e-9. Production preparation
// uses natural cubic splines within each segment; source_readers retains
// the original source interpolation for comparisons.
// The model copies every buffer during
// creation and owns its interpolation data afterwards, so the caller
// may free the inputs once physics_create_3pcf returns.
//
// The provider supplies Omega_m(z) for the halo collapse correction.
// Pressureless matter permits Omega_m0*(1+z)^3/[H(z)/H0]^2 in the
// provider; exact massive-neutrino fractions must include their actual
// density evolution. C never reconstructs a separate CPL background.
// Flat lensing geometry remains the provider's responsibility.
//
// Smoothing snapshots below are retained as an adapter data type. They
// belong to the theory producer, not to this projection model.
typedef struct {
  int nz; // anchor count
  const double* z; // piecewise-uniform redshift nodes [nz]
  const double* rnl; // positive nonlinear radii [nz], Mpc/h
  const double* neff; // effective spectral indices [nz]
  const double* sigma8; // positive linear rms amplitudes [nz]
} physics_variance_input_3pcf;

// LOS nodes are positive, strictly increasing, at least two in
// number, and must lie inside the tabulated redshift range; they
// define a fixed trapezoidal distance integral. kernel_nz, the
// resolution of the internal lensing-kernel grid, must be at least
// four.
typedef struct {
  int nk;        // wave-number nodes
  int nz;        // background redshift nodes
  int nbin;      // tomographic source bins
  int nlos;      // line-of-sight integration nodes
  int kernel_nz; // internal lensing-kernel redshift resolution
  const double* k;           // wave numbers, increasing [nk]
  const double* p_linear0;   // linear power at z = 0 [nk]
  const double* z;           // background redshifts from zero [nz]
  const double* p_nonlinear; // nonlinear power [nz,nk], row-major
  const double* chi;         // comoving distance chi(z) [nz]
  const double* dz_dchi;     // Jacobian dz/dchi [nz]
  const double* growth;      // growth factor D(z), D(0) = 1 [nz]
  const double* sources; // densities per unit redshift [nbin,nsource]
  const double* los_z;   // line-of-sight node redshifts [nlos]
  double omega_m;  // matter density fraction today
  double omega_de; // dark-energy density fraction today
  double w0;       // dark-energy equation of state today
  double wa;       // its evolution, w(a) = w0 + wa (1 - a)
  double ns;       // primordial spectral index
  double sigma8;   // legacy interface scalar; never rescales supplied power
  int source_readers; // 1 keeps cubic source reads for comparisons
  int table_boost; // dense-grid refinement factor; zero means eight
  int fit_nz; // retained ABI field; fit sampling belongs to the provider
  // Optional independent source coordinates. A NULL source_z with
  // nsource = 0 reuses the background z grid and keeps its exact
  // preparation arithmetic. Source resolution never changes the
  // number of nonlinear matter-power rows above.
  int nsource;            // source nodes; zero reuses nz
  const double* source_z; // source redshifts [nsource], or NULL
  const feedback_input_3pcf* feedback; // optional paired response; copied
  int nwindow; // zero disables the angular pixel window
  const double* window; // W(ell) at integers 0..nwindow-1; copied
  // Required theory-supplied matter B. The provider owns BiHalofit or any
  // replacement prescription. omega_de/w0/wa/ns are retained ABI fields,
  // unused here; the supplied flat background owns the expansion history.
  const baseline_input_3pcf* baseline; // copied absolute matter B table
  const double* omega_m_z; // required exact matter fraction for halo covariance
  int nomega; // zero reuses nz; otherwise independent matter-fraction nodes
  const double* omega_m_redshift; // optional piecewise-uniform axis [nomega]
  // Retained ABI field; smoothing quantities belong to the theory producer.
  const physics_variance_input_3pcf* variance; // unused by this model
} physics_input_3pcf;

// TATT (tidal alignment and tidal torquing) parameters in the
// convention of Gomes et al. (arXiv:2601.09133) as fastnc implements
// it, plus the baryon controls of the matter bispectrum. DMO means
// dark-matter-only.
typedef struct {
  double a1;     // tidal-alignment amplitude A_1
  double eta1;   // redshift power of A_1 about the pivot
  double a2;     // tidal-torque amplitude A_2
  double eta2;   // redshift power of A_2 about the pivot
  double bta;    // density-weighting amplitude b_TA
  double ct;     // velocity-shear amplitude C_t (see ct_mode)
  double zpivot; // pivot redshift of the (1 + z) power laws
  double c1rho;  // C_1 rho_crit normalization; 0.01389 by default
  double fb;     // fraction interpolating DMO to the baryon-fit ratio
  int power_mode; // 0: linear D^2; 1: nonlinear(z=0) D^2; 2: nonlinear(z)
  int ct_mode; // 0: physical Ct=C1*ct (fastnc); 1: physical Ct=ct
  int suppress_only; // cap the baryon transfer at one before scaling fb
  // Diagnostics select one physical operator before projection. Zero
  // keeps the complete prediction; 1 = NLA (nonlinear alignment, the
  // linear C_1 part alone), 2 = density weighting, 3 = tidal torque,
  // 4 = velocity shear. NLA includes the matter term with three
  // lensing legs (GGG; intrinsic_legs below defines the letters)
  // until intrinsic_legs selects a single placement.
  int operator_component;
  // -1 keeps all placements. Bits 0, 1 and 2 put the intrinsic field
  // I on the corresponding ordered leg; an unset bit keeps the
  // lensing field G. Thus 4 = GGI and 3 = IIG. The selector affects
  // projection only, never the nineteen unprojected IA entries.
  int intrinsic_legs;
  double feedback_amplitude; // shared log-response amplitude; zero is DMO
} physics_ia_3pcf;

typedef struct physics_model_3pcf physics_model_3pcf;
typedef struct physics_projection_3pcf physics_projection_3pcf;
typedef struct physics_projection_batch_3pcf physics_projection_batch_3pcf;

physics_model_3pcf* physics_create_3pcf(
  const physics_input_3pcf* input
);
void physics_destroy_3pcf(physics_model_3pcf* model);
void physics_default_ia_3pcf(physics_ia_3pcf* parameters);

// Angular smoothing applies to observables, never to the logged
// matter interpolation tables. The window convention: samples sit at
// integer ell, a read interpolates linearly between neighboring
// integers, and every ell above the last sample returns zero rather
// than a taper.
double physics_window_value_3pcf(int count, const double* window, double ell);
const double* physics_window_samples_3pcf(
  const physics_model_3pcf* model,
  int* count
);

double physics_power_3pcf(
  const physics_model_3pcf* model,
  double k,
  double z,
  int mode
);

// Return the positive paired power response. The model's ordinary
// power reader keeps the DMO spectra, which the perturbative IA and
// halo calculations require.
double physics_feedback_power_3pcf(
  const physics_model_3pcf* model,
  double k,
  double z,
  double amplitude
);

double physics_feedback_bispectrum_3pcf(
  const physics_model_3pcf* model,
  const double* sides,
  double z,
  double amplitude
);

// component: 0 = supplied full matter B, 1 = tree level from P_linear.
// Legacy components 2/3 are refused; C owns no matter-fit decomposition.
// A supplied full B cannot define the fitted pieces 2/3, so those requests
// stop explicitly. The covariance's independent halo integrals do not
// request those fitted pieces. Optional fb still multiplies the selected B;
// leave it zero if the supplied table already contains the same feedback.
double physics_matter_3pcf(
  const physics_model_3pcf* model,
  double k1,
  double k2,
  double k3,
  double z,
  int component,
  double fb
);

// The 19 entries follow the active fastnc tensor calculation, in
// order: ddE,dEd,Edd,dEE,EEd,EdE,EEE,ddB,dBd,Bdd,dEB,dBE,EBd,BEd,
// BdE,EdB,EEB,EBE,BEE. d denotes matter density; E and B are the two
// intrinsic shape modes. remove_alignment removes C1^p B_tree, with p
// the number of shape legs, from the seven all-E entries only. A supplied
// full matter B does not alter these perturbative IA templates. It enters
// the projected NLA baseline, while the signed quadratic residuals keep
// their existing power-spectrum prescription.
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

// Evaluate physical triangles that differ only by a positive common
// side scaling. All 19 IA components share one orientation and F2
// preparation. sides is [npoint,3]; z is [npoint]; out is
// [npoint,19].
void physics_ia_same_shape_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  int npoint,
  const double* sides,
  const double* z,
  int remove_alignment,
  double* out
);

// A projection owns radial weights for one ordered source-bin
// triplet and one IA parameter choice. It borrows its immutable
// model, which must outlive it. Both objects support simultaneous
// read-only calls from OpenMP workers.
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

// Evaluation for corners of fastnc's rectangular PUV interpolation
// grid: the triangle-coordinate table over the perimeter
// p = ell1 + ell2 + ell3 and the side fractions u = ell1/p and
// v = ell2/p, whose rectangular corners include side combinations
// that close no triangle. Off-triangle corners follow the active IA
// vector clipping, and the zero closing-side corner returns zero.
// Physical direct queries keep the strict triangle guard above.
double physics_projected_puv_extension_3pcf(
  const physics_projection_3pcf* projection,
  double ell1,
  double ell2,
  double ell3
);

// Batch projection requires the same immutable model and IA
// parameters for every projector; only the ordered source triplets
// differ. The function contracts one set of physical vertices with
// all radial weights at each redshift.
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

// Prepare one immutable batch before launching triangle workers. The
// batch owns packed radial weights [node, 8 terms, triplet] and
// borrows the first projector and its model, so destroy the batch
// before either borrowed owner. Every projector must use that same
// model and exactly equal IA settings; the ordered source bins may
// differ. During evaluation only the output buffers are mutable, and
// they must not overlap the packed weights or any physical input.
physics_projection_batch_3pcf* physics_projection_batch_create_3pcf(
  int nprojection, // number of ordered source triplets
  physics_projection_3pcf* const* projections // consumed during construction
);

void physics_projection_batch_evaluate_3pcf(
  const physics_projection_batch_3pcf* batch, // shared immutable preparation
  double ell1, // first angular Fourier side
  double ell2, // second angular Fourier side
  double ell3, // closing side; puv_extension below governs its corners
  int component, // total=0, NLA baseline=1, residual IA=2
  int puv_extension, // nonzero keeps fastnc's PUV corner continuation
  double* out // [nprojection], one private buffer per caller
);

void physics_projection_batch_destroy_3pcf(
  physics_projection_batch_3pcf* batch // releases owned weights; NULL allowed
);

// The shared Fourier integrator accepts either a direct physical
// projection or an immutable prepared table through this callback.
// The callback returns the total ordered E-mode bispectrum and must
// support concurrent read-only evaluations.
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

// b_L: the Legendre coefficients of the projected E-mode bispectrum
// in mu, the cosine of the angle between the two Fourier multipole
// legs:
//
//   b_L = (2L+1)/2 integral[-1,1] dmu P_L(mu) B(ell_i,ell_j,ell_3),
//   ell_3^2 = ell_i^2 + ell_j^2 - 2 ell_i ell_j mu.
//
// No Limber +1/2 shift enters any leg. out has shape
// [lmax+1,nell,nell], and the function imposes no leg-sorting
// symmetry.
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

// Full compensated-Gaussian aperture mean, with the Fourier filter
// U(ell R) = (ell R)^2 exp[-(ell R)^2/2]/2. filters is [nfilter,3]
// and keeps the ordered source legs. Explicit positive quadratures
// integrate dell1, dell2 and dphi.
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

// The covariance shares the same physical tables through these
// borrowed arrays, valid for the model's lifetime. dchi holds the
// trapezoidal node weights; lens and shape are the lensing efficiency
// g and the source kernel W_g of the fastnc convention, each
// [nbin,nlos].
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

// The Limber angular power of the combined lensing + NLA intrinsic
// kernel G + C_1 W at k = ell/chi, using the nonlinear power
// interpolated in redshift; a nonzero feedback amplitude applies its
// baryon response ratio per ell. physics_nla_power_3pcf is the
// single-multipole wrapper of the batch form below.
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
