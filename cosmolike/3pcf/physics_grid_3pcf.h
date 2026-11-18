#ifndef COSMOLIKE_PHYSICS_GRID_3PCF_H
#define COSMOLIKE_PHYSICS_GRID_3PCF_H

#include "physics_3pcf.h"

#ifdef __cplusplus
extern "C" {
#endif

// Every setting contributes to a prepared table's identity. The object is
// immutable: changing any setting, source bin, model or IA value requires
// a new object. Counts specify nodes, except that a log portion hands its
// last node to the linear portion that follows it, as in fastnc.
//
// The settings name two triangle charts that physics_grid_3pcf.c
// derives. RUV sorts the triangle sides - r is the middle side,
// u = smallest/middle and v = (largest-middle)/smallest - and holds the
// baseline of the NLA model, the nonlinear-alignment intrinsic-alignment
// model. PUV keeps the ordered sides - p is the perimeter, u = ell1/p
// and v = ell2/p - and holds the residual of the TATT model, the
// tidal-alignment-tidal-torque extension. GGG denotes the pure-gravity
// three-shear correlation, which the signed storage modes use as a
// positive reference. Two angles recur below: PSI, the opening angle
// arctan(ell2/ell1) of the two Fourier multipole legs, and mu, the
// cosine of the angle between those legs, in which the Legendre
// multipoles b_L expand the projected bispectrum. epmu caps that cosine
// at mu <= 1 - epmu, keeping the closing side of each sampled triangle
// away from its degenerate limit.
typedef struct {
  double ell_min; // smallest Fourier multipole on either leg
  double ell_max; // largest Fourier multipole on either leg
  double epmu; // caps mu, the two-leg angle cosine, at 1 - epmu
  double shape_stretch; // bounded-tanh concentration near squeezed edges
  double psi_scale; // 0=log PSI; positive=log PSI+PSI/scale in production
  int nr; // nodes in log r, the middle side of the sorted RUV table
  int nu; // nodes in log u, the smallest/middle side ratio
  int nv; // nodes in v = (largest-middle)/smallest, linear in [0,1]
  int np; // nodes in log p, the PUV residual perimeter
  int npu_log; // log-spaced PUV shape nodes beside each squeezed edge
  int npu_linear; // uniform PUV shape nodes across the central region
  int nell; // nodes in log ELL, the two-leg magnitude of the b_L table
  int npsi_log; // log-spaced PSI nodes below the 1e-3 rad join
  int npsi_linear; // uniform PSI nodes from the join up to pi/4
  int nmu_log; // log-spaced nodes in 1-mu against the epmu cap
  int nmu_linear; // uniform nodes in 1-mu over the rest of [-1,1]
  int lmax; // largest Legendre degree kept in the b_L table
  int reference_mode; // 1 preserves source irregular grids and linear reads
  int table_boost; // natural-cubic dense-grid factor for RUV/PUV tables
  int multipole_boost; // dense-grid factor for b_L(log ELL,angular chart)
  int npsi; // production node count uniform in the chosen angular chart
  int prepare_multipoles; // zero omits b_L when only full filters are needed
  int nshape; // production uniform bounded-tanh shape-coordinate count
  int signed_baseline; // 1 stores log GGG and the signed ratio NLA/GGG
  int residual_scaling; // 1 scales PUV by positive GGG at equal p/3 sides
} physics_grid_settings_3pcf;

typedef struct physics_grid_3pcf physics_grid_3pcf;
typedef struct physics_grid_query_3pcf physics_grid_query_3pcf;

// Fill the settings with the fastnc reference resolution, so a default
// grid reproduces the pinned source's sampling.
void physics_grid_defaults_3pcf(physics_grid_settings_3pcf* settings);

// Create one prepared grid for a single ordered source triplet.
physics_grid_3pcf* physics_grid_create_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const int* bins, // [3], ordered source legs
  const physics_grid_settings_3pcf* settings
);

// Create one grid per ordered source triplet, sharing every expensive
// line-of-sight evaluation across the batch.
void physics_grid_create_many_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  int ntriplet,
  const int* bins, // [ntriplet,3], ordered source legs
  const physics_grid_settings_3pcf* settings,
  physics_grid_3pcf** out // caller owns ntriplet returned objects
);

// Release a grid and every array it owns; NULL is allowed.
void physics_grid_destroy_3pcf(physics_grid_3pcf* grid);

// Interpolate the projected E-mode bispectrum at one triangle of
// Fourier multipoles; callers multiply any pixel window afterwards.
double physics_grid_projected_3pcf(
  const physics_grid_3pcf* grid, // prepared immutable tables
  double ell1, // first triangle side, a positive Fourier multipole
  double ell2, // second triangle side
  double ell3, // closing third side; the three must form a triangle
  int component // total=0, NLA baseline=1, residual IA=2
);

// Full compensated-filter aperture statistic with this grid standing in
// for the direct bispectrum provider; physics_3pcf.h declares the
// quadrature measure and the filter convention.
void physics_grid_aperture_full_3pcf(
  const physics_grid_3pcf* grid, // prepared tables replacing direct B
  int nell, // quadrature nodes on each Fourier leg
  const double* ell, // [nell], increasing leg-magnitude nodes
  const double* dell, // [nell], leg-magnitude quadrature weights
  int nphi, // quadrature nodes in the angle between the legs
  const double* phi, // [nphi], angles between the two legs, radians
  const double* dphi, // [nphi], angular quadrature weights
  int nfilter, // number of aperture filter triples
  const double* filters, // [nfilter,3], radii for the ordered legs
  double* out // [nfilter], one aperture value per filter triple
);

// b_L at every ordered (ell1,ell2) pair, read from the prepared table;
// physics_multipoles_3pcf in physics_3pcf.h declares the convention.
void physics_grid_multipoles_3pcf(
  const physics_grid_3pcf* grid,
  int nell,
  const double* ell,
  int component, // total=0, NLA baseline=1, residual IA=2
  double* out // [lmax+1,nell,nell], same convention as direct multipoles
);

// A query owns interpolation geometry for one fixed caller ell array.
// It borrows neither the source grid nor its physical multipoles, so the
// caller may destroy the source grid after construction. Matching
// compares the complete physical and chart axes, their sizes, L cutoff
// and chart choice. A changed cosmology or source triplet can reuse this
// geometry when those keys agree exactly; evaluation always reads fresh
// b_L values from the grid the caller passes. Changing the caller's ell
// values requires a new query even at the same size.
physics_grid_query_3pcf* physics_grid_query_create_3pcf(
  const physics_grid_3pcf* grid, // prepared b_L grid defining interpolation
  int nell, // number of requested Fourier values on each ordered leg
  const double* ell // [nell], positive multipoles; the query copies them
);

int physics_grid_query_matches_3pcf(
  const physics_grid_query_3pcf* query, // immutable owned geometry
  const physics_grid_3pcf* grid // returns zero for NULL or incompatible input
);

int physics_grid_query_nell_3pcf(
  const physics_grid_query_3pcf* query // requested output Fourier size
);

int physics_grid_query_lmax_3pcf(
  const physics_grid_query_3pcf* query // largest Legendre degree it keeps
);

void physics_grid_query_evaluate_3pcf(
  const physics_grid_query_3pcf* query, // immutable charts, cells and weights
  const physics_grid_3pcf* grid, // matching geometry, current physical values
  int component, // total=0, NLA baseline=1, residual IA=2
  double* out // [lmax+1,nell,nell], must not alias grid or query storage
);

void physics_grid_query_destroy_3pcf(
  physics_grid_query_3pcf* query // release owned arrays; NULL is allowed
);

// Expose the actual interpolation coordinates for diagnostics and records.
// axis: 0=log_r,1=log_u,2=v,3=log_p,4=pu,5=log_ELL,6=log_small_PSI,7=mu.
void physics_grid_axis_3pcf(
  const physics_grid_3pcf* grid,
  int axis,
  int* count,
  const double** values
);

#ifdef __cplusplus
}
#endif
#endif
