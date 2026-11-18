#ifndef COSMOLIKE_PHYSICS_GRID_3PCF_H
#define COSMOLIKE_PHYSICS_GRID_3PCF_H

#include "physics_3pcf.h"

#ifdef __cplusplus
extern "C" {
#endif

// Every setting contributes to a prepared table's identity. The object is
// immutable: changing any setting, source bin, model or IA value requires
// a new object. Counts specify nodes, except the log portions whose last
// node is shared with the subsequent linear portion, as in fastnc.
typedef struct {
  double ell_min;
  double ell_max;
  double epmu;
  int nr;
  int nu;
  int nv;
  int np;
  int npu_log;
  int npu_linear;
  int nell;
  int npsi_log;
  int npsi_linear;
  int nmu_log;
  int nmu_linear;
  int lmax;
  int reference_mode; // 1 preserves source irregular grids and linear reads
  int table_boost; // natural-cubic dense-grid factor for RUV/PUV tables
  int multipole_boost; // dense-grid factor for b_L(log ELL,log small PSI)
  int npsi; // production coarse uniform log-small-PSI node count
  int prepare_multipoles; // zero omits b_L when only full filters are needed
  int nshape; // production uniform bounded-tanh shape-coordinate count
  int signed_baseline; // 1 stores log GGG and the signed ratio NLA/GGG
} physics_grid_settings_3pcf;

typedef struct physics_grid_3pcf physics_grid_3pcf;
typedef struct physics_grid_query_3pcf physics_grid_query_3pcf;

void physics_grid_defaults_3pcf(physics_grid_settings_3pcf* settings);
physics_grid_3pcf* physics_grid_create_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const int* bins,
  const physics_grid_settings_3pcf* settings
);
void physics_grid_create_many_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  int ntriplet,
  const int* bins, // [ntriplet,3], ordered source legs
  const physics_grid_settings_3pcf* settings,
  physics_grid_3pcf** out // caller owns ntriplet returned objects
);
void physics_grid_destroy_3pcf(physics_grid_3pcf* grid);
double physics_grid_projected_3pcf(
  const physics_grid_3pcf* grid,
  double ell1,
  double ell2,
  double ell3,
  int component
);
void physics_grid_aperture_full_3pcf(
  const physics_grid_3pcf* grid,
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
void physics_grid_multipoles_3pcf(
  const physics_grid_3pcf* grid,
  int nell,
  const double* ell,
  int component,
  double* out // [lmax+1,nell,nell], same convention as direct multipoles
);

// A query owns interpolation geometry for one fixed caller ell array.
// It borrows neither the source grid nor its physical multipoles, so the
// source grid may be destroyed after construction. Matching compares the
// complete interpolation axes, their sizes, L cutoff and reader convention.
// A changed cosmology or source triplet can reuse this geometry when those
// keys agree exactly; evaluation always reads the supplied grid's fresh b_L.
// Changing the caller's ell values requires a new query even at the same size.
physics_grid_query_3pcf* physics_grid_query_create_3pcf(
  const physics_grid_3pcf* grid, // prepared b_L grid defining interpolation
  int nell, // number of requested Fourier values on each ordered leg
  const double* ell // [nell], positive values consumed during construction
);
int physics_grid_query_matches_3pcf(
  const physics_grid_query_3pcf* query, // immutable owned geometry
  const physics_grid_3pcf* grid // returns zero for NULL or incompatible input
);
int physics_grid_query_nell_3pcf(
  const physics_grid_query_3pcf* query // requested output Fourier size
);
int physics_grid_query_lmax_3pcf(
  const physics_grid_query_3pcf* query // maximum retained Legendre degree
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
