#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "threepcf_core_bridge.h"
#include "cosmolike/basics.h"
#include "cosmolike/structs.h"
#include "cosmolike/IA.h"
#include "log.c/src/log.h"

// ===========================================================================
// CORE-STRUCT BRIDGE OF THE THREE-POINT PHYSICS MODEL
// ===========================================================================
// threepcf_core_bridge.h states the contract: this file reads the
// global Cosmolike structs, adapts units and layout once at the
// boundary, and assembles the physics_input_3pcf that the immutable
// three-point model is created from. The model's own files stay
// untouched; everything specific to the struct world lives here.
//
// The file has four parts, in order: small local helpers, the
// supplied-tier setters, the struct-read conversions with the cached
// input factory, and the core struct installers the standalone module
// uses in place of generic_interface.cpp.

// ---------------------------------------------------------------------------
// Local helpers.
// ---------------------------------------------------------------------------

// Process-unique cache keys. The full core draws its keys from the
// C++ RandomNumber singleton in generic_interface.cpp; this C file
// cannot reach it, so a monotonic counter stands in. Uniqueness is
// the only property the cache guards need: a key must never repeat a
// value a consumer may still hold. The counter therefore never
// resets, not even in reset_threepcf_core_bridge.
static uint64_t bridge_draw(void)
{
  static uint64_t counter = 0;
  counter++;
  return counter;
}

// Exact element-wise difference test for the supplied tier. The
// supplied setters compare bitwise on purpose: a rounded tolerance
// (the fdiff of basics.h) could leave the nonce unchanged after a
// genuine sub-tolerance parameter step, and the cached model would
// silently keep the previous physics. NaN never reaches this loop:
// every setter scans its input first and aborts on NaN.
static int bridge_differs(
  const double* stored, // [n] values the bridge holds
  const double* incoming, // [n] values a setter received
  const int n // entry count
)
{
  for (int i=0; i<n; i++) {
    if (stored[i] != incoming[i]) {
      return 1;
    }
  }
  return 0;
}

// Abort on any NaN before a comparison or copy touches the values.
static void bridge_require_finite(
  const double* values, // [n] caller values
  const int n, // entry count
  const char* name // array name for the error message
)
{
  for (int i=0; i<n; i++) {
    if (isnan(values[i])) {
      log_fatal("threepcf bridge: %s contains NaN at entry %d", name, i);
      exit(1);
    }
  }
}

// Owned copy of a caller array; the bridge frees it on replacement
// and on reset. malloc1d (basics.c) aborts on allocation failure.
static double* bridge_copy(
  const double* values, // [n] caller values
  const int n // entry count
)
{
  double* copy = (double*) malloc1d(n);
  memcpy(copy, values, sizeof(double)*(size_t) n);
  return copy;
}

// ---------------------------------------------------------------------------
// Supplied-tier storage. One struct holds every owned copy and scalar
// the structs do not provide, plus the single bridge nonce. The
// resolution and prescription defaults repeat the Python
// constructor's and physics_default_ia_3pcf's defaults, so a caller
// who never touches a setter gets the same model the Python route
// builds with default arguments.
// ---------------------------------------------------------------------------
static struct {
  uint64_t nonce;     // redrawn by every setter whose value changed
  int have_scalars;   // ns/sigma8/w0/wa installed at least once
  double ns;          // primordial spectral index
  double sigma8;      // linear fluctuation amplitude at z = 0
  double w0;          // dark-energy equation of state today
  double wa;          // its evolution in w(a) = w0 + wa (1 - a)
  int nlos;           // line-of-sight node count; zero = not set
  double* los;        // owned node redshifts [nlos]
  int kernel_nz;      // physics_input_3pcf.kernel_nz
  int source_readers; // physics_input_3pcf.source_readers
  int table_boost;    // physics_input_3pcf.table_boost
  int fit_nz;         // physics_input_3pcf.fit_nz
  int nbin;           // supplied source bins; zero = struct-read path
  int nsource;        // supplied source nodes
  double* source_z;   // owned source redshifts [nsource]
  double* sources;    // owned densities [nbin*nsource], row-major
  int nwindow;        // pixel-window samples; zero = no window
  double* window;     // owned W(ell) samples [nwindow]
  struct {
    int nz;           // feedback redshift nodes; zero = no response
    int nk;           // feedback log k nodes
    int nu;           // feedback log u nodes
    int nv;           // feedback v nodes
    int boost;        // dense intervals per loaded interval
    int clamp;        // boundary policy (feedback_3pcf.h)
    double amplitude; // shared log-response amplitude for the IA dict
    double* z;        // owned axes and tables, as in feedback_3pcf.h
    double* logk;
    double* logu;
    double* v;
    double* log_power;
    double* log_bispectrum;
  } feedback;
  int power_mode;     // physics_ia_3pcf.power_mode
  int ct_mode;        // physics_ia_3pcf.ct_mode
  int suppress_only;  // physics_ia_3pcf.suppress_only
} supplied = {
  .kernel_nz = 101,
  .source_readers = 0,
  .table_boost = 8,
  .fit_nz = 33,
  .power_mode = 1,
  .ct_mode = 0,
  .suppress_only = 0,
};

// Free every owned supplied-tier array and restore the defaults the
// initializer above states. free(NULL) is a no-op, so a fresh bridge
// passes through unchanged.
static void supplied_clear(void)
{
  free(supplied.los);
  free(supplied.source_z);
  free(supplied.sources);
  free(supplied.window);
  free(supplied.feedback.z);
  free(supplied.feedback.logk);
  free(supplied.feedback.logu);
  free(supplied.feedback.v);
  free(supplied.feedback.log_power);
  free(supplied.feedback.log_bispectrum);
  memset(&supplied, 0, sizeof(supplied));
  supplied.kernel_nz = 101;
  supplied.source_readers = 0;
  supplied.table_boost = 8;
  supplied.fit_nz = 33;
  supplied.power_mode = 1;
  supplied.ct_mode = 0;
  supplied.suppress_only = 0;
}

// ---------------------------------------------------------------------------
// Supplied-tier setters. Shared discipline (threepcf_core_bridge.h):
// NaN scan first, exact comparison against the stored copy, owned
// replacement and one nonce redraw only when something changed.
// ---------------------------------------------------------------------------

void threepcf_bridge_set_scalars(
  const double ns, // primordial spectral index
  const double sigma8, // linear fluctuation amplitude at z = 0
  const double w0, // dark-energy equation of state today
  const double wa // its evolution in w(a) = w0 + wa (1 - a)
)
{
  const double values[4] = {ns, sigma8, w0, wa};
  bridge_require_finite(values, 4, "scalars (ns, sigma8, w0, wa)");
  if (!(sigma8 > 0.0)) {
    log_fatal("threepcf bridge: sigma8 = %e must be positive", sigma8);
    exit(1);
  }
  const double stored[4] = {supplied.ns, supplied.sigma8, supplied.w0,
                            supplied.wa};
  if (!supplied.have_scalars || bridge_differs(stored, values, 4)) {
    supplied.ns = ns;
    supplied.sigma8 = sigma8;
    supplied.w0 = w0;
    supplied.wa = wa;
    supplied.have_scalars = 1;
    supplied.nonce = bridge_draw();
  }
}

void threepcf_bridge_set_los(
  const double* z, // node redshifts [n]
  const int n // node count
)
{
  if (NULL == z
      || n < 2) {
    log_fatal("threepcf bridge: the line of sight needs at least two "
              "nodes (physics_3pcf.h)");
    exit(1);
  }
  bridge_require_finite(z, n, "los_z");
  if (supplied.nlos != n || bridge_differs(supplied.los, z, n)) {
    free(supplied.los);
    supplied.los = bridge_copy(z, n);
    supplied.nlos = n;
    supplied.nonce = bridge_draw();
  }
}

void threepcf_bridge_set_resolution(
  const int kernel_nz, // internal lensing-kernel redshift resolution
  const int source_readers, // 1 keeps cubic source reads for comparisons
  const int table_boost, // dense-grid refinement factor; zero means eight
  const int fit_nz // uniform BiHalofit anchor count; zero means 33
)
{
  // The model validates each knob's range itself (physics_3pcf.h);
  // the bridge only keeps the values and tracks changes.
  if (supplied.kernel_nz != kernel_nz ||
      supplied.source_readers != source_readers ||
      supplied.table_boost != table_boost ||
      supplied.fit_nz != fit_nz) {
    supplied.kernel_nz = kernel_nz;
    supplied.source_readers = source_readers;
    supplied.table_boost = table_boost;
    supplied.fit_nz = fit_nz;
    supplied.nonce = bridge_draw();
  }
}

void threepcf_bridge_set_sources(
  const double* source_z, // source redshifts [nsource], increasing
  const double* sources, // densities per unit z [nbin, nsource]
  const int nbin, // tomographic source bins
  const int nsource // source redshift nodes
)
{
  if (NULL == source_z && NULL == sources && 0 == nbin && 0 == nsource) {
    // clear the override; the struct-read path takes over
    if (0 != supplied.nbin) {
      free(supplied.source_z);
      free(supplied.sources);
      supplied.source_z = NULL;
      supplied.sources = NULL;
      supplied.nbin = 0;
      supplied.nsource = 0;
      supplied.nonce = bridge_draw();
    }
    return;
  }
  if (NULL == source_z
      || NULL == sources
      || nbin < 1
      || nsource < 2) {
    log_fatal("threepcf bridge: supplied sources need both arrays, at "
              "least one bin and at least two source nodes");
    exit(1);
  }
  bridge_require_finite(source_z, nsource, "source_z");
  bridge_require_finite(sources, nbin*nsource, "sources");
  if (supplied.nbin != nbin || supplied.nsource != nsource ||
      bridge_differs(supplied.source_z, source_z, nsource) ||
      bridge_differs(supplied.sources, sources, nbin*nsource)) {
    free(supplied.source_z);
    free(supplied.sources);
    supplied.source_z = bridge_copy(source_z, nsource);
    supplied.sources = bridge_copy(sources, nbin*nsource);
    supplied.nbin = nbin;
    supplied.nsource = nsource;
    supplied.nonce = bridge_draw();
  }
}

void threepcf_bridge_set_pixel_window(
  const double* window, // W(ell) samples [n], or NULL
  const int n // sample count, or zero
)
{
  if (NULL == window && 0 == n) {
    if (0 != supplied.nwindow) {
      free(supplied.window);
      supplied.window = NULL;
      supplied.nwindow = 0;
      supplied.nonce = bridge_draw();
    }
    return;
  }
  if (NULL == window
      || n < 2) {
    log_fatal("threepcf bridge: the pixel window needs at least two "
              "integer-ell samples (physics_3pcf.h)");
    exit(1);
  }
  bridge_require_finite(window, n, "pixel window");
  if (supplied.nwindow != n || bridge_differs(supplied.window, window, n)) {
    free(supplied.window);
    supplied.window = bridge_copy(window, n);
    supplied.nwindow = n;
    supplied.nonce = bridge_draw();
  }
}

void threepcf_bridge_set_feedback(
  const double* z, // redshift nodes [nz]
  const int nz, // redshift count
  const double* logk, // log(k/[h/Mpc]) nodes [nk]
  const int nk, // wave-number count
  const double* logu, // log(u) nodes [nu]
  const int nu, // side-ratio count
  const double* v, // v nodes [nv]
  const int nv, // shape-coordinate count
  const double* log_power, // ln R_P [nz, nk]
  const double* log_bispectrum, // ln R_B [nz, nk, nu, nv]
  const int boost, // dense intervals per loaded interval
  const int clamp, // 0 refuses out-of-domain; 1 clamps
  const double amplitude // shared log-response amplitude
)
{
  if (NULL == z && NULL == logk && NULL == logu && NULL == v &&
      0 == nz && 0 == nk && 0 == nu && 0 == nv) {
    if (0 != supplied.feedback.nz || 0.0 != supplied.feedback.amplitude) {
      free(supplied.feedback.z);
      free(supplied.feedback.logk);
      free(supplied.feedback.logu);
      free(supplied.feedback.v);
      free(supplied.feedback.log_power);
      free(supplied.feedback.log_bispectrum);
      memset(&supplied.feedback, 0, sizeof(supplied.feedback));
      supplied.nonce = bridge_draw();
    }
    return;
  }
  // feedback_create_3pcf requires at least four nodes per axis for
  // its cubic preparation; refuse earlier with the same bound
  if (NULL == z
      || NULL == logk
      || NULL == logu
      || NULL == v
      || NULL == log_power
      || NULL == log_bispectrum
      || nz < 4
      || nk < 4
      || nu < 4
      || nv < 4) {
    log_fatal("threepcf bridge: the paired feedback response needs six "
              "arrays with at least four nodes per axis (feedback_3pcf.h)");
    exit(1);
  }
  if (isnan(amplitude)) {
    log_fatal("threepcf bridge: feedback amplitude is NaN");
    exit(1);
  }
  bridge_require_finite(z, nz, "feedback z");
  bridge_require_finite(logk, nk, "feedback logk");
  bridge_require_finite(logu, nu, "feedback logu");
  bridge_require_finite(v, nv, "feedback v");
  bridge_require_finite(log_power, nz*nk, "feedback log_power");
  bridge_require_finite(log_bispectrum, nz*nk*nu*nv,
                        "feedback log_bispectrum");
  if (supplied.feedback.nz != nz || supplied.feedback.nk != nk ||
      supplied.feedback.nu != nu || supplied.feedback.nv != nv ||
      supplied.feedback.boost != boost ||
      supplied.feedback.clamp != clamp ||
      supplied.feedback.amplitude != amplitude ||
      bridge_differs(supplied.feedback.z, z, nz) ||
      bridge_differs(supplied.feedback.logk, logk, nk) ||
      bridge_differs(supplied.feedback.logu, logu, nu) ||
      bridge_differs(supplied.feedback.v, v, nv) ||
      bridge_differs(supplied.feedback.log_power, log_power, nz*nk) ||
      bridge_differs(supplied.feedback.log_bispectrum, log_bispectrum,
                     nz*nk*nu*nv)) {
    free(supplied.feedback.z);
    free(supplied.feedback.logk);
    free(supplied.feedback.logu);
    free(supplied.feedback.v);
    free(supplied.feedback.log_power);
    free(supplied.feedback.log_bispectrum);
    supplied.feedback.z = bridge_copy(z, nz);
    supplied.feedback.logk = bridge_copy(logk, nk);
    supplied.feedback.logu = bridge_copy(logu, nu);
    supplied.feedback.v = bridge_copy(v, nv);
    supplied.feedback.log_power = bridge_copy(log_power, nz*nk);
    supplied.feedback.log_bispectrum = bridge_copy(log_bispectrum,
                                                   nz*nk*nu*nv);
    supplied.feedback.nz = nz;
    supplied.feedback.nk = nk;
    supplied.feedback.nu = nu;
    supplied.feedback.nv = nv;
    supplied.feedback.boost = boost;
    supplied.feedback.clamp = clamp;
    supplied.feedback.amplitude = amplitude;
    supplied.nonce = bridge_draw();
  }
}

void threepcf_bridge_set_prescriptions(
  const int power_mode, // 0 linear D^2; 1 nonlinear(z=0) D^2; 2 full table
  const int ct_mode, // 0 physical Ct = C1*ct (fastnc); 1 Ct = ct
  const int suppress_only // cap the baryon transfer at one before scaling
)
{
  if (power_mode < 0
      || power_mode > 2
      || ct_mode < 0
      || ct_mode > 1
      || suppress_only < 0
      || suppress_only > 1) {
    log_fatal("threepcf bridge: prescription selectors out of range "
              "(physics_3pcf.h: power_mode 0..2, ct_mode 0..1, "
              "suppress_only 0..1)");
    exit(1);
  }
  if (supplied.power_mode != power_mode ||
      supplied.ct_mode != ct_mode ||
      supplied.suppress_only != suppress_only) {
    supplied.power_mode = power_mode;
    supplied.ct_mode = ct_mode;
    supplied.suppress_only = suppress_only;
    supplied.nonce = bridge_draw();
  }
}

// ---------------------------------------------------------------------------
// IA parameter translation: the single conversion site.
//
// Both the ordinary two-point core and the three-point module write
// the tidal-alignment weight as
//
//   C_1(z) = -A_1 ((1+z)/(1+z_pivot))^eta_1 c1rho Omega_m / D(z),
//
// with D the linear growth factor and c1rho the C_1 rho_crit
// normalization; the tidal-torque weight C_2 carries the same
// structure with its own amplitude, exponent and an extra factor
// present in both codes. The two conventions differ only in their
// fixed constants: the core stores nuisance.c1rhocrit_ia = 0.01389
// and the pivot nuisance.oneplusz0_ia = 1.62 (set_nuisance_IA pins
// both), while the module defaults to c1rho = 0.0134 with a free
// pivot (physics_default_ia_3pcf). Matching the physical C_1 at every
// redshift therefore means, with z_pivot taken from the core's pivot,
//
//   a1_module = A1_core / [ (c1rho/0.01389)
//                           * ((1 + 0.62)/(1 + z_pivot))^eta_1 ],
//
// the exact inverse of the module-to-core translation the project's
// des3pcf.core_two_point.core_ia_parameters applies; a2 follows with
// eta_2. With z_pivot = oneplusz0_ia - 1 the power-law base is one,
// so only the c1rho ratio acts, but the formula keeps the general
// pivot term so the two translation sites stay inverses line by line.
//
// The remaining slots translate without factors: b_TA (ia[2][0]) and
// the velocity-shear amplitude C_t (ia[1][2]) are dimensionless
// amplitudes defined by the same Gomes et al. (arXiv:2601.09133)
// convention in both codes, and no two-point path reads C_t at all.
// ---------------------------------------------------------------------------
void threepcf_bridge_ia_from_core(
  physics_ia_3pcf* parameters // receives the translated parameters
)
{
  if (NULL == parameters) {
    log_fatal("threepcf bridge: missing IA parameter output");
    exit(1);
  }
  // Registry guard (structs.h): under IA_REDSHIFT_BINNING the slot
  // ia[1][2] holds the per-bin tidal-torque amplitude A2 of shear
  // bin 2, so reading it as C_t would run the wrong physics. The
  // translation above is the power-law (IA_REDSHIFT_EVOLUTION) map,
  // and the module implements TATT operators, so both flags must
  // say so.
  if (IA_REDSHIFT_BINNING == nuisance.IA) {
    log_fatal("threepcf bridge: nuisance.IA = IA_REDSHIFT_BINNING is "
              "refused: ia[1][2] holds the bin-2 A2 amplitude there, "
              "not the velocity-shear C_t");
    exit(1);
  }
  if (IA_REDSHIFT_EVOLUTION != nuisance.IA
      || IA_MODEL_TATT != nuisance.IA_MODEL) {
    log_fatal("threepcf bridge: the IA translation requires "
              "nuisance.IA = IA_REDSHIFT_EVOLUTION and "
              "nuisance.IA_MODEL = IA_MODEL_TATT (got IA = %d, "
              "IA_MODEL = %d)", nuisance.IA, nuisance.IA_MODEL);
    exit(1);
  }
  if (!(nuisance.oneplusz0_ia > 0.0)) {
    log_fatal("threepcf bridge: nuisance.oneplusz0_ia = %e; install "
              "the IA amplitudes (set_nuisance_IA) before the "
              "translation", nuisance.oneplusz0_ia);
    exit(1);
  }
  if (!(nuisance.c1rhocrit_ia > 0.0)) {
    log_fatal("threepcf bridge: nuisance.c1rhocrit_ia = %e must be "
              "positive", nuisance.c1rhocrit_ia);
    exit(1);
  }
  // One scalar b_TA serves the whole model. A leftover per-bin table
  // (an earlier IA_REDSHIFT_BINNING setup in the same process) whose
  // entries differ cannot be represented; refuse instead of silently
  // collapsing it to bin 0. Under IA_REDSHIFT_EVOLUTION the setter
  // writes slot 0 only, so untouched slots hold their reset zeros.
  for (int i=1; i<redshift.shear_nbin; i++) {
    if (0.0 != nuisance.ia[2][i]
        && nuisance.ia[2][i] != nuisance.ia[2][0]) {
      log_fatal("threepcf bridge: per-bin b_TA values differ "
                "(ia[2][%d] = %e vs ia[2][0] = %e); one scalar b_TA "
                "cannot represent them", i, nuisance.ia[2][i],
                nuisance.ia[2][0]);
      exit(1);
    }
  }

  physics_default_ia_3pcf(parameters);
  parameters->zpivot = nuisance.oneplusz0_ia - 1.0;
  parameters->eta1 = nuisance.ia[0][1];
  parameters->eta2 = nuisance.ia[1][1];
  // the documented factor; its base is exactly one for the pivot
  // chosen above, but the general form keeps this site the line-by-
  // line inverse of core_ia_parameters in des3pcf.core_two_point
  const double ratio = parameters->c1rho/nuisance.c1rhocrit_ia;
  const double base = nuisance.oneplusz0_ia/(1.0 + parameters->zpivot);
  parameters->a1 = nuisance.ia[0][0]/(ratio*pow(base, parameters->eta1));
  parameters->a2 = nuisance.ia[1][0]/(ratio*pow(base, parameters->eta2));
  parameters->bta = nuisance.ia[2][0];
  parameters->ct = nuisance.ia[1][2];
  parameters->power_mode = supplied.power_mode;
  parameters->ct_mode = supplied.ct_mode;
  parameters->suppress_only = supplied.suppress_only;
  parameters->feedback_amplitude = supplied.feedback.amplitude;
}

// ---------------------------------------------------------------------------
// Cached-input factory: scratch, guard snapshot and assembly.
// ---------------------------------------------------------------------------

// Scratch buffers of the last assembly. physics_create_3pcf copies
// every table, so these only need to outlive the creation call; they
// are kept until the next assembly (or reset) so the returned input
// stays valid while the caller builds the model.
static struct {
  double* k;        // wave numbers [nk], h/Mpc
  double* plin0;    // linear power at z = 0 [nk], (Mpc/h)^3
  double* z;        // background redshifts [nz]
  double* pnl;      // nonlinear power [nz*nk], row-major, (Mpc/h)^3
  double* chi;      // comoving distance [nz], Mpc/h
  double* dzdchi;   // dz/dchi [nz], h/Mpc
  double* growth;   // growth factor D [nz], D(0) = 1
  double* sources;  // struct-read source repack [nbin*nz], or NULL
  physics_input_3pcf input;     // the assembled model input
  feedback_input_3pcf feedback; // pointed to by input.feedback
} scratch;

static void scratch_clear(void)
{
  free(scratch.k);
  free(scratch.plin0);
  free(scratch.z);
  free(scratch.pnl);
  free(scratch.chi);
  free(scratch.dzdchi);
  free(scratch.growth);
  free(scratch.sources);
  memset(&scratch, 0, sizeof(scratch));
}

// Guard snapshot of the last committed assembly. The allocation tier
// stores every count as an int; the refill tier stores the cache keys
// of the structs the assembly read plus the bridge nonce. A build
// exists only while have_build is one.
#define BRIDGE_NCOUNT 20
#define BRIDGE_NKEY 6
static struct {
  int have_build;
  int counts[BRIDGE_NCOUNT];
  uint64_t keys[BRIDGE_NKEY];
  uint64_t generation;
} guard;

// Gather the current counts (allocation tier) and keys (refill tier).
static void guard_current(
  int counts[BRIDGE_NCOUNT], // out: allocation-tier counts
  uint64_t keys[BRIDGE_NKEY] // out: refill-tier cache keys and the nonce
)
{
  counts[0] = cosmology.lnP_nk;
  counts[1] = cosmology.lnP_nz;
  counts[2] = cosmology.lnPL_nk;
  counts[3] = cosmology.lnPL_nz;
  counts[4] = cosmology.chi_nz;
  counts[5] = cosmology.G_nz;
  counts[6] = redshift.shear_nbin;
  counts[7] = redshift.shear_nzbins;
  counts[8] = supplied.nlos;
  counts[9] = supplied.nbin;
  counts[10] = supplied.nsource;
  counts[11] = supplied.kernel_nz;
  counts[12] = supplied.source_readers;
  counts[13] = supplied.table_boost;
  counts[14] = supplied.fit_nz;
  counts[15] = supplied.nwindow;
  counts[16] = supplied.feedback.nz;
  counts[17] = supplied.feedback.nk;
  counts[18] = supplied.feedback.nu;
  counts[19] = supplied.feedback.nv;
  keys[0] = cosmology.random;
  keys[1] = redshift.random_shear;
  keys[2] = nuisance.random_photoz_shear;
  keys[3] = nuisance.random_ia;
  // Ntable.random rides along for the accuracy-boost knobs that bump
  // it in full-core builds; the bridge's own resolution knobs travel
  // through the nonce instead.
  keys[4] = Ntable.random;
  keys[5] = supplied.nonce;
}

// dz/dchi at a node of the chi table, from the same finite-difference
// derivative chi_all (cosmo3D.c) interpolates: inside bracket
// [j, j+1], dchi/dz = down + dy (up - down), with up the centered
// slope over (j, j+2), down the centered slope over (j-1, j+1) (one-
// sided at j = 0) and dy the bracket fraction. Evaluated exactly at a
// node, dy vanishes for every interior node, so the derivative is the
// plain centered difference there; the last two nodes inherit
// chi_all's clamped bracket (dy = 1 and dy ~ 2). hoverh0 is the
// reciprocal of this same derivative, which is why the model's
// dz/dchi row is defined as its inverse.
static double bridge_dzdchi_node(
  const double* z, // [nz] redshift nodes of the chi table
  const double* c, // [nz] comoving distance chi at those nodes
  const int nz, // node count of the chi table
  const int i // node at which dz/dchi is evaluated
)
{
  int j = (i < nz - 2) ? i : nz - 2;    // piecewise_index clamp
  const int jc = (j > nz - 3) ? nz - 3 : j; // chi_all's j+2 clamp
  const double dy = (z[i] - z[jc])/(z[jc+1] - z[jc]);
  const double up = (c[jc+2] - c[jc])/(z[jc+2] - z[jc]);
  const double down = (jc > 0)
      ? (c[jc+1] - c[jc-1])/(z[jc+1] - z[jc-1])
      : (c[jc+1] - c[jc  ])/(z[jc+1] - z[jc  ]);
  return 1.0/(down + dy*(up - down));
}

uint64_t threepcf_bridge_generation(void)
{
  return guard.generation;
}

const physics_input_3pcf* threepcf_bridge_refresh(void)
{
  int counts[BRIDGE_NCOUNT];
  uint64_t keys[BRIDGE_NKEY];
  guard_current(counts, keys);

  if (guard.have_build) {
    int stale = 0;
    for (int i=0; i<BRIDGE_NCOUNT; i++) {
      if (guard.counts[i] != counts[i]) {
        stale = 1;
      }
    }
    for (int i=0; i<BRIDGE_NKEY; i++) {
      if (fdiff2(guard.keys[i], keys[i])) {
        stale = 1;
      }
    }
    if (!stale) {
      return NULL; // the caller's current model is still valid
    }
  }

  // --- refusals: everything the assembly reads must be installed ---
  if (NULL == cosmology.lnP
      || NULL == cosmology.lnPL) {
    log_fatal("threepcf bridge: install the nonlinear and linear power "
              "spectra before building the three-point model");
    exit(1);
  }
  if (NULL == cosmology.chi
      || NULL == cosmology.G) {
    log_fatal("threepcf bridge: install the distances and the growth "
              "table before building the three-point model");
    exit(1);
  }
  if (!(cosmology.Omega_m > 0.0)) {
    log_fatal("threepcf bridge: cosmology.Omega_m = %e; install the "
              "cosmological parameters first", cosmology.Omega_m);
    exit(1);
  }
  if (!supplied.have_scalars) {
    log_fatal("threepcf bridge: set the supplied scalars (ns, sigma8, "
              "w0, wa) before building the three-point model");
    exit(1);
  }
  if (0 == supplied.nlos) {
    log_fatal("threepcf bridge: set the line-of-sight nodes before "
              "building the three-point model");
    exit(1);
  }

  const int nk = cosmology.lnP_nk;
  const int nz = cosmology.lnP_nz;

  // The model's background grid is the nonlinear table's z axis, so
  // every background quantity must live on that same grid. Node
  // equality is exact: in a real likelihood all these axes descend
  // from the same arrays, and a bitwise gate cannot absorb a rounded
  // tolerance.
  const double* zrow = cosmology.lnP[nk]; // z axis of the lnP table
  if (cosmology.chi_nz != nz) {
    log_fatal("threepcf bridge: chi table has %d nodes but the "
              "nonlinear power has %d redshift nodes; install both on "
              "one grid", cosmology.chi_nz, nz);
    exit(1);
  }
  if (cosmology.G_nz != nz) {
    log_fatal("threepcf bridge: growth table has %d nodes but the "
              "nonlinear power has %d redshift nodes; install both on "
              "one grid", cosmology.G_nz, nz);
    exit(1);
  }
  for (int j=0; j<nz; j++) {
    if (cosmology.chi[0][j] != zrow[j]
        || cosmology.G[0][j] != zrow[j]) {
      log_fatal("threepcf bridge: background z grids disagree at node "
                "%d (lnP %e, chi %e, G %e)", j, zrow[j],
                cosmology.chi[0][j], cosmology.G[0][j]);
      exit(1);
    }
  }
  if (nz < 5) {
    log_fatal("threepcf bridge: the dz/dchi stencil needs at least "
              "five redshift nodes, got %d", nz);
    exit(1);
  }

  // The linear table shares the wave-number axis and contributes its
  // z = 0 row only (the model evolves the linear power with D^2
  // itself).
  if (cosmology.lnPL_nk != nk) {
    log_fatal("threepcf bridge: linear and nonlinear power tables "
              "have different wave-number counts (%d vs %d)",
              cosmology.lnPL_nk, nk);
    exit(1);
  }
  for (int i=0; i<nk; i++) {
    if (cosmology.lnPL[i][cosmology.lnPL_nz] !=
        cosmology.lnP[i][nz]) {
      log_fatal("threepcf bridge: linear and nonlinear log10 k axes "
                "disagree at node %d", i);
      exit(1);
    }
  }
  int j0 = -1;
  for (int j=0; j<cosmology.lnPL_nz; j++) {
    if (0.0 == cosmology.lnPL[cosmology.lnPL_nk][j]) {
      j0 = j;
      break;
    }
  }
  if (j0 < 0) {
    log_fatal("threepcf bridge: the linear power table has no z = 0 "
              "row; the model needs the z = 0 linear spectrum");
    exit(1);
  }

  // Source tier selection: a supplied override wins; otherwise the
  // struct table must live on the background grid with every photo-z
  // shift exactly zero (threepcf_core_bridge.h explains why shifted
  // sources are supplied, never recomputed here).
  if (0 == supplied.nbin) {
    if (NULL == redshift.shear_zdist_table
        || redshift.shear_nbin < 1) {
      log_fatal("threepcf bridge: no supplied sources and no source "
                "sample installed (set_source_sample)");
      exit(1);
    }
    if (redshift.shear_nzbins != nz) {
      log_fatal("threepcf bridge: the source table has %d redshift "
                "rows but the background grid has %d; supply shifted "
                "sources through the bridge setter instead",
                redshift.shear_nzbins, nz);
      exit(1);
    }
    const double* zv = redshift.shear_zdist_table[redshift.shear_nbin];
    for (int j=0; j<nz; j++) {
      if (zv[j] != zrow[j]) {
        log_fatal("threepcf bridge: source and background z grids "
                  "disagree at node %d (%e vs %e)", j, zv[j], zrow[j]);
        exit(1);
      }
    }
    for (int i=0; i<redshift.shear_nbin; i++) {
      if (0.0 != nuisance.photoz[0][0][i]) {
        log_fatal("threepcf bridge: source photo-z shift of bin %d is "
                  "%e; the bridge never replicates the Python shift "
                  "convention - apply the shifts in Python and supply "
                  "the result through the source setter", i,
                  nuisance.photoz[0][0][i]);
        exit(1);
      }
    }
  }

  // --- assembly: one fresh scratch per rebuild ---
  scratch_clear();
  scratch.k = (double*) malloc1d(nk);
  scratch.plin0 = (double*) malloc1d(nk);
  scratch.z = (double*) malloc1d(nz);
  scratch.pnl = (double*) malloc1d(nz*nk);
  scratch.chi = (double*) malloc1d(nz);
  scratch.dzdchi = (double*) malloc1d(nz);
  scratch.growth = (double*) malloc1d(nz);

  // Wave numbers and power leave their logarithms here, and nowhere
  // else: the structs store log10 k (k in h/Mpc) and ln P (P in
  // (Mpc/h)^3), which are already the model's physical units, so
  // pow/exp is the entire unit boundary. No c/H0 factor applies -
  // only the FAST-PT tables of the full core store c/H0 units.
  for (int i=0; i<nk; i++) {
    scratch.k[i] = pow(10.0, cosmology.lnP[i][nz]);
    scratch.plin0[i] = exp(cosmology.lnPL[i][j0]);
  }
  for (int j=0; j<nz; j++) {
    scratch.z[j] = zrow[j];
    for (int i=0; i<nk; i++) {
      scratch.pnl[(size_t) j*nk + i] = exp(cosmology.lnP[i][j]);
    }
  }

  // Distances: the chi table already stores Mpc/h (set_distances),
  // the model's unit, so the nodes copy through; dz/dchi inverts the
  // chi_all derivative at each node (bridge_dzdchi_node).
  for (int j=0; j<nz; j++) {
    scratch.chi[j] = cosmology.chi[1][j];
    scratch.dzdchi[j] = bridge_dzdchi_node(cosmology.chi[0],
                                           cosmology.chi[1], nz, j);
  }

  // Growth: the G table stores G = (1+z) D (set_growth's D = G a
  // convention), so D = G a with a = 1/(1+z), normalized by the z = 0
  // node so D(0) = 1 exactly, as the model contract requires. The
  // grid-equality guard above makes the z = 0 node the first one.
  if (!(0.0 != cosmology.G[1][0])) {
    log_fatal("threepcf bridge: growth table has G(z=0) = 0");
    exit(1);
  }
  for (int j=0; j<nz; j++) {
    const double a = 1.0/(1.0 + zrow[j]);
    scratch.growth[j] = (cosmology.G[1][j]*a)/cosmology.G[1][0];
  }

  memset(&scratch.input, 0, sizeof(scratch.input));
  scratch.input.nk = nk;
  scratch.input.nz = nz;
  scratch.input.k = scratch.k;
  scratch.input.p_linear0 = scratch.plin0;
  scratch.input.z = scratch.z;
  scratch.input.p_nonlinear = scratch.pnl;
  scratch.input.chi = scratch.chi;
  scratch.input.dz_dchi = scratch.dzdchi;
  scratch.input.growth = scratch.growth;

  if (supplied.nbin > 0) {
    scratch.input.nbin = supplied.nbin;
    scratch.input.sources = supplied.sources;
    scratch.input.nsource = supplied.nsource;
    scratch.input.source_z = supplied.source_z;
  }
  else {
    // Repack the padded malloc2d rows of the struct table into the
    // flat row-major block the model reads; nsource = 0 reuses the
    // background grid and keeps the model's exact preparation
    // arithmetic for sources on that grid (physics_3pcf.h).
    const int nbin = redshift.shear_nbin;
    scratch.sources = (double*) malloc1d(nbin*nz);
    for (int k=0; k<nbin; k++) {
      for (int j=0; j<nz; j++) {
        scratch.sources[(size_t) k*nz + j] =
            redshift.shear_zdist_table[k][j];
      }
    }
    scratch.input.nbin = nbin;
    scratch.input.sources = scratch.sources;
    scratch.input.nsource = 0;
    scratch.input.source_z = NULL;
  }

  scratch.input.nlos = supplied.nlos;
  scratch.input.los_z = supplied.los;
  scratch.input.kernel_nz = supplied.kernel_nz;
  scratch.input.source_readers = supplied.source_readers;
  scratch.input.table_boost = supplied.table_boost;
  scratch.input.fit_nz = supplied.fit_nz;

  scratch.input.omega_m = cosmology.Omega_m;
  // the core keeps flat background models (set_cosmological_parameters
  // writes Omega_v = 1 - Omega_m), and the model checks flatness
  scratch.input.omega_de = 1.0 - cosmology.Omega_m;
  scratch.input.w0 = supplied.w0;
  scratch.input.wa = supplied.wa;
  scratch.input.ns = supplied.ns;
  scratch.input.sigma8 = supplied.sigma8;

  if (supplied.nwindow > 0) {
    scratch.input.nwindow = supplied.nwindow;
    scratch.input.window = supplied.window;
  }
  if (supplied.feedback.nz > 0) {
    memset(&scratch.feedback, 0, sizeof(scratch.feedback));
    scratch.feedback.nz = supplied.feedback.nz;
    scratch.feedback.nk = supplied.feedback.nk;
    scratch.feedback.nu = supplied.feedback.nu;
    scratch.feedback.nv = supplied.feedback.nv;
    scratch.feedback.z = supplied.feedback.z;
    scratch.feedback.logk = supplied.feedback.logk;
    scratch.feedback.logu = supplied.feedback.logu;
    scratch.feedback.v = supplied.feedback.v;
    scratch.feedback.log_power = supplied.feedback.log_power;
    scratch.feedback.log_bispectrum = supplied.feedback.log_bispectrum;
    scratch.feedback.boost = supplied.feedback.boost;
    scratch.feedback.clamp = supplied.feedback.clamp;
    scratch.input.feedback = &scratch.feedback;
  }

  // Commit the snapshot before returning: the keys describe the
  // state this assembly read. physics_create_3pcf aborts the process
  // on any failure, so a committed snapshot can never describe a
  // model that silently failed to build.
  memcpy(guard.counts, counts, sizeof(counts));
  memcpy(guard.keys, keys, sizeof(keys));
  guard.have_build = 1;
  guard.generation++;

  return &scratch.input;
}

void reset_threepcf_core_bridge(void)
{
  supplied_clear();
  scratch_clear();
  guard.have_build = 0;
  memset(guard.counts, 0, sizeof(guard.counts));
  memset(guard.keys, 0, sizeof(guard.keys));
  // guard.generation stays monotonic on purpose: a consumer that
  // cached a generation number across the reset must still observe
  // the next build as new.
}

// ===========================================================================
// CORE STRUCT INSTALLERS (standalone-module equivalents)
// ===========================================================================
// Each installer mirrors its generic_interface.cpp original: the same
// units, the same validation, the same fdiff change detection on the
// stored values and the same cache-key bump on change (bridge_draw
// stands in for the RandomNumber singleton). Deliberate divergences,
// each tied to machinery this module does not compile:
//   - set_distances calls no set_chi_bucket_index (cosmo3D.c): the
//     bridge walks the stored z axis directly;
//   - set_growth leaves the growth-rate row f at zero (the five-point
//     stencil serves f_growth in cosmo3D.c; nothing here reads it);
//   - set_source_sample runs no nz_source_photoz warm-up and keeps no
//     spline-convention caches (redshift_spline.c);
//   - no OpenMP fill loops: these tables install once per model, and
//     the serial fill keeps this file free of parallel regions.
// In a full-core build the original installers run instead and these
// stay unused.

void threepcf_core_set_cosmological_parameters(
  const double omega_matter, // Omega_m today
  const double omega_baryon, // Omega_b today
  const double hubble, // H0 (km/s/Mpc)
  const double omega_nu_h2 // omega_nu h^2 of the massive neutrinos
)
{
  // the negated comparisons also reject NaN
  if (!(hubble > 0.0)) {
    log_fatal("threepcf bridge: H0 = %e km/s/Mpc must be > 0", hubble);
    exit(1);
  }
  if (!(omega_nu_h2 >= 0.0)) {
    log_fatal("threepcf bridge: omega_nu h^2 = %e must be >= 0",
              omega_nu_h2);
    exit(1);
  }
  const double h = hubble/100.0;
  const double omega_nu = omega_nu_h2/(h*h);
  if (!(omega_nu < omega_matter)) {
    log_fatal("threepcf bridge: Omega_nu = %e must be < Omega_m = %e "
              "(Omega_m includes the massive neutrinos)", omega_nu,
              omega_matter);
    exit(1);
  }
  int cache_update = 0;
  if (fdiff(cosmology.Omega_m, omega_matter) ||
      fdiff(cosmology.Omega_b, omega_baryon) ||
      fdiff(cosmology.Omega_nu, omega_nu) ||
      fdiff(cosmology.h0, hubble/100.0)) {
    cache_update = 1;
  }
  if (1 == cache_update) {
    cosmology.Omega_m = omega_matter;
    cosmology.Omega_v = 1.0 - omega_matter;
    cosmology.Omega_b = omega_baryon;
    cosmology.Omega_nu = omega_nu;
    cosmology.h0 = hubble/100.0;
    cosmology.MGSigma = 0.0;
    cosmology.MGmu = 0.0;
    cosmology.random = bridge_draw();
  }
}

void threepcf_core_set_distances(
  const double* z, // redshift grid [n], at least five nodes
  const double* chi, // comoving distance at z in Mpc/h [n]
  const int n // node count
)
{
  if (NULL == z
      || NULL == chi
      || n < 5) {
    log_fatal("threepcf bridge: set_distances needs matching z and chi "
              "arrays with at least five nodes");
    exit(1);
  }
  int cache_update = 0;
  if (cosmology.chi_nz != n || NULL == cosmology.chi) {
    cache_update = 1;
  }
  else {
    for (int i=0; i<n; i++) {
      if (fdiff(cosmology.chi[0][i], z[i]) ||
          fdiff(cosmology.chi[1][i], chi[i])) {
        cache_update = 1;
        break;
      }
    }
  }
  if (1 == cache_update) {
    cosmology.chi_nz = n;
    free(cosmology.chi);
    cosmology.chi = (double**) malloc2d(2, n);
    cosmology.chi_z_nseg = detect_uniform_segments(
        z, n, 1e-9, MAX_GRID_SEGMENTS,
        cosmology.chi_z_seg_start, cosmology.chi_z_seg_len,
        cosmology.chi_z_seg_xmin, cosmology.chi_z_seg_inv_dx,
        "chi z");
    for (int i=0; i<n; i++) {
      if (isnan(z[i])
          || isnan(chi[i])) {
        log_fatal("threepcf bridge: NaN in the distance table");
        exit(1);
      }
      cosmology.chi[0][i] = z[i];
      cosmology.chi[1][i] = chi[i];
    }
    cosmology.random = bridge_draw();
  }
}

void threepcf_core_set_growth(
  const double* z, // redshift grid [n], at least five nodes
  const double* G, // growth G at z, D = G * a [n]
  const int n // node count
)
{
  if (NULL == z
      || G == NULL
      || n < 5) {
    log_fatal("threepcf bridge: set_growth needs matching z and G "
              "arrays with at least five nodes");
    exit(1);
  }
  int cache_update = 0;
  if (cosmology.G_nz != n || NULL == cosmology.G) {
    cache_update = 1;
  }
  else {
    for (int i=0; i<n; i++) {
      if (fdiff(cosmology.G[0][i], z[i]) ||
          fdiff(cosmology.G[1][i], G[i])) {
        cache_update = 1;
        break;
      }
    }
  }
  if (1 == cache_update) {
    cosmology.G_nz = n;
    cosmology.G_z_nseg = detect_uniform_segments(
        z, n, 1e-9, MAX_GRID_SEGMENTS,
        cosmology.G_z_seg_start, cosmology.G_z_seg_len,
        cosmology.G_z_seg_xmin, cosmology.G_z_seg_inv_dx,
        "G z");
    free(cosmology.G);
    cosmology.G = (double**) malloc2d(3, n);
    for (int i=0; i<n; i++) {
      if (isnan(z[i])
          || isnan(G[i])) {
        log_fatal("threepcf bridge: NaN in the growth table");
        exit(1);
      }
      cosmology.G[0][i] = z[i];
      cosmology.G[1][i] = G[i];
      // row 2 (the growth rate f from set_growth's five-point
      // stencil) stays zero: its readers live in cosmo3D.c, which
      // this module does not compile, and the bridge reads rows 0-1
      cosmology.G[2][i] = 0.0;
    }
    cosmology.random = bridge_draw();
  }
}

// Shared body of the two power-spectrum installers: the linear and
// nonlinear tables use one layout (structs.h), so one routine fills
// either destination. is_linear selects the struct fields and the
// linear-only duty of dropping the P_cb companion table.
static void bridge_install_power(
  const double* log10k, // log10 k grid [nk], k in h/Mpc
  const int nk, // wave-number count
  const double* z, // redshift grid [nz]
  const int nz, // redshift count
  const double* lnP, // flattened ln P [nk*nz]
  const int is_linear // 1 fills the linear table; 0 the nonlinear one
)
{
  if (NULL == log10k
      || NULL == z
      || NULL == lnP
      || nk < 2
      || nz < 2) {
    log_fatal("threepcf bridge: a power-spectrum table needs log10k, "
              "z and lnP with at least two nodes per axis");
    exit(1);
  }
  double** stored = is_linear ? cosmology.lnPL : cosmology.lnP;
  const int stored_nk = is_linear ? cosmology.lnPL_nk : cosmology.lnP_nk;
  const int stored_nz = is_linear ? cosmology.lnPL_nz : cosmology.lnP_nz;

  int cache_update = 0;
  if (stored_nk != nk || stored_nz != nz || NULL == stored) {
    cache_update = 1;
  }
  else {
    for (int i=0; i<nk && !cache_update; i++) {
      for (int j=0; j<nz; j++) {
        if (fdiff(stored[i][j], lnP[(size_t) i*nz + j])) {
          cache_update = 1;
          break;
        }
      }
    }
    for (int i=0; i<nk && !cache_update; i++) {
      if (fdiff(stored[i][nz], log10k[i])) {
        cache_update = 1;
      }
    }
    for (int j=0; j<nz && !cache_update; j++) {
      if (fdiff(stored[nk][j], z[j])) {
        cache_update = 1;
      }
    }
  }
  if (0 == cache_update) {
    return;
  }

  // the log10 k axis must be one uniform segment (the full core's
  // direct-index readers require it; keeping the requirement here
  // means a table accepted by this module is accepted everywhere)
  {
    int s_start[MAX_GRID_SEGMENTS];
    int s_len[MAX_GRID_SEGMENTS];
    double s_xmin[MAX_GRID_SEGMENTS];
    double s_inv_dx[MAX_GRID_SEGMENTS];
    const int nseg = detect_uniform_segments(
        log10k, nk, 1e-9, MAX_GRID_SEGMENTS,
        s_start, s_len, s_xmin, s_inv_dx,
        is_linear ? "lnPL log10k" : "lnP log10k");
    if (1 != nseg) {
      log_fatal("threepcf bridge: the log10 k axis must be one uniform "
                "segment, got %d", nseg);
      exit(1);
    }
    if (is_linear) {
      cosmology.lnPL_log10k_min = s_xmin[0];
      cosmology.lnPL_log10k_inv_dx = s_inv_dx[0];
    }
    else {
      cosmology.lnP_log10k_min = s_xmin[0];
      cosmology.lnP_log10k_inv_dx = s_inv_dx[0];
    }
  }
  if (is_linear) {
    cosmology.lnPL_z_nseg = detect_uniform_segments(
        z, nz, 1e-9, MAX_GRID_SEGMENTS,
        cosmology.lnPL_z_seg_start, cosmology.lnPL_z_seg_len,
        cosmology.lnPL_z_seg_xmin, cosmology.lnPL_z_seg_inv_dx,
        "lnPL z");
  }
  else {
    cosmology.lnP_z_nseg = detect_uniform_segments(
        z, nz, 1e-9, MAX_GRID_SEGMENTS,
        cosmology.lnP_z_seg_start, cosmology.lnP_z_seg_len,
        cosmology.lnP_z_seg_xmin, cosmology.lnP_z_seg_inv_dx,
        "lnP z");
  }

  free(stored);
  double** table = (double**) malloc2d(nk + 1, nz + 1);
  for (int i=0; i<nk; i++) {
    if (isnan(log10k[i])) {
      log_fatal("threepcf bridge: NaN in the log10 k axis");
      exit(1);
    }
    table[i][nz] = log10k[i];
  }
  for (int j=0; j<nz; j++) {
    if (isnan(z[j])) {
      log_fatal("threepcf bridge: NaN in the power-spectrum z axis");
      exit(1);
    }
    table[nk][j] = z[j];
  }
  for (int i=0; i<nk; i++) {
    for (int j=0; j<nz; j++) {
      if (isnan(lnP[(size_t) i*nz + j])) {
        log_fatal("threepcf bridge: NaN in the ln P table");
        exit(1);
      }
      table[i][j] = lnP[(size_t) i*nz + j];
    }
  }
  if (is_linear) {
    cosmology.lnPL = table;
    cosmology.lnPL_nk = nk;
    cosmology.lnPL_nz = nz;
    // the P_cb companion of the replaced spectrum goes with it
    // (structs.h: the two are installed as a pair)
    free(cosmology.lnPL_cb);
    cosmology.lnPL_cb = NULL;
  }
  else {
    cosmology.lnP = table;
    cosmology.lnP_nk = nk;
    cosmology.lnP_nz = nz;
  }
  cosmology.random = bridge_draw();
}

void threepcf_core_set_linear_power_spectrum(
  const double* log10k, // log10 k grid [nk], k in h/Mpc, uniform
  const int nk, // wave-number count
  const double* z, // redshift grid [nz]
  const int nz, // redshift count
  const double* lnP // flattened ln P [nk*nz]
)
{
  bridge_install_power(log10k, nk, z, nz, lnP, 1);
}

void threepcf_core_set_non_linear_power_spectrum(
  const double* log10k, // log10 k grid [nk], k in h/Mpc, uniform
  const int nk, // wave-number count
  const double* z, // redshift grid [nz]
  const int nz, // redshift count
  const double* lnP // flattened ln P [nk*nz]
)
{
  bridge_install_power(log10k, nk, z, nz, lnP, 0);
}

void threepcf_core_set_source_sample_size(
  const int ntomo // number of source tomographic bins
)
{
  if (!(ntomo > 0)
      || ntomo > MAX_SIZE_ARRAYS) {
    log_fatal("threepcf bridge: Ntomo = %d outside 1..%d", ntomo,
              MAX_SIZE_ARRAYS);
    exit(1);
  }
  // photo-z mode flag; no code in cosmolike_core reads it
  redshift.shear_photoz = 4;
  redshift.shear_nbin = ntomo;
}

// The bin count the stored table was saved with; a table saved with
// another count is reloaded without an entry-by-entry comparison, as
// in generic_interface.cpp.
static int bridge_stored_shear_nbin = 0;

void threepcf_core_set_source_sample(
  const double* table, // row-major n(z) table [nzbins][ntomo+1]
  const int nzbins, // redshift rows
  const int ntomo // source bins; must equal redshift.shear_nbin
)
{
  const int Ntomo = redshift.shear_nbin;
  if (!(Ntomo > 0)
      || Ntomo > MAX_SIZE_ARRAYS) {
    log_fatal("threepcf bridge: declare the source tomography "
              "(set_source_sample_size) before installing n(z)");
    exit(1);
  }
  if (NULL == table
      || nzbins < 2
      || ntomo != Ntomo) {
    log_fatal("threepcf bridge: the n(z) table needs at least two "
              "redshift rows and ntomo = %d columns beside the z "
              "column (got ntomo = %d)", Ntomo, ntomo);
    exit(1);
  }
  const int ncol = Ntomo + 1; // z column plus one n(z) column per bin

  int cache_update = 0;
  if (redshift.shear_nzbins != nzbins ||
      bridge_stored_shear_nbin != Ntomo ||
      NULL == redshift.shear_zdist_table) {
    cache_update = 1;
  }
  else {
    double** tab = redshift.shear_zdist_table;
    double* z_v = redshift.shear_zdist_table[Ntomo];
    for (int i=0; i<nzbins && !cache_update; i++) {
      if (fdiff(z_v[i], table[(size_t) i*ncol])) {
        cache_update = 1;
        break;
      }
      for (int k=0; k<Ntomo; k++) {
        if (fdiff(tab[k][i], table[(size_t) i*ncol + k + 1])) {
          cache_update = 1;
          break;
        }
      }
    }
  }
  if (0 == cache_update) {
    return;
  }

  bridge_require_finite(table, nzbins*ncol, "source n(z) table");
  redshift.shear_nzbins = nzbins;
  bridge_stored_shear_nbin = Ntomo;
  free(redshift.shear_zdist_table);
  redshift.shear_zdist_table = (double**) malloc2d(Ntomo + 1, nzbins);
  double** tab = redshift.shear_zdist_table;
  double* z_v = redshift.shear_zdist_table[Ntomo];
  for (int i=0; i<nzbins; i++) {
    z_v[i] = table[(size_t) i*ncol];
    for (int k=0; k<Ntomo; k++) {
      tab[k][i] = table[(size_t) i*ncol + k + 1];
    }
  }

  redshift.shear_zdist_zall[RANGE_MIN] = fmax(z_v[0], 1.e-5);
  redshift.shear_zdist_zall[RANGE_MAX] =
      z_v[nzbins-1] + (z_v[nzbins-1] - z_v[0])/((double) nzbins - 1.);

  for (int k=0; k<Ntomo; k++) { // set tomography bin boundaries
    // the bin support is where n(z) exceeds 0.999e-8 of its maximum,
    // exactly as set_source_sample bounds it
    double nzmax = tab[k][0];
    for (int i=1; i<nzbins; i++) {
      nzmax = fmax(nzmax, tab[k][i]);
    }
    int first = -1;
    int last = -1;
    for (int i=0; i<nzbins; i++) {
      if (tab[k][i] > 0.999e-8*nzmax) {
        if (first < 0) {
          first = i;
        }
        last = i;
      }
    }
    if (first < 0) {
      log_fatal("threepcf bridge: n(z) of bin %d has no positive "
                "entry", k);
      exit(1);
    }
    redshift.shear_zdist_z[RANGE_MIN][k] = fmax(z_v[first], 1.001e-5);
    redshift.shear_zdist_z[RANGE_MAX][k] = z_v[last];
  }
  if (redshift.shear_zdist_zall[RANGE_MAX] <
          redshift.shear_zdist_z[RANGE_MAX][Ntomo-1]
      || redshift.shear_zdist_zall[RANGE_MIN] >
          redshift.shear_zdist_z[RANGE_MIN][0]) {
    log_fatal("threepcf bridge: n(z) bin support exceeds the table "
              "range (zall = [%e, %e])",
              redshift.shear_zdist_zall[RANGE_MIN],
              redshift.shear_zdist_zall[RANGE_MAX]);
    exit(1);
  }
  // no nz_source_photoz warm-up: the spline machinery lives in
  // redshift_spline.c, which this module does not compile
  redshift.random_shear = bridge_draw();
}

void threepcf_core_set_nuisance_shear_photoz(
  const double* shifts, // per-bin shifts [n]
  const int n // must equal redshift.shear_nbin
)
{
  if (0 == redshift.shear_nbin) {
    log_fatal("threepcf bridge: declare the source tomography before "
              "the photo-z shifts");
    exit(1);
  }
  if (NULL == shifts
      || redshift.shear_nbin != n) {
    log_fatal("threepcf bridge: expected %d photo-z shifts, got %d",
              redshift.shear_nbin, n);
    exit(1);
  }
  int cache_update = 0;
  for (int i=0; i<redshift.shear_nbin; i++) {
    if (isnan(shifts[i])) {
      log_fatal("threepcf bridge: photo-z shift %d is NaN", i);
      exit(1);
    }
    if (fdiff(nuisance.photoz[0][0][i], shifts[i])) {
      cache_update = 1;
      nuisance.photoz[0][0][i] = shifts[i];
    }
  }
  if (1 == cache_update) {
    nuisance.random_photoz_shear = bridge_draw();
  }
}

void threepcf_core_init_ia(
  const int ia_model, // 0 = IA_MODEL_NLA, 1 = IA_MODEL_TATT
  const int ia_redshift_evolution // NO_IA .. IA_REDSHIFT_EVOLUTION (IA.h)
)
{
  // every IA consumer keys on nuisance.random_ia: redraw it when the
  // model registry changes, as init_IA_fastpt does
  if (nuisance.IA_MODEL != ia_model ||
      nuisance.IA != ia_redshift_evolution) {
    nuisance.random_ia = bridge_draw();
  }
  if (0 == ia_model
      || 1 == ia_model) {
    nuisance.IA_MODEL = ia_model;
  }
  else {
    log_fatal("threepcf bridge: invalid IA_MODEL = %d", ia_model);
    exit(1);
  }
  if (NO_IA == ia_redshift_evolution
      || IA_NLA_LF == ia_redshift_evolution
      || IA_REDSHIFT_BINNING == ia_redshift_evolution
      || IA_REDSHIFT_EVOLUTION == ia_redshift_evolution) {
    nuisance.IA = ia_redshift_evolution;
  }
  else {
    log_fatal("threepcf bridge: invalid IA mode = %d",
              ia_redshift_evolution);
    exit(1);
  }
  nuisance.IA_code = 0; // CFASTPT; no python FAST-PT path here
}

void threepcf_core_set_nuisance_ia(
  const double* A1, // tidal-alignment amplitudes [n]
  const double* A2, // tidal-torque amplitudes [n]
  const double* BTA, // density-weighting amplitudes [n]
  const int n // entries supplied; at least shear_nbin
)
{
  if (0 == redshift.shear_nbin) {
    log_fatal("threepcf bridge: declare the source tomography before "
              "the IA amplitudes");
    exit(1);
  }
  if (A1 == NULL
      || A2 == NULL
      || BTA == NULL
      || n < redshift.shear_nbin) {
    log_fatal("threepcf bridge: the IA amplitudes need at least "
              "shear_nbin = %d entries, got %d", redshift.shear_nbin,
              n);
    exit(1);
  }
  int cache_update = 0;
  nuisance.c1rhocrit_ia = 0.01389;
  if (IA_REDSHIFT_BINNING == nuisance.IA) {
    for (int i=0; i<redshift.shear_nbin; i++) {
      if (isnan(A1[i])
          || isnan(A2[i])
          || isnan(BTA[i])) {
        log_fatal("threepcf bridge: IA amplitude %d is NaN", i);
        exit(1);
      }
      if (fdiff(nuisance.ia[0][i], A1[i]) ||
          fdiff(nuisance.ia[1][i], A2[i]) ||
          fdiff(nuisance.ia[2][i], BTA[i])) {
        nuisance.ia[0][i] = A1[i];
        nuisance.ia[1][i] = A2[i];
        nuisance.ia[2][i] = BTA[i];
        cache_update = 1;
      }
    }
  }
  else if (IA_REDSHIFT_EVOLUTION == nuisance.IA) {
    // the power law reads A1[0], A1[1], A2[0], A2[1] and BTA[0]
    if (n < 2) {
      log_fatal("threepcf bridge: the IA power law needs two entries "
                "of A1 and A2 (amplitude, redshift exponent)");
      exit(1);
    }
    if (isnan(A1[0])
        || isnan(A1[1])
        || isnan(A2[0])
        || isnan(A2[1])
        || isnan(BTA[0])) {
      log_fatal("threepcf bridge: an IA power-law entry is NaN");
      exit(1);
    }
    nuisance.oneplusz0_ia = 1.62;
    if (fdiff(nuisance.ia[0][0], A1[0]) ||
        fdiff(nuisance.ia[0][1], A1[1]) ||
        fdiff(nuisance.ia[1][0], A2[0]) ||
        fdiff(nuisance.ia[1][1], A2[1]) ||
        fdiff(nuisance.ia[2][0], BTA[0])) {
      nuisance.ia[0][0] = A1[0];
      nuisance.ia[0][1] = A1[1];
      nuisance.ia[1][0] = A2[0];
      nuisance.ia[1][1] = A2[1];
      nuisance.ia[2][0] = BTA[0];
      cache_update = 1;
    }
  }
  if (1 == cache_update) {
    nuisance.random_ia = bridge_draw();
  }
}

void threepcf_core_set_nuisance_ia_ct(
  const double ct // velocity-shear alignment amplitude
)
{
  if (isnan(ct)) {
    log_fatal("threepcf bridge: C_t is NaN");
    exit(1);
  }
  int cache_update = 0;
  if (fdiff(nuisance.ia[1][2], ct)) {
    nuisance.ia[1][2] = ct;
    cache_update = 1;
  }
  if (1 == cache_update) {
    nuisance.random_ia = bridge_draw();
  }
}

void threepcf_core_initial_setup(void)
{
  // the reset_*_struct family lives in structs.c, which this module
  // compiles; state the full initial_setup resets beyond the structs
  // (baryon scenarios, Pdelta dispatch, FAST-PT origins, point
  // masses, CMB products) does not exist here
  reset_redshift_struct();
  reset_nuisance_struct();
  reset_cosmology_struct();
  reset_tomo_struct();
  reset_Ntable_struct();
  reset_like_struct();
  reset_cmb_struct();
  bridge_stored_shear_nbin = 0;
  reset_threepcf_core_bridge();
}
