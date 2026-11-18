#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "simde/x86/sse2.h"
#include "physics_grid_3pcf.h"
#ifdef _OPENMP
#include <omp.h>
#else
static int omp_in_parallel(void) { return 0; }
static int omp_get_max_threads(void) { return 1; }
static int omp_get_thread_num(void) { return 0; }
#endif

// ===========================================================================
// Prepared projected-bispectrum and Legendre tables
// ===========================================================================
// Direct line-of-sight evaluation is useful as a reference, but it
// repeats the same physical fit at every point of the transform grid.
// fastnc removes that cost in two stages. First it samples the projected
// bispectrum on a small triangle table and interpolates that table. Then
// it integrates the table's angular dependence into the Legendre
// multipoles b_L(ELL,PSI) and interpolates those. This file implements
// both stages in C from the MIT-licensed fastnc tatt sources (revision
// 7b2336ce..., bispectrum.py and multipole.py); the fastnc method paper
// (Sugiyama, Gomes and Jarvis, arXiv:2407.01798) introduces the method,
// and the TATT three-point analysis (Gomes et al., arXiv:2601.09133)
// defines the intrinsic-alignment model whose tables this file prepares.
//
// Three model names recur below. NLA is the nonlinear-alignment
// intrinsic-alignment model, and TATT is the tidal-alignment-tidal-torque
// model that extends it. GGG is the pure-gravity three-shear correlation,
// with no intrinsic alignment at all.
//
// The baseline table stores log B on sorted triangles in the RUV
// coordinates: r is the middle side, u = smallest/middle, and
// v = (largest-middle)/smallest. The baseline's radial weight is a
// product of three NLA kernels, so sorting the sides loses nothing: this
// baseline is invariant under side exchange. With signed_baseline=1 the
// storage splits in two: a positive gravity-only GGG table supplies the
// logarithmic dynamic range, and a separate NLA/GGG ratio carries the
// negative values and zero crossings. The split changes only the
// numerical representation, never the product of physical NLA kernels.
//
// The signed TATT residual keeps its source-bin association, so its
// table uses the PUV coordinates without side sorting: p is the sum of
// the sides, u = ell1/p and v = ell2/p. With residual_scaling=1 the
// preparation divides the signed value by a positive gravity-only
// equilateral B at the same perimeter before interpolating; splitting
// the amplitude this way limits the radial variation each table must
// track, and it changes neither the PUV shape grid nor its continuation.
// The rectangular PUV fill extends the integrand outside the physical
// triangle domain through the source's rectangular continuation
// (puv_extension 1); physical queries stay inside the triangle.
//
// The multipole stage samples polar coordinates of the two Fourier
// legs: ELL = sqrt(ell1^2+ell2^2) is the two-leg magnitude, and PSI =
// atan2(ell2,ell1) is the opening angle arctan(ell2/ell1) in [0, pi/2]
// - an angle that encodes the ratio of the two multipole legs, not
// itself a ratio. b_L are the Legendre coefficients of the projected
// E-mode bispectrum in mu, the cosine of the angle between the two
// legs; the mu axis stops at the cap mu <= 1 - epmu, which keeps the
// closing third side away from its degenerate limit. The source fastnc
// folds PSI to <= pi/4, and folding assumes symmetry under exchanging
// the legs; the TATT residual of unequal source bins has no such
// symmetry, so this file stores two independent charts, PSI =
// small_angle and PSI = pi/2 - small_angle. Both charts use the same
// physical small-angle range and preserve the ordered legs, and no
// transpose or implicit bin swap creates the second residual chart.
//
// Production grids with psi_scale > 0 sample PSI uniformly in the mixed
// coordinate x = log(PSI) + PSI/psi_scale: at small PSI the logarithm
// dominates and gives logarithmic resolution, while at large PSI the
// linear term gives nearly uniform angular resolution. psi_scale = 0
// keeps the original uniform log-PSI coordinate. Only reference_mode=1
// retains the source's irregular sampling; production uses the uniform
// PSI chart and a uniform bounded-tanh coordinate for the PUV shape,
// upsamples the tables with natural cubics, and reads them back with
// arithmetic-index linear interpolation.
//
// These tables own their arrays and borrow their immutable model only
// during construction. Changing any cosmology, IA parameter, source
// distribution, source triplet, grid or interpolation choice creates a
// new table. There is no pointer-identity cache and no mutable global.

typedef simde__m128d grid_v2d;

struct physics_grid_3pcf {
  physics_grid_settings_3pcf settings; // complete construction-time copy
  int nwindow; // pixel-window samples; zero disables smoothing
  double* window; // owned [nwindow], because a grid can outlive its model
  int nr; // nodes on the RUV middle-side axis log_r
  int nu; // nodes on the RUV ratio axis log_u
  int nv; // nodes on the RUV difference axis v
  int np; // nodes on the PUV perimeter axis log_p
  int nell; // nodes on the b_L magnitude axis log_ell
  int npu; // nodes on each PUV shape axis
  int npsi; // nodes on the b_L angular axis
  int nmu; // quadrature nodes on the mu axis
  int has_residual; // false for exact pure-NLA parameter choices
  double shape_min; // smallest side/perimeter ratio of the PUV chart
  double* log_r; // [nr], log middle side of sorted RUV triangles
  double* log_u; // [nu], log of the smallest/middle side ratio
  double* v; // [nv], (largest-middle)/smallest in [0,1]
  double* log_p; // [np], log perimeter of PUV triangles
  double* pu; // [npu], physical side/perimeter shape nodes
  double* shape_coordinate; // [npu], interpolation chart of the pu nodes
  double* log_ell; // [nell], log two-leg magnitude ELL
  double* log_psi; // [npsi], physical log angles, kept for diagnostics
  double* psi_coordinate; // [npsi], interpolation chart, uniform in production
  double* mu; // [nmu], two-leg angle cosines from -1 up to 1-epmu
  double* log_baseline; // [nr,nu,nv], log NLA, or log GGG if signed_baseline
  double* baseline_ratio; // optional [nr,nu,nv], signed NLA/GGG
  double* residual; // [np,npu,npu], signed value or scaled signed ratio
  double* log_residual_scale; // optional [np], log GGG(p/3,p/3,p/3)
  // The b_L table, in [component,chart,L,nell,npsi] order: component 0
  // is the baseline and 1 the residual; chart 0 is PSI = small_angle
  // and chart 1 its mirror. The pixel window at the closing side, W3,
  // sits inside the mu integral; the query stage applies the two leg
  // windows W1*W2 after interpolation.
  double* multipoles;
};

// This owner retains geometry only: no sampled bispectrum or b_L value.
// Construction fixes its allocation dimensions. The copied axes
// and reader convention form the complete matching key for a later grid;
// comparing endpoints alone would miss a changed irregular reference axis.
// No pointer to the constructing grid, model or caller ell array survives.
struct physics_grid_query_3pcf {
  int nell; // requested Fourier values per leg, fixed for this query
  double* ell; // owned query axis; each grid supplies its own windows
  int grid_nell; // interpolation log_ELL axis size
  int grid_npsi; // angular interpolation axis size
  int lmax; // sets the physical table's L stride and output row count
  int reference_mode; // determines coordinate tolerance and cell reader
  double* log_ell; // owned complete interpolation-axis key
  double psi_scale; // selects the angular coordinate, even at equal counts
  double* log_psi; // owned complete physical-axis key
  double* psi_coordinate; // owned complete interpolation-axis key
  int* charts; // [nell,nell], which ordered-leg chart each pair reads
  size_t* cells; // [nell,nell], lower-left interpolation offsets
  double* fractions; // [nell,nell,4], ordered bilinear corner weights
};

// ============================================================================
// [SECTION] AXIS AND ALLOCATION HELPERS
// ============================================================================

// Stop with a message when a structural requirement fails. The grid code
// treats any such failure as a programming error, never as a recoverable
// runtime state.
static void grid_require(const int condition, const char* message)
{
  if (!condition) {
    fprintf(stderr, "physics_grid_3pcf: %s\n", message);
    exit(EXIT_FAILURE);
  }
}


// Allocate zero-initialized storage after checking that the byte count
// cannot overflow. The zero fill matters physically: it is what lets a
// pure-NLA grid keep untouched residual planes at exactly zero.
static void* grid_allocate(const size_t count, const size_t bytes)
{
  grid_require(count > 0 && count <= SIZE_MAX/bytes,
    "grid allocation overflows");
  void* result = calloc(count, bytes);
  grid_require(result != NULL, "grid allocation failed");
  return result;
}


// Allocate all worker storage before entering a parallel region. Each
// slice includes one cache-line gap, so adjacent workers cannot write
// the same line even when calloc's starting address is not 64-byte aligned.
static double* worker_storage(const size_t count, size_t* stride)
{
  grid_require(count <= SIZE_MAX - 15, "worker stride overflows");
  *stride = ((count + 7)/8 + 1)*8;
  const size_t workers = (size_t)omp_get_max_threads();
  grid_require(workers > 0 && *stride <= SIZE_MAX/workers,
    "worker storage size overflows");
  return grid_allocate(workers*(*stride), sizeof(double));
}


// Multiply three positive dimensions into a size_t element count,
// aborting on overflow before any allocation can use the product.
static size_t grid_product(const int first, const int second, const int third)
{
  grid_require(first > 0 && second > 0 && third > 0,
    "grid dimensions must be positive");
  grid_require((size_t)first <= SIZE_MAX/(size_t)second
    && (size_t)third <= SIZE_MAX/((size_t)first*second),
    "grid dimension product overflows");
  return (size_t)first*second*third;
}


// ---------------------------------------------------------------------------
// Fill the settings with the fastnc reference resolution, so a default
// grid reproduces the pinned source's sampling. The groups below
// control, in order:
//
//   ell_min..epmu - the per-leg multipole range and the cap
//     mu <= 1 - epmu on the cosine between the two Fourier legs;
//   shape_stretch, psi_scale - the production coordinate charts for the
//     PUV shape axes and the opening angle PSI;
//   nr, nu, nv - the sorted-triangle RUV baseline table;
//   np, npu_log, npu_linear - the PUV residual table;
//   nell, npsi_log, npsi_linear - the b_L(ELL,PSI) multipole table;
//   nmu_log, nmu_linear - the mu quadrature feeding each b_L;
//   lmax - the largest Legendre degree kept;
//   reference_mode..residual_scaling - chart, upsampling and storage
//     switches, at the production choices.
//
// Parameters:
//   settings - the structure to fill completely
//
// Returns:
//   Nothing; every member receives a value.
// ---------------------------------------------------------------------------
void physics_grid_defaults_3pcf(physics_grid_settings_3pcf* settings)
{
  grid_require(settings != NULL, "missing grid settings");
  settings->ell_min = 0.1;
  settings->ell_max = 1.0e5;
  settings->epmu = 1.0e-7;
  settings->shape_stretch = 3.0;
  settings->psi_scale = 0.0;
  settings->nr = 35;
  settings->nu = 35;
  settings->nv = 25;
  settings->np = 35;
  settings->npu_log = 8;
  settings->npu_linear = 22;
  settings->nell = 100;
  settings->npsi_log = 50;
  settings->npsi_linear = 80;
  settings->nmu_log = 30;
  settings->nmu_linear = 50;
  settings->lmax = 20;
  settings->reference_mode = 0;
  settings->table_boost = 2;
  settings->multipole_boost = 2;
  settings->npsi = 80;
  settings->prepare_multipoles = 1;
  settings->nshape = 80;
  settings->signed_baseline = 0;
  settings->residual_scaling = 0;
}


// Count the nodes of a dense axis that places boost subintervals inside
// each coarse interval, checking that the arithmetic fits in an int.
static int dense_grid_count(const int count, const int boost)
{
  grid_require(count >= 2 && boost >= 1
    && count - 1 <= (INT_MAX - 1)/boost, "dense grid count overflows");
  return (count - 1)*boost + 1;
}


// ---------------------------------------------------------------------------
// Upsample one uniform row through a natural cubic spline. The coarse
// nodes y_0..y_{count-1} sit one unit apart when distance is measured
// in coarse cells, so interval index i and offset t in [0,1] address
// the point i + t. On each interval the spline is the cubic
//
//   S(t) = y_i + t*(b + t*(c + t*d)),
//
// and the code names its coefficients linear (b), quadratic (c) and
// cubic (d). Writing M_i for the spline's second derivative at node i,
// continuity of value, slope and curvature at the interior nodes gives
// the tridiagonal system, in units of one coarse cell,
//
//   M_{i-1} + 4 M_i + M_{i+1} = 6*(y_{i+1} - 2 y_i + y_{i-1}),
//
// whose right-hand side is six times the second difference of the data.
// The natural boundary condition sets the curvature to zero at both
// ends, M_0 = M_{count-1} = 0. The Thomas algorithm solves the system
// in two sweeps: forward elimination subtracts each row's share of the
// row above it, leaving the modified pivots
// diagonal[i] = 4 - 1/diagonal[i-1] and a modified right-hand side,
// and back substitution walks from the last interior node to the first,
// turning that right-hand side into M_i. The caller prepares the pivots
// once per axis.
//
// The standard uniform-spacing identities then give the coefficients on
// interval i:
//
//   b = y_{i+1} - y_i - (M_{i+1} + 2 M_i)/6,
//   c = M_i/2,
//   d = (M_{i+1} - M_i)/6.
//
// Only preparation evaluates this cubic. Readers afterwards use
// arithmetic-index linear interpolation on the dense output, as in
// cosmo2D.c's Limber table preparation.
//
// Parameters:
//   coarse - [count], the row's coarse samples y_i
//   count - coarse node count, at least two
//   boost - dense subintervals per coarse interval
//   dense - [(count-1)*boost + 1], receives the upsampled row
//   second - [count], scratch holding the second derivatives M_i
//   diagonal - [count], prepared Thomas pivots for this count
//   offsets - [boost+1], prepared subcell coordinates t = step/boost
//
// Returns:
//   Nothing; dense holds the natural-cubic value at every dense node.
// ---------------------------------------------------------------------------
static void upsample_uniform_row(
  const double* coarse, // [count], coarse samples of one row
  const int count, // coarse node count, at least two
  const int boost, // dense subintervals per coarse interval
  double* dense, // [(count-1)*boost + 1], upsampled output
  double* second, // [count], second-derivative scratch
  const double* diagonal, // [count], prepared Thomas pivots
  const double* offsets // [boost+1], subcell coordinates t
)
{
  second[0] = 0.0;
  second[count - 1] = 0.0;
  // Forward elimination: second temporarily holds the modified
  // right-hand side, built from the second differences of the data.
  for (int index=1; index<count - 1; index++) {
    const double rhs = 6.0*(coarse[index + 1] - 2.0*coarse[index]
      + coarse[index - 1]);
    second[index] = rhs - (index == 1 ? 0.0
      : second[index - 1]/diagonal[index - 1]);
  }
  // Back substitution: from the last interior node down, each division
  // by the prepared pivot turns the stored value into the curvature M_i.
  for (int index=count - 2; index>=1; index--) {
    second[index] = (second[index] - second[index + 1])/diagonal[index];
  }
  // Neighboring dense samples share one cubic's coefficients. Evaluate
  // that polynomial with the same operations, using prepared offsets.
  // Only the final interval writes its right endpoint, so the loop
  // writes every dense cell exactly once, including on a coarse grid
  // with only two nodes.
  for (int index=0; index<count - 1; index++) {
    const double linear = coarse[index + 1] - coarse[index]
      - (second[index + 1] + 2.0*second[index])/6.0;
    const double quadratic = 0.5*second[index];
    const double cubic = (second[index + 1] - second[index])/6.0;
    const int steps = index == count - 2 ? boost + 1 : boost;
    for (int step=0; step<steps; step++) {
      const double offset = offsets[step];
      dense[index*boost + step] = coarse[index]
        + offset*(linear + offset*(quadratic + offset*cubic));
    }
  }
}


// Tensor-product upsampling applies the same one-dimensional solve to
// every line of a rectangular array, one axis after another. Each worker
// owns its line and scratch. A boost of one leaves that axis unchanged,
// so the routine never interpolates across component or L labels when
// it prepares the b_L arrays.
static double* upsample_uniform_grid(double* input, int* dimensions,
  const int* boosts)
{
  for (int axis=0; axis<3; axis++) {
    const int boost = boosts[axis];
    if (boost == 1) {
      continue;
    }
    const int old_count = dimensions[axis];
    const int new_count = dense_grid_count(old_count, boost);
    size_t before = 1;
    size_t after = 1;
    for (int index=0; index<axis; index++) {
      before *= (size_t)dimensions[index];
    }
    for (int index=axis + 1; index<3; index++) {
      after *= (size_t)dimensions[index];
    }
    grid_require(before <= SIZE_MAX/after
      && (size_t)new_count <= SIZE_MAX/(before*after),
      "upsampled tensor size overflows");
    double* output = grid_allocate(before*after*new_count, sizeof(double));
    // The natural-cubic matrix and subcell coordinates depend only on
    // this axis's count and boost. Share their exact original values
    // across all lines, before launching workers; no physical data
    // enter. diagonal[0] = 1.0 is a placeholder that keeps the array
    // fully initialized and is never read: the solve starts at the
    // first interior node, whose pivot is the constant 4.
    double* diagonal = grid_allocate((size_t)old_count, sizeof(double));
    diagonal[0] = 1.0;
    for (int index=1; index<old_count - 1; index++) {
      diagonal[index] = 4.0 - (index == 1 ? 0.0 : 1.0/diagonal[index - 1]);
    }
    double* offsets = grid_allocate((size_t)boost + 1, sizeof(double));
    for (int step=0; step<=boost; step++) {
      offsets[step] = (double)step/boost;
    }
    size_t worker_stride;
    double* scratch = worker_storage((size_t)2*old_count + new_count,
      &worker_stride);
    #pragma omp parallel if(!omp_in_parallel())
    {
      double* coarse = scratch + (size_t)omp_get_thread_num()*worker_stride;
      double* fine = coarse + old_count;
      double* second = fine + new_count;
      // One iteration upsamples one line: the samples obtained by
      // fixing every other index and letting this axis run. before
      // counts the index combinations left of the axis and after those
      // right of it, so line/after and line%after recover the fixed
      // outer and inner indices. Each worker gathers its line into
      // private scratch, solves the one-dimensional spline there, and
      // scatters the dense result, so no two workers touch the same
      // output element.
      #pragma omp for schedule(static)
      for (size_t line=0; line<before*after; line++) {
        const size_t outer = line/after;
        const size_t inner = line%after;
        for (int index=0; index<old_count; index++) {
          coarse[index] = input[(outer*old_count + index)*after + inner];
        }
        upsample_uniform_row(coarse, old_count, boost, fine, second,
          diagonal, offsets);
        for (int index=0; index<new_count; index++) {
          output[(outer*new_count + index)*after + inner] = fine[index];
        }
      }
    }
    free(scratch);
    free(diagonal);
    free(offsets);
    free(input);
    input = output;
    dimensions[axis] = new_count;
  }
  return input;
}


// Map a side/perimeter ratio onto the bounded-tanh interpolation chart.
// The residual varies fastest near the edges of the shape domain -
// ratios near the floor and near the equal-split value 0.5 - so the
// chart compresses those edges: uniform samples of the coordinate t in
// [-1,1] crowd toward both ends of the physical ratio in [minimum,0.5],
// and the uniform coordinate keeps arithmetic cell indexing, with no
// binary search on irregular physical ratios. A larger stretch resolves
// smaller edge ratios at the price of more nodes for the same central
// resolution.
static double shape_to_coordinate(const double ratio, const double minimum,
  const double stretch)
{
  // Pin the exact endpoints: evaluating the inverse hyperbolic tangent
  // at an argument that tanh roundoff pushed past one would amplify
  // that last-bit error into the chart.
  if (ratio <= minimum) return -1.0;
  if (ratio >= 0.5) return 1.0;
  const double normalized = 2.0*(ratio - minimum)/(0.5 - minimum) - 1.0;
  return atanh(tanh(stretch)*normalized)/stretch;
}


// Invert the bounded-tanh chart: map a coordinate in [-1,1] back to the
// physical side/perimeter ratio in [minimum, 0.5].
static double coordinate_to_shape(const double coordinate, const double minimum,
  const double stretch)
{
  if (coordinate <= -1.0) return minimum;
  if (coordinate >= 1.0) return 0.5;
  return minimum + 0.5*(0.5 - minimum)
    *(1.0 + tanh(stretch*coordinate)/tanh(stretch));
}


// Build a uniformly spaced axis and pin both endpoints to their exact
// requested values, so interval arithmetic never lands outside them.
static double* linear_axis(const double lower, const double upper,
  const int count)
{
  grid_require(count >= 2 && isfinite(lower) && isfinite(upper)
    && upper > lower, "invalid interpolation axis");
  double* result = grid_allocate((size_t)count, sizeof(double));
  for (int index=0; index<count; index++) {
    result[index] = lower + (upper - lower)*index/(count - 1);
  }
  result[0] = lower;
  result[count - 1] = upper;
  return result;
}


// Map the opening angle PSI onto the production interpolation
// coordinate. A positive scale selects x = log(PSI) + PSI/scale, which
// spaces samples logarithmically below that angle and nearly uniformly
// in angle above it; the coordinate is monotone because its derivative
// 1/PSI + 1/scale stays positive. A zero scale keeps the original
// log(PSI) expression, including its floating-point rounding and the
// reference arrays built from it.
static double psi_grid_coordinate(const double psi, const double scale)
{
  const double logarithm = log(psi);
  return scale > 0.0 ? logarithm + psi/scale : logarithm;
}


// Recover the physical log-angles of a uniform coordinate axis by
// inverting x = t + exp(t)/scale for t = log(PSI), only while preparing
// the axis. Each interior node needs one root of this monotone
// function, and the known physical endpoints bracket it. The fixed
// count of 64 bisection steps halves the bracket more times than the
// 52 fractional mantissa bits of a double, so the result resolves the
// root to the last bit without any convergence tolerance that could
// change the sampled nodes; no root search survives into a production
// reader. The endpoints keep the exact bits the caller passed.
static double* physical_psi_axis(const double* coordinates, const int count,
  const double lower, const double upper, const double scale)
{
  double* logarithms = grid_allocate((size_t)count, sizeof(double));
  if (scale == 0.0) {
    memcpy(logarithms, coordinates, (size_t)count*sizeof(double));
    return logarithms;
  }
  logarithms[0] = lower;
  logarithms[count - 1] = upper;
  for (int index=1; index<count - 1; index++) {
    double left = lower;
    double right = upper;
    for (int step=0; step<64; step++) {
      const double middle = left + 0.5*(right - left);
      const double value = middle + exp(middle)/scale;
      if (value < coordinates[index]) {
        left = middle;
      } else {
        right = middle;
      }
    }
    logarithms[index] = left + 0.5*(right - left);
    grid_require(logarithms[index] > logarithms[index - 1],
      "mixed PSI chart has indistinguishable physical nodes");
  }
  grid_require(logarithms[count - 1] > logarithms[count - 2],
    "mixed PSI chart has indistinguishable physical endpoints");
  return logarithms;
}


// Build the fastnc mixed axis: a logarithmic block from lower to the
// join, continued by a linear block from the join to upper. The log
// block stops one node short of the join, which the linear block then
// supplies, so the two blocks never duplicate a node. When a narrower
// requested domain starts above the join, the axis skips the
// logarithmic block entirely; keeping it would produce duplicate or
// decreasing nodes at that boundary.
static double* loglinear_axis(const double lower, const double join,
  const double upper, const int nlog, const int nlinear, int* count)
{
  grid_require(lower > 0.0 && upper > lower && nlog >= 0 && nlinear >= 2,
    "invalid mixed logarithmic/linear axis");
  const int logarithmic = lower < join ? nlog : 0;
  grid_require(logarithmic <= INT_MAX - nlinear, "mixed axis is too large");
  *count = logarithmic + nlinear;
  double* result = grid_allocate((size_t)*count, sizeof(double));
  for (int index=0; index<logarithmic; index++) {
    result[index] = exp(log(lower) + log(join/lower)*index/logarithmic);
  }
  const double start = logarithmic > 0 ? join : lower;
  for (int index=0; index<nlinear; index++) {
    result[logarithmic + index] = start + (upper - start)*index/(nlinear - 1);
  }
  result[0] = lower;
  result[*count - 1] = upper;
  return result;
}


// ============================================================================
// [SECTION] INTERPOLATION READERS
// ============================================================================

// Snap a query coordinate that falls just outside the table onto its
// nearest boundary, so that roundoff in a caller's logarithm or chart
// transform cannot abort an evaluation at the very edge of the domain.
// The tolerance exists for that snap alone; a query clearly outside the
// support still aborts below.
static double corrected_coordinate(const double coordinate,
  const double lower, const double upper, const int reference_mode)
{
  // The reference branch reproduces fastnc's edge_correction, which
  // calls numpy.isclose(x, bound, atol, rtol) with its local variables
  // atol = 1e-8 and rtol = 1e-5 passed positionally. numpy's signature
  // is isclose(a, b, rtol, atol), so numpy receives rtol = 1e-8 and
  // atol = 1e-5, and its test |x - bound| <= atol + rtol*|bound|
  // becomes |x - bound| <= 1e-5 + 1e-8*|bound|. The expressions below
  // keep those effective values exactly. The production branch instead
  // allows 64 machine epsilons, scaled by 1 + |bound| so the tolerance
  // is absolute near zero and relative at large bounds, and it never
  // extrapolates farther.
  const double lower_tolerance = reference_mode
    ? 1.0e-5 + 1.0e-8*fabs(lower) : 64.0*DBL_EPSILON*(1.0 + fabs(lower));
  const double upper_tolerance = reference_mode
    ? 1.0e-5 + 1.0e-8*fabs(upper) : 64.0*DBL_EPSILON*(1.0 + fabs(upper));
  if (coordinate < lower && lower - coordinate <= lower_tolerance) {
    return lower;
  }
  if (coordinate > upper && coordinate - upper <= upper_tolerance) {
    return upper;
  }
  grid_require(isfinite(coordinate) && coordinate >= lower
    && coordinate <= upper, "query lies outside prepared grid support");
  return coordinate;
}


// Locate the interpolation interval that contains a coordinate: by
// index arithmetic on a uniform axis, by bisection on an irregular
// reference axis. The result is the left node, clamped so the right
// neighbor index + 1 stays inside the table.
static int grid_interval(
  const double* axis, // [count], increasing interpolation nodes
  const int count, // node count, at least two
  const double coordinate, // query position inside the axis span
  const int uniform // nonzero selects arithmetic indexing
)
{
  if (uniform) {
    const double position = (coordinate - axis[0])*(count - 1)
      /(axis[count - 1] - axis[0]);
    return (int)floor(fmax(0.0, fmin(count - 2.0, position)));
  }
  int lower = 0;
  int upper = count - 1;
  while (upper - lower > 1) {
    const int middle = lower + (upper - lower)/2;
    if (axis[middle] <= coordinate) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return lower;
}


// ---------------------------------------------------------------------------
// Trilinear interpolation on a three-dimensional table. Each axis
// contributes an interval index and a fraction, the position inside
// that interval in [0,1]. The value is the weighted sum over the
// cell's eight corners: each corner's weight is a product of one
// factor per axis, the fraction for the far face (shift 1) or one
// minus the fraction for the near face (shift 0). Those products are
// the volumes of the sub-boxes opposite each corner, so they sum to
// one and the interpolant reproduces the corner values exactly.
//
// Every coordinate first passes through corrected_coordinate, which
// snaps boundary roundoff and aborts on a genuine miss, so this reader
// never extrapolates.
//
// Parameters:
//   values - [count0,count1,count2] table in C order
//   axis0, axis1, axis2 - the three node axes, each strictly increasing
//   count0, count1, count2 - node counts per axis
//   coordinate0..coordinate2 - the query position on each axis
//   reference_mode - selects fastnc tolerances and bisection reads
//
// Returns:
//   The trilinearly interpolated table value.
// ---------------------------------------------------------------------------
static double trilinear_grid(
  const double* values, // [count0,count1,count2] table in C order
  const double* axis0, // [count0], strictly increasing nodes
  const int count0, // first-axis node count
  const double* axis1, // [count1], strictly increasing nodes
  const int count1, // second-axis node count
  const double* axis2, // [count2], strictly increasing nodes
  const int count2, // third-axis node count
  double coordinate0, // query position on axis0
  double coordinate1, // query position on axis1
  double coordinate2, // query position on axis2
  const int reference_mode // fastnc tolerances and bisection when set
)
{
  coordinate0 = corrected_coordinate(coordinate0, axis0[0], axis0[count0 - 1],
    reference_mode);
  coordinate1 = corrected_coordinate(coordinate1, axis1[0], axis1[count1 - 1],
    reference_mode);
  coordinate2 = corrected_coordinate(coordinate2, axis2[0], axis2[count2 - 1],
    reference_mode);
  const int index0 = grid_interval(axis0, count0, coordinate0, !reference_mode);
  const int index1 = grid_interval(axis1, count1, coordinate1, !reference_mode);
  const int index2 = grid_interval(axis2, count2, coordinate2, !reference_mode);
  const double fraction0 = (coordinate0 - axis0[index0])
    /(axis0[index0 + 1] - axis0[index0]);
  const double fraction1 = (coordinate1 - axis1[index1])
    /(axis1[index1 + 1] - axis1[index1]);
  const double fraction2 = (coordinate2 - axis2[index2])
    /(axis2[index2 + 1] - axis2[index2]);
  double sum = 0.0;
  for (int shift0=0; shift0<2; shift0++) {
    for (int shift1=0; shift1<2; shift1++) {
      for (int shift2=0; shift2<2; shift2++) {
        const size_t index = ((size_t)(index0 + shift0)*count1
          + index1 + shift1)*count2 + index2 + shift2;
        sum += (shift0 ? fraction0 : 1.0 - fraction0)
          *(shift1 ? fraction1 : 1.0 - fraction1)
          *(shift2 ? fraction2 : 1.0 - fraction2)*values[index];
      }
    }
  }
  return sum;
}


// ---------------------------------------------------------------------------
// Interpolate the prepared projected bispectrum at one triangle of
// Fourier multipoles. The total is the sum of two stored pieces,
//
//   B(ell1,ell2,ell3) = baseline + residual,
//
// where the baseline is the NLA part on the sorted RUV table and the
// residual is the signed TATT remainder on the PUV table; the file head
// defines both coordinate systems and the NLA, TATT and GGG terms.
//
// Baseline read (component 0 or 1). The sides sort into largest, middle
// and smallest; the table coordinates are log(middle),
// log(smallest/middle) and (largest - middle)/smallest, and the stored
// value is log B, so the read exponentiates. With signed_baseline=1 the
// log table holds positive GGG and a second table holds the signed
// NLA/GGG ratio; their product restores the signed NLA value.
//
// Residual read (component 0 or 2, when the IA parameters produce a
// residual at all). The coordinates are log(perimeter) and the two side
// fractions ell1/perimeter and ell2/perimeter, each mapped onto the
// bounded-tanh chart in production. With residual_scaling=1 a
// one-dimensional positive scale table - the gravity-only equilateral B
// at the same perimeter - multiplies the interpolated ratio back to a
// physical value.
//
// Parameters:
//   grid - prepared immutable tables
//   ell1, ell2, ell3 - triangle sides, positive Fourier multipoles that
//     must satisfy the triangle inequality
//   component - total=0, NLA baseline=1, residual IA=2
//
// Returns:
//   The interpolated projected E-mode bispectrum. The pixel window is
//   not part of this value; callers that need it multiply afterwards.
// ---------------------------------------------------------------------------
double physics_grid_projected_3pcf(
  const physics_grid_3pcf* grid, // prepared immutable tables
  const double ell1, // first triangle side
  const double ell2, // second triangle side
  const double ell3, // closing third side
  const int component // total=0, NLA baseline=1, residual IA=2
)
{
  grid_require(grid && ell1 > 0.0 && ell2 > 0.0 && ell3 > 0.0
    && isfinite(ell1 + ell2 + ell3) && component >= 0 && component <= 2,
    "invalid projected-grid query");
  const physics_grid_settings_3pcf* settings = &grid->settings;
  const double maximum = fmax(ell1, fmax(ell2, ell3));
  const double minimum = fmin(ell1, fmin(ell2, ell3));
  const double perimeter = ell1 + ell2 + ell3;
  // The triangle test allows a slack of 32 machine epsilons on the
  // perimeter: quadrature callers build the closing side from the two
  // legs, and an exactly degenerate triangle would otherwise fail on
  // the last bit of that construction.
  grid_require(2.0*maximum <= perimeter*(1.0 + 32.0*DBL_EPSILON),
    "projected-grid query is not a triangle");
  double baseline = 0.0;
  double residual = 0.0;
  if (component != 2) {
    const double middle = perimeter - maximum - minimum;
    const double ratio = minimum/middle;
    const double difference = fmax(0.0, fmin(1.0, (maximum - middle)/minimum));
    baseline = exp(trilinear_grid(grid->log_baseline,
      grid->log_r, grid->nr, grid->log_u, grid->nu,
      grid->v, grid->nv, log(middle), log(ratio), difference,
      settings->reference_mode));
    if (settings->signed_baseline) {
      // GGG stays positive, so only this dimensionless ratio determines
      // the sign; the code never takes log of an absolute NLA value and
      // never replaces a negative one.
      baseline *= trilinear_grid(grid->baseline_ratio,
        grid->log_r, grid->nr, grid->log_u, grid->nu,
        grid->v, grid->nv, log(middle), log(ratio), difference,
        settings->reference_mode);
    }
  }
  if (component != 1 && grid->has_residual) {
    const double ratio1 = corrected_coordinate(ell1/perimeter,
      grid->shape_min, 0.5, settings->reference_mode);
    const double ratio2 = corrected_coordinate(ell2/perimeter,
      grid->shape_min, 0.5, settings->reference_mode);
    const double shape1 = settings->reference_mode ? ratio1
      : shape_to_coordinate(ratio1, grid->shape_min, settings->shape_stretch);
    const double shape2 = settings->reference_mode ? ratio2
      : shape_to_coordinate(ratio2, grid->shape_min, settings->shape_stretch);
    residual = trilinear_grid(grid->residual,
      grid->log_p, grid->np, grid->shape_coordinate, grid->npu,
      grid->shape_coordinate, grid->npu, log(perimeter), shape1, shape2,
      settings->reference_mode);
    if (settings->residual_scaling) {
      // The positive scale table has only a perimeter axis. This read
      // uses its current dense nodes with the same linear reader
      // convention as the ratio; the scale stays positive, so the
      // interpolated ratio alone carries the residual's sign.
      const double coordinate = corrected_coordinate(log(perimeter),
        grid->log_p[0], grid->log_p[grid->np - 1], settings->reference_mode);
      const int cell = grid_interval(grid->log_p, grid->np, coordinate,
        !settings->reference_mode);
      const double fraction = (coordinate - grid->log_p[cell])
        /(grid->log_p[cell + 1] - grid->log_p[cell]);
      const double log_scale = (1.0 - fraction)*grid->log_residual_scale[cell]
        + fraction*grid->log_residual_scale[cell + 1];
      residual *= exp(log_scale);
    }
  }
  return baseline + residual;
}


// ============================================================================
// [SECTION] MULTIPOLE PREPARATION
// ============================================================================

// ---------------------------------------------------------------------------
// Prepare the quadrature weights that turn samples of the projected
// bispectrum B(mu) into its Legendre coefficients
//
//   b_L = (2L+1)/2 * integral B(mu) P_L(mu) dmu,
//
// with mu the cosine of the angle between the two Fourier multipole
// legs. The integral runs from -1 up to the cap mu = 1 - epmu, which
// keeps the closing side away from degeneracy; the cap endpoint stays
// explicit in the last interval, and the weights invent no missing
// piece beyond it. The normalization (2L+1)/2 makes b_L the
// coefficients of the expansion B(mu) = sum_L b_L P_L(mu).
//
// The source method named 'gauss-legendre' does not sample at Gauss
// nodes: it integrates the piecewise-linear interpolant of B through
// the mu nodes exactly against P_L, and this file reproduces that
// choice. On one interval B = a*mu + b, so the interval contributes
// a*J1 + b*J0 with the two normalized interval moments
//
//   J0 = (2L+1)/2 * integral P_L(mu) dmu,
//   J1 = (2L+1)/2 * integral mu P_L(mu) dmu,
//
// both over that interval. Writing a and b in terms of the two endpoint
// values of B distributes a*J1 + b*J0 onto the endpoints once:
//
//   weight[lower] += (upper*J0 - J1)/width,
//   weight[upper] += (J1 - lower*J0)/width.
//
// The code evaluates J0 and J1 from exact Legendre identities rather
// than numerically. The identity (2L+1) integral P_L = P_{L+1} - P_{L-1}
// turns J0 into half a difference of neighboring polynomials, and
// (2L+1) mu P_L = (L+1) P_{L+1} + L P_{L-1} reduces J1 to the same
// identity applied two degrees up and down, which is why the recurrence
// below tabulates polynomials through degree lmax+2.
//
// Parameters:
//   grid - carries the mu axis and the Legendre cutoff lmax
//
// Returns:
//   Owned [lmax+1,nmu] weights; the caller frees them. Contracting
//   row L with samples B(mu[node]) yields b_L.
// ---------------------------------------------------------------------------
static double* source_legendre_weights(const physics_grid_3pcf* grid)
{
  const int lmax = grid->settings.lmax;
  const int count = grid->nmu;
  double* weights = grid_allocate((size_t)(lmax + 1)*count, sizeof(double));
  double* polynomial = grid_allocate((size_t)(lmax + 3)*count, sizeof(double));
  // Tabulate P_0..P_{lmax+2} at every mu node with the three-term
  // recurrence L P_L = (2L-1) mu P_{L-1} - (L-1) P_{L-2}; the two
  // extra degrees serve the J1 identity in the header.
  for (int node=0; node<count; node++) {
    const double coordinate = grid->mu[node];
    polynomial[node] = 1.0;
    polynomial[count + node] = coordinate;
    for (int order=2; order<=lmax + 2; order++) {
      polynomial[(size_t)order*count + node]
        = ((2.0*order - 1.0)*coordinate
        *polynomial[(size_t)(order - 1)*count + node]
        - (order - 1.0)*polynomial[(size_t)(order - 2)*count + node])/order;
    }
  }
  // Accumulate the endpoint weights. One outer iteration owns one
  // Legendre degree; the inner loop walks the mu intervals, forms the
  // moments J0 and J1 of that interval from the identities in the
  // header, and adds (upper*J0 - J1)/width to the left endpoint's
  // weight and (J1 - lower*J0)/width to the right one's, so an interior
  // node collects contributions from both neighboring intervals.
  // Degree zero takes both moments from the explicit integrals of 1
  // and mu, and degree one takes its first moment from the explicit
  // integral of mu^2, because the difference identities would need the
  // polynomials P_{L-1} and P_{L-2} that do not exist there.
  for (int order=0; order<=lmax; order++) {
    for (int node=0; node<count - 1; node++) {
      const double lower = grid->mu[node];
      const double upper = grid->mu[node + 1];
      double j0;
      double j1;
      if (order == 0) {
        j0 = 0.5*(upper - lower);
        j1 = 0.25*(upper*upper - lower*lower);
      } else {
        const double plus = polynomial[(size_t)(order + 1)*count + node + 1]
          - polynomial[(size_t)(order + 1)*count + node];
        const double minus = polynomial[(size_t)(order - 1)*count + node + 1]
          - polynomial[(size_t)(order - 1)*count + node];
        j0 = 0.5*(plus - minus);
        if (order == 1) {
          j1 = 0.5*(upper*upper*upper - lower*lower*lower);
        } else {
          const double central = polynomial[(size_t)order*count + node + 1]
            - polynomial[(size_t)order*count + node];
          const double high = polynomial[(size_t)(order + 2)*count + node + 1]
            - polynomial[(size_t)(order + 2)*count + node];
          const double low = polynomial[(size_t)(order - 2)*count + node + 1]
            - polynomial[(size_t)(order - 2)*count + node];
          j1 = 0.5*((order + 1.0)/(2.0*order + 3.0)*(high - central)
            + order/(2.0*order - 1.0)*(central - low));
        }
      }
      const double width = upper - lower;
      weights[(size_t)order*count + node] += (upper*j0 - j1)/width;
      weights[(size_t)order*count + node + 1] += (j1 - lower*j0)/width;
    }
  }
  free(polynomial);
  return weights;
}


// Flatten the b_L storage index [component,chart,L,scale,angle]; the
// struct declaration of physics_grid_3pcf documents each axis.
static size_t multipole_grid_index(const physics_grid_3pcf* grid,
  const int component, const int chart, const int order,
  const int scale, const int angle)
{
  return ((((size_t)component*2 + chart)*(grid->settings.lmax + 1) + order)
    *grid->nell + scale)*grid->npsi + angle;
}


// ---------------------------------------------------------------------------
// Integrate the prepared projected-bispectrum tables into the Legendre
// multipole table b_L(ELL,PSI) for both charts. At one two-leg
// magnitude ELL and opening angle PSI the legs are
//
//   ell1 = ELL cos(PSI),  ell2 = ELL sin(PSI),
//
// and the law of cosines closes the triangle at each quadrature node
// mu, the cosine of the angle between the legs. The routine samples the
// projected B on those triangles and contracts the sampled curve B(mu)
// with the weights of source_legendre_weights, which integrate the
// linear interpolant into the normalized Legendre coefficients b_L.
//
// The baseline is symmetric under exchanging the legs, so its chart-1
// plane equals its chart-0 plane and the code copies it; only the
// ordered TATT residual needs a genuine second-chart integration. The
// pixel window at the closing side, W3, multiplies the samples inside
// the mu integral; the query stage applies the leg windows W1 and W2
// after interpolation.
//
// Parameters:
//   grid - carries the coarse axes and receives grid->multipoles
//
// Returns:
//   Nothing; grid->multipoles holds [component,chart,L,nell,npsi].
// ---------------------------------------------------------------------------
static void prepare_multipole_grid(physics_grid_3pcf* grid)
{
  const int count = grid->nmu;
  const int lmax = grid->settings.lmax;
  const int nell = grid->nell;
  const int npsi = grid->npsi;
  const size_t per_chart = grid_product(lmax + 1, nell, npsi);
  grid_require(per_chart <= SIZE_MAX/4, "multipole charts are too large");
  grid->multipoles = grid_allocate(4*per_chart, sizeof(double));
  double* weights = source_legendre_weights(grid);
  // Interleave the weights of adjacent Legendre degrees so one vector
  // load below fills both lanes: entry [pair,node,0] holds degree
  // 2*pair and entry [pair,node,1] degree 2*pair + 1. The mu sequence
  // and every coefficient stay unchanged; only the storage order
  // differs. The original rows remain for the scalar tail degree when
  // lmax + 1 is odd, including the lmax = 0 case with no pairs at all.
  const int npair = (lmax + 1)/2;
  double* paired_weights = NULL;
  if (npair > 0) {
    paired_weights = grid_allocate(grid_product(npair, count, 2),
      sizeof(double));
    for (int pair=0; pair<npair; pair++) {
      for (int node=0; node<count; node++) {
        const size_t target = ((size_t)pair*count + node)*2;
        paired_weights[target] = weights[(size_t)(2*pair)*count + node];
        paired_weights[target + 1]
          = weights[(size_t)(2*pair + 1)*count + node];
      }
    }
  }
  size_t worker_stride;
  double* scratch = worker_storage((size_t)2*count, &worker_stride);
  #pragma omp parallel if(!omp_in_parallel())
  {
    double* baseline = scratch + (size_t)omp_get_thread_num()*worker_stride;
    double* residual = baseline + count;
    // One iteration owns one (ELL,PSI) sample of the multipole table:
    // it builds the two legs from that polar sample, closes the
    // triangle at every mu node, reads the projected B there, and
    // contracts the sampled curve B(mu) with the prepared Legendre
    // weights, which integrates the linear interpolant into b_L for
    // every degree at once. Workers write disjoint (scale,angle)
    // columns of every output plane, so the loop needs no
    // synchronization beyond its own schedule. The vector lanes inside
    // pair adjacent Legendre degrees L and L+1: both degrees reuse the
    // same sampled B(mu), so one pass over the nodes accumulates two
    // independent Legendre sums side by side.
    #pragma omp for collapse(2) schedule(static)
    for (int scale=0; scale<nell; scale++) {
      for (int angle=0; angle<npsi; angle++) {
        const double magnitude = exp(grid->log_ell[scale]);
        const double small = exp(grid->log_psi[angle]);
        for (int chart=0; chart<2; chart++) {
          if (chart == 1) {
            // The baseline reads the same sorted triangle in both
            // charts, so this copies its completed Legendre sums from
            // chart 0; only the ordered TATT residual still needs its
            // own second-chart calculation below.
            for (int order=0; order<=lmax; order++) {
              grid->multipoles[multipole_grid_index(grid, 0,
                1, order, scale, angle)]
                = grid->multipoles[multipole_grid_index(grid, 0,
                0, order, scale, angle)];
            }
            if (!grid->has_residual) {
              continue;
            }
          }
          // The literal is pi/2 to double precision: chart 1 holds the
          // mirrored opening angle PSI = pi/2 - small_angle.
          const double psi = chart == 0 ? small
            : 1.57079632679489661923 - small;
          const double first = magnitude*cos(psi);
          const double second = magnitude*sin(psi);
          for (int node=0; node<count; node++) {
            // The closing side follows the law of cosines, rearranged
            // as (ell1 - ell2)^2 + 2 ell1 ell2 (1 - mu). Near the cap
            // mu -> 1 the direct form ell1^2 + ell2^2 - 2 ell1 ell2 mu
            // subtracts two nearly equal numbers and loses the small
            // closing side to cancellation; the rearranged form adds
            // two nonnegative terms, and the difference of two close
            // legs is itself exact in floating point.
            const double difference = first - second;
            const double third = sqrt(difference*difference
              + 2.0*first*second*(1.0 - grid->mu[node]));
            if (chart == 0) {
              baseline[node] = physics_grid_projected_3pcf(grid,
                first, second, third, 1);
              if (grid->nwindow > 0) {
                baseline[node] *= physics_window_value_3pcf(grid->nwindow,
                  grid->window, third);
              }
            }
            if (grid->has_residual) {
              residual[node] = physics_grid_projected_3pcf(grid,
                first, second, third, 2);
              if (grid->nwindow > 0) {
                residual[node] *= physics_window_value_3pcf(grid->nwindow,
                  grid->window, third);
              }
            }
          }
          // A pure-NLA grid has exactly zero residual, and its
          // calloc-initialized planes keep +0 without this code ever
          // sampling or integrating zero values.
          const int ncomponent = grid->has_residual ? 2 : 1;
          for (int component=chart; component<ncomponent; component++) {
            const double* values = component == 0 ? baseline : residual;
            int order = 0;
            // Scalar form of the vector loop below, for one even
            // degree L and its neighbor:
            //
            //   double sum_L = 0.0, sum_L1 = 0.0;
            //   for (int node=0; node<count; node++) {
            //     sum_L  += weights[L*count + node]*values[node];
            //     sum_L1 += weights[(L+1)*count + node]*values[node];
            //   }
            //
            // Each sum is the contracted Legendre integral b_L of the
            // weights header. The vector loop runs both accumulations
            // at once: lane 0 (low) holds degree L and lane 1 (high)
            // holds L+1, and the lanes never mix because the two
            // Legendre sums stay independent until each lands in its
            // own output plane.
            for (; order + 1<=lmax; order += 2) {
              // Set both lanes of the running sums to zero: the two
              // neighboring degrees accumulate independently.
              grid_v2d sum = simde_mm_setzero_pd();
              const double* restrict pair_weights
                = paired_weights + (size_t)order*count;
              for (int node=0; node<count; node++) {
                // Load the interleaved pair from pair_weights[2*node]
                // and [2*node + 1] in one unaligned load (loadu needs
                // no vector-aligned address): the low lane takes index
                // 2*node, degree L, and the high lane takes
                // 2*node + 1, degree L+1.
                const grid_v2d weight = simde_mm_loadu_pd(pair_weights
                  + (size_t)node*2);
                // Broadcast the sampled B(mu[node]) into both lanes:
                // the two degrees weight the same physical sample.
                const grid_v2d value = simde_mm_set1_pd(values[node]);
                // Multiply lane by lane: each lane now holds the
                // scalar product weights[L,node]*values[node] for its
                // own degree.
                const grid_v2d term = simde_mm_mul_pd(weight, value);
                // Add the term into the running sums, keeping the
                // increasing mu-node order within each lane.
                sum = simde_mm_add_pd(sum, term);
              }
              double result[2];
              // Spill both lanes to memory - result[0] is the low
              // lane, degree L, and result[1] the high lane, L+1 -
              // before scattering them into their own planes.
              simde_mm_storeu_pd(result, sum);
              grid->multipoles[multipole_grid_index(grid, component,
                chart, order, scale, angle)] = result[0];
              grid->multipoles[multipole_grid_index(grid, component,
                chart, order + 1, scale, angle)] = result[1];
            }
            if (order <= lmax) {
              double sum = 0.0;
              for (int node=0; node<count; node++) {
                // Force the same rounding as the vector path: volatile
                // keeps the product and the addition separately
                // rounded, so the compiler cannot contract them into a
                // fused multiply-add. Without this, extending lmax by
                // one would move this degree from a fused scalar tail
                // into a separately rounded vector lane and change the
                // stored bits.
                volatile double term =
                  weights[(size_t)order*count + node]*values[node];
                sum += term;
              }
              grid->multipoles[multipole_grid_index(grid, component,
                chart, order, scale, angle)] = sum;
            }
          }
        }
      }
    }
  }
  free(scratch);
  free(paired_weights);
  free(weights);
}


// ============================================================================
// [SECTION] GRID CONSTRUCTION AND COARSE FILL
// ============================================================================

// ---------------------------------------------------------------------------
// Validate the settings, size every axis and allocate the coarse tables.
// No physical value enters here: the caller fills the tables afterwards
// and finish_grid upsamples them.
//
// Domain bounds. The multipole stage samples triangles whose legs are
//
//   ell1 = ELL cos(PSI),  ell2 = ELL sin(PSI),
//   ell3^2 = ell1^2 + ell2^2 - 2 ell1 ell2 mu,
//
// with the two-leg magnitude ELL in [sqrt(2) ell_min, sqrt(2) ell_max]
// (each leg runs over [ell_min, ell_max], so ELL runs from the
// equal-minimal to the equal-maximal pair), the opening angle PSI no
// smaller than psi_min = atan2(ell_min, ell_max), the most unequal leg
// pair, and mu, the cosine of the angle between the legs, capped at
// mu_max = 1 - epmu. The RUV and PUV axes must cover the sorted sides
// of every such triangle:
//
//   rmin, the smallest middle side. It arises in the mu -> 1 limit
//     where the triangle closes onto itself with one leg twice the
//     other, ell1 = 2 ell2 and ell3 = ell2: there the middle side is
//     ELL/sqrt(5). The expression reproduces the pinned fastnc bound
//     verbatim, including its companion sqrt(1 - 2 mu_max/5); fastnc
//     does not document the companion, and for every permitted epmu
//     (below 0.05) it exceeds 1/sqrt(5), so the fmin always selects
//     the closing-limit value.
//   rmax, the largest middle side: the longer leg at the widest
//     magnitude, ELL cos(psi_min), which the closing side can exceed
//     when mu < 0. The companion 1/sqrt(2) is the equal-leg value and
//     only acts as a floor, because cos(psi_min) >= cos(pi/4).
//   umin, the smallest smallest/middle ratio, from the two mechanisms
//     that squeeze a triangle: the mu cap closing an equal-leg pair,
//     where the ratio is ell3/ell1 = sqrt(2 epmu), and the most
//     unequal leg pair, where it is ell2/ell1 = tan(psi_min).
//   log_p, the PUV perimeter axis, runs from the smallest middle side,
//     a floor below any sampled perimeter, to 3 rmax, the perimeter of
//     the equilateral triangle with the largest middle side, as in
//     fastnc.
//   shape_min, the smallest side/perimeter fraction of the PUV chart.
//     The fastnc reference fixes it at 1e-6. Production also admits
//     0.25 ell_min/ell_max, a bound below ell_min/(3 ell_max), the
//     smallest fraction a triangle with all legs in range can reach,
//     so a very wide leg range extends the chart below the fastnc
//     floor.
//
// Node layouts. The reference PUV shape axis reproduces the pinned
// fastnc layout: one node at the floor 1e-6, npu_log log-spaced nodes
// from 1e-4 up to 3e-2 (the factor 300 in the code), npu_linear
// uniform nodes from 0.047 to 0.453, the mirrored log nodes
// approaching the equal-split value 0.5 from below, and one node at
// 0.5 itself; production replaces this layout with the uniform
// bounded-tanh chart. The mu axis applies loglinear_axis to the
// variable 1 - mu with the fastnc joins: logarithmic from epmu to
// 0.05, which crowds nodes against the cap mu_max where the closing
// side degenerates and the integrand varies fastest, then linear from
// 0.05 to 2.0, the value of 1 - mu at mu = -1 (anti-parallel legs);
// reversing and mapping back gives mu ascending from -1 to mu_max.
// The opening-angle axis joins its logarithmic and linear blocks at
// 1e-3 radians, again the fastnc value, before the production chart
// replaces it.
//
// Parameters:
//   settings - validated and copied into the grid
//   has_residual - nonzero allocates the PUV residual storage
//
// Returns:
//   An owned grid with every axis built and every table zeroed.
// ---------------------------------------------------------------------------
static physics_grid_3pcf* allocate_grid(
  const physics_grid_settings_3pcf* settings,
  const int has_residual
)
{
  grid_require(settings != NULL, "missing grid settings");
  grid_require(settings->ell_min > 0.0
    && settings->ell_max > settings->ell_min && isfinite(settings->ell_max)
    && settings->epmu > 0.0 && settings->epmu < 0.05
    && settings->psi_scale >= 0.0 && isfinite(settings->psi_scale)
    && (settings->psi_scale == 0.0
      || isfinite(0.78539816339744830962/settings->psi_scale))
    && settings->shape_stretch > 0.0 && isfinite(settings->shape_stretch)
    && tanh(settings->shape_stretch) < 1.0
    && settings->nr >= 2 && settings->nu >= 2 && settings->nv >= 2
    && settings->np >= 2 && settings->npu_log >= 2
    && settings->npu_linear >= 2 && settings->nell >= 2
    && settings->npsi_log >= 0 && settings->npsi_linear >= 2
    && settings->nmu_log >= 1 && settings->nmu_linear >= 2
    && settings->lmax >= 0 && settings->lmax < INT_MAX/4 - 1
    && settings->reference_mode >= 0 && settings->reference_mode <= 1
    && settings->table_boost >= 1 && settings->multipole_boost >= 1
    && settings->npsi >= 2 && settings->nshape >= 2
    && settings->prepare_multipoles >= 0
    && settings->prepare_multipoles <= 1
    && settings->signed_baseline >= 0 && settings->signed_baseline <= 1
    && settings->residual_scaling >= 0 && settings->residual_scaling <= 1,
    "invalid interpolation settings");
  grid_require(settings->npu_log <= (INT_MAX - settings->npu_linear - 2)/2,
    "PUV axis is too large");
  physics_grid_3pcf* grid = grid_allocate(1, sizeof(*grid));
  grid->settings = *settings;
  grid->has_residual = has_residual;
  grid->nr = settings->nr;
  grid->nu = settings->nu;
  grid->nv = settings->nv;
  grid->np = settings->np;
  grid->nell = settings->nell;
  // The header derives every bound and layout constant that follows.
  grid->shape_min = settings->reference_mode ? 1.0e-6
    : fmin(1.0e-6, 0.25*settings->ell_min/settings->ell_max);
  const double psi_min = atan2(settings->ell_min, settings->ell_max);
  const double magnitude_min = sqrt(2.0)*settings->ell_min;
  const double magnitude_max = sqrt(2.0)*settings->ell_max;
  const double mu_max = 1.0 - settings->epmu;
  const double rmin = magnitude_min*fmin(1.0/sqrt(5.0),
    sqrt(1.0 - mu_max*2.0/5.0));
  const double rmax = magnitude_max*fmax(1.0/sqrt(2.0), cos(psi_min));
  const double umin = fmin(sqrt(2.0*settings->epmu), tan(psi_min));
  grid->log_r = linear_axis(log(rmin), log(rmax), settings->nr);
  grid->log_u = linear_axis(log(umin), 0.0, settings->nu);
  grid->v = linear_axis(0.0, 1.0, settings->nv);
  grid->log_p = linear_axis(log(rmin), log(3.0*rmax), settings->np);
  grid->npu = 2*settings->npu_log + settings->npu_linear + 2;
  grid->pu = grid_allocate((size_t)grid->npu, sizeof(double));
  // The fastnc floor node of the shape fraction.
  grid->pu[0] = 1.0e-6;
  // The log block of the fastnc layout, from 1e-4 to 1e-4*300 = 3e-2,
  // and its mirror approaching the equal-split value 0.5 from below.
  for (int index=0; index<settings->npu_log; index++) {
    const double small = exp(log(1.0e-4) + log(300.0)
      *index/(settings->npu_log - 1));
    grid->pu[1 + index] = small;
    grid->pu[grid->npu - 2 - index] = 0.5 - small;
  }
  // The uniform central block of the fastnc layout.
  for (int index=0; index<settings->npu_linear; index++) {
    grid->pu[1 + settings->npu_log + index] = 0.047 + (0.453 - 0.047)
      *index/(settings->npu_linear - 1);
  }
  // The equal-split endpoint.
  grid->pu[grid->npu - 1] = 0.5;
  if (settings->reference_mode) {
    grid->shape_coordinate = grid_allocate((size_t)grid->npu, sizeof(double));
    memcpy(grid->shape_coordinate, grid->pu, (size_t)grid->npu*sizeof(double));
  } else {
    free(grid->pu);
    grid->npu = settings->nshape;
    grid->pu = grid_allocate((size_t)grid->npu, sizeof(double));
    grid->shape_coordinate = linear_axis(-1.0, 1.0, grid->npu);
    for (int index=0; index<grid->npu; index++) {
      grid->pu[index] = coordinate_to_shape(grid->shape_coordinate[index],
        grid->shape_min, settings->shape_stretch);
    }
  }
  grid->log_ell = linear_axis(log(magnitude_min), log(magnitude_max),
    settings->nell);
  // The upper literal is pi/4 to double precision, the fold point of
  // the opening-angle chart; 1e-3 radians is the fastnc join between
  // the logarithmic and linear blocks.
  double* psi = loglinear_axis(psi_min, 1.0e-3, 0.78539816339744830962,
    settings->npsi_log, settings->npsi_linear, &grid->npsi);
  grid->log_psi = grid_allocate((size_t)grid->npsi, sizeof(double));
  for (int index=0; index<grid->npsi; index++) {
    grid->log_psi[index] = log(psi[index]);
  }
  free(psi);
  if (!settings->reference_mode) {
    free(grid->log_psi);
    grid->npsi = settings->npsi;
    const double scale = settings->psi_scale;
    // pi/4 to double precision again: the production chart spans the
    // same physical angles as the reference axis above.
    const double maximum = 0.78539816339744830962;
    grid->psi_coordinate = linear_axis(
      psi_grid_coordinate(psi_min, scale),
      psi_grid_coordinate(maximum, scale), grid->npsi);
    grid->log_psi = physical_psi_axis(grid->psi_coordinate, grid->npsi,
      log(psi_min), log(maximum), scale);
  } else {
    // Reference sampling keeps its original irregular log-PSI
    // intervals, and setting psi_scale does not change its coordinate
    // definition.
    grid->psi_coordinate = grid_allocate((size_t)grid->npsi, sizeof(double));
    memcpy(grid->psi_coordinate, grid->log_psi,
      (size_t)grid->npsi*sizeof(double));
  }
  // Build 1 - mu with the fastnc joins - logarithmic from epmu up to
  // 0.05, linear from 0.05 to 2.0, the anti-parallel value - and then
  // reverse into ascending mu; the header derives this layout.
  double* reversed = loglinear_axis(settings->epmu, 0.05, 2.0,
    settings->nmu_log, settings->nmu_linear, &grid->nmu);
  grid->mu = grid_allocate((size_t)grid->nmu, sizeof(double));
  for (int index=0; index<grid->nmu; index++) {
    grid->mu[index] = 1.0 - reversed[grid->nmu - 1 - index];
  }
  free(reversed);
  const size_t nbaseline = grid_product(settings->nr, settings->nu,
    settings->nv);
  const size_t nresidual = grid_product(settings->np, grid->npu, grid->npu);
  grid->log_baseline = grid_allocate(nbaseline, sizeof(double));
  if (settings->signed_baseline) {
    grid->baseline_ratio = grid_allocate(nbaseline, sizeof(double));
  }
  if (has_residual) {
    grid->residual = grid_allocate(nresidual, sizeof(double));
    if (settings->residual_scaling) {
      grid->log_residual_scale = grid_allocate((size_t)grid->np,
        sizeof(double));
    }
  }
  return grid;
}


// Complete a grid whose coarse tables are already filled: upsample
// every table with natural cubics onto dense uniform axes, rebuild the
// axis arrays at the dense counts, and prepare the b_L multipole table.
// After this step every reader uses arithmetic cell indices and linear
// interpolation only.
static void finish_grid(physics_grid_3pcf* grid)
{
  const physics_grid_settings_3pcf* settings = &grid->settings;
  if (!settings->reference_mode) {
    const int boost = settings->table_boost;
    const int boosts[3] = {boost, boost, boost};
    int baseline_shape[3] = {grid->nr, grid->nu, grid->nv};
    int ratio_shape[3] = {grid->nr, grid->nu, grid->nv};
    int residual_shape[3] = {grid->np, grid->npu, grid->npu};
    int scale_shape[3] = {grid->np, 1, 1};
    grid->log_baseline = upsample_uniform_grid(grid->log_baseline,
      baseline_shape, boosts);
    if (settings->signed_baseline) {
      // The upsampler modifies its shape argument. Give the ratio its
      // original coarse dimensions, not the already enlarged log shape.
      grid->baseline_ratio = upsample_uniform_grid(grid->baseline_ratio,
        ratio_shape, boosts);
    }
    if (grid->has_residual) {
      grid->residual = upsample_uniform_grid(grid->residual,
        residual_shape, boosts);
      if (settings->residual_scaling) {
        // Both arrays start from their original coarse perimeter count.
        // The singleton shape axes are labels, not interpolated axes.
        const int scale_boosts[3] = {boost, 1, 1};
        grid->log_residual_scale = upsample_uniform_grid(
          grid->log_residual_scale, scale_shape, scale_boosts);
        grid_require(scale_shape[0] == residual_shape[0],
          "residual scale and ratio have different perimeter counts");
      }
    }
    double* log_r = linear_axis(grid->log_r[0], grid->log_r[grid->nr - 1],
      baseline_shape[0]);
    double* log_u = linear_axis(grid->log_u[0], grid->log_u[grid->nu - 1],
      baseline_shape[1]);
    double* v = linear_axis(0.0, 1.0, baseline_shape[2]);
    double* log_p = linear_axis(grid->log_p[0], grid->log_p[grid->np - 1],
      residual_shape[0]);
    free(grid->log_r);
    free(grid->log_u);
    free(grid->v);
    free(grid->log_p);
    free(grid->pu);
    free(grid->shape_coordinate);
    grid->log_r = log_r;
    grid->log_u = log_u;
    grid->v = v;
    grid->log_p = log_p;
    grid->nr = baseline_shape[0];
    grid->nu = baseline_shape[1];
    grid->nv = baseline_shape[2];
    grid->np = residual_shape[0];
    grid->npu = residual_shape[1];
    grid->shape_coordinate = linear_axis(-1.0, 1.0, grid->npu);
    grid->pu = grid_allocate((size_t)grid->npu, sizeof(double));
    for (int index=0; index<grid->npu; index++) {
      grid->pu[index] = coordinate_to_shape(grid->shape_coordinate[index],
        grid->shape_min, settings->shape_stretch);
    }
  }
  if (settings->prepare_multipoles) {
    prepare_multipole_grid(grid);
  }
  if (!settings->reference_mode && settings->prepare_multipoles) {
    const int boost = settings->multipole_boost;
    const int boosts[3] = {1, boost, boost};
    int shape[3] = {4*(settings->lmax + 1), grid->nell, grid->npsi};
    grid->multipoles = upsample_uniform_grid(grid->multipoles, shape, boosts);
    double* log_ell = linear_axis(grid->log_ell[0],
      grid->log_ell[grid->nell - 1], shape[1]);
    // Cubic preparation is uniform in the angular coordinate that
    // psi_scale selects. The code rebuilds the physical log-angles for
    // diagnostics only after creating the dense coordinate axis;
    // interpolating in physical log-PSI would mix two different
    // numerical definitions of the same table.
    double* psi_coordinate = linear_axis(grid->psi_coordinate[0],
      grid->psi_coordinate[grid->npsi - 1], shape[2]);
    double* log_psi = physical_psi_axis(psi_coordinate, shape[2],
      grid->log_psi[0], grid->log_psi[grid->npsi - 1], settings->psi_scale);
    free(grid->log_ell);
    free(grid->log_psi);
    free(grid->psi_coordinate);
    grid->log_ell = log_ell;
    grid->log_psi = log_psi;
    grid->psi_coordinate = psi_coordinate;
    grid->nell = shape[1];
    grid->npsi = shape[2];
  }
}


// ---------------------------------------------------------------------------
// Create one prepared grid per ordered source triplet, sharing every
// expensive line-of-sight evaluation across the batch. All triplets use
// the same cosmology, IA parameters and triangle geometry; only the
// radial source weights differ, so the batch contracts each sampled
// triangle's physical vertices with every triplet's weights at once.
//
// The coarse fill proceeds in three work-sharing loops: the sorted RUV
// baseline, the optional one-dimensional residual scale, and the PUV
// residual. finish_grid then upsamples each grid and prepares its b_L
// table. A pure-NLA parameter choice (a2 = bta = ct = 0, the TATT
// amplitudes all zero) produces no residual, and those grids skip the
// PUV storage entirely.
//
// The gravity-only companion. signed_baseline and residual_scaling
// both need a positive pure-gravity reference, so the batch builds a
// second set of projectors with every intrinsic-shape coefficient set
// to zero: that is GGG, the pure-gravity three-shear correlation.
// Cosmology, source distributions and the matter prescription stay
// identical, so each stored ratio compares two projected observables.
// The companion also holds its feedback amplitude at DMO
// (dark-matter-only, no baryon response): a changed baryon amplitude
// would otherwise change the interpolation error of the deliberately
// unchanged quadratic IA residual.
//
// Parameters:
//   model - immutable physical model, borrowed during construction
//   parameters - IA coefficients shared by every triplet
//   ntriplet - number of ordered source triplets
//   bins - [ntriplet,3], ordered source legs
//   settings - interpolation settings shared by every grid
//   out - [ntriplet], receives one owned grid per triplet
//
// Returns:
//   Nothing; the caller owns the ntriplet grids in out.
// ---------------------------------------------------------------------------
void physics_grid_create_many_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const int ntriplet,
  const int* bins,
  const physics_grid_settings_3pcf* settings,
  physics_grid_3pcf** out
)
{
  grid_require(model && parameters && bins && out && ntriplet > 0,
    "missing batch grid inputs");
  grid_require(settings != NULL, "missing batch grid settings");
  grid_require((parameters->operator_component == 0
      && parameters->intrinsic_legs == -1) || settings->signed_baseline,
    "component grids require signed_baseline=1 for common GGG scaling");
  physics_projection_3pcf** projections = grid_allocate((size_t)ntriplet,
    sizeof(*projections));
  // Pure NLA omits the residual entirely, including any scale table.
  const int has_residual = parameters->a2 != 0.0
    || parameters->bta != 0.0 || parameters->ct != 0.0;
  const int scaled_residual = settings->residual_scaling && has_residual;
  const int needs_gravity = settings->signed_baseline || scaled_residual;
  physics_projection_3pcf** gravity_projections = NULL;
  if (needs_gravity) {
    gravity_projections = grid_allocate((size_t)ntriplet,
      sizeof(*gravity_projections));
  }
  // Only the intrinsic-shape coefficients change for the denominator;
  // the header explains why this gravity-only companion stays at DMO.
  physics_ia_3pcf gravity_parameters = *parameters;
  gravity_parameters.a1 = 0.0;
  gravity_parameters.a2 = 0.0;
  gravity_parameters.bta = 0.0;
  gravity_parameters.ct = 0.0;
  gravity_parameters.operator_component = 0;
  gravity_parameters.intrinsic_legs = -1;
  gravity_parameters.feedback_amplitude = 0.0;
  for (int triplet=0; triplet<ntriplet; triplet++) {
    out[triplet] = allocate_grid(settings, has_residual);
    physics_grid_3pcf* grid = out[triplet];
    const double* window = physics_window_samples_3pcf(model, &grid->nwindow);
    if (grid->nwindow > 0) {
      grid->window = grid_allocate((size_t)grid->nwindow, sizeof(double));
      memcpy(grid->window, window, (size_t)grid->nwindow*sizeof(double));
    }
    projections[triplet] = physics_projector_create_3pcf(model, parameters,
      bins + (size_t)triplet*3);
    if (needs_gravity) {
      gravity_projections[triplet] = physics_projector_create_3pcf(model,
        &gravity_parameters, bins + (size_t)triplet*3);
    }
  }
  // Validate common model/IA once and copy radial weights into contiguous
  // triplet lanes. The batch borrows the first projector until destruction
  // below; workers neither allocate nor modify any shared preparation.
  physics_projection_batch_3pcf* batch = physics_projection_batch_create_3pcf(
    ntriplet, projections);
  physics_projection_batch_3pcf* gravity_batch = NULL;
  if (needs_gravity) {
    gravity_batch = physics_projection_batch_create_3pcf(ntriplet,
      gravity_projections);
  }
  const physics_grid_3pcf* first = out[0];
  // Every cell on one axis uses the same exponential. Prepare those
  // exact values once; retain all subsequent triangle-side operations.
  // These arrays belong to this call and remain read-only for workers.
  double* radii = grid_allocate((size_t)first->nr, sizeof(double));
  double* ratios = grid_allocate((size_t)first->nu, sizeof(double));
  for (int index=0; index<first->nr; index++) {
    radii[index] = exp(first->log_r[index]);
  }
  for (int index=0; index<first->nu; index++) {
    ratios[index] = exp(first->log_u[index]);
  }
  double* perimeters = NULL;
  if (has_residual) {
    perimeters = grid_allocate((size_t)first->np, sizeof(double));
    for (int index=0; index<first->np; index++) {
      perimeters[index] = exp(first->log_p[index]);
    }
  }
  const size_t nbaseline = grid_product(first->nr, first->nu, first->nv);
  const size_t nresidual = has_residual
    ? grid_product(first->np, first->npu, first->npu) : 0;
  const int nscale = scaled_residual ? first->np : 0;
  // Keep the original positive values only during coarse preparation.
  // All shape nodes at one perimeter divide by the same value, without
  // repeating an exponential or a line-of-sight integral for each shape.
  double* scales = scaled_residual
    ? grid_allocate(grid_product(nscale, ntriplet, 1), sizeof(double)) : NULL;
  // Every worker owns one geometry point and an ntriplet scratch
  // vector. The batch shares the physical bispectrum and IA vertices at
  // each redshift before the source-weight contraction, so memory grows
  // with the number of output tables, not with ntriplet times the
  // line-of-sight nodes times all sampled triangle geometries.
  size_t worker_stride;
  const size_t scratch_count = settings->signed_baseline
    ? 2*(size_t)ntriplet : (size_t)ntriplet;
  double* scratch = worker_storage(scratch_count, &worker_stride);
  #pragma omp parallel if(!omp_in_parallel())
  {
    double* values = scratch + (size_t)omp_get_thread_num()*worker_stride;
    double* gravity = settings->signed_baseline
      ? values + (size_t)ntriplet : NULL;
    // Fill the sorted RUV baseline. One iteration owns one (r,u,v)
    // cell: it decodes the flattened index into the three axis
    // indices, rebuilds the sorted sides by inverting the definitions
    // u = smallest/middle and v = (largest-middle)/smallest - largest
    // is r*(1 + u*v), middle is r, smallest is r*u - and evaluates the
    // batch there, so one line-of-sight pass serves every triplet's
    // table at that cell.
    #pragma omp for schedule(static)
    for (size_t index=0; index<nbaseline; index++) {
      const int iv = index%(size_t)first->nv;
      const int iu = (index/(size_t)first->nv)%(size_t)first->nu;
      const int ir = index/((size_t)first->nv*first->nu);
      const double radius = radii[ir];
      const double ratio = ratios[iu];
      // Sides ordered largest, middle, smallest; component 1 selects
      // the NLA baseline and puv_extension 0 keeps the triangle guard.
      physics_projection_batch_evaluate_3pcf(batch,
        radius*(1.0 + ratio*first->v[iv]), radius, radius*ratio, 1, 0,
        values);
      if (settings->signed_baseline) {
        // The same cell for the gravity-only denominator batch, with
        // the same component and extension arguments. Both batches were
        // prepared before workers began, share their own line-of-sight
        // physics across all source triplets, and write only their
        // separate private outputs.
        physics_projection_batch_evaluate_3pcf(gravity_batch,
          radius*(1.0 + ratio*first->v[iv]), radius, radius*ratio, 1, 0,
          gravity);
        for (int triplet=0; triplet<ntriplet; triplet++) {
          grid_require(gravity[triplet] > 0.0 && isfinite(gravity[triplet])
            && isfinite(values[triplet]),
            "signed RUV baseline requires finite NLA and positive GGG");
          const double signed_ratio = values[triplet]/gravity[triplet];
          grid_require(isfinite(signed_ratio),
            "signed RUV baseline ratio is not representable");
          out[triplet]->log_baseline[index] = log(gravity[triplet]);
          out[triplet]->baseline_ratio[index] = signed_ratio;
        }
      } else {
        for (int triplet=0; triplet<ntriplet; triplet++) {
          grid_require(values[triplet] > 0.0 && isfinite(values[triplet]),
            "RUV log-baseline requires positive B; use signed_baseline=1");
          out[triplet]->log_baseline[index] = log(values[triplet]);
        }
      }
    }
    // Tabulate the optional residual scale. One iteration owns one
    // perimeter node and evaluates the gravity-only batch on the
    // equilateral triangle with side p/3 (component 1 with
    // puv_extension 0 reads plain projected GGG there). The log of
    // each value becomes the stored scale table, and the positive
    // values stay in scales for the shape loop's division below.
    #pragma omp for schedule(static)
    for (int ip=0; ip<nscale; ip++) {
      const double side = perimeters[ip]/3.0;
      physics_projection_batch_evaluate_3pcf(gravity_batch,
        side, side, side, 1, 0, values);
      for (int triplet=0; triplet<ntriplet; triplet++) {
        grid_require(values[triplet] > 0.0 && isfinite(values[triplet]),
          "residual scaling requires finite positive equilateral GGG");
        scales[(size_t)ip*ntriplet + triplet] = values[triplet];
        out[triplet]->log_residual_scale[ip] = log(values[triplet]);
      }
    }
    // Fill the PUV residual. One iteration owns one (p,u,v) cell: the
    // ordered sides are ell1 = p*u, ell2 = p*v and the remainder
    // ell3 = p*(1 - u - v), so the three fractions sum to one by
    // construction. The preceding work-sharing barrier completes every
    // scale before this loop reads it, and workers own distinct PUV
    // cells throughout.
    #pragma omp for schedule(static)
    for (size_t index=0; index<nresidual; index++) {
      const int iv = index%(size_t)first->npu;
      const int iu = (index/(size_t)first->npu)%(size_t)first->npu;
      const int ip = index/((size_t)first->npu*first->npu);
      const double perimeter = perimeters[ip];
      const double side1 = perimeter*first->pu[iu];
      const double side2 = perimeter*first->pu[iv];
      const double side3 = perimeter*(1.0 - first->pu[iu] - first->pu[iv]);
      // component 2 selects the TATT residual; puv_extension 1 keeps
      // the source's rectangular continuation, so corner cells outside
      // the physical triangle still receive defined values for the
      // interpolation stencil.
      physics_projection_batch_evaluate_3pcf(batch,
        side1, side2, side3, 2, 1, values);
      for (int triplet=0; triplet<ntriplet; triplet++) {
        grid_require(isfinite(values[triplet]), "nonfinite PUV residual");
        const double residual = scaled_residual
          ? values[triplet]/scales[(size_t)ip*ntriplet + triplet]
          : values[triplet];
        grid_require(isfinite(residual),
          "scaled PUV residual is not representable");
        out[triplet]->residual[index] = residual;
      }
    }
  }
  free(scratch);
  free(radii);
  free(ratios);
  free(perimeters);
  free(scales);
  physics_projection_batch_destroy_3pcf(batch);
  physics_projection_batch_destroy_3pcf(gravity_batch);
  for (int triplet=0; triplet<ntriplet; triplet++) {
    physics_projector_destroy_3pcf(projections[triplet]);
    if (needs_gravity) {
      physics_projector_destroy_3pcf(gravity_projections[triplet]);
    }
    finish_grid(out[triplet]);
  }
  free(projections);
  free(gravity_projections);
}


// Create a single grid through the batched path with ntriplet = 1.
physics_grid_3pcf* physics_grid_create_3pcf(
  const physics_model_3pcf* model,
  const physics_ia_3pcf* parameters,
  const int* bins,
  const physics_grid_settings_3pcf* settings
)
{
  physics_grid_3pcf* grid;
  physics_grid_create_many_3pcf(model, parameters, 1, bins, settings, &grid);
  return grid;
}


// ============================================================================
// [SECTION] QUERY MACHINERY
// ============================================================================

// Adapt a prepared grid to the shared aperture integrator's callback:
// read the total projected B and apply the pixel window at all three
// sides, exactly as the direct reference provider does.
static double grid_aperture_reader(
  const void* context,
  const double ell1,
  const double ell2,
  const double ell3
)
{
  const physics_grid_3pcf* grid = context;
  double value = physics_grid_projected_3pcf(grid, ell1, ell2, ell3, 0);
  if (grid->nwindow > 0) {
    value *= physics_window_value_3pcf(grid->nwindow, grid->window, ell1)
      *physics_window_value_3pcf(grid->nwindow, grid->window, ell2)
      *physics_window_value_3pcf(grid->nwindow, grid->window, ell3);
  }
  return value;
}


void physics_grid_aperture_full_3pcf(
  const physics_grid_3pcf* grid,
  const int nell,
  const double* ell,
  const double* dell,
  const int nphi,
  const double* phi,
  const double* dphi,
  const int nfilter,
  const double* filters,
  double* out
)
{
  // The direct reference uses this same Fourier quadrature; only the
  // provider of B changes, and this route never touches the b_L table.
  physics_aperture_integrate_3pcf(grid, grid_aperture_reader,
    nell, ell, dell, nphi, phi, dphi, nfilter, filters, out);
}


void physics_grid_destroy_3pcf(physics_grid_3pcf* grid)
{
  if (grid == NULL) {
    return;
  }
  free(grid->log_r);
  free(grid->log_u);
  free(grid->v);
  free(grid->log_p);
  free(grid->pu);
  free(grid->shape_coordinate);
  free(grid->log_ell);
  free(grid->log_psi);
  free(grid->psi_coordinate);
  free(grid->mu);
  free(grid->log_baseline);
  free(grid->baseline_ratio);
  free(grid->residual);
  free(grid->log_residual_scale);
  free(grid->multipoles);
  free(grid->window);
  free(grid);
}


// ---------------------------------------------------------------------------
// Prepare the interpolation geometry for one fixed array of output
// Fourier multipoles. For every ordered leg pair (ell1,ell2) the
// evaluation reads the b_L table at the magnitude coordinate
// log(sqrt(ell1^2 + ell2^2)) and at the angular coordinate of the
// folded opening angle. This routine computes those transforms, the
// chart choice, the flattened lower-left cell and the four bilinear
// corner weights once, so the per-component evaluation later contains
// neither a logarithm nor a binary search nor repeated index
// arithmetic.
//
// The query owns copies of the complete interpolation axes instead of
// borrowing the grid; the header declares the matching rules that let
// one query serve many physical grids with identical geometry.
//
// Parameters:
//   grid - prepared b_L grid defining the interpolation geometry
//   nell - number of requested Fourier values on each ordered leg
//   ell - [nell], positive multipoles; copied, never kept by pointer
//
// Returns:
//   An owned query object; the caller destroys it.
// ---------------------------------------------------------------------------
physics_grid_query_3pcf* physics_grid_query_create_3pcf(
  const physics_grid_3pcf* grid, // prepared interpolation axes and L cutoff
  const int nell, // requested Fourier values per leg
  const double* ell // [nell], copied here and never kept by pointer
)
{
  grid_require(grid && ell && nell > 0, "invalid multipole query geometry");
  grid_require(grid->settings.prepare_multipoles && grid->multipoles,
    "this grid omitted b_L; recreate with prepare_multipoles=1");
  const int lmax = grid->settings.lmax;
  grid_require(lmax >= 0 && lmax < INT_MAX && grid->nell >= 2
    && grid->npsi >= 2 && grid->log_ell && grid->log_psi
    && grid->psi_coordinate,
    "invalid prepared multipole axes");
  const size_t output_count = grid_product(lmax + 1, nell, nell);
  grid_require(output_count <= SIZE_MAX/sizeof(double),
    "multipole query output size overflows");
  const size_t npoint = grid_product(1, nell, nell);
  grid_require(npoint <= SIZE_MAX/4, "multipole query weight count overflows");
  for (int index=0; index<nell; index++) {
    grid_require(isfinite(ell[index]) && ell[index] > 0.0,
      "invalid Fourier multipole");
  }
  physics_grid_query_3pcf* query = grid_allocate(1, sizeof(*query));
  query->nell = nell;
  query->ell = grid_allocate((size_t)nell, sizeof(double));
  memcpy(query->ell, ell, (size_t)nell*sizeof(double));
  query->grid_nell = grid->nell;
  query->grid_npsi = grid->npsi;
  query->lmax = lmax;
  query->reference_mode = grid->settings.reference_mode;
  query->psi_scale = grid->settings.psi_scale;
  query->log_ell = grid_allocate((size_t)grid->nell, sizeof(double));
  query->log_psi = grid_allocate((size_t)grid->npsi, sizeof(double));
  query->psi_coordinate = grid_allocate((size_t)grid->npsi, sizeof(double));
  memcpy(query->log_ell, grid->log_ell, (size_t)grid->nell*sizeof(double));
  memcpy(query->log_psi, grid->log_psi, (size_t)grid->npsi*sizeof(double));
  memcpy(query->psi_coordinate, grid->psi_coordinate,
    (size_t)grid->npsi*sizeof(double));
  query->charts = grid_allocate(npoint, sizeof(int));
  query->cells = grid_allocate(npoint, sizeof(size_t));
  query->fractions = grid_allocate(4*npoint, sizeof(double));
  int* charts = query->charts;
  size_t* cells = query->cells;
  double* fractions = query->fractions;
  // Every Legendre degree and both physical components share this
  // geometry, so the loop computes it once. One iteration owns one
  // ordered pair (first,second) and writes only that pair's chart,
  // cell and four weights; sharing the read-only axes across workers
  // changes neither the coordinates nor their rounding.
  #pragma omp parallel for collapse(2) if(!omp_in_parallel()) schedule(static)
  for (int first=0; first<nell; first++) {
    for (int second=0; second<nell; second++) {
      // The file head's two charts split at the fold: chart 1 is the
      // PSI = pi/2 - small_angle chart, which an ordered pair enters
      // exactly when its second leg exceeds its first.
      const int chart = ell[second] > ell[first] ? 1 : 0;
      const double small = atan2(fmin(ell[first], ell[second]),
        fmax(ell[first], ell[second]));
      const double x = corrected_coordinate(log(hypot(ell[first], ell[second])),
        grid->log_ell[0], grid->log_ell[grid->nell - 1],
        grid->settings.reference_mode);
      const double psi_scale = grid->settings.reference_mode
        ? 0.0 : grid->settings.psi_scale;
      const double y = corrected_coordinate(
        psi_grid_coordinate(small, psi_scale), grid->psi_coordinate[0],
        grid->psi_coordinate[grid->npsi - 1], grid->settings.reference_mode);
      const int ix = grid_interval(grid->log_ell, grid->nell, x,
        !grid->settings.reference_mode);
      const int iy = grid_interval(grid->psi_coordinate, grid->npsi, y,
        !grid->settings.reference_mode);
      const double fraction_x = (x - grid->log_ell[ix])
        /(grid->log_ell[ix + 1] - grid->log_ell[ix]);
      const double fraction_y = (y - grid->psi_coordinate[iy])
        /(grid->psi_coordinate[iy + 1] - grid->psi_coordinate[iy]);
      const size_t point = (size_t)first*nell + second;
      charts[point] = chart;
      cells[point] = (size_t)ix*grid->npsi + iy;
      fractions[point*4] = (1.0 - fraction_x)*(1.0 - fraction_y);
      fractions[point*4 + 1] = (1.0 - fraction_x)*fraction_y;
      fractions[point*4 + 2] = fraction_x*(1.0 - fraction_y);
      fractions[point*4 + 3] = fraction_x*fraction_y;
    }
  }
  return query;
}


int physics_grid_query_matches_3pcf(
  const physics_grid_query_3pcf* query, // complete immutable geometry key
  const physics_grid_3pcf* grid // physical values are deliberately not a key
)
{
  if (!query || !grid || !grid->settings.prepare_multipoles
      || !grid->multipoles || !grid->log_ell || !grid->log_psi
      || !grid->psi_coordinate
      || query->grid_nell != grid->nell || query->grid_npsi != grid->npsi
      || query->lmax != grid->settings.lmax
      || query->reference_mode != grid->settings.reference_mode
      || query->psi_scale != grid->settings.psi_scale) {
    return 0;
  }
  // The constructor checked these sizes before copying the arrays.
  // Exact byte comparison includes every interior axis value and
  // distinguishes signed zero. The query's own ell coordinates cannot
  // change after construction.
  return memcmp(query->log_ell, grid->log_ell,
      (size_t)query->grid_nell*sizeof(double)) == 0
    && memcmp(query->log_psi, grid->log_psi,
      (size_t)query->grid_npsi*sizeof(double)) == 0
    && memcmp(query->psi_coordinate, grid->psi_coordinate,
      (size_t)query->grid_npsi*sizeof(double)) == 0;
}


int physics_grid_query_nell_3pcf(
  const physics_grid_query_3pcf* query // fixed output Fourier size
)
{
  grid_require(query != NULL, "missing multipole query geometry");
  return query->nell;
}


int physics_grid_query_lmax_3pcf(
  const physics_grid_query_3pcf* query // fixed maximum Legendre degree
)
{
  grid_require(query != NULL, "missing multipole query geometry");
  return query->lmax;
}


// ---------------------------------------------------------------------------
// Evaluate the prepared b_L interpolation for one component on a
// matching physical grid. Each output element is a four-corner
// bilinear read of one stored multipole plane,
//
//   out[L,i,j] = sum over corners of weight[i,j,corner]
//                * b_L(chart[i,j], cell[i,j] + corner offset),
//
// summed over the baseline and residual parts that the component
// selects; the total adds both. The geometry - charts, cells, weights
// - comes from the query, and the physical values come from the grid,
// whose axes must match the query's exactly.
//
// The pixel window splits across the stages: the stored table already
// carries W3, the window at the closing side, inside the mu integral,
// and the leg windows W1 and W2 multiply the output after
// interpolation, so the interpolation stencil cannot leak smoothed
// values across the window's hard upper cutoff.
//
// Parameters:
//   query - immutable interpolation geometry for one fixed ell array
//   grid - matching axes, current physical multipoles
//   component - total=0, NLA baseline=1, residual IA=2
//   out - [lmax+1,nell,nell]; must not alias grid or query storage
//
// Returns:
//   Nothing; out holds b_L at every ordered (ell1,ell2) pair.
// ---------------------------------------------------------------------------
void physics_grid_query_evaluate_3pcf(
  const physics_grid_query_3pcf* query, // immutable geometric interpolation map
  const physics_grid_3pcf* grid, // matching axes, fresh physical multipoles
  const int component, // total=0, NLA baseline=1, residual IA=2
  double* out // [lmax+1,nell,nell], independent writable output
)
{
  grid_require(query && grid && out && component >= 0 && component <= 2,
    "invalid prepared multipole query");
  grid_require(physics_grid_query_matches_3pcf(query, grid),
    "multipole query axes differ; recreate the query for this grid");
  const int nell = query->nell;
  const int lmax = query->lmax;
  const int* charts = query->charts;
  const size_t* cells = query->cells;
  const double* fractions = query->fractions;
  const size_t plane = (size_t)grid->nell*grid->npsi;
  const size_t offsets[4] = {0, 1, (size_t)grid->npsi,
    (size_t)grid->npsi + 1};
  // One iteration owns one (L,ell1) output row and walks the ell2
  // entries two at a time. The vector lanes hold adjacent ell2 outputs
  // of that same row: both lanes run the identical four-corner
  // bilinear sum, each with its own prepared cell and weights, and the
  // lanes never interact until the paired results store into the
  // contiguous row. Workers write disjoint rows, so the loop shares
  // nothing.
  #pragma omp parallel for collapse(2) if(!omp_in_parallel()) schedule(static)
  for (int order=0; order<=lmax; order++) {
    for (int first=0; first<nell; first++) {
      double* restrict output = out + ((size_t)order*nell + first)*nell;
      const double* restrict tables = grid->multipoles;
      int second = 0;
      // Scalar form of the vector loop below, for one pair index p
      // (one ordered (ell1,ell2) pair):
      //
      //   double sum = 0.0;
      //   for (each part the component selects)
      //     for (int corner=0; corner<4; corner++)
      //       sum += fractions[p*4 + corner]
      //               *tables[base(part,p) + offsets[corner]];
      //   output[second] = sum;
      //
      // The vector loop evaluates this for pairs p and p + 1 at once:
      // lane 0 (low) accumulates the output at ell2[second] and lane 1
      // (high) the output at ell2[second+1].
      for (; second + 1<nell; second += 2) {
        const size_t point = (size_t)first*nell + second;
        // Set both running sums to zero; each lane accumulates its own
        // output element.
        grid_v2d sum = simde_mm_setzero_pd();
        for (int part=0; part<2; part++) {
          if ((component == 1 && part == 1) || (component == 2 && part == 0)) {
            continue;
          }
          const size_t base0 = ((size_t)(part*2 + charts[point])*(lmax + 1)
            + order)*plane + cells[point];
          const size_t base1 = ((size_t)(part*2 + charts[point + 1])*(lmax + 1)
            + order)*plane + cells[point + 1];
          for (int corner=0; corner<4; corner++) {
            // Gather the same corner for the two prepared cells.
            // simde_mm_set_pd(high, low) places its first argument in
            // the high lane: the high lane holds the ell2[second+1]
            // cell's value tables[base1 + ...] and the low lane the
            // ell2[second] cell's value tables[base0 + ...].
            const grid_v2d values = simde_mm_set_pd(
              tables[base1 + offsets[corner]], tables[base0 + offsets[corner]]);
            // The matching corner weights in the same lane order: the
            // high lane takes fractions[(point+1)*4 + corner] and the
            // low lane fractions[point*4 + corner].
            const grid_v2d weights = simde_mm_set_pd(
              fractions[(point + 1)*4 + corner], fractions[point*4 + corner]);
            // Multiply lane by lane: one corner's bilinear
            // contribution, weight times table value, for each output.
            const grid_v2d term = simde_mm_mul_pd(weights, values);
            // Add into the running sums; both lanes keep the same
            // part-then-corner addition order as the scalar tail.
            sum = simde_mm_add_pd(sum, term);
          }
        }
        // Store the low lane to output[second] and the high lane to
        // output[second + 1], contiguous in the C-order row; storeu
        // needs no vector-aligned address.
        simde_mm_storeu_pd(output + second, sum);
      }
      if (second < nell) {
        const size_t point = (size_t)first*nell + second;
        double sum = 0.0;
        for (int part=0; part<2; part++) {
          if ((component == 1 && part == 1) || (component == 2 && part == 0)) {
            continue;
          }
          const size_t base = ((size_t)(part*2 + charts[point])*(lmax + 1)
            + order)*plane + cells[point];
          for (int corner=0; corner<4; corner++) {
            // Force the separate rounding of the vector path: volatile
            // keeps the product and the addition separately rounded,
            // so the compiler cannot contract them into a fused
            // multiply-add - strict flags alone do not forbid that.
            // Without this, a query with one more ell value would
            // change earlier outputs when a former scalar tail becomes
            // a vector lane.
            volatile double term =
              fractions[point*4 + corner]*tables[base + offsets[corner]];
            sum += term;
          }
        }
        output[second] = sum;
      }
      if (grid->nwindow > 0) {
        // The leg windows W1 and W2 sit outside the mu integral, so
        // they multiply the output only after the b_L interpolation;
        // the header explains why the stencil must not cross the
        // window's hard upper cutoff. The table already carries W3.
        const double first_window = physics_window_value_3pcf(grid->nwindow,
          grid->window, query->ell[first]);
        for (int index=0; index<nell; index++) {
          output[index] *= first_window*physics_window_value_3pcf(grid->nwindow,
            grid->window, query->ell[index]);
        }
      }
    }
  }
}


void physics_grid_query_destroy_3pcf(
  physics_grid_query_3pcf* query // owned geometry only, never a physical grid
)
{
  if (query == NULL) {
    return;
  }
  free(query->log_ell);
  free(query->log_psi);
  free(query->psi_coordinate);
  free(query->charts);
  free(query->cells);
  free(query->fractions);
  free(query->ell);
  free(query);
}


// A one-shot convenience: build the query, evaluate it and destroy it.
// A repeated caller owns one query object instead and reuses it across
// physical grids; evaluate runs the same exact matching check either
// way.
void physics_grid_multipoles_3pcf(const physics_grid_3pcf* grid,
  const int nell, const double* ell, const int component, double* out)
{
  grid_require(out && component >= 0 && component <= 2,
    "invalid multipole table output or component");
  physics_grid_query_3pcf* query = physics_grid_query_create_3pcf(grid,
    nell, ell);
  physics_grid_query_evaluate_3pcf(query, grid, component, out);
  physics_grid_query_destroy_3pcf(query);
}


void physics_grid_axis_3pcf(const physics_grid_3pcf* grid,
  const int axis, int* count, const double** values)
{
  grid_require(grid && axis >= 0 && axis <= 7 && count && values,
    "invalid grid-axis query");
  const int counts[8] = {grid->nr, grid->nu, grid->nv, grid->np, grid->npu,
    grid->nell, grid->npsi, grid->nmu};
  const double* axes[8] = {grid->log_r, grid->log_u, grid->v, grid->log_p,
    grid->pu, grid->log_ell, grid->log_psi, grid->mu};
  *count = counts[axis];
  *values = axes[axis];
}
