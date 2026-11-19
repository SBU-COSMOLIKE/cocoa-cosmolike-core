#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "feedback_3pcf.h"
#include "baseline_3pcf.h"
#include "cosmolike/basics.h"
#include "log.c/src/log.h"

// Baryonic feedback - gas cooling and the energetic outflows from
// active galactic nuclei and supernovae - moves matter around inside
// halos, so a hydrodynamical simulation predicts slightly different
// clustering than a dark-matter-only (DMO) simulation of the same
// cosmology. This file stores that difference as two response ratios:
// a power spectrum response R_P = P_hydro/P_DMO and a bispectrum
// response R_B = B_hydro/B_DMO. The two responses constrain different
// moments of the matter redistribution, so neither determines the
// other, and the tables keep both as natural logarithms.
//
// A single amplitude A scales both responses through
// R_X(A) = exp(A ln R_X): A = 0 reproduces the DMO prediction and
// A = 1 reproduces the response pair the caller loaded. Interpolating
// in A between those anchors is a model assumption, not a new
// simulation calibration. The intrinsic alignment (IA) residual terms
// never read these tables; only their nonlinear-alignment baseline
// receives the feedback ratio.
//
// Preparation splits its independent spline lines across OpenMP
// workers and ends in one immutable response object. After creation,
// worker threads read uniform arrays through
// arithmetic index calculations: no GSL calls, no allocations and no
// mutable state remain on the query path. Storage is flat and
// unpadded, in the row order the header documents.
struct feedback_3pcf {
  int count[4]; // dense node counts on the z, log(k)=log(r), log(u), v axes
  double lower[4]; // first coordinate on each axis
  double upper[4]; // final coordinate on each axis
  double inverse[4]; // one over the dense grid spacing on each axis
  double* power; // dense ln R_P [nz, nk], last index contiguous
  double* bispectrum; // dense ln R_B [nz, nk, nu, nv], last contiguous
  int clamp; // 0 refuses out-of-domain queries; 1 holds boundary values
};


// Multiply the node counts of the leading axes into one flat element
// count. The check inside the loop proves, one factor at a time, that
// the running product still fits the int element count the core
// allocator malloc1d takes, so no later allocation can silently
// truncate or wrap around.
//
// Parameters:
//   counts     - node count per axis [dimensions], each positive
//   dimensions - number of axes to multiply (2 or 4 in this file)
// Returns: the product of the counts, as a flat element count.
static size_t table_size(
  const int* counts, // node count per axis [dimensions], each positive
  const int dimensions // number of axes to multiply (2 or 4 here)
)
{
  size_t size = 1;

  for (int axis=0; axis<dimensions; axis++) {
    if (!(counts[axis] > 0
          && size <= (size_t)INT_MAX/(size_t)counts[axis])) {
      log_fatal("3pcf logarithmic table: dimensions overflow the core "
        "allocator count");
      exit(1);
    }
    size *= (size_t)counts[axis];
  }

  return size;
}


// ---------------------------------------------------------------------------
// Resample a logarithmic table onto a grid boost times denser.
//
// The query path uses multilinear interpolation, because linear
// weights are cheap and branch-free inside parallel workers. Linear
// interpolation on the loaded grid alone would lose the curvature of
// the tabulated function between nodes. This preparation recovers that
// curvature once, at creation: it runs a natural cubic spline through the
// loaded nodes and evaluates it on a grid with boost intervals per
// loaded interval, so the later linear reads follow the cubic shape.
//
// The resampling is a tensor product, taken one axis at a time. A
// line is the set of table entries that share every coordinate except
// the one on the current axis; cubic interpolation along each such
// independent line, axis after axis, defines the tensor-product
// result.
//
// Parameters:
//   source     - loaded logarithmic table, flat, last axis contiguous
//   counts     - loaded node count per axis [dimensions]
//   axes       - loaded coordinate array of each axis [dimensions]
//   dimensions - 2 for the power response, 4 for the bispectrum
//   boost      - dense intervals per loaded interval, at least 1
// Returns: a newly allocated dense table that the caller owns and
// frees; boost = 1 returns a finiteness-checked copy of the source.
// ---------------------------------------------------------------------------
static double* dense_table(
  const double* source, // logarithmic values, flat, last axis contiguous
  const int* counts, // loaded node count per axis [dimensions]
  const double* const* axes, // loaded coordinates of each axis [dimensions]
  const int dimensions, // 2 for the power response, 4 for bispectrum
  const int boost // dense intervals per loaded interval, >= 1
)
{
  int current[4];
  memcpy(current, counts, sizeof(current));
  const size_t size = table_size(current, dimensions);
  double* table = malloc1d((int)size);

  for (size_t index=0; index<size; index++) {
    if (!(isfinite(source[index]))) {
      log_fatal("3pcf logarithmic table: nonfinite input value");
      exit(1);
    }
    table[index] = source[index];
  }

  if (1 == boost) {
    return table;
  }

  // Loop overview: each pass densifies one axis while every other axis
  // keeps its current count. To reach every line of the current axis,
  // the pass views the flat table as [blocks][old_count][stride]:
  // stride is the product of the counts of all later (faster-varying)
  // axes, so the entries of one line sit stride elements apart, and
  // blocks is the product of the counts of all earlier axes, so each
  // block interleaves stride lines. The two inner loops enumerate
  // every (block, inner) pair, gather that line into a contiguous
  // buffer, spline it, and scatter the dense result into the new
  // table. The pass grows the axis count to boost*(old_count - 1) + 1,
  // and the next pass reads the denser table as its input.
  //
  // Threading: inside one pass, every (block, inner) line reads only
  // the finished input table and writes only its own dense entries, so
  // the lines are shared among OpenMP workers with no locks. The passes
  // themselves depend on each other (pass a+1 reads what pass a wrote),
  // so each pass has its own parallel region, and the region's closing
  // barrier finishes every line before the table/dense swap publishes
  // the result. Each line runs the same serial arithmetic whichever
  // worker takes it, so the tables are bitwise independent of the
  // thread count.
  for (int axis=0; axis<dimensions; axis++) {
    const int old_count = current[axis];
    const int new_count = boost*(old_count - 1) + 1;
    size_t stride = 1;
    for (int other=axis + 1; other<dimensions; other++) {
      stride *= (size_t)current[other];
    }
    const size_t old_size = table_size(current, dimensions);
    const size_t blocks = old_size/((size_t)old_count*stride);
    current[axis] = new_count;

    double* dense = malloc1d((int)table_size(current, dimensions));

    // The spline reads this axis as the ideal uniform grid between its
    // endpoints, so one spacing h serves every interval of every line.
    const double h = (axes[axis][old_count - 1] - axes[axis][0])
      /(old_count - 1);

    #pragma omp parallel
    {
      // One scratch set per worker, private to this region: the
      // gathered line and its spline coefficients.
      double* line = malloc1d(old_count);
      double* c = malloc1d(old_count); // c_q = S''(x_q)/2 at each loaded node
      double* b = malloc1d(old_count - 1); // linear coefficient per interval
      double* d = malloc1d(old_count - 1); // cubic coefficient per interval
      const double* restrict input = table; // finished previous pass
      double* restrict output = dense; // this pass, disjoint lines

      // collapse(2) with a static schedule hands each worker one
      // contiguous run of (block, inner) pairs. Neighbouring inner lines
      // interleave in memory, so contiguous runs keep most cache lines
      // with a single writer.
      #pragma omp for collapse(2) schedule(static)
      for (size_t block=0; block<blocks; block++) {
        for (size_t inner=0; inner<stride; inner++) {
          for (int node=0; node<old_count; node++) {
            line[node] = input[(block*old_count + node)*stride + inner];
          }

          // Natural cubic spline through this line, in the house form of
          // spline2d_upsample_uniform (basics.c). The uniform solver
          // returns c_q, half the curvature at loaded node q. On interval
          // q the cubic is S(x_q + del) = y_q + b_q del + c_q del^2
          // + d_q del^3: b_q makes it reach y_{q+1} at the right node, and
          // d_q makes the curvature change linearly from 2 c_q to 2 c_{q+1}.
          spline_coeffs_uniform(line, old_count, h, c);
          for (int q=0; q<old_count - 1; q++) {
            b[q] = (line[q + 1] - line[q])/h - h*(c[q + 1] + 2.0*c[q])/3.0;
            d[q] = (c[q + 1] - c[q])/(3.0*h);
          }

          for (int node=0; node<new_count; node++) {
            // A dense node that falls on a loaded node copies that value
            // exactly, endpoints included: densifying never moves the
            // loaded data.
            double value;
            if (0 == node%boost) {
              value = line[node/boost];
            } else {
              // Each loaded interval holds boost dense steps, so this
              // node's interval is the integer quotient node/boost and its
              // offset inside that interval is (node % boost) h/boost:
              // arithmetic only, no search.
              const int q = node/boost;
              const double del = (node%boost)*h/boost;
              value = line[q] + del*(b[q] + del*(c[q] + del*d[q])); // Horner
            }
            if (!(isfinite(value))) {
              log_fatal("3pcf logarithmic table: nonfinite prepared value");
              exit(1);
            }
            output[(block*new_count + node)*stride + inner] = value;
          }
        }
      }

      free(d);
      free(b);
      free(c);
      free(line);
    }

    free(table);
    table = dense;
  }

  return table;
}


// Validate the uniform coordinates and prepare arithmetic index constants.
// Both physical baseline tables and response ratios use this grid machinery;
// their statistical meanings and their owners remain separate.
static void prepare_axes(
  const int* counts, // source counts [4]
  const double* const* axes, // source coordinate vectors [4]
  const int boost, // dense intervals per original interval
  int* dense_count, // output dense counts [4]
  double* lower, // output first coordinates [4]
  double* upper, // output last coordinates [4]
  double* inverse // output inverse dense spacings [4]
)
{
  for (int axis=0; axis<4; axis++) {
    const int count = counts[axis];
    if (!(count >= 4
          && axes[axis]
          && count - 1 <= (INT_MAX - 1)/boost)) {
      log_fatal("3pcf logarithmic table: each axis needs four nodes and "
        "representable dense size");
      exit(1);
    }

    const double low = axes[axis][0];
    const double high = axes[axis][count - 1];
    if (!(isfinite(low)
          && isfinite(high)
          && high > low)) {
      log_fatal("3pcf logarithmic table: coordinates must increase");
      exit(1);
    }

    const double step = (high - low)/(count - 1);
    if (!(isfinite(step)
          && step > 0.0)) {
      log_fatal("3pcf logarithmic table: spacing is not representable");
      exit(1);
    }

    // Purpose of this uniformity check: the query path converts a
    // coordinate to a grid index by multiplying with one stored
    // inverse spacing, and that shortcut is correct only on a uniform
    // axis. A file written with rounded decimals is uniform only to
    // roundoff, so the comparison allows a slack of 64 machine
    // epsilons scaled by the largest axis magnitude: tens of
    // last-digit rounding errors pass, while a genuinely nonuniform
    // axis, whose node offsets are comparable to the spacing itself,
    // still fails clearly.
    //
    // An accepted axis is then read as the ideal uniform grid its
    // endpoints define: the dense preparation never sees the roundoff
    // offsets of the individual loaded coordinates.
    for (int node=0; node<count; node++) {
      const double expected = low + node*step;
      if (!(isfinite(axes[axis][node])
            && (0 == node || axes[axis][node] > axes[axis][node - 1])
            && fabs(axes[axis][node] - expected)
              <= 64.0*DBL_EPSILON*fmax(1.0, fmax(fabs(low), fabs(high))))) {
        log_fatal("3pcf logarithmic table: coordinates must be uniform");
        exit(1);
      }
    }

    dense_count[axis] = boost*(count - 1) + 1;
    lower[axis] = low;
    upper[axis] = high;
    inverse[axis] = (dense_count[axis] - 1)/(high - low);
    if (!(isfinite(inverse[axis])
          && inverse[axis] > 0.0)) {
      log_fatal("3pcf logarithmic table: dense inverse spacing is not "
        "representable");
      exit(1);
    }
  }

}


// ---------------------------------------------------------------------------
// Build the immutable response object from the caller's loaded tables.
//
// Creation validates everything the query path later takes for
// granted: both log-response tables exist, every axis carries at
// least four strictly increasing uniform nodes (a cubic spline needs
// four), and the z, log(u) and v axes describe physical sorted
// triangles (z >= 0, u <= 1 so log(u) <= 0, and 0 <= v <= 1). It then
// densifies both tables with dense_table and stores the per-axis
// constants (dense count, first and last coordinate, inverse spacing)
// that turn each later query into pure arithmetic.
//
// Parameters:
//   input - description of the loaded tables; feedback_3pcf.h defines
//           every field. This call borrows the arrays and never keeps
//           them.
// Returns: the response object for the query functions and finally
// for feedback_destroy_3pcf. A NULL input returns NULL, so a caller
// without feedback tables passes the NULL straight through; queries
// on a NULL response then require amplitude zero.
// ---------------------------------------------------------------------------
feedback_3pcf* feedback_create_3pcf(
  const feedback_input_3pcf* input // borrowed loaded tables; NULL returns NULL
)
{
  if (NULL == input) {
    return NULL;
  }
  if (!(input->boost >= 1
        && input->clamp >= 0
        && input->clamp <= 1
        && input->log_power
        && input->log_bispectrum)) {
    log_fatal("feedback_3pcf: both response tables, boost>=1 and clamp=0/1 "
      "are required");
    exit(1);
  }

  const int counts[4] = {input->nz, input->nk, input->nu, input->nv};
  const double* axes[4] = {input->z, input->logk, input->logu, input->v};

  feedback_3pcf* response = calloc1d_generic(1, sizeof(*response));
  response->clamp = input->clamp;

  prepare_axes(counts, axes, input->boost, response->count,
    response->lower, response->upper, response->inverse);

  if (!(input->z[0] >= 0.0
        && input->logu[input->nu - 1] <= 0.0
        && input->v[0] >= 0.0
        && input->v[input->nv - 1] <= 1.0)) {
    log_fatal("feedback_3pcf: response z/u/v axes must describe physical "
      "sorted triangles");
    exit(1);
  }

  response->power = dense_table(input->log_power, counts, axes, 2,
    input->boost);
  response->bispectrum = dense_table(input->log_bispectrum, counts, axes, 4,
    input->boost);

  return response;
}


// Release the dense tables and the response object itself. A NULL
// response does nothing, matching the NULL that feedback_create_3pcf
// returns for a caller without feedback tables.
void feedback_destroy_3pcf(
  feedback_3pcf* response // owned response object; NULL does nothing
)
{
  if (NULL != response) {
    free(response->power);
    free(response->bispectrum);
    free(response);
  }
}


// Read one logarithmic value from a prepared uniform table. The corner
// order is shared by baseline B and paired responses and retains the
// existing feedback arithmetic exactly.
static double read_log_table(
  const int* count, // prepared counts [dimensions]
  const double* lower, // first coordinates [dimensions]
  const double* upper, // last coordinates [dimensions]
  const double* inverse, // inverse spacings [dimensions]
  const int clamp, // hold boundaries only when explicitly requested
  const double* table, // flat logarithmic table, last axis contiguous
  const double* coordinates, // query coordinates [dimensions]
  const int dimensions // two for power, four for bispectrum
)
{
  int cells[4];
  double fractions[4];

  for (int axis=0; axis<dimensions; axis++) {
    const double low = lower[axis];
    const double high = upper[axis];
    const double value = coordinates[axis];

    // Purpose of this domain tolerance: a caller often queries exactly
    // at a table edge, and the coordinate it computed - a log(k), for
    // example - can land a few rounding errors outside the stored
    // boundary. With clamp = 0 the check accepts a coordinate within
    // 64 machine epsilons (scaled by its magnitude) of the boundary
    // and refuses a query genuinely outside the domain. With
    // clamp = 1 every finite query passes, and the fmin/fmax below
    // hold the boundary value.
    const double tolerance = 64.0*DBL_EPSILON*fmax(1.0, fabs(value));
    if (!(isfinite(value)
          && (clamp
              || (value >= low - tolerance && value <= high + tolerance)))) {
      log_fatal("3pcf logarithmic table: query exceeds its declared table "
        "domain");
      exit(1);
    }

    const double position = (fmax(low, fmin(high, value)) - low)
      *inverse[axis];
    cells[axis] = (int)fmin(count[axis] - 2, floor(position));
    fractions[axis] = position - cells[axis];
  }

  double logarithm = 0.0;

  // A 2D cell has four corners and a 4D cell has sixteen. Products of
  // one-dimensional linear weights sum to one and preserve constant ratios.
  // Loop overview: multilinear interpolation averages the values at
  // the 2^dimensions corners of the grid cell around the query point.
  // The counter corner enumerates those corners through its binary
  // digits: bit number axis of corner says whether this corner sits
  // at the lower (0) or upper (1) node of that axis, and the
  // expression (corner >> axis)&1 extracts that digit. The corner's
  // flat index accumulates digit by digit in the table's row order,
  // and its weight is the product of the per-axis linear weights: the
  // fraction for an upper node, one minus the fraction for a lower
  // node.
  for (int corner=0; corner<(1 << dimensions); corner++) {
    size_t index = 0;
    double weight = 1.0;
    for (int axis=0; axis<dimensions; axis++) {
      const int upper = (corner >> axis)&1;
      index = index*count[axis] + cells[axis] + upper;
      weight *= upper ? fractions[axis] : 1.0 - fractions[axis];
    }
    logarithm += weight*table[index];
  }

  return logarithm;
}


// ---------------------------------------------------------------------------
// Evaluate one response ratio by multilinear interpolation.
//
// The stored tables hold ln R on a uniform dense grid. A query maps
// each coordinate to a fractional grid position, interpolates ln R
// linearly along every axis between the surrounding grid nodes,
// scales the result by the amplitude and exponentiates. Interpolating
// the logarithm keeps the rebuilt ratio positive, and exp(A ln R)
// makes the amplitude A act as a power of the stored ratio.
//
// Parameters:
//   response    - the prepared tables; needed whenever amplitude != 0
//   coordinates - query point, one value per axis [dimensions]
//   dimensions  - 2 selects the power table, 4 the bispectrum table
//   amplitude   - the feedback amplitude A; A = 0 returns exactly 1
// Returns: the multiplicative ratio exp(A ln R) that the caller
// applies to its DMO prediction; finite and positive, or the program
// stops.
// ---------------------------------------------------------------------------
static double read_response(
  const feedback_3pcf* response, // prepared tables; needed if amplitude != 0
  const double* coordinates, // query point, one value per axis [dimensions]
  const int dimensions, // 2 selects the power table, 4 the bispectrum
  const double amplitude // feedback amplitude A; A = 0 returns exactly 1
)
{
  if (!(isfinite(amplitude))) {
    log_fatal("feedback_3pcf: feedback amplitude must be finite");
    exit(1);
  }
  if (0.0 == amplitude) {
    return 1.0;
  }
  if (!(NULL != response)) {
    log_fatal("feedback_3pcf: nonzero feedback needs paired tables");
    exit(1);
  }

  const double* table = 2 == dimensions
    ? response->power : response->bispectrum;
  const double logarithm = read_log_table(response->count, response->lower,
    response->upper, response->inverse, response->clamp, table, coordinates,
    dimensions);

  const double result = exp(amplitude*logarithm);
  if (!(isfinite(result)
        && result > 0.0)) {
    log_fatal("feedback_3pcf: feedback amplitude produces an "
      "unrepresentable positive ratio");
    exit(1);
  }

  return result;
}


// ---------------------------------------------------------------------------
// Power spectrum feedback ratio at one wavenumber and redshift.
//
// Parameters:
//   response  - prepared tables; NULL only works at amplitude zero
//   k         - comoving wavenumber in h/Mpc; the table axis is log(k)
//   z         - redshift
//   amplitude - the feedback amplitude A; A = 0 returns exactly 1
// Returns: the multiplicative hydro-to-DMO power ratio exp(A ln R_P)
// that the caller applies to its DMO power spectrum.
// ---------------------------------------------------------------------------
double feedback_power_3pcf(
  const feedback_3pcf* response, // prepared tables; NULL only when A = 0
  const double k, // comoving wavenumber in h/Mpc
  const double z, // redshift
  const double amplitude // feedback amplitude A; 0 returns 1
)
{
  const double coordinates[2] = {z, log(k)};
  return read_response(response, coordinates, 2, amplitude);
}


// ---------------------------------------------------------------------------
// Bispectrum feedback ratio for one triangle of wavenumbers.
//
// The table parametrizes the triangle shape as the header documents:
// with the sides sorted a <= b <= c, the coordinates are the middle
// side r = b, the ratio u = a/b, and the folding parameter
// v = (c - b)/a, which is 0 when the two largest sides are equal and
// 1 for a flattened triangle with c = a + b. The code below extracts
// those sorted sides from the input, which may arrive in any order.
//
// Parameters:
//   response  - prepared tables; NULL only works at amplitude zero
//   sides     - the three triangle wavenumbers in h/Mpc [3], any order
//   z         - redshift
//   amplitude - the feedback amplitude A; A = 0 returns exactly 1
// Returns: the multiplicative hydro-to-DMO bispectrum ratio
// exp(A ln R_B) that the caller applies to its DMO bispectrum.
// ---------------------------------------------------------------------------
double feedback_bispectrum_3pcf(
  const feedback_3pcf* response, // prepared tables; NULL only when A = 0
  const double* sides, // triangle wavenumbers in h/Mpc [3]
  const double z, // redshift
  const double amplitude // feedback amplitude A; 0 returns 1
)
{
  // Amplitude zero needs no tables and no triangle sorting: the
  // hydro-to-DMO ratio is exactly 1.
  if (0.0 == amplitude) {
    return 1.0;
  }

  const double small = fmin(sides[0], fmin(sides[1], sides[2]));
  const double large = fmax(sides[0], fmax(sides[1], sides[2]));

  // The middle side comes from comparisons alone. The algebraic
  // shortcut - subtracting the two extremes from the sum of all three
  // sides - cancels large numbers in a very squeezed triangle and
  // loses the middle side to roundoff; comparisons cancel nothing.
  const double middle = fmax(fmin(sides[0], sides[1]),
    fmin(fmax(sides[0], sides[1]), sides[2]));

  // Purpose of this clip: a valid physical triangle that is nearly
  // degenerate can round to a v a few machine epsilons above 1. The
  // fmin keeps the shape coordinate inside its mathematical [0, 1]
  // range and absorbs only that roundoff.
  const double v = fmin(1.0, (large - middle)/small);

  const double coordinates[4] = {z, log(middle), log(small/middle),
    v};
  return read_response(response, coordinates, 4, amplitude);
}


// A supplied baseline is B itself, in (Mpc/h)^6, rather than B_hydro/B_DMO.
// It has no amplitude or power-response partner. The owner copies the input
// and shares only interpolation machinery with feedback objects above.
struct baseline_3pcf {
  int count[4]; // prepared z, log(k), log(u), v counts
  double lower[4]; // first coordinate on each axis
  double upper[4]; // last coordinate on each axis
  double inverse[4]; // inverse dense spacings
  double* logarithm; // ln B [nz,nk,nu,nv], v varies fastest
  int clamp; // explicit boundary holding policy
};

void baseline_validate_3pcf(
  const baseline_input_3pcf* input // required borrowed table
)
{
  if (!(NULL != input)) { log_fatal("baseline_3pcf: missing table"); exit(1); }
  if (!(1 <= input->boost)) { log_fatal(
    "baseline_3pcf: boost must be >=1"); exit(1); }
  if (!(0 == input->clamp
        || 1 == input->clamp)) { log_fatal(
    "baseline_3pcf: clamp must be 0 or 1"); exit(1); }
  if (!(NULL != input->log_bispectrum)) { log_fatal(
    "baseline_3pcf: missing ln B table"); exit(1); }
  const int counts[4] = {input->nz, input->nk, input->nu, input->nv};
  const double* axes[4] = {input->z, input->logk, input->logu, input->v};
  int dense_count[4];
  double lower[4];
  double upper[4];
  double inverse[4];
  prepare_axes(counts, axes, input->boost, dense_count, lower, upper, inverse);
  if (!(0.0 <= input->z[0])) { log_fatal(
    "baseline_3pcf: negative redshift axis"); exit(1); }
  if (!(0.0 >= input->logu[input->nu - 1])) { log_fatal(
    "baseline_3pcf: log(u) must be <=0"); exit(1); }
  if (!(0.0 <= input->v[0])) { log_fatal(
    "baseline_3pcf: v must be >=0"); exit(1); }
  if (!(1.0 >= input->v[input->nv - 1])) { log_fatal(
    "baseline_3pcf: v must be <=1"); exit(1); }
  const size_t size = table_size(counts, 4);
  (void)table_size(dense_count, 4);
  for (size_t index=0; index<size; index++) {
    if (!(isfinite(input->log_bispectrum[index]))) { log_fatal(
      "baseline_3pcf: ln B contains a nonfinite value"); exit(1); }
  }
}

baseline_3pcf* baseline_create_3pcf(
  const baseline_input_3pcf* input // borrowed table; NULL disables
)
{
  if (NULL == input) {
    return NULL;
  }
  if (!(1 <= input->boost)) { log_fatal(
    "baseline_3pcf: boost must be >=1"); exit(1); }
  if (!(0 == input->clamp
        || 1 == input->clamp)) { log_fatal(
    "baseline_3pcf: clamp must be 0 or 1"); exit(1); }
  if (!(NULL != input->log_bispectrum)) { log_fatal(
    "baseline_3pcf: missing ln B table"); exit(1); }
  const int counts[4] = {input->nz, input->nk, input->nu, input->nv};
  const double* axes[4] = {input->z, input->logk, input->logu, input->v};
  baseline_3pcf* baseline = calloc1d_generic(1, sizeof(*baseline));
  prepare_axes(counts, axes, input->boost, baseline->count, baseline->lower,
    baseline->upper, baseline->inverse);
  if (!(0.0 <= input->z[0])) { log_fatal(
    "baseline_3pcf: negative redshift axis"); exit(1); }
  if (!(0.0 >= input->logu[input->nu - 1])) { log_fatal(
    "baseline_3pcf: log(u) must be <=0"); exit(1); }
  if (!(0.0 <= input->v[0])) { log_fatal(
    "baseline_3pcf: v must be >=0"); exit(1); }
  if (!(1.0 >= input->v[input->nv - 1])) { log_fatal(
    "baseline_3pcf: v must be <=1"); exit(1); }
  // Prove the final dense count fits before allocating any intermediate
  // spline pass. The spline code still checks each intermediate size.
  (void)table_size(baseline->count, 4);
  baseline->clamp = input->clamp;
  baseline->logarithm = dense_table(input->log_bispectrum, counts, axes, 4,
    input->boost);
  return baseline;
}

void baseline_destroy_3pcf(
  baseline_3pcf* baseline // owned object; NULL does nothing
)
{
  if (NULL != baseline) {
    free(baseline->logarithm);
    free(baseline);
  }
}

double baseline_bispectrum_3pcf(
  const baseline_3pcf* baseline, // prepared physical B table
  const double* sides, // triangle sides [3], h/Mpc, already validated
  const double z // redshift
)
{
  if (!(NULL != baseline)) { log_fatal(
    "baseline_3pcf: missing prepared B table"); exit(1); }
  const double small = fmin(sides[0], fmin(sides[1], sides[2]));
  const double large = fmax(sides[0], fmax(sides[1], sides[2]));
  const double middle = fmax(fmin(sides[0], sides[1]),
    fmin(fmax(sides[0], sides[1]), sides[2]));
  const double coordinates[4] = {z, log(middle), log(small/middle),
    fmin(1.0, (large - middle)/small)};
  const double logarithm = read_log_table(baseline->count, baseline->lower,
    baseline->upper, baseline->inverse, baseline->clamp, baseline->logarithm,
    coordinates, 4);
  const double value = exp(logarithm);
  if (!(isfinite(value))) { log_fatal(
    "baseline_3pcf: B overflows floating-point storage"); exit(1); }
  if (!(0.0 < value)) { log_fatal(
    "baseline_3pcf: B must remain strictly positive"); exit(1); }
  return value;
}
