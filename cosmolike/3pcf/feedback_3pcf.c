#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_spline.h>
#include "feedback_3pcf.h"

// Feedback redistributes matter through gas cooling and energetic outflows.
// A power response and a bispectrum response constrain different moments
// of that redistribution: neither determines the other. These tables store
// both log ratios. A single declared amplitude A applies R_X(A)=exp(A lnR_X).
// A=0 is DMO and A=1 is the supplied pair; interpolation in A is a model
// assumption, not a new simulation calibration. IA residuals never read
// these tables. Only their nonlinear-alignment baseline receives feedback.
//
// Preparation is serial and owned by one immutable response. Workers read
// uniform arrays with arithmetic indices and no GSL calls, allocations or
// mutable state. Storage is flat, unpadded and in the documented row order.
struct feedback_3pcf {
  int count[4]; // z, log(r), log(u), v dense node counts
  double lower[4]; // first coordinate on each axis
  double upper[4]; // final coordinate on each axis
  double inverse[4]; // inverse dense spacing
  double* power; // [nz,nk], dense log R_P
  double* bispectrum; // [nz,nk,nu,nv], dense log R_B
  int clamp; // explicitly selected boundary extension
};

static void require_feedback(const int condition, const char* message)
{
  if (!condition) {
    fprintf(stderr, "feedback_3pcf: %s\n", message);
    exit(EXIT_FAILURE);
  }
}


static size_t table_size(const int* counts, const int dimensions)
{
  size_t size = 1;
  for (int axis=0; axis<dimensions; axis++) {
    require_feedback(counts[axis] > 0
      && size <= SIZE_MAX/sizeof(double)/(size_t)counts[axis],
      "response dimensions overflow addressable storage");
    size *= (size_t)counts[axis];
  }
  return size;
}


static double* allocate_feedback(const size_t size)
{
  require_feedback(size <= SIZE_MAX/sizeof(double), "allocation overflows");
  double* values = malloc(size*sizeof(double));
  require_feedback(values != NULL, "response allocation failed");
  return values;
}


// Resample one axis at a time. Each line has fixed values of all other
// coordinates. Cubic interpolation along these independent lines defines
// the tensor-product preparation; hot queries later use linear weights.
static double* dense_table(
  const double* source,
  const int* counts,
  const double* const* axes,
  const int dimensions,
  const int boost
)
{
  int current[4];
  memcpy(current, counts, sizeof(current));
  const size_t size = table_size(current, dimensions);
  double* table = allocate_feedback(size);
  for (size_t index=0; index<size; index++) {
    require_feedback(isfinite(source[index]), "nonfinite log response");
    table[index] = source[index];
  }
  if (boost == 1) {
    return table;
  }
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
    double* dense = allocate_feedback(table_size(current, dimensions));
    double* line = allocate_feedback((size_t)old_count);
    gsl_spline* spline = gsl_spline_alloc(gsl_interp_cspline, old_count);
    gsl_interp_accel* search = gsl_interp_accel_alloc();
    require_feedback(spline && search, "spline preparation allocation failed");
    for (size_t block=0; block<blocks; block++) {
      for (size_t inner=0; inner<stride; inner++) {
        for (int node=0; node<old_count; node++) {
          line[node] = table[(block*old_count + node)*stride + inner];
        }
        gsl_spline_init(spline, axes[axis], line, old_count);
        gsl_interp_accel_reset(search);
        for (int node=0; node<new_count; node++) {
          // Original knots are retained exactly, including both endpoints.
          double value;
          if (node%boost == 0) {
            value = line[node/boost];
          } else {
            const double coordinate = axes[axis][0]
              + (axes[axis][old_count - 1] - axes[axis][0])
              *node/(new_count - 1);
            value = gsl_spline_eval(spline, coordinate, search);
          }
          require_feedback(isfinite(value), "nonfinite prepared log response");
          dense[(block*new_count + node)*stride + inner] = value;
        }
      }
    }
    gsl_interp_accel_free(search);
    gsl_spline_free(spline);
    free(line);
    free(table);
    table = dense;
  }
  return table;
}


feedback_3pcf* feedback_create_3pcf(const feedback_input_3pcf* input)
{
  if (input == NULL) {
    return NULL;
  }
  require_feedback(input->boost >= 1 && input->clamp >= 0 && input->clamp <= 1
    && input->log_power && input->log_bispectrum,
    "both response tables, boost>=1 and clamp=0/1 are required");
  const int counts[4] = {input->nz, input->nk, input->nu, input->nv};
  const double* axes[4] = {input->z, input->logk, input->logu, input->v};
  feedback_3pcf* response = calloc(1, sizeof(*response));
  require_feedback(response != NULL, "response owner allocation failed");
  response->clamp = input->clamp;
  for (int axis=0; axis<4; axis++) {
    const int count = counts[axis];
    require_feedback(count >= 4 && axes[axis]
      && count - 1 <= (INT_MAX - 1)/input->boost,
      "each response axis needs four nodes and representable dense size");
    const double low = axes[axis][0];
    const double high = axes[axis][count - 1];
    require_feedback(isfinite(low) && isfinite(high) && high > low,
      "response coordinates must increase");
    const double step = (high - low)/(count - 1);
    require_feedback(isfinite(step) && step > 0.0,
      "response spacing is not representable");
    for (int node=0; node<count; node++) {
      const double expected = low + node*step;
      require_feedback(isfinite(axes[axis][node])
        && (node == 0 || axes[axis][node] > axes[axis][node - 1])
        && fabs(axes[axis][node] - expected)
          <= 64.0*DBL_EPSILON*fmax(1.0, fmax(fabs(low), fabs(high))),
        "response coordinates must be uniform");
    }
    response->count[axis] = input->boost*(count - 1) + 1;
    response->lower[axis] = low;
    response->upper[axis] = high;
    response->inverse[axis] = (response->count[axis] - 1)/(high - low);
    require_feedback(isfinite(response->inverse[axis])
      && response->inverse[axis] > 0.0,
      "dense response inverse spacing is not representable");
  }
  require_feedback(input->z[0] >= 0.0 && input->logu[input->nu - 1] <= 0.0
    && input->v[0] >= 0.0 && input->v[input->nv - 1] <= 1.0,
    "response z/u/v axes must describe physical sorted triangles");
  response->power = dense_table(input->log_power, counts, axes, 2,
    input->boost);
  response->bispectrum = dense_table(input->log_bispectrum, counts, axes, 4,
    input->boost);
  return response;
}


void feedback_destroy_3pcf(feedback_3pcf* response)
{
  if (response != NULL) {
    free(response->power);
    free(response->bispectrum);
    free(response);
  }
}


static double read_response(
  const feedback_3pcf* response,
  const double* coordinates,
  const int dimensions,
  const double amplitude
)
{
  require_feedback(isfinite(amplitude), "feedback amplitude must be finite");
  if (amplitude == 0.0) {
    return 1.0;
  }
  require_feedback(response != NULL, "nonzero feedback needs paired tables");
  int cells[4];
  double fractions[4];
  for (int axis=0; axis<dimensions; axis++) {
    const double low = response->lower[axis];
    const double high = response->upper[axis];
    const double value = coordinates[axis];
    const double tolerance = 64.0*DBL_EPSILON*fmax(1.0, fabs(value));
    require_feedback(isfinite(value) && (response->clamp
      || (value >= low - tolerance && value <= high + tolerance)),
      "feedback query exceeds its declared table domain");
    const double position = (fmax(low, fmin(high, value)) - low)
      *response->inverse[axis];
    cells[axis] = (int)fmin(response->count[axis] - 2, floor(position));
    fractions[axis] = position - cells[axis];
  }
  const double* table = dimensions == 2
    ? response->power : response->bispectrum;
  double logarithm = 0.0;
  // A 2D cell has four corners and a 4D cell has sixteen. Products of
  // one-dimensional linear weights sum to one and preserve constant ratios.
  for (int corner=0; corner<(1 << dimensions); corner++) {
    size_t index = 0;
    double weight = 1.0;
    for (int axis=0; axis<dimensions; axis++) {
      const int upper = (corner >> axis)&1;
      index = index*response->count[axis] + cells[axis] + upper;
      weight *= upper ? fractions[axis] : 1.0 - fractions[axis];
    }
    logarithm += weight*table[index];
  }
  const double result = exp(amplitude*logarithm);
  require_feedback(isfinite(result) && result > 0.0,
    "feedback amplitude produces an unrepresentable positive ratio");
  return result;
}


double feedback_power_3pcf(
  const feedback_3pcf* response,
  const double k,
  const double z,
  const double amplitude
)
{
  const double coordinates[2] = {z, log(k)};
  return read_response(response, coordinates, 2, amplitude);
}


double feedback_bispectrum_3pcf(
  const feedback_3pcf* response,
  const double* sides,
  const double z,
  const double amplitude
)
{
  if (amplitude == 0.0) {
    return 1.0;
  }
  const double small = fmin(sides[0], fmin(sides[1], sides[2]));
  const double large = fmax(sides[0], fmax(sides[1], sides[2]));
  // Select the middle value by comparisons. Subtracting extremes from
  // their sum loses the middle side in very squeezed triangles.
  const double middle = fmax(fmin(sides[0], sides[1]),
    fmin(fmax(sides[0], sides[1]), sides[2]));
  // Validated physical triangles can round just beyond a degenerate
  // boundary. Keep this shape coordinate in its mathematical [0,1] range.
  const double v = fmin(1.0, (large - middle)/small);
  const double coordinates[4] = {z, log(middle), log(small/middle),
    v};
  return read_response(response, coordinates, 4, amplitude);
}
