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
// Direct line-of-sight evaluation is useful as a reference but repeats
// the same physical fit at every transform grid point. fastnc avoids that
// cost in two stages. First it interpolates a smaller projected-B table.
// Then it integrates its angular dependence and interpolates the resulting
// b_L(ELL,PSI). This file implements those stages in C from the MIT-licensed
// fastnc tatt sources (7b2336ce...), bispectrum.py and multipole.py.
//
// The baseline uses log B on sorted RUV triangles: r=middle side,
// u=smallest/middle and v=(largest-middle)/smallest. Its radial weight is
// a product of three NLA kernels, so this *baseline* is invariant under
// side exchange. The signed TATT residual uses PUV without side sorting:
// p=sum sides, u=ell1/p, v=ell2/p. Its source-bin association is retained.
// The rectangular PUV fill uses the explicitly named source extension
// outside the physical triangle domain; physical queries stay inside it.
//
// ELL=sqrt(ell1^2+ell2^2), PSI=atan2(ell2,ell1). Source fastnc folds PSI
// to <=pi/4. That is insufficient for unequal-bin TATT. We store two
// independent charts: PSI=small_angle and PSI=pi/2-small_angle. Both use
// log(small_angle) as interpolation coordinate and preserve ordered legs.
// Only reference_mode=1 retains the source's irregular sampling. Production
// uses uniform log-small-PSI and a uniform bounded-tanh coordinate for PUV
// shape, followed by natural-cubic upsampling and arithmetic-index linear
// reads. No transpose or implicit bin swap creates the second residual chart.
//
// These tables own their arrays and borrow only their immutable model
// while being constructed. Changing any cosmology, IA parameter, source
// distribution, source triplet, grid or interpolation choice creates a
// new table. There is no pointer-identity cache and no mutable global.

typedef simde__m128d grid_v2d;

struct physics_grid_3pcf {
  physics_grid_settings_3pcf settings;
  int nr;
  int nu;
  int nv;
  int np;
  int nell;
  int npu;
  int npsi;
  int nmu;
  int has_residual; // false for exact pure-NLA parameter choices
  double shape_min;
  double* log_r;
  double* log_u;
  double* v;
  double* log_p;
  double* pu;
  double* shape_coordinate;
  double* log_ell;
  double* log_psi;
  double* mu;
  double* log_baseline; // [nr,nu,nv]
  double* residual; // [np,npu,npu]
  double* multipoles; // [baseline/residual,chart,L,nell,npsi]
};

static void grid_require(const int condition, const char* message)
{
  if (!condition) {
    fprintf(stderr, "physics_grid_3pcf: %s\n", message);
    exit(EXIT_FAILURE);
  }
}


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


static size_t grid_product(const int first, const int second, const int third)
{
  grid_require(first > 0 && second > 0 && third > 0,
    "grid dimensions must be positive");
  grid_require((size_t)first <= SIZE_MAX/(size_t)second
    && (size_t)third <= SIZE_MAX/((size_t)first*second),
    "grid dimension product overflows");
  return (size_t)first*second*third;
}


void physics_grid_defaults_3pcf(physics_grid_settings_3pcf* settings)
{
  grid_require(settings != NULL, "missing grid settings");
  settings->ell_min = 0.1;
  settings->ell_max = 1.0e5;
  settings->epmu = 1.0e-7;
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
}


static int dense_grid_count(const int count, const int boost)
{
  grid_require(count >= 2 && boost >= 1
    && count - 1 <= (INT_MAX - 1)/boost, "dense grid count overflows");
  return (count - 1)*boost + 1;
}


// Uniform coarse samples permit one tridiagonal natural-cubic solve per
// row. In units of one coarse cell, S(t)=y_i+t*(b+t*(c+t*d)); the second
// derivative vanishes at both boundaries. Only preparation evaluates this
// cubic. Its dense output is subsequently read by arithmetic-index linear
// interpolation, as in cosmo2D.c's Limber table preparation.
static void upsample_uniform_row(const double* coarse, const int count,
  const int boost, double* dense, double* second, double* diagonal)
{
  second[0] = 0.0;
  second[count - 1] = 0.0;
  diagonal[0] = 1.0;
  for (int index=1; index<count - 1; index++) {
    diagonal[index] = 4.0 - (index == 1 ? 0.0 : 1.0/diagonal[index - 1]);
    const double rhs = 6.0*(coarse[index + 1] - 2.0*coarse[index]
      + coarse[index - 1]);
    second[index] = rhs - (index == 1 ? 0.0
      : second[index - 1]/diagonal[index - 1]);
  }
  for (int index=count - 2; index>=1; index--) {
    second[index] = (second[index] - second[index + 1])/diagonal[index];
  }
  const int fine_count = dense_grid_count(count, boost);
  for (int fine=0; fine<fine_count; fine++) {
    const int index = fine/boost < count - 1 ? fine/boost : count - 2;
    const double offset = (double)(fine - index*boost)/boost;
    const double linear = coarse[index + 1] - coarse[index]
      - (second[index + 1] + 2.0*second[index])/6.0;
    const double quadratic = 0.5*second[index];
    const double cubic = (second[index + 1] - second[index])/6.0;
    dense[fine] = coarse[index]
      + offset*(linear + offset*(quadratic + offset*cubic));
  }
}


// Tensor-product upsampling applies the same one-dimensional solve to
// every line of a rectangular array. Each worker owns its line and scratch.
// A boost of one leaves that axis unchanged, so component/L labels are
// never interpolated when this routine prepares the b_L arrays.
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
    size_t worker_stride;
    double* scratch = worker_storage((size_t)3*old_count + new_count,
      &worker_stride);
    #pragma omp parallel if(!omp_in_parallel())
    {
      double* coarse = scratch + (size_t)omp_get_thread_num()*worker_stride;
      double* fine = coarse + old_count;
      double* second = fine + new_count;
      double* diagonal = second + old_count;
      #pragma omp for schedule(static)
      for (size_t line=0; line<before*after; line++) {
        const size_t outer = line/after;
        const size_t inner = line%after;
        for (int index=0; index<old_count; index++) {
          coarse[index] = input[(outer*old_count + index)*after + inner];
        }
        upsample_uniform_row(coarse, old_count, boost, fine, second, diagonal);
        for (int index=0; index<new_count; index++) {
          output[(outer*new_count + index)*after + inner] = fine[index];
        }
      }
    }
    free(scratch);
    free(input);
    input = output;
    dimensions[axis] = new_count;
  }
  return input;
}


static double shape_to_coordinate(const double ratio, const double minimum)
{
  // A bounded tanh chart places uniform-coordinate samples near both
  // squeezed edges without binary searches on irregular physical ratios.
  // Its inverse maps u in [1e-6,.5] to a uniform t in [-1,1].
  const double normalized = 2.0*(ratio - minimum)/(0.5 - minimum) - 1.0;
  return atanh(tanh(3.0)*normalized)/3.0;
}


static double coordinate_to_shape(const double coordinate, const double minimum)
{
  return minimum + 0.5*(0.5 - minimum)
    *(1.0 + tanh(3.0*coordinate)/tanh(3.0));
}


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


// The source joins a logarithmic lower region to a linear upper region.
// If a narrower requested domain starts above the join, no lower region
// is needed. This avoids duplicate or decreasing nodes at that boundary.
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


static double corrected_coordinate(const double coordinate,
  const double lower, const double upper, const int reference_mode)
{
  // fastnc calls numpy.isclose(x,bound,atol,rtol) positionally. Its actual
  // tolerances are therefore absolute 1e-5 and relative 1e-8. Preserve
  // those values explicitly; do not silently extrapolate farther away.
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


static int grid_interval(const double* axis, const int count,
  const double coordinate, const int uniform)
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


static double trilinear_grid(const double* values,
  const double* axis0, const int count0,
  const double* axis1, const int count1,
  const double* axis2, const int count2,
  double coordinate0, double coordinate1, double coordinate2,
  const int reference_mode)
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


double physics_grid_projected_3pcf(const physics_grid_3pcf* grid,
  const double ell1, const double ell2, const double ell3,
  const int component)
{
  grid_require(grid && ell1 > 0.0 && ell2 > 0.0 && ell3 > 0.0
    && isfinite(ell1 + ell2 + ell3) && component >= 0 && component <= 2,
    "invalid projected-grid query");
  const physics_grid_settings_3pcf* settings = &grid->settings;
  const double maximum = fmax(ell1, fmax(ell2, ell3));
  const double minimum = fmin(ell1, fmin(ell2, ell3));
  const double perimeter = ell1 + ell2 + ell3;
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
  }
  if (component != 1 && grid->has_residual) {
    const double ratio1 = corrected_coordinate(ell1/perimeter,
      grid->shape_min, 0.5, settings->reference_mode);
    const double ratio2 = corrected_coordinate(ell2/perimeter,
      grid->shape_min, 0.5, settings->reference_mode);
    const double shape1 = settings->reference_mode ? ratio1
      : shape_to_coordinate(ratio1, grid->shape_min);
    const double shape2 = settings->reference_mode ? ratio2
      : shape_to_coordinate(ratio2, grid->shape_min);
    residual = trilinear_grid(grid->residual,
      grid->log_p, grid->np, grid->shape_coordinate, grid->npu,
      grid->shape_coordinate, grid->npu, log(perimeter), shape1, shape2,
      settings->reference_mode);
  }
  return baseline + residual;
}


// The source method called 'gauss-legendre' integrates the *linear
// interpolant* of B(mu) exactly against P_L, rather than sampling at
// Gauss nodes. On one interval B=a*mu+b, its integral is a*J1+b*J0.
// Distribute those coefficients to the two endpoint values once. The
// endpoint mu=1-epmu remains explicit; no missing cap is invented here.
static double* source_legendre_weights(const physics_grid_3pcf* grid)
{
  const int lmax = grid->settings.lmax;
  const int count = grid->nmu;
  double* weights = grid_allocate((size_t)(lmax + 1)*count, sizeof(double));
  double* polynomial = grid_allocate((size_t)(lmax + 3)*count, sizeof(double));
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


static size_t multipole_grid_index(const physics_grid_3pcf* grid,
  const int component, const int chart, const int order,
  const int scale, const int angle)
{
  return ((((size_t)component*2 + chart)*(grid->settings.lmax + 1) + order)
    *grid->nell + scale)*grid->npsi + angle;
}


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
  size_t worker_stride;
  double* scratch = worker_storage((size_t)2*count, &worker_stride);
  #pragma omp parallel if(!omp_in_parallel())
  {
    double* baseline = scratch + (size_t)omp_get_thread_num()*worker_stride;
    double* residual = baseline + count;
    #pragma omp for collapse(2) schedule(static)
    for (int scale=0; scale<nell; scale++) {
      for (int angle=0; angle<npsi; angle++) {
        const double magnitude = exp(grid->log_ell[scale]);
        const double small = exp(grid->log_psi[angle]);
        for (int chart=0; chart<2; chart++) {
          const double psi = chart == 0 ? small
            : 1.57079632679489661923 - small;
          const double first = magnitude*cos(psi);
          const double second = magnitude*sin(psi);
          for (int node=0; node<count; node++) {
            const double difference = first - second;
            const double third = sqrt(difference*difference
              + 2.0*first*second*(1.0 - grid->mu[node]));
            if (chart == 0) {
              baseline[node] = physics_grid_projected_3pcf(grid,
                first, second, third, 1);
            }
            residual[node] = physics_grid_projected_3pcf(grid,
              first, second, third, 2);
          }
          for (int component=0; component<2; component++) {
            const double* values = component == 0 ? baseline : residual;
            int order = 0;
            for (; order + 1<=lmax; order += 2) {
              // Independent neighboring L sums start at zero.
              grid_v2d sum = simde_mm_setzero_pd();
              for (int node=0; node<count; node++) {
                // High lane is L+1, low lane is L.
                const grid_v2d weight = simde_mm_set_pd(
                  weights[(size_t)(order + 1)*count + node],
                  weights[(size_t)order*count + node]);
                // Both orders see the same sampled projected B.
                const grid_v2d value = simde_mm_set1_pd(values[node]);
                // Scalar form: sum_L += weight[L,node]*B[node].
                const grid_v2d term = simde_mm_mul_pd(weight, value);
                // Keep increasing mu-node order within each L.
                sum = simde_mm_add_pd(sum, term);
              }
              double result[2];
              // Save both L values before placing them in their own planes.
              simde_mm_storeu_pd(result, sum);
              grid->multipoles[multipole_grid_index(grid, component,
                chart, order, scale, angle)] = result[0];
              grid->multipoles[multipole_grid_index(grid, component,
                chart, order + 1, scale, angle)] = result[1];
            }
            if (order <= lmax) {
              double sum = 0.0;
              for (int node=0; node<count; node++) {
                // Match the vector product's rounding before addition.
                // Otherwise extending lmax moves this L from a fused
                // scalar tail to a separately rounded vector lane.
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
  free(weights);
}


static physics_grid_3pcf* allocate_grid(
  const physics_grid_settings_3pcf* settings,
  const int has_residual
)
{
  grid_require(settings != NULL, "missing grid settings");
  grid_require(settings->ell_min > 0.0
    && settings->ell_max > settings->ell_min && isfinite(settings->ell_max)
    && settings->epmu > 0.0 && settings->epmu < 0.05
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
    && settings->prepare_multipoles <= 1,
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
  grid->pu[0] = 1.0e-6;
  for (int index=0; index<settings->npu_log; index++) {
    const double small = exp(log(1.0e-4) + log(300.0)
      *index/(settings->npu_log - 1));
    grid->pu[1 + index] = small;
    grid->pu[grid->npu - 2 - index] = 0.5 - small;
  }
  for (int index=0; index<settings->npu_linear; index++) {
    grid->pu[1 + settings->npu_log + index] = 0.047 + (0.453 - 0.047)
      *index/(settings->npu_linear - 1);
  }
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
        grid->shape_min);
    }
  }
  grid->log_ell = linear_axis(log(magnitude_min), log(magnitude_max),
    settings->nell);
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
    grid->log_psi = linear_axis(log(psi_min),
      log(0.78539816339744830962), grid->npsi);
  }
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
  if (has_residual) {
    grid->residual = grid_allocate(nresidual, sizeof(double));
  }
  return grid;
}


// Coarse values are complete before interpolation preparation begins.
// Natural-cubic upsampling produces dense uniform axes, whose subsequent
// readers require arithmetic cell indices and linear interpolation only.
static void finish_grid(physics_grid_3pcf* grid)
{
  const physics_grid_settings_3pcf* settings = &grid->settings;
  if (!settings->reference_mode) {
    const int boost = settings->table_boost;
    const int boosts[3] = {boost, boost, boost};
    int baseline_shape[3] = {grid->nr, grid->nu, grid->nv};
    int residual_shape[3] = {grid->np, grid->npu, grid->npu};
    grid->log_baseline = upsample_uniform_grid(grid->log_baseline,
      baseline_shape, boosts);
    if (grid->has_residual) {
      grid->residual = upsample_uniform_grid(grid->residual,
        residual_shape, boosts);
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
        grid->shape_min);
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
    double* log_psi = linear_axis(grid->log_psi[0],
      grid->log_psi[grid->npsi - 1], shape[2]);
    free(grid->log_ell);
    free(grid->log_psi);
    grid->log_ell = log_ell;
    grid->log_psi = log_psi;
    grid->nell = shape[1];
    grid->npsi = shape[2];
  }
}


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
  physics_projection_3pcf** projections = grid_allocate((size_t)ntriplet,
    sizeof(*projections));
  // Pure NLA has no residual after its linear-alignment subtraction.
  // Keep no PUV values and perform no residual physics in that case.
  const int has_residual = parameters->a2 != 0.0
    || parameters->bta != 0.0 || parameters->ct != 0.0;
  for (int triplet=0; triplet<ntriplet; triplet++) {
    out[triplet] = allocate_grid(settings, has_residual);
    projections[triplet] = physics_projector_create_3pcf(model, parameters,
      bins + (size_t)triplet*3);
  }
  const physics_grid_3pcf* first = out[0];
  const size_t nbaseline = grid_product(first->nr, first->nu, first->nv);
  const size_t nresidual = has_residual
    ? grid_product(first->np, first->npu, first->npu) : 0;
  // Every worker owns one geometry point and an ntriplet scratch vector.
  // Physical B/IA at each z is shared before the source-weight contraction.
  // Memory grows with the number of output tables, not ntriplet*nlos
  // times all sampled triangle geometries.
  size_t worker_stride;
  double* scratch = worker_storage((size_t)ntriplet, &worker_stride);
  #pragma omp parallel if(!omp_in_parallel())
  {
    double* values = scratch + (size_t)omp_get_thread_num()*worker_stride;
    #pragma omp for schedule(static)
    for (size_t index=0; index<nbaseline; index++) {
      const int iv = index%(size_t)first->nv;
      const int iu = (index/(size_t)first->nv)%(size_t)first->nu;
      const int ir = index/((size_t)first->nv*first->nu);
      const double radius = exp(first->log_r[ir]);
      const double ratio = exp(first->log_u[iu]);
      physics_projected_batch_3pcf(ntriplet, projections,
        radius*(1.0 + ratio*first->v[iv]), radius, radius*ratio, 1, 0,
        values);
      for (int triplet=0; triplet<ntriplet; triplet++) {
        grid_require(values[triplet] > 0.0 && isfinite(values[triplet]),
          "RUV log-baseline requires positive B; use direct signed values");
        out[triplet]->log_baseline[index] = log(values[triplet]);
      }
    }
    #pragma omp for schedule(static)
    for (size_t index=0; index<nresidual; index++) {
      const int iv = index%(size_t)first->npu;
      const int iu = (index/(size_t)first->npu)%(size_t)first->npu;
      const int ip = index/((size_t)first->npu*first->npu);
      const double perimeter = exp(first->log_p[ip]);
      const double side1 = perimeter*first->pu[iu];
      const double side2 = perimeter*first->pu[iv];
      const double side3 = perimeter*(1.0 - first->pu[iu] - first->pu[iv]);
      physics_projected_batch_3pcf(ntriplet, projections,
        side1, side2, side3, 2, 1, values);
      for (int triplet=0; triplet<ntriplet; triplet++) {
        grid_require(isfinite(values[triplet]), "nonfinite PUV residual");
        out[triplet]->residual[index] = values[triplet];
      }
    }
  }
  free(scratch);
  for (int triplet=0; triplet<ntriplet; triplet++) {
    physics_projector_destroy_3pcf(projections[triplet]);
    finish_grid(out[triplet]);
  }
  free(projections);
}


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


static double grid_aperture_reader(
  const void* context,
  const double ell1,
  const double ell2,
  const double ell3
)
{
  return physics_grid_projected_3pcf(context, ell1, ell2, ell3, 0);
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
  // The same Fourier quadrature is used by the direct reference. Only
  // its B provider changes. The b_L table is unnecessary for this route.
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
  free(grid->mu);
  free(grid->log_baseline);
  free(grid->residual);
  free(grid->multipoles);
  free(grid);
}


void physics_grid_multipoles_3pcf(const physics_grid_3pcf* grid,
  const int nell, const double* ell, const int component, double* out)
{
  grid_require(grid && ell && out && nell > 0
    && component >= 0 && component <= 2, "invalid multipole table query");
  grid_require(grid->settings.prepare_multipoles,
    "this grid omitted b_L; recreate with prepare_multipoles=1");
  const int lmax = grid->settings.lmax;
  grid_product(lmax + 1, nell, nell);
  const size_t npoint = (size_t)nell*nell;
  int* charts = grid_allocate(npoint, sizeof(int));
  size_t* cells = grid_allocate(npoint, sizeof(size_t));
  double* fractions = grid_allocate(4*npoint, sizeof(double));
  // The geometry is shared by every L and both physical components.
  // Compute its coordinate transforms, integer cells and four bilinear
  // weights once. The later interpolation loop contains neither a
  // logarithm nor a binary search nor repeated cell-index arithmetic.
  for (int first=0; first<nell; first++) {
    for (int second=0; second<nell; second++) {
      grid_require(isfinite(ell[first]) && ell[first] > 0.0
        && isfinite(ell[second]) && ell[second] > 0.0,
        "invalid Fourier multipole");
      const int chart = ell[second] > ell[first] ? 1 : 0;
      const double small = atan2(fmin(ell[first], ell[second]),
        fmax(ell[first], ell[second]));
      const double x = corrected_coordinate(log(hypot(ell[first], ell[second])),
        grid->log_ell[0], grid->log_ell[grid->nell - 1],
        grid->settings.reference_mode);
      const double y = corrected_coordinate(log(small), grid->log_psi[0],
        grid->log_psi[grid->npsi - 1], grid->settings.reference_mode);
      const int ix = grid_interval(grid->log_ell, grid->nell, x,
        !grid->settings.reference_mode);
      const int iy = grid_interval(grid->log_psi, grid->npsi, y,
        !grid->settings.reference_mode);
      const double fraction_x = (x - grid->log_ell[ix])
        /(grid->log_ell[ix + 1] - grid->log_ell[ix]);
      const double fraction_y = (y - grid->log_psi[iy])
        /(grid->log_psi[iy + 1] - grid->log_psi[iy]);
      const size_t point = (size_t)first*nell + second;
      charts[point] = chart;
      cells[point] = (size_t)ix*grid->npsi + iy;
      fractions[point*4] = (1.0 - fraction_x)*(1.0 - fraction_y);
      fractions[point*4 + 1] = (1.0 - fraction_x)*fraction_y;
      fractions[point*4 + 2] = fraction_x*(1.0 - fraction_y);
      fractions[point*4 + 3] = fraction_x*fraction_y;
    }
  }
  const size_t plane = (size_t)grid->nell*grid->npsi;
  const size_t offsets[4] = {0, 1, (size_t)grid->npsi,
    (size_t)grid->npsi + 1};
  #pragma omp parallel for collapse(2) if(!omp_in_parallel()) schedule(static)
  for (int order=0; order<=lmax; order++) {
    for (int first=0; first<nell; first++) {
      double* restrict output = out + ((size_t)order*nell + first)*nell;
      const double* restrict tables = grid->multipoles;
      int second = 0;
      for (; second + 1<nell; second += 2) {
        const size_t point = (size_t)first*nell + second;
        // Lanes represent adjacent ell2 values in one fixed (L,ell1) row.
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
            // Gather this same corner for the two prepared query cells.
            const grid_v2d values = simde_mm_set_pd(
              tables[base1 + offsets[corner]], tables[base0 + offsets[corner]]);
            // High lane uses ell2[second+1], low lane ell2[second].
            const grid_v2d weights = simde_mm_set_pd(
              fractions[(point + 1)*4 + corner], fractions[point*4 + corner]);
            // Scalar equivalent: value += corner_weight*table[corner].
            const grid_v2d term = simde_mm_mul_pd(weights, values);
            // Both outputs retain the same component/corner addition order.
            sum = simde_mm_add_pd(sum, term);
          }
        }
        // Adjacent output ell2 values are contiguous in the C-order row.
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
            // An added query must not change earlier values when their
            // former scalar tail becomes a vector lane. Strict flags
            // alone do not forbid scalar multiply-add contraction.
            volatile double term =
              fractions[point*4 + corner]*tables[base + offsets[corner]];
            sum += term;
          }
        }
        output[second] = sum;
      }
    }
  }
  free(charts);
  free(cells);
  free(fractions);
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
