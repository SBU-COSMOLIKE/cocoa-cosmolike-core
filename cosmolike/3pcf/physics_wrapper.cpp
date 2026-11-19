#include <climits>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <spdlog/spdlog.h>
#include "physics_wrapper.hpp"

namespace cosmolike_interface {

// ---------------------------------------------------------------------------
// physics_wrapper.cpp: the notebook route to the 3PCF physics model.
// The notebook adapter receives private Armadillo arrays. Armadillo is
// column-major: it stores each column contiguously, so the first (row)
// index runs fastest in memory. The C model reads row-major tables,
// where the last index runs fastest. The loops here copy between those
// layouts; they neither evaluate a physical kernel nor choose an
// integration rule. physics_3pcf.h and physics_grid_3pcf.h hold the
// physics these calls reach.
// ---------------------------------------------------------------------------


// Build the immutable C model; physics_create_3pcf copies every input.
physics_owner_3pcf::physics_owner_3pcf(const physics_input_3pcf& input)
  : model(physics_create_3pcf(&input), physics_destroy_3pcf)
{
}


// Expose the borrowed C pointer; the owner must outlive every reader.
const physics_model_3pcf* physics_owner_3pcf::get() const
{
  return model.get();
}


physics_grid_owner_3pcf::physics_grid_owner_3pcf(
  const std::shared_ptr<physics_owner_3pcf>& model,
  const physics_ia_3pcf& parameters,
  const int* bins,
  const physics_grid_settings_3pcf& settings
) : grid(physics_grid_create_3pcf(model->get(), &parameters, bins, &settings),
      physics_grid_destroy_3pcf), maximum_order(settings.lmax)
{
  // All physical projections are complete at construction. The table
  // owns its values and remains valid even after the model is released.
}


// Adopt a table the batch factory already built; physics_wrapper.hpp
// explains why lmax travels alongside it.
physics_grid_owner_3pcf::physics_grid_owner_3pcf(
  physics_grid_3pcf* prepared,
  const int lmax
) : grid(prepared, physics_grid_destroy_3pcf), maximum_order(lmax)
{
}


const physics_grid_3pcf* physics_grid_owner_3pcf::get() const
{
  return grid.get();
}


int physics_grid_owner_3pcf::lmax() const
{
  return maximum_order;
}


// ---------------------------------------------------------------------------
// Notebook model factory. It validates the Armadillo tables, repacks
// the two matrices into row-major C storage, and hands the pointers to
// physics_create_3pcf, which copies every physical value into an
// immutable model. physics_3pcf.h defines the table contract and the
// units: k in h/Mpc, chi in Mpc/h, P in (Mpc/h)^3, a redshift grid
// that includes zero.
//
// Parameters:
//   documented argument by argument in physics_wrapper.hpp
//
// Returns:
//   shared owner of the C model; Python keeps it alive
// ---------------------------------------------------------------------------
std::shared_ptr<physics_owner_3pcf> physics_model_cpp_3pcf(
  const arma::Col<double>& k,
  const arma::Col<double>& p_linear0,
  const arma::Col<double>& z,
  const arma::Mat<double>& p_nonlinear,
  const arma::Col<double>& chi,
  const arma::Col<double>& dz_dchi,
  const arma::Col<double>& growth,
  const arma::Mat<double>& sources,
  const double omega_m,
  const double omega_de,
  const double w0,
  const double wa,
  const double ns,
  const double sigma8,
  const arma::Col<double>& los_z,
  const int kernel_nz,
  const bool source_readers,
  const int table_boost,
  const int fit_nz,
  const arma::Col<double>& source_z,
  const physics_feedback_tables_cpp_3pcf* feedback,
  const arma::Col<double>* pixel_window,
  const physics_baseline_table_cpp_3pcf* baseline,
  const arma::Col<double>* omega_m_z
)
{
  const arma::uword nsource = source_z.n_elem ? source_z.n_elem : z.n_elem;

  // One chained size check: cubic preparation needs at least four k
  // nodes, four z nodes and four source-coordinate nodes; the line of
  // sight needs at least two nodes; at least one source bin must exist;
  // every count must fit the C int fields of physics_input_3pcf.
  if (!(k.n_elem >= 4 && k.n_elem <= INT_MAX
        && z.n_elem >= 4 && z.n_elem <= INT_MAX
        && sources.n_rows > 0 && sources.n_rows <= INT_MAX
        && los_z.n_elem >= 2 && los_z.n_elem <= INT_MAX
        && nsource >= 4 && nsource <= INT_MAX)) {
    spdlog::critical("3pcf physics wrapper: invalid model dimensions");
    std::exit(1);
  }

  // One chained cross-table check: the linear power lives on the k
  // axis; the nonlinear power is [nz,nk]; chi, dz/dchi and the growth
  // factor live on the z axis; each source row holds one density per
  // source-coordinate node.
  if (!(p_linear0.n_elem == k.n_elem
        && p_nonlinear.n_rows == z.n_elem && p_nonlinear.n_cols == k.n_elem
        && chi.n_elem == z.n_elem && dz_dchi.n_elem == z.n_elem
        && growth.n_elem == z.n_elem && sources.n_cols == nsource)) {
    spdlog::critical("3pcf physics wrapper: model table shapes disagree");
    std::exit(1);
  }

  // Repack both matrices for C. Armadillo stores them column-major
  // (row index fastest); the C model reads row-major tables (last
  // index fastest), so each loop writes one row at a time.
  std::vector<double> nonlinear(p_nonlinear.n_elem);
  std::vector<double> source(sources.n_elem);

  // nonlinear[row*nk + column] lays P(z,k) out with k running fastest
  for (arma::uword row=0; row<p_nonlinear.n_rows; row++) {
    for (arma::uword column=0; column<p_nonlinear.n_cols; column++) {
      nonlinear[row*p_nonlinear.n_cols + column] = p_nonlinear(row, column);
    }
  }

  // source[row*nsource + column] puts the source coordinate fastest
  for (arma::uword row=0; row<sources.n_rows; row++) {
    for (arma::uword column=0; column<sources.n_cols; column++) {
      source[row*sources.n_cols + column] = sources(row, column);
    }
  }

  physics_input_3pcf input = {};
  input.nk = static_cast<int>(k.n_elem);
  input.nz = static_cast<int>(z.n_elem);
  input.nbin = static_cast<int>(sources.n_rows);
  input.nlos = static_cast<int>(los_z.n_elem);
  input.kernel_nz = kernel_nz;
  input.source_readers = source_readers ? 1 : 0;
  input.table_boost = table_boost;
  input.fit_nz = fit_nz;
  input.k = k.memptr();
  input.p_linear0 = p_linear0.memptr();
  input.z = z.memptr();
  input.p_nonlinear = nonlinear.data();
  input.chi = chi.memptr();
  input.dz_dchi = dz_dchi.memptr();
  input.growth = growth.memptr();
  input.sources = source.data();
  if (source_z.n_elem) {
    input.nsource = static_cast<int>(source_z.n_elem);
    input.source_z = source_z.memptr();
  }
  input.los_z = los_z.memptr();
  input.omega_m = omega_m;
  input.omega_de = omega_de;
  input.w0 = w0;
  input.wa = wa;
  input.ns = ns;
  input.sigma8 = sigma8;

  feedback_input_3pcf response = {};
  if (feedback != nullptr) {
    // The four axis counts must fit the C int fields before any size_t
    // arithmetic uses them.
    if (!(feedback->z.n_elem <= INT_MAX
          && feedback->logk.n_elem <= INT_MAX
          && feedback->logu.n_elem <= INT_MAX
          && feedback->v.n_elem <= INT_MAX)) {
      spdlog::critical("3pcf physics wrapper: feedback axis sizes exceed C "
        "integer storage");
      std::exit(1);
    }

    const size_t nz = feedback->z.n_elem;
    const size_t nk = feedback->logk.n_elem;
    const size_t nu = feedback->logu.n_elem;
    const size_t nv = feedback->v.n_elem;

    // One chained check: each axis needs at least four nodes for cubic
    // preparation; each division guard proves the next multiplication
    // cannot wrap around SIZE_MAX; the flattened tables must then match
    // their axis products exactly.
    if (!(nz >= 4 && nk >= 4 && nu >= 4 && nv >= 4
          && nz <= SIZE_MAX/nk && nz*nk <= SIZE_MAX/nu
          && nz*nk*nu <= SIZE_MAX/nv
          && feedback->log_power.n_elem == nz*nk
          && feedback->log_bispectrum.n_elem == nz*nk*nu*nv)) {
      spdlog::critical("3pcf physics wrapper: flattened paired feedback "
        "arrays disagree with their axes");
      std::exit(1);
    }

    response.nz = static_cast<int>(nz);
    response.nk = static_cast<int>(nk);
    response.nu = static_cast<int>(nu);
    response.nv = static_cast<int>(nv);
    response.z = feedback->z.memptr();
    response.logk = feedback->logk.memptr();
    response.logu = feedback->logu.memptr();
    response.v = feedback->v.memptr();
    response.log_power = feedback->log_power.memptr();
    response.log_bispectrum = feedback->log_bispectrum.memptr();
    response.boost = feedback->boost;
    response.clamp = feedback->clamp;
    input.feedback = &response;
  }

  if (pixel_window != nullptr) {
    // The window reader interpolates linearly between integer
    // multipoles, so it needs at least two samples.
    if (!(pixel_window->n_elem >= 2
          && pixel_window->n_elem <= INT_MAX && pixel_window->is_finite())) {
      spdlog::critical("3pcf physics wrapper: pixel window requires a "
        "finite vector with at least two samples");
      std::exit(1);
    }
    input.nwindow = static_cast<int>(pixel_window->n_elem);
    input.window = pixel_window->memptr();
  }

  baseline_input_3pcf supplied = {};
  if (nullptr != baseline) {
    if (!(nullptr != omega_m_z)) { spdlog::critical(
      "3pcf: supplied B needs Omega_m(z)"); std::exit(1); }
    if (!(z.n_elem == omega_m_z->n_elem)) { spdlog::critical(
      "3pcf: Omega_m(z) must match background z"); std::exit(1); }
    const size_t counts[4] = {baseline->z.n_elem, baseline->logk.n_elem,
      baseline->logu.n_elem, baseline->v.n_elem};
    size_t size = 1;
    for (int axis=0; axis<4; axis++) {
      if (!(4 <= counts[axis])) { spdlog::critical(
        "3pcf: B axes need four nodes"); std::exit(1); }
      if (!(counts[axis] <= INT_MAX/size)) { spdlog::critical(
        "3pcf: B dimensions exceed C storage"); std::exit(1); }
      size *= counts[axis];
    }
    if (!(size == baseline->log_bispectrum.n_elem)) { spdlog::critical(
      "3pcf: B table size disagrees with axes"); std::exit(1); }
    supplied.nz = static_cast<int>(counts[0]);
    supplied.nk = static_cast<int>(counts[1]);
    supplied.nu = static_cast<int>(counts[2]);
    supplied.nv = static_cast<int>(counts[3]);
    supplied.z = baseline->z.memptr();
    supplied.logk = baseline->logk.memptr();
    supplied.logu = baseline->logu.memptr();
    supplied.v = baseline->v.memptr();
    supplied.log_bispectrum = baseline->log_bispectrum.memptr();
    supplied.boost = baseline->boost;
    supplied.clamp = baseline->clamp;
    input.baseline = &supplied;
    input.omega_m_z = omega_m_z->memptr();
  }

  return std::make_shared<physics_owner_3pcf>(input);
}


// ---------------------------------------------------------------------------
// Notebook point evaluator: the Armadillo twin of the production
// dispatch in physics_interface.cpp. One integer selects the physical
// operation, and physics_3pcf.h holds each C function it reaches:
// 0 evaluates the matter bispectrum, 1 the nineteen IA bispectra,
// 2 the projected bispectrum through a shared projector (three source
// bins instead of redshifts), 3 the nineteen IA bispectra for
// triangles of one common shape, and 4 the baryon-feedback bispectrum
// response.
//
// This route stays serial on purpose, while the production route
// parallelizes over triangles: the notebook adapter serves interactive
// exploration, where a simple loop is worth more than speed.
//
// Parameters:
//   owner      - the immutable model
//   sides      - [point,3] triangle sides: k in h/Mpc, or multipoles
//                ell for operation 2
//   redshifts  - [point] evaluation redshifts (operations 0, 1, 3, 4)
//   bins       - [3] ordered source bins (operation 2 only)
//   parameters - IA parameter choice (physics_ia_3pcf)
//   operation, component, remove_alignment - see above and
//                physics_3pcf.h
//
// Returns:
//   [point,ncomponent] matrix, checked finite
// ---------------------------------------------------------------------------
arma::Mat<double> physics_points_cpp_3pcf(
  const physics_owner_3pcf& owner,
  const arma::Mat<double>& sides,
  const arma::Col<double>& redshifts,
  const arma::Col<int>& bins,
  const physics_ia_3pcf& parameters,
  const int operation,
  const int component,
  const bool remove_alignment
)
{
  if (!(sides.n_rows > 0 && sides.n_rows <= INT_MAX
        && sides.n_cols == 3)) {
    spdlog::critical("3pcf physics wrapper: sides must have shape [point,3]");
    std::exit(1);
  }
  if (!(operation >= 0 && operation <= 4)) {
    spdlog::critical("3pcf physics wrapper: unknown point operation");
    std::exit(1);
  }
  if (!(operation == 2 ? bins.n_elem == 3
        : redshifts.n_elem == sides.n_rows)) {
    spdlog::critical("3pcf physics wrapper: point query shapes disagree");
    std::exit(1);
  }

  // Operations 1 and 3 return the nineteen density/E/B leg combinations
  // that physics_ia_bispectra_3pcf lists in physics_3pcf.h; every other
  // operation returns one value per triangle.
  const int ncomponent = operation == 1 || operation == 3 ? 19 : 1;
  arma::Mat<double> result(sides.n_rows, ncomponent);

  if (operation == 3) {
    // physics_ia_same_shape_3pcf takes every triangle in one C call,
    // so the adapter packs the sides into C layout first and unpacks
    // the results after the call returns.
    std::vector<double> triangles(sides.n_elem);
    std::vector<double> values(sides.n_rows*19); // [point,19], row-major

    // pack: Armadillo [point,3] column-major (point index fastest) to
    // C row-major, where the leg index runs fastest
    for (arma::uword point=0; point<sides.n_rows; point++) {
      for (arma::uword leg=0; leg<3; leg++) {
        triangles[point*3 + leg] = sides(point, leg);
      }
    }

    physics_ia_same_shape_3pcf(owner.get(), &parameters,
      static_cast<int>(sides.n_rows), triangles.data(), redshifts.memptr(),
      remove_alignment ? 1 : 0, values.data());

    // unpack: C row-major [point,19] back into the Armadillo matrix
    for (arma::uword point=0; point<sides.n_rows; point++) {
      // Warning: this loop index reuses the name component and hides
      // the function parameter component inside the loop body. The
      // parameter plays no role in operation 3, so the physics is
      // unchanged; keep the shadowing in mind when editing.
      for (int component=0; component<19; component++) {
        result(point, component) = values[point*19 + component];
      }
    }

    if (!(result.is_finite())) {
      spdlog::critical("3pcf physics wrapper: nonfinite common-shape IA");
      std::exit(1);
    }

    return result;
  }

  physics_projection_3pcf* projection = operation == 2
    ? physics_projector_create_3pcf(owner.get(), &parameters, bins.memptr())
    : nullptr;

  // Each iteration evaluates one physical triangle through the C
  // function that operation selects and copies its components into one
  // result row. The loop stays serial by design (see the header above).
  for (arma::uword point=0; point<sides.n_rows; point++) {
    double values[19]; // scratch row: up to the nineteen IA entries
    if (operation == 0) { // matter bispectrum
      values[0] = physics_matter_3pcf(owner.get(), sides(point, 0),
        sides(point, 1), sides(point, 2), redshifts(point), component,
        parameters.fb);
    } else if (operation == 1) { // nineteen IA bispectra
      physics_ia_bispectra_3pcf(owner.get(), &parameters, sides(point, 0),
        sides(point, 1), sides(point, 2), redshifts(point),
        remove_alignment ? 1 : 0, values);
    } else if (operation == 4) { // feedback bispectrum response
      const double triangle[3] = {sides(point, 0), sides(point, 1),
        sides(point, 2)};
      values[0] = physics_feedback_bispectrum_3pcf(owner.get(), triangle,
        redshifts(point), parameters.feedback_amplitude);
    } else { // operation 2: projection through the shared projector
      values[0] = physics_projected_3pcf(projection, sides(point, 0),
        sides(point, 1), sides(point, 2), component);
    }
    for (int index=0; index<ncomponent; index++) {
      if (!(std::isfinite(values[index]))) {
        spdlog::critical("3pcf physics wrapper: physical model returned a "
          "nonfinite value");
        std::exit(1);
      }
      result(point, index) = values[index];
    }
  }

  physics_projector_destroy_3pcf(projection);

  return result;
}


// ---------------------------------------------------------------------------
// Paired power response at each (k, z) pair, scaled by the shared
// amplitude; physics_feedback_power_3pcf in physics_3pcf.h holds the
// physics, and an amplitude of zero returns the DMO (dark-matter-only)
// value unchanged.
//
// Parameters:
//   k, z      - equal-length wavenumber and redshift arrays
//   amplitude - shared log-response amplitude
//
// Returns:
//   [n] positive power-response values
// ---------------------------------------------------------------------------
arma::Col<double> physics_feedback_power_cpp_3pcf(
  const physics_owner_3pcf& owner,
  const arma::Col<double>& k,
  const arma::Col<double>& z,
  const double amplitude
)
{
  if (!(k.n_elem == z.n_elem && k.n_elem > 0
        && k.is_finite() && z.is_finite())) {
    spdlog::critical("3pcf physics wrapper: power response axes disagree");
    std::exit(1);
  }

  arma::Col<double> result(k.n_elem);
  for (arma::uword index=0; index<k.n_elem; index++) {
    result[index] = physics_feedback_power_3pcf(owner.get(), k[index],
      z[index], amplitude);
  }

  return result;
}


// ---------------------------------------------------------------------------
// Direct Legendre multipoles b_L(ell1,ell2) of the projected
// bispectrum, through the mu quadrature of physics_multipoles_3pcf;
// physics_3pcf.h displays the defining integral.
//
// Parameters:
//   documented argument by argument in physics_wrapper.hpp
//
// Returns:
//   cube [lmax+1,nell,nell], checked finite entry by entry
// ---------------------------------------------------------------------------
arma::Cube<double> physics_multipoles_cpp_3pcf(
  const physics_owner_3pcf& owner,
  const arma::Col<double>& ell,
  const arma::Col<int>& bins,
  const physics_ia_3pcf& parameters,
  const int lmax,
  const int nmu,
  const int component
)
{
  if (!(ell.n_elem > 0 && ell.n_elem <= INT_MAX
        && bins.n_elem == 3 && lmax >= 0 && lmax < INT_MAX)) {
    spdlog::critical("3pcf physics wrapper: invalid multipole dimensions");
    std::exit(1);
  }

  const size_t nell = ell.n_elem;

  // Overflow guards before allocating [lmax+1,nell,nell]: first
  // nell*nell, then the full element count with lmax+1. The byte bound
  // uses SIZE_MAX/sizeof(double) because std::vector measures its
  // allocation with the unsigned size_t; the production twin in
  // physics_interface.cpp bounds the same count by NumPy's signed
  // ssize_t instead.
  if (!(nell <= SIZE_MAX/nell
        && static_cast<size_t>(lmax + 1) <= SIZE_MAX/(nell*nell)
        && static_cast<size_t>(lmax + 1)*nell*nell
        <= SIZE_MAX/sizeof(double))) {
    spdlog::critical("3pcf physics wrapper: multipole allocation size "
      "overflows");
    std::exit(1);
  }

  std::vector<double> buffer(static_cast<size_t>(lmax + 1)*nell*nell);

  physics_multipoles_3pcf(owner.get(), &parameters, bins.memptr(),
    static_cast<int>(nell), ell.memptr(), lmax, nmu, component, buffer.data());

  // The C buffer is row-major [L,ell1,ell2] with ell2 running fastest;
  // the Armadillo cube is column-major inside each slice. The loop
  // places each value one at a time and checks it finite on the way.
  arma::Cube<double> result(lmax + 1, nell, nell);
  for (int order=0; order<=lmax; order++) {
    for (size_t first=0; first<nell; first++) {
      for (size_t second=0; second<nell; second++) {
        const double value = buffer[(static_cast<size_t>(order)*nell
          + first)*nell + second];
        if (!(std::isfinite(value))) {
          spdlog::critical("3pcf physics wrapper: multipole returned a "
            "nonfinite value");
          std::exit(1);
        }
        result(order, first, second) = value;
      }
    }
  }

  return result;
}


// ---------------------------------------------------------------------------
// Full third-order aperture statistic by the direct (ell, phi)
// quadrature of physics_aperture_full_3pcf; physics_3pcf.h gives the
// compensated-Gaussian filter U. One value comes back per
// filter-radius triple.
//
// Parameters:
//   documented argument by argument in physics_wrapper.hpp; phi, dphi
//   and the filter radii are in radians
//
// Returns:
//   [nfilter] aperture means, checked finite
// ---------------------------------------------------------------------------
arma::Col<double> physics_aperture_full_cpp_3pcf(
  const physics_owner_3pcf& owner,
  const arma::Col<double>& ell,
  const arma::Col<double>& dell,
  const arma::Col<double>& phi,
  const arma::Col<double>& dphi,
  const arma::Mat<double>& filters,
  const arma::Col<int>& bins,
  const physics_ia_3pcf& parameters
)
{
  if (!(ell.n_elem > 0 && ell.n_elem <= INT_MAX
        && phi.n_elem > 0 && phi.n_elem <= INT_MAX
        && filters.n_rows > 0 && filters.n_rows <= INT_MAX
        && filters.n_cols == 3 && bins.n_elem == 3
        && dell.n_elem == ell.n_elem && dphi.n_elem == phi.n_elem)) {
    spdlog::critical("3pcf physics wrapper: full-aperture input shapes "
      "disagree");
    std::exit(1);
  }

  // pack the radii: Armadillo [nfilter,3] column-major to the C
  // row-major layout, where the leg index runs fastest
  std::vector<double> radii(filters.n_elem);
  for (arma::uword filter=0; filter<filters.n_rows; filter++) {
    for (arma::uword leg=0; leg<3; leg++) {
      radii[filter*3 + leg] = filters(filter, leg);
    }
  }

  arma::Col<double> result(filters.n_rows);
  physics_aperture_full_3pcf(owner.get(), &parameters, bins.memptr(),
    static_cast<int>(ell.n_elem), ell.memptr(), dell.memptr(),
    static_cast<int>(phi.n_elem), phi.memptr(), dphi.memptr(),
    static_cast<int>(filters.n_rows), radii.data(), result.memptr());

  if (!(result.is_finite())) {
    spdlog::critical("3pcf physics wrapper: nonfinite full-aperture mean");
    std::exit(1);
  }

  return result;
}


// ---------------------------------------------------------------------------
// Interpolated projected bispectrum: one read of the prepared grid
// table (physics_grid_3pcf.h) per row of multipole triangles.
//
// Parameters:
//   sides     - [point,3] angular Fourier triangles
//   component - total=0, NLA baseline=1, residual IA=2
//
// Returns:
//   [point] interpolated values, checked finite
// ---------------------------------------------------------------------------
arma::Col<double> physics_grid_projected_cpp_3pcf(
  const physics_grid_owner_3pcf& owner,
  const arma::Mat<double>& sides,
  const int component
)
{
  if (!(sides.n_rows > 0 && sides.n_rows <= INT_MAX
        && sides.n_cols == 3 && sides.is_finite())) {
    spdlog::critical("3pcf physics wrapper: invalid grid-query sides");
    std::exit(1);
  }

  arma::Col<double> result(sides.n_rows);
  for (arma::uword row=0; row<sides.n_rows; row++) {
    result(row) = physics_grid_projected_3pcf(owner.get(), sides(row, 0),
      sides(row, 1), sides(row, 2), component);
  }

  if (!(result.is_finite())) {
    spdlog::critical("3pcf physics wrapper: nonfinite interpolated bispectrum");
    std::exit(1);
  }

  return result;
}


// ---------------------------------------------------------------------------
// Interpolated Legendre multipoles b_L from the prepared grid table;
// the owner carries the Legendre cutoff the table was built with.
//
// Parameters:
//   ell       - Fourier sides of the b_L output grid
//   component - total=0, NLA baseline=1, residual IA=2
//
// Returns:
//   cube [lmax+1,nell,nell], checked finite
// ---------------------------------------------------------------------------
arma::Cube<double> physics_grid_multipoles_cpp_3pcf(
  const physics_grid_owner_3pcf& owner,
  const arma::Col<double>& ell,
  const int component
)
{
  if (!(ell.n_elem > 0 && ell.n_elem <= INT_MAX)) {
    spdlog::critical("3pcf physics wrapper: invalid interpolated-multipole "
      "dimensions");
    std::exit(1);
  }

  const size_t count = ell.n_elem;
  const size_t norder = static_cast<size_t>(owner.lmax() + 1);

  if (!(count <= SIZE_MAX/count
        && norder <= SIZE_MAX/(count*count)
        && norder*count*count <= SIZE_MAX/sizeof(double))) {
    spdlog::critical("3pcf physics wrapper: interpolated-multipole "
      "allocation overflows");
    std::exit(1);
  }

  std::vector<double> packed(norder*count*count);

  physics_grid_multipoles_3pcf(owner.get(), static_cast<int>(count),
    ell.memptr(), component, packed.data());

  // The C buffer is row-major [L,ell1,ell2] with ell2 running fastest;
  // the loop places each value into the column-major Armadillo cube.
  arma::Cube<double> result(norder, count, count);
  for (size_t order=0; order<norder; order++) {
    for (size_t first=0; first<count; first++) {
      for (size_t second=0; second<count; second++) {
        result(order, first, second) = packed[(order*count + first)*count
          + second];
      }
    }
  }

  if (!(result.is_finite())) {
    spdlog::critical("3pcf physics wrapper: nonfinite interpolated multipole");
    std::exit(1);
  }

  return result;
}


// ---------------------------------------------------------------------------
// Batch projection-grid factory: one prepared table per ordered source
// triplet. physics_grid_create_many_3pcf shares the triangle and
// redshift preparation across the whole batch, which is why building
// many tables together beats building them one by one.
//
// Parameters:
//   bins - [ntriplet,3] ordered source triplets; parameters and
//          settings apply to every table
//
// Returns:
//   one shared owner per triplet, in the input row order
// ---------------------------------------------------------------------------
std::vector<std::shared_ptr<physics_grid_owner_3pcf>>
physics_grids_cpp_3pcf(
  const physics_owner_3pcf& owner,
  const arma::Mat<int>& bins,
  const physics_ia_3pcf& parameters,
  const physics_grid_settings_3pcf& settings
)
{
  if (!(bins.n_rows > 0 && bins.n_rows <= INT_MAX
        && bins.n_cols == 3)) {
    spdlog::critical("3pcf physics wrapper: batch bins must have shape "
      "[ntriplet,3]");
    std::exit(1);
  }

  // pack the triplets: Armadillo [ntriplet,3] column-major to the C
  // row-major layout, where the leg index runs fastest
  std::vector<int> ordered(bins.n_elem);
  for (arma::uword row=0; row<bins.n_rows; row++) {
    for (arma::uword leg=0; leg<3; leg++) {
      ordered[row*3 + leg] = bins(row, leg);
    }
  }

  std::vector<physics_grid_3pcf*> prepared(bins.n_rows);
  physics_grid_create_many_3pcf(owner.get(), &parameters,
    static_cast<int>(bins.n_rows), ordered.data(), &settings,
    prepared.data());

  std::vector<std::shared_ptr<physics_grid_owner_3pcf>> result;
  result.reserve(bins.n_rows);
  for (physics_grid_3pcf* table : prepared) {
    result.push_back(std::make_shared<physics_grid_owner_3pcf>(table,
      settings.lmax));
  }

  return result;
}


// ---------------------------------------------------------------------------
// Aperture statistic from the prepared grid table: the same (ell, phi)
// quadrature as the direct route, reading interpolated values instead
// of line-of-sight projections.
//
// Parameters:
//   documented argument by argument in physics_wrapper.hpp; phi, dphi
//   and the filter radii are in radians
//
// Returns:
//   [nfilter] aperture means, checked finite
// ---------------------------------------------------------------------------
arma::Col<double> physics_grid_aperture_cpp_3pcf(
  const physics_grid_owner_3pcf& owner,
  const arma::Col<double>& ell,
  const arma::Col<double>& dell,
  const arma::Col<double>& phi,
  const arma::Col<double>& dphi,
  const arma::Mat<double>& filters
)
{
  // One chained check: both quadrature node arrays must be nonempty and
  // fit the C int counts; each filter row holds three radii; each
  // weight array matches its node array.
  if (!(ell.n_elem > 0 && ell.n_elem <= INT_MAX
        && phi.n_elem > 0 && phi.n_elem <= INT_MAX
        && filters.n_rows > 0 && filters.n_rows <= INT_MAX
        && filters.n_cols == 3 && dell.n_elem == ell.n_elem
        && dphi.n_elem == phi.n_elem)) {
    spdlog::critical("3pcf physics wrapper: grid aperture input shapes "
      "disagree");
    std::exit(1);
  }

  // pack the radii: Armadillo [nfilter,3] column-major to the C
  // row-major layout, where the leg index runs fastest
  std::vector<double> radii(filters.n_elem);
  for (arma::uword filter=0; filter<filters.n_rows; filter++) {
    for (arma::uword leg=0; leg<3; leg++) {
      radii[filter*3 + leg] = filters(filter, leg);
    }
  }

  arma::Col<double> result(filters.n_rows);
  physics_grid_aperture_full_3pcf(owner.get(), static_cast<int>(ell.n_elem),
    ell.memptr(), dell.memptr(), static_cast<int>(phi.n_elem), phi.memptr(),
    dphi.memptr(), static_cast<int>(filters.n_rows), radii.data(),
    result.memptr());

  if (!(result.is_finite())) {
    spdlog::critical("3pcf physics wrapper: nonfinite grid aperture mean");
    std::exit(1);
  }

  return result;
}


// ---------------------------------------------------------------------------
// Shear power spectrum C_ell with NLA kernels for one source-bin pair,
// through physics_nla_power_batch_3pcf; no TATT loop integrals enter
// this diagnostic.
//
// Parameters:
//   ell  - output multipoles of the shear C_ell
//   bins - [2] source bins of the power spectrum
//
// Returns:
//   [nell] C_ell values, checked finite
// ---------------------------------------------------------------------------
arma::Col<double> physics_nla_power_cpp_3pcf(
  const physics_owner_3pcf& owner,
  const arma::Col<double>& ell,
  const arma::Col<int>& bins,
  const physics_ia_3pcf& parameters
)
{
  if (!(ell.n_elem > 0 && ell.n_elem <= INT_MAX
        && ell.is_finite() && bins.n_elem == 2)) {
    spdlog::critical("3pcf physics wrapper: invalid NLA power arrays");
    std::exit(1);
  }

  arma::Col<double> result(ell.n_elem);
  physics_nla_power_batch_3pcf(owner.get(), &parameters, bins[0], bins[1],
    static_cast<int>(ell.n_elem), ell.memptr(), result.memptr());

  if (!(result.is_finite())) {
    spdlog::critical("3pcf physics wrapper: nonfinite NLA power result");
    std::exit(1);
  }

  return result;
}

}
