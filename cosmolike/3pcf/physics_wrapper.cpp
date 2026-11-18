#include <climits>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <spdlog/spdlog.h>
#include "physics_wrapper.hpp"

namespace cosmolike_interface {

// The notebook adapter receives private Armadillo arrays. Armadillo
// stores columns contiguously, whereas C expects its last axis to be
// contiguous. The loops here copy between those layouts; they neither
// evaluate a physical kernel nor choose an integration rule.
static void require_physics_cpp(
  const bool condition,
  const char* message
)
{
  if (!condition) {
    spdlog::critical("3pcf physics wrapper: {}", message);
    std::exit(1);
  }
}


physics_owner_3pcf::physics_owner_3pcf(const physics_input_3pcf& input)
  : model(physics_create_3pcf(&input), physics_destroy_3pcf)
{
}


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
  const physics_feedback_tables_cpp_3pcf* feedback
)
{
  const arma::uword nsource = source_z.n_elem ? source_z.n_elem : z.n_elem;
  require_physics_cpp(k.n_elem >= 4 && k.n_elem <= INT_MAX
    && z.n_elem >= 4 && z.n_elem <= INT_MAX
    && sources.n_rows > 0 && sources.n_rows <= INT_MAX
    && los_z.n_elem >= 2 && los_z.n_elem <= INT_MAX
    && nsource >= 4 && nsource <= INT_MAX,
    "invalid model dimensions");
  require_physics_cpp(p_linear0.n_elem == k.n_elem
    && p_nonlinear.n_rows == z.n_elem && p_nonlinear.n_cols == k.n_elem
    && chi.n_elem == z.n_elem && dz_dchi.n_elem == z.n_elem
    && growth.n_elem == z.n_elem && sources.n_cols == nsource,
    "model table shapes disagree");
  std::vector<double> nonlinear(p_nonlinear.n_elem);
  std::vector<double> source(sources.n_elem);
  for (arma::uword row=0; row<p_nonlinear.n_rows; row++) {
    for (arma::uword column=0; column<p_nonlinear.n_cols; column++) {
      nonlinear[row*p_nonlinear.n_cols + column] = p_nonlinear(row, column);
    }
  }
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
    require_physics_cpp(feedback->z.n_elem <= INT_MAX
      && feedback->logk.n_elem <= INT_MAX
      && feedback->logu.n_elem <= INT_MAX && feedback->v.n_elem <= INT_MAX,
      "feedback axis sizes exceed C integer storage");
    const size_t nz = feedback->z.n_elem;
    const size_t nk = feedback->logk.n_elem;
    const size_t nu = feedback->logu.n_elem;
    const size_t nv = feedback->v.n_elem;
    require_physics_cpp(nz >= 4 && nk >= 4 && nu >= 4 && nv >= 4
      && nz <= SIZE_MAX/nk && nz*nk <= SIZE_MAX/nu
      && nz*nk*nu <= SIZE_MAX/nv
      && feedback->log_power.n_elem == nz*nk
      && feedback->log_bispectrum.n_elem == nz*nk*nu*nv,
      "flattened paired feedback arrays disagree with their axes");
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
  return std::make_shared<physics_owner_3pcf>(input);
}


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
  require_physics_cpp(sides.n_rows > 0 && sides.n_rows <= INT_MAX
    && sides.n_cols == 3, "sides must have shape [point,3]");
  require_physics_cpp(operation >= 0 && operation <= 4,
    "unknown point operation");
  require_physics_cpp(operation == 2 ? bins.n_elem == 3
    : redshifts.n_elem == sides.n_rows, "point query shapes disagree");
  const int ncomponent = operation == 1 || operation == 3 ? 19 : 1;
  arma::Mat<double> result(sides.n_rows, ncomponent);
  if (operation == 3) {
    std::vector<double> triangles(sides.n_elem);
    std::vector<double> values(sides.n_rows*19);
    for (arma::uword point=0; point<sides.n_rows; point++) {
      for (arma::uword leg=0; leg<3; leg++) {
        triangles[point*3 + leg] = sides(point, leg);
      }
    }
    physics_ia_same_shape_3pcf(owner.get(), &parameters,
      static_cast<int>(sides.n_rows), triangles.data(), redshifts.memptr(),
      remove_alignment ? 1 : 0, values.data());
    for (arma::uword point=0; point<sides.n_rows; point++) {
      for (int component=0; component<19; component++) {
        result(point, component) = values[point*19 + component];
      }
    }
    require_physics_cpp(result.is_finite(), "nonfinite common-shape IA");
    return result;
  }
  physics_projection_3pcf* projection = operation == 2
    ? physics_projector_create_3pcf(owner.get(), &parameters, bins.memptr())
    : nullptr;
  for (arma::uword point=0; point<sides.n_rows; point++) {
    double values[19];
    if (operation == 0) {
      values[0] = physics_matter_3pcf(owner.get(), sides(point, 0),
        sides(point, 1), sides(point, 2), redshifts(point), component,
        parameters.fb);
    } else if (operation == 1) {
      physics_ia_bispectra_3pcf(owner.get(), &parameters, sides(point, 0),
        sides(point, 1), sides(point, 2), redshifts(point),
        remove_alignment ? 1 : 0, values);
    } else if (operation == 4) {
      const double triangle[3] = {sides(point, 0), sides(point, 1),
        sides(point, 2)};
      values[0] = physics_feedback_bispectrum_3pcf(owner.get(), triangle,
        redshifts(point), parameters.feedback_amplitude);
    } else {
      values[0] = physics_projected_3pcf(projection, sides(point, 0),
        sides(point, 1), sides(point, 2), component);
    }
    for (int index=0; index<ncomponent; index++) {
      require_physics_cpp(std::isfinite(values[index]),
        "physical model returned a nonfinite value");
      result(point, index) = values[index];
    }
  }
  physics_projector_destroy_3pcf(projection);
  return result;
}


arma::Col<double> physics_feedback_power_cpp_3pcf(
  const physics_owner_3pcf& owner,
  const arma::Col<double>& k,
  const arma::Col<double>& z,
  const double amplitude
)
{
  require_physics_cpp(k.n_elem == z.n_elem && k.n_elem > 0
    && k.is_finite() && z.is_finite(), "power response axes disagree");
  arma::Col<double> result(k.n_elem);
  for (arma::uword index=0; index<k.n_elem; index++) {
    result[index] = physics_feedback_power_3pcf(owner.get(), k[index],
      z[index], amplitude);
  }
  return result;
}


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
  require_physics_cpp(ell.n_elem > 0 && ell.n_elem <= INT_MAX
    && bins.n_elem == 3 && lmax >= 0 && lmax < INT_MAX,
    "invalid multipole dimensions");
  const size_t nell = ell.n_elem;
  require_physics_cpp(nell <= SIZE_MAX/nell
    && static_cast<size_t>(lmax + 1) <= SIZE_MAX/(nell*nell)
    && static_cast<size_t>(lmax + 1)*nell*nell <= SIZE_MAX/sizeof(double),
    "multipole allocation size overflows");
  std::vector<double> buffer(static_cast<size_t>(lmax + 1)*nell*nell);
  physics_multipoles_3pcf(owner.get(), &parameters, bins.memptr(),
    static_cast<int>(nell), ell.memptr(), lmax, nmu, component, buffer.data());
  arma::Cube<double> result(lmax + 1, nell, nell);
  for (int order=0; order<=lmax; order++) {
    for (size_t first=0; first<nell; first++) {
      for (size_t second=0; second<nell; second++) {
        const double value = buffer[(static_cast<size_t>(order)*nell
          + first)*nell + second];
        require_physics_cpp(std::isfinite(value),
          "multipole returned a nonfinite value");
        result(order, first, second) = value;
      }
    }
  }
  return result;
}


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
  require_physics_cpp(ell.n_elem > 0 && ell.n_elem <= INT_MAX
    && phi.n_elem > 0 && phi.n_elem <= INT_MAX
    && filters.n_rows > 0 && filters.n_rows <= INT_MAX
    && filters.n_cols == 3 && bins.n_elem == 3
    && dell.n_elem == ell.n_elem && dphi.n_elem == phi.n_elem,
    "full-aperture input shapes disagree");
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
  require_physics_cpp(result.is_finite(), "nonfinite full-aperture mean");
  return result;
}


arma::Col<double> physics_grid_projected_cpp_3pcf(
  const physics_grid_owner_3pcf& owner,
  const arma::Mat<double>& sides,
  const int component
)
{
  require_physics_cpp(sides.n_rows > 0 && sides.n_rows <= INT_MAX
    && sides.n_cols == 3 && sides.is_finite(), "invalid grid-query sides");
  arma::Col<double> result(sides.n_rows);
  for (arma::uword row=0; row<sides.n_rows; row++) {
    result(row) = physics_grid_projected_3pcf(owner.get(), sides(row, 0),
      sides(row, 1), sides(row, 2), component);
  }
  require_physics_cpp(result.is_finite(), "nonfinite interpolated bispectrum");
  return result;
}


arma::Cube<double> physics_grid_multipoles_cpp_3pcf(
  const physics_grid_owner_3pcf& owner,
  const arma::Col<double>& ell,
  const int component
)
{
  require_physics_cpp(ell.n_elem > 0 && ell.n_elem <= INT_MAX,
    "invalid interpolated-multipole dimensions");
  const size_t count = ell.n_elem;
  const size_t norder = static_cast<size_t>(owner.lmax() + 1);
  require_physics_cpp(count <= SIZE_MAX/count
    && norder <= SIZE_MAX/(count*count)
    && norder*count*count <= SIZE_MAX/sizeof(double),
    "interpolated-multipole allocation overflows");
  std::vector<double> packed(norder*count*count);
  physics_grid_multipoles_3pcf(owner.get(), static_cast<int>(count),
    ell.memptr(), component, packed.data());
  arma::Cube<double> result(norder, count, count);
  for (size_t order=0; order<norder; order++) {
    for (size_t first=0; first<count; first++) {
      for (size_t second=0; second<count; second++) {
        result(order, first, second) = packed[(order*count + first)*count
          + second];
      }
    }
  }
  require_physics_cpp(result.is_finite(), "nonfinite interpolated multipole");
  return result;
}


std::vector<std::shared_ptr<physics_grid_owner_3pcf>>
physics_grids_cpp_3pcf(
  const physics_owner_3pcf& owner,
  const arma::Mat<int>& bins,
  const physics_ia_3pcf& parameters,
  const physics_grid_settings_3pcf& settings
)
{
  require_physics_cpp(bins.n_rows > 0 && bins.n_rows <= INT_MAX
    && bins.n_cols == 3, "batch bins must have shape [ntriplet,3]");
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


arma::Col<double> physics_grid_aperture_cpp_3pcf(
  const physics_grid_owner_3pcf& owner,
  const arma::Col<double>& ell,
  const arma::Col<double>& dell,
  const arma::Col<double>& phi,
  const arma::Col<double>& dphi,
  const arma::Mat<double>& filters
)
{
  require_physics_cpp(ell.n_elem > 0 && ell.n_elem <= INT_MAX
    && phi.n_elem > 0 && phi.n_elem <= INT_MAX
    && filters.n_rows > 0 && filters.n_rows <= INT_MAX
    && filters.n_cols == 3 && dell.n_elem == ell.n_elem
    && dphi.n_elem == phi.n_elem, "grid aperture input shapes disagree");
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
  require_physics_cpp(result.is_finite(), "nonfinite grid aperture mean");
  return result;
}


arma::Col<double> physics_nla_power_cpp_3pcf(
  const physics_owner_3pcf& owner,
  const arma::Col<double>& ell,
  const arma::Col<int>& bins,
  const physics_ia_3pcf& parameters
)
{
  require_physics_cpp(ell.n_elem > 0 && ell.n_elem <= INT_MAX
    && ell.is_finite() && bins.n_elem == 2, "invalid NLA power arrays");
  arma::Col<double> result(ell.n_elem);
  physics_nla_power_batch_3pcf(owner.get(), &parameters, bins[0], bins[1],
    static_cast<int>(ell.n_elem), ell.memptr(), result.memptr());
  require_physics_cpp(result.is_finite(), "nonfinite NLA power result");
  return result;
}

}
