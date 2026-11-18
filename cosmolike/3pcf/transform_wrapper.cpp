#include <algorithm>
#include <climits>
#include <cstdlib>
#include <spdlog/spdlog.h>
#include "coupling_3pcf.h"
#include "fftlog_3pcf.h"
#include "transform_wrapper.hpp"

namespace cosmolike_interface {

// ---------------------------------------------------------------------------
// transform_wrapper.cpp: notebook adapters for the transform stage of the
// shear three-point pipeline - the angular coupling table, the
// double-Bessel FFTLog transform, the natural-component reconstruction
// and the prepared-grid aperture contraction.
//
// Each wrapper exists to bridge two storage orders. The C kernels read
// flat row-major buffers, where the last index changes fastest, while
// Armadillo stores its first index fastest. Every function here owns the
// Armadillo copies it receives from the pybind layer, packs them into
// explicit row-major work buffers, calls the same C routine the
// production path calls, and unpacks the row-major result. The physical
// meaning of each axis never changes; only the memory order does.
//
// Cache invalidation:
// These wrappers keep no state. Every call owns its packed buffers and
// its result, so a later call cannot replace values an earlier caller
// still holds. The one borrowed object, the transform plan, belongs to
// the notebook plan owner and carries only geometry and scratch.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Angular coupling table G_ML(psi). The C routine evaluates
//
//   G_ML(psi) = 4 pi integral_0^pi P_L(cos delta)
//               cos(2 beta_bar + M delta) d delta,
//
// where P_L is the Legendre polynomial of degree L, M is the angular
// Fourier mode, delta is the integration angle and beta_bar is the
// triangle phase that the coupling geometry fixes (coupling_3pcf.c
// derives it). The query variable psi is the opening angle
// arctan(ell2/ell1) in [0,pi/2]: it encodes the ratio of the two
// multipole legs, but it is itself an angle, not a ratio.
//
// npsi = 0 evaluates the exact finite Fourier sum at every query angle;
// npsi >= 2 builds a uniform table of that length and interpolates it
// linearly, which reproduces fastnc's tabulation convention (the
// reference table holds 200 nodes).
//
// Returns:
//   arma::Cube [mmax+1, lmax+1, nquery] of coupling values.
// ---------------------------------------------------------------------------
arma::Cube<double> coupling_cpp_3pcf(
    const arma::Col<double>& psi, // query angles
    const int lmax, // Legendre cutoff
    const int mmax, // Fourier cutoff
    const int npsi // interpolation grid length
  )
{
  // One guard covers three groups of requirements: the angle grid is
  // populated, counts within a C int and holds finite values; each
  // cutoff is nonnegative and leaves room for the +1 of its table axis;
  // and lmax + mmax stays within int, because the coupling's Fourier
  // sum reaches the combined harmonic M + L.
  if (!(psi.n_elem > 0 && psi.n_elem <= INT_MAX
        && psi.is_finite() && lmax >= 0 && lmax < INT_MAX-1
        && mmax >= 0 && mmax < INT_MAX-1
        && lmax <= INT_MAX-1-mmax)) {
    spdlog::critical("3pcf transform wrapper: invalid coupling input");
    std::exit(1);
  }

  const arma::uword count = (arma::uword)(mmax+1)*(lmax+1)*psi.n_elem;
  arma::Col<double> packed(count);
  legendre_coupling_3pcf(lmax, mmax, static_cast<int>(psi.n_elem),
      psi.memptr(), npsi, packed.memptr());
  arma::Cube<double> result(mmax+1, lmax+1, psi.n_elem);

  // The C output is row-major, the rule pack_cube_3pcf spells out in
  // threepcf_wrapper.cpp: cell (mode, degree, query) lives at flat
  // offset (mode*(lmax+1) + degree)*nquery + query. This loop reads
  // each cell at that offset and writes it into the Armadillo cube,
  // which arranges the same entries in its own storage order.
  for (int mode=0; mode<=mmax; mode++) {
    for (int degree=0; degree<=lmax; degree++) {
      for (arma::uword query=0; query<psi.n_elem; query++) {
        result(mode, degree, query) =
            packed(((arma::uword)mode*(lmax+1)+degree)*psi.n_elem+query);
      }
    }
  }

  if (!(result.is_finite())) {
    spdlog::critical("3pcf transform wrapper: nonfinite coupling");
    std::exit(1);
  }

  return result;
}


// ---------------------------------------------------------------------------
// Double-Bessel transform of a tabulated integrand. The C workspace
// evaluates
//
//   g(theta1, theta2) = integral dln(ell1) dln(ell2) f(ell1, ell2)
//                       J_order1(ell1 theta1) J_order2(ell2 theta2),
//
// where J_n is the cylindrical Bessel function of the first kind and f
// arrives sampled on the logarithmic grid ell. FFTLog evaluates this
// integral as follows: the bias nu is a power-law tilt divided out of
// the integrand so the remaining integral on the logarithmic grid
// becomes a convolution, which the FFT turns into a product of
// coefficients. The constant xy fixes where the output grid lands,
// through theta = xy/ell on the reversed grid. The optional taper
// suppresses the highest log-frequency coefficients, damping the
// ringing that the FFT's periodic assumption causes; window enables it
// and window_width sets the suppressed fraction. A positive dlnt
// averages each output over the radial bin [theta, theta e^dlnt] with
// measure theta dtheta instead of returning point values.
//
// Returns:
//   arma::Mat [nell, nell] of g on the theta = xy/reversed(ell) grid.
// ---------------------------------------------------------------------------
arma::Mat<double> bessel_cpp_3pcf(
    const arma::Col<double>& ell, // logarithmic Fourier grid
    const arma::Mat<double>& integrand, // dln ell integrand
    const int order1, // first Bessel order
    const int order2, // second Bessel order
    const double nu1, // first bias
    const double nu2, // second bias
    const double xy, // grid alignment
    const double dlnt, // radial-bin width
    const int padding, // zero padding
    const bool window, // taper enabled
    const double window_width // taper fraction
  )
{
  // One guard covers three groups of requirements: the Fourier grid is
  // populated and counts within a C int; the integrand is square on
  // that grid and every value, grid included, is finite; and both
  // Bessel orders are nonnegative.
  if (!(ell.n_elem > 0 && ell.n_elem <= INT_MAX
        && integrand.n_rows == ell.n_elem
        && integrand.n_cols == ell.n_elem && ell.is_finite()
        && integrand.is_finite() && order1 >= 0 && order2 >= 0)) {
    spdlog::critical("3pcf transform wrapper: invalid two-Bessel input");
    std::exit(1);
  }

  const int nell = static_cast<int>(ell.n_elem);

  // Create a one-shot workspace. The literal 1 is the worker count: one
  // scratch slice suffices because this helper fills, transforms and
  // frees the workspace within a single serial call.
  auto* work = fftlog_create_3pcf(nell, ell.memptr(),
      std::max(order1, order2), 1, nu1, nu2, xy, dlnt, padding, window,
      window_width);

  // Pack the integrand into the workspace's row-major input buffer:
  // cell (first, second) lives at first*nell + second, the same rule
  // pack_cube_3pcf spells out in threepcf_wrapper.cpp.
  double* packed = fftlog_input_3pcf(work, 0);
  for (int first=0; first<nell; first++) {
    for (int second=0; second<nell; second++) {
      packed[(size_t)first*nell+second] = integrand(first, second);
    }
  }

  // Forward pass: compute the FFT coefficients of the biased integrand.
  fftlog_forward_3pcf(work, 0);

  // Inverse pass: contract those coefficients with the Bessel kernels
  // of the requested orders and land on the theta grid.
  const double* output = fftlog_inverse_3pcf(work, 0, order1, order2);

  // Unpack the row-major [nell,nell] result into an Armadillo matrix,
  // reversing the packing rule above cell by cell.
  arma::Mat<double> result(nell, nell);
  for (int first=0; first<nell; first++) {
    for (int second=0; second<nell; second++) {
      result(first, second) = output[(size_t)first*nell+second];
    }
  }

  // Release the workspace before validating, so an abort cannot leak it.
  fftlog_destroy_3pcf(work);

  if (!(result.is_finite())) {
    spdlog::critical("3pcf transform wrapper: nonfinite transform");
    std::exit(1);
  }

  return result;
}


// ---------------------------------------------------------------------------
// Natural components from bispectrum multipoles, on a prepared plan.
// The input cube holds b_L, the Legendre coefficients in mu of the
// projected E-mode bispectrum; mu is the cosine in the triangle closure
// ell3^2 = ell1^2 + ell2^2 - 2 ell1 ell2 mu. The C transform turns them
// into the four natural components Gamma_0..Gamma_3 of the shear
// three-point function: the complex correlators that differ by which of
// the three shears is conjugated (Gamma_0 none, Gamma_i the shear at
// vertex i; Schneider & Lombardi, arXiv:astro-ph/0207454). The result
// travels split-complex: real and imaginary parts in two real cubes.
//
// The output pair axis has length ntheta*ntheta and packs the two
// radial bins (first, second) as first*ntheta + second; the caller
// splits it back into two axes.
// ---------------------------------------------------------------------------
natural_result_cpp_3pcf natural_cpp_3pcf(
    transform_plan_3pcf* plan, // prepared geometry
    const arma::Cube<double>& multipoles, // [L,nell,nell]
    const int ntheta, // count of radial bins the plan keeps
    const int nphi // output angular count
  )
{
  if (!(multipoles.is_finite())) {
    spdlog::critical("3pcf transform wrapper: nonfinite bispectrum multipoles");
    std::exit(1);
  }

  // Pack b_L into the row-major order the C transform reads: cell
  // (degree, first, second) lives at (degree*nell + first)*nell +
  // second, the rule pack_cube_3pcf spells out in threepcf_wrapper.cpp.
  arma::Col<double> packed(multipoles.n_elem);
  for (arma::uword degree=0; degree<multipoles.n_rows; degree++) {
    for (arma::uword first=0; first<multipoles.n_cols; first++) {
      for (arma::uword second=0; second<multipoles.n_slices; second++) {
        const arma::uword cell =
            (degree*multipoles.n_cols+first)*multipoles.n_slices+second;
        packed(cell) = multipoles(degree, first, second);
      }
    }
  }

  const arma::uword pairs = (arma::uword)ntheta*ntheta;
  const arma::uword triangles = pairs*nphi;
  arma::Col<double> real(4*triangles);
  arma::Col<double> imag(4*triangles);

  transform_natural_3pcf(plan, packed.memptr(), real.memptr(),
      imag.memptr());

  natural_result_cpp_3pcf result = {
    arma::Cube<double>(4, pairs, nphi),
    arma::Cube<double>(4, pairs, nphi)
  };

  // Unpack the row-major C output [4, ntheta*ntheta, nphi]: cell
  // (component, pair, angle) lives at (component*pairs + pair)*nphi +
  // angle, again pack_cube_3pcf's storage rule read in reverse.
  for (int component=0; component<4; component++) {
    for (arma::uword pair=0; pair<pairs; pair++) {
      for (int angle=0; angle<nphi; angle++) {
        const arma::uword cell = (component*pairs+pair)*nphi+angle;
        result.real(component, pair, angle) = real(cell);
        result.imag(component, pair, angle) = imag(cell);
      }
    }
  }

  if (!(result.real.is_finite()
        && result.imag.is_finite())) {
    spdlog::critical("3pcf transform wrapper: nonfinite natural components");
    std::exit(1);
  }

  return result;
}


// ---------------------------------------------------------------------------
// Aperture moments from a prepared physical grid. The plan reads fresh
// bispectrum multipoles from the grid, transforms them to natural
// components and contracts them with the caller's aperture weights, all
// inside the C engine; this wrapper only validates the weights and
// converts their storage order. The weight cubes carry the real and
// imaginary parts of the integration weights for the four natural
// components on every triangle bin.
// ---------------------------------------------------------------------------
arma::Col<double> aperture_grid_cpp_3pcf(
    transform_plan_3pcf* plan,
    const physics_grid_3pcf* grid,
    const arma::Cube<double>& real,
    const arma::Cube<double>& imag,
    const int ntriangle
  )
{
  // One guard covers four groups of requirements: the plan and grid
  // exist; the filter axis is populated and counts within a C int; the
  // component axis holds the four natural components and the triangle
  // axis matches the plan the caller prepared; and the imaginary cube
  // matches the real cube in shape, with every value finite.
  if (!(plan && grid && real.n_rows > 0
        && real.n_rows <= INT_MAX && real.n_cols == 4 && ntriangle > 0
        && real.n_slices == static_cast<arma::uword>(ntriangle)
        && imag.n_rows == real.n_rows && imag.n_cols == real.n_cols
        && imag.n_slices == real.n_slices && real.is_finite()
        && imag.is_finite())) {
    spdlog::critical("3pcf transform wrapper: invalid notebook "
      "prepared-grid aperture weights");
    std::exit(1);
  }

  // Armadillo stores its first axis contiguously, and a notebook cube
  // may arrive strided or read-only. Pack the weights explicitly into
  // the row-major order the C engine reads - pack_cube_3pcf's rule in
  // threepcf_wrapper.cpp - so the identical C kernel serves both routes.
  arma::Col<double> packed_real(real.n_elem);
  arma::Col<double> packed_imag(imag.n_elem);
  for (arma::uword filter=0; filter<real.n_rows; filter++) {
    for (int component=0; component<4; component++) {
      for (int triangle=0; triangle<ntriangle; triangle++) {
        const arma::uword cell = (filter*4 + component)*ntriangle + triangle;
        packed_real(cell) = real(filter, component, triangle);
        packed_imag(cell) = imag(filter, component, triangle);
      }
    }
  }

  arma::Col<double> result(real.n_rows);
  transform_aperture_grid_3pcf(plan, grid, static_cast<int>(real.n_rows),
      packed_real.memptr(), packed_imag.memptr(), result.memptr());

  if (!(result.is_finite())) {
    spdlog::critical("3pcf transform wrapper: nonfinite aperture moment");
    std::exit(1);
  }

  return result;
}

}
