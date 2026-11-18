"""Fisher forecasting helpers shared by the EXAMPLE_EVALUATE notebooks.

A Fisher matrix forecasts parameter uncertainties from the
derivatives of the theory data vector. Near the fiducial parameter
vector p0, expand the predicted data vector mu to first order,
mu(p0 + dp) ~ mu0 + D dp, with D the (n_data, n_params) matrix
whose column p is the derivative of mu with respect to parameter p.
For a Gaussian data likelihood with a covariance C that is held
fixed (no parameter dependence of C), the quadratic part of the
change of -2 ln L about the fiducial is then dp^T F dp with

    F = D^T C^-1 D,

the Fisher matrix: it measures how strongly parameter changes move
the predictions relative to their correlated measurement
uncertainty. Entry F_ij carries the reciprocal units of parameter i
times parameter j. The forecast is local (first order about the
fiducial), differentiates no covariance, and knows nothing of flat
prior walls; within it, the inverse F^-1 is the forecast parameter
covariance. Two different errors follow for one parameter p:
1/sqrt(F_pp) is the error with every other parameter held fixed,
while sqrt((F^-1)_pp), from the diagonal of the inverse, is the
marginalized error after every other parameter (nuisances included)
is free to move; the second is what an analysis quotes. Everything in
this module is pure numpy/getdist mathematics except the data vector
itself, which needs a project's compiled cosmolike interface. That
dependency arrives as an explicit argument, never by replacing a
function at run time (a monkey patch): get_ddv, get_ddv_dkit and
get_Fisher2 take the notebook's data-vector function as their first
argument (`dv`, called as dv(param=<1D parameter array>,
AccuracyBoost=<float>) and returning the theory vector), and
get_Fisher takes a derivative function built from it (`ddv`).
This module therefore never imports a cosmolike interface, and every
project shares it; each project binds its own dv, fiducial values,
priors, and covariance in thin wrappers
(interface/cosmolike_<project>_notebook_wrappers.py).

Contents:
  get_ddv             five-point finite-difference derivative of dv
  get_ddv_dkit        the same derivative via the derivkit package
  get_Fisher          F = D^T C^-1 D from a derivative function
  get_Fisher2         the same, with derivkit derivatives
  add_gaussian_priors add 1/sigma^2 to the diagonal for Gaussian
                      priors
  get_FoM             figure of merit of one parameter pair
  get_stencil_noise   numerical noise of one evaluation along one
                      parameter, measured from the five stencil points
                      (fourth difference), the spurious Fisher prior a
                      it implies, and the five-point derivative from the
                      same evaluations
  get_fom_noise_weights
                      R_j^2, sigma_j^2 and the lever of every column of
                      F: how strongly the noise of column j reaches the
                      FoM of one pair
  predict_fom_noise   change of the FoM of one pair from diagonal
                      additions a_j to F, to first order and exactly
  plot_Fisher         getdist triangle plot of Fisher contours,
                      optionally against MCMC chains
  plot_Fisher_amplitude_cut
                      the amplitude-bias prior box with the region
                      A_s b_i^2/(A_s b_i^2)_fid > cut shaded and the
                      Fisher 1 and 2 sigma contours about the fiducial
  plot_Fisher_prior_walls
                      Fisher 1 and 2 sigma contours against the walls of
                      a flat prior (boxes and linear walls such as
                      w0 + wa), the excluded region shaded
"""

import itertools

import numpy as np
from getdist import plots
from getdist.mcsamples import MCSamples, loadMCSamples


def get_ddv(dv, index=0, h=0.02, CV=None, AccuracyBoost=1.0):
    """Differentiate the data vector along one parameter (5-point rule).

    Four evaluations at +-h and +-2h around the fiducial, combined
    with the standard five-point-stencil weights (the center point
    has weight zero, so it is never evaluated). The step is relative
    (p0 * (1 + s)) so one h suits parameters of any magnitude; a
    parameter whose fiducial is zero at the 1e-12 absolute tolerance
    (|p0| <= 1e-12) takes absolute steps instead. One test,
    np.isclose(p0, 0.0, atol=1e-12), selects both the steps and the
    final divisor (h for absolute steps, h * p0 for relative ones),
    so every fiducial gets a consistent five-point derivative: a
    tiny nonzero fiducial below the tolerance is differentiated with
    absolute steps, exactly as a fiducial at 0.0 is.

    Arguments:
      dv    = the data-vector function, called as
              dv(param=<1D array>, AccuracyBoost=AccuracyBoost).
      index = position of the differentiated parameter inside CV.
      h     = relative step size (0.02 = two percent); the absolute
              step when |p0| <= 1e-12. Must be > 0 (assert).
      CV    = 1D array of fiducial parameter values (required; the
              None default is a placeholder). Never modified: each
              evaluation works on a copy.
      AccuracyBoost = forwarded to every dv evaluation.

    Returns:
      1D array: d(data vector)/d(parameter) at the fiducial.
    """
    assert h > 0.0, "h must be > 0"
    # the five-point-stencil weights (-1, 8, -8, 1)/12 paired with
    # the steps (2h, h, -h, -2h); zip walks the two arrays in step,
    # handing one (step, weight) pair per loop turn
    coeffs = np.array([-1.0/12.0, 8.0/12.0, -8.0/12.0, +1.0/12.0], dtype=np.float64)
    steps  = np.array([2*h, h, -h, -2*h], dtype=np.float64)
    p0 = float(CV[index])
    absolute_step = np.isclose(p0, 0.0, atol=1e-12)
    result = None
    for s, c in zip(steps, coeffs):
        # copy=True: p is an independent copy, so writing p[index]
        # below cannot touch the caller's CV array
        p = np.array(CV, dtype = np.float64, copy=True)
        # ternary a if cond else b: absolute step when the fiducial
        # is (numerically) zero, relative step otherwise
        p[index] = p0 + s if absolute_step else p0 * (1.0 + s)
        vec = dv(param=p, AccuracyBoost=AccuracyBoost)
        # accumulate c * vec; the first turn initializes result
        result = c*vec if result is None else result + c*vec
    # divide by the step scale: h for absolute steps, h * p0 for
    # relative ones. absolute_step is the same test the loop used to
    # pick the steps, so divisor and steps always describe one and
    # the same stencil.
    return result / (h if absolute_step else h * p0)


def get_ddv_dkit(dv, index=0, CV=None, AccuracyBoost=1.0,
                 min_samples=7, fallback_mode="poly_at_floor"):
    """Differentiate the data vector along one parameter with derivkit.

    derivkit fits a polynomial of degree equal to the derivative
    order (a straight line for a first derivative) through
    adaptively chosen sample points and differentiates the fit:
    numerical noise in the data vector averages out over the fitted
    points instead of entering the derivative at full strength, as
    it does through the fixed step of a finite-difference stencil.
    Its inverse-distance weights pin the line to the center sample,
    so the slope is the mean of the symmetric central differences;
    points are pruned only when a relative residual exceeds the 5%
    fit tolerance, far above the residuals smooth data vectors
    produce.

    Arguments:
      dv    = the data-vector function (see get_ddv).
      index = position of the differentiated parameter inside CV.
      CV    = 1D array of fiducial parameter values (required; the
              None default is a placeholder). Never modified.
      AccuracyBoost = forwarded to every dv evaluation.
      min_samples   = minimum number of sample points for the fit.
      fallback_mode = derivkit's strategy when the fit rejects the
              samples (see the derivkit documentation).

    Returns:
      1D array: d(data vector)/d(parameter) at the fiducial.
    """
    # imported here, not at module top: derivkit is optional and only
    # this function and get_Fisher2 need it
    from derivkit import adaptive_fit

    p0 = float(CV[index])
    # func is a one-argument closure over CV and dv: derivkit varies
    # x, and func rebuilds the full parameter array around it
    def func(x):
        p = np.array(CV, dtype = np.float64, copy=True)
        p[index] = x
        return dv(param=p, AccuracyBoost=AccuracyBoost)
    dk = adaptive_fit.AdaptiveFitDerivative(function=func, central_value=p0)
    return dk.compute(diagnostics=False, min_samples=min_samples, fallback_mode=fallback_mode)


def add_gaussian_priors(F, priors):
    """Add Gaussian priors to a Fisher matrix.

    A Gaussian prior of width sigma on parameter i adds 1/sigma^2 to
    F[i, i] and nothing else.

    Arguments:
      F      = Fisher matrix (2D square array), not modified.
      priors = {parameter index: (mean, sigma)}; only sigma enters a
               Fisher matrix (the mean shifts no curvature).

    Returns:
      a copy of the Fisher matrix with the priors added; the input
      matrix is not modified.
    """
    Fp = F.copy()
    # np.fromiter builds an array straight from an iterator:
    # priors.keys() yields the parameter indices, and the generator
    # (v[1] for v in priors.values()) yields each (mean, sigma)
    # tuple's second entry, the sigma
    idxs = np.fromiter(priors.keys(), dtype=int)
    sigmas = np.fromiter((v[1] for v in priors.values()), dtype=float)
    # np.add.at adds in place at the listed positions; the index pair
    # (idxs, idxs) targets the diagonal entries (i, i) only
    np.add.at(Fp, (idxs, idxs), 1.0 / (sigmas ** 2))
    return Fp


def get_Fisher(ddv, CV=None, h=0.02, AccuracyBoost=3.1, priors=None,
               invcov=None):
    """Assemble F = D^T C^-1 D from a derivative function.

    The module docstring derives the expression: the local Gaussian
    forecast with a fixed data covariance, valid to first order
    about the fiducial CV. For marginalized errors invert the full
    returned matrix; the diagonal of F itself gives fixed-parameter
    errors only.

    Arguments:
      ddv   = derivative function, called as
              ddv(index=p, h=h, CV=CV, AccuracyBoost=AccuracyBoost)
              for every parameter p (the project wrappers pass their
              own get_ddv, which calls this module's get_ddv with the
              project's dv).
      CV    = 1D array of fiducial parameter values (required); its
              length sets the number of columns of D.
      h     = relative step forwarded to ddv.
      AccuracyBoost = forwarded to ddv (derivatives want accurate
              vectors: differences amplify numerical noise).
      priors  = None, or {index: (mean, sigma)} Gaussian priors
              added on the diagonal.
      invcov  = inverse covariance of the data vector, a 2D
              (n_data, n_data) array matching the length of the
              vectors ddv returns (required).

    Returns:
      the Fisher matrix (n_params x n_params array).
    """
    # D[a, p] = d mu_a / d p, shape (n_data, n_params). The
    # comprehension evaluates ddv once per parameter (four
    # data-vector evaluations per call with get_ddv); np.column_stack
    # glues the returned 1D arrays side by side as the columns of D
    D = np.column_stack([ddv(index=p, h=h, CV=CV, AccuracyBoost=AccuracyBoost) for p in list(range(len(CV)))]).astype(np.float64)
    # @ is numpy matrix multiplication: D^T (C^-1 D). The inverse
    # covariance weights coherent changes of the predictions: two
    # parameters with nearly parallel derivative columns move the
    # data the same way, so the data constrain their combination
    # rather than each one (the degeneracy shows up as nearly
    # dependent columns of D, as a small eigenvalue of F, and as
    # large entries of F^-1). How correlations between data entries
    # change the information depends on the direction of the
    # response: at fixed per-entry variances, positive correlation
    # weakens what a common-mode response measures and sharpens
    # what a difference-mode response measures. Visual curve
    # separation in the data-vector plots measures none of this.
    F = D.T @ (invcov @ D)
    if priors is not None:
        F = add_gaussian_priors(F=F, priors=priors)
    return F


def get_Fisher2(dv, CV=None, AccuracyBoost=3.0, priors=None,
                invcov=None, min_samples=7,
                fallback_mode="poly_at_floor"):
    """Assemble F = D^T C^-1 D with derivkit derivatives.

    Arguments:
      dv    = the data-vector function (see get_ddv); the derivkit
              derivative is built from it per parameter.
      CV    = 1D array of fiducial parameter values (required).
      AccuracyBoost = forwarded to every dv evaluation.
      priors  = None, or {index: (mean, sigma)} Gaussian priors.
      invcov  = inverse covariance of the data vector, a 2D
              (n_data, n_data) array (required).
      min_samples, fallback_mode = forwarded to get_ddv_dkit.

    Returns:
      the Fisher matrix (n_params x n_params array).
    """
    # one derivkit derivative per parameter, glued as the columns of
    # D as in get_Fisher
    D = np.column_stack([get_ddv_dkit(dv, index=p, CV=CV, AccuracyBoost=AccuracyBoost, min_samples=min_samples, fallback_mode=fallback_mode) for p in list(range(len(CV)))]).astype(np.float64)
    F = D.T @ (invcov @ D)
    if priors is not None:
        F = add_gaussian_priors(F=F, priors=priors)
    return F


def get_FoM(i, j, F):
    """Compute the figure of merit of the parameter pair (i, j).

    FoM = 1/sqrt(det C_ij), with C_ij the 2x2 block of the parameter
    covariance (the inverse of the full Fisher matrix) for
    parameters i and j: a larger FoM means a smaller error ellipse.
    Inverting the full matrix first marginalizes every other
    parameter of F (cosmology and nuisances alike), so the block is
    the marginalized pair covariance, not the inverse of the pair's
    2x2 Fisher block. The FoM is proportional to the inverse area of
    the pair's error ellipse, with the probability-contour constant
    (the delta chi2 of a chosen confidence level) left out, and it
    carries the reciprocal units of parameter i times parameter j:
    values compare only between forecasts of the same two parameters
    in the same normalization with a stated nuisance set. The
    forecast requires a nonsingular, positive-definite Fisher
    matrix.

    Arguments:
      i, j = the two parameter indices.
      F    = Fisher matrix.

    Returns:
      the figure of merit as a float. A returned value always comes
      from a positive-definite covariance block: the eigenvalue
      check raises before anything else is computed.

    Raises:
      ValueError when the marginalized 2x2 block is not positive
      definite (any eigenvalue <= 0): such a block bounds no error
      ellipse, so no figure of merit exists for it. The test is on
      the eigenvalues, not on the determinant, because a block with
      two negative eigenvalues also has det C_ij > 0: a positive
      determinant alone does not establish a positive-definite
      block.
    """
    # 0.5 * (F + F^T) symmetrizes away numerical asymmetry before
    # inverting; np.ix_([i, j], [i, j]) selects the 2x2 block with
    # rows and columns (i, j)
    C = np.linalg.inv(0.5 * (F + F.T))
    C = C[np.ix_([i, j], [i, j])]
    eigenvalues = np.linalg.eigvalsh(C)
    if np.any(eigenvalues <= 0):
        raise ValueError(
            "get_FoM: the marginalized covariance block of parameters "
            f"({i}, {j}) has eigenvalues {eigenvalues}; a figure of merit "
            "needs a positive-definite block (both eigenvalues > 0). The "
            "Fisher matrix is singular or indefinite: check its "
            "derivatives, priors and parameter list")
    detC = np.linalg.det(C)
    if detC <= 0:
        # both eigenvalues are positive here (the check above raised
        # otherwise), so a nonpositive np.linalg.det can only be
        # round-off of its LU factorization on a nearly singular
        # block. Recompute the determinant as the product of the
        # eigvalsh eigenvalues (eigvalsh = eigenvalues of a symmetric
        # matrix); the clip is a no-op on this all-positive spectrum
        # and merely keeps the product nonnegative
        w = np.clip(np.linalg.eigvalsh(C), 0, None)
        detC = w.prod()
    return float(1.0/np.sqrt(detC))


def get_stencil_noise(dv, index=0, h=0.02, CV=None, AccuracyBoost=1.0,
                      invcov=None):
    """Measure the numerical noise of one derivative column from its stencil.

    A computed data vector is the smooth model plus a small numerical
    error (integration grids, distance grids, CAMB tolerances) that
    changes irregularly from one parameter value to the next. A finite
    difference divides that error by the step, and the Fisher matrix
    F = D^T C^-1 D built from noisy columns exceeds the true one on its
    diagonal, on average: noise looks like information. This function
    measures the error of one evaluation along the parameter
    p = CV[index] from the five equally spaced points

        p0 + k eps,   k = -2, -1, 0, 1, 2,

    with the step convention of get_ddv (eps = h p0 for a relative
    step, eps = h when |p0| <= 1e-12), and it also returns the
    five-point derivative from the same evaluations: a caller that
    wants both pays five evaluations, not nine.

    The estimate (the Fisher note of the lssty1xdesi project,
    projects/lssty1xdesi/notes/fisher_machinery, section "Derivative
    accuracy") takes three steps:
      - the fourth difference
            Delta4 m = m(-2) - 4 m(-1) + 6 m(0) - 4 m(1) + m(2)
        removes the smooth part of the model: expanding each point in a
        Taylor series, every derivative below the fourth cancels, so a
        smooth vector leaves only eps^4 m''''(p0) + O(eps^6), which the
        few-percent steps used here suppress to a negligible size.
        Noise that is independent from point to point, with a common
        size, passes with 1 + 16 + 36 + 16 + 1 = 70 times the variance
        of one point;
      - whitening (transforming a vector so its errors become unit and
        uncorrelated; its squared length is then a chi^2) gives the
        noise of one evaluation in the data's own error bars,
            sigma = sqrt(Delta4 m^T C^-1 Delta4 m / 70),
        in sqrt(chi^2) units;
      - the noise the stencil leaves in the derivative column,
            e = (n(-2) - 8 n(-1) + 8 n(1) - n(2))/(12 eps),
        adds on average e^T C^-1 e to F[index, index]; its four
        independent errors add through the squared weights
        1 + 64 + 64 + 1 = 130, so
            a = <e^T C^-1 e> = (130/144) (sigma/eps)^2.
        Off the diagonal nothing is added on average, so the noise acts
        exactly like a Gaussian prior of width a^-1/2 on p -- a prior
        the data do not carry. predict_fom_noise turns a into a change
        of a figure of merit.

    What the estimate does not see: an error that varies smoothly
    across the five points, or that is odd about p0, cancels in
    Delta4 m but can still bias the derivative; errors correlated
    between neighbouring points partly cancel and are under-counted.
    A small sigma therefore excludes the point-to-point jitter only,
    not every numerical error.

    Arguments:
      dv    = the data-vector function, called as
              dv(param=<1D array>, AccuracyBoost=AccuracyBoost); it must
              return a finite 1D array of length n_data (the size of
              invcov).
      index = position of the parameter inside CV, 0 <= index < len(CV).
      h     = relative step (0.02 = two percent); the absolute step when
              |p0| <= 1e-12. Finite and > 0.
      CV    = 1D array of fiducial parameter values (required; the None
              default is a placeholder). Never modified: each evaluation
              works on a copy.
      AccuracyBoost = forwarded to every dv evaluation.
      invcov = inverse covariance of the data vector, a 2D
              (n_data, n_data) array (required); it sets the units of
              sigma. A masked inverse covariance whose masked rows and
              columns are zero (positive semi-definite) is accepted.

    Returns:
      dict with
        "sigma" = float >= 0, the whitened noise of one evaluation, in
                  sqrt(chi^2) units;
        "eps"   = float > 0, the step in the parameter's own units:
                  h |p0| for a relative step, h for an absolute one;
        "a"     = float >= 0, (130/144) (sigma/eps)^2, the average
                  spurious addition to F[index, index], in one over the
                  parameter's unit squared (the units of F_pp);
        "ddv"   = 1D array (n_data,), the five-point derivative
                  d(data vector)/d(parameter) from the four off-center
                  evaluations, computed with get_ddv's steps, weights
                  and accumulation order, so for the same dv it equals
                  get_ddv's result bit for bit.
      The call evaluates dv five times: the four points of get_ddv,
      then the center (which the derivative does not use; only the
      fourth difference reads it).

    Raises:
      ValueError when CV or invcov is missing or malformed, when index
      is not an integer inside CV, when h is not finite and positive,
      when a dv result is not a finite 1D array of length n_data, or
      when Delta4 m^T C^-1 Delta4 m is negative (invcov is not positive
      semi-definite).
    """
    if CV is None:
        raise ValueError(
            "get_stencil_noise: CV is None; pass the 1D array of fiducial "
            "parameter values")
    fiducial = np.asarray(CV, dtype=np.float64)
    if fiducial.ndim != 1:
        raise ValueError(
            f"get_stencil_noise: CV has shape {fiducial.shape}; it must be "
            "a 1D array of fiducial parameter values")
    if not np.all(np.isfinite(fiducial)):
        raise ValueError(
            f"get_stencil_noise: CV = {fiducial} has a nonfinite entry; "
            "every fiducial value must be finite")
    if not isinstance(index, (int, np.integer)):
        raise ValueError(
            f"get_stencil_noise: index = {index!r} is not an integer; pass "
            "the position of the parameter inside CV")
    if index < 0 or index >= len(fiducial):
        raise ValueError(
            f"get_stencil_noise: index = {index} is outside CV, which has "
            f"{len(fiducial)} entries; pass 0 <= index < {len(fiducial)}")
    if not isinstance(h, (int, float, np.integer, np.floating)):
        raise ValueError(
            f"get_stencil_noise: h = {h!r} is not a number; pass the "
            "relative step as a float (0.02 = two percent)")
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError(
            f"get_stencil_noise: h = {h}; the step must be finite and > 0 "
            "(0.02 = a two-percent relative step)")
    if invcov is None:
        raise ValueError(
            "get_stencil_noise: invcov is None; pass the inverse covariance "
            "of the data vector, which sets the units of the noise")
    inverse_covariance = np.asarray(invcov, dtype=np.float64)
    if (inverse_covariance.ndim != 2
            or inverse_covariance.shape[0] != inverse_covariance.shape[1]):
        raise ValueError(
            f"get_stencil_noise: invcov has shape {inverse_covariance.shape}; "
            "it must be a square (n_data, n_data) matrix")
    if not np.all(np.isfinite(inverse_covariance)):
        raise ValueError(
            "get_stencil_noise: invcov has a nonfinite entry; pass a finite "
            "inverse covariance")
    n_data = inverse_covariance.shape[0]

    def evaluate(param):
        """Call dv at one point and check its result (shape, finiteness)."""
        vector = np.asarray(dv(param=param, AccuracyBoost=AccuracyBoost),
                            dtype=np.float64)
        if vector.ndim != 1 or vector.shape[0] != n_data:
            raise ValueError(
                f"get_stencil_noise: dv returned shape {vector.shape} at "
                f"param[{index}] = {param[index]}, but invcov is "
                f"({n_data}, {n_data}); dv must return the 1D data vector "
                "that invcov describes")
        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"get_stencil_noise: dv returned a nonfinite entry at "
                f"param[{index}] = {param[index]}; the data vector must be "
                "finite at every stencil point")
        return vector

    # the four off-center points of get_ddv, in its order and with its
    # arithmetic (the same steps, the same weights, the same
    # accumulation), so the derivative below equals get_ddv's bit for
    # bit: steps (2h, h, -h, -2h), derivative weights (-1, 8, -8, 1)/12.
    # The fourth difference takes the same four vectors with the weights
    # (1, -4, -4, 1); the center enters it alone, with weight 6.
    coeffs = np.array([-1.0/12.0, 8.0/12.0, -8.0/12.0, +1.0/12.0],
                      dtype=np.float64)
    steps  = np.array([2*h, h, -h, -2*h], dtype=np.float64)
    fourth_weights = np.array([1.0, -4.0, -4.0, 1.0], dtype=np.float64)
    p0 = float(fiducial[index])
    absolute_step = np.isclose(p0, 0.0, atol=1e-12)
    derivative = None
    fourth_difference = None
    # zip walks the three arrays in step, handing one triple (step,
    # derivative weight, fourth-difference weight) per loop turn
    for s, c, c4 in zip(steps, coeffs, fourth_weights):
        p = np.array(fiducial, dtype=np.float64, copy=True)
        p[index] = p0 + s if absolute_step else p0 * (1.0 + s)
        vec = evaluate(param=p)
        derivative = c*vec if derivative is None else derivative + c*vec
        if fourth_difference is None:
            fourth_difference = c4*vec
        else:
            fourth_difference = fourth_difference + c4*vec
    center = evaluate(param=np.array(fiducial, dtype=np.float64, copy=True))
    fourth_difference = fourth_difference + 6.0*center
    derivative = derivative / (h if absolute_step else h * p0)

    # Delta4 m^T C^-1 Delta4 m: the squared whitened length of the
    # fourth difference, a chi^2. It is >= 0 for any positive
    # semi-definite invcov, so a negative value means invcov is not one
    weighted = inverse_covariance @ fourth_difference
    chi2_fourth = float(fourth_difference @ weighted)
    if chi2_fourth < 0.0:
        raise ValueError(
            f"get_stencil_noise: Delta4 m^T invcov Delta4 m = {chi2_fourth} "
            "< 0; invcov is not positive semi-definite. Pass the inverse "
            "covariance of the data vector")
    # 70 = 1 + 16 + 36 + 16 + 1, the sum of the squared fourth-difference
    # weights: independent noise of variance sigma^2 per point gives the
    # fourth difference the variance 70 sigma^2
    sigma = np.sqrt(chi2_fourth/70.0)
    # the step in the parameter's own units; the stencil above moves by
    # h p0, which is negative for a negative fiducial, so the magnitude
    # is taken
    eps = h if absolute_step else h*abs(p0)
    # 130/144 = (1 + 64 + 64 + 1)/12^2, the sum of the squared five-point
    # weights (-1, 8, -8, 1)/12: the average e^T C^-1 e of the derivative
    # noise is (130/144) (sigma/eps)^2
    spurious_prior = (130.0/144.0)*(sigma/eps)**2
    return {
        "sigma": float(sigma),
        "eps": float(eps),
        "a": float(spurious_prior),
        "ddv": derivative,
    }


def get_fom_noise_weights(F, i, j, eps=None):
    """Weigh every column of F by how strongly its noise reaches FoM(i, j).

    Write w = (p_i, p_j) for the pair, C_w = (F^-1)_ww for its 2x2
    marginalized covariance (the block get_FoM reads,
    FoM = det(C_w)^-1/2) and sigma_k^2 = (F^-1)_kk for the
    marginalized variance of parameter k. For every column k,

        R_k^2 = (F^-1)_{k,w} C_w^-1 (F^-1)_{w,k} / sigma_k^2,

    with (F^-1)_{w,k} the two entries of column k of F^-1 in the
    pair's rows. R_k^2 is the squared multiple correlation of p_k with
    the pair: the share of p_k's forecast variance that a linear fit to
    (p_i, p_j) explains. It is 1 for p_i and p_j themselves, 0 for a
    parameter uncorrelated with the pair, and between 0 and 1
    otherwise. To first order, an addition a_k to F_kk changes
    ln FoM(i, j) by R_k^2 a_k sigma_k^2 / 2 (predict_fom_noise): a
    column's noise reaches the figure of merit only through the
    column's correlation with the pair. With
    a_k = (130/144) (sigma_hat_k/eps_k)^2 from get_stencil_noise
    (sigma_hat_k the measured noise of one evaluation, not to be
    confused with the forecast error sigma_k), the change from
    column k is (130/288) sigma_hat_k^2 W_k, with the dimensionless
    lever

        W_k = R_k^2 sigma_k^2 / eps_k^2 = R_k^2 (sigma_k/eps_k)^2.

    The lever needs F and the steps only, so it ranks the columns
    before any noise is measured: at each candidate accuracy setting,
    only the columns with the largest levers need get_stencil_noise;
    the rest cannot move the figure of merit even if their noise is
    sizable.

    F is symmetrized as get_FoM symmetrizes it, 0.5 (F + F^T), and
    must be positive definite, so that the forecast covariance F^-1
    exists.

    Arguments:
      F    = Fisher matrix, a 2D (n, n) array, priors included.
      i, j = the indices of the pair, two distinct integers in [0, n).
      eps  = None, or a 1D array (n,) with the step of every column in
             the parameters' own units (the "eps" of get_stencil_noise:
             h |p0_k|, or h when |p0_k| <= 1e-12); finite and > 0.

    Returns:
      dict with
        "R2"     = 1D array (n,), R_k^2, dimensionless;
        "sigma2" = 1D array (n,), sigma_k^2 = (F^-1)_kk, in the unit of
                   p_k squared;
        "lever"  = 1D array (n,), W_k, dimensionless, or None when eps
                   is None.

    Raises:
      ValueError when F is not a finite square matrix, when F is not
      positive definite, when i or j is not an integer in [0, n) or
      i == j, or when eps has the wrong shape or an entry that is not
      finite and positive.
    """
    fisher = np.asarray(F, dtype=np.float64)
    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError(
            f"get_fom_noise_weights: F has shape {fisher.shape}; it must be "
            "a square (n, n) Fisher matrix")
    if not np.all(np.isfinite(fisher)):
        raise ValueError(
            "get_fom_noise_weights: F has a nonfinite entry; check its "
            "derivatives and priors")
    n_params = fisher.shape[0]
    for name, value in (("i", i), ("j", j)):
        if not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"get_fom_noise_weights: {name} = {value!r} is not an "
                "integer; pass a parameter index")
        if value < 0 or value >= n_params:
            raise ValueError(
                f"get_fom_noise_weights: {name} = {value} is outside F, "
                f"which has {n_params} parameters; pass 0 <= {name} < "
                f"{n_params}")
    if i == j:
        raise ValueError(
            f"get_fom_noise_weights: i = j = {i}; a figure of merit needs "
            "two different parameters")
    steps = None
    if eps is not None:
        steps = np.asarray(eps, dtype=np.float64)
        if steps.shape != (n_params,):
            raise ValueError(
                f"get_fom_noise_weights: eps has shape {steps.shape}, but F "
                f"has {n_params} parameters; pass one step per column")
        if not np.all(np.isfinite(steps)) or np.any(steps <= 0.0):
            raise ValueError(
                f"get_fom_noise_weights: eps = {steps}; every step must be "
                "finite and > 0 (the 'eps' of get_stencil_noise)")
    fisher = 0.5 * (fisher + fisher.T)
    # the Cholesky factorization F = L L^T exists only for a positive
    # definite F, so it doubles as the test; "from None" drops numpy's
    # own traceback, which the message below replaces
    try:
        np.linalg.cholesky(fisher)
    except np.linalg.LinAlgError:
        raise ValueError(
            "get_fom_noise_weights: F is not positive definite, so no "
            "forecast covariance F^-1 exists; check its derivatives, priors "
            "and parameter list") from None

    covariance = np.linalg.inv(fisher)
    # np.diag of a 2D array returns a read-only view of its diagonal;
    # the copy is an independent array
    sigma2 = np.diag(covariance).copy()
    pair = [i, j]
    pair_covariance = covariance[np.ix_(pair, pair)]
    pair_information = np.linalg.inv(pair_covariance)
    # cross[:, k] = (F^-1)_{w,k}, the pair's rows of column k of F^-1:
    # shape (2, n). The product with pair_information is (2, n) as well,
    # and the sum over axis 0 of the elementwise product gives, column
    # by column, the quadratic form (F^-1)_{k,w} C_w^-1 (F^-1)_{w,k}
    cross = covariance[pair, :]
    quadratic = np.sum(cross * (pair_information @ cross), axis=0)
    R2 = quadratic / sigma2
    lever = None
    if steps is not None:
        lever = R2 * sigma2 / steps**2
    return {
        "R2": R2,
        "sigma2": sigma2,
        "lever": lever,
    }


def predict_fom_noise(F, i, j, a):
    """Predict the change of FoM(i, j) from diagonal additions a to F.

    The noise of derivative column k adds, on average, a_k to F_kk and
    nothing elsewhere (get_stencil_noise), so the average noisy Fisher
    matrix is F + diag(a). Two predictions of what that does to the
    figure of merit of get_FoM, FoM = det(C_w)^-1/2 with
    C_w = (F^-1)_ww the pair's marginalized covariance:
      - first order,
            dlnFoM = 1/2 sum_k R_k^2 a_k sigma_k^2,
        with R_k^2 and sigma_k^2 from get_fom_noise_weights. It follows
        from delta(F^-1) = -F^-1 diag(a) F^-1 and
        delta ln det C_w = tr(C_w^-1 delta C_w). Each term
        a_k sigma_k^2 is the squared ratio of the marginalized error
        sigma_k to the spurious prior width a_k^-1/2 -- how informative
        the fake prior is against what the data leave after
        marginalization -- and R_k^2 passes it to the pair;
      - exact,
            ratio = FoM(F + diag(a)) / FoM(F),
        the same diagonal addition without the expansion. Once the
        terms are of order one the first order overshoots and only the
        ratio holds. To compare an accuracy setting with a reference
        setting, divide the setting's ratio by the reference's own
        ratio: the reference is not noise free.

    Both keep only the average addition: the zero-mean parts of the
    noise (the cross terms 2 d_k^T C^-1 e_k, the off-diagonal gains
    e_k^T C^-1 e_l) move one computed forecast by a random amount that
    is not predicted here. The estimate is a convergence diagnostic,
    not a correction: a setting has settled when
    sum_k R_k^2 a_k sigma_k^2 (= 2 dlnFoM) is below about 1e-2, which
    keeps the figure of merit within about half a percent. And it sees
    only the point-to-point jitter that get_stencil_noise measures:
    numerical errors that vary smoothly with the parameters sit
    outside it.

    Arguments:
      F    = Fisher matrix, a 2D (n, n) array, priors included, at a
             normal accuracy: R_k and sigma_k barely depend on the
             accuracy setting, so one F serves every setting.
      i, j = the indices of the pair (as in get_FoM).
      a    = 1D array (n,) of the diagonal additions, in one over the
             unit of p_k squared (the "a" of get_stencil_noise for each
             measured column), with zeros for the columns whose noise
             was not measured; finite and >= 0.

    Returns:
      dict with
        "dlnFoM" = float, the first-order change of ln FoM(i, j);
        "terms"  = 1D array (n,), R_k^2 a_k sigma_k^2 per column,
                   dimensionless (dlnFoM = sum(terms)/2);
        "ratio"  = float, FoM(F + diag(a)) / FoM(F);
        "R2", "sigma2" = the arrays of get_fom_noise_weights.

    Raises:
      ValueError from get_fom_noise_weights (a malformed F, i or j),
      when a has the wrong shape or an entry that is negative or not
      finite, and from get_FoM.
    """
    # get_fom_noise_weights validates F, i and j before computing
    weights = get_fom_noise_weights(F=F, i=i, j=j)
    n_params = len(weights["R2"])
    additions = np.asarray(a, dtype=np.float64)
    if additions.shape != (n_params,):
        raise ValueError(
            f"predict_fom_noise: a has shape {additions.shape}, but F has "
            f"{n_params} parameters; pass one addition per column, 0 for "
            "the columns whose noise was not measured")
    if not np.all(np.isfinite(additions)):
        raise ValueError(
            f"predict_fom_noise: a = {additions} has a nonfinite entry; "
            "every addition must be finite")
    if np.any(additions < 0.0):
        raise ValueError(
            f"predict_fom_noise: a = {additions} has a negative entry; an "
            "average noise addition e^T C^-1 e is >= 0")
    terms = weights["R2"] * additions * weights["sigma2"]
    dlnFoM = 0.5 * float(np.sum(terms))
    fisher = np.asarray(F, dtype=np.float64)
    # get_FoM symmetrizes each matrix itself; np.diag of a 1D array
    # builds the diagonal matrix diag(a)
    fom = get_FoM(i, j, fisher)
    fom_noisy = get_FoM(i, j, fisher + np.diag(additions))
    return {
        "dlnFoM": dlnFoM,
        "terms": terms,
        "ratio": fom_noisy / fom,
        "R2": weights["R2"],
        "sigma2": weights["sigma2"],
    }


def plot_Fisher(F, mu, F2=None, root=None, select=None, labels=None,
                names=None, filled=True, flat_priors=None,
                chain_names=None, rng=None):
    """Draw a getdist triangle plot of Fisher contours, optionally vs MCMC.

    Each Fisher matrix is turned into a cloud of samples from the
    Gaussian it defines (mean mu, covariance F^-1), truncated to the
    flat-prior boxes, and handed to getdist, which draws the familiar
    triangle of 1D and 2D marginals. An MCMC chain loaded from root
    joins the same figure, so forecast and chain overlay directly.

    The truncation clips the drawn samples in the selected
    coordinates only, after the covariance sub-block has already
    marginalized the dropped parameters: a flat prior on a parameter
    that was marginalized out cannot be imposed afterwards by
    cutting the shown coordinates, and the clipped cloud illustrates
    a truncated Gaussian, not a recomputed posterior. The wall
    plotters below (plot_Fisher_amplitude_cut,
    plot_Fisher_prior_walls) make the complementary choice:
    untruncated Gaussian contours next to shaded walls.

    Arguments:
      F      = one full-size Fisher matrix or a list of them (one
               contour set each); each is inverted before select cuts
               its covariance down.
      mu     = 1D array of fiducial parameter values (the Gaussian
               mean), full length; select cuts it down.
      F2     = a second matrix or list, sampled as F and drawn after
               it with the next styles of the cycle.
      root   = getdist chain root to load and overlay, or None.
      select = list of parameter indices to show, or None for all.
      labels = full-length list of LaTeX parameter labels.
      names  = full-length list of parameter names (getdist ids).
      filled = True fills the 2D contours.
      flat_priors = {position in select: (min, max)} truncation
               boxes, or None for no truncation. The keys count the
               selected parameters (the columns of the sample array),
               not the full parameter list: they agree with full
               indices only when select[j] == j for every key j, as
               with select None.
      chain_names = legend labels, one per drawn set, or None.
      rng    = numpy random Generator for the sample clouds; None
               creates a fresh seeded one (reproducible, but then
               repeated calls reuse identical sample noise).

    Returns:
      the getdist subplot plotter holding the figure. The figure is
      drawn on getdist's matplotlib figure; nothing calls plt.show()
      and nothing is saved.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if flat_priors is None:
        flat_priors = {}
    if select is None:
        select = list(range(len(mu)))
    idx = list(select)

    # cut the mean and the label lists down to the selected
    # parameters; the comprehensions keep the order of idx
    mu = np.asarray(mu, float)[idx]
    names  = [names[k]  for k in idx]
    labels = [labels[k] for k in idx]

    # accept one matrix or a list of them: a bare matrix is wrapped
    # into a one-element list so the loop below covers both cases
    F = F if isinstance(F, (list, tuple)) else [F]

    samples = []
    for Fi in F:
        Fi = 0.5 * (Fi + Fi.T)
        C = np.linalg.inv(Fi)
        # np.ix_ selects the (idx x idx) block: marginalizing a
        # Gaussian over the dropped parameters is taking the
        # covariance sub-block
        C = C[np.ix_(idx, idx)]
        # 40000 draws from the Gaussian the Fisher matrix defines
        y = rng.multivariate_normal(mu, C, size=40000)
        # one boolean mask per flat prior, one entry per draw: True
        # where column j of y (position j in select) sits inside the
        # box
        conds = [(y[:,j] >= lo) & (y[:,j] <= hi) for j, (lo, hi) in flat_priors.items()]
        # np.all(conds, axis=0) stacks the masks as rows and ands them
        # draw by draw: a draw survives only when every box contains
        # it; with no boxes, np.ones(..., dtype=bool) keeps every draw
        valid = np.all(conds, axis=0) if conds else np.ones(y.shape[0], dtype=bool)
        # {name: [lo, hi]} tells getdist where hard prior edges sit,
        # so its density estimate does not smooth across them
        ranges = {names[j]: [lo, hi] for j, (lo, hi) in flat_priors.items()}
        x = MCSamples(samples=y[valid], names=names, ranges=ranges, labels=labels)
        samples.append(x)

    # getdist analysis settings: smooth_scale_1D/2D = 0.1 smooth with
    # a Gaussian kernel 0.1 standard deviations wide (getdist's default
    # -1 picks the width automatically); ignore_rows 0 keeps every draw
    # (independent samples, no burn-in); range_confidence 0.005 is the
    # 1D tail probability that sets each parameter's plotted range;
    # fine_bins_2D is the 2D density grid. getdist 1.7 has no settings
    # named bins_2D, fine_bins_1D or bins_1D and ignores those entries.
    analysissettings={'smooth_scale_1D':0.1,
                      'smooth_scale_2D':0.1,
                      'ignore_rows': u'0.0',
                      'range_confidence' : u'0.005',
                      'fine_bins_2D': 2048,
                      'bins_2D': 1024,
                      'fine_bins_1D': 512,
                      'bins_1D': 256}

    if F2 is not None:
        F2 = F2 if isinstance(F2, (list, tuple)) else [F2]
        for Fi in F2:
            Fi = 0.5 * (Fi + Fi.T)
            C = np.linalg.inv(Fi)
            C = C[np.ix_(idx, idx)]
            y = rng.multivariate_normal(mu, C, size=40000)
            conds = [(y[:,j] >= lo) & (y[:,j] <= hi) for j, (lo, hi) in flat_priors.items()]
            valid = np.all(conds, axis=0) if conds else np.ones(y.shape[0], dtype=bool)
            ranges = {names[j]: [lo, hi] for j, (lo, hi) in flat_priors.items()}
            x = MCSamples(samples=y[valid], names=names, ranges=ranges, labels=labels)
            samples.append(x)

    # ignore_rows 0.0 removes no burn-in: the chain at root is used as
    # stored
    if root:
        samples.append(loadMCSamples(root,settings={'ignore_rows': u'0.0'}))

    g = plots.get_subplot_plotter(width_inch=10.5,analysis_settings=analysissettings)
    g.settings.axis_tick_x_rotation=65
    g.settings.lw_contour=1.0
    g.settings.legend_rect_border = False
    g.settings.figure_legend_frame = False
    g.settings.axes_fontsize = 16.0
    g.settings.legend_fontsize = 16.5
    g.settings.alpha_filled_add = 0.85
    g.settings.lab_fontsize=15.5
    g.legend_labels=False

    # itertools.cycle repeats each style list forever: every drawn
    # set pulls the next color/dash/width, wrapping at the end
    cc  = itertools.cycle(["cornflowerblue", "darkorange", "seagreen", "firebrick", "purple", "gold"])
    lsc = itertools.cycle(["solid", "dashed", "dotted", "dashdot"])
    lwc = itertools.cycle([1.2, 1.5])

    line_args = []
    contour_cols = []
    contour_ls = []
    contour_lws = []

    for _ in samples:
        c  = next(cc)
        ls = next(lsc)
        lw = next(lwc)
        line_args.append({'lw': lw, 'ls': ls, 'color': c})
        contour_cols.append(c)
        contour_ls.append(ls)
        contour_lws.append(lw)

    g.triangle_plot(samples,
                    params=names,
                    filled=filled,
                    line_args=line_args,
                    contour_colors=contour_cols,
                    contour_ls=contour_ls,
                    contour_lws=contour_lws,
                   legend_labels=chain_names)
    return g


def _gaussian_contour(center, cov2, delta_chi2, n_points=240):
    """Return the (x, y) points of one contour of a 2D Gaussian.

    The contour of the Gaussian with mean `center` and covariance `cov2`
    where (p - center)^T cov2^-1 (p - center) = delta_chi2. With
    cov2 = L L^T (Cholesky), a point at angle t is
    center + sqrt(delta_chi2) L (cos t, sin t).

    Arguments:
      center     = 1D array of the two means.
      cov2       = 2x2 covariance matrix.
      delta_chi2 = the contour level (2.30 for 68.3%, 6.18 for 95.4%).
      n_points   = points along the contour.

    Returns:
      (x, y), two 1D arrays of length n_points.
    """
    chol = np.linalg.cholesky(cov2)
    angle = np.linspace(0.0, 2.0*np.pi, n_points)
    circle = np.sqrt(delta_chi2)*np.vstack([np.cos(angle), np.sin(angle)])
    contour = chol @ circle
    return center[0] + contour[0], center[1] + contour[1]


def plot_Fisher_amplitude_cut(F, names, fiducial, amplitude, biases, boxes,
                              labels=None, cut=8.0, chain_names=None,
                              figsize=None, show=None):
    """Draw the amplitude-bias prior box, an amplitude cut and Fisher contours.

    The linear galaxy power of lens bin i scales as A_s b_i^2 (the
    full prediction adds nonlinear, RSD and magnification terms that
    do not follow this scaling; the ratio below uses the linear
    scaling as a proxy). A prior box that lets A_s and b_i range
    freely reaches amplitude ratios
    A_i = (A_s/A_s,fid)(b_i/b_i,fid)^2 far above 1, where a fixed
    relative numerical error of the data vector costs a large delta
    chi2 against the fiducial covariance. A cut A_i <= cut removes that
    corner of the box. This triangle is an overlap diagnostic for such
    a cut: it shades the excluded region gray and draws each Fisher
    forecast's 1 and 2 sigma contours about the fiducial (68.3% and
    95.4% of a two-dimensional Gaussian, delta chi2 = 2.30 and 6.18).
    The contours come from the local Gaussian approximation the Fisher
    matrix defines (module docstring): a cut far outside the 2 sigma
    contours overlaps little of each forecast Gaussian, which within
    that approximation marks the removed corner as prior volume the
    data already disfavor. The Gaussian keeps nonzero probability
    beyond any contour, and the figure says nothing about the true
    likelihood far from the fiducial, where the local approximation
    itself degrades.

    The panels form a lower triangle over [amplitude] + biases:
      - an amplitude - b_i panel shades bin i's own cut;
      - a b_i - b_j panel shades the region either bin excludes at the
        fiducial amplitude;
      - a diagonal panel shades the part of the box excluded with the
        other parameters at the fiducial, and draws each forecast's
        marginal Gaussian (peak 1).
    Each Fisher matrix is inverted once; the contours of a pair use the
    2x2 block of that covariance, which marginalizes every other
    parameter of the matrix (cosmology and nuisances alike). The Fisher
    contours ignore the flat priors; the axes show the boxes.

    Arguments:
      F           = one Fisher matrix or a list of them (one contour set
                    each).
      names       = the parameter names of the rows of F: one list for
                    every matrix, or a list of lists, one per matrix.
      fiducial    = {name: value} of the amplitude and the biases (the
                    contour centers).
      amplitude   = the name of the amplitude parameter (As_1e9).
      biases      = the names of the linear-bias parameters to show.
      boxes       = {name: (min, max)} flat-prior boxes of the amplitude
                    and the biases; they set the axis ranges.
      labels      = {name: LaTeX label without $}, or None to show the
                    names.
      cut         = the largest allowed amplitude ratio A_i.
      chain_names = legend labels, one per matrix, or None.
      figsize     = (width, height) in inches, or None for 1.9-inch
                    panels.
      show        = 1 draws the figure; None returns (fig, axes).

    Returns:
      None when show is 1; (fig, axes) when show is None, axes the
      (n, n) array of panels, n = 1 + len(biases).

    Raises:
      ValueError when F and names differ in number, when a shown
      parameter is missing from a matrix's names, from fiducial or from
      boxes, or when cut is not positive.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # one matrix or a list of them; names follow the same shape
    matrices = list(F) if isinstance(F, (list, tuple)) else [F]
    if len(names) > 0 and isinstance(names[0], (list, tuple)):
        name_lists = [list(n) for n in names]
    else:
        name_lists = []
        for _ in matrices:
            name_lists.append(list(names))
    if len(name_lists) != len(matrices):
        raise ValueError(
            f"{len(matrices)} Fisher matrices but {len(name_lists)} name "
            "lists: pass one list for all matrices or one per matrix")
    shown = [amplitude] + list(biases)
    for name in shown:
        if name not in fiducial or name not in boxes:
            raise ValueError(
                f"{name!r} needs a fiducial value and a box: add it to "
                "fiducial and to boxes")
        for k in range(len(name_lists)):
            if name not in name_lists[k]:
                raise ValueError(
                    f"{name!r} is not a parameter of Fisher matrix {k} "
                    f"(names {name_lists[k]})")
    if not cut > 0:
        raise ValueError(f"cut = {cut}: the amplitude cut must be positive")
    if labels is None:
        labels = {}
    if chain_names is None:
        chain_names = []
        for k in range(len(matrices)):
            chain_names.append(f"Fisher {k + 1}")

    # the marginalized covariance of the shown parameters, per matrix
    covariances = []
    for k in range(len(matrices)):
        matrix = np.asarray(matrices[k], dtype=float)
        covariance = np.linalg.inv(0.5*(matrix + matrix.T))
        rows = []
        for name in shown:
            rows.append(name_lists[k].index(name))
        covariances.append(covariance[np.ix_(rows, rows)])
    center = np.array([fiducial[name] for name in shown])

    def ratio(amplitude_value, bias_name, bias_value):
        """A_i = (A_s/A_s,fid)(b_i/b_i,fid)^2 (arrays welcome)."""
        return ((amplitude_value/fiducial[amplitude])
                *(bias_value/fiducial[bias_name])**2)

    def axis_label(name):
        if name in labels:
            return f"${labels[name]}$"
        return name

    # Okabe-Ito (colorblind safe, never red with green), each forecast
    # also told apart by its line width
    colors = ["#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#D55E00"]
    widths = [2.4, 1.8, 1.3, 0.9, 0.7]
    gray = ListedColormap(["white", "0.80"])
    n = len(shown)
    if figsize is None:
        figsize = (1.9*n + 0.6, 1.9*n)
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=figsize,
                             squeeze=False,
                             gridspec_kw={"wspace": 0.06, "hspace": 0.06})
    for row in range(n):
        for col in range(n):
            ax = axes[row][col]
            if col > row:
                ax.axis("off")
                continue
            x_name = shown[col]
            y_name = shown[row]
            x_grid = np.linspace(boxes[x_name][0], boxes[x_name][1], 300)
            if col == row:
                # the box, with the part excluded at the fiducial of the
                # others shaded, and each forecast's marginal Gaussian
                if x_name == amplitude:
                    excluded = np.zeros(x_grid.shape, dtype=bool)
                    for bias_name in biases:
                        excluded |= ratio(x_grid, bias_name,
                                          fiducial[bias_name]) > cut
                else:
                    excluded = ratio(fiducial[amplitude], x_name,
                                     x_grid) > cut
                ax.fill_between(x_grid, 0.0, 1.1, where=excluded,
                                color="0.80", step="mid")
                for k in range(len(covariances)):
                    sigma = np.sqrt(covariances[k][col, col])
                    gauss = np.exp(-0.5*((x_grid - center[col])/sigma)**2)
                    ax.plot(x_grid, gauss, color=colors[k % len(colors)],
                            lw=widths[k % len(widths)])
                ax.set_ylim(0.0, 1.1)
                ax.set_yticks([])
            else:
                y_grid = np.linspace(boxes[y_name][0], boxes[y_name][1], 300)
                x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
                if x_name == amplitude:
                    excluded = ratio(x_mesh, y_name, y_mesh) > cut
                else:
                    excluded = ((ratio(fiducial[amplitude], x_name, x_mesh)
                                 > cut)
                                | (ratio(fiducial[amplitude], y_name, y_mesh)
                                   > cut))
                ax.pcolormesh(x_mesh, y_mesh, excluded.astype(float),
                              cmap=gray, vmin=0.0, vmax=1.0,
                              shading="auto")
                if np.any(excluded):
                    ax.contour(x_mesh, y_mesh, excluded.astype(float),
                               levels=[0.5], colors="k", linewidths=0.7)
                pair = [col, row]
                for k in range(len(covariances)):
                    cov2 = covariances[k][np.ix_(pair, pair)]
                    for level, style in ((2.30, "-"), (6.18, "--")):
                        x_line, y_line = _gaussian_contour(center[pair],
                                                           cov2, level)
                        ax.plot(x_line, y_line,
                                color=colors[k % len(colors)],
                                linestyle=style, lw=widths[k % len(widths)])
                ax.plot(center[col], center[row], marker="*", color="k",
                        ms=9, mec="white", mew=0.5)
                ax.set_ylim(boxes[y_name])
            ax.set_xlim(boxes[x_name])
            ax.tick_params(labelsize=8)
            # axis labels on the outer edges of the triangle only
            if row == n - 1:
                ax.set_xlabel(axis_label(x_name), fontsize=13)
            else:
                ax.tick_params(labelbottom=False)
            if col == 0 and row > 0:
                ax.set_ylabel(axis_label(y_name), fontsize=13)
            elif col > 0 and col != row:
                ax.tick_params(labelleft=False)

    handles = []
    for k in range(len(covariances)):
        handles.append(Line2D([], [], color=colors[k % len(colors)],
                              lw=widths[k % len(widths)],
                              label=chain_names[k]))
    handles.append(Line2D([], [], color="0.3", lw=1.3, linestyle="-",
                          label=r"$1\sigma$ (68.3%)"))
    handles.append(Line2D([], [], color="0.3", lw=1.3, linestyle="--",
                          label=r"$2\sigma$ (95.4%)"))
    handles.append(Patch(facecolor="0.80", edgecolor="k", lw=0.7,
                         label=(r"excluded: $A_\mathrm{s} b_i^2/"
                                r"(A_\mathrm{s} b_i^2)_\mathrm{fid} > "
                                f"{cut:g}$")))
    handles.append(Line2D([], [], linestyle="", marker="*", color="k",
                          ms=9, label="fiducial"))
    # the empty upper-right triangle holds the legend
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.93, 0.93), frameon=False, fontsize=11)
    if show == 1:
        plt.show()
        return None
    return fig, axes


def plot_Fisher_prior_walls(F, names, fiducial, params, ranges, boxes=None,
                            walls=None, labels=None, chain_names=None,
                            figsize=None, show=None):
    """Draw Fisher contours against the walls of a prior.

    A flat prior is a set of walls: the box of each parameter and, for
    combinations, linear walls such as lo <= w0 + wa <= hi. A wall can
    only move an inference if the data allow points near it. This
    triangle is an overlap diagnostic: it shades the region the walls
    exclude and draws each Fisher forecast's 1 and 2 sigma contours
    about the fiducial (68.3% and 95.4% of a two-dimensional Gaussian,
    delta chi2 = 2.30 and 6.18), every other parameter of the matrix
    marginalized. The contours come from the local Gaussian
    approximation the Fisher matrix defines (module docstring): a wall
    far outside the 2 sigma contours overlaps little of each forecast
    Gaussian, and within that approximation it removes only prior
    volume the data disfavor. The Gaussian keeps nonzero probability
    beyond any contour, and the figure says nothing about the true
    likelihood far from the fiducial.

    The panels form a lower triangle over params: a two-parameter panel
    shades the points outside either parameter's box or outside a wall
    that involves only those two parameters; a diagonal panel shades the
    part of the axis outside the box and draws each forecast's marginal
    Gaussian (peak 1).

    Arguments:
      F           = one Fisher matrix or a list of them (one contour set
                    each).
      names       = the parameter names of the rows of F: one list for
                    every matrix, or a list of lists, one per matrix.
      fiducial    = {name: value} of the shown parameters (the contour
                    centers).
      params      = the names of the parameters to show.
      ranges      = {name: (min, max)} axis ranges of the shown
                    parameters; wider than the boxes, so the walls show.
      boxes       = {name: (min, max)} flat-prior boxes of shown
                    parameters, or None; a parameter without a box has
                    no wall of its own.
      walls       = list of linear walls ({name: coefficient}, lo, hi):
                    the prior keeps lo <= sum coefficient*value <= hi
                    (None for an open side); a wall shades only panels
                    whose two parameters are exactly its names.
      labels      = {name: LaTeX label without $}, or None to show the
                    names.
      chain_names = legend labels, one per matrix, or None.
      figsize     = (width, height) in inches, or None for 2.3-inch
                    panels.
      show        = 1 draws the figure; None returns (fig, axes).

    Returns:
      None when show is 1; (fig, axes) when show is None, axes the
      (n, n) array of panels, n = len(params).

    Raises:
      ValueError when F and names differ in number, when a shown
      parameter is missing from a matrix's names, from fiducial or from
      ranges, or when a wall names a parameter that is not shown.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    matrices = list(F) if isinstance(F, (list, tuple)) else [F]
    if len(names) > 0 and isinstance(names[0], (list, tuple)):
        name_lists = [list(n) for n in names]
    else:
        name_lists = []
        for _ in matrices:
            name_lists.append(list(names))
    if len(name_lists) != len(matrices):
        raise ValueError(
            f"{len(matrices)} Fisher matrices but {len(name_lists)} name "
            "lists: pass one list for all matrices or one per matrix")
    if boxes is None:
        boxes = {}
    if walls is None:
        walls = []
    if labels is None:
        labels = {}
    for name in params:
        if name not in fiducial or name not in ranges:
            raise ValueError(
                f"{name!r} needs a fiducial value and an axis range: add "
                "it to fiducial and to ranges")
        for k in range(len(name_lists)):
            if name not in name_lists[k]:
                raise ValueError(
                    f"{name!r} is not a parameter of Fisher matrix {k} "
                    f"(names {name_lists[k]})")
    for coefficients, _, _ in walls:
        for name in coefficients:
            if name not in params:
                raise ValueError(
                    f"the wall on {sorted(coefficients)} names {name!r}, "
                    f"which is not shown (params {params})")
    if chain_names is None:
        chain_names = []
        for k in range(len(matrices)):
            chain_names.append(f"Fisher {k + 1}")

    covariances = []
    for k in range(len(matrices)):
        matrix = np.asarray(matrices[k], dtype=float)
        covariance = np.linalg.inv(0.5*(matrix + matrix.T))
        rows = []
        for name in params:
            rows.append(name_lists[k].index(name))
        covariances.append(covariance[np.ix_(rows, rows)])
    center = np.array([fiducial[name] for name in params])

    def outside_box(name, values):
        """True where a value of `name` falls outside its box."""
        if name not in boxes:
            return np.zeros(np.shape(values), dtype=bool)
        low, high = boxes[name]
        return (values < low) | (values > high)

    def axis_label(name):
        if name in labels:
            return f"${labels[name]}$"
        return name

    # Okabe-Ito (colorblind safe, never red with green), each forecast
    # also told apart by its line width
    colors = ["#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#D55E00"]
    widths = [2.4, 1.8, 1.3, 0.9, 0.7]
    gray = ListedColormap(["white", "0.80"])
    n = len(params)
    if figsize is None:
        figsize = (2.3*n + 0.8, 2.3*n)
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=figsize,
                             squeeze=False,
                             gridspec_kw={"wspace": 0.06, "hspace": 0.06})
    for row in range(n):
        for col in range(n):
            ax = axes[row][col]
            if col > row:
                ax.axis("off")
                continue
            x_name = params[col]
            y_name = params[row]
            x_grid = np.linspace(ranges[x_name][0], ranges[x_name][1], 400)
            if col == row:
                excluded = outside_box(x_name, x_grid)
                ax.fill_between(x_grid, 0.0, 1.1, where=excluded,
                                color="0.80", step="mid")
                for k in range(len(covariances)):
                    sigma = np.sqrt(covariances[k][col, col])
                    gauss = np.exp(-0.5*((x_grid - center[col])/sigma)**2)
                    ax.plot(x_grid, gauss, color=colors[k % len(colors)],
                            lw=widths[k % len(widths)])
                ax.set_ylim(0.0, 1.1)
                ax.set_yticks([])
            else:
                y_grid = np.linspace(ranges[y_name][0], ranges[y_name][1], 400)
                x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
                excluded = outside_box(x_name, x_mesh) | outside_box(y_name, y_mesh)
                for coefficients, low, high in walls:
                    if set(coefficients) != {x_name, y_name}:
                        continue
                    combination = (coefficients[x_name]*x_mesh
                                   + coefficients[y_name]*y_mesh)
                    if low is not None:
                        excluded |= combination < low
                    if high is not None:
                        excluded |= combination > high
                ax.pcolormesh(x_mesh, y_mesh, excluded.astype(float),
                              cmap=gray, vmin=0.0, vmax=1.0, shading="auto")
                if np.any(excluded):
                    ax.contour(x_mesh, y_mesh, excluded.astype(float),
                               levels=[0.5], colors="k", linewidths=0.7)
                pair = [col, row]
                for k in range(len(covariances)):
                    cov2 = covariances[k][np.ix_(pair, pair)]
                    for level, style in ((2.30, "-"), (6.18, "--")):
                        x_line, y_line = _gaussian_contour(center[pair],
                                                           cov2, level)
                        ax.plot(x_line, y_line,
                                color=colors[k % len(colors)],
                                linestyle=style, lw=widths[k % len(widths)])
                ax.plot(center[col], center[row], marker="*", color="k",
                        ms=9, mec="white", mew=0.5)
                ax.set_ylim(ranges[y_name])
            ax.set_xlim(ranges[x_name])
            ax.tick_params(labelsize=8)
            if row == n - 1:
                ax.set_xlabel(axis_label(x_name), fontsize=13)
            else:
                ax.tick_params(labelbottom=False)
            if col == 0 and row > 0:
                ax.set_ylabel(axis_label(y_name), fontsize=13)
            elif col > 0 and col != row:
                ax.tick_params(labelleft=False)

    handles = []
    for k in range(len(covariances)):
        handles.append(Line2D([], [], color=colors[k % len(colors)],
                              lw=widths[k % len(widths)],
                              label=chain_names[k]))
    handles.append(Line2D([], [], color="0.3", lw=1.3, linestyle="-",
                          label=r"$1\sigma$ (68.3%)"))
    handles.append(Line2D([], [], color="0.3", lw=1.3, linestyle="--",
                          label=r"$2\sigma$ (95.4%)"))
    handles.append(Patch(facecolor="0.80", edgecolor="k", lw=0.7,
                         label="excluded by the prior"))
    handles.append(Line2D([], [], linestyle="", marker="*", color="k",
                          ms=9, label="fiducial"))
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.93, 0.93), frameon=False, fontsize=11)
    if show == 1:
        plt.show()
        return None
    return fig, axes
