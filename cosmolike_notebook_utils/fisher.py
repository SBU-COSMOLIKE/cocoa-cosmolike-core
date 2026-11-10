"""Fisher forecasting helpers shared by the EXAMPLE_EVALUATE notebooks.

A Fisher matrix forecasts parameter uncertainties from the
derivatives of the theory data vector: F = D^T C^-1 D, with D the
matrix whose column p is the derivative of the data vector with
respect to parameter p, and C the data covariance. Everything in
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
    parameter whose fiducial is zero (|p0| <= 1e-12) falls back to
    an absolute step.

    Arguments:
      dv    = the data-vector function, called as
              dv(param=<1D array>, AccuracyBoost=AccuracyBoost).
      index = position of the differentiated parameter inside CV.
      h     = relative step size (0.02 = two percent); the absolute
              step when the fiducial is zero. Must be > 0 (assert).
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
    result = None
    for s, c in zip(steps, coeffs):
        # copy=True: p is an independent copy, so writing p[index]
        # below cannot touch the caller's CV array
        p = np.array(CV, dtype = np.float64, copy=True)
        # ternary a if cond else b: absolute step when the fiducial
        # is (numerically) zero, relative step otherwise
        p[index] = p0 + s if np.isclose(p0, 0.0, atol=1e-12) else p0 * (1.0 + s)
        vec = dv(param=p, AccuracyBoost=AccuracyBoost)
        # accumulate c * vec; the first turn initializes result
        result = c*vec if result is None else result + c*vec
    # divide by the step scale: h for absolute steps, h * p0 for
    # relative ones. This test is exact (p0 == 0.0) while the loop
    # takes absolute steps for |p0| <= 1e-12, so the two agree for
    # every fiducial except 0 < |p0| <= 1e-12.
    return result / (h if p0 == 0.0 else h * p0)


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
    fit tolerance, which never happens on data vectors.

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
    # the comprehension evaluates ddv once per parameter (four
    # data-vector evaluations per call with get_ddv); np.column_stack
    # glues the returned 1D arrays side by side as the columns of D
    D = np.column_stack([ddv(index=p, h=h, CV=CV, AccuracyBoost=AccuracyBoost) for p in list(range(len(CV)))]).astype(np.float64)
    # @ is numpy matrix multiplication: D^T (C^-1 D)
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
    covariance (the inverse Fisher matrix) for parameters i and j: a
    larger FoM means a smaller error ellipse.

    Arguments:
      i, j = the two parameter indices.
      F    = Fisher matrix.

    Returns:
      the figure of merit as a float; inf (or a very large value)
      when det C_ij <= 0, instead of an error. A block with two
      negative eigenvalues has det C_ij > 0 and passes unnoticed.
    """
    # 0.5 * (F + F^T) symmetrizes away numerical asymmetry before
    # inverting; np.ix_([i, j], [i, j]) selects the 2x2 block with
    # rows and columns (i, j)
    C = np.linalg.inv(0.5 * (F + F.T))
    C = C[np.ix_([i, j], [i, j])]
    detC = np.linalg.det(C)
    if detC <= 0:
        # a numerically indefinite block: clip negative eigenvalues
        # to zero (eigvalsh = eigenvalues of a symmetric matrix) and
        # use the product of the clipped ones as the determinant. The
        # product is then zero (or tiny), so the FoM returned below is
        # inf (or very large): this branch reports no error
        w = np.clip(np.linalg.eigvalsh(C), 0, None)
        detC = w.prod()
    return float(1.0/np.sqrt(detC))


def plot_Fisher(F, mu, F2=None, root=None, select=None, labels=None,
                names=None, filled=True, flat_priors=None,
                chain_names=None, rng=None):
    """Draw a getdist triangle plot of Fisher contours, optionally vs MCMC.

    Each Fisher matrix is turned into a cloud of samples from the
    Gaussian it defines (mean mu, covariance F^-1), truncated to the
    flat-prior boxes, and handed to getdist, which draws the familiar
    triangle of 1D and 2D marginals. An MCMC chain loaded from root
    joins the same figure, so forecast and chain overlay directly.

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

    The galaxy power of lens bin i is proportional to A_s b_i^2. A prior
    box that lets A_s and b_i range freely reaches amplitude ratios
    A_i = (A_s/A_s,fid)(b_i/b_i,fid)^2 far above 1, where a fixed
    relative numerical error of the data vector costs a large delta
    chi2 against the fiducial covariance. A cut A_i <= cut removes that
    corner of the box. This triangle tells whether such a cut can move
    an inference: it shades the excluded region gray and draws each
    Fisher forecast's 1 and 2 sigma contours about the fiducial (68.3%
    and 95.4% of a two-dimensional Gaussian, delta chi2 = 2.30 and
    6.18). A cut far outside the 2 sigma contours removes only prior
    volume the data already exclude.

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

    colors = ["C0", "C3", "C2", "C1", "C4"]
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
                            lw=1.5)
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
                                linestyle=style, lw=1.3)
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
        handles.append(Line2D([], [], color=colors[k % len(colors)], lw=1.8,
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
    triangle shades the region the walls exclude and draws each Fisher
    forecast's 1 and 2 sigma contours about the fiducial (68.3% and
    95.4% of a two-dimensional Gaussian, delta chi2 = 2.30 and 6.18),
    every other parameter of the matrix marginalized. A wall far outside
    the 2 sigma contours removes only prior volume the data exclude.

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

    colors = ["C0", "C3", "C2", "C1", "C4"]
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
                            lw=1.5)
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
                                linestyle=style, lw=1.3)
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
        handles.append(Line2D([], [], color=colors[k % len(colors)], lw=1.8,
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
