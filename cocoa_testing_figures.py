"""The shared figures of the data-vector checks.

Two figure families live here.

The photo-z convention figures: every project's
tests/data_vector/generate_photoz_convention_figure.py evaluates the
frozen cosmic-shear fiducial under the four runtime photo-z settings
(cspline/Z_LOW default, linear, Steffen, Z_MID; see the core README's
shared data-vector checks for what they mean) and plots the fractional
data-vector differences against the default, per tomographic pair and
per angular bin. The machinery is the same in every project, so it
lives here; the thin script keeps the survey data (binning, mask file,
panel grid) and calls photoz_convention_figures with it.

The N-random-models figure: the N-random-models accuracy check
(test_ax99_nmodels of every project's test_accuracy.py) draws N
reproducible points across the prior and scores each with the pushed
accuracy settings against its own default-settings vector.
random_models_triangle draws each point's delta chi2 against the two
quantities that set it, Omega_m and the clustering amplitude A_s b^2,
plus the plane of the two, so the reader sees where in the prior the
numerical error grows.

This module imports matplotlib, so the test shims never import it at
module level: the figure scripts import it after their own
OMP_NUM_THREADS line, and the N-random-models check imports it inside
its test method, after the models are evaluated.
"""

import os
import shutil
import tempfile

# "Agg" is matplotlib's file-only backend: figures are written to PNG
# files without opening a window, so the scripts also run on a machine
# without a display. It must be chosen before pyplot is imported.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (report tag, photoz_interpolation_type, photoz_zmid_convention); the
# first entry is the default every other setting is compared with
SETTINGS = (("cspline/Z_LOW (default)", 0, 0), ("linear", 1, 0),
            ("steffen", 2, 0), ("Z_MID", 0, 1))

# The fixed y half-ranges of the two figures: fractional differences
# blow up where the signal crosses zero, and a fixed range keeps the
# flat offsets, which are the physics, readable (the spikes run
# off-panel). The Z_MID shift is percent-level; the interpolant
# differences are 1e-4-level.
ZMID_YLIM = 4.0
INTERP_YLIM = 8.0


def _datavectors(u, example):
    """Return the printed theory vector under each setting, keyed by tag.

    Builds one model per SETTINGS entry on the frozen fiducial, with
    print_datavector writing the theory vector into a temporary folder
    that is removed afterwards, also after an error.

    Arguments:
      u       = the project's cocoa_test_utils module.
      example = the key of u.EXAMPLES to evaluate.

    Returns:
      {tag: 1D float array of the full data-vector length}.
    """
    vectors_dir = tempfile.mkdtemp(prefix="photoz_conventions_fig_")
    out = {}
    try:
        for tag, interp, zmid in SETTINGS:
            print(f"building model ({example}, NLA, {tag}) ...",
                  flush=True)
            info = u.load_frozen_info(example, tatt=False)
            block = info["likelihood"][u.EXAMPLES[example]["likelihood"]]
            block["photoz_interpolation_type"] = interp
            block["photoz_zmid_convention"] = zmid
            path = os.path.join(vectors_dir,
                                f"dv_{interp}_{zmid}.modelvector")
            block["print_datavector"] = True
            block["print_datavector_file"] = path
            model = u.make_model(info)
            point = u.build_point(model, example, tatt=False)
            u.evaluate_chi2(model, point)
            out[tag] = u._load_datavector(path)
    finally:
        shutil.rmtree(vectors_dir, ignore_errors=True)
    return out


def _plot(curves, path, title, grid, x, x_label, y_label, real,
          panel_fontsize, scale, ylim):
    """Draw one grid of glued panels, one per source pair, and save it.

    The layout of the notebook plotters: the source pairs (i <= j)
    fill the grid row by row, panels share both axes, and the bin
    labels count from 1. A panel the pair count does not reach is
    switched off, and the panel above a switched-off one carries the
    x label of its column.

    Arguments:
      curves         = {label: per-pair curve data}; in a real-space
                       figure each value is a (dxi_plus, dxi_minus)
                       pair of (npair, nlen) arrays, in a Fourier one
                       a single (npair, nlen) array. NaN marks masked
                       bins.
      path           = the PNG path to write.
      title          = figure title.
      grid           = (nrows, ncols) of the panel grid.
      x              = the angular-bin or band centers.
      x_label        = the x-axis label of the bottom row.
      y_label        = the y-axis label of the first column.
      real           = True for xi_pm curves (solid xi_+, dashed
                       xi_-), False for one C_l curve per label.
      panel_fontsize = the font size of the per-panel pair label.
      scale          = factor applied to the fractions before
                       plotting (100 for percent).
      ylim           = None, or the half-width of a fixed symmetric
                       y range in plotted units.

    Returns:
      nothing; the PNG file is written and the figure closed.
    """
    nrows, ncols = grid
    # the number of pairs comes from any curve's first dimension
    first = next(iter(curves.values()))
    npair = (first[0] if real else first).shape[0]
    # ntomo from npair = ntomo (ntomo + 1) / 2, solved by search
    ntomo = 1
    while ntomo * (ntomo + 1) // 2 < npair:
        ntomo += 1
    # every source pair (i, j) with i <= j, in data-vector order
    pairs = [(i, j) for i in range(ntomo) for j in range(i, ntomo)]
    # the figure height follows the row count (the widths the
    # projects always used: 7 for two rows, 9 for three, 16 beyond)
    height = {2: 7, 3: 9}.get(nrows, 16)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(20, height),
                             sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0, "hspace": 0})
    cm = plt.get_cmap("gist_rainbow")
    # axes.ravel() lists the panels row by row; panel p shows pair p.
    # Only panel 0 labels its curves, so the legend lists each once.
    for p, (i, j) in enumerate(pairs):
        ax = axes.ravel()[p]
        for q, (label, data) in enumerate(curves.items()):
            color = cm(q / max(len(curves) - 1, 1) * 0.8)
            if real:
                dp, dm = data
                ax.semilogx(x, scale * dp[p], color=color, lw=1.6,
                            label=rf"$\xi_+$: {label}"
                            if p == 0 else None)
                ax.semilogx(x, scale * dm[p], color=color, lw=1.6,
                            ls="--",
                            label=rf"$\xi_-$: {label}"
                            if p == 0 else None)
            else:
                ax.semilogx(x, scale * data[p], color=color, lw=1.6,
                            label=label if p == 0 else None)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.text(0.08, 0.85, f"$({i+1},{j+1})$",
                transform=ax.transAxes, fontsize=panel_fontsize)
        # the bottom row carries the x labels; p + ncols >= npair
        # also catches a panel whose below-neighbor is switched off
        if p + ncols >= len(pairs):
            ax.set_xlabel(x_label, fontsize=16)
        if p % ncols == 0:
            ax.set_ylabel(y_label, fontsize=16 if real else 14)
    # a grid larger than the pair count leaves blank panels at the
    # end; switching them off removes their frames
    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")
    if ylim is not None:
        axes.ravel()[0].set_ylim(-ylim, ylim)
    axes.ravel()[0].legend(fontsize=9, loc="lower left")
    fig.suptitle(title, fontsize=17)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.basename(path)}")


def photoz_convention_figures(u, out_dir, kind, ntomo, nlen,
                              x_min, x_max, mask_file,
                              grid, example="example1"):
    """Evaluate the four settings and write the two figures.

    Arguments:
      u         = the project's cocoa_test_utils module.
      out_dir   = the folder the PNG files go into (the figure
                  script's own folder, tests/data_vector/).
      kind      = "real" (xi_pm over theta) or "fourier" (C_l over
                  the bands).
      ntomo     = the number of source tomographic bins.
      nlen      = angular bins (real) or bands (fourier) per block.
      x_min     = the lower edge of the angular or band range.
      x_max     = the upper edge.
      mask_file = the scale-cut mask of the frozen contract (two
                  columns, entry index and 0/1; one column accepted).
      grid      = (nrows, ncols) of the panel grid; the pairs fill
                  it row by row.
      example   = the key of u.EXAMPLES to evaluate.

    Returns:
      nothing; photoz_zmid_*.png and photoz_interp_*.png are written
      into out_dir.

    Raises:
      RuntimeError when Cocoa is not activated; AssertionError when
      the frozen state does not match the manifest.
    """
    u.require_cocoa_environment()
    u.verify_frozen()
    dv = _datavectors(u, example)

    # the mask file has two columns (entry index, 0/1); keep the
    # second
    mask = np.loadtxt(mask_file)
    mask = mask[:, 1] if mask.ndim == 2 else mask
    npair = ntomo * (ntomo + 1) // 2
    # entries of one block: xi_+ or xi_- in real space, the leading
    # cosmic-shear block in Fourier space
    nblock = npair * nlen

    real = kind == "real"
    edges = np.geomspace(x_min, x_max, nlen + 1)
    if real:
        # the area-weighted centers of the log-spaced bins,
        # (2/3)(t_max^3 - t_min^3)/(t_max^2 - t_min^2) per bin, the
        # convention of cosmolike's angular bins
        x = (2.0 / 3.0) * (edges[1:]**3 - edges[:-1]**3) \
                        / (edges[1:]**2 - edges[:-1]**2)
    else:
        # geometric band centers
        x = np.sqrt(edges[1:] * edges[:-1])

    def frac(tag):
        """One setting's fractional difference against the default.

        Real space returns the (dxi_plus, dxi_minus) pair, Fourier
        space the single C_l array, each (npair, nlen) with NaN at
        masked bins. errstate silences numpy's division warnings on
        masked entries (0 in both vectors), which np.where replaces
        by NaN anyway.
        """
        ref, cur = dv[SETTINGS[0][0]], dv[tag]
        out = []
        for s in range(2 if real else 1):
            sl = slice(s * nblock, (s + 1) * nblock)
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.where(mask[sl] > 0, cur[sl] / ref[sl] - 1.0,
                             np.nan)
            out.append(d.reshape(npair, nlen))
        return tuple(out) if real else out[0]

    # the figure files and labels of the two spaces
    project = os.path.basename(os.path.dirname(u.TESTS_DIR))
    suffix = "dxi" if real else "dcl"
    x_label = r"$\theta$ [arcmin]" if real else r"$\ell$"
    y_unit_pct = "%"
    y_unit_1em4 = r"$10^{-4}$"
    if real:
        y_label_pct = rf"$\Delta\xi_\pm/\xi_\pm$ [{y_unit_pct}]"
        y_label_1em4 = rf"$\Delta\xi_\pm/\xi_\pm$ [{y_unit_1em4}]"
    else:
        y_label_pct = rf"$\Delta C_\ell/C_\ell$ [{y_unit_pct}]"
        y_label_1em4 = rf"$\Delta C_\ell/C_\ell$ [{y_unit_1em4}]"
    panel_fontsize = 15 if real else 13

    _plot({"Z_MID": frac("Z_MID")},
          os.path.join(out_dir, f"photoz_zmid_{suffix}.png"),
          f"{project} cosmic shear: n(z) z-column read as Z_MID "
          "instead of Z_LOW (frozen fiducial)",
          grid, x, x_label, y_label_pct, real, panel_fontsize,
          scale=100.0, ylim=ZMID_YLIM)
    _plot({"linear": frac("linear"), "steffen": frac("steffen")},
          os.path.join(out_dir, f"photoz_interp_{suffix}.png"),
          f"{project} cosmic shear: linear and Steffen n(z) "
          "interpolation vs cubic spline (frozen fiducial)",
          grid, x, x_label, y_label_1em4, real, panel_fontsize,
          scale=1.0e4, ylim=INTERP_YLIM)


# ---- the N-random-models figure ---------------------------------------------

# The floor of the delta chi2 axes and of the color scale: a
# logarithmic axis has no place for zero, and two settings whose chi2
# agree to 1e-6 are identical for every purpose of the 0.2 budget
NMODELS_DELTA_FLOOR = 1.0e-6

# The top of the delta chi2 color scale: the scale resolves the range
# around the 0.2 limit, and every delta above 100 is far outside the
# budget, so those points saturate at the top color instead of
# stretching the scale over decades nothing distinguishes
NMODELS_COLOR_CEILING = 100.0


def random_models_amplitude(point, fiducial):
    """Return the largest clustering-amplitude ratio of one drawn point.

    The galaxy power spectrum of lens bin i is proportional to
    A_s b_i^2. A numerical error that is a fixed fraction of the signal
    therefore grows by A_i = (A_s b_i^2) / (A_s b_i^2)_fid at a drawn
    point, and its delta chi2 against a covariance held at the
    fiducial grows by A_i^2. The figure plots the largest A_i of each
    point: the lens bin whose error is amplified most.

    Arguments:
      point    = the drawn {parameter name: value} point.
      fiducial = the frozen {parameter name: value} point, with the
                 same names.

    Returns:
      the largest A_i as a float, or None when the point samples no
      linear bias (a cosmic-shear configuration) or no As_1e9.
    """
    # the harness owns the definition, so the draw's cut and the
    # figure's axis can never disagree
    from cocoa_testing import largest_amplitude_ratio
    return largest_amplitude_ratio(point, fiducial)


def random_models_triangle(points, deltas, fiducial, path, limit,
                           colorbarlabel, held=None, label=None):
    """Draw the N-random-models delta chi2 as an Omega_m - amplitude triangle.

    Two quantities set the numerical error of a random point. The
    matter density Omega_m raises the shear signal and pushes the
    matter power deeper into the nonlinear regime, where the relative
    error itself grows. The clustering-amplitude ratio of
    random_models_amplitude multiplies the galaxy power, so a fixed
    relative error grows by its square against the fiducial
    covariance. The two often move together, so the figure is a
    triangle: the diagonal panels show delta chi2 against each
    quantity (logarithmic delta chi2, the limit as a dashed line), and
    the lower panel shows Omega_m against the amplitude ratio, which
    tells whether a failing point at low amplitude sits at high
    Omega_m. All x axes are linear. Every point is colored by its
    delta chi2 on one logarithmic scale shared by the panels and the
    color bar, and the largest deltas are drawn last, on top. A
    configuration that samples no linear bias (cosmic shear) draws the
    Omega_m panel only.

    Models that held Omega_m, As and the biases at the fiducial (the
    odd models of a run with a hold list) are drawn as black-edged
    diamonds: they stack at Omega_m = fiducial and amplitude ratio 1,
    and the height of the stack is the error the other parameters
    cause.

    Arguments:
      points        = the drawn points, one {parameter name: value}
                      dict per model, in model order; they must
                      include omegam.
      deltas        = the per-model delta chi2 floats, in the same
                      order.
      fiducial      = the frozen {parameter name: value} point.
      path          = the PNG path to write.
      limit         = the delta chi2 of the dashed line (the pass
                      limit, or the 0.2 band of an advisory check).
      colorbarlabel = the text of the color bar.
      held          = None, or one bool per model: True draws the
                      model as a held diamond.
      label         = None, or the text naming the case (project,
                      example, number of models, seed base), drawn
                      above the panels.

    Returns:
      nothing; the PNG file is written and the figure closed.

    Raises:
      ValueError when the points and the deltas differ in number, when
      a delta is not finite, or when the points do not sample omegam.
    """
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    if len(points) != len(deltas):
        raise ValueError(
            f"{len(points)} points but {len(deltas)} delta chi2 values: "
            "pass one delta per drawn point, in model order")
    delta_values = np.array(deltas, dtype=float)
    if not np.all(np.isfinite(delta_values)):
        raise ValueError(
            f"non-finite delta chi2 among {deltas}: every model must "
            "evaluate before the figure is drawn")
    if "omegam" not in points[0]:
        raise ValueError(
            f"the points do not sample omegam (names {list(points[0])}): "
            "the figure needs the drawn matter density")
    # a delta chi2 is a chi2 against the point's own vector, so it is
    # negative only by round-off; the floor puts zero on the log axes
    delta_plot = np.maximum(np.abs(delta_values), NMODELS_DELTA_FLOOR)
    omegam = np.array([point["omegam"] for point in points])
    amplitudes = []
    for point in points:
        amplitudes.append(random_models_amplitude(point, fiducial))
    has_amplitude = amplitudes[0] is not None
    omegam_label = r"$\Omega_\mathrm{m}$"
    amplitude_label = r"$A_\mathrm{s} b^2/(A_\mathrm{s} b^2)_\mathrm{fid}$"

    delta_low = min(float(np.min(delta_plot)), limit) / 3.0
    delta_high = max(float(np.max(delta_plot)), limit) * 3.0
    # the scale stops at NMODELS_COLOR_CEILING: larger deltas saturate
    color_high = NMODELS_COLOR_CEILING
    color_low = min(float(np.min(delta_plot)), color_high / 10.0)
    norm = LogNorm(vmin=color_low, vmax=color_high, clip=True)
    if held is None:
        held_mask = np.zeros(len(points), dtype=bool)
    else:
        held_mask = np.array(held, dtype=bool)
    # each kind of model, drawn in its own call: (index array, marker,
    # edge color, legend label). argsort lists a kind's models from the
    # smallest delta to the largest, so the largest land on top.
    kinds = []
    for mask, marker, edge, kind_label in (
            (~held_mask, "o", "white", "drawn across the prior"),
            (held_mask, "D", "black",
             r"$\Omega_\mathrm{m}$, $A_\mathrm{s}$, $b$ at the fiducial")):
        indices = np.nonzero(mask)[0]
        if len(indices) == 0:
            continue
        indices = indices[np.argsort(delta_plot[indices])]
        kinds.append((indices, marker, edge, kind_label))

    def scatter_kinds(ax, x_values, y_values):
        """Draw every kind of model at (x, y); return the last mappable."""
        mappable = None
        for indices, marker, edge, kind_label in kinds:
            mappable = ax.scatter(x_values[indices], y_values[indices],
                                  c=delta_plot[indices], cmap="viridis",
                                  norm=norm, s=55, marker=marker,
                                  edgecolors=edge, linewidths=0.6)
        return mappable

    def delta_panel(ax, x_values):
        """One diagonal panel: delta chi2 against one quantity."""
        mappable = scatter_kinds(ax, x_values, delta_plot)
        ax.set_yscale("log")
        ax.set_ylim(delta_low, delta_high)
        ax.axhline(limit, color="0.5", linestyle="--", linewidth=1.0)
        ax.grid(alpha=0.3)
        return mappable

    if not has_amplitude:
        fig, ax = plt.subplots(figsize=(5.8, 4.6))
        mappable = delta_panel(ax, omegam)
        ax.set_xlabel(omegam_label, fontsize=13)
        ax.set_ylabel(r"$\Delta\chi^2$", fontsize=13)
        # extend="max" draws the triangle that marks the saturation
        colorbar = fig.colorbar(mappable, ax=ax, pad=0.02, extend="max")
        legend_axes = ax
        legend_place = "upper left"
        title_height = 0.98
    else:
        amplitude = np.array(amplitudes)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9.5, 9),
                                 gridspec_kw={"wspace": 0.06,
                                              "hspace": 0.06})
        # top left: delta chi2 against Omega_m, sharing the x range of
        # the plane below it
        delta_panel(axes[0][0], omegam)
        axes[0][0].set_ylabel(r"$\Delta\chi^2$", fontsize=13)
        axes[0][0].tick_params(labelbottom=False)
        # bottom left: the Omega_m - amplitude plane
        mappable = scatter_kinds(axes[1][0], omegam, amplitude)
        axes[1][0].set_xlabel(omegam_label, fontsize=13)
        axes[1][0].set_ylabel(amplitude_label, fontsize=13)
        axes[1][0].grid(alpha=0.3)
        axes[0][0].set_xlim(axes[1][0].get_xlim())
        # bottom right: delta chi2 against the amplitude; its delta
        # axis moves to the right, the left edge belonging to the row
        delta_panel(axes[1][1], amplitude)
        axes[1][1].yaxis.tick_right()
        axes[1][1].set_xlabel(amplitude_label, fontsize=13)
        # the empty upper-right corner holds the color bar (figure
        # coordinates: left, bottom, width, height)
        axes[0][1].axis("off")
        colorbar_axes = fig.add_axes([0.60, 0.58, 0.03, 0.30])
        colorbar = fig.colorbar(mappable, cax=colorbar_axes,
                                extend="max")
        # the legend sits in the same empty corner, under the color bar
        legend_axes = axes[0][1]
        legend_place = "lower left"
        # the default top of the panels is 0.88: the title sits just above
        title_height = 0.93
    colorbar.set_label(colorbarlabel, fontsize=12)
    if label is not None:
        fig.suptitle(label, fontsize=13, y=title_height)
    if len(kinds) > 1:
        # the legend names the two kinds by their markers alone (gray,
        # since the color is the delta chi2)
        handles = []
        for indices, marker, edge, kind_label in kinds:
            handles.append(Line2D([], [], linestyle="", marker=marker,
                                  markersize=8, markerfacecolor="0.6",
                                  markeredgecolor=edge, label=kind_label))
        legend_axes.legend(handles=handles, loc=legend_place,
                           frameon=False, fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
