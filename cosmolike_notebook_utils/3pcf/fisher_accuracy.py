"""Draw saved Fisher-accuracy diagnostics without calculating a forecast.

The caller supplies numbers and their scientific meanings. A noise prediction
assumes independent errors; a partial-column replacement uses actual selected
derivatives; a full check recalculates every derivative. These are different
tests and remain separately labeled. Functions never set rcParams, save files,
load a compiled interface, or choose a numerical acceptance threshold.
"""

import numpy as np
from matplotlib import pyplot as plt


def _finish(fig, axes, handles, labels, legendfontsize, show):
    """Place a shared legend outside all data rectangles and verify clearance.

    Disjoint display-coordinate rectangles are a stronger check than sampling
    the plotted lines: no curve inside an axis can intersect this legend.
    Labels below categorical axes have their own reserved bottom margin.
    """
    if handles:
        legend = fig.legend(
            handles=handles, labels=labels, loc="upper center",
            bbox_to_anchor=(0.5, 1.0), ncol=len(handles), frameon=False,
            fontsize=legendfontsize,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = legend.get_window_extent(renderer=renderer)
        for axis in np.asarray(axes, dtype=object).flat:
            if bounds.overlaps(axis.get_window_extent(renderer=renderer)):
                plt.close(fig)
                raise ValueError("The outside legend overlaps the axes; enlarge figsize.")
    if show is not None:
        plt.show()
        return None
    return fig, axes


def _range(values):
    """Keep zero and pad the observed linear range without an invented floor."""
    finite = values[np.isfinite(values)]
    lower = min(0.0, float(np.min(finite)))
    upper = max(0.0, float(np.max(finite)))
    width = upper - lower
    if width == 0.0:
        # Matplotlib chooses a nonsingular display range around exact zero.
        # No nonzero scientific datum or error threshold is inserted.
        return None
    return lower - 0.08 * width, upper + 0.10 * width


def plot_noise_contributions(
    contributions, names, selected, *, figsize=(10, 3.8), show=1,
    legendfontsize=12, xaxisticklabelsize=11, yaxisticklabelsize=11,
    yaxislabelsize=13,
):
    """Show each parameter's measured share of the baseline noise estimate.

    Arguments:
      contributions = nonnegative R_squared * variance * noise arrays [npar].
      names = display labels [npar], in the same order; no sorting is inferred.
      selected = boolean [npar]; these columns are measured at new settings.
      figsize = figure size in inches; show = None returns the figure.
      legendfontsize = shared outside legend size.
      xaxisticklabelsize, yaxisticklabelsize = tick font sizes.
      yaxislabelsize = font size of the percentage-share axis label.
    The sum must be positive to define a percentage. An omitted column's
    small baseline share is not a bound on its unknown noise in another run.
    Returns: 0 for bad input, None after showing, or (fig, axes) for show=None.
    """
    try:
        values = np.asarray(contributions, dtype=float)
        keep = np.asarray(selected)
        if (values.ndim != 1 or values.size == 0 or len(names) != values.size
                or keep.shape != values.shape or keep.dtype.kind != "b"
                or not np.all(np.isfinite(values)) or np.any(values < 0.0)):
            raise ValueError("finite nonnegative contributions and matching labels/mask")
        total = float(np.sum(values))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("a positive finite total is needed for percentage shares")
        percentages = 100.0 * (values / total)
    except (TypeError, ValueError) as error:
        print(f"Bad Input (noise contributions): {error}")
        return 0
    fig, axis = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.28, top=0.80)
    positions = np.arange(values.size)
    handles = []
    labels = []
    for mask, color, hatch, label in (
        (keep, "#0072B2", None, "Selected for new settings"),
        (~keep, "#E69F00", "///", "Measured only at the baseline"),
    ):
        if np.any(mask):
            bars = axis.bar(
                x=positions[mask], height=percentages[mask], width=0.72,
                color=color, edgecolor="black", linewidth=0.6, hatch=hatch,
            )
            handles.append(bars)
            labels.append(label)
    axis.set_xticks(ticks=positions, labels=names, rotation=45, ha="right")
    axis.set_ylabel("Share of baseline noise estimate [%]", fontsize=yaxislabelsize)
    axis.tick_params(axis="x", labelsize=xaxisticklabelsize)
    axis.tick_params(axis="y", labelsize=yaxisticklabelsize)
    axis.set_ylim(0.0, 1.10 * float(np.max(percentages)))
    return _finish(
        fig=fig, axes=np.asarray([axis]), handles=handles, labels=labels,
        legendfontsize=legendfontsize, show=show,
    )


def _ratio_panel(ratios, labels, legend, figsize, show, legendfontsize):
    """Draw two differently interpreted ratios on the same percent scale."""
    try:
        values = np.asarray(ratios, dtype=float)
        if (values.ndim != 2 or values.shape[0] != 2 or values.shape[1] == 0
                or len(labels) != values.shape[1] or len(legend) != 2
                or np.any(np.isinf(values))
                or np.any(values[np.isfinite(values)] <= 0.0)
                or not np.any(np.isfinite(values))):
            raise ValueError(
                "two positive ratio rows, matching labels; NaN means unavailable"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            differences = 100.0 * (values - 1.0)
        if np.any(np.isinf(differences)):
            raise ValueError("percentage conversion overflowed")
    except (TypeError, ValueError) as error:
        print(f"Bad Input (FoM ratios): {error}")
        return 0
    fig, axis = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.30, top=0.80)
    positions = np.arange(values.shape[1])
    handles = []
    for index, (color, marker, width, style) in enumerate((
        ("#0072B2", "o", 2.4, "solid"),
        ("#E69F00", "s", 1.8, "dashed"),
    )):
        line, = axis.plot(
            positions, differences[index], color=color, marker=marker,
            linewidth=width, linestyle=style, markersize=5,
        )
        handles.append(line)
    axis.axhline(y=0.0, color="0.45", linewidth=0.8, zorder=0)
    axis.set_xticks(ticks=positions, labels=labels, rotation=40, ha="right")
    axis.set_ylabel("FoM change from baseline [%]", fontsize=13)
    axis.tick_params(axis="both", labelsize=11)
    limits = _range(values=differences)
    if limits is not None:
        axis.set_ylim(*limits)
    return _finish(
        fig=fig, axes=np.asarray([axis]), handles=handles, labels=legend,
        legendfontsize=legendfontsize, show=show,
    )


def plot_accuracy_screen(
    ratios, labels, *, figsize=(11, 4), show=1, legendfontsize=11,
):
    """Compare a diagonal-noise expectation with selected derivative changes.

    Arguments:
      ratios = [2,ncase]: shared noise prediction divided by the baseline
               prediction; then actual selected-column replacement FoM/base.
      labels = numerical-setting labels [ncase].
      figsize = figure size in inches; show = None returns the figure.
      legendfontsize = font size of the shared outside legend.
    NaN marks unavailable or rank-unresolved entries and creates a gap. Neither
    row is a full candidate Fisher recalculation. Ratios display as 100*(r-1).
    Returns: 0 for bad input, None after showing, or (fig, axes) for show=None.
    """
    return _ratio_panel(
        ratios=ratios, labels=labels,
        legend=("Diagonal-noise expectation", "Selected derivative replacement"),
        figsize=figsize, show=show, legendfontsize=legendfontsize,
    )


def plot_full_checks(
    ratios, labels, *, figsize=(9, 4), show=1, legendfontsize=12,
):
    """Display full recalculations at both finite-difference step lengths.

    Arguments:
      ratios = [2,ncase]: full h and h/2 FoMs divided by one fixed baseline.
      labels = baseline/candidate display labels [ncase].
      figsize = figure size in inches; show = None returns the figure.
      legendfontsize = shared outside legend font size.
    The caller supplies only completed calculations. NaN denotes an unresolved
    Fisher matrix; missing files are reported by the loader, not drawn as zero.
    Returns: 0 for bad input, None after showing, or (fig, axes) for show=None.
    """
    return _ratio_panel(
        ratios=ratios, labels=labels, legend=("All derivatives at h", "All at h/2"),
        figsize=figsize, show=show, legendfontsize=legendfontsize,
    )


def plot_xi_preflight(
    distances, labels, *, figsize=(11, 4), show=1,
):
    """Show saved covariance-weighted distances of fiducial xi vectors.

    Arguments:
      distances = nonnegative Delta_xi^T C_xi^-1 Delta_xi [ncase].
      labels = [ncase] labels naming N_ell, N_theta and ell_max.
      figsize = figure size in inches; show = None returns the figure.
    Exact zero distances are omitted from the logarithmic axis and reported;
    no display floor is substituted. The finest vector is a numerical
    reference, not an observed data vector or an exact physical solution.
    Returns: 0 for bad input, None after showing, or (fig, axes) for show=None.
    """
    try:
        values = np.asarray(distances, dtype=float)
        if (values.ndim != 1 or values.size == 0 or len(labels) != values.size
                or not np.all(np.isfinite(values)) or np.any(values < 0.0)):
            raise ValueError("finite nonnegative distances and matching case labels")
        positive = values > 0.0
        if not np.any(positive):
            raise ValueError(
                "all distances are exactly zero; a logarithmic plot is undefined"
            )
    except (TypeError, ValueError) as error:
        print(f"Bad Input (xi preflight): {error}")
        return 0
    zeros = int(np.count_nonzero(~positive))
    if zeros:
        print(f"The log plot omits {zeros} exact-zero reference/identical cases.")
    positions = np.arange(values.size)[positive]
    kept_labels = []
    for index in positions:
        kept_labels.append(labels[index])
    fig, axis = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.31, top=0.96)
    axis.plot(positions, values[positive], color="#0072B2", marker="o",
              linewidth=2.4, linestyle="solid", markersize=5)
    axis.set_yscale("log")
    axis.set_xticks(ticks=positions, labels=kept_labels, rotation=40, ha="right")
    axis.set_xlabel(r"Quadrature settings: $N_\ell$ / $N_\theta$ / $\ell_{\max}$",
                    fontsize=13)
    axis.set_ylabel(r"$\Delta\xi^{\mathsf{T}} C_\xi^{-1}\Delta\xi$", fontsize=14)
    axis.tick_params(axis="both", labelsize=11)
    return _finish(
        fig=fig, axes=np.asarray([axis]), handles=[], labels=[],
        legendfontsize=12, show=show,
    )
