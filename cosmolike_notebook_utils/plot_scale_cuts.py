"""Per-bin scale cuts against k_max, for the EXAMPLE_EVALUATE notebooks.

A scale cut removes the small scales a model cannot be trusted to
predict. The UV-fraction cut of arXiv:2603.13031 turns one wavenumber
k_max into a cut for every tomographic bin: each spectrum is split
into the part sourced by modes k <= k_max/D(z) (the IR part) and the
rest (the UV part), with D(z) the linear growth factor, D(0) = 1, and
a data point is kept while its UV fraction stays below a threshold
alpha. k_max is therefore the z = 0 value of a cut that moves to
larger k at higher redshift, and the axes of this module say so.

The compiled interfaces compute the cuts (gg_scale_cuts_tomo for the
Fourier C_gg band powers returns l_max per lens bin and per k_max).
This module only draws them: one curve per bin of the per-bin cut
(l_max in Fourier space, theta_min in real space) against k_max.

Conventions shared with the plot_datavectors family:

- show = None returns (fig, axes); show = 1 draws the figure.
- Malformed input prints one message that names the problem and
  returns 0 before any figure is created.
- Bin arguments count from 0, as the arrays do; the bin labels of the
  legend count from 1.
- The functions never change rcParams: fonts and usetex belong to the
  notebook.

Figure style: every bin is encoded by three cues at once, so the eye
separates the curves even in grayscale: a color of the Okabe-Ito
palette (colorblind safe, with no red next to green), a line width
and a marker. The legend has no frame and is kept off the data by a
check in display coordinates: every drawn line is sampled densely,
and a legend position is accepted only when no sample falls inside
the legend box.

Pure matplotlib and numpy: nothing here touches CAMB or a compiled
interface, so every project shares the module unchanged.
"""

import itertools

import numpy as np
from matplotlib import pyplot as plt


# Okabe-Ito colors in the order of use. Yellow (#F0E442) is left out:
# it reads only as a thick line or a fill, and the curves of this plot
# take thin widths too. Bluish green (#009E73) is left out because the
# list holds vermillion, and the two must never sit side by side.
_BIN_COLORS = ["#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#D55E00", "black"]

# Line widths in points, widest first; four steps stay distinguishable
# at notebook size, and their cycle (length 4) is coprime with the color
# cycle (length 6), so the (color, width) pair of a bin repeats only
# after 12 bins.
_BIN_LINEWIDTHS = [2.4, 1.8, 1.3, 0.9]

# One marker per bin up to 12 bins: filled shapes that differ in
# outline, so a bin is identified even where two curves share a color.
_BIN_MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "h", "P", "X", "*"]

# Legend corners tried, in this order, before the y headroom grows.
_LEGEND_CORNERS = ["upper left", "lower right", "upper right", "lower left"]

# Headroom steps for the legend, as fractions of the data's y span
# (measured in log10 for a log axis): 0 keeps the data range. The search
# stops at the first step whose legend clears the data; it ends at 20%,
# because more headroom leaves a band of white space, and a legend above
# the axes reads better than that.
_HEADROOM_STEPS = [0.0, 0.1, 0.2]

# Most legend columns of a legend above the axes: five entries per row
# keep the row within the width of a notebook figure.
_ABOVE_MAX_COLUMNS = 5


def _line_points(ax):
    """Returns display-coordinate samples along every drawn line of ax.

    Each line's vertices are converted to display coordinates (pixels)
    and every segment between two finite vertices is sampled at 60
    points. Sampling in display space is what makes the check exact on
    a log axis: a segment drawn straight on screen is straight there.
    Only samples inside the axes box are returned, since a line is
    clipped at the axes edges.

    Arguments:
      ax = the matplotlib axes whose lines (ax.get_lines(), the curves
           and the reference lines) are sampled.

    Returns:
      2D float array (n_points, 2) of (x, y) pixel positions.
    """
    box = ax.get_window_extent()
    # t holds 60 fractions from 0 to 1 as a column, so t*(b - a) is the
    # (60, 2) array of points between vertices a and b
    t = np.linspace(0.0, 1.0, 60)[:, None]
    pieces = [np.zeros((0, 2))]
    for line in ax.get_lines():
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
        vertices = np.column_stack([xdata, ydata])
        display_xy = line.get_transform().transform(vertices)
        finite = np.all(np.isfinite(display_xy), axis=1)
        pieces.append(display_xy[finite])
        for i in range(len(display_xy) - 1):
            if finite[i] and finite[i + 1]:
                segment = display_xy[i] + t*(display_xy[i + 1] - display_xy[i])
                pieces.append(segment)
    points = np.vstack(pieces)
    inside = ((points[:, 0] >= box.x0) & (points[:, 0] <= box.x1)
              & (points[:, 1] >= box.y0) & (points[:, 1] <= box.y1))
    return points[inside]


def _legend_clear(fig, ax, legend):
    """Tells whether a legend box holds no sample of a drawn line.

    The legend box is padded by 3% in width and 8% in height, so a
    curve that only grazes the box edge also counts as touching.

    Arguments:
      fig    = the figure (drawn once here, so the legend has its size).
      ax     = the axes whose lines are checked.
      legend = the matplotlib Legend to test.

    Returns:
      True when no line sample falls inside the padded legend box.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = legend.get_window_extent(renderer).expanded(1.03, 1.08)
    points = _line_points(ax)
    hit = ((points[:, 0] > box.x0) & (points[:, 0] < box.x1)
           & (points[:, 1] > box.y0) & (points[:, 1] < box.y1))
    return not np.any(hit)


def _grown_limits(low, high, fraction, side, log):
    """Returns axis limits widened on one side by a fraction of the span.

    Arguments:
      low, high = the current limits (positive for a log axis).
      fraction  = the widening as a fraction of the span (0 = none).
      side      = "top" widens above high, "bottom" below low.
      log       = True measures the span in log10 (a log axis).

    Returns:
      (low, high) after the widening.
    """
    if log:
        span = np.log10(high) - np.log10(low)
        if side == "top":
            return (low, high*10.0**(fraction*span))
        return (low/10.0**(fraction*span), high)
    span = high - low
    if side == "top":
        return (low, high + fraction*span)
    return (low - fraction*span, high)


def plot_scale_cuts(kmax, cut, ylabel, binlabels = None, yscale = "log",
                    xscale = "linear", ylim = None, yref = None,
                    linewidth = None, marker = None, markersize = 5,
                    legend = 1, legendloc = None, legendncols = None,
                    legendfontsize = None, yaxislabelsize = 16,
                    xaxislabelsize = 16, yaxisticklabelsize = 13,
                    xaxisticklabelsize = 13, figsize = (8, 5), show = 1):
    """Draws a per-bin scale cut against k_max, one curve per bin.

    The y quantity is whatever per-bin cut the caller computed at each
    k_max: l_max of the Fourier band powers (gg_scale_cuts_tomo returns
    it as 'lmax', the center of the last band kept) or theta_min of a
    real-space correlation function. The x axis is k_max in h/Mpc at
    z = 0: each bin is cut at k_max/D(z), D the linear growth factor
    with D(0) = 1, so the drawn k_max is the z = 0 value of a cut that
    moves to larger k at higher redshift.

    Each bin gets a color, a line width and a marker of its own
    (module docstring). The legend is placed by a check in display
    coordinates: with legendloc = None the four corners are tried
    first, then the y range grows above (or below) the data by the
    least step, at most 20% of the span, that frees an upper (or
    lower) corner, and as the last resort the legend moves above the
    axes, in up to five columns, where it cannot touch a curve. A
    requested corner that touches a curve is refused with an error
    instead of being drawn over the data.

    Arguments:
      kmax      = 1D array of k_max values in h/Mpc at z = 0, finite,
                  positive and strictly ascending (n_k entries).
      cut       = 2D array (n_bins, n_k): the cut of bin b at kmax[k]
                  is cut[b, k] (for example the first nautos rows of
                  the 'lmax' entry of gg_scale_cuts_tomo).
      ylabel    = the y-axis label (LaTeX string), e.g.
                  r"$\\ell_{\\rm max}$".
      binlabels = one legend label per bin, or None for "bin 1",
                  "bin 2", ... (bins count from 1 in labels).
      yscale, xscale = "log" or "linear" for each axis. On a log y axis
                  an entry <= 0 (gg_scale_cuts_tomo writes 0 when no
                  band survives) is not drawn: that point is missing
                  from its curve.
      ylim      = (low, high) of the y axis before any legend headroom,
                  or None to fit the data with 10% padding (in log10
                  on a log axis).
      yref      = list of y values drawn as thin gray dashed reference
                  lines without a legend entry (for example the first
                  and last band centers), or None.
      linewidth, marker = lists cycled across bins, or None for the
                  module's width and marker cycles.
      markersize = marker size in points.
      legend    = 1 draws the bin legend; None draws none.
      legendloc = None for the automatic placement (corners, then
                  headroom, then above the axes); one of the four
                  corner strings ("upper left", "lower right",
                  "upper right", "lower left") to request that corner,
                  refused with a RuntimeError when it touches a curve;
                  "above" for a legend above the axes.
      legendncols = legend columns inside the axes, or None for 1 up to
                  four bins and 2 above; a legend above the axes takes
                  up to five columns.
      legendfontsize = legend font size in points, or None for
                  matplotlib's legend.fontsize rcParam.
      yaxislabelsize, xaxislabelsize = axis-label font sizes in points;
      yaxisticklabelsize, xaxisticklabelsize = tick-number sizes.
      figsize   = (width, height) in inches.
      show      = 1 draws the figure; None returns (fig, axes), axes
                  being the one matplotlib Axes.

    Returns:
      0 on malformed input (a printed message names the problem),
      None after drawing, or (fig, axes) when show is None.

    Raises:
      RuntimeError when a requested legendloc touches a curve; the
      message names the location and the automatic alternative.
    """
    # --- validation, before any figure exists ---------------------------
    kmax = np.asarray(kmax, dtype=float)
    cut = np.asarray(cut, dtype=float)
    if kmax.ndim != 1 or len(kmax) < 1:
        print("Bad Input (kmax): need a 1D array with at least one entry")
        return 0
    if not np.all(np.isfinite(kmax)) or np.any(kmax <= 0):
        print("Bad Input (kmax): every k_max must be finite and positive")
        return 0
    if len(kmax) > 1 and np.any(np.diff(kmax) <= 0):
        print("Bad Input (kmax): the k_max values must be strictly ascending")
        return 0
    if cut.ndim != 2 or cut.shape[1] != len(kmax):
        print(f"Bad Input (cut): shape {cut.shape}, need (n_bins, {len(kmax)}),"
              " one row per bin and one column per k_max")
        return 0
    if not np.all(np.isfinite(cut)):
        print("Bad Input (cut): every entry must be finite")
        return 0
    nbins = cut.shape[0]
    if not (binlabels is None) and len(binlabels) != nbins:
        print(f"Bad Input (binlabels): {len(binlabels)} labels for {nbins} bins")
        return 0
    if yscale not in ("log", "linear") or xscale not in ("log", "linear"):
        print("Bad Input (yscale/xscale): each must be 'log' or 'linear'")
        return 0
    ylog = (yscale == "log")
    if ylog and not np.any(cut > 0):
        print("Bad Input (cut): a log y axis needs at least one entry > 0")
        return 0
    if not (ylim is None):
        if len(ylim) != 2 or not (ylim[0] < ylim[1]) or (ylog and ylim[0] <= 0):
            print("Bad Input (ylim): need (low, high) with low < high"
                  " (and low > 0 on a log axis)")
            return 0
    if not (yref is None):
        yref = [float(value) for value in yref]
        if ylog and any(value <= 0 for value in yref):
            print("Bad Input (yref): reference values must be > 0 on a log axis")
            return 0
    legendloc_known = ((legendloc is None) or (legendloc == "above")
                       or (legendloc in _LEGEND_CORNERS))
    if not legendloc_known:
        print("Bad Input (legendloc): use None, 'above' or one of "
              + ", ".join(_LEGEND_CORNERS))
        return 0

    if binlabels is None:
        binlabels = []
        for b in range(nbins):
            binlabels.append(f"bin {b + 1}")
    if legendncols is None:
        legendncols = 1 if nbins <= 4 else 2

    # itertools.cycle repeats a list forever: next(...) pulls the entry
    # of the following bin, wrapping at the end
    colorcycler = itertools.cycle(_BIN_COLORS)
    if linewidth is None:
        widthcycler = itertools.cycle(_BIN_LINEWIDTHS)
    else:
        widthcycler = itertools.cycle(linewidth)
    if marker is None:
        markercycler = itertools.cycle(_BIN_MARKERS)
    else:
        markercycler = itertools.cycle(marker)

    # --- the curves -----------------------------------------------------
    fig, ax = plt.subplots(figsize = figsize)
    drawn_values = []
    for b in range(nbins):
        values = cut[b].copy()
        if ylog:
            # a log axis cannot show 0 (no band kept): the point is left
            # out of the curve instead of being clipped to the bottom
            values[values <= 0] = np.nan
        drawn_values.append(values[np.isfinite(values)])
        ax.plot(kmax,
                values,
                color = next(colorcycler),
                linewidth = next(widthcycler),
                marker = next(markercycler),
                markersize = markersize,
                label = binlabels[b])

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    # the x range is the k_max grid plus a small margin on each side, so
    # the first and last markers sit inside the frame: 4% of the span on
    # a linear axis, a factor 1.08 on a log axis
    if len(kmax) > 1:
        xpad = 0.04*(kmax[-1] - kmax[0])
    else:
        xpad = 0.04*kmax[0]
    if xscale == "log":
        ax.set_xlim(kmax[0]/1.08, kmax[-1]*1.08)
    else:
        ax.set_xlim(kmax[0] - xpad, kmax[-1] + xpad)

    # the y range: the data with 10% padding (in log10 on a log axis),
    # the reference lines included, unless the caller fixed it
    if ylim is None:
        all_values = np.concatenate(drawn_values)
        if not (yref is None):
            all_values = np.concatenate((all_values, np.array(yref)))
        low = float(np.min(all_values))
        high = float(np.max(all_values))
        if ylog:
            span = max(np.log10(high) - np.log10(low), 0.1)
            ylim = (low/10.0**(0.1*span), high*10.0**(0.1*span))
        else:
            span = max(high - low, 1e-12)
            ylim = (low - 0.1*span, high + 0.1*span)
    ax.set_ylim(ylim)

    if not (yref is None):
        for value in yref:
            ax.axhline(y = value, color = "gray", linestyle = "dashed",
                       linewidth = 0.8, zorder = 0)

    ax.set_xlabel(r"$k_{\rm max}$ at $z = 0$ [$h\,{\rm Mpc}^{-1}$]",
                  fontsize = xaxislabelsize)
    ax.set_ylabel(ylabel, fontsize = yaxislabelsize)
    for item in ax.get_yticklabels():
        item.set_fontsize(yaxisticklabelsize)
    for item in ax.get_xticklabels():
        item.set_fontsize(xaxisticklabelsize)

    # --- the legend, kept off the data ----------------------------------
    if not (legend is None):
        # the options every placement shares (frameless, as the house
        # style asks); the placements below add loc and the columns
        legend_options = {
            "fontsize": legendfontsize,
            "frameon": False,
            "handlelength": 2.4,
            "columnspacing": 1.0,
            "borderaxespad": 0.4,
        }
        # a legend above the axes: its lower center is pinned to the
        # top center of the axes (bbox_to_anchor in axes fractions)
        above_columns = min(nbins, _ABOVE_MAX_COLUMNS)
        if legendloc == "above":
            ax.legend(loc = "lower center", bbox_to_anchor = (0.5, 1.0),
                      ncols = above_columns, **legend_options)
        elif not (legendloc is None):
            requested = ax.legend(loc = legendloc, ncols = legendncols,
                                  **legend_options)
            if not _legend_clear(fig, ax, requested):
                plt.close(fig)
                raise RuntimeError(
                    f"plot_scale_cuts: the legend at legendloc = '{legendloc}'"
                    " touches a curve; pass legendloc = None for the"
                    " automatic placement or 'above' for a legend above"
                    " the axes")
        else:
            placed = False
            base_low, base_high = ax.get_ylim()
            # step 1: the four corners on the data range; step 2: the
            # least headroom above (upper corners) or below (lower
            # corners) that clears one of them
            for fraction in _HEADROOM_STEPS:
                for corner in _LEGEND_CORNERS:
                    if corner.startswith("upper"):
                        side = "top"
                    else:
                        side = "bottom"
                    limits = _grown_limits(base_low, base_high, fraction,
                                           side, ylog)
                    ax.set_ylim(limits)
                    trial = ax.legend(loc = corner, ncols = legendncols,
                                      **legend_options)
                    if _legend_clear(fig, ax, trial):
                        placed = True
                        break
                    trial.remove()
                if placed:
                    break
            if not placed:
                # step 3: above the axes, where no curve is drawn
                ax.set_ylim(base_low, base_high)
                ax.legend(loc = "lower center", bbox_to_anchor = (0.5, 1.0),
                          ncols = above_columns, **legend_options)

    if not (show is None):
        fig.show(warn=False)
    else:
        return (fig, ax)
