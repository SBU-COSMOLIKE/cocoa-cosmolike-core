"""Plot supplied three-point arrays without changing their scientific model.

Map3 is an aperture third moment. Gamma contains the four complex natural
shear correlations. Neither a signed IA component nor an imaginary Gamma
that vanishes is an excluded measurement. Exclusion is an explicit flag.
No input is changed, interpolated, reordered or saved by these functions.

Reference comparisons have four named meanings: pointwise percent,
difference divided by a supplied standard deviation, percent of one fixed
reference scale per panel, or an absolute difference in the original units.
Pointwise percent is undefined at a zero reference. Such points stay masked
and are counted in a diagnostic; absolute and sigma modes retain them.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from ..plot_datavectors import _hide_glued_edge_ticklabels


def _frames(curves, name, complex_values=False):
    """Validate (coordinate, [panel,coordinate]) pairs before drawing.

    Coordinates must agree exactly across curves: interpolating a reference
    would add a numerical operation to a plot advertised as a comparison.
    """
    if not isinstance(curves, (list, tuple)) or not curves:
        raise ValueError(f"{name} needs a nonempty list of (coordinate, values)")
    grid = None
    arrays = []
    for entry in curves:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"{name} entries must contain coordinate and values")
        coordinate = np.asarray(entry[0], dtype=float)
        values = np.asarray(entry[1])
        if (coordinate.ndim != 1 or coordinate.size == 0
                or not np.all(np.isfinite(coordinate))
                or np.any(np.diff(coordinate) <= 0)):
            raise ValueError("coordinate must be finite, increasing and nonempty")
        if (values.ndim != 2 or values.shape[1] != coordinate.size
                or values.shape[0] == 0 or not np.all(np.isfinite(values))):
            raise ValueError("values must be finite [panel,coordinate] arrays")
        if not complex_values and np.iscomplexobj(values):
            raise ValueError("these values must be real; select a Gamma component")
        if grid is None:
            grid = coordinate
        elif not np.array_equal(grid, coordinate):
            raise ValueError("all coordinate grids must match exactly")
        if arrays and values.shape != arrays[0].shape:
            raise ValueError("all curve arrays must have the same shape")
        arrays.append(values)
    return grid, np.asarray(arrays)


def _triplets(values, count):
    """Keep ordered zero-based source labels; print their one-based labels."""
    triples = np.asarray(values)
    if (triples.shape != (count, 3) or triples.dtype.kind not in "iu"
            or np.any(triples < 0)):
        raise ValueError("triplets must contain nonnegative integers [panel,3]")
    labels = []
    for first, second, third in triples:
        labels.append(f"({first+1},{second+1},{third+1})")
    return labels


def _residual(values, reference, mode, sigma, reference_scale):
    """Transform a curve bank [curve,panel,x] while retaining signed values."""
    difference = values - reference[None, :, :]
    if not np.all(np.isfinite(difference)):
        raise ValueError("subtraction overflowed; use consistent physical units")
    if mode == "absolute":
        return difference, r"$\Delta$", "original"
    if mode == "sigma":
        errors = np.asarray(sigma, dtype=float)
        if (errors.shape != reference.shape or np.any(errors <= 0)
                or not np.all(np.isfinite(errors))):
            raise ValueError("sigma needs finite positive values [panel,coordinate]")
        with np.errstate(over="ignore", invalid="ignore"):
            answer = difference / errors[None, :, :]
        if not np.all(np.isfinite(answer)):
            raise ValueError("sigma residual overflowed; check the error units")
        return answer, r"$\Delta/\sigma$", "sigma"
    if mode == "scaled":
        if reference_scale is None:
            scale = np.max(np.abs(reference), axis=1)
        else:
            scale = np.asarray(reference_scale, dtype=float)
            if scale.ndim == 0:
                scale = np.full(reference.shape[0], float(scale))
        if (scale.shape != (reference.shape[0],) or np.any(scale <= 0)
                or not np.all(np.isfinite(scale))):
            raise ValueError(
                "scaled residual needs positive reference_scale per panel; "
                "a zero reference has no automatic scale"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            answer = 100.0 * difference / scale[None, :, None]
        if not np.all(np.isfinite(answer)):
            raise ValueError("scaled residual overflowed; check reference_scale units")
        return answer, r"$100\Delta/s_{\rm ref}$ [%]", "scaled"
    if mode != "percent":
        raise ValueError("residual must be percent, sigma, scaled or absolute")
    answer = np.full(difference.shape, np.nan)
    # No division is evaluated where the reference is exactly zero. A NaN
    # is a display mask, not a replacement physical prediction.
    np.divide(difference, reference[None, :, :], out=answer,
              where=reference[None, :, :] != 0)
    answer *= 100.0
    if np.any(~np.isfinite(answer) & (reference[None, :, :] != 0)):
        raise ValueError("percent residual overflowed; use scaled or absolute mode")
    undefined = int(np.count_nonzero(reference == 0))
    if undefined:
        print(f"Percent residual undefined at {undefined} reference entries; "
              "use absolute or sigma mode to display those differences.")
    return answer, r"$100(f/f_{\rm ref}-1)$ [%]", "percent"


def _range(values):
    """Fit a signed linear band with zero and ten-percent padding."""
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lower = min(0.0, float(np.min(finite)))
    upper = max(0.0, float(np.max(finite)))
    span = upper - lower
    if span == 0:
        # The whole panel is exactly zero. This is an axis convention only.
        return -1.0, 1.0
    return lower - 0.1 * span, upper + 0.1 * span


def _styles(count, param, cmap, legend, marker, linestyle, linewidth):
    """Share curve colors with the parameter colorbar, without pale endpoints."""
    if legend is not None and len(legend) != count:
        raise ValueError("legend needs one label per curve")
    if param is None:
        category_colors = (
            "#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#D55E00", "black",
        )
        palette = []
        for index in range(count):
            palette.append(category_colors[index % len(category_colors)])
        mapping = None
    else:
        parameters = np.asarray(param, dtype=float)
        if parameters.shape != (count,) or not np.all(np.isfinite(parameters)):
            raise ValueError("param needs one finite value per curve")
        low = float(np.min(parameters))
        high = float(np.max(parameters))
        if low == high:
            low -= 0.5
            high += 0.5
        source = plt.get_cmap(cmap)
        sampled = source(np.linspace(0.12, 0.8, 256))
        readable = colors.LinearSegmentedColormap.from_list("threepcf", sampled)
        mapping = ScalarMappable(norm=colors.Normalize(low, high), cmap=readable)
        palette = mapping.to_rgba(parameters)
    settings = []
    for index in range(count):
        line = {
            "color": palette[index],
            "linewidth": (2.4, 1.8, 1.3, 0.9)[index % 4],
            "marker": ("o", "s", "D", "^")[index % 4],
            "linestyle": ["solid", "dashed", "dashdot", "dotted"][index % 4],
            "label": None if legend is None else legend[index],
        }
        for name, sequence in (("marker", marker), ("linestyle", linestyle),
                               ("linewidth", linewidth)):
            if sequence is not None:
                if (not isinstance(sequence, (list, tuple, np.ndarray))
                        or not len(sequence)):
                    raise ValueError(f"{name} must be a nonempty sequence")
                line[name] = sequence[index % len(sequence)]
        settings.append(line)
        # Validate style choices without creating a figure. A misspelled
        # linestyle must not leave a half-constructed notebook plot behind.
        Line2D([], [], **line)
    return settings, mapping


def _line_panels(
    curves, reference, labels, coordinate_label, quantity_label,
    param, colorbarlabel, sigma, residual, reference_scale,
    legend, marker, linestyle, linewidth, ylim, rescale, excluded,
    ncols, figsize, show, xscale, cmap, data=None,
    legendloc=None, legendfontsize=13, xaxislabelsize=17, yaxislabelsize=17,
    xaxisticklabelsize=13, yaxisticklabelsize=13,
    bintextpos=(0.06, 0.86), bintextsize=16, markersize=4,
    colorbar=1, colorbarshrink=0.75,
):
    """Draw checked real arrays using the shared data-vector panel conventions."""
    coordinate, values = _frames(curves=curves, name="curves")
    count, panels, _ = values.shape
    if len(labels) != panels:
        raise ValueError("one panel label is required for each value row")
    if xscale not in ("linear", "log"):
        raise ValueError("xscale must be linear or log")
    if xscale == "log" and np.any(coordinate <= 0):
        raise ValueError("a logarithmic coordinate must be strictly positive")
    if isinstance(ncols, bool) or int(ncols) != ncols or ncols < 1:
        raise ValueError("ncols must be a positive integer")
    ncols = min(int(ncols), panels)
    nrows = (panels + ncols - 1) // ncols
    if excluded is None:
        excluded = np.zeros(panels, dtype=bool)
    else:
        excluded = np.asarray(excluded)
        if excluded.shape != (panels,) or excluded.dtype.kind != "b":
            raise ValueError("excluded must be a Boolean flag per panel")
    settings, mapping = _styles(count=count, param=param, cmap=cmap,
                               legend=legend, marker=marker,
                               linestyle=linestyle, linewidth=linewidth)
    observed = None
    uncertainty = None
    if data is not None:
        observed_grid, observed_bank = _frames(curves=[data], name="data")
        if (not np.array_equal(observed_grid, coordinate)
                or observed_bank.shape[1:] != values.shape[1:]):
            raise ValueError("data must use exactly the curve grid and panel shape")
        observed = observed_bank[0]
        uncertainty = np.asarray(sigma, dtype=float)
        if (uncertainty.shape != observed.shape or np.any(uncertainty < 0)
                or not np.all(np.isfinite(uncertainty))):
            raise ValueError("data need finite nonnegative sigma [panel,coordinate]")
        if reference is not None:
            raise ValueError("data error bars belong to an absolute-value plot")
    ordinate = quantity_label
    scaled = np.ones(panels)
    if reference is not None:
        reference_grid, reference_bank = _frames(curves=[reference], name="reference")
        if (not np.array_equal(reference_grid, coordinate)
                or reference_bank.shape[1:] != values.shape[1:]):
            raise ValueError("reference must use the exact curve grid and shape")
        values, notation, units = _residual(
            values=values, reference=reference_bank[0], mode=residual,
            sigma=sigma, reference_scale=reference_scale,
        )
        ordinate = notation if units != "original" else notation + " " + quantity_label
    elif rescale is not None:
        if rescale != 1:
            raise ValueError("rescale is None or 1 for an annotated power-of-ten scale")
        peaks = np.max(np.abs(values), axis=(0, 2))
        nonzero = peaks > 0
        scaled[nonzero] = 10.0 ** (-np.floor(np.log10(peaks[nonzero])))
        values = values * scaled[None, :, None]
        if observed is not None:
            observed = observed * scaled[:, None]
            uncertainty = uncertainty * scaled[:, None]
        ordinate = r"$\alpha$ " + quantity_label
    bounds = None
    if ylim is not None:
        bounds = np.asarray(ylim, dtype=float)
        if bounds.shape == (2,):
            bounds = np.tile(bounds, (nrows, 1))
        if (bounds.shape != (nrows, 2) or not np.all(np.isfinite(bounds))
                or np.any(bounds[:, 1] <= bounds[:, 0])):
            raise ValueError("ylim is one increasing pair or one pair per panel row")
    if figsize is None:
        figsize = (4.2 * ncols, 3.2 * nrows + 0.6)
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                                squeeze=False, sharex=True, constrained_layout=True)
    active = []
    for panel, axis in enumerate(axes.flat):
        if panel >= panels:
            axis.set_visible(False)
            continue
        active.append(axis)
        axis.set_xscale(xscale)
        if bintextpos is None:
            axis.set_title(label=labels[panel], fontsize=bintextsize, pad=5)
        else:
            axis.text(x=bintextpos[0], y=bintextpos[1], s=labels[panel],
                      transform=axis.transAxes, fontsize=bintextsize)
        # A small aperture bank is clearer at its actual radii than with
        # several crowded logarithmic minor labels such as 2x10^1,3x10^1.
        if xscale == "log" and coordinate.size <= 6:
            axis.xaxis.set_major_locator(FixedLocator(coordinate))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            axis.xaxis.set_minor_formatter(NullFormatter())
        axis.tick_params(axis="x", labelsize=xaxisticklabelsize)
        axis.tick_params(axis="y", labelsize=yaxisticklabelsize)
        if excluded[panel]:
            axis.text(0.5, 0.45, "excluded", ha="center", transform=axis.transAxes)
            continue
        for curve in range(count):
            axis.plot(coordinate, values[curve, panel], markersize=markersize,
                      **settings[curve])
        axis.axhline(y=0.0, color="0.5", linewidth=0.6)
        if observed is not None:
            axis.errorbar(coordinate, observed[panel], yerr=uncertainty[panel],
                          fmt="o", color="black", markersize=markersize, capsize=2,
                          label="Data", zorder=5)
        if reference is None and rescale == 1 and scaled[panel] != 1:
            exponent = int(np.rint(np.log10(scaled[panel])))
            factor = rf"$\alpha=10^{{{exponent}}}$"
            if bintextpos is None:
                axis.set_title(label=labels[panel] + "  " + factor,
                               fontsize=bintextsize, pad=5)
            else:
                axis.text(0.06, 0.06, factor, transform=axis.transAxes,
                          fontsize=max(10, bintextsize-3))
        if not np.any(np.isfinite(values[:, panel])):
            axis.text(0.5, 0.45, "undefined reference ratio", ha="center",
                      transform=axis.transAxes, fontsize=10)
    # Some Matplotlib versions merge logarithmic error-bar collection
    # coordinates into the automatic data limits. All curves and data use
    # this one declared coordinate grid, so set its bounds explicitly.
    if coordinate.size > 1:
        limits = np.array([coordinate[0], coordinate[-1]])
        if xscale == "log":
            limits = np.log(limits)
        margin = 0.03 * (limits[1] - limits[0])
        limits = limits + np.array([-margin, margin])
        if xscale == "log":
            limits = np.exp(limits)
        axes[0, 0].set_xlim(limits)
    # Residuals and rescaled panels share one band per row. Absolute
    # unscaled panels retain their own physical amplitude range.
    shared = reference is not None or rescale == 1
    for row in range(nrows):
        selection = slice(row * ncols, min((row + 1) * ncols, panels))
        displayed_row = values[:, selection].ravel()
        if observed is not None:
            displayed_row = np.concatenate((displayed_row,
                (observed[selection] - uncertainty[selection]).ravel(),
                (observed[selection] + uncertainty[selection]).ravel()))
        row_range = _range(values=displayed_row)
        if bounds is not None:
            row_range = bounds[row]
        for column in range(ncols):
            panel = row * ncols + column
            if panel >= panels:
                continue
            limits = row_range
            if not shared and bounds is None:
                displayed = values[:, panel].ravel()
                if observed is not None:
                    displayed = np.concatenate((displayed,
                        observed[panel] - uncertainty[panel],
                        observed[panel] + uncertainty[panel]))
                limits = _range(values=displayed)
            axes[row, column].set_ylim(limits)
    if legend is not None or data is not None:
        for axis in active:
            handles, names = axis.get_legend_handles_labels()
            if handles:
                location = "outside upper center" if legendloc is None else legendloc
                figure.legend(handles=handles, labels=names, loc=location,
                              ncol=len(handles), frameon=False, fontsize=legendfontsize)
                break
    if mapping is not None and colorbar not in (None, 0, False):
        bar = figure.colorbar(mapping, ax=active, shrink=colorbarshrink, pad=0.02)
        if colorbarlabel is not None:
            bar.set_label(colorbarlabel, fontsize=15)
    figure.supxlabel(coordinate_label, fontsize=xaxislabelsize)
    figure.supylabel(ordinate, fontsize=yaxislabelsize)
    if shared and coordinate.size > 1:
        figure.get_layout_engine().set(wspace=0.02, hspace=0.02)
        figure.canvas.draw()
        for row in range(nrows):
            visible = []
            for column in range(ncols):
                panel = row * ncols + column
                if panel < panels:
                    if column > 0:
                        axes[row, column].sharey(axes[row, 0])
                        axes[row, column].tick_params(axis="y", labelleft=False)
                    visible.append((axes[row, column], column == 0,
                                    panel == min((row+1)*ncols, panels)-1))
            _hide_glued_edge_ticklabels(panels=visible, lo=coordinate[0],
                                       hi=coordinate[-1], axis="x", log=xscale == "log")
    if show is not None:
        plt.show()
        return None
    return figure, axes


def plot_Map3(
    map3, map3_ref=None, param=None, colorbarlabel=None, *, triplets,
    sigma=None, data=None, residual="percent", reference_scale=None,
    legend=None, marker=None, linestyle=None, linewidth=None, ylim=None,
    rescale=None, excluded=None, ncols=5, figsize=None, show=1, cmap="viridis",
    legendloc=None, legendfontsize=13, xaxislabelsize=17, yaxislabelsize=17,
    xaxisticklabelsize=13, yaxisticklabelsize=13,
    bintextpos=(0.06, 0.86), bintextsize=16, markersize=4,
    colorbar=1, colorbarshrink=0.75,
):
    """Plot equal-radius aperture skewness, one panel per ordered source triplet.

    Arguments:
      map3 = list of (R, values), with R[nR] in arcmin and values[ntriplet,nR].
          R is the common radius of all three apertures, not a triangle-side cut.
      map3_ref = optional same-form reference on exactly the same R grid.
      param, colorbarlabel = one sweep value per curve and its colorbar label.
      triplets = integer[ntriplet,3] zero-based labels; order is preserved.
      sigma = matching standard deviations, for residual="sigma" or data bars.
      data = optional (R, observed values), requiring sigma, in absolute mode.
      residual = percent, sigma, scaled or absolute; used only with a reference.
      reference_scale = scalar or[ntriplet] positive scale for scaled residuals;
          None uses max(abs(reference)) in each panel, refusing a zero panel.
      legend = one label per curve; marker/linestyle/linewidth = cycled sequences.
      ylim = optional (lower,upper), or one pair per panel row, in displayed units.
      rescale = 1 annotates a power-of-ten factor per absolute panel; None disables.
      excluded = optional Boolean[ntriplet]; zeros alone never mean excluded.
      ncols, figsize, show, cmap = panel columns, inches, display and sweep colors.
      legendloc, legendfontsize = shared legend location and font size.
      xaxislabelsize, yaxislabelsize = shared axis-label sizes in points.
      xaxisticklabelsize, yaxisticklabelsize = tick-label sizes in points.
      bintextpos, bintextsize = panel-label position in axes fractions and size;
          bintextpos=None places the labels above the axes instead.
      markersize, colorbar, colorbarshrink = marker points, colorbar toggle/size.
    Returns:
      0 with a named diagnostic for invalid input, None after display, or
      (figure, axes[nrows,ncols]) when show=None. Inputs are never modified.

    Percent is 100*(value/reference-1). A zero denominator is masked, never
    replaced by a floor. Absolute or sigma residuals retain those points.
    Diagonal error bars do not show cross correlations or a Fisher constraint.
    """
    try:
        _, values = _frames(curves=map3, name="map3")
        labels = _triplets(values=triplets, count=values.shape[1])
        return _line_panels(
            curves=map3, reference=map3_ref, labels=labels,
            coordinate_label=r"$R$ [arcmin]",
            quantity_label=r"$\langle M_{\rm ap}^3\rangle$",
            param=param, colorbarlabel=colorbarlabel, sigma=sigma, residual=residual,
            reference_scale=reference_scale, legend=legend, marker=marker,
            linestyle=linestyle, linewidth=linewidth, ylim=ylim, rescale=rescale,
            excluded=excluded, ncols=ncols, figsize=figsize, show=show,
            xscale="log", cmap=cmap, data=data,
            legendloc=legendloc, legendfontsize=legendfontsize,
            xaxislabelsize=xaxislabelsize, yaxislabelsize=yaxislabelsize,
            xaxisticklabelsize=xaxisticklabelsize, yaxisticklabelsize=yaxisticklabelsize,
            bintextpos=bintextpos, bintextsize=bintextsize, markersize=markersize,
            colorbar=colorbar, colorbarshrink=colorbarshrink,
        )
    except (ValueError, TypeError) as error:
        print(f"Bad Input (Map3): {error}")
        return 0


def plot_ia_components(
    components, components_ref=None, param=None, colorbarlabel=None, *,
    component_labels, coordinate_label, quantity_label,
    sigma=None, residual="percent", reference_scale=None,
    legend=None, marker=None, linestyle=None, linewidth=None, ylim=None,
    rescale=None, ncols=3, figsize=None, show=1, xscale="log", cmap="viridis",
    legendloc=None, legendfontsize=13, xaxislabelsize=17, yaxislabelsize=17,
    xaxisticklabelsize=13, yaxisticklabelsize=13,
    bintextpos=(0.06, 0.86), bintextsize=16, markersize=4,
    colorbar=1, colorbarshrink=0.75,
):
    """Show named signed IA contributions without adding or flipping any term.

    Arguments:
      components = list of (x, values), values[ncomponent,nx], in supplied units.
      components_ref = optional same-form reference on the exact same grid.
      param, colorbarlabel = sweep values and their colorbar label, or None.
      component_labels = exact operator/field labels, one per row; no inferred sum.
      coordinate_label, quantity_label = labels including the supplied units.
      sigma, residual, reference_scale = comparison definitions from plot_Map3.
      legend, marker, linestyle, linewidth = per-curve labels/style sequences.
      ylim, rescale = display bounds and optional annotated power-of-ten factors.
      ncols, figsize, show, xscale, cmap = panel/layout/color controls.
      legendloc, legendfontsize, xaxislabelsize, yaxislabelsize,
      xaxisticklabelsize, yaxisticklabelsize, bintextpos, bintextsize,
      markersize, colorbar, colorbarshrink = same display controls as plot_Map3.
    Returns:
      0 with a named diagnostic for invalid input, None after display, or
      (figure, axes) when show=None. Inputs are never modified.

    A component can cross zero or vanish exactly. Linear signed axes retain
    both cases. Operator coefficients and ordered field placements are
    different decompositions; summing overlapping contributions here would
    double count physics, so no Total curve is manufactured by the plotter.
    """
    try:
        return _line_panels(
            curves=components, reference=components_ref, labels=component_labels,
            coordinate_label=coordinate_label, quantity_label=quantity_label,
            param=param, colorbarlabel=colorbarlabel, sigma=sigma, residual=residual,
            reference_scale=reference_scale, legend=legend, marker=marker,
            linestyle=linestyle, linewidth=linewidth, ylim=ylim, rescale=rescale,
            excluded=None, ncols=ncols, figsize=figsize, show=show,
            xscale=xscale, cmap=cmap,
            legendloc=legendloc, legendfontsize=legendfontsize,
            xaxislabelsize=xaxislabelsize, yaxislabelsize=yaxislabelsize,
            xaxisticklabelsize=xaxisticklabelsize, yaxisticklabelsize=yaxisticklabelsize,
            bintextpos=bintextpos, bintextsize=bintextsize, markersize=markersize,
            colorbar=colorbar, colorbarshrink=colorbarshrink,
        )
    except (ValueError, TypeError) as error:
        print(f"Bad Input (IA components): {error}")
        return 0


def plot_natural_components(
    gamma, gamma_ref=None, param=None, colorbarlabel=None, *,
    coordinate_label, slice_label, projection,
    sigma=None, residual="absolute", reference_scale=None,
    legend=None, marker=None, linestyle=None, linewidth=None, ylim=None,
    rescale=None, figsize=None, show=1, xscale="linear", cmap="viridis",
    legendloc=None, legendfontsize=13, xaxislabelsize=17, yaxislabelsize=17,
    xaxisticklabelsize=13, yaxisticklabelsize=13,
    bintextpos=(0.06, 0.86), bintextsize=16, markersize=4,
    colorbar=1, colorbarshrink=0.75,
):
    """Show real and imaginary parts of all four Gamma components on one slice.

    Arguments:
      gamma = list of (x, complex[4,nx]) arrays on one declared triangle slice.
      gamma_ref = optional matching reference; no phase conversion occurs here.
      param, colorbarlabel = parameter values and colorbar label, or None.
      coordinate_label = horizontal coordinate with units, e.g. phi[rad].
      slice_label = fixed side lengths/ratio and source triplet, shown in the legend.
      projection = explicit shear-rotation convention, shown beside slice_label.
      sigma = optional[8,nx], real rows0..3 then imaginary rows0..3.
      residual, reference_scale = comparison definitions from plot_Map3.
      legend, marker, linestyle, linewidth = per-curve labels and styles.
      ylim, rescale = displayed bounds and optional power-of-ten panel scaling.
      figsize, show, xscale, cmap = inches, display, coordinate scale and colors.
      legendloc, legendfontsize, xaxislabelsize, yaxislabelsize,
      xaxisticklabelsize, yaxisticklabelsize, bintextpos, bintextsize,
      markersize, colorbar, colorbarshrink = same display controls as plot_Map3.
    Returns:
      0 with a named diagnostic for invalid input, None after display, or
      (figure, axes[2,4]) when show=None. Inputs are never modified.

    Conjugation and component ordering belong to the caller's physical
    convention. Imaginary zeros are valid signals, not excluded panels.
    The default absolute residual remains defined at those zeros.
    """
    try:
        coordinate, values = _frames(curves=gamma, name="gamma", complex_values=True)
        if values.shape[1] != 4 or not slice_label or not projection:
            raise ValueError("four components, slice_label and projection are required")
        curves = []
        for curve in values:
            curves.append((coordinate, np.concatenate((curve.real, curve.imag), axis=0)))
        reference = None
        if gamma_ref is not None:
            ref_grid, ref_values = _frames(curves=[gamma_ref], name="gamma_ref",
                                           complex_values=True)
            reference = (ref_grid, np.concatenate(
                (ref_values[0].real, ref_values[0].imag), axis=0))
        labels = []
        for part in (r"\operatorname{Re}", r"\operatorname{Im}"):
            for component in range(4):
                labels.append(rf"${part}\Gamma_{component}$")
        result = _line_panels(
            curves=curves, reference=reference, labels=labels,
            coordinate_label=coordinate_label, quantity_label=r"$\Gamma$",
            param=param, colorbarlabel=colorbarlabel, sigma=sigma, residual=residual,
            reference_scale=reference_scale, legend=legend, marker=marker,
            linestyle=linestyle, linewidth=linewidth, ylim=ylim, rescale=rescale,
            excluded=None, ncols=4, figsize=figsize, show=None,
            xscale=xscale, cmap=cmap,
            legendloc=legendloc, legendfontsize=legendfontsize,
            xaxislabelsize=xaxislabelsize, yaxislabelsize=yaxislabelsize,
            xaxisticklabelsize=xaxisticklabelsize, yaxisticklabelsize=yaxisticklabelsize,
            bintextpos=bintextpos, bintextsize=bintextsize, markersize=markersize,
            colorbar=colorbar, colorbarshrink=colorbarshrink,
        )
        figure, axes = result
        # This label carries scientific information absent from the arrays;
        # it is not an automatic title such as "natural components".
        figure.supxlabel(coordinate_label + "\n" + slice_label + "; " + projection,
                         fontsize=xaxislabelsize)
        if show is not None:
            plt.show()
            return None
        return figure, axes
    except (ValueError, TypeError) as error:
        print(f"Bad Input (natural components): {error}")
        return 0


def plot_Gamma(
    gamma, gamma_ref=None, param=None, colorbarlabel=None, *, triplets,
    opening_angle_deg, projection, component=0, part="real",
    residual="percent", reference_scale=None, sigma=None,
    limits=None, ncols=4, figsize=None, show=1, cmap="RdBu_r",
    xaxislabelsize=17, yaxislabelsize=17,
    xaxisticklabelsize=13, yaxisticklabelsize=13,
    bintextpos=(0.06, 0.88), bintextsize=16,
    colorbar=1, colorbarshrink=0.85,
):
    """Draw one Gamma component over its two triangle sides at a fixed angle.

    Arguments:
      gamma = (theta1,theta2,values), positive arcmin axes and complex values
          [ntriplet,4,ntheta1,ntheta2]. Neither side nor source order is changed.
      gamma_ref = optional matching tuple; reference comparison uses residual.
      param = accepted for shared plotting signatures; must be None (map colors
          encode signal, so a parameter sweep needs separate calls).
      colorbarlabel = optional explicit colorbar label including normalization.
      triplets = zero-based integer[ntriplet,3] source labels.
      opening_angle_deg, projection = declared angle and shear convention.
      component, part = natural index 0..3 and real/imaginary part ("real"/"imag").
      residual = percent, sigma, scaled or absolute as in plot_Map3.
      reference_scale = optional positive scalar or[ntriplet] for scaled residual.
      sigma = optional positive[ntriplet,ntheta1,ntheta2] for sigma residual.
      limits = optional increasing color bounds; must include every finite value.
      ncols, figsize, show, cmap = panel columns, inches, display and signed map.
      xaxislabelsize, yaxislabelsize, xaxisticklabelsize, yaxisticklabelsize,
      bintextpos, bintextsize, colorbar, colorbarshrink = familiar label/color
          controls, with the same meanings as plot_Map3.
    Returns:
      0 with a named diagnostic for invalid input, None after display, or
      (figure, axes) when show=None. Inputs are never modified.

    This is the two-side diagnostic used for ReGamma0 in the TATT paper's
    Figures 1–2. It draws supplied calculations; it does not reproduce that
    paper's physical inputs. Gray cells mark an undefined percent ratio,
    and the colorbar states their count. No percentile clipping is applied.
    """
    try:
        if param is not None or part not in ("real", "imag"):
            raise ValueError("param must be None and part must be real or imag")
        if isinstance(component, bool) or component not in (0, 1, 2, 3):
            raise ValueError("component must be an integer0..3")
        if (not np.isfinite(opening_angle_deg) or not 0 < opening_angle_deg < 180
                or not projection):
            raise ValueError("declare 0<opening_angle_deg<180 and projection")
        first, second, values = _gamma_map(gamma=gamma)
        labels = _triplets(values=triplets, count=values.shape[0])
        selected = values[:, int(component)].real
        symbol = rf"$\operatorname{{Re}}\Gamma_{component}$"
        if part == "imag":
            selected = values[:, int(component)].imag
            symbol = rf"$\operatorname{{Im}}\Gamma_{component}$"
        if gamma_ref is not None:
            ref_first, ref_second, ref_values = _gamma_map(gamma=gamma_ref)
            if (not np.array_equal(first, ref_first)
                    or not np.array_equal(second, ref_second)
                    or values.shape != ref_values.shape):
                raise ValueError("reference needs identical side grids and array shape")
            baseline = ref_values[:, int(component)].real
            if part == "imag":
                baseline = ref_values[:, int(component)].imag
            errors = None
            if sigma is not None:
                errors = np.asarray(sigma)
                if errors.shape != selected.shape:
                    raise ValueError("sigma needs exactly [ntriplet,ntheta1,ntheta2]")
                errors = errors.reshape(selected.shape[0], -1)
            transformed, notation, units = _residual(
                values=selected.reshape(1, selected.shape[0], -1),
                reference=baseline.reshape(baseline.shape[0], -1), mode=residual,
                sigma=errors, reference_scale=reference_scale,
            )
            selected = transformed[0].reshape(selected.shape)
            symbol = notation if units != "original" else notation + " " + symbol
        if isinstance(ncols, bool) or int(ncols) != ncols or ncols < 1:
            raise ValueError("ncols must be a positive integer")
        ncols = min(int(ncols), len(labels))
        nrows = (len(labels) + ncols - 1) // ncols
        finite = selected[np.isfinite(selected)]
        if limits is None:
            maximum = 1.0 if finite.size == 0 else float(np.max(np.abs(finite)))
            if maximum == 0:
                maximum = 1.0
            limits = (-maximum, maximum)
        else:
            limits = np.asarray(limits, dtype=float)
            if (limits.shape != (2,) or not np.all(np.isfinite(limits))
                    or limits[1] <= limits[0]):
                raise ValueError("limits must be a finite increasing pair")
            if finite.size and (np.min(finite) < limits[0] or np.max(finite) > limits[1]):
                raise ValueError("limits would clip values; include the complete range")
        if figsize is None:
            figsize = (4 * ncols + 0.7, 3.6 * nrows)
        palette = plt.get_cmap(cmap).copy()
        palette.set_bad("0.55")
        figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                                    squeeze=False, sharex=True, sharey=True,
                                    constrained_layout=True)
        active = []
        for panel, axis in enumerate(axes.flat):
            if panel >= len(labels):
                axis.set_visible(False)
                continue
            active.append(axis)
            # pcolormesh expects [y,x]; this transpose preserves the public
            # [theta1,theta2] axes and does not exchange their source labels.
            image = axis.pcolormesh(first, second, selected[panel].T,
                                   shading="nearest", cmap=palette,
                                   vmin=limits[0], vmax=limits[1], rasterized=True)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.text(*bintextpos, labels[panel], transform=axis.transAxes,
                      fontsize=bintextsize, bbox={"facecolor": "white", "alpha": 0.8,
                                         "edgecolor": "none"})
            axis.tick_params(axis="x", labelsize=xaxisticklabelsize)
            axis.tick_params(axis="y", labelsize=yaxisticklabelsize)
        label = symbol if colorbarlabel is None else colorbarlabel
        undefined = int(np.count_nonzero(~np.isfinite(selected)))
        if undefined:
            label += f"; gray: {undefined} undefined cells"
        if colorbar not in (None, 0, False):
            bar = figure.colorbar(image, ax=active, shrink=colorbarshrink, pad=0.02)
            bar.set_label(label, fontsize=15)
        figure.supxlabel(r"$\theta_1$ [arcmin]" + "\n"
                         + rf"$\phi={opening_angle_deg:g}^\circ$; " + projection,
                         fontsize=xaxislabelsize)
        figure.supylabel(r"$\theta_2$ [arcmin]", fontsize=yaxislabelsize)
        if show is not None:
            plt.show()
            return None
        return figure, axes
    except (ValueError, TypeError) as error:
        print(f"Bad Input (Gamma map): {error}")
        return 0


def _gamma_map(gamma):
    """Validate a rectangular two-side Gamma grid without changing its order."""
    if not isinstance(gamma, (list, tuple)) or len(gamma) != 3:
        raise ValueError("gamma needs (theta1,theta2,values)")
    first = np.asarray(gamma[0], dtype=float)
    second = np.asarray(gamma[1], dtype=float)
    for axis in (first, second):
        if (axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis))
                or np.any(axis <= 0) or np.any(np.diff(axis) <= 0)):
            raise ValueError(
                "both side grids need at least two positive increasing nodes"
            )
    values = np.asarray(gamma[2])
    if (values.ndim != 4 or values.shape[0] == 0
            or values.shape[1:] != (4, first.size, second.size)
            or not np.all(np.isfinite(values))):
        raise ValueError("Gamma must be finite [ntriplet,4,ntheta1,ntheta2]")
    return first, second, values


def plot_code_comparison(
    x, reference, ported, labels=None, *, coordinate_label,
    quantity_label, residual="scaled", reference_scale=None, sigma=None,
    legend=("C port",), ncols=3, figsize=None, show=1, xscale="linear",
    **plot_options,
):
    """Show a source-versus-port residual without computing either prediction.

    Arguments:
      x = common increasing[nx] coordinates with declared units.
      reference, ported = real[nseries,nx] arrays; complete arrays belong in
          the comparison record even if these plotted rows are only slices.
      labels = one physical component label per row, or numbered series.
      coordinate_label, quantity_label = axes, including supplied units.
      residual = scaled(default), absolute, sigma or pointwise percent.
      reference_scale = positive scalar/[nseries] for scaled mode; None uses
          each reference row's peak absolute value and refuses all-zero rows.
      sigma = positive[nseries,nx] errors for sigma mode.
      legend, ncols, figsize, show, xscale = label and display controls.
      plot_options = additional named display controls accepted by plot_ia_components.
    Returns:
      0 with a named diagnostic for invalid input, None after display, or
      (figure, axes) when show=None. No array is changed or saved.

    Scaled residuals are percent of a fixed reference amplitude, not a
    pointwise percent difference or statistical significance. For exactly
    zero reference rows choose absolute mode or supply a physical scale.
    """
    if labels is None:
        shape = np.shape(reference)
        if len(shape) != 2:
            print("Bad Input (code comparison): reference must have two axes")
            return 0
        labels = [f"Series {index+1}" for index in range(shape[0])]
    return plot_ia_components(
        components=[(x, ported)], components_ref=(x, reference),
        component_labels=labels, coordinate_label=coordinate_label,
        quantity_label=quantity_label, residual=residual,
        reference_scale=reference_scale, sigma=sigma, legend=legend,
        ncols=ncols, figsize=figsize, show=show, xscale=xscale,
        **plot_options,
    )


def plot_parameter_errors(
    names, errors, errors_ref=None, legend=None, labels=None, *,
    figsize=None, show=1,
):
    """Compare supplied marginalized parameter errors or their reference ratios.

    Arguments:
      names = unique ordered parameter identifiers; the plot never sorts them.
      errors = list of finite positive[nparameter] marginalized sigma arrays.
      errors_ref = optional finite positive[nparameter] reference errors;
          ratios below1 mean smaller marginalized errors for that parameter.
      legend = one calculation label per error array, or numbered curves.
      labels = optional mathematical display labels matching names.
      figsize, show = inches and display/return control.
    Returns:
      0 with a named diagnostic for invalid input, None after display, or
      (figure, axis) when show=None. No Fisher inversion is performed.

    Without a reference, one axis mixes parameter units and is useful only
    when the supplied errors have an agreed common unit or normalization.
    Ratios compare the same parameter and prior definition. Conditional
    errors 1/sqrt(F_aa) must not be labeled marginalized errors here.
    Lines join categorical positions only to identify a calculation; they
    do not interpolate between different physical parameters. Colors, marker
    shapes and widths distinguish calculations. An outside legend is checked
    against the whole axes rectangle after layout, which also excludes all
    clipped data and reference lines from its bounding rectangle.
    """
    try:
        if (not isinstance(names, (list, tuple)) or not names
                or len(set(names)) != len(names)):
            raise ValueError("names must be a nonempty sequence of unique identifiers")
        values = np.asarray(errors, dtype=float)
        if (values.ndim != 2 or values.shape[0] == 0
                or values.shape[1] != len(names) or np.any(values <= 0)
                or not np.all(np.isfinite(values))):
            raise ValueError("errors must be positive finite [calculation,parameter]")
        labels = names if labels is None else labels
        if len(labels) != len(names):
            raise ValueError("labels must match the parameter count")
        ylabel = r"Marginalized $\sigma(p)$ [declared units]"
        if errors_ref is not None:
            baseline = np.asarray(errors_ref, dtype=float)
            if (baseline.shape != (len(names),) or np.any(baseline <= 0)
                    or not np.all(np.isfinite(baseline))):
                raise ValueError("errors_ref must be positive finite [parameter]")
            values = values / baseline[None, :]
            ylabel = r"$\sigma(p)/\sigma_{\rm ref}(p)$"
        if legend is None:
            legend = [f"Calculation {index+1}" for index in range(len(values))]
        if len(legend) != len(values):
            raise ValueError("legend needs one label per calculation")
        if figsize is None:
            figsize = (max(7.0, 0.7 * len(names)), 4.8)
        figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
        locations = np.arange(len(names))
        width = 0.7 / len(values)
        colors = (
            "#0072B2",
            "#E69F00",
            "#CC79A7",
            "#56B4E9",
            "#D55E00",
            "#000000",
        )
        markers = (
            "o",
            "s",
            "D",
            "^",
            "v",
            "P",
        )
        widths = (
            2.4,
            1.8,
            1.3,
            0.9,
        )
        for index, row in enumerate(values):
            shifted = locations + (index - (len(values)-1)/2) * width
            axis.plot(
                shifted, row, markersize=6, marker=markers[index % 6],
                linewidth=widths[index % 4], color=colors[index % 6],
                label=legend[index],
            )
        if errors_ref is not None:
            axis.axhline(y=1.0, color="0.4", linestyle="dashed", linewidth=1)
        axis.set_xticks(locations, labels=labels, rotation=35, ha="right")
        axis.set_ylabel(ylabel, fontsize=17)
        axis.tick_params(axis="both", labelsize=14)
        displayed = values.ravel()
        if errors_ref is not None:
            displayed = np.concatenate((displayed, np.array([1.0])))
        axis.set_ylim(_range(values=displayed))
        key = figure.legend(
            loc="outside upper center", ncol=min(len(values), 2),
            frameon=False, fontsize=13,
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        key_box = key.get_window_extent(renderer=renderer).padded(2.0)
        axes_box = axis.get_tightbbox(renderer=renderer)
        if key_box.overlaps(axes_box):
            plt.close(fig=figure)
            raise ValueError("Outside legend overlaps the parameter axes.")
        if (key_box.x0 < figure.bbox.x0 or key_box.x1 > figure.bbox.x1
                or key_box.y0 < figure.bbox.y0
                or key_box.y1 > figure.bbox.y1):
            plt.close(fig=figure)
            raise ValueError("Increase figsize to contain the outside legend.")
        if show is not None:
            plt.show()
            return None
        return figure, axis
    except (ValueError, TypeError) as error:
        print(f"Bad Input (parameter errors): {error}")
        return 0
