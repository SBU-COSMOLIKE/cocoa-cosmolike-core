"""Compare local parameter responses in the measurement's error units.

Diagonal errors keep each plotted point attached to its observable and angle.
Fits use the full covariance, including correlations between observables.
These array operations calculate no physical prediction and add no priors.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker

from .fisher_accuracy import _finish, _range


def compare_templates(target, mimic, covariance, two_point):
    """Fit one supplied parameter response with another, for xi and jointly.

    Arguments:
      target, mimic = finite [nrow] responses Delta_p * dmu/dp, in the
          observable units of each row. The caller declares Delta_p;
          neither a finite-difference step nor a prior width is inferred.
      covariance = symmetric positive definite [nrow,nrow] covariance.
      two_point = boolean [nrow], True for xi and False for Map3.
    Returns:
      Dictionary containing templates[2,nrow] divided by sqrt(C_ii), and
      two_point/joint dictionaries with the fitted coefficient, signed
      template cosine, reproduced information fraction and residual[nrow]
      in the same plotted units. The xi fit is extended to all Map3 rows
      without refitting. Residual norms use the full covariance; their
      two-point plus conditional-three-point decomposition is retained.
    Raises:
      ValueError for inconsistent inputs or a zero response in either fitted
      selection; LinAlgError for nonpositive covariance. Inputs remain unchanged.
    Interpretation:
      Other parameters are fixed. The cosine is a derivative overlap,
      not a correlation after marginalizing other parameters and priors.
    """
    responses = np.asarray([target, mimic], dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    second = np.asarray(two_point)
    if (responses.ndim != 2 or responses.shape[1] < 2
            or not np.all(np.isfinite(responses))):
        raise ValueError("Supply two finite response vectors with at least two rows.")
    nrow = responses.shape[1]
    if (matrix.shape != (nrow, nrow) or not np.all(np.isfinite(matrix))
            or np.any(np.diag(matrix) <= 0.0)):
        raise ValueError("Supply a finite covariance with positive row variances.")
    if (second.shape != (nrow,) or second.dtype.kind != "b"
            or not np.any(second) or np.all(second)):
        raise ValueError("Supply a boolean row selection containing xi and Map3.")

    # Removing observable units before factorization avoids comparing
    # tiny Map3 variances with much larger xi variances in a symmetry test.
    error = np.sqrt(np.diag(matrix))
    correlation = matrix / error[:, None] / error[None, :]
    if not np.allclose(correlation, correlation.T, rtol=1.e-12, atol=1.e-14):
        raise ValueError("Covariance is not symmetric in row-error units.")
    templates = responses / error[None, :]
    factor = np.linalg.cholesky(correlation)
    block22 = correlation[np.ix_(second, second)]
    block23 = correlation[np.ix_(second, ~second)]
    factor22 = np.linalg.cholesky(block22)
    whitened23 = np.linalg.solve(factor22, block23)
    conditional = correlation[np.ix_(~second, ~second)]
    conditional = conditional - whitened23.T @ whitened23
    factor_conditional = np.linalg.cholesky(conditional)
    result = {"templates": templates.copy()}
    for name, selected, selected_factor in (
        ("two_point", second, factor22),
        ("joint", np.ones(nrow, dtype=bool), factor),
    ):
        whitened = np.linalg.solve(selected_factor, templates[:, selected].T)
        target_norm2 = float(whitened[:, 0] @ whitened[:, 0])
        mimic_norm2 = float(whitened[:, 1] @ whitened[:, 1])
        if target_norm2 <= 0.0 or mimic_norm2 <= 0.0:
            raise ValueError(f"The {name} selection has a zero parameter response.")
        overlap = float(whitened[:, 0] @ whitened[:, 1])
        coefficient = overlap / mimic_norm2
        cosine = overlap / np.sqrt(target_norm2) / np.sqrt(mimic_norm2)
        residual = templates[0] - coefficient * templates[1]
        residual2 = np.linalg.solve(factor22, residual[second])
        # Subtract the Map3 fluctuation predicted by the correlated xi
        # residual before weighting with its conditional covariance.
        residual3 = residual[~second] - whitened23.T @ residual2
        residual3 = np.linalg.solve(factor_conditional, residual3)
        residual_joint = np.linalg.solve(factor, residual)
        norm2 = float(residual_joint @ residual_joint)
        norm2_second = float(residual2 @ residual2)
        norm2_third = float(residual3 @ residual3)
        result[name] = {
            "coefficient": coefficient,
            "cosine": cosine,
            "reproduced_fraction": cosine**2,
            "target_norm2": target_norm2,
            "residual": residual,
            "residual_joint_norm2": norm2,
            "residual_two_point_norm2": norm2_second,
            "residual_conditional_norm2": norm2_third,
        }
    return result


def plot_derivative_templates(
    panels, parameter_labels, *, figsize=(12, 6), show=1,
    legendfontsize=11,
):
    """Draw scaled derivatives above the residuals of xi and joint fits.

    Arguments:
      panels = sequence of dictionaries with coordinate[n], templates[2,n],
          residuals[2,n], title and coordinate_label. Rows of templates
          are target/mimic; rows of residuals are xi-fit/joint-fit. All
          vertical quantities already contain their declared row errors.
      parameter_labels = two display labels including parameter scales.
      figsize = width/height in inches; legendfontsize = outside legend size.
      show = None returns (fig, axes); otherwise show the figure.
    No curve is renormalized to its peak. Each row shares a linear y range.
    Titles and legends stay outside data rectangles. No files are written,
    no rcParams are changed, and no cosmology is calculated.
    Returns: 0 for bad input, None after showing, or (fig, axes).
    """
    try:
        if not panels or len(parameter_labels) != 2:
            raise ValueError("supply panels and two parameter labels")
        checked = []
        for panel in panels:
            coordinate = np.asarray(panel["coordinate"], dtype=float)
            templates = np.asarray(panel["templates"], dtype=float)
            residuals = np.asarray(panel["residuals"], dtype=float)
            if (coordinate.ndim != 1 or coordinate.size < 2
                    or not np.all(np.isfinite(coordinate))
                    or np.any(coordinate <= 0.0)
                    or np.any(np.diff(coordinate) <= 0.0)
                    or templates.shape != (2, coordinate.size)
                    or residuals.shape != templates.shape
                    or not np.all(np.isfinite(templates))
                    or not np.all(np.isfinite(residuals))):
                raise ValueError("use increasing positive angles and finite [2,n] curves")
            checked.append((coordinate, templates, residuals))
            str(panel["title"])
            str(panel["coordinate_label"])
    except (KeyError, TypeError, ValueError) as error:
        print(f"Bad Input (derivative templates): {error}")
        return 0

    fig, axes = plt.subplots(
        nrows=2, ncols=len(panels), figsize=figsize, sharey="row", squeeze=False,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.12, top=0.87,
                        hspace=0.12, wspace=0.08)
    handles = []
    for column, (panel, values) in enumerate(zip(panels, checked)):
        coordinate, templates, residuals = values
        for row, curves in enumerate((templates, residuals)):
            axis = axes[row, column]
            for index, (color, width, marker, style) in enumerate((
                ("#0072B2", 2.4, "o", "solid"),
                ("#E69F00", 1.8, "s", "dashed"),
            )):
                line, = axis.plot(
                    coordinate, curves[index], color=color, linewidth=width,
                    marker=marker, linestyle=style, markersize=4,
                )
                if column == 0:
                    handles.append(line)
            axis.axhline(y=0.0, color="0.6", linewidth=0.7, zorder=0)
            axis.set_xscale("log")
            if coordinate.size <= 5:
                # Sparse aperture radii read directly as 7,14,25,40 arcmin;
                # logarithmic minor labels would crowd the small panels.
                axis.set_xticks(coordinate, labels=[f"{value:g}" for value in coordinate])
                axis.xaxis.set_minor_locator(ticker.NullLocator())
            axis.tick_params(axis="both", labelsize=11)
            axis.grid(False)
        axes[0, column].set_title(panel["title"], fontsize=12)
        axes[0, column].tick_params(which="both", labelbottom=False)
        axes[1, column].set_xlabel(panel["coordinate_label"], fontsize=13)
    for row in (0, 1):
        values = np.concatenate([item[row + 1].ravel() for item in checked])
        limits = _range(values=values)
        if limits is not None:
            axes[row, 0].set_ylim(*limits)
    axes[0, 0].set_ylabel(r"$\Delta_p\,(\partial\mu_i/\partial p)/\sigma_i$",
                          fontsize=14)
    axes[1, 0].set_ylabel(r"$r_i/\sigma_i$", fontsize=14)
    labels = list(parameter_labels) + ["Residual: fit xi", "Residual: fit jointly"]
    return _finish(
        fig=fig, axes=axes, handles=handles, labels=labels,
        legendfontsize=legendfontsize, show=show,
    )
