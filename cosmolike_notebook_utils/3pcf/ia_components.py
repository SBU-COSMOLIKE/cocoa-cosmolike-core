"""Plot signed aperture IA contributions without evaluating a physical model.

The first panel groups terms by the physical alignment operator. The second
groups higher-order TATT by the number of intrinsic galaxies, excluding the
velocity-shear extension. Dividing by the same gravity-only moment makes the
curves dimensionless; it does not remove any covariance or parameter.
"""

import numpy as np
from matplotlib import pyplot as plt

def plot_ia_operator_placement(radius, values, title="", show=None):
    """Compare operator and galaxy-placement contributions at one source triple.

    Arguments:
      radius = increasing equal-aperture radii [nrow], in arcminutes.
      values = signed moments [5,8,nrow]; operator order is gravity, linear
        alignment, density weighting, tidal torque, velocity shear. Placement
        order is GGG, GGI, GIG, IGG, GII, IIG, IGI, III.
      title = optional source-triple/assumption label above both panels.
      show = plotting convention: None returns figure/axes without displaying.
    Returns:
      (figure, axes[2]) or prints one input error and returns 0.
    Side effects:
      Creates a Matplotlib figure; no rcParams or physical state is changed.
    """
    angles = np.asarray(radius, dtype=float)
    components = np.asarray(values, dtype=float)
    if (angles.ndim != 1 or angles.size < 2 or np.any(angles <= 0.0)
            or np.any(np.diff(angles) <= 0.0) or not np.all(np.isfinite(angles))
            or components.shape != (5, 8, angles.size)
            or not np.all(np.isfinite(components))
            or np.any(components[0, 0] == 0.0)):
        print("Supply increasing radii and finite [5,8,nrow] terms with nonzero GGG.")
        return 0
    gravity = components[0, 0]
    operator_curves = np.sum(components, axis=1) / gravity[None, :]
    residual = np.sum(components[2:4], axis=0) / gravity[None, :]
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(10.0, 5.2))
    colors = ("#0072B2", "#E69F00", "#CC79A7", "#56B4E9")
    styles = ("-", "--", "-.", ":")
    markers = ("o", "s", "^", "D")
    widths = (2.4, 1.8, 1.3, 0.9)
    labels = ("Linear alignment", "Density weighting", "Tidal torque", "Velocity shear")
    for index in range(4):
        axes[0].plot(angles, operator_curves[index + 1], color=colors[index],
                     linestyle=styles[index], marker=markers[index],
                     linewidth=widths[index], label=labels[index])
    axes[0].plot(angles, np.sum(operator_curves[1:], axis=0), color="black",
                 linewidth=2.2, label="Total IA")
    groups = ((1, 2, 3), (4, 5, 6), (7,))
    labels = ("GGI + GIG + IGG", "GII + IIG + IGI", "III")
    for index, group in enumerate(groups):
        curve = np.sum(residual[list(group)], axis=0)
        axes[1].plot(angles, curve, color=colors[index], linestyle=styles[index],
                     marker=markers[index], linewidth=widths[index], label=labels[index])
    for axis in axes:
        axis.set_xscale("log")
        axis.axhline(y=0.0, color="gray", linewidth=0.8)
        axis.set_xlabel(r"Aperture radius $\theta$ [arcmin]")
        axis.set_ylabel(r"Signed contribution / $\langle M_{\rm ap}^3\rangle_{GGG}$")
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                    frameon=False, fontsize=9)
    axes[0].set_title("Physical operators, including linear IA")
    axes[1].set_title("Higher-order TATT, excluding velocity shear")
    figure.suptitle(title)
    figure.subplots_adjust(bottom=0.32, top=0.85, wspace=0.32)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis in axes:
        bounds = axis.get_legend().get_window_extent(renderer=renderer)
        # Disjoint rectangles guarantee clearance for every curve, even
        # between plotted samples. The legends have their own lower band.
        for panel in axes:
            if bounds.overlaps(panel.get_window_extent(renderer=renderer)):
                plt.close(figure)
                raise ValueError("An IA legend overlaps the data; enlarge the layout.")
    if show is not None:
        plt.show()
        return None
    return figure, axes
