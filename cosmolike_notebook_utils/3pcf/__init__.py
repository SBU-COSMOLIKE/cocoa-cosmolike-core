"""Array-only figures for three-point shear and aperture statistics.

Python identifiers cannot begin with a digit. Import this requested
directory with importlib.import_module("cosmolike_notebook_utils.3pcf").
The functions calculate no cosmology and import no compiled interface.
See README.md for array axes, estimator labels and paper comparisons.
"""

from .plots import (
    plot_Map3,
    plot_Gamma,
    plot_natural_components,
    plot_ia_components,
    plot_code_comparison,
    plot_parameter_errors,
)
from ..plot_covariances import (
    plot_correlation,
    plot_covariance_components,
    plot_covariance_diagonal,
)
from .ia_components import plot_ia_operator_placement
from .figure_checks import check_figure_labels

__all__ = [
    "plot_Map3",
    "plot_Gamma",
    "plot_natural_components",
    "plot_ia_components",
    "plot_ia_operator_placement",
    "plot_code_comparison",
    "plot_parameter_errors",
    "plot_correlation",
    "plot_covariance_components",
    "plot_covariance_diagonal",
    "check_figure_labels",
]
