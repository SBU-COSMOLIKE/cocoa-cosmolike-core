"""Check that a displayed comparison preserves its physical array meaning.

Small algebraic arrays test plotting operations only. They are not survey
predictions. The separate render_examples.py uses pinned public Y3 arrays,
the saved original IA reference and an analytic shear reference for visual QA.
"""

import os
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["OMP_THREAD_LIMIT"] = "4"

import importlib
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pytest


CORE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CORE))
plots = importlib.import_module("cosmolike_notebook_utils.3pcf")


def test_map3_percent_zero_and_ownership(capsys):
    """An exact reference zero stays undefined; negative values retain signs."""
    radius = np.array([7.0, 14.0, 25.0])
    baseline = np.array([[2.0, 0.0, -4.0]])
    values = np.array([[3.0, 1.0, -2.0]])
    before = values.copy()
    figure, axes = plots.plot_Map3(
        map3=[(radius, values)], map3_ref=(radius, baseline),
        triplets=np.array([[0, 2, 3]]), show=None,
    )
    plotted = axes[0, 0].lines[0].get_ydata()
    np.testing.assert_allclose(plotted[[0, 2]], [50.0, -50.0])
    assert np.isnan(plotted[1])
    assert "undefined at 1" in capsys.readouterr().out
    np.testing.assert_array_equal(values, before)
    assert axes[0, 0].texts[0].get_text() == "(1,3,4)"
    plt.close(figure)


def test_sigma_residual_uses_given_errors():
    """A plotted sigma means the supplied diagonal error, not a fitted scale."""
    radius = np.array([7.0, 14.0])
    figure, axes = plots.plot_Map3(
        map3=[(radius, np.array([[3.0, -1.0]]))],
        map3_ref=(radius, np.array([[1.0, 0.0]])),
        triplets=np.array([[1, 0, 0]]), sigma=np.array([[2.0, 0.5]]),
        residual="sigma", show=None,
    )
    np.testing.assert_allclose(axes[0, 0].lines[0].get_ydata(), [[1.0, -2.0]][0])
    plt.close(figure)


def test_zero_component_is_not_excluded():
    """A genuine zero IA contribution remains a visible zero curve."""
    coordinate = np.array([0.0, 1.0, 2.0])
    figure, axes = plots.plot_ia_components(
        components=[(coordinate, np.zeros((1, 3)))], component_labels=["zero operator"],
        coordinate_label="sample", quantity_label="signal", xscale="linear", show=None,
    )
    assert not any(text.get_text() == "excluded" for text in axes[0, 0].texts)
    np.testing.assert_array_equal(axes[0, 0].lines[0].get_ydata(), 0)
    plt.close(figure)


def test_gamma_map_axis_order_and_mask():
    """A rectangular grid must not swap its two Fourier/source legs."""
    first = np.array([1.0, 2.0])
    second = np.array([3.0, 5.0, 9.0])
    values = np.zeros((1, 4, 2, 3), dtype=complex)
    values[0, 0] = np.arange(6).reshape(2, 3) - 2
    original = values.copy()
    figure, axes = plots.plot_Gamma(
        gamma=(first, second, values), triplets=np.array([[0, 1, 2]]),
        opening_angle_deg=60, projection="centroid TATT", show=None,
    )
    np.testing.assert_array_equal(axes[0, 0].collections[0].get_array(),
                                  values[0, 0].real.T)
    np.testing.assert_array_equal(values, original)
    plt.close(figure)
    figure, axes = plots.plot_Gamma(
        gamma=(first, second, values), gamma_ref=(first, second, original),
        triplets=np.array([[0, 1, 2]]), opening_angle_deg=60,
        projection="centroid TATT", residual="percent", show=None,
    )
    displayed = axes[0, 0].collections[0].get_array()
    assert np.ma.count_masked(displayed) == 1
    plt.close(figure)


def test_natural_real_imaginary_order():
    """The first row contains ReGamma0..3 and the second ImGamma0..3."""
    phi = np.array([0.5, 1.0])
    values = np.arange(8).reshape(4, 2) + 1j * (10 + np.arange(8).reshape(4, 2))
    figure, axes = plots.plot_natural_components(
        gamma=[(phi, values)], coordinate_label="phi [rad]",
        slice_label="equal sides; source(1,1,1)", projection="centroid", show=None,
    )
    for component in range(4):
        np.testing.assert_array_equal(axes[0, component].lines[0].get_ydata(),
                                      values[component].real)
        np.testing.assert_array_equal(axes[1, component].lines[0].get_ydata(),
                                      values[component].imag)
    plt.close(figure)


@pytest.mark.parametrize("change", ["grid", "scale", "style"])
def test_invalid_comparison_does_not_create_figure(change):
    """Malformed requests must not leave a partially displayed comparison."""
    coordinate = np.array([1.0, 2.0])
    reference = (coordinate, np.ones((1, 2)))
    arguments = {
        "components": [(coordinate, np.ones((1, 2)))],
        "components_ref": reference,
        "component_labels": ["term"],
        "coordinate_label": "x",
        "quantity_label": "value",
        "show": None,
    }
    if change == "grid":
        arguments["components_ref"] = (coordinate + 0.1, np.ones((1, 2)))
    elif change == "scale":
        arguments["residual"] = "scaled"
        arguments["reference_scale"] = 0.0
    else:
        arguments["linestyle"] = ["invalid-style"]
    before = plt.get_fignums()
    assert plots.plot_ia_components(**arguments) == 0
    assert plt.get_fignums() == before


def test_parameter_error_ratios_and_reference_line():
    """Small ratios must not cause the reference ratio1 to leave the axes."""
    figure, axis = plots.plot_parameter_errors(
        names=["a1", "a2"], errors=[np.array([0.3, 0.6])],
        errors_ref=np.array([1.0, 2.0]), legend=["joint"], show=None,
    )
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [0.3, 0.3])
    assert axis.get_ylim()[0] <= 1 <= axis.get_ylim()[1]
    plt.close(figure)


def test_rescaled_error_bars_and_shared_tick_labels():
    """Shared bands include observed errors, even above the model's amplitude."""
    radius = np.array([7.0, 14.0])
    values = np.array([[1.e-8, 0.5e-8], [2.e-8, 1.e-8]])
    errors = np.full_like(values, 1.e-7)
    figure, axes = plots.plot_Map3(
        map3=[(radius, values)], triplets=np.array([[0, 0, 0], [0, 0, 1]]),
        data=(radius, values), sigma=errors, rescale=1, ncols=2,
        bintextsize=21, legendfontsize=18, markersize=6, show=None,
    )
    assert axes[0, 0].get_ylim()[1] > 11
    assert 6 < axes[0, 0].get_xlim()[0] < 7
    assert axes[0, 0].get_shared_y_axes().joined(axes[0, 0], axes[0, 1])
    assert not axes[0, 1].get_yticklabels()
    assert axes[0, 0].texts[0].get_fontsize() == 21
    plt.close(figure)


@pytest.mark.parametrize("mode", ["sigma", "scaled"])
def test_residual_overflow_refused_before_drawing(mode, capsys):
    """Finite inputs can overflow a ratio; do not draw an infinite ordinate."""
    coordinate = np.array([1.0, 2.0])
    before = plt.get_fignums()
    result = plots.plot_ia_components(
        components=[(coordinate, np.full((1, 2), 1.e300))],
        components_ref=(coordinate, np.ones((1, 2))),
        component_labels=["term"], coordinate_label="x", quantity_label="value",
        residual=mode, sigma=np.full((1, 2), 1.e-200),
        reference_scale=1.e-200, show=None,
    )
    assert result == 0
    assert "overflowed" in capsys.readouterr().out
    assert plt.get_fignums() == before


def test_panel_titles_do_not_cover_comparison_curves():
    """A component label can sit above the axes instead of hiding an extremum."""
    coordinate = np.array([1.0, 2.0])
    figure, axes = plots.plot_code_comparison(
        x=coordinate, reference=np.ones((1, 2)), ported=np.array([[1.0, 2.0]]),
        labels=["ReGamma0"], coordinate_label="x", quantity_label="Gamma",
        bintextpos=None, show=None,
    )
    assert axes[0, 0].get_title() == "ReGamma0"
    assert not axes[0, 0].texts
    np.testing.assert_array_equal(axes[0, 0].lines[0].get_ydata(), [0.0, 100.0])
    plt.close(figure)
