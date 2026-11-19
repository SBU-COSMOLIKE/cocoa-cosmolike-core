"""Check optional fit anchors, owned model inputs and bridge state validity.

The deliberately simple anchor arrays are controlled numerical inputs. They
establish selection and ownership, not the accuracy of the shared FFTLog fit.
The companion test_snapshot.py compares that engine with separate integrals.
"""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import cosmolike_des_3pcf_interface as compiled
from data_vector.test_physics_model import physical_inputs
from test_supplied_baseline import baseline_input, install_bridge


SIDES = np.array(object=[[0.3, 0.4, 0.5], [0.8, 1.0, 1.3]])
REDSHIFTS = np.array(object=[0.2, 0.8])


def fit_input():
    """Return positive smooth anchors on the model's entire redshift range."""
    z = np.linspace(start=0.0, stop=2.0, num=33)
    return {
        "z": z,
        "rnl": 1.0 / (1.0 + z)**2,
        "neff": np.full(shape=z.size, fill_value=-1.8),
        "sigma8": 0.8 / (1.0 + z),
    }


def matter(model):
    """Read two nonsymmetric triangles from one immutable owner."""
    return model.matter(sides=SIDES, redshifts=REDSHIFTS)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_fit_anchors_select_amplitude_and_own_inputs(adapter):
    """Supplying fit arrays keeps P's amplitude and consumes each fit field."""
    inputs = physical_inputs()
    fit = fit_input()
    factory = getattr(compiled, adapter).PhysicsModel
    model = factory(**inputs, variance=fit)
    wave = np.ascontiguousarray(inputs["k"][10:60:8])
    expected = inputs["p_linear0"][10:60:8]
    actual = model.power(k=wave, z=np.zeros_like(a=wave), mode=0)
    np.testing.assert_allclose(actual=actual, desired=expected, rtol=2.0e-14)
    reference = matter(model=model)
    inputs["sigma8"] = 1.1
    other_scalar = factory(**inputs, variance=fit)
    np.testing.assert_array_equal(x=matter(model=other_scalar), y=reference)
    for field in ("rnl", "neff", "sigma8"):
        changed = {name: values.copy() for name, values in fit.items()}
        changed[field] *= 1.1
        different = factory(**inputs, variance=changed)
        assert not np.array_equal(a1=matter(model=different), a2=reference)
    for values in fit.values():
        values[:] = 99.0
    inputs["p_linear0"][:] = 99.0
    np.testing.assert_array_equal(x=matter(model=model), y=reference)


def test_fit_notebook_accepts_strides_and_matches_production():
    """Notebook copying accepts sliced anchors and keeps the same C result."""
    inputs = physical_inputs()
    fit = fit_input()
    copied = {}
    for name, values in fit.items():
        storage = np.repeat(a=values, repeats=2)
        copied[name] = storage[::2]
    first = compiled.production.PhysicsModel(**inputs, variance=fit)
    second = compiled.notebook.PhysicsModel(**inputs, variance=copied)
    np.testing.assert_array_equal(x=matter(model=first), y=matter(model=second))


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_supplied_b_accepts_fit_amplitude_contract(adapter):
    """Supplied B remains unchanged while its linear IA power keeps input P."""
    inputs = physical_inputs()
    for name in ("omega_de", "w0", "wa", "ns"):
        del inputs[name]
    inputs["omega_m_z"] = np.full(shape=inputs["z"].size, fill_value=0.3)
    model = getattr(compiled, adapter).PhysicsModelWithBispectrum(
        **inputs, baseline=baseline_input(varying=False), variance=fit_input())
    np.testing.assert_allclose(actual=matter(model=model), desired=7.0,
                               rtol=2.0e-14)
    wave = np.ascontiguousarray(inputs["k"][10:60:8])
    np.testing.assert_allclose(
        actual=model.power(k=wave, z=np.zeros_like(a=wave), mode=0),
        desired=inputs["p_linear0"][10:60:8], rtol=2.0e-14)


def test_bridge_variance_reuse_replace_clear_reset():
    """A/B/A, identical resends and reset preserve retained model owners."""
    inputs = physical_inputs()
    install_bridge(compiled=compiled, inputs=inputs, install_scalars=True)
    core = compiled.core
    legacy = core.model_from_core()
    legacy_values = matter(model=legacy)
    fit = fit_input()
    core.set_variance(variance=fit)
    first = core.model_from_core()
    values = matter(model=first)
    assert first is not legacy
    core.set_variance(variance=fit)
    assert core.model_from_core() is first
    changed = {name: row.copy() for name, row in fit.items()}
    changed["rnl"] *= 1.1
    core.set_variance(variance=changed)
    second = core.model_from_core()
    assert not np.array_equal(a1=matter(model=second), a2=values)
    core.set_variance(variance=fit)
    np.testing.assert_array_equal(x=matter(model=core.model_from_core()),
                                  y=values)
    fit["rnl"][:] = 99.0
    np.testing.assert_array_equal(x=matter(model=first), y=values)
    core.set_variance()
    np.testing.assert_array_equal(x=matter(model=core.model_from_core()),
                                  y=legacy_values)
    core.initial_setup()
    np.testing.assert_array_equal(x=matter(model=first), y=values)
    install_bridge(compiled=compiled, inputs=inputs, install_scalars=True)
    np.testing.assert_array_equal(x=matter(model=core.model_from_core()),
                                  y=legacy_values)


@pytest.mark.parametrize("change", [
    "power", "cosmology", "mass_nodes", "scale_factor_nodes",
    "gaussian_radius_nodes", "gaussian_scale_factor_nodes",
    "gaussian_inverse_nodes",
])
def test_bridge_refuses_stale_fit_and_accepts_resupply(change):
    """State changes refuse stale arrays; explicit resupply renews validity."""
    setup = (
        "from test_fit_input import *\n"
        "inputs = physical_inputs()\n"
        "install_bridge(compiled=compiled, inputs=inputs, install_scalars=True)\n"
        "core = compiled.core\n"
        "fit = fit_input()\n"
        "core.set_variance(variance=fit)\n"
        "first = core.model_from_core()\n"
        "before = matter(model=first)\n"
    )
    if change == "power":
        mutation = (
            "linear = inputs['growth'][:, None]**2 * inputs['p_linear0'][None, :]\n"
            "core.set_linear_power_spectrum(log10k=np.log10(inputs['k']), "
            "z=inputs['z'], lnP=np.log(1.1 * linear).ravel(order='F'))\n"
        )
    elif change == "cosmology":
        mutation = (
            "core.set_cosmological_parameters(omega_matter=0.31, "
            "omega_baryon=0.05, hubble=70.0, omega_nu_h2=0.0)\n"
        )
    else:
        mutation = (
            "sampling = core.variance_sampling()\n"
            f"sampling[{change!r}] *= 2\n"
            "core.set_variance_sampling(**sampling)\n"
        )
    failed = subprocess.run(
        args=[sys.executable, "-c", setup + mutation + "core.model_from_core()"],
        capture_output=True, text=True)
    assert failed.returncode != 0
    assert "supplied variance is stale" in failed.stdout + failed.stderr
    finish = (
        "core.set_variance(variance=fit)\n"
        "second = core.model_from_core()\n"
        "assert second is not first\n"
        "np.testing.assert_array_equal(x=matter(model=first), y=before)\n"
        "core.set_variance()\n"
        "core.model_from_core()\n"
    )
    subprocess.run(args=[sys.executable, "-c", setup + mutation + finish],
                   check=True, capture_output=True, text=True)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
@pytest.mark.parametrize("bad", ["missing", "extra", "shape", "nonfinite",
                                  "negative", "segments"])
def test_fit_refusals(adapter, bad):
    """Malformed dictionaries or physical anchors fail before a model exists."""
    source = (
        "from test_fit_input import *\n"
        "inputs = physical_inputs()\n"
        "fit = fit_input()\n"
    )
    changes = {
        "missing": "del fit['neff']\n",
        "extra": "fit['other'] = fit['neff']\n",
        "shape": "fit['neff'] = fit['neff'][:-1]\n",
        "nonfinite": "fit['neff'][2] = np.inf\n",
        "negative": "fit['rnl'][2] = -1.0\n",
        "segments": "fit['z'] = np.linspace(start=0.0, stop=2.0, num=33)**1.2\n",
    }
    source += changes[bad]
    source += "print('CONSTRUCTOR_REACHED', flush=True)\n"
    source += f"compiled.{adapter}.PhysicsModel(**inputs, variance=fit)\n"
    result = subprocess.run(args=[sys.executable, "-c", source],
                            capture_output=True, text=True)
    assert "CONSTRUCTOR_REACHED" in result.stdout
    assert result.returncode != 0


def test_piecewise_fit_anchors_are_supported():
    """Two uniform redshift segments need no binary-search interpolation."""
    z = np.concatenate((np.linspace(start=0.0, stop=1.0, num=17)[:-1],
                        np.linspace(start=1.0, stop=2.0, num=33)))
    fit = {"z": z, "rnl": 1.0 / (1.0 + z)**2,
           "neff": np.full(shape=z.size, fill_value=-1.8),
           "sigma8": 0.8 / (1.0 + z)}
    inputs = physical_inputs()
    first = compiled.production.PhysicsModel(**inputs, variance=fit)
    second = compiled.notebook.PhysicsModel(**inputs, variance=fit)
    np.testing.assert_array_equal(x=matter(model=first), y=matter(model=second))


def test_shared_snapshot_feeds_direct_and_bridge_models():
    """Actual shared FFTLog anchors reach both immutable model constructors."""
    inputs = physical_inputs()
    install_bridge(compiled=compiled, inputs=inputs, install_scalars=True)
    core = compiled.core
    z = np.linspace(start=0.0, stop=40.0, num=4001)
    linear = inputs["p_linear0"][:, None] / (1.0 + z[None, :])**2
    core.set_linear_power_spectrum(
        log10k=np.log10(inputs["k"]), z=z, lnP=np.log(linear).ravel())
    snapshot = core.variance_snapshot(
        z=np.linspace(start=0.0, stop=2.0, num=33), include_fit=True)
    fit = {name: snapshot[name] for name in ("z", "rnl", "neff", "sigma8")}
    core.set_variance(variance=fit)
    bridge = core.model_from_core()
    direct = compiled.production.PhysicsModel(**inputs, variance=fit)
    np.testing.assert_allclose(actual=matter(model=bridge),
                               desired=matter(model=direct), rtol=3.0e-12)
    before = matter(model=bridge)
    core.initial_setup()
    np.testing.assert_array_equal(x=matter(model=bridge), y=before)
