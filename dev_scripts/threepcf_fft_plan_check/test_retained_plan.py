"""Check preparation reuse separately from physics and runtime speed.

These tests run only in the privately instrumented extension produced by
build_validation.py. Every FFT input is checked for finite initialized data,
and every execution checks its new-array alignment class. Counters measure
plan lifetime, not performance. Existing NumPy/SciPy transform references
supply an independent implementation of the finite operator.
"""

import gc
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_transform import numpy_two_bessel, small_transform_inputs


def settings(case="baseline"):
    """Return immutable geometry, changing one control per named case."""
    arguments = {
        "ell": np.geomspace(start=0.05, stop=30.0, num=24),
        "nu1": 1.13,
        "nu2": 0.89,
        "xy": 1.7,
        "dlnt": 0.05,
        "padding": 2,
        "window": True,
        "window_width": 0.5,
    }
    if case == "grid":
        arguments["ell"] *= 1.2
    elif case == "spacing":
        arguments["ell"] = np.geomspace(start=0.1, stop=30.0, num=24)
    elif case == "same_fft_size":
        arguments["ell"] = np.geomspace(start=0.05, stop=30.0, num=20)
        arguments["padding"] = 4
    elif case == "size":
        arguments["ell"] = np.geomspace(start=0.05, stop=30.0, num=32)
    elif case == "nu1":
        arguments["nu1"] = 0.81
    elif case == "nu2":
        arguments["nu2"] = 1.53
    elif case == "xy":
        arguments["xy"] = 0.9
    elif case == "dlnt":
        arguments["dlnt"] = 0.0
    elif case == "padding":
        arguments["padding"] = 0
    elif case == "window":
        arguments["window"] = False
    elif case == "width":
        arguments["window_width"] = 0.8
    elif case != "baseline":
        raise ValueError("Unknown geometry case: " + case)
    return arguments


def inputs(ell, variant):
    """Supply two signed, asymmetric inputs with different shapes."""
    values = ell[:, None] ** 1.2 * ell[None, :] ** 0.8
    values *= np.exp(-0.2 * ell[:, None] - 0.4 * ell[None, :])
    if variant == "B":
        values *= np.sin(0.2 * ell[:, None]) - 0.3 * np.cos(ell[None, :])
    return np.ascontiguousarray(values)


def copied_view(values):
    """Make a read-only sliced array that needs notebook conversion."""
    shape = list(values.shape)
    shape[-1] *= 2
    storage = np.empty(shape=shape, dtype=np.float64)
    view = storage[..., ::2]
    view[...] = values
    view.setflags(write=False)
    return view


def assert_bits(actual, expected):
    """Compare all bytes after accounting for C/F array layout."""
    left = np.ascontiguousarray(actual).view(dtype=np.uint64)
    right = np.ascontiguousarray(expected).view(dtype=np.uint64)
    np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
@pytest.mark.parametrize("case", [
    "baseline", "grid", "spacing", "same_fft_size", "size", "nu1",
    "nu2", "xy", "dlnt", "padding", "window", "width",
])
def test_all_geometry_and_replacements(compiled, adapter, case):
    """A-B-A and many inverse orders agree with one-shot and NumPy."""
    arguments = settings(case=case)
    if adapter == "production":
        plan = compiled.production.TwoBesselPlan(max_order=103, **arguments)
    else:
        notebook_arguments = arguments.copy()
        notebook_arguments["ell"] = copied_view(values=arguments["ell"])
        plan = compiled.notebook.two_bessel_plan(
            max_order=103, **notebook_arguments)
    first = None
    for variant in ("A", "B", "A"):
        values = inputs(ell=arguments["ell"], variant=variant)
        if adapter == "production":
            plan.forward(integrand=values)
        else:
            compiled.notebook.two_bessel_forward(
                plan=plan, integrand=copied_view(values=values))
        for order1, order2 in ((0, 0), (1, 3), (6, 2), (97, 103)):
            if adapter == "production":
                actual = plan.inverse(order1=order1, order2=order2)
            else:
                actual = compiled.notebook.two_bessel_inverse(
                    plan=plan, order1=order1, order2=order2)
            reference = compiled.production.two_bessel(
                integrand=values, order1=order1, order2=order2, **arguments)
            assert_bits(actual=actual, expected=reference)
            independent = numpy_two_bessel(
                integrand=values, order1=order1, order2=order2, **arguments)
            scale = np.max(np.abs(independent))
            np.testing.assert_allclose(actual, independent, rtol=8.0e-11,
                                       atol=8.0e-13 * scale)
            if (order1, order2) == (0, 0) and variant == "A":
                if first is None:
                    first = actual.copy()
                else:
                    assert_bits(actual=actual, expected=first)
            # Altering a returned array must never alter stored coefficients.
            actual.fill(-999.0)


def test_lifetime_counts_and_output_ownership(compiled):
    """Ten inputs require twenty one-shot plans or two retained plans."""
    gc.collect()
    arguments = settings()
    values = inputs(ell=arguments["ell"], variant="A")
    start = compiled._fftw_trace_counts()
    for index in range(10):
        compiled.production.two_bessel(
            integrand=values * (index + 1), order1=2, order2=3, **arguments)
    once = compiled._fftw_trace_counts()
    assert once["created"] - start["created"] == 20
    assert once["destroyed"] - start["destroyed"] == 20
    plan = compiled.production.TwoBesselPlan(max_order=6, **arguments)
    outputs = []
    for index in range(10):
        outputs.append(plan.evaluate(
            integrand=values * (index + 1), order1=2, order2=3))
    retained = compiled._fftw_trace_counts()
    assert retained["created"] - once["created"] == 2
    assert retained["destroyed"] == once["destroyed"]
    frozen = outputs[0].copy()
    plan.forward(integrand=values)
    values.fill(0.0)
    np.testing.assert_array_equal(plan.inverse(order1=2, order2=3), frozen)
    del plan
    gc.collect()
    after = compiled._fftw_trace_counts()
    assert after["live"] == start["live"]
    assert after["destroyed"] - retained["destroyed"] == 2
    assert_bits(actual=outputs[0], expected=frozen)


def test_independent_owners_and_core_reset(compiled):
    """Different geometries and global resets leave retained inputs intact."""
    arguments = settings()
    values = inputs(ell=arguments["ell"], variant="A")
    plan = compiled.production.TwoBesselPlan(max_order=6, **arguments)
    reference = plan.evaluate(integrand=values, order1=2, order2=3)
    theta = plan.theta
    expected_theta = theta.copy()
    theta.fill(0.0)
    arguments["ell"].fill(-5.0)
    assert_bits(actual=plan.theta, expected=expected_theta)
    others = []
    for case in ("same_fft_size", "size", "grid", "nu1"):
        other_arguments = settings(case=case)
        other = compiled.production.TwoBesselPlan(
            max_order=2, **other_arguments)
        other.evaluate(integrand=inputs(ell=other_arguments["ell"],
                                       variant="B"), order1=0, order2=2)
        others.append(other)
    compiled.core.initial_setup()
    assert_bits(actual=plan.inverse(order1=2, order2=3), expected=reference)
    del other
    others.clear()
    gc.collect()
    assert_bits(actual=plan.inverse(order1=2, order2=3), expected=reference)


def test_destroy_first_compatible_owner(compiled):
    """Deleting one identically configured owner leaves the other usable."""
    arguments = settings()
    values = inputs(ell=arguments["ell"], variant="B")
    first = compiled.production.TwoBesselPlan(max_order=6, **arguments)
    second = compiled.production.TwoBesselPlan(max_order=6, **arguments)
    first.forward(integrand=values)
    reference = second.evaluate(integrand=values, order1=2, order2=3)
    del first
    gc.collect()
    assert_bits(actual=second.inverse(order1=2, order2=3), expected=reference)


def test_notebook_evaluate_and_strict_production_types(compiled):
    """The copied adapter accepts lists/views; production refuses conversion."""
    arguments = settings()
    values = inputs(ell=arguments["ell"], variant="B")
    plan = compiled.notebook.two_bessel_plan(max_order=6, **arguments)
    actual = compiled.notebook.two_bessel_evaluate(
        plan=plan, integrand=values.tolist(), order1=2, order2=3)
    expected = compiled.production.two_bessel(
        integrand=values, order1=2, order2=3, **arguments)
    assert_bits(actual=actual, expected=expected)
    for invalid in (values.astype(np.float32), copied_view(values=values)):
        with pytest.raises(TypeError):
            plan.evaluate(integrand=invalid, order1=2, order2=3)
    assert_bits(actual=plan.inverse(order1=2, order2=3), expected=expected)


def test_natural_plan_reuses_fft_pair_with_six_workers(compiled):
    """Retained finite-triangle plans use two recipes at 1, 4, 6 workers."""
    ell, phi, indices, multipoles = small_transform_inputs()
    references = None
    try:
        for workers in (1, 4, 6):
            compiled.set_omp_threads(nthreads=workers)
            before = compiled._fftw_trace_counts()
            plan = compiled.production.TransformPlan(
                ell=ell, phi=phi, indices=indices, lmax=3, mmax=5,
                npsi=0, projection="X", padding=3)
            after = compiled._fftw_trace_counts()
            assert after["created"] - before["created"] == 2
            result = plan.natural_components(b_l=multipoles)
            changed = np.ascontiguousarray(multipoles[:, ::-1, :])
            plan.natural_components(b_l=changed)
            restored = plan.natural_components(b_l=multipoles)
            if references is None:
                references = (result[0].copy(), result[1].copy())
            for part in range(2):
                assert_bits(actual=result[part], expected=references[part])
                assert_bits(actual=restored[part], expected=references[part])
            final = compiled._fftw_trace_counts()
            assert final["created"] == after["created"]
            assert final["destroyed"] == after["destroyed"]
            del plan
            gc.collect()
    finally:
        compiled.set_omp_threads(nthreads=1)


@pytest.mark.parametrize("case,diagnostic", [
    ("unready", "call forward before inverse"),
    ("order", "orders must be in [0, 6]"),
    ("negative", "orders must be in [0, 6]"),
    ("shape", "integrand shape mismatch"),
    ("nonfinite", "finite values"),
])
def test_named_refusals(case, diagnostic):
    """Invalid calls stop before FFTW can consume invalid coefficients."""
    process = subprocess.run(
        args=[sys.executable, "-B", str(Path(__file__)), case],
        check=False, capture_output=True, text=True, timeout=30)
    message = process.stdout + process.stderr
    assert process.returncode == 1
    assert diagnostic in message
    assert "runtime error:" not in message
    assert "Traceback" not in message


def test_fresh_process_result(compiled, tmp_path):
    """A changed retained input agrees bitwise with a new interpreter."""
    arguments = settings()
    plan = compiled.production.TwoBesselPlan(max_order=6, **arguments)
    plan.forward(integrand=inputs(ell=arguments["ell"], variant="A"))
    actual = plan.evaluate(integrand=inputs(ell=arguments["ell"], variant="B"),
                           order1=2, order2=3)
    output = tmp_path / "fresh.npy"
    subprocess.run(args=[sys.executable, "-B", str(Path(__file__)),
                         "fresh", str(output)], check=True, timeout=30)
    assert_bits(actual=actual, expected=np.load(file=output))


def child(case):
    """Select a private extension by path before importing it."""
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project / "interface"))
    import cosmolike_des_3pcf_interface as compiled

    compiled.set_omp_threads(nthreads=1)
    arguments = settings()
    plan = compiled.production.TwoBesselPlan(max_order=6, **arguments)
    values = inputs(ell=arguments["ell"], variant="B")
    if case == "fresh":
        result = plan.evaluate(integrand=values, order1=2, order2=3)
        np.save(file=sys.argv[2], arr=result)
    elif case == "unready":
        plan.inverse(order1=0, order2=2)
    elif case == "order":
        plan.evaluate(integrand=values, order1=7, order2=0)
    elif case == "negative":
        plan.evaluate(integrand=values, order1=-1, order2=0)
    elif case == "shape":
        plan.forward(integrand=np.zeros(shape=(2, 2)))
    elif case == "nonfinite":
        values[0, 0] = np.nan
        plan.forward(integrand=values)
    else:
        raise ValueError("Unknown child case: " + case)


if __name__ == "__main__":
    child(case=sys.argv[1])
