"""Check copied core variances and their optional halo-covariance consumer.

Use a private extension on PYTHONPATH and put the DES 3PCF test directory
and its data_vector directory there too. These checks use a power-law
oracle and controlled arrays. They establish neither survey convergence
nor equivalence of factorized growth and an evolving linear spectrum.
"""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import simpson
from scipy.optimize import brentq
from scipy.special import gamma

import cosmolike_des_3pcf_interface as compiled
from covariance.test_physical_covariance import covariance_arguments
from covariance.test_physical_covariance import make_covariance_model


def install_power(amplitude=30.0, reset=True):
    """Install P(k,z)=A k^-2.2 a^2, with k in h/Mpc and no neutrinos.

    This power-law slope matches FFTLog's bias and checks normalization.
    It does not test the response to other slopes or spectral curvature;
    the independent quadrature case below supplies the curved comparison.
    Dense redshift sampling limits interpolation of log power in redshift.
    """
    compiled.set_omp_threads(nthreads=1)
    core = compiled.core
    if reset:
        core.initial_setup()
    core.set_cosmological_parameters(
        omega_matter=0.3, omega_baryon=0.05, hubble=70.0, omega_nu_h2=0.0)
    wave = np.geomspace(start=1.0e-5, stop=1.0e3, num=257)
    redshift = np.linspace(start=0.0, stop=40.0, num=4001)
    power = amplitude * wave[:, None]**-2.2 / (1.0 + redshift[None, :])**2
    core.set_linear_power_spectrum(
        log10k=np.log10(wave), z=redshift, lnP=np.log(power).ravel())
    return core


def request_snapshot(core, z, mass, include_fit=True):
    """Request the current covariance's total-matter radius convention."""
    return core.variance_snapshot(
        z=z, mass=mass, reference_rho=2.77536627e11 * 0.3,
        include_fit=include_fit)


def attach_snapshot(arguments, snapshot):
    """Keep the coordinates and density with the variance values."""
    result = dict(arguments)
    result["halo_z"] = snapshot["z"]
    result["halo_mass"] = snapshot["mass"]
    result["halo_sigma2"] = snapshot["halo_sigma2"]
    result["halo_dlnsigma_dlnm"] = snapshot["halo_dlnsigma_dlnm"]
    result["halo_reference_rho"] = snapshot["reference_rho"]
    return result


def test_power_law_oracle():
    """Compare Gaussian/top-hat quantities with analytic infinite integrals."""
    core = install_power()
    z = np.array(object=[0.0, 0.5, 1.0])
    mass = np.array(object=[1.0e11, 1.0e12, 1.0e14])
    result = request_snapshot(core=core, z=z, mass=mass)
    growth2 = (1.0 + z)**-2
    radius = np.cbrt(3.0 * mass / (4.0 * np.pi * result["reference_rho"]))
    exponent = 0.8
    # Integrating k^exponent against each window gives its Mellin
    # integral. These closed forms set the infinite-range reference.
    tophat = (4.5 * np.pi * gamma(4.0 - exponent) * gamma(exponent / 2.0)
              / (2.0**(4.0 - exponent) * gamma((5.0 - exponent) / 2.0)**2
                 * gamma(4.0 - exponent / 2.0)))
    expected_sigma2 = (30.0 * tophat / (2.0 * np.pi**2)
                       * growth2[:, None] * radius[None, :]**-exponent)
    expected_rnl = (30.0 * gamma(exponent / 2.0) * growth2
                    / (4.0 * np.pi**2))**(1.0 / exponent)
    expected_sigma8 = np.sqrt(
        30.0 * tophat * growth2 / (2.0 * np.pi**2 * 8.0**exponent))
    np.testing.assert_allclose(result["rnl"], expected_rnl, rtol=1.0e-3)
    np.testing.assert_allclose(result["neff"], -2.2, rtol=1.0e-3)
    np.testing.assert_allclose(result["sigma8"], expected_sigma8, rtol=1.0e-3)
    np.testing.assert_allclose(result["halo_sigma2"], expected_sigma2, rtol=1.0e-3)
    np.testing.assert_allclose(result["halo_dlnsigma_dlnm"], -exponent / 6.0,
                               rtol=1.0e-3)


def test_curved_spectrum_quadrature():
    """Compare a turnover spectrum with direct smoothing integrals.

    The reference integrates the supplied log-linear reconstruction, with
    its endpoint power laws, on 200001 log-k nodes. It shares no FFTLog
    coefficients or variance lookup with the C calculation.
    """
    core = install_power()
    wave = np.geomspace(start=1.0e-5, stop=1.0e3, num=513)
    redshift = np.linspace(start=0.0, stop=40.0, num=4001)
    linear = 2.0e4 * wave / (1.0 + (wave / 0.15)**2)**1.8
    power = linear[:, None] / (1.0 + redshift[None, :])**2
    core.set_linear_power_spectrum(
        log10k=np.log10(wave), z=redshift, lnP=np.log(power).ravel())
    z = np.array(object=[0.0, 0.5, 1.0])
    mass = np.array(object=[1.0e11, 1.0e12, 1.0e14])
    actual = request_snapshot(core=core, z=z, mass=mass)
    log_wave = np.log(wave)
    log_power = np.log(linear)
    nodes = np.linspace(start=np.log(1.0e-7), stop=np.log(1.0e4), num=200001)
    values = np.interp(x=nodes, xp=log_wave, fp=log_power)
    low = nodes < log_wave[0]
    high = nodes > log_wave[-1]
    values[low] = (log_power[0] + (nodes[low] - log_wave[0])
                   * (log_power[1] - log_power[0]) / (log_wave[1] - log_wave[0]))
    values[high] = (log_power[-1] + (nodes[high] - log_wave[-1])
                    * (log_power[-1] - log_power[-2])
                    / (log_wave[-1] - log_wave[-2]))
    k = np.exp(nodes)
    delta = np.exp(values + 3.0 * nodes) / (2.0 * np.pi**2)
    expected_rnl = []
    expected_neff = []
    for value in z:
        growth2 = (1.0 + value)**-2

        def crossing(radius):
            return simpson(y=delta * np.exp(-(k * radius)**2), x=nodes) * growth2 - 1.0

        radius = brentq(f=crossing, a=1.0e-3, b=100.0)
        slope_integrand = delta * (k * radius)**2 * np.exp(-(k * radius)**2)
        expected_rnl.append(radius)
        expected_neff.append(-3.0 + 2.0 * simpson(y=slope_integrand, x=nodes) * growth2)
    radius = np.cbrt(3.0 * mass / (4.0 * np.pi * actual["reference_rho"]))
    expected_variance = []
    for value in radius:
        argument = k * value
        small = argument < 1.0e-3
        window = np.empty_like(argument)
        window[small] = 1.0 - argument[small]**2 / 10.0 + argument[small]**4 / 280.0
        large = argument[~small]
        window[~small] = 3.0 * (np.sin(large) - large * np.cos(large)) / large**3
        expected_variance.append(simpson(y=delta * window**2, x=nodes))
    expected_variance = np.asarray(expected_variance)[None, :] / (1.0 + z[:, None])**2
    np.testing.assert_allclose(actual["rnl"], expected_rnl, rtol=1.0e-3)
    np.testing.assert_allclose(actual["neff"], expected_neff, rtol=1.0e-3)
    np.testing.assert_allclose(actual["halo_sigma2"], expected_variance, rtol=1.0e-3)


def test_owned_snapshot_and_replacement():
    """A/B/A installation restores arrays without modifying an old snapshot."""
    z = np.array(object=[0.0, 1.0])
    mass = np.array(object=[1.0e12, 1.0e14])
    core = install_power()
    first = request_snapshot(core=core, z=z, mass=mass)
    saved = first["halo_sigma2"].copy()
    generation = core.model_generation()
    repeat = request_snapshot(core=core, z=z, mass=mass)
    assert generation == core.model_generation()
    np.testing.assert_array_equal(repeat["halo_sigma2"], saved)
    core = install_power(amplitude=60.0, reset=False)
    changed = request_snapshot(core=core, z=z, mass=mass)
    np.testing.assert_allclose(changed["halo_sigma2"], 2.0 * saved, rtol=1.0e-12)
    np.testing.assert_array_equal(first["halo_sigma2"], saved)
    core = install_power(reset=False)
    restored = request_snapshot(core=core, z=z, mass=mass)
    np.testing.assert_array_equal(restored["halo_sigma2"], saved)
    z[:] += 0.2
    mass[:] *= 2.0
    np.testing.assert_array_equal(first["z"], [0.0, 1.0])
    np.testing.assert_array_equal(first["mass"], [1.0e12, 1.0e14])


def test_reset_and_thread_count():
    """Explicit reset and worker changes preserve results and owned snapshots."""
    core = install_power()
    z = np.array(object=[0.0, 1.0])
    mass = np.array(object=[1.0e12, 1.0e14])
    first = request_snapshot(core=core, z=z, mass=mass)
    compiled.set_omp_threads(nthreads=6)
    parallel = request_snapshot(core=core, z=z, mass=mass)
    core = install_power()
    reset = request_snapshot(core=core, z=z, mass=mass)
    for name in ("rnl", "neff", "sigma8", "halo_sigma2", "halo_dlnsigma_dlnm"):
        np.testing.assert_array_equal(parallel[name], first[name])
        np.testing.assert_array_equal(reset[name], first[name])


def test_halo_only_and_fit_only():
    """Halo-only works when the nonlinear radius lies outside the fit table."""
    core = install_power(amplitude=1.0e-7)
    result = request_snapshot(
        core=core, z=np.array(object=[0.0]), mass=np.array(object=[1.0e12]),
        include_fit=False)
    assert result["rnl"].size == 0
    assert np.all(result["halo_sigma2"] > 0.0)
    core = install_power()
    fit = core.variance_snapshot(z=np.array(object=[0.0]), mass=None)
    assert fit["halo_sigma2"].shape == (1, 0)
    assert fit["rnl"].shape == (1,)


@pytest.mark.parametrize("adapter_name", ["production", "notebook"])
def test_covariance_consumes_exact_snapshot(adapter_name):
    """Both adapters use supplied variances and the correct mass derivative."""
    core = install_power()
    model = make_covariance_model(compiled=compiled)
    arguments = covariance_arguments()
    z = model.radial()["z"]
    snapshot = request_snapshot(core=core, z=z, mass=arguments["mass"],
                                include_fit=False)
    values = attach_snapshot(arguments=arguments, snapshot=snapshot)
    adapter = getattr(compiled, adapter_name)
    actual = adapter.halo_tables(
        model=model, inputs=values, z=z,
        k_profile=np.array(object=[0.0, 0.1]))
    np.testing.assert_array_equal(actual[0], snapshot["halo_sigma2"])
    peak = values["delta_c"]**2 / snapshot["halo_sigma2"]
    apeak = values["st_a"] * peak
    multiplicity = (values["st_norm"] * np.sqrt(2.0 * apeak / np.pi)
                    * (1.0 + apeak**-values["st_p"]) * np.exp(-0.5 * apeak))
    expected_abundance = (-values["halo_reference_rho"]
                          / values["mass"][None, :]**2
                          * snapshot["halo_dlnsigma_dlnm"] * multiplicity)
    np.testing.assert_allclose(actual[1], expected_abundance, rtol=2.0e-14)
    np.testing.assert_array_equal(actual[4][:, :, 0], 1.0)


def test_covariance_term_selection():
    """Replacing halo variances preserves the mean and non-halo contractions."""
    core = install_power()
    model = make_covariance_model(compiled=compiled)
    arguments = covariance_arguments()
    snapshot = request_snapshot(
        core=core, z=model.radial()["z"], mass=arguments["mass"],
        include_fit=False)
    values = attach_snapshot(arguments=arguments, snapshot=snapshot)
    old_mean, old_terms = compiled.production.physical_covariance(
        model=model, inputs=arguments)
    mean, terms = compiled.production.physical_covariance(
        model=model, inputs=values)
    copied_mean, copied_terms = compiled.notebook.physical_covariance(
        model=model, inputs=values)
    np.testing.assert_array_equal(mean, old_mean)
    np.testing.assert_array_equal(copied_mean, mean)
    np.testing.assert_array_equal(copied_terms, terms)
    np.testing.assert_array_equal(terms[[0, 1, 3]], old_terms[[0, 1, 3]])
    assert np.all(np.isfinite(terms))
    assert np.any(terms[4] != old_terms[4])


def refusal_case(name):
    """Exercise process-stopping C guards without ending the pytest process."""
    core = install_power()
    if name == "reset_without_power":
        core.variance_snapshot(z=np.array(object=[0.0]))
        core.initial_setup()
        print("VARIANCE_REQUEST_REACHED", flush=True)
        core.variance_snapshot(z=np.array(object=[0.0]))
        return
    if name == "no_outputs":
        print("VARIANCE_REQUEST_REACHED", flush=True)
        core.variance_snapshot(z=np.array(object=[0.0]), include_fit=False)
        return
    if name == "negative_density":
        print("VARIANCE_REQUEST_REACHED", flush=True)
        core.variance_snapshot(z=np.array(object=[0.0]), reference_rho=-1.0)
        return
    if name == "descending_z":
        print("VARIANCE_REQUEST_REACHED", flush=True)
        core.variance_snapshot(z=np.array(object=[1.0, 0.0]))
        return
    model = make_covariance_model(compiled=compiled)
    arguments = covariance_arguments()
    z = model.radial()["z"]
    snapshot = request_snapshot(core=core, z=z, mass=arguments["mass"],
                                include_fit=False)
    values = attach_snapshot(arguments=arguments, snapshot=snapshot)
    if name == "mass_changed":
        values["mass"] = 1.01 * values["mass"]
    elif name == "redshift_changed":
        values["halo_z"] = values["halo_z"] + 0.01
    elif name == "density_changed":
        values["halo_reference_rho"] *= 1.01
    elif name == "negative_variance":
        values["halo_sigma2"][0, 0] = -1.0
    elif name == "positive_slope":
        values["halo_dlnsigma_dlnm"][0, 0] = 1.0
    elif name == "partial":
        del values["halo_mass"]
    elif name == "shape":
        values["halo_sigma2"] = values["halo_sigma2"][:, :1].copy()
    else:
        raise ValueError(f"Unknown refusal case {name}")
    print("VARIANCE_REQUEST_REACHED", flush=True)
    compiled.production.halo_tables(
        model=model, inputs=values, z=z,
        k_profile=np.array(object=[0.0, 0.1]))


@pytest.mark.parametrize("name", [
    "reset_without_power",
    "no_outputs",
    "negative_density",
    "descending_z",
    "mass_changed",
    "redshift_changed",
    "density_changed",
    "negative_variance",
    "positive_slope",
    "partial",
    "shape",
])
def test_refusal(name):
    """Invalid or stale snapshot inputs stop before a covariance is returned."""
    result = subprocess.run(
        args=[sys.executable, str(Path(__file__).resolve()), "--refusal", name],
        capture_output=True, text=True, check=False)
    assert result.returncode == 1, result.stdout + result.stderr
    # An import or setup failure is not evidence that the intended guard
    # rejected its input. The marker is emitted immediately before it.
    assert "VARIANCE_REQUEST_REACHED" in result.stdout
    assert "variance" in (result.stdout + result.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--refusal":
        raise SystemExit("Use pytest, or --refusal followed by one case name.")
    refusal_case(name=sys.argv[2])
