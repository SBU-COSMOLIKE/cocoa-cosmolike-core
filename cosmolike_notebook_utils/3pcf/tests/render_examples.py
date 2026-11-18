"""Render each plotting family from declared arrays without running C physics.

Public simulated Y3 inputs, saved original fastnc kernels, and an analytic
Gaussian shear reference answer different questions. The manifest identifies
each source; none of these figures claims a new survey likelihood result.
"""

import os
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["OMP_THREAD_LIMIT"] = "4"

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
from astropy.io import fits


CORE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CORE))
plots = importlib.import_module("cosmolike_notebook_utils.3pcf")


def checked_file(path, expected):
    """Check the actual Git blob identity before using a public input."""
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode()
    actual = hashlib.sha1(header + content).hexdigest()
    if actual != expected:
        raise ValueError(f"Public file hash differs: {path}")
    return path


def main():
    """Write labeled visual checks and their provenance to an explicit folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    project = arguments.project_dir.resolve()
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "mathtext.fontset": "stix", "font.family": "STIXGeneral",
        "axes.edgecolor": "black", "axes.linewidth": 1.0,
        "axes.labelsize": 19, "xtick.labelsize": 17, "ytick.labelsize": 17,
        "legend.fontsize": 16, "savefig.bbox": "tight", "savefig.dpi": 160,
        "xtick.top": False, "ytick.right": False,
    })
    records = []

    def save(name, result, title, source):
        """Keep the scientific provenance beside, and on, each saved figure."""
        if not isinstance(result, tuple):
            raise ValueError(f"Plot failed before saving {name}")
        figure, _ = result
        # The shared plotters own the top legend. Its own title stacks
        # provenance above the entries without overlapping a suptitle.
        if figure.legends:
            figure.legends[0].set_title(title, prop={"size": 17})
        else:
            figure.suptitle(title, fontsize=17)
        if name.startswith("covariance_diagonal"):
            for axis in figure.axes:
                axis.xaxis.set_major_locator(FixedLocator(radius))
                axis.xaxis.set_major_formatter(
                    FuncFormatter(lambda value, _: f"{value:g}")
                )
                axis.xaxis.set_minor_formatter(NullFormatter())
        figure.savefig(output / f"{name}.png")
        plt.close(figure)
        records.append({"file": f"{name}.png", "title": title, "source": source})

    style = {
        "xaxislabelsize": 21, "yaxislabelsize": 21,
        "xaxisticklabelsize": 17, "yaxisticklabelsize": 17,
        "bintextsize": 20, "bintextpos": None,
        "legendfontsize": 18, "markersize": 5,
    }
    public = project / "data/upstream_y3/data"
    moment_path = checked_file(
        path=public / "dv/sim_map3_emu-NLA-cosmoCosmogrid-24FEBRUARY2025_WITH_COV.fits",
        expected="5e26a7bafd42fdcbe837b2eb0639f18c3845ae5a",
    )
    with fits.open(moment_path) as table:
        rows = table["map3"].data.copy()
        covariance = table["COVMAT"].data.copy()
    radius = np.unique(rows["THETA1"])
    bin_columns = np.column_stack((rows["BIN1"], rows["BIN2"], rows["BIN3"]))
    triples = bin_columns[::radius.size] - 1
    expected_bins = np.repeat(triples + 1, radius.size, axis=0)
    if not np.array_equal(bin_columns, expected_bins):
        raise ValueError("Public Map3 ordering differs from panel-major ordering")
    for column in ("THETA1", "THETA2", "THETA3"):
        if not np.array_equal(rows[column], np.tile(radius, len(triples))):
            raise ValueError("Public Map3 filters are not equal-radius panel grids")
    moment = rows["VALUE"].reshape(len(triples), radius.size)
    errors = np.sqrt(np.diag(covariance)).reshape(moment.shape)
    public_source = "Pinned public simulated Y3 Map3 input; no C prediction."
    save("map3_absolute", plots.plot_Map3(
        map3=[(radius, moment)], triplets=triples, marker=["o"],
        figsize=(19, 12), show=None, **style,
    ), "Public simulated Y3 aperture moment", public_source)
    save("map3_rescaled_errors", plots.plot_Map3(
        map3=[(radius, moment)], triplets=triples, data=(radius, moment),
        sigma=errors, rescale=1, figsize=(19, 12), show=None, **style,
    ), "Public simulated Y3 input with supplied diagonal errors", public_source)
    save("map3_sigma", plots.plot_Map3(
        map3=[(radius, moment)], map3_ref=(radius, np.zeros_like(moment)),
        triplets=triples, sigma=errors, residual="sigma", marker=["o"],
        figsize=(19, 12), show=None, **style,
    ), "Supplied Y3 signal / diagonal error; zero-signal reference",
        public_source + " Per-entry significance, not a joint signal-to-noise.")

    reference_path = project / "tests/data_vector/reference_physics_fastnc.json"
    record = json.loads(reference_path.read_text())
    if record["source_commit"] != "7b2336ce37a909bed51f07f32a2a4f096e71b6ce":
        raise ValueError("Saved IA reference has an unexpected source commit")
    linear = np.asarray(record["alignment"]["0"]).T[:6]
    nonlinear = np.asarray(record["alignment"]["1"]).T[:6]
    samples = np.arange(1, linear.shape[1] + 1)
    names = ["ddE", "dEd", "Edd", "dEE", "EEd", "EdE"]
    ia_source = (
        "Saved original fastnc 7b2336 IA kernels on six declared test triangles; "
        "linear versus nonlinear P(k,0)D(z)^2 is a physical input choice. "
        "SHA256 of saved record: "
        + hashlib.sha256(reference_path.read_bytes()).hexdigest()
    )
    ia_options = {
        "component_labels": names, "coordinate_label": "Declared triangle index",
        "quantity_label": r"$B_{\rm IA}$ [$(\mathrm{Mpc}/h)^6$]",
        "xscale": "linear", "figsize": (15, 8), "show": None,
    }
    for scaled in (None, 1):
        suffix = "absolute" if scaled is None else "rescaled"
        save(f"ia_{suffix}", plots.plot_ia_components(
            components=[(samples, linear), (samples, nonlinear)],
            legend=["Original linear power", "Original nonlinear power"],
            rescale=scaled, **ia_options, **style,
        ), "Signed original IA kernels: two physical power choices", ia_source)
    for mode in ("percent", "scaled", "absolute"):
        save(f"ia_residual_{mode}", plots.plot_ia_components(
            components=[(samples, nonlinear)], components_ref=(samples, linear),
            residual=mode, **ia_options, **style,
        ), f"Original nonlinear minus linear IA choice: {mode}", ia_source)
    save("ia_parameter_colorbar", plots.plot_ia_components(
        components=[(samples, linear), (samples, nonlinear)],
        param=[0, 1], colorbarlabel="Source power mode (0=linear; 1=nonlinear)",
        rescale=1, **ia_options, **style,
    ), "Original IA kernel power-mode comparison", ia_source)
    save("comparison_identity", plots.plot_code_comparison(
        x=samples, reference=linear, ported=linear.copy(), labels=names,
        coordinate_label="Declared triangle index", quantity_label="IA kernel",
        residual="scaled", legend=["Saved reference copied unchanged"],
        figsize=(15, 8), show=None, **style,
    ), "Comparison-plot identity check; no C result is used", ia_source)

    analytic_path = project / "tests/data_vector/test_transform_physics.py"
    specification = importlib.util.spec_from_file_location(
        "threepcf_plot_analytic_reference", analytic_path
    )
    analytic = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(analytic)
    analytic_source = (
        "Exact Cartesian Gaussian moments for B=(ell1 ell2 ell3)^2 "
        "exp[-a sum ell_i^2]; a=0.5 or0.7 arcmin^2; no DES model or C calls."
    )
    theta = np.geomspace(0.3, 4, 23)
    angle = np.array([np.pi / 3])
    first = analytic.gaussian_centroid_components(theta=theta, phi=angle, scale=0.5)
    second = analytic.gaussian_centroid_components(theta=theta, phi=angle, scale=0.7)
    map_first = (theta, theta, first[:, :, :, 0][None])
    map_second = (theta, theta, second[:, :, :, 0][None])
    map_options = {
        "triplets": np.array([[0, 0, 0]]), "opening_angle_deg": 60,
        "projection": "analytic centroid", "show": None, "figsize": (7, 6),
        "xaxislabelsize": 20, "yaxislabelsize": 20, "bintextsize": 20,
    }
    for part in ("real", "imag"):
        component = 0 if part == "real" else 1
        save(f"gamma_map_{part}", plots.plot_Gamma(
            gamma=map_first, part=part, component=component, **map_options,
        ), f"Analytic Gaussian shear: {part} component{component}", analytic_source)
    for mode in ("percent", "scaled", "absolute"):
        save(f"gamma_map_{mode}", plots.plot_Gamma(
            gamma=map_second, gamma_ref=map_first, residual=mode, **map_options,
        ), f"Analytic Gaussian scale change: {mode}", analytic_source)
    save("gamma_map_sigma", plots.plot_Gamma(
        gamma=map_second, gamma_ref=map_first, residual="sigma",
        sigma=np.ones((1, theta.size, theta.size)), **map_options,
    ), "Sigma display check with illustrative unit errors; no survey errors",
        analytic_source + " Errors are illustrative ones, not survey predictions.")
    # A declared zero reference tests undefined-ratio rendering, rather than
    # assigning a numerical floor or presenting this as a physics residual.
    save("gamma_undefined", plots.plot_Gamma(
        gamma=map_first, gamma_ref=(theta, theta, np.zeros_like(map_first[2])),
        residual="percent", **map_options,
    ), "Display check: zero reference makes every ratio undefined", analytic_source)
    phi = np.linspace(0.15, 3, 64)
    sides = np.array([1.0, 1.7])
    first = analytic.gaussian_centroid_components(theta=sides, phi=phi, scale=0.5)
    second = analytic.gaussian_centroid_components(theta=sides, phi=phi, scale=0.7)
    slice_options = {
        "coordinate_label": r"$\phi$ [rad]",
        "slice_label": r"$\theta_1=1$, $\theta_2=1.7$ arcmin; analytic field",
        "projection": "centroid", "figsize": (18, 8), "show": None,
    }
    save("gamma_components", plots.plot_natural_components(
        gamma=[(phi, first[:, 0, 1]), (phi, second[:, 0, 1])],
        legend=[r"$a=0.5$ arcmin$^2$", r"$a=0.7$ arcmin$^2$"],
        **slice_options, **style,
    ), "Four complex natural shear components: analytic Gaussian reference",
        analytic_source)
    save("gamma_components_residual", plots.plot_natural_components(
        gamma=[(phi, second[:, 0, 1])], gamma_ref=(phi, first[:, 0, 1]),
        residual="absolute", **slice_options, **style,
    ), "Analytic Gaussian scale change: signed absolute differences", analytic_source)
    for mode in ("percent", "scaled", "sigma"):
        save(f"gamma_components_{mode}", plots.plot_natural_components(
            gamma=[(phi, second[:, 0, 1])], gamma_ref=(phi, first[:, 0, 1]),
            residual=mode, sigma=np.ones((8, phi.size)),
            **slice_options, **style,
        ), f"Analytic scale change: {mode}; ImGamma0 is roundoff",
            analytic_source + " Sigma mode uses illustrative unit errors only.")

    joint_path = checked_file(
        path=public / "covariance/xip_xim_map3_covariance_8ARCMINCUT_25Feb25.fits",
        expected="963bb0fcb42c00e9b7b47d4a87c99efc00edd688",
    )
    with fits.open(joint_path) as table:
        joint = table["COVMAT"].data.copy()
        header = table["COVMAT"].header
        if [header[f"STRT_{index}"] for index in range(3)] != [0, 200, 400]:
            raise ValueError("Public joint covariance ordering changed")
    block_options = {
        "block_sizes": [200, 200, 80],
        "block_labels": [r"$\xi_+$", r"$\xi_-$", r"$M_{\rm ap}^3$"],
        "show": None,
    }
    cov_source = "Pinned public joint Y3 sample covariance; no new analytic covariance."
    save("correlation", plots.plot_correlation(
        covariance=joint, labels=("Supplied covariance", "Reference"),
        figsize=(9, 8), **block_options,
    ), "Public joint Y3 correlation matrix", cov_source)
    save("correlation_identity", plots.plot_correlation(
        covariance=joint, covariance_ref=joint.copy(),
        labels=("Original supplied matrix", "Unchanged copy"),
        figsize=(10, 8), **block_options,
    ), "Matrix-triangle comparison: identity check", cov_source)
    c22 = np.zeros_like(joint)
    c22[:400, :400] = joint[:400, :400]
    c33 = np.zeros_like(joint)
    c33[400:, 400:] = joint[400:, 400:]
    c23 = joint - c22 - c33
    for normalization in ("diagonal", "element"):
        save(f"covariance_blocks_{normalization}", plots.plot_covariance_components(
            total=joint, components={"Two-point block": c22,
                                     "Three-point block": c33,
                                     "Cross blocks": c23},
            normalization=normalization, percent=True, figsize=(18, 9),
            **block_options,
        ), "Estimator blocks of the supplied matrix; not a physical-term split",
            cov_source)
    selected_covariance = covariance[:12, :12]
    for baseline in (None, selected_covariance):
        suffix = "absolute" if baseline is None else "identity"
        save(f"covariance_diagonal_{suffix}", plots.plot_covariance_diagonal(
            theta_arcmin=radius, covariances={"Supplied": selected_covariance},
            panel_labels=["(1,1,1)", "(1,1,2)", "(1,1,3)"],
            covariance_ref=baseline, coordinate_label=r"$R$ [arcmin]",
            figsize=(15, 5), show=None,
        ), "Diagonal errors of three supplied aperture triplets", public_source)
    for baseline in (None, np.array([1.0, 1.0, 1.0])):
        suffix = "normalized" if baseline is None else "ratios"
        save(f"parameter_errors_{suffix}", plots.plot_parameter_errors(
            names=["p1", "p2", "p3"], errors=[np.array([0.5, 0.8, 1.2])],
            errors_ref=baseline, legend=["Illustrative normalized errors"],
            show=None,
        ), "Plot convention only: illustrative errors, not a forecast",
            "Explicit illustrative dimensionless errors [0.5,0.8,1.2].")
    manifest = {"figures": records, "blas_threads": 1, "omp_threads": 1}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(records)} figures and their manifest to {output}")


if __name__ == "__main__":
    main()
