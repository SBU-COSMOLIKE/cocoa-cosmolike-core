"""Freeze small bridge inputs and compare numerical outputs across builds.

Run freeze once, then run capture in separate processes against the original
and changed extensions. The input NPZ contains every physical array, scalar,
query coordinate and response table consumed below. No survey data,
covariance or mask is used: these are controlled raw-observable comparisons.
The copied test fixtures define the inputs; capture always reloads the saved
arrays. Model-generation records are separate because reuse should change
those records while leaving all numerical output arrays unchanged.
The raw IA and projected three-point outputs retain each state's TATT
coefficients. The two-point read uses a separate NLA-only parameter dictionary,
with a2, bta, ct and the legacy fb set to zero, as that API requires.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(*, path):
    """This helper identifies the complete bytes of a saved file.

    A filename can stay unchanged after an input is edited. Recording a hash
    lets the later comparison establish which bytes each process actually read.

    Arguments:
        path = existing file to identify.
    Returns:
        Hexadecimal SHA256 string.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(*, project, threads):
    """This helper selects copied project sources and bounded worker counts.

    BLAS libraries each receive one worker; OpenMP receives the requested
    workers. Setting this before importing NumPy avoids nested thread pools.

    Arguments:
        project = project tree holding the copied Python code and test inputs.
        threads = OpenMP workers, from one to six.
    Returns:
        None. Updates this process's environment and Python import search path.
    """
    for name in (
        "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OMP_THREAD_LIMIT"] = "6"
    sys.path.insert(0, str(project / "python"))
    sys.path.insert(0, str(project / "tests"))
    # The original fixture modules import neighboring test helpers by name.
    sys.path.insert(0, str(project / "tests/data_vector"))


def freeze(*, project, output):
    """This function saves controlled inputs before comparing two builds.

    The fixture functions provide small numerical examples. Saving their
    complete results once means later processes compare the same physical
    arrays, rather than independently generating apparently similar inputs.

    Arguments:
        project = copied project tree defining the input examples.
        output = destination NPZ archive; its neighboring JSON records hashes.
    Returns:
        None. Writes the input archive and its identification record.
    """
    environment(project=project, threads=1)
    import numpy as np
    from data_vector.test_feedback import paired_input
    from data_vector.test_physics_model import physical_inputs
    from test_supplied_baseline import baseline_input

    arrays = {}
    for name, value in physical_inputs().items():
        arrays["physical__" + name] = np.asarray(value)
    for name, value in baseline_input(varying=False).items():
        arrays["baseline__" + name] = np.asarray(value)
    for name, value in paired_input().native.items():
        arrays["feedback__" + name] = np.asarray(value)
    arrays["omega_m_z"] = np.full(shape=arrays["physical__z"].shape,
                                 fill_value=0.3)
    arrays["angular"] = np.array([[120.0, 180.0, 250.0]])
    arrays["sides"] = np.array([[0.3, 0.4, 0.5]])
    arrays["redshifts"] = np.array([0.4])
    arrays["ell"] = np.array([120.0, 250.0])
    arrays["triplet"] = np.array([0, 1, 0], dtype=np.int32)
    arrays["pair"] = np.array([0, 1], dtype=np.int32)
    arrays["A1"] = np.array([0.4, -0.2])
    arrays["A2"] = np.array([0.7, 0.3])
    arrays["BTA"] = np.array([0.9, 0.0])
    arrays["ct"] = np.asarray(0.2)
    controls = {
        "bridge_omega_m": 0.3,
        "bridge_omega_b": 0.05,
        "bridge_hubble": 70.0,
        "bridge_omega_nu_h2": 0.0,
        "kernel_nz": 101,
        "source_readers": 1,
        "table_boost": 8,
        "fit_nz": 33,
        "ia_model": 1,
        "ia_redshift_evolution": 3,
        "changed_a1_increment": 0.15,
        "changed_feedback_amplitude": 0.7,
        "changed_power_mode": 2,
        "default_power_mode": 1,
        "ct_mode": 0,
        "suppress_only": 0,
        "nla_zero_fields": ["a2", "bta", "ct", "fb"],
        "nla_zero_value": 0.0,
    }
    arrays["controls_json"] = np.asarray(json.dumps(controls, sort_keys=True))
    # Keyword expansion writes one named NPZ member per input dictionary key.
    np.savez_compressed(file=output, **arrays)
    record = {"input_sha256": digest(path=output), "fixture_sha256": {}}
    for name in (
        "tests/data_vector/test_physics_model.py",
        "tests/data_vector/test_feedback.py", "tests/test_supplied_baseline.py",
    ):
        record["fixture_sha256"][name] = digest(path=project / name)
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")


def capture(*, project, interface, inputs, output, threads):
    """This function evaluates each state from the previously saved inputs.

    Each process imports one explicitly selected extension. The model build
    counts may change when table reuse improves, but the numerical arrays must
    retain the same float64 bits at every input state.

    Arguments:
        project = copied project Python/test tree.
        interface = directory containing the selected compiled extension.
        inputs = previously saved NPZ containing all physical inputs.
        output = output NPZ; a neighboring JSON records identities and controls.
        threads = OpenMP workers, from one to six; BLAS always uses one.
    Returns:
        None. Writes arrays and state records, then clears process-global core
        input after each internal/supplied-bispectrum sequence.
    """
    environment(project=project, threads=threads)
    sys.path.insert(0, str(interface))
    import numpy as np
    import cosmolike_des_3pcf_interface as compiled
    from test_supplied_baseline import install_bridge

    compiled.set_omp_threads(nthreads=threads)
    # The context manager closes the archive even if reading one member fails.
    with np.load(file=inputs, allow_pickle=False) as archive:
        frozen = {}
        for name in archive.files:
            frozen[name] = np.array(archive[name], copy=True)
    dictionaries = {}
    for prefix in ("physical", "baseline", "feedback"):
        values = {}
        for name, value in frozen.items():
            if name.startswith(prefix + "__"):
                key = name[len(prefix) + 2:]
                if value.ndim == 0:
                    values[key] = value.item()
                else:
                    values[key] = value
        dictionaries[prefix] = values
    results = {}
    records = []
    controls = json.loads(frozen["controls_json"].item())
    core = compiled.core
    phases = ("initial", "changed_ia", "restored_ia", "amplitude",
              "prescription", "restored")
    for supplied in (False, True):
        install_bridge(compiled=compiled, inputs=dictionaries["physical"],
                       install_scalars=True)
        core.set_cosmological_parameters(
            omega_matter=controls["bridge_omega_m"],
            omega_baryon=controls["bridge_omega_b"],
            hubble=controls["bridge_hubble"],
            omega_nu_h2=controls["bridge_omega_nu_h2"],
        )
        core.set_resolution(
            kernel_nz=controls["kernel_nz"],
            source_readers=controls["source_readers"],
            table_boost=controls["table_boost"], fit_nz=controls["fit_nz"],
        )
        if supplied:
            core.set_baseline_bispectrum(
                baseline=dictionaries["baseline"],
                omega_m_z=frozen["omega_m_z"],
                sigma8=dictionaries["physical"]["sigma8"],
            )
        generations = []
        parameters_record = []
        for phase in phases:
            first = frozen["A1"].copy()
            if phase == "changed_ia":
                first[0] += controls["changed_a1_increment"]
            amplitude = 0.0
            if phase == "amplitude":
                amplitude = controls["changed_feedback_amplitude"]
            power_mode = controls["default_power_mode"]
            if phase == "prescription":
                power_mode = controls["changed_power_mode"]
            core.init_IA(ia_model=controls["ia_model"],
                         ia_redshift_evolution=controls["ia_redshift_evolution"])
            core.set_nuisance_ia(A1=first, A2=frozen["A2"], B_TA=frozen["BTA"])
            core.set_nuisance_ia_ct(ct=float(frozen["ct"]))
            core.set_prescriptions(power_mode=power_mode,
                                   ct_mode=controls["ct_mode"],
                                   suppress_only=controls["suppress_only"])
            core.set_feedback(feedback=dictionaries["feedback"],
                              amplitude=amplitude)
            model = core.model_from_core()
            parameters = core.ia_parameters_from_core()
            prefix = f"supplied{int(supplied)}__{phase}__"
            results[prefix + "matter"] = model.matter(
                sides=frozen["sides"], redshifts=frozen["redshifts"],
            )
            results[prefix + "ia"] = model.ia(
                sides=frozen["sides"], redshifts=frozen["redshifts"],
                parameters=parameters,
            )
            # The two-point NLA read deliberately excludes quadratic IA and
            # the bispectrum-only legacy modifier; the other outputs use TATT.
            nla_parameters = dict(parameters)
            for name in controls["nla_zero_fields"]:
                nla_parameters[name] = controls["nla_zero_value"]
            results[prefix + "nla"] = model.nla_power(
                ell=frozen["ell"], bins=frozen["pair"],
                parameters=nla_parameters,
            )
            for component in range(3):
                results[prefix + f"projected{component}"] = model.projected(
                    sides=frozen["angular"], bins=frozen["triplet"],
                    parameters=parameters, component=component,
                )
            generations.append(core.model_generation())
            parameters_record.append(parameters)
        records.append({
            "supplied": supplied,
            "phases": phases,
            "generations": generations,
            "ia": parameters_record,
        })
        core.initial_setup()
    # Every result dictionary key becomes a named NPZ member in this archive.
    np.savez_compressed(file=output, **results)
    binary = Path(compiled.__file__).resolve()
    record = {
        "binary": str(binary),
        "binary_sha256": digest(path=binary),
        "inputs_sha256": digest(path=inputs),
        "capture_source_sha256": digest(path=Path(__file__)),
        "threads": threads,
        "array_count": len(results),
        "controls": controls,
        "survey_data_covariance_mask": "none; raw-observable comparison",
        "states": records,
    }
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {len(results)} bridge arrays to {output}", flush=True)


def main():
    """This entry point selects input saving or a single-build evaluation.

    Separate invocations ensure the old and new extensions never coexist in
    one Python process. Explicit paths identify every saved artifact.

    Arguments:
        Command-line mode and paths described by --help.
    Returns:
        None. Runs the selected stage and writes its requested output files.
    Raises:
        SystemExit with a parser error if capture lacks its input/build paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "capture"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interface", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--threads", type=int, choices=range(1, 7), default=1)
    options = parser.parse_args()
    if options.mode == "freeze":
        freeze(project=options.project, output=options.output)
    else:
        if options.interface is None or options.inputs is None:
            parser.error("capture requires --interface and --inputs")
        capture(project=options.project, interface=options.interface,
                inputs=options.inputs, output=options.output,
                threads=options.threads)


if __name__ == "__main__":
    main()
