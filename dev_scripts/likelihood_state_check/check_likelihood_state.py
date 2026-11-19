"""Compare LSST Y1 likelihood results across processes with fixed inputs.

The Gaussian statistic is (theory - data)^T C^-1 (theory - data).
Identical theory vectors need not give identical statistics if a reference
data file changes between runs. This check copies the frozen inputs before
starting workers, verifies their manifest, and records their SHA-256 hashes.
Each worker compares the likelihood statistic with the statistic obtained
by passing the returned theory vector to the same compiled calculation.

Run from a started Cocoa environment, with an unused output directory:

    python dev_scripts/likelihood_state_check/check_likelihood_state.py \
        --build /absolute/path/to/lsst/interface --output /tmp/state_check

Ten fresh processes alternate one and four OpenMP threads and two case
orders. BLAS always uses one thread. The binary and the frozen-input
snapshot must retain the same bytes throughout the check. No reference
value is regenerated, and the project is never an output destination.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def file_hash(path):
    """Return the SHA-256 of a file, reading at most one MiB at a time.

    Arguments:
        path = pathlib.Path of the input file.
    Returns:
        Hexadecimal digest string identifying the file's bytes.
    """
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            result.update(block)
    return result.hexdigest()


def verify_snapshot(tests):
    """Check every frozen file against the copied manifest; return its hashes.

    Arguments:
        tests = private directory containing frozen/ and manifest_sha256.json.
    Returns:
        Dictionary mapping relative filenames to SHA-256 strings.
    Raises:
        RuntimeError if any file differs from the recorded input version.
    """
    manifest = json.loads((tests / "manifest_sha256.json").read_text())
    hashes = manifest["files"]
    for name, expected in hashes.items():
        actual = file_hash(tests / name)
        if actual != expected:
            raise RuntimeError(
                f"Input {name} has SHA-256 {actual}, expected {expected}. "
                "Repeat the check from a consistent frozen-input snapshot.")
    return hashes


def copy_inputs(project, destination):
    """Copy and verify the project's frozen files without following later edits.

    Arguments:
        project = project path containing tests/frozen and its manifest.
        destination = unused directory that will hold the private tests/ copy.
    Returns:
        The verified snapshot's file hashes.
    """
    source = project / "tests"
    destination.mkdir(parents=True)
    # The manifest is copied first. A refreeze during the subsequent copy
    # causes a hash failure instead of silently mixing two input versions.
    for name in ("manifest_sha256.json", "cocoa_test_utils.py"):
        shutil.copy2(source / name, destination / name)
    shutil.copytree(source / "frozen", destination / "frozen",
                    ignore=shutil.ignore_patterns("__pycache__"))
    return verify_snapshot(tests=destination)


def run_worker(arguments, cocoa, output):
    """Evaluate four cases in one process and save each statistic and vector.

    Arguments:
        arguments = parsed command-line options, including worker number.
        cocoa = root directory of the Cocoa installation.
        output = parent output directory containing the private inputs.
    Returns:
        None; writes one JSON record and numerical arrays for this worker.
    Raises:
        AssertionError if a likelihood statistic differs from its vector's
        statistic, or if a repeated evaluation changes either result.
    """
    core = Path(__file__).resolve().parents[2]
    tests = output / "inputs" / "tests"
    sys.path.insert(0, str(core))
    sys.path.insert(0, str(tests))
    sys.path.insert(0, str(arguments.build))
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits
    import cosmolike_lsst_y1_interface as native
    import cocoa_test_utils as helpers

    assert Path(native.__file__).parent.resolve() == arguments.build
    expected = json.loads((output / "inputs.json").read_text())
    assert file_hash(Path(native.__file__)) == expected["binary_sha256"]
    assert verify_snapshot(tests=tests) == expected["files"]
    threadpool_limits(limits=1, user_api="blas")
    native.set_omp_threads(arguments.threads)
    os.chdir(cocoa)

    cases = [
        ("example1", False, False),
        ("example1", True, False),
        ("example2", False, False),
        ("example2", True, True),
    ]
    # The unmasked TATT case is last in one order and first in the other.
    # This separates the previous model's state from the requested model.
    if arguments.worker % 2:
        cases.reverse()
    records = []
    folder = output / f"worker_{arguments.worker:02d}"
    folder.mkdir()
    for example, tatt, all_scales in cases:
        tag = example + ("_tatt" if tatt else "_nla")
        if all_scales:
            tag += "_unmasked"
        info = helpers.load_frozen_info(example=example, tatt=tatt)
        name = helpers.EXAMPLES[example]["likelihood"]
        if all_scales:
            info["likelihood"][name]["data_file"] = "tatt_lsst_y1_ones.dataset"
        model = helpers.make_model(info=info)
        point = helpers.build_point(model=model, example=example, tatt=tatt)
        repeats = 2 if all_scales else 1
        previous = None
        for repeat in range(repeats):
            chi2 = helpers.evaluate_chi2(model=model, point=point)
            vector = np.asarray(native.compute_data_vector_masked()).copy()
            vector_chi2 = float(native.compute_chi2(vector))
            assert np.isfinite(vector).all(), tag
            assert chi2.hex() == vector_chi2.hex(), (tag, chi2, vector_chi2)
            digest = hashlib.sha256(vector.tobytes()).hexdigest()
            result = {
                "case": tag,
                "repeat": repeat,
                "chi2": chi2,
                "chi2_hex": chi2.hex(),
                "vector_chi2": vector_chi2,
                "vector_sha256": digest,
            }
            if previous is not None:
                assert previous["chi2_hex"] == result["chi2_hex"], tag
                assert previous["vector_sha256"] == digest, tag
            previous = result
            records.append(result)
            np.save(folder / f"{tag}_{repeat}.npy", vector)
        model.close()
    for pool in threadpool_info():
        if pool["user_api"] == "blas":
            assert pool["num_threads"] == 1, pool
    assert verify_snapshot(tests=tests) == expected["files"]
    assert file_hash(Path(native.__file__)) == expected["binary_sha256"]
    (folder / "results.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    """Parse paths, create an input snapshot, and check fresh-process agreement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, choices=(1, 4), default=1,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runs < 4:
        parser.error("--runs must be at least 4 to test both orders and threads")
    cocoa = Path(os.environ["ROOTDIR"]).resolve()
    args.build = args.build.resolve()
    output = args.output.resolve()
    if args.worker is not None:
        run_worker(arguments=args, cocoa=cocoa, output=output)
        return
    binary = args.build / "cosmolike_lsst_y1_interface.so"
    if not binary.is_file():
        parser.error(f"Build does not contain {binary.name}: {args.build}")
    project = cocoa / "projects" / "lsst_y1"
    if output == project or project in output.parents:
        parser.error("--output must be outside the LSST Y1 project")
    output.mkdir(parents=True, exist_ok=False)
    hashes = copy_inputs(project=project,
                         destination=output / "inputs" / "tests")
    record = {"files": hashes, "binary_sha256": file_hash(binary)}
    (output / "inputs.json").write_text(json.dumps(record, indent=2) + "\n")
    baseline = {}
    reports = []
    for worker in range(args.runs):
        # Pairs of orders at one thread, then the same pair at four threads.
        threads = 1 if (worker // 2) % 2 == 0 else 4
        environment = dict(os.environ)
        blas_variables = (
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
        for key in blas_variables:
            environment[key] = "1"
        environment["OMP_NUM_THREADS"] = str(threads)
        environment["OMP_THREAD_LIMIT"] = "4"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--build", str(args.build),
            "--output", str(output),
            "--worker", str(worker),
            "--threads", str(threads),
        ]
        with (output / f"worker_{worker:02d}.log").open("w") as stream:
            subprocess.run(args=command, cwd=cocoa, env=environment,
                           stdout=stream, stderr=subprocess.STDOUT,
                           check=True)
        results = json.loads(
            (output / f"worker_{worker:02d}" / "results.json").read_text())
        for result in results:
            case = result["case"]
            signature = (result["chi2_hex"], result["vector_sha256"])
            if case in baseline:
                assert baseline[case] == signature, (worker, case, signature)
            else:
                baseline[case] = signature
        reports.append({
            "worker": worker,
            "threads": threads,
            "reversed_order": bool(worker % 2),
            "passed": True,
        })
        (output / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(f"Process {worker + 1}/{args.runs}: exact agreement", flush=True)


if __name__ == "__main__":
    main()
