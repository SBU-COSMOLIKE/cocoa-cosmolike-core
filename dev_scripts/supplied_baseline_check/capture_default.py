#!/usr/bin/env python3
"""Capture unchanged-route outputs for comparison across private builds.

This script evaluates small analytic inputs from the project's component
fixtures. It does not construct a cosmology, load DES observations, or time
an implementation. Run it in separate processes against the pinned old and
new extensions, then compare every saved array exactly. Use one and four
OpenMP threads in separate processes to check preparation ownership.

Arguments select the interface folder, pinned fixture folder, output NPZ,
and thread count. The fixture folder must contain test_physics_model.py.
All arrays are saved; a JSON companion records the binary and fixture hashes.
BLAS stays at one thread while OpenMP owns the requested worker count.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def capture(*, interface, fixtures, output, threads):
    """Save deterministic default-route predictions and their source identity.

    Arguments:
        interface: Directory containing cosmolike_des_3pcf_interface.so.
        fixtures: Directory containing the pinned toy physical-input fixture.
        output: Destination NPZ; its parent must already exist.
        threads: OpenMP workers, between one and six.

    Returns:
        Nothing. Writes the NPZ and a neighboring JSON identity record.

    Raises:
        ValueError for unsupported worker counts or non-unit BLAS settings.
        FileNotFoundError when the extension or fixture is absent.
    """
    if threads < 1 or threads > 6:
        raise ValueError("threads must be between one and six")
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(name, "1") != "1":
            raise ValueError(f"{name} must be one; assign workers to OpenMP")
        os.environ[name] = "1"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    binary = interface / "cosmolike_des_3pcf_interface.so"
    fixture = fixtures / "test_physics_model.py"
    for path in (binary, fixture):
        if not path.is_file():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(interface))
    sys.path.insert(0, str(fixtures))
    import numpy as np
    import cosmolike_des_3pcf_interface as compiled
    from test_physics_model import compact_grid_settings, physical_inputs

    compiled.set_omp_threads(nthreads=threads)
    sides = np.array([[0.03, 0.04, 0.05], [0.2, 0.3, 0.4]])
    redshifts = np.array([0.2, 1.3])
    angular = np.array([[120.0, 180.0, 250.0], [200.0, 270.0, 340.0]])
    bins = np.array([0, 1, 0], dtype=np.int32)
    parameters = {
        "a1": 0.4,
        "a2": 0.7,
        "bta": 0.9,
        "ct": 0.2,
    }
    results = {}
    for source_readers in (False, True):
        inputs = physical_inputs()
        inputs["source_readers"] = source_readers
        model = compiled.PhysicsModel(**inputs)
        prefix = f"source{int(source_readers)}"
        for component in range(4):
            results[f"{prefix}_matter{component}"] = model.matter(
                sides=sides, redshifts=redshifts, component=component
            )
        for mode in range(3):
            parameters["power_mode"] = mode
            results[f"{prefix}_power{mode}"] = model.power(
                k=sides[:, 0].copy(), z=redshifts, mode=mode
            )
            results[f"{prefix}_ia{mode}"] = model.ia(
                sides=sides, redshifts=redshifts, parameters=parameters
            )
        for component in range(3):
            results[f"{prefix}_projected{component}"] = model.projected(
                sides=angular, bins=bins, parameters=parameters, component=component
            )
        grid = model.projection_grid(
            bins=bins,
            parameters=parameters,
            settings=compact_grid_settings(reference_mode=0),
        )
        for component in range(3):
            results[f"{prefix}_grid{component}"] = grid.projected(
                sides=angular, component=component
            )
        for name, values in model.radial().items():
            results[f"{prefix}_radial_{name}"] = values
    np.savez(file=output, **results)
    identity = {
        "binary": str(binary.resolve()),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "fixture": str(fixture.resolve()),
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "threads": threads,
        "blas_threads": 1,
        "array_count": len(results),
    }
    output.with_suffix(".json").write_text(
        json.dumps(obj=identity, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(results)} arrays to {output}")


def main():
    """Parse explicit paths and run one deterministic capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    arguments = parser.parse_args()
    capture(
        interface=arguments.interface,
        fixtures=arguments.fixtures,
        output=arguments.output,
        threads=arguments.threads,
    )


if __name__ == "__main__":
    main()
