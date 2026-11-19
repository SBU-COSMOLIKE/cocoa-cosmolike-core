#!/usr/bin/env python3
"""Build and check the actual core allocators without a survey likelihood.

The reference revision supplies only basics.c. Both builds use the current
declarations, structs and logging code. Positive-shape output records logical
row offsets and must match byte for byte. Empty shapes are checked only in
the candidate because the reference has no portable non-NULL contract for
zero-byte requests. No runtime function or module is replaced.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    """Return the SHA-256 of a source, executable or saved test output."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_command(command, directory, environment, output):
    """Run one command and retain combined output; abort on failure."""
    with output.open(mode="w") as stream:
        subprocess.run(
            args=command,
            cwd=directory,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main():
    """Compile strict, UBSan and ASan checks and compare their layouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    core = here.parent.parent
    required = ["ROOTDIR", "CONDA_PREFIX", "C_COMPILER"]
    for name in required:
        if not os.environ.get(name):
            raise RuntimeError(
                f"{name} is unset; activate the Cocoa environment and source "
                "start_cocoa.sh before running this check."
            )
    cocoa = Path(os.environ["ROOTDIR"])
    conda = Path(os.environ["CONDA_PREFIX"])
    compiler = os.environ["C_COMPILER"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reference = output / "reference_basics.c"
    reference.write_bytes(subprocess.check_output(
        args=["git", "show", f"{args.reference}:cosmolike/basics.c"],
        cwd=core,
    ))
    environment = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    environment["OMP_NUM_THREADS"] = "6"
    environment["OMP_THREAD_LIMIT"] = "6"
    environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    environment["ASAN_OPTIONS"] = "halt_on_error=1:detect_leaks=0"
    includes = [core, core / "cosmolike", cocoa / ".local/include",
                conda / "include", cocoa / "external_modules/code/simde"]
    flags = ["-std=gnu99", "-fopenmp", "-fno-fast-math",
             "-fno-associative-math", "-frounding-math", "-ftrapping-math"]
    for directory in includes:
        flags.append("-I" + str(directory))
    libraries = []
    for directory in (cocoa / ".local/lib", conda / "lib"):
        libraries.extend(["-L" + str(directory), "-Wl,-rpath," + str(directory)])
    libraries.extend(["-lgsl", "-lgslcblas", "-lm"])
    common_sources = [core / "cosmolike/structs.c", core / "log.c/src/log.c",
                      here / "check_allocators.c"]
    modes = {
        "strict": ["-O3", "-march=native"],
        "debug": ["-O0", "-g", "-fsanitize=undefined,float-divide-by-zero",
                  "-fno-sanitize-recover=all"],
        "address": ["-O1", "-g", "-fsanitize=address,undefined",
                    "-fno-omit-frame-pointer", "-fno-sanitize-recover=all"],
    }
    record = {
        "reference": args.reference,
        "core_head": subprocess.check_output(
            args=["git", "rev-parse", "HEAD"], cwd=core, text=True,
        ).strip(),
        "compiler": compiler,
        "blas_threads": 1,
        "source_hashes": {},
        "modes": {},
        "empty_reference_scope": (
            "Empty reference allocations are not required to fail on this "
            "platform; POSIX permits either NULL or a freeable pointer for "
            "a zero-byte request. Candidate checks require a non-NULL owner."
        ),
    }
    sources = common_sources + [core / "cosmolike/basics.c",
                                core / "cosmolike/basics.h", reference]
    for source in sources:
        record["source_hashes"][str(source)] = digest(path=source)
    result_path = output / "results.json"
    result_path.write_text(json.dumps(record, indent=2) + "\n")
    for mode, mode_flags in modes.items():
        directory = output / mode
        directory.mkdir()
        mode_record = {"builds": {}, "runs": {}}
        record["modes"][mode] = mode_record
        for label, source in (("reference", reference),
                              ("candidate", core / "cosmolike/basics.c")):
            executable = directory / label
            command = [compiler] + flags + mode_flags
            command.extend([str(source)] + [str(path) for path in common_sources])
            command.extend(libraries + ["-o", str(executable)])
            log = directory / (label + "_build.log")
            run_command(command=command, directory=core,
                        environment=environment, output=log)
            mode_record["builds"][label] = {
                "command": command, "executable_sha256": digest(path=executable),
                "log_sha256": digest(path=log),
            }
        for label, positive, workers in (
                ("reference", True, 1), ("candidate", True, 1),
                ("candidate", True, 6), ("candidate", False, 1),
                ("candidate", False, 6)):
            name = f"{label}_{'positive' if positive else 'all'}_{workers}"
            command = [str(directory / label)]
            if positive:
                command.append("--positive")
            environment["OMP_NUM_THREADS"] = str(workers)
            log = directory / (name + ".log")
            run_command(command=command, directory=core,
                        environment=environment, output=log)
            text = log.read_text()
            if not text.endswith("allocator checks passed\n"):
                raise RuntimeError(f"Allocator check did not complete: {log}")
            mode_record["runs"][name] = {
                "command": command, "omp_threads": workers,
                "output_sha256": digest(path=log),
            }
        old = (directory / "reference_positive_1.log").read_bytes()
        for name in ("candidate_positive_1.log", "candidate_positive_6.log"):
            if old != (directory / name).read_bytes():
                raise RuntimeError(f"Positive allocation layout differs: {name}")
        if ((directory / "candidate_all_1.log").read_bytes()
                != (directory / "candidate_all_6.log").read_bytes()):
            raise RuntimeError(f"Allocator output depends on worker count: {mode}")
        mode_record["positive_layout_equal"] = True
        mode_record["empty_and_positive_worker_outputs_equal"] = True
        result_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{mode}: five runs passed; positive layouts match; workers 1/6 agree.")


if __name__ == "__main__":
    main()
