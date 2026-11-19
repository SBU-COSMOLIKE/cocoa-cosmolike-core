"""Build a private FFTW-instrumented extension and copy transform checks.

The supplied output directory receives its own source snapshot, extension,
project test copies, and build artifacts. The installed project and core
sources remain unchanged. Run the generated tests in a separate process;
its import directory selects the private extension without monkey patches.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--conda", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    arguments = parser.parse_args()
    here = Path(__file__).resolve().parent
    core = here.parents[1]  # The directory above dev_scripts/.
    project = arguments.root / "projects" / "des_3pcf"
    output = arguments.output.resolve()
    root = arguments.root.resolve()
    if output == root or root in output.parents:
        raise ValueError("Validation output must be outside the Cocoa tree")
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source"
    source.mkdir(exist_ok=True)
    hashes = {}
    for folder in ("cosmolike", "log.c"):
        shutil.copytree(src=core / folder, dst=source / folder,
                        dirs_exist_ok=True)
        for path in sorted((source / folder).rglob("*")):
            if path.is_file():
                relative = str(path.relative_to(source))
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "source_sha256.json").write_text(
        json.dumps(hashes, indent=2) + "\n")
    interface = output / "interface"
    interface.mkdir(exist_ok=True)
    makefile = project / "interface" / "MakefileCosmolike"
    shutil.copy2(src=makefile, dst=interface / makefile.name)
    entry = (project / "interface" / "interface.cpp").read_text()
    entry = '#include "trace_fftw.hpp"\n' + entry
    marker = "  cosmolike_interface::bind_threepcf(module);"
    if entry.count(marker) != 1:
        raise RuntimeError("Private registration marker must occur once")
    entry = entry.replace(marker, "  bind_fftw_trace(module);\n" + marker)
    (interface / "interface.cpp").write_text(entry)
    for name in ("trace_fftw.cpp", "trace_fftw.hpp"):
        shutil.copy2(src=here / name, dst=interface / name)
    extra = """
OBJECTS += $(BUILD)/trace_fftw.o
$(TARGET): $(BUILD)/trace_fftw.o
$(BUILD)/trace_fftw.o: trace_fftw.cpp | $(BUILD)
\t$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD)/fftlog_3pcf.o: CPPFLAGS += -Dfftw_malloc=trace_malloc
$(BUILD)/fftlog_3pcf.o: CPPFLAGS += -Dfftw_destroy_plan=trace_destroy
$(BUILD)/fftlog_3pcf.o: CPPFLAGS += -Dfftw_plan_dft_r2c_2d=trace_plan_r2c
$(BUILD)/fftlog_3pcf.o: CPPFLAGS += -Dfftw_plan_dft_c2r_2d=trace_plan_c2r
$(BUILD)/fftlog_3pcf.o: CPPFLAGS += -Dfftw_execute_dft_r2c=trace_execute_r2c
$(BUILD)/fftlog_3pcf.o: CPPFLAGS += -Dfftw_execute_dft_c2r=trace_execute_c2r
"""
    with (interface / makefile.name).open("a") as handle:
        handle.write(extra)
    tests = output / "tests" / "data_vector"
    tests.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src=project / "tests" / "conftest.py",
                 dst=tests.parent / "conftest.py")
    for path in sorted((project / "tests" / "data_vector").glob(
            "test_transform*.py")):
        shutil.copy2(src=path, dst=tests / path.name)
    shutil.copy2(src=here / "test_retained_plan.py",
                 dst=tests / "test_retained_plan.py")
    environment = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        environment[name] = "1"
    environment["OMP_NUM_THREADS"] = "6"
    environment["OMP_DYNAMIC"] = "FALSE"
    environment["CONDA_PREFIX"] = str(arguments.conda)
    command = [
        "make",
        "-f", makefile.name,
        "-j2",
        "all",
        "CORE=" + str(source),
        "ROOTDIR=" + str(arguments.root),
        "PYTHON=" + str(arguments.python),
        "C_COMPILER=" + str(arguments.conda / "bin" / "clang"),
        "CXX_COMPILER=" + str(arguments.conda / "bin" / "clang++"),
    ]
    if arguments.debug:
        command.append("COSMOLIKE_DEBUG_MODE=1")
    subprocess.run(args=command, cwd=interface, env=environment, check=True)


if __name__ == "__main__":
    main()
