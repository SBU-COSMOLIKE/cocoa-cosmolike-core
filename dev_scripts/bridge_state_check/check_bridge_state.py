#!/usr/bin/env python3
"""Check the state transitions of the three-point core bridge.

What is being checked. cosmolike/3pcf/threepcf_core_bridge.c fills the
global Cosmolike structs through its own installers (set_source_sample,
init_IA, set_nuisance_ia, ...) and builds the immutable three-point model
from them. A notebook or a likelihood calls those installers many times in
one process, so the bridge must stay correct across every change of state,
not only after a fresh setup. Two transitions failed before the fix that
this check guards (red-team review of 2026-11-18):

  - IA mode switch. The registry slot nuisance.ia[1][2] is the A2
    amplitude of source bin 2 under IA_REDSHIFT_BINNING, but the
    velocity-shear amplitude C_t under IA_REDSHIFT_EVOLUTION. The
    power-law setter writes only entries 0 and 1 of the A2 row, so an old
    binned A2 of bin 2 survived a switch to the power law and was read as
    C_t (A2 = [0.7, 0.8, 1.7] gave ct = 1.7; a fresh setup gives 0.0). The
    fix resets every IA amplitude on a mode change, and the C_t setter
    refuses every mode other than the power law.
  - Source-table size. set_source_sample_size changes the declared bin
    count without touching the stored n(z) table, which has one row per
    bin plus the redshift row. Building the model before a matching table
    arrived read the row past the end of the allocation (signal 11). The
    fix refuses the build through log_fatal until the matching table is
    installed.

How the check works, stage by stage:
  1. validate the repository layout, the des_3pcf build files, the cocoa
     environment and the reference revision;
  2. build two des_3pcf extensions out of tree, with the project's
     MakefileCosmolike, into a scratch folder: the LIVE build from the
     working tree, and a REFERENCE build that is the same working tree
     with the two bridge files (threepcf_core_bridge.c and .h) taken from
     the reference revision, the last commit before the fix;
  3. binned to power law: a three-bin binned install with
     A2 = [0.7, 0.8, 1.7] and zero B_TA, then a switch to the power law
     and its amplitudes. C_t must read 0.0, and the IA dictionary and the
     predicted b_L multipoles must equal those of a fresh process that
     only ever installed the power law, bitwise;
  4. round trip: the power law with an explicitly installed
     C_t = 0.25 (read back before the switch, to prove it was set), then
     binned (nonzero bin-2 A2) and back to the power law, reinstalling
     the amplitudes at each step as a caller would. C_t must read 0.0
     until the caller installs it again, because a mode change resets
     every IA amplitude, an explicit C_t included; the IA dictionary and
     b_L must equal the fresh process bitwise. A retained C_t would not
     be harmless: at the default ct_mode = 0 the velocity-shear term is
     C_t times C_1(z), not switched off;
  5. C_t refusal: the C_t setter called under the binned mode must stop
     the child through log_fatal with exit code 1, within a timeout;
  6. source-table size: after a two-bin install, declaring three bins
     and requesting the model must exit 1 with the bridge's diagnostic,
     never on a signal; the same holds for shrinking from three bins to
     two. Then, in one process, 2 -> 3 -> 2 bins each with a matching
     reinstall must build every time, and the final two-bin model must
     reproduce the first one bitwise;
  7. control: one consistent sequence (a correct power-law setup, a
     model, a C_t change and a rebuild) gives bitwise-identical IA
     dictionaries, b_L multipoles and radial tables in the live and the
     reference build. The fix must be invisible to every consistent
     sequence.

Every sequence runs in its own child process (this script calls itself
with --child), so a crash or a log_fatal exit is observed from outside,
and two sequences never share struct state. Every child's output must be
free of sanitizer reports, so a --debug run (both extensions built with
COSMOLIKE_DEBUG_MODE=1, which adds UBSan) checks the same paths for
undefined behavior.

The reference revision. Stage 7 compares against the bridge files of
REFERENCE_REVISION and keeps every other source live, so only the bridge
differs between the two builds. Once the bridge changes for another
reason, pass --reference with a revision whose bridge differs from the
live one only by changes meant to be invisible.

How to run it (from the Cocoa/ folder, cocoa environment active, so the
compiler, pybind11 and the gsl/fftw headers of the environment are
found):

    conda activate cocoapy311
    source start_cocoa.sh
    python ./external_modules/code/cosmolike_core/dev_scripts/\\
bridge_state_check/check_bridge_state.py

Optional arguments: --debug builds both extensions with UBSan;
--threads N sets the OpenMP threads of the children (default 4);
--reference REV replaces the reference revision.

Expected output: one PASS line per stage and a final line
"bridge state check: PASS" with exit code 0. On any failure an
AssertionError names the sequence and the disagreeing values, and the
exit code is nonzero. The builds land in a scratch_* folder created next
to this script and are removed before the script returns, pass or fail.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

# The last commit before the fix (the parent of dbb0a0f, "Reset the IA
# registry on a mode change and guard the source table"). Stage 7 builds
# its bridge files.
REFERENCE_REVISION = "c13325993bb51b8687c21651263e05aadfd128a8"

# The two files of the fix that the extension compiles (the third file of
# the fix, generic_interface.cpp, is not part of the des_3pcf build).
BRIDGE_FILES = (
    os.path.join("cosmolike", "3pcf", "threepcf_core_bridge.c"),
    os.path.join("cosmolike", "3pcf", "threepcf_core_bridge.h"),
)

# Name of the compiled module; the des_3pcf makefile writes it into the
# folder make runs in.
MODULE = "cosmolike_des_3pcf_interface"

# A child builds at most a few small models; seconds suffice. The timeout
# only exists so a hanging child fails the check instead of stalling it.
CHILD_TIMEOUT_SECONDS = 300

# A child prints its result as one JSON line that starts with this tag.
RESULT_TAG = "RESULT "

# The diagnostics the two refusals print through log_fatal.
CT_DIAGNOSTIC = "C_t needs nuisance.IA = IA_REDSHIFT_EVOLUTION"
TABLE_DIAGNOSTIC = "the source table was installed for {} bins but {} are declared"

# Text that UBSan prints for every undefined-behavior report.
SANITIZER_TEXT = "runtime error:"

# IA registry flags of IA.h, as the bridge's init_IA receives them.
IA_MODEL_TATT = 1
IA_REDSHIFT_BINNING = 2
IA_REDSHIFT_EVOLUTION = 3

# The final power-law amplitudes every IA sequence ends with: entry 0 is
# the amplitude, entry 1 the redshift exponent, and B_TA uses entry 0
# only. The arrays have three entries because the setter needs at least
# one per declared source bin.
POWER_LAW = {
    "A1": (0.8, 0.35, 0.0),
    "A2": (-1.1, -0.65, 0.0),
    "B_TA": (0.9, 0.0, 0.0),
}

# A different earlier power law for the round trip of stage 4.
EARLY_POWER_LAW = {
    "A1": (1.2, 0.1, 0.0),
    "A2": (0.4, 0.2, 0.0),
    "B_TA": (0.5, 0.0, 0.0),
}

# The binned amplitudes of the red-team reproduction: one entry per source
# bin. The bin-2 A2 value 1.7 is the one that leaked into C_t.
BINNED = {
    "A1": (0.3, 0.4, 0.5),
    "A2": (0.7, 0.8, 1.7),
    "B_TA": (0.0, 0.0, 0.0),
}


def locate_repository():
    """Resolve the cosmolike_core and des_3pcf paths and validate them.

    The script lives in cosmolike_core/dev_scripts/bridge_state_check/,
    so the repository root is two levels up and the Cocoa root three more
    (Cocoa/external_modules/code/cosmolike_core). Every file the build
    needs is checked here, so a moved file fails with its path instead of
    as a make error.

    Returns:
      a dict with the repository root ("core"), the Cocoa root ("cocoa"),
      the des_3pcf makefile ("makefile") and registration file
      ("interface").

    Raises:
      AssertionError naming the first missing file.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    core = os.path.dirname(os.path.dirname(here))
    cocoa = os.path.dirname(os.path.dirname(os.path.dirname(core)))
    project = os.path.join(cocoa, "projects", "des_3pcf", "interface")
    paths = {
        "core": core,
        "cocoa": cocoa,
        "makefile": os.path.join(project, "MakefileCosmolike"),
        "interface": os.path.join(project, "interface.cpp"),
    }
    required = [paths["makefile"], paths["interface"]]
    for name in BRIDGE_FILES:
        required.append(os.path.join(core, name))
    for path in required:
        assert os.path.isfile(path), (
            f"missing file for the extension build: {path}; run this "
            "script from a Cocoa checkout that has the des_3pcf project")
    return paths


def locate_toolchain():
    """Check the cocoa environment the des_3pcf makefile builds with.

    The makefile takes its include and library paths from CONDA_PREFIX
    and asks the Python interpreter for the pybind11 and numpy headers,
    so the environment must be active and this interpreter must provide
    both packages.

    Returns:
      a dict with "make" (absolute path of make) and "python" (this
      interpreter, handed to the makefile so the headers match it).

    Raises:
      AssertionError with the activation commands when something is
      missing.
    """
    prefix = os.environ.get("CONDA_PREFIX", "")
    assert prefix != "", (
        "CONDA_PREFIX is not set: activate the cocoa environment first "
        "(conda activate cocoapy311; source start_cocoa.sh)")
    make = shutil.which("make")
    assert make is not None, "no make found on PATH"
    for package in ("numpy", "pybind11"):
        probe = subprocess.run(
            [sys.executable, "-c", f"import {package}"],
            capture_output=True, text=True)
        assert probe.returncode == 0, (
            f"this interpreter ({sys.executable}) cannot import {package}; "
            "activate the cocoa environment first")
    return {"make": make, "python": sys.executable}


def validate_reference(paths, revision):
    """Check that the reference revision holds both bridge files.

    Arguments:
      paths = the dict from locate_repository
      revision = a git revision of cosmolike_core

    Returns:
      nothing.

    Raises:
      AssertionError when git cannot read a bridge file at the revision.
    """
    for name in BRIDGE_FILES:
        probe = subprocess.run(
            ["git", "-C", paths["core"], "cat-file", "-e",
             f"{revision}:{name}"],
            capture_output=True, text=True)
        assert probe.returncode == 0, (
            f"git cannot read {name} at revision {revision}: "
            f"{probe.stderr.strip()}")


def prepare_reference_tree(paths, revision, scratch):
    """Write the reference source tree: live sources, reference bridge.

    The des_3pcf makefile reads cosmolike/ and log.c/ under its CORE
    folder, so both are copied from the working tree, and then the two
    bridge files are replaced by their content at the reference revision
    (git show prints a file as stored in that commit).

    Arguments:
      paths = the dict from locate_repository
      revision = the reference revision
      scratch = existing scratch folder

    Returns:
      absolute path of the reference CORE folder.
    """
    tree = os.path.join(scratch, "reference_core")
    for folder in ("cosmolike", "log.c"):
        shutil.copytree(os.path.join(paths["core"], folder),
                        os.path.join(tree, folder),
                        ignore=shutil.ignore_patterns("__pycache__"))
    for name in BRIDGE_FILES:
        show = subprocess.run(
            ["git", "-C", paths["core"], "show", f"{revision}:{name}"],
            capture_output=True, check=True)
        with open(os.path.join(tree, name), "wb") as handle:
            handle.write(show.stdout)
    return tree


def build_extension(paths, toolchain, core, folder, debug):
    """Build the des_3pcf extension from one source tree.

    make runs in the build folder, where the makefile expects the
    project's interface.cpp and writes its objects and the extension;
    CORE points it at the source tree, ROOTDIR at the Cocoa root for the
    shared libraries, and PYTHON at this interpreter.

    Arguments:
      paths = the dict from locate_repository
      toolchain = the dict from locate_toolchain
      core = the source tree (the live repository or the reference tree)
      folder = new build folder inside the scratch folder
      debug = True for the UBSan debug build (COSMOLIKE_DEBUG_MODE=1)

    Returns:
      absolute path of the build folder that holds the extension.

    Raises:
      AssertionError with the end of make's output when the build fails.
    """
    os.makedirs(folder)
    shutil.copy(paths["interface"], folder)
    command = [
        toolchain["make"], "-f", paths["makefile"], "-j8",
        f"ROOTDIR={paths['cocoa']}", f"CORE={core}",
        f"PYTHON={toolchain['python']}",
    ]
    if debug:
        command.append("COSMOLIKE_DEBUG_MODE=1")
    build = subprocess.run(command, cwd=folder, capture_output=True,
                           text=True)
    assert build.returncode == 0, (
        f"extension build in {folder} failed with exit code "
        f"{build.returncode}:\n{build.stdout[-3000:]}\n{build.stderr[-3000:]}")
    library = os.path.join(folder, MODULE + ".so")
    assert os.path.isfile(library), f"make produced no {library}"
    return folder


def run_child(name, build, threads):
    """Run one sequence in a child process and return what it reported.

    Arguments:
      name = the sequence name (a key of SEQUENCES)
      build = build folder of the extension the child imports
      threads = OMP_NUM_THREADS of the child

    Returns:
      a dict with "code" (the exit code; negative for a signal), "output"
      (stdout and stderr joined) and "result" (the parsed RESULT line, or
      None when the child printed none).

    Raises:
      AssertionError when the child's output holds a sanitizer report;
      subprocess.TimeoutExpired when the child hangs.
    """
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = str(threads)
    child = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", name,
         "--build", build, "--threads", str(threads)],
        capture_output=True, text=True, env=environment,
        timeout=CHILD_TIMEOUT_SECONDS)
    output = child.stdout + child.stderr
    assert SANITIZER_TEXT not in output, (
        f"sequence {name} in {build} printed a sanitizer report:\n{output}")
    result = None
    for line in child.stdout.splitlines():
        if line.startswith(RESULT_TAG):
            result = json.loads(line[len(RESULT_TAG):])
    return {"code": child.returncode, "output": output, "result": result}


def require_success(run, name):
    """Assert that a child finished normally and reported a result.

    Arguments:
      run = the dict from run_child
      name = the sequence name, for the message

    Returns:
      the child's result dict.
    """
    assert run["code"] == 0 and run["result"] is not None, (
        f"sequence {name} must succeed; it exited {run['code']} with "
        f"output:\n{run['output']}")
    return run["result"]


def require_refusal(run, name, diagnostic):
    """Assert that a child stopped through log_fatal with a diagnostic.

    log_fatal is followed by exit(1), so the exit code must be exactly 1:
    a negative code is a signal (a crash), and 0 means the refused call
    went through.

    Arguments:
      run = the dict from run_child
      name = the sequence name, for the message
      diagnostic = text the refusal must print

    Returns:
      nothing.
    """
    assert run["code"] == 1, (
        f"sequence {name} must exit 1 through log_fatal; it exited "
        f"{run['code']} with output:\n{run['output']}")
    assert diagnostic in run["output"], (
        f"sequence {name} exited 1 without the diagnostic "
        f"{diagnostic!r}; output:\n{run['output']}")
    assert run["result"] is None, (
        f"sequence {name} reported a result after the refused call")


def check_ia_switch(live, threads):
    """Stage 3: binned to power law against a fresh power-law process.

    Arguments:
      live = build folder of the live extension
      threads = OpenMP threads of the children

    Returns:
      the fresh power-law result, reused by stage 4.
    """
    fresh = require_success(run_child("fresh_power_law", live, threads),
                            "fresh_power_law")
    switched = require_success(
        run_child("binned_to_power_law", live, threads),
        "binned_to_power_law")
    assert switched["ct"] == 0.0, (
        f"after the binned-to-power-law switch C_t = {switched['ct']}; "
        "the old bin-2 A2 amplitude 1.7 leaked into the C_t slot")
    assert switched["ia"] == fresh["ia"], (
        "the IA dictionary after the switch differs from a fresh "
        f"power-law process:\n{switched['ia']}\n{fresh['ia']}")
    assert switched["multipoles"] == fresh["multipoles"], (
        "the b_L multipoles after the switch differ from a fresh "
        "power-law process")
    print("PASS stage 3: binned -> power law gives C_t = 0.0 and the "
          "fresh process's IA dictionary and b_L bitwise")
    return fresh


def check_round_trip(live, threads, fresh):
    """Stage 4: power law -> binned -> power law against a fresh process.

    Arguments:
      live = build folder of the live extension
      threads = OpenMP threads of the children
      fresh = the fresh power-law result of stage 3

    Returns:
      nothing.
    """
    trip = require_success(run_child("power_law_round_trip", live, threads),
                           "power_law_round_trip")
    assert trip.pop("ct_installed") == 0.25, (
        "the explicitly installed C_t = 0.25 did not read back before the "
        "round trip, so the sequence does not test its reset")
    assert trip["ct"] == 0.0, (
        f"after the round trip C_t = {trip['ct']}; neither the old C_t "
        "0.25 nor the bin-2 A2 amplitude 1.7 may survive the mode changes")
    assert trip["ia"] == fresh["ia"], (
        "the IA dictionary after the round trip differs from a fresh "
        f"power-law process:\n{trip['ia']}\n{fresh['ia']}")
    assert trip["multipoles"] == fresh["multipoles"], (
        "the b_L multipoles after the round trip differ from a fresh "
        "power-law process")
    print("PASS stage 4: power law with C_t = 0.25 -> binned -> power law "
          "reads C_t = 0.0 and equals the fresh process bitwise")


def check_ct_refusal(live, threads):
    """Stage 5: the C_t setter under the binned mode exits through log_fatal.

    Arguments:
      live = build folder of the live extension
      threads = OpenMP threads of the children

    Returns:
      nothing.
    """
    run = run_child("ct_under_binned", live, threads)
    require_refusal(run, "ct_under_binned", CT_DIAGNOSTIC)
    print("PASS stage 5: C_t under IA_REDSHIFT_BINNING exits 1 through "
          "log_fatal")


def check_source_resize(live, threads):
    """Stage 6: bin-count changes refuse until the matching table arrives.

    Arguments:
      live = build folder of the live extension
      threads = OpenMP threads of the children

    Returns:
      nothing.
    """
    grow = run_child("grow_before_install", live, threads)
    require_refusal(grow, "grow_before_install",
                    TABLE_DIAGNOSTIC.format(2, 3))
    shrink = run_child("shrink_before_install", live, threads)
    require_refusal(shrink, "shrink_before_install",
                    TABLE_DIAGNOSTIC.format(3, 2))
    trip = require_success(run_child("resize_round_trip", live, threads),
                           "resize_round_trip")
    assert trip["generations"][0] < trip["generations"][1] < \
        trip["generations"][2], (
            f"every resize must build a new model: generations "
            f"{trip['generations']}")
    assert trip["three_bin_finite"], (
        "the three-bin model's b_L multipoles are not finite")
    assert trip["first"] == trip["last"], (
        "the two-bin model after 2 -> 3 -> 2 bins differs from the first "
        "two-bin model")
    print("PASS stage 6: 2 -> 3 and 3 -> 2 bins without a new table exit 1; "
          "2 -> 3 -> 2 with matching tables builds and returns bitwise")


def check_control(live, reference, threads):
    """Stage 7: a consistent sequence is unchanged by the fix.

    Arguments:
      live = build folder of the live extension
      reference = build folder of the reference extension
      threads = OpenMP threads of the children

    Returns:
      nothing.
    """
    new = require_success(run_child("consistent", live, threads),
                          "consistent (live)")
    old = require_success(run_child("consistent", reference, threads),
                          "consistent (reference)")
    for key in new:
        assert new[key] == old[key], (
            f"the consistent sequence differs between the live and the "
            f"reference build in {key!r}:\n{new[key]}\n{old[key]}")
    print("PASS stage 7: the consistent sequence gives bitwise-identical "
          "IA dictionaries, b_L and radial tables before and after the fix")


# ---------------------------------------------------------------------------
# Child side: the sequences, run inside the child process.
# ---------------------------------------------------------------------------


def analytic_point(numpy, nbin):
    """Deterministic analytic inputs for one model, with nbin source bins.

    The forms repeat physical_inputs of the des_3pcf data-vector tests: a
    smooth linear spectrum, a nonlinear table that evolves differently
    from D^2, chi(z) = 3000 z / (1 + 0.35 z) in Mpc/h, D(z) = 1/(1+z) (so
    the core's G = D (1+z) is one), and source densities
    z^2 exp(-(z/z0)^1.5) that vanish at z = 0, with z0 = 0.6, 0.95, 1.3.

    Arguments:
      numpy = the numpy module
      nbin = number of source bins (at most three)

    Returns:
      a dict of the installer arrays.
    """
    k = numpy.geomspace(1.0e-5, 1.0e3, 96)
    z = numpy.linspace(0.0, 2.0, 41)
    linear = 2.0e4 * k / (1.0 + (k / 0.15) ** 2) ** 1.8
    nonlinear0 = linear * (1.0 + (k / 0.35) ** 2) ** 0.22
    nonlinear = nonlinear0[None, :] / (1.0 + z[:, None]) ** 2.4
    growth = 1.0 / (1.0 + z)
    linear_table = linear[None, :] * growth[:, None] ** 2
    sources = []
    for z0 in (0.6, 0.95, 1.3)[:nbin]:
        sources.append(z ** 2 * numpy.exp(-(z / z0) ** 1.5))
    # The installers take ln P flattened as lnP[i*nz + j] with i the k
    # index and j the z index; Fortran order of an [nz, nk] array is
    # exactly that layout.
    return {
        "log10k": numpy.ascontiguousarray(numpy.log10(k)),
        "z": z,
        "lnPnl": numpy.ascontiguousarray(
            numpy.log(nonlinear).flatten(order="F")),
        "lnPlin": numpy.ascontiguousarray(
            numpy.log(linear_table).flatten(order="F")),
        "chi": 3000.0 * z / (1.0 + 0.35 * z),
        "G": numpy.ascontiguousarray(growth * (1.0 + z)),
        "nofz": numpy.ascontiguousarray(
            numpy.column_stack([z] + sources)),
        "los": numpy.linspace(0.005, 2.0, 25),
    }


def install_background(core, numpy, nbin):
    """Install cosmology, tables, sources and scalars; no IA state.

    Arguments:
      core = the extension's core submodule
      numpy = the numpy module
      nbin = number of source bins

    Returns:
      nothing.
    """
    point = analytic_point(numpy, nbin)
    core.set_cosmological_parameters(omega_matter=0.3, omega_baryon=0.048,
                                     hubble=70.0, omega_nu_h2=0.0)
    core.set_non_linear_power_spectrum(log10k=point["log10k"],
                                       z=point["z"], lnP=point["lnPnl"])
    core.set_linear_power_spectrum(log10k=point["log10k"], z=point["z"],
                                   lnP=point["lnPlin"])
    core.set_distances(z=point["z"], chi=point["chi"])
    core.set_growth(z=point["z"], G=point["G"])
    install_sources(core, numpy, nbin)
    core.set_scalars(ns=0.965, sigma8=0.8, w0=-1.0, wa=0.0)
    core.set_los(z=point["los"])
    core.set_resolution(kernel_nz=101, source_readers=1, table_boost=8,
                        fit_nz=33)


def install_sources(core, numpy, nbin):
    """Declare nbin source bins and install their matching n(z) table.

    Arguments:
      core = the extension's core submodule
      numpy = the numpy module
      nbin = number of source bins

    Returns:
      nothing.
    """
    point = analytic_point(numpy, nbin)
    core.set_source_sample_size(ntomo=nbin)
    core.set_source_sample(nofz=point["nofz"])
    core.set_nuisance_shear_photoz(bias=numpy.zeros(nbin))


def install_ia(core, numpy, mode, values):
    """Select an IA redshift mode (TATT model) and install its amplitudes.

    Arguments:
      core = the extension's core submodule
      numpy = the numpy module
      mode = IA_REDSHIFT_BINNING or IA_REDSHIFT_EVOLUTION
      values = one of the amplitude dicts (POWER_LAW, BINNED, ...)

    Returns:
      nothing.
    """
    core.init_IA(ia_model=IA_MODEL_TATT, ia_redshift_evolution=mode)
    core.set_nuisance_ia(A1=numpy.array(values["A1"]),
                         A2=numpy.array(values["A2"]),
                         B_TA=numpy.array(values["B_TA"]))


def fingerprint(numpy, value):
    """Exact, JSON-safe form of a result, for bitwise comparison.

    Floats become their hexadecimal form (float.hex is exact), arrays
    their shape, dtype and the SHA-256 of their bytes; dicts are
    fingerprinted entry by entry.

    Arguments:
      numpy = the numpy module
      value = a float, int, numpy array or dict of those

    Returns:
      the fingerprint (str, int, list or dict).
    """
    if isinstance(value, dict):
        return {key: fingerprint(numpy, value[key]) for key in sorted(value)}
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, int):
        return value
    array = numpy.ascontiguousarray(value)
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    return [list(array.shape), str(array.dtype), digest]


def multipoles(core, numpy, bins):
    """Build (or reuse) the model and sample its b_L multipoles.

    The b_L table is the model's IA-dependent prediction; it is sampled
    with the IA dictionary the bridge translates from the structs.

    Arguments:
      core = the extension's core submodule
      numpy = the numpy module
      bins = three source-bin indices of the sampled triplet

    Returns:
      (ia, table): the IA dictionary and the b_L array.
    """
    model = core.model_from_core()
    ia = core.ia_parameters_from_core()
    ell = numpy.geomspace(20.0, 3000.0, 12)
    triplet = numpy.array(bins, dtype=numpy.int32)
    table = model.multipoles(ell=ell, bins=triplet, parameters=ia, lmax=3,
                             nmu=16)
    return ia, table


def ia_result(core, numpy):
    """The IA part of a stage 3 or 4 result, after the final power law.

    Arguments:
      core = the extension's core submodule
      numpy = the numpy module

    Returns:
      a dict with the plain C_t value and the fingerprints.
    """
    ia, table = multipoles(core, numpy, (0, 1, 2))
    return {"ct": ia["ct"], "ia": fingerprint(numpy, ia),
            "multipoles": fingerprint(numpy, table)}


def sequence_fresh_power_law(core, numpy):
    """A process that only ever installs the final power law."""
    core.initial_setup()
    install_background(core, numpy, 3)
    install_ia(core, numpy, IA_REDSHIFT_EVOLUTION, POWER_LAW)
    return ia_result(core, numpy)


def sequence_binned_to_power_law(core, numpy):
    """The red-team sequence: binned amplitudes, then the power law."""
    core.initial_setup()
    install_background(core, numpy, 3)
    install_ia(core, numpy, IA_REDSHIFT_BINNING, BINNED)
    install_ia(core, numpy, IA_REDSHIFT_EVOLUTION, POWER_LAW)
    return ia_result(core, numpy)


def sequence_power_law_round_trip(core, numpy):
    """Power law with C_t and a built model, binned, then the power law."""
    core.initial_setup()
    install_background(core, numpy, 3)
    install_ia(core, numpy, IA_REDSHIFT_EVOLUTION, EARLY_POWER_LAW)
    core.set_nuisance_ia_ct(ct=0.25)
    ia, table = multipoles(core, numpy, (0, 1, 2))
    installed = ia["ct"]
    install_ia(core, numpy, IA_REDSHIFT_BINNING, BINNED)
    install_ia(core, numpy, IA_REDSHIFT_EVOLUTION, POWER_LAW)
    result = ia_result(core, numpy)
    result["ct_installed"] = installed
    return result


def sequence_ct_under_binned(core, numpy):
    """The C_t setter in binned mode; the setter must stop the process."""
    core.initial_setup()
    core.set_source_sample_size(ntomo=3)
    install_ia(core, numpy, IA_REDSHIFT_BINNING, BINNED)
    core.set_nuisance_ia_ct(ct=0.3)
    return {"reached": "the C_t setter returned under the binned mode"}


def sequence_grow_before_install(core, numpy):
    """A two-bin model, then three bins declared without a new table."""
    core.initial_setup()
    install_background(core, numpy, 2)
    install_ia(core, numpy, IA_REDSHIFT_EVOLUTION, POWER_LAW)
    multipoles(core, numpy, (0, 1, 0))
    core.set_source_sample_size(ntomo=3)
    core.model_from_core()
    return {"reached": "a model was built from the two-bin table"}


def sequence_shrink_before_install(core, numpy):
    """A three-bin model, then two bins declared without a new table."""
    core.initial_setup()
    install_background(core, numpy, 3)
    install_ia(core, numpy, IA_REDSHIFT_EVOLUTION, POWER_LAW)
    multipoles(core, numpy, (0, 1, 2))
    core.set_source_sample_size(ntomo=2)
    core.model_from_core()
    return {"reached": "a model was built from the three-bin table"}


def sequence_resize_round_trip(core, numpy):
    """2 -> 3 -> 2 source bins, each with its matching table."""
    core.initial_setup()
    install_background(core, numpy, 2)
    install_ia(core, numpy, IA_REDSHIFT_EVOLUTION, POWER_LAW)
    generations = []
    ia, first = multipoles(core, numpy, (0, 1, 0))
    generations.append(core.model_generation())
    install_sources(core, numpy, 3)
    ia, middle = multipoles(core, numpy, (0, 1, 2))
    generations.append(core.model_generation())
    install_sources(core, numpy, 2)
    ia, last = multipoles(core, numpy, (0, 1, 0))
    generations.append(core.model_generation())
    return {"generations": generations,
            "three_bin_finite": bool(numpy.all(numpy.isfinite(middle))),
            "first": fingerprint(numpy, first),
            "last": fingerprint(numpy, last)}


def sequence_consistent(core, numpy):
    """A correct power-law setup, a model, a C_t change and a rebuild."""
    core.initial_setup()
    install_background(core, numpy, 3)
    install_ia(core, numpy, IA_REDSHIFT_EVOLUTION, POWER_LAW)
    core.set_nuisance_ia_ct(ct=0.25)
    ia_a, table_a = multipoles(core, numpy, (0, 1, 2))
    radial = core.model_from_core().radial()
    core.set_nuisance_ia_ct(ct=0.4)
    ia_b, table_b = multipoles(core, numpy, (0, 2, 1))
    return {"ia_first": fingerprint(numpy, ia_a),
            "multipoles_first": fingerprint(numpy, table_a),
            "radial": fingerprint(numpy, dict(radial)),
            "ia_second": fingerprint(numpy, ia_b),
            "multipoles_second": fingerprint(numpy, table_b)}


SEQUENCES = {
    "fresh_power_law": sequence_fresh_power_law,
    "binned_to_power_law": sequence_binned_to_power_law,
    "power_law_round_trip": sequence_power_law_round_trip,
    "ct_under_binned": sequence_ct_under_binned,
    "grow_before_install": sequence_grow_before_install,
    "shrink_before_install": sequence_shrink_before_install,
    "resize_round_trip": sequence_resize_round_trip,
    "consistent": sequence_consistent,
}


def child_main(name, build, threads):
    """Run one sequence against the extension in build and print its result.

    Arguments:
      name = the sequence name
      build = build folder holding the extension
      threads = OpenMP threads of the extension

    Returns:
      nothing; prints one RESULT line.
    """
    import numpy
    sys.path.insert(0, build)
    import cosmolike_des_3pcf_interface as compiled
    where = os.path.dirname(os.path.abspath(compiled.__file__))
    assert where == os.path.abspath(build), (
        f"imported {compiled.__file__} instead of the extension in {build}")
    compiled.set_omp_threads(nthreads=threads)
    result = SEQUENCES[name](compiled.core, numpy)
    print(RESULT_TAG + json.dumps(result), flush=True)


def main():
    """Validate, build both extensions, run every stage, clean up.

    Returns:
      nothing; prints the stage results and the final PASS line.
    """
    parser = argparse.ArgumentParser(
        description="Check the state transitions of the three-point bridge.")
    parser.add_argument("--debug", action="store_true",
                        help="build both extensions with UBSan")
    parser.add_argument("--threads", type=int, default=4,
                        help="OpenMP threads of every child (default 4)")
    parser.add_argument("--reference", default=REFERENCE_REVISION,
                        help="revision of the reference bridge files")
    parser.add_argument("--child", choices=sorted(SEQUENCES),
                        help=argparse.SUPPRESS)
    parser.add_argument("--build", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.child is not None:
        child_main(arguments.child, arguments.build, arguments.threads)
        return
    assert arguments.threads > 0, "--threads must be positive"

    paths = locate_repository()
    toolchain = locate_toolchain()
    validate_reference(paths, arguments.reference)
    print(f"repository: {paths['core']}")
    print(f"reference:  {arguments.reference} (bridge files only)")
    print(f"build:      {'debug (UBSan)' if arguments.debug else 'strict'}")
    print("PASS stage 1: layout, environment and reference revision")
    # The scratch folder lands next to this script with a scratch_
    # prefix; mkdtemp creates it with a unique suffix so two runs cannot
    # collide. The finally block removes it on every exit path.
    here = os.path.dirname(os.path.abspath(__file__))
    scratch = tempfile.mkdtemp(prefix="scratch_", dir=here)
    try:
        live = build_extension(paths, toolchain, paths["core"],
                               os.path.join(scratch, "live"),
                               arguments.debug)
        tree = prepare_reference_tree(paths, arguments.reference, scratch)
        reference = build_extension(paths, toolchain, tree,
                                    os.path.join(scratch, "reference"),
                                    arguments.debug)
        print("PASS stage 2: live and reference extensions built")
        fresh = check_ia_switch(live, arguments.threads)
        check_round_trip(live, arguments.threads, fresh)
        check_ct_refusal(live, arguments.threads)
        check_source_resize(live, arguments.threads)
        check_control(live, reference, arguments.threads)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print("bridge state check: PASS")


if __name__ == "__main__":
    main()
