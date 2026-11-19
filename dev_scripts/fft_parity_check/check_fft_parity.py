#!/usr/bin/env python3
"""Check the FFT even-length selection of the live sources for parity bugs.

What is being checked. FFTLog and FAST-PT hand their transform lengths to
next_fft_size (cosmolike/basics.c), which returns the smallest length at or
above the request whose prime factors are all in {2, 3, 5, 7} - the radices
FFTW transforms fastest. Such a length is called admissible below. Several
callers additionally need an EVEN length, because they split the transform
at N/2 (the Nyquist index, the highest frequency a length-N grid resolves):
J_abl in cfastpt/cfastpt.c and the FFTLog spectra of cosmolike/cosmo2D.c and
cosmolike/cosmo2D_cluster.c. Those callers round the request up to even,
call next_fft_size, and retry from the next integer until the result lands
even.

The failure mode this guards. Consecutive admissible lengths can share
parity: the even request 242 meets 243 = 3^5 and then 245 = 5 * 7^2, both
odd, before the even 250. A selection that retries only once can therefore
still return an odd length. With an odd N, J_abl's padding loop fills its
work buffer only up to index 2*(N/2) = N - 1, the next loop reads the
unwritten entry N back while mirroring the negative-frequency coefficients,
and that uninitialized read made the scalar FAST-PT bias spectra differ
between identical calls. The input 6056 reaches this through a public
accuracy setting (init_fpt_internal_boost at 1.187 selects 1306 internal
points, padded to a raw request of 6056, whose first two admissible lengths
6075 and 6125 are both odd).

How the check works, stage by stage:
  1. validate the repository layout and the compiler toolchain;
  2. compile fft_parity_probe.c (same folder) together with the live
     cosmolike/basics.c, cosmolike/structs.c and log.c/src/log.c into a
     scratch folder, so the probe runs the selection code of the working
     tree, never an extracted snapshot;
  3. enumerate every admissible length up to 2^63 by multiplying prime
     powers of 2, 3, 5 and 7 (about 80000 values). This reference never
     repeats the production algorithm's trial division of candidates, so
     the two implementations cannot share a bug;
  4. probe every request from 1 to 100000: next_fft_size must return the
     smallest admissible length at or above the request, and the caller
     selection must return the smallest EVEN admissible length at or above
     the even-rounded request, with exactly the retry count the
     enumeration predicts (1040 of these requests need more than one
     retry, among them the named cases 242, 722 and 6056);
  5. probe requests around INT_MAX: the selection stays minimal-even in
     long arithmetic, and the driver reports which results exceed
     INT_MAX - at the call sites such a result triggers log_fatal instead
     of being truncated by the (int) cast;
  6. run the refusal cases in child processes with a timeout: a request
     below 1 and a request above the largest representable admissible
     length must terminate the probe through log_fatal (exit code 1)
     instead of hanging or returning an unusable 0;
  7. run the PRODUCTION kernel route. The selection probe of stages 4-6
     replicates the call sites' retry loop, and a replica cannot guard
     the code that actually runs, so a second probe
     (fft_kernel_probe.c) compiles the live cfastpt/cfastpt.c (with
     its utils files) and calls the real J_abl on the configuration
     that formerly selected the odd length: 1306 input points, the
     public internal-boost-1.187 setting, padded raw request 6056.
     The driver asserts that the selected lengths are even and minimal
     (derived through the live next_fft_size, since J_abl keeps its
     length in function-local state with no read-out), that the two
     in-process calls, a second fresh process and an OMP_NUM_THREADS=1
     process all agree bitwise (the output prints in exact
     hexadecimal-float form), and that the peak matches the repaired
     kernel's reference value 42.849612101071031 - with an odd length
     the kernel read an unwritten work-buffer entry and these
     comparisons failed. The unboosted default route (1100 points)
     runs once and must reproduce itself bitwise in-process.

How to run it (from the Cocoa/ folder, cocoa environment active, so the
compiler and the gsl/fftw headers of the environment are found):

    conda activate cocoapy311
    source start_cocoa.sh
    python ./external_modules/code/cosmolike_core/dev_scripts/\\
fft_parity_check/check_fft_parity.py

Expected output: one PASS line per stage and a final line
"fft parity check: PASS" with exit code 0. On any failure an
AssertionError names the request and the disagreeing values, and the exit
code is nonzero. The build lands in a scratch_* folder created next to
this script and is removed before the script returns, pass or fail.
"""

import bisect
import os
import shutil
import subprocess
import sys
import tempfile

# The exhaustive request range of stage 4. Probing every request up to
# 100000 covers all 1040 multi-retry requests below that bound (the three
# named cases included) and finishes in seconds; the INT_MAX neighborhood
# is covered separately in stage 5.
RANGE_LIMIT = 100000

# The three requests the 2026-11-18 defect report demonstrated: for each,
# the first two admissible lengths above the even-rounded request are both
# odd, so the selection needs two retries (242 -> 250, 722 -> 750,
# 6056 -> 6144). They sit inside the stage-4 range and are also asserted
# individually so a regression names them directly.
NAMED_CASES = (242, 722, 6056)

# Signed 32-bit bounds of the call sites that store the selected length in
# an int: results above INT_MAX must be refused there, never truncated.
INT_MAX = 2 ** 31 - 1

# Signed 64-bit bound of next_fft_size's own long arithmetic; the
# enumeration runs up to this bound so the largest representable
# admissible length is known exactly.
LONG_MAX = 2 ** 63 - 1

# Each refusal child is given ample time; the pre-repair behavior for a
# request below 1 was an infinite scan, so a generous timeout still
# converts a hang into a clear test failure instead of a stuck run.
REFUSAL_TIMEOUT_SECONDS = 60

# The kernel-route input sizes: 1306 points is the public
# internal-boost-1.187 setting whose padded raw request 6056 formerly
# selected the odd length 6125; 1100 points is the unboosted default,
# whose selection was always even, kept as the no-regression control.
KERNEL_BOOST_SIZE = 1306
KERNEL_DEFAULT_SIZE = 1100

# Peak absolute value of the repaired kernel's output at 1306 points
# (measured on the repaired sources, IEEE-strict optimized build). The
# odd-length defect produced peaks near 2098 or 1e298 that also varied
# between runs, so a loose relative tolerance separates the two cases
# cleanly while allowing the tiny floating-point drift between this
# -O2 probe build and the projects' -O3 LTO builds.
KERNEL_REFERENCE_PEAK = 42.849612101071031
KERNEL_PEAK_TOLERANCE = 1.0e-9

# Each kernel child plans FFTs and runs three terms; seconds suffice,
# and the timeout only exists so a wedged child fails the check
# instead of hanging it.
KERNEL_TIMEOUT_SECONDS = 300


def locate_repository():
    """Resolve the cosmolike_core paths this check compiles and validate them.

    The script lives in cosmolike_core/dev_scripts/fft_parity_check/, so
    the repository root is two directory levels up from this file. Every
    source file is checked for existence before any build starts, so a
    moved file fails here with its path instead of as a compiler error.

    Returns:
      a dict with the repository root ("core"), the three C sources the
      probe links ("basics", "structs", "log"), and the probe source
      ("probe"), all as absolute paths.

    Raises:
      AssertionError naming the first missing file.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    core = os.path.dirname(os.path.dirname(here))
    paths = {
        "core": core,
        "basics": os.path.join(core, "cosmolike", "basics.c"),
        "structs": os.path.join(core, "cosmolike", "structs.c"),
        "log": os.path.join(core, "log.c", "src", "log.c"),
        "probe": os.path.join(here, "fft_parity_probe.c"),
        "kernel_probe": os.path.join(here, "fft_kernel_probe.c"),
        "cfastpt": os.path.join(core, "cfastpt", "cfastpt.c"),
        "cfastpt_utils": os.path.join(core, "cfastpt",
                                      "utils_cfastpt.c"),
        "cfastpt_complex": os.path.join(core, "cfastpt",
                                        "utils_complex_cfastpt.c"),
        # basics.h includes "simde/x86/avx2.h"; in a Cocoa checkout the
        # simde package sits beside cosmolike_core in the same
        # external_modules/code folder, exactly where the project
        # makefiles point their include path.
        "simde_root": os.path.join(os.path.dirname(core), "simde"),
    }
    for name in ("basics", "structs", "log", "probe", "kernel_probe",
                 "cfastpt", "cfastpt_utils", "cfastpt_complex"):
        assert os.path.isfile(paths[name]), (
            f"missing source file for the probe build: {paths[name]}; "
            "run this script from an intact cosmolike_core checkout")
    simde_header = os.path.join(paths["simde_root"],
                                "simde", "x86", "avx2.h")
    assert os.path.isfile(simde_header), (
        f"simde headers not found at {simde_header}; basics.h needs "
        "them, and a Cocoa checkout provides them next to "
        "cosmolike_core (run setup_cocoa.sh first)")
    return paths


def locate_toolchain():
    """Find the C compiler and the gsl/fftw headers of the cocoa environment.

    basics.c includes gsl and fftw3 headers, which live in the conda
    environment, so the environment must be active (CONDA_PREFIX set).
    The compiler preference order matches the project makefiles: the
    environment's clang, then its gcc, then whatever "cc" is on PATH.

    Returns:
      a dict with "compiler" (absolute path or command name) and
      "prefix" (the conda environment root holding include/ and lib/).

    Raises:
      AssertionError with the activation commands when the environment
      is not active, the headers are absent, or no compiler is found.
    """
    prefix = os.environ.get("CONDA_PREFIX", "")
    assert prefix != "", (
        "CONDA_PREFIX is not set: activate the cocoa environment first "
        "(conda activate cocoapy311; source start_cocoa.sh)")
    fftw_header = os.path.join(prefix, "include", "fftw3.h")
    gsl_dir = os.path.join(prefix, "include", "gsl")
    assert os.path.isfile(fftw_header), (
        f"fftw3.h not found at {fftw_header}; the active environment "
        "does not provide the fftw the cosmolike sources include")
    assert os.path.isdir(gsl_dir), (
        f"gsl headers not found at {gsl_dir}; the active environment "
        "does not provide the gsl the cosmolike sources include")
    # The kernel probe links the fftw library itself (the project
    # makefiles link -lfftw3 from this same environment); accept the
    # platform spellings of the shared or static library.
    fftw_library = None
    for library_name in ("libfftw3.dylib", "libfftw3.so", "libfftw3.a"):
        candidate = os.path.join(prefix, "lib", library_name)
        if os.path.isfile(candidate):
            fftw_library = candidate
            break
    assert fftw_library is not None, (
        f"no fftw3 library found under {os.path.join(prefix, 'lib')} "
        "(looked for libfftw3.dylib, libfftw3.so, libfftw3.a); the "
        "cocoa environment provides it - activate it first "
        "(conda activate cocoapy311; source start_cocoa.sh)")
    compiler = None
    for candidate in (os.path.join(prefix, "bin", "clang"),
                      os.path.join(prefix, "bin", "gcc")):
        if os.path.isfile(candidate):
            compiler = candidate
            break
    if compiler is None:
        # shutil.which returns the absolute path of a command found on
        # PATH, or None when no such command exists.
        compiler = shutil.which("cc")
    assert compiler is not None, (
        "no C compiler found: neither clang nor gcc in the environment "
        f"at {prefix}/bin, and no cc on PATH")
    return {"compiler": compiler, "prefix": prefix}


def build_probe(paths, toolchain, scratch):
    """Compile the probe against the live repository sources.

    One compile-and-link command produces the probe executable in the
    scratch folder. The probe needs structs.c because basics.c reads the
    global Ntable configuration structure defined there, and log.c
    because next_fft_size reports refused requests through log_fatal.
    The flags repeat the project makefiles' feature macros (HAVE_INLINE
    and GSLLIB for the gsl headers) at plain -O2: this is a logic check,
    not a performance build, and skipping LTO keeps the compile inside
    the one-minute budget.

    Arguments:
      paths = the dict from locate_repository
      toolchain = the dict from locate_toolchain
      scratch = existing folder that receives the executable

    Returns:
      absolute path of the compiled probe executable.

    Raises:
      AssertionError with the compiler's output when the build fails.
    """
    executable = os.path.join(scratch, "fft_parity_probe")
    command = [
        toolchain["compiler"], "-std=gnu99", "-O2", "-fopenmp",
        "-DHAVE_INLINE", "-DGSLLIB",
        "-I", paths["core"],
        "-I", paths["simde_root"],
        "-I", os.path.join(toolchain["prefix"], "include"),
        paths["probe"], paths["basics"], paths["structs"], paths["log"],
        "-o", executable,
        "-L", os.path.join(toolchain["prefix"], "lib"),
        "-Wl,-rpath," + os.path.join(toolchain["prefix"], "lib"),
        "-lgsl", "-lgslcblas", "-lfftw3", "-lm",
    ]
    build = subprocess.run(command, capture_output=True, text=True)
    assert build.returncode == 0, (
        "probe build failed with exit code "
        f"{build.returncode}:\n{build.stderr}")
    return executable


def build_kernel_probe(paths, toolchain, scratch):
    """Compile the production-kernel probe against the live sources.

    Same toolchain and flags as build_probe, with the live
    cfastpt/cfastpt.c and its two utils files added (they hold the
    window and complex-gamma helpers J_abl calls) and the fftw library
    linked, since J_abl executes real transforms. Nothing on the
    parity-sensitive path is replicated: the J_abl that runs is the
    working tree's.

    Arguments:
      paths = the dict from locate_repository
      toolchain = the dict from locate_toolchain
      scratch = existing folder that receives the executable

    Returns:
      absolute path of the compiled kernel-probe executable.

    Raises:
      AssertionError with the compiler's output when the build fails.
    """
    executable = os.path.join(scratch, "fft_kernel_probe")
    command = [
        toolchain["compiler"], "-std=gnu99", "-O2", "-fopenmp",
        "-DHAVE_INLINE", "-DGSLLIB",
        "-I", paths["core"],
        "-I", paths["simde_root"],
        "-I", os.path.join(toolchain["prefix"], "include"),
        paths["kernel_probe"], paths["cfastpt"],
        paths["cfastpt_utils"], paths["cfastpt_complex"],
        paths["basics"], paths["structs"], paths["log"],
        "-o", executable,
        "-L", os.path.join(toolchain["prefix"], "lib"),
        "-Wl,-rpath," + os.path.join(toolchain["prefix"], "lib"),
        "-lgsl", "-lgslcblas", "-lfftw3", "-lm",
    ]
    build = subprocess.run(command, capture_output=True, text=True)
    assert build.returncode == 0, (
        "kernel probe build failed with exit code "
        f"{build.returncode}:\n{build.stderr}")
    return executable


def run_kernel_probe(executable, size, omp_threads):
    """Run the kernel probe once and parse its output.

    One run calls the production J_abl twice in the same process (the
    first call builds the cached transform plans and grids, the second
    exercises the cache) and prints the exact bits of every output
    value in hexadecimal-float form.

    Arguments:
      executable = the compiled kernel probe
      size = number of input k points (KERNEL_BOOST_SIZE or
        KERNEL_DEFAULT_SIZE)
      omp_threads = OMP_NUM_THREADS for the child process; the OpenMP
        runtime reads it at startup, which is why the thread-count
        comparison needs separate processes

    Returns:
      a dict with "sizes" = (raw, selected, ntotal) and "calls" = list
      of two dicts, each with "finite" (bool), "peak" (float) and
      "values" (the call's hexadecimal lines joined into one string,
      compared verbatim for bitwise equality).

    Raises:
      AssertionError when the output does not have the expected shape.
    """
    child_env = dict(os.environ)
    child_env["OMP_NUM_THREADS"] = str(omp_threads)
    probe = subprocess.run(
        [executable, str(size)],
        capture_output=True, text=True, check=True,
        timeout=KERNEL_TIMEOUT_SECONDS, env=child_env)
    lines = probe.stdout.splitlines()
    header = lines[0].split()
    assert header[0] == "sizes" and len(header) == 8, (
        f"malformed kernel header line: {lines[0]!r}")
    assert int(header[1]) == size, (
        f"kernel probe echoed size {header[1]}, expected {size}")
    sizes = (int(header[3]), int(header[5]), int(header[7]))
    # Each call contributes one summary line plus 3*size value lines.
    values_per_call = 3 * size
    expected_lines = 1 + 2 * (1 + values_per_call)
    assert len(lines) == expected_lines, (
        f"kernel probe printed {len(lines)} lines, expected "
        f"{expected_lines}")
    calls = []
    cursor = 1
    for call in range(2):
        summary = lines[cursor].split()
        assert summary[0] == "call" and int(summary[1]) == call, (
            f"malformed call summary line: {lines[cursor]!r}")
        block_start = cursor + 1
        block_end = block_start + values_per_call
        calls.append({
            "finite": summary[3] == "1",
            "peak": float(summary[5]),
            "values": "\n".join(lines[block_start:block_end]),
        })
        cursor = block_end
    return {"sizes": sizes, "calls": calls}


def check_production_route(executable, even_lengths):
    """Stage 7: the production J_abl on the formerly failing route.

    Three child processes run the 1306-point configuration (the public
    internal-boost-1.187 setting, padded raw request 6056): two at
    4 OpenMP threads and one at 1 thread. Asserted, in order: the
    selected transform length is even and is the minimal even
    admissible length (derived through the live next_fft_size; J_abl
    itself keeps the length in function-local state with no read-out);
    every call is finite; the two calls inside each process agree
    bitwise; the two 4-thread processes agree bitwise; the 1-thread
    process agrees bitwise with them; and the peak matches
    KERNEL_REFERENCE_PEAK within KERNEL_PEAK_TOLERANCE. Before the
    repair this configuration selected the odd length 6125, J_abl read
    an unwritten work-buffer entry, and the repeatability and peak
    comparisons failed. One 1100-point run checks that the unboosted
    default route reproduces itself bitwise in-process.

    Arguments:
      executable = the compiled kernel probe
      even_lengths = the even admissible lengths from
        enumerate_admissible

    Returns:
      nothing; prints one line per run and a PASS line.
    """
    boost_a = run_kernel_probe(executable=executable,
                               size=KERNEL_BOOST_SIZE, omp_threads=4)
    boost_b = run_kernel_probe(executable=executable,
                               size=KERNEL_BOOST_SIZE, omp_threads=4)
    boost_single = run_kernel_probe(executable=executable,
                                    size=KERNEL_BOOST_SIZE,
                                    omp_threads=1)
    for label, run in (("4 threads, process a", boost_a),
                       ("4 threads, process b", boost_b),
                       ("1 thread", boost_single)):
        raw, selected, ntotal = run["sizes"]
        minimal_even = even_lengths[bisect.bisect_left(even_lengths,
                                                       raw)]
        assert selected % 2 == 0, (
            f"kernel route selected the odd length {selected} for raw "
            f"request {raw}")
        assert selected == minimal_even, (
            f"kernel route selected {selected} for raw request {raw}; "
            f"the minimal even admissible length is {minimal_even}")
        for call in run["calls"]:
            assert call["finite"], (
                f"kernel output not finite ({label})")
            drift = abs(call["peak"] / KERNEL_REFERENCE_PEAK - 1.0)
            assert drift < KERNEL_PEAK_TOLERANCE, (
                f"kernel peak {call['peak']} differs from the "
                f"reference {KERNEL_REFERENCE_PEAK} by a relative "
                f"{drift} ({label}); the odd-length defect produced "
                "peaks near 2098 or 1e298")
        assert run["calls"][0]["values"] == run["calls"][1]["values"], (
            f"the two in-process calls differ bitwise ({label}); "
            "an uninitialized read on the transform buffers is the "
            "known cause of exactly this failure")
        print(f"  boost route ({label}): raw {raw} -> even {selected} "
              f"(ntotal {ntotal}), two calls bitwise identical, peak "
              f"{run['calls'][0]['peak']:.17g}")
    assert (boost_a["calls"][0]["values"]
            == boost_b["calls"][0]["values"]), (
        "two fresh 4-thread processes disagree bitwise on the boost "
        "route; the odd-length defect showed exactly this, because "
        "the uninitialized value depends on allocation history")
    assert (boost_a["calls"][0]["values"]
            == boost_single["calls"][0]["values"]), (
        "1-thread and 4-thread runs disagree bitwise on the boost "
        "route")
    default = run_kernel_probe(executable=executable,
                               size=KERNEL_DEFAULT_SIZE, omp_threads=4)
    raw, selected, ntotal = default["sizes"]
    assert selected % 2 == 0, (
        f"default route selected the odd length {selected}")
    assert (default["calls"][0]["values"]
            == default["calls"][1]["values"]), (
        "the default route does not reproduce itself bitwise")
    print(f"  default route: raw {raw} -> even {selected} "
          f"(ntotal {ntotal}), two calls bitwise identical, peak "
          f"{default['calls'][0]['peak']:.17g}")
    print("PASS stage 7: production kernel route even, bitwise "
          "repeatable across calls, processes and thread counts")


def enumerate_admissible():
    """Enumerate every admissible length representable as a 64-bit long.

    Builds the set of products 2^a * 3^b * 5^c * 7^d <= LONG_MAX by
    repeated multiplication, one prime at a time: starting from {1},
    each pass replaces the set with every member times every power of
    the next prime that stays in range. This is the independent
    reference: it never trial-divides a candidate, so it cannot share a
    bug with next_fft_size's scan.

    Returns:
      (all_lengths, even_lengths): two sorted lists, every admissible
      length and the even ones alone, both ending at the largest
      admissible value not exceeding LONG_MAX (about 80000 and 76000
      entries).
    """
    values = {1}
    for prime in (2, 3, 5, 7):
        grown = set()
        for value in values:
            while value <= LONG_MAX:
                grown.add(value)
                value = value * prime
        values = grown
    all_lengths = sorted(values)
    even_lengths = []
    for value in all_lengths:
        if value % 2 == 0:
            even_lengths.append(value)
    return all_lengths, even_lengths


def expected_selection(request, all_lengths, even_lengths):
    """Predict the probe's full output line for one request.

    Reproduces, from the enumeration alone, what a correct selection
    must do: the direct result is the first admissible length at or
    above the request; the caller result is the first EVEN admissible
    length at or above the even-rounded request; the retry count walks
    the same path the caller's loop walks (first admissible, then the
    first admissible above each odd intermediate) and counts the steps,
    so the probe's loop structure is checked, not only its endpoint.

    Arguments:
      request = the transform-length request, at least 1
      all_lengths = sorted admissible lengths from enumerate_admissible
      even_lengths = the even subset, same order

    Returns:
      (direct, caller, retries) as integers.
    """
    # bisect_left returns the index of the first list entry >= request,
    # so all_lengths[index] is the minimal admissible length at or
    # above the request.
    direct = all_lengths[bisect.bisect_left(all_lengths, request)]
    even_start = request + request % 2
    caller = even_lengths[bisect.bisect_left(even_lengths, even_start)]
    retries = 0
    step = all_lengths[bisect.bisect_left(all_lengths, even_start)]
    while step % 2 == 1:
        step = all_lengths[bisect.bisect_left(all_lengths, step + 1)]
        retries = retries + 1
    assert step == caller, (
        f"reference inconsistency at request {request}: walked to {step}, "
        f"minimal even is {caller}")
    return direct, caller, retries


def parse_probe_lines(text):
    """Turn the probe's stdout into a list of 4-integer tuples.

    Arguments:
      text = the probe's stdout: lines of "n direct caller retries".

    Returns:
      list of (request, direct, caller, retries) tuples in file order.

    Raises:
      AssertionError on a line that does not hold exactly 4 integers.
    """
    rows = []
    for line in text.splitlines():
        fields = line.split()
        assert len(fields) == 4, f"malformed probe line: {line!r}"
        rows.append((int(fields[0]), int(fields[1]),
                     int(fields[2]), int(fields[3])))
    return rows


def check_exhaustive_range(executable, all_lengths, even_lengths):
    """Stage 4: verify every request from 1 to RANGE_LIMIT.

    One probe run prints all RANGE_LIMIT lines; each is compared with
    the enumeration's prediction. The multi-retry count is reported so
    the log shows the parity trap was actually exercised.

    Arguments:
      executable = the compiled probe
      all_lengths, even_lengths = the enumeration reference

    Returns:
      nothing; prints one PASS line.

    Raises:
      AssertionError naming the first disagreeing request.
    """
    probe = subprocess.run(
        [executable, "range", "1", str(RANGE_LIMIT)],
        capture_output=True, text=True, check=True)
    rows = parse_probe_lines(probe.stdout)
    assert len(rows) == RANGE_LIMIT, (
        f"probe printed {len(rows)} lines, expected {RANGE_LIMIT}")
    multi_retry = 0
    for request, direct, caller, retries in rows:
        expected = expected_selection(request, all_lengths, even_lengths)
        assert (direct, caller, retries) == expected, (
            f"request {request}: probe returned direct={direct}, "
            f"caller={caller}, retries={retries}; the enumeration "
            f"requires direct={expected[0]}, caller={expected[1]}, "
            f"retries={expected[2]}")
        if retries > 1:
            multi_retry = multi_retry + 1
    assert multi_retry > 0, (
        "no request below RANGE_LIMIT needed more than one retry; the "
        "range no longer covers the parity trap this check exists for")
    print(f"PASS stage 4: requests 1..{RANGE_LIMIT} minimal and "
          f"minimal-even ({multi_retry} needed more than one retry)")


def check_named_cases(executable, all_lengths, even_lengths):
    """Stage 4b: assert the three documented failing requests one by one.

    These are inside the exhaustive range, but asserting them separately
    makes a regression report name the historical cases directly.

    Arguments:
      executable = the compiled probe
      all_lengths, even_lengths = the enumeration reference

    Returns:
      nothing; prints one line per named case and a PASS line.
    """
    arguments = [executable, "cases"]
    for request in NAMED_CASES:
        arguments.append(str(request))
    probe = subprocess.run(arguments, capture_output=True, text=True,
                           check=True)
    rows = parse_probe_lines(probe.stdout)
    for request, direct, caller, retries in rows:
        expected = expected_selection(request, all_lengths, even_lengths)
        assert (direct, caller, retries) == expected, (
            f"named case {request}: probe returned {caller} after "
            f"{retries} retries; the enumeration requires {expected[1]} "
            f"after {expected[2]}")
        assert retries >= 2, (
            f"named case {request} needed only {retries} retries; it is "
            "documented as needing at least two")
        print(f"  named case {request}: caller selection {caller} "
              f"after {retries} retries")
    print("PASS stage 4b: named cases 242, 722, 6056")


def check_int_boundary(executable, all_lengths, even_lengths):
    """Stage 5: verify the selection around the callers' (int) boundary.

    The call sites store the selected length in an int, so a result
    above INT_MAX must make them refuse through log_fatal; the long
    arithmetic of the selection itself must stay minimal-even on both
    sides of the boundary. The requests cover values whose result still
    fits an int and values whose result exceeds it; the printed line
    says which side each case lands on.

    Arguments:
      executable = the compiled probe
      all_lengths, even_lengths = the enumeration reference

    Returns:
      nothing; prints one line per boundary case and a PASS line.
    """
    requests = [
        INT_MAX // 2 - 1,   # result fits an int comfortably
        INT_MAX // 2 + 1,   # result fits an int comfortably
        INT_MAX - 5000,     # close below the boundary
        INT_MAX - 1,
        INT_MAX,
        INT_MAX + 1,        # the request itself no longer fits an int
        INT_MAX + 5000,
    ]
    arguments = [executable, "cases"]
    for request in requests:
        arguments.append(str(request))
    probe = subprocess.run(arguments, capture_output=True, text=True,
                           check=True)
    rows = parse_probe_lines(probe.stdout)
    for request, direct, caller, retries in rows:
        expected = expected_selection(request, all_lengths, even_lengths)
        assert (direct, caller, retries) == expected, (
            f"boundary case {request}: probe returned direct={direct}, "
            f"caller={caller}, retries={retries}; the enumeration "
            f"requires {expected}")
        if caller > INT_MAX:
            side = "exceeds INT_MAX, so the call sites refuse it"
        else:
            side = "fits an int"
        print(f"  boundary case {request}: caller selection {caller} "
              f"({side})")
    print("PASS stage 5: INT_MAX boundary selections minimal-even")


def check_refusals(executable, all_lengths):
    """Stage 6: the requests next_fft_size must refuse, in timed children.

    Three requests must terminate the probe through log_fatal with exit
    code 1: 0 and -4 (below the domain; the scan would otherwise walk
    through candidate 0, where the repeated division by 2 never
    terminates) and a request so high that no admissible length exists
    between it and LONG_MAX (exhaustion; the scan would otherwise
    return an unusable 0). Each runs as a child process with a timeout,
    so a regression back to the hang becomes a clear test failure
    instead of a stuck run.

    The exhaustion request is LONG_MAX - 16, not one above the largest
    admissible length: the gap between that length and LONG_MAX spans
    about 10^14 integers, and the scan visits every one of them, so
    starting just above the last admissible length would take hours.
    Sixteen candidates prove the same refusal in microseconds; the
    enumeration first confirms that none of them is admissible.

    Arguments:
      executable = the compiled probe
      all_lengths = the enumeration; its last entry is the largest
        admissible length representable as a long, and the assertion
        below places it under the exhaustion request

    Returns:
      nothing; prints one line per refusal and a PASS line.

    Raises:
      AssertionError when a child exits 0; subprocess.TimeoutExpired
      when a child hangs.
    """
    exhaustion_request = LONG_MAX - 16
    assert all_lengths[-1] < exhaustion_request, (
        f"the largest admissible length {all_lengths[-1]} is not below "
        f"{exhaustion_request}; the exhaustion case would succeed")
    for request in (0, -4, exhaustion_request):
        child = subprocess.run(
            [executable, "one", str(request)],
            capture_output=True, text=True,
            timeout=REFUSAL_TIMEOUT_SECONDS)
        assert child.returncode == 1, (
            f"next_fft_size({request}) must terminate through log_fatal "
            f"with exit code 1; the child exited {child.returncode} with "
            f"stdout {child.stdout!r}")
        print(f"  refusal {request}: child exited 1 (log_fatal, no hang)")
    print("PASS stage 6: domain and exhaustion refusals")


def main():
    """Run every stage in order and clean the scratch build up afterwards.

    Returns:
      nothing; prints the stage results and the final PASS line.
    """
    paths = locate_repository()
    toolchain = locate_toolchain()
    print(f"repository: {paths['core']}")
    print(f"compiler:   {toolchain['compiler']}")
    # The scratch folder lands next to this script with a scratch_
    # prefix; mkdtemp creates it with a unique suffix so two runs
    # cannot collide. The finally block removes it on every exit path.
    here = os.path.dirname(os.path.abspath(__file__))
    scratch = tempfile.mkdtemp(prefix="scratch_", dir=here)
    try:
        executable = build_probe(paths=paths, toolchain=toolchain,
                                 scratch=scratch)
        kernel_executable = build_kernel_probe(paths=paths,
                                               toolchain=toolchain,
                                               scratch=scratch)
        print("PASS stage 2: both probes compiled against the live "
              "sources")
        all_lengths, even_lengths = enumerate_admissible()
        print(f"PASS stage 3: enumerated {len(all_lengths)} admissible "
              f"lengths up to LONG_MAX ({len(even_lengths)} even)")
        check_exhaustive_range(executable, all_lengths, even_lengths)
        check_named_cases(executable, all_lengths, even_lengths)
        check_int_boundary(executable, all_lengths, even_lengths)
        check_refusals(executable, all_lengths)
        check_production_route(kernel_executable, even_lengths)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print("fft parity check: PASS")


if __name__ == "__main__":
    main()
