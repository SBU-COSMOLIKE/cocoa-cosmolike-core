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
     instead of hanging or returning an unusable 0.

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
        # basics.h includes "simde/x86/avx2.h"; in a Cocoa checkout the
        # simde package sits beside cosmolike_core in the same
        # external_modules/code folder, exactly where the project
        # makefiles point their include path.
        "simde_root": os.path.join(os.path.dirname(core), "simde"),
    }
    for name in ("basics", "structs", "log", "probe"):
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
        print("PASS stage 2: probe compiled against the live sources")
        all_lengths, even_lengths = enumerate_admissible()
        print(f"PASS stage 3: enumerated {len(all_lengths)} admissible "
              f"lengths up to LONG_MAX ({len(even_lengths)} even)")
        check_exhaustive_range(executable, all_lengths, even_lengths)
        check_named_cases(executable, all_lengths, even_lengths)
        check_int_boundary(executable, all_lengths, even_lengths)
        check_refusals(executable, all_lengths)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print("fft parity check: PASS")


if __name__ == "__main__":
    main()
