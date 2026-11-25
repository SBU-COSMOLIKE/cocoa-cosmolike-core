"""Shared harness machinery for the Cocoa project unit tests.

Every Cocoa project repository (roman_real, desy1xplanck, des_y3,
roman_fourier, roman_kl, ...) carries the same test architecture:
frozen configurations pinned by a SHA-256 manifest, reference chi2
values recorded at freeze time, race checks that share one model
instance, accuracy advisory scans, baryonic-feedback checks, the
CFASTPT-vs-FASTPT comparison, and the Halofit-vs-EE2 comparison.
The machinery is identical from project to project; only the
project DATA differs (the examples table, the TATT point, the
accuracy knobs, the dataset names, the comparison contract). This
module holds the machinery once. Each project's
tests/cocoa_test_utils.py is a "shim", a thin module that adapts
this shared code to one project: it holds the project's data,
builds ONE CocoaTestHarness from it, and re-exports the harness's
bound methods as module-level names. The test modules and the
generator (generate_frozen_reference.py) import everything from
their project's shim, never from this module directly. A WORKED
EXAMPLE below shows the complete shim a new project writes.

The candidate matter-spectrum prior check is a separate accuracy study:
run_matter_emulator_prior_check reads the chosen current example, compares
two freshly computed predictions, and records the source, points and code
revisions under chains/. It never refreshes a frozen regression reference.

THE FROZEN-STATE DOCTRINE

Everything a test evaluates is FROZEN: stored under the project's
tests/frozen/ and pinned by a SHA-256 hash (a 64-character
fingerprint that changes when any byte of the file changes) in
tests/manifest_sha256.json. The live project configuration is never
read, so a user can edit the examples, the likelihood default yaml
files, or ../data without touching the tests. Frozen means the
INPUTS and the recorded reference outputs. The code under test (the
likelihood Python, the compiled cosmolike library, CAMB, the
numerical dependencies) is deliberately current: a regression test
compares today's behavior on the saved inputs against the saved
result, and the manifest is an identity check on the stored bytes,
not a judgment of their scientific correctness. The frozen state has
three parts:

  - frozen/frozen_config_*.py: one auto-generated module per
    configuration holding (a) the complete cobaya configuration as a
    yaml string, with every likelihood option and every parameter
    written out, including the ones that normally come from the
    likelihood default files, and (b) the exact sampled-parameter
    point the reference chi2 was evaluated at. Because every default
    is materialized in the frozen copy, a later edit to a live
    default file is shadowed and cannot reach the test.
  - frozen/data/: the tests' own copy of the data vectors,
    covariances, n(z), and masks.
  - frozen/reference_chi2.json: the chi2 values recorded at freeze
    time; each reference test compares its freshly computed chi2
    against one of them. (frozen/EXAMPLE_*.yaml snapshots sit next
    to them as human-readable provenance; no test reads them.)

Every test first verifies the manifest (verify_frozen) and refuses
to run when any frozen file changed: a tampered frozen state must
not produce a plausible-looking chi2. Refreshing the frozen state is
a deliberate maintainer action: generate_frozen_reference.py
--overwrite.

WORKER ISOLATION

Every test quantity is computed in its own worker subprocess:
single_model_chi2, ten_in_a_row_chi2 and the baryon checks each
spawn a fresh python that imports the project's cocoa_test_utils by
file path, evaluates in-process, and hands the numbers back through
a temporary json file, while its progress lines stream to the same
terminal. The reason is the cosmolike C layer: it keeps the
data-vector dimensions in C globals and aborts the whole process
when a second configuration with different dimensions initializes
after the first ("IP::set_mask: inconsistent mask"). In a project
whose examples share one data set the isolation is preventive rather
than required; every project keeps the one architecture so they all
behave identically. Inside its one worker, the race check
deliberately runs its 11 evaluations on ONE model instance, because
leaked state inside that instance is exactly what it hunts; the
baryon accuracy check builds its default and pushed models there one
after the other, which the C layer accepts because both carry
example1's dimensions. The environment variable COCOA_TESTS_WORKER
separates the two roles: a process where it equals "1" is a worker
and evaluates in-process; the parent sets it, together with
OMP_NUM_THREADS, only in the environment copy it hands the worker.

THE ZERO-BASED COMPARISONS

Three families of checks refuse to compare chi2 values taken against
shipped data far from its minimum, where the chi2 responds LINEARLY
to tiny numerical changes and a harmless rounding-level shift reads
as an alarming difference:

  - the baryon accuracy checks write the default-settings theory
    vector at the fiducial point, make it the data of a temporary
    dataset descriptor, and evaluate the pushed-settings model
    against it: the default chi2 against its own vector is zero by
    construction, so the pushed model's chi2 IS the delta, a pure
    numerics response at a minimum (_baryon_accuracy_delta_impl);
  - the CFASTPT-vs-FASTPT comparison prints each implementation's
    theory vector at the same 30 IA points and computes
    delta^T C^-1 delta between the vectors, with C^-1 the masked
    inverse covariance from the compiled interface: the chi2 of the
    implementation DIFFERENCE, zero when the vectors agree. Each of
    the three blocks (cfastpt; python FAST-PT at the pinned low
    settings; python FAST-PT at the doubled grids) runs in its own
    fresh subprocess, so no cobaya cache, CAMB state, or C global
    survives from one block into the next
    (_fastpt_comparison_block);
  - the Halofit-vs-EE2 comparison (advisory checks NL1/NL2) applies
    the same quadratic form to two nonlinear-P(k) sources at ten
    cosmologies: at each cosmology the EuclidEmulator2 vector is the
    reference the Halofit vector is measured against, again with one
    fresh subprocess per block (_nonlinear_comparison_block).

RUNNING A PROJECT SUITE

Each project keeps its test modules in tests/data_vector/; the
shim, conftest.py, generate_frozen_reference.py, frozen/, the
manifest, and pytest.ini sit one level up, in tests/. The
pytest.ini adds -s to every run, so each test's printed report
reaches the terminal instead of a bare PASSED. From the Cocoa/
folder, with the cocoa conda environment active:

    source start_cocoa.sh
    python -m pytest ./projects/<project>/tests/data_vector

A path to one file (.../data_vector/test_fastpt.py) runs that file
alone. Two options steer the comparison sweeps (conftest_addoption):
--high=1 runs the CFASTPT-vs-FASTPT sweep at the high-accuracy
settings instead of the frozen defaults, and --mask=<name> runs the
CFASTPT-vs-FASTPT and Halofit-vs-EE2 sweeps under another scale-cut
mask the project offers. Run one project per pytest invocation:
every project names its shim cocoa_test_utils, and a python process
keeps the first module it imports under a name, so a second
project's tests would bind to the first project's shim.

A WORKED EXAMPLE: A PROJECT SHIM

A new project "demo" whose compiled interface imports as
cosmolike_demo_interface writes tests/cocoa_test_utils.py as below
(data values abbreviated; the test modules, the generator,
conftest.py and the worker subprocesses reach this module only
through the shim's names):

    import os
    import sys

    TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
    FROZEN_DIR = os.path.join(TESTS_DIR, "frozen")
    MANIFEST_FILE = os.path.join(TESTS_DIR, "manifest_sha256.json")
    REFERENCE_FILE = os.path.join(FROZEN_DIR, "reference_chi2.json")

    # path-based import so it works before start_cocoa.sh's
    # python-path setup
    _CORE_DIR = os.path.abspath(os.path.join(
        TESTS_DIR, "..", "..", "..", "external_modules", "code",
        "cosmolike_core"))
    if _CORE_DIR not in sys.path:
        sys.path.insert(0, _CORE_DIR)
    import cocoa_testing as _cct

    # ---- the project data --------------------------------------
    TATT_POINT = {"demo_A2_1": 0.05, ...}
    TATT_GENERATORS = {"tatt_demo.dataset": "example2"}
    EXAMPLES = {
        "example1": {
            "frozen_module": "frozen_config_example1.py",
            "provenance": "EXAMPLE_EVALUATE1.yaml",
            "likelihood": "demo.cosmic_shear",
            "tatt_dataset": "tatt_demo.dataset",
        },
        "example2": {...},
    }
    HIGH_ACCURACY_LIKELIHOOD = {...}
    ACCURACY_KNOBS = [...]
    FASTPT_COMPARISON_TOLERANCE = 0.2
    FASTPT_LOW_SETTINGS = {...}
    FASTPT_HIGH_SETTINGS = {...}

    # ---- project-independent constants, re-exported ------------
    REQUIRED_OMP_THREADS = _cct.REQUIRED_OMP_THREADS
    CHI2_TOLERANCE = _cct.CHI2_TOLERANCE
    RACE_TOLERANCE = _cct.RACE_TOLERANCE
    RACE_PERTURBATIONS = _cct.RACE_PERTURBATIONS
    HIGH_ACCURACY_CAMB_EXTRA_ARGS = (
        _cct.HIGH_ACCURACY_CAMB_EXTRA_ARGS)
    BARYON_METHODS = _cct.BARYON_METHODS
    BARYON_POINT_OVERRIDES = _cct.BARYON_POINT_OVERRIDES
    FASTPT_COMPARISON_POINTS = (
        _cct.fastpt_comparison_points("demo"))
    NONLINEAR_COMPARISON_POINTS = _cct.NONLINEAR_COMPARISON_POINTS

    # ---- the harness -------------------------------------------
    _H = _cct.CocoaTestHarness(
        worker_file=__file__,
        interface_module="cosmolike_demo_interface",
        examples=EXAMPLES,
        tatt_point=TATT_POINT,
        accuracy_knobs=ACCURACY_KNOBS,
        high_accuracy_likelihood=HIGH_ACCURACY_LIKELIHOOD,
        fastpt_low_settings=FASTPT_LOW_SETTINGS,
        fastpt_high_settings=FASTPT_HIGH_SETTINGS,
        fastpt_points=FASTPT_COMPARISON_POINTS,
    )

    # ---- module functions, re-exported -------------------------
    require_cocoa_environment = _cct.require_cocoa_environment
    assert_omp_threads = _cct.assert_omp_threads
    sha256_of = _cct.sha256_of
    make_model = _cct.make_model
    evaluate_chi2 = _cct.evaluate_chi2
    _evaluate_cached = _cct._evaluate_cached
    _load_datavector = _cct._load_datavector
    _baryon_method = _cct._baryon_method
    _baryon_dataset = _cct._baryon_dataset
    report_chi2_test = _cct.report_chi2_test
    report_race_test = _cct.report_race_test
    report_accuracy = _cct.report_accuracy
    report_knob = _cct.report_knob
    report_fastpt_comparison = _cct.report_fastpt_comparison
    report_nonlinear_comparison = _cct.report_nonlinear_comparison

    # ---- bound harness methods, re-exported --------------------
    compute_manifest = _H.compute_manifest
    verify_frozen = _H.verify_frozen
    load_reference = _H.load_reference
    _frozen_module = _H._frozen_module
    load_frozen_info = _H.load_frozen_info
    load_frozen_point = _H.load_frozen_point
    build_point = _H.build_point
    _single_model_chi2_impl = _H._single_model_chi2_impl
    _ten_in_a_row_impl = _H._ten_in_a_row_impl
    single_model_chi2 = _H.single_model_chi2
    ten_in_a_row_chi2 = _H.ten_in_a_row_chi2
    baryon_accuracy_delta = _H.baryon_accuracy_delta
    baryon_drift_chi2 = _H.baryon_drift_chi2
    _worker = _H._worker
    _run_isolated = _H._run_isolated
    _fastpt_comparison_info = _H._fastpt_comparison_info
    _fastpt_comparison_block = _H._fastpt_comparison_block
    _run_fastpt_comparison_worker = _H._run_fastpt_comparison_worker
    cfastpt_vs_fastpt_chi2s = _H.cfastpt_vs_fastpt_chi2s
    _nonlinear_comparison_block = _H._nonlinear_comparison_block
    halofit_vs_ee2_dchi2s = _H.halofit_vs_ee2_dchi2s

A project with one project-wide synthetic NLA vector also passes
nla_dataset="synthetic_demo.dataset" to the constructor; a project
whose EXAMPLES entries carry their own "nla_dataset" keys needs
nothing extra (the per-example key wins); a project that evaluates
NLA against its shipped data_file passes neither. A project whose
comparison sweeps offer more scale-cut masks passes, for example,
fastpt_masks=("frozen", "ones") and ships the matching frozen
dataset variants (the class docstring gives the naming rule). A
project with extra data blocks the generator reads
(SYNTHETIC_VECTORS, a NLA_DATASET constant) or project-only report
printers keeps them in the shim next to the data.

Some shim names are looked up by name from another process or file,
so they must keep their exact spelling: _worker (called by
_WORKER_DRIVER inside every worker), _fastpt_comparison_block and
_nonlinear_comparison_block (called by the child programs of the
comparison sweeps, so a project that runs those sweeps needs them),
and _cct and _H (read by the project's conftest.py; see the banner
before conftest_addoption). A missing name surfaces only where it is
looked up, as an AttributeError.

ADDING A TEST MODULE

A new test file in tests/data_vector/ copies the opening of an
existing one; the order of its first lines matters:

    import os

    # OpenMP reads OMP_NUM_THREADS once, when the compiled libraries
    # load: set it before ANY cobaya/cosmolike import
    os.environ["OMP_NUM_THREADS"] = "4"

    import sys
    import unittest

    # the shim lives one folder up, in tests/
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    import cocoa_test_utils as u

    class TestDemo(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            u.require_cocoa_environment()
            u.verify_frozen()
            cls.reference = u.load_reference()

        def test_1_chi2_matches_frozen_reference(self):
            chi2 = u.single_model_chi2("example1", tatt=False)
            ref = self.reference["example1_nla"]
            delta = u.report_chi2_test(
                1, "example1 (cosmic shear, NLA)", chi2, ref,
                u.CHI2_TOLERANCE)
            self.assertLess(
                delta, u.CHI2_TOLERANCE,
                msg=f"chi2 = {chi2:.6f} vs frozen reference "
                    f"{ref:.6f} (|delta| >= {u.CHI2_TOLERANCE})")

Each test computes its quantity through a harness method (which
spawns the worker), prints it with the matching report_* printer,
and asserts on the difference the printer returns; the printers
never assert. A race test calls u.assert_omp_threads() first.
Reference values enter frozen/reference_chi2.json only through
generate_frozen_reference.py --overwrite: a hand edit fails
verify_frozen.

MAP OF THIS FILE

  Section 1: MODULE CONSTANTS (identical in every project)
    REQUIRED_OMP_THREADS, CHI2_TOLERANCE, RACE_TOLERANCE
    RACE_PERTURBATIONS, HIGH_ACCURACY_CAMB_EXTRA_ARGS
    BARYON_METHODS, BARYON_POINT_OVERRIDES
    _baryon_method, _baryon_dataset  label helpers of BARYON_METHODS
    fastpt_comparison_points  the 30 comparison points, built for a
                              project's parameter prefix
    NONLINEAR_COMPARISON_POINTS  the ten Halofit-vs-EE2 cosmologies

  Section 2: MODULE FUNCTIONS (no project state)
    require_cocoa_environment, assert_omp_threads, sha256_of
    make_model, evaluate_chi2, _evaluate_cached, _load_datavector
    report_chi2_test, report_race_test, report_accuracy, report_knob
    report_fastpt_comparison, report_nonlinear_comparison

  Section 3: WORKER-SUBPROCESS PLUMBING (shared text)
    _WORKER_FLAG, _WORKER_DRIVER, _SEQUENCE_DRIVER
    setting sequences        SEQUENCE_OPS, SEQUENCE_EVALUATORS,
                             SEQUENCE_CHECKS (the settings changes, the
                             compared vectors and the reach checks of
                             setting_sequence_suite)

  Section 4: class CocoaTestHarness (bound to one project's spec)
    frozen-state integrity   compute_manifest, verify_frozen,
                             load_reference
    the chi2 pipeline        _frozen_module, load_frozen_info,
                             load_frozen_point, build_point,
                             _single_model_chi2_impl,
                             _ten_in_a_row_impl
    baryonic feedback        _baryon_accuracy_delta_impl,
                             _baryon_drift_chi2_impl
    worker isolation         _worker, _run_isolated
    setting sequences        _sequence_build, _sequence_worker,
                             run_sequence_leg
    test quantities          single_model_chi2, ten_in_a_row_chi2,
                             baryon_accuracy_delta, baryon_drift_chi2
    CFASTPT vs FASTPT        _fastpt_comparison_info,
                             _fastpt_comparison_block,
                             _run_fastpt_comparison_worker,
                             cfastpt_vs_fastpt_chi2s
    Halofit vs EE2           _nonlinear_comparison_block,
                             _run_nonlinear_comparison_worker,
                             halofit_vs_ee2_dchi2s
                             (points: NONLINEAR_COMPARISON_POINTS,
                             printer: report_nonlinear_comparison)

Module functions after the class: conftest_addoption and
conftest_configure, the shared implementation of every project's
conftest.py (see the section banner at the end of the file).
"""

import hashlib
import json
import os
import shutil
import tempfile


# =============================================================================
# SECTION 1: MODULE CONSTANTS (identical in every project)
# =============================================================================

# The race tests must run multi-threaded: with one thread there is no
# thread scheduling, so an OpenMP race could never show up. Every
# worker subprocess runs at this count too (_run_isolated and the
# comparison sweeps' child programs set it). The value is a string,
# not a number: environment variables only carry text.
# Four is the reproducible default. A long validation run can explicitly
# select another multi-threaded count before importing any native library.
# Project entry points read the same override before setting OMP_NUM_THREADS.
REQUIRED_OMP_THREADS = os.environ.get("COCOA_TEST_OMP_THREADS", "4")
if (not REQUIRED_OMP_THREADS.isdecimal()
        or int(REQUIRED_OMP_THREADS) < 2):
    raise ValueError(
        f"COCOA_TEST_OMP_THREADS={REQUIRED_OMP_THREADS!r}; "
        "choose an integer >= 2 so the race checks use multiple threads")

# Reference tests: |chi2(now) - chi2(frozen reference)| must stay
# below this. The bound is a chosen regression tolerance: wide enough
# to tolerate compiler and library-version noise, tight enough that a
# code or input change is unlikely to hide under it. It is NOT a
# physical detectability limit: with r the residual of the reference
# prediction against the data and e the prediction change,
#   delta chi2 = 2 r^T C^-1 e + e^T C^-1 e,
# so far from a minimum the signed cross term can cancel a large
# prediction change. The ZERO-BASED COMPARISONS of the module
# docstring are built at r = 0 exactly so that their delta chi2 IS
# the squared prediction distance e^T C^-1 e. The reports print the
# difference with more decimals than the bound needs, so a drift
# below it stays visible.
CHI2_TOLERANCE = 0.2

# Race tests: |chi2(10th of a row) - chi2(fresh model)|. The two
# numbers come from the same code on the same inputs, so only float
# noise is allowed; a state leak produces a much larger shift.
RACE_TOLERANCE = 1.0e-4

# The nine cosmologies a race test evaluates between its two
# evaluations of the fiducial point. Each entry replaces the named
# parameters in the frozen point; a name the configuration does not
# sample is skipped (_ten_in_a_row_impl), so one table serves every
# project. They stay inside the priors of the frozen configurations
# (an out-of-prior point would evaluate to -inf and abort the test),
# and they change the chi2 by orders of magnitude, so state leaked
# from any of them would visibly move the final fiducial evaluation.
RACE_PERTURBATIONS = [
    {"As_1e9": 1.95},
    {"As_1e9": 2.25},
    {"omegam": 0.28},
    {"omegam": 0.33},
    {"H0": 64.0},
    {"H0": 71.0},
    {"ns": 0.95},
    # a configuration that samples w0pwa = w + wa in place of wa
    # keeps wa = 0 when w0pwa moves together with w
    {"w": -1.1, "w0pwa": -1.1},
    {"omegab": 0.052, "mnu": 0.15},
]

# High-accuracy CAMB settings for the accuracy advisory checks: the
# same physics evaluated with the numerical knobs pushed far beyond
# the defaults. The LIKELIHOOD-side table is project data (real-space
# projects push lmax; Fourier-space projects have no lmax option), so
# it lives in each project's spec. halofit_version, dark_energy_model
# and accurate_massive_neutrino_transfers are physics choices, not
# accuracy knobs: they repeat the frozen configurations' values, so
# the update changes the numerics only.
HIGH_ACCURACY_CAMB_EXTRA_ARGS = {
    "halofit_version": "takahashi",
    "AccuracyBoost": 2.0,       # default 1.05
    "dark_energy_model": "ppf",
    "accurate_massive_neutrino_transfers": False,
    "k_per_logint": 50,         # default 10
    "kmax": 50.0,               # default 5.0-7.5
}

# ---- baryonic feedback methods (bfmt theory block) --------------------------

# One entry per feedback method the bfmt theory block implements:
# (label, theory-block options selecting it, fixed evaluation point
# for its sampled parameters). The SP(k) points are pyspk's
# documented examples; the emulator points are the fiducial values
# quoted in the example yamls. test_accuracy_baryons.py evaluates
# each method at the default and the pushed numerical settings.
BARYON_METHODS = [
    ("spk power law", {"baryon_model": 1, "spk_fb_model": 1},
     {"fb_a_spk": 0.4, "fb_pow_spk": 0.3}),
    ("spk akino", {"baryon_model": 1, "spk_fb_model": 2},
     {"alpha_spk": 4.189, "beta_spk": 1.273, "gamma_spk": 0.298}),
    # pyspk's documented double-power-law example (epsilon 0.3,
    # alpha 1.1, beta 0.2, gamma 0.5) pushes fb outside SP(k)'s
    # calibrated band at z >~ 1.4 (pyspk then returns NaN and the
    # block falls back to unity per redshift): a check with that
    # point would test the fallback, not the method. This point
    # matches the Akino relation's amplitude and mass slope at the
    # pivot and stays inside the band over the full z grid.
    ("spk double power law", {"baryon_model": 1, "spk_fb_model": 3},
     {"epsilon_spk": 0.66, "alpha_spk": 0.35, "beta_spk": 0.2,
      "gamma_spk": 0.3}),
    ("bcemu", {"baryon_model": 2},
     {"log10Mc_bcemu": 13.32, "mu_bcemu": 0.93, "thej_bcemu": 4.235,
      "gamma_bcemu": 2.25, "delta_bcemu": 6.40, "eta_bcemu": 0.15,
      "deta_bcemu": 0.14}),
    ("flamingo", {"baryon_model": 3},
     {"fgas_sigma_flamingo": 0.0, "mstar_sigma_flamingo": 0.0,
      "jet_frac_flamingo": 0.0}),
    ("baccoemu", {"baryon_model": 4},
     {"M_c_baccoemu": 14.0, "eta_baccoemu": -0.3,
      "beta_baccoemu": -0.22, "M1_z0_cen_baccoemu": 10.5,
      "theta_inn_baccoemu": -0.86}),
    ("bcemu2025", {"baryon_model": 5},
     {"Theta_co_bcemu25": 0.3, "log10Mc_bcemu25": 13.1,
      "mu_bcemu25": 1.0, "delta_bcemu25": 6.0, "eta_bcemu25": 0.10,
      "deta_bcemu25": 0.22, "Nstar_bcemu25": 0.028}),
]

# Cosmology shifts a method needs so its OWN training box contains
# the evaluation point. BACCOemu's omega_baryon floor is 0.04001,
# just above the fiducial omegab = 0.04, so its checks (and its
# generated data vector) evaluate at omegab = 0.049, inside the box
# and inside the yaml prior. Generator and checks apply the SAME
# override, so the chi2 still sits at the minimum by construction.
BARYON_POINT_OVERRIDES = {
    "baccoemu": {"omegab": 0.049},
}


def _baryon_method(label):
    """Look one BARYON_METHODS entry up by its label.

    Arguments:
      label = the first field of one BARYON_METHODS entry.

    Returns:
      the (label, theory options, parameter point) tuple.

    Raises:
      ValueError when label names no entry.
    """
    matches = [b for b in BARYON_METHODS if b[0] == label]
    if len(matches) != 1:
        raise ValueError(f"unknown baryon method {label!r}")
    return matches[0]


def _baryon_dataset(label):
    """Return the descriptor name of one feedback method's own vector.

    Arguments:
      label = a BARYON_METHODS label.

    Returns:
      the frozen/data descriptor file name, e.g.
      baryon_spk_akino.dataset for "spk akino".

    Raises:
      nothing.
    """
    return "baryon_" + label.replace(" ", "_") + ".dataset"


# ---- the CFASTPT-vs-FASTPT comparison points --------------------------------

# The comparison evaluates the SAME intrinsic-alignment points in
# every project: each project's five TATT parameters carry the same
# [-5, 5] and [0, 2] prior boxes, so the same tuples are valid
# everywhere and a divergence can be chased at one shared point.
# The scalar per-point numbers are still each project's own - the
# observables, masks and covariance weights differ - so they are
# not directly interchangeable between projects. The first 20
# points are hard-coded draws:
# drawn ONCE, uniformly across the prior boxes, with
# numpy.random.default_rng(20250922), and written out here so every
# run evaluates exactly these points. Points 21-30 are the
# one-parameter-at-a-time family that names the TATT parameter
# driving a divergence: an all-zero null, each amplitude alone at
# both signs, and the redshift powers and BTA riding on the amplitude
# that activates them (alone they are exact nulls). Each tuple is
# (A1_1, A1_2, A2_1, A2_2, BTA_1); fastpt_comparison_points attaches
# a project's parameter prefix.
_FASTPT_COMPARISON_VALUES = [
    (-2.347878, -2.631529, -4.548776, 4.089803, 1.299591),
    (-4.5757, 1.842436, -2.778512, 2.349448, 1.771619),
    (-1.885483, -4.988485, -1.485879, -4.495962, 0.884889),
    (4.467901, -3.113044, 0.788438, 4.321531, 0.595105),
    (2.925418, 2.106288, 3.514003, -4.307538, 0.106922),
    (0.737236, -2.601612, -3.525379, -2.532121, 1.551288),
    (-2.208689, 1.166122, -3.89846, -3.625951, 1.458385),
    (3.577859, -2.150368, 3.68132, -0.800727, 1.338299),
    (-3.301099, -1.899991, 2.247241, -2.085669, 1.21194),
    (-1.957003, 2.044701, 1.357407, 3.861245, 0.112379),
    (1.56864, -1.191915, 2.709754, -3.689616, 0.581821),
    (0.092758, 4.358014, -0.665741, -1.574787, 0.158704),
    (1.205006, 1.544994, -2.161392, -0.047689, 1.065058),
    (2.623534, -4.583449, 4.401911, 3.545763, 1.463577),
    (-4.459096, 3.114698, -0.526033, 2.268589, 1.739542),
    (0.206121, -4.811619, 4.734078, -2.90544, 1.16909),
    (3.56276, -0.662237, -0.174979, 4.150983, 1.275072),
    (3.593366, -4.875329, 2.680328, -3.169156, 0.538465),
    (1.290369, 0.975332, -0.422358, -2.637129, 0.641767),
    (1.835056, -1.493955, 1.138184, 4.009877, 1.228736),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (4.0, 0.0, 0.0, 0.0, 0.0),
    (-4.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 4.0, 0.0, 0.0),
    (0.0, 0.0, -4.0, 0.0, 0.0),
    (4.0, 4.0, 0.0, 0.0, 0.0),
    (4.0, -4.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 4.0, 4.0, 0.0),
    (0.0, 0.0, 4.0, -4.0, 0.0),
    (4.0, 0.0, 0.0, 0.0, 2.0),
]


def fastpt_comparison_points(prefix):
    """Return the 30 comparison points under one project's parameter prefix.

    Arguments:
      prefix = the sampled-parameter prefix of the project's five
               TATT parameters, e.g. "roman" (roman_A1_1, ...),
               "DES", or "ROMAN_KL".

    Returns:
      a list of 30 {parameter name: value} dictionaries, one per
      comparison point, index-aligned with _FASTPT_COMPARISON_VALUES.

    Raises:
      nothing.
    """
    names = [f"{prefix}_A1_1", f"{prefix}_A1_2", f"{prefix}_A2_1",
             f"{prefix}_A2_2", f"{prefix}_BTA_1"]
    return [dict(zip(names, values))
            for values in _FASTPT_COMPARISON_VALUES]


# ---- the Halofit-vs-EE2 comparison points -----------------------------------

# Advisory checks NL1/NL2 evaluate the SAME ten cosmologies with the
# two nonlinear-P(k) sources the likelihoods can consume
# (EuclidEmulator2, non_linear_emul: 1, and CAMB's Takahashi halofit,
# non_linear_emul: 2, the frozen contracts' setting) and compare the
# data vectors. The points are hard-coded draws shared by every
# project: drawn ONCE, uniformly in omegam [0.26, 0.38], ns
# [0.93, 0.99], and As_1e9 [1.8, 2.4] (a box inside the sampled
# priors and EE2's training box) with
# numpy.random.default_rng(20260923), and written out here so every
# run of every project evaluates exactly these cosmologies (the
# parameter names are cosmology-level, so no project prefix is
# involved). Every other parameter keeps the project's frozen
# fiducial value.
NONLINEAR_COMPARISON_POINTS = [
    {"omegam": 0.378775, "ns": 0.961148, "As_1e9": 2.311486},
    {"omegam": 0.285929, "ns": 0.948020, "As_1e9": 2.395017},
    {"omegam": 0.264959, "ns": 0.937756, "As_1e9": 1.960748},
    {"omegam": 0.351034, "ns": 0.972406, "As_1e9": 1.905972},
    {"omegam": 0.374796, "ns": 0.956464, "As_1e9": 2.191216},
    {"omegam": 0.355271, "ns": 0.978809, "As_1e9": 1.844406},
    {"omegam": 0.299391, "ns": 0.985365, "As_1e9": 2.345031},
    {"omegam": 0.338220, "ns": 0.953241, "As_1e9": 2.074333},
    {"omegam": 0.377881, "ns": 0.930398, "As_1e9": 2.193133},
    {"omegam": 0.291721, "ns": 0.965774, "As_1e9": 2.088667},
]


# =============================================================================
# SECTION 2: MODULE FUNCTIONS (no project state)
# =============================================================================
def require_cocoa_environment():
    """Refuse to run outside a started Cocoa shell, then move to ROOTDIR.

    start_cocoa.sh exports ROOTDIR (the absolute path of the Cocoa/
    folder) and prepares the library paths the compiled cosmolike
    interface needs. Without it, importing the likelihood would fail
    with a confusing linker error, so this check turns that failure
    into an instruction. The chdir matters because component paths in
    the frozen configuration (for example CAMB's
    ./external_modules/code/CAMB) are relative to ROOTDIR.

    Arguments:
      none.

    Returns:
      nothing; on success the process working directory is ROOTDIR.

    Raises:
      RuntimeError telling the user to activate the cocoa environment
      and source start_cocoa.sh when ROOTDIR is not exported.
    """
    if "ROOTDIR" not in os.environ:
        raise RuntimeError(
            "ROOTDIR is not set. Activate the cocoa conda environment and run "
            "`source start_cocoa.sh` from the Cocoa/ folder before running "
            "these tests."
        )
    os.chdir(os.environ["ROOTDIR"])


def assert_omp_threads():
    """Require the configured race-test thread count (four by default).

    OpenMP reads OMP_NUM_THREADS once, when the compiled library is
    first loaded, so the value must be in the environment before any
    cobaya or cosmolike import. The test modules set it at their first
    line. This check is an environment prerequisite read at call time:
    it verifies the variable's CURRENT value, but it cannot see when
    an already-loaded library read it, so a stack imported earlier
    under a different value can still pass. The stronger control is
    the worker isolation: each worker subprocess receives the variable
    in its environment before its first import (_run_isolated).

    Arguments:
      none.

    Returns:
      nothing when OMP_NUM_THREADS equals REQUIRED_OMP_THREADS.

    Raises:
      RuntimeError naming the observed value and the required one.
    """
    # .get returns None when the variable is unset, so the error can
    # show "None" rather than crash on a missing key
    observed = os.environ.get("OMP_NUM_THREADS")
    if observed != REQUIRED_OMP_THREADS:
        # !r prints the value in its python literal form: None and
        # the text '4' stay distinguishable in the message
        raise RuntimeError(
            f"OMP_NUM_THREADS={observed!r}; the race-condition tests require "
            f"OMP_NUM_THREADS={REQUIRED_OMP_THREADS} and it must be set "
            "before cobaya/cosmolike are imported."
        )


def assert_aborts(attempt):
    """Run attempt() in a forked child; require a nonzero exit status.

    The compiled layers validate their inputs by printing one line
    through the shared logger ([critical] with the function name and
    the reason) and ending the whole process with exit(1). They never
    raise a Python exception, so pytest.raises cannot observe a
    refusal - the exit would take the test runner down with it.

    assert_aborts therefore runs the call in a forked copy of the
    test process. os.fork() duplicates the process at the call site,
    so the child inherits the already-initialized interface, tables
    and test inputs at no cost; only the one malformed call runs
    there. The child must die with a nonzero exit status:

      - compiled layer: the validation prints [critical] and calls
        exit(1);
      - a Python-level check that raises instead: the uncaught
        exception also ends the child with status 1.

    Either way the parent's waitpid sees a nonzero status, so this
    check does not depend on which layer catches the bad input. A
    child that reaches os._exit(0) means the malformed input was
    ACCEPTED, and the test fails.

    Forking a process that already started OpenMP threads is safe
    here because a refused call ends before any parallel region
    starts; an accepted call could misbehave in the child, but
    acceptance is already the failure this check exists to catch.

    Arguments:
      attempt = argument-free function performing exactly one call
        with one malformed input. It runs only in the child process.

    Returns:
      nothing; an AssertionError in the parent fails the test when
      the child accepted the input (exit status 0) or died by a
      signal.
    """
    child = os.fork()
    if child == 0:
        # Child: the expected [critical] line is noise in the test
        # log, so stdout and stderr go to /dev/null before the call.
        quiet = os.open(os.devnull, os.O_WRONLY)
        os.dup2(quiet, 1)
        os.dup2(quiet, 2)
        # The child must never return into the test runner it
        # inherited from the fork: every outcome of attempt() ends in
        # os._exit, which leaves immediately without unwinding into
        # pytest. BaseException also catches SystemExit, so a
        # Python-level refusal (raise) and an accepted call both take
        # a defined exit.
        try:
            attempt()
        except BaseException:
            os._exit(1)
        os._exit(0)
    status = os.waitpid(child, 0)[1]
    # WIFEXITED is False when a signal killed the child (for example
    # a crash inside the call): that is a bug, not a refusal.
    assert os.WIFEXITED(status), \
        "malformed-input child died by a signal"
    assert os.WEXITSTATUS(status) != 0, "malformed input was accepted"


# The cosmolike C layer keeps the data-vector dimensions in C globals
# and aborts a process that initializes a second configuration with
# different dimensions (WORKER ISOLATION above). In a project whose
# suite holds more than one data-vector layout, the tests that go
# through the harness are covered by its workers, but a test that
# must build its models IN PROCESS (it reads the full data vector
# from the compiled interface, or needs several model builds to share
# the C caches) is not. The own_process decorator gives such a test a
# python process of its own, so two layouts never meet in one
# process. The variable marks the process that runs such a test's
# body.
OWN_PROCESS_FLAG = "COCOA_TESTS_OWN_PROCESS"


def own_process(test_file):
    """Decorator: run one test method in a python process of its own.

    Used as @own_process(__file__) on a unittest test method. In the
    process pytest (or unittest discover) collected the test in, the
    decorated method does not run its body: it starts a fresh python
    on the test's own file, naming that one test on the command line
    (unittest.main runs exactly the test it is given), and passes
    when the child exits with code 0. The child carries
    OWN_PROCESS_FLAG, so there the decorated method runs the body in
    process, as written. The child's report streams to the same
    terminal, so a failure prints its assertion there; the parent
    only relays the verdict.

    Arguments:
      test_file = the path of the test module (__file__ in that
                  module), the file the child process runs.

    Returns:
      the decorator to apply to the test method.

    Raises:
      nothing at decoration time; the decorated method fails with an
      AssertionError naming the test when its process exits nonzero.
    """
    # functools.wraps copies the method's name and docstring onto the
    # wrapper, so unittest and pytest still collect and report the
    # test under its own name
    import functools

    def decorator(test_method):
        """Return the replacement of test_method running it in a child.

        Arguments:
          test_method = the unittest test method being decorated.

        Returns:
          wrapper, which keeps test_method's name and docstring.
        """
        @functools.wraps(test_method)
        def wrapper(self):
            """Run the body here in the child, else start the child.

            functools.wraps replaces this docstring by the test
            method's.

            Arguments:
              self = the unittest.TestCase instance of the test.

            Returns:
              the body's result in the child process; nothing in the
              parent.

            Raises:
              AssertionError in the parent when the child exits
              nonzero.
            """
            # .get returns None when the variable is absent: only
            # the child process, which carries the flag, runs the
            # body
            if os.environ.get(OWN_PROCESS_FLAG) == "1":
                return test_method(self)
            import subprocess

            # "Class.method", the form unittest.main accepts as a
            # command-line test name inside the file it runs
            test_id = f"{type(self).__name__}.{self._testMethodName}"
            # dict(os.environ) is a COPY of the environment: the
            # edits below reach only the child process
            environment = dict(os.environ)
            environment[OWN_PROCESS_FLAG] = "1"
            # OpenMP reads this at library load inside the fresh
            # process (the test file sets it at its first line too)
            environment["OMP_NUM_THREADS"] = REQUIRED_OMP_THREADS
            print(f"  running {test_id} in its own process ...",
                  flush=True)
            # subprocess.run starts the child (sys.executable = this
            # same python) and BLOCKS until it exits
            import sys
            completed = subprocess.run(
                [sys.executable, os.path.abspath(test_file), test_id],
                env=environment)
            self.assertEqual(
                completed.returncode, 0,
                f"{test_id} failed in its own process (exit code "
                f"{completed.returncode}); its report is printed "
                "above")
        return wrapper
    return decorator


def sha256_of(path):
    """Fingerprint one file with SHA-256.

    Arguments:
      path = absolute path of the file to hash.

    Returns:
      the 64-character lowercase hexadecimal SHA-256 digest of the
      file's bytes. Reading happens in 1 MiB blocks so a frozen
      covariance (some reach hundreds of MB) never sits in memory at
      once.

    Raises:
      OSError when the file cannot be opened or read.
    """
    hasher = hashlib.sha256()
    # "rb" reads raw bytes (hashing is byte-level); the with block
    # closes the file on every exit, an exception included
    with open(path, "rb") as f:
        # the lambda is an unnamed one-line function wrapping f.read;
        # two-argument iter calls it again and again until it returns
        # b"" (end of file); 1 << 20 is 2**20 bytes = 1 MiB per read
        for block in iter(lambda: f.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def make_model(info):
    """Build a cobaya model (theory plus likelihood, ready to evaluate).

    Arguments:
      info = a cobaya input dictionary from load_frozen_info.

    Returns:
      the cobaya Model. Building one loads CAMB and the compiled
      cosmolike interface and reads the frozen data files, which takes
      a few seconds; the callers print a progress line first.

    Raises:
      whatever cobaya's get_model raises on an invalid configuration
      (the frozen strings were valid at freeze time, so a failure
      here signals an environment or code change, not a data change).
    """
    from cobaya.model import get_model

    return get_model(info)


def evaluate_chi2(model, point):
    """Evaluate one point and return the likelihood chi2.

    cached=False turns off cobaya's own component reuse, so CAMB and
    the likelihood recompute at the point: the race tests evaluate
    the same point twice on one model, and letting cobaya return a
    cached value would compare a number with itself. Tables cached
    inside the compiled layers follow their own rebuild rules; the
    race sequence exercises them by changing the cosmology between
    its two fiducial evaluations.

    Arguments:
      model = the cobaya Model to evaluate on.
      point = {parameter name: value} covering the sampled parameters.

    Returns:
      chi2 = -2 ln L of the single likelihood, as a plain float.

    Raises:
      RuntimeError when the model holds more than one likelihood
      (the -2*loglikes[0] extraction would then be ambiguous);
      AssertionError when the chi2 is not finite, which is how an
      out-of-prior or rejected point shows up.
    """
    import numpy as np

    # logposterior runs the full pipeline (theory + likelihood) at the
    # point; cached=False forces recomputation (see docstring)
    posterior = model.logposterior(point, cached=False)
    # loglikes = one ln L per likelihood component, in model order
    if len(posterior.loglikes) != 1:
        raise RuntimeError(
            f"expected exactly one likelihood: {posterior.loglikes}")
    chi2 = -2.0 * posterior.loglikes[0]
    if not np.isfinite(chi2):
        raise AssertionError(f"non-finite chi2 at point {point}")
    # float() converts the numpy scalar into a plain python float,
    # which json can store and print cleanly
    return float(chi2)


def _evaluate_cached(model, point):
    """Evaluate one point with cobaya's caching left on.

    The CFASTPT-vs-FASTPT comparison points share one cosmology and
    differ only in the five IA parameters, so letting cobaya skip
    components whose inputs did not change (CAMB, and the fastpt
    theory block) makes each point cost only the likelihood's
    data-vector rebuild. evaluate_chi2 forces cached=False because
    the race tests need full recomputation; this comparison does not.

    Arguments:
      model = the cobaya Model to evaluate on.
      point = {parameter name: value} covering the sampled
              parameters.

    Returns:
      chi2 = -2 ln L of the single likelihood, as a plain float.

    Raises:
      RuntimeError on more than one likelihood; AssertionError on a
      non-finite chi2 (an out-of-prior or rejected point).
    """
    import numpy as np

    posterior = model.logposterior(point, cached=True)
    if len(posterior.loglikes) != 1:
        raise RuntimeError(
            f"expected exactly one likelihood: {posterior.loglikes}")
    chi2 = -2.0 * posterior.loglikes[0]
    if not np.isfinite(chi2):
        raise AssertionError(f"non-finite chi2 at point {point}")
    return float(chi2)


def _load_datavector(path):
    """Read one printed theory data vector as a 1D numpy array.

    print_datavector writes one line per data point; cosmolike's
    format carries the entry index in the first column and the value
    in the second, so a two-column file is reduced to its value
    column. A one-column file is used as it is.

    Arguments:
      path = the .modelvector file a likelihood evaluation printed.

    Returns:
      the vector values as a 1D float array.

    Raises:
      RuntimeError when the file has more than two columns.
    """
    import numpy as np

    table = np.loadtxt(path)
    # ndim == 1 means one number per line: the values themselves
    if table.ndim == 1:
        return table
    if table.shape[1] == 2:
        # [:, 1] selects the second column (all rows): the values
        return table[:, 1]
    raise RuntimeError(
        f"{path}: expected 1 or 2 columns, found {table.shape[1]}")


# ---- terminal reports -------------------------------------------------------
# Each printer returns the difference it printed; the assertion on
# that difference lives in the calling test method, not here.
# The frozen FAST-PT references (the tests that evaluate TATT with the
# python FAST-PT package against their own stored chi2) were computed
# on one 1100-point grid shared by the convolutions and the output
# table. In the fastpt block's boost convention (1.0 = the converged
# two-grid configuration) that table is accuracyboost = 1/5120;
# pinning it here keeps those tests on the configuration their
# references were computed with.
FASTPT_FROZEN_SETTINGS = {
    "accuracyboost": 1.0 / 5120.0,
    "internal_accuracyboost": 1.0,
    "kmax_boltzmann": 7.5,
    "extrap_kmax": 250.0,
}


# Base of the per-model random seeds of the N-random-models accuracy
# check (CocoaTestHarness.nmodels_points): model m draws its point from
# numpy.random.default_rng(RANDOM_MODEL_SEED + m). The value 137 is
# arbitrary; what matters is that it is fixed, so every run on every
# machine evaluates the same points, and that each model has its own
# seed, so model m's point does not depend on how many models exist.
RANDOM_MODEL_SEED = 137

# The number of points of the N-random-models check. It is fixed, like
# the seed: the high-accuracy vector of every point is computed once by
# generate_frozen_reference.py --nmodels and stored in
# frozen/nmodels_<example>.npz, and the check compares the default
# settings against those stored vectors. Other points (another
# cosmology, another prior) mean regenerating the stored vectors.
NMODELS_COUNT = 50

# The EuclidEmulator2 commit that precedes the Cocoa modifications, in
# the repository EE2_URL of set_installation_options.sh (the
# commented-out EE2_GIT_COMMIT alternative there). The EE2 modification
# test builds it side by side and scores the installed, modified EE2
# against it.
EE2_ORIGINAL_COMMIT = "ff59f6683069417f6b4d2fb5d59197044d424445"

# The compatibility patch the original needs to run inside Cocoa at
# all, injected into the original-EE2 worker before the model builds
# (it replaces euclidemu2.get_boost2 in the child process only):
#   - the original has no get_boost2 (the Cocoa addition that takes a
#     pre-built PyEuclidEmulator), so the adapter maps it onto the
#     original get_boost, which builds its own emulator internally;
#   - the original computes at most 101 redshifts per call (its
#     training-grid size) and silently overflows beyond that (the
#     likelihood sends about 110), so the adapter batches the
#     redshifts in chunks of 100 and stacks the per-chunk results
#     (the original returns a per-redshift dict, the likelihood
#     expects a 2D array). The numerics inside each chunk are the
#     original's, untouched.
EE2_ORIGINAL_SHIM = """
import numpy as _np
_orig_get_boost = euclidemu2.get_boost
def _get_boost2(cosmo_par_in, redshifts, ee2_obj, custom_kvec=None):
    redshifts = _np.atleast_1d(_np.asarray(redshifts, dtype=float))
    rows = []
    k = None
    for start in range(0, len(redshifts), 100):
        k, b = _orig_get_boost(cosmo_par_in,
                               redshifts[start:start + 100],
                               custom_kvec)
        if isinstance(b, dict):
            rows.extend(_np.asarray(b[i]) for i in sorted(b))
        else:
            rows.extend(_np.asarray(b))
    return k, _np.array(rows)
euclidemu2.get_boost2 = _get_boost2
"""


def report_chi2_test(number, label, chi2, ref, tol):
    """Print one reference-comparison test as a readable block.

    A bare pytest PASSED does not say what was compared, so each test
    prints its own numbers: the freshly computed chi2, the frozen
    reference, their absolute difference, and the limit the assertion
    uses. flush=True makes the block appear immediately (pytest runs
    with -s, so nothing buffers it).

    Arguments:
      number = the test number shown in the header.
      label  = one line naming the example, probe, and IA model.
      chi2   = the chi2 computed in this run.
      ref    = the frozen reference chi2.
      tol    = the pass limit on |chi2 - ref| (CHI2_TOLERANCE).

    Returns:
      |chi2 - ref|, the printed difference.

    Raises:
      nothing.
    """
    delta = abs(chi2 - ref)
    # one multi-line f-string: '-' * 66 repeats the dash 66 times (a
    # rule), :.6f prints a fixed six decimals, and the a-if-else
    # inside the last braces picks the verdict word
    print(f"""
{'-' * 66}
TEST {number}: {label}
  chi2 (this run)     = {chi2:.6f}
  frozen reference    = {ref:.6f}
  |delta chi2|        = {delta:.6f}   (limit: < {tol})
  -> {'OK' if delta < tol else 'EXCEEDS LIMIT'}
{'-' * 66}""", flush=True)
    return delta


def report_race_test(number, label, fresh, tenth, tol):
    """Print one race-condition test as a readable block.

    Arguments:
      number = the test number shown in the header.
      label  = one line naming the example, probe, and IA model.
      fresh  = chi2 of the fiducial point evaluated first on the model.
      tenth  = chi2 of the same point evaluated as the 10th of a row.
      tol    = the pass limit on |tenth - fresh| (RACE_TOLERANCE).

    Returns:
      |tenth - fresh|, the printed difference.

    Raises:
      nothing.
    """
    delta = abs(tenth - fresh)
    # the same construct as report_chi2_test's block, at :.8f (a
    # fixed eight decimals: the race limit is 1e-4)
    print(f"""
{'-' * 66}
TEST {number}: {label}
  fresh-model chi2    = {fresh:.8f}
  10th of 10 in a row = {tenth:.8f}
  |delta chi2|        = {delta:.8f}   (limit: < {tol})
  OMP_NUM_THREADS     = {os.environ.get('OMP_NUM_THREADS')}
  -> {'OK' if delta < tol else 'EXCEEDS LIMIT'}
{'-' * 66}""", flush=True)
    return delta


def report_accuracy(label, chi2_high, default_ref,
                    default_name="default, frozen"):
    """Print one default-vs-high-accuracy check. Advisory only.

    The default-settings chi2 is the frozen reference (recorded at
    freeze time); the high-accuracy chi2 is computed in this run. The
    difference is a refinement comparison at this one point: the
    measured sensitivity of the chi2 to pushing every numerical
    setting, under the same physical model. The pushed settings are a
    finer calculation, not the continuum answer, so the difference
    estimates the defaults' numerical error rather than measuring it
    exactly.
    There is no pass/fail because how much numerical error an
    analysis tolerates is a judgment call, not a fixed bound.

    Arguments:
      label        = one line naming the probe and IA model.
      chi2_high    = chi2 with HIGH_ACCURACY settings, this run.
      default_ref  = the frozen default-settings reference chi2.
      default_name = the label printed in parentheses on the
                     default-settings line.

    Returns:
      chi2_high - default_ref, the printed difference.

    Raises:
      nothing.
    """
    delta = chi2_high - default_ref
    # ljust(28) pads each name to the same column so the = signs line
    # up whatever the default is called; :+.6f prints six decimals
    # with the sign ALWAYS shown, so a shift reads as +0.01 at a
    # glance
    line_high = "  chi2 (high accuracy)".ljust(28)
    line_default = f"  chi2 ({default_name})".ljust(28)
    line_delta = "  delta chi2 (high-default) "
    print(f"""
{'-' * 66}
ACCURACY: {label}
{line_high}= {chi2_high:.6f}
{line_default}= {default_ref:.6f}
{line_delta}= {delta:+.6f}
{'-' * 66}""", flush=True)
    return delta


def report_knob(label, chi2, default_ref):
    """Print one entry of the one-knob-at-a-time scan. Advisory only.

    Arguments:
      label       = the ACCURACY_KNOBS entry evaluated.
      chi2        = chi2 with only that knob changed, this run.
      default_ref = the frozen default-settings reference chi2.

    Returns:
      chi2 - default_ref, the printed difference.

    Raises:
      nothing.
    """
    delta = chi2 - default_ref
    # :30s pads the label to 30 characters so the columns line up;
    # :12.6f = width 12 with six decimals; the + forces the sign
    print(f"  KNOB {label:30s} chi2 = {chi2:12.6f}  "
          f"delta = {delta:+12.6f}", flush=True)
    return delta


def report_fastpt_comparison(number, label, chi2_cfastpt,
                             chi2_fastpt_low, chi2_fastpt_high,
                             dchi2_low, dchi2_high, tol):
    """Print the CFASTPT-vs-FASTPT sweep as a readable block.

    Two lines per comparison point. The first carries the raw chi2
    of each configuration against the shipped data; those values are
    large across the IA prior and are printed only as information,
    because their differences ride the local chi2 slope. The second
    line carries the tested quantities: the chi2 of each FASTPT
    vector against the CFASTPT vector at the same point. The
    fastpt-low entry is the pass/fail quantity; the doubled-grid
    entry is advisory.

    Arguments:
      number           = the test number shown in the header.
      label            = one line naming the example, probe, and the
                         camb/cosmolike settings.
      chi2_cfastpt     = the per-point chi2 list under IA_code 0.
      chi2_fastpt_low  = the list under IA_code 1 at
                         FASTPT_LOW_SETTINGS.
      chi2_fastpt_high = the list under IA_code 1 at
                         FASTPT_HIGH_SETTINGS.
      dchi2_low        = per point, the chi2 of the FASTPT(low)
                         vector against the CFASTPT vector.
      dchi2_high       = the same for FASTPT(high).
      tol              = the pass limit on max of dchi2_low.

    Returns:
      max of dchi2_low over the points, the quantity the calling
      test asserts on.

    Raises:
      nothing.
    """
    import numpy as np

    largest = float(np.max(dchi2_low))
    middle = float(np.median(dchi2_low))
    high_largest = float(np.max(dchi2_high))
    print(f"""
{'-' * 66}
TEST {number}: {label}""", flush=True)
    # zip walks the five lists in step, handing every per-point
    # quantity at once; start=1 makes the rows read 1..30
    for i, (c, fl, fh, dl, dh) in enumerate(
            zip(chi2_cfastpt, chi2_fastpt_low, chi2_fastpt_high,
                dchi2_low, dchi2_high), start=1):
        # :14.6f = six decimals in a 14-wide field so the columns
        # line up
        print(f"  point {i:2d}:  CFASTPT = {c:14.6f}   "
              f"FASTPT(low) = {fl:14.6f}   FASTPT(high) = {fh:14.6f}",
              flush=True)
        print(f"             dchi2 vs the CFASTPT vector:  "
              f"low = {dl:.6f}   high = {dh:.6f}", flush=True)
    print(f"""  max dchi2(FASTPT low vs CFASTPT)    = {largest:.6f}   \
(limit: < {tol})
  median dchi2(FASTPT low vs CFASTPT) = {middle:.6f}
  max dchi2(FASTPT high vs CFASTPT)   = {high_largest:.6f}   (advisory)
  -> {'OK' if largest < tol else 'EXCEEDS LIMIT'}
{'-' * 66}""", flush=True)
    return largest


def build_original_ee2(prefix_dir):
    """Build the pre-modification EE2 into prefix_dir, offline.

    Clones the local euclidemu2 checkout (no internet: the pinned
    clone's history carries the original commit), checks out
    EE2_ORIGINAL_COMMIT, and pip-installs it with the offline flags
    the compile scripts use plus --ignore-installed: without that
    flag pip uninstalls the same-name distribution from .local
    before installing into the prefix, breaking the environment.

    Arguments:
      prefix_dir = the directory the build is installed under (a
                temporary directory owned by the caller).

    Returns:
      the site-packages path inside prefix_dir, for PYTHONPATH.

    Raises:
      RuntimeError when the original commit is not present in the
      local clone (a shallow checkout); the caller turns this into
      a test skip rather than a failure.
    """
    import subprocess
    import sys

    clone_src = os.path.join(os.environ["ROOTDIR"], "external_modules",
                             "code", "euclidemu2")
    probe = subprocess.run(
        ["git", "-C", clone_src, "cat-file", "-e", EE2_ORIGINAL_COMMIT],
        capture_output=True)
    if probe.returncode != 0:
        raise RuntimeError(
            f"the local euclidemu2 clone does not carry commit "
            f"{EE2_ORIGINAL_COMMIT} (shallow checkout?)")
    workdir = os.path.join(prefix_dir, "src")
    subprocess.run(["git", "clone", "-q", clone_src, workdir],
                   check=True)
    subprocess.run(["git", "-C", workdir, "checkout", "-q",
                    EE2_ORIGINAL_COMMIT], check=True)
    env = dict(os.environ)
    # the compile scripts export the compiler pair; pass it through
    # the same way when present
    if "CXX_COMPILER" in env:
        env["CXX"] = env["CXX_COMPILER"]
    if "C_COMPILER" in env:
        env["CC"] = env["C_COMPILER"]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", workdir,
         "--no-dependencies", "--no-index", "--no-build-isolation",
         "--ignore-installed", f"--prefix={prefix_dir}", "--quiet"],
        check=True, env=env)
    return os.path.join(
        prefix_dir, "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages")


def report_fastpt_test(number, label, chi2, ref, cfastpt_ref, tol):
    """Print one frozen-FASTPT reference test as a readable block.

    Two numbers matter here: how far the FASTPT chi2 moved from its
    own frozen reference (the pass/fail criterion, same rule as every
    other reference test), and the difference between the FASTPT and
    cfastpt scalar chi2 values at the same point (informational).
    Both chi2 values are measured against the shipped data, so their
    difference rides the local chi2 slope: it is a difference of
    likelihood scores, not the chi2 distance between the two theory
    vectors. The comparison sweep computes that distance directly,
    as delta^T C^-1 delta of the vectors (_fastpt_comparison_block).

    Arguments:
      number      = the test number shown in the header.
      label       = one line naming the example and IA model.
      chi2        = the FASTPT chi2 computed in this run.
      ref         = the frozen FASTPT reference chi2.
      cfastpt_ref = the frozen cfastpt (IA_code: 0) reference chi2.
      tol         = the pass limit on |chi2 - ref| (CHI2_TOLERANCE).

    Returns:
      |chi2 - ref|, the printed pass/fail difference.
    """
    delta = abs(chi2 - ref)
    # :+.6f = fixed six decimals with the + forcing a sign, so the
    # direction of the FASTPT-minus-CFASTPT difference always shows
    print(f"""
{'-' * 66}
TEST {number}: {label}
  chi2 (this run, FASTPT)  = {chi2:.6f}
  frozen FASTPT reference  = {ref:.6f}
  |delta chi2|             = {delta:.6f}   (limit: < {tol})
  frozen CFASTPT reference = {cfastpt_ref:.6f}
  FASTPT - CFASTPT         = {chi2 - cfastpt_ref:+.6f}   (informational)
  -> {'OK' if delta < tol else 'EXCEEDS LIMIT'}
{'-' * 66}""", flush=True)
    return delta


# The most draws the amplitude cut of nmodels_points may spend
# on one model before it stops: a box of which the cut keeps at
# least 0.1% needs far fewer, so reaching this ceiling means the cut
# excludes almost the whole box (a cut below the fiducial ratio of 1,
# or a typo)
NMODELS_MAX_DRAWS = 100000

# Every project names its linear galaxy-bias parameters
# <SURVEY>_B1_<bin> (LSST_B1_1, DESI_B1_3, ...), and its amplitude
# As_1e9 (10^9 A_s)
LINEAR_BIAS_TAG = "_B1_"
AMPLITUDE_NAME = "As_1e9"


def largest_amplitude_ratio(point, fiducial):
    """Return the largest clustering-amplitude ratio of a point.

    For linear, density-only galaxy clustering at fixed shape, the
    power of lens bin i is proportional to A_s b_i^2 (A_s the
    primordial amplitude, b_i the bin's linear bias), so a point's
    amplitude relative to the fiducial is
    A_i = (A_s/A_s,fid)(b_i/b_i,fid)^2. IF an entire error vector
    scaled by one common factor A_i at fixed covariance, its delta
    chi2 would grow by A_i^2. Velocity terms, other probes,
    nonlinear shape changes and unequal per-bin factors break that
    single-factor picture, so A_i is a diagnostic proxy for sorting
    a scan, not a decomposition of each error's cause.

    Arguments:
      point    = {parameter name: value}.
      fiducial = the frozen {parameter name: value} point.

    Returns:
      the largest A_i over the linear biases (LINEAR_BIAS_TAG) as a
      float, or None when the point samples no bias or no amplitude.
    """
    if AMPLITUDE_NAME not in point:
        return None
    as_ratio = point[AMPLITUDE_NAME]/fiducial[AMPLITUDE_NAME]
    largest = None
    for name in point:
        if LINEAR_BIAS_TAG not in name:
            continue
        amplitude = as_ratio*(point[name]/fiducial[name])**2
        if largest is None or amplitude > largest:
            largest = amplitude
    return largest


def held_parameter(name, hold):
    """Return whether the N-random-models hold list covers one parameter.

    An entry ending in "_" is a name prefix (<SURVEY>_B1_ covers every
    <SURVEY>_B1_<bin>); any other entry is an exact name.

    Arguments:
      name = a sampled parameter name.
      hold = the hold list, a tuple of names and prefixes.

    Returns:
      True when an entry covers the name.
    """
    for entry in hold:
        if entry.endswith("_"):
            if name.startswith(entry):
                return True
        elif name == entry:
            return True
    return False


def report_random_model(index, n_models, point, chi2_high):
    """Print one N-random-models block. Advisory only.

    The delta is (v_high - v_default)^T C^-1 (v_high - v_default),
    with v_high the stored high-accuracy vector of the point and
    v_default the vector at the default settings, both masked, and C
    the frozen covariance. The three printed parameters locate the
    point inside the prior at a glance; the full point is
    reproducible from RANDOM_MODEL_SEED + index.

    Arguments:
      index     = the model counter, 0-based (printed 1-based).
      n_models  = how many models the run evaluates in total.
      point     = the drawn {parameter name: value} point.
      chi2_high = the delta chi2 of the point.

    Returns:
      chi2_high, the printed delta.
    """
    # inside the triple-quoted f-string, a backslash at the end of a
    # source line eats the newline: As_1e9, omegam, and ns print on
    # one line
    print(f"""
{'-' * 66}
RANDOM MODEL {index + 1}/{n_models}
  As_1e9 = {point['As_1e9']:.6f}   omegam = {point['omegam']:.6f}   \
ns = {point['ns']:.6f}
  delta chi2 = {chi2_high:+.6f}
{'-' * 66}""", flush=True)
    return chi2_high


def report_random_model_summary(deltas):
    """Print the spread of the N-random-models deltas. Advisory only.

    The median rather than the mean: one point near a prior edge can
    carry a delta far above the rest, and the median keeps the
    typical numerical error readable next to that outlier (which the
    max still shows).

    Arguments:
      deltas = the per-model delta chi2 list of the
               N-random-models check, one float per model.

    Returns:
      (minimum, median, maximum) of the deltas as floats.
    """
    import numpy as np

    lowest = float(np.min(deltas))
    middle = float(np.median(deltas))
    highest = float(np.max(deltas))
    print(f"  RANDOM MODELS SUMMARY ({len(deltas)} models): delta chi2 "
          f"min = {lowest:+.6f}  median = {middle:+.6f}  "
          f"max = {highest:+.6f}", flush=True)
    return lowest, middle, highest


def report_emul2_advisory(label, chi2, frozen_ref, exact_ref, limit):
    """Print one EMUL2 accuracy check: measurements and an agreement verdict.

    There is no pass/fail here. An emulator is an approximation, so
    the useful outputs are the numbers themselves: the change against
    the frozen emulator reference (nonzero: the installed emulator no
    longer reproduces the chi2 it gave at freeze time), the
    difference against the non-emulated reference calculation at the
    same cosmology, and a verdict saying whether that difference
    stays below the threshold. The non-emulated counterpart is itself
    a numerical model (the frozen pipeline without the emulator), so
    the number measures agreement between two calculations at this
    one configuration, mask and covariance - not a general accuracy
    certificate - and the printed verdict claims agreement or
    disagreement at this configuration only. Emulated configuration
    and non-emulated counterpart both evaluate the counterpart's
    synthetic NLA vector (see load_frozen_info), and the
    counterpart's own reference is 0.000000 there by construction, so
    |emulator - non-emulated| is the disagreement at the same data
    and nothing else.

    Arguments:
      label      = one line naming the emulated configuration.
      chi2       = the emulator chi2 computed in this run.
      frozen_ref = the frozen emulator reference chi2.
      exact_ref  = the non-emulated reference chi2 (from the matching
                   example's frozen reference).
      limit      = the threshold on |chi2 - exact_ref| that separates
                   the two printed verdicts.

    Returns:
      |chi2 - exact_ref|, the difference the printed verdict reports.
    """
    drift = chi2 - frozen_ref
    delta_exact = abs(chi2 - exact_ref)
    if delta_exact < limit:
        verdict = f"agrees at this configuration (|delta chi2| < {limit})"
    else:
        verdict = f"does not agree at this configuration (|delta chi2| >= {limit})"
    # one multi-line f-string; the drift prints at :+.6f (fixed six
    # decimals, sign always shown)
    print(f"""
{'-' * 66}
EMUL2 ADVISORY: {label}
  chi2 (this run, emulator)   = {chi2:.6f}
  frozen emulator reference   = {frozen_ref:.6f}  (drift {drift:+.6f})
  non-emulated reference      = {exact_ref:.6f}
  |emulator - non-emulated|   = {delta_exact:.6f}   (threshold: {limit})
  -> {verdict}
{'-' * 66}""", flush=True)
    return delta_exact


def report_emul2_race(label, fresh, tenth):
    """Print one EMUL2 race check, advisory only.

    Arguments:
      label = one line naming the emulated configuration.
      fresh = chi2 of the point evaluated first on the model.
      tenth = chi2 of the same point as the 10th of a row.

    Returns:
      |tenth - fresh|, the printed difference. A value above
      RACE_TOLERANCE is flagged as a possible race or state leak, but
      nothing fails: the emulator checks only alert.
    """
    delta = abs(tenth - fresh)
    # the a-if-else expression picks the note from the comparison
    note = ("consistent" if delta < RACE_TOLERANCE
            else "WARNING: possible race condition or state leak")
    print(f"""
{'-' * 66}
EMUL2 ADVISORY: {label}
  fresh-model chi2    = {fresh:.8f}
  10th of 10 in a row = {tenth:.8f}
  |delta chi2|        = {delta:.8f}   ({note})
  OMP_NUM_THREADS     = {os.environ.get('OMP_NUM_THREADS')}
{'-' * 66}""", flush=True)
    return delta


def report_nonlinear_comparison(label, dchi2s):
    """Print the Halofit-vs-EE2 sweep as a readable block (advisory).

    One row per cosmology with its drawn parameters and the chi2 of
    the Halofit vector against the EE2 vector, then the max and the
    median. There is no pass limit: the caller only checks that
    every cosmology produced a number.

    Arguments:
      label  = one line naming the example, probe, and mask.
      dchi2s = the per-cosmology list from halofit_vs_ee2_dchi2s,
               index-aligned with NONLINEAR_COMPARISON_POINTS.

    Returns:
      nothing; the report goes to stdout.
    """
    import numpy as np

    print(f"\n==== {label} ====", flush=True)
    for i, (cosmology, d) in enumerate(
            zip(NONLINEAR_COMPARISON_POINTS, dchi2s), start=1):
        print(f"  cosmology {i:2d}: omegam = {cosmology['omegam']:.6f}"
              f"  ns = {cosmology['ns']:.6f}"
              f"  As_1e9 = {cosmology['As_1e9']:.6f}"
              f"   dchi2(HALOFIT vs EE2) = {d:.4f}", flush=True)
    print(f"  max dchi2    = {max(dchi2s):.4f}   (advisory; no pass "
          "limit)", flush=True)
    print(f"  median dchi2 = {float(np.median(dchi2s)):.4f}",
          flush=True)


# =============================================================================
# SECTION 3: WORKER-SUBPROCESS PLUMBING (shared text)
# =============================================================================
# One flag separates the two roles: the parent process (pytest or the
# generator) spawns workers; a process carrying this environment
# variable IS a worker and runs the physics in-process.
_WORKER_FLAG = "COCOA_TESTS_WORKER"

# The driver handed to `python -c` inside the worker: load the
# project's cocoa_test_utils from its file path (which builds its
# harness and re-exports _worker) and call _worker with the seven
# command-line arguments. The * in _worker(*sys.argv[2:9]) spreads
# the seven-element slice into seven separate arguments.
_WORKER_DRIVER = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location("
    "'cocoa_test_utils_worker', sys.argv[1])\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(module)\n"
    "module._worker(*sys.argv[2:9])\n"
)

# The setting-sequence worker's driver: the same file-path import, then
# _sequence_worker(spec file, result file).
_SEQUENCE_DRIVER = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location("
    "'cocoa_test_utils_worker', sys.argv[1])\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(module)\n"
    "module._sequence_worker(sys.argv[2], sys.argv[3])\n"
)


# ---- setting sequences: the named settings changes --------------------------
#
# A setting sequence (setting_sequence_suite in cocoa_testing_suites.py)
# changes cosmolike settings in one process and compares the result with
# a process that saw only the final settings. Each operation below is one
# settings change a notebook can make between evaluations. It receives
# the worker's context dictionary (ci = the compiled interface, like =
# the likelihood object, model, point, harness) and keyword arguments
# read from the sequence's json description. Every array crossing into
# the compiled interface is a numpy array: carma turns a plain list of
# more than 16 entries into zeros.

def _sequence_nz_table(ctx, kind, nbin):
    """The likelihood's n(z) table cut or extended to nbin bins.

    The table has the z grid in column 0 and one n(z) per bin after it
    (read_nz_sample's layout). Extra bins repeat the last bin's n(z), a
    valid sample with a known support.

    Arguments:
      ctx  = the worker context (ci, like).
      kind = "lens" or "source".
      nbin = the number of bins of the returned table.

    Returns:
      a numpy array of shape (n_z, nbin + 1).
    """
    import numpy as np
    like = ctx["like"]
    tables = ctx["ci"].read_redshift_distributions(
        lens_multihisto_file=like.lens_file,
        lens_ntomo=int(like.lens_ntomo),
        source_multihisto_file=like.source_file,
        source_ntomo=int(like.source_ntomo))
    if kind == "lens":
        table = np.array(tables[0], dtype=float)
    else:
        table = np.array(tables[1], dtype=float)
    loaded = table.shape[1] - 1
    if nbin <= loaded:
        return np.ascontiguousarray(table[:, :nbin + 1])
    columns = [table]
    for _ in range(nbin - loaded):
        columns.append(table[:, -1:])
    return np.ascontiguousarray(np.hstack(columns))


def _sequence_op_source_bins(ctx, nbin, recount=True):
    """Install nbin source bins with zero shear calibration and photo-z.

    Arguments:
      ctx     = the worker context.
      nbin    = the new number of source bins.
      recount = True also calls init_ntomo_powerspectra (the pair
                counts follow the new bins).
    """
    import numpy as np
    ci = ctx["ci"]
    ci.init_source_sample_size(int(nbin))
    ci.set_source_sample(_sequence_nz_table(ctx, "source", int(nbin)))
    ci.set_nuisance_shear_calib(M=np.zeros(int(nbin)))
    ci.set_nuisance_shear_photoz(bias=np.zeros(int(nbin)))
    if recount:
        ci.init_ntomo_powerspectra()


def _sequence_op_lens_bins(ctx, nbin, b1=1.5, recount=True):
    """Install nbin lens bins with bias b1, no other bias and no photo-z.

    Arguments:
      ctx     = the worker context.
      nbin    = the new number of lens bins.
      b1      = the linear bias of every bin.
      recount = True also calls init_ntomo_powerspectra.
    """
    import numpy as np
    ci = ctx["ci"]
    n = int(nbin)
    ci.init_lens_sample_size(n)
    ci.set_lens_sample(_sequence_nz_table(ctx, "lens", n))
    ci.set_nuisance_bias(B1=np.full(n, float(b1)), B2=np.zeros(n),
                         B_MAG=np.zeros(n), B3nl=np.zeros(n),
                         BK=np.zeros(n))
    try:
        ci.set_nuisance_clustering_photoz(bias=np.zeros(n))
    except TypeError:
        # some interfaces bind the setter with a second required
        # argument, the photo-z stretch (a width factor per bin, 1 =
        # unstretched); calling their binding without it raises
        # TypeError before any C++ runs, so the retry below is safe
        ci.set_nuisance_clustering_photoz(bias=np.zeros(n),
                                          stretch=np.ones(n))
    if hasattr(ci, "set_point_mass"):
        ci.set_point_mass(PMV=np.zeros(n))
    if recount:
        ci.init_ntomo_powerspectra()


def _sequence_op_recount(ctx):
    """Recount the tomographic pairs (init_ntomo_powerspectra)."""
    ctx["ci"].init_ntomo_powerspectra()


def _sequence_op_ggl_exclude(ctx, pairs):
    """Replace the excluded (lens, source) pairs.

    Arguments:
      ctx   = the worker context.
      pairs = a list of [lens bin, source bin] pairs.
    """
    import numpy as np
    flat = np.array(pairs, dtype=np.int32).flatten()
    ctx["ci"].init_ggl_exclude(flat)


def _sequence_op_call(ctx, name, **kwargs):
    """Call one interface setter by name with keyword arguments.

    Lists among the arguments become numpy arrays (integers stay
    integers), so long lists cross the pybind boundary intact.

    Arguments:
      ctx      = the worker context.
      name     = the interface function, e.g. "init_IA".
      **kwargs = its keyword arguments.
    """
    import numpy as np
    arguments = {}
    for key, value in kwargs.items():
        if isinstance(value, list):
            value = np.array(value)
        arguments[key] = value
    getattr(ctx["ci"], name)(**arguments)


def _sequence_op_init_data(ctx, mask="frozen", drop=0):
    """Reload the covariance, mask and data vector (init_data).

    Arguments:
      ctx  = the worker context.
      mask = "frozen" for the likelihood's mask file, or a file name in
             the same folder.
      drop = the number of leading kept entries to mask in addition, so
             two calls can load two different masks.
    """
    import numpy as np
    like = ctx["like"]
    mask_file = like.mask_file
    if mask != "frozen":
        mask_file = os.path.join(os.path.dirname(like.mask_file), mask)
    if drop > 0:
        table = np.loadtxt(mask_file)
        kept = np.flatnonzero(table[:, 1] > 0)
        table[kept[:int(drop)], 1] = 0.0
        mask_file = os.path.join(ctx["scratch"], f"mask_drop{int(drop)}.mask")
        np.savetxt(mask_file, table, fmt=["%d", "%.1f"])
    ci = ctx["ci"]
    extended = (hasattr(ci, "compute_data_vector_fourier_with_gg_cross_sizes")
                and int(ci.compute_data_vector_fourier_with_gg_cross_sizes()[3]) > 0)
    if extended:
        ci.init_data_fourier_gg_cross(like.cov_file, mask_file,
                                      like.data_vector_file)
    elif hasattr(ci, "init_data_fourier") and ctx["space"] == "fourier":
        ci.init_data_fourier(like.cov_file, mask_file, like.data_vector_file)
    else:
        ci.init_data_real(like.cov_file, mask_file, like.data_vector_file)


def _sequence_op_baryons(ctx, sim, allsims):
    """Contaminate P(k) with one baryon scenario of an HDF5 library.

    Arguments:
      ctx     = the worker context.
      sim     = the scenario label, e.g. "antilles-2".
      allsims = the library file name, in the folder of the likelihood's
                mask file (the dataset folder).
    """
    folder = os.path.dirname(ctx["like"].mask_file)
    ctx["ci"].init_baryons_contamination(
        sim=sim, allsims=os.path.join(folder, allsims))


def _sequence_op_ia_model(ctx, model):
    """Switch the intrinsic-alignment model, keeping the likelihood's
    redshift evolution and FAST-PT code (init_IA).

    Arguments:
      ctx   = the worker context.
      model = 0 (NLA) or 1 (TATT).
    """
    like = ctx["like"]
    ctx["ci"].init_IA(ia_model=int(model),
                      ia_redshift_evolution=int(like.IA_redshift_evolution),
                      ia_code=int(like.IA_code))


def _sequence_op_bias_b1(ctx, code=None):
    """Replace the b1 evolution code of the likelihood's bias model.

    Arguments:
      ctx  = the worker context.
      code = the new b1 code; None restores the likelihood's own model.
    """
    import numpy as np
    model = np.array(ctx["like"].bias_model, dtype=float)
    if code is not None:
        model[0] = float(code)
    ctx["ci"].init_bias(bias_model=model)


def _sequence_op_halve_binning(ctx):
    """Keep every other angular bin or band center (a new layout)."""
    import numpy as np
    like = ctx["like"]
    if hasattr(like, "band_ells"):
        # every other center; a contiguous copy (carma refuses a strided view)
        ells = np.ascontiguousarray(np.array(like.band_ells, dtype=float)[::2])
        ctx["ci"].init_binning_fourier_ells(
            ells=ells, lmax_shear=int(np.ceil(ells[-1])))
    elif hasattr(like, "ncl"):
        # log-spaced bands: (count, l_min, l_max, l_max of shear), under
        # the two attribute spellings the Fourier likelihoods use
        lmin = getattr(like, "l_min", getattr(like, "lmin", None))
        lmax = getattr(like, "l_max", getattr(like, "lmax", None))
        lmax_shear = getattr(like, "l_max_shear",
                             getattr(like, "lmax_shear", None))
        ctx["ci"].init_binning(int(like.ncl)//2, int(lmin), int(lmax),
                               int(lmax_shear))
    else:
        ctx["ci"].init_binning(int(like.ntheta)//2, like.theta_min_arcmin,
                               like.theta_max_arcmin)


def _sequence_op_cluster_ytransform(ctx):
    """Flip the cluster-lensing Y transform, keeping the likelihood's
    other cluster model choices (init_cluster_model).

    Arguments:
      ctx = the worker context.
    """
    like = ctx["like"]
    current = int(getattr(like, "cluster_ytransform", 1))
    ctx["ci"].init_cluster_model(
        mor_model=0,
        kernel_mode=int(getattr(like, "cluster_kernel_mode", 0)),
        selection_model=int(like.cluster_selection_model),
        ytransform=1 - current,
        include_ia=int(getattr(like, "cluster_include_ia", 1)),
        magnification=float(getattr(like, "cluster_magnification", -2.0)))


def _sequence_op_scale_band_centers(ctx, factor):
    """Move every Fourier band center by one factor (the same count).

    Arguments:
      ctx    = the worker context.
      factor = the multiplier of the likelihood's band centers.
    """
    import numpy as np
    ells = float(factor)*np.array(ctx["like"].band_ells, dtype=float)
    ctx["ci"].init_binning_fourier_ells(
        ells=ells, lmax_shear=int(np.ceil(ells[-1])))


def _sequence_op_rebuild(ctx, example, overrides=None):
    """Build a second likelihood in the same process and evaluate it.

    The likelihood's initialize runs initial_setup and its own setters,
    as a notebook building a second model does.

    Arguments:
      ctx       = the worker context; model, like and point are replaced.
      example   = the key of EXAMPLES to build.
      overrides = optional {likelihood key: value}.
    """
    ctx["harness"]._sequence_build(ctx, example, overrides)
    evaluate_chi2(ctx["model"], ctx["point"])


SEQUENCE_OPS = {
    "source_bins": _sequence_op_source_bins,
    "lens_bins": _sequence_op_lens_bins,
    "recount": _sequence_op_recount,
    "ggl_exclude": _sequence_op_ggl_exclude,
    "call": _sequence_op_call,
    "init_data": _sequence_op_init_data,
    "baryons": _sequence_op_baryons,
    "ia_model": _sequence_op_ia_model,
    "bias_b1": _sequence_op_bias_b1,
    "halve_binning": _sequence_op_halve_binning,
    "cluster_ytransform": _sequence_op_cluster_ytransform,
    "scale_band_centers": _sequence_op_scale_band_centers,
    "rebuild": _sequence_op_rebuild,
}


# ---- setting sequences: what is compared ------------------------------------

# Multipoles of the Fourier spectra evaluators: integers and band-like
# centers from the non-Limber range up to the Limber range.
SEQUENCE_ELLS = (2.0, 10.5, 30.0, 75.5, 140.0, 212.5, 400.0, 1000.0)


def _sequence_masked(ctx):
    """The masked theory vector the likelihood scores (data-vector space).

    Returns:
      a numpy array, the masked vector of the native space.
    """
    import numpy as np
    ci = ctx["ci"]
    # a single-space interface binds the generic name to its own space; a
    # dual-space one keeps it on the real layout and adds Fourier twins
    if (ctx["space"] == "fourier"
            and hasattr(ci, "compute_data_vector_fourier_masked")):
        if (hasattr(ci, "compute_data_vector_fourier_with_gg_cross_sizes")
                and int(ci.compute_data_vector_fourier_with_gg_cross_sizes()[3]) > 0):
            return np.array(ci.compute_data_vector_fourier_gg_cross_masked())
        return np.array(ci.compute_data_vector_fourier_masked())
    return np.array(ci.compute_data_vector_masked())


def _sequence_cluster(ctx):
    """The masked theory vector of the joint cluster analysis."""
    import numpy as np
    return np.array(ctx["ci"].compute_data_vector_cluster_masked())


def _sequence_sizes(ctx):
    """The probe-block sizes of the current layout (the pair counts)."""
    import numpy as np
    ci = ctx["ci"]
    if (ctx["space"] == "fourier"
            and hasattr(ci, "compute_data_vector_3x2pt_fourier_sizes")):
        return np.array(ci.compute_data_vector_3x2pt_fourier_sizes(),
                        dtype=float)
    return np.array(ci.compute_data_vector_3x2pt_real_sizes(), dtype=float)


def _sequence_chi2(ctx):
    """The masked vector followed by its chi2 against the loaded data."""
    import numpy as np
    vector = _sequence_masked(ctx)
    chi2 = ctx["ci"].compute_chi2(vector)
    return np.append(vector, float(chi2))


def _sequence_spectra(name):
    """An evaluator returning one family of tomographic spectra, flattened.

    Arguments:
      name = the interface function: xi_pm_tomo, w_gammat_tomo, w_gg_tomo,
             w_gk_tomo, w_ks_tomo (real space, no arguments), or
             C_ss_tomo_limber, C_gs_tomo_limber, C_gg_tomo_limber,
             C_gg_tomo, C_gk_tomo_limber, C_ks_tomo_limber (at
             SEQUENCE_ELLS).

    Returns:
      a function of the worker context returning a flat numpy array.
    """
    def evaluate(ctx):
        import numpy as np
        function = getattr(ctx["ci"], name)
        if name.startswith("C_"):
            result = function(np.array(SEQUENCE_ELLS))
        else:
            result = function()
        parts = result if isinstance(result, tuple) else (result,)
        flat = []
        for part in parts:
            flat.append(np.array(part, dtype=float).flatten())
        return np.concatenate(flat)
    return evaluate


SEQUENCE_EVALUATORS = {
    "masked": _sequence_masked,
    "chi2": _sequence_chi2,
    "sizes": _sequence_sizes,
    "cluster": _sequence_cluster,
}
for _name in ("xi_pm_tomo", "w_gammat_tomo", "w_gg_tomo", "w_gk_tomo",
              "w_ks_tomo", "C_ss_tomo_limber", "C_gs_tomo_limber",
              "C_gg_tomo_limber", "C_gg_tomo", "C_gk_tomo_limber",
              "C_ks_tomo_limber"):
    SEQUENCE_EVALUATORS[_name] = _sequence_spectra(_name)


# ---- setting sequences: reach checks ----------------------------------------

def _sequence_check_band_centers(ctx, low, high):
    """Raise unless a Fourier band center lies in [low, high).

    A multipole cap raised from low to high changes only band centers
    between the two caps; without one, the sequence never reaches the
    storage it tests.
    """
    import numpy as np
    ells = np.array(ctx["like"].band_ells, dtype=float)
    inside = (ells >= low) & (ells < high)
    if not np.any(inside):
        raise AssertionError(
            f"no band center in [{low}, {high}): the band centers run from "
            f"{ells.min()} to {ells.max()}; choose caps around a band center")


SEQUENCE_CHECKS = {
    "band_centers": _sequence_check_band_centers,
}


# =============================================================================
# SECTION 4: THE PROJECT-BOUND HARNESS
# =============================================================================
class CocoaTestHarness:
    """The shared test machinery, bound to one project's data.

    A project's tests/cocoa_test_utils.py builds ONE instance and
    re-exports its bound methods under the names the test modules
    already import, so the project file reduces to its data plus the
    binding.

    Constructor arguments (the project spec):
      worker_file      = the project's cocoa_test_utils.py path
                         (__file__ in that module); the tests/ paths
                         and the worker-subprocess target derive from
                         it.
      interface_module = the compiled interface's import name, e.g.
                         "cosmolike_roman_real_interface" (the
                         CFASTPT-vs-FASTPT comparison reads the
                         masked inverse covariance from it).
      examples         = the project's EXAMPLES table, {name: entry}.
                         The harness reads these entry keys:
                         "frozen_module" (the configuration's file in
                         frozen/), "likelihood" (the cobaya likelihood
                         component name), "tatt_dataset" (the data
                         descriptor of the TATT variant, needed by
                         tatt=True and by a non-frozen mask), the
                         optional "nla_dataset" (a synthetic NLA
                         descriptor), and "emulator": True with
                         "exact_example" (the non-emulated entry
                         whose ggl_exclude layout an emulator entry
                         copies; see load_frozen_info). Other keys
                         serve the generator and the test modules.
      tatt_point       = the project's TATT_POINT, {parameter name:
                         value} replacing the frozen point's TATT
                         values.
      accuracy_knobs   = the project's ACCURACY_KNOBS list of (label,
                         likelihood overrides, camb extra_args
                         overrides) tuples, each evaluated alone.
      high_accuracy_likelihood = the project's likelihood-side
                         high-accuracy settings, {option: value},
                         applied together with
                         HIGH_ACCURACY_CAMB_EXTRA_ARGS.
      fastpt_low_settings, fastpt_high_settings = the comparison's
                         pinned fastpt extra_args tables.
      fastpt_points    = the comparison points under the project's
                         parameter prefix (fastpt_comparison_points).
      nla_dataset      = None, or one synthetic-NLA dataset name used
                         by EVERY example (a project whose examples
                         name their own carries "nla_dataset" keys in
                         EXAMPLES instead, which take precedence).
      fastpt_masks     = the scale-cut masks the CFASTPT-vs-FASTPT
                         comparison can run under (the --mask option
                         of the tests). "frozen" (always first) keeps
                         each example's own tatt_dataset; every other
                         name selects the frozen dataset variant
                         "<tatt_dataset stem>_<name>.dataset", a copy
                         of that descriptor whose mask_file line
                         names the chosen mask (e.g. "ones" for the
                         no-scale-cuts all-ones mask). The default
                         offers "frozen" alone.
    """

    def __init__(self, worker_file, interface_module, examples,
                 tatt_point, accuracy_knobs, high_accuracy_likelihood,
                 fastpt_low_settings, fastpt_high_settings,
                 fastpt_points, nla_dataset=None,
                 fastpt_masks=("frozen",)):
        """Bind the shared machinery to one project's data.

        Arguments:
          see the class docstring, which documents every constructor
          argument once, next to the project spec it belongs to.

        Returns:
          nothing; the instance carries the spec plus the tests/
          paths derived from worker_file.

        Raises:
          nothing.
        """
        # Everything the tests read or write lives relative to the
        # project's tests folder, so the suite works no matter which
        # directory pytest is launched from.
        self.worker_file = os.path.abspath(worker_file)
        self.tests_dir = os.path.dirname(self.worker_file)
        self.frozen_dir = os.path.join(self.tests_dir, "frozen")
        self.manifest_file = os.path.join(self.tests_dir,
                                          "manifest_sha256.json")
        self.reference_file = os.path.join(self.frozen_dir,
                                           "reference_chi2.json")
        self.interface_module = interface_module
        self.examples = examples
        self.tatt_point = tatt_point
        self.accuracy_knobs = accuracy_knobs
        self.high_accuracy_likelihood = high_accuracy_likelihood
        self.fastpt_low_settings = fastpt_low_settings
        self.fastpt_high_settings = fastpt_high_settings
        self.fastpt_points = fastpt_points
        self.nla_dataset = nla_dataset
        self.fastpt_masks = tuple(fastpt_masks)

    # ---- frozen-state integrity ---------------------------------------------
    def compute_manifest(self):
        """Hash every file currently under tests/frozen/.

        __pycache__ folders and .pyc files are skipped: Python writes
        them as a side effect of importing the frozen modules, so
        hashing them would make the manifest fail after the first
        run. .DS_Store files (macOS Finder metadata) are skipped for
        the same reason.

        Arguments:
          none.

        Returns:
          a dictionary {relative path: sha256 digest}, with paths
          relative to the tests/ folder using "/" separators, sorted
          by path so the manifest file is stable across platforms.

        Raises:
          OSError when a frozen file cannot be read.
        """
        files = {}
        # os.walk visits every folder under frozen/, handing back the
        # folder path, its subfolder names, and its file names
        for base, dirs, names in os.walk(self.frozen_dir):
            # the comprehension keeps every subfolder name except
            # __pycache__; assigning through dirs[:] rewrites
            # os.walk's own list in place, which stops the walk from
            # entering the dropped folders (a plain dirs = ... would
            # not)
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in sorted(names):
                if name == ".DS_Store" or name.endswith(".pyc"):
                    continue
                full = os.path.join(base, name)
                rel = os.path.relpath(full, self.tests_dir).replace(
                    os.sep, "/")
                files[rel] = sha256_of(full)
        # sorted(files.items()) orders the (path, digest) pairs by
        # path; dict() rebuilds the table in that order, so the
        # manifest json written from it never reshuffles between runs
        return dict(sorted(files.items()))

    def verify_frozen(self):
        """Fail every test up front when the frozen state was edited.

        Compares the stored manifest with a fresh hash of
        tests/frozen/ in both directions, so an edited file
        (CHANGED), a deleted file (MISSING), and a new file (EXTRA)
        are all reported. This runs before any model is built: a
        tampered frozen state must not produce a plausible-looking
        chi2.

        Arguments:
          none.

        Returns:
          nothing when every frozen file matches the manifest.

        Raises:
          AssertionError listing every mismatched path and pointing
          to generate_frozen_reference.py --overwrite for a
          deliberate refresh; AssertionError also when the manifest
          file itself is absent (the frozen state was never
          generated).
        """
        if not os.path.isfile(self.manifest_file):
            raise AssertionError(
                "tests/manifest_sha256.json is missing; run "
                "generate_frozen_reference.py --overwrite to create the "
                "frozen test state."
            )
        # expected = the {relative path: sha256 digest} table written
        # at freeze time; it is the definition of "untouched"
        with open(self.manifest_file) as f:
            expected = json.load(f)["files"]
        # actual = the same table computed from the files on disk now
        actual = self.compute_manifest()
        # collect every discrepancy before raising: a report naming
        # all problem files at once beats failing on the first one
        problems = []
        for rel, digest in expected.items():
            if rel not in actual:
                # the manifest lists it but the file is gone from disk
                problems.append(f"MISSING  {rel}")
            elif actual[rel] != digest:
                # the file exists but at least one byte differs
                problems.append(f"CHANGED  {rel}")
        # both directions matter: a file ADDED to frozen/ is as
        # suspicious as an edited one, so the reverse scan runs too
        for rel in actual:
            if rel not in expected:
                problems.append(f"EXTRA    {rel}")
        if problems:
            raise AssertionError(
                "Frozen test data does not match tests/manifest_sha256.json "
                "(someone edited the frozen copies; the tests refuse to "
                "run):\n  "
                + "\n  ".join(problems)
                + "\nIf the change is deliberate, regenerate with "
                "generate_frozen_reference.py --overwrite."
            )

    def load_reference(self):
        """Read the frozen reference chi2 values.

        Arguments:
          none.

        Returns:
          the dictionary stored in frozen/reference_chi2.json: one
          entry per configuration plus a "_meta" entry recording when
          and how the references were generated. The file sits inside
          frozen/, so verify_frozen() also protects it from editing.

        Raises:
          OSError when the reference file is absent (the frozen state
          was never generated).
        """
        with open(self.reference_file) as f:
            return json.load(f)

    # ---- the chi2 pipeline --------------------------------------------------
    def has_tatt(self, example):
        """Tell whether one example carries a TATT variant.

        The harness reads the TATT data vector of a configuration
        from its "tatt_dataset" key, so that key IS the statement
        "this example runs TATT": an example without it never gets
        the variant (the cluster combinations, whose lensing code has
        no TATT, are the standing case). The freeze generator asks
        before it evaluates a TATT reference; test modules of such
        examples simply never request the variant.

        Arguments:
          example = a key of the project's EXAMPLES table.

        Returns:
          True when the example has a TATT variant.

        Raises:
          KeyError when example names no EXAMPLES entry.
        """
        return "tatt_dataset" in self.examples[example]

    # cobaya and numpy are imported inside the functions, not at the
    # top of this module. The reason is OpenMP: OMP_NUM_THREADS must
    # be in the environment before the compiled libraries load, and
    # it is the TEST modules that set it, on their first line, before
    # importing this module's callers.
    def _frozen_module(self, example):
        """Load one frozen configuration module from its file path.

        importlib is used instead of a plain import statement because
        the frozen modules live inside frozen/, which is data, not a
        package: it has no __init__.py and is never on sys.path.
        Loading by path also guarantees the file that verify_frozen()
        hashed is exactly the file being executed.

        Arguments:
          example = a key of the project's EXAMPLES table.

        Returns:
          the loaded module, carrying the attributes `yaml_string`
          (the complete configuration) and `point` (the frozen
          evaluation point).

        Raises:
          KeyError when example names no EXAMPLES entry; OSError
          when the frozen module file is absent.
        """
        import importlib.util

        path = os.path.join(self.frozen_dir,
                            self.examples[example]["frozen_module"])
        # the importlib three-step: describe the file (spec), create
        # an empty module object from the description, then run the
        # file's code inside that object to fill in its attributes
        spec = importlib.util.spec_from_file_location(
            f"frozen_{example}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def load_frozen_info(self, example, tatt, high_accuracy=False,
                         overrides=None, baryon=None, fastpt=False,
                         fastpt_extra_args=None):
        """Build the cobaya input dictionary for one frozen configuration.

        Starts from the frozen module's yaml string and applies the
        run-time adjustments the tests need: the likelihood `path` is
        pointed at the absolute location of frozen/data, IA_model
        selects the intrinsic-alignment model, the data_file switches
        to the generated TATT or synthetic NLA vector when the
        project carries one, and cobaya's log level is raised so the
        component-loading chatter does not bury the test reports.

        An EXAMPLES entry carrying the "emulator" flag gets two more
        adjustments, both serving the accuracy measurement of
        test_emul2.py: its data_file becomes the non-emulated
        counterpart's synthetic vector (so |emulator - non-emulated|
        is the disagreement at the same data and nothing else), and
        the counterpart's ggl_exclude is copied in, because cosmolike
        only accepts a mask whose length equals the data-vector
        layout it computes from ggl_exclude.

        Arguments:
          example = a key of the project's EXAMPLES table.
          tatt    = True selects the TATT IA model, False keeps NLA.
          high_accuracy = True applies the project's high-accuracy
                    likelihood settings and
                    HIGH_ACCURACY_CAMB_EXTRA_ARGS on top of the
                    frozen configuration (accuracy advisory checks
                    only; not available for emulator configurations,
                    which have no camb block).
          overrides = None, or a pair (likelihood overrides, camb
                    extra_args overrides) applied on top of the
                    frozen configuration; _single_model_chi2_impl
                    builds this pair from one ACCURACY_KNOBS entry.
          baryon  = None, or a BARYON_METHODS label: the bfmt theory
                    block is added with that method selected, the
                    likelihood's external_baryon_suppression switch
                    is turned on, and the method's parameter point
                    enters the params block as fixed values.
          fastpt  = True computes the TATT terms with the python
                    FAST-PT package (IA_code: 1) instead of the C
                    cfastpt, adding the fastpt theory block the
                    computation requires. IA_code only matters under
                    TATT, so fastpt=True is combined with tatt=True.
          fastpt_extra_args = the fastpt block's settings table; None
                    selects FASTPT_FROZEN_SETTINGS, the configuration
                    the frozen FAST-PT references were computed with.

        Returns:
          the input dictionary ready for cobaya's get_model.

        Raises:
          ValueError when tatt or high_accuracy is requested for an
          emulator entry (the emulators were trained for the shipped
          IA settings and carry no camb accuracy knobs to push), or
          when a TATT_POINT name is absent from the frozen params.
        """
        from cobaya.yaml import yaml_load

        cfg = self.examples[example]
        # cfg.get("emulator") is None (falsy) for the non-emulated
        # entries, which do not carry the flag at all
        if cfg.get("emulator") and (tatt or high_accuracy):
            raise ValueError(
                f"{example}: the emulator configurations run only at "
                "their shipped IA settings (no TATT variant) and carry "
                "no camb accuracy knobs (no high_accuracy variant)")
        # _frozen_module loads frozen/<frozen_module>.py by path and
        # hands back its yaml_string attribute: the complete
        # configuration with every option and parameter written out
        # at freeze time
        info = yaml_load(self._frozen_module(example).yaml_string)
        # the tests drive the model directly, so a sampler or output
        # block left in the info would only confuse cobaya; pop's
        # second argument makes the removal a no-op instead of an
        # error when the key is absent
        info.pop("sampler", None)
        info.pop("output", None)
        # log level WARNING (30): component-loading chatter would
        # bury the test reports
        info["debug"] = 30
        info["timing"] = False
        likelihood_block = info["likelihood"][cfg["likelihood"]]
        # the frozen string stores the ROOTDIR-relative data path;
        # the absolute path is independent of the working directory
        likelihood_block["path"] = os.path.join(self.frozen_dir, "data")
        # tests must not write files as a side effect: an example can
        # ship print_datavector: True aimed at chains/, which does
        # not exist in a fresh clone (the vector generators re-enable
        # printing deliberately, into their own directories)
        likelihood_block["print_datavector"] = False
        # intrinsic-alignment model selection: 0 = NLA, 1 = TATT (the
        # `1 if tatt else 0` form yields 1 when tatt is True, else 0)
        likelihood_block["IA_model"] = 1 if tatt else 0
        if tatt:
            # TATT evaluates against its own generated data vector so
            # the chi2 sits at a minimum (see the TATT_GENERATORS
            # comment in the project file)
            likelihood_block["data_file"] = cfg["tatt_dataset"]
        else:
            # a project with synthetic NLA vectors does the same for
            # NLA: the shipped data_file is real data (or an
            # off-minimum modelvector) and the fiducial point sits
            # away from its minimum. Per-example "nla_dataset" keys
            # win over the project-wide constant; a project with
            # neither keeps the shipped data_file.
            nla = cfg.get("nla_dataset") or self.nla_dataset
            if nla is not None:
                likelihood_block["data_file"] = nla
        if cfg.get("emulator"):
            # the dataset just selected belongs to the exact
            # counterpart, and cosmolike only accepts a mask whose
            # length equals the data-vector layout it computes from
            # ggl_exclude (see the docstring); copy the counterpart's
            # frozen layout so the two configurations describe the
            # same data vector
            exact_cfg = self.examples[cfg["exact_example"]]
            exact_module = self._frozen_module(cfg["exact_example"])
            exact_info = yaml_load(exact_module.yaml_string)
            exact_block = exact_info["likelihood"][exact_cfg["likelihood"]]
            # a cosmic-shear counterpart has no galaxy-galaxy lensing
            # part and therefore no ggl_exclude key; without the key
            # both configurations already use the default layout, so
            # there is nothing to copy
            if "ggl_exclude" in exact_block:
                likelihood_block["ggl_exclude"] = (
                    exact_block["ggl_exclude"])
        if high_accuracy:
            # dict.update merges the pushed settings into the block
            # in place, overwriting any key both sides carry
            likelihood_block.update(self.high_accuracy_likelihood)
            info["theory"]["camb"]["extra_args"].update(
                HIGH_ACCURACY_CAMB_EXTRA_ARGS)
        if overrides is not None:
            # one knob at a time (an ACCURACY_KNOBS entry): the same
            # mechanism as high_accuracy, restricted to a single
            # setting; the overrides pair unpacks into its two
            # dictionaries
            like_over, camb_over = overrides
            likelihood_block.update(like_over)
            info["theory"]["camb"]["extra_args"].update(camb_over)
        if tatt:
            # a TATT parameter can be SAMPLED in the frozen
            # configuration (it has a prior; build_point then sets
            # its value in the evaluation point) or FIXED (a block
            # with a value and no prior; it must be replaced here,
            # before the model is built, because a fixed parameter
            # cannot change per evaluation)
            for name, value in self.tatt_point.items():
                # .get returns None instead of raising when the name
                # is missing, so the error below can name the
                # parameter
                block = info["params"].get(name)
                if block is None:
                    raise ValueError(
                        f"TATT parameter {name} is not in the frozen "
                        "configuration")
                if isinstance(block, dict) and "prior" not in block:
                    block["value"] = value
        if fastpt:
            likelihood_block["IA_code"] = 1
            # the frozen configurations carry no fastpt block (the
            # live examples ship it commented out), so it is added
            # here with the same path the examples use. setdefault
            # hands back the existing "theory" dictionary, or first
            # inserts the empty {} and hands that back, so the
            # assignment lands inside info either way
            info.setdefault("theory", {})["fastpt"] = {
                "path": "./external_modules/code/FAST-PT",
            }
            if fastpt_extra_args is None:
                # the frozen references were computed on one shared
                # grid; FASTPT_FROZEN_SETTINGS reproduces it (see its
                # comment)
                fastpt_extra_args = FASTPT_FROZEN_SETTINGS
            # dict(...) copies, so a caller's settings table is never
            # shared with (or mutated through) the built model
            info["theory"]["fastpt"]["extra_args"] = dict(fastpt_extra_args)
        if baryon is not None:
            _, theory_options, baryon_point = _baryon_method(baryon)
            # the likelihood requests the suppression product only
            # when this switch is on (see external_baryon_suppression
            # in likelihood/_cosmolike_prototype_base.py)
            likelihood_block["external_baryon_suppression"] = True
            # dict(a, **b) builds a new dictionary with a's entries
            # plus b's: python_path tells cobaya where the bfmt class
            # lives
            info["theory"]["bfmt"] = dict(
                {"python_path": os.path.join(
                    os.environ["ROOTDIR"], "external_modules", "code",
                    "baryon_suppression")},
                **theory_options)
            for name, value in baryon_point.items():
                info["params"][name] = value
        return info

    def load_frozen_point(self, example):
        """Read the frozen evaluation point of one example.

        Arguments:
          example = a key of the project's EXAMPLES table.

        Returns:
          a fresh {parameter name: value} dictionary (copied, so a
          caller may modify it without affecting later calls).

        Raises:
          KeyError when example names no EXAMPLES entry (through
          _frozen_module).
        """
        # dict(...) builds a COPY of the module's point table: the
        # edits a caller makes stay in its copy, never in the module
        return dict(self._frozen_module(example).point)

    def build_point(self, model, example, tatt):
        """Assemble the exact point a test evaluates, with a safety check.

        The frozen point must cover the model's sampled parameters
        one to one. When likelihood or theory code changes its
        parameter set (a new nuisance parameter appears, or one is
        removed), evaluating would either fail cryptically or
        silently pick up a new default, so the mismatch is reported
        here by name instead.

        Arguments:
          model   = the cobaya Model the point will be evaluated on.
          example = a key of the project's EXAMPLES table.
          tatt    = True replaces the TATT_POINT values (nonzero
                    A2/BTA) in the frozen point; False evaluates it
                    unchanged.

        Returns:
          a {parameter name: value} dictionary covering every sampled
          parameter of the model.

        Raises:
          AssertionError listing the parameters that appeared or
          vanished when the model's sampled set differs from the
          frozen point.
        """
        # load_frozen_point returns a copy of the frozen module's
        # point: the exact {parameter: value} table the references
        # were computed at
        point = self.load_frozen_point(example)
        # sampled = the parameters THIS model, built from today's
        # code, expects to receive; the frozen point must cover them
        # exactly. set() turns the name list into a set so the two
        # sides can be compared and subtracted; set(point) takes the
        # dictionary's KEYS
        sampled = set(model.parameterization.sampled_params())
        if sampled != set(point):
            # sampled - set(point) = names only the model has; the
            # reversed difference = names only the point has;
            # sorted() fixes the order so the message is reproducible
            raise AssertionError(
                "sampled-parameter set differs from the frozen point (the "
                "likelihood/theory code changed its parameters):\n"
                f"  new since freeze: {sorted(sampled - set(point))}\n"
                f"  gone since freeze: {sorted(set(point) - sampled)}"
            )
        if tatt:
            # only SAMPLED TATT parameters appear in the point; the
            # fixed ones were already replaced inside the
            # configuration by load_frozen_info (which also catches
            # unknown names)
            for name, value in self.tatt_point.items():
                if name in point:
                    point[name] = value
        return point

    def _single_model_chi2_impl(self, example, tatt,
                                high_accuracy=False, knob=None,
                                baryon=None, fastpt=False):
        """Evaluate single_model_chi2's quantity in-process (worker side).

        Runs inside the worker subprocess only: building a model
        here, next to a model of different dimensions, would abort
        the process (see _run_isolated). It chains the pipeline:
        load_frozen_info, make_model, build_point, evaluate_chi2.

        Arguments:
          example = a key of the project's EXAMPLES table.
          tatt    = True evaluates the TATT variant, False the NLA
                    one.
          high_accuracy = True evaluates with the pushed numerical
                    settings (see load_frozen_info).
          knob    = None, or the label of one ACCURACY_KNOBS entry;
                    that knob's overrides are applied alone (the
                    one-at-a-time scan of test_accuracy.py).
          baryon  = None, or a BARYON_METHODS label: the bfmt theory
                    block computes the baryonic suppression of the
                    nonlinear power spectrum
                    (test_accuracy_baryons.py).

        Returns:
          the chi2 as a float, or None when, with a baryon method
          enabled, the evaluation returned a non-finite chi2. The
          usual cause is the method rejecting the frozen fiducial (a
          training-range violation, identified by the bfmt block's
          own log line); the None alone does not name the cause.

        Raises:
          ValueError when knob names no ACCURACY_KNOBS entry;
          AssertionError when the sampled-parameter set differs from
          the frozen point (build_point) or, without baryon, when the
          chi2 is not finite (evaluate_chi2).
        """
        # `"TATT" if tatt else "NLA"` picks the first name when tatt
        # is True, the second otherwise
        ia_label = "TATT" if tatt else "NLA"
        if fastpt:
            ia_label += "+FASTPT"
        if high_accuracy:
            ia_label += ", high accuracy"
        overrides = None
        if knob is not None:
            # knob = a label from ACCURACY_KNOBS; the comprehension
            # keeps only the entries whose first field k[0] equals
            # it, so matches is a list with one member (or none:
            # unknown label)
            matches = [k for k in self.accuracy_knobs if k[0] == knob]
            if len(matches) != 1:
                raise ValueError(f"unknown accuracy knob {knob!r}")
            overrides = (matches[0][1], matches[0][2])
            ia_label += f", knob: {knob}"
        if baryon is not None:
            ia_label += f", baryons: {baryon}"
        print(f"  building model ({example}, {ia_label}) ...", flush=True)
        # load_frozen_info returns the frozen configuration
        # dictionary with the run-time adjustments applied;
        # make_model turns it into an evaluable cobaya Model (loads
        # CAMB and the cosmolike interface)
        info = self.load_frozen_info(example, tatt,
                                     high_accuracy=high_accuracy,
                                     overrides=overrides, baryon=baryon,
                                     fastpt=fastpt)
        model = make_model(info)
        # build_point returns the frozen evaluation point after
        # checking that the point and the model name the same
        # sampled parameters: if the likelihood or theory code gained
        # or lost a sampled parameter since the freeze, the mismatch
        # is reported by name instead of failing deep inside cobaya
        point = self.build_point(model, example, tatt)
        if baryon is not None:
            # dict(point) copies before the in-place update below
            point = dict(point)
            point.update(BARYON_POINT_OVERRIDES.get(baryon, {}))
        print("  evaluating the fiducial point ...", flush=True)
        if baryon is not None:
            # With the feedback on, a non-finite chi2 is expected
            # when the method rejects the frozen fiducial (a
            # training-range violation; the bfmt block logs a warning
            # naming the offending parameter). The non-finite value
            # alone does not identify the cause - that log line does.
            # The docstring makes this case a None return the caller
            # can report, so hand back None instead of letting the
            # assertion kill the worker.
            try:
                return evaluate_chi2(model, point)
            except AssertionError:
                return None
        return evaluate_chi2(model, point)

    def _ten_in_a_row_impl(self, example, tatt, ee2=False):
        """Run ten_in_a_row_chi2's race check in-process (worker side).

        Race check: the fiducial evaluated fresh and as 10th of a
        row. On ONE model instance, in order: the fiducial point (the
        fresh value), then the nine RACE_PERTURBATIONS cosmologies,
        then the fiducial again as the 10th point of the row. State
        leaked between evaluations, or an OpenMP race under
        REQUIRED_OMP_THREADS threads, shifts the second fiducial
        value away from the first; correct code reproduces it to
        float noise. Agreement is evidence for this sequence,
        observable and thread count - it does not prove the absence
        of every possible race or state leak. Each evaluation prints
        its chi2, so a stuck or slow run is visible line by line.

        Arguments:
          example = a key of the project's EXAMPLES table.
          tatt    = True runs the TATT variant, False the NLA one.
          ee2     = True sources the nonlinear P(k) from
                    EuclidEmulator2 (non_linear_emul: 1) instead of
                    the frozen setting; EE2's compute is
                    OpenMP-threaded, so this variant exercises its
                    threading inside the race sequence.

        Returns:
          (fresh, tenth): chi2 of the first fiducial evaluation and
          chi2 of the fiducial as the 10th point of the row, both
          floats.

        Raises:
          AssertionError when the sampled-parameter set differs from
          the frozen point (build_point) or a chi2 is not finite
          (evaluate_chi2), or when the restored full vector differs.
          Equal chi-squared values alone do not establish vector equality.
        """
        import numpy as np

        # `"TATT" if tatt else "NLA"` picks the first name when tatt
        # is True, the second otherwise
        ia_label = "TATT" if tatt else "NLA"
        if ee2:
            ia_label += "+EE2"
        print(f"  building model ({example}, {ia_label}) ...", flush=True)
        # one model instance for the whole sequence: sharing the
        # instance is the point, since leaked state lives inside it
        info = self.load_frozen_info(example, tatt)
        if ee2:
            info["likelihood"][self.examples[example]["likelihood"]][
                "non_linear_emul"] = 1
        model = make_model(info)
        likelihood = model.likelihood[self.examples[example]["likelihood"]]
        original_score = likelihood.compute_logp
        score_was_local = "compute_logp" in vars(likelihood)
        vectors = []

        def record_score(datavector, *args, **kwargs):
            # Copy the exact argument being scored, including Python-side
            # contributions such as baryon PCs. A second native getter could
            # omit those contributions or rebuild the state under test.
            # This hook belongs only to this isolated test worker; it always
            # delegates to the real likelihood and is restored below.
            vectors.append(np.array(datavector, dtype=float, copy=True))
            return original_score(datavector, *args, **kwargs)

        likelihood.compute_logp = record_score
        try:
            point = self.build_point(model, example, tatt)
            # the fresh value: the fiducial evaluated before anything
            # else touched this model instance
            fresh = evaluate_chi2(model, point)
            # :.8f prints the value with a fixed eight decimals (:.4f
            # below: four); :2d pads the row counter to a width of two
            print(f"  fresh model, fiducial point:  chi2 = {fresh:.8f}",
                  flush=True)
            # enumerate yields (counter, entry) pairs; start=1 makes the
            # printed row numbers begin at 1 instead of 0
            for i, perturbation in enumerate(RACE_PERTURBATIONS, start=1):
                # an emulator configuration can sample fewer parameters
                # than the exact ones (mnu is fixed inside the emulator
                # training), so a perturbation key the model does not
                # sample is dropped rather than kept in a separate
                # perturbation table per configuration. The comprehension
                # builds a new dictionary from the (name, value) pairs
                # whose name the point carries.
                applied = {k: v for k, v in perturbation.items()
                           if k in point}
                # {**point, **applied} is a NEW dictionary: point's
                # entries with the applied ones written over them; point
                # itself stays untouched for the final fiducial
                # evaluation
                chi2 = evaluate_chi2(model, {**point, **applied})
                # one "name=value" text per changed parameter, glued with
                # ", " into the progress line
                changed = ", ".join(f"{k}={v}" for k, v in applied.items())
                print(f"  row {i:2d}/10 ({changed}):  chi2 = {chi2:.4f}",
                      flush=True)
            tenth = evaluate_chi2(model, point)
            print(f"  row 10/10 (fiducial again):  chi2 = {tenth:.8f}",
                  flush=True)
            if len(vectors) != len(RACE_PERTURBATIONS) + 2:
                raise AssertionError(
                    "The race sequence did not capture every scored vector.")
            # The score is a scalar: different residual vectors can lie on
            # the same chi-squared surface. Check every restored entry too.
            np.testing.assert_allclose(
                vectors[-1], vectors[0], rtol=1.e-11, atol=1.e-30,
                equal_nan=False,
                err_msg="Returning to the fiducial changed the scored data vector.",
            )
            print("  first/final scored vectors agree (rtol=1e-11)", flush=True)
            return fresh, tenth
        finally:
            if score_was_local:
                likelihood.compute_logp = original_score
            else:
                del likelihood.compute_logp
            model.close()

    # ---- baryonic feedback --------------------------------------------------
    def _baryon_accuracy_delta_impl(self, baryon, knob=None):
        """Compute one feedback method's delta chi2 against its own vector.

        The zero-based mechanism at the frozen fiducial: a
        DEFAULT-settings model with this method's feedback on writes
        its theory vector during evaluation (print_datavector); that
        vector becomes the data of a temporary dataset descriptor, so
        the default chi2 against it is zero by construction; a second
        model (high accuracy, or one accuracy knob alone) evaluates
        at the SAME point against that descriptor, and its chi2 IS

            delta chi2 = chi2(pushed settings) - chi2(default)

        a pure numerics response at the minimum. Nothing is written
        into frozen/ (the manifest pins every byte there); the
        vector, the descriptor, and the symlinked data folder live
        and die inside a temporary directory. The evaluation point is
        the frozen fiducial plus the method's cosmology override
        (BARYON_POINT_OVERRIDES), applied to BOTH evaluations. Both
        models share example1's data-vector dimensions, so building
        them one after another inside one worker process is safe.

        Arguments:
          baryon = a BARYON_METHODS label.
          knob   = None for the all-knobs high-accuracy comparison,
                   or an ACCURACY_KNOBS label evaluated alone.

        Returns:
          the delta chi2 as a float.

        Raises:
          ValueError when knob names no ACCURACY_KNOBS entry;
          RuntimeError when the evaluation printed no vector, the
          vector's length disagrees with the original data, or the
          descriptor's data_file line is not unique.
        """
        cfg = self.examples["example1"]
        frozen_data_dir = os.path.join(self.frozen_dir, "data")
        info = self.load_frozen_info("example1", tatt=False,
                                     baryon=baryon)
        likelihood_block = info["likelihood"][cfg["likelihood"]]
        workdir = tempfile.mkdtemp(prefix="cocoa_baryon_model_")
        try:
            slug = baryon.replace(" ", "_")
            vector_name = f"baryon_{slug}.modelvector"
            descriptor_name = f"baryon_{slug}.dataset"
            # the likelihood joins path + filename for EVERY file a
            # descriptor names, so the temporary directory must look
            # like a complete data folder: symlink each frozen data
            # file in, except the two files written below. frozen/
            # holds a vector and a descriptor of these names (the
            # drift tests' own), and a write to a symlink goes through
            # to its target: it would overwrite the frozen copies
            for name in sorted(os.listdir(frozen_data_dir)):
                if name in (vector_name, descriptor_name):
                    continue
                os.symlink(os.path.join(frozen_data_dir, name),
                           os.path.join(workdir, name))
            vector_path = os.path.join(workdir, vector_name)
            # the default model's evaluation writes the theory
            # vector; its chi2 (against the frozen no-feedback data)
            # plays no role
            likelihood_block["print_datavector"] = True
            likelihood_block["print_datavector_file"] = vector_path
            print(f"  building the default model ({baryon}) ...",
                  flush=True)
            model = make_model(info)
            point = dict(self.build_point(model, "example1",
                                          tatt=False))
            point.update(BARYON_POINT_OVERRIDES.get(baryon, {}))
            print("  evaluating (writes the synthetic vector) ...",
                  flush=True)
            evaluate_chi2(model, point)
            if not os.path.isfile(vector_path):
                raise RuntimeError(
                    f"print_datavector wrote no file at {vector_path}")
            # full-length check: the covariance and the masks select
            # entries by position, so a short vector would misalign
            # them. sum(1 for _ in f) counts lines: the generator
            # yields 1 per line of the file and sum adds them up
            with open(vector_path) as f:
                generated_lines = sum(1 for _ in f)
            with open(os.path.join(frozen_data_dir,
                                   likelihood_block["data_file"])) as f:
                descriptor = f.read()
            original_vector = None
            for line in descriptor.splitlines():
                if line.strip().startswith("data_file"):
                    # split("=", 1) cuts at the first "=" only; [1] is
                    # the text after it, the vector's file name
                    original_vector = line.split("=", 1)[1].strip()
            with open(os.path.join(frozen_data_dir,
                                   original_vector)) as f:
                original_lines = sum(1 for _ in f)
            if generated_lines != original_lines:
                raise RuntimeError(
                    f"generated vector has {generated_lines} lines; "
                    f"the original {original_vector} has "
                    f"{original_lines}")
            # the temporary descriptor: the frozen text with only the
            # data_file line renamed, so the same covariance, n(z),
            # and masks are read but the synthetic vector is the data
            replaced = 0
            out_lines = []
            for line in descriptor.splitlines(keepends=True):
                if line.strip().startswith("data_file"):
                    out_lines.append(f"data_file = {vector_name}\n")
                    replaced += 1
                else:
                    out_lines.append(line)
            if replaced != 1:
                raise RuntimeError(
                    "expected exactly one data_file line, found "
                    f"{replaced}")
            with open(os.path.join(workdir, descriptor_name), "w") as f:
                f.write("".join(out_lines))
            # the pushed-settings model, at the same point, against
            # the synthetic vector: its chi2 is the delta by
            # construction
            overrides = None
            high_accuracy = knob is None
            if knob is not None:
                matches = [k for k in self.accuracy_knobs
                           if k[0] == knob]
                if len(matches) != 1:
                    raise ValueError(f"unknown accuracy knob {knob!r}")
                overrides = (matches[0][1], matches[0][2])
            info_high = self.load_frozen_info(
                "example1", tatt=False, baryon=baryon,
                high_accuracy=high_accuracy, overrides=overrides)
            block_high = info_high["likelihood"][cfg["likelihood"]]
            block_high["path"] = workdir
            block_high["data_file"] = descriptor_name
            label = knob if knob is not None else "high accuracy"
            print(f"  building the pushed model ({label}) ...",
                  flush=True)
            model_high = make_model(info_high)
            print("  evaluating against the synthetic vector ...",
                  flush=True)
            return float(evaluate_chi2(model_high, point))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _baryon_drift_chi2_impl(self, baryon):
        """Compute one feedback method's drift chi2 against its FROZEN vector.

        The frozen vector was written at freeze time by
        generate_frozen_reference.py --baryons: the default-settings
        theory prediction with this method's feedback on, at the
        frozen fiducial plus the method's cosmology override. At
        freeze time the chi2 against it was zero by construction, so
        any chi2 above the tolerance today means cosmolike or the
        bfmt theory block changed its prediction since the freeze:
        the same pinning idea as the reference tests, applied to the
        feedback pipeline.

        Arguments:
          baryon = a BARYON_METHODS label.

        Returns:
          the chi2 as a float.

        Raises:
          AssertionError when the sampled-parameter set differs from
          the frozen point (build_point) or the chi2 is not finite
          (evaluate_chi2).
        """
        cfg = self.examples["example1"]
        info = self.load_frozen_info("example1", tatt=False,
                                     baryon=baryon)
        likelihood_block = info["likelihood"][cfg["likelihood"]]
        # the method's own frozen dataset: same covariance, n(z), and
        # masks, but the freeze-time feedback prediction as the data
        likelihood_block["data_file"] = _baryon_dataset(baryon)
        print(f"  building model ({baryon}, frozen vector) ...",
              flush=True)
        model = make_model(info)
        point = dict(self.build_point(model, "example1", tatt=False))
        point.update(BARYON_POINT_OVERRIDES.get(baryon, {}))
        print("  evaluating the fiducial point ...", flush=True)
        return float(evaluate_chi2(model, point))

    # ---- worker-subprocess isolation ----------------------------------------
    # ---- the N-random-models check -------------------------------------------
    def draw_uniform_point(self, info, example, rng, box_overrides=None):
        """Draw one random sampled-parameter point across the priors.

        Every freely varied parameter with a uniform box prior (a
        `prior` block with `min` and `max`) is drawn uniformly across
        its box, which is a draw from the prior. Two kinds of sampled
        parameters are not drawn and keep their frozen point value
        instead: those whose prior is a Gaussian rather than a box
        (`dist: norm`, no min/max; typically the photo-z shifts and
        the shear calibrations), and any parameter without a prior
        block at all. Keeping their frozen values leaves the draw well
        inside those Gaussian priors while the box-prior parameters
        explore the full volume.

        A box override narrows the box of one parameter for the draw
        (a project whose examples tightened a prior after the freeze
        passes the new box instead of refreezing). The draw still
        consumes one random number per parameter in the same order, so
        an override changes only that parameter's values: every other
        parameter of a seeded point is the same with and without it.

        Arguments:
          info    = the cobaya input dictionary of the frozen
                    configuration (a load_frozen_info result); its
                    params block holds every parameter's prior.
          example = a key of the EXAMPLES table; names the frozen
                    point that fills the parameters no box prior
                    covers.
          rng     = a numpy random Generator; the caller seeds it,
                    which is what makes the draw reproducible.
          box_overrides = None, or {parameter name: (min, max)}: the
                    box to draw that parameter from instead of its
                    frozen one (nmodels_points validates it).

        Returns:
          {parameter name: value} with the same names as the frozen
          point, every box-prior parameter replaced by a uniform draw.
        """
        # load_frozen_point returns a copy of the frozen evaluation
        # point; its keys are exactly the sampled parameters
        point = self.load_frozen_point(example)
        drawn = {}
        # sorted() fixes the order the rng is consumed in: dict order
        # would also be stable here (the generator writes the point
        # sorted), but reproducibility must not hang on that detail
        for name in sorted(point):
            spec = info["params"][name]
            prior = spec.get("prior")
            # `"min" in prior` asks whether the dictionary carries
            # that key; a prior dict holding min and max is a uniform
            # box
            if isinstance(prior, dict) and "min" in prior and "max" in prior:
                # a uniform box prior: the draw is a prior sample
                low = prior["min"]
                high = prior["max"]
                if box_overrides is not None and name in box_overrides:
                    low, high = box_overrides[name]
                drawn[name] = float(rng.uniform(low, high))
            else:
                # Gaussian prior or no prior: keep the frozen value
                drawn[name] = point[name]
        return drawn

    def nmodels_points(self, example, box_overrides=None, hold=None,
                       amplitude_cut=None, n_models=NMODELS_COUNT,
                       seed=RANDOM_MODEL_SEED):
        """Return the fixed points of the N-random-models check.

        The points are a designed stress sample, not a draw from the
        whole joint prior. Model m draws every box-prior parameter
        uniformly with numpy.random.default_rng(seed + m)
        (draw_uniform_point) while the Gaussian-prior parameters keep
        their frozen values; box_overrides can narrow a box,
        amplitude_cut can reject and redraw, and with a hold list
        every odd model resets the listed parameters to the frozen
        point, so half of the models measure what the other
        parameters alone do to the accuracy. Statistics over the
        points (the maximum, the fraction above a limit) describe
        these evaluated points, not probabilities under the
        scientific prior.

        The points never change: the project's stored high-accuracy
        vectors (frozen/nmodels_<example>.npz) were evaluated at exactly
        these points, and the check redraws them and compares, so a
        changed prior or draw setting is caught instead of comparing
        vectors of different points.

        Arguments:
          example  = a key of the project's EXAMPLES table.
          box_overrides = None, or {parameter name: (min, max)}: a
                     narrower box for a box-prior parameter (see
                     draw_uniform_point); each must lie inside the
                     frozen box.
          hold     = None, or a tuple of parameter names (an entry
                     ending in "_" is a name prefix; held_parameter)
                     that every odd model (m = 1, 3, ...) resets to the
                     frozen point after the draw.
          amplitude_cut = None, or the largest clustering-amplitude
                     ratio (largest_amplitude_ratio) a draw may have: a
                     draw above it is replaced by the next draw of the
                     same generator, so the points stay a uniform,
                     reproducible sample of the box inside the cut. It
                     must exceed 1, the ratio of the fiducial.
          n_models = the number of points (NMODELS_COUNT).
          seed     = the seed base (RANDOM_MODEL_SEED).

        Returns:
          (points, held): the list of n_models point dictionaries
          {sampled parameter name: value}, and the list of n_models
          booleans, True for a model that holds the hold list.

        Raises:
          ValueError when a box override, a hold entry or the
          amplitude cut does not fit the frozen configuration;
          RuntimeError when no draw falls inside the amplitude cut in
          NMODELS_MAX_DRAWS draws.
        """
        import numpy as np

        info = self.load_frozen_info(example, tatt=False)
        frozen_point = self.load_frozen_point(example)
        if box_overrides is not None:
            for name in box_overrides:
                low, high = box_overrides[name]
                spec = info["params"].get(name)
                prior = None
                if isinstance(spec, dict):
                    prior = spec.get("prior")
                if not (isinstance(prior, dict) and "min" in prior
                        and "max" in prior):
                    raise ValueError(
                        f"box override for {name!r}: {example} has no "
                        "box prior for it in its frozen configuration; "
                        "override only parameters with a min/max prior")
                if not (prior["min"] <= low < high <= prior["max"]):
                    raise ValueError(
                        f"box override for {name!r} is ({low}, {high}); "
                        "it must satisfy min < max inside the frozen box "
                        f"({prior['min']}, {prior['max']})")
        if hold is not None:
            for entry in hold:
                covered = False
                for name in frozen_point:
                    if held_parameter(name, (entry,)):
                        covered = True
                if not covered:
                    raise ValueError(
                        f"hold entry {entry!r} covers no sampled "
                        f"parameter of {example} "
                        f"({sorted(frozen_point)}): use an exact name, "
                        "or a prefix ending in '_'")
        if amplitude_cut is not None:
            if not amplitude_cut > 1.0:
                raise ValueError(
                    f"amplitude_cut = {amplitude_cut}: the cut must exceed "
                    "1, the amplitude ratio of the fiducial")
            if largest_amplitude_ratio(frozen_point, frozen_point) is None:
                raise ValueError(
                    f"amplitude_cut needs {AMPLITUDE_NAME} and linear "
                    f"biases (*{LINEAR_BIAS_TAG}*) among the sampled "
                    f"parameters of {example} ({sorted(frozen_point)})")
        points = []
        held = []
        for m in range(n_models):
            rng = np.random.default_rng(seed + m)
            point = self.draw_uniform_point(info, example, rng,
                                            box_overrides=box_overrides)
            if amplitude_cut is not None:
                # redraw from the same generator until every bin is
                # inside the cut: a uniform sample of the restricted box
                draws = 1
                while largest_amplitude_ratio(point, frozen_point) > amplitude_cut:
                    if draws >= NMODELS_MAX_DRAWS:
                        raise RuntimeError(
                            f"model {m}: no draw inside amplitude_cut = "
                            f"{amplitude_cut} in {draws} draws; the cut "
                            "excludes almost the whole box")
                    point = self.draw_uniform_point(
                        info, example, rng, box_overrides=box_overrides)
                    draws += 1
            is_held = hold is not None and m % 2 == 1
            if is_held:
                for name in point:
                    if held_parameter(name, hold):
                        point[name] = frozen_point[name]
            points.append(point)
            held.append(is_held)
        return points, held

    def nmodels_config_hash(self, example):
        """Return the fingerprint of the example's high-accuracy setup.

        The stored N-random-models vectors hold the theory at the
        pushed accuracy settings on top of the frozen configuration.
        The fingerprint is the SHA-256 of that configuration's
        resolved likelihood and theory dictionaries, with every
        "path" entry left out (they are absolute paths that differ
        between machines). A change of those recorded settings - the
        frozen configuration, the high-accuracy overrides, or the
        likelihood defaults the freeze expanded - changes the
        fingerprint, and the check then asks for the stored vectors
        to be regenerated.

        The fingerprint identifies only those recorded settings. It
        does not hash the params block, the source code, the
        compiled libraries, or the CONTENT of input files: a data
        file changed under the same filename leaves it unchanged.
        Matching it therefore does not prove that a retained vector
        was produced by the current calculation; regenerating after
        such a change is a separate maintainer decision (see the
        preservation note in cocoa_testing_freeze's freeze_main).

        Arguments:
          example = a key of the project's EXAMPLES table.

        Returns:
          the hexadecimal SHA-256 string.
        """
        import hashlib
        import json

        def without_paths(value):
            # a copy of nested dictionaries and lists with every "path"
            # key dropped; other values pass through unchanged
            if isinstance(value, dict):
                kept = {}
                for key in value:
                    if key != "path":
                        kept[key] = without_paths(value[key])
                return kept
            if isinstance(value, list):
                items = []
                for item in value:
                    items.append(without_paths(item))
                return items
            return value

        info = self.load_frozen_info(example, tatt=False, high_accuracy=True)
        blocks = {"likelihood": without_paths(info.get("likelihood")),
                  "theory": without_paths(info.get("theory"))}
        # sort_keys makes the text independent of dictionary order;
        # default=str prints any non-JSON value (a numpy number) the
        # same way every time
        text = json.dumps(blocks, sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()

    # ---- the EE2 modification test --------------------------------------------
    def _ee2_comparison_block(self, example, label, vectors_dir,
                              reference_label, original_site=None):
        """Run the ten-cosmology sweep on one EE2 build (the pin's worker).

        The same construction as the Halofit-vs-EE2 blocks: the frozen
        NLA configuration with non_linear_emul: 1, evaluated at every
        NONLINEAR_COMPARISON_POINTS entry with the printed data vector
        per cosmology; a block given a reference_label reports the
        per-cosmology delta^T C^-1 delta against the reference block's
        vectors (the reference is the installed, modified EE2, so its
        own chi2 against itself is zero by construction). The worker
        asserts which euclidemu2 binary it imported, so a path mistake
        cannot silently compare a build against itself. Unlike the
        other workers, the child program here is written out in full
        (it does not call a block function of the shim).

        Arguments:
          example = a key of the EXAMPLES table (the cosmic-shear NLA
                    entry).
          label   = the file-name tag of this block's printed vectors
                    ("cocoa", "original").
          vectors_dir = the directory the per-cosmology vectors are
                    printed into, shared by the two blocks.
          reference_label = None to only print vectors, or the label
                    of the block to measure against.
          original_site = None for the installed build, or the
                    site-packages path of the original build (the
                    worker then expects euclidemu2 to resolve there;
                    the caller prepends it to the child's PYTHONPATH
                    and the shim is applied).

        Returns:
          {"dchi2_vs_reference": per-cosmology list or None}.
        """
        import subprocess
        import sys

        workdir = tempfile.mkdtemp(prefix="cocoa_ee2_check_")
        out_path = os.path.join(workdir, "result.json")
        # expect = a path fragment the child's euclidemu2 file must
        # contain: the original build's site-packages, or .local
        # (where Cocoa installs the modified EE2); shim = the
        # EE2_ORIGINAL_SHIM text, pasted into the child program for
        # the original build only
        expect = original_site if original_site else ".local"
        shim = EE2_ORIGINAL_SHIM if original_site else ""
        child_code = (
            "import os\n"
            f"os.environ['OMP_NUM_THREADS'] = {REQUIRED_OMP_THREADS!r}\n"
            "import json\n"
            "import sys\n"
            f"sys.path.insert(0, {self.tests_dir!r})\n"
            "import euclidemu2\n"
            f"assert {expect!r} in euclidemu2.__file__, euclidemu2.__file__\n"
            "print('  euclidemu2 =', euclidemu2.__file__, flush=True)\n"
            f"{shim}\n"
            "import numpy as np\n"
            "import cocoa_test_utils as u\n"
            "u.require_cocoa_environment()\n"
            f"info = u.load_frozen_info({example!r}, tatt=False)\n"
            f"lb = info['likelihood'][u.EXAMPLES[{example!r}]['likelihood']]\n"
            "lb['non_linear_emul'] = 1\n"
            f"cur = os.path.join({vectors_dir!r}, {label!r}"
            " + '_cur.modelvector')\n"
            "lb['print_datavector'] = True\n"
            "lb['print_datavector_file'] = cur\n"
            "model = u.make_model(info)\n"
            f"base = u.build_point(model, {example!r}, tatt=False)\n"
            "for i, cosmo in enumerate(u.NONLINEAR_COMPARISON_POINTS, 1):\n"
            # _evaluate_cached: only the printed data vector matters
            # here, and the ten points share everything but the
            # cosmology, so cobaya may reuse unchanged components
            "    u._evaluate_cached(model, dict(base, **cosmo))\n"
            "    assert os.path.isfile(cur), 'no vector printed'\n"
            f"    os.replace(cur, os.path.join({vectors_dir!r},\n"
            f"        {label!r} + '_pt%02d.modelvector' % i))\n"
            f"    print('  ' + {label!r} + ' cosmology %2d/10' % i,\n"
            "          flush=True)\n"
            "result = {}\n"
            f"ref = {reference_label!r}\n"
            "if ref is not None:\n"
            f"    import {self.interface_module} as ci\n"
            "    icov = np.array(ci.get_inv_cov_masked())\n"
            "    dchi2s = []\n"
            "    for i in range(1, 11):\n"
            f"        own = u._load_datavector(os.path.join({vectors_dir!r},\n"
            f"            {label!r} + '_pt%02d.modelvector' % i))\n"
            f"        rv = u._load_datavector(os.path.join({vectors_dir!r},\n"
            "            ref + '_pt%02d.modelvector' % i))\n"
            "        assert own.shape == rv.shape == (icov.shape[0],)\n"
            "        d = own - rv\n"
            "        dchi2s.append(float(d @ icov @ d))\n"
            "    result['dchi2_vs_reference'] = dchi2s\n"
            f"with open({out_path!r}, 'w') as f:\n"
            "    json.dump(result, f)\n"
        )
        # the child's environment is a copy of this one; for the
        # original build its site-packages goes first on PYTHONPATH,
        # so the child's `import euclidemu2` finds the original before
        # the installed one
        env = dict(os.environ)
        if original_site:
            env["PYTHONPATH"] = (original_site + os.pathsep
                                 + env.get("PYTHONPATH", ""))
        try:
            subprocess.run([sys.executable, "-c", child_code],
                           check=True, env=env)
            with open(out_path) as f:
                return json.load(f)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def ee2_original_vs_cocoa_dchi2s(self, example):
        """Return the per-cosmology differences of the two EE2 builds.

        Builds the pre-modification EE2 into a temporary prefix
        (build_original_ee2), then runs the two blocks: the installed,
        modified EE2 prints the reference vectors, and the original
        (through the compatibility shim) is measured against them at
        the ten shared cosmologies. Everything temporary dies with
        this call; the installed environment is never touched
        (--ignore-installed protects .local).

        Arguments:
          example = a key of the EXAMPLES table (the cosmic-shear NLA
                    entry).

        Returns:
          the per-cosmology delta^T C^-1 delta list, index-aligned
          with NONLINEAR_COMPARISON_POINTS.

        Raises:
          RuntimeError from build_original_ee2 when the original
          commit is unavailable locally (the test turns this into a
          skip).
        """
        prefix_dir = tempfile.mkdtemp(prefix="cocoa_ee2_original_")
        vectors_dir = tempfile.mkdtemp(prefix="cocoa_ee2_vectors_")
        try:
            print("  building the pre-modification EE2 "
                  f"({EE2_ORIGINAL_COMMIT[:7]}) ...", flush=True)
            original_site = build_original_ee2(prefix_dir)
            self._ee2_comparison_block(example, "cocoa", vectors_dir,
                                       None)
            original = self._ee2_comparison_block(
                example, "original", vectors_dir, "cocoa",
                original_site=original_site)
        finally:
            shutil.rmtree(vectors_dir, ignore_errors=True)
            shutil.rmtree(prefix_dir, ignore_errors=True)
        return original["dchi2_vs_reference"]

    def _worker(self, function, example, tatt, high_accuracy, knob,
                baryon, result_path):
        """Worker-side entry: run one evaluation and save the numbers.

        Arguments:
          function      = "single" or "single_fastpt" (one chi2;
                          the second computes the TATT terms with
                          python FAST-PT); "race" or "race_ee2"
                          (fresh, tenth; "race_ee2" takes the
                          nonlinear P(k) from EuclidEmulator2);
                          "bdelta" or "bdrift" (the baryon checks).
          example       = a key of the project's EXAMPLES table.
          tatt          = "1" for the TATT variant, "0" for NLA.
          high_accuracy = "1" for the pushed numerical settings, "0"
                          not.
          knob          = an ACCURACY_KNOBS label evaluated alone, or
                          the empty string for none.
          baryon        = a BARYON_METHODS label switching the bfmt
                          theory block on, or the empty string for
                          none.
          result_path   = file the result is written into as json;
                          the parent reads it back. Progress prints
                          go to the inherited stdout, so the terminal
                          streams them.

        Returns:
          nothing; the result lands in result_path.

        Raises:
          whatever the selected implementation raises; the parent
          then sees a nonzero exit code instead of a result file
          (_run_isolated turns that into a RuntimeError).
        """
        require_cocoa_environment()
        # the flags arrive as text (command-line arguments are
        # strings); comparing against "1" turns them back into
        # booleans
        tatt = tatt == "1"
        high_accuracy = high_accuracy == "1"
        if function in ("single", "single_fastpt"):
            # `knob or None` turns the empty string (no knob) into
            # None: `or` hands back its second operand when the first
            # is empty. The fastpt flag rides in the function name
            # (the race_ee2 pattern) so the argv stays unchanged.
            value = self._single_model_chi2_impl(
                example, tatt, high_accuracy=high_accuracy,
                knob=knob or None, baryon=baryon or None,
                fastpt=(function == "single_fastpt"))
        elif function == "bdelta":
            value = self._baryon_accuracy_delta_impl(baryon,
                                                     knob=knob or None)
        elif function == "bdrift":
            value = self._baryon_drift_chi2_impl(baryon)
        else:
            # "race" and "race_ee2" both land here; the ee2 flag is
            # carried by the function name so the worker argv stays
            # unchanged
            value = list(self._ten_in_a_row_impl(
                example, tatt, ee2=(function == "race_ee2")))
        # json.dump writes the value into the file as json text; the
        # parent reads it back with json.load. The with block closes
        # the file even when the dump fails
        with open(result_path, "w") as f:
            json.dump(value, f)

    def _sequence_build(self, ctx, example, overrides=None, tatt=False):
        """Build one frozen example's model into the worker context.

        Arguments:
          ctx       = the worker context; model, like, point and space
                      are (re)placed.
          example   = a key of the project's EXAMPLES table.
          overrides = optional {likelihood key: value}.
          tatt      = True builds the TATT variant (frozen TATT point).
        """
        import importlib
        info = self.load_frozen_info(example, tatt=tatt)
        lk_name = self.examples[example]["likelihood"]
        for key, value in (overrides or {}).items():
            info["likelihood"][lk_name][key] = value
        ctx["model"] = make_model(info)
        ctx["like"] = ctx["model"].likelihood[lk_name]
        ctx["point"] = self.build_point(ctx["model"], example, tatt=tatt)
        ctx["ci"] = importlib.import_module(self.interface_module)
        # the native space of the example: a Fourier likelihood carries
        # band centers or an n_cl count
        fourier = (hasattr(ctx["like"], "band_ells")
                   or hasattr(ctx["like"], "ncl"))
        ctx["space"] = "fourier" if fourier else "real"

    def _sequence_worker(self, spec_path, result_path):
        """Worker-side entry of one setting-sequence leg.

        The spec (json) holds the sequence of setting_sequence_suite
        plus "leg": "inprocess" or "fresh", and for the fresh leg
        "upto", the number of steps it applies.

        The in-process leg builds the base, evaluates it through cobaya
        (cosmology and nuisance parameters set, every table filled),
        runs the optional reach check, then applies one step at a time
        and evaluates after each. The fresh leg applies the operations
        of steps 1..upto at once (of "fresh_steps" when the spec gives
        them: the operations a correct code needs for the same final
        state) and evaluates one time. With "fresh":
        "ops_first" (the default) the operations run before the first
        cobaya evaluation, so no table ever saw an earlier setting. With
        "base_first" (steps that change bin counts, which the
        likelihood's own per-point setters cannot follow) the base is
        evaluated first. A step holding a "rebuild" starts the fresh
        leg from that model.

        Arguments:
          spec_path   = the json file of the spec.
          result_path = the npz file the vectors are written to (v0 =
                        the base, v<k> = after step k; the fresh leg
                        writes v<upto>).
        """
        import numpy as np
        require_cocoa_environment()
        with open(spec_path) as f:
            spec = json.load(f)
        ctx = {"harness": self, "scratch": os.path.dirname(result_path)}
        evaluators = spec["evaluate"]
        if isinstance(evaluators, str):
            evaluators = [evaluators]

        def evaluate():
            parts = []
            for name in evaluators:
                parts.append(SEQUENCE_EVALUATORS[name](ctx))
            return np.concatenate(parts)

        def apply(ops):
            for name, kwargs in ops:
                SEQUENCE_OPS[name](ctx, **kwargs)

        def check():
            if spec.get("check"):
                name, kwargs = spec["check"]
                SEQUENCE_CHECKS[name](ctx, **kwargs)

        vectors = {}
        tatt = bool(spec.get("tatt", False))
        if spec["leg"] == "inprocess":
            self._sequence_build(ctx, spec["base"], spec.get("overrides"), tatt)
            evaluate_chi2(ctx["model"], ctx["point"])
            check()
            vectors["v0"] = evaluate()
            for k, step in enumerate(spec["steps"]):
                apply(step)
                vectors[f"v{k + 1}"] = evaluate()
        else:
            upto = int(spec["upto"])
            base = spec["base"]
            overrides = spec.get("overrides")
            ops = []
            for step in spec.get("fresh_steps", spec["steps"])[:upto]:
                for name, kwargs in step:
                    if name == "rebuild":
                        # the fresh leg starts from the rebuilt model
                        base = kwargs["example"]
                        overrides = kwargs.get("overrides")
                        tatt = False
                        ops = []
                    else:
                        ops.append((name, kwargs))
            self._sequence_build(ctx, base, overrides, tatt)
            if spec.get("fresh", "ops_first") == "ops_first":
                apply(ops)
                evaluate_chi2(ctx["model"], ctx["point"])
                check()
            else:
                evaluate_chi2(ctx["model"], ctx["point"])
                check()
                apply(ops)
            vectors[f"v{upto}"] = evaluate()
        np.savez(result_path, **vectors)

    def run_sequence_leg(self, spec, leg, upto=None):
        """Spawn one setting-sequence worker and hand back its output.

        Arguments:
          spec = the sequence (see setting_sequence_suite).
          leg  = "inprocess" or "fresh".
          upto = the fresh leg's number of applied steps.

        Returns:
          (exit code, captured output text, {name: vector} or None).
        """
        import subprocess
        import sys

        import numpy as np
        folder = tempfile.mkdtemp(prefix="cocoa_sequence_")
        try:
            spec_path = os.path.join(folder, "spec.json")
            result_path = os.path.join(folder, "result.npz")
            leg_spec = dict(spec)
            leg_spec["leg"] = leg
            leg_spec["upto"] = upto
            with open(spec_path, "w") as f:
                json.dump(leg_spec, f)
            environment = dict(os.environ)
            environment[_WORKER_FLAG] = "1"
            environment["OMP_NUM_THREADS"] = REQUIRED_OMP_THREADS
            # stdout and stderr are captured together: an expected fatal
            # is recognized by its message, and a failure prints them
            completed = subprocess.run(
                [sys.executable, "-c", _SEQUENCE_DRIVER, self.worker_file,
                 spec_path, result_path],
                env=environment, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True)
            vectors = None
            if completed.returncode == 0 and os.path.exists(result_path):
                with np.load(result_path) as data:
                    vectors = {}
                    for name in data.files:
                        vectors[name] = np.array(data[name])
            return completed.returncode, completed.stdout, vectors
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def _run_isolated(self, function, example, tatt,
                      high_accuracy=False, knob=None, baryon=None):
        """Spawn one worker subprocess and hand back its result.

        Every test quantity runs in its own worker: in a project
        whose examples use different data-set dimensions, the
        cosmolike C layer aborts a process that initializes both,
        and every project keeps one architecture so they all behave
        identically.

        Arguments:
          function      = "single", "race", "race_ee2", "bdelta", or
                          "bdrift" (see _worker).
          example       = a key of the project's EXAMPLES table.
          tatt          = True for the TATT variant.
          high_accuracy = True for the pushed numerical settings.
          knob          = an ACCURACY_KNOBS label evaluated alone, or
                          None.
          baryon        = a BARYON_METHODS label, or None.

        Returns:
          the json value the worker wrote: a float for "single"
          (None when a baryon method rejected the fiducial), "bdelta"
          and "bdrift"; a two-element list [fresh, tenth] for "race"
          and "race_ee2".

        Raises:
          RuntimeError naming the configuration when the worker dies
          without writing a result (the cosmolike C layer aborts the
          process on an internal inconsistency instead of raising).
        """
        import subprocess
        import sys

        # delete=False keeps the file when the with block closes it:
        # only a fresh unique NAME is needed here; the worker writes
        # the file and the finally below removes it
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as tmp:
            result_path = tmp.name
        # dict(os.environ) is a COPY of the environment: the edits
        # below reach only the worker subprocess, never this process
        environment = dict(os.environ)
        environment[_WORKER_FLAG] = "1"
        # OpenMP reads this at library load inside the fresh worker,
        # so the requirement holds for every spawned evaluation
        environment["OMP_NUM_THREADS"] = REQUIRED_OMP_THREADS
        # subprocess.run starts the worker and BLOCKS until it exits;
        # env=environment hands the child the edited environment
        # copy. The booleans travel as "1"/"0" text (command-line
        # arguments are strings; _worker decodes them), and `knob or
        # ""` turns None into the empty string the same way
        completed = subprocess.run(
            [sys.executable, "-c", _WORKER_DRIVER, self.worker_file,
             function, example, "1" if tatt else "0",
             "1" if high_accuracy else "0",
             knob or "", baryon or "", result_path],
            env=environment)
        # the finally below runs on EVERY exit from the try, an
        # exception included, so once the worker has exited the
        # temporary file never outlives this call
        try:
            if completed.returncode != 0:
                raise RuntimeError(
                    f"worker for ({function}, {example}, tatt={tatt}) "
                    f"exited with code {completed.returncode} before "
                    "writing a result; a cosmolike-level abort prints "
                    "its reason (e.g. IP::set_mask) just above")
            # json.load parses the worker's file back into the value
            # the worker json.dump-ed (a float, or a two-element
            # list)
            with open(result_path) as f:
                return json.load(f)
        finally:
            if os.path.exists(result_path):
                os.unlink(result_path)

    # ---- test quantities ----------------------------------------------------
    def single_model_chi2(self, example, tatt, high_accuracy=False,
                          knob=None, baryon=None, fastpt=False):
        """Evaluate the chi2 of the frozen fiducial point in a fresh worker.

        This is the quantity the reference tests compare against the
        frozen reference and the quantity the generator stores as
        that reference. The evaluation runs in a subprocess (see
        _run_isolated for why isolation is mandatory).

        Arguments:
          example = a key of the project's EXAMPLES table.
          tatt    = True evaluates the TATT variant, False the NLA
                    one.
          high_accuracy = True evaluates with the pushed numerical
                    settings; expect minutes instead of seconds.
          knob    = None, or the label of one ACCURACY_KNOBS entry;
                    forwarded to the worker, which applies that
                    knob's overrides alone (the one-at-a-time scan of
                    test_accuracy.py).
          baryon  = None, or a BARYON_METHODS label.
          fastpt  = True computes the TATT terms with the python
                    FAST-PT package at FASTPT_FROZEN_SETTINGS (the
                    frozen FAST-PT reference checks; see
                    load_frozen_info).

        Returns:
          the chi2 as a float, or None when a baryon method rejected
          the frozen fiducial.

        Raises:
          RuntimeError when the worker dies without writing a result
          (_run_isolated).
        """
        # .get returns None when the variable is absent, so a normal
        # (parent) process fails this test and spawns a worker
        # instead
        if os.environ.get(_WORKER_FLAG) == "1":
            return self._single_model_chi2_impl(
                example, tatt, high_accuracy=high_accuracy, knob=knob,
                baryon=baryon, fastpt=fastpt)
        # the fastpt flag travels in the worker function name (the
        # race_ee2 pattern), so the worker argv stays unchanged
        value = self._run_isolated(
            "single_fastpt" if fastpt else "single", example, tatt,
            high_accuracy=high_accuracy, knob=knob, baryon=baryon)
        # a None from the worker means a baryon method rejected the
        # frozen fiducial (see _single_model_chi2_impl); it travels
        # as json null
        return None if value is None else float(value)

    def baryon_accuracy_delta(self, baryon, knob=None):
        """Compute one feedback method's delta chi2 in a fresh worker.

        See _baryon_accuracy_delta_impl for the mechanism (the
        synthetic on-the-fly data vector). Both models of the pair
        run inside ONE worker subprocess.

        Arguments:
          baryon = a BARYON_METHODS label.
          knob   = None for the all-knobs high-accuracy comparison,
                   or an ACCURACY_KNOBS label evaluated alone.

        Returns:
          the delta chi2 as a float.

        Raises:
          RuntimeError when the worker dies without writing a result
          (_run_isolated).
        """
        if os.environ.get(_WORKER_FLAG) == "1":
            return self._baryon_accuracy_delta_impl(baryon, knob=knob)
        return float(self._run_isolated("bdelta", "example1", False,
                                        knob=knob, baryon=baryon))

    def baryon_drift_chi2(self, baryon):
        """Compute one feedback method's drift chi2 in a fresh worker.

        See _baryon_drift_chi2_impl for the mechanism (the frozen
        feedback vector written at freeze time).

        Arguments:
          baryon = a BARYON_METHODS label.

        Returns:
          the chi2 as a float.

        Raises:
          RuntimeError when the worker dies without writing a result
          (_run_isolated).
        """
        if os.environ.get(_WORKER_FLAG) == "1":
            return self._baryon_drift_chi2_impl(baryon)
        return float(self._run_isolated("bdrift", "example1", False,
                                        baryon=baryon))

    def ten_in_a_row_chi2(self, example, tatt, ee2=False):
        """Run the race check in one fresh worker subprocess.

        The whole 11-evaluation sequence runs inside ONE worker: the
        race check needs the evaluations to share a model instance,
        and the worker boundary only isolates this configuration from
        the other configurations' dimensions.

        Arguments:
          example = a key of the project's EXAMPLES table.
          tatt    = True runs the TATT variant, False the NLA one.
          ee2     = True runs the row with the nonlinear P(k) from
                    EuclidEmulator2 (non_linear_emul: 1), the EE2
                    race check.

        Returns:
          (fresh, tenth): chi2 of the first fiducial evaluation and
          chi2 of the fiducial as the 10th point of the row, both
          floats.

        Raises:
          RuntimeError when the worker dies without writing a result
          (_run_isolated).
        """
        # .get returns None when the variable is absent, so a normal
        # (parent) process fails this test and spawns a worker
        # instead
        if os.environ.get(_WORKER_FLAG) == "1":
            return self._ten_in_a_row_impl(example, tatt, ee2=ee2)
        # the worker's two-element json list unpacks into the two
        # names; the ee2 flag travels as the function name so the
        # worker argv stays unchanged
        fresh, tenth = self._run_isolated(
            "race_ee2" if ee2 else "race", example, tatt)
        return float(fresh), float(tenth)

    # ---- the CFASTPT vs FASTPT comparison -----------------------------------
    def _fastpt_comparison_info(self, example, fastpt, high,
                                fastpt_settings, mask="frozen"):
        """Build the cobaya input of one comparison block.

        Starts from the project's frozen TATT configuration
        (load_frozen_info) and applies the comparison's two
        adjustments: IA_code selects the perturbation-theory
        implementation (0 keeps cfastpt, 1 asks the python FAST-PT
        package through the fastpt theory block, which is added here
        with the path the examples use), and fastpt_settings becomes
        that block's extra_args.

        A project can FIX some TATT second-order parameters
        (value: 0.0) in its configuration instead of sampling them;
        the comparison varies all five IA parameters, so any fixed
        comparison parameter is promoted to a sampled one carrying
        its prior box (a no-op in the projects that sample all five).

        Arguments:
          example = a key of the project's EXAMPLES table; the
                    comparison uses "example1" (cosmic shear).
          fastpt  = False keeps cfastpt (IA_code 0), True selects
                    python FAST-PT (IA_code 1).
          high    = True applies the HIGH_ACCURACY settings on top
                    (the --high=1 option of the tests).
          fastpt_settings = None, or the fastpt block's extra_args
                    table (FASTPT_LOW_SETTINGS or
                    FASTPT_HIGH_SETTINGS).
          mask    = a fastpt_masks entry: "frozen" (the default)
                    keeps the example's own tatt_dataset; any other
                    name swaps the data_file for the frozen variant
                    "<tatt_dataset stem>_<mask>.dataset", identical
                    except for its mask_file line (the --mask option
                    of the tests).

        Returns:
          the input dictionary ready for cobaya's get_model.

        Raises:
          ValueError when mask is not a fastpt_masks entry, or from
          load_frozen_info when example names an emulator entry (the
          comparison never evaluates one).
        """
        if mask not in self.fastpt_masks:
            raise ValueError(
                f"mask {mask!r} is not offered by this project; the "
                f"choices are {self.fastpt_masks}")
        info = self.load_frozen_info(example, tatt=True,
                                     high_accuracy=high)
        likelihood_block = info["likelihood"][
            self.examples[example]["likelihood"]]
        if mask != "frozen":
            # the variant descriptor follows the naming rule the
            # class docstring states; str.removesuffix drops the
            # ".dataset" tail so the mask lands before it
            stem = self.examples[example]["tatt_dataset"].removesuffix(
                ".dataset")
            likelihood_block["data_file"] = f"{stem}_{mask}.dataset"
        if fastpt:
            likelihood_block["IA_code"] = 1
            # setdefault hands back the existing "theory" dictionary,
            # or first inserts the empty {} and hands that back, so
            # the assignment lands inside info either way
            info.setdefault("theory", {})["fastpt"] = {
                "path": "./external_modules/code/FAST-PT",
            }
            if fastpt_settings is not None:
                # dict(...) copies, so a caller's settings table is
                # never shared with (or mutated through) the built
                # model
                info["theory"]["fastpt"]["extra_args"] = dict(
                    fastpt_settings)

        # promote any fixed comparison parameter to a sampled one
        # with its prior box; the evaluation base gains it with a
        # zero default in _fastpt_comparison_block. ref and proposal
        # only serve a sampler, which these tests never run.
        for name in self.fastpt_points[0]:
            spec = info["params"].get(name)
            if (isinstance(spec, dict) and "value" in spec
                    and "prior" not in spec):
                lo, hi = ((0.0, 2.0) if name.endswith("BTA_1")
                          else (-5.0, 5.0))
                info["params"][name] = {
                    "prior": {"min": lo, "max": hi},
                    "ref": spec["value"],
                    "proposal": 0.1,
                    "latex": spec.get("latex", name),
                }
        return info

    def _fastpt_comparison_block(self, example, fastpt, high,
                                 fastpt_settings=None, label=None,
                                 vectors_dir=None,
                                 reference_label=None, mask="frozen"):
        """Run the 30-point sweep on ONE model: the comparison's worker half.

        Builds the frozen TATT configuration with one
        perturbation-theory implementation selected and evaluates
        every comparison point on it. Each evaluation prints its
        theory data vector into vectors_dir, and a block given a
        reference_label measures itself against the vectors a
        previous block left there: at every point it computes

            delta chi2 = delta^T C^-1 delta,
            delta = dv(this block) - dv(reference block),

        with C^-1 the masked inverse covariance from the compiled
        interface. That is exactly the chi2 this block would score
        against a dataset whose data vector IS the reference block's
        prediction at the same point: the reference block's own chi2
        against it is zero, so the number is a pure second-order
        deviation and never rides the slope of the distance to the
        shipped data.

        The vector contract: every printed theory vector keeps the
        FULL data-vector length [n_full] of the frozen configuration;
        masking never shortens it. The inverse covariance is
        [n_full, n_full] as the compiled interface supplies it, with
        the rows and columns of masked entries zeroed, so excluded
        entries carry zero weight in the quadratic form instead of
        being dropped. Each covariance entry carries the product of
        its two observables' units, which is what makes
        delta^T C^-1 delta dimensionless.

        This function is meant to run inside a FRESH python process
        (see cfastpt_vs_fastpt_chi2s): process isolation is what
        guarantees that nothing computed under the other
        implementation, or under the other accuracy settings,
        survives into this block: cobaya's caches, CAMB's state, and
        the C globals of the compiled cosmolike interface all die
        with their process, so there is no cache to flush by hand.

        Arguments:
          example = a key of the project's EXAMPLES table.
          fastpt  = False evaluates with cfastpt (IA_code 0), True
                    with python FAST-PT (IA_code 1).
          high    = False keeps the frozen default settings; True
                    applies the HIGH_ACCURACY settings on top.
          fastpt_settings = None for the cfastpt block; the fastpt
                    blocks pass FASTPT_LOW_SETTINGS or
                    FASTPT_HIGH_SETTINGS.
          label   = the file-name tag of this block's printed
                    vectors.
          vectors_dir = the directory the per-point vectors are
                    printed into, shared by the three blocks of one
                    sweep.
          reference_label = None to only print vectors, or the label
                    of the block to measure against.
          mask    = a fastpt_masks entry, forwarded to
                    _fastpt_comparison_info (see there): the
                    scale-cut mask this block's dataset carries.
                    Under a non-frozen mask the shipped TATT data
                    vector does not apply (it was generated under
                    the frozen mask), so the block regenerates the
                    baseline: the CFASTPT fiducial vector under the
                    chosen mask becomes the sweep's data vector and
                    every reported chi2 is measured against it,
                    from a zero baseline.

        Returns:
          {"chi2s": the per-point chi2 list (informational; against
          the shipped data under the frozen mask, against the CFASTPT
          fiducial vector under any other), "dchi2_vs_reference": the
          per-point delta^T C^-1 delta list or None, "eval_seconds":
          per-point wall-clock seconds}. The block's first evaluation
          carries CAMB plus the PT tables: under the frozen mask that
          is eval_seconds[0]; under another mask it is the baseline
          evaluation, timed and printed separately before the loop.

        Raises:
          AssertionError when the sampled-parameter set differs from
          the frozen point plus the promoted IA parameters;
          RuntimeError when an evaluation did not print its vector.
        """
        import time

        import numpy as np

        require_cocoa_environment()
        code = "FASTPT" if fastpt else "CFASTPT"
        if fastpt_settings is not None:
            # name the fastpt settings by their accuracyboost, which
            # differs between the low and high tables
            code += f"(boost {fastpt_settings['accuracyboost']:g})"
        setting = "high accuracy" if high else "default settings"
        print(f"  building model ({example}, TATT, {code}, {setting}, "
              f"mask {mask}) ...", flush=True)
        info = self._fastpt_comparison_info(example, fastpt, high,
                                            fastpt_settings, mask=mask)
        # every evaluation rewrites this one file; the loop below
        # moves it to a per-point name right after each evaluation
        current_path = os.path.join(vectors_dir,
                                    f"{label}_current.modelvector")
        likelihood_block = info["likelihood"][
            self.examples[example]["likelihood"]]
        likelihood_block["print_datavector"] = True
        likelihood_block["print_datavector_file"] = current_path
        model = make_model(info)
        # the frozen point does not carry a promoted parameter (it is
        # fixed in the shipped configuration), so the evaluation base
        # is assembled here: the frozen point plus a zero for each
        # promoted name, checked against the model's sampled set the
        # same way build_point checks the unpromoted configurations
        base = dict(self.load_frozen_point(example))
        for name in self.fastpt_points[0]:
            if name not in base:
                base[name] = 0.0
        sampled = set(model.parameterization.sampled_params())
        if sampled != set(base):
            raise AssertionError(
                "sampled-parameter set differs from the frozen point "
                "plus the promoted IA parameters: model only "
                f"{sorted(sampled - set(base))}; point only "
                f"{sorted(set(base) - sampled)}")
        n_points = len(self.fastpt_points)
        fiducial_truth = None
        if mask != "frozen":
            # Under a non-frozen mask the shipped TATT data vector
            # does not apply: it was generated under the frozen
            # contract's mask, so rows this mask newly unmasks would
            # be compared against zeros and the chi2 against it
            # would be meaningless. The baseline is regenerated
            # instead, the same construction the accuracy checks use
            # when they change cosmology: the fiducial TATT point is
            # evaluated first, and the CFASTPT block's printed
            # vector at it becomes the data vector of the whole
            # sweep, so the reported chi2 starts from a zero
            # baseline (CFASTPT at the fiducial scores exactly
            # zero). The fastpt blocks read the cfastpt fiducial
            # from vectors_dir, where the cfastpt block (always
            # first) left it.
            started = time.perf_counter()
            _evaluate_cached(model, base)
            elapsed = time.perf_counter() - started
            if not os.path.isfile(current_path):
                raise RuntimeError(
                    f"{code} fiducial: the evaluation printed no "
                    f"data vector at {current_path}")
            os.replace(current_path,
                       os.path.join(vectors_dir,
                                    f"{label}_fiducial.modelvector"))
            # the compiled interface was initialized by the
            # likelihood build above; its masked inverse covariance
            # carries the chosen mask
            import importlib

            ci = importlib.import_module(self.interface_module)
            icov_fid = np.array(ci.get_inv_cov_masked())
            # "cfastpt" is the label the driver gives the reference
            # block, which always runs first, so this file exists
            # for every block (the cfastpt block reads its own)
            fiducial_truth = _load_datavector(
                os.path.join(vectors_dir,
                             "cfastpt_fiducial.modelvector"))
            print(f"  {code} fiducial under mask {mask}: baseline "
                  f"regenerated ({elapsed:.2f} s); the chi2 below "
                  "is measured against the CFASTPT fiducial vector",
                  flush=True)
        chi2s = []
        eval_seconds = []
        # enumerate pairs each point with a counter; start=1 makes
        # the printed rows read 1..30 instead of 0..29
        for i, ia_values in enumerate(self.fastpt_points, start=1):
            # perf_counter is a monotonic wall clock; the difference
            # of two readings is the elapsed time of the evaluation
            # alone
            started = time.perf_counter()
            # {**base, **ia_values} builds a NEW dictionary: the
            # fiducial entries first, then the five IA values
            # replacing their fiducial counterparts; base itself
            # stays untouched
            chi2 = _evaluate_cached(model, {**base, **ia_values})
            elapsed = time.perf_counter() - started
            eval_seconds.append(elapsed)
            if not os.path.isfile(current_path):
                raise RuntimeError(
                    f"{code} point {i}: the evaluation printed no "
                    f"data vector at {current_path} (did "
                    "print_datavector's path handling change?)")
            # os.replace moves the freshly printed vector to its
            # per-point name; the next evaluation must write
            # current_path anew, which the isfile check above
            # enforces point by point
            os.replace(current_path,
                       os.path.join(vectors_dir,
                                    f"{label}_point{i:02d}.modelvector"))
            if fiducial_truth is not None:
                # the reported chi2 is measured against the
                # regenerated baseline: the quadratic form of this
                # point's printed vector against the CFASTPT
                # fiducial vector, which IS the chi2 against a
                # dataset whose data vector is that fiducial (the
                # likelihood's own chi2 is measured against the
                # frozen dataset's data vector, not this baseline,
                # and is discarded)
                own = _load_datavector(
                    os.path.join(vectors_dir,
                                 f"{label}_point{i:02d}.modelvector"))
                delta = own - fiducial_truth
                chi2 = float(delta @ icov_fid @ delta)
            chi2s.append(chi2)
            print(f"  {code} point {i:2d}/{n_points}: chi2 = "
                  f"{chi2:.6f}   ({elapsed:.2f} s)", flush=True)
        # the first evaluation carries the one-time work (CAMB plus
        # this implementation's perturbation-theory tables); the mean
        # of the rest is the steady per-point cost. eval_seconds[1:]
        # is the list without its first entry.
        print(f"  {code}: first evaluation {eval_seconds[0]:.2f} s "
              f"(CAMB + PT tables + data vector); later evaluations "
              f"mean {float(np.mean(eval_seconds[1:])):.3f} s",
              flush=True)
        if reference_label is None:
            return {"chi2s": chi2s, "dchi2_vs_reference": None,
                    "eval_seconds": eval_seconds}
        # the compiled interface was initialized by the likelihood
        # build above, so its masked inverse covariance (masked rows
        # and columns zeroed, full data-vector dimensions) is
        # available here; the import resolves because start_cocoa.sh
        # puts the project interface folder on the python path
        import importlib

        ci = importlib.import_module(self.interface_module)

        icov = np.array(ci.get_inv_cov_masked())
        dchi2s = []
        for i in range(1, n_points + 1):
            own = _load_datavector(
                os.path.join(vectors_dir,
                             f"{label}_point{i:02d}.modelvector"))
            ref = _load_datavector(
                os.path.join(vectors_dir,
                             f"{reference_label}_point{i:02d}"
                             ".modelvector"))
            if own.shape != ref.shape or icov.shape[0] != own.shape[0]:
                raise RuntimeError(
                    f"point {i}: vector/covariance shapes disagree "
                    f"({own.shape}, {ref.shape}, {icov.shape})")
            delta = own - ref
            # delta @ icov @ delta is the quadratic form delta^T C^-1
            # delta: the chi2 of the difference vector in units of
            # the statistical error the covariance defines
            dchi2s.append(float(delta @ icov @ delta))
        return {"chi2s": chi2s, "dchi2_vs_reference": dchi2s,
                "eval_seconds": eval_seconds}

    def _run_fastpt_comparison_worker(self, example, fastpt, high,
                                      fastpt_settings, label,
                                      vectors_dir, reference_label,
                                      mask="frozen"):
        """Run one _fastpt_comparison_block in a fresh python subprocess.

        A fresh process is the cache flush: cobaya's component
        caches, CAMB, and the C globals of the compiled cosmolike
        interface start empty in each block, so no block can reuse
        anything another block computed. The child inherits this
        process's environment, sets OMP_NUM_THREADS before its first
        import, adds the project's tests/ to its import path, calls
        _fastpt_comparison_block, and writes the result as json into
        a temporary file the parent reads back; its per-point
        progress lines stream to the same terminal because stdout is
        inherited.

        Arguments:
          example, fastpt, high, fastpt_settings, label, vectors_dir,
          reference_label, mask = forwarded to
          _fastpt_comparison_block.

        Returns:
          the block's result dictionary.

        Raises:
          subprocess.CalledProcessError when the worker fails (its
          traceback already streamed to the terminal); RuntimeError
          when the worker returns the wrong number of chi2 values.
        """
        import subprocess
        import sys

        workdir = tempfile.mkdtemp(prefix="cocoa_fastpt_compare_")
        out_path = os.path.join(workdir, "result.json")
        # the child program, line by line: pin the OpenMP thread
        # count BEFORE any import can load the compiled libraries,
        # make the project's tests/ importable, run the block, dump
        # the result. !r prints each interpolated value as python
        # source, so the generated program is valid python.
        child_code = (
            "import os\n"
            f"os.environ['OMP_NUM_THREADS'] = "
            f"{REQUIRED_OMP_THREADS!r}\n"
            "import json\n"
            "import sys\n"
            f"sys.path.insert(0, {self.tests_dir!r})\n"
            "import cocoa_test_utils as u\n"
            f"result = u._fastpt_comparison_block({example!r}, "
            f"fastpt={fastpt!r}, high={high!r}, "
            f"fastpt_settings={fastpt_settings!r}, "
            f"label={label!r}, vectors_dir={vectors_dir!r}, "
            f"reference_label={reference_label!r}, mask={mask!r})\n"
            f"with open({out_path!r}, 'w') as f:\n"
            "    json.dump(result, f)\n"
        )
        try:
            # sys.executable is this python interpreter; check=True
            # raises when the child exits nonzero (its traceback has
            # already streamed to the inherited stderr)
            subprocess.run([sys.executable, "-c", child_code],
                           check=True)
            with open(out_path) as f:
                result = json.load(f)
        finally:
            # a finally block runs on EVERY exit from the try, so no
            # failure mode leaves the temporary directory behind
            shutil.rmtree(workdir, ignore_errors=True)
        if len(result["chi2s"]) != len(self.fastpt_points):
            raise RuntimeError(
                f"worker returned {len(result['chi2s'])} chi2 values "
                f"for {len(self.fastpt_points)} comparison points")
        return result

    def cfastpt_vs_fastpt_chi2s(self, example, high=False,
                                mask="frozen"):
        """Compute the per-point quantities under the three configurations.

        The same 30 hard-coded intrinsic-alignment points evaluated
        three times with everything else identical:

          1. cfastpt (IA_code 0), the reference: its printed data
             vector at each point becomes the fiducial the other
             blocks are measured against (its own chi2 against that
             vector is zero by construction);
          2. python FAST-PT (IA_code 1) at FASTPT_LOW_SETTINGS, the
             pass configuration;
          3. python FAST-PT at FASTPT_HIGH_SETTINGS, the doubled
             boosts.

        Block 2's per-point delta^T C^-1 delta against block 1 is the
        pass/fail quantity; block 3's shows how the deviation
        responds to the FAST-PT grids (advisory). Each configuration
        runs in its own subprocess, one after the other; the
        per-point vectors travel through one shared temporary
        directory that dies with this call.

        Arguments:
          example = a key of the project's EXAMPLES table; the
                    comparison uses "example1".
          high    = False compares at the frozen default
                    camb/cosmolike settings; True repeats all three
                    blocks with the HIGH_ACCURACY settings (the
                    --high=1 command line option of the tests).
          mask    = a fastpt_masks entry, applied to all three
                    blocks: "frozen" (the default) keeps each
                    example's own tatt_dataset; any other name swaps
                    in the matching frozen dataset variant (the
                    --mask command line option of the tests).

        Returns:
          (chi2_cfastpt, chi2_fastpt_low, chi2_fastpt_high,
          dchi2_low, dchi2_high): five lists index-aligned with the
          comparison points.

        Raises:
          subprocess.CalledProcessError when a block's worker fails;
          RuntimeError when a worker returns the wrong number of
          chi2 values (_run_fastpt_comparison_worker).
        """
        vectors_dir = tempfile.mkdtemp(prefix="cocoa_fastpt_vectors_")
        try:
            cfastpt = self._run_fastpt_comparison_worker(
                example, fastpt=False, high=high, fastpt_settings=None,
                label="cfastpt", vectors_dir=vectors_dir,
                reference_label=None, mask=mask)
            fastpt_low = self._run_fastpt_comparison_worker(
                example, fastpt=True, high=high,
                fastpt_settings=self.fastpt_low_settings,
                label="fastpt_low", vectors_dir=vectors_dir,
                reference_label="cfastpt", mask=mask)
            fastpt_high = self._run_fastpt_comparison_worker(
                example, fastpt=True, high=high,
                fastpt_settings=self.fastpt_high_settings,
                label="fastpt_high", vectors_dir=vectors_dir,
                reference_label="cfastpt", mask=mask)
        finally:
            # the shared vectors die with the sweep, whether it
            # finished or an exception is on its way out
            shutil.rmtree(vectors_dir, ignore_errors=True)
        return (cfastpt["chi2s"], fastpt_low["chi2s"],
                fastpt_high["chi2s"],
                fastpt_low["dchi2_vs_reference"],
                fastpt_high["dchi2_vs_reference"])

    def _nonlinear_comparison_block(self, example, emul, label=None,
                                    vectors_dir=None,
                                    reference_label=None,
                                    mask="frozen"):
        """Run the ten-cosmology sweep on ONE nonlinear-P(k) source.

        The worker half of the Halofit-vs-EE2 advisory checks
        (NL1/NL2). Builds the frozen NLA configuration with one
        nonlinear-P(k) source selected (non_linear_emul 1 =
        EuclidEmulator2, 2 = CAMB's Takahashi halofit) and evaluates
        every NONLINEAR_COMPARISON_POINTS entry on it. Each
        evaluation prints its theory data vector into vectors_dir,
        and a block given a reference_label measures itself against
        the vectors a previous block left there: at every cosmology

            delta chi2 = delta^T C^-1 delta,
            delta = dv(this block) - dv(reference block),

        with C^-1 the masked inverse covariance of the chosen mask.
        The reference block's own chi2 against its vector is zero by
        construction, so the number IS the chi2 the Halofit vector
        would score against a dataset whose data vector is the EE2
        prediction at the same cosmology: a zero-baseline
        measurement at every point. Each cosmology carries its own
        regenerated fiducial, so no stored data vector enters the
        metric anywhere.

        Runs inside a FRESH python process
        (_run_nonlinear_comparison_worker); unlike the FASTPT sweep,
        every point changes the cosmology, so each evaluation pays a
        CAMB run. Ten points stay cheap.

        Arguments:
          example = a key of the project's EXAMPLES table; NL1 uses
                    the cosmic-shear example, NL2 the 3x2pt one.
          emul    = the likelihood's non_linear_emul: 1 evaluates
                    with EuclidEmulator2, 2 with CAMB's Takahashi
                    halofit.
          label   = the file-name tag of this block's printed
                    vectors ("ee2", "halofit").
          vectors_dir = the directory the per-cosmology vectors are
                    printed into, shared by the two blocks of one
                    sweep.
          reference_label = None to only print vectors (the
                    reference block), or the label of the block to
                    measure against.
          mask    = a fastpt_masks entry: "frozen" (the default)
                    keeps the configuration's own dataset; any other
                    name swaps in the example's tatt_dataset variant
                    "<stem>_<mask>.dataset", whose mask and
                    covariance weight the difference (the data
                    vector it names plays no role in this check's
                    metric, so no baseline regeneration is needed).

        Returns:
          {"dchi2_vs_reference": the per-cosmology delta^T C^-1
          delta list, or None for the reference block,
          "eval_seconds": the per-cosmology wall-clock seconds}.

        Raises:
          ValueError when mask is not a fastpt_masks entry or a
          comparison-point parameter is not sampled by the model;
          RuntimeError when an evaluation did not print its vector
          or the vector/covariance shapes disagree.
        """
        import time

        import numpy as np

        require_cocoa_environment()
        if mask not in self.fastpt_masks:
            raise ValueError(
                f"mask {mask!r} is not offered by this project; the "
                f"choices are {self.fastpt_masks}")
        code = "EE2" if emul == 1 else "HALOFIT"
        print(f"  building model ({example}, NLA, {code}, "
              f"mask {mask}) ...", flush=True)
        info = self.load_frozen_info(example, tatt=False)
        likelihood_block = info["likelihood"][
            self.examples[example]["likelihood"]]
        likelihood_block["non_linear_emul"] = emul
        if mask != "frozen":
            # the variant descriptor follows the fastpt_masks naming
            # rule; its data vector is never read by the metric
            stem = self.examples[example][
                "tatt_dataset"].removesuffix(".dataset")
            likelihood_block["data_file"] = f"{stem}_{mask}.dataset"
        current_path = os.path.join(vectors_dir,
                                    f"{label}_current.modelvector")
        likelihood_block["print_datavector"] = True
        likelihood_block["print_datavector_file"] = current_path
        model = make_model(info)
        base = self.build_point(model, example, tatt=False)
        n_points = len(NONLINEAR_COMPARISON_POINTS)
        eval_seconds = []
        for i, cosmology in enumerate(NONLINEAR_COMPARISON_POINTS,
                                      start=1):
            for name in cosmology:
                if name not in base:
                    raise ValueError(
                        f"comparison parameter {name} is not sampled "
                        f"by {example}; the parameterization changed "
                        "since NONLINEAR_COMPARISON_POINTS was drawn")
            started = time.perf_counter()
            # the chi2 against the dataset's stored vector is
            # discarded: the evaluation's purpose is the printed
            # theory vector. {**base, **cosmology} is a NEW
            # dictionary: the fiducial with this cosmology's three
            # values written over it
            _evaluate_cached(model, {**base, **cosmology})
            elapsed = time.perf_counter() - started
            eval_seconds.append(elapsed)
            if not os.path.isfile(current_path):
                raise RuntimeError(
                    f"{code} cosmology {i}: the evaluation printed "
                    f"no data vector at {current_path} (did "
                    "print_datavector's path handling change?)")
            os.replace(current_path,
                       os.path.join(vectors_dir,
                                    f"{label}_point{i:02d}"
                                    ".modelvector"))
            print(f"  {code} cosmology {i:2d}/{n_points} evaluated "
                  f"({elapsed:.2f} s)", flush=True)
        if reference_label is None:
            return {"dchi2_vs_reference": None,
                    "eval_seconds": eval_seconds}
        # the compiled interface was initialized by the likelihood
        # build above, so the masked inverse covariance of the
        # chosen mask is available here
        import importlib

        ci = importlib.import_module(self.interface_module)

        icov = np.array(ci.get_inv_cov_masked())
        dchi2s = []
        for i in range(1, n_points + 1):
            own = _load_datavector(
                os.path.join(vectors_dir,
                             f"{label}_point{i:02d}.modelvector"))
            ref = _load_datavector(
                os.path.join(vectors_dir,
                             f"{reference_label}_point{i:02d}"
                             ".modelvector"))
            if own.shape != ref.shape or icov.shape[0] != own.shape[0]:
                raise RuntimeError(
                    f"cosmology {i}: vector/covariance shapes "
                    f"disagree ({own.shape}, {ref.shape}, "
                    f"{icov.shape})")
            delta = own - ref
            # the quadratic form delta^T C^-1 delta, as in
            # _fastpt_comparison_block
            dchi2s.append(float(delta @ icov @ delta))
        return {"dchi2_vs_reference": dchi2s,
                "eval_seconds": eval_seconds}

    def _run_nonlinear_comparison_worker(self, example, emul, label,
                                         vectors_dir, reference_label,
                                         mask):
        """Run one _nonlinear_comparison_block in a fresh subprocess.

        The same isolation mechanism as
        _run_fastpt_comparison_worker (see there): the fresh process
        is the cache flush; the child pins OMP_NUM_THREADS, puts the
        project's tests/ first on its import path, imports
        cocoa_test_utils, and calls the shim's re-exported
        _nonlinear_comparison_block.

        Arguments:
          example, emul, label, vectors_dir, reference_label, mask =
                    forwarded to _nonlinear_comparison_block.

        Returns:
          the block's result dictionary.

        Raises:
          subprocess.CalledProcessError when the worker fails (its
          traceback already streamed to the terminal).
        """
        import subprocess
        import sys

        workdir = tempfile.mkdtemp(prefix="cocoa_nonlinear_compare_")
        out_path = os.path.join(workdir, "result.json")
        child_code = (
            "import os\n"
            f"os.environ['OMP_NUM_THREADS'] = "
            f"{REQUIRED_OMP_THREADS!r}\n"
            "import json\n"
            "import sys\n"
            f"sys.path.insert(0, {self.tests_dir!r})\n"
            "import cocoa_test_utils as u\n"
            f"result = u._nonlinear_comparison_block({example!r}, "
            f"emul={emul!r}, label={label!r}, "
            f"vectors_dir={vectors_dir!r}, "
            f"reference_label={reference_label!r}, mask={mask!r})\n"
            f"with open({out_path!r}, 'w') as f:\n"
            "    json.dump(result, f)\n"
        )
        try:
            subprocess.run([sys.executable, "-c", child_code],
                           check=True)
            with open(out_path) as f:
                result = json.load(f)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        return result

    def halofit_vs_ee2_dchi2s(self, example, mask="frozen"):
        """Compute the per-cosmology Halofit-vs-EE2 differences (NL1/NL2).

        The ten hard-coded cosmologies
        (NONLINEAR_COMPARISON_POINTS) evaluated twice with
        everything else identical:

          1. EuclidEmulator2 (non_linear_emul 1), the reference of
             the comparison (a chosen baseline, not exact physics):
             its printed data vector at each cosmology becomes the
             fiducial the Halofit block is measured against;
          2. CAMB's Takahashi halofit (non_linear_emul 2, the frozen
             contracts' setting).

        Block 2 returns, at every cosmology, the chi2 of its
        data-vector difference against block 1 (delta^T C^-1 delta
        under the chosen mask). The check is advisory: the numbers
        say how much of the statistical error budget the
        Halofit-vs-emulator difference consumes under those scale
        cuts. Each block runs in its own subprocess; the
        per-cosmology vectors travel through one shared temporary
        directory that dies with this call.

        Arguments:
          example = a key of the project's EXAMPLES table; NL1 uses
                    the cosmic-shear example, NL2 the 3x2pt one.
          mask    = a fastpt_masks entry (the --mask command line
                    option of the tests).

        Returns:
          the per-cosmology delta^T C^-1 delta list, index-aligned
          with NONLINEAR_COMPARISON_POINTS.

        Raises:
          subprocess.CalledProcessError when a block's worker fails
          (_run_nonlinear_comparison_worker).
        """
        vectors_dir = tempfile.mkdtemp(
            prefix="cocoa_nonlinear_vectors_")
        try:
            self._run_nonlinear_comparison_worker(
                example, emul=1, label="ee2", vectors_dir=vectors_dir,
                reference_label=None, mask=mask)
            halofit = self._run_nonlinear_comparison_worker(
                example, emul=2, label="halofit",
                vectors_dir=vectors_dir, reference_label="ee2",
                mask=mask)
        finally:
            shutil.rmtree(vectors_dir, ignore_errors=True)
        return halofit["dchi2_vs_reference"]


# =============================================================================
# SHARED pytest CONFTEST IMPLEMENTATION
# =============================================================================
# pytest discovers conftest.py only by walking up from the collected
# test files, so every project must keep a physical conftest.py inside
# its tests/ folder. The IMPLEMENTATION lives here once; a project's
# conftest.py reduces to a shim binding these two functions:
#
#     import os
#     import sys
#     sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#     import cocoa_test_utils as u
#
#     def pytest_addoption(parser):
#         u._cct.conftest_addoption(parser, u._H.fastpt_masks)
#
#     def pytest_configure(config):
#         u._cct.conftest_configure(config)
#
# A run collecting several projects at once loads each project's
# conftest.py; the FIRST registration of an option name wins and the
# later ones step aside (the try/except below), so in a multi-project
# run the --mask choices are the first project's. The chosen values
# still reach every project through the environment variables, and a
# mask a project does not offer is refused by its harness
# (_fastpt_comparison_info and _nonlinear_comparison_block raise
# ValueError).


def conftest_addoption(parser, mask_choices=("frozen",)):
    """Register --high and --mask: the comparison sweeps' switches.

    A project's conftest.py forwards pytest's option parser here.
    --high=0 (the default) runs the CFASTPT-vs-FASTPT sweeps at the
    frozen default settings; --high=1 repeats them with the
    HIGH_ACCURACY settings applied to both implementations (the full
    comparison is both invocations). --mask selects the scale-cut
    mask of the CFASTPT-vs-FASTPT and Halofit-vs-EE2 sweeps:
    "frozen" (the default) keeps each example's own dataset, and
    every other offered name selects the matching frozen dataset
    variant.

    Arguments:
      parser       = pytest's option parser (supplied by pytest to
                     the conftest hook).
      mask_choices = the mask names this project offers, normally the
                     harness's fastpt_masks tuple; "frozen" alone by
                     default.

    Returns:
      nothing; the options become readable through config.getoption.
    """
    # action="store" keeps the given text as the option's value;
    # choices rejects anything except the documented settings. The
    # second registration of the same option name (another project's
    # conftest in the same run) raises ValueError; the first
    # registration already serves every project, so this one steps
    # aside.
    try:
        parser.addoption(
            "--high", action="store", default="0", choices=("0", "1"),
            help="1 repeats the CFASTPT-vs-FASTPT comparison at the "
                 "HIGH_ACCURACY settings instead of the frozen "
                 "defaults")
    except ValueError:
        pass
    try:
        parser.addoption(
            "--mask", action="store", default="frozen",
            choices=tuple(mask_choices),
            help="the scale-cut mask of the CFASTPT-vs-FASTPT "
                 "comparison: frozen (the contract mask, the "
                 "default) or another mask this project offers")
    except ValueError:
        pass


def conftest_configure(config):
    """Copy the option values where the test classes read them.

    A project's conftest.py forwards pytest's configuration object
    here after the command line is parsed. The values land in the
    COCOA_FASTPT_HIGH and COCOA_FASTPT_MASK environment variables
    (the tests are unittest.TestCase classes, whose methods cannot
    receive pytest fixtures), which the test modules read with the
    "0" and "frozen" defaults, so a run without the options and a
    run outside pytest behave the same.

    Arguments:
      config = pytest's configuration object (supplied by pytest to
               the conftest hook).

    Returns:
      nothing; the environment of this process gains the variables.
    """
    os.environ["COCOA_FASTPT_HIGH"] = str(config.getoption("--high"))
    os.environ["COCOA_FASTPT_MASK"] = str(config.getoption("--mask"))


def run_matter_emulator_prior_check(
    project,
    configuration,
    get_precision,
    n_models=50,
    seed=137,
    limit=1.0,
    points_file=None,
    fisher=False,
    output_dir=None,
    fisher_sample_power=18,
    fisher_center=None,
    fisher_parameters=None,
    fisher_steps=None,
    fisher_probe=None,
    get_covariance=None,
    advisory=False,
    candidate_extra_args=None,
    use_emulator_switch=True,
    select_vector=None,
    focused_models=25,
    local_points_file=None,
    derivative_provider=None,
    amplitude_bias_cuts=None,
    mpi_comm=None,
    fisher_accuracy_likelihood=None,
    fisher_accuracy_camb=None,
    fisher_fom_tolerance=0.01,
):
    """Compare a candidate matter emulator with CAMB on random cosmologies.

    Project tests call this routine with their ordinary Cobaya evaluate
    example and native precision reader, as the other shared Cocoa tests
    do. Each cosmology changes the theory state; cached=False forces the
    likelihood to run. Random draws retain the example's nuisance override;
    an explicit points_file may also specify the nuisance parameters.
    Both routes use CAMB distances and plain HMCode2020, with mnu=0.06 eV.
    The candidate uses the project's existing EMUL2 likelihood route.

    Draw from each project's cosmological box with 0.2 < Omega_m < 0.5,
    w0 > -2, 1 < 1e9 As <= 3.5 and H0 < 85 for both the mean and Fisher,
    and keep fixed cosmological parameters fixed. Do not redraw failures
    or silently restrict to the emulator training domain. The result is
    delta_model.T @ precision @ delta_model, not a difference between two
    chi-squared values against the measured data. A value below limit is
    the requested agreement criterion at that point, not a proof that
    importance weights have adequate effective sample size.

    Arguments:
      project = project directory below Cocoa/projects.
      configuration = existing evaluate YAML, including its nuisance point.
      get_precision = the initialized native masked precision reader.
      n_models = number of random cosmologies; default 50.
      seed = model i uses numpy.random.default_rng(seed + i).
      limit = strict upper bound on the prediction-distance statistic.
      points_file = optional named-column text table of explicit parameter
        points; supplied values replace the example's overrides. Its row
        count replaces n_models and no random cosmologies are drawn.
      fisher = also compute the CAMB Fisher matrix and bounded-prior plot.
        Its physical theory follows the example, with mnu fixed to 0.06 eV;
        the plain-mead2020 matching applies to mean comparisons only.
      output_dir = optional destination below the project chains directory.
      fisher_sample_power = log2 proposal count per bounded-Fisher repetition.
      fisher_center, fisher_parameters, fisher_steps = optional reference
        point, differentiated parameter names and absolute derivative steps.
      fisher_probe = None for the complete vector, or "xi" for the initial
        real-space shear rows. The latter requires get_covariance.
      get_covariance = initialized native full covariance reader.
      candidate_extra_args = explicit additional emulator theory options.
      use_emulator_switch = whether the likelihood uses the ordinary
        use_emulator/non_linear_emul flags. Provider-only likelihoods do not.
      select_vector = optional native mask/compression callable, applied
        after the full vector is checked against its likelihood.
      advisory = report prediction errors without failing on their size;
        an incomplete Fisher calculation still fails.
      focused_models = additional draws from the bounded Fisher distribution
        after a broad-prior test; zero skips this stage for diagnostics.
      mpi_comm = optional MPI communicator. Each rank owns one live model;
        point indices are distributed in rank order and only rank zero writes.
        Completed checkpoint rows are retained on resubmission.
      local_points_file = optional predetermined focused sample, replacing
        the Fisher draws. It must contain focused_models rows.
      fisher_accuracy_likelihood, fisher_accuracy_camb = optional overrides
        of the explicit next numerical setting used for the Fisher check.
        Without them, native sampling doubles up to the established boost
        cap of three, integration advances one level, CAMB AccuracyBoost
        reaches at least two and k_per_logint at least fifty. 3PCF's separate
        sampling/integration controls double. Existing larger boosts stay.
      fisher_fom_tolerance = relative target-pair FoM tolerance, default 1%,
        required at both derivative steps and between numerical settings.
        This is the test criterion, not a universal science requirement.

    Returns:
      number of failed or nonfinite comparisons. Generated vectors, points,
      status records and a summary are written under the project's chains/.
      EMULATOR_CODE_DIR and EMULATOR_DATA_DIR select candidate checkouts.
      The project driver selects the native interface before calling this
      routine; production configuration files remain unchanged.
    """
    import copy
    import io
    import subprocess
    import sys
    from pathlib import Path

    import numpy as np
    from cobaya.model import get_model
    from cobaya.yaml import yaml_load_file
    from cosmolike_notebook_utils.mpi_batching import run_owned_jobs

    rank = 0 if mpi_comm is None else mpi_comm.Get_rank()
    size = 1 if mpi_comm is None else mpi_comm.Get_size()

    if not isinstance(focused_models, int) or focused_models < 0:
        raise ValueError("focused_models must be a nonnegative integer.")

    project = Path(project).resolve()
    root = project.parents[1]
    source = project / configuration
    code_dir = Path(os.environ.get(
        "EMULATOR_CODE_DIR",
        root / "external_modules/code/emulators/emultrf",
    ))
    data_dir = Path(os.environ.get(
        "EMULATOR_DATA_DIR",
        root / "external_modules/data/emultrf",
    ))
    sys.path.insert(0, str(code_dir))
    from emulmps.emulmps import emulmps

    info = yaml_load_file(str(source))
    base_point = dict(info["sampler"]["evaluate"]["override"])
    base_point.pop("mnu", None)
    info.pop("sampler")
    info.pop("output", None)
    info["params"]["mnu"] = {"value": 0.06}
    # Intersect the example's priors before constructing either theory
    # or Fisher. In CPL, CAMB's w is today's w0; w0pwa has its own prior.
    # Move open endpoints inward by one representable float so that a
    # bounded Fisher draw and an explicitly supplied point obey the same cut.
    w_today = "w0" if "w0" in info["params"] else "w"
    test_bounds = {
        w_today: (float(np.nextafter(-2.0, np.inf)), np.inf),
        "As_1e9": (float(np.nextafter(1.0, np.inf)), 3.5),
        "H0": (-np.inf, float(np.nextafter(85.0, -np.inf))),
        "omegam": (float(np.nextafter(0.2, np.inf)), float(np.nextafter(0.5, -np.inf))),
    }
    for name, (lo, hi) in test_bounds.items():
        spec = info["params"].get(name, -1.0 if name == w_today else None)
        if isinstance(spec, dict) and "prior" in spec:
            prior = spec["prior"]
            lo, hi = max(lo, float(prior["min"])), min(hi, float(prior["max"]))
            if not lo < hi:
                raise ValueError(f"Empty emulator-test prior for {name}.")
            prior.update(min=lo, max=hi)
            test_bounds[name] = (lo, hi)
            value = base_point[name]
        else:
            value = spec.get("value") if isinstance(spec, dict) else spec
        if value is None or not lo <= value <= hi:
            raise ValueError(f"{name} fiducial {value} outside emulator-test bounds.")
    if len(info["likelihood"]) != 1:
        raise ValueError("Choose one project's likelihood for this comparison.")
    likelihood_name = next(iter(info["likelihood"]))
    # Preserve the example's physical theory and ordinary accuracy for
    # Fisher. Only the requested fixed neutrino mass above is shared with
    # the separate matched-mead2020 emulator comparison below.
    fisher_settings = copy.deepcopy(info)
    if use_emulator_switch:
        info["likelihood"][likelihood_name]["non_linear_emul"] = 2
        info["likelihood"][likelihood_name]["print_datavector"] = False
    camb_args = info["theory"]["camb"]["extra_args"]
    camb_args["halofit_version"] = "mead2020"
    camb_args.pop("HMCode_logT_AGN", None)
    camb_args["AccuracyBoost"] = 3.0
    camb_args["k_per_logint"] = 50
    # Background parameters and the number of massive species remain the
    # project's choices. Only the mass and nonlinear prescription change.
    names = ["As_1e9", "ns", "H0", "omegab", "omegam", w_today, "w0pwa", "wa"]
    points = []
    for index in range(n_models if points_file is None else 0):
        random = np.random.default_rng(seed + index)
        point = dict(base_point)
        for name in names:
            specification = info["params"].get(name)
            if not isinstance(specification, dict):
                continue
            prior = specification.get("prior")
            if prior is None:
                continue
            if "min" not in prior or "max" not in prior:
                raise ValueError(f"Cosmological prior for {name} is not a uniform box.")
            lower = float(prior["min"])
            upper = float(prior["max"])
            if lower >= upper:
                raise ValueError(f"Empty requested prior interval for {name}.")
            point[name] = float(random.uniform(low=lower, high=upper))
        points.append(point)

    if points_file is not None:
        points_path = Path(points_file).resolve()
        supplied = np.atleast_1d(np.genfromtxt(points_path, names=True))
        if not supplied.dtype.names or not len(supplied):
            raise ValueError("The parameter table needs named columns and rows.")
        unknown = set(supplied.dtype.names) - set(base_point)
        if unknown:
            raise ValueError(f"Unknown or fixed parameter columns: {sorted(unknown)}")
        for row in supplied:
            point = dict(base_point)
            for name in supplied.dtype.names:
                point[name] = float(row[name])
            if not all(np.isfinite(value) for value in point.values()):
                raise ValueError("Supplied parameter points must be finite.")
            for name, (lo, hi) in test_bounds.items():
                if name in point and not lo <= point[name] <= hi:
                    raise ValueError(f"{name} in supplied table is outside the emulator-test bounds.")
            points.append(point)
        n_models = len(points)
        output = project / "chains/matter_emulator_points" / source.stem / points_path.stem
    else:
        output = project / "chains/matter_emulator_prior" / source.stem
    if fisher:
        output = output / "with_fisher"
    if output_dir is not None:
        output = Path(output_dir).resolve()
        if not output.is_relative_to(project / "chains"):
            raise ValueError("Comparison output must stay below project chains/.")
    metadata_error = None
    if rank == 0:
        try:
            output.mkdir(parents=True, exist_ok=True)
            if (output / "source.yaml").exists():
                if (output / "source.yaml").read_text() != source.read_text():
                    raise ValueError("Saved YAML differs; choose a new --output directory.")
            columns = sorted(points[0])
            table = []
            for point in points:
                table.append([point[name] for name in columns])
            if (output / "points.txt").exists():
                previous = np.loadtxt(output / "points.txt", ndmin=2)
                if previous.shape != np.shape(table) or not np.array_equal(previous, table):
                    raise ValueError("Saved points differ; choose a new output directory.")
            np.savetxt(
                fname=output / "points.txt",
                X=table,
                header=" ".join(columns),
            )
            (output / "source.yaml").write_text(source.read_text())
            with io.StringIO() as record:
                record.write(f"models={n_models}\nseed={seed}\nlimit={limit}\n")
                record.write(f"effective emulator test bounds={test_bounds}\n")
                if points_file is not None:
                    record.write(f"explicit parameter table={points_path}; seed unused\n")
                record.write("mnu=0.06 eV; plain mead2020; CAMB AccuracyBoost=3; k_per_logint=50\n")
                record.write("CAMB background in both; candidate replaces the requested power spectra\n")
                record.write(f"candidate_extra_args={candidate_extra_args}; native_selection={select_vector is not None}\n")
                record.write(f"use_emulator_switch={use_emulator_switch}; "
                             f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}\n")
                for directory in (project, Path(__file__).parent, code_dir, data_dir):
                    revision = subprocess.check_output(
                        ["git", "-C", str(directory), "rev-parse", "HEAD"],
                        text=True,
                    ).strip()
                    record.write(f"{directory}: {revision}\n")
                # A Git revision survives local edits. Hash the survey inputs,
                # including the measured vector (its score is checkpointed),
                # and the actual compiled module and networks selected here.
                files = set()
                options = info["likelihood"][likelihood_name]
                folder = root / options.get("path", project / "data")
                dataset = folder / options["data_file"]
                files.add(dataset)
                for line in dataset.read_text().splitlines():
                    entry = line.split("#", 1)[0]
                    if "=" in entry:
                        value = entry.split("=", 1)[1].strip()
                        if value and (folder / value).is_file():
                            files.add(folder / value)
                for value in options.values():
                    if isinstance(value, str) and value and (root / value).is_file():
                        files.add(root / value)
                # Cobaya expands likelihood defaults from the project's package.
                for pattern in ("*.yaml", "*.py"):
                    files.update((project / "likelihood").glob(pattern))
                native = sys.modules[f"cosmolike_{project.name}_interface"]
                files.add(Path(native.__file__))
                camb_package = root / info["theory"]["camb"]["path"] / "camb"
                files.add(camb_package / "camblib.so")
                files.update(camb_package.glob("*.py"))
                for theory in info["theory"]:
                    package = Path(__file__).parent / "theory" / theory
                    if package.is_dir():
                        files.update(package.glob("*.py"))
                        files.update(package.glob("*.yaml"))
                linear = root / "external_modules/data/emultrf/emulmps/npce_emul"
                nonlinear = data_dir / "emulmps/w0wa_mead2020"
                networks = {
                    "model_file": linear / "emulator_npce.keras",
                    "metadata_file": linear / "metadata.joblib",
                    "nl_model_file": nonlinear / "emulator_mead2020.keras",
                    "nl_metadata_file": nonlinear / "metadata.joblib",
                }
                for key, path in networks.items():
                    files.add(Path((candidate_extra_args or {}).get(key, path)))
                files.update((code_dir / "emulmps").rglob("*.py"))
                files.add(Path(__file__))
                for path in sorted(files):
                    record.write(f"{path.resolve()} {sha256_of(path)}\n")
                provenance = record.getvalue()
            previous_provenance = output / "provenance.txt"
            if not previous_provenance.exists() and any(output.glob("*_vectors.npy")):
                raise ValueError("Missing checkpoint provenance; use a fresh output directory.")
            if previous_provenance.exists() and previous_provenance.read_text() != provenance:
                raise ValueError("Saved model provenance differs; choose a new output directory.")
            previous_provenance.write_text(provenance)

        except Exception as error:
            metadata_error = f"{type(error).__name__}: {error}"
    if mpi_comm is not None:
        metadata_error = mpi_comm.bcast(metadata_error, root=0)
    if metadata_error is not None:
        raise ValueError(metadata_error)

    vectors = {}
    loglikes = {}
    reference_precision = None
    fisher_failed = False
    status = (output / "status.txt").open("a") if rank == 0 else io.StringIO()
    try:
        # Finish one model before initializing the next: the C library has
        # one global state. Holding two live likelihood instances would make
        # their setters and caches interfere with each other.
        for route in ("camb", "candidate"):
            settings = copy.deepcopy(info)
            if use_emulator_switch:
                settings["likelihood"][likelihood_name]["use_emulator"] = 0
            if route == "candidate":
                if use_emulator_switch:
                    settings["likelihood"][likelihood_name]["use_emulator"] = 2
                settings["params"]["As_1e9"]["drop"] = False
                settings["theory"]["emulmps"] = {
                    "external": emulmps,
                    "stop_at_error": True,
                    "provides": ["Pk_interpolator", "Pk_grid"],
                    "extra_args": {
                        "model_file": str(root / "external_modules/data/emultrf/emulmps/npce_emul/emulator_npce.keras"),
                        "metadata_file": str(root / "external_modules/data/emultrf/emulmps/npce_emul/metadata.joblib"),
                        "nl_model_file": str(data_dir / "emulmps/w0wa_mead2020/emulator_mead2020.keras"),
                        "nl_metadata_file": str(data_dir / "emulmps/w0wa_mead2020/metadata.joblib"),
                        "param_order": ["As_1e9", "ns", "H0", "omegab", "omegam", "w", "wa"],
                        "fixed_params": {"mnu": 0.06},
                    },
                }
                if candidate_extra_args is not None:
                    settings["theory"]["emulmps"]["extra_args"].update(candidate_extra_args)
                # These inputs are dropped by CAMB-only project examples,
                # but the matter emulator consumes them directly.
                for name in ("omegam", "omegab"):
                    if isinstance(settings["params"].get(name), dict):
                        settings["params"][name]["drop"] = False
            model_error = None
            try:
                model = get_model(info_or_yaml_or_file=settings)
            except Exception as error:
                model_error = f"{type(error).__name__}: {error}"
            errors = [model_error] if mpi_comm is None else mpi_comm.allgather(model_error)
            if any(error is not None for error in errors):
                raise RuntimeError(f"MPI model initialization failed: {errors}")
            likelihood = model.likelihood[likelihood_name]
            precision = np.asarray(get_precision())
            precision_error = None
            try:
                if reference_precision is None:
                    reference_precision = precision.copy()
                    if rank == 0:
                        if (output / "precision.npy").exists():
                            np.testing.assert_array_equal(np.load(output / "precision.npy"), precision)
                        else:
                            np.save(file=output / "precision.npy", arr=precision)
                else:
                    np.testing.assert_array_equal(precision, reference_precision)
            except Exception as error:
                precision_error = f"{type(error).__name__}: {error}"
            errors = [precision_error] if mpi_comm is None else mpi_comm.allgather(precision_error)
            if any(error is not None for error in errors):
                raise RuntimeError(f"Saved/native precision differs: {errors}")
            rows = np.full((n_models, len(precision)), np.nan)
            loglike_rows = np.full(n_models, np.nan)
            done = np.zeros(n_models, dtype=bool)
            resume_error = None
            if rank == 0:
                try:
                    vector_path = output / f"{route}_vectors.npy"
                    loglike_path = output / f"{route}_loglikes.npy"
                    done_path = output / f"{route}_done.npy"
                    if vector_path.exists() and loglike_path.exists():
                        saved_rows = np.load(vector_path)
                        saved_loglikes = np.load(loglike_path)
                        if saved_rows.shape != rows.shape or saved_loglikes.shape != loglike_rows.shape:
                            raise ValueError("Saved comparison checkpoint has another shape.")
                        rows, loglike_rows = saved_rows, saved_loglikes
                        done = (np.all(np.isfinite(rows), axis=1)
                                & np.isfinite(loglike_rows))
                        if done_path.exists():
                            saved_done = np.load(done_path)
                            if saved_done.shape != done.shape:
                                raise ValueError("Saved completion flags have another shape.")
                            done = saved_done.astype(bool)
                except Exception as error:
                    resume_error = f"{type(error).__name__}: {error}"
            if mpi_comm is not None:
                resume_error = mpi_comm.bcast(resume_error, root=0)
                rows, loglike_rows, done = mpi_comm.bcast((rows, loglike_rows, done), root=0)
            if resume_error is not None:
                raise ValueError(resume_error)
            def evaluate_point(job):
                index = job[0]
                point = points[index]
                vector = None
                loglike = np.nan
                try:
                    posterior = model.logposterior(params_values=point, cached=False)
                    if not np.isfinite(posterior.loglike):
                        raise ValueError("Nonfinite likelihood; no current prediction accepted.")
                    if hasattr(likelihood, "last_datavector"):
                        # The uncached posterior call just evaluated this
                        # vector, including any Python stochastic templates.
                        # Retain it without repeating expensive native setters.
                        vector = np.array(likelihood.last_datavector, copy=True)
                    else:
                        # to_input(dict) changes Cobaya's internal sampled
                        # ordering. Its canonical check prevents assigning
                        # values to the wrong names on the next evaluation.
                        inputs = model.parameterization.to_input(
                            sampled_params_values=model.parameterization.check_sampled(point)
                        )
                        vector = np.asarray(likelihood.get_datavector(**inputs))
                    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
                        raise ValueError("Prediction shape or finite-value check failed.")
                    # Re-score the saved vector through the same likelihood.
                    # A disagreement means the repeated getter changed state;
                    # such a row is invalid, not an emulator accuracy result.
                    repeated_loglike = likelihood.compute_logp(datavector=vector)
                    if not np.isclose(
                        a=repeated_loglike,
                        b=posterior.loglike,
                        rtol=1.e-12,
                        atol=1.e-8,
                    ):
                        raise ValueError("Saved prediction does not reproduce its likelihood.")
                    if select_vector is not None:
                        vector = np.asarray(select_vector(vector))
                    if vector.shape != (len(precision),) or not np.all(np.isfinite(vector)):
                        raise ValueError("Selected prediction does not match its precision.")
                    loglike = posterior.loglike
                    message = f"{route} cosmology {index + 1}/{n_models}: evaluated"
                except Exception as error:
                    vector = None
                    loglike = np.nan
                    # Keep the failed row as NaN. Never reuse a previous
                    # vector, redraw a point, or count a rejection as a pass.
                    message = f"{route} cosmology {index + 1}/{n_models}: FAILED {type(error).__name__}: {error}"
                return index, vector, loglike, message

            # Use the same shared MPI job owner as the established DESI
            # Fisher driver. Small rounds let rank zero save checkpoints
            # between rounds, without another scheduling implementation.
            for first in range(0, n_models, size):
                jobs = [(index,) for index in range(first, min(first + size, n_models))
                        if not done[index]]
                if not jobs:
                    continue
                if mpi_comm is None:
                    gathered = [evaluate_point(job) for job in jobs]
                else:
                    results, _ = run_owned_jobs(mpi_comm, jobs, evaluate_point)
                    gathered = list(results.values()) if rank == 0 else None
                checkpoint_error = None
                if rank == 0:
                    try:
                        for item in gathered:
                            if item is None:
                                continue
                            index, vector, loglike, message = item
                            if vector is not None:
                                rows[index] = vector
                                loglike_rows[index] = loglike
                            done[index] = True
                            print(message, flush=True)
                            status.write(message + "\n")
                        status.flush()
                        np.save(output / f"{route}_vectors.npy", rows)
                        np.save(output / f"{route}_loglikes.npy", loglike_rows)
                        np.save(output / f"{route}_done.npy", done)
                    except Exception as error:
                        checkpoint_error = f"{type(error).__name__}: {error}"
                if mpi_comm is not None:
                    checkpoint_error = mpi_comm.bcast(checkpoint_error, root=0)
                if checkpoint_error is not None:
                    raise RuntimeError(checkpoint_error)
            if mpi_comm is not None:
                rows, loglike_rows = mpi_comm.bcast((rows, loglike_rows), root=0)
            vectors[route], loglikes[route] = rows, loglike_rows
            if route == "camb" and fisher:
                # Close the matched-mean reference before changing native
                # accuracy. CosmoLike has one global model state per rank.
                model.close()
                model = None
                try:
                    normal = copy.deepcopy(fisher_settings)
                    if use_emulator_switch:
                        normal["likelihood"][likelihood_name]["use_emulator"] = 0
                    # A parameter-dependent early exit creates jumps in a
                    # derivative. Disable it for both Fisher accuracies;
                    # the separate mean comparisons keep the YAML values.
                    for key in ("nonlimber_tolerance_gg", "nonlimber_tolerance_gs"):
                        if hasattr(likelihood, key):
                            normal["likelihood"][likelihood_name][key] = 0.0
                    refined = None
                    for stage in ("default_accuracy", "refined_accuracy"):
                        current = normal if refined is None else refined
                        error = None
                        try:
                            model = get_model(info_or_yaml_or_file=current)
                            likelihood = model.likelihood[likelihood_name]
                            np.testing.assert_array_equal(get_precision(), reference_precision)
                        except Exception as caught:
                            error = f"{type(caught).__name__}: {caught}"
                        errors = [error] if mpi_comm is None else mpi_comm.allgather(error)
                        if any(item is not None for item in errors):
                            raise ValueError(f"Fisher model setup failed: {errors}")
                        if stage == "default_accuracy":
                            # One explicit next setting, recorded below. This
                            # is a convergence trial, not a claimed validated
                            # production setting. Per-project overrides can
                            # supply an already documented numerical ladder.
                            refined = copy.deepcopy(normal)
                            options = refined["likelihood"][likelihood_name]
                            boost = float(likelihood.accuracyboost)
                            options["accuracyboost"] = max(boost, min(3., 2.*boost))
                            options["integration_accuracy"] = abs(int(likelihood.integration_accuracy))+1
                            if hasattr(likelihood, "threepcf_accuracy"):
                                controls = dict(likelihood.threepcf_accuracy or {})
                                controls["accuracy_boost"] = 2.*float(controls.get("accuracy_boost", 1.))
                                controls["integration_accuracy_boost"] = 2.*float(controls.get("integration_accuracy_boost", 1.))
                                options["threepcf_accuracy"] = controls
                            options.update(fisher_accuracy_likelihood or {})
                            fine_camb = refined["theory"]["camb"]["extra_args"]
                            fine_camb["AccuracyBoost"] = max(2., float(fine_camb.get("AccuracyBoost", 1.)))
                            fine_camb["k_per_logint"] = max(50, int(fine_camb.get("k_per_logint", 0)))
                            fine_camb.update(fisher_accuracy_camb or {})
                        fisher_rows, fisher_covariance = None, None
                        if fisher_probe is not None:
                            if fisher_probe != "xi" or get_covariance is None:
                                raise ValueError("Shear Fisher requires probe xi and a covariance reader.")
                            nsource = likelihood.source_ntomo
                            nshear = nsource*(nsource+1)*likelihood.ntheta
                            fisher_rows = np.flatnonzero((np.diag(reference_precision)>0.)
                                                         & (np.arange(len(reference_precision))<nshear))
                            fisher_covariance = np.asarray(get_covariance())
                        destination = output/"fisher"/"default_accuracy" if stage == "default_accuracy" else output/"fisher"
                        compute_matter_emulator_fisher(
                            model=model, likelihood=likelihood,
                            center=base_point if fisher_center is None else fisher_center,
                            precision=reference_precision, params=model.info()["params"],
                            output=destination, sample_power=fisher_sample_power,
                            parameters=fisher_parameters, steps=fisher_steps,
                            rows=fisher_rows, covariance=fisher_covariance,
                            select_vector=select_vector, derivative_provider=derivative_provider,
                            amplitude_bias_cuts=amplitude_bias_cuts, mpi_comm=mpi_comm,
                            sample=stage == "refined_accuracy", noise_screen=True,
                            noise_reference=None if stage == "default_accuracy" else output/"fisher"/"default_accuracy",
                        )
                        if rank == 0:
                            (destination/"numerical_settings.txt").write_text(
                                f"stage={stage}\n"
                                f"likelihood={current['likelihood'][likelihood_name]}\n"
                                f"CAMB={current['theory']['camb']['extra_args']}\n"
                                f"FoM relative tolerance={fisher_fom_tolerance}\n"
                                "Fisher uses the example's physical theory with mnu=0.06 eV.\n"
                                + (destination/"fiducial_likelihood.txt").read_text()
                            )
                        model.close()
                        model = None
                    gate_error = None
                    if rank == 0:
                        try:
                            compare_matter_emulator_fisher_accuracy(
                                base=output/"fisher"/"default_accuracy", refined=output/"fisher",
                                tolerance=fisher_fom_tolerance,
                            )
                        except Exception as caught:
                            gate_error = f"{type(caught).__name__}: {caught}"
                    if mpi_comm is not None:
                        gate_error = mpi_comm.bcast(gate_error, root=0)
                    if gate_error is not None:
                        raise ValueError(gate_error)
                except Exception as error:
                    fisher_failed = True
                    message = f"Fisher FAILED {type(error).__name__}: {error}"
                    if rank == 0:
                        print(message, flush=True)
                        status.write(message+"\n")
                        status.flush()
            if model is not None:
                model.close()
    finally:
        status.close()

    if rank == 0:
        residuals = vectors["candidate"] - vectors["camb"]
        distances = np.einsum("ij,jk,ik->i", residuals, reference_precision, residuals)
        finite = np.isfinite(distances)
        passed = finite & (distances >= 0.0) & (distances < limit)
        log_weights = loglikes["camb"] - loglikes["candidate"]
        result_table = np.column_stack((
            np.arange(n_models), distances, passed.astype(int), log_weights,
        ))
        np.savetxt(
            fname=output / "distances.txt",
            X=result_table,
            header="model_index delta_chi2 passes_strict_limit log_L_CAMB_minus_log_L_emulator",
        )
        # Prior draws include points that fit the data very poorly. Compare their
        # CAMB likelihoods with the best saved point before interpreting an error
        # as relevant to inference. Nuisance values are prescribed (constant for
        # prior draws, supplied per row for explicit inputs); they are not fitted.
        # This is neither a profile likelihood nor a marginalized exclusion test.
        fit_penalty = np.full(n_models, np.nan)
        finite_reference = np.isfinite(loglikes["camb"])
        if finite_reference.any():
            best_loglike = np.max(loglikes["camb"][finite_reference])
            fit_penalty[finite_reference] = -2.0 * (
                loglikes["camb"][finite_reference] - best_loglike
            )
        np.savetxt(
            fname=output / "likelihood_relevance.txt",
            X=np.column_stack((
                np.arange(n_models), loglikes["camb"], fit_penalty, distances,
            )),
            header=("index logL_CAMB "
                    + ("fixed_nuisance_fit_penalty" if points_file is None
                       else "prescribed_nuisance_fit_penalty")
                    + " emulator_vector_error"),
        )
        summary = f"{int(passed.sum())}/{n_models} pass delta_chi2 < {limit}; {int((~finite).sum())} nonfinite\n"
        if finite.any():
            summary += f"median={np.median(distances[finite]):.9g}; maximum={np.max(distances[finite]):.9g}\n"
        for threshold in (0.2, 0.5, 1.0, 4.0):
            count = np.count_nonzero(finite & (distances >= 0.0) & (distances < threshold))
            summary += f"delta_chi2 < {threshold}: {count}/{n_models}\n"
        summary += f"delta_chi2 >= 4: {np.count_nonzero(finite & (distances >= 4.0))}/{n_models}\n"
        (output / "summary.txt").write_text(summary)
    focused_failed = 0
    local_input = None
    setup_error = None
    if rank == 0 and fisher and not fisher_failed:
        try:
            if points_file is None and focused_models > 0:
                local_input = local_points_file
                if local_input is None:
                    local_input = draw_matter_emulator_focused_points(
                        output=output / "fisher", n_models=focused_models,
                        seed=seed + 10000,
                    )
                supplied = np.atleast_1d(np.genfromtxt(local_input, names=True))
                if len(supplied) != focused_models:
                    raise ValueError("Focused table row count differs from focused_models.")
                for name in supplied.dtype.names:
                    if name not in names and name in base_point:
                        if not np.all(supplied[name] == base_point[name]):
                            raise ValueError(f"Focused precision points vary nuisance {name}; "
                                             "keep its evaluate YAML value fixed.")
        except Exception as error:
            setup_error = f"{type(error).__name__}: {error}"
    if mpi_comm is not None:
        local_input, setup_error = mpi_comm.bcast((local_input, setup_error), root=0)
    if setup_error is None and local_input is not None:
        focused_failed = run_matter_emulator_prior_check(
            project=project, configuration=configuration,
            get_precision=get_precision, points_file=local_input,
            output_dir=output / "local", limit=limit, advisory=advisory,
            candidate_extra_args=candidate_extra_args,
            use_emulator_switch=use_emulator_switch, select_vector=select_vector,
            focused_models=0, mpi_comm=mpi_comm,
        )
    result = None
    if rank == 0:
        if setup_error is not None:
            fisher_failed = True
            summary += f"Focused setup FAILED {setup_error}\n"
        if fisher and not fisher_failed:
            try:
                plot_matter_emulator_fisher(
                    output=output / "fisher", points_file=output / "points.txt",
                    distances_file=output / "distances.txt", project_name=project.name,
                    local_points_file=output / "local/points.txt" if local_input is not None else None,
                    local_distances_file=output / "local/distances.txt" if local_input is not None else None,
                )
                if local_input is not None:
                    summary += f"Focused comparison: {focused_models} points; see local/summary.txt.\n"
            except Exception as error:
                fisher_failed = True
                summary += f"Fisher plot FAILED {type(error).__name__}: {error}\n"
        if fisher_failed:
            summary += "Fisher overlay incomplete; inspect status and fisher diagnostics.\n"
        calculation_failed = (fisher_failed or focused_failed or not finite_reference.all()
                              or not np.any(finite & (distances >= 0.0)))
        if not finite_reference.all():
            summary += "CAMB reference evaluations incomplete; comparison cannot pass.\n"
        if not np.any(finite & (distances >= 0.0)):
            summary += "No finite nonnegative prediction distances were obtained.\n"
        (output / "summary.txt").write_text(summary)
        print(summary, flush=True)
        result = int(calculation_failed) if advisory else int((~passed).sum()) + int(fisher_failed) + focused_failed
    return result if mpi_comm is None else mpi_comm.bcast(result, root=0)


def draw_matter_emulator_focused_points(output, n_models=25, seed=10137):
    """Save parameter draws from a validated bounded CAMB Fisher calculation.

    Sample the full weighted Fisher distribution, then retain cosmological
    coordinates only. This integrates over nuisance uncertainty in selecting
    the cosmological region; it does not condition the Fisher on fixed
    nuisance parameters. The subsequent precision comparison uses the YAML
    nuisance values in both theories. The draws are deterministic for a given
    archive and seed, and use no emulator errors or observed-data fit scores.
    These are local Fisher draws, not samples from an inferred posterior.

    Arguments:
      output = Fisher directory containing bounded_samples.npz.
      n_models = number of additional parameter points.
      seed = random seed for weighted sampling without replacement.
    Returns:
      path of focused_points.txt; the file contains named parameter columns.
    """
    from pathlib import Path
    import numpy as np

    output = Path(output)
    if not isinstance(n_models, int) or n_models < 1:
        raise ValueError("The focused sample must contain a positive integer number of points.")
    with np.load(file=output / "bounded_samples.npz") as record:
        if not bool(record["sampling_valid"]):
            raise ValueError("Focused draws require converged bounded Fisher sampling.")
        names = record["names"].tolist()
        values = record["values"].copy()
        weights = record["weights"].copy()
        lower = record["lower"]
        upper = record["upper"]
        valid = np.all(np.isfinite(values), axis=1)
        valid &= np.all((values >= lower) & (values <= upper), axis=1)
        valid &= np.isfinite(weights) & (weights > 0.0)
        if "omegam" in names:
            omega_m = values[:, names.index("omegam")]
            valid &= (omega_m > 0.2) & (omega_m < 0.5)
        if "As_1e9" in names:
            valid &= values[:, names.index("As_1e9")] <= 3.5
    if np.count_nonzero(valid) < n_models:
        raise ValueError("Too few bounded Fisher samples for the requested focused points.")
    values = values[valid]
    weights = weights[valid]
    weights /= np.sum(weights)
    random = np.random.default_rng(seed=seed)
    selected = random.choice(a=len(values), size=n_models, replace=False, p=weights)
    cosmological = ("As_1e9", "ns", "H0", "omegab", "omegam", "w", "w0pwa", "wa")
    columns = []
    selected_names = []
    for index, name in enumerate(names):
        if name in cosmological:
            columns.append(index)
            selected_names.append(name)
    if not columns:
        raise ValueError("The Fisher archive contains no supported cosmological parameters.")
    destination = output / "focused_points.txt"
    np.savetxt(fname=destination, X=values[np.ix_(selected, columns)],
               header=" ".join(selected_names))
    (output / "focused_definition.txt").write_text(
        f"Weighted bounded Fisher draws; count={n_models}; seed={seed}.\n"
        "Cosmological marginal of the full nuisance-marginalized Fisher.\n"
        "0.2 < Omega_m < 0.5; 1e9 As <= 3.5; no selection on emulator error.\n"
        "All nuisance inputs keep the evaluate YAML overrides in precision evaluations.\n"
    )
    return destination


def compute_matter_emulator_fisher(
    model, likelihood, center, precision, params, output, steps=None,
    rows=None, covariance=None, sample_power=18, seeds=(137, 138),
    min_effective=200, parameters=None, select_vector=None,
    derivative_provider=None, amplitude_bias_cuts=None, mpi_comm=None,
    sample=True, noise_reference=None, noise_fraction=0.95, noise_screen=False,
):
    """Compute the reference model's local Fisher information and contours.

    Arguments:
      model, likelihood = initialized CAMB Cobaya model and its likelihood.
      center = sampled parameter dictionary; fixed inputs come from Cobaya.
      precision = native precision, including zero masked rows.
      params = parameter specifications of this same configuration.
      output = directory for derivatives, priors and bounded samples.
      steps = optional named absolute stencil steps (not relative steps).
      parameters = optional differentiated names. The complete sampled
        center still reaches Cobaya, including undifferentiated nuisances.
        Without this override, ordinary NLA excludes its exact TATT-only
        A2_1, A2_2 and BTA_1 coordinates: their independent priors factor out.
      select_vector = optional native mask/compression before derivatives.
      derivative_provider = optional callable(model, likelihood, center, steps)
        returning {name: (full derivative at h, full derivative at h/2)}.
        Supplied columns replace expensive finite differences for those names;
        the remaining parameters retain the common stencil and all priors.
      amplitude_bias_cuts = optional {external prior name:
        (amplitude name, bias name, reference amplitude, reference bias, limit)}.
        Each explicitly declared hard prior is (amplitude/A0)*(bias/b0)^2
        <= limit. Its interior curvature is zero; the coupled boundary is
        retained in every bounded integral. Other external priors are refused.
      mpi_comm = optional communicator passed to the existing shared
        mpi_batching.run_owned_jobs. A complete h/h2 derivative column is
        one job; each rank owns a model, and rank zero assembles the result.
      sample = integrate bounded contours after saving the full Fisher.
      noise_reference = optional default-accuracy Fisher directory. Its
        measured noise contributions choose which refined columns run first.
        The rest still run once for the final full-matrix convergence check.
      noise_fraction = fraction of baseline noise contribution prioritized;
        omitted columns are subsequently checked, never assumed noise-free.
      noise_screen = save the shared fourth-difference/FoM noise diagnostics;
        requires a resolved full Fisher matrix and the target cosmology pair.
      rows = optional unmasked data subset; requires the full covariance.
      covariance = full covariance, only used when selecting data rows.
      sample_power, seeds, min_effective = bounded integration controls.
    Returns:
      sample_matter_emulator_fisher's dictionary of weighted samples.
    Side effects:
      Evaluates the live reference model and restores its fiducial state.
      Saves Fisher information before sampling, permitting inexpensive retries.

    This is a centered forecast approximation. It does not fit the observed
    data score or move the fiducial to the nuisance-prior means. Consequently
    its bounded contours are not measured posterior credible regions.
    """
    from pathlib import Path
    import numpy as np
    from cosmolike_notebook_utils import fisher as cf

    from cosmolike_notebook_utils.mpi_batching import run_owned_jobs

    rank = 0 if mpi_comm is None else mpi_comm.Get_rank()
    output = Path(output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    cuts = {} if amplitude_bias_cuts is None else dict(amplitude_bias_cuts)
    external = getattr(model.prior, "external", {})
    if set(external) != set(cuts):
        raise ValueError("External priors require explicit matching amplitude-bias hard cuts; "
                         "other external/correlated prior curvature is unsupported.")
    sampled = list(model.parameterization.sampled_params())
    names = sampled if parameters is None else list(parameters)
    inactive_nla = []
    if parameters is None and getattr(likelihood, "IA_model", None) == 0:
        # These are the ordinary likelihoods' amplitude, redshift slope
        # and density-weighting names. IA.c and the IA_MODEL_TATT branches
        # in cosmo2D.c exclude their terms throughout NLA, not just at the
        # fiducial. Their independent prior integrals are constants. Keep
        # the full center supplied to Cobaya, but do not differentiate an
        # unused BTA at its common zero-valued hard boundary. TATT and an
        # explicit caller parameter list retain their existing behavior.
        inactive_nla = [
            name for name in sampled
            if name.endswith(("_A2_1", "_A2_2", "_BTA_1"))
        ]
        names = [name for name in sampled if name not in inactive_nla]
    if not names or len(set(names)) != len(names) or set(names) - set(sampled):
        raise ValueError("Differentiated parameters must be unique sampled names.")
    center = {name: float(center[name]) for name in sampled}
    lower, upper, widths, absolute_steps = [], [], {}, {}
    steps = {} if steps is None else dict(steps)
    if set(steps) - set(names):
        raise ValueError("Fisher step overrides must name sampled parameters.")
    for name in names:
        specification = params[name]
        prior = specification["prior"]
        if not isinstance(prior, dict):
            raise ValueError(f"Prior for {name} must be an explicit mapping.")
        distribution = prior.get("dist", "uniform")
        if distribution == "uniform":
            lo, hi = float(prior["min"]), float(prior["max"])
            scale = (hi - lo) / 10.0
        elif distribution in ("norm", "truncnorm"):
            scale = float(prior.get("scale", 1.0))
            widths[name] = scale
            if distribution == "truncnorm":
                location = float(prior.get("loc", 0.0))
                lo = location + float(prior["a"]) * scale
                hi = location + float(prior["b"]) * scale
            else:
                lo, hi = -np.inf, np.inf
        else:
            raise ValueError(f"Fisher prior {distribution} for {name} is unsupported.")
        if not lo < center[name] < hi or not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Fisher center for {name} must lie inside a proper prior.")
        lower.append(lo)
        upper.append(hi)
        # Proposal widths choose numerical steps, never prior curvature.
        # Comparing h and h/2 below tests this initial numerical choice.
        proposal = specification.get("proposal", scale)
        proposal = scale if proposal is None else float(proposal)
        step = float(steps.get(name, 0.1 * proposal))
        if name not in steps:
            step = min(step, 0.2*(center[name]-lo), 0.2*(hi-center[name]))
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError(f"Fisher step for {name} must be finite and positive.")
        absolute_steps[name] = step

    cut_parameters, cut_scales = [], []
    for label, specification in cuts.items():
        if len(specification) != 5:
            raise ValueError(f"Hard cut {label} needs two names and three scales.")
        amplitude, bias, amplitude0, bias0, limit = specification
        if amplitude not in names or bias not in names:
            raise ValueError(f"Hard cut {label} must use differentiated parameters.")
        scales = np.array([amplitude0, bias0, limit], dtype=float)
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.):
            raise ValueError(f"Hard cut {label} needs finite positive scales.")
        if (center[amplitude]/amplitude0)*(center[bias]/bias0)**2 >= limit:
            raise ValueError(f"Fisher center must be strictly inside hard cut {label}.")
        cut_parameters.append([amplitude, bias])
        cut_scales.append(scales)

    precision = np.asarray(precision, dtype=float)
    if precision.ndim != 2 or precision.shape[0] != precision.shape[1]:
        raise ValueError("Native precision must be square.")
    cf.symmetric_matrix(precision, name="Native precision")
    if np.any(np.diag(precision) < 0.0):
        raise ValueError("Native precision has negative diagonal entries.")
    active = np.flatnonzero(np.diag(precision) > 0.0)
    if rows is None:
        selected = active
        metric = precision[np.ix_(selected, selected)]
        arguments = {"precision": metric}
    else:
        selected = np.asarray(rows)
        if (selected.ndim != 1 or selected.dtype.kind not in "iu"
                or len(np.unique(selected)) != len(selected)
                or not np.all(np.isin(selected, active))):
            raise ValueError("Fisher rows must be unique unmasked integer indices.")
        if covariance is None or np.shape(covariance) != precision.shape:
            raise ValueError("A data subset requires the full matching covariance.")
        # Selecting the full precision would condition on omitted data.
        # A marginal observable subset instead uses this covariance block.
        metric = np.asarray(covariance)[np.ix_(selected, selected)]
        arguments = {"covariance": metric}
    cf.symmetric_matrix(metric, name="Fisher data metric", positive=True)

    def predict(point):
        # Only theory may reuse identical inputs. The likelihood owns native
        # state and last_datavector, so it must execute on every cache miss.
        manage_cache = hasattr(likelihood, "set_cache_size") and hasattr(likelihood, "_states")
        if manage_cache:
            previous_size = likelihood._states.maxlen
            likelihood.set_cache_size(0)
        try:
            result = model.logposterior(params_values=point, cached=manage_cache)
        finally:
            if manage_cache:
                likelihood.set_cache_size(previous_size)
        if not np.isfinite(result.loglike):
            raise ValueError("Fisher reference likelihood is nonfinite.")
        if hasattr(likelihood, "last_datavector"):
            vector = np.array(likelihood.last_datavector, dtype=float, copy=True)
        else:
            inputs = model.parameterization.to_input(
                sampled_params_values=model.parameterization.check_sampled(point)
            )
            vector = np.array(likelihood.get_datavector(**inputs), dtype=float, copy=True)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            raise ValueError("Fisher prediction has wrong shape or nonfinite entries.")
        if not np.isclose(likelihood.compute_logp(datavector=vector), result.loglike,
                          rtol=1.e-12, atol=1.e-8):
            raise ValueError("Fisher saved prediction does not reproduce its likelihood.")
        if select_vector is not None:
            vector = np.asarray(select_vector(vector))
        if vector.shape != (len(precision),) or not np.all(np.isfinite(vector)):
            raise ValueError("Selected Fisher prediction does not match its precision.")
        return vector[selected]

    # The existing cache reuses identical shared stencil points only within
    # this fixed model. It never treats nearby cosmologies as the same point.
    reference = predict(center)
    if hasattr(likelihood, "last_datavector"):
        center_loglike = float(likelihood.compute_logp(datavector=likelihood.last_datavector))
    else:
        center_loglike = float(model.logposterior(params_values=center, cached=False).loglike)
    if rank == 0:
        (output / "fiducial_likelihood.txt").write_text(
            f"CAMB Fisher reference: logL={center_loglike:.17g}; -2logL={-2*center_loglike:.17g}\n"
            "This is the declared Fisher fiducial and theory; consult numerical_settings.txt for the runner's choices.\n"
            "It need not have zero residual against a synthetic vector made with another prescription.\n"
        )
    # Subtract one fixed vector before differencing. An inactive parameter
    # then gives exactly zero at every stencil point, rather than a tiny
    # artificial slope from adding rounded multiples of the same vector.
    cache = cf.ExactMeanCache(lambda point: predict(point) - reference)
    bounds = dict(zip(names, zip(lower, upper)))
    try:
        supplied = {} if derivative_provider is None else derivative_provider(
            model=model, likelihood=likelihood, center=dict(center),
            steps=dict(absolute_steps),
        )
        if set(supplied) - set(names):
            raise ValueError("Supplied derivative columns must name differentiated parameters.")
        columns = {}
        stencil_columns = {}
        for name, pair in supplied.items():
            if len(pair) != 2:
                raise ValueError(f"Supplied derivative {name} needs both h and h/2 columns.")
            selected_pair = []
            for vector in pair:
                vector = np.asarray(vector, dtype=float)
                if select_vector is not None:
                    vector = np.asarray(select_vector(vector))
                if vector.shape != (len(precision),) or not np.all(np.isfinite(vector)):
                    raise ValueError(f"Supplied derivative {name} does not match the data metric.")
                selected_pair.append(vector[selected])
            columns[name] = selected_pair
        remaining = [name for name in names if name not in columns]

        def evaluate_column(name):
            """Evaluate both derivative steps together on the owning rank.

            Return an error as data so every rank reaches the shared gather;
            the caller then refuses the entire Fisher matrix on all ranks.
            """
            try:
                _, _, coarse_column, fine_column = cf.differentiate(
                    predict=cache,
                    parameters=center,
                    steps={name: absolute_steps[name]},
                    bounds=bounds,
                )
                # These points are already in ExactMeanCache. Reading them
                # retains the fourth difference without another model call.
                stencils = []
                for divisor in (1., 2.):
                    values = []
                    for multiple in (-2., -1., 0., 1., 2.):
                        point = dict(center)
                        if multiple != 0.:
                            point[name] += multiple * (absolute_steps[name]/divisor)
                        values.append(cache(point))
                    stencils.append(np.asarray(values))
                return None, coarse_column[:, 0], fine_column[:, 0], stencils
            except Exception as error:
                return f"{name}: {type(error).__name__}: {error}", None, None, None

        # One default full base supplies both marginalization and ranking.
        # At refined accuracy evaluate its important columns first; finish
        # the rest only once for the independent full-Fisher gate below.
        baseline = None
        selected_names = []
        if noise_reference is not None:
            baseline = np.load(Path(noise_reference)/"fisher.npz")
            baseline_noise = matter_emulator_fisher_noise(baseline)
            if not 0. < noise_fraction <= 1.:
                raise ValueError("noise_fraction must lie in (0,1].")
            importance = np.maximum(baseline_noise["h"]["terms"],
                                    baseline_noise["half_h"]["terms"])
            available = [names.index(name) for name in remaining]
            order = sorted(available, key=lambda index: importance[index], reverse=True)
            total = sum(importance[index] for index in order)
            accumulated = 0.
            for index in order:
                selected_names.append(names[index])
                accumulated += importance[index]
                if total > 0. and accumulated >= noise_fraction*total:
                    break
                if total == 0. and len(selected_names) >= min(2, len(order)):
                    break
            # Always inspect the target coordinates, even when a baseline
            # fourth difference happens to cancel exactly for either one.
            for name in baseline_noise["pair"]:
                if name in remaining and name not in selected_names:
                    selected_names.append(name)
        groups = [remaining] if baseline is None else [
            selected_names, [name for name in remaining if name not in selected_names],
        ]
        for group_index, group in enumerate(groups):
            results = {}
            if group:
                if mpi_comm is None:
                    for name in group:
                        results[name] = evaluate_column(name)
                else:
                    results, _ = run_owned_jobs(comm=mpi_comm, jobs=group, evaluate=evaluate_column)
            derivative_error = None
            if rank == 0:
                try:
                    for name in group:
                        error, coarse_column, fine_column, stencils = results[name]
                        if error is not None:
                            raise ValueError(error)
                        columns[name] = [coarse_column, fine_column]
                        stencil_columns[name] = stencils
                    if baseline is not None and group_index == 0:
                        partial = dict(baseline)
                        partial.update(
                            stencil_names=np.asarray(selected_names),
                            stencils_h=np.asarray([stencil_columns[n][0] for n in selected_names]),
                            stencils_half_h=np.asarray([stencil_columns[n][1] for n in selected_names]),
                            analytic_names=np.asarray(list(supplied)),
                            steps=np.asarray([absolute_steps[n] for n in names]),
                        )
                        screen = matter_emulator_fisher_noise(partial, reference=baseline, complete=False)
                        # Compare the SAME columns at both accuracies. The
                        # default setting also contains numerical noise;
                        # its own inflation must divide the refined one.
                        baseline_partial = dict(partial)
                        positions = [baseline["stencil_names"].tolist().index(n)
                                     for n in selected_names]
                        for key in ("stencils_h", "stencils_half_h"):
                            baseline_partial[key] = baseline[key][positions]
                        default_screen = matter_emulator_fisher_noise(
                            baseline_partial, reference=baseline, complete=False,
                        )
                        ratios = []
                        for suffix in ("h", "half_h"):
                            ratios.append(
                                f"{suffix}: predicted refined/default FoM ratio="
                                f"{screen[suffix]['ratio']/default_screen[suffix]['ratio']:.17g}; "
                                f"first-order delta log FoM="
                                f"{screen[suffix]['dlnFoM']-default_screen[suffix]['dlnFoM']:.17g}\n"
                            )
                            score = screen[suffix]
                            np.savetxt(output/f"selected_noise_{suffix}.txt",
                                       np.c_[np.arange(len(names)), score["sigma"], score["a"], score["terms"]],
                                       header="index sigma_evaluation false_information twice_dlnFoM; names="+",".join(names))
                        omitted = screen["unmeasured"]
                        (output/"selected_noise_definition.txt").write_text(
                            "Default full Fisher weights; refined selected stencils.\n"
                            "Measured: "+", ".join(selected_names)+"\n"
                            "Not yet measured: "+", ".join(omitted)+"\n"
                            "Zeros for unmeasured columns omit unknown contributions.\n"
                            + "".join(ratios)
                            + "These ratios use the same default full Fisher and selected columns.\n"
                            "Compare with the measured full ratios in fom_convergence.txt; omitted noise is not bounded.\n"
                            "The remaining columns run next for the full-Fisher gate.\n"
                        )
                except Exception as error:
                    derivative_error = f"{type(error).__name__}: {error}"
            if mpi_comm is not None:
                derivative_error = mpi_comm.bcast(derivative_error, root=0)
            if derivative_error is not None:
                raise ValueError(derivative_error)
    finally:
        predict(center)

    # Only the writer assembles and samples the Fisher matrix. Other ranks
    # receive its success or failure before continuing the comparison.
    result = None
    assembly_error = None
    if rank == 0:
        try:
            coarse = np.column_stack([columns[name][0] for name in names])
            fine = np.column_stack([columns[name][1] for name in names])
            first = cf.fisher_matrix(derivatives=coarse, names=names, priors=widths, **arguments)
            second = cf.fisher_matrix(derivatives=fine, names=names, priors=widths, **arguments)
            zero = np.all(coarse == 0.0, axis=0) & np.all(fine == 0.0, axis=0)
            np.savez_compressed(
                file=output / "fisher.npz", names=names,
                center=[center[name] for name in names], lower=lower, upper=upper,
                gaussian_width=[widths.get(name, np.nan) for name in names],
                steps=[absolute_steps[name] for name in names], rows=selected,
                reference=reference, metric=metric, metric_is_covariance=rows is not None,
                derivatives_coarse=coarse, derivatives_fine=fine,
                stencil_names=remaining,
                stencils_h=np.asarray([stencil_columns[n][0] for n in remaining]),
                stencils_half_h=np.asarray([stencil_columns[n][1] for n in remaining]),
                analytic_names=list(supplied),
                fisher_coarse=first, fisher_fine=second, zero_response=zero,
                amplitude_bias_cut_names=list(cuts),
                amplitude_bias_cut_parameters=np.asarray(cut_parameters, dtype=str).reshape(-1, 2),
                amplitude_bias_cut_scales=np.asarray(cut_scales, dtype=float).reshape(-1, 3),
            )
            (output / "definition.txt").write_text(
                "Centered local CAMB Fisher diagnostic, not a fitted posterior.\n"
                "Differentiated parameters: " + ", ".join(names) + "\n"
                "Fixed in this Fisher calculation: " + ", ".join(n for n in sampled if n not in names) + "\n"
                "Exact NLA-independent prior factors omitted: " + ", ".join(inactive_nla) + "\n"
                "Independent Gaussian prior curvature and all selected hard bounds "
                "retained. Data/prior linear scores not fitted.\n"
                "Explicit amplitude-bias hard cuts retained: " + ", ".join(cuts) + "\n"
                "Supplied derivative columns: " + ", ".join(supplied) + "\n"
                "Fixed data covariance; no covariance derivatives.\n"
                "Exactly zero local response: " + ", ".join(np.asarray(names)[zero]) + "\n"
                "A zero response can hide nonlinear dependence, such as an IA slope "
                "at zero amplitude. These parameters remain in the bounded integral.\n"
            )
            if noise_screen or noise_reference is not None:
                record = np.load(output/"fisher.npz")
                baseline = None if noise_reference is None else np.load(Path(noise_reference)/"fisher.npz")
                screen = matter_emulator_fisher_noise(record, reference=baseline)
                for suffix in ("h", "half_h"):
                    score = screen[suffix]
                    np.savetxt(output/f"noise_{suffix}.txt",
                               np.c_[np.arange(len(names)), score["sigma"], score["a"],
                                     score["R2"], score["sigma2"], score["terms"]],
                               header="index sigma_evaluation false_information R2 parameter_variance twice_dlnFoM; names="+",".join(names))
                (output/"noise_definition.txt").write_text(
                    f"pair={screen['pair']}; full default Fisher with nuisance priors.\n"
                    f"h: dlnFoM={screen['h']['dlnFoM']:.17g}; ratio={screen['h']['ratio']:.17g}\n"
                    f"h/2: dlnFoM={screen['half_h']['dlnFoM']:.17g}; ratio={screen['half_h']['ratio']:.17g}\n"
                    "Analytic columns: "+", ".join(screen["analytic"])+"\n"
                    "Smooth curvature and correlated errors can invalidate the noise interpretation.\n"
                )
            if sample:
                result = sample_matter_emulator_fisher(
                    output=output, sample_power=sample_power, seeds=seeds,
                    min_effective=min_effective,
                )

        except Exception as error:
            assembly_error = f"{type(error).__name__}: {error}"
    if mpi_comm is not None:
        assembly_error = mpi_comm.bcast(assembly_error, root=0)
    if assembly_error is not None:
        raise ValueError(assembly_error)
    return result


def matter_emulator_fisher_noise(record, reference=None, complete=True):
    """Score saved stencils using one full, default-accuracy Fisher matrix.

    record and reference are mappings returned by numpy.load(fisher.npz).
    No physical model is evaluated. reference fixes the marginalization
    and prior curvature while candidate stencils measure their own noise.
    Returned arrays use the reference parameter order. Analytic columns
    are marked separately; their zeros are not measured numerical noise.

    For CPL sampled as (w, w0pwa=w+wa), the pair transformation to (w,wa)
    has determinant one. Its ellipse area and noise weights are therefore
    exactly those of (w,w0pwa), with all other parameters marginalized.
    """
    import numpy as np
    from cosmolike_notebook_utils import fisher as cf

    base = record if reference is None else reference
    names = base["names"].tolist()
    if "w0" in names and "wa" in names:
        pair = ("w0", "wa")
    elif "w" in names and "wa" in names:
        pair = ("w", "wa")
    elif "w" in names and "w0pwa" in names:
        pair = ("w", "w0pwa")
    elif "As_1e9" in names and "omegam" in names:
        pair = ("As_1e9", "omegam")
    else:
        raise ValueError("FoM screen needs sampled dark-energy coordinates or As_1e9 and omegam.")
    i, j = (names.index(name) for name in pair)
    if record["names"].tolist() != names:
        raise ValueError("Accuracy screening requires the complete matching parameter order.")
    for key in ("center", "lower", "upper", "gaussian_width", "steps", "rows", "metric"):
        if not np.array_equal(record[key], base[key], equal_nan=True):
            raise ValueError(f"Accuracy screening changed {key}; keep the physical test fixed.")
    for key in ("amplitude_bias_cut_names", "amplitude_bias_cut_parameters",
                "amplitude_bias_cut_scales"):
        if not np.array_equal(record[key], base[key]):
            raise ValueError("Accuracy screening changed a coupled prior boundary.")
    if bool(record["metric_is_covariance"]) != bool(base["metric_is_covariance"]):
        raise ValueError("Accuracy screening changed covariance/precision interpretation.")
    metric = np.asarray(record["metric"])
    if bool(record["metric_is_covariance"]):
        factor = np.linalg.cholesky(metric)
        inverse = np.linalg.solve(factor, np.eye(len(factor)))
        precision = inverse.T @ inverse
    else:
        precision = metric
    measured_names = record["stencil_names"].tolist()
    measured = np.array([names.index(name) for name in measured_names], dtype=int)
    analytic = set(record["analytic_names"].tolist())
    known = set(measured_names) | analytic
    if set(measured_names) & analytic or known - set(names) or (complete and known != set(names)):
        raise ValueError("Every derivative must have a measured stencil or an analytic declaration.")
    answer = {"names": names, "pair": pair, "measured": measured, "analytic": sorted(analytic),
              "unmeasured": [name for name in names if name not in known]}
    for suffix, divisor, field in (("h", 1., "stencils_h"), ("half_h", 2., "stencils_half_h")):
        steps = np.asarray(record["steps"])/divisor
        sigma, additions = np.zeros(len(names)), np.zeros(len(names))
        if len(measured):
            score = cf.retained_stencil_noise(
                stencil_values=record[field], steps=steps[measured], invcov=precision,
            )
            sigma[measured], additions[measured] = score["sigma"], score["a"]
        # Both candidate step choices use the SAME full default Fisher.
        # A refined setting must not alter its own screening weights.
        prediction = cf.predict_fom_noise(F=base["fisher_fine"], i=i, j=j, a=additions)
        weights = cf.get_fom_noise_weights(F=base["fisher_fine"], i=i, j=j, eps=steps)
        answer[suffix] = dict(prediction, sigma=sigma, a=additions, steps=steps,
                              lever=weights["lever"])
    return answer


def compare_matter_emulator_fisher_accuracy(base, refined, tolerance=0.01):
    """Require target-pair FoM stability in full h/h2 and accuracy comparisons.

    This test's default tolerance is one percent, not a universal scientific
    requirement. Priors, fiducial, data rows and covariance remain fixed.
    A small fourth-difference estimate alone never passes this gate.
    """
    from pathlib import Path
    import numpy as np
    from cosmolike_notebook_utils import fisher as cf

    base, refined = Path(base), Path(refined)
    first, second = np.load(base/"fisher.npz"), np.load(refined/"fisher.npz")
    score = matter_emulator_fisher_noise(second, reference=first)
    if not np.isfinite(tolerance) or tolerance <= 0.:
        raise ValueError("FoM convergence tolerance must be positive and finite.")
    names, pair = score["names"], score["pair"]
    values = np.array([
        cf.figure_of_merit(record[key], names=names, pair=pair)
        for record in (first, second)
        for key in ("fisher_coarse", "fisher_fine")
    ])
    changes = np.array([values[1]/values[0]-1., values[3]/values[2]-1.,
                        values[2]/values[0]-1., values[3]/values[1]-1.])
    passed = bool(np.all(np.isfinite(changes)) and np.all(np.abs(changes) < tolerance))
    np.savetxt(refined/"fom_convergence.txt", np.c_[np.arange(4), values, changes],
               header="row FoM relative_change; FoMs=base_h,base_h2,refined_h,refined_h2; "
                      "changes=base_step,refined_step,accuracy_h,accuracy_h2")
    (refined/"fom_convergence_status.txt").write_text(
        f"pair={pair}; relative_tolerance={tolerance}; passed={passed}\n"
        "The full Fisher retains all selected nuisance parameters and priors.\n"
        "The noise screen is diagnostic; correlated errors can evade it.\n"
    )
    if not passed:
        raise ValueError("Full Fisher FoM is not stable at the requested step/accuracy tolerance; "
                         "inspect fom_convergence.txt and refine further.")
    return values


def sample_matter_emulator_fisher(output, sample_power=18, seeds=(137, 138), min_effective=200):
    """Integrate a saved centered Fisher Gaussian over all prior bounds.

    Arguments:
      output = directory holding compute_matter_emulator_fisher's archive.
      sample_power = base-two sample exponent per independent Sobol scramble.
      seeds = at least two distinct integer scrambling seeds.
      min_effective = minimum importance effective count in each scramble.
    Returns:
      dictionary with names, values, weights, covariance and diagnostics.
    Side effects:
      Saves weighted samples and seed/step convergence diagnostics. Derivatives
      are never recomputed. Low effective count raises after saving diagnostics.

    A resolved positive-definite Fisher matrix supplies a joint Gaussian
    proposal, including all nuisance correlations. Rejecting outside ALL
    prior bounds gives the bounded marginal. For unresolved or null directions,
    retain the uniform cosmological-box and conditional-nuisance proposal;
    proper finite bounds then provide the information absent from the Fisher.
    """
    from pathlib import Path
    import numpy as np
    from scipy.special import ndtri, logsumexp
    from scipy.stats import qmc
    from cosmolike_notebook_utils import fisher as cf

    output = Path(output)
    record = np.load(file=output / "fisher.npz")
    names = record["names"].tolist()
    center, lower, upper = (record[key] for key in ("center", "lower", "upper"))
    fisher = cf.symmetric_matrix(record["fisher_fine"], name="Fisher")
    coarse = cf.symmetric_matrix(record["fisher_coarse"], name="Coarse Fisher")
    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Use at least two distinct sampling seeds.")
    if not isinstance(sample_power, int) or not 1 <= sample_power <= 24:
        raise ValueError("sample_power must be an integer between 1 and 24.")
    if not np.isfinite(min_effective) or min_effective <= 0.0:
        raise ValueError("min_effective must be positive.")
    cosmological = {"As_1e9", "ns", "H0", "omegab", "omegam", "w", "wa", "w0",
                    "w0pwa", "sigma8", "S8"}
    finite = np.isfinite(lower) & np.isfinite(upper)
    null = (np.diag(fisher) == 0.0) & (np.diag(coarse) == 0.0)
    if np.any(null & ~finite):
        raise ValueError("A zero-information parameter needs proper finite bounds.")
    uniform = np.array([i for i, name in enumerate(names)
                        if finite[i] and (name in cosmological or null[i])], dtype=int)
    conditional = np.array([i for i in range(len(names)) if i not in uniform], dtype=int)
    schur = fisher[np.ix_(uniform, uniform)].copy()
    if len(conditional):
        block = fisher[np.ix_(conditional, conditional)]
        # Only the conditional block requires inversion. A bounded flat
        # coordinate is integrated uniformly, without a pseudoinverse.
        conditional_covariance = cf.parameter_covariance(
            fisher=block, names=[names[i] for i in conditional],
        )
        response = -np.linalg.solve(block, fisher[np.ix_(conditional, uniform)])
        factor = np.linalg.cholesky(conditional_covariance)
        schur += fisher[np.ix_(uniform, conditional)] @ response
    if len(uniform):
        width = upper[uniform] - lower[uniform]
        scaled = 0.5*(schur+schur.T) * width[:, None] * width[None, :]
        eigenvalues = np.linalg.eigvalsh(scaled)
        if eigenvalues[0] < -1.e-10 * max(1.0, eigenvalues[-1]):
            raise ValueError("Fisher Schur complement has negative curvature.")

    gaussian_factor = None
    if not np.any(null):
        try:
            gaussian_factor = np.linalg.cholesky(
                cf.parameter_covariance(fisher=fisher, names=names))
        except (ValueError, np.linalg.LinAlgError):
            # The bounded integral can remain proper even when the full
            # unbounded Gaussian has an unresolved direction. The existing
            # conditional construction above has already checked its blocks.
            pass
    proposal = "joint Gaussian" if gaussian_factor is not None else "uniform and conditional Gaussian"
    values_list, weights_list, coarse_logs, seed_list, diagnostics, masses = [], [], [], [], [], []
    for seed in seeds:
        draws = qmc.Sobol(d=len(names), scramble=True, seed=seed).random_base2(sample_power)
        if gaussian_factor is not None:
            normal = ndtri(np.clip(draws, np.finfo(float).eps,
                                   1.0-np.finfo(float).eps))
            values = center + normal @ gaussian_factor.T
            # The proposal is the full fine-step Fisher Gaussian itself.
            # Its density cancels from the importance ratio; only the
            # prior boundaries below reject a point. Nuisances remain joint.
            log_weight = np.zeros(len(draws))
        else:
            values = np.tile(center, (len(draws), 1))
            values[:, uniform] = lower[uniform] + draws[:, uniform]*(upper[uniform]-lower[uniform])
            displacement = values[:, uniform] - center[uniform]
            if len(conditional):
                normal = ndtri(np.clip(draws[:, conditional], np.finfo(float).eps,
                                       1.0-np.finfo(float).eps))
                values[:, conditional] = center[conditional] + displacement @ response.T + normal @ factor.T
            distance = np.einsum("ij,jk,ik->i", displacement, schur, displacement)
            log_weight = -0.5*distance
        accepted = np.all((values >= lower) & (values <= upper), axis=1)
        # These boundaries couple a cosmological amplitude to a nuisance
        # bias. Applying them before marginalization keeps the same support
        # as the explicitly declared external hard priors of the model.
        if "amplitude_bias_cut_parameters" in record:
            for pair, scales in zip(record["amplitude_bias_cut_parameters"],
                                    record["amplitude_bias_cut_scales"]):
                amplitude, bias = (names.index(name) for name in pair)
                amplitude0, bias0, limit = scales
                accepted &= ((values[:, amplitude]/amplitude0)
                             * (values[:, bias]/bias0)**2 <= limit)
        weight = np.exp(log_weight) * accepted
        total = np.sum(weight)
        effective = total**2 / np.sum(weight**2) if total > 0.0 else 0.0
        if total > 0.0:
            normalized = weight / total
            mean = normalized @ values
            sigma = np.sqrt(normalized @ ((values-mean)**2))
        else:
            mean = sigma = np.full(len(names), np.nan)
        diagnostics.append([seed, effective, *mean, *sigma])
        masses.append(total / len(weight))
        values_list.append(values)
        weights_list.append(weight)
        seed_list.append(np.full(len(values), seed))
        full_displacement = values-center
        changed = np.einsum("ij,jk,ik->i", full_displacement, coarse-fisher, full_displacement)
        coarse_logs.append(np.where(accepted, log_weight-0.5*changed, -np.inf))
    values, weights = np.concatenate(values_list), np.concatenate(weights_list)
    coarse_log = np.concatenate(coarse_logs)
    finite_log = np.isfinite(coarse_log)
    coarse_weight = np.zeros(len(values))
    if finite_log.any():
        coarse_weight[finite_log] = np.exp(coarse_log[finite_log]-np.max(coarse_log[finite_log]))
    diagnostics = np.asarray(diagnostics)
    seed_means = diagnostics[:, 2:2+len(names)]
    seed_widths = diagnostics[:, 2+len(names):]
    average_width = np.mean(seed_widths, axis=0)
    width_spread = np.max(np.ptp(seed_widths, axis=0) / average_width)
    mean_spread = np.max(np.ptp(seed_means, axis=0) / average_width)
    mass_spread = np.ptp(masses) / np.mean(masses) if np.mean(masses) > 0.0 else np.inf
    # Effective count alone is not a convergence test. Independent
    # scramblings must also agree on means, widths and normalization.
    sampling_valid = bool(
        np.all(diagnostics[:, 1] >= min_effective)
        and width_spread <= 0.2 and mean_spread <= 0.25 and mass_spread <= 0.25
    )
    # Fine-proposal convergence does not establish coarse reweighting
    # convergence: a broader coarse Gaussian may need tails absent from
    # the fine sample. Check its independent scramblings separately.
    coarse_diagnostics, coarse_masses = [], []
    for seed, sample, log_weights in zip(seeds, values_list, coarse_logs):
        finite_here = np.isfinite(log_weights)
        if not finite_here.any():
            coarse_diagnostics.append([seed, 0., *([np.nan]*(2*len(names)))])
            coarse_masses.append(-np.inf)
            continue
        norm = logsumexp(log_weights[finite_here])
        normalized_here = np.zeros(len(sample))
        normalized_here[finite_here] = np.exp(log_weights[finite_here]-norm)
        mean_here = normalized_here @ sample
        sigma_here = np.sqrt(normalized_here @ ((sample-mean_here)**2))
        coarse_diagnostics.append([seed, 1./np.sum(normalized_here**2), *mean_here, *sigma_here])
        coarse_masses.append(norm-np.log(len(sample)))
    coarse_diagnostics = np.asarray(coarse_diagnostics)
    coarse_average_width = np.mean(coarse_diagnostics[:, 2+len(names):], axis=0)
    coarse_width_spread = np.max(np.ptp(coarse_diagnostics[:, 2+len(names):], axis=0)/coarse_average_width)
    coarse_mean_spread = np.max(np.ptp(coarse_diagnostics[:, 2:2+len(names)], axis=0)/coarse_average_width)
    coarse_masses = np.exp(np.asarray(coarse_masses)-np.max(coarse_masses))
    coarse_mass_spread = np.ptp(coarse_masses)/np.mean(coarse_masses)
    coarse_valid = bool(np.all(coarse_diagnostics[:, 1] >= min_effective)
                        and coarse_width_spread <= .2 and coarse_mean_spread <= .25
                        and coarse_mass_spread <= .25)
    np.savetxt(output/"coarse_sampling_convergence.txt", coarse_diagnostics,
               header="seed effective_count "+" ".join("mean_"+n for n in names)
                      +" "+" ".join("sigma_"+n for n in names))
    (output/"coarse_sampling_checks.txt").write_text(
        f"coarse_sampling_valid={coarse_valid}; minimum_effective={min_effective}\n"
        f"maximum_relative_seed_width_spread={coarse_width_spread:.8g}\n"
        f"maximum_seed_mean_spread_in_sigma={coarse_mean_spread:.8g}\n"
        f"relative_seed_normalization_spread={coarse_mass_spread:.8g}\n"
    )
    np.savez_compressed(
        file=output / "bounded_samples.npz", names=names, values=values,
        weights=weights, coarse_weights=coarse_weight, seed=np.concatenate(seed_list),
        lower=lower, upper=upper, proposal=proposal,
        uniform_indices=uniform if gaussian_factor is None else np.array([], dtype=int),
        sampling_valid=sampling_valid, coarse_sampling_valid=coarse_valid,
    )
    np.savetxt(
        fname=output / "sampling_convergence.txt", X=diagnostics,
        header="seed effective_count " + " ".join("mean_"+n for n in names)
        + " " + " ".join("sigma_"+n for n in names),
    )
    (output / "sampling_checks.txt").write_text(
        f"maximum_relative_seed_width_spread={width_spread:.8g}; limit=0.2\n"
        f"maximum_seed_mean_spread_in_sigma={mean_spread:.8g}; limit=0.25\n"
        f"relative_seed_normalization_spread={mass_spread:.8g}; limit=0.25\n"
        f"minimum_effective={min_effective}; sampling_valid={sampling_valid}\n"
    )
    if not sampling_valid:
        raise ValueError("Bounded-Fisher sampling did not converge; increase sample_power "
                         "with sample_matter_emulator_fisher. Derivatives remain saved.")
    if not coarse_valid:
        raise ValueError("Coarse-Fisher reweighting is unresolved; fine contours remain saved, "
                         "but the h/h2 width comparison is not valid. Increase sampling "
                         "or sample the coarse matrix independently.")
    normalized = weights/np.sum(weights)
    mean = normalized @ values
    delta = values-mean
    bounded_covariance = delta.T @ (normalized[:, None]*delta)
    cn = coarse_weight/coarse_weight.sum()
    cm = cn @ values
    cc = (values-cm).T @ (cn[:, None]*(values-cm))
    sigma, coarse_sigma = np.sqrt(np.diag(bounded_covariance)), np.sqrt(np.diag(cc))
    np.savetxt(
        fname=output / "bounded_widths.txt",
        X=np.column_stack((np.arange(len(names)), mean, sigma, coarse_sigma)),
        header="index mean sigma_fine sigma_coarse; names=" + " ".join(names),
    )
    (output / "sampling_definition.txt").write_text(
        "All bounds imposed before marginalization; centered local Fisher, not data posterior.\n"
        f"Proposal: {proposal}.\n"
        "Importance effective count measures weight concentration; Sobol points "
        "are not independent posterior draws. Compare independent scramblings.\n"
        f"sample_power={sample_power}; seeds={seeds}; minimum_effective={min_effective}\n"
        f"coarse_reweight_effective={1.0/np.sum(cn**2):.8g}\n"
        f"maximum_relative_width_change_h_to_half_h={np.max(np.abs(coarse_sigma/sigma-1.0)):.8g}\n"
    )
    return {"names": names, "values": values, "weights": weights,
            "covariance": bounded_covariance, "diagnostics": diagnostics}


def plot_matter_emulator_fisher(
    output,
    points_file,
    distances_file,
    names=None,
    project_name=None,
    local_points_file=None,
    local_distances_file=None,
    amplitude_max=3.5,
    omega_m_bounds=(0.2, 0.5),
):
    """Plot direct emulator errors with bounded local Fisher marginals.

    The project name identifies the survey. Colors and marker shapes
    distinguish the accuracy bands; open markers show prior draws and
    filled markers show an optional local sample. Solid and dashed contours enclose 68 and 95
    percent of the bounded local Fisher distribution, respectively.

    Arguments:
      output = Fisher directory containing bounded_samples.npz.
      points_file = named-column comparison parameters in saved row order.
      distances_file = distances.txt with matching sequential row indices.
      names = optional displayed cosmological names.
      project_name = project label printed on the figure.
      local_points_file, local_distances_file = optional second comparison.
      amplitude_max = upper plotting and integration bound on 1e9 As;
        a tighter saved prior bound remains in force.
      omega_m_bounds = comparison interval imposed before marginalization;
        intersects, rather than widens, the saved prior.
    Returns:
      saved PNG and PDF paths. No model evaluations occur.
    """
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from getdist import MCSamples

    output = Path(output)
    if project_name is None:
        project_name = next((p.name for p in output.parents if p.parent.name == "projects"), "Project")
    if (local_points_file is None) != (local_distances_file is None):
        raise ValueError("The local comparison requires both points and distances.")
    record = np.load(file=output / "bounded_samples.npz")
    if not bool(record["sampling_valid"]):
        raise ValueError("Bounded sampling has not passed its convergence checks.")
    all_names = record["names"].tolist()
    lower, upper = record["lower"].copy(), record["upper"].copy()
    if "As_1e9" in all_names:
        i = all_names.index("As_1e9")
        if not np.isfinite(amplitude_max) or amplitude_max <= lower[i]:
            raise ValueError("Amplitude cap must exceed the saved lower bound.")
        upper[i] = min(upper[i], amplitude_max)
    if "omegam" in all_names:
        i = all_names.index("omegam")
        lower[i] = max(lower[i], omega_m_bounds[0])
        upper[i] = min(upper[i], omega_m_bounds[1])
        if not lower[i] < upper[i]:
            raise ValueError("No overlap with the Omega_m comparison interval.")
    # Impose a revised amplitude prior on the full-dimensional samples
    # before selecting columns. Cropping an already marginalized ellipse
    # would omit the change in its probability normalization.
    weights = record["weights"]
    positive = ((weights > 0.0)
                & np.all((record["values"] >= lower)
                         & (record["values"] <= upper), axis=1))
    if np.count_nonzero(positive) < 3:
        raise ValueError("Too few Fisher samples remain within the plot bounds.")
    comparison_files = [(points_file, distances_file, False)]
    if local_points_file is not None:
        comparison_files.append((local_points_file, local_distances_file, True))
    comparisons = []
    for point_path, distance_path, local in comparison_files:
        points = np.atleast_1d(np.genfromtxt(point_path, names=True))
        distances = np.atleast_2d(np.loadtxt(distance_path))
        if (len(points) != len(distances)
                or not np.array_equal(distances[:, 0], np.arange(len(points)))):
            raise ValueError("Comparison points and distance row indices disagree.")
        comparisons.append((points, distances[:, 1], local))
    if names is None:
        names = [name for name in ("As_1e9", "omegam", "H0", "omegab", "ns", "w", "wa", "w0pwa")
                 if name in all_names and name in comparisons[0][0].dtype.names]
    if len(names) < 2 or any(name not in all_names for name in names):
        raise ValueError("Select at least two cosmological names in the Fisher sample.")
    for points, _, _ in comparisons:
        if any(name not in points.dtype.names for name in names):
            raise ValueError("Every comparison must contain the displayed parameters.")
    index = [all_names.index(name) for name in names]
    ranges = {name: [float(lower[i]), float(upper[i])] for name, i in zip(names, index)}
    labels = {"As_1e9": r"10^9 A_s", "omegam": r"\Omega_m",
              "H0": r"H_0\;[\mathrm{km\,s^{-1}\,Mpc^{-1}}]",
              "omegab": r"\Omega_b", "ns": r"n_s", "w": r"w_0",
              "wa": r"w_a", "w0pwa": r"w_0+w_a"}
    samples = MCSamples(
        samples=record["values"][positive][:, index],
        weights=weights[positive],
        names=names,
        labels=[labels.get(name, name) for name in names],
        ranges=ranges,
        sampler="uncorrelated",
        settings={"ignore_rows": 0},
    )
    colors = ["#0072B2", "#009E73", "#8B5AA7", "#E69F00", "#333333"]
    markers = ["o", "s", "D", "^", "P"]
    error_labels = [r"$E<0.2$", r"$0.2\leq E<0.5$", r"$0.5\leq E<1$",
                    r"$1\leq E<4$", r"$E\geq4$"]
    with plt.rc_context():
        plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
        amplitude = "As_1e9" if "As_1e9" in names else names[0]
        horizontal = [name for name in names if name != amplitude]
        columns = min(2, len(horizontal))
        rows = (len(horizontal)+columns-1)//columns
        figure, axes = plt.subplots(rows, columns, squeeze=False,
                                    figsize=(5.5*columns, 4.4*rows), sharey=True)
        for panel, (axis, name) in enumerate(zip(axes.flat, horizontal)):
            density = samples.get2DDensity(name, amplitude)
            levels = np.sort(density.getContourLevels([0.68, 0.95]))
            axis.contour(density.x, density.y, density.P, levels=levels,
                         colors="black", linestyles=["--", "-"], linewidths=1.5)
            for points, error, local in comparisons:
                valid = np.isfinite(error) & (error >= 0.0)
                within = np.ones(len(points), dtype=bool)
                for j, parameter in enumerate(all_names):
                    if parameter in points.dtype.names:
                        within &= (points[parameter] >= lower[j]) & (points[parameter] <= upper[j])
                categories = [valid & (error < 0.2), valid & (error >= 0.2) & (error < 0.5),
                              valid & (error >= 0.5) & (error < 1.0),
                              valid & (error >= 1.0) & (error < 4.0), valid & (error >= 4.0)]
                for chosen, color, marker in zip(categories, colors, markers):
                    chosen = chosen & within
                    axis.scatter(points[name][chosen], points[amplitude][chosen],
                                 facecolors=color if local else "none", marker=marker,
                                 s=48 if local else 38, edgecolors="black" if local else color,
                                 linewidths=0.7 if local else 1.2, zorder=3 if local else 2)
                failed = ~valid & within
                axis.scatter(points[name][failed], points[amplitude][failed],
                             color="black", marker="x", s=44, linewidths=1.2,
                             zorder=4)
            axis.set_xlabel("$" + labels.get(name, name) + "$", fontsize=14)
            if panel % columns == 0:
                axis.set_ylabel("$" + labels.get(amplitude, amplitude) + "$", fontsize=15)
            axis.tick_params(labelsize=11)
            for variable, setter in ((name, axis.set_xlim), (amplitude, axis.set_ylim)):
                if np.all(np.isfinite(ranges[variable])):
                    setter(*ranges[variable])
            if amplitude == "As_1e9":
                lo, hi = ranges[amplitude]
                ticks = np.arange(np.ceil(np.nextafter(lo, -np.inf)*2)/2, hi, 0.5)
                axis.set_yticks(np.append(ticks, hi))
        for axis in list(axes.flat)[len(horizontal):]:
            axis.set_visible(False)
        handles = [Line2D([], [], color=color, marker=marker, linestyle="none", label=label)
                   for color, marker, label in zip(colors, markers, error_labels)]
        handles.append(Line2D([], [], color="black", marker="x", linestyle="none",
                              label="Failed evaluation"))
        handles.append(Line2D([], [], color="black", marker="o", linestyle="none",
                              markerfacecolor="none", label="Prior (open)"))
        if local_points_file is not None:
            handles.append(Line2D([], [], color="black", marker="o", linestyle="none",
                                  label=f"Local {len(comparisons[1][0])}"))
        handles.extend([Line2D([], [], color="black", linestyle="-", label="Bounded Fisher 68%"),
                        Line2D([], [], color="black", linestyle="--", label="Bounded Fisher 95%")])
        figure.suptitle(str(project_name).replace("_", " ").upper(), fontsize=18, y=0.995)
        legend = figure.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.96),
            ncol=min(4, 2 * columns), frameon=False, fontsize=11,
        )
        # Reserve the legend's measured height, including the extra error
        # bands. A fixed top margin clips labels in the one-panel test.
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        legend_box = legend.get_window_extent(renderer=renderer)
        legend_fraction = legend_box.transformed(figure.transFigure.inverted())
        figure.tight_layout(rect=(0, 0, 1, legend_fraction.y0 - 0.025))
        figure.canvas.draw()
        for axis in axes.flat:
            if axis.get_visible() and legend_box.overlaps(axis.get_window_extent()):
                raise ValueError("The Fisher legend overlaps a panel; increase the figure height.")
        png, pdf = output / "emulator_fisher.png", output / "emulator_fisher.pdf"
        figure.savefig(png, dpi=180, bbox_inches="tight")
        figure.savefig(pdf, bbox_inches="tight")
        plt.close(figure)
    return png, pdf
