"""Factories that build the shared test classes for one project.

Every project's test suite runs the same checks on its own survey:
the test classes are therefore shared code, and this module is where
they live. Each factory takes a project's cocoa_test_utils module
(the shim that binds the shared harness in cocoa_testing.py to that
project's tables) and returns a ready test class closed over it. A
project's test file binds the class to the module-level name pytest
collects:

    import cocoa_test_utils as u
    TestBaryonDrift = u.baryon_drift_suite(u)

The binding changes nothing about collection or test ids: pytest
finds the class through the module attribute exactly as it found the
class statement the factory replaced. The docstrings on the classes
and their methods are the documentation the test reports show, and
keeping them here keeps them identical across the projects.

What each check means physically is documented once in this
repository's README (the "shared data-vector checks" section); the
factories' docstrings describe what runs, not why.
"""

import os
import re
import unittest


def _interface(u):
    """The project's compiled interface module, imported on first use.

    The harness already knows the module's name (the shim passes it to
    CocoaTestHarness); importing through it keeps the factories free
    of project names. The import happens inside the test bodies, so
    collecting the tests never loads the compiled library.

    Arguments:
      u = the project's cocoa_test_utils module.

    Returns:
      the imported interface module.
    """
    import importlib
    return importlib.import_module(u._H.interface_module)


def _native_space(u, ci):
    """The binning space the project's frozen examples score in.

    A single-space interface binds the generic getter names
    (compute_data_vector_masked, the 3x2pt sizes) to its native
    space, so the space follows from which size binding exists. A
    dual-space interface binds both families and keeps the generic
    names on the real-space layout; its shim says which space the
    frozen examples use through a module constant BINNING_SPACE
    ("real" or "fourier").

    Arguments:
      u  = the project's cocoa_test_utils module.
      ci = the imported interface module.

    Returns:
      "real" or "fourier".
    """
    space = getattr(u, "BINNING_SPACE", None)
    if space is None:
        space = ("real"
                 if hasattr(ci, "compute_data_vector_3x2pt_real_sizes")
                 else "fourier")
    return space


def _masked_vector_fn(u, ci):
    """The masked theory-vector getter matching _native_space.

    compute_data_vector_masked returns the theory vector of the
    state the last evaluation left in cosmolike. On a dual-space
    interface that name keeps the real-space layout and indexes out
    of bounds when the loaded dataset holds Fourier band powers, so
    the Fourier twin must be read instead.

    Arguments:
      u  = the project's cocoa_test_utils module.
      ci = the imported interface module.

    A Fourier project may append a gg-cross block after the three
    probe blocks; the plain Fourier getter leaves those entries
    zero, so the extended getter is the vector the likelihood
    scores. The choice is made at call time, because the cross-pair
    list is installed by the first model build, after this function
    is bound.

    Returns:
      a zero-argument callable returning the masked theory vector.
    """
    def masked_vector():
        if (_native_space(u, ci) == "fourier"
                and hasattr(ci, "compute_data_vector_fourier_masked")):
            if _ggx_len(ci) > 0:
                return ci.compute_data_vector_fourier_gg_cross_masked()
            return ci.compute_data_vector_fourier_masked()
        return ci.compute_data_vector_masked()
    return masked_vector


def _ggx_len(ci):
    """The appended gg-cross block's entry count (0 when absent).

    Interfaces with an extended [EE | gs | gg | gg-cross] Fourier
    layout expose the four block sizes; the fourth stays zero until
    the likelihood installs its cross-pair list.

    Arguments:
      ci = the imported interface module.

    Returns:
      the number of appended gg-cross entries.
    """
    if hasattr(ci, "compute_data_vector_fourier_with_gg_cross_sizes"):
        return int(
            ci.compute_data_vector_fourier_with_gg_cross_sizes()[3])
    return 0


def _ggl_excluded(like):
    """The excluded (lens, source) ggl pairs as a set of tuples.

    The likelihood yaml accepts a list of pairs or the string "all"
    (a clustering-only likelihood drops every pair).

    Arguments:
      like = the likelihood object.

    Returns:
      {(lens bin, source bin), ...} of the excluded pairs.
    """
    raw = getattr(like, "ggl_exclude", None)
    if isinstance(raw, str):
        return {(zl, zs)
                for zl in range(int(like.lens_ntomo))
                for zs in range(int(like.source_ntomo))}
    return {(int(a), int(b)) for a, b in (raw or [])}


def _fourier_nlen(like):
    """The entries per spectrum of a Fourier data vector.

    Fourier likelihoods carry the multipole count under one of two
    names: an n_cl dataset key (ncl), or an explicit list of band
    centers (band_ells).

    Arguments:
      like = the likelihood object.

    Returns:
      the number of multipole entries per spectrum.
    """
    if hasattr(like, "ncl"):
        return int(like.ncl)
    return len(like.band_ells)


def _chi2_method(u, example, number, label, tatt):
    """Build one chi2-vs-frozen-reference test method.

    Arguments:
      u       = the project's cocoa_test_utils module.
      example = the key of u.EXAMPLES the method evaluates.
      number  = the test number as a string, possibly x-prefixed; the
                prefix stays in the method name and drops out of the
                printed report.
      label   = the opening of the printed report line, up to the IA
                model, e.g. "example1 (cosmic shear".
      tatt    = True for the TATT variant of the frozen point.

    Returns:
      the test method, named and documented; example_suite installs
      it on the class it builds.
    """
    ia_key = "tatt" if tatt else "nla"
    ia_label = "TATT" if tatt else "NLA"

    def test(self):
        chi2 = u.single_model_chi2(example, tatt=tatt)
        # reference_chi2.json names each entry <example>_<IA model>
        ref = self.reference[f"{example}_{ia_key}"]
        # report_chi2_test prints chi2, the reference, their
        # difference and the limit; it never asserts (the assertion
        # follows)
        u.report_chi2_test(
            int(number.lstrip("x")),
            f"{label}, {ia_label}) chi2 vs frozen reference",
            chi2, ref, u.CHI2_TOLERANCE)
        # assertLess(a, b) passes when a < b and fails with msg
        # otherwise; in that message, :.6f prints fixed six decimals
        self.assertLess(
            abs(chi2 - ref), u.CHI2_TOLERANCE,
            msg=f"chi2 = {chi2:.6f} vs frozen reference {ref:.6f} "
                f"(|delta| >= {u.CHI2_TOLERANCE})")

    # the method name carries the test number; pytest and unittest
    # find the method under this name on the built class
    test.__name__ = (f"test_{number}_chi2_matches_frozen_reference"
                     + ("_tatt" if tatt else ""))
    if tatt:
        test.__doc__ = (
            "The chi2 check repeated with the TATT IA model.\n\n"
            "        TATT replaces the NLA point's zero second-order\n"
            "        amplitudes with the nonzero values of TATT_POINT,\n"
            "        so the frozen TATT reference genuinely exercises\n"
            "        the tidal second-order terms.\n        ")
    else:
        test.__doc__ = (
            "chi2 at the frozen NLA point stays within 0.2 of the "
            "reference.\n\n"
            "        u.single_model_chi2 starts one worker, which\n"
            "        builds the cobaya model (CAMB plus the\n"
            "        likelihood) from the frozen configuration, checks\n"
            "        that the frozen point names exactly the\n"
            "        parameters the model samples, and returns\n"
            "        chi2 = -2 ln L at that point. The pass limit\n"
            "        CHI2_TOLERANCE = 0.2 is the project's bound for\n"
            "        every frozen chi2: compiler and library-version\n"
            "        noise stays below it, and no physics change is\n"
            "        detectable below a chi2 shift of 0.2, so a\n"
            "        failure means the code or the frozen inputs\n"
            "        changed the prediction.\n        ")
    return test


def _race_method(u, example, number, label, tatt):
    """Build one ten-in-a-row race-check test method.

    Arguments: as in _chi2_method.

    Returns:
      the test method, named and documented; example_suite installs
      it on the class it builds.
    """
    ia_label = "TATT" if tatt else "NLA"

    def test(self):
        # with one thread no race can occur: a process that did not
        # get the 4 threads set at the top of the test file stops here
        u.assert_omp_threads()
        # the function returns a (fresh, tenth) pair; the assignment
        # unpacks it into the two names
        fresh, tenth = u.ten_in_a_row_chi2(example, tatt=tatt)
        u.report_race_test(
            int(number.lstrip("x")),
            f"{label}, {ia_label}) race check: "
            "10 cosmologies in a row",
            fresh, tenth, u.RACE_TOLERANCE)
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg=f"10th-in-a-row chi2 = {tenth:.8f} vs fresh "
                f"{fresh:.8f}")

    test.__name__ = (f"test_{number}_no_race_condition_ten_in_a_row"
                     + ("_tatt" if tatt else ""))
    if tatt:
        test.__doc__ = ("The race check repeated with the TATT IA "
                        "model.\n        ")
    else:
        test.__doc__ = (
            "The fiducial as 10th of 10 cosmologies matches a fresh "
            "run.\n\n"
            "        u.ten_in_a_row_chi2 starts one worker that\n"
            "        builds one model and evaluates on it, in order:\n"
            "        the frozen fiducial point (the fresh value), the\n"
            "        nine cosmologies of RACE_PERTURBATIONS, and the\n"
            "        fiducial again, the 10th point of the row.\n"
            "        Correct code gives the same chi2 both times. A\n"
            "        table kept from an earlier cosmology (leaked\n"
            "        state) or two OpenMP threads writing the same\n"
            "        memory (a race) shift the second value.\n"
            "        RACE_TOLERANCE = 1e-4: both values come from the\n"
            "        same code on the same inputs, so only\n"
            "        floating-point noise is allowed, while leaked\n"
            "        state shifts the chi2 by much more.\n        ")
    return test


def example_suite(u, name, example, label, numbers, tatt_numbers=None):
    """Build one frozen-example test class (chi2 + race, maybe TATT).

    Arguments:
      u            = the project's cocoa_test_utils module.
      name         = the class name to build; the caller assigns the
                     class to this same module-level name, which is
                     what pytest collects, so the test ids do not
                     change.
      example      = the key of u.EXAMPLES the class evaluates.
      label        = the opening of every printed report line, up to
                     the IA model, e.g. "example1 (cosmic shear".
      numbers      = the (chi2 test, race test) numbers as strings,
                     e.g. ("1", "2") or ("x11", "x12"); an x prefix
                     only keeps unittest's alphabetical ordering
                     aligned with the numbering (test_11 would sort
                     before test_2) and drops out of the printed
                     reports.
      tatt_numbers = the same pair for the TATT variants, or None for
                     a configuration without them (the cluster
                     combinations use no TATT variant).

    Returns:
      a unittest.TestCase subclass named `name`.
    """
    # each helper returns a ready method; the dict keys must equal
    # the methods' own __name__ for unittest to find and run them
    methods = [
        _chi2_method(u, example, numbers[0], label, tatt=False),
        _race_method(u, example, numbers[1], label, tatt=False),
    ]
    if tatt_numbers is not None:
        methods.append(_chi2_method(u, example, tatt_numbers[0],
                                    label, tatt=True))
        methods.append(_race_method(u, example, tatt_numbers[1],
                                    label, tatt=True))

    @classmethod
    def setUpClass(cls):
        """Move to ROOTDIR, verify the frozen state, load the references."""
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    # the class docstring names the test-number range; int(...) of
    # the x-stripped strings recovers the printed numbers
    spanned = [int(n.lstrip("x"))
               for n in tuple(numbers) + tuple(tatt_numbers or ())]
    doc = f"""Tests {min(spanned)}-{max(spanned)}, sharing one frozen-state verification.

    setUpClass runs once before the tests: it moves to ROOTDIR,
    verifies every frozen file against the SHA-256 manifest (an edited
    frozen state must fail loudly before any physics runs), and loads
    the frozen reference chi2 values.
    """

    # type(name, bases, namespace) builds the class exactly as a
    # class statement would, with the name chosen at run time
    namespace = {"__doc__": doc, "setUpClass": setUpClass}
    for method in methods:
        namespace[method.__name__] = method
    return type(name, (unittest.TestCase,), namespace)


def nmodels_file(u, example):
    """Return the path of an example's stored N-random-models vectors.

    generate_frozen_reference.py --nmodels writes the file; the
    manifest pins it with the rest of frozen/.

    Arguments:
      u       = the project's cocoa_test_utils module.
      example = a key of u.EXAMPLES.

    Returns:
      the path frozen/nmodels_<example>.npz.
    """
    return os.path.join(u.FROZEN_DIR, f"nmodels_{example}.npz")


def nmodels_draw(u, example):
    """Return the fixed N-random-models points of one example.

    The draw settings live in the project's cocoa_test_utils
    (NMODELS_BOX, NMODELS_HOLD, NMODELS_AMPLITUDE_CUT), the one place
    that both the check and generate_frozen_reference.py --nmodels
    read, so the two always draw the same points.

    Arguments:
      u       = the project's cocoa_test_utils module.
      example = a key of u.EXAMPLES.

    Returns:
      (points, held) of u.nmodels_points.
    """
    return u.nmodels_points(
        example, box_overrides=getattr(u, "NMODELS_BOX", None),
        hold=getattr(u, "NMODELS_HOLD", None),
        amplitude_cut=getattr(u, "NMODELS_AMPLITUDE_CUT", None))


def nmodels_vectors(u, example, points, high_accuracy):
    """Return the masked theory vector of every point, from one model.

    One cobaya model is built for the whole list and each point costs
    one evaluation. The compiled interface holds one model at a time,
    so the default and the high-accuracy vectors come from two
    separate calls, never from two live models.

    Arguments:
      u             = the project's cocoa_test_utils module.
      example       = a key of u.EXAMPLES.
      points        = the {sampled name: value} points.
      high_accuracy = True builds the model with the project's pushed
                      accuracy settings (load_frozen_info's
                      high_accuracy), False with the frozen defaults.

    Returns:
      (vectors, inv_cov): a (len(points), n) float64 array of masked
      theory vectors, and the (n, n) masked inverse covariance of the
      frozen data set.

    Raises:
      AssertionError when a point and the model disagree on the
      sampled parameters (the likelihood or theory code changed its
      parameters since the freeze).
    """
    import time
    import numpy as np

    ci = _interface(u)
    masked_vector = _masked_vector_fn(u, ci)
    info = u.load_frozen_info(example, tatt=False, high_accuracy=high_accuracy)
    if high_accuracy:
        label = "high-accuracy"
    else:
        label = "default-settings"
    print(f"  building the {label} model of {example} ...", flush=True)
    start = time.time()
    model = u.make_model(info)
    print(f"  built in {time.time() - start:.0f} s", flush=True)
    sampled = set(model.parameterization.sampled_params())
    vectors = []
    for index in range(len(points)):
        point = points[index]
        if sampled != set(point):
            raise AssertionError(
                "sampled-parameter set differs from the frozen point "
                "(the likelihood/theory code changed its parameters):\n"
                f"  new since freeze: {sorted(sampled - set(point))}\n"
                f"  gone since freeze: {sorted(set(point) - sampled)}")
        start = time.time()
        u.evaluate_chi2(model, point)
        vectors.append(np.array(masked_vector(), dtype="float64"))
        print(f"  model {index + 1}/{len(points)}: {label} vector "
              f"({time.time() - start:.1f} s)", flush=True)
    inv_cov = np.array(ci.get_inv_cov_masked(), dtype="float64")
    del model
    return np.array(vectors), inv_cov


def _nmodels_figure_dir():
    """Return the folder the N-random-models figure is written into.

    COCOA_ACCURACY_NMODELS_FIGURES names it. Unset, the figure goes
    into the system temporary folder, so an ordinary test run changes
    no project file. The check calls this before evaluating any model,
    so a wrong folder fails in seconds, not after minutes of models.

    Returns:
      the folder path.

    Raises:
      AssertionError when the named folder does not exist.
    """
    import tempfile

    figure_dir = os.environ.get("COCOA_ACCURACY_NMODELS_FIGURES", "")
    if figure_dir == "":
        return tempfile.gettempdir()
    if not os.path.isdir(figure_dir):
        raise AssertionError(
            f"COCOA_ACCURACY_NMODELS_FIGURES={figure_dir!r} is not an "
            "existing folder: create it, or unset the variable to write "
            "the figure into the system temporary folder")
    return figure_dir


def _nmodels_figure(u, example, points, deltas, figure_dir, seed, limit,
                    held=None, case=None):
    """Draw the N-random-models figure of one example; return its path.

    The file name carries the example and the seed base, so the figure
    names how to redraw its points. matplotlib is imported here, inside
    the call, so importing the suites never loads it.

    Arguments:
      u          = the project's cocoa_test_utils module.
      example    = the key of u.EXAMPLES the models were drawn for.
      points     = the drawn points, one {name: value} dict per model.
      deltas     = the per-model delta chi2, in the same order.
      figure_dir = the folder of _nmodels_figure_dir.
      seed       = the seed base of the points.
      limit      = the delta chi2 of the dashed line.
      held       = None, or one bool per model: True for a model that
                   kept the hold list at the frozen point.
      case       = None, or the check's label of the example (for
                   instance "example4 (Delta z = 0.2, NLA)"); with the
                   project name, the number of models and the seed base
                   it titles the figure.

    Returns:
      the PNG path.
    """
    import cocoa_testing_figures as figures

    path = os.path.join(figure_dir,
                        f"accuracy_nmodels_{example}_seed{seed}.png")
    # the project is the folder that holds the tests/ folder
    project = os.path.basename(os.path.dirname(u.TESTS_DIR))
    if case is None:
        case = example
    label = (f"{project}, {case.replace('Delta z', 'Δz')}: "
             f"{len(points)} models, seed base {seed}")
    figures.random_models_triangle(
        points=points, deltas=deltas,
        fiducial=u.load_frozen_point(example), path=path, limit=limit,
        held=held, label=label,
        colorbarlabel=r"$\Delta\chi^2$ (pushed settings $-$ default)")
    return path


def accuracy_suite(u, name, checks, knob_scan=None):
    """Build the accuracy-advisory class (A-checks, K-scan, N-models).

    Arguments:
      u               = the project's cocoa_test_utils module.
      name            = the class name to build; the caller assigns
                        the class to this same module-level name, so
                        pytest collects unchanged test ids.
      checks          = one (method_name, advisory, example, tatt,
                        label) tuple per A-check: the full collected
                        method name (e.g. "test_a1_cosmic_shear_nla"),
                        the printed advisory label (e.g. "A1"), the
                        key of u.EXAMPLES, the IA-model flag, and the
                        printed report line (e.g. "example1 (cosmic
                        shear, NLA)").
      knob_scan       = the EXAMPLES key the one-knob-at-a-time scan
                        evaluates (the method
                        test_a0_one_knob_at_a_time, reading the
                        "<key>_nla" reference), a tuple of keys for
                        one scan per example (the first keeps that
                        name, each further key adds
                        test_a0_one_knob_at_a_time_<key>), or None
                        for a project without the scan.
    The N-random-models check (test_ax99_nmodels, one per example) is
    configured in the project's cocoa_test_utils, which
    generate_frozen_reference.py --nmodels also reads:
      NMODELS_EXAMPLES      = (EXAMPLES key, display label) pairs, or
                              () for a project without the check; the
                              first keeps the name test_ax99_nmodels,
                              each further key adds
                              test_ax99_nmodels_<key>.
      NMODELS_LIMIT         = None keeps the check advisory (finite
                              deltas only); a number makes every
                              model's |delta chi2| stay below it.
      NMODELS_BOX, NMODELS_HOLD, NMODELS_AMPLITUDE_CUT
                            = the draw settings of nmodels_points.

    Returns:
      a unittest.TestCase subclass named `name`.
    """

    class TestAccuracy(unittest.TestCase):
        """The advisory accuracy checks, one test method each.

        They share one frozen-state verification (setUpClass) and one
        implementation (_accuracy_check), which evaluates a
        configuration in a worker subprocess and prints the report
        block. There is no pass/fail on the size of the delta: how
        much numerical error an analysis tolerates is a judgment
        call, so only a broken evaluation fails.
        """

        # @classmethod is a decorator (a line starting with @ that
        # wraps the function defined below it): it hands the method
        # the class itself (cls), not an instance; unittest calls
        # setUpClass once before the first test of the class
        @classmethod
        def setUpClass(cls):
            """Check the environment and the frozen state once for all tests.

            Runs once, before the first test of the class: it moves
            to ROOTDIR (the Cocoa/ folder; the frozen configurations
            name paths relative to it), verifies every frozen file
            against the SHA-256 manifest (an edited frozen state must
            fail loudly before any physics runs), and loads the
            frozen reference chi2 values.

            Arguments:
              cls = this test class (the classmethod decorator passes
                    the class itself, not an instance).

            Returns:
              nothing; cls.reference holds the dictionary of
              frozen/reference_chi2.json: {"<example>_nla" or
              "<example>_tatt": reference chi2} plus a "_meta" entry.

            Raises:
              RuntimeError when ROOTDIR is not set (start_cocoa.sh
              was not sourced); AssertionError when a frozen file
              differs from the manifest.
            """
            u.require_cocoa_environment()
            u.verify_frozen()
            cls.reference = u.load_reference()

        def _accuracy_check(self, check, example, tatt, label):
            """Evaluate one configuration at high accuracy and report.

            u.single_model_chi2(..., high_accuracy=True) starts a
            worker subprocess (a fresh python process, so no
            cosmolike state of another model can reach it) that loads
            the frozen configuration, applies the high-accuracy
            settings on top of it, builds the cobaya model, evaluates
            the chi2 at the frozen point, and hands it back as a
            float. The default-settings value is the frozen
            reference, so the printed difference is the numerical
            error the default settings carry.

            Arguments:
              check   = the advisory label (e.g. "A1") for the
                        report.
              example = a key of u.EXAMPLES (exact-physics
                        configurations only).
              tatt    = True evaluates the TATT variant against the
                        TATT-generated data vector, False the NLA
                        one.
              label   = one line naming the probe and IA model.

            Returns:
              nothing; the delta is printed and nothing is asserted.
            """
            chi2_high = u.single_model_chi2(example, tatt,
                                            high_accuracy=True)
            # `"tatt" if tatt else "nla"` picks the reference-key
            # suffix, the naming the frozen reference file uses
            suffix = "tatt" if tatt else "nla"
            default_ref = self.reference[f"{example}_{suffix}"]
            u.report_accuracy(f"{check}: {label}", chi2_high,
                              default_ref)

    # the A-check methods: one small closure per roster row, named
    # by the row's method_name so the collected ids do not change
    def make_check(check_name, advisory, example, tatt, label):
        def test(self):
            self._accuracy_check(advisory, example, tatt, label)
        test.__name__ = check_name
        test.__doc__ = (f"{advisory}: {label}, default vs high "
                        "accuracy.\n        ")
        return test

    for row in checks:
        method = make_check(*row)
        setattr(TestAccuracy, method.__name__, method)

    # the K scans: one method per bound example; a single key (a str)
    # is the one-scan case every project used before tuples existed
    scans = ((knob_scan,) if isinstance(knob_scan, str)
             else tuple(knob_scan or ()))

    def make_scan(example):
        def test_a0_one_knob_at_a_time(self):
            """K scan: each accuracy knob alone (check A0).

            Advisory: each knob's chi2 and its difference to the
            frozen default reference print as the scan runs, on the
            NLA configuration the project binds for the scan. A knob
            whose delta rivals the all-knobs delta is the driver; a
            knob whose delta explodes (orders of magnitude beyond the
            others) signals an interface breakdown, not a numerics
            improvement.
            """
            default_ref = self.reference[f"{example}_nla"]
            print("", flush=True)
            # each entry is (label, likelihood overrides, camb
            # overrides); the two _ discard the overrides, only the
            # label is needed here (the worker looks the knob up
            # again by its label)
            for label, _, _ in u.ACCURACY_KNOBS:
                chi2 = u.single_model_chi2(example, False,
                                           knob=label)
                # with several scans, each line names its example
                tag = f"{example}: " if len(scans) > 1 else ""
                u.report_knob(tag + label, chi2, default_ref)
        return test_a0_one_knob_at_a_time

    for k, example in enumerate(scans):
        method = make_scan(example)
        method.__name__ = ("test_a0_one_knob_at_a_time" if k == 0
                           else f"test_a0_one_knob_at_a_time_{example}")
        setattr(TestAccuracy, method.__name__, method)

    # one (example, label) pair is the single-check case every
    nmodels_pairs = tuple(tuple(pair) for pair
                          in getattr(u, "NMODELS_EXAMPLES", ()))
    nmodels_limit = getattr(u, "NMODELS_LIMIT", None)

    def make_nmodels(nm_example, nm_label):
        def test_ax99_nmodels(self):
                """N-random-models check across the prior.

                The A-checks measure the numerical error of the default
                settings at the one frozen fiducial point. This check
                measures it at NMODELS_COUNT fixed points across the
                prior. For each point

                    delta chi2 = (v_high - v_default)^T C^-1 (v_high - v_default),

                with v_high the stored vector at the pushed accuracy
                settings (frozen/nmodels_<example>.npz, written once by
                generate_frozen_reference.py --nmodels), v_default the
                vector at the default settings, computed here, and C the
                frozen covariance. Storing the expensive half keeps the
                check to one model build and one evaluation per point.

                The points are fixed: model m draws with the seed
                RANDOM_MODEL_SEED + m from the project's draw settings
                (NMODELS_BOX, NMODELS_HOLD and NMODELS_AMPLITUDE_CUT in
                cocoa_test_utils). The check redraws them and requires
                the stored points and the fingerprint of the
                high-accuracy configuration (nmodels_config_hash) to
                match, so a changed prior, draw or configuration asks
                for regenerated vectors instead of comparing the wrong
                ones. Without a stored file the check is skipped with
                that instruction.

                The deltas have to be finite; with NMODELS_LIMIT set,
                each |delta| must also stay below it, otherwise the
                check is advisory. Before the limit applies, the check
                draws the Omega_m - clustering-amplitude triangle of the
                points (cocoa_testing_figures.random_models_triangle)
                into the folder COCOA_ACCURACY_NMODELS_FIGURES names, or
                the system temporary folder. The ax99 in the method name
                sorts this check after the A-checks (unittest runs
                methods in name order).
                """
                import math
                import numpy as np

                path = nmodels_file(u, nm_example)
                if not os.path.isfile(path):
                    self.skipTest(
                        f"no stored N-random-models vectors for "
                        f"{nm_example} (frozen/{os.path.basename(path)}); "
                        "generate them with generate_frozen_reference.py "
                        "--nmodels")
                # checked before the models run, which take minutes
                figure_dir = _nmodels_figure_dir()
                seed = u.RANDOM_MODEL_SEED
                points, held = nmodels_draw(u, nm_example)
                n_models = len(points)
                print(f"\n  N-random-models check: {n_models} models, "
                      f"{nm_label}, seed base {seed}", flush=True)
                nmodels_box = getattr(u, "NMODELS_BOX", None)
                nmodels_hold = getattr(u, "NMODELS_HOLD", None)
                nmodels_cut = getattr(u, "NMODELS_AMPLITUDE_CUT", None)
                if nmodels_box is not None:
                    for box_name in nmodels_box:
                        box_low, box_high = nmodels_box[box_name]
                        print(f"  box override: {box_name} drawn in "
                              f"[{box_low}, {box_high}]", flush=True)
                if nmodels_cut is not None:
                    print("  amplitude cut: (A_s/A_s,fid)(b_i/b_i,fid)^2 <= "
                          f"{nmodels_cut:g} in every bin", flush=True)
                if nmodels_hold is not None:
                    print("  odd models hold at the fiducial: "
                          f"{', '.join(nmodels_hold)}", flush=True)

                # the stored vectors must belong to these points and to
                # today's high-accuracy configuration
                stored = np.load(path, allow_pickle=False)
                regenerate = ("regenerate the stored vectors with "
                              "generate_frozen_reference.py --nmodels")
                names = []
                for name in stored["names"]:
                    names.append(str(name))
                self.assertEqual(
                    names, sorted(points[0]),
                    msg="the stored points have other parameters than "
                        f"today's draw; {regenerate}")
                values = np.zeros((n_models, len(names)))
                for index in range(n_models):
                    for column in range(len(names)):
                        values[index, column] = points[index][names[column]]
                self.assertTrue(
                    np.array_equal(values, stored["values"]),
                    msg="the stored points differ from today's draw (the "
                        "prior, the draw settings or the fiducial "
                        f"changed); {regenerate}")
                self.assertEqual(
                    str(stored["config_hash"]),
                    u.nmodels_config_hash(nm_example),
                    msg="the high-accuracy configuration changed since "
                        f"the stored vectors were computed; {regenerate}")

                default_vectors, inv_cov = nmodels_vectors(
                    u, nm_example, points, high_accuracy=False)
                high_vectors = stored["vectors"]
                self.assertEqual(
                    high_vectors.shape, default_vectors.shape,
                    msg="the stored vectors have another length than "
                        f"today's masked data vector; {regenerate}")
                deltas = []
                for index in range(n_models):
                    difference = high_vectors[index] - default_vectors[index]
                    delta = float(difference @ inv_cov @ difference)
                    u.report_random_model(index, n_models, points[index],
                                          delta)
                    deltas.append(delta)
                # enumerate pairs each delta with its position, counted
                # from 0, so a broken model is named by number
                for index, delta in enumerate(deltas):
                    # advisory: the size of the delta is a judgment
                    # call, but a non-finite one means the evaluation
                    # itself broke
                    self.assertTrue(
                        math.isfinite(delta),
                        msg=f"model {index}: non-finite delta chi2 "
                            f"{delta!r}")
                u.report_random_model_summary(deltas)
                if nmodels_hold is not None:
                    drawn_deltas = []
                    held_deltas = []
                    for index in range(n_models):
                        if held[index]:
                            held_deltas.append(deltas[index])
                        else:
                            drawn_deltas.append(deltas[index])
                    print("  models drawn across the prior:", flush=True)
                    u.report_random_model_summary(drawn_deltas)
                    if len(held_deltas) > 0:
                        print("  models holding "
                              f"{', '.join(nmodels_hold)} at the fiducial:",
                              flush=True)
                        u.report_random_model_summary(held_deltas)
                # drawn before the limit applies, so a failing run still
                # shows where in the prior its deltas sit; an advisory
                # check draws the 0.2 band of the reference tests
                if nmodels_limit is not None:
                    figure_limit = nmodels_limit
                else:
                    figure_limit = u.CHI2_TOLERANCE
                figure_path = _nmodels_figure(
                    u, example=nm_example, points=points, deltas=deltas,
                    figure_dir=figure_dir, seed=seed,
                    limit=figure_limit, held=held, case=nm_label)
                print(f"  N-random-models figure: {figure_path}",
                      flush=True)
                if nmodels_limit is not None:
                    # the limit applies to every model; the message
                    # names each model (and its seed) above it
                    above = [(index, delta) for index, delta
                             in enumerate(deltas)
                             if abs(delta) >= nmodels_limit]
                    self.assertFalse(
                        above,
                        msg="models with |delta chi2| >= "
                            f"{nmodels_limit}: " + ", ".join(
                                f"{index} (seed {seed + index}): "
                                f"{delta:+.4f}"
                                for index, delta in above))

        return test_ax99_nmodels

    for k, (nm_example, nm_label) in enumerate(nmodels_pairs):
        method = make_nmodels(nm_example, nm_label)
        method.__name__ = ("test_ax99_nmodels" if k == 0
                           else f"test_ax99_nmodels_{nm_example}")
        setattr(TestAccuracy, method.__name__, method)

    # rename the class last; type(name, ...) is not needed because
    # the class statement above already built it, so only the two
    # name attributes change
    TestAccuracy.__name__ = name
    TestAccuracy.__qualname__ = name
    return TestAccuracy


def fastpt_reference_suite(u, name, cases):
    """Build the FASTPT frozen-reference class (TATT via python FAST-PT).

    Arguments:
      u     = the project's cocoa_test_utils module.
      name  = the class name to build (assigned by the caller to the
              same module-level name, so test ids do not change).
      cases = one (method_name, number, example, label) tuple per
              check: the collected method name, the printed test
              number, the key of u.EXAMPLES (one that carries a
              frozen FASTPT reference), and the full report line,
              e.g. "example1 (cosmic shear, TATT+FASTPT) chi2 vs
              frozen reference".

    Returns:
      a unittest.TestCase subclass named `name`.
    """

    class TestFastptReference(unittest.TestCase):
        """The FASTPT TATT checks, sharing one frozen-state verification.

        Each check evaluates the frozen TATT point with the python
        FAST-PT package (IA_code: 1) instead of the built-in cfastpt,
        compares the chi2 with its own frozen FASTPT reference under
        the usual CHI2_TOLERANCE rule, and prints the frozen
        FASTPT-minus-CFASTPT difference next to it, so a numerics
        change in either implementation is visible at a glance.
        """

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR, verify the frozen state, load the references."""
            u.require_cocoa_environment()
            u.verify_frozen()
            cls.reference = u.load_reference()

        def _run_fastpt_case(self, number, example, label):
            """Evaluate one FASTPT TATT case against its frozen reference.

            Arguments:
              number  = the test number shown in the report header.
              example = a key of u.EXAMPLES.
              label   = one line naming the example and probe.

            Returns:
              nothing; the assertion fails the test when
              |chi2 - ref| >= CHI2_TOLERANCE.
            """
            chi2 = u.single_model_chi2(example, tatt=True, fastpt=True)
            ref = self.reference[f"{example}_tatt_fastpt"]
            cfastpt_ref = self.reference[f"{example}_tatt"]
            u.report_fastpt_test(number, label, chi2, ref,
                                 cfastpt_ref, u.CHI2_TOLERANCE)
            frozen_diff = self.reference[
                f"{example}_fastpt_minus_cfastpt"]
            # the + in :+.6f forces a sign, so the direction of the
            # frozen difference always shows
            print("  frozen FASTPT - CFASTPT difference = "
                  f"{frozen_diff:+.6f}", flush=True)
            self.assertLess(
                abs(chi2 - ref), u.CHI2_TOLERANCE,
                msg=f"FASTPT chi2 = {chi2:.6f} vs frozen reference "
                    f"{ref:.6f} (|delta| >= {u.CHI2_TOLERANCE})")

    def make_case(method_name, number, example, label):
        def test(self):
            self._run_fastpt_case(number, example, label)
        test.__name__ = method_name
        test.__doc__ = (f"{example} TATT with python FAST-PT stays "
                        "on its reference.\n        ")
        return test

    for row in cases:
        method = make_case(*row)
        setattr(TestFastptReference, method.__name__, method)
    TestFastptReference.__name__ = name
    TestFastptReference.__qualname__ = name
    return TestFastptReference


def fastpt_sweep_suite(u, name, cases):
    """Build the CFASTPT-vs-FASTPT sweep class.

    Arguments:
      u     = the project's cocoa_test_utils module.
      name  = the class name to build (assigned by the caller to the
              same module-level name, so test ids do not change).
      cases = one (method_name, number, example, label) tuple per
              sweep: the collected method name, the printed test
              number, the key of u.EXAMPLES, and the opening of the
              report line up to the IA model, e.g. "example2 (3x2pt".

    Returns:
      a unittest.TestCase subclass named `name`.
    """

    class TestFastptSweep(unittest.TestCase):
        """The direct CFASTPT-vs-FASTPT comparison, one sweep per probe.

        Each sweep evaluates the project's hard-coded points across
        the intrinsic-alignment prior (u.FASTPT_COMPARISON_POINTS,
        cosmology fixed at the frozen fiducial) three times: with
        cfastpt, with FASTPT at the converged two-grid defaults, and
        with FASTPT at the doubled boosts. The CFASTPT vector is the
        fiducial of each point, so the pass rule is the chi2 of the
        FASTPT(low) vector against it (delta^T C^-1 delta), under
        u.FASTPT_COMPARISON_TOLERANCE; FASTPT(high)'s deviation
        prints as the advisory grid response. The sweeps read the
        --high and --mask options (conftest.py). setUpClass loads no
        reference chi2: these tests compare the two implementations
        against each other, so the frozen state only supplies the
        configuration and the data files.
        """

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR and verify the frozen state, once per class."""
            u.require_cocoa_environment()
            u.verify_frozen()

    def make_sweep(method_name, number, example, label):
        def test(self):
            # the conftest copies the --high command line option into
            # this variable; .get with the "0" default keeps a run
            # outside pytest on the default settings unless the
            # variable is exported by hand
            high = os.environ.get("COCOA_FASTPT_HIGH", "0") == "1"
            setting = "high accuracy" if high else "default settings"
            mask = os.environ.get("COCOA_FASTPT_MASK", "frozen")
            (chi2_cfastpt, chi2_fastpt_low, chi2_fastpt_high,
             dchi2_low, dchi2_high) = u.cfastpt_vs_fastpt_chi2s(
                example, high=high, mask=mask)
            largest = u.report_fastpt_comparison(
                number,
                f"{label}, TATT, camb/cosmolike {setting}, "
                f"mask {mask}): CFASTPT vs FASTPT at 30 "
                "hard-coded points",
                chi2_cfastpt, chi2_fastpt_low, chi2_fastpt_high,
                dchi2_low, dchi2_high, u.FASTPT_COMPARISON_TOLERANCE)
            self.assertLess(
                largest, u.FASTPT_COMPARISON_TOLERANCE,
                msg="max chi2 of the FASTPT(low)-vs-CFASTPT "
                    f"data-vector difference = {largest:.6f} over "
                    f"the comparison points ({setting}); limit "
                    f"{u.FASTPT_COMPARISON_TOLERANCE}")
        test.__name__ = method_name
        test.__doc__ = (f"{example}: cfastpt and FASTPT agree at the "
                        "30 IA points.\n        ")
        return test

    for row in cases:
        method = make_sweep(*row)
        setattr(TestFastptSweep, method.__name__, method)
    TestFastptSweep.__name__ = name
    TestFastptSweep.__qualname__ = name
    return TestFastptSweep


def ee2_suite(u, name, race, original=None):
    """Build the EuclidEmulator2 class (race check, maybe original).

    Arguments:
      u        = the project's cocoa_test_utils module.
      name     = the class name to build (assigned by the caller to
                 the same module-level name, so test ids do not
                 change).
      race     = a (method_name, number, example, label) tuple for
                 the EE2 race check, label up to " race check", e.g.
                 "example1 (cosmic shear, NLA+EE2)".
      original = a (method_name, number, example) tuple enabling the
                 comparison of the installed, modified EE2 against
                 the pre-modification build (u.EE2_ORIGINAL_COMMIT),
                 or None for a project without that check.

    Returns:
      a unittest.TestCase subclass named `name`.
    """

    class TestEE2(unittest.TestCase):
        """The EuclidEmulator2 checks, sharing one frozen-state verification.

        setUpClass loads no reference chi2: the original-vs-modified
        check compares the two EE2 builds against each other and the
        race check compares one build against itself, so the frozen
        state only supplies the configuration and the data files.
        """

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR and verify the frozen state, once per class."""
            u.require_cocoa_environment()
            u.verify_frozen()

    race_name, race_number, race_example, race_label = race

    def race_test(self):
        """The fiducial with EE2 as 10th of 10 matches a fresh run.

        EE2's OpenMP-threaded compute runs inside every evaluation
        of the row, so a thread race or leaked state in it moves
        the second fiducial value.
        """
        u.assert_omp_threads()
        fresh, tenth = u.ten_in_a_row_chi2(race_example, tatt=False,
                                           ee2=True)
        u.report_race_test(
            race_number,
            f"{race_label} race check: 10 cosmologies in a row",
            fresh, tenth, u.RACE_TOLERANCE)
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg=f"10th-in-a-row chi2 = {tenth:.8f} vs fresh "
                f"{fresh:.8f}")

    race_test.__name__ = race_name
    setattr(TestEE2, race_name, race_test)

    if original is not None:
        orig_name, orig_number, orig_example = original

        def original_test(self):
            """The modified EE2 reproduces the original's data vectors.

            The per-cosmology delta chi2 of the pre-modification
            build against the installed one, at the ten shared
            cosmologies, must stay below CHI2_TOLERANCE (0.2), the
            band of the reference tests.
            """
            try:
                dchi2s = u.ee2_original_vs_cocoa_dchi2s(orig_example)
            except RuntimeError as error:
                # a shallow local clone cannot supply the original
                # commit; that is an installation limitation, not an
                # EE2 regression
                self.skipTest(str(error))
            print(f"\n==== test {orig_number}: modified EE2 vs "
                  f"original EE2 ({orig_example}, NLA, 10 "
                  "cosmologies) ====", flush=True)
            for i, d in enumerate(dchi2s, start=1):
                print(f"  cosmology {i:2d}: dchi2 = {d:.8f}",
                      flush=True)
            largest = max(dchi2s)
            print(f"  max dchi2 = {largest:.8f}   (limit: < "
                  f"{u.CHI2_TOLERANCE})", flush=True)
            self.assertLess(
                largest, u.CHI2_TOLERANCE,
                msg=f"max dchi2 of the original-vs-modified EE2 data "
                    f"vectors = {largest:.8f} over the ten "
                    f"cosmologies (limit {u.CHI2_TOLERANCE})")

        original_test.__name__ = orig_name
        setattr(TestEE2, orig_name, original_test)

    TestEE2.__name__ = name
    TestEE2.__qualname__ = name
    return TestEE2


def nonlinear_suite(u, name, cases):
    """Build the Halofit-vs-EuclidEmulator2 advisory class.

    Arguments:
      u     = the project's cocoa_test_utils module.
      name  = the class name to build (assigned by the caller to the
              same module-level name, so test ids do not change).
      cases = one (method_name, advisory, example, label) tuple per
              check: the collected method name, the printed advisory
              label (e.g. "NL1"), the key of u.EXAMPLES, and the
              opening of the report line up to the IA model, e.g.
              "example1 (cosmic shear".

    Returns:
      a unittest.TestCase subclass named `name`.
    """

    class TestNonlinear(unittest.TestCase):
        """Halofit scored against EE2, one advisory check per probe.

        Each check evaluates the probe's data vector with both
        nonlinear-P(k) sources at the ten fixed cosmologies of
        u.NONLINEAR_COMPARISON_POINTS and reports, per cosmology, the
        chi2 of the Halofit vector against the EE2 vector (the EE2
        vector is that cosmology's fiducial, so the baseline is zero
        by construction). Advisory: the printed report is the
        product, and the only assertion is structural. The checks
        read the --mask option (conftest.py). setUpClass loads no
        reference chi2: the frozen state only supplies the
        configuration and the data files.
        """

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR and verify the frozen state, once per class."""
            u.require_cocoa_environment()
            u.verify_frozen()

    def make_check(method_name, advisory, example, label):
        def test(self):
            mask = os.environ.get("COCOA_FASTPT_MASK", "frozen")
            dchi2s = u.halofit_vs_ee2_dchi2s(example, mask=mask)
            u.report_nonlinear_comparison(
                f"{advisory}: {label}, NLA, mask {mask}): HALOFIT "
                "vs EE2 at 10 fixed cosmologies", dchi2s)
            self.assertEqual(len(dchi2s),
                             len(u.NONLINEAR_COMPARISON_POINTS))
        test.__name__ = method_name
        test.__doc__ = (f"{advisory}: Halofit scored against EE2 at "
                        "ten cosmologies, advisory.\n        ")
        return test

    for row in cases:
        method = make_check(*row)
        setattr(TestNonlinear, method.__name__, method)
    TestNonlinear.__name__ = name
    TestNonlinear.__qualname__ = name
    return TestNonlinear


def baryon_accuracy_suite(u):
    """Build the BF0-BF7 baryonic-feedback accuracy class for a project.

    Arguments:
      u = the project's cocoa_test_utils module. The class reads its
          BARYON_METHODS labels, ACCURACY_KNOBS table and
          baryon_accuracy_delta through it, so the returned class is
          bound to that project's frozen state and tables.

    Returns:
      a unittest.TestCase subclass; the caller assigns it to the name
      TestBaryonAccuracyAdvisory so pytest collects it with the same
      test ids as before the factory existed.
    """

    class TestBaryonAccuracyAdvisory(unittest.TestCase):
        """Advisory checks BF0-BF7: accuracy with baryonic feedback on.

        Each check evaluates this project's example1 configuration
        (cosmic shear, NLA) with the bfmt theory block computing one
        feedback method, and reports

            delta chi2 = chi2(high accuracy) - chi2(default)

        against a vector the default-settings model generated on the
        fly (so the default chi2 is zero by construction and nothing
        is stored in frozen/). The delta is advisory: only a
        non-finite value fails. BF0 additionally runs the
        one-knob-at-a-time scan with the Akino SP(k) method on, so a
        large delta names the knob causing it.

        setUpClass runs once: it moves to ROOTDIR and verifies every
        frozen file against the SHA-256 manifest before any physics
        runs.
        """

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR and verify the frozen state, once per class."""
            u.require_cocoa_environment()
            u.verify_frozen()

        def _baryon_accuracy_check(self, name, baryon, label):
            """Print the default-vs-high-accuracy delta with one method on.

            Arguments:
              name   = the advisory label (BF1-BF7) for the report.
              baryon = a label of cocoa_test_utils.BARYON_METHODS.
              label  = one line naming the feedback method.

            Returns:
              nothing; only a non-finite delta fails the check.
            """
            # the default chi2 is zero by construction (the default
            # model produced the very vector it is compared with), so
            # the pushed evaluation's chi2 is the delta; only that is
            # printed
            delta = u.baryon_accuracy_delta(baryon)
            # delta == delta is False only for NaN (NaN compares
            # unequal to itself); with the infinity check, the
            # assertion demands a finite delta
            self.assertTrue(
                delta == delta and abs(delta) != float("inf"),
                f"{name}: non-finite delta")
            print(f"""
{'-' * 66}
ACCURACY: {name}: {label}
  delta chi2 (high-default) = {delta:+.6f}
{'-' * 66}""", flush=True)

        def test_bf0_one_knob_at_a_time(self):
            """BF0: each accuracy knob alone, Akino SP(k) feedback on.

            Advisory: the same K-scan as test_accuracy.py, with the
            bfmt block computing the Akino SP(k) suppression and the
            chi2 measured against that method's own generated vector.
            A knob whose delta rivals the all-knobs delta of BF2 is
            the driver of the numerical error under feedback.
            """
            print("", flush=True)
            # each entry is (label, likelihood overrides, camb
            # overrides); the two _ discard the tables, which
            # baryon_accuracy_delta looks up again by label
            for label, _, _ in u.ACCURACY_KNOBS:
                # each knob's chi2 against the on-the-fly vector is
                # its delta (the default against that vector is zero)
                delta = u.baryon_accuracy_delta("spk akino", knob=label)
                print(f"  KNOB {label:30s} delta chi2 = {delta:+12.6f}",
                      flush=True)

        def test_bf1_spk_power_law(self):
            """BF1: SP(k) with the power-law fb relation."""
            self._baryon_accuracy_check("BF1", "spk power law",
                                        "SP(k), power-law fb relation")

        def test_bf2_spk_akino(self):
            """BF2: SP(k) with the Akino et al. 2022 fb relation."""
            self._baryon_accuracy_check("BF2", "spk akino",
                                        "SP(k), Akino et al. 2022")

        def test_bf3_spk_double_power_law(self):
            """BF3: SP(k) with the double power-law fb relation."""
            self._baryon_accuracy_check("BF3", "spk double power law",
                                        "SP(k), double power-law fb relation")

        def test_bf4_bcemu(self):
            """BF4: BCEmu."""
            self._baryon_accuracy_check("BF4", "bcemu", "BCEmu")

        def test_bf5_flamingo(self):
            """BF5: FlamingoBaryonResponseEmulator."""
            self._baryon_accuracy_check("BF5", "flamingo",
                                        "FlamingoBaryonResponseEmulator")

        def test_bf6_baccoemu(self):
            """BF6: BACCOemu."""
            self._baryon_accuracy_check("BF6", "baccoemu", "BACCOemu")

        def test_bf7_bcemu2025(self):
            """BF7: BCemu2025."""
            self._baryon_accuracy_check("BF7", "bcemu2025", "BCemu2025")

    return TestBaryonAccuracyAdvisory


def baryon_drift_suite(u):
    """Build the BD1-BD7 baryonic-feedback drift class for a project.

    Arguments:
      u = the project's cocoa_test_utils module. The class reads its
          BARYON_METHODS labels, CHI2_TOLERANCE and baryon_drift_chi2
          through it, so the returned class is bound to that
          project's frozen state and tables.

    Returns:
      a unittest.TestCase subclass; the caller assigns it to the name
      TestBaryonDrift so pytest collects it with the same test ids as
      before the factory existed.
    """

    class TestBaryonDrift(unittest.TestCase):
        """Drift tests BD1-BD7: the feedback pipeline against its freeze.

        Each test evaluates this project's example1 configuration
        (cosmic shear, NLA) with the bfmt theory block computing one
        feedback method, against that method's frozen data vector
        (written at freeze time by generate_frozen_reference.py
        --baryons, when the chi2 against it was zero by
        construction). The assertion chi2 <= CHI2_TOLERANCE therefore
        pins the whole feedback pipeline: a failure means cosmolike
        or the bfmt theory block changed its prediction since the
        freeze. The accuracy checks (BF) regenerate their vector per
        run and can never see drift; these tests hold the frozen
        vector still, so they measure drift and nothing else.

        setUpClass runs once: it moves to ROOTDIR and verifies every
        frozen file against the SHA-256 manifest before any physics
        runs.
        """

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR and verify the frozen state, once per class."""
            u.require_cocoa_environment()
            u.verify_frozen()

        def _baryon_drift_check(self, name, baryon, label):
            """Check one method's chi2 against its frozen feedback vector.

            Arguments:
              name   = the test label (BD1-BD7) for the report.
              baryon = a label of cocoa_test_utils.BARYON_METHODS.
              label  = one line naming the feedback method.

            Returns:
              nothing; the assertion fails when the chi2 exceeds
              CHI2_TOLERANCE.
            """
            chi2 = u.baryon_drift_chi2(baryon)
            print(f"""
{'-' * 66}
DRIFT: {name}: {label}
  chi2 against the frozen feedback vector = {chi2:.6f}
  (zero at freeze time; tolerance {u.CHI2_TOLERANCE})
{'-' * 66}""", flush=True)
            self.assertLessEqual(
                chi2, u.CHI2_TOLERANCE,
                f"{name}: chi2 {chi2:.6f} exceeds the tolerance "
                f"{u.CHI2_TOLERANCE}; cosmolike or the bfmt theory "
                "block changed its prediction since the freeze")

        def test_bd1_spk_power_law(self):
            """BD1: SP(k) with the power-law fb relation."""
            self._baryon_drift_check("BD1", "spk power law",
                                     "SP(k), power-law fb relation")

        def test_bd2_spk_akino(self):
            """BD2: SP(k) with the Akino et al. 2022 fb relation."""
            self._baryon_drift_check("BD2", "spk akino",
                                     "SP(k), Akino et al. 2022")

        def test_bd3_spk_double_power_law(self):
            """BD3: SP(k) with the double power-law fb relation."""
            self._baryon_drift_check("BD3", "spk double power law",
                                     "SP(k), double power-law fb relation")

        def test_bd4_bcemu(self):
            """BD4: BCEmu."""
            self._baryon_drift_check("BD4", "bcemu", "BCEmu")

        def test_bd5_flamingo(self):
            """BD5: FlamingoBaryonResponseEmulator."""
            self._baryon_drift_check("BD5", "flamingo",
                                     "FlamingoBaryonResponseEmulator")

        def test_bd6_baccoemu(self):
            """BD6: BACCOemu."""
            self._baryon_drift_check("BD6", "baccoemu", "BACCOemu")

        def test_bd7_bcemu2025(self):
            """BD7: BCemu2025."""
            self._baryon_drift_check("BD7", "bcemu2025", "BCemu2025")

    return TestBaryonDrift


# ---- the interface-level families (in-process cache checks) -----------------

# The photo-z convention sweep is built inside the factory from the
# project's default pair (interp_default, zmid_default): the default,
# each alternative stage-1 interpolant, the other z-column reading,
# and the default again (the round trip).


def photoz_conventions_suite(u, name, example="example1",
                             interp_default=0, zmid_default=0):
    """Build the runtime photo-z convention sweep class.

    The likelihood exposes two runtime knobs that control how the
    n(z) table files are turned into the smooth distributions the
    Limber integrals consume: photoz_interpolation_type (the stage-1
    interpolant) and photoz_zmid_convention (how the z column of the
    n(z) file is read). The class evaluates the frozen cosmic-shear
    fiducial under the five settings of its sweep in one process, measures
    each alternative's delta chi2 = delta^T C^-1 delta against the
    default vector, and asserts the dead-flag floors, the frozen
    reference, and the bitwise round trip back to the default.
    Running everything in one process is the point: the n(z) caches
    inside the compiled interface must notice a runtime flag change
    and rebuild.

    Arguments:
      u       = the project's cocoa_test_utils module.
      name    = the class name to build (assigned by the caller to
                the same module-level name, so test ids do not
                change).
      example = the key of u.EXAMPLES the sweep evaluates.
      interp_default = the project's photoz_interpolation_type
                default (0 cubic spline, 1 linear, 2+ Steffen).
      zmid_default = the project's photoz_zmid_convention default
                (0: the z column holds left bin edges, 1: it holds
                the sample points). The frozen references carry the
                project defaults, so the sweep's baseline must sit
                there, not at a universal (0, 0).

    Returns:
      a unittest.TestCase subclass named `name`.
    """

    # The sweep: the default pair, the two alternative stage-1
    # interpolants at the default z reading, the other z reading at
    # the default interpolant, and the default pair again (the round
    # trip).
    interp_names = {0: "cspline", 1: "linear", 2: "steffen"}
    zmid_names = {0: "Z_LOW", 1: "Z_MID"}
    default_tag = (f"{interp_names[interp_default]}/"
                   f"{zmid_names[zmid_default]} (default)")
    alternatives = [(interp_names[i], i, zmid_default)
                    for i in sorted(interp_names)
                    if i != interp_default]
    alternatives.append((zmid_names[1 - zmid_default], interp_default,
                         1 - zmid_default))
    settings = ((default_tag, interp_default, zmid_default),
                *alternatives,
                ("default again (round trip)", interp_default,
                 zmid_default))
    # The alternatives must be seen (a stale n(z) cache would give
    # exactly zero); the floors are orders of magnitude below the
    # measured deltas, so they only catch a dead flag, never normal
    # numerical drift. The z-reading flip shifts every distribution
    # by dz/2, hence its larger floor.
    floors = {tag: (1.0e-2 if tag in zmid_names.values() else 1.0e-8)
              for tag, _, _ in alternatives}

    class TestPhotozConventions(unittest.TestCase):
        """The five-setting sweep, sharing one frozen-state verification."""

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR, verify the frozen state, load the references."""
            u.require_cocoa_environment()
            u.verify_frozen()
            cls.reference = u.load_reference()

        def test_photoz_conventions(self):
            """Every alternative is seen, and the default round trip is exact."""
            import shutil
            import tempfile

            import numpy as np
            ci = _interface(u)

            vectors_dir = tempfile.mkdtemp(prefix="photoz_conventions_")
            # addCleanup registers a call unittest makes after the
            # test, pass or fail: here it removes the temporary folder
            self.addCleanup(shutil.rmtree, vectors_dir,
                            ignore_errors=True)

            results = {}
            icov = None
            for tag, interp, zmid in settings:
                print(f"  building model ({example}, NLA, {tag}) ...",
                      flush=True)
                info = u.load_frozen_info(example, tatt=False)
                likelihood_block = \
                    info["likelihood"][u.EXAMPLES[example]["likelihood"]]
                likelihood_block["photoz_interpolation_type"] = interp
                likelihood_block["photoz_zmid_convention"] = zmid
                vector_path = os.path.join(
                    vectors_dir, f"dv_{interp}_{zmid}.modelvector")
                likelihood_block["print_datavector"] = True
                likelihood_block["print_datavector_file"] = vector_path
                model = u.make_model(info)
                point = u.build_point(model, example, tatt=False)
                chi2 = u.evaluate_chi2(model, point)
                if not os.path.isfile(vector_path):
                    raise RuntimeError(
                        f"print_datavector wrote no file at "
                        f"{vector_path}")
                results[tag] = (chi2, u._load_datavector(vector_path))
                if icov is None:
                    # all five settings share one dataset (mask,
                    # covariance), so the masked inverse covariance of
                    # the first build serves every comparison
                    icov = np.array(ci.get_inv_cov_masked())

            chi2_default, dv_default = results[settings[0][0]]
            reference_key = f"{example}_nla"

            print(f"\n  chi2 report ({example}, NLA):")
            print(f"    default: chi2 = {chi2_default:.6f} "
                  f"(frozen reference "
                  f"{self.reference[reference_key]:.6f})")
            for tag, _, _ in alternatives:
                chi2, dv = results[tag]
                delta = dv - dv_default
                dchi2 = float(delta @ icov @ delta)
                print(f"    {tag:8s}: chi2 = {chi2:.6f}, "
                      f"delta^T C^-1 delta vs default = {dchi2:.6e}")
                self.assertGreater(
                    dchi2, floors[tag],
                    f"{tag}: the flag change was not seen by the n(z) "
                    "cache (delta chi2 at or below the dead-flag "
                    "floor)")

            # the default must agree with the frozen reference within
            # the band the reference tests demand
            self.assertLess(
                abs(chi2_default - self.reference[reference_key]),
                u.CHI2_TOLERANCE)

            # round trip: after all the flips, the default settings
            # must reproduce the first vector identically: the cache
            # rebuilt back to the same state
            _, dv_return = results[settings[-1][0]]
            self.assertTrue(
                np.array_equal(dv_return, dv_default),
                "returning to the default settings did not reproduce "
                "the default data vector bit for bit; the n(z) cache "
                "did not rebuild cleanly on the way back")

    TestPhotozConventions.__name__ = name
    TestPhotozConventions.__qualname__ = name
    return TestPhotozConventions


def nonlimber_suite(u, name, kind, measured, default=0, rtol=0.05,
                    floor=1.0e-6, example="example2"):
    """Build one Limber-vs-non-Limber class (clustering or ggl).

    The likelihood key adopt_limber_gg (clustering) or adopt_limber_gs
    (galaxy-galaxy lensing) chooses between the exact low-l projection
    and the Limber approximation at every multipole. The class
    evaluates the frozen fiducial three times in one process (the
    default, the other setting, the default again) and asserts: the
    delta is seen (dead-flag floor), only the probe's own block
    changes, the round trip reproduces the first vector bitwise, the
    delta chi2 matches the project's measured value to the relative
    band, and the default reproduces the frozen reference (last, so a
    stale snapshot cannot hide the other checks).

    Arguments:
      u        = the project's cocoa_test_utils module.
      name     = the class name to build (assigned by the caller to
                 the same module-level name, so test ids do not
                 change).
      kind     = "gg" (clustering, adopt_limber_gg, the block after
                 cosmic shear and ggl) or "ggl" (galaxy-galaxy
                 lensing, adopt_limber_gs, the block after cosmic
                 shear).
      measured = the delta chi2 measured for this project's frozen
                 fiducial (survey data).
      default  = the project's adopt_limber_* default (0 non-Limber,
                 1 Limber).
      rtol     = the relative band around `measured`.
      floor    = the dead-flag floor on the delta chi2.
      example  = the key of u.EXAMPLES the sweep evaluates.

    Returns:
      a unittest.TestCase subclass named `name`.
    """
    flag_key = {"gg": "adopt_limber_gg", "ggl": "adopt_limber_gs"}[kind]
    probe = {"gg": "clustering", "ggl": "ggl"}[kind]

    # (report tag, flag value): the default, the other setting, the
    # default again
    names = {0: "non-Limber", 1: "Limber"}
    settings = (
        (f"{names[default]} (default)", default),
        (names[1 - default], 1 - default),
        (f"{names[default]} again (round trip)", default),
    )

    class TestNonLimber(unittest.TestCase):
        """Limber vs non-Limber on the frozen fiducial."""

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR, verify the frozen state, load the references."""
            u.require_cocoa_environment()
            u.verify_frozen()
            cls.reference = u.load_reference()

        def _run_sweep(self):
            """Evaluate the three settings and assert the five checks."""
            import time

            import numpy as np
            ci = _interface(u)
            masked_vector = _masked_vector_fn(u, ci)

            reference_key = f"{example}_nla"
            vectors = {}
            chi2s = {}
            icov = None
            sizes = None
            nlen = None
            pairs = None
            for tag, flag in settings:
                print(f"  building model ({example}, NLA, {tag}) ...",
                      flush=True)
                info = u.load_frozen_info(example, tatt=False)
                lk_name = u.EXAMPLES[example]["likelihood"]
                info["likelihood"][lk_name][flag_key] = flag
                model = u.make_model(info)
                point = u.build_point(model, example, tatt=False)
                start = time.perf_counter()
                chi2s[tag] = u.evaluate_chi2(model, point)
                elapsed = time.perf_counter() - start
                print(f"    chi2 = {chi2s[tag]:.6f}  ({elapsed:.1f} s, "
                      "first evaluation of the model)", flush=True)
                # full-precision model vector (the printed one keeps
                # 9 digits)
                vectors[tag] = np.array(masked_vector())
                if icov is None:
                    # all settings share one data set (mask,
                    # covariance), so the first build's masked inverse
                    # covariance serves all
                    icov = np.array(ci.get_inv_cov_masked())
                    like = model.likelihood[lk_name]
                    if _native_space(u, ci) == "real":
                        sizes = ci.compute_data_vector_3x2pt_real_sizes()
                        nlen = int(like.ntheta)
                    else:
                        sizes = \
                            ci.compute_data_vector_3x2pt_fourier_sizes()
                        nlen = _fourier_nlen(like)
                    # clustering physics may extend past the three
                    # probe blocks (an appended gg-cross block)
                    ggx = _ggx_len(ci)
                    if kind == "ggl":
                        # the kept (lens, source) pairs, lens bin
                        # outer, for the per-pair report labels
                        excluded = _ggl_excluded(like)
                        pairs = [(zl, zs)
                                 for zl in range(int(like.lens_ntomo))
                                 for zs in range(int(like.source_ntomo))
                                 if (zl, zs) not in excluded]

            # tags = {flag value: report tag} of the first two
            # settings, so tags[0] names the non-Limber run and
            # tags[1] the Limber one (the delta enters quadratically,
            # so its sign never matters)
            tags = {flag: tag for tag, flag in settings[:2]}
            dv_default = vectors[settings[0][0]]
            delta = vectors[tags[0]] - vectors[tags[1]]
            dchi2 = float(delta @ icov @ delta)

            if kind == "gg":
                # the clustering block follows cosmic shear and
                # galaxy-galaxy lensing in every probe combination:
                # entries [b0, b1), one block of nlen entries per
                # spectrum. An appended gg-cross block is clustering
                # physics, so the window spans it too.
                b0 = int(sizes[0]) + int(sizes[1])
                b1 = b0 + int(sizes[2]) + ggx
                nblocks = (int(sizes[2]) + ggx) // nlen
                block_label = "lens bin" if ggx == 0 else "spectrum"
                per_block_note = (
                    "per lens bin (the bin's block alone):"
                    if ggx == 0 else
                    "per clustering spectrum (the spectrum's block "
                    "alone):")
            else:
                # the ggl block follows cosmic shear: entries
                # [b0, b1), one block of nlen entries per kept
                # (lens, source) pair
                b0 = int(sizes[0])
                b1 = b0 + int(sizes[1])
                nblocks = int(sizes[1]) // nlen
                block_label = "pair"
                per_block_note = ("per lens-source pair (the pair's "
                                  "block alone):")

            print(f"\n  delta chi2 report ({example}, NLA):")
            print(f"    {settings[0][0]}: chi2 = "
                  f"{chi2s[settings[0][0]]:.6f} "
                  f"(frozen reference "
                  f"{self.reference[reference_key]:.6f})")
            print(f"    {settings[1][0]}: chi2 = "
                  f"{chi2s[settings[1][0]]:.6f}")
            print(f"    delta^T C^-1 delta = {dchi2:.4f} "
                  f"(measured {measured:.4f})")
            print(f"    {per_block_note}")
            # per-block contributions, printed largest first; the
            # loop stops at the first one below 0.1% of the total
            rows = []
            for b in range(nblocks):
                block = np.zeros_like(delta)
                sl = slice(b0 + b*nlen, b0 + (b + 1)*nlen)
                block[sl] = delta[sl]
                rows.append((float(block @ icov @ block), b))
            for contribution, b in sorted(rows, reverse=True):
                if contribution < 1.0e-3*max(dchi2, floor):
                    break
                if kind == "ggl" and pairs is not None \
                        and len(pairs) == nblocks:
                    label = f"(lens {pairs[b][0]}, source {pairs[b][1]})"
                    print(f"      {label:24s} {contribution:.4f}")
                else:
                    print(f"      {block_label} {b:<14d} "
                          f"{contribution:.4f}")

            self.assertGreater(
                dchi2, floor,
                f"non-Limber {probe} did not change the data vector: "
                f"the {flag_key} flag did not reach the C code, or "
                f"the {probe} cache did not rebuild")

            outside = np.concatenate((delta[:b0], delta[b1:]))
            self.assertTrue(
                np.all(outside == 0.0),
                f"entries outside the {probe} block changed with "
                f"{flag_key}")

            self.assertTrue(
                np.array_equal(vectors[settings[-1][0]], dv_default),
                "returning to the default did not reproduce the first "
                f"data vector bit for bit; the {probe} cache did not "
                "rebuild cleanly")

            self.assertLess(
                abs(dchi2/measured - 1.0), rtol,
                f"delta chi2 = {dchi2:.4f} differs from the measured "
                f"{measured:.4f} by more than {rtol:.0%}")

            # last, so that a stale frozen snapshot does not hide the
            # four checks above
            self.assertLess(
                abs(chi2s[settings[0][0]]
                    - self.reference[reference_key]),
                u.CHI2_TOLERANCE,
                f"{settings[0][0]}: chi2 = "
                f"{chi2s[settings[0][0]]:.6f} vs frozen reference "
                f"{self.reference[reference_key]:.6f}")

    def test(self):
        self._run_sweep()
    test.__name__ = f"test_nonlimber_{kind}"
    test.__doc__ = (f"Limber vs non-Limber {probe}: the five checks "
                    "of the factory docstring.\n        ")
    setattr(TestNonLimber, test.__name__, test)
    TestNonLimber.__name__ = name
    TestNonLimber.__qualname__ = name
    return TestNonLimber


def lowrank_gg_suite(u, name, measured, rtol=0.10, floor=1.0e-8,
                     example="example2", default=0, overrides=None):
    """Build one FKEM-vs-low-rank-SVD clustering class.

    The likelihood key nonlimber_lowrank_gg picks the non-Limber
    clustering engine: 0 = FKEM (every project's default), 1 = the
    low-rank SVD thin-bin engine (C_cl_tomo_lowrank in cosmo2D.c).
    The delta measures FKEM's approximations - its Limber treatment
    of the nonlinear density term, and its multipole cap
    lmax_nonlimber - against the SVD's rank truncation. Its size
    depends on the bins: small for photometric ones, large for thin
    spectroscopic ones unless FKEM's cap covers every band.

    The class evaluates the frozen fiducial three times in one
    process (the default engine, the other one, the default again)
    and asserts:

      1. the engine delta chi2 is above a dead-flag floor (the flag
         reached the C code and the clustering cache rebuilt);
      2. only clustering entries change;
      3. the round trip reproduces the first vector bitwise;
      4. the delta chi2 matches the project's measured value to the
         relative band;
      5. the default engine reproduces the frozen reference chi2
         (last, so a stale snapshot cannot hide checks 1-4).

    It also reports each engine's steady-state runtime (one second
    evaluation per model; informational, never asserted).

    Arguments:
      u        = the project's cocoa_test_utils module.
      name     = the class name to build (assigned by the caller to
                 the same module-level name, so test ids do not
                 change).
      measured = the engine delta chi2 measured for this project's
                 frozen fiducial.
      rtol     = the relative band around `measured`.
      floor    = the dead-flag floor on the delta chi2.
      example  = the key of u.EXAMPLES the sweep evaluates.
      default  = the project's nonlimber_lowrank_gg default (0 = FKEM,
                 1 = the low-rank SVD engine): the sweep starts and
                 ends on it, with the other engine in between.
      overrides = optional {likelihood key: value} applied to all
                 three builds, e.g. {"lmax_nonlimber": 550} for an
                 FKEM cap that covers every band. Each engine ignores
                 the other's keys, so one dictionary serves both; with
                 None every build keeps the frozen configuration.

    Returns:
      a unittest.TestCase subclass named `name`.
    """
    names = {0: "FKEM", 1: "low-rank SVD"}
    settings = (
        (f"{names[default]} (default)", default),
        (names[1 - default], 1 - default),
        (f"{names[default]} again (round trip)", default),
    )

    class TestLowrank(unittest.TestCase):
        """FKEM vs low-rank SVD clustering on the frozen fiducial."""

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR, verify the frozen state, load the references."""
            u.require_cocoa_environment()
            u.verify_frozen()
            cls.reference = u.load_reference()

        def _run_sweep(self):
            """Evaluate the three settings and assert the five checks."""
            import time

            import numpy as np
            ci = _interface(u)
            masked_vector = _masked_vector_fn(u, ci)

            reference_key = f"{example}_nla"
            vectors = {}
            chi2s = {}
            steady = {}
            icov = None
            sizes = None
            for tag, flag in settings:
                print(f"  building model ({example}, NLA, {tag}) ...",
                      flush=True)
                info = u.load_frozen_info(example, tatt=False)
                lk_name = u.EXAMPLES[example]["likelihood"]
                info["likelihood"][lk_name]["nonlimber_lowrank_gg"] = \
                    flag
                # the same overrides in every build, so the engine
                # flag is the only difference between them
                for key, value in (overrides or {}).items():
                    info["likelihood"][lk_name][key] = value
                model = u.make_model(info)
                point = u.build_point(model, example, tatt=False)
                chi2s[tag] = u.evaluate_chi2(model, point)
                start = time.perf_counter()
                u.evaluate_chi2(model, point)
                steady[tag] = time.perf_counter() - start
                print(f"    chi2 = {chi2s[tag]:.6f}  (steady-state "
                      f"evaluation {steady[tag]:.2f} s)", flush=True)
                vectors[tag] = \
                    np.array(masked_vector())
                if icov is None:
                    # all settings share one data set (mask,
                    # covariance), so the first build's masked
                    # inverse covariance serves all
                    icov = np.array(ci.get_inv_cov_masked())
                    if _native_space(u, ci) == "real":
                        sizes = ci.compute_data_vector_3x2pt_real_sizes()
                    else:
                        sizes = \
                            ci.compute_data_vector_3x2pt_fourier_sizes()
                    # an appended gg-cross block is clustering too
                    ggx = _ggx_len(ci)

            dv_default = vectors[settings[0][0]]
            delta = vectors[settings[1][0]] - dv_default
            dchi2 = float(delta @ icov @ delta)

            # the clustering block follows cosmic shear and
            # galaxy-galaxy lensing in every probe combination; an
            # appended gg-cross block belongs to it as well
            b0 = int(sizes[0]) + int(sizes[1])
            b1 = b0 + int(sizes[2]) + ggx

            print(f"\n  engine report ({example}, NLA"
                  f"{', ' + str(overrides) if overrides else ''}):")
            print(f"    {settings[0][0]}: chi2 = "
                  f"{chi2s[settings[0][0]]:.6f} (frozen reference "
                  f"{self.reference[reference_key]:.6f}), "
                  f"{steady[settings[0][0]]:.2f} s")
            print(f"    {settings[1][0]}: chi2 = "
                  f"{chi2s[settings[1][0]]:.6f}, "
                  f"{steady[settings[1][0]]:.2f} s")
            print(f"    delta^T C^-1 delta = {dchi2:.6e} "
                  f"(measured {measured:.6e})")

            self.assertGreater(
                dchi2, floor,
                "switching the engine did not change the data vector: "
                "the nonlimber_lowrank_gg flag did not reach the C "
                "code, or the clustering cache did not rebuild")

            outside = np.concatenate((delta[:b0], delta[b1:]))
            self.assertTrue(
                np.all(outside == 0.0),
                "entries outside the clustering block changed with "
                "nonlimber_lowrank_gg")

            self.assertTrue(
                np.array_equal(vectors[settings[-1][0]], dv_default),
                "returning to the default engine did not reproduce "
                "the first data vector bit for bit; the clustering "
                "cache did not rebuild cleanly")

            self.assertLess(
                abs(dchi2/measured - 1.0), rtol,
                f"engine delta chi2 = {dchi2:.6e} differs from the "
                f"measured {measured:.6e} by more than {rtol:.0%}")

            # last, so that a stale frozen snapshot does not hide the
            # four checks above
            self.assertLess(
                abs(chi2s[settings[0][0]]
                    - self.reference[reference_key]),
                u.CHI2_TOLERANCE,
                f"{settings[0][0]}: chi2 = "
                f"{chi2s[settings[0][0]]:.6f} vs frozen reference "
                f"{self.reference[reference_key]:.6f}")

    def test(self):
        self._run_sweep()
    test.__name__ = "test_nonlimber_lowrank_gg"
    test.__doc__ = ("FKEM vs low-rank SVD clustering: the five "
                    "checks of the factory docstring.\n        ")
    setattr(TestLowrank, test.__name__, test)
    TestLowrank.__name__ = name
    TestLowrank.__qualname__ = name
    return TestLowrank


# ---- the sector-ladder cache checks -----------------------------------------

# Sector membership by sampled-parameter name. re.compile builds a
# regular-expression matcher: ^(a|b)$ matches exactly the names a or
# b, "_A1_|_A2_" matches a name containing either text, and "_M[0-9]+$"
# a name ending in _M plus digits. _ladder_sector_of returns the first
# sector whose pattern matches, so the order matters and "other" (the
# pattern "." matches any name) collects the rest. "fog" holds the
# finger-of-god velocity dispersions of spectroscopic samples. A sector
# outside the suite's phases (by default bias, fog and other) moves only
# in the scramble step.
LADDER_SECTORS = (
    ("cosmo",
     re.compile(r"^(As_1e9|H0|ns|omegab|omegam|mnu|w|w0pwa)$")),
    ("ia", re.compile(r"_A1_|_A2_|_BTA_")),
    ("dz_source", re.compile(r"_DZ_S")),
    ("dz_lens", re.compile(r"_DZ_L")),
    ("m", re.compile(r"_M[0-9]+$")),
    ("bias", re.compile(r"_B1_|_B2_|_BMAG_")),
    ("fog", re.compile(r"_SIGV_")),
    ("other", re.compile(r".")),
)

# Ladder phases (the default sector order of the forward walk; a
# project passes its own through cache_ladder_suite) and the per-step
# offsets: parameter value = fiducial + step * delta, deterministic.
# A key of LADDER_DELTAS is either an exact parameter name (a string)
# or a compiled pattern matched against the names. The offsets are
# small moves inside the priors that still change the data vector
# visibly.
LADDER_PHASES = ("cosmo", "ia", "dz_source", "dz_lens", "m")
LADDER_DELTAS = {
    "cosmo": {"omegam": 0.002, "H0": 0.2, "As_1e9": 0.02},
    "ia": {re.compile(r"_A1_1$"): 0.05, re.compile(r"_A1_2$"): 0.05,
           re.compile(r"_A2_1$"): 0.05, re.compile(r"_A2_2$"): 0.05,
           re.compile(r"_BTA_1$"): 0.05},
    "dz_source": {re.compile(r"_DZ_S"): 0.001},
    "dz_lens": {re.compile(r"_DZ_L"): 0.001},
    "m": {re.compile(r"_M[0-9]+$"): 0.005},
    "bias": {re.compile(r"_B1_"): 0.05},
    "fog": {re.compile(r"_SIGV_"): 5.0},  # km/s
}
# LADDER_NSTEP = steps per sector on the ladder; LADDER_SCRAMBLE_STEP
# = the step count of every sector at the scramble point, beyond the
# ladder's end; LADDER_RESCALE_RTOL = relative tolerance of the
# analytic shear-calibration rescale, a few roundings of a product of
# double-precision factors.
LADDER_NSTEP = 3
LADDER_SCRAMBLE_STEP = 4  # every sector at step 4, bias included
LADDER_RESCALE_RTOL = 1.0e-12


def _ladder_sector_of(name):
    """Return the sector of one sampled-parameter name.

    Arguments:
      name = a sampled-parameter name, e.g. a survey's "..._DZ_S1".

    Returns:
      the name of the first LADDER_SECTORS entry whose pattern
      matches (pat.search looks for the pattern anywhere in the
      name).
    """
    for sector, pat in LADDER_SECTORS:
        if pat.search(name):
            return sector
    return "other"


def _ladder_deltas_for(sector, names):
    """Return {parameter: per-step delta} for one sector's sampled names.

    Arguments:
      sector = a LADDER_SECTORS name.
      names  = the sampled-parameter names that belong to the sector.

    Returns:
      a dictionary with one entry per name that a LADDER_DELTAS key
      of the sector matches; names no key matches are left out.
    """
    # .get(sector, {}) returns an empty table for a sector without
    # offsets (bias moves only in the scramble, "other" never)
    table = LADDER_DELTAS.get(sector, {})
    out = {}
    for n in names:
        for key, d in table.items():
            # the condition reads: when key is a string, compare it
            # with n; otherwise key is a compiled pattern, so search
            # n for it
            if (key == n) if isinstance(key, str) else key.search(n):
                out[n] = d
                break
    return out


def cache_ladder_suite(u, name, example="example2",
                       required_sectors=("cosmo", "dz_source", "m"),
                       phases=LADDER_PHASES):
    """Build the sector-wise cache-invalidation class (the ladder).

    cosmolike keeps the result of every expensive stage in memory and
    recomputes it only when an input it depends on changed. The class
    walks a deterministic parameter ladder in one process (three
    steps per sector: cosmology, IA, source photo-z, lens photo-z,
    shear calibration; a sector the configuration samples no
    parameters in drops out), evaluating after every step, and
    asserts: every step changes the vector; each M-only step is the
    analytic (1+m_i)(1+m_j) rescale; a no-op update is bitwise
    neutral; returning to the ladder's final point after a scramble
    reproduces vector and chi2 bit for bit; and a second model
    walking the mirrored ladder lands on the same final vector. Both
    IA models run (the TATT ladder exercises the FAST-PT rebuild
    machinery NLA never touches), and every evaluation bypasses
    cobaya's cache, so the checks test cosmolike's own invalidation.

    Arguments:
      u       = the project's cocoa_test_utils module.
      name    = the class name to build (assigned by the caller to
                the same module-level name, so test ids do not
                change).
      example = the key of u.EXAMPLES the ladder evaluates (the
                flagship combination, so every sector enters its
                data vector).
      required_sectors = the sectors that must sample at least one
                parameter: the typo net (an accidentally empty
                sector would pass every later check vacuously).
                Projects that declare no source nuisance parameters
                shrink this to the sectors they sample.
      phases  = the sectors of the ladder, in forward order (the
                mirrored walk reverses it); every other sector moves
                only at the scramble point. A clustering-only project
                walks ("cosmo", "bias", "fog").

    Returns:
      a unittest.TestCase subclass named `name`.
    """

    class TestCacheLadder(unittest.TestCase):
        """Sector-ladder cache-invalidation check on the frozen fiducial."""

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR and verify the frozen state, once per class."""
            u.require_cocoa_environment()
            u.verify_frozen()

        def _point_at(self, fid, steps):
            """Return the ladder point with each sector at its step count.

            Arguments:
              fid   = the fiducial {parameter name: value} point.
              steps = {sector name: step count}; each parameter of
                      the sector moves to fiducial + step * delta.

            Returns:
              a new {parameter name: value} dictionary (fid is not
              changed).
            """
            point = dict(fid)
            for sector, step in steps.items():
                for n, d in self.sector_deltas[sector].items():
                    point[n] = fid[n] + step * d
            return point

        def _mpairs(self, like, np_):
            """Per data-vector entry, the source bins of its (1+m) factors.

            Cosmic shear scales by both bins' factors (1+m_i)(1+m_j),
            gamma_t (and the ks cross where present) by the source
            bin's, clustering, gk and kk by nothing. Fourier data
            vectors use ncl entries per block, real ones ntheta; the
            xi_pm split doubles the shear block in real space.

            Arguments:
              like = the likelihood object (read for its bin counts
                     and, when present, its ggl_exclude pairs).
              np_  = the numpy module, passed in because numpy is
                     imported inside the test methods.

            Returns:
              int array [n_data, 2]: row k = (i, j), the source bins
              whose factors multiply entry k; -1 marks "no factor"
              (for gamma_t the row is (-1, source bin); for
              clustering both are -1).
            """
            ci = _interface(u)
            real = _native_space(u, ci) == "real"
            sizes = [int(x) for x in
                     (ci.compute_data_vector_3x2pt_real_sizes()
                      if real else
                      ci.compute_data_vector_3x2pt_fourier_sizes())]
            nlen = int(like.ntheta) if real else _fourier_nlen(like)
            nsrc = int(like.source_ntomo)
            # sspairs = every source pair (i, j) with i <= j, in
            # data-vector order (the two for clauses nest like two C
            # loops, i outer)
            sspairs = [(i, j) for i in range(nsrc)
                       for j in range(i, nsrc)]
            # excluded = the set of (lens, source) pairs the
            # likelihood drops from gamma_t ("all" excludes every
            # pair)
            excluded = _ggl_excluded(like)
            # gglpairs = every (lens, source) pair kept in gamma_t,
            # lens bin outer, source bin inner
            gglpairs = [(zl, zs)
                        for zl in range(int(like.lens_ntomo))
                        for zs in range(nsrc)
                        if (zl, zs) not in excluded]
            # start with -1 ("no factor") everywhere; the loops below
            # fill the shear and gamma_t rows block by block
            # an appended gg-cross block carries no factor: its rows
            # stay (-1, -1) like the rest of clustering
            fac = np_.zeros((sum(sizes) + _ggx_len(ci), 2),
                            dtype=int) - 1
            k = 0
            # xi_plus + xi_minus in real space
            ssrep = sspairs + sspairs if real else sspairs
            for (i, j) in ssrep:
                for t in range(nlen):
                    fac[k] = (i, j)
                    k += 1
            for (zl, zs) in gglpairs:
                for t in range(nlen):
                    fac[k] = (-1, zs)
                    k += 1
            # skip clustering
            k = sizes[0] + sizes[1] + sizes[2]
            # 6x2pt: gk (no factor), ks (source), kk (none)
            if len(sizes) > 3:
                k += sizes[3]
                for zs in range(nsrc):
                    for t in range(nlen):
                        fac[k] = (-1, zs)
                        k += 1
            return fac

        def _run_ladder(self, tatt):
            """Walk the forward and mirrored ladders; assert checks 1-5.

            Each order builds its own model in this process (the two
            models share the compiled library's C globals); check 5
            compares their final vectors bit for bit.

            Arguments:
              tatt = True runs the TATT variant, False the NLA one.

            Returns:
              nothing; a failed check raises AssertionError.
            """
            import numpy as np
            ci = _interface(u)
            masked_vector = _masked_vector_fn(u, ci)

            lk_name = u.EXAMPLES[example]["likelihood"]
            results = {}
            for order in ("forward", "mirrored"):
                info = u.load_frozen_info(example, tatt=tatt)
                model = u.make_model(info)
                fid = dict(u.build_point(model, example, tatt=tatt))
                # sector_deltas = {sector: {parameter: per-step
                # delta}}; the inner comprehension lists the
                # fiducial's names that belong to sector s
                self.sector_deltas = {
                    s: _ladder_deltas_for(
                        s, [n for n in fid
                            if _ladder_sector_of(n) == s])
                    for s, _ in LADDER_SECTORS}
                for s in required_sectors:
                    self.assertTrue(
                        self.sector_deltas[s],
                        f"no sampled parameters in sector {s}")
                # a sector nothing samples drops out of the ladder:
                # lenses that are the source sample carry no separate
                # DZ_L shifts, and a configuration may fix every IA
                # amplitude
                active = tuple(s for s in phases
                               if self.sector_deltas[s])

                walk = (active if order == "forward"
                        else tuple(reversed(active)))
                # every sector starts at step 0 (the fiducial point)
                steps = {s: 0 for s in self.sector_deltas}
                u.evaluate_chi2(model, self._point_at(fid, steps))
                # compute_data_vector_masked returns the theory
                # vector of the state the last evaluation left in
                # cosmolike
                prev = np.array(masked_vector())
                mfac = self._mpairs(model.likelihood[lk_name], np)

                for sector in walk:
                    for r in range(1, LADDER_NSTEP + 1):
                        m_prev = {
                            n: fid[n] + steps["m"] * d
                            for n, d in
                            self.sector_deltas["m"].items()}
                        steps[sector] = r
                        point = self._point_at(fid, steps)
                        u.evaluate_chi2(model, point)
                        dv = np.array(masked_vector())
                        self.assertFalse(
                            np.array_equal(dv, prev),
                            f"{order}: {sector} step {r} left the "
                            "data vector unchanged (dead sector flag "
                            "or stale cache)")
                        if sector == "m":
                            # check 2: the expected ratio dv/prev of
                            # every entry is the product of
                            # (1+m_new)/(1+m_old) over the source
                            # bins mfac names for it
                            m_now = {n: point[n]
                                     for n in self.sector_deltas["m"]}
                            # M1..M5 in bin order
                            mp = sorted(m_prev)
                            ratio = np.ones(dv.size)
                            for k in range(dv.size):
                                i, j = mfac[k]
                                if j >= 0:
                                    ratio[k] *= ((1 + m_now[mp[j]]) /
                                                 (1 + m_prev[mp[j]]))
                                if i >= 0:
                                    ratio[k] *= ((1 + m_now[mp[i]]) /
                                                 (1 + m_prev[mp[i]]))
                            # nz = boolean mask of the entries where
                            # prev is nonzero (masked entries are 0
                            # and cannot carry a ratio); rel =
                            # relative deviation from the analytic
                            # rescale on those entries
                            nz = prev != 0
                            rel = np.abs(
                                dv[nz]/(prev[nz]*ratio[nz]) - 1.0)
                            self.assertLess(
                                rel.max(), LADDER_RESCALE_RTOL,
                                f"{order}: M step {r} is not the "
                                "analytic (1+m_i)(1+m_j) rescale "
                                f"(max {rel.max():.2e})")
                        prev = dv

                final_point = self._point_at(fid, steps)
                final_chi2 = u.evaluate_chi2(model, final_point)
                final_dv = np.array(masked_vector())

                # no-op probe: identical point again, bitwise
                u.evaluate_chi2(model, dict(final_point))
                self.assertTrue(
                    np.array_equal(
                        np.array(masked_vector()),
                        final_dv),
                    f"{order}: a no-op re-evaluation changed the "
                    "data vector")

                # scramble: every sector at once, bias included
                scr = {s: LADDER_SCRAMBLE_STEP
                       for s in self.sector_deltas}
                u.evaluate_chi2(model, self._point_at(fid, scr))

                # return: the ladder's final point must reproduce
                # bitwise
                back_chi2 = u.evaluate_chi2(model, final_point)
                back_dv = np.array(masked_vector())
                self.assertTrue(
                    np.array_equal(back_dv, final_dv),
                    f"{order}: returning after the scramble did not "
                    "reproduce the data vector bit for bit (stale "
                    "sector cache)")
                self.assertEqual(
                    back_chi2, final_chi2,
                    f"{order}: chi2 after the scramble return "
                    "differs")
                results[order] = final_dv
                print(f"  {order} ladder "
                      f"({'TATT' if tatt else 'NLA'}): "
                      f"final chi2 = {final_chi2:.6f}", flush=True)

            self.assertTrue(
                np.array_equal(results["forward"],
                               results["mirrored"]),
                "the mirrored-order ladder landed on a different "
                "data vector: the answer depends on the invalidation "
                "history")

        def test_cache_consistency_nla(self):
            """The ladder checks 1-5 hold for the NLA configuration."""
            self._run_ladder(tatt=False)

        def test_cache_consistency_tatt(self):
            """The ladder checks 1-5 hold for the TATT configuration."""
            if not getattr(u, "TATT_POINT", None):
                self.skipTest(
                    "the project declares no TATT parameters")
            self._run_ladder(tatt=True)

    TestCacheLadder.__name__ = name
    TestCacheLadder.__qualname__ = name
    return TestCacheLadder
