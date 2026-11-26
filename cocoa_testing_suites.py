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


def _make_evaluate_model(evaluate_inputs, example, tatt):
    """Build a single-survey model from the input and point of an Evaluate test.

    Arguments:
      evaluate_inputs = callable accepting example and tatt keyword arguments
                        and returning a Cobaya configuration and its point.
      example = configuration name passed through to the caller.
      tatt = select the TATT variant when the test supplies two IA models.

    Returns:
      The Cobaya model, its likelihood name and a sampled-parameter dictionary
      in the order resolved by Cobaya. The caller closes the model.
    """
    from cocoa_testing import make_model

    info, point = evaluate_inputs(example=example, tatt=tatt)
    likelihood_name, = info["likelihood"]
    model = make_model(info=info)
    try:
        sampled = {name: point[name] for name in model.parameterization.sampled_params()}
    except KeyError:
        model.close()
        raise
    return model, likelihood_name, sampled


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

    A Fourier project may append a gg-cross block after the three
    probe blocks; the plain Fourier getter leaves those entries
    zero, so the extended getter is the vector the likelihood
    scores. The choice is made at call time, because the cross-pair
    list is installed by the first model build, after this function
    is bound.

    Arguments:
      u  = the project's cocoa_test_utils module.
      ci = the imported interface module.

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
            "        CHI2_TOLERANCE = 0.2 is a chosen regression\n"
            "        tolerance for every frozen chi2: wide enough for\n"
            "        compiler and library-version noise, tight enough\n"
            "        that a code or input change is unlikely to hide\n"
            "        under it; it is not a physical detectability\n"
            "        limit. A failure means the code or the frozen\n"
            "        inputs changed the prediction.\n        ")
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
            "        The worker captures the actual scored vectors\n"
            "        and checks every restored entry at rtol=1e-11.\n"
            "        This separate scalar check uses RACE_TOLERANCE\n"
            "        = 1e-4. Equal scores alone would not establish\n"
            "        equal vectors: different residuals can lie on\n"
            "        the same chi-squared surface. Agreement tests\n"
            "        this sequence, not every possible race.\n        ")
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


def nmodels_current_info(u, example, high_accuracy=False):
    """The current-prior check uses the NLA version of the live evaluation.

    Keep its survey files, parameter prescriptions and physical theory.
    The ordinary prediction uses the common CAMB resolution. The pushed
    prediction uses the project's established likelihood accuracy settings
    and the shared higher CAMB transfer resolution.

    Arguments:
      u = the project's cocoa_test_utils module.
      example = an EXAMPLES key whose provenance names the evaluation YAML.
      high_accuracy = use all the numerical refinements together.

    Returns:
      The fully resolved Cobaya input and the evaluation's fiducial overrides.
    """
    from pathlib import Path

    from cobaya.input import update_info
    from cobaya.yaml import yaml_load_file
    from cocoa_testing import HIGH_ACCURACY_CAMB_EXTRA_ARGS

    project = Path(u.TESTS_DIR).parent
    case = u.EXAMPLES[example]
    path = project / case["provenance"]
    info = yaml_load_file(str(path))
    fiducial = info.pop("sampler")["evaluate"]["override"]
    info.pop("output", None)
    info["debug"] = 30
    info["timing"] = False
    info["stop_at_error"] = True
    likelihood = info["likelihood"][case["likelihood"]]
    likelihood["IA_model"] = 0
    likelihood["print_datavector"] = False
    camb = info["theory"]["camb"]
    camb["stop_at_error"] = True
    arguments = camb["extra_args"]
    arguments["AccuracyBoost"] = 1.05
    arguments["k_per_logint"] = 10
    arguments["kmax"] = 10.0
    if high_accuracy:
        likelihood.update(u.HIGH_ACCURACY_LIKELIHOOD)
        arguments.update(HIGH_ACCURACY_CAMB_EXTRA_ARGS)
    return update_info(info), fiducial


def nmodels_draw(u, example):
    """Return the fixed N-random-models points of one example.

    The current-prior projects sample every coordinate from the
    resolved Cobaya prior. Historical suites use their recorded
    box-draw, hold and amplitude settings. The check and explicit
    reference generator share this function.

    Arguments:
      u       = the project's cocoa_test_utils module.
      example = a key of u.EXAMPLES.

    Returns:
      (points, held), sampled parameter dictionaries and hold flags.
    """
    if getattr(u, "NMODELS_CURRENT_PRIOR", False):
        from cocoa_testing import draw_nmodels_prior_rows

        info, _ = nmodels_current_info(u=u, example=example)
        names, values = draw_nmodels_prior_rows(info=info)
        points = [dict(zip(names, row, strict=True)) for row in values]
        return points, [False] * len(points)
    return u.nmodels_points(
        example, box_overrides=getattr(u, "NMODELS_BOX", None),
        hold=getattr(u, "NMODELS_HOLD", None),
        amplitude_cut=getattr(u, "NMODELS_AMPLITUDE_CUT", None))


def nmodels_vectors(u, example, points, high_accuracy, expected_precision=None):
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
                      accuracy settings, False with ordinary settings.
      expected_precision = saved masked inverse covariance. The current-prior
                           replay checks this before calculating any vectors.

    Returns:
      (vectors, inv_cov): a (len(points), n) float64 array of masked
      theory vectors, and the (n, n) masked inverse covariance of the
      frozen data set.

    Raises:
      AssertionError when a point and the model disagree on the
      sampled parameters (the likelihood or theory code changed its
      parameters since the freeze).
    """
    from contextlib import chdir, nullcontext
    import time

    import numpy as np

    ci = _interface(u)
    masked_vector = _masked_vector_fn(u, ci)
    current = getattr(u, "NMODELS_CURRENT_PRIOR", False)
    if current:
        info, _ = nmodels_current_info(u=u, example=example, high_accuracy=high_accuracy)
        where = chdir(os.environ["ROOTDIR"])
    else:
        info = u.load_frozen_info(example, tatt=False, high_accuracy=high_accuracy)
        where = nullcontext()
    if high_accuracy:
        label = "high-accuracy"
    else:
        label = "default-settings"
    print(f"  building the {label} model of {example} ...", flush=True)
    start = time.time()
    with where, u.make_model(info) as model:
        print(f"  built in {time.time() - start:.0f} s", flush=True)
        inv_cov = np.array(ci.get_inv_cov_masked(), dtype="float64")
        if expected_precision is not None:
            np.testing.assert_allclose(inv_cov, expected_precision, rtol=1e-10, atol=1e-12)
        if current:
            like_name = u.EXAMPLES[example]["likelihood"]
            like = model.likelihood[like_name]
            options = info["likelihood"][like_name]
            effective = {}
            for name in u.HIGH_ACCURACY_LIKELIHOOD:
                effective[name] = getattr(like, name)
                assert effective[name] == options[name], (example, name, effective[name])
            print(f"  native controls {effective}", flush=True)
        sampled = set(model.parameterization.sampled_params())
        vectors = []
        for index, point in enumerate(points):
            if sampled != set(point):
                raise AssertionError(
                    "sampled-parameter set differs from the saved point:\n"
                    f"  new: {sorted(sampled - set(point))}\n"
                    f"  gone: {sorted(set(point) - sampled)}")
            start = time.time()
            u.evaluate_chi2(model, point)
            vector = np.array(masked_vector(), dtype="float64")
            if vector.shape != (len(inv_cov),) or not np.isfinite(vector).all():
                raise ValueError(f"{example} model {index + 1} returned an invalid vector.")
            vectors.append(vector)
            if current and index == 0:
                from ctypes import CDLL

                from threadpoolctl import threadpool_info

                _, transfer = model.theory["camb.transfers"].get_CAMB_transfers()
                params = transfer.Params
                requested = info["theory"]["camb"]["extra_args"]
                boost = params.Accuracy.AccuracyBoost
                spacing = params.Transfer.k_per_logint
                assert boost == requested["AccuracyBoost"]
                assert spacing == requested["k_per_logint"]
                assert params.Transfer.kmax >= requested["kmax"]
                print(f"  CAMB boost={boost:g}, k_per_logint={spacing}, "
                      f"requested/effective kmax={requested['kmax']:g}/"
                      f"{params.Transfer.kmax:g}", flush=True)
                native_threads = CDLL(ci.__file__).omp_get_max_threads()
                blas = {pool["num_threads"] for pool in threadpool_info()
                        if pool["user_api"] == "blas"}
                print(f"  native OpenMP={native_threads}, BLAS={sorted(blas)}", flush=True)
                expected_threads = int(os.environ["OMP_NUM_THREADS"])
                assert native_threads == expected_threads and blas == {1}, (
                    f"{example}: measured OpenMP={native_threads}, BLAS={sorted(blas)}. "
                    f"Expected OpenMP={expected_threads} and BLAS=[1]."
                )
            print(f"  model {index + 1}/{len(points)}: {label} vector "
                  f"({time.time() - start:.1f} s)", flush=True)
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
    import matplotlib

    import cocoa_testing_figures as figures

    path = os.path.join(figure_dir,
                        f"accuracy_nmodels_{example}_seed{seed}.png")
    if case is None:
        case = example
    color_ceiling = figures.NMODELS_COLOR_CEILING
    data_path = None
    current = getattr(u, "NMODELS_CURRENT_PRIOR", False)
    if current:
        _, fiducial = nmodels_current_info(u=u, example=example)
        label = case
    else:
        fiducial = u.load_frozen_point(example)
        project = os.path.basename(os.path.dirname(u.TESTS_DIR))
        label = (f"{project}, {case.replace('Delta z', 'Δz')}: "
                 f"{len(points)} models, seed base {seed}")
    fisher_panel = None
    reference = u.EXAMPLES[example].get("nmodels_fisher")
    if reference is not None:
        fisher_panel = figures.saved_nmodels_fisher_panel(
            path=os.path.join(u.TESTS_DIR, reference),
        )
    if current or fisher_panel is not None:
        color_ceiling = 1.0
        data_path = os.path.splitext(path)[0] + ".txt"
    with matplotlib.rc_context():
        if current:
            # Match the other current survey figures without changing the legacy style.
            matplotlib.rcParams['mathtext.fontset'] = 'stix'
            matplotlib.rcParams['font.family'] = 'STIXGeneral'
            matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
            matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
            matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
            matplotlib.rcParams['xtick.bottom'] = True
            matplotlib.rcParams['xtick.top'] = False
            matplotlib.rcParams['ytick.right'] = False
            matplotlib.rcParams['axes.edgecolor'] = 'black'
            matplotlib.rcParams['axes.linewidth'] = '1.0'
            matplotlib.rcParams['axes.labelsize'] = 'medium'
            matplotlib.rcParams['axes.grid'] = True
            matplotlib.rcParams['grid.linewidth'] = '0.0'
            matplotlib.rcParams['grid.alpha'] = '0.18'
            matplotlib.rcParams['grid.color'] = 'lightgray'
            matplotlib.rcParams['legend.labelspacing'] = 0.77
            matplotlib.rcParams['savefig.bbox'] = 'tight'
            matplotlib.rcParams['savefig.format'] = 'pdf'
        figures.random_models_triangle(
            points=points, deltas=deltas,
            fiducial=fiducial, path=path, limit=limit,
            held=held, label=label,
            colorbarlabel=r"$\Delta\chi^2$", color_ceiling=color_ceiling,
            data_path=data_path, fisher_panel=fisher_panel)
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
      NMODELS_CURRENT_PRIOR = True uses the corresponding current NLA
                              evaluation and all its sampled priors.
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
        block. For the A-checks and the K scan there is no pass/fail
        on the size of the delta: how much numerical error an
        analysis tolerates is a judgment call, so only a broken
        evaluation fails. The N-random-models check on this class is
        the exception: with NMODELS_LIMIT set it enforces that bound
        on every model's delta.
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
            reference, so the printed difference is a refinement
            comparison at the frozen point: the measured sensitivity
            of the chi2 to pushing every numerical setting under the
            same physical model. It estimates the defaults' numerical
            error; the pushed settings are a finer calculation, not
            the continuum answer.

            Arguments:
              check   = the advisory label (e.g. "A1") for the
                        report.
              example = a key of u.EXAMPLES (non-emulated
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

    # the K scans: one method per bound example. A single key (a str)
    # builds the one scan most projects bind; a tuple builds one scan
    # per listed example, and None builds none (tuple(None or ()) is
    # the empty tuple).
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

    # NMODELS_EXAMPLES is a tuple of (example key, display label)
    # pairs; the generator copies each pair into a plain tuple, and a
    # project without the constant yields no pairs and no check
    nmodels_pairs = tuple(tuple(pair) for pair
                          in getattr(u, "NMODELS_EXAMPLES", ()))
    nmodels_limit = getattr(u, "NMODELS_LIMIT", None)

    def make_nmodels(nm_example, nm_label):
        def test_ax99_nmodels(self):
            """N-random-models check at the fixed stress points.

            The A-checks report the measured sensitivity to the
            pushed accuracy settings at the one frozen fiducial
            point. This check measures that sensitivity at
            NMODELS_COUNT fixed points of the designed stress sample
            (nmodels_points). For each point

                delta chi2 = (v_high - v_default)^T C^-1 (v_high - v_default),

            with v_high the stored vector at the pushed accuracy
            settings (frozen/nmodels_<example>.npz, written once by
            generate_frozen_reference.py --nmodels), v_default the
            vector at the default settings, computed here, and C the
            frozen covariance. Storing the expensive half keeps the
            check to one model build and one evaluation per point.

            Model m draws with seed RANDOM_MODEL_SEED + m. Current
            checks use the full Cobaya prior and historical checks
            use their recorded stress settings. The test verifies
            that the points and reference settings match the saved
            file. Current checks also compare the precision matrix
            before calculating their ordinary predictions.

            The deltas have to be finite. With NMODELS_LIMIT set,
            each |delta| must also stay below it; with the limit
            None the check is advisory. Before the limit applies, the check
            draws the Omega_m - clustering-amplitude triangle of the
            points (cocoa_testing_figures.random_models_triangle)
            into the folder COCOA_ACCURACY_NMODELS_FIGURES names, or
            the system temporary folder. The ax99 in the method name
            sorts this check after the A-checks (unittest runs
            methods in name order).
            """
            import json
            import math

            import numpy as np
            from cocoa_testing import (
                RANDOM_MODEL_SEED, report_random_model, report_random_model_summary,
            )

            path = nmodels_file(u, nm_example)
            if not os.path.isfile(path):
                self.skipTest(
                    f"no stored N-random-models vectors for "
                    f"{nm_example} (frozen/{os.path.basename(path)}); "
                    "generate them with generate_frozen_reference.py "
                    "--nmodels")
            # checked before the models run, which take minutes
            figure_dir = _nmodels_figure_dir()
            seed = RANDOM_MODEL_SEED
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
            self.assertEqual(int(stored["seed"]), seed)
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
            expected_precision = None
            if getattr(u, "NMODELS_CURRENT_PRIOR", False):
                pushed, _ = nmodels_current_info(u=u, example=nm_example, high_accuracy=True)
                configuration = json.dumps(pushed, sort_keys=True)
                self.assertEqual(str(stored["pushed_configuration"]), configuration,
                                 msg=f"the pushed reference configuration changed; {regenerate}")
                expected_precision = stored["precision"]
            else:
                self.assertEqual(
                    str(stored["config_hash"]),
                    u.nmodels_config_hash(nm_example),
                    msg=f"the pushed reference configuration changed; {regenerate}")
            high_vectors = stored["vectors"]
            stored.close()
            self.assertTrue(high_vectors.ndim == 2 and len(high_vectors) == n_models)
            self.assertTrue(np.isfinite(high_vectors).all())

            default_vectors, inv_cov = nmodels_vectors(
                u, nm_example, points, high_accuracy=False,
                expected_precision=expected_precision)
            self.assertEqual(
                high_vectors.shape, default_vectors.shape,
                msg="the stored vectors have another length than "
                    f"today's masked data vector; {regenerate}")
            deltas = []
            for index in range(n_models):
                difference = high_vectors[index] - default_vectors[index]
                delta = float(difference @ inv_cov @ difference)
                report_random_model(index, n_models, points[index], delta)
                deltas.append(delta)
            # enumerate pairs each delta with its position, counted
            # from 0, so a broken model is named by number
            for index, delta in enumerate(deltas):
                # advisory: the size of the delta is a judgment
                # call, but a non-finite one means the evaluation
                # itself broke
                self.assertTrue(
                    math.isfinite(delta) and delta >= 0.0,
                    msg=f"model {index}: invalid delta chi2 "
                        f"{delta!r}")
            report_random_model_summary(deltas)
            if nmodels_hold is not None:
                drawn_deltas = []
                held_deltas = []
                for index in range(n_models):
                    if held[index]:
                        held_deltas.append(deltas[index])
                    else:
                        drawn_deltas.append(deltas[index])
                print("  models drawn across the prior:", flush=True)
                report_random_model_summary(drawn_deltas)
                if len(held_deltas) > 0:
                    print("  models holding "
                          f"{', '.join(nmodels_hold)} at the fiducial:",
                          flush=True)
                    report_random_model_summary(held_deltas)
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
                # the comprehension collects the (index, delta) pair
                # of every model at or over the limit, so the failure
                # message names each offending model and its seed
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
                             interp_default=0, zmid_default=0,
                             zmid_floor=1.0e-2):
    """Build the runtime photo-z convention sweep class.

    The likelihood exposes two runtime knobs that control how the
    n(z) table files are turned into the smooth distributions the
    Limber integrals consume: photoz_interpolation_type (the stage-1
    interpolant) and photoz_zmid_convention (how the z column of the
    n(z) file is read). The class evaluates the frozen fiducial of the
    bound example under the five settings of its sweep in one process, measures
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
      zmid_floor = the dead-flag floor of the z-reading flip. The
                flip shifts every distribution by half a histogram
                cell, which moves chi2 by about 0.01-0.05 on narrow
                bins but less on wide ones (about 1e-3 on Delta z =
                0.2 bins); a dead flag gives exactly zero, so any
                floor far above numerical noise (1e-8) detects it.

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
    # The alternatives must be seen (an n(z) cache that failed to
    # rebuild would give exactly zero); the floors are orders of
    # magnitude below the measured deltas, so they only catch a dead
    # flag, never normal numerical drift. The z-reading flip shifts
    # every distribution by dz/2, hence its larger floor.
    floors = {tag: (zmid_floor if tag in zmid_names.values() else 1.0e-8)
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
    (galaxy-galaxy lensing) chooses between the full non-Limber
    projection integral at low multipoles (within the adopted
    modeling prescriptions) and the Limber approximation at every
    multipole. The class
    evaluates the frozen fiducial three times in one process (the
    default, the other setting, the default again) and asserts: the
    delta is seen (dead-flag floor), only the probe's own block
    changes, the round trip reproduces the first vector bitwise, the
    delta chi2 matches the project's measured value to the relative
    band, and the default reproduces the frozen reference (last, so an
    out-of-date snapshot cannot hide the other checks).

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
            # Per-block scores, printed largest first; the loop stops
            # at the first one below 0.1% of the total. Each score
            # zeroes every entry outside one bin pair and applies the
            # FULL precision matrix W = C^-1: for a difference split
            # over two disjoint blocks, delta = delta_1 + delta_2,
            #   delta^T W delta = delta_1^T W delta_1
            #                     + delta_2^T W delta_2
            #                     + 2 delta_1^T W delta_2,
            # and the printed scores omit the cross term, so they
            # rank the sensitive blocks without adding up to the
            # total (with W = [[1, 1/2], [1/2, 1]] and delta = (1, 1)
            # the two blocks score 1 and 1 while the total is 3).
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

            # last, so that an out-of-date frozen snapshot does not
            # hide the four checks above
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
    clustering engine: 0 = FKEM, 1 = the low-rank SVD thin-bin
    engine (C_cl_tomo_lowrank in cosmo2D.c); the `default` argument
    names the engine the project's likelihood defaults to. The
    delta compares the two implementations at the frozen
    configuration: their main modeling differences are FKEM's
    Limber treatment of the nonlinear density term and its
    multipole cap lmax_nonlimber against the SVD's rank truncation,
    on top of the numerical controls each engine carries. Neither
    engine is the exact physical answer, so agreement with the
    recorded `measured` value checks that the comparison
    reproduces, not that either engine is right. The delta's size
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
         (last, so an out-of-date frozen snapshot cannot hide
         checks 1-4).

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
    if default not in (0, 1):
        raise ValueError(
            f"lowrank_gg_suite: default = {default!r}; the project's "
            "nonlimber_lowrank_gg default is 0 (FKEM) or 1 (low-rank SVD)")
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
                info["likelihood"][lk_name]["nonlimber_lowrank_gg"] = flag
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
                vectors[tag] = np.array(masked_vector())
                if icov is None:
                    # all settings share one data set (mask,
                    # covariance), so the first build's masked
                    # inverse covariance serves all
                    icov = np.array(ci.get_inv_cov_masked())
                    if _native_space(u, ci) == "real":
                        sizes = ci.compute_data_vector_3x2pt_real_sizes()
                    else:
                        sizes = (
                            ci.compute_data_vector_3x2pt_fourier_sizes())
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

            # last, so that an out-of-date frozen snapshot does not
            # hide the four checks above
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
                    "checks of the factory docstring.")
    setattr(TestLowrank, test.__name__, test)
    TestLowrank.__name__ = name
    TestLowrank.__qualname__ = name
    return TestLowrank


# ---- the early-exit tolerance of the non-Limber C_l --------------------------

# The Fisher step of the example notebooks: a relative step h = 0.025 (an
# absolute one for a parameter whose fiducial is 0, such as wa = 0).
NONLIMBER_TOL_STEP = 0.025
# The FKEM caps (limits.LMAX_NOLIMBER) of the hand-off check: the default
# 150 and the higher cap the comparison is judged against, 200. The
# low-rank (SVD) engine runs at the same two caps, with its own cap.
NONLIMBER_TOL_FKEM_LMAX = (150, 200)
NONLIMBER_TOL_SVD_LMAX = (150, 200)
# The core defaults of the two tolerances, (gg, gs): the values
# reset_Ntable_struct stores and the default arguments of
# init_nonlimber_tolerance. Tests 2-4 and 8 switch the early exit on
# with them, so they keep testing the setting in a project whose
# production tolerances are 0 (no early exit).
NONLIMBER_TOL_CORE_DEFAULTS = (0.002, 0.01)
# SVD ranks kept per lens bin in the low-rank runs: the likelihoods'
# default (nonlimber_lowrank_rank: 5)
NONLIMBER_TOL_SVD_RANK = 5
# Absolute slack of every comparison between two noise values, in
# sqrt(chi^2) units: far below every measured noise floor (1e-4 to 1e-3),
# far above the round-off noise of a column on which the data vector
# depends as a polynomial of degree below 4 (a linear bias: about 1e-12),
# whose fourth difference vanishes.
NONLIMBER_TOL_NOISE_FLOOR = 1.0e-6
# Dead-setter floor of the real-space change delta^T C^-1 delta between
# tolerance 0 and the core defaults: the measured changes are 1.4e-4 to
# 4.1e-2, so a value below this floor means the setter never reached the
# C code or the real-space cache did not rebuild.
NONLIMBER_TOL_CHANGE_FLOOR = 1.0e-8


def _cube_diagonal_at_one_ell(cube):
    """Return the auto spectra of a (one ell, bin, bin) C_l cube.

    C_gg_tomo and C_gg_tomo_limber return an arma::Cube with the
    multipoles on its first axis and the bin pair on the other two; with
    a single multipole the multipole axis has length 1.

    Arguments:
      cube = the array the binding returns for one multipole.

    Returns:
      1D array (n_lens,): the (b, b) entries.

    Raises:
      ValueError when the array is not three-dimensional with a
      multipole axis of length 1.
    """
    import numpy as np

    array = np.asarray(cube, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(
            f"a C_l cube for one multipole has shape {array.shape}; "
            "a three-dimensional array was expected")
    if array.shape[0] == 1:
        matrix = array[0]
    elif array.shape[1] == 1:
        matrix = array[:, 0, :]
    else:
        raise ValueError(
            f"a C_l cube for one multipole has shape {array.shape}; "
            "no axis of length 1 holds the multipole")
    return np.diag(matrix).copy()


def nonlimber_tolerance_suite(u, name, example, bias_parameter,
                              expected_defaults, change_bound=None,
                              noise_bound=None, handoff_margin=None,
                              svd_rule=None, svd_margin=None,
                              expected_lmax_nonlimber=None, *,
                              evaluate_inputs=None, interface=None):
    """Build the class of the non-Limber early-exit and hand-off checks.

    The non-Limber real-space C_l (w_gg_tomo for w(theta), w_gammat_tomo
    for gamma_t) are computed in blocks of 16 multipoles, and a lens bin
    (gg) or lens-source pair (gs) whose C_l agrees with Limber within a
    tolerance at the last multipole of a block takes the Limber values
    from there: the early exit. Its switch multipole is an integer that
    moves with the parameters, so a finite-difference derivative whose
    stencil straddles a switch sees a step in the model, which a Fisher
    matrix reads as information. init_nonlimber_tolerance sets the two
    tolerances (core defaults Ntable.NL_tol_gg = 0.002 and
    Ntable.NL_tol_gs = 0.01, NONLIMBER_TOL_CORE_DEFAULTS; 0 = no early
    exit). A project's likelihood yamls set its production tolerances
    (keys nonlimber_tolerance_gg/gs), which may differ from the core
    defaults: 0 in a project where the exit's error across the prior is
    too large. The Fourier entries C_gg_tomo_ells and C_gs_tomo_ells
    never take the exit.

    A five-point fourth-difference diagnostic is computed with
    cosmolike_notebook_utils.fisher.get_stencil_noise: the fourth
    difference of the data vector at the five points p0 + k eps,
    k = -2..2, whitened by the masked inverse covariance,
    sigma = sqrt(Delta4 m^T C^-1 Delta4 m/70), in sqrt(chi^2) units.
    A smooth model gives sigma ~ eps^4 m''''; a step between two stencil
    points gives sigma of the order of the step's whitened size. Calling
    sigma independent per-evaluation noise requires an additional model
    for the numerical errors; the statistic also contains smooth fourth-
    order curvature. The engine comparison below holds the galaxy-shear
    cap fixed, but does not separate these contributions. The columns are
    w0 and wa (Cobaya names w = w0 and w0pwa = w0 + wa, so varying w0
    at fixed wa moves both names together) and the bias of the highest
    lens bin, with the notebooks' step h = 0.025 (absolute for wa = 0).
    The data vector is the example's masked vector. A frozen example
    keeps its recorded settings. An Evaluate input uses the settings
    selected by its configuration loader.

    One model is built for the class; every test restores the production
    settings (the tolerances, the FKEM cap and the gg engine the
    likelihood set) in a finally block, and the last test checks that
    the fiducial vector after the suite equals the one before it bit for
    bit. The early exit of tests 2-4 and 8 is the one of the core
    defaults, whatever the production tolerances. The tests:

      1. the production settings reproduce the frozen reference chi2
         or the native masked quadratic for an Evaluate input,
         the likelihood applied this project's production tolerances
         (expected_defaults), and the binding's default arguments give
         the vector of the core defaults bit for bit;
      2. the core defaults, then tolerance 0, then the production
         tolerances give the production vector back bit for bit (the
         real-space caches key on the tolerance values);
      3. real space: tolerance 0 against the core defaults changes only
         the non-Limber blocks, by a whitened delta^T C^-1 delta above
         a dead-setter floor and below change_bound; Fourier space:
         nothing changes;
      4. real space: the stencil noise of the three columns at the core
         defaults and at 0, at the production FKEM cap; at 0 every
         column stays below noise_bound;
      5. the FKEM hand-off at the fixed cap, tolerance 0: the noise at
         lmax_nonlimber = 150 and 200; at 150 it is at most
         (1 + handoff_margin) times the 200 value and below noise_bound.
         This empirical one-sided bound can flag a hand-off contribution;
         passing it does not exclude one. It
         also prints |C_l^FKEM/C_l^Limber - 1| at l = lmax - 1 per lens
         bin and the whitened change of the fiducial vector between the
         caps 150 and 200;
      6. the low-rank (SVD) engine, tolerance 0, at its caps 150 and
         200, next to FKEM with the same galaxy-shear cap in each pair:
         svd_rule "not_larger" asserts
         sigma_SVD <= (1 + svd_margin) sigma_FKEM at each cap, "bound"
         asserts sigma_SVD < noise_bound;
      7. a negative or nonfinite tolerance stops the process with a
         message naming it (checked in a subprocess, since the C code
         exits);
      8. the global accuracy boost AB divides the tolerances in use by
         1 + 2 (AB - 1) above AB = 1 (get_nonlimber_tolerance at AB = 1,
         2, 3 for the core defaults; a setting of 0 stays 0), and at
         AB = 2 the vector at the scaled core defaults lies at least as
         close to the tolerance-0 vector of the same boost as the vector
         at the unscaled ones (the AB = 1 distance is printed next to
         them; Fourier space: no change at either boost);
      9. the production state is back: the fiducial vector equals the
         first one bit for bit.

    A pre-change comparison (the default against the build before the
    tolerance became a setting) needs the old library and runs outside
    this suite.

    The bounds are set from this project's measurement; None turns the
    corresponding assertion into a printed record only.

    Arguments:
      u              = the project's cocoa_test_utils module.
      name           = the class name to build (assigned by the caller
                       to the same module-level name).
      example        = the key of u.EXAMPLES the suite evaluates (a
                       configuration with galaxy clustering).
      bias_parameter = the sampled linear-bias parameter of the highest
                       lens bin, e.g. "LSST_B1_5".
      expected_defaults = (gg, gs): the production tolerances this
                       project's likelihood yamls set, which test 1
                       asserts, e.g. (0.002, 0.01) (the core defaults)
                       or (0.0, 0.0) (no early exit).
      change_bound   = upper bound of the real-space delta^T C^-1 delta
                       of test 3, or None.
      noise_bound    = upper bound of sigma at tolerance 0 and the
                       production cap (tests 4, 5 and the "bound" rule
                       of 6), or None.
      handoff_margin = relative margin of test 5 (sigma at cap 150 over
                       sigma at cap 200), or None.
      svd_rule       = "not_larger", "bound" or None (record only).
      svd_margin     = relative margin of the "not_larger" rule.
      expected_lmax_nonlimber = optional production FKEM cap checked
                       against the resolved likelihood configuration.
      evaluate_inputs = optional callable receiving example and tatt
                       keywords and returning a Cobaya configuration
                       and fiducial. This uses the live native masked
                       quadratic for check 1.
      interface      = the compiled survey module when evaluate_inputs
                       is supplied.

    Returns:
      a unittest.TestCase subclass named `name`.

    Raises:
      ValueError for an unknown svd_rule, "not_larger" without a
      margin, or expected_defaults that are not two finite values
      >= 0.
    """
    import math

    if (evaluate_inputs is None) != (interface is None):
        raise ValueError("supply evaluate_inputs and interface together")
    if len(expected_defaults) != 2:
        raise ValueError(
            f"nonlimber_tolerance_suite: expected_defaults = "
            f"{expected_defaults!r}; give the pair (gg, gs) of the "
            "production tolerances")
    expected_defaults = (float(expected_defaults[0]),
                         float(expected_defaults[1]))
    for value in expected_defaults:
        if not (math.isfinite(value) and value >= 0.0):
            raise ValueError(
                f"nonlimber_tolerance_suite: expected_defaults = "
                f"{expected_defaults!r}; each tolerance is finite and "
                ">= 0")
    if svd_rule not in (None, "not_larger", "bound"):
        raise ValueError(
            f"nonlimber_tolerance_suite: svd_rule = {svd_rule!r}; use "
            "'not_larger', 'bound' or None")
    if svd_rule == "not_larger" and svd_margin is None:
        raise ValueError(
            "nonlimber_tolerance_suite: svd_rule = 'not_larger' needs "
            "svd_margin (the relative margin of sigma_SVD over sigma_FKEM)")
    columns = ("w0", "wa", bias_parameter)
    reference_key = f"{example}_nla"
    if evaluate_inputs is None:
        evaluate_chi2 = u.evaluate_chi2
        require_cocoa_environment = u.require_cocoa_environment
    else:
        from cocoa_testing import evaluate_chi2, require_cocoa_environment

    class TestNonLimberTolerance(unittest.TestCase):
        """Early-exit tolerance, FKEM hand-off and SVD noise checks."""

        @classmethod
        def setUpClass(cls):
            """Build the model, evaluate the fiducial, keep the state."""
            import numpy as np

            require_cocoa_environment()
            print(f"\n  building model ({example}, NLA) ...", flush=True)
            if evaluate_inputs is None:
                u.verify_frozen()
                cls.reference = u.load_reference()
                cls.ci = _interface(u)
                info = u.load_frozen_info(example, tatt=False)
                lk_name = u.EXAMPLES[example]["likelihood"]
                cls.model = u.make_model(info)
                cls.point = u.build_point(cls.model, example, tatt=False)
            else:
                cls.ci = interface
                cls.model, lk_name, cls.point = _make_evaluate_model(
                    evaluate_inputs=evaluate_inputs, example=example, tatt=False)
                cls.addClassCleanup(cls.model.close)
            # staticmethod: a plain function stored on the class would be
            # bound to the instance (called with self) when a test method
            # reads it as self.masked_vector
            cls.masked_vector = staticmethod(_masked_vector_fn(u, cls.ci))
            cls.like = cls.model.likelihood[lk_name]
            for key in ("w", "w0pwa", bias_parameter):
                if key not in cls.point:
                    raise ValueError(
                        f"the frozen point of {example} has no sampled "
                        f"parameter {key!r}; the w0, wa and bias columns "
                        "need w, w0pwa and the bias of the highest lens "
                        "bin")
            # the settings the likelihood applied at initialize, read
            # back with the defaults of its own getattr calls
            cls.production = {
                "gg_tol": float(getattr(cls.like, "nonlimber_tolerance_gg",
                                        0.002)),
                "gs_tol": float(getattr(cls.like, "nonlimber_tolerance_gs",
                                        0.01)),
                "lmax_nonlimber": int(getattr(cls.like, "lmax_nonlimber",
                                              150)),
                "lowrank_flag": int(getattr(cls.like,
                                            "nonlimber_lowrank_gg", 0)),
                "lowrank_rank": int(getattr(cls.like,
                                            "nonlimber_lowrank_rank", 5)),
                "lowrank_lmax": int(getattr(cls.like,
                                            "nonlimber_lowrank_lmax", 150)),
                "accuracy_boost": float(getattr(cls.like, "accuracyboost",
                                                1.0)),
                "integration_accuracy": int(getattr(
                    cls.like, "integration_accuracy", 0)),
                "probe_integration_factors": {
                    "integration_accuracy_ss": float(getattr(
                        cls.like, "integration_accuracy_ss", 1.0)),
                    "integration_accuracy_gs": float(getattr(
                        cls.like, "integration_accuracy_gs", 1.0)),
                    "integration_accuracy_gg": float(getattr(
                        cls.like, "integration_accuracy_gg", 1.0)),
                    "integration_accuracy_gk": float(getattr(
                        cls.like, "integration_accuracy_gk", 1.0)),
                    "integration_accuracy_ks": float(getattr(
                        cls.like, "integration_accuracy_ks", 1.0)),
                    "integration_accuracy_kk": float(getattr(
                        cls.like, "integration_accuracy_kk", 1.0)),
                    "integration_accuracy_cg": float(getattr(
                        cls.like, "integration_accuracy_cg", 1.0)),
                    "integration_accuracy_cc": float(getattr(
                        cls.like, "integration_accuracy_cc", 1.0)),
                    "integration_accuracy_cs": float(getattr(
                        cls.like, "integration_accuracy_cs", 1.0)),
                    "integration_accuracy_n": float(getattr(
                        cls.like, "integration_accuracy_n", 1.0)),
                },
            }
            # the emulator path of the likelihoods replaces the yaml
            # accuracy by a fixed low one (boost 0.35, level -1)
            if int(getattr(cls.like, "use_emulator", 0)) == 1:
                cls.production["accuracy_boost"] = 0.35
                cls.production["integration_accuracy"] = -1
            cls.space = _native_space(u, cls.ci)
            cls.chi2 = evaluate_chi2(cls.model, cls.point)
            cls.vector = np.array(cls.masked_vector())
            cls.icov = np.array(cls.ci.get_inv_cov_masked())
            # {(engine, cap, tolerance): record} shared by tests 4-6, so
            # each setting is measured once whatever the test order
            cls.measurements = {}
            cls.zero_vector = None
            print(f"  production settings: {cls.production}; "
                  f"{cls.space} space, chi2 = {cls.chi2:.6f}", flush=True)

        @classmethod
        def _restore_production(cls):
            """Restore the accuracy settings, tolerances, FKEM cap and gg engine.

            init_accuracy_boost rescales every boosted table size from the
            baselines the likelihood's own call captured, so the production
            boost gives back the production sizes exactly.
            """
            production = cls.production
            cls.ci.init_accuracy_boost(
                accuracy_boost=production["accuracy_boost"],
                integration_accuracy=production["integration_accuracy"],
                **production["probe_integration_factors"],
            )
            cls.ci.init_nonlimber_tolerance(gg_tol=production["gg_tol"],
                                            gs_tol=production["gs_tol"])
            cls.ci.init_lmax_nonlimber(lmax=production["lmax_nonlimber"])
            cls.ci.init_lowrank_nonlimber_gg(
                flag=production["lowrank_flag"],
                nrank=production["lowrank_rank"],
                lmax=production["lowrank_lmax"])

        @classmethod
        def _vector_at(cls, param, AccuracyBoost=1.0):
            """Evaluate the masked vector at (w0, wa, b) (the dv form).

            Arguments:
              param         = 1D array (3,): w0, wa and the bias.
              AccuracyBoost = unused: the frozen accuracy settings hold.

            Returns:
              1D array, the masked theory vector at the point.
            """
            import numpy as np

            point = dict(cls.point)
            point["w"] = float(param[0])
            point["w0pwa"] = float(param[0] + param[1])
            point[bias_parameter] = float(param[2])
            if evaluate_inputs is not None:
                names = cls.model.parameterization.sampled_params()
                values = np.array([point[name] for name in names])
                assert np.isfinite(cls.model.prior.logps_internal(values)), (
                    f"the (w0, wa, bias) stencil point {param} must lie in the sampled prior")
            evaluate_chi2(cls.model, point)
            vector = np.array(cls.masked_vector())
            cls.last_vector = vector
            return vector

        @classmethod
        def _fiducial_vector(cls):
            """Evaluate the fiducial and return its masked vector."""
            import numpy as np

            evaluate_chi2(cls.model, cls.point)
            return np.array(cls.masked_vector())

        @classmethod
        def _measure(cls, engine, lmax, tolerance):
            """Measure sigma of the three columns at one setting.

            Arguments:
              engine    = "FKEM" (cap = lmax) or "SVD" (the low-rank
                          engine with its own cap = lmax).
              lmax      = the engine's non-Limber cap and the shared
                          FKEM cap used by galaxy shear in both cases.
              tolerance = the pair (gg_tol, gs_tol), e.g.
                          NONLIMBER_TOL_CORE_DEFAULTS or (0.0, 0.0).

            Returns:
              dict: "sigma" {column: sigma}, "center" (the fiducial
              vector at this setting), "dev" (FKEM only: the auto
              spectra's C_l^FKEM/C_l^Limber - 1 at l = lmax - 1, one per
              lens bin), "seconds".
            """
            import time

            import numpy as np
            from cosmolike_notebook_utils.fisher import get_stencil_noise

            key = (engine, lmax, tolerance)
            if key in cls.measurements:
                return cls.measurements[key]
            production = cls.production
            fiducial = np.array([
                cls.point["w"],
                cls.point["w0pwa"] - cls.point["w"],
                cls.point[bias_parameter],
            ], dtype=np.float64)
            start = time.perf_counter()
            try:
                cls.ci.init_nonlimber_tolerance(gg_tol=tolerance[0],
                                                gs_tol=tolerance[1])
                if engine == "FKEM":
                    cls.ci.init_lowrank_nonlimber_gg(
                        flag=0, nrank=production["lowrank_rank"],
                        lmax=production["lowrank_lmax"])
                else:
                    cls.ci.init_lowrank_nonlimber_gg(
                        flag=1, nrank=NONLIMBER_TOL_SVD_RANK, lmax=lmax)
                # SVD replaces only gg. Give gs the same FKEM cap in
                # both runs so its contribution to the fourth difference
                # cannot change when the gg engine is compared.
                cls.ci.init_lmax_nonlimber(lmax=lmax)
                sigma = {}
                for index, column in enumerate(columns):
                    result = get_stencil_noise(
                        cls._vector_at, index=index, h=NONLIMBER_TOL_STEP,
                        CV=fiducial, AccuracyBoost=1.0, invcov=cls.icov)
                    sigma[column] = result["sigma"]
                # get_stencil_noise evaluates the center last, so the
                # last vector is the fiducial one at this setting
                center = cls.last_vector
                dev = None
                if engine == "FKEM":
                    # C_gg_tomo keeps the Limber value from cap - 1 on, so
                    # the cap goes up by one for the FKEM value at
                    # l = lmax - 1, the last multipole the real-space
                    # path computes with FKEM
                    ells = np.array([lmax - 1.0])
                    cls.ci.init_lmax_nonlimber(lmax=lmax + 1)
                    nonlimber = _cube_diagonal_at_one_ell(
                        cls.ci.C_gg_tomo(l=ells))
                    limber = _cube_diagonal_at_one_ell(
                        cls.ci.C_gg_tomo_limber(l=ells))
                    cls.ci.init_lmax_nonlimber(lmax=lmax)
                    dev = nonlimber/limber - 1.0
            finally:
                cls._restore_production()
            record = {
                "sigma": sigma,
                "center": center,
                "dev": dev,
                "seconds": time.perf_counter() - start,
            }
            cls.measurements[key] = record
            label = f"{tolerance[0]}/{tolerance[1]}"
            if tolerance[0] == 0.0 and tolerance[1] == 0.0:
                label = "0"
            print(f"  measured {engine} cap {lmax}, tolerance {label}: "
                  + ", ".join(f"{c} {sigma[c]:.3e}" for c in columns)
                  + f" ({record['seconds']:.0f} s)", flush=True)
            return record

        def _noise_table(self, rows):
            """Print sigma per column for a list of (label, record)."""
            header = "    " + f"{'setting':34s}" + "".join(
                f"{c:>16s}" for c in columns)
            print(header)
            for label, record in rows:
                values = "".join(f"{record['sigma'][c]:16.3e}"
                                 for c in columns)
                print(f"    {label:34s}{values}")

        def _check_default_tolerances(self):
            """Check the production score and the Python binding defaults.

            A frozen model reproduces its stored chi2. An Evaluate model
            reproduces the native masked quadratic against the survey data.
            The default setter restores the core-default vector bit for bit.
            """
            import numpy as np

            print(f"\n  chi2 = {self.chi2:.6f}; production tolerances "
                  f"(gg, gs) = ({self.production['gg_tol']}, "
                  f"{self.production['gs_tol']}), expected "
                  f"{expected_defaults}")
            self.assertEqual(
                (self.production["gg_tol"], self.production["gs_tol"]),
                expected_defaults,
                "the likelihood applied other production tolerances than "
                "the ones this project's test file expects")
            if expected_lmax_nonlimber is not None:
                self.assertEqual(
                    self.production["lmax_nonlimber"],
                    expected_lmax_nonlimber,
                    "the likelihood applied a different FKEM cap from "
                    "the one this project's test file expects")
            if evaluate_inputs is None:
                reference = self.reference[reference_key]
                print(f"  frozen reference {reference:.6f}, "
                      f"difference {self.chi2 - reference:+.6f}")
                self.assertLess(
                    abs(self.chi2 - reference), u.CHI2_TOLERANCE,
                    f"chi2 = {self.chi2:.6f} vs frozen reference {reference:.6f}")
            else:
                data = np.asarray(self.ci.get_dv_masked())
                residual = self.vector - data
                quadratic = float(residual @ self.icov @ residual)
                np.testing.assert_allclose(self.chi2, quadratic, rtol=1e-12, atol=1e-9)
                boost = self.production["accuracy_boost"]
                divisor = max(1.0, 1.0 + 2.0 * (boost - 1.0))
                expected = np.asarray(expected_defaults) / divisor
                np.testing.assert_array_equal(self.ci.get_nonlimber_tolerance(), expected)
            core = NONLIMBER_TOL_CORE_DEFAULTS
            try:
                self._fiducial_vector()
                self.ci.init_nonlimber_tolerance(gg_tol=core[0],
                                                 gs_tol=core[1])
                explicit = np.array(self.masked_vector())
                self.ci.init_nonlimber_tolerance(gg_tol=0.0, gs_tol=0.0)
                self.masked_vector()
                self.ci.init_nonlimber_tolerance()
                restored = np.array(self.masked_vector())
            finally:
                self._restore_production()
            self.assertTrue(
                np.array_equal(restored, explicit),
                "init_nonlimber_tolerance() with its default arguments "
                f"did not give the vector of the core defaults {core}: "
                "the binding defaults differ from the core's")

        if evaluate_inputs is None:
            test_1_default_reproduces_frozen_reference = _check_default_tolerances
        else:
            test_1_default_reproduces_native_chi2 = _check_default_tolerances

        @classmethod
        def _tolerance_zero_vectors(cls):
            """Evaluate the fiducial, then read it with and without the exit.

            The cosmology stays at the fiducial: only the tolerance
            changes between the reads (core defaults, 0, production),
            so the real-space caches must rebuild on the tolerance key
            alone.

            Returns:
              (fiducial, early, zero, back): the masked vectors at the
              production tolerances, at the core defaults, at 0, and at
              the production values again; computed once per class.
            """
            import numpy as np

            if cls.zero_vector is not None:
                return cls.zero_vector
            production = cls.production
            core = NONLIMBER_TOL_CORE_DEFAULTS
            try:
                fiducial = cls._fiducial_vector()
                cls.ci.init_nonlimber_tolerance(gg_tol=core[0],
                                                gs_tol=core[1])
                early = np.array(cls.masked_vector())
                cls.ci.init_nonlimber_tolerance(gg_tol=0.0, gs_tol=0.0)
                zero = np.array(cls.masked_vector())
                cls.ci.init_nonlimber_tolerance(
                    gg_tol=production["gg_tol"],
                    gs_tol=production["gs_tol"])
                back = np.array(cls.masked_vector())
            finally:
                cls._restore_production()
            cls.zero_vector = (fiducial, early, zero, back)
            return cls.zero_vector

        def test_2_tolerance_round_trip_is_bitwise(self):
            """A detour through other tolerances gives the vector back bitwise."""
            import numpy as np

            (fiducial, early, zero, back) = self._tolerance_zero_vectors()
            self.assertTrue(np.array_equal(fiducial, self.vector),
                            "re-evaluating the fiducial changed the vector")
            self.assertTrue(
                np.array_equal(back, self.vector),
                "returning to the production tolerances did not give the "
                "production vector back bit for bit: a real-space cache "
                "did not rebuild")

        def test_3_tolerance_zero_change(self):
            """Tolerance 0 changes only the non-Limber blocks, by little."""
            import numpy as np

            (fiducial, early, zero, back) = self._tolerance_zero_vectors()
            delta = zero - early
            dchi2 = float(delta @ self.icov @ delta)
            if self.space != "real":
                print(f"\n  Fourier space: tolerance 0 changes "
                      f"{int(np.count_nonzero(delta))} entries "
                      f"(delta^T C^-1 delta = {dchi2:.3e})")
                self.assertTrue(
                    np.array_equal(zero, self.vector),
                    "a Fourier data vector changed with the early-exit "
                    "tolerance, which only the real-space path reads")
                return
            covariance = np.array(self.ci.get_cov_masked())
            sigma = np.sqrt(np.abs(np.diag(covariance)))
            used = sigma > 0.0
            worst = float(np.max(np.abs(delta[used])/sigma[used]))
            sizes = self.ci.compute_data_vector_3x2pt_real_sizes()
            xi_end = int(sizes[0])
            gs_end = xi_end + int(sizes[1])
            gg_end = gs_end + int(sizes[2])
            # each block's own delta^T C^-1 delta, the entries outside it
            # set to zero (so the cross terms between blocks are left out)
            block_scores = {}
            for label, begin, end in (("gamma_t", xi_end, gs_end),
                                      ("w(theta)", gs_end, gg_end)):
                block = np.zeros_like(delta)
                block[begin:end] = delta[begin:end]
                block_scores[label] = float(block @ self.icov @ block)
            print(f"\n  tolerance 0 against the core defaults "
                  f"{NONLIMBER_TOL_CORE_DEFAULTS} (gg, gs): "
                  f"delta^T C^-1 delta = {dchi2:.4e}, max |delta|/sigma "
                  f"= {worst:.4e}; blocks: gamma_t "
                  f"{block_scores['gamma_t']:.3e}, w(theta) "
                  f"{block_scores['w(theta)']:.3e}; bound {change_bound}")
            self.assertGreater(
                dchi2, NONLIMBER_TOL_CHANGE_FLOOR,
                "tolerance 0 did not change the real-space vector: the "
                "setter did not reach the C code, or a cache did not "
                "rebuild")
            self.assertTrue(np.all(delta[:xi_end] == 0.0),
                            "cosmic shear changed with the tolerance")
            self.assertTrue(np.all(delta[gg_end:] == 0.0),
                            "entries after the clustering block changed")
            if int(getattr(self.like, "adopt_limber_gs", 1)) == 1:
                self.assertTrue(np.all(delta[xi_end:gs_end] == 0.0),
                                "Limber gamma_t changed with the tolerance")
            if change_bound is not None:
                self.assertLess(
                    dchi2, change_bound,
                    f"delta^T C^-1 delta = {dchi2:.4e} >= {change_bound}")

        def test_4_stencil_noise_without_early_exit(self):
            """Real space: the stencil noise at the core defaults and at 0."""
            if self.space != "real":
                self.skipTest("the Fourier data vectors never take the "
                              "early exit (test 3: identical vectors)")
            lmax = self.production["lmax_nonlimber"]
            core = NONLIMBER_TOL_CORE_DEFAULTS
            default = self._measure("FKEM", lmax, core)
            zero = self._measure("FKEM", lmax, (0.0, 0.0))
            print(f"\n  stencil noise sigma (h = {NONLIMBER_TOL_STEP}, FKEM "
                  f"cap {lmax}), bound at tolerance 0: {noise_bound}")
            self._noise_table([
                (f"tolerance {core[0]}/{core[1]} (gg/gs)", default),
                ("tolerance 0", zero),
            ])
            if noise_bound is not None:
                for column in columns:
                    self.assertLess(
                        zero["sigma"][column], noise_bound,
                        f"{column}: sigma = {zero['sigma'][column]:.3e} "
                        f"at tolerance 0 is not below {noise_bound}")

        def test_5_fkem_handoff_adds_no_jump(self):
            """FKEM at caps 150 and 200 with tolerance 0."""
            import numpy as np

            records = {}
            for lmax in NONLIMBER_TOL_FKEM_LMAX:
                records[lmax] = self._measure("FKEM", lmax, (0.0, 0.0))
            low = NONLIMBER_TOL_FKEM_LMAX[0]
            high = NONLIMBER_TOL_FKEM_LMAX[-1]
            print(f"\n  FKEM hand-off, tolerance 0 (margin "
                  f"{handoff_margin}, bound {noise_bound}):")
            rows = []
            for lmax in NONLIMBER_TOL_FKEM_LMAX:
                rows.append((f"FKEM cap {lmax}", records[lmax]))
            self._noise_table(rows)
            for lmax in NONLIMBER_TOL_FKEM_LMAX:
                dev = records[lmax]["dev"]
                print(f"    |C_FKEM/C_Limber - 1| at l = {lmax - 1}, per "
                      "lens bin: " + ", ".join(f"{abs(d):.2e}" for d in dev))
            delta = records[high]["center"] - records[low]["center"]
            print(f"    fiducial vector, cap {high} against {low}: "
                  f"delta^T C^-1 delta = "
                  f"{float(delta @ self.icov @ delta):.4e}")
            for column in columns:
                at_low = records[low]["sigma"][column]
                at_high = records[high]["sigma"][column]
                # A larger low-cap fourth difference can flag a hand-off
                # contribution. This empirical one-sided bound does not
                # exclude one: smooth curvature and the added non-Limber
                # multipoles also contribute to the high-cap statistic.
                if handoff_margin is not None:
                    self.assertLessEqual(
                        at_low,
                        (1.0 + handoff_margin)*at_high
                        + NONLIMBER_TOL_NOISE_FLOOR,
                        f"{column}: sigma = {at_low:.3e} at cap {low} "
                        f"exceeds (1 + {handoff_margin}) times its "
                        f"{at_high:.3e} at cap {high}: the low-cap fourth "
                        "difference exceeds the regression bound")
                if noise_bound is not None:
                    self.assertLess(
                        at_low, noise_bound,
                        f"{column}: sigma = {at_low:.3e} at cap {low} is "
                        f"not below {noise_bound}")

        def test_6_svd_noise(self):
            """The low-rank engine's noise at caps 150 and 200, next to FKEM."""
            fkem = {}
            svd = {}
            for lmax in NONLIMBER_TOL_SVD_LMAX:
                fkem[lmax] = self._measure("FKEM", lmax, (0.0, 0.0))
                svd[lmax] = self._measure("SVD", lmax, (0.0, 0.0))
            print(f"\n  SVD (rank {NONLIMBER_TOL_SVD_RANK}) against FKEM, "
                  f"tolerance 0 (rule {svd_rule}, margin {svd_margin}, "
                  f"bound {noise_bound}):")
            rows = []
            for lmax in NONLIMBER_TOL_SVD_LMAX:
                rows.append((f"FKEM cap {lmax}", fkem[lmax]))
                rows.append((f"SVD cap {lmax}", svd[lmax]))
            self._noise_table(rows)
            for lmax in NONLIMBER_TOL_SVD_LMAX:
                for column in columns:
                    value = svd[lmax]["sigma"][column]
                    other = fkem[lmax]["sigma"][column]
                    if svd_rule == "not_larger":
                        self.assertLessEqual(
                            value,
                            (1.0 + svd_margin)*other
                            + NONLIMBER_TOL_NOISE_FLOOR,
                            f"{column}, cap {lmax}: sigma_SVD = "
                            f"{value:.3e} > (1 + {svd_margin}) sigma_FKEM "
                            f"= {other:.3e}")
                    if svd_rule == "bound":
                        self.assertLess(
                            value, noise_bound,
                            f"{column}, cap {lmax}: sigma_SVD = "
                            f"{value:.3e} is not below {noise_bound}")

        def test_7_invalid_tolerance_is_refused(self):
            """A negative or nonfinite tolerance stops with its name.

            The C setter refuses the value with a message and exit(1), so
            each case runs in a fresh python process that imports the
            interface and calls the setter.
            """
            import subprocess
            import sys

            code = (
                "import sys\n"
                f"import {self.ci.__name__} as ci\n"
                "ci.initial_setup()\n"
                "ci.init_nonlimber_tolerance(gg_tol=float(sys.argv[1]), "
                "gs_tol=float(sys.argv[2]))\n"
                "print('accepted')\n")
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(sys.path)
            cases = (
                ("-0.001", "0.01", "gg_tol"),
                ("0.002", "-0.001", "gs_tol"),
                ("nan", "0.01", "gg_tol"),
            )
            for gg_tol, gs_tol, refused in cases:
                completed = subprocess.run(
                    [sys.executable, "-c", code, gg_tol, gs_tol],
                    capture_output=True, text=True, env=environment,
                    timeout=300)
                output = completed.stdout + completed.stderr
                print(f"\n  gg_tol = {gg_tol}, gs_tol = {gs_tol}: return "
                      f"code {completed.returncode}")
                self.assertNotEqual(completed.returncode, 0,
                                    f"gg_tol = {gg_tol}, gs_tol = {gs_tol} "
                                    "was accepted")
                self.assertNotIn("accepted", completed.stdout)
                self.assertIn(f"invalid {refused}", output,
                              f"the refusal does not name {refused}")

        def test_8_accuracy_boost_divides_tolerance(self):
            """The tolerance in use is the setting over 1 + 2 (AB - 1).

            get_nonlimber_tolerance returns what w_gg_tomo and
            w_gammat_tomo use: the setting at AB = 1, a third at AB = 2, a
            fifth at AB = 3, and 0 at every boost when the setting is 0.
            The nonzero setting is the core defaults.
            """
            production = self.production
            settings = NONLIMBER_TOL_CORE_DEFAULTS
            rows = []
            try:
                for boost in (1.0, 2.0, 3.0):
                    self.ci.init_accuracy_boost(
                        accuracy_boost=boost,
                        integration_accuracy=production[
                            "integration_accuracy"],
                        **production["probe_integration_factors"],
                    )
                    self.ci.init_nonlimber_tolerance(gg_tol=settings[0],
                                                     gs_tol=settings[1])
                    effective = self.ci.get_nonlimber_tolerance()
                    self.ci.init_nonlimber_tolerance(gg_tol=0.0, gs_tol=0.0)
                    zero = self.ci.get_nonlimber_tolerance()
                    rows.append((boost, effective, zero))
            finally:
                self._restore_production()
            print("\n  tolerances in use (gg, gs) against the accuracy boost:")
            for boost, effective, zero in rows:
                print(f"    boost {boost}: setting {settings} -> "
                      f"({effective[0]:.6g}, {effective[1]:.6g}); setting "
                      f"0 -> {tuple(zero)}")
                divisor = 1.0
                if boost > 1.0:
                    divisor = 1.0 + 2.0*(boost - 1.0)
                for index in range(2):
                    self.assertAlmostEqual(
                        effective[index], settings[index]/divisor,
                        delta=1.0e-15*settings[index],
                        msg=f"boost {boost}: tolerance in use "
                            f"{effective[index]} != {settings[index]}/"
                            f"{divisor}")
                    self.assertEqual(zero[index], 0.0,
                                     f"boost {boost}: a setting of 0 became "
                                     f"{zero[index]}")

        def test_8_boosted_vector_moves_toward_tolerance_zero(self):
            """At AB = 2 the scaled tolerance brings the vector closer to 0's.

            At each boost the fiducial vector at the core defaults is
            compared with the one at tolerance 0 at the same boost, so the
            other boosted tables cancel and only the switching step stays;
            its whitened size delta^T C^-1 delta is printed at AB = 1 and
            AB = 2. The assertion isolates the scaling at AB = 2: the
            vector at the scaled tolerance (a third) is at least as close
            to the tolerance-0 vector as the one at the unscaled tolerance
            (settings times 3, so the tolerance in use is the core
            default). A smaller tolerance can only move a switch to a
            later block, so fewer multipoles keep the Limber value; the
            distance stays equal when no switch moves. Fourier space:
            every distance is 0 (no early exit).
            """
            import numpy as np

            production = self.production
            core = NONLIMBER_TOL_CORE_DEFAULTS
            distances = {}
            try:
                self._fiducial_vector()
                for boost in (1.0, 2.0):
                    self.ci.init_accuracy_boost(
                        accuracy_boost=boost,
                        integration_accuracy=production[
                            "integration_accuracy"],
                        **production["probe_integration_factors"],
                    )
                    self.ci.init_nonlimber_tolerance(gg_tol=0.0, gs_tol=0.0)
                    zero = np.array(self.masked_vector())
                    # 1 + 2 (AB - 1): the divisor of the scaled tolerance
                    divisor = 1.0 + 2.0*(boost - 1.0)
                    for label, factor in (("scaled", 1.0),
                                          ("unscaled", divisor)):
                        if boost == 1.0 and label == "unscaled":
                            continue
                        self.ci.init_nonlimber_tolerance(
                            gg_tol=core[0]*factor,
                            gs_tol=core[1]*factor)
                        vector = np.array(self.masked_vector())
                        delta = vector - zero
                        distances[(boost, label)] = float(
                            delta @ self.icov @ delta)
            finally:
                self._restore_production()
            print(f"\n  delta^T C^-1 delta to the tolerance-0 vector of the "
                  f"same boost: boost 1: {distances[(1.0, 'scaled')]:.4e}; "
                  f"boost 2, scaled tolerance: "
                  f"{distances[(2.0, 'scaled')]:.4e}, unscaled: "
                  f"{distances[(2.0, 'unscaled')]:.4e}")
            if self.space == "real":
                self.assertLessEqual(
                    distances[(2.0, "scaled")],
                    distances[(2.0, "unscaled")]*(1.0 + 1.0e-9),
                    "the scaled tolerance did not bring the vector closer "
                    "to the one without early exit")
            else:
                for key in distances:
                    self.assertEqual(distances[key], 0.0,
                                     f"a Fourier vector changed with the "
                                     f"tolerance at {key}")

        def test_9_production_state_restored(self):
            """After the suite the fiducial vector is the first one, bitwise."""
            import numpy as np

            vector = self._fiducial_vector()
            self.assertTrue(
                np.array_equal(vector, self.vector),
                "the fiducial vector after the suite differs from the one "
                "before it: a setting was not restored")

    TestNonLimberTolerance.__name__ = name
    TestNonLimberTolerance.__qualname__ = name
    return TestNonLimberTolerance


# ---- the UV-fraction scale cut of w(theta) ----------------------------------

# The cut grid of the w(theta) scale-cut checks, in h/Mpc at z = 0: the
# grid of the example notebooks, (first, last, count) of log-spaced
# values, then one cut far above every scale the angular bins resolve,
# where the UV fraction must be small in every bin.
W_GG_SCALE_CUT_KMAX_RANGE = (0.05, 1.0, 12)
W_GG_SCALE_CUT_KMAX_HIGH = 20.0
W_GG_SCALE_CUT_ALPHA = 0.05

# The three clustering projections the cut runs on, as likelihood
# overrides: Limber at every multipole, the FKEM non-Limber split and the
# low-rank (SVD) non-Limber engine. The cut follows whichever the
# likelihood selected.
W_GG_SCALE_CUT_ENGINES = {
    "Limber": {"adopt_limber_gg": 1, "nonlimber_lowrank_gg": 0},
    "FKEM": {"adopt_limber_gg": 0, "nonlimber_lowrank_gg": 0},
    "SVD": {"adopt_limber_gg": 0, "nonlimber_lowrank_gg": 1},
}

# Window off, the non-Limber driver differs from w_gg_tomo: the Fourier
# entry it uses below the cap has no early exit, while w_gg_tomo switches
# each bin to Limber once |C_l/C_l^limber - 1| < 0.002. The difference
# is relative to the largest |w| of the lens bin, the scale that stays
# finite where w crosses zero at large angles; it stays below 4e-5 on
# FKEM and SVD in the four real-space projects.
W_GG_SCALE_CUT_NONLIMBER_RTOL = 1.0e-4

# The step splits every mode into an IR or a UV mode, and the projection
# is linear, so w_IR + w_UV = w up to rounding on every engine (the
# low-rank engine forms its UV part as full - IR). The tolerance is
# relative to |w_IR| + |w_UV|, the scale of the two summands (w itself
# can cross zero at large angles).
W_GG_SCALE_CUT_PARTITION_RTOL = 1.0e-12
# the cuts of the partition check: below, inside and above the range
# where the notebooks' cuts bite
W_GG_SCALE_CUT_PARTITION_KMAX = (0.1, 0.3, 1.0)

# The Limber cross-check compares the driver's w_IR with a reference
# built from the exact windowed Limber C_l at every integer multipole.
# The driver interpolates its C_l table linearly in ln l above l = 20,
# so their difference is the table's interpolation error. That error
# enters the Legendre sum at the size of its terms, so it is bounded
# relative to S_i = sum_l |Pl[i, l] C_l|, the sum of the absolute
# terms of angular bin i. w itself is no usable scale: at large angles
# it cancels down to ~2e-3 of S_i near its zero crossing, and a bound
# relative to |w| must be loosened everywhere to cover that one place.
# Measured: up to 2.25e-3 of S_i (des_y3, the largest of the four
# real-space projects; the other three stay at or below 1.1e-3);
# doubling the table nodes (accuracyboost 2) shrinks the window-off
# error fourfold, the signature of linear interpolation.
W_GG_SCALE_CUT_LIMBER_RTOL = 1.0e-2
W_GG_SCALE_CUT_LIMBER_KMAX = 0.3   # h/Mpc at z = 0, the notebooks' panel
# multipoles per call of the reference spectra: the Limber batch holds
# per-node work arrays of size (lens bins) x (multipoles) x (nodes), so
# the integers up to lmax ~ 1e5 go through it in chunks
W_GG_SCALE_CUT_ELL_CHUNK = 2048


def _w_gg_scale_cut_kmax():
    """The cut grid of the w(theta) checks, ascending, in h/Mpc at z = 0.

    Returns:
      numpy array: the log-spaced W_GG_SCALE_CUT_KMAX_RANGE grid
      followed by W_GG_SCALE_CUT_KMAX_HIGH.
    """
    import numpy as np
    (first, last, count) = W_GG_SCALE_CUT_KMAX_RANGE
    grid = np.geomspace(first, last, count)
    return np.append(grid, W_GG_SCALE_CUT_KMAX_HIGH)


def _w_gg_legendre_kernel(theta_min_arcmin, theta_max_arcmin, ntheta,
                          lmax):
    """The bin-averaged Legendre kernel of w(theta), coded independently.

    w(theta_i) = sum_l Pl[i, l] C_l, with Pl[i, l] the average of
    (2l + 1)/(4 pi) P_l(cos theta) over angular bin i. With x = cos theta
    the antiderivative of P_l is (P_{l+1} - P_{l-1})/(2l + 1), so

        Pl[i, l] = [P_{l+1}(x_lo) - P_{l+1}(x_hi)
                    - P_{l-1}(x_lo) + P_{l-1}(x_hi)]
                   / (4 pi (x_lo - x_hi)),

    where x_lo and x_hi are the cosines of the bin's smaller and larger
    angle (x_lo > x_hi). The Legendre polynomials come from the upward
    recurrence (l + 1) P_{l+1} = (2l + 1) x P_l - l P_{l-1}, which is
    stable for |x| <= 1; the C code uses GSL instead, so the two kernels
    are independent implementations of the same formula.

    The bin edges are log-spaced between theta_min and theta_max, the
    binning of init_binning.

    Arguments:
      theta_min_arcmin = lower edge of the angular range, arcmin.
      theta_max_arcmin = upper edge of the angular range, arcmin.
      ntheta           = number of angular bins.
      lmax             = one past the last multipole of the sum
                         (Ntable.LMAX, the likelihood key lmax).

    Returns:
      numpy array (ntheta, lmax): Pl[i, l], with Pl[:, 0] = 0 (the sum
      starts at l = 1, as in w_gg_tomo).
    """
    import numpy as np
    arcmin = np.pi/10800.0   # one arcmin in radians
    ln_edges = np.linspace(np.log(theta_min_arcmin*arcmin),
                           np.log(theta_max_arcmin*arcmin), ntheta + 1)
    edges = np.exp(ln_edges)
    x_lo = np.cos(edges[:-1])   # cosine of each bin's smaller angle
    x_hi = np.cos(edges[1:])    # cosine of each bin's larger angle

    # P[l] = P_l at the 2*ntheta edge cosines: columns 0 .. ntheta-1 are
    # x_lo, columns ntheta .. 2*ntheta-1 are x_hi
    x = np.concatenate((x_lo, x_hi))
    P = np.empty((lmax + 1, 2*ntheta))
    P[0] = 1.0
    P[1] = x
    for l in range(1, lmax):
        P[l + 1] = ((2*l + 1)*x*P[l] - l*P[l - 1])/(l + 1)

    # dP[l, i] = P_l(x_lo[i]) - P_l(x_hi[i])
    dP = P[:, :ntheta] - P[:, ntheta:]
    kernel = np.zeros((ntheta, lmax))
    width = 4.0*np.pi*(x_lo - x_hi)
    # rows 2 .. lmax of dP are l + 1 for l = 1 .. lmax - 1, rows
    # 0 .. lmax - 2 are l - 1; .T turns (l, bin) into (bin, l)
    kernel[:, 1:lmax] = (dP[2:lmax + 1] - dP[0:lmax - 1]).T/width[:, None]
    return kernel


def w_gg_scale_cut_suite(u, name, example="example2"):
    """Build the class that checks the UV-fraction scale cut of w(theta).

    The cut (arXiv:2603.13031) splits each clustering spectrum as
    C_l = C_l^IR + C_l^UV with a step on the 3D power spectrum at
    k_max/D(z), projects both parts with the Legendre kernel of
    w(theta), and keeps an angular bin while
    |w_UV|/(|w_UV| + |w_IR|) < alpha. The bindings under test:
    w_gg_kwindow_tomo (the driver: w, w_IR or w_UV), w_gg_scale_cuts_tomo
    (the cut on a k_max grid) and C_gg_kwindow_tomo_ells (the windowed
    spectra at given multipoles, used as a reference).

    The cut runs the projection the likelihood selected. One test method
    per engine of W_GG_SCALE_CUT_ENGINES (Limber, FKEM, SVD) builds the
    frozen fiducial of `example` with that engine and checks:
      1. window off, the driver against the data vector's w_gg_tomo:
         bit for bit on the Limber path, within
         W_GG_SCALE_CUT_NONLIMBER_RTOL on the non-Limber engines;
      2. w_IR (keep_ir = 1) plus an independently computed w_UV
         (keep_ir = 0) equals w to W_GG_SCALE_CUT_PARTITION_RTOL;
      4. at the largest cut, W_GG_SCALE_CUT_KMAX_HIGH (20 h/Mpc), the
         UV fraction is below alpha in every bin;
      5. theta_min does not increase as k_max increases, per lens bin,
         and every row of the mask is the enforced suffix: the kept
         bins of one cut are the largest angles, and each cut keeps
         every bin a smaller cut keeps;
      and that running the cut leaves the model as it was: the next
      ordinary evaluation gives the same data vector bit for bit.
    test_engine_limber_reference adds, on the Limber path:
      3. the driver's w_IR agrees with a reference summed from the
         exact windowed Limber C_l at every integer l and an
         independently coded Legendre kernel, on every angular bin,
         within W_GG_SCALE_CUT_LIMBER_RTOL of sum_l |Pl C_l|, the
         cancellation-free scale of the Legendre sum.
    Runtimes of the driver, the cut and the reference are printed
    (informational, never asserted).

    Arguments:
      u       = the project's cocoa_test_utils module.
      name    = the class name to build (the caller assigns the class
                to this same module-level name).
      example = the key of u.EXAMPLES the checks evaluate; its
                likelihood must compute galaxy clustering in real space.

    Returns:
      a unittest.TestCase subclass named `name`.
    """

    class TestWggScaleCut(unittest.TestCase):
        """The UV-fraction scale cut of w(theta) on the frozen fiducial."""

        # the engine of the last build with its model, point, likelihood
        # and masked data vector: an engine lives in the compiled
        # interface's global state, so switching it means a new model
        _engine = None
        _model = None
        _point = None
        _like = None
        _dv = None

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR and verify the frozen state."""
            u.require_cocoa_environment()
            u.verify_frozen()

        @classmethod
        def _ensure(cls, engine):
            """Leave the interface at the frozen fiducial with this engine.

            Builds and evaluates a new model when the last one used
            another engine; otherwise the interface already holds the
            state.

            Arguments:
              engine = a key of W_GG_SCALE_CUT_ENGINES.

            Returns:
              the likelihood object of the model (its lmax and angular
              binning are the interface's).
            """
            import numpy as np
            if cls._engine == engine:
                return cls._like
            lk_name = u.EXAMPLES[example]["likelihood"]
            print(f"  building model ({example}, NLA, {engine}) ...",
                  flush=True)
            info = u.load_frozen_info(example, tatt=False)
            for key, value in W_GG_SCALE_CUT_ENGINES[engine].items():
                info["likelihood"][lk_name][key] = value
            model = u.make_model(info)
            point = u.build_point(model, example, tatt=False)
            u.evaluate_chi2(model, point)
            ci = _interface(u)
            cls._engine = engine
            cls._model = model
            cls._point = point
            cls._like = model.likelihood[lk_name]
            cls._dv = np.array(_masked_vector_fn(u, ci)())
            return cls._like

        def _engine_checks(self, engine):
            """Checks 1, 2, 4 and 5 and the untouched model, one engine."""
            import time

            import numpy as np
            ci = _interface(u)
            self._ensure(engine)

            # 1. window off against the data vector's w_gg_tomo, read
            # before any cut runs: (ntheta, nbin, nbin), autos on the
            # diagonal
            wgg = np.array(ci.w_gg_tomo())
            start = time.perf_counter()
            full = np.array(ci.w_gg_kwindow_tomo(kmax=-1.0, keep_ir=1))
            elapsed = time.perf_counter() - start
            nbin = full.shape[0]
            print(f"    {engine}: driver {elapsed:.3f} s, {nbin} lens "
                  f"bins x {full.shape[1]} angular bins", flush=True)
            for s in range(nbin):
                reference = wgg[:, s, s]
                if engine == "Limber":
                    self.assertTrue(
                        np.array_equal(full[s], reference),
                        f"{engine}, lens bin {s}: the window-off driver "
                        "differs from w_gg_tomo (max |difference| = "
                        f"{np.max(np.abs(full[s] - reference)):.3e}); "
                        "its fill no longer repeats w_gg_tomo's")
                else:
                    scale = np.max(np.abs(reference))
                    worst = float(np.max(np.abs(full[s] - reference))/scale)
                    print(f"    {engine}, lens bin {s}: max |w - w_gg_tomo|"
                          f"/max|w_gg_tomo| = {worst:.2e}", flush=True)
                    self.assertLess(
                        worst, W_GG_SCALE_CUT_NONLIMBER_RTOL,
                        f"{engine}, lens bin {s}: the window-off driver "
                        f"differs from w_gg_tomo by {worst:.3e} of the "
                        f"bin's largest |w| (limit "
                        f"{W_GG_SCALE_CUT_NONLIMBER_RTOL:.0e})")

            # 2. the partition
            for kmax in W_GG_SCALE_CUT_PARTITION_KMAX:
                w_ir = np.array(ci.w_gg_kwindow_tomo(kmax=kmax, keep_ir=1))
                w_uv = np.array(ci.w_gg_kwindow_tomo(kmax=kmax, keep_ir=0))
                scale = np.abs(w_ir) + np.abs(w_uv)
                residual = np.abs(w_ir + w_uv - full)
                worst = float(np.max(residual/scale))
                print(f"    {engine}, k_max = {kmax}: max |w_IR + w_UV - w|"
                      f"/(|w_IR| + |w_UV|) = {worst:.3e}", flush=True)
                self.assertTrue(
                    np.all(residual <= W_GG_SCALE_CUT_PARTITION_RTOL*scale),
                    f"{engine}, k_max = {kmax}: w_IR + w_UV misses w by "
                    f"{worst:.3e} of |w_IR| + |w_UV| (limit "
                    f"{W_GG_SCALE_CUT_PARTITION_RTOL:.0e})")

            # 4. and 5., on the cut grid
            kmax = _w_gg_scale_cut_kmax()
            start = time.perf_counter()
            cuts = ci.w_gg_scale_cuts_tomo(kmax=kmax,
                                           alpha=W_GG_SCALE_CUT_ALPHA)
            elapsed = time.perf_counter() - start
            print(f"    {engine}: w_gg_scale_cuts_tomo, {len(kmax)} cuts in "
                  f"{elapsed:.2f} s", flush=True)
            uv = np.array(cuts["uv_fraction"])[:, :, -1]
            print(f"    {engine}, k_max = {W_GG_SCALE_CUT_KMAX_HIGH}: max UV "
                  f"fraction = {float(np.max(uv)):.3e}", flush=True)
            self.assertTrue(
                np.all(uv < W_GG_SCALE_CUT_ALPHA),
                f"{engine}, k_max = {W_GG_SCALE_CUT_KMAX_HIGH} h/Mpc: UV "
                f"fraction up to {float(np.max(uv)):.3e}, not below "
                f"alpha = {W_GG_SCALE_CUT_ALPHA}")

            theta_min = np.array(cuts["theta_min"])   # (nbin, nk)
            mask = np.array(cuts["mask"])             # (nbin, ntheta, nk)
            for s in range(theta_min.shape[0]):
                print(f"    {engine}, lens bin {s}: theta_min [arcmin] = "
                      + " ".join(f"{t:.3g}" for t in theta_min[s]),
                      flush=True)
            self.assertTrue(
                np.all(np.diff(theta_min, axis=1) <= 0.0),
                f"{engine}: theta_min grows with k_max in some lens bin: "
                f"{theta_min}")
            # a suffix in theta: along the theta axis a mask row only
            # steps from 0 to 1; nested in k_max: a larger cut keeps at
            # least the bins of a smaller one
            self.assertTrue(
                np.all(np.diff(mask, axis=1) >= 0.0),
                f"{engine}: a mask row is not a suffix in theta")
            self.assertTrue(
                np.all(np.diff(mask, axis=2) >= 0.0),
                f"{engine}: a smaller k_max keeps a bin a larger k_max cuts")

            # the cut leaves the model as it was: the next ordinary
            # evaluation gives the data vector of the build bit for bit
            u.evaluate_chi2(self._model, self._point)
            after = np.array(_masked_vector_fn(u, ci)())
            self.assertTrue(
                np.array_equal(after, self._dv),
                f"{engine}: the data vector after the cut differs from "
                "the one before (max |difference| = "
                f"{np.max(np.abs(after - self._dv)):.3e}); the cut left "
                "state behind")

        def test_engine_fkem(self):
            """FKEM: checks 1, 2, 4 and 5, and the untouched model."""
            self._engine_checks("FKEM")

        def test_engine_limber(self):
            """Limber at every l: checks 1, 2, 4 and 5, and the model."""
            self._engine_checks("Limber")

        def test_engine_limber_reference(self):
            """Check 3: the Limber w_IR against the exact-C_l reference."""
            import time

            import numpy as np
            ci = _interface(u)
            like = self._ensure("Limber")
            kmax = W_GG_SCALE_CUT_LIMBER_KMAX
            lmax = int(like.lmax)

            kernel = _w_gg_legendre_kernel(
                theta_min_arcmin=float(like.theta_min_arcmin),
                theta_max_arcmin=float(like.theta_max_arcmin),
                ntheta=int(like.ntheta), lmax=lmax)

            # the exact windowed Limber C_l at l = 1 .. lmax - 1; column
            # 0 (l = 0) stays zero, as in the driver
            start = time.perf_counter()
            cl = None
            for first in range(1, lmax, W_GG_SCALE_CUT_ELL_CHUNK):
                last = min(first + W_GG_SCALE_CUT_ELL_CHUNK, lmax)
                ells = np.arange(first, last, dtype=float)
                (autos, pairs) = ci.C_gg_kwindow_tomo_ells(
                    kmax=kmax, keep_ir=1, ells=ells)
                autos = np.array(autos)
                if cl is None:
                    cl = np.zeros((autos.shape[0], lmax))
                cl[:, first:last] = autos
            reference = cl @ kernel.T   # (nbin, ntheta)
            elapsed = time.perf_counter() - start
            print(f"    reference: exact C_l at {lmax - 1} multipoles, "
                  f"{elapsed:.1f} s", flush=True)

            w_ir = np.array(ci.w_gg_kwindow_tomo(kmax=kmax, keep_ir=1))
            # the cancellation-free scale of the Legendre sum: the
            # interpolation error of the driver's C_l table enters
            # w_IR at the size of the sum's terms, while w_IR itself
            # cancels toward zero at large angles
            scale = np.abs(cl) @ np.abs(kernel).T
            relative = np.abs(w_ir - reference)/scale
            worst = float(np.max(relative))
            print(f"    k_max = {kmax}: max |w_IR - reference|"
                  f"/sum_l |Pl C_l| = {worst:.3e}", flush=True)
            self.assertLess(
                worst, W_GG_SCALE_CUT_LIMBER_RTOL,
                f"k_max = {kmax}: the driver's w_IR differs from the "
                f"exact-C_l reference by {worst:.3e} of the sum of "
                f"absolute Legendre terms (limit "
                f"{W_GG_SCALE_CUT_LIMBER_RTOL:.0e})")

        def test_engine_svd(self):
            """Low-rank SVD: checks 1, 2, 4 and 5, and the untouched model."""
            self._engine_checks("SVD")

    TestWggScaleCut.__name__ = name
    TestWggScaleCut.__qualname__ = name
    return TestWggScaleCut


# ---- the UV-fraction scale cut of the Fourier C_ss and C_gs -----------------

# The cut grid of the C_ss and C_gs scale-cut checks, in h/Mpc at z = 0:
# (first, last, count) of log-spaced values covering the range where the
# notebooks' cuts bite, then one cut far above every scale the band
# centers resolve, where the UV fraction must be small in every band.
# 50 h/Mpc, not 20: the lensing kernel of the lowest source bin reaches
# k > 20 h/Mpc at the highest band (roman_fourier, l = 3398: UV fraction
# up to 7.5e-2 at 20, 5.1e-3 at 50).
SS_GS_SCALE_CUT_KMAX_RANGE = (0.05, 1.0, 12)
SS_GS_SCALE_CUT_KMAX_HIGH = 50.0
SS_GS_SCALE_CUT_ALPHA = 0.05

# The two galaxy-galaxy lensing projections the cut runs on, as likelihood
# overrides: Limber at every band center and the FKEM non-Limber split.
# The cut follows whichever the likelihood selected. Cosmic shear is
# Limber in every configuration (the core has no non-Limber shear
# engine), so its checks run on whichever model is loaded.
SS_GS_SCALE_CUT_GS_ENGINES = {
    "Limber": {"adopt_limber_gs": 1},
    "FKEM": {"adopt_limber_gs": 0},
}

# The step splits every quadrature node (and every FFTLog summand of the
# FKEM sum) into an IR or a UV contribution, and each spectrum is a
# linear sum of them, so C_IR + C_UV = C up to the rounding of the two
# sums. The tolerance is relative to |C_IR| + |C_UV|, the scale of the
# summands (C_gs can pass through zero).
SS_GS_SCALE_CUT_PARTITION_RTOL = 1.0e-12
# the cuts of the partition check: below, inside and above the range
# where the notebooks' cuts bite
SS_GS_SCALE_CUT_PARTITION_KMAX = (0.1, 0.3, 1.0)

# The Limber cross-check of C_ss compares the driver's IR part, with the
# intrinsic alignment switched off, against an independent numpy Limber
# integral: the lensing efficiency from the project's n(z) by cumulative
# trapezoids, P_NL and the growth factor D from CAMB's interpolators, the
# step at k_max/D(z), all on SS_GS_SCALE_CUT_REFERENCE_NZ redshift nodes.
# The two differ by quadrature and interpolation, not physics. The
# difference is the driver's Gauss-Legendre quadrature of an integrand
# the step makes discontinuous: at the frozen integration_accuracy it
# reaches 4.8e-3 (roman_fourier, level 0) and 2.8e-3 (roman_kl, level 1)
# of |C_IR| on a kept band, and it falls to 8e-4 at level 3 and 1.5e-4 at
# level 6, while the reference moves by 2.6e-4 from 8000 to 16000 nodes.
# The tolerance is relative to |C_IR| on the kept bands.
SS_GS_SCALE_CUT_LIMBER_RTOL = 1.0e-2
SS_GS_SCALE_CUT_LIMBER_KMAX = 0.3   # h/Mpc at z = 0, the notebooks' panel
# redshift nodes of the reference integral, uniform from the n(z) table's
# lower edge to its shifted upper edge
SS_GS_SCALE_CUT_REFERENCE_NZ = 8000


def _ss_gs_scale_cut_kmax():
    """The cut grid of the C_ss and C_gs checks, ascending, h/Mpc at z = 0.

    Returns:
      numpy array: the log-spaced SS_GS_SCALE_CUT_KMAX_RANGE grid
      followed by SS_GS_SCALE_CUT_KMAX_HIGH.
    """
    import numpy as np
    (first, last, count) = SS_GS_SCALE_CUT_KMAX_RANGE
    grid = np.geomspace(first, last, count)
    return np.append(grid, SS_GS_SCALE_CUT_KMAX_HIGH)


def _source_bin_values(values, suffix, nbin):
    """One nuisance value per source bin, read from a parameter table.

    The likelihoods name a per-source-bin nuisance <survey><suffix><i>,
    with i counting from 1 (roman_M1, ROMAN_KL_DZ_S3); a bin without its
    parameter takes 0, as the likelihood does (params.get(p, 0)).

    Arguments:
      values = {parameter name: value}, the sampled point merged with
               the model's fixed parameters.
      suffix = the name part before the bin number, e.g. "_M" (shear
               calibration) or "_DZ_S" (source photo-z shift).
      nbin   = the number of source bins.

    Returns:
      numpy array (nbin,) of the values, bin 0 first.
    """
    import numpy as np
    pattern = re.compile(re.escape(suffix) + r"([0-9]+)$")
    out = np.zeros(nbin)
    for key, value in values.items():
        found = pattern.search(key)
        if found is None:
            continue
        b = int(found.group(1)) - 1
        if 0 <= b < nbin:
            out[b] = float(value)
    return out


def _ss_limber_reference(ci, like, values, kmax, ells, pairs, nz_nodes):
    """The Limber C_ss^IR of pure lensing, coded independently in numpy.

    With the intrinsic alignment off, the shear spectrum of source bins
    i and j is the Limber integral

        C_l^ij = F_l int dchi W_i(chi) W_j(chi)/chi^2
                 P_NL((l + 1/2)/chi, z(chi)) Theta(k <= k_max/D(z)),

    with the curved-sky prefactor F_l = l (l - 1)(l + 1)(l + 2)/(l + 1/2)^4
    of cosmolike's shear spectra, the lensing kernel

        W_i(chi) = (3/2) Omega_m (H0/c)^2 chi (1 + z) g_i(chi),
        g_i(chi) = int_z^zmax dz' n_i(z') (1 - chi/chi(z')),

    and Theta the IR step at the redshift's own threshold. g_i is the
    difference A - chi B of two tail integrals of n_i and n_i/chi, each
    a cumulative trapezoid. n_i(z) is the model's curve: the histogram
    heights placed at the cell centers (the file's z column holds the
    left edges), divided by the histogram area, joined by a natural cubic
    spline (scipy here, GSL in cosmolike), zero outside the first and
    last centers, and evaluated at z - Delta z_i (the photo-z shift).
    P_NL and the growth factor D(z) = sqrt(P_lin(z, k_g)/P_lin(0, k_g))
    at the likelihood's growth_k come from CAMB's interpolators, as the
    likelihood reads them. Units: chi in Mpc/h, k in h/Mpc, P in
    (Mpc/h)^3.

    Arguments:
      ci       = the compiled interface (reads the n(z) file).
      like     = the likelihood object (its provider holds CAMB's
                 results at the evaluated point).
      values   = {parameter name: value} of the evaluated point.
      kmax     = the cut at z = 0 in h/Mpc.
      ells     = numpy array (nell,), the band centers.
      pairs    = numpy array (nspec, 2) of source bins (i, j) per row.
      nz_nodes = number of uniform redshift nodes of the integral.

    Returns:
      numpy array (nspec, nell), C_l^IR of every pair.
    """
    import numpy as np
    from scipy.integrate import cumulative_trapezoid
    from scipy.integrate import trapezoid
    from scipy.interpolate import CubicSpline

    c_over_H0 = 2997.92458   # c/H0 in Mpc/h
    provider = like.provider
    h = provider.get_param("H0")/100.0
    omegam = provider.get_param("omegam")

    # the source n(z) table as cosmolike reads it: column 0 = the left
    # edges of uniform cells, one column per source bin after it
    tables = ci.read_redshift_distributions(
        lens_multihisto_file=like.lens_file,
        lens_ntomo=int(like.lens_ntomo),
        source_multihisto_file=like.source_file,
        source_ntomo=int(like.source_ntomo))
    table = np.array(tables[1], dtype=float)
    nbin = table.shape[1] - 1
    z_left = table[:, 0]
    cell = (z_left[-1] - z_left[0])/(len(z_left) - 1)
    centers = z_left + 0.5*cell
    shifts = _source_bin_values(values, "_DZ_S", nbin)

    # the line-of-sight nodes, from the table's lower edge (cosmolike's
    # Limber range starts there) to its upper edge plus the largest
    # positive shift
    z_low = max(float(z_left[0]), 1.0e-6)
    z_high = float(z_left[-1]) + cell + max(0.0, float(np.max(shifts)))
    z = np.linspace(z_low, z_high, nz_nodes)
    # CAMB gives distances only at the redshifts the likelihood requested
    # (its 1D grid z_interp_1D); a cubic spline through them gives chi on
    # the reference nodes
    chi_on_grid = provider.get_comoving_radial_distance(like.z_interp_1D)*h
    chi = CubicSpline(like.z_interp_1D, chi_on_grid)(z)

    # n_i(z) on the nodes: zero outside the spline's support
    nz = np.zeros((nbin, nz_nodes))
    for b in range(nbin):
        heights = table[:, b + 1]
        density = heights/(np.sum(heights)*cell)
        spline = CubicSpline(centers, density, bc_type="natural")
        z_true = z - shifts[b]
        inside = (z_true > centers[0]) & (z_true < centers[-1])
        nz[b, inside] = spline(z_true[inside])

    # tail integrals int_z^z_high: the full integral minus the running
    # one from z_low (cumulative_trapezoid with initial=0 starts at 0)
    total_n = trapezoid(nz, x=z, axis=1)
    total_q = trapezoid(nz/chi, x=z, axis=1)
    tail_n = total_n[:, None] - cumulative_trapezoid(nz, x=z, axis=1,
                                                     initial=0.0)
    tail_q = total_q[:, None] - cumulative_trapezoid(nz/chi, x=z, axis=1,
                                                     initial=0.0)
    efficiency = tail_n - chi*tail_q
    kernel = 1.5*omegam/c_over_H0**2*chi*(1.0 + z)*efficiency

    extrap_kmax = 2.5e2*float(like.accuracyboost)
    p_nonlinear = provider.get_Pk_interpolator(
        ("delta_tot", "delta_tot"), nonlinear=True,
        extrap_kmin=1e-6, extrap_kmax=extrap_kmax)
    p_linear = provider.get_Pk_interpolator(
        ("delta_tot", "delta_tot"), nonlinear=False,
        extrap_kmin=1e-6, extrap_kmax=extrap_kmax)
    growth_k = float(getattr(like, "growth_k", 0.05))   # 1/Mpc
    growth = np.sqrt(p_linear.P(z, growth_k)/p_linear.P(0.0, growth_k))
    threshold = kmax/growth   # h/Mpc, per node

    out = np.zeros((len(pairs), len(ells)))
    for n, ell in enumerate(ells):
        prefactor = ell*(ell - 1.0)*(ell + 1.0)*(ell + 2.0)/(ell + 0.5)**4
        k = (ell + 0.5)/chi   # h/Mpc
        keep = k <= threshold
        # P is read only on the kept nodes (k <= k_max/D, well inside
        # CAMB's range); the others contribute zero to the IR part
        power = np.zeros(nz_nodes)
        power[keep] = (p_nonlinear.P(z[keep], k[keep]*h, grid=False)
                       *h**3)
        for s in range(len(pairs)):
            (i, j) = (int(pairs[s][0]), int(pairs[s][1]))
            integrand = kernel[i]*kernel[j]/chi**2*power
            out[s, n] = prefactor*trapezoid(integrand, x=chi)
    return out


def fourier_ss_gs_scale_cut_suite(u, name, example="example2"):
    """Build the class that checks the UV-fraction cut of C_ss and C_gs.

    The cut (arXiv:2603.13031) splits each Fourier spectrum as
    C_l = C_l^IR + C_l^UV with a step on the 3D power spectrum at
    k_max/D(z), and keeps a band while |C_l^UV|/(|C_l^UV| + |C_l^IR|) <
    alpha. The bindings under test: C_ss_kwindow_tomo and
    C_gs_kwindow_tomo (the drivers: C, C_IR or C_UV at the data-vector
    band centers) and ss_scale_cuts_tomo and gs_scale_cuts_tomo (the cut
    on a k_max grid).

    Cosmic shear is Limber. Galaxy-galaxy lensing runs the projection the
    likelihood selected; one test method per engine of
    SS_GS_SCALE_CUT_GS_ENGINES (Limber, FKEM) builds the frozen fiducial
    of `example` with that engine. The checks, per probe and gs engine:
      1. window off, the driver against the data-vector path, bit for
         bit: C_ss against the EE spectra of C_ss_tomo_limber at the band
         centers, C_gs against the gs block of the masked data vector
         (each kept entry there is the driver's value times 1 + m, m the
         shear calibration of the pair's source bin);
      2. C_IR (keep_ir = 1) plus an independently computed C_UV
         (keep_ir = 0) equals C to SS_GS_SCALE_CUT_PARTITION_RTOL;
      4. at the largest cut, SS_GS_SCALE_CUT_KMAX_HIGH (50 h/Mpc), the UV
         fraction is below alpha in every band;
      5. l_max does not decrease as k_max increases, per pair, and every
         row of the mask is the enforced prefix: the kept bands of one
         cut are the lowest multipoles, and each cut keeps every band a
         smaller cut keeps;
      and that running the cut leaves the model as it was: the next
      ordinary evaluation gives the same data vector bit for bit.
    test_shear_limber_reference adds:
      3. with the intrinsic alignment off, the driver's C_ss^IR agrees on
         the kept bands with the independent numpy Limber integral of
         _ss_limber_reference, to SS_GS_SCALE_CUT_LIMBER_RTOL.
    Runtimes of the drivers and the cuts are printed (informational,
    never asserted).

    Arguments:
      u       = the project's cocoa_test_utils module.
      name    = the class name to build (the caller assigns the class
                to this same module-level name).
      example = the key of u.EXAMPLES the checks evaluate; its
                likelihood must compute cosmic shear and galaxy-galaxy
                lensing in Fourier space.

    Returns:
      a unittest.TestCase subclass named `name`.
    """

    class TestSsGsScaleCut(unittest.TestCase):
        """The UV-fraction scale cut of C_ss and C_gs on the frozen fiducial."""

        # the gs engine of the last build with its model, point,
        # likelihood and masked data vector: the engine lives in the
        # compiled interface's global state, so switching it means a new
        # model
        _engine = None
        _model = None
        _point = None
        _like = None
        _dv = None

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR and verify the frozen state."""
            u.require_cocoa_environment()
            u.verify_frozen()

        @classmethod
        def _ensure(cls, engine):
            """Leave the interface at the frozen fiducial with this engine.

            Builds and evaluates a new model when the last one used
            another gs engine; otherwise the interface already holds the
            state. engine = None accepts whichever model is loaded (the
            shear checks do not depend on the gs engine) and builds the
            example's own configuration when none is.

            Arguments:
              engine = a key of SS_GS_SCALE_CUT_GS_ENGINES, or None.

            Returns:
              the likelihood object of the model.
            """
            import numpy as np
            if engine is None and cls._model is not None:
                return cls._like
            if engine is not None and cls._engine == engine:
                return cls._like
            lk_name = u.EXAMPLES[example]["likelihood"]
            label = "default gs engine" if engine is None else engine
            print(f"  building model ({example}, NLA, {label}) ...",
                  flush=True)
            info = u.load_frozen_info(example, tatt=False)
            if engine is not None:
                overrides = SS_GS_SCALE_CUT_GS_ENGINES[engine]
                for key, value in overrides.items():
                    info["likelihood"][lk_name][key] = value
            model = u.make_model(info)
            point = u.build_point(model, example, tatt=False)
            u.evaluate_chi2(model, point)
            ci = _interface(u)
            cls._engine = engine
            cls._model = model
            cls._point = point
            cls._like = model.likelihood[lk_name]
            cls._dv = np.array(_masked_vector_fn(u, ci)())
            return cls._like

        def _point_values(self):
            """The evaluated point merged with the model's fixed parameters.

            Returns:
              {parameter name: value} of every sampled and fixed input
              parameter.
            """
            values = dict(self._model.parameterization.constant_params())
            values.update(self._point)
            return values

        def _partition(self, probe, driver, full):
            """Check 2 for one probe: C_IR + C_UV = C at three cuts.

            Arguments:
              probe  = "ss" or "gs" plus the engine, for the messages.
              driver = the binding (C_ss_kwindow_tomo or
                       C_gs_kwindow_tomo).
              full   = numpy array (nspec, nell), the window-off spectra.
            """
            import numpy as np
            for kmax in SS_GS_SCALE_CUT_PARTITION_KMAX:
                c_ir = np.array(driver(kmax=kmax, keep_ir=1))
                c_uv = np.array(driver(kmax=kmax, keep_ir=0))
                scale = np.abs(c_ir) + np.abs(c_uv)
                residual = np.abs(c_ir + c_uv - full)
                # entries with scale = 0 have residual 0 too; the maximum
                # below skips them
                ratio = np.zeros_like(residual)
                nonzero = scale > 0.0
                ratio[nonzero] = residual[nonzero]/scale[nonzero]
                worst = float(np.max(ratio))
                print(f"    {probe}, k_max = {kmax}: max |C_IR + C_UV - C|"
                      f"/(|C_IR| + |C_UV|) = {worst:.3e}", flush=True)
                self.assertTrue(
                    np.all(residual <= SS_GS_SCALE_CUT_PARTITION_RTOL*scale),
                    f"{probe}, k_max = {kmax}: C_IR + C_UV misses C by "
                    f"{worst:.3e} of |C_IR| + |C_UV| (limit "
                    f"{SS_GS_SCALE_CUT_PARTITION_RTOL:.0e})")

        def _cut_grid(self, probe, cutter):
            """Checks 4 and 5 for one probe on the cut grid.

            Arguments:
              probe  = "ss" or "gs" plus the engine, for the messages.
              cutter = the binding (ss_scale_cuts_tomo or
                       gs_scale_cuts_tomo).

            Returns:
              the dict the binding returned.
            """
            import time

            import numpy as np
            kmax = _ss_gs_scale_cut_kmax()
            start = time.perf_counter()
            cuts = cutter(kmax=kmax, alpha=SS_GS_SCALE_CUT_ALPHA)
            elapsed = time.perf_counter() - start
            print(f"    {probe}: {len(kmax)} cuts in {elapsed:.2f} s",
                  flush=True)

            uv = np.array(cuts["uv_fraction"])[:, :, -1]
            print(f"    {probe}, k_max = {SS_GS_SCALE_CUT_KMAX_HIGH}: max "
                  f"UV fraction = {float(np.max(uv)):.3e}", flush=True)
            self.assertTrue(
                np.all(uv < SS_GS_SCALE_CUT_ALPHA),
                f"{probe}, k_max = {SS_GS_SCALE_CUT_KMAX_HIGH} h/Mpc: UV "
                f"fraction up to {float(np.max(uv)):.3e}, not below "
                f"alpha = {SS_GS_SCALE_CUT_ALPHA}")

            lmax = np.array(cuts["lmax"])     # (nspec, nk)
            mask = np.array(cuts["mask"])     # (nspec, nell, nk)
            pairs = np.array(cuts["pairs"])   # (nspec, 2)
            for s in range(lmax.shape[0]):
                print(f"    {probe}, pair ({pairs[s][0]}, {pairs[s][1]}): "
                      "l_max = " + " ".join(f"{v:.0f}" for v in lmax[s]),
                      flush=True)
            self.assertTrue(
                np.all(np.diff(lmax, axis=1) >= 0.0),
                f"{probe}: l_max decreases with k_max in some pair: {lmax}")
            # a prefix in l: along the l axis a mask row only steps from
            # 1 to 0; nested in k_max: a larger cut keeps at least the
            # bands of a smaller one
            self.assertTrue(
                np.all(np.diff(mask, axis=1) <= 0.0),
                f"{probe}: a mask row is not a prefix in l")
            self.assertTrue(
                np.all(np.diff(mask, axis=2) >= 0.0),
                f"{probe}: a smaller k_max keeps a band a larger k_max cuts")
            return cuts

        def _model_untouched(self, probe):
            """The next ordinary evaluation gives the build's data vector.

            Arguments:
              probe = the label of the messages.
            """
            import numpy as np
            ci = _interface(u)
            u.evaluate_chi2(self._model, self._point)
            after = np.array(_masked_vector_fn(u, ci)())
            self.assertTrue(
                np.array_equal(after, self._dv),
                f"{probe}: the data vector after the cut differs from "
                "the one before (max |difference| = "
                f"{np.max(np.abs(after - self._dv)):.3e}); the cut left "
                "state behind")

        def _ggl_checks(self, engine):
            """Checks 1, 2, 4 and 5 of C_gs and the untouched model."""
            import time

            import numpy as np
            ci = _interface(u)
            like = self._ensure(engine)
            probe = f"gs {engine}"

            start = time.perf_counter()
            full = np.array(ci.C_gs_kwindow_tomo(kmax=-1.0, keep_ir=1))
            elapsed = time.perf_counter() - start
            print(f"    {probe}: driver {elapsed:.3f} s, {full.shape[0]} "
                  f"pairs x {full.shape[1]} band centers", flush=True)

            cuts = self._cut_grid(probe, ci.gs_scale_cuts_tomo)
            pairs = np.array(cuts["pairs"])
            self.assertTrue(
                np.array_equal(np.array(cuts["full"]), full),
                f"{probe}: the cut's full spectra differ from the "
                "window-off driver")

            # 1. window off against the gs block of the masked data
            # vector: the block follows cosmic shear, one row of nell
            # entries per ggl pair; a kept entry is C_gs times
            # (1 + m) of its source bin, a masked one is zero
            sizes = ci.compute_data_vector_3x2pt_fourier_sizes()
            start_gs = int(sizes[0])
            nell = full.shape[1]
            nspec = full.shape[0]
            self.assertEqual(
                int(sizes[1]), nspec*nell,
                f"{probe}: the gs block has {int(sizes[1])} entries, the "
                f"driver {nspec} x {nell}")
            calibration = _source_bin_values(self._point_values(), "_M",
                                             int(like.source_ntomo))
            block = self._dv[start_gs:start_gs + nspec*nell]
            block = block.reshape((nspec, nell))
            expected = np.zeros_like(full)
            for s in range(nspec):
                zs = int(pairs[s][1])
                expected[s] = full[s]*(1.0 + calibration[zs])
            kept = block != 0.0
            print(f"    {probe}: window off vs the gs block, "
                  f"{int(kept.sum())} kept entries", flush=True)
            self.assertGreater(
                int(kept.sum()), 0,
                f"{probe}: the gs block is masked entirely, so the "
                "window-off check compares nothing")
            self.assertTrue(
                np.array_equal(block[kept], expected[kept]),
                f"{probe}: the window-off driver differs from the gs "
                "block of the data vector (max |difference| = "
                f"{np.max(np.abs(block[kept] - expected[kept])):.3e}); "
                "its calls no longer repeat the data vector's")

            self._partition(probe, ci.C_gs_kwindow_tomo, full)
            self._model_untouched(probe)

        def test_ggl_fkem(self):
            """C_gs, FKEM: checks 1, 2, 4 and 5, and the untouched model."""
            self._ggl_checks("FKEM")

        def test_ggl_limber(self):
            """C_gs, Limber: checks 1, 2, 4 and 5, and the untouched model."""
            self._ggl_checks("Limber")

        def test_shear(self):
            """C_ss: checks 1, 2, 4 and 5, and the untouched model."""
            import time

            import numpy as np
            ci = _interface(u)
            self._ensure(None)
            probe = "ss"

            start = time.perf_counter()
            full = np.array(ci.C_ss_kwindow_tomo(kmax=-1.0, keep_ir=1))
            elapsed = time.perf_counter() - start
            print(f"    {probe}: driver {elapsed:.3f} s, {full.shape[0]} "
                  f"pairs x {full.shape[1]} band centers", flush=True)

            cuts = self._cut_grid(probe, ci.ss_scale_cuts_tomo)
            pairs = np.array(cuts["pairs"])
            self.assertTrue(
                np.array_equal(np.array(cuts["full"]), full),
                f"{probe}: the cut's full spectra differ from the "
                "window-off driver")

            # 1. window off against C_ss_tomo_limber at the band centers:
            # (EE, BB) cubes (nell, nbin, nbin), the pair (i, j) with
            # i <= j at [:, i, j]; the dict's ell holds the band centers
            # as one column
            ells = np.ascontiguousarray(np.ravel(cuts["ell"]))
            (ee, bb) = ci.C_ss_tomo_limber(l=ells)
            ee = np.array(ee)
            for s in range(full.shape[0]):
                (i, j) = (int(pairs[s][0]), int(pairs[s][1]))
                self.assertTrue(
                    np.array_equal(full[s], ee[:, i, j]),
                    f"{probe}, pair ({i}, {j}): the window-off driver "
                    "differs from C_ss_tomo_limber (max |difference| = "
                    f"{np.max(np.abs(full[s] - ee[:, i, j])):.3e})")

            self._partition(probe, ci.C_ss_kwindow_tomo, full)
            self._model_untouched(probe)

        def test_shear_limber_reference(self):
            """Check 3: the C_ss IR part against the numpy Limber integral."""
            import time

            import numpy as np
            ci = _interface(u)
            like = self._ensure(None)
            # the reference models CAMB's halofit P_NL without baryons,
            # what the likelihood hands cosmolike in these settings
            self.assertEqual(
                int(like.use_emulator), 0,
                "the Limber reference needs the CAMB path (use_emulator "
                "= 0)")
            self.assertEqual(
                int(like.non_linear_emul), 2,
                "the Limber reference needs CAMB's halofit "
                "(non_linear_emul = 2)")
            self.assertFalse(
                bool(getattr(like, "external_baryon_suppression", False)),
                "the Limber reference has no baryon suppression")

            # the intrinsic alignment off: pure lensing (the next
            # evaluation of the model restores it)
            nbin = int(like.source_ntomo)
            zeros = np.zeros(nbin)
            ci.set_nuisance_ia(A1=zeros, A2=zeros, B_TA=zeros)

            kmax = SS_GS_SCALE_CUT_LIMBER_KMAX
            c_ir = np.array(ci.C_ss_kwindow_tomo(kmax=kmax, keep_ir=1))
            cuts = ci.ss_scale_cuts_tomo(kmax=np.array([kmax]),
                                         alpha=SS_GS_SCALE_CUT_ALPHA)
            kept = np.array(cuts["mask"])[:, :, 0] > 0.5
            pairs = np.array(cuts["pairs"])
            ells = np.ravel(cuts["ell"])

            start = time.perf_counter()
            reference = _ss_limber_reference(
                ci=ci, like=like, values=self._point_values(), kmax=kmax,
                ells=ells, pairs=pairs,
                nz_nodes=SS_GS_SCALE_CUT_REFERENCE_NZ)
            elapsed = time.perf_counter() - start
            print(f"    reference: numpy Limber integral on "
                  f"{SS_GS_SCALE_CUT_REFERENCE_NZ} nodes, {elapsed:.1f} s",
                  flush=True)

            relative = np.abs(c_ir - reference)/np.abs(reference)
            worst_kept = float(np.max(relative[kept]))
            worst_all = float(np.max(relative))
            print(f"    k_max = {kmax}: max |C_IR - reference|/|reference| "
                  f"= {worst_kept:.3e} on the {int(kept.sum())} kept bands "
                  f"({worst_all:.3e} on all bands)", flush=True)
            self.assertGreater(
                int(kept.sum()), 0,
                f"k_max = {kmax}: the cut keeps no band, so the "
                "cross-check compares nothing")
            self.assertLess(
                worst_kept, SS_GS_SCALE_CUT_LIMBER_RTOL,
                f"k_max = {kmax}: the driver's C_IR differs from the "
                f"numpy Limber integral by {worst_kept:.3e} on a kept "
                f"band (limit {SS_GS_SCALE_CUT_LIMBER_RTOL:.0e})")

            self._model_untouched("ss reference")

    TestSsGsScaleCut.__name__ = name
    TestSsGsScaleCut.__qualname__ = name
    return TestSsGsScaleCut


# ---- the pair-quadrature accuracy checks ------------------------------------

# The cross-pair Gauss-Legendre rule (gg_pair_glrule in cosmo2D.c) has
# 128 + 64*abs(level) nodes per panel, capped at 8192 nodes from level
# 126 on, where level is the likelihood key integration_accuracy. These
# tolerances judge the pair block (the appended gg-cross entries of the
# data vector) against a converged reference.
#
# The default level is an MCMC setting, so it must pass the reference
# rule of every frozen test, |delta chi2| < CHI2_TOLERANCE (0.2). A
# Fisher run raises integration_accuracy; at the project's Fisher level
# the pair block must sit 200 times deeper, and no kept band power (one
# entry of the pair block) may move by more than 1% of its error bar.
# These are accuracy targets on the FIDUCIAL vector at different
# quadrature settings, under one fixed covariance. A Fisher derivative
# divides the difference of nearby parameter points by a step, so a
# small error at the fiducial does not by itself bound the derivative
# error: derivative stability is a separate check, run with the Fisher
# machinery itself.
PAIR_QUADRATURE_FISHER_DCHI2 = 1.0e-3
PAIR_QUADRATURE_FISHER_BAND = 1.0e-2
# The reference must agree with a rule twice as fine to this level, far
# below every other tolerance here, so that the errors measured against
# it are the tested level's own (measured at or below 1e-9 on thin- and
# wide-bin layouts).
PAIR_QUADRATURE_REFERENCE_DCHI2 = 1.0e-8
# Above this delta chi2 the pair-block error falls as nodes are added;
# below it the error oscillates from one rule to the next (the nodes
# move across the integrands' sharp features), so two levels are
# compared only when one of them is above it.
PAIR_QUADRATURE_FLOOR = 1.0e-6
# Set this environment variable to 1 to run the full ladder sweep
# (test_pair_quadrature_ladder). At the factory's default arguments the
# sweep builds eight models per layout; the routine check builds four.
PAIR_QUADRATURE_DEEP_FLAG = "COCOA_PAIR_QUADRATURE_DEEP"


def _pair_block_scores(delta, icov, sigma, pair, used):
    """The delta chi2 of a vector difference, whole and pair block alone.

    The pair and outside scores zero the complementary entries and
    apply the full precision matrix, so they omit the cross term
    2 delta_pair^T C^-1 delta_outside and need not add up to the
    whole-vector score; they locate where the difference sits, not a
    partition of it. The precision matrix is the masked inverse of
    the full covariance, not the inverse of a separately
    marginalized sub-covariance.

    Arguments:
      delta = theory-vector difference (tested level minus reference),
              the masked layout of the likelihood.
      icov  = the masked inverse covariance (zero rows and columns on
              masked entries).
      sigma = the square root of the covariance diagonal, per entry.
      pair  = boolean mask of the appended gg-cross entries.
      used  = boolean mask of the entries the likelihood keeps.

    Returns:
      (dchi2 of the whole vector, dchi2 of the pair entries alone, the
      largest |delta|/sigma over the kept pair entries, the dchi2 of
      every entry outside the pair block).
    """
    import numpy as np
    in_pairs = np.where(pair, delta, 0.0)
    outside = np.where(pair, 0.0, delta)
    kept = pair & used & (sigma > 0.0)
    band = float(np.max(np.abs(delta[kept])/sigma[kept])) if np.any(kept) else 0.0
    return (float(delta @ icov @ delta), float(in_pairs @ icov @ in_pairs),
            band, float(outside @ icov @ outside))


def pair_quadrature_suite(u, name, example, fisher_level=1,
                          reference_level=30, check_level=62,
                          deep_levels=(2, 3, 6, 14)):
    """Build the pair-quadrature accuracy class of one layout.

    The class evaluates the frozen fiducial of `example` at several
    values of the likelihood key integration_accuracy, in one process,
    and compares the appended gg-cross block (the cross-bin pair
    spectra) with the same block at `reference_level`.

    test_pair_quadrature (always runs; four models at the default
    arguments) asserts:
      1. the level reaches the pair rule: the pair block at level 0
         differs from the reference;
      2. the sign of the level is ignored: level -1 reproduces level 1
         bit for bit, pair block and every other entry;
      3. the default level passes the MCMC rule against the reference,
         |delta chi2| < CHI2_TOLERANCE, for the whole vector and for
         the pair block;
      4. the next level is no worse than the default: its pair-block
         delta chi2 is at most the default's, unless both sit below
         PAIR_QUADRATURE_FLOOR (both converged);
      5. the Fisher level meets the Fisher tolerances: pair-block
         delta chi2 below PAIR_QUADRATURE_FISHER_DCHI2 and every kept
         pair band within PAIR_QUADRATURE_FISHER_BAND of its error.

    test_pair_quadrature_ladder (runs when the environment variable
    COCOA_PAIR_QUADRATURE_DEEP is 1; eight models at the default
    arguments) adds:
      6. the reference is converged: it agrees with `check_level` to
         PAIR_QUADRATURE_REFERENCE_DCHI2 in the pair block;
      7. every level of the sweep passes the MCMC rule, and every level
         from the Fisher level up meets the Fisher tolerances;
      8. above the floor, every finer level is no worse than the
         default.
    It prints the whole table and, when the environment variable
    COCOA_PAIR_QUADRATURE_RESULTS names a folder, writes it there as
    pair_quadrature_<example>.json.

    Arguments:
      u               = the project's cocoa_test_utils module.
      name            = the class name to build (the caller assigns
                        the class to this same module-level name).
      example         = the key of u.EXAMPLES; its frozen dataset must
                        install a gg-cross pair list.
      fisher_level    = the integration_accuracy the project's Fisher
                        runs use.
      reference_level = the level of the reference vector.
      check_level     = the finer level that shows the reference is
                        converged (deep tier).
      deep_levels     = the further levels of the deep sweep.

    Returns:
      a unittest.TestCase subclass named `name`.

    Raises:
      ValueError unless 0 <= fisher_level < reference_level <
      check_level and every deep level lies below reference_level: a
      higher level is a finer quadrature rule, every assertion measures
      a level against the reference, and only the check level (which
      certifies the reference itself) may sit above it.
    """
    if not (0 <= fisher_level < reference_level < check_level):
        raise ValueError(
            f"pair_quadrature_suite: fisher_level = {fisher_level}, "
            f"reference_level = {reference_level}, check_level = "
            f"{check_level}; they must satisfy 0 <= fisher_level < "
            "reference_level < check_level")
    for level in deep_levels:
        if not (0 <= level < reference_level):
            raise ValueError(
                f"pair_quadrature_suite: deep level {level} must lie in "
                f"[0, reference_level = {reference_level})")

    class TestPairQuadrature(unittest.TestCase):
        """The cross-pair Gauss-Legendre ladder on one frozen layout."""

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR and verify the frozen state."""
            u.require_cocoa_environment()
            u.verify_frozen()

        def _vectors(self, levels):
            """Evaluate the frozen fiducial at every level, in order.

            Arguments:
              levels = the integration_accuracy values to evaluate.

            Returns:
              (dict level -> masked theory vector, the masked inverse
              covariance, sigma per entry, the pair-block mask, the
              kept-entry mask).
            """
            import numpy as np
            ci = _interface(u)
            # the interface alone decides this: check it before any build
            if not hasattr(ci, "compute_data_vector_3x2pt_fourier_sizes"):
                raise ValueError(
                    f"{example}: the pair quadrature suite needs a Fourier "
                    "project (compute_data_vector_3x2pt_fourier_sizes)")
            masked_vector = _masked_vector_fn(u, ci)
            lk_name = u.EXAMPLES[example]["likelihood"]
            vectors = {}
            icov = sigma = pair = used = None
            for level in levels:
                print(f"  building model ({example}, NLA, "
                      f"integration_accuracy = {level}) ...", flush=True)
                info = u.load_frozen_info(example, tatt=False)
                info["likelihood"][lk_name]["integration_accuracy"] = level
                model = u.make_model(info)
                point = u.build_point(model, example, tatt=False)
                u.evaluate_chi2(model, point)
                vectors[level] = np.array(masked_vector())
                if icov is None:
                    icov = np.array(ci.get_inv_cov_masked())
                    # masked rows of the covariance are zero; the clip
                    # keeps a round-off negative diagonal entry out of
                    # the square root (the kept mask drops sigma = 0)
                    sigma = np.sqrt(np.clip(np.diag(
                        np.array(ci.get_cov_masked())), 0.0, None))
                    sizes = ci.compute_data_vector_3x2pt_fourier_sizes()
                    # the cross-pair list is installed by the first
                    # model build, so this check cannot run earlier
                    ggx = _ggx_len(ci)
                    if ggx == 0:
                        raise ValueError(
                            f"{example}: the frozen dataset installs no "
                            "gg-cross pair list, so there is no pair "
                            "block to test")
                    start = int(sizes[0]) + int(sizes[1]) + int(sizes[2])
                    pair = np.zeros(len(vectors[level]), dtype=bool)
                    pair[start:start + ggx] = True
                    used = icov.diagonal() > 0.0
            return vectors, icov, sigma, pair, used

        def _report(self, vectors, icov, sigma, pair, used, levels):
            """Score every level against the reference and print the table.

            Returns:
              dict level -> (dchi2 whole, dchi2 pair block, max band
              |delta|/sigma, dchi2 outside the pair block).
            """
            ref = vectors[reference_level]
            scores = {}
            print(f"\n  pair-quadrature report ({example}, reference "
                  f"level {reference_level} = "
                  f"{128 + 64*abs(reference_level)} nodes per panel):")
            print("    level  nodes  dchi2 whole   dchi2 pairs   "
                  "max |dd|/sigma  dchi2 outside")
            for level in levels:
                if level == reference_level:
                    continue
                score = _pair_block_scores(vectors[level] - ref, icov,
                                           sigma, pair, used)
                scores[level] = score
                print(f"    {level:5d}  {128 + 64*abs(level):5d}  "
                      f"{score[0]:.3e}     {score[1]:.3e}     "
                      f"{score[2]:.3e}       {score[3]:.3e}")
            return scores

        def _assert_fisher(self, scores, level):
            """Assert the Fisher tolerances at one level."""
            self.assertLess(
                scores[level][1], PAIR_QUADRATURE_FISHER_DCHI2,
                f"{example}: level {level} leaves pair-block delta chi2 "
                f"= {scores[level][1]:.3e}, above the Fisher tolerance "
                f"{PAIR_QUADRATURE_FISHER_DCHI2:.1e}; a Fisher run needs "
                "a higher integration_accuracy")
            self.assertLess(
                scores[level][2], PAIR_QUADRATURE_FISHER_BAND,
                f"{example}: level {level} moves a pair band by "
                f"{scores[level][2]:.3e} of its error, above "
                f"{PAIR_QUADRATURE_FISHER_BAND:.0e}")

        def test_pair_quadrature(self):
            """Default, next and Fisher levels against the reference (checks 1-5)."""
            import numpy as np
            levels = sorted({0, 1, fisher_level}) + [-1, reference_level]
            vectors, icov, sigma, pair, used = self._vectors(levels)
            scores = self._report(vectors, icov, sigma, pair, used,
                                  [level for level in levels if level != -1])

            self.assertFalse(
                np.array_equal(vectors[0][pair], vectors[reference_level][pair]),
                f"{example}: the pair block is identical at levels 0 and "
                f"{reference_level}: integration_accuracy does not reach "
                "the pair rule")
            self.assertTrue(
                np.array_equal(vectors[-1], vectors[1]),
                f"{example}: integration_accuracy = -1 does not reproduce "
                "level 1; the pair rule must read only abs(level)")
            self.assertLess(
                abs(scores[0][0]), u.CHI2_TOLERANCE,
                f"{example}: the default level misses the MCMC rule on the "
                f"whole vector: delta chi2 = {scores[0][0]:.3e}")
            self.assertLess(
                abs(scores[0][1]), u.CHI2_TOLERANCE,
                f"{example}: the default level misses the MCMC rule on the "
                f"pair block: delta chi2 = {scores[0][1]:.3e}")
            if max(scores[0][1], scores[1][1]) > PAIR_QUADRATURE_FLOOR:
                self.assertLessEqual(
                    scores[1][1], scores[0][1],
                    f"{example}: level 1 ({scores[1][1]:.3e}) is worse "
                    f"than level 0 ({scores[0][1]:.3e}) in the pair block")
            self._assert_fisher(scores, fisher_level)

        @unittest.skipUnless(
            os.environ.get(PAIR_QUADRATURE_DEEP_FLAG) == "1",
            f"set {PAIR_QUADRATURE_DEEP_FLAG}=1 for the full ladder sweep")
        def test_pair_quadrature_ladder(self):
            """The full ladder sweep (checks 6-8)."""
            import json
            levels = (sorted({0, 1, fisher_level, *deep_levels})
                      + [reference_level, check_level])
            vectors, icov, sigma, pair, used = self._vectors(levels)
            scores = self._report(vectors, icov, sigma, pair, used, levels)

            self.assertLess(
                scores[check_level][1], PAIR_QUADRATURE_REFERENCE_DCHI2,
                f"{example}: the reference (level {reference_level}) and "
                f"level {check_level} differ by pair-block delta chi2 = "
                f"{scores[check_level][1]:.3e}; the reference is not "
                "converged")
            sweep = [level for level in levels
                     if level not in (reference_level, check_level)]
            for level in sweep:
                self.assertLess(
                    abs(scores[level][0]), u.CHI2_TOLERANCE,
                    f"{example}: level {level} misses the MCMC rule: "
                    f"delta chi2 = {scores[level][0]:.3e}")
                if level >= fisher_level:
                    self._assert_fisher(scores, level)
            for fine in sweep:
                if fine <= 0:
                    continue
                if max(scores[0][1], scores[fine][1]) > PAIR_QUADRATURE_FLOOR:
                    self.assertLessEqual(
                        scores[fine][1], scores[0][1],
                        f"{example}: level {fine} ({scores[fine][1]:.3e}) "
                        f"is worse than the default level 0 "
                        f"({scores[0][1]:.3e}) in the pair block")

            folder = os.environ.get("COCOA_PAIR_QUADRATURE_RESULTS")
            if folder:
                os.makedirs(folder, exist_ok=True)
                path = os.path.join(folder, f"pair_quadrature_{example}.json")
                with open(path, "w") as handle:
                    json.dump({
                        "example": example,
                        "reference_level": reference_level,
                        "check_level": check_level,
                        "nodes_per_panel": {str(level): 128 + 64*abs(level)
                                            for level in levels},
                        "columns": ["dchi2_whole", "dchi2_pairs",
                                    "max_band_over_sigma", "dchi2_outside"],
                        "scores": {str(level): list(score)
                                   for level, score in scores.items()},
                    }, handle, indent=1)
                print(f"  wrote {path}")

    TestPairQuadrature.__name__ = name
    TestPairQuadrature.__qualname__ = name
    return TestPairQuadrature


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
    ("nz_source_modes", re.compile(r"_U_S[0-9]+$")),
    ("nz_lens_modes", re.compile(r"_U_L[0-9]+_[0-9]+$")),
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
    # The n(z) coefficients are in units of a standard-normal draw. The
    # furthest point, the four-step scramble, moves each mode by 0.20.
    "nz_source_modes": {re.compile(r"_U_S[0-9]+$"): 0.05},
    "nz_lens_modes": {re.compile(r"_U_L[0-9]+_[0-9]+$"): 0.05},
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
                       phases=LADDER_PHASES, *, evaluate_inputs=None, interface=None):
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
      evaluate_inputs = optional callable receiving example and tatt
                keywords and returning a Cobaya configuration and
                fiducial. Supply this for a ladder on live Evaluate inputs.
      interface = the compiled survey module when evaluate_inputs is supplied.

    Returns:
      a unittest.TestCase subclass named `name`.
    """

    if (evaluate_inputs is None) != (interface is None):
        raise ValueError("supply evaluate_inputs and interface together")
    if evaluate_inputs is None:
        evaluate_chi2 = u.evaluate_chi2
        require_cocoa_environment = u.require_cocoa_environment
    else:
        from cocoa_testing import evaluate_chi2, require_cocoa_environment

    class TestCacheLadder(unittest.TestCase):
        """Sector-ladder cache check at the configuration's fiducial."""

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR and verify frozen inputs when supplied."""
            require_cocoa_environment()
            if evaluate_inputs is None:
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
            ci = _interface(u) if evaluate_inputs is None else interface
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
            gglpairs = []
            if sizes[1] > 0:
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
            ci = _interface(u) if evaluate_inputs is None else interface
            masked_vector = _masked_vector_fn(u, ci)

            results = {}
            for order in ("forward", "mirrored"):
                if evaluate_inputs is None:
                    lk_name = u.EXAMPLES[example]["likelihood"]
                    info = u.load_frozen_info(example, tatt=tatt)
                    model = u.make_model(info)
                    fid = dict(u.build_point(model, example, tatt=tatt))
                else:
                    model, lk_name, fid = _make_evaluate_model(
                        evaluate_inputs=evaluate_inputs, example=example, tatt=tatt)
                    self.addCleanup(model.close)
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
                if evaluate_inputs is not None:
                    # The endpoints bound the independent scalar priors along
                    # the ladder. The evaluations also check the coupled priors.
                    scramble = dict.fromkeys(self.sector_deltas, LADDER_SCRAMBLE_STEP)
                    for point in (fid, self._point_at(fid, scramble)):
                        names = model.parameterization.sampled_params()
                        values = np.array([point[name] for name in names])
                        self.assertTrue(
                            np.isfinite(model.prior.logps_internal(values)),
                            f"the ladder endpoint {point} must lie in the sampled prior")
                evaluate_chi2(model, self._point_at(fid, steps))
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
                        evaluate_chi2(model, point)
                        dv = np.array(masked_vector())
                        self.assertFalse(
                            np.array_equal(dv, prev),
                            f"{order}: {sector} step {r} left the "
                            "data vector unchanged (dead sector flag, "
                            "or a sector cache that did not rebuild)")
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
                final_chi2 = evaluate_chi2(model, final_point)
                final_dv = np.array(masked_vector())

                # no-op probe: identical point again, bitwise
                evaluate_chi2(model, dict(final_point))
                self.assertTrue(
                    np.array_equal(
                        np.array(masked_vector()),
                        final_dv),
                    f"{order}: a no-op re-evaluation changed the "
                    "data vector")

                # scramble: every sector at once, bias included
                scr = {s: LADDER_SCRAMBLE_STEP
                       for s in self.sector_deltas}
                evaluate_chi2(model, self._point_at(fid, scr))

                # return: the ladder's final point must reproduce
                # bitwise
                back_chi2 = evaluate_chi2(model, final_point)
                back_dv = np.array(masked_vector())
                self.assertTrue(
                    np.array_equal(back_dv, final_dv),
                    f"{order}: returning after the scramble did not "
                    "reproduce the data vector bit for bit (a "
                    "sector cache that did not rebuild)")
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
            if evaluate_inputs is None and not getattr(u, "TATT_POINT", None):
                self.skipTest(
                    "the project declares no TATT parameters")
            self._run_ladder(tatt=True)

    TestCacheLadder.__name__ = name
    TestCacheLadder.__qualname__ = name
    return TestCacheLadder


# ---- the setting-sequence checks --------------------------------------------

def nonlimber_endpoint_suite(u, name, example, project_endpoint,
                            tatt=True, cluster=False, galaxy_lensing=True):
    """Check endpoint validation, reset behavior and cached projections.

    The endpoint sets the largest redshift of the FFTLog distance grid.
    It does not set the number of nodes. Each numerical sequence starts
    at zmax = 4, moves to 5, then 4.5, then returns to 4. The shared
    sequence tests compare the intermediate and restored vectors against
    fresh processes, bit for bit. Galaxy tests exercise FKEM and rank-5
    SVD clustering, with NLA and, when supported, TATT. A cluster test
    reads the joint cluster vector with both cluster projections enabled.

    Arguments:
      u = the project's cocoa_test_utils module.
      name = name assigned to the returned unittest class.
      example = frozen example containing the lens and source samples.
      project_endpoint = endpoint explicitly selected by the project YAML.
      tatt = whether this example has a TATT variant.
      cluster = whether to read the joint cluster vector.
      galaxy_lensing = whether the likelihood supports adopt_limber_gs.
                       Shear-plus-clustering layouts can omit this probe.
    Returns:
      A unittest.TestCase subclass. Invalid settings run in child processes
      because the C++ initialization contract exits on invalid input.
    """
    sequences = []
    variants = [False]
    if tatt:
        variants.append(True)
    for use_tatt in variants:
        for engine in [0, 1]:
            suffix = "tatt" if use_tatt else "nla"
            overrides = {
                "nonlimber_zmax": 4.0,
                "adopt_limber_gg": 0,
                "nonlimber_lowrank_gg": engine,
                "nonlimber_lowrank_rank": 5,
                "nonlimber_lowrank_lmax": 200,
                "lmax_nonlimber": 200,
                "nonlimber_tolerance_gg": 0.0,
                "nonlimber_tolerance_gs": 0.0,
            }
            if galaxy_lensing:
                overrides["adopt_limber_gs"] = 0
            evaluator = "masked"
            if cluster:
                overrides["cluster_adopt_limber_cc"] = 0
                overrides["cluster_adopt_limber_cg"] = 0
                evaluator = "cluster"
            steps = []
            for endpoint in [5.0, 4.5, 4.0]:
                operation = ("call", {
                    "name": "init_nonlimber_zmax",
                    "zmax": endpoint,
                })
                steps.append([operation])
            sequences.append({
                "name": f"endpoint_{suffix}_engine{engine}",
                "base": example,
                "tatt": use_tatt,
                "overrides": overrides,
                "steps": steps,
                "evaluate": evaluator,
                "compare": [1, 2, 3],
                "expect": "bitwise",
            })
    numerical = setting_sequence_suite(u=u, name=name, sequences=sequences)

    class TestEndpoint(numerical):
        """The radial endpoint is a validated, resettable numerical setting."""

        def test_default_and_accuracy_independence(self):
            """Reset restores 4 and sampling boosts preserve an explicit endpoint."""
            ci = _interface(u)
            ci.initial_setup()
            self.assertEqual(ci.get_nonlimber_zmax(), 4.0)
            ci.init_nonlimber_zmax(zmax=5.0)
            ci.init_accuracy_boost(accuracy_boost=2.0, integration_accuracy=3)
            ci.init_nonlimber_accuracy_boost(nonlimber_boost=2.0)
            self.assertEqual(ci.get_nonlimber_zmax(), 5.0)
            ci.initial_setup()
            self.assertEqual(ci.get_nonlimber_zmax(), 4.0)

        def test_project_yaml_endpoint(self):
            """An omitted example override receives the explicit project preset."""
            info = u.load_frozen_info(example=example, tatt=False)
            likelihood_name = u.EXAMPLES[example]["likelihood"]
            # Remove any explicit value so this check tests the live YAML.
            info["likelihood"][likelihood_name].pop("nonlimber_zmax", None)
            model = u.make_model(info=info)
            try:
                endpoint = _interface(u).get_nonlimber_zmax()
                self.assertEqual(endpoint, project_endpoint)
            finally:
                model.close()

        def test_notebook_endpoint_validation(self):
            """A rejected notebook setting preserves its previous configuration."""
            import importlib.util
            from pathlib import Path

            native = _interface(u)
            project = native.__name__[len("cosmolike_"):-len("_interface")]
            module_name = "cosmolike_" + project + "_notebook_wrappers"
            directory = Path(os.environ["ROOTDIR"]) / "projects" / project
            filename = directory / "interface" / (module_name + ".py")
            spec = importlib.util.spec_from_file_location(
                name=module_name, location=filename)
            wrapper = importlib.util.module_from_spec(spec=spec)
            spec.loader.exec_module(module=wrapper)
            for endpoint in [4.0, 4.5, 5.0]:
                wrapper.configure(nonlimber_zmax=endpoint)
                self.assertEqual(wrapper._CONFIG["nonlimber_zmax"], endpoint)
                self.assertEqual(native.get_nonlimber_zmax(), endpoint)
            for value in [3.999, 3.0, 0.0, -1.0,
                          float("nan"), float("inf"), -float("inf")]:
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "finite value >= 4.0"):
                        wrapper.configure(nonlimber_zmax=value)
                    self.assertEqual(wrapper._CONFIG["nonlimber_zmax"], 5.0)
                    self.assertEqual(native.get_nonlimber_zmax(), 5.0)

        def test_endpoint_changes_projection(self):
            """The two endpoints must produce different finite predictions."""
            import numpy as np

            info = u.load_frozen_info(example=example, tatt=False)
            likelihood_name = u.EXAMPLES[example]["likelihood"]
            block = info["likelihood"][likelihood_name]
            block["nonlimber_zmax"] = 4.0
            block["adopt_limber_gg"] = 0
            if galaxy_lensing:
                block["adopt_limber_gs"] = 0
            if cluster:
                block["cluster_adopt_limber_cc"] = 0
                block["cluster_adopt_limber_cg"] = 0
            model = u.make_model(info=info)
            try:
                point = u.build_point(model=model, example=example, tatt=False)
                u.evaluate_chi2(model=model, point=point)
                ci = _interface(u)
                evaluate = _masked_vector_fn(u, ci)
                if cluster:
                    evaluate = ci.compute_data_vector_cluster_masked
                first = np.asarray(evaluate()).copy()
                ci.init_nonlimber_zmax(zmax=5.0)
                second = np.asarray(evaluate()).copy()
                self.assertTrue(np.isfinite(first).all())
                self.assertTrue(np.isfinite(second).all())
                self.assertFalse(np.array_equal(first, second))
                ci.init_nonlimber_zmax(zmax=4.0)
                restored = np.asarray(evaluate()).copy()
                self.assertTrue(np.array_equal(first, restored))
            finally:
                model.close()

        def test_invalid_endpoints(self):
            """Reject values below four, NaN and either infinity before prediction."""
            import subprocess
            import sys

            ci = _interface(u)
            directory = os.path.dirname(os.path.abspath(ci.__file__))
            module = ci.__name__
            code = (
                "import importlib, sys\n"
                "sys.path.insert(0, sys.argv[1])\n"
                "ci = importlib.import_module(sys.argv[2])\n"
                "ci.initial_setup()\n"
                "ci.init_nonlimber_zmax(zmax=float(sys.argv[3]))\n"
            )
            for value in ["3.999", "3", "0", "-1", "nan", "inf", "-inf"]:
                with self.subTest(value=value):
                    command = [
                        sys.executable, "-B", "-c", code,
                        directory, module, value,
                    ]
                    result = subprocess.run(args=command, capture_output=True,
                                            text=True, check=False)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("must be finite and >= 4.0",
                                  result.stdout + result.stderr)

    TestEndpoint.__name__ = name
    TestEndpoint.__qualname__ = name
    return TestEndpoint


def common_sequences(space, shear_example=None, lens_example=None,
                     nsource=0, nlens=0, tatt=False, rebuild_example=None,
                     baryons=None, lens_bins=True):
    """The setting sequences every project with these examples can run.

    Each sequence reproduces one class of notebook-safety gap (the
    catalog of the notebook-safety plan): bin counts changed in one
    process (S1, S2, and S2-without-recount, which must stop with a
    named error), the IA model (S6), the bias model (S7), the run mode
    (S8), a baryon scenario loaded and reset (S9), a second model built
    after initial_setup (S10), init_data called twice (S11), and a
    binning or probe change after init_data (S12, a named error).

    Arguments:
      space           = "real" or "fourier", the examples' binning space.
      shear_example   = an EXAMPLES key whose evaluation computes cosmic
                        shear (S1, S6, S8), or None.
      lens_example    = an EXAMPLES key with lens bins in its vector
                        (S2, S7, S9, S11, S12), or None.
      nsource, nlens  = the source and lens bin counts of those examples.
      tatt            = True when shear_example has a TATT variant (S6
                        starts from the TATT point, whose second-order
                        amplitudes are not zero).
      rebuild_example = the EXAMPLES key S10 builds after lens_example,
                        or None to skip S10.
      baryons         = (scenario, HDF5 library file in the dataset
                        folder) for S9 and S10, or None.
      lens_bins       = False skips S2 (a gg cross-pair list names lens
                        bins, so the lens count cannot change under it).

    Returns:
      a list of sequence dictionaries (setting_sequence_suite).
    """
    if space == "real":
        shear = ["xi_pm_tomo"]
        lens = ["w_gg_tomo", "w_gammat_tomo"]
    else:
        shear = ["C_ss_tomo_limber"]
        lens = ["C_gg_tomo", "C_gs_tomo_limber"]
    sequences = []
    if shear_example is not None:
        sequences.append({
            "name": "S1_source_bins", "base": shear_example,
            "steps": [[("source_bins", {"nbin": nsource - 1})],
                      [("source_bins", {"nbin": nsource + 1})],
                      [("source_bins", {"nbin": nsource})]],
            "evaluate": shear, "fresh": "base_first", "compare": [2, 3],
            "expect": "bitwise"})
        sequences.append({
            "name": "S8_runmode_round_trip", "base": shear_example,
            "steps": [[("call", {"name": "init_cosmo_runmode",
                                 "is_linear": True})],
                      [("call", {"name": "init_cosmo_runmode",
                                 "is_linear": False})]],
            "evaluate": "masked", "compare": [1, 2], "expect": "bitwise"})
        if tatt:
            sequences.append({
                "name": "S6_ia_model_round_trip", "base": shear_example,
                "tatt": True,
                "steps": [[("ia_model", {"model": 0})],
                          [("ia_model", {"model": 1})]],
                "evaluate": "masked", "compare": [1, 2],
                "expect": "bitwise"})
    if lens_example is not None:
        recount = [("call", {"name": "init_adopt_limber_gg",
                             "adopt_limber_gg": 0}),
                   ("lens_bins", {"nbin": nlens - 1})]
    if lens_example is not None and lens_bins:
        sequences.append({
            "name": "S2_lens_bins", "base": lens_example,
            "steps": [recount, [("lens_bins", {"nbin": nlens + 1})]],
            "evaluate": lens, "fresh": "base_first", "compare": [1, 2],
            "expect": "bitwise"})
        if space == "real":
            sequences.append({
                "name": "S2_lens_bins_without_recount", "base": lens_example,
                "steps": [recount,
                          [("lens_bins", {"nbin": nlens + 1,
                                          "recount": False})]],
                "evaluate": "w_gg_tomo", "fresh": "base_first",
                "expect": "fatal:call init_ntomo_powerspectra"})
    if lens_example is not None:
        sequences.append({
            "name": "S7_bias_model_round_trip", "base": lens_example,
            "steps": [[("bias_b1", {"code": 2})], [("bias_b1", {})]],
            "evaluate": "masked", "compare": [1, 2], "expect": "bitwise"})
        sequences.append({
            "name": "S11_init_data_twice", "base": lens_example,
            "steps": [[("init_data", {"drop": 40})],
                      [("init_data", {"drop": 120})]],
            "evaluate": "chi2", "compare": [1, 2], "expect": "bitwise"})
        sequences.append({
            "name": "S12_binning_change_after_init_data",
            "base": lens_example, "steps": [[("halve_binning", {})]],
            "evaluate": "masked", "expect": "fatal:re-run init_data"})
        sequences.append({
            "name": "S12_probe_change_after_init_data",
            "base": lens_example,
            "steps": [[("call", {"name": "init_probes",
                                 "possible_probes": "xi"})]],
            "evaluate": "masked", "expect": "fatal:re-run init_data"})
        if baryons is not None:
            load = ("baryons", {"sim": baryons[0], "allsims": baryons[1]})
            sequences.append({
                "name": "S9_baryons_load_and_reset", "base": lens_example,
                "steps": [[load], [("call", {"name": "reset_bary_struct"})]],
                "evaluate": "masked", "compare": [1, 2],
                "expect": "bitwise"})
        if rebuild_example is not None:
            # the first model turns settings on (linear P(k), a baryon
            # scenario); the rebuild then constructs a second model in
            # the same process, whose initialize runs initial_setup, so
            # any first-model state that survives the reset shows up
            # against a fresh build of the second model
            setup = [("call", {"name": "init_cosmo_runmode",
                               "is_linear": True})]
            if baryons is not None:
                setup.append(("baryons", {"sim": baryons[0],
                                          "allsims": baryons[1]}))
            setup.append(("rebuild", {"example": rebuild_example}))
            sequences.append({
                "name": "S10_second_model_after_initial_setup",
                "base": lens_example, "steps": [setup],
                "evaluate": "masked", "expect": "bitwise"})
    return sequences


def setting_sequence_suite(u, name, sequences):
    """Build the class that changes settings in one process.

    A notebook changes cosmolike settings between evaluations without
    restarting the kernel. Every such change must give the numbers a
    process seeing only the final settings gives. Each sequence is a
    plain dictionary:

      {"name": str, "base": EXAMPLES key, "overrides": optional
       {likelihood key: value} of the base build,
       "steps": [[(op, kwargs), ...], ...], "evaluate": an evaluator
       name or a list of them, "expect": "bitwise" or "fatal:<text>",
       "compare": optional step numbers whose vectors are compared
       (default: the last), "fresh": optional "ops_first" or
       "base_first", "fresh_steps": optional steps of the fresh leg
       (default: "steps"), "check": optional (check name, kwargs)}

    The operations, evaluators and checks are the SEQUENCE_OPS,
    SEQUENCE_EVALUATORS and SEQUENCE_CHECKS tables of cocoa_testing.py;
    _sequence_worker documents the two legs. For "bitwise" the class
    runs the in-process leg once and one fresh leg per compared step,
    each in its own worker, and asserts exact float64 equality. For
    "fatal:<text>" the in-process leg must exit with an error whose
    output contains <text>.

    Arguments:
      u         = the project's cocoa_test_utils module.
      name      = the class name to build (the caller assigns the class
                  to this same module-level name).
      sequences = the list of sequence dictionaries; each becomes one
                  test method test_<sequence name>.

    Returns:
      a unittest.TestCase subclass named `name`.
    """

    class TestSettingSequences(unittest.TestCase):
        """Settings changed in one process against a fresh process."""

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR and verify saved inputs when the project uses them."""
            u.require_cocoa_environment()
            if u._H.sequence_inputs is None:
                u.verify_frozen()

    def make_test(sequence):
        def test(self):
            import numpy as np
            print(f"\n  sequence {sequence['name']} (base "
                  f"{sequence['base']}, {len(sequence['steps'])} steps)",
                  flush=True)
            code, output, inprocess = u._H.run_sequence_leg(
                sequence, "inprocess")
            expect = sequence["expect"]
            if expect.startswith("fatal:"):
                text = expect[len("fatal:"):]
                self.assertNotEqual(
                    code, 0,
                    f"{sequence['name']}: the sequence ran to the end; it "
                    f"must stop with an error naming '{text}'")
                self.assertIn(
                    text, output,
                    f"{sequence['name']}: the process stopped (exit code "
                    f"{code}) without the message '{text}':\n"
                    f"{output[-3000:]}")
                return
            self.assertEqual(
                code, 0,
                f"{sequence['name']}: the in-process leg failed (exit code "
                f"{code}):\n{output[-3000:]}")
            compare = sequence.get("compare", [len(sequence["steps"])])
            for upto in compare:
                code, output, fresh = u._H.run_sequence_leg(
                    sequence, "fresh", upto=upto)
                self.assertEqual(
                    code, 0,
                    f"{sequence['name']}: the fresh leg of step {upto} "
                    f"failed (exit code {code}):\n{output[-3000:]}")
                key = f"v{upto}"
                same = np.array_equal(inprocess[key], fresh[key])
                print(f"    step {upto}: {inprocess[key].size} entries, "
                      f"in-process {'==' if same else '!='} fresh",
                      flush=True)
                self.assertTrue(
                    same,
                    f"{sequence['name']}: after step {upto} the in-process "
                    "vector differs from a fresh process at the same "
                    "settings: a cached table did not follow the change")

        test.__name__ = f"test_{sequence['name']}"
        test.__doc__ = f"Setting sequence {sequence['name']}."
        return test

    for sequence in sequences:
        test = make_test(sequence)
        setattr(TestSettingSequences, test.__name__, test)
    TestSettingSequences.__name__ = name
    TestSettingSequences.__qualname__ = name
    return TestSettingSequences
