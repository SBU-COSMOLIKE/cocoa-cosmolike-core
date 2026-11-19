# Double-Bessel preparation: validation record

The production change retains standalone FFTLog preparation through
`production.TwoBesselPlan` and corresponding copied notebook adapters.
The finite-triangle transform keeps its existing retained workspace.
The only C-engine change checks actual worker-buffer FFTW alignment during
construction. FFT arithmetic, `FFTW_ESTIMATE`, and serial FFTW under outer
OpenMP are unchanged.

## Completed checks

The strict `-O3 -march=native` build and the `-O0` build with
`-fsanitize=undefined,float-divide-by-zero -fno-sanitize-recover=all` both
passed **94 tests**. Each suite contains 35 retained-owner cases and 59
existing transform cases copied from the DES 3PCF project. The compiler is
Clang 19.1.7. Environment limits were six OpenMP workers, dynamic OpenMP
teams disabled, and one thread for OpenBLAS, MKL, BLIS and Accelerate.
These are correctness runs; their elapsed time is not a speed measurement.

Both builds used private FFTW instrumentation. Each fresh workspace
allocation was filled with NaNs. Every forward and inverse FFT checked all
of its input cells for finiteness, including padding, and compared its actual
input/output alignment classes with those used in planning. No such check
failed. The private recipe map is read-only during OpenMP execution; there
is no per-execution lock or counter. Production code has no test bindings
or lifecycle instrumentation.

The lifecycle test observed twenty plan creations and twenty destructions
for ten one-shot calculations. Ten calls through one retained standalone
object created two plans, kept both alive between calls, and destroyed both
when the owner was released. Retained finite-triangle objects also created
exactly two plans and reused them across replaced input arrays at one, four
and six workers. The comparisons use changed mathematical integrands or
bispectrum multipoles; they are not ten full cosmological predictions.

The numerical and state checks cover:

- Both C++ adapter routes; C-order borrowed inputs and copied lists,
  Fortran-order, sliced and read-only notebook arrays.
- Input A, changed input B, and restored A through one retained object;
  multiple inverse orders through 103 from the same forward coefficients.
- Bitwise equality of complete output arrays with the one-shot route;
  comparison with independent NumPy FFT/SciPy log-Gamma references.
- Physical grid size, equal padded FFT sizes with different physical size
  and padding, ell endpoints and spacing, both biases, xy, bin averaging,
  padding, taper selection and taper width.
- Caller mutation of constructor arrays and returned theta/output arrays;
  live independent owners, both destruction orders, and core reset.
- Fresh-interpreter agreement, named invalid-input refusals, strict
  production dtype/layout refusal, and finite-triangle worker changes.
- Existing angular quadrature, projection, aperture and finite-operator
  checks in the copied project transform suite.

Both compilers returned success. The sanitizer suite emitted no runtime
error. The debug compiler reported three existing unused-parameter warnings
in `basics.c`; no edited production source emitted a warning. The four edited
production files were checked byte for byte against both tested source
snapshots. Later formatting of private instrumentation comments/signatures
preserved every C++ token. `git diff --check`, the 80-column source check,
and Python syntax compilation passed. The build script's output-directory
guard was checked separately: a destination inside Cocoa is refused before
creating or copying files.

The JSON beside this record saves source hashes and test totals. Raw logs,
JUnit results, full source manifests and privately compiled extensions are
in `/private/tmp/fftw_retained_3pcf_20261118/`. The installed project extension
was not rebuilt: the project remains read-only for this task. The optional
API becomes available there after its usual build from the committed core.
The existing one-shot and finite-triangle APIs remain compatible.

## Reproducing the checks

Run these commands from `cosmolike_core` after starting the Cocoa environment.
`ROOTDIR` must name Cocoa and `CONDA_PREFIX` the active compiler environment.
Choose a directory outside the Cocoa source tree for each private build.
The script copies the project interface registration and transform tests;
it does not edit the project. Its numerical tests do not require survey data.
Coordinate the CPU allocation with any other active numerical run.

```bash
"${ROOTDIR:?}/.local/bin/python" \
  dev_scripts/threepcf_fft_plan_check/build_validation.py \
  --output /private/tmp/fftw_retained_3pcf_20261118/strict \
  --root "${ROOTDIR:?}" \
  --python "${ROOTDIR:?}/.local/bin/python" \
  --conda "${CONDA_PREFIX:?}"
```

```bash
env OMP_NUM_THREADS=6 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  "${ROOTDIR:?}/.local/bin/python" -B -m pytest -q -p no:cacheprovider \
  /private/tmp/fftw_retained_3pcf_20261118/strict/tests/data_vector \
  --junitxml=/private/tmp/fftw_retained_3pcf_20261118/strict_tests.xml
```

```bash
"${ROOTDIR:?}/.local/bin/python" \
  dev_scripts/threepcf_fft_plan_check/build_validation.py \
  --output /private/tmp/fftw_retained_3pcf_20261118/debug \
  --root "${ROOTDIR:?}" \
  --python "${ROOTDIR:?}/.local/bin/python" \
  --conda "${CONDA_PREFIX:?}" \
  --debug
```

```bash
env OMP_NUM_THREADS=6 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  "${ROOTDIR:?}/.local/bin/python" -B -m pytest -q -p no:cacheprovider \
  /private/tmp/fftw_retained_3pcf_20261118/debug/tests/data_vector \
  --junitxml=/private/tmp/fftw_retained_3pcf_20261118/debug_tests.xml
```

## Limits

No timing study was run on the busy machine. These tests establish reuse,
state independence and finite-operator agreement; they do not establish DES
accuracy convergence or a science-loop speedup. The science engine already
retains its finite-triangle transform. Standalone callers found in the
project are its original-library comparison and component checks.

Sharing FFTW pairs across distinct geometry owners and changing planner
policy remain conditional on preparation/execution profiles and numerical
checks. No global plan pool, `FFTW_MEASURE` default, threaded FFTW, or project
caller rewrite is part of this change.
