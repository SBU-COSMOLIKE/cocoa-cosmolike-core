# Core bridge table reuse and borrowed input checks

The core bridge assembles distances, power spectra, source distributions and
optional four-dimensional bispectrum tables into an immutable model. An
intrinsic-alignment amplitude changes how that model is queried. It does not
change the arrays from which the model was prepared. The feedback amplitude
has the same role: it multiplies the logarithmic response at query time.

The permanent project test is
`Cocoa/projects/des_3pcf/tests/test_bridge_cache.py`. It checks that changing
these query parameters reuses the model while changing its prediction, and
that returning to the original parameters recovers the original values.
Cosmological scalars and response table definitions must still replace the
model. An older model or projected-grid owner must retain its previous
values after another state is installed or the core is reset.

The test also compiles a small C caller against the actual public headers.
That caller resubmits arrays borrowed from the bridge itself. For example,
a request can use the current response arrays but increase their dense-grid
upsampling factor. The setter must copy all incoming arrays before freeing
any old allocation. The same requirement applies when changing source
densities while borrowing their redshift axis, or when retaining a shorter
prefix of the integration nodes or pixel window. Python never invents a
native struct layout, and no function is replaced during these checks.

`tests/test_bridge_sizes.py` checks a separate safety condition. The flat
source and feedback array counts must fit the C `int` count before any
multiplication, scan or copy occurs. Six child processes pass positive axis
counts whose product is too large, together with one unread NaN sentinel.
Requiring the size-specific error proves that refusal precedes even the
first array read. No large table is allocated for these tests.

## Comparing numerical output across builds

`capture.py` separates saving the inputs from evaluating either build.
The input NPZ stores every physical array, response table, scalar and query
control. The output NPZ stores 72 arrays: internal and supplied matter-B
routes, six successive IA/feedback states, and six numerical outputs per
state. The raw IA and projected three-point outputs use the state's TATT
coefficients. The two-point output uses the NLA-only API with a separately
named dictionary: quadratic IA, its density weighting, the counterterm and
the legacy bispectrum modifier are set to zero. It is an NLA two-point
check, not a full-TATT two-point prediction. The input record names those
zeroed fields. No survey data, covariance matrix or mask enters this
comparison.
Model-generation counts are recorded separately because fewer model builds
are expected, while the observable arrays must keep the same float64 bits.

The commands below run from this directory. Replace the project and
interface paths with the copied source tree and the extension being checked.
Each process selects only one extension. BLAS uses one worker; the command's
`--threads` controls OpenMP and permits at most six workers.

```bash
python capture.py freeze --project /path/to/copied/des_3pcf --output /tmp/bridge_inputs.npz
```

```bash
python capture.py capture --project /path/to/copied/des_3pcf --interface /path/to/reference/interface --inputs /tmp/bridge_inputs.npz --output /tmp/reference_bridge.npz --threads 1
```

```bash
python capture.py capture --project /path/to/copied/des_3pcf --interface /path/to/candidate/interface --inputs /tmp/bridge_inputs.npz --output /tmp/candidate_bridge.npz --threads 1
```

Use separate processes at one and four workers for the numerical comparison.
The input and output hashes identify the arrays used; they do not measure
forecast accuracy. The controlled examples exercise dependency handling and
ownership, rather than a survey's integration convergence.

## Recorded results

The [validation record](validation.json) reports 829 passing project tests
in each of the strict and UBSan builds, including 40 focused bridge and
array-size cases. The saved comparisons cover 72 bridge arrays and 66
stage-zero arrays, plus 11 entries from a repeated first-model evaluation.
Their bits agree across the reference and candidate builds at one and four
OpenMP workers. The record counts repeated comparisons separately from
these stored array entries.

Four targeted memory checks exercise replacing arrays borrowed from the
bridge. Each reproduces a read from released storage in the preceding
implementation and passes when replacement inputs are copied first.
Only the bridge translation unit is instrumented in these ASan checks.
They establish the tested input lifetimes, not simultaneous Python entry:
setters, refresh and reset remain sequential process-global operations.
The record retains setup failures and their corrections as well as the
successful results.
