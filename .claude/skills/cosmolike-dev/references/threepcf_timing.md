# Three-point CLI timing against LSST Y1

Read this before changing `projects/des_3pcf/scripts/benchmark_scaling.py`
or interpreting its results. The project owns the driver, physical input
preparation and figure. Core numerical routines retain their existing cache
rules. The project uses `main`; core work uses `lowrank_fftlog`. Preserve
other teams' edits and commit only the intended paths.

## Distinguish preparation from the measured calculation

The study runs LSST Y1 NLA, LSST Y1 TATT, DES Y3 three-point NLA and DES Y3
three-point TATT, in that order. LSST uses its existing production interface
and all 1560 3x2pt entries. Each case's three thread-count blocks must finish
and agree before the next case begins. Three-point cases contain 80 aperture
entries, twenty source triplets and four filter radii. The two data vectors
have different observables; compare each case against its own one-thread
result. Within each data vector, NLA and TATT share cosmological points and
numerical grids.

`benchmark_scaling.py preflight` hashes inputs and draws parameter points.
It performs no CAMB call or physical prediction. `run` prepares eleven
physical table archives per case: one separate warmup point and ten
random points. Every thread count reads the same archives outside its
measured interval. CAMB is excluded from data-vector timings. The separately
recorded preparation duration includes source processing and file writing,
so it must not be called a CAMB-only duration. IA does not change these
matter tables. The helpers allow reuse between NLA and TATT only after
matching physical preparation inputs and recording the prior archive hash.
Three-point reuse checks actual k/z/line-of-sight/source arrays as well as
cosmology, producer source and source-file identity. No timed prediction is
reused. The optional CLI `--reuse-from` imports verified LSST tables from an
earlier two-workload run; it does not import its timing measurements.

Follow the LSST separation between the Boltzmann solve range and interpolation
support. In the project producer, kmax_boltzmann defaults to 5 and extrap_kmax
is a separate tail endpoint; both use physical 1/Mpc. The output table uses
h/Mpc, so query CAMB at k*h and convert its Mpc^3 power by h^3. Do not request
transfer evolution to the output table's endpoint near 1000 h/Mpc. Resolve
coverage before running CAMB and record both limits. With WantCls=False,
power_spectra_from_transfer recalculates powers after the sigma8 amplitude
adjustment without repeating transfer evolution. The project tests compare
this result against a fresh final-amplitude solve using the opposite unit API.
Changes to either limit must invalidate the physical provider and outer mean
cache, and incompatible reference-table provenance must be rejected.

The LSST interval installs cosmology and nuisance arrays through its native
setters and calls `compute_data_vector_masked`. The three-point interval
checks point identity, constructs fresh physical C tables and evaluates
the retained aperture geometry. Fixed geometry, FFT plans, imports, file
reads, covariance loading and output checks are excluded. The LSST covariance
load initializes a shared likelihood entry point; it is not a covariance
benchmark. These boundaries do not measure total Fisher or sampler wall time.

## Change physical states rather than manufacturing cache misses

The LSST point bank comes from the existing `nmodels_points` test sampler.
Retain its explicit boxes, amplitude rejection and alternating held
parameters. The DES bank uses the same seed-plus-index convention with
declared intervals. DES varies sigma8; LSST varies As. Their banks are
separate, and the LSST test bank is not a draw from a joint survey prior.

Process all ten points at one thread count before starting a fresh child
for the next count. Timing one unchanged point repeatedly can measure a
cache hit. Do not change an internal cache identifier or toggle the run
mode to manufacture a miss. Parameter changes use ordinary setters; angular
geometry remains reusable. A separate warmup prevents the first allocation
and lazy table construction from dominating a reported sample.

LSST uses native IA_model 0 for NLA and 1 for TATT, with CFASTPT code 0 and
redshift evolution 3. Its TATT warmup has nonzero quadratic amplitudes so
first-use CFASTPT preparation stays outside the measurements. Three-point
NLA fixes a2, eta2, bta and ct to zero after drawing the same random sequence
as TATT. This omits physical residual-grid allocation and preparation;
the declared Legendre and natural-component transform loops remain.

## Preserve thread ownership and evidence

Set all BLAS vendors to one thread before numerical imports and inspect
loaded pools through `threadpoolctl`. OpenMP owns the parallel work. The
declared study uses 1, 4 and 8 threads, with fresh children in order 4, 1, 8.
Eight threads are authorized for this timing study; routine tests retain
their four-thread cap. Wait for the user to make the shared machine quiet
before running this study. Never run another numerical test alongside it.

The driver copies both extension binaries and records their hashes. It
checks the runtime's actual maximum OpenMP team and loaded BLAS pools.
Every point must retain identical float64 bits across thread counts;
consecutive different points must produce different vectors. A failed
comparison stops publication of the scaling figure and preserves logs.
The ten dots represent different physical points. Their spread includes
parameter dependence and machine variation, not a confidence interval.

## Keep the timing source policy separate from scientific inputs

The supplied Y3 nz_source BIN1 contains twelve negative samples, with an
integrated magnitude about 0.041% of the signed normalization.
`configs/des_y3_tatt_scaling.yaml` declares explicit timing clipping and
renormalization. The helper
records raw file hashes, negative area, normalization and source shifts.
Its physical and line-of-sight grids extend to 3.1 for positive shifts.

Preserve the original FITS data and strict generic source reader. The user
also authorized an explicitly labeled demonstration forecast with recorded
clipping and normalization; that authorization does not need to be requested
again. Give the demonstration its own derived source file and processing
record, and use that same file for the two- and three-point means. Clipping
is not an established source-calibration choice for a scientific result. Likewise,
the chosen finite-triangle geometry and starting grids have not established
survey-level accuracy merely because the timing run completes.

## Separate covariance performance from covariance convergence

`tests/studies/covariance_scaling.py` in the project supplies an independent
offline workload. Its physical case includes all internal C preparation and
the six implemented term families; Python model construction and CAMB are
excluded. It uses gravity-only full-aperture rows, a large-area limit for
short modes, and declared test area/noise. It is not matched to the finite-SAS
TATT mean. Even a much shorter covariance runtime cannot be interpreted as
the cost of that matched survey covariance. Record the quadrature counts,
integration bounds and term-by-term refinement before comparing costs.

The Gaussian-only case is distinct: it times supplied-spectrum pair transforms
and all fifteen finite-window Wick contractions. The large-area physical
branch has six surviving Gaussian contractions. Do not combine those timing
records under one unnamed Gaussian workload.
