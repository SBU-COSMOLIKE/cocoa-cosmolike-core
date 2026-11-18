# Three-point CLI timing against LSST Y1

Read this before changing `projects/des_3pcf/scripts/benchmark_scaling.py`
or interpreting its results. The project owns the driver, physical input
preparation and figure. Core numerical routines retain their existing cache
rules. The project uses `main`; core work uses `lowrank_fftlog`. Preserve
other teams' edits and commit only the intended paths.

## Distinguish preparation from the measured calculation

The study starts with LSST Y1, using its existing production interface and
all 1560 3x2pt entries. Its three thread-count blocks must finish and agree
before preparation of the DES Y3 three-point workload begins. The latter
contains 80 aperture entries, twenty source triplets and four filter radii.
The two workloads have different observables and IA models; compare each
one's speedup against its own one-thread result.

`benchmark_scaling.py preflight` hashes inputs and draws parameter points.
It performs no CAMB call or physical prediction. `run` prepares eleven
physical table archives per workload: one separate warmup point and ten
random points. Every thread count reads the same archives outside its
measured interval. CAMB is excluded from data-vector timings. The separately
recorded preparation duration includes source processing and file writing,
so it must not be called a CAMB-only duration.

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
integrated magnitude about 0.041% of the signed normalization. The user
authorized clipping those samples to zero and renormalizing for timing
only. `configs/des_y3_tatt_scaling.yaml` declares that policy. The helper
records raw file hashes, negative area, normalization and source shifts.
Its physical and line-of-sight grids extend to 3.1 for positive shifts.

Do not change the original FITS data, the strict generic source reader or
forecast configurations to implement this exception. The clipping is not
an accepted source-calibration choice for a scientific result. Likewise,
the chosen finite-triangle geometry and starting grids have not established
survey-level accuracy merely because the timing run completes.
