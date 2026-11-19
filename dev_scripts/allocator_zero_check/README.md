# Empty arrays and core allocation layouts

The core allocators return one aligned block that the caller releases with
one `free`. An empty logical array still has a non-NULL owner. A zero-byte
request reserves 64 bytes without creating readable elements. For a
multidimensional array, pointer levels before a zero extent remain present;
there are no entries at that extent or any later dimension. Existing
positive-size row padding and pointer offsets are preserved.

This case occurs in the three-point covariance. A request containing only
two-point rows has no cubic-row entities, and its contraction list can be
empty. `covariance_physics_3pcf.c` can then call `malloc1d_int(0)` or
`calloc1d(0)`. A zero-byte `posix_memalign` call is allowed to succeed with
a NULL pointer; using an explicit nonzero allocation avoids that
platform-dependent outcome before initialization and cleanup.

## What the check exercises

`check_allocators.c` calls the twelve typed allocation functions in the
actual `basics.c`. It covers zero in every dimension and combinations of
zero extents, positive extents crossing a padding boundary, 64-byte owner
and data-row alignment, pointer-table access, zero initialization, distinct
row contents and one-owner cleanup. It never reads padding or treats an
empty array as containing a data element.

`run_checks.py` builds the current implementation and the `basics.c` from
a named Git revision. Each uses the current declarations, structs and
logging code. It compares positive-shape row offsets byte for byte and
runs the candidate's empty/positive checks with one and six OpenMP workers.
BLAS stays at one. The builds use strict arithmetic, UBSan, and a separate
AddressSanitizer mode. ASan leak detection is disabled for platform
compatibility; invalid memory access and frees remain checked.

The old empty case need not fail on this machine: its allocator may return
a non-NULL pointer for a zero-byte request. The comparison therefore does
not claim a reproduced historical crash. It requires the candidate to
honor the declared empty-array contract and preserves the old positive
layouts. Array-count overflow and negative dimensions are outside this
check; callers must supply nonnegative counts whose allocation arithmetic
fits the existing size types.

## Run

Activate and source the normal Cocoa environment first. From the core
repository, select the revision before the allocator change and a fresh
output directory:

```bash
python dev_scripts/allocator_zero_check/run_checks.py \
  --reference 2bc4500a9a55c1c4dcbe26952e0cfd658846fdc5 \
  --output /tmp/cosmolike_allocator_zero_check
```

The script writes compiler commands, source/executable/output hashes and
comparison results to `results.json`. Each build and process has its own
log. Any failed build, assertion, sanitizer exit or layout mismatch stops
the script. It runs no survey likelihood and measures no performance.
Full data-vector and covariance comparisons belong to their project checks.

## Recorded results

The [saved validation record](validation_20261119.json) contains five
successful runs in each of the strict, UBSan and ASan modes. Positive
layouts match the preceding implementation, and the candidate's empty and
positive cases give identical output at one and six OpenMP workers.
The record identifies the source, compiler commands, executables and output
hashes; it does not extend the checks to negative or overflowing counts.
