# Supplied matter bispectrum validation

The optional input is an absolute matter bispectrum, in `(Mpc/h)^6`.
A response ratio answers a different question: how much feedback changes an
already chosen matter model. The two owners remain separate. They share
uniform-grid validation, natural-cubic preparation, and multilinear reads of
logarithmic values in `feedback_3pcf.c`; no dummy power response is allocated
for the baseline.

For ordered magnitudes `a <= b <= c`, the baseline coordinates are
`z, ln(b/[h/Mpc]), ln(a/b), (c-b)/a`. `log_bispectrum` is a flat C-order array,
with the last coordinate varying fastest. Every input axis is uniform and
has at least four points. `boost` controls dense intervals per loaded
interval. `clamp=0` refuses queries outside the specified table. Boundary
holding is possible only through the explicit `clamp=1` setting.

The additive production and notebook factories are both named
`PhysicsModelWithBispectrum`. They require a `baseline` dictionary and
`omega_m_z` on the background redshift axis. They omit the internal fit's
`omega_de, w0, wa, ns` arguments. `omega_m` still sets the lensing and IA
normalizations, while `sigma8` retains the existing linear-power
normalization. Removing that normalization would change another numerical
contract and is outside this change.

The supplied route never prepares the internal fit's nonlinear radius,
effective slope, or redshift-dependent fit coefficients. The full-B query
selects the supplied table centrally, so direct matter queries, projected
NLA, and covariance terms containing B agree about the selected model.
Tree-level diagnostics still use the supplied linear power. The fitted
one-/three-halo pieces are undefined for a general full table and are
rejected. The covariance's independent halo integrals remain unchanged.
Raw perturbative IA entries and signed quadratic IA residuals also retain
their existing power prescription.

`core.set_baseline_bispectrum` owns copies, compares every value and size,
and rebuilds only after a change. Clearing the supplied table restores the
legacy route and its separately stored scalars. Resetting the bridge cannot
invalidate an older Python model: that model owns private native arrays.
New table comparisons include the bit representation of signed zero. Copies
are made before old buffers are released, so native callers can resupply a
view of the bridge's current input.

## Checks and their limits

The project test file `tests/test_supplied_baseline.py` exercises analytic
multilinear logarithms, permutation symmetry, both adapters, input mutation,
retained model/grid lifetime, unchanged tree/P/IA values, explicit refusals,
bridge replacement/clear/reset, and every baseline cache input. Its covariance
check doubles B while holding P and the background fixed: the cubic mean and
PB must double, BB must quadruple, and the other covariance families must
remain unchanged. These identities check routing and normalization. They do
not establish that a particular cosmological B table is accurate.

`capture_default.py` saves a small set of default-route predictions in a
fresh process and records hashes of its binary and fixture. Compare the
saved array bytes between the pinned pre-edit binary and the new binary,
and between one and four OpenMP threads. BLAS must remain at one thread.
The original stage-0 battery additionally checks three physical test points,
production/notebook routes, joint xi/Map3 predictions, multipoles, natural
components, and all six covariance families. Its original script and outputs
are retained with the private validation artifacts; they are not replaced by
the smaller capture.

The final strict validation covered 767 project tests. Its first
collection passed 763 tests and skipped the original-code comparison because
the private project copy lacked the sibling source directories. After those
reference files were copied and hashed, that comparison and three newly
added cases passed in a separate strict run. The complete debug build passed
767 tests with no skips under undefined-behavior and floating-division
sanitizers. All tests used the private extensions, including CLI children;
the installed project binary was not replaced.

The final Python producer subsequently received a near-zero upper-gamma
correction and the joint test switched to committed, hashed inputs and a
committed covariance. Those final files passed 57 targeted checks in each
native build: 54 producer cases, the fixed joint-metric test and two squeezed
triangle cases. [producer_followup.json](producer_followup.json) preserves
their file hashes and test output separately from the earlier full-project
snapshot. Neither follow-up required a native source or binary change.

The earlier 42-array default capture matched the pre-edit binary at one and
four workers. The final complete stage-0 comparison matched every array in
all fourteen output-file comparisons for each build, including a repeated
first point. An additional curved logarithmic feedback table checks the
shared interpolation code at nonzero feedback amplitude. Across strict and
debug builds, all 344 compared arrays in 32 output files agree byte for byte
with the pre-edit reference. The full test collection also checks the new
absolute-B path: 29 focused cases cover interpolation, cache changes and
covariance routing.

[validation.json](validation.json) records the tested native source hashes,
compiler modes, extension hashes and individual output comparisons.
[reference](reference/) retains the small analytic reference outputs;
[manifests](manifests/) retains hashes of the pre-edit source/input copies,
the stage-0 inputs, the copied current project and the original libraries.
These tests establish regression and specified routing identities. A
supplied cosmological table still needs coarse/dense refinement and a
final-observable comparison weighted by the same fixed joint covariance.
The project owns that separate survey-level check.
