# FFTLog endpoint correction and its validation

The smoothing variance is an integral of the supplied dimensionless power,
`Delta^2(k) = k^3 P(k)/(2 pi^2)`, against a window squared. FFTLog divides
this spectrum by `k^b` and represents the result as periodic in `ln k`.
Here `b = 0.8`. If an endpoint continuation has `Delta^2 = A k^s`, the
weighted function grows towards high k when `s > b`, or towards low k
when `s < b`. Its periodic image can contaminate radii well inside the
requested range. Extending a growing continuation further does not fix it.
The original unbroken `P proportional to k^-2` test had a 9.72% nonlinear
radius error and an 11.69% sigma8 error. These were analytic stress tests,
not measured errors of a DES forecast.

## Mathematical identity

For each problematic continuation, subtract `A k^s` at every wavenumber,
transform the residual, and add `A R^-s M_W(s)` to the variance. Here
`M_W(s) = integral dx x^(s-1) W(x)^2` is the analytic Mellin integral of
the selected window. For the Gaussian window squared it is
`Gamma(s/2)/2`. For the spherical top-hat window squared it is

```math
M_{\rm TH}(s)=
\frac{9\pi\,\Gamma(4-s)\Gamma(s/2)}
{2^{5-s}\Gamma^2((5-s)/2)\Gamma(4-s/2)}.
```

The top-hat expression requires `0 < s < 4`. The Gaussian expression
requires `s > 0`. These are convergence conditions of the analytic
counterterm. A nonpositive low-k exponent makes the original infinite
variance diverge; a top-hat high-k exponent at least four also diverges.
Those cases are refused rather than assigned a finite truncated answer.
The radial derivative of each added term is `-s` times that term.

The input spectrum and its endpoint continuation remain the same. The
subtraction changes the numerical representation of their integral. Each
row owns two small endpoint descriptors; no new persistent cache exists.
The fixed-bias FFT kernels, retained plans and threading remain unchanged.
Outside an endpoint, the matching residual is zero by algebra. Evaluating
that zero by subtracting two large rounded exponentials manufactured false
power in the initial independent experiment, so the C implementation avoids
that subtraction. Interior near-cancellation uses `expm1`.

A bias-matched high-k tail is included in the correction. Leaving that
single spectrum on the zero-padded path gave a small mean offset but a
0.366% error in a spectral-slope derivative at step 0.00025. The corrected
path passes the unchanged 0.1% derivative bound at steps 0.001, 0.0005 and
0.00025. The shared threshold on endpoint slopes prevents roundoff from
selecting different treatments for an exactly matched spectrum.

## Checks and limits

The expanded component suite passes 54 tests in strict mode and 54 under
UBSan. It includes five analytic spectral slopes, curved spectra activating
neither, either or both corrections, derivative-step checks, one/six-worker
equality, reset, copied ownership and stale-state refusals. The power-law
and curved-spectrum accuracy bounds remain 0.1%.

The independent low-tail quadrature must extend below k = 1e-7 h/Mpc.
For `Delta^2 proportional to sqrt(k)`, truncating there omits approximately
0.001 of variance. Its reference now extends to 1e-16 h/Mpc. The low-tail
curved case has a z = 1 unit crossing near R = 0.005 Mpc/h. At the default
Gaussian-table resolution its radius error is 0.1088%; refining the existing
scale-factor and inverse tables reduces it. The test requires a decrease
by more than a factor of three and the original 0.1% bound after two nested
refinements. This is a recorded resolution requirement, not evidence that
default sampling suffices for every spectrum.

Separate processes captured an ordinary turnover spectrum and the saved
DES fiducial spectrum, with identical input arrays. All 18 snapshot fields
are byte-identical before and after the correction. The mass queries span
401 samples inside the density-adjusted supported mass range, at three
redshifts. These checks cover total matter with factorized input growth;
they do not establish a cb-field forecast or the final shared-variance
adoption for the three-point provider.

The workspace record is `to_reproduce/threepcf_consolidation_20261119/`.
It contains `fftlog_tail_probe.py`, `variance_refinement_probe.py`,
`capture_variance.py` and their result records. Private build and run
artifacts are in `/private/tmp/threepcf_consolidation_20261119/`.
