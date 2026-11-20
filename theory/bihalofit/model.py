"""Prepare the BiHalofit matter bispectrum outside the C integration code.

The fit is Takahashi et al., arXiv:1911.07886. Its coefficients and numerical
conventions follow the retired C matter fit and Sunao Sugiyama's fastnc
TATT branch, revision 7b2336ce37a909bed51f07f32a2a4f096e71b6ce. Rafael Gomes's
three-point analyses motivated the surrounding projection pipeline. This
module supplies the dark-matter-only baseline; feedback remains a separate
physical response, applied by the existing project machinery.

Units are k in h/Mpc, P in (Mpc/h)^3 and B in (Mpc/h)^6. The background is
provided as arrays. No w0-wa expansion model or flatness relation is assumed
here. The reference fit uses P_lin(k,0) D(z)^2 and explicit scalar sigma8
normalization; it does not promote this approximation to a calibrated
massive-neutrino bispectrum model.

Original fastnc license: MIT License, Copyright (c) 2024 Sunao Sugiyama.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import hashlib
import json

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import exp1, exprel, gammaincc, gammaln, zeta

def uniform_segments(coordinates, name, maximum=10):
    """Return inclusive endpoints of the declared uniform coordinate pieces.

    Cocoa's grid detector compares each interval with the first spacing
    in its current piece, allowing relative rounding error 1e-9. Adjacent
    pieces share their endpoint: [(0, 20), (20, 44)] describes 45 nodes.
    This check runs before constructing native models, where an invalid
    grid would otherwise terminate the process. It changes no samples.
    """
    axis = np.asarray(coordinates, dtype=np.float64)
    if (axis.ndim != 1 or axis.size < 2
            or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0)):
        raise ValueError(f"{name} requires at least two finite increasing nodes.")
    pieces = []
    begin = 0
    spacing = axis[1] - axis[0]
    for index in range(1, axis.size):
        following = spacing
        if index < axis.size - 1:
            following = axis[index + 1] - axis[index]
        if index == axis.size - 1 or abs(following - spacing) > 1.0e-9 * spacing:
            pieces.append((begin, index))
            if len(pieces) > maximum:
                raise ValueError(
                    f"{name} has more than {maximum} uniform pieces; "
                    "construct each coordinate segment with np.linspace."
                )
            begin = index
            spacing = following
    return pieces


# Retain the C literal for 1/(2 pi^2) while comparing numerical conventions.
# It converts dimensional P into Delta^2=k^3 P/(2 pi^2).
INVERSE_TWO_PI_SQUARED = 0.05066059182116888572
# Historical source fingerprint of the C equation used for this port.
# The current C projection library no longer implements that equation.
C_SOURCE_SHA256 = (
    "af3c533d7a85ddf493ab5840823134206fcd154ac2d5cd648875185de41ac351"
)


def _axis(values, name, uniform=False):
    """Copy four or more finite increasing coordinates, checking uniformity."""
    result = np.array(values, dtype=np.float64, order="C", copy=True)
    if result.ndim != 1 or result.size < 4:
        raise ValueError(f"{name} needs at least four nodes on one axis.")
    if not np.all(np.isfinite(result)) or np.any(np.diff(result) <= 0.0):
        raise ValueError(f"{name} must contain finite increasing coordinates.")
    if uniform:
        regular = np.linspace(start=result[0], stop=result[-1], num=result.size)
        scale = max(1.0, float(np.max(np.abs(result))))
        tolerance = 64.0 * np.finfo(float).eps * scale
        if not np.allclose(result, regular, rtol=0.0, atol=tolerance):
            raise ValueError(f"{name} must have uniform stored coordinates.")
    return result


def _integer(value, name, minimum):
    """Require an explicit integer count before an allocation or grid build."""
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    if not np.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    if value > np.iinfo(np.int32).max:
        raise ValueError(f"{name} exceeds the C integer range.")
    return int(value)


def _values(values, shape, name, positive=False):
    """Own finite values with declared axes and optional positivity."""
    result = np.array(values, dtype=np.float64, order="C", copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} requires finite values with shape {shape}.")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")
    return result


def input_hashes(arrays, settings):
    """Hash names, shapes, dtypes, every array byte and scalar/control values.

    This is the calculation's input record, not a hash of its answer.
    Dictionary keys are sorted so insertion order cannot change the record.
    File-based A/B gates must additionally record data, covariance and masks;
    the equation-level producer does not read those survey inputs.
    """
    result = {}
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        header = json.dumps({"shape": array.shape, "dtype": array.dtype.str})
        digest = hashlib.sha256(header.encode("utf-8"))
        digest.update(array.tobytes(order="C"))
        result[name] = digest.hexdigest()
    controls = json.dumps(settings, sort_keys=True, allow_nan=False)
    result["settings"] = hashlib.sha256(controls.encode("utf-8")).hexdigest()
    return result


class _Curve:
    """Prepare a cubic once, then use arithmetic cells on a dense uniform grid.

    Production uses natural boundaries on each declared uniform piece.
    Cubic evaluation prepares dense samples once; repeated read() calls
    select a piece and find a cell by index arithmetic. Reference mode
    retains the original not-a-knot cubic for comparisons with fastnc.
    """

    def __init__(self, x, y, natural=False):
        self.x = np.array(x, dtype=np.float64, copy=True)
        self.y = np.array(y, dtype=np.float64, copy=True)
        boundary = "natural" if natural else "not-a-knot"
        self.pieces = [(0, self.x.size - 1)]
        if natural:
            self.pieces = uniform_segments(coordinates=self.x, name="BiHalofit curve")
        self.cubics = []
        for first, last in self.pieces:
            curve = CubicSpline(
                x=self.x[first:last + 1], y=self.y[first:last + 1],
                bc_type=boundary,
            )
            self.cubics.append(curve)
        self.low_slope = (self.y[1] - self.y[0]) / (self.x[1] - self.x[0])
        self.high_slope = (self.y[-1] - self.y[-2]) / (
            self.x[-1] - self.x[-2])
        self.inverse_spacing = None

    def shift(self, offset):
        """Add a constant without refitting second derivatives or end slopes."""
        self.y += offset
        for cubic in self.cubics:
            cubic.c[-1, :] += offset

    def prepare_uniform(self, boost):
        """Refine original intervals before any repeated triangle evaluation."""
        count = (self.x.size - 1) * boost + 1
        if count > np.iinfo(np.int32).max:
            raise ValueError("Dense BiHalofit reader exceeds the C range.")
        axis = np.empty(shape=count)
        values = np.empty(shape=count)
        pieces = []
        inverse = []
        for segment, (first, last) in enumerate(self.pieces):
            lower = first * boost
            upper = last * boost
            nodes = np.linspace(start=self.x[first], stop=self.x[last],
                                num=upper - lower + 1)
            axis[lower:upper + 1] = nodes
            values[lower:upper + 1] = self.cubics[segment](x=nodes)
            pieces.append((lower, upper))
            inverse.append((upper - lower) / (nodes[-1] - nodes[0]))
        self.y = values
        self.x = axis
        self.pieces = pieces
        self.inverse_spacing = inverse
        self.cubics = None

    def read(self, coordinates, endpoint_power=False):
        """Read y; endpoint_power extends log P with its original slope."""
        query = np.asarray(coordinates, dtype=np.float64)
        # At most ten uniform pieces need inspection, independent of the
        # number of samples. Queries beyond the ends use the edge piece.
        values = np.empty_like(query)
        for segment, (first, last) in enumerate(self.pieces):
            if self.inverse_spacing is None:
                candidate = self.cubics[segment](x=query)
            else:
                position = (query - self.x[first]) * self.inverse_spacing[segment]
                local = np.floor(np.clip(position, 0, last - first - 1))
                index = first + local.astype(np.intp)
                fraction = position - local
                candidate = self.y[index] + fraction * (
                    self.y[index + 1] - self.y[index])
            if segment == 0:
                values = candidate
            else:
                values = np.where(query >= self.x[first], candidate, values)
        if endpoint_power:
            below = self.y[0] + self.low_slope * (query - self.x[0])
            above = self.y[-1] + self.high_slope * (query - self.x[-1])
            values = np.where(query < self.x[0], below, values)
            values = np.where(query > self.x[-1], above, values)
        return values


def _simpson_uniform(values, spacing):
    """Integrate the last axis with the C Simpson/Cartwright end correction."""
    count = values.shape[-1]
    odd_count = count - 1 if count % 2 == 0 else count
    total = values[..., 0] + values[..., odd_count - 1]
    total += 4.0 * np.sum(values[..., 1:odd_count - 1:2], axis=-1)
    total += 2.0 * np.sum(values[..., 2:odd_count - 1:2], axis=-1)
    total *= spacing / 3.0
    if count % 2 == 0:
        total += spacing * (5.0 * values[..., -1] / 12.0
                            + 2.0 * values[..., -2] / 3.0
                            - values[..., -3] / 12.0)
    return total


def _gamma_fraction(shape, lower):
    """Evaluate the upper-gamma fraction for x>1 and negative or small a.

    This expression returns the unnormalized upper integral directly, so
    it also avoids the large-x cancellation of downward recurrence when a
    is negative. No Gamma(a) normalization or division by a is involved.
    """
    x = lower
    epsilon = np.finfo(float).eps
    # Modified Lentz evaluation of the upper-gamma continued fraction.
    # Tiny-denominator protection bounds reciprocals within float range.
    tiny = np.finfo(float).tiny / epsilon
    denominator = x + 1.0 - shape
    c_value = np.full(shape=x.shape, fill_value=1.0 / tiny)
    d_value = 1.0 / denominator
    fraction = d_value.copy()
    active = np.ones(shape=x.shape, dtype=bool)
    for order in range(1, 1024):
        numerator = -order * (order - shape)
        denominator += 2.0
        d_value = numerator * d_value + denominator
        c_value = denominator + numerator / c_value
        d_value = np.where(np.abs(d_value) < tiny,
                           np.copysign(tiny, d_value), d_value)
        c_value = np.where(np.abs(c_value) < tiny,
                           np.copysign(tiny, c_value), c_value)
        d_value = 1.0 / d_value
        multiplier = d_value * c_value
        fraction[active] *= multiplier[active]
        active &= np.abs(multiplier - 1.0) > 4.0 * epsilon
        if not np.any(active):
            break
    if np.any(active):
        raise ValueError("The upper-gamma fraction did not converge.")
    with np.errstate(under="ignore"):
        result = np.exp(-x + shape * np.log(x)) * fraction
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("The upper-gamma fraction failed its finite check.")
    return result


def _gamma_near_zero(shape, lower):
    """Evaluate Gamma(a,x) near a=0 without dividing a cancellation by a.

    For x<=1, subtract the lower-gamma series from Gamma(a) analytically.
    Writing L=log Gamma(1+a)/a removes its pole before floating arithmetic:
      Gamma(a,x)=x^a [(L-log x) exprel(a(L-log x)) - S],
      S=sum(n>=1) (-x)^n/[n! (a+n)].
    Here exprel(t)=(exp(t)-1)/t is itself evaluated stably by SciPy. The
    convergent zeta series for L is retained through a^5; for |a|<1e-3,
    the next term is below 2e-19. For x>1, the usual upper-gamma continued
    fraction converges without this subtraction. Both iterations have a
    fixed maximum length and an explicit failure; neither changes the
    physical integration nodes of the BiHalofit fit.
    """
    epsilon = np.finfo(float).eps
    result = np.empty_like(lower)
    small = lower <= 1.0
    if np.any(small):
        x = lower[small]
        coefficient = 0.0
        for order in range(6, 1, -1):
            value = (-1.0)**order * zeta(order, 1.0) / order
            coefficient = value + shape * coefficient
        log_gamma_ratio = -np.euler_gamma + shape * coefficient
        difference = log_gamma_ratio - np.log(x)
        first = difference * exprel(shape * difference)
        power = -x
        series = power / (shape + 1.0)
        for order in range(2, 128):
            power *= -x / order
            term = power / (shape + order)
            series += term
            if np.all(np.abs(term) <= epsilon * np.abs(series)):
                break
        else:
            raise ValueError("The near-zero gamma series did not converge.")
        result[small] = np.exp(shape * np.log(x)) * (first - series)
    if np.any(~small):
        result[~small] = _gamma_fraction(shape=shape, lower=lower[~small])
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("The near-zero gamma evaluation failed its check.")
    return result


def _upper_gamma(shape, lower):
    """Return Gamma(shape,lower), including nonpositive shape if needed.

    SciPy's normalized gammaincc accepts positive shape. Recurrence extends
    the unnormalized integral to negative shape; Gamma(0,x)=E1(x) handles
    its zero step. Large lower bounds can underflow to a zero tail, as GSL
    permits in the C calculation. Negative or nonfinite answers are refused.
    """
    lower = np.asarray(lower, dtype=np.float64)
    if np.any(lower <= 0.0):
        raise ValueError("Gaussian tail requires a positive lower bound.")
    if abs(shape) < 1.0e-3:
        if shape == 0.0:
            return exp1(lower)
        return _gamma_near_zero(shape=shape, lower=lower)
    if shape < 0.0 and np.any(lower > 1.0):
        # One split sends every large-x value to the direct fraction.
        # The remaining call contains only x<=1 and cannot split again.
        large = lower > 1.0
        value = np.empty_like(lower)
        value[large] = _gamma_fraction(shape=shape, lower=lower[large])
        if np.any(~large):
            value[~large] = _upper_gamma(shape=shape, lower=lower[~large])
        return value
    steps = max(0, int(np.floor(-shape)) + 1)
    raised = shape + steps
    with np.errstate(divide="ignore", under="ignore"):
        value = np.exp(gammaln(raised) + np.log(gammaincc(raised, lower)))
        for step in range(steps):
            current = raised - step - 1.0
            if current == 0.0:
                value = exp1(lower)
            elif abs(current) < 1.0e-3:
                value = _gamma_near_zero(shape=current, lower=lower)
            else:
                boundary = np.exp(current * np.log(lower) - lower)
                value = (value - boundary) / current
    if np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("Gaussian upper-gamma tail failed its finite check.")
    return value


def fit_redshifts(zmax, fit_nz=33, source_readers=False):
    """Return uniform production or upstream-comparison fit anchors."""
    fit_nz = _integer(value=fit_nz, name="fit_nz", minimum=4)
    if not np.isfinite(zmax) or zmax <= 0.0:
        raise ValueError("The fit requires a finite positive redshift maximum.")
    if not source_readers:
        return np.linspace(start=0.0, stop=zmax, num=fit_nz)
    nodes = []
    index = 0
    while 0.15 * index < min(0.5, zmax):
        nodes.append(0.15 * index)
        index += 1
    if zmax > 0.5:
        index = 0
        while 0.5 + 0.3 * index < zmax:
            nodes.append(0.5 + 0.3 * index)
            index += 1
    nodes.append(zmax)
    if len(nodes) < 4:
        return np.linspace(start=0.0, stop=zmax, num=4)
    return np.asarray(nodes, dtype=np.float64)


def reference_smoothed_variance(linear, radius, growth_squared, window, logk_min, count, tail):
    """Integrate k^3 P W^2/(2pi^2), sharing k nodes across redshift rows."""
    coordinates = np.linspace(start=logk_min, stop=np.log(100.0), num=count)
    wave = np.exp(coordinates)
    radius = np.asarray(radius, dtype=np.float64)
    x = radius[..., None] * wave
    if window == 0:
        square = x * x
        small = np.abs(x) < 1.0e-3
        # Do not evaluate the cancellation-prone expression on the small
        # branch; np.where would evaluate both of its array arguments.
        kernel = np.empty_like(x)
        kernel[small] = (1.0 - square[small] / 10.0
                         + square[small]**2 / 280.0)
        large = x[~small]
        kernel[~small] = 3.0 * (
            np.sin(large) - large * np.cos(large)) / large**3
    else:
        kernel = np.exp(-0.5 * x * x)
        if window == 2:
            kernel *= x
    power = np.exp(linear.read(coordinates, endpoint_power=True))
    integrand = power * wave * wave * wave * INVERSE_TWO_PI_SQUARED
    integrand = integrand * np.asarray(growth_squared)[..., None]
    integrand = integrand * kernel * kernel
    spacing = (coordinates[-1] - coordinates[0]) / (count - 1)
    result = _simpson_uniform(values=integrand, spacing=spacing)
    if tail:
        end = coordinates[-1]
        previous = end - spacing
        log_last = linear.read(end, endpoint_power=True) + 3.0 * end
        log_previous = linear.read(
            previous, endpoint_power=True) + 3.0 * previous
        slope = float((log_last - log_previous) / spacing)
        gamma = _upper_gamma(shape=0.5 * slope, lower=(100.0 * radius)**2)
        result += (0.5 * np.exp(log_last - slope * end)
                   * radius**(-slope) * gamma
                   * INVERSE_TWO_PI_SQUARED * growth_squared)
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("BiHalofit variance must be positive and finite.")
    return result


class BiHalofit:
    """Own the supplied background and the fitted matter-bispectrum equation.

    Arguments:
      k, p_linear0 = increasing h/Mpc grid and positive P(k,0), (Mpc/h)^3.
      z, growth = redshift grid beginning at zero and D(z), with D(0)=1.
      omega_m_z = supplied matter-density fraction, including massive nu.
      omega_m_redshift = its coordinate grid; None means the background z.
      ns, sigma8 = primordial slope and target normalization used by the
        independent comparison calculation. Trusting raw P normalization is a
        separate numerical/physical change and is not silently selected.
      fit_nz, table_boost, source_readers = existing C preparation controls.
      variance_nodes, slope_nodes, radius_rtol = explicit refinement controls;
        defaults use 1000/2000-node rules and a 1e-5 root tolerance.

    The fit is a DMO prescription. Its full, tree, one-halo and three-halo
    components are diagnostics of this equation. Only the positive full B
    is used to prepare a supplied baseline table for C.
    """

    def __init__(self, k, p_linear0, z, growth, omega_m_z, ns, sigma8,
                 fit_nz=33, table_boost=8, source_readers=False,
                 omega_m_redshift=None, variance_nodes=1000,
                 slope_nodes=2000, radius_rtol=1.0e-5, variance=None):
        self.k = _axis(values=k, name="power k")
        self.z = _axis(values=z, name="background z")
        if self.k[0] <= 0.0 or self.z[0] != 0.0:
            raise ValueError("Require k>0 and background z starting at 0.")
        self.p_linear0 = _values(values=p_linear0, shape=self.k.shape,
                                    name="p_linear0", positive=True)
        self.growth = _values(values=growth, shape=self.z.shape,
                               name="growth", positive=True)
        if abs(self.growth[0] - 1.0) > 1.0e-12:
            raise ValueError("Growth must be normalized to D(0)=1.")
        if omega_m_redshift is None:
            omega_m_redshift = self.z
        self.omega_m_redshift = _axis(
            values=omega_m_redshift, name="Omega_m redshift")
        self.omega_m_z = _values(
            values=omega_m_z, shape=self.omega_m_redshift.shape,
            name="omega_m_z", positive=True)
        if (self.omega_m_redshift[0] > 0.0
                or self.omega_m_redshift[-1] < self.z[-1]):
            raise ValueError("Omega_m samples must cover the whole background.")
        if (not np.isscalar(ns) or not np.isscalar(sigma8)
                or not np.isfinite(ns) or not np.isfinite(sigma8)
                or sigma8 <= 0.0):
            raise ValueError("Require finite ns and positive finite sigma8.")
        if not isinstance(source_readers, (bool, np.bool_)):
            raise ValueError("source_readers must be an explicit Boolean.")
        if (not np.isfinite(radius_rtol) or radius_rtol <= 0.0
                or radius_rtol >= 1.0):
            raise ValueError("radius_rtol must lie between zero and one.")
        self.ns = float(ns)
        self.sigma8 = float(sigma8)
        self.fit_nz = _integer(value=fit_nz, name="fit_nz", minimum=4)
        self.table_boost = _integer(
            value=table_boost, name="table_boost", minimum=1)
        self.source_readers = bool(source_readers)
        self.variance_nodes = _integer(
            value=variance_nodes, name="variance_nodes", minimum=4)
        self.slope_nodes = _integer(
            value=slope_nodes, name="slope_nodes", minimum=4)
        self.radius_rtol = float(radius_rtol)
        logk = np.log(self.k)
        uniform_segments(coordinates=logk, name="BiHalofit log k", maximum=1)
        natural = not self.source_readers
        self._linear = _Curve(x=logk, y=np.log(self.p_linear0), natural=natural)
        self._growth = _Curve(x=self.z, y=self.growth, natural=natural)
        omega = _Curve(x=self.omega_m_redshift, y=self.omega_m_z, natural=natural)
        if np.array_equal(self.omega_m_redshift, self.z):
            self.background_omega_m_z = self.omega_m_z.copy()
        else:
            self.background_omega_m_z = omega.read(coordinates=self.z)
        if np.any(self.background_omega_m_z <= 0.0):
            raise ValueError("Interpolated Omega_m must be positive.")

        if variance is None:
            # The reference first normalizes with a 1e-3 h/Mpc lower bound.
            # Fit variances then use 1e-4. Keeping both bounds is necessary for
            # equation comparisons; replacing the normalization is separate work.
            normalization = self._variance(
                radius=np.asarray(8.0), growth_squared=1.0, window=0,
                logk_min=np.log(1.0e-3), count=self.variance_nodes, tail=False)
            self.power_normalization = self.sigma8**2 / float(normalization)
            self._linear.shift(offset=np.log(self.power_normalization))
            anchors = fit_redshifts(zmax=self.z[-1], fit_nz=self.fit_nz,
                                    source_readers=self.source_readers)
            growth_squared = self._growth.read(coordinates=anchors)**2
            radius = self._nonlinear_radius(growth_squared=growth_squared)
            spectral = -3.0 + 2.0 * self._variance(
                radius=radius, growth_squared=growth_squared, window=2,
                logk_min=np.log(1.0e-4), count=self.slope_nodes, tail=False)
            sigma8z = np.sqrt(self._variance(
                radius=np.full(anchors.shape, 8.0),
                growth_squared=growth_squared, window=0,
                logk_min=np.log(1.0e-4), count=self.variance_nodes, tail=False))
        else:
            if set(variance) != {"z", "rnl", "neff", "sigma8"}:
                raise ValueError("Shared fit variance requires exactly z/rnl/neff/sigma8.")
            anchors = _axis(values=variance["z"], name="shared fit redshift")
            uniform_segments(coordinates=anchors, name="shared fit redshift")
            if anchors.size < 4 or anchors[0] != 0.0 or anchors[-1] != self.z[-1]:
                raise ValueError("Shared fit anchors must span exactly 0..background zmax with at least four nodes.")
            radius = _values(values=variance["rnl"], shape=anchors.shape,
                             name="shared nonlinear radius", positive=True)
            spectral = _values(values=variance["neff"], shape=anchors.shape,
                               name="shared effective index", positive=False)
            sigma8z = _values(values=variance["sigma8"], shape=anchors.shape,
                              name="shared sigma8(z)", positive=True)
            self.power_normalization = 1.0
        # Supplied anchors change who evaluates the smoothing integral,
        # not the interpolation convention selected for the comparison.
        natural = not self.source_readers
        self._radius = _Curve(x=anchors, y=radius, natural=natural)
        self._neff = _Curve(x=anchors, y=spectral, natural=natural)
        self._sigma8z = _Curve(x=anchors, y=sigma8z, natural=natural)
        self._omega = _Curve(x=anchors, y=omega.read(coordinates=anchors),
                             natural=natural)
        self.fit_anchors = anchors
        self.radius_anchors = radius
        self.neff_anchors = spectral
        self.sigma8_anchors = sigma8z
        if not self.source_readers:
            for curve in (self._linear, self._growth, self._radius,
                          self._neff, self._sigma8z, self._omega):
                curve.prepare_uniform(boost=self.table_boost)

    def _variance(self, radius, growth_squared, window, logk_min, count, tail):
        """Evaluate the independent finite-grid reference on this spectrum."""
        return reference_smoothed_variance(
            linear=self._linear, radius=radius, growth_squared=growth_squared,
            window=window, logk_min=logk_min, count=count, tail=tail,
        )

    def _nonlinear_radius(self, growth_squared):
        """Solve sigma_G(R,z)=1 with the legacy bracket and stop criterion."""
        lower = np.full(growth_squared.shape, 0.01)
        upper = np.full(growth_squared.shape, 1.0e4)
        options = {
            "growth_squared": growth_squared,
            "window": 1,
            "logk_min": np.log(1.0e-4),
            "count": self.variance_nodes,
            "tail": True,
        }
        at_lower = self._variance(radius=lower, **options) - 1.0
        for retry in range(10):
            retrying = at_lower < 0.0
            if not np.any(retrying):
                break
            lower[retrying] *= 0.5
            at_lower = self._variance(radius=lower, **options) - 1.0
        at_upper = self._variance(radius=upper, **options) - 1.0
        if np.any(at_lower < 0.0) or np.any(at_upper > 0.0):
            raise ValueError("BiHalofit nonlinear radius is not bracketed.")
        active = np.ones(lower.shape, dtype=bool)
        for iteration in range(100):
            middle = 0.5 * (lower + upper)
            value = self._variance(radius=middle, **options) - 1.0
            move_lower = active & (value > 0.0)
            move_upper = active & ~move_lower
            lower[move_lower] = middle[move_lower]
            upper[move_upper] = middle[move_upper]
            active &= upper - lower > self.radius_rtol * middle
            if not np.any(active):
                break
        if np.any(active):
            raise ValueError("The nonlinear-radius solve did not converge.")
        return 0.5 * (lower + upper)

    def _coefficients(self, redshift):
        """Evaluate Takahashi's fitted polynomials in neff and log10 sigma8."""
        spectral = self._neff.read(coordinates=redshift)
        spectral2 = spectral * spectral
        spectral3 = spectral2 * spectral
        amplitude = np.log10(self._sigma8z.read(coordinates=redshift))
        amplitude2 = amplitude * amplitude
        amplitude3 = amplitude2 * amplitude
        matter = self._omega.read(coordinates=redshift)
        return {
            "radius": self._radius.read(coordinates=redshift),
            "loga": -2.167 - 2.944 * amplitude - 1.106 * amplitude2
                    - 2.865 * amplitude3,
            "b": 10.0**(-3.428 - 2.681 * amplitude + 1.624 * amplitude2
                         - 0.095 * amplitude3),
            "c": 10.0**(0.159 - 1.107 * spectral),
            "logalpha": -4.348 - 3.006 * spectral - 0.5745 * spectral2,
            "alpha_shape": 10.0**(-0.9 + 0.2 * spectral),
            "logbeta": -1.731 - 2.845 * spectral - 1.4995 * spectral2
                       - 0.2811 * spectral3,
            "gamma": 10.0**(0.182 + 0.57 * spectral),
            "f": 10.0**(-10.533 - 16.838 * spectral - 9.3048 * spectral2
                         - 1.8263 * spectral3),
            "g": 10.0**(2.787 + 2.405 * spectral + 0.4577 * spectral2),
            "h": 10.0**(-1.118 - 0.394 * spectral),
            "m": 10.0**(-2.605 - 2.434 * amplitude + 5.71 * amplitude2),
            "n": 10.0**(-4.468 - 3.08 * amplitude + 1.035 * amplitude2),
            "mu": 10.0**(15.312 + 22.977 * spectral + 10.9579 * spectral2
                          + 1.6586 * spectral3),
            "nu": 10.0**(1.347 + 1.246 * spectral + 0.4525 * spectral2),
            "p": 10.0**(0.071 - 0.433 * spectral),
            "e": 10.0**(-0.632 + 0.646 * spectral),
            "d": 10.0**(-0.483 + 0.892 * amplitude - 0.086 * matter),
        }

    def evaluate(self, k1, k2, k3, z, component=0):
        """Evaluate broadcast triangles, returning B in (Mpc/h)^6.

        component is 0/full, 1/tree, 2/one-halo or 3/three-halo. Three
        positive finite lengths must close; redshifts stay inside the supplied
        background. All triangle operations are NumPy arrays. Sorting and
        power-of-two rescaling keep the squeezed-limit subtractions stable.
        """
        if component not in (0, 1, 2, 3):
            raise ValueError("BiHalofit component must be 0, 1, 2 or 3.")
        query_z = np.asarray(z, dtype=np.float64)
        if (np.any(~np.isfinite(query_z)) or np.any(query_z < 0.0)
                or np.any(query_z > self.z[-1])):
            raise ValueError("Bispectrum redshift lies outside the background.")
        arrays = np.broadcast_arrays(k1, k2, k3, query_z)
        sides = np.stack(arrays[:3], axis=-1).astype(np.float64)
        if np.any(~np.isfinite(sides)) or np.any(sides <= 0.0):
            raise ValueError("Bispectrum sides must be positive and finite.")
        sides.sort(axis=-1)
        maximum = sides[..., 2]
        tolerance = 32.0 * np.finfo(float).eps
        if np.any(2.0 * maximum > np.sum(sides, axis=-1) * (1.0 + tolerance)):
            raise ValueError("Bispectrum sides do not close into a triangle.")
        exponent = np.frexp(maximum)[1]
        normalized = np.ldexp(sides, -exponent[..., None])
        if np.any(normalized <= 0.0):
            raise ValueError("Triangle side ratio underflows double precision.")
        growth = self._growth.read(coordinates=query_z)
        linear = np.exp(self._linear.read(np.log(sides), endpoint_power=True))
        linear *= growth[..., None] * growth[..., None]
        if component == 1:
            return _tree_sum(sides=normalized, power=linear)
        coefficients = self._coefficients(redshift=query_z)
        small, middle, large = np.moveaxis(normalized, -1, 0)
        ratio1 = small / large
        ratio2 = (small - (large - middle)) / large
        amplitude = 10.0**(coefficients["loga"]
                          - 0.310 * ratio1**coefficients["gamma"])
        alpha = np.minimum(10.0**(coefficients["logalpha"]
                                 + coefficients["alpha_shape"] * ratio2**2),
                           1.0 - 2.0 * self.ns / 3.0)
        beta = 10.0**(coefficients["logbeta"] + 0.007 * ratio2)
        scaled = sides * coefficients["radius"][..., None]
        one_halo = np.ones(sides.shape[:-1])
        damping = np.ones(sides.shape[:-1])
        effective = np.empty_like(sides)
        # There are three physical legs. The Python loop is over those three
        # terms only; every leg evaluates the full triangle block at once.
        with np.errstate(over="ignore", under="ignore"):
            for leg in range(3):
                wave = scaled[..., leg]
                one_halo /= ((amplitude * wave**alpha
                              + coefficients["b"] * wave**beta)
                             * (1.0 + 1.0 / (coefficients["c"] * wave)))
                effective[..., leg] = (
                    (1.0 + coefficients["f"] * wave * wave)
                    / (1.0 + coefficients["g"] * wave
                       + coefficients["h"] * wave * wave) * linear[..., leg]
                    + 1.0 / ((coefficients["m"] * wave**coefficients["mu"]
                              + coefficients["n"] * wave**coefficients["nu"])
                             * (1.0 + (coefficients["p"] * wave)**(-3.0))))
                damping /= 1.0 + coefficients["e"] * wave
        three_halo = _tree_sum(sides=normalized, power=effective)
        for leg in range(3):
            next_leg = (leg + 1) % 3
            opposite = (leg + 2) % 3
            three_halo += (2.0 * coefficients["d"] * scaled[..., opposite]
                           * effective[..., leg] * effective[..., next_leg])
        three_halo *= damping
        if component == 2:
            result = one_halo
        elif component == 3:
            result = three_halo
        else:
            result = one_halo + three_halo
        if np.any(~np.isfinite(result)):
            raise ValueError("BiHalofit is not finite on this domain.")
        return result


def _pair_cosine(first, second, opposite):
    """Factor unequal squared sides before dividing by the shortest side."""
    small = np.minimum(first, second)
    large = np.maximum(first, second)
    return (((opposite - large) * (opposite + large)) / small - small) / (
        2.0 * large)


def _gravity(first, second, opposite):
    """Evaluate F2 with the C cancellation treatment for antiparallel legs."""
    small = np.minimum(first, second)
    large = np.maximum(first, second)
    cosine = _pair_cosine(first=small, second=large, opposite=opposite)
    regular = (5.0 / 7.0 + 0.5 * (cosine * (small / large)
                + (cosine / small) * large) + 2.0 * cosine * cosine / 7.0)
    difference = large - small
    one_plus = 0.5 * ((opposite - difference) / small) * (
        (opposite + difference) / large)
    cancellation = (one_plus * (3.0 + 2.0 * one_plus) / 7.0
                    + 0.5 * cosine * (difference / small)
                    * (difference / large))
    return np.where(cosine < -0.5, cancellation, regular)


def _tree_sum(sides, power):
    """Combine squeezed terms before subtracting their large parts."""
    small, first, second = np.moveaxis(sides, -1, 0)
    cosine1 = _pair_cosine(first=small, second=first, opposite=second)
    cosine2 = _pair_cosine(first=second, second=small, opposite=first)
    regular1 = (13.0 / 28.0 + 2.0 * cosine1 * cosine1 / 7.0
                + 0.5 * cosine1 * (small / first))
    regular2 = (13.0 / 28.0 + 2.0 * cosine2 * cosine2 / 7.0
                + 0.5 * cosine2 * (small / second))
    difference = np.zeros_like(small)
    unequal = first != second
    difference[unequal] = (0.25 * ((second[unequal] - first[unequal])
                                   / small[unequal])
                           * (((second[unequal] + first[unequal])
                               / small[unequal])
                              * (power[..., 1][unequal]
                                 - power[..., 2][unequal])))
    gravity = _gravity(first=first, second=second, opposite=small)
    return 2.0 * (power[..., 0] * (regular1 * power[..., 1]
                                 + regular2 * power[..., 2] + difference)
                  + gravity * power[..., 1] * power[..., 2])


def prepare_reference_bispectrum(tables, axes, fit_nz=33, table_boost=8,
                                 source_readers=False):
    """Prepare a complete B table and its consistently normalized power.

    This explicit theory operation retains the reference finite-integral
    convention. Shared-variance callers instead pass their snapshot to
    BiHalofitProducer.prepare. The returned table mapping is a new owner;
    the input spectrum is never modified. Projection constructors do no fit.
    """
    if "omega_m_z" not in tables:
        raise ValueError(
            "Theory B requires provider omega_m_z; regenerate the physical "
            "tables with include_bispectrum_background=True."
        )
    from .variance import ReferenceVariance

    fit, multiplier, unused = ReferenceVariance().prepare(
        tables=tables, fit_nz=fit_nz, table_boost=table_boost,
        source_readers=source_readers,
    )
    prepared = dict(tables)
    prepared["p_linear0"] = np.asarray(tables["p_linear0"]) * multiplier
    producer = BiHalofitProducer(**axes)
    baseline = producer.prepare(
        tables=prepared, fit_nz=fit_nz, table_boost=table_boost,
        source_readers=source_readers, variance=fit,
    )
    return prepared, baseline


def bispectrum_axes(zmax, counts=(49, 193, 97, 33)):
    """Return the existing supplied-theory example's uniform sampling axes.

    The domain covers log(k_middle)=-18..12 and log(k_small/k_middle)
    =-24..0, with the remaining triangle coordinate in [0,1]. Counts are
    starting settings, not a survey accuracy claim. Refine them with the
    bispectrum accuracy control and check the actual likelihood or forecast.
    Unit tests may pass smaller explicit counts to exercise ownership only.
    """
    if len(counts) != 4:
        raise ValueError("Bispectrum sampling requires four counts.")
    bounds = ((0.0, zmax), (-18.0, 12.0), (-24.0, 0.0), (0.0, 1.0))
    axes = {}
    for name, endpoints, count in zip(("z", "logk", "logu", "v"), bounds, counts):
        count = _integer(value=count, name=name + " count", minimum=4)
        axes[name] = np.linspace(start=endpoints[0], stop=endpoints[1], num=count)
    axes["boost"] = 1
    return axes


class BiHalofitProducer:
    """Retain output geometry and the latest table for an exact input state.

    z/logk/logu/v are immutable uniform output axes. boost belongs to the C
    table upsampling; table_boost in prepare() belongs to the Python fit's
    power/background readers. They refine different approximations.
    block_size bounds the number of triangles in one vectorized evaluation;
    the final four-dimensional ln B table is still an explicitly owned array.
    Changing a physical array or numerical setting rebuilds the result.
    Older returned owners remain usable after a later cosmology is prepared.
    """

    def __init__(self, z, logk, logu, v, boost=2, domain="error",
                 block_size=65536):
        self.z = _axis(values=z, name="baseline z", uniform=True)
        self.logk = _axis(values=logk, name="baseline logk", uniform=True)
        self.logu = _axis(values=logu, name="baseline logu", uniform=True)
        self.v = _axis(values=v, name="baseline v", uniform=True)
        if self.z[0] < 0.0 or self.logu[-1] > 0.0:
            raise ValueError("Baseline coordinates require z>=0 and logu<=0.")
        if self.v[0] < 0.0 or self.v[-1] > 1.0:
            raise ValueError("Baseline v must lie in [0,1].")
        self.boost = _integer(value=boost, name="baseline boost", minimum=1)
        self.block_size = _integer(
            value=block_size, name="triangle block size", minimum=1)
        if domain not in ("error", "clamp"):
            raise ValueError("Baseline domain must be error or clamp.")
        for name, axis in (("logk", self.logk), ("logu", self.logu)):
            endpoints = np.exp(axis[[0, -1]])
            if np.any(~np.isfinite(endpoints)) or np.any(endpoints <= 0.0):
                raise ValueError(f"Invalid positive baseline exp({name}).")
        for axis in (self.z, self.logk, self.logu, self.v):
            if self.boost * (axis.size - 1) + 1 > np.iinfo(np.int32).max:
                raise ValueError("Dense baseline axis exceeds the C range.")
        self.domain = domain
        self._latest_cache = None

    def prepare(self, tables, fit_nz=33, table_boost=8, source_readers=False,
                variance_nodes=1000, slope_nodes=2000, radius_rtol=1.0e-5,
                variance=None):
        """Return the table for explicit supplied arrays, reusing exact matches.

        tables contains k,p_linear0,z,growth,omega_m_z,ns,sigma8; optional
        omega_m_redshift declares the density-fraction array's own axis.
        No source distribution, IA amplitude or baryon response enters this
        DMO fit. Hashes include every consumed input and every numeric control.
        Returned background_omega_m_z is on tables['z'] for the C factory.
        """
        required = ("k", "p_linear0", "z", "growth", "omega_m_z")
        arrays = {}
        for name in required:
            if name not in tables:
                raise ValueError(f"Supplied BiHalofit inputs lack {name}.")
            arrays[name] = np.array(tables[name], dtype=np.float64,
                                    order="C", copy=True)
        arrays["omega_m_redshift"] = np.array(
            tables.get("omega_m_redshift", tables["z"]),
            dtype=np.float64, order="C", copy=True)
        for name in ("ns", "sigma8"):
            if (name not in tables or not np.isscalar(tables[name])
                    or not np.isfinite(tables[name])):
                raise ValueError(f"BiHalofit needs a finite scalar {name}.")
        controls = {
            "ns": float(tables["ns"]),
            "sigma8": float(tables["sigma8"]),
            "fit_nz": _integer(value=fit_nz, name="fit_nz", minimum=4),
            "table_boost": _integer(
                value=table_boost, name="table_boost", minimum=1),
            "variance_nodes": _integer(
                value=variance_nodes, name="variance_nodes", minimum=4),
            "slope_nodes": _integer(
                value=slope_nodes, name="slope_nodes", minimum=4),
            "radius_rtol": float(radius_rtol),
            "source_readers": bool(source_readers),
            "baseline_boost": self.boost,
            "domain": self.domain,
            "block_size": self.block_size,
        }
        if not isinstance(source_readers, (bool, np.bool_)):
            raise ValueError("source_readers must be an explicit Boolean.")
        geometry = {
            "output_z": self.z,
            "output_logk": self.logk,
            "output_logu": self.logu,
            "output_v": self.v,
        }
        all_inputs = dict(arrays)
        all_inputs.update(geometry)
        fit_variance = None
        if variance is not None:
            if set(variance) != {"z", "rnl", "neff", "sigma8"}:
                raise ValueError("Shared fit variance requires exactly z/rnl/neff/sigma8.")
            fit_variance = {}
            for name in ("z", "rnl", "neff", "sigma8"):
                if name not in variance:
                    raise ValueError(f"Shared fit variance lacks {name}.")
                all_inputs["variance_" + name] = np.array(
                    variance[name], dtype=np.float64, order="C", copy=True)
                fit_variance[name] = all_inputs["variance_" + name]
        controls["variance_source"] = "local" if variance is None else "supplied"
        hashes = input_hashes(arrays=all_inputs, settings=controls)
        key = json.dumps(hashes, sort_keys=True)
        latest = self._latest_cache
        if latest is not None and key == latest[0]:
            return latest[1]
        arrays["z"] = _axis(values=arrays["z"], name="background z")
        if self.z[-1] > arrays["z"][-1]:
            raise ValueError("Baseline table extends beyond the background.")
        model = BiHalofit(
            **arrays, ns=controls["ns"], sigma8=controls["sigma8"],
            fit_nz=controls["fit_nz"], table_boost=controls["table_boost"],
            source_readers=controls["source_readers"],
            variance_nodes=controls["variance_nodes"],
            slope_nodes=controls["slope_nodes"],
            radius_rtol=controls["radius_rtol"], variance=fit_variance)
        shape = (self.z.size, self.logk.size, self.logu.size, self.v.size)
        logarithm = np.empty(shape=shape, dtype=np.float64)
        count = self.logk.size * self.logu.size * self.v.size
        for row, redshift in enumerate(self.z):
            output = logarithm[row].reshape(-1)
            for start in range(0, count, self.block_size):
                stop = min(start + self.block_size, count)
                offsets = np.arange(start=start, stop=stop, dtype=np.intp)
                # v varies fastest in C order, then u, then the middle side.
                v_index = offsets % self.v.size
                u_index = (offsets // self.v.size) % self.logu.size
                k_index = offsets // (self.v.size * self.logu.size)
                middle = np.exp(self.logk[k_index])
                small = middle * np.exp(self.logu[u_index])
                large = middle + self.v[v_index] * small
                values = model.evaluate(k1=small, k2=middle, k3=large,
                                        z=redshift, component=0)
                if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
                    raise ValueError("Full B must be positive to supply ln B.")
                output[start:stop] = np.log(values)
        record = {
            "model": "Takahashi et al. BiHalofit, DMO",
            "reference": "https://arxiv.org/abs/1911.07886",
            "source_url": "https://github.com/git-sunao/fastnc/tree/tatt",
            "fastnc_revision": "7b2336ce37a909bed51f07f32a2a4f096e71b6ce",
            "ported_c_source_sha256": C_SOURCE_SHA256,
            "matter_field": "total",
            "growth_convention": "P_linear(k,0) times D(z)^2",
            "normalization": "legacy sigma8 renormalization; k>=1e-3 h/Mpc",
            "power_normalization": model.power_normalization,
            "omega_m_source": "explicit supplied density-fraction array",
            "input_sha256": hashes,
            "controls": controls,
        }
        if variance is not None:
            record["normalization"] = "supplied power amplitude; no scalar sigma8 rescaling"
            record["variance_source"] = "supplied owned fit arrays"
            record["growth_convention"] = "factorized tree power; fit variances supplied separately"
        # These are the seven arrays/settings already accepted by C. The
        # likelihood checks supplied theory tables before projection; no
        # special Python table class is required from a theory producer.
        native = {
            "z": self.z.copy(),
            "logk": self.logk.copy(),
            "logu": self.logu.copy(),
            "v": self.v.copy(),
            "log_bispectrum": logarithm.ravel(order="C"),
            "boost": self.boost,
            "clamp": int(self.domain == "clamp"),
        }
        record["coordinates"] = "z, ln(middle side), ln(short/middle), gap/short"
        record["bispectrum_units"] = "(Mpc/h)^6"
        record["domain"] = self.domain
        record["boost"] = self.boost
        native_arrays = {}
        for name in ("z", "logk", "logu", "v", "log_bispectrum"):
            native_arrays[name] = native[name]
        native_arrays["background_z"] = model.z.copy()
        native_arrays["background_omega_m_z"] = model.background_omega_m_z.copy()
        record["native_input_sha256"] = input_hashes(
            arrays=native_arrays,
            settings={"boost": self.boost, "clamp": native["clamp"]},
        )
        result = {
            "native": native,
            "metadata": record,
            "background_z": native_arrays["background_z"],
            "background_omega_m_z": native_arrays["background_omega_m_z"],
        }
        # Publish the key and owner together. NumPy can release Python's GIL
        # during preparation, so two callers may compute independent results.
        # A single tuple assignment cannot associate A's key with B's owner.
        # An exception leaves the last complete valid result reusable.
        self._latest_cache = (key, result)
        return result
