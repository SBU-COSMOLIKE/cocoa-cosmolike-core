"""Prepare a total-matter BACCO response for the existing native P/B reader.

BACCO predicts cold-matter (CDM+baryon) power and bispectrum responses.
This module declares how they are used for lensing with massive neutrinos
and beyond the trained Fourier/redshift domain. It is a response model,
not a hydrodynamic calibration of mixed neutrino or intrinsic-shape terms.
The derivation and alternatives are in the project's BACCO.md.

Every cosmology or physical feedback parameter change produces a new
paired response. Python evaluates the installed neural networks only while
preparing uniform tables; the unchanged C projector reads the completed
tables with its usual arithmetic linear interpolation. No callback to
Python or emulator inference occurs in an OpenMP worker.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Real

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .emulator import BaccoColdResponses, installed_bacco_weights



# Use the same physical coordinate names as Cocoa's bfmt theory block.
# All five coordinates are log10 values, including beta and eta.
BACCO_PARAMETERS = (
    "M_c_baccoemu",
    "eta_baccoemu",
    "beta_baccoemu",
    "M1_z0_cen_baccoemu",
    "theta_inn_baccoemu",
)
_EMULATOR_NAMES = ("M_c", "eta", "beta", "M1_z0_cen", "theta_inn")


class FeedbackPair:
    """Own uniform ratio tables and their scientific interpretation.

    Arguments:
      z, logk, logu, v = uniform increasing axes, each with >=4 nodes.
        logk=ln(k/[h/Mpc]). For sorted B sides a<=b<=c, k=b,
        u=a/b and v=(c-b)/a. z>=0, logu<=0, and 0<=v<=1.
      log_power = ln(P_hydro/P_DMO), [nz,nk].
      log_bispectrum = ln(B_hydro/B_DMO), [nz,nk,nu,nv].
      metadata = source/name and the explicit conventions checked below.
        Only total-matter, fixed-ratio, NLA-baseline feedback is supported.
        domain='clamp' explicitly holds each boundary value outside the
        measured range; domain='error' refuses those queries.
      boost = positive integer dense-grid interval multiplier. The C model
        uses natural cubic preparation and uniform linear hot-path reads.
    Returns:
      immutable owned arrays, metadata and native constructor dictionary.
    Raises:
      ValueError for incomplete provenance, unsupported physics or bad arrays.
    """

    def __init__(self, z, logk, logu, v, log_power, log_bispectrum, metadata, boost=2):
        """Validate before native construction or any physical evaluation."""
        required = {
            "name",
            "source_url",
            "matter_field",
            "cosmology_dependence",
            "ia_response",
            "amplitude_model",
            "domain",
        }
        if not isinstance(metadata, dict) or not required.issubset(metadata):
            raise ValueError(f"Feedback metadata requires {sorted(required)}.")
        choices = {
            "matter_field": "total",
            "cosmology_dependence": "fixed_ratio",
            "ia_response": "nla_only",
            "amplitude_model": "log_response",
        }
        for name, expected in choices.items():
            if metadata[name] != expected:
                raise ValueError(f"Feedback {name} must be {expected} for this model.")
        if (not isinstance(metadata["name"], str)
                or not metadata["name"].strip()
                or not isinstance(metadata["source_url"], str)
                or not metadata["source_url"].strip()
                or metadata["domain"] not in ("clamp", "error")):
            raise ValueError("Feedback needs a name, source URL and clamp/error domain.")
        if (isinstance(boost, bool) or not np.isscalar(boost)
                or not np.isfinite(boost) or int(boost) != boost or boost < 1):
            raise ValueError("Feedback boost must be a positive integer.")
        axes = {}
        for name, supplied in (("z", z), ("logk", logk), ("logu", logu), ("v", v)):
            values = np.array(supplied, dtype=np.float64, order="C", copy=True)
            if (values.ndim != 1 or values.size < 4 or not np.all(np.isfinite(values))
                    or np.any(np.diff(values) <= 0.0)):
                raise ValueError(f"Feedback {name} needs four increasing finite nodes.")
            uniform = np.linspace(values[0], values[-1], values.size)
            tolerance = 64 * np.finfo(float).eps * max(1., np.max(np.abs(values)))
            if not np.allclose(values, uniform, rtol=0., atol=tolerance):
                raise ValueError(f"Feedback {name} must be uniform; prepare it first.")
            if int(boost) * (values.size - 1) + 1 > np.iinfo(np.int32).max:
                raise ValueError("Dense feedback axis exceeds the native integer range.")
            width = float(values[-1]) - float(values[0])
            dense_intervals = int(boost) * (values.size - 1)
            if (not np.isfinite(width) or width <= 0.0
                    or width < dense_intervals / np.finfo(float).max):
                raise ValueError("Feedback axis has unrepresentable dense spacing.")
            axes[name] = values
        if axes["z"][0] < 0. or axes["logu"][-1] > 0.:
            raise ValueError("Feedback needs z>=0 and logu<=0.")
        if axes["v"][0] < 0. or axes["v"][-1] > 1.:
            raise ValueError("Feedback v must stay within [0,1].")
        power = np.asarray(log_power, dtype=np.float64)
        bispectrum = np.asarray(log_bispectrum, dtype=np.float64)
        shape = tuple(axes[name].size for name in ("z", "logk", "logu", "v"))
        if (power.shape != shape[:2] or bispectrum.shape != shape
                or not np.all(np.isfinite(power))
                or not np.all(np.isfinite(bispectrum))):
            raise ValueError("Feedback log ratios must be finite on their declared axes.")
        self.metadata_json = json.dumps(metadata, allow_nan=False, sort_keys=True)
        self.metadata = json.loads(self.metadata_json)
        self.native = dict(axes)
        self.native["log_power"] = np.ascontiguousarray(power).ravel()
        self.native["log_bispectrum"] = np.ascontiguousarray(bispectrum).ravel()
        self.native["boost"] = int(boost)
        self.native["clamp"] = int(metadata["domain"] == "clamp")

    def record(self):
        """Return an independent JSON-compatible copy of this fixed pair's provenance."""
        return json.loads(self.metadata_json)

def feedback_power_tables(model, tables, amplitude):
    """Modify only nonlinear P for the existing two-point CFASTPT calculation.

    Arguments:
      model = C model owning the paired response table.
      tables = DMO physical-table dictionary with [nz,nk] nonlinear power.
      amplitude = finite common log-response amplitude used for Map3 too.
    Returns:
      a shallow dictionary copy with a new nonlinear-P array. Linear P,
      growth and all source tables are retained unchanged. The input is
      never modified, including after an A/B/A notebook sequence.
    """
    if not np.isfinite(amplitude):
        raise ValueError("Feedback amplitude must be finite.")
    result = dict(tables)
    if amplitude == 0.0:
        return result
    wave, redshift = np.meshgrid(tables["k"], tables["z"])
    ratio = model.power_response(
        k=wave.ravel(), z=redshift.ravel(), amplitude=float(amplitude),
    ).reshape(tables["p_nonlinear"].shape)
    result["p_nonlinear"] = tables["p_nonlinear"] * ratio
    return result

@dataclass(frozen=True)
class BaccoSettings:
    """Keep response sampling and continuation assumptions immutable.

    Arguments:
      neutrino_model = coherent_linear_neutrinos or cold_on_total.
      low_k_anchor, high_k_anchor = interior emulator anchors, h/Mpc.
      high_k_mode = shape_preserving (rescale all sides) or side_clamp.
      high_z_mode = hold the z=3 boundary or fade_to_unity above it.
      high_z_width = redshift interval over which the smooth fade completes.
      nz, nk, nu, nv = uniform preparation counts; separate accuracy boosts
        refine these intervals. The C reader also has its dense-table boost.
      logk_min/max, logu_min = natural-log native response-table limits.
      boost = native uniform cubic-to-linear upsampling factor.
    Raises:
      ValueError for unsupported choices or unusable numerical grids.
      The installed emulator's exact bounds are checked when it is loaded.
    """

    neutrino_model: str = "coherent_linear_neutrinos"
    low_k_anchor: float = 0.02
    high_k_anchor: float = 10.0
    high_k_mode: str = "shape_preserving"
    high_z_mode: str = "hold"
    high_z_width: float = 1.0
    nz: int = 17
    nk: int = 97
    nu: int = 33
    nv: int = 9
    logk_min: float = -15.0
    logk_max: float = 14.0
    logu_min: float = -16.0
    boost: int = 2

    def __post_init__(self):
        """Refuse a bad physical choice before allocating its response table."""
        choices = {
            "neutrino_model": ("coherent_linear_neutrinos", "cold_on_total"),
            "high_k_mode": ("shape_preserving", "side_clamp"),
            "high_z_mode": ("hold", "fade_to_unity"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"BACCO {name} must be one of {allowed}.")
        for name in ("nz", "nk", "nu", "nv", "boost"):
            value = getattr(self, name)
            minimum = 1 if name == "boost" else 4
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or int(value) != value or value < minimum):
                raise ValueError(f"BACCO {name} must be an integer >= {minimum}.")
        for name in ("low_k_anchor", "high_k_anchor", "high_z_width",
                     "logk_min", "logk_max", "logu_min"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value)):
                raise ValueError(f"BACCO {name} must be a finite number.")
        if (not 0.0 < self.low_k_anchor < self.high_k_anchor
                or self.high_z_width <= 0.0 or self.logu_min >= 0.0
                or self.logk_min + self.logu_min <= -700.0
                or not -700.0 < self.logk_min < self.logk_max < 700.0):
            raise ValueError(
                "BACCO requires ordered physical anchors and finite domains."
            )
        if (np.exp(self.logk_min) >= self.low_k_anchor
                or np.exp(self.logk_max) <= self.high_k_anchor):
            raise ValueError("The native table must extend beyond both response anchors.")


def smooth_turn_on(value):
    """Return a smooth factor from zero to one for a dimensionless interval.

    The cubic has zero slope at either endpoint. Applied to log R, it keeps
    R positive and joins a boundary response to unity without a hard step.
    """
    fraction = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return fraction**2 * (3.0 - 2.0 * fraction)


def continued_triangles(sides, settings):
    """Map physical triangles to a stated emulator boundary and return a taper.

    Arguments:
      sides = positive valid triangles [n,3] in h/Mpc; arbitrary leg order.
      settings = BaccoSettings with k anchors and high-k policy.
    Returns:
      sorted float32 interior triangles and a log-response factor [n].
      Only an all-soft triangle tapers to unity. One soft leg alone does
      not erase feedback on its two short-scale partners.
    Raises:
      ValueError for invalid physical sides or float32 boundary failures.
    """
    original = np.asarray(sides, dtype=float)
    if (original.ndim != 2 or original.shape[1] != 3
            or not np.all(np.isfinite(original)) or np.any(original <= 0.0)):
        raise ValueError("BACCO continuation needs positive finite [n,3] sides.")
    ordered = np.sort(original, axis=1)
    # A native v=1 grid node is exactly flattened. Allow that boundary;
    # the actual neural-network query below uses its one-sided interior limit.
    tolerance = 32.0 * np.finfo(float).eps * ordered[:, 2]
    if np.any(ordered[:, 0] + ordered[:, 1] < ordered[:, 2] - tolerance):
        raise ValueError("The supplied sides do not form a triangle.")
    longest = ordered[:, 2]
    if settings.high_k_mode == "shape_preserving":
        scale = np.minimum(1.0, settings.high_k_anchor / longest)
        mapped = ordered * scale[:, None]
    else:
        mapped = np.minimum(ordered, settings.high_k_anchor)
    mapped = np.maximum(mapped, settings.low_k_anchor).astype(np.float32)
    flattened = mapped[:, 2] >= mapped[:, 0] + mapped[:, 1]
    mapped[flattened, 2] = np.nextafter(
        mapped[flattened, 0] + mapped[flattened, 1], np.float32(0.0),
    )
    if np.any(mapped[:, 2] < mapped[:, 1]):
        raise ValueError("Anchors cannot represent an interior triangle in float32.")
    taper = smooth_turn_on(value=longest / settings.low_k_anchor)
    return mapped, taper


def total_power_response(cold_response, cold_weight, mode):
    """Apply an independent cold response to a declared cold/neutrino mixture.

    Arguments:
      cold_response = positive R_P,cb, any shape.
      cold_weight = a=C/sqrt(P_m), in [0,1] with matching shape. The
        neutrino amplitude fraction is 1-a; neutrino auto-power stays fixed.
      mode = coherent_linear_neutrinos or cold_on_total.
    Returns:
      positive total response. The coherent choice gives
      R_P,m=(1-a+a*sqrt(R_P,cb))^2 and preserves the component covariance's
      positivity, unlike freezing a nonzero cold-neutrino cross-spectrum.
    """
    response, weight = np.broadcast_arrays(cold_response, cold_weight)
    if (not np.all(np.isfinite(response)) or np.any(response <= 0.0)
            or not np.all(np.isfinite(weight)) or np.any(weight < 0.0)
            or np.any(weight > 1.0)):
        raise ValueError("Responses must be positive and cold weights lie in [0,1].")
    if mode == "cold_on_total":
        return np.array(response, copy=True)
    if mode != "coherent_linear_neutrinos":
        raise ValueError("Unknown BACCO neutrino response model.")
    mixed = (1.0 - weight + weight * np.sqrt(response))**2
    return np.where(weight == 1.0, response, mixed)


def total_bispectrum_response(cold_response, leg_power, cold_weight, mode):
    """Retain the calibrated cold B response within a declared mixed-B closure.

    Arguments:
      cold_response = positive independent R_B,cb[n].
      leg_power = R_P,cb[n,3], evaluated at the physical three sides.
      cold_weight = a_i[n,3] from the same amplitude mixture as power.
      mode = coherent_linear_neutrinos or cold_on_total.
    Returns:
      R_B,m[n]. Seven mixed placements use a_i*sqrt(R_P,i) or 1-a_i;
      the pure-cold placement uses prod(a_i)*R_B,cb. This is an explicit
      phenomenological closure, not a calibrated neutrino bispectrum.
      Summing nonnegative terms avoids cancellation near the massless limit.
    """
    response = np.asarray(cold_response, dtype=float)
    power = np.asarray(leg_power, dtype=float)
    weight = np.asarray(cold_weight, dtype=float)
    if (response.ndim != 1 or power.shape != (response.size, 3)
            or weight.shape != power.shape):
        raise ValueError("B response needs [n] ratios and [n,3] powers/weights.")
    # This shared check also rejects invalid modes, nonfinite values and
    # negative power responses before square roots or component products.
    total_power_response(cold_response=power, cold_weight=weight, mode=mode)
    if not np.all(np.isfinite(response)) or np.any(response <= 0.0):
        raise ValueError("Cold bispectrum response must be positive and finite.")
    if mode == "cold_on_total":
        return response.copy()
    cold = weight * np.sqrt(power)
    neutrino = 1.0 - weight
    result = np.prod(weight, axis=1) * response
    for placement in range(7):
        term = np.ones(response.size)
        for leg in range(3):
            # A set bit assigns that leg to cold matter; an unset bit
            # assigns it to the unchanged neutrino amplitude. The all-cold
            # placement (binary 111) was supplied independently above.
            term *= cold[:, leg] if placement & (1 << leg) else neutrino[:, leg]
        result += term
    return result


class BaccoFeedback:
    """Produce one immutable native response at each physical parameter point.

    Arguments:
      settings = BaccoSettings or a dictionary of its explicit options.
    Side effects:
      Checks installed weight files but loads no emulator until preparation.
      Keeps only the last paired table; old C models own their independent
      copies and remain valid after another parameter point is prepared.
    """

    def __init__(self, settings):
        """Validate settings and freeze the installed-weight identity."""
        if isinstance(settings, BaccoSettings):
            self.settings = settings
        else:
            self.settings = BaccoSettings(**settings)
        self.weights = installed_bacco_weights()
        self._cached_key = None
        self._cached_pair = None

    def parameter_key(self, parameters):
        """Return exact physical feedback coordinates, refusing missing values."""
        values = []
        for name in BACCO_PARAMETERS:
            value = parameters.get(name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value)):
                raise ValueError(f"BACCO requires finite {name}.")
            values.append(float(value).hex())
        return tuple(values)

    def verify_weights(self):
        """Refuse a changed weight file even when a retained prediction can be reused.

        Checks read file bytes but evaluate no network. Native tables retain
        their own copies; a changed installation requires a new producer and
        fresh process so its provenance cannot describe a different network.
        """
        if installed_bacco_weights() != self.weights:
            raise ValueError(
                "BACCO weights changed; construct a new producer in a fresh process."
            )

    def prepare(self, parameters, tables, metadata, accuracy=None):
        """Prepare total response tables for one exact cosmology and BCM point.

        Arguments:
          parameters = current cosmology plus all five BACCO coordinates.
          tables = current DMO arrays; P_m,NL and, at nonzero neutrino mass,
            explicit CAMB P_nu,L are required, with z and k in h/Mpc.
          metadata = current CAMB record with omega_nu_h2 and sigma8_cb.
          accuracy = global and individual response-table sampling controls.
        Returns:
          an immutable FeedbackPair for the native total-matter reader.
          Each pair is fixed at its recorded point; the producer regenerates
          it when any physics, source spectrum or sampling choice changes.
        Raises:
          ValueError for missing neutrino inputs, calibration violations,
          negative inferred cold amplitudes or unsupported source ranges.
        """
        self.verify_weights()
        controls = dict(accuracy or {})
        sampling = controls.get("accuracy_boost", 1.0)
        for name, value in controls.items():
            if not np.isfinite(value) or value < 1.0:
                raise ValueError(f"{name} must be finite and >=1.")
        feedback_key = self.parameter_key(parameters=parameters)
        # An independent caller must supply the same cosmology that produced
        # these CAMB spectra and sigma8_cb. Validate before a cache return.
        recorded = metadata.get("parameters", {})
        for name in ("omega_m", "omega_b", "h", "sigma8", "ns", "w0", "wa"):
            if name not in recorded or name not in parameters:
                raise ValueError(f"BACCO needs complete CAMB provenance for {name}.")
            value = parameters[name]
            if not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"BACCO {name} must be a finite real number.")
        defaults = {
            "mnu": 0.,
            "num_massive_neutrinos": 1,
            "nnu": 3.046,
        }
        for name in ("omega_m", "omega_b", "h", "sigma8", "ns", "w0", "wa",
                     "mnu", "num_massive_neutrinos", "nnu"):
            requested = parameters.get(name, defaults.get(name))
            produced = recorded.get(name, defaults.get(name))
            if requested != produced:
                raise ValueError(f"Regenerate CAMB tables after changing {name}.")
        for name in ("omega_nu_h2", "sigma8_cb"):
            if name not in metadata or not np.isfinite(metadata[name]):
                raise ValueError(f"Regenerate CAMB tables with BACCO input {name}.")
        h = float(parameters["h"])
        omega_m = float(parameters["omega_m"])
        if h <= 0.0 or omega_m <= 0.0:
            raise ValueError(
                "BACCO requires positive h and Omega_m; check the cosmology."
            )
        omega_nu = float(metadata["omega_nu_h2"]) / h**2
        if not 0.0 <= omega_nu < omega_m:
            raise ValueError("CAMB neutrino density must lie between zero and Omega_m.")
        fraction = omega_nu / omega_m
        if fraction > 0.0 and "p_linear_nu" not in tables:
            raise ValueError("Regenerate CAMB tables with explicit p_linear_nu.")
        # Hash the physical arrays used below, even for a caller outside
        # PhysicalProvider. A notebook can pass a mutable array directly;
        # its address alone cannot identify its current numerical contents.
        digest = hashlib.sha256()
        names = ["k", "z", "p_nonlinear"]
        if fraction > 0.0:
            names.append("p_linear_nu")
        for name in names:
            values = np.asarray(tables[name], dtype=np.float64)
            digest.update(name.encode())
            digest.update(str(values.shape).encode())
            digest.update(values.tobytes(order="C"))
        coordinates = {
            "omega_cold": omega_m - omega_nu,
            "omega_baryon": parameters["omega_b"],
            "sigma8_cold": float(metadata["sigma8_cb"]),
        }
        for name, native in zip(BACCO_PARAMETERS, _EMULATOR_NAMES):
            coordinates[native] = float(parameters[name])
        key = (feedback_key, tuple(sorted(coordinates.items())),
               float(fraction).hex(), digest.hexdigest(), controls)
        source_k = np.asarray(tables["k"], dtype=float)
        source_z = np.asarray(tables["z"], dtype=float)
        for axis, name in ((source_k, "k"), (source_z, "z")):
            if (axis.ndim != 1 or axis.size < 4 or not np.all(np.isfinite(axis))
                    or np.any(np.diff(axis) <= 0.0)):
                raise ValueError(
                    f"BACCO needs at least four increasing finite {name} nodes."
                )
        if source_k[0] <= 0.0 or source_z[0] != 0.0:
            raise ValueError(
                "BACCO tables need positive k and a redshift grid from zero."
            )
        for name in names[2:]:
            values = np.asarray(tables[name])
            if (values.shape != (source_z.size, source_k.size)
                    or not np.all(np.isfinite(values)) or np.any(values <= 0.0)):
                raise ValueError(f"BACCO needs positive finite {name}[nz,nk].")
        if key == self._cached_key:
            return self._cached_pair
        emulator = BaccoColdResponses(coordinates=coordinates)
        lower, upper = emulator.bispectrum.emulator["baryon"]["k_range"]
        settings = self.settings
        if not lower < settings.low_k_anchor < settings.high_k_anchor < upper:
            raise ValueError(
                "BACCO continuation anchors must be inside trained side bounds."
            )
        counts = {}
        for name, stage in (("nz", "feedback_z_boost"), ("nk", "feedback_k_boost"),
                            ("nu", "feedback_u_boost"), ("nv", "feedback_v_boost")):
            factor = sampling * controls.get(stage, 1.0)
            counts[name] = int(np.ceil((getattr(settings, name)-1)*factor)) + 1
        factor = sampling * controls.get("feedback_table_boost", 1.0)
        dense_boost = int(np.ceil(settings.boost * factor))
        dense_cells = 1
        for count in counts.values():
            dense_cells *= (count - 1) * dense_boost + 1
        if np.prod(list(counts.values()), dtype=float) > 8.e6 or dense_cells > 128.e6:
            raise ValueError(
                "BACCO response exceeds the preparation memory limit; refine one axis "
                "at a time or reduce the native feedback-table boost."
            )
        z = np.linspace(0.0, float(tables["z"][-1]), counts["nz"])
        logk = np.linspace(settings.logk_min, settings.logk_max, counts["nk"])
        logu = np.linspace(settings.logu_min, 0.0, counts["nu"])
        v = np.linspace(0.0, 1.0, counts["nv"])
        middle, ratio, opening = np.meshgrid(np.exp(logk), np.exp(logu), v,
                                             indexing="ij")
        shortest = middle * ratio
        longest = middle + shortest * opening
        sides = np.column_stack((shortest.ravel(), middle.ravel(), longest.ravel()))
        mapped, taper_b = continued_triangles(sides=sides, settings=settings)
        physical_k = np.exp(logk)
        mapped_k = np.clip(physical_k, settings.low_k_anchor, settings.high_k_anchor)
        taper_p = smooth_turn_on(value=physical_k / settings.low_k_anchor)
        log_power = np.empty((z.size, logk.size))
        log_bispectrum = np.empty((z.size, logk.size, logu.size, v.size))
        # Interpolation here is cold preparation, not a projection hot path.
        # Cubic log-power surfaces keep positive spectra. Outside the saved
        # k range their endpoint ratios are held; the neutrino component is
        # already small at the high-k boundary and this choice is recorded.
        source_k = np.log(source_k)
        nonlinear = RectBivariateSpline(source_z, source_k,
                                         np.log(tables["p_nonlinear"]))
        neutrino = None
        if fraction > 0.0:
            neutrino = RectBivariateSpline(source_z, source_k,
                                          np.log(tables["p_linear_nu"]))
        for index, redshift in enumerate(z):
            evaluated_z = min(float(redshift), 3.0)
            fade = 1.0
            if settings.high_z_mode == "fade_to_unity":
                fade = 1.0 - smooth_turn_on(
                    value=(redshift - 3.0) / settings.high_z_width,
                )
            cold_p, cold_b = emulator.evaluate(z=evaluated_z, k=mapped_k, sides=mapped)
            cold_p = np.exp(np.log(cold_p) * taper_p * fade)
            cold_b = np.exp(np.log(cold_b) * taper_b * fade)
            # Power responses on B's individual legs must use the same
            # continuation as P, not the k values of the mapped B triangle.
            leg_k = np.clip(sides.ravel(), settings.low_k_anchor,
                            settings.high_k_anchor)
            # Many triangles reuse a side. Evaluate each distinct power
            # wavenumber once, then put the results back in triangle order.
            unique_k, inverse = np.unique(leg_k, return_inverse=True)
            returned_k, leg_response = emulator.power.get_baryonic_boost(
                k=unique_k, **dict(coordinates, expfactor=1.0 / (1.0 + evaluated_z)),
            )
            if not np.allclose(returned_k, unique_k, rtol=2.e-7, atol=0.0):
                raise ValueError("BACCO changed the requested power grid.")
            leg_response = np.asarray(leg_response, dtype=float).reshape(-1)
            leg_response = leg_response[inverse]
            leg_response = leg_response.reshape(sides.shape)
            if np.any(leg_response <= 0.0) or not np.all(np.isfinite(leg_response)):
                raise ValueError("BACCO supplied invalid triangle-leg power responses.")
            leg_taper = smooth_turn_on(value=sides / settings.low_k_anchor)
            leg_response = np.exp(np.log(leg_response) * leg_taper * fade)
            weight_p = _cold_weights(k=physical_k, z=redshift, fraction=fraction,
                                     nonlinear=nonlinear, neutrino=neutrino,
                                     source_logk=source_k)
            weight_b = _cold_weights(k=sides, z=redshift, fraction=fraction,
                                     nonlinear=nonlinear, neutrino=neutrino,
                                     source_logk=source_k)
            log_power[index] = np.log(total_power_response(
                cold_response=cold_p, cold_weight=weight_p, mode=settings.neutrino_model,
            ))
            log_bispectrum[index] = np.log(total_bispectrum_response(
                cold_response=cold_b, leg_power=leg_response, cold_weight=weight_b,
                mode=settings.neutrino_model,
            )).reshape((logk.size, logu.size, v.size))
        record = {
            "name": "BACCO Burger2025 with declared neutrino/continuation model",
            "source_url": "https://arxiv.org/abs/2506.18974",
            "matter_field": "total",
            "cosmology_dependence": "fixed_ratio",
            "ia_response": "nla_only",
            "amplitude_model": "log_response",
            "domain": "clamp",
            "producer": "dynamic_bacco",
            "regenerated_per_physical_point": True,
            "coordinates": coordinates,
            "neutrino_fraction": fraction,
            "settings": asdict(settings),
            "counts": counts,
            "table_sha256": digest.hexdigest(),
            "dense_response_cells": dense_cells,
            "weights": {name: dict(value) for name, value in self.weights.items()},
            "B_neutrinos": "phenomenological mixed-response closure, not calibration",
            "independent_response_directions_omitted": [
                "h",
                "ns",
                "mnu",
                "w0",
                "wa",
            ],
            "neutrino_ratio_outside_table": "hold at saved k boundaries",
            "grid_fractions": {
                "one_or_more_low_sides": float(
                    np.mean(sides[:, 0] < settings.low_k_anchor)
                ),
                "all_soft": float(np.mean(sides[:, 2] < settings.low_k_anchor)),
                "high_side": float(np.mean(sides[:, 2] > settings.high_k_anchor)),
            },
        }
        pair = FeedbackPair(z=z, logk=logk, logu=logu, v=v, log_power=log_power,
                            log_bispectrum=log_bispectrum, metadata=record,
                            boost=settings.boost)
        self._cached_key = key
        self._cached_pair = pair
        return pair


def _cold_weights(k, z, fraction, nonlinear, neutrino, source_logk):
    """Infer a cold amplitude from total nonlinear and linear-neutrino powers.

    Arguments:
      k = physical h/Mpc queries, any shape; z = scalar redshift.
      fraction = exact CAMB Omega_nu/Omega_m.
      nonlinear, neutrino = log-power spline surfaces; neutrino may be None
        only for fraction=0. source_logk bounds the supplied table domain.
    Returns:
      a=1-f_nu*sqrt(P_nu,L/P_m,NL), same shape as k.
    Raises:
      ValueError when the neutrino amplitude exceeds total nonlinear matter.
    """
    values = np.asarray(k)
    if fraction == 0.0:
        return np.ones_like(values)
    wave = np.clip(np.log(values.ravel()), source_logk[0], source_logk[-1])
    redshifts = np.full(wave.size, z)
    neutrino_weight = fraction * np.exp(0.5 * (
        neutrino.ev(redshifts, wave) - nonlinear.ev(redshifts, wave)
    ))
    if (not np.all(np.isfinite(neutrino_weight)) or np.any(neutrino_weight > 1.0)
            or np.any(neutrino_weight < 0.0)):
        raise ValueError("Linear neutrino power leaves no nonnegative cold amplitude.")
    return (1.0 - neutrino_weight).reshape(values.shape)
