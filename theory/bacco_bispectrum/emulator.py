"""Load paired BACCO power and bispectrum responses for the feedback theory.

These routines check installed weights and physical triangle inputs. They
perform no calculation or download when imported.
"""

import hashlib
import importlib.util
from pathlib import Path
import numpy as np


def installed_bacco_weights():
    """Locate both Burger2025 weight files without triggering a download.

    Returns:
      absolute paths and SHA-256 hashes of the installed P and B weights.
    Raises:
      FileNotFoundError when Cocoa's existing package or weights are absent.
      The error is resolved through Cocoa setup, never a notebook pip command.
    """
    spec = importlib.util.find_spec(name="baccoemu")
    if spec is None:
        raise FileNotFoundError("Enable BACCOemu through Cocoa's installation keys.")
    directory = Path(spec.origin).parent / "bispectrum_baryonic_boost"
    result = {}
    for kind, filename in (("P", "powerspectra_ratio_bcm_emu_folded.pkl"),
                           ("B", "bispectra_ratio_bcm_emu_folded.pkl")):
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Cocoa setup must supply {path} before evaluation.")
        result[kind] = {"path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def checked_bacco_triangles(sides, k_range):
    """Check triangles before BACCO's float32 conversion can drop a row.

    Arguments:
      sides = positive side lengths [ntriangle,3] in h/Mpc; any leg order.
      k_range = minimum and maximum side accepted by the installed weights.
    Returns:
      sorted float32 triangles with the original row order retained.
    Raises:
      ValueError for invalid, degenerate or out-of-range triangles, including
      degeneracy caused by rounding. Side bounds alone do not certify the
      emulator's accuracy for every squeezed shape; that needs validation.
    """
    values = np.asarray(sides, dtype=float)
    bounds = np.asarray(k_range, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 1
            or not np.all(np.isfinite(values)) or np.any(values <= 0.0)
            or bounds.shape != (2,) or not np.all(np.isfinite(bounds))
            or not 0.0 < bounds[0] < bounds[1]):
        raise ValueError("Supply positive finite [ntriangle,3] sides and k bounds.")
    values = np.sort(values, axis=1)
    if (np.any(values[:, 0] + values[:, 1] <= values[:, 2])
            or np.any(values < bounds[0]) or np.any(values > bounds[1])):
        raise ValueError("BACCO triangles must be nondegenerate and inside side bounds.")
    rounded = values.astype(np.float32)
    if (np.any(rounded[:, 0] + rounded[:, 1] <= rounded[:, 2])
            or np.any(rounded.astype(float) < bounds[0])
            or np.any(rounded.astype(float) > bounds[1])):
        raise ValueError("Float32 rounding leaves BACCO's triangle/side domain.")
    return rounded


class BaccoColdResponses:
    """Load the matched Burger2025 power and bispectrum responses for cold matter.

    Cold matter means CDM plus baryons. Massive neutrinos are excluded, so
    these responses cannot be labeled total-matter responses for lensing
    without an additional approximation. This class deliberately returns
    plain diagnostic arrays, not the native total-matter FeedbackPair.

    Arguments:
      coordinates = omega_cold, omega_baryon, sigma8_cold and five BCM
        parameters M_c/eta/beta/M1_z0_cen/theta_inn. All five BCM numbers
        are log10 coordinates; negative eta/beta are expected.
    Side effects:
      Loads the two installed weight files; never downloads missing data.
      No CAMB or CosmoLike calculation occurs here.
    """

    def __init__(self, coordinates):
        """Check both parameter boxes once and retain a private coordinate copy."""
        self.weights = installed_bacco_weights()
        import baccoemu

        self.power = baccoemu.Matter_powerspectrum(
            linear=False, no_wiggles=False, nonlinear_boost=False,
            compute_sigma8=False, baryonic_model_name="Burger2025",
            baryonic_emu_path=self.weights["P"]["path"], verbose=False,
        )
        self.bispectrum = baccoemu.Matter_bispectrum(
            baryonic_emu_path=self.weights["B"]["path"], verbose=False,
        )
        self._coordinates = dict(coordinates)
        for emulator in (self.power, self.bispectrum):
            trained = emulator.emulator["baryon"]
            expected = set(trained["keys"]) - {"expfactor"}
            if set(coordinates) != expected:
                raise ValueError(f"BACCO needs exactly these coordinates: {expected}.")
            for name, bounds in zip(trained["keys"], trained["bounds"]):
                if name == "expfactor":
                    continue
                value = coordinates[name]
                if (not np.isscalar(value) or not np.isfinite(value)
                        or not bounds[0] <= value <= bounds[1]):
                    raise ValueError(f"BACCO {name} is outside {bounds}.")

    def evaluate(self, z, k, sides):
        """Evaluate both responses at one redshift without extending their domain.

        Arguments:
          z = finite scalar redshift, inside both installed calibration boxes.
          k = positive power wavenumbers [nk] in h/Mpc, inside P side bounds.
          sides = bispectrum triangles [ntriangle,3] in h/Mpc, inside B bounds.
        Returns:
          independent float64 R_P[nk], R_B[ntriangle]. The ratios are queried
          independently; no relation such as R_B=R_P**2 is imposed.
        Raises:
          ValueError for domain violations, dropped/reordered triangles,
          nonpositive outputs or an upstream extrapolation flag.
        """
        if not np.isscalar(z) or not np.isfinite(z) or z < 0.0:
            raise ValueError("BACCO redshift must be finite and nonnegative.")
        coordinates = dict(self._coordinates, expfactor=1.0 / (1.0 + z))
        for emulator in (self.power, self.bispectrum):
            trained = emulator.emulator["baryon"]
            index = trained["keys"].index("expfactor")
            low, high = trained["bounds"][index]
            if not low <= coordinates["expfactor"] <= high:
                raise ValueError("BACCO redshift is outside the installed calibration.")
        wave = np.asarray(k, dtype=float)
        bounds = self.power.emulator["baryon"]["k_range"]
        if (wave.ndim != 1 or wave.size < 1 or not np.all(np.isfinite(wave))
                or np.any(wave < bounds[0]) or np.any(wave > bounds[1])):
            raise ValueError("BACCO power k must stay inside the installed side bounds.")
        triangles = checked_bacco_triangles(
            sides=sides, k_range=self.bispectrum.emulator["baryon"]["k_range"],
        )
        returned_k, power, flags_p = self.power.get_baryonic_boost(
            k=wave, check_k_extrapolation=True, **coordinates,
        )
        returned_sides, bispectrum, flags_b = self.bispectrum.get_baryonic_boost(
            k1=triangles[:, 0], k2=triangles[:, 1], k3=triangles[:, 2],
            check_k_extrapolation=True, **coordinates,
        )
        if (np.any(flags_p) or np.any(flags_b)
                or not np.allclose(returned_k, wave, rtol=2.e-7, atol=0.0)
                or not np.array_equal(returned_sides, triangles)):
            raise ValueError("BACCO changed the requested rows or extrapolated.")
        result = []
        for values, size in ((power, wave.size), (bispectrum, triangles.shape[0])):
            array = np.array(values, dtype=float, copy=True).reshape(-1)
            if array.shape != (size,) or not np.all(np.isfinite(array)):
                raise ValueError("BACCO returned the wrong number of finite responses.")
            if np.any(array <= 0.0):
                raise ValueError("BACCO returned a nonpositive response.")
            result.append(array)
        return result
