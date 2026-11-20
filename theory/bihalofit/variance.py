"""Prepare owned three-point variance inputs with the shared CosmoLike engine.

The ordinary core already smooths a supplied linear spectrum with Gaussian
and top-hat windows. This adapter installs that spectrum serially, resolves
the shared sampling counts and copies the requested outputs. It leaves the
choice of evolving or factorized linear growth explicit. Neither choice
rescales power to a second sigma8 scalar.

ReferenceVariance separately retains the older finite-integral and scalar
normalization convention for independent comparisons.
"""

import hashlib
import json

from copy import deepcopy

import numpy as np

from .model import uniform_segments


def reference_matter_fraction(tables, redshift):
    """Return the legacy CPL fraction for explicit upstream comparisons only.

    This preserves the upstream equation independently of the toy distance
    tables used by those tests. Production derives a pressureless fraction
    from its expansion history or supplies its exact provider background.
    """
    scale = 1.0 / (1.0 + np.asarray(redshift, dtype=np.float64))
    dark = scale**(-1.0 - 3.0 * (tables["w0"] + tables["wa"]))
    dark *= np.exp(-3.0 * (1.0 - scale) * tables["wa"])
    omega = tables["omega_m"]
    vacuum = tables["omega_de"]
    total = 1.0 + (omega + vacuum - 1.0) / (
        1.0 - omega - vacuum + vacuum * dark + omega / scale
    )
    return total * omega / (omega + vacuum * scale * dark)


def reference_bihalofit(inputs):
    """Return the exact Python matter fit for upstream equation comparisons.

    This function does not construct a projection model or interpolate a
    four-dimensional B table. Tests of the original fit decomposition belong
    here; native projections receive only its complete bispectrum.
    """
    from .model import BiHalofit, fit_redshifts

    arguments = {}
    for name in ("k", "p_linear0", "z", "growth", "ns", "sigma8"):
        arguments[name] = inputs[name]
    for name in ("fit_nz", "table_boost", "source_readers", "variance"):
        if name in inputs:
            arguments[name] = inputs[name]
    axis = inputs.get("omega_m_redshift", inputs["z"])
    if "omega_m_z" in inputs:
        matter = inputs["omega_m_z"]
    else:
        axis = fit_redshifts(zmax=inputs["z"][-1],
                             fit_nz=inputs.get("fit_nz", 33),
                             source_readers=inputs.get("source_readers", False))
        matter = reference_matter_fraction(tables=inputs, redshift=axis)
    return BiHalofit(**arguments, omega_m_redshift=axis, omega_m_z=matter)


def reference_physics_model(factory, **inputs):
    """Construct a comparison projection from an explicit Python theory table.

    This adapter retains the upstream power normalization in Python. Its
    small B grid tests projection/state machinery, not survey convergence.
    Equation tests use reference_bihalofit directly and frozen upstream
    values; no native one-/three-halo decomposition is available.
    """
    from .model import BiHalofitProducer, bispectrum_axes

    if "omega_de" in inputs:
        if abs(inputs["omega_m"] + inputs["omega_de"] - 1.0) > 1.0e-8:
            raise ValueError(
                "flat geometry requires abs(omega_m + omega_de - 1) <= 1e-8"
            )
    prepared = dict(inputs)
    source_readers = inputs.get("source_readers", False)
    if inputs.get("variance") is not None:
        fit = dict(inputs["variance"])
        multiplier = 1.0
    elif inputs.get("baseline") is None:
        fit, multiplier, unused = ReferenceVariance().prepare(
            tables=inputs, fit_nz=inputs.get("fit_nz", 33),
            table_boost=inputs.get("table_boost", 8),
            source_readers=source_readers,
        )
    else:
        from .model import _Curve, reference_smoothed_variance

        curve = _Curve(x=np.log(inputs["k"]), y=np.log(inputs["p_linear0"]),
                       natural=not source_readers)
        value = reference_smoothed_variance(
            linear=curve, radius=np.asarray(8.0), growth_squared=1.0,
            window=0, logk_min=np.log(1.0e-3), count=1000, tail=False,
        )
        multiplier = inputs["sigma8"]**2 / float(value)
        fit = None
    prepared["p_linear0"] = np.asarray(inputs["p_linear0"]) * multiplier
    if inputs.get("baseline") is None:
        if "omega_m_z" not in inputs:
            prepared["omega_m_redshift"] = fit["z"]
            prepared["omega_m_z"] = reference_matter_fraction(
                tables=inputs, redshift=fit["z"],
            )
        axes = bispectrum_axes(zmax=inputs["z"][-1], counts=(9, 65, 33, 17))
        # Extreme squeezed stress tests reach below the example survey domain.
        axes["logu"] = np.linspace(start=-40.0, stop=0.0, num=33)
        producer = BiHalofitProducer(**axes)
        baseline = producer.prepare(
            tables=prepared, fit_nz=inputs.get("fit_nz", 33),
            table_boost=inputs.get("table_boost", 8),
            source_readers=source_readers, variance=fit,
        )
        prepared["baseline"] = dict(baseline["native"])
    prepared.pop("variance", None)
    return factory(**prepared)


class ReferenceVariance:
    """Retain the independent Python calculation for legacy comparisons.

    This reference integrates the original finite log-k rules and rescales
    power to the separate sigma8 value. SharedVariance never applies that
    rescaling. These fit arrays feed the Python theory producer; C receives
    its complete bispectrum and performs no smoothing or BiHalofit fit.
    """

    def __init__(self):
        """Start without a retained reference spectrum or fit arrays."""
        self._latest = None

    def prepare(self, tables, fit_nz=33, table_boost=8, source_readers=False):
        """Return owned fit arrays, the power multiplier and their input record.

        Arguments:
          tables = validated linear power, growth and cosmological scalars.
          fit_nz = coefficient redshift count in the uniform production mode.
          table_boost = dense-reader ratio used by the reference calculation.
          source_readers = retain the upstream cubic-reader convention.
        Returns:
          fit dictionary, scalar power multiplier and independent metadata.
        Side effects:
          Retains one immutable result; never changes core globals or inputs.
        """
        from .model import BiHalofit, input_hashes

        arguments = {}
        for name in ("k", "p_linear0", "z", "growth", "ns", "sigma8"):
            arguments[name] = tables[name]
        # BiHalofit also validates a background, although density fractions
        # do not enter its smoothing integrals. Use a supplied background
        # when available; otherwise retain the internal fit's flat CPL
        # background for this explicit legacy reference only.
        if "omega_m_z" in tables:
            matter_fraction = tables["omega_m_z"]
        else:
            redshift = np.asarray(tables["z"], dtype=np.float64)
            matter = tables["omega_m"] * (1.0 + redshift)**3
            exponent = 3.0 * (1.0 + tables["w0"] + tables["wa"])
            dark = tables["omega_de"] * (1.0 + redshift)**exponent
            dark *= np.exp(-3.0 * tables["wa"] * redshift / (1.0 + redshift))
            matter_fraction = matter / (matter + dark)
        arguments["omega_m_z"] = matter_fraction
        settings = {
            "fit_nz": fit_nz,
            "table_boost": table_boost,
            "source_readers": source_readers,
        }
        hashes = input_hashes(arrays=arguments, settings=settings)
        key = json.dumps(hashes, sort_keys=True)
        if self._latest is not None and self._latest[0] == key:
            fit, multiplier, record = self._latest[1:]
            return deepcopy(x=fit), multiplier, json.loads(record)
        model = BiHalofit(**arguments, **settings)
        fit = deepcopy(x={
            "z": model.fit_anchors,
            "rnl": model.radius_anchors,
            "neff": model.neff_anchors,
            "sigma8": model.sigma8_anchors,
        })
        multiplier = float(model.power_normalization)
        record = {
            "engine": "Python reference Simpson integration and bisection",
            "growth_mode": "factorized",
            "normalization": "legacy scalar sigma8 normalization",
            "power_normalization": multiplier,
            "sampling": {
                "variance_nodes": model.variance_nodes,
                "slope_nodes": model.slope_nodes,
                "radius_rtol": model.radius_rtol,
                **settings,
            },
            "input_sha256": hashes,
        }
        encoded = json.dumps(record, sort_keys=True)
        self._latest = (key, fit, multiplier, encoded)
        return deepcopy(x=fit), multiplier, json.loads(encoded)


class SharedVariance:
    """Own copied snapshots while reusing the core's retained FFT plans.

    Construction calls core.initial_setup and records its unboosted counts.
    Subsequent physical changes replace inputs without resetting the core,
    allowing plans of the same size to survive. Calls use the core's usual
    serial preparation contract; its builders manage their OpenMP workers.
    Previously returned snapshots remain valid after any later core changes.
    The provider's own saved baseline and Accuracy object select resolution;
    unrelated manual core settings do not redefine that baseline.
    """

    def __init__(self, compiled):
        self.compiled = compiled
        compiled.core.initial_setup()
        self.baseline = deepcopy(x=compiled.core.variance_sampling())
        self._latest = None

    def prepare(self, tables, parameters, metadata, z, accuracy=None,
                growth_mode="factorized", mass=None, reference_rho=0.0,
                include_fit=True):
        """Return a copied snapshot and its complete numerical input record.

        tables supplies k [h/Mpc], z, positive linear power [(Mpc/h)^3],
        growth D(0)=1 and omega_m. 'factorized' installs P(k,0)D(z)^2,
        retaining the three-point effective-growth approximation. 'evolving'
        requires the full supplied p_linear[z,k]; no fallback is inferred.
        parameters supplies omega_b and h. Massive-neutrino density comes
        from metadata['omega_nu_h2'], as computed by the Boltzmann provider.
        Output z and optional mass [Msun/h] are requested coordinates, not
        the core's internal interpolation nodes. The returned record states
        both. Only fit arrays should be passed to a physical-model factory.
        """
        if growth_mode not in ("factorized", "evolving"):
            raise ValueError("Variance growth_mode must be factorized or evolving.")
        if not isinstance(include_fit, (bool, np.bool_)):
            raise ValueError("Variance include_fit must be Boolean.")
        controls = self.compiled.init_3pcf_accuracy(value=accuracy)
        sampling = self.compiled.accuracy_variance_settings(accuracy=controls, baseline=self.baseline)
        wave = np.ascontiguousarray(tables["k"], dtype=np.float64)
        redshift = np.ascontiguousarray(tables["z"], dtype=np.float64)
        if np.any(wave <= 0.0):
            raise ValueError("Shared variance requires positive k in h/Mpc.")
        uniform_segments(coordinates=np.log10(wave), name="variance log10(k)", maximum=1)
        uniform_segments(coordinates=redshift, name="variance redshift")
        if redshift[0] != 0.0:
            raise ValueError("Shared variance power must include z=0.")
        if growth_mode == "evolving":
            if "p_linear" not in tables:
                raise ValueError("Evolving variance requires supplied p_linear[z,k].")
            power = np.ascontiguousarray(tables["p_linear"], dtype=np.float64)
        else:
            linear = np.asarray(tables["p_linear0"], dtype=np.float64)
            growth = np.asarray(tables["growth"], dtype=np.float64)
            if linear.shape != wave.shape or growth.shape != redshift.shape:
                raise ValueError("Factorized variance requires P(k,0) and D(z) on their declared axes.")
            if (not np.all(np.isfinite(growth)) or np.any(growth <= 0.0)
                    or abs(growth[0] - 1.0) > 1.0e-12):
                raise ValueError("Factorized variance requires positive growth with D(0)=1.")
            power = growth[:, None]**2 * linear[None, :]
        if (power.shape != (redshift.size, wave.size)
                or not np.all(np.isfinite(power)) or np.any(power <= 0.0)):
            raise ValueError("Shared linear power must be positive finite [nz,nk].")
        query = np.array(z, dtype=np.float64, order="C", copy=True)
        if (query.ndim != 1 or query.size == 0 or not np.all(np.isfinite(query))
                or np.any(np.diff(query) <= 0.0) or query[0] < redshift[0]
                or query[-1] > redshift[-1]):
            raise ValueError("Variance output redshifts must increase within the supplied power range.")
        masses = None
        if mass is not None:
            masses = np.array(mass, dtype=np.float64, order="C", copy=True)
            if (masses.ndim != 1 or masses.size == 0 or not np.all(np.isfinite(masses))
                    or np.any(masses <= 0.0) or np.any(np.diff(masses) <= 0.0)):
                raise ValueError("Variance masses must be a positive finite increasing vector.")
        if not include_fit and masses is None:
            raise ValueError("Variance request must include fit quantities or halo masses.")
        if not np.isfinite(reference_rho) or reference_rho < 0.0:
            raise ValueError("Variance reference_rho must be finite and nonnegative.")
        matter = float(tables["omega_m"])
        baryon = float(parameters["omega_b"])
        hubble = 100.0 * float(parameters["h"])
        if parameters.get("mnu", 0.0) > 0.0 and "omega_nu_h2" not in metadata:
            raise ValueError("Massive-neutrino variance needs provider omega_nu_h2 metadata.")
        neutrino = float(metadata.get("omega_nu_h2", 0.0))
        if (not np.all(np.isfinite([matter, baryon, hubble, neutrino]))
                or hubble <= 0.0 or baryon < 0.0 or matter <= baryon
                or neutrino < 0.0 or neutrino / (hubble / 100.0)**2 >= matter):
            raise ValueError("Variance cosmology requires finite physical matter/baryon/neutrino densities and h>0.")
        arrays = {"k": wave, "z": redshift, "linear_power": power, "output_z": query}
        if masses is not None:
            arrays["mass"] = masses
        hashes = {}
        for name, values in arrays.items():
            hashes[name] = hashlib.sha256(values.tobytes(order="C")).hexdigest()
        record = {
            "engine": "shared CosmoLike FFTLog",
            "growth_mode": growth_mode,
            "normalization": "supplied power amplitude; no scalar sigma8 rescaling",
            "matter_field": "total",
            "sampling": sampling,
            "omega_m": matter, "omega_b": baryon, "hubble": hubble,
            "omega_nu_h2": neutrino, "reference_rho": float(reference_rho),
            "include_fit": bool(include_fit), "input_sha256": hashes,
        }
        key = json.dumps(record, sort_keys=True)
        if self._latest is not None and key == self._latest[0]:
            return deepcopy(x=self._latest[1]), json.loads(self._latest[0])
        # Prepare the layout before installing any global state. Returning
        # an earlier cached snapshot above requires no current-core state:
        # the snapshot owns exactly the values recorded by its input key.
        flattened = np.ascontiguousarray(np.log(power).T).ravel()
        core = self.compiled.core
        core.set_variance_sampling(**sampling)
        core.set_cosmological_parameters(omega_matter=matter, omega_baryon=baryon,
                                         hubble=hubble, omega_nu_h2=neutrino)
        core.set_linear_power_spectrum(log10k=np.log10(wave), z=redshift, lnP=flattened)
        snapshot = core.variance_snapshot(z=query, mass=masses,
                                          reference_rho=reference_rho,
                                          include_fit=bool(include_fit))
        owned = deepcopy(x=snapshot)
        self._latest = (key, owned)
        return deepcopy(x=owned), json.loads(key)
