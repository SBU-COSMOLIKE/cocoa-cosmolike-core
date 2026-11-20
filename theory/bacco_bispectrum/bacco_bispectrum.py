"""Supply paired baryonic responses through Cobaya's CAMB provider.

BiHalofit (or another matter theory) supplies the gravity-only bispectrum.
This independent block supplies R_P=P_hydro/P_DMO and R_B=B_hydro/B_DMO;
it does not replace that bispectrum. The likelihood passes both ratios to
the existing native readers, applying each response once. CAMB is run by
the YAML's camb component, never by this block.

Burger2025 predicts cold-matter responses. The existing response module
owns the declared conversion to total matter and its continuation outside
the trained domain. Those assumptions apply equally to the offline and
Cobaya routes. The covariance and quadratic IA retain their DMO model.
"""

import numpy as np
from cobaya.theory import Theory

from .response import BaccoFeedback


class bacco_bispectrum(Theory):
    """Return paired response tables for the provider's current cosmology.

    response_settings contains the existing BaccoSettings options, including
    the separate response-axis counts. background_nz and power_nk sample
    the DMO spectra used in the neutrino conversion, on uniform z/log-k
    grids. kmax_boltzmann is the actual CAMB limit in 1/Mpc. log10k_min/max
    instead set the supplied h/Mpc table, with explicit power-law extension.

    CAMBdata is requested only to obtain sigma8_cb at exactly 8 Mpc/h and
    the neutrino density from the same solved cosmology. Cobaya's generic
    sigma_R request fixes radii in Mpc before h is known. Reading the solved
    CAMB result avoids a second solve or an interpolation over radii.
    """

    response_settings = None
    background_nz = 125
    zmax = 3.1
    power_nk = 256
    log10k_min = -5.0
    log10k_max = 3.0
    kmax_boltzmann = 5.0
    input_params = [
        "M_c_baccoemu",
        "eta_baccoemu",
        "beta_baccoemu",
        "M1_z0_cen_baccoemu",
        "theta_inn_baccoemu",
    ]

    def initialize(self):
        """Prepare grids and locate installed weights; run no cosmology."""
        for name, value in (
            ("background_nz", self.background_nz),
            ("power_nk", self.power_nk),
        ):
            if (isinstance(value, bool) or not np.isfinite(value)
                    or int(value) != value or value < 4):
                raise ValueError(f"BACCO {name} must be an integer >=4.")
        if (
            not np.isfinite(self.zmax)
            or self.zmax <= 0.0
            or not np.isfinite(self.kmax_boltzmann)
            or self.kmax_boltzmann <= 0.0
            or not np.isfinite(self.log10k_min)
            or not np.isfinite(self.log10k_max)
            or not -100.0 < self.log10k_min < self.log10k_max < 100.0
        ):
            raise ValueError("BACCO needs positive zmax/kmax and ordered log-k bounds.")
        self.z = np.linspace(
            start=0.0,
            stop=self.zmax,
            num=self.background_nz,
        )
        log10k = np.linspace(
            start=self.log10k_min,
            stop=self.log10k_max,
            num=self.power_nk,
        )
        self.k = 10.0**log10k
        self.response = BaccoFeedback(settings=self.response_settings or {})

    def get_requirements(self):
        """Declare CAMB dependencies so cosmology changes invalidate this result."""
        return {
            "CAMBdata": None,
            "Hubble": {"z": [0.0]},
            "Omega_b": {"z": [0.0]},
            "Omega_cdm": {"z": [0.0]},
            "Omega_nu_massive": {"z": [0.0]},
            "Pk_interpolator": {
                "z": self.z,
                "k_max": self.kmax_boltzmann,
                "nonlinear": (False, True),
                "vars_pairs": [
                    ("delta_tot", "delta_tot"),
                    ("delta_nu", "delta_nu"),
                ],
            },
        }

    def get_can_provide(self):
        """Expose a response product, leaving matter_bispectrum to its producer."""
        return ["baryon_response"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        """Prepare total-matter P/B ratios from the current solved CAMB state.

        The saved result contains the native uniform tables and their model
        record. Older Cobaya states retain their own arrays. The response
        producer's existing content cache also checks spectra and sampling.
        """
        results = self.provider.get_CAMBdata()
        hubble = self.provider.get_Hubble(
            z=[0.0],
            units="km/s/Mpc",
        )
        little_h = float(hubble[0]) / 100.0
        omega_b = float(self.provider.get_Omega_b(z=[0.0])[0])
        omega_cdm = float(self.provider.get_Omega_cdm(z=[0.0])[0])
        omega_nu = float(self.provider.get_Omega_nu_massive(z=[0.0])[0])
        transfer_z = np.asarray(results.Params.Transfer.PK_redshifts)
        zero_index = int(np.argmin(np.abs(transfer_z)))
        if transfer_z[zero_index] != 0.0:
            raise ValueError("BACCO requires CAMB transfer functions at z=0.")
        sigma_cb = results.get_sigmaR(
            R=8.0,
            z_indices=[zero_index],
            var1="delta_nonu",
            var2="delta_nonu",
            hubble_units=True,
        )
        parameters = {
            "omega_m": omega_b + omega_cdm + omega_nu,
            "omega_b": omega_b,
            "h": little_h,
            "sigma8": float(results.get_sigma8_0()),
            "ns": float(results.Params.InitPower.ns),
            "w0": float(results.Params.DarkEnergy.w),
            "wa": float(results.Params.DarkEnergy.wa),
            "M_c_baccoemu": params_values_dict["M_c_baccoemu"],
            "eta_baccoemu": params_values_dict["eta_baccoemu"],
            "beta_baccoemu": params_values_dict["beta_baccoemu"],
            "M1_z0_cen_baccoemu": params_values_dict["M1_z0_cen_baccoemu"],
            "theta_inn_baccoemu": params_values_dict["theta_inn_baccoemu"],
        }
        metadata = {
            "parameters": dict(parameters),
            "omega_nu_h2": omega_nu * little_h**2,
            "sigma8_cb": float(np.asarray(sigma_cb).ravel()[0]),
        }
        physical_k = little_h * self.k
        nonlinear = self.provider.get_Pk_interpolator(
            var_pair=("delta_tot", "delta_tot"),
            nonlinear=True,
            extrap_kmin=float(physical_k[0]),
            extrap_kmax=max(self.kmax_boltzmann, 1.01 * physical_k[-1]),
        )
        nonlinear_power = nonlinear.P(
            z=self.z,
            k=physical_k,
        ) * little_h**3
        tables = {
            "k": self.k,
            "z": self.z,
            "p_nonlinear": nonlinear_power,
        }
        # With no massive neutrinos their transfer spectrum is zero, so it
        # must not enter the logarithmic interpolation of a positive power.
        if omega_nu > 0.0:
            neutrino = self.provider.get_Pk_interpolator(
                var_pair=("delta_nu", "delta_nu"),
                nonlinear=False,
                extrap_kmin=float(physical_k[0]),
                extrap_kmax=max(self.kmax_boltzmann, 1.01 * physical_k[-1]),
            )
            tables["p_linear_nu"] = neutrino.P(
                z=self.z,
                k=physical_k,
            ) * little_h**3
        pair = self.response.prepare(
            parameters=parameters,
            tables=tables,
            metadata=metadata,
        )
        state["baryon_response"] = {
            "native": pair.native,
            "metadata": pair.record(),
        }
        return True

    def get_baryon_response(self):
        """Return the active paired tables; the consumer must not modify them."""
        return self.current_state["baryon_response"]
