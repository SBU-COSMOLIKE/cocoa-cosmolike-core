# Intrinsic Alignment Halo Model Extension

Extension of the Cosmolike halo model to compute **intrinsic alignment (IA) power
spectra** split by galaxy population — red/blue and central/satellite — following
[Fortuna et al. 2021 (MNRAS 501, 2983; arXiv:2003.02700)](https://arxiv.org/abs/2003.02700)
and Schneider & Bridle (2010).

> **Branch:** `halo_ia_debug` (project `roman_real`)

---

## Overview

This extension adds the IA contributions that a spherical halo model predicts:

| Term | Population | Regime | Physics |
|------|------------|--------|---------|
| **II 2-halo** | red centrals | large scales | NLA / linear alignment |
| **δI 2-halo** | red centrals | large scales | NLA / linear alignment |
| **II 1-halo** | red satellites | small scales | radial alignment (Schneider & Bridle) |
| **δI 1-halo** | red satellites | small scales | radial alignment (Schneider & Bridle) |

Blue galaxies are assumed to contribute zero IA. The red fraction is applied
through a tunable sigmoid in `log10(M)` for centrals and satellites separately.

---

## What changed

### 1. `generic_interface.cpp`
- Added **`set_nuisance_halo_model`**, which configures the HOD parameters and IA
  amplitudes per lens bin.
- Added the Python-facing compute wrappers:
  `compute_n_red_cen`, `compute_n_red_sat`, `compute_p_II_1h`, `compute_p_II_2h_cen`,
  `compute_p_dI_2h_cen`, `compute_p_dI_1h`, `compute_b_red_cen`, `compute_test_u_ia_sat`.

### 2. `halo.c`
New physics functions:

| Function | Purpose | Reference |
|----------|---------|-----------|
| `f_red_cen`, `f_red_sat` | red-fraction sigmoids for centrals / satellites | — |
| `int_u_ia_sat`, `u_ia_sat` | satellite IA Fourier profile `γ̂(k\|M)` | Eqs. 16, C8–C9 |
| `test_u_ia_sat` | standalone accessor to validate `γ̂(k\|M)` vs Fig. C1 | — |
| `int_for_IA`, `I_for_IA_nointerp` | 1-halo IA mass-integral integrand + wrapper | Eqs. 17–18 |
| `n_red_sat_bar` | red-satellite number density `n̄_s` | — |
| `p_II_1h_nointerp`, `p_dI_1h_nointerp` | 1-halo satellite II and δI power spectra | Eqs. 17–18 |
| `int_for_bred_cen`, `I_bred_cen_nointerp`, `b_red_cen` | red-central effective bias | — |
| `A_nla_cen` | NLA amplitude for red centrals | Eqs. 1–2 |
| `p_II_2h_cen_nointerp`, `p_dI_2h_cen_nointerp` | 2-halo central II and δI power spectra | Eqs. 1–2 |

### 3. `interface.cpp` (`roman_real`)
- Python bindings for all `compute_*` functions above.

---

## Usage

```python
import cosmolike_roman_real_interface as ci

coverH0 = 2997.92458  # c/H0 in Mpc/h  (code-unit length scale)

# --- required setup (abbreviated) ---
# initial_setup -> init_* -> set_cosmology -> set_nuisance_* -> set_nuisance_halo_model

# --- evaluate an IA power spectrum ---
ni = 0                       # lens bin
a  = 1.0 / (1.0 + 0.5)       # scale factor at z = 0.5
k_hMpc = 0.1                 # wavenumber in h/Mpc

k_code = k_hMpc * coverH0                       # h/Mpc -> code units
P_code = ci.compute_p_II_2h_cen(ni, k_code, a)
P_hMpc3 = P_code * coverH0**3                    # code units -> (Mpc/h)^3
```

> **Units.** Internally the code uses **code units** (length `= c/H0`). All
> `compute_p_*` functions expect **k in code units** and return **P in code
> units**. Convert at the Python boundary:
> - input: `k_code = k_hMpc * coverH0`
> - output: `P_hMpc3 = P_code * coverH0**3`
>
> `compute_test_u_ia_sat` returns a dimensionless profile — apply the **k
> conversion only**, no `coverH0³` factor on the output.

---

## Model details

### Red fraction
Centrals and satellites are split red/blue with independent sigmoids:

```
f_red_cen(M) = 0.5 * (1 + tanh[(log10 M - M_trans_cen) / w_cen])
f_red_sat(M) = 0.5 * (1 + tanh[(log10 M - M_trans_sat) / w_sat])
```

The transition masses and widths are stored in `hod[ni][6..9]`. This sigmoid
parametrisation is a modelling choice for this pipeline (the paper reads red
fractions from mocks); tune or replace as needed.

### Satellite IA profile `γ̂(k|M)`
Built from the density-weighted satellite shear (Eq. 16), with:
- the **ℓ = 2** spherical Bessel term (the leading multipole; Appendix C),
- a radial alignment `γ̄(r) ∝ (r/r_vir)^b` with **b = −2** (G19),
- a small-radius floor at **r = 0.06 Mpc/h** (Eq. 20),
- **NFW-mass normalisation**, matching the paper's Appendix C convention.

### 2-halo NLA amplitude
```
A_NLA(a) = -A_IA * C1 * ρ_crit * Ω_m / D(a) * [(1+z)/(1+z0)]^η
```

---

## Validation

| Check | Expectation | Reference |
|-------|-------------|-----------|
| `b_red_cen` | ~2 (healthy red-central bias) | — |
| `test_u_ia_sat` vs k | rises ~k² at low k, peaks, then declines | Fig. C1 |
| Peak vs mass | more massive haloes peak at lower k | Fig. C1 |
| 1-halo vs 2-halo (II) | 2-halo dominates at low k; 1-halo emerges at high k | Fig. 8 |

The total II and δI spectra are the signed sums of the 1-halo and 2-halo terms
(kept separate for II and δI):

```
P_II_total  = P_II_1h_sat  + P_II_2h_cen
P_dI_total  = P_dI_1h_sat  + P_dI_2h_cen
```

---

## Known limitations / TODO

- **Amplitudes** (`A_IA`, `η`, `a_1h`) are placeholders per lens bin; set them from
  Table 3 of the paper or fit to data/simulations. Note `a_1h ~ 0.001` (not ~1).
- **ℓ = 2 only.** The paper truncates the multipole expansion at `l_max = 6`;
  extending to ℓ = 4, 6 is a percent-level refinement.
- **Halo exclusion.** The intermediate-scale truncations (Appendix B,
  k ≈ 4–6 h/Mpc) are not implemented — relevant only for k ≳ few h/Mpc.
- **HOD bins.** Bins beyond the calibrated set use placeholder values copied from
  the last calibrated bin.

---

## References

- Fortuna et al. 2021, *The halo model as a versatile tool to predict intrinsic
  alignments*, MNRAS 501, 2983 — [arXiv:2003.02700](https://arxiv.org/abs/2003.02700)
- Schneider & Bridle 2010, MNRAS 402, 2127
- Georgiou et al. 2019, A&A 628, A31 (radial alignment measurement)
