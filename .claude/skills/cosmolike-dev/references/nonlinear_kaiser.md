# The nonlinear Kaiser terms of the non-Limber gg engines

The switch `use_nonlinear_kaiser_GG` (cosmo2D.c; setter
`set_nonlinear_kaiser_GG`, binding `init_nonlinear_kaiser_gg`,
likelihood yaml key `nonlinear_kaiser`, default 0) replaces the linear
velocity spectra of the redshift-space (RSD) rows of every non-Limber
gg engine by their one-loop values in standard perturbation theory.
Default 0 is the previous code bit for bit. This reference holds the
design, the conventions, the measured validation and the remaining
limits. Measurements dated 2026-11-14/15, at the lssty1xdesi fiducial
point.

## Problem

Thin spectroscopic bins resolve radial modes, so the RSD rows stay
comparable to the density rows at every multipole, and the linear
Kaiser approximation of the velocity spectra becomes the leading
model question on those rows. The red-team check E3 of the
lssty1xdesi forecast measured the one-loop correction externally
(exact projections per spectrum); the switch puts the same physics in
the engines, behind a flag, so a likelihood can sample it.

## Conventions

Theta = -div v/(aH) is the velocity divergence; its linear part is
f delta_L (f = dln D/dln a, D the linear growth factor, D(0) = 1).
theta = Theta/f. The templates are theta spectra (no f); f enters
exactly once per velocity leg, through the RSD kernel -f D^m j_l''.
With D_a = D(z_a), f_a = f(z_a), the unequal-time one-loop spectra
are

    P_dTheta(k; z1, z2) = f2 [D1 D2 P_L + D1^2 D2^2 P22_dt
                              + D1 D2^3 A_t + D1^3 D2 A_d]
    P_ThetaTheta        = f1 f2 [D1 D2 P_L + D1^2 D2^2 P22_tt
                                 + (D1^3 D2 + D1 D2^3) A_t]

with A_d = P13_dd/2 and A_t = P13_tt/2 the propagator halves. Every
term is one function of z1 times one function of z2 times a fixed
k-shape: exactly separable, so the loops ride plain FFTLog rows and
never touch the low-rank SVD (which exists for the one-loop density
spectrum, whose scale-dependent growth does not separate). Both
orderings of the propagator growth factors are kept; unlike the
density's geometric mean, nothing here is a convention.

## The pieces

- **Switch** (`cosmo2D.c`): `set_nonlinear_kaiser_GG` /
  `get_use_nonlinear_kaiser_GG`, in the style of `set_lowrank_DD_GG`.
  Flag 1 refuses `include_HOD_GX = 1`; `initial_setup` resets to 0.
  Cache keys: its own bit (<< 48) in the packed `w_gg_tomo` key, and
  slot 5 of the cross-pair stash key `ggx_stash_key`, so one process
  can flip the flag and get each data vector back bitwise. The
  Fourier path recomputes every call and has no cache to key.
- **Templates** (`get_FPT_kaiser`, pt_cfastpt.c; global table
  `FPTkaiser`, structs.c/.h, like FPTbias): rows P22_dt, P22_tt, A_d,
  A_t, k, P_lin at a = 1; k in H0/c, P in (c/H0)^3. C-FAST-PT has no
  kernel table for G2 or the propagator terms (its `J_abl` ignores
  the P13 flag; the F2 G2 expansion needs the `J_{2,-2,0}` term that
  FAST-PT regularizes in Python), so the templates come from direct
  quadrature on `p_lin(k, 1)`: P22 on the half domain |k - q| >= q,
  doubled, 32 Gauss-Legendre nodes in the angle and a cubic spline of
  ln P; P13 with the closed-form brackets of Carlson, White &
  Padmanabhan (arXiv:0905.0479), summed as series of the logarithm
  below r = 0.1 and above r = 10, where the polynomial and the
  logarithm cancel. Grid: 680 nodes over k = 1.7e-5 to 100 h/Mpc
  (100 per decade), times (1 + FPTboost); `fpt_regrid` upsamples four
  times for the engines' linear reads; zero outside the range.
  Rebuilt when `cosmology.random` or `Ntable.random` changes; each
  engine warms it serially before its parallel regions.
- **Rows** (`nl_kaiser_rows`): per bin, a D^2 row and a D^3 row (the
  bin's density and RSD kernels times D and D^2), magnification slot
  masked off through the `active` mask. All three engines:
  `C_cl_tomo_core` (rows nbins + i, 2 nbins + i; per-row skip flags
  follow the bin), `C_cl_tomo_lowrank_core` (after the rank rows, in
  pass 1 and the pair pass) and `C_cl_tomo_fkem_pairs_core`.
  `nl_kaiser_loop` contracts the transformed rows against the
  templates, read once per multipole block
  (`nl_kaiser_templates_block`) on the shared wavenumber grid. The
  allocation guards key the row count (SIZE1), per pitfall 14. The
  rows exist only with the flag on, `include_RSD_GG = 1` and no HOD.
- **Limber match** (`nl_kaiser_limber`): the equal-time limit,
  D^4 (P22_dt + A_t + A_d) and D^4 (P22_tt + 2 A_t), added to the
  Limber *linear* tables of `C_gg_tomo_limber_work` and
  `C_gg_cross_tomo_limber_work` (output `table_lin` only), under the
  node mask, in a separate sum so the existing terms keep their exact
  order. With the flag on the spectrum is

      C = N[lin + loop] + L[halofit] - L[lin + loop]

  (N the exact FFTLog projection, L the engine's Limber sum), so the
  loops cancel at high l as the linear Kaiser terms do, and the
  halofit hand-off of the RSD rows stays, never counted twice. The
  full and DD Limber outputs are unchanged.

## The high-l convention is a design choice

E3's headline swap is A - B: the exact loop projection minus the
engine's whole Limber halofit hand-off on the RSD rows. The flag
instead keeps that hand-off and subtracts only the loops' own Limber
limit (A - Lloop above). The two Limber terms differ by 3.5%
(Delta z = 0.2) to 8.9% (Delta z = 0.02, the lowest BGS auto) of the
change, maximized over each layout's retained rows — small because
the W_RSD Limber kernel sits one to two orders below the exact RSD
projection on these bins. Papers quoting E3's numbers quote the A - B
definition; the code computes A - Lloop. Which definition a project
adopts is the maintainer's decision, recorded per project.

## Validation (2026-11-14/15)

"Bitwise" means float64 equality of the whole array.

| check | configuration | result |
|---|---|---|
| flag 0 = previous code (c155815) | Delta z = 0.2 example, data vector and all gg spectra, every band; also the derivative-cache final tier, the stored m0 derivative base, and lsst_y1 EXAMPLE_EVALUATE2 (real-space 3x2pt) | bitwise |
| flag 1 vs E3's exact A - Lloop | all 667 retained rows of the 23 spectra (Delta z = 0.2); all 7,516 of the 223 spectra (Delta z = 0.02); per spectrum, relative to its largest change | max 2.65e-3 / 2.67e-3, median 1.7e-4 / 1.9e-4 |
| linear limit | templates zeroed: flag 1 vs flag 0; change at A_s x 0.5 vs 0.25 x change | bitwise; 1.2e-13 of the change |
| toggles | flags 0, 1, 0, 1 in one process; fresh process at 1; real-space w_gg cache round trip | bitwise |
| engines agree | low-rank vs FKEM pairs (Delta z = 0.2); C_cl_tomo_core vs low-rank (LSST Y1 Fourier 3x2pt) | 9.9e-13 (autos), 3.7e-13 (pairs); 2.5e-12 |
| templates | C core vs E3's spectra at the nodes; brackets vs 40-digit mpmath; 100 vs 200 nodes per decade | P22 within 1e-5; 1.6e-12; 1.1e-5 |
| size of the change | chi2 against the example data vectors (shipped mask) | 0.0811 (Delta z = 0.2), 50.35 (Delta z = 0.02) |

The 0.1-0.27% residual against E3 is a smooth ratio offset of the two
pipelines' independent growth inputs (E3 reads D off its own CAMB
template ratio at k = 5e-3/Mpc; the code uses cosmolike's growth
table; the loops scale as D^4 P_L^2). FFTLog vs direct quadrature,
Limber at high l and E3's k <= 1/Mpc projection range were checked
and excluded as sources.

Runtime (eight-core laptop, AC power; contended, not benchmarks):
Delta z = 0.2 one evaluation 9.4 s -> 10.8 s; Delta z = 0.02 (90
bins) 62 s -> 82 s, peak memory about 10.4 GB; the low-rank pool
grows by 2 nbins rows (540 -> 720 on Delta z = 0.02). A template
build is 0.24 s on one thread (680 nodes).

The project-level test is lssty1xdesi
`tests/studies/test_nonlinear_kaiser.py` (default off, gg-only
change, bitwise toggles, A_s^2 scaling, the E3-record comparison).

## Remaining limits

- The templates are zero above 100 h/Mpc. One-loop SPT is unphysical
  well below that; the loops enter only through N - L and the scale
  cut decides what is kept. Stopping the quadrature at 100 h/Mpc
  costs 1.4e-5 to 2.4e-5 of A_d and A_t against E3's extended
  extrapolation.
- Einstein-de Sitter second-order kernels and a single
  scale-independent D(z), like the rest of the perturbative model.
- Galaxies keep linear bias in the RSD rows; the redshift-space map
  stays linear (no one-loop redshift-space mapping terms, no
  bias-velocity loops).
- With `include_RSD_GG = 0` the flag is inert (there are no RSD rows
  to correct). `set_nonlinear_kaiser_GG(1)` refuses an active HOD,
  but `set_include_HOD_GX(1)` does not check the Kaiser flag (the
  same one-way guard the low-rank switch has): HOD enabled after the
  flag silently drops the loop rows through the engines' gates.
