"""Shared helpers for the EXAMPLE_EVALUATE notebooks of every project.

Each Cosmolike project ships Jupyter notebooks that drive the compiled
interface directly (their own CAMB run, set_cosmology, the nuisance
setters, compute_data_vector_masked). The functions that do not
depend on a project live here, once: private copies in each notebook
would drift apart, and every interface upgrade would have to be
repeated in each one. Each notebook imports them once:

    sys.path.insert(0, os.environ['ROOTDIR']
                    + '/external_modules/code/cosmolike_core')
    import cosmolike_notebook_utils as cnu

Design rule for this package: it never imports a project's compiled
interface (cosmolike_<project>_interface). Each project compiles its
own interface module, so a function that needs cosmolike receives the
notebook's interface-bound callable as an argument instead (the `dv`
and `ddv` parameters of the Fisher helpers). Shared groups:

  camb_cosmology     get_camb_cosmology: one CAMB run packaged into
                     the tuple set_cosmology consumes, on the nested
                     dyadic grids shared with the likelihoods.
  plot_datavectors   tomographic plots of the data vectors: cosmic
                     shear (plot_C_ss_tomo_limber, plot_xi),
                     galaxy-galaxy lensing (plot_C_gs_tomo_limber,
                     plot_gammat_tomo_limber), galaxy clustering
                     (plot_C_gg_tomo, plot_wtheta_tomo), and the
                     baryonic-feedback (bfmt) parameter sweeps
                     (plot_baryon_suppression).
  plot_response      plot_response_function: curves of a data
                     vector's response to the matter power spectrum,
                     both d ln DV / d ln k and cumulative R(k_max).
  plot_scale_cuts    plot_scale_cuts: a per-bin scale cut (l_max of the
                     Fourier band powers, theta_min in real space)
                     against k_max at z = 0, one curve per bin, from
                     the cuts the compiled interface computed.
  fisher             finite-difference and derivkit derivatives of a
                     data-vector function passed as an argument, the
                     five-point stencil the project drivers share
                     (FIVE_POINT_OFFSETS, FIVE_POINT_WEIGHTS,
                     five_point), Fisher-matrix assembly and algebra
                     (fisher_bias, combine, marginalized_sigma),
                     figures of merit, the FoM-noise estimate
                     (get_stencil_noise, get_fom_noise_weights,
                     predict_fom_noise: the numerical noise of a
                     derivative column and the change of a figure of
                     merit it predicts), and Fisher contour plots via
                     getdist.

The two covariance modules are not imported by this file, so
`import cosmolike_notebook_utils as cnu` does not bind them; the
covariance notebooks import them as submodules:

    from cosmolike_notebook_utils import covariance as cov
    from cosmolike_notebook_utils import plot_covariances as pcov

  covariance         component preparation, Gaussian blocks, real-space
                     and Fourier survey matrices, dense-linear lookup
                     tables and covariance eigenvalue diagnostics.
  plot_covariances   correlation triangles, component maps/histograms,
                     standard deviations per angular or multipole bin,
                     and halo trispectrum terms, from supplied arrays.

caches and mpi_batching are also imported as submodules, usually by a
project's notebook wrappers, which re-export their functions:

    from cosmolike_notebook_utils import caches, mpi_batching

  caches             caches of notebook results in a project's chains/
                     folder: read_or_build reads a cache when the file
                     is there and otherwise computes and writes it; a
                     cache made with other settings stops the read with
                     a message naming every difference, and no file is
                     replaced without overwrite=True.
  mpi_batching       independent jobs on MPI ranks: a greedy coloring
                     that groups tomographic bins into batches so no
                     batch holds both bins of a cross pair, and the
                     fixed job-to-rank assignment and loop, with the
                     MPI communicator passed in.

The notebooks keep everything project-specific: fiducial parameter
values, tomographic-bin layouts, the interface init sequence, and
thin wrappers that bind those to the functions here.
"""

from .camb_cosmology import get_camb_cosmology
from .plot_datavectors import (plot_C_gg_tomo, plot_C_gs_tomo_limber,
                               plot_C_ss_tomo_limber,
                               plot_baryon_suppression,
                               plot_gammat_tomo_limber, plot_wtheta_tomo,
                               plot_xi)
from .fisher import (add_gaussian_priors, get_Fisher, get_Fisher2,
                     get_FoM, get_ddv, get_ddv_dkit,
                     get_fom_noise_weights, get_stencil_noise,
                     plot_Fisher, plot_Fisher_amplitude_cut,
                     plot_Fisher_prior_walls, predict_fom_noise)
from .plot_response import plot_response_function
from .plot_scale_cuts import plot_scale_cuts
