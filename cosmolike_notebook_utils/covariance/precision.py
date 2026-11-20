"""Correct an analytic joint precision using independent simulation vectors.

The precision-matrix expansion of Friedrich and Eifler (2018),
arXiv:1703.07786, expands around an analytic covariance M. Only M needs
inversion; the sample covariance can be singular. The second-order
Wishart expression subtracts the bias caused by multiplying a noisy
sample covariance by itself. Its expectation is the truncated expansion,
not the exact inverse of the unknown true covariance.

This module uses the entire joint sample covariance. Subtracting a
two-point block and treating the remainder as Wishart is not valid: a
matrix with zero variance and nonzero cross-covariance is not a covariance.
Correlated sky patches are not independent samples. The caller must
establish the independent realization count before using this estimator.
"""

import numpy as np
from scipy.linalg import solve_triangular

from cosmolike_notebook_utils.fisher import symmetric_matrix


def whiten_covariance(covariance, factor):
    """Apply L^-1 C L^-T using two triangular solves.

    Arguments:
      covariance = finite symmetric [ndata,ndata] covariance.
      factor = lower Cholesky factor of the analytic covariance.
    Returns:
      dimensionless symmetric covariance in analytic-whitened coordinates.
    Raises:
      ValueError when the shapes differ.
    """
    if covariance.shape != factor.shape:
        raise ValueError("Sample and analytic covariances have different sizes.")
    left = solve_triangular(factor, covariance, lower=True)
    result = solve_triangular(factor, left.T, lower=True).T
    return 0.5 * (result + result.T)


def _restore_precision(whitened, factor):
    """Map a dimensionless precision back to the original measurement units.

    Arguments:
      whitened = symmetric precision in the coordinates L^-1 d.
      factor = lower Cholesky factor of analytic covariance M=L L^T.
    Returns:
      L^-T whitened L^-1; no eigenvalue repair is performed.
    """
    left = solve_triangular(factor.T, whitened, lower=False)
    result = solve_triangular(factor.T, left.T, lower=False).T
    return 0.5 * (result + result.T)


def precision_expansion(analytic, sample, independent_count, order=2):
    """Estimate the first- or second-order joint precision expansion.

    With S=L^-1 sample L^-T and nu=N-1, the unbiased estimate of
    (E[S])^2 is Q=(nu^2 S^2-nu S tr(S))/(nu^2+nu-2).
    The first-order polynomial is 2I-S; the second is 3I-3S+Q.
    Their realization can be indefinite and is returned for diagnosis;
    a Fisher calculation separately requires a positive precision.

    Arguments:
      analytic = positive definite joint model covariance, fixed independently.
      sample = unbiased full sample covariance from independent realizations.
      independent_count = number of independent Gaussian summary vectors >2.
      order = 1 or 2, the expansion truncation order.
    Returns:
      precision, diagnostics containing residual spectral radius and
      the minimum eigenvalue of the whitened precision polynomial.
    Raises:
      ValueError for invalid matrices, count, order or a negative sample mode.
    """
    model = symmetric_matrix(analytic, name="Analytic covariance", positive=True)
    measured = symmetric_matrix(sample, name="Sample covariance")
    if (isinstance(independent_count, bool)
            or int(independent_count) != independent_count or independent_count <= 2):
        raise ValueError("Wishart correction requires an integer independent count >2.")
    if order not in (1, 2):
        raise ValueError("Precision expansion order must be 1 or 2.")
    factor = np.linalg.cholesky(model)
    normalized = whiten_covariance(covariance=measured, factor=factor)
    eigenvalues = np.linalg.eigvalsh(normalized)
    tolerance = 128.0 * np.finfo(np.float64).eps * np.max(np.abs(eigenvalues))
    if eigenvalues[0] < -tolerance:
        raise ValueError("Sample covariance has a negative mode; check its construction.")
    identity = np.eye(model.shape[0])
    polynomial = 2.0 * identity - normalized
    if order == 2:
        nu = float(independent_count - 1)
        quadratic = nu**2 * (normalized @ normalized)
        quadratic -= nu * normalized * np.trace(normalized)
        quadratic /= nu**2 + nu - 2.0
        polynomial = 3.0 * identity - 3.0 * normalized + quadratic
    diagnostics = {
        "estimated_residual_spectral_radius": float(np.max(np.abs(eigenvalues - 1.0))),
        "minimum_polynomial_eigenvalue": float(np.linalg.eigvalsh(polynomial)[0]),
        "independent_count": int(independent_count),
        "order": int(order),
    }
    precision = _restore_precision(whitened=polynomial, factor=factor)
    return precision, diagnostics


def split_precision_expansion(analytic, sample_first, sample_second):
    """Use independent sample groups for an unbiased quadratic product.

    Independence gives E[(S1 S2+S2 S1)/2]=(E[S])^2 even when the
    summary vectors are not Gaussian. Each group's covariance must still
    be unbiased for the same physical covariance. This removes the
    Wishart assumption for this expectation, not the truncation error.

    Arguments:
      analytic = positive definite model fixed independently of both groups.
      sample_first = unbiased covariance of the first independent group.
      sample_second = unbiased covariance of the second independent group.
    Returns:
      second-order precision estimate, which can be indefinite.
    Raises:
      ValueError for invalid shapes, symmetry, analytic positivity or a
      materially negative sample-covariance mode. Rank deficiency is allowed.
    """
    model = symmetric_matrix(analytic, name="Analytic covariance", positive=True)
    first = symmetric_matrix(sample_first, name="First sample covariance")
    second = symmetric_matrix(sample_second, name="Second sample covariance")
    factor = np.linalg.cholesky(model)
    first = whiten_covariance(covariance=first, factor=factor)
    second = whiten_covariance(covariance=second, factor=factor)
    # Each group must be a covariance even though neither needs inversion.
    # Use the same scale-relative roundoff allowance as the Wishart route;
    # a zero eigenvalue is permitted for a finite independent sample set.
    for name, normalized in (("First", first), ("Second", second)):
        eigenvalues = np.linalg.eigvalsh(normalized)
        scale = np.max(np.abs(eigenvalues))
        tolerance = 128.0 * np.finfo(np.float64).eps * scale
        if eigenvalues[0] < -tolerance:
            raise ValueError(
                f"{name} sample covariance has a negative mode; "
                "check its construction."
            )
    average = 0.5 * (first + second)
    quadratic = 0.5 * (first @ second + second @ first)
    polynomial = 3.0 * np.eye(model.shape[0]) - 3.0 * average + quadratic
    return _restore_precision(whitened=polynomial, factor=factor)


def component_precision_expansion(
    analytic, component_model, component_sample, independent_count, order=2,
):
    """Apply Friedrich--Eifler Eq12 to a separately simulated covariance B.

    The physical decomposition is C=A+B, with known covariance A and model
    M=A+Bm. A and Bm must be positive semidefinite covariance components;
    they are not the two-point and three-point blocks of C. Simulations
    supply Bhat directly with the A-generating random contribution absent.
    Subtracting a deterministic A from a full sample covariance does not
    create a Wishart Bhat and must not be used with this estimator.

    Whiten by M=L L.T. Write S=L^-1 Bhat L^-T and T=L^-1 Bm L^-T.
    The first polynomial is I-S+T. At second order add
    Q-S T-T S+T T, where Q is unbiased for (E[S]) squared. Keeping both
    multiplication orders matters: these matrices need not commute.

    Arguments:
      analytic = positive definite model M[ndata,ndata], fixed independently
          of the simulation group used to estimate component_sample.
      component_model = positive semidefinite Bm with A=M-Bm also positive
          semidefinite; the remainder can be zero.
      component_sample = unbiased Bhat from independent Gaussian B-only
          summary vectors. A singular sample covariance is permitted.
      independent_count = integer number of B-only realizations greater2.
      order = 1 or 2, the Taylor truncation order.
    Returns:
      Precision and diagnostics. No noisy matrix is inverted and no negative
      precision eigenvalue is repaired. The expectation equals the truncated
      expansion, which is generally different from the exact true precision.
    Raises:
      ValueError for invalid components, shapes, sample count or order.
    """
    model = symmetric_matrix(
        values=analytic, name="Analytic covariance", positive=True,
    )
    baseline = symmetric_matrix(
        values=component_model, name="Modeled B covariance",
    )
    measured = symmetric_matrix(
        values=component_sample, name="Sample B covariance",
    )
    if (isinstance(independent_count, bool)
            or int(independent_count) != independent_count
            or independent_count <= 2):
        raise ValueError("Component PME needs integer independent count >2.")
    if order not in (1, 2):
        raise ValueError("Precision expansion order must be 1 or 2.")
    if baseline.shape != model.shape or measured.shape != model.shape:
        raise ValueError("Every covariance component must have the M shape.")
    factor = np.linalg.cholesky(model)
    baseline = whiten_covariance(covariance=baseline, factor=factor)
    measured = whiten_covariance(covariance=measured, factor=factor)
    identity = np.eye(model.shape[0])
    # The known component may have zero modes. Test in dimensionless model
    # units so rounding a zero mode does not depend on mixed observable units.
    for name, matrix in (
        ("Modeled B covariance", baseline),
        ("Known A covariance", identity - baseline),
        ("Sample B covariance", measured),
    ):
        eigenvalues = np.linalg.eigvalsh(matrix)
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        tolerance = 128.0 * np.finfo(np.float64).eps * scale
        if eigenvalues[0] < -tolerance:
            raise ValueError(f"{name} must be positive semidefinite.")
    residual = measured - baseline
    polynomial = identity - residual
    if order == 2:
        nu = float(independent_count - 1)
        quadratic = nu**2 * (measured @ measured)
        quadratic -= nu * measured * np.trace(measured)
        quadratic /= nu**2 + nu - 2.0
        polynomial += quadratic - measured @ baseline
        polynomial -= baseline @ measured
        polynomial += baseline @ baseline
    polynomial = 0.5 * (polynomial + polynomial.T)
    diagnostics = {
        "estimated_residual_spectral_radius": float(
            np.max(np.abs(np.linalg.eigvalsh(residual)))
        ),
        "minimum_polynomial_eigenvalue": float(
            np.linalg.eigvalsh(polynomial)[0]
        ),
        "independent_count": int(independent_count),
        "order": int(order),
        "sample_scope": "separately_simulated_B_covariance",
    }
    precision = _restore_precision(whitened=polynomial, factor=factor)
    return precision, diagnostics
