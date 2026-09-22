"""Aligned-spin coordinate changes with the mass ratio retained."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _inputs(
    first: ArrayLike, second: ArrayLike, mass_ratio: ArrayLike
) -> tuple[NDArray, NDArray, NDArray]:
    arrays = [np.asarray(v) for v in (first, second, mass_ratio)]
    if any(v.dtype.kind not in "biuf" for v in arrays):
        raise ValueError(
            "Spin coordinates and mass ratio must be real numbers"
        )
    dtype = np.result_type(*[v.dtype for v in arrays])
    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(float)
    first, second, q = np.broadcast_arrays(
        *[np.asarray(v, dtype=dtype) for v in arrays]
    )
    return first, second, q


def aligned_spin_coordinates(
    chi1z: ArrayLike, chi2z: ArrayLike, mass_ratio: ArrayLike
) -> tuple[tuple[NDArray, NDArray], NDArray]:
    """Map component spins to effective spin and half their difference.

    Parameters
    ----------
    chi1z, chi2z : array_like
        Finite dimensionless component spins. Physical bounds are the
        caller's responsibility; the algebra also applies outside [-1, 1].
    mass_ratio : array_like
        Retained q = m2/m1, finite and in (0, 1]. Inputs must broadcast.

    Returns
    -------
    coordinates : tuple of ndarray
        chi_eff = (chi1z + q*chi2z)/(1+q), chi_diff = (chi1z-chi2z)/2.
    log_jacobian : ndarray
        Forward log-absolute determinant -log(2), also when q varies.

    Notes
    -----
    NumPy arrays have the broadcast shape, including () for scalars, and a
    common floating dtype (float32 and float64 supported; integers convert
    to float). Inputs are not mutated. Invalid inputs raise ValueError.
    The component-spin prior induces coupled, q-dependent support.
    """
    a, b, q = _inputs(chi1z, chi2z, mass_ratio)
    if np.any(~np.isfinite(a) | ~np.isfinite(b)) or np.any(
        ~np.isfinite(q) | (q <= 0) | (q > 1)
    ):
        raise ValueError("Require finite spins and mass ratio 0 < q <= 1")
    with np.errstate(over="ignore", invalid="ignore"):
        eff = (a + q * b) / (1 + q)
        diff = (a - b) / 2
    if np.any(~np.isfinite(eff) | ~np.isfinite(diff)):
        raise ValueError("Spin coordinates overflow the floating dtype")
    jac = np.full_like(eff, -np.log(np.asarray(2, dtype=eff.dtype)))
    return (np.asarray(eff), np.asarray(diff)), np.asarray(jac)


def inverse_aligned_spin_coordinates(
    chi_eff: ArrayLike, chi_diff: ArrayLike, mass_ratio: ArrayLike
) -> tuple[tuple[NDArray, NDArray], NDArray]:
    """Recover component spins with the original mass ratio retained.

    Parameters
    ----------
    chi_eff, chi_diff : array_like
        Effective spin and half the component-spin difference.
    mass_ratio : array_like
        Retained q = m2/m1 in (0, 1]. All inputs must broadcast.

    Returns
    -------
    coordinates : tuple of ndarray
        chi1z = chi_eff + 2*q*chi_diff/(1+q),
        chi2z = chi_eff - 2*chi_diff/(1+q). Shapes and dtypes follow
        aligned_spin_coordinates. Invalid rows return NaN coordinates.
    log_jacobian : ndarray
        Inverse log-absolute determinant +log(2); -inf for invalid rows.

    Notes
    -----
    Nonfinite inputs/results and q outside (0, 1] are invalid proposals.
    Finite recovered spins are not clipped or checked against physical
    bounds; the caller must enforce the original component-spin prior.
    """
    eff, diff, q = _inputs(chi_eff, chi_diff, mass_ratio)
    valid = (
        np.isfinite(eff)
        & np.isfinite(diff)
        & np.isfinite(q)
        & (q > 0)
        & (q <= 1)
    )
    safe_q = np.where(valid, q, 1)
    with np.errstate(invalid="ignore", over="ignore"):
        a = eff + 2 * safe_q * diff / (1 + safe_q)
        b = eff - 2 * diff / (1 + safe_q)
    valid &= np.isfinite(a) & np.isfinite(b)
    jac = np.full_like(eff, np.log(np.asarray(2, dtype=eff.dtype)))
    return (
        np.asarray(np.where(valid, a, np.nan)),
        np.asarray(np.where(valid, b, np.nan)),
    ), np.asarray(np.where(valid, jac, -np.inf))
