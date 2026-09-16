"""Circular amplitudes in explicit distance units."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _check_scale(distance_scale: float) -> None:
    if not np.isfinite(distance_scale) or distance_scale <= 0:
        raise ValueError("distance_scale must be finite and positive")


def circular_amplitudes(
    cos_inclination: ArrayLike, distance: ArrayLike, *, distance_scale: float
) -> tuple[tuple[NDArray[np.float64], ...], NDArray[np.float64]]:
    """Convert cosine inclination and distance to circular amplitudes.

    Parameters
    ----------
    cos_inclination : array_like
        Inclination cosine c in [-1, 1].
    distance : array_like
        Positive distance D, broadcastable with c.
    distance_scale : float
        Positive finite reference scale S, in the same units as D.

    Returns
    -------
    coordinates : tuple of ndarray
        (right, left) = (S*(1+c)**2/(4*D), S*(1-c)**2/(4*D)).
        Each array has the broadcast shape, including () for scalar inputs.
    log_jacobian : ndarray
        log|d(right, left)/d(c, D)|, with the broadcast shape.

    Notes
    -----
    Exact face-on/off inclination has a singular determinant (-inf in log
    space). Invalid physical inputs retain NumPy nonfinite arithmetic; no
    clipping or regularization is applied.
    """
    _check_scale(distance_scale)
    c, d = np.broadcast_arrays(
        np.asarray(cos_inclination, dtype=float),
        np.asarray(distance, dtype=float),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        right = distance_scale * (1 + c) ** 2 / (4 * d)
        left = distance_scale * (1 - c) ** 2 / (4 * d)
        jac = (
            2 * np.log(distance_scale)
            + np.log1p(-(c**2))
            - np.log(4.0)
            - 3 * np.log(d)
        )
    return (np.asarray(right), np.asarray(left)), np.asarray(jac)


def inverse_circular_amplitudes(
    right: ArrayLike, left: ArrayLike, *, distance_scale: float
) -> tuple[tuple[NDArray[np.float64], ...], NDArray[np.float64]]:
    """Recover cosine inclination and distance from circular amplitudes.

    Parameters
    ----------
    right, left : array_like
        Broadcastable, finite, strictly positive circular amplitudes.
    distance_scale : float
        Positive finite scale in the desired distance units; match the forward
        map's scale.

    Returns
    -------
    coordinates : tuple of ndarray
        (cos_inclination, distance), each with the broadcast shape, including
        () for scalars. Invalid pairs use cosine 0.5 and distance -scale.
    log_jacobian : ndarray
        log|d(c, D)/d(right, left)|, same shape; -inf for invalid pairs.

    Notes
    -----
    Zero amplitudes are rejected, including exact face-on/off forward images.
    Negative-distance placeholders must be rejected by the downstream prior.
    """
    _check_scale(distance_scale)
    ar, al = np.broadcast_arrays(
        np.asarray(right, dtype=float), np.asarray(left, dtype=float)
    )
    valid = np.isfinite(ar) & np.isfinite(al) & (ar > 0) & (al > 0)
    r, root_left = (
        np.sqrt(np.where(valid, ar, 1.0)),
        np.sqrt(np.where(valid, al, 1.0)),
    )
    total = r + root_left
    c, d = (r - root_left) / total, distance_scale / total**2
    jac = (
        np.log(4.0)
        + 3 * np.log(d)
        - 2 * np.log(distance_scale)
        - (np.log(4.0) + np.log(r) + np.log(root_left) - 2 * np.log(total))
    )
    return (
        np.where(valid, c, 0.5),
        np.where(valid, d, -distance_scale),
    ), np.where(valid, jac, -np.inf)
