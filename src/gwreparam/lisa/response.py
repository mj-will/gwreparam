"""Affine frozen-response helicity coordinates, independent of sampler parameter names."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _latitude(sin_latitude: ArrayLike) -> NDArray:
    b = np.asarray(sin_latitude)
    if not np.issubdtype(b.dtype, np.floating):
        b = b.astype(float)
    if np.any(~np.isfinite(b) | (b < 0) | (b > 1)):
        raise ValueError(
            "Folded sine latitude must be finite and in [0, 1]; "
            "apply the LISA hemisphere reflection before this transform."
        )
    return b


def affine_frozen_response_log_jacobian(sin_latitude: ArrayLike) -> NDArray:
    """Return log|dH_det/dH_src| = -2 log(1+kappa) at fixed folded sky.

    Parameters
    ----------
    sin_latitude : array_like
        Folded sine latitude in [0, 1], arbitrary shape. No clipping is used.

    Returns
    -------
    ndarray
        Forward log determinant, same shape and floating dtype as latitude.
    """
    b = _latitude(sin_latitude)
    return np.asarray(-2 * np.log1p(((1 - b) / (1 + b)) ** 2))


def inverse_affine_frozen_response_log_jacobian(
    sin_latitude: ArrayLike,
) -> NDArray:
    """Return log|dH_src/dH_det|, opposite to the forward log determinant.

    Parameters
    ----------
    sin_latitude : array_like
        Folded sine latitude in [0, 1], arbitrary shape.

    Returns
    -------
    ndarray
        Inverse log determinant, same shape and floating dtype as latitude.
    """
    return -affine_frozen_response_log_jacobian(sin_latitude)


def _inputs(
    x: ArrayLike,
    y: ArrayLike,
    sin_latitude: ArrayLike,
    longitude: ArrayLike,
    hand: ArrayLike,
) -> tuple[NDArray, ...]:
    h = np.asarray(hand)
    if not np.issubdtype(h.dtype, np.integer) or not np.isin(h, [-1, 1]).all():
        raise ValueError("Invalid hand branch labels: expected integers -1/+1")
    b = _latitude(sin_latitude)
    # Exclude discrete labels from dtype promotion (int64 would widen float32).
    dtype = np.result_type(
        *[np.asarray(v).dtype for v in (x, y, b, longitude)]
    )
    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(float)
    x, y, b, lam, s = np.broadcast_arrays(
        *[np.asarray(v, dtype=dtype) for v in (x, y, b, longitude, h)]
    )
    if not np.isfinite(lam).all():
        raise ValueError("Longitude must be finite")
    angle = 4 * s * (lam % (2 * np.pi))
    return x, y, ((1 - b) / (1 + b)) ** 2, np.cos(angle), np.sin(angle)


def affine_frozen_response(
    x: ArrayLike,
    y: ArrayLike,
    sin_latitude: ArrayLike,
    longitude: ArrayLike,
    *,
    hand: ArrayLike,
) -> tuple[tuple[NDArray, NDArray], NDArray]:
    """Map H_src to affine frozen-response helicity coordinates H_det.

    Parameters
    ----------
    x, y : array_like
        Cartesian source helicity coordinates H_src, |H_src| <= 1.
    sin_latitude : array_like
        Already folded sine latitude in [0, 1].
    longitude : array_like
        Folded longitude in radians, periodic modulo 2*pi.
    hand : array_like of int
        Retained handedness branch, -1/+1; never inferred from H_src.

    Returns
    -------
    coordinates : tuple of ndarray
        H_det_x, H_det_y with the broadcast shape and common floating dtype.
    log_jacobian : ndarray
        Forward log determinant, broadcast shape and common floating dtype.

    Notes
    -----
    NumPy CPU arrays are supported. Inputs are not mutated. The affine map is
    defined beyond the disk; physical support is the caller's responsibility.
    H_det = exp(-4j*hand*longitude)*(H_src+kappa)/(1+kappa), where
    kappa = ((1-sin_latitude)/(1+sin_latitude))**2. H_det lies in a
    sky-dependent translated disk, not an independently uniform unit disk. See docs/affine_frozen_response.rst for response assumptions.
    """
    x, y, k, c, s = _inputs(x, y, sin_latitude, longitude, hand)
    return (
        (
            np.asarray((c * (x + k) + s * y) / (1 + k)),
            np.asarray((c * y - s * (x + k)) / (1 + k)),
        ),
        np.asarray(-2 * np.log1p(k)),
    )


def inverse_affine_frozen_response(
    x: ArrayLike,
    y: ArrayLike,
    sin_latitude: ArrayLike,
    longitude: ArrayLike,
    *,
    hand: ArrayLike,
) -> tuple[tuple[NDArray, NDArray], NDArray]:
    """Recover source helicity coordinates H_src from H_det.

    Parameters
    ----------
    x, y : array_like
        Cartesian H_det coordinates; nonfinite values propagate for rejection.
    sin_latitude, longitude : array_like
        Retained folded sine latitude in [0, 1] and longitude in radians.
    hand : array_like of int
        Retained -1/+1 handedness label. All inputs must broadcast.

    Returns
    -------
    coordinates : tuple of ndarray
        H_src_x, H_src_y with the broadcast shape and common floating dtype.
    log_jacobian : ndarray
        Inverse log determinant, broadcast shape and common floating dtype.

    Notes
    -----
    H_src = (1+kappa)*exp(4j*hand*longitude)*H_det-kappa.
    Finite and invertible even in the LISA plane. This performs no disk
    clipping; reject recovered H_src outside physical support downstream.
    """
    x, y, k, c, s = _inputs(x, y, sin_latitude, longitude, hand)
    return (
        (
            np.asarray((1 + k) * (c * x - s * y) - k),
            np.asarray((1 + k) * (s * x + c * y)),
        ),
        np.asarray(2 * np.log1p(k)),
    )


def longitude_rotation(
    x: ArrayLike,
    y: ArrayLike,
    longitude: ArrayLike,
    *,
    hand: ArrayLike,
    inverse: bool = False,
) -> tuple[tuple[NDArray, NDArray], NDArray]:
    """Rotate Cartesian helicity coordinates by -4*hand*longitude radians.

    Inputs broadcast as NumPy arrays; hand contains integer -1/+1 labels.
    Set inverse=True for the opposite rotation. Returns (x, y) and a zero
    log determinant. Inputs are not mutated; nonfinite coordinates propagate.
    """
    x, y, _, c, s = _inputs(x, y, 1.0, longitude, hand)
    if inverse:
        s = -s
    return (c * x + s * y, c * y - s * x), np.zeros_like(x)


def longitude_phase_shift(longitude: ArrayLike, *, hand: ArrayLike) -> NDArray:
    """Return the additive phase shift 2*hand*longitude in radians.

    Longitude must be finite and hand contains integer -1/+1 labels.
    Inputs broadcast. Add/subtract this shift and wrap the phase modulo 2*pi
    for forward/inverse longitude alignment, with unit phase derivative.
    """
    _inputs(0.0, 0.0, 1.0, longitude, hand)
    return np.asarray(2 * np.asarray(hand) * np.asarray(longitude))


def _dominant_response(
    x: ArrayLike, y: ArrayLike, sin_latitude: ArrayLike
) -> tuple[NDArray, NDArray, NDArray]:
    b = _latitude(sin_latitude)
    x, y, b = np.broadcast_arrays(x, y, b)
    kappa = ((1 - b) / (1 + b)) ** 2
    return 1 + kappa * x, kappa * y, b


def frozen_response_phase_correction(
    x: ArrayLike, y: ArrayLike, sin_latitude: ArrayLike
) -> NDArray:
    """Return arg(1+kappa*H_src), in radians, with broadcast NumPy shape.

    Supply Cartesian source helicity coordinates and folded sine latitude in [0, 1].
    Add/subtract the result for forward/inverse phase correction; its phase
    derivative is one. At the exact response null the phase is undefined
    (NaN), not regularized. Inputs are not mutated.
    """
    real, imag, _ = _dominant_response(x, y, sin_latitude)
    return np.asarray(
        np.where((real == 0) & (imag == 0), np.nan, np.arctan2(imag, real))
    )


def frozen_response_log_amplitude_correction(
    x: ArrayLike, y: ArrayLike, sin_latitude: ArrayLike
) -> NDArray:
    """Return log((1+b)**2/4 * abs(1+kappa*H_src)).

    Inputs are broadcast Cartesian source helicity coordinates and folded sine latitude
    b in [0, 1]. Add/subtract this shift for forward/inverse log-amplitude
    correction, with unit log-amplitude derivative. A response null yields
    -inf; there is no clipping. Inputs are not mutated.
    """
    real, imag, b = _dominant_response(x, y, sin_latitude)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(
            2 * np.log1p(b) - np.log(4.0) + np.log(np.hypot(real, imag))
        )
