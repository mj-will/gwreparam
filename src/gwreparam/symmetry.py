"""Deterministic periodic folds; branch probabilities belong to consumers."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_modes(modes: ArrayLike, n_modes: int) -> NDArray[np.int64]:
    values = np.asarray(modes)
    if not np.issubdtype(values.dtype, np.integer) or np.any(
        (values < 0) | (values >= n_modes)
    ):
        raise ValueError("Branch labels must be integers in the valid range.")
    return values.astype(np.int64, copy=False)


def _span(period: float, n_modes: int) -> float:
    if (
        not np.isfinite(period)
        or period <= 0
        or isinstance(n_modes, (bool, np.bool_))
        or not isinstance(n_modes, (int, np.integer))
        or n_modes < 1
    ):
        raise ValueError("Require a positive period and integer mode count.")
    return period / n_modes


def fold_periodic(
    values: ArrayLike, period: float, n_modes: int, *, origin: float = 0.0
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Fold a periodic coordinate into equal-width branches.

    Parameters
    ----------
    values : array_like
        Finite real coordinates, in the same units as period and origin.
    period : float
        Positive finite full period (e.g. 2*pi for a radian angle).
    n_modes : int
        Positive number of equal-width branches.
    origin : float, optional
        Finite lower endpoint of the full and folded intervals.

    Returns
    -------
    values : ndarray
        Folded coordinates in [origin, origin+period/n_modes), same shape as
        input. Scalar inputs produce zero-dimensional arrays.
    log_jacobian : ndarray
        Zeros, with the input shape.
    labels : ndarray of int64
        Branches in [0, n_modes), determined modulo the full period.

    Raises
    ------
    ValueError
        If coordinates, period, origin or branch count are invalid.
    """
    span = _span(period, n_modes)
    values = np.asarray(values, dtype=float)
    if not np.isfinite(origin) or not np.isfinite(values).all():
        raise ValueError(
            "Periodic folding requires finite coordinates and origin."
        )
    normalized = np.mod(values - origin, period)
    modes = np.minimum(
        np.floor(normalized / span).astype(np.int64), n_modes - 1
    )
    return (
        np.asarray(np.mod(values - origin, span) + origin),
        np.zeros_like(values),
        np.asarray(modes, dtype=np.int64),
    )


def unfold_periodic(
    values: ArrayLike,
    period: float,
    n_modes: int,
    modes: ArrayLike,
    *,
    origin: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Invert a periodic fold using explicit branch labels.

    Parameters
    ----------
    values : array_like
        Folded coordinates, normally in [origin, origin+period/n_modes).
        Nonfinite values propagate.
    period : float
        Positive finite full period, in the coordinate's units.
    n_modes : int
        Positive branch count matching the forward fold.
    modes : array_like of int
        Labels in [0, n_modes), broadcastable with values.
    origin : float, optional
        Finite lower endpoint matching the forward fold.

    Returns
    -------
    values : ndarray
        Reconstructed coordinates modulo period about origin, with the
        broadcast shape, including () for scalar inputs.
    log_jacobian : ndarray
        Zeros with the broadcast shape, excluding branch probabilities.

    Raises
    ------
    ValueError
        If labels, period, branch count or origin are invalid.
    """
    span = _span(period, n_modes)
    if not np.isfinite(origin):
        raise ValueError("origin must be finite")
    labels = _validate_modes(modes, n_modes)
    result = np.asarray(
        (np.asarray(values) - origin + labels * span) % period + origin,
        dtype=float,
    )
    return result, np.zeros_like(result)


def fold_phase_polarization(
    phase: ArrayLike, polarization: ArrayLike
) -> tuple[
    tuple[NDArray[np.float64], ...], NDArray[np.float64], NDArray[np.int64]
]:
    """Fold the coupled dominant-(2, 2) phase/polarization action.

    Parameters
    ----------
    phase, polarization : array_like
        Finite broadcastable radian angles, with periods 2*pi and pi.

    Returns
    -------
    coordinates : tuple of ndarray
        (phase, polarization) in [0, pi/2) x [0, pi). Each array has the
        broadcast shape, including () for scalar inputs.
    log_jacobian : ndarray
        Zeros with the broadcast shape.
    labels : ndarray of int64
        Branch b in 0..3, removing (b*pi/2, (b%2)*pi/2).

    Notes
    -----
    This is a waveform symmetry only under the dominant-mode assumptions.
    It is not an independent fold of each periodic coordinate.
    """
    phase, polarization = np.broadcast_arrays(phase, polarization)
    phi, jac, modes = fold_periodic(phase, 2 * np.pi, 4)
    if not np.isfinite(polarization).all():
        raise ValueError("Polarization must be finite")
    return (
        (phi, np.asarray((polarization - (modes % 2) * np.pi / 2) % np.pi)),
        jac,
        modes,
    )


def unfold_phase_polarization(
    phase: ArrayLike, polarization: ArrayLike, modes: ArrayLike
) -> tuple[tuple[NDArray[np.float64], ...], NDArray[np.float64]]:
    """Invert the coupled dominant-22 fold at fixed labels.

    Parameters
    ----------
    phase, polarization : array_like
        Folded radian coordinates. Nonfinite proposals propagate.
    modes : array_like of int
        Explicit branches in 0..3. All inputs must be broadcastable.

    Returns
    -------
    coordinates : tuple of ndarray
        (phase, polarization) in [0, 2*pi) x [0, pi), with the broadcast shape,
        including () for scalars.
    log_jacobian : ndarray
        Zeros with the broadcast shape; no branch probability is included.
    """
    phase, polarization, modes = np.broadcast_arrays(
        phase, polarization, _validate_modes(modes, 4)
    )
    phi, jac = unfold_periodic(phase, 2 * np.pi, 4, modes)
    return (
        phi,
        np.asarray((polarization + (modes % 2) * np.pi / 2) % np.pi),
    ), jac


def _check_cosine(cos_inclination: ArrayLike) -> NDArray[np.float64]:
    c = np.asarray(cos_inclination, dtype=float)
    if np.any(~np.isfinite(c) | (c == 0)):
        raise ValueError(
            "Delta-phase symmetry requires finite, non-edge-on inclination."
        )
    return c


def fold_delta_phase_polarization(
    delta: ArrayLike, polarization: ArrayLike, cos_inclination: ArrayLike
) -> tuple[
    tuple[NDArray[np.float64], ...], NDArray[np.float64], NDArray[np.int64]
]:
    """Fold dominant-22 branches in delta-phase coordinates.

    Parameters
    ----------
    delta, polarization : array_like
        Finite radian angles with periods 2*pi and pi, respectively.
    cos_inclination : array_like
        Finite nonzero inclination cosine, broadcastable with both angles.
        Exactly edge-on inclination is excluded; the physical domain is
        [-1, 1]. Either sign of the delta-phase convention is supported.

    Returns
    -------
    coordinates : tuple of ndarray
        (delta, polarization) in [0, pi) x [0, pi/2). Each array has the
        broadcast shape, including () for scalar inputs.
    log_jacobian : ndarray
        Zeros with the broadcast shape.
    labels : ndarray of int64
        2*delta_branch + polarization_branch, in 0..3.

    Raises
    ------
    ValueError
        If inputs are nonfinite or inclination is exactly edge-on.
    """
    delta, polarization, _ = np.broadcast_arrays(
        delta, polarization, _check_cosine(cos_inclination)
    )
    d, jac, db = fold_periodic(delta, 2 * np.pi, 2)
    p, _, pb = fold_periodic(polarization, np.pi, 2)
    return (d, p), jac, np.asarray(2 * db + pb, dtype=np.int64)


def unfold_delta_phase_polarization(
    delta: ArrayLike,
    polarization: ArrayLike,
    cos_inclination: ArrayLike,
    modes: ArrayLike,
) -> tuple[tuple[NDArray[np.float64], ...], NDArray[np.float64]]:
    """Invert the delta-phase fold with explicit branches.

    Parameters
    ----------
    delta, polarization : array_like
        Folded radian coordinates; nonfinite proposals propagate.
    cos_inclination : array_like
        Finite nonzero inclination cosine, physically in [-1, 1].
    modes : array_like of int
        Branches in 0..3. All inputs must be broadcastable.

    Returns
    -------
    coordinates : tuple of ndarray
        (delta, polarization) in [0, 2*pi) x [0, pi). Each array has the
        broadcast shape, including () for scalar inputs.
    log_jacobian : ndarray
        Zeros with the broadcast shape, excluding branch probabilities.

    Raises
    ------
    ValueError
        If labels are invalid or inclination is nonfinite or exactly edge-on.
    """
    delta, polarization, _, modes = np.broadcast_arrays(
        delta,
        polarization,
        _check_cosine(cos_inclination),
        _validate_modes(modes, 4),
    )
    d, jac = unfold_periodic(delta, 2 * np.pi, 2, modes // 2)
    p, _ = unfold_periodic(polarization, np.pi, 2, modes % 2)
    return (d, p), jac
