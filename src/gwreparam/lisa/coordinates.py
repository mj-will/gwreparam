"""Helicity coordinates, separate from sky folding and phase conventions."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..amplitude import _check_scale
from ..symmetry import _validate_modes


def _check_options(
    orientation: str, representation: str, polarization_frame: str
) -> None:
    if orientation not in ("W", "Z"):
        raise ValueError("orientation must be W or Z")
    if representation not in ("cartesian", "polar-area"):
        raise ValueError("representation must be cartesian or polar-area")
    if polarization_frame not in ("sky-rotated", "unrotated"):
        raise ValueError("polarization_frame must be sky-rotated or unrotated")


def _log_jacobian(
    u: NDArray[np.float64],
    distance: NDArray[np.float64] | None,
    orientation: str,
    representation: str,
) -> NDArray[np.float64]:
    result = (
        np.log(32.0) + 3 * np.log1p(-u) - 5 * np.log1p(u)
        if orientation == "W"
        else np.full_like(u, np.log(4.0))
    )
    if representation == "polar-area":
        result = result + np.log(2.0)
    if distance is not None:
        result = result - np.log(distance)
    return result


def _angles(
    phase: NDArray[np.float64],
    w: NDArray[np.complex128],
    longitude: NDArray[np.float64],
    hand: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    phi_dom = (phase - 2 * hand * longitude) % (2 * np.pi)
    psi = (hand * np.angle(w * np.exp(4j * hand * longitude)) / 4) % (
        np.pi / 2
    )
    return psi, phi_dom / 2 + hand * psi


def helicity_coordinates(
    phase: ArrayLike,
    cos_inclination: ArrayLike,
    polarization: ArrayLike,
    longitude: ArrayLike = 0.0,
    *,
    distance: ArrayLike | None = None,
    distance_scale: float = 3.085677581491367e25,
    orientation: str = "W",
    representation: str = "cartesian",
    polarization_frame: str = "sky-rotated",
) -> tuple[
    tuple[NDArray[np.float64], ...],
    NDArray[np.float64],
    dict[str, NDArray[np.int64]],
]:
    """Convert orbital phase and orientation to helicity coordinates.

    Parameters
    ----------
    phase, cos_inclination, polarization : array_like
        Finite orbital phase and polarization in radians (periods 2*pi, pi)
        and inclination cosine c with 0 < abs(c) < 1.
    longitude : array_like, optional
        Finite longitude in radians, used by the sky-rotated frame. This
        coordinate is unchanged and must be supplied again at inversion.
    distance : array_like or None, optional
        Explicit positive finite distance; omitted by default. All supplied
        coordinates must broadcast, including distance.
    distance_scale : float, optional
        Positive finite reference distance in the same units as distance.
        Default is one Gpc in metres (3.085677581491367e25).
    orientation : {"W", "Z"}, optional
        Radius ((1-abs(c))/(1+abs(c)))**2 or sqrt(1-abs(c)), respectively.
    representation : {"cartesian", "polar-area"}, optional
        Cartesian components or squared radius and angle in [0, 2*pi).
    polarization_frame : {"sky-rotated", "unrotated"}, optional
        Use polarization-longitude (default) or polarization in the coordinates.

    Returns
    -------
    coordinates : tuple of ndarray
        (Phi, a, b), plus log(scale*(1+abs(c))**2/(4*distance)) when distance
        is supplied. Phi is in [0, 2*pi). a,b are Cartesian components or
        area and angle. Arrays have the broadcast shape, including () for
        scalars. Longitude is not returned or folded.
    log_jacobian : ndarray
        Forward log-absolute coordinate determinant, with the broadcast shape.
    labels : dict of str to ndarray of int64
        Broadcast-shaped hand (-1/+1), psi (0/1), and phase (0/1) branches.

    Raises
    ------
    ValueError
        If coordinates, coordinate options or scale are invalid. Exact edge-on
        and face-on/off points are excluded singularities.

    Notes
    -----
    No sky folding or delta-phase conversion occurs here. Compose them
    explicitly downstream if desired. Changing labels is a proposal mapping,
    not a claim of waveform or target invariance. No randomness is used.
    """
    _check_options(orientation, representation, polarization_frame)
    _check_scale(distance_scale)
    inputs = [phase, cos_inclination, polarization, longitude]
    if distance is not None:
        inputs.append(distance)
    arrays = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in inputs])
    phi, c, psi, lam = arrays[:4]
    d = arrays[4] if distance is not None else None
    u = np.abs(c)
    if not all(np.isfinite(v).all() for v in arrays) or np.any(
        (u == 0) | (u >= 1)
    ):
        raise ValueError(
            "Helicity coordinates require finite, non-edge-on, non-face-on coordinates."
        )
    s = np.sign(c).astype(np.int64)
    if d is not None and np.any(d <= 0):
        raise ValueError("Distance must be positive.")
    if polarization_frame == "unrotated":
        lam = np.zeros_like(lam)
    k = ((1 - u) / (1 + u)) ** 2 if orientation == "W" else np.sqrt(1 - u)
    phase_out = (2 * (phi - s * (psi - lam))) % (2 * np.pi)
    w = k * np.exp(4j * s * (psi - lam))
    theta = (4 * s * (psi - lam)) % (2 * np.pi)
    psi0, base = _angles(
        phase_out,
        w if representation == "cartesian" else np.exp(1j * theta),
        lam,
        s,
    )
    pb = np.rint((psi - psi0) / (np.pi / 2)).astype(np.int64) % 2
    fb = (
        np.rint((phi - base - s * pb * np.pi / 2) / np.pi).astype(np.int64) % 2
    )
    a, b = (
        (w.real, w.imag)
        if representation == "cartesian"
        else (k**2 if orientation == "W" else 1 - u, theta)
    )
    result: tuple[NDArray[np.float64], ...] = (phase_out, a, b)
    if d is not None:
        result += (
            np.log(distance_scale) + 2 * np.log1p(u) - np.log(4.0) - np.log(d),
        )
    return (
        tuple(np.asarray(v) for v in result),
        np.asarray(_log_jacobian(u, d, orientation, representation)),
        {
            "hand": np.asarray(s),
            "psi": np.asarray(pb),
            "phase": np.asarray(fb),
        },
    )


def inverse_helicity_coordinates(
    phase: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
    longitude: ArrayLike = 0.0,
    *,
    hand: ArrayLike,
    psi_branch: ArrayLike,
    phase_branch: ArrayLike,
    log_amplitude: ArrayLike | None = None,
    distance_scale: float = 3.085677581491367e25,
    orientation: str = "W",
    representation: str = "cartesian",
    polarization_frame: str = "sky-rotated",
) -> tuple[tuple[NDArray[np.float64], ...], NDArray[np.float64]]:
    """Invert helicity coordinates using all three branch labels.

    Parameters
    ----------
    phase, a, b : array_like
        Phi and helicity coordinates. Cartesian a,b must be in the open unit
        disk, excluding its center; polar-area a must lie in (0, 1) and b
        is a radian angle. Phase has period 2*pi.
    longitude : array_like, optional
        Longitude supplied to the forward transform, in radians.
    hand : array_like of int
        Explicit handedness labels, -1 or +1.
    psi_branch, phase_branch : array_like of int
        Explicit polarization and phase labels, each 0 or 1.
    log_amplitude : array_like or None, optional
        Optional distance-related coordinate from the forward transform. All
        coordinates and labels must be broadcastable.
    distance_scale : float, optional
        Positive finite scale matching the forward map, default one Gpc in
        metres. Reconstructed distance has the same units.
    orientation : {"W", "Z"}, optional
        Radial convention matching the forward transform.
    representation : {"cartesian", "polar-area"}, optional
        Representation matching the forward transform.
    polarization_frame : {"sky-rotated", "unrotated"}, optional
        Frame matching the forward transform.

    Returns
    -------
    coordinates : tuple of ndarray
        (orbital phase, cosine inclination, polarization), plus distance if
        log_amplitude is supplied. Angles are canonical modulo 2*pi and pi.
        Each array has the broadcast shape, including () for scalars.
    log_jacobian : ndarray
        Inverse log-absolute determinant at fixed branches, same shape.
        Branch probabilities are excluded.

    Raises
    ------
    ValueError
        If labels, coordinate options or reference scale are invalid.

    Notes
    -----
    Nonfinite proposals, invalid radii, recovered singular inclinations, and
    nonpositive or overflowing distances produce finite placeholders with
    cosine inclination 2 and log Jacobian -inf. They must be rejected by the
    consumer. The transform is never clipped or regularized.
    """
    _check_options(orientation, representation, polarization_frame)
    _check_scale(distance_scale)
    h = np.asarray(hand)
    if not np.issubdtype(h.dtype, np.integer) or not np.isin(h, [-1, 1]).all():
        raise ValueError("Invalid hand branch labels")
    pb, fb = _validate_modes(psi_branch, 2), _validate_modes(phase_branch, 2)
    inputs = [phase, a, b, longitude]
    if log_amplitude is not None:
        inputs.append(log_amplitude)
    arrays = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in inputs], h, pb, fb
    )
    phase, a, b, lam = arrays[:4]
    s, pb, fb = arrays[-3:]
    radius = np.hypot(a, b) if representation == "cartesian" else a
    valid = (
        np.logical_and.reduce([np.isfinite(v) for v in arrays])
        & (radius > 0)
        & (radius < 1)
    )
    safe = [np.where(valid, v, 0.0) for v in arrays[: len(inputs)]]
    k = np.where(valid, radius, 0.25)
    if orientation == "W":
        root = np.sqrt(k) if representation == "cartesian" else k**0.25
        u = (1 - root) / (1 + root)
    else:
        u = 1 - (k**2 if representation == "cartesian" else k)
    valid &= (u > 0) & (u < 1)
    u = np.where(valid, u, 0.5)
    w = (
        safe[1] + 1j * safe[2]
        if representation == "cartesian"
        else np.exp(1j * (safe[2] % (2 * np.pi)))
    )
    lam = (
        safe[3] if polarization_frame == "sky-rotated" else np.zeros_like(lam)
    )
    psi, base = _angles(safe[0], w, lam, s)
    phi = (base + s * pb * np.pi / 2 + fb * np.pi) % (2 * np.pi)
    psi = (psi + pb * np.pi / 2) % np.pi
    d = None
    if log_amplitude is not None:
        with np.errstate(over="ignore", under="ignore"):
            d = np.exp(
                np.log(distance_scale)
                + 2 * np.log1p(u)
                - np.log(4.0)
                - safe[4]
            )
        valid &= np.isfinite(d) & (d > 0)
        d = np.where(valid, d, distance_scale)
    result: tuple[NDArray[np.float64], ...] = (
        phi,
        np.where(valid, s * u, 2.0),
        psi,
    )
    if d is not None:
        result += (d,)
    return tuple(np.asarray(v) for v in result), np.where(
        valid, -_log_jacobian(u, d, orientation, representation), -np.inf
    )


def source_helicity(
    phase: ArrayLike,
    cos_inclination: ArrayLike,
    polarization: ArrayLike,
    *,
    distance: ArrayLike | None = None,
    distance_scale: float = 3.085677581491367e25,
) -> tuple[
    tuple[NDArray[np.float64], ...],
    NDArray[np.float64],
    dict[str, NDArray[np.int64]],
]:
    """Return (Phi_src, H_src_x, H_src_y[, log_A_dom]), Jacobian and labels.

    This is the primary API for Cartesian source helicity coordinates.
    Angles are in radians, with 0 < abs(cos_inclination) < 1. Positive
    distance, if supplied, uses distance_scale units. NumPy inputs broadcast.
    H_src denotes complex source helicity coordinates; hand is the discrete
    handedness label. This uses Cartesian W coordinates without sky folding;
    branch and invalid-input conventions match helicity_coordinates.
    """
    return helicity_coordinates(
        phase,
        cos_inclination,
        polarization,
        distance=distance,
        distance_scale=distance_scale,
        polarization_frame="unrotated",
    )


def inverse_source_helicity(
    phase: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    *,
    hand: ArrayLike,
    psi_branch: ArrayLike,
    phase_branch: ArrayLike,
    log_amplitude: ArrayLike | None = None,
    distance_scale: float = 3.085677581491367e25,
) -> tuple[tuple[NDArray[np.float64], ...], NDArray[np.float64]]:
    """Invert source_helicity with retained hand, psi and phase labels.

    Returns (orbital phase, cos inclination, polarization[, distance]) and
    inverse log determinant. NumPy inputs broadcast; phase is periodic 2*pi.
    Invalid source radii or amplitudes return finite out-of-prior placeholders
    and -inf Jacobians, as in inverse_helicity_coordinates.
    """
    return inverse_helicity_coordinates(
        phase,
        x,
        y,
        hand=hand,
        psi_branch=psi_branch,
        phase_branch=phase_branch,
        log_amplitude=log_amplitude,
        distance_scale=distance_scale,
        polarization_frame="unrotated",
    )
