"""LISA approximate sky folds, independent of parameter names and priors."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..symmetry import _validate_modes, fold_periodic

KnownSymmetryGroup = Literal["reflected-antipodal", "8-mode"]
KnownBetaMode = Literal["angle", "sin"]
KnownIotaMode = Literal["angle", "cos"]


def sky_mode_count(symmetry_group: str) -> int:
    """Return the number of branches in a supported LISA sky group.

    Parameters
    ----------
    symmetry_group : {"reflected-antipodal", "8-mode"}
        Approximate LISA sky symmetry group.

    Returns
    -------
    n_modes : int
        Four or eight branches, respectively.

    Raises
    ------
    ValueError
        If the group is unknown.
    """
    if symmetry_group not in ("reflected-antipodal", "8-mode"):
        raise ValueError(f"Unknown symmetry_group: {symmetry_group}")
    return 8 if symmetry_group == "8-mode" else 4


def _check_conventions(latitude_mode: str, inclination_mode: str) -> None:
    if latitude_mode not in ("angle", "sin") or inclination_mode not in (
        "angle",
        "cos",
    ):
        raise ValueError(
            "Require latitude angle/sin and inclination angle/cos."
        )


def fold_sky(
    longitude: ArrayLike,
    latitude: ArrayLike,
    polarization: ArrayLike,
    inclination: ArrayLike,
    *,
    symmetry_group: str = "reflected-antipodal",
    latitude_mode: str = "sin",
    inclination_mode: str = "cos",
) -> tuple[
    tuple[NDArray[np.float64], ...], NDArray[np.float64], NDArray[np.int64]
]:
    """Fold LISA longitude and latitude branches with explicit conventions.

    Parameters
    ----------
    longitude, latitude, polarization, inclination : array_like
        Finite broadcastable coordinates. Longitude and polarization are
        radians, with periods 2*pi and pi. Physical latitude is in
        [-pi/2, pi/2] or [-1, 1]; inclination is in [0, pi] or [-1, 1],
        according to the explicit conventions below.
    symmetry_group : {"reflected-antipodal", "8-mode"}, optional
        Two or four longitude branches, respectively, plus two hemispheres.
    latitude_mode : {"angle", "sin"}, optional
        Radian latitude or sine latitude (default).
    inclination_mode : {"angle", "cos"}, optional
        Radian inclination or cosine inclination (default).

    Returns
    -------
    coordinates : tuple of ndarray
        (longitude, latitude, polarization, inclination), folded north with
        longitude in [0, 2*pi/n_longitude) and polarization in [0, pi).
        Arrays have the broadcast shape, including () for scalar inputs.
    log_jacobian : ndarray
        Zeros with the broadcast shape.
    labels : ndarray of int64
        longitude_bin + n_longitude*(latitude >= 0), with the broadcast shape.

    Notes
    -----
    Southern reflection reverses latitude, inclination cosine and polarization.
    Eight-mode folding also subtracts longitude_bin*pi/2 from polarization.
    These approximate LISA degeneracies need not preserve a full waveform or
    target density. Bounds are not inferred or clipped.
    """
    _check_conventions(latitude_mode, inclination_mode)
    n = sky_mode_count(symmetry_group) // 2
    lon, lat, psi, inc = np.broadcast_arrays(
        *[
            np.asarray(v, dtype=float)
            for v in (longitude, latitude, polarization, inclination)
        ]
    )
    if not all(np.isfinite(v).all() for v in (lon, lat, psi, inc)):
        raise ValueError("Sky folding requires finite coordinates")
    folded_lon, jac, lb = fold_periodic(lon, 2 * np.pi, n)
    north = lat >= 0
    psi = (psi - (lb * np.pi / 2 if n == 4 else 0)) % np.pi
    reflected_inc = np.pi - inc if inclination_mode == "angle" else -inc
    return (
        (
            folded_lon,
            np.asarray(np.abs(lat)),
            np.where(north, psi, (-psi) % np.pi),
            np.where(north, inc, reflected_inc),
        ),
        jac,
        np.asarray(lb + north.astype(np.int64) * n),
    )


def unfold_sky(
    longitude: ArrayLike,
    latitude: ArrayLike,
    polarization: ArrayLike,
    inclination: ArrayLike,
    modes: ArrayLike,
    *,
    symmetry_group: str = "reflected-antipodal",
    latitude_mode: str = "sin",
    inclination_mode: str = "cos",
) -> tuple[tuple[NDArray[np.float64], ...], NDArray[np.float64]]:
    """Invert a LISA sky fold with explicit integer branches.

    Parameters
    ----------
    longitude, latitude, polarization, inclination : array_like
        Folded coordinates in the units and domains documented by fold_sky.
        Nonfinite proposals propagate.
    modes : array_like of int
        Branches in [0, sky_mode_count(symmetry_group)). All inputs broadcast.
    symmetry_group : {"reflected-antipodal", "8-mode"}, optional
        Group matching the forward fold.
    latitude_mode : {"angle", "sin"}, optional
        Radian or sine latitude, matching the forward fold.
    inclination_mode : {"angle", "cos"}, optional
        Radian or cosine inclination, matching the forward fold.

    Returns
    -------
    coordinates : tuple of ndarray
        (longitude, latitude, polarization, inclination), with canonical
        longitude in [0, 2*pi) and polarization in [0, pi). Each array has
        the broadcast shape, including () for scalar inputs.
    log_jacobian : ndarray
        Zeros with the broadcast shape, excluding branch probabilities.

    Raises
    ------
    ValueError
        If labels or mathematical conventions are invalid.
    """
    _check_conventions(latitude_mode, inclination_mode)
    n = sky_mode_count(symmetry_group) // 2
    lon, lat, psi, inc, b = np.broadcast_arrays(
        longitude,
        latitude,
        polarization,
        inclination,
        _validate_modes(modes, 2 * n),
    )
    lb, north = b % n, b // n
    p = np.where(north, psi, (-psi) % np.pi)
    p = (p + (lb * np.pi / 2 if n == 4 else 0)) % np.pi
    i = np.where(
        north, inc, np.pi - inc if inclination_mode == "angle" else -inc
    )
    return (
        np.asarray((lon + lb * 2 * np.pi / n) % (2 * np.pi)),
        np.where(north, lat, -lat),
        np.asarray(p),
        i,
    ), np.zeros(lon.shape)
