"""Deterministic coordinates extracted from MBHB analysis utilities."""

import numpy as np
from numpy.typing import NDArray


class CircularAmplitude:
    """Map (cosine inclination, distance) to right/left circular amplitudes.

    Distance retains its physical units; amplitudes use distance_scale / D.
    Invalid amplitude proposals map outside the distance prior and have zero
    density. No reflection or clipping changes the proposal distribution.

    Parameters
    ----------
    inclination_index, distance_index : int
        Distinct column indices for cosine inclination and positive distance.
    distance_scale : float
        Positive finite reference distance, in the same units as distance.
    """

    def __init__(
        self,
        inclination_index: int,
        distance_index: int,
        distance_scale: float,
    ) -> None:
        self.inclination_index = inclination_index
        self.distance_index = distance_index
        self.distance_scale = float(distance_scale)
        if not np.isfinite(self.distance_scale) or self.distance_scale <= 0:
            raise ValueError("distance_scale must be finite and positive")

    def forward(
        self, particles: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], None]:
        """Return amplitudes, forward log Jacobian and no auxiliary labels.

        Parameters
        ----------
        particles : ndarray, shape (..., d)
            Physical coordinates; abs(cosine inclination) <= 1, distance > 0.

        Returns
        -------
        values : ndarray, shape (..., d)
            Copied rows with the two selected columns replaced.
        log_det : ndarray, shape (...,)
            log|d amplitudes / d physical|; -infinity at face-on/off.
        auxiliary : None
            No discrete branches are required.
        """
        result = np.array(particles, dtype=float, copy=True)
        c = result[..., self.inclination_index].copy()
        d = result[..., self.distance_index].copy()
        result[..., self.inclination_index] = (
            self.distance_scale * (1 + c) ** 2 / (4 * d)
        )
        result[..., self.distance_index] = (
            self.distance_scale * (1 - c) ** 2 / (4 * d)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            log_det = (
                2 * np.log(self.distance_scale)
                + np.log1p(-(c**2))
                - np.log(4.0)
                - 3 * np.log(d)
            )
        return result, log_det, None

    def inverse(
        self, values: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return physical rows and log|d physical / d amplitudes|.

        Parameters
        ----------
        values : ndarray, shape (..., d)
            Amplitude coordinates. Both amplitudes must be finite and positive.

        Returns
        -------
        physical : ndarray, shape (..., d)
            Restored rows; invalid pairs use cosine=0.5 and distance=-scale.
        log_det : ndarray, shape (...,)
            Inverse log Jacobian; invalid pairs return -infinity.
        """
        result = np.array(values, dtype=float, copy=True)
        ar = result[..., self.inclination_index].copy()
        al = result[..., self.distance_index].copy()
        valid = np.isfinite(ar) & np.isfinite(al) & (ar > 0) & (al > 0)
        # Safe arithmetic placeholders; invalid rows are explicitly rejected.
        root_r = np.sqrt(np.where(valid, ar, 1.0))
        root_l = np.sqrt(np.where(valid, al, 1.0))
        total = root_r + root_l
        c = (root_r - root_l) / total
        d = self.distance_scale / total**2
        # Delta-phase unfolding excludes exactly edge-on inclination. Invalid
        # proposals therefore use a non-edge-on placeholder before rejection.
        result[..., self.inclination_index] = np.where(valid, c, 0.5)
        result[..., self.distance_index] = np.where(
            valid, d, -self.distance_scale
        )
        # 1-c^2 = 4*r*l/(r+l)^2 avoids cancellation near face-on/off.
        log_det = (
            np.log(4.0)
            + 3 * np.log(d)
            - 2 * np.log(self.distance_scale)
            - (
                np.log(4.0)
                + np.log(root_r)
                + np.log(root_l)
                - 2 * np.log(total)
            )
        )
        return result, np.where(valid, log_det, -np.inf)
