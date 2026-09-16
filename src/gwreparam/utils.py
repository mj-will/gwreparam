"""Shared numerical utilities for gravitational-wave extensions."""

import numpy as np
from numpy.typing import NDArray


def calculate_angular_shift(
    values: np.ndarray,
    lower: float,
    upper: float,
) -> float:
    """Return a shift that centers the circular mean in an interval.

    Parameters
    ----------
    values : ndarray
        Angular samples, pooled over all axes.
    lower, upper : float
        Endpoints of a nonzero-width periodic interval, in the samples' units.

    Returns
    -------
    shift : float
        Additive shift in the input units. Apply modulo the interval width.
        Empty inputs or a zero resultant retain NumPy's mean/angle behavior.
    """
    angles = 2 * np.pi * (values - lower) / (upper - lower)
    mean_angle = np.angle(np.mean(np.exp(1j * angles)))
    return (lower + upper) / 2 - mean_angle * (upper - lower) / (2 * np.pi)


__all__ = ["calculate_angular_shift", "UnclippedLogitMapping"]


class UnclippedLogitMapping:
    """Exact radial logit: the standard clipped map would truncate W's prior.

    Nonfinite latent values and floating-point saturation are rejected by the
    chart inverse, with finite physical placeholders. No radial clipping.
    """

    def forward(
        self, values: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Map an open-unit-interval radius to the real line.

        Parameters
        ----------
        values : ndarray
            Finite radii strictly between zero and one, of arbitrary shape.

        Returns
        -------
        values : ndarray
            Logit coordinates, with the input shape.
        log_jacobian : ndarray
            Forward log-absolute derivative, with the input shape.

        Raises
        ------
        ValueError
            If any radius is nonfinite or outside the open interval.
        """
        if np.any(~np.isfinite(values) | (values <= 0) | (values >= 1)):
            raise ValueError("Polar-area radius must lie strictly in (0, 1).")
        return np.log(values) - np.log1p(-values), -np.log(values) - np.log1p(
            -values
        )

    def inverse(
        self, values: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Map real coordinates back to radius without clipping saturation.

        Parameters
        ----------
        values : ndarray
            Real coordinates of arbitrary shape; nonfinite inputs are allowed.

        Returns
        -------
        radius : ndarray
            Logistic values, with the input shape. Floating-point saturation
            can reach zero or one; a downstream chart must reject boundaries.
        log_jacobian : ndarray
            Inverse log-absolute derivative, same shape; -inf for nonfinite
            inputs. NaN coordinates propagate to the returned radius.
        """
        # Match the original expit saturation, without requiring SciPy.
        with np.errstate(over="ignore"):
            radial = 1.0 / (1.0 + np.exp(-values))
        jac = -np.logaddexp(0, -values) - np.logaddexp(0, values)
        return radial, np.where(np.isfinite(values), jac, -np.inf)
