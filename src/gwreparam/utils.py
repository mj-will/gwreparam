"""Shared numerical utilities for gravitational-wave extensions."""

import numpy as np
from numpy.typing import NDArray


def calculate_angular_shift(
    values: np.ndarray,
    lower: float,
    upper: float,
) -> float:
    """Return the shift that centers the circular mean in an interval."""
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
        if np.any(~np.isfinite(values) | (values <= 0) | (values >= 1)):
            raise ValueError("Polar-area radius must lie strictly in (0, 1).")
        return np.log(values) - np.log1p(-values), -np.log(values) - np.log1p(
            -values
        )

    def inverse(
        self, values: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # Match the original expit saturation, without requiring SciPy.
        with np.errstate(over="ignore"):
            radial = 1.0 / (1.0 + np.exp(-values))
        jac = -np.logaddexp(0, -values) - np.logaddexp(0, values)
        return radial, np.where(np.isfinite(values), jac, -np.inf)
