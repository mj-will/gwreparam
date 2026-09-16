"""Shared numerical utilities for gravitational-wave extensions."""

import numpy as np


def calculate_angular_shift(
    values: np.ndarray,
    lower: float,
    upper: float,
) -> float:
    """Return the shift that centers the circular mean in an interval."""
    angles = 2 * np.pi * (values - lower) / (upper - lower)
    mean_angle = np.angle(np.mean(np.exp(1j * angles)))
    return (lower + upper) / 2 - mean_angle * (upper - lower) / (2 * np.pi)


__all__ = ["calculate_angular_shift"]
