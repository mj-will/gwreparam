from __future__ import annotations

import numpy as np
import pytest

from gwreparam.symmetry import (
    Dominant22PhasePolarizationSymmetry,
    PeriodicParameterSymmetry,
)


@pytest.fixture
def generic_particles() -> np.ndarray:
    eps = 1.0e-12
    return np.asarray(
        [
            [0.0, 0.0],
            [0.5 * np.pi - eps, 0.5 * np.pi - eps],
            [0.5 * np.pi + eps, 0.5 * np.pi + eps],
            [np.pi + 0.2, 0.8],
            [2 * np.pi - eps, np.pi - eps],
        ]
    )


@pytest.fixture
def periodic_symmetry() -> PeriodicParameterSymmetry:
    return PeriodicParameterSymmetry(0, 2 * np.pi, 4)


@pytest.fixture
def dominant_symmetry() -> Dominant22PhasePolarizationSymmetry:
    return Dominant22PhasePolarizationSymmetry(0, 1)
