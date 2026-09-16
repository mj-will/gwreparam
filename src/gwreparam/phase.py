"""Orbital/delta-phase coordinate conversion (radians)."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


class DeltaPhase:
    """Replace orbital phase with a convention-dependent delta phase.

    ``delta = (phase + sign * sign(cos(inclination)) * polarization) % (2*pi)``.
    ``sign`` defaults to +1; use -1 for the opposite convention. Inclination
    is in radians unless ``inclination_mode="cos"``. Phase is an angle on
    [0, 2*pi); inversion recovers it modulo 2*pi. Other coordinates are
    unchanged. This is a unit-Jacobian shear away from the inclination branch
    boundary, not a symmetry fold, and requires no auxiliary mode labels.
    """

    def __init__(
        self,
        phase_index: int,
        polarization_index: int,
        inclination_index: int,
        *,
        sign: int = 1,
        inclination_mode: str = "angle",
    ) -> None:
        if isinstance(sign, (bool, np.bool_)) or sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1.")
        if inclination_mode not in ("angle", "cos"):
            raise ValueError("inclination_mode must be 'angle' or 'cos'.")
        indices = (phase_index, polarization_index, inclination_index)
        if (
            any(
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                or index < 0
                for index in indices
            )
            or len(set(indices)) != 3
        ):
            raise ValueError(
                "Parameter indices must be distinct nonnegative integers."
            )
        self.phase_index = int(phase_index)
        self.polarization_index = int(polarization_index)
        self.inclination_index = int(inclination_index)
        self.sign = int(sign)
        self.inclination_mode = inclination_mode

    def _apply(
        self, particles: NDArray[np.float64], direction: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        values = np.array(particles, dtype=float, copy=True)
        inclination = values[:, self.inclination_index]
        if self.inclination_mode == "angle":
            cosine = np.cos(inclination)
            # Give exact edge-on angles the same branch as cos(inclination)=0.
            cosine = np.where(inclination == np.pi / 2, 0.0, cosine)
        else:
            cosine = inclination
        values[:, self.phase_index], _ = delta_phase(
            values[:, self.phase_index],
            values[:, self.polarization_index],
            cosine,
            sign=self.sign,
            inverse=direction == -1,
        )
        return values, np.zeros(len(values), dtype=values.dtype)

    def forward(
        self, particles: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], None]:
        """Return delta-phase rows (n, d), zero log Jacobians (n,), and None."""
        values, log_det = self._apply(particles, 1)
        return values, log_det, None

    def inverse(
        self, values: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return orbital-phase rows (n, d) and zero log Jacobians (n,)."""
        return self._apply(values, -1)


def delta_phase(
    phase: ArrayLike,
    polarization: ArrayLike,
    cos_inclination: ArrayLike,
    *,
    sign: int = 1,
    inverse: bool = False,
    wrap: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert broadcastable angles using an explicitly supplied cosine.

    Angles are radians. Return values and log|d output/d input| (zero).
    ``sign`` selects delta = phase + sign*sign(cos_inclination)*psi.
    ``inverse`` reverses the shear; ``wrap`` selects [0, 2*pi) output.
    Zero cosine leaves phase unchanged; nonfinite inputs propagate.
    """
    if isinstance(sign, (bool, np.bool_)) or sign not in (-1, 1):
        raise ValueError("sign must be +1 or -1.")
    result = np.asarray(phase, dtype=float) + (
        -1 if inverse else 1
    ) * sign * np.sign(cos_inclination) * np.asarray(polarization)
    if wrap:
        result = np.mod(result, 2 * np.pi)
    result = np.asarray(result, dtype=float)
    return result, np.zeros_like(result)
