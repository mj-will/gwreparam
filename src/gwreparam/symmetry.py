"""Detector-independent gravitational-wave phase symmetries."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

KNOWN_PHASE_PARAMETERS = frozenset(
    ["delta_phi", "phi_ref", "phase", "coa_phase"]
)

_DOMINANT_22_PHASE_POLARIZATION_SHIFTS = np.asarray(
    [
        # dphase, dpsi
        [0.0, 0.0],
        [0.5 * np.pi, 0.5 * np.pi],
        [np.pi, 0.0],
        [1.5 * np.pi, 0.5 * np.pi],
    ],
    dtype=float,
)


class Symmetry(Protocol):
    """Finite symmetry represented by folding into a reduced space."""

    @property
    def n_modes(self) -> int: ...

    def determine_mode(self, particles: np.ndarray) -> np.ndarray: ...

    def fold(self, particles: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def unfold(self, reduced: np.ndarray, mode: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class PeriodicParameterSymmetry:
    """Fold one periodic particle coordinate into a finite number of modes."""

    parameter_index: int
    period: float
    n_modes: int
    origin: float = 0.0

    def __post_init__(self) -> None:
        if self.period <= 0.0:
            raise ValueError("period must be positive.")
        if self.n_modes < 1:
            raise ValueError("n_modes must be at least one.")

    @property
    def reduced_period(self) -> float:
        return self.period / self.n_modes

    def determine_mode(self, particles: np.ndarray) -> np.ndarray:
        values = np.asarray(particles)[:, self.parameter_index]
        normalized = np.mod(values - self.origin, self.period)
        return np.floor(normalized / self.reduced_period).astype(int)

    def fold(self, particles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        reduced = np.array(particles, copy=True, dtype=float)
        values = np.asarray(particles)[:, self.parameter_index]
        mode_ids = self.determine_mode(particles)
        reduced[:, self.parameter_index] = (
            np.mod(values - self.origin, self.reduced_period) + self.origin
        )
        return reduced, mode_ids

    def unfold(
        self,
        reduced: np.ndarray,
        mode_ids: np.ndarray,
    ) -> np.ndarray:
        full = np.array(reduced, copy=True, dtype=float)
        values = np.asarray(reduced)[:, self.parameter_index]
        modes = np.asarray(mode_ids, dtype=int)
        if modes.shape != values.shape:
            raise ValueError("Expected one symmetry mode per particle.")
        if np.any((modes < 0) | (modes >= self.n_modes)):
            raise ValueError("Symmetry mode is outside the valid range.")
        full[:, self.parameter_index] = (
            np.mod(
                values - self.origin + modes * self.reduced_period,
                self.period,
            )
            + self.origin
        )
        return full

    def forward(
        self, particles: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        """Return folded rows, log|d folded/d full|, and branch labels."""
        values, labels = self.fold(particles)
        return values, np.zeros(len(values)), labels

    def inverse(
        self, values: NDArray[np.float64], *, auxiliary: NDArray[np.int64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return unfolded rows and log|d full/d folded| at fixed labels."""
        full = self.unfold(values, auxiliary)
        return full, np.zeros(len(full))


@dataclass(frozen=True)
class Dominant22PhasePolarizationSymmetry:
    """Exact phase/polarization symmetry of a dominant-(2, 2) waveform."""

    phase_index: int
    polarization_index: int

    @property
    def n_modes(self) -> int:
        return len(_DOMINANT_22_PHASE_POLARIZATION_SHIFTS)

    @property
    def shifts(self) -> np.ndarray:
        return _DOMINANT_22_PHASE_POLARIZATION_SHIFTS.copy()

    def determine_mode(self, particles: np.ndarray) -> np.ndarray:
        phase = np.asarray(particles)[:, self.phase_index]
        phase_mod = np.mod(phase, 2 * np.pi)
        return np.floor(phase_mod / (0.5 * np.pi)).astype(int)

    def fold(self, particles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        reduced = np.array(particles, copy=True, dtype=float)
        phase = np.asarray(particles)[:, self.phase_index]
        polarization = np.asarray(particles)[:, self.polarization_index]
        phase_mod = np.mod(phase, 2 * np.pi)
        mode_ids = self.determine_mode(particles)
        shifts = _DOMINANT_22_PHASE_POLARIZATION_SHIFTS[mode_ids]
        reduced[:, self.phase_index] = np.mod(
            phase_mod - shifts[:, 0],
            0.5 * np.pi,
        )
        reduced[:, self.polarization_index] = np.mod(
            np.mod(polarization, np.pi) - shifts[:, 1],
            np.pi,
        )
        return reduced, mode_ids

    def unfold(
        self,
        reduced: np.ndarray,
        mode_ids: np.ndarray,
    ) -> np.ndarray:
        full = np.array(reduced, copy=True, dtype=float)
        modes = np.asarray(mode_ids, dtype=int)
        if modes.shape != (full.shape[0],):
            raise ValueError("Expected one symmetry mode per particle.")
        if np.any((modes < 0) | (modes >= self.n_modes)):
            raise ValueError("Symmetry mode is outside the valid range.")
        shifts = _DOMINANT_22_PHASE_POLARIZATION_SHIFTS[modes]
        full[:, self.phase_index] = np.mod(
            full[:, self.phase_index] + shifts[:, 0],
            2 * np.pi,
        )
        full[:, self.polarization_index] = np.mod(
            full[:, self.polarization_index] + shifts[:, 1],
            np.pi,
        )
        return full

    def apply_mode(self, particles: np.ndarray, mode_id: int) -> np.ndarray:
        """Apply one symmetry shift to full-space particles."""
        modes = np.full(len(particles), mode_id, dtype=int)
        return self.unfold(particles, modes)

    def forward(
        self, particles: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        """Return folded rows, log|d folded/d full|, and branch labels."""
        values, labels = self.fold(particles)
        return values, np.zeros(len(values)), labels

    def inverse(
        self, values: NDArray[np.float64], *, auxiliary: NDArray[np.int64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return unfolded rows and log|d full/d folded| at fixed labels."""
        full = self.unfold(values, auxiliary)
        return full, np.zeros(len(full))


@dataclass(frozen=True)
class Dominant22DeltaPhasePolarizationSymmetry(
    Dominant22PhasePolarizationSymmetry
):
    """Four dominant-22 branches in delta-phase coordinates.

    For delta = phi +/- sign(cos(iota))*psi, the generators are independent
    delta += pi and psi += pi/2. The reduced domain is [0, pi) x [0, pi/2).
    Mode IDs are 2*delta_branch + polarization_branch, independent of the
    convention sign. Inclination is required to reject exactly edge-on
    particles, where sign(0)=0 makes delta=phi and this action is not valid.
    This class does not implement LISA sky/reflection symmetries.
    """

    inclination_index: int
    inclination_mode: str = "cos"

    def __post_init__(self) -> None:
        indices = (
            self.phase_index,
            self.polarization_index,
            self.inclination_index,
        )
        if (
            any(
                isinstance(i, (bool, np.bool_))
                or not isinstance(i, (int, np.integer))
                or i < 0
                for i in indices
            )
            or len(set(indices)) != 3
        ):
            raise ValueError(
                "Parameter indices must be distinct nonnegative integers."
            )
        if self.inclination_mode not in {"angle", "cos"}:
            raise ValueError("inclination_mode must be 'angle' or 'cos'.")

    @property
    def shifts(self) -> NDArray[np.float64]:
        return np.asarray(
            [[0, 0], [0, np.pi / 2], [np.pi, 0], [np.pi, np.pi / 2]]
        )

    def _check_inclination(self, particles: NDArray[np.float64]) -> None:
        inclination = np.asarray(particles)[:, self.inclination_index]
        boundary = 0.0 if self.inclination_mode == "cos" else np.pi / 2
        if np.any(~np.isfinite(inclination)) or np.any(
            inclination == boundary
        ):
            raise ValueError(
                "Delta-phase symmetry requires finite, non-edge-on inclination."
            )

    def determine_mode(
        self, particles: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        self._check_inclination(particles)
        particles = np.asarray(particles)
        phase_branch = np.floor(
            np.mod(particles[:, self.phase_index], 2 * np.pi) / np.pi
        )
        psi_branch = np.floor(
            np.mod(particles[:, self.polarization_index], np.pi) / (np.pi / 2)
        )
        return (2 * phase_branch + psi_branch).astype(int)

    def fold(
        self, particles: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        modes = self.determine_mode(particles)
        reduced = np.array(particles, copy=True, dtype=float)
        reduced[:, self.phase_index] %= np.pi
        reduced[:, self.polarization_index] %= np.pi / 2
        return reduced, modes

    def unfold(
        self, reduced: NDArray[np.float64], mode_ids: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        self._check_inclination(reduced)
        modes = np.asarray(mode_ids)
        if modes.shape != (len(reduced),):
            raise ValueError("Expected one symmetry mode per particle.")
        if not np.issubdtype(modes.dtype, np.integer) or np.any(
            (modes < 0) | (modes >= 4)
        ):
            raise ValueError(
                "Symmetry mode must be an integer in the valid range [0, 4)."
            )
        full = np.array(reduced, copy=True, dtype=float)
        shifts = self.shifts[modes]
        full[:, self.phase_index] = (
            full[:, self.phase_index] + shifts[:, 0]
        ) % (2 * np.pi)
        full[:, self.polarization_index] = (
            full[:, self.polarization_index] + shifts[:, 1]
        ) % np.pi
        return full

    def apply_mode(
        self, particles: NDArray[np.float64], mode_id: int
    ) -> NDArray[np.float64]:
        return self.unfold(particles, np.full(len(particles), mode_id))


__all__ = [
    "Dominant22DeltaPhasePolarizationSymmetry",
    "Dominant22PhasePolarizationSymmetry",
    "PeriodicParameterSymmetry",
    "fold_periodic",
    "unfold_periodic",
    "Symmetry",
]


def fold_periodic(
    values: ArrayLike, period: float, n_modes: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Fold broadcastable angles, returning values, zero log Jacobian, labels.

    Radian inputs are reduced to [0, period/n_modes). The branch is taken
    modulo the full period. Nonfinite inputs retain NumPy's NaN/cast behavior.
    """
    if (
        not np.isfinite(period)
        or period <= 0
        or n_modes < 1
        or int(n_modes) != n_modes
    ):
        raise ValueError("Require a positive period and integer mode count.")
    values = np.asarray(values)
    span = period / n_modes
    modes = np.floor(np.mod(values, period) / span).astype(int)
    return (
        np.asarray(np.mod(values, span), dtype=float),
        np.zeros_like(values, dtype=float),
        np.asarray(modes),
    )


def unfold_periodic(
    values: ArrayLike, span: float, modes: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Add explicit branch shifts, returning values and zero log Jacobian.

    This is a lift, without wrapping or validating the branch range. Consumers
    choose branch support. Inputs broadcast, and nonfinite values propagate.
    """
    result = np.asarray(
        np.asarray(values) + np.asarray(modes) * span, dtype=float
    )
    return result, np.zeros_like(result, dtype=float)
