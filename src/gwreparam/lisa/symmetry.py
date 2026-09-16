"""LISA sky and extrinsic symmetry descriptions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

from ..phase import DeltaPhase as DeltaPhaseTransform
from ..symmetry import (
    KNOWN_PHASE_PARAMETERS,
    Dominant22DeltaPhasePolarizationSymmetry,
    Dominant22PhasePolarizationSymmetry,
)
from ..utils import calculate_angular_shift

KnownSymmetryGroup = Literal["reflected-antipodal", "8-mode"]
KnownBetaMode = Literal["angle", "sin"]
KnownIotaMode = Literal["angle", "cos"]


SYMMETRY_GROUP_CONFIGS = {
    "8-mode": {"n_longitude_modes": 4, "lambda_step": np.pi / 2},
    "reflected-antipodal": {"n_longitude_modes": 2, "lambda_step": np.pi},
}

KNOWN_LAMBDA_PARAMETERS = frozenset(["eclipticlongitude", "lambda", "lam"])
KNOWN_BETA_PARAMETERS = frozenset(
    [
        "eclipticlatitude",
        "beta",
        "sin_beta",
        "sinbeta",
        "sin_eclipticlatitude",
    ]
)
KNOWN_PSI_PARAMETERS = frozenset(["polarization", "psi"])
KNOWN_IOTA_PARAMETERS = frozenset(
    [
        "iota",
        "inc",
        "inclination",
        "cos_iota",
        "cosiota",
        "cos_inclination",
        "cosinc",
        "cos_inc",
    ]
)


def is_periodic_parameter(parameter_name: str) -> bool:
    """Return whether a known gravitational-wave coordinate is periodic."""
    return parameter_name in (
        KNOWN_LAMBDA_PARAMETERS | KNOWN_PSI_PARAMETERS | KNOWN_PHASE_PARAMETERS
    )


@overload
def _determine_parameter_name(
    parameters: Sequence[str],
    known_parameters: frozenset[str],
    *,
    required: Literal[True] = True,
) -> str: ...


@overload
def _determine_parameter_name(
    parameters: Sequence[str],
    known_parameters: frozenset[str],
    *,
    required: Literal[False],
) -> str | None: ...


@overload
def _determine_parameter_name(
    parameters: Sequence[str],
    known_parameters: frozenset[str],
    *,
    required: bool,
) -> str | None: ...


def _determine_parameter_name(
    parameters: Sequence[str],
    known_parameters: frozenset[str],
    *,
    required: bool = True,
) -> str | None:
    matches = list(set(parameters).intersection(known_parameters))
    if len(matches) > 1:
        raise ValueError(
            f"Multiple parameters match. Expected one of {known_parameters}. "
            f"Received {set(parameters)}."
        )
    if not matches:
        if required:
            raise ValueError(
                f"No parameters match. Expected one of {known_parameters}. "
                f"Received {set(parameters)}."
            )
        return None
    return matches[0]


def _resolve_beta_mode(bounds: tuple[float, float]) -> KnownBetaMode:
    lower, upper = map(float, bounds)
    if np.isclose(lower, -np.pi / 2) and np.isclose(upper, np.pi / 2):
        return "angle"
    if np.isclose(lower, -1.0) and np.isclose(upper, 1.0):
        return "sin"
    raise ValueError(f"Unsupported beta bounds: {bounds}")


def _resolve_iota_mode(bounds: tuple[float, float]) -> KnownIotaMode:
    lower, upper = map(float, bounds)
    if np.isclose(lower, 0.0) and np.isclose(upper, np.pi):
        return "angle"
    if np.isclose(lower, -1.0) and np.isclose(upper, 1.0):
        return "cos"
    raise ValueError(f"Unsupported inclination bounds: {bounds}")


def _validate_periodic_bounds(
    name: str,
    bounds: tuple[float, float],
    expected_upper: float,
) -> None:
    lower, upper = map(float, bounds)
    if not np.isclose(lower, 0.0):
        raise ValueError(f"{name} must start at 0. Received bounds: {bounds}")
    if not np.isclose(upper, expected_upper):
        raise ValueError(
            f"{name} must end at {expected_upper}. Received bounds: {bounds}"
        )


def _lambda_step(symmetry_group: KnownSymmetryGroup) -> float:
    return SYMMETRY_GROUP_CONFIGS[symmetry_group]["lambda_step"]


def _n_longitude_modes(symmetry_group: KnownSymmetryGroup) -> int:
    return int(SYMMETRY_GROUP_CONFIGS[symmetry_group]["n_longitude_modes"])


def _n_sky_modes(symmetry_group: KnownSymmetryGroup) -> int:
    return 2 * _n_longitude_modes(symmetry_group)


def _reflect_iota(iota: np.ndarray, iota_mode: KnownIotaMode) -> np.ndarray:
    if iota_mode == "angle":
        return np.pi - iota
    return -iota


def _reflect_psi(psi: np.ndarray) -> np.ndarray:
    return np.mod(np.pi - psi, np.pi)


def _psi_longitude_shift(
    long_num: np.ndarray,
    symmetry_group: KnownSymmetryGroup,
) -> np.ndarray:
    if symmetry_group == "8-mode":
        return np.asarray(long_num, dtype=float) * 0.5 * np.pi
    return np.zeros_like(long_num, dtype=float)


@dataclass(frozen=True)
class LISASkySymmetry:
    """LISA longitude and latitude/reflection symmetry."""

    longitude_index: int
    latitude_index: int
    polarization_index: int
    inclination_index: int
    symmetry_group: KnownSymmetryGroup
    latitude_mode: KnownBetaMode
    inclination_mode: KnownIotaMode

    def __post_init__(self) -> None:
        if self.symmetry_group not in SYMMETRY_GROUP_CONFIGS:
            raise ValueError(f"Unknown symmetry_group: {self.symmetry_group}")

    @property
    def longitude_step(self) -> float:
        return _lambda_step(self.symmetry_group)

    @property
    def n_longitude_modes(self) -> int:
        return _n_longitude_modes(self.symmetry_group)

    @property
    def n_modes(self) -> int:
        return _n_sky_modes(self.symmetry_group)

    def decode_mode(
        self,
        mode_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return longitude and latitude labels for flat sky mode IDs."""
        mode_ids = np.asarray(mode_ids, dtype=int)
        if np.any((mode_ids < 0) | (mode_ids >= self.n_modes)):
            raise ValueError("LISA sky mode is outside the valid range.")
        longitude_mode = mode_ids % self.n_longitude_modes
        latitude_mode = mode_ids // self.n_longitude_modes
        return longitude_mode, latitude_mode

    def determine_mode(self, particles: np.ndarray) -> np.ndarray:
        particles = np.asarray(particles)
        longitude_mode = np.floor(
            np.mod(particles[:, self.longitude_index], 2 * np.pi)
            / self.longitude_step
        ).astype(int)
        latitude_mode = (particles[:, self.latitude_index] >= 0.0).astype(int)
        return longitude_mode + latitude_mode * self.n_longitude_modes

    def fold(self, particles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        particles = np.asarray(particles)
        reduced = np.array(particles, copy=True, dtype=float)
        mode_ids = self.determine_mode(particles)
        longitude_mode, latitude_mode = self.decode_mode(mode_ids)
        polarization_shift = _psi_longitude_shift(
            longitude_mode,
            self.symmetry_group,
        )

        reduced[:, self.longitude_index] = np.mod(
            particles[:, self.longitude_index]
            - longitude_mode * self.longitude_step,
            self.longitude_step,
        )
        reduced[:, self.latitude_index] = np.where(
            latitude_mode,
            particles[:, self.latitude_index],
            -particles[:, self.latitude_index],
        )
        reduced[:, self.polarization_index] = np.mod(
            particles[:, self.polarization_index] - polarization_shift,
            np.pi,
        )
        reduced[:, self.polarization_index] = np.where(
            latitude_mode,
            reduced[:, self.polarization_index],
            _reflect_psi(reduced[:, self.polarization_index]),
        )
        reduced[:, self.inclination_index] = np.where(
            latitude_mode,
            particles[:, self.inclination_index],
            _reflect_iota(
                particles[:, self.inclination_index],
                self.inclination_mode,
            ),
        )
        return reduced, mode_ids

    def unfold(self, reduced: np.ndarray, mode: np.ndarray) -> np.ndarray:
        full = np.array(reduced, copy=True, dtype=float)
        mode_ids = np.asarray(mode, dtype=int)
        if mode_ids.shape != (full.shape[0],):
            raise ValueError("Expected one LISA sky mode per particle.")
        longitude_mode, latitude_mode = self.decode_mode(mode_ids)
        polarization_shift = _psi_longitude_shift(
            longitude_mode,
            self.symmetry_group,
        )

        full[:, self.longitude_index] = np.mod(
            full[:, self.longitude_index]
            + longitude_mode * self.longitude_step,
            2 * np.pi,
        )
        full[:, self.latitude_index] = np.where(
            latitude_mode,
            full[:, self.latitude_index],
            -full[:, self.latitude_index],
        )
        full[:, self.inclination_index] = np.where(
            latitude_mode,
            full[:, self.inclination_index],
            _reflect_iota(
                full[:, self.inclination_index],
                self.inclination_mode,
            ),
        )
        full[:, self.polarization_index] = np.where(
            latitude_mode,
            full[:, self.polarization_index],
            _reflect_psi(full[:, self.polarization_index]),
        )
        full[:, self.polarization_index] = np.mod(
            full[:, self.polarization_index] + polarization_shift,
            np.pi,
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


class LISAExtrinsicSymmetry:
    """Compose LISA sky degeneracies with optional generic GW symmetries."""

    def __init__(
        self,
        sampling_parameters: Sequence[str],
        bounds: Mapping[str, tuple[float, float]],
        *,
        symmetry_group: KnownSymmetryGroup = "reflected-antipodal",
        phase_polarization: Dominant22PhasePolarizationSymmetry | None = None,
        roll: bool = False,
    ) -> None:
        if symmetry_group not in SYMMETRY_GROUP_CONFIGS:
            raise ValueError(f"Unknown symmetry_group: {symmetry_group}")
        self.sampling_parameters = tuple(sampling_parameters)
        self.bounds = {
            name: (float(parameter_bounds[0]), float(parameter_bounds[1]))
            for name, parameter_bounds in bounds.items()
        }
        self.symmetry_group = symmetry_group
        self.phase_polarization = phase_polarization
        self.roll = bool(roll)

        self.lambda_parameter = _determine_parameter_name(
            self.sampling_parameters, KNOWN_LAMBDA_PARAMETERS
        )
        self.beta_parameter = _determine_parameter_name(
            self.sampling_parameters, KNOWN_BETA_PARAMETERS
        )
        self.psi_parameter = _determine_parameter_name(
            self.sampling_parameters, KNOWN_PSI_PARAMETERS
        )
        self.iota_parameter = _determine_parameter_name(
            self.sampling_parameters, KNOWN_IOTA_PARAMETERS
        )
        self.phase_parameter = _determine_parameter_name(
            self.sampling_parameters,
            KNOWN_PHASE_PARAMETERS,
            required=self.phase_polarization is not None,
        )

        _validate_periodic_bounds(
            self.lambda_parameter,
            self.bounds[self.lambda_parameter],
            2 * np.pi,
        )
        _validate_periodic_bounds(
            self.psi_parameter,
            self.bounds[self.psi_parameter],
            np.pi,
        )
        if self.phase_parameter is not None:
            _validate_periodic_bounds(
                self.phase_parameter,
                self.bounds[self.phase_parameter],
                2 * np.pi,
            )

        self.lambda_index = self.sampling_parameters.index(
            self.lambda_parameter
        )
        self.beta_index = self.sampling_parameters.index(self.beta_parameter)
        self.psi_index = self.sampling_parameters.index(self.psi_parameter)
        self.iota_index = self.sampling_parameters.index(self.iota_parameter)
        self.phase_index = (
            None
            if self.phase_parameter is None
            else self.sampling_parameters.index(self.phase_parameter)
        )
        self.beta_mode = _resolve_beta_mode(self.bounds[self.beta_parameter])
        self.iota_mode = _resolve_iota_mode(self.bounds[self.iota_parameter])
        self.sky_symmetry = LISASkySymmetry(
            longitude_index=self.lambda_index,
            latitude_index=self.beta_index,
            polarization_index=self.psi_index,
            inclination_index=self.iota_index,
            symmetry_group=self.symmetry_group,
            latitude_mode=self.beta_mode,
            inclination_mode=self.iota_mode,
        )

        if self.phase_polarization is not None:
            if self.phase_parameter is None or self.phase_index is None:
                raise ValueError(
                    "dominant-22 phase/polarization symmetry requires a phase "
                    "parameter to be present."
                )
            if (
                self.phase_polarization.phase_index != self.phase_index
                or self.phase_polarization.polarization_index != self.psi_index
            ):
                raise ValueError(
                    "Phase/polarization symmetry indices do not match the "
                    "resolved LISA coordinates."
                )

    @property
    def lambda_step(self) -> float:
        return self.sky_symmetry.longitude_step

    @property
    def n_sky_modes(self) -> int:
        return self.sky_symmetry.n_modes

    @property
    def n_longitude_modes(self) -> int:
        """Number of longitude branches in the configured sky symmetry."""
        return self.sky_symmetry.n_longitude_modes

    def determine_sky_modes(self, particles: np.ndarray) -> np.ndarray:
        """Classify particles into LISA longitude and latitude branches."""
        return self.sky_symmetry.determine_mode(particles)

    def decode_sky_mode(
        self,
        mode: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return longitude and latitude labels for flat sky mode IDs."""
        return self.sky_symmetry.decode_mode(mode)

    @property
    def n_phase_polarization_modes(self) -> int:
        if self.phase_polarization is None:
            return 1
        return self.phase_polarization.n_modes

    @property
    def transformed_phase_span(self) -> float:
        if self.phase_polarization is not None:
            return 0.5 * np.pi
        return 2 * np.pi

    @property
    def transformed_psi_span(self) -> float:
        return np.pi

    def transformed_bounds(self) -> dict[int, tuple[float, float]]:
        transformed = {
            i: self.bounds[name]
            for i, name in enumerate(self.sampling_parameters)
        }
        transformed[self.lambda_index] = (0.0, self.lambda_step)
        transformed[self.beta_index] = (
            (0.0, np.pi / 2) if self.beta_mode == "angle" else (0.0, 1.0)
        )
        transformed[self.psi_index] = (0.0, self.transformed_psi_span)
        if (
            self.phase_index is not None
            and self.phase_polarization is not None
        ):
            transformed[self.phase_index] = (0.0, self.transformed_phase_span)
        return transformed

    def periodic_parameters(self) -> tuple[str, ...]:
        periodic = {
            name
            for name in self.sampling_parameters
            if is_periodic_parameter(name)
        }
        periodic.update([self.lambda_parameter, self.psi_parameter])
        if self.phase_parameter is not None:
            periodic.add(self.phase_parameter)
        return tuple(
            name for name in self.sampling_parameters if name in periodic
        )

    @property
    def n_modes(self) -> int:
        n_modes = self.sky_symmetry.n_modes
        if self.phase_polarization is not None:
            n_modes *= self.n_phase_polarization_modes
        return n_modes

    def encode_mode(
        self,
        sky_mode: np.ndarray,
        phase_polarization_mode: np.ndarray | None = None,
    ) -> np.ndarray:
        """Combine component labels into contiguous composite mode IDs."""
        sky = np.asarray(sky_mode, dtype=int)
        if np.any((sky < 0) | (sky >= self.n_sky_modes)):
            raise ValueError("LISA sky mode is outside the valid range.")
        if self.phase_polarization is None:
            if phase_polarization_mode is not None:
                raise ValueError(
                    "Phase/polarization modes require a configured symmetry."
                )
            phase_polarization = np.zeros_like(sky)
        else:
            if phase_polarization_mode is None:
                raise ValueError("Missing phase/polarization mode labels.")
            phase_polarization = np.asarray(
                phase_polarization_mode,
                dtype=int,
            )
            if phase_polarization.shape != sky.shape:
                raise ValueError(
                    "Composite mode components must have equal shape."
                )
            if np.any(
                (phase_polarization < 0)
                | (phase_polarization >= self.n_phase_polarization_modes)
            ):
                raise ValueError(
                    "Phase/polarization mode is outside the valid range."
                )
        return sky * self.n_phase_polarization_modes + phase_polarization

    def decode_mode(
        self,
        mode: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Split contiguous composite IDs into their component labels."""
        mode_ids = np.asarray(mode, dtype=int)
        if np.any((mode_ids < 0) | (mode_ids >= self.n_modes)):
            raise ValueError("LISA extrinsic mode is outside the valid range.")
        sky_mode, phase_polarization_mode = np.divmod(
            mode_ids,
            self.n_phase_polarization_modes,
        )
        if self.phase_polarization is None:
            return sky_mode, None
        return sky_mode, phase_polarization_mode

    def determine_mode(self, particles: np.ndarray) -> np.ndarray:
        """Return contiguous composite mode labels for full-space particles."""
        _, mode = self.fold(particles)
        return mode

    def fold(
        self,
        particles: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        folded, sky_mode = self.sky_symmetry.fold(particles)

        phase_polarization_mode = None
        if self.phase_polarization is not None:
            if self.phase_index is None:
                raise RuntimeError(
                    "Phase/polarization folding is enabled but no phase "
                    "parameter was resolved."
                )
            folded, phase_polarization_mode = self.phase_polarization.fold(
                folded
            )

        return folded, self.encode_mode(sky_mode, phase_polarization_mode)

    def unfold(
        self,
        folded: np.ndarray,
        mode: np.ndarray,
    ) -> np.ndarray:
        unfolded = np.array(folded, copy=True, dtype=float)
        mode_ids = np.asarray(mode, dtype=int)
        if mode_ids.shape != (unfolded.shape[0],):
            raise ValueError("Expected one LISA extrinsic mode per particle.")
        sky_mode, phase_polarization_mode = self.decode_mode(mode_ids)

        if self.phase_polarization is not None:
            if self.phase_index is None or phase_polarization_mode is None:
                raise ValueError(
                    "Missing phase/polarization mode for dominant-22 "
                    "phase/polarization "
                    "unfolding."
                )
            unfolded = self.phase_polarization.unfold(
                unfolded,
                phase_polarization_mode,
            )

        return self.sky_symmetry.unfold(unfolded, sky_mode)

    def compute_roll_shifts(self, folded: np.ndarray) -> dict[str, float]:
        if not self.roll:
            return {"lambda": 0.0, "psi": 0.0, "phase": 0.0}
        shifts = {
            "lambda": calculate_angular_shift(
                folded[:, self.lambda_index], 0.0, self.lambda_step
            ),
            "psi": calculate_angular_shift(
                folded[:, self.psi_index], 0.0, self.transformed_psi_span
            ),
            "phase": 0.0,
        }
        if (
            self.phase_index is not None
            and self.phase_polarization is not None
        ):
            shifts["phase"] = calculate_angular_shift(
                folded[:, self.phase_index], 0.0, self.transformed_phase_span
            )
        return shifts

    def apply_roll(
        self,
        folded: np.ndarray,
        shifts: Mapping[str, float],
    ) -> np.ndarray:
        if not self.roll:
            return np.array(folded, copy=True)
        rolled = np.array(folded, copy=True)
        rolled[:, self.lambda_index] = np.mod(
            rolled[:, self.lambda_index] + shifts["lambda"],
            self.lambda_step,
        )
        rolled[:, self.psi_index] = np.mod(
            rolled[:, self.psi_index] + shifts["psi"],
            self.transformed_psi_span,
        )
        if (
            self.phase_index is not None
            and self.phase_polarization is not None
        ):
            rolled[:, self.phase_index] = np.mod(
                rolled[:, self.phase_index] + shifts["phase"],
                self.transformed_phase_span,
            )
        return rolled

    def remove_roll(
        self,
        folded: np.ndarray,
        shifts: Mapping[str, float],
    ) -> np.ndarray:
        if not self.roll:
            return np.array(folded, copy=True)
        unrolled = np.array(folded, copy=True)
        unrolled[:, self.lambda_index] = np.mod(
            unrolled[:, self.lambda_index] - shifts["lambda"],
            self.lambda_step,
        )
        unrolled[:, self.psi_index] = np.mod(
            unrolled[:, self.psi_index] - shifts["psi"],
            self.transformed_psi_span,
        )
        if (
            self.phase_index is not None
            and self.phase_polarization is not None
        ):
            unrolled[:, self.phase_index] = np.mod(
                unrolled[:, self.phase_index] - shifts["phase"],
                self.transformed_phase_span,
            )
        return unrolled

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


class LISADeltaPhaseExtrinsicSymmetry(LISAExtrinsicSymmetry):
    """Compose LISA sky actions with optional delta-phase/polarization folding.

    Sky actions are conjugated through delta = phi + phase_sign*sign(cos(iota))*psi.
    Folding applies the sky action first, then delta-phase folding; inversion
    reverses that order. The LISA sky degeneracies retain the approximations
    of the original raw-phase implementation.
    """

    def __init__(
        self,
        sampling_parameters: Sequence[str],
        bounds: Mapping[str, tuple[float, float]],
        *,
        symmetry_group: KnownSymmetryGroup = "reflected-antipodal",
        phase_polarization: Dominant22DeltaPhasePolarizationSymmetry
        | None = None,
        phase_sign: int = 1,
        roll: bool = False,
    ) -> None:
        super().__init__(
            sampling_parameters,
            bounds,
            symmetry_group=symmetry_group,
            phase_polarization=phase_polarization,
            roll=roll,
        )
        if self.phase_parameter != "delta_phi" or self.phase_index is None:
            raise ValueError("LISA delta-phase symmetry requires delta_phi.")
        if phase_polarization is not None and (
            not isinstance(
                phase_polarization, Dominant22DeltaPhasePolarizationSymmetry
            )
            or phase_polarization.inclination_index != self.iota_index
            or phase_polarization.inclination_mode != self.iota_mode
        ):
            raise ValueError(
                "LISA delta-phase folding requires matching delta-phase symmetry and inclination."
            )
        self.phase_transform = DeltaPhaseTransform(
            self.phase_index,
            self.psi_index,
            self.iota_index,
            sign=phase_sign,
            inclination_mode=self.iota_mode,
        )

    @property
    def transformed_phase_span(self) -> float:
        return np.pi if self.phase_polarization is not None else 2 * np.pi

    @property
    def transformed_psi_span(self) -> float:
        return np.pi / 2 if self.phase_polarization is not None else np.pi

    def fold(
        self, particles: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        raw, _ = self.phase_transform.inverse(particles)
        raw_folded, sky_mode = self.sky_symmetry.fold(raw)
        folded, _, _ = self.phase_transform.forward(raw_folded)
        phase_mode = None
        if self.phase_polarization is not None:
            folded, phase_mode = self.phase_polarization.fold(folded)
        return folded, self.encode_mode(sky_mode, phase_mode)

    def unfold(
        self, folded: NDArray[np.float64], mode: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        if np.asarray(mode).shape != (len(folded),):
            raise ValueError("Expected one LISA symmetry mode per particle.")
        sky_mode, phase_mode = self.decode_mode(mode)
        values = np.asarray(folded)
        if self.phase_polarization is not None:
            if phase_mode is None:
                raise ValueError("Missing phase/polarization mode labels.")
            values = self.phase_polarization.unfold(values, phase_mode)
        raw, _ = self.phase_transform.inverse(values)
        raw_unfolded = self.sky_symmetry.unfold(raw, sky_mode)
        return self.phase_transform.forward(raw_unfolded)[0]


__all__ = [
    "KnownBetaMode",
    "KnownIotaMode",
    "KnownSymmetryGroup",
    "LISADeltaPhaseExtrinsicSymmetry",
    "LISAExtrinsicSymmetry",
    "LISASkySymmetry",
    "is_periodic_parameter",
]
