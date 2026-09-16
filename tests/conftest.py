from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from gwreparam.lisa.symmetry import (
    KnownSymmetryGroup,
    LISADeltaPhaseExtrinsicSymmetry,
    LISAExtrinsicSymmetry,
)
from gwreparam.symmetry import (
    Dominant22DeltaPhasePolarizationSymmetry,
    Dominant22PhasePolarizationSymmetry,
    PeriodicParameterSymmetry,
)


@pytest.fixture
def lisa_configuration_factory() -> Callable:
    def _factory(
        *,
        beta_mode: str = "angle",
        inclination_mode: str = "angle",
        include_phase_polarization: bool = False,
    ):
        beta_name = "beta" if beta_mode == "angle" else "sin_beta"
        beta_bounds = (
            (-0.5 * np.pi, 0.5 * np.pi)
            if beta_mode == "angle"
            else (-1.0, 1.0)
        )
        inclination_name = (
            "iota" if inclination_mode == "angle" else "cos_iota"
        )
        inclination_bounds = (
            (0.0, np.pi) if inclination_mode == "angle" else (-1.0, 1.0)
        )
        parameters = (
            "phase",
            "psi",
            "lam",
            beta_name,
            inclination_name,
            "distance",
        )
        bounds = {
            "phase": (0.0, 2 * np.pi),
            "psi": (0.0, np.pi),
            "lam": (0.0, 2 * np.pi),
            beta_name: beta_bounds,
            inclination_name: inclination_bounds,
            "distance": (1.0, 10.0),
        }
        phase_polarization = (
            Dominant22PhasePolarizationSymmetry(0, 1)
            if include_phase_polarization
            else None
        )
        return parameters, bounds, phase_polarization

    return _factory


@pytest.fixture
def lisa_symmetry_factory(lisa_configuration_factory) -> Callable:
    def _factory(
        *,
        symmetry_group: KnownSymmetryGroup = "8-mode",
        beta_mode: str = "angle",
        inclination_mode: str = "angle",
        include_phase_polarization: bool = False,
        roll: bool = False,
    ) -> LISAExtrinsicSymmetry:
        parameters, bounds, phase_polarization = lisa_configuration_factory(
            beta_mode=beta_mode,
            inclination_mode=inclination_mode,
            include_phase_polarization=include_phase_polarization,
        )
        return LISAExtrinsicSymmetry(
            parameters,
            bounds,
            symmetry_group=symmetry_group,
            phase_polarization=phase_polarization,
            roll=roll,
        )

    return _factory


@pytest.fixture
def lisa_particles_factory() -> Callable:
    def _factory(
        *,
        beta_mode: str = "angle",
        inclination_mode: str = "angle",
    ) -> np.ndarray:
        eps = 1.0e-12
        beta = (
            np.asarray([-0.4, -eps, eps, 0.4])
            if beta_mode == "angle"
            else np.asarray([-0.8, -eps, eps, 0.8])
        )
        inclination = (
            np.asarray([0.2, 1.0, 2.0, np.pi - 0.2])
            if inclination_mode == "angle"
            else np.asarray([0.8, 0.2, -0.3, -0.8])
        )
        return np.column_stack(
            [
                [0.0, 0.5 * np.pi + eps, np.pi + eps, 2 * np.pi - eps],
                [0.0, 0.4, 0.5 * np.pi, np.pi - eps],
                [0.0, 0.5 * np.pi + eps, np.pi + eps, 2 * np.pi - eps],
                beta,
                inclination,
                [2.0, 3.0, 4.0, 5.0],
            ]
        )

    return _factory


@pytest.fixture
def lisa_symmetry(lisa_symmetry_factory) -> LISAExtrinsicSymmetry:
    return lisa_symmetry_factory(include_phase_polarization=True)


@pytest.fixture
def lisa_particles(lisa_particles_factory) -> np.ndarray:
    return lisa_particles_factory()


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


@pytest.fixture
def relative_case():
    bounds = dict(
        lam=(0, 2 * np.pi),
        sin_beta=(-1, 1),
        phi_ref=(0, 2 * np.pi),
        cos_inc=(-1, 1),
        psi=(0, np.pi),
    )
    sym = LISAExtrinsicSymmetry(list(bounds), bounds)
    x = np.random.default_rng(114).uniform(
        *np.array(list(bounds.values())).T, (20000, 5)
    )
    return sym, x


@pytest.fixture(
    params=[
        ("W", "cartesian"),
        ("Z", "cartesian"),
        ("W", "polar-area"),
        ("Z", "polar-area"),
    ],
    ids=["W-cartesian", "Z-cartesian", "W-polar-area", "Z-polar-area"],
)
def chart_options(request):
    return {
        "orientation": request.param[0],
        "representation": request.param[1],
    }


@pytest.fixture
def relative_case_factory():
    def _factory(
        distance=True,
        delta=True,
        sign=-1,
        group="reflected-antipodal",
        phase_fold=True,
    ):
        bounds = {
            "mchirp": (1.0, 3.0),
            "q": (0.2, 1.0),
            "chi1z": (-1.0, 1.0),
            "chi2z": (-1.0, 1.0),
            "t_ref": (0.0, 10.0),
            "lam": (0.0, 2 * np.pi),
            "sin_beta": (-1.0, 1.0),
        }
        if distance:
            bounds["distance"] = (1.0, 10.0)
        bounds.update(
            {
                "delta_phi" if delta else "phi_ref": (0.0, 2 * np.pi),
                "cos_inc": (-1.0, 1.0),
                "psi": (0.0, np.pi),
            }
        )
        names = list(bounds)
        pi, ci, wi = [
            names.index(n)
            for n in ("delta_phi" if delta else "phi_ref", "cos_inc", "psi")
        ]
        phase = (
            (
                Dominant22DeltaPhasePolarizationSymmetry(pi, wi, ci)
                if delta
                else Dominant22PhasePolarizationSymmetry(pi, wi)
            )
            if phase_fold
            else None
        )
        cls = (
            LISADeltaPhaseExtrinsicSymmetry if delta else LISAExtrinsicSymmetry
        )
        sym = cls(
            names,
            bounds,
            symmetry_group=group,
            phase_polarization=phase,
            **({"phase_sign": sign} if delta else {}),
        )
        low, high = np.array(list(bounds.values())).T
        x = np.random.default_rng(915).uniform(low, high, (256, len(names)))
        return sym, None, x

    return _factory
