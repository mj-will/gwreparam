from __future__ import annotations

import numpy as np
import pytest

from gwreparam.symmetry import PeriodicParameterSymmetry


@pytest.mark.parametrize(
    "batch_slice", [slice(None), slice(2, 3)], ids=["batch", "single"]
)
def test_periodic_symmetry_round_trip_and_domain(
    periodic_symmetry,
    generic_particles,
    batch_slice,
) -> None:
    particles = generic_particles[batch_slice]

    folded, modes = periodic_symmetry.fold(particles)
    restored = periodic_symmetry.unfold(folded, modes)

    np.testing.assert_allclose(restored, particles)
    assert np.all(folded[:, 0] >= periodic_symmetry.origin)
    assert np.all(
        folded[:, 0]
        < periodic_symmetry.origin + periodic_symmetry.reduced_period
    )
    assert modes.shape == (len(particles),)
    assert np.issubdtype(modes.dtype, np.integer)
    assert np.all((modes >= 0) & (modes < periodic_symmetry.n_modes))
    np.testing.assert_array_equal(
        modes,
        periodic_symmetry.determine_mode(particles),
    )


def test_periodic_symmetries_compose_by_coordinate() -> None:
    particles = np.asarray(
        [
            [0.1, 0.2],
            [0.6 * np.pi, 0.6 * np.pi],
            [1.1 * np.pi, 0.9 * np.pi],
            [1.8 * np.pi, 0.1 * np.pi],
        ]
    )
    phase = PeriodicParameterSymmetry(0, 2 * np.pi, 4)
    polarization = PeriodicParameterSymmetry(1, np.pi, 2)

    folded, phase_modes = phase.fold(particles)
    folded, polarization_modes = polarization.fold(folded)
    restored = phase.unfold(
        polarization.unfold(folded, polarization_modes),
        phase_modes,
    )

    np.testing.assert_array_equal(phase_modes, [0, 1, 2, 3])
    np.testing.assert_array_equal(polarization_modes, [0, 1, 1, 0])
    np.testing.assert_allclose(restored, particles)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"period": 0.0, "n_modes": 2}, "period must be positive"),
        ({"period": -1.0, "n_modes": 2}, "period must be positive"),
        ({"period": 1.0, "n_modes": 0}, "n_modes must be at least one"),
    ],
)
def test_periodic_symmetry_rejects_invalid_configuration(
    kwargs, match
) -> None:
    with pytest.raises(ValueError, match=match):
        PeriodicParameterSymmetry(0, **kwargs)


@pytest.mark.parametrize(
    "modes",
    [np.asarray([0, 1]), np.asarray([-1]), np.asarray([4])],
    ids=["wrong-shape", "negative", "too-large"],
)
def test_periodic_symmetry_rejects_invalid_modes(
    periodic_symmetry, modes
) -> None:
    folded = np.asarray([[0.1, 0.2]])
    match = "one symmetry mode" if len(modes) != 1 else "valid range"
    with pytest.raises(ValueError, match=match):
        periodic_symmetry.unfold(folded, modes)


@pytest.mark.parametrize(
    "batch_slice", [slice(None), slice(1, 2)], ids=["batch", "single"]
)
def test_dominant_symmetry_operations_agree(
    dominant_symmetry,
    generic_particles,
    batch_slice,
) -> None:
    particles = generic_particles[batch_slice]

    folded, modes = dominant_symmetry.fold(particles)
    restored = dominant_symmetry.unfold(folded, modes)

    np.testing.assert_allclose(restored, particles)
    np.testing.assert_array_equal(
        modes,
        dominant_symmetry.determine_mode(particles),
    )
    assert np.all((0.0 <= folded[:, 0]) & (folded[:, 0] < 0.5 * np.pi))
    assert np.all((0.0 <= folded[:, 1]) & (folded[:, 1] < np.pi))
    assert modes.shape == (len(particles),)
    assert np.issubdtype(modes.dtype, np.integer)
    assert np.all((0 <= modes) & (modes < dominant_symmetry.n_modes))

    for mode in range(dominant_symmetry.n_modes):
        expected = dominant_symmetry.unfold(
            folded,
            np.full(len(folded), mode),
        )
        np.testing.assert_allclose(
            dominant_symmetry.apply_mode(folded, mode),
            expected,
        )


def test_dominant_symmetry_shifts_are_defensive_copy(
    dominant_symmetry,
) -> None:
    shifts = dominant_symmetry.shifts
    shifts[0] = -1.0

    np.testing.assert_array_equal(dominant_symmetry.shifts[0], [0.0, 0.0])


@pytest.mark.parametrize(
    "modes",
    [np.asarray([0, 1]), np.asarray([-1]), np.asarray([4])],
    ids=["wrong-shape", "negative", "too-large"],
)
def test_dominant_symmetry_rejects_invalid_modes(
    dominant_symmetry, modes
) -> None:
    folded = np.asarray([[0.1, 0.2]])
    match = "one symmetry mode" if len(modes) != 1 else "valid range"
    with pytest.raises(ValueError, match=match):
        dominant_symmetry.unfold(folded, modes)


def test_dominant_symmetry_apply_mode_rejects_invalid_mode(
    dominant_symmetry,
) -> None:
    with pytest.raises(ValueError, match="valid range"):
        dominant_symmetry.apply_mode(np.asarray([[0.1, 0.2]]), 4)
