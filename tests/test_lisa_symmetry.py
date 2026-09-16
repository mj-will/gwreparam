from __future__ import annotations

import numpy as np
import pytest

from gwreparam.lisa.symmetry import (
    LISAExtrinsicSymmetry,
    is_periodic_parameter,
)
from gwreparam.symmetry import Dominant22PhasePolarizationSymmetry


def _assert_periodic_allclose(actual, expected, period) -> None:
    difference = (
        np.mod(actual - expected + 0.5 * period, period) - 0.5 * period
    )
    np.testing.assert_allclose(difference, 0.0, atol=1.0e-12)


@pytest.mark.parametrize(
    "symmetry_group",
    ["reflected-antipodal", "8-mode"],
)
@pytest.mark.parametrize("beta_mode", ["angle", "sin"])
@pytest.mark.parametrize("inclination_mode", ["angle", "cos"])
@pytest.mark.parametrize(
    "include_phase_polarization",
    [False, True],
    ids=["sky-only", "with-phase-polarization"],
)
def test_lisa_symmetry_round_trip_domains_and_modes(
    lisa_symmetry_factory,
    lisa_particles_factory,
    symmetry_group,
    beta_mode,
    inclination_mode,
    include_phase_polarization,
) -> None:
    symmetry = lisa_symmetry_factory(
        symmetry_group=symmetry_group,
        beta_mode=beta_mode,
        inclination_mode=inclination_mode,
        include_phase_polarization=include_phase_polarization,
    )
    particles = lisa_particles_factory(
        beta_mode=beta_mode,
        inclination_mode=inclination_mode,
    )

    folded, modes = symmetry.fold(particles)
    restored = symmetry.unfold(folded, modes)

    _assert_periodic_allclose(restored[:, 0], particles[:, 0], 2 * np.pi)
    _assert_periodic_allclose(restored[:, 1], particles[:, 1], np.pi)
    _assert_periodic_allclose(restored[:, 2], particles[:, 2], 2 * np.pi)
    np.testing.assert_allclose(restored[:, 3:], particles[:, 3:])
    assert np.all(
        (0.0 <= folded[:, 2]) & (folded[:, 2] < symmetry.lambda_step)
    )
    assert np.all(folded[:, 3] >= 0.0)
    if include_phase_polarization:
        assert np.all((0.0 <= folded[:, 0]) & (folded[:, 0] < 0.5 * np.pi))
    assert modes.shape == (len(particles),)
    assert np.issubdtype(modes.dtype, np.integer)
    assert np.all((0 <= modes) & (modes < symmetry.n_modes))
    np.testing.assert_array_equal(modes, symmetry.determine_mode(particles))


@pytest.mark.parametrize(
    ("symmetry_group", "n_longitude", "n_sky"),
    [("reflected-antipodal", 2, 4), ("8-mode", 4, 8)],
)
def test_lisa_sky_mode_component_ordering(
    lisa_symmetry_factory,
    symmetry_group,
    n_longitude,
    n_sky,
) -> None:
    symmetry = lisa_symmetry_factory(symmetry_group=symmetry_group)
    sky_modes = np.arange(n_sky)

    longitude, latitude = symmetry.decode_sky_mode(sky_modes)

    np.testing.assert_array_equal(longitude, sky_modes % n_longitude)
    np.testing.assert_array_equal(latitude, sky_modes // n_longitude)
    assert symmetry.n_longitude_modes == n_longitude
    assert symmetry.n_sky_modes == n_sky


@pytest.mark.parametrize(
    "include_phase_polarization",
    [False, True],
    ids=["sky-only", "composite"],
)
def test_lisa_composite_mode_encoding_round_trip(
    lisa_symmetry_factory,
    include_phase_polarization,
) -> None:
    symmetry = lisa_symmetry_factory(
        include_phase_polarization=include_phase_polarization
    )
    modes = np.arange(symmetry.n_modes)

    sky_modes, phase_modes = symmetry.decode_mode(modes)
    encoded = symmetry.encode_mode(sky_modes, phase_modes)

    np.testing.assert_array_equal(encoded, modes)
    if include_phase_polarization:
        np.testing.assert_array_equal(
            modes,
            sky_modes * symmetry.n_phase_polarization_modes + phase_modes,
        )
    else:
        assert phase_modes is None
        np.testing.assert_array_equal(sky_modes, modes)


def test_lisa_folded_representative_can_generate_every_mode(
    lisa_symmetry,
    lisa_particles,
) -> None:
    folded, _ = lisa_symmetry.fold(lisa_particles[:1])

    for mode in range(lisa_symmetry.n_modes):
        full = lisa_symmetry.unfold(folded, np.asarray([mode]))
        refolded, recovered_mode = lisa_symmetry.fold(full)
        np.testing.assert_allclose(refolded, folded)
        np.testing.assert_array_equal(recovered_mode, [mode])


def test_lisa_transformed_bounds_and_periodic_parameters(
    lisa_symmetry_factory,
) -> None:
    sky_only = lisa_symmetry_factory(include_phase_polarization=False)
    composite = lisa_symmetry_factory(include_phase_polarization=True)

    sky_bounds = sky_only.transformed_bounds()
    composite_bounds = composite.transformed_bounds()

    np.testing.assert_allclose(
        sky_bounds[sky_only.lambda_index],
        (0.0, sky_only.lambda_step),
    )
    np.testing.assert_allclose(
        sky_bounds[sky_only.beta_index], (0.0, np.pi / 2)
    )
    np.testing.assert_allclose(
        sky_bounds[sky_only.phase_index], (0.0, 2 * np.pi)
    )
    np.testing.assert_allclose(
        composite_bounds[composite.phase_index],
        (0.0, 0.5 * np.pi),
    )
    assert composite.periodic_parameters() == ("phase", "psi", "lam")


@pytest.mark.parametrize(
    ("parameter", "expected"),
    [("phase", True), ("psi", True), ("lam", True), ("beta", False)],
)
def test_lisa_periodic_parameter_detection(parameter, expected) -> None:
    assert is_periodic_parameter(parameter) is expected


@pytest.mark.parametrize("roll", [False, True])
def test_lisa_roll_application_and_removal_are_inverses(
    lisa_symmetry_factory,
    lisa_particles,
    roll,
) -> None:
    symmetry = lisa_symmetry_factory(
        include_phase_polarization=True,
        roll=roll,
    )
    folded, _ = symmetry.fold(lisa_particles)
    shifts = symmetry.compute_roll_shifts(folded)

    rolled = symmetry.apply_roll(folded, shifts)
    restored = symmetry.remove_roll(rolled, shifts)

    np.testing.assert_allclose(restored, folded)


@pytest.mark.parametrize(
    ("method", "argument", "match"),
    [
        ("decode_sky_mode", np.asarray([-1]), "sky mode"),
        ("decode_mode", np.asarray([32]), "extrinsic mode"),
        ("unfold", np.asarray([0, 1]), "one LISA extrinsic mode"),
    ],
)
def test_lisa_symmetry_rejects_invalid_modes(
    lisa_symmetry,
    lisa_particles,
    method,
    argument,
    match,
) -> None:
    with pytest.raises(ValueError, match=match):
        if method == "unfold":
            lisa_symmetry.unfold(lisa_particles[:1], argument)
        else:
            getattr(lisa_symmetry, method)(argument)


def test_lisa_composite_mode_validation(lisa_symmetry) -> None:
    sky = np.asarray([0, 1])

    with pytest.raises(ValueError, match="Missing phase/polarization"):
        lisa_symmetry.encode_mode(sky)
    with pytest.raises(ValueError, match="equal shape"):
        lisa_symmetry.encode_mode(sky, np.asarray([0]))
    with pytest.raises(ValueError, match="outside the valid range"):
        lisa_symmetry.encode_mode(sky, np.asarray([0, 4]))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"symmetry_group": "unknown"}, "Unknown symmetry_group"),
        ({"beta": (0.0, 1.0)}, "Unsupported beta bounds"),
        ({"iota": (-1.0, 2.0)}, "Unsupported inclination bounds"),
        ({"lam": (-np.pi, np.pi)}, "must start at 0"),
        ({"psi": (0.0, 2 * np.pi)}, "must end at"),
    ],
)
def test_lisa_symmetry_rejects_invalid_configuration(
    lisa_configuration_factory,
    change,
    match,
) -> None:
    parameters, bounds, phase_polarization = lisa_configuration_factory(
        include_phase_polarization=True
    )
    symmetry_group = change.get("symmetry_group", "8-mode")
    bounds.update(
        {name: value for name, value in change.items() if name in bounds}
    )

    with pytest.raises(ValueError, match=match):
        LISAExtrinsicSymmetry(
            parameters,
            bounds,
            symmetry_group=symmetry_group,
            phase_polarization=phase_polarization,
        )


def test_lisa_symmetry_rejects_mismatched_phase_indices(
    lisa_configuration_factory,
) -> None:
    parameters, bounds, _ = lisa_configuration_factory()

    with pytest.raises(ValueError, match="indices do not match"):
        LISAExtrinsicSymmetry(
            parameters,
            bounds,
            phase_polarization=Dominant22PhasePolarizationSymmetry(1, 0),
        )
