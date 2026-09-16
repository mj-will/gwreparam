import numpy as np
import pytest

from gwreparam.phase import DeltaPhase as DeltaPhaseTransform
from gwreparam.symmetry import (
    Dominant22DeltaPhasePolarizationSymmetry,
    Dominant22PhasePolarizationSymmetry,
)


@pytest.fixture
def particles():
    rng = np.random.default_rng(931)
    return rng.uniform([0, 0, -1], [2 * np.pi, np.pi, 1], (100, 3))


@pytest.mark.parametrize("sign", [-1, 1])
def test_same_physical_orbits_as_raw_phase(particles, sign) -> None:
    conversion = DeltaPhaseTransform(
        0, 1, 2, sign=sign, inclination_mode="cos"
    )
    raw = Dominant22PhasePolarizationSymmetry(0, 1)
    delta_symmetry = Dominant22DeltaPhasePolarizationSymmetry(0, 1, 2)
    delta, _, _ = conversion.forward(particles)
    expected = np.stack(
        [conversion.forward(raw.apply_mode(particles, m))[0] for m in range(4)]
    )
    actual = np.stack([delta_symmetry.apply_mode(delta, m) for m in range(4)])
    # Compare sets, since inclination and convention can permute branch IDs.
    for candidate in actual:
        error = expected - candidate[None, :, :]
        error[:, :, 0] = (error[:, :, 0] + np.pi) % (2 * np.pi) - np.pi
        error[:, :, 1] = (error[:, :, 1] + np.pi / 2) % np.pi - np.pi / 2
        assert np.all(np.min(np.max(np.abs(error), axis=2), axis=0) < 1e-12)


def test_fold_domains_modes_and_roundtrip() -> None:
    particles = np.array(
        [
            [0.1, 0.2, 1],
            [0.1, 2, -1],
            [4, 0.2, 1],
            [4, 2, -1],
            [np.pi, np.pi / 2, 0.1],
            [0, 0, -0.1],
        ]
    )
    original = particles.copy()
    symmetry = Dominant22DeltaPhasePolarizationSymmetry(0, 1, 2)
    reduced, modes = symmetry.fold(particles)
    np.testing.assert_array_equal(modes, [0, 1, 2, 3, 3, 0])
    assert np.all((reduced[:, 0] >= 0) & (reduced[:, 0] < np.pi))
    assert np.all((reduced[:, 1] >= 0) & (reduced[:, 1] < np.pi / 2))
    np.testing.assert_allclose(symmetry.unfold(reduced, modes), original)
    np.testing.assert_array_equal(particles, original)
    for mode in range(4):
        folded, _ = symmetry.fold(symmetry.apply_mode(particles, mode))
        np.testing.assert_allclose(folded, reduced, atol=1e-12)


@pytest.mark.parametrize(
    "mode, boundary", [("cos", 0.0), ("angle", np.pi / 2)]
)
def test_edge_on_rejected(mode, boundary) -> None:
    symmetry = Dominant22DeltaPhasePolarizationSymmetry(
        0, 1, 2, inclination_mode=mode
    )
    x = np.array([[1.0, 0.2, boundary]])
    with pytest.raises(ValueError, match="non-edge-on"):
        symmetry.fold(x)
    with pytest.raises(ValueError, match="non-edge-on"):
        symmetry.apply_mode(x, 1)


@pytest.mark.parametrize("modes", [[-1], [4], [0.5], [True], [0, 1]])
def test_invalid_modes(modes) -> None:
    symmetry = Dominant22DeltaPhasePolarizationSymmetry(0, 1, 2)
    with pytest.raises(ValueError):
        symmetry.unfold(np.array([[1.0, 0.2, 1.0]]), np.array(modes))
