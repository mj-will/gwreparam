"""Reusable source, longitude, phase and amplitude operations."""

import numpy as np
import pytest

from gwreparam.lisa.coordinates import inverse_source_helicity, source_helicity
from gwreparam.lisa.response import (
    frozen_response_log_amplitude_correction,
    frozen_response_phase_correction,
    longitude_phase_shift,
    longitude_rotation,
)


@pytest.mark.parametrize("distance", [None, 2.0])
def test_source_equations_and_inverse(distance: float | None) -> None:
    phi = np.array([[0.2], [1.3]])
    c = np.array([-0.7, 0.4, 0.8])
    psi = np.array([0.3, 0.2, 1.0])
    values, jac, labels = source_helicity(
        phi, c, psi, distance=distance, distance_scale=1
    )
    phase, x, y = values[:3]
    h = ((1 - abs(c)) / (1 + abs(c))) ** 2 * np.exp(4j * np.sign(c) * psi)
    np.testing.assert_allclose(x + 1j * y, np.broadcast_to(h, (2, 3)))
    np.testing.assert_allclose(
        np.exp(1j * phase), np.exp(2j * (phi - np.sign(c) * psi))
    )
    physical, inverse = inverse_source_helicity(
        phase,
        x,
        y,
        hand=labels["hand"],
        psi_branch=labels["psi"],
        phase_branch=labels["phase"],
        log_amplitude=values[3] if distance is not None else None,
        distance_scale=1,
    )
    np.testing.assert_allclose(physical[0], np.broadcast_to(phi, (2, 3)))
    np.testing.assert_allclose(physical[1], np.broadcast_to(c, (2, 3)))
    np.testing.assert_allclose(physical[2], np.broadcast_to(psi, (2, 3)))
    np.testing.assert_allclose(jac + inverse, 0, atol=1e-13)
    if distance is not None:
        np.testing.assert_allclose(
            values[3],
            np.broadcast_to(
                np.log((1 + abs(c)) ** 2 / (4 * distance)), (2, 3)
            ),
        )
        np.testing.assert_allclose(physical[3], distance)


def test_longitude_primitives() -> None:
    h = np.array([0.2 + 0.1j, -0.4 + 0.3j])
    lam, hand = np.array([0.3, 1.2]), np.array([-1, 1])
    (x, y), jac = longitude_rotation(h.real, h.imag, lam, hand=hand)
    np.testing.assert_allclose(x + 1j * y, h * np.exp(-4j * hand * lam))
    (xr, yr), inverse = longitude_rotation(x, y, lam, hand=hand, inverse=True)
    np.testing.assert_allclose(xr + 1j * yr, h)
    np.testing.assert_array_equal(jac + inverse, 0)
    np.testing.assert_allclose(
        longitude_phase_shift(lam, hand=hand), 2 * hand * lam
    )


def test_response_shifts_and_null() -> None:
    h = np.array([0.2 + 0.1j, -0.4 + 0.3j, -1 + 0j])
    b = np.array([0.1, 1.0, 0.0])
    k = ((1 - b) / (1 + b)) ** 2
    z = 1 + k * h
    phase = frozen_response_phase_correction(h.real, h.imag, b)
    amplitude = frozen_response_log_amplitude_correction(h.real, h.imag, b)
    np.testing.assert_allclose(phase[:2], np.angle(z[:2]))
    np.testing.assert_allclose(
        amplitude[:2], np.log((1 + b[:2]) ** 2 / 4 * abs(z[:2]))
    )
    assert np.isnan(phase[-1])
    assert np.isneginf(amplitude[-1])
    assert phase[1] == amplitude[1] == 0


@pytest.mark.parametrize(
    "function",
    [
        frozen_response_phase_correction,
        frozen_response_log_amplitude_correction,
    ],
)
def test_shift_latitude_validation(function) -> None:
    with pytest.raises(ValueError, match="Folded sine latitude"):
        function(0.2, 0.1, -0.1)
