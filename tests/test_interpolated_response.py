"""Interpolated detector chart invariants and numerical boundaries."""

import numpy as np
import pytest

from gwreparam.lisa.response import (
    affine_frozen_response,
    frozen_response_ratio,
    interpolated_frozen_response_log_jacobian,
    inverse_affine_frozen_response,
    inverse_frozen_response_ratio,
    inverse_interpolated_frozen_response_log_jacobian,
)
from gwreparam.lisa.response import (
    interpolated_frozen_response as forward,
)
from gwreparam.lisa.response import (
    inverse_interpolated_frozen_response as inverse,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("fraction", [0.0, 0.5, 0.995, 1.0])
def test_broadcast_formula_and_roundtrip(dtype, fraction: float) -> None:
    rng = np.random.default_rng(711)
    h = (
        0.8
        * np.sqrt(rng.random((2, 64)))
        * np.exp(2j * np.pi * rng.random((2, 64)))
    )
    x, y = h.real.astype(dtype), h.imag.astype(dtype)
    saved = x.copy(), y.copy()
    b = np.linspace(0.05, 1, 64, dtype=dtype)
    lon = np.array([[0.2], [0.9]], dtype=dtype)
    hand = np.array([[-1], [1]])
    kwargs = dict(hand=hand, detector_ratio_fraction=fraction)
    values, jf = forward(x, y, b, lon, **kwargs)
    restored, ji = inverse(*values, b, lon, **kwargs)
    k = ((1 - b) / (1 + b)) ** 2
    expected = (
        np.exp(-4j * hand * lon)
        * (x + 1j * y + k)
        / (1 + (1 - fraction) * k + fraction * k * (x + 1j * y))
    )
    tol = 5e-5 if dtype == np.float32 else 2e-13
    np.testing.assert_allclose(values[0] + 1j * values[1], expected, atol=tol)
    np.testing.assert_allclose(restored, (x, y), atol=tol, rtol=0)
    np.testing.assert_allclose(jf + ji, 0, atol=tol)
    np.testing.assert_array_equal((x, y), saved)
    for result in (*values, jf, *restored, ji):
        assert result.shape == h.shape
        assert result.dtype == dtype
    np.testing.assert_array_equal(
        jf, interpolated_frozen_response_log_jacobian(x, y, b, lon, **kwargs)
    )
    np.testing.assert_array_equal(
        ji,
        inverse_interpolated_frozen_response_log_jacobian(
            *values, b, lon, **kwargs
        ),
    )


@pytest.mark.parametrize(
    "fraction,old,old_inverse",
    [
        (0, affine_frozen_response, inverse_affine_frozen_response),
        (1, frozen_response_ratio, inverse_frozen_response_ratio),
    ],
)
def test_exact_endpoints(fraction, old, old_inverse) -> None:
    b = np.array([1e-12, 0.2, 1.0])
    args = (0.2, -0.3, b, 0.7)
    values, jf = forward(*args, hand=-1, detector_ratio_fraction=fraction)
    expected, expected_jac = old(*args, hand=-1)
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(jf, expected_jac)
    back, ji = inverse(
        *values, b, 0.7, hand=-1, detector_ratio_fraction=fraction
    )
    expected, expected_jac = old_inverse(*values, b, 0.7, hand=-1)
    np.testing.assert_array_equal(back, expected)
    np.testing.assert_array_equal(ji, expected_jac)


@pytest.mark.parametrize("fraction", [0.2, 0.995, 0.999])
@pytest.mark.parametrize("hand", [-1, 1])
def test_plane_and_support(fraction: float, hand: int) -> None:
    b = np.array([0, 1e-16, 1e-8, 0.2, 1.0])[:, None]
    h = np.exp(1j * np.linspace(0, 2 * np.pi, 65))[None, :]
    kwargs = dict(hand=hand, detector_ratio_fraction=fraction)
    values, jf = forward(h.real, h.imag, b, 0.7, **kwargs)
    restored, ji = inverse(*values, b, 0.7, **kwargs)
    np.testing.assert_allclose(
        restored[0] + 1j * restored[1],
        np.broadcast_to(h, restored[0].shape),
        atol=3e-12,
    )
    np.testing.assert_allclose(jf + ji, 0, atol=3e-12)
    k = ((1 - b) / (1 + b)) ** 2
    denominator = 1 + k - 2 * fraction * k
    center = np.exp(-4j * hand * 0.7) * (1 - fraction) * k / denominator
    radius = (1 - fraction * k) / denominator
    np.testing.assert_allclose(
        abs(values[0] + 1j * values[1] - center),
        np.broadcast_to(radius, values[0].shape),
        atol=3e-12,
    )
    assert np.isfinite(jf).all()


@pytest.mark.parametrize("mapping", [forward, inverse])
@pytest.mark.parametrize("fraction", [0.3, 0.995])
@pytest.mark.parametrize("b", [0.0, 0.2, 1.0])
def test_numerical_jacobian(mapping, fraction: float, b: float) -> None:
    kwargs = dict(hand=-1, detector_ratio_fraction=fraction)
    z = np.array([0.23, -0.19])
    _, jac = mapping(*z, b, 0.7, **kwargs)
    columns = []
    for step in np.eye(2) * 1e-6:
        plus = mapping(*(z + step), b, 0.7, **kwargs)[0]
        minus = mapping(*(z - step), b, 0.7, **kwargs)[0]
        columns.append((np.array(plus) - minus) / 2e-6)
    np.testing.assert_allclose(
        np.linalg.slogdet(np.array(columns).T)[1], jac, atol=2e-7
    )


@pytest.mark.parametrize("mapping", [forward, inverse])
@pytest.mark.parametrize(
    "fraction",
    [
        None,
        True,
        np.bool_(False),
        "0.5",
        [],
        np.array(0.5),
        0.5j,
        np.nan,
        np.inf,
        -0.1,
        1.1,
    ],
)
def test_invalid_fraction(mapping, fraction) -> None:
    with pytest.raises(ValueError, match="detector_ratio_fraction"):
        mapping(0.2, 0.1, 0.3, 0.7, hand=1, detector_ratio_fraction=fraction)


@pytest.mark.parametrize("mapping", [forward, inverse])
def test_validation_and_invalid_coordinates(mapping) -> None:
    kwargs = dict(hand=1, detector_ratio_fraction=0.5)
    for b in [-0.1, 1.1, np.nan]:
        with pytest.raises(ValueError, match="latitude"):
            mapping(0, 0, b, 0, **kwargs)
    with pytest.raises(ValueError, match="Longitude"):
        mapping(0, 0, 0.5, np.inf, **kwargs)
    with pytest.raises(ValueError, match="hand"):
        mapping(0, 0, 0.5, 0, hand=1.0, detector_ratio_fraction=0.5)
    for x in [np.nan, np.inf]:
        values, jac = mapping(x, 0, 0.3, 0, **kwargs)
        assert np.isnan(values).all()
        assert np.isneginf(jac)
    values, jac = mapping([2, 3], [0, 0], 1, 0, **kwargs)
    np.testing.assert_array_equal(values, [[2, 3], [0, 0]])
    np.testing.assert_array_equal(jac, [0, 0])


def test_poles_and_ratio_plane() -> None:
    # At b=0 and fraction=1/2 the forward pole is H=-3, inverse pole U=2.
    for mapping, x in [(forward, -3), (inverse, 2)]:
        values, jac = mapping(x, 0, 0, 0, hand=1, detector_ratio_fraction=0.5)
        assert np.isnan(values).all()
        assert np.isneginf(jac)
    with pytest.raises(ValueError, match="exact plane"):
        forward(0.2, 0.1, 0, 0, hand=1, detector_ratio_fraction=1)
    values, jac = inverse(0.2, 0.1, 0, 0, hand=1, detector_ratio_fraction=1)
    assert np.isnan(values).all()
    assert np.isneginf(jac)
