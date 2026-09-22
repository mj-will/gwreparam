"""Aligned-spin conventions, domains and full retained-q determinants."""

import numpy as np
import pytest

from gwreparam.spins import (
    aligned_spin_coordinates as forward,
)
from gwreparam.spins import (
    inverse_aligned_spin_coordinates as inverse,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scalar", [False, True])
def test_roundtrip_dtype_and_broadcast(dtype, scalar: bool) -> None:
    a = np.asarray(0.75 if scalar else [[-1], [0.25], [1]], dtype=dtype)
    b = np.asarray(-0.5 if scalar else [-1, 0.5, 1], dtype=dtype)
    q = np.asarray(0.25 if scalar else [0.01, 0.25, 1], dtype=dtype)
    saved = [v.copy() for v in (a, b, q)]
    values, jf = forward(a, b, q)
    restored, ji = inverse(*values, q)
    expected = np.broadcast_arrays(a, b, q)
    tol = 5 * np.finfo(dtype).eps
    for result, original in zip(restored, expected):
        np.testing.assert_allclose(result, original, atol=tol, rtol=0)
    for result in (*values, jf, *restored, ji):
        assert isinstance(result, np.ndarray)
        assert result.shape == expected[0].shape
        assert result.dtype == dtype
    np.testing.assert_allclose(
        values[0], (a + q * b) / (1 + q), atol=tol, rtol=0
    )
    np.testing.assert_allclose(values[1], (a - b) / 2, atol=tol, rtol=0)
    np.testing.assert_allclose(jf, -np.log(2), atol=tol)
    np.testing.assert_array_equal(jf + ji, np.zeros_like(jf))
    for result, original in zip((a, b, q), saved):
        np.testing.assert_array_equal(result, original)


@pytest.mark.parametrize("q", [0.25, 1.0])
def test_known_normalization_and_boundaries(q: float) -> None:
    a, b = np.array([-1, -1, 1, 1]), np.array([-1, 1, -1, 1])
    (eff, diff), jf = forward(a, b, q)
    np.testing.assert_array_equal(diff, [0, -1, 1, 0])
    np.testing.assert_allclose(
        eff, [-1, (q - 1) / (1 + q), (1 - q) / (1 + q), 1]
    )
    restored, ji = inverse(eff, diff, q)
    np.testing.assert_allclose(restored, (a, b), atol=1e-15)
    np.testing.assert_array_equal(jf + ji, np.zeros(4))


@pytest.mark.parametrize("mapping", [forward, inverse])
@pytest.mark.parametrize("q", [0.05, 0.3, 0.8])
def test_full_numerical_jacobian(mapping, q: float) -> None:
    point = np.array([q, 0.23, -0.61])

    def apply(p):
        values, jac = mapping(p[1], p[2], p[0])
        return np.array([p[0], *values]), jac

    _, jac = apply(point)
    step = 1e-6
    columns = []
    for delta in step * np.eye(3):
        columns.append(
            (apply(point + delta)[0] - apply(point - delta)[0]) / (2 * step)
        )
    np.testing.assert_allclose(
        np.linalg.slogdet(np.array(columns).T)[1], jac, atol=2e-10
    )


@pytest.mark.parametrize("q", [0, -0.1, 1.1, np.nan, np.inf])
def test_invalid_mass_ratio(q: float) -> None:
    with pytest.raises(ValueError, match="mass ratio"):
        forward(0.1, 0.2, q)
    values, jac = inverse([0.1, 0.1], 0.2, [q, 0.5])
    assert all(np.isnan(v[0]) for v in values)
    assert np.isneginf(jac[0])
    assert all(np.isfinite(v[1]) for v in values)
    assert jac[1] == np.log(2)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("column", [0, 1])
def test_invalid_coordinates(value: float, column: int) -> None:
    inputs = [0.1, 0.2, 0.5]
    inputs[column] = value
    with pytest.raises(ValueError, match="finite spins"):
        forward(*inputs)
    coords, jac = inverse(*inputs)
    assert np.isnan(coords).all()
    assert np.isneginf(jac)


def test_conversion_and_no_implicit_prior() -> None:
    coords, jac = forward([1, -1], [0, 0], [1, 1])
    assert coords[0].dtype == np.float64
    assert jac.dtype == np.float64
    # These are algebraic primitives, not a rectangular transformed prior.
    coords, jac = inverse(1.0, 1.0, 0.5)
    assert coords[0] > 1
    assert jac == np.log(2)
    np.testing.assert_allclose(forward(*coords, 0.5)[0], (1, 1))
    for mapping in (forward, inverse):
        with pytest.raises(ValueError, match="real numbers"):
            mapping(1j, 0, 0.5)
        with pytest.raises(ValueError):
            mapping([0, 1], [0, 1, 2], 0.5)
    coords, jac = inverse(np.finfo(float).max, np.finfo(float).max, 0.1)
    assert np.isnan(coords).all()
    assert np.isneginf(jac)


@pytest.mark.parametrize("q", [0.05, 0.4, 1.0])
def test_induced_support_area(q: float) -> None:
    # A rectangle in component spins becomes a q-dependent parallelogram.
    a = np.array([-0.7, 0.8, 0.8, -0.7])
    b = np.array([-0.3, -0.3, 0.9, 0.9])
    (eff, diff), jac = forward(a, b, q)
    area = (
        abs(np.dot(eff, np.roll(diff, 1)) - np.dot(diff, np.roll(eff, 1))) / 2
    )
    physical_area = 1.5 * 1.2
    np.testing.assert_allclose(area, physical_area / 2)
    # The induced uniform-component density is doubled inside that support.
    np.testing.assert_allclose(area * np.exp(-jac[0]) / physical_area, 1)
