"""Pure ratio invariants, independent determinants and singular domains."""

import numpy as np
import pytest

from gwreparam.lisa.response import (
    frozen_response_ratio as forward,
)
from gwreparam.lisa.response import (
    frozen_response_ratio_log_jacobian,
    inverse_frozen_response_ratio_log_jacobian,
)
from gwreparam.lisa.response import (
    inverse_frozen_response_ratio as inverse,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_broadcast_and_branches(dtype) -> None:
    rng = np.random.default_rng(817)
    h = (
        0.8
        * np.sqrt(rng.uniform(size=(2, 128)))
        * np.exp(2j * np.pi * rng.uniform(size=(2, 128)))
    )
    b = rng.uniform(0.1, 1, size=h.shape).astype(dtype)
    lam = np.array([[0], [2 * np.pi]], dtype=dtype)
    hand = np.array([[-1], [1]])
    x, y = h.real.astype(dtype), h.imag.astype(dtype)
    saved = x.copy()
    values, jf = forward(x, y, b, lam, hand=hand)
    restored, ji = inverse(*values, b, lam, hand=hand)
    tol = 5e-6 if dtype == np.float32 else 2e-14
    np.testing.assert_allclose(restored, (x, y), atol=tol)
    np.testing.assert_allclose(jf + ji, 0, atol=tol)
    np.testing.assert_array_equal(x, saved)
    k = ((1 - b) / (1 + b)) ** 2
    expected = (
        np.exp(-4j * hand * lam) * (x + 1j * y + k) / (1 + k * (x + 1j * y))
    )
    np.testing.assert_allclose(values[0] + 1j * values[1], expected, atol=tol)
    assert np.all(np.hypot(*values) < 1)
    for result in (*values, jf, *restored, ji):
        assert result.dtype == dtype
        assert result.shape == h.shape
    np.testing.assert_array_equal(
        jf, frozen_response_ratio_log_jacobian(x, y, b, lam, hand=hand)
    )
    np.testing.assert_array_equal(
        ji,
        inverse_frozen_response_ratio_log_jacobian(*values, b, lam, hand=hand),
    )


@pytest.mark.parametrize("b", [0.01, 0.3, 1.0])
@pytest.mark.parametrize("hand", [-1, 1])
@pytest.mark.parametrize("mapping", [forward, inverse])
def test_numerical_jacobian(b, hand, mapping) -> None:
    h = 0.23 - 0.41j
    _, jac = mapping(h.real, h.imag, b, 1.13, hand=hand)
    columns = []
    for delta in (1e-6, 1e-6j):
        plus = mapping((h + delta).real, (h + delta).imag, b, 1.13, hand=hand)[
            0
        ]
        minus = mapping(
            (h - delta).real, (h - delta).imag, b, 1.13, hand=hand
        )[0]
        columns.append((np.array(plus) - minus) / 2e-6)
    np.testing.assert_allclose(
        np.linalg.slogdet(np.array(columns).T)[1], jac, atol=2e-8
    )


@pytest.mark.parametrize("b", [1e-3, 1e-6, 1e-9, 1e-12])
@pytest.mark.parametrize("hand", [-1, 1])
def test_near_plane(b: float, hand: int) -> None:
    h = -0.37 + 0.21j
    values, jf = forward(h.real, h.imag, b, 0.71, hand=hand)
    restored, ji = inverse(*values, b, 0.71, hand=hand)
    # The inverse condition number grows as 1/b: do not promise uniform
    # accuracy arbitrarily near the singular plane.
    tol = 5 * np.finfo(float).eps / b
    np.testing.assert_allclose(restored, (h.real, h.imag), atol=tol, rtol=0)
    np.testing.assert_allclose(jf + ji, 0, atol=10 * tol)
    assert np.isfinite(jf)


def test_exact_plane_and_invalid_proposals() -> None:
    with pytest.raises(ValueError, match="exact plane"):
        forward(0.2, 0.1, 0, 0, hand=1)
    for x, y, b in [
        (0, 0, 0),
        (1 / ((1 - 1 / 3) / (1 + 1 / 3)) ** 2, 0, 1 / 3),
        (np.nan, 0, 0.5),
        (0, np.inf, 0.5),
    ]:
        values, jac = inverse(x, y, b, 0, hand=1)
        assert np.isnan(values).all()
        assert np.isneginf(jac)
    # Domain support belongs downstream; the finite map is not clipped.
    values, jac = inverse(2, 0, 1, 0, hand=-1)
    np.testing.assert_array_equal(values, (2, 0))
    assert jac == 0


@pytest.mark.parametrize("mapping", [forward, inverse])
def test_validation_and_scalar_lists(mapping) -> None:
    for b in [-0.1, 1.1, np.nan, np.inf]:
        with pytest.raises(ValueError, match="hemisphere"):
            mapping(0, 0, b, 0, hand=1)
    for hand in [0, 2, 1.0]:
        with pytest.raises(ValueError, match="hand"):
            mapping(0, 0, 0.5, 0, hand=hand)
    with pytest.raises(ValueError, match="Longitude"):
        mapping(0, 0, 0.5, np.inf, hand=1)
    values, jac = mapping([0.2, 0.4], [0, 0], 1, 0, hand=1)
    np.testing.assert_array_equal(values, [[0.2, 0.4], [0, 0]])
    np.testing.assert_array_equal(jac, [0, 0])


@pytest.mark.parametrize("dtype,b", [(np.float32, 1e-9), (np.float64, 1e-20)])
def test_no_near_plane_cutoff(dtype, b) -> None:
    # At H=-1 the ratio is exactly -1, even when k rounds to one.
    b = dtype(b)
    values, jf = forward(dtype(-1), dtype(0), b, dtype(0), hand=1)
    restored, ji = inverse(*values, b, dtype(0), hand=1)
    np.testing.assert_allclose(values, (-1, 0), atol=np.finfo(dtype).eps)
    np.testing.assert_allclose(restored, (-1, 0), atol=np.finfo(dtype).eps)
    np.testing.assert_allclose(jf + ji, 0, atol=100 * np.finfo(dtype).eps)
    assert np.isfinite(jf)
