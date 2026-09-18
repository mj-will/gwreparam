import numpy as np
import pytest

from gwreparam.lisa.response import (
    affine_frozen_response as forward,
)
from gwreparam.lisa.response import (
    affine_frozen_response_log_jacobian,
    inverse_affine_frozen_response_log_jacobian,
)
from gwreparam.lisa.response import (
    inverse_affine_frozen_response as inverse,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_response(dtype) -> None:
    rng = np.random.default_rng(815)
    shape = (2, 128)
    w = np.sqrt(rng.uniform(size=shape)) * np.exp(
        2j * np.pi * rng.uniform(size=shape)
    )
    w[:, :3] = [0, 1 - 1e-7, -1]
    b = rng.uniform(size=shape).astype(dtype)
    b[:, :2] = [0, 1]
    lam = rng.uniform(0, 2 * np.pi, size=shape).astype(dtype)
    lam[:, :3] = [0, 2 * np.pi - 1e-6, 2 * np.pi]
    hand = np.array([[-1], [1]])
    x, y = w.real.astype(dtype), w.imag.astype(dtype)
    (vx, vy), jf = forward(x, y, b, lam, hand=hand)
    (wx, wy), ji = inverse(vx, vy, b, lam, hand=hand)
    tol = 3e-6 if dtype == np.float32 else 3e-14
    for actual, expected in ((wx, x), (wy, y)):
        np.testing.assert_allclose(actual, expected, atol=tol)
    for result in (vx, vy, jf, wx, wy, ji):
        assert result.dtype == dtype
        assert result.shape == shape
        assert np.isfinite(result).all()
    assert np.all(np.hypot(vx, vy) <= 1 + tol)
    k = ((1 - b) / (1 + b)) ** 2
    expected = np.exp(-4j * hand * lam) * (x + 1j * y + k) / (1 + k)
    np.testing.assert_allclose(vx + 1j * vy, expected, atol=tol)
    np.testing.assert_allclose(jf + ji, 0, atol=tol)
    np.testing.assert_allclose(jf, affine_frozen_response_log_jacobian(b))
    np.testing.assert_allclose(
        ji, inverse_affine_frozen_response_log_jacobian(b)
    )
    (vxx, vyy), _ = forward(wx, wy, b, lam, hand=hand)
    np.testing.assert_allclose(vxx, vx, atol=tol)
    np.testing.assert_allclose(vyy, vy, atol=tol)
    (period_x, period_y), _ = forward(x, y, b, lam + 2 * np.pi, hand=hand)
    np.testing.assert_allclose(period_x, vx, atol=1e-5)
    np.testing.assert_allclose(period_y, vy, atol=1e-5)


@pytest.mark.parametrize("latitude", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("hand", [-1, 1])
def test_limits_and_determinant(latitude: float, hand: int) -> None:
    w, lam = 0.23 - 0.71j, 1.13
    (x, y), jf = forward(w.real, w.imag, latitude, lam, hand=hand)
    k = ((1 - latitude) / (1 + latitude)) ** 2
    np.testing.assert_allclose(
        x + 1j * y, np.exp(-4j * hand * lam) * (w + k) / (1 + k)
    )
    columns = []
    for delta in (1e-6, 1e-6j):
        plus = forward(
            (w + delta).real, (w + delta).imag, latitude, lam, hand=hand
        )[0]
        minus = forward(
            (w - delta).real, (w - delta).imag, latitude, lam, hand=hand
        )[0]
        columns.append((np.array(plus) - minus) / 2e-6)
    np.testing.assert_allclose(
        np.linalg.slogdet(np.array(columns).T)[1], jf, atol=1e-9
    )


@pytest.mark.parametrize("latitude", [-0.01, 1.01, np.nan, np.inf])
@pytest.mark.parametrize("transform", [forward, inverse])
def test_invalid_latitude(latitude, transform) -> None:
    with pytest.raises(ValueError, match="hemisphere reflection"):
        transform(0.0, 0.0, latitude, 0.0, hand=1)


@pytest.mark.parametrize("hand", [0, 2, 1.0, np.nan])
def test_invalid_hand(hand) -> None:
    with pytest.raises(ValueError, match="hand"):
        forward(0, 0, 0.5, 0, hand=hand)


def test_inverse_forward_arbitrary_values_and_lists() -> None:
    v = np.random.default_rng(53).normal(size=(2, 19))
    w, _ = inverse(v[0], v[1], 0.0, 2 * np.pi - 1e-7, hand=-1)
    restored, _ = forward(*w, 0.0, 2 * np.pi - 1e-7, hand=-1)
    np.testing.assert_allclose(restored, v, atol=1e-14)
    values, jac = forward([0.0, 0.25], [0.0, -0.5], [0.0, 1.0], 0.0, hand=1)
    np.testing.assert_allclose(values, [[0.5, 0.25], [0.0, -0.5]])
    np.testing.assert_allclose(jac, [-2 * np.log(2), 0])
