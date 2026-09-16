"""Coordinate contracts and independent numerical invariants."""

from itertools import product

import numpy as np
import pytest

from gwreparam.amplitude import (
    circular_amplitudes,
    inverse_circular_amplitudes,
)
from gwreparam.lisa.coordinates import (
    inverse_relative_polarization,
    relative_polarization,
)
from gwreparam.lisa.symmetry import fold_sky, unfold_sky
from gwreparam.phase import delta_phase
from gwreparam.symmetry import (
    fold_delta_phase_polarization,
    fold_periodic,
    fold_phase_polarization,
    unfold_delta_phase_polarization,
    unfold_periodic,
    unfold_phase_polarization,
)


def numerical_log_jacobian(function, point, steps):
    columns = []
    for i, step in enumerate(steps):
        change = np.zeros_like(point)
        change[i] = step
        columns.append(
            (
                np.asarray(function(*(point + change))[0])
                - np.asarray(function(*(point - change))[0])
            )
            / (2 * step)
        )
    return np.linalg.slogdet(np.column_stack(columns))[1]


@pytest.mark.parametrize("shape", [(), (5,), (2, 3)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_delta_broadcast(shape, sign):
    c = np.full(shape, -0.4)
    out, jac = delta_phase(6.0, 2.0, c, sign=sign)
    back, inv = delta_phase(out, 2.0, c, sign=sign, inverse=True)
    assert out.shape == jac.shape == shape
    np.testing.assert_allclose(back, 6.0)
    np.testing.assert_array_equal(jac + inv, 0.0)
    assert delta_phase(6.0, 2.0, 0.0)[0] == 6.0
    assert delta_phase(6.0, 2.0, 1.0, wrap=False)[0] == 8.0
    assert np.isnan(delta_phase(np.nan, 0.0, 1.0)[0])


@pytest.mark.parametrize("scale", [1.0, 3.085677581491367e25])
def test_amplitudes(scale):
    c, d = np.array([[-0.4], [0.3]]), scale * np.array([1.0, 2.0, 3.0])
    coords, jac = circular_amplitudes(c, d, distance_scale=scale)
    back, inv = inverse_circular_amplitudes(*coords, distance_scale=scale)
    assert jac.shape == (2, 3)
    np.testing.assert_allclose(back[0], np.broadcast_to(c, (2, 3)))
    np.testing.assert_allclose(back[1], np.broadcast_to(d, (2, 3)))
    np.testing.assert_allclose(jac + inv, 0.0, atol=4e-14)

    def fn(c, d):
        return circular_amplitudes(c, d, distance_scale=scale)

    point = np.array([0.3, 2.5 * scale])
    np.testing.assert_allclose(
        numerical_log_jacobian(fn, point, [1e-6, scale * 1e-6]),
        fn(*point)[1],
        atol=1e-8,
    )
    bad, inv = inverse_circular_amplitudes(
        [0.0, -1.0, np.inf, np.nan], 1.0, distance_scale=scale
    )
    np.testing.assert_array_equal(bad[0], 0.5)
    np.testing.assert_array_equal(bad[1], -scale)
    assert np.isneginf(inv).all()
    assert np.isneginf(fn(1.0, scale)[1])


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("origin", [0.0, -np.pi])
def test_periodic_seams(count, origin):
    values = origin + np.array(
        [-1e-12, 0.0, 1e-12, np.pi / 2, np.pi, 2 * np.pi - 1e-12, 2 * np.pi]
    )
    y, jac, labels = fold_periodic(values, 2 * np.pi, count, origin=origin)
    back, inv = unfold_periodic(y, 2 * np.pi, count, labels, origin=origin)
    np.testing.assert_allclose(
        (back - values + np.pi) % (2 * np.pi) - np.pi, 0.0, atol=2e-15
    )
    np.testing.assert_array_equal(jac + inv, 0.0)
    assert ((labels >= 0) & (labels < count)).all()
    assert fold_periodic(0.3, 2 * np.pi, count)[0].shape == ()


@pytest.mark.parametrize("delta", [False, True])
def test_phase_folds(delta):
    phase = np.array([[0.0], [np.pi / 2], [np.pi], [2 * np.pi - 1e-12]])
    psi = np.array([0.0, np.pi / 2, np.pi - 1e-12])
    fold = fold_delta_phase_polarization if delta else fold_phase_polarization
    unfold = (
        unfold_delta_phase_polarization if delta else unfold_phase_polarization
    )
    extra = (0.3,) if delta else ()
    values, jac, labels = fold(phase, psi, *extra)
    back, inv = unfold(*values, *extra, labels)
    assert jac.shape == (4, 3)
    np.testing.assert_allclose(back[0], np.broadcast_to(phase, (4, 3)))
    np.testing.assert_allclose(back[1], np.broadcast_to(psi, (4, 3)))
    np.testing.assert_array_equal(jac + inv, 0.0)
    if delta:
        with pytest.raises(ValueError, match="edge-on"):
            fold(0.0, 0.0, 0.0)


@pytest.mark.parametrize("group", ["reflected-antipodal", "8-mode"])
@pytest.mark.parametrize(
    "latitude_mode,inclination_mode",
    list(product(["sin", "angle"], ["cos", "angle"])),
)
def test_sky(group, latitude_mode, inclination_mode):
    lon = np.array([[0.0], [np.pi / 2], [np.pi], [2 * np.pi - 1e-12]])
    lat = np.array([-0.3, 0.0, 0.3])
    inc = 0.4
    options = dict(
        symmetry_group=group,
        latitude_mode=latitude_mode,
        inclination_mode=inclination_mode,
    )
    values, jac, labels = fold_sky(lon, lat, 0.0, inc, **options)
    back, inv = unfold_sky(*values, labels, **options)
    for actual, expected in zip(back, np.broadcast_arrays(lon, lat, 0.0, inc)):
        np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_array_equal(jac + inv, 0.0)

    def fn(*x):
        return fold_sky(*x, **options)

    np.testing.assert_allclose(
        numerical_log_jacobian(
            fn, np.array([3.7, -0.3, 0.7, 0.4]), [1e-6] * 4
        ),
        0.0,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "orientation,representation,frame,distance",
    list(
        product(
            ["W", "Z"],
            ["cartesian", "polar-area"],
            ["sky-rotated", "unrotated"],
            [False, True],
        )
    ),
)
def test_relative(orientation, representation, frame, distance):
    options = dict(
        orientation=orientation,
        representation=representation,
        polarization_frame=frame,
        distance_scale=2.0,
    )
    phi = np.array([[0.0], [np.pi], [2 * np.pi - 1e-12]])
    c = np.array([-0.8, 0.3])
    psi = np.pi / 2
    lam = 0.4
    coords, jac, branches = relative_polarization(
        phi, c, psi, lam, distance=3.0 if distance else None, **options
    )
    back, inv = inverse_relative_polarization(
        *coords[:3],
        lam,
        hand=branches["hand"],
        psi_branch=branches["psi"],
        phase_branch=branches["phase"],
        log_amplitude=coords[3] if distance else None,
        **options,
    )
    assert jac.shape == (3, 2)
    for actual, expected, period in zip(
        back[:3], np.broadcast_arrays(phi, c, psi), [2 * np.pi, None, np.pi]
    ):
        diff = actual - expected
        if period:
            diff = (diff + period / 2) % period - period / 2
        np.testing.assert_allclose(diff, 0.0, atol=1e-12)
    np.testing.assert_allclose(jac + inv, 0.0, atol=1e-12)
    if distance:
        np.testing.assert_allclose(back[3], 3.0)
    point = np.array([1.2, 0.3, 0.7] + ([3.0] if distance else []))

    def fn(phi, c, psi, *d):
        return relative_polarization(
            phi, c, psi, lam, distance=d[0] if d else None, **options
        )

    np.testing.assert_allclose(
        numerical_log_jacobian(fn, point, [1e-6] * len(point)),
        fn(*point)[1],
        atol=1e-8,
    )
    assert fn(*point)[1].shape == ()
    bad, inv = inverse_relative_polarization(
        1.0,
        [0.0, 1.0, 1.1, np.inf, np.nan],
        0.0,
        hand=1,
        psi_branch=0,
        phase_branch=0,
        **options,
    )
    assert all(np.isfinite(v).all() for v in bad)
    np.testing.assert_array_equal(bad[1], 2.0)
    assert np.isneginf(inv).all()


@pytest.mark.parametrize("labels", [-1, 4, 1.2, True, np.nan])
def test_invalid_labels(labels):
    with pytest.raises(ValueError):
        unfold_periodic(0.1, 2 * np.pi, 4, labels)
    with pytest.raises(ValueError):
        unfold_phase_polarization(0.1, 0.1, labels)
    with pytest.raises(ValueError):
        unfold_sky(0.1, 0.1, 0.1, 0.1, labels)


@pytest.mark.parametrize("c", [0.0, -1.0, 1.0, np.nan])
def test_relative_invalid_physical(c):
    with pytest.raises(ValueError):
        relative_polarization(1.0, c, 0.3)


def test_relative_distance_optional_and_broadcast():
    a, _, _ = relative_polarization(1.0, 0.3, 0.2)
    b, jac, _ = relative_polarization(1.0, 0.3, 0.2, distance=np.ones((2, 3)))
    assert len(a) == 3 and len(b) == 4 and jac.shape == (2, 3)
    with pytest.raises(ValueError):
        relative_polarization(1.0, 0.3, 0.2, distance=-1.0)
    with pytest.raises(ValueError):
        inverse_relative_polarization(
            1.0, 0.2, 0.2, hand=0, psi_branch=0, phase_branch=0
        )


@pytest.mark.parametrize(
    "fold", [fold_phase_polarization, fold_delta_phase_polarization]
)
def test_phase_fold_numerical_jacobian(fold):
    def fn(phase, psi):
        return fold(
            phase,
            psi,
            *([0.4] if fold is fold_delta_phase_polarization else []),
        )

    np.testing.assert_allclose(
        numerical_log_jacobian(fn, np.array([4.2, 2.1]), [1e-6] * 2),
        0.0,
        atol=1e-8,
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_delta_numerical_jacobian(sign):
    def fn(phase, psi, cosine):
        delta, jac = delta_phase(phase, psi, cosine, sign=sign)
        return (delta, psi, cosine), jac

    np.testing.assert_allclose(
        numerical_log_jacobian(fn, np.array([3.0, 0.4, -0.3]), [1e-6] * 3),
        0.0,
        atol=1e-8,
    )


def test_scalar_array_contract():
    outputs = [
        delta_phase(0.3, 0.2, 0.4),
        circular_amplitudes(0.4, 2.0, distance_scale=1.0),
        fold_periodic(0.3, 2 * np.pi, 4),
        fold_phase_polarization(0.3, 0.2),
        fold_delta_phase_polarization(0.3, 0.2, 0.4),
        fold_sky(0.3, 0.2, 0.4, 0.5),
        relative_polarization(0.3, 0.4, 0.2),
    ]
    for result in outputs:
        coordinates, jac, *labels = result
        for value in (
            coordinates if isinstance(coordinates, tuple) else (coordinates,)
        ):
            assert isinstance(value, np.ndarray) and value.shape == ()
        assert isinstance(jac, np.ndarray) and jac.shape == ()
        if labels:
            for value in (
                labels[0].values() if isinstance(labels[0], dict) else labels
            ):
                assert isinstance(value, np.ndarray) and value.shape == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"period": 0.0, "n_modes": 2},
        {"period": np.inf, "n_modes": 2},
        {"period": 1.0, "n_modes": 2.5},
        {"period": 1.0, "n_modes": True},
    ],
)
def test_periodic_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        fold_periodic(0.2, **kwargs)


def test_inverse_label_broadcast_and_nonmutation():
    values, jac, labels = relative_polarization(1.2, 0.3, 0.7, 0.4)
    originals = [v.copy() for v in values]
    back, inv = inverse_relative_polarization(
        *values,
        0.4,
        hand=np.array([[-1], [1]]),
        psi_branch=np.array([0, 1]),
        phase_branch=0,
    )
    assert inv.shape == (2, 2)
    repeated, forward, branches = relative_polarization(*back, 0.4)
    for before, after, expected in zip(values, originals, repeated):
        np.testing.assert_array_equal(before, after)
        np.testing.assert_allclose(expected, before, atol=2e-14)
    np.testing.assert_allclose(forward + inv, 0.0, atol=2e-14)
    np.testing.assert_array_equal(branches["hand"], [[-1, -1], [1, 1]])


@pytest.mark.parametrize("log_amplitude", [np.inf, np.nan, 1000.0, -1000.0])
def test_relative_invalid_distance_proposal(log_amplitude):
    values, jac = inverse_relative_polarization(
        1.0,
        0.2,
        0.1,
        hand=1,
        psi_branch=0,
        phase_branch=0,
        log_amplitude=log_amplitude,
    )
    assert all(np.isfinite(v).all() for v in values)
    assert values[1] == 2 and np.isneginf(jac)
