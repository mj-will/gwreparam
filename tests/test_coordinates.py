"""Independent density, singularity and Jacobian checks."""

import numpy as np
import pytest

from gwreparam.amplitude import CircularAmplitude
from gwreparam.lisa.coordinates import RelativePolarization
from gwreparam.phase import DeltaPhase, delta_phase
from gwreparam.utils import UnclippedLogitMapping


@pytest.mark.parametrize("orientation", ["W", "Z"])
@pytest.mark.parametrize("representation", ["cartesian", "polar-area"])
def test_prior_measure_and_branch_multiplicity(
    relative_case, orientation, representation
) -> None:
    sym, x = relative_case
    chart = RelativePolarization(
        sym, orientation=orientation, representation=representation
    )
    y, jf, labels = chart.forward(x)
    area = (
        y[:, 3] ** 2 + y[:, 4] ** 2
        if representation == "cartesian"
        else y[:, 3]
    )
    cdf = 2 * area**0.25 / (1 + area**0.25) if orientation == "W" else area
    # Deterministic empirical CDF bound for isotropic cosine inclination.
    assert (
        np.max(abs(np.sort(cdf) - np.arange(1, len(x) + 1) / len(x))) < 0.015
    )
    counts = np.bincount(
        ((labels["hand"] > 0) * 4 + 2 * labels["psi"] + labels["phase"]),
        minlength=8,
    )
    assert np.max(abs(counts / len(x) - 1 / 8)) < 0.012
    # Integrate the physical orientation prior in transformed coordinates,
    # including all eight orientation branches. Gauss-Legendre nodes avoid
    # both singular endpoints; k=t^2 keeps W's integrand smooth.
    nodes, weights = np.polynomial.legendre.leggauss(80)
    t = (nodes + 1) / 2
    u = (1 - t) / (1 + t) if orientation == "W" else 1 - t**2
    physical = np.tile(x[0], (len(t), 1))
    physical[:, 3] = u
    jac = np.exp(chart.forward(physical)[1])
    measure = 2 * t**3 if orientation == "W" else t
    if representation == "polar-area":
        measure *= 2
    mass = 8 * np.sum(weights / 2 * measure / jac)
    np.testing.assert_allclose(mass, 1, atol=2e-13)
    restored, ji = chart.inverse(y, auxiliary=labels)
    np.testing.assert_allclose(jf + ji, 0, atol=1e-10)
    np.testing.assert_allclose(restored[:, 3], x[:, 3], atol=1e-12)


@pytest.mark.parametrize("orientation", ["W", "Z"])
@pytest.mark.parametrize("representation", ["cartesian", "polar-area"])
def test_relative_rejection(
    relative_case, orientation, representation
) -> None:
    sym, x = relative_case
    chart = RelativePolarization(
        sym, orientation=orientation, representation=representation
    )
    for c in [0, -1, 1, np.nan]:
        point = x[:1].copy()
        point[:, 3] = c
        with pytest.raises(ValueError):
            chart.forward(point)
    y, _, labels = chart.forward(x[:5])
    y[:, 3] = [0, 1, 1.1, np.inf, np.nan]
    y[:, 4] = 0
    back, jac = chart.inverse(y, auxiliary=labels)
    assert np.isfinite(back).all()
    assert np.all(back[:, 3] == 2)
    assert np.isneginf(jac).all()
    labels["hand"][0] = 0
    with pytest.raises(ValueError, match="hand"):
        chart.inverse(y, auxiliary=labels)


@pytest.mark.parametrize("scale", [1, 3.085677581491367e25])
def test_amplitude_jacobian_and_rejection(scale) -> None:
    chart = CircularAmplitude(0, 1, scale)
    x = np.array([[0.3, 2.5 * scale]])
    y, jf, _ = chart.forward(x)
    back, ji = chart.inverse(y)
    np.testing.assert_allclose(back, x)
    np.testing.assert_allclose(jf + ji, 0, atol=1e-13)
    jac = np.empty((2, 2))
    for i, step in enumerate([1e-6, scale * 1e-6]):
        dx = np.zeros_like(x)
        dx[:, i] = step
        jac[:, i] = (chart.forward(x + dx)[0] - chart.forward(x - dx)[0])[
            0
        ] / (2 * step)
    np.testing.assert_allclose(np.linalg.slogdet(jac)[1], jf[0], atol=1e-8)
    back, ji = chart.inverse(
        np.array([[0, 1], [-1, 1], [np.nan, 1], [np.inf, 1]])
    )
    assert np.isfinite(back).all()
    assert np.all(back[:, 1] < 0)
    assert np.isneginf(ji).all()
    _, jf, _ = chart.forward(np.array([[1, scale], [-1, scale]]))
    assert np.isneginf(jf).all()


@pytest.mark.parametrize("sign", [-1, 1])
def test_delta_conventions_and_edge_on(sign) -> None:
    x = np.array([[6, 2, 0], [6, 2, -0.5], [6, 2, 0.5]])
    chart = DeltaPhase(0, 1, 2, sign=sign, inclination_mode="cos")
    y, jf, aux = chart.forward(x)
    back, ji = chart.inverse(y)
    assert aux is None
    np.testing.assert_allclose(back, x)
    np.testing.assert_array_equal(jf + ji, 0)
    assert y[0, 0] == 6
    unwrapped, _ = delta_phase(6, 2, 0.5, wrap=False)
    assert unwrapped == 8


def test_unclipped_radial_map() -> None:
    mapping = UnclippedLogitMapping()
    x = np.array([1e-30, 1e-12, 0.2, 1 - 1e-12])
    y, jf = mapping.forward(x)
    back, ji = mapping.inverse(y)
    np.testing.assert_allclose(back, x, rtol=1e-14)
    np.testing.assert_allclose(jf + ji, 0, atol=1e-14)
    with pytest.raises(ValueError):
        mapping.forward(np.array([0, 1]))
    radial, jac = mapping.inverse(np.array([-np.inf, np.inf, -1000, 1000]))
    np.testing.assert_array_equal(radial, [0, 1, 0, 1])
    assert np.isneginf(jac[:2]).all()


def test_radial_saturation_matches_original_expit() -> None:
    radial, _ = UnclippedLogitMapping().inverse(np.array([-710.0, 36.0, 37.0]))
    assert radial[0] == 0.0
    assert radial[1] < 1.0
    assert radial[2] == 1.0
