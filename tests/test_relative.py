"""Numerical invariants extracted from MBHB's relative-coordinate tests."""

import numpy as np
import pytest

from gwreparam.lisa.coordinates import (
    RelativePolarization as RelativePolarizationTransform,
)


def assert_angles_equal(x, y, sym, atol=2e-12):
    diff = y - x
    for i, period in (
        (sym.phase_index, 2 * np.pi),
        (sym.psi_index, np.pi),
        (sym.lambda_index, 2 * np.pi),
    ):
        diff[:, i] = (diff[:, i] + period / 2) % period - period / 2
    np.testing.assert_allclose(diff, 0, atol=atol)


@pytest.mark.parametrize("polarization_frame", ["sky-rotated", "unrotated"])
@pytest.mark.parametrize("distance", [False, True])
@pytest.mark.parametrize(
    ("delta", "sign"), [(False, -1), (True, -1), (True, 1)]
)
@pytest.mark.parametrize("group", ["reflected-antipodal", "8-mode"])
@pytest.mark.parametrize("phase_fold", [False, True])
def test_roundtrip_and_branches(
    relative_case_factory,
    polarization_frame,
    distance,
    delta,
    sign,
    group,
    phase_fold,
    chart_options,
) -> None:
    sym, _, x = relative_case_factory(distance, delta, sign, group, phase_fold)
    chart = RelativePolarizationTransform(
        sym,
        distance_scale=2.0,
        polarization_frame=polarization_frame,
        **chart_options,
    )
    y, jf, aux = chart.forward(x)
    restored, ji = chart.inverse(y, auxiliary=aux)
    assert_angles_equal(x, restored, sym)
    np.testing.assert_allclose(jf + ji, 0, atol=3e-12)
    assert set(aux["hand"]) == {-1, 1}
    assert set(aux["sky"]) == set(range(sym.n_sky_modes))
    assert set(zip(aux["psi"], aux["phase"], strict=False)) == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    chosen = np.array([7, 2, 11])
    back, _ = chart.inverse(
        y[chosen], auxiliary={k: v[chosen] for k, v in aux.items()}
    )
    assert_angles_equal(x[chosen], back, sym)
    # Check the requested formula independently in raw, sky-folded coordinates.
    raw = sym.phase_transform.inverse(x)[0] if delta else x
    f, _ = sym.sky_symmetry.fold(raw)
    s = np.sign(f[:, sym.iota_index])
    u = abs(f[:, sym.iota_index])
    radius = (
        ((1 - u) / (1 + u)) ** 2
        if chart.orientation == "W"
        else np.sqrt(1 - u)
    )
    offset = (
        f[:, sym.lambda_index] if polarization_frame == "sky-rotated" else 0
    )
    theta = (4 * s * (f[:, sym.psi_index] - offset)) % (2 * np.pi)
    if chart.representation == "cartesian":
        z = radius * np.exp(1j * theta)
        np.testing.assert_allclose(
            y[:, sym.iota_index] + 1j * y[:, sym.psi_index], z, atol=1e-14
        )
    else:
        np.testing.assert_allclose(y[:, sym.iota_index], radius**2)
        np.testing.assert_allclose(y[:, sym.psi_index], theta)
    for mode in range(sym.n_modes):
        folded, _ = sym.fold(x)
        moved = sym.unfold(folded, np.full(len(x), mode))
        v, _, a = chart.forward(moved)
        assert_angles_equal(moved, chart.inverse(v, auxiliary=a)[0], sym)


@pytest.mark.parametrize("polarization_frame", ["sky-rotated", "unrotated"])
@pytest.mark.parametrize("distance", [False, True])
@pytest.mark.parametrize("sign", [-1, 1])
def test_numerical_jacobian(
    relative_case_factory, polarization_frame, distance, sign, chart_options
) -> None:
    sym, _, x = relative_case_factory(distance, sign=sign)
    chart = RelativePolarizationTransform(
        sym,
        distance_scale=2.0,
        polarization_frame=polarization_frame,
        **chart_options,
    )
    for row in x[:10]:
        _, expected, _ = chart.forward(row[None])
        jac = np.empty((len(row), len(row)))
        for j in range(len(row)):
            d = np.zeros_like(row)
            d[j] = 1e-6
            jac[:, j] = (
                chart.forward((row + d)[None])[0][0]
                - chart.forward((row - d)[None])[0][0]
            ) / (2e-6)
        np.testing.assert_allclose(
            np.linalg.slogdet(jac)[1], expected[0], atol=2e-7
        )


@pytest.mark.parametrize("sign", [-1, 1])
def test_handedness_proposal_is_involution_and_unit_jacobian(
    relative_case_factory, sign
) -> None:
    from gwreparam.lisa.coordinates import (
        RelativeHandedness as RelativeHandednessProposal,
    )

    sym, _, x = relative_case_factory(sign=sign)
    proposal = RelativeHandednessProposal(sym)
    reduced, modes = proposal.fold(x)
    assert_angles_equal(x, proposal.unfold(reduced, modes), sym)
    flipped = proposal.unfold(reduced, 1 - modes)
    y, j, aux = proposal.chart.forward(x)
    yf, jf, af = proposal.chart.forward(flipped)
    np.testing.assert_allclose(y, yf, atol=1e-12)
    np.testing.assert_allclose(j, jf, atol=1e-12)
    np.testing.assert_array_equal(aux["hand"], -af["hand"])
    row = x[:1].copy()

    def flip(p):
        r, m = proposal.fold(p)
        return proposal.unfold(r, 1 - m)

    jac = np.empty((row.shape[1], row.shape[1]))
    for col in range(row.shape[1]):
        d = np.zeros_like(row)
        d[:, col] = 1e-6
        jac[:, col] = (flip(row + d) - flip(row - d))[0] / 2e-6
    np.testing.assert_allclose(np.linalg.slogdet(jac)[1], 0, atol=1e-7)


def test_unrotated_frame_relation_and_legacy_checkpoint(
    relative_case_factory, chart_options
) -> None:
    import pickle

    sym, _, x = relative_case_factory(delta=False)
    rotated = RelativePolarizationTransform(sym, **chart_options)
    ordinary = RelativePolarizationTransform(
        sym, polarization_frame="unrotated", **chart_options
    )
    y, jac, labels = rotated.forward(x)
    z, ordinary_jac, ordinary_labels = ordinary.forward(x)
    sky, _ = sym.sky_symmetry.fold(x)
    s = np.sign(sky[:, sym.iota_index])
    lam = sky[:, sym.lambda_index]
    np.testing.assert_allclose(
        np.exp(1j * y[:, sym.phase_index]),
        np.exp(1j * (z[:, sym.phase_index] + 2 * s * lam)),
        atol=2e-14,
    )
    if chart_options["representation"] == "cartesian":
        w_rot = y[:, sym.iota_index] + 1j * y[:, sym.psi_index]
        w = z[:, sym.iota_index] + 1j * z[:, sym.psi_index]
        np.testing.assert_allclose(
            w_rot, w * np.exp(-4j * s * lam), atol=2e-14
        )
    else:
        np.testing.assert_allclose(y[:, sym.iota_index], z[:, sym.iota_index])
        np.testing.assert_allclose(
            np.exp(1j * y[:, sym.psi_index]),
            np.exp(1j * (z[:, sym.psi_index] - 4 * s * lam)),
            atol=2e-14,
        )
    np.testing.assert_array_equal(jac, ordinary_jac)
    # Existing serialized charts do not contain the new convention field.
    del rotated.polarization_frame
    legacy = pickle.loads(pickle.dumps(rotated))
    np.testing.assert_array_equal(legacy.forward(x)[0], y)
    assert_angles_equal(x, legacy.inverse(y, auxiliary=labels)[0], sym)
    ordinary = pickle.loads(pickle.dumps(ordinary))
    assert_angles_equal(
        x, ordinary.inverse(z, auxiliary=ordinary_labels)[0], sym
    )


def test_unrotated_seams_and_rejection(
    relative_case_factory, chart_options
) -> None:
    sym, _, x = relative_case_factory(delta=False)
    chart = RelativePolarizationTransform(
        sym, polarization_frame="unrotated", **chart_options
    )
    x = x[:8].copy()
    x[:, sym.psi_index] = np.arange(8) * np.pi / 8
    x[:, sym.phase_index] = np.arange(8) * np.pi / 4
    x[:, sym.iota_index] = np.tile([-0.4, 0.4], 4)
    y, jac, labels = chart.forward(x)
    restored, inverse = chart.inverse(y, auxiliary=labels)
    assert_angles_equal(x, restored, sym)
    np.testing.assert_allclose(jac + inverse, 0, atol=1e-13)
    y[:, sym.iota_index] = 2
    physical, inverse = chart.inverse(y, auxiliary=labels)
    assert np.isfinite(physical).all()
    assert np.isneginf(inverse).all()
    with pytest.raises(ValueError, match="polarization_frame"):
        RelativePolarizationTransform(sym, polarization_frame="unknown")
