"""Deterministic coordinates extracted from MBHB analysis utilities."""

import numpy as np
from numpy.typing import NDArray

from .symmetry import LISAExtrinsicSymmetry


class RelativePolarization:
    """Sky-folded chart with particle-aligned sky, handedness and phase labels.

    Columns retain their positions: phase -> Phi, cos_inc/psi ->
    W_x/W_y (default), Z_x/Z_y, or a_W/theta, a_Z/theta.
    Distance (if present) -> log(distance_scale * A_dom).
    The fixed default distance_scale is one Gpc in metres. No distance is
    accessed when it is absent from sampling_parameters.

    This replaces phase/polarization folding for mutation, even when the
    supplied symmetry also configures that fold for other sampler operations.
    Exact edge-on and face-on points are singular and excluded (zero measure
    under the supported continuous priors); no epsilon regularizes the chart.

    Parameters
    ----------
    symmetry : LISAExtrinsicSymmetry
        Physical column names, bounds and sky-folding convention.
    distance_scale : float, optional
        Positive reference distance in the same units as sampled distance.
    orientation : {"W", "Z"}, optional
        Radial coordinate, default W.
    representation : {"cartesian", "polar-area"}, optional
        Cartesian components or squared radius and angle.
    """

    def __init__(
        self,
        symmetry: LISAExtrinsicSymmetry,
        *,
        distance_scale: float = 3.085677581491367e25,
        orientation: str = "W",
        representation: str = "cartesian",
    ) -> None:
        if orientation not in ("W", "Z"):
            raise ValueError(
                "relative_polarization_orientation must be W or Z."
            )
        if representation not in ("cartesian", "polar-area"):
            raise ValueError(
                "relative_polarization_representation must be cartesian or polar-area."
            )
        self.orientation = orientation
        self.representation = representation
        self.symmetry = symmetry
        self.names = tuple(symmetry.sampling_parameters)
        if symmetry.iota_mode != "cos" or symmetry.beta_mode != "sin":
            raise ValueError(
                "Relative polarization requires cosine inclination and sine latitude."
            )
        if symmetry.phase_index is None:
            raise ValueError("Relative polarization requires a sampled phase.")
        self.ci, self.pi = symmetry.iota_index, symmetry.phase_index
        self.wi, self.li = symmetry.psi_index, symmetry.lambda_index
        self.di = (
            self.names.index("distance") if "distance" in self.names else None
        )
        self.distance_scale = float(distance_scale)
        if not np.isfinite(self.distance_scale) or self.distance_scale <= 0:
            raise ValueError("distance_scale must be finite and positive.")
        self.delta = getattr(symmetry, "phase_transform", None)
        if symmetry.phase_parameter == "delta_phi" and self.delta is None:
            raise ValueError(
                "delta_phi requires a convention-aware LISA symmetry."
            )

    def _log_jacobian(
        self,
        u: NDArray[np.float64],
        distance: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        # |d(Phi, Wx, Wy)/d(phi, c, psi)| = 8*k*|dk/dc|.
        if self.orientation == "W":
            result = np.log(32.0) + 3 * np.log1p(-u) - 5 * np.log1p(u)
        else:
            result = np.full_like(u, np.log(4.0))
        # d(a, theta)/d(x, y) = 2 for a = x*x + y*y.
        if self.representation == "polar-area":
            result = result + np.log(2.0)
        if distance is not None:
            result = result - np.log(distance)
        return result

    def forward(
        self, particles: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.float64], NDArray[np.float64], dict[str, NDArray[np.int64]]
    ]:
        """Return chart rows, forward log Jacobian, and four branch arrays.

        Parameters
        ----------
        particles : ndarray, shape (n, d)
            Finite physical coordinates in the configured column order.
            Angles are radians; 0 < abs(cosine inclination) < 1 is required.

        Returns
        -------
        values : ndarray, shape (n, d)
            Relative coordinates, retaining original column positions.
        log_det : ndarray, shape (n,)
            log|d chart / d physical|, excluding branch probability.
        auxiliary : dict[str, ndarray]
            Integer sky, hand (-1/+1), psi (0/1) and phase (0/1) labels.
        """
        raw = np.array(particles, dtype=float, copy=True)
        if self.delta is not None:
            raw = self.delta.inverse(raw)[0]
        folded, sky = self.symmetry.sky_symmetry.fold(raw)
        c, psi = folded[:, self.ci], folded[:, self.wi]
        phi = folded[:, self.pi]
        lam = folded[:, self.li]
        u, s = np.abs(c), np.sign(c)
        if np.any(~np.isfinite(folded)) or np.any((u == 0) | (u >= 1)):
            raise ValueError(
                "Relative polarization requires finite, non-edge-on, non-face-on particles."
            )
        distance = None if self.di is None else folded[:, self.di].copy()
        if distance is not None and np.any(distance <= 0):
            raise ValueError("Distance must be positive.")
        k = (
            ((1 - u) / (1 + u)) ** 2
            if self.orientation == "W"
            else np.sqrt(1 - u)
        )
        phase = (2 * (phi - s * (psi - lam))) % (2 * np.pi)
        w = k * np.exp(4j * s * (psi - lam))
        # Derive labels from the SAME wrapped inverse used for proposals. This
        # also works on angular seams, where floor(psi / (pi/2)) is fragile.
        theta = (4 * s * (psi - lam)) % (2 * np.pi)
        angle_value = (
            w if self.representation == "cartesian" else np.exp(1j * theta)
        )
        psi0, base = self._angles(phase, angle_value, lam, s)
        psi_branch = np.rint((psi - psi0) / (np.pi / 2)).astype(int) % 2
        phase_branch = (
            np.rint((phi - base - s * psi_branch * np.pi / 2) / np.pi).astype(
                int
            )
            % 2
        )
        result = folded.copy()
        result[:, self.pi] = phase
        if self.representation == "cartesian":
            result[:, self.ci], result[:, self.wi] = w.real, w.imag
        else:
            result[:, self.ci] = k**2 if self.orientation == "W" else 1 - u
            result[:, self.wi] = theta
        if distance is not None:
            result[:, self.di] = (
                np.log(self.distance_scale)
                + 2 * np.log1p(u)
                - np.log(4.0)
                - np.log(distance)
            )
        auxiliary = {
            "sky": sky,
            "hand": s.astype(int),
            "psi": psi_branch,
            "phase": phase_branch,
        }
        return result, self._log_jacobian(u, distance), auxiliary

    @staticmethod
    def _angles(
        phase: NDArray[np.float64],
        w_rotated: NDArray[np.complex128],
        lam: NDArray[np.float64],
        s: NDArray[np.float64] | NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        phi_dom = (phase - 2 * s * lam) % (2 * np.pi)
        w = w_rotated * np.exp(4j * s * lam)
        # Phi_weak = (Phi_dom + arg(W)) mod 2*pi; use the phase
        # difference directly to avoid subtracting two wrapped phases.
        psi = (s * np.angle(w) / 4) % (np.pi / 2)
        return psi, phi_dom / 2 + s * psi

    def inverse(
        self,
        values: NDArray[np.float64],
        *,
        auxiliary: dict[str, NDArray[np.int64]] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return physical rows and the inverse log Jacobian at fixed labels.

        Parameters
        ----------
        values : ndarray, shape (n, d)
            Chart coordinates in the open unit disk or area interval.
        auxiliary : dict[str, ndarray]
            All four branch arrays, each of shape (n,). Bad labels raise.

        Returns
        -------
        physical : ndarray, shape (n, d)
            Restored rows. Rejected proposals have finite placeholders and
            cosine inclination 2, outside the physical prior.
        log_det : ndarray, shape (n,)
            log|d physical / d chart|; -infinity for rejected proposals.
        """
        result = np.array(values, dtype=float, copy=True)
        if not isinstance(auxiliary, dict) or set(auxiliary) != {
            "sky",
            "hand",
            "psi",
            "phase",
        }:
            raise ValueError(
                "Relative polarization inverse requires all branch labels."
            )
        for key, allowed in (
            ("hand", (-1, 1)),
            ("psi", (0, 1)),
            ("phase", (0, 1)),
            ("sky", tuple(range(self.symmetry.n_sky_modes))),
        ):
            label = np.asarray(auxiliary[key])
            if (
                label.shape != (len(result),)
                or not np.isin(label, allowed).all()
            ):
                raise ValueError(f"Invalid {key} branch labels.")
        s = np.asarray(auxiliary["hand"])
        radius = (
            np.hypot(result[:, self.ci], result[:, self.wi])
            if self.representation == "cartesian"
            else result[:, self.ci]
        )
        valid = np.isfinite(result).all(axis=1) & (radius > 0) & (radius < 1)
        # Finite placeholders are only used on rejected rows, never projected
        # onto the physical disk. Mark them outside the inclination prior below.
        safe = np.where(valid[:, None], result, 0.0)
        k = np.where(valid, radius, 0.25)
        if self.orientation == "W":
            root = (
                np.sqrt(k) if self.representation == "cartesian" else k**0.25
            )
            u = (1 - root) / (1 + root)
        else:
            u = 1 - (k**2 if self.representation == "cartesian" else k)
        # Reject values whose recovered inclination rounds to a singularity.
        valid &= (u > 0) & (u < 1)
        u = np.where(valid, u, 0.5)
        w = (
            safe[:, self.ci] + 1j * safe[:, self.wi]
            if self.representation == "cartesian"
            else np.exp(1j * (safe[:, self.wi] % (2 * np.pi)))
        )
        lam = safe[:, self.li]
        psi, base = self._angles(safe[:, self.pi], w, lam, s)
        psi_shift = np.asarray(auxiliary["psi"]) * np.pi / 2
        safe[:, self.wi] = (psi + psi_shift) % np.pi
        safe[:, self.pi] = (
            base + s * psi_shift + np.asarray(auxiliary["phase"]) * np.pi
        ) % (2 * np.pi)
        safe[:, self.ci] = s * u
        distance = None
        if self.di is not None:
            with np.errstate(over="ignore", under="ignore"):
                distance = np.exp(
                    np.log(self.distance_scale)
                    + 2 * np.log1p(u)
                    - np.log(4.0)
                    - safe[:, self.di]
                )
            valid &= np.isfinite(distance) & (distance > 0)
            distance = np.where(valid, distance, self.distance_scale)
            safe[:, self.di] = distance
        jac = -self._log_jacobian(u, distance)
        physical = self.symmetry.sky_symmetry.unfold(safe, auxiliary["sky"])
        if self.delta is not None:
            physical = self.delta.forward(physical)[0]
        physical[~valid, self.ci] = 2.0
        return physical, np.where(valid, jac, -np.inf)


class RelativeHandedness:
    """Two representatives at fixed relative coordinates, with unit absolute Jacobian.

    This is a proposal, not a waveform invariance: the existing mode MH step
    evaluates the full physical target before accepting the other handedness.
    """

    n_modes = 2

    def __init__(self, symmetry: LISAExtrinsicSymmetry) -> None:
        self.chart = RelativePolarization(symmetry)

    def determine_mode(
        self, particles: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        return self.fold(particles)[1]

    def fold(
        self, particles: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        values, _, labels = self.chart.forward(particles)
        mode = (labels["hand"] > 0).astype(int)
        labels["hand"] = np.ones(len(values), dtype=int)
        return self.chart.inverse(values, auxiliary=labels)[0], mode

    def unfold(
        self, reduced: NDArray[np.float64], mode: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        values, _, labels = self.chart.forward(reduced)
        labels["hand"] = 2 * np.asarray(mode, dtype=int) - 1
        return self.chart.inverse(values, auxiliary=labels)[0]

    def forward(
        self, particles: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        """Return folded rows, log|d folded/d full|, and branch labels."""
        values, labels = self.fold(particles)
        return values, np.zeros(len(values)), labels

    def inverse(
        self,
        values: NDArray[np.float64],
        *,
        auxiliary: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return unfolded rows and log|d full/d folded| at fixed labels."""
        full = self.unfold(values, auxiliary)
        return full, np.zeros(len(full))
