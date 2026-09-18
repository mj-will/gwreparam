"""Validate metadata and install each distribution in a fresh environment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

INSTALLATION_CHECK = """
import importlib.util
import numpy as np
from gwreparam.amplitude import circular_amplitudes, inverse_circular_amplitudes
from gwreparam.lisa.coordinates import helicity_coordinates
from gwreparam.phase import delta_phase

for name in ("sequince", "nessai", "bilby", "bbhx"):
    assert importlib.util.find_spec(name) is None, name
coordinates, jacobian = circular_amplitudes(0.3, 2.0, distance_scale=1.0)
restored, inverse_jacobian = inverse_circular_amplitudes(
    *coordinates, distance_scale=1.0
)
np.testing.assert_allclose(restored, (0.3, 2.0))
np.testing.assert_allclose(jacobian + inverse_jacobian, 0, atol=1e-14)
assert delta_phase(1.0, 0.2, 0.3)[0] == 1.2
values, jacobian, labels = helicity_coordinates(1.0, 0.3, 0.2)
assert len(values) == 3 and set(labels) == {"hand", "psi", "phase"}

"""


def main() -> None:
    """Check one wheel and one sdist produced by the Pixi build task."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=root / "dist")
    directory = parser.parse_args().dist_dir
    distributions = []
    for pattern in ("*.whl", "*.tar.gz"):
        matches = list(directory.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one {pattern} in {directory}, found {len(matches)}. "
                "Remove stale distributions and rebuild."
            )
        distributions.append(matches[0])
    subprocess.run(
        [sys.executable, "-m", "twine", "check", "--strict", *distributions],
        check=True,
    )
    for distribution in distributions:
        with tempfile.TemporaryDirectory(prefix="gwreparam-install-") as path:
            workdir = Path(path)
            environment = workdir / "environment"
            venv.EnvBuilder(with_pip=True).create(environment)
            python = environment / (
                "Scripts/python.exe" if os.name == "nt" else "bin/python"
            )
            subprocess.run(
                [python, "-m", "pip", "install", str(distribution)],
                cwd=workdir,
                check=True,
            )
            subprocess.run(
                [python, "-I", "-c", INSTALLATION_CHECK],
                cwd=workdir,
                check=True,
            )
        print(f"Clean installation passed: {distribution.name}", flush=True)


if __name__ == "__main__":
    main()
