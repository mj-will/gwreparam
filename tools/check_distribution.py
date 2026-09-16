"""Validate metadata and install each distribution in a fresh environment."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

INSTALLATION_CHECK = """
import importlib.util
import numpy as np
from gwreparam.amplitude import CircularAmplitude
from gwreparam.lisa.coordinates import RelativePolarization
from gwreparam.phase import DeltaPhase

for name in ("sequince", "nessai", "bilby", "bbhx"):
    assert importlib.util.find_spec(name) is None, name
mapping = CircularAmplitude(0, 1, distance_scale=1.0)
x = np.array([[0.3, 2.0]])
y, jacobian, _ = mapping.forward(x)
restored, inverse_jacobian = mapping.inverse(y)
np.testing.assert_allclose(restored, x)
np.testing.assert_allclose(jacobian + inverse_jacobian, 0, atol=1e-14)
"""


def main() -> None:
    """Check one wheel and one sdist produced by the Pixi build task."""
    root = Path(__file__).resolve().parents[1]
    distributions = []
    for pattern in ("*.whl", "*.tar.gz"):
        matches = list((root / "dist").glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one {pattern} in dist/, found {len(matches)}. "
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
