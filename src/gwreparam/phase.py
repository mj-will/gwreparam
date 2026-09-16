"""Orbital/delta-phase coordinate conversion (radians)."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def delta_phase(
    phase: ArrayLike,
    polarization: ArrayLike,
    cos_inclination: ArrayLike,
    *,
    sign: int = 1,
    inverse: bool = False,
    wrap: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert orbital and delta phase using an explicit inclination cosine.

    Parameters
    ----------
    phase, polarization, cos_inclination : array_like
        Broadcastable real coordinates. Angles are radians; inclination is
        supplied as its cosine in [-1, 1]. Nonfinite inputs propagate.
    sign : {-1, 1}, optional
        Convention in delta = phase + sign*sign(cos_inclination)*polarization.
    inverse : bool, optional
        Reverse the shear, interpreting phase as delta phase.
    wrap : bool, optional
        Reduce the result to [0, 2*pi). False retains unwrapped values.

    Returns
    -------
    phase : ndarray
        Converted phase with the broadcast shape, including () for scalars.
    log_jacobian : ndarray
        Zero log-absolute determinant in the evaluated direction, same shape.

    Notes
    -----
    Zero cosine leaves phase unchanged. This is a coordinate shear away from
    the inclination sign boundary, not a symmetry fold. No labels are needed.
    """
    if isinstance(sign, (bool, np.bool_)) or sign not in (-1, 1):
        raise ValueError("sign must be +1 or -1.")
    result = np.asarray(phase, dtype=float) + (
        -1 if inverse else 1
    ) * sign * np.sign(cos_inclination) * np.asarray(polarization)
    if wrap:
        result = np.mod(result, 2 * np.pi)
    result = np.asarray(result, dtype=float)
    return result, np.zeros_like(result)
