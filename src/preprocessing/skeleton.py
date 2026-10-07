from __future__ import annotations
import numpy as np

def validate_skeleton_sequence(skeletons) -> np.ndarray:
    """Return training-compatible 3D skeleton array.

    The current dataset builders did not apply additional root-centering, scaling,
    or coordinate standardization after VideoPose3D. Therefore V1 only validates
    shape and dtype and preserves the generated coordinates.
    """
    x = np.asarray(skeletons, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (17,3):
        raise ValueError(f"3D skeleton expected (N,17,3), got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("3D skeleton contains NaN/Inf")
    return x
