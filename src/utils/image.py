from __future__ import annotations

import numpy as np


def preprocess_intensity(voxel: np.ndarray, clip_std: float = 3.0) -> np.ndarray:
    """Clip positive-foreground outliers, z-score, then map to [-1, 1]."""
    voxel = np.asarray(voxel, dtype=np.float32)
    nonzero = voxel[voxel > 0]
    if nonzero.size:
        upper = float(nonzero.mean() + clip_std * nonzero.std())
        voxel = np.clip(voxel, 0.0, upper)
        mu = float(nonzero.mean())
        sigma = float(nonzero.std())
        sigma = sigma if sigma > 1e-6 else 1.0
        voxel = (voxel - mu) / sigma
    vmin = float(voxel.min())
    vmax = float(voxel.max())
    if vmax > vmin:
        voxel = (voxel - vmin) / (vmax - vmin)
    else:
        voxel = np.zeros_like(voxel)
    return (voxel * 2.0 - 1.0).astype(np.float32)


def normalize(voxel: np.ndarray, phase: str, mode: str = "t1") -> np.ndarray:
    """Normalize intensity for a given pipeline phase ('cropping', 'stripping', 'parcellation', 'hemisphere')."""
    if mode == "t1c" and phase in ("stripping", "parcellation", "hemisphere"):
        return preprocess_intensity(voxel, clip_std=3.0)

    voxel = np.asarray(voxel, dtype=np.float32)
    nonzero = voxel[voxel > 0]
    if phase in ("cropping", "stripping"):
        clip = 2.0
    elif phase in ("parcellation", "hemisphere"):
        clip = 3.0
    else:
        raise ValueError(f"Unknown normalization phase: {phase}")

    if nonzero.size:
        voxel = np.clip(voxel, 0.0, float(np.mean(nonzero) + np.std(nonzero) * clip))
    vmin = float(np.min(voxel))
    vmax = float(np.max(voxel))
    if vmax > vmin:
        voxel = (voxel - vmin) / (vmax - vmin)
    else:
        voxel = np.zeros_like(voxel)
    voxel = (voxel * 2.0) - 1.0
    return voxel.astype("float32")
