"""Multi-view 2.5D brain parcellation (142 classes for OpenMAP-T1, 139 classes for OpenMAP-T1c)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from utils.image import normalize
from utils.pipeline.infer import DIRECTION_VALUE, forward_plane


def _num_classes(model: nn.Module, default: int = 142) -> int:
    dconv0 = getattr(model, "dconv0", None)
    if dconv0 is not None and hasattr(dconv0, "out_channels"):
        return int(dconv0.out_channels)
    return default


def parcellate(
    voxel: np.ndarray,
    model: nn.Module,
    device: torch.device,
    mode: str,
    n_classes: int | None = None,
    batch_size: int = 16,
    use_amp: bool = False,
    out_view: torch.Tensor | None = None,
    accumulate: bool = False,
) -> torch.Tensor | None:
    """Perform 2.5D neural network inference for brain parcellation along a specific anatomical plane."""
    plane_key = mode.lower()
    if plane_key not in DIRECTION_VALUE:
        raise ValueError("mode must be one of {'Axial', 'Coronal', 'Sagittal'}")
    if n_classes is None:
        n_classes = _num_classes(model)
    return forward_plane(
        voxel.astype(np.float32, copy=False),
        model=model,
        device=device,
        n_classes=n_classes,
        direction=DIRECTION_VALUE[plane_key],
        batch_size=batch_size,
        use_amp=use_amp,
        binary=False,
        out_view=out_view,
        accumulate=accumulate,
    )


def parcellation(
    voxel: np.ndarray,
    pnet: nn.Module,
    device: torch.device,
    mode: str = "t1",
    batch_size: int = 16,
    use_amp: bool = False,
    fp16_accum: bool = False,
) -> np.ndarray:
    """Perform full 3D brain parcellation across Coronal, Sagittal, and Axial planes.

    With `fp16_accum`, the 3-plane probability sum is kept in float16 (half the memory); the
    result can differ from the float32 default at a few near-tie boundary voxels.
    """
    n_classes = _num_classes(pnet)
    voxel = normalize(voxel, "parcellation", mode=mode)

    coronal = voxel.transpose(1, 2, 0)
    sagittal = voxel
    axial = voxel.transpose(2, 1, 0)

    # One (n_classes, X, Y, Z) accumulator; each plane writes its probabilities straight into a
    # permuted view of it, so no per-plane full-volume tensor is allocated. The planes are summed
    # in the same order as before (coronal + sagittal + axial), so the result is unchanged.
    acc = torch.empty((n_classes, *voxel.shape), dtype=torch.float16 if fp16_accum else torch.float32)
    planes = (
        (coronal, "Coronal", (2, 0, 3, 1)),
        (sagittal, "Sagittal", (1, 0, 2, 3)),
        (axial, "Axial", (3, 0, 2, 1)),
    )
    for k, (plane, name, inverse) in enumerate(planes):
        parcellate(
            plane,
            pnet,
            device,
            name,
            n_classes=n_classes,
            batch_size=batch_size,
            use_amp=use_amp,
            out_view=acc.permute(*inverse),
            accumulate=k > 0,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return torch.argmax(acc, dim=0).numpy()
