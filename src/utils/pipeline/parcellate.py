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
) -> torch.Tensor:
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
    )


def parcellation(
    voxel: np.ndarray,
    pnet: nn.Module,
    device: torch.device,
    mode: str = "t1",
    batch_size: int = 16,
    use_amp: bool = False,
) -> np.ndarray:
    """Perform full 3D brain parcellation across Coronal, Sagittal, and Axial planes."""
    n_classes = _num_classes(pnet)
    voxel = normalize(voxel, "parcellation", mode=mode)

    coronal = voxel.transpose(1, 2, 0)
    sagittal = voxel
    axial = voxel.transpose(2, 1, 0)

    out_c = parcellate(coronal, pnet, device, "Coronal", n_classes=n_classes, batch_size=batch_size, use_amp=use_amp).permute(1, 3, 0, 2)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out_s = parcellate(sagittal, pnet, device, "Sagittal", n_classes=n_classes, batch_size=batch_size, use_amp=use_amp).permute(1, 0, 2, 3)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out_e = out_c + out_s
    del out_c, out_s

    out_a = parcellate(axial, pnet, device, "Axial", n_classes=n_classes, batch_size=batch_size, use_amp=use_amp).permute(1, 3, 2, 0)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out_e = out_e + out_a
    del out_a

    return torch.argmax(out_e, dim=0).numpy()
