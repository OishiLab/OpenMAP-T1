"""Hemisphere separation on coronal and axial views (3-channel for OpenMAP-T1, 4-channel with direction for OpenMAP-T1c)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_dilation

from utils.image import normalize
from utils.pipeline.infer import DIRECTION_VALUE, forward_plane

N_CLASSES = 3
HEMI_DILATE = 5


def _in_channels(model: nn.Module, default: int = 3) -> int:
    econv0 = getattr(model, "econv0", None)
    if econv0 is not None and hasattr(econv0, "in_channels"):
        return int(econv0.in_channels)
    return default


def separate(
    voxel: np.ndarray,
    model: nn.Module,
    device: torch.device,
    direction: float | None = None,
    batch_size: int = 16,
    use_amp: bool = False,
) -> torch.Tensor:
    """Perform slice-wise inference using the hemisphere separation model."""
    return forward_plane(
        voxel.astype(np.float32, copy=False),
        model=model,
        device=device,
        n_classes=N_CLASSES,
        direction=direction,
        batch_size=batch_size,
        use_amp=use_amp,
        binary=False,
    )


def hemisphere(
    voxel: np.ndarray,
    hnet: nn.Module,
    device: torch.device,
    mode: str = "t1",
    batch_size: int = 16,
    use_amp: bool = False,
) -> np.ndarray:
    """Perform hemisphere separation on a 224³ brain volume."""
    voxel = normalize(voxel, "hemisphere", mode=mode)
    use_direction = _in_channels(hnet) == 4

    coronal = voxel.transpose(1, 2, 0)
    transverse = voxel.transpose(2, 1, 0)

    dir_c = DIRECTION_VALUE["coronal"] if use_direction else None
    dir_a = DIRECTION_VALUE["axial"] if use_direction else None

    out_c = separate(coronal, hnet, device, direction=dir_c, batch_size=batch_size, use_amp=use_amp).permute(1, 3, 0, 2)
    out_a = separate(transverse, hnet, device, direction=dir_a, batch_size=batch_size, use_amp=use_amp).permute(1, 3, 2, 0)

    out_e = torch.argmax(out_c + out_a, dim=0).cpu().numpy().astype("int16")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    dilated_mask_1 = binary_dilation(out_e == 1, iterations=HEMI_DILATE).astype("int16")
    dilated_mask_1[out_e == 2] = 2

    dilated_mask_2 = binary_dilation(dilated_mask_1 == 2, iterations=HEMI_DILATE).astype("int16") * 2
    dilated_mask_2[dilated_mask_1 == 1] = 1

    return dilated_mask_2
