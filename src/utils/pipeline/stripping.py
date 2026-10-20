from __future__ import annotations

import numpy as np
import torch

from utils.image import normalize
from utils.output.output_space import save_mask_and_masked
from utils.pipeline.infer import forward_plane


def strip(voxel: np.ndarray, model: torch.nn.Module, device: torch.device, batch_size: int = 16, use_amp: bool = False) -> torch.Tensor:
    """Perform slice-wise inference using the brain stripping model."""
    return forward_plane(
        voxel,
        model=model,
        device=device,
        n_classes=1,
        direction=None,
        batch_size=batch_size,
        use_amp=use_amp,
        binary=True,
    )


def stripping(
    output_dir: str,
    basename: str,
    voxel: np.ndarray,
    odata,
    data,
    ssnet: torch.nn.Module,
    shift: tuple[int, int, int],
    device: torch.device,
    output_ext: str = ".nii.gz",
    output_space: str = "native",
    mode: str = "t1",
    batch_size: int = 16,
    use_amp: bool = False,
) -> np.ndarray:
    """Perform full 3D brain stripping on the 224³ cropped volume."""
    original = voxel.copy()
    voxel = normalize(voxel, "stripping", mode=mode)

    coronal = voxel.transpose(1, 2, 0)
    sagittal = voxel
    axial = voxel.transpose(2, 1, 0)

    out_c = strip(coronal, ssnet, device, batch_size=batch_size, use_amp=use_amp).permute(2, 0, 1)
    out_s = strip(sagittal, ssnet, device, batch_size=batch_size, use_amp=use_amp)
    out_a = strip(axial, ssnet, device, batch_size=batch_size, use_amp=use_amp).permute(2, 1, 0)

    out_e = (((out_c + out_s + out_a) / 3.0) > 0.5).cpu().numpy()

    stripped = original * out_e.astype(np.float32)

    out_e_256 = np.pad(out_e, [(16, 16), (16, 16), (16, 16)], "constant", constant_values=0)
    out_e_256 = np.roll(out_e_256, (-shift[0], -shift[1], -shift[2]), axis=(0, 1, 2))

    save_mask_and_masked(output_dir, basename, "stripped", out_e_256, odata, data, output_ext, output_space)

    return stripped
