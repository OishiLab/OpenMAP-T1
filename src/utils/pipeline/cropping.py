from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import binary_closing

from utils.image import normalize
from utils.output.output_space import save_mask_and_masked
from utils.pipeline.infer import forward_plane

CROP = 16


def crop(voxel: np.ndarray, model: torch.nn.Module, device: torch.device, batch_size: int = 16, use_amp: bool = False) -> torch.Tensor:
    """Apply a 2.5D cropping network across slices of `voxel` (shape: N, 256, 256)."""
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


def closing(voxel: np.ndarray) -> np.ndarray:
    """Apply binary morphological closing to a 3D mask."""
    selem = np.ones((3, 3, 3), dtype=bool)
    return binary_closing(voxel, structure=selem, iterations=3)


def cropping(
    output_dir: str,
    basename: str,
    odata,
    data,
    cnet: torch.nn.Module,
    device: torch.device,
    output_ext: str = ".nii.gz",
    output_space: str = "native",
    batch_size: int = 16,
    use_amp: bool = False,
):
    """
    Perform 3D brain region cropping using CNet.

    Returns:
        tuple:
            - numpy.ndarray: Cropped brain volume of shape (224, 224, 224).
            - tuple[int, int, int]: (xd, yd, zd) shift applied to center the brain.
    """
    volume = data.get_fdata().astype("float32")
    voxel = normalize(volume, "cropping", mode="t1")

    coronal = voxel.transpose(1, 2, 0)
    sagittal = voxel

    out_c = crop(coronal, cnet, device, batch_size=batch_size, use_amp=use_amp).permute(2, 0, 1)
    out_s = crop(sagittal, cnet, device, batch_size=batch_size, use_amp=use_amp)

    out_e = (((out_c + out_s) / 2.0) > 0.5).cpu().numpy()
    out_e = closing(out_e)

    cropped = volume * out_e.astype("float32")

    save_mask_and_masked(output_dir, basename, "cropped", out_e, odata, data, output_ext, output_space)

    if not np.any(out_e):
        return cropped[CROP:-CROP, CROP:-CROP, CROP:-CROP].copy(), (0, 0, 0)

    x, y, z = map(int, ndimage.center_of_mass(out_e))
    xd = 128 - x
    yd = 120 - y
    zd = 128 - z

    cropped = np.roll(cropped, (xd, yd, zd), axis=(0, 1, 2))
    cropped = cropped[CROP:-CROP, CROP:-CROP, CROP:-CROP]

    return cropped, (xd, yd, zd)
