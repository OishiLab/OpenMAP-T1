import os

import nibabel as nib
import numpy as np
from nibabel import processing

OUTPUT_SPACE_CHOICES = ("native", "conform", "both")


def spaces_to_save(output_space: str) -> tuple[str, ...]:
    if output_space == "both":
        return ("native", "conform")
    return (output_space,)


def output_subdir(base_subdir: str, space: str) -> str:
    if space == "conform":
        return os.path.join("conform", base_subdir)
    return base_subdir


def filename_suffix(space: str) -> str:
    if space == "conform":
        return "_1mm"
    return ""


def to_native_nifti(array, odata, data, dtype):
    nii = nib.Nifti1Image(array.astype(dtype), affine=data.affine)
    header = odata.header
    return processing.conform(
        nii,
        out_shape=(header["dim"][1], header["dim"][2], header["dim"][3]),
        voxel_size=(header["pixdim"][1], header["pixdim"][2], header["pixdim"][3]),
        order=0,
    )


def to_conform_nifti(array, data, dtype):
    return nib.Nifti1Image(array.astype(dtype), affine=data.affine)


def save_label_volume(
    output_dir,
    subdir,
    basename,
    name,
    array,
    odata,
    data,
    output_ext,
    output_space,
    dtype=np.uint16,
):
    for space in spaces_to_save(output_space):
        dest = os.path.join(output_dir, output_subdir(subdir, space))
        os.makedirs(dest, exist_ok=True)
        if space == "native":
            nii = to_native_nifti(array, odata, data, dtype)
        else:
            nii = to_conform_nifti(array, data, dtype)
        nib.save(nii, os.path.join(dest, f"{basename}_{name}{filename_suffix(space)}{output_ext}"))


def save_mask_and_masked(
    output_dir,
    basename,
    suffix,
    mask,
    odata,
    data,
    output_ext,
    output_space,
):
    for space in spaces_to_save(output_space):
        dest = os.path.join(output_dir, output_subdir(suffix, space))
        os.makedirs(dest, exist_ok=True)
        if space == "native":
            mask_nii = to_native_nifti(mask, odata, data, np.uint16)
            masked = odata.get_fdata().astype(np.float32) * mask_nii.get_fdata().astype(np.int16)
            masked_nii = nib.Nifti1Image(masked.astype(np.float32), affine=odata.affine)
        else:
            mask_nii = to_conform_nifti(mask, data, np.uint16)
            masked = data.get_fdata().astype(np.float32) * mask.astype(np.float32)
            masked_nii = nib.Nifti1Image(masked.astype(np.float32), affine=data.affine)
        nib.save(mask_nii, os.path.join(dest, f"{basename}_{suffix}_mask{filename_suffix(space)}{output_ext}"))
        nib.save(masked_nii, os.path.join(dest, f"{basename}_{suffix}{filename_suffix(space)}{output_ext}"))
