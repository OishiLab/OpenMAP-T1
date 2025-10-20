import os

import nibabel as nib
import numpy as np
from nibabel import processing


def normalize(voxel, mode):
    nonzero = voxel[voxel > 0]
    if mode in ["cropping", "stripping"]:
        clip = 2
    elif mode in ["parcellation", "hemisphere"]:
        clip = 3
    voxel = np.clip(voxel, 0, np.mean(nonzero) + np.std(nonzero) * clip)
    voxel = (voxel - np.min(voxel)) / (np.max(voxel) - np.min(voxel))
    voxel = (voxel * 2) - 1
    return voxel.astype("float32")


def reimburse_conform(output_dir, basename, suffix, odata, data, output):
    nii = nib.Nifti1Image(output.astype(np.uint16), affine=data.affine)
    header = odata.header
    nii = processing.conform(
        nii,
        out_shape=(header["dim"][1], header["dim"][2], header["dim"][3]),
        voxel_size=(header["pixdim"][1], header["pixdim"][2], header["pixdim"][3]),
        order=0,
    )
    os.makedirs(os.path.join(output_dir, f"{suffix}"), exist_ok=True)
    nib.save(nii, os.path.join(output_dir, f"{suffix}/{basename}_{suffix}_mask.nii"))

    result = odata.get_fdata().astype("float32") * nii.get_fdata().astype("int16")
    nii = nib.Nifti1Image(result.astype(np.float32), affine=odata.affine)
    nib.save(nii, os.path.join(output_dir, f"{suffix}/{basename}_{suffix}.nii"))
    return
