import numpy as np


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
