import numpy as np


def normalize(voxel):
    nonzero = voxel[voxel > 0]
    voxel = np.clip(voxel, 0, np.mean(nonzero) + np.std(nonzero) * 2)
    voxel = (voxel - np.min(voxel)) / (np.max(voxel) - np.min(voxel))
    voxel = (voxel * 2) - 1
    return voxel.astype("float32")
