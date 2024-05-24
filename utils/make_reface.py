import ants
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation

def closing(out_e):
    selem = np.ones((5, 5, 5), dtype="bool")
    out_e = binary_closing(out_e, structure=selem, iterations=3)
    return out_e

def dilation(out_e):
    selem = np.ones((11, 11, 11), dtype="bool")
    out_e = binary_dilation(out_e, structure=selem, iterations=1)
    return out_e

def preprosess_skull(voxel, skull):
    nonzero = voxel[voxel>0]
    skullnonzero = skull[skull>0]
    voxel = (voxel-np.mean(nonzero)) / np.std(nonzero) * np.std(skullnonzero) + np.mean(skullnonzero)
    voxel = (voxel - np.min(voxel)) / (np.max(voxel) - np.min(voxel))
    return voxel

def preprosess(voxel):
    nonzero = voxel[voxel>0]
    voxel = np.clip(voxel, 0, np.mean(nonzero)+np.std(nonzero)*3)
    voxel = (voxel - np.min(voxel)) / (np.max(voxel) - np.min(voxel))
    return voxel

def make_reface(idata, bdata, vdata, ldata):
    idata_np = idata.get_fdata().astype("float32")
    ldata_np = ldata.get_fdata().astype("float32")
    bdata_np = bdata.get_fdata().astype("float32")
    vdata_np = vdata.get_fdata().astype("float32")
    
    idata_np_copy = idata_np.copy()
    idata_np = idata_np * (ldata_np > 0)
    
    idata_ants = ants.from_numpy(idata_np)
    bdata_ants = ants.from_numpy(bdata_np)
    vdata_ants = ants.from_numpy(vdata_np)
    
    tx = ants.registration(idata_ants, bdata_ants, type_of_transform = 'TRSAA', grad_step = 0.2, shrinkfactors = (8,4,2,1), smoothingsigmas =(3,2,1,0), reg_iterations = (500,500,200,50))
    warp = tx['warpedmovout']
    reg_voxel = ants.apply_transforms(idata_ants, vdata_ants, transformlist = tx['fwdtransforms'], interpolator = "linear")
    reg_voxel = reg_voxel.numpy()
    
    ldata_np = dilation(closing(ldata_np))
    reg_voxel[ldata_np > 0] = 0
    idata_np_skull = idata_np_copy * (ldata_np == 0)
    idata_np_brain = idata_np_copy * (ldata_np > 0)
    
    reg_voxel = preprosess_skull(reg_voxel,idata_np_skull)
    idata_np_brain = preprosess(idata_np_brain)
    reface_img = (reg_voxel + idata_np_brain) * 255
    return reface_img