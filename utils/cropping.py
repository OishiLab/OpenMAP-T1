import numpy as np
import torch
from scipy.ndimage import binary_closing

from utils.functions import normalize


def crop(voxel, model, device):
    model.eval()
    with torch.inference_mode():
        output = torch.zeros(256, 256, 256).to(device)
        for i, v in enumerate(voxel):
            image = v.reshape(1, 1, 256, 256)
            image = torch.tensor(image).to(device)
            x_out = torch.sigmoid(model(image)).detach()
            if i == 0:
                output[0] = x_out
            else:
                output[i] = x_out
        return output.reshape(256, 256, 256)


def closing(voxel):
    selem = np.ones((3, 3, 3), dtype="bool")
    voxel = binary_closing(voxel, structure=selem, iterations=3)
    return voxel


def cropping(data, cnet, device):
    voxel = data.get_fdata()
    voxel = normalize(voxel)

    coronal = voxel.transpose(1, 2, 0)
    sagittal = voxel
    out_c = crop(coronal, cnet, device).permute(2, 0, 1)
    out_s = crop(sagittal, cnet, device)
    out_e = ((out_c + out_s) / 2) > 0.5
    out_e = out_e.cpu().numpy()
    out_e = closing(out_e)
    cropped = data.get_fdata() * out_e
    return cropped
