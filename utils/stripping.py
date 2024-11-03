import numpy as np
import torch
from scipy import ndimage

from utils.functions import normalize


def strip(voxel, model, device):
    """
    Applies a given model to a 3D voxel array and returns the processed output.

    Args:
        voxel (numpy.ndarray): A 3D numpy array of shape (256, 256, 256) representing the input voxel data.
        model (torch.nn.Module): A PyTorch model to be used for processing the voxel data.
        device (torch.device): The device (CPU or GPU) on which the model and data should be loaded.

    Returns:
        torch.Tensor: A 3D tensor of shape (256, 256, 256) containing the processed output.
    """
    model.eval()
    with torch.inference_mode():
        output = torch.zeros(256, 256, 256).to(device)
        for i, v in enumerate(voxel):
            image = v.reshape(1, 1, 256, 256)
            image = torch.tensor(image).to(device)
            x_out = torch.sigmoid(model(image)).detach()
            output[i] = x_out
        return output.reshape(256, 256, 256)


def stripping(voxel, data, ssnet, device):
    """
    Perform brain stripping on a given voxel using a specified neural network.

    This function normalizes the input voxel, applies brain stripping in three anatomical planes
    (coronal, sagittal, and axial), and combines the results to produce a final stripped brain image.
    The stripped image is then centered and cropped.

    Args:
        voxel (numpy.ndarray): The input 3D voxel data to be stripped.
        data (nibabel.Nifti1Image): The original neuroimaging data.
        ssnet (torch.nn.Module): The neural network model used for brain stripping.
        device (torch.device): The device on which the neural network model is loaded (e.g., CPU or GPU).

    Returns:
        tuple: A tuple containing:
            - stripped (numpy.ndarray): The stripped and processed brain image.
            - (xd, yd, zd) (tuple of int): The shifts applied to center the brain image in the x, y, and z directions.
    """
    voxel = normalize(voxel)

    coronal = voxel.transpose(1, 2, 0)
    sagittal = voxel
    axial = voxel.transpose(2, 1, 0)
    out_c = strip(coronal, ssnet, device).permute(2, 0, 1)
    out_s = strip(sagittal, ssnet, device)
    out_a = strip(axial, ssnet, device).permute(2, 1, 0)
    out_e = ((out_c + out_s + out_a) / 3) > 0.5
    out_e = out_e.cpu().numpy()
    stripped = data.get_fdata() * out_e

    x, y, z = map(int, ndimage.center_of_mass(out_e))
    xd = 128 - x
    yd = 120 - y
    zd = 128 - z
    stripped = np.roll(stripped, (xd, yd, zd), axis=(0, 1, 2))
    stripped = stripped[32:-32, 16:-16, 32:-32]
    return stripped, (xd, yd, zd)
