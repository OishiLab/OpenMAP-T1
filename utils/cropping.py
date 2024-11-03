import numpy as np
import torch
from scipy.ndimage import binary_closing

from utils.functions import normalize


def crop(voxel, model, device):
    """
    Crops the given voxel data using the provided model and device.

    Args:
        voxel (numpy.ndarray): The input voxel data to be cropped, expected to be of shape (N, 256, 256).
        model (torch.nn.Module): The PyTorch model used for cropping.
        device (torch.device): The device (CPU or GPU) on which the computation will be performed.

    Returns:
        torch.Tensor: The cropped output tensor of shape (256, 256, 256).
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


def closing(voxel):
    """
    Perform a binary closing operation on a 3D voxel array.

    This function applies a binary closing operation using a 3x3x3 structuring element
    and performs the operation for a specified number of iterations.

    Parameters:
    voxel (numpy.ndarray): A 3D numpy array representing the voxel data to be processed.

    Returns:
    numpy.ndarray: The voxel data after the binary closing operation.
    """
    selem = np.ones((3, 3, 3), dtype="bool")
    voxel = binary_closing(voxel, structure=selem, iterations=3)
    return voxel


def cropping(data, cnet, device):
    """
    Crops the input medical imaging data using a neural network model.

    Args:
        data (nibabel.Nifti1Image): The input medical imaging data in NIfTI format.
        cnet (torch.nn.Module): The neural network model used for cropping.
        device (torch.device): The device (CPU or GPU) on which the model is run.

    Returns:
        numpy.ndarray: The cropped medical imaging data.
    """
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
