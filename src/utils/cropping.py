import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import binary_closing

from utils.functions import normalize, reimburse_conform


def crop(voxel, model, device):
    """
    Apply a neural network-based cropping operation on 3D voxel data.

    This function slides a 3-slice window across the input volume along the first axis
    and predicts a binary mask for each slice using the given model. The outputs are then
    aggregated into a full 3D prediction volume.

    Args:
        voxel (numpy.ndarray): Input 3D array of shape (N, 256, 256). The first dimension
            corresponds to the slice index (typically coronal or sagittal).
        model (torch.nn.Module): The trained PyTorch model that predicts binary masks
            for each input slice triplet.
        device (torch.device): The device (CPU, CUDA, or MPS) on which inference will run.

    Returns:
        torch.Tensor: The predicted 3D binary mask of shape (256, 256, 256).
    """
    # Pad the input volume by one slice at each end to allow 3-slice context
    voxel = np.pad(voxel, [(1, 1), (0, 0), (0, 0)], "constant", constant_values=voxel.min())
    model.eval()

    with torch.inference_mode():
        box = torch.zeros(256, 256, 256)

        # Iterate through each target slice and predict using a 3-slice input context
        for i in range(1, 257):
            image = np.stack([voxel[i - 1], voxel[i], voxel[i + 1]])
            image = torch.tensor(image.reshape(1, 3, 256, 256)).to(device)
            x_out = torch.sigmoid(model(image)).detach().cpu()
            box[i - 1] = x_out

        return box.reshape(256, 256, 256)


def closing(voxel):
    """
    Apply binary morphological closing to a 3D mask.

    The closing operation helps remove small holes and connect adjacent regions,
    thereby producing a more continuous mask.

    Args:
        voxel (numpy.ndarray): 3D binary array representing the voxel mask.

    Returns:
        numpy.ndarray: Smoothed 3D mask after applying binary closing.
    """
    selem = np.ones((3, 3, 3), dtype=bool)
    voxel = binary_closing(voxel, structure=selem, iterations=3)
    return voxel


def cropping(output_dir, basename, odata, data, cnet, device):
    """
    Perform 3D brain region cropping using a deep learning model.

    The function normalizes the image, runs dual-view (coronal and sagittal) inference
    to estimate the brain mask, refines it using morphological operations, and finally
    centers and crops the resulting image around the brain.

    Args:
        output_dir (str): Directory where intermediate and final outputs are saved.
        basename (str): Base filename (without extension) for saving outputs.
        odata (nibabel.Nifti1Image): Original input image (pre-conformation).
        data (nibabel.Nifti1Image): Preprocessed and conformed input image.
        cnet (torch.nn.Module): Cropping network model.
        device (torch.device): Device used for inference.

    Returns:
        tuple:
            - numpy.ndarray: Cropped brain volume of shape approximately (224, 224, 224).
            - tuple[int, int, int]: (xd, yd, zd) shift applied to center the brain.
    """
    # Convert to float32 and normalize intensity
    voxel = data.get_fdata().astype("float32")
    voxel = normalize(voxel, "cropping")

    # Generate two orthogonal views for model inference
    coronal = voxel.transpose(1, 2, 0)
    sagittal = voxel

    # Run model inference for both views
    out_c = crop(coronal, cnet, device).permute(2, 0, 1)
    out_s = crop(sagittal, cnet, device)

    # Average predictions from both views and threshold
    out_e = ((out_c + out_s) / 2) > 0.5
    out_e = out_e.cpu().numpy()

    # Refine mask via binary closing
    out_e = closing(out_e)

    # Apply the mask to the original image
    cropped = data.get_fdata().astype("float32") * out_e

    # Save the binary mask in the output directory
    reimburse_conform(output_dir, basename, "cropped", odata, data, out_e)

    # Compute center of mass for the masked brain
    x, y, z = map(int, ndimage.center_of_mass(out_e))

    # Compute shifts required to center the brain
    xd = 128 - x
    yd = 120 - y
    zd = 128 - z

    # Translate (roll) the image to center the brain region
    cropped = np.roll(cropped, (xd, yd, zd), axis=(0, 1, 2))

    # Crop out boundary padding to reduce size and focus on the centered brain
    cropped = cropped[16:-16, 16:-16, 16:-16]

    return cropped, (xd, yd, zd)
