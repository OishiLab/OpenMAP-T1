import numpy as np
import torch
from scipy import ndimage

from utils.functions import normalize, reimburse_conform


def strip(voxel, model, device):
    """
    Perform slice-wise inference using the brain stripping model.

    This function processes the input 3D volume slice by slice (along the first axis),
    using a three-slice context window for each prediction. The output is a 3D mask
    representing the brain region.

    Args:
        voxel (numpy.ndarray): Input voxel data of shape (N, 224, 224), typically
            a single anatomical orientation (e.g., coronal or sagittal view).
        model (torch.nn.Module): The trained PyTorch brain stripping model.
        device (torch.device): Device used for inference (CPU, CUDA, or MPS).

    Returns:
        torch.Tensor: A tensor of shape (224, 224, 224) representing the predicted
        binary brain mask.
    """
    model.eval()

    # Pad one slice on both ends to ensure valid 3-slice context at the boundaries
    voxel = np.pad(voxel, [(1, 1), (0, 0), (0, 0)], "constant", constant_values=voxel.min())

    with torch.inference_mode():
        box = torch.zeros(224, 224, 224)

        # Perform model inference for each slice using a 3-slice context
        for i in range(1, 225):
            image = np.stack([voxel[i - 1], voxel[i], voxel[i + 1]])
            image = torch.tensor(image.reshape(1, 3, 224, 224)).to(device)
            x_out = torch.sigmoid(model(image)).detach().cpu()
            box[i - 1] = x_out

        # Return as a 3D mask tensor
        return box.reshape(224, 224, 224)


def stripping(output_dir, basename, voxel, odata, data, ssnet, shift, device):
    """
    Perform full 3D brain stripping using a deep learning model.

    This function applies a neural network-based skull-stripping algorithm to
    isolate the brain region from a 3D MRI volume. It performs inference along
    three anatomical orientations—coronal, sagittal, and axial—and fuses the
    predictions to obtain a robust binary mask. The mask is then applied to the
    input image, recentred, and saved.

    Args:
        output_dir (str): Directory where intermediate and final results will be saved.
        basename (str): Base name of the current case (used for file naming).
        voxel (numpy.ndarray): Input 3D voxel data (preprocessed MRI image).
        odata (nibabel.Nifti1Image): Original NIfTI image before preprocessing.
        data (nibabel.Nifti1Image): Preprocessed NIfTI image used for model input.
        ssnet (torch.nn.Module): Trained brain stripping network.
        shift (tuple[int, int, int]): The (x, y, z) offsets applied previously during cropping.
        device (torch.device): Device used for inference (CPU, CUDA, or MPS).

    Returns:
        numpy.ndarray: The skull-stripped 3D brain volume.
    """
    # Preserve original intensity data for later restoration
    original = voxel.copy()

    # Normalize the voxel intensities for model input
    voxel = normalize(voxel, "stripping")

    # Prepare data in three anatomical orientations
    coronal = voxel.transpose(1, 2, 0)
    sagittal = voxel
    axial = voxel.transpose(2, 1, 0)

    # Apply the model along each anatomical plane
    out_c = strip(coronal, ssnet, device).permute(2, 0, 1)  # coronal → native orientation
    out_s = strip(sagittal, ssnet, device)  # sagittal
    out_a = strip(axial, ssnet, device).permute(2, 1, 0)  # axial → native orientation

    # Fuse predictions by averaging across the three planes and apply threshold
    out_e = ((out_c + out_s + out_a) / 3) > 0.5
    out_e = out_e.cpu().numpy()

    # Apply the binary mask to extract the brain region
    stripped = original * out_e

    # Restore the mask to the original conformed geometry
    # Pad to original full size and reverse the previously applied shift
    out_e = np.pad(out_e, [(16, 16), (16, 16), (16, 16)], "constant", constant_values=0)
    out_e = np.roll(out_e, (-shift[0], -shift[1], -shift[2]), axis=(0, 1, 2))

    # Save the binary brain mask in conformed space for reference
    reimburse_conform(output_dir, basename, "stripped", odata, data, out_e)

    return stripped
