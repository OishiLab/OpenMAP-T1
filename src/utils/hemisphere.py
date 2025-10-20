import numpy as np
import torch
from scipy.ndimage import binary_dilation

from utils.functions import normalize


def separate(voxel, model, device):
    """
    Perform slice-wise inference using a hemisphere separation model.

    This function runs a 2.5D neural network across slices of a 3D input volume.
    Each slice is processed in the context of its immediate neighbors (previous
    and next slices) to improve spatial coherence. The model outputs a
    three-class probability map distinguishing background, left hemisphere,
    and right hemisphere regions.

    Args:
        voxel (numpy.ndarray): Input voxel data of shape (N, 224, 224).
        model (torch.nn.Module): Trained hemisphere segmentation model (U-Net architecture).
        device (torch.device): Computational device (CPU, CUDA, or MPS).

    Returns:
        torch.Tensor: A tensor of shape (224, 3, 224, 224) containing softmax
        probabilities for each class at every voxel.
    """
    model.eval()

    # Pad the volume by one slice on both ends to provide full 3-slice context
    voxel = np.pad(voxel, [(1, 1), (0, 0), (0, 0)], "constant", constant_values=voxel.min())

    with torch.inference_mode():
        # Output tensor for storing model predictions (class probabilities)
        box = torch.zeros(224, 3, 224, 224)

        # Iterate slice-by-slice along the first axis
        for i in range(1, 225):
            image = np.stack([voxel[i - 1], voxel[i], voxel[i + 1]])
            image = torch.tensor(image.reshape(1, 3, 224, 224)).to(device)

            # Model inference with softmax normalization across classes
            x_out = torch.softmax(model(image), dim=1).detach().cpu()
            box[i - 1] = x_out

        # Return complete 3D probability map
        return box.reshape(224, 3, 224, 224)


def hemisphere(voxel, hnet, device):
    """
    Perform hemisphere separation on a brain MRI volume using a deep learning model.

    The function predicts left and right hemisphere regions from a normalized
    3D MRI volume using multi-view inference (coronal and transverse planes).
    Predictions from both orientations are fused to improve robustness. The final
    label map is post-processed using binary dilation to smooth and expand hemisphere
    boundaries, ensuring anatomical continuity.

    Args:
        voxel (numpy.ndarray): Input 3D brain volume to be separated into hemispheres.
        hnet (torch.nn.Module): Trained hemisphere segmentation model.
        device (torch.device): Target device for computation (e.g., 'cuda', 'cpu').

    Returns:
        numpy.ndarray: A 3D integer array representing the hemisphere mask:
            - 0: Background
            - 1: Left hemisphere
            - 2: Right hemisphere
    """
    # Normalize voxel intensities for inference
    voxel = normalize(voxel, "hemisphere")

    # Prepare different anatomical orientations for inference
    coronal = voxel.transpose(1, 2, 0)
    transverse = voxel.transpose(2, 1, 0)

    # Perform inference for both coronal and transverse orientations
    out_c = separate(coronal, hnet, device).permute(1, 3, 0, 2)
    out_a = separate(transverse, hnet, device).permute(1, 3, 2, 0)

    # Fuse both outputs by summing class probabilities
    out_e = out_c + out_a

    # Determine final class labels (0, 1, or 2) by selecting the most probable class
    out_e = torch.argmax(out_e, dim=0).cpu().numpy()

    # Release any residual GPU memory
    torch.cuda.empty_cache()

    # --------------------------
    # Post-processing step: binary dilation
    # --------------------------

    # First, dilate the left hemisphere (class 1)
    dilated_mask_1 = binary_dilation(out_e == 1, iterations=5).astype("int16")
    # Preserve right hemisphere voxels from the original prediction
    dilated_mask_1[out_e == 2] = 2

    # Then, dilate the right hemisphere (class 2) symmetrically
    dilated_mask_2 = binary_dilation(dilated_mask_1 == 2, iterations=5).astype("int16") * 2
    # Restore left hemisphere voxels to prevent overwriting
    dilated_mask_2[dilated_mask_1 == 1] = 1

    # Return the final dilated and fused hemisphere mask
    return dilated_mask_2
