import numpy as np
import torch
from tqdm import tqdm

from utils.functions import normalize


def parcellate(
    voxel: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    mode: str,
    n_classes: int = 142,
) -> torch.Tensor:
    """
    Perform 2.5D neural network inference for brain parcellation along a specific anatomical plane.

    The function processes a 3D volume slice by slice using a 3-slice context window (previous,
    current, next). An additional constant-valued fourth channel encodes the orientation mode
    (Axial, Coronal, or Sagittal), allowing the network to distinguish the processing plane.

    Args:
        voxel (numpy.ndarray): 3D voxel data of shape (N, 224, 224), representing a single anatomical view.
        model (torch.nn.Module): The trained PyTorch parcellation model.
        device (torch.device): Device for inference (CPU, CUDA, or MPS).
        mode (str): The anatomical plane used for inference. Must be one of {'Axial', 'Coronal', 'Sagittal'}.
        n_classes (int, optional): Number of output anatomical labels. Defaults to 142.

    Returns:
        torch.Tensor: A tensor of shape (224, n_classes, 224, 224) containing softmax probabilities
        for each class at each voxel position.
    """
    model.eval()
    voxel = voxel.astype(np.float32)

    # Set the constant value for the 4th channel to encode plane orientation
    if mode == "Axial":
        section_value = 1.0
    elif mode == "Coronal":
        section_value = -1.0
    elif mode == "Sagittal":
        section_value = 0.0
    else:
        raise ValueError("mode must be one of {'Axial','Coronal','Sagittal'}")

    # Pad one slice on both ends to safely allow 3-slice context
    voxel_pad = np.pad(
        voxel,
        [(1, 1), (0, 0), (0, 0)],
        mode="constant",
        constant_values=float(voxel.min()),
    )

    # Initialize a container for the network outputs (CPU for accumulation)
    box = torch.empty((224, n_classes, 224, 224), dtype=torch.float32, device="cpu")

    # Inference loop: iterate over slices and feed triplets to the model
    with torch.inference_mode():
        for i in range(1, 225):
            prev_ = voxel_pad[i - 1]
            curr_ = voxel_pad[i]
            next_ = voxel_pad[i + 1]

            # Build 4-channel input (3 context slices + orientation encoding)
            four_ch = np.empty((4, 224, 224), dtype=np.float32)
            four_ch[0] = prev_
            four_ch[1] = curr_
            four_ch[2] = next_
            four_ch[3].fill(section_value)

            inp = torch.from_numpy(four_ch).unsqueeze(0).to(device)

            # Model inference with softmax normalization
            logits = model(inp)
            probs = torch.softmax(logits, dim=1)

            # Store softmax output for this slice
            box[i - 1] = probs

    return box


def parcellation(voxel, pnet, device):
    """
    Perform full 3D brain parcellation by aggregating predictions across multiple anatomical planes.

    The function normalizes the input MRI volume, generates three differently oriented representations
    (coronal, sagittal, axial), and performs 2.5D inference on each using a shared parcellation network.
    The resulting probability maps are fused by summation and converted into a discrete segmentation map
    via argmax over anatomical classes.

    Args:
        voxel (numpy.ndarray): Input 3D brain volume (float array).
        pnet (torch.nn.Module): Trained parcellation network (U-Net or similar architecture).
        device (torch.device): Device on which inference will be executed (CPU or GPU).

    Returns:
        numpy.ndarray: Final 3D parcellation map (integer label image) with voxel-wise anatomical labels.
    """
    # Normalize input intensities for network inference
    voxel = normalize(voxel, "parcellation")

    # Prepare three anatomical views for 2.5D inference
    coronal = voxel.transpose(1, 2, 0)
    sagittal = voxel
    axial = voxel.transpose(2, 1, 0)

    # ------------------------
    # Coronal view inference
    # ------------------------
    out_c = parcellate(coronal, pnet, device, "Coronal").permute(1, 3, 0, 2)
    torch.cuda.empty_cache()

    # ------------------------
    # Sagittal view inference
    # ------------------------
    out_s = parcellate(sagittal, pnet, device, "Sagittal").permute(1, 0, 2, 3)
    torch.cuda.empty_cache()

    # Fuse coronal and sagittal predictions
    out_e = out_c + out_s
    del out_c, out_s

    # ------------------------
    # Axial view inference
    # ------------------------
    out_a = parcellate(axial, pnet, device, "Axial").permute(1, 3, 2, 0)
    torch.cuda.empty_cache()

    # Combine outputs from all three anatomical orientations
    out_e = out_e + out_a
    del out_a

    # Convert probability maps to final integer labels
    parcellated = torch.argmax(out_e, 0).numpy()

    return parcellated
