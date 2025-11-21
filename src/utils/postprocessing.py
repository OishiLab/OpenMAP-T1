import os
import pickle

import numpy as np
import torch

# Get the absolute path of the current file (postprocessing.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the 'split_map.pkl' lookup table
SPLIT_MAP_PATH = os.path.join(CURRENT_DIR, "split_map.pkl")


def postprocessing(parcellated, separated, shift, device):
    """
    Perform post-processing to combine parcellation and hemisphere segmentation results.

    This function fuses the outputs of two neural networks:
      - The *parcellation* network, which labels fine-grained anatomical regions.
      - The *hemisphere* network, which distinguishes left and right hemispheres.

    It uses a predefined mapping (`split_map.pkl`) to merge region and hemisphere
    labels into a unified integer-encoded segmentation map. The output is then
    spatially restored to the original coordinate system using the recorded shift
    and padding offsets.

    Args:
        parcellated (numpy.ndarray): 3D integer array from the parcellation network,
            where each voxel corresponds to an anatomical label (1–142).
        separated (numpy.ndarray): 3D integer array from the hemisphere network,
            where voxel values indicate hemisphere classification:
                0 = background, 1 = left hemisphere, 2 = right hemisphere.
        shift (tuple[int, int, int]): Offsets (xd, yd, zd) used during cropping to
            center the brain; used here to roll the output back to its original location.
        device (torch.device): Device (CPU, CUDA, or MPS) for tensor-based computation.

    Returns:
        numpy.ndarray: The final 3D integer segmentation map where each voxel’s value
        encodes both hemisphere and regional identity, aligned to the original space.
    """
    # -----------------------------------------------------------
    # Step 1: Load the hemisphere–region label correspondence map
    # -----------------------------------------------------------
    # The dictionary in split_map.pkl maps pairs of (hemisphere_label, region_label)
    # to unified segmentation indices.
    with open(SPLIT_MAP_PATH, "rb") as tf:
        dictionary = pickle.load(tf)

    # -----------------------------------------------------------
    # Step 2: Convert input arrays to PyTorch tensors for efficient computation
    # -----------------------------------------------------------
    pmap = torch.tensor(parcellated.astype("int16"), requires_grad=False).to(device)
    hmap = torch.tensor(separated.astype("int16"), requires_grad=False).to(device)

    # Combine flattened hemisphere and parcellation labels into a two-column tensor
    # Each row represents (hemisphere_label, region_label)
    combined = torch.stack((torch.flatten(hmap), torch.flatten(pmap)), axis=-1)

    # Initialize an empty flattened output tensor
    output = torch.zeros_like(hmap).ravel()

    # -----------------------------------------------------------
    # Step 3: Map combined (hemisphere, region) label pairs to final class IDs
    # -----------------------------------------------------------
    # For each entry in the lookup dictionary:
    #   - 'key' represents a pair (hemisphere_label, region_label)
    #   - 'value' is the corresponding unified label in the final segmentation
    for key, value in dictionary.items():
        key = torch.tensor(key, requires_grad=False).to(device)
        mask = torch.all(combined == key, axis=1)
        output[mask] = value

    # Reshape flattened output back to 3D volume
    output = output.reshape(hmap.shape)

    # Move tensor to CPU and convert back to NumPy array
    output = output.cpu().detach().numpy()

    # -----------------------------------------------------------
    # Step 4: Mask irrelevant voxels to clean up final segmentation
    # -----------------------------------------------------------
    # Retain only voxels belonging to hemispheres or specific parcellation indices (87, 138),
    # which likely correspond to midline or reference structures.
    output = output * (np.logical_or(np.logical_or(separated > 0, parcellated == 87), parcellated == 136))

    # -----------------------------------------------------------
    # Step 5: Restore original spatial position
    # -----------------------------------------------------------
    # Undo the cropping offsets by applying padding and rolling back shifts.
    output = np.pad(output, [(16, 16), (16, 16), (16, 16)], "constant", constant_values=0)
    output = np.roll(output, (-shift[0], -shift[1], -shift[2]), axis=(0, 1, 2))

    # Return the final postprocessed segmentation map
    return output
