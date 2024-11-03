import numpy as np
import torch

from utils.functions import normalize


def parcellate(voxel, model, device, mode):
    """
    Parcellates a given voxel volume using a specified model and mode.

    Args:
        voxel (numpy.ndarray): The input voxel volume to be parcellated.
        model (torch.nn.Module): The neural network model used for parcellation.
        device (torch.device): The device (CPU or GPU) on which the model is run.
        mode (str): The mode of parcellation. Can be 'c', 's', or 'a', which determines the stack dimensions.

    Returns:
        torch.Tensor: The parcellated voxel volume.
    """
    if mode == "c":
        stack = (224, 192, 192)
    elif mode == "s":
        stack = (192, 224, 192)
    elif mode == "a":
        stack = (192, 224, 192)

    # Set the model to evaluation mode
    model.eval()

    # Pad the voxel volume to handle edge cases
    voxel = np.pad(voxel, [(1, 1), (0, 0), (0, 0)], "constant", constant_values=voxel.min())

    # Disable gradient calculation for inference
    with torch.inference_mode():
        # Initialize an empty tensor to store the parcellation results
        box = torch.zeros(stack[0], 142, stack[1], stack[2])

        # Iterate over each slice in the stack dimension
        for i in range(1, stack[0] + 1):
            # Stack three consecutive slices to form the input image
            image = np.stack([voxel[i - 1], voxel[i], voxel[i + 1]])
            image = torch.tensor(image.reshape(1, 3, stack[1], stack[2]))
            image = image.to(device)

            # Perform the forward pass through the model and apply softmax
            x_out = torch.softmax(model(image), 1).detach().cpu()

            # Store the output in the corresponding slice of the box tensor
            box[i - 1] = x_out

        # Reshape the box tensor to the desired output shape
        return box.reshape(stack[0], 142, stack[1], stack[2])


def parcellation(voxel, pnet_c, pnet_s, pnet_a, device):
    """
    Perform parcellation on the given voxel data using provided neural networks for coronal, sagittal, and axial views.

    Args:
        voxel (torch.Tensor): The input 3D voxel data to be parcellated.
        pnet_c (torch.nn.Module): The neural network model for coronal view parcellation.
        pnet_s (torch.nn.Module): The neural network model for sagittal view parcellation.
        pnet_a (torch.nn.Module): The neural network model for axial view parcellation.
        device (torch.device): The device (CPU or GPU) to perform computations on.

    Returns:
        numpy.ndarray: The parcellated output as a numpy array.
    """
    # Normalize the voxel data
    voxel = normalize(voxel)

    # Prepare the voxel data for different views
    coronal = voxel.transpose(1, 2, 0)
    sagittal = voxel
    axial = voxel.transpose(2, 1, 0)

    # Perform parcellation for the coronal view
    out_c = parcellate(coronal, pnet_c, device, "c").permute(1, 3, 0, 2)
    torch.cuda.empty_cache()

    # Perform parcellation for the sagittal view
    out_s = parcellate(sagittal, pnet_s, device, "s").permute(1, 0, 2, 3)
    torch.cuda.empty_cache()

    # Combine the results from coronal and sagittal views
    out_e = out_c + out_s
    del out_c, out_s

    # Perform parcellation for the axial view
    out_a = parcellate(axial, pnet_a, device, "a").permute(1, 3, 2, 0)
    torch.cuda.empty_cache()

    # Combine the results from all views
    out_e = out_e + out_a
    del out_a

    # Get the final parcellated output by taking the argmax
    parcellated = torch.argmax(out_e, 0).numpy()

    return parcellated
