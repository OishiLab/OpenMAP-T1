import numpy as np
import torch

from utils.functions import normalize


def parcellate(voxel, model, device, mode):
    if mode == "c":
        stack = (224, 192, 192)
    elif mode == "s":
        stack = (192, 224, 192)
    elif mode == "a":
        stack = (192, 224, 192)
    model.eval()
    voxel = np.pad(
        voxel, [(1, 1), (0, 0), (0, 0)], "constant", constant_values=voxel.min()
    )
    with torch.inference_mode():
        box = torch.zeros(stack[0], 142, stack[1], stack[2])
        for i in range(stack[0]):
            i += 1
            image = np.stack([voxel[i - 1], voxel[i], voxel[i + 1]])
            image = torch.tensor(image.reshape(1, 3, stack[1], stack[2]))
            image = image.to(device)
            x_out = torch.softmax(model(image), 1).detach().cpu()
            if i == 1:
                box[0] = x_out
            else:
                box[i - 1] = x_out
        return box.reshape(stack[0], 142, stack[1], stack[2])


def parcellation(voxel, pnet_c, pnet_s, pnet_a, device):
    voxel = normalize(voxel)

    coronal = voxel.transpose(1, 2, 0)
    sagittal = voxel
    axial = voxel.transpose(2, 1, 0)
    out_c = parcellate(coronal, pnet_c, device, "c").permute(1, 3, 0, 2)
    torch.cuda.empty_cache()
    out_s = parcellate(sagittal, pnet_s, device, "s").permute(1, 0, 2, 3)
    torch.cuda.empty_cache()
    out_e = out_c + out_s
    del out_c, out_s
    out_a = parcellate(axial, pnet_a, device, "a").permute(1, 3, 2, 0)
    torch.cuda.empty_cache()
    out_e = out_e + out_a
    del out_a
    parcellated = torch.argmax(out_e, 0).numpy()
    return parcellated
