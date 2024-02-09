import pickle

import numpy as np
import torch


def postprocessing(parcellated, separated, shift, device):
    with open("utils/split_map.pkl", "rb") as tf:
        dictionary = pickle.load(tf)
        
    label_map_t = torch.tensor(parcellated.astype("int16"), requires_grad=False).to(device)
    label_half_t = torch.tensor(separated.astype("int16"), requires_grad=False).to(device)
    combined = torch.stack((torch.flatten(label_half_t), torch.flatten(label_map_t)), axis=-1)
    output = torch.zeros_like(label_half_t).ravel()
    for key, value in dictionary.items():
        key = torch.tensor(key, requires_grad=False).to(device)
        mask = torch.all(combined == key, axis=1)
        output[mask] = value
    output = output.reshape(label_half_t.shape)
    output = output.cpu().detach().numpy()
    output = output * (np.logical_or(np.logical_or(separated > 0, parcellated==87), parcellated==138))
    output = np.pad(output, [(32,32),(16,16),(32,32)], "constant", constant_values=0)
    output = np.roll(output, (-shift[0],-shift[1],-shift[2]), axis=(0,1,2))
    return output