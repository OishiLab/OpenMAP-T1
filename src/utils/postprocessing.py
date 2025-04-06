import os
import pickle

import numpy as np
import torch

# このファイル(postprocessing.py)のあるディレクトリを取得
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# split_map.pkl は同じディレクトリ内にあるため、CURRENT_DIR を基にパスを作成
SPLIT_MAP_PATH = os.path.join(CURRENT_DIR, "split_map.pkl")


def postprocessing(parcellated, separated, shift, device):
    # 絶対パスを用いて split_map.pkl を読み込む
    with open(SPLIT_MAP_PATH, "rb") as tf:
        dictionary = pickle.load(tf)

    pmap = torch.tensor(parcellated.astype("int16"), requires_grad=False).to(device)
    hmap = torch.tensor(separated.astype("int16"), requires_grad=False).to(device)
    combined = torch.stack((torch.flatten(hmap), torch.flatten(pmap)), axis=-1)
    output = torch.zeros_like(hmap).ravel()
    for key, value in dictionary.items():
        key = torch.tensor(key, requires_grad=False).to(device)
        mask = torch.all(combined == key, axis=1)
        output[mask] = value
    output = output.reshape(hmap.shape)
    output = output.cpu().detach().numpy()
    output = output * (
        np.logical_or(np.logical_or(separated > 0, parcellated == 87), parcellated == 138)
    )
    output = np.pad(output, [(32, 32), (16, 16), (32, 32)], "constant", constant_values=0)
    output = np.roll(output, (-shift[0], -shift[1], -shift[2]), axis=(0, 1, 2))
    return output
