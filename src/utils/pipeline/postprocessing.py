"""Post-processing to combine parcellation and hemisphere masks for OpenMAP-T1 and OpenMAP-T1c."""

from __future__ import annotations

import os
import pickle

import numpy as np
import torch
from scipy import ndimage

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_MAP_T1_PATH = os.path.join(CURRENT_DIR, "split_map.pkl")
SPLIT_MAP_T1C_PATH = os.path.join(CURRENT_DIR, "split_map_t1c.pkl")
CROP = 16

_LUT_CACHE: dict[str, np.ndarray] = {}


def _get_lut(mode: str) -> np.ndarray:
    if mode not in _LUT_CACHE:
        path = SPLIT_MAP_T1C_PATH if mode == "t1c" else SPLIT_MAP_T1_PATH
        with open(path, "rb") as handle:
            split_map = pickle.load(handle)
        max_h = max(int(k[0]) for k in split_map)
        max_p = max(int(k[1]) for k in split_map)
        lut = np.zeros((max_h + 1, max_p + 1), dtype=np.int16)
        for (hemi, parc), value in split_map.items():
            lut[int(hemi), int(parc)] = int(value)
        _LUT_CACHE[mode] = lut
    return _LUT_CACHE[mode]


def _insert_label(seg: np.ndarray, start: int, end: int) -> np.ndarray:
    out = seg.copy()
    for num in reversed(range(start, end)):
        out[seg == num] = num + 6
    return out


def _relabel_segmentation(parc: np.ndarray) -> np.ndarray:
    seg = _insert_label(parc, 251, 281)
    seg[seg == 257] = 252
    seg[seg == 258] = 254
    seg[seg == 259] = 256
    seg[seg == 260] = 258
    seg[seg == 261] = 259
    seg[seg == 262] = 260
    seg[seg == 263] = 262
    seg[seg == 281] = 251
    seg[seg == 282] = 253
    seg[seg == 283] = 255
    seg[seg == 284] = 257
    seg[seg == 285] = 261
    seg[seg == 286] = 263
    return seg


def _fuse_280_t1c(parcellated: np.ndarray, separated: np.ndarray) -> np.ndarray:
    """Map (hemisphere, 139-class label) to 274 regions, then split sulcal CSF into internal/external to 280."""
    hemi = separated.astype(np.int16, copy=False)
    parc = parcellated.astype(np.int16, copy=False)
    lut = _get_lut("t1c")
    h = np.clip(hemi, 0, lut.shape[0] - 1)
    p = np.clip(parc, 0, lut.shape[1] - 1)
    output = lut[h, p]
    output = output * np.logical_or.reduce((hemi > 0, parc == 87, parc == 136))
    output = output.astype(np.int16)

    boundary = (hemi == 1) & ndimage.binary_dilation(hemi == 2, structure=np.ones((3, 3, 3)))
    coords = np.column_stack(np.where(boundary))
    if coords.shape[0] < 32:
        return _relabel_segmentation(output)

    centroid = coords.mean(axis=0)
    centered = coords - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, int(np.argmin(eigvals))]
    target = np.array([1.0, 0.0, 0.0])
    v = np.cross(normal, target)
    s = np.linalg.norm(v)
    c = float(np.dot(normal, target))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    rot = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2 + 1e-8))

    mask_brain = output > 0
    coords_brain = np.column_stack(np.where(mask_brain))
    if coords_brain.shape[0] == 0:
        return _relabel_segmentation(output)
    centroid_brain = coords_brain.mean(axis=0)

    parc_rot = ndimage.affine_transform(output, rot, offset=centroid - rot @ centroid, order=0)
    parc_rot = ndimage.median_filter(parc_rot, size=3)
    cx_b, _, cz_b = rot @ (centroid_brain - centroid) + centroid
    grid_x, _, grid_z = np.meshgrid(
        np.arange(parc_rot.shape[0]),
        np.arange(parc_rot.shape[1]),
        np.arange(parc_rot.shape[2]),
        indexing="ij",
    )
    dx = grid_x - cx_b
    dz = grid_z - cz_b
    radius = np.sqrt(dx**2 + dz**2)
    mask_radius = (radius >= 5) & (radius <= 100)
    theta = np.arctan2(dx, dz)
    wedge = (theta >= np.deg2rad(-30)) & (theta <= np.deg2rad(30)) & mask_radius

    new_parc = parc_rot.copy()
    new_parc[wedge & (parc_rot == 250)] = 275
    new_parc[wedge & (parc_rot == 252)] = 277
    new_parc[wedge & (parc_rot == 256)] = 279
    new_parc[wedge & (parc_rot == 251)] = 276
    new_parc[wedge & (parc_rot == 253)] = 278
    new_parc[wedge & (parc_rot == 257)] = 280

    rot_inv = rot.T
    back = ndimage.affine_transform(new_parc, rot_inv, offset=centroid - rot_inv @ centroid, order=0)
    return _relabel_segmentation(back)


def _fuse_280_t1(parcellated: np.ndarray, separated: np.ndarray, device: torch.device) -> np.ndarray:
    """Map (hemisphere, 142-class label) to 280 regions for OpenMAP-T1."""
    hemi = separated.astype(np.int16, copy=False)
    parc = parcellated.astype(np.int16, copy=False)
    lut = _get_lut("t1")
    h = np.clip(hemi, 0, lut.shape[0] - 1)
    p = np.clip(parc, 0, lut.shape[1] - 1)
    output = lut[h, p]
    output = output * np.logical_or.reduce((hemi > 0, parc == 87, parc == 138))
    return output.astype(np.int16)


def uncrop_256(volume224: np.ndarray, shift: tuple[int, int, int]) -> np.ndarray:
    padded = np.pad(volume224, [(CROP, CROP)] * 3, mode="constant", constant_values=0)
    return np.roll(padded, (-shift[0], -shift[1], -shift[2]), axis=(0, 1, 2))


def postprocessing(
    parcellated: np.ndarray,
    separated: np.ndarray,
    shift: tuple[int, int, int],
    device: torch.device | None = None,
    mode: str = "t1",
) -> np.ndarray:
    """Fuse parcellation and hemisphere results and restore the 256³ processing grid."""
    if mode == "t1c":
        labels224 = _fuse_280_t1c(parcellated, separated)
    else:
        labels224 = _fuse_280_t1(parcellated, separated, device)
    return uncrop_256(labels224, shift).astype(np.uint16)
