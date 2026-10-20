"""2.5D slice inference shared by cropping, stripping, parcellation, and hemisphere."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

DIRECTION_VALUE = {
    "coronal": -1.0,
    "sagittal": 0.0,
    "axial": 1.0,
}


def forward_plane(
    volume: np.ndarray,
    model: nn.Module,
    device: torch.device,
    *,
    n_classes: int,
    direction: float | None,
    batch_size: int = 16,
    use_amp: bool = False,
    binary: bool = False,
) -> torch.Tensor:
    """Run a 2.5D network along axis 0. `volume` is (depth, height, width)."""
    depth, height, width = volume.shape
    padded = np.pad(volume, [(1, 1), (0, 0), (0, 0)], mode="constant", constant_values=float(volume.min()))
    n_in = 3 if direction is None else 4
    if binary:
        out = torch.empty((depth, height, width), dtype=torch.float32)
    else:
        out = torch.empty((depth, n_classes, height, width), dtype=torch.float32)
    amp = use_amp and device.type == "cuda"
    with torch.inference_mode(), autocast("cuda", enabled=amp):
        for start in range(0, depth, batch_size):
            stop = min(start + batch_size, depth)
            batch = np.empty((stop - start, n_in, height, width), dtype=np.float32)
            for j, i in enumerate(range(start + 1, stop + 1)):
                batch[j, 0] = padded[i - 1]
                batch[j, 1] = padded[i]
                batch[j, 2] = padded[i + 1]
                if direction is not None:
                    batch[j, 3] = direction
            logits = model(torch.from_numpy(batch).to(device, non_blocking=True))
            if binary:
                out[start:stop] = torch.sigmoid(logits)[:, 0].float().cpu()
            else:
                out[start:stop] = torch.softmax(logits.float(), dim=1).cpu()
    return out
