import os

import torch

from utils.network import UNet

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(opt, device):
    """
    Load and initialize the pretrained neural network models required for the OpenMAP-T1 pipeline.

    This function loads four U-Net–based models from the specified pretrained model directory.
    Each model is moved to the target device (CPU, CUDA, or MPS) and set to evaluation mode.

    Models loaded:
        1. **CNet (Cropping Network)** — Performs face cropping and brain localization.
        2. **SSNet (Skull Stripping Network)** — Removes non-brain tissues from MRI scans.
        3. **PNet (Parcellation Network)** — Predicts fine-grained anatomical labels across 142 regions.
        4. **HNet (Hemisphere Network)** — Segments the brain into hemispheric masks (left/right/other).

    Args:
        opt (argparse.Namespace): Parsed command-line arguments containing the pretrained model directory path (`opt.m`).
        device (torch.device): Target device on which to load models (e.g., `torch.device('cuda')`).

    Returns:
        tuple:
            A tuple containing four initialized and evaluation-ready models:
            (cnet, ssnet, pnet, hnet).
    """
    model_dir = opt.m  # Base directory where pretrained model weights are stored

    # --------------------------
    # Load CNet (Cropping Network)
    # --------------------------
    # Input: 3-channel (neighboring slices), Output: 1-channel binary mask
    cnet = UNet(3, 1)
    cnet.load_state_dict(torch.load(os.path.join(model_dir, "CNet", "CNet.pth"), weights_only=True))
    cnet.to(device)
    cnet.eval()

    # ------------------------------
    # Load SSNet (Skull Stripping Network)
    # ------------------------------
    # Input: 3-channel (neighboring slices), Output: 1-channel brain mask
    ssnet = UNet(3, 1)
    ssnet.load_state_dict(torch.load(os.path.join(model_dir, "SSNet", "SSNet.pth"), weights_only=True))
    ssnet.to(device)
    ssnet.eval()

    # -----------------------------
    # Load PNet (Parcellation Network)
    # -----------------------------
    # Input: 4 channels (multi-modal or augmented context), Output: 142 anatomical regions
    pnet = UNet(4, 142)
    pnet.load_state_dict(torch.load(os.path.join(model_dir, "PNet", "PNet.pth"), weights_only=True))
    pnet.to(device)
    pnet.eval()

    # -----------------------------
    # Load HNet (Hemisphere Network)
    # -----------------------------
    # Input: 3 channels, Output: 3-class hemisphere mask (left, right, background)
    hnet = UNet(3, 3)
    hnet.load_state_dict(torch.load(os.path.join(model_dir, "HNet", "HNet.pth"), weights_only=True))
    hnet.to(device)
    hnet.eval()

    # Return all loaded, device-initialized, and evaluation-ready models
    return cnet, ssnet, pnet, hnet
