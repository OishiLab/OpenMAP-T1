from dataclasses import dataclass, fields
from typing import Optional

import torch
from loguru import logger

from openmap_t1.models.unet import UNet


@dataclass
class UNetModels(object):
    cnet: UNet
    ssnet: UNet
    pnet_c: UNet
    pnet_s: UNet
    pnet_a: UNet
    hnet_c: UNet
    hnet_a: UNet

    def to(self, device: torch.device) -> None:
        for field in fields(self):
            logger.debug(f"Moving {field.name} to {device}")
            model = getattr(self, field.name)
            model.to(device)

    def eval(self) -> None:
        for field in fields(self):
            logger.debug(f"Setting {field.name} to evaluation mode")
            model = getattr(self, field.name)
            model.eval()


def get_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_models(device: torch.device) -> UNetModels:
    """This function loads multiple pre-trained models and sets them to evaluation mode.

    The models loaded are:
    1. CNet: A U-Net model for some specific task.
    2. SSNet: Another U-Net model for a different task.
    3. PNet coronal: A U-Net model for coronal plane predictions.
    4. PNet sagittal: A U-Net model for sagittal plane predictions.
    5. PNet axial: A U-Net model for axial plane predictions.
    6. HNet coronal: A U-Net model for coronal plane predictions with different input/output channels.
    7. HNet axial: A U-Net model for axial plane predictions with different input/output channels.

    Parameters:
        device (torch.device): The device on which to load the models (CPU or GPU).

    Returns:
        UNetModels: A dataclass containing all the loaded models
    """

    models = UNetModels(
        cnet=UNet.from_pretrained(
            "OishiLab/OpenMAP-T1",
            subfolder="CNet",
            revision="v2.0.0",
        ),
        ssnet=UNet.from_pretrained(
            "OishiLab/OpenMAP-T1",
            subfolder="SSNet",
            revision="v2.0.0",
        ),
        pnet_c=UNet.from_pretrained(
            "OishiLab/OpenMAP-T1",
            subfolder="PNet/coronal",
            revision="v2.0.0",
        ),
        pnet_s=UNet.from_pretrained(
            "OishiLab/OpenMAP-T1",
            subfolder="PNet/sagittal",
            revision="v2.0.0",
        ),
        pnet_a=UNet.from_pretrained(
            "OishiLab/OpenMAP-T1",
            subfolder="PNet/axial",
            revision="v2.0.0",
        ),
        hnet_c=UNet.from_pretrained(
            "OishiLab/OpenMAP-T1",
            subfolder="HNet/coronal",
            revision="v2.0.0",
        ),
        hnet_a=UNet.from_pretrained(
            "OishiLab/OpenMAP-T1",
            subfolder="HNet/axial",
            revision="v2.0.0",
        ),
    )
    models.to(device)
    models.eval()
    return models
