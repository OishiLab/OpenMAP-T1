from dataclasses import dataclass, fields

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
        cnet=UNet.from_pretrained("OishiLab/OpenMAP-T1/CNet"),
        ssnet=UNet.from_pretrained("OishiLab/OpenMAP-T1/SSNet"),
        pnet_c=UNet.from_pretrained("OishiLab/OpenMAP-T1/PNet/coronal"),
        pnet_s=UNet.from_pretrained("OishiLab/OpenMAP-T1/PNet/sagittal"),
        pnet_a=UNet.from_pretrained("OishiLab/OpenMAP-T1/PNet/axial"),
        hnet_c=UNet.from_pretrained("OishiLab/OpenMAP-T1/HNet/coronal"),
        hnet_a=UNet.from_pretrained("OishiLab/OpenMAP-T1/HNet/axial"),
    )
    models.to(device)
    models.eval()
    return models
