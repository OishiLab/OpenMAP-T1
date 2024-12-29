import torch

from openmap_t1.models.unet import UNet, UNetConfig
from openmap_t1.testing import OpenmapT1TestCase


class TestLoadPretrainedModel(OpenmapT1TestCase):
    def test_load_cnet(self):
        model_dir = self.PROJECT_ROOT / "data" / "OpenMAP-T1-V2.0.0" / "CNet"
        model_path = model_dir / "CNet.pth"

        config = UNetConfig(ch_in=1, ch_out=1)
        cnet = UNet(config)

        state_dict = torch.load(model_path, weights_only=True)
        cnet.load_state_dict(state_dict, strict=True)
        cnet.save_pretrained("OishiLab/OpenMAP-T1/CNet")

    def test_load_ssnet(self):
        model_dir = self.PROJECT_ROOT / "data" / "OpenMAP-T1-V2.0.0" / "SSNet"
        model_path = model_dir / "SSNet.pth"

        config = UNetConfig(ch_in=1, ch_out=1)
        ssnet = UNet(config)

        state_dict = torch.load(model_path, weights_only=True)
        ssnet.load_state_dict(state_dict, strict=True)
        ssnet.save_pretrained("OishiLab/OpenMAP-T1/SSNet")

    def test_load_pnet_coronal(self):
        model_dir = self.PROJECT_ROOT / "data" / "OpenMAP-T1-V2.0.0" / "PNet"
        model_path = model_dir / "coronal.pth"

        config = UNetConfig(ch_in=3, ch_out=142)
        pnet_c = UNet(config)

        state_dict = torch.load(model_path, weights_only=True)
        pnet_c.load_state_dict(state_dict, strict=True)
        pnet_c.save_pretrained("OishiLab/OpenMAP-T1/PNet/coronal")

    def test_load_pnet_sagittal(self):
        model_dir = self.PROJECT_ROOT / "data" / "OpenMAP-T1-V2.0.0" / "PNet"
        model_path = model_dir / "sagittal.pth"

        config = UNetConfig(ch_in=3, ch_out=142)
        pnet_s = UNet(config)

        state_dict = torch.load(model_path, weights_only=True)
        pnet_s.load_state_dict(state_dict, strict=True)
        pnet_s.save_pretrained("OishiLab/OpenMAP-T1/PNet/sagittal")

    def test_load_pnet_axial(self):
        model_dir = self.PROJECT_ROOT / "data" / "OpenMAP-T1-V2.0.0" / "PNet"
        model_path = model_dir / "axial.pth"

        config = UNetConfig(ch_in=3, ch_out=142)
        pnet_a = UNet(config)

        state_dict = torch.load(model_path, weights_only=True)
        pnet_a.load_state_dict(state_dict, strict=True)
        pnet_a.save_pretrained("OishiLab/OpenMAP-T1/PNet/axial")

    def test_load_hnet_coronal(self):
        model_dir = self.PROJECT_ROOT / "data" / "OpenMAP-T1-V2.0.0" / "HNet"
        model_path = model_dir / "coronal.pth"

        config = UNetConfig(ch_in=1, ch_out=3)
        hnet_c = UNet(config)

        state_dict = torch.load(model_path, weights_only=True)
        hnet_c.load_state_dict(state_dict, strict=True)
        hnet_c.save_pretrained("OishiLab/OpenMAP-T1/HNet/coronal")

    def test_load_hnet_axial(self):
        model_dir = self.PROJECT_ROOT / "data" / "OpenMAP-T1-V2.0.0" / "HNet"
        model_path = model_dir / "axial.pth"

        config = UNetConfig(ch_in=1, ch_out=3)
        hnet_a = UNet(config)

        state_dict = torch.load(model_path, weights_only=True)
        hnet_a.load_state_dict(state_dict, strict=True)
        hnet_a.save_pretrained("OishiLab/OpenMAP-T1/HNet/axial")
