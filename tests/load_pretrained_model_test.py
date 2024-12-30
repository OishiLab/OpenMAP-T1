from tempfile import TemporaryDirectory

import pytest
import torch
from huggingface_hub import HfApi

from openmap_t1.models.unet import UNet, UNetConfig
from openmap_t1.testing import OpenmapT1TestCase


class TestLoadPretrainedModel(OpenmapT1TestCase):
    @pytest.fixture
    def org_name(self) -> str:
        return "OishiLab"

    @pytest.fixture
    def repo_name(self) -> str:
        return "OpenMAP-T1"

    @pytest.fixture
    def repo_id(self, org_name: str, repo_name: str) -> str:
        return f"{org_name}/{repo_name}"

    @pytest.fixture
    def revision(self) -> str:
        return "v2.0.0"

    @pytest.fixture
    def hf_api(self, repo_id: str, revision: str) -> HfApi:
        api = HfApi()
        api.create_branch(repo_id=repo_id, branch=revision, exist_ok=True)
        return api

    def test_load_cnet(
        self, repo_id: str, hf_api: HfApi, revision: str, model_name: str = "CNet"
    ):
        model_dir = self.PROJECT_ROOT / "data" / "OpenMAP-T1-V2.0.0" / model_name
        model_path = model_dir / f"{model_name}.pth"

        config = UNetConfig(ch_in=1, ch_out=1)
        cnet = UNet(config)

        state_dict = torch.load(model_path, weights_only=True)
        cnet.load_state_dict(state_dict, strict=True)

        with TemporaryDirectory() as temp_dir:
            cnet.save_pretrained(temp_dir)

            hf_api.upload_folder(
                repo_id=repo_id,
                folder_path=temp_dir,
                path_in_repo=model_name,
                revision=revision,
                repo_type="model",
            )

    def test_load_ssnet(
        self, repo_id: str, hf_api: HfApi, revision: str, model_name: str = "SSNet"
    ):
        model_dir = self.PROJECT_ROOT / "data" / "OpenMAP-T1-V2.0.0" / model_name
        model_path = model_dir / f"{model_name}.pth"

        config = UNetConfig(ch_in=1, ch_out=1)
        ssnet = UNet(config)

        state_dict = torch.load(model_path, weights_only=True)
        ssnet.load_state_dict(state_dict, strict=True)

        with TemporaryDirectory() as temp_dir:
            ssnet.save_pretrained(temp_dir)

            hf_api.upload_folder(
                repo_id=repo_id,
                folder_path=temp_dir,
                path_in_repo=model_name,
                revision=revision,
                repo_type="model",
            )

    def test_load_pnet_coronal(
        self, repo_id: str, hf_api: HfApi, revision: str, model_name: str = "coronal"
    ):
        model_dir = self.PROJECT_ROOT / "data" / "OpenMAP-T1-V2.0.0" / "PNet"
        model_path = model_dir / f"{model_name}.pth"

        config = UNetConfig(ch_in=3, ch_out=142)
        pnet_c = UNet(config)

        state_dict = torch.load(model_path, weights_only=True)
        pnet_c.load_state_dict(state_dict, strict=True)

        with TemporaryDirectory() as temp_dir:
            pnet_c.save_pretrained(temp_dir)

            hf_api.upload_folder(
                repo_id=repo_id,
                folder_path=temp_dir,
                path_in_repo=f"PNet/{model_name}",
                revision=revision,
                repo_type="model",
            )

    def test_load_pnet_sagittal(
        self, repo_id: str, hf_api: HfApi, revision: str, model_name: str = "sagittal"
    ):
        model_dir = self.PROJECT_ROOT / "data" / "OpenMAP-T1-V2.0.0" / "PNet"
        model_path = model_dir / f"{model_name}.pth"
        save_path = f"{repo_id}/PNet/{model_name}"

        config = UNetConfig(ch_in=3, ch_out=142)
        pnet_s = UNet(config)

        state_dict = torch.load(model_path, weights_only=True)
        pnet_s.load_state_dict(state_dict, strict=True)

        with TemporaryDirectory() as temp_dir:
            pnet_s.save_pretrained(temp_dir)

            hf_api.upload_folder(
                repo_id=repo_id,
                folder_path=temp_dir,
                path_in_repo=f"PNet/{model_name}",
                revision=revision,
                repo_type="model",
            )

    def test_load_pnet_axial(
        self, repo_id: str, hf_api: HfApi, revision: str, model_name: str = "axial"
    ):
        model_dir = self.PROJECT_ROOT / "data" / "OpenMAP-T1-V2.0.0" / "PNet"
        model_path = model_dir / f"{model_name}.pth"

        config = UNetConfig(ch_in=3, ch_out=142)
        pnet_a = UNet(config)

        state_dict = torch.load(model_path, weights_only=True)
        pnet_a.load_state_dict(state_dict, strict=True)

        with TemporaryDirectory() as temp_dir:
            pnet_a.save_pretrained(temp_dir)

            hf_api.upload_folder(
                repo_id=repo_id,
                folder_path=temp_dir,
                path_in_repo=f"PNet/{model_name}",
                revision=revision,
                repo_type="model",
            )

    def test_load_hnet_coronal(
        self, repo_id: str, hf_api: HfApi, revision: str, model_name: str = "coronal"
    ):
        model_dir = self.PROJECT_ROOT / "data" / "OpenMAP-T1-V2.0.0" / "HNet"
        model_path = model_dir / f"{model_name}.pth"

        config = UNetConfig(ch_in=1, ch_out=3)
        hnet_c = UNet(config)

        state_dict = torch.load(model_path, weights_only=True)
        hnet_c.load_state_dict(state_dict, strict=True)

        with TemporaryDirectory() as temp_dir:
            hnet_c.save_pretrained(temp_dir)

            hf_api.upload_folder(
                repo_id=repo_id,
                folder_path=temp_dir,
                path_in_repo=f"HNet/{model_name}",
                revision=revision,
                repo_type="model",
            )

    def test_load_hnet_axial(
        self, repo_id: str, hf_api: HfApi, revision: str, model_name: str = "axial"
    ):
        model_dir = self.PROJECT_ROOT / "data" / "OpenMAP-T1-V2.0.0" / "HNet"
        model_path = model_dir / f"{model_name}.pth"

        config = UNetConfig(ch_in=1, ch_out=3)
        hnet_a = UNet(config)

        state_dict = torch.load(model_path, weights_only=True)
        hnet_a.load_state_dict(state_dict, strict=True)

        with TemporaryDirectory() as temp_dir:
            hnet_a.save_pretrained(temp_dir)

            hf_api.upload_folder(
                repo_id=repo_id,
                folder_path=temp_dir,
                path_in_repo=f"HNet/{model_name}",
                revision=revision,
                repo_type="model",
            )
