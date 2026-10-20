"""Load pretrained weights for OpenMAP-T1 (142-class PNet / 3ch HNet) and OpenMAP-T1c (139-class PNet / 4ch HNet)."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from utils.models.network import UNet

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]

MODE_CHOICES = ("auto", "t1", "t1c")


def _resolve_weight(model_dir: Path, phase: str, folder: str, filename: str) -> Path:
    direct = model_dir / folder / filename
    if direct.is_file():
        return direct

    models = model_dir / phase / "models"
    final = models / "model.pth"
    if final.is_file():
        return final
    checkpoint = models / "checkpoint.pt"
    if checkpoint.is_file():
        return checkpoint
    epochs = sorted(models.glob("epoch_*.pth"))
    if epochs:
        return epochs[-1]

    # For CNet, allow falling back to the shared CNet in model/OpenMAP-T1c or model/V3
    # when loading from a training checkpoint folder (e.g. 20260821/).
    if folder == "CNet":
        for fallback in (
            PROJECT_ROOT / "model" / "OpenMAP-T1c" / "CNet" / "CNet.pth",
            PROJECT_ROOT / "model" / "V3" / "CNet" / "CNet.pth",
        ):
            if fallback.is_file():
                return fallback

    raise FileNotFoundError(
        f"No weights for {phase}. Expected {direct} or {models / 'model.pth'}."
    )


def _read_state_dict(path: Path, device: torch.device) -> dict:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
        return ckpt["model"]
    return ckpt


def _infer_channels(state_dict: dict) -> tuple[int, int]:
    ch_in = int(state_dict["econv0.weight"].shape[1])
    ch_out = int(state_dict["dconv0.weight"].shape[0])
    return ch_in, ch_out


def _detect_mode_from_pnet(pnet_out_channels: int) -> str:
    if pnet_out_channels == 142:
        return "t1"
    if pnet_out_channels == 139:
        return "t1c"
    raise ValueError(
        f"Unsupported PNet output channels: {pnet_out_channels} (expected 142 for OpenMAP-T1 or 139 for OpenMAP-T1c)."
    )


def _build_and_load(path: Path, state_dict: dict, expected_in: int, expected_out: int, device: torch.device) -> nn.Module:
    ch_in, ch_out = _infer_channels(state_dict)
    if (ch_in, ch_out) != (expected_in, expected_out):
        raise ValueError(
            f"Weight shape mismatch for {path}: got (in={ch_in}, out={ch_out}), "
            f"expected (in={expected_in}, out={expected_out})."
        )
    model = UNet(ch_in, ch_out)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_model(opt, device: torch.device):
    """
    Load and initialize the pretrained neural network models for OpenMAP-T1 or OpenMAP-T1c.

    Automatically detects 't1' (PNet 142 classes, HNet 3 input channels) vs
    't1c' (PNet 139 classes, HNet 4 input channels) from the model weights and
    stores the resolved mode on `opt.mode`.

    Returns:
        tuple: (cnet, ssnet, pnet, hnet)
    """
    model_dir = Path(os.path.abspath(opt.m))
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model folder does not exist: {model_dir}")

    cnet_path = _resolve_weight(model_dir, "cropping", "CNet", "CNet.pth")
    ssnet_path = _resolve_weight(model_dir, "stripping", "SSNet", "SSNet.pth")
    pnet_path = _resolve_weight(model_dir, "parcellation", "PNet", "PNet.pth")
    hnet_path = _resolve_weight(model_dir, "hemisphere", "HNet", "HNet.pth")

    cnet_sd = _read_state_dict(cnet_path, device)
    ssnet_sd = _read_state_dict(ssnet_path, device)
    pnet_sd = _read_state_dict(pnet_path, device)
    hnet_sd = _read_state_dict(hnet_path, device)

    _, pnet_out = _infer_channels(pnet_sd)
    detected_mode = _detect_mode_from_pnet(pnet_out)

    requested_mode = getattr(opt, "mode", "auto") or "auto"
    if requested_mode != "auto" and requested_mode != detected_mode:
        raise ValueError(
            f"Requested --mode {requested_mode}, but weights in {model_dir} match '{detected_mode}' "
            f"(PNet out_channels={pnet_out})."
        )
    opt.mode = detected_mode

    expected_pnet_out = 142 if detected_mode == "t1" else 139
    expected_hnet_in = 3 if detected_mode == "t1" else 4

    cnet = _build_and_load(cnet_path, cnet_sd, 3, 1, device)
    ssnet = _build_and_load(ssnet_path, ssnet_sd, 3, 1, device)
    pnet = _build_and_load(pnet_path, pnet_sd, 4, expected_pnet_out, device)
    hnet = _build_and_load(hnet_path, hnet_sd, expected_hnet_in, 3, device)

    label = "OpenMAP-T1c" if detected_mode == "t1c" else "OpenMAP-T1"
    print(f"Pipeline mode: {label} ({detected_mode})")
    print(f"  CNet : {cnet_path}")
    print(f"  SSNet: {ssnet_path}")
    print(f"  PNet : {pnet_path}")
    print(f"  HNet : {hnet_path}")

    return cnet, ssnet, pnet, hnet
