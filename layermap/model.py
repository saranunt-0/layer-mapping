"""Depth Anything V2 Small (Apache-2.0) as a layer-index predictor."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

# Only the Small checkpoint of Depth Anything V2 is Apache-2.0; Base/Large/Giant are CC-BY-NC-4.0.
DEFAULT_BACKBONE = "depth-anything/Depth-Anything-V2-Small-hf"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
PATCH = 14


def make_linear_head(model: nn.Module) -> nn.Module:
    """Drop the final ReLU so regions predicted at the back still receive gradients."""
    model.head.activation2 = nn.Identity()
    model.config.layermap_linear_head = True
    return model


def make_multi_head(model: nn.Module, n: int) -> nn.Module:
    """Give the head n output channels (one per training target), each initialised from the pretrained one."""
    old = model.head.conv3
    if old.out_channels != n:
        new = nn.Conv2d(old.in_channels, n, kernel_size=1)
        with torch.no_grad():
            new.weight.copy_(old.weight[:1].repeat(n, 1, 1, 1))
            new.bias.copy_(old.bias[:1].repeat(n))
        model.head.conv3 = new
    model.config.layermap_out_channels = n
    return model


def load_model(name_or_path: str = DEFAULT_BACKBONE, for_training: bool = False,
               device: str | torch.device | None = None, out_channels: int | None = None) -> nn.Module:
    """Load a pretrained backbone or a fine-tuned layermap checkpoint.

    Fine-tuned checkpoints (and any model with for_training=True) use a linear
    output; the stock model keeps its ReLU so zero-shot results match upstream.
    `out_channels` > 1 adds output channels for multi-target training; checkpoints
    remember their channel count.
    """
    from transformers import AutoConfig, AutoModelForDepthEstimation

    config = AutoConfig.from_pretrained(name_or_path)
    saved_channels = getattr(config, "layermap_out_channels", 1)
    if saved_channels > 1:
        # The stock class builds a 1-channel output conv: load everything else the normal way, then
        # widen the head and restore its saved weights.
        from safetensors import safe_open

        from transformers.utils import logging as hf_logging

        level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()  # the size-mismatch report for conv3 is expected here
        try:
            model = AutoModelForDepthEstimation.from_pretrained(name_or_path, ignore_mismatched_sizes=True)
        finally:
            hf_logging.set_verbosity(level)
        make_multi_head(model, saved_channels)
        conv3 = {}
        for f in Path(name_or_path).glob("*.safetensors"):
            with safe_open(str(f), framework="pt") as sf:
                conv3.update({k.rsplit(".", 1)[1]: sf.get_tensor(k) for k in sf.keys() if k.startswith("head.conv3.")})
        if set(conv3) != {"weight", "bias"}:
            raise RuntimeError(f"no head.conv3 weights in {name_or_path}")
        with torch.no_grad():
            model.head.conv3.weight.copy_(conv3["weight"])
            model.head.conv3.bias.copy_(conv3["bias"])
    else:
        model = AutoModelForDepthEstimation.from_pretrained(name_or_path)
    if out_channels and out_channels != saved_channels:
        make_multi_head(model, out_channels)
    if for_training or getattr(model.config, "layermap_linear_head", False):
        make_linear_head(model)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device)


def save_model(model: nn.Module, out_dir: str | Path) -> None:
    model.save_pretrained(str(out_dir))


def preprocess(rgb: np.ndarray, size: int = 518) -> torch.Tensor:
    """uint8 RGB (H, W, 3) -> normalised (1, 3, h, w) with the long side ~size and h, w multiples of 14."""
    h, w = rgb.shape[:2]
    s = size / max(h, w)
    nh, nw = (max(PATCH, int(round(d * s / PATCH)) * PATCH) for d in (h, w))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    x = cv2.resize(rgb, (nw, nh), interpolation=interp).astype(np.float32) / 255
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


@torch.no_grad()
def predict(model: nn.Module, rgb: np.ndarray, size: int = 518, flip_tta: bool = True) -> np.ndarray:
    """Prediction at the input resolution; larger values are closer to the viewer.

    float32 (H, W) for single-output models, (C, H, W) for multi-target models
    (channels in the order of the training targets).
    """
    model.eval()
    device = next(model.parameters()).device
    x = preprocess(rgb, size).to(device)
    out = model(pixel_values=x).predicted_depth
    if flip_tta:
        out = 0.5 * (out + model(pixel_values=x.flip(-1)).predicted_depth.flip(-1))
    if out.dim() == 3:
        out = out[:, None]
    out = torch.nn.functional.interpolate(out, size=rgb.shape[:2], mode="bilinear", align_corners=False)
    out = out[0].float().cpu().numpy()
    return out[0] if out.shape[0] == 1 else out
