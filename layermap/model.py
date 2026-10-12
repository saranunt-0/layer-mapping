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


def load_model(name_or_path: str = DEFAULT_BACKBONE, for_training: bool = False,
               device: str | torch.device | None = None) -> nn.Module:
    """Load a pretrained backbone or a fine-tuned layermap checkpoint.

    Fine-tuned checkpoints (and any model with for_training=True) use a linear
    output; the stock model keeps its ReLU so zero-shot results match upstream.
    """
    from transformers import AutoModelForDepthEstimation

    model = AutoModelForDepthEstimation.from_pretrained(name_or_path)
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
    """Layer depth at the input resolution, float32 (H, W); larger values are closer to the viewer."""
    model.eval()
    device = next(model.parameters()).device
    x = preprocess(rgb, size).to(device)
    out = model(pixel_values=x).predicted_depth
    if flip_tta:
        out = 0.5 * (out + model(pixel_values=x.flip(-1)).predicted_depth.flip(-1))
    out = torch.nn.functional.interpolate(out[:, None], size=rgb.shape[:2], mode="bilinear", align_corners=False)
    return out[0, 0].float().cpu().numpy()
