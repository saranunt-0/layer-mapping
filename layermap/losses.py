"""Losses for layer-index regression.

The network predicts one scalar per pixel and only the ordering of those
values matters, so every term here is invariant to scaling and shifting the
prediction.

References:
    Ranftl et al., "Towards Robust Monocular Depth Estimation: Mixing Datasets
    for Zero-shot Cross-dataset Transfer" (MiDaS), TPAMI 2020: scale/shift
    invariant normalisation and multi-scale gradient matching.
    Chen et al., "Single-Image Depth Perception in the Wild", NeurIPS 2016:
    pairwise ordinal (ranking) loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    ssi: float = 1.0
    grad: float = 0.5
    rank: float = 0.25


def normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-image shift by the median and scale by the mean absolute deviation. x: (B, H, W)."""
    flat = x.reshape(x.shape[0], -1)
    med = flat.median(dim=1, keepdim=True).values
    scale = (flat - med).abs().mean(dim=1, keepdim=True).clamp_min(eps)
    return ((flat - med) / scale).reshape_as(x)


def ssi_l1(pred_n: torch.Tensor, target_n: torch.Tensor) -> torch.Tensor:
    return (pred_n - target_n).abs().mean()


def gradient_matching(pred_n: torch.Tensor, target_n: torch.Tensor, scales: int = 4) -> torch.Tensor:
    """L1 on the image gradients of the residual, over a pyramid of scales."""
    loss = pred_n.new_zeros(())
    d = pred_n - target_n
    for _ in range(scales):
        loss = loss + (d[:, :, 1:] - d[:, :, :-1]).abs().mean() + (d[:, 1:, :] - d[:, :-1, :]).abs().mean()
        d = d[:, ::2, ::2]
    return loss / scales


def pairwise_ranking(pred_n: torch.Tensor, target: torch.Tensor, n_pairs: int = 8192,
                     local_radius: int = 24) -> torch.Tensor:
    """Logistic ranking loss on pixel pairs in different layers, L2 on pairs in the same layer.

    Half the pairs are drawn uniformly over the image; the other half are drawn
    within `local_radius` pixels of each other, so many straddle a layer boundary.
    """
    b, h, w = pred_n.shape
    dev = pred_n.device
    iy = torch.randint(0, h, (b, n_pairs), device=dev)
    ix = torch.randint(0, w, (b, n_pairs), device=dev)
    half = n_pairs // 2
    jy = torch.randint(0, h, (b, n_pairs), device=dev)
    jx = torch.randint(0, w, (b, n_pairs), device=dev)
    off = torch.randint(-local_radius, local_radius + 1, (2, b, n_pairs - half), device=dev)
    jy[:, half:] = (iy[:, half:] + off[0]).clamp(0, h - 1)
    jx[:, half:] = (ix[:, half:] + off[1]).clamp(0, w - 1)

    flat_p, flat_t = pred_n.reshape(b, -1), target.reshape(b, -1).to(pred_n.dtype)
    i, j = iy * w + ix, jy * w + jx
    dp = flat_p.gather(1, i) - flat_p.gather(1, j)
    sign = torch.sign(flat_t.gather(1, i) - flat_t.gather(1, j))
    ordered = sign != 0
    loss = torch.where(ordered, F.softplus(-sign * dp), dp.pow(2))
    return loss.mean()


def layer_loss(pred: torch.Tensor, target: torch.Tensor, weights: LossWeights = LossWeights()):
    """pred: (B, H, W) raw network output; target: (B, H, W) layer index, larger = closer to the viewer."""
    pred = pred.float()
    target = target.float()
    pred_n, target_n = normalize(pred), normalize(target)
    terms = {
        "ssi": ssi_l1(pred_n, target_n),
        "grad": gradient_matching(pred_n, target_n),
        "rank": pairwise_ranking(pred_n, target),
    }
    total = weights.ssi * terms["ssi"] + weights.grad * terms["grad"] + weights.rank * terms["rank"]
    return total, {k: v.detach() for k, v in terms.items()}
