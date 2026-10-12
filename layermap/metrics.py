"""Evaluation metrics for predicted layer orderings.

Landscapes are strongly ordered top to bottom, so "lower in the frame = closer"
already gets most random pixel pairs right. Report all three pair sets:

    all    uniform random pairs in different layers
    local  pairs within a few pixels of each other, i.e. across layer boundaries
    hard   pairs where the upper pixel is in front, which the vertical prior always gets wrong

Only `local` and `hard` show whether a model has learned occlusion.
"""

from __future__ import annotations

import numpy as np
import torch

from .losses import normalize

PAIR_MODES = ("all", "local", "hard")


def _as_tensor(x) -> torch.Tensor:
    return (x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))).float().cpu()


def ordinal_accuracy(pred, target, n_pairs: int = 100_000, seed: int = 0, mode: str = "all",
                     radius: int = 8) -> float:
    """Share of pixel pairs in different layers whose order the prediction gets right.

    For mode="all" this is 1 - WHDR (Bell et al. 2014) with uniform weights.
    A tie in the prediction counts as wrong. pred and target are (H, W).
    """
    if mode not in PAIR_MODES:
        raise ValueError(f"mode must be one of {PAIR_MODES}")
    p, t = _as_tensor(pred), _as_tensor(target)
    h, w = t.shape
    g = torch.Generator().manual_seed(seed)
    iy, ix = torch.randint(0, h, (n_pairs,), generator=g), torch.randint(0, w, (n_pairs,), generator=g)
    if mode == "local":
        off = torch.randint(-radius, radius + 1, (2, n_pairs), generator=g)
        jy, jx = (iy + off[0]).clamp(0, h - 1), (ix + off[1]).clamp(0, w - 1)
    else:
        jy, jx = torch.randint(0, h, (n_pairs,), generator=g), torch.randint(0, w, (n_pairs,), generator=g)
    st = torch.sign(t[iy, ix] - t[jy, jx])
    keep = st != 0
    if mode == "hard":
        keep &= st == -torch.sign((iy - jy).float())
    if not keep.any():
        return float("nan")
    return (torch.sign(p[iy, ix] - p[jy, jx])[keep] == st[keep]).float().mean().item()


def ssi_mae(pred, target) -> float:
    """Mean absolute error after median / mean-absolute-deviation normalisation of both maps."""
    p, t = _as_tensor(pred)[None], _as_tensor(target)[None]
    return (normalize(p) - normalize(t)).abs().mean().item()


def all_metrics(pred, target, n_pairs: int = 20_000) -> dict:
    out = {f"acc_{m}": ordinal_accuracy(pred, target, n_pairs, mode=m) for m in PAIR_MODES}
    out["ssi_mae"] = ssi_mae(pred, target)
    return out


def vertical_prior(shape: tuple[int, int]) -> np.ndarray:
    """Baseline "depth" that only says lower in the frame = closer."""
    h, w = shape
    return np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
