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


# --------------------------------------------------------------------------- #
# Hybrid (layered + continuous) depth
# --------------------------------------------------------------------------- #


def within_layer_rho(pred, depth_gt, labels, min_frac: float = 0.01, min_std: float = 1e-3,
                     max_px: int = 4000, seed: int = 0) -> float:
    """Area-weighted Spearman correlation between prediction and true depth inside each layer whose true
    depth actually varies (roads, plains, hills).

    1 = continuous and correctly oriented inside layers; about 0 = flat inside layers (discrete maps).
    """
    from scipy.stats import rankdata

    p, d, lab = (np.asarray(a) for a in (pred, depth_gt, labels))
    rng = np.random.default_rng(seed)
    total, acc = 0.0, 0.0
    for k in np.unique(lab):
        sel = np.flatnonzero(lab.ravel() == k)
        if sel.size < min_frac * lab.size:
            continue
        if sel.size > max_px:
            sel = rng.choice(sel, max_px, replace=False)
        dv, pv = d.ravel()[sel], p.ravel()[sel]
        if dv.std() < min_std:
            continue
        rp, rd = rankdata(pv), rankdata(dv)
        rho = 0.0 if rp.std() == 0 else float(np.corrcoef(rp, rd)[0, 1])
        total += sel.size
        acc += rho * sel.size
    return acc / total if total else float("nan")


def edge_accuracy(pred, labels, n_pairs: int = 100_000, radius: int = 3, seed: int = 0) -> float:
    """Share of nearby pixel pairs in different layers that the prediction orders like the paint order."""
    p, lab = _as_tensor(pred), _as_tensor(labels)
    h, w = lab.shape
    g = torch.Generator().manual_seed(seed)
    iy, ix = torch.randint(0, h, (n_pairs,), generator=g), torch.randint(0, w, (n_pairs,), generator=g)
    off = torch.randint(-radius, radius + 1, (2, n_pairs), generator=g)
    jy, jx = (iy + off[0]).clamp(0, h - 1), (ix + off[1]).clamp(0, w - 1)
    st = torch.sign(lab[iy, ix] - lab[jy, jx])
    keep = st != 0
    if not keep.any():
        return float("nan")
    return (torch.sign(p[iy, ix] - p[jy, jx])[keep] == st[keep]).float().mean().item()


def hybrid_metrics(pred, target, depth_gt, labels) -> dict:
    """Scores for a layered-continuous depth prediction.

    The first two do not depend on how the hybrid target is defined:
    acc_edge    pairs across layer boundaries ordered like the paint order (artistic layering)
    rho_within  continuity and direction of depth inside layers, vs true depth (roads, plains)
    The last two compare with the hybrid target (`fusion.hybrid_target`):
    acc_hard    pairs where the upper pixel is in front
    ssi_mae     overall distance after scale/shift normalisation
    """
    return {
        "acc_edge": edge_accuracy(pred, labels),
        "rho_within": within_layer_rho(pred, depth_gt, labels),
        "acc_hard": ordinal_accuracy(pred, target, 50_000, mode="hard"),
        "ssi_mae": ssi_mae(pred, target),
    }
