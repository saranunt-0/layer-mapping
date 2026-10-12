"""Combine a discrete layer map (e.g. Illustrator's Depth) with a continuous depth map (e.g. Depth Anything).

Conventions: `depth` is relative inverse depth (larger = closer) with arbitrary scale and shift;
`labels` are integer layer indices (larger = painted later = in front). Every fuser returns a float32 map
normalised to [0, 1] (1 = closest).

Methods:
    multiply  n(labels) * n(depth)                         (pixel-wise blend, no ordering guarantee)
    blend     a * n(labels) + (1 - a) * n(depth)           (pixel-wise blend, no ordering guarantee)
    stacked   layer index + depth rescaled inside each layer (strict global layer order)
    ordered   depth + one offset per layer, the smallest that puts every layer in front of the layers it
              touches by `margin` (keeps the depth shape inside layers, orders only where layers meet)
    poisson   least-squares integration of depth gradients inside layers and ordered jumps across them
    guided    guided filter of depth with the layer map as guide (snaps edges, does not reorder)

`masked_depth` re-runs the depth model once per layer with the layers in front of it removed, so each
layer's depth is predicted without its occluders, then aligns everything to the full-image prediction.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np


def normalize01(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Robust min-max to [0, 1] using percentiles."""
    x = x.astype(np.float32)
    a, b = np.percentile(x, [lo, hi])
    return np.clip((x - a) / max(b - a, 1e-6), 0, 1).astype(np.float32)


def _dense_labels(labels: np.ndarray) -> np.ndarray:
    _, inv = np.unique(labels, return_inverse=True)
    return inv.reshape(labels.shape)


# --------------------------------------------------------------------------- #
# Pixel-wise blends
# --------------------------------------------------------------------------- #


def fuse_multiply(depth, labels, floor: float = 0.1):
    l = _dense_labels(labels).astype(np.float32)
    nl = floor + (1 - floor) * l / max(1.0, l.max())
    nd = floor + (1 - floor) * normalize01(depth)
    return normalize01(nl * nd, 0, 100)


def fuse_blend(depth, labels, alpha: float = 0.5):
    l = _dense_labels(labels).astype(np.float32)
    return normalize01(alpha * l / max(1.0, l.max()) + (1 - alpha) * normalize01(depth), 0, 100)


# --------------------------------------------------------------------------- #
# Layer-structured fusions
# --------------------------------------------------------------------------- #


def fuse_stacked(depth, labels, within: float = 0.8):
    """Lexicographic order: the layer decides, depth only orders pixels inside a layer."""
    l = _dense_labels(labels)
    d = normalize01(depth)
    out = l.astype(np.float32)
    for k in range(l.max() + 1):
        sel = l == k
        v = d[sel]
        lo, hi = np.percentile(v, [1, 99]) if v.size > 10 else (v.min(), v.max())
        out[sel] += within * np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    return normalize01(out, 0, 100)


def boundary_pairs(labels: np.ndarray, radii=(1, 3)):
    """Flat indices (front, back) of pixel pairs that straddle a layer boundary."""
    h, w = labels.shape
    idx = np.arange(h * w).reshape(h, w)
    fronts, backs = [], []
    for r in radii:
        for dy, dx in ((0, r), (r, 0), (r, r), (r, -r)):
            ys, ye = 0, h - dy
            xs, xe = max(0, -dx), w - max(0, dx)
            a, b = labels[ys:ye, xs:xe], labels[ys + dy:ye + dy, xs + dx:xe + dx]
            ia, ib = idx[ys:ye, xs:xe], idx[ys + dy:ye + dy, xs + dx:xe + dx]
            m = a != b
            front_is_a = a[m] > b[m]
            fronts.append(np.where(front_is_a, ia[m], ib[m]))
            backs.append(np.where(front_is_a, ib[m], ia[m]))
    return np.concatenate(fronts), np.concatenate(backs)


def layer_offsets(depth: np.ndarray, labels: np.ndarray, margin: float = 0.02, quantile: float = 0.1,
                  min_pairs: int = 20) -> np.ndarray:
    """Smallest non-negative per-layer offsets b such that, along every boundary between layers f > k,
    the `quantile` of (depth_f + b_f) - (depth_k + b_k) is at least `margin`.

    Constraints only point from back to front, so processing layers back to front gives the exact
    minimal solution (a longest-path problem on the layer adjacency graph).
    """
    l = _dense_labels(labels)
    n = l.max() + 1
    f_idx, b_idx = boundary_pairs(l)
    lf, lb = l.ravel()[f_idx], l.ravel()[b_idx]
    diff = depth.ravel()[f_idx] - depth.ravel()[b_idx]
    order = np.lexsort((lb, lf))
    lf, lb, diff = lf[order], lb[order], diff[order]
    need = {}
    if lf.size:
        keys = lf.astype(np.int64) * n + lb
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        ends = np.r_[starts[1:], keys.size]
        for s, e in zip(starts, ends):
            if e - s >= min_pairs:
                need.setdefault(int(lf[s]), []).append((int(lb[s]), margin - float(np.quantile(diff[s:e], quantile))))
    b = np.zeros(n, np.float32)
    for f in range(n):
        for k, c in need.get(f, []):
            b[f] = max(b[f], b[k] + c)
    return b


def fuse_ordered(depth, labels, margin: float = 0.02, quantile: float = 0.1):
    """Keep the depth shape inside every layer; shift layers forward just enough to sit in front of
    the layers they touch."""
    l = _dense_labels(labels)
    d = normalize01(depth)
    return normalize01(d + layer_offsets(d, l, margin, quantile)[l], 0, 100)


def fuse_poisson(depth, labels, margin: float = 0.02, data_weight: float = 1e-3, max_side: int = 512):
    """Screened Poisson: match depth gradients inside layers, require front-minus-back >= margin
    (or the depth's own jump if larger) across boundaries, weakly anchored to the depth values."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    H, W = depth.shape
    s = min(1.0, max_side / max(H, W))
    d = normalize01(depth)
    l = _dense_labels(labels)
    if s < 1:
        d = cv2.resize(d, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        l = cv2.resize(l.astype(np.int32), (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST)
    h, w = d.shape
    n = h * w
    idx = np.arange(n).reshape(h, w)
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for p, q in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
        p, q = p.ravel(), q.ravel()
        lp, lq = l.ravel()[p], l.ravel()[q]
        dp, dq = d.ravel()[p], d.ravel()[q]
        # Orient every equation as (first - second); across boundaries "first" is the front pixel.
        swap = lq > lp
        a, b = np.where(swap, q, p), np.where(swap, p, q)
        da, db = np.where(swap, dq, dp), np.where(swap, dp, dq)
        target = da - db
        cross = lp != lq
        target[cross] = np.maximum(target[cross], margin)
        m = len(a)
        eq = np.arange(r, r + m)
        rows += [eq, eq]
        cols += [a, b]
        vals += [np.ones(m), -np.ones(m)]
        rhs.append(target)
        r += m
    eq = np.arange(r, r + n)
    rows.append(eq)
    cols.append(np.arange(n))
    vals.append(np.full(n, np.sqrt(data_weight)))
    rhs.append(np.sqrt(data_weight) * d.ravel())
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r + n, n))
    x = spsolve((A.T @ A).tocsc(), A.T @ np.concatenate(rhs)).reshape(h, w).astype(np.float32)
    if s < 1:
        x = _upsample_by_layer(x, l, labels, (H, W))
    return normalize01(x, 0, 100)


def _upsample_by_layer(x_small, l_small, labels_full, shape):
    """Upsample without blending values across layer boundaries (per-layer normalised convolution)."""
    H, W = shape
    l_full = _dense_labels(labels_full)
    out = cv2.resize(x_small, (W, H), interpolation=cv2.INTER_LINEAR)
    for k in np.unique(l_small):
        m = (l_small == k).astype(np.float32)
        num = cv2.resize(x_small * m, (W, H), interpolation=cv2.INTER_LINEAR)
        den = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        sel = (l_full == k) & (den > 1e-3)
        out[sel] = num[sel] / den[sel]
    return out


def fuse_guided(depth, labels, radius: int = 8, eps: float = 1e-3):
    """Guided filter (He et al. 2010) with the layer map as guide: edges follow layer boundaries."""
    l = _dense_labels(labels).astype(np.float32)
    guide = l / max(1.0, l.max())
    p = normalize01(depth)
    k = (2 * radius + 1, 2 * radius + 1)
    mean_i, mean_p = cv2.blur(guide, k), cv2.blur(p, k)
    cov_ip = cv2.blur(guide * p, k) - mean_i * mean_p
    var_i = cv2.blur(guide * guide, k) - mean_i ** 2
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return normalize01(cv2.blur(a, k) * guide + cv2.blur(b, k), 0, 100)


FUSERS: dict[str, Callable] = {
    "multiply": fuse_multiply,
    "blend": fuse_blend,
    "stacked": fuse_stacked,
    "ordered": fuse_ordered,
    "poisson": fuse_poisson,
    "guided": fuse_guided,
}


def hybrid_target(depth_gt: np.ndarray, labels_gt: np.ndarray, margin: float = 0.02) -> np.ndarray:
    """Training / evaluation target: true depth with layers shifted forward where paint order requires.

    Uses a low quantile so that a few pixels of stroke spill do not dominate the offsets.
    """
    return fuse_ordered(depth_gt, labels_gt, margin=margin, quantile=0.02)


# --------------------------------------------------------------------------- #
# Masked (occluder-removed) depth
# --------------------------------------------------------------------------- #


def fill_holes(rgb: np.ndarray, holes: np.ndarray, levels: int = 8) -> np.ndarray:
    """Fill masked pixels with a smooth push-pull interpolation of the surrounding colours."""
    img = rgb.astype(np.float32)
    wgt = (~holes).astype(np.float32)
    pyr = [(img * wgt[..., None], wgt)]
    for _ in range(levels):
        c, m = pyr[-1]
        if min(m.shape) < 4:
            break
        pyr.append((cv2.pyrDown(c), cv2.pyrDown(m)))
    c, m = pyr[-1]
    filled = c / np.maximum(m, 1e-6)[..., None]
    for c, m in reversed(pyr[:-1]):
        up = cv2.resize(filled, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_LINEAR)
        own = c / np.maximum(m, 1e-6)[..., None]
        a = np.clip(m, 0, 1)[..., None]
        filled = a * own + (1 - a) * up
    out = rgb.copy()
    out[holes] = np.clip(filled[holes], 0, 255).astype(np.uint8)
    return out


def masked_depth(predict: Callable[[np.ndarray], np.ndarray], rgb: np.ndarray, labels: np.ndarray,
                 min_area: float = 0.03, base: np.ndarray | None = None) -> np.ndarray:
    """Depth assembled from one prediction per layer, each made with the layers in front removed.

    Each per-layer prediction is mapped onto the full-image prediction with a least-squares scale and
    shift over that layer's pixels, so the result can be fused like any other depth map.
    """
    l = _dense_labels(labels)
    base = predict(rgb) if base is None else base
    out = base.astype(np.float32).copy()
    for k in range(l.max() + 1):
        sel = l == k
        occluders = l > k
        if sel.mean() < min_area or not occluders.any():
            continue
        d = predict(fill_holes(rgb, occluders))
        A = np.stack([d[sel], np.ones(sel.sum(), np.float32)], 1)
        (scale, shift), *_ = np.linalg.lstsq(A, base[sel], rcond=None)
        if scale > 0:
            out[sel] = scale * d[sel] + shift
    return out
