"""Helpers for the layered-continuous depth experiments (notebook 03)."""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from . import fusion


def degrade_labels(labels: np.ndarray, seed: int = 0, jitter: float = 4.0, merge_prob: float = 0.15,
                   smooth: float = 20.0) -> np.ndarray:
    """Imitate an imperfect layer predictor: wobble boundaries by ~`jitter` px and merge some layers."""
    rng = np.random.default_rng(seed)
    l = fusion._dense_labels(labels)
    mapping = np.arange(l.max() + 1)
    for i in range(1, len(mapping)):
        if rng.random() < merge_prob:
            mapping[i] = mapping[i - 1]
    l = mapping[l]
    h, w = l.shape
    disp = []
    for _ in range(2):
        n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), smooth)
        disp.append(n * jitter / (n.std() + 1e-6))
    map_x = (np.arange(w, dtype=np.float32)[None, :] + disp[0]).astype(np.float32)
    map_y = (np.arange(h, dtype=np.float32)[:, None] + disp[1]).astype(np.float32)
    out = cv2.remap(l.astype(np.float32), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return fusion._dense_labels(out.astype(np.int32))


def run_fusions(depth: np.ndarray, labels: np.ndarray, methods=None, **kwargs) -> dict[str, np.ndarray]:
    """Every fuser in `methods` (default: all) applied to one depth map and one layer map."""
    methods = methods or list(fusion.FUSERS)
    return {m: fusion.FUSERS[m](depth, labels, **kwargs.get(m, {})) for m in methods}


def depth_profile(maps: dict[str, np.ndarray], labels: np.ndarray, column: int | None = None):
    """Values of each map down one image column (default: the column crossing the most layers)."""
    if column is None:
        changes = (np.diff(labels.astype(int), axis=0) != 0).sum(0)
        column = int(np.argmax(changes))
    return column, {k: fusion.normalize01(v)[:, column] for k, v in maps.items()}


def predictor(model, size: int = 518) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a layermap/Depth Anything model as rgb -> depth for `fusion.masked_depth`."""
    from .model import predict

    def fn(rgb):
        out = predict(model, rgb, size=size, flip_tta=False)
        return out if out.ndim == 2 else out[0]

    return fn
