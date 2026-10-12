"""Turn a continuous layer-depth map into a small number of discrete, editable layers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def kmeans_1d(values: np.ndarray, k: int, weights: np.ndarray | None = None, iters: int = 50) -> np.ndarray:
    """Sorted cluster centres of 1-D data (Lloyd's algorithm, quantile initialisation)."""
    values = values.astype(np.float64).ravel()
    weights = np.ones_like(values) if weights is None else weights.astype(np.float64).ravel()
    k = max(1, min(k, len(np.unique(values))))
    centres = np.quantile(values, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        assign = np.abs(values[:, None] - centres[None, :]).argmin(1)
        new = np.array([
            np.average(values[assign == c], weights=weights[assign == c]) if (assign == c).any() else centres[c]
            for c in range(k)
        ])
        if np.allclose(new, centres):
            break
        centres = new
    return np.sort(centres)


def depth_to_layers(depth: np.ndarray, n_layers: int = 5, rgb: np.ndarray | None = None,
                    n_segments: int = 1200, compactness: float = 10.0) -> np.ndarray:
    """Quantise a layer-depth map (larger = closer) into n_layers labels, 0 = back.

    With `rgb`, the depth is first averaged inside SLIC superpixels so layer
    boundaries follow colour edges in the painting instead of depth noise.
    """
    d = depth.astype(np.float32)
    if rgb is not None:
        from skimage.segmentation import slic

        if rgb.shape[:2] != d.shape:
            rgb = cv2.resize(rgb, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA)
        seg = slic(rgb, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=-1)
        counts = np.bincount(seg.ravel())
        means = np.bincount(seg.ravel(), weights=d.ravel()) / np.maximum(counts, 1)
        centres = kmeans_1d(means[counts > 0], n_layers, counts[counts > 0])
        seg_label = np.abs(means[:, None] - centres[None, :]).argmin(1)
        return seg_label[seg].astype(np.uint8)
    sample = d.ravel()[:: max(1, d.size // 200_000)]
    centres = kmeans_1d(sample, n_layers)
    return np.abs(d[..., None] - centres).argmin(-1).astype(np.uint8)


def colorize(labels: np.ndarray, n_layers: int | None = None) -> np.ndarray:
    """RGB visualisation of a label map (dark = back, bright = front)."""
    n = int(labels.max()) + 1 if n_layers is None else n_layers
    scaled = (labels.astype(np.float32) / max(1, n - 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    lo, hi = np.percentile(d, [1, 99])
    scaled = (np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)[..., ::-1].copy()


def export_layers(rgb: np.ndarray, labels: np.ndarray, out_dir: str | Path, name: str = "layer") -> list[Path]:
    """Write one RGBA PNG per layer (back to front), the label map and a preview."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rgb.shape[:2] != labels.shape:
        labels = cv2.resize(labels, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    paths = []
    for k in range(int(labels.max()) + 1):
        rgba = np.dstack([rgb, np.where(labels == k, 255, 0).astype(np.uint8)])
        p = out / f"{name}_{k:02d}.png"
        cv2.imwrite(str(p), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        paths.append(p)
    cv2.imwrite(str(out / "layer_map.png"), labels)
    cv2.imwrite(str(out / "preview.png"), colorize(labels)[..., ::-1])
    return paths
