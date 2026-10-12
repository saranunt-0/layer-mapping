"""Parallax previews: the quickest way to judge whether a depth map has smooth ground and clean layers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .fusion import normalize01


def parallax_frames(rgb: np.ndarray, depth: np.ndarray, amplitude: float = 0.025, n: int = 16,
                    max_side: int = 512) -> list[np.ndarray]:
    """Frames of a horizontal camera wiggle: near pixels move more than far ones (z-buffered forward warp)."""
    s = min(1.0, max_side / max(rgb.shape[:2]))
    if s < 1:
        rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    d = cv2.resize(normalize01(depth), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    h, w = d.shape
    ys, xs = np.mgrid[0:h, 0:w]
    ys, xs, dv = ys.ravel(), xs.ravel(), d.ravel()
    colours = rgb.reshape(-1, 3)
    frames = []
    for t in range(n):
        shift = amplitude * w * np.sin(2 * np.pi * t / n)
        xt = np.round(xs + shift * (dv - 0.5)).astype(int)
        ok = (xt >= 0) & (xt < w)
        flat = ys[ok] * w + xt[ok]
        zbuf = np.full(h * w, -1.0)
        np.maximum.at(zbuf, flat, dv[ok])
        win = dv[ok] >= zbuf[flat]
        out = np.zeros((h * w, 3), np.uint8)
        out[flat[win]] = colours[ok][win]
        holes = (zbuf < 0).reshape(h, w).astype(np.uint8)
        frame = out.reshape(h, w, 3)
        frames.append(cv2.inpaint(frame, holes, 3, cv2.INPAINT_TELEA) if holes.any() else frame)
    return frames


def save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 12) -> Path:
    from PIL import Image

    path = Path(path)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
    return path
