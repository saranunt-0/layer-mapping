"""Dataset generation and loading.

On-disk layout, shared by synthetic data and hand-labelled paintings:

    root/images/<stem>.jpg|png   RGB image
    root/labels/<stem>.png       layer map: any grey levels, darker = further back

Labels are re-ranked on load, so a painting labelled with a handful of grey
levels works the same as a synthetic map with values 0..N-1.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .model import IMAGENET_MEAN, IMAGENET_STD
from .synth import SynthConfig, render

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _write_one(args):
    root, seed, cfg = args
    stem = f"{seed:07d}"
    img_p, lab_p = root / "images" / f"{stem}.jpg", root / "labels" / f"{stem}.png"
    if img_p.exists() and lab_p.exists():
        return
    rgb, labels = render(seed, cfg)
    cv2.imwrite(str(img_p), rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(lab_p), labels)


def generate_dataset(root: str | Path, n: int, start_seed: int = 0, cfg: SynthConfig | None = None,
                     workers: int | None = None, progress: bool = True) -> Path:
    """Render n synthetic scenes (seeds start_seed..start_seed+n-1). Existing files are skipped, so it resumes."""
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    cfg = cfg or SynthConfig()
    jobs = [(root, s, cfg) for s in range(start_seed, start_seed + n)]
    workers = workers or os.cpu_count() or 1
    step = max(1, n // 20)
    with ProcessPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(_write_one, jobs, chunksize=8), 1):
            if progress and (i % step == 0 or i == n):
                print(f"\r  generated {i}/{n}", end="", flush=True)
    if progress:
        print()
    return root


def read_label(path: str | Path) -> np.ndarray:
    """Load a label image as dense ranks 0..L-1 (0 = darkest = back)."""
    lab = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise FileNotFoundError(path)
    if lab.ndim == 3:
        lab = cv2.cvtColor(lab[..., :3], cv2.COLOR_BGR2GRAY)
    _, ranks = np.unique(lab, return_inverse=True)
    return ranks.reshape(lab.shape).astype(np.uint16)


def list_pairs(root: str | Path) -> list[tuple[Path, Path]]:
    root = Path(root)
    pairs = []
    for img in sorted((root / "images").iterdir()):
        lab = root / "labels" / f"{img.stem}.png"
        if img.suffix.lower() in IMAGE_EXTS and lab.exists():
            pairs.append((img, lab))
    return pairs


class LayerFolderDataset(Dataset):
    """(image, layer map) pairs as square tensors of side `size`.

    Training mode applies a random crop, horizontal flip and colour jitter;
    eval mode resizes the whole image.
    """

    def __init__(self, root: str | Path, size: int = 518, train: bool = True, max_items: int | None = None):
        self.pairs = list_pairs(root)[:max_items]
        if not self.pairs:
            raise ValueError(f"No image/label pairs found under {root}")
        self.size = size
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def _jitter(self, rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        r = torch.rand(4).numpy()
        hsv[..., 0] = (hsv[..., 0] + (r[0] - 0.5) * 20) % 180
        hsv[..., 1] *= 0.7 + 0.6 * r[1]
        hsv[..., 2] *= 0.8 + 0.4 * r[2]
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        if r[3] < 0.05:
            out = np.repeat(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
        return out

    def __getitem__(self, idx):
        img_p, lab_p = self.pairs[idx]
        rgb = cv2.imread(str(img_p), cv2.IMREAD_COLOR)[..., ::-1]
        lab = read_label(lab_p)
        if lab.shape != rgb.shape[:2]:
            lab = cv2.resize(lab, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        if self.train:
            h, w = lab.shape
            side = int(min(h, w) * (0.6 + 0.4 * torch.rand(1).item()))
            y0 = int(torch.randint(0, h - side + 1, (1,)))
            x0 = int(torch.randint(0, w - side + 1, (1,)))
            rgb, lab = rgb[y0:y0 + side, x0:x0 + side], lab[y0:y0 + side, x0:x0 + side]
            if torch.rand(1).item() < 0.5:
                rgb, lab = rgb[:, ::-1], lab[:, ::-1]
        s = self.size
        interp = cv2.INTER_AREA if max(rgb.shape[:2]) > s else cv2.INTER_CUBIC
        rgb = cv2.resize(np.ascontiguousarray(rgb), (s, s), interpolation=interp)
        lab = cv2.resize(np.ascontiguousarray(lab), (s, s), interpolation=cv2.INTER_NEAREST)
        if self.train:
            rgb = self._jitter(rgb)
        x = (rgb.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(lab.astype(np.float32))
