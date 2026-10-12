"""Dataset generation and loading.

On-disk layout, shared by synthetic data and hand-labelled paintings:

    root/images/<stem>.jpg|png   RGB image
    root/labels/<stem>.png       layer map: any grey levels, darker = further back
    root/depth/<stem>.png        optional 16-bit disparity (synthetic data with with_depth=True)

Labels are re-ranked on load, so a painting labelled with a handful of grey
levels works the same as a synthetic map with values 0..N-1.

Training targets (`LayerFolderDataset(targets=...)`):
    layer   paint-order layer index
    depth   true continuous disparity (needs depth/)
    hybrid  layered continuous depth, `fusion.hybrid_target(depth, layer)` (needs depth/)
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .fusion import hybrid_target
from .model import IMAGENET_MEAN, IMAGENET_STD
from .synth import SynthConfig, render

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _write_one(args):
    root, seed, cfg, with_depth = args
    stem = f"{seed:07d}"
    img_p, lab_p, dep_p = (root / "images" / f"{stem}.jpg", root / "labels" / f"{stem}.png",
                           root / "depth" / f"{stem}.png")
    if img_p.exists() and lab_p.exists() and (dep_p.exists() or not with_depth):
        return
    rgb, labels, depth = render(seed, cfg, return_depth=True)
    cv2.imwrite(str(img_p), rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(lab_p), labels)
    if with_depth:
        cv2.imwrite(str(dep_p), np.round(np.clip(depth, 0, 1) * 65535).astype(np.uint16))


def generate_dataset(root: str | Path, n: int, start_seed: int = 0, cfg: SynthConfig | None = None,
                     workers: int | None = None, progress: bool = True, with_depth: bool = False) -> Path:
    """Render n synthetic scenes (seeds start_seed..start_seed+n-1). Existing files are skipped, so it resumes.

    with_depth also writes the true disparity map, needed for the "depth" and "hybrid" targets.
    """
    root = Path(root)
    for sub in ("images", "labels") + (("depth",) if with_depth else ()):
        (root / sub).mkdir(parents=True, exist_ok=True)
    cfg = cfg or SynthConfig()
    jobs = [(root, s, cfg, with_depth) for s in range(start_seed, start_seed + n)]
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


def read_depth(path: str | Path) -> np.ndarray:
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    return d.astype(np.float32) / 65535


def list_pairs(root: str | Path) -> list[tuple[Path, Path]]:
    root = Path(root)
    pairs = []
    for img in sorted((root / "images").iterdir()):
        lab = root / "labels" / f"{img.stem}.png"
        if img.suffix.lower() in IMAGE_EXTS and lab.exists():
            pairs.append((img, lab))
    return pairs


TARGETS = ("layer", "depth", "hybrid")


class LayerFolderDataset(Dataset):
    """(image, target) pairs as square tensors of side `size`.

    The target is (size, size) for a single target and (len(targets), size, size) otherwise.
    Training mode applies a random crop, horizontal flip and colour jitter;
    eval mode resizes the whole image.
    """

    def __init__(self, root: str | Path, size: int = 518, train: bool = True, max_items: int | None = None,
                 targets: tuple[str, ...] | list[str] = ("layer",), margin: float = 0.02):
        self.root = Path(root)
        self.pairs = list_pairs(root)[:max_items]
        if not self.pairs:
            raise ValueError(f"No image/label pairs found under {root}")
        unknown = set(targets) - set(TARGETS)
        if unknown:
            raise ValueError(f"unknown targets {unknown}; choose from {TARGETS}")
        self.targets = tuple(targets)
        self.needs_depth = bool({"depth", "hybrid"} & set(self.targets))
        if self.needs_depth and not (self.root / "depth" / f"{self.pairs[0][0].stem}.png").exists():
            raise ValueError(f"targets {self.targets} need {self.root / 'depth'}: generate with with_depth=True")
        self.margin = margin
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

    def _target_maps(self, img_p: Path, lab_p: Path, shape) -> np.ndarray:
        lab = read_label(lab_p)
        if lab.shape != shape:
            lab = cv2.resize(lab, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        maps = {"layer": lab.astype(np.float32)}
        if self.needs_depth:
            dep = read_depth(self.root / "depth" / f"{img_p.stem}.png")
            if dep.shape != shape:
                dep = cv2.resize(dep, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            maps["depth"] = dep
            if "hybrid" in self.targets:
                # Computed on the full image: the layer offsets depend on every boundary.
                maps["hybrid"] = hybrid_target(dep, lab, self.margin)
        return np.stack([maps[t] for t in self.targets], axis=-1)

    def __getitem__(self, idx):
        img_p, lab_p = self.pairs[idx]
        rgb = cv2.imread(str(img_p), cv2.IMREAD_COLOR)[..., ::-1]
        y = self._target_maps(img_p, lab_p, rgb.shape[:2])
        if self.train:
            h, w = y.shape[:2]
            side = int(min(h, w) * (0.6 + 0.4 * torch.rand(1).item()))
            y0 = int(torch.randint(0, h - side + 1, (1,)))
            x0 = int(torch.randint(0, w - side + 1, (1,)))
            rgb, y = rgb[y0:y0 + side, x0:x0 + side], y[y0:y0 + side, x0:x0 + side]
            if torch.rand(1).item() < 0.5:
                rgb, y = rgb[:, ::-1], y[:, ::-1]
        s = self.size
        interp = cv2.INTER_AREA if max(rgb.shape[:2]) > s else cv2.INTER_CUBIC
        rgb = cv2.resize(np.ascontiguousarray(rgb), (s, s), interpolation=interp)
        y = cv2.resize(np.ascontiguousarray(y), (s, s), interpolation=cv2.INTER_NEAREST).reshape(s, s, -1)
        if self.train:
            rgb = self._jitter(rgb)
        x = (rgb.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        y = torch.from_numpy(y.transpose(2, 0, 1).copy())
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), (y[0] if len(self.targets) == 1 else y)
