"""Fine-tune Depth Anything V2 Small to predict paint-order layer indices.

Usable from a notebook (`train(TrainConfig(...))`) or the command line:

    python -m layermap.train --train-dirs data/synth_train --val-dir data/synth_val --steps 2000

Checkpoints go to <out_dir>/last (every save_every steps, for resuming after a
Colab disconnect) and <out_dir>/best (highest validation score).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .data import LayerFolderDataset
from .losses import LossWeights, layer_loss
from .metrics import PAIR_MODES, all_metrics
from .model import DEFAULT_BACKBONE, load_model, save_model


@dataclass
class TrainConfig:
    train_dirs: list[str]
    val_dir: str | None = None
    out_dir: str = "runs/layermap"
    backbone: str = DEFAULT_BACKBONE
    size: int = 518
    batch_size: int = 8
    steps: int = 2000
    lr_backbone: float = 5e-6
    lr_head: float = 5e-5
    weight_decay: float = 0.01
    warmup: int = 100
    grad_clip: float = 1.0
    num_workers: int = 2
    log_every: int = 25
    eval_every: int = 500
    save_every: int = 500
    val_items: int = 200
    resume: bool = True
    seed: int = 0
    loss: LossWeights = field(default_factory=LossWeights)


@torch.no_grad()
def evaluate(model, dataset, batch_size: int = 8, max_items: int | None = None) -> dict:
    """Mean metrics (see layermap.metrics) over an eval-mode LayerFolderDataset.

    `score` is the mean of the three pair accuracies and picks the best checkpoint.
    """
    model.eval()
    device = next(model.parameters()).device
    n = len(dataset) if max_items is None else min(max_items, len(dataset))
    rows = []
    for start in range(0, n, batch_size):
        items = [dataset[i] for i in range(start, min(n, start + batch_size))]
        x = torch.stack([it[0] for it in items]).to(device)
        with torch.autocast(device.type, dtype=_amp_dtype(device), enabled=device.type == "cuda"):
            pred = model(pixel_values=x).predicted_depth.float().cpu()
        rows += [all_metrics(p, t) for p, (_, t) in zip(pred, items)]
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    out["score"] = float(np.mean([out[f"acc_{m}"] for m in PAIR_MODES]))
    return out


def _amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float16


def _save(model, opt, sched, scaler, step, best, out: Path, name: str):
    d = out / name
    save_model(model, d)
    torch.save({"step": step, "best": best, "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict()}, d / "train_state.pt")


def train(cfg: TrainConfig) -> list[dict]:
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resume_dir = out / "last"
    resuming = cfg.resume and (resume_dir / "train_state.pt").exists()
    model = load_model(str(resume_dir) if resuming else cfg.backbone, for_training=True, device=device)

    train_ds = ConcatDataset([LayerFolderDataset(d, cfg.size, train=True) for d in cfg.train_dirs])
    val_ds = LayerFolderDataset(cfg.val_dir, cfg.size, train=False, max_items=cfg.val_items) if cfg.val_dir else None
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=cfg.num_workers > 0)
    print(f"train images: {len(train_ds)}  val images: {len(val_ds) if val_ds else 0}  device: {device}")

    backbone = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone}
    rest = [p for p in model.parameters() if id(p) not in backbone_ids]
    opt = torch.optim.AdamW([{"params": backbone, "lr": cfg.lr_backbone}, {"params": rest, "lr": cfg.lr_head}],
                            weight_decay=cfg.weight_decay)

    def lr_lambda(step):
        if step < cfg.warmup:
            return (step + 1) / cfg.warmup
        progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    amp_dtype = _amp_dtype(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    step, best = 0, -1.0
    history_path = out / "history.jsonl"
    if resuming:
        state = torch.load(resume_dir / "train_state.pt", map_location=device, weights_only=False)
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        step, best = state["step"], state["best"]
        print(f"resumed from step {step}")
    history = [json.loads(line) for line in history_path.read_text().splitlines()] if history_path.exists() else []

    def log(rec):
        history.append(rec)
        with history_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    model.train()
    running, t0 = {}, time.time()
    while step < cfg.steps:
        for x, y in loader:
            if step >= cfg.steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                pred = model(pixel_values=x).predicted_depth
            loss, terms = layer_loss(pred, y, cfg.loss)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1

            for k, v in {"loss": loss.detach(), **terms}.items():
                running[k] = running.get(k, 0.0) + float(v)
            if step % cfg.log_every == 0:
                rec = {"step": step, **{k: v / cfg.log_every for k, v in running.items()},
                       "lr": sched.get_last_lr()[0], "sec_per_step": (time.time() - t0) / cfg.log_every}
                log(rec)
                print(f"step {step:6d}  loss {rec['loss']:.4f}  ssi {rec['ssi']:.4f}  rank {rec['rank']:.4f}  "
                      f"{rec['sec_per_step']:.2f}s/step", flush=True)
                running, t0 = {}, time.time()
            if val_ds is not None and (step % cfg.eval_every == 0 or step == cfg.steps):
                metrics = evaluate(model, val_ds)
                log({"step": step, **{f"val_{k}": v for k, v in metrics.items()}})
                print(f"  val @ {step}: " + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
                if metrics["score"] > best:
                    best = metrics["score"]
                    _save(model, opt, sched, scaler, step, best, out, "best")
                model.train()
                t0 = time.time()
            if step % cfg.save_every == 0 or step == cfg.steps:
                _save(model, opt, sched, scaler, step, best, out, "last")
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dirs", nargs="+", required=True)
    for f in dataclasses.fields(TrainConfig):
        if f.name in ("train_dirs", "loss"):
            continue
        typ = {"int": int, "float": float, "bool": lambda s: s.lower() in ("1", "true", "yes")}.get(
            f.type if isinstance(f.type, str) else f.type.__name__, str)
        p.add_argument(f"--{f.name.replace('_', '-')}", type=typ, default=f.default)
    for f in dataclasses.fields(LossWeights):
        p.add_argument(f"--loss-{f.name}", type=float, default=f.default)
    a = vars(p.parse_args())
    loss = LossWeights(**{f.name: a.pop(f"loss_{f.name}") for f in dataclasses.fields(LossWeights)})
    train(TrainConfig(**a, loss=loss))


if __name__ == "__main__":
    main()
