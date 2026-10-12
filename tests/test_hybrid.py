"""Tests for layered-continuous depth: synthetic depth, fusion, metrics and two-channel training."""

import numpy as np
import pytest
import torch

from layermap import fusion, hybrid_eval
from layermap.data import LayerFolderDataset, generate_dataset
from layermap.metrics import hybrid_metrics, within_layer_rho
from layermap.model import load_model, predict
from layermap.synth import SynthConfig, render, scene_kinds
from layermap.train import TrainConfig, train

SMALL = SynthConfig(size=96)


def test_render_depth():
    rgb, lab, dep = render(4, SMALL, return_depth=True)
    rgb2, lab2 = render(4, SMALL)
    assert np.array_equal(rgb, rgb2) and np.array_equal(lab, lab2)
    assert dep.shape == lab.shape and dep.dtype == np.float32
    assert 0 <= dep.min() and dep.max() <= 1.01
    assert np.allclose(dep[lab == 0], 0)  # sky is at infinity
    assert scene_kinds(4, SMALL)[0] == "sky"


def test_layer_offsets_minimal():
    lab = np.zeros((20, 20), int)
    lab[10:] = 1
    dep = np.where(lab == 1, 0.2, 0.5).astype(np.float32)  # front layer predicted behind: needs a shift
    b = fusion.layer_offsets(dep, lab, margin=0.1, quantile=0.0)
    assert b[0] == 0 and np.isclose(b[1], 0.4)
    ok = np.where(lab == 1, 0.9, 0.1).astype(np.float32)  # already in front by more than the margin
    assert np.allclose(fusion.layer_offsets(ok, lab, margin=0.1), 0)


@pytest.mark.parametrize("name", list(fusion.FUSERS))
def test_fusers_shape_and_range(name):
    _, lab, dep = render(9, SMALL, return_depth=True)
    out = fusion.FUSERS[name](dep, lab)
    assert out.shape == lab.shape and out.dtype == np.float32
    assert out.min() >= 0 and out.max() <= 1


def test_ordered_fusion_restores_layer_order():
    _, lab, dep = render(11, SMALL, return_depth=True)
    target = fusion.hybrid_target(dep, lab)
    flat = np.full_like(dep, 0.5)
    m = hybrid_metrics(fusion.fuse_ordered(flat, lab), target, dep, lab)
    assert m["acc_edge"] > 0.95  # order comes from the layers even when depth carries none
    assert within_layer_rho(dep, dep, lab) > 0.99


def test_masked_depth_and_degrade():
    rgb, lab, dep = render(12, SMALL, return_depth=True)
    calls = []

    def fake_predict(img):
        calls.append(1)
        return img.mean(-1).astype(np.float32)

    out = fusion.masked_depth(fake_predict, rgb, lab, min_area=0.0)
    assert out.shape == lab.shape and len(calls) > 1
    noisy = hybrid_eval.degrade_labels(lab, seed=1)
    assert noisy.shape == lab.shape and noisy.max() <= lab.max()


def test_two_channel_training(tmp_path, tiny_backbone):
    root = tmp_path / "synth"
    generate_dataset(root / "train", 8, 0, SMALL, workers=2, progress=False, with_depth=True)
    generate_dataset(root / "val", 2, 500, SMALL, workers=2, progress=False, with_depth=True)
    ds = LayerFolderDataset(root / "train", 70, targets=["hybrid", "layer"])
    x, y = ds[0]
    assert y.shape == (2, 70, 70)
    cfg = TrainConfig(train_dirs=[str(root / "train")], val_dir=str(root / "val"), out_dir=str(tmp_path / "run"),
                      backbone=tiny_backbone, size=70, batch_size=2, steps=2, warmup=1, num_workers=0,
                      log_every=1, eval_every=2, save_every=2, targets=["hybrid", "layer"])
    train(cfg)
    model = load_model(str(tmp_path / "run" / "best"), device="cpu")
    assert model.head.conv3.out_channels == 2
    out = predict(model, render(1, SMALL)[0], size=70)
    assert out.shape == (2, 96, 96)
    cfg.steps = 3
    train(cfg)  # resumes a two-channel checkpoint
    assert torch.load(tmp_path / "run" / "last" / "train_state.pt", weights_only=False)["step"] == 3
    cfg.targets = ["layer"]
    with pytest.raises(ValueError, match="trained with targets"):
        train(cfg)  # resuming with different outputs would silently drop a channel
