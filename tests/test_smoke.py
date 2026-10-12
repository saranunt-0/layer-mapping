"""CPU smoke tests. A tiny randomly initialised Depth Anything stands in for the real checkpoint."""

import json

import numpy as np
import pytest
import torch

from layermap.data import LayerFolderDataset, generate_dataset, read_label
from layermap.layers import depth_to_layers, export_layers
from layermap.losses import layer_loss
from layermap.metrics import ordinal_accuracy, ssi_mae, vertical_prior
from layermap.model import load_model, predict
from layermap.synth import SynthConfig, render
from layermap.train import TrainConfig, evaluate, train

SMALL = SynthConfig(size=96)


@pytest.fixture(scope="module")
def tiny_backbone(tmp_path_factory):
    from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation, Dinov2Config

    backbone = Dinov2Config(image_size=70, patch_size=14, hidden_size=32, num_hidden_layers=4, num_attention_heads=2,
                            intermediate_size=64, out_indices=[1, 2, 3, 4], apply_layernorm=True,
                            reshape_hidden_states=False)
    cfg = DepthAnythingConfig(backbone_config=backbone, reassemble_hidden_size=32, neck_hidden_sizes=[8, 16, 16, 16],
                              fusion_hidden_size=16, head_hidden_size=8)
    path = tmp_path_factory.mktemp("tiny")
    DepthAnythingForDepthEstimation(cfg).save_pretrained(str(path))
    return str(path)


@pytest.fixture(scope="module")
def synth_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    generate_dataset(root / "train", 12, start_seed=0, cfg=SMALL, workers=2, progress=False)
    generate_dataset(root / "val", 4, start_seed=1000, cfg=SMALL, workers=2, progress=False)
    return root / "train", root / "val"


def test_render_is_deterministic_with_dense_labels():
    rgb, lab = render(3, SMALL)
    rgb2, lab2 = render(3, SMALL)
    assert rgb.shape == (96, 96, 3) and rgb.dtype == np.uint8
    assert np.array_equal(lab, lab2) and np.array_equal(rgb, rgb2)
    assert np.array_equal(np.unique(lab), np.arange(lab.max() + 1))
    assert lab.max() >= 1


def test_losses_reward_correct_order():
    lab = torch.from_numpy(render(5, SMALL)[1].astype(np.float32))[None]
    perfect = 3.0 * lab + 7.0
    _, terms = layer_loss(perfect, lab)
    assert terms["ssi"] < 1e-5 and terms["grad"] < 1e-5
    total_good, _ = layer_loss(perfect, lab)
    total_bad, _ = layer_loss(-perfect, lab)
    assert total_good < total_bad


def test_metrics():
    lab = render(7, SMALL)[1].astype(np.float32)
    for mode in ("all", "local", "hard"):
        assert ordinal_accuracy(lab * 2 + 1, lab, mode=mode) == 1.0
        assert ordinal_accuracy(-lab, lab, mode=mode) == 0.0
    assert ordinal_accuracy(vertical_prior(lab.shape), lab, mode="hard") == 0.0
    assert ssi_mae(lab * 5 - 2, lab) < 1e-5


def test_depth_to_layers_is_ordered(tmp_path):
    depth = np.tile(np.linspace(0, 1, 64, dtype=np.float32)[:, None], (1, 48))
    labels = depth_to_layers(depth, n_layers=4)
    assert labels.max() == 3 and labels[0, 0] == 0 and labels[-1, 0] == 3
    assert np.all(np.diff(labels[:, 0].astype(int)) >= 0)
    rgb = (np.random.default_rng(0).random((64, 48, 3)) * 255).astype(np.uint8)
    sp = depth_to_layers(depth, n_layers=4, rgb=rgb, n_segments=50)
    assert sp.shape == depth.shape and sp.max() <= 3
    paths = export_layers(rgb, labels, tmp_path)
    assert len(paths) == 4 and (tmp_path / "layer_map.png").exists()


def test_label_reranking(tmp_path):
    import cv2

    lab = np.zeros((10, 10), np.uint8)
    lab[5:] = 200
    lab[8:] = 90
    cv2.imwrite(str(tmp_path / "l.png"), lab)
    assert sorted(np.unique(read_label(tmp_path / "l.png"))) == [0, 1, 2]


def test_predict_and_train_resume(tiny_backbone, synth_dirs, tmp_path):
    train_dir, val_dir = synth_dirs
    model = load_model(tiny_backbone, device="cpu")
    rgb = render(1, SMALL)[0]
    depth = predict(model, rgb, size=70)
    assert depth.shape == rgb.shape[:2]

    ds = LayerFolderDataset(train_dir, size=70, train=True)
    x, y = ds[0]
    assert x.shape == (3, 70, 70) and y.shape == (70, 70)
    assert set(evaluate(model, LayerFolderDataset(val_dir, 70, train=False))) == {
        "acc_all", "acc_local", "acc_hard", "ssi_mae", "score"}

    cfg = TrainConfig(train_dirs=[str(train_dir)], val_dir=str(val_dir), out_dir=str(tmp_path / "run"),
                      backbone=tiny_backbone, size=70, batch_size=2, steps=4, warmup=1, num_workers=0,
                      log_every=2, eval_every=2, save_every=2)
    train(cfg)
    assert (tmp_path / "run" / "last" / "train_state.pt").exists()
    assert (tmp_path / "run" / "best" / "config.json").exists()

    cfg.steps = 6
    train(cfg)
    state = torch.load(tmp_path / "run" / "last" / "train_state.pt", weights_only=False)
    assert state["step"] == 6
    steps = [json.loads(line)["step"] for line in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    assert max(steps) == 6

    finetuned = load_model(str(tmp_path / "run" / "best"), device="cpu")
    assert isinstance(finetuned.head.activation2, torch.nn.Identity)
