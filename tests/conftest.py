"""Shared fixtures."""

import pytest


@pytest.fixture(scope="session")
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
