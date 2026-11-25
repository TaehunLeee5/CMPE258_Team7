from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from wm.demo import load_model
from wm.models import Encoder, UNetSEEncoder, build_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("side", [128, 130])
def test_unet_shape_range_and_message_gradient(small_cfg, side):
    small_cfg["model"]["encoder_type"] = "unet_se"
    model = build_model(small_cfg)
    assert isinstance(model.encoder, UNetSEEncoder)

    cover = torch.rand(2, 3, side, side) * 0.8 + 0.1
    message = torch.randint(
        0, 2, (2, small_cfg["message"]["length"])
    ).float()

    watermarked, attacked, logits = model(cover, message)

    assert watermarked.shape == cover.shape
    assert attacked.shape == cover.shape
    assert logits.shape == message.shape
    assert torch.isfinite(watermarked).all()
    assert watermarked.min().item() >= 0
    assert watermarked.max().item() <= 1

    F.binary_cross_entropy_with_logits(logits, message).backward()
    gradient = model.encoder.message_fc.weight.grad
    assert gradient is not None
    assert gradient.abs().sum().item() > 0


def test_default_encoder_loads_existing_checkpoint():
    checkpoint = ROOT / "results/checkin3/runs/hard_fgas_s42/best.pt"
    model = load_model(checkpoint)
    assert isinstance(model.encoder, Encoder)