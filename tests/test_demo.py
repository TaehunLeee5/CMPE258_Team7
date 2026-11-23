from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from wm.demo import (
    apply_attack,
    bits_to_text,
    decode,
    embed,
    load_model,
    text_to_bits,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("CMPE", "CMPE"),
        ("AI", "AI  "),
        ("", "    "),
        ("CMPE258", "CMPE"),
    ],
)
def test_message_round_trip(text, expected):
    bits = text_to_bits(text)

    assert len(bits) == 32
    assert set(bits) <= {0, 1}
    assert bits_to_text(bits) == expected


@pytest.mark.parametrize("bits", [[0] * 31, [0] * 33, [2] * 32])
def test_invalid_bits_are_rejected(bits):
    with pytest.raises(ValueError):
        bits_to_text(bits)


def test_checkpoint_embed_resize_decode():
    checkpoint = ROOT / "results/checkin3/runs/hard_fgas_s42/best.pt"
    model = load_model(checkpoint)
    assert not model.training

    expected = text_to_bits("CMPE")

    # Use a synthetic image to exercise cropping and resizing.
    image = Image.new("RGB", (240, 180), (128, 128, 128))
    cover, watermarked, psnr = embed(model, image, "CMPE")

    assert cover.shape == (1, 3, 128, 128)
    assert watermarked.shape == cover.shape
    assert torch.isfinite(watermarked).all()
    assert watermarked.min().item() >= 0
    assert watermarked.max().item() <= 1
    assert np.isfinite(psnr)

    attacked = apply_attack(watermarked, "resize", 0.5)
    assert attacked.shape == watermarked.shape

    bits, text, errors = decode(model, attacked, expected)
    assert bits == expected
    assert text == "CMPE"
    assert errors == 0


def test_unsupported_attack_is_rejected():
    image = torch.zeros(1, 3, 128, 128)

    with pytest.raises(ValueError, match="Unsupported demo attack"):
        apply_attack(image, "unknown", 0.5)