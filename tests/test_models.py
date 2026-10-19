import torch

from wm.messages import random_messages
from wm.models import build_model


def test_messages_are_binary_with_requested_shape():
    msg = random_messages(5, 32, generator=torch.Generator().manual_seed(0))
    assert msg.shape == (5, 32)
    assert set(msg.unique().tolist()) <= {0.0, 1.0}


def test_encoder_decoder_shapes_and_range(small_cfg):
    model = build_model(small_cfg)
    size, length = small_cfg["data"]["image_size"], small_cfg["message"]["length"]
    cover = torch.rand(4, 3, size, size)
    message = random_messages(4, length)
    watermarked, attacked, logits = model(cover, message)
    assert watermarked.shape == cover.shape
    assert attacked.shape == cover.shape
    assert logits.shape == (4, length)
    assert watermarked.min() >= 0 and watermarked.max() <= 1


def test_gradients_reach_encoder(small_cfg):
    model = build_model(small_cfg)
    size, length = small_cfg["data"]["image_size"], small_cfg["message"]["length"]
    cover = torch.rand(4, 3, size, size) * 0.8 + 0.1
    message = random_messages(4, length)
    _, _, logits = model(cover, message)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, message).backward()
    for name, module in (("encoder", model.encoder), ("decoder", model.decoder)):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads), f"no gradient in {name}"
