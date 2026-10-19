import math

import torch

from wm.attacks import build_attacks
from wm.data import build_loaders
from wm.evaluate import evaluate_attackwise
from wm.metrics import bit_accuracy, message_accuracy, psnr, ssim
from wm.models import build_model


def test_bit_and_message_accuracy():
    message = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    logits = torch.tensor([[3.0, -2.0, 1.0, -1.0], [3.0, 2.0, -1.0, 1.0]])  # one wrong bit in row 2
    assert bit_accuracy(logits, message) == 7 / 8
    assert message_accuracy(logits, message) == 0.5


def test_psnr_known_value():
    x = torch.zeros(1, 3, 8, 8)
    y = torch.full_like(x, 0.1)  # MSE = 0.01 -> 20 dB
    assert math.isclose(psnr(x, y), 20.0, rel_tol=1e-5)


def test_ssim_identity_and_degradation():
    x = torch.rand(2, 3, 32, 32)
    assert math.isclose(ssim(x, x), 1.0, rel_tol=1e-4)
    assert ssim(x, (x + 0.2 * torch.randn_like(x)).clamp(0, 1)) < 0.9


def test_attackwise_evaluation_is_deterministic(small_cfg):
    torch.manual_seed(0)
    model = build_model(small_cfg)
    _, val_loader = build_loaders(small_cfg)
    attacks = build_attacks(small_cfg["attacks"]["eval"], small_cfg["attacks"]["params"])
    device = torch.device("cpu")
    a = evaluate_attackwise(model, val_loader, attacks, device, seed=1)
    b = evaluate_attackwise(model, val_loader, attacks, device, seed=1)
    assert a == b
    assert set(a["attacks"]) == set(small_cfg["attacks"]["eval"])
    assert a["worst_case"]["attack"] != "clean"
