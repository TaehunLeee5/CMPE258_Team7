import pytest
import torch

from wm.attacks import ATTACKS, AttackSampler, build_attack, build_attacks, is_trainable
from wm.messages import random_messages
from wm.models import build_model

DIFFERENTIABLE = ["noise", "blur", "brightness", "resize", "jpeg", "crop", "occlusion"]


def _textured(n=2, size=32, seed=0):
    """Smooth images with some detail, closer to photos than uniform noise."""
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand(n, 3, 4, 4, generator=generator)
    smooth = torch.nn.functional.interpolate(coarse, size=(size, size), mode="bicubic", align_corners=False)
    return (smooth + 0.03 * torch.randn(n, 3, size, size, generator=generator)).clamp(0, 1)


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_attack_preserves_shape_and_range(name):
    attack = build_attacks([name])[name]
    x = torch.rand(2, 3, 33, 47)  # odd sizes on purpose
    y = attack(x)
    assert y.shape == x.shape
    assert y.min() >= 0 and y.max() <= 1


@pytest.mark.parametrize("name", DIFFERENTIABLE + ["jpeg_real", "combo"])
def test_attack_changes_image(name):
    x = torch.rand(2, 3, 32, 32) * 0.5 + 0.25
    assert not torch.allclose(build_attacks([name])[name](x), x)


def test_attack_strength_is_configurable():
    attacks = build_attacks(["noise", "blur"], {"noise": {"std": 0.2}, "blur": {"kernel_size": 7, "sigma": 2.0}})
    assert attacks["noise"].std == 0.2
    assert attacks["blur"].kernel.shape[-1] == 7


@pytest.mark.parametrize("name", DIFFERENTIABLE)
def test_gradients_flow_through_attack(small_cfg, name):
    model = build_model(small_cfg)
    size, length = small_cfg["data"]["image_size"], small_cfg["message"]["length"]
    cover = torch.rand(4, 3, size, size) * 0.6 + 0.2
    message = random_messages(4, length)
    _, _, logits = model(cover, message, build_attacks([name])[name])
    torch.nn.functional.binary_cross_entropy_with_logits(logits, message).backward()
    assert model.encoder.to_residual.weight.grad.abs().sum() > 0


def _mse(a, b):
    return torch.mean((a - b) ** 2).item()


def test_diff_jpeg_high_quality_is_near_identity_and_low_quality_degrades():
    x = _textured(4, 64)
    high = build_attack("jpeg", {"quality": 95})(x)
    low = build_attack("jpeg", {"quality": 10})(x)
    assert _mse(high, x) < 1e-3  # remaining error is mostly 4:2:0 chroma subsampling, as in real JPEG
    assert _mse(low, x) > 3 * _mse(high, x)


def test_diff_jpeg_matches_real_jpeg_at_quality_50():
    x = (_textured(4, 64) * 255).round() / 255  # real JPEG sees uint8 pixels
    simulated = build_attack("jpeg", {"quality": 50})(x)
    real = build_attack("jpeg_real", {"quality": 50})(x)
    assert _mse(simulated, real) < 0.1 * _mse(real, x)


@pytest.mark.parametrize("rounding", ["ste", "cubic"])
def test_diff_jpeg_rounding_modes_pass_gradients(rounding):
    x = _textured(2, 32).requires_grad_(True)
    build_attack("jpeg", {"quality": 50, "rounding": rounding})(x).sum().backward()
    assert x.grad.abs().sum() > 0


def test_real_jpeg_and_chains_with_it_are_evaluation_only():
    assert not is_trainable(build_attack("jpeg_real"))
    assert not is_trainable(build_attack("combo"))
    assert is_trainable(build_attack("combo", {"stages": [["resize", {"scale": 0.5}], ["jpeg", {"quality": 50}]]}))
    with pytest.raises(ValueError):
        build_attacks(["noise", "jpeg_real"], trainable_only=True)


@pytest.mark.parametrize("name", ["noise", "jpeg", "crop", "occlusion", "resize"])
def test_seeded_attacks_are_reproducible(name):
    attack = build_attack(name, {"resize": {"scale": [0.3, 0.7]}}.get(name))
    x = _textured(3, 32)
    torch.manual_seed(7)
    first = attack(x)
    torch.manual_seed(7)
    assert torch.equal(first, attack(x))


def test_noise_range_is_sampled_per_image_within_bounds():
    torch.manual_seed(0)
    noise = build_attack("noise", {"std": [0.05, 0.2]})
    x = torch.full((64, 3, 32, 32), 0.5)
    std = (noise(x) - x).std(dim=(1, 2, 3))
    assert std.min() > 0.04 and std.max() < 0.21
    assert std.std() > 0.02  # different strength per image


def test_range_strength_is_sampled_per_image_within_bounds():
    torch.manual_seed(0)
    occlusion = build_attack("occlusion", {"area": [0.2, 0.4], "fill": 0.0})
    x = torch.ones(64, 3, 64, 64)
    covered = 1 - occlusion(x)[:, 0].mean(dim=(1, 2))
    assert covered.min() > 0.15 and covered.max() < 0.45  # pixel rounding at the rectangle edges
    assert covered.std() > 0.02  # different area per image


def test_crop_keep_one_is_identity():
    x = _textured(2, 32)
    assert torch.allclose(build_attack("crop", {"keep": 1.0})(x), x, atol=1e-5)


def test_aliased_attack_instances():
    params = {"jpeg_q30": {"type": "jpeg_real", "quality": 30}, "noise_hard": {"type": "noise", "std": 0.2}}
    attacks = build_attacks(["jpeg_q30", "noise_hard"], params)
    assert attacks["jpeg_q30"].quality == 30 and not is_trainable(attacks["jpeg_q30"])
    assert attacks["noise_hard"].std == 0.2
    with pytest.raises(KeyError):
        build_attack("bogus", {"type": "nope"})


def test_uniform_sampler_draws_every_attack_and_counts():
    names = ["noise", "blur", "brightness", "resize"]
    sampler = AttackSampler(names, seed=0)
    assert all(abs(p - 0.25) < 1e-6 for p in sampler.probs.values())
    for _ in range(4000):
        sampler.sample()
    assert sum(sampler.counts.values()) == 4000
    assert all(800 < c < 1200 for c in sampler.counts.values())


def test_sampler_follows_updated_probs():
    sampler = AttackSampler(["a", "b"], seed=0)
    sampler.set_probs({"a": 0.9, "b": 0.1})
    for _ in range(2000):
        sampler.sample()
    assert sampler.counts["a"] > 1600
