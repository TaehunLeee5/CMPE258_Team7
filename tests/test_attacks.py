import pytest
import torch

from wm.attacks import ATTACKS, AttackSampler, build_attacks
from wm.messages import random_messages
from wm.models import build_model


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_attack_preserves_shape_and_range(name):
    attack = build_attacks([name])[name]
    x = torch.rand(2, 3, 33, 47)  # odd sizes on purpose
    y = attack(x)
    assert y.shape == x.shape
    assert y.min() >= 0 and y.max() <= 1


@pytest.mark.parametrize("name", ["noise", "blur", "brightness", "resize"])
def test_attack_changes_image(name):
    x = torch.rand(2, 3, 32, 32) * 0.5 + 0.25
    assert not torch.allclose(build_attacks([name])[name](x), x)


def test_attack_strength_is_configurable():
    attacks = build_attacks(["noise", "blur"], {"noise": {"std": 0.2}, "blur": {"kernel_size": 7, "sigma": 2.0}})
    assert attacks["noise"].std == 0.2
    assert attacks["blur"].kernel.shape[-1] == 7


@pytest.mark.parametrize("name", ["noise", "blur", "brightness", "resize"])
def test_gradients_flow_through_attack(small_cfg, name):
    model = build_model(small_cfg)
    size, length = small_cfg["data"]["image_size"], small_cfg["message"]["length"]
    cover = torch.rand(4, 3, size, size) * 0.6 + 0.2
    message = random_messages(4, length)
    _, _, logits = model(cover, message, build_attacks([name])[name])
    torch.nn.functional.binary_cross_entropy_with_logits(logits, message).backward()
    assert model.encoder.to_residual.weight.grad.abs().sum() > 0


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
