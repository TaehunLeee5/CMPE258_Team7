"""Differentiable image distortions and the attack sampler.

Every attack maps a (B, 3, H, W) tensor in [0, 1] to a tensor of the same shape,
so the same modules are used for training (gradients flow back to the encoder)
and for attack-wise evaluation. Strengths come from `attacks.params` in the config.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    name = "clean"

    def forward(self, x):
        return x


class GaussianNoise(nn.Module):
    name = "noise"

    def __init__(self, std=0.03):
        super().__init__()
        self.std = std

    def forward(self, x):
        return (x + self.std * torch.randn_like(x)).clamp(0, 1)


class GaussianBlur(nn.Module):
    name = "blur"

    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-coords**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        self.register_buffer("kernel", kernel_2d.expand(3, 1, kernel_size, kernel_size).clone())
        self.padding = kernel_size // 2

    def forward(self, x):
        x = F.pad(x, [self.padding] * 4, mode="reflect")
        return F.conv2d(x, self.kernel.to(x.dtype), groups=x.shape[1])


class Brightness(nn.Module):
    """Multiply each image by a random factor in [min_factor, max_factor]."""

    name = "brightness"

    def __init__(self, min_factor=0.7, max_factor=1.3):
        super().__init__()
        self.min_factor, self.max_factor = min_factor, max_factor

    def forward(self, x):
        factor = torch.empty(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
        factor.uniform_(self.min_factor, self.max_factor)
        return (x * factor).clamp(0, 1)


class Resize(nn.Module):
    """Downsample by `scale` and restore to the original size (bilinear)."""

    name = "resize"

    def __init__(self, scale=0.5):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        height, width = x.shape[-2:]
        small = (max(1, math.floor(height * self.scale)), max(1, math.floor(width * self.scale)))
        x = F.interpolate(x, size=small, mode="bilinear", align_corners=False)
        return F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)


ATTACKS = {cls.name: cls for cls in (Identity, GaussianNoise, GaussianBlur, Brightness, Resize)}


def build_attack(name, params=None):
    if name not in ATTACKS:
        raise KeyError(f"Unknown attack {name!r}; available: {sorted(ATTACKS)}")
    return ATTACKS[name](**(params or {}))


def build_attacks(names, params=None):
    """Return an ordered {name: module} dict using strengths from `params[name]`."""
    params = params or {}
    return nn.ModuleDict({name: build_attack(name, params.get(name)) for name in names})


class AttackSampler:
    """Draws one attack name per training batch from a categorical distribution.

    Starts uniform unless `probs` is given; FGAS replaces the distribution via
    `set_probs`. Per-attack draw counts are kept so the training log records
    which attack each batch actually used.
    """

    def __init__(self, names, probs=None, seed=0):
        self.names = list(names)
        self.generator = torch.Generator().manual_seed(seed)
        self.set_probs(probs or {name: 1.0 / len(self.names) for name in self.names})
        self.reset_counts()

    def set_probs(self, probs):
        missing = set(self.names) - set(probs)
        if missing:
            raise ValueError(f"Missing probabilities for {sorted(missing)}")
        values = torch.tensor([float(probs[name]) for name in self.names])
        if (values < 0).any() or values.sum() <= 0:
            raise ValueError(f"Invalid attack probabilities: {probs}")
        values = values / values.sum()
        self.probs = {name: float(p) for name, p in zip(self.names, values)}
        self._weights = values

    def sample(self):
        index = torch.multinomial(self._weights, 1, generator=self.generator).item()
        name = self.names[index]
        self.counts[name] += 1
        return name

    def reset_counts(self):
        self.counts = {name: 0 for name in self.names}

    def state_dict(self):
        return {"names": self.names, "probs": self.probs, "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        self.set_probs(state["probs"])
        self.generator.set_state(state["generator"])
