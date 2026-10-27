"""Image distortions and the attack sampler.

Every attack maps a (B, 3, H, W) tensor in [0, 1] to a tensor of the same shape,
so the same modules are used for training (gradients flow back to the encoder)
and for attack-wise evaluation. Strengths come from `attacks.params` in the config.

A strength given as a `[min, max]` pair is drawn uniformly per image (per batch
for resize), so training covers a range instead of a single setting. Attacks
with `trainable = False` (real JPEG, chains that contain it) have no gradient
and are for evaluation only.

`params[name]` may carry a `type` key to build a second, differently configured
instance of an attack under its own name, e.g.
`jpeg_q30: {type: jpeg_real, quality: 30}`.
"""

import io
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def _per_image(value, n, device, dtype=torch.float32):
    """Strength for each of `n` images: a scalar is repeated, a [min, max] pair is sampled uniformly."""
    if isinstance(value, (list, tuple)):
        low, high = float(value[0]), float(value[1])
        return torch.empty(n, device=device, dtype=dtype).uniform_(low, high)
    return torch.full((n,), float(value), device=device, dtype=dtype)


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
        std = _per_image(self.std, x.shape[0], x.device, x.dtype).view(-1, 1, 1, 1)
        return (x + std * torch.randn_like(x)).clamp(0, 1)


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
        scale = float(_per_image(self.scale, 1, "cpu")[0])  # one scale per batch (interpolate needs one size)
        small = (max(1, math.floor(height * scale)), max(1, math.floor(width * scale)))
        x = F.interpolate(x, size=small, mode="bilinear", align_corners=False)
        return F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)


# Standard JPEG (ITU-T T.81 Annex K) quantization tables at quality 50.
_LUMA_TABLE = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
]
_CHROMA_TABLE = [[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
                 [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99]] + [[99] * 8] * 4


def _dct_matrix(size=8):
    k = torch.arange(size, dtype=torch.float32).view(-1, 1)
    n = torch.arange(size, dtype=torch.float32).view(1, -1)
    mat = torch.cos(math.pi * (2 * n + 1) * k / (2 * size)) * math.sqrt(2 / size)
    mat[0] /= math.sqrt(2)
    return mat  # orthonormal DCT-II: coeffs = M @ block @ M.T


def _scaled_tables(base, quality):
    """IJG quality scaling of a (8, 8) table for a (B,) quality tensor -> (B, 1, 1, 8, 8)."""
    quality = quality.clamp(1, 100)
    scale = torch.where(quality < 50, 5000 / quality, 200 - 2 * quality).view(-1, 1, 1, 1, 1)
    return torch.floor((base * scale + 50) / 100).clamp(1, 255)


class DiffJPEG(nn.Module):
    """Differentiable JPEG simulation (YCbCr, 4:2:0 chroma, 8x8 DCT, IJG tables).

    Quantization rounding uses a straight-through estimator (`rounding="ste"`:
    forward = round, backward = identity) or the cubic approximation
    round(x) + (x - round(x))^3 (`rounding="cubic"`). Entropy coding is lossless,
    so it is omitted. `quality` may be a [min, max] range, sampled per image.
    """

    name = "jpeg"

    def __init__(self, quality=(50, 90), rounding="ste"):
        super().__init__()
        if rounding not in ("ste", "cubic"):
            raise ValueError(f"rounding must be 'ste' or 'cubic', got {rounding!r}")
        self.quality, self.rounding = quality, rounding
        self.register_buffer("dct", _dct_matrix())
        self.register_buffer("luma_table", torch.tensor(_LUMA_TABLE, dtype=torch.float32))
        self.register_buffer("chroma_table", torch.tensor(_CHROMA_TABLE, dtype=torch.float32))

    def _round(self, x):
        if self.rounding == "ste":
            return x + (torch.round(x) - x).detach()
        r = torch.round(x)
        return r + (x - r) ** 3

    def _compress_channel(self, channel, table):
        """Quantize one (B, H, W) channel (H, W multiples of 8) in DCT space with per-image tables."""
        b, h, w = channel.shape
        blocks = channel.view(b, h // 8, 8, w // 8, 8).permute(0, 1, 3, 2, 4)
        coeffs = self.dct @ blocks @ self.dct.T
        coeffs = self._round(coeffs / table) * table
        blocks = self.dct.T @ coeffs @ self.dct
        return blocks.permute(0, 1, 3, 2, 4).reshape(b, h, w)

    def forward(self, x):
        b, _, height, width = x.shape
        pad_h, pad_w = (-height) % 16, (-width) % 16
        if pad_h or pad_w:
            x = F.pad(x, [0, pad_w, 0, pad_h], mode="replicate")
        rgb = x * 255
        r, g, bl = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * bl
        cb = -0.168736 * r - 0.331264 * g + 0.5 * bl + 128
        cr = 0.5 * r - 0.418688 * g - 0.081312 * bl + 128
        chroma = F.avg_pool2d(torch.stack([cb, cr], dim=1), 2)

        quality = _per_image(self.quality, b, x.device)
        luma_q = _scaled_tables(self.luma_table, quality)
        chroma_q = _scaled_tables(self.chroma_table, quality)
        y = self._compress_channel(y - 128, luma_q) + 128
        cb = self._compress_channel(chroma[:, 0] - 128, chroma_q)
        cr = self._compress_channel(chroma[:, 1] - 128, chroma_q)
        chroma = F.interpolate(torch.stack([cb, cr], dim=1), scale_factor=2, mode="bilinear", align_corners=False)
        cb, cr = chroma[:, 0], chroma[:, 1]

        out = torch.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], dim=1) / 255
        return out[..., :height, :width].clamp(0, 1)


class RealJPEG(nn.Module):
    """Actual JPEG encode/decode with PIL (no gradient; evaluation only).

    Measures the gap between the differentiable simulation and real codecs.
    Images are quantized to uint8 first, as a saved file would be.
    """

    name = "jpeg_real"
    trainable = False

    def __init__(self, quality=50):
        super().__init__()
        self.quality = quality

    @torch.no_grad()
    def forward(self, x):
        qualities = _per_image(self.quality, x.shape[0], "cpu").round().int().tolist()
        arrays = (x.detach().clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        out = []
        for array, quality in zip(arrays, qualities):
            buffer = io.BytesIO()
            Image.fromarray(array).save(buffer, format="JPEG", quality=int(quality))
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                out.append(np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy())
        decoded = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2)
        return decoded.to(device=x.device, dtype=x.dtype) / 255


class Crop(nn.Module):
    """Keep a random square window covering `keep` of the image area, resized back to full size.

    Combines translation and scale change. The window is drawn per image and
    resampled with grid_sample, so the attack stays batched and differentiable.
    """

    name = "crop"

    def __init__(self, keep=(0.5, 0.8)):
        super().__init__()
        self.keep = keep

    def forward(self, x):
        b = x.shape[0]
        side = _per_image(self.keep, b, x.device, x.dtype).clamp(1e-3, 1).sqrt()  # fraction of each side
        max_shift = 1 - side
        cx = (torch.rand(b, device=x.device, dtype=x.dtype) * 2 - 1) * max_shift
        cy = (torch.rand(b, device=x.device, dtype=x.dtype) * 2 - 1) * max_shift
        theta = torch.zeros(b, 2, 3, device=x.device, dtype=x.dtype)
        theta[:, 0, 0], theta[:, 0, 2] = side, cx
        theta[:, 1, 1], theta[:, 1, 2] = side, cy
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


class Occlusion(nn.Module):
    """Fill one random rectangle covering `area` of each image with a constant (cropout)."""

    name = "occlusion"

    def __init__(self, area=(0.25, 0.5), fill=0.5):
        super().__init__()
        self.area, self.fill = area, fill

    def forward(self, x):
        b, _, height, width = x.shape
        area = _per_image(self.area, b, x.device, x.dtype)
        aspect = torch.exp(torch.empty(b, device=x.device, dtype=x.dtype).uniform_(math.log(0.5), math.log(2.0)))
        rect_h = (area * aspect).sqrt().clamp(max=1) * height
        rect_w = (area / aspect).sqrt().clamp(max=1) * width
        top = torch.rand(b, device=x.device, dtype=x.dtype) * (height - rect_h)
        left = torch.rand(b, device=x.device, dtype=x.dtype) * (width - rect_w)
        rows = torch.arange(height, device=x.device, dtype=x.dtype).view(1, -1, 1)
        cols = torch.arange(width, device=x.device, dtype=x.dtype).view(1, 1, -1)
        inside_rows = (rows >= top.view(-1, 1, 1)) & (rows < (top + rect_h).view(-1, 1, 1))
        inside_cols = (cols >= left.view(-1, 1, 1)) & (cols < (left + rect_w).view(-1, 1, 1))
        mask = (inside_rows & inside_cols).unsqueeze(1).to(x.dtype)
        return x * (1 - mask) + self.fill * mask


class Compose(nn.Module):
    """Apply several attacks in sequence, e.g. `stages: [[resize, {scale: 0.5}], [jpeg_real, {quality: 50}]]`."""

    name = "combo"

    def __init__(self, stages=(("resize", {"scale": 0.5}), ("jpeg_real", {"quality": 50}))):
        super().__init__()
        self.stages = nn.ModuleList(build_attack(stage_name, stage_params) for stage_name, stage_params in stages)
        self.trainable = all(is_trainable(stage) for stage in self.stages)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


ATTACKS = {cls.name: cls for cls in (Identity, GaussianNoise, GaussianBlur, Brightness, Resize,
                                     DiffJPEG, RealJPEG, Crop, Occlusion, Compose)}


def is_trainable(attack):
    return getattr(attack, "trainable", True)


def build_attack(name, params=None):
    """Build attack `name`; a `type` entry in `params` selects the class for an aliased instance."""
    params = dict(params or {})
    kind = params.pop("type", name)
    if kind not in ATTACKS:
        raise KeyError(f"Unknown attack {kind!r}; available: {sorted(ATTACKS)}")
    return ATTACKS[kind](**params)


def build_attacks(names, params=None, trainable_only=False):
    """Return an ordered {name: module} dict using strengths from `params[name]`.

    With `trainable_only`, evaluation-only attacks (no gradient) are rejected.
    """
    params = params or {}
    attacks = nn.ModuleDict({name: build_attack(name, params.get(name)) for name in names})
    if trainable_only:
        bad = [name for name, attack in attacks.items() if not is_trainable(attack)]
        if bad:
            raise ValueError(f"Attacks {bad} are evaluation-only and cannot be used for training")
    return attacks


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
