"""Watermark metrics: bit accuracy, full-message recovery, PSNR, SSIM.

All functions take batched tensors and return the mean over the batch, with
images in [0, 1]. SSIM follows Wang et al. (2004): an 11x11 Gaussian window
(sigma 1.5), K1 = 0.01, K2 = 0.03, computed per channel and averaged.
"""

import torch
import torch.nn.functional as F


def decode_bits(logits):
    return (logits > 0).float()


def bit_accuracy(logits, message):
    """Fraction of individual bits recovered correctly."""
    return (decode_bits(logits) == message).float().mean().item()


def message_accuracy(logits, message):
    """Fraction of samples whose whole message is recovered exactly."""
    return (decode_bits(logits) == message).all(dim=1).float().mean().item()


def psnr(x, y, data_range=1.0):
    """Mean PSNR (dB) over the batch, computed per image."""
    mse = (x - y).pow(2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return (10 * torch.log10(data_range**2 / mse)).mean().item()


def _gaussian_window(size, sigma, channels, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g).expand(channels, 1, size, size).contiguous()


def ssim(x, y, data_range=1.0, window_size=11, sigma=1.5):
    """Mean SSIM over the batch (valid convolution, no padding)."""
    channels = x.shape[1]
    window = _gaussian_window(window_size, sigma, channels, x.device, x.dtype)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2

    def filt(t):
        return F.conv2d(t, window, groups=channels)

    mu_x, mu_y = filt(x), filt(y)
    sigma_x = filt(x * x) - mu_x**2
    sigma_y = filt(y * y) - mu_y**2
    sigma_xy = filt(x * y) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
    return ssim_map.flatten(1).mean(dim=1).mean().item()
