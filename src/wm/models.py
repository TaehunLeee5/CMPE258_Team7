"""HiDDeN-inspired encoder and decoder (Zhu et al., ECCV 2018), kept intentionally small.

Encoder: conv features of the cover image are concatenated with a message map
and the cover itself; a final 1x1 conv predicts a residual that is added to the
cover:

    watermarked = clamp(cover + residual, 0, 1)

Message map (`message_embed`):
  - "replicate": bits mapped to +/-1 and copied to every pixel (original HiDDeN).
  - "spatial":   a linear layer maps the bits to a (message_channels x grid x grid)
                 map that is upsampled to the image size (StegaStamp-style). This gives the encoder a spatially varying pattern to
                 embed and converges much faster at our small training budget.

Decoder: conv stack (three stride-2 stages) -> adaptive average pooling to a
`decoder_pool` x `decoder_pool` grid (1 = global average) -> linear layer that
produces one logit per message bit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Encoder(nn.Module):
    def __init__(self, message_length, channels=64, blocks=4, message_embed="spatial",
                 message_grid=16, message_channels=8):
        super().__init__()
        if message_embed not in ("replicate", "spatial"):
            raise ValueError(f"Unknown message_embed {message_embed!r}")
        layers = [ConvBNReLU(3, channels)] + [ConvBNReLU(channels, channels) for _ in range(blocks - 1)]
        self.features = nn.Sequential(*layers)
        self.message_embed = message_embed
        if message_embed == "spatial":
            self.message_grid, self.message_channels = message_grid, message_channels
            self.message_fc = nn.Linear(message_length, message_channels * message_grid * message_grid)
            map_channels = message_channels
        else:
            map_channels = message_length
        self.fuse = ConvBNReLU(channels + map_channels + 3, channels)
        self.to_residual = nn.Conv2d(channels, 3, kernel_size=1)

    def message_map(self, message, height, width):
        signed = 2 * message - 1
        if self.message_embed == "replicate":
            return signed[:, :, None, None].expand(-1, -1, height, width)
        grid = self.message_fc(signed).view(-1, self.message_channels, self.message_grid, self.message_grid)
        if height % self.message_grid == 0 and width % self.message_grid == 0:
            # Block upsampling via repeat_interleave: deterministic backward on CUDA (bilinear is not).
            grid = grid.repeat_interleave(height // self.message_grid, dim=2)
            return grid.repeat_interleave(width // self.message_grid, dim=3)
        return F.interpolate(grid, size=(height, width), mode="bilinear", align_corners=False)

    def forward(self, cover, message):
        _, _, height, width = cover.shape
        x = torch.cat([self.features(cover), self.message_map(message, height, width), cover], dim=1)
        residual = self.to_residual(self.fuse(x))
        return (cover + residual).clamp(0, 1)


class Decoder(nn.Module):
    def __init__(self, message_length, channels=64, blocks=7, pool=1):
        super().__init__()
        # Downsample at three evenly spaced blocks to keep compute low at 128x128.
        stride_at = {blocks * (i + 1) // 4 for i in range(3)}
        layers = [ConvBNReLU(3, channels)]
        layers += [ConvBNReLU(channels, channels, stride=2 if i in stride_at else 1) for i in range(1, blocks)]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(pool)
        self.head = nn.Linear(channels * pool * pool, message_length)

    def forward(self, image):
        return self.head(self.pool(self.features(image)).flatten(1))


class WatermarkModel(nn.Module):
    """Encoder + decoder; the distortion is applied by the caller in between."""

    def __init__(self, message_length=32, encoder_channels=64, encoder_blocks=4,
                 decoder_channels=64, decoder_blocks=7, message_embed="spatial",
                 message_grid=16, message_channels=8, decoder_pool=1):
        super().__init__()
        self.message_length = message_length
        self.encoder = Encoder(message_length, encoder_channels, encoder_blocks,
                               message_embed, message_grid, message_channels)
        self.decoder = Decoder(message_length, decoder_channels, decoder_blocks, decoder_pool)

    def forward(self, cover, message, attack=None):
        watermarked = self.encoder(cover, message)
        attacked = attack(watermarked) if attack is not None else watermarked
        return watermarked, attacked, self.decoder(attacked)


def build_model(cfg):
    m = cfg["model"]
    return WatermarkModel(
        message_length=cfg["message"]["length"],
        encoder_channels=m["encoder_channels"],
        encoder_blocks=m["encoder_blocks"],
        decoder_channels=m["decoder_channels"],
        decoder_blocks=m["decoder_blocks"],
        message_embed=m.get("message_embed", "spatial"),
        message_grid=m.get("message_grid", 16),
        message_channels=m.get("message_channels", 8),
        decoder_pool=m.get("decoder_pool", 1),
    )
