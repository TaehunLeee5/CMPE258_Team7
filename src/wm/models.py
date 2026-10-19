"""HiDDeN-inspired encoder and decoder (Zhu et al., ECCV 2018), kept intentionally small.

Encoder: conv features of the cover image are concatenated with the message
(bits mapped to +/-1 and replicated over every spatial position) and the cover
itself; a final 1x1 conv predicts a residual that is added to the cover:

    watermarked = clamp(cover + residual, 0, 1)

Decoder: conv stack (three stride-2 stages) -> global average pooling -> linear
layer producing one logit per message bit.
"""

import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Encoder(nn.Module):
    def __init__(self, message_length, channels=64, blocks=4):
        super().__init__()
        layers = [ConvBNReLU(3, channels)] + [ConvBNReLU(channels, channels) for _ in range(blocks - 1)]
        self.features = nn.Sequential(*layers)
        self.fuse = ConvBNReLU(channels + message_length + 3, channels)
        self.to_residual = nn.Conv2d(channels, 3, kernel_size=1)

    def forward(self, cover, message):
        _, _, height, width = cover.shape
        signed = (2 * message - 1)[:, :, None, None].expand(-1, -1, height, width)
        x = torch.cat([self.features(cover), signed, cover], dim=1)
        residual = self.to_residual(self.fuse(x))
        return (cover + residual).clamp(0, 1)


class Decoder(nn.Module):
    def __init__(self, message_length, channels=64, blocks=7):
        super().__init__()
        # Downsample at three evenly spaced blocks to keep compute low at 128x128.
        stride_at = {blocks * (i + 1) // 4 for i in range(3)}
        layers = [ConvBNReLU(3, channels)]
        layers += [ConvBNReLU(channels, channels, stride=2 if i in stride_at else 1) for i in range(1, blocks)]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels, message_length)

    def forward(self, image):
        return self.head(self.pool(self.features(image)).flatten(1))


class WatermarkModel(nn.Module):
    """Encoder + decoder; the distortion is applied by the caller in between."""

    def __init__(self, message_length=32, encoder_channels=64, encoder_blocks=4,
                 decoder_channels=64, decoder_blocks=7):
        super().__init__()
        self.message_length = message_length
        self.encoder = Encoder(message_length, encoder_channels, encoder_blocks)
        self.decoder = Decoder(message_length, decoder_channels, decoder_blocks)

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
    )
