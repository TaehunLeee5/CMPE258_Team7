import numpy as np
from PIL import Image, ImageOps
import torch
from wm.attacks import build_attack
from wm.models import build_model


def text_to_bits(text: str) -> list[int]:
    """Encode up to four ASCII characters as 32 bits, padding with spaces."""
    message = text[:4].encode("ascii", errors="replace").ljust(4, b" ")
    return [(byte >> shift) & 1 for byte in message for shift in range(7, -1, -1)]


def bits_to_text(bits: list[int]) -> str:
    """Decode 32 bits into four characters."""
    if len(bits) != 32 or any(bit not in (0, 1) for bit in bits):
        raise ValueError("Expected exactly 32 binary bits.")

    message = bytes(
        sum(bits[start + offset] << (7 - offset) for offset in range(8))
        for start in range(0, 32, 8)
    )
    return message.decode("ascii", errors="replace")


def load_model(checkpoint):
    """Load the team's trained watermark model for CPU inference."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(saved["cfg"])
    model.load_state_dict(saved["model"])
    return model.eval()
@torch.inference_mode()
def apply_attack(image, name, strength):
    """Apply the selected distortion to a watermarked image."""
    if name == "clean":
        return image
    if name == "resize":
        attack = build_attack("demo_resize", {"type": "resize", "scale": strength})
        return attack(image)
    raise ValueError(f"Unsupported demo attack: {name}")


@torch.inference_mode()
def decode(model, image, expected_bits):
    """Recover the message and count incorrect bits for one image."""
    bits = (model.decoder(image)[0] > 0).int().tolist()
    errors = sum(actual != expected for actual, expected in zip(bits, expected_bits))
    return bits, bits_to_text(bits), errors

@torch.inference_mode()
def embed(model, image, text):
    """Return the prepared cover, watermarked tensor, and PSNR in dB."""
    prepared = ImageOps.fit(
        image.convert("RGB"), (128, 128), method=Image.Resampling.BILINEAR
    )
    pixels = np.array(prepared, dtype=np.float32) / 255.0
    cover = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    message = torch.tensor([text_to_bits(text)], dtype=torch.float32)

    watermarked = model.encoder(cover, message)
    mse = (watermarked - cover).square().mean().item()
    psnr = float("inf") if mse == 0 else -10 * np.log10(mse)
    return cover, watermarked, float(psnr)