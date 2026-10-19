"""Fixed-length binary watermark messages."""

import torch


def random_messages(batch_size, length, device=None, generator=None):
    """Uniform random bits as a float tensor of shape (batch_size, length) in {0, 1}."""
    bits = torch.randint(0, 2, (batch_size, length), generator=generator,
                         device=generator.device if generator is not None else device)
    return bits.float().to(device) if device is not None else bits.float()


def bits_to_string(bits):
    return "".join(str(int(b)) for b in bits)
