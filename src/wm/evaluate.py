"""Attack-wise evaluation.

Each validation batch is watermarked once and then decoded separately after
every attack, so all attacks are compared on identical watermarked images.
Messages and attack randomness are seeded, and the global RNG state is restored
afterwards so evaluating in the middle of training does not change training.
"""

import torch

from .metrics import bit_accuracy, message_accuracy, psnr, ssim
from .messages import random_messages


@torch.no_grad()
def evaluate_attackwise(model, loader, attacks, device, seed=0):
    """Return {"attacks": {name: {bit_acc, msg_acc}}, "image_quality": {psnr, ssim},
    "worst_case": {...}, "num_samples": N}."""
    was_training = model.training
    model.eval()
    sums = {name: {"bit_acc": 0.0, "msg_acc": 0.0} for name in attacks}
    quality = {"psnr": 0.0, "ssim": 0.0}
    total = 0
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        message_gen = torch.Generator().manual_seed(seed)
        for cover in loader:
            cover = cover.to(device, non_blocking=True)
            n = cover.shape[0]
            message = random_messages(n, model.message_length, generator=message_gen).to(device)
            watermarked = model.encoder(cover, message)
            quality["psnr"] += psnr(watermarked, cover) * n
            quality["ssim"] += ssim(watermarked, cover) * n
            for name, attack in attacks.items():
                logits = model.decoder(attack(watermarked))
                sums[name]["bit_acc"] += bit_accuracy(logits, message) * n
                sums[name]["msg_acc"] += message_accuracy(logits, message) * n
            total += n
    model.train(was_training)

    per_attack = {name: {k: v / total for k, v in vals.items()} for name, vals in sums.items()}
    attacked = {k: v for k, v in per_attack.items() if k != "clean"} or per_attack
    worst = min(attacked, key=lambda k: attacked[k]["bit_acc"])
    return {
        "attacks": per_attack,
        "image_quality": {k: v / total for k, v in quality.items()},
        "worst_case": {"attack": worst, "bit_acc": attacked[worst]["bit_acc"]},
        "mean_attacked_bit_acc": sum(v["bit_acc"] for v in attacked.values()) / len(attacked),
        "num_samples": total,
    }
