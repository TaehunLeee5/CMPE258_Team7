"""Render original, watermarked, and amplified-difference validation crops.

    python scripts/figure1.py --checkpoint results/baseline_uniform/best.pt
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm.config import apply_override, resolve_device, set_seed  # noqa: E402
from wm.data import build_val_dataset, validate_prepared_manifest  # noqa: E402
from wm.models import build_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="results/baseline_uniform/best.pt")
    parser.add_argument("--out", default="results/figure1.png")
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 100, 200])
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "cfg" not in checkpoint or "model" not in checkpoint:
        parser.error(f"{args.checkpoint} is not a training checkpoint with cfg and model state")
    cfg = checkpoint["cfg"]
    for item in args.set:
        apply_override(cfg, item)
    if cfg["data"].get("synthetic", False):
        parser.error("Figure 1 requires real DIV2K validation crops (data.synthetic=false)")
    set_seed(cfg["experiment"]["seed"])
    validate_prepared_manifest(cfg["data"])
    dataset = build_val_dataset(cfg)
    if any(index < 0 or index >= len(dataset) for index in args.indices):
        parser.error(f"--indices must be in [0, {len(dataset) - 1}]")

    device = resolve_device(cfg["experiment"].get("device", "auto"))
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    generator = torch.Generator().manual_seed(cfg["experiment"]["seed"])
    figure, axes = plt.subplots(len(args.indices), 3, figsize=(12, 3.6 * len(args.indices)), squeeze=False)
    columns = ("Original", "Watermarked", "Amplified |difference| (x10)")

    with torch.inference_mode():
        for row, index in enumerate(args.indices):
            cover = dataset[index].unsqueeze(0).to(device)
            message = torch.randint(0, 2, (1, cfg["message"]["length"]), generator=generator).to(device).float()
            watermarked = model.encoder(cover, message)
            difference = (watermarked - cover).abs().mul(10).clamp(0, 1)
            for axis, image, column in zip(axes[row], (cover, watermarked, difference), columns):
                axis.imshow(image[0].cpu().permute(1, 2, 0).numpy())
                axis.set_title(f"Crop {index}: {column}")
                axis.set_axis_off()

    figure.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved Figure 1 with {len(args.indices)} validation crops to {out}")


if __name__ == "__main__":
    main()