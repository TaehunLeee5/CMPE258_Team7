"""Validate the prepared DIV2K cache and export deterministic validation crops.

    python scripts/validate_div2k.py --samples-out results/div2k_samples
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm.config import load_config  # noqa: E402
from wm.data import build_val_dataset, validate_prepared_manifest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--samples-out", default="results/div2k_samples")
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")

    cfg = load_config(args.config)
    if cfg["data"].get("synthetic", False):
        parser.error("DIV2K validation requires data.synthetic=false")
    summary = validate_prepared_manifest(cfg["data"])
    val_dataset = build_val_dataset(cfg)
    count = min(args.count, len(val_dataset))
    out_dir = Path(args.samples_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        crop = (val_dataset[index].permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        Image.fromarray(crop).save(out_dir / f"val_crop_{index:03d}.png")

    print(f"Validated {summary['train_images']} train / {summary['val_images']} val images")
    print(f"Validation crops: {len(val_dataset)} ({cfg['data']['val_crops_per_image']} per image)")
    print(f"Saved {count} deterministic crop samples to {out_dir}")


if __name__ == "__main__":
    main()