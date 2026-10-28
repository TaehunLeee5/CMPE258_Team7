"""Validate the prepared COCO cache and export deterministic validation crops.

    python scripts/validate_coco.py --config configs/hard_base.yaml --samples-out results/coco_samples

Checks the manifest against the config and the PNG files on disk, and that the
train and val splits are disjoint with every val filename sorting after every
train filename.
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm.config import load_config  # noqa: E402
from wm.data import active_data_cfg, build_val_dataset, validate_prepared_manifest  # noqa: E402


def check_split_order(manifest):
    train = [entry["source"] for entry in manifest["train"]]
    val = [entry["source"] for entry in manifest["val"]]
    if set(train) & set(val):
        raise ValueError("COCO train and val splits overlap")
    if max(train) >= min(val):
        raise ValueError("COCO val images must sort after all train images")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/hard_base.yaml")
    parser.add_argument("--samples-out", default="results/coco_samples")
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")

    cfg = load_config(args.config)
    if cfg["data"].get("synthetic", False) or cfg["data"].get("dataset") != "coco":
        parser.error("COCO validation requires data.dataset=coco and data.synthetic=false")
    data_cfg = active_data_cfg(cfg["data"])
    summary = validate_prepared_manifest(data_cfg)
    with open(Path(data_cfg["prepared_root"]) / "manifest.json", encoding="utf-8") as f:
        check_split_order(json.load(f))

    val_dataset = build_val_dataset(cfg)
    count = min(args.count, len(val_dataset))
    out_dir = Path(args.samples_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        crop = (val_dataset[index].permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        Image.fromarray(crop).save(out_dir / f"val_crop_{index:03d}.png")

    print(f"Validated {summary['train_images']} train / {summary['val_images']} val COCO images")
    print(f"Validation crops: {len(val_dataset)} ({cfg['data']['val_crops_per_image']} per image)")
    print(f"Saved {count} deterministic crop samples to {out_dir}")


if __name__ == "__main__":
    main()
