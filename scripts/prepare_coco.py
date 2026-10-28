"""Create the cached COCO val2017 train/val split used by the hardened benchmark.

    python scripts/prepare_coco.py --config configs/hard_base.yaml

COCO val2017 images are sorted by filename; images whose short side is below
128 px are dropped. The last `data.num_val_images` of the remaining images form
the validation split and the first `data.num_train_images` before them the
training split (4500 / 500 for all 5000 images). Each image is resized so its
short side equals `data.prepare_short_side` (256: COCO is ~640x480, so this
downscales) and written as PNG (never JPEG, which would add compression
artifacts before any attack) to `data.coco_prepared_root/{train,val}`, with a
manifest.json in the same format as the DIV2K one.
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm.config import load_config  # noqa: E402
from wm.data import list_images  # noqa: E402

MIN_SHORT_SIDE = 128


def split_coco(sizes, num_train, num_val, min_short_side=MIN_SHORT_SIDE):
    """Split {filename: (width, height)} by sorted filename into (train, val, dropped) name lists."""
    names = sorted(sizes)
    kept = [name for name in names if min(sizes[name]) >= min_short_side]
    dropped = [name for name in names if min(sizes[name]) < min_short_side]
    if len(kept) < num_train + num_val:
        raise RuntimeError(f"Need {num_train + num_val} images with short side >= {min_short_side}, "
                           f"found {len(kept)} ({len(dropped)} dropped); lower data.num_train_images")
    val = kept[len(kept) - num_val:]
    train = kept[:len(kept) - num_val][:num_train]
    return train, val, dropped


def prepare_images(paths, dst_dir, short_side):
    dst_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in tqdm(paths, desc=dst_dir.name):
        out = dst_dir / (path.stem + ".png")
        with Image.open(path) as img:
            orig_size = img.size
            scale = short_side / min(orig_size)
            new_size = (round(orig_size[0] * scale), round(orig_size[1] * scale))
            if not out.exists():
                img.convert("RGB").resize(new_size, Image.Resampling.LANCZOS).save(out, format="PNG")
        entries.append({"source": path.name, "original_size": orig_size, "prepared_size": new_size})
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/hard_base.yaml")
    args = parser.parse_args()

    data_cfg = load_config(args.config)["data"]
    src_dir = Path(data_cfg["coco_raw_root"]) / "val2017"
    out_root = Path(data_cfg["coco_prepared_root"])
    short_side = data_cfg["prepare_short_side"]

    paths = {path.name: path for path in list_images(src_dir)}
    if not paths:
        raise FileNotFoundError(f"No images in {src_dir}. Run scripts/download_coco.py first.")
    sizes = {}
    for name, path in paths.items():
        with Image.open(path) as img:  # reads the header only
            sizes[name] = img.size
    train, val, dropped = split_coco(sizes, data_cfg["num_train_images"], data_cfg["num_val_images"])
    print(f"{len(paths)} images, {len(dropped)} dropped (short side < {MIN_SHORT_SIDE})")

    manifest = {"short_side": short_side, "dataset": "coco_val2017", "min_short_side": MIN_SHORT_SIDE,
                "dropped": dropped}
    for split, names in (("train", train), ("val", val)):
        manifest[split] = prepare_images([paths[name] for name in names], out_root / split, short_side)
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"Prepared {len(train)} train, {len(val)} val images in {out_root}")


if __name__ == "__main__":
    main()
