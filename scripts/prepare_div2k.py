"""Create the cached DIV2K subset used for training and validation.

    python scripts/prepare_div2k.py --config configs/baseline.yaml

Takes the first `data.num_train_images` DIV2K train HR images (sorted by name)
and the first `data.num_val_images` DIV2K valid HR images, downscales each so
its short side equals `data.prepare_short_side`, and writes them as PNG to
`data.prepared_root/{train,val}` together with a manifest.json.
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


def prepare_split(src_dir, dst_dir, count, short_side):
    paths = list_images(src_dir)[:count]
    if len(paths) < count:
        raise RuntimeError(f"Need {count} images in {src_dir}, found {len(paths)}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in tqdm(paths, desc=dst_dir.name):
        out = dst_dir / (path.stem + ".png")
        with Image.open(path) as img:
            img = img.convert("RGB")
            orig_size = img.size
            scale = short_side / min(img.size)
            new_size = (round(img.width * scale), round(img.height * scale))
            if not out.exists():
                img.resize(new_size, Image.Resampling.LANCZOS).save(out)
        entries.append({"source": path.name, "original_size": orig_size, "prepared_size": new_size})
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"],
                        help="e.g. --splits val when DIV2K is only needed as a test set")
    args = parser.parse_args()

    data_cfg = load_config(args.config)["data"]
    raw_root, out_root = Path(data_cfg["raw_root"]), Path(data_cfg["prepared_root"])
    short_side = data_cfg["prepare_short_side"]

    sources = {"train": ("DIV2K_train_HR", data_cfg["num_train_images"]),
               "val": ("DIV2K_valid_HR", data_cfg["num_val_images"])}
    manifest = {"short_side": short_side}
    for split in args.splits:
        folder, count = sources[split]
        manifest[split] = prepare_split(raw_root / folder, out_root / split, count, short_side)
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"Prepared {', '.join(f'{len(manifest[s])} {s}' for s in args.splits)} images in {out_root}")


if __name__ == "__main__":
    main()
