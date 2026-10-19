"""Download and extract DIV2K HR images.

    python scripts/download_div2k.py --config configs/baseline.yaml
    python scripts/download_div2k.py --splits valid              # only the 100 validation images
    python scripts/download_div2k.py --cache-dir /content/drive/MyDrive/CMPE258_Team7/cache

With --cache-dir the zip files are kept there (e.g. on Google Drive) and reused
on later runs instead of being downloaded again.
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm.config import load_config  # noqa: E402

BASE_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/"
SPLITS = {
    "train": ("DIV2K_train_HR.zip", "DIV2K_train_HR", 800),
    "valid": ("DIV2K_valid_HR.zip", "DIV2K_valid_HR", 100),
}


def _progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    print(f"\r  {done / 2**20:7.1f} / {total_size / 2**20:.1f} MB", end="", flush=True)


def download(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, tmp, _progress)
    print()
    tmp.replace(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"], choices=sorted(SPLITS))
    parser.add_argument("--cache-dir", default=None, help="Directory to keep/reuse the zip files")
    args = parser.parse_args()

    raw_root = Path(load_config(args.config)["data"]["raw_root"])
    raw_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else raw_root
    cache_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        zip_name, folder, expected = SPLITS[split]
        target = raw_root / folder
        if target.is_dir() and len(list(target.glob("*.png"))) >= expected:
            print(f"{target} already has {expected} images, skipping")
            continue
        zip_path = cache_dir / zip_name
        if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
            download(BASE_URL + zip_name, zip_path)
        print(f"Extracting {zip_path} -> {raw_root}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_root)
        count = len(list(target.glob("*.png")))
        if count < expected:
            raise RuntimeError(f"Expected {expected} images in {target}, found {count}")
        print(f"{split}: {count} images in {target}")
        if args.cache_dir is None and zip_path.parent == raw_root:
            zip_path.unlink()  # keep only the extracted images locally

    macosx = raw_root / "__MACOSX"
    if macosx.exists():
        shutil.rmtree(macosx)


if __name__ == "__main__":
    main()
