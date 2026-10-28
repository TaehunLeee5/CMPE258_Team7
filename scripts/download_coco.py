"""Download and extract COCO val2017 (5000 images), the training data of the hardened benchmark.

    python scripts/download_coco.py --config configs/hard_base.yaml
    python scripts/download_coco.py --cache-dir /content/drive/MyDrive/CMPE258_Team7/cache

Images are extracted to `data.coco_raw_root/val2017`. With --cache-dir the zip
(~780 MB) is kept there (e.g. on Google Drive) and reused on later runs instead
of being downloaded again.
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm.config import load_config  # noqa: E402

URL = "http://images.cocodataset.org/zips/val2017.zip"
ZIP_NAME, FOLDER, EXPECTED = "val2017.zip", "val2017", 5000


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


def count_images(folder):
    return len(list(folder.glob("*.jpg"))) if folder.is_dir() else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/hard_base.yaml")
    parser.add_argument("--cache-dir", default=None, help="Directory to keep/reuse the zip file")
    args = parser.parse_args()

    raw_root = Path(load_config(args.config)["data"]["coco_raw_root"])
    raw_root.mkdir(parents=True, exist_ok=True)
    target = raw_root / FOLDER
    if count_images(target) == EXPECTED:
        print(f"{target} already has {EXPECTED} images, skipping")
        return

    cache_dir = Path(args.cache_dir) if args.cache_dir else raw_root
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / ZIP_NAME
    if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
        download(URL, zip_path)
    print(f"Extracting {zip_path} -> {raw_root}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_root)
    count = count_images(target)
    assert count == EXPECTED, f"Expected {EXPECTED} images in {target}, found {count}"
    print(f"val2017: {count} images in {target}")
    if args.cache_dir is None:
        zip_path.unlink()  # keep only the extracted images locally


if __name__ == "__main__":
    main()
