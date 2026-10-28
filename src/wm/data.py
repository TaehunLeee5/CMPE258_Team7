"""DIV2K / COCO loading and 128x128 crop preprocessing.

`scripts/prepare_div2k.py` (or `scripts/prepare_coco.py` with `data.dataset: coco`)
downscales a fixed subset once (short side = `data.prepare_short_side`) into
`<prepared root>/{train,val}`, where the prepared root is `data.prepared_root`
for DIV2K and `data.coco_prepared_root` for COCO. Those prepared images are
small enough to keep in RAM as uint8 tensors, so crops are cheap and no worker
processes are needed.

Training samples are random crops (+ horizontal flip); validation samples are
fixed crops whose positions are drawn once from the experiment seed, so every
evaluation sees exactly the same pixels.

The optional held-out test set (`data.test`, prepared DIV2K) uses fixed crops
drawn from `data.test.seed` instead, so every training seed is tested on the
same pixels.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def list_images(directory):
    directory = Path(directory)
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_image(path):
    """Load an RGB image as a uint8 CHW tensor."""
    with Image.open(path) as img:
        array = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()


def load_images(directory, max_images=None):
    paths = list_images(directory)
    if max_images is not None:
        paths = paths[:max_images]
    return [load_image(p) for p in paths]


def dataset_name(data_cfg):
    name = data_cfg.get("dataset", "div2k")
    if name not in ("div2k", "coco"):
        raise ValueError(f"data.dataset must be 'div2k' or 'coco', got {name!r}")
    return name


def active_data_cfg(data_cfg):
    """`data_cfg` with `prepared_root` pointing at the configured dataset's prepared cache."""
    if dataset_name(data_cfg) == "coco":
        return {**data_cfg, "prepared_root": data_cfg["coco_prepared_root"]}
    return data_cfg


def validate_prepared_manifest(data_cfg):
    """Validate cached files (DIV2K or COCO) against the preprocessing manifest and config."""
    root = Path(data_cfg["prepared_root"])
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared dataset manifest not found at {manifest_path}")
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    short_side = data_cfg["prepare_short_side"]
    if manifest.get("short_side") != short_side:
        raise ValueError(
            f"Manifest short_side is {manifest.get('short_side')}, expected {short_side}"
        )

    summary = {"short_side": short_side}
    for split, count_key in (("train", "num_train_images"), ("val", "num_val_images")):
        entries = manifest.get(split)
        expected_count = data_cfg[count_key]
        if not isinstance(entries, list) or len(entries) != expected_count:
            actual_count = len(entries) if isinstance(entries, list) else "missing"
            raise ValueError(f"Manifest {split} count is {actual_count}, expected {expected_count}")

        split_dir = root / split
        expected_names = [Path(entry["source"]).with_suffix(".png").name for entry in entries]
        actual_names = [path.name for path in list_images(split_dir)]
        if len(set(expected_names)) != len(expected_names) or sorted(actual_names) != sorted(expected_names):
            raise ValueError(f"Prepared {split} filenames do not match {manifest_path}")

        for entry, filename in zip(entries, expected_names):
            original_size = tuple(entry["original_size"])
            prepared_size = tuple(entry["prepared_size"])
            if len(original_size) != 2 or len(prepared_size) != 2:
                raise ValueError(f"Invalid dimensions in {split} manifest entry for {filename}")
            scale = short_side / min(original_size)
            expected_size = (round(original_size[0] * scale), round(original_size[1] * scale))
            if prepared_size != expected_size or min(prepared_size) != short_side:
                raise ValueError(f"Invalid prepared dimensions in {split} manifest entry for {filename}")
            with Image.open(split_dir / filename) as image:
                if image.mode != "RGB" or image.size != prepared_size:
                    raise ValueError(
                        f"{split_dir / filename} has mode/size {image.mode}/{image.size}, "
                        f"expected RGB/{prepared_size}"
                    )
        summary[f"{split}_images"] = expected_count
    return summary


def synthetic_images(count, height, width, seed):
    """Random smooth RGB images for tests and smoke runs (no download needed)."""
    generator = torch.Generator().manual_seed(seed)
    images = []
    for _ in range(count):
        coarse = torch.rand(1, 3, 8, 8, generator=generator)
        smooth = F.interpolate(coarse, size=(height, width), mode="bicubic", align_corners=False)
        detail = 0.05 * torch.randn(1, 3, height, width, generator=generator)
        image = (smooth + detail).clamp(0, 1)[0]
        images.append((image * 255).round().to(torch.uint8))
    return images


def _check_size(images, image_size):
    for image in images:
        if min(image.shape[1:]) < image_size:
            raise ValueError(f"Image of shape {tuple(image.shape)} is smaller than crop size {image_size}")


class RandomCropDataset(Dataset):
    """`samples_per_epoch` random crops drawn from a list of uint8 images."""

    def __init__(self, images, image_size, samples_per_epoch, hflip=True):
        _check_size(images, image_size)
        self.images = images
        self.image_size = image_size
        self.samples_per_epoch = samples_per_epoch
        self.hflip = hflip

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        image = self.images[torch.randint(len(self.images), (1,)).item()]
        _, height, width = image.shape
        size = self.image_size
        top = torch.randint(height - size + 1, (1,)).item()
        left = torch.randint(width - size + 1, (1,)).item()
        crop = image[:, top:top + size, left:left + size]
        if self.hflip and torch.rand(1).item() < 0.5:
            crop = crop.flip(-1)
        return crop.float() / 255.0


class FixedCropDataset(Dataset):
    """Deterministic crops: positions are drawn once from `seed`."""

    def __init__(self, images, image_size, crops_per_image, seed):
        _check_size(images, image_size)
        self.images = images
        self.image_size = image_size
        rng = np.random.default_rng(seed)
        self.crops = []
        for idx, image in enumerate(images):
            _, height, width = image.shape
            for _ in range(crops_per_image):
                top = int(rng.integers(0, height - image_size + 1))
                left = int(rng.integers(0, width - image_size + 1))
                self.crops.append((idx, top, left))

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, index):
        idx, top, left = self.crops[index]
        size = self.image_size
        crop = self.images[idx][:, top:top + size, left:left + size]
        return crop.float() / 255.0


def load_split_images(data_cfg, split, seed):
    """Return the prepared `split` ("train" or "val") images as uint8 CHW tensors."""
    count = data_cfg["num_train_images"] if split == "train" else data_cfg["num_val_images"]
    if data_cfg.get("synthetic", False):
        side = max(data_cfg["image_size"] * 2, 64)
        return synthetic_images(count, side, side, seed if split == "train" else seed + 1)
    name = dataset_name(data_cfg)
    directory = Path(active_data_cfg(data_cfg)["prepared_root"]) / split
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Prepared {name} split not found at {directory}. Run scripts/download_{name}.py and "
            f"scripts/prepare_{name}.py first (or set data.synthetic=true for a smoke test)."
        )
    return load_images(directory, count)


def build_train_dataset(cfg):
    data_cfg = cfg["data"]
    images = load_split_images(data_cfg, "train", cfg["experiment"]["seed"])
    return RandomCropDataset(images, data_cfg["image_size"], data_cfg["train_samples_per_epoch"])


def build_val_dataset(cfg):
    data_cfg, seed = cfg["data"], cfg["experiment"]["seed"]
    images = load_split_images(data_cfg, "val", seed)
    return FixedCropDataset(images, data_cfg["image_size"], data_cfg["val_crops_per_image"], seed)


def build_test_dataset(cfg):
    """Fixed crops of the held-out test set (`data.test`); crops come from `data.test.seed`."""
    data_cfg = cfg["data"]
    test_cfg = data_cfg.get("test")
    if not test_cfg:
        raise ValueError("No held-out test set configured (data.test)")
    seed, count = test_cfg.get("seed", 0), test_cfg["num_images"]
    if data_cfg.get("synthetic", False):
        side = max(data_cfg["image_size"] * 2, 64)
        images = synthetic_images(count, side, side, seed + 2)
    else:
        directory = Path(test_cfg["prepared_root"]) / test_cfg["split"]
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Prepared test split not found at {directory}. Run scripts/download_div2k.py --splits valid "
                "and scripts/prepare_div2k.py --splits val first."
            )
        images = load_images(directory, count)
        if len(images) != count:
            raise ValueError(f"Test set needs {count} images in {directory}, found {len(images)}")
    return FixedCropDataset(images, data_cfg["image_size"], test_cfg["crops_per_image"], seed)


def build_datasets(cfg):
    return build_train_dataset(cfg), build_val_dataset(cfg)


def _loader(dataset, cfg, drop_last):
    return DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, drop_last=drop_last,
                      num_workers=cfg["data"].get("num_workers", 0), pin_memory=torch.cuda.is_available())


def build_val_loader(cfg):
    return _loader(build_val_dataset(cfg), cfg, drop_last=False)


def build_test_loader(cfg):
    return _loader(build_test_dataset(cfg), cfg, drop_last=False)


def build_loaders(cfg):
    return _loader(build_train_dataset(cfg), cfg, drop_last=True), build_val_loader(cfg)
