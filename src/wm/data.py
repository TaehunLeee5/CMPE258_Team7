"""DIV2K loading and 128x128 crop preprocessing.

`scripts/prepare_div2k.py` downscales a fixed subset of DIV2K once (short side =
`data.prepare_short_side`) into `data.prepared_root/{train,val}`. Those prepared
images are small enough to keep in RAM as uint8 tensors, so crops are cheap and
no worker processes are needed.

Training samples are random crops (+ horizontal flip); validation samples are
fixed crops whose positions are drawn once from the experiment seed, so every
evaluation sees exactly the same pixels.
"""

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
    directory = Path(data_cfg["prepared_root"]) / split
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Prepared DIV2K split not found at {directory}. Run scripts/download_div2k.py and "
            "scripts/prepare_div2k.py first (or set data.synthetic=true for a smoke test)."
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


def build_datasets(cfg):
    return build_train_dataset(cfg), build_val_dataset(cfg)


def _loader(dataset, cfg, drop_last):
    return DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, drop_last=drop_last,
                      num_workers=cfg["data"].get("num_workers", 0), pin_memory=torch.cuda.is_available())


def build_val_loader(cfg):
    return _loader(build_val_dataset(cfg), cfg, drop_last=False)


def build_loaders(cfg):
    return _loader(build_train_dataset(cfg), cfg, drop_last=True), build_val_loader(cfg)
