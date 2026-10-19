import torch

from wm.config import apply_override
from wm.data import FixedCropDataset, build_datasets, synthetic_images


def test_override_parses_yaml_types():
    cfg = {"train": {"lr": 0.1}}
    apply_override(cfg, "train.lr=0.002")
    apply_override(cfg, "data.synthetic=true")
    assert cfg["train"]["lr"] == 0.002 and cfg["data"]["synthetic"] is True


def test_crops_have_expected_shape_and_range(small_cfg):
    train_ds, val_ds = build_datasets(small_cfg)
    size = small_cfg["data"]["image_size"]
    assert len(train_ds) == small_cfg["data"]["train_samples_per_epoch"]
    assert len(val_ds) == small_cfg["data"]["num_val_images"] * small_cfg["data"]["val_crops_per_image"]
    for sample in (train_ds[0], val_ds[0]):
        assert sample.shape == (3, size, size)
        assert sample.dtype == torch.float32
        assert 0.0 <= sample.min() and sample.max() <= 1.0


def test_validation_crops_are_deterministic():
    images = synthetic_images(3, 64, 80, seed=0)
    a = FixedCropDataset(images, 32, 2, seed=7)
    b = FixedCropDataset(images, 32, 2, seed=7)
    assert a.crops == b.crops
    assert torch.equal(a[3], b[3])
