import importlib.util
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from wm.config import apply_override, load_config
from wm.data import active_data_cfg, build_test_dataset, build_train_dataset, validate_prepared_manifest

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare_coco = _script("prepare_coco")
validate_coco = _script("validate_coco")


def test_split_is_by_sorted_filename_and_drops_small_images():
    sizes = {f"{i:012d}.jpg": (640, 480) for i in range(8)}
    sizes["000000000003.jpg"] = (640, 120)  # short side < 128 -> dropped
    train, val, dropped = prepare_coco.split_coco(dict(reversed(sizes.items())), num_train=4, num_val=2)
    assert dropped == ["000000000003.jpg"]
    assert val == ["000000000006.jpg", "000000000007.jpg"]
    assert train == ["000000000000.jpg", "000000000001.jpg", "000000000002.jpg", "000000000004.jpg"]
    with pytest.raises(RuntimeError, match="lower data.num_train_images"):
        prepare_coco.split_coco(sizes, num_train=6, num_val=2)


def test_prepared_coco_matches_manifest_and_is_png(tmp_path):
    src = tmp_path / "val2017"
    src.mkdir()
    for i, size in enumerate([(640, 480), (480, 640), (500, 375), (612, 612), (640, 100)]):
        Image.new("RGB", size, (i * 40, 80, 120)).save(src / f"{i:012d}.jpg")
    sizes = {p.name: Image.open(p).size for p in sorted(src.iterdir())}
    train, val, dropped = prepare_coco.split_coco(sizes, num_train=3, num_val=1)
    assert dropped == ["000000000004.jpg"]

    out = tmp_path / "coco_prepared"
    manifest = {"short_side": 256}
    for split, names in (("train", train), ("val", val)):
        manifest[split] = prepare_coco.prepare_images([src / n for n in names], out / split, 256)
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    for path in (out / "train").iterdir():
        with Image.open(path) as img:
            assert img.format == "PNG" and min(img.size) == 256
    data_cfg = {"dataset": "coco", "prepared_root": "data/div2k_prepared", "coco_prepared_root": str(out),
                "prepare_short_side": 256, "num_train_images": 3, "num_val_images": 1}
    assert validate_prepared_manifest(active_data_cfg(data_cfg)) == {
        "short_side": 256, "train_images": 3, "val_images": 1}
    validate_coco.check_split_order(manifest)
    with pytest.raises(ValueError, match="sort after"):
        validate_coco.check_split_order({"train": manifest["val"], "val": manifest["train"][:1]})


def test_coco_dataset_reads_coco_prepared_root(tmp_path):
    cfg = load_config(ROOT / "configs" / "hard_base.yaml", [
        f"data.coco_prepared_root={tmp_path.as_posix()}", "data.num_train_images=2", "data.image_size=32"])
    (tmp_path / "train").mkdir()
    for i in range(2):
        Image.new("RGB", (64, 48)).save(tmp_path / "train" / f"{i}.png")
    assert len(build_train_dataset(cfg).images) == 2
    apply_override(cfg, "data.dataset=imagenet")
    with pytest.raises(ValueError, match="data.dataset"):
        build_train_dataset(cfg)


def test_test_set_ignores_experiment_seed(small_cfg):
    apply_override(small_cfg, "data.test={num_images: 2, crops_per_image: 3, seed: 0}")
    a = build_test_dataset(small_cfg)
    apply_override(small_cfg, "experiment.seed=43")
    b = build_test_dataset(small_cfg)
    assert len(a) == 6 and a.crops == b.crops
    assert torch.equal(a[5], b[5])
    apply_override(small_cfg, "data.test.seed=1")
    assert build_test_dataset(small_cfg).crops != a.crops


def test_test_set_reads_prepared_div2k(small_cfg, tmp_path):
    (tmp_path / "val").mkdir()
    for i in range(3):
        Image.new("RGB", (80, 64), (i, i, i)).save(tmp_path / "val" / f"{i:04d}.png")
    apply_override(small_cfg, "data.synthetic=false")
    apply_override(small_cfg, f"data.test={{prepared_root: {tmp_path.as_posix()}, split: val, "
                              "num_images: 3, crops_per_image: 2, seed: 0}")
    assert len(build_test_dataset(small_cfg)) == 6
    apply_override(small_cfg, "data.test.num_images=4")
    with pytest.raises(ValueError, match="needs 4 images"):
        build_test_dataset(small_cfg)
