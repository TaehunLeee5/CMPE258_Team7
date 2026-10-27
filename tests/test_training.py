import csv
import json

import pytest
import torch

from wm.config import apply_override
from wm.trainer import Trainer


def test_one_epoch_writes_artifacts(small_cfg, tmp_path):
    apply_override(small_cfg, f"experiment.output_dir={tmp_path.as_posix()}")
    apply_override(small_cfg, "train.epochs=1")
    result = Trainer(small_cfg).fit()

    run_dir = tmp_path / small_cfg["experiment"]["name"]
    for name in ("config.yaml", "run_info.json", "metrics.csv", "metrics.jsonl", "best.pt", "last.pt", "best_eval.json"):
        assert (run_dir / name).exists(), name
    with open(run_dir / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    batches = small_cfg["data"]["train_samples_per_epoch"] // small_cfg["train"]["batch_size"]
    assert sum(int(rows[0][f"count_{a}"]) for a in small_cfg["attacks"]["train"]) == batches
    assert set(result["attacks"]) == set(small_cfg["attacks"]["eval"])
    assert json.loads((run_dir / "best_eval.json").read_text())["epoch"] == 1


def test_resume_continues_from_last_epoch(small_cfg, tmp_path):
    apply_override(small_cfg, f"experiment.output_dir={tmp_path.as_posix()}")
    apply_override(small_cfg, "train.epochs=1")
    Trainer(small_cfg).fit()
    last = tmp_path / small_cfg["experiment"]["name"] / "last.pt"
    apply_override(small_cfg, "train.epochs=2")
    apply_override(small_cfg, f"train.resume={last.as_posix()}")
    trainer = Trainer(small_cfg)
    assert trainer.start_epoch == 2
    trainer.fit()
    assert torch.load(last, weights_only=False)["epoch"] == 2


def test_fgas_run_updates_sampler_from_validation(small_cfg, tmp_path):
    apply_override(small_cfg, f"experiment.output_dir={tmp_path.as_posix()}")
    apply_override(small_cfg, "train.epochs=2")
    apply_override(small_cfg, "fgas.enabled=true")
    apply_override(small_cfg, "fgas.update_at_start=true")
    trainer = Trainer(small_cfg)
    trainer.fit()

    history = json.loads((tmp_path / small_cfg["experiment"]["name"] / "fgas_history.json").read_text())
    assert [h["step"] for h in history] == [0, 1]  # at start, after epoch 1 (not after the final epoch)
    latest = {r["attack"]: r["new_prob"] for r in history[-1]["rows"]}
    assert trainer.sampler.probs == pytest.approx(latest)
    with open(tmp_path / small_cfg["experiment"]["name"] / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    first_update = {r["attack"]: r["new_prob"] for r in history[0]["rows"]}
    assert float(rows[0]["prob_resize"]) == pytest.approx(first_update["resize"])


def test_image_loss_weight_waits_for_trigger_then_ramps():
    from wm.trainer import image_loss_weight
    cfg = {"image_loss_weight": 20.0, "image_loss_weight_start": 1.0, "image_loss_warmup_epochs": 4}
    assert image_loss_weight(cfg, 5, ramp_start=None) == 1.0
    assert [image_loss_weight(cfg, e, ramp_start=3) for e in (3, 4, 5, 6, 9)] == [5.75, 10.5, 15.25, 20.0, 20.0]
    assert image_loss_weight({**cfg, "image_loss_warmup_epochs": 0}, 1, ramp_start=None) == 20.0


def test_best_checkpoint_is_scored_on_the_test_set(small_cfg, tmp_path, monkeypatch):
    import wm.data
    from wm.data import build_val_loader

    calls = []

    def fake_test_loader(cfg):  # stands in for the held-out test split
        calls.append(cfg["data"]["test"]["seed"])
        return build_val_loader(cfg)

    monkeypatch.setattr(wm.data, "build_test_loader", fake_test_loader, raising=False)
    apply_override(small_cfg, f"experiment.output_dir={tmp_path.as_posix()}")
    apply_override(small_cfg, "train.epochs=1")
    apply_override(small_cfg, "data.test={seed: 0}")
    apply_override(small_cfg, "attacks.extra_eval=[jpeg_real]")
    Trainer(small_cfg).fit()

    report = json.loads((tmp_path / small_cfg["experiment"]["name"] / "best_test_eval.json").read_text())
    assert calls == [0] and report["split"] == "test" and report["epoch"] == 1
    assert "jpeg_real" in report["attacks"]  # evaluation-only attacks are reported...
    assert report["worst_case"]["attack"] in small_cfg["attacks"]["train"]  # ...but never the primary metric


def test_no_test_evaluation_without_test_block(small_cfg, tmp_path):
    apply_override(small_cfg, f"experiment.output_dir={tmp_path.as_posix()}")
    apply_override(small_cfg, "train.epochs=1")
    Trainer(small_cfg).fit()
    assert not (tmp_path / small_cfg["experiment"]["name"] / "best_test_eval.json").exists()
