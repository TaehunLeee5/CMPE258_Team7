import csv
import json

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
