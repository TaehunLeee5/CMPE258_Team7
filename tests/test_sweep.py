import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from wm.config import apply_override
from wm.models import build_model

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sweep.py"
spec = importlib.util.spec_from_file_location("sweep", SCRIPT)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def test_level_labels_and_params_keep_base_settings():
    blur = [{"kernel_size": 9, "sigma": 1.0}, {"kernel_size": 13, "sigma": 3.0}]
    brightness = [{"min_factor": 0.7, "max_factor": 1.3}, {"min_factor": 0.5, "max_factor": 1.5}]
    assert sweep.level_label(blur, 1) == "3.0"
    assert sweep.level_label(brightness, 0) == "0.7-1.3"
    assert sweep.level_label([{"keep": 0.9}], 0) == "0.9"

    params = sweep.sweep_attack_params({"occlusion": [{"area": 0.4}]},
                                       {"occlusion": {"area": [0.1, 0.25], "fill": 0.5}})
    assert params == {"occlusion_l0": {"type": "occlusion", "area": 0.4, "fill": 0.5}}


def _write(path, method, seed, accs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(sweep.COLUMNS)
        writer.writerow([method, seed, "clean", 0, "", 1.0, 1.0])
        for index, acc in enumerate(accs):
            writer.writerow([method, seed, "crop", index, ["0.9", "0.5"][index], acc, acc / 2])


def test_plot_aggregates_seeds_per_method(tmp_path):
    paths = []
    for method, seed, accs in [("hard_uniform", 42, [0.9, 0.6]), ("hard_uniform", 43, [0.8, 0.6]),
                               ("hard_fgas", 42, [0.95, 0.7])]:
        paths.append(tmp_path / f"{method}_s{seed}.csv")
        _write(paths[-1], method, seed, accs)
    summary = sweep.summarize(sweep.load_sweeps(paths))
    assert set(summary) == {"crop"}  # clean is not plotted
    index, label, mean, std, n = summary["crop"]["hard_uniform"][0]
    assert (index, label, n) == (0, "0.9", 2)
    assert mean == pytest.approx(0.85) and std == pytest.approx(0.0707, abs=1e-4)

    out = tmp_path / "figures" / "severity_sweep.png"
    sys.argv = ["sweep.py", "--plot", *map(str, paths), "--out", str(out)]
    sweep.main()
    assert out.stat().st_size > 0


def test_sweep_checkpoint_on_synthetic_test_set(small_cfg, tmp_path):
    apply_override(small_cfg, "experiment.name=hard_uniform_s43")
    apply_override(small_cfg, "experiment.seed=43")
    apply_override(small_cfg, "data.test={num_images: 2, crops_per_image: 2, seed: 0}")
    apply_override(small_cfg, "attacks.sweep={noise: [{std: 0.05}, {std: 0.2}], crop: [{keep: 0.7}]}")
    ckpt = tmp_path / "best.pt"
    torch.save({"cfg": small_cfg, "model": build_model(small_cfg).state_dict(), "epoch": 1}, ckpt)

    out = tmp_path / "sweep.csv"
    sys.argv = ["sweep.py", "--checkpoint", str(ckpt), "--out", str(out)]
    sweep.main()
    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == sweep.COLUMNS
    assert [(r["attack"], r["level_idx"], r["level"]) for r in rows] == [
        ("clean", "0", ""), ("noise", "0", "0.05"), ("noise", "1", "0.2"), ("crop", "0", "0.7")]
    assert {(r["method"], r["seed"]) for r in rows} == {("hard_uniform", "43")}
