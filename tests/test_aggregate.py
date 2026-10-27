import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_seeds.py"
spec = importlib.util.spec_from_file_location("aggregate_seeds", SCRIPT)
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)


def _result(noise, jpeg, psnr, clean=1.0):
    attacks = {"clean": clean, "noise": noise, "jpeg": jpeg}
    worst = min(("noise", "jpeg"), key=lambda a: attacks[a])
    return {
        "epoch": 60,
        "attacks": {a: {"bit_acc": v, "msg_acc": v / 2} for a, v in attacks.items()},
        "image_quality": {"psnr": psnr, "ssim": 0.98},
        "worst_case": {"attack": worst, "bit_acc": attacks[worst]},
        "mean_attacked_bit_acc": (noise + jpeg) / 2,
    }


def _write_runs(tmp_path, runs):
    dirs = []
    for name, result in runs.items():
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "best_test_eval.json").write_text(json.dumps(result))
        dirs.append(run_dir)
    return dirs


def test_groups_strip_seed_suffix_and_compute_mean_std(tmp_path):
    dirs = _write_runs(tmp_path, {
        "hard_uniform_s42": _result(0.95, 0.80, 38.0),
        "hard_uniform_s43": _result(0.97, 0.84, 38.2),
        "hard_fgas_s42": _result(0.95, 0.86, 38.1),
    })
    groups = aggregate.group_runs(aggregate.load_runs(dirs, "best_test_eval.json"))
    assert sorted(groups) == ["hard_fgas", "hard_uniform"]
    summary = aggregate.summarize(groups["hard_uniform"])
    mean, std, n = summary["jpeg_bit_acc"]
    assert n == 2 and mean == pytest.approx(0.82) and std == pytest.approx(0.0283, abs=1e-4)
    assert summary["worst_attack"] == "jpeg"
    assert aggregate.summarize(groups["hard_fgas"])["psnr"][1] == 0.0  # single seed -> std 0


def test_criterion_flags(tmp_path):
    dirs = _write_runs(tmp_path, {
        "u_s42": _result(0.95, 0.80, 38.0),
        "f_s42": _result(0.95, 0.83, 38.3),   # +3 pp worst, mean +1.5 pp, PSNR +0.3 dB
        "g_s42": _result(0.90, 0.81, 39.0),   # +1 pp worst, PSNR +1.0 dB, mean -2 pp
    })
    out = tmp_path / "table"
    summaries = aggregate.write_grouped(aggregate.group_runs(aggregate.load_runs(dirs, "best_test_eval.json")), out)
    good = aggregate.check_criterion(summaries, "u", "f")
    assert good["met"] and good["delta_worst_pp"] == pytest.approx(3.0)
    bad = aggregate.check_criterion(summaries, "u", "g")
    assert not bad["met"]
    assert not bad["checks"]["worst_case_gain_ge_2pp"]
    assert not bad["checks"]["psnr_within_0.5db"]
    assert not bad["checks"]["mean_attacked_drop_le_0.5pp"]
    assert out.with_suffix(".csv").exists() and "jpeg_bit_acc" in out.with_suffix(".md").read_text(encoding="utf-8")


def test_per_run_table_reports_spread(tmp_path):
    dirs = _write_runs(tmp_path, {"calib_w50": _result(0.99, 0.85, 37.0), "calib_w100": _result(0.97, 0.75, 38.5)})
    out = tmp_path / "summary"
    aggregate.write_per_run(aggregate.load_runs(dirs, "best_test_eval.json"), out)
    rows = out.with_suffix(".csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].endswith("spread_pp") and len(rows) == 3
    assert rows[1].split(",")[-1] == "14.0"
