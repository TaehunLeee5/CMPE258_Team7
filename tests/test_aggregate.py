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


def test_paired_deltas_over_common_seeds(tmp_path):
    dirs = _write_runs(tmp_path, {
        "u_s42": _result(0.95, 0.80, 38.0),
        "u_s43": _result(0.95, 0.84, 38.2),
        "u_s44": _result(0.95, 0.82, 38.0),   # no FGAS partner -> not paired
        "f_s42": _result(0.95, 0.86, 37.8),   # worst +6 pp, PSNR -0.2 dB
        "f_s43": _result(0.95, 0.82, 38.2),   # worst -2 pp, PSNR 0
    })
    groups = aggregate.group_runs(aggregate.load_runs(dirs, "best_test_eval.json"))
    paired = aggregate.paired_deltas(groups, "u", "f")
    assert paired["seeds"] == [42, 43] and paired["num_paired_seeds"] == 2
    worst = paired["summary"]["worst_bit_acc"]
    assert [row["worst_bit_acc"] for row in paired["per_seed"]] == pytest.approx([6.0, -2.0])
    assert worst["mean"] == pytest.approx(2.0) and worst["std"] == pytest.approx(5.6569, abs=1e-4)
    assert paired["summary"]["psnr"]["mean"] == pytest.approx(-0.1)
    # msg_acc = bit_acc / 2 in _result, so the message delta is half the mean bit delta
    assert paired["summary"]["mean_attacked_msg_acc"]["mean"] == pytest.approx(
        paired["summary"]["mean_attacked_bit_acc"]["mean"] / 2)
    assert paired["fgas_better_worst_case"] == 1
    assert "1 of 2 seeds" in aggregate.format_paired(paired)


def test_mean_msg_acc_uses_config_train_attacks_for_old_jsons(tmp_path):
    result = _result(0.9, 0.8, 38.0)
    result["attacks"]["jpeg_real"] = {"bit_acc": 0.5, "msg_acc": 0.0}  # evaluation-only attack
    (run_dir,) = _write_runs(tmp_path, {"u_s42": result})
    (run,) = aggregate.load_runs([run_dir], "best_test_eval.json")
    assert aggregate.mean_attacked_msg_acc(run) == pytest.approx((0.9 + 0.8 + 0.0) / 2 / 3)  # no config: all attacked
    (run_dir / "config.yaml").write_text("attacks:\n  train: [noise, jpeg]\n")
    (run,) = aggregate.load_runs([run_dir], "best_test_eval.json")
    assert aggregate.mean_attacked_msg_acc(run) == pytest.approx((0.45 + 0.40) / 2)
    run["mean_attacked_msg_acc"] = 0.123  # new JSONs carry it directly
    assert aggregate.mean_attacked_msg_acc(run) == 0.123


def test_seed42_reproduces_committed_comparison(tmp_path):
    runs_dir = Path(__file__).resolve().parents[1] / "results" / "checkin3" / "runs"
    dirs = [runs_dir / "hard_uniform_s42", runs_dir / "hard_fgas_s42"]
    if not all((d / "best_test_eval.json").is_file() for d in dirs):
        pytest.skip("committed seed-42 runs not available")
    summaries = aggregate.write_grouped(aggregate.group_runs(aggregate.load_runs(dirs, "best_test_eval.json")),
                                        tmp_path / "table")
    assert summaries["hard_uniform"]["mean_attacked_msg_acc"][0] == pytest.approx(0.7825, abs=5e-5)
    assert summaries["hard_fgas"]["mean_attacked_msg_acc"][0] == pytest.approx(0.7729, abs=5e-5)
    result = aggregate.check_criterion(summaries, "hard_uniform", "hard_fgas")
    assert result["met"] and result["delta_worst_pp"] == pytest.approx(7.359375)
