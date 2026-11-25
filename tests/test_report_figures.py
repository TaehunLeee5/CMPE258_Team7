import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report_figures.py"
spec = importlib.util.spec_from_file_location("report_figures", SCRIPT)
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


def _run(base, name, worst, mean, psnr):
    run_dir = base / "runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / "best_test_eval.json").write_text(json.dumps({
        "image_quality": {"psnr": psnr, "ssim": 0.98},
        "worst_case": {"attack": "crop", "bit_acc": worst},
        "mean_attacked_bit_acc": mean,
    }))


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    main, variants = tmp_path / "main", tmp_path / "variants"
    for seed, shift in ((42, 0.0), (43, 0.01)):
        _run(main, f"hard_uniform_s{seed}", 0.86 + shift, 0.975, 36.0 + shift)
        _run(main, f"hard_fgas_s{seed}", 0.93 + shift, 0.983, 35.7 + shift)
    _run(main, "hard_fgas_p2_s42", 0.5, 0.5, 30.0)  # must not be mistaken for a hard_fgas seed
    _run(variants, "hard_uniform_w100_s42", 0.90, 0.985, 33.0)
    _run(variants, "hard_fgas_w400_s42", 0.88, 0.97, 38.5)
    _run(variants, "hard_fgas_lossweight_s42", 0.89, 0.979, 35.9)
    monkeypatch.setattr(figures, "HARD_DIR", main)
    monkeypatch.setattr(figures, "VARIANTS_DIR", variants)
    return main, variants


def test_weight_200_point_uses_only_main_seeds(dirs):
    main, _ = dirs
    runs = figures._runs(main, "hard_fgas_s*")
    assert len(runs) == 2 and all(r["psnr"] > 35 for r in runs)


def test_trade_off_and_ablations_write_figures_with_partial_runs(dirs, tmp_path):
    out = tmp_path / "figures"
    out.mkdir()
    figures.trade_off(out)
    figures.ablations(out)
    assert (out / "trade_off.png").exists() and (out / "ablations.png").exists()


def test_figures_skip_without_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "HARD_DIR", tmp_path / "none")
    monkeypatch.setattr(figures, "VARIANTS_DIR", tmp_path / "none")
    figures.trade_off(tmp_path)
    figures.ablations(tmp_path)
    assert not list(tmp_path.glob("*.png"))


def _per_run_csv(path, runs):
    attacks = figures.HARD_ATTACKS
    header = ["run", "psnr"] + [f"{a}_bit_acc" for a in attacks]
    lines = [",".join(header)] + [",".join([name, "36.0"] + [str(acc)] * len(attacks)) for name, acc in runs]
    path.write_text("\n".join(lines) + "\n")


def test_per_run_strip_writes_figure_with_a_stuck_run(tmp_path, monkeypatch):
    _per_run_csv(tmp_path / "per_run.csv", [("hard_uniform_s42", 0.97), ("hard_uniform_s43", 0.96),
                                            ("hard_fgas_s42", 0.98), ("hard_fgas_s43", 0.5)])
    monkeypatch.setattr(figures, "HARD_DIR", tmp_path)
    figures.per_run_strip(tmp_path)
    assert (tmp_path / "per_run_strip.png").exists()


def test_per_run_strip_handles_unknown_group_names(tmp_path, monkeypatch):
    _per_run_csv(tmp_path / "per_run.csv", [("hard_uniform_warm0_s42", 0.97), ("hard_static_warm0_s42", 0.95),
                                            ("hard_fgas_warm0_s42", 0.98), ("hard_fgas_warm0_s43", 0.97)])
    monkeypatch.setattr(figures, "HARD_DIR", tmp_path)
    figures.per_run_strip(tmp_path)
    assert (tmp_path / "per_run_strip.png").exists()


def test_per_run_strip_skips_missing_csv(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(figures, "HARD_DIR", tmp_path / "none")
    figures.per_run_strip(tmp_path)
    assert "skipped:" in capsys.readouterr().out and not list(tmp_path.glob("*.png"))


def test_warm0_groups_get_bars_and_fgas_history(tmp_path, monkeypatch):
    groups = ("hard_uniform_warm0", "hard_static_warm0", "hard_fgas_warm0")
    lines = ["group,metric,mean,std,n"]
    for group in groups:
        lines += [f"{group},{a}_bit_acc,0.97,0.01,5" for a in figures.HARD_ATTACKS] + [f"{group},psnr,35.5,0.3,5"]
    (tmp_path / "main_comparison.csv").write_text("\n".join(lines) + "\n")
    run = tmp_path / "runs" / "hard_fgas_warm0_s42"  # no hard_fgas_s42 here: the warm0 run is used
    run.mkdir(parents=True)
    rows = [{"attack": a, "old_prob": 0.5, "new_prob": 0.5} for a in ("noise", "crop")]
    run.joinpath("fgas_history.json").write_text(json.dumps([{"step": 2, "rows": rows}, {"step": 4, "rows": rows}]))
    monkeypatch.setattr(figures, "HARD_DIR", tmp_path)
    figures.method_bars(tmp_path)
    figures.fgas_over_time(tmp_path)
    assert (tmp_path / "method_bars.png").exists() and (tmp_path / "fgas_over_time.png").exists()
