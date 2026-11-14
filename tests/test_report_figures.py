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
