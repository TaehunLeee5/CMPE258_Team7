"""Aggregate evaluation results over seeds (mean +- std) and check the success criterion.

    python scripts/aggregate_seeds.py runs/hard_uniform_s* runs/hard_fgas_s* \
        --out results/checkin3/main_comparison --criterion hard_uniform hard_fgas
    python scripts/aggregate_seeds.py runs/calib_* --file best_eval.json --no-group \
        --out results/checkin3/calibration/summary

Each run directory contains an evaluation JSON (`--file`, default
best_test_eval.json = DIV2K test set). Runs are grouped by name with the
`_s<seed>` suffix removed (hard_fgas_s43 -> hard_fgas). Writes <out>.csv and
<out>.md; with --no-group, one row per run instead (calibration table).

--criterion UNIFORM FGAS checks the pre-registered comparison on the group means:
worst-case attack bit accuracy at least +2 pp, PSNR within 0.5 dB, mean attacked
bit accuracy not lower by more than 0.5 pp, FGAS clean bit accuracy >= 98%.
The result goes to <out>_criterion.json.
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

SEED_SUFFIX = re.compile(r"_s\d+$")


def load_runs(run_dirs, filename):
    runs = []
    for run_dir in run_dirs:
        path = Path(run_dir) / filename
        if not path.is_file():
            print(f"skipping {run_dir}: no {filename}")
            continue
        with open(path, encoding="utf-8") as f:
            runs.append({"name": Path(run_dir).name, **json.load(f)})
    return runs


def run_metrics(run):
    """Flat {metric: value} for one evaluation result."""
    metrics = {}
    for attack, vals in run["attacks"].items():
        metrics[f"{attack}_bit_acc"] = vals["bit_acc"]
        metrics[f"{attack}_msg_acc"] = vals["msg_acc"]
    metrics["psnr"] = run["image_quality"]["psnr"]
    metrics["ssim"] = run["image_quality"]["ssim"]
    metrics["worst_bit_acc"] = run["worst_case"]["bit_acc"]
    metrics["mean_attacked_bit_acc"] = run["mean_attacked_bit_acc"]
    return metrics


def group_runs(runs):
    groups = {}
    for run in runs:
        groups.setdefault(SEED_SUFFIX.sub("", run["name"]), []).append(run)
    return groups


def summarize(runs):
    """{metric: (mean, std, n)} over runs; std is the sample std (0 for one run)."""
    per_run = [run_metrics(run) for run in runs]
    summary = {}
    for metric in per_run[0]:
        values = [m[metric] for m in per_run if metric in m]
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[metric] = (statistics.fmean(values), std, len(values))
    worst = [run["worst_case"]["attack"] for run in runs]
    summary["worst_attack"] = max(set(worst), key=worst.count)
    return summary


def _fmt(metric, mean, std):
    if metric == "psnr":
        return f"{mean:.2f} ± {std:.2f}"
    if metric == "ssim":
        return f"{mean:.4f} ± {std:.4f}"
    return f"{100 * mean:.2f} ± {100 * std:.2f}"  # accuracies in percent


def write_grouped(groups, out):
    summaries = {name: summarize(runs) for name, runs in groups.items()}
    metrics = [m for m in next(iter(summaries.values())) if m != "worst_attack"]
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "metric", "mean", "std", "n"])
        for name, summary in summaries.items():
            for metric in metrics:
                if metric in summary:
                    writer.writerow([name, metric, *(f"{v:.6f}" for v in summary[metric][:2]), summary[metric][2]])
            writer.writerow([name, "worst_attack", summary["worst_attack"], "", len(groups[name])])
    names = list(summaries)
    lines = ["| metric | " + " | ".join(f"{n} (n={len(groups[n])})" for n in names) + " |",
             "|---|" + "---|" * len(names)]
    for metric in metrics:
        cells = [_fmt(metric, *summaries[n][metric][:2]) if metric in summaries[n] else "" for n in names]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.append("| worst_attack | " + " | ".join(summaries[n]["worst_attack"] for n in names) + " |")
    lines.append("")
    lines.append("Accuracies in %, PSNR in dB; mean ± sample std over seeds.")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries


def write_per_run(runs, out):
    attacks = list(runs[0]["attacks"])
    header = ["run", "epoch", "psnr", "ssim", *[f"{a}_bit_acc" for a in attacks],
              "worst_attack", "worst_bit_acc", "mean_attacked_bit_acc", "spread_pp"]
    rows = []
    for run in runs:
        attacked = [v["bit_acc"] for a, v in run["attacks"].items() if a != "clean"]
        rows.append([run["name"], run.get("epoch"), f"{run['image_quality']['psnr']:.2f}",
                     f"{run['image_quality']['ssim']:.4f}",
                     *[f"{run['attacks'][a]['bit_acc']:.4f}" if a in run["attacks"] else "" for a in attacks],
                     run["worst_case"]["attack"], f"{run['worst_case']['bit_acc']:.4f}",
                     f"{run['mean_attacked_bit_acc']:.4f}", f"{100 * (max(attacked) - min(attacked)):.1f}"])
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_criterion(summaries, uniform, fgas):
    u, f = summaries[uniform], summaries[fgas]
    delta_worst_pp = 100 * (f["worst_bit_acc"][0] - u["worst_bit_acc"][0])
    delta_psnr = f["psnr"][0] - u["psnr"][0]
    delta_mean_pp = 100 * (f["mean_attacked_bit_acc"][0] - u["mean_attacked_bit_acc"][0])
    clean = f["clean_bit_acc"][0] if "clean_bit_acc" in f else None
    checks = {
        "worst_case_gain_ge_2pp": delta_worst_pp >= 2.0,
        "psnr_within_0.5db": abs(delta_psnr) <= 0.5,
        "mean_attacked_drop_le_0.5pp": delta_mean_pp >= -0.5,
        "clean_ge_98": clean is None or clean >= 0.98,
    }
    return {"uniform": uniform, "fgas": fgas, "delta_worst_pp": delta_worst_pp, "delta_psnr_db": delta_psnr,
            "delta_mean_attacked_pp": delta_mean_pp, "fgas_clean_bit_acc": clean,
            "checks": checks, "met": all(checks.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--file", default="best_test_eval.json", help="evaluation JSON inside each run directory")
    parser.add_argument("--out", required=True, help="output prefix (writes .csv and .md)")
    parser.add_argument("--no-group", action="store_true", help="one row per run (e.g. calibration)")
    parser.add_argument("--criterion", nargs=2, metavar=("UNIFORM", "FGAS"))
    args = parser.parse_args()

    runs = load_runs(sorted(args.run_dirs), args.file)
    if not runs:
        raise SystemExit(f"No {args.file} found in the given run directories")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.no_group:
        write_per_run(runs, out)
    else:
        summaries = write_grouped(group_runs(runs), out)
        if args.criterion:
            missing = [g for g in args.criterion if g not in summaries]
            if missing:
                raise SystemExit(f"Groups {missing} not found; have {sorted(summaries)}")
            result = check_criterion(summaries, *args.criterion)
            with open(f"{out}_criterion.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(json.dumps(result, indent=2))
    print(out.with_suffix(".md").read_text(encoding="utf-8"))
    print(f"Wrote {out.with_suffix('.csv')} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
