"""Attack severity sweep on the held-out DIV2K test set (Figure 5).

    python scripts/sweep.py --checkpoint runs/hard_uniform_s42/best.pt --out runs/hard_uniform_s42/sweep.csv
    python scripts/sweep.py --plot runs/hard_*_s*/sweep.csv --out results/checkin3/figures/severity_sweep.png

Sweep mode evaluates one checkpoint under every level of `attacks.sweep` (a list
of parameter dicts per attack, mild to beyond the training strength). Each
level is merged over the attack's `attacks.params`, so unchanged settings such
as the occlusion fill value are kept. All levels are decoded from the same
watermarked test crops; crops and messages come from `data.test.seed`, so every
run is scored on identical inputs. The grid comes from the checkpoint's config,
or from --config if given. Writes a CSV with columns
method, seed, attack, level_idx, level, bit_acc, msg_acc
(method = experiment name without the `_s<seed>` suffix).

Plot mode reads one or more such CSVs and draws one panel per attack with the
bit accuracy per level, mean +- std over seeds for each method (uniform and
FGAS overlaid).
"""

import argparse
import csv
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEED_SUFFIX = re.compile(r"_s\d+$")
COLUMNS = ["method", "seed", "attack", "level_idx", "level", "bit_acc", "msg_acc"]
METHOD_COLORS = [("fgas_p2", "#dd8a2c"), ("fgas", "#c0392b"), ("uniform", "#9aa5b1"), ("baseline", "#9aa5b1")]
FALLBACK_COLORS = ["#2b6cb0", "#2f855a", "#6b46c1", "#b7791f"]


def level_label(levels, index):
    """Short tick label for level `index`: the value of the last parameter that changes across
    levels (blur -> sigma), or "lo-hi" for a min_*/max_* pair (brightness)."""
    level = levels[index]
    varying = [key for key in level if len({repr(other.get(key)) for other in levels}) > 1] or list(level)
    if len(varying) == 2 and varying[0].startswith("min_") and varying[1].startswith("max_"):
        return f"{level[varying[0]]}-{level[varying[1]]}"
    return str(level[varying[-1]])


def sweep_attack_params(sweep, params):
    """{"<attack>_l<i>": params} for every sweep level, merged over the attack's base params."""
    out = {}
    for attack, levels in sweep.items():
        base = params.get(attack, {})
        for index, level in enumerate(levels):
            out[f"{attack}_l{index}"] = {"type": base.get("type", attack), **base, **level}
    return out


def run_sweep(args):
    import torch

    from wm.attacks import build_attacks
    from wm.config import apply_override, load_config, resolve_device, set_seed
    from wm.data import build_test_loader
    from wm.evaluate import evaluate_attackwise
    from wm.models import build_model

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    for item in args.set:
        apply_override(cfg, item)
    sweep = (load_config(args.config) if args.config else cfg)["attacks"].get("sweep")
    if not sweep:
        raise SystemExit("No attacks.sweep grid in the checkpoint config; pass --config configs/hard_base.yaml")
    if not cfg["data"].get("test"):
        raise SystemExit("No held-out test set (data.test) in the checkpoint config; pass it with --set")
    test_seed = cfg["data"]["test"].get("seed", 0)
    set_seed(test_seed)
    device = resolve_device(cfg["experiment"].get("device", "auto"))

    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    attack_params = {"clean": {}, **sweep_attack_params(sweep, cfg["attacks"]["params"])}
    attacks = build_attacks(list(attack_params), attack_params).to(device)
    result = evaluate_attackwise(model, build_test_loader(cfg), attacks, device, seed=test_seed)

    method, seed = SEED_SUFFIX.sub("", cfg["experiment"]["name"]), cfg["experiment"]["seed"]
    clean = result["attacks"]["clean"]
    rows = [[method, seed, "clean", 0, "", clean["bit_acc"], clean["msg_acc"]]]
    for attack, levels in sweep.items():
        for index in range(len(levels)):
            vals = result["attacks"][f"{attack}_l{index}"]
            rows.append([method, seed, attack, index, level_label(levels, index), vals["bit_acc"], vals["msg_acc"]])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(row[:5] + [f"{row[5]:.6f}", f"{row[6]:.6f}"])
    print(f"Swept {len(rows) - 1} attack levels on {result['num_samples']} test crops from {args.checkpoint}")
    for row in rows:
        print(f"  {row[2]:<11} {row[4]:<28} bit_acc {row[5]:.4f}  msg_acc {row[6]:.4f}")
    print(f"Wrote {out}")


def load_sweeps(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def summarize(rows):
    """{attack: {method: [(level_idx, label, mean, std, n)]}} of bit accuracy over seeds (clean excluded)."""
    values, labels = {}, {}
    for row in rows:
        if row["attack"] == "clean":
            continue
        key = (row["attack"], row["method"], int(row["level_idx"]))
        values.setdefault(key, []).append(float(row["bit_acc"]))
        labels[(row["attack"], int(row["level_idx"]))] = row["level"]
    order = {attack: i for i, attack in enumerate(dict.fromkeys(key[0] for key in values))}  # CSV (config) order
    summary = {}
    for (attack, method, index), accs in sorted(values.items(), key=lambda kv: (order[kv[0][0]], *kv[0][1:])):
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        summary.setdefault(attack, {}).setdefault(method, []).append(
            (index, labels[(attack, index)], statistics.fmean(accs), std, len(accs)))
    return summary


def method_color(method, index):
    for key, color in METHOD_COLORS:
        if key in method:
            return color
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = summarize(load_sweeps(args.plot))
    if not summary:
        raise SystemExit("No sweep rows to plot")
    methods = sorted({method for per_method in summary.values() for method in per_method})
    ncols = min(4, len(summary))
    nrows = math.ceil(len(summary) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows), squeeze=False)
    for ax, (attack, per_method) in zip(axes.flat, summary.items()):
        for m_index, method in enumerate(methods):
            if method not in per_method:
                continue
            points = per_method[method]
            x = [p[0] for p in points]
            mean = [100 * p[2] for p in points]
            std = [100 * p[3] for p in points]
            color = method_color(method, m_index)
            n = max(p[4] for p in points)
            ax.plot(x, mean, marker="o", ms=4, color=color, label=f"{method} (n={n})")
            ax.fill_between(x, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
                            color=color, alpha=0.2, linewidth=0)
        ticks = sorted({(p[0], p[1]) for points in per_method.values() for p in points})
        ax.set_xticks([t[0] for t in ticks])
        ax.set_xticklabels([t[1] for t in ticks], fontsize=7, rotation=20 if max(len(t[1]) for t in ticks) > 5 else 0)
        ax.axhline(50, color="#999999", lw=0.8, ls=":")
        ax.set_title(attack, fontsize=10)
        ax.set_ylim(40, 101)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(summary):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("bit accuracy (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Severity sweep on the DIV2K test set (mean ± std over seeds)", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", help="best.pt to sweep")
    mode.add_argument("--plot", nargs="+", metavar="CSV", help="sweep CSVs to plot")
    parser.add_argument("--out", required=True, help="sweep CSV (sweep mode) or PNG (plot mode)")
    parser.add_argument("--config", default=None, help="take attacks.sweep from this config instead")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    if args.plot:
        plot(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
