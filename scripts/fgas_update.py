"""Experiment C: compute an FGAS probability update from real validation results.

    python scripts/fgas_update.py --eval-json results/baseline_eval.json --out results/fgas_update.json

Reads the attack-wise bit accuracies written by scripts/evaluate.py, takes the
sampling distribution the checkpoint was trained with as the old distribution,
and applies the FGAS rule with the settings from --config (fgas.min_prob,
fgas.power, fgas.momentum). Writes <out>.json and the Table 2 data <out>.csv.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm.config import load_config  # noqa: E402
from wm.fgas import build_fgas, format_table  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-json", default="results/baseline_eval.json")
    parser.add_argument("--config", default="configs/fgas.yaml")
    parser.add_argument("--out", default="results/fgas_update.json")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    with open(args.eval_json, encoding="utf-8") as f:
        evaluation = json.load(f)
    accuracies = {name: evaluation["attacks"][name]["bit_acc"] for name in cfg["attacks"]["train"]}

    fgas = build_fgas(cfg, initial_probs=evaluation.get("sampler_probs"))
    fgas.update(accuracies, step=evaluation.get("epoch"))
    rows = fgas.history[-1]["rows"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "source_eval": Path(args.eval_json).as_posix(),
            "source_checkpoint": evaluation.get("checkpoint"),
            "source_epoch": evaluation.get("epoch"),
            "fgas_settings": {k: cfg["fgas"][k] for k in ("min_prob", "power", "momentum")},
            "rule": "p = min_prob + (1 - K*min_prob) * failure^power / sum(failure^power); failure = 1 - bit_acc",
            "rows": rows,
            "new_probs": fgas.probs,
        }, f, indent=2)
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    print(format_table(rows))
    print(f"Wrote {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
