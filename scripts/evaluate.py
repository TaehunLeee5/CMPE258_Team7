"""Attack-wise evaluation of a trained checkpoint (independent of training).

    python scripts/evaluate.py --checkpoint runs/baseline_uniform/best.pt --out results/baseline_eval.json

The model/data/attack settings stored in the checkpoint are used, so results
match training-time validation. Use --set to change e.g. data paths on another
machine (`--set data.prepared_root=/content/data/div2k_prepared`) or attack
strengths for a stress test. Attacks in `attacks.extra_eval` (e.g. real JPEG,
combo) are evaluated too; worst-case and mean use the training attacks only.
`--split test` evaluates the held-out test set (`data.test`, DIV2K) instead of
the validation split; its crops and messages come from `data.test.seed`.
Writes <out>.json and a table <out>.csv.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm.attacks import build_attacks  # noqa: E402
from wm.config import apply_override, resolve_device, set_seed  # noqa: E402
from wm.data import build_test_loader, build_val_loader  # noqa: E402
from wm.evaluate import evaluate_attackwise  # noqa: E402
from wm.models import build_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="results/eval.json")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    for item in args.set:
        apply_override(cfg, item)
    seed = cfg["experiment"]["seed"]
    set_seed(seed)
    device = resolve_device(cfg["experiment"].get("device", "auto"))

    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    attack_cfg = cfg["attacks"]
    names = list(attack_cfg["eval"]) + [n for n in attack_cfg.get("extra_eval", []) if n not in attack_cfg["eval"]]
    attacks = build_attacks(names, attack_cfg["params"]).to(device)
    if args.split == "test":
        loader, eval_seed = build_test_loader(cfg), cfg["data"]["test"].get("seed", 0)
    else:
        loader, eval_seed = build_val_loader(cfg), seed
    result = evaluate_attackwise(model, loader, attacks, device, seed=eval_seed, primary=attack_cfg["train"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": Path(args.checkpoint).as_posix(),
        "epoch": ckpt.get("epoch"),
        "experiment": cfg["experiment"]["name"],
        "split": args.split,
        "sampler_probs": ckpt.get("sampler_probs"),
        "attack_params": cfg["attacks"]["params"],
        "device": str(device),
        **result,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["attack", "bit_acc", "msg_acc"])
        for name, vals in result["attacks"].items():
            writer.writerow([name, f"{vals['bit_acc']:.4f}", f"{vals['msg_acc']:.4f}"])

    q = result["image_quality"]
    print(f"Evaluated {result['num_samples']} {args.split} crops from {args.checkpoint} (epoch {ckpt.get('epoch')})")
    print(f"PSNR {q['psnr']:.2f} dB | SSIM {q['ssim']:.4f}")
    for name, vals in result["attacks"].items():
        print(f"  {name:<12} bit_acc {vals['bit_acc']:.4f}  msg_acc {vals['msg_acc']:.4f}")
    print(f"Worst case: {result['worst_case']['attack']} ({result['worst_case']['bit_acc']:.4f})")
    print(f"Wrote {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
