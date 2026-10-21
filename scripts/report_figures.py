"""Report figures built from committed Check-In 2 artifacts.

    python scripts/report_figures.py                 # all figures -> results/figures/
    python scripts/report_figures.py --only architecture fgas_probs

Figures:
  architecture     pipeline diagram (encoder -> sampled attack -> decoder, FGAS loop)
  robustness       per-attack bit / full-message accuracy (results/baseline_eval.json)
  training_curves  accuracy, PSNR and image-loss weight per epoch (results/baseline_uniform/metrics.csv)
  fgas_probs       uniform vs FGAS sampling probabilities (results/fgas_update.csv)
  attack_examples  one validation crop: cover, watermarked, and each attack
                   (needs results/baseline_uniform/best.pt and the prepared DIV2K cache)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
ATTACKS = ["noise", "blur", "brightness", "resize"]
COLORS = {"bit": "#2b6cb0", "msg": "#dd8a2c", "uniform": "#9aa5b1", "fgas": "#c0392b", "psnr": "#2f855a"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, out_dir, name):
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def architecture(out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(-1, 124)
    ax.set_ylim(-1, 45)
    ax.axis("off")

    def box(x, y, w, h, title, body="", color="#e8f0fb", edge="#2b6cb0", size=8.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    facecolor=color, edgecolor=edge, linewidth=1.4))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=10, fontweight="bold")
        if body:
            ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=size, linespacing=1.4)

    def arrow(x1, y1, x2, y2, text="", color="#333333", ls="-", rad=0.0, tx=0.0, ty=1.2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=color,
                                     linewidth=1.4, linestyle=ls, connectionstyle=f"arc3,rad={rad}"))
        if text:
            ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, text, ha="center", fontsize=8, color=color)

    box(0, 31, 16, 11, "Cover image", "DIV2K crop\n3 × 128 × 128", "#f4f4f4", "#666666")
    box(0, 16, 16, 11, "Message m", "32 random bits", "#f4f4f4", "#666666")
    box(22, 16, 27, 26, "Encoder",
        "4 × Conv-BN-ReLU on cover\n+ learned message map\n(Linear 32 → 8×16×16,\n"
        "block-upsampled to 128×128)\n→ fuse conv → 1×1 residual r\nI_w = clamp(I + r, 0, 1)", size=7.8)
    box(55, 16, 23, 26, "Distortion layer",
        "one attack per batch, a ~ p(a)\nGaussian noise σ = 0.1\nGaussian blur 9×9, σ = 2\n"
        "brightness × U[0.3, 1.7]\nresize ×0.25 and back", "#fdf1e3", "#dd8a2c", size=7.8)
    box(84, 16, 18, 26, "Decoder", "7 × Conv-BN-ReLU\n(3 stride-2 stages)\n→ 4×4 avg-pool\n→ Linear\n→ 32 logits",
        size=7.8)
    box(108, 24, 15, 18, "Loss", "20 · MSE(I_w, I)\n+ BCE(logits, m)\n\nimage weight\nwarm-up 1 → 20",
        "#eef7ee", "#2f855a", size=7.8)

    arrow(16, 36.5, 22, 33)
    arrow(16, 21.5, 22, 25)
    arrow(49, 29, 55, 29, "I_w")
    arrow(78, 29, 84, 29, "a(I_w)")
    arrow(102, 33, 108, 33)

    box(40, 0, 52, 10, "Failure-Guided Attack Sampling (FGAS)",
        "every k epochs: validation bit acc per attack → failure_a = 1 − acc_a\n"
        "p(a) = floor + (1 − K·floor) · failure_a / Σ failure", "#fbeaea", "#c0392b", size=7.8)
    arrow(93, 16, 92.5, 6, "attack-wise\nvalidation", color="#c0392b", ls="--", rad=-0.35, tx=9, ty=-1)
    arrow(66.5, 10.4, 66.5, 16, "updated p(a)", color="#c0392b", ls="--", tx=7, ty=-1.2)
    ax.text(0, 6, "Baseline:  p(a) = 1/4 for every attack\nProposed: p(a) re-estimated by FGAS",
            fontsize=8.5, va="center", color="#444444")
    _save(fig, out_dir, "architecture")


def robustness(out_dir):
    ev = json.loads((RESULTS / "baseline_eval.json").read_text())
    names = ["clean"] + ATTACKS
    bit = [100 * ev["attacks"][n]["bit_acc"] for n in names]
    msg = [100 * ev["attacks"][n]["msg_acc"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = range(len(names))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], bit, w, label="Bit accuracy", color=COLORS["bit"])
    b2 = ax.bar([i + w / 2 for i in x], msg, w, label="Full-message accuracy (all 32 bits)", color=COLORS["msg"])
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(list(x), [n.capitalize() for n in names])
    ax.set_ylim(95, 100.6)
    ax.set_ylabel("Accuracy (%)  [axis starts at 95%]")
    q = ev["image_quality"]
    ax.set_title(f"Uniform baseline, epoch {ev['epoch']}, {ev['num_samples']} validation crops "
                 f"(PSNR {q['psnr']:.2f} dB, SSIM {q['ssim']:.3f})", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8, frameon=False)
    _save(fig, out_dir, "robustness")


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def training_curves(out_dir):
    rows = _read_csv(RESULTS / "baseline_uniform" / "metrics.csv")
    ep = [int(r["epoch"]) for r in rows]
    best = json.loads((RESULTS / "baseline_eval.json").read_text())["epoch"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    a1.plot(ep, [100 * float(r["train_bit_acc"]) for r in rows], label="Train (sampled attack)", color="#718096")
    a1.plot(ep, [100 * float(r["val_mean_attacked_bit_acc"]) for r in rows], label="Val, mean over 4 attacks",
            color=COLORS["bit"])
    a1.plot(ep, [100 * float(r["val_worst_bit_acc"]) for r in rows], label="Val, worst attack", color=COLORS["fgas"])
    a1.set_ylim(45, 101)
    a1.set_xlabel("Epoch")
    a1.set_ylabel("Bit accuracy (%)")
    a1.set_title("Message recovery")
    a1.legend(fontsize=8, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.7))
    inset = a1.inset_axes([0.36, 0.1, 0.6, 0.36])
    tail = [i for i, e in enumerate(ep) if e >= 5]
    inset.plot([ep[i] for i in tail], [100 * float(rows[i]["val_mean_attacked_bit_acc"]) for i in tail],
               color=COLORS["bit"], lw=1)
    inset.plot([ep[i] for i in tail], [100 * float(rows[i]["val_worst_bit_acc"]) for i in tail],
               color=COLORS["fgas"], lw=1)
    inset.set_ylim(98.8, 100.05)
    inset.tick_params(labelsize=7)
    inset.set_title("epochs 5-60 (zoom)", fontsize=7)

    a2.plot(ep, [float(r["val_psnr"]) for r in rows], color=COLORS["psnr"], label="Val PSNR")
    a2.axvline(best, color="#888888", ls=":", lw=1)
    a2.text(best - 1, 18.5, f"best.pt\n(epoch {best})", ha="right", fontsize=8, color="#555555")
    a2.set_xlabel("Epoch")
    a2.set_ylabel("PSNR (dB)", color=COLORS["psnr"])
    a2.set_title("Imperceptibility vs. image-loss weight")
    tw = a2.twinx()
    tw.plot(ep, [float(r["image_loss_weight"]) for r in rows], color="#999999", ls="--", lw=1.2)
    tw.set_ylabel("image-loss weight (dashed)", color="#777777")
    tw.spines["top"].set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir, "training_curves")


def fgas_probs(out_dir):
    rows = _read_csv(RESULTS / "fgas_update.csv")
    names = [r["attack"] for r in rows]
    old = [100 * float(r["old_prob"]) for r in rows]
    new = [100 * float(r["new_prob"]) for r in rows]
    fail = [100 * float(r["failure"]) for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
    a1.bar(names, fail, color="#c9a227")
    for i, v in enumerate(fail):
        a1.text(i, v + 0.004, f"{v:.2f}%", ha="center", fontsize=8)
    a1.set_ylabel("Failure = 1 - bit acc (%)")
    a1.set_title("Step 1: validation failure per attack")
    x = range(len(names))
    w = 0.38
    a2.bar([i - w / 2 for i in x], old, w, label="Uniform (baseline)", color=COLORS["uniform"])
    b = a2.bar([i + w / 2 for i in x], new, w, label="FGAS update", color=COLORS["fgas"])
    for bar in b:
        a2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f"{bar.get_height():.0f}%",
                ha="center", fontsize=8)
    a2.axhline(10, color="#555555", ls=":", lw=1)
    a2.text(-0.55, 10.8, "10% floor", ha="left", fontsize=7.5, color="#555555")
    a2.set_xticks(list(x), names)
    a2.set_ylabel("Sampling probability (%)")
    a2.set_title("Step 2: new attack sampling distribution")
    a2.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, "fgas_probs")


def attack_examples(out_dir, index=100):
    import torch

    from wm.attacks import build_attacks
    from wm.config import set_seed
    from wm.data import build_val_dataset
    from wm.metrics import psnr

    from wm.models import build_model

    ckpt = torch.load(RESULTS / "baseline_uniform" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    set_seed(cfg["experiment"]["seed"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ev = json.loads((RESULTS / "baseline_eval.json").read_text())
    attacks = build_attacks(ATTACKS, cfg["attacks"]["params"])
    cover = build_val_dataset(cfg)[index].unsqueeze(0)
    message = torch.randint(0, 2, (1, cfg["message"]["length"]), generator=torch.Generator().manual_seed(7)).float()
    labels = {"noise": "Gaussian noise s=0.1", "blur": "Blur 9x9, s=2", "brightness": "Brightness x[0.3,1.7]",
              "resize": "Resize x0.25 -> x4"}
    panels = []
    with torch.no_grad():
        wm = model.encoder(cover, message)
        panels.append((cover, "Original"))
        panels.append((wm, f"Watermarked\nPSNR {psnr(wm, cover):.1f} dB"))
        for name in ATTACKS:
            attacked = attacks[name](wm)
            correct = int(((model.decoder(attacked) > 0).float() == message).sum())
            panels.append((attacked, f"{labels[name]}\n{correct}/32 bits here | "
                                     f"val {100 * ev['attacks'][name]['bit_acc']:.2f}%"))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels), 2.9))
    for axis, (img, title) in zip(axes, panels):
        axis.imshow(img[0].permute(1, 2, 0).clamp(0, 1).numpy())
        axis.set_title(title, fontsize=7.8)
        axis.axis("off")
    fig.tight_layout()
    _save(fig, out_dir, "attack_examples")


FIGURES = {"architecture": architecture, "robustness": robustness, "training_curves": training_curves,
           "fgas_probs": fgas_probs, "attack_examples": attack_examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=sorted(FIGURES))
    parser.add_argument("--out-dir", default=str(RESULTS / "figures"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        FIGURES[name](out_dir)


if __name__ == "__main__":
    main()
