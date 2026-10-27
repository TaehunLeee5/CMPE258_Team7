"""End-to-end training: cover + message -> encoder -> sampled attack -> decoder.

Loss = w_img * MSE(watermarked, cover)
     + message_loss_weight * BCEWithLogits(decoder logits, message bits)

Image-loss warm-up: with a large image weight from the start the encoder learns
to embed nothing (PSNR -> inf, bit accuracy -> 0.5). So w_img stays at
`image_loss_weight_start` until the training bit accuracy of an epoch reaches
`image_loss_ramp_trigger_acc` (or epoch `image_loss_ramp_latest_epoch` is
reached), then ramps linearly to `image_loss_weight` over
`image_loss_warmup_epochs` epochs: the message is learned first and
imperceptibility is tightened afterwards. Only epochs at the final weight are
eligible for best.pt.

One attack is drawn per batch by `AttackSampler` (uniform for the baseline).
After every epoch the model is evaluated separately under every attack in
`attacks.eval`. With `fgas.enabled`, every `fgas.update_interval` epochs (and
optionally once before training, `fgas.update_at_start`) those validation
accuracies are turned into a new sampling distribution by FGAS, which the
sampler uses from the next batch on. Run artifacts in `<output_dir>/<name>/`:

    config.yaml    resolved config          run_info.json  environment / git SHA
    metrics.csv    one row per epoch        metrics.jsonl  full per-epoch records
    best.pt        weights of best epoch    last.pt        full state for --resume
    best_eval.json attack-wise validation results of best.pt
    fgas_history.json  every FGAS update (FGAS runs only)
"""

import csv
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .attacks import AttackSampler, build_attacks
from .config import resolve_device, save_config, set_seed
from .data import build_loaders
from .evaluate import evaluate_attackwise
from .fgas import build_fgas, format_table
from .messages import random_messages
from .metrics import bit_accuracy, psnr
from .models import build_model


def image_loss_weight(train_cfg, epoch, ramp_start):
    """Image-loss weight for `epoch`; `ramp_start` is the first ramp epoch (None = not triggered yet)."""
    final = train_cfg["image_loss_weight"]
    warmup = train_cfg.get("image_loss_warmup_epochs", 0)
    if warmup <= 0:
        return final
    start = train_cfg.get("image_loss_weight_start", final)
    if ramp_start is None:
        return start
    return start + (final - start) * min(1.0, (epoch - ramp_start + 1) / warmup)


def compute_losses(cover, watermarked, logits, message, image_weight, message_weight):
    image_loss = F.mse_loss(watermarked, cover)
    message_loss = F.binary_cross_entropy_with_logits(logits, message)
    total = image_weight * image_loss + message_weight * message_loss
    return total, image_loss, message_loss


def collect_run_info(cfg, device):
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None
    return {
        "git_sha": sha,
        "seed": cfg["experiment"]["seed"],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def flatten_eval(result, prefix="val"):
    row = {}
    for name, vals in result["attacks"].items():
        row[f"{prefix}_{name}_bit_acc"] = vals["bit_acc"]
        row[f"{prefix}_{name}_msg_acc"] = vals["msg_acc"]
    row[f"{prefix}_psnr"] = result["image_quality"]["psnr"]
    row[f"{prefix}_ssim"] = result["image_quality"]["ssim"]
    row[f"{prefix}_mean_attacked_bit_acc"] = result["mean_attacked_bit_acc"]
    row[f"{prefix}_worst_attack"] = result["worst_case"]["attack"]
    row[f"{prefix}_worst_bit_acc"] = result["worst_case"]["bit_acc"]
    return row


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        exp, train_cfg, attack_cfg = cfg["experiment"], cfg["train"], cfg["attacks"]
        self.seed = exp["seed"]
        set_seed(self.seed)
        self.device = resolve_device(exp.get("device", "auto"))
        self.run_dir = Path(exp["output_dir"]) / exp["name"]
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader, self.val_loader = build_loaders(cfg)
        self.model = build_model(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_cfg["lr"])
        self.train_attacks = build_attacks(attack_cfg["train"], attack_cfg["params"],
                                           trainable_only=True).to(self.device)
        self.eval_attacks = build_attacks(attack_cfg["eval"], attack_cfg["params"]).to(self.device)
        self.sampler = AttackSampler(attack_cfg["train"], attack_cfg.get("initial_probs"), seed=self.seed)
        self.fgas = build_fgas(cfg, self.sampler.probs) if cfg["fgas"]["enabled"] else None

        self.start_epoch = 1
        self.best_score = -math.inf
        self.ramp_start = None  # epoch at which the image-loss ramp began
        self._load_initial_state()

    # ------------------------------------------------------------------ state
    def _load_initial_state(self):
        train_cfg = self.cfg["train"]
        if train_cfg.get("resume"):
            ckpt = torch.load(train_cfg["resume"], map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.sampler.load_state_dict(ckpt["sampler"])
            self.start_epoch = ckpt["epoch"] + 1
            self.best_score = ckpt.get("best_score", -math.inf)
            self.ramp_start = ckpt.get("ramp_start")
            if self.fgas is not None and ckpt.get("fgas"):
                self.fgas.load_state_dict(ckpt["fgas"])
            print(f"Resumed from {train_cfg['resume']} at epoch {self.start_epoch}")
        elif train_cfg.get("init_checkpoint"):
            ckpt = torch.load(train_cfg["init_checkpoint"], map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            print(f"Initialized model weights from {train_cfg['init_checkpoint']}")

    def save_checkpoint(self, path, epoch, full):
        state = {"model": self.model.state_dict(), "epoch": epoch, "cfg": self.cfg,
                 "sampler_probs": dict(self.sampler.probs)}
        if full:
            state.update(optimizer=self.optimizer.state_dict(), sampler=self.sampler.state_dict(),
                         best_score=self.best_score, ramp_start=self.ramp_start, fgas=self.fgas.state_dict() if self.fgas else None)
        torch.save(state, path)

    # --------------------------------------------------------------- training
    def train_epoch(self, epoch):
        self.model.train()
        train_cfg, length = self.cfg["train"], self.cfg["message"]["length"]
        image_weight = image_loss_weight(train_cfg, epoch, self.ramp_start)
        self.sampler.reset_counts()
        totals = {"loss": 0.0, "image_loss": 0.0, "message_loss": 0.0, "bit_acc": 0.0, "psnr": 0.0}
        steps = 0
        bar = tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False)
        for cover in bar:
            cover = cover.to(self.device, non_blocking=True)
            message = random_messages(cover.shape[0], length, device=self.device)
            attack_name = self.sampler.sample()
            watermarked, _, logits = self.model(cover, message, self.train_attacks[attack_name])
            loss, image_loss, message_loss = compute_losses(cover, watermarked, logits, message,
                                                            image_weight, train_cfg["message_loss_weight"])

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            totals["loss"] += loss.item()
            totals["image_loss"] += image_loss.item()
            totals["message_loss"] += message_loss.item()
            totals["bit_acc"] += bit_accuracy(logits.detach(), message)
            totals["psnr"] += psnr(watermarked.detach(), cover)
            steps += 1
            if steps % train_cfg.get("log_interval", 50) == 0:
                bar.set_postfix(loss=totals["loss"] / steps, acc=totals["bit_acc"] / steps,
                                psnr=totals["psnr"] / steps)
        stats = {f"train_{k}": v / max(steps, 1) for k, v in totals.items()}
        stats["image_loss_weight"] = image_weight
        return stats

    def validate(self):
        return evaluate_attackwise(self.model, self.val_loader, self.eval_attacks, self.device, seed=self.seed,
                                   primary=self.sampler.names)

    def fgas_update(self, result, step):
        """Feed attack-wise validation accuracy to FGAS and hand the new distribution to the sampler."""
        accuracies = {name: result["attacks"][name]["bit_acc"] for name in self.sampler.names}
        self.sampler.set_probs(self.fgas.update(accuracies, step=step))
        with open(self.run_dir / "fgas_history.json", "w", encoding="utf-8") as f:
            json.dump(self.fgas.history, f, indent=2)
        print(f"FGAS update (after epoch {step}):", flush=True)
        print(format_table(self.fgas.history[-1]["rows"]), flush=True)

    def _log(self, row, record):
        csv_path = self.run_dir / "metrics.csv"
        new_file = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def fit(self):
        save_config(self.cfg, self.run_dir / "config.yaml")
        with open(self.run_dir / "run_info.json", "w", encoding="utf-8") as f:
            json.dump(collect_run_info(self.cfg, self.device), f, indent=2)

        epochs = self.cfg["train"]["epochs"]
        result = None
        if self.fgas is not None and self.cfg["fgas"].get("update_at_start") and self.start_epoch == 1:
            self.fgas_update(self.validate(), step=0)
        for epoch in range(self.start_epoch, epochs + 1):
            start = time.time()
            probs_used = dict(self.sampler.probs)
            train_stats = self.train_epoch(epoch)
            result = self.validate()

            row = {"epoch": epoch, **train_stats, **flatten_eval(result)}
            row.update({f"prob_{n}": p for n, p in probs_used.items()})
            row.update({f"count_{n}": c for n, c in self.sampler.counts.items()})
            row["epoch_time_s"] = time.time() - start
            self._log(row, {"epoch": epoch, "train": train_stats, "val": result,
                            "sampler_probs": probs_used, "sampler_counts": dict(self.sampler.counts)})

            # Warm-up epochs trade image quality for accuracy, so they are not eligible as "best".
            train_cfg = self.cfg["train"]
            final_weight = train_stats["image_loss_weight"] == train_cfg["image_loss_weight"]
            triggered = (train_stats["train_bit_acc"] >= train_cfg.get("image_loss_ramp_trigger_acc", 0.0)
                         or epoch + 1 >= train_cfg.get("image_loss_ramp_latest_epoch", math.inf))
            if self.ramp_start is None and train_cfg.get("image_loss_warmup_epochs", 0) > 0 and triggered:
                self.ramp_start = epoch + 1
                print(f"Train bit accuracy {train_stats['train_bit_acc']:.3f}: image-loss ramp starts at epoch "
                      f"{self.ramp_start}", flush=True)
            score = result["mean_attacked_bit_acc"]
            if (final_weight or epoch == epochs) and score > self.best_score:
                self.best_score = score
                self.save_checkpoint(self.run_dir / "best.pt", epoch, full=False)
                with open(self.run_dir / "best_eval.json", "w", encoding="utf-8") as f:
                    json.dump({"epoch": epoch, **result}, f, indent=2)
            self.save_checkpoint(self.run_dir / "last.pt", epoch, full=True)

            accs = " ".join(f"{n}={v['bit_acc']:.3f}" for n, v in result["attacks"].items())
            print(f"[epoch {epoch}/{epochs}] loss={train_stats['train_loss']:.4f} "
                  f"psnr={result['image_quality']['psnr']:.2f} ssim={result['image_quality']['ssim']:.3f} "
                  f"bit_acc: {accs} ({row['epoch_time_s']:.0f}s)", flush=True)

            interval = self.cfg["fgas"].get("update_interval", 1)
            if self.fgas is not None and epoch % interval == 0 and epoch < epochs:
                self.fgas_update(result, step=epoch)
        return result
