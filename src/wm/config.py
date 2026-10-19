"""Experiment configuration: YAML loading, CLI overrides, seeding, device selection."""

import copy
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_override(cfg, item):
    """Apply a single `dotted.key=value` override; the value is parsed as YAML."""
    key, sep, raw = item.partition("=")
    if not sep or not key:
        raise ValueError(f"Override must look like key.subkey=value, got {item!r}")
    *parents, leaf = key.strip().split(".")
    node = cfg
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = yaml.safe_load(raw)


def load_config(path, overrides=None):
    """Load a YAML config. A top-level `_base_: other.yaml` is merged underneath."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = _deep_merge(load_config(path.parent / base), cfg)
    for item in overrides or []:
        apply_override(cfg, item)
    return cfg


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def set_seed(seed):
    """Seed every RNG we use and disable non-deterministic cuDNN autotuning."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
