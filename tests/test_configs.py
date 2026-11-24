import copy
from pathlib import Path

import pytest

from wm.attacks import AttackSampler, build_attacks
from wm.config import load_config
from wm.fgas import FGAS

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _without(cfg, *keys):
    cfg = copy.deepcopy(cfg)
    for key in keys:
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node[part]
        node.pop(leaf)
    return cfg


def test_uniform_and_fgas_differ_only_in_fgas_block():
    uniform = load_config(CONFIGS / "hard_uniform.yaml")
    fgas = load_config(CONFIGS / "hard_fgas.yaml")
    assert _without(uniform, "fgas", "experiment.name") == _without(fgas, "fgas", "experiment.name")
    assert fgas["fgas"]["enabled"] and not uniform["fgas"]["enabled"]


def test_fallback_changes_only_power():
    base, p2 = load_config(CONFIGS / "hard_fgas.yaml"), load_config(CONFIGS / "hard_fgas_p2.yaml")
    assert _without(base, "fgas.power", "experiment.name") == _without(p2, "fgas.power", "experiment.name")
    assert p2["fgas"]["power"] == 2.0


ABLATIONS = {"floor002": ("min_prob", 0.02), "floor010": ("min_prob", 0.10), "int1": ("update_interval", 1),
             "int5": ("update_interval", 5), "lossweight": ("mode", "loss_weight")}


@pytest.mark.parametrize("variant", sorted(ABLATIONS))
def test_ablation_changes_exactly_one_fgas_key(variant):
    key, value = ABLATIONS[variant]
    base = load_config(CONFIGS / "hard_fgas.yaml")
    cfg = load_config(CONFIGS / f"hard_fgas_{variant}.yaml")
    assert cfg["fgas"][key] == value != base["fgas"].get(key)
    assert cfg["experiment"]["name"] == f"hard_fgas_{variant}"
    base_rest = _without(base, "experiment.name", *([f"fgas.{key}"] if key in base["fgas"] else []))
    assert _without(cfg, "experiment.name", f"fgas.{key}") == base_rest


WARM0 = {"hard_uniform_warm0": "hard_uniform", "hard_fgas_warm0": "hard_fgas"}


@pytest.mark.parametrize("name", sorted(WARM0))
def test_warm0_changes_only_the_warmup_image_weight(name):
    cfg, parent = load_config(CONFIGS / f"{name}.yaml"), load_config(CONFIGS / f"{WARM0[name]}.yaml")
    assert cfg["train"]["image_loss_weight_start"] == 0.0 != parent["train"]["image_loss_weight_start"]
    assert cfg["experiment"]["name"] == name
    keys = ("train.image_loss_weight_start", "experiment.name")
    assert _without(cfg, *keys) == _without(parent, *keys)


def test_warm0_uniform_and_fgas_differ_only_in_fgas_block():
    uniform = load_config(CONFIGS / "hard_uniform_warm0.yaml")
    fgas = load_config(CONFIGS / "hard_fgas_warm0.yaml")
    assert _without(uniform, "fgas", "experiment.name") == _without(fgas, "fgas", "experiment.name")


def test_static_schedule_changes_only_the_attack_mix():
    static = load_config(CONFIGS / "hard_static_warm0.yaml")
    uniform = load_config(CONFIGS / "hard_uniform_warm0.yaml")
    keys = ("attacks.initial_probs", "experiment.name")
    assert _without(static, *keys) == _without(uniform, *keys)
    assert static["experiment"]["name"] == "hard_static_warm0" and not static["fgas"]["enabled"]
    probs = static["attacks"]["initial_probs"]
    assert set(probs) == set(static["attacks"]["train"])
    assert all(p > 0 for p in probs.values()) and sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
    sampler = AttackSampler(static["attacks"]["train"], probs)
    assert sampler.probs["crop"] == pytest.approx(0.345) and max(sampler.probs, key=sampler.probs.get) == "crop"


def test_lossweight_warm0_changes_only_fgas_mode():
    cfg = load_config(CONFIGS / "hard_fgas_lossweight_warm0.yaml")
    base = load_config(CONFIGS / "hard_fgas_warm0.yaml")
    assert cfg["fgas"]["mode"] == "loss_weight" != base["fgas"]["mode"]
    assert _without(cfg, "fgas.mode", "experiment.name") == _without(base, "fgas.mode", "experiment.name")


@pytest.mark.parametrize("name", ["hard_base", "hard_uniform", "hard_fgas", "hard_fgas_p2",
                                  *(f"hard_fgas_{v}" for v in sorted(ABLATIONS)),
                                  *sorted(WARM0), "hard_static_warm0", "hard_fgas_lossweight_warm0",
                                  "calib_w100", "calib_w200", "calib_w400", "calib_w100_strong"])
def test_hardened_configs_build(name):
    cfg = load_config(CONFIGS / f"{name}.yaml")
    attacks = cfg["attacks"]
    build_attacks(attacks["train"], attacks["params"], trainable_only=True)
    build_attacks(attacks["eval"] + attacks["extra_eval"], attacks["params"])
    for attack, levels in attacks["sweep"].items():
        build_attacks([f"{attack}_l{i}" for i in range(len(levels))],
                      {f"{attack}_l{i}": {"type": attack, **level} for i, level in enumerate(levels)})
    assert set(attacks["eval"]) - {"clean"} == set(attacks["train"])


def test_fgas_with_seven_attacks_keeps_the_floor():
    cfg = load_config(CONFIGS / "hard_fgas.yaml")
    names, fg = cfg["attacks"]["train"], cfg["fgas"]
    fgas = FGAS(names, min_prob=fg["min_prob"], power=fg["power"], momentum=fg["momentum"])
    accuracies = {n: 0.99 for n in names} | {"crop": 0.75}
    for _ in range(10):
        probs = fgas.update(accuracies)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert min(probs.values()) >= fg["min_prob"] - 1e-9
    assert max(probs, key=probs.get) == "crop"
