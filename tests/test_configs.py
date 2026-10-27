import copy
from pathlib import Path

import pytest

from wm.attacks import build_attacks
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


@pytest.mark.parametrize("name", ["hard_base", "hard_uniform", "hard_fgas", "hard_fgas_p2",
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
