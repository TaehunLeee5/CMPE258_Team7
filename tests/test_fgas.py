import math

import pytest

from wm.fgas import FGAS

NAMES = ["noise", "blur", "brightness", "resize"]
ACCS = {"noise": 0.95, "blur": 0.80, "brightness": 0.90, "resize": 0.60}


def test_starts_uniform():
    assert FGAS(NAMES).probs == {n: 0.25 for n in NAMES}


def test_update_is_a_distribution_with_floor():
    fgas = FGAS(NAMES, min_prob=0.1)
    probs = fgas.update(ACCS)
    assert math.isclose(sum(probs.values()), 1.0)
    assert all(p >= 0.1 - 1e-9 for p in probs.values())


def test_weaker_attacks_get_more_probability():
    probs = FGAS(NAMES, min_prob=0.05).update(ACCS)
    ranked_by_prob = sorted(NAMES, key=probs.get, reverse=True)
    ranked_by_weakness = sorted(NAMES, key=ACCS.get)
    assert ranked_by_prob == ranked_by_weakness
    assert probs["resize"] > 0.25 > probs["noise"]


def test_expected_values():
    # failures 0.05, 0.20, 0.10, 0.40 (sum 0.75); floor 0.1 leaves 0.6 to distribute
    probs = FGAS(NAMES, min_prob=0.1).update(ACCS)
    assert math.isclose(probs["noise"], 0.1 + 0.6 * 0.05 / 0.75)
    assert math.isclose(probs["resize"], 0.1 + 0.6 * 0.40 / 0.75)


def test_floor_keeps_perfect_attack_alive():
    probs = FGAS(NAMES, min_prob=0.1).update({"noise": 1.0, "blur": 1.0, "brightness": 1.0, "resize": 0.5})
    assert math.isclose(probs["noise"], 0.1)
    assert math.isclose(probs["resize"], 0.7)


def test_equal_accuracy_gives_uniform():
    probs = FGAS(NAMES).update({n: 0.7 for n in NAMES})
    assert all(math.isclose(p, 0.25) for p in probs.values())
    probs = FGAS(NAMES).update({n: 1.0 for n in NAMES})  # zero failure everywhere
    assert all(math.isclose(p, 0.25) for p in probs.values())


def test_momentum_moves_partially():
    fgas = FGAS(NAMES, min_prob=0.1, momentum=0.5)
    target = FGAS(NAMES, min_prob=0.1).update(ACCS)
    probs = fgas.update(ACCS)
    for n in NAMES:
        assert math.isclose(probs[n], 0.5 * 0.25 + 0.5 * target[n])


def test_history_records_table_rows():
    fgas = FGAS(NAMES)
    fgas.update(ACCS, step=3)
    entry = fgas.history[0]
    assert entry["step"] == 3
    assert [r["attack"] for r in entry["rows"]] == NAMES
    assert entry["rows"][0]["old_prob"] == 0.25


def test_rejects_infeasible_floor():
    with pytest.raises(ValueError):
        FGAS(NAMES, min_prob=0.3)
