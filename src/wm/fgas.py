"""Failure-Guided Attack Sampling (FGAS).

Given attack-wise validation bit accuracies acc_a for the K training attacks:

    failure_a = 1 - acc_a                                  (high = model is weak there)
    target_a  = failure_a^power / sum_b failure_b^power     (uniform if every failure is 0)
    floored_a = min_prob + (1 - K * min_prob) * target_a    (every attack keeps >= min_prob)
    p_a       = momentum * p_old_a + (1 - momentum) * floored_a

The result is a valid distribution in which weaker attacks get more training
batches, and no attack ever disappears from training. Every update is kept in
`history` (old probabilities, accuracies, failures, new probabilities).
"""

import copy


class FGAS:
    def __init__(self, attack_names, min_prob=0.1, power=1.0, momentum=0.0, initial_probs=None):
        self.names = list(attack_names)
        k = len(self.names)
        if not 0 <= min_prob * k <= 1:
            raise ValueError(f"min_prob={min_prob} is infeasible for {k} attacks (need min_prob * K <= 1)")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        self.min_prob, self.power, self.momentum = min_prob, power, momentum
        self.probs = dict(initial_probs) if initial_probs else {n: 1.0 / k for n in self.names}
        self.history = []

    def failure_scores(self, accuracies):
        return {n: min(max(1.0 - float(accuracies[n]), 0.0), 1.0) for n in self.names}

    def target_probs(self, failures):
        weights = {n: failures[n] ** self.power for n in self.names}
        total = sum(weights.values())
        k = len(self.names)
        raw = {n: (w / total if total > 0 else 1.0 / k) for n, w in weights.items()}
        return {n: self.min_prob + (1 - k * self.min_prob) * raw[n] for n in self.names}

    def update(self, accuracies, step=None):
        """Update and return the sampling distribution from {attack: bit accuracy}."""
        missing = set(self.names) - set(accuracies)
        if missing:
            raise ValueError(f"Missing validation accuracy for {sorted(missing)}")
        old = dict(self.probs)
        failures = self.failure_scores(accuracies)
        target = self.target_probs(failures)
        new = {n: self.momentum * old[n] + (1 - self.momentum) * target[n] for n in self.names}
        total = sum(new.values())
        self.probs = {n: p / total for n, p in new.items()}
        self.history.append({
            "step": step,
            "rows": [{"attack": n, "val_bit_acc": float(accuracies[n]), "failure": failures[n],
                      "old_prob": old[n], "new_prob": self.probs[n]} for n in self.names],
        })
        return dict(self.probs)

    def state_dict(self):
        return {"probs": dict(self.probs), "history": copy.deepcopy(self.history)}

    def load_state_dict(self, state):
        self.probs = dict(state["probs"])
        self.history = copy.deepcopy(state["history"])


def build_fgas(cfg, initial_probs=None):
    f = cfg["fgas"]
    return FGAS(cfg["attacks"]["train"], min_prob=f["min_prob"], power=f.get("power", 1.0),
                momentum=f.get("momentum", 0.0), initial_probs=initial_probs)


def format_table(rows):
    """Plain-text table of one FGAS update (attack, accuracy, failure, old/new probability)."""
    lines = [f"{'attack':<12}{'val_bit_acc':>12}{'failure':>10}{'old_prob':>10}{'new_prob':>10}"]
    for r in rows:
        lines.append(f"{r['attack']:<12}{r['val_bit_acc']:>12.4f}{r['failure']:>10.4f}"
                     f"{r['old_prob']:>10.4f}{r['new_prob']:>10.4f}")
    return "\n".join(lines)
