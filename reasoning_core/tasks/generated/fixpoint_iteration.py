import ast
import random
from dataclasses import dataclass

from reasoning_core.template import Entry, Config, Task, edict, stochastic_rounding as sround
from ._base import GeneratedMixin, exact


@dataclass
class FixpointIterationConfig(Config):
    n_sets: int = 4
    universe_size: int = 7
    n_rules: int = 7
    max_shift: int = 2
    min_passes: int = 2
    max_attempts: int = 300

    def apply_difficulty(self, level):
        self.n_sets = sround(self.n_sets + 0.45 * level)
        self.universe_size = sround(self.universe_size + 0.8 * level)
        self.n_rules = sround(self.n_rules + 1.2 * level)
        self.max_shift = sround(self.max_shift + 0.2 * level)
        self.min_passes = sround(self.min_passes + 0.35 * level)
        self.max_attempts = sround(self.max_attempts + 40 * level)


def _shift_set(values, delta, universe_size):
    return {x + delta for x in values if 0 <= x + delta < universe_size}


def _fixpoint(initial, rules, universe_size):
    sets = [set(x) for x in initial]
    passes, changes = 0, 0
    while True:
        changed = False
        passes += 1
        for dst, src, shift, mask in rules:
            add = _shift_set(sets[src], shift, universe_size)
            if mask is not None:
                add &= set(mask)
            before = len(sets[dst])
            sets[dst] |= add
            if len(sets[dst]) != before:
                changes += len(sets[dst]) - before
                changed = True
        if not changed:
            return sets, passes - 1, changes


def _set_text(xs):
    return "{" + ",".join(map(str, sorted(xs))) + "}"


class FixpointIteration(GeneratedMixin, Task):
    summary = "Compute a least fixpoint of monotone finite-set propagation rules."
    config_cls = FixpointIterationConfig

    def generate_entry(self):
        cfg = self.config
        for _ in range(cfg.max_attempts):
            initial = []
            for _ in range(cfg.n_sets):
                initial.append({x for x in range(cfg.universe_size) if random.random() < 0.12})
            if not any(initial):
                initial[random.randrange(cfg.n_sets)].add(random.randrange(cfg.universe_size))
            rules = []
            for _ in range(cfg.n_rules):
                dst, src = random.sample(range(cfg.n_sets), 2)
                shift = random.randint(-cfg.max_shift, cfg.max_shift)
                mask = None
                if random.random() < 0.4:
                    mask = tuple(x for x in range(cfg.universe_size) if random.random() < 0.55)
                    if not mask:
                        mask = (random.randrange(cfg.universe_size),)
                rule = (dst, src, shift, mask)
                if rule not in rules:
                    rules.append(rule)
            final, passes, changes = _fixpoint(initial, rules, cfg.universe_size)
            candidates = [
                i for i in range(cfg.n_sets)
                if 2 <= len(final[i]) < cfg.universe_size
                and len(final[i] - initial[i]) >= 2
            ]
            if passes < cfg.min_passes or not candidates:
                continue
            target = max(candidates, key=lambda i: len(final[i] - initial[i]))
            metadata = edict(initial=[sorted(x) for x in initial], rules=rules,
                             universe_size=cfg.universe_size, target=target,
                             passes=passes, changes=changes)
            return Entry(metadata=metadata, answer=_set_text(final[target]))
        raise RuntimeError("Failed to generate a nontrivial fixpoint instance")

    def render_prompt(self, metadata):
        start = "; ".join(f"X{i}={_set_text(xs)}" for i, xs in enumerate(metadata.initial))
        def rule_text(rule):
            dst, src, shift, mask = rule
            text = f"X{dst} |= shift(X{src},{shift:+d})"
            return text if mask is None else text + " & " + _set_text(mask)

        rules = "; ".join(rule_text(rule) for rule in metadata.rules)
        return (
            f"Universe: 0..{metadata.universe_size - 1}. Start: {start}\n"
            f"Rules: {rules}\n"
            "Apply the rules repeatedly in listed order until no set changes. "
            "shift(S,d) = {x+d in the universe : x in S}.\n"
            f"What is X{metadata.target} at the fixed point? The answer is a sorted set like {{0,2,5}}."
        )

    def score_answer(self, answer, entry):
        try:
            a = ast.literal_eval(str(answer).strip())
            b = ast.literal_eval(str(entry.answer).strip())
            return float(set(a) == set(b))
        except Exception:
            return exact(answer, entry)

    def balancing_key(self, problem):
        return min(4, problem.metadata.passes)
