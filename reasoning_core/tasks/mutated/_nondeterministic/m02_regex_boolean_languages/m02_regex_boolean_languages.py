import random
from dataclasses import dataclass
from collections import deque

from greenery import parse as gparse
from gramforge import generate
from reasoning_core.template import Task, Entry, Config, stochastic_rounding as sround
from easydict import EasyDict as edict

from reasoning_core.tasks.regex import regex_grammar, ALPHA


def _sample_regex(G, depth, min_depth, mode="sequential", max_tries=60):
    for _ in range(max_tries):
        x = generate(G.start(), depth=depth, min_depth=min_depth, mode=mode,
                     seed=random.randrange(2**32))
        if len(x.leaves) <= 1:
            continue
        r = x @ "re"
        try:
            f = gparse(r).to_fsm()
            if not f.empty():
                return r, f
        except Exception:
            continue
    return None, None


@dataclass
class RegexBooleanConfig(Config):
    max_depth: int = 3
    min_depth: int = 2
    n_alpha: int = 3
    gramforge_algorithm: str = "sequential"

    def apply_difficulty(self, level):
        self.max_depth += level
        self.min_depth += level
        self.n_alpha = sround(self.n_alpha + 0.5 * level)


def _shortest_witness(fsm, alphabet):
    queue = deque([(fsm.initial, "")])
    visited = {fsm.initial}
    while queue:
        state, path = queue.popleft()
        if state in fsm.finals:
            return path
        for symbol in sorted(alphabet):
            next_state = next(
                target
                for charclass, target in fsm.map[state].items()
                if charclass.accepts(symbol)
            )
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + symbol))
    return None


class RegexBooleanLanguages(Task):
    summary = "Reason about Boolean set operations over regular languages with witnesses."
    config_cls = RegexBooleanConfig

    def __init__(self, config=None):
        super().__init__(config=config or RegexBooleanConfig())
        self.balancing_key_ratio = 0.25

    def generate_entry(self):
        cfg = self.config
        alpha = ALPHA[: max(2, cfg.n_alpha)]
        words = [a + b for a in alpha for b in alpha][:6]
        G = regex_grammar(
            fsm_subset=True,
            alpha=alpha,
            words=random.sample(words, min(len(words), 4)),
        )

        qtype = random.choice(["sd", "sub"])
        for _ in range(200):
            r1, f1 = _sample_regex(G, cfg.max_depth, cfg.min_depth, cfg.gramforge_algorithm)
            if f1 is None:
                continue
            r2, f2 = _sample_regex(G, cfg.max_depth, cfg.min_depth, cfg.gramforge_algorithm)
            if f2 is None:
                continue

            if qtype == "sd":
                # A symmetric-difference B must be non-empty so the answer carries a witness.
                sd = f1.symmetric_difference(f2)
                witness = _shortest_witness(sd, alpha)
                if not witness:
                    continue
                meta = edict(qtype="sd", regex_a=r1, regex_b=r2)
                return Entry(meta, witness)

            r3, f3 = _sample_regex(G, cfg.max_depth, cfg.min_depth, cfg.gramforge_algorithm)
            if f3 is None:
                continue
            # Counterexample to (A intersect not B) subset C == a string in A \ (B U C).
            diff = f1.difference(f2).difference(f3)
            witness = _shortest_witness(diff, alpha)
            if not witness:
                continue
            meta = edict(qtype="sub", regex_a=r1, regex_b=r2, regex_c=r3)
            return Entry(meta, witness)

        raise RuntimeError("Could not generate a non-empty boolean-language witness")

    def render_prompt(self, metadata):
        meta = metadata
        if meta["qtype"] == "sd":
            return (
                f"A = {meta['regex_a']}\nB = {meta['regex_b']}\n"
                "The answer is the shortest string accepted by exactly one of A or B "
                "(the symmetric difference). Break ties lexicographically."
            )
        return (
            f"A = {meta['regex_a']}\nB = {meta['regex_b']}\nC = {meta['regex_c']}\n"
            "The answer is the shortest string in A but in neither B nor C, "
            "providing a counterexample to '(A intersect not B) is a subset of C'. "
            "Break ties lexicographically."
        )

    def score_answer(self, answer, entry):
        pred = str(answer).strip()
        meta = entry.metadata
        try:
            if meta["qtype"] == "sd":
                fa = gparse(meta["regex_a"]).to_fsm()
                fb = gparse(meta["regex_b"]).to_fsm()
                if not (fa.accepts(pred) != fb.accepts(pred)):
                    return 0.0
            else:
                fa = gparse(meta["regex_a"]).to_fsm()
                fb = gparse(meta["regex_b"]).to_fsm()
                fc = gparse(meta["regex_c"]).to_fsm()
                if not (fa.accepts(pred) and not fb.accepts(pred) and not fc.accepts(pred)):
                    return 0.0
        except Exception:
            return 0.0
        expected = entry.answer
        if pred == expected:
            return 1.0
        if len(pred) > len(expected):
            return 1.0 / (1.0 + len(pred) - len(expected))
        return 0.0

    def balancing_key(self, problem):
        n = len(problem.answer)
        bucket = "1" if n == 1 else "2" if n == 2 else "3+"
        return f"{problem.metadata.qtype}:len={bucket}"


TASK_META = {'parent_source_id': '1a726b2c0003885c08045ad7cac014a6adc6ac3a8ee4b259264d455e71b2d31e',
 'idea': 'Test shallow composition of familiar finite-state primitives.',
 'hypothesis': 'H10',
 'changes': 'Add depth-2/3 Boolean language expressions and exact '
            'relation/emptiness queries.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 117786424,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 28,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
