import random

from dataclasses import dataclass
from fractions import Fraction

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'kraft_feasibility (draw 1 of 2)',
 'hypothesis': 'W1-069',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/kraft_feasibility',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 9834260,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _parse_fraction(s):
    if isinstance(s, str):
        s = s.strip().strip('`').strip()
    return Fraction(s)


@dataclass
class KraftFeasibilityConfig(Config):
    n_len: int = 3
    max_len: int = 6

    def apply_difficulty(self, level):
        self.n_len = sround(self.n_len + level)
        self.max_len = sround(self.max_len + 1 + level)


class KraftFeasibility(Task):
    summary = ("Given prefix-code word lengths, decide Kraft's inequality "
               "feasibility by computing the exact Kraft sum (sum of 2^-len) "
               "as a reduced fraction; instances vary in length count, "
               "codeword length range, and bound-crossing vs within-bound sums.")

    config_cls = KraftFeasibilityConfig

    def generate_entry(self):
        cfg = self.config
        n = int(cfg.n_len)
        m = int(cfg.max_len)
        lengths = [random.randint(1, m) for _ in range(n)]
        total = sum(Fraction(1, 2 ** ell) for ell in lengths)
        if not (Fraction(0) < total <= n):
            raise RuntimeError("Kraft sum out of domain")
        answer = str(total)
        metadata = edict({
            "lengths": lengths,
            "n": n,
            "max_len": m,
            "kraft": answer,
            "feasible": total <= 1,
        })
        metadata.payload = {"lengths": lengths}
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (render_payload(metadata.payload)
                + "\n\nThese are the word lengths of a binary prefix code. By "
                  "Kraft's inequality a code with these lengths exists exactly "
                  "when the Kraft sum sum(2^-len) is at most 1, and it never "
                  "exists when the sum exceeds 1. Compute this Kraft sum as an "
                  "exact reduced fraction and report it as the answer (for "
                  "example \"13/16\", or \"1\" if it is an integer).")

    def score_answer(self, answer, entry):
        try:
            return 1.0 if _parse_fraction(answer) == _parse_fraction(entry.answer) else 0.0
        except Exception:
            return 0.0

    def distractor_candidates(self, entry):
        lengths = list(entry.metadata.lengths)
        n = len(lengths)
        for i in range(n):
            original = lengths[i]
            for delta in (-1, 1):
                new = original + delta
                if new < 1:
                    continue
                l2 = list(lengths)
                l2[i] = new
                yield str(sum(Fraction(1, 2 ** ell) for ell in l2))
