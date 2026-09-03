import random
from dataclasses import dataclass
from fractions import Fraction

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround
from reasoning_core.utils import score_scalar

TASK_META = {'parent_source_id': None,
 'idea': 'Continued-fraction expansion of a rational number.',
 'hypothesis': 'S61',
 'changes': 'New task; the answer is a list of integers that reconstructs the '
            'input.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1513341689,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class ContinuedFractionConfig(Config):
    max_size: int = 100
    min_negative: int = 0

    def apply_difficulty(self, level):
        self.max_size = sround(self.max_size * (level + 1))
        self.min_negative = sround(self.min_negative + level)


def _list_to_fraction(terms):
    frac = Fraction(terms[-1])
    for a in reversed(terms[:-1]):
        frac = a + 1 / frac
    return frac


class ContinuedFraction(Task):
    config_cls = ContinuedFractionConfig

    def generate_entry(self):
        max_size = self.config.max_size
        while True:
            num = random.randint(1, max_size)
            den = random.randint(1, max_size)
            sign = random.choice([1, 1, 1, -1])
            frac = Fraction(sign * num, den)
            if abs(frac) < 1:
                continue
            break

        terms = []
        x = frac
        while x != 0:
            whole = int(x.numerator // x.denominator)
            terms.append(whole)
            x = x - whole
            if x == 0:
                break
            x = 1 / x

        if len(terms) >= 2 and terms[-1] == 1:
            terms[-2] += 1
            terms.pop()

        assert len(terms) >= 1
        if len(terms) >= 2:
            assert terms[-1] != 1, terms

        rebuilt = _list_to_fraction(terms)
        assert rebuilt == frac, (terms, rebuilt, frac)

        answer = ", ".join(str(t) for t in terms)

        metadata = edict({
            "frac": f"{frac.numerator}/{frac.denominator}",
            "terms": terms,
        })
        metadata.payload = {
            "fraction": f"{frac.numerator}/{frac.denominator}",
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (
            f"Give the continued fraction expansion of the rational number "
            f"{metadata.payload['fraction']}. "
            f"Write it as a comma-separated list of integers, whole part first, "
            f"e.g. 3, 7, 15, 1. The last term is never 1 (it is folded into the "
            f"one before it). The answer is that list."
        )

    def score_answer(self, answer, entry):
        answer = answer.strip()
        if not answer:
            return 0.0
        try:
            parts = [int(p.strip()) for p in answer.split(",")]
        except (ValueError, TypeError):
            return 0.0
        if not parts:
            return 0.0
        if len(parts) >= 2 and parts[-1] == 1:
            return 0.0
        try:
            rebuilt = _list_to_fraction(parts)
        except (ZeroDivisionError, ValueError):
            return 0.0
        target = Fraction(entry.metadata["frac"])
        if rebuilt == target:
            return 1.0
        return 0.0
