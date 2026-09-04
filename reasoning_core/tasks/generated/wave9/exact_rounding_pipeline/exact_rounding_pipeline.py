import math
import random
from dataclasses import dataclass
from fractions import Fraction

from reasoning_core.template import Entry, Config, Task, edict, stochastic_rounding as sround


@dataclass
class ExactRoundingPipelineConfig(Config):
    n_stages: int = 3
    val_range: int = 25
    max_places: int = 3
    max_attempts: int = 300

    def apply_difficulty(self, level):
        self.n_stages = sround(self.n_stages + 0.8 * level)
        self.val_range = sround(self.val_range + 6 * level)
        self.max_places = sround(self.max_places + 0.4 * level)


MULTIPLIERS = [Fraction(1, 3), Fraction(2, 3), Fraction(3, 4), Fraction(3, 2),
               Fraction(5, 3), Fraction(7, 4), Fraction(7, 6), Fraction(6, 5),
               Fraction(8, 5), Fraction(9, 5), Fraction(11, 6), Fraction(5, 2)]
ADDENDS = [Fraction(1, 2), Fraction(2, 3), Fraction(1, 3), Fraction(5, 4),
           Fraction(4, 3), Fraction(3, 4), Fraction(7, 6), Fraction(6, 5)]
RULES = ["truncate", "floor", "ceiling", "half-even"]


def _round_frac(f, places, rule):
    t = f * (10 ** places)
    n = t.numerator
    d = t.denominator
    if rule == "truncate":
        if n >= 0:
            q = n // d
        else:
            q = -((-n) // d)
    elif rule == "floor":
        q = n // d
    elif rule == "ceiling":
        q = -((-n) // d)
    elif rule == "half-even":
        sign = 1 if n >= 0 else -1
        a = abs(n)
        q, r = divmod(a, d)
        if 2 * r > d:
            q += 1
        elif 2 * r == d and q % 2 == 1:
            q += 1
        q = sign * q
    return Fraction(q, 10 ** places)


def _frac_text(f):
    if f.denominator == 1:
        return str(f.numerator)
    if f.denominator in (2, 4, 5, 8, 10, 20, 25, 50, 100):
        return f"{float(f):g}"
    return f"{f.numerator}/{f.denominator}"


class ExactRoundingPipeline(Task):
    summary = ("Execute multi-stage fixed-point calculations with explicit "
               "truncation, floor, ceiling, or half-even rules, returning the exact final value.")
    config_cls = ExactRoundingPipelineConfig

    def generate_entry(self):
        cfg = self.config
        for _ in range(cfg.max_attempts):
            x0 = Fraction(random.randint(-cfg.val_range, cfg.val_range), 1) \
                + random.choice([Fraction(0, 1), Fraction(1, 2), Fraction(1, 3),
                                 Fraction(2, 3), Fraction(1, 4), Fraction(3, 4),
                                 Fraction(1, 5), Fraction(2, 5)])
            stages = []
            x = x0
            for s in range(cfg.n_stages):
                m = random.choice(MULTIPLIERS)
                b = random.choice(ADDENDS)
                places = random.randint(1, cfg.max_places)
                rule = random.choice(RULES)
                x = _round_frac(x * m + b, places, rule)
                stages.append((m, b, places, rule))
            final = _round_frac(x, 0, "half-even")
            final_int = final.numerator
            if final.denominator != 1:
                continue
            if not -1000000 <= final_int <= 1000000:
                continue
            metadata = edict(x0=_frac_text(x0), stages=[
                [_frac_text(m), _frac_text(b), places, rule] for (m, b, places, rule) in stages
            ], final=final_int)
            return Entry(metadata=metadata, answer=str(final_int))
        raise RuntimeError("Failed to generate a nontrivial rounding pipeline instance")

    def render_prompt(self, metadata):
        lines = [f"A fixed-point pipeline starts with x0 = {metadata.x0} and applies stages in order."]
        lines.append("Each stage k computes xk = round(x(k-1) * m + b, p, rule), where rounding "
                     "to p decimals keeps only p digits after the point:")
        lines.append("  'truncate' discards extra decimals towards zero;")
        lines.append("  'floor' rounds down towards minus infinity;")
        lines.append("  'ceiling' rounds up towards plus infinity;")
        lines.append("  'half-even' rounds the half to the nearest even final digit.")
        lines.append("Stage list (m, b, p, rule):")
        for i, (m, b, p, rule) in enumerate(metadata.stages, 1):
            lines.append(f"  stage {i}: m = {m}, b = {b}, p = {p}, rule = {rule}")
        lines.append("What is the exact integer xN after the final stage? "
                     "The answer is a single integer.")
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        try:
            return float(int(str(answer).strip()) == entry.metadata.final)
        except Exception:
            return 0.0

    def balancing_key(self, problem):
        return min(5, len(problem.metadata.stages)), problem.metadata.final % 7


TASK_META = {'parent_source_id': None,
 'idea': 'exact_rounding_pipeline (draw 1 of 1)',
 'hypothesis': 'HV-070',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/exact_rounding_pipeline',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3948247022,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
