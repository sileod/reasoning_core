from dataclasses import dataclass
from fractions import Fraction
import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload


@dataclass
class LayeredTariffsConfig(Config):
    n_brackets: int = 3
    max_extra: int = 4
    rate_denom: int = 10
    amount_bound: int = 200

    def apply_difficulty(self, level):
        self.n_brackets = int(self.n_brackets) + level
        self.max_extra = int(self.max_extra) + level * 2
        self.amount_bound = int(self.amount_bound) + level * 120
        self.rate_denom = int(self.rate_denom) + level * 4


_TARIFF_NAMES = ["import tariff", "excise duty", "sales tax", "processing fee",
                 "customs surcharge", "harbour fee", "storage levy",
                 "inspection duty", "broker charge", "document tax"]


def _bracket_rate(rng, denom):
    num = rng.randint(1, denom - 1)
    while num % 5 == 0:
        num = rng.randint(1, denom - 1)
    return Fraction(num, denom)


def _compute(amount_cents, brackets, discount_threshold, discount_rate,
             rebate_cents, rate_scale):
    if amount_cents < 0:
        return Fraction(-1)
    amount = Fraction(amount_cents, 1)
    for lo, hi, rate in brackets:
        lo = Fraction(lo, 1)
        hi = Fraction(hi, 1)
        if hi <= lo:
            return Fraction(-1)
        taxed = max(Fraction(0), min(amount, hi) - lo)
        amount += taxed * rate
    if amount >= discount_threshold:
        amount -= amount * discount_rate
    amount -= rebate_cents
    if amount < 0:
        return Fraction(-1)
    amount = amount * rate_scale
    return amount


class LayeredTariffs(Task):
    config_cls = LayeredTariffsConfig

    def generate_entry(self):
        c = self.config
        rng = random
        while True:
            n = c.n_brackets
            lo_prev = 0
            brackets = []
            for i in range(n):
                lo = lo_prev
                hi = lo_prev + int(rng.randint(1, 20) * (0.5 + 0.5 * (i + 1)))
                rate = _bracket_rate(rng, c.rate_denom)
                brackets.append((lo, hi, rate))
                lo_prev = hi
            threshold = int(lo_prev * rng.uniform(0.3, 0.9))
            discount_rate = _bracket_rate(rng, c.rate_denom)
            rebate_cents = int(rng.randint(1, 60))
            scale_num = rng.randint(1, 3)
            scale_den = rng.randint(1, 3)
            while scale_num == scale_den:
                scale_num = rng.randint(1, 3)
            rate_scale = Fraction(scale_num, scale_den)

            amount_cents = int(rng.randint(5, c.amount_bound * 100))

            crossed = 0
            for lo, hi, _ in brackets:
                if amount_cents < hi and amount_cents >= lo:
                    crossed += 1
            if crossed < 1:
                continue

            final = _compute(amount_cents, brackets, threshold, discount_rate,
                             rebate_cents, rate_scale)
            if final <= 0 or final.denominator > 2000:
                continue
            names = _TARIFF_NAMES[:n]
            answer = f"{final.numerator}/{final.denominator}"
            metadata = edict({
                "scenario": self._render(amount_cents, brackets, threshold,
                                         discount_rate, rebate_cents,
                                         rate_scale, names),
                "answer": answer,
            })
            metadata.payload = {}
            return Entry(metadata=metadata, answer=answer)

    def _render(self, amount_cents, brackets, threshold, discount_rate,
                rebate_cents, rate_scale, names):
        amount_eur = Fraction(amount_cents, 100)
        lines = [
            f"A shipment valued at EUR {amount_eur} crosses through {len(names)} stages.",
            f"It starts as EUR {amount_eur}. For each stage in order, apply:",
        ]
        for idx, (lo, hi, rate) in enumerate(brackets):
            lines.append(f"  {idx + 1}. {names[idx]}: any part of the value above EUR {Fraction(lo, 100)} and up to EUR {Fraction(hi, 100)} is charged at {rate} of that part.")
        lines.append(f"After the stages, a discount of {discount_rate} applies, but only if the resulting value is at least EUR {Fraction(threshold, 100)}.")
        lines.append(f"Then subtract a fixed rebate of EUR {Fraction(rebate_cents, 100)}.")
        lines.append(f"Finally convert to dollars at a rate of 1 EUR = {rate_scale} USD.")
        prompt = "\n".join(lines)
        return f"{prompt}\n\nThe answer is the final amount in USD, as an exact reduced fraction in the form `n/d`, where n is the integer numerator and d is the integer denominator."

    def render_prompt(self, metadata):
        return metadata.scenario

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        answer = answer.strip()
        try:
            given = Fraction(answer)
        except Exception:
            return 0.0
        if given == Fraction(entry.answer):
            return 1.0
        return 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'Add layered tariff and discount arithmetic with exact rational '
         'answers.',
 'hypothesis': 'S48',
 'changes': 'Ask for the amount payable after a chain of bracketed rates, '
            'discounts and rebates.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 709414788,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
