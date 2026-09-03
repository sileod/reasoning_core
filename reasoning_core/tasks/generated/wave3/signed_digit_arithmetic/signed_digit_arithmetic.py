import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'Add arithmetic in signed-digit and mixed-radix positional systems.',
 'hypothesis': 'S33',
 'changes': 'Ask for a value converted into balanced ternary or a stated mixed '
            'radix, or for a sum computed in that system.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3574012244,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _to_balanced_ternary(value):
    if value == 0:
        return "0"
    digits = []
    n = value
    while n != 0:
        r = n % 3
        if r == 2:
            digits.append("T")
            n = n // 3 + 1
        elif r == 1:
            digits.append("1")
            n = n // 3
        else:
            digits.append("0")
            n = n // 3
    return "".join(reversed(digits))


def _from_balanced_ternary(s):
    total = 0
    for ch in s:
        total *= 3
        if ch == "T":
            total -= 1
        elif ch == "1":
            total += 1
    return total


def _from_mixed_radix(s, radices):
    total = 0
    for i, ch in enumerate(s):
        digit = 1 if ch == "1" else -1 if ch == "T" else 0
        total += digit * radices[i]
    return total


def _to_mixed_radix(value, radices):
    digits = []
    n = value
    for r in reversed(radices):
        if n >= r:
            digits.append(1)
            n -= r
        elif n <= -r:
            digits.append(-1)
            n += r
        else:
            digits.append(0)
    if n == 0:
        return "".join("1" if d == 1 else "T" if d == -1 else "0" for d in digits)
    return None


@dataclass
class SignedDigitConfig(Config):
    n_digits: int = 4
    system: str = "balanced_ternary"

    def apply_difficulty(self, level):
        self.n_digits = sround(self.n_digits + level)


class SignedDigitArithmetic(Task):
    config_cls = SignedDigitConfig

    def generate_entry(self):
        if random.random() < 0.5:
            return self._gen_convert()
        return self._gen_sum()

    def _gen_convert(self):
        n = self.config.n_digits
        limit = 3 ** n
        value = random.randint(-limit, limit)
        if value == 0:
            value = 1
        answer = _to_balanced_ternary(value)
        metadata = edict({
            "operation": "convert",
            "system": "balanced ternary (digits T=(-1), 0, 1)",
            "place_values": "powers of 3 (..., 27, 9, 3, 1)",
            "given": value,
            "answer": answer,
        })
        metadata.payload = {
            "op": "convert",
            "given": value,
            "system": metadata.system,
            "place_values": metadata.place_values,
        }
        return Entry(metadata=metadata, answer=answer)

    def _gen_sum(self):
        n = self.config.n_digits
        a_str = _random_digit_string(n)
        b_str = _random_digit_string(n)
        a = _from_balanced_ternary(a_str)
        b = _from_balanced_ternary(b_str)
        total = a + b
        answer = _to_balanced_ternary(total)
        metadata = edict({
            "operation": "sum",
            "system": "balanced ternary (digits T=(-1), 0, 1)",
            "place_values": "powers of 3 (..., 27, 9, 3, 1)",
            "a": a_str,
            "b": b_str,
            "answer": answer,
        })
        metadata.payload = {
            "op": "sum",
            "a": a_str,
            "b": b_str,
            "system": metadata.system,
            "place_values": metadata.place_values,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        if metadata.operation == "convert":
            body = (f"Convert the integer {metadata.given} into balanced ternary. Write the "
                    "answer as a single balanced ternary numeral.")
        else:
            body = ("Compute the sum of the two balanced ternary numerals "
                    f"`{metadata.a}` and `{metadata.b}` using the standard carry rules, "
                    "and write the result in balanced ternary. The answer is a single "
                    "balanced ternary numeral.")
        system = ("Balanced ternary uses the digits T (= -1), 0 and 1, with place values "
                  "the powers of 3 (..., 27, 9, 3, 1) so that a digit string d_k...d_1 d_0 "
                  "denotes the value sum_i d_i * 3^i.")
        return f"{system}\n\n{body}\n\nThe answer is a balanced ternary numeral (digits T, 0, 1)."

    def score_answer(self, answer, entry):
        gold = entry.answer
        if answer is None:
            return 0.0
        s = str(answer).strip()
        if s == gold:
            return 1.0
        if s.startswith("[") and s.endswith("]"):
            s = s.strip("[]").strip()
            s = s.replace("'", "").replace('"', "").replace(",", "")
        if s == gold:
            return 1.0
        return 0.0


def _random_digit_string(n):
    while True:
        s = "".join(random.choice("T01") for _ in range(n))
        if s[0] in "1T":
            return s
