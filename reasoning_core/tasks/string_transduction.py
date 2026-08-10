import random
import string
from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, edict, stochastic_rounding as sround


ALPHA = string.ascii_lowercase[:8]
WORDS = "nova river amber delta orbit pixel quiet signal vector winter".split()


@dataclass
class StringTransductionConfig(Config):
    length: int = 8
    n_ops: int = 2
    alphabet_size: int = 5
    edit_ops: int = 3
    edit_rate: float = 0.25
    exclude_spaces: float = 0.9
    noop_example_rate: float = 0.2
    cancel_example_rate: float = 0.15
    effective_example_rate: float = 0.85
    min_effective_ops: int = 1
    max_noop_rate: float = 0.34

    def apply_difficulty(self, level):
        self.length = sround(self.length + 2 * level)
        self.n_ops = sround(min(8, self.n_ops + level))
        self.alphabet_size = sround(min(8, self.alphabet_size + level // 2))
        self.edit_ops = sround(self.edit_ops + level)
        self.min_effective_ops = sround(min(5, self.min_effective_ops + 0.8 * level))


def caesar(s, k):
    return "".join(chr(97 + (ord(c) - 97 + k) % 26) if c.isalpha() else c for c in s)


def rotate(s, k):
    if not s:
        return s
    k %= len(s)
    return s[k:] + s[:k]


def dedupe(s):
    out = []
    for c in s:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def apply_edits(s, edits):
    xs = list(s)
    for op, i, x in edits:
        i = max(0, min(i, len(xs)))
        if op == "insert":
            xs.insert(i, x)
        elif op == "delete" and i < len(xs):
            del xs[i]
        elif op == "replace" and i < len(xs):
            xs[i] = x
    return "".join(xs)


def apply_steps(s, steps):
    """Apply unary string operations and return the output and local change flags."""
    changed = []
    for step in steps:
        updated = step(s)
        changed.append(updated != s)
        s = updated
    return s, changed


def analyze_steps(source, steps):
    """Measure local changes and each step's causal effect on the final output."""
    target, changed = apply_steps(source, steps)
    effective = [apply_steps(source, steps[:i] + steps[i + 1:])[0] != target
                 for i in range(len(steps))]
    return target, changed, effective


class StringTransduction(Task):
    summary = "Apply string transduction operations including Caesar cipher and rotation."
    def __init__(self, config=None):
        super().__init__(config=config or StringTransductionConfig())

    def _changing_op(self, s, alphabet):
        """Sample an operation conditioned to matter at its current program slot."""
        candidates = [
            ("reverse", lambda x: x[::-1]),
            ("sort ascending", lambda x: "".join(sorted(x))),
            ("sort descending", lambda x: "".join(sorted(x, reverse=True))),
            ("dedupe adjacent repeats", dedupe),
        ]
        if len(s) > 1:
            r = random.randint(1, len(s) - 1)
            candidates.append((f"rotate left by {r}", lambda x, r=r: rotate(x, r)))
        k = random.randint(1, 5)
        candidates.append((f"caesar shift by {k}", lambda x, k=k: caesar(x, k)))
        present = sorted(set(s) & set(string.ascii_lowercase))
        if present:
            a = random.choice(present)
            b = random.choice([x for x in alphabet if x != a] or ["z"])
            candidates.append((f"replace {a} with {b}", lambda x, a=a, b=b: x.replace(a, b)))
        if len(present) > 2:
            a, b = random.sample(present, 2)
            candidates.append(
                (f"keep only {a} and {b}",
                 lambda x, a=a, b=b: "".join(c for c in x if c in {a, b}))
            )
        random.shuffle(candidates)
        return next(((name, op) for name, op in candidates if op(s) and op(s) != s), None)

    def _program(self, source, alphabet, want_noop, want_cancel):
        n_ops = max(1, int(self.config.n_ops))
        want_cancel = want_cancel and n_ops >= 3
        reserved = int(want_noop) + 2 * int(want_cancel)
        if reserved >= n_ops:
            want_noop = False
            reserved -= 1
        program, current = [], source
        for _ in range(n_ops - reserved):
            choice = self._changing_op(current, alphabet)
            if choice is None:
                break
            program.append(choice)
            current = choice[1](current)
        if want_noop:
            noop = ("caesar shift by 26", lambda s: caesar(s, 26))
            program.insert(random.randrange(len(program) + 1), noop)
        if want_cancel:
            k = random.randint(1, 5)
            program.extend((
                (f"caesar shift by {k}", lambda s, k=k: caesar(s, k)),
                (f"caesar shift by {26 - k}", lambda s, k=26 - k: caesar(s, k)),
            ))
        return program, int(want_cancel)

    def _edits(self, s, alphabet, want_noop=False):
        edits, xs = [], list(s)
        n_edits = max(1, int(self.config.edit_ops))
        noop_slot = random.randrange(n_edits) if want_noop and xs else -1
        for step in range(n_edits):
            op = "replace" if step == noop_slot else random.choice(["insert", "delete", "replace"])
            if not xs:
                op = "insert"
            i = random.randrange(len(xs) + (op == "insert"))
            x = xs[i] if step == noop_slot and op == "replace" else random.choice(alphabet)
            edits.append((op, i, x))
            xs = list(apply_edits("".join(xs), [edits[-1]]))
        return edits

    def generate_entry(self):
        if not 0 <= self.config.max_noop_rate <= 1:
            raise ValueError("max_noop_rate must be between 0 and 1")
        for rate in (self.config.noop_example_rate, self.config.cancel_example_rate,
                     self.config.effective_example_rate):
            if not 0 <= rate <= 1:
                raise ValueError("operation-control rates must be between 0 and 1")
        for _ in range(80):
            alphabet = ALPHA[: self.config.alphabet_size]
            mode = "edit" if random.random() < self.config.edit_rate else "program"
            want_noop = (self.config.noop_example_rate > 0
                         and random.random() < self.config.noop_example_rate)
            want_cancel = (mode == "program" and self.config.cancel_example_rate > 0
                           and random.random() < self.config.cancel_example_rate)
            require_effective = (self.config.effective_example_rate > 0
                                 and random.random() < self.config.effective_example_rate)
            if mode != "edit" and random.random() < 0.25:
                xs = random.sample(WORDS, random.randint(4, min(8, len(WORDS))))
                source = " ".join(xs)
            else:
                source = "".join(random.choice(alphabet) for _ in range(self.config.length))

            if mode == "edit":
                edits = self._edits(source, alphabet, want_noop)
                steps = [lambda s, edit=edit: apply_edits(s, [edit]) for edit in edits]
                target, changed, effective = analyze_steps(source, steps)
                meta = edict(mode=mode, source=source, edits=edits)
                cancelled_pairs = 0
            else:
                program, cancelled_pairs = self._program(source, alphabet, want_noop, want_cancel)
                steps = [f for _, f in program]
                target, changed, effective = analyze_steps(source, steps)
                meta = edict(mode=mode, source=source, ops=[name for name, _ in program])
            expected_steps = int(self.config.edit_ops if mode == "edit" else self.config.n_ops)
            if len(steps) != max(1, expected_steps):
                continue
            noop_rate = changed.count(False) / len(changed)
            if noop_rate > self.config.max_noop_rate:
                continue
            min_effective = min(len(steps), int(self.config.min_effective_ops))
            if require_effective and sum(effective) < min_effective:
                continue
            meta.noop_rate = noop_rate
            meta.local_change_flags = changed
            meta.effective_flags = effective
            meta.effective_op_count = sum(effective)
            meta.dead_op_count = effective.count(False)
            meta.cancelled_pair_count = cancelled_pairs
            meta.required_effective_ops = min_effective if require_effective else 0
            exclude_spaces = " " in target and random.random() < self.config.exclude_spaces
            meta.exclude_spaces = exclude_spaces
            if exclude_spaces:
                target = target.replace(" ", "")
            elif target != target.strip():
                continue
            if target:
                return Entry(meta, target)
        raise RuntimeError("failed to generate nonempty string transduction")

    def render_prompt(self, m):
        if m.mode == "edit":
            lines = []
            for op, i, x in m.edits:
                lines.append(f"- insert {x} at index {i}" if op == "insert" else f"- delete at index {i}" if op == "delete" else f"- replace index {i} with {x}")
            prompt = f"String: {m.source}\nEdits:\n" + "\n".join(lines)
        else:
            prompt = f"String: {m.source}\nOperations:\n" + "\n".join(f"- {x}" for x in m.ops)
        prompt += "\nAnswer with the final string, excluding spaces." if m.exclude_spaces else "\nAnswer with the final string."
        return prompt

    def score_answer(self, answer, entry):
        from rapidfuzz.distance import Levenshtein
        pred, gold = str(answer).strip(), str(entry.answer).strip()
        return Levenshtein.normalized_similarity(pred, gold) if pred else 0.0
