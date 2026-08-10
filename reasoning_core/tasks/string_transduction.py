import ast
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
    max_noop_rate: float = 0.05

    def apply_difficulty(self, level):
        self.length = sround(self.length + 2 * level)
        self.n_ops = sround(min(8, self.n_ops + level))
        self.alphabet_size = sround(min(8, self.alphabet_size + level // 2))
        self.edit_ops = sround(self.edit_ops + level)


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
    """Apply unary string operations and report their no-op rate."""
    no_ops = 0
    for step in steps:
        updated = step(s)
        no_ops += updated == s
        s = updated
    return s, no_ops / len(steps) if steps else 0.0


class StringTransduction(Task):
    summary = "Apply string transduction operations including Caesar cipher and rotation."
    def __init__(self, config=None):
        super().__init__(config=config or StringTransductionConfig())

    def _program(self, alphabet):
        r, k = random.randint(1, 3), random.randint(1, 5)
        ops = [
            ("reverse", lambda s: s[::-1]),
            ("sort ascending", lambda s: "".join(sorted(s))),
            ("sort descending", lambda s: "".join(sorted(s, reverse=True))),
            ("dedupe adjacent repeats", dedupe),
            (f"rotate left by {r}", lambda s, r=r: rotate(s, r)),
            (f"caesar shift by {k}", lambda s, k=k: caesar(s, k)),
        ]
        a, b = random.sample(alphabet, 2)
        ops += [
            (f"replace {a} with {b}", lambda s, a=a, b=b: s.replace(a, b)),
            (f"keep only {a} and {b}", lambda s, a=a, b=b: "".join(c for c in s if c in {a, b})),
        ]
        return random.sample(ops, max(1, int(self.config.n_ops)))

    def _edits(self, s, alphabet):
        edits, xs = [], list(s)
        for _ in range(max(1, int(self.config.edit_ops))):
            op = random.choice(["insert", "delete", "replace"])
            if not xs:
                op = "insert"
            i = random.randrange(len(xs) + (op == "insert"))
            x = random.choice(alphabet)
            edits.append((op, i, x))
            xs = list(apply_edits("".join(xs), [edits[-1]]))
        return edits

    def generate_entry(self):
        if not 0 <= self.config.max_noop_rate <= 1:
            raise ValueError("max_noop_rate must be between 0 and 1")
        for _ in range(20):
            alphabet = ALPHA[: self.config.alphabet_size]
            mode = "edit" if random.random() < self.config.edit_rate else "program"
            if mode != "edit" and random.random() < 0.25:
                xs = random.sample(WORDS, random.randint(4, min(8, len(WORDS))))
                source = " ".join(xs)
            else:
                source = "".join(random.choice(alphabet) for _ in range(self.config.length))

            if mode == "edit":
                edits = self._edits(source, alphabet)
                steps = [lambda s, edit=edit: apply_edits(s, [edit]) for edit in edits]
                target, noop_rate = apply_steps(source, steps)
                meta = edict(mode=mode, source=source, edits=edits)
            else:
                program = self._program(alphabet)
                target, noop_rate = apply_steps(source, [f for _, f in program])
                meta = edict(mode=mode, source=source, ops=[name for name, _ in program])
            if noop_rate > self.config.max_noop_rate:
                continue
            meta.noop_rate = noop_rate
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
