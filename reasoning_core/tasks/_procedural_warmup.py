"""
Procedural warm-up baseline task generator.

Paper-inspired by:

    Jiang et al., "Procedural Pretraining:
    Warming Up Language Models with Abstract Data", arXiv:2601.21725
    https://arxiv.org/pdf/2601.21725

Implements one aggregate task that samples among simple procedural subtasks:
IDENTITY, REVERSE, SORT, SET, UNION, DELETE, STACK, k-DYCK,
k-DYCK SHUFFLE, and a textual ECA Rule 110 surrogate.

This is a lightweight reasoning-core task generator, not an exact reproduction
of the paper's pretraining pipeline.
"""

import json
import random
from collections import Counter
from dataclasses import dataclass

from easydict import EasyDict as edict
from reasoning_core.template import Config, Entry, Task, stochastic_rounding as sround


@dataclass
class ProceduralWarmupConfig(Config):
    task: str = "mixed"
    seq_len: int = 32
    vocab_size: int = 100
    k: int = 4
    p_open: float = 0.49
    p_open_shuffle: float = 0.5
    max_depth: int = 10**9

    tasks = (
        "identity",
        "reverse",
        "sort",
        "set",
        "union",
        "delete",
        "stack",
        "dyck",
        "dyck_shuffle",
        "eca110",
    )

    def apply_difficulty(self, level):
        self.seq_len = sround(max(8, self.seq_len + 8 * level))
        self.vocab_size = sround(max(16, self.vocab_size + 8 * level))
        self.k = sround(min(16, max(2, self.k + 0.5 * level)))
        self.max_depth = sround(max(4, self.max_depth + 2 * level))


def _canon_task(task):
    s = str(task).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "all": "mixed",
        "basic": "mixed",
        "baseline": "mixed",
        "procedural": "mixed",
        "procedural_warmup": "mixed",
        "procedural_baseline": "mixed",
        "dyckshuffle": "dyck_shuffle",
        "dyck_sh": "dyck_shuffle",
        "shuffle_dyck": "dyck_shuffle",
        "k_dyck": "dyck",
        "k_dyck_shuffle": "dyck_shuffle",
        "eca": "eca110",
        "eca_110": "eca110",
        "rule110": "eca110",
        "rule_110": "eca110",
        "eca_rule_110": "eca110",
    }
    return aliases.get(s, s)


def _tok(i):
    return f"x{i}"


def _tok_id(x):
    return int(x[1:])


def _sample_seq(n, vocab_size):
    return [_tok(random.randint(1, vocab_size)) for _ in range(n)]


def _dedup_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _missing_or_random_token(seq, vocab_size):
    missing = [_tok(i) for i in range(1, vocab_size + 1) if _tok(i) not in seq]
    if missing:
        return random.choice(missing)
    return _tok(random.randint(1, vocab_size))


def _dyck_tokens(k):
    return [(f"o{i}", f"c{i}") for i in range(k)]


def _even_len(n):
    return max(2, n + n % 2)


def generate_dyck(k, length, p_open=0.49):
    length = _even_len(length)
    pairs = _dyck_tokens(k)
    seq, stack = [], []
    remaining = length

    while remaining:
        if not stack:
            t = random.randrange(k)
            seq.append(pairs[t][0])
            stack.append(t)
        elif remaining <= len(stack):
            t = stack.pop()
            seq.append(pairs[t][1])
        elif random.random() < p_open and len(stack) < remaining - 1:
            t = random.randrange(k)
            seq.append(pairs[t][0])
            stack.append(t)
        else:
            t = stack.pop()
            seq.append(pairs[t][1])
        remaining -= 1

    return seq


def generate_dyck_shuffle(k, length, p_open=0.5, max_depth=10**9):
    length = _even_len(length)
    pairs = _dyck_tokens(k)
    seq = []
    counts = Counter()
    remaining = length

    while remaining:
        depth = sum(counts.values())

        if depth == 0:
            t = random.randrange(k)
            seq.append(pairs[t][0])
            counts[t] += 1
        else:
            must_close = remaining <= depth or depth >= max_depth

            if not must_close and random.random() < p_open:
                t = random.randrange(k)
                seq.append(pairs[t][0])
                counts[t] += 1
            else:
                open_types = [t for t, n in counts.items() if n > 0]
                t = random.choice(open_types)
                seq.append(pairs[t][1])
                counts[t] -= 1

        remaining -= 1

    return seq


def _is_valid_dyck(seq, k):
    pairs = _dyck_tokens(k)
    opens = {o: i for i, (o, _) in enumerate(pairs)}
    closes = {c: i for i, (_, c) in enumerate(pairs)}
    stack = []

    for x in seq:
        if x in opens:
            stack.append(opens[x])
        elif x in closes:
            if not stack or stack.pop() != closes[x]:
                return False
        else:
            return False

    return not stack


def _is_valid_dyck_shuffle(seq, k):
    pairs = _dyck_tokens(k)
    opens = {o: i for i, (o, _) in enumerate(pairs)}
    closes = {c: i for i, (_, c) in enumerate(pairs)}
    counts = Counter()

    for x in seq:
        if x in opens:
            counts[opens[x]] += 1
        elif x in closes:
            t = closes[x]
            if counts[t] <= 0:
                return False
            counts[t] -= 1
        else:
            return False

    return sum(counts.values()) == 0


def _eca110_next(bits):
    rule = {
        (1, 1, 1): 0,
        (1, 1, 0): 1,
        (1, 0, 1): 1,
        (1, 0, 0): 0,
        (0, 1, 1): 1,
        (0, 1, 0): 1,
        (0, 0, 1): 1,
        (0, 0, 0): 0,
    }
    n = len(bits)
    return [rule[(bits[(i - 1) % n], bits[i], bits[(i + 1) % n])] for i in range(n)]


def _generate_stack(n, vocab_size):
    ops, stack = [], []
    available = [_tok(i) for i in range(1, vocab_size + 1)]

    for i in range(n):
        early = i < 2 * n // 3
        p_push = 0.75 if early else 0.25

        can_push = bool(available)
        can_pop = bool(stack)
        do_push = can_push and (not can_pop or random.random() < p_push)

        if do_push:
            j = random.randrange(len(available))
            x = available.pop(j)
            stack.append(x)
            ops.append(x)
        elif can_pop:
            stack.pop()
            ops.append("P")
        else:
            x = _tok(random.randint(1, vocab_size))
            stack.append(x)
            ops.append(x)

    return ops, list(reversed(stack)) or ["EMPTY"]


def _meta_get(meta, key, default=None):
    return meta.get(key, default) if isinstance(meta, dict) else getattr(meta, key, default)


def _answer_tokens(answer):
    if answer is None:
        return []

    text = str(answer).strip()
    if not text:
        return []

    toks = text.replace("\n", " ").split()
    if "|" in toks:
        i = len(toks) - 1 - toks[::-1].index("|")
        toks = toks[i + 1:]

    return toks


def _make_prompt(task, seq, k=None, n=None):
    task = _canon_task(task)

    if task == "identity":
        return f"IDENTITY {' '.join(seq)} |"

    if task == "reverse":
        return f"REVERSE {' '.join(seq)} |"

    if task == "sort":
        return f"SORT {' '.join(seq)} |"

    if task == "set":
        return f"SET {' '.join(seq)} |"

    if task == "union":
        a, b = seq
        return f"UNION {' '.join(a)} | {' '.join(b)} |"

    if task == "delete":
        xs, target = seq
        return f"DELETE {' '.join(xs)} | {target} |"

    if task == "stack":
        return f"STACK {' '.join(seq)} |"

    if task == "dyck":
        return f"DYCK-{k} {n} |"

    if task == "dyck_shuffle":
        return f"DYCK-SHUFFLE-{k} {n} |"

    if task == "eca110":
        return f"ECA110 {' '.join(seq)} |"

    raise ValueError(f"unknown task: {task}")


class ProceduralWarmup(Task):
    summary = "Sequence tasks including reverse, sort, stack operations, and ECA rule 110."
    def __init__(self, config=None):
        super().__init__(config=config or ProceduralWarmupConfig())
        self.balancing_key_ratio = 0.2

    def _choose_task(self):
        task = _canon_task(self.config.task)
        if task != "mixed":
            return task
        return random.choice(self.config.tasks)

    def generate_entry(self):
        task = _canon_task(self._choose_task())
        c = self.config
        n = c.seq_len

        if task == "identity":
            seq = _sample_seq(n, c.vocab_size)
            answer = seq

        elif task == "reverse":
            seq = _sample_seq(n, c.vocab_size)
            answer = list(reversed(seq))

        elif task == "sort":
            seq = _sample_seq(n, c.vocab_size)
            answer = sorted(seq, key=_tok_id)

        elif task == "set":
            seq = _sample_seq(n, c.vocab_size)
            answer = _dedup_order(seq)

        elif task == "union":
            n1 = max(1, n // 2)
            n2 = max(1, n - n1)
            seq1 = _sample_seq(n1, c.vocab_size)
            seq2 = _sample_seq(n2, c.vocab_size)
            seq = [seq1, seq2]
            answer = _dedup_order(seq1 + seq2)

        elif task == "delete":
            seq = _sample_seq(n, c.vocab_size)
            target = random.choice(seq) if random.random() < 0.8 else _missing_or_random_token(seq, c.vocab_size)
            answer = [x for x in seq if x != target]
            seq = [seq, target]

        elif task == "stack":
            seq, answer = _generate_stack(n, c.vocab_size)

        elif task == "dyck":
            seq = []
            answer = generate_dyck(c.k, n, c.p_open)

        elif task == "dyck_shuffle":
            seq = []
            answer = generate_dyck_shuffle(c.k, n, c.p_open_shuffle, c.max_depth)

        elif task == "eca110":
            bits = [random.getrandbits(1) for _ in range(n)]
            seq = [str(b) for b in bits]
            answer = [str(b) for b in _eca110_next(bits)]

        else:
            raise ValueError(f"unknown task: {task}")

        answer = " ".join(answer)
        length = len(answer.split()) if task in {"dyck", "dyck_shuffle"} else n
        question = _make_prompt(task, seq, c.k, length)
        meta = {
            "task_name": f"procedural_warmup.{task}",
            "source_collection": "procedural_warmup",
            "source_task": task,
            "effective_level": c.level,
            "max_level": None,
            "_question": question,
            "seq": seq,
            "k": c.k,
            "n": length,
            "vocab_size": c.vocab_size,
        }
        return Entry(json.loads(json.dumps(meta, default=str)), answer)

    def render_prompt(self, metadata):
        question = _meta_get(metadata, "_question")
        if question is not None:
            return question

        task = _canon_task(_meta_get(metadata, "source_task", _meta_get(metadata, "task")))
        return _make_prompt(
            task,
            _meta_get(metadata, "seq"),
            _meta_get(metadata, "k"),
            _meta_get(metadata, "n"),
        )

    def score_answer(self, answer, entry):
        ref = entry["answer"].strip().split()
        ans = _answer_tokens(answer)
        meta = entry["metadata"]
        task = _canon_task(_meta_get(meta, "source_task", _meta_get(meta, "task")))

        if ans == ref:
            return 1.0

        if task == "dyck":
            k = _meta_get(meta, "k")
            n = _meta_get(meta, "n")
            return float(len(ans) == n and _is_valid_dyck(ans, k))

        if task == "dyck_shuffle":
            k = _meta_get(meta, "k")
            n = _meta_get(meta, "n")
            return float(len(ans) == n and _is_valid_dyck_shuffle(ans, k))

        if len(ans) != len(ref) or not ref:
            return 0.0

        return sum(a == b for a, b in zip(ans, ref)) / len(ref)
