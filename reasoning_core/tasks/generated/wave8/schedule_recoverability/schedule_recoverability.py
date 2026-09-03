import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, \
    stochastic_rounding as sround


@dataclass
class ScheduleRecoverabilityV2Config(Config):
    n_trans: int = 2
    n_items: int = 3
    ops_per_xact: int = 2

    def apply_difficulty(self, level):
        self.n_trans = sround(self.n_trans + level)
        self.n_items = sround(self.n_items + level)
        self.ops_per_xact = sround(self.ops_per_xact + level)


def _commit_index(ops, t):
    for i, (kind, wt, _it) in enumerate(ops, start=1):
        if kind == "C" and wt == t:
            return i
    return None


def _classify(ops):
    """Return (class_name, witness_index). Witness is a 1-based op index, 0 for strict."""
    dirty_reads = []
    dirty_writes = []
    unrecoverable = []
    for i, (kind, t, item) in enumerate(ops, start=1):
        if item is None:
            continue
        writer = None
        for j, (k2, wt, it2) in enumerate(ops):
            if j + 1 >= i:
                break
            if it2 == item and k2 == "W":
                writer = (wt, j + 1)
        if writer is None or writer[0] == t:
            continue
        wc = _commit_index(ops, writer[0])
        if wc is not None and wc <= i - 1:
            continue
        if kind == "R":
            dirty_reads.append(i)
            rc = _commit_index(ops, t)
            if rc is None or wc is None or rc < wc:
                unrecoverable.append(i)
        elif kind == "W":
            dirty_writes.append(i)
    if not dirty_reads and not dirty_writes:
        return "strict", 0
    if not dirty_reads:
        return "cascadeless", dirty_writes[0]
    if not unrecoverable:
        return "recoverable", dirty_reads[0]
    return "unrecoverable", unrecoverable[0]


def _serial_block(t, n_items, ops_per_xact):
    """A fully-run committed transaction block (safe: no dirty reads/writes)."""
    block = []
    nops = max(0, ops_per_xact + random.randrange(-1, 2))
    for _ in range(nops):
        kind = random.choice(("R", "W"))
        block.append((kind, t, random.randrange(n_items)))
    block.append(("C", t, None))
    return block


def _make(target, n_trans, n_items, ops_per_xact):
    ops = []
    X = random.randrange(n_items)
    # A short committed prelude shifts witness indices for variety.
    prelude = random.randrange(0, 3)
    for t in range(prelude):
        ops.extend(_serial_block(t, n_items, ops_per_xact))
    start = prelude
    if target == "strict":
        for t in range(n_trans):
            ops.extend(_serial_block(start + t, n_items, ops_per_xact))
    else:
        A, B = start, start + 1
        if target == "unrecoverable":
            ops.append(("W", A, X))
            ops.append(("R", B, X))
            ops.append(("C", B, None))
            ops.append(("C", A, None))
        elif target == "recoverable":
            ops.append(("W", A, X))
            ops.append(("R", B, X))
            ops.append(("C", A, None))
            ops.append(("C", B, None))
        elif target == "cascadeless":
            ops.append(("W", A, X))
            ops.append(("W", B, X))
            ops.append(("C", A, None))
            ops.append(("C", B, None))
        for t in range(start + 2, n_trans):
            ops.extend(_serial_block(t, n_items, ops_per_xact))
    return ops


def _build(n_trans, n_items, ops_per_xact):
    weights = {"unrecoverable": 1.0, "recoverable": 1.0, "cascadeless": 1.0, "strict": 0.6}
    labels = list(weights.keys())
    probs = [weights[k] for k in labels]
    for _ in range(300):
        target = random.choices(labels, weights=probs, k=1)[0]
        ops = _make(target, n_trans, n_items, ops_per_xact)
        cls, wit = _classify(ops)
        if cls == target:
            return ops, cls, wit
    raise RuntimeError("schedule_recoverability: could not construct a valid schedule")


class ScheduleRecoverability(Task):
    summary = ("Classify a database schedule as strict, cascadeless, "
               "recoverable-only, or unrecoverable, reporting the index of the "
               "first dirty read, dirty write, or unrecoverable read that "
               "determines the class.")
    config_cls = ScheduleRecoverabilityV2Config
    task_version = 2

    def generate_entry(self):
        ops, cls, wit = _build(self.config.n_trans, self.config.n_items,
                               self.config.ops_per_xact)
        n_items = self.config.n_items
        lines = []
        for idx, (kind, t, item) in enumerate(ops, start=1):
            name = {0: "T0", 1: "T1", 2: "T2", 3: "T3", 4: "T4", 5: "T5",
                    6: "T6", 7: "T7"}.get(t, f"T{t}")
            if kind == "C":
                lines.append(f"{idx}. {name} commits")
            else:
                it = f"X{item}"
                verb = "reads" if kind == "R" else "writes"
                lines.append(f"{idx}. {name} {verb} {it}")
        payload = {"schedule": "\n".join(lines)}
        answer = f"{cls} {wit}"
        metadata = edict({
            "payload": payload,
            "n_items": int(n_items),
            "transactions": int(self.config.n_trans),
        })
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        rules = (
            "A schedule interleaves the read, write and commit operations of database "
            "transactions T0,T1,... on data items X0,X1,.... Each line is one operation "
            "\"Tn reads Xk\", \"Tn writes Xk\", or \"Tn commits\". A write stays "
            "uncommitted until the writing transaction commits. Reading an item written "
            "by an uncommitted transaction is a dirty read; a transaction overwriting an "
            "item that an uncommitted transaction wrote is a dirty write.\n\n"
            "Classify by the strictest property the schedule violates:\n"
            "- unrecoverable: some transaction commits after reading data whose writing "
            "transaction has not yet committed.\n"
            "- recoverable: dirty reads occur, but no transaction commits before every "
            "writer of a value it read has committed.\n"
            "- cascadeless: no dirty reads occur, but some dirty write happens.\n"
            "- strict: no dirty writes and no dirty reads occur.\n"
            "The four classes are exclusive and exhaustive; report the one that applies."
        )
        fmt = (
            "Answer with the class name, a space, then the 1-based index of the first "
            "operation that justifies it: the first dirty read whose reader commits "
            "before its writer for unrecoverable, the first dirty read for recoverable, "
            "the first dirty write for cascadeless, and 0 for strict."
        )
        return f"{rules}\n\n{render_payload(metadata.payload)}\n\n{fmt}\n\nExample: for a schedule whose first bad operation is operation 3, answer \"recoverable 3\"."

    def score_answer(self, answer, entry):
        if not isinstance(answer, str) and not isinstance(answer, (int, float)):
            return 0.0
        a = " ".join(str(answer).split()).lower()
        g = " ".join(str(entry.answer).split()).lower()
        return 1.0 if a == g else 0.0

    def distractor_candidates(self, entry):
        gold = entry.answer.split()[0]
        for label in ("strict", "cascadeless", "recoverable", "unrecoverable"):
            if label != gold:
                yield f"{label} 0"
                yield f"{label} {len(entry.metadata.payload['schedule'])}"
        correct = entry.answer.split()[1]
        for off in (-1, 1, 2):
            v = str(int(correct) + off)
            if v != correct:
                yield f"{gold} {v}"


TASK_META = {'parent_source_id': None,
 'idea': 'schedule_recoverability (draw 2 of 2)',
 'hypothesis': 'W1-032',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/schedule_recoverability',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1669468372,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
