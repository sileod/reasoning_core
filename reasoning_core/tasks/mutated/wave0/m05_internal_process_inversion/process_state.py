import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround
from reasoning_core.utils import score_scalar

UNITS = "stamps marbles coins books apples cards tokens beads tiles cookies shells stickers pebbles buttons pencils".split()

MUL = {2: "doubled", 3: "tripled", 4: "quadrupled", 5: "quintupled"}
ORD = {2: "half", 3: "a third", 4: "a quarter", 5: "a fifth"}


def ri(a, b):
    return random.randint(int(a), int(b))


def _step_text(op, k, unit):
    if op == "add":
        return f"{k} more {unit} were added"
    if op == "sub":
        return f"{k} {unit} were removed"
    if op == "mul":
        return f"the count was {MUL[k] if random.random() < 0.6 else 'multiplied by %d' % k}"
    return "the count was cut to " + ORD[k]


@dataclass
class ProcessStateConfig(Config):
    min_steps: int = 3
    max_steps: int = 4
    min_query: int = 2
    max_n: int = 9

    def apply_difficulty(self, level):
        self.min_steps = sround(self.min_steps + level)
        self.max_steps = sround(self.max_steps + level)
        self.min_query = sround(self.min_query + level)
        self.max_n = sround(self.max_n + 3 * level)


def _build_chain(cfg):
    base = ri(3, 8)
    cur = base
    states = [cur]
    steps = []
    n_steps = ri(cfg.min_steps, cfg.max_steps)
    for _ in range(n_steps):
        op = random.choice(["add", "sub", "mul", "div"])
        if op == "add":
            k = ri(2, cfg.max_n)
            cur += k
        elif op == "sub":
            if cur <= 4:
                k = ri(2, cfg.max_n)
                cur += k
                op = "add"
            else:
                k = ri(2, cur - 1)
                cur -= k
        elif op == "mul":
            k = ri(2, 5)
            cur *= k
        else:
            ks = [d for d in (2, 3, 4, 5) if cur % d == 0 and cur // d >= 2]
            if not ks:
                k = ri(2, cfg.max_n)
                cur += k
                op = "add"
            else:
                k = random.choice(ks)
                cur //= k
        steps.append((op, k))
        states.append(cur)
    return steps, states, base


class ProcessState(Task):
    config_cls = ProcessStateConfig
    summary = "Identify the process state immediately before or after a given internal step."

    def generate_entry(self):
        while True:
            steps, states, base = _build_chain(self.config)
            n_steps = len(steps)
            if n_steps < 3 or self.config.min_query >= n_steps:
                continue
            k = ri(2, n_steps - 1)
            target = random.choice(["before", "after"])
            idx = k - 1 if target == "before" else k
            val = states[idx]
            if val == base or val == states[-1] or idx < self.config.min_query:
                continue
            unit = random.choice(UNITS)
            metadata = edict(
                unit=unit,
                base=base,
                observed=states[-1],
                steps=steps,
                target=target,
                k=k,
                answer_val=val,
                forward_distance=idx,
            )
            return Entry(metadata=metadata, answer=str(val))

    def render_prompt(self, m):
        segs = [_step_text(op, k, m.unit) for op, k in m.steps]
        segs[0] = segs[0][0].upper() + segs[0][1:]
        chain = "; then ".join(segs)
        word = "before" if m.target == "before" else "after"
        return (
            f"A jar starts with {m.base} {m.unit}. {chain}. "
            f"The jar now holds {m.observed} {m.unit}. "
            f"How many {m.unit} were in the jar immediately {word} step {m.k}? "
            f"Answer with a number."
        )

    def score_answer(self, answer, entry):
        return score_scalar(answer, entry)

    def deduplication_key(self, problem):
        m = problem.metadata
        return str((m.base, m.observed, tuple(map(tuple, m.steps)), m.target, m.k))


TASK_META = {'parent_source_id': 'c267a83e5953e4862bec61fb7c72a249dc6d8d945f1116585ac947e52ef26f35',
 'idea': 'Test bidirectional process reasoning at controlled internal depth.',
 'hypothesis': 'H1',
 'changes': 'Hide and query an internal process state instead of only the '
            'initial value.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2033032770,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 28,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
