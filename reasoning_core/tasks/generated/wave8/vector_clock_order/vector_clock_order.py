import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


def _classify(u, v):
    n = len(u)
    less = any(u[i] < v[i] for i in range(n))
    greater = any(u[i] > v[i] for i in range(n))
    if not less and not greater:
        return "equal"
    if not greater:
        return "before"
    if not less:
        return "after"
    return "concurrent"


def _gen_pair(rel, length, vmax):
    for _ in range(500):
        v = [random.randint(0, vmax) for _ in range(length)]
        if rel == "equal":
            u = list(v)
        elif rel == "before":
            u = list(v)
            hi = [i for i in range(length) if v[i] > 0]
            if not hi:
                continue
            u[random.choice(hi)] -= 1
        elif rel == "after":
            u = list(v)
            u[random.randint(0, length - 1)] += 1
        else:
            u = list(v)
            hi = [i for i in range(length) if v[i] > 0]
            if not hi or length < 2:
                continue
            a = random.choice(hi)
            b = random.choice([i for i in range(length) if i != a])
            u[a] -= 1
            u[b] += 1
        if _classify(u, v) == rel:
            return u, v
    raise RuntimeError("vector_clock_order: failed to build a valid " + rel + " pair")


@dataclass
class VectorClockOrderConfig(Config):
    length: int = 2
    vmax: int = 2

    def apply_difficulty(self, level):
        self.length = sround(self.length + level)
        self.vmax = sround(self.vmax + level)


class VectorClockOrder(Task):
    summary = ("Classify the causal order of two vector timestamps as equal, "
               "before, after, or concurrent, over balanced relations, varied "
               "lengths and clock-value ranges.")
    config_cls = VectorClockOrderConfig

    def generate_entry(self):
        rel = random.choice(["equal", "before", "after", "concurrent"])
        u, v = _gen_pair(rel, self.config.length, self.config.vmax)
        metadata = edict({"u": u, "v": v, "relation": rel})
        metadata.payload = {"u": u, "v": v}
        return Entry(metadata=metadata, answer=rel)

    def render_prompt(self, metadata):
        return (
            "A vector clock timestamp is a tuple of counters, one per process; "
            "process i's vec[i] is the number of events process i has sent before "
            f"that point in time. Two timestamps u = {tuple(metadata.payload['u'])} "
            f"and v = {tuple(metadata.payload['v'])} fix two moments. "
            "The order relation between u and v is determined componentwise: "
            "u is 'before' v when every u[i] <= v[i] and at least one is strict; "
            "u is 'after' v when every u[i] >= v[i] and at least one is strict; "
            "u 'equals' v when all components match; otherwise the two are "
            "'concurrent' (one component is strictly smaller and another strictly "
            "greater). "
            "What is the order relation between u and v? "
            "The answer is one of the exact words: equal, before, after, or concurrent."
        )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        return 1.0 if answer.strip().lower() == entry.answer else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'vector_clock_order (draw 1 of 2)',
 'hypothesis': 'W1-036',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/vector_clock_order',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2985613391,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
