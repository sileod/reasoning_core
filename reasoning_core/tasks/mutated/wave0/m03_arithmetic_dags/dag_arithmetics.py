from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround
from reasoning_core.utils import score_scalar
from dataclasses import dataclass
import random
import operator
from fractions import Fraction

TASK_META = {
    'parent_source_id': 'c267a83e5953e4862bec61fb7c72a249dc6d8d945f1116585ac947e52ef26f35',
    'idea': 'Test whether shared-state reuse improves transfer over expression trees.',
    'hypothesis': 'H7',
    'changes': 'Replace arithmetic trees with let-bound DAGs reusing intermediates two to four times.',
    'generation': {'provider_name': 'albert',
                   'model_name': 'deepseek-v4-flash',
                   'adapter_name': 'harness-link',
                   'adapter_version': 'harness-link albert 0.3.0',
                   'harness_name': 'opencode',
                   'harness_version': '1.18.20',
                   'agent_name': 'task-search-worker',
                   'settings': {'variant': None,
                                'requested_seed': 2588473867,
                                'seed_forwarded': True,
                                'temperature': None,
                                'top_p': None,
                                'pure': True,
                                'max_steps': 28,
                                'timeout_seconds': 1200,
                                'sandbox': {'name': 'bubblewrap',
                                            'version': 'bubblewrap 0.8.0'}}}}

OPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "floordiv": operator.floordiv,
}
OP_SYM = {"add": "+", "sub": "-", "mul": "*", "floordiv": "//"}


def _eval_dag(decls):
    env = {}
    for lhs, op, a, b in decls:
        va = env[a] if isinstance(a, str) else a
        vb = env[b] if isinstance(b, str) else b
        env[lhs] = Fraction(OPS[op](va, vb))
    return env


def _eval_dag_steps(decls):
    env = {}
    steps = []
    for lhs, op, a, b in decls:
        va = env[a] if isinstance(a, str) else a
        vb = env[b] if isinstance(b, str) else b
        res = Fraction(OPS[op](va, vb))
        env[lhs] = res
        steps.append(f"{lhs} = {va} {OP_SYM[op]} {vb} = {res}")
    return env, steps


def _display(decls):
    return "\n".join(f"{lhs} = {a} {OP_SYM[op]} {b}" for lhs, op, a, b in decls)


def _count_reuse(decls):
    bound = [d[0] for d in decls]
    reuse = 0
    for idx, (lhs, op, a, b) in enumerate(decls):
        for x in (a, b):
            if isinstance(x, str) and x in bound[:idx]:
                reuse += 1
    return reuse


def _referenced(decls):
    refs = set()
    for _, op, a, b in decls:
        if isinstance(a, str):
            refs.add(a)
        if isinstance(b, str):
            refs.add(b)
    return refs


@dataclass
class DagArithmeticsConfig(Config):
    min_depth: int = 3
    max_depth: int = 5
    min_reuse: int = 2
    max_reuse: int = 4
    min_val: int = 1
    max_val: int = 30
    reuse_prob: float = 0.75

    def apply_difficulty(self, level):
        self.min_depth = sround(self.min_depth + level)
        self.max_depth = sround(self.max_depth + level)
        self.min_reuse = sround(self.min_reuse + level)
        self.max_reuse = sround(self.max_reuse + level)
        self.min_depth = max(self.min_depth, self.min_reuse + 2)
        self.max_depth = max(self.max_depth, self.max_reuse + 2)
        self.max_val = sround(self.max_val + 10 * level)


class DagArithmetics(Task):
    config_cls = DagArithmeticsConfig
    summary = "Evaluate let-bound arithmetic DAGs with reused intermediate bindings."

    def generate_entry(self):
        cfg = self.config
        for _ in range(500):
            n = random.randint(cfg.min_depth, cfg.max_depth)
            names = [f"x{i}" for i in range(n)]
            k = random.randint(min(cfg.min_reuse, n - 1), min(cfg.max_reuse, n - 1))
            reuse_lines = set(random.sample(range(1, n), min(k, n - 1)))
            decls = []
            for i in range(n):
                op = random.choice(list(OPS))
                if i in reuse_lines:
                    a = random.choice(names[:i])
                    b = random.randint(cfg.min_val, cfg.max_val)
                else:
                    a = random.randint(cfg.min_val, cfg.max_val)
                    b = random.randint(cfg.min_val, cfg.max_val)
                if op == "floordiv" and not isinstance(b, str):
                    b = max(1, b)
                decls.append((names[i], op, a, b))

            env = _eval_dag(decls)
            value = env[names[-1]]
            if value.denominator != 1 or abs(value.numerator) > 100000:
                continue

            answer = str(int(value))
            metadata = edict(
                decls=decls,
                root=names[-1],
                reuses=_count_reuse(decls),
                cot="\n".join(_eval_dag_steps(decls)[1]),
            )
            metadata.payload = {"program": _display(decls), "root": names[-1]}
            return Entry(metadata=metadata, answer=answer)
        raise RuntimeError("No feasible DAG found")

    def render_prompt(self, metadata):
        return (
            f"Evaluate the variable {metadata.root} in this let-bound program:\n"
            f"{metadata.payload['program']}\n\n"
            f"The answer is a number."
        )

    def score_answer(self, answer, entry):
        return score_scalar(answer, entry)
