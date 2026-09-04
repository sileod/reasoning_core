import random
from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, edict, render_payload
from reasoning_core.template import stochastic_rounding as sround


@dataclass
class CounterfactualTwinConfig(Config):
    min_vars: int = 3
    max_vars: int = 5
    coef_span: int = 2
    val_span: int = 5

    def apply_difficulty(self, level):
        self.min_vars = sround(min(3 + level, 6))
        self.max_vars = sround(min(4 + level, 7))
        self.coef_span = 2 + level
        self.val_span = 4 + level


def _predict(n, coef, v, k, j, w):
    u = [0] * n
    u[0] = v[0]
    for i in range(1, n):
        u[i] = v[i] - coef[i] * v[i - 1]
    V = [0] * n
    for i in range(n):
        if i == 0:
            V[i] = w if k == 0 else u[0]
        elif i == k:
            V[i] = w
        else:
            V[i] = coef[i] * V[i - 1] + u[i]
    return V[j]


def _answer_formula(n, coef, v, k, j, w):
    prod = 1
    for i in range(k + 1, j + 1):
        prod *= coef[i]
    return prod * (w - v[k]) + v[j]


class CounterfactualTwinModel(Task):
    summary = (
        "Perform abduction, intervention, and prediction in deterministic finite "
        "structural causal models (chains of linear equations), returning a queried "
        "counterfactual integer value V_j after intervening V_k."
    )
    config_cls = CounterfactualTwinConfig
    task_version = 2

    def generate_entry(self):
        cfg = self.config
        for _ in range(200):
            n = random.randint(cfg.min_vars, cfg.max_vars)
            coef = [0] * n
            for i in range(1, n):
                while True:
                    c = random.randint(-cfg.coef_span, cfg.coef_span)
                    if c != 0:
                        break
                coef[i] = c
            v = [random.randint(-cfg.val_span, cfg.val_span) for _ in range(n)]
            k = random.randint(0, n - 2)
            j = random.randint(k + 1, n - 1)
            w = random.randint(-cfg.val_span, cfg.val_span)
            while w == v[k]:
                w = random.randint(-cfg.val_span, cfg.val_span)
            ans = _answer_formula(n, coef, v, k, j, w)
            check = _predict(n, coef, v, k, j, w)
            if ans != check:
                continue
            if ans < -100000 or ans > 100000:
                continue
            metadata = edict(
                {
                    "n": n,
                    "coef": coef,
                    "v": v,
                    "k": k,
                    "j": j,
                    "w": w,
                }
            )
            metadata.payload = {
                "n": n,
                "coef": coef,
                "v": v,
                "k": k,
                "j": j,
                "w": w,
            }
            return Entry(metadata=metadata, answer=str(int(ans)))
        raise RuntimeError("could not generate a valid counterfactual instance")

    def render_prompt(self, metadata):
        n = metadata.n
        lines = [
            f"Consider a deterministic structural causal model on variables "
            f"V0, V1, ..., V{n - 1}.",
            "The structural equations are (U_i are the unobserved exogenous terms):",
        ]
        lines.append("  V0 = U0")
        for i in range(1, n):
            lines.append(f"  V{i} = {metadata.coef[i]} * V{i - 1} + U{i}")
        obs = ", ".join(f"V{i} = {metadata.v[i]}" for i in range(n))
        lines.append(f"You observe the evidence: {obs}.")
        lines.append(
            f"An intervention sets V{metadata.k} := {metadata.w}, replacing "
            f"V{metadata.k}'s structural equation."
        )
        lines.append(
            f"Using Pearl's three-step procedure (abduction, action, prediction), "
            f"what would V{metadata.j} equal after this intervention?"
        )
        lines.append(
            "The answer is the single integer value of V{0}.".format(metadata.j)
        )
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        return _score(answer, entry)


def _score(answer, entry):
    try:
        got = int(answer.strip())
    except (ValueError, AttributeError, TypeError):
        return 0.0
    if got == int(entry.answer):
        return 1.0
    return 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'counterfactual_twin_model (draw 1 of 1)',
 'hypothesis': 'HV-009',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/counterfactual_twin_model',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3867019559,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
