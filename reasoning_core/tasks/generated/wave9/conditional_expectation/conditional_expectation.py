from dataclasses import dataclass
from fractions import Fraction

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

import random


def _fmt(x):
    """Render a Fraction in reduced p/q form, as an integer if whole."""
    if x.denominator == 1:
        return str(x.numerator)
    return "%d/%d" % (x.numerator, x.denominator)


def _parse(answer):
    """Parse 'p/q' or 'int' into a Fraction, else None."""
    if isinstance(answer, (int, Fraction)):
        return Fraction(answer)
    if isinstance(answer, float):
        return Fraction(answer).limit_denominator()
    if not isinstance(answer, str):
        return None
    s = answer.strip()
    if s.count("/") == 1:
        a, b = s.split("/")
        try:
            return Fraction(int(a.strip()), int(b.strip()))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return Fraction(int(s))
    except ValueError:
        return None


@dataclass
class ConditionalExpectationConfig(Config):
    n_outcomes: int = 5
    n_vars: int = 2

    def apply_difficulty(self, level):
        self.n_outcomes = sround(self.n_outcomes + level * 2)
        self.n_vars = sround(self.n_vars + level // 2)


class ConditionalExpectation(Task):
    summary = "Compute exact rational conditional expectations of a stated variable over a finite joint distribution after restricting to one- or two-variable observations or range events, over varied marginal and correlated supports."
    config_cls = ConditionalExpectationConfig

    def generate_entry(self):
        n = int(self.config.n_outcomes)
        nv = int(self.config.n_vars) if self.config.n_vars >= 2 else 2

        while True:
            vals = []
            for _ in range(nv):
                vals.append([random.randint(-9, 9) for _ in range(n)])
            probs = [random.randint(1, 6) for _ in range(n)]
            tot = sum(probs)
            mass = [Fraction(p, tot) for p in probs]

            mode = random.randint(0, 2)
            if mode == 0:
                vi = random.randrange(nv)
                threshold = random.randint(-10, 10)
                op = random.choice(["\u2265", "\u2264"])
                if op == "\u2265":
                    idx = [i for i in range(n) if vals[vi][i] >= threshold]
                    cond = "V%d %s %d" % (vi + 1, "\u2265", threshold)
                else:
                    idx = [i for i in range(n) if vals[vi][i] <= threshold]
                    cond = "V%d %s %d" % (vi + 1, "\u2264", threshold)
                if len(idx) < 2 or len(idx) == n:
                    continue
            elif mode == 1:
                vi = random.randrange(nv)
                target = vals[vi][random.randrange(n)]
                idx = [i for i in range(n) if vals[vi][i] == target]
                if len(idx) < 2 or len(idx) == n:
                    continue
                cond = "V%d = %d" % (vi + 1, target)
            else:
                vi = random.randrange(nv)
                vj = (vi + 1) % nv
                ai = vals[vi][random.randrange(n)]
                aj = vals[vj][random.randrange(n)]
                idx = [i for i in range(n) if vals[vi][i] == ai and vals[vj][i] == aj]
                if len(idx) < 2 or len(idx) == n:
                    continue
                cond = "V%d = %d and V%d = %d" % (vi + 1, ai, vj + 1, aj)

            denom = sum(mass[i] for i in idx)
            if denom == 0:
                continue

            while True:
                target = random.randrange(nv)
                num = sum(mass[i] * vals[target][i] for i in idx)
                e = Fraction(num, denom)
                lo = min(vals[target])
                hi = max(vals[target])
                if Fraction(lo) <= e <= Fraction(hi):
                    break

            metadata = edict({
                "n_outcomes": n,
                "n_vars": nv,
                "values": vals,
                "probs": [p for p in probs],
                "condition": cond,
                "target_var": target,
                "support_count": len(idx),
            })
            metadata.payload = {
                "n_outcomes": n,
                "n_vars": nv,
                "values": vals,
                "probs": [p for p in probs],
                "condition": cond,
                "target": "V%d" % (target + 1),
            }
            return Entry(metadata=metadata, answer=_fmt(e))

    def render_prompt(self, metadata):
        p = metadata.payload
        n = p["n_outcomes"]
        nv = p["n_vars"]
        total_w = sum(p["probs"])
        header = "outcome | " + " | ".join("V%d" % (v + 1) for v in range(nv)) + " | weight"
        sep = "-" * len(header)
        rows = []
        for i in range(n):
            cell_vals = " | ".join(str(p["values"][v][i]) for v in range(nv))
            rows.append("%d | %s | %d" % (i + 1, cell_vals, p["probs"][i]))
        table = "\n".join([header, sep] + rows)
        cond = p["condition"]
        target = p["target"]
        out = (
            "Below is a finite joint distribution over %d equally numbered outcomes "
            "(the probability of each row is its weight divided by %d).\n"
            "%s\n\n"
            "Restrict to the rows where %s, renormalize those weights, and compute the "
            "conditional expectation of %s over that restricted distribution.\n"
            "The answer is the exact reduced rational value a/b, or a single integer if it is whole."
            % (n, total_w, table, cond, target)
        )
        return out

    def score_answer(self, answer, entry):
        parsed = _parse(answer)
        if parsed is None:
            return 0.0
        gold = _parse(entry.answer)
        return 1.0 if parsed == gold else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'conditional_expectation (draw 1 of 1)',
 'hypothesis': 'HV-005',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/conditional_expectation',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 729651269,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
