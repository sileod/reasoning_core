import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'abstract_interval_analysis (draw 1 of 1)',
 'hypothesis': 'HV-030',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/abstract_interval_analysis',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1549547534,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

INF = float("inf")
NEGINF = float("-inf")


def _fmt(x):
    if x == INF:
        return "+inf"
    if x == NEGINF:
        return "-inf"
    return str(int(x))


def _shifted(iv, op, c):
    lo, hi = iv
    if op == "+":
        return (lo + c, hi + c)
    return (lo - c, hi - c)


def _lub(intervals):
    return (min(iv[0] for iv in intervals), max(iv[1] for iv in intervals))


def _parse_interval(text):
    if not isinstance(text, str):
        return None
    s = text.strip().strip("()").split(",")
    if len(s) != 2:
        return None
    out = []
    for tok in s:
        tok = tok.strip()
        if tok == "+inf":
            out.append(INF)
        elif tok == "-inf":
            out.append(NEGINF)
        else:
            try:
                out.append(float(tok))
            except (TypeError, ValueError):
                return None
    lo, hi = out
    if lo > hi:
        return None
    return (lo, hi)


@dataclass
class IntervalConfig(Config):
    n_stmt: int = 4
    max_const: int = 6
    n_opts: int = 3
    widens: int = 0

    def apply_difficulty(self, level):
        self.n_stmt = sround(self.n_stmt + 2 * level)
        self.max_const = sround(self.max_const + 2 * level)
        self.n_opts = sround(self.n_opts + (level > 1))
        self.widens = sround(self.widens + (level > 1))


class AbstractIntervalAnalysis(Task):
    summary = "Execute interval abstract interpretation through arithmetic, branches, joins, and stated loop widening, returning a queried abstract value."

    config_cls = IntervalConfig

    def generate_entry(self):
        cfg = self.config
        VARS = ["a", "b", "c"]
        state = {v: (0.0, 0.0) for v in VARS}
        prog = []
        for _ in range(cfg.n_stmt):
            kind = random.random()
            if kind < 0.55 or cfg.n_stmt <= 2:
                # arithmetic assignment x := <shift of another var> or constant
                v = random.choice(VARS)
                u = random.choice(VARS)
                op = random.choice(["+", "-"])
                c = random.randint(-cfg.max_const, cfg.max_const)
                c = c if c != 0 else 1
                state[v] = _shifted(state[u], op, float(c))
                prog.append(("SET", v, u, op, c))
            elif kind < 0.9:
                # branch / join
                v = random.choice(VARS)
                nopt = random.randint(2, cfg.n_opts)
                intervals = []
                texts = []
                for _i in range(nopt):
                    r = random.random()
                    if r < 0.4:
                        c = random.randint(-cfg.max_const, cfg.max_const)
                        intervals.append((float(c), float(c)))
                        texts.append(f"{c}")
                    elif r < 0.7:
                        u = random.choice(VARS)
                        intervals.append(state[u])
                        texts.append(u)
                    else:
                        u = random.choice(VARS)
                        op = random.choice(["+", "-"])
                        c = random.randint(-cfg.max_const, cfg.max_const)
                        c = c if c != 0 else 1
                        intervals.append(_shifted(state[u], op, float(c)))
                        texts.append(f"{u}{op}{c}")
                intervals.append(state[v])
                texts.append(v)
                state[v] = _lub(intervals)
                prog.append(("BRANCH", v, texts))
            else:
                # loop with widening
                v = random.choice(VARS)
                delta = random.randint(-cfg.max_const, cfg.max_const)
                delta = delta if delta != 0 else 1
                lo, hi = state[v]
                if delta > 0:
                    state[v] = (lo, INF)
                else:
                    state[v] = (NEGINF, hi)
                prog.append(("WIDEN", v, delta))

        target = random.choice(VARS)
        lo, hi = state[target]
        if lo == NEGINF and hi == INF:
            for v in VARS:
                l2, h2 = state[v]
                if not (l2 == NEGINF and h2 == INF):
                    target = v
                    lo, hi = state[v]
                    break
        assert lo <= hi, (lo, hi, prog)
        answer = f"({_fmt(lo)}, {_fmt(hi)})"

        lines = [
            "Variables a, b and c are initialized to 0. The program below is analyzed with "
            "interval abstract interpretation: every expression yields an integer interval [lo, hi].",
            "An assignment x := <expr> replaces x's interval by the interval of <expr>.",
            "A join 'x := any{ p1 , p2 , ... }' puts x into the least upper bound (the interval "
            "spanning all of those options together with x's previous interval).",
            "A loop that repeatedly does x := x + d is resolved by standard widening: a positive "
            "increment raises x's upper bound to +infinity and a negative increment lowers its "
            "lower bound to -infinity.",
            "",
        ]
        for st in prog:
            if st[0] == "SET":
                _, v, u, op, c = st
                lines.append(f"{v} := {u} {op} {c}")
            elif st[0] == "BRANCH":
                _, v, texts = st
                lines.append(f"{v} := any{{ {' , '.join(texts)} }}")
            else:
                _, v, delta = st
                lines.append(f"loop: {v} := {v} {('+' if delta > 0 else '-')} {abs(delta)}   (widened)")

        payload = {"program": "\n".join(lines).rstrip()}
        metadata = edict({
            "payload": payload,
            "query": f"the abstract interval of variable '{target}' after the whole program has been analyzed",
        })
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        payload = render_payload(metadata.payload)
        return f"{payload}\n\nCompute {metadata.query}.\n\nThe answer is the interval in the form (lo, hi), using -inf or +inf for unbounded ends, for example (-inf, 5) or (3, +inf)."

    def score_answer(self, answer, entry):
        got = _parse_interval(answer)
        if got is None:
            return 0.0
        want = _parse_interval(entry.answer)
        return 1.0 if got == want else 0.0


_TASK = AbstractIntervalAnalysis
