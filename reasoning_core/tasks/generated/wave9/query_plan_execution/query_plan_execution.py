import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, edict, render_payload, stochastic_rounding as sround


@dataclass
class QueryPlanExecutionConfig(Config):
    n_r: int = 3
    n_s: int = 2
    n_dom: int = 2
    a_max: int = 3

    def apply_difficulty(self, level):
        self.n_r = sround(self.n_r + 0.5 * level)
        self.n_s = sround(self.n_s + 0.4 * level)
        self.n_dom = sround(self.n_dom + 0.6 * level)
        self.a_max = sround(self.a_max + level)


def _execute(meta):
    rows = []
    for a, b in meta.r_rows:
        for bb, c in meta.s_rows:
            if b == bb:
                rows.append([a, b, c])
    for op in meta.plan:
        if op[0] == "filter":
            idx = {"a": 0, "b": 1, "c": 2}[op[1]]
            kind, v = op[2]
            if kind == "ge":
                rows = [r for r in rows if r[idx] >= v]
            elif kind == "lt":
                rows = [r for r in rows if r[idx] < v]
            elif kind == "eq":
                rows = [r for r in rows if r[idx] == v]
            elif kind == "parity":
                rows = [r for r in rows if r[idx] % 2 == v]
    mode = meta.mode
    if mode == "count":
        cnt = Counter(r[2] for r in rows)
        return ", ".join(f"{k}:{cnt[k]}" for k in sorted(cnt))
    if mode == "sum":
        acc = defaultdict(int)
        for r in rows:
            acc[r[2]] += r[0]
        return ", ".join(f"{k}:{acc[k]}" for k in sorted(acc))
    vals = sorted({r[2] for r in rows})
    return ", ".join(str(v) for v in vals)


class QueryPlanExecution(Task):
    summary = ("Execute relational-algebra plans over two keyed relations with filter, inner "
               "join, projection, group-by (count/sum) and sort operators, returning an exact "
               "projected result set.")
    config_cls = QueryPlanExecutionConfig

    def generate_entry(self):
        self.config.seed = random.randrange(2**32)
        cfg = self.config
        for _ in range(400):
            r_rows = [[random.randint(0, cfg.a_max), random.randrange(cfg.n_dom)] for _ in range(cfg.n_r)]
            s_rows = [[random.randrange(cfg.n_dom), random.randrange(cfg.n_dom)] for _ in range(cfg.n_s)]
            mode = random.choice(["count", "sum", "distinct"])
            filt = random.choice([
                ("filter", "a", ("ge", random.randint(0, cfg.a_max))),
                ("filter", "a", ("lt", random.randint(1, cfg.a_max + 1))),
                ("filter", "a", ("eq", random.randint(0, cfg.a_max))),
                ("filter", "c", ("parity", 0)),
                ("filter", "c", ("parity", 1)),
            ])
            meta = edict(r_rows=r_rows, s_rows=s_rows, plan=[filt], mode=mode)
            ans = _execute(meta)
            if ans and _execute(meta) == ans:
                return Entry(metadata=meta, answer=ans)
        raise RuntimeError("query_plan_execution: failed to generate a valid instance")

    def render_prompt(self, meta):
        filt = meta.plan[0]
        col = {"a": "a", "c": "c"}[filt[1]]
        kind, v = filt[2]
        if kind == "ge":
            pred = f"{col} >= {v}"
        elif kind == "lt":
            pred = f"{col} < {v}"
        elif kind == "eq":
            pred = f"{col} == {v}"
        else:
            pred = f"{col} % 2 == {v}"
        if meta.mode == "count":
            tail = "group the joined rows by c and count the rows in each group, then sort by c."
        elif meta.mode == "sum":
            tail = "group the joined rows by c and sum the a column within each group, then sort by c."
        else:
            tail = "project onto column c, drop duplicates, and sort ascending."
        payload = {
            "relation R (columns a b)": "\n".join(f"{a} {b}" for a, b in meta.r_rows),
            "relation S (columns b c)": "\n".join(f"{b} {c}" for b, c in meta.s_rows),
            "plan": (
                f"1. Scan both relations and inner-join them on column b.\n"
                f"2. Filter (σ) rows where {pred}.\n"
                f"3. Project (π) onto the needed columns, then {tail}"
            ),
        }
        return (
            f"{render_payload(payload)}\n\n"
            "Execute the relational-algebra plan. The answer is the exact projected result: "
            "pairs 'c:value' separated by ', ' sorted by c (for count/sum) or the c values "
            "themselves separated by ', ' (for distinct projection)."
        )

    def score_answer(self, answer, entry):
        norm = lambda x: " ".join(str(x).replace(",", " ").split())
        return float(norm(answer) == norm(entry.answer))


TASK_META = {'parent_source_id': None,
 'idea': 'query_plan_execution (draw 1 of 1)',
 'hypothesis': 'HV-040',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/query_plan_execution',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 390250470,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
