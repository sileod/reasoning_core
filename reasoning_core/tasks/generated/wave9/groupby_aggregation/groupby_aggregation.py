import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'groupby_aggregation (draw 1 of 1)',
 'hypothesis': 'HV-032',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/groupby_aggregation',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 349546629,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

OPS = ["count", "sum", "min", "max"]


@dataclass
class GroupbyAggregationConfig(Config):
    n_groups: int = 3
    n_rows: int = 6
    val_max: int = 20

    def apply_difficulty(self, level):
        self.n_groups = sround(self.n_groups + level)
        self.n_rows = sround(self.n_rows + 2 * level)


class GroupbyAggregation(Task):
    summary = ("Partition rows by one or more keys and compute exact grouped count, sum, minimum, "
               "or maximum values followed by filtering and ordering.")
    config_cls = GroupbyAggregationConfig

    def generate_entry(self):
        cfg = self.config
        n_groups = int(cfg.n_groups)
        n_rows = int(cfg.n_rows)
        val_max = int(cfg.val_max)
        op = random.choice(OPS)

        groups = ["G%d" % i for i in range(n_groups)]

        while True:
            rows = []
            for _ in range(n_rows):
                rows.append([random.choice(groups), random.randint(0, val_max)])
            agg = {g: [] for g in groups}
            for g, v in rows:
                agg[g].append(v)
            if all(len(agg[g]) > 0 for g in groups):
                break

        if op == "count":
            aggval = {g: len(agg[g]) for g in groups}
            measure = "row count"
        elif op == "sum":
            aggval = {g: sum(agg[g]) for g in groups}
            measure = "sum"
        elif op == "min":
            aggval = {g: min(agg[g]) for g in groups}
            measure = "minimum"
        else:
            aggval = {g: max(agg[g]) for g in groups}
            measure = "maximum"

        vals = list(aggval.values())
        low = min(min(vals), 1) if op == "count" else min(vals)
        threshold = random.randint(low, low + random.randint(0, 5))
        if op != "count" and threshold < 0:
            threshold = 0

        surviving = [g for g in groups if aggval[g] >= threshold]
        surviving.sort(key=lambda g: (-aggval[g], g))
        answer = ",".join(surviving) if surviving else "none"

        # verifier rebuild
        check = {g: [] for g in groups}
        for g, v in rows:
            check[g].append(v)
        if op == "count":
            cval = {g: len(check[g]) for g in groups}
        elif op == "sum":
            cval = {g: sum(check[g]) for g in groups}
        elif op == "min":
            cval = {g: min(check[g]) for g in groups}
        else:
            cval = {g: max(check[g]) for g in groups}
        assert cval == aggval
        kept = [g for g in groups if cval[g] >= threshold]
        kept.sort(key=lambda g: (-cval[g], g))
        assert (",".join(kept) if kept else "none") == answer

        metadata = edict({
            "rows": rows,
            "op": op,
            "threshold": threshold,
            "aggval": {g: aggval[g] for g in groups},
        })
        metadata.payload = {
            "rows": [list(r) for r in rows],
            "op": op,
            "threshold": threshold,
        }
        metadata.measure = measure

        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        rows_str = " ".join("%s:%d" % (g, v) for g, v in metadata.rows)
        measure = metadata.measure
        text = (
            "We have a table with rows [group, value]. Rows: " + rows_str + ".\n"
            "Partition the rows by the group letter and for each group compute the "
            + measure + " of that group's values (for 'row count', the number of rows). "
            "Keep only the groups whose " + measure + " is at least the threshold "
            + str(metadata.threshold) + ". "
            "List the kept group letters in decreasing order of their " + measure + "; "
            "when two groups tie on the " + measure + ", order them alphabetically. "
            "Answer with the comma-separated list of kept group letters, e.g. G0,G2. "
            "If no group is kept, answer the single word none."
        )
        return text

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        a = answer.strip()
        if a == entry.answer:
            return 1.0
        return 0.0
