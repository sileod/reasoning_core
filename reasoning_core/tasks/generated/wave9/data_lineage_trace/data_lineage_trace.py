import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict


TASK_META = {'parent_source_id': None,
 'idea': 'data_lineage_trace (draw 1 of 1)',
 'hypothesis': 'HV-037',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/data_lineage_trace',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 895421081,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


class LineageConfig(Config):
    level: int = 2
    n_rows: int = 6
    n_tables: int = 3
    n_filters: int = 1
    n_joins: int = 1
    n_aggs: int = 1
    n_transforms: int = 1

    def apply_difficulty(self, level):
        self.n_rows = 5 + level
        self.n_tables = min(3 + level, 7)
        self.n_filters = min(1 + level, 4)
        self.n_joins = min(1 + level // 2, 3)
        self.n_aggs = 1
        self.n_transforms = min(level, 4)


class DataLineageTrace(Task):
    summary = ("Follow records through filters, joins, aggregations, and transformations "
               "to return the source records contributing to a requested output; "
               "modes: keep/drop filter on a key value, inner join on equal values "
               "(dimension rows added to the lineage), sum aggregation (ids combined), "
               "and add/sub/multiply/integer-divide transforms on the aggregate; answered "
               "by the non-empty set of original source row ids sorted ascending and "
               "comma-separated.")
    config_cls = LineageConfig
    task_version = 2

    def generate_entry(self):
        cfg = self.config
        # Each source row gets a globally unique integer id.
        # tables[t] is a list of {"id": int, "val": int}.
        base_id = 1000
        next_id = [base_id]
        tables = []
        for t in range(cfg.n_tables):
            nrows = random.randint(cfg.n_rows, cfg.n_rows + 2)
            rows = []
            for _ in range(nrows):
                rows.append({"id": next_id[0], "val": random.randint(1, 25)})
                next_id[0] += 1
            tables.append(rows)

        # Track lineage: a set of contributing source ids on each running row,
        # plus a numeric value. Start from table 0.
        current = [dict(src={r["id"]}, val=r["val"]) for r in tables[0]]
        ops_desc = []

        # 1) filters. Each filter is tuned so that at least one row always survives:
        # a "greater than" filter drops everything at or below a threshold below the
        # current max (the max row survives); a "less than or equal to" filter keeps
        # everything at or below a threshold above the current min (the min survives).
        for _ in range(cfg.n_filters):
            vals = [r["val"] for r in current]
            lo, hi = min(vals), max(vals)
            keep_gt = random.choice([True, False])
            if keep_gt:
                threshold = random.randint(lo, hi - 1) if hi > lo else lo - 1
                current = [r for r in current if r["val"] > threshold]
            else:
                threshold = random.randint(lo, hi) if hi > lo else lo
                current = [r for r in current if r["val"] <= threshold]
            ops_desc.append(("filter", threshold, keep_gt))

        # 2) joins (inner on equal values). Guarantee a match so the join is not empty:
        # a dimension row with a value that already occurs in the current set must exist.
        for i in range(cfg.n_joins):
            t = random.randint(1, cfg.n_tables - 1)
            other = tables[t]
            cur_vals = {r["val"] for r in current}
            if not any(orow["val"] in cur_vals for orow in other):
                other.append({"id": next_id[0], "val": random.choice(list(cur_vals))})
                next_id[0] += 1
            merged = []
            for row in current:
                for orow in other:
                    if row["val"] == orow["val"]:
                        merged.append({"src": row["src"] | {orow["id"]},
                                       "val": row["val"]})
            current = merged
            ops_desc.append(("join", t, None))

        # 3) aggregate (sum): collapse all remaining rows into one
        total_src = set()
        total_val = 0
        for row in current:
            total_src |= row["src"]
            total_val += row["val"]
        current = [{"src": total_src, "val": total_val}]
        ops_desc.append(("aggregate",))

        # 4) transforms on the aggregate value
        cur_val = total_val
        for _ in range(cfg.n_transforms):
            op = random.choice(["add", "mul", "sub", "floordiv"])
            operand = random.randint(1, 6)
            if op == "add":
                cur_val = cur_val + operand
            elif op == "mul":
                cur_val = cur_val * operand
            elif op == "sub":
                cur_val = cur_val - operand
            else:
                cur_val = cur_val // operand
            ops_desc.append(("transform", op, operand))
        final_val = cur_val

        # Answer: the contributing source ids sorted ascending
        ans_list = sorted(total_src)
        answer = ", ".join(str(x) for x in ans_list)

        # Self-check: non-empty, all ids are real, joined ids present, ids unique.
        assert ans_list, "pipeline unexpectedly empty"
        assert min(ans_list) >= base_id
        assert len(ans_list) == len(set(ans_list))
        base_ids = {r["id"] for r in tables[0]}
        for oid in ans_list:
            assert oid in {r["id"] for tab in tables for r in tab}


        # ---- render the prompt ----
        lines = []
        lines.append("A data warehouse task: a query reads from a base fact table T0 and several dimension tables.")
        lines.append("The base table T0 is a list of source rows; each row shows its source id and a key value:")
        for idx, row in enumerate(tables[0]):
            lines.append("  T0 row {}: source_id {}, key {}".format(idx, row["id"], row["val"]))
        for t in range(1, cfg.n_tables):
            lines.append("Table T{} (dimension) is a list of source rows; each row shows its source id and a key value:".format(t))
            for idx, row in enumerate(tables[t]):
                lines.append("  T{} row {}: source_id {}, key {}".format(
                    t, idx, row["id"], row["val"]))

        lines.append("A SQL-like pipeline runs over T0 in this order:")
        step = 0
        for op in ops_desc:
            step += 1
            if op[0] == "filter":
                _, thr, gt = op
                cmp = "greater than" if gt else "less than or equal to"
                lines.append("  step {}: keep only rows whose key is {} {} (drop the rest)".format(step, cmp, thr))
            elif op[0] == "join":
                _, t, _ = op
                lines.append("  step {}: inner-join with table T{} on equal key values".format(step, t))
            elif op[0] == "aggregate":
                lines.append("  step {}: aggregate by summing the keys of all remaining rows into one row; the source-id set of that row is the union of the source-id sets of all remaining rows".format(step))
            elif op[0] == "transform":
                _, o, operand = op
                nm = {"add": "add", "mul": "multiply by", "sub": "subtract",
                      "floordiv": "integer-divide by"}[o]
                lines.append("  step {}: {} {} the aggregated key value".format(step, nm, operand))

        lines.append("Question: which source rows (by source id) end up contributing to the final output key value {}? Contributing rows are the T0 rows kept after filtering plus every dimension row added by a join that survives to the aggregation.".format(final_val))

        lines.append("Give the answer as a single line: the contributing source ids, sorted in ascending order, separated by commas and spaces. Every pipeline in this exercise leaves at least one source row reaching the output.")

        prompt = "\n".join(lines)

        metadata = edict({"prompt": prompt, "final_val": final_val, "n_sources": len(ans_list)})
        metadata.payload = {"prompt": prompt}
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return metadata.prompt

    def score_answer(self, answer, entry):
        gold = entry.answer
        a = (answer or "").strip()
        if not a:
            return 0.0
        if gold == "NONE":
            return 1.0 if a.upper() == "NONE" else 0.0
        if a.upper() == "NONE":
            return 0.0
        try:
            got = [int(x.strip()) for x in a.split(",") if x.strip() != ""]
        except ValueError:
            return 0.0
        gold_nums = {int(x.strip()) for x in gold.split(",") if x.strip()}
        if sorted(got) == sorted(gold_nums):
            return 1.0
        return 0.0
