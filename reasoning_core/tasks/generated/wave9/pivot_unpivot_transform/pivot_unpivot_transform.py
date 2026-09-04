import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'pivot_unpivot_transform (draw 1 of 1)',
 'hypothesis': 'HV-034',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/pivot_unpivot_transform',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3773979110,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}

AGG_FUNCS = ["sum", "avg", "min", "max", "count"]


def _fmt(v):
    return str(int(v)) if float(v).is_integer() else f"{float(v):g}"


def _compute_agg(vals, agg):
    if agg == "sum":
        return sum(vals)
    if agg == "count":
        return len(vals)
    if agg == "avg":
        return sum(vals) / len(vals)
    if agg == "min":
        return min(vals)
    return max(vals)


def _pivot_columns(cols, id_keys, agg):
    groups = {}
    order = []
    for r in range(len(cols[0])):
        key = tuple(id_keys[r])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)
    result = []
    for key in order:
        rows = groups[key]
        vals = [cols[c][r] for r in rows for c in range(len(cols))]
        result.append((key, _compute_agg(vals, agg)))
    result.sort(key=lambda item: item[0])
    return result


class PivotUnpivotTransformConfig(Config):
    n_rows: int = 3
    n_cols: int = 3
    n_ids: int = 3
    max_val: int = 9

    def apply_difficulty(self, level):
        self.n_rows = sround(self.n_rows + level)
        self.n_cols = sround(self.n_cols + level)
        self.n_ids = sround(self.n_ids + 1 + level)
        self.max_val = 9 + level * 10


class PivotUnpivotTransform(Task):
    summary = "Reshape records between long and wide forms under explicit aggregation and missing-value rules, returning requested output records or cells."
    config_cls = PivotUnpivotTransformConfig

    def generate_entry(self):
        cfg = self.config
        kind = random.choice(["pivot", "unpivot"])

        if kind == "pivot":
            n_rows = cfg.n_rows
            n_ids = cfg.n_ids
            max_val = cfg.max_val
            cols = _build_matrix(n_rows, 2, max_val)
            agg = random.choice(AGG_FUNCS)
            id_keys = [_gen_key(n_ids) for _ in range(n_rows)]

            expected = _pivot_columns(cols, id_keys, agg)
            out_lines = []
            for key, val in expected:
                out_lines.append(f"({key[0]}, {key[1]}) -> {_fmt(val)}")
            answer = " ".join(out_lines)

            header = "id1,id2,value1,value2"
            data_rows = []
            for r in range(n_rows):
                data_rows.append(",".join([str(id_keys[r][0]), str(id_keys[r][1]),
                                           _fmt(cols[0][r]), _fmt(cols[1][r])]))

            agg_text = {
                "sum": "the sum of all values in that cell",
                "avg": "the average of all values in that cell",
                "min": "the minimum of all values in that cell",
                "max": "the maximum of all values in that cell",
                "count": "the count of rows in that cell",
            }[agg]
            agg_word = {"sum": "sum", "avg": "average", "min": "minimum",
                        "max": "maximum", "count": "count"}[agg]

            metadata = edict({
                "operation": "pivot",
                "agg": agg,
                "header": header,
                "data_rows": data_rows,
                "answer": answer,
            })
            metadata.payload = {
                "agg_text": agg_text,
                "agg_word": agg_word,
                "table_block": [header] + data_rows,
                "request": (
                    "Reshape this table from long to wide form: group the rows by the id1,id2 "
                    "pair so each distinct pair becomes one output cell holding the " + agg_text +
                    ". Report, for each distinct id pair, the resulting value. Write the answer "
                    "as lines \"(id1, id2) -> value\" separated by spaces, in ascending id1 then id2 order."
                ),
            }
            return Entry(metadata=metadata, answer=answer)

        else:
            n_rows = cfg.n_rows
            n_ids = cfg.n_ids
            n_cols = cfg.n_cols
            max_val = cfg.max_val
            col_names = [f"c{i}" for i in range(n_cols)]
            ids = [random.randint(0, n_ids - 1) for _ in range(n_rows)]
            vals = [[_rand_or_none(max_val) for _ in range(n_cols)] for _ in range(n_rows)]

            missing_row = random.random() < 0.5
            missing_val = random.randint(0, max_val) if not missing_row else None

            records = []
            for r in range(n_rows):
                for c in range(n_cols):
                    v = vals[r][c]
                    if v is None:
                        cell = "NA" if missing_row else _fmt(missing_val)
                    else:
                        cell = _fmt(v)
                    records.append((ids[r], col_names[c], cell))
            records.sort(key=lambda rec: (rec[0],))
            out_lines = [f"(id{rec[0]}, {rec[1]}, {rec[2]})" for rec in records]
            answer = " ".join(out_lines)

            header = "id," + ",".join(col_names)
            data_rows = []
            for r in range(n_rows):
                row = [str(ids[r])] + [(_fmt(v) if v is not None else "NA") for v in vals[r]]
                data_rows.append(",".join(row))

            missing_text = (
                "Any missing cell shown as NA is treated as missing and must be written as NA in the output."
                if missing_row else
                f"Any missing cell shown as NA is treated as the value {_fmt(missing_val)}."
            )

            metadata = edict({
                "operation": "unpivot",
                "missing_row": missing_row,
                "missing_val": missing_val,
                "header": header,
                "data_rows": data_rows,
                "answer": answer,
            })
            metadata.payload = {
                "missing_text": missing_text,
                "table_block": [header] + data_rows,
                "request": (
                    "Reshape this table from wide to long form: one long-form record per cell. "
                    + missing_text + " List every resulting record as \"(id, column, value)\", "
                    "columns in their table order and records grouped and ordered by ascending id."
                ),
            }
            return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        if metadata.operation == "pivot":
            return (
                "The table below is in long form. Reshape it into wide form as described.\n\n"
                + metadata.payload["request"] + "\n\n"
                + render_payload({"table": metadata.payload["table_block"]})
                + "\n\nThe answer is a list of lines separated by spaces."
            )
        else:
            return (
                "The table below is in wide form. Reshape it into long form as described.\n\n"
                + metadata.payload["request"] + "\n\n"
                + render_payload({"table": metadata.payload["table_block"]})
                + "\n\nThe answer is a list of records separated by spaces."
            )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        a = " ".join(answer.strip().split())
        g = " ".join(entry.answer.strip().split())
        return 1.0 if a == g else 0.0


def _build_matrix(n_rows, n_cols, max_val):
    return [[random.randint(1, max_val) for _ in range(n_rows)] for _ in range(n_cols)]


def _gen_key(n_ids):
    a = random.randint(0, n_ids - 1)
    b = random.randint(0, n_ids - 1)
    return (min(a, b), max(a, b))


def _rand_or_none(max_val):
    if random.random() < 0.2:
        return None
    return random.randint(0, max_val)
