import json
import pandas as pd
import duckdb
import numpy as np
from faker import Faker
import random
from html import escape
from babel.dates import format_date
from babel.numbers import format_decimal
from tabulate import tabulate
from dataclasses import asdict, dataclass, field, replace
from reasoning_core.template import Task, DevTask, Entry, Config, render_payload, stochastic_rounding as sround
from reasoning_core.utils import score_scalar
import csv
import yaml
import io
import re
from numbers import Number
from decimal import Decimal

try:
    from sklearn.metrics import normalized_mutual_info_score
except Exception:
    normalized_mutual_info_score = None

@dataclass
class TableQAConfig(Config):
    num_rows: int = 10
    column_slack: float = 1.0
    complexity: float = 1.0

    def apply_difficulty(self, level):
        self.num_rows = sround(self.num_rows * 1.12 ** level)
        self.column_slack *= 1.08 ** level
        self.complexity *= 1.22 ** level


@dataclass
class TableStatisticsConfig(Config):
    num_rows: int = 12
    num_numeric: int = 4
    num_categories: int = 3
    margin: float = 0.45
    def apply_difficulty(self, level):
        # Difficulty is one branch-free continuum: slightly more objects and
        # closer winners, with a hard prompt-size ceiling.
        self.num_rows = min(20, sround(self.num_rows + 2 * level))
        self.num_numeric = min(8, sround(self.num_numeric + 0.75 * level))
        self.num_categories = min(5, sround(self.num_categories + 0.5 * level))
        self.margin = max(0.08, self.margin * (0.85 ** level))


_faker = Faker()

LOCALES = ["en_US", "fr_FR"]
DATE_FORMATS = ["yyyy-MM-dd", "d MMM yyyy", "MMM d, yyyy", "yyyy/MM/dd"]
NUMBER_FORMATS = ["#,##0.##", "#,##0.00", "#,##0.##", "#,##0.00", "0.###E0"]
TABLEQA_NUMBER_FORMATS = ["0.##", "0.00", "0.###"]
BOOL_FORMATS = [
    {True: "true", False: "false"},
    {True: "yes", False: "no"},
    {True: "Y", False: "N"},
    {True: "1", False: "0"},
    {True: "✓", False: "✗"},
]
SQL_NULL_MARKER = "—"

def generate_random_table(config):
    f = _faker
    pool = [
        ('customer', f.name), ('city', f.city), ('country', f.country), ('email', f.email),
        ('company', f.company), ('product', lambda: f.word().capitalize()), ('job', f.job),
        ('date', lambda: f.date_between('-1y')), ('qty', lambda: random.randint(1, 1000)),
        ('revenue', lambda: round(random.uniform(10, 1000), 2)),
        ('price', lambda: round(random.uniform(5, 500), 2)),
        ('rating', lambda: round(random.uniform(1, 5), 1))
    ]
    width = getattr(config, "num_columns", sround(4 + config.column_slack))
    cols = random.sample(pool, min(width, len(pool)))
    return pd.DataFrame({n: [g() for _ in range(config.num_rows)] for n, g in cols})

def is_date_series(s):
    xs = s.dropna()
    return len(xs) and xs.map(lambda x: hasattr(x, "strftime")).all()


def render_date_series(s):
    locale, fmt = random.choice(LOCALES), random.choice(DATE_FORMATS)
    out = s.map(lambda x: format_date(x, fmt, locale=locale) if pd.notna(x) else x)
    return out, {"kind": "date", "format": fmt, "locale": locale}


def render_number_series(s, number_formats=NUMBER_FORMATS, number_locales=LOCALES):
    locale, fmt = random.choice(number_locales), random.choice(number_formats)
    out = s.map(lambda x: format_decimal(x, format=fmt, locale=locale) if pd.notna(x) else x)
    return out, {"kind": "number", "format": fmt, "locale": locale}


def render_bool_series(s):
    mapping = random.choice(BOOL_FORMATS)
    meta_mapping = {str(k): v for k, v in mapping.items()}
    return s.map(lambda x: mapping.get(x, x)), {"kind": "bool", "mapping": meta_mapping}


def render_nulls(s):
    return s.map(lambda x: SQL_NULL_MARKER if pd.isna(x) else x), {"null": SQL_NULL_MARKER}


def make_display_dataframe(dataframe, number_formats=NUMBER_FORMATS, number_locales=LOCALES):
    df, meta = dataframe.copy(), {}
    for c in df.columns:
        s = df[c]
        if is_date_series(s):
            df[c], meta[c] = render_date_series(s)
        elif pd.api.types.is_bool_dtype(s):
            df[c], meta[c] = render_bool_series(s)
        elif pd.api.types.is_numeric_dtype(s):
            df[c], meta[c] = render_number_series(
                s,
                number_formats=number_formats,
                number_locales=number_locales,
            )
        if df[c].isna().any():
            df[c], null_meta = render_nulls(df[c])
            meta.setdefault(c, {}).update(null_meta)
    return df.astype(object), {"display": meta}


def make_statistics_display_dataframe(dataframe):
    return dataframe.copy().astype(object), {"display": {}}


def apply_display_formats(dataframe, display_meta):
    df = dataframe.copy()
    for c, spec in display_meta.get("display", {}).items():
        if c not in df.columns:
            continue
        if spec.get("kind") == "date":
            df[c] = df[c].map(lambda x: format_date(x, spec["format"], locale=spec["locale"]) if pd.notna(x) else x)
        elif spec.get("kind") == "number":
            df[c] = df[c].map(lambda x: format_decimal(x, format=spec["format"], locale=spec["locale"]) if pd.notna(x) else x)
        elif spec.get("kind") == "bool":
            df[c] = df[c].map(lambda x: spec["mapping"].get(str(x), x))
        if "null" in spec:
            df[c] = df[c].map(lambda x: spec["null"] if pd.isna(x) else x)
    return df.astype(object)


def scalar_kind(x):
    if pd.isna(x):
        return "null"
    if isinstance(x, (bool, np.bool_)):
        return "bool"
    if hasattr(x, "strftime"):
        return "date"
    if isinstance(x, Number):
        return "number"
    return None


def canonical_scalar(x):
    kind = scalar_kind(x)
    if kind == "null":
        return "NULL"
    if kind == "bool":
        return str(bool(x)).lower()
    if kind == "date":
        return x.strftime("%Y-%m-%d")
    return str(x)


def rows(dataframe, index=False):
    return dataframe.reset_index() if index else dataframe


def to_tab(dataframe, tablefmt, index=False):
    df = rows(dataframe, index=index)
    return tabulate(df, headers="keys", tablefmt=tablefmt, showindex=False, disable_numparse=True)


def to_html(dataframe, index=False):
    df = rows(dataframe, index=index)
    out = ["<table>"]
    out.append("<thead><tr>" + "".join(f"<th>{escape(str(c))}</th>" for c in df.columns) + "</tr></thead>")
    out.append("<tbody>")
    for _, row in df.iterrows():
        out.append("<tr>" + "".join(f"<td>{escape(str(row[c]))}</td>" for c in df.columns) + "</tr>")
    out += ["</tbody>", "</table>"]
    return "\n".join(out)


def to_kv_rows(dataframe, index=False):
    df = rows(dataframe, index=index)
    return "\n".join("; ".join(f"{c}: {row[c]}" for c in df.columns) for _, row in df.iterrows())


def to_jsonl(dataframe, index=False):
    return "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows(dataframe, index=index).to_dict("records"))


def to_python_records(dataframe, index=False):
    return repr(rows(dataframe, index=index).to_dict("records"))


def get_renderers(dataframe):
    return {
        "to_string": lambda index=False: to_tab(dataframe, "plain", index=index),
        "to_markdown": lambda index=False: to_tab(dataframe, "pipe", index=index),
        "to_grid": lambda index=False: to_tab(dataframe, "grid", index=index),
        "to_csv": lambda index=False: rows(dataframe, index=index).to_csv(index=False),
        "to_tsv": lambda index=False: rows(dataframe, index=index).to_csv(index=False, sep="\t"),
        "to_pipe": lambda index=False: rows(dataframe, index=index).to_csv(index=False, sep="|"),
        "to_html": lambda index=False: to_html(dataframe, index=index),
        "to_latex": lambda index=False: to_tab(dataframe, "latex", index=index),
        "to_json": lambda index=False: rows(dataframe, index=index).to_json(orient="records", indent=4, force_ascii=False),
        "to_jsonl": lambda index=False: to_jsonl(dataframe, index=index),
        "to_yaml": lambda index=False: yaml.dump(rows(dataframe, index=index).to_dict("records"), default_flow_style=False, sort_keys=False),
        "to_python_records": lambda index=False: to_python_records(dataframe, index=index),
        "to_kv": lambda index=False: to_kv_rows(dataframe, index=index),
    }


TABLE_RENDERER_WEIGHTS = {
    "to_csv": 18, "to_tsv": 15, "to_pipe": 12, "to_string": 10,
    "to_markdown": 9, "to_jsonl": 7, "to_kv": 6,
    "to_python_records": 5, "to_yaml": 5, "to_latex": 4,
    "to_html": 4, "to_json": 3, "to_grid": 2,
}


def sample_renderer(names=None):
    names = list(names or TABLE_RENDERER_WEIGHTS)
    return random.choices(names, weights=[TABLE_RENDERER_WEIGHTS[n] for n in names], k=1)[0]


def sample_distinct_renderers(names, k=2):
    pool, selected = list(names), []
    for _ in range(k):
        choice = sample_renderer(pool)
        selected.append(choice)
        pool.remove(choice)
    return selected


def split_table(dataframe, n):
    n = max(1, min(n, len(dataframe) or 1))
    q, r = divmod(len(dataframe), n)
    out = []
    start = 0
    for i in range(n):
        stop = start + q + (i < r)
        out.append(dataframe.iloc[start:stop])
        start = stop
    return out


def ident(c):
    return f'"{c}"'


def literal(x):
    if pd.isna(x):
        return "NULL"
    if hasattr(x, "isoformat"):
        return f"DATE '{x.isoformat()}'"
    if isinstance(x, str):
        return "'" + x.replace("'", "''") + "'"
    if isinstance(x, (float, np.floating)):
        return str(float(x))
    return str(x)


@dataclass(frozen=True)
class Predicate:
    column: str
    value: object


@dataclass
class QueryPlan:
    predicates: list[Predicate] = field(default_factory=list)
    projection: list[str] = field(default_factory=list)
    expression: str | None = None
    aggregate: str | None = None
    group_by: list[str] = field(default_factory=list)
    order_by: list[str] = field(default_factory=list)
    descending: bool = True
    distinct: bool = False
    limit: int | None = None

    @property
    def scalar(self):
        return bool(self.aggregate and not self.group_by)


_COLUMN_VALUES = {
    "row_id": lambda n: [f"R{i:04d}" for i in range(n)],
    "country": lambda n: np.random.choice(["France", "Germany", "Spain", "Italy"], n),
    "category": lambda n: np.random.choice(["Books", "Electronics", "Clothing", "Food"], n),
    "status": lambda n: np.random.choice(["paid", "pending", "cancelled"], n),
    "customer": lambda n: [f"C{random.randint(1, max(3, n // 3)):03d}" for _ in range(n)],
    "segment": lambda n: np.random.choice(["consumer", "corporate", "education"], n),
    "date": lambda n: [_faker.date_between("-18M", "today") for _ in range(n)],
    "qty": lambda n: np.random.randint(2, 20, n),
    "unit_price": lambda n: np.round(np.random.lognormal(3.1, 0.65, n), 2),
    "discount": lambda n: np.random.choice([0, 0.05, 0.1, 0.2, 0.3], n),
}
_PREDICATE_VALUES = {
    "country": ["France", "Germany", "Spain", "Italy"],
    "status": ["paid", "pending", "cancelled"],
}
_OP_COST = {
    "predicate": 0.8, "expression": 1.0, "aggregate": 1.0,
    "group": 1.4, "order": 0.7, "limit": 0.5, "distinct": 0.8,
    "secondary_sort": 0.8, "projection": 0.6,
}


def legal_extensions(plan):
    out = []
    if len(plan.predicates) < 2:
        out.append("predicate")
    if not plan.expression and not plan.aggregate and not plan.distinct:
        out.append("expression")
    if not plan.aggregate and not plan.order_by and not plan.distinct:
        out.append("aggregate")
    if plan.aggregate and not plan.group_by:
        out.append("group")
    if not plan.order_by and (not plan.aggregate or plan.group_by):
        out.append("order")
    if plan.order_by and plan.limit is None:
        out.append("limit")
    if not plan.aggregate and not plan.expression and not plan.distinct and not plan.order_by:
        out.append("distinct")
    if len(plan.order_by) == 1 and not plan.distinct and not plan.group_by:
        out.append("secondary_sort")
    if not plan.aggregate and not plan.expression and len(plan.projection) < 2:
        out.append("projection")
    return out


def _apply_extension(plan, op):
    if op == "predicate":
        used = {p.column for p in plan.predicates}
        column = random.choice([c for c in _PREDICATE_VALUES if c not in used])
        plan.predicates.append(Predicate(column, random.choice(_PREDICATE_VALUES[column])))
    elif op == "expression":
        plan.expression = f"{ident('qty')} * {ident('unit_price')}"
    elif op == "aggregate":
        plan.aggregate = random.choice(["count", "sum", "avg", "max", "min"])
        if plan.aggregate != "count":
            plan.expression = plan.expression or ident("qty")
        plan.projection = []
    elif op == "group":
        plan.group_by = plan.projection = ["category"]
    elif op == "order":
        plan.order_by = [
            "aggregate" if plan.group_by else
            "expression" if plan.expression else
            plan.projection[0] if plan.distinct else "qty"
        ]
        plan.descending = random.choice([False, True])
    elif op == "limit":
        plan.limit = random.randint(1, 4)
    elif op == "distinct":
        plan.distinct = True
        if plan.projection == ["row_id"]:
            plan.projection = [random.choice(["country", "category", "status"])]
    elif op == "secondary_sort":
        plan.order_by.append("category" if plan.group_by else "row_id")
    elif op == "projection":
        choices = [c for c in ("row_id", "country", "category", "status") if c not in plan.projection]
        plan.projection.append(random.choice(choices))


def sample_query_plan(config):
    plan = QueryPlan(projection=[random.choice(["row_id", "country", "category", "status"])])
    budget = np.random.gamma(shape=max(0.5, config.complexity), scale=1.0)
    while budget > 0:
        candidates = legal_extensions(plan)
        if not candidates:
            break
        weights = [1.3 if op == "predicate" else 1.0 for op in candidates]
        op = random.choices(candidates, weights=weights, k=1)[0]
        cost = _OP_COST[op] * np.random.lognormal(0, 0.18)
        if random.random() > min(1.0, budget / cost):
            break
        _apply_extension(plan, op)
        budget -= cost
    return plan


def schema_for_plan(plan, config):
    required = {
        *(p.column for p in plan.predicates), *plan.projection, *plan.group_by,
        *(c for c in plan.order_by if c not in {"aggregate", "expression"}),
        *re.findall(r'"([^"]+)"', plan.expression or ""),
    }
    if plan.aggregate:
        required.add("qty")
    if not required:
        required.add("row_id")
    available = [c for c in _COLUMN_VALUES if c not in required]
    n_extra = min(np.random.poisson(config.column_slack), max(1, len(required)), len(available))
    columns = [*required, *random.sample(available, n_extra)]
    random.shuffle(columns)
    return columns


def minimum_witness_rows(plan):
    needs = [4, 3 + len(plan.predicates)]
    if plan.limit:
        needs.append(2 * plan.limit + len(plan.predicates))
    if plan.group_by:
        needs.append(2 * max(3, plan.limit or 2) + len(plan.predicates))
    if plan.distinct or len(plan.order_by) > 1:
        needs.append(6)
    return max(needs)


def _condition_dataframe(df, plan):
    n_match = max(2, (plan.limit or 0) + 2,
                  2 * max(3, (plan.limit or 0) + 1) if plan.group_by else 0)
    n_match = min(n_match, len(df) - len(plan.predicates))
    for predicate in plan.predicates:
        alternatives = [x for x in _PREDICATE_VALUES[predicate.column] if x != predicate.value]
        df[predicate.column] = random.choice(alternatives)
    for i in range(n_match):
        for predicate in plan.predicates:
            df.loc[i, predicate.column] = predicate.value
    for offset, predicate in enumerate(plan.predicates):
        row = n_match + offset
        for other in plan.predicates:
            df.loc[row, other.column] = other.value
        df.loc[row, predicate.column] = random.choice(
            [x for x in _PREDICATE_VALUES[predicate.column] if x != predicate.value]
        )

    if "qty" in df:
        df["qty"] = np.arange(2, len(df) + 2)
    if "unit_price" in df:
        df["unit_price"] = np.round(8 + (np.arange(len(df)) * 7 % 23), 2)
    for row in range(n_match, n_match + len(plan.predicates)):
        if "qty" in df:
            df.loc[row, "qty"] = 100 + row
        if "unit_price" in df:
            df.loc[row, "unit_price"] = 100 + 3 * row

    if plan.group_by:
        groups = ["Books", "Electronics", "Clothing", "Food", "Office"][:max(3, (plan.limit or 0) + 1)]
        for i in range(n_match):
            df.loc[i, "category"] = groups[i % len(groups)]
    if plan.distinct and n_match > 1:
        for column in plan.projection:
            df.loc[1, column] = df.loc[0, column]
    if len(plan.order_by) > 1:
        if plan.order_by[0] == "expression":
            df.loc[1, ["qty", "unit_price"]] = df.loc[0, ["qty", "unit_price"]]
        elif plan.order_by[0] in df:
            df.loc[1, plan.order_by[0]] = df.loc[0, plan.order_by[0]]
        if not plan.descending and "row_id" in df:
            df.loc[0, "row_id"], df.loc[1, "row_id"] = df.loc[1, "row_id"], df.loc[0, "row_id"]
    return df


def realize_table(schema, plan, config):
    n = max(
        minimum_witness_rows(plan),
        sround(config.num_rows * np.exp(np.random.normal(0, 0.12))),
    )
    df = pd.DataFrame({column: _COLUMN_VALUES[column](n) for column in schema})
    return _condition_dataframe(df, plan)


def _aggregate_sql(plan):
    if plan.aggregate == "count":
        return "COUNT(*)"
    return f"{plan.aggregate.upper()}({plan.expression})"


def render_query(plan):
    where = " AND ".join(
        f"{ident(p.column)} = {literal(p.value)}" for p in plan.predicates
    ) or "TRUE"
    if plan.aggregate:
        aggregate = _aggregate_sql(plan)
        select = ", ".join(map(ident, plan.projection)) if plan.group_by else (
            aggregate if plan.aggregate == "count" else f"ROUND({aggregate}, 2)"
        )
    else:
        fields = [ident(c) for c in plan.projection]
        if plan.expression:
            fields.append(f"ROUND({plan.expression}, 2) AS value")
        select = ", ".join(fields)
    query = f"SELECT {'DISTINCT ' if plan.distinct else ''}{select} FROM dataframe WHERE {where}"
    if plan.group_by:
        query += " GROUP BY " + ", ".join(map(ident, plan.group_by))
    if plan.order_by:
        keys = [
            _aggregate_sql(plan) if key == "aggregate" else
            plan.expression if key == "expression" else ident(key)
            for key in plan.order_by
        ]
        direction = " DESC" if plan.descending else " ASC"
        query += " ORDER BY " + ", ".join(key + direction for key in keys)
    if plan.limit is not None:
        query += f" LIMIT {plan.limit}"
    return query


def semantic_ablations(plan):
    for i in range(len(plan.predicates)):
        yield f"predicate_{i}", replace(plan, predicates=plan.predicates[:i] + plan.predicates[i + 1:])
    if plan.expression and "*" in plan.expression:
        yield "expression", replace(plan, expression=ident("qty"))
    if plan.aggregate:
        aggregate = "sum" if plan.aggregate == "count" else "max" if plan.aggregate != "max" else "min"
        yield "aggregate", replace(plan, aggregate=aggregate, expression=plan.expression or ident("qty"))
    if plan.group_by:
        yield "group", replace(plan, projection=[], group_by=[], order_by=[], limit=None)
    if plan.order_by:
        yield "order_direction", replace(plan, descending=not plan.descending)
    if plan.limit is not None:
        yield "limit", replace(plan, limit=None)
    if plan.distinct:
        yield "distinct", replace(plan, distinct=False)
    if len(plan.order_by) > 1:
        yield "secondary_sort", replace(plan, order_by=plan.order_by[:1])


def interesting_result(result, plan):
    if result.empty or (plan.limit is not None and len(result) != plan.limit):
        return False
    if plan.scalar or result.size == 1:
        return canonical_scalar(result.iloc[0, 0]) not in {"0", "1", "NULL", "nan", "None"}
    if len(result) > 6:
        return False
    return result.nunique(dropna=False).sum() > 1


def sample_query(conn, config, max_tries=80):
    for _ in range(max_tries):
        plan = sample_query_plan(config)
        schema = schema_for_plan(plan, config)
        df = realize_table(schema, plan, config)
        conn.register("dataframe", df)
        try:
            result = conn.execute(render_query(plan)).df()
            checks = {
                name: not conn.execute(render_query(ablated)).df().equals(result)
                for name, ablated in semantic_ablations(plan)
            }
        except Exception:
            continue
        if interesting_result(result, plan) and all(checks.values()):
            spec = {
                "query_conditioned": True,
                "plan": asdict(plan),
                "n_predicates": len(plan.predicates),
                "has_expression": plan.expression is not None,
                "has_aggregate": plan.aggregate is not None,
                "n_group_keys": len(plan.group_by),
                "n_order_keys": len(plan.order_by),
                "has_limit": plan.limit is not None,
                "distinct": plan.distinct,
                "is_scalar": plan.scalar,
                "feature_checks": checks,
                "num_rows": len(df),
                "num_columns": len(df.columns),
            }
            return df, render_query(plan), result, spec
    raise RuntimeError("Could not synthesize an interesting compositional table query")


class TableQA(Task):
    summary = "Answer queries on tabular data by executing SQL queries over dataframes."
    task_version = 5

    def __init__(self, config=None):
        super().__init__(config=config or TableQAConfig())
        self.balancing_key_ratio = 0.5

    def _result_bucket(self, result):
        if result.shape != (1, 1):
            return f"rows={min(len(result), 4)}"
        try:
            x = float(result.iloc[0, 0])
            return "0" if x == 0 else "1" if x == 1 else "2-3" if x <= 3 else "4+"
        except Exception:
            return "text"
    
    def generate_entry(self):
        conn = duckdb.connect()
        semantic_df, q, result, query_spec = sample_query(conn, self.config)
        display_df, display_meta = make_display_dataframe(
            semantic_df,
            number_formats=TABLEQA_NUMBER_FORMATS,
            number_locales=["en_US"],
        )
        renderers = get_renderers(display_df)
        fmt_name = sample_renderer(renderers)
        render_func = renderers[fmt_name]
        is_scalar = result.shape == (1, 1)
        answer = (
            canonical_scalar(result.iloc[0, 0])
            if is_scalar
            else apply_display_formats(result, display_meta).to_csv(index=False, header=False).strip()
        )
        
        tables = [render_func(index=False)]

        return Entry(
            metadata={
                "table": tables[0],
                "tables": tables,
                "query": q,
                "query_spec": query_spec,
                "result_bucket": self._result_bucket(result),
                "is_scalar": is_scalar,
                "scalar_kind": scalar_kind(result.iloc[0, 0]) if is_scalar else None,
                "table_format": fmt_name,
                **display_meta,
            },
            answer=answer,
        )

    def render_prompt(self, m):
        scalar_formats = {
            "date": "a single date in YYYY-MM-DD format",
            "bool": "a single boolean (`true` or `false`)",
            "null": "the literal NULL",
            "number": "a single number without display formatting",
        }
        fmt = scalar_formats.get(m.get("scalar_kind"), "a single value") if m['is_scalar'] else (
            "CSV format (rows separated by newlines, values by commas). Do not include column headers."
        )
        tables = m.get('tables') or [m['table']]
        if len(tables) == 1:
            preamble = "Execute this SQL query on the table named dataframe:"
        else:
            preamble = "The following tables are row-wise shards of one logical table named dataframe. Concatenate them in order to reconstruct dataframe, then execute the SQL query:"
        presentation = "\n\n".join(f"Table {i}:\n{table}" for i, table in enumerate(tables, 1))
        return (
            f"{preamble}\n\n{presentation}\n\n"
            f"In this table, {SQL_NULL_MARKER} represents SQL NULL.\n\n"
            f"SQL: {m['query']}\n\nThe answer is the result as {fmt}."
        )

    def score_answer(self, ans, entry):
        def isnumeric(x):
            try: float(x); return True
            except: return False
                
        if entry.metadata['is_scalar'] and isnumeric(entry.answer):
            return score_scalar(ans, entry)
        
        # Strip potential header line: if first line matches column names from query, remove it
        def strip_header(s, reference):
            lines = s.strip().splitlines()
            ref_lines = reference.strip().splitlines()
            if len(lines) == len(ref_lines) + 1:
                # First line might be a header — check if remaining lines match
                candidate = "\n".join(lines[1:])
                if candidate.strip():
                    return candidate
            return s
        
        ans = strip_header(ans, entry.answer)
        
        if ans.strip() == entry.answer.strip(): return 1.0
        
        try:
            parse = lambda s: list(csv.reader(io.StringIO(s.strip())))
            a, e = parse(ans), parse(entry.answer)
            
            if len(a) != len(e): return 0.0
            for ar, er in zip(a, e):
                if len(ar) != len(er): return 0.0
                for av, ev in zip(ar, er):
                    try:
                        if abs(float(av) - float(ev)) > 0.01: return 0.0
                    except:
                        # Normalize date formats before comparing
                        av_clean = av.strip().replace("T00:00:00.000", "").replace("T00:00:00", "")
                        ev_clean = ev.strip().replace("T00:00:00.000", "").replace("T00:00:00", "")
                        if av_clean != ev_clean: return 0.0
            return 1.0
        except:
            return 0.0

    def balancing_key(self, problem):
        m = problem.metadata
        s = m.query_spec
        return (
            f"pred={min(s['n_predicates'], 2)}:expr={int(s['has_expression'])}:"
            f"agg={int(s['has_aggregate'])}:group={int(bool(s['n_group_keys']))}:"
            f"order={min(s['n_order_keys'], 2)}:limit={int(s['has_limit'])}:"
            f"distinct={int(s['distinct'])}:rows={min(3, len(problem.answer.splitlines()))}"
        )

EQUIV_RENDERERS = list(TABLE_RENDERER_WEIGHTS)
STAT_RENDERERS = list(TABLE_RENDERER_WEIGHTS)


def permute_table(dataframe):
    def permuted(items):
        out = random.sample(list(items), len(items))
        return out[1:] + out[:1] if len(out) > 1 and out == list(items) else out

    df = dataframe[permuted(dataframe.columns)]
    return df.iloc[permuted(range(len(df)))].reset_index(drop=True)


def generate_equivalence_table(config):
    df = generate_random_table(config).apply(
        lambda column: column.map(
            lambda x: f"{x}_" if isinstance(x, str) and x in {"NULL", "—"} else x
        )
    )
    while len(df.columns) < 2:
        df[f"field_{len(df.columns)}"] = range(len(df))
    first, second = df.columns[:2]
    df = df.rename(columns={first: "event_date", second: "amount"})
    df["event_date"] = [_faker.date_between("-2y", "today") for _ in range(len(df))]
    df["amount"] = np.round(np.random.uniform(1, 2500, len(df)), 2)
    df = df.astype(object)
    duplicate = len(df) > 1 and random.random() < 0.5
    stop = len(df) - duplicate
    df.loc[random.randrange(stop), random.choice(list(df.columns))] = None
    if duplicate:
        df.iloc[-1] = df.iloc[random.randrange(len(df) - 1)]
    return df


def equivalence_display(dataframe, style):
    def display(x):
        kind = scalar_kind(x)
        if kind == "null":
            return "—" if style == "plain" else "NULL"
        if kind == "date":
            return x.strftime("%Y-%m-%d" if style == "plain" else "%b %d, %Y")
        if kind == "number":
            value = Decimal(str(x))
            places = max(2, -value.as_tuple().exponent)
            return canonical_scalar(x) if style == "plain" else f"{value:,.{places}f}"
        return x

    return dataframe.apply(lambda column: column.map(display)).astype(object)


def canonical_table(dataframe):
    def canonical(x):
        kind = scalar_kind(x)
        if kind == "null":
            return kind, ""
        if kind == "date":
            return kind, x.strftime("%Y-%m-%d")
        if kind == "bool":
            return kind, str(bool(x)).lower()
        if kind == "number":
            value = Decimal(str(x))
            return kind, format(Decimal(0) if value == 0 else value.normalize(), "f")
        return "text", str(x)

    df = dataframe.copy()
    df.columns = [str(c) for c in df.columns]
    cols = tuple(sorted(df.columns))
    df = df.reindex(cols, axis=1)
    body = sorted(tuple(canonical(row[c]) for c in cols) for _, row in df.iterrows())
    return cols, tuple(body)


def canonical_table_pair(table_a, table_b, answer):
    """Canonical semantic instance; table sides are interchangeable."""
    pair = sorted((canonical_table(table_a), canonical_table(table_b)), key=repr)
    return tuple(pair), answer


def mutate_cell(x):
    if pd.isna(x):
        return "not missing"
    s = str(x)
    try:
        return str(float(s.replace(",", "")) + random.choice([-1, 1, 10]))
    except Exception:
        pass
    if len(s) >= 3:
        i = random.randrange(len(s))
        choices = [ch for ch in "abcdefghijklmnopqrstuvwxyz" if ch != s[i].lower()]
        return s[:i] + random.choice(choices) + s[i + 1:]
    return s + "_x"


def add_noise_row(df):
    row = {c: mutate_cell(random.choice(df[c].tolist())) if len(df[c]) else "x" for c in df.columns}
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def _corrupt_once(dataframe):
    df = dataframe.copy().astype(object)
    choices = ["cell", "column_name", "drop_row", "add_row", "drop_column", "add_column", "duplicate_row"]
    if len(df) <= 1:
        choices.remove("drop_row")
    if len(df.columns) <= 1:
        choices.remove("drop_column")

    kind = random.choice(choices)
    if kind == "cell":
        r, c = random.randrange(len(df)), random.choice(list(df.columns))
        df.loc[df.index[r], c] = mutate_cell(df.loc[df.index[r], c])
    elif kind == "column_name":
        c = random.choice(list(df.columns))
        name = f"{c}_x"
        while name in df.columns:
            name += "_x"
        df = df.rename(columns={c: name})
    elif kind == "drop_row":
        df = df.drop(df.index[random.randrange(len(df))]).reset_index(drop=True)
    elif kind == "add_row":
        df = add_noise_row(df)
    elif kind == "drop_column":
        df = df.drop(columns=[random.choice(list(df.columns))])
    elif kind == "add_column":
        name = "extra"
        while name in df.columns:
            name += "_x"
        df[name] = [f"x{i}" for i in range(len(df))]
    elif kind == "duplicate_row":
        df = pd.concat([df, df.iloc[[random.randrange(len(df))]]], ignore_index=True)
    return df, kind


def corrupt_table(dataframe, count=None):
    for _ in range(20):
        df, mutations = dataframe.copy(), []
        for _ in range(count or random.randint(2, 3)):
            df, kind = _corrupt_once(df)
            mutations.append(kind)
        if canonical_table(df) != canonical_table(dataframe):
            return df, mutations
    raise RuntimeError("Could not produce a distinct corrupted table")


def corrupt_duplicate_count(dataframe):
    duplicates = dataframe.index[dataframe.duplicated(keep=False)].tolist()
    if duplicates:
        return dataframe.drop(random.choice(duplicates)).reset_index(drop=True), "drop_duplicate"
    row = dataframe.iloc[[random.randrange(len(dataframe))]]
    return pd.concat([dataframe, row], ignore_index=True), "duplicate_row"


# DevTask, not Task: prompts run to a median of 1931 characters (max 3052 over 18 sampled examples,
# levels 0/2/4) against a roster median near 340 -- 5-6x every other task, for a 2-character answer.
# Prompt length that far out competes with the aux budget rather than teaching anything the answer
# needs, so it stays available for development and out of list_tasks()/the roster.
class TableEquivalence(DevTask):
    summary = "Decide if two rendered tables are semantically equivalent under mutations."
    def __init__(self, config=None):
        super().__init__(config=config or TableQAConfig())
        self.balancing_key_ratio = 0.5

    def generate_entry(self):
        semantic_df = generate_equivalence_table(self.config)

        if random.random() < 0.5:
            other_df, answer, corruptions = semantic_df.copy(), "Yes", []
        else:
            if random.random() < 0.3:
                other_df, mutation = corrupt_duplicate_count(semantic_df)
                corruptions = [mutation]
            else:
                other_df, corruptions = corrupt_table(semantic_df)
            answer = "No"

        deduplication_canonical = canonical_table_pair(semantic_df, other_df, answer)
        other_df = permute_table(other_df)
        style_a, style_b = random.sample(["plain", "formatted"], 2)
        display_a = equivalence_display(semantic_df, style_a)
        display_b = equivalence_display(other_df, style_b)
        fmt_a, fmt_b = sample_distinct_renderers(EQUIV_RENDERERS)
        entry = Entry(
            metadata={
                "table_a": get_renderers(display_a)[fmt_a](index=False),
                "table_b": get_renderers(display_b)[fmt_b](index=False),
                "format_a": fmt_a,
                "format_b": fmt_b,
                "mutation": "+".join(corruptions) or "none",
                "transformations": ["row_order", "column_order", "syntax", "numeric", "date", "null"],
                "corruptions": corruptions,
                "duplicate_sensitive": len(corruptions) == 1 and corruptions[0] in {"duplicate_row", "drop_duplicate"},
            },
            answer=answer,
        )
        entry._deduplication_canonical = deduplication_canonical
        return entry

    def render_prompt(self, m):
        return (
            "Do these tables contain the same data?\n"
            "Ignore row order, column order, and table syntax; match values by column name.\n"
            "Treat numeric grouping and trailing zeros as formatting, ISO and English month-name dates as dates, and — and NULL as missing. Repeated rows count.\n\n"
            f"Table A:\n{m['table_a']}\n\n"
            f"Table B:\n{m['table_b']}\n\n"
            "Answer Yes or No."
        )

    def score_answer(self, answer, entry):
        ans = str(answer).strip().lower().strip(".")
        if ans in {"yes", "y", "true", "same"}:
            ans = "Yes"
        elif ans in {"no", "n", "false", "different"}:
            ans = "No"
        return float(ans == entry.answer)

    def balancing_key(self, problem):
        return problem.answer

    def deduplication_key(self, problem):
        return problem._deduplication_canonical


def pearson(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return 0.0 if np.std(a) == 0 or np.std(b) == 0 else float(np.corrcoef(a, b)[0, 1])


def abs_pearson(a, b):
    return abs(pearson(a, b))


def eta_squared(values, labels):
    values, labels = np.asarray(values, dtype=float), np.asarray(labels)
    mean, total = values.mean(), ((values - values.mean()) ** 2).sum()
    if total == 0:
        return 0.0
    return float(sum((labels == g).sum() * (values[labels == g].mean() - mean) ** 2 for g in set(labels)) / total)


def winner_with_margin(scores, need):
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else ordered[0][1]
    return (ordered[0][0], margin) if margin >= need else (None, margin)


def permute_statistics_identifiers(df, spec):
    df, spec = df.copy(), spec.copy()
    if spec["family"] == "row_pearson":
        identifiers = list(df["row_id"])
        mapping = dict(zip(identifiers, random.sample(identifiers, len(identifiers))))
        df["row_id"] = df["row_id"].map(mapping)
    else:
        identifiers = [c for c in df.columns if c not in {"label", "group"}]
        mapping = dict(zip(identifiers, random.sample(identifiers, len(identifiers))))
        df = df.rename(columns=mapping)

    identifiers_pattern = "|".join(re.escape(identifier) for identifier in mapping)
    spec["find"] = re.sub(
        rf"\b(?:{identifiers_pattern})\b",
        lambda match: mapping[match.group()],
        spec["find"],
    )
    spec["answer"] = mapping[spec["answer"]]
    return df, spec


STAT_IDENTIFIERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")


def compact_statistics_identifiers(df, spec):
    """Use short labels for selectable objects so the answer stays one token."""
    df, spec = df.copy(), spec.copy()
    is_row = spec["family"] == "row_pearson"
    identifiers = list(df["row_id"]) if is_row else [
        c for c in df.columns if c not in {"label", "group"}
    ]
    if len(identifiers) > len(STAT_IDENTIFIERS):
        raise RuntimeError("Too many statistical identifiers for compact labels")
    mapping = dict(zip(identifiers, random.sample(STAT_IDENTIFIERS, len(identifiers))))
    if is_row:
        df["row_id"] = df["row_id"].map(mapping)
    else:
        df = df.rename(columns=mapping)
    pattern = "|".join(re.escape(identifier) for identifier in mapping)
    spec["find"] = re.sub(
        rf"\b(?:{pattern})\b", lambda match: mapping[match.group()], spec["find"]
    )
    spec["answer"] = mapping[spec["answer"]]
    return df, spec


def gen_column_pearson(config):
    n, p = max(8, config.num_rows), max(4, config.num_numeric)
    for _ in range(300):
        # A random covariance geometry: the winner is measured, not assigned to a
        # fixed near-copy column.
        latent = np.random.normal(size=(n, random.randint(2, min(5, p))))
        values = latent @ np.random.normal(size=(latent.shape[1], p))
        values += np.random.normal(size=(n, p)) * np.random.uniform(0.3, 1.5, p)
        df = pd.DataFrame(np.round(values, 2), columns=[f"x{i}" for i in range(p)])
        target = random.choice(list(df.columns))
        scores = {c: abs_pearson(df[target], df[c]) for c in df.columns if c != target}
        answer, margin = winner_with_margin(scores, config.margin)
        if answer is not None:
            return df, {
                "find": f"column name most associated with column {target}",
                "metric": "absolute Pearson correlation",
                "answer": answer, "family": "column_pearson", "margin": margin,
            }
    raise RuntimeError("Could not generate column Pearson table with sufficient margin")


def gen_row_pearson(config):
    n, p = max(8, config.num_rows), max(4, config.num_numeric)
    for _ in range(300):
        values = np.random.normal(size=(n, p))
        values += np.random.normal(size=(n, 2)) @ np.random.normal(size=(2, p))
        df = pd.DataFrame(np.round(values, 2), columns=[f"x{i}" for i in range(p)])
        df.insert(0, "row_id", [f"R{i}" for i in range(n)])
        target_i = random.randrange(n)
        target = df.iloc[target_i]
        scores = {
            r.row_id: pearson(target[df.columns[1:]], r[df.columns[1:]])
            for i, r in df.iterrows() if i != target_i
        }
        answer, margin = winner_with_margin(scores, config.margin)
        if answer is not None:
            return df, {
                "find": f"row_id with highest Pearson correlation to row {target.row_id}",
                "metric": "Pearson correlation over numeric columns",
                "answer": answer, "family": "row_pearson", "margin": margin,
            }
    raise RuntimeError("Could not generate row Pearson table with sufficient margin")


def gen_label_eta2(config):
    n, p, k = max(9, config.num_rows), max(4, config.num_numeric), max(3, config.num_categories)
    for _ in range(300):
        labels = np.array([f"L{i % k}" for i in range(n)])
        np.random.shuffle(labels)
        effects = np.random.normal(size=(p, k)) * np.random.uniform(0.0, 2.5, (p, 1))
        noise = np.random.uniform(0.35, 1.8, p)
        label_index = np.array([int(x[1:]) for x in labels])
        data = {"label": labels}
        for i in range(p):
            data[f"x{i}"] = np.round(effects[i, label_index] + np.random.normal(0, noise[i], n), 2)
        df = pd.DataFrame(data)
        scores = {c: eta_squared(df[c], df["label"]) for c in df.columns if c != "label"}
        answer, margin = winner_with_margin(scores, config.margin)
        if answer is not None:
            return df, {
                "find": "numeric column name most associated with column label",
                "metric": "eta squared",
                "answer": answer, "family": "label_eta2", "margin": margin,
            }
    raise RuntimeError("Could not generate eta squared table with sufficient margin")


def gen_categorical_nmi(config):
    if normalized_mutual_info_score is None:
        raise RuntimeError("scikit-learn is required for categorical NMI generation")
    n, p, k = max(9, config.num_rows), max(4, config.num_categories), max(3, config.num_categories)
    for _ in range(300):
        label = np.array([f"L{i % k}" for i in range(n)])
        np.random.shuffle(label)
        data = {"label": label}
        for i, error_rate in enumerate(np.random.uniform(0.05, 0.95, p)):
            values = label.copy()
            mask = np.random.random(n) < error_rate
            values[mask] = np.random.choice([f"L{j}" for j in range(k)], mask.sum())
            data[f"c{i}"] = values
        df = pd.DataFrame(data)
        scores = {c: normalized_mutual_info_score(df["label"], df[c]) for c in df.columns if c != "label"}
        answer, margin = winner_with_margin(scores, config.margin)
        if answer is not None:
            return df, {
                "find": "categorical column name most associated with column label",
                "metric": "normalized mutual information",
                "answer": answer, "family": "categorical_nmi", "margin": margin,
            }
    raise RuntimeError("Could not generate categorical NMI table with sufficient margin")


def partial_pearson(a, b, controls):
    z = np.column_stack([np.ones(len(controls)), np.asarray(controls, dtype=float)])
    residual = lambda x: np.asarray(x, dtype=float).reshape(-1) - z @ np.linalg.lstsq(
        z, np.asarray(x, dtype=float).reshape(-1), rcond=None
    )[0]
    ra, rb = residual(a), residual(b)
    return 0.0 if min(np.std(ra), np.std(rb)) < 1e-12 else pearson(ra, rb)


def gen_partial_pearson(config):
    n, p = max(12, config.num_rows), max(5, config.num_numeric)
    for _ in range(300):
        z, signal = np.random.normal(size=(2, n))
        data = {
            "x0": 1.8 * z + signal + np.random.normal(0, 0.35, n),
            "x1": 2.2 * z + np.random.normal(0, 0.35, n),
            "x2": signal + np.random.normal(0, 0.45, n),
            "x3": z - 0.7 * signal + np.random.normal(0, 0.8, n),
        }
        for i in range(4, p):
            data[f"x{i}"] = np.random.normal(0, 1.2, n)
        df = pd.DataFrame({k: np.round(v, 2) for k, v in data.items()})
        scores = {
            c: abs(partial_pearson(df["x0"], df[c], df[["x1"]]))
            for c in df.columns if c not in {"x0", "x1"}
        }
        answer, margin = winner_with_margin(scores, config.margin)
        if answer is not None:
            return df, {
                "find": "column name most associated with column x0 while controlling for x1",
                "metric": "absolute partial Pearson correlation",
                "answer": answer, "family": "partial_pearson", "margin": margin,
            }
    raise RuntimeError("Could not generate partial Pearson table with sufficient margin")


def gen_pearson_change(config):
    n, p = max(12, config.num_rows), max(5, config.num_numeric)
    for _ in range(300):
        confounder, signal = np.random.normal(size=(2, n))
        data = {
            "x0": 2.5 * confounder + signal + np.random.normal(0, 0.2, n),
            "x1": 2.5 * confounder + np.random.normal(0, 0.2, n),
            "x2": confounder + np.random.normal(0, 0.15, n),
            "x3": confounder + signal + np.random.normal(0, 0.35, n),
            "x4": signal + np.random.normal(0, 1.2, n),
        }
        for i in range(5, p):
            data[f"x{i}"] = np.random.normal(0, 1.2, n)
        df = pd.DataFrame({k: np.round(v, 2) for k, v in data.items()})
        scores = {
            c: abs(partial_pearson(df["x0"], df[c], df[["x1"]]) - pearson(df["x0"], df[c]))
            for c in df.columns if c not in {"x0", "x1"}
        }
        answer, margin = winner_with_margin(scores, config.margin)
        if answer is not None:
            return df, {
                "find": "column name whose correlation with x0 changes most after controlling for x1",
                "metric": "absolute difference between partial and ordinary Pearson correlation",
                "answer": answer, "family": "pearson_change", "margin": margin,
            }
    raise RuntimeError("Could not generate Pearson-change table with sufficient margin")


def _two_groups(n):
    groups = np.array([f"G{i % 2}" for i in range(n)])
    np.random.shuffle(groups)
    return groups


def gen_group_robust_pearson(config):
    n, p = max(12, config.num_rows), max(4, config.num_numeric)
    for _ in range(300):
        groups, target = _two_groups(n), np.random.normal(size=n)
        first = groups == "G0"
        data = {
            "group": groups,
            "x0": target,
            "x1": target + np.random.normal(0, 0.15, n),
            "x2": np.where(first, target + np.random.normal(0, 0.15, n), np.random.normal(size=n)),
            "x3": target + np.random.normal(0, 1.5, n),
        }
        for i in range(4, p):
            data[f"x{i}"] = np.random.normal(size=n)
        df = pd.DataFrame({k: v if k == "group" else np.round(v, 2) for k, v in data.items()})
        scores = {
            c: min(abs_pearson(part["x0"], part[c]) for _, part in df.groupby("group"))
            for c in df.columns if c not in {"group", "x0"}
        }
        answer, margin = winner_with_margin(scores, config.margin)
        if answer is not None:
            return df, {
                "find": "column name with the strongest worst-group association with x0",
                "metric": "minimum absolute Pearson correlation across groups in group",
                "answer": answer, "family": "group_robust_pearson", "margin": margin,
            }
    raise RuntimeError("Could not generate robust grouped-correlation table with sufficient margin")


def gen_group_heterogeneity(config):
    n, p = max(12, config.num_rows), max(4, config.num_numeric)
    for _ in range(300):
        groups, target = _two_groups(n), np.random.normal(size=n)
        sign = np.where(groups == "G0", 1.0, -1.0)
        data = {
            "group": groups,
            "x0": target,
            "x1": sign * target + np.random.normal(0, 0.15, n),
            "x2": target + np.random.normal(0, 0.3, n),
            "x3": np.random.normal(size=n),
        }
        for i in range(4, p):
            data[f"x{i}"] = np.random.normal(size=n)
        df = pd.DataFrame({k: v if k == "group" else np.round(v, 2) for k, v in data.items()})
        scores = {}
        for c in df.columns:
            if c not in {"group", "x0"}:
                correlations = [pearson(part["x0"], part[c]) for _, part in df.groupby("group")]
                scores[c] = max(correlations) - min(correlations)
        answer, margin = winner_with_margin(scores, config.margin)
        if answer is not None:
            return df, {
                "find": "column name whose association with x0 varies most between groups",
                "metric": "range of Pearson correlation across groups in group",
                "answer": answer, "family": "group_heterogeneity", "margin": margin,
            }
    raise RuntimeError("Could not generate heterogeneous grouped-correlation table with sufficient margin")


def standardized_mean_difference(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled_variance = (
        (len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)
    ) / (len(a) + len(b) - 2)
    pooled = np.sqrt(pooled_variance)
    return 0.0 if pooled < 1e-12 else abs(float(a.mean() - b.mean())) / pooled


def gen_distribution_shift(config):
    n, p = max(12, config.num_rows), max(4, config.num_numeric)
    for _ in range(300):
        groups = _two_groups(n)
        effects = np.random.uniform(0, 2.5, p)
        scales = np.random.uniform(0.5, 1.5, p)
        data = {"group": groups}
        for i in range(p):
            data[f"x{i}"] = np.round(
                effects[i] * (groups == "G1") + np.random.normal(0, scales[i], n), 2
            )
        df = pd.DataFrame(data)
        scores = {
            c: standardized_mean_difference(df.loc[groups == "G0", c], df.loc[groups == "G1", c])
            for c in df.columns if c != "group"
        }
        answer, margin = winner_with_margin(scores, config.margin)
        if answer is not None:
            return df, {
                "find": "column name with largest absolute standardized mean difference between G0 and G1",
                "metric": "absolute standardized mean difference",
                "answer": answer, "family": "distribution_shift", "margin": margin,
            }
    raise RuntimeError("Could not generate distribution-shift table with sufficient margin")


STAT_GENERATORS = (
    gen_column_pearson, gen_row_pearson, gen_label_eta2, gen_partial_pearson,
    gen_pearson_change, gen_group_robust_pearson, gen_group_heterogeneity,
    gen_distribution_shift,
)


class TableStatistics(Task):
    summary = "Select rows or columns using associations, conditioning, group robustness, and shifts."
    def __init__(self, config=None):
        super().__init__(config=config or TableStatisticsConfig())
        self.balancing_key_ratio = 0.5

    def generate_entry(self):
        generators = list(STAT_GENERATORS)
        if normalized_mutual_info_score is not None:
            generators.append(gen_categorical_nmi)
        semantic_df, spec = random.choice(generators)(self.config)
        semantic_df, spec = permute_statistics_identifiers(semantic_df, spec)
        semantic_df, spec = compact_statistics_identifiers(semantic_df, spec)
        display_df, display_meta = make_statistics_display_dataframe(semantic_df)
        fmt = sample_renderer(STAT_RENDERERS)
        table = get_renderers(display_df)[fmt](index=False)

        return Entry(
            metadata={
                "table": table,
                "find": spec["find"],
                "metric": spec["metric"],
                "payload": {"table": table, "find": spec["find"], "metric": spec["metric"]},
                "family": spec["family"],
                "margin": spec["margin"],
                "table_format": fmt,
                **display_meta,
            },
            answer=spec["answer"],
        )

    def render_prompt(self, m):
        return (
            f"{render_payload(m.payload)}\n\n"
            "Answer with only the identifier."
        )

    def score_answer(self, answer, entry):
        clean = lambda s: str(s).strip().strip("`'\"").lower()
        return float(clean(answer) == clean(entry.answer))

    def balancing_key(self, problem):
        m = problem.metadata
        return f"{m.family}:format={m.table_format}"
