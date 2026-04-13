import json
import pandas as pd
import duckdb
from faker import Faker
import random
import re
from dataclasses import dataclass
from nltk.metrics.distance import edit_distance
from reasoning_core.template import Task, Problem, Config
from reasoning_core.utils import score_scalar
import csv
import yaml
import io
from rapidfuzz.distance import Levenshtein

@dataclass
class TableQAConfig(Config):
    num_rows: int = 5
    num_columns: int = 2
    num_tables: int = 1
    def update(self, c):
        self.num_rows = int(self.num_rows * (1+c))
        self.num_columns += c
        self.num_tables = min(self.num_tables+c, 2)

_faker = Faker()

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
    cols = random.sample(pool, min(config.num_columns, len(pool)))
    return pd.DataFrame({n: [g() for _ in range(config.num_rows)] for n, g in cols})

def format_float(x):
    s = f"{x:.12f}".rstrip('0').rstrip('.')
    return s if '.' in s else f"{s}.0"


def get_renderers(dataframe):
    return {
        'to_string': lambda index=False: dataframe.to_string(index=index, float_format=format_float),
        'to_markdown': lambda index=False: dataframe.to_markdown(index=index, floatfmt='.12g', disable_numparse=True),
        'to_csv': lambda index=False: dataframe.to_csv(index=index, float_format=format_float),
        'to_html': lambda index=False: dataframe.to_html(index=index, float_format=format_float),
        'to_latex': lambda index=False: dataframe.to_latex(index=index, float_format=format_float),
        'to_json': lambda index=False: dataframe.to_json(orient='records', date_format='iso', indent=4),
        'to_yaml': lambda index=False: yaml.dump(dataframe.to_dict(orient='records'), default_flow_style=False, sort_keys=False),
    }


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


def canonicalize_floats(dataframe):
    dataframe = dataframe.copy()
    for c in dataframe.select_dtypes(include='float').columns:
        dataframe[c] = dataframe[c].map(format_float)
    return dataframe

class TableQA(Task):
    def __init__(self, config=TableQAConfig()):
        super().__init__(config=config)
    
    def _query(self, dataframe):
        if len(dataframe) == 0: return "SELECT COUNT(*) FROM dataframe"

        num = dataframe.select_dtypes('number').columns.tolist()
        cat = dataframe.select_dtypes(exclude='number').columns.tolist()
        order = random.choice(['ASC', 'DESC'])
        esc = lambda s: str(s).replace("'", "''")
        
        queries = []
        if num:
            c = random.choice(num)
            queries += [
                f"SELECT ROUND({random.choice(['SUM', 'AVG', 'MAX', 'MIN'])}({c}), 2) FROM dataframe",
                f"SELECT COUNT(*) FROM dataframe WHERE {c} > {dataframe[c].quantile(random.choice([0.3, 0.5, 0.7]))}",
                f"SELECT * FROM dataframe ORDER BY {c} {order} LIMIT {random.randint(1, 3)}"
            ]
            if len(num) >= 2:
                n1, n2 = random.sample(num, 2)
                queries.append(f"SELECT ROUND(AVG({n1} * {n2}), 2) FROM dataframe")

        if num and cat:
            n, c = random.choice(num), random.choice(cat)
            queries.append(
                f"SELECT {c}, SUM({n}) as v FROM dataframe GROUP BY {c} "
                f"ORDER BY v {order} LIMIT {random.randint(1, 3)}"
            )
            val = esc(dataframe[c].iloc[0])
            queries.append(f"SELECT COUNT(*) FROM dataframe WHERE {c} = '{val}' AND {n} > {dataframe[n].mean()}")

        if cat:
            c = random.choice(cat)
            val = esc(dataframe[c].iloc[random.randint(0, len(dataframe)-1)])
            queries.append(f"SELECT COUNT(DISTINCT {c}) FROM dataframe")
            queries.append(f"SELECT COUNT(*) FROM dataframe WHERE {c} = '{val}'")
            if len(val) > 1:
                queries.append(f"SELECT COUNT(*) FROM dataframe WHERE {c} LIKE '%{val[1:]}%'")
        
        return random.choice(queries) if queries else "SELECT COUNT(*) FROM dataframe"
    
    def generate(self):
        dataframe = generate_random_table(self.config)
        q = self._query(dataframe)
        conn = duckdb.connect()
        result = conn.execute(q).df()
        renderers = get_renderers(dataframe)
        fmt_name = random.choice(list(renderers))
        render_func = renderers[fmt_name]
        is_scalar = result.shape == (1, 1)
        
        tables = [render_func(index=False)]
        if self.config.level > 0 and self.config.num_tables > 1:
            tables = [get_renderers(part)[fmt_name](index=False) for part in split_table(dataframe, self.config.num_tables)]

        return Problem(
            metadata={
                "table": tables[0],
                "tables": tables,
                "query": q,
                "is_scalar": is_scalar,
                "table_format": fmt_name
            },
            answer=result.to_csv(index=False, header=False).strip()
        )

    def prompt(self, m):
        fmt = "single value" if m['is_scalar'] else "CSV format (rows separated by newlines, values by commas). Do not include column headers."
        tables = m.get('tables') or [m['table']]
        if len(tables) == 1:
            preamble = "Execute this SQL query on the table named dataframe:"
        else:
            preamble = "The following tables are row-wise shards of one logical table named dataframe. Concatenate them in order to reconstruct dataframe, then execute the SQL query:"
        presentation = "\n\n".join(f"Table {i}:\n{table}" for i, table in enumerate(tables, 1))
        return f"{preamble}\n\n{presentation}\n\nSQL: {m['query']}\n\nThe answer is the result as {fmt}."

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

class TableConversion(Task):
    def __init__(self, config=TableQAConfig()):
        super().__init__(config=config)

    def generate(self):
        dataframe = canonicalize_floats(generate_random_table(self.config))
        renderers = get_renderers(dataframe)
        src_name, tgt_name = random.sample(list(renderers), 2)
        src_func, tgt_func = renderers[src_name], renderers[tgt_name]
        
        return Problem(
            metadata={
                "source_table": src_func(index=False),
                "source_format": src_name,
                "target_format": tgt_name
            },
            answer=tgt_func(index=False)
        )

    def prompt(self, m):
        return (
            f"Convert the following table from {m['source_format']} to {m['target_format']}.\n\n"
            f"{m['source_table']}\n\n"
            f"Output only the converted table."
        )

    def score_answer(self, answer, entry):
        reference = entry['answer']
        if not answer: return 0.0

        # Semantic pre-check for structured formats
        fmt = entry['metadata'].get('target_format', '')
        try:
            if fmt == 'to_json':
                if json.loads(str(answer).strip()) == json.loads(reference):
                    return 1.0
            elif fmt == 'to_yaml':
                if yaml.safe_load(str(answer).strip()) == yaml.safe_load(reference):
                    return 1.0
            elif fmt == 'to_csv':
                parse = lambda s: list(csv.reader(io.StringIO(s.strip())))
                if parse(str(answer)) == parse(reference):
                    return 1.0
        except Exception:
            pass

        return Levenshtein.normalized_similarity(
            str(answer).strip(), str(reference).strip())