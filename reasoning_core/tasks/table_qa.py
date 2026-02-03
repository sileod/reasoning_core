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
import io
from rapidfuzz.distance import Levenshtein

@dataclass
class TableQAConfig(Config):
    num_rows: int = 5
    num_columns: int = 2
    def update(self, c):
        self.num_rows = int(self.num_rows * (1+c))
        self.num_columns += c

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

def get_renderers(dataframe):
    return [
        (dataframe.to_string, 'to_string'),
        (dataframe.to_markdown, 'to_markdown'),
        (dataframe.to_csv, 'to_csv'),
        (dataframe.to_html, 'to_html'),
        (dataframe.to_latex, 'to_latex'),
        (lambda index=False: dataframe.to_json(orient='records', date_format='iso', indent=4), 'to_json')
    ]

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
        conn = duckdb.connect()  # fresh in-memory connection
        result = conn.execute(q).df()
        render_func, fmt_name = random.choice(get_renderers(dataframe))
        is_scalar = result.shape == (1, 1)
        
        return Problem(
            metadata={
                "table": render_func(index=False), 
                "query": q,
                "is_scalar": is_scalar,
                "table_format": fmt_name
            },
            answer=result.to_csv(index=False, header=False).strip()
        )
    
    def prompt(self, m):
        fmt = "single value" if m['is_scalar'] else "CSV format (rows separated by newlines, values by commas)"
        return f"Execute this SQL query on the table:\n\n{m['table']}\n\nSQL: {m['query']}\n\nReturn result as {fmt}."
    
    def score_answer(self, ans, entry):
        def isnumeric(x):
            try: float(x); return True
            except: return False
                
        if entry.metadata['is_scalar'] and isnumeric(entry.answer):
            return score_scalar(ans, entry)
        
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
                        if av.strip() != ev.strip(): return 0.0
            return 1.0
        except:
            return 0.0

class TableConversion(Task):
    def __init__(self, config=TableQAConfig()):
        super().__init__(config=config)

    def generate(self):
        dataframe = generate_random_table(self.config)
        renderers = get_renderers(dataframe)
        
        (src_func, src_name), (tgt_func, tgt_name) = random.sample(renderers, 2)
        
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
        
        # normalized_similarity returns 0.0 - 1.0 (float)
        return Levenshtein.normalized_similarity(str(answer).strip(), str(reference).strip())