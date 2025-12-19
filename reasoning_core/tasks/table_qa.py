import pandas as pd
import duckdb
from faker import Faker
import random
from dataclasses import dataclass
from reasoning_core.template import Task, Problem, Config
from reasoning_core.utils import score_scalar

@dataclass
class TableQAConfig(Config):
    num_rows: int = 5
    num_columns: int = 2
    def update(self, c):
        self.num_rows *= (1+c)
        self.num_columns += c

class TableQA(Task):
    def __init__(self, config=TableQAConfig()):
        super().__init__(config=config)
    
    def _table(self):
        f = Faker()
        pool = [
            ('customer', f.name), ('city', f.city), ('country', f.country), ('email', f.email),
            ('company', f.company), ('product', lambda: f.word().capitalize()), ('job', f.job),
            ('date', lambda: f.date_between('-1y')), ('qty', lambda: random.randint(1, 1000)),
            ('revenue', lambda: round(random.uniform(10, 1000), 2)),
            ('price', lambda: round(random.uniform(5, 500), 2)),
            ('rating', lambda: round(random.uniform(1, 5), 1))
        ]
        cols = random.sample(pool, min(self.config.num_columns, len(pool)))
        return pd.DataFrame({n: [g() for _ in range(self.config.num_rows)] for n, g in cols})
    
    def _query(self, df):
        num = df.select_dtypes('number').columns.tolist()
        cat = df.select_dtypes(exclude='number').columns.tolist()
        order = random.choice(['ASC', 'DESC'])
        
        queries = []
        if num:
            c = random.choice(num)
            queries += [
                f"SELECT ROUND({random.choice(['SUM', 'AVG', 'MAX', 'MIN'])}({c}), 2) FROM df",
                f"SELECT COUNT(*) FROM df WHERE {c} > {df[c].quantile(random.choice([0.3, 0.5, 0.7]))}",
            ]
        if num and cat:
            n, c = random.choice(num), random.choice(cat)
            queries.append(
                f"SELECT {c} FROM (SELECT {c}, SUM({n}) v FROM df GROUP BY {c}) "
                f"ORDER BY v {order} LIMIT {random.randint(1, 3)}"
            )
        if cat:
            c = random.choice(cat)
            queries.append(f"SELECT COUNT(DISTINCT {c}) FROM df")
            v = df[c].iloc[random.randint(0, len(df)-1)]
            queries.append(f"SELECT COUNT(*) FROM df WHERE {c} = '{v}'")
        
        return random.choice(queries) if queries else "SELECT COUNT(*) FROM df"
    
    def generate(self):
        df = self._table()
        q = self._query(df)
        result = result = duckdb.sql(q).df()
        
        render = random.choice([df.to_string, df.to_markdown, df.to_csv, df.to_html, df.to_latex])
        is_scalar = result.shape == (1, 1)
        
        return Problem(
            metadata={
                "table": render(index=False), 
                "query": q,
                "is_scalar": is_scalar,
                "table_format": render.__name__
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
            return score_scalar(ans, entry) # provides a scalar reward between 0 and 1  based on two floats (answer is interpreted with "eval")
        
        if ans.strip() == entry.answer.strip(): return 1.0
        try:
            parse = lambda s: [[v.strip() for v in r.split(',')] for r in s.strip().split('\n')]
            a, e = parse(ans), parse(entry.answer)
            if len(a) != len(e): return 0.0
            for ar, er in zip(a, e):
                if len(ar) != len(er): return 0.0
                for av, ev in zip(ar, er):
                    try:
                        if abs(float(av) - float(ev)) > 0.01: return 0.0
                    except:
                        if av != ev: return 0.0
            return 1.0
        except:
            return 0.0
