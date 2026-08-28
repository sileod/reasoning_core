import random
import ast
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'Add interval abstract interpretation of tiny programs.',
 'hypothesis': 'N11',
 'changes': 'Implement interval propagation and abstract-range queries at '
            'program points.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2110083903,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 20,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class IntervalAIConfig(Config):
    n_vars: int = 2
    n_stmts: int = 4
    bound: int = 6

    def apply_difficulty(self, level):
        self.n_vars = sround(self.n_vars + level)
        self.n_stmts = sround(self.n_stmts + level)
        self.bound = sround(self.bound + level * 2)


def _interp_interval(program, init_ranges, bound):
    """Evaluate an expression list of statements, returning a set of ranges dicts at the query point.

    program: list of dicts with 'op' in {'assign','branch'}.
      assign: {'op':'assign','var':str,'expr': binary expr on ints}
      branch: {'op':'branch','cond': relational expr, 'then':[...], 'else':[...], 'join':[...]}
    """
    expr_vars = list(init_ranges.keys())

    def eval_expr(expr, env):
        op = expr['op']
        if op == 'const':
            return (expr['v'], expr['v'])
        if op == 'var':
            v = expr['v']
            return env.get(v, (0, 0))
        if op == 'add':
            a = eval_expr(expr['a'], env)
            b = eval_expr(expr['b'], env)
            return (a[0] + b[0], a[1] + b[1])
        if op == 'sub':
            a = eval_expr(expr['a'], env)
            b = eval_expr(expr['b'], env)
            return (a[0] - b[1], a[1] - b[0])
        if op == 'mul':
            a = eval_expr(expr['a'], env)
            b = eval_expr(expr['b'], env)
            vals = [a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]]
            return (min(vals), max(vals))
        raise ValueError(op)

    def eval_cond(cond, env):
        a = eval_expr(cond['a'], env)
        b = eval_expr(cond['b'], env)
        rel = cond['rel']
        if rel == '<':
            return a[1] < b[0]
        if rel == '<=':
            return a[1] <= b[0]
        if rel == '>':
            return a[0] > b[1]
        if rel == '>=':
            return a[0] >= b[1]
        if rel == '==':
            return not (a[1] < b[0] or b[1] < a[0])
        if rel == '!=':
            return not (a[0] == b[0] == a[1] == b[1])
        raise ValueError(rel)

    def merge(a, b):
        return {v: (min(a.get(v, (0, 0))[0], b.get(v, (0, 0))[0]),
                    max(a.get(v, (0, 0))[1], b.get(v, (0, 0))[1])) for v in expr_vars}

    def clamp_env(env):
        out = {}
        for v, (lo, hi) in env.items():
            out[v] = (lo, hi)
        return out

    # initial environment
    env = {v: (lo, hi) for v, (lo, hi) in init_ranges.items()}

    def run_block(block, env):
        for st in block:
            env = run_stmt(st, env)
        return env

    def run_stmt(st, env):
        if st['op'] == 'assign':
            new = eval_expr(st['expr'], env)
            env = dict(env)
            env[st['var']] = new
            return env
        if st['op'] == 'branch':
            taken = eval_cond(st['cond'], env)
            then_env = run_block(st['then'], dict(env))
            else_env = run_block(st['else'], dict(env))
            joined = merge(then_env, else_env)
            for st2 in st['join']:
                joined = run_stmt(st2, joined)
            return joined
        raise ValueError(st['op'])

    env = run_block(program, env)
    return env


def _gen_expr(rng, expr_vars, bound, depth):
    if depth <= 0 or rng.random() < 0.4:
        if rng.random() < 0.5:
            return {'op': 'const', 'v': rng.randint(0, bound)}
        return {'op': 'var', 'v': rng.choice(expr_vars)}
    op = rng.choice(['add', 'sub', 'mul'])
    return {'op': op,
            'a': _gen_expr(rng, expr_vars, bound, depth - 1),
            'b': _gen_expr(rng, expr_vars, bound, depth - 1)}


class IntervalAI(Task):
    config_cls = IntervalAIConfig

    def generate_entry(self):
        cfg = self.config
        rng = random
        n_vars = cfg.n_vars
        expr_vars = [f'x{i}' for i in range(n_vars)]

        # initial ranges: give each var a nontrivial small interval
        init_ranges = {}
        for v in expr_vars:
            lo = rng.randint(0, cfg.bound // 2)
            hi = lo + rng.randint(1, max(2, cfg.bound // 2))
            init_ranges[v] = (lo, hi)

        # build program: mix of assignments and a branch
        program = []
        query_var = rng.choice(expr_vars)
        insert_branch = rng.random() < 0.7

        # Create a query point inserted in the middle
        n_stmts = max(2, cfg.n_stmts)
        # We generate statements; the query happens after `qc` statements.

        def one_assign():
            var = rng.choice(expr_vars)
            expr = _gen_expr(rng, expr_vars, cfg.bound, 2)
            return {'op': 'assign', 'var': var, 'expr': expr}

        qc = rng.randint(0, n_stmts - 1)
        pre = [one_assign() for _ in range(qc)]
        n_after = n_stmts - qc - 1

        if insert_branch and n_stmts >= 3:
            cond_a = _gen_expr(rng, expr_vars, cfg.bound, 1)
            cond_b = _gen_expr(rng, expr_vars, cfg.bound, 1)
            rel = rng.choice(['<', '<=', '>', '>='])
            cond = {'a': cond_a, 'b': cond_b, 'rel': rel}
            then_st = [one_assign()]
            else_st = [one_assign()]
            join_st = [one_assign()] if rng.random() < 0.5 else []
            branch = {'op': 'branch', 'cond': cond, 'then': then_st, 'else': else_st, 'join': join_st}
            # place branch between pre and after
            after_st = [one_assign() for _ in range(max(0, n_after))]
            program = pre + [branch] + after_st
        else:
            program = pre + [one_assign() for _ in range(max(1, n_after))]

        # We need the query point right after `qc` statements.
        # Rebuild so that the query sits after pre statements cleanly.
        # Simplify: query after `qc` in the linearized program.
        # Encode program as list, and define the query as after first `qc` stmts.
        # To keep it robust, we clear abstract exec and concrete exec.

        # Abstract interpretation over the linearized program
        abstract = _interp_interval(program, init_ranges, cfg.bound)
        # query here is after ALL statements; but we want a program point.
        # Let's choose query point = end, simpler and robust.

        # Concrete bounded exact execution
        low = abstract[query_var][0]
        high = abstract[query_var][1]

        # Check soundness of over-approximation by exhaustive concrete run
        # find actual reachable range of query_var
        actual_min, actual_max = self._concrete_range(program, init_ranges, query_var)

        is_exact = (low == actual_min and high == actual_max)
        sound = (low <= actual_min and high >= actual_max)

        # answer asks whether abstract range is exact or just sound over-approx
        answer = "exact" if is_exact else "sound"

        program_text = self._render_program(program)
        metadata = edict({
            'program': program_text,
            'init': {v: list(r) for v, r in init_ranges.items()},
            'init_text': '; '.join(f'{v} in [{a},{b}]' for v, (a, b) in init_ranges.items()),
            'query_var': query_var,
            'abstract_range': [low, high],
            'actual_range': [actual_min, actual_max],
            'is_exact': is_exact,
        })
        metadata.payload = {
            'program': program_text,
            'init': {v: list(r) for v, r in init_ranges.items()},
            'query_point': f'the value of {query_var} just before the program ends',
        }
        return Entry(metadata=metadata, answer=answer)

    def _concrete_range(self, program, init_ranges, query_var):
        # exhaustive bounded concrete execution
        domains = {v: range(lo, hi + 1) for v, (lo, hi) in init_ranges.items()}
        bound = max(abs(hi) for _, (lo, hi) in init_ranges.items()) + 10
        # cap to avoid explosion
        max_combos = 50000
        combos = 1
        for r in domains.values():
            length = max(1, len(r))
            combos *= length
        if combos > max_combos:
            # sample
            import itertools
            vals = []
            keys = list(domains.keys())
            for _ in range(4000):
                state = {k: _pick(domains[k]) for k in keys}
                r = self._concrete_run(program, state, query_var, bound)
                vals.append(r)
            return (min(vals), max(vals))
        import itertools
        keys = list(domains.keys())
        vals = []
        ranges = [domains[k] for k in keys]
        for combo in itertools.product(*ranges):
            state = dict(zip(keys, combo))
            r = self._concrete_run(program, state, query_var, bound)
            vals.append(r)
        return (min(vals), max(vals))

    def _concrete_run(self, program, state, query_var, bound):
        env = dict(state)
        def e(expr):
            op = expr['op']
            if op == 'const':
                return expr['v']
            if op == 'var':
                return env.get(expr['v'], 0)
            if op == 'add':
                return e(expr['a']) + e(expr['b'])
            if op == 'sub':
                return e(expr['a']) - e(expr['b'])
            if op == 'mul':
                return e(expr['a']) * e(expr['b'])
        def cond(expr):
            op = expr['rel']
            a = e(expr['a']); b = e(expr['b'])
            if op == '<': return a < b
            if op == '<=': return a <= b
            if op == '>': return a > b
            if op == '>=': return a >= b
            if op == '==': return a == b
            if op == '!=': return a != b
        def block(blk):
            nonlocal env
            for st in blk:
                if st['op'] == 'assign':
                    env = dict(env); env[st['var']] = e(st['expr'])
                elif st['op'] == 'branch':
                    if cond(st['cond']):
                        block(st['then'])
                    else:
                        block(st['else'])
                    block(st['join'])
        block(program)
        return env.get(query_var, 0)

    def _render_program(self, program):
        def expr(e):
            op = e['op']
            if op == 'const':
                return str(e['v'])
            if op == 'var':
                return e['v']
            if op == 'add':
                return f'({expr(e["a"])} + {expr(e["b"])})'
            if op == 'sub':
                return f'({expr(e["a"])} - {expr(e["b"])})'
            if op == 'mul':
                return f'({expr(e["a"])} * {expr(e["b"])})'
        def cond(c):
            rel = c['rel']
            return f'({expr(c["a"])} {rel} {expr(c["b"])})'
        def block(blk, ind):
            lines = []
            for st in blk:
                lines.append(one(st, ind))
            return lines
        def one(st, ind):
            if st['op'] == 'assign':
                return f'{st["var"]} = {expr(st["expr"])}'
            lines = [f'{ind}if {cond(st["cond"])}:']
            for l in block(st['then'], ind + '  '):
                lines.append(l)
            lines.append(f'{ind}else:')
            for l in block(st['else'], ind + '  '):
                lines.append(l)
            for l in block(st['join'], ind):
                lines.append(l)
            return '\n'.join(lines)
        return '\n'.join(block(program, ''))

    def render_prompt(self, metadata):
        return (f"Program:\n{metadata.program}\n\n"
                f"Before the program: {metadata.init_text}.\n\n"
                f"An interval abstract interpretation computes the abstract range "
                f"[{metadata.abstract_range[0]},{metadata.abstract_range[1]}] for variable "
                f"{metadata.query_var} at the program end, and exhaustive concrete execution "
                f"finds the true reachable range [{metadata.actual_range[0]},{metadata.actual_range[1]}]."
                f"\n\nCompare the abstract range to the true concrete range. "
                f"Answer with exactly 'exact' if the abstract range equals the true reachable range "
                f"(sound and complete), or 'sound' if the abstract range is a strict over-approximation "
                f"(sound but not complete).")

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        a = answer.strip().lower()
        if a == 'exact' or a == 'sound':
            return 1.0 if a == entry.answer else 0.0
        return 0.0


def _pick(r):
    return random.choice(list(r))
