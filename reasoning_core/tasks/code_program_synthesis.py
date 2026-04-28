from dataclasses import dataclass
import multiprocessing as _mp
import random
import re
import cvc5
from cvc5 import Kind
from gramforge import init_grammar, generate
from reasoning_core.template import Task, DevTask, Problem, Config, edict


# --- grammar: python + smt in parallel --------------------------------------

def _build_grammar():
    R = init_grammar(['py', 'smt'], preprocess_template=lambda x: x)

    R('var', 's', 's')
    for lit in ['""', '" "', '"-"', '"_"']:
        R('strlit', lit, lit)
    for n in range(4):
        R('iconst', str(n), str(n))

    # index-safe integers: provably non-negative
    R('idx(iconst)',            '{0}',           '{0}')
    R('idx(sexpr)',             'len({0})',      '(str.len {0})')
    R('idx(idx,idx)',           '({0} + {1})',   '(+ {0} {1})', weight=0.3)

    # general integers
    R('iexpr(idx)',             '{0}',           '{0}')
    R('iexpr(iexpr,iexpr)',     '({0} - {1})',   '(- {0} {1})', weight=0.3)
    R('iexpr(sexpr,sexpr)',     '{0}.find({1})', '(str.indexof {0} {1} 0)', weight=0.25)

    # booleans — every condition must depend on `s`
    R('bexpr(sexpr,strlit)',    '({1} in {0})',  '(str.contains {0} {1})', weight=0.5)
    R('bexpr(strlit,sexpr)',    '({0} == {1})',  '(= {0} {1})', weight=0.3)
    R('bexpr(idx,iconst)',      '(len({0}) < {1})', '(< (str.len {0}) {1})', weight=0.35)  # len-based
    R('bexpr(sexpr,sexpr)',     '({0} == {1})',  '(= {0} {1})', weight=0.15)
    R('bexpr(bexpr)',           '(not {0})',     '(not {0})', weight=0.1)
    # (skipping 'and' for now: compound conditions muddy branching signal and
    # are rarely the minimum anyway)

    # string expressions
    R('sexpr(var)',                 '{0}',                   '{0}')
    R('sexpr(strlit)',              '{0}',                   '{0}')
    R('sexpr(sexpr,sexpr)',         '({0} + {1})',           '(str.++ {0} {1})', weight=0.55)
    R('sexpr(sexpr,idx,idx)',       '{0}[{1}:({1})+({2})]',  '(str.substr {0} {1} {2})', weight=0.35)
    R('sexpr(sexpr,sexpr,sexpr)',   '{0}.replace({1}, {2}, 1)', '(str.replace {0} {1} {2})', weight=0.2)
    R('sexpr(bexpr,sexpr,sexpr)',   '({1} if {0} else {2})', '(ite {0} {1} {2})', weight=0.9)

    R('start(sexpr)',               '{0}',                   '{0}')
    return R


# Note on the 'len({0}) < {1}' production: the first slot is an `idx` nonterminal,
# but we render it by wrapping in len(). This keeps SMT's `(< (str.len s) k)` aligned.
# A cleaner grammar would introduce a separate 'str_for_len' nonterminal; the shortcut
# above works because `idx` already admits `len(sexpr)` forms, so `len(len(s))` etc.
# can appear but is rare and correct.

_GRAMMAR = _build_grammar()


# --- body -> function wrapper -----------------------------------------------



def _expr_to_stmts(expr: str, depth: int) -> list:
    """Indented Python statements equivalent to `return expr`, with top-level
    conditionals rewritten as if/else."""
    pad = "    " * depth
    expr = expr.strip()
    if expr.startswith('(') and _matches_outer(expr):
        expr = expr[1:-1].strip()
    split = _split_top_level_conditional(expr)
    if split is None:
        return [f"{pad}return {expr}"]
    then_expr, cond, else_expr = split
    return (
        [f"{pad}if {cond}:"]
        + _expr_to_stmts(then_expr, depth + 1)
        + [f"{pad}else:"]
        + _expr_to_stmts(else_expr, depth + 1)
    )

def _wrap_as_function(body: str, name: str = "f") -> str:
    body = body.strip()
    stmts = _expr_to_stmts(body, depth=1)
    return f"def {name}(s: str) -> str:\n" + "\n".join(stmts)

def _matches_outer(s: str) -> bool:
    """Check that s[0]='(' matches s[-1]=')' as the outermost pair."""
    if not (s.startswith('(') and s.endswith(')')): return False
    depth, in_str = 0, False
    for i, ch in enumerate(s):
        if ch == '"': in_str = not in_str
        elif not in_str:
            if ch == '(': depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0: return i == len(s) - 1
    return False


def _split_top_level_conditional(s: str):
    """Find top-level 'X if Y else Z' and return (X, Y, Z) or None."""
    depth, in_str, if_pos, else_pos = 0, False, -1, -1
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == '(': depth += 1
            elif ch == ')': depth -= 1
            elif depth == 0:
                if s[i:i+4] == ' if ' and if_pos < 0:
                    if_pos = i
                elif s[i:i+6] == ' else ' and if_pos > 0 and else_pos < 0:
                    else_pos = i
                    break
        i += 1
    if if_pos < 0 or else_pos < 0: return None
    return s[:if_pos].strip(), s[if_pos+4:else_pos].strip(), s[else_pos+6:].strip()


def _reindent(block: str, indent: str) -> str:
    return "\n".join(indent + line if line.strip() else line for line in block.splitlines())


# --- smt -> python ----------------------------------------------------------

def _const_fold(expr):
    for _ in range(4):
        new = re.sub(r'\((\d+)\s*\+\s*(\d+)\)',
                     lambda m: str(int(m.group(1)) + int(m.group(2))), expr)
        new = re.sub(r'\((\d+)\s*-\s*(\d+)\)',
                     lambda m: str(int(m.group(1)) - int(m.group(2))), new)
        if new == expr: break
        expr = new
    return expr

def _smt_to_py(smt: str) -> str:
    s = smt.strip()
    m = re.match(r'\(define-fun\s+\w+\s+\(\([^)]+\)\)\s+\w+\s+(.*)\)\s*$', s, re.DOTALL)
    if m: s = m.group(1).strip()
    m = re.match(r'\(lambda\s+\(\([^)]+\)\)\s+(.*)\)\s*$', s, re.DOTALL)
    if m: s = m.group(1).strip()

    tokens = re.findall(r'\(|\)|"(?:[^"]|"")*"|[^\s()]+', s)
    pos = [0]
    OPS = {
        'str.++':      lambda a, b:    f'({a} + {b})',
        'str.substr':  lambda a, b, c: f'{a}[{b}:({b})+({c})]',
        'str.replace': lambda a, b, c: f'{a}.replace({b}, {c}, 1)',
        'str.len':     lambda a:       f'len({a})',
        'str.indexof': lambda a, b, c=None: f'{a}.find({b})',
        'str.contains':lambda a, b:    f'({b} in {a})',
        'str.prefixof':lambda a, b:    f'{b}.startswith({a})',
        'str.suffixof':lambda a, b:    f'{b}.endswith({a})',
        '+':           lambda a, b:    f'({a} + {b})',
        '-':           lambda a, b:    f'({a} - {b})',
        '<':           lambda a, b:    f'({a} < {b})',
        '=':           lambda a, b:    f'({a} == {b})',
        'and':         lambda *xs:     '(' + ' and '.join(xs) + ')',
        'or':          lambda *xs:     '(' + ' or '.join(xs) + ')',
        'not':         lambda a:       f'(not {a})',
        'ite':         lambda c, a, b: f'({a} if {c} else {b})',
    }
    def parse():
        t = tokens[pos[0]]; pos[0] += 1
        if t == '(':
            head = tokens[pos[0]]; pos[0] += 1
            args = []
            while tokens[pos[0]] != ')':
                args.append(parse())
            pos[0] += 1
            if head in OPS: return OPS[head](*args)
            raise ValueError(f"unknown op: {head}")
        if t.startswith('"'):
            return '"' + t[1:-1].replace('""', '\\"') + '"'
        return t
    return _const_fold(parse())


# --- cvc5 SyGuS -------------------------------------------------------------

def _synth_smallest(io_pairs, size_bound, timeout_ms):
    slv = cvc5.Solver()
    slv.setOption("sygus", "true")
    slv.setOption("sygus-abort-size", str(size_bound))
    slv.setOption("tlimit", str(timeout_ms))
    slv.setLogic("SLIA")

    S, I, B = slv.getStringSort(), slv.getIntegerSort(), slv.getBooleanSort()
    s = slv.mkVar(S, "s")
    ntS = slv.mkVar(S, "S")
    ntIdx = slv.mkVar(I, "Idx")
    ntI = slv.mkVar(I, "I")
    ntB = slv.mkVar(B, "B")
    g = slv.mkGrammar([s], [ntS, ntIdx, ntI, ntB])

    lits = [slv.mkString(x) for x in ("", " ", "-", "_")]
    g.addRules(ntS, [
        s, *lits,
        slv.mkTerm(Kind.STRING_CONCAT, ntS, ntS),
        slv.mkTerm(Kind.STRING_SUBSTR, ntS, ntIdx, ntIdx),
        #slv.mkTerm(Kind.STRING_REPLACE, ntS, ntS, ntS),
        slv.mkTerm(Kind.ITE, ntB, ntS, ntS),
    ])
    g.addRules(ntIdx, [
        slv.mkInteger(0), slv.mkInteger(1), slv.mkInteger(2), slv.mkInteger(3),
        slv.mkTerm(Kind.STRING_LENGTH, ntS),
        slv.mkTerm(Kind.ADD, ntIdx, ntIdx),
    ])
    g.addRules(ntI, [
        ntIdx,
        slv.mkTerm(Kind.SUB, ntI, ntI),
        slv.mkTerm(Kind.STRING_INDEXOF, ntS, ntS, slv.mkInteger(0)),
    ])
    # boolean productions: every condition mentions `s` directly or via len/contains
    g.addRules(ntB, [
        slv.mkTerm(Kind.STRING_CONTAINS, ntS, lits[1]),  # " " in S
        slv.mkTerm(Kind.STRING_CONTAINS, ntS, lits[2]),  # "-" in S
        slv.mkTerm(Kind.STRING_CONTAINS, ntS, lits[3]),  # "_" in S
        slv.mkTerm(Kind.EQUAL, ntS, lits[0]),
        slv.mkTerm(Kind.EQUAL, ntS, lits[1]),
        slv.mkTerm(Kind.EQUAL, ntS, lits[2]),
        slv.mkTerm(Kind.EQUAL, ntS, lits[3]),
        slv.mkTerm(Kind.LT, slv.mkTerm(Kind.STRING_LENGTH, ntS), slv.mkInteger(1)),
        slv.mkTerm(Kind.LT, slv.mkTerm(Kind.STRING_LENGTH, ntS), slv.mkInteger(2)),
        slv.mkTerm(Kind.LT, slv.mkTerm(Kind.STRING_LENGTH, ntS), slv.mkInteger(3)),
        slv.mkTerm(Kind.NOT, ntB),
    ])

    f = slv.synthFun("f", [s], S, g)
    for inp, out in io_pairs:
        slv.addSygusConstraint(slv.mkTerm(
            Kind.EQUAL,
            slv.mkTerm(Kind.APPLY_UF, f, slv.mkString(inp)),
            slv.mkString(out),
        ))
    res = slv.checkSynth()
    return str(slv.getSynthSolution(f)) if res.hasSolution() else None


def _synth_worker_proc(io_pairs, size_bound, timeout_ms, queue):
    try:
        queue.put(_synth_smallest(io_pairs, size_bound, timeout_ms))
    except Exception:
        queue.put(None)


def _synth_with_timeout(io_pairs, size_bound, timeout_ms):
    """Run synthesis in a subprocess so CVC5 can be hard-killed if it ignores its own timeout."""
    ctx = _mp.get_context('fork')
    queue = ctx.Queue()
    proc = ctx.Process(target=_synth_worker_proc,
                       args=(io_pairs, size_bound, timeout_ms, queue),
                       daemon=True)
    proc.start()
    proc.join(timeout=timeout_ms / 1000 + 3)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return None
    return queue.get_nowait() if not queue.empty() else None


# --- filters ----------------------------------------------------------------

_TRIVIAL_BODIES = {'s', '""', '" "', '"-"', '"_"'}

def _is_trivial_behavior(pairs):
    outs = {o for _, o in pairs}
    if len(outs) < 2: return True
    if all(o == i for i, o in pairs): return True
    if all(o == '' for _, o in pairs): return True
    return False

def _is_trivial_answer(py_body):
    return py_body.strip() in _TRIVIAL_BODIES

def _has_meaningful_branching(source_py, inputs):
    """At least one 'X if cond else Y' condition must split the inputs."""
    conditions = re.findall(r'\sif\s+(.+?)\s+else\s', source_py)
    if not conditions: return True  # non-branching source is fine; filter only if ite was used
    env_builtins = {'__builtins__': {'len': len}}
    for cond in conditions:
        results = set()
        for inp in inputs:
            try: results.add(bool(eval(cond, env_builtins, {'s': inp})))
            except Exception: pass
            if len(results) == 2: return True
    return False


# --- task -------------------------------------------------------------------

_RANDOM_POOL = ['', 'a', 'ab', 'abc', 'hello', 'x-y', 'a_b',
                ' ', '--', 'foo bar', 'zz', 'A-B-C', 'a-b_c', '__',
                'ab-cd', 'x_y_z', '-a-', '_ _', 'aa-bb']
_EDGE_CASES = ['', ' ', 'a', '-', '_']


def _run(py_body, s):
    return eval(py_body, {'__builtins__': {'len': len}}, {'s': s})


@dataclass
class ProgramSynthesisCfg(Config):
    depth: int = 6
    size_bound: int = 10
    n_io: int = 6
    n_holdout: int = 5
    timeout_ms: int = 10000
    max_attempts: int = 80

    def update(self, c=1):
        self.depth += c
        self.size_bound += 2 * c
        self.n_io += c // 2
        self.timeout_ms += 2000 * c


class ProgramSynthesis(DevTask):
    def __init__(self, config=ProgramSynthesisCfg()):
        super().__init__(config=config, timeout=120)

    def _sample_inputs(self, k_shown, k_holdout):
        shown_budget = max(0, k_shown - len(_EDGE_CASES))
        remaining = [x for x in _RANDOM_POOL if x not in _EDGE_CASES]
        shown = list(_EDGE_CASES) + random.sample(remaining, min(shown_budget, len(remaining)))
        shown = shown[:k_shown]
        holdout = random.sample([x for x in _RANDOM_POOL if x not in shown],
                                min(k_holdout, len([x for x in _RANDOM_POOL if x not in shown])))
        return shown, holdout

    def _execute_all(self, py_body, inputs):
        pairs = []
        for inp in inputs:
            try: out = _run(py_body, inp)
            except Exception: return None
            if not isinstance(out, str) or len(out) > 32: return None
            pairs.append((inp, out))
        return pairs

    def generate(self) -> Problem:
        cfg = self.config
        for _ in range(cfg.max_attempts):
            tree = generate(_GRAMMAR, depth=cfg.depth, min_depth=max(2, cfg.depth - 1))
            py_body = tree @ 0
            shown_in, holdout_in = self._sample_inputs(cfg.n_io, cfg.n_holdout)

            # Fix B: if the source contains conditionals, at least one of them
            # must actually branch across the shown inputs.
            if 'if' in py_body and not _has_meaningful_branching(py_body, shown_in):
                continue

            shown = self._execute_all(py_body, shown_in)
            if shown is None or _is_trivial_behavior(shown): continue
            holdout = self._execute_all(py_body, holdout_in)
            if holdout is None: continue

            smt_answer = _synth_with_timeout(shown, cfg.size_bound, cfg.timeout_ms)
            if smt_answer is None: continue
            try: py_expr = _smt_to_py(smt_answer)
            except Exception: continue
            if _is_trivial_answer(py_expr): continue

            try:
                if not all(_run(py_expr, i) == o for i, o in shown + holdout):
                    continue
            except Exception:
                continue

            py_function = _wrap_as_function(py_expr, name="f")

            meta = edict(
                io_pairs=shown,
                holdout=holdout,
                source_py=py_body,
                source_smt=tree @ 1,
                answer_smt=smt_answer,
                answer_expr=py_expr,
            )
            return Problem(metadata=meta, answer=py_function)
        raise RuntimeError(f"no non-trivial instance after {cfg.max_attempts} attempts")

    def prompt(self, metadata) -> str:
        examples = "\n".join(f"  f({i!r}) = {o!r}" for i, o in metadata.io_pairs)
        return (
            "Write a Python function `f(s: str) -> str` consistent with these examples:\n"
            f"{examples}\n\n"
            "You may use: the parameter `s`; string literals \"\", \" \", \"-\", \"_\"; "
            "integer constants 0..3; `+` (string concat or int add); `-` (int subtraction); "
            "`len(x)`; slicing `x[a:b]`; `x.replace(a, b, 1)` (replace first occurrence); "
            "`x.find(y)` (first index of y in x, or -1); `y in x`; `if/else` statements or "
            "conditional expressions; and boolean operators `not`, `==`, `<`.\n\n"
            "Return a complete function definition starting with `def f(s: str) -> str:`."
        )

    def score_answer(self, answer, entry) -> float:
        # Try to extract and run the candidate function.
        candidate = answer.strip()
        ns = {'__builtins__': {'len': len, 'str': str, 'int': int, 'range': range}}

        # Grab the `def f(...)` block if present; otherwise treat as an expression body.
        m = re.search(r'(def\s+\w+\s*\([^)]*\)[^:]*:.*?)(?=\n\S|\Z)', candidate, re.DOTALL)
        if m:
            src = m.group(1)
            func_name = re.match(r'def\s+(\w+)', src).group(1)
        else:
            # fallback: maybe just a body expression or 'return EXPR'
            rm = re.search(r'return\s+(.+?)\s*$', candidate, re.DOTALL)
            body_expr = rm.group(1).strip() if rm else candidate
            src = f"def f(s):\n    return {body_expr}"
            func_name = "f"

        try:
            exec(src, ns)
            fn = ns[func_name]
        except Exception as e:
            print("ERRORED:",e)
            print(src)
            return 0.0

        all_pairs = list(entry.metadata.io_pairs) + list(entry.metadata.holdout)
        hits = 0
        for inp, expected in all_pairs:
            try:
                if fn(inp) == expected: hits += 1
            except Exception:
                pass
        return hits / len(all_pairs)