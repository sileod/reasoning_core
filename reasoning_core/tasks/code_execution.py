import ast, io, re, sys, os, time, random, signal, inspect, contextlib, multiprocessing as mp
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import product

from reasoning_core.template import Task, DevTask, Entry, Config, edict, stochastic_rounding as sround
from gramforge import generate
from gramforge.grammars import mesopy_grammar

@dataclass
class MesopyCodeCfg(Config):
    difficulty: float = 0.0
    min_depth: int = 4
    max_depth: int = 15
    max_attempts: int = 180
    timeout: float = 0.5
    magnitude: int = 3
    n_functions: int = 2
    max_answer_len: int = 80
    max_steps: int = 10_000
    min_steps: int = 4
    shortcut_prob: float = 0.08
    trivial_accept_prob: float = 0.05
    trivial_probes: int = 3

    def apply_difficulty(self, level):
        self.difficulty += level
        self.min_depth = sround(self.min_depth + 0.5 * level)
        self.max_depth = sround(self.max_depth + level)
        self.n_functions = sround(self.n_functions + 0.5 * level)
        self.magnitude = sround(self.magnitude + 0.5 * level)
        self.min_steps = sround(self.min_steps + 2 * level)


@dataclass
class RunReport:
    ok: bool = False
    value: str | None = None
    error: str | None = None
    args: list | None = None
    stdout: str = ""
    stderr: str = ""
    steps: int = 0
    elapsed: float = 0.0


class CapIO(io.StringIO):
    def __init__(self, cap=2000):
        super().__init__()
        self.cap = cap

    def write(self, s):
        if self.tell() < self.cap:
            super().write(s[: self.cap - self.tell()])
        return len(s)


class StepLimit(BaseException):
    pass


def fake(t, magnitude=3):
    n = max(1, int(magnitude))
    return {
        int: lambda: random.randint(-n, n),
        str: lambda: "".join(random.choices("abcxyz", k=random.randint(0, n))),
        list: lambda: [random.randint(-n, n) for _ in range(random.randint(0, n))],
    }.get(t, lambda: None)()


def endpoint_args(fn, magnitude=3):
    return [fake(p.annotation, magnitude) for p in inspect.signature(fn).parameters.values()]


def call_src(args):
    return f"endpoint({', '.join(map(repr, args))})"


def soft_score(answer, reference):
    norm = lambda x: re.sub(r"\s+", " ", str(x).strip())
    a, b = norm(answer), norm(reference)
    return 1.0 if a == b else SequenceMatcher(None, a, b).ratio()


def value_kind(x):
    return "list" if x.startswith("[") else "str" if x.startswith(("'", '"')) else "int"


def accept_steps(steps, cfg):
    if steps >= cfg.min_steps:
        return True
    if cfg.min_steps <= 0:
        return True
    return random.random() < cfg.shortcut_prob * ((steps + 1) / cfg.min_steps) ** 2


def function_triviality(reports):
    good = [r for r in reports if r.ok and r.args is not None]
    if len(good) < 2:
        return None
    if len({r.value for r in good}) == 1:
        return "constant"
    for i in range(min(map(lambda r: len(r.args), good), default=0)):
        if all(r.value == repr(r.args[i]) for r in good):
            return "identity"
    return None


def kill(p):
    if p.is_alive():
        p.terminate()
        p.join(0.05)
        if p.is_alive():
            os.kill(p.pid, signal.SIGKILL)
            p.join()


def _worker(send, code, magnitude, recursionlimit, max_steps, call_args=None, batch=False, reports=False):
    out, err = CapIO(), CapIO()
    ns = {"__builtins__": __builtins__}
    steps, t0 = 0, time.perf_counter()
    sys.setrecursionlimit(recursionlimit)

    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    except Exception:
        pass

    def trace(frame, event, arg):
        nonlocal steps
        if event == "line":
            steps += 1
            if steps > max_steps:
                raise StepLimit
        return trace

    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exec(compile(code, "<mesopy>", "exec"), ns, ns)
            args = call_args if call_args is not None else endpoint_args(ns["endpoint"], magnitude)
            args_list = args if batch else None
            values = []
            for a in args_list or [args]:
                steps = 0
                if not reports:
                    sys.settrace(trace)
                try:
                    value = repr(ns["endpoint"](*a))[:500]
                    values.append(RunReport(ok=True, value=value, args=a) if reports else value)
                except StepLimit:
                    if not reports:
                        raise
                    values.append(RunReport(error="TimeoutError", args=a, steps=steps))
                except Exception as e:
                    if not reports:
                        raise
                    values.append(RunReport(error=type(e).__name__, args=a, steps=steps))
                finally:
                    sys.settrace(None)

        r = values if reports else RunReport(
            True,
            values if args_list else values[0],
            None,
            args,
            out.getvalue(),
            err.getvalue(),
            steps,
            time.perf_counter() - t0,
        )

    except StepLimit:
        sys.settrace(None)
        r = RunReport(
            False,
            None,
            "TimeoutError",
            locals().get("args"),
            out.getvalue(),
            err.getvalue(),
            steps,
            time.perf_counter() - t0,
        )

    except Exception as e:
        sys.settrace(None)
        r = RunReport(
            False,
            None,
            type(e).__name__,
            locals().get("args"),
            out.getvalue(),
            err.getvalue(),
            steps,
            time.perf_counter() - t0,
        )

    try:
        send.send(r)
    except Exception:
        pass
    send.close()


def run_code(code, cfg, recursionlimit=80, call_args=None, batch=False, reports=False):
    ctx = mp.get_context("fork")
    recv, send = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_worker, args=(send, code, cfg.magnitude, recursionlimit, cfg.max_steps, call_args, batch, reports))
    p.start()
    send.close()
    timeout = min(4.0, cfg.timeout + 0.01 * len(call_args)) if batch else cfg.timeout

    try:
        if recv.poll(timeout):
            r = recv.recv()
            p.join(0.05)
            kill(p)
            return r

        kill(p)
        return [] if reports else RunReport(error="TimeoutError", args=call_args, elapsed=timeout)

    except KeyboardInterrupt:
        kill(p)
        raise

    finally:
        recv.close()


def make_code(cfg, failure_rate, profile="full"):
    fast = profile == "runnability"
    g = mesopy_grammar(
        mode="function",
        n_functions=max(1, int(cfg.n_functions)),
        max_number=max(4, int(8 + 2 * cfg.difficulty)),
        max_params=2 + int(cfg.difficulty >= 4),
        param_types=("int", "str", "list"),
        return_types=("int", "str", "list"),
        emit_endpoint=True,
        emit_result=False,
        failure_rate=failure_rate,
        triviality_rate=max(0.05, 0.5 - 0.05 * cfg.difficulty),
        allow_recursion=not fast,
        include_loops=not fast,
        include_try_except=not fast,
        include_comprehensions=not fast,
        include_fstrings=not fast,
        include_break_continue=not fast,
        min_body_stmts=2 if fast else 1,
    )
    return generate(g, depth=cfg.max_depth, min_depth=cfg.min_depth) @ "py"


def endpoint_probes(code, cfg, limit=24):
    try:
        tree = ast.parse(code)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "endpoint")
    except (SyntaxError, StopIteration):
        return []
    n = max(3, int(cfg.magnitude))
    values = {
        "int": list(range(-n, n + 1)),
        "str": ["", "a", "ab", "xyz", "0"],
        "list": [[], [0], [1], [-1, 1], [0, 2, -2]],
    }
    pools = [values.get(getattr(arg.annotation, "id", None), [None]) for arg in fn.args.args]
    probes = [list(xs) for xs in product(*pools)]
    random.shuffle(probes)
    return probes[:limit]


def organic_mutations(code):
    """Small mutations of Mesopy statements; callers still determine the label."""
    tree, lines = ast.parse(code), code.splitlines()
    arities = {
        n.name: len(n.args.args) for n in tree.body if isinstance(n, ast.FunctionDef)
    }
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    def ancestor(node, kind):
        while node in parents:
            node = parents[node]
            if isinstance(node, kind):
                return node
        return None

    scopes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != "endpoint"]
    all_locals = {
        n.name: {a.arg for a in n.args.args} | {
            x.id for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)
        }
        for n in scopes
    }
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in arities:
            arity = arities[node.func.id]
            if len(node.args) == arity and ancestor(node, ast.If):
                args = node.args[:-1] if node.args else [ast.Constant(0)]
                replacement = f"{node.func.id}({', '.join(map(ast.unparse, args))})"
                edits.append((node.lineno - 1, node.col_offset, node.end_col_offset, replacement, "arity"))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            fn = ancestor(node, ast.FunctionDef)
            used = {x.id for x in ast.walk(node.left) if isinstance(x, ast.Name)}
            names = list(all_locals.get(getattr(fn, "name", ""), ()) - used)
            if names:
                edits.append((node.right.lineno - 1, node.right.col_offset, node.right.end_col_offset, random.choice(names), "denominator"))
        elif (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult))
            and any(isinstance(x, ast.Name) for x in ast.walk(node.right))
            and not (
                {x.id for x in ast.walk(node.left) if isinstance(x, ast.Name)}
                & {x.id for x in ast.walk(node.right) if isinstance(x, ast.Name)}
            )
        ):
            between = lines[node.lineno - 1][node.left.end_col_offset:node.right.col_offset]
            match = re.search(r"[+*-]", between)
            if match:
                start = node.left.end_col_offset + match.start()
                edits.append((node.lineno - 1, start, start + 1, "//", "operator"))

    for fn in scopes:
        foreign = set().union(*(v for k, v in all_locals.items() if k != fn.name)) - all_locals[fn.name]
        param_types = {a.arg: getattr(a.annotation, "id", None) for a in fn.args.args}
        ints = [name for name, kind in param_types.items() if kind == "int"]
        lists = [name for name, kind in param_types.items() if kind == "list"]
        strings = [name for name, kind in param_types.items() if kind == "str"]
        seqs = lists + strings
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "len"
                and ints and seqs
            ):
                seq, index = random.choice(seqs), random.choice(ints)
                edits.append((node.lineno - 1, node.col_offset, node.end_col_offset, f"{seq}[{index}]", "index"))
                if lists:
                    edits.append((node.lineno - 1, node.col_offset, node.end_col_offset, f"{random.choice(lists)}.index({index})", "lookup"))
                if len(strings) >= 2:
                    left, right = random.sample(strings, 2)
                    edits.append((node.lineno - 1, node.col_offset, node.end_col_offset, f"{left}.index({right})", "lookup"))
            if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and ancestor(node, ast.If)):
                continue
            if node.id in all_locals[fn.name] and foreign:
                edits.append((node.lineno - 1, node.col_offset, node.end_col_offset, random.choice(tuple(foreign)), "name"))
            alternatives = [
                name for name, kind in param_types.items()
                if kind != param_types.get(node.id) and kind is not None
            ]
            if alternatives:
                edits.append((node.lineno - 1, node.col_offset, node.end_col_offset, random.choice(alternatives), "type"))

    random.shuffle(edits)
    for line_no, start, end, replacement, kind in edits:
        candidate = lines[:]
        candidate[line_no] = candidate[line_no][:start] + replacement + candidate[line_no][end:]
        candidate = "\n".join(candidate) + "\n"
        try:
            compile(candidate, "<mesopy>", "exec")
        except SyntaxError:
            continue
        yield kind, candidate


def runnability_pair(cfg):
    """Find two calls with different outcomes in an organically generated program."""
    for _ in range(max(1, cfg.max_attempts)):
        base = make_code(cfg, failure_rate=random.uniform(0.05, 0.35), profile="runnability")
        probes = endpoint_probes(base, cfg)
        if len(probes) < 2:
            continue
        mutations = list(organic_mutations(base))
        first_by_kind = {kind: candidate for kind, candidate in mutations}
        perturbed = list(first_by_kind.items()) + mutations[:4]
        random.shuffle(perturbed)
        perturbed.sort(key=lambda item: item[0] in {"operator", "denominator"})
        candidates = [("organic", base)] + perturbed
        for kind, code in candidates:
            reports = run_code(code, cfg, call_args=probes, batch=True, reports=True)
            good = next((r for r in reports if r.ok), None)
            bad = next((r for r in reports if r.error), None)
            if good and bad:
                verified = [run_code(code, cfg, call_args=r.args) for r in (bad, good)]
                if (
                    verified[0].error == bad.error
                    and verified[1].ok
                    and min(r.steps for r in verified) >= min(6, 4 + int(cfg.difficulty) // 3)
                ):
                    return kind, ((code, verified[0]), (code, verified[1]))
    raise RuntimeError("Failed to generate a mixed-outcome Mesopy program")


def meta(code, r):
    return edict(
        code=code,
        args=r.args,
        call=call_src(r.args),
        steps=r.steps,
        elapsed=r.elapsed,
        stdout=r.stdout,
        stderr=r.stderr,
        triviality=getattr(r, "triviality", None),
    )


def sample_problem(cfg, want_error, failure_rate, profile="full"):
    for _ in range(cfg.max_attempts):
        code = make_code(cfg, failure_rate, profile)
        if "def endpoint" not in code:
            continue

        r = run_code(code, cfg)

        if want_error:
            if r.error and r.args is not None and r.error != "TimeoutError":
                return code, r

        elif r.ok and r.value is not None and len(r.value) <= cfg.max_answer_len:
            if profile == "runnability":
                triviality = None
            else:
                probes = [r] + [run_code(code, cfg) for _ in range(cfg.trivial_probes - 1)]
                triviality = function_triviality(probes)
            if (
                accept_steps(r.steps, cfg)
                and (not triviality or random.random() < cfg.trivial_accept_prob)
            ):
                r.triviality = triviality
                return code, r

    raise RuntimeError(f"Failed to generate valid task. Config: {cfg}")


class CodeRunnability(Task):
    summary = "Predict if a given Python code snippet runs successfully or raises an exception."
    def __init__(self, config=None):
        super().__init__(config=config or MesopyCodeCfg())

    def generate_entry(self, _case=None, _paired_cases=None):
        if _case is None:
            family, pair = runnability_pair(self.config)
            pair = list(pair)
            random.shuffle(pair)
            _case = family, pair.pop()
            if _paired_cases is not None:
                _paired_cases.append((family, pair.pop()))
        family, (code, r) = _case
        metadata = meta(code, r)
        metadata.family = family
        return Entry(metadata=metadata, answer=r.error or "OK")

    def generate_examples(self, **kwargs):
        paired_cases = []
        first = self.generate_example(_paired_cases=paired_cases, **kwargs)
        second = self.generate_example(_case=paired_cases.pop(), **kwargs)
        return [first, second]

    def generate_balanced_batch(self, batch_size=32, **kwargs):
        if batch_size % 2:
            raise ValueError("CodeRunnability requires an even batch_size for atomic pairs")
        return super().generate_balanced_batch(batch_size=batch_size, **kwargs)

    def render_prompt(self, metadata):
        return (
            "Predict whether this Python call runs successfully or raises an exception.\n"
            f"```python\n{metadata.code}\n```\n"
            f"Call: `{metadata.call}`\n"
            "The answer is `OK` if it runs successfully; otherwise the exception class name."
        )

    def score_answer(self, answer, entry):
        reference = entry["answer"] if isinstance(entry, dict) else entry.answer
        return float(str(answer).strip() == str(reference).strip())

    def balancing_key(self, problem):
        return None  # Pairing already guarantees exact OK/error balance.


class CodeExecution(Task):
    summary = "Predict the return value or stdout of executing generated Python code blocks."
    def __init__(self, config=None):
        super().__init__(config=config or MesopyCodeCfg())

    def generate_entry(self):
        code, r = sample_problem(self.config, want_error=False, failure_rate=0.05)
        return Entry(metadata=meta(code, r), answer=r.value)

    def render_prompt(self, metadata):
        return (
            "Predict the value returned by this Python call.\n"
            f"```python\n{metadata.code}\n```\n"
            f"Call: `{metadata.call}`\n"
            "The answer is the exact Python `repr` of the returned value."
        )

    def score_answer(self, answer, entry):
        reference = entry["answer"] if isinstance(entry, dict) else entry.answer
        return soft_score(answer, reference)

    def balancing_key(self, problem):
        return value_kind(problem.answer)


@dataclass
class CodeInputDeductionCfg(MesopyCodeCfg):
    min_depth: int = 3
    max_depth: int = 8
    n_functions: int = 1
    lo: int = -6
    hi: int = 9
    max_len: int = 3
    alphabet: str = "abc"
    max_attempts: int = 100

    def apply_difficulty(self, level):
        self.lo = sround(self.lo - level)
        self.hi = sround(self.hi + level)
        self.max_len = sround(self.max_len + 0.5 * level)


def bounded_strings(alphabet, max_len):
    return [
        "".join(xs)
        for n in range(1, max_len + 1)
        for xs in product(alphabet, repeat=n)
    ]


class CodeInputDeduction(DevTask):
    summary = "Deduce the Python function input that yields a target output value or condition."
    def __init__(self, config=None):
        super().__init__(config=config or CodeInputDeductionCfg())
        self.balancing_key_ratio = 1 / 3

    def generate_entry(self):
        cfg = self.config
        modes = ["int", "tuple", "str"]
        random.shuffle(modes)
        for mode in modes:
            for _ in range(max(1, cfg.max_attempts // len(modes))):
                if mode == "int":
                    domain = list(range(cfg.lo, cfg.hi + 1))
                    sig, call = (("int",), "int"), lambda x: [x]
                    goal, call_text, answer_hint = f"smallest integer x in [{cfg.lo}, {cfg.hi}]", "endpoint(x)", "Answer with the integer."
                    endpoint = f"def endpoint(x):\n    return f0(x) % {random.choice((3, 4, 5))}\n"
                elif mode == "tuple":
                    domain = [(x, y) for x in range(cfg.lo, cfg.hi + 1) for y in range(cfg.lo, cfg.hi + 1)]
                    sig, call = (("int", "int"), "int"), lambda xy: list(xy)
                    goal, call_text, answer_hint = f"lexicographically smallest integer pair (x, y) with each value in [{cfg.lo}, {cfg.hi}]", "endpoint(x, y)", "Answer as `x y`."
                    endpoint = f"def endpoint(x, y):\n    return f0(x, y) % {random.choice((3, 4, 5))}\n"
                else:
                    domain = bounded_strings(cfg.alphabet, cfg.max_len)
                    sig, call = (("int",), "int"), lambda s: [sum((len(cfg.alphabet) ** i) * cfg.alphabet.index(ch) for i, ch in enumerate(reversed(s)))]
                    goal, call_text, answer_hint = f"lexicographically smallest string s over `{cfg.alphabet}` with length 1..{cfg.max_len}", "endpoint(s)", "Answer with the string."
                    endpoint = f"def endpoint(s):\n    z = 0\n    for ch in s:\n        z = {len(cfg.alphabet)} * z + {repr(cfg.alphabet)}.index(ch)\n    return f0(z) % {random.choice((3, 4, 5))}\n"
                core = generate(
                    mesopy_grammar(
                        mode="function",
                        n_functions=max(1, int(cfg.n_functions)),
                        main_signature=sig,
                        max_number=max(4, int(8 + 2 * cfg.difficulty)),
                        max_params=len(sig[0]),
                        param_types=("int",),
                        return_types=("int",),
                        emit_endpoint=False,
                        emit_result=False,
                        failure_rate=0.05,
                        triviality_rate=0.25,
                        include_print=False,
                        include_assert=False,
                        include_try_except=False,
                    ),
                    depth=cfg.max_depth,
                    min_depth=cfg.min_depth,
                ) @ "py"
                code = f"{core}\n\n{endpoint}"
                call_args = [call(x) for x in domain]
                r = run_code(code, cfg, call_args=call_args, batch=True)
                reports = [
                    RunReport(True, value, None, args, r.stdout, r.stderr, r.steps, r.elapsed)
                    for args, value in zip(call_args, r.value or [])
                ] if r.ok else [r]
                buckets = {}
                for x, r in zip(domain, reports):
                    if r.ok and r.value is not None:
                        buckets.setdefault(r.value, []).append(x)
                if function_triviality(reports):
                    continue
                choices = [(y, min(xs)) for y, xs in buckets.items() if 1 < len(xs) < len(domain)]
                choices = [c for c in choices if c[1] != domain[0]] or choices
                if choices:
                    target, answer = random.choice(choices)
                    if isinstance(answer, tuple):
                        answer = " ".join(map(str, answer))
                    else:
                        answer = str(answer)
                    return Entry(
                        edict(code=code, mode=mode, goal=goal, call_text=call_text, answer_hint=answer_hint, target=target),
                        answer,
                    )
        raise RuntimeError("failed to generate code input deduction task")

    def render_prompt(self, m):
        return (
            f"Find the {m.goal} such that `{m.call_text} == target`.\n"
            f"{m.answer_hint}\n\n"
            f"```python\n{m.code}\n```\n\n"
            f"Target: {m.target}"
        )

    def score_answer(self, answer, entry):
        reference = entry["answer"] if isinstance(entry, dict) else entry.answer
        return float(str(answer).strip().strip("\"'") == reference)


# ============================================================================
# Code Consolidation
# ----------------------------------------------------------------------------
# Extracts the minimal source program from a merge of several mesopy-generated
# programs, after stripping intra-program dead code and (optionally) entangling
# one dead-program function into the call chain of the correct answer.
#
# Ported onto the same mesopy_grammar / generate() / fork-based runner used by
# the tasks above (instead of the old pygram_grammar + Fuzzer + run_sandboxed
# stack), so it generates and executes at comparable speed and scales the
# same way under task_diagnostics / parallel workers.
# ============================================================================

import ast, io, re, sys, os, time, random, signal, inspect, contextlib, multiprocessing as mp
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import product

from reasoning_core.template import Task, DevTask, Entry, Config, edict, stochastic_rounding as sround
from gramforge import generate
from gramforge.grammars import mesopy_grammar
import copy


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _too_many_pass(code, threshold=0.25):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    stmts = [n for n in ast.walk(tree) if isinstance(n, ast.stmt)]
    passes = [n for n in stmts if isinstance(n, ast.Pass)]
    return bool(stmts) and len(passes) / len(stmts) > threshold


def _line_count(code):
    return len(code.strip().splitlines())


def _get_called_functions(node):
    return {
        n.func.id
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }


def _safe_parse(code):
    try:
        return ast.parse(code)
    except Exception:
        return None


def _func_defs(tree):
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _prune_to_reachable(code):
    """Drop function defs not reachable from any top-level exec statement."""
    tree = _safe_parse(code)
    if tree is None:
        return None

    func_defs = _func_defs(tree)
    exec_stmts = [n for n in tree.body if not isinstance(n, ast.FunctionDef)]

    reachable, frontier = set(), set()
    for stmt in exec_stmts:
        frontier |= _get_called_functions(stmt)

    while frontier:
        name = frontier.pop()
        if name in reachable or name not in func_defs:
            continue
        reachable.add(name)
        frontier |= _get_called_functions(func_defs[name])

    tree.body = [n for n in tree.body if not isinstance(n, ast.FunctionDef) or n.name in reachable]
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:
        return None


def _call_chain_depth(code):
    """Max transitive call-chain depth reachable from a top-level exec statement."""
    tree = _safe_parse(code)
    if tree is None:
        return 0

    func_defs = _func_defs(tree)
    exec_stmts = [n for n in tree.body if not isinstance(n, ast.FunctionDef)]
    memo = {}

    def _depth(name, visiting):
        key = (name, visiting)
        if key in memo:
            return memo[key]
        if name not in func_defs or name in visiting:
            return 0
        called = _get_called_functions(func_defs[name]) & func_defs.keys()
        d = 1 + max((_depth(c, visiting | {name}) for c in called), default=0)
        memo[key] = d
        return d

    top_calls = set()
    for stmt in exec_stmts:
        top_calls |= _get_called_functions(stmt)
    top_calls &= func_defs.keys()

    return max((_depth(c, frozenset()) for c in top_calls), default=0)


def _func_call_signature(code):
    """Shape-only fingerprint of a program's call graph."""
    tree = _safe_parse(code)
    if tree is None:
        return ""

    func_defs = _func_defs(tree)
    exec_stmts = [n for n in tree.body if not isinstance(n, ast.FunctionDef)]
    memo = {}

    def sig(name, visiting):
        key = (name, visiting)
        if key in memo:
            return memo[key]
        if name not in func_defs or name in visiting:
            return "leaf"
        called = sorted(_get_called_functions(func_defs[name]) & func_defs.keys())
        s = f"node({','.join(sig(c, visiting | {name}) for c in called)})" if called else "leaf"
        memo[key] = s
        return s

    top_calls = sorted(set().union(*[_get_called_functions(s) for s in exec_stmts]) & func_defs.keys()) if exec_stmts else []
    return f"root({','.join(sig(c, frozenset()) for c in top_calls)})"


class _ReturnWrapper(ast.NodeTransformer):
    """Rewrites `return <expr>` to `return dead_fn(param=<expr>)`."""

    def __init__(self, dead_fn_name, param_name):
        self.dead_fn_name = dead_fn_name
        self.param_name = param_name
        self.modified = False

    def visit_Return(self, node):
        if node.value is None:
            return node
        self.modified = True
        wrapped = ast.Return(
            value=ast.Call(
                func=ast.Name(id=self.dead_fn_name, ctx=ast.Load()),
                args=[],
                keywords=[ast.keyword(arg=self.param_name, value=node.value)],
            )
        )
        ast.fix_missing_locations(wrapped)
        return wrapped


def _try_entangle(p_correct, dead_funcs):
    """
    Inject one dead-program leaf function into P_correct's call chain, bumping
    its depth by 1. Returns (new_p_correct_source, shared_fn_name), or None if
    no compatible (leaf, injectable dead function) pair exists.
    """
    tree = _safe_parse(p_correct)
    if tree is None:
        return None

    p_func_defs = _func_defs(tree)
    p_func_names = set(p_func_defs.keys())

    leaves = [name for name, fdef in p_func_defs.items() if not (_get_called_functions(fdef) & p_func_names)]
    if not leaves:
        return None

    def is_injectable(fdef):
        if len(fdef.args.args) != 1:
            return False
        for node in ast.walk(fdef):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                callee = node.func.id
                if callee == "print":
                    return False
                if callee not in p_func_names and callee != fdef.name:
                    return False
        return True

    safe_dead = [f for f in dead_funcs if is_injectable(f)]
    if not safe_dead:
        return None

    random.shuffle(leaves)
    random.shuffle(safe_dead)

    for leaf_name in leaves:
        for dead_fn in safe_dead:
            try:
                leaf_def = copy.deepcopy(p_func_defs[leaf_name])
                dead_fn_copy = copy.deepcopy(dead_fn)
                param_name = dead_fn.args.args[0].arg

                wrapper = _ReturnWrapper(dead_fn.name, param_name)
                new_leaf = wrapper.visit(leaf_def)
                if not wrapper.modified:
                    continue

                new_body = []
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name == leaf_name:
                        new_body.append(dead_fn_copy)
                        new_body.append(new_leaf)
                    else:
                        new_body.append(node)

                new_tree = ast.Module(body=new_body, type_ignores=[])
                ast.fix_missing_locations(new_tree)
                return ast.unparse(new_tree), dead_fn.name

            except Exception:
                continue

    return None


def _patch_ren_with_entangled(ren, entangled, original):
    """Propagate _try_entangle's rewritten leaf body back into ren so merge(ren)
    stays consistent with the entangled P_correct."""
    orig_tree, ent_tree = _safe_parse(original), _safe_parse(entangled)
    if orig_tree is None or ent_tree is None:
        return ren

    orig_funcs = {name: ast.unparse(fd) for name, fd in _func_defs(orig_tree).items()}
    ent_funcs = _func_defs(ent_tree)
    modified = {name: node for name, node in ent_funcs.items() if name in orig_funcs and ast.unparse(node) != orig_funcs[name]}
    if not modified:
        return ren

    new_ren = []
    for prog in ren:
        try:
            tree = ast.parse(prog)
            new_body, changed = [], False
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in modified:
                    new_body.append(copy.deepcopy(modified[node.name]))
                    changed = True
                else:
                    new_body.append(node)
            if changed:
                tree.body = new_body
                ast.fix_missing_locations(tree)
                new_ren.append(ast.unparse(tree))
            else:
                new_ren.append(prog)
        except Exception:
            new_ren.append(prog)

    return new_ren


class _ConsolidationRegistry:
    """Prevent structurally identical problems: keyed on (call-graph shape, depth)."""

    def __init__(self):
        self._seen = set()

    def _key(self, p_correct):
        return (_func_call_signature(p_correct), _call_chain_depth(p_correct))

    def is_duplicate(self, p_correct):
        return self._key(p_correct) in self._seen

    def register(self, p_correct):
        self._seen.add(self._key(p_correct))

    def reset(self):
        self._seen.clear()


def _inline_result_print(code):
    """Ensure the program has a top-level print statement.

    If `_result = ...; print(_result)` exists, compress it.
    Otherwise, append a call to the first function with RANDOM arguments
    so outputs differ across programs.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    # 1. Try the original _result+print compression
    new_body, i = [], 0
    while i < len(tree.body):
        s = tree.body[i]
        n = tree.body[i + 1] if i + 1 < len(tree.body) else None
        if (
            isinstance(s, ast.Assign)
            and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name)
            and s.targets[0].id == "_result"
            and n
            and isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id == "print"
            and len(n.value.args) == 1
            and isinstance(n.value.args[0], ast.Name)
            and n.value.args[0].id == "_result"
        ):
            new_body.append(
                ast.Expr(
                    ast.Call(
                        func=ast.Name("print", ast.Load()),
                        args=[s.value],
                        keywords=[],
                    )
                )
            )
            i += 2
        else:
            new_body.append(s)
            i += 1
    tree.body = new_body

    # 2. If still no executable statement, add a synthetic call with random args
    has_exec = any(not isinstance(n, ast.FunctionDef) for n in tree.body)
    if not has_exec and any(isinstance(n, ast.FunctionDef) for n in tree.body):
        first_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        fake_args = []
        for arg in first_fn.args.args:
            typ = getattr(arg.annotation, "id", "int") if arg.annotation else "int"
            if typ == "int":
                fake_args.append(ast.Constant(value=random.randint(-3, 7)))
            elif typ == "str":
                fake_args.append(ast.Constant(value=random.choice(["x", "y", "z", "w"])))
            elif typ == "list":
                length = random.randint(0, 2)
                fake_args.append(ast.List(elts=[ast.Constant(value=random.randint(0,5)) for _ in range(length)], ctx=ast.Load()))
            else:
                fake_args.append(ast.Constant(value=0))
        call = ast.Call(
            func=ast.Name(id=first_fn.name, ctx=ast.Load()),
            args=fake_args,
            keywords=[],
        )
        print_stmt = ast.Expr(
            ast.Call(func=ast.Name("print", ast.Load()), args=[call], keywords=[])
        )
        tree.body.append(print_stmt)

    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:
        return None


def _collect_func_names(code):
    tree = _safe_parse(code)
    return list(_func_defs(tree)) if tree else []


def _apply_name_map(code, name_map):
    tree = _safe_parse(code)
    if tree is None:
        return None

    class R(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.name = name_map.get(node.name, node.name)
            self.generic_visit(node)
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name):
                node.func.id = name_map.get(node.func.id, node.func.id)
            return node

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                node.id = name_map.get(node.id, node.id)
            return node

    tree = R().visit(tree)
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:
        return None


def rename_all(progs):
    """Safely rename function names in each program by appending a unique suffix.

    Program 0 will have functions f0__0, f1__0, etc.
    Program 1 will have f0__1, f1__1, etc.
    All definitions and call sites are updated consistently.
    """
    def rename_one(code, suffix):
        func_names = _collect_func_names(code)
        if not func_names:
            return None
        # Sort longest first to avoid partial replacements (e.g., f12 before f1)
        func_names.sort(key=len, reverse=True)
        new_code = code
        for old in func_names:
            new_name = f"{old}__{suffix}"
            # Replace all occurrences of the exact name (whole word)
            new_code = re.sub(rf'\b{re.escape(old)}\b', new_name, new_code)
        # Verify that the result is still parseable
        try:
            compile(new_code, "<rename>", "exec")
        except SyntaxError:
            return None
        return new_code

    out = []
    for i, prog in enumerate(progs):
        renamed = rename_one(prog, i)
        if renamed is None:
            return None
        out.append(renamed)
    return out


def merge(progs):
    try:
        trees = [ast.parse(p) for p in progs]
    except Exception:
        return None
    defs, execs = [], []
    for t in trees:
        defs.extend(n for n in t.body if isinstance(n, ast.FunctionDef))
        execs.append([n for n in t.body if not isinstance(n, ast.FunctionDef)])
    random.shuffle(defs)
    order = list(range(len(progs)))
    random.shuffle(order)
    mx = max((len(e) for e in execs), default=0)
    body = []
    for i in range(mx):
        for idx in order:
            if i < len(execs[idx]):
                body.append(execs[idx][i])
    mod = ast.Module(body=defs + body, type_ignores=[])
    ast.fix_missing_locations(mod)
    try:
        return ast.unparse(mod)
    except Exception:
        return None


def _normalise(code):
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    fdefs = sorted((n for n in tree.body if isinstance(n, ast.FunctionDef)), key=lambda n: n.name)
    other = [n for n in tree.body if not isinstance(n, ast.FunctionDef)]
    tree.body = fdefs + other
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:
        return None


def _program_metrics(code):
    tree = _safe_parse(code)
    fns = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)) if tree else 0
    return {"lines": _line_count(code), "functions": fns}


def consolidation_program(cfg):
    """Generate one standalone mesopy program."""
    d = random.randint(*cfg.depth_range)
    min_d = max(d - 6, 3)

    g = mesopy_grammar(
        mode="function",
        n_functions=random.randint(*cfg.n_functions_range),
        max_number=max(4, int(8 + 2 * cfg.difficulty)),
        max_params=2 + int(cfg.difficulty >= 4),
        param_types=("int", "str", "list"),
        return_types=("int", "str", "list"),
        emit_endpoint=False,
        emit_result=True,
        failure_rate=random.uniform(*cfg.failure_rate_range),
        triviality_rate=0.0,
        allow_recursion=False,
        include_loops=True,
        include_try_except=cfg.include_try_except,
        include_comprehensions=cfg.include_comprehensions,
        include_fstrings=cfg.include_fstrings,
        include_break_continue=cfg.include_break_continue,
        include_ternary=cfg.include_ternary,
        min_body_stmts=random.randint(*cfg.min_body_stmts_range),
    )
    try:
        code = generate(g, depth=d, min_depth=min_d) @ "py"
    except Exception:
        return None

    return None if _too_many_pass(code) else code


def _consolidation_worker(send, code, max_steps, recursionlimit):
    out, err = CapIO(), CapIO()
    ns = {"__builtins__": __builtins__}
    steps = 0
    sys.setrecursionlimit(recursionlimit)

    def trace(frame, event, arg):
        nonlocal steps
        if event == "line":
            steps += 1
            if steps > max_steps:
                raise StepLimit
        return trace

    ok = False
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            sys.settrace(trace)
            try:
                exec(compile(code, "<mesopy>", "exec"), ns, ns)
            finally:
                sys.settrace(None)
        ok = True
    except BaseException:
        ok = False

    try:
        send.send((ok, out.getvalue()))
    except Exception:
        pass
    send.close()


def kill(p):
    if p.is_alive():
        p.terminate()
        p.join(0.05)
        if p.is_alive():
            os.kill(p.pid, signal.SIGKILL)
            p.join()


def run_program(code, cfg, recursionlimit=80):
    ctx = mp.get_context("fork")
    recv, send = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_consolidation_worker, args=(send, code, cfg.max_steps, recursionlimit))
    p.start()
    send.close()
    try:
        if recv.poll(cfg.timeout):
            ok, stdout = recv.recv()
            p.join(0.05)
            kill(p)
            if not ok:
                return None
            out = stdout.strip()
            return out if out else None
        else:
            kill(p)
            return None
    except KeyboardInterrupt:
        kill(p)
        raise
    finally:
        recv.close()


def _is_trivial_output(out: str) -> bool:
    stripped = out.strip()
    if not stripped:
        return True
    if len(stripped) <= 2:
        return True
    if re.fullmatch(r'-?\d+', stripped) and len(stripped) <= 2:
        return True
    if re.fullmatch(r'\[(\s*\d+\s*,?)*\]', stripped) and len(stripped) <= 4:
        return True
    return False


# ---------------------------------------------------------------------------
# Configuration and Task class
# ---------------------------------------------------------------------------

class StepLimit(BaseException):
    pass


class CapIO(io.StringIO):
    def __init__(self, cap=2000):
        super().__init__()
        self.cap = cap

    def write(self, s):
        if self.tell() < self.cap:
            super().write(s[: self.cap - self.tell()])
        return len(s)


@dataclass
class ConsolidationConfig(Config):
    difficulty: float = 0.0
    timeout: float = 15.0
    max_steps: int = 20_000
    n_programs: int = 3
    depth_range: tuple = (6, 18)
    n_functions_range: tuple = (3, 8)
    min_body_stmts_range: tuple = (3, 8)
    failure_rate_range: tuple = (0.0, 0.15)
    include_comprehensions: bool = True
    include_fstrings: bool = True
    include_break_continue: bool = True
    include_try_except: bool = True
    include_ternary: bool = True
    include_classes: bool = False
    min_call_depth: int = 2
    entangle: bool = True
    deduplicate: bool = False
    max_attempts: int = 60

    def apply_difficulty(self, level):
        self.difficulty += level
        self.n_functions_range = (
            max(sround(self.n_functions_range[0] + 0.5 * level), 1),
            sround(self.n_functions_range[1] + level),
        )
        self.min_body_stmts_range = (
            max(sround(self.min_body_stmts_range[0] + 0.5 * level), 1),
            sround(self.min_body_stmts_range[1] + level),
        )
        self.depth_range = (
            max(sround(self.depth_range[0] + level), 3),
            sround(self.depth_range[1] + level),
        )
        self.failure_rate_range = (
            self.failure_rate_range[0],
            min(self.failure_rate_range[1] + 0.05 * level, 0.5),
        )
        self.min_call_depth = min(self.min_call_depth + int(level >= 3), 4)


class CodeConsolidation(Task):
    summary = "Extract the minimal source program from a merge of several generated programs."

    def __init__(self, config=None):
        super().__init__(config=config or ConsolidationConfig())
        self._registry = _ConsolidationRegistry()
        self.balancing_key_ratio = 1.0

    def reset_registry(self):
        self._registry.reset()

    def balancing_key(self, problem):
        md = problem.metadata
        return f"d{md.call_depth}_len{len(md.correct_stdout)}"

    def generate_entry(self):
        cfg = self.config
        for _ in range(cfg.max_attempts):
            raw = [consolidation_program(cfg) for _ in range(cfg.n_programs)]
            if any(p is None for p in raw):
                continue

            inl = [_inline_result_print(p) for p in raw]
            if any(p is None for p in inl):
                continue

            ren = rename_all(inl)
            if ren is None:
                continue

            runnable_raw = [(p, out) for p in ren if (out := run_program(p, cfg)) is not None]
            if len(runnable_raw) < 2:
                continue

            runnable = []
            for p_raw, expected_out in runnable_raw:
                pruned = _prune_to_reachable(p_raw)
                if pruned is None:
                    continue
                if run_program(pruned, cfg) != expected_out:
                    continue
                runnable.append((pruned, expected_out))
            if len(runnable) < 2:
                continue

            deep_runnable = [(p, o) for p, o in runnable if _call_chain_depth(p) >= cfg.min_call_depth]
            if not deep_runnable:
                continue

            p_correct, out = min(deep_runnable, key=lambda x: _line_count(x[0]))

            if _is_trivial_output(out):
                continue

            entangled_flag = False
            if cfg.entangle:
                p_correct_funcs = set(_func_defs(ast.parse(p_correct)))
                dead_funcs = []
                for p in ren:
                    tree = _safe_parse(p)
                    if tree is None:
                        continue
                    for node in tree.body:
                        if isinstance(node, ast.FunctionDef) and node.name not in p_correct_funcs:
                            dead_funcs.append(node)
                entangle_result = _try_entangle(p_correct, dead_funcs)
                if entangle_result is not None:
                    entangled_code, _shared_fn = entangle_result
                    if (new_out := run_program(entangled_code, cfg)) is not None:
                        p_correct, out = entangled_code, new_out
                        entangled_flag = True
                        ren = _patch_ren_with_entangled(ren, p_correct, deep_runnable[0][0])

            merged = merge(ren)
            if merged is None or run_program(merged, cfg) is None:
                continue

            if entangled_flag:
                final_funcs = set(_func_defs(ast.parse(p_correct)))
                merged_funcs = set(_func_defs(ast.parse(merged)))
                if not final_funcs.issubset(merged_funcs):
                    continue

            metadata = edict(
                merged_code=merged,
                correct_stdout=out,
                n_programs=cfg.n_programs,
                n_runnable=len(runnable),
                call_depth=_call_chain_depth(p_correct),
                entangled=entangled_flag,
                metrics=_program_metrics(merged),
            )
            return Entry(metadata=metadata, answer=p_correct)

        raise RuntimeError(f"CodeConsolidation: failed after {cfg.max_attempts} attempts")

    def render_prompt(self, m):
        depth_hint = f"The correct program has a call chain of depth {m.call_depth} (functions calling other functions)."
        entangle_hint = (
            "\nNote: some functions in the dead code are also called by the correct "
            "program — you must trace the full call graph to know what to keep."
            if m.get("entangled")
            else ""
        )
        return (
            "You are given a Python program P created by merging several independent programs.\n\n"
            f"Program P:\n```python\n{m.merged_code}\n```\n\n"
            "Each original program defined its own functions, then called them to produce "
            "output. Their definitions were collected, renamed to neutral tokens (g1234...), "
            "shuffled, and all executable statements interleaved.\n\n"
            f"Exactly ONE program produces this output on its own:\n\n```\n{m.correct_stdout}\n```\n\n"
            f"{depth_hint}{entangle_hint}\n\n"
            "Extract the MINIMAL program. Remove ALL dead code.\n"
            "RULES (any violation = failure):\n"
            "1. NEVER rename any function, class, method, or variable.\n"
            "2. Do NOT modify the body of any function you keep.\n"
            "3. Do NOT add new code.\n"
            "4. Your cleaned program must print EXACTLY the output above.\n"
            "5. It must contain ONLY what is needed — trace the full call graph from the "
            "exec statement(s) to find every required function.\n\n"
            "Return ONLY the cleaned Python program, no explanation."
        )

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        meta_ = entry["metadata"] if isinstance(entry, dict) else entry.metadata
        reference = entry["answer"] if isinstance(entry, dict) else entry.answer
        out = meta_["correct_stdout"] if isinstance(meta_, dict) else meta_.correct_stdout
        matches_output = run_program(answer, self.config) == out
        matches_source = _normalise(answer) == _normalise(reference)
        return 1.0 if matches_output and matches_source else 0.0

"""
Inverse Object Trace – recover hidden inputs from a composed object chain.
"""

import ast, copy, os, random, re, threading
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

from easydict import EasyDict as edict
from reasoning_core.template import Task, Problem, Config
from gramforge.grammars import mesopy_grammar
from gramforge import generate
# from ._gramforge_helpers.prerequisites import run_sandboxed


# ── Value domains ──────────────────────────────────────────────────────────

@dataclass
class ValueDomain:
    kind: str          # 'int' | 'str' | 'bool'
    values: list

    def describe(self) -> str:
        if self.kind == "int":
            lo, hi = self.values[0], self.values[-1]
            return f"an integer in the range [{lo}, {hi}]"
        if self.kind == "bool":
            return "a boolean (True or False)"
        if self.kind == "str":
            lengths = {len(v) for v in self.values}
            alphabet = sorted(set("".join(self.values)))
            if len(lengths) == 1:
                return f"a string of length {lengths.pop()} using only characters from {{{', '.join(alphabet)}}}"
            return f"a string using only characters from {{{', '.join(alphabet)}}}"
        raise ValueError(f"Unknown domain kind: {self.kind!r}")


def _build_domain(kind: str, rng: random.Random, cfg) -> ValueDomain:
    if kind == "int":
        return ValueDomain("int", list(range(cfg.domain_min, cfg.domain_max + 1)))
    if kind == "bool":
        return ValueDomain("bool", [True, False])
    if kind == "str":
        length = rng.randint(*cfg.str_length_range)
        values = ["".join(p) for p in product(cfg.str_alphabet, repeat=length)]
        return ValueDomain("str", values)
    raise ValueError(f"Unknown value kind: {kind!r}")


# ── Leaf extraction from GramForge output ──────────────────────────────────

def _method_has_print(method_node: ast.FunctionDef) -> bool:
    return any(
        isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "print"
        for call in ast.walk(method_node)
    )

def _extract_leaf_classes(code: str) -> list[tuple[ast.ClassDef, str]]:
    """Return (ClassDef, method_name) for classes with single-param __init__ and a clean, returning method."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    leaves = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        init_fn = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if init_fn is None or len([a.arg for a in init_fn.args.args if a.arg != "self"]) != 1:
            continue
        value_method_name = None
        for n in node.body:
            if not isinstance(n, ast.FunctionDef):
                continue
            if n.name == "__init__" or (n.name.startswith("__") and n.name.endswith("__")):
                continue
            if [a.arg for a in n.args.args if a.arg != "self"]:
                continue
            if _method_has_print(n):
                continue
            if any(isinstance(s, ast.Return) and s.value is not None for s in ast.walk(n)):
                value_method_name = n.name
                break
        if value_method_name:
            leaves.append((node, value_method_name))
    return leaves

def _canonicalize_leaf(node: ast.ClassDef, value_method_name: str, new_class_name: str, strip_annotations: bool = False) -> str:
    new_node = copy.deepcopy(node)
    new_node.name = new_class_name
    for n in new_node.body:
        if not isinstance(n, ast.FunctionDef):
            continue
        if n.name == "__init__" and strip_annotations:
            for a in n.args.args:
                if a.arg != "self":
                    a.annotation = None
        if n.name == value_method_name:
            n.name = "value"
            if strip_annotations:
                n.returns = None
    ast.fix_missing_locations(new_node)
    return ast.unparse(new_node)

def _structural_signature(method_node: ast.FunctionDef) -> str:
    blanked = copy.deepcopy(method_node)
    blanked.name = "_"
    for node in ast.walk(blanked):
        if isinstance(node, ast.arg):
            node.arg = "_"
        elif isinstance(node, ast.Name):
            node.id = "_"
        elif isinstance(node, ast.Attribute):
            node.attr = "_"
    return ast.dump(blanked, annotate_fields=False)

def _dedupe_leaves(leaves: list[tuple[ast.ClassDef, str]]) -> list[tuple[ast.ClassDef, str]]:
    seen: set[str] = set()
    unique = []
    for node, method in leaves:
        method_node = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
        sig = _structural_signature(method_node)
        if sig not in seen:
            seen.add(sig)
            unique.append((node, method))
    return unique


# ── Glue wrappers (hand-written) ───────────────────────────────────────────

_GLUE_OPS = ("add", "sub", "mul")
_GLUE_WEIGHTS = (0.30, 0.15, 0.55)  # favour mul

def _glue_ops_and_weights_for_kinds(kinds: list[str]) -> tuple[tuple, tuple]:
    if "str" in kinds:
        return _GLUE_OPS, (0.40, 0.05, 0.55)
    return _GLUE_OPS, _GLUE_WEIGHTS

def _render_glue_wrapper(class_name: str, op: str) -> str:
    expr = {"add": "self.prev.value() + self.leaf.value()",
            "sub": "self.prev.value() - self.leaf.value()",
            "mul": "self.prev.value() * self.leaf.value()"}[op]
    return (f"class {class_name}:\n"
            f"    def __init__(self, prev, leaf):\n"
            f"        self.prev = prev\n"
            f"        self.leaf = leaf\n"
            f"    def value(self):\n"
            f"        return {expr}\n")

def _render_final_transform_class(class_name: str, op: str, coeff) -> str:
    expr = {"add": f"self.prev.value() + {repr(coeff)}",
            "sub": f"self.prev.value() - {repr(coeff)}",
            "mul": f"self.prev.value() * {repr(coeff)}"}[op]
    return (f"class {class_name}:\n"
            f"    def __init__(self, prev):\n"
            f"        self.prev = prev\n"
            f"    def value(self):\n"
            f"        return {expr}\n")

def _build_chain_plan(leaf_srcs: list[str], wrapper_ops: list[str], final_transform: Optional[tuple]) -> dict:
    n_hidden = len(leaf_srcs)
    class_defs = list(leaf_srcs)
    wrapper_names = []
    for i, op in enumerate(wrapper_ops):
        wname = f"Wrap{i+2}"
        class_defs.append(_render_glue_wrapper(wname, op))
        wrapper_names.append(wname)
    final_name = None
    if final_transform is not None:
        op, coeff = final_transform
        final_name = "Final"
        class_defs.append(_render_final_transform_class(final_name, op, coeff))
    return dict(class_defs=class_defs, n_hidden=n_hidden, wrapper_names=wrapper_names, final_name=final_name)

def _render_wiring_lines(plan: dict, val_strs: list[str]) -> tuple[list[str], str]:
    lines = [f"obj1 = Leaf1({val_strs[0]})"]
    prev = "obj1"
    for i, wname in enumerate(plan["wrapper_names"]):
        cur = f"obj{i+2}"
        lines.append(f"{cur} = {wname}({prev}, Leaf{i+2}({val_strs[i+1]}))")
        prev = cur
    if plan["final_name"]:
        lines.append(f"final = {plan['final_name']}({prev})")
        prev = "final"
    return lines, prev

def _render_display(plan: dict) -> str:
    class_block = "\n\n".join(plan["class_defs"])
    val_decls = "\n".join(f"val{i+1} = ?" for i in range(plan["n_hidden"]))
    wiring, outer = _render_wiring_lines(plan, [f"val{i+1}" for i in range(plan["n_hidden"])])
    body = "\n".join(wiring)
    return f"{class_block}\n\n{val_decls}\n{body}\nprint({outer}.value())"

def _render_executable(plan: dict, values: list) -> str:
    class_block = "\n\n".join(plan["class_defs"])
    wiring, outer = _render_wiring_lines(plan, [repr(v) for v in values])
    body = "\n".join(wiring)
    return f"{class_block}\n\n{body}\nprint({outer}.value())"

def _render_probe_script(plan: dict, domains: list, allowed_output_types: tuple) -> str:
    class_block = "\n\n".join(plan["class_defs"])
    n = plan["n_hidden"]
    arg_names = [f"val{i+1}" for i in range(n)]
    wiring, outer = _render_wiring_lines(plan, arg_names)
    probe_body = "\n    ".join(wiring)
    probe_fn = f"def _probe({', '.join(arg_names)}):\n    {probe_body}\n    return {outer}.value()\n"
    domain_lines = "\n".join(f"_domain_{i} = {domains[i].values!r}" for i in range(n))
    allowed_repr = "(" + ", ".join(t.__name__ for t in allowed_output_types) + ")"
    axis_names = ", ".join(f"_domain_{i}" for i in range(n))
    sweep = (
        f"import itertools\n{domain_lines}\n"
        f"for combo in itertools.product({axis_names}):\n"
        f"    try:\n"
        f"        out = _probe(*combo)\n"
        f"        if type(out) not in {allowed_repr}:\n"
        f"            continue\n"
        f"        print(repr(combo) + '|' + repr(out))\n"
        f"    except Exception:\n"
        f"        pass\n"
    )
    return f"{class_block}\n\n{probe_fn}\n{sweep}"

def _parse_probe_output(stdout: str) -> dict:
    table = defaultdict(list)
    for line in stdout.strip().splitlines():
        if "|" not in line:
            continue
        combo_repr, out_repr = line.split("|", 1)
        try:
            combo = ast.literal_eval(combo_repr)
            out = ast.literal_eval(out_repr)
        except (ValueError, SyntaxError):
            continue
        table[out].append(combo)
    return table

def _boundary_score(inputs: tuple, domains: list) -> int:
    score = 0
    for v, dom in zip(inputs, domains):
        if dom.kind == "int" and (v == dom.values[0] or v == dom.values[-1]):
            score += 1
    return score

def _output_within_bounds(out, max_output_abs: int, max_output_str_len: int) -> bool:
    if isinstance(out, bool):
        return True
    if isinstance(out, int):
        return abs(out) <= max_output_abs
    if isinstance(out, str):
        return len(out) <= max_output_str_len
    return True

def _table_violates_bounds(table: dict, max_output_abs: int, max_output_str_len: int) -> bool:
    return any(not _output_within_bounds(out, max_output_abs, max_output_str_len) for out in table)

def _pick_ground_truth(table: dict, domains: list, max_boundary_inputs: int, max_output_abs: int, max_output_str_len: int, rng: random.Random, strict: bool = True) -> Optional[tuple]:
    unique = {out: combos[0] for out, combos in table.items() if len(combos) == 1}
    if not unique:
        return None
    good = {out: inp for out, inp in unique.items()
            if _boundary_score(inp, domains) <= max_boundary_inputs
            and _output_within_bounds(out, max_output_abs, max_output_str_len)}
    if good:
        out = rng.choice(list(good.keys()))
        return out, good[out]
    if strict:
        return None
    out = rng.choice(list(unique.keys()))
    return out, unique[out]


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class InverseObjectTraceConfig(Config):
    depth_range: tuple = (6, 14)
    n_functions_range: tuple = (3, 8)
    min_body_stmts_range: tuple = (3, 8)
    failure_rate_range: tuple = (0.0, 0.15)
    usage_bias_range: tuple = (0.2, 0.6)
    include_comprehensions: bool = True
    include_fstrings: bool = True
    include_ternary: bool = True
    include_break_continue: bool = True
    include_try_except: bool = True
    n_classes_range: tuple = (2, 6)
    inherit_rate_range: tuple = (0.0, 0.0)

    chain_depth_range: tuple = (2, 2)
    domain_min: int = 0
    domain_max: int = 12
    max_output_abs: int = 50_000
    max_output_str_len: int = 200
    min_unique_outputs: int = 3
    max_boundary_inputs: int = 1
    include_final_transform: bool = True
    final_transform_coeff_range: tuple = (2, 5)

    value_kinds: tuple = ("int",)
    str_alphabet: str = "abcdefghij"
    str_length_range: tuple = (1, 2)
    allowed_output_types: tuple = field(default_factory=lambda: (int, str, bool))

    n_leaf_oversample_calls: int = 6
    n_wrapper_op_trials: int = 8
    max_attempts: int = 300
    partial_credit: bool = False

    def update(self, c: int) -> None:
        lo, hi = self.chain_depth_range
        self.chain_depth_range = (lo, min(hi + (c // 2), 6))
        self.domain_max = min(self.domain_max + 3 * c, 30)
        self.max_output_abs = min(self.max_output_abs + 20_000 * c, 500_000)
        self.max_attempts = max(int(self.max_attempts * (1 + 0.1 * c)), 200)


# ── Task ───────────────────────────────────────────────────────────────────

_VAL_LINE_RE = re.compile(r"val[_\s]*(\d+)\s*:?\s*=?\s*(.+)", re.IGNORECASE)

def _kind_matches(val, kind: str) -> bool:
    if kind == "int":
        return isinstance(val, int) and not isinstance(val, bool)
    if kind == "bool":
        return isinstance(val, bool)
    if kind == "str":
        return isinstance(val, str)
    return False

def _coerce_token(token: str, kind: str):
    token = token.strip()
    try:
        val = ast.literal_eval(token)
    except (ValueError, SyntaxError):
        val = None
    if val is not None and _kind_matches(val, kind):
        return val
    if kind == "bool":
        low = token.lower()
        if low == "true": return True
        if low == "false": return False
    if kind == "str":
        return token.strip("'\"")
    return None

def _parse_answer(ans: Optional[str], domain_kinds: list) -> Optional[tuple]:
    if not ans:
        return None
    found = {}
    for line in ans.splitlines():
        m = _VAL_LINE_RE.search(line)
        if not m:
            continue
        idx = int(m.group(1))
        if not (1 <= idx <= len(domain_kinds)):
            continue
        val = _coerce_token(m.group(2), domain_kinds[idx - 1])
        if val is None:
            continue
        found[idx] = val
    if not all(i in found for i in range(1, len(domain_kinds) + 1)):
        return None
    return tuple(found[i] for i in range(1, len(domain_kinds) + 1))

def _strict_eq(p, c) -> bool:
    return type(p) == type(c) and p == c


class CodeInverseTrace(Task):
    task_name = "code_inverse_trace"
    balancing_key_ratio = 1.0
    _seed_counter = 0
    _seed_lock = threading.Lock()

    def __init__(self, config=None):
        super().__init__(config=config or InverseObjectTraceConfig())

    def _next_seed(self) -> int:
        with self._seed_lock:
            CodeInverseTrace._seed_counter += 1
            return (os.getpid() << 32) + CodeInverseTrace._seed_counter

    def _sample_kwargs(self, rng: random.Random) -> dict:
        c = self.config
        return dict(
            mode="function",
            n_functions=rng.randint(*c.n_functions_range),
            max_params=2,
            min_body_stmts=rng.randint(*c.min_body_stmts_range),
            include_classes=True,
            n_classes=max(rng.randint(*c.n_classes_range), rng.randint(*c.chain_depth_range)),
            include_instance_use=True,
            allow_recursion=True,
            allow_cross_calls=True,
            safe_returns=True,
            failure_rate=rng.uniform(*c.failure_rate_range),
            triviality_rate=0.0,
            usage_bias=rng.uniform(*c.usage_bias_range),
            inherit_rate=0.5,
            include_loops=True,
            include_conditionals=True,
            include_augmented_assigns=True,
            include_ternary=c.include_ternary,
            include_assert=True,
            include_comprehensions=c.include_comprehensions,
            include_fstrings=c.include_fstrings,
            include_extra_ops=True,
            include_swap=True,
            include_break_continue=c.include_break_continue,
            include_try_except=c.include_try_except,
            include_print=True,
            include_dunders=True,
            endpoint_mode="auto",
            emit_endpoint=True,
            emit_result=False,
            print_result=False,
        )

    def _sample_gramforge_classes(self, rng: random.Random) -> list[tuple[ast.ClassDef, str]]:
        kwargs = self._sample_kwargs(rng)
        try:
            g = mesopy_grammar(**kwargs)
            code = generate(g) @ "py"
        except Exception:
            return []
        return _extract_leaf_classes(code)

    def _collect_leaves(self, rng: random.Random, n_needed: int) -> Optional[list[tuple[ast.ClassDef, str]]]:
        c = self.config
        pool = []
        for _ in range(c.n_leaf_oversample_calls):
            pool.extend(self._sample_gramforge_classes(rng))
            pool = _dedupe_leaves(pool)
            if len(pool) >= n_needed:
                return pool
        return None

    def generate(self) -> Problem:
        cfg = self.config
        n_strict_attempts = int(cfg.max_attempts * 0.8)
        stage_fail = {"no_leaves_collected": 0, "no_wrapper_combo_found": 0, "no_ground_truth_found": 0, "sandbox_mismatch": 0}
        last_leaf_pool_size = 0

        for attempt_i in range(cfg.max_attempts):
            rng = random.Random(self._next_seed())
            strict = attempt_i < n_strict_attempts

            n_hidden = rng.randint(*cfg.chain_depth_range)
            leaf_pool = self._collect_leaves(rng, n_hidden)
            if leaf_pool is None:
                stage_fail["no_leaves_collected"] += 1
                continue
            last_leaf_pool_size = len(leaf_pool)

            chosen = rng.sample(leaf_pool, n_hidden)
            kinds = [rng.choice(cfg.value_kinds) for _ in range(n_hidden)]
            domains = [_build_domain(k, rng, cfg) for k in kinds]

            leaf_srcs = [_canonicalize_leaf(node, method, f"Leaf{i+1}", strip_annotations=(kinds[i] != "int"))
                         for i, (node, method) in enumerate(chosen)]

            final_transform = None
            if cfg.include_final_transform:
                if "str" in kinds:
                    op = rng.choice(("add", "mul"))
                    if op == "add":
                        coeff = "".join(rng.choices(cfg.str_alphabet, k=rng.randint(1,2)))
                    else:
                        coeff = rng.randint(*cfg.final_transform_coeff_range)
                else:
                    op = rng.choice(_GLUE_OPS)
                    coeff = rng.randint(*cfg.final_transform_coeff_range)
                final_transform = (op, coeff)

            glue_ops_pool, glue_weights = _glue_ops_and_weights_for_kinds(kinds)
            found_plan, found_table = None, None

            for _ in range(cfg.n_wrapper_op_trials):
                wrapper_ops = list(rng.choices(glue_ops_pool, weights=glue_weights, k=n_hidden-1)) if n_hidden>1 else []
                plan = _build_chain_plan(leaf_srcs, wrapper_ops, final_transform)
                probe = _render_probe_script(plan, domains, cfg.allowed_output_types)
                r = run_sandboxed(probe, timeout=10.0)
                if not r.success:
                    continue
                table = _parse_probe_output(r.stdout)
                n_unique = sum(1 for v in table.values() if len(v) == 1)
                if n_unique >= cfg.min_unique_outputs and not _table_violates_bounds(table, cfg.max_output_abs, cfg.max_output_str_len):
                    found_plan, found_table = plan, table
                    break

            if found_plan is None:
                stage_fail["no_wrapper_combo_found"] += 1
                continue

            picked = _pick_ground_truth(found_table, domains, cfg.max_boundary_inputs, cfg.max_output_abs, cfg.max_output_str_len, rng, strict=strict)
            if picked is None:
                stage_fail["no_ground_truth_found"] += 1
                continue
            gt_output, gt_inputs = picked

            exe = _render_executable(found_plan, list(gt_inputs))
            r = run_sandboxed(exe, timeout=3.0)
            if not r.success or r.stdout.strip() != str(gt_output):
                stage_fail["sandbox_mismatch"] += 1
                continue

            display = _render_display(found_plan)
            meta = edict(
                program=display,
                n_hidden=n_hidden,
                domain_kinds=kinds,
                domain_descriptions=[d.describe() for d in domains],
                output=gt_output,
                output_type=type(gt_output).__name__,
                inputs=list(gt_inputs),
            )
            return Problem(metadata=meta, answer=tuple(gt_inputs))

        breakdown = "\n".join(f"    {k}: {v}" for k,v in stage_fail.items())
        raise RuntimeError(
            f"InverseObjectTrace: failed after {cfg.max_attempts} attempts.\n"
            f"  Failure breakdown:\n{breakdown}\n"
            f"  Last leaf-pool size: {last_leaf_pool_size} (need >= {cfg.chain_depth_range[1]})"
        )

    def prompt(self, m) -> str:
        val_list = ", ".join(f"val{i+1}" for i in range(m.n_hidden))
        domain_lines = "\n".join(f"  val{i+1}: {desc}" for i, desc in enumerate(m.domain_descriptions))
        answer_lines = "\n".join(f"val{i+1}: <value>" for i in range(m.n_hidden))
        return (
            f"You are given a Python program that builds a nested object structure. {m.n_hidden} initial value(s) — {val_list} — "
            f"are hidden (shown as `?`). The program prints a single result.\n\n"
            f"{m.program}\n\n"
            f"Printed output: {m.output}  (type: {m.output_type})\n\n"
            f"Hidden value domains:\n{domain_lines}\n\n"
            f"Exactly one combination of hidden values produces this output.\n\n"
            f"Determine the hidden values. Write integer values as plain numbers (e.g. val1: 7), boolean values as True or False, "
            f"and string values in quotes (e.g. val2: 'ab'). Return your answer in exactly this format, with no other text:\n{answer_lines}"
        )

    def score_answer(self, a, entry):
        domain_kinds = entry["metadata"].domain_kinds
        parsed = _parse_answer(a, domain_kinds)
        correct = tuple(entry["answer"])
        if parsed is None:
            return 0.0 if self.config.partial_credit else 0
        if not self.config.partial_credit:
            return 1 if all(_strict_eq(p, c) for p, c in zip(parsed, correct)) else 0
        hits = sum(_strict_eq(p, c) for p, c in zip(parsed, correct))
        return hits / len(correct)

    def balancing_key(self, p) -> str:
        kinds = ",".join(sorted(set(p.metadata.domain_kinds)))
        return f"{p.metadata.n_hidden}:{kinds}"


#### TODO !!!
