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


def _limit_worker(cpu_seconds=1):
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    except Exception:
        pass


def _execute(code, magnitude, recursionlimit, max_steps, call_args=None, batch=False, reports=False):
    out, err = CapIO(), CapIO()
    ns = {"__builtins__": __builtins__}
    steps, t0 = 0, time.perf_counter()
    sys.setrecursionlimit(recursionlimit)

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

        return values if reports else RunReport(
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
        return RunReport(
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
        return RunReport(
            False,
            None,
            type(e).__name__,
            locals().get("args"),
            out.getvalue(),
            err.getvalue(),
            steps,
            time.perf_counter() - t0,
        )


def _worker(send, code, magnitude, recursionlimit, max_steps, call_args=None, batch=False, reports=False):
    _limit_worker()
    try:
        send.send(_execute(code, magnitude, recursionlimit, max_steps, call_args, batch, reports))
    except Exception:
        pass
    send.close()


def _candidate_worker(send, candidates, magnitude, recursionlimit, max_steps, probes):
    _limit_worker(2)
    try:
        send.send([
            _execute(code, magnitude, recursionlimit, max_steps, probes, True, True)
            for _, code in candidates
        ])
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


def run_candidates(candidates, probes, cfg, recursionlimit=80):
    """Probe related programs together while retaining process isolation."""
    ctx = mp.get_context("fork")
    recv, send = ctx.Pipe(duplex=False)
    p = ctx.Process(
        target=_candidate_worker,
        args=(send, candidates, cfg.magnitude, recursionlimit, cfg.max_steps, probes),
    )
    p.start()
    send.close()
    timeout = min(4.0, cfg.timeout + 0.02 * len(candidates))
    try:
        if recv.poll(timeout):
            try:
                reports = recv.recv()
            except EOFError:
                reports = []
            p.join(0.05)
            kill(p)
            return reports
        kill(p)
        return []
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
        candidates = list(dict.fromkeys([("organic", base)] + perturbed))
        for (kind, code), reports in zip(candidates, run_candidates(candidates, probes, cfg)):
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

    def balancing_key(self, problem):
        return problem.metadata.mode
