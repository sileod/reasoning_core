"""Goal-directed imperative Python program synthesis for synthetic reasoning tasks.

The generator builds runnable programs by construction and exposes the same program
distribution to execution, runnability, profiling, input-deduction, and related
tasks.  Runnability pairs use identical source with different calls, avoiding the
strongest "planted bug" shortcut.

The implementation deliberately keeps the semantic state small enough to reason
about while composing independently sampled phenomena and surface realizations.
Execution remains the final oracle, not the search procedure.
"""

import ast
import random
from dataclasses import dataclass, field


PHENOMENA = (
    "aliasing",
    "closure_late_binding",
    "default_capture",
    "mutation_call",
    "loop_carried_state",
    "rebinding_vs_aliasing",
    "conditional_flow",
    "helper_chain",
    "comprehension",
    "mapping_bridge",
)


@dataclass
class MesopyGoal:
    runnable: bool | None = True
    paired_runnability: bool = False
    error: str | None = None
    phenomena: tuple[str, ...] = ()
    min_phenomena: int = 3
    max_phenomena: int = 6
    result_kind: str | None = None
    input_arity: int | None = None


@dataclass
class MesopyConfig:
    magnitude: int = 5
    list_size: int = 4
    min_segments: int = 3
    max_segments: int = 7
    input_arity: tuple[int, int] = (0, 2)
    safe_hazard_rate: float = 0.35
    noise_rate: float = 0.35
    type_hints: bool = False
    max_attempts: int = 40


@dataclass
class CallOutcome:
    args: tuple
    ok: bool
    value: str | None = None
    error: str | None = None


@dataclass
class MesopySample:
    code: str
    phenomena: tuple[str, ...]
    calls: tuple[CallOutcome, ...]
    features: dict = field(default_factory=dict)

    @property
    def call(self):
        return self.calls[0]

    @property
    def args(self):
        return self.call.args

    @property
    def answer(self):
        return self.call.value if self.call.ok else self.call.error


class _Names:
    def __init__(self, rng):
        self.rng = rng
        self.used = set()
        self.pools = {
            "state": ["state", "values", "buf", "items", "work", "data"],
            "acc": ["acc", "total", "score", "carry", "offset"],
            "tmp": ["tmp", "part", "piece", "delta", "hold", "cache"],
            "alias": ["alias", "view", "ref", "other", "shared"],
            "fn": ["f", "step", "adjust", "mix", "apply", "transform"],
            "loop": ["i", "j", "k", "v", "q"],
            "map": ["table", "mapping", "slots", "lookup"],
        }

    def take(self, kind):
        pool = [x for x in self.pools[kind] if x not in self.used]
        base = self.rng.choice(pool or self.pools[kind])
        name = base
        suffix = 2
        while name in self.used:
            name = f"{base}{suffix}"
            suffix += 1
        self.used.add(name)
        return name


def _name(x, ctx=ast.Load()):
    return ast.Name(id=x, ctx=ctx)


def _const(x):
    return ast.Constant(value=x)


def _sub(name, i, ctx=ast.Load()):
    return ast.Subscript(value=_name(name), slice=_const(i), ctx=ctx)


def _assign(target, value):
    return ast.Assign(targets=[target], value=value)


def _aug(target, op, value):
    return ast.AugAssign(target=target, op=op, value=value)


def _call(fn, *args):
    return ast.Call(func=_name(fn), args=list(args), keywords=[])


def _bin(a, op, b):
    return ast.BinOp(left=a, op=op, right=b)


def _compare(a, op, b):
    return ast.Compare(left=a, ops=[op], comparators=[b])


class ImperativeMesopy:
    def __init__(self, config=None, seed=None):
        self.config = config or MesopyConfig()
        self.rng = random.Random(seed)

    def generate(self, goal=None):
        goal = goal or MesopyGoal()
        for _ in range(self.config.max_attempts):
            sample = self._generate_once(goal)
            if self._valid(sample, goal):
                return sample
        raise RuntimeError(f"failed to generate imperative Mesopy sample: {goal}")

    def execution(self, **kwargs):
        return self.generate(MesopyGoal(runnable=True, **kwargs))

    def runnability_pair(self, **kwargs):
        return self.generate(MesopyGoal(paired_runnability=True, runnable=None, **kwargs))

    def _generate_once(self, goal):
        cfg = self.config
        rng = self.rng
        names = _Names(rng)

        n = max(3, int(cfg.list_size))
        arity = goal.input_arity
        if arity is None:
            lo, hi = cfg.input_arity
            needs_input = goal.paired_runnability or goal.runnable is False or bool(goal.error)
            arity = rng.randint(max(1 if needs_input else 0, lo), max(1 if needs_input else 0, hi))
        if goal.paired_runnability or goal.runnable is False or goal.error:
            arity = max(1, arity)
        params = [f"x{i}" for i in range(arity)]
        names.used.update(params)

        state = names.take("state")
        acc = names.take("acc")
        init = []
        for i in range(n):
            if params:
                p = _name(params[i % len(params)])
                k = rng.randint(-cfg.magnitude, cfg.magnitude)
                expr = _bin(p, ast.Add(), _const(k))
            else:
                expr = _const(rng.randint(-cfg.magnitude, cfg.magnitude))
            init.append(expr)

        body = [
            _assign(_name(state, ast.Store()), ast.List(elts=init, ctx=ast.Load())),
            _assign(_name(acc, ast.Store()), _const(rng.randint(-cfg.magnitude, cfg.magnitude))),
        ]

        requested = list(goal.phenomena)
        unknown = set(requested) - set(PHENOMENA)
        if unknown:
            raise ValueError(f"unknown phenomena: {sorted(unknown)}")
        lower = max(cfg.min_segments, goal.min_phenomena, len(requested))
        upper = max(lower, min(cfg.max_segments, max(goal.max_phenomena, len(requested))))
        target_n = rng.randint(lower, upper)
        available = [x for x in PHENOMENA if x not in requested]
        rng.shuffle(available)
        phenomena = requested + available[: max(0, target_n - len(requested))]
        rng.shuffle(phenomena)

        depth = 1
        for k, phenomenon in enumerate(phenomena):
            stmts, delta = getattr(self, f"_p_{phenomenon}")(
                state, acc, params, n, names, k
            )
            body.extend(stmts)
            depth += delta
            if rng.random() < cfg.noise_rate:
                body.extend(self._noise(state, acc, params, n, names))

        hazard = None
        if goal.paired_runnability or goal.runnable is False or goal.error or rng.random() < cfg.safe_hazard_rate:
            hazard = self._pick_hazard(goal.error)
            body.extend(self._hazard(hazard, state, acc, params, n, names))

        result_kind = goal.result_kind or rng.choice(("list", "int", "tuple"))
        body.append(ast.Return(value=self._result_expr(result_kind, state, acc, n)))

        fn = ast.FunctionDef(
            name="endpoint",
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg=p, annotation=_name("int") if cfg.type_hints else None)
                    for p in params
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=body,
            decorator_list=[],
            returns=None,
        )
        module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
        code = ast.unparse(module) + "\n"

        if goal.paired_runnability:
            calls = self._paired_calls(code, arity, hazard)
        else:
            args = self._bad_args(arity, hazard) if (goal.runnable is False or goal.error) else self._safe_args(arity, hazard)
            calls = (self._execute(code, args),)

        counts = self._features(module)
        counts.update(
            dataflow_depth=depth,
            result_kind=result_kind,
            hazard=hazard,
            input_arity=arity,
        )
        return MesopySample(code, tuple(phenomena), calls, counts)

    def _pick_hazard(self, requested):
        aliases = {
            None: None,
            "IndexError": "index",
            "ZeroDivisionError": "division",
            "ValueError": "lookup",
        }
        if requested not in aliases:
            raise ValueError("supported errors: IndexError, ZeroDivisionError, ValueError")
        return aliases[requested] or self.rng.choice(("index", "division", "lookup"))

    def _safe_args(self, arity, hazard):
        mag = self.config.magnitude
        if arity == 0:
            return ()
        xs = [self.rng.randint(-mag, mag) for _ in range(arity)]
        if hazard == "index":
            xs[0] = self.rng.randrange(max(3, self.config.list_size))
        elif hazard == "division":
            xs[0] = self.rng.choice([x for x in range(-mag, mag + 1) if x != 1])
        elif hazard == "lookup":
            xs[0] = self.rng.choice((-1, 0, 1))
        return tuple(xs)

    def _bad_args(self, arity, hazard):
        xs = list(self._safe_args(arity, hazard))
        if hazard == "index":
            xs[0] = max(3, self.config.list_size) + self.rng.randint(1, 3)
        elif hazard == "division":
            xs[0] = 1
        elif hazard == "lookup":
            xs[0] = self.config.magnitude + 17
        return tuple(xs)

    def _paired_calls(self, code, arity, hazard):
        safe = self._safe_args(arity, hazard)
        bad = self._bad_args(arity, hazard)
        outcomes = [self._execute(code, safe), self._execute(code, bad)]
        self.rng.shuffle(outcomes)
        return tuple(outcomes)

    def _execute(self, code, args):
        allowed = {
            "range": range,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
        }
        ns = {"__builtins__": allowed}
        try:
            exec(compile(code, "<imperative-mesopy>", "exec"), ns, ns)
            value = ns["endpoint"](*args)
            return CallOutcome(tuple(args), True, repr(value), None)
        except Exception as e:
            return CallOutcome(tuple(args), False, None, type(e).__name__)

    def _valid(self, sample, goal):
        if goal.paired_runnability:
            return (
                len(sample.calls) == 2
                and {x.ok for x in sample.calls} == {True, False}
                and (
                    goal.error is None
                    or any(x.error == goal.error for x in sample.calls)
                )
            )
        if goal.runnable is True:
            return sample.call.ok
        if goal.runnable is False:
            return not sample.call.ok
        return True

    def _result_expr(self, kind, state, acc, n):
        if kind == "list":
            return _name(state)
        if kind == "int":
            return _bin(_call("sum", _name(state)), ast.Add(), _name(acc))
        i, j = self.rng.sample(range(n), 2)
        return ast.Tuple(elts=[_name(acc), _sub(state, i), _sub(state, j)], ctx=ast.Load())

    def _noise(self, state, acc, params, n, names):
        tmp = names.take("tmp")
        i = self.rng.randrange(n)
        if self.rng.random() < 0.5:
            expr = _bin(_sub(state, i), ast.Add(), _const(self.rng.randint(-3, 3)))
        else:
            expr = _call("abs", _bin(_sub(state, i), ast.Sub(), _name(acc)))
        return [_assign(_name(tmp, ast.Store()), expr)]

    def _p_aliasing(self, state, acc, params, n, names, k):
        alias = names.take("alias")
        i, j = self.rng.sample(range(n), 2)
        d = self._small_nonzero()
        mutate = self._update(_sub(alias, i, ast.Store()), ast.Add(), _const(d))
        use = self._update(_name(acc, ast.Store()), ast.Add(), _sub(state, i))
        if self.rng.random() < 0.5:
            use = self._update(_sub(state, j, ast.Store()), ast.Add(), _sub(alias, i))
        return [
            _assign(_name(alias, ast.Store()), _name(state)),
            mutate,
            use,
        ], 2

    def _p_rebinding_vs_aliasing(self, state, acc, params, n, names, k):
        alias = names.take("alias")
        i, j = self.rng.sample(range(n), 2)
        d = self._small_nonzero()
        return [
            _assign(_name(alias, ast.Store()), _name(state)),
            _assign(
                _name(state, ast.Store()),
                ast.Subscript(
                    value=_name(state),
                    slice=ast.Slice(lower=None, upper=None, step=None),
                    ctx=ast.Load(),
                ),
            ),
            self._update(_sub(state, i, ast.Store()), ast.Add(), _const(d)),
            self._update(_sub(alias, j, ast.Store()), ast.Add(), _sub(state, i)),
            self._update(_name(acc, ast.Store()), ast.Add(), _sub(alias, j)),
        ], 3

    def _p_closure_late_binding(self, state, acc, params, n, names, k):
        bias = names.take("tmp")
        fn = names.take("fn")
        i, j = self.rng.sample(range(n), 2)
        a, b = self._small_nonzero(), self._small_nonzero()
        arg = ast.arg(arg="v")
        helper = ast.FunctionDef(
            name=fn,
            args=ast.arguments(
                posonlyargs=[], args=[arg], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=[
                ast.Return(
                    value=_bin(
                        _bin(_name("v"), ast.Add(), _name(bias)),
                        ast.Add(),
                        _sub(state, j),
                    )
                )
            ],
            decorator_list=[],
        )
        return [
            _assign(_name(bias, ast.Store()), _const(a)),
            helper,
            self._update(_name(bias, ast.Store()), ast.Add(), _const(b)),
            _assign(_sub(state, i, ast.Store()), _call(fn, _sub(state, i))),
        ], 3

    def _p_default_capture(self, state, acc, params, n, names, k):
        bias = names.take("tmp")
        fn = names.take("fn")
        i, j = self.rng.sample(range(n), 2)
        a, b = self._small_nonzero(), self._small_nonzero()
        helper = ast.FunctionDef(
            name=fn,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="v"), ast.arg(arg="bias")],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[_name(bias)],
            ),
            body=[
                ast.Return(
                    value=_bin(
                        _bin(_name("v"), ast.Add(), _name("bias")),
                        ast.Add(),
                        _sub(state, j),
                    )
                )
            ],
            decorator_list=[],
        )
        return [
            _assign(_name(bias, ast.Store()), _const(a)),
            helper,
            self._update(_name(bias, ast.Store()), ast.Add(), _const(b)),
            _assign(_sub(state, i, ast.Store()), _call(fn, _sub(state, i))),
        ], 3

    def _p_mutation_call(self, state, acc, params, n, names, k):
        fn = names.take("fn")
        i, j = self.rng.sample(range(n), 2)
        d = self._small_nonzero()
        helper = ast.FunctionDef(
            name=fn,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="seq"), ast.arg(arg="d")],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=[
                self._update(_sub("seq", i, ast.Store()), ast.Add(), _name("d")),
                ast.Return(value=_bin(_sub("seq", i), ast.Sub(), _sub("seq", j))),
            ],
            decorator_list=[],
        )
        return [
            helper,
            self._update(
                _sub(state, j, ast.Store()),
                ast.Add(),
                _call(fn, _name(state), _const(d)),
            ),
        ], 3

    def _p_loop_carried_state(self, state, acc, params, n, names, k):
        i, j = self.rng.sample(range(n), 2)
        loop = names.take("loop")
        vals = [self.rng.randint(-3, 3) for _ in range(self.rng.randint(2, 4))]
        mul = self.rng.choice((-2, -1, 1, 2))
        loop_body = [
            _assign(
                _sub(state, i, ast.Store()),
                _bin(
                    _bin(_const(mul), ast.Mult(), _sub(state, i)),
                    ast.Add(),
                    _name(loop),
                ),
            ),
            self._update(_sub(state, j, ast.Store()), ast.Add(), _sub(state, i)),
        ]
        if self.rng.random() < 0.55:
            stmt = ast.For(
                target=_name(loop, ast.Store()),
                iter=ast.List(elts=[_const(x) for x in vals], ctx=ast.Load()),
                body=loop_body,
                orelse=[],
            )
            return [stmt], 3

        seq = names.take("tmp")
        idx = names.take("loop")
        while_stmt = ast.While(
            test=_compare(_name(idx), ast.Lt(), _call("len", _name(seq))),
            body=[
                _assign(_name(loop, ast.Store()), _sub(seq, 0)),
            ] + loop_body + [
                self._update(_name(idx, ast.Store()), ast.Add(), _const(1))
            ],
            orelse=[],
        )
        while_stmt.body[0] = _assign(
            _name(loop, ast.Store()),
            ast.Subscript(value=_name(seq), slice=_name(idx), ctx=ast.Load()),
        )
        return [
            _assign(_name(seq, ast.Store()), ast.List(elts=[_const(x) for x in vals], ctx=ast.Load())),
            _assign(_name(idx, ast.Store()), _const(0)),
            while_stmt,
        ], 4

    def _p_conditional_flow(self, state, acc, params, n, names, k):
        i, j = self.rng.sample(range(n), 2)
        if params:
            lhs = _name(self.rng.choice(params))
        else:
            lhs = _sub(state, self.rng.randrange(n))
        rhs = _const(self.rng.randint(-self.config.magnitude, self.config.magnitude))
        op = self.rng.choice((ast.Lt(), ast.Gt(), ast.Eq(), ast.NotEq()))
        yes = self._update(_sub(state, i, ast.Store()), ast.Add(), _name(acc))
        no = self._update(_sub(state, j, ast.Store()), ast.Sub(), _name(acc))
        if self.rng.random() < 0.5:
            test = _compare(lhs, op, rhs)
            body, orelse = [yes], [no]
        else:
            test = ast.UnaryOp(op=ast.Not(), operand=_compare(lhs, op, rhs))
            body, orelse = [no], [yes]
        return [ast.If(test=test, body=body, orelse=orelse)], 2

    def _p_helper_chain(self, state, acc, params, n, names, k):
        f = names.take("fn")
        g = names.take("fn")
        i = self.rng.randrange(n)
        a, b = self._small_nonzero(), self._small_nonzero()
        fdef = self._unary_helper(f, _bin(_name("v"), ast.Add(), _const(a)))
        gdef = self._unary_helper(
            g, _bin(_call(f, _name("v")), self.rng.choice((ast.Add(), ast.Sub())), _const(b))
        )
        return [
            fdef,
            gdef,
            _assign(_sub(state, i, ast.Store()), _call(g, _sub(state, i))),
            self._update(_name(acc, ast.Store()), ast.Add(), _sub(state, i)),
        ], 4

    def _p_comprehension(self, state, acc, params, n, names, k):
        v = names.take("loop")
        d = self._small_nonzero()
        op = self.rng.choice((ast.Add(), ast.Sub()))
        elt = _bin(_name(v), op, _const(d))
        comp = ast.ListComp(
            elt=elt,
            generators=[
                ast.comprehension(
                    target=_name(v, ast.Store()),
                    iter=_name(state),
                    ifs=[],
                    is_async=0,
                )
            ],
        )
        return [
            _assign(_name(state, ast.Store()), comp),
            self._update(_name(acc, ast.Store()), ast.Add(), _sub(state, self.rng.randrange(n))),
        ], 2

    def _p_mapping_bridge(self, state, acc, params, n, names, k):
        table = names.take("map")
        i, j = self.rng.sample(range(n), 2)
        mapping = ast.Dict(
            keys=[_const(i), _const(j)],
            values=[_sub(state, i), _sub(state, j)],
        )
        get_i = ast.Subscript(value=_name(table), slice=_const(i), ctx=ast.Load())
        get_j_store = ast.Subscript(value=_name(table), slice=_const(j), ctx=ast.Store())
        get_j = ast.Subscript(value=_name(table), slice=_const(j), ctx=ast.Load())
        return [
            _assign(_name(table, ast.Store()), mapping),
            self._update(get_j_store, ast.Add(), _name(acc)),
            _assign(_sub(state, i, ast.Store()), _bin(get_i, ast.Add(), get_j)),
        ], 2

    def _hazard(self, kind, state, acc, params, n, names):
        if not params:
            return []
        x = _name(params[0])
        if kind == "index":
            tmp = names.take("tmp")
            return [
                _assign(_name(tmp, ast.Store()), ast.Subscript(value=_name(state), slice=x, ctx=ast.Load())),
                self._update(_name(acc, ast.Store()), ast.Add(), _name(tmp)),
            ]
        if kind == "division":
            den = names.take("tmp")
            return [
                _assign(_name(den, ast.Store()), _bin(x, ast.Sub(), _const(1))),
                self._update(
                    _name(acc, ast.Store()),
                    ast.Add(),
                    _bin(_sub(state, 0), ast.FloorDiv(), _name(den)),
                ),
            ]
        lookup = names.take("tmp")
        return [
            _assign(
                _name(lookup, ast.Store()),
                ast.List(elts=[_const(-1), _const(0), _const(1)], ctx=ast.Load()),
            ),
            self._update(
                _name(acc, ast.Store()),
                ast.Add(),
                ast.Call(
                    func=ast.Attribute(value=_name(lookup), attr="index", ctx=ast.Load()),
                    args=[x],
                    keywords=[],
                ),
            ),
        ]

    def _small_nonzero(self):
        mag = max(1, self.config.magnitude)
        return self.rng.choice([x for x in range(-mag, mag + 1) if x])

    def _update(self, target, op, value):
        if self.rng.random() < 0.5:
            return _aug(target, op, value)
        if isinstance(target, ast.Name):
            load = _name(target.id)
        elif isinstance(target, ast.Subscript):
            load = ast.Subscript(value=target.value, slice=target.slice, ctx=ast.Load())
        else:
            raise TypeError(f"unsupported update target: {type(target).__name__}")
        return _assign(target, _bin(load, op, value))

    def _unary_helper(self, name, expr):
        return ast.FunctionDef(
            name=name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="v")],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=[ast.Return(value=expr)],
            decorator_list=[],
        )

    def _features(self, module):
        nodes = list(ast.walk(module))
        return {
            "ast_nodes": len(nodes),
            "functions": sum(isinstance(x, ast.FunctionDef) for x in nodes),
            "calls": sum(isinstance(x, ast.Call) for x in nodes),
            "branches": sum(isinstance(x, ast.If) for x in nodes),
            "loops": sum(isinstance(x, (ast.For, ast.While, ast.comprehension)) for x in nodes),
            "mutations": sum(isinstance(x, (ast.AugAssign, ast.Subscript)) for x in nodes),
        }


def generate_imperative_mesopy(goal=None, config=None, seed=None):
    return ImperativeMesopy(config=config, seed=seed).generate(goal)
