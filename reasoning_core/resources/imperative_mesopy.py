"""Fast goal-directed Python synthesis for code reasoning tasks.

Programs are built recursively from typed AST constructors under explicit structural
budgets. The same generator supports execution, runnability, profiling, and other
tasks without giving each task a recognisably different source distribution.
"""

import ast
import random
import sys
import time
from dataclasses import dataclass, field


ERRORS = ("IndexError", "ZeroDivisionError", "ValueError", "KeyError", "RecursionError")
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
    "recursion",
)


@dataclass(frozen=True)
class MesopyComplexity:
    statements: int = 10
    expr_depth: int = 3
    control_depth: int = 2
    functions: int = 3
    call_depth: int = 2
    dataflow_depth: int = 5
    loop_bound: int = 4

    @classmethod
    def level(cls, level):
        level = max(0, int(level))
        return cls(
            statements=8 + 3 * level,
            expr_depth=2 + level // 2,
            control_depth=1 + level // 2,
            functions=1 + level // 2,
            call_depth=1 + level // 2,
            dataflow_depth=3 + level,
            loop_bound=3 + level // 2,
        )


@dataclass(frozen=True)
class MesopyGoal:
    runnable: bool | None = True
    paired_runnability: bool = False
    error: str | None = None
    result_kind: str | None = None
    input_arity: int | None = None
    phenomena: tuple[str, ...] = ()
    complexity: MesopyComplexity | None = None
    allow_recursion: bool = True
    require_recursion: bool = False


@dataclass
class MesopyConfig:
    magnitude: int = 6
    list_size: tuple[int, int] = (3, 7)
    input_arity: tuple[int, int] = (1, 3)
    complexity: MesopyComplexity = field(default_factory=MesopyComplexity)
    safe_hazard_rate: float = 0.4
    recursion_rate: float = 0.3
    phenomena_rate: float = 0.55
    max_attempts: int = 12


@dataclass(frozen=True)
class CallOutcome:
    args: tuple
    ok: bool
    value: str | None = None
    error: str | None = None
    steps: int | None = None
    elapsed: float | None = None


@dataclass
class MesopySample:
    code: str
    calls: tuple[CallOutcome, ...]
    phenomena: tuple[str, ...]
    features: dict

    @property
    def call(self):
        return self.calls[0]

    @property
    def args(self):
        return self.call.args

    @property
    def answer(self):
        return self.call.value if self.call.ok else self.call.error


@dataclass
class _Env:
    ints: list[str] = field(default_factory=list)
    lists: dict[str, int] = field(default_factory=dict)
    funcs: list[str] = field(default_factory=list)
    depth: dict[str, int] = field(default_factory=dict)

    def copy(self):
        return _Env(self.ints[:], dict(self.lists), self.funcs[:], dict(self.depth))

    def add_int(self, name, depth=1):
        if name not in self.ints:
            self.ints.append(name)
        self.depth[name] = depth

    def add_list(self, name, length, depth=1):
        self.lists[name] = length
        self.depth[name] = depth


class _Names:
    def __init__(self):
        self.n = 0

    def take(self, prefix="v"):
        self.n += 1
        return f"{prefix}{self.n}"


class ImperativeMesopy:
    def __init__(self, config=None, seed=None):
        self.config = config or MesopyConfig()
        self.rng = random.Random(seed)
        self.names = _Names()

    def generate(self, goal=None):
        goal = goal or MesopyGoal()
        if goal.error is not None and goal.error not in ERRORS:
            raise ValueError(f"supported errors: {', '.join(ERRORS)}")
        if goal.require_recursion and not goal.allow_recursion:
            raise ValueError("require_recursion needs allow_recursion=True")
        for _ in range(self.config.max_attempts):
            sample = self._generate_once(goal)
            if self._valid(sample, goal):
                return sample
        raise RuntimeError(f"failed to generate imperative Mesopy sample: {goal}")

    def execution(self, **kwargs):
        return self.generate(MesopyGoal(runnable=True, **kwargs))

    def runnability_pair(self, **kwargs):
        return self.generate(MesopyGoal(runnable=None, paired_runnability=True, **kwargs))

    def profile(self, sample, call=0, max_steps=100_000):
        outcome = sample.calls[call]
        return self._execute(sample.code, outcome.args, profile=True, max_steps=max_steps)

    def _generate_once(self, goal):
        self.names = _Names()
        rng, cfg = self.rng, self.config
        cx = goal.complexity or cfg.complexity
        needs_input = goal.paired_runnability or goal.runnable is False or goal.error is not None
        arity = goal.input_arity
        if arity is None:
            lo, hi = cfg.input_arity
            arity = rng.randint(max(lo, int(needs_input)), max(hi, int(needs_input)))
        arity = max(1 if needs_input else 0, arity)
        params = [f"x{i}" for i in range(arity)]
        env = _Env()
        for p in params:
            env.add_int(p, 0)

        recursive = goal.allow_recursion and (
            goal.require_recursion
            or goal.error == "RecursionError"
            or "recursion" in goal.phenomena
            or rng.random() < cfg.recursion_rate
        )
        helpers, helper_names = self._gen_helpers(cx, recursive)
        env.funcs.extend(helper_names)

        n = rng.randint(*cfg.list_size)
        state = self.names.take("xs")
        init = [
            self._gen_int_expr(env, cx.expr_depth, helper_names, force_dep=bool(params))
            for _ in range(n)
        ]
        body = [ast.Assign([ast.Name(state, ast.Store())], ast.List(init, ast.Load()))]
        env.add_list(state, n, 1)

        acc = self.names.take("a")
        body.append(ast.Assign(
            [ast.Name(acc, ast.Store())],
            self._gen_int_expr(env, max(1, cx.expr_depth - 1), helper_names),
        ))
        env.add_int(acc, 1)

        phenomena = set(goal.phenomena)
        unknown = phenomena - set(PHENOMENA)
        if unknown:
            raise ValueError(f"unknown phenomena: {sorted(unknown)}")
        if recursive:
            phenomena.add("recursion")

        target_stmts = max(2, cx.statements)
        for _ in range(target_stmts):
            stmts, tags = self._gen_stmt(
                env, cx.control_depth, cx.expr_depth, cx, state, helper_names
            )
            body.extend(stmts)
            phenomena.update(tags)

        for phenomenon in [p for p in goal.phenomena if p not in phenomena]:
            body.extend(self._inject_phenomenon(phenomenon, env, state, helper_names, cx))
            phenomena.add(phenomenon)

        while max(env.depth.values(), default=0) < cx.dataflow_depth:
            src = max(env.ints, key=lambda x: env.depth.get(x, 0))
            name = self.names.take("flow")
            env.add_int(name, env.depth.get(src, 0) + 1)
            body.append(ast.Assign(
                [ast.Name(name, ast.Store())],
                ast.BinOp(
                    ast.Name(src, ast.Load()),
                    rng.choice((ast.Add(), ast.Sub())),
                    ast.Constant(rng.choice((-2, -1, 1, 2))),
                ),
            ))

        hazard = None
        if needs_input or rng.random() < cfg.safe_hazard_rate:
            hazard = self._choose_hazard(goal, recursive)
            body.extend(self._hazard_nodes(hazard, params, env, state, n, helper_names))

        result_kind = goal.result_kind or rng.choice(("int", "list", "tuple"))
        body.append(ast.Return(self._result_expr(result_kind, env, state, helper_names, cx.expr_depth)))
        endpoint = ast.FunctionDef(
            "endpoint",
            ast.arguments([], [ast.arg(p) for p in params], None, [], [], None, []),
            body,
            [],
        )
        module = ast.fix_missing_locations(ast.Module(helpers + [endpoint], []))
        code = ast.unparse(module) + "\n"

        if goal.paired_runnability:
            safe_args = self._safe_args(arity, hazard, n)
            bad_args = self._bad_args(arity, hazard, n)
            calls = [self._execute(code, safe_args), self._execute(code, bad_args)]
            rng.shuffle(calls)
        else:
            args = (
                self._bad_args(arity, hazard, n)
                if needs_input
                else self._safe_args(arity, hazard, n)
            )
            calls = [self._execute(code, args)]

        features = self._features(module)
        features.update(
            dataflow_depth=max(env.depth.values(), default=0),
            requested_statements=target_stmts,
            hazard=hazard,
            result_kind=result_kind,
            recursive=recursive,
            input_arity=arity,
        )
        return MesopySample(code, tuple(calls), tuple(sorted(phenomena)), features)

    def _gen_helpers(self, cx, recursive):
        helpers = []
        names = []
        for i in range(max(0, cx.functions)):
            name = f"f{i}"
            x = ast.Name("x", ast.Load())
            env = _Env(["x", "y"], {}, names[:], {"x": 0, "y": 0})
            expr = self._gen_int_expr(env, cx.expr_depth, names[:], force_dep=True)
            if i and i < cx.call_depth + 1:
                prev = ast.Call(ast.Name(names[-1], ast.Load()), [x, ast.Name("y", ast.Load())], [])
                expr = ast.BinOp(prev, self.rng.choice((ast.Add(), ast.Sub())), expr)
            helpers.append(ast.FunctionDef(
                name,
                ast.arguments([], [ast.arg("x"), ast.arg("y")], None, [], [], None, []),
                [ast.Return(expr)],
                [],
            ))
            names.append(name)

        if recursive:
            n = ast.Name("n", ast.Load())
            z = ast.Name("z", ast.Load())
            base = ast.If(
                ast.Compare(n, [ast.Eq()], [ast.Constant(0)]),
                [ast.Return(z)],
                [],
            )
            step = ast.Call(
                ast.Name("rec", ast.Load()),
                [
                    ast.BinOp(n, ast.Sub(), ast.Constant(1)),
                    ast.BinOp(z, ast.Add(), n),
                ],
                [],
            )
            helpers.append(ast.FunctionDef(
                "rec",
                ast.arguments([], [ast.arg("n"), ast.arg("z")], None, [], [], None, []),
                [base, ast.Return(step)],
                [],
            ))
            names.append("rec")
        return helpers, names

    def _gen_int_expr(self, env, depth, helpers, force_dep=False):
        rng = self.rng
        if depth <= 0:
            if env.ints and (force_dep or rng.random() < 0.72):
                return ast.Name(rng.choice(env.ints), ast.Load())
            return ast.Constant(rng.randint(-self.config.magnitude, self.config.magnitude))

        choices = ["leaf", "bin", "ternary"]
        if env.lists:
            choices += ["index", "length"]
        if helpers:
            choices += ["call"]
        kind = rng.choice(choices)
        if kind == "leaf":
            return self._gen_int_expr(env, 0, helpers, force_dep)
        if kind == "bin":
            left = self._gen_int_expr(env, depth - 1, helpers, force_dep)
            right = self._gen_int_expr(env, depth - 1, helpers)
            op = rng.choice((ast.Add(), ast.Sub(), ast.Mult(), ast.Mod()))
            if isinstance(op, ast.Mod):
                right = ast.BinOp(
                    ast.Call(ast.Name("abs", ast.Load()), [right], []),
                    ast.Add(),
                    ast.Constant(1),
                )
            elif isinstance(op, ast.Mult):
                right = ast.Constant(rng.choice((-3, -2, -1, 1, 2, 3)))
            return ast.BinOp(left, op, right)
        if kind == "ternary":
            left = self._gen_int_expr(env, depth - 1, helpers, force_dep)
            right = self._gen_int_expr(env, max(0, depth - 2), helpers)
            return ast.IfExp(
                ast.Compare(left, [rng.choice((ast.Lt(), ast.GtE(), ast.NotEq()))], [right]),
                self._gen_int_expr(env, depth - 1, helpers),
                self._gen_int_expr(env, depth - 1, helpers),
            )
        if kind == "index":
            name, length = rng.choice(list(env.lists.items()))
            raw = self._gen_int_expr(env, depth - 1, helpers, force_dep)
            idx = ast.BinOp(
                ast.Call(ast.Name("abs", ast.Load()), [raw], []),
                ast.Mod(),
                ast.Constant(length),
            )
            return ast.Subscript(ast.Name(name, ast.Load()), idx, ast.Load())
        if kind == "length":
            name = rng.choice(list(env.lists))
            return ast.Call(ast.Name("len", ast.Load()), [ast.Name(name, ast.Load())], [])

        fn = rng.choice(helpers)
        if fn == "rec":
            a = ast.BinOp(
                ast.Call(
                    ast.Name("abs", ast.Load()),
                    [self._gen_int_expr(env, depth - 1, helpers)],
                    [],
                ),
                ast.Mod(),
                ast.Constant(max(2, self.config.magnitude)),
            )
            b = self._gen_int_expr(env, depth - 1, helpers, force_dep)
            return ast.Call(ast.Name(fn, ast.Load()), [a, b], [])
        return ast.Call(
            ast.Name(fn, ast.Load()),
            [
                self._gen_int_expr(env, depth - 1, helpers, force_dep),
                self._gen_int_expr(env, depth - 1, helpers),
            ],
            [],
        )

    def _gen_stmt(self, env, control_depth, expr_depth, cx, state, helpers):
        rng = self.rng
        kinds = ["assign", "aug", "mutate", "alias", "helper"]
        if control_depth > 0:
            kinds += ["if", "for", "while"]
        kind = rng.choice(kinds)
        tags = set()

        if kind == "assign":
            name = self.names.take("v")
            expr = self._gen_int_expr(env, expr_depth, helpers, force_dep=True)
            dep = 1 + max((env.depth.get(v, 0) for v in env.ints), default=0)
            env.add_int(name, dep)
            return [ast.Assign([ast.Name(name, ast.Store())], expr)], tags

        if kind == "aug":
            target = rng.choice(env.ints)
            expr = self._gen_int_expr(env, max(0, expr_depth - 1), helpers)
            env.depth[target] = env.depth.get(target, 0) + 1
            return [
                ast.AugAssign(
                    ast.Name(target, ast.Store()),
                    rng.choice((ast.Add(), ast.Sub())),
                    expr,
                )
            ], tags

        if kind == "mutate":
            list_name, length = rng.choice(list(env.lists.items()))
            expr = self._gen_int_expr(env, max(0, expr_depth - 1), helpers)
            env.depth[list_name] = env.depth.get(list_name, 0) + 1
            tags.add("mutation_call")
            if rng.random() < 0.5:
                node = ast.AugAssign(
                    ast.Subscript(
                        ast.Name(list_name, ast.Load()),
                        ast.Constant(rng.randrange(length)),
                        ast.Store(),
                    ),
                    ast.Add(),
                    expr,
                )
            else:
                node = ast.Expr(ast.Call(
                    ast.Attribute(ast.Name(list_name, ast.Load()), "append", ast.Load()),
                    [expr],
                    [],
                ))
                env.lists[list_name] += 1
            return [node], tags

        if kind == "alias":
            src, length = rng.choice(list(env.lists.items()))
            alias = self.names.take("alias")
            env.add_list(alias, length, env.depth.get(src, 0))
            tags.add("aliasing")
            nodes = [ast.Assign([ast.Name(alias, ast.Store())], ast.Name(src, ast.Load()))]
            if rng.random() < 0.5:
                nodes.append(ast.AugAssign(
                    ast.Subscript(
                        ast.Name(alias, ast.Load()),
                        ast.Constant(rng.randrange(length)),
                        ast.Store(),
                    ),
                    ast.Add(),
                    ast.Constant(rng.choice((-2, -1, 1, 2))),
                ))
            else:
                copy = self.names.take("copy")
                nodes.append(ast.Assign(
                    [ast.Name(copy, ast.Store())],
                    ast.Subscript(
                        ast.Name(alias, ast.Load()),
                        ast.Slice(None, None, None),
                        ast.Load(),
                    ),
                ))
                env.add_list(copy, length, env.depth.get(src, 0) + 1)
                tags.add("rebinding_vs_aliasing")
            return nodes, tags

        if kind == "helper":
            if not helpers:
                return self._gen_stmt(env, 0, expr_depth, cx, state, helpers)
            name = self.names.take("h")
            expr = self._gen_int_expr(env, expr_depth, helpers, force_dep=True)
            env.add_int(name, max(env.depth.values(), default=0) + 1)
            tags.add("helper_chain")
            return [ast.Assign([ast.Name(name, ast.Store())], expr)], tags

        if kind == "if":
            tags.add("conditional_flow")
            cond = ast.Compare(
                self._gen_int_expr(env, max(0, expr_depth - 1), helpers, True),
                [rng.choice((ast.Lt(), ast.Gt(), ast.NotEq()))],
                [self._gen_int_expr(env, max(0, expr_depth - 1), helpers)],
            )
            then_env, else_env = env.copy(), env.copy()
            then_nodes, then_tags = self._gen_stmt(
                then_env, control_depth - 1, expr_depth, cx, state, helpers
            )
            else_nodes, else_tags = self._gen_stmt(
                else_env, control_depth - 1, expr_depth, cx, state, helpers
            )
            tags.update(then_tags | else_tags)
            return [ast.If(cond, then_nodes, else_nodes)], tags

        if kind == "for":
            tags.add("loop_carried_state")
            loop = self.names.take("i")
            loop_env = env.copy()
            loop_env.add_int(loop, 0)
            inner, inner_tags = self._gen_stmt(
                loop_env, control_depth - 1, expr_depth, cx, state, helpers
            )
            tags.update(inner_tags)
            bound = rng.randint(1, max(1, cx.loop_bound))
            return [ast.For(
                ast.Name(loop, ast.Store()),
                ast.Call(ast.Name("range", ast.Load()), [ast.Constant(bound)], []),
                inner,
                [],
            )], tags

        tags.add("loop_carried_state")
        counter = self.names.take("i")
        limit = rng.randint(1, max(1, cx.loop_bound))
        loop_env = env.copy()
        inner, inner_tags = self._gen_stmt(
            loop_env, control_depth - 1, expr_depth, cx, state, helpers
        )
        env.add_int(counter, 0)
        tags.update(inner_tags)
        update = ast.AugAssign(ast.Name(counter, ast.Store()), ast.Add(), ast.Constant(1))
        return [
            ast.Assign([ast.Name(counter, ast.Store())], ast.Constant(0)),
            ast.While(
                ast.Compare(ast.Name(counter, ast.Load()), [ast.Lt()], [ast.Constant(limit)]),
                inner + [update],
                [],
            ),
        ], tags

    def _inject_phenomenon(self, phenomenon, env, state, helpers, cx):
        if phenomenon == "aliasing":
            alias = self.names.take("alias")
            env.add_list(alias, env.lists[state], env.depth.get(state, 0))
            return [
                ast.Assign([ast.Name(alias, ast.Store())], ast.Name(state, ast.Load())),
                ast.AugAssign(
                    ast.Subscript(ast.Name(alias, ast.Load()), ast.Constant(0), ast.Store()),
                    ast.Add(),
                    ast.Constant(1),
                ),
            ]
        if phenomenon == "rebinding_vs_aliasing":
            alias = self.names.take("alias")
            env.add_list(alias, env.lists[state], env.depth.get(state, 0))
            return [
                ast.Assign([ast.Name(alias, ast.Store())], ast.Name(state, ast.Load())),
                ast.Assign(
                    [ast.Name(state, ast.Store())],
                    ast.Subscript(
                        ast.Name(state, ast.Load()),
                        ast.Slice(None, None, None),
                        ast.Load(),
                    ),
                ),
            ]
        if phenomenon == "closure_late_binding":
            bias = self.names.take("bias")
            fn = self.names.take("closure")
            env.add_int(bias, 1)
            return [
                ast.Assign([ast.Name(bias, ast.Store())], ast.Constant(2)),
                ast.FunctionDef(
                    fn,
                    ast.arguments([], [ast.arg("z")], None, [], [], None, []),
                    [ast.Return(ast.BinOp(
                        ast.Name("z", ast.Load()), ast.Add(), ast.Name(bias, ast.Load())
                    ))],
                    [],
                ),
                ast.AugAssign(ast.Name(bias, ast.Store()), ast.Add(), ast.Constant(1)),
                ast.Assign(
                    [ast.Name(bias, ast.Store())],
                    ast.Call(ast.Name(fn, ast.Load()), [ast.Name(bias, ast.Load())], []),
                ),
            ]
        if phenomenon == "default_capture":
            bias = self.names.take("bias")
            fn = self.names.take("default")
            out = self.names.take("v")
            env.add_int(bias, 1)
            env.add_int(out, 2)
            return [
                ast.Assign([ast.Name(bias, ast.Store())], ast.Constant(2)),
                ast.FunctionDef(
                    fn,
                    ast.arguments(
                        [], [ast.arg("z"), ast.arg("b")], None, [], [], None,
                        [ast.Name(bias, ast.Load())],
                    ),
                    [ast.Return(ast.BinOp(
                        ast.Name("z", ast.Load()), ast.Add(), ast.Name("b", ast.Load())
                    ))],
                    [],
                ),
                ast.AugAssign(ast.Name(bias, ast.Store()), ast.Add(), ast.Constant(1)),
                ast.Assign(
                    [ast.Name(out, ast.Store())],
                    ast.Call(ast.Name(fn, ast.Load()), [ast.Name(bias, ast.Load())], []),
                ),
            ]
        if phenomenon == "mutation_call":
            fn = self.names.take("mut")
            return [
                ast.FunctionDef(
                    fn,
                    ast.arguments([], [ast.arg("ys")], None, [], [], None, []),
                    [
                        ast.AugAssign(
                            ast.Subscript(
                                ast.Name("ys", ast.Load()), ast.Constant(0), ast.Store()
                            ),
                            ast.Add(), ast.Constant(1),
                        ),
                        ast.Return(ast.Subscript(
                            ast.Name("ys", ast.Load()), ast.Constant(0), ast.Load()
                        )),
                    ],
                    [],
                ),
                ast.Expr(ast.Call(ast.Name(fn, ast.Load()), [ast.Name(state, ast.Load())], [])),
            ]
        if phenomenon == "loop_carried_state":
            v = self.names.take("i")
            target = env.ints[0]
            return [ast.For(
                ast.Name(v, ast.Store()),
                ast.Call(ast.Name("range", ast.Load()), [ast.Constant(max(2, cx.loop_bound))], []),
                [ast.AugAssign(
                    ast.Name(target, ast.Store()), ast.Add(), ast.Name(v, ast.Load())
                )],
                [],
            )]
        if phenomenon == "conditional_flow":
            target = env.ints[0]
            return [ast.If(
                ast.Compare(ast.Name(target, ast.Load()), [ast.GtE()], [ast.Constant(0)]),
                [ast.AugAssign(ast.Name(target, ast.Store()), ast.Add(), ast.Constant(1))],
                [ast.AugAssign(ast.Name(target, ast.Store()), ast.Sub(), ast.Constant(1))],
            )]
        if phenomenon == "helper_chain":
            name = self.names.take("v")
            env.add_int(name, max(env.depth.values(), default=0) + 1)
            return [ast.Assign(
                [ast.Name(name, ast.Store())],
                self._gen_int_expr(env, cx.expr_depth, helpers, True),
            )]
        if phenomenon == "comprehension":
            name = self.names.take("lc")
            v = self.names.take("i")
            length = max(2, cx.loop_bound)
            env.add_list(name, length, 2)
            return [ast.Assign(
                [ast.Name(name, ast.Store())],
                ast.ListComp(
                    ast.BinOp(ast.Name(v, ast.Load()), ast.Mult(), ast.Name(v, ast.Load())),
                    [ast.comprehension(
                        ast.Name(v, ast.Store()),
                        ast.Call(ast.Name("range", ast.Load()), [ast.Constant(length)], []),
                        [], 0,
                    )],
                ),
            )]
        if phenomenon == "mapping_bridge":
            name = self.names.take("d")
            out = self.names.take("v")
            env.add_int(out, 2)
            return [
                ast.Assign(
                    [ast.Name(name, ast.Store())],
                    ast.Dict(
                        [ast.Constant(0), ast.Constant(1)],
                        [ast.Name(env.ints[0], ast.Load()), ast.Name(env.ints[-1], ast.Load())],
                    ),
                ),
                ast.Assign(
                    [ast.Name(out, ast.Store())],
                    ast.Subscript(
                        ast.Name(name, ast.Load()),
                        ast.Constant(self.rng.randrange(2)),
                        ast.Load(),
                    ),
                ),
            ]
        if phenomenon == "recursion":
            return []
        raise ValueError(phenomenon)

    def _choose_hazard(self, goal, recursive):
        if goal.error:
            return goal.error
        choices = list(ERRORS[:-1])
        if recursive:
            choices.append("RecursionError")
        return self.rng.choice(choices)

    def _hazard_nodes(self, hazard, params, env, state, n, helpers):
        if not params:
            return []
        x = ast.Name(params[0], ast.Load())
        out = self.names.take("haz")
        if hazard == "IndexError":
            expr = ast.Subscript(ast.Name(state, ast.Load()), x, ast.Load())
        elif hazard == "ZeroDivisionError":
            expr = ast.BinOp(
                self._gen_int_expr(env, 1, helpers),
                ast.FloorDiv(),
                ast.BinOp(x, ast.Sub(), ast.Constant(1)),
            )
        elif hazard == "ValueError":
            expr = ast.Call(
                ast.Attribute(
                    ast.List([ast.Constant(i) for i in range(n)], ast.Load()),
                    "index",
                    ast.Load(),
                ),
                [x],
                [],
            )
        elif hazard == "KeyError":
            expr = ast.Subscript(
                ast.Dict(
                    [ast.Constant(i) for i in range(n)],
                    [ast.Constant(i * i + 1) for i in range(n)],
                ),
                x,
                ast.Load(),
            )
        else:
            expr = ast.Call(
                ast.Name("rec", ast.Load()),
                [x, self._gen_int_expr(env, 1, helpers)],
                [],
            )
        env.add_int(out, max(env.depth.values(), default=0) + 1)
        return [ast.Assign([ast.Name(out, ast.Store())], expr)]

    def _safe_args(self, arity, hazard, n):
        if arity == 0:
            return ()
        mag = self.config.magnitude
        xs = [self.rng.randint(-mag, mag) for _ in range(arity)]
        if hazard in ("IndexError", "ValueError", "KeyError"):
            xs[0] = self.rng.randrange(n)
        elif hazard == "ZeroDivisionError":
            xs[0] = self.rng.choice([x for x in range(-mag, mag + 1) if x != 1])
        elif hazard == "RecursionError":
            xs[0] = self.rng.randint(0, max(1, mag))
        return tuple(xs)

    def _bad_args(self, arity, hazard, n):
        xs = list(self._safe_args(arity, hazard, n))
        if not xs:
            return ()
        if hazard in ("IndexError", "ValueError", "KeyError"):
            xs[0] = n + self.rng.randint(1, 4)
        elif hazard == "ZeroDivisionError":
            xs[0] = 1
        elif hazard == "RecursionError":
            xs[0] = -1
        return tuple(xs)

    def _result_expr(self, kind, env, state, helpers, depth):
        if kind == "list":
            return ast.Name(state, ast.Load())
        if kind == "tuple":
            return ast.Tuple(
                [
                    self._gen_int_expr(env, min(depth, 2), helpers, True),
                    ast.Call(ast.Name("len", ast.Load()), [ast.Name(state, ast.Load())], []),
                ],
                ast.Load(),
            )
        return self._gen_int_expr(env, depth, helpers, True)

    def _execute(self, code, args, profile=False, max_steps=100_000):
        builtins = {
            "range": range,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
        }
        ns = {"__builtins__": builtins}
        steps = 0

        def trace(frame, event, arg):
            nonlocal steps
            if event == "line":
                steps += 1
                if steps > max_steps:
                    raise RuntimeError("StepLimit")
            return trace

        t0 = time.perf_counter()
        try:
            exec(compile(code, "<imperative-mesopy>", "exec"), ns, ns)
            if profile:
                sys.settrace(trace)
            value = ns["endpoint"](*args)
            return CallOutcome(
                tuple(args), True, repr(value), None,
                steps if profile else None,
                time.perf_counter() - t0 if profile else None,
            )
        except Exception as e:
            return CallOutcome(
                tuple(args), False, None, type(e).__name__,
                steps if profile else None,
                time.perf_counter() - t0 if profile else None,
            )
        finally:
            if profile:
                sys.settrace(None)

    def _valid(self, sample, goal):
        if goal.paired_runnability:
            if len(sample.calls) != 2 or {x.ok for x in sample.calls} != {True, False}:
                return False
            return not goal.error or any(x.error == goal.error for x in sample.calls)
        if goal.runnable is True:
            return sample.call.ok
        if goal.runnable is False:
            return not sample.call.ok and (
                goal.error is None or sample.call.error == goal.error
            )
        return True

    @staticmethod
    def _features(tree):
        nodes = list(ast.walk(tree))

        def depth(node):
            children = list(ast.iter_child_nodes(node))
            return 1 + max(map(depth, children), default=0)

        def control_depth(node, current=0):
            here = current + int(isinstance(node, (ast.If, ast.For, ast.While, ast.Try)))
            return max([here] + [control_depth(c, here) for c in ast.iter_child_nodes(node)])

        funcs = [n for n in nodes if isinstance(n, ast.FunctionDef)]
        names = {f.name for f in funcs}
        edges = set()
        n_calls = 0

        class Calls(ast.NodeVisitor):
            current = None

            def visit_FunctionDef(self, node):
                prev, self.current = self.current, node.name
                self.generic_visit(node)
                self.current = prev

            def visit_Call(self, node):
                nonlocal n_calls
                n_calls += 1
                if self.current and isinstance(node.func, ast.Name) and node.func.id in names:
                    edges.add((self.current, node.func.id))
                self.generic_visit(node)

        Calls().visit(tree)

        def longest(src, seen):
            nxt = [b for a, b in edges if a == src and b not in seen]
            return 1 + max((longest(x, seen | {x}) for x in nxt), default=0)

        return {
            "ast_nodes": len(nodes),
            "ast_depth": depth(tree),
            "control_depth": control_depth(tree),
            "functions": len(funcs),
            "call_depth": max((longest(f.name, {f.name}) for f in funcs), default=0),
            "loops": sum(isinstance(n, (ast.For, ast.While)) for n in nodes),
            "branches": sum(isinstance(n, ast.If) for n in nodes),
            "calls": n_calls,
        }
