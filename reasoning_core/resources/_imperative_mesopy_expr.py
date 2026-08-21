from __future__ import annotations

import ast

from ._imperative_mesopy_types import BOOL, DICT_INT, INT, LIST_INT, ExprSpec, Risk, _Env


def _combine_int(a: ExprSpec, b: ExprSpec, op: ast.operator) -> ExprSpec:
    return ExprSpec(
        ast.BinOp(a.node, op, b.node),
        INT,
        a.deps | b.deps,
        a.risks + b.risks,
        1 + max(a.depth, b.depth),
    )


class _ExpressionMixin:
    def _leaf_int(self, env: _Env, required: set[str] | None = None) -> ExprSpec:
        required = set(required or ())
        candidates = env.names(INT)
        if required:
            covering = [n for n in candidates if required <= set(env.info(n).deps)]
            if covering:
                name = self.rng.choice(covering)
                info = env.info(name)
                return ExprSpec(ast.Name(name, ast.Load()), INT, info.deps, (), info.depth)
            param = self.rng.choice(sorted(required))
            if param in env.vars:
                info = env.info(param)
                return ExprSpec(ast.Name(param, ast.Load()), INT, info.deps, (), info.depth)
        if candidates and self.rng.random() < 0.76:
            name = self.rng.choice(candidates)
            info = env.info(name)
            return ExprSpec(ast.Name(name, ast.Load()), INT, info.deps, (), info.depth)
        return ExprSpec(ast.Constant(self.rng.randint(-self.config.magnitude, self.config.magnitude)), INT)

    def _gen_int_expr(
        self,
        env: _Env,
        depth: int,
        required: set[str] | None = None,
    ) -> ExprSpec:
        required = set(required or ())
        if depth <= 0:
            if len(required) <= 1:
                return self._leaf_int(env, required)
            deps = sorted(required)
            mid = max(1, len(deps) // 2)
            left = self._gen_int_expr(env, 0, set(deps[:mid]))
            right = self._gen_int_expr(env, 0, set(deps[mid:]))
            return _combine_int(left, right, ast.Add())

        kinds = ["leaf", "bin", "ternary", "call"]
        if env.names(LIST_INT):
            kinds += ["index", "length", "list_index"]
        if env.names(DICT_INT):
            kinds += ["dict_get"]
        kind = self.rng.choice(kinds)
        if required and kind in {"index", "list_index", "dict_get", "call"} and self.rng.random() < 0.45:
            kind = "bin"

        if kind == "leaf":
            return self._leaf_int(env, required)

        if kind == "bin":
            deps = list(required)
            self.rng.shuffle(deps)
            cut = self.rng.randint(0, len(deps)) if deps else 0
            left = self._gen_int_expr(env, depth - 1, set(deps[:cut]))
            right = self._gen_int_expr(env, depth - 1, set(deps[cut:]))
            op = self.rng.choice((ast.Add(), ast.Sub(), ast.Mult(), ast.FloorDiv(), ast.Mod()))
            risks = left.risks + right.risks
            right_node = right.node
            if isinstance(op, (ast.FloorDiv, ast.Mod)):
                if not self._take_risk():
                    right_node = ast.BinOp(
                        ast.Call(ast.Name("abs", ast.Load()), [right.node], []),
                        ast.Add(),
                        ast.Constant(1),
                    )
                else:
                    risks += (Risk("ZeroDivisionError", right.deps),)
            elif isinstance(op, ast.Mult) and self.rng.random() < 0.55:
                right_node = ast.Constant(self.rng.choice((-3, -2, -1, 1, 2, 3)))
            return ExprSpec(
                ast.BinOp(left.node, op, right_node),
                INT,
                left.deps | right.deps,
                risks,
                1 + max(left.depth, right.depth),
            )

        if kind == "ternary":
            cond = self._gen_bool_expr(env, depth - 1)
            yes = self._gen_int_expr(env, depth - 1, required)
            no = self._gen_int_expr(env, depth - 1, required)
            return ExprSpec(
                ast.IfExp(cond.node, yes.node, no.node),
                INT,
                cond.deps | yes.deps | no.deps,
                cond.risks + yes.risks + no.risks,
                1 + max(cond.depth, yes.depth, no.depth),
            )

        if kind == "call" and env.functions:
            fn = self.rng.choice(env.functions)
            if fn.startswith("recur"):
                seed = self._gen_int_expr(env, max(0, depth - 1), required)
                rank_raw = self._gen_int_expr(env, max(0, depth - 1))
                rank = ast.BinOp(
                    ast.Call(ast.Name("abs", ast.Load()), [rank_raw.node], []),
                    ast.Mod(),
                    ast.Constant(max(2, self.config.magnitude)),
                )
                return ExprSpec(
                    ast.Call(ast.Name(fn, ast.Load()), [rank, seed.node], []),
                    INT,
                    seed.deps | rank_raw.deps,
                    seed.risks + rank_raw.risks,
                    1 + max(seed.depth, rank_raw.depth),
                )
            a = self._gen_int_expr(env, depth - 1, required)
            b = self._gen_int_expr(env, depth - 1)
            return ExprSpec(
                ast.Call(ast.Name(fn, ast.Load()), [a.node, b.node], []),
                INT,
                a.deps | b.deps,
                a.risks + b.risks,
                1 + max(a.depth, b.depth),
            )

        if kind == "index":
            name = self.rng.choice(env.names(LIST_INT))
            info = env.info(name)
            idx = self._gen_int_expr(env, depth - 1, required)
            node = idx.node
            risks = idx.risks
            if not self._take_risk() and info.length:
                node = ast.BinOp(
                    ast.Call(ast.Name("abs", ast.Load()), [idx.node], []),
                    ast.Mod(),
                    ast.Constant(info.length),
                )
            else:
                risks += (Risk("IndexError", idx.deps),)
            return ExprSpec(
                ast.Subscript(ast.Name(name, ast.Load()), node, ast.Load()),
                INT,
                info.deps | idx.deps,
                risks,
                max(info.depth, idx.depth) + 1,
            )

        if kind == "length":
            name = self.rng.choice(env.names(LIST_INT))
            info = env.info(name)
            return ExprSpec(
                ast.Call(ast.Name("len", ast.Load()), [ast.Name(name, ast.Load())], []),
                INT,
                info.deps,
                (),
                info.depth + 1,
            )

        if kind == "list_index":
            name = self.rng.choice(env.names(LIST_INT))
            info = env.info(name)
            value = self._gen_int_expr(env, depth - 1, required)
            if not self._take_risk() and info.length:
                value_node = ast.Subscript(
                    ast.Name(name, ast.Load()),
                    ast.Constant(self.rng.randrange(info.length)),
                    ast.Load(),
                )
                risks = value.risks
            else:
                value_node = value.node
                risks = value.risks + (Risk("ValueError", value.deps),)
            return ExprSpec(
                ast.Call(ast.Attribute(ast.Name(name, ast.Load()), "index", ast.Load()), [value_node], []),
                INT,
                info.deps | value.deps,
                risks,
                max(info.depth, value.depth) + 1,
            )

        if kind == "dict_get" and env.names(DICT_INT):
            name = self.rng.choice(env.names(DICT_INT))
            info = env.info(name)
            key = self._gen_int_expr(env, depth - 1, required)
            if not self._take_risk():
                node = ast.Call(
                    ast.Attribute(ast.Name(name, ast.Load()), "get", ast.Load()),
                    [key.node, ast.Constant(0)],
                    [],
                )
                risks = key.risks
            else:
                node = ast.Subscript(ast.Name(name, ast.Load()), key.node, ast.Load())
                risks = key.risks + (Risk("KeyError", key.deps),)
            return ExprSpec(node, INT, info.deps | key.deps, risks, max(info.depth, key.depth) + 1)

        return self._leaf_int(env, required)

    def _gen_bool_expr(self, env: _Env, depth: int) -> ExprSpec:
        if depth <= 0 or self.rng.random() < 0.55:
            a = self._gen_int_expr(env, max(0, depth - 1))
            b = self._gen_int_expr(env, max(0, depth - 1))
            return ExprSpec(
                ast.Compare(a.node, [self.rng.choice((ast.Lt(), ast.LtE(), ast.Gt(), ast.GtE(), ast.Eq(), ast.NotEq()))], [b.node]),
                BOOL,
                a.deps | b.deps,
                a.risks + b.risks,
                1 + max(a.depth, b.depth),
            )
        a = self._gen_bool_expr(env, depth - 1)
        b = self._gen_bool_expr(env, depth - 1)
        return ExprSpec(
            ast.BoolOp(self.rng.choice((ast.And(), ast.Or())), [a.node, b.node]),
            BOOL,
            a.deps | b.deps,
            a.risks + b.risks,
            1 + max(a.depth, b.depth),
        )

    def _gen_list_expr(self, env: _Env, depth: int, required: set[str] | None = None) -> ExprSpec:
        required = set(required or ())
        if env.names(LIST_INT) and self.rng.random() < 0.45:
            name = self.rng.choice(env.names(LIST_INT))
            info = env.info(name)
            if self.rng.random() < 0.55:
                node = ast.Subscript(
                    ast.Name(name, ast.Load()),
                    ast.Slice(None, None, self.rng.choice((None, ast.Constant(2), ast.Constant(-1)))),
                    ast.Load(),
                )
            else:
                node = ast.Name(name, ast.Load())
            return ExprSpec(node, LIST_INT, info.deps, (), info.depth + 1)
        n = self.rng.randint(2, max(2, self.config.list_size[1]))
        parts = self._covering_exprs(env, n, max(0, depth - 1), required)
        return ExprSpec(
            ast.List([p.node for p in parts], ast.Load()),
            LIST_INT,
            frozenset().union(*(p.deps for p in parts)) if parts else frozenset(),
            tuple(r for p in parts for r in p.risks),
            1 + max((p.depth for p in parts), default=0),
        )

    def _gen_dict_expr(self, env: _Env, depth: int) -> ExprSpec:
        size = self.rng.randint(2, 4)
        values = [self._gen_int_expr(env, max(0, depth - 1)) for _ in range(size)]
        return ExprSpec(
            ast.Dict([ast.Constant(i) for i in range(size)], [x.node for x in values]),
            DICT_INT,
            frozenset().union(*(x.deps for x in values)) if values else frozenset(),
            tuple(r for x in values for r in x.risks),
            1 + max((x.depth for x in values), default=0),
        )

    def _force_risky_int_expr(self, env: _Env, depth: int) -> ExprSpec:
        choices = []
        if env.names(LIST_INT):
            choices += ["index", "list_index"]
        if env.names(DICT_INT):
            choices += ["dict"]
        choices += ["division"]
        kind = self.rng.choice(choices)
        if kind == "division":
            a = self._gen_int_expr(env, max(0, depth - 1))
            b = self._gen_int_expr(env, max(0, depth - 1))
            return ExprSpec(
                ast.BinOp(a.node, ast.FloorDiv(), b.node),
                INT,
                a.deps | b.deps,
                a.risks + b.risks + (Risk("ZeroDivisionError", b.deps),),
                1 + max(a.depth, b.depth),
            )
        if kind == "index":
            name = self.rng.choice(env.names(LIST_INT))
            idx = self._gen_int_expr(env, max(0, depth - 1))
            info = env.info(name)
            return ExprSpec(
                ast.Subscript(ast.Name(name, ast.Load()), idx.node, ast.Load()),
                INT,
                info.deps | idx.deps,
                idx.risks + (Risk("IndexError", idx.deps),),
                max(info.depth, idx.depth) + 1,
            )
        if kind == "list_index":
            name = self.rng.choice(env.names(LIST_INT))
            value = self._gen_int_expr(env, max(0, depth - 1))
            info = env.info(name)
            return ExprSpec(
                ast.Call(ast.Attribute(ast.Name(name, ast.Load()), "index", ast.Load()), [value.node], []),
                INT,
                info.deps | value.deps,
                value.risks + (Risk("ValueError", value.deps),),
                max(info.depth, value.depth) + 1,
            )
        name = self.rng.choice(env.names(DICT_INT))
        key = self._gen_int_expr(env, max(0, depth - 1))
        info = env.info(name)
        return ExprSpec(
            ast.Subscript(ast.Name(name, ast.Load()), key.node, ast.Load()),
            INT,
            info.deps | key.deps,
            key.risks + (Risk("KeyError", key.deps),),
            max(info.depth, key.depth) + 1,
        )
