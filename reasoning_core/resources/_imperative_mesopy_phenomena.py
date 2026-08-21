from __future__ import annotations

import ast

from ._imperative_mesopy_types import LIST_INT, Risk, VarInfo, _Env, MesopyComplexity


class _PhenomenaMixin:
    def _phenomenon(
        self,
        phenomenon: str,
        env: _Env,
        focus: str,
        state: str,
        cx: MesopyComplexity,
    ) -> tuple[list[ast.stmt], list[Risk]]:
        rng = self.rng
        risks: list[Risk] = []
        if phenomenon == "aliasing":
            alias = self.names.take("ref")
            idx = rng.randrange(max(1, env.info(state).length or 1))
            delta = rng.choice((-2, -1, 1, 2))
            env.add(alias, VarInfo(LIST_INT, env.info(state).deps, env.info(state).depth, env.info(state).length))
            return [
                ast.Assign([ast.Name(alias, ast.Store())], ast.Name(state, ast.Load())),
                ast.AugAssign(ast.Subscript(ast.Name(alias, ast.Load()), ast.Constant(idx), ast.Store()), ast.Add(), ast.Constant(delta)),
                ast.AugAssign(ast.Name(focus, ast.Store()), ast.Add(), ast.Subscript(ast.Name(state, ast.Load()), ast.Constant(idx), ast.Load())),
            ], risks

        if phenomenon == "rebinding_vs_aliasing":
            alias = self.names.take("ref")
            env.add(alias, VarInfo(LIST_INT, env.info(state).deps, env.info(state).depth, env.info(state).length))
            idx = rng.randrange(max(1, env.info(state).length or 1))
            if rng.random() < 0.5:
                copy_node = ast.Subscript(ast.Name(state, ast.Load()), ast.Slice(None, None, None), ast.Load())
            else:
                copy_node = ast.Call(ast.Name("list", ast.Load()), [ast.Name(state, ast.Load())], [])
            return [
                ast.Assign([ast.Name(alias, ast.Store())], ast.Name(state, ast.Load())),
                ast.Assign([ast.Name(state, ast.Store())], copy_node),
                ast.AugAssign(ast.Subscript(ast.Name(alias, ast.Load()), ast.Constant(idx), ast.Store()), ast.Add(), ast.Constant(1)),
                ast.AugAssign(ast.Name(focus, ast.Store()), ast.Add(), ast.BinOp(
                    ast.Subscript(ast.Name(alias, ast.Load()), ast.Constant(idx), ast.Load()),
                    ast.Sub(),
                    ast.Subscript(ast.Name(state, ast.Load()), ast.Constant(idx), ast.Load()),
                )),
            ], risks

        if phenomenon == "closure_late_binding":
            funcs = self.names.take("funcs")
            loop = self.names.take("i")
            arg = self.names.take("z")
            call_index = rng.choice((0, 1, 2))
            lam = ast.Lambda(
                ast.arguments([], [ast.arg(arg)], None, [], [], None, []),
                ast.BinOp(ast.Name(arg, ast.Load()), ast.Add(), ast.Name(loop, ast.Load())),
            )
            return [
                ast.Assign([ast.Name(funcs, ast.Store())], ast.List([], ast.Load())),
                ast.For(
                    ast.Name(loop, ast.Store()),
                    ast.Call(ast.Name("range", ast.Load()), [ast.Constant(3)], []),
                    [ast.Expr(ast.Call(ast.Attribute(ast.Name(funcs, ast.Load()), "append", ast.Load()), [lam], []))],
                    [],
                ),
                ast.Assign([ast.Name(focus, ast.Store())], ast.Call(
                    ast.Subscript(ast.Name(funcs, ast.Load()), ast.Constant(call_index), ast.Load()),
                    [ast.Name(focus, ast.Load())],
                    [],
                )),
            ], risks

        if phenomenon == "default_capture":
            bias = self.names.take("bias")
            fn = self.names.take("inner")
            z = self.names.take("z")
            default = self.names.take("d")
            c = rng.randint(1, 4)
            return [
                ast.Assign([ast.Name(bias, ast.Store())], ast.BinOp(ast.Name(focus, ast.Load()), ast.Mod(), ast.Constant(5))),
                ast.FunctionDef(
                    fn,
                    ast.arguments([], [ast.arg(z), ast.arg(default)], None, [], [], None, [ast.Name(bias, ast.Load())]),
                    [ast.Return(ast.BinOp(ast.Name(z, ast.Load()), ast.Add(), ast.Name(default, ast.Load())))],
                    [],
                ),
                ast.AugAssign(ast.Name(bias, ast.Store()), ast.Add(), ast.Constant(c)),
                ast.Assign([ast.Name(focus, ast.Store())], ast.Call(ast.Name(fn, ast.Load()), [ast.Name(focus, ast.Load())], [])),
            ], risks

        if phenomenon == "mutable_default":
            fn = self.names.take("inner")
            z = self.names.take("z")
            box = self.names.take("box")
            return [
                ast.FunctionDef(
                    fn,
                    ast.arguments([], [ast.arg(z), ast.arg(box)], None, [], [], None, [ast.List([], ast.Load())]),
                    [
                        ast.Expr(ast.Call(ast.Attribute(ast.Name(box, ast.Load()), "append", ast.Load()), [ast.Name(z, ast.Load())], [])),
                        ast.Return(ast.BinOp(ast.Name(z, ast.Load()), ast.Add(), ast.Call(ast.Name("len", ast.Load()), [ast.Name(box, ast.Load())], []))),
                    ],
                    [],
                ),
                ast.Assign([ast.Name(focus, ast.Store())], ast.Call(ast.Name(fn, ast.Load()), [ast.Name(focus, ast.Load())], [])),
                ast.Assign([ast.Name(focus, ast.Store())], ast.Call(ast.Name(fn, ast.Load()), [ast.Name(focus, ast.Load())], [])),
            ], risks

        if phenomenon == "mutation_call":
            fn = self.names.take("mut")
            ys = self.names.take("ys")
            idx = rng.randrange(max(1, env.info(state).length or 1))
            return [
                ast.FunctionDef(
                    fn,
                    ast.arguments([], [ast.arg(ys)], None, [], [], None, []),
                    [
                        ast.AugAssign(ast.Subscript(ast.Name(ys, ast.Load()), ast.Constant(idx), ast.Store()), ast.Add(), ast.Constant(1)),
                        ast.Return(ast.Subscript(ast.Name(ys, ast.Load()), ast.Constant(idx), ast.Load())),
                    ],
                    [],
                ),
                ast.AugAssign(ast.Name(focus, ast.Store()), ast.Add(), ast.Call(ast.Name(fn, ast.Load()), [ast.Name(state, ast.Load())], [])),
            ], risks

        if phenomenon == "loop_carried_state":
            it = self.names.take("i")
            return [ast.For(
                ast.Name(it, ast.Store()),
                ast.Call(ast.Name("range", ast.Load()), [ast.Constant(max(2, cx.loop_bound))], []),
                [ast.AugAssign(ast.Name(focus, ast.Store()), ast.Add(), ast.Name(it, ast.Load()))],
                [],
            )], risks

        raise ValueError(phenomenon)

    def _force_structural_phenomenon(
        self,
        phenomenon: str,
        env: _Env,
        focus: str,
        state: str,
        cx: MesopyComplexity,
    ) -> tuple[list[ast.stmt], set[str], list[Risk]]:
        if phenomenon == "conditional_flow":
            cond = self._gen_bool_expr(env, max(0, cx.expr_depth - 1))
            a = self._gen_int_expr(env, max(0, cx.expr_depth - 1))
            b = self._gen_int_expr(env, max(0, cx.expr_depth - 1))
            return [ast.If(
                cond.node,
                [ast.AugAssign(ast.Name(focus, ast.Store()), ast.Add(), a.node)],
                [ast.AugAssign(ast.Name(focus, ast.Store()), ast.Sub(), b.node)],
            )], {phenomenon}, list(cond.risks + a.risks + b.risks)
        if phenomenon == "try_except":
            risky = self._force_risky_int_expr(env, max(1, cx.expr_depth - 1))
            exc = next(iter({r.kind for r in risky.risks}), "Exception")
            return [ast.Try(
                [ast.AugAssign(ast.Name(focus, ast.Store()), ast.Add(), risky.node)],
                [ast.ExceptHandler(ast.Name(exc, ast.Load()), None, [ast.AugAssign(ast.Name(focus, ast.Store()), ast.Sub(), ast.Constant(1))])],
                [], [],
            )], {phenomenon}, list(risky.risks)
        if phenomenon == "early_return":
            cond = self._gen_bool_expr(env, max(0, cx.expr_depth - 1))
            return [ast.If(cond.node, [ast.Return(ast.Name(focus, ast.Load()))], [])], {phenomenon}, list(cond.risks)
        if phenomenon == "comprehension":
            it = self.names.take("i")
            name = self.names.take("comp")
            bound = max(2, cx.loop_bound)
            env.add(name, VarInfo(LIST_INT, env.info(focus).deps, env.info(focus).depth + 1, bound))
            return [
                ast.Assign([ast.Name(name, ast.Store())], ast.ListComp(
                    ast.BinOp(ast.Name(it, ast.Load()), ast.Add(), ast.Name(focus, ast.Load())),
                    [ast.comprehension(ast.Name(it, ast.Store()), ast.Call(ast.Name("range", ast.Load()), [ast.Constant(bound)], []), [], 0)],
                )),
                ast.AugAssign(ast.Name(focus, ast.Store()), ast.Add(), ast.Call(ast.Name("sum", ast.Load()), [ast.Name(name, ast.Load())], [])),
            ], {phenomenon}, []
        if phenomenon == "mapping":
            expr = self._gen_dict_expr(env, cx.expr_depth)
            name = self.names.take("map")
            env.add(name, expr)
            return [
                ast.Assign([ast.Name(name, ast.Store())], expr.node),
                ast.AugAssign(ast.Name(focus, ast.Store()), ast.Add(), ast.Call(
                    ast.Attribute(ast.Name(name, ast.Load()), "get", ast.Load()), [ast.Constant(0), ast.Constant(0)], []
                )),
            ], {phenomenon}, list(expr.risks)
        raise ValueError(phenomenon)
