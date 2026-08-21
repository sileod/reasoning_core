from __future__ import annotations

import ast
from typing import Iterable

from ._imperative_mesopy_types import INT, LIST_INT, OBSERVED_ERRORS, Risk, VarInfo, _Env, MesopyComplexity


class _StatementMixin:
    def _merge_existing(self, env: _Env, children: Iterable[_Env], names: Iterable[str]) -> None:
        children = list(children)
        for name in names:
            infos = [child.vars.get(name) for child in children if name in child.vars]
            if not infos:
                continue
            base = env.info(name)
            deps = base.deps | frozenset().union(*(info.deps for info in infos))
            depth = max([base.depth] + [info.depth for info in infos])
            lengths = {info.length for info in infos if info.length is not None}
            length = base.length if not lengths else (next(iter(lengths)) if len(lengths) == 1 else None)
            env.vars[name] = VarInfo(base.typ, deps, depth, length)

    def _gen_effect(
        self,
        env: _Env,
        focus: str,
        state: str,
        control_depth: int,
        expr_depth: int,
        cx: MesopyComplexity,
    ) -> tuple[list[ast.stmt], set[str], list[Risk]]:
        kinds = ["focus", "mutate"]
        if control_depth > 0:
            kinds += ["if", "for", "while"]
        kind = self.rng.choice(kinds)

        if kind == "focus":
            expr = self._gen_int_expr(env, max(0, expr_depth - 1))
            old = env.info(focus)
            env.vars[focus] = VarInfo(INT, old.deps | expr.deps, max(old.depth, expr.depth) + 1)
            return [ast.AugAssign(
                ast.Name(focus, ast.Store()),
                self.rng.choice((ast.Add(), ast.Sub())),
                expr.node,
            )], set(), list(expr.risks)

        if kind == "mutate":
            info = env.info(state)
            value = self._gen_int_expr(env, max(0, expr_depth - 1))
            idx = self._gen_int_expr(env, max(0, expr_depth - 1))
            idx_node = idx.node
            risks = list(value.risks + idx.risks)
            if not self._take_risk() and info.length:
                idx_node = ast.BinOp(
                    ast.Call(ast.Name("abs", ast.Load()), [idx.node], []),
                    ast.Mod(),
                    ast.Constant(info.length),
                )
            else:
                risks.append(Risk("IndexError", idx.deps))
            env.vars[state] = VarInfo(LIST_INT, info.deps | value.deps | idx.deps, info.depth + 1, info.length)
            old = env.info(focus)
            env.vars[focus] = VarInfo(INT, old.deps | env.info(state).deps, old.depth + 1)
            return [
                ast.AugAssign(
                    ast.Subscript(ast.Name(state, ast.Load()), idx_node, ast.Store()),
                    self.rng.choice((ast.Add(), ast.Sub())),
                    value.node,
                ),
                ast.AugAssign(
                    ast.Name(focus, ast.Store()), ast.Add(),
                    ast.Call(ast.Name("len", ast.Load()), [ast.Name(state, ast.Load())], []),
                ),
            ], {"mutation_call"}, risks

        if kind == "if":
            cond = self._gen_bool_expr(env, max(0, expr_depth - 1))
            yes_env, no_env = env.copy(), env.copy()
            yes, yes_tags, yes_risks = self._gen_effect(
                yes_env, focus, state, control_depth - 1, expr_depth, cx
            )
            no, no_tags, no_risks = self._gen_effect(
                no_env, focus, state, control_depth - 1, expr_depth, cx
            )
            self._merge_existing(env, (yes_env, no_env), (focus, state))
            return [ast.If(cond.node, yes, no)], {"conditional_flow"} | yes_tags | no_tags, list(cond.risks) + yes_risks + no_risks

        if kind == "for":
            child = env.copy()
            inner, tags, risks = self._gen_effect(
                child, focus, state, control_depth - 1, expr_depth, cx
            )
            self._merge_existing(env, (child,), (focus, state))
            it = self.names.take("i")
            bound = self.rng.randint(1, max(1, cx.loop_bound))
            return [ast.For(
                ast.Name(it, ast.Store()),
                ast.Call(ast.Name("range", ast.Load()), [ast.Constant(bound)], []),
                inner,
                [],
            )], {"loop_carried_state"} | tags, risks

        child = env.copy()
        inner, tags, risks = self._gen_effect(
            child, focus, state, control_depth - 1, expr_depth, cx
        )
        self._merge_existing(env, (child,), (focus, state))
        counter = self.names.take("i")
        limit = self.rng.randint(1, max(1, cx.loop_bound))
        return [
            ast.Assign([ast.Name(counter, ast.Store())], ast.Constant(0)),
            ast.While(
                ast.Compare(ast.Name(counter, ast.Load()), [ast.Lt()], [ast.Constant(limit)]),
                inner + [ast.AugAssign(ast.Name(counter, ast.Store()), ast.Add(), ast.Constant(1))],
                [],
            ),
        ], {"loop_carried_state"} | tags, risks

    def _gen_stmt(
        self,
        env: _Env,
        focus: str,
        state: str,
        control_depth: int,
        expr_depth: int,
        cx: MesopyComplexity,
    ) -> tuple[list[ast.stmt], set[str], list[Risk]]:
        kinds = ["focus", "mutate", "assign", "dict", "swap", "comprehension"]
        if control_depth > 0:
            kinds += ["if", "for", "while", "try"]
        kind = self.rng.choice(kinds)
        tags: set[str] = set()
        risks: list[Risk] = []

        if kind == "focus":
            expr = self._gen_int_expr(env, expr_depth, required=set(env.params) if self.rng.random() < 0.35 else None)
            op = self.rng.choice((ast.Add(), ast.Sub()))
            node = ast.AugAssign(ast.Name(focus, ast.Store()), op, expr.node)
            old = env.info(focus)
            env.add(focus, VarInfo(INT, old.deps | expr.deps, max(old.depth, expr.depth) + 1))
            return [node], tags, list(expr.risks)

        if kind == "mutate":
            list_name = self.rng.choice(env.names(LIST_INT))
            info = env.info(list_name)
            value = self._gen_int_expr(env, max(0, expr_depth - 1))
            if self.rng.random() < 0.45:
                node = ast.Expr(ast.Call(ast.Attribute(ast.Name(list_name, ast.Load()), "append", ast.Load()), [value.node], []))
                env.vars[list_name] = VarInfo(LIST_INT, info.deps | value.deps, info.depth + 1, None if info.length is None else info.length + 1)
            else:
                idx = self._gen_int_expr(env, max(0, expr_depth - 1))
                idx_node = idx.node
                node_risks = list(value.risks + idx.risks)
                if not self._take_risk() and info.length:
                    idx_node = ast.BinOp(ast.Call(ast.Name("abs", ast.Load()), [idx.node], []), ast.Mod(), ast.Constant(info.length))
                else:
                    node_risks.append(Risk("IndexError", idx.deps))
                node = ast.AugAssign(
                    ast.Subscript(ast.Name(list_name, ast.Load()), idx_node, ast.Store()),
                    self.rng.choice((ast.Add(), ast.Sub())),
                    value.node,
                )
                env.vars[list_name] = VarInfo(LIST_INT, info.deps | value.deps | idx.deps, info.depth + 1, info.length)
                risks.extend(node_risks)
            use = ast.AugAssign(
                ast.Name(focus, ast.Store()),
                ast.Add(),
                ast.Call(ast.Name("len", ast.Load()), [ast.Name(list_name, ast.Load())], []),
            )
            old = env.info(focus)
            env.add(focus, VarInfo(INT, old.deps | env.info(list_name).deps, old.depth + 1))
            return [node, use], {"mutation_call"}, risks + list(value.risks)

        if kind == "assign":
            expr = self._gen_int_expr(env, expr_depth)
            name = self.names.take("v")
            env.add(name, expr)
            nodes = [ast.Assign([ast.Name(name, ast.Store())], expr.node)]
            if self.rng.random() > self.config.noise_rate:
                nodes.append(ast.AugAssign(ast.Name(focus, ast.Store()), ast.Add(), ast.Name(name, ast.Load())))
                old = env.info(focus)
                env.add(focus, VarInfo(INT, old.deps | expr.deps, max(old.depth, expr.depth) + 1))
            return nodes, tags, list(expr.risks)

        if kind == "dict":
            expr = self._gen_dict_expr(env, expr_depth)
            name = self.names.take("map")
            env.add(name, expr)
            key = self._gen_int_expr(env, max(0, expr_depth - 1))
            if not self._take_risk():
                access = ast.Call(ast.Attribute(ast.Name(name, ast.Load()), "get", ast.Load()), [key.node, ast.Constant(0)], [])
                access_risks = key.risks
            else:
                access = ast.Subscript(ast.Name(name, ast.Load()), key.node, ast.Load())
                access_risks = key.risks + (Risk("KeyError", key.deps),)
            nodes = [
                ast.Assign([ast.Name(name, ast.Store())], expr.node),
                ast.AugAssign(ast.Name(focus, ast.Store()), ast.Add(), access),
            ]
            old = env.info(focus)
            env.add(focus, VarInfo(INT, old.deps | expr.deps | key.deps, max(old.depth, expr.depth, key.depth) + 1))
            return nodes, {"mapping"}, list(expr.risks + access_risks)

        if kind == "swap":
            other = self.names.take("v")
            expr = self._gen_int_expr(env, max(0, expr_depth - 1))
            env.add(other, expr)
            nodes = [
                ast.Assign([ast.Name(other, ast.Store())], expr.node),
                ast.Assign(
                    [ast.Tuple([ast.Name(focus, ast.Store()), ast.Name(other, ast.Store())], ast.Store())],
                    ast.Tuple([ast.Name(other, ast.Load()), ast.Name(focus, ast.Load())], ast.Load()),
                ),
                ast.AugAssign(ast.Name(focus, ast.Store()), ast.Add(), ast.Name(other, ast.Load())),
            ]
            old = env.info(focus)
            env.add(focus, VarInfo(INT, old.deps | expr.deps, max(old.depth, expr.depth) + 2))
            return nodes, tags, list(expr.risks)

        if kind == "comprehension":
            it = self.names.take("i")
            name = self.names.take("comp")
            bound = self.rng.randint(2, max(2, cx.loop_bound))
            expr = self._gen_int_expr(env, max(0, expr_depth - 1))
            comp = ast.ListComp(
                ast.BinOp(ast.Name(it, ast.Load()), ast.Add(), expr.node),
                [ast.comprehension(ast.Name(it, ast.Store()), ast.Call(ast.Name("range", ast.Load()), [ast.Constant(bound)], []), [], 0)],
            )
            env.add(name, VarInfo(LIST_INT, expr.deps, expr.depth + 1, bound))
            nodes = [
                ast.Assign([ast.Name(name, ast.Store())], comp),
                ast.AugAssign(ast.Name(focus, ast.Store()), ast.Add(), ast.Call(ast.Name("sum", ast.Load()), [ast.Name(name, ast.Load())], [])),
            ]
            old = env.info(focus)
            env.add(focus, VarInfo(INT, old.deps | expr.deps, max(old.depth, expr.depth) + 2))
            return nodes, {"comprehension"}, list(expr.risks)

        if kind in {"if", "for", "while"}:
            nodes, nested_tags, nested_risks = self._gen_effect(
                env, focus, state, max(0, control_depth - 1), expr_depth, cx
            )
            if kind == "if" and not isinstance(nodes[0], ast.If):
                cond = self._gen_bool_expr(env, max(0, expr_depth - 1))
                yes_env, no_env = env.copy(), env.copy()
                yes, yes_tags, yes_risks = self._gen_effect(yes_env, focus, state, max(0, control_depth - 1), expr_depth, cx)
                no, no_tags, no_risks = self._gen_effect(no_env, focus, state, max(0, control_depth - 1), expr_depth, cx)
                self._merge_existing(env, (yes_env, no_env), (focus, state))
                return [ast.If(cond.node, yes, no)], {"conditional_flow"} | yes_tags | no_tags, list(cond.risks) + yes_risks + no_risks
            if kind == "for" and not isinstance(nodes[0], ast.For):
                child = env.copy()
                inner, inner_tags, inner_risks = self._gen_effect(child, focus, state, max(0, control_depth - 1), expr_depth, cx)
                self._merge_existing(env, (child,), (focus, state))
                it = self.names.take("i")
                bound = self.rng.randint(1, max(1, cx.loop_bound))
                return [ast.For(ast.Name(it, ast.Store()), ast.Call(ast.Name("range", ast.Load()), [ast.Constant(bound)], []), inner, [])], {"loop_carried_state"} | inner_tags, inner_risks
            if kind == "while" and not isinstance(nodes[0], ast.While):
                child = env.copy()
                inner, inner_tags, inner_risks = self._gen_effect(child, focus, state, max(0, control_depth - 1), expr_depth, cx)
                self._merge_existing(env, (child,), (focus, state))
                counter = self.names.take("i")
                limit = self.rng.randint(1, max(1, cx.loop_bound))
                return [
                    ast.Assign([ast.Name(counter, ast.Store())], ast.Constant(0)),
                    ast.While(
                        ast.Compare(ast.Name(counter, ast.Load()), [ast.Lt()], [ast.Constant(limit)]),
                        inner + [ast.AugAssign(ast.Name(counter, ast.Store()), ast.Add(), ast.Constant(1))],
                        [],
                    ),
                ], {"loop_carried_state"} | inner_tags, inner_risks
            return nodes, nested_tags, nested_risks

        risky = self._force_risky_int_expr(env, max(1, expr_depth - 1))
        fallback = self._gen_int_expr(env, max(0, expr_depth - 1))
        exc = self.rng.choice(tuple({r.kind for r in risky.risks}) or OBSERVED_ERRORS)
        node = ast.Try(
            [ast.AugAssign(ast.Name(focus, ast.Store()), ast.Add(), risky.node)],
            [ast.ExceptHandler(ast.Name(exc, ast.Load()), None, [ast.AugAssign(ast.Name(focus, ast.Store()), ast.Sub(), fallback.node)])],
            [],
            [],
        )
        old = env.info(focus)
        env.add(focus, VarInfo(INT, old.deps | risky.deps | fallback.deps, max(old.depth, risky.depth, fallback.depth) + 1))
        return [node], {"try_except"}, list(risky.risks + fallback.risks)
