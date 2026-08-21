from __future__ import annotations

import ast
import random

from ._imperative_mesopy_analysis import _alpha_rename, _liveness_metrics, _structural_features, structural_fingerprint
from ._imperative_mesopy_types import (
    CONTROLLED_PHENOMENA, INT, LIST_INT, OBSERVED_ERRORS, PHENOMENA, TUPLE_INT,
    ExprSpec, MesopyComplexity, MesopyConfig, MesopyGoal, MesopySample, Risk, VarInfo, _Env, _Names,
)


class _GenerationMixin:
    def __init__(self, config: MesopyConfig | None = None, seed: int | None = None):
        self.config = config or MesopyConfig()
        self.rng = random.Random(seed)
        self.names = _Names()
        self._seen: set[str] = set()
        self._risk_remaining = 0

    def execution(self, **kwargs) -> MesopySample:
        return self.generate(MesopyGoal(runnable=True, **kwargs))

    def runnability_pair(self, **kwargs) -> MesopySample:
        return self.generate(MesopyGoal(runnable=None, paired_runnability=True, **kwargs))

    def generate(self, goal: MesopyGoal | None = None) -> MesopySample:
        goal = goal or MesopyGoal()
        self._validate_goal(goal)
        for _ in range(self.config.max_attempts):
            candidate = self._build_candidate(goal)
            if candidate is None:
                continue
            if self.config.deduplicate and candidate.fingerprint in self._seen:
                continue
            self._seen.add(candidate.fingerprint)
            return candidate
        raise RuntimeError(f"failed to generate imperative Mesopy sample: {goal}")

    def _validate_goal(self, goal: MesopyGoal) -> None:
        if goal.error is not None and goal.error not in OBSERVED_ERRORS:
            raise ValueError(f"supported observed errors: {', '.join(OBSERVED_ERRORS)}")
        if goal.require_recursion and not goal.allow_recursion:
            raise ValueError("require_recursion needs allow_recursion=True")
        if goal.input_arity is not None and goal.input_arity < 0:
            raise ValueError("input_arity must be non-negative")
        if (goal.runnable is False or goal.paired_runnability or goal.error) and goal.input_arity == 0:
            raise ValueError("runnability goals need at least one input")
        if goal.result_kind not in (None, INT, LIST_INT, TUPLE_INT):
            raise ValueError(f"unsupported result_kind: {goal.result_kind}")
        unknown = set(goal.phenomena) - set(PHENOMENA)
        if unknown:
            raise ValueError(f"unknown phenomena: {sorted(unknown)}")

    def _build_candidate(self, goal: MesopyGoal) -> MesopySample | None:
        self.names = _Names()
        cfg = self.config
        cx = goal.complexity or cfg.complexity
        arity = goal.input_arity
        if arity is None:
            lo, hi = cfg.input_arity
            if goal.runnable is False or goal.paired_runnability or goal.error:
                lo = max(1, lo)
            arity = self.rng.randint(lo, hi)
        self._risk_remaining = 0
        params = [self.names.take("arg") for _ in range(arity)]
        env = _Env(params)

        recursive = goal.allow_recursion and (
            goal.require_recursion
            or "recursion" in goal.phenomena
            or self.rng.random() < cfg.recursion_rate
        )
        helpers, helper_names = self._gen_helpers(cx, recursive)
        env.functions = helper_names
        self._risk_remaining = min(cfg.max_risk_sites, max(1, 1 + cx.statements // 8))

        body: list[ast.stmt] = []
        phenomena: set[str] = set()
        risks: list[Risk] = []

        list_len = self.rng.randint(*cfg.list_size)
        state = self.names.take("seq")
        state_exprs = self._covering_exprs(env, list_len, cx.expr_depth, set(params))
        body.append(ast.Assign([ast.Name(state, ast.Store())], ast.List([x.node for x in state_exprs], ast.Load())))
        state_deps = frozenset().union(*(x.deps for x in state_exprs)) if state_exprs else frozenset()
        env.add(state, VarInfo(LIST_INT, state_deps, 1, list_len))
        for x in state_exprs:
            risks.extend(x.risks)

        focus = self.names.take("acc")
        focus_spec = self._gen_int_expr(env, cx.expr_depth, required=set(params))
        body.append(ast.Assign([ast.Name(focus, ast.Store())], focus_spec.node))
        env.add(focus, focus_spec)
        risks.extend(focus_spec.risks)

        requested = list(goal.phenomena)
        ordinary_count = max(2, cx.statements)
        for _ in range(ordinary_count):
            nodes, tags, new_risks = self._gen_stmt(
                env, focus, state, cx.control_depth, cx.expr_depth, cx
            )
            body.extend(nodes)
            phenomena.update(tags)
            risks.extend(new_risks)

            if self.rng.random() < cfg.phenomenon_rate:
                choices = [p for p in CONTROLLED_PHENOMENA if p not in phenomena]
                if choices:
                    p = self.rng.choice(choices)
                    nodes, new_risks = self._phenomenon(p, env, focus, state, cx)
                    self._splice(body, nodes)
                    phenomena.add(p)
                    risks.extend(new_risks)

        for p in requested:
            if p == "recursion":
                phenomena.add("recursion")
                continue
            if p in phenomena:
                continue
            if p in ("conditional_flow", "try_except", "early_return", "comprehension", "mapping"):
                nodes, tags, new_risks = self._force_structural_phenomenon(p, env, focus, state, cx)
                self._splice(body, nodes)
                phenomena.update(tags)
                risks.extend(new_risks)
            else:
                nodes, new_risks = self._phenomenon(p, env, focus, state, cx)
                self._splice(body, nodes)
                phenomena.add(p)
                risks.extend(new_risks)

        if recursive:
            phenomena.add("recursion")

        # Keep semantic depth productive without a conspicuous terminal padding chain:
        # dependency-preserving updates are inserted among existing top-level statements.
        while env.info(focus).depth < cx.dataflow_depth:
            expr = self._gen_int_expr(env, max(1, cx.expr_depth - 1), required=set(params))
            node = ast.Assign(
                [ast.Name(focus, ast.Store())],
                ast.BinOp(ast.Name(focus, ast.Load()), self.rng.choice((ast.Add(), ast.Sub())), expr.node),
            )
            combined = VarInfo(
                INT,
                env.info(focus).deps | expr.deps,
                max(env.info(focus).depth, expr.depth) + 1,
            )
            env.add(focus, combined)
            risks.extend(expr.risks)
            insert_at = self.rng.randrange(max(1, len(body) // 2), len(body) + 1)
            body.insert(insert_at, node)

        result_kind = goal.result_kind or self.rng.choice((INT, INT, LIST_INT, TUPLE_INT))
        return_spec = self._return_expr(result_kind, env, focus, state, cx.expr_depth, set(params))
        body.append(ast.Return(return_spec.node))
        risks.extend(return_spec.risks)

        endpoint = "endpoint"
        endpoint_def = ast.FunctionDef(
            endpoint,
            ast.arguments([], [ast.arg(p) for p in params], None, [], [], None, []),
            body,
            [],
        )
        module_body = helpers + [endpoint_def]
        self.rng.shuffle(module_body)
        module = ast.fix_missing_locations(ast.Module(module_body, []))

        liveness = _liveness_metrics(endpoint_def)
        if liveness["live_fraction"] < goal.min_live_fraction:
            return None

        code = ast.unparse(module) + "\n"
        if cfg.max_source_chars is not None and len(code) > cfg.max_source_chars:
            return None

        probes = self._probe_args(arity, cfg.probe_count)
        outcomes = self._execute_many(code, endpoint, probes)
        selected = self._select_outcomes(goal, outcomes)
        if selected is None:
            return None

        sensitivity, diversity = self._probe_metrics(code, endpoint, selected[0], arity)
        if sensitivity < goal.min_param_sensitivity:
            return None

        anonymize = cfg.anonymize_names if goal.anonymize_names is None else goal.anonymize_names
        if anonymize:
            renamed, renamed_entrypoint = _alpha_rename(module, endpoint, self.rng)
            renamed_code = ast.unparse(renamed) + "\n"
            if cfg.max_source_chars is not None and len(renamed_code) > cfg.max_source_chars:
                return None
            check = self._execute_many(renamed_code, renamed_entrypoint, [x.args for x in selected])
            if any(
                (a.ok, a.value, a.error) != (b.ok, b.value, b.error)
                for a, b in zip(selected, check)
            ):
                return None
            module, code, endpoint = renamed, renamed_code, renamed_entrypoint

        profile = selected[0]
        if cfg.profile_accepted:
            profile = self._profile(code, endpoint, selected[0].args, cfg.max_profile_steps)
            if profile.ok != selected[0].ok or profile.value != selected[0].value or profile.error != selected[0].error:
                return None
            selected = (profile,) + tuple(selected[1:])

        fingerprint = structural_fingerprint(code)
        features = _structural_features(module)
        features.update(liveness)
        features.update(
            dataflow_depth=env.info(focus).depth,
            recursive=recursive,
            input_arity=arity,
            result_kind=result_kind,
            risk_sites=len(risks),
            risk_kinds=tuple(sorted({r.kind for r in risks})),
            probe_errors=tuple(sorted({x.error for x in outcomes if x.error})),
            probe_output_diversity=diversity,
            param_sensitivity=sensitivity,
            dynamic_steps=profile.steps,
            dynamic_lines=profile.distinct_lines,
        )
        return MesopySample(
            code,
            endpoint,
            tuple(selected),
            tuple(sorted(phenomena)),
            features,
            fingerprint,
        )

    def _take_risk(self) -> bool:
        if self._risk_remaining <= 0 or self.rng.random() >= self.config.risk_rate:
            return False
        self._risk_remaining -= 1
        return True

    def _covering_exprs(
        self, env: _Env, count: int, depth: int, required: set[str]
    ) -> list[ExprSpec]:
        buckets = [set() for _ in range(count)]
        for i, dep in enumerate(sorted(required)):
            buckets[i % count].add(dep)
        return [self._gen_int_expr(env, depth, required=b) for b in buckets]

    def _gen_helpers(self, cx: MesopyComplexity, recursive: bool) -> tuple[list[ast.stmt], list[str]]:
        helpers: list[ast.stmt] = []
        names: list[str] = []
        for i in range(max(0, cx.functions)):
            fn = self.names.take("fn")
            x = self.names.take("p")
            y = self.names.take("p")
            local = _Env((x, y))
            local.functions = names[:]
            expr = self._gen_int_expr(local, max(1, cx.expr_depth - 1), required={x})
            if names and i <= cx.call_depth:
                prev = self.rng.choice(names)
                call = ast.Call(ast.Name(prev, ast.Load()), [ast.Name(x, ast.Load()), ast.Name(y, ast.Load())], [])
                expr = ExprSpec(
                    ast.BinOp(call, self.rng.choice((ast.Add(), ast.Sub())), expr.node),
                    INT,
                    expr.deps | {x, y},
                    expr.risks,
                    expr.depth + 1,
                )
            helpers.append(ast.FunctionDef(
                fn,
                ast.arguments([], [ast.arg(x), ast.arg(y)], None, [], [], None, []),
                [ast.Return(expr.node)],
                [],
            ))
            names.append(fn)

        if recursive:
            fn = self.names.take("recur")
            n = self.names.take("n")
            z = self.names.take("z")
            base = ast.If(
                ast.Compare(ast.Name(n, ast.Load()), [ast.LtE()], [ast.Constant(0)]),
                [ast.Return(ast.Name(z, ast.Load()))],
                [],
            )
            step = ast.Call(
                ast.Name(fn, ast.Load()),
                [
                    ast.BinOp(ast.Name(n, ast.Load()), ast.Sub(), ast.Constant(1)),
                    ast.BinOp(ast.Name(z, ast.Load()), ast.Add(), ast.Name(n, ast.Load())),
                ],
                [],
            )
            helpers.append(ast.FunctionDef(
                fn,
                ast.arguments([], [ast.arg(n), ast.arg(z)], None, [], [], None, []),
                [base, ast.Return(step)],
                [],
            ))
            names.append(fn)
        return helpers, names

    def _splice(self, body: list[ast.stmt], nodes: list[ast.stmt]) -> None:
        if not nodes:
            return
        # Do not place before the two initial state/focus bindings.
        pos = self.rng.randint(min(2, len(body)), len(body))
        body[pos:pos] = nodes

    def _return_expr(
        self,
        kind: str,
        env: _Env,
        focus: str,
        state: str,
        depth: int,
        required: set[str],
    ) -> ExprSpec:
        focus_info = env.info(focus)
        if kind == LIST_INT:
            # Include focus in the returned list so scalar control/dataflow remains relevant.
            state_info = env.info(state)
            node = ast.BinOp(
                ast.Name(state, ast.Load()),
                ast.Add(),
                ast.List([ast.Name(focus, ast.Load())], ast.Load()),
            )
            return ExprSpec(node, LIST_INT, state_info.deps | focus_info.deps, (), max(state_info.depth, focus_info.depth) + 1)
        if kind == TUPLE_INT:
            second = self._gen_int_expr(env, min(depth, 2), required)
            return ExprSpec(
                ast.Tuple([ast.Name(focus, ast.Load()), second.node], ast.Load()),
                TUPLE_INT,
                focus_info.deps | second.deps,
                second.risks,
                max(focus_info.depth, second.depth) + 1,
            )
        if required <= set(focus_info.deps) and self.rng.random() < 0.75:
            return ExprSpec(ast.Name(focus, ast.Load()), INT, focus_info.deps, (), focus_info.depth)
        extra = self._gen_int_expr(env, min(depth, 2), required)
        return ExprSpec(
            ast.BinOp(ast.Name(focus, ast.Load()), ast.Add(), extra.node),
            INT,
            focus_info.deps | extra.deps,
            extra.risks,
            max(focus_info.depth, extra.depth) + 1,
        )
