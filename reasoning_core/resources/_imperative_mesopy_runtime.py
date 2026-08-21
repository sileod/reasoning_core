from __future__ import annotations

import ast
import copy
import sys
import time
from typing import Iterable

from ._imperative_mesopy_types import (
    CallOutcome, MinimalPair, MesopyGoal, MesopySample, OBSERVED_ERRORS,
)


class _RuntimeMixin:
    def evaluate(
        self,
        sample: MesopySample,
        args_list: Iterable[tuple[int, ...]],
        *,
        fresh: bool = False,
    ) -> tuple[CallOutcome, ...]:
        """Evaluate explicit calls, optionally resetting program state per call."""
        args_list = [tuple(args) for args in args_list]
        if fresh:
            try:
                compiled = compile(sample.code, "<imperative-mesopy>", "exec")
            except Exception as exc:
                return tuple(
                    CallOutcome(args, False, None, type(exc).__name__)
                    for args in args_list
                )
            outcomes = []
            for args in args_list:
                namespace = self._namespace()
                start = time.perf_counter()
                try:
                    exec(compiled, namespace, namespace)
                    value = namespace[sample.entrypoint](*args)
                    outcomes.append(CallOutcome(
                        args, True, repr(value), None, time.perf_counter() - start
                    ))
                except Exception as exc:
                    outcomes.append(CallOutcome(
                        args, False, None, type(exc).__name__, time.perf_counter() - start
                    ))
            return tuple(outcomes)
        return tuple(self._execute_many(sample.code, sample.entrypoint, args_list))

    def minimal_pair(self, sample: MesopySample, attempts: int = 24) -> MinimalPair:
        tree = ast.parse(sample.code)
        mutations = self._mutation_sites(tree)
        self.rng.shuffle(mutations)
        for path, mutation in mutations[:attempts]:
            mutated = copy.deepcopy(tree)
            description = self._apply_mutation(mutated, path, mutation)
            if description is None:
                continue
            ast.fix_missing_locations(mutated)
            code = ast.unparse(mutated) + "\n"
            outcome = self._profile(code, sample.entrypoint, sample.args, self.config.max_profile_steps)
            if sample.call.ok and outcome.ok and outcome.value != sample.call.value:
                return MinimalPair(sample, code, sample.entrypoint, outcome, description)
        raise RuntimeError("could not find output-changing minimal mutation")

    def _probe_args(self, arity: int, count: int) -> list[tuple[int, ...]]:
        if arity == 0:
            return [()]
        mag = self.config.magnitude
        special = [0, 1, -1, 2, -2, mag, -mag, mag + 1, -(mag + 1)]
        probes: list[tuple[int, ...]] = []
        for i in range(min(count, len(special))):
            xs = [self.rng.randint(-mag, mag) for _ in range(arity)]
            xs[i % arity] = special[i]
            probes.append(tuple(xs))
        while len(probes) < count:
            probes.append(tuple(self.rng.randint(-mag - 2, mag + 2) for _ in range(arity)))
        return list(dict.fromkeys(probes))

    def _select_outcomes(
        self, goal: MesopyGoal, outcomes: list[CallOutcome]
    ) -> tuple[CallOutcome, ...] | None:
        oks = [x for x in outcomes if x.ok]
        fails = [x for x in outcomes if not x.ok and x.error in OBSERVED_ERRORS]
        if goal.error:
            fails = [x for x in fails if x.error == goal.error]
        if goal.paired_runnability:
            if not oks or not fails:
                return None
            pair = [self.rng.choice(oks), self.rng.choice(fails)]
            self.rng.shuffle(pair)
            return tuple(pair)
        if goal.runnable is True:
            return (self.rng.choice(oks),) if oks else None
        if goal.runnable is False:
            return (self.rng.choice(fails),) if fails else None
        if goal.error:
            return (self.rng.choice(fails),) if fails else None
        return (self.rng.choice(outcomes),) if outcomes else None

    def _probe_metrics(
        self,
        code: str,
        entrypoint: str,
        selected: CallOutcome,
        arity: int,
    ) -> tuple[float, int]:
        if arity == 0:
            return 1.0, 1
        base = list(selected.args)
        probes = [tuple(base)]
        for i in range(arity):
            changed = base[:]
            changed[i] += 1 if changed[i] != 0 else 2
            probes.append(tuple(changed))
        outcomes = self._execute_many(code, entrypoint, probes)
        base_out = outcomes[0]
        changed = sum(
            (out.ok, out.value, out.error) != (base_out.ok, base_out.value, base_out.error)
            for out in outcomes[1:]
        )
        diversity = len({(x.ok, x.value, x.error) for x in outcomes})
        return changed / arity, diversity

    @staticmethod
    def _namespace():
        return {
            "__builtins__": {
                "abs": abs,
                "len": len,
                "list": list,
                "max": max,
                "min": min,
                "range": range,
                "reversed": reversed,
                "sorted": sorted,
                "sum": sum,
                "enumerate": enumerate,
                "zip": zip,
                "IndexError": IndexError,
                "ZeroDivisionError": ZeroDivisionError,
                "ValueError": ValueError,
                "KeyError": KeyError,
                "Exception": Exception,
            }
        }

    def _execute_many(
        self, code: str, entrypoint: str, args_list: Iterable[tuple[int, ...]]
    ) -> list[CallOutcome]:
        ns = self._namespace()
        args_list = list(args_list)
        try:
            compiled = compile(code, "<imperative-mesopy>", "exec")
            exec(compiled, ns, ns)
            fn = ns[entrypoint]
        except Exception as exc:
            err = type(exc).__name__
            return [CallOutcome(tuple(args), False, None, err) for args in args_list]
        results = []
        for args in args_list:
            start = time.perf_counter()
            try:
                value = fn(*args)
                results.append(CallOutcome(tuple(args), True, repr(value), None, time.perf_counter() - start))
            except Exception as exc:
                results.append(CallOutcome(tuple(args), False, None, type(exc).__name__, time.perf_counter() - start))
        return results

    def _profile(
        self, code: str, entrypoint: str, args: tuple[int, ...], max_steps: int
    ) -> CallOutcome:
        ns = self._namespace()
        filename = "<imperative-mesopy-profile>"
        compiled = compile(code, filename, "exec")
        exec(compiled, ns, ns)
        fn = ns[entrypoint]
        steps = 0
        lines: set[int] = set()

        class _StepLimit(Exception):
            pass

        def trace(frame, event, arg):
            nonlocal steps
            if event == "line" and frame.f_code.co_filename == filename:
                steps += 1
                lines.add(frame.f_lineno)
                if steps > max_steps:
                    raise _StepLimit
            return trace

        start = time.perf_counter()
        old_trace = sys.gettrace()
        try:
            sys.settrace(trace)
            value = fn(*args)
            return CallOutcome(args, True, repr(value), None, time.perf_counter() - start, steps, len(lines))
        except _StepLimit:
            return CallOutcome(args, False, None, "StepLimit", time.perf_counter() - start, steps, len(lines))
        except Exception as exc:
            return CallOutcome(args, False, None, type(exc).__name__, time.perf_counter() - start, steps, len(lines))
        finally:
            sys.settrace(old_trace)

    @staticmethod
    def _mutation_sites(tree: ast.AST):
        sites = []
        for idx, node in enumerate(ast.walk(tree)):
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                sites.append((idx, "compare"))
            elif isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                sites.append((idx, "constant"))
            elif isinstance(node, ast.AugAssign) and isinstance(node.op, (ast.Add, ast.Sub)):
                sites.append((idx, "augop"))
            elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
                sites.append((idx, "binop"))
        return sites

    def _apply_mutation(self, tree: ast.AST, path: int, mutation: str) -> str | None:
        nodes = list(ast.walk(tree))
        if path >= len(nodes):
            return None
        node = nodes[path]
        if mutation == "compare" and isinstance(node, ast.Compare):
            op = node.ops[0]
            swaps = {
                ast.Lt: ast.LtE,
                ast.LtE: ast.Lt,
                ast.Gt: ast.GtE,
                ast.GtE: ast.Gt,
                ast.Eq: ast.NotEq,
                ast.NotEq: ast.Eq,
            }
            cls = swaps.get(type(op))
            if cls:
                node.ops[0] = cls()
                return f"{type(op).__name__}->{cls.__name__}"
        if mutation == "constant" and isinstance(node, ast.Constant) and isinstance(node.value, int):
            delta = self.rng.choice((-1, 1))
            old = node.value
            node.value += delta
            return f"constant {old}->{node.value}"
        if mutation == "augop" and isinstance(node, ast.AugAssign):
            old = type(node.op)
            node.op = ast.Sub() if isinstance(node.op, ast.Add) else ast.Add()
            return f"{old.__name__}->{type(node.op).__name__}"
        if mutation == "binop" and isinstance(node, ast.BinOp):
            old = type(node.op)
            node.op = ast.Sub() if isinstance(node.op, ast.Add) else ast.Add()
            return f"{old.__name__}->{type(node.op).__name__}"
        return None
