from __future__ import annotations

import ast
import copy
import hashlib
import random

from ._imperative_mesopy_types import _REALISTIC_NAMES

def _bound_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
            names.update(arg.arg for arg in node.args.args)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names

def _alpha_rename(module: ast.Module, entrypoint: str, rng: random.Random) -> tuple[ast.Module, str]:
    module = copy.deepcopy(module)
    bound = sorted(_bound_names(module))
    pool = list(_REALISTIC_NAMES)
    rng.shuffle(pool)
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for old in bound:
        while pool:
            candidate = pool.pop()
            if candidate not in used:
                break
        else:
            candidate = f"item{len(mapping) + 1}"
        used.add(candidate)
        mapping[old] = candidate

    class Rename(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.name = mapping.get(node.name, node.name)
            for arg in node.args.args:
                arg.arg = mapping.get(arg.arg, arg.arg)
            self.generic_visit(node)
            return node

        def visit_arg(self, node):
            node.arg = mapping.get(node.arg, node.arg)
            return node

        def visit_Name(self, node):
            node.id = mapping.get(node.id, node.id)
            return node

    module = Rename().visit(module)
    ast.fix_missing_locations(module)
    return module, mapping.get(entrypoint, entrypoint)

def structural_fingerprint(code: str) -> str:
    tree = ast.parse(code)
    mapping: dict[str, str] = {}

    def anon(name: str) -> str:
        if name not in mapping:
            mapping[name] = f"v{len(mapping)}"
        return mapping[name]

    bound = _bound_names(tree)

    class Normalize(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.name = anon(node.name)
            self.generic_visit(node)
            return node

        def visit_arg(self, node):
            node.arg = anon(node.arg)
            return node

        def visit_Name(self, node):
            if node.id in bound:
                node.id = anon(node.id)
            return node

        def visit_Constant(self, node):
            if isinstance(node.value, bool) or node.value is None:
                return node
            if isinstance(node.value, int):
                return ast.copy_location(ast.Constant(0), node)
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(""), node)
            return node

    norm = Normalize().visit(tree)
    ast.fix_missing_locations(norm)
    dump = ast.dump(norm, annotate_fields=False, include_attributes=False)
    return hashlib.sha256(dump.encode()).hexdigest()

def _rw(stmt: ast.AST) -> tuple[set[str], set[str]]:
    reads: set[str] = set()
    writes: set[str] = set()
    for node in ast.walk(stmt):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                reads.add(node.id)
            elif isinstance(node.ctx, ast.Store):
                writes.add(node.id)
    return reads, writes

def _has_effect(stmt: ast.stmt) -> bool:
    if isinstance(stmt, (ast.Return, ast.Raise, ast.Try)):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return True
    for node in ast.walk(stmt):
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Subscript):
            return True
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Subscript) for t in node.targets):
            return True
    return False

def _dependency_depth(fn: ast.FunctionDef) -> int:
    def block(stmts: list[ast.stmt], incoming: dict[str, int]) -> tuple[dict[str, int], int]:
        depths = dict(incoming)
        returned = 0
        for stmt in stmts:
            if isinstance(stmt, ast.Return):
                reads, _ = _rw(stmt)
                returned = max(returned, 1 + max((depths.get(name, 0) for name in reads), default=0))
                continue
            if isinstance(stmt, ast.If):
                yes, yes_ret = block(stmt.body, depths)
                no, no_ret = block(stmt.orelse, depths)
                for name in set(yes) | set(no):
                    depths[name] = max(depths.get(name, 0), yes.get(name, 0), no.get(name, 0))
                returned = max(returned, yes_ret, no_ret)
                continue
            if isinstance(stmt, (ast.For, ast.While)):
                child, child_ret = block(stmt.body, depths)
                for name, value in child.items():
                    depths[name] = max(depths.get(name, 0), value)
                returned = max(returned, child_ret)
                continue
            if isinstance(stmt, ast.Try):
                branches = [block(stmt.body, depths)]
                branches += [block(handler.body, depths) for handler in stmt.handlers]
                if stmt.finalbody:
                    branches.append(block(stmt.finalbody, depths))
                for child, child_ret in branches:
                    for name, value in child.items():
                        depths[name] = max(depths.get(name, 0), value)
                    returned = max(returned, child_ret)
                continue
            reads, writes = _rw(stmt)
            base = max((depths.get(name, 0) for name in reads), default=0)
            for name in writes:
                if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == name:
                    base = max(base, depths.get(name, 0))
                depths[name] = max(depths.get(name, 0), base + 1)
        return depths, returned

    initial = {arg.arg: 0 for arg in fn.args.args}
    _, returned = block(fn.body, initial)
    return returned

def _liveness_metrics(fn: ast.FunctionDef) -> dict:
    total = 0
    live = 0
    max_slice_depth = 0

    def slice_block(stmts: list[ast.stmt], live_names: set[str], depth: int = 0) -> set[str]:
        nonlocal total, live, max_slice_depth
        current = set(live_names)
        for stmt in reversed(stmts):
            total += 1
            if isinstance(stmt, ast.Return):
                reads, _ = _rw(stmt)
                current |= reads
                live += 1
                max_slice_depth = max(max_slice_depth, depth + 1)
                continue

            if isinstance(stmt, ast.If):
                before = set(current)
                yes = slice_block(stmt.body, set(current), depth + 1)
                no = slice_block(stmt.orelse, set(current), depth + 1)
                if yes != before or no != before or _has_effect(stmt):
                    cond_reads, _ = _rw(stmt.test)
                    current |= yes | no | cond_reads
                    live += 1
                    max_slice_depth = max(max_slice_depth, depth + 1)
                continue

            if isinstance(stmt, (ast.For, ast.While)):
                before = set(current)
                body_live = slice_block(stmt.body, set(current), depth + 1)
                if body_live != before or _has_effect(stmt):
                    reads, writes = _rw(stmt)
                    current |= body_live | reads
                    current -= {w for w in writes if w not in before}
                    live += 1
                    max_slice_depth = max(max_slice_depth, depth + 1)
                continue

            if isinstance(stmt, ast.Try):
                branches = [slice_block(stmt.body, set(current), depth + 1)]
                branches += [slice_block(h.body, set(current), depth + 1) for h in stmt.handlers]
                branches.append(slice_block(stmt.finalbody, set(current), depth + 1))
                current |= set().union(*branches)
                live += 1
                max_slice_depth = max(max_slice_depth, depth + 1)
                continue

            reads, writes = _rw(stmt)
            needed = bool(writes & current) or _has_effect(stmt)
            if needed:
                live += 1
                current -= writes
                current |= reads
                max_slice_depth = max(max_slice_depth, depth + 1)
        return current

    slice_block(fn.body, set())
    return {
        "live_statements": live,
        "total_statements": total,
        "live_fraction": live / total if total else 1.0,
        "backward_slice_depth": _dependency_depth(fn),
        "control_slice_depth": max_slice_depth,
    }

def _structural_features(tree: ast.AST) -> dict:
    nodes = list(ast.walk(tree))

    def depth(node: ast.AST) -> int:
        children = list(ast.iter_child_nodes(node))
        return 1 + max((depth(child) for child in children), default=0)

    def control_depth(node: ast.AST, current: int = 0) -> int:
        here = current + int(isinstance(node, (ast.If, ast.For, ast.While, ast.Try)))
        return max([here] + [control_depth(child, here) for child in ast.iter_child_nodes(node)])

    funcs = [n for n in nodes if isinstance(n, ast.FunctionDef)]
    fn_names = {f.name for f in funcs}
    edges: set[tuple[str, str]] = set()

    class Calls(ast.NodeVisitor):
        current: str | None = None

        def visit_FunctionDef(self, node):
            prev, self.current = self.current, node.name
            self.generic_visit(node)
            self.current = prev

        def visit_Call(self, node):
            if self.current and isinstance(node.func, ast.Name) and node.func.id in fn_names:
                edges.add((self.current, node.func.id))
            self.generic_visit(node)

    Calls().visit(tree)

    def longest(src: str, seen: set[str]) -> int:
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
        "try_blocks": sum(isinstance(n, ast.Try) for n in nodes),
        "mutations": sum(
            isinstance(n, ast.AugAssign) or (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in {"append", "insert", "pop", "reverse"}
            )
            for n in nodes
        ),
    }
