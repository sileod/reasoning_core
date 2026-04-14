"""
type_inhabitation_fixed.py
──────────────────────────
Drop-in replacement for the TypeInhabitation task section.

Key changes vs the original:
  1. All candidates (valid AND invalid) are generated from the TOOLKIT only.
     No random functions from the global index can appear.
  2. Structural type-checker (_type_check_expr) validates expressions without
     eval, subprocess, or any real imports.  It walks the expression tree and
     checks each call's argument types against the function signature.
  3. Tree generation (_generate_expr_tree) produces random compositions of
     exact `depth` from the toolkit, with correct or deliberately wrong wiring.
  4. The generate() loop order is fixed: build toolkit first, then synthesise.
"""

from __future__ import annotations

import random
import string
import re

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


# ── assume these are already imported from the main file ──────────────────────
# FunctionRecord, TypeIndex, _ProgramBase, Task, Problem, Config,
# WEAK_RETURN_TYPES, _is_clean_type, _get_clean_functions
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL TYPE CHECKER  (no eval, no deps, no subprocess)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_func_lookup(funcs: list) -> dict[str, object]:
    """Map function name → FunctionRecord for quick lookup."""
    return {f.name: f for f in funcs}

def _build_masking_map(funcs):
    name_map = {}
    reverse_map = {}

    for i, f in enumerate(funcs):
        masked = f"f{i}"
        name_map[f.name] = masked
        reverse_map[masked] = f.name

    return name_map, reverse_map


# def _mask_expr(expr: str, name_map: dict) -> str:
#     """
#     Safely replace function names in expressions.
#     Example: bernoulli(x=...) -> f0(x=...)
#     """
#     # Sort by length to avoid partial replacement issues
#     for real in sorted(name_map.keys(), key=len, reverse=True):
#         expr = expr.replace(real + "(", name_map[real] + "(")
#     return expr

def _mask_expr(expr: str, name_map: dict) -> str:
    for real in sorted(name_map.keys(), key=len, reverse=True):
        expr = re.sub(rf'\b{re.escape(real)}\s*\(', name_map[real] + "(", expr)
    return expr

def _type_check_expr(
    expr_str: str,
    func_lookup: dict[str, object],
    var_types: dict[str, str],
) -> tuple[bool, str, str]:
    """
    Structurally type-check an expression string WITHOUT eval.

    Returns:
        (is_valid, inferred_return_type, explanation)

    Strategy:
        Recursive descent over the expression string.
        We don't actually parse a full AST — we use a lightweight
        tokeniser that recognises `name(k=v, k=v, ...)` and bare names.

    Limitations (acceptable for dataset generation):
        - Only handles keyword arguments (name=expr form).
        - Does not handle *args / **kwargs.
        - Nested calls are resolved recursively.
        - If a function name is unknown, returns (False, '?', reason).
    """
    expr_str = expr_str.strip()

    # ── Base case: bare variable name ─────────────────────────────────────────
    if expr_str in var_types:
        return True, var_types[expr_str], f"`{expr_str}` is a known variable"

    # ── Find the outermost function call ──────────────────────────────────────
    paren_pos = expr_str.find("(")
    if paren_pos == -1:
        # Not a call and not a known variable
        return False, "?", f"Unknown token `{expr_str}`"

    func_name = expr_str[:paren_pos].strip()
    if func_name not in func_lookup:
        return False, "?", f"Function `{func_name}` not in toolkit"

    func = func_lookup[func_name]

    # Strip outer parens and split keyword args
    inner = expr_str[paren_pos + 1:].rstrip(")")
    if not inner.strip():
        # Zero-argument call
        if func.inputs:
            return (
                False,
                "?",
                f"`{func_name}` requires {len(func.inputs)} args but got 0",
            )
        return True, func.output, f"`{func_name}()` → {func.output}"

    # Split on top-level commas only
    arg_parts = _split_top_level_commas(inner)

    # Parse each keyword argument
    provided: dict[str, str] = {}   # param_name → (sub_expr string)
    for part in arg_parts:
        if "=" not in part:
            return False, "?", f"Non-keyword arg in `{expr_str}`: `{part}`"
        k, _, v = part.partition("=")
        provided[k.strip()] = v.strip()

    # Resolve each parameter
    expected_params = {name: typ for name, typ in func.inputs}
    errors = []

    for pname, ptype in func.inputs:
        if pname not in provided:
            errors.append(f"Missing arg `{pname}: {ptype}`")
            continue
        sub_expr = provided[pname]
        sub_ok, sub_type, sub_reason = _type_check_expr(
            sub_expr, func_lookup, var_types
        )
        if not sub_ok:
            errors.append(f"Arg `{pname}`: {sub_reason}")
            continue
        if not _types_compatible(sub_type, ptype):
            errors.append(
                f"Arg `{pname}` expected `{ptype}` but got `{sub_type}`"
            )

    for k in provided:
        if k not in expected_params:
            errors.append(f"Unknown param `{k}`")

    if errors:
        return False, "?", "; ".join(errors)

    return True, func.output, f"`{func_name}(...)` → {func.output}"


def _types_compatible(actual: str, expected: str) -> bool:
    """
    Loose type compatibility.
    Handles Union types like `torch.Tensor | float`.
    """
    if actual == expected:
        return True
    # Expand Union on expected side
    if "|" in expected:
        parts = [p.strip() for p in expected.split("|")]
        return any(_types_compatible(actual, p) for p in parts)
    # Expand Union on actual side
    if "|" in actual:
        parts = [p.strip() for p in actual.split("|")]
        return any(_types_compatible(p, expected) for p in parts)
    return False


def _split_top_level_commas(s: str) -> list[str]:
    """Split on commas that are not inside parentheses."""
    parts = []
    depth = 0
    buf = []
    for ch in s:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


# ═══════════════════════════════════════════════════════════════════════════════
# EXPRESSION TREE GENERATOR  (from toolkit only)
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_expr_tree(
    target_type: str,
    funcs: list,
    var_types: dict[str, str],      # name → type
    rng: random.Random,
    depth: int,
    introduce_error: bool = False,  # if True, deliberately mis-wire one arg
) -> Optional[str]:
    """
    Recursively build an expression of `target_type` with exact `depth` levels.

    - At depth 0: return a variable of the right type, or None if unavailable.
    - At depth > 0: pick a function returning target_type, fill its args
      recursively at depth-1.
    - introduce_error: at the topmost call, swap one argument's type to a
      wrong type to create a deliberately invalid expression.

    Returns an expression string, or None if construction fails.
    """
    by_output: dict[str, list] = defaultdict(list)
    for f in funcs:
        by_output[f.output].append(f)

    vars_by_type: dict[str, list[str]] = defaultdict(list)
    for name, t in var_types.items():
        vars_by_type[t].append(name)

    def build(req_type: str, d: int, apply_error: bool) -> Optional[str]:
        if d == 0:
            # Leaf: use a variable
            candidates = vars_by_type.get(req_type, [])
            if not candidates:
                # Try Union components
                for vname, vtype in var_types.items():
                    if _types_compatible(vtype, req_type):
                        candidates.append(vname)
            if not candidates:
                return None
            return rng.choice(candidates)

        pool = by_output.get(req_type, [])
        if not pool:
            # Fall back to leaf
            # return build(req_type, 0, False)
            return None

        rng.shuffle(pool)
        for func in pool:
            if not func.inputs:
                return f"{func.name}()"

            # For error injection: choose one param to mis-wire
            error_param = None
            if apply_error and func.inputs:
                error_param = rng.choice(func.inputs)[0]

            args = []
            failed = False
            for pname, ptype in func.inputs:
                if error_param and pname == error_param:
                    # Intentionally pick a variable of the WRONG type
                    wrong_vars = [
                        n for n, t in var_types.items()
                        if not _types_compatible(t, ptype)
                    ]
                    if wrong_vars:
                        args.append(f"{pname}={rng.choice(wrong_vars)}")
                    else:
                        # Can't find a wrong-type var; skip error for this param
                        sub = build(ptype, d - 1, False)
                        if sub is None:
                            failed = True
                            break
                        args.append(f"{pname}={sub}")
                else:
                    sub = build(ptype, d - 1, False)
                    if sub is None:
                        failed = True
                        break
                    args.append(f"{pname}={sub}")

            if not failed:
                return f"{func.name}({', '.join(args)})"

        # All functions exhausted — fall back to leaf
        return build(req_type, 0, False)

    return build(target_type, depth, introduce_error)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOLKIT SELECTOR  (unchanged logic, kept here for completeness)
# ═══════════════════════════════════════════════════════════════════════════════

def _select_toolkit_v2(
    funcs: list,
    target_type: str,
    rng: random.Random,
    n_funcs: int = 8,
) -> tuple[list, dict[str, str]]:
    """
    Select toolkit and return:
        (selected_functions, var_types_dict)

    var_types_dict maps variable_name → type_str.
    """
    by_output: dict[str, list] = defaultdict(list)
    for f in funcs:
        by_output[f.output].append(f)

    if not by_output.get(target_type):
        return [], {}

    selected = []
    seen_names: set[str] = set()

    # Always include 1-2 functions returning target type
    target_funcs = rng.sample(
        by_output[target_type], min(2, len(by_output[target_type]))
    )
    for f in target_funcs:
        if f.name not in seen_names:
            selected.append(f)
            seen_names.add(f.name)

    # Collect needed input types and add producers
    needed_types: set[str] = set()
    for f in selected:
        for _, t in f.inputs:
            needed_types.add(t)

    for t in list(needed_types):
        if t == target_type:
            continue
        producers = by_output.get(t, [])
        if producers and len(selected) < n_funcs:
            f = rng.choice(producers)
            if f.name not in seen_names:
                selected.append(f)
                seen_names.add(f.name)

    # Fill remaining with diverse functions
    seen_outputs = {f.output for f in selected}
    remaining = [
        f for f in funcs
        if f.name not in seen_names and f.output not in seen_outputs
    ]
    rng.shuffle(remaining)
    for f in remaining:
        if len(selected) >= n_funcs:
            break
        selected.append(f)
        seen_names.add(f.name)
        seen_outputs.add(f.output)

    # Derive leaf input variables
    produced_types = {f.output for f in selected}
    leaf_types: set[str] = set()
    for f in selected:
        for _, t in f.inputs:
            if t not in produced_types:
                leaf_types.add(t)
    for f in target_funcs:
        for _, t in f.inputs:
            leaf_types.add(t)

    type_counts: dict[str, int] = defaultdict(int)
    var_types: dict[str, str] = {}
    for t in sorted(leaf_types):
        prefix = t.lower()[:3].replace(" ", "_").replace("|", "")[:3]
        vname = f"{prefix}_{type_counts[t]}"
        type_counts[t] += 1
        var_types[vname] = t
        if rng.random() < 0.4:
            vname2 = f"{prefix}_{type_counts[t]}"
            type_counts[t] += 1
            var_types[vname2] = t

    return selected, var_types

def _generate_expr_tree_strict(
    target_type: str,
    funcs: list,
    var_types: dict[str, str],
    rng: random.Random,
    depth: int,
    introduce_error: bool = False,
) -> Optional[str]:
    """
    Generate ONE valid expression of exact depth
    with constraint: each function used at most once.
    """

    from collections import defaultdict

    by_output: dict[str, list] = defaultdict(list)
    for f in funcs:
        by_output[f.output].append(f)

    def build(req_type: str, d: int, used_funcs: set[str]) -> Optional[str]:

        # ── BASE CASE ─────────────────────────────────────────────
        if d == 0:
            candidates = [
                name for name, t in var_types.items()
                if _types_compatible(t, req_type)
            ]
            return rng.choice(candidates) if candidates else None

        # ── FUNCTION CASE ─────────────────────────────────────────
        pool = by_output.get(req_type, [])
        if not pool:
            return None

        rng.shuffle(pool)

        for func in pool:

            # enforce uniqueness
            if func.name in used_funcs:
                continue

            # zero-arg function
            if not func.inputs:
                if d == 1:
                    return f"{func.name}()"
                continue

            chain_param = rng.choice(func.inputs)[0]

            args = []
            failed = False

            for pname, ptype in func.inputs:

                if pname == chain_param:
                    sub = build(ptype, d - 1, used_funcs | {func.name})
                else:
                    sub = build(ptype, 0, used_funcs)

                if sub is None:
                    failed = True
                    break

                args.append(f"{pname}={sub}")

            if not failed:
                return f"{func.name}({', '.join(args)})"

        return None

    return build(target_type, depth, set())

def _expr_depth(expr: str) -> int:
    return expr.count("(")

def _enumerate_valid_exprs(
    target_type: str,
    funcs: list,
    var_types: dict[str, str],
    depth: int,
    max_results: int = 50,
) -> list[str]:

    from collections import defaultdict
    from itertools import product

    by_output: dict[str, list] = defaultdict(list)
    for f in funcs:
        by_output[f.output].append(f)

    cache = {}

    def build(req_type: str, d: int, used: frozenset):
        key = (req_type, d, used)
        if key in cache:
            return cache[key]

        results = []

        # ── BASE CASE ─────────────────────────────
        if d == 0:
            for name, t in var_types.items():
                if _types_compatible(t, req_type):
                    results.append(name)
            cache[key] = results
            return results

        # ── FUNCTION CASE ─────────────────────────
        for func in by_output.get(req_type, []):

            if func.name in used:
                continue  # enforce uniqueness

            if not func.inputs:
                if d == 1:
                    results.append(f"{func.name}()")
                continue

            for chain_param, chain_type in func.inputs:

                sub_chains = build(
                    chain_type,
                    d - 1,
                    used | {func.name}
                )
                if not sub_chains:
                    continue

                arg_options = []

                for pname, ptype in func.inputs:
                    if pname == chain_param:
                        arg_options.append([
                            f"{pname}={sc}" for sc in sub_chains
                        ])
                    else:
                        leafs = build(ptype, 0, used)
                        if not leafs:
                            arg_options = []
                            break
                        arg_options.append([
                            f"{pname}={lf}" for lf in leafs
                        ])

                if not arg_options:
                    continue

                for combo in product(*arg_options):
                    expr = f"{func.name}({', '.join(combo)})"
                    results.append(expr)

                    if len(results) >= max_results:
                        cache[key] = results
                        return results

        cache[key] = results
        return results

    return build(target_type, depth, frozenset())

# ═══════════════════════════════════════════════════════════════════════════════
# TASK CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TypeInhabitationCfg:
    """
    depth          — exact composition depth for candidate expressions.
                     depth=1 → single function call f(var)
                     depth=2 → f(g(var)) or f(var1, g(var2))
                     depth=3 → deeper nesting
    n_candidates   — total candidates shown (valid + invalid).
    n_valid        — how many valid candidates to include.
    max_gen_tries  — retry budget.
    db_path        — path to functions DB.
    seed           — random seed.
    """
    depth: int = 2
    n_candidates: int = 4
    n_valid: int = 1
    max_gen_tries: int = 2000
    db_path: str = "functions.db"
    seed: Optional[int] = None

    difficulty_mode: Optional[str] = None 

    def update(self, delta: int):
        self.depth += delta
        self.n_valid = min(self.n_valid + delta, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK
# ═══════════════════════════════════════════════════════════════════════════════

class TypeInhabitation:
    """
    Given a typed toolkit of functions and variables, identify the valid
    expression(s) producing the target type.

    All candidates are generated from the toolkit only — no outside functions
    appear in the prompt or candidates list.

    Type-checking is structural (no eval, no subprocess, no imports).
    """

    def __init__(self, config=None):
        if config is None:
            config = TypeInhabitationCfg()
        self.config = config
        self._index_cache: dict[str, object] = {}

    def _get_index(self):
        db = self.config.db_path
        if db not in self._index_cache:
            from program_composition_task import TypeIndex
            self._index_cache[db] = TypeIndex(db)
        return self._index_cache[db]

    def generate(self):
        from program_composition_task import (
            _get_clean_functions, _is_clean_type
        )

        index = self._get_index()
        rng = random.Random(self.config.seed)
        cfg = self.config

        # ── difficulty → toolkit size ─────────────────────────────────────────
        if cfg.difficulty_mode == "easy":
            n_funcs = 8
        elif cfg.difficulty_mode == "medium":
            n_funcs = 16
        else:
            n_funcs = 8

        clean_funcs = _get_clean_functions(index)
        if not clean_funcs:
            raise RuntimeError("No clean functions found in database.")

        # ── group functions by return type ─────────────────────────────────────
        from collections import defaultdict
        clean_by_output: dict[str, list] = defaultdict(list)
        for f in clean_funcs:
            clean_by_output[f.output].append(f)

        # ── pick viable target types ───────────────────────────────────────────
        viable_targets = [
            t for t, fs in clean_by_output.items()
            if len(fs) >= 2 and _is_clean_type(t)
        ]
        if not viable_targets:
            raise RuntimeError("No viable target types.")

        # ──────────────────────────────────────────────────────────────────────
        # MAIN GENERATION LOOP
        # ──────────────────────────────────────────────────────────────────────
        for attempt in range(cfg.max_gen_tries):

            # ── 1. Pick target + toolkit ──────────────────────────────────────
            target_type = rng.choice(viable_targets)

            selected_funcs, var_types = _select_toolkit_v2(
                funcs=clean_funcs,
                target_type=target_type,
                rng=rng,
                n_funcs=n_funcs,
            )
            if not selected_funcs or not var_types:
                continue

            funcs_for_target = [
                f for f in selected_funcs if f.output == target_type
            ]
            if not funcs_for_target:
                continue

            func_lookup = _build_func_lookup(selected_funcs)
            name_map, reverse_map = _build_masking_map(selected_funcs)

            # ── 2. Enumerate ALL valid expressions ────────────────────────────
            # valid_exprs_full = _enumerate_valid_exprs(
            #     target_type=target_type,
            #     funcs=selected_funcs,
            #     var_types=var_types,
            #     depth=cfg.depth,
            #     max_results=200,
            # )
            valid_exprs_full = list(set(_enumerate_valid_exprs(
                target_type=target_type,
                funcs=selected_funcs,
                var_types=var_types,
                depth=cfg.depth,
                max_results=200,
            )))

            if not valid_exprs_full:
                continue  # retry toolkit

            # ── 3. Select valid expressions for candidates ────────────────────
            # valid_exprs = valid_exprs_full[:cfg.n_valid]
            valid_exprs = rng.sample(valid_exprs_full, cfg.n_valid)

            # ── 4. Generate invalid expressions ───────────────────────────────
            n_invalid = cfg.n_candidates - len(valid_exprs)
            invalid_exprs = []

            for _ in range(n_invalid * 30):
                expr = _generate_expr_tree(
                    target_type=target_type,
                    funcs=selected_funcs,
                    var_types=var_types,
                    rng=rng,
                    depth=cfg.depth,
                    introduce_error=True,
                )
                if expr is None:
                    continue

                ok, _, _ = _type_check_expr(expr, func_lookup, var_types)

                if not ok and "(" in expr:
                    if expr not in invalid_exprs and expr not in valid_exprs_full:
                        invalid_exprs.append(expr)

                if len(invalid_exprs) >= n_invalid:
                    break

            # If not enough invalid → retry
            if len(invalid_exprs) < n_invalid:
                continue

            # ── 5. Build candidates ──────────────────────────────────────────
            # candidates = (
            #     [{"expr": e, "is_correct": True} for e in valid_exprs] +
            #     [{"expr": e, "is_correct": False} for e in invalid_exprs]
            # )
            candidates = (
                [{"expr": _mask_expr(e, name_map), "is_correct": True} for e in valid_exprs] +
                [{"expr": _mask_expr(e, name_map), "is_correct": False} for e in invalid_exprs]
            )

            rng.shuffle(candidates)

            # ── 6. Annotate candidates ───────────────────────────────────────
            # for c in candidates:
            #     ok, ret_type, explanation = _type_check_expr(
            #         c["expr"], func_lookup, var_types
            #     )
            #     c["type_ok"] = ok
            #     c["inferred_type"] = ret_type
            #     c["explanation"] = explanation
            for c in candidates:
                # UNMASK before type checking
                real_expr = c["expr"]
                # for masked, real in reverse_map.items():
                #     real_expr = real_expr.replace(masked + "(", real + "(")
                for masked, real in reverse_map.items():
                    real_expr = re.sub(rf'\b{re.escape(masked)}\s*\(', real + "(", real_expr)

                ok, ret_type, explanation = _type_check_expr(
                    real_expr, func_lookup, var_types
                )

                # Mask explanation back for display
                explanation = _mask_expr(explanation, name_map)

                c["type_ok"] = ok
                c["inferred_type"] = ret_type
                c["explanation"] = explanation

            # ── 7. Difficulty ────────────────────────────────────────────────
            # if len(valid_exprs_full) == 1:
            #     difficulty = "easy"
            # elif len(valid_exprs_full) <= 3:
            #     difficulty = "medium"
            # else:
            #     difficulty = "hard"
            # ── 7. Difficulty (depth-based) ─────────────────────────────────
            if cfg.difficulty_mode:
                difficulty = cfg.difficulty_mode
            else:
                if cfg.depth <= 2:
                    difficulty = "easy"
                elif cfg.depth <= 4:
                    difficulty = "medium"
                else:
                    difficulty = "hard"

            # ── 8. Format toolkit ────────────────────────────────────────────
            # toolkit_funcs_str = "\n".join(
            #     f"  {f.name}({', '.join(f'{n}: {t}' for n, t in f.inputs)}) → {f.output}"
            #     for f in selected_funcs
            # )

            toolkit_funcs_str = "\n".join(
                f"  {name_map[f.name]}({', '.join(f'{n}: {t}' for n, t in f.inputs)}) → {f.output}"
                for f in selected_funcs
            )

            toolkit_vars_str = "\n".join(
                f"  {name}: {t}" for name, t in sorted(var_types.items())
            )

            # ── 9. Build CoT ─────────────────────────────────────────────────
            cot_lines = [
                "Reasoning step by step:",
                f"1. Target type is `{target_type}`.",
                "2. Valid expressions must:",
                "   - match target type",
                "   - respect argument types",
                "   - use each function at most once",
                f"   - have exact depth {cfg.depth}",
                f"3. Found {len(valid_exprs_full)} valid expressions.",
                "4. Type-checking candidates:",
            ]

            for c in candidates:
                tick = "✓" if c["is_correct"] else "✗"
                cot_lines.append(
                    f"   - `{c['expr']}`\n"
                    f"     {tick} {c['explanation']}"
                )

            cot_lines.append(
                "5. Valid expression(s): "
                + ", ".join(f"`{e}`" for e in valid_exprs)
            )

            # cot = "\n".join(cot_lines)
            cot = _mask_expr("\n".join(cot_lines), name_map)

            # ── 10. Metadata ────────────────────────────────────────────────
            meta = {
                "func_lookup": func_lookup,
                "var_types": var_types,
                "target_type": target_type,
                "depth": cfg.depth,
                "toolkit_funcs": toolkit_funcs_str,
                "toolkit_vars": toolkit_vars_str,
                "candidates": candidates,
                "valid_exprs": [_mask_expr(e, name_map) for e in valid_exprs_full],
                "all_valid_exprs": [_mask_expr(e, name_map) for e in valid_exprs_full],
                "invalid_exprs": invalid_exprs,
                "difficulty": difficulty,
                "task_type": "type_inhabitation",
                "cot": cot,
                "name_map": name_map,
                "code_hash": str(
                    hash(valid_exprs_full[0] + target_type + str(attempt))
                ),
            }

            # ── 11. Problem wrapper ─────────────────────────────────────────
            class _Problem:
                def __init__(self, metadata, answer):
                    self.metadata = metadata
                    self.answer = answer
                    self.prompt = None
                    self.cot = metadata.get("cot", "")

            prob = _Problem(metadata=meta, answer=valid_exprs_full)

            prob.prompt = self.prompt(meta)
            return prob

        # ── failure ─────────────────────────────────────────────────────────
        raise RuntimeError(
            f"TypeInhabitation: could not generate example in "
            f"{cfg.max_gen_tries} attempts."
        )

    def prompt(self, metadata: dict) -> str:
        cand_block = "\n".join(
            f"  {chr(65 + i)}) {c['expr']}"
            for i, c in enumerate(metadata["candidates"])
        )
        return (
            "You are given a typed toolkit of functions and variables.\n\n"
            f"Functions:\n{metadata['toolkit_funcs']}\n\n"
            f"Variables:\n{metadata['toolkit_vars']}\n\n"
            f"Target type: `{metadata['target_type']}`  "
            f"(composition depth: {metadata['depth']})\n\n"
            "Write a valid expression (function chain) of EXACT depth "
            f"{metadata['depth']} that produces the target type.\n\n"
            "Constraints:\n"
            "- Use only the given functions and variables\n"
            "- Each function may be used AT MOST ONCE\n"
            "- All argument types must be valid\n"
        )

    def score_answer(self, answer, entry) -> float:
        ans = str(answer).strip()
        norm = lambda x: x.strip().replace(" ", "")

        # ── 1. Multiple choice ─────────────────────────────
        candidates = entry.metadata.get("candidates", [])
        if len(ans) == 1 and ans.upper() in string.ascii_uppercase:
            idx = ord(ans.upper()) - ord("A")
            if 0 <= idx < len(candidates):
                return float(candidates[idx]["is_correct"])

        # ── 2. Exact match (strong signal) ─────────────────
        valid_exprs = entry.metadata.get("valid_exprs", [])
        if any(norm(ans) == norm(v) for v in valid_exprs):
            return 1.0

        # ── 3. Type-check fallback (CRUCIAL) ───────────────
        func_lookup = entry.metadata.get("func_lookup")
        var_types = entry.metadata.get("var_types")
        target_type = entry.metadata.get("target_type")
        depth = entry.metadata.get("depth")

        # ok, ret_type, _ = _type_check_expr(ans, func_lookup, var_types)
        name_map = entry.metadata.get("name_map", {})
        reverse_map = {v: k for k, v in name_map.items()}

        real_ans = ans
        for masked, real in reverse_map.items():
            real_ans = real_ans.replace(masked + "(", real + "(")

        ok, ret_type, _ = _type_check_expr(real_ans, func_lookup, var_types)

        if ok and ret_type == target_type:
            # optionally check depth here
            return 0.5

        return 0.0

    def deduplication_key(self, problem):
        return problem.metadata.get("code_hash", None)


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE DEMO  (python type_inhabitation_fixed.py --n 3 --depth 2)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="functions.db")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-valid", type=int, default=1)
    parser.add_argument("--n-candidates", type=int, default=4)
    args = parser.parse_args()

    cfg = TypeInhabitationCfg(
        db_path=args.db,
        depth=args.depth,
        seed=args.seed,
        n_valid=args.n_valid,
        n_candidates=args.n_candidates,
    )
    task = TypeInhabitation(config=cfg)
    # for i in range(args.n):
    #     print(f"\n{'═' * 60}")
    #     print(f"  [INHABIT] EXAMPLE {i + 1}  (depth={cfg.depth})")
    #     print(f"{'═' * 60}")
    #     # Vary seed per example
    #     cfg.seed = (args.seed or 0) + i
    #     ex = task.generate()
    #     print("\n── Prompt ──")
    #     print(ex.prompt)
    #     print("\n── All valid expressions (full set) ──")
    #     for ve in ex.metadata.get("all_valid_exprs", []):
    #         print(f"  {ve}")
    #     # print("\n── All candidates with type-check detail ──")
    #     # for c in ex.metadata["candidates"]:
    #     #     mark = "✓" if c["is_correct"] else "✗"
    #     #     print(f"  {mark} {c['expr']}")
    #     #     print(f"    {c['explanation']}")
    #     # print("\n── CoT ──")
    #     # print(ex.cot)
    #     print(f"\n── Difficulty: {ex.metadata['difficulty']} ──")
    #     # print(f"── Self-check score: {task.score_answer(ex.answer, ex)} ──")
    for i in range(args.n):
        print(f"\n{'═' * 60}")
        print(f"  [INHABIT] EXAMPLE {i + 1}  (depth={cfg.depth})")
        print(f"{'═' * 60}")

        success = False
        retries = 0
        max_retries = 100  # you can tune this

        while not success and retries < max_retries:
            try:
                cfg.seed = (args.seed or 0) + i + retries
                ex = task.generate()
                success = True
            except RuntimeError as e:
                if "could not generate example" in str(e):
                    retries += 1
                    print(f"---- Retry {retries}/{max_retries}...")
                else:
                    raise  # real bug → crash

        if not success:
            print("---- Skipping example after repeated failures.")
            continue

        print("\n── Prompt ──")
        print(ex.prompt)

        print("\n── All valid expressions (full set) ──")
        for ve in ex.metadata.get("all_valid_exprs", []):
            print(f"  {ve}")

        print(f"\n── Difficulty: {ex.metadata['difficulty']} ──")