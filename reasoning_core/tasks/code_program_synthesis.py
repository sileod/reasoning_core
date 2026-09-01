from __future__ import annotations

import ast
import hashlib
import itertools
import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from types import CodeType
from typing import Iterable

from reasoning_core.template import Config, Entry, Task, edict, stochastic_rounding as sround


DSL_NAME = "StringFrag-v1"
OPS = (
    "concat",
    "substr",
    "replace1",
    "ite",
    "len",
    "find",
    "add",
    "sub",
    "contains",
    "eq_str",
    "lt",
    "not",
)
OP_INDEX = {name: i for i, name in enumerate(OPS)}
COST_DESCRIPTION = "nodes,ops,source_len,source_lex"
OP_SYNTAX = {
    "concat": "str + str",
    "substr": "str[int:(int)+(int)]",
    "replace1": "str.replace(str, str, 1)",
    "ite": "(str if bool else str)",
    "len": "len(str)",
    "find": "str.find(str)",
    "add": "int + int",
    "sub": "int - int",
    "contains": "str in str",
    "eq_str": "str == str",
    "lt": "int < int",
    "not": "not bool",
}
OP_COST_ORDER = ", ".join(OPS)
SAFE_BUILTINS = {"len": len}
STR_LITS = ('""', '" "', '"-"', '"_"')
INT_LITS = ("0", "1", "2", "3")
SORTS = ("str", "int", "bool")

MIN_INT = -16
MAX_INT = 64
MAX_STR_LEN = 64

ADVERSARIAL_INPUTS = (
    "",
    " ",
    "-",
    "_",
    "a",
    "aa",
    "ab",
    "a-b",
    "a_b",
    "abc",
    " abc",
    "abc ",
    "--",
    "__",
    "0",
)
GLOBAL_PROBES = ADVERSARIAL_INPUTS + (
    "b",
    "ba",
    "a--b",
    "a__b",
    "a b",
    "c-a",
    "_a_",
    "-a-",
    "abc-def",
    "abc_def",
    "cab",
    "bbb",
)
RANDOM_ALPHABET = ("a", "b", "c", " ", "-", "_")


@dataclass(frozen=True)
class Expr:
    sort: str
    py: str
    nodes: int
    ops: tuple[int, ...]
    code: CodeType = field(compare=False, repr=False)
    depends_on_s: bool = False

    @property
    def cost(self) -> tuple:
        return (self.nodes, *self.ops, len(self.py), self.py)

    def run(self, s: str):
        return eval(self.code, {"__builtins__": SAFE_BUILTINS}, {"s": s})

    def signature(self, inputs: tuple[str, ...]) -> tuple | None:
        vals = []
        for inp in inputs:
            try:
                val = self.run(inp)
            except Exception:
                return None
            if not _valid_value(self.sort, val):
                return None
            vals.append(val)
        return tuple(vals)


def _valid_value(sort: str, val) -> bool:
    if sort == "str":
        return isinstance(val, str) and len(val) <= MAX_STR_LEN
    if sort == "int":
        return isinstance(val, int) and not isinstance(val, bool) and MIN_INT <= val <= MAX_INT
    if sort == "bool":
        return isinstance(val, bool)
    return False


def _compile_expr(py: str) -> CodeType:
    return compile(py, "<stringfrag-v1>", "eval")


def _op_counts(*children: Expr, op: str | None = None) -> tuple[int, ...]:
    counts = [0] * len(OPS)
    for child in children:
        for i, n in enumerate(child.ops):
            counts[i] += n
    if op is not None:
        counts[OP_INDEX[op]] += 1
    return tuple(counts)


def _expr(sort: str, py: str, nodes: int, ops: tuple[int, ...], depends_on_s: bool) -> Expr:
    return Expr(sort, py, nodes, ops, _compile_expr(py), depends_on_s)


def _leaves() -> list[Expr]:
    zero_ops = (0,) * len(OPS)
    out = [_expr("str", "s", 1, zero_ops, True)]
    out.extend(_expr("str", lit, 1, zero_ops, False) for lit in STR_LITS)
    out.extend(_expr("int", lit, 1, zero_ops, False) for lit in INT_LITS)
    return out


def _valid_on_inputs(e: Expr, inputs: tuple[str, ...]) -> tuple | None:
    return e.signature(inputs)


def _add_expr(
    e: Expr,
    inputs: tuple[str, ...],
    by_size: dict[str, dict[int, dict[tuple, Expr]]],
    best_for_sig: dict[tuple[str, tuple], Expr],
):
    sig = _valid_on_inputs(e, inputs)
    if sig is None:
        return
    key = (e.sort, sig)
    old = best_for_sig.get(key)
    if old is not None and old.cost <= e.cost:
        return
    best_for_sig[key] = e
    slot = by_size[e.sort].setdefault(e.nodes, {})
    same_size = slot.get(sig)
    if same_size is None or e.cost < same_size.cost:
        slot[sig] = e


def _parts(total: int, arity: int) -> Iterable[tuple[int, ...]]:
    if arity == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(1, total - arity + 2):
        for rest in _parts(total - first, arity - 1):
            yield (first, *rest)


def _children(by_size, sorts: tuple[str, ...], sizes: tuple[int, ...]):
    pools = []
    for sort, size in zip(sorts, sizes):
        pool = list(by_size[sort].get(size, {}).values())
        if not pool:
            return ()
        pools.append(pool)
    return itertools.product(*pools)


def _make_unary(sort: str, op: str, a: Expr, py: str) -> Expr:
    return _expr(sort, py, a.nodes + 1, _op_counts(a, op=op), a.depends_on_s)


def _make_binary(sort: str, op: str, a: Expr, b: Expr, py: str) -> Expr:
    return _expr(sort, py, a.nodes + b.nodes + 1, _op_counts(a, b, op=op), a.depends_on_s or b.depends_on_s)


def _make_ternary(sort: str, op: str, a: Expr, b: Expr, c: Expr, py: str) -> Expr:
    return _expr(
        sort,
        py,
        a.nodes + b.nodes + c.nodes + 1,
        _op_counts(a, b, c, op=op),
        a.depends_on_s or b.depends_on_s or c.depends_on_s,
    )


def _enumerate_core(
    inputs: tuple[str, ...],
    max_nodes: int,
    stop_key: tuple[str, tuple] | None = None,
) -> tuple[dict[tuple[str, tuple], Expr], tuple[Expr, ...], Expr | None]:
    by_size: dict[str, dict[int, dict[tuple, Expr]]] = {sort: {} for sort in SORTS}
    best_for_sig: dict[tuple[str, tuple], Expr] = {}

    for leaf in _leaves():
        _add_expr(leaf, inputs, by_size, best_for_sig)
    if stop_key is not None and stop_key in best_for_sig:
        return best_for_sig, tuple(best_for_sig.values()), best_for_sig[stop_key]

    for nodes in range(2, max_nodes + 1):
        for (n,) in _parts(nodes - 1, 1):
            for (s_expr,) in _children(by_size, ("str",), (n,)):
                _add_expr(_make_unary("int", "len", s_expr, f"len({s_expr.py})"), inputs, by_size, best_for_sig)
            for (c,) in _children(by_size, ("bool",), (n,)):
                _add_expr(_make_unary("bool", "not", c, f"(not {c.py})"), inputs, by_size, best_for_sig)

        for sizes in _parts(nodes - 1, 2):
            for a, b in _children(by_size, ("str", "str"), sizes):
                _add_expr(_make_binary("str", "concat", a, b, f"({a.py} + {b.py})"), inputs, by_size, best_for_sig)
                _add_expr(_make_binary("int", "find", a, b, f"{a.py}.find({b.py})"), inputs, by_size, best_for_sig)
                _add_expr(_make_binary("bool", "contains", a, b, f"({b.py} in {a.py})"), inputs, by_size, best_for_sig)
                _add_expr(_make_binary("bool", "eq_str", a, b, f"({a.py} == {b.py})"), inputs, by_size, best_for_sig)
            for a, b in _children(by_size, ("int", "int"), sizes):
                _add_expr(_make_binary("int", "add", a, b, f"({a.py} + {b.py})"), inputs, by_size, best_for_sig)
                _add_expr(_make_binary("int", "sub", a, b, f"({a.py} - {b.py})"), inputs, by_size, best_for_sig)
                _add_expr(_make_binary("bool", "lt", a, b, f"({a.py} < {b.py})"), inputs, by_size, best_for_sig)

        for sizes in _parts(nodes - 1, 3):
            for a, b, c in _children(by_size, ("str", "int", "int"), sizes):
                _add_expr(_make_ternary("str", "substr", a, b, c, f"{a.py}[{b.py}:({b.py})+({c.py})]"), inputs, by_size, best_for_sig)
            for a, b, c in _children(by_size, ("str", "str", "str"), sizes):
                if b.py == '""':
                    continue
                _add_expr(_make_ternary("str", "replace1", a, b, c, f"{a.py}.replace({b.py}, {c.py}, 1)"), inputs, by_size, best_for_sig)
            for c, a, b in _children(by_size, ("bool", "str", "str"), sizes):
                _add_expr(_make_ternary("str", "ite", c, a, b, f"({a.py} if {c.py} else {b.py})"), inputs, by_size, best_for_sig)

        if stop_key is not None and stop_key in best_for_sig:
            return best_for_sig, tuple(best_for_sig.values()), best_for_sig[stop_key]

    return best_for_sig, tuple(best_for_sig.values()), None


@lru_cache(maxsize=128)
def _enumerate(inputs: tuple[str, ...], max_nodes: int) -> tuple[dict[tuple[str, tuple], Expr], tuple[Expr, ...]]:
    best_for_sig, exprs, _ = _enumerate_core(inputs, max_nodes)
    return best_for_sig, exprs


def canonical_expr(examples: list[tuple[str, str]] | tuple[tuple[str, str], ...], max_nodes: int = 10) -> Expr | None:
    inputs = tuple(inp for inp, _ in examples)
    target = tuple(out for _, out in examples)
    _, _, found = _enumerate_core(inputs, max_nodes, ("str", target))
    return found


def run_expr(expr: Expr | str, s: str):
    if isinstance(expr, Expr):
        return expr.run(s)
    return eval(_compile_expr(expr), {"__builtins__": SAFE_BUILTINS}, {"s": s})


def expr_to_function(expr: Expr | str) -> str:
    py = expr.py if isinstance(expr, Expr) else expr
    return f"def f(s: str) -> str:\n    return {py}"


def _random_input(max_len: int = 12) -> str:
    return "".join(random.choice(RANDOM_ALPHABET) for _ in range(random.randint(0, max_len)))


def _signature_hash(sig: tuple) -> str:
    body = repr(sig).encode("utf-8")
    return hashlib.sha256(body).hexdigest()[:16]


def _op_dict(e: Expr) -> dict[str, int]:
    return {op: n for op, n in zip(OPS, e.ops) if n}


def _prompt_ops(e: Expr, distractor_rate: float) -> list[str]:
    used = {op for op, n in zip(OPS, e.ops) if n}
    chosen = set(used)
    for op in OPS:
        if op not in used and random.random() < distractor_rate:
            chosen.add(op)
    return [op for op in OPS if op in chosen]


def _target_quality(e: Expr, sig: tuple, min_nodes: int) -> bool:
    if e.sort != "str" or not e.depends_on_s or e.nodes < min_nodes:
        return False
    if len(set(sig)) < 2:
        return False
    if e.py in {"s", *STR_LITS}:
        return False
    if re.search(r'"[^"]*"\s*(?:\.replace|\.find|\[)', e.py):
        return False
    if re.search(r'\bs\s+in\s+"[^"]*"', e.py):
        return False
    return True


@lru_cache(maxsize=16)
def _target_frontier(max_nodes: int, min_nodes: int) -> tuple[Expr, ...]:
    frontier_nodes = min(max_nodes, 7)
    best_for_sig, _ = _enumerate(tuple(GLOBAL_PROBES), frontier_nodes)
    out = []
    seen_py = set()
    for (sort, sig), e in best_for_sig.items():
        if sort != "str" or e.py in seen_py:
            continue
        if _target_quality(e, sig, min_nodes):
            out.append(e)
            seen_py.add(e.py)
    out.sort(key=lambda e: e.cost)
    return tuple(out)


def _counterexample(p: Expr, q: Expr, max_len: int) -> str | None:
    candidates = list(ADVERSARIAL_INPUTS) + [_random_input(max_len) for _ in range(128)]
    seen = set()
    for s in candidates:
        if s in seen:
            continue
        seen.add(s)
        try:
            if p.run(s) != q.run(s):
                return s
        except Exception:
            continue
    return None


def _make_examples(target: Expr, max_nodes: int, max_examples: int, max_input_len: int) -> tuple[list[tuple[str, str]], int] | None:
    seed_pool = list(ADVERSARIAL_INPUTS)
    random.shuffle(seed_pool)
    shown_inputs = []
    for s in seed_pool:
        if len(shown_inputs) >= 2:
            break
        try:
            if _valid_value("str", target.run(s)):
                shown_inputs.append(s)
        except Exception:
            pass
    if not shown_inputs:
        return None

    shown = [(s, target.run(s)) for s in shown_inputs]
    killed = 0
    for _ in range(max_examples):
        q = canonical_expr(shown, max_nodes)
        if q is None:
            return None
        if q.py == target.py:
            return shown, killed
        x = _counterexample(target, q, max_input_len)
        if x is None or len(x) > max_input_len or any(x == old for old, _ in shown):
            return None
        shown.append((x, target.run(x)))
        killed += 1
        if len(shown) > max_examples:
            return None
    return None


def _nontrivial_examples(examples: list[tuple[str, str]]) -> bool:
    if len({out for _, out in examples}) < 2:
        return False
    if all(inp == out for inp, out in examples):
        return False
    return True


def _holdout(target: Expr, shown: list[tuple[str, str]], k: int, max_len: int) -> list[tuple[str, str]]:
    shown_inputs = {inp for inp, _ in shown}
    candidates = list(GLOBAL_PROBES) + [_random_input(max_len) for _ in range(128)]
    out = []
    seen = set(shown_inputs)
    for s in candidates:
        if s in seen:
            continue
        seen.add(s)
        try:
            y = target.run(s)
        except Exception:
            continue
        if _valid_value("str", y):
            out.append((s, y))
        if len(out) >= k:
            break
    return out


def _validate_problem(examples: list[tuple[str, str]], holdout: list[tuple[str, str]], target: Expr, max_nodes: int) -> bool:
    expr = canonical_expr(examples, max_nodes)
    if expr is None or expr.py != target.py:
        return False
    if any(expr.run(inp) != out for inp, out in examples):
        return False
    if any(expr.run(inp) != target.run(inp) or target.run(inp) != out for inp, out in holdout):
        return False
    return True


@dataclass
class ProgramSynthesisCfg(Config):
    max_nodes: int = 10
    n_holdout: int = 12
    max_examples: int = 8
    max_input_len: int = 16
    max_attempts: int = 80
    min_nodes: int = 4
    prompt_distractor_rate: float = 0.25

    def apply_difficulty(self, level):
        self.max_nodes = sround(self.max_nodes + 0.5 * level)
        self.n_holdout = sround(self.n_holdout + level)
        self.max_attempts = sround(self.max_attempts + 10 * level)
        self.min_nodes = sround(self.min_nodes + level / 3)


class ProgramSynthesis(Task):
    summary = "Synthesize minimum-cost bounded string-transformation programs from examples across compositional StringFrag operations."

    def __init__(self, config=None):
        super().__init__(config=config or ProgramSynthesisCfg())

    def _sample_target(self) -> Expr:
        frontier = _target_frontier(self.config.max_nodes, self.config.min_nodes)
        if not frontier:
            raise RuntimeError("StringFrag-v1 target frontier is empty")
        by_nodes: dict[int, list[Expr]] = {}
        for e in frontier:
            by_nodes.setdefault(e.nodes, []).append(e)
        sizes = sorted(by_nodes)
        weights = [1 + i for i, _ in enumerate(sizes)]
        size = random.choices(sizes, weights=weights, k=1)[0]
        return random.choice(by_nodes[size])

    def generate_entry(self) -> Entry:
        cfg = self.config
        for _ in range(cfg.max_attempts):
            target = self._sample_target()
            made = _make_examples(target, cfg.max_nodes, cfg.max_examples, cfg.max_input_len)
            if made is None:
                continue
            examples, killed = made
            if not _nontrivial_examples(examples):
                continue
            holdout = _holdout(target, examples, cfg.n_holdout, cfg.max_input_len)
            if len(holdout) < cfg.n_holdout:
                continue
            if not _validate_problem(examples, holdout, target, cfg.max_nodes):
                continue
            probe_sig = target.signature(tuple(GLOBAL_PROBES))
            prompt_ops = _prompt_ops(target, cfg.prompt_distractor_rate)
            meta = edict(
                dsl=DSL_NAME,
                cost=COST_DESCRIPTION,
                max_nodes=cfg.max_nodes,
                io_pairs=examples,
                examples=examples,
                holdout=holdout,
                solution_expr=target.py,
                answer_expr=target.py,
                solution_function=expr_to_function(target),
                nodes=target.nodes,
                ops=_op_dict(target),
                prompt_ops=prompt_ops,
                target_signature_hash=_signature_hash(probe_sig or ()),
                difficulty=edict(
                    num_examples=len(examples),
                    cegis_rounds=max(0, len(examples) - 2),
                    cheaper_hypotheses_killed=killed,
                ),
            )
            return Entry(metadata=meta, answer=meta.solution_function)
        raise RuntimeError(f"no StringFrag-v1 instance after {cfg.max_attempts} attempts")

    def render_prompt(self, metadata) -> str:
        examples = "\n".join(f"f({inp!r}) = {out!r}" for inp, out in metadata.io_pairs)
        op_lines = "\n".join(f"- {op}: {OP_SYNTAX[op]}" for op in metadata.prompt_ops)
        return (
            "Write f(s: str) -> str.\n\n"
            "Target: return the minimum-cost StringFrag-v1 expression matching the examples.\n\n"
            "Always allowed: s, string literals \"\", \" \", \"-\", \"_\", and integer literals 0, 1, 2, 3.\n"
            f"Allowed operators for this problem:\n{op_lines}\n"
            "Bounds: strings have length <= 64; integers are between -16 and 64. Use Python string semantics.\n"
            "Cost: AST nodes, then operator-count tuple in this global order "
            f"({OP_COST_ORDER}), then source length, then lexicographic source order.\n\n"
            f"Examples:\n{examples}\n\n"
            "Return only:\n"
            "def f(s: str) -> str:\n"
            "    return <expression>"
        )

    def score_answer(self, answer, entry) -> float:
        import ast
        from reasoning_core.tasks.code_program_synthesis import (
            SAFE_BUILTINS, _extract_return_expr, _valid_value, _validate_candidate,
        )
        expr = _extract_return_expr(answer)
        if expr is None:
            return 0.0
        try:
            tree = ast.parse(expr, mode="eval")
            target_tree = ast.parse(entry.metadata["solution_expr"], mode="eval")
        except SyntaxError:
            return 0.0
        if ast.dump(tree, include_attributes=False) == ast.dump(target_tree, include_attributes=False):
            return 1.0
        try:
            _, used_ops = _validate_candidate(tree.body)
        except ValueError:
            return 0.0
        valid_dsl = used_ops <= set(entry.metadata["prompt_ops"])

        pairs = list(entry.metadata["io_pairs"]) + list(entry.metadata.get("holdout", ()))
        if not pairs:
            return 0.0
        code = compile(tree, "<stringfrag-v1-answer>", "eval")
        hits = 0
        for inp, expected in pairs:
            try:
                got = eval(code, {"__builtins__": SAFE_BUILTINS}, {"s": inp})
            except Exception:
                continue
            hits += got == expected and _valid_value("str", got)
        return (0.9 if valid_dsl else 0.45) * hits / len(pairs)


def _validate_candidate(node: ast.AST) -> tuple[str, set[str]]:
    """Type-check the public DSL and return its sort and used operators."""
    if isinstance(node, ast.Name) and node.id == "s":
        return "str", set()
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str) and len(node.value) <= MAX_STR_LEN:
            marker = set() if node.value in {"", " ", "-", "_"} else {"literal_outside"}
            return "str", marker
        if type(node.value) is int and MIN_INT <= node.value <= MAX_INT:
            marker = set() if node.value in {0, 1, 2, 3} else {"literal_outside"}
            return "int", marker
        raise ValueError("literal outside StringFrag-v1")

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left_sort, left_ops = _validate_candidate(node.left)
        right_sort, right_ops = _validate_candidate(node.right)
        if isinstance(node.op, ast.Add) and left_sort == right_sort == "str":
            return "str", left_ops | right_ops | {"concat"}
        if left_sort == right_sort == "int":
            op = "add" if isinstance(node.op, ast.Add) else "sub"
            return "int", left_ops | right_ops | {op}
        raise ValueError("ill-typed binary operation")

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "len" and len(node.args) == 1:
            sort, ops = _validate_candidate(node.args[0])
            if sort == "str" and not node.keywords:
                return "int", ops | {"len"}
        if isinstance(node.func, ast.Attribute):
            base_sort, ops = _validate_candidate(node.func.value)
            args = [_validate_candidate(arg) for arg in node.args]
            if base_sort == "str" and not node.keywords:
                if node.func.attr == "find" and len(args) == 1 and args[0][0] == "str":
                    return "int", ops | args[0][1] | {"find"}
                if (node.func.attr == "replace" and len(args) == 3
                        and [sort for sort, _ in args] == ["str", "str", "int"]
                        and isinstance(node.args[2], ast.Constant) and node.args[2].value == 1):
                    return "str", ops | set().union(*(item[1] for item in args)) | {"replace1"}
        raise ValueError("call outside StringFrag-v1")

    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        left_sort, left_ops = _validate_candidate(node.left)
        right_sort, right_ops = _validate_candidate(node.comparators[0])
        op = node.ops[0]
        if isinstance(op, ast.In) and left_sort == right_sort == "str":
            return "bool", left_ops | right_ops | {"contains"}
        if isinstance(op, ast.Eq) and left_sort == right_sort == "str":
            return "bool", left_ops | right_ops | {"eq_str"}
        if isinstance(op, ast.Lt) and left_sort == right_sort == "int":
            return "bool", left_ops | right_ops | {"lt"}
        raise ValueError("comparison outside StringFrag-v1")

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        sort, ops = _validate_candidate(node.operand)
        if sort == "bool":
            return "bool", ops | {"not"}
        raise ValueError("ill-typed not")

    if isinstance(node, ast.IfExp):
        test_sort, test_ops = _validate_candidate(node.test)
        body_sort, body_ops = _validate_candidate(node.body)
        else_sort, else_ops = _validate_candidate(node.orelse)
        if test_sort == "bool" and body_sort == else_sort == "str":
            return "str", test_ops | body_ops | else_ops | {"ite"}
        raise ValueError("ill-typed conditional")

    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
        value_sort, value_ops = _validate_candidate(node.value)
        sl = node.slice
        if sl.lower is None or sl.upper is None or sl.step is not None:
            raise ValueError("slice outside StringFrag-v1")
        start_sort, start_ops = _validate_candidate(sl.lower)
        lower = ast.dump(sl.lower, include_attributes=False)
        if isinstance(sl.upper, ast.BinOp) and isinstance(sl.upper.op, ast.Add):
            upper_start = ast.dump(sl.upper.left, include_attributes=False)
            length_sort, length_ops = _validate_candidate(sl.upper.right)
            if value_sort == "str" and start_sort == length_sort == "int" and upper_start == lower:
                return "str", value_ops | start_ops | length_ops | {"substr"}
        upper_sort, upper_ops = _validate_candidate(sl.upper)
        if value_sort == "str" and start_sort == upper_sort == "int":
            return "str", value_ops | start_ops | upper_ops | {"substr", "slice_syntax_outside"}
        raise ValueError("ill-typed slice")

    raise ValueError("syntax outside StringFrag-v1")


def _extract_return_expr(answer: str) -> str | None:
    candidate = answer.strip()
    if candidate == "":
        return None
    match = re.search(r"^\s*return\s+(.+?)\s*$", candidate, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()
    try:
        mod = ast.parse(candidate)
    except SyntaxError:
        return candidate
    if len(mod.body) == 1 and isinstance(mod.body[0], ast.FunctionDef):
        fn = mod.body[0]
        if len(fn.body) == 1 and isinstance(fn.body[0], ast.Return):
            return ast.get_source_segment(candidate, fn.body[0].value).strip()
    match = re.search(r"return\s+(.+?)\s*$", candidate, re.DOTALL)
    if match:
        return match.group(1).strip()
    return candidate
