"""
program_composition_task.py
───────────────────────────
Four Task subclasses built on the ProgramGenerator engine,
fully integrated with reasoning_core/template.py conventions.

Tasks
─────
  TypePrediction   — Given code, predict the return type of the outermost call.
  NodeCompletion   — Given code with one node masked, pick the correct function
                     from execution-validated candidates.
  ExecutionTracing — Given code, produce the depth-first call sequence.
  OutputPrediction — Given fully-literal code, predict the printed output.

NodeCompletion difficulty tiers (automatic)
───────────────────────────────────────────
  execution  — sandbox runs all candidates; correct one succeeds, ≥1 distractor
               fails or produces different output. Strongest signal.
  semantic   — registry constraint discriminates; correct answer is VALID,
               ≥1 distractor is INVALID. Medium signal.
  type-only  — fallback; all candidates share return type, differ by module.

Fixes in this version
─────────────────────
  1. Axis constraint contradiction — semantic tasks now only emit when the
     correct answer itself is VALID under the constraint. Previously a masked
     axis-sensitive function with an OOB axis would mark itself as invalid.

  2. Degenerate example filtering —
     a) Programs where the entire body is just `result = <MASKED>` are rejected
        (no surrounding code = no reasoning context).
     b) Masks whose return type is `Any`, `module`, or `NoneType` are rejected
        (these types give the model no real constraint to reason from).
     c) Dummy variant names (e.g. `func_variant_1`) are suppressed; when the
        distractor pool is too small we retry instead of padding with fakes.

  3. Evaluation dataset script — see generate_eval_set() at the bottom.
     Produces a fixed JSON file of N examples per difficulty tier, suitable
     for reproducible LLM evaluation.

Sandbox safety
──────────────
  Each candidate substitution runs in a fresh subprocess (SIGKILL timeout).
  Imports are checked against IMPORT_BLACKLIST before any subprocess is spawned.
"""

import io
import ast
import sys
import json
import random
import sqlite3
import string
import contextlib
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Callable, Any

from reasoning_core.template import Task, Problem, Config
from reasoning_core.utils import score_scalar


MAX_LINE = 100
INDENT   = "    "

# Return types that give the model no real constraint — skip masking these.
WEAK_RETURN_TYPES = {"Any", "module", "NoneType", "Optional", "Decorator", "Union"}


# Modules rejected from sandbox execution regardless of generated code.
IMPORT_BLACKLIST = {
    "os", "sys", "subprocess", "socket", "urllib", "http",
    "pathlib", "shutil", "glob", "tempfile", "atexit",
    "multiprocessing", "threading", "ctypes", "cffi",
}


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProgramCompositionCfg(Config):
    """
    max_depth        — maximum nesting depth of function calls.
    db_path          — path to the typed functions SQLite DB.
    target_type      — if set, root must return this type. None = random.
    literal_ratio    — probability a primitive leaf becomes a real literal.
    exec_timeout     — per-candidate subprocess wall-clock timeout (seconds).
    n_candidates     — total candidates shown (1 correct + rest distractors).
    max_gen_tries    — retry budget for the generation loop.
    prefer_semantic  — prefer masking nodes in the semantic registry.
    max_depth_node   — depth cap for NodeCompletion (keeps programs readable).
    use_sandbox      — set False to skip execution (faster, weaker difficulty).
    min_pool_size    — minimum distractor pool size; retry if pool too small
                       (prevents dummy variant padding).
    """
    max_depth       : int   = 3
    db_path         : str   = "functions.db"
    target_type     : str   = None
    literal_ratio   : float = 1.0
    exec_timeout    : int   = 4
    n_candidates    : int   = 4
    max_gen_tries   : int   = 500
    prefer_semantic : bool  = True
    max_depth_node  : int   = 2
    use_sandbox     : bool  = True
    min_pool_size   : int   = 3   # require at least this many real distractors

    def update(self, delta: int):
        self.max_depth      += delta
        self.max_depth_node += delta


# ═══════════════════════════════════════════════════════════════════════════════
# CORE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

PRIMITIVE_TYPES = {"int", "float", "str", "bool", "bytes"}


class Node:
    """Base class — eliminates scattered isinstance checks."""
    @property
    def return_type(self) -> str:
        raise NotImplementedError

    @property
    def state(self) -> dict:
        return {}


class FunctionRecord:
    __slots__ = ("id", "library", "module", "name", "inputs", "output")

    def __init__(self, id_, library, module, name, inputs, output):
        self.id      = id_
        self.library = library
        self.module  = module
        self.name    = name
        self.inputs  = inputs    # list of (param_name, type_str)
        self.output  = output

    @property
    def is_leaf(self):
        return len(self.inputs) == 0

    def import_path(self):
        return f"from {self.module} import {self.name}"

    def signature(self):
        params = ", ".join(f"{n}: {t}" for n, t in self.inputs)
        return f"{self.name}({params}) -> {self.output}"


class ExpressionNode(Node):
    __slots__ = ("func", "args", "_state")

    def __init__(self, func, args):
        self.func   = func
        self.args   = args
        self._state = {}

    @property
    def return_type(self):
        return self.func.output

    @property
    def state(self):
        return self._state


class LeafNode(Node):
    __slots__ = ("type_str", "value", "is_literal", "_state")

    def __init__(self, type_str, value, is_literal):
        self.type_str   = type_str
        self.value      = value
        self.is_literal = is_literal
        self._state     = {}

    @property
    def return_type(self):
        return self.type_str

    @property
    def state(self):
        return self._state


class MaskedNode(Node):
    __slots__ = ("original",)

    def __init__(self, original: ExpressionNode):
        self.original = original

    @property
    def return_type(self):
        return self.original.return_type

    @property
    def state(self):
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE INDEX
# ═══════════════════════════════════════════════════════════════════════════════

class TypeIndex:
    def __init__(self, db_path: str):
        self._by_output : dict[str, list[FunctionRecord]] = defaultdict(list)
        self._all       : list[FunctionRecord] = []
        self._load(db_path)

    @staticmethod
    def _parse_inputs(s: str):
        if not s or not s.strip():
            return []
        result = []
        for part in s.split(","):
            part = part.strip()
            if ":" in part:
                name, _, typ = part.partition(":")
                result.append((name.strip(), typ.strip()))
            else:
                result.append((f"arg{len(result)}", part.strip()))
        return result

    def _load(self, db_path: str):
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        cur.execute(
            "SELECT id, library, module, function_name, inputs, outputs "
            "FROM functions"
        )
        for row in cur.fetchall():
            id_, lib, mod, name, inp_str, out = row
            out = (out or "").strip()
            if not out:
                continue
            inputs = self._parse_inputs(inp_str or "")
            rec = FunctionRecord(id_, lib, mod, name, inputs, out)
            self._by_output[out].append(rec)
            self._all.append(rec)
        conn.close()

    def functions_returning(self, t: str) -> list[FunctionRecord]:
        return self._by_output.get(t, [])

    def random_function(self, rng: random.Random) -> FunctionRecord:
        return rng.choice(self._all)

    def unique_functions_returning(self, t: str) -> list[FunctionRecord]:
        """One entry per unique function name."""
        seen, result = set(), []
        for f in self._by_output.get(t, []):
            if f.name not in seen:
                seen.add(f.name)
                result.append(f)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# LITERAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _random_literal(type_str: str, rng: random.Random) -> Optional[str]:
    t = type_str.lower()
    if t == "int":
        return str(rng.randint(-100, 100))
    if t == "float":
        return repr(round(rng.uniform(-100.0, 100.0), 4))
    if t == "str":
        k = rng.randint(3, 8)
        return repr("".join(rng.choices(string.ascii_lowercase, k=k)))
    if t == "bool":
        return rng.choice(["True", "False"])
    if t == "bytes":
        n = rng.randint(1, 4)
        return repr(bytes(rng.randint(0, 255) for _ in range(n)))
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRAM GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ProgramGenerator:
    def __init__(self, index: TypeIndex, cfg: ProgramCompositionCfg,
                 rng: random.Random):
        self.index = index
        self.cfg   = cfg
        self.rng   = rng
        self._const_counter: dict[str, int] = defaultdict(int)

    def reset_counters(self):
        self._const_counter.clear()

    def _fresh_const(self, type_str: str) -> str:
        safe = (type_str
                .replace(" ", "_").replace("|", "Or").replace("[", "_")
                .replace("]", "_").replace(".", "_").replace(",", "")
                .replace("(", "").replace(")", ""))[:40]
        idx = self._const_counter[safe]
        self._const_counter[safe] += 1
        return f"CONST_{safe}_{idx}"

    def _make_leaf(self, type_str: str) -> LeafNode:
        if type_str in PRIMITIVE_TYPES:
            if self.rng.random() < self.cfg.literal_ratio:
                lit = _random_literal(type_str, self.rng)
                if lit is not None:
                    node = LeafNode(type_str, lit, True)
                    if type_str == "int":
                        node._state["value"] = int(lit)
                    return node
        node = LeafNode(type_str, self._fresh_const(type_str), False)
        if type_str in ("Any", "Tensor", "ndarray"):
            rank = self.rng.randint(1, 4)
            node._state["shape"] = tuple(self.rng.randint(2, 8) for _ in range(rank))
        return node

    def _build(self, required_type: str, depth: int) -> Node:
        pool = self.index.functions_returning(required_type)
        if depth == 0 or not pool:
            return self._make_leaf(required_type)
        func = self.rng.choice(pool)
        if func.is_leaf:
            return ExpressionNode(func, [])
        args = [self._build(pt, depth - 1) for _, pt in func.inputs]
        node = ExpressionNode(func, args)
        for a in args:
            if "shape" in a.state:
                node._state["shape"] = a.state["shape"]
                break
        for a in args:
            if "value" in a.state:
                node._state["value"] = a.state["value"]
                break
        return node

    def build_tree(self, target_type: Optional[str] = None,
                   max_depth: Optional[int] = None) -> ExpressionNode:
        self.reset_counters()
        depth = max_depth if max_depth is not None else self.cfg.max_depth
        t = target_type or self.cfg.target_type
        if t:
            pool = self.index.functions_returning(t)
            if not pool:
                raise ValueError(f"No functions returning '{t}'")
            root_func = self.rng.choice(pool)
        else:
            root_func = self.index.random_function(self.rng)
        if root_func.is_leaf:
            return ExpressionNode(root_func, [])
        args = [self._build(pt, depth - 1) for _, pt in root_func.inputs]
        return ExpressionNode(root_func, args)


# ═══════════════════════════════════════════════════════════════════════════════
# TREE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_placeholders(node: Node) -> list:
    if isinstance(node, LeafNode):
        return [node] if not node.is_literal else []
    if isinstance(node, MaskedNode):
        return []
    return [p for a in node.args for p in _collect_placeholders(a)]


def _collect_call_nodes(node: Node, acc: list):
    if isinstance(node, ExpressionNode):
        acc.append(node)
        for a in node.args:
            _collect_call_nodes(a, acc)


def _tree_to_dict(node: Node) -> dict:
    if isinstance(node, LeafNode):
        return {"kind": "leaf", "type": node.type_str,
                "value": node.value, "is_literal": node.is_literal}
    if isinstance(node, MaskedNode):
        return {"kind": "masked", "type": node.return_type}
    return {
        "kind": "call",
        "function": node.func.name,
        "module": node.func.module,
        "return_type": node.func.output,
        "args": [
            {
                "param": node.func.inputs[i][0] if i < len(node.func.inputs) else f"arg{i}",
                "expected_type": node.func.inputs[i][1] if i < len(node.func.inputs) else "?",
                "node": _tree_to_dict(a),
            }
            for i, a in enumerate(node.args)
        ],
    }


def _emit_expr(node: Node, imports: list, level: int = 0) -> str:
    if isinstance(node, LeafNode):
        return node.value
    if isinstance(node, MaskedNode):
        return "<MASKED>"
    imports.append(node.func.import_path())
    arg_strs = []
    for i, a in enumerate(node.args):
        expr  = _emit_expr(a, imports, level + 1)
        pname = node.func.inputs[i][0] if i < len(node.func.inputs) else None
        arg_strs.append(f"{pname}={expr}" if pname else expr)
    inline = f"{node.func.name}({', '.join(arg_strs)})"
    if len(inline) <= MAX_LINE and "\n" not in inline:
        return inline
    inner = INDENT * (level + 1)
    body  = ",\n".join(f"{inner}{a}" for a in arg_strs)
    return f"{node.func.name}(\n{body}\n{INDENT * level})"


# def _emit_code(root: Node, substitute_masked: Optional[str] = None
#                ) -> tuple[str, list[str]]:
#     imports: list[str] = []
#     consts  = _collect_placeholders(root)
#     expr    = _emit_expr(root, imports)

#     if "np." in expr:
#         unique_imports.insert(0, "import numpy as np")

#     if substitute_masked is not None:
#         expr = expr.replace("<MASKED>", substitute_masked, 1)
#     seen, unique_imports = set(), []
#     for imp in imports:
#         if imp not in seen:
#             seen.add(imp)
#             unique_imports.append(imp)
#     lines = []
#     if unique_imports:
#         lines += unique_imports
#         lines.append("")
#     if consts:
#         lines.append("# ── placeholder constants ──")
#         seen_c: set[str] = set()
#         for leaf in consts:
#             if leaf.value not in seen_c:
#                 lines.append(f"{leaf.value} = None  # type: {leaf.type_str}")
#                 seen_c.add(leaf.value)
#         lines.append("")
#     lines.append(f"result = {expr}")
#     lines.append("print(result)")
#     return "\n".join(lines), unique_imports

def _emit_code(root: Node, substitute_masked: Optional[str] = None
               ) -> tuple[str, list[str]]:
    imports: list[str] = []
    consts  = _collect_placeholders(root)
    expr    = _emit_expr(root, imports)

    if substitute_masked is not None:
        expr = expr.replace("<MASKED>", substitute_masked, 1)

    # build unique imports first
    seen, unique_imports = set(), []
    for imp in imports:
        if imp not in seen:
            seen.add(imp)
            unique_imports.append(imp)

    if "np." in expr and "import numpy as np" not in unique_imports:
        unique_imports.insert(0, "import numpy as np")

    if "jnp." in expr and "import jax.numpy as jnp" not in unique_imports:
        unique_imports.insert(0, "import jax.numpy as jnp")
    if "torch." in expr and "import torch" not in unique_imports:
        unique_imports.insert(0, "import torch")
    if "PIL.Image.new" in expr and "import PIL.Image" not in unique_imports:
        unique_imports.insert(0, "import PIL.Image")

    lines = []
    if unique_imports:
        lines += unique_imports
        lines.append("")

    if consts:
        lines.append("# ── placeholder constants ──")
        seen_c: set[str] = set()
        for leaf in consts:
            if leaf.value not in seen_c:
                lines.append(f"{leaf.value} = None  # type: {leaf.type_str}")
                seen_c.add(leaf.value)
        lines.append("")

    lines.append(f"result = {expr}")
    lines.append("print(result)")

    return "\n".join(lines), unique_imports


def _dfs_call_sequence(node: Node) -> list[str]:
    if isinstance(node, (LeafNode, MaskedNode)):
        return []
    seq = []
    for a in node.args:
        seq.extend(_dfs_call_sequence(a))
    seq.append(node.func.name)
    return seq


def _has_placeholder(node: Node) -> bool:
    if isinstance(node, LeafNode):
        return not node.is_literal
    if isinstance(node, MaskedNode):
        return True
    return any(_has_placeholder(a) for a in node.args)


def _mask_node(root: Node, target: ExpressionNode) -> Node:
    if root is target:
        return MaskedNode(target)
    if isinstance(root, (LeafNode, MaskedNode)):
        return root
    return ExpressionNode(root.func, [_mask_node(a, target) for a in root.args])


def _is_degenerate_masked(masked_root: Node) -> bool:
    """
    True when the masked program body is just `result = <MASKED>` with no
    surrounding calls — the model has nothing to reason about.
    """
    return isinstance(masked_root, MaskedNode)

def evaluate_candidates(
    candidates: list[FunctionRecord],
    target_type: str,
    build_program,      # fn(candidate) -> full code string
    sandbox_run         # fn(code) -> SandboxResult
):
    # 1. Filter by return type
    valid = [c for c in candidates if c.output == target_type]

    if not valid:
        return None, "no-solution"

    # 2. If only one → type-only
    if len(valid) == 1:
        return valid[0], "type-only"

    # 3. MULTIPLE VALID → FORCE EXECUTION (THIS IS YOUR MISSING LOGIC)
    results = []
    for c in valid:
        try:
            code = build_program(c)
            res = sandbox_run(code)
            results.append((c, res))
        except Exception as e:
            results.append((c, None))

    # 4. Keep only successful executions
    ok = [(c, r) for (c, r) in results if r is not None and r.ok]

    # 5. If exactly one executes successfully → execution case
    if len(ok) == 1:
        return ok[0][0], "execution"

    # 6. If multiple execute → still ambiguous
    if len(ok) > 1:
        # optionally compare outputs here
        return ok[0][0], "execution-ambiguous"

    # 7. If none execute → fallback
    return valid[0], "type-only"


def _dummy_value(type_str: str) -> str:
    t = type_str.lower().strip()
    if t == "int": return "1"
    if t == "float": return "1.0"
    if t == "str": return "'x'"
    if t == "bool": return "True"
    if t == "bytes": return "b'\\x00'"
    if "array" in t: return "jnp.ones((2,))"  # jax
    if "tensor" in t: return "torch.ones(2)"
    if "ndarray" in t: return "np.ones((2,))"
    if "image" in t: return "PIL.Image.new('RGB', (4,4))"
    # For Union/Any/Optional — use a simple int, it might work
    return "1"

# ═══════════════════════════════════════════════════════════════════════════════
# SAFE SANDBOX EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

_WORKER_SCRIPT = r"""
import sys, json, io, contextlib

code = sys.stdin.read()
buf  = io.StringIO()
ns   = {}
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<sandbox>", "exec"), ns)
    out = buf.getvalue().strip()
    print(json.dumps({"ok": True, "output": out}))
except Exception as e:
    print(json.dumps({"ok": False, "error": type(e).__name__}))
"""


@dataclass
class SandboxResult:
    ok     : bool
    output : str = ""
    error  : str = ""


def _check_imports_safe(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in IMPORT_BLACKLIST:
                    return False
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in IMPORT_BLACKLIST:
                return False
    return True


def _run_in_sandbox(code: str, timeout: int) -> SandboxResult:
    if not _check_imports_safe(code):
        return SandboxResult(ok=False, error="BlacklistedImport")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _WORKER_SCRIPT],
            input=code, capture_output=True, text=True, timeout=timeout,
        )
        raw = proc.stdout.strip()
        if not raw:
            return SandboxResult(ok=False, error="NoOutput")
        data = json.loads(raw)
        if data.get("ok"):
            return SandboxResult(ok=True, output=data.get("output", ""))
        return SandboxResult(ok=False, error=data.get("error", "UnknownError"))
    except subprocess.TimeoutExpired:
        return SandboxResult(ok=False, error="Timeout")
    except Exception as e:
        return SandboxResult(ok=False, error=type(e).__name__)


@dataclass
class CandidateVerdict:
    name   : str
    result : SandboxResult

def _dummy_value(type_str: str) -> str:
    t = type_str.lower()

    if t == "int":
        return "1"
    if t == "float":
        return "1.0"
    if t == "str":
        return "'x'"
    if t == "bool":
        return "True"
    if t == "bytes":
        return "b'0'"
    if t in ("ndarray", "tensor"):
        return "np.zeros((2, 2))"

    return "None"


def _build_candidate_call(func: FunctionRecord) -> str:
    if not func.inputs:
        return f"{func.name}()"

    args = ", ".join(
        f"{name}={_dummy_value(type_str)}"
        for name, type_str in func.inputs
    )
    return f"{func.name}({args})"


def _run_candidates_in_sandbox(
    masked_root : Node,
    candidates  : list[FunctionRecord],
    timeout     : int,
    rng         : random.Random,
) -> list[CandidateVerdict]:
    verdicts = []
    for func in candidates:
        call_expr = _build_candidate_call(func)
        code, _   = _emit_code(masked_root, substitute_masked=call_expr)
        candidate_import = func.import_path()
        if candidate_import not in code:
            code = candidate_import + "\n" + code
        result = _run_in_sandbox(code, timeout)
        verdicts.append(CandidateVerdict(name=func.name, result=result))
    return verdicts


def _classify_by_execution(
    correct_name : str,
    verdicts     : list[CandidateVerdict],
) -> tuple[bool, str]:
    """
    Returns (is_discriminated, human_readable_description).

    Discrimination requires:
      - correct candidate ran successfully, AND
      - at least one distractor crashed OR produced different output.
    """
    correct_v    = next((v for v in verdicts if v.name == correct_name), None)
    distractor_v = [v for v in verdicts if v.name != correct_name]
    if correct_v is None or not correct_v.result.ok:
        return False, ""
    correct_out  = correct_v.result.output
    discriminated = any(
        not v.result.ok or v.result.output != correct_out
        for v in distractor_v
    )
    if not discriminated:
        return False, ""
    parts = []
    for v in verdicts:
        if v.name == correct_name:
            parts.append(
                f"`{v.name}`: runs successfully → output `{correct_out or '(no output)'}`"
            )
        elif not v.result.ok:
            parts.append(f"`{v.name}`: raises `{v.result.error}` → invalid")
        elif v.result.output != correct_out:
            parts.append(f"`{v.name}`: produces `{v.result.output}` ≠ expected → wrong")
        else:
            parts.append(f"`{v.name}`: runs but is the wrong function")
    return True, "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC CONSTRAINT SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SemanticContext:
    tensor_shapes : dict = field(default_factory=dict)
    axis_values   : dict = field(default_factory=dict)
    range_values  : dict = field(default_factory=dict)
    extra         : dict = field(default_factory=dict)

    def axis_valid_for(self, tensor_param: str, axis_param: str) -> bool:
        shape = self.tensor_shapes.get(tensor_param)
        axis  = self.axis_values.get(axis_param)
        if shape is None or axis is None:
            return True
        return -len(shape) <= axis < len(shape)

    def shapes_compatible(self) -> bool:
        shapes = list(self.tensor_shapes.values())
        return len(shapes) < 2 or all(s == shapes[0] for s in shapes)

    def range_valid_for(self, val_param: str, n_param: str) -> bool:
        v = self.range_values.get(val_param, 0)
        n = self.range_values.get(n_param, 1)
        return 0 <= v < n

    def has_concrete_info(self) -> bool:
        return bool(self.tensor_shapes or self.axis_values or self.range_values)


@dataclass
class SemanticSpec:
    build_context   : Callable    # (node, rng) -> SemanticContext
    is_valid        : Callable    # (func, ctx) -> bool
    describe        : Callable    # (ctx) -> str
    correct_is_valid: Callable    # (ctx) -> bool  ← NEW: checks if correct fn is valid


def _sample_shape(rng: random.Random, rank: int = None) -> tuple:
    r = rank if rank is not None else rng.randint(1, 4)
    return tuple(rng.randint(2, 8) for _ in range(r))


# ── Axis ───────────────────────────────────────────────────────────────────────
#
# FIX: correct_is_valid for axis checks whether the axis is IN RANGE
# (because the correct function needs a valid axis to work).
# Previously a masked xp_take_along_axis with OOB axis would be marked invalid
# by its own spec, making the CoT contradict itself.
#
# The semantic task is now only emitted when the correct answer IS valid under
# the constraint, and at least one distractor is NOT valid.

_AXIS_SENSITIVE = {"xp_take_along_axis", "xp_moveaxis_to_end", "xp_vector_norm"}

def _axis_build_real(node: ExpressionNode) -> SemanticContext:
    ctx = SemanticContext()
    tensor_param = axis_param = None
    for pname, ptype in node.func.inputs:
        if ptype in ("Any", "ndarray", "Tensor") and tensor_param is None:
            tensor_param = pname
        elif ptype == "int" and axis_param is None:
            axis_param = pname
    if tensor_param is None or axis_param is None:
        return ctx
    for arg, (pname, _) in zip(node.args, node.func.inputs):
        if pname == tensor_param and "shape" in arg.state:
            ctx.tensor_shapes[tensor_param] = arg.state["shape"]
        if pname == axis_param and "value" in arg.state:
            ctx.axis_values[axis_param] = arg.state["value"]
    return ctx

def _axis_is_valid(func: FunctionRecord, ctx: SemanticContext) -> bool:
    """Axis-sensitive functions fail when axis is out of range."""
    if func.name not in _AXIS_SENSITIVE:
        return True    # non-axis-sensitive functions always valid
    if not ctx.tensor_shapes or not ctx.axis_values:
        return True
    return ctx.axis_valid_for(
        next(iter(ctx.tensor_shapes)),
        next(iter(ctx.axis_values))
    )

def _axis_correct_is_valid(ctx: SemanticContext) -> bool:
    """
    The correct answer (an axis-sensitive function) is only valid when
    the axis is within range. If axis is OOB, we cannot emit a semantic
    example — the correct answer would be marked invalid by its own constraint.
    """
    if not ctx.tensor_shapes or not ctx.axis_values:
        return True
    return ctx.axis_valid_for(
        next(iter(ctx.tensor_shapes)),
        next(iter(ctx.axis_values))
    )

def _axis_describe(ctx: SemanticContext) -> str:
    if not ctx.tensor_shapes or not ctx.axis_values:
        return ""
    tname = next(iter(ctx.tensor_shapes))
    pname = next(iter(ctx.axis_values))
    shape = ctx.tensor_shapes[tname]
    rank  = len(shape)
    axis  = ctx.axis_values[pname]
    valid = ctx.axis_valid_for(tname, pname)
    return (
        f"Tensor shape={shape} (rank={rank}). "
        f"axis={axis} is {'valid' if valid else 'OUT OF RANGE'} "
        f"(valid range: [{-rank}, {rank - 1}])."
    )


# ── Shape ──────────────────────────────────────────────────────────────────────

_SHAPE_SENSITIVE = {
    "xp_copysign",
    "_cumulative_simpson_equal_intervals",
    "_cumulative_simpson_unequal_intervals",
    "_cumulatively_sum_simpson_integrals",
}

def _shape_build(node: ExpressionNode, rng: random.Random) -> SemanticContext:
    ctx = SemanticContext()
    base  = _sample_shape(rng)
    count = 0
    for pname, ptype in node.func.inputs:
        if ptype in ("Any", "ndarray", "Tensor"):
            ctx.tensor_shapes[pname] = base if (count == 0 or rng.random() < 0.5) \
                                       else _sample_shape(rng)
            count += 1
    return ctx

def _shape_is_valid(func: FunctionRecord, ctx: SemanticContext) -> bool:
    if func.name not in _SHAPE_SENSITIVE:
        return True
    return ctx.shapes_compatible()

def _shape_correct_is_valid(ctx: SemanticContext) -> bool:
    return ctx.shapes_compatible()

def _shape_describe(ctx: SemanticContext) -> str:
    if len(ctx.tensor_shapes) < 2:
        return ""
    parts = ", ".join(f"`{n}`: {s}" for n, s in ctx.tensor_shapes.items())
    match = ctx.shapes_compatible()
    return f"Shapes: {parts}. {'Compatible ✓' if match else 'Incompatible ✗'}"


# ── Range ──────────────────────────────────────────────────────────────────────

_RANGE_SENSITIVE = {
    "_get_data_description_by_id",
    "_get_data_features",
    "_get_data_qualities",
    "_get_num_samples",
}

def _range_build(node: ExpressionNode, rng: random.Random) -> SemanticContext:
    ctx = SemanticContext()
    int_params = [p for p, t in node.func.inputs if t == "int"]
    if int_params:
        n = rng.randint(2, 10)
        ctx.range_values[int_params[0]] = rng.randint(0, n - 1)  # always valid
        ctx.range_values["_N"] = n
    return ctx

def _range_is_valid(func: FunctionRecord, ctx: SemanticContext) -> bool:
    if func.name not in _RANGE_SENSITIVE:
        return True
    keys = list(ctx.range_values)
    return len(keys) < 2 or ctx.range_valid_for(keys[0], keys[1])

def _range_correct_is_valid(ctx: SemanticContext) -> bool:
    keys = list(ctx.range_values)
    return len(keys) < 2 or ctx.range_valid_for(keys[0], keys[1])

def _range_describe(ctx: SemanticContext) -> str:
    keys = list(ctx.range_values)
    if len(keys) < 2:
        return ""
    v, n  = ctx.range_values[keys[0]], ctx.range_values[keys[1]]
    valid = 0 <= v < n
    return (
        f"Integer argument value={v}, N={n}. "
        f"{'Valid' if valid else 'OUT OF RANGE'} (need 0 ≤ value < N={n})."
    )


# ── Registry ───────────────────────────────────────────────────────────────────

_AXIS_SPEC  = SemanticSpec(_axis_build_real, _axis_is_valid,  _axis_describe,  _axis_correct_is_valid)
_SHAPE_SPEC = SemanticSpec(_shape_build,     _shape_is_valid, _shape_describe, _shape_correct_is_valid)
_RANGE_SPEC = SemanticSpec(_range_build,     _range_is_valid, _range_describe, _range_correct_is_valid)

SEMANTIC_REGISTRY: dict[str, SemanticSpec] = {
    "xp_take_along_axis"                    : _AXIS_SPEC,
    "xp_moveaxis_to_end"                    : _AXIS_SPEC,
    "xp_vector_norm"                        : _AXIS_SPEC,
    "xp_copysign"                           : _SHAPE_SPEC,
    "_cumulative_simpson_equal_intervals"   : _SHAPE_SPEC,
    "_cumulative_simpson_unequal_intervals" : _SHAPE_SPEC,
    "_cumulatively_sum_simpson_integrals"   : _SHAPE_SPEC,
    "_get_data_description_by_id"           : _RANGE_SPEC,
    "_get_data_features"                    : _RANGE_SPEC,
    "_get_data_qualities"                   : _RANGE_SPEC,
    "_get_num_samples"                      : _RANGE_SPEC,
}


# ── Distractor pool ────────────────────────────────────────────────────────────

def _build_distractor_pool(
    correct : FunctionRecord,
    index   : TypeIndex,
    ctx     : Optional[SemanticContext],
    spec    : Optional[SemanticSpec],
    n_total : int,
    rng     : random.Random,
) -> Optional[list[FunctionRecord]]:
    """
    Returns (n_total - 1) distractors, or None if the pool is too small.
    None signals the caller to retry with a different tree.
    No dummy variant padding — we'd rather retry than emit fake function names.
    """
    pool   = index.unique_functions_returning(correct.output)
    pool   = [f for f in pool if f.name != correct.name]
    needed = n_total - 1

    if len(pool) < needed:
        return None    # pool too small — retry

    if spec and ctx and ctx.has_concrete_info():
        invalid = [f for f in pool if not spec.is_valid(f, ctx)]
        valid   = [f for f in pool if spec.is_valid(f, ctx)]
        distractors: list[FunctionRecord] = []
        if invalid:
            take = min(needed, len(invalid))
            distractors += rng.sample(invalid, take)
            needed -= take
        if needed > 0 and valid:
            take = min(needed, len(valid))
            distractors += rng.sample(valid, take)
            needed -= take
    else:
        distractors = rng.sample(pool, min(needed, len(pool)))
        needed -= len(distractors)

    if needed > 0:
        return None    # couldn't fill — retry

    return distractors


# ── CoT builder ────────────────────────────────────────────────────────────────

def _build_cot(
    correct    : FunctionRecord,
    candidates : list[dict],
    spec       : Optional[SemanticSpec],
    ctx        : Optional[SemanticContext],
    exec_desc  : str,
    difficulty : str,
) -> str:
    lines = ["Reasoning step by step:"]
    lines.append(
        f"1. The masked position must return type `{correct.output}`. "
        "All candidates share this return type, so type alone is not enough."
    )
    if difficulty == "execution":
        lines.append("2. Running each candidate in the program:")
        lines.append(exec_desc)
        lines.append(f"3. Therefore the correct answer is `{correct.name}`.")
    elif difficulty == "semantic" and spec and ctx:
        lines.append(f"2. Semantic context: {spec.describe(ctx)}")
        lines.append("3. Evaluating each candidate:")
        for c in candidates:
            f_stub  = FunctionRecord(-1, "", "", c["name"], [], correct.output)
            valid   = spec.is_valid(f_stub, ctx)
            verdict = "✓ satisfies constraint" if valid else "✗ violates constraint"
            lines.append(f"   - `{c['name']}`: {verdict}")
        lines.append(f"4. Therefore the correct answer is `{correct.name}`.")
    else:
        lines.append(
            f"2. No semantic or execution discrimination. "
            f"The correct function is `{correct.name}` "
            f"with signature: `{correct.signature()}`."
        )
        lines.append(
            "3. The other candidates share the return type but are from "
            "different modules or have different semantics."
        )
        lines.append(f"4. Therefore the answer is `{correct.name}`.")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED BASE TASK
# ═══════════════════════════════════════════════════════════════════════════════

class _ProgramBase(Task):
    _index_cache: dict[str, TypeIndex] = {}

    def __init__(self, config=None):
        if config is None:
            config = ProgramCompositionCfg()
        super().__init__(config=config)

    def _get_index(self) -> TypeIndex:
        db = self.config.db_path
        if db not in _ProgramBase._index_cache:
            _ProgramBase._index_cache[db] = TypeIndex(db)
        return _ProgramBase._index_cache[db]

    def _make_generator(self) -> ProgramGenerator:
        rng = random.Random(self.config.seed)
        return ProgramGenerator(self._get_index(), self.config, rng)

    def score_answer(self, answer, entry) -> float:
        norm = lambda x: str(x).strip().lower()
        a, ref = norm(answer), norm(entry.answer)
        if a == ref:
            return 1.0
        try:
            return score_scalar(answer, entry)
        except Exception:
            return 0.0

    def deduplication_key(self, problem):
        return problem.metadata.get("code_hash", None)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — TYPE PREDICTION (UNTOUCHED)
# ═══════════════════════════════════════════════════════════════════════════════

class TypePrediction(_ProgramBase):
    def generate(self) -> Problem:
        gen     = self._make_generator()
        root    = gen.build_tree()
        code, _ = _emit_code(root)
        answer  = root.return_type
        cot = (
            f"Step-by-step type derivation:\n"
            f"1. The outermost call is `{root.func.name}` from `{root.func.module}`.\n"
            f"2. Its declared return type is `{root.func.output}`.\n"
            f"3. Therefore `result` has type `{root.func.output}`."
        )
        meta = {
            "code": code, "tree": _tree_to_dict(root),
            "task_type": "type_prediction", "cot": cot,
            "code_hash": str(hash(code)),
        }
        return Problem(metadata=meta, answer=answer)

    def prompt(self, metadata: dict) -> str:
        return (
            "Consider the following Python program:\n\n"
            f"```python\n{metadata['code']}\n```\n\n"
            "What is the return type of `result`? "
            "Give only the type name (e.g. `str`, `int`, `ndarray`)."
        )

    def score_answer(self, answer, entry) -> float:
        norm = lambda x: str(x).strip().lower().strip("`")
        return float(norm(answer) == norm(entry.answer))


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — NODE COMPLETION (WORK IN PROGRESS)
# ═══════════════════════════════════════════════════════════════════════════════

class NodeCompletion(_ProgramBase):
    """
    Mask one call node. Model picks the correct function from candidates.

    Generation filters (all must pass before emitting an example):
      - masked return type not in WEAK_RETURN_TYPES
      - masked program is not degenerate (body is not just <MASKED>)
      - distractor pool has at least (n_candidates - 1) real functions
      - for semantic tier: correct answer is valid under its own constraint
      - for semantic tier: at least one distractor is invalid
    """

    def generate(self) -> Problem:
        gen   = self._make_generator()
        rng   = random.Random(self.config.seed)
        index = self._get_index()
        calls = []


        for _ in range(self.config.max_gen_tries):
            root  = gen.build_tree(max_depth=self.config.max_depth_node)
            calls : list[ExpressionNode] = []
            _collect_call_nodes(root, calls)
            inner = calls[1:] if len(calls) > 1 else calls
            if len(calls) < 3:   # require at least 3 call nodes for interesting programs
                continue

                # Also reject if the program only has 2 unique function names
            unique_funcs = len({n.func.name for n in calls})
            if unique_funcs < 2:
                continue

            # ── prefer semantic registry nodes ────────────────────────────
            if self.config.prefer_semantic:
                sem = [n for n in inner if n.func.name in SEMANTIC_REGISTRY]
                target = rng.choice(sem) if sem else rng.choice(inner)
            else:
                target = rng.choice(inner)

            # Filter: weak return types give no useful constraint
            if target.func.output in WEAK_RETURN_TYPES:
                continue

            # Filter: degenerate masked program (body is just <MASKED>)
            masked_root = _mask_node(root, target)
            if _is_degenerate_masked(masked_root):
                continue

            # ── semantic context ──────────────────────────────────────────
            spec = SEMANTIC_REGISTRY.get(target.func.name)
            ctx  = None
            if spec:
                if spec is _AXIS_SPEC:
                    ctx = _axis_build_real(target)
                else:
                    ctx = spec.build_context(target, rng)
                if ctx and not ctx.has_concrete_info():
                    ctx = None

            # ── distractor pool ───────────────────────────────────────────
            distractors = _build_distractor_pool(
                correct = target.func,
                index   = index,
                ctx     = ctx,
                spec    = spec,
                n_total = self.config.n_candidates,
                rng     = rng,
            )
            if distractors is None:
                continue    # pool too small — retry

            all_funcs = [target.func] + distractors

            # ── sandbox execution ─────────────────────────────────────────
            difficulty   = "type-only"
            exec_desc    = ""
            context_desc = spec.describe(ctx) if (spec and ctx) else ""

            if self.config.use_sandbox:
                verdicts = _run_candidates_in_sandbox(
                    masked_root = masked_root,
                    candidates  = all_funcs,
                    timeout     = self.config.exec_timeout,
                    rng         = rng,
                )
                is_disc, exec_desc = _classify_by_execution(
                    target.func.name, verdicts
                )
                if is_disc:
                    difficulty = "execution"

            # ── semantic tier (only if correct answer is itself valid) ─────
            if difficulty == "type-only" and spec and ctx and ctx.has_concrete_info():
                # FIX: only emit semantic examples when the correct answer
                # satisfies its own constraint — prevents CoT contradiction.
                correct_valid = spec.correct_is_valid(ctx)
                if correct_valid:
                    any_invalid = any(
                        not spec.is_valid(
                            FunctionRecord(-1,"","",d.name,[],target.func.output), ctx
                        )
                        for d in distractors
                    )
                    if any_invalid:
                        difficulty = "semantic"

            # ── build candidate list (shuffled) ───────────────────────────
            candidates = [{"name": target.func.name, "is_correct": True}] + [
                {"name": d.name, "is_correct": False} for d in distractors
            ]
            rng.shuffle(candidates)

            code, _ = _emit_code(masked_root)
            cot = _build_cot(
                correct    = target.func,
                candidates = candidates,
                spec       = spec,
                ctx        = ctx,
                exec_desc  = exec_desc,
                difficulty = difficulty,
            )

            meta = {
                "code":          code,
                "tree":          _tree_to_dict(masked_root),
                "masked_type":   target.func.output,
                "masked_module": target.func.module,
                "task_type":     "node_completion",
                "difficulty":    difficulty,
                "candidates":    candidates,
                "context_desc":  context_desc,
                "exec_desc":     exec_desc,
                "cot":           cot,
                "code_hash":     str(hash(code)),
            }
            return Problem(metadata=meta, answer=target.func.name)

        raise RuntimeError(
            f"NodeCompletion: could not generate a valid example in "
            f"{self.config.max_gen_tries} attempts."
        )

    def prompt(self, metadata: dict) -> str:
        cand_block = "\n".join(
            f"  {chr(65 + i)}) {c['name']}"
            for i, c in enumerate(metadata["candidates"])
        )
        ctx_section = (
            f"\nContext: {metadata['context_desc']}\n"
            if metadata.get("context_desc") else ""
        )
        exec_section = ""
        if metadata.get("exec_desc"):
            exec_section = (
                "\nExecution hint: running each candidate gives:\n"
                + "\n".join(f"  {l}" for l in metadata["exec_desc"].splitlines())
                + "\n"
            )
        return (
            "The following Python program has one call replaced with `<MASKED>`:\n\n"
            f"```python\n{metadata['code']}\n```\n"
            f"{ctx_section}{exec_section}\n"
            f"The masked expression must return type `{metadata['masked_type']}`.\n\n"
            "Which function correctly fills `<MASKED>`?\n\n"
            f"{cand_block}\n\n"
            "Answer with the letter (A, B, C, ...) or the function name."
        )

    def score_answer(self, answer, entry) -> float:
        ans        = str(answer).strip()
        candidates = entry.metadata.get("candidates", [])
        if len(ans) == 1 and ans.upper() in string.ascii_uppercase:
            idx = ord(ans.upper()) - ord("A")
            if 0 <= idx < len(candidates):
                return float(candidates[idx]["name"] == entry.answer)
        norm = lambda x: x.strip().lower().strip("`")
        return float(norm(ans) == norm(entry.answer))


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3 — EXECUTION TRACING (UNTOUCHED)
# ═══════════════════════════════════════════════════════════════════════════════

class ExecutionTracing(_ProgramBase):
    def generate(self) -> Problem:
        gen = self._make_generator()
        for _ in range(self.config.max_gen_tries):
            root  = gen.build_tree()
            calls : list[ExpressionNode] = []
            _collect_call_nodes(root, calls)
            if len(calls) >= 2:
                break
        code, _ = _emit_code(root)
        seq     = _dfs_call_sequence(root)
        answer  = ", ".join(seq)
        lines   = [
            "Depth-first evaluation order "
            "(arguments evaluated before the call that uses them):",
        ]
        for i, name in enumerate(seq, 1):
            lines.append(f"  {i}. {name}()")
        lines.append(f"Final answer: {', '.join(seq)}")
        meta = {
            "code": code, "tree": _tree_to_dict(root),
            "sequence": seq, "task_type": "execution_tracing",
            "cot": "\n".join(lines), "code_hash": str(hash(code)),
        }
        return Problem(metadata=meta, answer=answer)

    def prompt(self, metadata: dict) -> str:
        return (
            "Consider the following Python program:\n\n"
            f"```python\n{metadata['code']}\n```\n\n"
            "List the function calls in the order they would be evaluated "
            "(innermost / deepest arguments first). "
            "Give a comma-separated list of function names only."
        )

    def score_answer(self, answer, entry) -> float:
        norm  = lambda x: [s.strip().lower() for s in str(x).split(",")]
        a_seq = norm(answer)
        r_seq = norm(entry.answer)
        if a_seq == r_seq:
            return 1.0
        if not r_seq:
            return 0.0
        return sum(a == r for a, r in zip(a_seq, r_seq)) / len(r_seq)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4 — OUTPUT PREDICTION (UNTOUCHED)
# ═══════════════════════════════════════════════════════════════════════════════

class OutputPrediction(_ProgramBase):
    def generate(self) -> Problem:
        gen        = self._make_generator()
        orig_ratio = self.config.literal_ratio
        for _ in range(self.config.max_gen_tries):
            self.config.literal_ratio = 1.0
            root = gen.build_tree()
            self.config.literal_ratio = orig_ratio
            if _has_placeholder(root):
                continue
            code, _ = _emit_code(root)
            sr = _run_in_sandbox(code, self.config.exec_timeout)
            if not sr.ok or not sr.output:
                continue
            seq   = _dfs_call_sequence(root)
            lines = ["Tracing execution bottom-up:"]
            for i, name in enumerate(seq, 1):
                lines.append(f"  {i}. evaluate {name}()")
            lines.append(f"Final printed value: {sr.output}")
            meta = {
                "code": code, "tree": _tree_to_dict(root),
                "task_type": "output_prediction",
                "cot": "\n".join(lines),
                "code_hash": str(hash(code + sr.output)),
            }
            return Problem(metadata=meta, answer=sr.output)
        raise RuntimeError(
            f"OutputPrediction: no executable program found in "
            f"{self.config.max_gen_tries} attempts."
        )

    def prompt(self, metadata: dict) -> str:
        return (
            "Consider the following Python program:\n\n"
            f"```python\n{metadata['code']}\n```\n\n"
            "What does this program print? Give only the printed output, nothing else."
        )

    def score_answer(self, answer, entry) -> float:
        norm = lambda x: str(x).strip()
        if norm(answer) == norm(entry.answer):
            return 1.0
        return score_scalar(answer, entry)

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5 — TYPE INHABITATION
# ═══════════════════════════════════════════════════════════════════════════════

def _enumerate_compositions(
    target_type : str,
    index       : TypeIndex,
    inputs      : list[tuple[str, str]],   # [(name, type_str), ...]
    max_depth   : int,
    max_results : int = 50,
) -> list[str]:
    """
    Enumerate all valid expression strings producing `target_type`
    using functions from `index` and typed input variables `inputs`.
    Returns up to `max_results` expressions, shortest first.
    """
    # available_by_type: type_str -> list of expression strings
    available: dict[str, list[str]] = defaultdict(list)
    for name, t in inputs:
        available[t].append(name)

    results = []
    visited = set()

    def search(req_type: str, depth: int) -> list[str]:
        """Return all expressions of req_type reachable at this depth."""
        exprs = list(available.get(req_type, []))

        if depth == 0:
            return exprs

        for func in index.functions_returning(req_type):
            if func.output in WEAK_RETURN_TYPES:
                continue
            if not func.inputs:
                candidate = f"{func.name}()"
                if candidate not in visited:
                    exprs.append(candidate)
                continue

            # build all argument combinations
            arg_options = []
            feasible = True
            for pname, ptype in func.inputs:
                sub = search(ptype, depth - 1)
                if not sub:
                    feasible = False
                    break
                arg_options.append([(pname, e) for e in sub])

            if not feasible:
                continue

            # cartesian product of argument choices (capped to avoid explosion)
            from itertools import product as iproduct
            for combo in iproduct(*arg_options):
                arg_str = ", ".join(f"{n}={e}" for n, e in combo)
                expr = f"{func.name}({arg_str})"
                if expr not in visited:
                    visited.add(expr)
                    exprs.append(expr)
                if len(exprs) > max_results * 3:
                    return exprs[:max_results]

        return exprs

    raw = search(target_type, max_depth)
    # sort by length (shorter = simpler = easier)
    return sorted(set(raw), key=len)[:max_results]


def _make_type_error_distractor(
    valid_expr  : str,
    index       : TypeIndex,
    inputs      : list[tuple[str, str]],
    target_type : str,
    rng         : random.Random,
) -> Optional[str]:
    """
    Build a type-invalid expression that *looks* plausible.
    Strategy: take a valid expression, swap one argument with 
    a wrong-typed variable or swap function order.
    """
    # Strategy 1: find a function whose input type doesn't match
    # what we're feeding it — swap variable names
    input_names = [n for n, _ in inputs]
    input_types = {n: t for n, t in inputs}

    # Strategy 2: compose in wrong order
    # e.g. if valid is g(f(x)), try f(g(x)) — type error
    funcs_by_output = {}
    for n, t in inputs:
        funcs_by_output[t] = n

    # Find two functions that could be swapped to create a type error
    returning_target = index.unique_functions_returning(target_type)
    rng.shuffle(returning_target)

    for f in returning_target:
        if not f.inputs:
            continue
        for pname, ptype in f.inputs:
            # find something of the WRONG type for this slot
            wrong_exprs = [
                n for n, t in inputs if t != ptype
            ]
            if wrong_exprs:
                wrong = rng.choice(wrong_exprs)
                # build the broken call
                args = []
                for p, pt in f.inputs:
                    if p == pname:
                        args.append(f"{p}={wrong}")
                    else:
                        rights = [n for n, t in inputs if t == pt]
                        if not rights:
                            args.append(f"{p}=None")
                        else:
                            args.append(f"{p}={rng.choice(rights)}")
                distractor = f"{f.name}({', '.join(args)})"
                if distractor != valid_expr:
                    return distractor
    return None


def _is_clean_type(t: str) -> bool:
    """Reject opaque/polymorphic types that break reasoning."""
    bad = {"Union", "Optional", "Any", "Sequence", "Iterable",
           "Callable", "Iterator", "NoneType", "Decorator", "module"}
    return t not in bad and not any(b in t for b in bad)


def _get_clean_functions(index: TypeIndex) -> list[FunctionRecord]:
    """Return functions where all input and output types are concrete."""
    clean = []
    for f in index._all:
        if not _is_clean_type(f.output):
            continue
        if f.output in WEAK_RETURN_TYPES:
            continue
        if not f.inputs:
            continue
        if any(not _is_clean_type(t) for _, t in f.inputs):
            continue
        clean.append(f)
    return clean


def _select_toolkit(
    funcs       : list[FunctionRecord],
    target_type : str,
    rng         : random.Random,
    n_funcs     : int = 8,
) -> tuple[list[FunctionRecord], list[tuple[str, str]]]:
    """
    Select a small diverse toolkit of functions and derive input variables.
    Returns (selected_functions, input_variables).

    Strategy:
    - Always include at least one function returning target_type
    - Include functions returning types needed as inputs
    - Derive concrete typed input variables from leaf input types
    """
    # Group by output type
    by_output: dict[str, list[FunctionRecord]] = defaultdict(list)
    for f in funcs:
        by_output[f.output].append(f)

    # Must have functions for target type
    if not by_output.get(target_type):
        return [], []

    selected: list[FunctionRecord] = []
    seen_names: set[str] = set()

    # Pick 1-2 functions returning target type
    target_funcs = rng.sample(
        by_output[target_type],
        min(2, len(by_output[target_type]))
    )
    for f in target_funcs:
        if f.name not in seen_names:
            selected.append(f)
            seen_names.add(f.name)

    # Collect all input types needed by selected functions
    needed_types: set[str] = set()
    for f in selected:
        for _, t in f.inputs:
            needed_types.add(t)

    # For each needed input type, try to add a function producing it
    # (so compositions are possible)
    for t in list(needed_types):
        if t == target_type:
            continue
        producers = by_output.get(t, [])
        if producers and len(selected) < n_funcs:
            f = rng.choice(producers)
            if f.name not in seen_names:
                selected.append(f)
                seen_names.add(f.name)

    # Fill remaining slots with diverse functions (different output types)
    seen_outputs = {f.output for f in selected}
    remaining = [f for f in funcs if f.name not in seen_names
                 and f.output not in seen_outputs]
    rng.shuffle(remaining)
    for f in remaining:
        if len(selected) >= n_funcs:
            break
        selected.append(f)
        seen_names.add(f.name)
        seen_outputs.add(f.output)

    # Derive input variables: leaf types (types with no function producing them
    # in the selected toolkit, i.e., must come from outside)
    produced_types = {f.output for f in selected}
    leaf_types: set[str] = set()
    for f in selected:
        for _, t in f.inputs:
            if t not in produced_types:
                leaf_types.add(t)

    # Also add target type's direct input types as variables
    for f in target_funcs:
        for _, t in f.inputs:
            leaf_types.add(t)

    # Build named input variables
    type_counts: dict[str, int] = defaultdict(int)
    inputs: list[tuple[str, str]] = []
    for t in sorted(leaf_types):  # sorted for determinism
        prefix = t.lower()[:3].replace(" ", "_").replace("|", "")[:3]
        varname = f"{prefix}_{type_counts[t]}"
        type_counts[t] += 1
        inputs.append((varname, t))
        # Add a second variable of the same type for richer compositions
        if rng.random() < 0.4:
            varname2 = f"{prefix}_{type_counts[t]}"
            type_counts[t] += 1
            inputs.append((varname2, t))

    return selected, inputs

def _synthesize_chain(funcs, rng, depth):
    """
    Build a VALID composition chain of exact depth.
    Returns (expr_str, target_type, inputs)
    """

    chain = []
    current_type = None

    for _ in range(depth):
        if current_type is None:
            f = rng.choice(funcs)
        else:
            candidates = [fn for fn in funcs if fn.output == current_type]
            if not candidates:
                return None
            f = rng.choice(candidates)

        chain.append(f)
        if f.inputs:
            current_type = f.inputs[0][1]
        else:
            return None

    # Build expression
    inputs = []
    var_count = 0

    expr = None
    for f in reversed(chain):
        args = []
        for name, t in f.inputs:
            if expr is None:
                var = f"x{var_count}"
                inputs.append((var, t))
                var_count += 1
                args.append(f"{name}={var}")
            else:
                args.append(f"{name}={expr}")
        expr = f"{f.name}({', '.join(args)})"

    target_type = chain[0].output
    return expr, target_type, inputs




@dataclass
class TypeInhabitationCfg(Config):
    """
    max_depth      — max composition depth for valid solutions
    n_candidates   — total options shown (valid + invalid)
    n_valid        — how many valid compositions to include (1=easy, 2+=medium)
    max_gen_tries  — generation retry budget
    db_path        — path to functions DB
    """
    max_depth     : int = 2
    n_candidates  : int = 4
    n_valid       : int = 1
    max_gen_tries : int = 300
    db_path       : str = "functions.db"

    def update(self, delta: int):
        self.max_depth += delta
        self.n_valid   = min(self.n_valid + delta, 3)


class TypeInhabitation(_ProgramBase):
    """
    Given a typed toolkit of functions and variables, identify the valid
    expression(s) producing the target type.

    The correct answer always involves at least one function call.
    Distractors have type errors — wrong argument types fed to functions.

    Difficulty tiers:
      easy   — one valid composition
      medium — multiple valid compositions (model picks shortest/minimal)
      hard   — multiple valid + deeper chains required
    """

    def __init__(self, config=None):
        if config is None:
            config = TypeInhabitationCfg()
        Task.__init__(self, config=config)

    def _get_index(self) -> TypeIndex:
        db = self.config.db_path
        if db not in _ProgramBase._index_cache:
            _ProgramBase._index_cache[db] = TypeIndex(db)
        return _ProgramBase._index_cache[db]

    def generate(self) -> Problem:
        index = self._get_index()
        rng   = random.Random(self.config.seed)
        cfg   = self.config

        clean_funcs = _get_clean_functions(index)
        if not clean_funcs:
            raise RuntimeError("No clean functions found in database.")

        # Group clean functions by output type
        clean_by_output: dict[str, list[FunctionRecord]] = defaultdict(list)
        for f in clean_funcs:
            clean_by_output[f.output].append(f)

        # Only consider target types with enough functions for distractors
        viable_targets = [
            t for t, fs in clean_by_output.items()
            if len(fs) >= cfg.n_candidates
        ]
        if not viable_targets:
            raise RuntimeError("No viable target types with enough functions.")

        for attempt in range(cfg.max_gen_tries):

            # ── 1. Pick target type ───────────────────────────────────────
            # target_type = rng.choice(viable_targets)
            # target_type = f"Target_{rng.randint(0, 10**6)}"
            result = _synthesize_chain(
                selected_funcs,
                rng,
                depth=cfg.max_depth
            )

            if result is None:
                continue

            correct, target_type, toolkit_inputs = result

            # ── 2. Build toolkit ──────────────────────────────────────────
            selected_funcs, toolkit_inputs = _select_toolkit(
                funcs       = clean_funcs,
                target_type = target_type,
                rng         = rng,
                n_funcs     = 8,
            )

            if not selected_funcs or not toolkit_inputs:
                continue

            funcs_for_target = [f for f in selected_funcs
                                if f.output == target_type]
            if not funcs_for_target:
                continue

            # ── 3. Enumerate valid compositions (must include function call) ─
            filtered_index = FilteredIndex(selected_funcs)

            valid_exprs = _enumerate_compositions(
                target_type = target_type,
                index       = filtered_index,
                inputs      = toolkit_inputs,
                max_depth   = cfg.max_depth,
                max_results = 30,
            )

            # Filter: must involve at least one function call
            valid_exprs = [e for e in valid_exprs if "(" in e]
            if not valid_exprs:
                continue

            # ── 4. Select answers by difficulty ───────────────────────────
            n_valid = min(cfg.n_valid, len(valid_exprs))
            correct_exprs = valid_exprs[:n_valid]
            correct = correct_exprs[0]  # shortest = simplest

            if n_valid == 1:
                difficulty = "easy"
            elif n_valid == 2:
                difficulty = "medium"
            else:
                difficulty = "hard"

            # ── 5. Build distractors ──────────────────────────────────────
            n_distractors = cfg.n_candidates - 1
            distractors: list[str] = []

            # Primary strategy: type-error distractors
            for _ in range(n_distractors * 5):
                d = _make_type_error_distractor(
                    valid_expr  = correct,
                    index       = index,
                    inputs      = toolkit_inputs,
                    target_type = target_type,
                    rng         = rng,
                )
                if d and d not in distractors and d not in correct_exprs:
                    distractors.append(d)
                if len(distractors) >= n_distractors:
                    break

            # Fallback: valid expressions of wrong type
            if len(distractors) < n_distractors:
                other_types = [
                    t for t in clean_by_output
                    if t != target_type and _is_clean_type(t)
                ]
                rng.shuffle(other_types)
                for wt in other_types[:6]:
                    wrong_exprs = _enumerate_compositions(
                        target_type = wt,
                        index       = index,
                        inputs      = toolkit_inputs,
                        max_depth   = 1,
                        max_results = 4,
                    )
                    for we in wrong_exprs:
                        if "(" in we and we not in distractors and we not in correct_exprs:
                            distractors.append(we)
                            break
                    if len(distractors) >= n_distractors:
                        break

            if len(distractors) < 1:
                continue

            distractors = distractors[:n_distractors]

            # ── 6. Build candidate list ───────────────────────────────────
            candidates = [{"expr": correct, "is_correct": True}] + [
                {"expr": d, "is_correct": False} for d in distractors
            ]
            rng.shuffle(candidates)

            # ── 7. Build prompt strings ───────────────────────────────────
            # Show only selected toolkit functions
            toolkit_funcs_str = "\n".join(
                f"  {f.name}({', '.join(f'{n}: {t}' for n,t in f.inputs)}) → {f.output}"
                for f in selected_funcs
            )
            toolkit_vars_str = "\n".join(
                f"  {name}: {t}" for name, t in toolkit_inputs
            )

            # ── 8. Build CoT ──────────────────────────────────────────────
            cot_lines = [
                "Reasoning step by step:",
                f"1. Target type is `{target_type}`.",
                f"2. Available inputs: {', '.join(f'{n}: {t}' for n,t in toolkit_inputs)}",
                f"3. Functions in toolkit returning `{target_type}`: "
                   f"{', '.join(f.name for f in funcs_for_target)}",
                f"4. Check each candidate by tracing argument types:",
            ]
            for c in candidates:
                expr = c["expr"]
                valid = c["is_correct"]
                cot_lines.append(
                    f"   - `{expr}`: {'✓ all argument types match' if valid else '✗ type mismatch in arguments'}"
                )
            cot_lines.append(f"5. Therefore the answer is `{correct}`.")
            cot = "\n".join(cot_lines)

            meta = {
                "target_type":    target_type,
                "toolkit_funcs":  toolkit_funcs_str,
                "toolkit_vars":   toolkit_vars_str,
                "toolkit_inputs": toolkit_inputs,
                "candidates":     candidates,
                "valid_exprs":    correct_exprs,
                "difficulty":     difficulty,
                "task_type":      "type_inhabitation",
                "cot":            cot,
                "code_hash":      str(hash(correct + target_type + str(attempt))),
            }
            return Problem(metadata=meta, answer=correct)

        raise RuntimeError(
            f"TypeInhabitation: could not generate example in "
            f"{cfg.max_gen_tries} attempts."
        )

    def prompt(self, metadata: dict) -> str:
        cand_block = "\n".join(
            f"  {chr(65+i)}) {c['expr']}"
            for i, c in enumerate(metadata["candidates"])
        )
        return (
            "You are given a typed toolkit of functions and variables:\n\n"
            f"Functions:\n{metadata['toolkit_funcs']}\n\n"
            f"Variables:\n{metadata['toolkit_vars']}\n\n"
            f"Target type: `{metadata['target_type']}`\n\n"
            "Which of the following expressions correctly produces the target "
            "type with valid argument types?\n\n"
            f"{cand_block}\n\n"
            "Answer with the letter (A, B, C, ...) or the full expression."
        )

    def score_answer(self, answer, entry) -> float:
        ans        = str(answer).strip()
        candidates = entry.metadata.get("candidates", [])

        if len(ans) == 1 and ans.upper() in string.ascii_uppercase:
            idx = ord(ans.upper()) - ord("A")
            if 0 <= idx < len(candidates):
                return float(candidates[idx]["is_correct"])

        norm = lambda x: x.strip().replace(" ", "")
        valid_exprs = entry.metadata.get("valid_exprs", [entry.answer])
        return float(any(norm(ans) == norm(v) for v in valid_exprs))

    def deduplication_key(self, problem):
        return problem.metadata.get("code_hash", None)


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION DATASET GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_eval_set(
    db_path      : str  = "functions.db",
    output_path  : str  = "eval_set.json",
    n_per_tier   : int  = 100,
    seed         : int  = 42,
    use_sandbox  : bool = True,
) -> dict:
    """
    Generate a fixed evaluation dataset with balanced difficulty tiers.

    Args:
        db_path     : path to functions.db
        output_path : where to write the JSON file
        n_per_tier  : target number of examples per difficulty tier
        seed        : random seed for full reproducibility
        use_sandbox : whether to allow execution-tier examples

    Returns:
        dict with keys "execution", "semantic", "type-only", each a list
        of serialisable example dicts.

    Usage:
        python program_composition_task.py --generate-eval \
            --db functions.db --out eval_set.json --n 100

    Evaluation:
        Load eval_set.json, send each example["prompt"] to your LLM, then
        call score_example(model_answer, example) to get 0.0 or 1.0.
    """
    cfg = ProgramCompositionCfg(
        db_path     = db_path,
        seed        = seed,
        use_sandbox = use_sandbox,
        max_depth_node = 2,
    )
    task   = NodeCompletion(config=cfg)
    tiers  = {"execution": [], "semantic": [], "type-only": []}
    target = {t: n_per_tier for t in tiers}
    rng    = random.Random(seed)

    print(f"Generating eval set: {n_per_tier} examples per tier …")
    attempts = 0
    while any(len(tiers[t]) < target[t] for t in tiers):
        attempts += 1
        if attempts > n_per_tier * 50:
            print("Warning: hit attempt limit, some tiers may be short.")
            break
        try:
            ex = task.generate_example()
        except Exception:
            continue
        tier = ex.metadata.get("difficulty", "type-only")
        if len(tiers.get(tier, [])) >= target.get(tier, 0):
            continue
        tiers[tier].append({
            "prompt":      ex.prompt,
            "answer":      ex.answer,
            "difficulty":  tier,
            "candidates":  ex.metadata["candidates"],
            "masked_type": ex.metadata["masked_type"],
            "masked_module": ex.metadata["masked_module"],
            "code":        ex.metadata["code"],
            "cot":         ex.cot,
        })
        done  = sum(len(v) for v in tiers.values())
        total = sum(target.values())
        if done % 10 == 0:
            counts = {t: len(v) for t, v in tiers.items()}
            print(f"  {done}/{total}  {counts}")

    with open(output_path, "w") as f:
        json.dump(tiers, f, indent=2)

    counts = {t: len(v) for t, v in tiers.items()}
    print(f"\nSaved {sum(counts.values())} examples to {output_path}")
    print(f"Tier counts: {counts}")
    return tiers


def score_example(model_answer: str, example: dict) -> float:
    """
    Score one model answer against one example from the eval set.

    Args:
        model_answer : the model's raw response string
        example      : one dict from the eval set JSON

    Returns:
        1.0 if correct, 0.0 otherwise.
        Also accepts letter answers (A/B/C/D).
    """
    ans        = str(model_answer).strip()
    candidates = example.get("candidates", [])
    correct    = example["answer"]

    if len(ans) == 1 and ans.upper() in string.ascii_uppercase:
        idx = ord(ans.upper()) - ord("A")
        if 0 <= idx < len(candidates):
            return float(candidates[idx]["name"] == correct)

    return float(ans.strip().lower().strip("`") == correct.strip().lower())


def compute_accuracy(results: list[dict]) -> dict:
    """
    Compute per-tier accuracy from a list of scored results.

    Each result dict should have:
        {"difficulty": str, "score": float}

    Returns:
        {"execution": float, "semantic": float, "type-only": float, "overall": float}
    """
    from collections import defaultdict
    tier_scores: dict[str, list[float]] = defaultdict(list)
    for r in results:
        tier_scores[r["difficulty"]].append(r["score"])

    accuracy = {}
    all_scores = []
    for tier, scores in tier_scores.items():
        accuracy[tier] = sum(scores) / len(scores) if scores else 0.0
        all_scores.extend(scores)
    accuracy["overall"] = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return accuracy


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    TASKS = {
        "type":    TypePrediction,
        "node":    NodeCompletion,
        "trace":   ExecutionTracing,
        "output":  OutputPrediction,
        "inhabit": TypeInhabitation,
    }

    parser = argparse.ArgumentParser(description="Program composition tasks")
    parser.add_argument("--task",          choices=list(TASKS), default="node")
    parser.add_argument("--db",            default="functions.db")
    parser.add_argument("--depth",         type=int,  default=3)
    parser.add_argument("--n",             type=int,  default=5)
    parser.add_argument("--seed",          type=int,  default=None)
    parser.add_argument("--target",        default=None)
    parser.add_argument("--no-sandbox",    action="store_true")
    parser.add_argument("--generate-eval", action="store_true")
    parser.add_argument("--out",           default="eval_set.json")
    parser.add_argument("--n-per-tier",    type=int, default=100)
    args = parser.parse_args()


    if args.generate_eval:
        generate_eval_set(
            db_path     = args.db,
            output_path = args.out,
            n_per_tier  = args.n_per_tier,
            seed        = args.seed or 42,
            use_sandbox = not args.no_sandbox,
        )
        sys.exit(0)

    if args.task == "inhabit":
        cfg = TypeInhabitationCfg(
            db_path   = args.db,
            max_depth = args.depth,
            seed      = args.seed,
        )
    else:
        cfg = ProgramCompositionCfg(
            db_path     = args.db,
            max_depth   = args.depth,
            seed        = args.seed,
            target_type = args.target,
            use_sandbox = not args.no_sandbox,
        )

    task = TASKS[args.task](config=cfg)

    for i in range(args.n):
        print(f"\n{'═'*60}")
        print(f"  [{args.task.upper()}] EXAMPLE {i+1}")
        print(f"{'═'*60}")
        ex = task.generate_example()
        print("\n── Prompt ──")
        print(ex.prompt)
        print("\n── Answer ──")
        print(ex.answer)
        print("\n── CoT ──")
        print(ex.cot)
        print("\n── Difficulty ──")
        print(ex.metadata.get("difficulty", "n/a"))
        print("\n── Score (self-check) ──")
        print(task.score_answer(ex.answer, ex))