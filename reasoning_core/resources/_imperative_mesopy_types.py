from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Iterable

INT = "int"
BOOL = "bool"
LIST_INT = "list[int]"
DICT_INT = "dict[int,int]"
TUPLE_INT = "tuple[int,int]"

OBSERVED_ERRORS = ("IndexError", "ZeroDivisionError", "ValueError", "KeyError")
CONTROLLED_PHENOMENA = (
    "aliasing",
    "closure_late_binding",
    "default_capture",
    "mutation_call",
    "loop_carried_state",
    "rebinding_vs_aliasing",
)
PHENOMENA = CONTROLLED_PHENOMENA + (
    "mutable_default",
    "conditional_flow",
    "try_except",
    "early_return",
    "comprehension",
    "mapping",
    "recursion",
)

_REALISTIC_NAMES = (
    "total", "idx", "buf", "scale", "offset", "rows", "acc", "tmp", "n", "k",
    "value", "values", "items", "data", "count", "step", "delta", "result", "out",
    "left", "right", "current", "next_value", "limit", "pos", "key", "table", "seq",
    "work", "score", "base", "factor", "part", "cache", "state", "nums", "target",
)

@dataclass(frozen=True)
class MesopyComplexity:
    statements: int = 10
    expr_depth: int = 3
    control_depth: int = 2
    functions: int = 2
    call_depth: int = 2
    dataflow_depth: int = 5
    loop_bound: int = 4

    @classmethod
    def level(cls, level: int) -> "MesopyComplexity":
        level = max(0, int(level))
        # Recursive expression/control expansion is multiplicative, so keep its
        # depth bounded and spend higher levels on semantic interactions instead.
        structural_tier = min(1, (level + 1) // 3)
        return cls(
            statements=7 + level // 3,
            expr_depth=2 + structural_tier,
            control_depth=1 + structural_tier,
            functions=1 + level // 3,
            call_depth=1 + level // 3,
            dataflow_depth=3 + level,
            loop_bound=3 + level,
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
    min_live_fraction: float = 0.35
    min_param_sensitivity: float = 0.0
    anonymize_names: bool | None = None

@dataclass
class MesopyConfig:
    magnitude: int = 7
    list_size: tuple[int, int] = (3, 7)
    input_arity: tuple[int, int] = (1, 3)
    complexity: MesopyComplexity = field(default_factory=MesopyComplexity)
    probe_count: int = 16
    risk_rate: float = 0.35
    max_risk_sites: int = 2
    phenomenon_rate: float = 0.22
    noise_rate: float = 0.12
    recursion_rate: float = 0.24
    anonymize_names: bool = False
    deduplicate: bool = False
    max_attempts: int = 28
    profile_accepted: bool = True
    max_profile_steps: int = 100_000
    max_source_chars: int | None = None

@dataclass(frozen=True)
class Risk:
    kind: str
    deps: frozenset[str]

@dataclass(frozen=True)
class ExprSpec:
    node: ast.expr
    typ: str
    deps: frozenset[str] = frozenset()
    risks: tuple[Risk, ...] = ()
    depth: int = 1

@dataclass
class VarInfo:
    typ: str
    deps: frozenset[str]
    depth: int = 1
    length: int | None = None

@dataclass(frozen=True)
class CallOutcome:
    args: tuple[int, ...]
    ok: bool
    value: str | None = None
    error: str | None = None
    elapsed: float | None = None
    steps: int | None = None
    distinct_lines: int | None = None

@dataclass
class MesopySample:
    code: str
    entrypoint: str
    calls: tuple[CallOutcome, ...]
    phenomena: tuple[str, ...]
    features: dict
    fingerprint: str

    @property
    def call(self) -> CallOutcome:
        return self.calls[0]

    @property
    def args(self) -> tuple[int, ...]:
        return self.call.args

    @property
    def answer(self) -> str | None:
        return self.call.value if self.call.ok else self.call.error

@dataclass(frozen=True)
class MinimalPair:
    original: MesopySample
    code: str
    entrypoint: str
    outcome: CallOutcome
    mutation: str

class _Names:
    def __init__(self):
        self.n = 0

    def take(self, prefix: str = "v") -> str:
        self.n += 1
        return f"{prefix}{self.n}"

class _Env:
    def __init__(self, params: Iterable[str] = ()):
        self.vars: dict[str, VarInfo] = {}
        self.params = tuple(params)
        self.functions: list[str] = []
        for p in self.params:
            self.vars[p] = VarInfo(INT, frozenset({p}), 0)

    def copy(self) -> "_Env":
        other = _Env()
        other.vars = dict(self.vars)
        other.params = self.params
        other.functions = self.functions[:]
        return other

    def add(self, name: str, spec: ExprSpec | VarInfo) -> None:
        if isinstance(spec, ExprSpec):
            self.vars[name] = VarInfo(spec.typ, spec.deps, spec.depth)
        else:
            self.vars[name] = spec

    def names(self, typ: str | None = None) -> list[str]:
        if typ is None:
            return list(self.vars)
        return [name for name, info in self.vars.items() if info.typ == typ]

    def info(self, name: str) -> VarInfo:
        return self.vars[name]
