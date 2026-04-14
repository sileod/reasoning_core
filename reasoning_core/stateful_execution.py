import random
import sys 
import ast 
import io 

from dataclasses import dataclass
from typing import Optional
from easydict import EasyDict as edict
from template import Task, Problem, Config

@dataclass
class StatefulConfig(Config):
    n_vars: int = 5
    n_steps: int = 7
    allow_aliasing: bool = True
    query_type: str = "final"
    data_structure: str = "dict"  # "list", "dict", "random"


    def update(self, c):
        self.n_steps += int(c * 2)
        self.n_vars += int(c)
        if c>=2: 
            self.allow_aliasing = True

#### XXX: Code generator
def make_list_program(n_vars: int, n_steps: int, rng: random.Random):
    """
    Generate a simple program to manipulate a list with possible aliasing. 

    Return (source_code, var_names)
    """

    var_names = [chr(ord('a') + i) for i in range(n_vars)]
    lines = []

    ### NOTE: init variables as lists
    for v in var_names:
        init = [rng.randint(0, 9) for _ in range(rng.randint(1, 3))]
        lines.append(f"{v} = {init}")

    ops = ["append", "pop", "index_assign", "augment"]
    for _ in range(n_steps):
        v = rng.choice(var_names)
        op = rng.choice(ops)

        if op == "append":
            val = rng.randint(0, 9)
            lines.append(f"{v}.append({val})")

        elif op == "pop":
            ### NOTE: only pop if list is non-empty so guard with if
            lines.append(f"if {v}: {v}.pop()")

        elif op == "index_assign":
            val = rng.randint(0, 9)
            #### NOTE: guard against empty list
            lines.append(f"if {v}: {v}[0] = {val}")

        elif op == "augment":
            val = rng.randint(0, 9)
            lines.append(f"if {v}: {v}[0] += {val}")

    return lines, var_names

def make_dict_program(n_vars: int, n_steps: int, rng: random.Random):
    """
    Generate a simple program to manipulate dictionaries.

    Return (lines, var_names)
    """
    var_names = [chr(ord('a') + i) for i in range(n_vars)]
    lines = []

    ### NOTE: init variables as dicts
    for v in var_names:
        d = {}
        for _ in range(rng.randint(1, 3)):
            key = rng.randint(0, 5)
            val = rng.randint(0, 9)
            d[key] = val
        lines.append(f"{v} = {d}")

    if rng.random() < 0.3 and len(var_names) >= 2:
        v1, v2 = rng.sample(var_names, 2)
        lines.append(f"{v2} = {v1}")

        # FORCE rebind after alias
        if rng.random() < 0.5:
            new_dict = {rng.randint(0,5): rng.randint(0,9)}
            lines.append(f"{v1} = {new_dict}")

    if rng.random() < 0.2:
        v = rng.choice(var_names)
        new_dict = {rng.randint(0,5): rng.randint(0,9)}
        lines.append(f"{v} = {new_dict}")

    if rng.random() < 0.2:
        fake_key = 999
        v = rng.choice(var_names)
        lines.append(f"if {fake_key} in {v}: {v}[{fake_key}] = {rng.randint(0,9)}")

    ops = ["set", "update", "delete", "augment"]

    for _ in range(n_steps):
        v = rng.choice(var_names)
        op = rng.choice(ops)

        key = rng.randint(0, 5) 
        val = rng.randint(0, 9)

        if op == "set":
            lines.append(f"{v}[{key}] = {val}")
        elif op == "update":
            lines.append(f"{v}.update({{{key}: {val}}})")
        elif op == "delete":
            lines.append(f"if {key} in {v}: del {v}[{key}]")
        elif op == "augment":
            lines.append(f"if {key} in {v}: {v}[{key}] += {val}")
    
    return lines, var_names



def _make_alias_program(rng: random.Random):
    """
    Generate a program with a deliberate aliasing trap

    Return (lines, alias_var, original_var)
    """
    init = [rng.randint(0, 9) for _ in range(3)]
    lines = [
        f"x = {init}",
        f"y = x"
    ]
    for _ in range(rng.randint(1, 3)):
        lines.append(f"x.append({rng.randint(0,9)})")

    lines.append(f"y.append({val})")
    return lines, "x", "y"

def _make_dict_alias_program(rng):
    d = {rng.randint(0,5): rng.randint(0,9) for _ in range(2)}
    lines = [
        f"x = {d}",
        "y = x",
        f"y[{rng.randint(0,5)}] = {rng.randint(0,9)}"
    ]
    return lines, "x", "y"


#### XXX: Tracer
def _trace_execution(source: str):
    snapshots = []
    locals_dict = {}

    code_lines = source.splitlines()

    for i in range(len(code_lines)):
        partial_code = "\n".join(code_lines[:i+1])

        try:
            exec(partial_code, {}, locals_dict)
        except Exception:
            return []

        snapshots.append({
            "line": i+1,
            "code": code_lines[i],
            "locals": {
                k: (
                    list(v) if isinstance(v, list)
                    else {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict)
                    else v
                )
                for k, v in locals_dict.items()
            }
        })

    return snapshots


#### XXX: CoT Builder
def _build_cot(snapshots, query_var, query_line=None):
    lines = ["Tracing execution step by step:"]
    for snap in snapshots:
        state_str = ", ".join(
            f"{k}={v}" for k,v in snap["locals"].items()
        )
        lines.append(f" Line {snap['line']}: '{snap['code']}' -> {state_str}")

    if query_line is not None:
        ### NOTE: find the snapshot at or just before query_line
        target = None
        for snap in snapshots:
            if snap["line"] <= query_line:
                target = snap
        if target and query_var in target["locals"]:
            lines.append(
                f"At line {query_line}, {query_var} = {target['locals'][query_var]}"
            )
    else:
        last = snapshots[-1] if snapshots else {}
        locs = last.get("locals", {})
        if query_var in locs:
            lines.append(f"Final value of {query_var} = {locs[query_var]}")

    return "\n".join(lines)


#### XXX: Task
class StatefulExecution(Task):

    def __init__(self, config=StatefulConfig()):
        super().__init__(config=config)

    def generate(self, attempt=0) -> Problem:
        cfg = self.config
        rng = random.Random(cfg.seed + attempt if cfg.seed is not None else None)
        query_type = cfg.query_type

        if attempt > 50:
            raise RuntimeError("Failed to generate valid StatefulExecution task")

        if query_type == "count_keys":
            answer = str(len(last_locals[query_var]))

        ### NOTE: Build program
        if query_type == "alias_trap" or (
            cfg.allow_aliasing and rng.random() < 0.4
        ):
            if cfg.data_structure == "dict" or (
                cfg.data_structure == "random" and rng.random() < 0.5
            ):
                lines, query_var, alias_var = _make_dict_alias_program(rng)
            else:
                lines, query_var, alias_var = _make_alias_program(rng)

            query_type = "alias_trap"
            query_line = None
        else:
            if cfg.data_structure == "random":
                dtype = rng.choice(["list", "dict"])
            else:
                dtype = cfg.data_structure

            if dtype == "list":
                lines, var_names = make_list_program(cfg.n_vars, cfg.n_steps, rng)
            else:
                lines, var_names = make_dict_program(cfg.n_vars, cfg.n_steps, rng)


            query_var = rng.choice(var_names)
            query_line = (
                rng.randint(len(var_names), len(lines))
                if query_type == "intermediate"
                else None
            )

        source = "\n".join(lines)

        ### NOTE: trace
        snapshots = _trace_execution(source)
        if not snapshots:
            ### NOTE: retry
            return self.generate(attempt + 1)

        ### NOTE: Extract answer
        if query_type == "intermediate" and query_line is not None:
            target_snap = None
            for snap in snapshots:
                if snap["line"] <= query_line:
                    target_snap = snap
            if target_snap is None or query_var not in target_snap["locals"]:
                return self.generate(attempt + 1)
            answer = str(target_snap["locals"][query_var])
        else:
            last_locals = snapshots[-1]["locals"]
            if query_var not in last_locals:
                return self.generate(attempt + 1)
            answer = str(last_locals[query_var])

        ### NOTE: CoT
        cot = _build_cot(snapshots, query_var, query_line)
        meta = edict({
            "source": source,
            "query_var": query_var,
            "query_type": query_type,
            "query_line": query_line,
            "snapshots": snapshots,
            "cot": cot,
        })
        return Problem(metadata=meta, answer=answer)

    def prompt(self, metadata) -> str:
        src = metadata.source
        qv = metadata.query_var
        qt = metadata.query_type
        ql = metadata.query_line

        if qt == "intermediate":
            return (
                f"```python\n{src}\n```\n\n"
                f"What is the value of `{qv}` after line {ql} executes?\n"
                f"Answer with the exact Python value (e.g. [1, 2, 3])"
            )
        elif qt == "alias_trap":
            return (
                f"```python\n{src}\n```\n\n"
                f"what is the final value of `{qv}`?\n"
                f"Answer with the exact Python value."
            )
        elif qt == "count_keys":
            return f"... How many keys are in `{qv}` at the end?"
        else:
            return (
                f"```python\n{src}\n```\n\n"
                f"What is the final value of `{qv}`?\n"
                f"Answer with the exact Python value (e.g. [1, 2, 3])."
            )

    def normalize_dict(self, d):
        if isinstance(d, dict):
            return {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in d.items()}
        return d

    def score_answer(self, answer, entry) -> float:
        try:
            pred = ast.literal_eval(answer)
            truth = ast.literal_eval(entry.answer)

            pred = self.normalize_dict(pred)
            truth = self.normalize_dict(truth)
            if pred == truth:
                return 1.0

            # list partial credit
            if isinstance(pred, list) and isinstance(truth, list):
                if len(truth) == 0:
                    return 1.0 if len(pred) == 0 else 0.0
                matches = sum(
                    1 for a, b in zip(pred, truth) if a == b
                )
                return 0.5 * matches / max(len(pred), len(truth))

            # dict exact match
            if isinstance(pred, dict) and isinstance(truth, dict):
                return 1.0 if pred == truth else 0.0

            return 0.0

        except Exception:
            return 0.0