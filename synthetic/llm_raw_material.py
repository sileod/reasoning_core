"""
llm_raw_material.py

For each in-scope task, this module swaps out ONLY the one seam that normally
calls gramforge (or scrapes real libraries) for LLM-authored raw material,
while leaving every other line of the task's real, unmodified pipeline in
place: sandboxed execution, retry/acceptance logic, Entry construction,
render_prompt, score_answer, generate_example()'s metadata bookkeeping, etc.

This is what makes the LLM-generated rows land in the exact same TaskRow
schema as the procedural cache, with zero changes needed to task_influence.py
or task_diagnostics/cache.py.

Each `rig_*` function returns a `Rig` with:
  - .task            the constructed Task instance (already monkeypatched)
  - .fewshot()        -> a dict of strings to show the LLM as a structural example
  - .inject(**kwargs)  install one piece of LLM-authored material before generating
  - .prompt_spec()    -> dict describing what the LLM should produce (used to build
                         the actual chat prompt in llm_generate_tasks.py)

Injection points, one per task:
  code_execution        -- module-level `make_code()` in code_execution.py
  code_runnability       -- CodeRunnability.generate_entry(_case=...) (public API, no patch needed)
  code_analysis          -- CodeAnalysis._make_kripke (bound instance method)
  code_iterations        -- TemporalReasoning._make_driver (bound instance method)
  code_input_deduction   -- CodeInputDeduction.generate_entry (whole method, mode-aware)
  type_inhabitation      -- reasoning_core.tasks.code_reasoning._func_cache[db_path] (module cache)
  code_repair            -- same _func_cache injection, shared with type_inhabitation
"""
from __future__ import annotations

import ast
import random
import types
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Optional

from reasoning_core.template import edict


@dataclass
class Rig:
    task: Any
    inject: Callable[..., None]
    fewshot: Callable[[], dict]
    prompt_spec: dict
    # Only set for code_runnability: takes freshly-injected code and returns a
    # list of *both* Problems (bad + good) it was able to extract from it.
    harvest_pair: Optional[Callable[[Any, Any], list]] = None


# --------------------------------------------------------------------------
# code_execution
# --------------------------------------------------------------------------

def rig_code_execution(cfg=None):
    import reasoning_core.tasks.code_execution as ce

    task = ce.CodeExecution(config=cfg)
    state = {"code": None}
    original_make_code = ce.make_code

    def patched_make_code(cfg_, failure_rate, profile="full"):
        # Real generate_entry()/sample_problem() retries this call internally
        # (cfg.max_attempts times) purely to vary *random endpoint call args*
        # against the SAME LLM-authored program -- that's still a legitimate,
        # free (no extra API call) source of variety, so we deliberately keep
        # returning the same fixed code here rather than regenerating it.
        return state["code"]

    def inject(code_text):
        state["code"] = code_text
        ce.make_code = patched_make_code

    def restore():
        ce.make_code = original_make_code

    def fewshot():
        real_task = ce.CodeExecution(config=cfg)
        entry = real_task.generate_entry()
        return {"program": entry.metadata.code, "call": entry.metadata.call, "value": entry.metadata.value if hasattr(entry.metadata, "value") else entry.answer}

    task._llm_restore = restore
    return Rig(
        task=task,
        inject=inject,
        fewshot=fewshot,
        prompt_spec=dict(
            kind="mesopy_program",
            instructions=(
                "Write a small, self-contained Python program with genuine, meaningful logic -- "
                "the kind of small utility you'd actually find in a real codebase (for example: a "
                "string parser, a checksum/hash-like routine, a data-validation check, a small "
                "numeric algorithm, a list/dict transformation, a simple encoder/decoder). Do NOT "
                "write arbitrary or meaningless toy code just to satisfy the constraints below -- "
                "the logic itself should be genuinely purposeful, with clear, meaningful function "
                "and variable names that you choose yourself (not f0/f1/a/b).\n\n"
                "Hard constraints the code must satisfy:\n"
                "- One or more small functions with type-annotated int/str/list parameters and "
                "return types.\n"
                "- Only basic control flow: if/elif/else, for/while loops, arithmetic, string and "
                "list operations. No imports, no file/network I/O, no randomness.\n"
                "- A final `def endpoint(...) -> ...:` function that calls into your other "
                "function(s) and returns the result -- this is the entry point that gets executed "
                "and checked.\n"
                "- Under 15 lines total.\n"
                "- Must run to completion quickly (sandboxed: 1 CPU-second limit, 512MB memory, "
                "recursion depth ~80) and return a plain int, str, or list -- avoid unbounded "
                "loops or deep recursion."
            ),
        ),
    )


# --------------------------------------------------------------------------
# code_runnability
# --------------------------------------------------------------------------

def rig_code_runnability(cfg=None):
    import reasoning_core.tasks.code_execution as ce

    task = ce.CodeRunnability(config=cfg)

    def llm_runnability_pair(cfg_, base_code):
        """Mirrors ce.runnability_pair(), but takes the base program as input
        instead of generating it via make_code(). Faithfully reuses the real
        probing/mutation/verification logic."""
        probes = ce.endpoint_probes(base_code, cfg_)
        if len(probes) < 2:
            return None
        mutations = list(ce.organic_mutations(base_code))
        first_by_kind = {}
        for kind, candidate in mutations:
            first_by_kind.setdefault(kind, candidate)
        perturbed = list(first_by_kind.items()) + mutations[:4]
        random.shuffle(perturbed)
        perturbed.sort(key=lambda item: item[0] in {"operator", "denominator"})
        candidates = [("organic", base_code)] + perturbed
        for kind, code in candidates:
            reports = ce.run_code(code, cfg_, call_args=probes, batch=True, reports=True)
            good = next((r for r in reports if r.ok), None)
            bad = next((r for r in reports if r.error), None)
            if good and bad:
                verified = [ce.run_code(code, cfg_, call_args=r.args) for r in (bad, good)]
                if (
                    verified[0].error == bad.error
                    and verified[1].ok
                    and min(r.steps for r in verified) >= min(6, 4 + int(cfg_.difficulty) // 3)
                ):
                    return kind, ((code, verified[0]), (code, verified[1]))
        return None

    def harvest_pair(llm_task, code_text):
        result = llm_runnability_pair(llm_task.config, code_text)
        if result is None:
            return []
        kind, (bad_case, good_case) = result
        cases = [(kind, bad_case), (kind, good_case)]
        random.shuffle(cases)
        return [dict(_case=case) for case in cases]  # kwargs to feed generate_example(**kw)

    def inject(_code_text):
        pass  # nothing to monkeypatch; harvest_pair() consumes the code directly

    def fewshot():
        real_task = ce.CodeRunnability(config=cfg)
        entry = real_task.generate_entry()
        return {"program": entry.metadata.code, "call": entry.metadata.call, "answer": entry.answer}

    return Rig(
        task=task,
        inject=inject,
        fewshot=fewshot,
        harvest_pair=harvest_pair,
        prompt_spec=dict(
            kind="mesopy_program",
            instructions=(
                "Write a small, self-contained Python program with genuine, meaningful logic -- "
                "the kind of small utility you'd actually find in a real codebase (for example: a "
                "string parser, a data-validation check, a small numeric algorithm, a list/dict "
                "lookup or transformation). Do NOT write arbitrary or meaningless toy code -- the "
                "logic itself should be genuinely purposeful, with clear, meaningful function and "
                "variable names that you choose yourself (not f0/f1/a/b).\n\n"
                "Hard constraints the code must satisfy:\n"
                "- One or more small functions with type-annotated int/str/list parameters and "
                "return types.\n"
                "- Only basic control flow: if/elif/else, for/while loops, arithmetic, string and "
                "list operations. No imports, no file/network I/O, no randomness.\n"
                "- A final `def endpoint(...) -> ...:` function that calls into your other "
                "function(s) and returns the result.\n"
                "- Under 15 lines total.\n\n"
                "CRITICAL requirement -- read carefully: your code MUST behave correctly for MOST "
                "of its possible test inputs (listed below), but MUST also genuinely fail (raise an "
                "exception) for AT LEAST ONE of them, via a real, natural bug -- not an artificial "
                "crash. Pick ONE concrete bug category and build it in on purpose:\n"
                "  * indexing a list/string at a fixed position (e.g. items[1] or items[-2]) without "
                "checking length first, so it fails on [] or a single-element list\n"
                "  * dividing or taking modulo by a value derived from the input, so it fails when "
                "that value is 0\n"
                "  * looking up a dict key that isn't guaranteed to exist\n"
                "  * assuming a string has a certain minimum length or a specific character in it\n"
                "Do NOT add bounds-checks, try/except, or `if not x: return` guards that would "
                "prevent this bug from firing. This is the single most important requirement; code "
                "that always succeeds no matter what is not useful here."
            ),
        ),
    )


# --------------------------------------------------------------------------
# code_analysis
# --------------------------------------------------------------------------

def _literal_for_domain(value):
    """Mirrors reasoning_core.tasks.code_analysis._literal: bare True/False/digits
    render unquoted, everything else (category labels) renders as a Python repr."""
    if value in ("True", "False") or value.isdigit():
        return value
    return repr(value)


def rig_code_analysis(cfg=None):
    from reasoning_core.tasks.code_analysis import CodeAnalysis, _pick_variables, _reachable, _BadProgram

    task = CodeAnalysis(config=cfg)
    state = {"vars_": None, "program": None, "kripke": None}

    def build_kripke(self_task, vars_, program):
        states = list(product(*[range(len(domain)) for _, domain in vars_]))
        index = {state: i for i, state in enumerate(states)}
        compile(program, "<llm_code_analysis_program>", "exec")
        if len(program.splitlines()) > 40:
            raise _BadProgram("program too long")
        succ = []
        for state in states:
            choices = [index[s] for s in self_task._execute_successors(program, vars_, state)]
            succ.append(sorted(set(choices)) or [index[state]])
        reachable = _reachable(succ, 0)
        features = sorted({
            type(node).__name__.lower()
            for node in ast.walk(ast.parse(program))
            if isinstance(node, (ast.If, ast.Match, ast.Return, ast.Tuple, ast.Dict, ast.IfExp))
        })
        return edict(
            vars=vars_, states=states, succ=succ, initial=0, reachable=reachable,
            program=program, syntax="+".join(features or ["assign"]),
        )

    def patched_make_kripke(self_task):
        # Cached: real _make_kripke() picks a brand new random program every
        # call, but generate_entry() retries this up to cfg.max_retries times
        # (default 200) across different random query_types/formulas *layered
        # on top of* whatever kripke it got -- so returning the SAME kripke on
        # every retry is faithful (just skips useless recomputation) rather
        # than a behavior change.
        if state["kripke"] is None:
            state["kripke"] = build_kripke(self_task, state["vars_"], state["program"])
        return state["kripke"]

    task._make_kripke = types.MethodType(patched_make_kripke, task)

    def wrap_step_body(vars_, body_text):
        """Wrap an LLM-authored step() body into the full program text that
        _execute_successors()/exec() expects: initial assignments + a def
        step(): global ...: <body> block -- mirrors _transition_grammar's
        render_program()."""
        from textwrap import indent
        names = ", ".join(name for name, _ in vars_)
        initial = ", ".join(_literal_for_domain(domain[0]) for _, domain in vars_)
        body = body_text.strip("\n")
        needs_random = "random." in body_text
        header = "import random\n\n" if needs_random else ""
        return f"{header}{names} = {initial}\n\ndef step():\n    global {names}\n{indent(body, '    ')}\n"

    def inject(body_text, vars_=None):
        state["vars_"] = vars_ if vars_ is not None else state["vars_"]
        state["program"] = wrap_step_body(state["vars_"], body_text)
        state["kripke"] = None

    def pick_vars():
        return _pick_variables(task.config)

    def fewshot():
        from reasoning_core.tasks.code_analysis import _transition_grammar
        from gramforge import generate as gf_generate
        demo_vars = _pick_variables(task.config)
        demo_program = gf_generate(_transition_grammar(demo_vars), depth=8, min_depth=4) @ "py"
        return {"vars": demo_vars, "program": demo_program}

    task._llm_pick_vars = pick_vars
    return Rig(
        task=task,
        inject=inject,
        fewshot=fewshot,
        prompt_spec=dict(
            kind="step_function_body",
            instructions=(
                "Write ONLY the body of a Python `step()` function that mutates the given global "
                "variables in place (no `def step():` line, no `global` line, no imports -- just "
                "the statements inside it; the caller adds those).\n\n"
                "Write REALISTIC, meaningful transition logic, not arbitrary state-shuffling: pick "
                "whatever plausible real-world system these variable names and possible values "
                "could represent (e.g. an order's status plus a retry counter, a traffic light's "
                "phase plus a pedestrian-request flag, a device's mode plus a locked flag -- "
                "choose an interpretation that actually fits the names/domains below) and write "
                "the kind of state-transition logic that real system would genuinely have.\n\n"
                "Use only: plain assignments, tuple assignments, `match`/`case` on one variable, "
                "`if`/`elif`/`else`, and occasionally an early-return guard (`if <cond>:\\n    "
                "<assignment>\\n    return`). For nondeterministic branches use "
                "`random.choice([...])` on the right-hand side of an assignment. Keep values "
                "within each variable's domain at all times, and make sure the natural starting "
                "state can reach several different combinations of these variables over repeated "
                "calls -- don't get stuck cycling through only 1-2 states. 4-10 lines."
            ),
        ),
    )


# --------------------------------------------------------------------------
# code_iterations (temporal_reasoning)
# --------------------------------------------------------------------------

def rig_code_iterations(cfg=None):
    from reasoning_core.tasks.code_reasoning import TemporalReasoning

    task = TemporalReasoning(config=cfg)
    state = {"core": None}
    original_make_driver = TemporalReasoning._make_driver  # unbound, for fewshot use

    def patched_make_driver(self_task, cfg_):
        x0 = random.randint(cfg_.lo, cfg_.hi)
        code = (
            f"{state['core']}\nSTATE = {x0}\n\n"
            "def endpoint():\n    global STATE\n    STATE = f0(STATE)\n    return STATE\n"
        )
        return code, x0

    task._make_driver = types.MethodType(patched_make_driver, task)

    def inject(core_text):
        state["core"] = core_text

    def fewshot():
        demo_task = TemporalReasoning(config=cfg)
        code, x0 = original_make_driver(demo_task, demo_task.config)
        core = code.split("\nSTATE =")[0]
        return {"core": core, "x0": x0}

    return Rig(
        task=task,
        inject=inject,
        fewshot=fewshot,
        prompt_spec=dict(
            kind="int_mutator_function",
            instructions=(
                "Write a small Python function `def f0(s: int) -> int:` (optionally with one or two "
                "int->int helper functions f1, f2 that f0 calls) implementing a genuine, meaningful "
                "state-update rule -- as if it were one real step of an actual algorithm (e.g. one "
                "round of a hash/checksum-style mix, a counter with a real wraparound or decay "
                "rule, one step of a simple simulation). It must really depend on `s` (not return a "
                "constant, not just `return s`). No imports, no randomness, no I/O, no recursion. "
                "Under 12 lines. Use clear, meaningful variable names where it helps readability, "
                "not arbitrary single letters."
            ),
        ),
    )


# --------------------------------------------------------------------------
# code_input_deduction
# --------------------------------------------------------------------------

def rig_code_input_deduction(cfg=None, include_tuple_mode=False):
    from reasoning_core.tasks.code_execution import (
        CodeInputDeduction, bounded_strings, run_code, RunReport, function_triviality,
    )
    from reasoning_core.template import Entry

    task = CodeInputDeduction(config=cfg)
    state = {"core1": None, "core2": None}  # arity-1 core (int/str modes), arity-2 core (tuple mode)
    modes_available = ["int", "str"] + (["tuple"] if include_tuple_mode else [])

    def llm_generate_entry(self_task):
        # Faithful copy of CodeInputDeduction.generate_entry(), with the single
        # `core = generate(mesopy_grammar(...)) @ "py"` line replaced by a
        # lookup into our pre-fetched, arity-matched LLM cores.
        cfg_ = self_task.config
        modes = list(modes_available)
        random.shuffle(modes)
        for mode in modes:
            for _ in range(max(1, cfg_.max_attempts // len(modes))):
                if mode == "int":
                    domain = list(range(cfg_.lo, cfg_.hi + 1))
                    call = lambda x: [x]
                    goal = f"smallest integer x in [{cfg_.lo}, {cfg_.hi}]"
                    call_text, answer_hint = "endpoint(x)", "Answer with the integer."
                    endpoint = f"def endpoint(x):\n    return f0(x) % {random.choice((3, 4, 5))}\n"
                    core = state["core1"]
                elif mode == "tuple":
                    domain = [(x, y) for x in range(cfg_.lo, cfg_.hi + 1) for y in range(cfg_.lo, cfg_.hi + 1)]
                    call = lambda xy: list(xy)
                    goal = f"lexicographically smallest integer pair (x, y) with each value in [{cfg_.lo}, {cfg_.hi}]"
                    call_text, answer_hint = "endpoint(x, y)", "Answer as `x y`."
                    endpoint = f"def endpoint(x, y):\n    return f0(x, y) % {random.choice((3, 4, 5))}\n"
                    core = state["core2"]
                else:
                    domain = bounded_strings(cfg_.alphabet, cfg_.max_len)
                    call = lambda s: [sum((len(cfg_.alphabet) ** i) * cfg_.alphabet.index(ch) for i, ch in enumerate(reversed(s)))]
                    goal = f"lexicographically smallest string s over `{cfg_.alphabet}` with length 1..{cfg_.max_len}"
                    call_text, answer_hint = "endpoint(s)", "Answer with the string."
                    endpoint = (
                        f"def endpoint(s):\n    z = 0\n    for ch in s:\n        z = {len(cfg_.alphabet)} * z + "
                        f"{repr(cfg_.alphabet)}.index(ch)\n    return f0(z) % {random.choice((3, 4, 5))}\n"
                    )
                    core = state["core1"]

                code = f"{core}\n\n{endpoint}"
                call_args = [call(x) for x in domain]
                r = run_code(code, cfg_, call_args=call_args, batch=True)
                reports = [
                    RunReport(True, value, None, args, r.stdout, r.stderr, r.steps, r.elapsed)
                    for args, value in zip(call_args, r.value or [])
                ] if r.ok else [r]
                buckets = {}
                for x, rep in zip(domain, reports):
                    if rep.ok and rep.value is not None:
                        buckets.setdefault(rep.value, []).append(x)
                if function_triviality(reports):
                    continue
                choices = [(y, min(xs)) for y, xs in buckets.items() if 1 < len(xs) < len(domain)]
                choices = [c for c in choices if c[1] != domain[0]] or choices
                if choices:
                    target, answer = random.choice(choices)
                    answer = " ".join(map(str, answer)) if isinstance(answer, tuple) else str(answer)
                    return Entry(
                        edict(code=code, mode=mode, goal=goal, call_text=call_text,
                              answer_hint=answer_hint, target=target),
                        answer,
                    )
        raise RuntimeError("failed to generate code input deduction task from LLM material")

    task.generate_entry = types.MethodType(llm_generate_entry, task)

    def inject(core1_text, core2_text=None):
        state["core1"] = core1_text
        if core2_text is not None:
            state["core2"] = core2_text

    def fewshot():
        real_task = CodeInputDeduction(config=cfg)
        entry = real_task.generate_entry()
        return {"program": entry.metadata.code, "mode": entry.metadata.mode}

    return Rig(
        task=task,
        inject=inject,
        fewshot=fewshot,
        prompt_spec=dict(
            kind="int_chain_function",
            instructions=(
                "Write a small Python function `def f0(x: int) -> int:` (optionally with one or two "
                "int->int helper functions f1, f2 that f0 calls) implementing a genuine, meaningful "
                "computation on its input -- as if it were a real small numeric routine (e.g. a "
                "checksum-like fold, a simple encoding/hashing step, a bounded counter update, a "
                "digit-manipulation routine). It must be a genuine (non-constant, non-identity) "
                "function of its input. No imports, no randomness, no I/O, no recursion, no "
                "printing. Under 12 lines."
            ),
        ),
    )


# --------------------------------------------------------------------------
# type_inhabitation / code_repair -- shared synthetic function-toolkit pool
# --------------------------------------------------------------------------

_BAD_TOOLKIT_NAMES = {"print", "exec", "eval", "compile", "open", "input", "__import__"}


def _is_clean_type_str(t):
    from reasoning_core.tasks.code_reasoning import _is_clean_type
    return _is_clean_type(t)


def valid_toolkit_record(name, inputs, output):
    if not name.isidentifier() or name in _BAD_TOOLKIT_NAMES or name.startswith("_"):
        return False
    if not inputs or len(inputs) > 4:
        return False
    if not _is_clean_type_str(output):
        return False
    seen_params = set()
    for pname, ptype in inputs:
        if not str(pname).isidentifier() or pname in seen_params:
            return False
        seen_params.add(pname)
        if not _is_clean_type_str(ptype):
            return False
    return True


def inject_synthetic_functions(fake_db_path, records, extend=False):
    """records: iterable of {"name": str, "inputs": [[pname, ptype], ...], "output": str}.
    Populates reasoning_core.tasks.code_reasoning._func_cache[fake_db_path] directly, so
    _load_functions() (called unmodified inside TypeInhabitation/CodeRepair.generate_entry())
    sees a ready-made pool and never touches the real scraper.

    IMPORTANT: reasoning_core's _mask_names()/func_lookup are both plain dicts keyed by
    function NAME (not name+signature) -- they implicitly assume every name in a toolkit is
    unique. A real scraped functions.db rarely violates this, but an LLM inventing function
    names across many independent toolkit-building calls can easily reuse a generic name
    (e.g. "transform") with a DIFFERENT signature. If two same-named, different-signature
    functions both end up selected into one toolkit, name_map[name] silently collapses to
    whichever was processed last, and the rendered toolkit_text shows the SAME masked label
    (e.g. "f1") twice with two different signatures -- a corrupted, confusing prompt. So we
    enforce name-uniqueness here, at the pool level, rather than leave it to chance which
    functions _select_toolkit happens to draw into the same problem."""
    from reasoning_core.tasks.code_reasoning import _func_cache, FunctionRecord

    existing = list(_func_cache.get(fake_db_path, [])) if extend else []
    seen_names = {f.name for f in existing}
    seen = {(f.name, tuple(f.inputs), f.output) for f in existing}
    clean = list(existing)
    n_seen_input, n_rejected, n_name_collision = 0, 0, 0
    for r in records:
        n_seen_input += 1
        try:
            name = str(r["name"])
            inputs = [(str(p), str(t)) for p, t in r["inputs"]]
            output = str(r["output"])
        except Exception:
            n_rejected += 1
            continue
        if not valid_toolkit_record(name, inputs, output):
            n_rejected += 1
            continue
        key = (name, tuple(inputs), output)
        if key in seen:
            continue  # exact duplicate (same name, same signature) -- harmless, just skip
        if name in seen_names:
            # Same invented name already in the pool under a DIFFERENT signature --
            # would corrupt the f0/f1/... masking scheme. Reject it rather than risk it.
            n_name_collision += 1
            continue
        seen.add(key)
        seen_names.add(name)
        clean.append(FunctionRecord(name, inputs, output))
    _func_cache[fake_db_path] = clean
    return dict(pool_size=len(clean), seen_input=n_seen_input, rejected=n_rejected,
                name_collisions_rejected=n_name_collision)


def real_functions_sample(real_db_path, libraries=None, k=6):
    """A few real FunctionRecords, for few-shotting the toolkit-writing LLM prompt."""
    from reasoning_core.tasks.code_reasoning import _load_functions
    funcs = _load_functions(real_db_path, libraries)
    sample = random.sample(funcs, min(k, len(funcs))) if funcs else []
    return [{"name": f.name, "inputs": list(f.inputs), "output": f.output} for f in sample]


def rig_toolkit_task(task_name, fake_db_path, cfg=None, extra_cfg_kwargs=None):
    """task_name: 'type_inhabitation' or 'code_repair'. Assumes _func_cache[fake_db_path]
    has already been populated via inject_synthetic_functions().

    IMPORTANT: db_path must be passed into the Config constructor, not set as a
    post-construction attribute. reasoning_core's Config.set_level() (called
    every time generate_example(level=...) runs) resets self.__dict__ back to
    the snapshot captured in __post_init__ at construction time -- a
    post-construction `cfg.db_path = ...` assignment would silently get wiped
    on the very first set_level() call, falling back to the class default
    ("functions.db") and pulling in the real scraped DB instead of our
    synthetic pool.
    """
    from reasoning_core.tasks.code_reasoning import (
        TypeInhabitation, TypeInhabitationCfg, CodeRepair, CodeRepairCfg,
    )
    extra_cfg_kwargs = extra_cfg_kwargs or {}
    if cfg is not None and cfg.db_path != fake_db_path:
        raise ValueError("cfg.db_path must already equal fake_db_path -- construct it with "
                          "TypeInhabitationCfg(db_path=fake_db_path, ...) / CodeRepairCfg(...)")
    if task_name == "type_inhabitation":
        cfg = cfg or TypeInhabitationCfg(db_path=fake_db_path, **extra_cfg_kwargs)
        task = TypeInhabitation(config=cfg)
    elif task_name == "code_repair":
        cfg = cfg or CodeRepairCfg(db_path=fake_db_path, **extra_cfg_kwargs)
        task = CodeRepair(config=cfg)
    else:
        raise ValueError(task_name)
    return task