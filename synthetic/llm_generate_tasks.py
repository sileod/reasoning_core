#!/usr/bin/env python3
"""
llm_generate_tasks.py

Generate LLM-authored examples for ONE reasoning-core task at a time, writing
them into the exact same TaskRow Parquet cache format that the procedural
pipeline uses -- so `task_influence.py --taskrow-cache <dir>` runs unmodified
against either arm for a direct, apples-to-apples comparison.

Design (see handoff doc, section 3-5): for each task we swap out ONLY the
one seam that normally calls gramforge or scrapes real libraries with
LLM-authored raw material, then let the task's own real, unmodified pipeline
(sandboxed execution, retry/acceptance checks, Entry construction) do
everything else. See scripts/llm_raw_material.py for the per-task rigs.

Run ONE task at a time so cost is easy to watch live:

    python scripts/llm_generate_tasks.py --task code_execution --levels 0 1 2 --n-per-level 40
    python scripts/llm_generate_tasks.py --task code_runnability --levels 0 --n-per-level 100
    python scripts/llm_generate_tasks.py --task code_analysis --n-per-level 50
    python scripts/llm_generate_tasks.py --task code_iterations --n-per-level 50
    python scripts/llm_generate_tasks.py --task code_input_deduction --n-per-level 50
    python scripts/llm_generate_tasks.py --task type_inhabitation --toolkit-calls 24
    python scripts/llm_generate_tasks.py --task code_repair --toolkit-calls 0   # reuse the pool above

Requires an LLM provider key in the environment for whichever model you pick
(e.g. OPENROUTER_API_KEY for the default openrouter/deepseek/deepseek-v4-flash).
Cost is tracked live via litlm.cost_breakdown("session") and printed every
--cost-print-every accepted rows; --call-ceiling is a hard stop independent
of that.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llm_raw_material as lrm  # noqa: E402
import reasoning_core as _rc  # noqa: E402
from reasoning_core.template import TimeoutException  # noqa: E402
print(f"(using reasoning_core from: {_rc.__file__})")

TASKS = [
    "code_execution", "code_runnability", "code_analysis", "code_iterations",
    "code_input_deduction", "type_inhabitation", "code_repair",
]
TOOLKIT_TASKS = {"type_inhabitation", "code_repair"}
REAL_TASK_NAME = {  # get_task() key, where it differs from the public name
    "code_iterations": "temporal_reasoning",
}


# --------------------------------------------------------------------------
# LLM plumbing
# --------------------------------------------------------------------------

class Budget:
    def __init__(self, ceiling, print_every):
        self.ceiling = ceiling
        self.print_every = print_every
        self.calls = 0
        self.accepted_rows = 0

    def spend(self):
        import litlm
        return sum(litlm.cost_breakdown("session").values())

    def note_call(self):
        self.calls += 1
        if self.calls >= self.ceiling:
            raise SystemExit(
                f"\nHit --call-ceiling ({self.ceiling} calls). "
                f"Session spend so far: ${self.spend():.4f}. Re-run with --resume to continue."
            )

    def maybe_print(self, task_name):
        if self.accepted_rows and self.accepted_rows % self.print_every == 0:
            print(f"  [{task_name}] {self.accepted_rows} rows saved | {self.calls} LLM calls | "
                  f"${self.spend():.4f} session spend so far")


def make_progress_bar(total, desc):
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, unit="row")


def call_llm(prompt, model, max_tokens, temperature, budget, reasoning_effort="none", system=None, pbar=None):
    import litlm
    budget.note_call()
    write = pbar.write if pbar is not None else print
    t0 = time.time()
    res = litlm.complete(prompt, model=model, max_tokens=max_tokens, temperature=temperature,
                          reasoning_effort=reasoning_effort, system=system, json=False,
                          show_progress=False)
    dt = time.time() - t0
    if getattr(res, "failed", False):
        write(f"  call {budget.calls}: FAILED after {dt:.1f}s "
              f"({getattr(res, 'error', 'unknown error')})")
        return None, None
    text = str(res)
    if not text.strip():
        # The model spent its whole max_tokens budget on internal reasoning
        # (reasoning_content) and never got to emit the actual answer. This is
        # a real, distinct failure mode for reasoning-capable models -- NOT
        # the same as the model choosing not to produce a useful answer.
        reasoning_len = len(getattr(res, "reasoning_content", None) or "")
        write(f"  call {budget.calls}: EMPTY content after {dt:.1f}s ({reasoning_len} chars of "
              f"internal reasoning ate the whole --max-tokens budget) | ${budget.spend():.4f} so far")
        return None, None
    write(f"  call {budget.calls}: ok in {dt:.1f}s, {len(text)} chars back | "
          f"${budget.spend():.4f} session spend so far")
    return text, getattr(res, "call_id", None)


def extract_code(text):
    """Pull a fenced ```python ...``` block if present, else use the raw text."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip("\n")


# --------------------------------------------------------------------------
# Per-task-kind prompt builders (kind comes from Rig.prompt_spec["kind"])
# --------------------------------------------------------------------------

def build_prompt(rig, level):
    kind = rig.prompt_spec["kind"]
    instructions = rig.prompt_spec["instructions"]
    level_hint = (
        f"\nTarget complexity: level {level} out of 4 (0=simplest, 4=most intricate); "
        f"scale the amount of logic accordingly (more helper functions / branches / "
        f"nesting at higher levels) while still satisfying every constraint above."
    )

    if kind in ("mesopy_program", "int_mutator_function", "int_chain_function"):
        # Deliberately NOT shown a CFG-derived example here. The procedural (mesopy)
        # generator's own output is arbitrary/meaningless by construction (random
        # variable names, nonsense operations) -- showing it as "match this structure"
        # would anchor the LLM into mimicking that same meaninglessness, which defeats
        # the point of comparing genuinely LLM-authored code against it. Instead the
        # interface constraints are just stated directly in `instructions`.
        extra = ""
        if rig.harvest_pair is not None:
            # code_runnability specifically: the probe set that will actually test the
            # code is FIXED size regardless of level (see endpoint_probes() -- only the
            # int range grows slightly with magnitude; str/list pools never change). If
            # the model scales up code complexity with level (the normal level_hint),
            # the deliberate bug gets proportionally harder for that fixed small probe
            # set to actually discover -- this was observed directly: near-100% miss
            # rate at level 2 vs a much better hit rate at level 0. So: give the EXACT
            # values that will be used, and don't scale complexity for this task.
            magnitude = max(3, int(getattr(rig.task.config, "magnitude", 3)))
            int_vals = list(range(-magnitude, magnitude + 1))
            extra = (
                f"\n\nThe EXACT input values your code will actually be tested against, per "
                f"parameter type (nothing outside this list will ever be used):\n"
                f"  int params: {int_vals}\n"
                f"  str params: ['', 'a', 'ab', 'xyz', '0']\n"
                f"  list params: [[], [0], [1], [-1, 1], [0, 2, -2]]\n"
                f"Your deliberate bug MUST be triggered by at least one of these EXACT values -- "
                f"not just 'small inputs' in general. The empty list `[]`, empty string `''`, and "
                f"`0` are the most reliable choices (e.g. indexing position 0 or 1 of a str/list "
                f"parameter fails immediately on `''`/`[]`; a modulo/divide fails on `0`)."
            )
            level_hint = (
                f"\nTarget level {level} out of 4 -- for THIS task, keep the code itself simple "
                f"and roughly the same size regardless of level (extra complexity makes the bug "
                f"harder for the fixed test values above to actually reach, which defeats the "
                f"point). Let any extra 'difficulty' come from a slightly less obvious bug "
                f"category instead, not from more lines of code."
            )
        return (
            f"{instructions}{level_hint}{extra}\n\n"
            "Return ONLY the Python code (a ```python fenced block is fine), no explanation."
        )

    if kind == "step_function_body":
        vars_ = rig.task._llm_pick_vars()
        # Render each domain value the same way it will ACTUALLY appear at runtime
        # (bare True/False/digits, quoted otherwise) -- not Python's default list
        # repr, which always quotes strings and would show e.g. an int-valued
        # variable's domain as ['0', '1'], misleading the model into writing
        # `count == '0'` (a string comparison that's always False against the
        # real runtime int) instead of the correct `count == 0`.
        vars_desc = "\n".join(
            f"  {name}: one of [{', '.join(lrm._literal_for_domain(v) for v in domain)}]"
            for name, domain in vars_
        )
        return (
            f"{instructions}{level_hint}\n\n"
            f"The exact variables you must use (these names and value sets are fixed -- "
            f"do not rename them or invent values outside these domains; note some domains "
            f"below are shown WITHOUT quotes, meaning they are real numbers/booleans at "
            f"runtime -- compare against them accordingly, e.g. `count == 0` not "
            f"`count == '0'`, and `flag` / `not flag` not `flag == 'True'`):\n{vars_desc}\n\n"
            "Return ONLY the step() body code, no explanation, no ```fences, no `def step():` line."
        ), vars_

    raise ValueError(f"unhandled prompt kind: {kind}")


def build_toolkit_prompt(real_sample, n_new, level_hint=""):
    example = "\n".join(
        f"  {{\"name\": \"{f['name']}\", \"inputs\": {json.dumps(f['inputs'])}, \"output\": \"{f['output']}\"}}"
        for f in real_sample
    )
    return (
        "Here are a few real typed function signatures, scraped from installed Python libraries, "
        "as a STRUCTURAL example (format only -- do NOT reuse these names/types):\n\n"
        f"--- EXAMPLE FUNCTIONS ---\n[\n{example}\n]\n--- END EXAMPLE ---\n\n"
        f"Invent {n_new} NEW, original synthetic function signatures in the same style: realistic-"
        "looking (but fictional) function names, each taking 1-4 typed keyword parameters and "
        "returning a typed value. Use plain bare type names (e.g. `int`, `str`, `list`, `float`, "
        "`bool`, or invented class-like names such as `Dataset`, `Model`, `Summary`) -- never "
        "`None`, `Any`, or `object`. Make some functions PRODUCE types that other functions in your "
        "set CONSUME as input (so they can be chained/composed), while also using plain primitive "
        "types (str, int, float, bool, list) as inputs somewhere, since those won't have a producer "
        "and will become the toolkit's leaf variables.\n\n"
        "Return ONLY a JSON array of objects, each with exactly the keys \"name\", \"inputs\" "
        "(a list of [param_name, type] pairs), and \"output\" (a type string). No explanation, "
        "no markdown fences."
    )


# --------------------------------------------------------------------------
# TaskRow writing (reuses task_diagnostics.cache/schemas verbatim)
# --------------------------------------------------------------------------

def make_row_builder(task, public_name):
    import task_diagnostics.cache as tdc
    from task_diagnostics.schemas import TaskRow, canonical_json

    # TaskRow.task must be the REAL reasoning_core-registered name (e.g. "temporal_reasoning"),
    # not our friendly CLI alias ("code_iterations") -- task_influence.py's --tasks argument
    # and the GPU trainer both filter TaskRow rows by exact match against the real registered
    # name (see load_task_rows: df[df["task"].isin(tasks)]). Using the alias here would make
    # this task's rows silently invisible to --tasks temporal_reasoning. The JSONL sidecar and
    # everything CLI-facing still uses public_name for our own consistency.
    real_name = REAL_TASK_NAME.get(public_name, public_name)

    def to_row(problem, call_id, model):
        problem.metadata = problem.metadata  # already an edict
        problem.metadata["_llm_call_id"] = call_id
        problem.metadata["_llm_model"] = model
        problem.metadata["_llm_source"] = "deepseek-v4-flash" if "deepseek" in model else model
        d = problem.to_dict()
        md = canonical_json(d.get("metadata", "{}"))
        try:
            alvl = int(json.loads(md).get("_level", 0))
        except Exception:
            alvl = 0
        prompt, answer = d.get("prompt", ""), str(d.get("answer", ""))
        meta = tdc._metadata_dict({"metadata": md})
        return TaskRow(
            task=real_name, level=alvl, prompt=prompt, answer=answer, metadata=md, mode="instruct",
            task_version=str(getattr(task, "task_version", getattr(task, "version", "0"))),
            behavior_hash=tdc._behavior_hash(task),
            config=canonical_json(task.config.to_dict() if hasattr(task.config, "to_dict") else {}),
            prompt_tokens=int(meta.get("_prompt_tokens", -1)),
            answer_tokens=int(meta.get("_answer_tokens", -1)),
            gen_time_s=round(float(meta.get("_time", -1)), 5),
            row_hash=TaskRow.compute_hash(real_name, alvl, prompt, answer, md),
        )

    return to_row


def write_cache(rows, public_name, task, out_root, mode="instruct"):
    import task_diagnostics.cache as tdc
    from task_diagnostics.schemas import CacheManifest, canonical_json

    real_name = REAL_TASK_NAME.get(public_name, public_name)
    cfg = getattr(task, "config", None)
    config_json = canonical_json(cfg.to_dict() if hasattr(cfg, "to_dict") else {})
    bh = tdc._behavior_hash(task)
    tv = str(getattr(task, "task_version", getattr(task, "version", "0")))
    levels = tuple(sorted({r.level for r in rows}))
    n_per_task = max(1, len(rows) // max(1, len(levels)))
    gen_ver = "llm-1"
    cid = tdc._cache_id([real_name], levels, n_per_task, mode, gen_ver,
                         {real_name: bh}, {real_name: tv}, {real_name: config_json})
    source = tdc._task_source(task)
    manifest = CacheManifest(
        cache_id=cid, source="fresh", tasks=(real_name,), levels=levels, n_per_task=n_per_task,
        mode=mode, generator_version=gen_ver, behavior_hashes={real_name: bh},
        task_versions={real_name: tv}, configs={real_name: config_json},
        tokenizer="HuggingFaceTB/SmolLM2-135M", generator_commit=tdc._generator_commit(),
        sources={real_name: source}, source_hashes={real_name: tdc._source_hash(source)},
    )
    return tdc._write_cache(rows, manifest, out_dir=out_root, analyze=True)


def taskrow_from_jsonl_row(row):
    """Rebuild a real TaskRow purely from a JSONL row's own stored metadata --
    generate_example() already captured everything a TaskRow needs into
    metadata at generation time (_task, _level, _task_version,
    _task_behavior_hash, _config, _prompt_tokens, _answer_tokens, _time), so
    no live reasoning_core task object needs to be reconstructed. This is
    what makes both --resume and cache recovery from JSONL alone possible."""
    from task_diagnostics.schemas import TaskRow, canonical_json

    meta = row.get("metadata") or {}
    real_name = meta.get("_task") or row.get("task")
    level = int(meta.get("_level", row.get("level", 0)))
    prompt, answer = row.get("prompt", ""), str(row.get("answer", ""))
    md = canonical_json(meta)
    config = meta.get("_config", "{}")
    config = config if isinstance(config, str) else canonical_json(config)
    return TaskRow(
        task=real_name, level=level, prompt=prompt, answer=answer, metadata=md, mode="instruct",
        task_version=str(meta.get("_task_version", "0")),
        behavior_hash=meta.get("_task_behavior_hash", "?"),
        config=config,
        prompt_tokens=int(meta.get("_prompt_tokens") or -1),
        answer_tokens=int(meta.get("_answer_tokens") or -1),
        gen_time_s=round(float(meta.get("_time") or -1), 5),
        row_hash=TaskRow.compute_hash(real_name, level, prompt, answer, md),
    )


def load_existing_rows(jsonl_path):
    """Returns (list_of_TaskRow, {level: count}) from an existing JSONL sidecar,
    or ([], {}) if it doesn't exist yet. Malformed lines are skipped defensively
    rather than aborting the whole load."""
    if not jsonl_path.exists():
        return [], {}
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(taskrow_from_jsonl_row(json.loads(line)))
            except Exception:
                continue
    return rows, Counter(r.level for r in rows)


def append_jsonl(rows_payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for payload in rows_payload:
            f.write(json.dumps(payload) + "\n")


# --------------------------------------------------------------------------
# Per-task generation loops
# --------------------------------------------------------------------------

def gen_simple_task(rig, public_name, levels, n_per_level, model, max_tokens, temperature,
                     budget, to_row, jsonl_path, max_material_retries, reasoning_effort="none"):
    """code_execution / code_iterations / code_input_deduction: one fresh raw
    material candidate per attempt; generate_example()'s own internal retry
    loop reuses it (varying only random call args / x0) for free."""
    rows, existing_counts = load_existing_rows(jsonl_path)
    if rows:
        print(f"  [{public_name}] resuming: {len(rows)} existing rows found in {jsonl_path} "
              f"({dict(sorted(existing_counts.items()))})")
    for level in levels:
        rig.task.config.set_level(level)
        have = existing_counts.get(level, 0)
        if have >= n_per_level:
            print(f"  [{public_name}] L{level}: already have {have}/{n_per_level}, skipping")
            continue
        stale_material_tries = 0
        pbar = make_progress_bar(n_per_level, f"{public_name} L{level}")
        if pbar is not None:
            pbar.update(have)
        try:
            while have < n_per_level:
                prompt = build_prompt(rig, level)
                text, call_id = call_llm(prompt, model, max_tokens, temperature, budget, reasoning_effort=reasoning_effort, pbar=pbar)
                if text is None:
                    continue
                code = extract_code(text)
                rig.inject(code)
                try:
                    problem = rig.task.generate_example(level=level)
                except (Exception, TimeoutException):
                    stale_material_tries += 1
                    if stale_material_tries > max_material_retries:
                        (pbar.write if pbar else print)(
                            f"  [{public_name}] giving up on this material after "
                            f"{max_material_retries} failed attempts, trying fresh material")
                        stale_material_tries = 0
                    continue
                stale_material_tries = 0
                row = to_row(problem, call_id, model)
                rows.append(row)
                have += 1
                budget.accepted_rows += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix_str(f"calls={budget.calls} ${budget.spend():.4f}")
                budget.maybe_print(public_name)
                append_jsonl([dict(task=public_name, level=row.level, prompt=row.prompt,
                                    answer=row.answer, metadata=json.loads(row.metadata),
                                    call_id=call_id, source=model)], jsonl_path)
        finally:
            if pbar is not None:
                pbar.close()
    return rows


def gen_code_analysis(rig, public_name, levels, n_per_level, model, max_tokens, temperature,
                       budget, to_row, jsonl_path, max_material_retries, reasoning_effort="none"):
    rows, existing_counts = load_existing_rows(jsonl_path)
    if rows:
        print(f"  [{public_name}] resuming: {len(rows)} existing rows found in {jsonl_path} "
              f"({dict(sorted(existing_counts.items()))})")
    for level in levels:
        rig.task.config.set_level(level)
        have = existing_counts.get(level, 0)
        if have >= n_per_level:
            print(f"  [{public_name}] L{level}: already have {have}/{n_per_level}, skipping")
            continue
        stale_material_tries = 0
        pbar = make_progress_bar(n_per_level, f"{public_name} L{level}")
        if pbar is not None:
            pbar.update(have)
        try:
            while have < n_per_level:
                result = build_prompt(rig, level)
                prompt, vars_ = result
                text, call_id = call_llm(prompt, model, max_tokens, temperature, budget, reasoning_effort=reasoning_effort, pbar=pbar)
                if text is None:
                    continue
                body = extract_code(text)
                rig.inject(body, vars_)
                try:
                    problem = rig.task.generate_example(level=level)
                except (Exception, TimeoutException):
                    stale_material_tries += 1
                    if stale_material_tries > max_material_retries:
                        stale_material_tries = 0
                    continue
                stale_material_tries = 0
                row = to_row(problem, call_id, model)
                rows.append(row)
                have += 1
                budget.accepted_rows += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix_str(f"calls={budget.calls} ${budget.spend():.4f}")
                budget.maybe_print(public_name)
                append_jsonl([dict(task=public_name, level=row.level, prompt=row.prompt,
                                    answer=row.answer, metadata=json.loads(row.metadata),
                                    call_id=call_id, source=model)], jsonl_path)
        finally:
            if pbar is not None:
                pbar.close()
    return rows


def gen_code_runnability(rig, public_name, levels, n_per_level, model, max_tokens, temperature,
                          budget, to_row, jsonl_path, max_material_retries, reasoning_effort="none"):
    rows, existing_counts = load_existing_rows(jsonl_path)
    if rows:
        print(f"  [{public_name}] resuming: {len(rows)} existing rows found in {jsonl_path} "
              f"({dict(sorted(existing_counts.items()))})")
    for level in levels:
        rig.task.config.set_level(level)
        have = existing_counts.get(level, 0)
        if have >= n_per_level:
            print(f"  [{public_name}] L{level}: already have {have}/{n_per_level}, skipping")
            continue
        pbar = make_progress_bar(n_per_level, f"{public_name} L{level}")
        if pbar is not None:
            pbar.update(have)
        try:
            while have < n_per_level:
                prompt = build_prompt(rig, level)
                text, call_id = call_llm(prompt, model, max_tokens, temperature, budget, reasoning_effort=reasoning_effort, pbar=pbar)
                if text is None:
                    continue
                code = extract_code(text)
                try:
                    kwargs_list = rig.harvest_pair(rig.task, code)
                except (Exception, TimeoutException):
                    kwargs_list = []
                if not kwargs_list:
                    (pbar.write if pbar else print)(
                        f"  [{public_name}] that program didn't yield a mixed OK/error pair, "
                        f"trying a fresh one")
                    continue
                for kw in kwargs_list:
                    if have >= n_per_level:
                        break
                    try:
                        problem = rig.task.generate_example(level=level, **kw)
                    except (Exception, TimeoutException):
                        continue
                    row = to_row(problem, call_id, model)
                    rows.append(row)
                    have += 1
                    budget.accepted_rows += 1
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix_str(f"calls={budget.calls} ${budget.spend():.4f}")
                    budget.maybe_print(public_name)
                    append_jsonl([dict(task=public_name, level=row.level, prompt=row.prompt,
                                        answer=row.answer, metadata=json.loads(row.metadata),
                                        call_id=call_id, source=model)], jsonl_path)
        finally:
            if pbar is not None:
                pbar.close()
    return rows


def build_toolkit_pool(task_kind, fake_db_path, real_db_path, libraries, target_size, max_new_calls,
                        model, max_tokens, temperature, budget, pool_file, reasoning_effort="none"):
    import litlm

    existing_records = []
    if pool_file.exists():
        existing_records = json.loads(pool_file.read_text())

    current = lrm.inject_synthetic_functions(fake_db_path, existing_records, extend=False)
    print(f"  pool file has {len(existing_records)} raw candidates -> {current['pool_size']} clean "
          f"functions right now (target: {target_size})")
    if current["pool_size"] >= target_size:
        print(f"  already at/above target size -- skipping toolkit-building calls entirely "
              f"(0 new LLM calls). Raise --toolkit-target-size if you want more diversity.")
        return current

    all_records = list(existing_records)
    calls_made = 0
    pbar = make_progress_bar(max_new_calls, "building synthetic toolkit pool")
    try:
        while calls_made < max_new_calls:
            stats = lrm.inject_synthetic_functions(fake_db_path, all_records, extend=False)
            if stats["pool_size"] >= target_size:
                (pbar.write if pbar else print)(
                    f"  reached target size ({stats['pool_size']}/{target_size}) after "
                    f"{calls_made} calls -- stopping early")
                break
            sample = lrm.real_functions_sample(real_db_path, libraries, k=6)
            prompt = build_toolkit_prompt(sample, n_new=10)
            text, call_id = call_llm(prompt, model, max_tokens, temperature, budget,
                                      reasoning_effort=reasoning_effort, pbar=pbar)
            calls_made += 1
            if pbar is not None:
                pbar.update(1)
            if text is None:
                continue
            try:
                batch = litlm.extract_json(text, default=[])
            except Exception:
                batch = []
            if isinstance(batch, dict):
                batch = batch.get("functions") or list(batch.values())
            all_records.extend(batch)
            (pbar.write if pbar else print)(
                f"  toolkit call {calls_made}/{max_new_calls}: +{len(batch)} candidate functions")
            if pbar is not None:
                pbar.set_postfix_str(f"${budget.spend():.4f}")
    finally:
        if pbar is not None:
            pbar.close()

    pool_file.parent.mkdir(parents=True, exist_ok=True)
    pool_file.write_text(json.dumps(all_records, indent=2))
    stats = lrm.inject_synthetic_functions(fake_db_path, all_records, extend=False)
    print(f"  synthetic pool: {stats['pool_size']} accepted / {stats['seen_input']} candidates "
          f"({stats['rejected']} malformed, {stats['name_collisions_rejected']} rejected for "
          f"reusing an existing name with a different signature) -> {pool_file}  "
          f"({calls_made} new LLM calls)")
    return stats


def gen_toolkit_task(task, public_name, levels, n_per_level, budget, to_row, jsonl_path):
    rows, existing_counts = load_existing_rows(jsonl_path)
    if rows:
        print(f"  [{public_name}] resuming: {len(rows)} existing rows found in {jsonl_path} "
              f"({dict(sorted(existing_counts.items()))})")
    for level in levels:
        task.config.set_level(level)
        have = existing_counts.get(level, 0)
        if have >= n_per_level:
            print(f"  [{public_name}] L{level}: already have {have}/{n_per_level}, skipping")
            continue
        attempts = 0
        pbar = make_progress_bar(n_per_level, f"{public_name} L{level} (free, from pool)")
        if pbar is not None:
            pbar.update(have)
        try:
            while have < n_per_level and attempts < n_per_level * 20:
                attempts += 1
                try:
                    problem = task.generate_example(level=level)
                except (Exception, TimeoutException):
                    continue
                row = to_row(problem, call_id=None, model="synthetic-pool")
                rows.append(row)
                have += 1
                budget.accepted_rows += 1
                if pbar is not None:
                    pbar.update(1)
                budget.maybe_print(public_name)
                append_jsonl([dict(task=public_name, level=row.level, prompt=row.prompt,
                                    answer=row.answer, metadata=json.loads(row.metadata),
                                    call_id=None, source="synthetic-pool")], jsonl_path)
        finally:
            if pbar is not None:
                pbar.close()
        if have < n_per_level:
            print(f"  [{public_name}] level {level}: only {have}/{n_per_level} rows found in "
                  f"{attempts} attempts -- pool may need more toolkit-building calls (--toolkit-calls).")
    return rows


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def print_preview(rows, n):
    if n <= 0 or not rows:
        return
    import random as _random
    sample = _random.sample(rows, min(n, len(rows)))
    print(f"\n{'=' * 78}\n{len(sample)} sample row(s) from this run, for a quick sanity read\n{'=' * 78}")
    for i, row in enumerate(sample, 1):
        print(f"\n--- sample {i}/{len(sample)}  (level {row.level}) ---")
        print("PROMPT:")
        print(row.prompt)
        print("ANSWER:")
        print(row.answer)
    print(f"\n{'=' * 78}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-per-level", type=int, default=200,
                     help="Rows to collect per level (default 200 x 5 levels = 1000 total, "
                          "matching the handoff's target volume).")
    ap.add_argument("--model", default="openrouter/deepseek/deepseek-v4-flash")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--reasoning-effort", default="none", choices=["none", "low", "medium", "high"],
                     help="Passed to litlm.complete(). Reasoning-capable models (like DeepSeek V4) "
                          "spend part of --max-tokens on an internal reasoning trace before writing "
                          "the actual answer; if that trace runs long it can eat the whole budget and "
                          "leave an EMPTY response. Default 'none' skips/minimizes that for this task, "
                          "since we just need a short code snippet, not a reasoning trace.")
    ap.add_argument("--call-ceiling", type=int, default=3000)
    ap.add_argument("--cost-print-every", type=int, default=10)
    ap.add_argument("--max-material-retries", type=int, default=3,
                     help="How many times generate_example() may internally fail on ONE piece of "
                          "LLM material (varying random args) before we fetch fresh material.")
    ap.add_argument("--out-root", default="task_diagnostics/cache/task_rows")
    ap.add_argument("--jsonl-root", default="generated_data/llm_tasks")
    ap.add_argument("--fresh", action="store_true",
                     help="Ignore any existing rows in this task's JSONL sidecar and start "
                          "completely from scratch (the old file is renamed aside with a "
                          "timestamp, not deleted). Default behavior is to RESUME: count what's "
                          "already there per level and only generate the shortfall -- use this if "
                          "you've changed prompts/config and don't want old and new rows mixed "
                          "together in one dataset.")
    ap.add_argument("--preview", type=int, default=3,
                     help="Print this many full sample rows (prompt + answer) from THIS run at the "
                          "end, so you can eyeball quality before scaling up. 0 to disable.")
    # toolkit-task-only options
    ap.add_argument("--toolkit-calls", type=int, default=None,
                     help="[type_inhabitation/code_repair only] MAX new LLM calls this run may make "
                          "while growing the shared synthetic function pool toward "
                          "--toolkit-target-size -- a safety cap, not a forced count. Building stops "
                          "as soon as the target size is reached (or even makes ZERO new calls if "
                          "the pool is already there), so repeated small test runs don't keep "
                          "growing it unnecessarily. Default: 24 for type_inhabitation, 0 for "
                          "code_repair (reuse type_inhabitation's pool for free -- pass a number "
                          "explicitly to let code_repair grow it too).")
    ap.add_argument("--toolkit-target-size", type=int, default=80,
                     help="[type_inhabitation/code_repair only] Desired size of the CLEAN synthetic "
                          "function pool. Small values (e.g. 20-30) are plenty for a handful of test "
                          "rows; the default (80) comfortably supports a full 1000-row run via "
                          "resampling, similar to how a real scraped functions.db (~100-300 "
                          "functions) supports many more rows than its own size.")
    ap.add_argument("--toolkit-pool-file", default="generated_data/llm_tasks/synthetic_toolkit_pool.json")
    ap.add_argument("--functions-db", default="functions.db",
                     help="Real scraped function DB, used only to build the toolkit few-shot prompt.")
    ap.add_argument("--libraries", nargs="*", default=None)
    args = ap.parse_args()

    budget = Budget(args.call_ceiling, args.cost_print_every)
    public_name = args.task
    jsonl_path = Path(args.jsonl_root) / f"{public_name}.jsonl"
    if args.fresh and jsonl_path.exists():
        backup = jsonl_path.with_suffix(f".jsonl.bak.{int(time.time())}")
        jsonl_path.rename(backup)
        print(f"  --fresh: moved existing {jsonl_path.name} aside to {backup.name} "
              f"(nothing was deleted)")
    t0 = time.time()

    if public_name in TOOLKIT_TASKS:
        fake_db_path = "__llm_synthetic_pool__"
        pool_file = Path(args.toolkit_pool_file)
        toolkit_calls = args.toolkit_calls
        if toolkit_calls is None:
            toolkit_calls = 0 if public_name == "code_repair" else 24
            print(f"  (--toolkit-calls not given, defaulting to {toolkit_calls} for {public_name})")
        if toolkit_calls > 0:
            build_toolkit_pool(public_name, fake_db_path, args.functions_db, args.libraries,
                                args.toolkit_target_size, toolkit_calls, args.model, args.max_tokens,
                                args.temperature, budget, pool_file, reasoning_effort=args.reasoning_effort)
        else:
            if not pool_file.exists():
                raise SystemExit(f"--toolkit-calls 0 but {pool_file} doesn't exist yet -- run once "
                                  f"with --toolkit-calls > 0 first (for this task or its sibling).")
            records = json.loads(pool_file.read_text())
            stats = lrm.inject_synthetic_functions(fake_db_path, records, extend=False)
            print(f"  reused {stats['pool_size']} clean functions from {len(records)} raw candidates "
                  f"in {pool_file} ({stats['name_collisions_rejected']} name collisions filtered out, "
                  f"0 new LLM calls)")

        task = lrm.rig_toolkit_task(public_name, fake_db_path)
        to_row = make_row_builder(task, public_name)
        rows = gen_toolkit_task(task, public_name, args.levels, args.n_per_level, budget, to_row, jsonl_path)
    else:
        rig_fn = {
            "code_execution": lrm.rig_code_execution,
            "code_runnability": lrm.rig_code_runnability,
            "code_analysis": lrm.rig_code_analysis,
            "code_iterations": lrm.rig_code_iterations,
            "code_input_deduction": lrm.rig_code_input_deduction,
        }[public_name]
        rig = rig_fn()
        to_row = make_row_builder(rig.task, public_name)
        if public_name == "code_runnability":
            rows = gen_code_runnability(rig, public_name, args.levels, args.n_per_level, args.model,
                                         args.max_tokens, args.temperature, budget, to_row, jsonl_path,
                                         args.max_material_retries, reasoning_effort=args.reasoning_effort)
        elif public_name == "code_analysis":
            rows = gen_code_analysis(rig, public_name, args.levels, args.n_per_level, args.model,
                                      args.max_tokens, args.temperature, budget, to_row, jsonl_path,
                                      args.max_material_retries, reasoning_effort=args.reasoning_effort)
        else:
            rows = gen_simple_task(rig, public_name, args.levels, args.n_per_level, args.model,
                                    args.max_tokens, args.temperature, budget, to_row, jsonl_path,
                                    args.max_material_retries, reasoning_effort=args.reasoning_effort)
        task = rig.task

    if not rows:
        raise SystemExit("No rows were produced -- nothing to write.")

    manifest, out_dir = write_cache(rows, public_name, task, args.out_root)
    dt = time.time() - t0
    print(f"\n{public_name}: {len(rows)} rows -> {out_dir}  ({dt:.1f}s, {budget.calls} LLM calls, "
          f"${budget.spend():.4f} session spend)")
    print(f"cache_id={manifest.cache_id}")
    print(f"Run:  python task_diagnostics/task_influence.py --run-influence --taskrow-cache {out_dir}")
    print_preview(rows, args.preview)


if __name__ == "__main__":
    main()