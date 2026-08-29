"""Plan-driven, folder-scoped task-search workers."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import textwrap
import json
import os
from pathlib import Path, PurePosixPath
import pprint
import re
import shutil
import subprocess
import sys

import yaml

from ..source_store import SourceStore


@dataclass(frozen=True)
class Trial:
    trial_id: str
    instruction: str
    owned_path: str
    validation: tuple
    hypothesis: str = ""
    parent: str = ""
    idea: str = ""
    changes: str = ""


@dataclass(frozen=True)
class SearchPlan:
    name: str
    base_ref: str
    context_files: tuple
    trials: tuple
    queues: dict
    path: Path
    # Hashed at load, not at record time: editing the plan mid-wave would otherwise
    # stamp later trials with a hash of a file that is not the one they ran under.
    sha256: str = ""


def _relative_path(value, field):
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{field} must be a repository-relative path: {value}")
    return path.as_posix().rstrip("/")


def load_plan(path):
    """Load and validate a task-search YAML plan."""
    path = Path(path).resolve()
    plan_bytes = path.read_bytes()
    data = yaml.safe_load(plan_bytes.decode("utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("task-search plans require version: 1")
    name = str(data.get("name", "")).strip()
    if not name:
        raise ValueError("plan name is required")
    defaults = data.get("defaults") or {}
    base_ref = str(defaults.get("base_ref", "HEAD"))
    contexts = tuple(
        _relative_path(value, "context file")
        for value in data.get("context_files", ())
    )
    trials = []
    for raw in data.get("trials", ()):
        if not isinstance(raw, dict):
            raise ValueError("each trial must be a mapping")
        trial_id = str(raw.get("id", "")).strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", trial_id):
            raise ValueError(f"invalid trial id: {trial_id!r}")
        instruction = str(raw.get("instruction", "")).strip()
        if not instruction:
            raise ValueError(f"{trial_id}: instruction is required")
        idea = str(raw.get("idea", "")).strip()
        changes = str(raw.get("changes", "")).strip()
        if not idea or not changes:
            raise ValueError(f"{trial_id}: idea and changes are required")
        owned_path = _relative_path(raw.get("owned_path", ""), f"{trial_id} owned_path")
        validation = tuple(str(command).strip() for command in raw.get("validation", ()))
        if not validation or any(not command for command in validation):
            raise ValueError(f"{trial_id}: at least one validation command is required")
        trials.append(Trial(
            trial_id=trial_id,
            instruction=instruction,
            owned_path=owned_path,
            validation=validation,
            hypothesis=str(raw.get("hypothesis", "")).strip(),
            parent=_relative_path(raw["parent"], f"{trial_id} parent") if raw.get("parent") else "",
            idea=idea,
            changes=changes,
        ))
    if not trials:
        raise ValueError("at least one trial is required")
    if len({trial.trial_id for trial in trials}) != len(trials):
        raise ValueError("trial IDs must be unique")
    owned = sorted((trial.owned_path, trial.trial_id) for trial in trials)
    for index, (path_a, id_a) in enumerate(owned):
        for path_b, id_b in owned[index + 1:]:
            if (path_a == path_b or path_b.startswith(path_a + "/")
                    or path_a.startswith(path_b + "/")):
                raise ValueError(f"owned paths overlap: {id_a} and {id_b}")
    known_ids = {trial.trial_id for trial in trials}
    queues = {}
    for queue, members in (data.get("queues") or {}).items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", str(queue)):
            raise ValueError(f"invalid queue name: {queue!r}")
        if (not isinstance(members, list) or not members
                or any(not isinstance(member, str) for member in members)):
            raise ValueError(f"{queue}: queue must be a non-empty ID list")
        if len(set(members)) != len(members):
            raise ValueError(f"{queue}: duplicate trial IDs")
        unknown = set(members) - known_ids
        if unknown:
            raise ValueError(
                f"{queue}: unknown trial IDs: {', '.join(sorted(unknown))}")
        queues[str(queue)] = tuple(members)
    return SearchPlan(name, base_ref, contexts, tuple(trials), queues, path,
                      hashlib.sha256(plan_bytes).hexdigest())


def _repo_root(start):
    output = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=start, text=True)
    return Path(output.strip()).resolve()


# How hard the worker is told to hurry. The step budget does not set this on its own:
# a worker told to explore spends the same budget differently from one told to start
# writing. It lives in a table because it is an assumption about where the bottleneck
# is rather than a fact -- every review of this system has questioned it -- and an
# assumption you can only test by A/B is one worth being able to set from the CLI.
# The wave's pace is recorded in generation metadata, so waves stay comparable.
PACE = {
    "hurry": {
        "stance": "That is enough only if you do not explore: the assignment and the"
                  " guides above already contain everything you need, and a working"
                  " task has been written from this prompt alone with no repository"
                  " reads at all. Hurry, and work in this order:",
        "first_step": "Start writing immediately -- one call to read the parent if you"
                      " have one, then a single call that writes the whole module",
    },
    "steady": {
        "stance": "Spend the first two or three calls understanding the assignment --"
                  " read the parent, and skim one neighbouring task for the house"
                  " style -- then commit to a design and write it. Work in this order:",
        "first_step": "Read the parent if you have one and at most one neighbouring"
                      " task, then write the whole module in a single call",
    },
    "deliberate": {
        "stance": "Design before you implement. Write down two or three formulations of"
                  " this task, say for each what a lazy solver could exploit and what"
                  " makes level 5 harder than level 0, pick one and say why, and only"
                  " then write code. A wrong design polished for twenty steps scores"
                  " worse than a right one written in five. Work in this order:",
        "first_step": "State the formulation you chose and why in two sentences, then"
                      " write the whole module",
    },
}
DEFAULT_PACE = "hurry"


def _sha256(data):
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def render_prompt(plan, trial, repo_root, task_meta=None, pace=DEFAULT_PACE):
    """Compose stable global context with one compact assignment."""
    pacing = PACE[pace]
    sections = [
        f"# Task-search assignment {trial.trial_id}",
        "",
        "You are one worker in a reproducible task-search wave.",
        "Read the global context, then implement only this assignment.",
    ]
    for relative in plan.context_files:
        source = repo_root / relative
        sections.extend(("", f"## Global context: `{relative}`", "", source.read_text().rstrip()))
    sections.extend((
        "",
        "## Assignment",
        "",
        trial.instruction,
        "",
        f"Hypothesis: `{trial.hypothesis or 'unassigned'}`",
        (f"Parent module: `{trial.parent}` -- read this one file with the read tool"
         " before you write, and reuse its machinery instead of importing a new"
         " library." if trial.parent else "Parent module: `none (new task)`"),
        f"Owned path: `{trial.owned_path}/`",
        "",
        "Design constraint, measured on this wave: the answer has to vary across",
        "examples. Yes/no answers and small fixed label sets lose the gameability gate",
        "below -- three tasks have died there already. When the question is naturally a",
        "decision, make the answer carry the witness: the value, the cut, the",
        "counterexample, the derivation.",
        "",
        "You may read the whole repository but may write only under the owned path,",
        "which needs no `__init__.py`. Do not commit, push, or move the assignment.",
        "Read files with the read, glob and grep tools: `cat`, `head`, `sed` and `wc`",
        "are denied. bash allows only python, git status/diff, ls, pwd, cd, mkdir and",
        "the self-check below. One command per call, and nothing after it: `;`, `|`,",
        "`&&` and `2>&1` split the line and each piece is checked on its own, so",
        "`... | tail -30`, `... 2>/dev/null` and `...; echo done` are all denied as a",
        "whole -- 149 of the 183 denied calls measured on this prompt were exactly",
        "that, and the self-check already prints only what you need. Type the allowed",
        "command literally: an env assignment inserted in the middle misses the",
        "allowance too. A denied call costs a step and re-sending it costs another --",
        "change tool instead. The owned path is created by your first write, so never",
        "`ls` or `mkdir` to check it.",
        *textwrap.wrap(
            f"You have {_budget_phrase(task_meta)}: one tool call is one step and a"
            f" denied call still counts. {pacing['stance']}", 79),
        *textwrap.wrap(
            f"1. {pacing['first_step']} under the owned path: a `Config` subclass, a"
            " `Task` subclass whose name does not contain `Task`, and the exact"
            " TASK_META below, pasted rather than retyped.",
            79, subsequent_indent="   "),
        f"2. In one more call write both a `test_<your_module>.py` next to the module --",
        "   pytest collects only files named `test_*.py` containing `test_*` functions --",
        f"   and `generate_samples_{trial.trial_id}.py`, seeded with{_seed_phrase(task_meta)}",
        "   so it is byte-reproducible.",
        "3. Then run the self-check. It is the whole harness in one command, it takes",
        "   about half a minute, and it is the only verification command you need:",
        f"   `{_selfcheck_command(trial)}`",
        "   It reports eleven gates -- implementation, discovery, task_meta, smoke,",
        "   speed, samples, sections, reproducible, pytest, contract, gameability -- and",
        "   PASS on all eleven is what the harness scores as a success, so fix what it",
        "   names and run it again. Do not verify any other way: a hand-written python -c costs the same",
        "   step and checks less. Run it early, while there is budget left to act on it;",
        "   trials are being lost to gates their author never saw.",
        f"4. Spend whatever steps remain widening the tests, re-running the self-check",
        "   after each change. Leave the last word to a run with eleven PASSes.",
        "",
        "Failure modes measured on one-shot attempts at this prompt, all caught by the",
        "self-check:",
        "- `Task` has no `self.rng`; seed the `random` module instead.",
        "- metadata must be JSON-serializable: cast numpy scalars with `int`/`float`.",
        f"- third-party imports must already be installed; {_available_libs()} are.",
        "- `score_scalar` parses its argument as a float, so it cannot score a yes/no,",
        "  a list or a symbolic answer; write the comparison your answer format needs.",
        "- `random.Random()` with no argument draws from the OS and makes the samples",
        "  irreproducible; call the module-level `random` functions instead, and do not",
        "  seed inside the task -- only the sample script seeds. A helper you call from",
        "  the parent module may carry its own generator that `random.seed` never",
        "  reaches, which is why the self-check compares bytes, not source.",
        "- `gramforge.generate` calls `random.seed(seed)` on entry and its `seed`",
        "  defaults to `None`, so every call silently reseeds the global RNG from the",
        "  OS. If you build on a grammar, pass `seed=random.randrange(2**32)`: drawn",
        "  from the seeded module RNG it stays reproducible, and it still differs",
        "  between examples.",
        "- `validate()` re-scores the gold answer, so `score_answer` must return 1.0",
        "  on it and must match the answer format your prompt asks for.",
        "- `score_answer` runs with a mock `self` that raises on any attribute access,",
        "  so it must not touch `self` at all: no `self._parse_interval(answer)`, no",
        "  `self.config`. Put shared parsing in a module-level function and call it",
        "  directly. S11 in wave1 lost an otherwise-passing task to one helper call.",
        "- generation must survive every level: enforce construction invariants by",
        "  resampling in a loop, never with an `assert` that only holds at level 0.",
    ))
    if task_meta is not None:
        sections.extend((
            "The task module must contain this exact module-level provenance mapping:",
            "",
            "```python",
            f"TASK_META = {pprint.pformat(task_meta, sort_dicts=False)}",
            "```",
            "",
        ))
    sections.extend((
        "Gates worth knowing before you write, because they are the ones trials lose on:",
        "",
        "- `gameability` scores the single most frequent of 30 answers against all of",
        "  them and fails the trial if that constant guess wins more than 0.4. It prints",
        "  the rate even when it passes, and the 70 shipped tasks sit at a median of",
        "  0.23, so read anything above 0.25 as an answer space too narrow to be worth",
        "  shipping rather than as a pass. Widen it in the generator -- more distinct",
        "  answers, spread over a wider range -- and never weaken score_answer.",
        "  The same gate then answers with the last number in the prompt, the first, the",
        "  largest and the last word, and fails the trial if any of them wins more than",
        "  0.4. A word problem that narrates its own total -- \"working together they",
        "  produce 12 pounds. What is the total number of pounds?\" -- scores 1.00 here",
        "  and has already lost a trial that passed every other gate. State the givens",
        "  and ask for something no given spells out. It also generates at levels 0, 3",
        "  and 6 and fails the trial if any of them cannot produce an example at all --",
        "  `speed` only times the default config, so a top level whose search never",
        "  terminates used to pass unseen. Bound the search at every level, not just",
        "  the one you tried.",
        "- `reproducible` runs the sample generator in five fresh processes, two at each",
        "  of two string-hash salts and one at a third, and compares the bytes. Three",
        "  faults break it and none of them shows up when you run the generator twice in",
        "  one process: state kept between calls; a set or dict of strings whose order",
        "  reaches the output, because Python salts string hashing per process; and a",
        "  dict or set keyed on objects, whose hash is the memory address and which no",
        "  salt pins at all. Iterate `sorted(...)` over anything whose order is visible,",
        "  and key on a string or a tuple of ints, never on an object.",
        f"- `sections` needs `samples_{trial.trial_id}.md` to carry two complete",
        "  prompt/answer examples at each of levels 0, 2 and 5. The headings `Level 0`,",
        "  `Level 2` and `Level 5` are matched literally and the word `Answer` is counted",
        "  under each: one example per level fails, however well the file reads.",
        "- `contract` generates 64 examples and requires score_answer to return 1.0 on",
        "  every gold answer and less than 1.0 on empty and junk strings.",
        "- `speed` times generate_example at the DEFAULT config, which is the one the",
        "  contract audit uses. The harness kills any validation command at 300 seconds,",
        "  so a generator averaging more than about 4 seconds an example loses the trial",
        "  whatever else it does. Bound every rejection-sampling loop; the cost is",
        "  heavy-tailed and one pathological instance is enough.",
        "",
        "Do not end your turn before the self-check has printed a line for every one of",
        "the eleven gates and none of them says FAIL. Three of fifteen trials in the last",
        "wave stopped of their own accord inside three minutes, having used twelve of",
        "twenty-eight steps and written the task module but neither the test, the sample",
        "script nor the samples file, and all three were scored as failures: a trial that",
        "stops early scores exactly as badly as one that runs out. Reasoning about whether",
        "the code is right is not a substitute for running the check -- it is one call and",
        "it answers the question.",
        "",
        "Then finish with a concise summary of changes and self-check results.",
        "",
    ))
    return "\n".join(sections)


def generation_metadata(model, harness_version, agent, variant=None,
                        requested_seed=None, seed_forwarded=False,
                        temperature=None, top_p=None, sandbox_name="bubblewrap",
                        sandbox_version=None, max_steps=56,
                        timeout_seconds=1800, provider_name=None,
                        adapter_name="direct", adapter_version=None,
                        harness_name="opencode"):
    settings = {
        "variant": variant,
        "requested_seed": requested_seed,
        "seed_forwarded": seed_forwarded,
        "temperature": temperature,
        "top_p": top_p,
        "pure": True,
        "max_steps": max_steps,
        "timeout_seconds": timeout_seconds,
        "sandbox": {
            "name": sandbox_name,
            "version": sandbox_version,
        },
    }
    return {
        "provider_name": provider_name or model.split("/", 1)[0],
        "model_name": model,
        "adapter_name": adapter_name,
        "adapter_version": adapter_version,
        "harness_name": harness_name,
        "harness_version": harness_version,
        "agent_name": agent,
        "settings": settings,
    }


def _module_prefix(trial):
    return trial.owned_path.replace("/", ".")


_CANDIDATE_LIBS = ("z3", "sympy", "networkx", "numpy", "scipy", "nltk", "lark",
                   "pyparsing", "regex", "automata", "pandas")


def _available_libs():
    import importlib.util
    return ", ".join(m for m in _CANDIDATE_LIBS if importlib.util.find_spec(m))


def _budget_phrase(task_meta):
    steps = ((task_meta or {}).get("generation", {})
             .get("settings", {}).get("max_steps"))
    return f"exactly {steps} steps" if steps else "a very small step budget"


def _seed_phrase(task_meta):
    seed = ((task_meta or {}).get("generation", {})
            .get("settings", {}).get("requested_seed"))
    return f" seed {seed}" if seed is not None else " the recorded requested_seed"


def _sample_command_for(owned_path, trial_id):
    # PYTHONPATH=. so running the script by path still imports the worktree.
    return (
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python "
        f"{owned_path}/generate_samples_{trial_id}.py"
    )


def _sample_command(trial):
    return _sample_command_for(trial.owned_path, trial.trial_id)


def _selfcheck_command_for(owned_path, trial_id):
    # One call that runs every gate, including the three -- TASK_META, the contract
    # audit, the sample headings -- that otherwise surface only in run.json, after the
    # trial is already lost. Measured at half a minute for all ten.
    return ("PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m"
            f" reasoning_core.task_search.selfcheck {owned_path} {trial_id}")


def _selfcheck_command(trial):
    return _selfcheck_command_for(trial.owned_path, trial.trial_id)


def _prior_audit_command(trial):
    # A task a single fixed answer wins is not measuring reasoning, however well
    # it validates; measured on the first six wave0 tasks, two of them lost.
    return (
        "PYTHONDONTWRITEBYTECODE=1 python -m reasoning_core.task_search.prior_audit"
        f" --path {trial.owned_path} --n 30 --max-const 0.4 --max-shortcut 0.4 --budget-seconds 45"
    )


def opencode_permissions(trial):
    bash = {
        "*": "deny",
        "git status*": "allow",
        "git diff*": "allow",
        "python -c *": "allow",
        "PYTHONDONTWRITEBYTECODE=1 python -c *": "allow",
        # The same call the two above already allow, with the env prefix workers
        # actually type. Matching is prefix-anchored, so an assignment in the middle
        # misses every pattern: 33 of 183 denials in the first six waves were this
        # spelling of a command that was already permitted.
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -c *": "allow",
        # Navigation only; none of these can read file contents, so the
        # *.env read deny still holds. Denying them wasted ~30% of turns.
        "ls*": "allow",
        "pwd*": "allow",
        "cd *": "allow",
        "mkdir *": "allow",
    }
    # Trailing "*" so added flags and pipes still match the allowed command.
    for command in (list(trial.validation) + [_sample_command(trial),
                                              _selfcheck_command(trial),
                                              _prior_audit_command(trial)]):
        bash[command] = "allow"
        bash[command + "*"] = "allow"
    permissions = {
        "read": {"*": "allow", "*.env": "deny", "*.env.*": "deny"},
        "glob": "allow",
        "grep": "allow",
        "list": "allow",
        # Path-scoped edit globs are unreliable in OpenCode 1.18.20. The
        # bubblewrap mount namespace is the write boundary; this permission
        # lets edit tools operate inside that boundary.
        "edit": "allow",
        "bash": bash,
        "task": "deny",
        "external_directory": "deny",
        "question": "deny",
        "webfetch": "deny",
        "websearch": "deny",
    }
    return permissions


def opencode_config(trial, agent, *, requested_seed=None, forward_seed=False,
                    temperature=None, top_p=None, max_steps=56):
    permissions = opencode_permissions(trial)
    agent_config = {
        "description": "Folder-scoped task-search worker",
        "mode": "primary",
        "steps": max_steps,
        "permission": permissions,
    }
    if forward_seed:
        agent_config["seed"] = requested_seed
    if temperature is not None:
        agent_config["temperature"] = temperature
    if top_p is not None:
        agent_config["top_p"] = top_p
    return {
        "$schema": "https://opencode.ai/config.json",
        "permission": permissions,
        "agent": {agent: agent_config},
    }


def _opencode_command(opencode_bin, *, model, agent, worktree, prompt,
                      variant=None):
    # OpenCode's --file option accepts an array and greedily treats a following
    # positional message as another filename. Pass the complete prompt as the
    # positional message and keep prompt.md only as the durable run artifact.
    command = [
        opencode_bin, "run", "--pure", "--model", model, "--agent", agent,
        "--format", "json", "--dir", str(worktree),
    ]
    if variant:
        command.extend(("--variant", variant))
    command.append(prompt)
    return command


def _mini_command(mini_bin, *, prompt, config_path, trajectory_path):
    return [
        mini_bin,
        "-c", "mini.yaml",
        "-c", str(config_path),
        "-t", prompt,
        "-y",
        "--exit-immediately",
        "-o", str(trajectory_path),
    ]


def _mini_config(worktree, *, max_steps, timeout_seconds, requested_seed=None,
                 forward_seed=False, temperature=None, top_p=None):
    model_kwargs = {"drop_params": True}
    if forward_seed:
        model_kwargs["seed"] = requested_seed
    if temperature is not None:
        model_kwargs["temperature"] = temperature
    if top_p is not None:
        model_kwargs["top_p"] = top_p
    return {
        "agent": {
            "step_limit": max_steps,
            "wall_time_limit_seconds": timeout_seconds,
            "cost_limit": 0,
        },
        "environment": {
            "cwd": str(worktree),
            "timeout": min(300, timeout_seconds),
        },
        "model": {
            "cost_tracking": "ignore_errors",
            "model_kwargs": model_kwargs,
        },
    }


def _adapter_command(command, *, adapter, provider=None, adapter_bin=None,
                     model=None, harness="opencode"):
    """Wrap a harness command with a provider adapter when requested."""
    if adapter == "direct":
        return command
    if adapter != "harness-link":
        raise ValueError(f"unsupported adapter: {adapter}")
    if not provider:
        raise ValueError("--provider is required with --adapter harness-link")
    executable = shutil.which(adapter_bin or provider)
    if executable is None:
        raise RuntimeError(
            f"Harness Link provider command not found: {adapter_bin or provider!r}")
    return [executable, harness, "--model", model, "--", *command[1:]]


def _resolve_harness_executable(harness, adapter, harness_bin):
    # Harness Link resolves its own harness from PATH and discards the
    # executable in the wrapped command. Probe exactly the binary it will use.
    if harness == "mini" and adapter == "direct":
        raise ValueError("the mini harness currently requires --adapter harness-link")
    requested = harness if adapter == "harness-link" else harness_bin
    executable = shutil.which(requested)
    if executable is None:
        context = " on Harness Link PATH" if adapter == "harness-link" else ""
        raise RuntimeError(
            f"{harness} executable not found{context}: {requested!r}")
    return executable


def _resolve_opencode_executable(adapter, opencode_bin):
    """Compatibility wrapper for callers that only select OpenCode."""
    return _resolve_harness_executable("opencode", adapter, opencode_bin)


def _harness_version(harness, executable):
    if harness == "opencode":
        return subprocess.check_output(
            [executable, "--version"], text=True).strip()
    if harness == "mini":
        python = Path(executable).resolve().parent / "python"
        return subprocess.check_output([
            str(python), "-c",
            "import importlib.metadata as m; print(m.version('mini-swe-agent'))",
        ], text=True).strip()
    raise ValueError(f"unsupported harness: {harness}")


def _changed_paths(worktree):
    raw = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=worktree,
    )
    chunks = raw.split(b"\0")
    paths, index = [], 0
    while index < len(chunks) and chunks[index]:
        chunk = chunks[index]
        status = chunk[:2].decode("ascii", "replace")
        paths.append(chunk[3:].decode("utf-8", "surrogateescape"))
        index += 1
        if "R" in status or "C" in status:
            if index < len(chunks) and chunks[index]:
                paths.append(chunks[index].decode("utf-8", "surrogateescape"))
                index += 1
    return sorted(set(paths))


def _outside_owned(paths, owned_path):
    owned = PurePosixPath(owned_path)
    return [
        path for path in paths
        if PurePosixPath(path) != owned and owned not in PurePosixPath(path).parents
    ]


def _task_metadata(worktree, owned_path):
    found = []
    for path in sorted((worktree / owned_path).rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "TASK_META"):
                found.append((path.relative_to(worktree).as_posix(), ast.literal_eval(node.value)))
    return found


# The sections a samples file has to show. Named once, because the self-check
# reports this gate to the worker and the two rules going out of step would tell
# the worker it had passed something the coordinator then failed it on.
SAMPLE_LEVELS = ("0", "2", "5")
SAMPLE_EXAMPLES = 2


def sample_shortfall(body):
    """Which required levels does the samples file not actually show twice?

    The gate used to be four substrings, which a file carrying the right headings and
    a single example passed just as happily as one carrying two. Measured over 480
    sample files, three did exactly that, and their authors were told they had passed.
    """
    body = body.lower()
    hits = [(m.start(), m.group(1)) for m in re.finditer(r"level\s*([025])\b", body)]
    counts = dict.fromkeys(SAMPLE_LEVELS, 0)
    for index, (position, level) in enumerate(hits):
        end = hits[index + 1][0] if index + 1 < len(hits) else len(body)
        counts[level] += body.count("answer", position, end)
    return [f"level {level} shows {counts[level]} of {SAMPLE_EXAMPLES} answers"
            for level in SAMPLE_LEVELS if counts[level] < SAMPLE_EXAMPLES]


def _step_usage(events_path, max_steps):
    """How much of the step budget the worker spent, and whether it ran out.

    A trial that runs out of steps mid-fix is filed under whichever artifact it had
    not written yet, so the status names the missing file and not the cause: M10 in
    wave 20260829T092634Z fixed its NameError on its 27th of 28 steps and was
    recorded as sample_review_failed. Measured over the 64 trials run at 28 steps,
    trials that finished under the ceiling succeed 0.63 of the time and trials that
    hit it 0.29, so this is the first number to look at on any failure.
    """
    try:
        lines = Path(events_path).read_text().splitlines()
    except FileNotFoundError:
        return None
    used = sum(1 for line in lines if '"step_start"' in line)
    return {"used": used, "max": max_steps,
            "exhausted": bool(max_steps) and used >= max_steps}


def _sample_review(worktree, owned_path, trial_id, events_path):
    sample_name = f"samples_{trial_id}.md"
    script_name = f"generate_samples_{trial_id}.py"
    sample_path = Path(worktree) / owned_path / sample_name
    script_path = Path(worktree) / owned_path / script_name
    result = {
        "path": f"{owned_path}/{sample_name}",
        "generator_path": f"{owned_path}/{script_name}",
        "generator_exists": script_path.is_file(),
        "exists": sample_path.is_file(),
        "size_bytes": sample_path.stat().st_size if sample_path.is_file() else 0,
        "required_sections": False,
        "read_after_last_edit": False,
        "command_succeeded": False,
    }
    if sample_path.is_file():
        shortfall = sample_shortfall(sample_path.read_text())
        result["required_sections"] = not shortfall
        result["sample_shortfall"] = shortfall
    target = "/" + result["path"]
    expected_command = _sample_command_for(owned_path, trial_id)
    last_write = -1
    last_read = -1
    try:
        lines = Path(events_path).read_text().splitlines()
    except FileNotFoundError:
        lines = []
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "tool_use":
            continue
        part = event.get("part", {})
        state = part.get("state", {})
        if state.get("status") != "completed":
            continue
        file_path = str(state.get("input", {}).get("filePath", ""))
        bash_command = state.get("input", {}).get("command", "")
        command_matches = any(
            segment.strip().startswith(expected_command)
            for segment in bash_command.split("&&")
        )
        if (part.get("tool") == "bash" and command_matches
                and state.get("metadata", {}).get("exit") == 0):
            result["command_succeeded"] = True
            last_write = index
        if not file_path.endswith(target):
            continue
        if part.get("tool") in {"write", "edit", "apply_patch"}:
            last_write = index
        elif part.get("tool") == "read":
            last_read = index
    result["read_after_last_edit"] = last_read > last_write
    # Event parsing is deliberately observational. The durable files and the
    # coordinator's independent deterministic replay are the cross-harness
    # correctness boundary.
    result["ok"] = (
        result["generator_exists"] and result["exists"]
        and result["size_bytes"] > 0 and result["required_sections"]
    )
    return result


def _plan_problems(plan, repo_root):
    """The checks `check` cannot make by reading the YAML alone.

    load_plan validates properties of the text. These need the checkout, and every one
    of them used to surface only at launch, after the worktrees had been made -- a plan
    could pass `check` cleanly and still have nowhere to run.
    """
    repo_root = Path(repo_root)
    problems = []
    def at_base(relative):
        return subprocess.run(["git", "cat-file", "-e", f"{plan.base_ref}:{relative}"],
                              cwd=repo_root, capture_output=True).returncode == 0
    if subprocess.run(["git", "rev-parse", "--verify", f"{plan.base_ref}^{{commit}}"],
                      cwd=repo_root, capture_output=True).returncode != 0:
        return [f"base_ref does not resolve to a commit: {plan.base_ref}"]
    for relative in plan.context_files:
        # render_prompt reads these from the live checkout, not from base_ref.
        if not (repo_root / relative).is_file():
            problems.append(f"context file missing from the checkout: {relative}")
    for trial in plan.trials:
        # _task_classes turns an owned module into an import path by taking it relative
        # to reasoning_core/tasks. Anywhere else and the contract audit imports nothing.
        if not trial.owned_path.startswith("reasoning_core/tasks/"):
            problems.append(f"{trial.trial_id}: owned_path is outside"
                            f" reasoning_core/tasks: {trial.owned_path}")
        if trial.parent and not at_base(trial.parent):
            problems.append(f"{trial.trial_id}: parent not at {plan.base_ref}: {trial.parent}")
    return problems


def _frozen_module_drift(repo_root, base_ref):
    """Do the modules a worker runs match the ones the coordinator enforces?

    Workers run inside a worktree checked out at base_ref, so every harness module they
    invoke is frozen at that commit while the coordinator runs whatever is in the working
    tree. Let those drift apart and the harness starts telling workers they passed
    something it is about to fail them on -- and the worker has no way to find out. Worse
    for prior_audit: the coordinator builds its command line live, so a flag added here
    is an argparse error inside every worktree pinned before it.
    """
    problems = []
    for relative in ("reasoning_core/task_search/selfcheck.py",
                     "reasoning_core/task_search/prior_audit.py"):
        live = (Path(repo_root) / relative).read_bytes()
        try:
            pinned = subprocess.check_output(["git", "show", f"{base_ref}:{relative}"],
                                             cwd=repo_root)
        except subprocess.CalledProcessError:
            problems.append(f"{base_ref} has no {relative}: workers cannot run it at all")
            continue
        if pinned != live:
            problems.append(f"{relative} at {base_ref} differs from the working tree")
    if problems:
        return ("workers are judged by code they cannot see. Move base_ref forward.\n  "
                + "\n  ".join(problems))
    return ""


def _owned_digest(worktree, owned_path, exclude=()):
    """sha256 per file under the owned path, plus one hash over the lot.

    The gates that certify a candidate -- TASK_META and the contract audit -- run
    before the model-authored sample generator and pytest suite, and those run with
    the owned directory writable, because the generator has to write into it. Without
    a freeze check a test that rewrote task.py after the audit passed would still be
    accepted, and the run record would carry no hash of what was accepted.
    """
    root = Path(worktree) / owned_path
    files = {}
    for parent, directories, names in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if name != "__pycache__"]
        # Directory symlinks are listed but never descended, so record them here or
        # a swapped-in link to a tree outside the worktree leaves no trace at all.
        for name in sorted(names + directories):
            path = Path(parent, name)
            relative = path.relative_to(root).as_posix()
            if relative in exclude:
                continue
            # Content alone is not the file: a task.py replaced by a symlink to an
            # identical file elsewhere, or one that only gained the executable bit,
            # moves nothing in a digest that hashes bytes and nothing else.
            mode = oct(path.lstat().st_mode)
            if path.is_symlink():
                files[relative] = ["link", mode, os.readlink(path)]
            elif path.is_file():
                files[relative] = ["file", mode,
                                   hashlib.sha256(path.read_bytes()).hexdigest()]
    return {"files": files, "tree_sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()).hexdigest()}


def _undiscoverable(classes):
    """Modules the contract audit imports happily that reasoning_core would never list.

    _discover_tasks skips a file whose name starts with "_", anything under a dotted or
    underscored directory, and anything under "deprecated". _task_classes below does
    not, so without this a task can pass every gate and still never reach DATASETS.
    """
    hidden = []
    for module, _ in classes:
        parts = module.split(".")[2:]
        if (parts[-1].startswith("_") or "deprecated" in parts
                or any(part.startswith("_") for part in parts[:-1])):
            hidden.append(module)
    return sorted(set(hidden))


def _task_classes(worktree, owned_path):
    """Return import paths for direct Task subclasses in the owned path."""
    worktree = Path(worktree)
    tasks_root = worktree / "reasoning_core" / "tasks"
    found = []
    for path in sorted((worktree / owned_path).rglob("*.py")):
        if path.name == "__init__.py" or path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {
                base.id if isinstance(base, ast.Name) else base.attr
                for base in node.bases
                if isinstance(base, (ast.Name, ast.Attribute))
            }
            if "Task" in bases:
                relative = path.relative_to(tasks_root).with_suffix("")
                module = "reasoning_core.tasks." + ".".join(relative.parts)
                found.append((module, node.name))
    return found


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _resource_command(command, resource_limits):
    if not resource_limits.get("enabled"):
        return command
    return [
        resource_limits["executable"],
        "--user", "--scope", "--quiet", "--collect",
        "-p", f"MemoryMax={resource_limits['memory_max']}",
        "-p", f"TasksMax={resource_limits['tasks_max']}",
        "-p", f"CPUQuota={resource_limits['cpu_quota']}",
        "--",
        *command,
    ]


def _resolve_resource_limits(mode, *, systemd_run_bin="systemd-run",
                             memory_max="8G", tasks_max=512,
                             cpu_quota="400%"):
    if mode == "none":
        return {"enabled": False, "mode": mode}
    executable = shutil.which(systemd_run_bin)
    true_executable = shutil.which("true")
    error = None
    if executable and true_executable:
        probe = subprocess.run(
            [
                executable, "--user", "--scope", "--quiet", "--collect",
                "-p", f"MemoryMax={memory_max}",
                "-p", f"TasksMax={tasks_max}",
                "-p", f"CPUQuota={cpu_quota}",
                "--", true_executable,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if probe.returncode == 0:
            version = subprocess.check_output(
                [executable, "--version"], text=True).splitlines()[0]
            return {
                "enabled": True,
                "mode": mode,
                "name": "systemd-run user scope",
                "version": version,
                "executable": executable,
                "memory_max": memory_max,
                "tasks_max": tasks_max,
                "cpu_quota": cpu_quota,
            }
        error = probe.stderr.strip() or f"exit code {probe.returncode}"
    elif not executable:
        error = f"command not found: {systemd_run_bin}"
    else:
        error = "command not found: true"
    if mode == "required":
        raise RuntimeError(f"resource limits unavailable: {error}")
    return {"enabled": False, "mode": mode, "reason": error}


def _public_resource_limits(resource_limits):
    fields = {
        "enabled", "mode", "name", "version", "memory_max", "tasks_max",
        "cpu_quota",
    }
    return {
        key: value for key, value in resource_limits.items()
        if key in fields
    }


def _sanitized_environment(credential_env_names=()):
    environment = dict(os.environ)
    for name in credential_env_names:
        environment.pop(name, None)
    return environment


def _sandbox_command(command, *, worktree, owned_path, runtime_root,
                     bwrap_bin="bwrap"):
    """Wrap a worker so only its owned repository directory is writable."""
    executable = shutil.which(bwrap_bin)
    if executable is None:
        raise RuntimeError(
            f"bubblewrap executable not found: {bwrap_bin!r}; "
            "strict task-search runs require bubblewrap"
        )
    worktree = Path(worktree).resolve()
    owned = (worktree / owned_path).resolve()
    if worktree not in owned.parents:
        raise ValueError(f"owned path escapes worktree: {owned_path}")
    runtime_root = Path(runtime_root).resolve()
    for path, label in ((worktree, "worktree"), (runtime_root, "runtime root")):
        if path == Path("/tmp") or Path("/tmp") in path.parents:
            raise ValueError(
                f"{label} cannot be under /tmp because strict runs hide host /tmp: {path}")
        if path == Path("/run") or Path("/run") in path.parents:
            raise ValueError(
                f"{label} cannot be under /run because strict runs hide host /run: {path}")
    runtime_dirs = {
        "XDG_DATA_HOME": runtime_root / "data",
        "XDG_CACHE_HOME": runtime_root / "cache",
        "XDG_STATE_HOME": runtime_root / "state",
        "TMPDIR": runtime_root / "tmp",
        "MPLCONFIGDIR": runtime_root / "matplotlib",
    }
    for path in runtime_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    wrapped = [
        executable,
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--cap-drop", "ALL",
        "--ro-bind", "/", "/",
        # Do not expose host daemon and desktop sockets. Read-only socket files
        # can still be connected to, so a read-only root alone is insufficient.
        "--tmpfs", "/run",
        "--tmpfs", "/tmp",
        # Bun/OpenCode needs live device and proc mounts. Replacing the
        # read-only recursive binds also avoids a Bun startup crash.
        "--dev", "/dev",
        "--proc", "/proc",
        "--bind", str(owned), str(owned),
        "--bind", str(runtime_root), str(runtime_root),
        "--chdir", str(worktree),
    ]
    for name, value in runtime_dirs.items():
        wrapped.extend(("--setenv", name, str(value)))
    wrapped.extend(("--setenv", "PYTHONDONTWRITEBYTECODE", "1"))
    wrapped.extend(("--setenv", "TASK_SEARCH_SPEC",
                    str(runtime_root / "trial_spec.json")))
    wrapped.extend(command)
    return wrapped


def _run_validation(worktree, commands, log_path, *, owned_path, runtime_root,
                    bwrap_bin, resource_limits, timeout_seconds,
                    credential_env_names=()):
    results = []
    environment = _sanitized_environment(credential_env_names)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with log_path.open("w") as log:
        for command in commands:
            log.write(f"$ {command}\n")
            log.flush()
            sandboxed = _sandbox_command(
                ["/bin/bash", "-c", command],
                worktree=worktree,
                owned_path=owned_path,
                runtime_root=runtime_root,
                bwrap_bin=bwrap_bin,
            )
            sandboxed = _resource_command(sandboxed, resource_limits)
            try:
                completed = subprocess.run(
                    sandboxed,
                    cwd=worktree,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds,
                )
                exit_code = completed.returncode
                timed_out = False
            except subprocess.TimeoutExpired:
                log.write(f"TIMEOUT after {timeout_seconds} seconds\n")
                log.flush()
                exit_code = 124
                timed_out = True
            results.append({
                "command": command,
                "exit_code": exit_code,
                "timed_out": timed_out,
            })
            if exit_code:
                break
    return results


_CONTRACT_AUDIT = r"""
import importlib
import json
import random
import sys

classes = json.loads(sys.argv[1])
seed = int(sys.argv[2])
for offset, (module_name, class_name) in enumerate(classes):
    task_class = getattr(importlib.import_module(module_name), class_name)
    task = task_class()
    random.seed(seed + offset)
    task.validate(n_samples=10)
    for sample in range(64):
        entry = task.generate_example()
        assert task.score_answer(entry.answer, entry) == 1, (
            module_name, class_name, sample, "gold answer rejected")
        for bad in ("", " ", "reajrjrje9595!"):
            if bad != str(entry.answer):
                assert task.score_answer(bad, entry) < 1, (
                    module_name, class_name, sample,
                    f"invalid answer scored as correct: {bad!r}; gold={entry.answer!r}")
print(f"CONTRACT_AUDIT_OK {len(classes)} task class(es)")
"""


def _run_contract_audit(worktree, owned_path, seed, log_path, *, runtime_root,
                        bwrap_bin, resource_limits, timeout_seconds,
                        credential_env_names=()):
    try:
        classes = _task_classes(worktree, owned_path)
    except (SyntaxError, ValueError) as error:
        # Candidate code that does not parse is a bad candidate, not a runner bug.
        log_path.write_text(f"{type(error).__name__}: {error}\n")
        return {"classes": [], "exit_code": 2, "parse_error": str(error)}
    if not classes:
        return {"classes": [], "exit_code": 2}
    environment = _sanitized_environment(credential_env_names)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = _sandbox_command(
        [sys.executable, "-c", _CONTRACT_AUDIT, json.dumps(classes), str(seed)],
        worktree=worktree,
        owned_path=owned_path,
        runtime_root=runtime_root,
        bwrap_bin=bwrap_bin,
    )
    command = _resource_command(command, resource_limits)
    with log_path.open("w") as log:
        try:
            completed = subprocess.run(
                command,
                cwd=worktree,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
            )
            exit_code = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            log.write(f"TIMEOUT after {timeout_seconds} seconds\n")
            exit_code = 124
            timed_out = True
    return {"classes": classes, "exit_code": exit_code, "timed_out": timed_out}


def _run_trial(plan, trial, repo_root, invocation_root, base_commit,
               model, harness, agent, variant, harness_bin, harness_version,
               base_seed, forward_seed, temperature, top_p, bwrap_bin,
               sandbox_version, max_steps, timeout_seconds, adapter, provider,
               adapter_bin, adapter_version, resource_limits,
               validation_timeout_seconds, credential_env_names,
               pace=DEFAULT_PACE):
    trial_root = invocation_root / trial.trial_id
    worktree = trial_root / "worktree"
    trial_root.mkdir(parents=True)
    subprocess.run(
        ["git", "worktree", "add", "-q", "--detach", str(worktree), base_commit],
        cwd=repo_root,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    (worktree / trial.owned_path).mkdir(parents=True, exist_ok=True)
    requested_seed = int(_sha256(f"{base_seed}:{trial.trial_id}")[:8], 16)
    generation_agent = agent if harness == "opencode" else "mini-default"
    generation = generation_metadata(
        model, harness_version, generation_agent, variant,
        requested_seed=requested_seed,
        seed_forwarded=forward_seed,
        temperature=temperature,
        top_p=top_p,
        sandbox_version=sandbox_version,
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
        provider_name=provider,
        adapter_name=adapter,
        adapter_version=adapter_version,
        harness_name=harness,
    )
    parent_source_id = None
    if trial.parent:
        parent_source = subprocess.check_output(
            ["git", "show", f"{base_commit}:{trial.parent}"],
            cwd=repo_root,
        ).decode("utf-8")
        parent_source_id = SourceStore(
            repo_root / ".evolution" / "objects").put(parent_source)
    task_meta = {
        "parent_source_id": parent_source_id,
        "idea": trial.idea,
        "hypothesis": trial.hypothesis,
        "changes": trial.changes,
        "generation": generation,
    }
    prompt = render_prompt(plan, trial, repo_root, task_meta, pace)
    prompt_path = trial_root / "prompt.md"
    prompt_path.write_text(prompt)
    runtime_root = trial_root / "runtime"
    runtime_root.mkdir()
    # Out of the worktree: anything inside it would count as a changed path and the
    # trial would lose on scope_violation.
    _write_json(runtime_root / "trial_spec.json",
                {"trial_id": trial.trial_id, "owned_path": trial.owned_path,
                 "task_meta": task_meta})
    started = datetime.now(timezone.utc).isoformat()
    if harness == "opencode":
        config_path = trial_root / "opencode.json"
        _write_json(config_path, opencode_config(
            trial,
            agent,
            requested_seed=requested_seed,
            forward_seed=forward_seed,
            temperature=temperature,
            top_p=top_p,
            max_steps=max_steps,
        ))
        harness_model = model if adapter == "direct" else f"{provider}/{model}"
        command = _opencode_command(
            harness_bin,
            model=harness_model,
            agent=agent,
            worktree=worktree,
            prompt=prompt,
            variant=variant,
        )
        events_path = trial_root / "events.jsonl"
        trajectory_path = None
    elif harness == "mini":
        config_path = trial_root / "mini.yaml"
        config_path.write_text(yaml.safe_dump(
            _mini_config(
                worktree, max_steps=max_steps,
                timeout_seconds=timeout_seconds,
                requested_seed=requested_seed,
                forward_seed=forward_seed,
                temperature=temperature,
                top_p=top_p,
            ),
            sort_keys=False,
        ))
        trajectory_path = runtime_root / "trajectory.json"
        command = _mini_command(
            harness_bin,
            prompt=prompt,
            config_path=config_path,
            trajectory_path=trajectory_path,
        )
        events_path = trial_root / "harness.log"
    else:
        raise ValueError(f"unsupported harness: {harness}")
    command = _adapter_command(
        command,
        adapter=adapter,
        provider=provider,
        adapter_bin=adapter_bin,
        model=model,
        harness=harness,
    )
    command = _sandbox_command(
        command,
        worktree=worktree,
        owned_path=trial.owned_path,
        runtime_root=runtime_root,
        bwrap_bin=bwrap_bin,
    )
    command = _resource_command(command, resource_limits)
    environment = dict(os.environ)
    if harness == "opencode":
        environment["OPENCODE_CONFIG_CONTENT"] = config_path.read_text()
        environment["OPENCODE_DISABLE_EXTERNAL_SKILLS"] = "true"
        environment["OPENCODE_DISABLE_CLAUDE_CODE_SKILLS"] = "true"
    else:
        environment["MSWEA_CONFIGURED"] = "true"
    timed_out = False
    with events_path.open("w") as stdout, (trial_root / "stderr.log").open("w") as stderr:
        try:
            completed = subprocess.run(
                command,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_seconds,
            )
            harness_exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            harness_exit_code = 124
    initial_changed_paths = _changed_paths(worktree)
    initial_outside = _outside_owned(initial_changed_paths, trial.owned_path)
    metadata_error = None
    try:
        discovered_meta = _task_metadata(worktree, trial.owned_path) if not initial_outside else []
    except (SyntaxError, ValueError) as error:
        # A syntax error or a non-literal TASK_META = dict(...) used to escape all the
        # way to run_plan and be recorded as orchestration_error with no run.json at
        # all, which reads as a runner bug rather than as the candidate failure it is.
        discovered_meta, metadata_error = [], f"{type(error).__name__}: {error}"
    metadata_ok = len(discovered_meta) == 1 and discovered_meta[0][1] == task_meta
    sample_review = _sample_review(
        worktree, trial.owned_path, trial.trial_id, events_path)
    gates_open = not initial_outside and metadata_ok and harness_exit_code == 0
    validation_runtime = trial_root / "validation_runtime"
    contract_audit = ({"classes": [], "exit_code": None} if not gates_open else
                      _run_contract_audit(
                          worktree, trial.owned_path, requested_seed,
                          trial_root / "contract_audit.log",
                          runtime_root=validation_runtime,
                          bwrap_bin=bwrap_bin,
                          resource_limits=resource_limits,
                          timeout_seconds=validation_timeout_seconds,
                          credential_env_names=credential_env_names))
    contract_ok = contract_audit["exit_code"] == 0
    hidden_modules = _undiscoverable(contract_audit["classes"])
    sample_name = f"samples_{trial.trial_id}.md"
    # Everything the contract audit just certified, hashed. The sample generator is
    # allowed to rewrite its own output and nothing else.
    frozen = _owned_digest(worktree, trial.owned_path, exclude=(sample_name,))
    sample_path = worktree / trial.owned_path / f"samples_{trial.trial_id}.md"
    sample_sha256_before = (
        hashlib.sha256(sample_path.read_bytes()).hexdigest()
        if sample_path.is_file() else None
    )
    validation_commands = (_sample_command(trial), *trial.validation,
                           _prior_audit_command(trial))
    validation = [] if not gates_open else _run_validation(
        worktree, validation_commands, trial_root / "validation.log",
        owned_path=trial.owned_path,
        runtime_root=validation_runtime,
        bwrap_bin=bwrap_bin,
        resource_limits=resource_limits,
        timeout_seconds=validation_timeout_seconds,
        credential_env_names=credential_env_names)
    sample_sha256_after = (
        hashlib.sha256(sample_path.read_bytes()).hexdigest()
        if sample_path.is_file() else None
    )
    # Run the generator five times, twice under each of two string-hash salts and once
    # under a third. A same-salt pair disagreeing means the generator is stateful or
    # keys on an object whose hash is its id, which PYTHONHASHSEED does not pin at all;
    # only cross-salt disagreement means an unsorted set of strings reaches the output.
    # Two same-salt pairs rather than one because M1 in wave 20260829T072855Z agreed
    # with itself on four consecutive runs at salt 0 and disagreed on the fifth: at
    # three runs this gate passes an id-hashing generator about a quarter of the time.
    # A samples file that was merely stale is recorded, not gated -- the run above has
    # already refreshed it.
    recheck, recheck_digests = [], []
    for salt in ("0", "0", "1", "1", "2"):
        if not gates_open:
            break
        recheck += _run_validation(
            worktree, (f"PYTHONHASHSEED={salt} {_sample_command(trial)}",),
            trial_root / f"sample_recheck_{len(recheck_digests)}.log",
            owned_path=trial.owned_path,
            runtime_root=validation_runtime,
            bwrap_bin=bwrap_bin,
            resource_limits=resource_limits,
            timeout_seconds=validation_timeout_seconds,
            credential_env_names=credential_env_names)
        recheck_digests.append(
            hashlib.sha256(sample_path.read_bytes()).hexdigest()
            if sample_path.is_file() else None)
    after_validation = _owned_digest(worktree, trial.owned_path, exclude=(sample_name,))
    mutated_paths = sorted(
        set(frozen["files"]) ^ set(after_validation["files"])
        | {name for name, digest in frozen["files"].items()
           if after_validation["files"].get(name, digest) != digest})
    # The section markers are re-read here, not where _sample_review looked: that check
    # ran on whatever file the worker left behind, and the deterministic replay above
    # has since overwritten it. A stale file with the right markers used to pass.
    replayed_shortfall = ["samples file is missing"]
    if sample_path.is_file():
        replayed_shortfall = sample_shortfall(sample_path.read_text())
    sample_review["required_sections_after_replay"] = not replayed_shortfall
    sample_review["shortfall_after_replay"] = replayed_shortfall
    sample_validation = {
        "sha256_before": sample_sha256_before,
        "sha256_after": sample_sha256_after,
        "sha256_recheck": recheck_digests,
        "stale": (sample_sha256_before is not None
                  and sample_sha256_before != sample_sha256_after),
        "reproducible": (len(recheck_digests) == 5
                         and recheck_digests[0] is not None
                         and len(set(recheck_digests)) == 1),
        "irreproducible_as": (
            None if len(recheck_digests) < 5 or len(set(recheck_digests)) == 1
            else "stateful" if (recheck_digests[0] != recheck_digests[1]
                                or recheck_digests[2] != recheck_digests[3])
            else "hash_order"),
        "checked": bool(recheck) and all(r["exit_code"] == 0 for r in recheck),
    }
    validation_ok = (
        bool(validation)
        and all(item["exit_code"] == 0 for item in validation)
        and sample_validation["reproducible"]
        and not replayed_shortfall
    )
    candidate_frozen = not mutated_paths
    changed_paths = _changed_paths(worktree)
    outside = _outside_owned(changed_paths, trial.owned_path)
    if timed_out:
        status = "timed_out"
    elif harness_exit_code:
        status = "harness_failed"
    elif outside:
        status = "scope_violation"
    elif not changed_paths:
        # Nothing outside owned + nothing at all == the trial wrote no files.
        status = "no_implementation"
    elif not metadata_ok:
        status = "metadata_mismatch"
    elif not sample_review["ok"]:
        status = "sample_review_failed"
    elif not contract_ok:
        status = "contract_failed"
    elif hidden_modules:
        status = "undiscoverable"
    elif gates_open and not candidate_frozen:
        status = "candidate_mutated"
    elif (sample_validation["checked"]
          and not sample_validation["reproducible"]):
        status = "sample_not_reproducible"
    elif not validation_ok:
        status = "validation_failed"
    else:
        status = "success"
    record = {
        "schema_version": 1,
        "wave": plan.name,
        "trial_id": trial.trial_id,
        "hypothesis": trial.hypothesis,
        "base_commit": base_commit,
        "plan_sha256": plan.sha256,
        # Wave-level, deliberately not inside TASK_META: anything in there is pasted
        # into the worker's prompt, and a pace A/B whose treatment also edits the
        # provenance mapping is not measuring the pacing alone.
        "pace": pace,
        "prompt_sha256": _sha256(prompt),
        "generation": generation,
        "parent_source_id": parent_source_id,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "harness_exit_code": harness_exit_code,
        "harness_log": str(events_path),
        "steps": _step_usage(events_path, max_steps),
        "trajectory": str(trajectory_path) if trajectory_path else None,
        "timed_out": timed_out,
        "sandbox": {"name": "bubblewrap", "version": sandbox_version},
        "resource_limits": _public_resource_limits(resource_limits),
        "scrubbed_credential_env_names": sorted(credential_env_names),
        "changed_paths": changed_paths,
        "outside_owned_path": outside,
        "task_metadata": discovered_meta,
        "task_metadata_matches": metadata_ok,
        "task_metadata_error": metadata_error,
        # The accepted candidate, hashed, so the result has a referent that outlives
        # the mutable worktree.
        "candidate": {"tree_sha256": after_validation["tree_sha256"],
                      "files": after_validation["files"],
                      "frozen": candidate_frozen,
                      "mutated_paths": mutated_paths,
                      "undiscoverable_modules": hidden_modules},
        "sample_review": sample_review,
        "sample_validation": sample_validation,
        "contract_audit": contract_audit,
        "validation": validation,
        "status": status,
        "worktree": str(worktree),
    }
    _write_json(trial_root / "run.json", record)
    return record


def _select_trials(plan, trial_ids=(), queue_names=()):
    unknown_trials = set(trial_ids) - {trial.trial_id for trial in plan.trials}
    if unknown_trials:
        raise ValueError(f"unknown trial IDs: {', '.join(sorted(unknown_trials))}")
    unknown_queues = set(queue_names) - set(plan.queues)
    if unknown_queues:
        raise ValueError(
            f"unknown queues: {', '.join(sorted(unknown_queues))}")
    selected_ids = set(trial_ids)
    for queue in queue_names:
        selected_ids.update(plan.queues[queue])
    if not selected_ids:
        return list(plan.trials)
    return [trial for trial in plan.trials if trial.trial_id in selected_ids]


def run_plan(plan_path, *, model, jobs=1, trial_ids=(), agent="task-search-worker",
             variant=None, seed=0, forward_seed=True, temperature=None, top_p=None,
             harness="opencode", opencode_bin="opencode", mini_bin="mini",
             bwrap_bin="bwrap", runs_root=None,
             repo_root=None, max_steps=56, timeout_seconds=1800,
             queue_names=(), adapter="direct", provider=None, adapter_bin=None,
             resource_limit_mode="auto", systemd_run_bin="systemd-run",
             memory_max="8G", tasks_max=512, cpu_quota="400%",
             validation_timeout_seconds=300, credential_env_names=(),
             pace=DEFAULT_PACE):
    """Run selected trials concurrently in isolated Git worktrees."""
    if pace not in PACE:
        raise ValueError(f"unknown pace: {pace!r}; choose from {', '.join(sorted(PACE))}")
    plan = load_plan(plan_path)
    repo_root = Path(repo_root).resolve() if repo_root else _repo_root(plan.path.parent)
    selected = _select_trials(plan, trial_ids, queue_names)
    base_commit = subprocess.check_output(
        ["git", "rev-parse", plan.base_ref], cwd=repo_root, text=True).strip()
    problems = _plan_problems(plan, repo_root)
    if problems:
        raise ValueError("plan cannot run:\n  " + "\n  ".join(problems))
    drift = _frozen_module_drift(repo_root, base_commit)
    if drift:
        print(f"WARNING: {drift}", file=sys.stderr)
    if adapter == "harness-link" and not provider:
        raise ValueError("--provider is required with --adapter harness-link")
    if harness not in {"opencode", "mini"}:
        raise ValueError(f"unsupported harness: {harness}")
    if adapter == "harness-link":
        resolved_adapter = shutil.which(adapter_bin or provider)
        if resolved_adapter is None:
            raise RuntimeError(
                f"Harness Link provider command not found: {adapter_bin or provider!r}")
        adapter_version = subprocess.check_output(
            [resolved_adapter, "--version"], text=True).strip()
    elif adapter == "direct":
        adapter_version = None
    else:
        raise ValueError(f"unsupported adapter: {adapter}")
    requested_harness_bin = opencode_bin if harness == "opencode" else mini_bin
    harness_executable = _resolve_harness_executable(
        harness, adapter, requested_harness_bin)
    harness_version = _harness_version(harness, harness_executable)
    bwrap_path = shutil.which(bwrap_bin)
    if bwrap_path is None:
        raise RuntimeError(
            f"bubblewrap executable not found: {bwrap_bin!r}; "
            "strict task-search runs require bubblewrap"
        )
    sandbox_version = subprocess.check_output(
        [bwrap_path, "--version"], text=True).strip()
    resource_limits = _resolve_resource_limits(
        resource_limit_mode,
        systemd_run_bin=systemd_run_bin,
        memory_max=memory_max,
        tasks_max=tasks_max,
        cpu_quota=cpu_quota,
    )
    root = Path(runs_root).resolve() if runs_root else repo_root.parent / f".{repo_root.name}-task-search"
    invocation = root / plan.name / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    invocation.mkdir(parents=True)
    results = []

    def write_summary():
        _write_json(invocation / "summary.json", {
            "wave": plan.name,
            "queues": list(queue_names),
            "base_commit": base_commit,
            "model": model,
            "harness": {"name": harness, "version": harness_version},
            "provider": provider or model.split("/", 1)[0],
            "adapter": {"name": adapter, "version": adapter_version},
            "seed": seed,
            "seed_forwarded": forward_seed,
            "max_steps": max_steps,
            "timeout_seconds": timeout_seconds,
            "validation_timeout_seconds": validation_timeout_seconds,
            "pace": pace,
            "sandbox": {"name": "bubblewrap", "version": sandbox_version},
            "resource_limits": _public_resource_limits(resource_limits),
            "scrubbed_credential_env_names": sorted(credential_env_names),
            "results": sorted(results, key=lambda item: item["trial_id"]),
        })

    write_summary()
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {
            pool.submit(
                _run_trial, plan, trial, repo_root, invocation, base_commit,
                model, harness, agent, variant, harness_executable, harness_version,
                seed, forward_seed, temperature, top_p, bwrap_path,
                sandbox_version, max_steps, timeout_seconds,
                adapter, provider, adapter_bin, adapter_version, resource_limits,
                validation_timeout_seconds, tuple(credential_env_names), pace,
            ): trial.trial_id
            for trial in selected
        }
        for future in as_completed(futures):
            trial_id = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "schema_version": 1,
                    "wave": plan.name,
                    "trial_id": trial_id,
                    "status": "orchestration_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            results.append(result)
            write_summary()
    return sorted(results, key=lambda item: item["trial_id"])


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="validate and summarize a plan")
    check.add_argument("plan")
    render = subparsers.add_parser("render", help="render one worker prompt")
    render.add_argument("plan")
    render.add_argument("trial_id")
    render.add_argument("--pace", choices=sorted(PACE), default=DEFAULT_PACE)
    run = subparsers.add_parser("run", help="launch folder-scoped coding workers")
    run.add_argument("plan")
    run.add_argument("--model", required=True)
    run.add_argument("--harness", choices=("opencode", "mini"), default="opencode")
    run.add_argument(
        "--adapter", choices=("direct", "harness-link"), default="direct")
    run.add_argument("--provider", help="provider command, e.g. albert or nim")
    run.add_argument("--adapter-bin", help="explicit Harness Link provider executable")
    run.add_argument(
        "--credential-env", action="append", default=[],
        help="environment variable to remove from candidate validation processes",
    )
    run.add_argument("--jobs", type=int, default=1)
    run.add_argument("--trial", action="append", default=[])
    run.add_argument("--queue", action="append", default=[])
    run.add_argument("--agent", default="task-search-worker")
    run.add_argument("--variant")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--forward-seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="forward each derived trial seed to OpenCode (default: enabled)",
    )
    run.add_argument("--temperature", type=float)
    run.add_argument("--top-p", type=float)
    run.add_argument("--max-steps", type=int, default=56)
    run.add_argument("--pace", choices=sorted(PACE), default=DEFAULT_PACE,
                     help="how hard the worker is told to hurry; recorded in generation metadata so waves stay comparable")
    run.add_argument("--timeout-seconds", type=int, default=1800)
    run.add_argument("--validation-timeout-seconds", type=int, default=300)
    run.add_argument("--opencode-bin", default="opencode")
    run.add_argument("--mini-bin", default="mini")
    run.add_argument("--bwrap-bin", default="bwrap")
    run.add_argument(
        "--resource-limits", choices=("auto", "required", "none"),
        default="auto",
        help="apply a user systemd scope to every worker and validation process",
    )
    run.add_argument("--systemd-run-bin", default="systemd-run")
    run.add_argument("--memory-max", default="8G")
    run.add_argument("--tasks-max", type=int, default=512)
    run.add_argument("--cpu-quota", default="400%")
    run.add_argument("--runs-root")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    plan = load_plan(args.plan)
    if args.command == "check":
        print(f"{plan.name}: {len(plan.trials)} trials from {plan.base_ref}")
        repo_root = _repo_root(plan.path.parent)
        for problem in _plan_problems(plan, repo_root):
            print(f"PROBLEM: {problem}")
        drift = _frozen_module_drift(repo_root, plan.base_ref)
        if drift:
            print(f"WARNING: {drift}")
        for name, members in plan.queues.items():
            print(f"queue\t{name}\t{','.join(members)}")
        for trial in plan.trials:
            print(f"{trial.trial_id}\t{trial.hypothesis or '-'}\t{trial.owned_path}")
    elif args.command == "render":
        trial = next((item for item in plan.trials if item.trial_id == args.trial_id), None)
        if trial is None:
            raise SystemExit(f"unknown trial: {args.trial_id}")
        # A template preview: execution builds a model-, seed- and budget-dependent
        # TASK_META and passes it in. The prompt a worker actually got is its
        # prompt.md, in the trial directory.
        print(render_prompt(plan, trial, _repo_root(plan.path.parent),
                            pace=args.pace), end="")
    else:
        results = run_plan(
            args.plan,
            model=args.model,
            jobs=args.jobs,
            trial_ids=args.trial,
            queue_names=args.queue,
            agent=args.agent,
            variant=args.variant,
            seed=args.seed,
            forward_seed=args.forward_seed,
            temperature=args.temperature,
            top_p=args.top_p,
            max_steps=args.max_steps,
            timeout_seconds=args.timeout_seconds,
            harness=args.harness,
            opencode_bin=args.opencode_bin,
            mini_bin=args.mini_bin,
            bwrap_bin=args.bwrap_bin,
            runs_root=args.runs_root,
            adapter=args.adapter,
            provider=args.provider,
            adapter_bin=args.adapter_bin,
            resource_limit_mode=args.resource_limits,
            systemd_run_bin=args.systemd_run_bin,
            memory_max=args.memory_max,
            tasks_max=args.tasks_max,
            cpu_quota=args.cpu_quota,
            validation_timeout_seconds=args.validation_timeout_seconds,
            credential_env_names=args.credential_env,
            pace=args.pace,
        )
        for result in results:
            print(f"{result['trial_id']}\t{result['status']}\t"
                  f"{result.get('worktree', '-')}")
        if any(result["status"] != "success" for result in results):
            raise SystemExit(1)
