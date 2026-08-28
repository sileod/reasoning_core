"""Plan-driven, folder-scoped task-search workers."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
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


def _relative_path(value, field):
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{field} must be a repository-relative path: {value}")
    return path.as_posix().rstrip("/")


def load_plan(path):
    """Load and validate a task-search YAML plan."""
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
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
    return SearchPlan(name, base_ref, contexts, tuple(trials), queues, path)


def _repo_root(start):
    output = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=start, text=True)
    return Path(output.strip()).resolve()


def _sha256(data):
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def render_prompt(plan, trial, repo_root, task_meta=None):
    """Compose stable global context with one compact assignment."""
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
        f"Parent module: `{trial.parent or 'none (new task)'}`",
        f"Owned path: `{trial.owned_path}/`",
        "",
        "You may read the whole repository but may modify files only under the owned path.",
        "Do not commit, push, move the assignment, or edit shared files.",
        "Keep tests for this trial inside the owned path.",
        f"Before finishing, create `generate_samples_{trial.trial_id}.py` and use it to",
        f"generate `samples_{trial.trial_id}.md` from the actual task generator.",
        "Include at least two complete prompt/answer examples at each of levels 0, 2, and 5,",
        "covering the task's important query modes where possible.",
        f"Run `{_sample_command(trial)}` and then read `samples_{trial.trial_id}.md`;",
        "review wording, answers, trivial cases, and",
        "difficulty progression, then revise the task and regenerate the file if needed.",
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
        "Run the following validation commands before finishing:",
        "",
        "```text",
        *trial.validation,
        "```",
        "",
        "Finish with a concise summary of changes and validation results.",
        "",
    ))
    return "\n".join(sections)


def generation_metadata(model, harness_version, agent, variant=None,
                        requested_seed=None, seed_forwarded=False,
                        temperature=None, top_p=None, sandbox_name="bubblewrap",
                        sandbox_version=None, max_steps=48,
                        timeout_seconds=1800):
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
        "provider_name": model.split("/", 1)[0],
        "model_name": model,
        "harness_name": "opencode",
        "harness_version": harness_version,
        "agent_name": agent,
        "settings": settings,
    }


def _sample_command(trial):
    return (
        "PYTHONDONTWRITEBYTECODE=1 python "
        f"{trial.owned_path}/generate_samples_{trial.trial_id}.py"
    )


def opencode_permissions(trial):
    bash = {
        "*": "deny",
        "git status*": "allow",
        "git diff*": "allow",
        "python -c *": "allow",
        "PYTHONDONTWRITEBYTECODE=1 python -c *": "allow",
    }
    for command in trial.validation:
        bash[command] = "allow"
    bash[_sample_command(trial)] = "allow"
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
                    temperature=None, top_p=None, max_steps=48):
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
    }
    if sample_path.is_file():
        content = sample_path.read_text().lower()
        result["required_sections"] = all(
            marker in content for marker in ("level 0", "level 2", "level 5", "answer")
        )
    target = "/" + result["path"]
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
        if not file_path.endswith(target):
            continue
        if part.get("tool") in {"write", "edit", "apply_patch"}:
            last_write = index
        elif part.get("tool") == "read":
            last_read = index
    result["read_after_last_edit"] = last_read > last_write
    result["ok"] = (
        result["generator_exists"] and result["exists"] and result["size_bytes"] > 0
        and result["required_sections"] and result["read_after_last_edit"]
    )
    return result


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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


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
        "--ro-bind", "/", "/",
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
    wrapped.extend(command)
    return wrapped


def _run_validation(worktree, commands, log_path, *, owned_path, runtime_root,
                    bwrap_bin):
    results = []
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
            completed = subprocess.run(
                sandboxed,
                cwd=worktree,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            results.append({"command": command, "exit_code": completed.returncode})
            if completed.returncode:
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
                        bwrap_bin):
    classes = _task_classes(worktree, owned_path)
    if not classes:
        return {"classes": [], "exit_code": 2}
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = _sandbox_command(
        [sys.executable, "-c", _CONTRACT_AUDIT, json.dumps(classes), str(seed)],
        worktree=worktree,
        owned_path=owned_path,
        runtime_root=runtime_root,
        bwrap_bin=bwrap_bin,
    )
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=worktree,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return {"classes": classes, "exit_code": completed.returncode}


def _run_trial(plan, trial, repo_root, invocation_root, base_commit,
               model, agent, variant, opencode_bin, harness_version,
               base_seed, forward_seed, temperature, top_p, bwrap_bin,
               sandbox_version, max_steps, timeout_seconds):
    trial_root = invocation_root / trial.trial_id
    worktree = trial_root / "worktree"
    trial_root.mkdir(parents=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), base_commit],
        cwd=repo_root,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    (worktree / trial.owned_path).mkdir(parents=True, exist_ok=True)
    requested_seed = int(_sha256(f"{base_seed}:{trial.trial_id}")[:8], 16)
    generation = generation_metadata(
        model, harness_version, agent, variant,
        requested_seed=requested_seed,
        seed_forwarded=forward_seed,
        temperature=temperature,
        top_p=top_p,
        sandbox_version=sandbox_version,
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
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
    prompt = render_prompt(plan, trial, repo_root, task_meta)
    prompt_path = trial_root / "prompt.md"
    prompt_path.write_text(prompt)
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
    started = datetime.now(timezone.utc).isoformat()
    command = _opencode_command(
        opencode_bin,
        model=model,
        agent=agent,
        worktree=worktree,
        prompt=prompt,
        variant=variant,
    )
    command = _sandbox_command(
        command,
        worktree=worktree,
        owned_path=trial.owned_path,
        runtime_root=trial_root / "runtime",
        bwrap_bin=bwrap_bin,
    )
    environment = dict(os.environ)
    environment["OPENCODE_CONFIG"] = str(config_path)
    environment["OPENCODE_DISABLE_EXTERNAL_SKILLS"] = "true"
    environment["OPENCODE_DISABLE_CLAUDE_CODE_SKILLS"] = "true"
    timed_out = False
    with (trial_root / "events.jsonl").open("w") as stdout, (trial_root / "stderr.log").open("w") as stderr:
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
    discovered_meta = _task_metadata(worktree, trial.owned_path) if not initial_outside else []
    metadata_ok = len(discovered_meta) == 1 and discovered_meta[0][1] == task_meta
    sample_review = _sample_review(
        worktree, trial.owned_path, trial.trial_id, trial_root / "events.jsonl")
    gates_open = not initial_outside and metadata_ok and harness_exit_code == 0
    validation_runtime = trial_root / "validation_runtime"
    contract_audit = ({"classes": [], "exit_code": None} if not gates_open else
                      _run_contract_audit(
                          worktree, trial.owned_path, requested_seed,
                          trial_root / "contract_audit.log",
                          runtime_root=validation_runtime,
                          bwrap_bin=bwrap_bin))
    contract_ok = contract_audit["exit_code"] == 0
    validation = [] if not gates_open else _run_validation(
        worktree, trial.validation, trial_root / "validation.log",
        owned_path=trial.owned_path,
        runtime_root=validation_runtime,
        bwrap_bin=bwrap_bin)
    validation_ok = bool(validation) and all(item["exit_code"] == 0 for item in validation)
    changed_paths = _changed_paths(worktree)
    outside = _outside_owned(changed_paths, trial.owned_path)
    if timed_out:
        status = "timed_out"
    elif harness_exit_code:
        status = "harness_failed"
    elif outside:
        status = "scope_violation"
    elif not metadata_ok:
        status = "metadata_mismatch"
    elif not sample_review["ok"]:
        status = "sample_review_failed"
    elif not contract_ok:
        status = "contract_failed"
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
        "plan_sha256": hashlib.sha256(plan.path.read_bytes()).hexdigest(),
        "prompt_sha256": _sha256(prompt),
        "generation": generation,
        "parent_source_id": parent_source_id,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "harness_exit_code": harness_exit_code,
        "timed_out": timed_out,
        "sandbox": {"name": "bubblewrap", "version": sandbox_version},
        "changed_paths": changed_paths,
        "outside_owned_path": outside,
        "task_metadata": discovered_meta,
        "task_metadata_matches": metadata_ok,
        "sample_review": sample_review,
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
             opencode_bin="opencode", bwrap_bin="bwrap", runs_root=None,
             repo_root=None, max_steps=48, timeout_seconds=1800,
             queue_names=()):
    """Run selected trials concurrently in isolated Git worktrees."""
    plan = load_plan(plan_path)
    repo_root = Path(repo_root).resolve() if repo_root else _repo_root(plan.path.parent)
    selected = _select_trials(plan, trial_ids, queue_names)
    base_commit = subprocess.check_output(
        ["git", "rev-parse", plan.base_ref], cwd=repo_root, text=True).strip()
    harness_version = subprocess.check_output([opencode_bin, "--version"], text=True).strip()
    bwrap_path = shutil.which(bwrap_bin)
    if bwrap_path is None:
        raise RuntimeError(
            f"bubblewrap executable not found: {bwrap_bin!r}; "
            "strict task-search runs require bubblewrap"
        )
    sandbox_version = subprocess.check_output(
        [bwrap_path, "--version"], text=True).strip()
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
            "seed": seed,
            "seed_forwarded": forward_seed,
            "max_steps": max_steps,
            "timeout_seconds": timeout_seconds,
            "sandbox": {"name": "bubblewrap", "version": sandbox_version},
            "results": sorted(results, key=lambda item: item["trial_id"]),
        })

    write_summary()
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {
            pool.submit(
                _run_trial, plan, trial, repo_root, invocation, base_commit,
                model, agent, variant, opencode_bin, harness_version,
                seed, forward_seed, temperature, top_p, bwrap_path,
                sandbox_version, max_steps, timeout_seconds,
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
    run = subparsers.add_parser("run", help="launch folder-scoped OpenCode workers")
    run.add_argument("plan")
    run.add_argument("--model", required=True)
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
    run.add_argument("--max-steps", type=int, default=48)
    run.add_argument("--timeout-seconds", type=int, default=1800)
    run.add_argument("--opencode-bin", default="opencode")
    run.add_argument("--bwrap-bin", default="bwrap")
    run.add_argument("--runs-root")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    plan = load_plan(args.plan)
    if args.command == "check":
        print(f"{plan.name}: {len(plan.trials)} trials from {plan.base_ref}")
        for name, members in plan.queues.items():
            print(f"queue\t{name}\t{','.join(members)}")
        for trial in plan.trials:
            print(f"{trial.trial_id}\t{trial.hypothesis or '-'}\t{trial.owned_path}")
    elif args.command == "render":
        trial = next((item for item in plan.trials if item.trial_id == args.trial_id), None)
        if trial is None:
            raise SystemExit(f"unknown trial: {args.trial_id}")
        print(render_prompt(plan, trial, _repo_root(plan.path.parent)), end="")
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
            opencode_bin=args.opencode_bin,
            bwrap_bin=args.bwrap_bin,
            runs_root=args.runs_root,
        )
        for result in results:
            print(f"{result['trial_id']}\t{result['status']}\t{result['worktree']}")
        if any(result["status"] != "success" for result in results):
            raise SystemExit(1)
