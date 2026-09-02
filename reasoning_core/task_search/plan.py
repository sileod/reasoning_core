"""Task-search plan model, loading, validation, and selection."""

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess

import yaml


@dataclass(frozen=True)
class Trial:
    trial_id: str
    instruction: str
    owned_path: str
    validation: tuple
    hypothesis: str = ""
    parent: str = ""
    # The one approach this trial is told to take, when the wave is comparing approaches
    # rather than seeds. Empty is the normal case and renders no prompt bytes at all.
    design_choice: str = ""
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
    # Two different things used to share one name. `name` is the implementation wave --
    # one attempt at building tasks, repeatable with new seeds against the same ideas.
    # `proposal_wave` is where the ideas came from. wave8 was two implementation runs of
    # the eighty wave0 proposals and no run record said so, which made "how did the
    # legacy list do against kimi's" unanswerable from the records alone. Empty means a
    # plan written before the field existed, and `check` reports it as unrecorded.
    proposal_wave: str = ""
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
    proposal_wave = str(data.get("proposal_wave", "")).strip()
    defaults = data.get("defaults") or {}
    base_ref = str(defaults.get("base_ref", "HEAD"))
    contexts = tuple(
        _relative_path(value, "context file") for value in data.get("context_files", ())
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
        validation = tuple(
            str(command).strip() for command in raw.get("validation", ())
        )
        if not validation or any(not command for command in validation):
            raise ValueError(f"{trial_id}: at least one validation command is required")
        trials.append(
            Trial(
                trial_id=trial_id,
                instruction=instruction,
                owned_path=owned_path,
                validation=validation,
                hypothesis=str(raw.get("hypothesis", "")).strip(),
                parent=(
                    _relative_path(raw["parent"], f"{trial_id} parent")
                    if raw.get("parent")
                    else ""
                ),
                idea=idea,
                changes=changes,
                design_choice=str(raw.get("design_choice", "")).strip(),
            )
        )
    if not trials:
        raise ValueError("at least one trial is required")
    if len({trial.trial_id for trial in trials}) != len(trials):
        raise ValueError("trial IDs must be unique")
    owned = sorted((trial.owned_path, trial.trial_id) for trial in trials)
    for index, (path_a, id_a) in enumerate(owned):
        for path_b, id_b in owned[index + 1 :]:
            if (
                path_a == path_b
                or path_b.startswith(path_a + "/")
                or path_a.startswith(path_b + "/")
            ):
                raise ValueError(f"owned paths overlap: {id_a} and {id_b}")
    known_ids = {trial.trial_id for trial in trials}
    queues = {}
    for queue, members in (data.get("queues") or {}).items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", str(queue)):
            raise ValueError(f"invalid queue name: {queue!r}")
        if (
            not isinstance(members, list)
            or not members
            or any(not isinstance(member, str) for member in members)
        ):
            raise ValueError(f"{queue}: queue must be a non-empty ID list")
        if len(set(members)) != len(members):
            raise ValueError(f"{queue}: duplicate trial IDs")
        unknown = set(members) - known_ids
        if unknown:
            raise ValueError(
                f"{queue}: unknown trial IDs: {', '.join(sorted(unknown))}"
            )
        queues[str(queue)] = tuple(members)
    return SearchPlan(
        name,
        base_ref,
        contexts,
        tuple(trials),
        queues,
        path,
        proposal_wave=proposal_wave,
        sha256=hashlib.sha256(plan_bytes).hexdigest(),
    )


def _plan_problems(plan, repo_root):
    """The checks `check` cannot make by reading the YAML alone.

    load_plan validates properties of the text. These need the checkout, and every one
    of them used to surface only at launch, after the worktrees had been made -- a plan
    could pass `check` cleanly and still have nowhere to run.
    """
    repo_root = Path(repo_root)
    problems = []

    def at_base(relative):
        return (
            subprocess.run(
                ["git", "cat-file", "-e", f"{plan.base_ref}:{relative}"],
                cwd=repo_root,
                capture_output=True,
            ).returncode
            == 0
        )

    if (
        subprocess.run(
            ["git", "rev-parse", "--verify", f"{plan.base_ref}^{{commit}}"],
            cwd=repo_root,
            capture_output=True,
        ).returncode
        != 0
    ):
        return [f"base_ref does not resolve to a commit: {plan.base_ref}"]
    for relative in plan.context_files:
        # Prompt rendering reads these from the live checkout, not from base_ref.
        if not (repo_root / relative).is_file():
            problems.append(f"context file missing from the checkout: {relative}")
    for trial in plan.trials:
        # _task_classes turns an owned module into an import path by taking it relative
        # to reasoning_core/tasks. Anywhere else and the contract audit imports nothing.
        if not trial.owned_path.startswith("reasoning_core/tasks/"):
            problems.append(
                f"{trial.trial_id}: owned_path is outside"
                f" reasoning_core/tasks: {trial.owned_path}"
            )
        if trial.parent and not at_base(trial.parent):
            problems.append(
                f"{trial.trial_id}: parent not at {plan.base_ref}: {trial.parent}"
            )
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
    # The whole import closure of the worker's self-check, not just its entry point:
    # selfcheck imports validation, which imports sandbox and implementor_prompt, and a
    # worker silently runs the base_ref copy of every one of them.
    for relative in (
        "reasoning_core/task_search/selfcheck.py",
        "reasoning_core/task_search/validation.py",
        "reasoning_core/task_search/sandbox.py",
        "reasoning_core/task_search/implementor_prompt.py",
        "reasoning_core/task_search/prior_audit.py",
    ):
        live = (Path(repo_root) / relative).read_bytes()
        try:
            pinned = subprocess.check_output(
                ["git", "show", f"{base_ref}:{relative}"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            problems.append(
                f"{base_ref} has no {relative}: workers cannot run it at all"
            )
            continue
        if pinned != live:
            problems.append(f"{relative} at {base_ref} differs from the working tree")
    if problems:
        return (
            "workers are judged by code they cannot see. Move base_ref forward.\n  "
            + "\n  ".join(problems)
        )
    return ""


def _select_trials(plan, trial_ids=(), queue_names=()):
    unknown_trials = set(trial_ids) - {trial.trial_id for trial in plan.trials}
    if unknown_trials:
        raise ValueError(f"unknown trial IDs: {', '.join(sorted(unknown_trials))}")
    unknown_queues = set(queue_names) - set(plan.queues)
    if unknown_queues:
        raise ValueError(f"unknown queues: {', '.join(sorted(unknown_queues))}")
    selected_ids = set(trial_ids)
    for queue in queue_names:
        selected_ids.update(plan.queues[queue])
    if not selected_ids:
        return list(plan.trials)
    return [trial for trial in plan.trials if trial.trial_id in selected_ids]
