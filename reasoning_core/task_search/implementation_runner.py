"""Plan-driven, folder-scoped task-search workers."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from ..source_store import SourceStore
from .implementor_prompt import (
    DEFAULT_PACE,
    PACE,
    _prior_audit_command,
    _sample_command,
    _selfcheck_command,
    render_implementor_prompt,
)
from .plan import _frozen_module_drift, _plan_problems, _select_trials, load_plan
from .sandbox import (
    _agy_writable_overlays,
    _public_resource_limits,
    _resolve_resource_limits,
    _resource_command,
    _sandbox_command,
    _write_json,
)
from .validation import _step_usage, validate_candidate


def _repo_root(start):
    output = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=start, text=True
    )
    return Path(output.strip()).resolve()


def _sha256(data):
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def generation_metadata(
    model,
    harness_version,
    agent,
    variant=None,
    requested_seed=None,
    seed_forwarded=False,
    temperature=None,
    top_p=None,
    sandbox_name="bubblewrap",
    sandbox_version=None,
    max_steps=56,
    timeout_seconds=1800,
    provider_name=None,
    harness_name="opencode",
):
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
        "harness_name": harness_name,
        "harness_version": harness_version,
        "agent_name": agent,
        "settings": settings,
    }


# Every env prefix a worker has been observed to type in front of a command the
# harness itself prescribed. Matching is prefix-anchored, so each spelling is a
# separate pattern or it is a denial.
_ENV_PREFIXES = (
    "",
    "PYTHONDONTWRITEBYTECODE=1 ",
    "PYTHONPATH=. ",
    "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. ",
)


def _spellings(command, owned_path):
    """The patterns that allow one prescribed command however the worker writes it.

    Two things vary and neither changes what the command does: the env prefix, and the
    flags after the owned path. Both were costing whole trials -- wave7's T008 declared
    itself finished with its prior_audit never run, because every spelling of it that the
    worker tried was denied. Truncating at the owned path keeps the wildcard from
    widening the command to a different target.
    """
    body = command
    for prefix in sorted(_ENV_PREFIXES, key=len, reverse=True):
        if prefix and body.startswith(prefix):
            body = body[len(prefix):]
            break
    heads = [body, body + "*"]
    cut = body.find(owned_path)
    if cut >= 0:
        heads.append(body[: cut + len(owned_path)] + "*")
    return [prefix + head for prefix in _ENV_PREFIXES for head in heads]


def opencode_permissions(trial):
    bash = {
        "*": "deny",
        "git status*": "allow",
        "git diff*": "allow",
        # Read-only history. Denying it wasted turns and it exposes no file content
        # that the already-allowed read tool does not.
        "git log*": "allow",
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
    for command in list(trial.validation) + [
        _sample_command(trial),
        _selfcheck_command(trial),
        _prior_audit_command(trial),
    ]:
        for pattern in _spellings(command, trial.owned_path):
            bash[pattern] = "allow"
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


def opencode_config(
    trial,
    agent,
    *,
    requested_seed=None,
    forward_seed=False,
    temperature=None,
    top_p=None,
    max_steps=56,
):
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


def _mini_config(
    worktree,
    *,
    max_steps,
    timeout_seconds,
    requested_seed=None,
    forward_seed=False,
    temperature=None,
    top_p=None,
):
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


def _prepare_harness(
    hlink,
    harness,
    *,
    worktree,
    prompt,
    model,
    provider,
    agent,
    variant,
    config_path,
    trajectory_path,
    timeout_seconds,
    agy_log_path,
):
    """Build one normalized Harness Link launch plus task-search native options."""
    command = [
        hlink,
        harness,
        "-C",
        str(worktree),
        "-p",
        "-" if harness == "agy" else prompt,
        "-m",
        model,
        "-y",
    ]
    if provider:
        command.extend(("--provider", provider))
    native = []
    if harness == "opencode":
        native = ["--pure", "--agent", agent, "--format", "json"]
        if variant:
            native.extend(("--variant", variant))
    elif harness == "mini":
        native = [
            "-c",
            "mini.yaml",
            "-c",
            str(config_path),
            "--exit-immediately",
            "-o",
            str(trajectory_path),
        ]
    elif harness == "agy":
        native = [
            "--mode",
            "accept-edits",
            "--new-project",
            "--add-dir",
            str(worktree),
            "--output-format",
            "stream-json",
            "--log-file",
            str(agy_log_path),
            "--print-timeout",
            f"{timeout_seconds}s",
            "--disable-slash-commands",
        ]
    else:
        raise ValueError(f"unsupported harness: {harness}")
    return [*command, "--", *native]


def _run_trial(
    plan,
    trial,
    repo_root,
    invocation_root,
    base_commit,
    model,
    harness,
    agent,
    variant,
    hlink_bin,
    hlink_version,
    base_seed,
    forward_seed,
    temperature,
    top_p,
    bwrap_bin,
    sandbox_version,
    max_steps,
    timeout_seconds,
    provider,
    resource_limits,
    validation_timeout_seconds,
    credential_env_names,
    pace=DEFAULT_PACE,
):
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
    generation_agent = {
        "opencode": agent,
        "mini": "mini-default",
        "agy": "agy-default",
    }[harness]
    effective_max_steps = None if harness == "agy" else max_steps
    generation = generation_metadata(
        model,
        None,
        generation_agent,
        variant,
        requested_seed=requested_seed,
        seed_forwarded=(forward_seed and harness != "agy"),
        temperature=(None if harness == "agy" else temperature),
        top_p=(None if harness == "agy" else top_p),
        sandbox_version=sandbox_version,
        max_steps=effective_max_steps,
        timeout_seconds=timeout_seconds,
        provider_name=("antigravity" if harness == "agy" else provider),
        harness_name=harness,
    )
    parent_source_id = None
    if trial.parent:
        parent_source = subprocess.check_output(
            ["git", "show", f"{base_commit}:{trial.parent}"],
            cwd=repo_root,
        ).decode("utf-8")
        parent_source_id = SourceStore(repo_root / ".evolution" / "objects").put(
            parent_source
        )
    task_meta = {
        "parent_source_id": parent_source_id,
        "idea": trial.idea,
        "hypothesis": trial.hypothesis,
        "changes": trial.changes,
        "generation": generation,
    }
    prompt = render_implementor_prompt(plan, trial, repo_root, task_meta, pace)
    prompt_path = trial_root / "prompt.md"
    prompt_path.write_text(prompt)
    runtime_root = trial_root / "runtime"
    runtime_root.mkdir()
    # Out of the worktree: anything inside it would count as a changed path and the
    # trial would lose on scope_violation.
    _write_json(
        runtime_root / "trial_spec.json",
        {
            "trial_id": trial.trial_id,
            "owned_path": trial.owned_path,
            "task_meta": task_meta,
        },
    )
    started = datetime.now(timezone.utc).isoformat()
    if harness == "opencode":
        config_path = trial_root / "opencode.json"
        _write_json(
            config_path,
            opencode_config(
                trial,
                agent,
                requested_seed=requested_seed,
                forward_seed=forward_seed,
                temperature=temperature,
                top_p=top_p,
                max_steps=max_steps,
            ),
        )
        events_path = trial_root / "events.jsonl"
        trajectory_path = None
    elif harness == "mini":
        config_path = trial_root / "mini.yaml"
        config_path.write_text(
            yaml.safe_dump(
                _mini_config(
                    worktree,
                    max_steps=max_steps,
                    timeout_seconds=timeout_seconds,
                    requested_seed=requested_seed,
                    forward_seed=forward_seed,
                    temperature=temperature,
                    top_p=top_p,
                ),
                sort_keys=False,
            )
        )
        trajectory_path = runtime_root / "trajectory.json"
        events_path = trial_root / "harness.log"
    elif harness == "agy":
        config_path = None
        trajectory_path = None
        events_path = trial_root / "events.jsonl"
    else:
        raise ValueError(f"unsupported harness: {harness}")
    command = _prepare_harness(
        hlink_bin,
        harness,
        worktree=worktree,
        prompt=prompt,
        model=model,
        provider=provider,
        agent=agent,
        variant=variant,
        config_path=config_path,
        trajectory_path=trajectory_path,
        timeout_seconds=timeout_seconds,
        agy_log_path=runtime_root / "agy.log",
    )
    command = _sandbox_command(
        command,
        worktree=worktree,
        owned_path=trial.owned_path,
        runtime_root=runtime_root,
        bwrap_bin=bwrap_bin,
        writable_overlays=(
            _agy_writable_overlays(runtime_root) if harness == "agy" else ()
        ),
    )
    command = _resource_command(command, resource_limits)
    environment = dict(os.environ)
    if harness == "opencode":
        environment["OPENCODE_CONFIG_CONTENT"] = config_path.read_text()
        environment["OPENCODE_DISABLE_EXTERNAL_SKILLS"] = "true"
        environment["OPENCODE_DISABLE_CLAUDE_CODE_SKILLS"] = "true"
    elif harness == "mini":
        environment["MSWEA_CONFIGURED"] = "true"
    timed_out = False
    with (
        events_path.open("w") as stdout,
        (trial_root / "stderr.log").open("w") as stderr,
        prompt_path.open() if harness == "agy" else open(os.devnull) as stdin,
    ):
        try:
            completed = subprocess.run(
                command,
                env=environment,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_seconds,
            )
            harness_exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            harness_exit_code = 124
    audit = validate_candidate(
        worktree,
        trial,
        events_path,
        harness_exit_code,
        timed_out,
        requested_seed,
        trial_root,
        task_meta=task_meta,
        bwrap_bin=bwrap_bin,
        resource_limits=resource_limits,
        timeout_seconds=validation_timeout_seconds,
        credential_env_names=credential_env_names,
    )
    record = {
        "schema_version": 1,
        "wave": plan.name,
        "proposal_wave": plan.proposal_wave,
        "trial_id": trial.trial_id,
        "hypothesis": trial.hypothesis,
        "design_choice": trial.design_choice,
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
        "steps": _step_usage(events_path, effective_max_steps),
        "launcher": {"name": "hlink", "version": hlink_version},
        "trajectory": str(trajectory_path) if trajectory_path else None,
        "timed_out": timed_out,
        "sandbox": {"name": "bubblewrap", "version": sandbox_version},
        "resource_limits": _public_resource_limits(resource_limits),
        "scrubbed_credential_env_names": sorted(credential_env_names),
        **audit,
        "worktree": str(worktree),
    }
    _write_json(trial_root / "run.json", record)
    return record


_TRANSIENT_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _retryable_harness_failure(result):
    """Return a stable retry reason for infrastructure failures, never gate failures."""
    if result.get("status") != "harness_failed":
        return None
    exit_code = result.get("harness_exit_code") or 0
    if exit_code < 0:
        return f"signal_{-exit_code}"
    log_path = result.get("harness_log")
    if not log_path:
        return None
    try:
        lines = Path(log_path).read_text().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "error":
            continue
        data = event.get("error", {}).get("data", {})
        status = data.get("statusCode")
        if data.get("isRetryable") is True or status in _TRANSIENT_HTTP_STATUS:
            return f"provider_{status or 'transient'}"
    return None


def run_plan(
    plan_path,
    *,
    model,
    jobs=1,
    trial_ids=(),
    agent="task-search-worker",
    variant=None,
    seed=0,
    forward_seed=True,
    temperature=None,
    top_p=None,
    harness="opencode",
    hlink_bin="hlink",
    bwrap_bin="bwrap",
    runs_root=None,
    repo_root=None,
    max_steps=56,
    timeout_seconds=1800,
    queue_names=(),
    provider=None,
    resource_limit_mode="auto",
    systemd_run_bin="systemd-run",
    memory_max="8G",
    tasks_max=512,
    cpu_quota="400%",
    validation_timeout_seconds=300,
    credential_env_names=(),
    pace=DEFAULT_PACE,
    transient_retries=2,
    retry_backoff_seconds=30,
):
    """Run selected trials concurrently in isolated Git worktrees."""
    if pace not in PACE:
        raise ValueError(
            f"unknown pace: {pace!r}; choose from {', '.join(sorted(PACE))}"
        )
    if transient_retries < 0 or retry_backoff_seconds < 0:
        raise ValueError("retry count and backoff must be non-negative")
    plan = load_plan(plan_path)
    repo_root = Path(repo_root).resolve() if repo_root else _repo_root(plan.path.parent)
    selected = _select_trials(plan, trial_ids, queue_names)
    base_commit = subprocess.check_output(
        ["git", "rev-parse", plan.base_ref], cwd=repo_root, text=True
    ).strip()
    problems = _plan_problems(plan, repo_root)
    if problems:
        raise ValueError("plan cannot run:\n  " + "\n  ".join(problems))
    drift = _frozen_module_drift(repo_root, base_commit)
    if drift:
        print(f"WARNING: {drift}", file=sys.stderr)
    if harness not in {"opencode", "mini", "agy"}:
        raise ValueError(f"unsupported harness: {harness}")
    resolved_hlink = shutil.which(hlink_bin)
    if resolved_hlink is None:
        raise RuntimeError(f"Harness Link frontend not found: {hlink_bin!r}")
    hlink_version = subprocess.check_output(
        [resolved_hlink, "--version"], text=True
    ).strip()
    hlink_help = subprocess.check_output(
        [resolved_hlink, "--help"], text=True, stderr=subprocess.STDOUT
    )
    if harness not in hlink_help:
        raise RuntimeError(
            f"installed Harness Link does not support the {harness!r} harness"
        )
    bwrap_path = shutil.which(bwrap_bin)
    if bwrap_path is None:
        raise RuntimeError(
            f"bubblewrap executable not found: {bwrap_bin!r}; "
            "strict task-search runs require bubblewrap"
        )
    sandbox_version = subprocess.check_output(
        [bwrap_path, "--version"], text=True
    ).strip()
    resource_limits = _resolve_resource_limits(
        resource_limit_mode,
        systemd_run_bin=systemd_run_bin,
        memory_max=memory_max,
        tasks_max=tasks_max,
        cpu_quota=cpu_quota,
    )
    root = (
        Path(runs_root).resolve()
        if runs_root
        else repo_root.parent / f".{repo_root.name}-task-search"
    )
    invocation = (
        root / plan.name / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    invocation.mkdir(parents=True)
    results = []

    def write_summary():
        _write_json(
            invocation / "summary.json",
            {
                "wave": plan.name,
                "proposal_wave": plan.proposal_wave,
                "queues": list(queue_names),
                "base_commit": base_commit,
                "model": model,
                "harness": {"name": harness},
                "launcher": {"name": "hlink", "version": hlink_version},
                "provider": (
                    "antigravity"
                    if harness == "agy"
                    else provider or model.split("/", 1)[0]
                ),
                "seed": seed,
                "seed_forwarded": forward_seed and harness != "agy",
                "max_steps": None if harness == "agy" else max_steps,
                "timeout_seconds": timeout_seconds,
                "validation_timeout_seconds": validation_timeout_seconds,
                "pace": pace,
                "transient_retries": transient_retries,
                "retry_backoff_seconds": retry_backoff_seconds,
                "sandbox": {"name": "bubblewrap", "version": sandbox_version},
                "resource_limits": _public_resource_limits(resource_limits),
                "scrubbed_credential_env_names": sorted(credential_env_names),
                "results": sorted(results, key=lambda item: item["trial_id"]),
            },
        )

    def run_trial_retrying(*arguments):
        """Retry only explicit provider transients and killed harness processes."""
        history = []
        provider_retries = signal_retries = 0
        while True:
            result = _run_trial(*arguments)
            reason = _retryable_harness_failure(result)
            is_signal = bool(reason and reason.startswith("signal_"))
            retry = reason is not None and (
                (is_signal and signal_retries < 1)
                or (not is_signal and provider_retries < transient_retries)
            )
            if not retry:
                if history:
                    result["retry_history"] = history
                return result
            if is_signal:
                signal_retries += 1
            else:
                provider_retries += 1
            history.append(
                {
                    "reason": reason,
                    "status": result.get("status"),
                    "harness_exit_code": result.get("harness_exit_code"),
                }
            )
            trial_root = arguments[3] / arguments[1].trial_id
            archived = trial_root.with_name(
                f"{trial_root.name}.attempt{len(history)}-{reason}"
            )
            trial_root.rename(archived)
            # The detached worktree moved with its trial directory. Repair Git's
            # administrative pointer before recreating the canonical trial path;
            # otherwise `git worktree add` reports a missing-but-registered path.
            subprocess.run(
                ["git", "worktree", "repair", str(archived / "worktree")],
                cwd=arguments[2],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            if not is_signal and retry_backoff_seconds:
                delay = min(60, retry_backoff_seconds * (2 ** (provider_retries - 1)))
                time.sleep(delay)

    write_summary()
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {
            pool.submit(
                run_trial_retrying,
                plan,
                trial,
                repo_root,
                invocation,
                base_commit,
                model,
                harness,
                agent,
                variant,
                resolved_hlink,
                hlink_version,
                seed,
                forward_seed,
                temperature,
                top_p,
                bwrap_path,
                sandbox_version,
                max_steps,
                timeout_seconds,
                provider,
                resource_limits,
                validation_timeout_seconds,
                tuple(credential_env_names),
                pace,
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
                    "proposal_wave": plan.proposal_wave,
                    "trial_id": trial_id,
                    "status": "orchestration_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            results.append(result)
            write_summary()
    return sorted(results, key=lambda item: item["trial_id"])
