"""Process isolation and resource controls for task-search runs."""

import json
import os
from pathlib import Path
import shutil
import subprocess


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _resource_command(command, resource_limits):
    if not resource_limits.get("enabled"):
        return command
    return [
        resource_limits["executable"],
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "-p",
        f"MemoryMax={resource_limits['memory_max']}",
        "-p",
        f"TasksMax={resource_limits['tasks_max']}",
        "-p",
        f"CPUQuota={resource_limits['cpu_quota']}",
        "--",
        *command,
    ]


def _resolve_resource_limits(
    mode,
    *,
    systemd_run_bin="systemd-run",
    memory_max="8G",
    tasks_max=512,
    cpu_quota="400%",
):
    if mode == "none":
        return {"enabled": False, "mode": mode}
    executable = shutil.which(systemd_run_bin)
    true_executable = shutil.which("true")
    error = None
    if executable and true_executable:
        probe = subprocess.run(
            [
                executable,
                "--user",
                "--scope",
                "--quiet",
                "--collect",
                "-p",
                f"MemoryMax={memory_max}",
                "-p",
                f"TasksMax={tasks_max}",
                "-p",
                f"CPUQuota={cpu_quota}",
                "--",
                true_executable,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if probe.returncode == 0:
            version = subprocess.check_output(
                [executable, "--version"], text=True
            ).splitlines()[0]
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
        "enabled",
        "mode",
        "name",
        "version",
        "memory_max",
        "tasks_max",
        "cpu_quota",
    }
    return {key: value for key, value in resource_limits.items() if key in fields}


def _sanitized_environment(credential_env_names=()):
    environment = dict(os.environ)
    for name in credential_env_names:
        environment.pop(name, None)
    return environment


def _sandbox_command(
    command,
    *,
    worktree,
    owned_path,
    runtime_root,
    bwrap_bin="bwrap",
    writable_overlays=(),
):
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
                f"{label} cannot be under /tmp because strict runs hide host /tmp: {path}"
            )
        if path == Path("/run") or Path("/run") in path.parents:
            raise ValueError(
                f"{label} cannot be under /run because strict runs hide host /run: {path}"
            )
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
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/",
        "/",
        # Do not expose host daemon and desktop sockets. Read-only socket files
        # can still be connected to, so a read-only root alone is insufficient.
        "--tmpfs",
        "/run",
        "--tmpfs",
        "/tmp",
        # Bun/OpenCode needs live device and proc mounts. Replacing the
        # read-only recursive binds also avoids a Bun startup crash.
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--bind",
        str(owned),
        str(owned),
        "--bind",
        str(runtime_root),
        str(runtime_root),
        "--chdir",
        str(worktree),
    ]
    for source, target in writable_overlays:
        source = Path(source).resolve()
        target = Path(target).resolve()
        if runtime_root != source and runtime_root not in source.parents:
            raise ValueError(f"writable overlay is outside runtime root: {source}")
        if not source.exists() or not target.exists():
            raise ValueError(
                f"writable overlay endpoint does not exist: {source} -> {target}"
            )
        wrapped.extend(("--bind", str(source), str(target)))
    for name, value in runtime_dirs.items():
        wrapped.extend(("--setenv", name, str(value)))
    wrapped.extend(("--setenv", "PYTHONDONTWRITEBYTECODE", "1"))
    wrapped.extend(
        ("--setenv", "TASK_SEARCH_SPEC", str(runtime_root / "trial_spec.json"))
    )
    wrapped.extend(command)
    return wrapped


def _agy_writable_overlays(runtime_root):
    """Give AGY disposable tool state without opening its authenticated home.

    AGY rewrites ``bin/agentapi`` before every terminal call. The authenticated
    config, settings, conversations, and the rest of its installation stay on the
    read-only root. Its background-task logs similarly live under ``brain``; map
    that artifact tree to the trial runtime so a worker can inspect a long command.
    """
    home = Path.home() / ".gemini" / "antigravity-cli"
    helper_target = home / "bin" / "agentapi"
    brain_target = home / "brain"
    if not helper_target.is_file():
        raise RuntimeError(f"AGY terminal helper is missing: {helper_target}")
    if not brain_target.is_dir():
        raise RuntimeError(f"AGY artifact directory is missing: {brain_target}")
    helper_source = Path(runtime_root) / "agy-agentapi"
    brain_source = Path(runtime_root) / "agy-brain"
    helper_source.touch(mode=0o700)
    brain_source.mkdir(exist_ok=True)
    return ((helper_source, helper_target), (brain_source, brain_target))


def _run_validation(
    worktree,
    commands,
    log_path,
    *,
    owned_path,
    runtime_root,
    bwrap_bin,
    resource_limits,
    timeout_seconds,
    credential_env_names=(),
):
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
            results.append(
                {
                    "command": command,
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                }
            )
            if exit_code:
                break
    return results
