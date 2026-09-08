"""Check that this machine can actually run a wave, before a wave finds out for you.

Every credential path in this pipeline fails open, which is right: a reviewer outage must
not reject a task that is fine, and a probe that cannot reach a provider should record
`api-error` rather than crash a six-hour run. The cost is that a missing key is silent and
is discovered late. A review pass with no key reviews nothing and reports every trial
`unreviewed`; a wave with no worker key spends its whole queue on `harness_failed`. Both
have happened here, and both read as a broken pipeline rather than as an unset variable.

So the checks live in one command that runs in seconds and says which of them would have
been silent. It reports and exits non-zero; it never edits a config or a shell profile.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# Worker credentials reach the harness through a blanket copy of the environment, so what
# a provider needs is a fact about that provider, not about this repository.
PROVIDER_KEYS = {
    "albert": "ALBERT_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "openrouter": "OPENROUTER_FREE_API_KEY",
}
REVIEW_VARS = ("TASK_SEARCH_REVIEW_ENDPOINT", "TASK_SEARCH_REVIEW_MODEL",
               "TASK_SEARCH_REVIEW_KEY_ENV")
ENV_FILE = "~/.config/reasoning_core/env"


class Report:
    """PASS/WARN/FAIL lines, and the exit status is whether anything failed."""

    def __init__(self):
        self.rows = []

    def add(self, ok, name, detail, fix=""):
        self.rows.append((ok, name, detail, fix))

    def show(self):
        for ok, name, detail, fix in self.rows:
            mark = {True: "PASS", False: "FAIL", None: "WARN"}[ok]
            print(f"{mark:4}  {name:22} {detail}")
            if fix and ok is not True:
                print(f"      {'':22} -> {fix}")
        failed = [row for row in self.rows if row[0] is False]
        warned = [row for row in self.rows if row[0] is None]
        print(f"\n{len(failed)} failing, {len(warned)} warning, "
              f"{len(self.rows) - len(failed) - len(warned)} ok.")
        return 1 if failed else 0


def _binary(report, name, binary, why, version_args=("--version",)):
    found = shutil.which(binary)
    if not found:
        report.add(False, name, f"{binary} not on PATH", why)
        return None
    try:
        version = subprocess.check_output([found, *version_args], text=True,
                                          stderr=subprocess.STDOUT, timeout=30).strip()
    except (subprocess.SubprocessError, OSError) as error:
        report.add(None, name, f"{found} would not report a version: {error}")
        return found
    report.add(True, name, version.splitlines()[0])
    return found


def _ask(endpoint, model, key, timeout):
    """One tiny completion. Returns (ok, detail)."""
    body = json.dumps({"model": model, "max_tokens": 4,
                       "messages": [{"role": "user", "content": "say ok"}]}).encode()
    request = urllib.request.Request(
        endpoint, body, {"Authorization": "Bearer " + key,
                         "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            json.load(response)
        return True, f"{model} answered"
    except urllib.error.HTTPError as error:
        detail = error.read()[:200].decode("utf-8", "replace").strip()
        # The quota wall is per model and carries no rate-limit headers, so 429 is the
        # only warning anyone gets before every cell of a run comes back api-error.
        return False, f"HTTP {error.code}: {detail}"
    except Exception as error:  # noqa: BLE001 - any transport fault is the same verdict
        return False, f"unreachable: {error}"


def check(provider="albert", harness="opencode", live=False, timeout=60):
    report = Report()

    hlink = _binary(report, "harness link", "hlink",
                    "install Harness Link; it is what launches the coding worker")
    if hlink:
        try:
            help_text = subprocess.check_output([hlink, "--help"], text=True,
                                                stderr=subprocess.STDOUT, timeout=30)
            report.add(harness in help_text, f"harness {harness}",
                       "supported" if harness in help_text else "not built into this hlink",
                       f"install an hlink that supports {harness}")
        except (subprocess.SubprocessError, OSError) as error:
            report.add(None, f"harness {harness}", f"could not ask hlink: {error}")
    _binary(report, "sandbox", "bwrap",
            "install bubblewrap; strict runs refuse to start without it")
    systemd = shutil.which("systemd-run")
    report.add(True if systemd else None, "resource limits",
               "systemd-run available" if systemd else "systemd-run missing",
               "runs need --resource-limits none, which removes the memory and CPU cap")

    key_name = PROVIDER_KEYS.get(provider)
    if not key_name:
        report.add(None, f"worker key ({provider})", "unknown provider, cannot guess its key",
                   f"known providers: {', '.join(sorted(PROVIDER_KEYS))}")
    else:
        worker_key = os.environ.get(key_name, "")
        report.add(bool(worker_key), f"worker key ({provider})",
                   f"{key_name} is set" if worker_key else f"{key_name} is unset",
                   f"source {ENV_FILE} before launching; a wave without it spends its "
                   "whole queue on harness_failed")

    missing = [name for name in REVIEW_VARS if not os.environ.get(name)]
    review_key_name = os.environ.get("TASK_SEARCH_REVIEW_KEY_ENV", "")
    review_key = os.environ.get(review_key_name, "") if review_key_name else ""
    if missing or not review_key:
        report.add(False, "reviewer config",
                   f"missing {', '.join(missing)}" if missing
                   else f"{review_key_name} is named but unset",
                   f"source {ENV_FILE}; without it the semantic reviewer returns a null "
                   "verdict for every trial and land skips them all as unreviewed")
    else:
        report.add(True, "reviewer config",
                   f"{os.environ['TASK_SEARCH_REVIEW_MODEL']} via "
                   f"{os.environ['TASK_SEARCH_REVIEW_ENDPOINT']}")

    if not live:
        report.add(None, "provider reachable", "not checked",
                   "pass --live to spend one tiny completion proving the key works")
    elif not review_key:
        report.add(None, "provider reachable", "skipped, no reviewer key to try")
    else:
        ok, detail = _ask(os.environ["TASK_SEARCH_REVIEW_ENDPOINT"],
                          os.environ["TASK_SEARCH_REVIEW_MODEL"], review_key, timeout)
        report.add(ok, "provider reachable", detail,
                   "a 429 here is the daily quota, which is counted per model: another "
                   "model on the same key may still answer")
    return report


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", default="albert")
    parser.add_argument("--harness", default="opencode",
                        choices=("opencode", "mini", "agy"))
    parser.add_argument("--live", action="store_true",
                        help="spend one tiny completion proving the reviewer key works")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args(argv)
    raise SystemExit(check(provider=args.provider, harness=args.harness,
                           live=args.live, timeout=args.timeout).show())


if __name__ == "__main__":
    main(sys.argv[1:])
