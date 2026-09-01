"""Import the pre-pipeline candidate list as wave0, the reference proposal wave.

WAVE1.md holds 80 task candidates written by hand, before there was a proposer: one name
and one description each. They are a decent baseline, and a baseline is the thing the new
pipeline has been missing. Importing them as an ordinary proposal wave puts them in the
same format, the same archive and the same novelty catalog as generated waves, so a model
wave and the hand-written wave can be run through identical plan generation, identical
validation and identical scoring. If a proposed wave does not beat wave0, the proposer is
not earning its calls.

The descriptions need no rewriting: a WAVE1 description already is a one-line coverage
spec, which is what a proposal summary is now.
"""

from datetime import datetime, timezone
from pathlib import Path
import re

import yaml

from .wave_proposer import (
    _one_line,
    _snake,
    build_catalog,
    catalog_record,
    proposal_problems,
)


LEGACY_SOURCE = "reasoning_core/task_search/WAVE1.md"


def _fenced_yaml(text):
    match = re.search(r"```yaml\n(.*?)\n```", text, re.S)
    if not match:
        raise ValueError("no ```yaml block found in the legacy candidate list")
    return yaml.safe_load(match.group(1)) or {}


def read_legacy_candidates(repo_root, source=LEGACY_SOURCE):
    """Read (name, summary, origin_id) triples from the hand-written candidate list."""
    path = Path(repo_root) / source
    document = _fenced_yaml(path.read_text())
    candidates = []
    for entry in document.get("tasks", ()):
        name = _snake(entry.get("name"))
        summary = _one_line(entry.get("description"))
        if name and summary:
            candidates.append((name, summary, str(entry.get("id", "")).strip()))
    if not candidates:
        raise ValueError(f"{source} lists no usable candidates")
    return candidates


def build_legacy_wave(repo_root, *, name="wave0", source=LEGACY_SOURCE):
    """Build a proposal wave from the hand-written list, making no model calls."""
    repo_root = Path(repo_root).resolve()
    catalog = build_catalog(repo_root)
    proposals, rejected, seen = [], [], set()
    for candidate_name, summary, origin in read_legacy_candidates(repo_root, source):
        proposal = {"name": candidate_name, "summary": summary}
        problems = proposal_problems(proposal)
        if candidate_name in seen:
            problems = ["duplicate name in the legacy list"]
        if problems:
            rejected.append({"name": candidate_name, "verdict": "invalid",
                             "closest_id": None, "reason": "; ".join(problems)})
            continue
        seen.add(candidate_name)
        proposal["id"] = f"P{len(proposals) + 1:03d}"
        # Imported, not reviewed: say so in the record rather than fabricating a verdict
        # that no critic ever returned.
        proposal["novelty"] = {"source": "legacy", "verdict": "imported",
                               "origin_id": origin or candidate_name}
        proposals.append(proposal)
    return {
        "format_version": 1,
        "kind": "sft_task_proposals",
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": {"training_stage": "sft", "requested": len(proposals),
                      "accepted": len(proposals), "complete": True},
        "catalog": catalog_record(catalog),
        "generation": {"provider": "legacy", "source": source, "calls": []},
        "proposals": proposals,
        "rejected": rejected,
    }
