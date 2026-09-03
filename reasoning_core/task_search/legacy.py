"""Import the pre-pipeline candidate list as `external`, the reference proposal wave.

WAVE1.md holds 80 task candidates that came from outside this pipeline. Its own header says
so -- `kind: generated_only`, with GALLERY.md and WAVE0.md listed as the novelty screen it
was written against -- so some other model produced them and they were committed as a file.
No proposer here called for them and no critic here reviewed them.

The wave is named for that, not for a number. `wave0` used to mean both this and the
unrelated implementation plan in wave0.yaml, whose tasks ship under tasks/generated/wave0/.

Importing them as an ordinary proposal wave puts them in the same format, the same archive
and the same novelty catalog as generated waves, so a proposed wave and this one go through
identical plan generation, identical validation and identical scoring. It is a reference
point, but read the comparison carefully: these 80 were accepted 80-for-80 without passing
the critic a proposed wave has to pass, so beating them is not the same test.

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


LEGACY_SOURCE = "reasoning_core/task_search/plans/WAVE1.md"


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


def build_legacy_wave(repo_root, *, name="external", source=LEGACY_SOURCE):
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
