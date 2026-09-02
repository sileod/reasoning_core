"""Turn a proposal wave into an executable search plan.

A proposal is one line, so one proposal does not determine one implementation -- it leaves
the instance family, the generator, the verifier and the difficulty ladder open on purpose.
Rather than pretending a proposer can settle those on paper, this fans one proposal out into
`variants` independent trials carrying the identical instruction. The runner derives each
trial's seed from sha256(base_seed:trial_id), so identical instructions land on different
sampling and the wave produces several honest attempts at the same summary. Validation then
says which attempts survived, which is evidence rather than speculation.

Plan generation stays a separate command from proposal generation: proposals are reviewed by
a person before any model is paid to implement them.
"""

from pathlib import Path
import re

import yaml

from .wave_proposer import _snake


DEFAULT_CONTEXT_FILES = (
    "AGENTS.md",
    "TASK_AUTHORING_GUIDE.md",
    "reasoning_core/task_search/CONTEXT.md",
    "reasoning_core/task_search/AUTHORING_RULES.md",
)
PILOT_SIZE = 6
VALIDATION_COMMAND = (
    "PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider"
    " --import-mode=importlib {owned_path}"
)


def _instruction(task_name, summary):
    return (
        f"Author a new task from scratch: {task_name}.\n\n"
        "Cover exactly this, and put this line -- adapted to what you actually build -- in"
        f" the class `summary`:\n\n  {summary}\n\n"
        "Every decision the summary does not fix is yours: instance family, generator,"
        " verifier, difficulty scaling, prompt wording and answer format. Other trials in"
        " this wave are implementing the same summary independently and are not coordinating"
        " with you. Make the choices you believe produce the best training data.\n\n"
        "AUTHORING_RULES.md lists the checks generate_entry must perform and the failure"
        " modes that sank earlier waves -- read it before writing code, and treat it as part"
        " of this instruction."
    )


def build_plan(wave, *, name, base_ref="HEAD", variants=1,
               context_files=DEFAULT_CONTEXT_FILES, design_choices=None):
    """Build a task-search plan running every proposal in `wave` `variants` times.

    `design_choices` maps a proposal id to the approaches its variants should be split
    across, as `design_proposer` returns them. Pass none and the variants differ only by
    seed, which is what every wave so far has done.
    """
    if not isinstance(wave, dict) or wave.get("kind") != "sft_task_proposals":
        raise ValueError("expected an SFT proposal wave")
    # The wave name becomes a package directory under reasoning_core/tasks/generated, and
    # the contract audit imports candidates by module path. A dash there is not importable.
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("plan name must be a lowercase Python identifier")
    if variants < 1:
        raise ValueError("variants must be positive")
    proposals = wave.get("proposals") or []
    if not proposals:
        raise ValueError("proposal wave has no accepted proposals")
    design_choices = design_choices or {}
    trials, queues = [], {f"v{index}": [] for index in range(1, variants + 1)}
    for proposal in proposals:
        base_name = _snake(proposal.get("name"))
        summary = str(proposal.get("summary", "")).strip()
        if not base_name or not summary:
            raise ValueError(f"proposal {proposal.get('id', '?')} needs a name and a summary")
        # One choice per variant, so a proposal fanned three ways is three named
        # approaches and not one approach drawn three times.
        choices = tuple(design_choices.get(str(proposal.get("id", "")), ()))
        if choices and len(choices) != variants:
            raise ValueError(
                f"proposal {proposal.get('id', '?')} has {len(choices)} design choices"
                f" for {variants} variants: they have to match"
            )
        for index in range(1, variants + 1):
            task_name = f"{base_name}_v{index}" if variants > 1 else base_name
            trial_id = f"{proposal.get('id', base_name)}v{index}"
            owned_path = f"reasoning_core/tasks/generated/{name}/{task_name}"
            trials.append({
                "id": trial_id,
                "hypothesis": str((proposal.get("novelty") or {}).get("origin_id", "")
                                  or proposal.get("id", "")),
                "idea": f"{task_name} (draw {index} of {variants})",
                "changes": f"new task in {owned_path}",
                "instruction": _instruction(task_name, summary),
                "owned_path": owned_path,
                "validation": [VALIDATION_COMMAND.format(owned_path=owned_path)],
                **({"design_choice": choices[index - 1]} if choices else {}),
            })
            queues[f"v{index}"].append(trial_id)
    # One cheap draw to run first: the same six proposals every time, so a smoke run of two
    # waves is comparable and a broken harness costs six trials instead of the whole wave.
    queues["pilot"] = queues["v1"][:PILOT_SIZE]
    return {
        "version": 1,
        "name": name,
        "proposal_wave": wave.get("name", ""),
        "defaults": {"base_ref": base_ref},
        "context_files": list(context_files),
        "queues": {key: value for key, value in queues.items() if value},
        "trials": trials,
    }


def _block_string(dumper, value):
    """Dump instructions as literal blocks: these files get read by people."""
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


class _PlanDumper(yaml.SafeDumper):
    pass


_PlanDumper.add_representer(str, _block_string)


def write_plan(path, plan):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite plan: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.dump(plan, Dumper=_PlanDumper, sort_keys=False, width=100))
    temporary.replace(path)
