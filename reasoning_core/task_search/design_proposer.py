"""Different ways to implement one task summary, proposed per task by a cheap model.

A proposal is deliberately thin -- a name and one sentence -- so the implementer decides
everything else: what the inputs look like, what the answer is, where the difficulty
comes from. Running a summary twice therefore samples that decision twice without ever
recording it, and 21 of the 80 external proposals had one variant pass while its twin
failed. The wave learns that implementation is noisy; it cannot learn which choice won.

This asks a small model for N genuinely different approaches to one summary, so a wave
can fan its variants across named choices instead of across seeds alone. The choice each
worker was given is then in its plan, its prompt and its run record, and "what worked"
becomes a question the records can answer.

Off unless a plan asks for it: `build_plan` without design choices produces the trials it
always did, and a trial without one renders the prompt it always did, byte for byte.
"""
import argparse
import json
import os
from pathlib import Path

from .wave_proposer import ChatClient

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_ENDPOINT = "https://albert.api.etalab.gouv.fr/v1/chat/completions"
DEFAULT_API_KEY_ENV = "ALBERT_API_KEY"
# One sentence is not enough to constrain an implementation and a paragraph starts
# writing the code for the worker, which is the opposite of what the search is for.
# One line, and short enough to read beside its siblings. A choice now travels into the
# task as a class attribute, so it is read far more often than it is written: a paragraph
# there is a paragraph in every module built under it.
CHOICE_CHARS = 240
CHOICE_MIN_CHARS = 40
CHOICE_ATTEMPTS = 3

SYSTEM = """You design reasoning tasks for a machine-learning benchmark.

You are given one task summary. Propose distinct ways to IMPLEMENT it. A design choice
names a decision the summary leaves open -- what the instance looks like, what form the
answer takes, where the difficulty comes from, what a solver must do to get it right.

Rules:
- The choices must be mutually exclusive. Two phrasings of the same idea are one choice.
- Each choice must be implementable on its own, with no reference to the others.
- Describe the approach, not the code. Name no Python identifiers, files or libraries.
- Every choice must satisfy the summary. Do not propose a different task.
- Follow the worker's answer contract: compact, canonical strings, lists, fractions,
  or integers. No float or JSON answers, free-form prose, or open-ended semantic
  paraphrase grading. A tool call can be a constrained canonical call string.
- Answers must vary within a level. Balanced yes/no and small label sets are fine --
  the gameability gate scores the excess over the 1/k floor, not the raw rate -- but
  a level that emits one answer fails.

Reply with JSON only: {"choices": ["...", "..."]}"""


def _prompt(name, summary, count):
    return (
        f"TASK NAME: {name}\n"
        f"TASK SUMMARY: {summary}\n\n"
        f"Propose exactly {count} distinct design choices for implementing this task.\n"
        f"Each choice: ONE line, {CHOICE_MIN_CHARS}-{CHOICE_CHARS} characters, no line breaks."
    )


def propose_design_choices(name, summary, count, *, client=None, **client_options):
    """The `count` approaches this summary could be implemented under.

    Raises rather than returning fewer: a wave that quietly fans three variants across
    two choices would run one choice twice and report it as two independent draws.
    """
    if count < 1:
        raise ValueError("count must be positive")
    if not summary.strip():
        raise ValueError(f"{name}: a design proposer needs a summary to work from")
    client = client or ChatClient(**client_options)
    cleaned = []
    # One overlong choice used to cost the whole wave: the call returns all `count` at
    # once, so a single reply that runs past one line leaves a shortfall, and the raise
    # below aborts a plan covering every other proposal too. Ask again instead, keeping
    # what already passed, and only give up when the model cannot do it three times.
    for attempt in range(CHOICE_ATTEMPTS):
        reply = client.json(f"design-{name}-{attempt}" if attempt else f"design-{name}",
                            SYSTEM, _prompt(name, summary, count))
        choices = reply.get("choices")
        if not isinstance(choices, list):
            raise ValueError(f"{name}: design proposer response requires a choices list")
        for choice in choices:
            # Collapsing whitespace is what makes it one line; truncating mid-sentence
            # would leave a dangling clause in the task, so an overlong choice is dropped.
            text = " ".join(str(choice).split()).strip()
            if not CHOICE_MIN_CHARS <= len(text) <= CHOICE_CHARS:
                continue
            # Case-insensitive: the model repeats itself in different capitalisation more
            # often than it repeats itself exactly.
            if text.lower() not in {seen.lower() for seen in cleaned}:
                cleaned.append(text)
        if len(cleaned) >= count:
            return tuple(cleaned[:count])
    raise ValueError(
        f"{name}: asked for {count} distinct design choices, got {len(cleaned)} "
        f"in {CHOICE_ATTEMPTS} attempts"
    )


def propose_wave_design_choices(wave, count, *, client=None, **client_options):
    """One design-proposer call per proposal, keyed by proposal id."""
    client = client or ChatClient(**client_options)
    return {
        str(proposal["id"]): propose_design_choices(
            proposal["name"], proposal["summary"], count, client=client
        )
        for proposal in wave.get("proposals") or []
    }


def main(argv=None):
    import yaml

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("proposal_wave", type=Path)
    parser.add_argument("--count", type=int, default=2,
                        help="design choices per task; match it to --variants")
    parser.add_argument("--output", type=Path, help="write the choices as YAML")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is required for the design proposer")
    wave = yaml.safe_load(args.proposal_wave.read_text())
    choices = propose_wave_design_choices(
        wave, args.count, model=args.model, endpoint=args.endpoint, api_key=api_key,
        temperature=0.7, reasoning_effort=None,
    )
    if args.output:
        args.output.write_text(yaml.safe_dump(
            {"proposal_wave": wave.get("name", ""), "count": args.count,
             "choices": {key: list(value) for key, value in choices.items()}},
            sort_keys=False, width=100))
        print(f"{args.output}: {len(choices)} tasks x {args.count} choices")
    else:
        print(json.dumps({k: list(v) for k, v in choices.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
