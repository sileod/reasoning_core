"""Prompt construction for task-search implementor agents."""

import pprint
import textwrap

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


def render_implementor_prompt(
    plan, trial, repo_root, task_meta=None, pace=DEFAULT_PACE
):
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
        sections.extend(
            ("", f"## Global context: `{relative}`", "", source.read_text().rstrip())
        )
    sections.extend(
        (
            "",
            "## Assignment",
            "",
            trial.instruction,
            "",
            f"Hypothesis: `{trial.hypothesis or 'unassigned'}`",
            (
                f"Parent module: `{trial.parent}` -- read this one file with the read tool"
                " before you write, and reuse its machinery instead of importing a new"
                " library."
                if trial.parent
                else "Parent module: `none (new task)`"
            ),
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
                f" denied call still counts. {pacing['stance']}",
                79,
            ),
            *textwrap.wrap(
                f"1. {pacing['first_step']} under the owned path: a `Config` subclass, a"
                " `Task` subclass whose name does not contain `Task`, a literal class"
                " `summary` that packs the full task coverage into one sentence, and the exact"
                " TASK_META below, pasted rather than retyped.",
                79,
                subsequent_indent="   ",
            ),
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
            "- `summary` is a concise one-line coverage spec, not one example:",
            "  name the task's distinct modes, operations, input families and output regimes.",
            "- `score_answer` runs with a mock `self` that raises on any attribute access,",
            "  so it must not touch `self` at all: no `self._parse_interval(answer)`, no",
            "  `self.config`. Put shared parsing in a module-level function and call it",
            "  directly. S11 in wave1 lost an otherwise-passing task to one helper call.",
            "- generation must survive every level: enforce construction invariants by",
            "  resampling in a loop, never with an `assert` that only holds at level 0.",
            "- your self-check does not verify that your answer is mathematically possible;",
            "  it only checks that it is stable and hard to guess. A coordinator gate you",
            "  cannot run reads your samples after you stop and fails the trial outright",
            "  if the answers cannot be right. If the quantity you ask for has a",
            "  domain -- a count is a non-negative integer, a probability lies in [0, 1],",
            "  an expected time is positive, rows of a transition matrix sum to 1 -- assert",
            "  that domain in `generate_entry` and reject the draw when it fails. S17 in",
            "  wave2 passed every gate reporting an expected absorption time of -44/5,",
            "  from rows that summed above 1. Read your own samples file before you stop:",
            "  an answer no solver could produce is the one failure you cannot see coming.",
        )
    )
    if task_meta is not None:
        sections.extend(
            (
                "The task module must contain this exact module-level provenance mapping:",
                "",
                "```python",
                f"TASK_META = {pprint.pformat(task_meta, sort_dicts=False)}",
                "```",
                "",
            )
        )
    sections.extend(
        (
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
            '  0.4. A word problem that narrates its own total -- "working together they',
            '  produce 12 pounds. What is the total number of pounds?" -- scores 1.00 here',
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
            "  Print each example's prompt exactly as the task emits it and its gold answer",
            "  underneath, both verbatim and neither paraphrased nor left out. A reader with",
            "  only this file decides whether your task is answerable at all, and two wave4",
            "  trials were thrown out as unanswerable when the task was fine: one printed",
            "  headings with no prompt under them, one dropped an answer.",
            "- `contract` generates 64 examples and requires score_answer to return 1.0 on",
            "  every gold answer and less than 1.0 on empty and junk strings.",
            "- `speed` times generate_example at the DEFAULT config, which is the one the",
            "  contract audit uses. The harness kills any validation command at 300 seconds,",
            "  so a generator averaging more than about 4 seconds an example loses the trial",
            "  whatever else it does. Bound every rejection-sampling loop; the cost is",
            "  heavy-tailed and one pathological instance is enough.",
            f"- your working directory is the read-only worktree root and only the task",
            "  directory is writable, so the sample script must resolve its output next to",
            f'  itself -- `Path(__file__).with_name("samples_{trial.trial_id}.md")` -- and',
            "  never as a bare relative name. S37 in wave3 lost its trial at step 14 to",
            "  `OSError: [Errno 30] Read-only file system`, with the generator already written.",
            "- time-box the generator the first time you run it, before you build anything on",
            "  top of it. Two wave3 trials spent more than half their wall clock inside a",
            "  generate call that never returned; one lost to a level-6 timeout and one to the",
            "  harness dying under it. A level that cannot produce an example in a few seconds",
            "  means the algorithm is wrong for that level: count with a DP over states rather",
            "  than enumerating what you are counting.",
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
        )
    )
    return "\n".join(sections)


_CANDIDATE_LIBS = (
    "z3",
    "sympy",
    "networkx",
    "numpy",
    "scipy",
    "nltk",
    "lark",
    "pyparsing",
    "regex",
    "automata",
    "pandas",
)


def _available_libs():
    import importlib.util

    return ", ".join(m for m in _CANDIDATE_LIBS if importlib.util.find_spec(m))


def _budget_phrase(task_meta):
    steps = (task_meta or {}).get("generation", {}).get("settings", {}).get("max_steps")
    return f"exactly {steps} steps" if steps else "a very small step budget"


def _seed_phrase(task_meta):
    seed = (
        (task_meta or {})
        .get("generation", {})
        .get("settings", {})
        .get("requested_seed")
    )
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
    return (
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m"
        f" reasoning_core.task_search.selfcheck {owned_path} {trial_id}"
    )


def _selfcheck_command(trial):
    return _selfcheck_command_for(trial.owned_path, trial.trial_id)


def _prior_audit_command(trial):
    # A task a single fixed answer wins is not measuring reasoning, however well
    # it validates; measured on the first six wave0 tasks, two of them lost.
    return (
        "PYTHONDONTWRITEBYTECODE=1 python -m reasoning_core.task_search.prior_audit"
        f" --path {trial.owned_path} --n 30 --max-const 0.4 --max-shortcut 0.4 --budget-seconds 45"
    )
