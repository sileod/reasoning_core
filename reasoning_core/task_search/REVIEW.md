# Task-search architecture

The system-level design is stronger than the current Python package boundaries. This is
a refactoring problem, not a reason to redesign the execution model.

## Invariants to preserve

- A `SearchPlan` contains immutable `Trial` specifications.
- Every trial starts from the same resolved commit in a detached worktree.
- The worker can write only its owned task directory and private runtime directory.
- The coordinator independently reruns validation and rejects out-of-scope changes.
- Candidate files are hashed after validation, so accepted results retain an exact
  referent after worktree cleanup.
- Provenance, harness settings, resource limits, retries, and ordered failure status are
  recorded explicitly.
- Retries are limited to infrastructure failures; candidate failures are not silently
  retried into success.
- Ideation is separate from implementation. The wave proposer maintains durable novelty
  memory and emits SFT-oriented proposals before any coding worker is launched.

These choices make agent-generated task code auditable and reproducible enough for human
review. Keep them intact during refactoring.

## Current structural problem

`runner.py` is the architecture. Its roughly 2,200 lines currently own:

- plan loading and validation;
- prompt and pacing policy;
- OpenCode, Mini, and AGY configuration and command construction;
- provider adapters;
- Bubblewrap and resource limits;
- filesystem scope and candidate hashing;
- metadata extraction and task discovery;
- sample, contract, provenance, gameability, and semantic gates;
- retry policy, execution, artifact serialization, and CLI wiring.

The large `_run_trial(...)` and `run_plan(...)` argument surfaces are symptoms of these
mixed responsibilities. A new harness requires edits across orchestration conditionals
instead of implementing one bounded interface. Tests import many private helpers from
`runner.py`, creating an accidental module API.

There is also an important correctness seam: `selfcheck.py` and the authoritative
coordinator contain overlapping gate pipelines. Some implementation is copied, while
other pieces reach back into `runner.py`. Drift between the worker-visible check and the
coordinator verdict is therefore possible.

Prompt inputs are not fully frozen either. Workers execute from `base_commit`, but
context can be read from the live checkout when a prompt is rendered. A long wave can
therefore give later trials different context.

## Target boundaries

Keep the implementation lightweight—dataclasses and one small harness `Protocol` are
enough:

```text
task_search/
    model.py          Trial, SearchPlan, RunConfig, TrialResult, TrialStatus
    plan.py           load, validate, and freeze plans
    prompt.py         prompt rendering and pacing policy
    harness/          protocol plus OpenCode, Mini, and AGY adapters
    sandbox.py        Bubblewrap, resource limits, environment sanitation
    gates/            shared pure gate implementations and GateResult
    executor.py       prepare trial, execute harness, run gates, classify, retry
    artifacts.py      candidate digests, run.json, summary.json
    proposals/        catalog, proposer, and proposal validation
    cli.py             command-line wiring
```

The harness surface should stay narrow:

```python
class Harness(Protocol):
    def prepare(self, context): ...
    def command(self, context): ...
    def parse_usage(self, artifacts): ...
```

Gates should return data, preserving the current first-failure precedence without a long
conditional chain:

```python
@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    failure_status: str
    details: dict
```

JSON dictionaries should remain the persistence format, but execution should use typed
objects such as `TrialResult`, `GenerationInfo`, `CandidateDigest`, and
`SampleValidation` internally.

## Refactor order

1. Extract typed models and shared gate implementations. Make both `selfcheck.py` and the
   coordinator call the same pure gates; leave each as thin orchestration/formatting.
2. Freeze plan, context, and rendered prompts before any trial starts.
3. Extract artifact serialization and the executor while preserving status precedence
   byte-for-byte in recorded output.
4. Introduce the harness protocol and move existing harness branches behind adapters.
5. Split proposal catalog, validation, and model calls once the execution path is stable.

The first step offers the largest correctness and maintainability gain with the smallest
behavioral risk. Every extraction should be covered by characterization tests before old
code is removed.
