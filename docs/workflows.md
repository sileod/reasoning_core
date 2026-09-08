# Everyday workflows

Run repository examples from the checkout root. Python 3.10 or newer is required;
individual optional dependencies may require a newer Python version. A virtual
environment keeps authoring dependencies separate from the training stack:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For lightweight authoring only, use `bash scripts/install_authoring.sh` instead of
the full install. Install a task's solver dependencies when that task needs them.
The catalogue needs only the lightweight install.

## Find a task

```bash
python -m reasoning_core catalog 'graph'
python -m reasoning_core catalog 'graph' --all --json
```

Search matches all whitespace-separated words, ignoring case, across names,
summaries, status, origin, and source paths. JSON output is an array with `name`,
`summary`, `status`, `origin`, `source`, and `line`. No task modules are imported.
An empty result is `[]` with exit status zero.

Defaults match `list_tasks()`. `--include-generated` includes tasks under
`tasks/generated/`; `--all` also includes mutated tasks and discoverable `DevTask`
classes. `status` is `active` (a `Task`) or `dev` (a `DevTask`), and `origin` is
`core`, `generated`, or `mutated`. These describe declarations and file placement,
not a quality ranking. Deprecated directories and collection adapters are excluded
by registry discovery. Summaries absent from source are empty strings.

Python callers can use `from reasoning_core import task_catalog` with the same
filters. See [GALLERY.md](../GALLERY.md) for example prompts and answers.

## Sample data

With the full install:

```bash
python -m reasoning_core sample arithmetics --count 3 --level 0 --output /tmp/rc-samples.jsonl
```

This checks each reference answer scores 1, then writes JSONL with `prompt`,
`answer`, `task`, `metadata`, and `cot`. The command prints the output path and row
count. Existing output files are refused; choose a new path for each run.
Sampling uses each task's random generation and does not promise identical rows
between runs. Preserve the generated file when comparing models.

For larger balanced batches, use `get_task(name).generate_balanced_batch(...)`.
The production worker and Linux/GNU Parallel launcher are
`reasoning_core/generation_worker.py` and `reasoning_core/run_generate.sh`.
Inspect worker options with `python -m reasoning_core.generation_worker --help`.
Uploading is a separate operation: install `python -m pip install -e '.[collection]'`
and inspect `python -m reasoning_core.collect --help`. Uploads require Hub
credentials; the collector deletes uploaded input files by default (`--no-delete`
preserves them).

## Implement and validate a task

Read [TASK_AUTHORING_GUIDE.md](../TASK_AUTHORING_GUIDE.md) first. With the lightweight
install, save its [minimal skeleton](../TASK_AUTHORING_GUIDE.md#minimal-task-skeleton)
as `reasoning_core/tasks/example_maximum.py`, then validate it:

```bash
python -m reasoning_core validate example_maximum --samples 3
```

Expected output includes `"valid": true`. The new file is yours to edit into a
real task; it is not part of the shipped roster. `validate()` exercises difficulty
transitions as well as generation and scoring. No registry edit is necessary.
Follow [TASK_MUTATION_GUIDE.md](../TASK_MUTATION_GUIDE.md) for experimental tasks.

For another task, substitute its registry name. See the authoring guide for the
isolated scaling benchmark and task-specific dependency requirements.

## Score predictions

Keep every sampled row and add a `prediction` field containing the model's answer.
Score those files without an API call:

```bash
python -m reasoning_core score /tmp/rc-predictions.jsonl
```

Output contains `examples`, `mean_score`, and per-row `scores`. Task-native rewards
may be fractional; `mean_score` is not necessarily exact-match accuracy.

To smoke-test the file/scorer round trip using the sample file above:

```bash
python - <<'PY'
import json
from pathlib import Path
rows = [json.loads(line) for line in Path('/tmp/rc-samples.jsonl').read_text().splitlines()]
with Path('/tmp/rc-predictions.jsonl').open('x') as output:
    for row in rows:
        output.write(json.dumps({**row, 'prediction': row['answer']}) + '\n')
PY
python -m reasoning_core score /tmp/rc-predictions.jsonl
```

Expected `mean_score` is 1. This verifies scoring only, not model ability.

## Evaluate a model

For a fully offline model-evaluation smoke, install the training extra and run:

```bash
python -m pip install -e '.[training]'
CUDA_VISIBLE_DEVICES='' python scripts/smoke_influence.py --eval-only
```

The example constructs a tiny random GPT-2 model and tokenizer locally. It prints
an evaluation identity and finite answer-token `nll`; no weights, datasets, or API
credentials are downloaded. Its synthetic metric is a pipeline check.

For API-backed task evaluation, install `python -m pip install -e '.[eval]'`,
configure credentials for your `litlm` provider, and supply its model identifier:

```bash
export RC_EVAL_MODEL='your-provider/your-model'
python - <<'PY'
import os
import pandas as pd
from reasoning_core import evaluate_model
rows = pd.read_json('/tmp/rc-samples.jsonl', lines=True)
scored = evaluate_model(rows, model_name=os.environ['RC_EVAL_MODEL'])
scored.to_json('/tmp/rc-api-results.jsonl', orient='records', lines=True)
print(scored.groupby('task')[['format_check', 'score']].mean())
PY
```

This makes paid/provider-limited requests and saves raw outputs (`y`), extracted
answers (`pred`), format checks, and task scores. For local pretrained models and
benchmark batteries, use `evaluate_battery()` in
`reasoning_core.evaluation.battery`; [the training guide](evaluation.md)
explains battery identities and evaluator variants.

## Run a paired influence smoke

With the training extra installed:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/smoke_influence.py
```

This trains a baseline and a two-source group treatment for two steps each from
identical tiny-model weights, evaluates
both arms, and prints baseline/treatment metrics, `treatment - baseline` deltas,
and their artifact directories. Artifacts default to
`~/.reasoning_core/runs/arms/workflow-smoke-v1/`; `RC_HOME` can select another
directory under your home. Repeating an identical run reuses completed arm results.

The script includes all model, tokenizer, dataset factory, evaluation callback,
and immutable-ID setup. Replace the toy data/model for actual research and follow
the [influence protocol](influence.md). The smoke uses a synthetic
evaluation suite; its numbers cannot be compared to the published benchmark.

Custom battery manifests resolve relative leg paths beside the manifest. Passing
`data_dir` overrides that root. Shipped batteries unpack their frozen legs into
`data_cache/` (or `EVAL_DATA_DIR`); never rebuild those legs to refresh data.

## Run task search

Task search creates new tasks with coding agents. To find an existing task, use
the catalogue above. From a checkout with the full install, inspect the search
interface and render one historical trial without launching a model:

```bash
python -m reasoning_core.task_search --help
python -m reasoning_core.task_search proposal-catalog
python -m reasoning_core.task_search render reasoning_core/task_search/plans/wave8.yaml P001v1
```

The last command prints the implementor prompt. Historical plans pin old commits;
use newly generated plans for new runs. See the
[task-search guide](../reasoning_core/task_search/README.md) for runnable
`propose → check-proposals → plan → check → run` commands, Harness Link installation
requirements, provider/reviewer credentials, and `doctor --live` preflight.
`doctor --live` and actual proposal/worker runs make model requests.

Each trial retains its prompt, trajectory, validation logs, and `run.json` in the
run directory printed at startup; wave progress is in `summary.json`. Review and
landing are described in [BABYSITTING.md](../reasoning_core/task_search/BABYSITTING.md).

## Local checks

Install `pytest` explicitly (`python -m pip install pytest`). For catalogue and
core infrastructure changes:

```bash
python -m pytest -q tests/test_task_discovery.py tests/test_score_answer_dispatch.py \
  tests/test_template_validation.py tests/test_workflow_cli.py
```

For training changes, with the training extra installed:

```bash
python -m pytest -q tests/test_training_runtime.py
```

For search changes, run the relevant `tests/test_task_search_*.py` files. Individual
task tests can need external solvers; choose tests for the subsystem you changed.
