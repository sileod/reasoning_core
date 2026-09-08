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
`reasoning_core/generation/worker.py` and `reasoning_core/generation/run_generate.sh`.
Inspect worker options with `python -m reasoning_core.generation.worker --help`.
Uploading is a separate operation: install `python -m pip install -e '.[collection]'`
and inspect `python -m reasoning_core.generation.collect --help`. Uploads require Hub
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

Two different questions, answered by two different tools.

**How well does my model solve Reasoning Core tasks?** Generate data, let the model
answer it, score the answers. Install `python -m pip install -e '.[eval]'`, configure
credentials for your `litlm` provider, and supply its model identifier:

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

The input is the file [Sample data](#sample-data) wrote. This makes paid or
provider-limited requests. Each row keeps the raw output (`y`), the extracted answer
(`pred`), a `format_check`, and the task-native `score`. Read the two columns
together: `format_check` separates answering in the wrong shape from answering
wrongly, and a model that scores badly with a low format check is being measured on
its formatting rather than its reasoning. `system_prompt=` changes the answer-format
instruction and `scorer=` replaces task-native scoring.

**How does this checkpoint do on the benchmark battery?** The battery is the fixed
held-out suite that influence measurements are computed against — ARC, GSM8K, BBH,
MMLU, FineWeb NLL and others. It scores by teacher-forced likelihood rather than
free generation, so it works on base models that follow no instructions, and it
takes a loaded model rather than an API:

```bash
python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from reasoning_core.evaluation.battery import default_battery, evaluate_battery

name = 'EleutherAI/pythia-70m'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()

battery = default_battery(max_length=1024)
print(battery.identifier, len(battery.legs), 'legs')
result = evaluate_battery(model, tokenizer, battery, tokenizer.eos_token)
print(len(result.metrics), 'metrics')
for key in sorted(result.metrics)[:3]:
    print(f'{key:34s} {result.metrics[key]:.4f}')
PY
```

That prints `copyfree_battery_v8_tiny/battery@v1:c94e9ad44be0 39 legs` and 89
metrics, because each multiple-choice leg reports likelihood, accuracy, and margin.
The first call unpacks the shipped legs out of the package into `data_cache/`, so
no benchmark data is downloaded.

**Pass `max_length` explicitly.** It is part of the battery identifier, and the
manifest's own default is 512 (`...:5e8ae638a485`) while published results are
measured at 1024 (`...:c94e9ad44be0`). Numbers carrying different identifiers
describe different measurements and cannot be pooled, so build the battery at the
length you intend to compare at. [The training guide](evaluation.md) covers battery
identities, custom manifests, and the legacy `paper_battery()`.

The run above takes roughly 100 s of wall time on a many-core CPU host for a
70M-parameter model, and cost grows with parameter count; use a GPU for anything
larger.

## Run a paired influence smoke

These two commands exercise the training and evaluation pipeline end to end on a
tiny random model. They are plumbing checks, not measurements — the model is a
one-layer GPT-2 over an eight-token vocabulary and the evaluation suite is
synthetic, so no number either prints can be compared to a benchmark.

With the training extra installed (`python -m pip install -e '.[training]'`):

```bash
CUDA_VISIBLE_DEVICES='' python scripts/smoke_influence.py --eval-only
```

That constructs the model and tokenizer locally and prints an evaluation identity
and a finite answer-token `nll`; no weights, datasets, or API credentials are
downloaded. Dropping the flag runs the full paired protocol:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/smoke_influence.py
```

This trains a baseline and a two-source group treatment for two steps each from
identical initial weights, evaluates both arms, and prints baseline/treatment
metrics, `treatment - baseline` deltas, and their artifact directories. Artifacts
default to `~/.reasoning_core/runs/arms/workflow-smoke-v1/`; `RC_HOME` can select
another directory under your home. Repeating an identical run reuses completed arm
results.

The script includes all model, tokenizer, dataset factory, evaluation callback, and
immutable-ID setup. Replace the toy data and model for actual research and follow the
[influence protocol](influence.md).

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
