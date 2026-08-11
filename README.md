# Reasoning Core ◉

**Procedural reasoning data for language-model pre-training, post-training, evaluation, and RL.**

Reasoning Core generates verifiable textual tasks across first-order logic, formal mathematics with Lean and TPTP, planning, algorithms, syntax, and more. Use it as a Python library, generate datasets at scale, or plug it into modern reinforcement-learning environments.

More than **10B tokens** of pre-generated data are available in the 🤗 [Reasoning Core dataset collection](https://huggingface.co/collections/reasoning-core/datasets).

Tasks target compact, canonical answers and expose task-native scorers—clean targets for supervised fine-tuning (SFT), with deterministic rewards for RL and evaluation. The public [training and influence API](reasoning_core/training/README.md) provides reproducible paired baseline/treatment experiments. Follow the [task influence guide](task_influence/README.md) to measure a new or changed task, and compare it with the [reference results](task_influence/RESULTS.md).

## Quickstart

```bash
uv pip install reasoning-core
```

```python
from reasoning_core import get_task, score_answer

task = get_task("arithmetics")
example = task.generate_example()

print(example.prompt)
assert score_answer(example.answer, example) == 1
```

## Representative example

Reasoning Core includes compilation-checked formal reasoning tasks such as [`lean_candidate_compilation`](GALLERY.md#lean_candidate_compilation):

**Prompt**

```text
Does this Lean 4 tactic body close the theorem?
The answer is True or False.

THEOREM:
theorem ex (p2 p4 : Prop) : p2 → (p2 ∨ p4) := by
  ?

CANDIDATE:
linarith
```

**Answer:** `False`

Browse all [50 task examples](GALLERY.md).

## Task catalogue

[GALLERY](https://github.com/sileod/reasoning-core/blob/main/GALLERY.md) (names link to gallery examples)

[`arithmetics`](GALLERY.md#arithmetics) · [`math_word_problem`](GALLERY.md#math_word_problem) · [`equation_system`](GALLERY.md#equation_system) · [`combinatorics_formula`](GALLERY.md#combinatorics_formula) · [`function_manipulation`](GALLERY.md#function_manipulation) · [`lean_missing_line`](GALLERY.md#lean_missing_line) · [`lean_candidate_compilation`](GALLERY.md#lean_candidate_compilation) · [`planar_geometry_relations`](GALLERY.md#planar_geometry_relations) · [`metamath_entailment`](GALLERY.md#metamath_entailment) · [`metamath_core_select`](GALLERY.md#metamath_core_select) · [`lambda_reduction`](GALLERY.md#lambda_reduction) · [`rewrite_system`](GALLERY.md#rewrite_system) · [`unification_entailment`](GALLERY.md#unification_entailment) · [`most_probable_evidence`](GALLERY.md#most_probable_evidence) · [`most_probable_outcome`](GALLERY.md#most_probable_outcome) · [`multistep_nli`](GALLERY.md#multistep_nli) · [`defeasible_nli`](GALLERY.md#defeasible_nli) · [`multistep_evidence_retrieval`](GALLERY.md#multistep_evidence_retrieval) · [`multistep_abduction`](GALLERY.md#multistep_abduction) · [`logic_qa`](GALLERY.md#logic_qa) · [`logic_derivation`](GALLERY.md#logic_derivation) · [`planning`](GALLERY.md#planning) · [`set_missing_element`](GALLERY.md#set_missing_element) · [`set_expression`](GALLERY.md#set_expression) · [`sequential_induction`](GALLERY.md#sequential_induction) · [`qualitative_reasoning`](GALLERY.md#qualitative_reasoning) · [`grid_navigation`](GALLERY.md#grid_navigation) · [`reference_tracking`](GALLERY.md#reference_tracking) · [`belief_tracking`](GALLERY.md#belief_tracking) · [`coreference`](GALLERY.md#coreference) · [`constraint_satisfaction`](GALLERY.md#constraint_satisfaction) · [`graph_pathfinding`](GALLERY.md#graph_pathfinding) · [`graph_successors`](GALLERY.md#graph_successors) · [`regex_following`](GALLERY.md#regex_following) · [`regex_reasoning`](GALLERY.md#regex_reasoning) · [`analogical_case_matching`](GALLERY.md#analogical_case_matching) · [`parsing_derivation`](GALLERY.md#parsing_derivation) · [`syntax_error_detection`](GALLERY.md#syntax_error_detection) · [`constrained_continuation`](GALLERY.md#constrained_continuation) · [`table_qa`](GALLERY.md#table_qa) · [`table_equivalence`](GALLERY.md#table_equivalence) · [`table_statistics`](GALLERY.md#table_statistics) · [`string_transduction`](GALLERY.md#string_transduction) · [`game_best_move`](GALLERY.md#game_best_move) · [`game_forced_win`](GALLERY.md#game_forced_win) · [`qualitative_causal_reasoning`](GALLERY.md#qualitative_causal_reasoning) · [`code_analysis`](GALLERY.md#code_analysis) · [`code_runnability`](GALLERY.md#code_runnability) · [`code_execution`](GALLERY.md#code_execution) · [`program_synthesis`](GALLERY.md#program_synthesis)


## Task authoring guidelines

A task authoring guide describes the interface and guidelines.  
[TASK_AUTHORING_GUIDE](https://github.com/sileod/reasoning-core/blob/main/TASK_AUTHORING_GUIDE.md)
[TRAINING_AND_INFLUENCE](reasoning_core/training/README.md)
[TASK_INFLUENCE](task_influence/README.md)

## Ecosystem and integrations

- **[Prime Intellect](https://app.primeintellect.ai/dashboard/environments)** — install Reasoning Core from the Environments Hub for evaluation and RL workflows.
- **[OpenReward](https://openreward.ai/dsileo/reasoning-core)** — run Reasoning Core as an OpenReward-compatible environment.
- **[OpenEnv](https://huggingface.co/spaces/reasoning-core/reasoning-core-openenv)** — explore the interactive Reasoning Core OpenEnv on Hugging Face Spaces.
- **[reasoning-gym](https://github.com/open-thought/reasoning-gym)** — mix Reasoning Core and reasoning-gym tasks through either library's interface.
- **[SynLogic](https://github.com/MiniMax-AI/SynLogic)** — generate SynLogic games through the same task API, backed by SynLogic's native verifiers.

See the [integration guide](INTEGRATIONS.md) for runnable examples.



## Generate datasets at scale

Go from a single example to large pre-training, post-training, and evaluation corpora with balanced difficulty, token budgets, and verifiable answers. The generation pipeline supports parallel workers, resumable jobs, and JSONL shards and scalable postprocessing scripts ready for Hugging Face Datasets.


## Citation and paper

```bibtex
@article{reasoningcore2026,
  title={Reasoning Core: A Scalable Procedural Data Generation Suite for Symbolic Pre-training and Post-Training},
  author={Lacombe, Valentin and Quesnel, Valentin and Sileo, Damien},
  journal={arXiv preprint arXiv:2603.02208},
  year={2026},
  url={https://arxiv.org/abs/2603.02208}
}
```
https://arxiv.org/abs/2603.02208  
Contact: damien.sileo@inria.fr
