# Task influence reference results

These are the basic per-task influence measurements used while developing Reasoning
Core. They are a historical reference, not a claim that every current generator has
been re-measured with the public runner.

Each auxiliary task was mixed at 20% into an 80% FineWeb-Edu + dolci run on
SmolLM2-135M for 300 steps with answer-only loss and seed 43. The table reports the
mean percentage NLL reduction across six held-out legs, so **positive means
helpful**. Reward is task-native free-generation reward (`start → end`) and is
diagnostic rather than part of the score.

Protocol snapshot: 50 tasks, S43, T300, M20. Most rows use source snapshot `8aa8748`;
`⟳` marks generators re-measured on 2026-07-09. Use
[task influence guide](README.md) and the
[public training API](../reasoning_core/training/README.md) for new measurements.

## SmolLM2-135M

| # | Task | Mean NLL reduction | Reward |
| ---: | --- | ---: | ---: |
| 1 | `planar_geometry_relations` | **+4.30%** | 0.00 → 0.20 |
| 2 | `logic_qa` | **+4.21%** | 0.00 → 0.80 |
| 3 | `math_word_problem` | **+4.05%** | 0.00 → 0.00 |
| 4 | `constraint_satisfaction` | **+3.99%** | 0.00 → 0.00 |
| 5 | `qualitative_reasoning` | **+3.71%** | 0.00 → 0.04 |
| 6 | `coreference` ⟳ | **+3.63%** | 0.00 → 0.40 |
| 7 | `equation_system` | **+3.61%** | 0.00 → 0.00 |
| 8 | `defeasible_nli` | **+3.60%** | 0.00 → 0.40 |
| 9 | `qualitative_causal_reasoning` | **+3.48%** | 0.00 → 0.48 |
| 10 | `arithmetics` | **+3.47%** | 0.00 → 0.00 |
| 11 | `analogical_case_matching` ⟳ | **+3.43%** | 0.00 → 0.48 |
| 12 | `unification_entailment` | **+3.40%** | 0.00 → 0.04 |
| 13 | `graph_dependencies` | **+3.32%** | 0.00 → 0.00 |
| 14 | `game_forced_win` | **+3.28%** | 0.00 → 0.48 |
| 15 | `multistep_abduction` | **+3.22%** | 0.00 → 0.00 |
| 16 | `regex_reasoning` | **+3.21%** | 0.00 → 0.20 |
| 17 | `multistep_nli` | **+3.15%** | 0.00 → 0.44 |
| 18 | `grid_navigation` | **+3.03%** | 0.00 → 0.24 |
| 19 | `table_qa` ⟳ | **+3.02%** | 0.00 → 0.00 |
| 20 | `most_probable_evidence` | **+2.94%** | 0.00 → 0.20 |
| 21 | `game_best_move` | **+2.93%** | 0.00 → 0.52 |
| 22 | `sequential_induction` | **+2.93%** | 0.00 → 0.00 |
| 23 | `belief_tracking` | **+2.91%** | 0.00 → 0.28 |
| 24 | `most_probable_outcome` | **+2.88%** | 0.00 → 0.08 |
| 25 | `graph_successors` | **+2.81%** | 0.00 → 0.06 |
| 26 | `code_analysis` | **+2.77%** | 0.00 → 0.20 |
| 27 | `lean_candidate_compilation` | **+2.60%** | 0.00 → 0.60 |
| 28 | `logic_nli` | **+2.60%** | 0.00 → 0.33 |
| 29 | `reference_tracking` | **+2.58%** | 0.28 → 0.36 |
| 30 | `rewrite_system` | **+2.58%** | 0.00 → 0.04 |
| 31 | `multistep_evidence_retrieval` | **+2.56%** | 0.00 → 0.00 |
| 32 | `constrained_continuation` | **+2.52%** | 0.00 → 0.00 |
| 33 | `metamath_core_select` | **+2.49%** | 0.04 → 0.17 |
| 34 | `string_transduction` | **+2.46%** | 0.00 → 0.00 |
| 35 | `parsing_derivation` | **+2.38%** | 0.00 → 0.00 |
| 36 | `metamath_entailment` | **+2.23%** | 0.00 → 0.62 |
| 37 | `regex_following` ⟳ | **+2.23%** | 0.00 → 0.00 |
| 38 | `code_execution` | **+2.21%** | 0.01 → 0.03 |
| 39 | `lambda_reduction` | **+2.19%** | 0.00 → 0.00 |
| 40 | `lean_missing_line` | **+2.04%** | 0.00 → 0.00 |
| 41 | `graph_pathfinding` | **+1.81%** | 0.12 → 0.12 |
| 42 | `logic_formalization` | **+1.80%** | 0.00 → 0.08 |
| 43 | `code_runnability` | **+1.65%** | 0.00 → 0.00 |
| 44 | `syntax_error_detection` | **+1.61%** | 0.00 → 0.00 |
| 45 | `set_missing_element` | **+1.33%** | 0.00 → 0.00 |
| 46 | `planning` | **+1.12%** | 0.00 → 0.10 |
| 47 | `set_expression` | **+1.00%** | 0.00 → 0.00 |
| 48 | `program_synthesis` | **+0.85%** | — |
| 49 | `table_statistics` | **+0.60%** | 0.00 → 0.00 |
| 50 | `table_equivalence` | **+0.41%** | 0.00 → 0.00 |

## SmolLM2-360M

The companion cross-model sweep used the same fwdolci main, seed 43, 300 steps,
20% auxiliary mix, and six-leg aggregate. It measured 51 tasks; all were net
helpers. Its ranking had Spearman ρ = **+0.77** with the companion 135M sweep.
That 135M sweep used letter-form MMLU while 360M used cloze, so treat the correlation
as a robustness diagnostic rather than exact protocol parity. The detailed
[six-leg table](https://github.com/sileod/reasoning-core-experiments/blob/89e357ef2fa5dcf651d065f9d781677fefa77273/results/tables/xmodel_valuation.md)
and [reward report](https://github.com/sileod/reasoning-core-experiments/blob/89e357ef2fa5dcf651d065f9d781677fefa77273/results/sweeps/xmodel_SM360.md)
remain available as frozen source artifacts.

| # | Task | Mean NLL reduction | Reward |
| ---: | --- | ---: | ---: |
| 1 | `planar_geometry_relations` | **+4.93%** | 0.20 → 0.71 |
| 2 | `math_word_problem` | **+4.55%** | 0.02 → 0.58 |
| 3 | `qualitative_causal` | **+4.12%** | 0.42 → 0.85 |
| 4 | `constraint_satisfaction` | **+4.11%** | 0.43 → 0.67 |
| 5 | `logic_qa` | **+4.10%** | 0.00 → 0.93 |
| 6 | `most_probable_outcome` | **+3.88%** | 0.00 → 0.65 |
| 7 | `logic_nli` | **+3.87%** | 0.00 → 0.78 |
| 8 | `qualitative_reasoning` | **+3.74%** | 0.22 → 0.69 |
| 9 | `metamath_core_select` | **+3.73%** | 0.00 → 0.65 |
| 10 | `regex_reasoning` | **+3.72%** | 0.00 → 0.56 |
| 11 | `evidence_retrieval` | **+3.66%** | 0.23 → 0.60 |
| 12 | `navigation` | **+3.56%** | 0.35 → 0.82 |
| 13 | `equation_system` | **+3.49%** | 0.16 → 0.63 |
| 14 | `multistep_nli` | **+3.49%** | 0.00 → 0.68 |
| 15 | `count_elements` | **+3.40%** | 0.00 → 0.62 |
| 16 | `coreference` | **+3.09%** | 0.19 → 0.83 |
| 17 | `game_forced_win` | **+3.09%** | 0.00 → 0.71 |
| 18 | `tptp_entailment` | **+2.97%** | 0.00 → 0.88 |
| 19 | `multistep_abduction` | **+2.94%** | 0.00 → 0.61 |
| 20 | `most_probable_evidence` | **+2.89%** | 0.14 → 0.81 |
| 21 | `multistep_evidence_retrieval` | **+2.83%** | 0.53 → 0.78 |
| 22 | `reference_tracking` | **+2.68%** | 0.11 → 0.79 |
| 23 | `graph_dependencies` | **+2.63%** | 0.31 → 0.54 |
| 24 | `lean_candidate_compilation` | **+2.62%** | 0.00 → 0.78 |
| 25 | `graph_successors` | **+2.59%** | 0.00 → 0.59 |
| 26 | `graph_pathfinding` | **+2.56%** | 0.24 → 0.72 |
| 27 | `arithmetics` | **+2.55%** | 0.23 → 0.45 |
| 28 | `table_qa` | **+2.52%** | 0.47 → 0.74 |
| 29 | `set_expression` | **+2.51%** | 0.73 → 0.83 |
| 30 | `lambda_reduction` | **+2.43%** | 0.54 → 0.73 |
| 31 | `rewrite_system` | **+2.43%** | 0.61 → 0.82 |
| 32 | `constrained_continuation` | **+2.40%** | 0.25 → 0.56 |
| 33 | `table_statistics` | **+2.39%** | 0.20 → 0.88 |
| 34 | `game_best_move` | **+2.37%** | 0.04 → 0.84 |
| 35 | `code_input_deduction` | **+2.34%** | 0.00 → 0.73 |
| 36 | `code_execution` | **+2.30%** | 0.21 → 0.68 |
| 37 | `metamath_entailment` | **+2.29%** | 0.00 → 0.91 |
| 38 | `regex_following` | **+2.27%** | 0.01 → 0.15 |
| 39 | `sequential_induction` | **+2.21%** | 0.40 → 0.82 |
| 40 | `regex_induction` | **+2.20%** | 0.08 → 0.61 |
| 41 | `analogical_case_retrieval` | **+2.18%** | 0.65 → 0.80 |
| 42 | `set_missing_element` | **+2.08%** | 0.58 → 0.86 |
| 43 | `string_transduction` | **+2.00%** | 0.10 → 0.22 |
| 44 | `lean_missing_proof_line_selection` | **+1.96%** | 0.00 → 0.59 |
| 45 | `parsing_derivation` | **+1.96%** | 0.52 → 0.68 |
| 46 | `logic_formalization` | **+1.93%** | 0.00 → 0.82 |
| 47 | `locate_error` | **+1.90%** | 0.11 → 0.56 |
| 48 | `code_runnability` | **+1.62%** | 0.05 → 0.83 |
| 49 | `table_equivalence` | **+1.51%** | 0.00 → 0.75 |
| 50 | `planning` | **+1.23%** | 0.65 → 0.84 |
| 51 | `program_synthesis` | **+1.09%** | 0.65 → 0.86 |

Do not compare these percentages directly with runs using a different model, token
dose, formatter, data mix, seed policy, or evaluation battery. A new battery ID starts
a new comparison series.
