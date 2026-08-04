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

## Ranking

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

Do not compare these percentages directly with runs using a different model, token
dose, formatter, data mix, seed policy, or evaluation battery. A new battery ID starts
a new comparison series.
