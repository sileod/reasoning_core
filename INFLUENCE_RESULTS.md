# Task influence reference results

These are the basic per-task influence measurements used while developing Reasoning
Core. They are a historical reference, not a claim that every current generator has
been re-measured with the public runner.

Each auxiliary task was mixed at 20% into an 80% FineWeb-Edu + dolci run on
SmolLM2-135M for 300 steps with answer-only loss and seed 43. Each leg reports the
percentage NLL reduction against a task-free baseline, so **positive means helpful**.
`global` is the mean over six legs. `acc` is task-native free-generation reward
(`start → end`) and is diagnostic rather than part of the score.

Protocol snapshot: 50 tasks, S43, T300, M20. Most rows use source snapshot `8aa8748`;
`⟳` marks generators re-measured on 2026-07-09. Use
[`$evaluate-task-influence`](skills/evaluate-task-influence/SKILL.md) and the
[public training API](reasoning_core/training/README.md) for new measurements.

| # | task | global | bbh | mmlu_math | mmlu_logic | mbpp | fw | dolci | acc | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | planar_geometry_relations | **+4.30** | +9.37 | +11.34 | +4.45 | +0.57 | -0.00 | +0.05 | 0.00→0.20 | 55aa7d4 |
| 2 | logic_qa | **+4.21** | +7.86 | +12.79 | +4.01 | +0.59 | -0.03 | +0.07 | 0.00→0.80 | 8da17d7 |
| 3 | math_word_problem | **+4.05** | +7.65 | +13.80 | +2.25 | +0.60 | -0.04 | +0.03 | 0.00→0.00 | 3be023e |
| 4 | constraint_satisfaction | **+3.99** | +6.78 | +12.70 | +3.91 | +0.48 | -0.02 | +0.08 | 0.00→0.00 | cdf0128 |
| 5 | qualitative_reasoning | **+3.71** | +7.42 | +9.93 | +4.17 | +0.65 | -0.00 | +0.08 | 0.00→0.04 | d35670f |
| 6 | coreference ⟳ | **+3.63** | +7.47 | +9.72 | +4.20 | +0.44 | -0.03 | -0.02 | 0.00→0.40 | 7dd565e |
| 7 | equation_system | **+3.61** | +5.72 | +12.60 | +2.70 | +0.63 | +0.00 | +0.01 | 0.00→0.00 | a057bca |
| 8 | defeasible_nli | **+3.60** | +11.49 | +5.39 | +4.25 | +0.49 | -0.01 | +0.00 | 0.00→0.40 | 8da17d7 |
| 9 | qualitative_causal_reasoning | **+3.48** | +8.51 | +7.82 | +3.52 | +0.99 | -0.02 | +0.06 | 0.00→0.48 | eac479f |
| 10 | arithmetics | **+3.47** | +5.46 | +12.27 | +2.31 | +0.67 | -0.00 | +0.09 | 0.00→0.00 | 3be023e |
| 11 | analogical_case_matching ⟳ | **+3.43** | +8.82 | +8.21 | +4.10 | -0.18 | -0.04 | -0.33 | 0.00→0.48 | 1fa7e1e |
| 12 | unification_entailment | **+3.40** | +8.42 | +8.01 | +3.60 | +0.41 | -0.00 | -0.05 | 0.00→0.04 | fc54d11 |
| 13 | graph_dependencies | **+3.32** | +5.71 | +10.61 | +3.11 | +0.48 | -0.01 | +0.02 | 0.00→0.00 | 5f312b5 |
| 14 | game_forced_win | **+3.28** | +10.03 | +6.29 | +2.30 | +0.93 | -0.01 | +0.13 | 0.00→0.48 | 5bd8bd3 |
| 15 | multistep_abduction | **+3.22** | +5.25 | +11.10 | +2.71 | +0.31 | +0.01 | -0.08 | 0.00→0.00 | 8da17d7 |
| 16 | regex_reasoning | **+3.21** | +8.69 | +7.14 | +2.72 | +0.72 | -0.02 | +0.00 | 0.00→0.20 | d0ed155 |
| 17 | multistep_nli | **+3.15** | +9.52 | +5.78 | +3.49 | +0.20 | +0.01 | -0.08 | 0.00→0.44 | 8da17d7 |
| 18 | grid_navigation | **+3.03** | +6.46 | +8.37 | +2.89 | +0.43 | -0.01 | +0.07 | 0.00→0.24 | f099346 |
| 19 | table_qa ⟳ | **+3.02** | +6.24 | +8.10 | +3.30 | +0.57 | -0.06 | -0.06 | 0.00→0.00 | e08c044 |
| 20 | most_probable_evidence | **+2.94** | +6.04 | +8.87 | +2.38 | +0.39 | +0.01 | -0.08 | 0.00→0.20 | 11dc547 |
| 21 | game_best_move | **+2.93** | +6.52 | +7.14 | +2.85 | +1.02 | -0.02 | +0.09 | 0.00→0.52 | 5bd8bd3 |
| 22 | sequential_induction | **+2.93** | +3.72 | +9.73 | +3.18 | +0.81 | -0.02 | +0.15 | 0.00→0.00 | 1d6aa9f |
| 23 | belief_tracking | **+2.91** | +6.80 | +6.67 | +3.05 | +0.97 | -0.01 | -0.03 | 0.00→0.28 | 884048f |
| 24 | most_probable_outcome | **+2.88** | +7.81 | +6.28 | +2.54 | +0.58 | -0.01 | +0.05 | 0.00→0.08 | 11dc547 |
| 25 | graph_successors | **+2.81** | +4.79 | +9.00 | +2.57 | +0.45 | +0.02 | +0.00 | 0.00→0.06 | 5f312b5 |
| 26 | code_analysis | **+2.77** | +6.38 | +6.56 | +2.87 | +0.74 | +0.00 | +0.06 | 0.00→0.20 | a68fffe |
| 27 | lean_candidate_compilation | **+2.60** | +6.20 | +5.81 | +3.25 | +0.37 | -0.02 | +0.02 | 0.00→0.60 | 0461311 |
| 28 | logic_nli | **+2.60** | +7.99 | +4.81 | +2.80 | +0.19 | -0.00 | -0.19 | 0.00→0.33 | 93312bc |
| 29 | reference_tracking | **+2.58** | +6.30 | +6.30 | +2.32 | +0.61 | -0.02 | -0.01 | 0.28→0.36 | f6ca7cb |
| 30 | rewrite_system | **+2.58** | +4.55 | +6.17 | +3.42 | +1.32 | -0.00 | +0.02 | 0.00→0.04 | fc54d11 |
| 31 | multistep_evidence_retrieval | **+2.56** | +4.19 | +9.11 | +2.17 | +0.05 | -0.03 | -0.14 | 0.00→0.00 | 8da17d7 |
| 32 | constrained_continuation | **+2.52** | +5.85 | +4.82 | +3.44 | +0.94 | -0.03 | +0.12 | 0.00→0.00 | eb549a8 |
| 33 | metamath_core_select | **+2.49** | +8.83 | +4.09 | +1.97 | +0.14 | +0.00 | -0.11 | 0.04→0.17 | 2e016be |
| 34 | string_transduction | **+2.46** | +4.35 | +6.97 | +2.75 | +0.73 | -0.02 | +0.00 | 0.00→0.00 | 799e762 |
| 35 | parsing_derivation | **+2.38** | +5.10 | +5.35 | +3.19 | +0.68 | -0.02 | -0.01 | 0.00→0.00 | eb549a8 |
| 36 | metamath_entailment | **+2.23** | +5.64 | +4.80 | +2.87 | +0.21 | -0.01 | -0.11 | 0.00→0.62 | 2e016be |
| 37 | regex_following ⟳ | **+2.23** | +5.59 | +5.28 | +2.10 | +0.44 | -0.01 | +0.01 | 0.00→0.00 | 0ca06d4 |
| 38 | code_execution | **+2.21** | +3.95 | +6.98 | +1.89 | +0.47 | +0.01 | -0.06 | 0.01→0.03 | 6ab47de |
| 39 | lambda_reduction | **+2.19** | +3.95 | +5.05 | +3.35 | +0.83 | -0.01 | -0.05 | 0.00→0.00 | fc54d11 |
| 40 | lean_missing_line | **+2.04** | +5.34 | +5.05 | +1.65 | +0.23 | +0.01 | -0.03 | 0.00→0.00 | 0461311 |
| 41 | graph_pathfinding | **+1.81** | +3.50 | +6.26 | +1.62 | -0.09 | -0.02 | -0.38 | 0.12→0.12 | 5f312b5 |
| 42 | logic_formalization | **+1.80** | +5.49 | +3.36 | +2.00 | +0.28 | -0.02 | -0.29 | 0.00→0.08 | 93312bc |
| 43 | code_runnability | **+1.65** | +4.12 | +2.94 | +2.21 | +0.69 | -0.00 | -0.06 | 0.00→0.00 | 6ab47de |
| 44 | syntax_error_detection | **+1.61** | +4.39 | +2.40 | +2.12 | +0.68 | -0.02 | +0.08 | 0.00→0.00 | eb549a8 |
| 45 | set_missing_element | **+1.33** | +2.59 | +3.79 | +1.46 | +0.36 | -0.02 | -0.22 | 0.00→0.00 | 9c19246 |
| 46 | planning | **+1.12** | +2.97 | +1.77 | +1.19 | +0.86 | +0.00 | -0.04 | 0.00→0.10 | 8d57074 |
| 47 | set_expression | **+1.00** | +2.04 | +3.62 | +0.90 | -0.12 | -0.02 | -0.41 | 0.00→0.00 | 9c19246 |
| 48 | program_synthesis | **+0.85** | +0.74 | +0.86 | +0.82 | +2.84 | +0.02 | -0.16 | — | 95c16a3 |
| 49 | table_statistics | **+0.60** | +1.13 | +2.60 | +0.75 | -0.32 | -0.04 | -0.52 | 0.00→0.00 | 37e73b5 |
| 50 | table_equivalence | **+0.41** | +2.31 | +0.55 | +0.44 | -0.22 | -0.04 | -0.58 | 0.00→0.00 | 37e73b5 |

Do not compare these percentages directly with runs using a different model, token
dose, formatter, data mix, seed policy, or evaluation battery. A new battery ID starts
a new comparison series.
