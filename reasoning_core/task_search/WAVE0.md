# WAVE0 — task-search roadmap

## Objective

Test a small set of genuinely new reasoning primitives and controlled mutations
of strong existing tasks. Prefer short verifier-clean targets and structural
changes at fixed prompt size. The main scientific goal is to identify task
design principles that replicate across families, rather than accumulating
isolated wins.

Do not start by increasing `n`. The most valuable trials change what reasoning
is required while keeping surface size controlled.

## New task trials

These trials target `tasks/generated/` only inside their isolated worktrees.
They enter normal task discovery only if a reviewed worktree is explicitly
promoted; generated tasks remain in `generated/` rather than moving to the task
root.

### N1 — congruence closure (P0, very high ROI)

Ground equalities over nested function terms. Query whether two terms are equal,
or whether equalities plus a disequality are consistent. This is EUF congruence
closure, distinct from substitution/unification and directed rewriting. Use Z3
EUF or check a small implementation against Z3. Scale term count, nesting depth,
and equality-chain/core depth.

### N2 — modular congruence system (P0, very high ROI)

Solve systems such as `x = 3 mod 5` and `2x+y = 1 mod 7`; answer with a canonical
residue, `No solution`, or `Multiple`. This is distinct from arithmetic `%` and
ordinary linear systems. Use SymPy CRT for one dimension and Z3 modular
arithmetic for the general form.

### N3 — lambda type inference (P0, very high ROI)

Infer the principal simple type of a lambda term with canonicalized type
variables. Existing tasks cover beta reduction and exposed term unification,
not type inference. Generate a typed skeleton, erase annotations, and reconstruct
with unification.

### N4 — finite automaton execution (P0, high ROI)

Given an explicit DFA/NFA transition table and word, ask for the final state,
acceptance, or first rejecting position. Existing regex tasks reason from regex
syntax rather than explicit state execution. Scale states, word length, and NFA
active-set size.

### N5 — Hamming decode (P0, high ROI)

Given a parity-check matrix/code convention and corrupted word, identify the
error position or corrected payload. Use exact GF(2) syndrome verification.

### N6 — permutation algebra (P0, high ROI)

Compose, invert, and exponentiate permutations represented by cycles and
mappings; query an image or cycle. This is distinct from following repeated
edges in one permutation graph. Use SymPy `Permutation`.

### N7 — max flow / min cut (P1, high ROI)

Given a capacitated directed graph, ask for the maximum-flow value or a canonical
minimum-cut side. Use NetworkX and lexicographically canonicalize equal cuts.

### N8 — bipartite matching (P1, high-ish ROI)

Ask for maximum-cardinality or maximum-weight matching with a canonical answer
under ties. Generate controlled augmenting-path depth and verify with NetworkX.

### N9 — graph dominators (P1, high ROI)

In a directed CFG-like graph, ask whether `u` dominates `v` or ask for `v`'s
immediate dominator. Unlike reachability, this quantifies over all source-to-node
paths. Use NetworkX `immediate_dominators`.

### N10 — tree reconstruction (P1, medium-high ROI)

Reconstruct enough of a binary tree from preorder+inorder or
inorder+postorder traversals to answer parent, child, or subtree queries.
Generate the tree, derive traversals, and independently reconstruct/check it.

### N11 — interval abstract interpretation (P1, high ROI)

Propagate integer intervals through a tiny branching program and ask for the
abstract range at a program point. This is numerical abstraction, distinct from
concrete execution, finite-state CTL, and domain-neutral fixpoint iteration.
Check against exhaustive bounded concrete execution.

### N12 — minimum spanning tree (P2, medium ROI)

Given a weighted undirected graph, ask for the canonical MST edge list or total
weight. Use NetworkX and either reject ambiguous optima or canonicalize ties for
the selected query.

## Controlled mutations

Every mutation belongs under `tasks/mutated/`, snapshots its exact parent, and
implements one intended distributional change.

### M1 — regex intersection/disjointness (P0)

Parent: `regex_reasoning`. Add intersection/disjointness queries. For nonempty
intersection ask for the shortest common witness; for disjointness answer
`None`. Extend the existing exact FSM machinery with conjunction.

### M2 — Boolean regex languages (P0)

Parent: `regex_reasoning`. Add Boolean language expressions of depth 2–3, such
as `(A ∩ ¬B) ⊆ C` or emptiness of `(A△B)∩C`.

### M3 — let-bound arithmetic DAGs (P0)

Parent: `arithmetics`. Replace expression trees with let-bound DAGs whose
intermediates are reused two to four times. Add dependency tracking and shared
state without merely increasing AST depth.

### M4 — paired arithmetic counterfactuals (P0)

Parent: `arithmetics`. Evaluate an original expression and a version with one
marked literal/operator edit; answer `old,new` or the delta.

### M5 — internal process inversion (P0)

Parent: `math_word_problem`. Hide an internal state and ask for the amount
immediately before or after step `k`, rather than only recovering the start.

### M6 — coupled word-problem distractors (P0)

Parent: `math_word_problem`. Add connected distractor chains sharing entities
and units with the proof while holding the minimal proof core fixed.

### M7 — identifiable linear projection (P0)

Parent: `equation_system`. Generate underdetermined systems where a linear
functional such as `2X1-X3` is uniquely determined although individual variables
are not.

### M8 — controlled elimination depth (P0)

Parent: `equation_system`. Control elimination depth to the queried quantity
while holding variable/equation count roughly fixed; add nuisance equations and
variables.

### M9 — clue essentiality (P0)

Parent: `constraint_satisfaction`. Ask which clue can be removed without
changing the answer, or which clue is necessary. Reuse existing core/essentiality
metrics.

### M10 — paired CSP counterfactual (P0)

Parent: `constraint_satisfaction`. Show the original and one-clue intervention
together; ask for both answers or whether the answer flips.

### M11 — geometric invariance (P0)

Parent: `planar_geometry_relations`. Apply random exact similarities or
relation-safe affine transformations while preserving the query and changing
coordinate structure.

### M12 — geometric construction depth (P0)

Parent: `planar_geometry_relations`. Require the query to depend on two to four
generated intersections, midpoints, or constructions, with matched distractors.

### M13 — observation versus intervention pair (P0)

Parent: `qualitative_causal_reasoning`. On one confounded DAG ask for both
observational association and intervention effect as a canonical short pair.

### M14 — composed causal queries (P1)

Parent: `qualitative_causal_reasoning`. Compose `do()` with conditioning and
optionally two simultaneous interventions while retaining qualitative signed
answers.

### M15 — conjunctive logic branches (P1)

Parent: `logic_qa`. Require conjunctive queries whose predicates derive from
separate proof branches.

### M16 — matched defeasible conflict depth (P1)

Parent: `defeasible_nli`. Generate label-balanced default, exception, and
exception-to-exception cases with matched vocabulary.

### M17 — near-tie MAP contrast (P1)

Parent: `most_probable_evidence`. Present two plausible complete explanations
and choose the more probable one while controlling the log-probability margin.

### M18 — paired epistemic edit (P1)

Parent: `belief_tracking`. Present stories differing in one observation or
message-delivery event and ask for belief before/after the edit.

### M19 — scalar SQL rescue (P1)

Parent: `table_qa`. Restrict SQL tasks to one scalar or short identifier and
control relational-composition depth, testing whether long outputs caused poor
transfer.

### M20 — plan repair/localization rescue (P1)

Parent: `planning`. Provide a nearly valid plan and ask for the first invalid
action or unique replacement, preserving transition reasoning with a short
answer.

## Unified hypotheses

### H1 — core-depth curriculum

Difficulty driven by minimal proof/computation core depth transfers better than
difficulty driven by prompt size. Tests: M6, M8, M12, M15, M16.

### H2 — query-coupled distractors

Formally irrelevant distractors sharing solution variables/entities/vocabulary
are more useful than disconnected random distractors. Tests: M6, M12, M15 and
future CSP/causal variants.

### H3 — paired counterfactuals

Presenting a world and one-local-edit world together teaches sensitivity better
than independent counterfactual examples. Tests: M4, M10, M13, M18.

### H4 — short-answer bottleneck

For the same latent computation, short canonical targets transfer better than
long generation targets. Tests: M19, M20 and future candidate-selection forms.

### H5 — certificate beats label

A short canonical witness/core teaches more than a Boolean label. Tests: M1,
M9 and future causal-core variants.

### H6 — semantic invariance augmentation

Semantics-preserving representation changes improve transfer when structural
difficulty is fixed. Tests: M11 and future equation/regex/table renderings.

### H7 — shared-state reuse

Reusing latent intermediate values teaches working-memory/reference behavior
better than equivalent tree computation. Test: M3 and future code-task variants.

### H8 — decision-boundary sampling

Instances near a semantic decision boundary are more valuable than uniformly
sampled hard-looking instances. Tests: M17 and future regex/causal/rank margins.

### H9 — identifiable projection

Asking for a uniquely determined projection of an ambiguous global state is more
useful than forcing global uniqueness. Test: M7 and future CSP/probability forms.

### H10 — cross-primitive composition

Composing two familiar primitives at shallow depth is more valuable than taking
one primitive to extreme depth. Tests: M2, M14, M15.

## Execution and promotion

1. First screen N1–N6 and M1–M13, deferring M11/M12 if geometry generation is
   costly.
2. Materialize rows once and keep main data, formatter, seed, model revision,
   token dose, and evaluation battery identical between matched parent/child
   arms.
3. Never compare rankings across battery IDs.
4. Put exactly one hypothesis ID in each descendant's `TASK_META`.
5. Advance a hypothesis when the child-parent sign replicates across at least
   three task families, then run the three-model replication.
6. Allocate approximately 40% to mutations of robust tasks, 30% to new
   primitives, 20% to cross-task hypothesis replication, and 10% to rescue
   experiments.

Excluded as already represented natively, in generated/dev code, or in curated
Reasoning Gym: circuit evaluation, course scheduling, base conversion, generic
matrix manipulation, Sudoku, shortest path, regex induction/retrieval, RCC8,
interval algebra, generic DP, and generic SAT search.
