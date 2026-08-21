# 📖 Task Gallery

65 tasks

[`arithmetics`](#arithmetics) · [`math_word_problem`](#math_word_problem) · [`equation_system`](#equation_system) · [`combinatorics_formula`](#combinatorics_formula) · [`function_manipulation`](#function_manipulation) · [`lean_missing_line`](#lean_missing_line) · [`lean_candidate_compilation`](#lean_candidate_compilation) · [`planar_geometry_relations`](#planar_geometry_relations) · [`metamath_entailment`](#metamath_entailment) · [`metamath_core_select`](#metamath_core_select) · [`lambda_reduction`](#lambda_reduction) · [`rewrite_system`](#rewrite_system) · [`unification_entailment`](#unification_entailment) · [`most_probable_evidence`](#most_probable_evidence) · [`most_probable_outcome`](#most_probable_outcome) · [`multistep_nli`](#multistep_nli) · [`defeasible_nli`](#defeasible_nli) · [`multistep_evidence_retrieval`](#multistep_evidence_retrieval) · [`multistep_abduction`](#multistep_abduction) · [`logic_qa`](#logic_qa) · [`logic_derivation`](#logic_derivation) · [`planning`](#planning) · [`set_missing_element`](#set_missing_element) · [`set_expression`](#set_expression) · [`sequential_induction`](#sequential_induction) · [`qualitative_reasoning`](#qualitative_reasoning) · [`grid_navigation`](#grid_navigation) · [`reference_tracking`](#reference_tracking) · [`belief_tracking`](#belief_tracking) · [`coreference`](#coreference) · [`constraint_satisfaction`](#constraint_satisfaction) · [`graph_pathfinding`](#graph_pathfinding) · [`graph_successors`](#graph_successors) · [`regex_following`](#regex_following) · [`regex_reasoning`](#regex_reasoning) · [`analogical_case_matching`](#analogical_case_matching) · [`parsing_derivation`](#parsing_derivation) · [`syntax_error_detection`](#syntax_error_detection) · [`constrained_continuation`](#constrained_continuation) · [`table_qa`](#table_qa) · [`table_equivalence`](#table_equivalence) · [`table_statistics`](#table_statistics) · [`string_transduction`](#string_transduction) · [`game_best_move`](#game_best_move) · [`game_forced_win`](#game_forced_win) · [`qualitative_causal_reasoning`](#qualitative_causal_reasoning) · [`code_analysis`](#code_analysis) · [`code_runnability`](#code_runnability) · [`code_execution`](#code_execution) · [`program_synthesis`](#program_synthesis) · [`backtracking_search`](#backtracking_search) · [`boolean_propagation_search`](#boolean_propagation_search) · [`controlled_code_execution`](#controlled_code_execution) · [`dynamic_programming`](#dynamic_programming) · [`fixpoint_iteration`](#fixpoint_iteration) · [`matrix_induction`](#matrix_induction) · [`pattern_induction`](#pattern_induction) · [`schema_bound_query`](#schema_bound_query) · [`conditional_response_contract`](#conditional_response_contract) · [`protected_span_transformation`](#protected_span_transformation) · [`rule_switching`](#rule_switching) · [`shift_reduce_parsing`](#shift_reduce_parsing) · [`spatial_folding`](#spatial_folding) · [`typed_relation_extraction`](#typed_relation_extraction) · [`variable_elimination`](#variable_elimination)

---

## [arithmetics](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/arithmetics.py)

<!-- behavior-hash: f2b3d4e7e62e71bb -->

Compositional arithmetics with float/int/bool, varied operators, number theory.

**Prompt:**
```
Evaluate 6 // 14.
The answer is a number.
```

**Answer:**
```
0
```

---

## [math_word_problem](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/arithmetics.py)

<!-- behavior-hash: f2b3d4e7e62e71bb -->

Solve relational and process math word problems involving objects and values.

**Prompt:**
```
A jar holds 8 tokens. 4 more tokens added; then 6 tokens removed. How many tokens are in the jar now? Answer with a number.
```

**Answer:**
```
6
```

---

## [equation_system](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/equation_system.py)

<!-- behavior-hash: 8781164d0945f615 -->

Solve systems of linear equations or detect inconsistent/underdetermined systems.

**Prompt:**
```
Solve the following system of equations for the variable 'X2'.

System:
  X1 - 7 = 0
  -4*X1 + X2 + 10 = 0
  X2 - 18 = 0

The answer is the value of X2, or 'No solution' / 'Multiple solutions'.
```

**Answer:**
```
18
```

---

## [combinatorics_formula](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/combinatorics.py)

<!-- behavior-hash: 88227d20f937b52d -->

**Prompt:**
```
Write the counting expression. C(n,k) is unordered; P(n,k) is ordered.

Problem:
Make 2 ordered choices from 6 options, allowing repeats.

The answer must have the form:
X1^X2
where:
X1 := 7 | 2 | 6
X2 := 6 | 7 | 2
Answer with the complete expression.
```

**Answer:**
```
6^2
```

---

## [function_manipulation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/function_manipulation.py)

<!-- behavior-hash: 6cfeba2da9fd3a2c -->

Multistep symbolic function manipulation with composition, local inverses, calculus, and short exact answers.

**Prompt:**
```
Define $h(x)=\frac{d}{dx}\left(\frac{1}{\left(x\right)}\right)$.
Compute $h'(1)$.
The answer is a reduced rational number.
```

**Answer:**
```
2
```

---

## [lean_missing_line](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/math_lean.py)

<!-- behavior-hash: a2189324940e2d23 -->

Complete a Lean proof with a uniquely valid constrained proof line.

**Prompt:**
```
Fill `__ANSWER__` with a Lean proof line. Mathlib is imported.

THEOREM:
theorem ex (s t u : Finset Nat) : s ∩ (t ∪ u) = (s ∩ t) ∪ (s ∩ u) := by
  __ANSWER__

The answer must have the form:
simpa using inf_sup_left s X1 X2
where:
X1 := s | t | u
X2 := s | t | u
Answer with the complete Lean line.
```

**Answer:**
```
simpa using inf_sup_left s t u
```

---

## [lean_candidate_compilation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/math_lean.py)

<!-- behavior-hash: a2189324940e2d23 -->

Choose which complete Lean tactic body closes a theorem.

**Prompt:**
```
Which Lean 4 tactic body closes the theorem? Exactly one does.
The answer is A or B.

THEOREM:
theorem ex (a b c d : Nat) (h0 : a ∣ b) (h1 : b ∣ c) (h2 : c ∣ d) (junk0 : b ∣ d) : a ∣ d := by
  ?

A:
have step1 : a ∣ b := h0
have step2 : a ∣ c := dvd_trans step1 h0
have step3 : a ∣ d := dvd_trans step2 h2
exact step3

B:
have step1 : a ∣ b := h0
have step2 : a ∣ c := dvd_trans step1 h1
have step3 : a ∣ d := dvd_trans step2 h2
exact step3
```

**Answer:**
```
B
```

---

## [planar_geometry_relations](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/math_geometry.py)

<!-- behavior-hash: cb970570b347392f -->

Answer geometry queries about point intersections, angles, and distances.

**Prompt:**
```
Given points: A=(-5, 3); I=(3, -4); L=(-3, 5); N=(-53/20, 101/20); P=(-1, -4); Q=(-23/10, 51/10); T=(-63/20, 81/20); U=(-33/20, 141/20); Y=(-2, 3).
Question: Are points T, N, and U collinear?
Answer is either Yes or No.
```

**Answer:**
```
Yes
```

---

## [metamath_entailment](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/math_metamath.py)

<!-- behavior-hash: c23e14bdc3047f72 -->

Choose which structurally matched premise set derives a conjecture.

**Prompt:**
```
Which premise set makes the conjecture follow using only the listed rules?
Exactly one of A and B is sufficient.
Rules instantiate only by renaming variables.
The answer is A or B.

Premise Set A:
1. P1(x, D1)
2. P1(x, y)

Premise Set B:
1. P1(x, D1)
2. P1(y, x)

Allowed Rules:
r1: P1(z, D1) ==> P1(z, D4)
r2: P1(y, D5) ==> P3(P2(F1(y), C0), P2(y, D3))
r3: P1(z, D4); P1(y, z) ==> P1(y, D5)

Conjecture:
P3(P2(F1(y), C0), P2(y, D3))
```

**Answer:**
```
B
```

---

## [metamath_core_select](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/math_metamath.py)

<!-- behavior-hash: c23e14bdc3047f72 -->

**Prompt:**
```
Which option is sufficient to derive the conjecture?
Use only the listed premises and rules. No hidden background facts.
Rules may only rename variables, not substitute compound terms.
The answer is A, B, C, or D.

Premises:
1. P1(x, D1)
2. P1(y, D2)

Rule Catalog:
- r1: P1(x, D2); P1(y, D2) ==> P3(P2(y, C0), P4(F2(F1(x, y), y), x))
- r2: P4(z, F3(x)) ==> P1(z, D4)
- r3: P1(x, D1) ==> P1(x, D2)
- r4: P1(x, D2); P1(y, D2) ==> P5(P4(F4(x, y), C0), P4(x, y))

Conjecture:
P3(P2(y, C0), P4(F2(F1(x, y), y), x))

Options:
A. [r1, r3]
B. [r2, r3]
C. [r3, r4]
D. [r1]
```

**Answer:**
```
A
```

---

## [lambda_reduction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/binding.py)

<!-- behavior-hash: 16af264fca09c34c -->

Reduce lambda calculus terms to normal form with renaming and shadowing.

**Prompt:**
```
Reduce the following untyped λ-term to β-normal form.
Syntax: `\x.body` is λx.body; juxtaposition is left-associative application; free identifiers are constants.

Term: ((\_0.((\_3.(((\_5.(d (((\_1._5) ((\_2.b) (_5 b))) ((_5 ((_5 _0) (_3 _5))) d)))) a) (\v0.c))) (b c))) (\v0.((\_4.v0) a)))

The answer is the β-normal form (compared up to α-equivalence).
```

**Answer:**
```
((d (a ((a ((a (\x0.x0)) ((b c) a))) d))) (\x1.c))
```

---

## [rewrite_system](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/binding.py)

<!-- behavior-hash: 16af264fca09c34c -->

Normalize term rewrite systems under boolean, list, logic, or path rules.

**Prompt:**
```
Normalize by the ordered rewrite rules. At each step, scan subterm positions outermost-first and left-to-right. Stop at the first position matched by at least one rule, then apply the earliest matching rule in the listed order (position priority first; rule priority second).

Rules:
- or(X,true) -> true
- or(X,false) -> X
- and(X,true) -> X
- if(true,X,Y) -> X
- and(true,X) -> X

Term:
and(if(true,false,and(true,b)),if(true,c,false))

The answer is the normal form.
```

**Answer:**
```
and(false,c)
```

---

## [unification_entailment](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/binding.py)

<!-- behavior-hash: 16af264fca09c34c -->

Decide if an equality is implied by the most general unifier of equations.

**Prompt:**
```
Compute a most general unifier of the equations. Apply it to both sides of the candidate equality. Answer Yes if the instantiated candidate terms are identical, otherwise answer No. The equations are guaranteed to be unifiable.

Equations:
- g(x0) = g(c)

Candidate:
x0 = c
```

**Answer:**
```
Yes
```

---

## [most_probable_evidence](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/probabilistic_reasoning.py)

<!-- behavior-hash: ed992c8f6d367267 -->

Find the most probable configuration of hidden variables given evidence.

**Prompt:**
```
Factor a is independently true with probability 0.7.
Factor b is independently true with probability 0.7.
Factor c is independently true with probability 0.2.
The observation holds exactly when (factor c or (factor a and factor b is false)).
We observe it.
Which hidden fact values form the most probable complete explanation?

Hidden fact values:
0. not a
1. a
2. not b
3. b
4. not c
5. c

Choose one value for each hidden factor. Answer with space-separated indexes.
```

**Answer:**
```
1 2 4
```

---

## [most_probable_outcome](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/probabilistic_reasoning.py)

<!-- behavior-hash: ed992c8f6d367267 -->

Predict the most probable outcome or select hidden factor values in ProbLog.

**Prompt:**
```
A container has 8 red items, 2 blue items, 8 green items.
Draw 3 items in sequence.
After draw 1, replace the item before the next draw.
After draw 2, do not replace the item.
No draw result is observed in advance.
Which statement is more likely?
A: all 3 draws are red.
B: all 3 draws are green.

The answer is exactly one of: A, B, equal.
```

**Answer:**
```
equal
```

---

## [multistep_nli](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic_depth.py)

<!-- behavior-hash: f7d1c90f99134e14 -->

Multi-hop natural language inference over chained logic facts and rules.

**Prompt:**
```
Premise:
bruno is a parent of alice.
alice is a parent of david.
Whenever x is a parent of y, x is an ancestor of y.
Parent relations followed by ancestor relations imply ancestor relations.
From x is a spouse of y, it follows that y is a spouse of x.

Hypothesis:
bruno is an ancestor of david.

Is the hypothesis true given the premise? The answer is Yes, No, or Maybe.
```

**Answer:**
```
Yes
```

---

## [defeasible_nli](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic_depth.py)

<!-- behavior-hash: f7d1c90f99134e14 -->

NLI using defeasible logic rules and negation as failure.

**Prompt:**
```
An `unless` condition must be shown to block its rule.

Facts:
Clara is trained and a bird.
Bruno is trained, blocked, a bird, a penguin, and careful.
David is a penguin and blocked.
Alice is trusted.
Clara helps Bruno.

Rules:
Trained people are trusted unless blocked.
Trusted people are approved unless flagged.
Blocked people are not trusted.
People who are not trusted are not approved unless flagged.
Birds are approved unless abnormal.
Penguins are abnormal.
People who help a trusted person are approved unless flagged.

Hypothesis:
David is not approved.

Is the hypothesis true? Answer Yes, No, or Maybe.
```

**Answer:**
```
Yes
```

---

## [multistep_evidence_retrieval](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic_depth.py)

<!-- behavior-hash: f7d1c90f99134e14 -->

Retrieve the specific premise indexes required to prove a logical hypothesis.

**Prompt:**
```
Premise:
[0] david is a parent of alice.
[1] alice is a parent of bruno.
[2] bruno is not adult.
[3] From x is a parent of y, it follows that x is an ancestor of y.
[4] Whenever x is a parent of y and y is an ancestor of z, x is an ancestor of z.

Hypothesis:
david is not an ancestor of bruno.

Which premise statements are necessary to contradict the hypothesis, meaning removing any one of them breaks that result?
Answer with space-separated indexes.
```

**Answer:**
```
0 1 3 4
```

---

## [multistep_abduction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic_depth.py)

<!-- behavior-hash: f7d1c90f99134e14 -->

Find the missing facts from candidates to satisfy a target hypothesis.

**Prompt:**
```
Premise:
[0] alice is trusted.
[1] alice is verified.
[2] All things that are trusted are approved.
[3] Being approved implies being not careful.

Hypothesis:
david is careful.

Candidate Facts:
[0] clara is trusted.
[1] david is verified.
[2] david is trusted.
[3] bruno is careful.
[4] clara is active.
[5] david is not trusted.

Which smallest set of candidate facts, if added to the premise, make the premise contradict the hypothesis?
Do not include candidate facts that are not needed.
Answer with space-separated indexes.
```

**Answer:**
```
2
```

---

## [logic_qa](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic_depth.py)

<!-- behavior-hash: f7d1c90f99134e14 -->

Answer multi-step logical reasoning queries over rule-based theories.

**Prompt:**
```
Premise:
Alice is charlie tagged.
Alice is alpha tagged.
Clara is bravo tagged.
Whenever x is charlie tagged and x is alpha tagged, x is echo tagged.
For all x, if x is echo tagged, then x is foxtrot tagged.

Question:
How many entities can be shown to be foxtrot tagged?

Answer with one integer.
```

**Answer:**
```
1
```

---

## [logic_derivation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic_derivation.py)

<!-- behavior-hash: 910fcba0f7acd357 -->

Produce a canonical forward proof trace for a logical target.

**Prompt:**
```
Premise:
0: clara helps alice.
1: alice advises bruno.
2: bruno trusts clara.
3: For all x, y, if x advises y, then y trusts x.
4: From x trusts y and y helps z, it follows that x advises z.

Target:
alice trusts bruno.

Give derivation lines as Rule: Input... => Deduction.
Use premise IDs and @0, @1, ... for derived lines.
```

**Answer:**
```
4: 2 0 => bruno advises alice
3: @0 => alice trusts bruno
```

---

## [planning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/planning.py)

<!-- behavior-hash: 6d63a921f5474083 -->

Generate action plans to achieve goals in domains like Blocksworld.

**Prompt:**
```
Objects:
object_1, object_2, object_3

Actions:
action_2(x0, x1)
  Requires: fluent_0(x1)
  Effect: fluent_0(x0)
action_4(x0)
  Effect: fluent_0(x0)

Initial state:
True values: None
All facts not listed under True values are false.

Goal:
fluent_0(object_3)

Action format example: action_0(object1, object2).
The answer is a shortest valid plan, one action per line.
```

**Answer:**
```
action_4(object_3)
```

---

## [set_missing_element](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

<!-- behavior-hash: 68a061870893e11f -->

Identify missing elements from a shuffled sequence defined by set intension.

**Prompt:**
```
Answer with the missing elements in the ordered span of {466, 467, 465, 470, 468, 464, 471, 472, 463} as a Python set.
```

**Answer:**
```
{469}
```

---

## [set_expression](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

<!-- behavior-hash: 68a061870893e11f -->

Evaluate complex set expressions involving union, intersection, and nested lists.

**Prompt:**
```
A = {'administrative funeral', 'fair consequence', 'nice association', 'sexual recording', 'heavy tip', 'nasty address', 'emotional group', 'broad extent'}
B = {'gross attention', 'broad extent', 'sexual recording', 'guilty politics', 'sweet way', 'emotional group', 'complete damage', 'male bit'}
Evaluate (A | B).
```

**Answer:**
```
{'administrative funeral', 'broad extent', 'complete damage', 'emotional group', 'fair consequence', 'gross attention', 'guilty politics', 'heavy tip', 'male bit', 'nasty address', 'nice association', 'sexual recording', 'sweet way'}
```

---

## [sequential_induction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/sequential_induction.py)

<!-- behavior-hash: b4b67b4fcfc6a194 -->

Infer the canonical recurrence in a bounded polynomial DSL.

**Prompt:**
```
Infer U[n]. Max recurrence degree: 0. Ops: +, -, *.
Use n. Give the simplified polynomial RHS.
Sequence: [0, 31, 62, 93, 124, 155, 186, 217]
The answer is the RHS only.
```

**Answer:**
```
31 * n
```

---

## [qualitative_reasoning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/qstr.py)

<!-- behavior-hash: 029c65d1c05e5ead -->

Solve qualitative spatial and temporal reasoning problems over algebras.

**Prompt:**
```
There are 5 objects: E0, E1, E2, E3, E4.
They have distinct ages.
Facts:
- E2 is newer than E3.
- E3 is newer than E0.
- E4 is the 5th-newest.
- E0 is newer than E1.

Which object is the 4th-newest?
The answer is one object label.
```

**Answer:**
```
E1
```

---

## [grid_navigation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grid_navigation.py)

<!-- behavior-hash: 4bdc8a43b7bbbe26 -->

Infer object grid coordinates from spatial relations and step actions.

**Prompt:**
```
Grid [0,4]x[0,4], N=+y, E=+x.
Initial Facts:
- C is above B.
- C is in the same column as A.
- B is below A.
- C is below A.
- A is right of B.
- B is left of C.

Steps:
1. B and A swap positions.

What is the final spatial relation of A to C? The answer is (horizontal, vertical), where horizontal is left/right/aligned and vertical is above/below/aligned.
```

**Answer:**
```
(left, below)
```

---

## [reference_tracking](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/tracking.py)

<!-- behavior-hash: 4f3376f005d950c5 -->

Track locations of balls in boxes across moves, swaps, and coreferences.

**Prompt:**
```
Inventory:
- b1: blue
- b2: green
- b3: black
- b4: yellow

Initial State:
- b1 is in x2
- b2 is in x1
- b3 is in x2
- b4 is in x1

Moves:
- Relocate b3 from x2 to x1.
- Move b4 from x1 to x2.
- Transfer b3 from x1 into x2.
- Transfer b2 from x1 into x3.
Where is b4 now? The answer is a box tag, like x1.
```

**Answer:**
```
x2
```

---

## [belief_tracking](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/belief_tracking.py)

<!-- behavior-hash: 1c67a3f84e8f539f -->

Track ordered beliefs through observation and communication.

**Prompt:**
```
Initially, everyone knows that the ring is in the drawer.

Story: Alice moves the ring to the drawer. No one else sees the move. Alice moves the ring to the box. Alice and Eve watch together and can see one another. Alice moves the ring to the drawer. Alice and Eve watch together and can see one another. Alice moves the ring to the tin. Alice and Eve watch together and can see one another. Eve sends Dave the message "I think the ring is in the box". Dave confirms receipt. Eve sends Dave a message stating exactly what Eve believes about the location of the ring. Dave confirms receipt. Alice moves the ring to the box. Unknown to the others, Eve watches through a window. Eve sends Dave the message "I think the ring is in the box", but it is not delivered.

Question: Where does Dave think Eve thinks the ring is?

Answer with one container name.
```

**Answer:**
```
tin
```

---

## [coreference](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/coreference.py)

<!-- behavior-hash: 0d003634bece281b -->

Resolve references through ordered groups, later evidence, and branches.

**Prompt:**
```
(1) A quiet stern engineer named Sam watched a kind loud teacher named Noah.
(2) A kind young pilot named Alan met Noah.
(3) Noah called the pilot.
(4) Sam watched Noah.
(5) The teacher questioned an old stern chef named Mary.
(6) The teacher helped her.
(7) An old tall engineer named Adam helped her.

In sentence 7, what does the object expression 'her' refer to?
The answer is the person's name.
```

**Answer:**
```
Mary
```

---

## [constraint_satisfaction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/constraint_satisfaction.py)

<!-- behavior-hash: 8d908648ac925aad -->

Solve query-aware assignment, graph, scheduling, grid, set, and numeric CSPs.

**Prompt:**
```
In this 3x3 grid, each row and column contains 1..3 once.

Constraints:
1. (r2c1 = 3) xor (r2c1 < r1c1)
2. r3c2 = 2
3. r1c3 < r1c2

Question: What is r2c1?
Answer with one name or integer.
```

**Answer:**
```
3
```

---

## [graph_pathfinding](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

<!-- behavior-hash: 395bdd34b851e9bb -->

Find the shortest path or cost in weighted and unweighted directed graphs.

**Prompt:**
```
Find the shortest directed path from node 0 to node 1. Break ties lexicographically. Return space-separated nodes.

Graph:
0: 0->4 0->5; 1: 1->0 1->2; 2: 2->0 2->1 2->3; 3: 3->2; 4: 4->1 4->2 4->3 4->5; 5: 5->1 5->3 5->4
```

**Answer:**
```
0 4 1
```

---

## [graph_successors](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

<!-- behavior-hash: 395bdd34b851e9bb -->

Determine the k-th successor of a node in a permutation digraph topology.

**Prompt:**
```
For each query (x, k), give the k-th successor of x by following directed edges k times.
Answer with space-separated integers in query order.

Graph:
digraph { 0->5; 1->1; 2->3; 3->4; 4->2; 5->0 }

Queries:
[(0, 2)]
```

**Answer:**
```
0
```

---

## [regex_following](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

<!-- behavior-hash: d6859e492a128066 -->

Produce a string that matches a specified regular expression pattern.

**Prompt:**
```
The answer is the shortest non-empty visible non-whitespace ASCII string that fully matches this regular expression, with lexicographic tie-breaks: [J-k]+
```

**Answer:**
```
J
```

---

## [regex_reasoning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

<!-- behavior-hash: d6859e492a128066 -->

Reason about regular expression equivalence, containment, and witnesses.

**Prompt:**
```
A = a|a?
B = bc|cbb
Is every string accepted by A also accepted by B?
The answer is Yes or No.
```

**Answer:**
```
No
```

---

## [analogical_case_matching](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/formal_analogies.py)

<!-- behavior-hash: ed0ad5234d7bf6bc -->

Retrieve analogical cases matching query objects, links, and logical facts.

**Prompt:**
```
Which case can be embedded into Query? A case matches when every fact maps to a Query fact under one-to-one entity and relation renaming, with an optional consistent direction reversal for each relation. Query may contain additional facts. Answer with its ID.

M0: a alpha b, c alpha a, c alpha b, d alpha c
M1: a alpha b, a alpha c, b alpha c, d beta a
M2: a alpha c, e alpha a, e alpha d, a beta d
Query: u delta z, y delta u, y delta z, u gamma v, x gamma y, z gamma u
```

**Answer:**
```
M1
```

---

## [parsing_derivation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

<!-- behavior-hash: fc496aa462366030 -->

Determine the derivation production rule sequence parsing a given string.

**Prompt:**
```
(START)
S

(GRAMMAR)
R0: A ::= 'continue' A
R1: S ::= S A
R2: A ::= 'man'
R3: S ::= A

(STRING)
continue continue man

(QUESTION)
The answer is the rule labels used in the leftmost derivation of STRING, in order, separated by spaces.
```

**Answer:**
```
R3 R0 R0 R2
```

---

## [syntax_error_detection](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

<!-- behavior-hash: fc496aa462366030 -->

Locate syntax errors or grammatical perturbations in generated sentences.

**Prompt:**
```
(START)
S

(GRAMMAR)
A -> '<' A '>'
S -> A S
A -> 'answer'
A -> 'individual' A
S -> A
A -> A 'individual'
A -> 'edge' A
A -> 'quite' A

(STRING)
< < quite > individual > > individual individual

Answer OK, INCOMPLETE, or ERROR token for the first invalid token. If that token repeats in STRING, append its 1-based occurrence as @occurrence.
```

**Answer:**
```
ERROR >@1
```

---

## [constrained_continuation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

<!-- behavior-hash: fc496aa462366030 -->

Complete a uniquely determined fixed-length span using a formal grammar.

**Prompt:**
```
Complete <HOLE> according to the grammar.

GRAMMAR:
S ::= A
C ::= 'it'
A ::= B
A ::= '<' A '>'
B ::= C

SENTENCE:
< < <HOLE>

Return only the missing 3 tokens.
```

**Answer:**
```
it > >
```

---

## [table_qa](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/table_qa.py)

<!-- behavior-hash: 8e7b4b92489b4c3e -->

Answer queries on tabular data by executing SQL queries over dataframes.

**Prompt:**
```
Execute this SQL query on the table named dataframe:

Table 1:
discount	status	country
0	paid	France
0.1	cancelled	France
0.05	paid	Italy
0.2	pending	Italy
0.3	pending	Italy
0.1	paid	Italy
0.2	cancelled	Italy
0.05	pending	Italy
0	pending	Italy


In this table, — represents SQL NULL.

SQL: SELECT "status" FROM dataframe WHERE "country" = 'France'

The answer is the result as CSV format (rows separated by newlines, values by commas). Do not include column headers..
```

**Answer:**
```
paid
cancelled
```

---

## [table_equivalence](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/table_qa.py)

<!-- behavior-hash: 8e7b4b92489b4c3e -->

Decide if two rendered tables are semantically equivalent under mutations.

**Prompt:**
```
Do these tables contain the same data?
Ignore row order, column order, and table syntax; match values by column name.
Treat numeric grouping and trailing zeros as formatting, ISO and English month-name dates as dates, and — and NULL as missing. Repeated rows count.

Table A:
event_date    amount    country         rating    customer
Oct 18, 2025  2,111.98  Bulgaria        4.90      Kimberly Lawrence
Dec 14, 2025  1,744.19  Cuba            4.90      Brian Davis
Oct 21, 2024  1,714.95  New Zealand     2.70      Cheryl Long
Sep 08, 2024  754.69    Nauru           1.80      Gerald Maynard
Nov 16, 2024  1,000.62  Iceland         1.60      Mary Pham
Oct 03, 2024  1,644.70  Moldova         4.50      Angela Holder
Feb 24, 2025  1,770.16  Comoros         3.60      Catherine Harris
Jun 14, 2026  2,239.30  Gambia          4.50      Adam Moore
NULL          2,260.59  American Samoa  4.70      Carol Hurley
Oct 18, 2025  2,111.98  Bulgaria        4.90      Kimberly Lawrence

Table B:
event_date	country	customer	rating
2024-09-08	Nauru	Gerald Maynard	1.8
—	American Samoa	Carol Hurley	4.7
2025-12-14	Cuba	Brian Davis	4.9
2024-10-03	Moldova	Angela Holder	4.5
2024-10-21	New Zealand	Cheryl Lone	2.7
2025-10-18	Bulgaria	Kimberly Lawrence	4.9
2024-11-16	Iceland	Mary Pham	1.6
2025-10-18	Bulgaria	Kimberly Lawrence	4.9
2025-02-24	Comoros	Catherine Harris	3.6


Answer Yes or No.
```

**Answer:**
```
No
```

---

## [table_statistics](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/table_qa.py)

<!-- behavior-hash: 8e7b4b92489b4c3e -->

Select rows or columns using associations, conditioning, group robustness, and shifts.

**Prompt:**
```
Table:
W,K,A,H,T
-4.03,-3.75,-1.35,-0.93,2.48
-1.77,-4.55,0.68,-3.39,0.56
-3.61,-3.39,0.42,-2.47,0.16
0.08,0.6,-0.13,-0.14,-2.57
0.08,-1.77,0.65,-1.35,0.95
-1.24,-0.32,-0.56,0.31,0.66
0.94,0.37,1.1,-0.57,0.06
2.44,0.23,2.31,0.47,0.9
2.07,3.73,-0.24,1.69,0.62
-2.99,-3.49,-0.31,0.71,-0.48
-2.09,-3.27,0.42,-1.82,-1.46
1.4,1.38,0.68,1.01,1.89


Find:
column name most associated with column W while controlling for K

Metric:
absolute partial Pearson correlation

Answer with only the identifier.
```

**Answer:**
```
A
```

---

## [string_transduction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/string_transduction.py)

<!-- behavior-hash: 7e37b3a7153847d2 -->

Apply string transduction operations including Caesar cipher and rotation.

**Prompt:**
```
String: dcadecbd
Operations:
- caesar shift by 4
- keep only g and f
Answer with the final string.
```

**Answer:**
```
ggf
```

---

## [game_best_move](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/game_playing.py)

<!-- behavior-hash: 4b4daac8084b238d -->

Determine the minimax-optimal move for a player in a finite graph-based game.

**Prompt:**
```
In this graph game, choose player's best move. Player chooses on player turns; opponent chooses on opponent turns. Opponent minimizes player score.

Start: n0. Turns alternate player, opponent. Move along one edge per turn, for at most 3 moves. Play ends upon reaching a leaf or the move horizon; in either case, player's score is the current node's payoff. Node payoffs: n0:70; n1:0; n2:70; n3:80; n4:10; n5:40; n6:60. Edges: n0->n1,n5; n1->n6; n2->n4; n3->n5,n6.
Legal player moves now: n1, n5.
The answer is the destination node of the best move.
```

**Answer:**
```
n1
```

---

## [game_forced_win](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/game_playing.py)

<!-- behavior-hash: 4b4daac8084b238d -->

Decide if a player can force a win from a given state in a graph-based game.

**Prompt:**
```
In this graph game, decide whether player can force a win. Player chooses on player turns; opponent chooses on opponent turns. Opponent minimizes player score. A win means final player score is greater than 50.

Start: n0. Turns alternate player, opponent. Move along one edge per turn, for at most 3 moves. Play ends upon reaching a leaf or the move horizon; in either case, player's score is the current node's payoff. Node payoffs: n0:50; n1:20; n2:80; n3:90; n4:90; n5:0; n6:100. Edges: n0->n1,n2; n1->n3; n2->n5,n6; n3->n4,n6.
The answer is Yes or No.
```

**Answer:**
```
Yes
```

---

## [qualitative_causal_reasoning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/qualitative_causal_reasoning.py)

<!-- behavior-hash: c5133ef287038003 -->

Reason qualitatively about causal effects and associations in graphs.

**Prompt:**
```
Assume linear causal relations, independent noise, and no exact cancellations.

- X1 directly increases X4.
- X10 directly decreases X3.
- X4 directly decreases X2.
- X7 directly increases X5.
- X9 directly increases X0.

If we intervene to increase X1, what happens to X2?
Answer with: increase, decrease, no_effect, or ambiguous.
```

**Answer:**
```
decrease
```

---

## [code_analysis](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/code_analysis.py)

<!-- behavior-hash: ad434bfeb17e2165 -->

Analyze toy finite-state Python-like programs with CTL temporal formulas.

**Prompt:**
````
Program:
```python
y, count = 0, 0

def step():
    global y, count
    count, y = min(count + 1, 1), 1 if count >= 1 else y

```

Start from the assignments above; each transition calls `step()`.

Formula: AF(p0)
Property: on every execution, eventually (y == 1)

When does the initial state first enter the least fixed point?
Iteration 0 contains states satisfying the inner condition. Answer with an integer or never.
````

**Answer:**
```
2
```

---

## [code_runnability](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/code_execution.py)

<!-- behavior-hash: 9a2062859a1b2a7b -->

Predict if a given Python code snippet runs successfully or raises an exception.

**Prompt:**
````
Predict whether this Python call runs successfully or raises an exception.
```python
def fn4(p5, p6):
    return p5 * -2

def endpoint(arg1, arg2, arg3):
    seq7 = [arg1, arg2, arg3, 1, arg2 if arg3 < arg1 else arg1 // arg2, arg2]
    acc8 = arg1 * 2 // (abs(3 if -3 == arg1 else arg2) + 1)
    seq7[abs(arg2) % 6] -= 0 // (abs(acc8) + 1)
    acc8 += len(seq7)
    ref9 = seq7
    ref9[2] += -1
    acc8 += seq7[2]
    map10 = {0: len(seq7), 1: seq7[abs(arg2) % 6]}
    acc8 += map10.get(len(seq7), 0)
    ref9.append(fn4(arg3, acc8))
    acc8 += len(ref9)
    i11 = 0
    while i11 < 2:
        acc8 += arg2 % (abs(arg2) + 1)
        i11 += 1
    return acc8

```
Call: `endpoint(-3, -2, 0)`
The answer is `OK` if it runs successfully; otherwise the exception class name.
````

**Answer:**
```
OK
```

---

## [code_execution](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/code_execution.py)

<!-- behavior-hash: 9a2062859a1b2a7b -->

Predict the return value or stdout of executing generated Python code blocks.

**Prompt:**
````
Predict the value returned by this Python call.
```python
def endpoint(arg1, arg2):
    seq6 = [arg1, arg2 // arg2, arg2, 3 + arg1, arg2, arg1]
    acc7 = len(seq6) * -3
    try:
        acc7 += seq6.index(acc7)
    except ValueError:
        acc7 -= len(seq6)
    ref9 = seq6
    ref9[0] += -2
    acc7 += seq6[0]
    i8 = 0
    while i8 < 2:
        acc7 += len(seq6)
        i8 += 1
    acc7 += ref9.index(ref9[1])
    return acc7

```
Call: `endpoint(-4, 2)`
The answer is the exact Python `repr` of the returned value.
````

**Answer:**
```
-17
```

---

## [program_synthesis](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/code_program_synthesis.py)

<!-- behavior-hash: 199f17d08656bf4f -->

**Prompt:**
```
Write f(s: str) -> str.

Target: return the minimum-cost StringFrag-v1 expression matching the examples.

Always allowed: s, string literals "", " ", "-", "_", and integer literals 0, 1, 2, 3.
Allowed operators for this problem:
- concat: str + str
- eq_str: str == str
- not: not bool
Bounds: strings have length <= 64; integers are between -16 and 64. Use Python string semantics.
Cost: AST nodes, then operator-count tuple in this global order (concat, substr, replace1, ite, len, find, add, sub, contains, eq_str, lt, not), then source length, then lexicographic source order.

Examples:
f(' ') = '   '
f('abc') = '  abc'

Return only:
def f(s: str) -> str:
    return <expression>
```

**Answer:**
```
def f(s: str) -> str:
    return (" " + (" " + s))
```

---

## [backtracking_search](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/generated/backtracking_search.py)

<!-- behavior-hash: 794eed9f3f6f8c78 -->

Find the first finite-domain solution under deterministic backtracking with forward checking.

**Prompt:**
```
Variables X1..X5 each range over 1..4.
Constraints: X1 < X3; X3 != X4; X2 - X3 != 1; X3 + X4 != 5; X3 + X5 != 2; X3 > X4; X1 < X4
Search variables in order X1,X2,... and values in increasing order. After each assignment, remove from every later domain values that violate a constraint with the new assignment; backtrack immediately if a domain becomes empty.
What is the first complete solution found? The answer is the space-separated values of X1..Xn.
```

**Answer:**
```
1 1 4 2 1
```

---

## [boolean_propagation_search](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/generated/boolean_propagation_search.py)

<!-- behavior-hash: 78cb65ce574c1bb8 -->

Find the canonical first Boolean model under propagation and backtracking.

**Prompt:**
```
Formula: (x2 or not x3) and (not x5 or x2) and (x5 or not x1) and (not x4 or not x1 or x2) and (x3 or not x4) and (not x5 or not x2) and (x4 or x2 or x1)
Choose unassigned variables x1,x2,... in order and try False before True. Before each choice, repeatedly assign any value forced by a one-unassigned-literal clause; if several are forced, use the smallest variable first. Backtrack on contradiction.
What is the first satisfying assignment found? The answer is 5 space-separated T/F values for x1..x5.
```

**Answer:**
```
F T F F F
```

---

## [controlled_code_execution](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/generated/controlled_code_execution.py)

<!-- behavior-hash: 2dee25442d3b594b -->

Execute Python programs generated to require controlled semantic phenomena.

**Prompt:**
````
Predict the value returned by this Python call.
```python
def endpoint():
    state = [-2, 2, -2]
    alias0 = state
    alias0[2] += -4
    state[0] += alias0[2]
    bias1 = 1
    def f1(x):
        return x + bias1 + state[2]
    bias1 += 2
    state[0] = f1(state[0])
    return state
```
Call: `endpoint()`
The answer is the exact Python `repr` of the returned value.
````

**Answer:**
```
[-11, 2, -6]
```

---

## [dynamic_programming](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/generated/dynamic_programming.py)

<!-- behavior-hash: 66849ade8e988eb7 -->

Evaluate a max-sum dynamic program and reconstruct its optimal state sequence.

**Prompt:**
```
States: A B C
Observations: 1 2 2 2 1
Start: A=1 B=2 C=1
Transitions (rows=from, columns=A B C):
A: 3 1 -2
B: -1 0 2
C: 3 1 -2
Emissions (rows=state, columns=0..2):
A: -2 -3 -2
B: -3 3 2
C: 0 1 1
Score a state sequence by start + emissions + transitions. Find the maximum-score sequence; ties are lexicographic. The answer is the space-separated state labels.
```

**Answer:**
```
B C B C B
```

---

## [fixpoint_iteration](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/generated/fixpoint_iteration.py)

<!-- behavior-hash: 51bcd6d4cbb7ab36 -->

Compute a least fixpoint of monotone finite-set propagation rules.

**Prompt:**
```
Universe: 0..6. Start: X0={2,4}; X1={}; X2={2,3}; X3={2,3,6}
Rules: X0 |= shift(X1,+0); X3 |= shift(X1,-2) & {3,4,6}; X3 |= shift(X2,+0) & {3,4,5,6}; X0 |= shift(X3,+2); X2 |= shift(X1,+2); X1 |= shift(X2,+0); X1 |= shift(X3,+2)
Apply the rules repeatedly in listed order until no set changes. shift(S,d) = {x+d in the universe : x in S}.
What is X1 at the fixed point? The answer is a sorted set like {0,2,5}.
```

**Answer:**
```
{2,3,4,5,6}
```

---

## [matrix_induction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/generated/matrix_induction.py)

<!-- behavior-hash: 57ada1dea937871f -->

Infer a missing multi-attribute matrix cell under a certified finite rule family.

**Prompt:**
```
Complete the missing cell of the 3x3 matrix. Each attribute is independent and uses one fixed rule for every row and every column. Encode each listed domain by indices 0,1,... in the shown order. For a row or column with encoded values a,b,c, c is obtained from a,b by one of: left=a; right=b; min=min(a,b); max=max(a,b); add+t=(a+b+t) mod k for some t; xor+t=a xor b xor t for some t (xor is used only for power-of-two domain sizes). Different attributes may use different rules.
Domains:
- count: 1, 2, 3, 4
Matrix:
count=3 | count=4 | count=4
count=3 | count=1 | count=1
count=3 | count=1 | ?
The answer is the missing cell written with exactly the displayed attribute names as name=value pairs.
```

**Answer:**
```
count=1
```

---

## [pattern_induction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/generated/pattern_induction.py)

<!-- behavior-hash: 432c29d6e83dcf99 -->

Infer a shared symbolic sequence rule from examples and predict a uniquely determined continuation.

**Prompt:**
```
Infer one shared rule and continue the query.
Letters are cyclic in this order: A B C D E F.
Allowed rule family:
- Interleave m lanes, with m in 1..2. Before repetition, latent positions visit lanes 0,1,...,m-1 cyclically.
- Lane j has one fixed cyclic step s_j in {+1, -1}. Rows share m and all s_j, but may start from different letters.
- Direction is either straight, or uses one shared turn period p in {2, 3}. Straight uses occurrence multipliers 0,1,2,...; turn p uses 0,1,...,p,p-1,...,1,0,... periodically.
- Repeat every emitted letter r times, with r in 1..2. Rows share r and the turn choice.
Examples:
1. A F F A E B F -> A A F
Query: B C A D F
The answer is the next 2 letters, space-separated.
```

**Answer:**
```
E A
```

---

## [schema_bound_query](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/generated/response_contracts.py)

<!-- behavior-hash: b50cca7bfb835ccc -->

Execute a record query while satisfying a sampled exact nested JSON schema.

**Prompt:**
```
Records:
R1: group=B, value=5
R2: group=B, value=8
R3: group=C, value=-2
R4: group=C, value=0
R5: group=A, value=-5
R6: group=A, value=-6

Select records with group=A and value >= -9, preserving input order.
Answer as JSON matching exactly this schema, with no extra keys or prose:
{"ids":[string,...],"count":integer,"total":integer}
```

**Answer:**
```
{"ids":["R5","R6"],"count":2,"total":-11}
```

---

## [conditional_response_contract](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/generated/response_contracts.py)

<!-- behavior-hash: b50cca7bfb835ccc -->

Solve a selection problem and execute output transformations whose activation depends on the semantic result.

**Prompt:**
```
Records:
R1: score=9, eligible=yes, group=A, flag=yes
R2: score=11, eligible=yes, group=C, flag=no
R3: score=12, eligible=yes, group=A, flag=no
R4: score=9, eligible=no, group=C, flag=no
R5: score=9, eligible=yes, group=B, flag=no
R6: score=0, eligible=no, group=B, flag=yes

Choose the eligible record with the largest score; break ties by lexicographically smallest ID. Start the answer as that ID. Then apply these rules in order to the current answer:
1. If the winner's score is even, append token EVEN; otherwise do nothing.
2. If the winner's group is C, prepend token GROUP; otherwise do nothing.
The answer is the final transformed string and nothing else.
```

**Answer:**
```
R3 EVEN
```

---

## [protected_span_transformation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/generated/response_contracts.py)

<!-- behavior-hash: b50cca7bfb835ccc -->

Select and transform records while preserving opaque protected spans byte-for-byte.

**Prompt:**
```
Items:
<B00:q-5> value=-8
<D01:F-8> value=4
<G02:M-6> value=-8
<l03:Y-7> value=-2
<m04:z-7> value=-8
<d05:S-5> value=-1

Keep exactly the items whose absolute original value is even. For each kept item compute -3*value + 3. Sort kept items by the computed value ascending, then by protected span. Each answer line is PROTECTED_SPAN=COMPUTED_VALUE. Copy every protected span exactly, including case and punctuation.
```

**Answer:**
```
<D01:F-8>=-9
<l03:Y-7>=9
<B00:q-5>=27
<G02:M-6>=27
<m04:z-7>=27
```

---

## [rule_switching](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/generated/rule_switching.py)

<!-- behavior-hash: 36cb3d8da189c40b -->

Track symbolic state while identical opcodes change meaning across active rule modes.

**Prompt:**
```
Maintain the register values while executing the program. The current mode determines each opcode's meaning. For an instruction on registers (a,b,c): rotate-left maps their values to (b,c,a); rotate-right to (c,a,b); swap-first-two to (b,a,c); swap-last-two to (a,c,b); and swap-outer to (c,b,a). Mode changes affect following instructions.
Modes:
M1: X=rotate-left, Y=swap-last-two, Z=swap-outer
M2: X=swap-first-two, Y=swap-outer, Z=rotate-right
Initial state: r1=A r2=B r3=C r4=D r5=E
Start mode: M1
Program:
Z r3 r5 r4
X r5 r2 r3
mode M2
X r5 r4 r1
Z r4 r5 r2
mode M1
Y r1 r2 r4
X r1 r2 r5
mode M2
Y r1 r5 r3
What value is in r3 after the program? The answer is one value label.
```

**Answer:**
```
D
```

---

## [shift_reduce_parsing](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/generated/shift_reduce_parsing.py)

<!-- behavior-hash: eae75112036e50bb -->

Execute a deterministic shift-reduce parser and report one compact stack state.

**Prompt:**
```
Rules:
R0: N0 -> d
R1: N1 -> b
R2: N2 -> N0 N1 N1
R3: N3 -> d
R4: N4 -> N3 N2
Input: d d b b
Shift tokens left to right. After every shift, repeatedly reduce the longest stack suffix matching a rule RHS; ties use the lowest rule number.
What is the stack after consuming 3 tokens? The answer is the stack symbols from bottom to top, space-separated.
```

**Answer:**
```
N0 N0 N1
```

---

## [spatial_folding](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/generated/spatial_folding.py)

<!-- behavior-hash: 82c53586c052ccc4 -->

Track hole positions while folding and unfolding a square grid.

**Prompt:**
```
A 4x4 square sheet is divided into unit cells. Rows are numbered top-to-bottom and columns left-to-right, starting at 1. After every fold, renumber the visible folded rectangle from its new top-left corner.
Folds, in order:
1. fold the top half over the bottom half.
After all folds the sheet is 2x4. Punch holes through cells: 1,4.
Unfold the sheet completely. The answer is all punched cells as row,column pairs separated by semicolons, in row-major order.
```

**Answer:**
```
2,4; 3,4
```

---

## [typed_relation_extraction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/generated/structured_extraction.py)

<!-- behavior-hash: f597e57e2e8902b6 -->

Extract the complete set of typed relations with sentence provenance while ignoring negated and irrelevant statements.

**Prompt:**
```
Statements:
1. Dara is visited by Cleo.
2. Fara supports Dara.
3. Fara does not precede Cleo.
4. Dara precedes Fara.
5. Enzo supports Cleo.
6. Ivo manages Fara.
7. Dara supports Enzo.

Extract every affirmative supports, manages, and precedes relation. Interpret reversed wording semantically. Ignore negated statements and all other relation types. The answer is a JSON array in evidence-sentence order. Each object has exactly the keys relation, source, target, evidence, where evidence is the sentence number.
```

**Answer:**
```
[{"relation":"supports","source":"Fara","target":"Dara","evidence":2},{"relation":"precedes","source":"Dara","target":"Fara","evidence":4},{"relation":"supports","source":"Enzo","target":"Cleo","evidence":5},{"relation":"manages","source":"Ivo","target":"Fara","evidence":6},{"relation":"supports","source":"Dara","target":"Enzo","evidence":7}]
```

---

## [variable_elimination](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/generated/variable_elimination.py)

<!-- behavior-hash: 88fdc7c1321f9eb9 -->

Execute deterministic fraction-free elimination and report a compact residual equation.

**Prompt:**
```
Equations:
-x2 + 3*x3 - 3*x4 = -7
-5*x1 - 5*x2 - 5*x3 + 3*x4 = 6
-x1 + 2*x2 - 5*x3 = 4
-4*x2 + 2*x3 - x4 = 5
Eliminate x1, x2 in that order. For each variable, use the first remaining equation with a nonzero coefficient as pivot. For every later row with coefficient b and pivot coefficient a, replace it by a*row - b*pivot; then divide the entire row by the gcd of its integer coefficients and make its first nonzero coefficient positive.
After these eliminations, what is row 3? The answer is one simplified equation.
```

**Answer:**
```
25*x3 - 48*x4 = -91
```

---
