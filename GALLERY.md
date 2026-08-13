# 📖 Task Gallery

50 tasks

[`arithmetics`](#arithmetics) · [`math_word_problem`](#math_word_problem) · [`equation_system`](#equation_system) · [`combinatorics_formula`](#combinatorics_formula) · [`function_manipulation`](#function_manipulation) · [`lean_missing_line`](#lean_missing_line) · [`lean_candidate_compilation`](#lean_candidate_compilation) · [`planar_geometry_relations`](#planar_geometry_relations) · [`metamath_entailment`](#metamath_entailment) · [`metamath_core_select`](#metamath_core_select) · [`lambda_reduction`](#lambda_reduction) · [`rewrite_system`](#rewrite_system) · [`unification_entailment`](#unification_entailment) · [`most_probable_evidence`](#most_probable_evidence) · [`most_probable_outcome`](#most_probable_outcome) · [`multistep_nli`](#multistep_nli) · [`defeasible_nli`](#defeasible_nli) · [`multistep_evidence_retrieval`](#multistep_evidence_retrieval) · [`multistep_abduction`](#multistep_abduction) · [`logic_qa`](#logic_qa) · [`logic_derivation`](#logic_derivation) · [`planning`](#planning) · [`set_missing_element`](#set_missing_element) · [`set_expression`](#set_expression) · [`sequential_induction`](#sequential_induction) · [`qualitative_reasoning`](#qualitative_reasoning) · [`grid_navigation`](#grid_navigation) · [`reference_tracking`](#reference_tracking) · [`belief_tracking`](#belief_tracking) · [`coreference`](#coreference) · [`constraint_satisfaction`](#constraint_satisfaction) · [`graph_pathfinding`](#graph_pathfinding) · [`graph_successors`](#graph_successors) · [`regex_following`](#regex_following) · [`regex_reasoning`](#regex_reasoning) · [`analogical_case_matching`](#analogical_case_matching) · [`parsing_derivation`](#parsing_derivation) · [`syntax_error_detection`](#syntax_error_detection) · [`constrained_continuation`](#constrained_continuation) · [`table_qa`](#table_qa) · [`table_equivalence`](#table_equivalence) · [`table_statistics`](#table_statistics) · [`string_transduction`](#string_transduction) · [`game_best_move`](#game_best_move) · [`game_forced_win`](#game_forced_win) · [`qualitative_causal_reasoning`](#qualitative_causal_reasoning) · [`code_analysis`](#code_analysis) · [`code_runnability`](#code_runnability) · [`code_execution`](#code_execution) · [`program_synthesis`](#program_synthesis)

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

<!-- behavior-hash: 1bed6234031d1594 -->

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

<!-- behavior-hash: 1bed6234031d1594 -->

Determine if a candidate proof body successfully closes a theorem in Lean.

**Prompt:**
```
Does this Lean 4 tactic body close the theorem?
The answer is True or False.

THEOREM:
theorem ex (a : Int) : 0 ≤ (3 * a : Int) * (3 * a : Int) := by
  ?

CANDIDATE:
exact mul_self_nonneg (3 * a : Int)
```

**Answer:**
```
True
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

<!-- behavior-hash: 4edd58664bbad533 -->

**Prompt:**
```
Does the conjecture follow using only the listed premises and rules?
Rules instantiate only by renaming variables.
The answer is True or False.

Premises:
1. P1(x, D1)
2. P1(x, D2)

Allowed Rules:
r1: P1(x, D2) ==> P1(x, D1)
r2: P1(x, D1); P1(y, D1) ==> P2(F1(x, F2(y)), F3(x, y))

Conjecture:
P2(F1(x, F2(y)), F3(x, y))
```

**Answer:**
```
False
```

---

## [metamath_core_select](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/math_metamath.py)

<!-- behavior-hash: 4edd58664bbad533 -->

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
discount,qty
0.30,2
0.00,3
0.20,4
0.20,5
0.20,6
0.00,7
0.10,8
0.30,9
0.00,10
0.00,11
0.05,12
0.00,13


In this table, — represents SQL NULL.

SQL: SELECT ROUND(SUM("qty"), 2) FROM dataframe WHERE TRUE

The answer is the result as a single number without display formatting.
```

**Answer:**
```
90.0
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
label|J|Y|X|T
L0|L2|L0|L0|L0
L0|L0|L0|L0|L0
L2|L2|L0|L2|L2
L1|L2|L1|L1|L1
L2|L0|L0|L2|L1
L2|L2|L2|L2|L2
L1|L1|L1|L1|L0
L1|L1|L1|L1|L1
L2|L0|L2|L2|L2
L0|L0|L2|L0|L2
L0|L0|L1|L0|L0
L1|L1|L1|L1|L0


Find:
categorical column name most associated with column label

Metric:
normalized mutual information

Answer with only the identifier.
```

**Answer:**
```
X
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

<!-- behavior-hash: bbdf88e81797eb0a -->

Predict if a given Python code snippet runs successfully or raises an exception.

**Prompt:**
````
Predict whether this Python call runs successfully or raises an exception.
```python
def f0(c: int) -> list:
    assert 1 == 1
    return f1(c, "go")
def f1(f: int, y: str) -> list:
    print(f)
    f = 1 // f
    return [0, 1, 2]
def endpoint(x0: int) -> list:
    return f0(x0)

```
Call: `endpoint(-3)`
The answer is `OK` if it runs successfully; otherwise the exception class name.
````

**Answer:**
```
OK
```

---

## [code_execution](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/code_execution.py)

<!-- behavior-hash: bbdf88e81797eb0a -->

Predict the return value or stdout of executing generated Python code blocks.

**Prompt:**
````
Predict the value returned by this Python call.
```python
def f0(z: int, l: str) -> str:
    z = f"out={z}" + "go"
    return z
def f1(o: str) -> str:
    a = 0
    return o
def endpoint(x0: int, x1: str) -> str:
    return f0(x0, x1)

```
Call: `endpoint(1, 'ba')`
The answer is the exact Python `repr` of the returned value.
````

**Answer:**
```
'out=1go'
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
