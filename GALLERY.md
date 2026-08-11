# 📖 Task Gallery

50 tasks

[`arithmetics`](#arithmetics) · [`math_word_problem`](#math_word_problem) · [`equation_system`](#equation_system) · [`combinatorics_formula`](#combinatorics_formula) · [`function_manipulation`](#function_manipulation) · [`lean_missing_line`](#lean_missing_line) · [`lean_candidate_compilation`](#lean_candidate_compilation) · [`planar_geometry_relations`](#planar_geometry_relations) · [`metamath_entailment`](#metamath_entailment) · [`metamath_core_select`](#metamath_core_select) · [`lambda_reduction`](#lambda_reduction) · [`rewrite_system`](#rewrite_system) · [`unification_entailment`](#unification_entailment) · [`most_probable_evidence`](#most_probable_evidence) · [`most_probable_outcome`](#most_probable_outcome) · [`multistep_nli`](#multistep_nli) · [`defeasible_nli`](#defeasible_nli) · [`multistep_evidence_retrieval`](#multistep_evidence_retrieval) · [`multistep_abduction`](#multistep_abduction) · [`logic_qa`](#logic_qa) · [`logic_derivation`](#logic_derivation) · [`planning`](#planning) · [`set_missing_element`](#set_missing_element) · [`set_expression`](#set_expression) · [`sequential_induction`](#sequential_induction) · [`qualitative_reasoning`](#qualitative_reasoning) · [`grid_navigation`](#grid_navigation) · [`reference_tracking`](#reference_tracking) · [`belief_tracking`](#belief_tracking) · [`coreference`](#coreference) · [`constraint_satisfaction`](#constraint_satisfaction) · [`graph_pathfinding`](#graph_pathfinding) · [`graph_successors`](#graph_successors) · [`regex_following`](#regex_following) · [`regex_reasoning`](#regex_reasoning) · [`analogical_case_matching`](#analogical_case_matching) · [`parsing_derivation`](#parsing_derivation) · [`syntax_error_detection`](#syntax_error_detection) · [`constrained_continuation`](#constrained_continuation) · [`table_qa`](#table_qa) · [`table_equivalence`](#table_equivalence) · [`table_statistics`](#table_statistics) · [`string_transduction`](#string_transduction) · [`game_best_move`](#game_best_move) · [`game_forced_win`](#game_forced_win) · [`qualitative_causal_reasoning`](#qualitative_causal_reasoning) · [`code_analysis`](#code_analysis) · [`code_runnability`](#code_runnability) · [`code_execution`](#code_execution) · [`program_synthesis`](#program_synthesis)

---

## [arithmetics](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/arithmetics.py)

<!-- behavior-hash: f2b3d4e7e62e71bb -->

Compositional arithmetics with float/int/bool, varied operators, number theory.

**Prompt:**
```
Evaluate 0 * -14.
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
Yuki has 2 more stamps than Hana. Jon has 4 fewer stamps than Hana. Yuki has 11 stamps. How many stamps does Jon have? Answer with a number.
```

**Answer:**
```
5
```

---

## [equation_system](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/equation_system.py)

<!-- behavior-hash: 8781164d0945f615 -->

Solve systems of linear equations or detect inconsistent/underdetermined systems.

**Prompt:**
```
Solve the following system of equations for the variable 'X1'.

System:
  4*X1 + X2 + 93 = 0
  3*X1 + X2 + 66 = 0
  X1 + 27 = 0

The answer is the value of X1, or 'No solution' / 'Multiple solutions'.
```

**Answer:**
```
-27
```

---

## [combinatorics_formula](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/combinatorics.py)

<!-- behavior-hash: ea4ea56e7aa437ab -->

**Prompt:**
```
Write the counting expression. C(n,k) is unordered; P(n,k) is ordered.

Problem:
Place 9 identical tokens into 5 labeled boxes; empty boxes are allowed.

The answer must have the form:
C(X1,X2)
where:
X1 := 4 | 13 | 9
X2 := 14 | 5 | 4
Answer with the complete expression.
```

**Answer:**
```
C(13,4)
```

---

## [function_manipulation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/function_manipulation.py)

<!-- behavior-hash: 6cfeba2da9fd3a2c -->

Multistep symbolic function manipulation with composition, local inverses, calculus, and short exact answers.

**Prompt:**
```
Let $f(x)=-\frac{2}{3} + 2\left(x-2\right)$.
Define $h(x)=f\left(\left(x\right)-\left(2\left(x-2\right)\right)\right)$.
Compute $h'(2)$.
The answer is a reduced rational number.
```

**Answer:**
```
-2
```

---

## [lean_missing_line](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/math_lean.py)

<!-- behavior-hash: a9cebc298eaac71f -->

Complete a Lean proof with a uniquely valid constrained proof line.

**Prompt:**
```
Fill `__ANSWER__` with a Lean proof line. Mathlib is imported.

THEOREM:
theorem ex (s t u : Finset Nat) : t ∩ (s ∪ u) = (t ∩ s) ∪ (t ∩ u) := by
  __ANSWER__

The answer must have the form:
simpa using inf_sup_left X1 s X2
where:
X1 := s | u | t
X2 := s | t | u
Answer with the complete Lean line.
```

**Answer:**
```
simpa using inf_sup_left t s u
```

---

## [lean_candidate_compilation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/math_lean.py)

<!-- behavior-hash: a9cebc298eaac71f -->

Determine if a candidate proof body successfully closes a theorem in Lean.

**Prompt:**
```
Does this Lean 4 tactic body close the theorem?
The answer is True or False.

THEOREM:
theorem ex (s t u : Set Int) (h0 : u ⊆ s) : u ∩ t ⊆ s ∩ t := by
  ?

CANDIDATE:
exact Set.inter_subset_inter subset_rfl
```

**Answer:**
```
False
```

---

## [planar_geometry_relations](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/math_geometry.py)

<!-- behavior-hash: cb970570b347392f -->

Answer geometry queries about point intersections, angles, and distances.

**Prompt:**
```
Given points: B=(-5, 2); E=(81/17, -16/17); F=(-1, 1); I=(-1, 2); K=(80/9, -53/18); M=(11, -1); V=(5, 1); W=(5, 0); X=(68/9, -59/18).
Question: Are points X, K, and M collinear?
Answer is either Yes or No.
```

**Answer:**
```
No
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
1. P1(x, y)
2. P2(y, x)
3. P1(x, x)

Allowed Rules:
r1: P2(x, y); P1(y, z) ==> P1(x, z)
r2: P1(x, y); P1(y, z) ==> P1(x, z)

Conjecture:
P1(x, z)
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
- r1: P4(x, y); P4(y, x) ==> P5(x, y)
- r2: P1(x, D1); P1(y, D1) ==> P3(P2(F1(x), y), P2(F1(y), x))
- r3: P1(x, D2) ==> P1(x, D1)
- r4: P1(x, D3) ==> P3(P5(F2(x, C2), C0), P5(x, C0))

Conjecture:
P3(P2(F1(x), y), P2(F1(y), x))

Options:
A. [r1, r2]
B. [r2, r4]
C. [r2, r3]
D. [r3]
```

**Answer:**
```
C
```

---

## [lambda_reduction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/binding.py)

<!-- behavior-hash: 16af264fca09c34c -->

Reduce lambda calculus terms to normal form with renaming and shadowing.

**Prompt:**
```
Reduce the following untyped λ-term to β-normal form.
Syntax: `\x.body` is λx.body; juxtaposition is left-associative application; free identifiers are constants.

Term: (((\v0.(d (\v1.v1))) a) ((\v0.((\v1.(v1 v1)) ((\v0.v0) (b v0)))) (((d c) (\v0.d)) (((\v0.v0) b) c))))

The answer is the β-normal form (compared up to α-equivalence).
```

**Answer:**
```
((d (\x0.x0)) ((b (((d c) (\x1.d)) (b c))) (b (((d c) (\x2.d)) (b c)))))
```

---

## [rewrite_system](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/binding.py)

<!-- behavior-hash: 16af264fca09c34c -->

Normalize term rewrite systems under boolean, list, logic, or path rules.

**Prompt:**
```
Normalize by the ordered rewrite rules. At each step, scan subterm positions outermost-first and left-to-right. Stop at the first position matched by at least one rule, then apply the earliest matching rule in the listed order (position priority first; rule priority second).

Rules:
- snd(pair(X,Y)) -> Y
- fst(pair(X,Y)) -> X
- let(unit,X) -> X
- if(true,X,Y) -> X
- const(X,Y) -> X

Term:
if(if(if(false,c,unit),fst(pair(snd(pair(snd(id(unit)),let(unit,pair(false,true)))),c)),unit),pair(snd(b),b),pair(a,fst(a)))

The answer is the normal form.
```

**Answer:**
```
if(if(if(false,c,unit),pair(false,true),unit),pair(snd(b),b),pair(a,fst(a)))
```

---

## [unification_entailment](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/binding.py)

<!-- behavior-hash: 16af264fca09c34c -->

Decide if an equality is implied by the most general unifier of equations.

**Prompt:**
```
Compute a most general unifier of the equations. Apply it to both sides of the candidate equality. Answer Yes if the instantiated candidate terms are identical, otherwise answer No. The equations are guaranteed to be unifiable.

Equations:
- g(c) = g(x0)

Candidate:
c = x0
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
Factor b is independently true with probability 0.1.
Factor e is independently true with probability 0.2.
The observation holds exactly when (factor b or factor e).
We observe it.
Which hidden fact values form the most probable complete explanation?

Hidden fact values:
0. b
1. not b
2. e
3. not e

Choose one value for each hidden factor. Answer with space-separated indexes.
```

**Answer:**
```
1 2
```

---

## [most_probable_outcome](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/probabilistic_reasoning.py)

<!-- behavior-hash: ed992c8f6d367267 -->

Predict the most probable outcome or select hidden factor values in ProbLog.

**Prompt:**
```
A deck contains 5 blue cards and 5 red cards.
Two cards are drawn without replacing the first card.
Which statement is more likely?
A: both selected cards are blue.
B: the selected cards have different colors.

The answer is exactly one of: A, B, equal.
```

**Answer:**
```
B
```

---

## [multistep_nli](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic_depth.py)

<!-- behavior-hash: f7d1c90f99134e14 -->

Multi-hop natural language inference over chained logic facts and rules.

**Prompt:**
```
Premise:
Bruno is charlie tagged.

Hypothesis:
David is alpha-linked to Bruno.

Is the hypothesis true given the premise? The answer is Yes, No, or Maybe.
```

**Answer:**
```
Maybe
```

---

## [defeasible_nli](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic_depth.py)

<!-- behavior-hash: f7d1c90f99134e14 -->

NLI using defeasible logic rules and negation as failure.

**Prompt:**
```
An `unless` condition must be shown to block its rule.

Facts:
Bruno is trained, a bird, and trusted.
Clara is trained, blocked, a bird, a penguin, and careful.
Alice is not a penguin.
Bruno helps Clara.

Rules:
Trained people are trusted unless blocked.
Trusted people are approved unless flagged.
Blocked people are not trusted.
People who are not trusted are not approved unless flagged.
Penguins are abnormal.
Birds are approved unless abnormal.
Trained people are careful unless flagged.

Hypothesis:
Alice is not trusted.

Is the hypothesis true? Answer Yes, No, or Maybe.
```

**Answer:**
```
Maybe
```

---

## [multistep_evidence_retrieval](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic_depth.py)

<!-- behavior-hash: f7d1c90f99134e14 -->

Retrieve the specific premise indexes required to prove a logical hypothesis.

**Prompt:**
```
Premise:
[0] alice helps bruno.
[1] bruno trusts alice.
[2] From x helps y and y trusts z, it follows that x advises z.
[3] Whenever x advises z, x is careful.
[4] From x is trusted and x is approved, it follows that x is careful.

Hypothesis:
alice is not careful.

Which premise statements are necessary to contradict the hypothesis, meaning removing any one of them breaks that result?
Answer with space-separated indexes.
```

**Answer:**
```
0 1 2 3
```

---

## [multistep_abduction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic_depth.py)

<!-- behavior-hash: f7d1c90f99134e14 -->

Find the missing facts from candidates to satisfy a target hypothesis.

**Prompt:**
```
Premise:
[0] bruno trusts alice.
[1] david is trusted.
[2] If a person trusts a active person, then that person is careful.
[3] Whenever x is careful, x is trained.

Hypothesis:
bruno is trained.

Candidate Facts:
[0] david is careful.
[1] alice is careful.
[2] david is active.
[3] bruno is not careful.
[4] alice is active.
[5] alice is not active.

Which smallest set of candidate facts, if added to the premise, make the premise entail the hypothesis?
Do not include candidate facts that are not needed.
Answer with space-separated indexes.
```

**Answer:**
```
4
```

---

## [logic_qa](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic_depth.py)

<!-- behavior-hash: f7d1c90f99134e14 -->

Answer multi-step logical reasoning queries over rule-based theories.

**Prompt:**
```
Premise:
bruno is approved.
david is active.
alice is not careful.
Whenever x is approved, x is trained.
For all x, if x is trained, then x is active.
Every active entity is trusted.

Question:
Which entities can be shown to be trusted?

Answer with names in alphabetical order, comma-separated, or 'none'.
```

**Answer:**
```
bruno, david
```

---

## [logic_derivation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic_derivation.py)

<!-- behavior-hash: a99f22a5612d1966 -->

Produce a canonical forward proof trace for a logical target.

**Prompt:**
```
Premise:
0: clara advises david.
1: david trusts clara.
2: david is careful.
3: Advises relations followed by trusts relations imply helps relations.
4: From x helps z, it follows that x is active.

Target:
clara is active.

Give derivation lines as Rule: Input... => Deduction.
Use premise IDs and @0, @1, ... for derived lines.
```

**Answer:**
```
3: 0 1 => clara helps clara
4: @0 => clara is active
```

---

## [planning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/planning.py)

<!-- behavior-hash: 6d63a921f5474083 -->

Generate action plans to achieve goals in domains like Blocksworld.

**Prompt:**
```
Objects:
object_1, object_2

Actions:
action_1(x0)
  Effect: fluent_0(x0)

Initial state:
True values: None
All facts not listed under True values are false.

Goal:
fluent_0(object_2)

Action format example: action_0(object1, object2).
The answer is a shortest valid plan, one action per line.
```

**Answer:**
```
action_1(object_2)
```

---

## [set_missing_element](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

<!-- behavior-hash: 68a061870893e11f -->

Identify missing elements from a shuffled sequence defined by set intension.

**Prompt:**
```
Answer with the missing elements in the ordered span of {778, 777, 785, 782, 776, 783, 779} as a Python set.
```

**Answer:**
```
{780, 781, 784}
```

---

## [set_expression](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

<!-- behavior-hash: 68a061870893e11f -->

Evaluate complex set expressions involving union, intersection, and nested lists.

**Prompt:**
```
B = {3, 21, 16, 1, 13, 24, 10, 17}
C = {3, 21, 2, 1, 13, 24, 10, 17}
Evaluate len((B | (B | C))).
```

**Answer:**
```
9
```

---

## [sequential_induction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/sequential_induction.py)

<!-- behavior-hash: b4b67b4fcfc6a194 -->

Infer the canonical recurrence in a bounded polynomial DSL.

**Prompt:**
```
Infer U[n]. Max recurrence degree: 0. Ops: +, -, *.
Use n. Give the simplified polynomial RHS.
Sequence: [128, 130, 132, 134, 136, 138, 140, 142]
The answer is the RHS only.
```

**Answer:**
```
2 * n + 128
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
- E2 is immediately newer than E1.
- E4 is the 5th-newest.
- E3 is immediately newer than E0.
- E0 is newer than E2.

Which object is the 3rd-newest?
The answer is one object label.
```

**Answer:**
```
E2
```

---

## [grid_navigation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grid_navigation.py)

<!-- behavior-hash: 4bdc8a43b7bbbe26 -->

Infer object grid coordinates from spatial relations and step actions.

**Prompt:**
```
Grid [0,4]x[0,4], N=+y, E=+x.
Initial Facts:
- A is right of C.
- A is below C.
- C is in the same column as B.
- B is above C.
- B is above A.
- A is right of B.
- C starts at (0, 2).

Steps:
1. C and A swap positions.

What is the final coordinate of A? The answer is (x, y).
```

**Answer:**
```
(0, 2)
```

---

## [reference_tracking](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/tracking.py)

<!-- behavior-hash: 4f3376f005d950c5 -->

Track locations of balls in boxes across moves, swaps, and coreferences.

**Prompt:**
```
Inventory:
- b1: green
- b2: black
- b3: red
- b4: green

Initial State:
- b1 is in x2
- b2 is in x3
- b3 is in x1
- b4 is in x3

Moves:
- Move b1 from x2 to x1.
- Move it from x1 to x2.
- Transfer everything in x2 into x3.
- Transfer b2 from x3 into x2.
Where is b1 now? The answer is a box tag, like x1.
```

**Answer:**
```
x3
```

---

## [belief_tracking](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/belief_tracking.py)

<!-- behavior-hash: 2edebf66052d509d -->

Track ordered beliefs through observation and communication.

**Prompt:**
```
Initially, everyone knows that the button is in the crate.

Story: Grace moves the button to the crate. No one else sees the move. Grace moves the button to the crate. Grace and Eve watch together and can see one another. Grace moves the button to the case. Grace and Eve watch together and can see one another. Grace moves the button to the drawer. Grace and Eve watch together and can see one another. The button falls into the drawer. Nobody sees this happen. Grace sends Eve the message "The button is in the case", but it is not delivered.

Question: Where does Eve think the button is?

Answer with one container name.
```

**Answer:**
```
drawer
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

<!-- behavior-hash: fbe4336573ab3c89 -->

Solve query-aware assignment, graph, scheduling, grid, set, and numeric CSPs.

**Prompt:**
```
Each membership variable is 0 (not selected) or 1 (selected).

Constraints:
1. At most 0 of [m2 = 1; m4 = 1]
2. if m1 = 0, then if m0 = 1, then m1 = 1
3. Exactly 1 of [m0 = 1; m2 = 1; m4 = 1]

Question: Can m3 equal 1?
Answer Yes or No.
```

**Answer:**
```
Yes
```

---

## [graph_pathfinding](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

<!-- behavior-hash: 395bdd34b851e9bb -->

Find the shortest path or cost in weighted and unweighted directed graphs.

**Prompt:**
```
Find the shortest directed path from node 4 to node 3. If several paths are tied, return the lexicographically smallest one. Answer with space-separated nodes, or `None` if no path exists.

Graph:
0: 0->2 0->3 0->4 0->5; 1: 1->2; 2: 2->0 2->1 2->3 2->4; 3: 3->0 3->2 3->5; 4: 4->0 4->2; 5: 5->0 5->3
```

**Answer:**
```
4 0 3
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
Directed Edges: 0->4, 1->2, 2->1, 3->0, 4->5, 5->3

Queries:
[(0, 1)]
```

**Answer:**
```
4
```

---

## [regex_following](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

<!-- behavior-hash: 9f753ce41c94c3e6 -->

Produce a string that matches a specified regular expression pattern.

**Prompt:**
```
The answer is the shortest non-empty visible non-whitespace ASCII string that fully matches this regular expression, with lexicographic tie-breaks: \d{3,5}
```

**Answer:**
```
000
```

---

## [regex_reasoning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

<!-- behavior-hash: 9f753ce41c94c3e6 -->

Reason about regular expression equivalence, containment, and witnesses.

**Prompt:**
```
A = ac|c+
B = aab|b|ab
Do A and B accept exactly the same set of strings?
The answer is Yes or No.
```

**Answer:**
```
No
```

---

## [analogical_case_matching](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/formal_analogies.py)

<!-- behavior-hash: d4425f50bb96730a -->

Retrieve analogical cases matching query objects, links, and logical facts.

**Prompt:**
```
Which case matches Query under consistent entity/relation renaming and per-relation direction reversal? Answer with its ID.

M0: c alpha a, c alpha d, b beta a, d beta a
M1: a alpha b, c alpha d, a beta c, c beta d
M2: b alpha e, d beta b, a gamma e, e gamma b
Query: y delta v, v epsilon u, x epsilon v, u gamma v, z gamma v, z gamma x
```

**Answer:**
```
M1
```

---

## [parsing_derivation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

<!-- behavior-hash: f304ba7ec278a653 -->

Determine the derivation production rule sequence parsing a given string.

**Prompt:**
```
(START)
S

(GRAMMAR)
R0: A -> B
R1: S -> A
R2: B -> 'bank' S
R3: B -> S 'subject'
R4: B -> 'bank'
R5: S -> B 'dinner'

(STRING)
bank bank

(QUESTION)
The answer is the rule labels used in the leftmost derivation of STRING, in order, separated by spaces.
```

**Answer:**
```
R1 R0 R2 R1 R0 R4
```

---

## [syntax_error_detection](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

<!-- behavior-hash: f304ba7ec278a653 -->

Locate syntax errors or grammatical perturbations in generated sentences.

**Prompt:**
```
(START)
start

(GRAMMAR)
expr ::= '(' seq ')'
seq ::=
seq ::= expr seq
expr ::= '<' seq '>'
start ::= seq
expr ::= '[' seq ']'

(STRING)
[ < > ] ( ) <

Answer OK, INCOMPLETE, or ERROR token for the first invalid token. If that token repeats in STRING, append its 1-based occurrence as @occurrence.
```

**Answer:**
```
INCOMPLETE
```

---

## [constrained_continuation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

<!-- behavior-hash: f304ba7ec278a653 -->

Complete a uniquely determined fixed-length span using a formal grammar.

**Prompt:**
```
Complete <HOLE> according to the grammar.

GRAMMAR:
S ::= '[' S ']'
A ::= B
S ::= A
B ::= 'candidate'

SENTENCE:
[ <HOLE> ] ] ]

Return only the missing 5 tokens.
```

**Answer:**
```
[ [ [ candidate ]
```

---

## [table_qa](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/table_qa.py)

<!-- behavior-hash: 2d0436406b7e80db -->

Answer queries on tabular data by executing SQL queries over dataframes.

**Prompt:**
```
Execute this SQL query on the table named dataframe:

Table 1:
row_id,qty,country,date,unit_price,category,status
R0000,9,Spain,7 févr. 2026,9.75,Electronics,pending
R0001,1,Germany,26 janv. 2026,38.31,Office,paid
R0002,6,Netherlands,19 déc. 2025,22.03,Electronics,refunded
R0003,6,Germany,10 nov. 2025,17.71,Electronics,paid
R0004,9,Netherlands,9 oct. 2025,53.76,Office,paid
R0005,10,Italy,16 juil. 2025,7.40,Books,paid
R0006,1,Netherlands,1 sept. 2025,38.69,Food,paid
R0007,9,Italy,24 sept. 2025,27.92,Food,paid


In this table, — represents SQL NULL.

SQL: SELECT "category"
        FROM dataframe
        WHERE TRUE
        GROUP BY "category"
        ORDER BY MAX(("qty" + "unit_price")) ASC, "category" ASC
        LIMIT 1

The answer is the result as a single value.
```

**Answer:**
```
Books
```

---

## [table_equivalence](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/table_qa.py)

<!-- behavior-hash: 2d0436406b7e80db -->

Decide if two rendered tables are semantically equivalent under mutations.

**Prompt:**
```
Do these tables contain the same data?
Ignore row order, column order, and table syntax; match values by column name.
Treat numeric grouping and trailing zeros as formatting, ISO and English month-name dates as dates, and — and NULL as missing. Repeated rows count.

Table A:
event_date,amount,country,date,product,job
2024-12-21,390.87,Eritrea,2025-09-16,Add,—
2026-01-17,2019.04,Latvia,2026-07-04,Rise,Theatre stage manager
2025-08-03,123.15,Saint Lucia,2025-10-28,Phone,Contracting civil engineer
2024-07-15,1031.67,Egypt,2026-04-21,Rise,"Psychotherapist, dance movement"
2025-04-04,905.94,Zambia,2026-01-19,Similar,"Designer, exhibition/display"
2024-10-23,106.54,French Southern Territories,2025-07-24,We,"Therapist, drama"
2025-03-13,914.49,Bosnia and Herzegovina,2026-06-16,Full,Jewellery designer
2025-01-15,295.4,Libyan Arab Jamahiriya,2025-11-17,Hour,"Surveyor, hydrographic"


Table B:
event_date	job	amount	product	country	date
Jul 15, 2024	Psychotherapist, dance movement	1,031.67	Rise	Egypt	Apr 21, 2026
Aug 03, 2025	Contracting civil engineer	123.15	Phone	Saint Lucia	Oct 28, 2025
Jan 17, 2026	Theatre stage manager	2,019.04	Rise	Latvia	Jul 04, 2026
Mar 13, 2025	Jewellery designer	914.49	Full	Bosnia and Herzegovina	Jun 16, 2026
Dec 21, 2024	NULL	390.87	Add	Eritrea	Sep 16, 2025
Apr 04, 2025	Designer, exhibition/display	905.94	Similar	Zambia	Jan 19, 2026
Jan 15, 2025	Surveyor, hydrographic	295.40	Hour	Libyan Arab Jamahiriya	Nov 17, 2025
Oct 23, 2024	Therapist, drama	106.54	We	French Southern Territories	Jul 24, 2025


Answer Yes or No.
```

**Answer:**
```
Yes
```

---

## [table_statistics](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/table_qa.py)

<!-- behavior-hash: 2d0436406b7e80db -->

Select rows or columns using associations, conditioning, group robustness, and shifts.

**Prompt:**
```
Table:
label|E|W|G|C
L1|-1.02|-0.1|-0.71|-0.09
L1|-0.72|-0.08|-3.39|-2.29
L0|-1.96|0.73|2.05|0.59
L2|-0.71|0.29|-5.34|-1.28
L0|-1.59|-0.39|3.88|-0.5
L2|-1.66|0.12|-2.95|-0.24
L1|-1.86|-0.57|-2.63|0.29
L2|-1.66|0.33|-3.23|-0.92
L2|-1.54|0.92|-2.4|-2.17
L0|-0.69|0.21|4.12|1.03
L1|-1.7|-0.33|-1.31|-0.89
L0|0.01|0.87|3.95|-0.09


Find:
numeric column name most associated with column label

Metric:
eta squared

Answer with only the identifier.
```

**Answer:**
```
G
```

---

## [string_transduction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/string_transduction.py)

<!-- behavior-hash: 7e37b3a7153847d2 -->

Apply string transduction operations including Caesar cipher and rotation.

**Prompt:**
```
String: bbdcbcdc
Operations:
- caesar shift by 2
- keep only e and d
Answer with the final string.
```

**Answer:**
```
ddedee
```

---

## [game_best_move](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/game_playing.py)

<!-- behavior-hash: 4b4daac8084b238d -->

Determine the minimax-optimal move for a player in a finite graph-based game.

**Prompt:**
```
In this graph game, choose player's best move. Player chooses on player turns; opponent chooses on opponent turns. Opponent minimizes player score.

Start: n1. Turns alternate player, opponent. Move along one edge per turn, for at most 3 moves. Play ends upon reaching a leaf or the move horizon; in either case, player's score is the current node's payoff. Node payoffs: n0:80; n1:100; n2:100; n3:80; n4:100; n5:30; n6:30. Edges: n0->n5; n1->n4,n5; n2->n6; n3->n5.
Legal player moves now: n4, n5.
The answer is the destination node of the best move.
```

**Answer:**
```
n4
```

---

## [game_forced_win](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/game_playing.py)

<!-- behavior-hash: 4b4daac8084b238d -->

Decide if a player can force a win from a given state in a graph-based game.

**Prompt:**
```
In this graph game, decide whether player can force a win. Player chooses on player turns; opponent chooses on opponent turns. Opponent minimizes player score. A win means final player score is greater than 50.

Start: n0. Turns alternate player, opponent. Move along one edge per turn, for at most 3 moves. Play ends upon reaching a leaf or the move horizon; in either case, player's score is the current node's payoff. Node payoffs: n0:100; n1:60; n2:90; n3:0; n4:20; n5:0; n6:0. Edges: n0->n4,n6; n1->n5,n6; n2->n5; n3->n4.
The answer is Yes or No.
```

**Answer:**
```
No
```

---

## [qualitative_causal_reasoning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/qualitative_causal_reasoning.py)

<!-- behavior-hash: c5133ef287038003 -->

Reason qualitatively about causal effects and associations in graphs.

**Prompt:**
```
Assume linear causal relations, independent noise, and no exact cancellations.

- X1 directly increases X0.
- X2 directly increases X10.
- X2 directly increases X6.
- X4 directly increases X8.
- X7 directly increases X5.
- X9 directly increases X0.

Given X2, are X10 and X6 associated?
Answer with: associated or independent.
```

**Answer:**
```
independent
```

---

## [code_analysis](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/code_analysis.py)

<!-- behavior-hash: ad434bfeb17e2165 -->

Analyze toy finite-state Python-like programs with CTL temporal formulas.

**Prompt:**
````
Program:
```python
y, level = 0, 0

def step():
    global y, level
    y = (y + 1) % 2
    level = (level + y + 1) % 2

```

Start from the assignments above; each transition calls `step()`.

Formula: AF(p1)
Property: on every execution, eventually (level == 1)

Does the property hold from the initial state? Answer Yes or No.
````

**Answer:**
```
Yes
```

---

## [code_runnability](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/code_execution.py)

<!-- behavior-hash: 0213298e76d0d03d -->

Predict if a given Python code snippet runs successfully or raises an exception.

**Prompt:**
````
Predict whether this Python call runs successfully or raises an exception.
```python
def f0(s: int, f: int) -> int:
    print(s)
    print(f)
    return s // f
def f1(f: int, s: str) -> int:
    a = 2
    print(a)
    return 7
def endpoint(x0: int, x1: int) -> int:
    return f0(x0, x1)

```
Call: `endpoint(-2, -3)`
The answer is `OK` if it runs successfully; otherwise the exception class name.
````

**Answer:**
```
OK
```

---

## [code_execution](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/code_execution.py)

<!-- behavior-hash: 0213298e76d0d03d -->

Predict the return value or stdout of executing generated Python code blocks.

**Prompt:**
````
Predict the value returned by this Python call.
```python
def f0(t: str, n: int) -> int:
    print(n)
    return n + 1
def f1(c: str, w: int) -> str:
    w = len("go")
    return c
def endpoint(x0: str, x1: int) -> int:
    return f0(x0, x1)

```
Call: `endpoint('', -1)`
The answer is the exact Python `repr` of the returned value.
````

**Answer:**
```
0
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
- len: len(str)
- add: int + int
- not: not bool
Bounds: strings have length <= 64; integers are between -16 and 64. Use Python string semantics.
Cost: AST nodes, then operator-count tuple in this global order (concat, substr, replace1, ite, len, find, add, sub, contains, eq_str, lt, not), then source length, then lexicographic source order.

Examples:
f('0') = '0_-'
f(' ') = ' _-'

Return only:
def f(s: str) -> str:
    return <expression>
```

**Answer:**
```
def f(s: str) -> str:
    return ((s + "_") + "-")
```

---
