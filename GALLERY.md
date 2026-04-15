# 📖 Task Gallery

[`arithmetics`](#arithmetics) · [`symbolic_arithmetics`](#symbolic_arithmetics) · [`equation_system`](#equation_system) · [`conjecture_entailment`](#conjecture_entailment) · [`proof_reconstruction`](#proof_reconstruction) · [`bayesian_association`](#bayesian_association) · [`bayesian_intervention`](#bayesian_intervention) · [`logic_nli`](#logic_nli) · [`evidence_retrieval`](#evidence_retrieval) · [`planning`](#planning) · [`set_intersection`](#set_intersection) · [`set_missing_element`](#set_missing_element) · [`count_elements`](#count_elements) · [`set_equality`](#set_equality) · [`sequential_induction`](#sequential_induction) · [`qualitative_reasoning`](#qualitative_reasoning) · [`navigation`](#navigation) · [`reference_tracking`](#reference_tracking) · [`constraint_satisfaction`](#constraint_satisfaction) · [`graph_pathfinding`](#graph_pathfinding) · [`graph_node_centrality`](#graph_node_centrality) · [`graph_cycle_detection`](#graph_cycle_detection) · [`graph_isomorphism`](#graph_isomorphism) · [`graph_successors`](#graph_successors) · [`graph_dependencies`](#graph_dependencies) · [`regex_following`](#regex_following) · [`regex_induction`](#regex_induction) · [`regex_reasoning`](#regex_reasoning) · [`lexical_knowledge`](#lexical_knowledge) · [`parsability`](#parsability) · [`parsing`](#parsing) · [`continuation`](#continuation) · [`locate_error`](#locate_error) · [`constrained_continuation`](#constrained_continuation) · [`table_qa`](#table_qa) · [`table_conversion`](#table_conversion) · [`lambda_reduction`](#lambda_reduction) · [`term_unification`](#term_unification) · [`code_execution`](#code_execution) · [`diff_prediction`](#diff_prediction) · [`diff_patching`](#diff_patching)

---

## [arithmetics](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/arithmetics.py)

**Prompt:**
```
Evaluate 10 / 4.
The answer is a number.
```

**Answer:**
```
2.5
```

---

## [symbolic_arithmetics](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/arithmetics.py)

**Prompt:**
```
Simplify the following algebraic expression:
x + y - 9 + 6

The answer is the simplified expression.
```

**Answer:**
```
x + y - 3
```

---

## [equation_system](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/equation_system.py)

**Prompt:**
```
Solve the following system of equations for the variable 'X2'.

System:
  X2 + 14 = 0

The answer is the numerical value for X2, or 'No solution' / 'Multiple solutions' if a unique numerical solution does not exist.
```

**Answer:**
```
-14
```

---

## [conjecture_entailment](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/formal_maths.py)

**Prompt:**
``
Decide if the given premises entail the conjecture (i.e., the conjecture is provable) using Superposition/Resolution/Paramodulation.

Domain: Set Theory

Premises:
- (subset(X1,X2)|~subset(X3,X2)|~member(member_of_1_not_of_2(X1,X2),X3))

Conjecture: `(subset(X1,X2)|~subset(X3,X2)|~subset(X1,X3))`

The answer is `True` (provable) or `False` (not provable).
``

**Answer:**
```
False
```

---

## [proof_reconstruction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/formal_maths.py)

**Prompt:**
```
Reconstruct the proof dependency graph.
Domain: Analysis
Theorem: (less_or_equal(X1,X2)|X3!=minimum(X4,minimum(X5,X2))|X3!=X1)

Rules:
- Some clauses are axioms (no parents); do NOT list them
- All other clauses derive from exactly 2 parents
- Clauses can be reused as parents

Shuffled clauses:
1. (minimum(X1,X2)=X1|~less_or_equal(X1,X2))
2. (less_or_equal(X1,X2)|X1!=minimum(X3,X2))
3. (less_or_equal(X1,X2)|X3!=minimum(X4,minimum(X5,X2))|X3!=X1)
4. (less_or_equal(X1,X2)|X1!=X2)
5. (less_or_equal(X1,X3)|~less_or_equal(X1,minimum(X2,X3)))
6. (less_or_equal(X1,X2)|X1!=minimum(X3,minimum(X4,X2)))
7. (minimum(X1,X2)=X1|X1!=minimum(X3,X2))

The answer is the list of derivations for derived clauses, one per line: CHILD <- PARENT_1, PARENT_2
Example: 5 <- 2, 4

```

**Answer:**
```
2 <- 4, 5
3 <- 6, 7
6 <- 2, 5
7 <- 1, 2
```

---

## [bayesian_association](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/causal_reasoning.py)

**Prompt:**
```
System:
P(X_0) = {'0': 0.9, '1': 0.1} 
P(X_2|X_0=0) = {'0': 0.4, '1': 0.6} 
P(X_2|X_0=1) = {'0': 0.6, '1': 0.4} 
P(X_1) = {'0': 0.4, '1': 0.6}
Observed conditions:
Observing/Knowing that the state X_1 is equal to 1
Task: Compute probability distribution for X_0 (possible values: [0, 1]).

The answer is a Python dict mapping each value to its probability, rounded to 1 decimals.
Example: {0: 0.1, 1: 0.9}
```

**Answer:**
```
{0: 0.9, 1: 0.1}
```

---

## [bayesian_intervention](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/causal_reasoning.py)

**Prompt:**
```
System:
P(X_0) = {'0': 0.2, '1': 0.8} 
P(X_1) = {'0': 0.2, '1': 0.8} 
P(X_2) = {'0': 0.6, '1': 0.4}
Observed conditions:
Doing/Imposing that the state X_2 is equal to 0
Task: Compute probability distribution for X_1 (possible values: [0, 1]).

The answer is a Python dict mapping each value to its probability, rounded to 1 decimals.
Example: {0: 0.1, 1: 0.9}
```

**Answer:**
```
{0: 0.2, 1: 0.8}
```

---

## [logic_nli](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic.py)

**Prompt:**
```
Premise:
there is a room.
more than one person outside the room either is a quiet person or is not old, not quiet but not both
Fred has a pet dog
Mary is not old
all old people in the room are old
exactly five people in the room writes and illustrates their own graphic novels
Hypothesis:
Paul is quiet

If the Premise entails the Hypothesis, the label is 'entailment'.
If the Premise contradicts the Hypothesis, the label is 'contradiction'.
If neither, the label is 'neutral'.
The answer is exactly one word: neutral, contradiction, or entailment.
```

**Answer:**
```
neutral
```

---

## [evidence_retrieval](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic.py)

**Prompt:**
```
Premise:
[0] there is a room.
[1] everyone outside the room is old only if they are not quiet
[2] everyone in the room is not a quiet person if they are a quiet person
[3] Paul has a saltwater aquarium
[4] everyone in the room who is quiet is a professional photographer specializing in portrait photography
[5] Fred and Mary are quiet people
Hypothesis:
Mary is old

Which statements in the premise contradict the hypothesis?
The answer is the list of supporting statements, e.g. [0, 6, 7].
```

**Answer:**
```
[1, 2, 5]
```

---

## [planning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/planning.py)

**Prompt:**
```
[OBJECTS]
object_1, object_2

[ACTIONS]
action_0(x0)
  Requires: (not fluent_1(x0)), (not fluent_0)
  Effect: fluent_1(x0), fluent_0

[STATE]
Initial true values: None

[GOAL]

fluent_1(object_2)
Hint: Reference solution has 1 actions (but it may not be optimal).
The answer is the plan.
Answer format: Multiple lines, one action per line: action(obj1, obj2)
```

**Answer:**
```
    action_0(object_2)
```

---

## [set_intersection](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
```
Set1: {276, 935, 71, 124, 573, 155, 653, 681}
Set2: {573, 935, 681, 110, 660, 828}
The answer is the intersection of Set1 and Set2 as a Python set: {elem_1, elem_2, ..., elem_n}.
```

**Answer:**
```
{573, 681, 935}
```

---

## [set_missing_element](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
```
Set_A: {638, 629, 636, 637, 631, 634, 630}
The answer is the missing elements from Set_A as a Python set.
```

**Answer:**
```
{632, 633, 635}
```

---

## [count_elements](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
```
List: ['o', 'f', 'a', 'i', 'f', 'o', 'n', 'r', 'b', 'k']
How many times does 'o' appear? The answer is a number.
```

**Answer:**
```
2
```

---

## [set_equality](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
```
Set1: {49, 280, 114, 976, 296, 538, 840, 638}
Set2: {280, 638, 49, 114, 527, 840, 296, 538, 976}
The answer is True if Set1 and Set2 contain exactly the same elements, False otherwise.
```

**Answer:**
```
False
```

---

## [sequential_induction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/sequential_induction.py)

**Prompt:**
```
Infer a recurrence for a sequence indexed from 0: [U0, U1, ..., U7].
Max recurrence degree: 1.

Allowed binary ops: +, -, *, **
- Previous terms must be referenced exactly as: U[n - 1] ... U[n - 1]
- You may use "n" (current index).
- The answer is the right-hand side only (do not write "U[n] =").
- Your recurrence degree must be <= 1.

Sequence: [6, 7, 9, 12, 16, 21, 27, 34]
Degree of recurrence: 1
Initial terms: [6]

The answer must hold for all n >= d and be as simple as possible.
```

**Answer:**
```
n + U[n - 1]
```

---

## [qualitative_reasoning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/qstr.py)

**Prompt:**
```
Qualitative reasoning over vertical extents of 2D boxes.
There are 5 entities labeled 0 through 4.
You are given the following facts (read 'i rel j' as 'entity i is rel to entity j'):
  4 starts 2
  3 equals 4
  1 overlapped-by 4
  0 starts 1
  0 during 4

Question: what is the relation of the vertical extent of box 2 to that of box 0?
The answer is exactly one of: after, before, contains, during, equals, finished-by, finishes, meets, met-by, overlapped-by, overlaps, started-by, starts.
Respond with only the relation name as the answer.
```

**Answer:**
```
contains
```

---

## [navigation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/navigation.py)

**Prompt:**
```
Objects occupy distinct points on the integer grid [0, 4] x [0, 4].
North is +y and East is +x. Any object not mentioned in a step stays fixed.

Initial facts:
- A is right of B.
- B is in the same row as C.
- C is right of A.
- C is right of B.
- A is above B.
- A is above C.
- A starts at (1, 2).

Steps:
1. C moves by (2, 0).

What is the final Manhattan distance between B and C? Answer as an integer.

```

**Answer:**
```
4
```

---

## [reference_tracking](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/tracking.py)

**Prompt:**
```
Inventory:
- b1: black
- b2: red
- b3: white
- b4: red
Initial state:
- b1 is in x2
- b2 is in x3
- b3 is in x3
- b4 is in x1
Moves:
- Move b2 from x3 to x2.
- Transfer b2 from x2 into x1.
- Move it from x1 to x2.
- Relocate b4 from x1 to x2.
Question: Where is b1 now? Answer with a box tag like x1.
```

**Answer:**
```
x2
```

---

## [constraint_satisfaction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/constraint_satisfaction.py)

**Prompt:**
```
Variables/domains:
- 0 <= x0 <= 2
- 0 <= x1 <= 1

Constraints:
1. -3*x1 != -4
2. 2*x0 - x1 != -4
3. -2*x0 >= -4

Enumerate ALL satisfying assignments in variable order [x0, x1].
The answer is a Python list of lists of ints, sorted lexicographically, or UNSAT if no assignment exists.

```

**Answer:**
```
[[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
```

---

## [graph_pathfinding](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
```
Consider the graph:

graph { 0--3; 1--3; 1--4; 2--3; 2--4; 3--4 }

Find the lexicographically smallest shortest path from Node 4 to Node 1.
The answer is a Python list of nodes.
```

**Answer:**
```
[4, 1]
```

---

## [graph_node_centrality](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
``
Consider the following social network graph:

Edges: 0-1, 0-2, 0-3, 0-4, 1-4, 2-4

Based on the number of connections, identify all nodes that are the most central (i.e., have the highest degree centrality). There may be more than one.
The answer is a Python list of node integers, sorted in increasing order. Example: `[3, 8]`.
``

**Answer:**
```
[0]
```

---

## [graph_cycle_detection](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
``
Consider the graph below, which contains exactly one cycle.

Edges: 0-1, 1-2, 1-4, 2-3, 3-4

Identify all the nodes that form the cycle.
The answer is a Python list of nodes, sorted in increasing order. Example: `[2, 5, 7, 8]`.
``

**Answer:**
```
[1, 2, 3, 4]
```

---

## [graph_isomorphism](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
``
Consider two graphs described below.

Graph A:
Edges: 0-1, 1-2, 1-4, 2-3

Graph B:
graph { 0--2; 0--4; 1--0; 2--3 }

Do Graph A and Graph B have the exact same structure, just with different node labels? (In other words, are they isomorphic?)
The answer is `True` or `False`.
``

**Answer:**
```
True
```

---

## [graph_successors](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
```
Consider the directed graph:

Edges: 3->2, 5->1, 1->5, 2->4, 0->0, 4->3

Queries: [(1, 1)]
Each pair (x, k) asks for the k-th successor of x.
The answer is a Python list of integers in query order.
```

**Answer:**
```
[5]
```

---

## [graph_dependencies](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
```
Consider the dependency graph:

Edges (X->Y means Y depends on X): 1->2, 2->3, 2->4

List all prerequisites of node 4 (recursively), leaves first.
Do not include the query node itself.
If A depends on B and both appear in your answer, B must appear before A.
The answer is a Python list of integers.
```

**Answer:**
```
[1, 2]
```

---

## [regex_following](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

**Prompt:**
```
The answer is a 1-character string that fully matches the regular expression: E|(7)
```

**Answer:**
```
E
```

---

## [regex_induction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

**Prompt:**
``
The answer is the shortest regex that fully matches all POSITIVE strings and none of the NEGATIVE strings.
POSITIVE: '¬', '`', ',', '$', '¢', '»', ']', '['
NEGATIVE: 'win', '9', 'C', '<LPa', 'f', 'especially', '.last', '3583'
``

**Answer:**
```
((?:\W|[X-b]))
```

---

## [regex_reasoning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

**Prompt:**
```
Consider the regular expressions A = abb+ and B = c|c
Do A and B accept exactly the same set of strings?
The answer is Yes or No.
```

**Answer:**
```
No
```

---

## [lexical_knowledge](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/knowledge.py)

**Prompt:**
```
Context: WordNet (relation holds for any valid noun sense).

is_a(manual, training)
The answer is True or False.
```

**Answer:**
```
True
```

---

## [parsability](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
```
(GRAMMAR)
S -> D
D -> 'girl'
D -> D

(STRING)
girl

(QUESTION)
What is the parsability of this string?
The answer is exactly one word: unambiguous, ambiguous, or unparsable.
```

**Answer:**
```
ambiguous
```

---

## [parsing](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
```
(GRAMMAR)
decl -> decl_simple ',' conj decl_simple
are -> 'are'
root -> discourse '.'
start -> root
conj -> 'yet'
det_pl_indef -> 'some'
decl_simple -> there are det_pl_indef n_pl
discourse -> decl ',' conj decl
n_pl -> 'dogs'
there -> 'there'

(STRING)
there are some dogs , yet there are some dogs , yet there are some dogs , yet there are some dogs .

(QUESTION)
Identify the Part-of-Speech (immediate parent) and tree depth for each token.
format per token: token<POS:depth>
Example: the<Det:3> cat<Noun:3>
```

**Answer:**
```
there<there:6> are<are:6> some<det_pl_indef:6> dogs<n_pl:6> ,<decl:4> yet<conj:5> there<there:6> are<are:6> some<det_pl_indef:6> dogs<n_pl:6> ,<discourse:3> yet<conj:4> there<there:6> are<are:6> some<det_pl_indef:6> dogs<n_pl:6> ,<decl:4> yet<conj:5> there<there:6> are<are:6> some<det_pl_indef:6> dogs<n_pl:6> .<root:2>
```

---

## [continuation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
```
List all valid next tokens for this prefix. The answer is the list of valid tokens sorted alphabetically and separated by |, with STOP at the end if the prefix forms a complete string.
(GRAMMAR)
start -> seq
seq -> 
seq -> expr seq
expr -> '(' seq ')'
expr -> '[' seq ']'
expr -> '<' seq '>'
(PREFIX)
( ) ( )
```

**Answer:**
```
(|<|[|STOP
```

---

## [locate_error](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
```
(GRAMMAR)
S -> D
D -> 'control'
D -> 'avoid' D

(STRING)
avoid avoid avoid avoid

The answer is the shortest contiguous span from STRING that ends at the first invalid token and occurs only once in STRING.
Mark the invalid token as >>token<<.
If the token alone is enough, answer just >>token<<.
If STRING is fully grammatical, answer OK.
If all shown tokens are valid but more are needed, answer INCOMPLETE.
One line only.
```

**Answer:**
```
INCOMPLETE
```

---

## [constrained_continuation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
```
(GRAMMAR)
start -> seq
seq -> 
seq -> expr seq
expr -> '(' seq ')'
expr -> '[' seq ']'
expr -> '<' seq '>'

(PREFIX)
( )

(TEMPLATE)
___ ) ( ___

Fill in the 2 blanks (___) to form a grammatical continuation of PREFIX using exactly 4 tokens.
Fixed tokens must remain in place. The answer is all 4 tokens space-separated.
```

**Answer:**
```
( ) ( )
```

---

## [table_qa](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/table_qa.py)

**Prompt:**
```
Execute this SQL query on the table named dataframe:

Table 1:
 revenue   product
  359.11 Challenge
  313.47    Attack
  928.94     Treat
  630.61  Identify
  626.39 Available

SQL: SELECT COUNT(*) FROM dataframe WHERE revenue > 626.39

The answer is the result as single value.
```

**Answer:**
```
2
```

---

## [table_conversion](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/table_qa.py)

**Prompt:**
```
Convert the following table from to_yaml to to_html.

- date: 2025-07-28
  job: Office manager
- date: 2025-06-22
  job: Public relations officer
- date: 2026-02-21
  job: Seismic interpreter
- date: 2025-05-26
  job: Sports administrator
- date: 2025-06-13
  job: Energy manager


The answer is the converted table.
```

**Answer:**
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>date</th>
      <th>job</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2025-07-28</td>
      <td>Office manager</td>
    </tr>
    <tr>
      <td>2025-06-22</td>
      <td>Public relations officer</td>
    </tr>
    <tr>
      <td>2026-02-21</td>
      <td>Seismic interpreter</td>
    </tr>
    <tr>
      <td>2025-05-26</td>
      <td>Sports administrator</td>
    </tr>
    <tr>
      <td>2025-06-13</td>
      <td>Energy manager</td>
    </tr>
  </tbody>
</table>
```

---

## [lambda_reduction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/binding.py)

**Prompt:**
``
Reduce the following untyped λ-term to β-normal form.
Syntax: `\x.body` denotes λx.body; application is left-associative juxtaposition; free identifiers are treated as constants.

Term: ((\v0.v0) (d b))

The answer is the β-normal form (compared up to α-equivalence).
``

**Answer:**
```
(d b)
```

---

## [term_unification](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/binding.py)

**Prompt:**
```
Find the most general unifier (MGU) of the following first-order terms.
Uppercase identifiers are variables; lowercase are constants / function symbols.

T1 = p(f(g(c,X,q(b))))
T2 = p(f(g(c,c,Y)))

The answer is a Python dict mapping each bound variable (as a string key) to its fully-resolved ground term (as a string value), with keys sorted alphabetically.
Example: {'X': 'f(a)', 'Y': 'b'}
```

**Answer:**
```
{'X': 'c', 'Y': 'q(b)'}
```

---

## [code_execution](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/coding.py)

**Prompt:**
````
Predict the printed output of the following Python code:

```python
f = 2
b = 0 + 3
print("hi")
```

The answer is the exact printed output string.
````

**Answer:**
```
hi
```

---

## [diff_prediction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/coding.py)

**Prompt:**
```
Below is the version history of a file.

Version 909251b:
1    | General store crime
2    | Follow us rise message
3    | Building day himself expect role
4    | Foreign wear unit mind
5    | High improve control couple ability

Version cacc5d5:
1    | General store crime
2    | Foreign wear unit mind
3    | High improve control couple ability

Version 25972d6:
1    | General store crime
2    | Foreign wear unit mind
3    | High improve control couple ability

Generate the Unified Diff to transform version 909251b into version 25972d6.
The answer is the diff chunks only (no file headers), or empty if no changes.
```

**Answer:**
```
@@ -1,5 +1,3 @@
 General store crime
-Follow us rise message
-Building day himself expect role
 Foreign wear unit mind
 High improve control couple ability
```

---

## [diff_patching](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/coding.py)

**Prompt:**
```
Apply the following Unified Diff to the text.

Original Text (Version 171eec7):
1    | Meet go contain make force
2    | Information ask painting around
3    | End store whom
4    | Final what despite we record soon beat
5    | Morning adult care how player

Diff (171eec7 -> 42dc00e):


The answer is the resulting text.
```

**Answer:**
```
Meet go contain make force
Information ask painting around
End store whom
Final what despite we record soon beat
Morning adult care how player
```

---

