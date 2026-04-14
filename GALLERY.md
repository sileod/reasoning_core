# 📖 Task Gallery

[`arithmetics`](#arithmetics) · [`symbolic_arithmetics`](#symbolic_arithmetics) · [`equation_system`](#equation_system) · [`conjecture_entailment`](#conjecture_entailment) · [`proof_reconstruction`](#proof_reconstruction) · [`bayesian_association`](#bayesian_association) · [`bayesian_intervention`](#bayesian_intervention) · [`logic_nli`](#logic_nli) · [`evidence_retrieval`](#evidence_retrieval) · [`planning`](#planning) · [`set_intersection`](#set_intersection) · [`set_missing_element`](#set_missing_element) · [`count_elements`](#count_elements) · [`set_equality`](#set_equality) · [`sequential_induction`](#sequential_induction) · [`qualitative_reasoning`](#qualitative_reasoning) · [`navigation`](#navigation) · [`reference_tracking`](#reference_tracking) · [`constraint_satisfaction`](#constraint_satisfaction) · [`graph_pathfinding`](#graph_pathfinding) · [`graph_node_centrality`](#graph_node_centrality) · [`graph_cycle_detection`](#graph_cycle_detection) · [`graph_isomorphism`](#graph_isomorphism) · [`graph_successors`](#graph_successors) · [`graph_dependencies`](#graph_dependencies) · [`regex_following`](#regex_following) · [`regex_induction`](#regex_induction) · [`regex_reasoning`](#regex_reasoning) · [`lexical_knowledge`](#lexical_knowledge) · [`parsability`](#parsability) · [`parsing`](#parsing) · [`continuation`](#continuation) · [`locate_error`](#locate_error) · [`constrained_continuation`](#constrained_continuation) · [`table_qa`](#table_qa) · [`table_conversion`](#table_conversion) · [`lambda_reduction`](#lambda_reduction) · [`term_unification`](#term_unification) · [`code_execution`](#code_execution) · [`diff_prediction`](#diff_prediction) · [`diff_patching`](#diff_patching)

---

## [arithmetics](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/arithmetics.py)

**Prompt:**
```
Evaluate ((-5)).
The answer is a number.
```

**Answer:**
```
-5
```

---

## [symbolic_arithmetics](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/arithmetics.py)

**Prompt:**
```
Simplify the following algebraic expression:
x - 4 * y + min(2, y)

The answer is the simplified expression.
```

**Answer:**
```
x - 4*y + min(2, y)
```

---

## [equation_system](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/equation_system.py)

**Prompt:**
```
Solve the following system of equations for the variable 'X2'.

System:
  X1 - 17 = 0
  X1 - 17 = 0

The answer is the numerical value for X2, or 'No solution' / 'Multiple solutions' if a unique numerical solution does not exist.
```

**Answer:**
```
Multiple solutions
```

---

## [conjecture_entailment](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/formal_maths.py)

**Prompt:**
``
Decide if the given premises entail the conjecture (i.e., the conjecture is provable) using Superposition/Resolution/Paramodulation.

Domain: Group Theory

Premises:
- (product(identity,X1,identity)|~product(X2,inverse(X2),inverse(X1)))
- (inverse(inverse(X1))=X1)

Conjecture: `(multiply(multiply(X1,X2),inverse(X2))=X1)`

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
Domain: Set Theory
Theorem: (~subset(X1,empty_set)|~member(X2,apply(X1,X3)))

Rules:
- Some clauses are axioms (no parents); do NOT list them
- All other clauses derive from exactly 2 parents
- Clauses can be reused as parents

Shuffled clauses:
1. (member(X3,X2)|~subset(X1,X2)|~member(X3,X1))
2. (~subset(X1,empty_set)|~member(X2,X1))
3. (~member(X1,empty_set))
4. (~subset(X1,empty_set)|~member(X2,apply(X1,X3)))
5. (member(f28(X1,X2,X3),X2)|~member(X1,apply(X2,X3)))

The answer is the list of derivations for derived clauses, one per line: CHILD <- PARENT_1, PARENT_2
Example: 5 <- 2, 4

```

**Answer:**
```
2 <- 1, 3
4 <- 2, 5
```

---

## [bayesian_association](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/causal_reasoning.py)

**Prompt:**
```
System:
P(X_0) = {'0': 0.8, '1': 0.2} 
P(X_1) = {'0': 0.5, '1': 0.5} 
P(X_2) = {'0': 0.9, '1': 0.1}
Observed conditions:
Observing/Knowing that the state X_0 is equal to 0
Task: Compute probability distribution for X_2 (possible values: [0, 1]).

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
P(X_0) = {'0': 0.9, '1': 0.1} 
P(X_1) = {'0': 0.3, '1': 0.7} 
P(X_2) = {'0': 1.0, '1': 0.0}
Observed conditions:
Doing/Imposing that the state X_1 is equal to 1
Task: Compute probability distribution for X_0 (possible values: [0, 1]).

The answer is a Python dict mapping each value to its probability, rounded to 1 decimals.
Example: {0: 0.1, 1: 0.9}
```

**Answer:**
```
{0: 0.9, 1: 0.1}
```

---

## [logic_nli](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic.py)

**Prompt:**
```
Premise:
there is a room.
everyone in the room plays as a goalkeeper for a local amateur soccer team
no old person in the room is old
Gravity inverts in Oakhaven on Tuesdays.
all old people in the room are quiet
everyone in the room who is not quiet, not old is not old
Hypothesis:
Mary is not quiet

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
[0] Mary is the only person in the room.
[1] Fred is quiet
[2] Mary enjoys naked-eye stargazing
[3] everyone in the room is old
[4] Fred collects rare and antique scientific instruments
[5] Paul and Mary are respectively quiet and old
Hypothesis:
Fred is not quiet

Which statements in the premise contradict the hypothesis?
The answer is the list of supporting statements, e.g. [0, 6, 7].
```

**Answer:**
```
[1]
```

---

## [planning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/planning.py)

**Prompt:**
```
[OBJECTS]
object_1, object_2

[ACTIONS]
action_0(x0)
  Requires: (not fluent_4), (not fluent_3), (not fluent_2)
  Effect: fluent_4, fluent_3, fluent_2
action_1(x0, x1)
  Effect: fluent_1(x1), fluent_2

[STATE]
Default: False
Initial true values: None

[GOAL]

fluent_3
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
Set1: {279, 918, 219, 926, 606, 444, 870, 49}
Set2: {918, 444, 890, 463, 49, 852}
The answer is the intersection of Set1 and Set2 as a Python set: {elem_1, elem_2, ..., elem_n}.
```

**Answer:**
```
{49, 444, 918}
```

---

## [set_missing_element](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
```
Set_A: {5, 8, 2, 9, 6, 11, 4, 10, 3}
The answer is the missing elements from Set_A as a Python set.
```

**Answer:**
```
{7}
```

---

## [count_elements](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
```
List: [15, 6, 15, 6, 8, 2, 5, 4, 5, 2]
How many times does 5 appear? The answer is a number.
```

**Answer:**
```
2
```

---

## [set_equality](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
```
Set1: {'wf', 'eo', 'jh', 'po', 'fp', 'fj', 'ug', 'adv'}
Set2: {'po', 'fp', 'ug', 'eo', 'fj', 'nn', 'wf', 'adv', 'jh'}
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
Max recurrence degree: 0.

Allowed binary ops: +, -, *, **
- Previous terms must be referenced exactly as: U[n - 1] ... U[n - 0]
- You may use "n" (current index).
- The answer is the right-hand side only (do not write "U[n] =").
- Your recurrence degree must be <= 0.

Sequence: [5, 7, 9, 11, 13, 15, 17, 19]
Degree of recurrence: 0
Initial terms: []

The answer must hold for all n >= d and be as simple as possible.
```

**Answer:**
```
2*n + 5
```

---

## [qualitative_reasoning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/qstr.py)

**Prompt:**
```
Qualitative reasoning over horizontal extents of 2D boxes.
There are 5 entities labeled 0 through 4.
You are given the following facts (read 'i rel j' as 'entity i is rel to entity j'):
  0 starts 1
  3 finishes 1
  2 overlapped-by 0
  4 before 3
  2 during 3

Question: what is the relation of the horizontal extent of box 4 to that of box 2?
The answer is exactly one of: after, before, contains, during, equals, finished-by, finishes, meets, met-by, overlapped-by, overlaps, started-by, starts.
Respond with only the relation name as the answer.
```

**Answer:**
```
before
```

---

## [navigation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/navigation.py)

**Prompt:**
```
Objects occupy distinct points on the integer grid [0, 4] x [0, 4].
North is +y and East is +x. Any object not mentioned in a step stays fixed.

Initial facts:
- C starts at (3, 0).
- A is above C.
- A is in the same column as B.
- C is below B.
- A is left of C.
- A is below B.
- C is right of B.

Steps:
1. C and A swap positions.

What is the final coordinate of A? Answer as (x, y).

```

**Answer:**
```
(3, 0)
```

---

## [reference_tracking](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/tracking.py)

**Prompt:**
```
Inventory:
- b1: blue
- b2: black
- b3: yellow
- b4: red
Initial state:
- b1 is in x2
- b2 is in x1
- b3 is in x2
- b4 is in x2
Moves:
- Move b4 from x2 to x1.
- Relocate b4 from x1 to x3.
- Move it from x3 to x2.
- Relocate b3 from x2 to x1.
Question: Where is b2 now? Answer with a box tag like x1.
```

**Answer:**
```
x1
```

---

## [constraint_satisfaction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/constraint_satisfaction.py)

**Prompt:**
```
Variables/domains:
- 0 <= x0 <= 2
- 0 <= x1 <= 1

Constraints:
1. x1 == 0
2. x0 - x1 != -3
3. -x0 + 3*x1 != 3

Enumerate ALL satisfying assignments in variable order [x0, x1].
The answer is a Python list of lists of ints, sorted lexicographically, or UNSAT if no assignment exists.

```

**Answer:**
```
[[0, 0], [1, 0], [2, 0]]
```

---

## [graph_pathfinding](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
```
Consider the graph:

graph { 0--3; 0--4; 1--4; 2--3; 3--4 }

Find the lexicographically smallest shortest path from Node 2 to Node 3.
The answer is a Python list of nodes.
```

**Answer:**
```
[2, 3]
```

---

## [graph_node_centrality](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
``
Consider the following social network graph:

Edges: 0-1, 0-2, 0-3, 0-4, 1-2, 1-3, 1-4, 2-3, 2-4, 3-4

Based on the number of connections, identify all nodes that are the most central (i.e., have the highest degree centrality). There may be more than one.
The answer is a Python list of node integers, sorted in increasing order. Example: `[3, 8]`.
``

**Answer:**
```
[0, 1, 2, 3, 4]
```

---

## [graph_cycle_detection](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
``
Consider the graph below, which contains exactly one cycle.

graph { 0--1; 0--2; 1--2; 2--3; 3--4 }

Identify all the nodes that form the cycle.
The answer is a Python list of nodes, sorted in increasing order. Example: `[2, 5, 7, 8]`.
``

**Answer:**
```
[0, 1, 2]
```

---

## [graph_isomorphism](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
``
Consider two graphs described below.

Graph A:
Edges: 0-1, 1-2, 1-3, 1-4

Graph B:
Nodes [0, 1, 2, 3, 4] and edges: (2, 3), (3, 0), (3, 1), (3, 4).

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

Edges: 5->4, 3->3, 1->1, 0->0, 2->2, 4->5

Queries: [(5, 1)]
Each pair (x, k) asks for the k-th successor of x.
The answer is a Python list of integers in query order.
```

**Answer:**
```
[4]
```

---

## [graph_dependencies](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
```
Consider the dependency graph:

Edges (X->Y means Y depends on X): 4->2, 4->3, 4->5, 5->3

List all prerequisites of node 3 (recursively), leaves first.
Do not include the query node itself.
If A depends on B and both appear in your answer, B must appear before A.
The answer is a Python list of integers.
```

**Answer:**
```
[4, 5]
```

---

## [regex_following](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

**Prompt:**
```
The answer is a 7-character string that fully matches the regular expression: (1\.heart)
```

**Answer:**
```
1.heart
```

---

## [regex_induction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

**Prompt:**
```
The answer is the shortest regex that fully matches all POSITIVE strings and none of the NEGATIVE strings.
POSITIVE: '+Z', '+z', '+A', '+A', '+z', '+A', '+z', '+A'
NEGATIVE: 'g', 'W', 'BK', 'HHK', 'A', 'm', 'X', '>QC'
```

**Answer:**
```
(\+([zZA]))
```

---

## [regex_reasoning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

**Prompt:**
```
Consider the regular expressions A = babbc and B = cb*
Is every string accepted by A also accepted by B?
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

is_a(cent, coin)
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
S -> C
C -> 'consider'
C -> C

(STRING)
consider

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
start -> seq
seq -> 
seq -> expr seq
expr -> '(' seq ')'
expr -> '[' seq ']'
expr -> '<' seq '>'

(STRING)
( ( ) ) < >

(QUESTION)
Identify the Part-of-Speech (immediate parent) and tree depth for each token.
format per token: token<POS:depth>
Example: the<Det:3> cat<Noun:3>
```

**Answer:**
```
(<expr:3> (<expr:5> )<expr:5> )<expr:3> <<expr:4> ><expr:4>
```

---

## [continuation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
```
List all valid next tokens for this prefix. The answer is the list of valid tokens sorted alphabetically and separated by |, with STOP at the end if the prefix forms a complete string.
(GRAMMAR)
S -> B
B -> 'sense'
B -> B B
(PREFIX)
sense sense
```

**Answer:**
```
sense|STOP
```

---

## [locate_error](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
```
(GRAMMAR)
S -> B
B -> 'manage'
B -> 'same' B

(STRING)
same same same manage manage

The answer is the shortest contiguous span from STRING that ends at the first invalid token and occurs only once in STRING.
Mark the invalid token as >>token<<.
If the token alone is enough, answer just >>token<<.
If STRING is fully grammatical, answer OK.
If all shown tokens are valid but more are needed, answer INCOMPLETE.
One line only.
```

**Answer:**
```
manage >>manage<<
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
[ ___ ( ___

Fill in the 2 blanks (___) to form a grammatical continuation of PREFIX using exactly 4 tokens.
Fixed tokens must remain in place. The answer is all 4 tokens space-separated.
```

**Answer:**
```
[ ] ( )
```

---

## [table_qa](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/table_qa.py)

**Prompt:**
```
Execute this SQL query on the table named dataframe:

Table 1:
revenue,rating
601.38,2.9
885.34,4.1
511.7,3.4
571.91,4.1
19.07,4.8


SQL: SELECT COUNT(*) FROM dataframe WHERE revenue > 571.91

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
Convert the following table from to_latex to to_json.

\begin{tabular}{ll}
\toprule
date & price \\
\midrule
2025-06-15 & 136.6 \\
2025-11-11 & 415.08 \\
2025-05-22 & 493.45 \\
2026-02-12 & 386.93 \\
2026-02-08 & 353.97 \\
\bottomrule
\end{tabular}


The answer is the converted table.
```

**Answer:**
```
[
    {
        "date":"2025-06-15T00:00:00.000",
        "price":"136.6"
    },
    {
        "date":"2025-11-11T00:00:00.000",
        "price":"415.08"
    },
    {
        "date":"2025-05-22T00:00:00.000",
        "price":"493.45"
    },
    {
        "date":"2026-02-12T00:00:00.000",
        "price":"386.93"
    },
    {
        "date":"2026-02-08T00:00:00.000",
        "price":"353.97"
    }
]
```

---

## [lambda_reduction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/binding.py)

**Prompt:**
``
Reduce the following untyped λ-term to β-normal form.
Syntax: `\x.body` denotes λx.body; application is left-associative juxtaposition; free identifiers are treated as constants.

Term: ((\v0.v0) (\v0.v0))

The answer is the β-normal form (compared up to α-equivalence).
``

**Answer:**
```
(\v0.v0)
```

---

## [term_unification](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/binding.py)

**Prompt:**
```
Find the most general unifier (MGU) of the following first-order terms.
Uppercase identifiers are variables; lowercase are constants / function symbols.

T1 = g(h(d,e,g(a)),p(p(X,h(p(b,e,e)))))
T2 = g(h(d,e,g(a)),p(p(g(h(c),p(a,e)),Y)))

The answer is a Python dict mapping each bound variable (as a string key) to its fully-resolved ground term (as a string value), with keys sorted alphabetically.
Example: {'X': 'f(a)', 'Y': 'b'}
```

**Answer:**
```
{'X': 'g(h(c),p(a,e))', 'Y': 'h(p(b,e,e))'}
```

---

## [code_execution](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/coding.py)

**Prompt:**
````
Predict the printed output of the following Python code:

```python
q = 2
i = "sun"
print([4, 4, 8])
```

The answer is the exact printed output string.
````

**Answer:**
```
[4, 4, 8]
```

---

## [diff_prediction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/coding.py)

**Prompt:**
```
Below is the version history of a file.

Version ad411e1:
1    | Add service accept per success most today
2    | Manage technology edge ahead reflect nation turn
3    | Rise true hand perhaps heavy example
4    | Any like blood strong tend
5    | Remain especially board political

Version f6d20f3:
1    | Add service accept per success most today
2    | Manage technology edge ahead reflect nation turn
3    | Rise true hand perhaps heavy example
4    | Any like blood strong tend

Generate the Unified Diff to transform version f6d20f3 into version ad411e1.
The answer is the diff chunks only (no file headers), or empty if no changes.
```

**Answer:**
```
@@ -2,3 +2,4 @@
 Manage technology edge ahead reflect nation turn
 Rise true hand perhaps heavy example
 Any like blood strong tend
+Remain especially board political
```

---

## [diff_patching](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/coding.py)

**Prompt:**
```
Apply the following Unified Diff to the text.

Original Text (Version 4a730d8):
1    | Hospital rock do law own
2    | War defense sister movie to according enough
3    | Husband because wrong seem your national memory
4    | For across tend
5    | Camera government of she region although

Diff (4a730d8 -> 071f8f6):


The answer is the resulting text.
```

**Answer:**
```
Hospital rock do law own
War defense sister movie to according enough
Husband because wrong seem your national memory
For across tend
Camera government of she region although
```

---

