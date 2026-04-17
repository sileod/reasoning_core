# 📖 Task Gallery

[`arithmetics`](#arithmetics) · [`symbolic_arithmetics`](#symbolic_arithmetics) · [`equation_system`](#equation_system) · [`conjecture_entailment`](#conjecture_entailment) · [`proof_reconstruction`](#proof_reconstruction) · [`bayesian_association`](#bayesian_association) · [`bayesian_intervention`](#bayesian_intervention) · [`logic_nli`](#logic_nli) · [`evidence_retrieval`](#evidence_retrieval) · [`planning`](#planning) · [`set_intersection`](#set_intersection) · [`set_missing_element`](#set_missing_element) · [`count_elements`](#count_elements) · [`set_equality`](#set_equality) · [`sequential_induction`](#sequential_induction) · [`qualitative_reasoning`](#qualitative_reasoning) · [`navigation`](#navigation) · [`reference_tracking`](#reference_tracking) · [`constraint_satisfaction`](#constraint_satisfaction) · [`graph_pathfinding`](#graph_pathfinding) · [`graph_node_centrality`](#graph_node_centrality) · [`graph_isomorphism`](#graph_isomorphism) · [`graph_successors`](#graph_successors) · [`graph_dependencies`](#graph_dependencies) · [`regex_following`](#regex_following) · [`regex_induction`](#regex_induction) · [`regex_reasoning`](#regex_reasoning) · [`lexical_knowledge`](#lexical_knowledge) · [`parsability`](#parsability) · [`parsing`](#parsing) · [`continuation`](#continuation) · [`locate_error`](#locate_error) · [`constrained_continuation`](#constrained_continuation) · [`table_qa`](#table_qa) · [`table_conversion`](#table_conversion) · [`lambda_reduction`](#lambda_reduction) · [`term_unification`](#term_unification) · [`code_execution`](#code_execution) · [`diff_prediction`](#diff_prediction) · [`diff_patching`](#diff_patching)

---

## [arithmetics](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/arithmetics.py)

**Prompt:**
```
Evaluate 9 + -11.2.
The answer is a number.
```

**Answer:**
```
-2.2
```

---

## [symbolic_arithmetics](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/arithmetics.py)

**Prompt:**
```
Simplify the following algebraic expression:
min(x + (4)**2, 8)

The answer is the simplified expression.
```

**Answer:**
```
Min(8, x + 16)
```

---

## [equation_system](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/equation_system.py)

**Prompt:**
```
Solve the following system of equations for the variable 'X2'.

System:
  X1 - 9 = 0
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

Domain: Geometry

Premises:
- (on(point_on_plane(X1),X1)|~plane(X1))

Conjecture: `(X1=X2|on(X1,X1)|~plane(X2)|~plane(X1)|~line(X1)|~point(point_on_plane(X2))|~on(point_on_plane(X2),X1))`

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
Theorem: (minimum(n0,X1)=X1|half(X1)!=minimum(X2,n0))

Rules:
- Some clauses are axioms (no parents); do NOT list them
- All other clauses derive from exactly 2 parents
- Clauses can be reused as parents

Shuffled clauses:
1. (less_or_equal(X1,X3)|~less_or_equal(X1,minimum(X2,X3)))
2. (less_or_equal(X1,X2)|X1!=minimum(X3,X2))
3. (minimum(n0,X1)=X1|half(X1)!=minimum(X2,n0))
4. (less_or_equal(X1,X2)|X1!=X2)
5. (less_or_equal(X1,n0)|~less_or_equal(half(X1),n0))
6. (minimum(X2,X1)=X1|~less_or_equal(X1,X2))
7. (less_or_equal(X1,n0)|half(X1)!=minimum(X2,n0))

The answer is the list of derivations for derived clauses, one per line: CHILD <- PARENT_1, PARENT_2
Example: 5 <- 2, 4

```

**Answer:**
```
2 <- 1, 4
3 <- 6, 7
7 <- 2, 5
```

---

## [bayesian_association](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/causal_reasoning.py)

**Prompt:**
```
System:
P(X_0) = {'0': 0.8, '1': 0.2} 
P(X_1) = {'0': 0.3, '1': 0.7} 
P(X_2) = {'0': 0.3, '1': 0.7}
Observed conditions:
Observing/Knowing that the state X_0 is equal to 1
Task: Compute probability distribution for X_2 (possible values: [0, 1]).

The answer is a Python dict mapping each value to its probability, rounded to 1 decimals.
Example: {0: 0.1, 1: 0.9}
```

**Answer:**
```
{0: 0.3, 1: 0.7}
```

---

## [bayesian_intervention](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/causal_reasoning.py)

**Prompt:**
```
System:
P(X_0) = {'0': 0.6, '1': 0.4} 
P(X_1) = {'0': 0.7, '1': 0.3} 
P(X_2) = {'0': 0.7, '1': 0.3}
Observed conditions:
Doing/Imposing that the state X_1 is equal to 0
Task: Compute probability distribution for X_0 (possible values: [0, 1]).

The answer is a Python dict mapping each value to its probability, rounded to 1 decimals.
Example: {0: 0.1, 1: 0.9}
```

**Answer:**
```
{0: 0.6, 1: 0.4}
```

---

## [logic_nli](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic.py)

**Prompt:**
```
Premise:
Mary is the only person in the room.
Paul and Mary are old
Mary has a specialized collection of handmade artisan pottery
no old person in the room is quiet
someone who has a specialized collection of handmade artisan pottery hates someone who is old
all quiet people in the room are quiet
Hypothesis:
Paul is not old

If the Premise entails the Hypothesis, the label is 'entailment'.
If the Premise contradicts the Hypothesis, the label is 'contradiction'.
If neither, the label is 'neutral'.
The answer is exactly one word: neutral, contradiction, or entailment.
```

**Answer:**
```
contradiction
```

---

## [evidence_retrieval](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic.py)

**Prompt:**
```
Premise:
[0] Mary is the only person in the room.
[1] Fred practices pilates
[2] all quiet people in the room are quiet
[3] everyone in the room is not an old person if they are quiet
[4] it is not the case that “Paul and Mary are an old quiet people”
[5] Paul and Fred are respectively old and quiet
Hypothesis:
Fred is not quiet

Which statements in the premise contradict the hypothesis?
The answer is the list of supporting statements, e.g. [0, 6, 7].
```

**Answer:**
```
[5]
```

---

## [planning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/planning.py)

**Prompt:**
```
[OBJECTS]
object_1

[ACTIONS]
action_0(x0, x1)
  Requires: fluent_0(x1)
  Effect: not fluent_0(x1)

[STATE]
Default: True
Initial true values: fluent_0(object_1)

[GOAL]

(not fluent_0(object_1))
Hint: Reference solution has 1 actions (but it may not be optimal).
The answer is the plan.
Answer format: Multiple lines, one action per line: action(obj1, obj2)
```

**Answer:**
```
    action_0(object_1, object_1)
```

---

## [set_intersection](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
```
Set1: {58, 289, 197, 421, 218, 734, 75, 496}
Set2: {649, 869, 218, 958, 421, 289}
The answer is the intersection of Set1 and Set2 as a Python set: {elem_1, elem_2, ..., elem_n}.
```

**Answer:**
```
{218, 289, 421}
```

---

## [set_missing_element](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
```
Set_A: {571, 569, 576, 573, 568, 572, 577}
The answer is the missing elements from Set_A as a Python set.
```

**Answer:**
```
{570, 574, 575}
```

---

## [count_elements](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
```
List: [17, 6, 12, 10, 16, 13, 19, 8, 20, 9]
How many times does 9 appear? The answer is a number.
```

**Answer:**
```
1
```

---

## [set_equality](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
```
Set1: {978, 961, 36, 251, 67, 118, 214, 75}
Set2: {978, 961, 67, 214, 251, 36, 75, 118}
The answer is True if Set1 and Set2 contain exactly the same elements, False otherwise.
```

**Answer:**
```
True
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

Sequence: [-5, -3, 1, 7, 15, 25, 37, 51]
Degree of recurrence: 1
Initial terms: [-5]

The answer must hold for all n >= d and be as simple as possible.
```

**Answer:**
```
2*n + U[n - 1]
```

---

## [qualitative_reasoning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/qstr.py)

**Prompt:**
```
Qualitative reasoning over horizontal extents of 2D boxes.
There are 5 entities labeled 0 through 4.
You are given the following facts (read 'i rel j' as 'entity i is rel to entity j'):
  2 after 4
  0 overlaps 2
  1 after 0
  3 during 2

Question: what is the relation of the horizontal extent of box 4 to that of box 1?
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
- C is below A.
- B is above A.
- C is left of B.
- A is right of C.
- B is above C.
- A is right of B.

Steps:
1. C jumps to B's position offset by (-1, 0).

What is the final Manhattan distance between B and C? Answer as an integer.

```

**Answer:**
```
1
```

---

## [reference_tracking](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/tracking.py)

**Prompt:**
```
Inventory:
- b1: black
- b2: white
- b3: blue
- b4: white
Initial state:
- b1 is in x2
- b2 is in x3
- b3 is in x3
- b4 is in x3
Moves:
- Transfer b2 from x3 into x2.
- Move b4 from x3 to x1.
- Move b3 from x3 to x2.
- Move b3 from x2 to x3.
Question: Where is b4 now? Answer with a box tag like x1.
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
- 0 <= x1 <= 2

Constraints:
1. -x0 + 2*x1 != 6
2. -3*x0 + x1 != 5
3. x1 != 2

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
``
Consider the graph:

graph { 0--1; 1--3; 2--3; 3--4 }

Find the lexicographically smallest shortest path from Node 2 to Node 1.
If no path exists, answer `None`.
The answer is a Python list of nodes or `None`.
``

**Answer:**
```
[2, 3, 1]
```

---

## [graph_node_centrality](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
``
Consider the following social network graph:

graph { 0--2; 0--3; 1--3; 1--4; 2--4; 3--4 }

Based on the number of connections, identify all nodes that are the most central (i.e., have the highest degree centrality). There may be more than one.
The answer is a Python list of node integers, sorted in increasing order. Example: `[3, 8]`.
``

**Answer:**
```
[3, 4]
```

---

## [graph_isomorphism](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
``
Consider two graphs described below.

Graph A:
Edges: 0-4, 1-3, 2-3, 2-4

Graph B:
Edges: 0-1, 2-3, 2-4, 3-4

Do Graph A and Graph B have the exact same structure, just with different node labels? (In other words, are they isomorphic?)
The answer is `True` or `False`.
``

**Answer:**
```
False
```

---

## [graph_successors](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
```
Consider the directed graph:

Edges: 2->2, 4->4, 5->5, 1->1, 0->3, 3->0

Queries: [(4, 1)]
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

Edges (X->Y means Y depends on X): 3->1, 5->1, 5->4

List all prerequisites of node 1 (recursively), leaves first.
Do not include the query node itself.
If A depends on B and both appear in your answer, B must appear before A.
The answer is a Python list of integers.
```

**Answer:**
```
[3, 5]
```

---

## [regex_following](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

**Prompt:**
```
The answer is a 26-character string that fully matches the regular expression: (y)(often)+
```

**Answer:**
```
yoftenoftenoftenoftenoften
```

---

## [regex_induction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

**Prompt:**
```
The answer is the shortest regex that fully matches all POSITIVE strings and none of the NEGATIVE strings.
POSITIVE: 'XX', 'AXA', 'XoX', 'AoXA', 'X', 'oXoA', 'XAo', 'X'
NEGATIVE: '999', ';', 'j', '48', 'm', '\', 'amount2', ';'
```

**Answer:**
```
(?:[XAo])+
```

---

## [regex_reasoning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

**Prompt:**
```
Consider the regular expressions A = a|a* and B = a|a*
Do A and B accept exactly the same set of strings?
The answer is Yes or No.
```

**Answer:**
```
Yes
```

---

## [lexical_knowledge](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/knowledge.py)

**Prompt:**
```
Context: WordNet (relation holds for any valid noun sense).
Select all cohyponyms(motor)
From: [machinery, aspirin, cart, rein, weir, computer, depot]
The answer is a JSON list.
```

**Answer:**
```
["computer", "machinery"]
```

---

## [parsability](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
```
(GRAMMAR)
S -> D
D -> 'alone'
D -> D

(STRING)
alone

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
< < > > < [ ] > [ ]

(QUESTION)
The answer is the fully parenthesized parse tree of STRING in Lisp style.
Given G_ex: S -> NP VP, NP -> 'd' N, N -> 'n', VP -> 'v' and "d n v", correct is (S (NP d (N n)) (VP v)).
```

**Answer:**
```
(start (seq (expr < (seq (expr < (seq ) >) (seq )) >) (seq (expr < (seq (expr [ (seq ) ]) (seq )) >) (seq (expr [ (seq ) ]) (seq )))))
```

---

## [continuation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
```
List all valid next tokens for this prefix. The answer is the list of valid tokens sorted alphabetically and separated by |, with STOP at the end if the prefix forms a complete string.
(GRAMMAR)
S -> A
A -> D D
D -> 'account'
D -> A
(PREFIX)
account account account
```

**Answer:**
```
account|STOP
```

---

## [locate_error](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
```
(GRAMMAR)
S -> B
B -> 'quite' B
B -> 'woman'

(STRING)
quite quite quite quite quite woman woman

The answer is the shortest contiguous span from STRING that ends at the first invalid token and occurs only once in STRING.
Mark the invalid token as >>token<<.
If the token alone is enough, answer just >>token<<.
If STRING is fully grammatical, answer OK.
If all shown tokens are valid but more are needed, answer INCOMPLETE.
One line only.
```

**Answer:**
```
woman >>woman<<
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
[

(TEMPLATE)
( ___ ___

Fill in the 2 blanks (___) to form a grammatical continuation of PREFIX using exactly 3 tokens.
Fixed tokens must remain in place. The answer is all 3 tokens space-separated.
```

**Answer:**
```
( ) ]
```

---

## [table_qa](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/table_qa.py)

**Prompt:**
```
Execute this SQL query on the table named dataframe:

Table 1:
country,price
Bolivia,38.78
Brunei Darussalam,223.88
Myanmar,361.31
Macao,40.49
Turkmenistan,231.06


SQL: SELECT ROUND(MIN(price), 2) FROM dataframe

The answer is the result as single value.
```

**Answer:**
```
38.78
```

---

## [table_conversion](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/table_qa.py)

**Prompt:**
```
Convert the following table from to_string to to_html.

        country         customer
        Romania       Carol Shaw
Solomon Islands   Vincent Martin
      Swaziland   David Robinson
        Croatia Courtney Hoffman
       Pakistan   Jennifer Adams

The answer is the converted table.
```

**Answer:**
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>country</th>
      <th>customer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Romania</td>
      <td>Carol Shaw</td>
    </tr>
    <tr>
      <td>Solomon Islands</td>
      <td>Vincent Martin</td>
    </tr>
    <tr>
      <td>Swaziland</td>
      <td>David Robinson</td>
    </tr>
    <tr>
      <td>Croatia</td>
      <td>Courtney Hoffman</td>
    </tr>
    <tr>
      <td>Pakistan</td>
      <td>Jennifer Adams</td>
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

Term: ((((\_0.b) c) (\v0.v0)) (\v0.a))

The answer is the β-normal form (compared up to α-equivalence).
``

**Answer:**
```
((b (\v0.v0)) (\v0.a))
```

---

## [term_unification](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/binding.py)

**Prompt:**
```
Find the most general unifier (MGU) of the following first-order terms.
Uppercase identifiers are variables; lowercase are constants / function symbols.

T1 = q(f(Y,e,g(Y,f(p(c,b,c)),Y)))
T2 = q(f(b,e,g(b,X,b)))

The answer is a Python dict mapping each bound variable (as a string key) to its fully-resolved ground term (as a string value), with keys sorted alphabetically.
Example: {'X': 'f(a)', 'Y': 'b'}
```

**Answer:**
```
{'X': 'f(p(c,b,c))', 'Y': 'b'}
```

---

## [code_execution](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/coding.py)

**Prompt:**
````
Predict the printed output of the following Python code:

```python
f = 16
v = 13
print(v - f)
```

The answer is the exact printed output string.
````

**Answer:**
```
-3
```

---

## [diff_prediction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/coding.py)

**Prompt:**
```
Below is the version history of a file.

Version 1a9553b:
1    | Job executive visit available
2    | Wide do capital business kind manage world
3    | Natural will open make
4    | Near first purpose
5    | Democrat force issue national

Version b835196:
1    | Job executive visit available
2    | Wide do capital business kind manage world
3    | Near first purpose
4    | Democrat force issue national

Version ea569cc:
1    | Job executive visit available
2    | Wide do capital business kind manage world
3    | Near first purpose
4    | Democrat force issue national

Version 12e3347:
1    | Job executive visit available
2    | Near first purpose
3    | Democrat force issue national

Generate the Unified Diff to transform version 12e3347 into version b835196.
The answer is the diff chunks only (no file headers), or empty if no changes.
```

**Answer:**
```
@@ -1,3 +1,4 @@
 Job executive visit available
+Wide do capital business kind manage world
 Near first purpose
 Democrat force issue national
```

---

## [diff_patching](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/coding.py)

**Prompt:**
```
Apply the following Unified Diff to the text.

Original Text (Version 91bef3e):
1    | Listen also capital network
2    | Natural pass middle behavior score assume
3    | grow large dinner set important
4    | Leave industry since beyond blood laugh itself political
5    | Reason lose those between way best state

Diff (91bef3e -> ca364eb):


The answer is the resulting text.
```

**Answer:**
```
Listen also capital network
Natural pass middle behavior score assume
grow large dinner set important
Leave industry since beyond blood laugh itself political
Reason lose those between way best state
```

---

