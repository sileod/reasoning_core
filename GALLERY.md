# 📖 Task Gallery

[`arithmetics`](#arithmetics) · [`equation_system`](#equation_system) · [`conjecture_entailment`](#conjecture_entailment) · [`proof_reconstruction`](#proof_reconstruction) · [`bayesian_association`](#bayesian_association) · [`bayesian_intervention`](#bayesian_intervention) · [`logic_nli`](#logic_nli) · [`evidence_retrieval`](#evidence_retrieval) · [`logic_formalization`](#logic_formalization) · [`planning`](#planning) · [`set_intersection`](#set_intersection) · [`set_missing_element`](#set_missing_element) · [`count_elements`](#count_elements) · [`set_equality`](#set_equality) · [`sequential_induction`](#sequential_induction) · [`qualitative_reasoning`](#qualitative_reasoning) · [`navigation`](#navigation) · [`reference_tracking`](#reference_tracking) · [`coreference`](#coreference) · [`constraint_satisfaction`](#constraint_satisfaction) · [`graph_pathfinding`](#graph_pathfinding) · [`graph_isomorphism`](#graph_isomorphism) · [`graph_successors`](#graph_successors) · [`graph_dependencies`](#graph_dependencies) · [`regex_following`](#regex_following) · [`regex_induction`](#regex_induction) · [`regex_reasoning`](#regex_reasoning) · [`lexical_knowledge`](#lexical_knowledge) · [`parsability`](#parsability) · [`parsing`](#parsing) · [`continuation`](#continuation) · [`locate_error`](#locate_error) · [`constrained_continuation`](#constrained_continuation) · [`table_qa`](#table_qa) · [`table_conversion`](#table_conversion) · [`lambda_reduction`](#lambda_reduction) · [`term_unification`](#term_unification) · [`code_execution`](#code_execution) · [`diff_prediction`](#diff_prediction) · [`diff_patching`](#diff_patching)

---

## [arithmetics](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/arithmetics.py)

**Prompt:**
```
Evaluate 5 * -3.90.
The answer is a number.
```

**Answer:**
```
-19.5
```

---

## [equation_system](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/equation_system.py)

**Prompt:**
```
Solve the following system of equations for the variable 'X2'.

System:
  X1 - 2 = 0
  X2 + 9 = 0

The answer is the numerical value for X2, or 'No solution' / 'Multiple solutions' if a unique numerical solution does not exist.
```

**Answer:**
```
-9
```

---

## [conjecture_entailment](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/formal_maths.py)

**Prompt:**
```
Decide if the given premises entail the conjecture (i.e., the conjecture is provable) using Superposition/Resolution/Paramodulation.

Domain: Logic Calculi

Premises:
- (theorem(or(or(X1,or(X2,X3)),X4))|~theorem(or(X3,X4)))

Conjecture: `(theorem(or(X1,or(X2,or(X3,X4))))|~theorem(or(X4,X1)))`

The answer is `True` (provable) or `False` (not provable).
```

**Answer:**
```
False
```

---

## [proof_reconstruction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/formal_maths.py)

**Prompt:**
```
Reconstruct the proof dependency graph.
Domain: Geometry
Theorem: (~between_c(X1,X2,X2,X3))

Rules:
- Some clauses are axioms (no parents); do NOT list them
- All other clauses derive from exactly 2 parents
- Clauses can be reused as parents

Shuffled clauses:
1. (end_point(X2,ax1_sk1(X4,X3,X2,X1))|~between_c(X1,X2,X3,X4))
2. (~inner_point(X1,ax1_sk1(X2,X3,X1,X4))|~between_c(X4,X1,X3,X2))
3. (~between_c(X1,X2,X2,X3))
4. (inner_point(X3,ax1_sk1(X4,X3,X2,X1))|~between_c(X1,X2,X3,X4))
5. (~inner_point(X1,X2)|~end_point(X1,X2))

The answer is the list of derivations for derived clauses, one per line: CHILD <- PARENT_1, PARENT_2
Example: 5 <- 2, 4

```

**Answer:**
```
2 <- 1, 5
3 <- 2, 4
```

---

## [bayesian_association](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/causal_reasoning.py)

**Prompt:**
```
System:
P(X_0) = {'0': 0.5, '1': 0.5} 
P(X_1|X_0=0) = {'0': 0.5, '1': 0.5} 
P(X_1|X_0=1) = {'0': 0.3, '1': 0.7} 
P(X_2) = {'0': 0.3, '1': 0.7}
Observed conditions:
Observing/Knowing that the state X_0 is equal to 0
Task: Compute probability distribution for X_1 (possible values: [0, 1]).

The answer is a Python dict mapping each value to its probability, rounded to 1 decimals.
Example: {0: 0.1, 1: 0.9}
```

**Answer:**
```
{0: 0.5, 1: 0.5}
```

---

## [bayesian_intervention](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/causal_reasoning.py)

**Prompt:**
```
System:
P(X_0) = {'0': 0.7, '1': 0.3} 
X_2 ~ Noisy-AND(leak=0.0, weights={'X_0': 0.8, 'X_1': 0.8}) 
P(X_1) = {'0': 0.3, '1': 0.7}
Observed conditions:
Doing/Imposing that the state X_1 is equal to 0
Task: Compute probability distribution for X_2 (possible values: [0, 1]).

The answer is a Python dict mapping each value to its probability, rounded to 1 decimals.
Example: {0: 0.1, 1: 0.9}
```

**Answer:**
```
{0: 1.0, 1: 0.0}
```

---

## [logic_nli](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic.py)

**Prompt:**
```
Premise:
there is a room.
Fred collects vintage maps
Fred collects rare sneakers
more than one person in the room practices pilates
Paul trains for and competes in international triathlons
someone outside the room is a scotch connoisseur
Hypothesis:
Paul is not collects rare sneakers

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
[1] Mary is not old
[2] it is not the case that “Fred is not not quiet”
[3] Mary is not an old quiet person
[4] everyone in the room who is not old designs and sews custom cosplay costumes for conventions
[5] everyone in the room who enjoys white-water rafting is a drone photographer
Hypothesis:
Mary designs and sews custom cosplay costumes for conventions

Which statements in the premise entail the hypothesis?
The answer is the list of supporting statements, e.g. [0, 6, 7].
```

**Answer:**
```
[0, 1, 4]
```

---

## [logic_formalization](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic.py)

**Prompt:**
```
Premise:
Mary is the only person in the room.
Mary reads mystery novels
Mary has a vast collection of first-edition science fiction novels
it is not the case that “more than one person in the room is right-handed”
all old people in the room are quiet
everyone outside the room is a quiet person if they are old

Glossary (English phrase -> TPTP symbol):
  'has a vast collection of first-edition science fiction novels' -> predb
  'reads mystery novels' -> predf
  'is right-handed' -> predh

Translate the premise into a single TPTP first-order-logic formula, joining the lines with '&'.
Connectives: '&', '|', '~', '=>', '<=>'. Quantifiers: '![X]:...' (forall) and '?[X]:...' (exists). Equality: '='.
Use the symbols from the glossary for verbalized predicates. Names (mary, paul, ...), 'in_the_room', 'person', and adjectives (old, tall, ...) appear as-is.
The answer is the TPTP formula only (no fof(...) wrapper, no commentary).
```

**Answer:**
```
in_the_room(mary)&(![X]:(in_the_room(X)=>(X='mary')))&
(predf(mary))&
(predb(mary))&
(~((?[X,Y]:(in_the_room(X)&in_the_room(Y)&(predh(X)&predh(Y))&(X!=Y)))))&
(![X]:(in_the_room(X)=>(old(X)=>quiet(X))))&
(![X]:(~in_the_room(X)=>(((old(X))=>(quiet(X))))))
```

---

## [planning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/planning.py)

**Prompt:**
```
[OBJECTS]
object_1, object_2, object_3, object_4

[ACTIONS]
action_0(x0, x1)
  Requires: (not fluent_0)
  Effect: fluent_1, fluent_0

[STATE]
Default: False
Initial true values: fluent_1

[GOAL]

fluent_0
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
Set1: {594, 140, 945, 1, 593, 239, 63, 512}
Set2: {877, 188, 63, 945, 593, 273}
The answer is the intersection of Set1 and Set2 as a Python set: {elem_1, elem_2, ..., elem_n}.
```

**Answer:**
```
{63, 593, 945}
```

---

## [set_missing_element](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
```
Set_A: {386, 378, 381, 382, 384, 377, 379}
The answer is the missing elements from Set_A as a Python set.
```

**Answer:**
```
{380, 383, 385}
```

---

## [count_elements](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
```
List: [9, 2, 7, 15, 10, 18, 2, 19, 12, 3]
How many times does 2 appear? The answer is a number.
```

**Answer:**
```
2
```

---

## [set_equality](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
```
Set1: {'alf', 'akx', 'pj', 'mi', 'hs', 'ow', 'jy', 'ahl'}
Set2: {'akx', 'jy', 'ahl', 'pj', 'mi', 'ow', 'alf', 'hs'}
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

Sequence: [1, -6, 1, -6, 1, -6, 1, -6]
Degree of recurrence: 1
Initial terms: [1]

The answer must hold for all n >= d and be as simple as possible.
```

**Answer:**
```
-U[n - 1] - 5
```

---

## [qualitative_reasoning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/qstr.py)

**Prompt:**
```
There are 5 entities labeled 0 through 4.
You are given the following facts (read 'i rel j' as 'entity i is rel to entity j'):
  2 started-by 1
  4 starts 2
  0 contains 1
  3 equals 0
  0 contains 2

What is the relation of the horizontal extent of box 4 to that of box 3?
The answer is exactly one of: after, before, contains, during, equals, finished-by, finishes, meets, met-by, overlapped-by, overlaps, started-by, starts.
```

**Answer:**
```
during
```

---

## [navigation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/navigation.py)

**Prompt:**
```
Objects occupy distinct points on the integer grid [0, 4] x [0, 4].
North is +y and East is +x. Any object not mentioned in a step stays fixed.

Initial facts:
- B is in the same column as C.
- C is below B.
- B is above A.
- C starts at (0, 0).
- C is below A.
- C is left of A.
- A is right of B.

Steps:
1. C and A swap positions.

What is the final coordinate of A? Answer as (x, y).

```

**Answer:**
```
(0, 0)
```

---

## [reference_tracking](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/tracking.py)

**Prompt:**
```
Inventory:
- b1: red
- b2: red
- b3: yellow
- b4: yellow
Initial state:
- b1 is in x1
- b2 is in x3
- b3 is in x1
- b4 is in x3
Moves:
- Move all contents of x1 to x3.
- Transfer b1 from x3 into x1.
- Transfer b4 from x3 into x2.
- Move b4 from x2 to x3.
Where is b2 now? The answer is a box tag, like x1.
```

**Answer:**
```
x3
```

---

## [coreference](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/coreference.py)

**Prompt:**
```
(1) A kind short baker named Sam watched an old tall farmer named Ben.
(2) A stern tall doctor named Lucy thanked Ben.
(3) He thanked her.
(4) He thanked Sam.

In sentence 3, what does the object expression 'her' refer to?
The answer is the name of the person it refers to.
```

**Answer:**
```
Lucy
```

---

## [constraint_satisfaction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/constraint_satisfaction.py)

**Prompt:**
```
Variables/domains:
- 0 <= x0 <= 2
- 0 <= x1 <= 2

Constraints:
1. -2*x1 >= -2
2. 2*x0 <= 5
3. 3*x0 >= 0

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
Consider the directed graph:

digraph { 0->3; 1->2; 2->3; 3->0; 3->1; 3->2 }

Find the lexicographically smallest shortest directed path from Node 3 to Node 4.
If no path exists, answer `None`.
The answer is a Python list of nodes or `None`.
```

**Answer:**
```
None
```

---

## [graph_isomorphism](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
```
Consider two directed graphs described below.

Graph A:
digraph { 0->4; 1->0; 1->3; 2->3; 2->4; 3->1; 3->2 }

Graph B:
Directed Edges: 0->3, 1->0, 1->4, 2->3, 2->4, 3->1, 3->2

Do Graph A and Graph B have the exact same structure, just with different node labels? (In other words, are they isomorphic?)
The answer is `True` or `False`.
```

**Answer:**
```
False
```

---

## [graph_successors](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
```
Consider the directed graph:

Adjacency Dictionary (source to targets): {0: [2], 1: [4], 2: [3], 3: [5], 4: [0], 5: [1]}

Queries: [(4, 1)]
Each pair (x, k) asks for the k-th successor of x (following exact directed edges k times).
The answer is a Python list of integers in query order.
```

**Answer:**
```
[0]
```

---

## [graph_dependencies](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
```
Consider the directed graph:

Directed Edges: 0->1, 1->2, 4->3

In this scenario, a directed edge from U to V means V depends on U (so U is a prerequisite of V).
List all prerequisites of node 2 (recursively), making sure to order base prerequisites first.
Do not include the query node itself.
If A is a prerequisite of B and both appear in your answer, A must appear before B.
Tie-break nodes with no mutual dependency lexicographically (smaller node ID first).
The answer is a Python list of integers.
```

**Answer:**
```
[0, 1]
```

---

## [regex_following](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

**Prompt:**
```
The answer is a 2-character string that fully matches the regular expression: .W
```

**Answer:**
```
CW
```

---

## [regex_induction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

**Prompt:**
```
The answer is the shortest regex that fully matches all POSITIVE strings and none of the NEGATIVE strings.
POSITIVE: 'e', 'P', 'n'
NEGATIVE: '\5', 'J', 'lll', ']]]', 'MF', 'V7X', '[A', 'ZR'
```

**Answer:**
```
([nPe])
```

---

## [regex_reasoning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

**Prompt:**
```
Consider the regular expressions A = aab? and B = (a)|ab
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
Select all cohyponyms(acid)
From: [sludge, atom, metal, chloride, incense, asphalt, manure, nitrate]
The answer is a JSON list.
```

**Answer:**
```
["chloride", "incense", "nitrate"]
```

---

## [parsability](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
```
(GRAMMAR)
S -> B
B -> 'hold'
B -> B

(STRING)
hold

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
expr -> '⟨' seq '⟩'
expr -> '⟦' seq '⟧'
expr -> '⟪' seq '⟫'

(STRING)
[ < > ] < > ⟨ ⟩ ⟪ ⟫

(QUESTION)
Identify the Part-of-Speech (immediate parent) and tree depth for each token.
format per token: token<POS:depth>
Example: the<Det:3> cat<Noun:3>
```

**Answer:**
```
[<expr:3> <<expr:5> ><expr:5> ]<expr:3> <<expr:4> ><expr:4> ⟨<expr:5> ⟩<expr:5> ⟪<expr:6> ⟫<expr:6>
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
( ) (
```

**Answer:**
```
(|)|<|[
```

---

## [locate_error](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

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
< [ ) > < [ ] > < >

The answer is the shortest contiguous span from STRING that ends at the first invalid token and occurs only once in STRING.
Mark the invalid token as >>token<<.
If the token alone is enough, answer just >>token<<.
If STRING is fully grammatical, answer OK.
If all shown tokens are valid but more are needed, answer INCOMPLETE.
One line only.
```

**Answer:**
```
>>)<<
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
<

(TEMPLATE)
___ < ___

Fill in the 2 blanks (___) to form a grammatical continuation of PREFIX using exactly 3 tokens.
Fixed tokens must remain in place. The answer is all 3 tokens space-separated.
```

**Answer:**
```
> < >
```

---

## [table_qa](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/table_qa.py)

**Prompt:**
```
Execute this SQL query on the table named dataframe:

Table 1:
| price   | qty   |
|:--------|:------|
| 90.41   | 291.0 |
| 131.81  | 355.0 |
| 467.1   | 882.0 |
| 188.7   | 147.0 |
| 88.51   | 654.0 |

SQL: SELECT COUNT(*) FROM dataframe WHERE price > 131.81

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
Convert the following table from latex to html.

\begin{tabular}{ll}
\toprule
job & date \\
\midrule
Quality manager & 2025-06-23 \\
Planning and development surveyor & 2026-04-01 \\
Child psychotherapist & 2026-04-17 \\
Designer, television/film set & 2025-12-26 \\
Careers adviser & 2025-06-18 \\
\bottomrule
\end{tabular}


The answer is the converted table.
```

**Answer:**
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>job</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Quality manager</td>
      <td>2025-06-23</td>
    </tr>
    <tr>
      <td>Planning and development surveyor</td>
      <td>2026-04-01</td>
    </tr>
    <tr>
      <td>Child psychotherapist</td>
      <td>2026-04-17</td>
    </tr>
    <tr>
      <td>Designer, television/film set</td>
      <td>2025-12-26</td>
    </tr>
    <tr>
      <td>Careers adviser</td>
      <td>2025-06-18</td>
    </tr>
  </tbody>
</table>
```

---

## [lambda_reduction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/binding.py)

**Prompt:**
```
Reduce the following untyped λ-term to β-normal form.
Syntax: `\x.body` denotes λx.body; application is left-associative juxtaposition; free identifiers are treated as constants.

Term: ((((\_0.b) d) a) ((b d) d))

The answer is the β-normal form (compared up to α-equivalence).
```

**Answer:**
```
((b a) ((b d) d))
```

---

## [term_unification](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/binding.py)

**Prompt:**
```
Find the most general unifier (MGU) of the following first-order terms.
Uppercase identifiers are variables; lowercase are constants / function symbols.

T1 = h(f(f(h(p(d,d,c)),f(a,d,p(d,b,b)))))
T2 = h(f(f(X,Y)))

The answer is a Python dict mapping each bound variable (as a string key) to its fully-resolved ground term (as a string value), with keys sorted alphabetically.
Example: {'X': 'f(a)', 'Y': 'b'}
```

**Answer:**
```
{'X': 'h(p(d,d,c))', 'Y': 'f(a,d,p(d,b,b))'}
```

---

## [code_execution](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/code_execution.py)

**Prompt:**
````
Predict the printed output of the following Python code:

```python
s = 11
print(s)
```

The answer is the exact printed output string.
````

**Answer:**
```
11
```

---

## [diff_prediction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/code_diff.py)

**Prompt:**
```
Below is the version history of a file.

Version 0c7cad7:
1    | Away court manage think why sometimes happen
2    | Pass soon movement affect
3    | Sit occur everything
4    | Cost outside minute blue treatment heavy run
5    | Dog civil more really nation lay I

Version ae34d4f:
1    | Away court manage think why sometimes happen
2    | Pass soon movement affect
3    | Sit occur everything
4    | Cost herself minute blue treatment heavy run
5    | Dog civil more really nation lay I

Generate the Unified Diff to transform version ae34d4f into version 0c7cad7.
The answer is the diff chunks only (no file headers), or empty if no changes.
```

**Answer:**
```
@@ -1,5 +1,5 @@
 Away court manage think why sometimes happen
 Pass soon movement affect
 Sit occur everything
-Cost herself minute blue treatment heavy run
+Cost outside minute blue treatment heavy run
 Dog civil more really nation lay I
```

---

## [diff_patching](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/code_diff.py)

**Prompt:**
```
Apply the following Unified Diff to the text.

Original Text (Version b9c62f5):
1    | Later physical way name few little
2    | Cost consider daughter positive between
3    | Political bar direction shoulder plant room
4    | Particular nearly still performance itself half state
5    | Better clear easy effect find per today give

Diff (b9c62f5 -> 5d73cf8):


The answer is the resulting text.
```

**Answer:**
```
Later physical way name few little
Cost consider daughter positive between
Political bar direction shoulder plant room
Particular nearly still performance itself half state
Better clear easy effect find per today give
```

---

