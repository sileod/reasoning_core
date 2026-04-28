# 📖 Task Gallery

[`arithmetics`](#arithmetics) · [`symbolic_arithmetics`](#symbolic_arithmetics) · [`equation_system`](#equation_system) · [`conjecture_entailment`](#conjecture_entailment) · [`proof_reconstruction`](#proof_reconstruction) · [`bayesian_association`](#bayesian_association) · [`bayesian_intervention`](#bayesian_intervention) · [`logic_nli`](#logic_nli) · [`evidence_retrieval`](#evidence_retrieval) · [`logic_formalization`](#logic_formalization) · [`planning`](#planning) · [`set_intersection`](#set_intersection) · [`set_missing_element`](#set_missing_element) · [`count_elements`](#count_elements) · [`set_equality`](#set_equality) · [`sequential_induction`](#sequential_induction) · [`qualitative_reasoning`](#qualitative_reasoning) · [`navigation`](#navigation) · [`reference_tracking`](#reference_tracking) · [`coreference`](#coreference) · [`constraint_satisfaction`](#constraint_satisfaction) · [`graph_pathfinding`](#graph_pathfinding) · [`graph_node_centrality`](#graph_node_centrality) · [`graph_isomorphism`](#graph_isomorphism) · [`graph_successors`](#graph_successors) · [`graph_dependencies`](#graph_dependencies) · [`regex_following`](#regex_following) · [`regex_induction`](#regex_induction) · [`regex_reasoning`](#regex_reasoning) · [`lexical_knowledge`](#lexical_knowledge) · [`parsability`](#parsability) · [`parsing`](#parsing) · [`continuation`](#continuation) · [`locate_error`](#locate_error) · [`constrained_continuation`](#constrained_continuation) · [`table_qa`](#table_qa) · [`table_conversion`](#table_conversion) · [`lambda_reduction`](#lambda_reduction) · [`term_unification`](#term_unification) · [`code_execution`](#code_execution) · [`diff_prediction`](#diff_prediction) · [`diff_patching`](#diff_patching)

---

## [arithmetics](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/arithmetics.py)

**Prompt:**
````
Evaluate (1).
The answer is a number.
````

**Answer:**
````
1
````

---

## [symbolic_arithmetics](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/arithmetics.py)

**Prompt:**
````
Simplify the following algebraic expression:
((4) + x)

The answer is the simplified expression.
````

**Answer:**
````
x + 4
````

---

## [equation_system](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/equation_system.py)

**Prompt:**
````
Solve the following system of equations for the variable 'X1'.

System:
  X2 + 28 = 0

The answer is the numerical value for X1, or 'No solution' / 'Multiple solutions' if a unique numerical solution does not exist.
````

**Answer:**
````
Multiple solutions
````

---

## [conjecture_entailment](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/formal_maths.py)

**Prompt:**
``
Decide if the given premises entail the conjecture (i.e., the conjecture is provable) using Superposition/Resolution/Paramodulation.

Domain: Logic Calculi

Premises:
- (theorem(or(or(X1,X2),X3))|~theorem(or(X1,or(X3,X2))))

Conjecture: `(theorem(or(X1,X2))|~theorem(or(or(X1,X2),or(X2,X2))))`

The answer is `True` (provable) or `False` (not provable).
``

**Answer:**
````
False
````

---

## [proof_reconstruction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/formal_maths.py)

**Prompt:**
````
Reconstruct the proof dependency graph.
Domain: Number Theory
Theorem: (equalish(X1,X2)|divides(X3,X2)|~less(X3,X1)|~divides(X1,X2))

Rules:
- Some clauses are axioms (no parents); do NOT list them
- All other clauses derive from exactly 2 parents
- Clauses can be reused as parents

Shuffled clauses:
1. (less(X3,X2)|~less(X1,X2)|~less(X3,X1))
2. (divides(X1,X2)|~less(X1,X3)|~less(X3,X2))
3. (less(X1,X2)|equalish(X1,X2)|~divides(X1,X2))
4. (equalish(X1,X2)|divides(X3,X2)|~less(X3,X1)|~divides(X1,X2))
5. (divides(X1,X2)|~less(X1,X2))

The answer is the list of derivations for derived clauses, one per line: CHILD <- PARENT_1, PARENT_2
Example: 5 <- 2, 4

````

**Answer:**
````
2 <- 1, 5
4 <- 2, 3
````

---

## [bayesian_association](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/causal_reasoning.py)

**Prompt:**
````
System:
P(X_0) = {'0': 0.8, '1': 0.2} 
X_2 ~ Noisy-AND(leak=0.0, weights={'X_0': 0.7, 'X_1': 0.8}) 
P(X_1) = {'0': 0.6, '1': 0.4}
Observed conditions:
Observing/Knowing that the state X_0 is equal to 1
Task: Compute probability distribution for X_2 (possible values: [0, 1]).

The answer is a Python dict mapping each value to its probability, rounded to 1 decimals.
Example: {0: 0.1, 1: 0.9}
````

**Answer:**
````
{0: 0.8, 1: 0.2}
````

---

## [bayesian_intervention](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/causal_reasoning.py)

**Prompt:**
````
System:
P(X_0) = {'0': 0.9, '1': 0.1} 
P(X_1) = {'0': 0.4, '1': 0.6} 
P(X_2) = {'0': 0.5, '1': 0.5}
Observed conditions:
Doing/Imposing that the state X_1 is equal to 0
Task: Compute probability distribution for X_2 (possible values: [0, 1]).

The answer is a Python dict mapping each value to its probability, rounded to 1 decimals.
Example: {0: 0.1, 1: 0.9}
````

**Answer:**
````
{0: 0.5, 1: 0.5}
````

---

## [logic_nli](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic.py)

**Prompt:**
````
Premise:
there is a room.
all quiet people anywhere are old
Fred enjoys skydiving
everyone anywhere who enjoys landscape photography is an old person
no old person in the room is old
everyone in the room is quiet if they writes a travel blog
Hypothesis:
Fred and Paul are old

If the Premise entails the Hypothesis, the label is 'entailment'.
If the Premise contradicts the Hypothesis, the label is 'contradiction'.
If neither, the label is 'neutral'.
The answer is exactly one word: neutral, contradiction, or entailment.
````

**Answer:**
````
neutral
````

---

## [evidence_retrieval](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic.py)

**Prompt:**
````
Premise:
[0] Mary is the only person in the room.
[1] Paul is not old
[2] everyone outside the room is an active member of a local robotics club if they is an active member of a local robotics club
[3] everyone in the room collects comic books if they are a quiet person
[4] at least one person in the room collects comic books
[5] Mary collects luxury watches
Hypothesis:
Mary collects comic books

Which statements in the premise entail the hypothesis?
The answer is the list of supporting statements, e.g. [0, 6, 7].
````

**Answer:**
````
[0, 4]
````

---

## [logic_formalization](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/logic.py)

**Prompt:**
````
Premise:
there is a room.
“all old people in the room are old” unless “someone who participates in citizen science projects related to wildlife monitoring likes someone who is quiet”
Mary and Fred are not quiet old people, are not an old quiet people or are old
everyone in the room is an old person if they are old
Paul collects antique clocks
someone in the room is not enjoys stand-up paddleboarding

Glossary (English phrase -> TPTP symbol):
  'collects antique clocks' -> preda
  'enjoys stand-up paddleboarding' -> predc
  'participates in citizen science projects related to wildlife monitoring' -> predg

Translate the premise into a single TPTP first-order-logic formula, joining the lines with '&'.
Connectives: '&', '|', '~', '=>', '<=>'. Quantifiers: '![X]:...' (forall) and '?[X]:...' (exists). Equality: '='.
Use the symbols from the glossary for verbalized predicates. Names (mary, paul, ...), 'room', 'person', and adjectives (old, tall, ...) appear as-is.
The answer is the TPTP formula only (no fof(...) wrapper, no commentary).
````

**Answer:**
````
(there_is_a_room)&
(~(?[X,Y]:((predg(X))&(quiet(Y))&(like(X,Y))))=>(![X]:(room(X)=>(old(X)=>old(X)))))&
(((~(quiet(mary)&old(mary)&person(mary)))|(~(old(mary)&quiet(mary)&person(mary)))|(old(mary)))&((~(quiet(fred)&old(fred)&person(fred)))|(~(old(fred)&quiet(fred)&person(fred)))|(old(fred))))&
(![X]:(room(X)=>(((old(X))=>(old(X)&person(X))))))&
(preda(paul))&
(?[X]:(room(X)&(~predc(X))))
````

---

## [planning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/planning.py)

**Prompt:**
````
[OBJECTS]
object_1

[ACTIONS]
action_0(x0, x1)
  Requires: fluent_0
  Effect: not fluent_0
action_1(x0, x1)
  Requires: fluent_0
  Effect: not fluent_0
action_2(x0)
  Requires: (not fluent_0)
  Effect: fluent_0

[STATE]
Initial true values: None

[GOAL]

fluent_0
Hint: Reference solution has 1 actions (but it may not be optimal).
The answer is the plan.
Answer format: Multiple lines, one action per line: action(obj1, obj2)
````

**Answer:**
````
    action_2(object_1)
````

---

## [set_intersection](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
````
Set1: {566, 97, 332, 172, 930, 982, 421, 95}
Set2: {566, 653, 330, 192, 982, 172}
The answer is the intersection of Set1 and Set2 as a Python set: {elem_1, elem_2, ..., elem_n}.
````

**Answer:**
````
{172, 566, 982}
````

---

## [set_missing_element](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
````
Set_A: {348, 345, 347, 341, 343, 340, 339}
The answer is the missing elements from Set_A as a Python set.
````

**Answer:**
````
{342, 344, 346}
````

---

## [count_elements](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
````
List: [1, 15, 20, 16, 13, 9, 10, 2, 11, 7]
How many times does 16 appear? The answer is a number.
````

**Answer:**
````
1
````

---

## [set_equality](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
````
Set1: {270, 997, 487, 645, 51, 209, 335, 27}
Set2: {270, 335, 27, 487, 997, 645, 51}
The answer is True if Set1 and Set2 contain exactly the same elements, False otherwise.
````

**Answer:**
````
False
````

---

## [sequential_induction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/sequential_induction.py)

**Prompt:**
````
Infer a recurrence for a sequence indexed from 0: [U0, U1, ..., U7].
Max recurrence degree: 1.

Allowed binary ops: +, -, *, **
- Previous terms must be referenced exactly as: U[n - 1] ... U[n - 1]
- You may use "n" (current index).
- The answer is the right-hand side only (do not write "U[n] =").
- Your recurrence degree must be <= 1.

Sequence: [3, -2, 4, -1, 5, 0, 6, 1]
Degree of recurrence: 1
Initial terms: [3]

The answer must hold for all n >= d and be as simple as possible.
````

**Answer:**
````
n - U[n - 1]
````

---

## [qualitative_reasoning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/qstr.py)

**Prompt:**
````
Qualitative reasoning over time intervals.
There are 5 entities labeled 0 through 4.
You are given the following facts (read 'i rel j' as 'entity i is rel to entity j'):
  2 finishes 0
  1 before 0
  4 starts 1
  3 before 0

Question: what is the temporal relation of interval 4 to interval 2?
The answer is exactly one of: after, before, contains, during, equals, finished-by, finishes, meets, met-by, overlapped-by, overlaps, started-by, starts.
Respond with only the relation name as the answer.
````

**Answer:**
````
before
````

---

## [navigation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/navigation.py)

**Prompt:**
````
Objects occupy distinct points on the integer grid [0, 4] x [0, 4].
North is +y and East is +x. Any object not mentioned in a step stays fixed.

Initial facts:
- B is right of C.
- C is above A.
- A is above B.
- B is right of A.
- A starts at (1, 2).
- C is right of A.
- B is below C.

Steps:
1. B moves by (0, 2).

What is the final coordinate of A? Answer as (x, y).

````

**Answer:**
````
(1, 2)
````

---

## [reference_tracking](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/tracking.py)

**Prompt:**
````
Inventory:
- b1: green
- b2: blue
- b3: blue
- b4: red
Initial state:
- b1 is in x1
- b2 is in x2
- b3 is in x2
- b4 is in x1
Moves:
- Move b1 from x1 to x3.
- Relocate b4 from x1 to x3.
- Transfer b4 from x3 into x2.
- Move it from x2 to x3.
Question: Where is b2 now? Answer with a box tag like x1.
````

**Answer:**
````
x2
````

---

## [coreference](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/coreference.py)

**Prompt:**
````
(1) A quiet tall chef named Eve praised a short stern banker named Anna.
(2) A quiet short pilot named Tom watched Eve.
(3) He thanked her.
(4) Anna praised Eve.

In sentence 3, what does the subject expression 'He' refer to?
The answer is the name of the person it refers to.
````

**Answer:**
````
Tom
````

---

## [constraint_satisfaction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/constraint_satisfaction.py)

**Prompt:**
````
Variables/domains:
- 0 <= x0 <= 1
- 0 <= x1 <= 1

Constraints:
1. x0 == 0
2. x0 + 3*x1 != 5
3. -2*x0 >= -1

Enumerate ALL satisfying assignments in variable order [x0, x1].
The answer is a Python list of lists of ints, sorted lexicographically, or UNSAT if no assignment exists.

````

**Answer:**
````
[[0, 0], [0, 1]]
````

---

## [graph_pathfinding](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
``
Consider the directed graph:

Directed Edges: 0->1, 1->0, 1->2, 1->4, 2->1, 2->3

Find the lexicographically smallest shortest directed path from Node 1 to Node 4.
If no path exists, answer `None`.
The answer is a Python list of nodes or `None`.
``

**Answer:**
````
[1, 4]
````

---

## [graph_node_centrality](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
``
Consider the following directed network graph:

digraph { 0->4; 1->0; 2->1; 3->2; 3->4 }

Based on the total number of connections (summing both incoming and outgoing edges for each node), identify all nodes that are the most central (i.e., have the highest total degree).
There may be more than one.
The answer is a Python list of node integers, sorted in increasing order. Example: `[3, 8]`.
``

**Answer:**
````
[0, 1, 2, 3, 4]
````

---

## [graph_isomorphism](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
``
Consider two directed graphs described below.

Graph A:
Adjacency Dictionary (source to targets): {0: [1, 2], 1: [0, 3, 4], 2: [3, 4], 3: [1, 2], 4: [2]}

Graph B:
digraph { 0->2; 0->3; 1->0; 1->4; 2->0; 2->4; 3->0; 4->1; 4->2; 4->3 }

Do Graph A and Graph B have the exact same structure, just with different node labels? (In other words, are they isomorphic?)
The answer is `True` or `False`.
``

**Answer:**
````
True
````

---

## [graph_successors](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
````
Consider the directed graph:

digraph { 0->0; 1->2; 2->5; 3->1; 4->4; 5->3 }

Queries: [(3, 2)]
Each pair (x, k) asks for the k-th successor of x (following exact directed edges k times).
The answer is a Python list of integers in query order.
````

**Answer:**
````
[2]
````

---

## [graph_dependencies](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
````
Consider the directed graph:

Directed Edges: 3->2, 4->3, 5->0, 5->4

In this scenario, a directed edge from U to V means V depends on U (so U is a prerequisite of V).
List all prerequisites of node 2 (recursively), making sure to order base prerequisites first.
Do not include the query node itself.
If A is a prerequisite of B and both appear in your answer, A must appear before B.
Tie-break nodes with no mutual dependency lexicographically (smaller node ID first).
The answer is a Python list of integers.
````

**Answer:**
````
[5, 4, 3]
````

---

## [regex_following](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

**Prompt:**
````
The answer is a 7-character string that fully matches the regular expression: [^KnP]+\]+
````

**Answer:**
````
ag<;]]]
````

---

## [regex_induction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

**Prompt:**
````
The answer is the shortest regex that fully matches all POSITIVE strings and none of the NEGATIVE strings.
POSITIVE: '[p', '[%', '[Q', '[g', '['', '[]', '[5', '[s'
NEGATIVE: ')', 'Ý6õ', 'B', '#jjjj', '5777', 'T', 'W', 'Â'
````

**Answer:**
````
(?:\[)[^Sxe]
````

---

## [regex_reasoning](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/regex.py)

**Prompt:**
````
Consider the regular expressions A = (cb) and B = (cb)
Do A and B accept exactly the same set of strings?
The answer is Yes or No.
````

**Answer:**
````
Yes
````

---

## [lexical_knowledge](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/knowledge.py)

**Prompt:**
````
Context: WordNet (relation holds for any valid noun sense).

is_a(hat, clothing)
The answer is True or False.
````

**Answer:**
````
True
````

---

## [parsability](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
````
(GRAMMAR)
S -> C
C -> 'seem'
C -> C

(STRING)
seem

(QUESTION)
What is the parsability of this string?
The answer is exactly one word: unambiguous, ambiguous, or unparsable.
````

**Answer:**
````
ambiguous
````

---

## [parsing](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
````
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
⟨ ( ) ( ) ⟩ ⟪ ⟦ ⟧ ⟫ ⟪ ⟫ < >

(QUESTION)
Identify the Part-of-Speech (immediate parent) and tree depth for each token.
format per token: token<POS:depth>
Example: the<Det:3> cat<Noun:3>
````

**Answer:**
````
⟨<expr:3> (<expr:5> )<expr:5> (<expr:6> )<expr:6> ⟩<expr:3> ⟪<expr:4> ⟦<expr:6> ⟧<expr:6> ⟫<expr:4> ⟪<expr:5> ⟫<expr:5> <<expr:6> ><expr:6>
````

---

## [continuation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
````
List all valid next tokens for this prefix. The answer is the list of valid tokens sorted alphabetically and separated by |, with STOP at the end if the prefix forms a complete string.
(GRAMMAR)
S -> B
B -> 'yard'
B -> 'himself' B
(PREFIX)
himself himself himself
````

**Answer:**
````
himself|yard
````

---

## [locate_error](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
````
(GRAMMAR)
start -> seq
seq -> 
seq -> expr seq
expr -> '(' seq ')'
expr -> '[' seq ']'
expr -> '<' seq '>'

(STRING)
[ ( ) [ ] ) ]

The answer is the shortest contiguous span from STRING that ends at the first invalid token and occurs only once in STRING.
Mark the invalid token as >>token<<.
If the token alone is enough, answer just >>token<<.
If STRING is fully grammatical, answer OK.
If all shown tokens are valid but more are needed, answer INCOMPLETE.
One line only.
````

**Answer:**
````
] >>)<<
````

---

## [constrained_continuation](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
````
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
___ ] ___

Fill in the 2 blanks (___) to form a grammatical continuation of PREFIX using exactly 3 tokens.
Fixed tokens must remain in place. The answer is all 3 tokens space-separated.
````

**Answer:**
````
[ ] >
````

---

## [table_qa](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/table_qa.py)

**Prompt:**
````
Execute this SQL query on the table named dataframe:

Table 1:
 qty  rating
 305     2.7
 473     3.0
 499     4.9
 800     4.8
 561     5.0

SQL: SELECT ROUND(AVG(qty * rating), 2) FROM dataframe

The answer is the result as single value.
````

**Answer:**
````
2266.52
````

---

## [table_conversion](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/table_qa.py)

**Prompt:**
````
Convert the following table from to_csv to to_markdown.

job,company
"Therapist, occupational","Harding, Mendez and Wallace"
"Engineer, manufacturing systems",Lee-Jackson
Biomedical engineer,"Baker, Le and Sanchez"
Newspaper journalist,"Hall, Tran and Anderson"
Company secretary,"Burton, Gross and Sandoval"


The answer is the converted table.
````

**Answer:**
````
| job                             | company                     |
|:--------------------------------|:----------------------------|
| Therapist, occupational         | Harding, Mendez and Wallace |
| Engineer, manufacturing systems | Lee-Jackson                 |
| Biomedical engineer             | Baker, Le and Sanchez       |
| Newspaper journalist            | Hall, Tran and Anderson     |
| Company secretary               | Burton, Gross and Sandoval  |
````

---

## [lambda_reduction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/binding.py)

**Prompt:**
``
Reduce the following untyped λ-term to β-normal form.
Syntax: `\x.body` denotes λx.body; application is left-associative juxtaposition; free identifiers are treated as constants.

Term: (\v0.(((\_0.v0) a) v0))

The answer is the β-normal form (compared up to α-equivalence).
``

**Answer:**
````
(\v0.(v0 v0))
````

---

## [term_unification](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/binding.py)

**Prompt:**
````
Find the most general unifier (MGU) of the following first-order terms.
Uppercase identifiers are variables; lowercase are constants / function symbols.

T1 = h(q(q(d,X)),b)
T2 = h(q(q(d,g(f(d),c))),Y)

The answer is a Python dict mapping each bound variable (as a string key) to its fully-resolved ground term (as a string value), with keys sorted alphabetically.
Example: {'X': 'f(a)', 'Y': 'b'}
````

**Answer:**
````
{'X': 'g(f(d),c)', 'Y': 'b'}
````

---

## [code_execution](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/code_execution.py)

**Prompt:**
````
Predict the printed output of the following Python code:

```python
y = 12
v = 10
print("hi")
```

The answer is the exact printed output string.
````

**Answer:**
````
hi
````

---

## [diff_prediction](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/code_diff.py)

**Prompt:**
````
Below is the version history of a file.

Version 5ba64b6:
1    | Must home treat lose
2    | Provide despite girl vote method difficult quickly
3    | Former six fly as
4    | Window we open as book soon
5    | Million trip price possible

Version 1ec84c9:
1    | Must home treat lose
2    | Provide despite girl vote method difficult quickly
3    | Former movie fly as
4    | Window we open as book soon
5    | Million trip price possible

Generate the Unified Diff to transform version 1ec84c9 into version 5ba64b6.
The answer is the diff chunks only (no file headers), or empty if no changes.
````

**Answer:**
````
@@ -1,5 +1,5 @@
 Must home treat lose
 Provide despite girl vote method difficult quickly
-Former movie fly as
+Former six fly as
 Window we open as book soon
 Million trip price possible
````

---

## [diff_patching](https://github.com/sileod/reasoning-core/blob/main/reasoning_core/tasks/code_diff.py)

**Prompt:**
````
Apply the following Unified Diff to the text.

Original Text (Version f3cf0e1):
1    | Possible energy determine later again southern day
2    | Speak movement positive sign miss
3    | Speak buy cell through year involve less
4    | Expert consumer rate describe town painting church fire

Diff (f3cf0e1 -> ae774a8):


The answer is the resulting text.
````

**Answer:**
````
Possible energy determine later again southern day
Speak movement positive sign miss
Speak buy cell through year involve less
Expert consumer rate describe town painting church fire
````

---

