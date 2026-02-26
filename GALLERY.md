# 📖 Task Gallery

[`planning`](#planning) · [`table_qa`](#table_qa) · [`table_conversion`](#table_conversion) · [`equation_system`](#equation_system) · [`code_execution`](#code_execution) · [`diff_prediction`](#diff_prediction) · [`diff_patching`](#diff_patching) · [`regex_following`](#regex_following) · [`regex_induction`](#regex_induction) · [`graph_pathfinding`](#graph_pathfinding) · [`graph_node_centrality`](#graph_node_centrality) · [`graph_cycle_detection`](#graph_cycle_detection) · [`graph_isomorphism`](#graph_isomorphism) · [`arithmetics`](#arithmetics) · [`symbolic_arithmetics`](#symbolic_arithmetics) · [`sequential_induction`](#sequential_induction) · [`conjecture_entailment`](#conjecture_entailment) · [`proof_reconstruction`](#proof_reconstruction) · [`bayesian_association`](#bayesian_association) · [`bayesian_intervention`](#bayesian_intervention) · [`logic_nli`](#logic_nli) · [`evidence_retrieval`](#evidence_retrieval) · [`parsability`](#parsability) · [`parsing`](#parsing) · [`continuation`](#continuation) · [`set_intersection`](#set_intersection) · [`set_missing_element`](#set_missing_element) · [`count_elements`](#count_elements) · [`set_equality`](#set_equality)

---

## [planning](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/planning.py)

**Prompt:**
```
[OBJECTS]
object_1

[ACTIONS]
action_0(x0, x1)
  Requires: (not fluent_0)
  Effect: fluent_0

[STATE]
Default: False
Initial true values: None

[GOAL]

fluent_0

[OUTPUT]
Return only the plan.
Format: Multiple lines, one action per line: action(obj1, obj2)
```

**Answer:**
```
    action_0(object_1, object_1)
```

---

## [table_qa](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/table_qa.py)

**Prompt:**
```
Execute this SQL query on the table:

revenue,email
763.7,victorthompson@example.com
671.66,ihoover@example.org
833.77,andersontanner@example.net
815.8,austinronald@example.net
884.15,awatson@example.org


SQL: SELECT * FROM dataframe ORDER BY revenue DESC LIMIT 3

Return result as CSV format (rows separated by newlines, values by commas).
```

**Answer:**
```
884.15,awatson@example.org
833.77,andersontanner@example.net
815.8,austinronald@example.net
```

---

## [table_conversion](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/table_qa.py)

**Prompt:**
```
Convert the following table from to_markdown to to_json.

| customer        |   qty |
|:----------------|------:|
| Samuel Wright   |   844 |
| Christine Yates |   254 |
| Jessica Sanchez |   728 |
| Henry Sherman   |    32 |
| Ashley Moyer    |   617 |

Output only the converted table.
```

**Answer:**
```
[
    {
        "customer":"Samuel Wright",
        "qty":844
    },
    {
        "customer":"Christine Yates",
        "qty":254
    },
    {
        "customer":"Jessica Sanchez",
        "qty":728
    },
    {
        "customer":"Henry Sherman",
        "qty":32
    },
    {
        "customer":"Ashley Moyer",
        "qty":617
    }
]
```

---

## [equation_system](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/equation_system.py)

**Prompt:**
```
Solve the following system of equations for the variable 'X2'.

System:
  X1 - 19 = 0

Return the numerical value for X2. If a unique numerical solution does not exist, return either 'No solution' or 'Multiple solutions'.
```

**Answer:**
```
Multiple solutions
```

---

## [code_execution](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/coding.py)

**Prompt:**
````
Predict the printed output of the following Python code:

```python
y = 7
h = 12
print(h + h)
```

Return only the exact printed output string.
````

**Answer:**
```
24
```

---

## [diff_prediction](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/coding.py)

**Prompt:**
```
Below is the version history of a file.

Version e8f7c89:
1    | Involve traditional type still administration
2    | Agent generation serve street class
3    | Skin within far take of deep enter
4    | Up nearly single ball argue
5    | Upon rest anyone right foot despite professional

Version b5abc8a:
1    | Agent generation serve street class
2    | Skin within far take of deep enter
3    | Up nearly single ball argue
4    | Upon rest anyone right foot despite professional

Version 7c41e7e:
1    | Agent generation serve street class
2    | Skin within far take of deep enter
3    | Up nearly single ball argue
4    | support itself myself far arm
5    | Upon rest anyone right foot despite professional

Generate the Unified Diff to transform version e8f7c89 into version 7c41e7e.
Answer with the diff chunks only (no file headers). If no changes, return nothing.
```

**Answer:**
```
@@ -1,5 +1,5 @@
-Involve traditional type still administration
 Agent generation serve street class
 Skin within far take of deep enter
 Up nearly single ball argue
+support itself myself far arm
 Upon rest anyone right foot despite professional
```

---

## [diff_patching](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/coding.py)

**Prompt:**
```
Apply the following Unified Diff to the text.

Original Text (Version a68b5a2):
1    | Religious contain he face or want
2    | To scientist involve economy city resource cause build
3    | On loss available marriage kid what
4    | Determine expert miss play long
5    | Best bed impact peace when drug

Diff (a68b5a2 -> de8cf1c):


Answer with the resulting text only.
```

**Answer:**
```
Religious contain he face or want
To scientist involve economy city resource cause build
On loss available marriage kid what
Determine expert miss play long
Best bed impact peace when drug
```

---

## [regex_following](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/regex_following.py)

**Prompt:**
```
'daf' is a valid match for regex '[a-z]{3}' but not 'ab1'
Return a valid match for (\dnational)*
```

**Answer:**
```
0national8national2national
```

---

## [regex_induction](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/regex_following.py)

**Prompt:**
```
Return a regex that matches all POSITIVE strings and none of the NEGATIVE strings.
POSITIVE: '', '
', '.', ' .', '
', '	', '	.', ''
NEGATIVE: 'O', '\', '.2', 'lao', '..', '8', 'aÍ', 'suggestR'
```

**Answer:**
```
(\s|G)(\.?)
```

---

## [graph_pathfinding](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
``
Consider the graph:

0: 0-3 0-4; 1: 1-3; 2: 2-3; 3: 3-0 3-1 3-2; 4: 4-0

Find the shortest path from Node 3 to Node 2.
Answer with a Python list of integers. Example: `[0, 5, 3, 9]`.
``

**Answer:**
```
[3, 2]
```

---

## [graph_node_centrality](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
``
Consider the following social network graph:

graph { 0--1; 0--2; 0--4; 1--3 }

Based on the number of connections, identify all nodes that are the most central (i.e., have the highest degree centrality). There may be more than one.
Your answer must be a Python list of node integers, sorted in increasing order. Example: `[3, 8]`.
``

**Answer:**
```
[0]
```

---

## [graph_cycle_detection](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
``
Consider the graph below, which contains exactly one cycle.

Edges: 0-1, 0-4, 1-2, 2-3, 3-4

Identify all the nodes that form the cycle.
Your answer must be a Python list of node integers, sorted in increasing order. Example: `[2, 5, 7, 8]`.
``

**Answer:**
```
[0, 1, 2, 3, 4]
```

---

## [graph_isomorphism](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/graph_operations.py)

**Prompt:**
``
Consider two graphs described below.

Graph A:
Edges: 0-3, 0-4, 1-2, 1-3, 2-4

Graph B:
0: 0-1 0-3; 1: 1-0 1-2; 2: 2-1 2-4; 3: 3-0 3-4; 4: 4-2 4-3

Do Graph A and Graph B have the exact same structure, just with different node labels? (In other words, are they isomorphic?)
Answer with only `True` or `False`.
``

**Answer:**
```
True
```

---

## [arithmetics](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/arithmetics.py)

**Prompt:**
```
Evaluate (-10).
Answer with only a number.
```

**Answer:**
```
-10
```

---

## [symbolic_arithmetics](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/arithmetics.py)

**Prompt:**
```
Simplify the following algebraic expression:
x * x

Answer with the simplified expression.
```

**Answer:**
```
x**2
```

---

## [sequential_induction](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/sequential_induction.py)

**Prompt:**
```
Infer a recurrence for a sequence indexed from 0: [U0, U1, ..., U7].
Max recurrence degree: 1.

Allowed binary ops: +, -, *, **
- Previous terms must be referenced exactly as: U[n - 1] ... U[n - 1]
- You may use "n" (current index).
- Output ONLY the right-hand side (do not write "U[n] =").
- Your recurrence degree must be <= 1.

Sequence: [-1, 0, 4, 13, 29, 54, 90, 139]
Degree of recurrence: 1
Initial terms: [-1]

Answer must hold for all n >= d and be as simple as possible.
```

**Answer:**
```
n**2 + U[n - 1]
```

---

## [conjecture_entailment](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/formal_maths.py)

**Prompt:**
``
Decide if the given premises entail the conjecture (i.e., the conjecture is provable) using Superposition/Resolution/Paramodulation.

Domain: Ring Theory

Premises:
- (X1=additive_inverse(X2)|add(X2,add(X1,X3))!=X3)

Conjecture: `(X1=additive_inverse(X2)|add(X2,X3)!=add(X3,additive_inverse(X1)))`

Output only `True` (provable) or `False` (not provable).
``

**Answer:**
```
False
```

---

## [proof_reconstruction](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/formal_maths.py)

**Prompt:**
```
Reconstruct the proof dependency graph.
Domain: Logic Calculi
Theorem: (implies(not(X1),not(truth))=X1)

Rules:
- Some clauses are axioms (no parents); do NOT list them
- All other clauses derive from exactly 2 parents
- Clauses can be reused as parents

Shuffled clauses:
1. (implies(X1,X1)=implies(implies(X1,truth),truth))
2. (implies(implies(not(not(truth)),X1),X1)=truth)
3. (implies(not(X1),X2)=implies(not(X2),X1))
4. (implies(not(not(truth)),not(not(truth)))=truth)
5. (implies(not(X1),not(truth))=X1)
6. (implies(implies(not(X1),not(truth)),X1)=truth)
7. (not(not(truth))=truth)

Output derivations for derived clauses only, one per line: CHILD <- PARENT_1, PARENT_2
Example: 5 <- 2, 4

```

**Answer:**
```
2 <- 3, 6
4 <- 1, 2
5 <- 3, 7
7 <- 2, 4
```

---

## [bayesian_association](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/causal_reasoning.py)

**Prompt:**
```
System:
P(X_0) = {'0': 0.0, '1': 1.0} 
P(X_1) = {'0': 0.4, '1': 0.6} 
P(X_2) = {'0': 0.4, '1': 0.6}
Observed conditions:
Observing/Knowing that the state X_2 is equal to 0
Task: Compute probability distribution for X_1 (possible values: [0, 1]).

Output: Python dict mapping each value to its probability, rounded to 1 decimals.
Example: {0: 0.1, 1: 0.9}
```

**Answer:**
```
{0: 0.4, 1: 0.6}
```

---

## [bayesian_intervention](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/causal_reasoning.py)

**Prompt:**
```
System:
P(X_0) = {'0': 0.7, '1': 0.3} 
P(X_1) = {'0': 0.9, '1': 0.1} 
P(X_2) = {'0': 0.4, '1': 0.6}
Observed conditions:
Doing/Imposing that the state X_1 is equal to 1
Task: Compute probability distribution for X_2 (possible values: [0, 1]).

Output: Python dict mapping each value to its probability, rounded to 1 decimals.
Example: {0: 0.1, 1: 0.9}
```

**Answer:**
```
{0: 0.4, 1: 0.6}
```

---

## [logic_nli](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/logic.py)

**Prompt:**
```
Premise:
there is a room.
Mary is quiet
all quiet persons in the room are quiet
Mary has a piercing
Paul who enjoys stand-up paddleboarding is a quiet person
Paul has a piercing
Hypothesis:
Mary is not not quiet

If the Premise entails the Hypothesis, the label is 'entailment'.
If the Premise contradicts the Hypothesis, the label is 'contradiction'.
If neither, the label is 'neutral'.
Answer with exactly one word, neutral|contradiction|entailment
```

**Answer:**
```
entailment
```

---

## [evidence_retrieval](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/logic.py)

**Prompt:**
```
Premise:
[0] there is a room.
[1] someone in the room watches fantasy movies
[2] no quiet person in the room is quiet
[3] Paul is not quiet
[4] Mary who enjoys stargazing, has a vast collection of first-edition science fiction novels or is a quiet person is not allergic to anything
[5] all quiet persons in the room are quiet
Hypothesis:
Paul is not not quiet

Which statements in the premise contradict the hypothesis?
Only answer the list of supporting statements, e.g. [0, 6, 7].
```

**Answer:**
```
[3]
```

---

## [parsability](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
```
(GRAMMAR)
S -> B
B -> 'kid'
B -> B

(STRING)
kid

(QUESTION)
What is the parsability of this string?
Answer with exactly one word, unambiguous|ambiguous|unparsable
```

**Answer:**
```
ambiguous
```

---

## [parsing](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
```
(GRAMMAR)
S -> C
C -> 'check' C
C -> 'look'
D -> A

(STRING)
check check check look

(QUESTION)
Identify the Part-of-Speech (immediate parent) and tree depth for each token.
format per token: token<POS:depth>
Example: the<Det:3> cat<Noun:3>
```

**Answer:**
```
check<C:2> check<C:3> check<C:4> look<C:5>
```

---

## [continuation](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/grammar.py)

**Prompt:**
```
List all valid next tokens for this prefix. Answer sorted alphabetically separated by |, with STOP at the end if complete.
(GRAMMAR)
S -> B
B -> 'lay'
B -> 'recognize' B
B -> A
(PREFIX)
recognize
```

**Answer:**
```
lay|recognize
```

---

## [set_intersection](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
```
Set1: {322, 739, 450, 973, 96, 908, 354, 761}
Set2: {761, 322, 994, 908, 224, 992}
Only return the intersection of Set1 and Set2 as a Python set: {elem_1, elem_2, ..., elem_n}.
```

**Answer:**
```
{322, 761, 908}
```

---

## [set_missing_element](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
```
Set_A: {135, 131, 132, 138, 140, 134, 137}
Only return the missing elements from Set_A as a Python set.
```

**Answer:**
```
{133, 136, 139}
```

---

## [count_elements](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
```
List: [5, 20, 7, 6, 15, 6, 3, 3, 12, 6]
How many times does 3 appear? Only return the number.
```

**Answer:**
```
2
```

---

## [set_equality](https://github.com/sileod/reasoning_core/blob/main/reasoning_core/tasks/set_operations.py)

**Prompt:**
```
Set1: {340, 356, 531, 697, 990, 737, 241, 964}
Set2: {356, 990, 340, 964, 697, 531, 241, 737}
Only return True if Set1 and Set2 contain exactly the same elements, False otherwise.
```

**Answer:**
```
True
```

---

