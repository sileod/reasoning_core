# Samples: language separation (S38)


## Level 0


**Prompt:**

Two deterministic finite automata are given.

Automaton 1:
States: 0, 1, 2
Alphabet: a, b
Start state: 2
Accepting states: 2
Transition table (row i, column labeled by a symbol, entry = next state):
   | a b
---+---------
  0 | 1 0
  1 | 2 0
  2 | 1 0

Automaton 2:
States: 0, 1, 2
Alphabet: a, b
Start state: 0
Accepting states: 0, 2
Transition table (row i, column labeled by a symbol, entry = next state):
   | a b
---+---------
  0 | 2 0
  1 | 2 2
  2 | 1 2

Find the lexicographically smallest string of shortest length that is accepted by Automaton 1 and rejected by Automaton 2. If the two automata accept exactly the same set of strings and no such string exists, answer "none". The answer is the witness string, or "none".

**Answer:**

aa


**Prompt:**

Two deterministic finite automata are given.

Automaton 1:
States: 0, 1, 2
Alphabet: a, b
Start state: 0
Accepting states: 2
Transition table (row i, column labeled by a symbol, entry = next state):
   | a b
---+---------
  0 | 0 2
  1 | 2 2
  2 | 2 2

Automaton 2:
States: 0, 1, 2
Alphabet: a, b
Start state: 2
Accepting states: 2
Transition table (row i, column labeled by a symbol, entry = next state):
   | a b
---+---------
  0 | 1 1
  1 | 0 1
  2 | 1 2

Find the lexicographically smallest string of shortest length that is accepted by Automaton 1 and rejected by Automaton 2. If the two automata accept exactly the same set of strings and no such string exists, answer "none". The answer is the witness string, or "none".

**Answer:**

ab


## Level 2


**Prompt:**

Two deterministic finite automata are given.

Automaton 1:
States: 0, 1, 2, 3, 4
Alphabet: a, b
Start state: 0
Accepting states: 1, 3
Transition table (row i, column labeled by a symbol, entry = next state):
   | a b
---+---------
  0 | 0 2
  1 | 1 0
  2 | 3 3
  3 | 3 1
  4 | 2 4

Automaton 2:
States: 0, 1, 2, 3, 4
Alphabet: a, b
Start state: 1
Accepting states: 2
Transition table (row i, column labeled by a symbol, entry = next state):
   | a b
---+---------
  0 | 0 4
  1 | 1 4
  2 | 2 0
  3 | 1 1
  4 | 3 0

Find the lexicographically smallest string of shortest length that is accepted by Automaton 1 and rejected by Automaton 2. If the two automata accept exactly the same set of strings and no such string exists, answer "none". The answer is the witness string, or "none".

**Answer:**

ba


**Prompt:**

Two deterministic finite automata are given.

Automaton 1:
States: 0, 1, 2, 3, 4
Alphabet: a, b
Start state: 3
Accepting states: 2, 3, 4
Transition table (row i, column labeled by a symbol, entry = next state):
   | a b
---+---------
  0 | 2 1
  1 | 3 0
  2 | 4 2
  3 | 1 2
  4 | 3 1

Automaton 2:
States: 0, 1, 2, 3, 4
Alphabet: a, b
Start state: 3
Accepting states: 2, 3
Transition table (row i, column labeled by a symbol, entry = next state):
   | a b
---+---------
  0 | 1 1
  1 | 0 0
  2 | 3 4
  3 | 0 2
  4 | 3 0

Find the lexicographically smallest string of shortest length that is accepted by Automaton 1 and rejected by Automaton 2. If the two automata accept exactly the same set of strings and no such string exists, answer "none". The answer is the witness string, or "none".

**Answer:**

aa


## Level 5


**Prompt:**

Two deterministic finite automata are given.

Automaton 1:
States: 0, 1, 2, 3, 4, 5, 6, 7
Alphabet: a, b, c
Start state: 7
Accepting states: 6
Transition table (row i, column labeled by a symbol, entry = next state):
   | a b c
---+-------------
  0 | 0 0 1
  1 | 1 5 1
  2 | 3 1 1
  3 | 2 2 6
  4 | 5 2 6
  5 | 7 3 3
  6 | 4 2 0
  7 | 0 3 6

Automaton 2:
States: 0, 1, 2, 3, 4, 5, 6, 7
Alphabet: a, b, c
Start state: 5
Accepting states: 2, 3, 4, 6
Transition table (row i, column labeled by a symbol, entry = next state):
   | a b c
---+-------------
  0 | 6 4 6
  1 | 5 6 7
  2 | 2 5 0
  3 | 2 0 0
  4 | 5 1 7
  5 | 4 1 4
  6 | 4 0 3
  7 | 3 6 5

Find the lexicographically smallest string of shortest length that is accepted by Automaton 1 and rejected by Automaton 2. If the two automata accept exactly the same set of strings and no such string exists, answer "none". The answer is the witness string, or "none".

**Answer:**

bc


**Prompt:**

Two deterministic finite automata are given.

Automaton 1:
States: 0, 1, 2, 3, 4, 5, 6, 7
Alphabet: a, b, c
Start state: 3
Accepting states: 0
Transition table (row i, column labeled by a symbol, entry = next state):
   | a b c
---+-------------
  0 | 0 5 4
  1 | 7 7 4
  2 | 7 5 4
  3 | 1 1 4
  4 | 4 2 3
  5 | 0 5 7
  6 | 1 5 0
  7 | 4 1 2

Automaton 2:
States: 0, 1, 2, 3, 4, 5, 6, 7
Alphabet: a, b, c
Start state: 1
Accepting states: 0, 3, 5, 7
Transition table (row i, column labeled by a symbol, entry = next state):
   | a b c
---+-------------
  0 | 0 4 1
  1 | 7 2 2
  2 | 5 3 3
  3 | 6 6 0
  4 | 7 1 5
  5 | 0 1 4
  6 | 7 5 6
  7 | 4 3 2

Find the lexicographically smallest string of shortest length that is accepted by Automaton 1 and rejected by Automaton 2. If the two automata accept exactly the same set of strings and no such string exists, answer "none". The answer is the witness string, or "none".

**Answer:**

cbbaa

