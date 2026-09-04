## Level 0

### Example 1

**Prompt:**

```
A nondeterministic finite automaton (NFA) has states 0..4 and alphabet 0,1.
Its transition relation and epsilon transitions (moves possible without consuming a symbol) are:
From state 0: on 0 go to 4; epsilon to none.
From state 1: on 0 go to 2,4; epsilon to 4.
From state 2: on 0 go to 0,4; epsilon to none.
From state 3: on 0 go to 0,4; epsilon to 4.
From state 4: on 0 go to 2; on 1 go to 1; epsilon to 3.
The initial state is 0 and the accepting states are 1,3.
Apply the subset construction (with epsilon closure to take unreachable-at-epsilon states into account) to convert this NFA into a deterministic finite automaton (DFA) whose states are the reachable canonical subsets of NFA states. A DFA only has a transition when the source subset has at least one outgoing move on that symbol.
How many reachable states does the resulting DFA have?
The answer is the number of reachable DFA states, given as a single integer.
```

**Answer:**

```
4
```

### Example 2

**Prompt:**

```
A nondeterministic finite automaton (NFA) has states 0..4 and alphabet 0,1.
Its transition relation and epsilon transitions (moves possible without consuming a symbol) are:
From state 0: on 0 go to 4; on 1 go to 3; epsilon to none.
From state 1: on 1 go to 1,2; epsilon to 4.
From state 2: on 1 go to 2,3; epsilon to 4.
From state 3: on 0 go to 2; on 1 go to 2; epsilon to 4.
From state 4: on 0 go to 2; on 1 go to 0; epsilon to 0.
The initial state is 0 and the accepting states are 0,4.
Apply the subset construction (with epsilon closure to take unreachable-at-epsilon states into account) to convert this NFA into a deterministic finite automaton (DFA) whose states are the reachable canonical subsets of NFA states. A DFA only has a transition when the source subset has at least one outgoing move on that symbol.
How many reachable states does the resulting DFA have?
The answer is the number of reachable DFA states, given as a single integer.
```

**Answer:**

```
5
```

## Level 2

### Example 1

**Prompt:**

```
A nondeterministic finite automaton (NFA) has states 0..6 and alphabet 0,1.
Its transition relation and epsilon transitions (moves possible without consuming a symbol) are:
From state 0: on 0 go to 5,6; on 1 go to 1,6; epsilon to none.
From state 1: on 0 go to 3,4; on 1 go to 1,3; epsilon to 2.
From state 2: on 0 go to 0,5; on 1 go to 3,5; epsilon to none.
From state 3: on 0 go to 0,1; on 1 go to 2,6; epsilon to none.
From state 4: on 0 go to 0,6; on 1 go to 0,4; epsilon to 3,6.
From state 5: on 0 go to 1; on 1 go to 1,4,5; epsilon to 2,3,6.
From state 6: on 0 go to 3,4; on 1 go to 4,5; epsilon to 3,4.
The initial state is 0 and the accepting states are 1,4,6.
Apply the subset construction (with epsilon closure to take unreachable-at-epsilon states into account) to convert this NFA into a deterministic finite automaton (DFA) whose states are the reachable canonical subsets of NFA states. A DFA only has a transition when the source subset has at least one outgoing move on that symbol.
How many reachable states does the resulting DFA have?
The answer is the number of reachable DFA states, given as a single integer.
```

**Answer:**

```
4
```

### Example 2

**Prompt:**

```
A nondeterministic finite automaton (NFA) has states 0..6 and alphabet 0,1.
Its transition relation and epsilon transitions (moves possible without consuming a symbol) are:
From state 0: on 0 go to 6; on 1 go to 0,3; epsilon to none.
From state 1: on 0 go to 2,4; on 1 go to 0,2; epsilon to none.
From state 2: on 0 go to 5,6; on 1 go to 3,6; epsilon to 1,5,6.
From state 3: on 0 go to 3,6; on 1 go to 6; epsilon to 1.
From state 4: on 0 go to 0; on 1 go to 0,6; epsilon to 0,5.
From state 5: on 0 go to 4,6; on 1 go to 1,3; epsilon to 3.
From state 6: on 0 go to 2,3,4; on 1 go to 0; epsilon to 1,3,5.
The initial state is 0 and the accepting states are 0,1,2.
Apply the subset construction (with epsilon closure to take unreachable-at-epsilon states into account) to convert this NFA into a deterministic finite automaton (DFA) whose states are the reachable canonical subsets of NFA states. A DFA only has a transition when the source subset has at least one outgoing move on that symbol.
How many reachable states does the resulting DFA have?
The answer is the number of reachable DFA states, given as a single integer.
```

**Answer:**

```
5
```

## Level 5

### Example 1

**Prompt:**

```
A nondeterministic finite automaton (NFA) has states 0..9 and alphabet 0,1,2.
Its transition relation and epsilon transitions (moves possible without consuming a symbol) are:
From state 0: on 0 go to 1,7; on 1 go to 1,5,6; on 2 go to 8; epsilon to none.
From state 1: on 1 go to 1,3,4,5,6; on 2 go to 1,8; epsilon to 3,4,9.
From state 2: on 0 go to 5,9; on 1 go to 6; on 2 go to 2,8,9; epsilon to 0.
From state 3: on 0 go to 0,3,9; on 1 go to 3,9; on 2 go to 7; epsilon to none.
From state 4: on 0 go to 1,2,9; on 1 go to 3,6; on 2 go to 2,6; epsilon to none.
From state 5: on 0 go to 2,6,8; on 1 go to 3,8; on 2 go to 0,8; epsilon to none.
From state 6: on 0 go to 0,6,7; on 1 go to 0,2; on 2 go to 6,9; epsilon to 5.
From state 7: on 0 go to 0,1,5; on 1 go to 0,3; epsilon to 1.
From state 8: on 0 go to 1,3,9; on 1 go to 3,9; on 2 go to 5; epsilon to none.
From state 9: on 0 go to 3,7; on 1 go to 3,4; on 2 go to 5,7,9; epsilon to 4,6,8.
The initial state is 0 and the accepting states are 1,4,5,7,9.
Apply the subset construction (with epsilon closure to take unreachable-at-epsilon states into account) to convert this NFA into a deterministic finite automaton (DFA) whose states are the reachable canonical subsets of NFA states. A DFA only has a transition when the source subset has at least one outgoing move on that symbol.
How many reachable states does the resulting DFA have?
The answer is the number of reachable DFA states, given as a single integer.
```

**Answer:**

```
16
```

### Example 2

**Prompt:**

```
A nondeterministic finite automaton (NFA) has states 0..9 and alphabet 0,1,2.
Its transition relation and epsilon transitions (moves possible without consuming a symbol) are:
From state 0: on 0 go to 4,6,8,9; on 1 go to 5; on 2 go to 7,8; epsilon to none.
From state 1: on 0 go to 2,4,6,7; on 1 go to 0; on 2 go to 4,7; epsilon to 7.
From state 2: on 0 go to 0,1; on 1 go to 1; on 2 go to 3,4,7,9; epsilon to 8,9.
From state 3: on 0 go to 8; on 1 go to 2,4,6; on 2 go to 4,7; epsilon to 1.
From state 4: on 1 go to 3; on 2 go to 0,1,4,5,7,8; epsilon to none.
From state 5: on 0 go to 4; on 1 go to 4,5; on 2 go to 5,6,8; epsilon to 1,7.
From state 6: on 0 go to 0,3,5,7,8; on 1 go to 0,1; epsilon to 1,5.
From state 7: on 0 go to 0,1,8; on 1 go to 0; on 2 go to 0,9; epsilon to 6.
From state 8: on 0 go to 4,9; on 1 go to 1,3,8; epsilon to 3.
From state 9: on 0 go to 7,9; on 1 go to 0,6; on 2 go to 1; epsilon to 0.
The initial state is 0 and the accepting states are 2,6,7,8,9.
Apply the subset construction (with epsilon closure to take unreachable-at-epsilon states into account) to convert this NFA into a deterministic finite automaton (DFA) whose states are the reachable canonical subsets of NFA states. A DFA only has a transition when the source subset has at least one outgoing move on that symbol.
How many reachable states does the resulting DFA have?
The answer is the number of reachable DFA states, given as a single integer.
```

**Answer:**

```
7
```
