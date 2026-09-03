# Samples wait_for_deadlock (seed 1977587174)


## Level 0

### Example

Prompt:

```
Transactions:
['T0', 'T1', 'T2', 'T3', 'T4', 'T5']

Wait Edges:
['T0 waits for T2', 'T1 waits for T3', 'T2 waits for T5', 'T3 waits for T0', 'T5 waits for T2', 'T5 waits for T4']

Some transactions may be stuck in a deadlock cycle: a cycle of transactions where each waits for a resource the next holds. A transaction participates in a deadlock if it lies on at least one directed cycle of wait_edges.

List the labels of all deadlocked transactions, comma-separated in numeric order (for example "T1,T3,T5"), or exactly "none" if no transaction is deadlocked.
```

Answer:

```
T2,T5
```

### Example

Prompt:

```
Transactions:
['T0', 'T1', 'T2', 'T3', 'T4', 'T5']

Wait Edges:
['T0 waits for T3', 'T2 waits for T1', 'T3 waits for T2', 'T4 waits for T0', 'T5 waits for T1', 'T5 waits for T4']

Some transactions may be stuck in a deadlock cycle: a cycle of transactions where each waits for a resource the next holds. A transaction participates in a deadlock if it lies on at least one directed cycle of wait_edges.

List the labels of all deadlocked transactions, comma-separated in numeric order (for example "T1,T3,T5"), or exactly "none" if no transaction is deadlocked.
```

Answer:

```
none
```


## Level 2

### Example

Prompt:

```
Transactions:
['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']

Wait Edges:
['T0 waits for T4', 'T1 waits for T4', 'T2 waits for T7', 'T3 waits for T6', 'T4 waits for T2', 'T5 waits for T1', 'T6 waits for T5', 'T7 waits for T0']

Some transactions may be stuck in a deadlock cycle: a cycle of transactions where each waits for a resource the next holds. A transaction participates in a deadlock if it lies on at least one directed cycle of wait_edges.

List the labels of all deadlocked transactions, comma-separated in numeric order (for example "T1,T3,T5"), or exactly "none" if no transaction is deadlocked.
```

Answer:

```
T0,T2,T4,T7
```

### Example

Prompt:

```
Transactions:
['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']

Wait Edges:
['T0 waits for T5', 'T1 waits for T2', 'T1 waits for T7', 'T2 waits for T4', 'T3 waits for T0', 'T5 waits for T2', 'T6 waits for T1', 'T7 waits for T3']

Some transactions may be stuck in a deadlock cycle: a cycle of transactions where each waits for a resource the next holds. A transaction participates in a deadlock if it lies on at least one directed cycle of wait_edges.

List the labels of all deadlocked transactions, comma-separated in numeric order (for example "T1,T3,T5"), or exactly "none" if no transaction is deadlocked.
```

Answer:

```
none
```


## Level 5

### Example

Prompt:

```
Transactions:
['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']

Wait Edges:
['T0 waits for T9', 'T1 waits for T2', 'T2 waits for T3', 'T3 waits for T5', 'T4 waits for T6', 'T4 waits for T7', 'T5 waits for T0', 'T6 waits for T10', 'T7 waits for T4', 'T8 waits for T2', 'T9 waits for T1', 'T10 waits for T8']

Some transactions may be stuck in a deadlock cycle: a cycle of transactions where each waits for a resource the next holds. A transaction participates in a deadlock if it lies on at least one directed cycle of wait_edges.

List the labels of all deadlocked transactions, comma-separated in numeric order (for example "T1,T3,T5"), or exactly "none" if no transaction is deadlocked.
```

Answer:

```
T0,T1,T2,T3,T4,T5,T7,T9
```

### Example

Prompt:

```
Transactions:
['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']

Wait Edges:
['T0 waits for T10', 'T1 waits for T9', 'T2 waits for T5', 'T3 waits for T1', 'T3 waits for T4', 'T3 waits for T9', 'T4 waits for T6', 'T5 waits for T4', 'T6 waits for T3', 'T7 waits for T2', 'T8 waits for T0', 'T10 waits for T7']

Some transactions may be stuck in a deadlock cycle: a cycle of transactions where each waits for a resource the next holds. A transaction participates in a deadlock if it lies on at least one directed cycle of wait_edges.

List the labels of all deadlocked transactions, comma-separated in numeric order (for example "T1,T3,T5"), or exactly "none" if no transaction is deadlocked.
```

Answer:

```
T3,T4,T6
```
