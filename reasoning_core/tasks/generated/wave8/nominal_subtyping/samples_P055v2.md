# Nominal Subtyping v2 samples

## Level 0

### Example 1

**Prompt:**

Here is a nominal type hierarchy. Each line lists a type and the types it directly inherits from. A type X is a subtype of type Y exactly when X is Y itself or X inherits, possibly over several steps, from Y.
The hierarchy:
 - T0 inherits from: (none)
 - T1 inherits from: (none)
 - T2 inherits from: T1, T3, T5
 - T3 inherits from: T0, T4
 - T4 inherits from: T5
 - T5 inherits from: T0, T1, T2

Is T4 a subtype of T3?
If yes, answer YES followed by a colon and the chain of type names from T4 down to T3, separated by commas.
If no, answer NO followed by a colon and the full list of every type T4 inherits from, in sorted order, separated by commas.

**Answer:**

YES: T4, T5, T2, T3

### Example 2

**Prompt:**

Here is a nominal type hierarchy. Each line lists a type and the types it directly inherits from. A type X is a subtype of type Y exactly when X is Y itself or X inherits, possibly over several steps, from Y.
The hierarchy:
 - T0 inherits from: T2
 - T1 inherits from: T2, T3, T5
 - T2 inherits from: T0, T1, T3, T4
 - T3 inherits from: (none)
 - T4 inherits from: T0, T1, T2, T5
 - T5 inherits from: (none)

Is T2 a subtype of T2?
If yes, answer YES followed by a colon and the chain of type names from T2 down to T2, separated by commas.
If no, answer NO followed by a colon and the full list of every type T2 inherits from, in sorted order, separated by commas.

**Answer:**

YES: T2

## Level 2

### Example 1

**Prompt:**

Here is a nominal type hierarchy. Each line lists a type and the types it directly inherits from. A type X is a subtype of type Y exactly when X is Y itself or X inherits, possibly over several steps, from Y.
The hierarchy:
 - T0 inherits from: T8
 - T1 inherits from: T0, T7
 - T2 inherits from: (none)
 - T3 inherits from: T0, T5, T6, T8
 - T4 inherits from: T0
 - T5 inherits from: T0, T2, T6
 - T6 inherits from: T1, T4, T5, T9
 - T7 inherits from: T0
 - T8 inherits from: T3, T7, T9
 - T9 inherits from: (none)

Is T1 a subtype of T6?
If yes, answer YES followed by a colon and the chain of type names from T1 down to T6, separated by commas.
If no, answer NO followed by a colon and the full list of every type T1 inherits from, in sorted order, separated by commas.

**Answer:**

YES: T1, T0, T8, T3, T6

### Example 2

**Prompt:**

Here is a nominal type hierarchy. Each line lists a type and the types it directly inherits from. A type X is a subtype of type Y exactly when X is Y itself or X inherits, possibly over several steps, from Y.
The hierarchy:
 - T0 inherits from: T5, T9
 - T1 inherits from: T3, T6, T8, T9
 - T2 inherits from: (none)
 - T3 inherits from: T0, T1, T4, T8
 - T4 inherits from: T0, T2, T8, T9
 - T5 inherits from: T6, T8
 - T6 inherits from: (none)
 - T7 inherits from: T6, T9
 - T8 inherits from: T0, T3, T5
 - T9 inherits from: T1, T7, T8

Is T4 a subtype of T8?
If yes, answer YES followed by a colon and the chain of type names from T4 down to T8, separated by commas.
If no, answer NO followed by a colon and the full list of every type T4 inherits from, in sorted order, separated by commas.

**Answer:**

YES: T4, T8

## Level 5

### Example 1

**Prompt:**

Here is a nominal type hierarchy. Each line lists a type and the types it directly inherits from. A type X is a subtype of type Y exactly when X is Y itself or X inherits, possibly over several steps, from Y.
The hierarchy:
 - T0 inherits from: T7
 - T1 inherits from: (none)
 - T10 inherits from: T4
 - T11 inherits from: T10, T3
 - T12 inherits from: T8
 - T13 inherits from: T4, T5
 - T14 inherits from: T0, T15, T3, T6
 - T15 inherits from: T0, T13, T8
 - T2 inherits from: T3
 - T3 inherits from: T0, T11
 - T4 inherits from: T1, T11, T9
 - T5 inherits from: T12, T15, T9
 - T6 inherits from: T10, T13, T7
 - T7 inherits from: T14, T2, T6, T9
 - T8 inherits from: T9
 - T9 inherits from: T0

Is T8 a subtype of T9?
If yes, answer YES followed by a colon and the chain of type names from T8 down to T9, separated by commas.
If no, answer NO followed by a colon and the full list of every type T8 inherits from, in sorted order, separated by commas.

**Answer:**

YES: T8, T9

### Example 2

**Prompt:**

Here is a nominal type hierarchy. Each line lists a type and the types it directly inherits from. A type X is a subtype of type Y exactly when X is Y itself or X inherits, possibly over several steps, from Y.
The hierarchy:
 - T0 inherits from: (none)
 - T1 inherits from: T12, T5
 - T10 inherits from: T0, T1, T13
 - T11 inherits from: T7, T9
 - T12 inherits from: T2
 - T13 inherits from: T15, T3, T4, T7
 - T14 inherits from: T0, T1, T10, T6
 - T15 inherits from: T1, T10, T14
 - T2 inherits from: T14, T8
 - T3 inherits from: T6
 - T4 inherits from: T1, T14, T7, T9
 - T5 inherits from: T15, T6, T7, T8
 - T6 inherits from: T11, T15, T7
 - T7 inherits from: T5
 - T8 inherits from: T12, T13, T14, T6
 - T9 inherits from: T13, T4

Is T10 a subtype of T5?
If yes, answer YES followed by a colon and the chain of type names from T10 down to T5, separated by commas.
If no, answer NO followed by a colon and the full list of every type T10 inherits from, in sorted order, separated by commas.

**Answer:**

YES: T10, T1, T5
