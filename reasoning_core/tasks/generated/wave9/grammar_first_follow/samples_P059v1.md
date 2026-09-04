# Level 0

## Example 1

### Prompt

Nonterminals:
N0, N1, N2, N3, N4

Terminals:
a, b, c

Start:
N0

Productions:
N0 -> N1 b N1 b
N0 -> N3 c
N0 -> N2 N4
N0 -> N1 a N3 b
N1 -> N0 N3
N1 -> N1 N0 a
N2 -> N4
N2 -> N2 c N0
N2 -> c N4 N4 N4
N3 -> N3 N4 N2 N0
N3 -> N4 N0 b b
N4 -> b
N4 -> N3 N3 N3 N3
N4 -> N2 N2

Compute FIRST(N3).

The answer is the canonical FIRST set: distinct symbols that can start a string derived from N3, sorted, entries separated by spaces, using 'e' to denote the empty string epsilon.

### Answer

b c

## Example 2

### Prompt

Nonterminals:
N0, N1, N2, N3, N4

Terminals:
a, b, c

Start:
N0

Productions:
N0 -> b c
N1 -> N1 b N1
N1 -> c c N0 a
N2 -> N4 N3
N3 -> N2 b N2 b
N3 -> N1 N3 N0
N3 -> N4 N0 N2 N3
N4 -> N2 N3
N4 -> N2 a N4 c
N4 -> N1 N4 N1 N4

Compute FOLLOW(N1).

The answer is the canonical FOLLOW set: distinct symbols that can immediately follow N1 in a sentential form, sorted, entries separated by spaces, using '$' to denote the end-of-input marker; an empty set is written 'empty'.

### Answer

b c



# Level 2

## Example 1

### Prompt

Nonterminals:
N0, N1, N2, N3, N4, N5, N6

Terminals:
a, b, c, d

Start:
N0

Productions:
N0 -> a
N0 -> N1 N6
N0 -> N3 N4
N1 -> N4 N6 c N0
N2 -> N4 N5 N6 N0
N2 -> N4 N5 a c
N3 -> N6 N5 b b
N3 -> N4 N1
N4 -> N4 N0 d a
N4 -> N6 N4
N4 -> N1 N4 N5 N6
N5 -> d N4
N5 -> a d N6
N6 -> N3 a N0
N6 -> N2 N4
N6 -> N5 N0
N6 -> c

Compute FIRST(N2).

The answer is the canonical FIRST set: distinct symbols that can start a string derived from N2, sorted, entries separated by spaces, using 'e' to denote the empty string epsilon.

### Answer

a c d

## Example 2

### Prompt

Nonterminals:
N0, N1, N2, N3, N4, N5, N6

Terminals:
a, b, c, d

Start:
N0

Productions:
N0 -> N0 d b N4
N0 -> N1 a N3 a N5 b
N1 -> N2 N6
N2 -> N4 N6 N2 b
N3 -> c N5 c
N3 -> b N5 N5
N4 -> N4 a N4
N4 -> N5 N1 N0 N2
N5 -> N3
N5 -> c N3 a N2
N5 -> N3 N6 N1
N5 -> b N5 c
N6 -> N0 d
N6 -> N3 N4 N5 N2

Compute FIRST(N3).

The answer is the canonical FIRST set: distinct symbols that can start a string derived from N3, sorted, entries separated by spaces, using 'e' to denote the empty string epsilon.

### Answer

b c



# Level 5

## Example 1

### Prompt

Nonterminals:
N0, N1, N2, N3, N4, N5, N6, N7, N8, N9

Terminals:
a, b, c, d, e

Start:
N0

Productions:
N0 -> a N7 N7
N0 -> N1 N3
N0 -> N2 N9 N6
N0 -> N6 a N9 N0 N9 N6
N0 -> N2 N0
N0 -> N1 e N3 b N5 c N7 e N9 b
N1 -> e N7 d
N1 -> N1 N3
N1 -> N1 N7 N3 a N0 N0
N1 -> N9 N6 e
N1 -> N9 a
N2 -> a N7 N5
N2 -> N9 d N2
N2 -> N4 N9 N7 N5
N3 -> N1 N7 a a d N2
N3 -> N1 N9 a N4 N8 a
N3 -> c N9 b N1 N0
N4 -> N5 N3 N3 N6 e c
N4 -> N9 c
N4 -> N6
N4 -> e N6 a N9 N4
N4 -> N3 a
N5 -> N9 d c d
N5 -> c e N4
N5 -> N0 N2 N1 N6 N9
N6 -> N2
N6 -> N9 N3 d
N6 -> N0 N4
N7 -> N5 N3
N7 -> N3 N5 N9 N6 N0
N7 -> N3 c N9
N7 -> N4 N1 N8
N7 -> e
N8 -> e N1 N1 N4 N2 N7
N8 -> N9 N2 N1 a
N8 -> N0 N6 N6 N5
N8 -> d
N8 -> N3 c c d
N9 -> N7 N1
N9 -> b d b
N9 -> N9 N0 N5 N1 N8
N9 -> a N9
N9 -> N5 a N2 N0

Compute FOLLOW(N7).

The answer is the canonical FOLLOW set: distinct symbols that can immediately follow N7 in a sentential form, sorted, entries separated by spaces, using '$' to denote the end-of-input marker; an empty set is written 'empty'.

### Answer

$ a b c d

## Example 2

### Prompt

Nonterminals:
N0, N1, N2, N3, N4, N5, N6, N7, N8, N9

Terminals:
a, b, c, d, e

Start:
N0

Productions:
N0 -> N9 N5 N7
N0 -> N4 c
N0 -> N9 N0
N0 -> N2 N9 N4 N5 N1
N1 -> N0 N2 N8
N1 -> N0 N9 N0
N1 -> N3 N7 N4
N2 -> a N5 N6 c e
N2 -> N4 N4
N2 -> N6
N2 -> c N4 N7 N5 a
N3 -> a N5 N7 N0 N3 N6
N3 -> N4 N7 d
N4 -> d
N5 -> b
N5 -> c N2 d
N5 -> N9
N6 -> N5 N7
N6 -> N0 N9 N2 N6 N8 N5
N6 -> N6 N7 N9 N0 N1 N3
N6 -> N5 N4 N1
N7 -> N3 N4 N5
N7 -> N8 N8
N7 -> a N7 N3
N7 -> c d N4 N5 a
N7 -> e N9 d
N8 -> e N6 d N4 N8
N9 -> N6

Compute FIRST(N8).

The answer is the canonical FIRST set: distinct symbols that can start a string derived from N8, sorted, entries separated by spaces, using 'e' to denote the empty string epsilon.

### Answer

a b c d


