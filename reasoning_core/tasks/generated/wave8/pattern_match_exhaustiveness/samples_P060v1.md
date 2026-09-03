# samples_P060v1

Pattern-match exhaustiveness: decide whether all constructors are covered,
naming the smallest uncovered constructor as a counterexample.

## Level 0

### Example 1

Consider an algebraic datatype whose constructors are: c0, c1, c3, c2.
The following pattern-match clauses together cover constructors:
- clause 1: covers c2
- clause 2: covers c1
Determine how many constructors are NOT covered by any clause.
Answer with that number alone (for example "2"). It is 0 exactly when every constructor is covered.

**Answer:** 2

### Example 2

Consider an algebraic datatype whose constructors are: c3, c5, c0, c2, c1, c4.
The following pattern-match clauses together cover constructors:
- clause 1: covers (no clause matches any constructor)
Determine how many constructors are NOT covered by any clause.
Answer with that number alone (for example "2"). It is 0 exactly when every constructor is covered.

**Answer:** 6

## Level 2

### Example 1

Consider an algebraic datatype whose constructors are: c2, c3, c0, c1, c4, c5.
The following pattern-match clauses together cover constructors:
- clause 1: covers (no clause matches any constructor)
Determine how many constructors are NOT covered by any clause.
Answer with that number alone (for example "2"). It is 0 exactly when every constructor is covered.

**Answer:** 6

### Example 2

Consider an algebraic datatype whose constructors are: c3, c5, c2, c4, c0, c1.
The following pattern-match clauses together cover constructors:
- clause 1: covers c1
- clause 2: covers c0
- clause 3: covers c4
Determine how many constructors are NOT covered by any clause.
Answer with that number alone (for example "2"). It is 0 exactly when every constructor is covered.

**Answer:** 3

## Level 5

### Example 1

Consider an algebraic datatype whose constructors are: c9, c5, c8, c3, c1, c7, c6, c2, c10, c4, c0.
The following pattern-match clauses together cover constructors:
- clause 1: covers (no clause matches any constructor)
Determine how many constructors are NOT covered by any clause.
Answer with that number alone (for example "2"). It is 0 exactly when every constructor is covered.

**Answer:** 11

### Example 2

Consider an algebraic datatype whose constructors are: c0, c8, c4, c7, c5, c1, c3, c9, c2, c6.
The following pattern-match clauses together cover constructors:
- clause 1: covers c8
- clause 2: covers c9
- clause 3: covers c7
- clause 4: covers c5
- clause 5: covers c2
Determine how many constructors are NOT covered by any clause.
Answer with that number alone (for example "2"). It is 0 exactly when every constructor is covered.

**Answer:** 5

