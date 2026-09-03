## Level 0

**Prompt:**

```
Attributes:
['A0', 'A1', 'A2', 'A3', 'A4', 'A5']

Functional Dependencies:
['A0 -> A2', 'A1 -> A3', 'A0 -> A4', 'A1 -> A5']

Set:
['A3', 'A5']

A superkey is a set of attributes whose closure contains all attributes; a candidate key is a minimal superkey. Classify the given attribute set with respect to the functional dependencies. The answer is exactly one of: 'candidate key', 'nonminimal superkey', 'non-superkey'.
```

**Answer:**

non-superkey

**Prompt:**

```
Attributes:
['A0', 'A1', 'A2', 'A3', 'A4', 'A5']

Functional Dependencies:
['A0 -> A2', 'A1 -> A3', 'A0 -> A4', 'A1 -> A5']

Set:
['A0', 'A1']

A superkey is a set of attributes whose closure contains all attributes; a candidate key is a minimal superkey. Classify the given attribute set with respect to the functional dependencies. The answer is exactly one of: 'candidate key', 'nonminimal superkey', 'non-superkey'.
```

**Answer:**

candidate key

## Level 2

**Prompt:**

```
Attributes:
['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']

Functional Dependencies:
['A0 -> A2', 'A1 -> A3', 'A0 -> A4', 'A1 -> A5', 'A0 -> A6', 'A1 -> A7']

Set:
['A2', 'A3', 'A4', 'A5', 'A6', 'A7']

A superkey is a set of attributes whose closure contains all attributes; a candidate key is a minimal superkey. Classify the given attribute set with respect to the functional dependencies. The answer is exactly one of: 'candidate key', 'nonminimal superkey', 'non-superkey'.
```

**Answer:**

non-superkey

**Prompt:**

```
Attributes:
['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']

Functional Dependencies:
['A0 -> A3', 'A1 -> A4', 'A2 -> A5', 'A0 -> A6', 'A1 -> A7']

Set:
['A0', 'A1', 'A2']

A superkey is a set of attributes whose closure contains all attributes; a candidate key is a minimal superkey. Classify the given attribute set with respect to the functional dependencies. The answer is exactly one of: 'candidate key', 'nonminimal superkey', 'non-superkey'.
```

**Answer:**

candidate key

## Level 5

**Prompt:**

```
Attributes:
['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']

Functional Dependencies:
['A0 -> A4', 'A1 -> A5', 'A2 -> A6', 'A3 -> A7', 'A0 -> A8', 'A1 -> A9', 'A2 -> A10']

Set:
['A10']

A superkey is a set of attributes whose closure contains all attributes; a candidate key is a minimal superkey. Classify the given attribute set with respect to the functional dependencies. The answer is exactly one of: 'candidate key', 'nonminimal superkey', 'non-superkey'.
```

**Answer:**

non-superkey

**Prompt:**

```
Attributes:
['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']

Functional Dependencies:
['A0 -> A2', 'A1 -> A3', 'A0 -> A4', 'A1 -> A5', 'A0 -> A6', 'A1 -> A7', 'A0 -> A8', 'A1 -> A9', 'A0 -> A10']

Set:
['A0', 'A1', 'A4']

A superkey is a set of attributes whose closure contains all attributes; a candidate key is a minimal superkey. Classify the given attribute set with respect to the functional dependencies. The answer is exactly one of: 'candidate key', 'nonminimal superkey', 'non-superkey'.
```

**Answer:**

nonminimal superkey
