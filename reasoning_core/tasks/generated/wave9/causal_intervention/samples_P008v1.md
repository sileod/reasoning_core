## Level 0

### Example 1

**Prompt:**

```
Consider a structural causal model over nodes X_0, X_1, X_2, X_3, ordered topologically (a parent always has a smaller index than its child).
Each node X_i follows the structural equation
X_i = c_i + sum_{j in pa(i)} beta_ij * X_j + eps_i,  eps_i ~ N(0, 1.0) independent.
The structural equations are:
X_0 = 1.0 + eps_0
X_1 = -2.0 + eps_1
X_2 = 1.0 + -0.5*X_0 + eps_2
X_3 = 1.0 + -0.5*X_0 + -1.0*X_1 + eps_3
A do-intervention sets some nodes to constants and deletes their incoming edges:
do(X_0 = 2.0)

Query A1: compute the value of X_3 after the intervention.
Query A2: compute the value of X_2 after the intervention.
Query B3: compute the probability that X_1 > 1.0 after the intervention (a number between 0 and 1).

The answer is the list of values in query order, separated by spaces.
```

**Answer:**

```
2.0 0.0 0.0013
```

### Example 2

**Prompt:**

```
Consider a structural causal model over nodes X_0, X_1, X_2, X_3, ordered topologically (a parent always has a smaller index than its child).
Each node X_i follows the structural equation
X_i = c_i + sum_{j in pa(i)} beta_ij * X_j + eps_i,  eps_i ~ N(0, 1.0) independent.
The structural equations are:
X_0 = 2.0 + eps_0
X_1 = 1.0 + eps_1
X_2 = -2.0 + eps_2
X_3 = -1.0 + eps_3
A do-intervention sets some nodes to constants and deletes their incoming edges:
do(X_0 = 3.0)

Query A1: compute the value of X_2 after the intervention.
Query A2: compute the value of X_3 after the intervention.
Query B3: compute the probability that X_3 > 1.0 after the intervention (a number between 0 and 1).

The answer is the list of values in query order, separated by spaces.
```

**Answer:**

```
-2.0 -1.0 0.0228
```

## Level 2

### Example 1

**Prompt:**

```
Consider a structural causal model over nodes X_0, X_1, X_2, X_3, X_4, X_5, ordered topologically (a parent always has a smaller index than its child).
Each node X_i follows the structural equation
X_i = c_i + sum_{j in pa(i)} beta_ij * X_j + eps_i,  eps_i ~ N(0, 1.0) independent.
The structural equations are:
X_0 = -2.0 + eps_0
X_1 = 2.0 + 0.3*X_0 + eps_1
X_2 = 0.0 + 1.0*X_0 + -0.5*X_1 + eps_2
X_3 = -2.0 + eps_3
X_4 = 2.0 + -0.5*X_1 + 0.3*X_3 + eps_4
X_5 = -2.0 + -0.5*X_0 + 2.0*X_3 + eps_5
A do-intervention sets some nodes to constants and deletes their incoming edges:
do(X_3 = 2.0)
do(X_5 = -1.0)

Query A1: compute the value of X_2 after the intervention.
Query A2: compute the value of X_1 after the intervention.
Query B3: compute the probability that X_2 > 2.0 after the intervention (a number between 0 and 1).

The answer is the list of values in query order, separated by spaces.
```

**Answer:**

```
-2.7 1.4 0.0004
```

### Example 2

**Prompt:**

```
Consider a structural causal model over nodes X_0, X_1, X_2, X_3, X_4, X_5, ordered topologically (a parent always has a smaller index than its child).
Each node X_i follows the structural equation
X_i = c_i + sum_{j in pa(i)} beta_ij * X_j + eps_i,  eps_i ~ N(0, 1.0) independent.
The structural equations are:
X_0 = 1.0 + eps_0
X_1 = -2.0 + eps_1
X_2 = -2.0 + 1.0*X_0 + 1.0*X_1 + eps_2
X_3 = -2.0 + 1.0*X_0 + -0.5*X_2 + eps_3
X_4 = 1.0 + 0.3*X_0 + eps_4
X_5 = 1.0 + -1.0*X_0 + -0.5*X_4 + eps_5
A do-intervention sets some nodes to constants and deletes their incoming edges:
do(X_0 = 0.0)
do(X_2 = 1.0)

Query A1: compute the value of X_4 after the intervention.
Query A2: compute the value of X_3 after the intervention.
Query B3: compute the probability that X_1 > 0.0 after the intervention (a number between 0 and 1).

The answer is the list of values in query order, separated by spaces.
```

**Answer:**

```
1.0 -2.5 0.0228
```

## Level 5

### Example 1

**Prompt:**

```
Consider a structural causal model over nodes X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, ordered topologically (a parent always has a smaller index than its child).
Each node X_i follows the structural equation
X_i = c_i + sum_{j in pa(i)} beta_ij * X_j + eps_i,  eps_i ~ N(0, 1.0) independent.
The structural equations are:
X_0 = -2.0 + eps_0
X_1 = -1.0 + 1.0*X_0 + eps_1
X_2 = 2.0 + eps_2
X_3 = 2.0 + -0.5*X_1 + eps_3
X_4 = -2.0 + 1.0*X_0 + 1.0*X_1 + 2.0*X_2 + eps_4
X_5 = -1.0 + -0.5*X_0 + 2.0*X_4 + eps_5
X_6 = -2.0 + eps_6
X_7 = 2.0 + eps_7
X_8 = 1.0 + eps_8
A do-intervention sets some nodes to constants and deletes their incoming edges:
do(X_0 = 2.0)
do(X_4 = 1.0)
do(X_7 = 0.0)

Query A1: compute the value of X_3 after the intervention.
Query A2: compute the value of X_8 after the intervention.
Query A3: compute the value of X_1 after the intervention.
Query B4: compute the probability that X_6 > 0.0 after the intervention (a number between 0 and 1).

The answer is the list of values in query order, separated by spaces.
```

**Answer:**

```
1.5 1.0 1.0 0.0228
```

### Example 2

**Prompt:**

```
Consider a structural causal model over nodes X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, ordered topologically (a parent always has a smaller index than its child).
Each node X_i follows the structural equation
X_i = c_i + sum_{j in pa(i)} beta_ij * X_j + eps_i,  eps_i ~ N(0, 1.0) independent.
The structural equations are:
X_0 = 0.0 + eps_0
X_1 = -2.0 + 0.3*X_0 + eps_1
X_2 = 1.0 + 1.0*X_0 + eps_2
X_3 = 1.0 + 1.0*X_0 + -0.5*X_1 + 1.0*X_2 + eps_3
X_4 = 0.0 + eps_4
X_5 = 2.0 + 1.0*X_4 + eps_5
X_6 = -1.0 + 1.0*X_0 + 1.0*X_5 + eps_6
X_7 = -1.0 + eps_7
X_8 = 2.0 + -1.0*X_4 + -0.5*X_6 + 2.0*X_7 + eps_8
A do-intervention sets some nodes to constants and deletes their incoming edges:
do(X_0 = 0.0)
do(X_6 = -2.0)
do(X_7 = 0.0)

Query A1: compute the value of X_8 after the intervention.
Query A2: compute the value of X_5 after the intervention.
Query A3: compute the value of X_2 after the intervention.
Query B4: compute the probability that X_5 > 0.0 after the intervention (a number between 0 and 1).

The answer is the list of values in query order, separated by spaces.
```

**Answer:**

```
3.0 2.0 1.0 0.9214
```
