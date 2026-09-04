# query_plan_execution samples (P040v1)

## Level 0

### Example 1

Prompt:

```
Relation R (Columns A B):
2 0
3 0
3 0

Relation S (Columns B C):
0 0
1 1

Plan:
1. Scan both relations and inner-join them on column b.
2. Filter (σ) rows where c % 2 == 0.
3. Project (π) onto the needed columns, then group the joined rows by c and count the rows in each group, then sort by c.

Execute the relational-algebra plan. The answer is the exact projected result: pairs 'c:value' separated by ', ' sorted by c (for count/sum) or the c values themselves separated by ', ' (for distinct projection).
```

Answer:

```
0:3
```

### Example 2

Prompt:

```
Relation R (Columns A B):
3 1
3 1
2 1

Relation S (Columns B C):
0 1
1 0

Plan:
1. Scan both relations and inner-join them on column b.
2. Filter (σ) rows where a >= 1.
3. Project (π) onto the needed columns, then group the joined rows by c and sum the a column within each group, then sort by c.

Execute the relational-algebra plan. The answer is the exact projected result: pairs 'c:value' separated by ', ' sorted by c (for count/sum) or the c values themselves separated by ', ' (for distinct projection).
```

Answer:

```
0:8
```

## Level 2

### Example 1

Prompt:

```
Relation R (Columns A B):
1 3
2 1
0 1
4 3

Relation S (Columns B C):
0 2
1 1
1 1

Plan:
1. Scan both relations and inner-join them on column b.
2. Filter (σ) rows where a >= 0.
3. Project (π) onto the needed columns, then project onto column c, drop duplicates, and sort ascending.

Execute the relational-algebra plan. The answer is the exact projected result: pairs 'c:value' separated by ', ' sorted by c (for count/sum) or the c values themselves separated by ', ' (for distinct projection).
```

Answer:

```
1
```

### Example 2

Prompt:

```
Relation R (Columns A B):
4 2
4 1
1 1
3 1

Relation S (Columns B C):
0 2
1 0
1 1

Plan:
1. Scan both relations and inner-join them on column b.
2. Filter (σ) rows where a == 3.
3. Project (π) onto the needed columns, then group the joined rows by c and sum the a column within each group, then sort by c.

Execute the relational-algebra plan. The answer is the exact projected result: pairs 'c:value' separated by ', ' sorted by c (for count/sum) or the c values themselves separated by ', ' (for distinct projection).
```

Answer:

```
0:3, 1:3
```

## Level 5

### Example 1

Prompt:

```
Relation R (Columns A B):
4 0
6 4
7 3
1 2
7 2
2 0

Relation S (Columns B C):
2 0
3 1
0 2
3 0

Plan:
1. Scan both relations and inner-join them on column b.
2. Filter (σ) rows where c % 2 == 0.
3. Project (π) onto the needed columns, then group the joined rows by c and sum the a column within each group, then sort by c.

Execute the relational-algebra plan. The answer is the exact projected result: pairs 'c:value' separated by ', ' sorted by c (for count/sum) or the c values themselves separated by ', ' (for distinct projection).
```

Answer:

```
0:15, 2:6
```

### Example 2

Prompt:

```
Relation R (Columns A B):
3 4
5 2
6 4
2 1
7 3
1 3

Relation S (Columns B C):
2 1
1 3
1 2
3 4

Plan:
1. Scan both relations and inner-join them on column b.
2. Filter (σ) rows where a >= 7.
3. Project (π) onto the needed columns, then group the joined rows by c and sum the a column within each group, then sort by c.

Execute the relational-algebra plan. The answer is the exact projected result: pairs 'c:value' separated by ', ' sorted by c (for count/sum) or the c values themselves separated by ', ' (for distinct projection).
```

Answer:

```
4:7
```
