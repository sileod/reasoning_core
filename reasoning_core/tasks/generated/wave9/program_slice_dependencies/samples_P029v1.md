## Level 0
### Example 1
Program:
0: v0 = scale(1)
1: v1 = merge(v0, 6)
2: v2 = combine(v0)
3: v3 = merge(v0, v2, 9)
4: v4 = scale(v1, v2, 9)
5: v5 = fold(v2)

Query:
Final output value of v0

Trace the data and control dependencies backward from the final output and report the minimal set of statements that must be kept to compute that final value. Here 0-indexed statement numbers are shown at the start of each line; a line '3: if v2: v3 = merge(v1, 4)' means statement 3 depends on statement 1 (data) and on statement 2 (control, its guard). List the kept statement numbers in increasing order, separated by spaces (for example: '0 2 5'). The answer is exactly that list.
**Answer:** 0

### Example 2
Program:
0: v0 = combine()
1: v1 = combine(v0)
2: if v0: v2 = combine(v1)
3: v3 = scale(v0, v1)
4: if v3: v4 = merge(v1, v2)
5: v5 = combine(v0, v4)

Query:
Final output value of v1

Trace the data and control dependencies backward from the final output and report the minimal set of statements that must be kept to compute that final value. Here 0-indexed statement numbers are shown at the start of each line; a line '3: if v2: v3 = merge(v1, 4)' means statement 3 depends on statement 1 (data) and on statement 2 (control, its guard). List the kept statement numbers in increasing order, separated by spaces (for example: '0 2 5'). The answer is exactly that list.
**Answer:** 0 1

## Level 2
### Example 1
Program:
0: v0 = merge(1)
1: v1 = fold(v0)
2: v2 = combine(v0, v1, 4)
3: v3 = combine(v1)
4: if v3: v4 = merge(v1, 1)
5: v5 = merge(v3, v4)
6: v6 = scale(v2, v5)
7: v7 = merge(v4, v5, 2)
8: v8 = scale(v2, v4)
9: if v5: v9 = fold(v1, 6)

Query:
Final output value of v0

Trace the data and control dependencies backward from the final output and report the minimal set of statements that must be kept to compute that final value. Here 0-indexed statement numbers are shown at the start of each line; a line '3: if v2: v3 = merge(v1, 4)' means statement 3 depends on statement 1 (data) and on statement 2 (control, its guard). List the kept statement numbers in increasing order, separated by spaces (for example: '0 2 5'). The answer is exactly that list.
**Answer:** 0

### Example 2
Program:
0: v0 = combine()
1: v1 = scale(v0)
2: v2 = fold(v0, v1)
3: if v0: v3 = scale(v2)
4: if v0: v4 = merge(v2, v3)
5: if v3: v5 = fold(v4)
6: v6 = fold(v0, v4)
7: v7 = scale(v0, 6)
8: if v6: v8 = combine(v0, v7)
9: v9 = scale(v6, v8, 9)

Query:
Final output value of v9

Trace the data and control dependencies backward from the final output and report the minimal set of statements that must be kept to compute that final value. Here 0-indexed statement numbers are shown at the start of each line; a line '3: if v2: v3 = merge(v1, 4)' means statement 3 depends on statement 1 (data) and on statement 2 (control, its guard). List the kept statement numbers in increasing order, separated by spaces (for example: '0 2 5'). The answer is exactly that list.
**Answer:** 0 1 2 3 4 6 7 8 9

## Level 5
### Example 1
Program:
0: v0 = combine()
1: v1 = fold(v0)
2: v2 = fold(v0, v1)
3: v3 = fold(v0)
4: if v2: v4 = combine(v0)
5: if v0: v5 = combine(v3, v4)
6: if v2: v6 = fold(v1, v3)
7: if v2: v7 = fold(v1, 8)
8: if v1: v8 = fold(v3, v7)
9: if v2: v9 = merge(v0)
10: if v1: v10 = merge(v4, v9)
11: v11 = combine(v2, 7)
12: v12 = combine(v4)
13: v13 = fold(v1, v4)
14: if v3: v14 = fold(v5, 5)
15: if v11: v15 = scale(v9, 8)
16: v16 = scale(v3, v11)
17: v17 = merge(v5, v12, 4)
18: if v14: v18 = merge(v6, v8)
19: v19 = scale(v11, 5)

Query:
Final output value of v4

Trace the data and control dependencies backward from the final output and report the minimal set of statements that must be kept to compute that final value. Here 0-indexed statement numbers are shown at the start of each line; a line '3: if v2: v3 = merge(v1, 4)' means statement 3 depends on statement 1 (data) and on statement 2 (control, its guard). List the kept statement numbers in increasing order, separated by spaces (for example: '0 2 5'). The answer is exactly that list.
**Answer:** 0 1 2 4

### Example 2
Program:
0: v0 = fold(9)
1: v1 = fold(v0)
2: v2 = scale(v0, v1)
3: v3 = scale(v1)
4: v4 = scale(v3, 6)
5: if v4: v5 = merge(v0)
6: v6 = combine(v1)
7: v7 = fold(v0, v5, 5)
8: if v3: v8 = scale(v4, v7)
9: v9 = merge(v0, v4)
10: v10 = scale(v4, 7)
11: v11 = scale(v9)
12: if v2: v12 = fold(v0)
13: v13 = fold(v4)
14: if v10: v14 = fold(v3)
15: if v4: v15 = scale(v8)
16: if v13: v16 = scale(v12)
17: if v4: v17 = merge(v13, 2)
18: if v1: v18 = merge(v9, 6)
19: v19 = combine(v3, v9)

Query:
Final output value of v0

Trace the data and control dependencies backward from the final output and report the minimal set of statements that must be kept to compute that final value. Here 0-indexed statement numbers are shown at the start of each line; a line '3: if v2: v3 = merge(v1, 4)' means statement 3 depends on statement 1 (data) and on statement 2 (control, its guard). List the kept statement numbers in increasing order, separated by spaces (for example: '0 2 5'). The answer is exactly that list.
**Answer:** 0
