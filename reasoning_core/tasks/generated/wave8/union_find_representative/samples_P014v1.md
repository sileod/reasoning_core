## Level 0
```
We have a set of 6 elements labeled 0 through 5, initially each in its own set.
Operations are applied left to right:
  union 2 with 3
  query 0
  query 3
  query 3
  query 5
  union 0 with 3
  query 1
  union 1 with 4

After applying all operations in order, what is the representative (root) of element 1? Use union by rank: the root with the higher rank becomes the parent; if the two roots have equal rank, the larger-labeled root becomes the parent and the rank of the chosen root increases by one.

The answer is a single integer.
```

Answer: 4

```
We have a set of 6 elements labeled 0 through 5, initially each in its own set.
Operations are applied left to right:
  union 3 with 4
  query 4
  union 2 with 4
  query 3
  query 1
  query 3
  union 3 with 0
  query 2

After applying all operations in order, what is the representative (root) of element 2? Use union by rank: the root with the higher rank becomes the parent; if the two roots have equal rank, the larger-labeled root becomes the parent and the rank of the chosen root increases by one.

The answer is a single integer.
```

Answer: 4

## Level 2
```
We have a set of 12 elements labeled 0 through 11, initially each in its own set.
Operations are applied left to right:
  union 6 with 9
  query 7
  union 1 with 4
  query 0
  union 9 with 8
  union 2 with 1
  union 6 with 2
  union 9 with 1
  union 7 with 3
  query 6
  union 9 with 7
  union 10 with 0
  union 0 with 11
  query 11

After applying all operations in order, what is the representative (root) of element 11? Use union by rank: the root with the higher rank becomes the parent; if the two roots have equal rank, the larger-labeled root becomes the parent and the rank of the chosen root increases by one.

The answer is a single integer.
```

Answer: 10

```
We have a set of 12 elements labeled 0 through 11, initially each in its own set.
Operations are applied left to right:
  query 0
  union 6 with 1
  union 7 with 10
  union 3 with 9
  query 0
  query 6
  union 10 with 2
  union 3 with 7
  union 3 with 0
  union 8 with 4
  union 11 with 6
  union 11 with 10
  query 0
  query 9

After applying all operations in order, what is the representative (root) of element 9? Use union by rank: the root with the higher rank becomes the parent; if the two roots have equal rank, the smaller-labeled root becomes the parent and the rank of the chosen root increases by one.

The answer is a single integer.
```

Answer: 3

## Level 5
```
We have a set of 21 elements labeled 0 through 20, initially each in its own set.
Operations are applied left to right:
  query 13
  query 5
  union 15 with 10
  query 8
  union 3 with 5
  union 13 with 15
  union 3 with 10
  union 11 with 5
  union 4 with 14
  query 5
  union 13 with 19
  union 10 with 13
  union 20 with 16
  query 7
  query 14
  query 12
  union 9 with 19
  query 3
  query 18
  union 8 with 12
  query 5
  union 3 with 0
  union 5 with 13

After applying all operations in order, what is the representative (root) of element 5? Use union by rank: the root with the higher rank becomes the parent; if the two roots have equal rank, the larger-labeled root becomes the parent and the rank of the chosen root increases by one.

The answer is a single integer.
```

Answer: 15

```
We have a set of 21 elements labeled 0 through 20, initially each in its own set.
Operations are applied left to right:
  query 6
  query 0
  union 19 with 11
  union 12 with 15
  query 0
  union 0 with 3
  union 18 with 17
  query 11
  query 9
  union 17 with 18
  union 11 with 1
  query 15
  union 6 with 19
  query 15
  query 6
  union 1 with 12
  union 3 with 20
  union 17 with 12
  query 12
  union 15 with 17
  union 18 with 8
  query 8
  union 3 with 16

After applying all operations in order, what is the representative (root) of element 8? Use union by rank: the root with the higher rank becomes the parent; if the two roots have equal rank, the smaller-labeled root becomes the parent and the rank of the chosen root increases by one.

The answer is a single integer.
```

Answer: 11

