# Samples P031v1

## Level 0

### Example 1

Prompt:
```
Consider the following transaction schedule. Operations are rT(A) (transaction T reads item A) and wT(A) (transaction T writes item A). Two operations from different transactions conflict if they touch the same item and at least one is a write. A schedule is conflict-serializable iff the precedence graph (edge Ti->Tj when an operation of Ti precedes a conflicting operation of Tj) is acyclic, and its unique topological order is the conflict-equivalent serial order.

Schedule: r2(2),w1(4),r1(1),w1(2),w1(1),w2(1)

Decide whether the schedule is conflict-serializable. If YES, give the unique serial order of transactions. The answer format is exactly 'YES:T1,T2,...' (the serial order, a full permutation of all transactions) if it is conflict-serializable, or exactly 'NO' if it is not.
Example: 'YES:2,1,3' or 'NO'. The answer is a single exact string.
```
Answer:
```
NO
```

### Example 2

Prompt:
```
Consider the following transaction schedule. Operations are rT(A) (transaction T reads item A) and wT(A) (transaction T writes item A). Two operations from different transactions conflict if they touch the same item and at least one is a write. A schedule is conflict-serializable iff the precedence graph (edge Ti->Tj when an operation of Ti precedes a conflicting operation of Tj) is acyclic, and its unique topological order is the conflict-equivalent serial order.

Schedule: r1(3),w2(1),w4(1),w1(2),w1(1),r2(1),r3(1),r4(3),r4(1),r2(3),r1(3),r4(2),r3(2),w3(2)

Decide whether the schedule is conflict-serializable. If YES, give the unique serial order of transactions. The answer format is exactly 'YES:T1,T2,...' (the serial order, a full permutation of all transactions) if it is conflict-serializable, or exactly 'NO' if it is not.
Example: 'YES:2,1,3' or 'NO'. The answer is a single exact string.
```
Answer:
```
YES:1,2,4,3
```

## Level 2

### Example 1

Prompt:
```
Consider the following transaction schedule. Operations are rT(A) (transaction T reads item A) and wT(A) (transaction T writes item A). Two operations from different transactions conflict if they touch the same item and at least one is a write. A schedule is conflict-serializable iff the precedence graph (edge Ti->Tj when an operation of Ti precedes a conflicting operation of Tj) is acyclic, and its unique topological order is the conflict-equivalent serial order.

Schedule: w3(2),w2(4),r4(3),r3(2),r4(2),r3(4),r1(2),r2(5),w1(5),w4(4),w3(6),r5(4),w2(3),r2(3),w5(5),r4(5),w5(6)

Decide whether the schedule is conflict-serializable. If YES, give the unique serial order of transactions. The answer format is exactly 'YES:T1,T2,...' (the serial order, a full permutation of all transactions) if it is conflict-serializable, or exactly 'NO' if it is not.
Example: 'YES:2,1,3' or 'NO'. The answer is a single exact string.
```
Answer:
```
YES:1,4,5,2,3
```

### Example 2

Prompt:
```
Consider the following transaction schedule. Operations are rT(A) (transaction T reads item A) and wT(A) (transaction T writes item A). Two operations from different transactions conflict if they touch the same item and at least one is a write. A schedule is conflict-serializable iff the precedence graph (edge Ti->Tj when an operation of Ti precedes a conflicting operation of Tj) is acyclic, and its unique topological order is the conflict-equivalent serial order.

Schedule: w5(6),w2(6),w4(4),r3(5),w5(3),r4(3),r1(1),w1(3),w2(1),r3(1),r5(4),w4(6),r2(3),r1(3),w5(6)

Decide whether the schedule is conflict-serializable. If YES, give the unique serial order of transactions. The answer format is exactly 'YES:T1,T2,...' (the serial order, a full permutation of all transactions) if it is conflict-serializable, or exactly 'NO' if it is not.
Example: 'YES:2,1,3' or 'NO'. The answer is a single exact string.
```
Answer:
```
YES:1,4,5,2,3
```

## Level 5

### Example 1

Prompt:
```
Consider the following transaction schedule. Operations are rT(A) (transaction T reads item A) and wT(A) (transaction T writes item A). Two operations from different transactions conflict if they touch the same item and at least one is a write. A schedule is conflict-serializable iff the precedence graph (edge Ti->Tj when an operation of Ti precedes a conflicting operation of Tj) is acyclic, and its unique topological order is the conflict-equivalent serial order.

Schedule: r2(5),w1(4),r2(4),w1(5)

Decide whether the schedule is conflict-serializable. If YES, give the unique serial order of transactions. The answer format is exactly 'YES:T1,T2,...' (the serial order, a full permutation of all transactions) if it is conflict-serializable, or exactly 'NO' if it is not.
Example: 'YES:2,1,3' or 'NO'. The answer is a single exact string.
```
Answer:
```
NO
```

### Example 2

Prompt:
```
Consider the following transaction schedule. Operations are rT(A) (transaction T reads item A) and wT(A) (transaction T writes item A). Two operations from different transactions conflict if they touch the same item and at least one is a write. A schedule is conflict-serializable iff the precedence graph (edge Ti->Tj when an operation of Ti precedes a conflicting operation of Tj) is acyclic, and its unique topological order is the conflict-equivalent serial order.

Schedule: r3(6),r2(2),w2(5),w4(6),w8(8),w4(7),w5(6),w6(7),w6(7),r5(1),r8(3),r4(3),w7(1),r7(8),w1(9),r9(5),r4(4),r1(3),r8(7),w10(1),r9(2),r10(7),r1(1),r3(3),w7(3),r10(9),w9(1),r1(4),r6(1),r5(3),r4(1),w2(1),r6(4),w2(5),w6(1)

Decide whether the schedule is conflict-serializable. If YES, give the unique serial order of transactions. The answer format is exactly 'YES:T1,T2,...' (the serial order, a full permutation of all transactions) if it is conflict-serializable, or exactly 'NO' if it is not.
Example: 'YES:2,1,3' or 'NO'. The answer is a single exact string.
```
Answer:
```
YES:2,9,1,10,6,4,8,7,5,3
```
