## Level 0

### Prompt
```
Consider a database schedule of transactions. Each line lists one operation 'T<id><r|w><item>': transaction T<id> reads or writes a single item, in the order the lines appear. Two operations conflict if they act on the same item and at least one is a write. Build the conflict-precedence graph: for a write by Ti then a read or write by Tj on the same item, and for a read by Ti then a write by Tj on the same item, put edge Ti -> Tj.

1. T0w2
2. T1w2
3. T0r2
4. T2w0
5. T2w2
6. T0r0

The schedule is conflict-serializable exactly when this graph has no directed cycle, and then a serial order is any topological ordering of the graph. Decide whether the schedule is conflict-serializable. If it is, give the answer as 'serial T0,T2,T1' with the transaction ids in the lexicographically first topological order, comma-separated. If it is not serializable, give the answer as 'nonserial T1,T3' listing (in sorted order) every transaction id that lies on at least one directed precedence cycle.
```

### Answer
```
nonserial T0,T1,T2
```


### Prompt
```
Consider a database schedule of transactions. Each line lists one operation 'T<id><r|w><item>': transaction T<id> reads or writes a single item, in the order the lines appear. Two operations conflict if they act on the same item and at least one is a write. Build the conflict-precedence graph: for a write by Ti then a read or write by Tj on the same item, and for a read by Ti then a write by Tj on the same item, put edge Ti -> Tj.

1. T1w1
2. T1w1
3. T0w0
4. T1w1
5. T2r0
6. T2r1

The schedule is conflict-serializable exactly when this graph has no directed cycle, and then a serial order is any topological ordering of the graph. Decide whether the schedule is conflict-serializable. If it is, give the answer as 'serial T0,T2,T1' with the transaction ids in the lexicographically first topological order, comma-separated. If it is not serializable, give the answer as 'nonserial T1,T3' listing (in sorted order) every transaction id that lies on at least one directed precedence cycle.
```

### Answer
```
serial T0,T1,T2
```

## Level 2

### Prompt
```
Consider a database schedule of transactions. Each line lists one operation 'T<id><r|w><item>': transaction T<id> reads or writes a single item, in the order the lines appear. Two operations conflict if they act on the same item and at least one is a write. Build the conflict-precedence graph: for a write by Ti then a read or write by Tj on the same item, and for a read by Ti then a write by Tj on the same item, put edge Ti -> Tj.

1. T4r1
2. T1r4
3. T1w0
4. T2w1
5. T0r0
6. T4r3
7. T3r0
8. T3r2
9. T3w3
10. T1w2
11. T1r1
12. T1w1
13. T1r2

The schedule is conflict-serializable exactly when this graph has no directed cycle, and then a serial order is any topological ordering of the graph. Decide whether the schedule is conflict-serializable. If it is, give the answer as 'serial T0,T2,T1' with the transaction ids in the lexicographically first topological order, comma-separated. If it is not serializable, give the answer as 'nonserial T1,T3' listing (in sorted order) every transaction id that lies on at least one directed precedence cycle.
```

### Answer
```
nonserial T1,T3
```


### Prompt
```
Consider a database schedule of transactions. Each line lists one operation 'T<id><r|w><item>': transaction T<id> reads or writes a single item, in the order the lines appear. Two operations conflict if they act on the same item and at least one is a write. Build the conflict-precedence graph: for a write by Ti then a read or write by Tj on the same item, and for a read by Ti then a write by Tj on the same item, put edge Ti -> Tj.

1. T1r4
2. T4r4
3. T0w1
4. T1r2
5. T4r1
6. T4r3
7. T2w4
8. T3r3
9. T2w4
10. T4w3
11. T0w0
12. T2w0
13. T2r4

The schedule is conflict-serializable exactly when this graph has no directed cycle, and then a serial order is any topological ordering of the graph. Decide whether the schedule is conflict-serializable. If it is, give the answer as 'serial T0,T2,T1' with the transaction ids in the lexicographically first topological order, comma-separated. If it is not serializable, give the answer as 'nonserial T1,T3' listing (in sorted order) every transaction id that lies on at least one directed precedence cycle.
```

### Answer
```
serial T0,T1,T3,T4,T2
```

## Level 5

### Prompt
```
Consider a database schedule of transactions. Each line lists one operation 'T<id><r|w><item>': transaction T<id> reads or writes a single item, in the order the lines appear. Two operations conflict if they act on the same item and at least one is a write. Build the conflict-precedence graph: for a write by Ti then a read or write by Tj on the same item, and for a read by Ti then a write by Tj on the same item, put edge Ti -> Tj.

1. T2w3
2. T0w0
3. T2r7
4. T3w0
5. T4w3
6. T6w3
7. T6r7
8. T6w5
9. T6w6
10. T0w7
11. T0r0
12. T7r2
13. T7r3
14. T2r0
15. T3w7
16. T7w2
17. T0w0
18. T1w5
19. T7w0
20. T5w4
21. T0w4

The schedule is conflict-serializable exactly when this graph has no directed cycle, and then a serial order is any topological ordering of the graph. Decide whether the schedule is conflict-serializable. If it is, give the answer as 'serial T0,T2,T1' with the transaction ids in the lexicographically first topological order, comma-separated. If it is not serializable, give the answer as 'nonserial T1,T3' listing (in sorted order) every transaction id that lies on at least one directed precedence cycle.
```

### Answer
```
nonserial T0,T2,T3,T4,T6
```


### Prompt
```
Consider a database schedule of transactions. Each line lists one operation 'T<id><r|w><item>': transaction T<id> reads or writes a single item, in the order the lines appear. Two operations conflict if they act on the same item and at least one is a write. Build the conflict-precedence graph: for a write by Ti then a read or write by Tj on the same item, and for a read by Ti then a write by Tj on the same item, put edge Ti -> Tj.

1. T0w5
2. T4w7
3. T4w6
4. T1r7
5. T2w2
6. T7w7
7. T1r5
8. T2r0
9. T3w2
10. T7w4
11. T2r0
12. T7w4
13. T6w1
14. T5r1
15. T2r7
16. T6w1
17. T2r4
18. T6r7
19. T5w4
20. T6w6
21. T6w5

The schedule is conflict-serializable exactly when this graph has no directed cycle, and then a serial order is any topological ordering of the graph. Decide whether the schedule is conflict-serializable. If it is, give the answer as 'serial T0,T2,T1' with the transaction ids in the lexicographically first topological order, comma-separated. If it is not serializable, give the answer as 'nonserial T1,T3' listing (in sorted order) every transaction id that lies on at least one directed precedence cycle.
```

### Answer
```
nonserial T5,T6
```
