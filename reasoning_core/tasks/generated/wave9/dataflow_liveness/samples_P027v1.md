# Samples P027v1

## Level 0

### Example 1

**Prompt**

```
A control-flow graph has blocks indexed 0..4. The successor (outgoing edge) lists are:
successors = B0->[2], B1->[2,3,4], B2->[4], B3->[4], B4->[]
For each block, Gen is the set of variables defined (made live) there and Kill is the set of variables killed (not live on exit):
B0: Gen={b,c} Kill={a,d}
B1: Gen={a,b,c,d} Kill={}
B2: Gen={a,b,c,d} Kill={}
B3: Gen={a,b,d} Kill={}
B4: Gen={a,b} Kill={c,d}
Compute live-in and live-out variable sets using the backward dataflow equations LiveOut[b] = union of LiveIn of successors, LiveIn[b] = (LiveOut[b] - Kill[b]) | Gen[b], iterated over blocks in reverse post-order order 4,3,2,1,0 until fixpoint.
Give the answer as the live-in sets of blocks 0..4 in order, space-separated, each set written as comma-separated variables inside braces, e.g. for 2 blocks: {a} {b,c}.
The live-in sets are:
```

**Answer**

```
{b,c} {a,b,c,d} {a,b,c,d} {a,b,d} {a,b}
```

### Example 2

**Prompt**

```
A control-flow graph has blocks indexed 0..4. The successor (outgoing edge) lists are:
successors = B0->[1], B1->[3], B2->[4], B3->[4], B4->[]
For each block, Gen is the set of variables defined (made live) there and Kill is the set of variables killed (not live on exit):
B0: Gen={a} Kill={b,c,d}
B1: Gen={a} Kill={b,c,d}
B2: Gen={a} Kill={c}
B3: Gen={a,b} Kill={c}
B4: Gen={a,c,d} Kill={b}
Compute live-in and live-out variable sets using the backward dataflow equations LiveOut[b] = union of LiveIn of successors, LiveIn[b] = (LiveOut[b] - Kill[b]) | Gen[b], iterated over blocks in reverse post-order order 4,3,1,2,0 until fixpoint.
Give the answer as the live-in sets of blocks 0..4 in order, space-separated, each set written as comma-separated variables inside braces, e.g. for 2 blocks: {a} {b,c}.
The live-in sets are:
```

**Answer**

```
{a} {a} {a,d} {a,b,d} {a,c,d}
```

## Level 2

### Example 1

**Prompt**

```
A control-flow graph has blocks indexed 0..6. The successor (outgoing edge) lists are:
successors = B0->[1,2,4,5], B1->[4,5,6], B2->[5,6], B3->[5,6], B4->[5], B5->[6], B6->[]
For each block, Gen is the set of variables defined (made live) there and Kill is the set of variables killed (not live on exit):
B0: Gen={b,c,f} Kill={}
B1: Gen={a,b,c,e,f} Kill={}
B2: Gen={a,b,c,f} Kill={d}
B3: Gen={a,c,d,e,f} Kill={}
B4: Gen={b,e,f} Kill={a,c}
B5: Gen={a,c,d,e,f} Kill={b}
B6: Gen={b,c,d} Kill={a}
Compute live-in and live-out variable sets using the backward dataflow equations LiveOut[b] = union of LiveIn of successors, LiveIn[b] = (LiveOut[b] - Kill[b]) | Gen[b], iterated over blocks in reverse post-order order 6,5,4,2,1,3,0 until fixpoint.
Give the answer as the live-in sets of blocks 0..6 in order, space-separated, each set written as comma-separated variables inside braces, e.g. for 2 blocks: {a} {b,c}.
The live-in sets are:
```

**Answer**

```
{a,b,c,d,e,f} {a,b,c,d,e,f} {a,b,c,e,f} {a,b,c,d,e,f} {b,d,e,f} {a,c,d,e,f} {b,c,d}
```

### Example 2

**Prompt**

```
A control-flow graph has blocks indexed 0..6. The successor (outgoing edge) lists are:
successors = B0->[1,3,4,5,6], B1->[5,6], B2->[4], B3->[5], B4->[5], B5->[6], B6->[]
For each block, Gen is the set of variables defined (made live) there and Kill is the set of variables killed (not live on exit):
B0: Gen={a,b,c,d,e,f} Kill={}
B1: Gen={a} Kill={b,c,d}
B2: Gen={a,b,e,f} Kill={}
B3: Gen={a,c,d,f} Kill={b,e}
B4: Gen={b} Kill={a,c,d,f}
B5: Gen={b,c,e,f} Kill={a,d}
B6: Gen={b,c,d,e} Kill={f}
Compute live-in and live-out variable sets using the backward dataflow equations LiveOut[b] = union of LiveIn of successors, LiveIn[b] = (LiveOut[b] - Kill[b]) | Gen[b], iterated over blocks in reverse post-order order 6,5,4,3,1,2,0 until fixpoint.
Give the answer as the live-in sets of blocks 0..6 in order, space-separated, each set written as comma-separated variables inside braces, e.g. for 2 blocks: {a} {b,c}.
The live-in sets are:
```

**Answer**

```
{a,b,c,d,e,f} {a,e,f} {a,b,e,f} {a,c,d,f} {b,e} {b,c,e,f} {b,c,d,e}
```

## Level 5

### Example 1

**Prompt**

```
A control-flow graph has blocks indexed 0..9. The successor (outgoing edge) lists are:
successors = B0->[1,2,4,5,7,8,9], B1->[2,3,4,5,7,9], B2->[4,5,6,7,8,9], B3->[7,8,9], B4->[5,6,8,9], B5->[6,7,8,9], B6->[7,8,9], B7->[9], B8->[9], B9->[]
For each block, Gen is the set of variables defined (made live) there and Kill is the set of variables killed (not live on exit):
B0: Gen={a,b,c,d,e,f,g,h,i} Kill={}
B1: Gen={b} Kill={a,c,d,e,f,g,h,i}
B2: Gen={a,b,c,d,e,f,g,h} Kill={}
B3: Gen={a,b,c,d,e,f,g,h,i} Kill={}
B4: Gen={a,d,e,h,i} Kill={f,g}
B5: Gen={a,b,c,f,g,h,i} Kill={d,e}
B6: Gen={a,d} Kill={b,c,e,f,g,h}
B7: Gen={a,c,d,f,g,h} Kill={b,e,i}
B8: Gen={a,b,c,d,e,f,g,h,i} Kill={}
B9: Gen={a,b,f,g,i} Kill={d,e,h}
Compute live-in and live-out variable sets using the backward dataflow equations LiveOut[b] = union of LiveIn of successors, LiveIn[b] = (LiveOut[b] - Kill[b]) | Gen[b], iterated over blocks in reverse post-order order 9,8,7,6,5,4,3,2,1,0 until fixpoint.
Give the answer as the live-in sets of blocks 0..9 in order, space-separated, each set written as comma-separated variables inside braces, e.g. for 2 blocks: {a} {b,c}.
The live-in sets are:
```

**Answer**

```
{a,b,c,d,e,f,g,h,i} {b} {a,b,c,d,e,f,g,h,i} {a,b,c,d,e,f,g,h,i} {a,b,c,d,e,h,i} {a,b,c,f,g,h,i} {a,d,i} {a,c,d,f,g,h} {a,b,c,d,e,f,g,h,i} {a,b,f,g,i}
```

### Example 2

**Prompt**

```
A control-flow graph has blocks indexed 0..9. The successor (outgoing edge) lists are:
successors = B0->[1,2,3,4,6,7,8,9], B1->[3,4,5,6,7,8,9], B2->[4,5,6,7,8,9], B3->[4,5,6,7,8], B4->[5,6,7,8,9], B5->[6,7,8,9], B6->[7,8,9], B7->[8,9], B8->[9], B9->[]
For each block, Gen is the set of variables defined (made live) there and Kill is the set of variables killed (not live on exit):
B0: Gen={a,b,d,h,i} Kill={c,e,f,g}
B1: Gen={a,b,c,d,f} Kill={e,h,i}
B2: Gen={b,f} Kill={a,e,h,i}
B3: Gen={a,b,c,d,e,f,g,h,i} Kill={}
B4: Gen={f,i} Kill={a,c,d,e,g,h}
B5: Gen={c,d,f,i} Kill={a,b,e,h}
B6: Gen={a,c,e,h,i} Kill={}
B7: Gen={a,b,c,e,g,i} Kill={d,f}
B8: Gen={i} Kill={a,g,h}
B9: Gen={a,f,g,h,i} Kill={d}
Compute live-in and live-out variable sets using the backward dataflow equations LiveOut[b] = union of LiveIn of successors, LiveIn[b] = (LiveOut[b] - Kill[b]) | Gen[b], iterated over blocks in reverse post-order order 9,8,7,6,5,4,3,2,1,0 until fixpoint.
Give the answer as the live-in sets of blocks 0..9 in order, space-separated, each set written as comma-separated variables inside braces, e.g. for 2 blocks: {a} {b,c}.
The live-in sets are:
```

**Answer**

```
{a,b,d,h,i} {a,b,c,d,f,g} {b,c,d,f,g} {a,b,c,d,e,f,g,h,i} {b,f,i} {c,d,f,g,i} {a,b,c,e,f,g,h,i} {a,b,c,e,g,h,i} {f,i} {a,f,g,h,i}
```
