## Level 0

### Example 1

**Prompt:**

```
A Fenwick tree (binary indexed tree) is indexed 1..64. Its prefix sum query for prefix 26 visits node 26, adds its value, then subtracts the lowest set bit of the current node (i & -i) to move to the next node, repeating until the node becomes 0.

List the nodes visited, starting at 26 and going down to (but not including) 0, in decreasing order.

The answer is a comma-separated list of integers, e.g. "8,0" would be wrong because it includes 0; give just the visited nonzero nodes, like "12,8".
```

**Answer:**

```
26,24,16
```

### Example 2

**Prompt:**

```
A Fenwick tree (binary indexed tree) is indexed 1..64. Its prefix sum query for prefix 53 visits node 53, adds its value, then subtracts the lowest set bit of the current node (i & -i) to move to the next node, repeating until the node becomes 0.

List the nodes visited, starting at 53 and going down to (but not including) 0, in decreasing order.

The answer is a comma-separated list of integers, e.g. "8,0" would be wrong because it includes 0; give just the visited nonzero nodes, like "12,8".
```

**Answer:**

```
53,52,48,32
```


## Level 2

### Example 1

**Prompt:**

```
A Fenwick tree (binary indexed tree) is indexed 1..256. Its prefix sum query for prefix 60 visits node 60, adds its value, then subtracts the lowest set bit of the current node (i & -i) to move to the next node, repeating until the node becomes 0.

List the nodes visited, starting at 60 and going down to (but not including) 0, in decreasing order.

The answer is a comma-separated list of integers, e.g. "8,0" would be wrong because it includes 0; give just the visited nonzero nodes, like "12,8".
```

**Answer:**

```
60,56,48,32
```

### Example 2

**Prompt:**

```
A Fenwick tree (binary indexed tree) is indexed 1..256. Its prefix sum query for prefix 90 visits node 90, adds its value, then subtracts the lowest set bit of the current node (i & -i) to move to the next node, repeating until the node becomes 0.

List the nodes visited, starting at 90 and going down to (but not including) 0, in decreasing order.

The answer is a comma-separated list of integers, e.g. "8,0" would be wrong because it includes 0; give just the visited nonzero nodes, like "12,8".
```

**Answer:**

```
90,88,80,64
```


## Level 5

### Example 1

**Prompt:**

```
A Fenwick tree (binary indexed tree) is indexed 1..2048. Its prefix sum query for prefix 1912 visits node 1912, adds its value, then subtracts the lowest set bit of the current node (i & -i) to move to the next node, repeating until the node becomes 0.

List the nodes visited, starting at 1912 and going down to (but not including) 0, in decreasing order.

The answer is a comma-separated list of integers, e.g. "8,0" would be wrong because it includes 0; give just the visited nonzero nodes, like "12,8".
```

**Answer:**

```
1912,1904,1888,1856,1792,1536,1024
```

### Example 2

**Prompt:**

```
A Fenwick tree (binary indexed tree) is indexed 1..2048. Its prefix sum query for prefix 1231 visits node 1231, adds its value, then subtracts the lowest set bit of the current node (i & -i) to move to the next node, repeating until the node becomes 0.

List the nodes visited, starting at 1231 and going down to (but not including) 0, in decreasing order.

The answer is a comma-separated list of integers, e.g. "8,0" would be wrong because it includes 0; give just the visited nonzero nodes, like "12,8".
```

**Answer:**

```
1231,1230,1228,1224,1216,1152,1024
```

