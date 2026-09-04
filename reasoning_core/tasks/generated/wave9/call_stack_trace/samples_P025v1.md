## Level 0
### Example 1
**Prompt:**
A recursive function sum_subtree(n) adds a node's value to the sums returned by its children: it returns n.value + sum(sum_subtree(c) for c in n.children). A tree is written in prefix notation as value(child1,...,childk); leaves are bare values.

Tree: 2(-8,5,-9)

Count the nodes of the tree in the order they are written (left to right, parent before its children); the first node written is position 0. What does sum_subtree return for the node at position 2?

The answer is a single integer.

**Answer:**
5

### Example 2
**Prompt:**
A recursive function sum_subtree(n) adds a node's value to the sums returned by its children: it returns n.value + sum(sum_subtree(c) for c in n.children). A tree is written in prefix notation as value(child1,...,childk); leaves are bare values.

Tree: 8(-8(1(-5)))

Count the nodes of the tree in the order they are written (left to right, parent before its children); the first node written is position 0. What does sum_subtree return for the node at position 0?

The answer is a single integer.

**Answer:**
-4

## Level 2
### Example 1
**Prompt:**
A recursive function sum_subtree(n) adds a node's value to the sums returned by its children: it returns n.value + sum(sum_subtree(c) for c in n.children). A tree is written in prefix notation as value(child1,...,childk); leaves are bare values.

Tree: -1(-7,0(0,-2,-5(7)))

Count the nodes of the tree in the order they are written (left to right, parent before its children); the first node written is position 0. What does sum_subtree return for the node at position 6?

The answer is a single integer.

**Answer:**
7

### Example 2
**Prompt:**
A recursive function sum_subtree(n) adds a node's value to the sums returned by its children: it returns n.value + sum(sum_subtree(c) for c in n.children). A tree is written in prefix notation as value(child1,...,childk); leaves are bare values.

Tree: 1(-9(1),-6(1(-9)),6)

Count the nodes of the tree in the order they are written (left to right, parent before its children); the first node written is position 0. What does sum_subtree return for the node at position 5?

The answer is a single integer.

**Answer:**
-9

## Level 5
### Example 1
**Prompt:**
A recursive function sum_subtree(n) adds a node's value to the sums returned by its children: it returns n.value + sum(sum_subtree(c) for c in n.children). A tree is written in prefix notation as value(child1,...,childk); leaves are bare values.

Tree: -8(-5(-3(-2),2(-2)),4,-4,-1(-2),2)

Count the nodes of the tree in the order they are written (left to right, parent before its children); the first node written is position 0. What does sum_subtree return for the node at position 10?

The answer is a single integer.

**Answer:**
2

### Example 2
**Prompt:**
A recursive function sum_subtree(n) adds a node's value to the sums returned by its children: it returns n.value + sum(sum_subtree(c) for c in n.children). A tree is written in prefix notation as value(child1,...,childk); leaves are bare values.

Tree: -3(-4(-6),6(-8,-8(-4)),6(6),3(-8))

Count the nodes of the tree in the order they are written (left to right, parent before its children); the first node written is position 0. What does sum_subtree return for the node at position 1?

The answer is a single integer.

**Answer:**
-10

