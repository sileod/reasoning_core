# Samples for P012v1

## Level 0

### Prompt
```
Consider the directed acyclic graph on nodes {0 1 2 3}.
Edges:
1 -> 0
1 -> 2
3 -> 1
3 -> 2
Rule: remove the smallest-indexed zero-indegree node first.
The nodes are removed in rounds, 2 per round, always removing zero-indegree nodes in the order the rule specifies, until the graph is empty.
Which nodes are removed in round 1?
The answer is the comma-separated list of those node numbers in ascending order.
```
### Answer
```
1,3
```

### Prompt
```
Consider the directed acyclic graph on nodes {0 1 2 3}.
Edges:
1 -> 0
2 -> 0
2 -> 3
3 -> 0
Rule: remove the smallest-indexed zero-indegree node first.
The nodes are removed one at a time, always removing zero-indegree nodes in the order the rule specifies, until the graph is empty.
List every node in the exact order it is removed, as a comma-separated list of node numbers.
```
### Answer
```
1,2,3,0
```

## Level 2

### Prompt
```
Consider the directed acyclic graph on nodes {0 1 2 3 4 5}.
Edges:
0 -> 1
2 -> 0
2 -> 4
2 -> 5
3 -> 0
3 -> 1
3 -> 2
3 -> 5
4 -> 1
5 -> 4
Rule: remove the smallest-indexed zero-indegree node first.
The nodes are removed in rounds, 2 per round, always removing zero-indegree nodes in the order the rule specifies, until the graph is empty.
Which nodes are removed in round 1?
The answer is the comma-separated list of those node numbers in ascending order.
```
### Answer
```
2,3
```

### Prompt
```
Consider the directed acyclic graph on nodes {0 1 2 3 4 5}.
Edges:
0 -> 1
0 -> 2
0 -> 3
0 -> 5
1 -> 4
2 -> 1
2 -> 3
4 -> 3
5 -> 2
5 -> 3
Rule: remove the largest-indexed zero-indegree node first.
The nodes are removed one at a time, always removing zero-indegree nodes in the order the rule specifies, until the graph is empty.
List every node in the exact order it is removed, as a comma-separated list of node numbers.
```
### Answer
```
0,5,2,1,4,3
```

## Level 5

### Prompt
```
Consider the directed acyclic graph on nodes {0 1 2 3 4 5 6 7 8}.
Edges:
0 -> 1
0 -> 2
0 -> 7
2 -> 1
2 -> 7
3 -> 0
3 -> 1
3 -> 4
3 -> 7
4 -> 1
4 -> 2
4 -> 7
5 -> 2
5 -> 3
5 -> 7
6 -> 0
6 -> 1
6 -> 2
6 -> 5
6 -> 7
7 -> 1
8 -> 0
8 -> 5
8 -> 7
Rule: remove the largest-indexed zero-indegree node first.
The nodes are removed in rounds, 2 per round, always removing zero-indegree nodes in the order the rule specifies, until the graph is empty.
Which nodes are removed in round 3?
The answer is the comma-separated list of those node numbers in ascending order.
```
### Answer
```
0,4
```

### Prompt
```
Consider the directed acyclic graph on nodes {0 1 2 3 4 5 6 7 8}.
Edges:
0 -> 2
0 -> 4
0 -> 5
0 -> 7
1 -> 0
1 -> 2
1 -> 5
1 -> 7
1 -> 8
2 -> 4
2 -> 5
2 -> 6
2 -> 7
2 -> 8
3 -> 2
3 -> 4
3 -> 5
3 -> 7
5 -> 4
6 -> 5
6 -> 7
7 -> 4
8 -> 5
8 -> 7
Rule: remove the smallest-indexed zero-indegree node first.
The nodes are removed in rounds, 2 per round, always removing zero-indegree nodes in the order the rule specifies, until the graph is empty.
Which nodes are removed in round 2?
The answer is the comma-separated list of those node numbers in ascending order.
```
### Answer
```
2,3
```
