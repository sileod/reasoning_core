## Level 0

### Prompt

A graph has 8 nodes numbered 0 through 7.
Its edges, added in this order, are: Edges:
[[0, 1], [4, 3], [2, 7], [2, 4], [5, 3], [5, 0], [1, 1], [0, 7]].
We maintain a union-find data structure over the nodes using union-by-rank with path compression.
Process every edge in the listed order with a union call. Then, for each query pair (a, b) in the listed order, take the find of both ends.
For every query, write '1' if a and b are in the same component and '0' otherwise, followed by the smallest-numbered node in a's component.
Queries: Queries:
[[6, 1], [7, 0], [0, 6]].
Give only the answer: concatenated pairs of (connect bit, canonical node of a), separated by spaces, in query order.

### Answer

0 6 1 0 0 0

### Prompt

A graph has 8 nodes numbered 0 through 7.
Its edges, added in this order, are: Edges:
[[0, 5], [4, 2], [2, 6], [2, 6], [5, 0], [1, 1], [5, 0], [1, 1]].
We maintain a union-find data structure over the nodes using union-by-rank with path compression.
Process every edge in the listed order with a union call. Then, for each query pair (a, b) in the listed order, take the find of both ends.
For every query, write '1' if a and b are in the same component and '0' otherwise, followed by the smallest-numbered node in a's component.
Queries: Queries:
[[0, 7], [4, 4], [7, 6]].
Give only the answer: concatenated pairs of (connect bit, canonical node of a), separated by spaces, in query order.

### Answer

0 0 1 2 0 7

## Level 2

### Prompt

A graph has 16 nodes numbered 0 through 15.
Its edges, added in this order, are: Edges:
[[1, 3], [11, 4], [13, 5], [11, 3], [13, 1], [12, 7], [7, 0], [0, 0], [1, 1], [12, 4], [12, 4], [15, 8], [14, 11], [0, 11], [15, 7], [12, 13], [11, 3], [10, 15], [15, 1], [3, 1]].
We maintain a union-find data structure over the nodes using union-by-rank with path compression.
Process every edge in the listed order with a union call. Then, for each query pair (a, b) in the listed order, take the find of both ends.
For every query, write '1' if a and b are in the same component and '0' otherwise, followed by the smallest-numbered node in a's component.
Queries: Queries:
[[14, 3], [1, 10], [6, 8], [13, 13], [4, 11], [13, 3], [6, 5]].
Give only the answer: concatenated pairs of (connect bit, canonical node of a), separated by spaces, in query order.

### Answer

1 0 1 0 0 6 1 0 1 0 1 0 0 6

### Prompt

A graph has 16 nodes numbered 0 through 15.
Its edges, added in this order, are: Edges:
[[7, 5], [0, 15], [10, 12], [5, 1], [15, 7], [2, 6], [3, 6], [12, 5], [13, 13], [4, 15], [10, 4], [4, 5], [12, 8], [0, 13], [1, 10], [11, 0], [5, 3], [12, 13], [7, 10], [8, 9]].
We maintain a union-find data structure over the nodes using union-by-rank with path compression.
Process every edge in the listed order with a union call. Then, for each query pair (a, b) in the listed order, take the find of both ends.
For every query, write '1' if a and b are in the same component and '0' otherwise, followed by the smallest-numbered node in a's component.
Queries: Queries:
[[3, 11], [9, 8], [14, 6], [0, 4], [15, 13], [12, 2], [4, 1]].
Give only the answer: concatenated pairs of (connect bit, canonical node of a), separated by spaces, in query order.

### Answer

1 0 1 0 0 14 1 0 1 0 1 0 1 0

## Level 5

### Prompt

A graph has 28 nodes numbered 0 through 27.
Its edges, added in this order, are: Edges:
[[13, 6], [26, 5], [12, 17], [18, 19], [1, 16], [7, 23], [0, 3], [4, 11], [16, 1], [25, 14], [0, 27], [11, 6], [23, 26], [14, 16], [27, 21], [21, 8], [22, 6], [1, 22], [21, 8], [12, 10], [22, 19], [6, 5], [3, 27], [4, 5], [21, 19], [15, 16], [9, 22], [19, 20], [10, 18], [10, 17], [15, 9], [5, 20], [18, 9], [17, 9], [13, 7], [3, 7], [5, 25], [6, 8]].
We maintain a union-find data structure over the nodes using union-by-rank with path compression.
Process every edge in the listed order with a union call. Then, for each query pair (a, b) in the listed order, take the find of both ends.
For every query, write '1' if a and b are in the same component and '0' otherwise, followed by the smallest-numbered node in a's component.
Queries: Queries:
[[0, 24], [21, 14], [4, 16], [23, 26], [20, 14], [2, 0], [2, 13], [6, 0], [10, 10], [4, 17], [12, 19], [26, 14], [15, 27]].
Give only the answer: concatenated pairs of (connect bit, canonical node of a), separated by spaces, in query order.

### Answer

0 0 1 0 1 0 1 0 1 0 0 2 0 2 1 0 1 0 1 0 1 0 1 0 1 0

### Prompt

A graph has 28 nodes numbered 0 through 27.
Its edges, added in this order, are: Edges:
[[24, 27], [17, 27], [15, 9], [25, 15], [1, 16], [17, 8], [5, 4], [6, 11], [6, 27], [7, 27], [3, 4], [11, 20], [1, 6], [4, 17], [19, 15], [24, 23], [4, 11], [20, 26], [21, 15], [18, 3], [10, 20], [20, 27], [22, 18], [17, 19], [3, 4], [16, 0], [23, 1], [15, 16], [5, 2], [21, 0], [19, 7], [27, 18], [26, 9], [26, 26], [15, 17], [6, 14], [27, 20], [2, 13]].
We maintain a union-find data structure over the nodes using union-by-rank with path compression.
Process every edge in the listed order with a union call. Then, for each query pair (a, b) in the listed order, take the find of both ends.
For every query, write '1' if a and b are in the same component and '0' otherwise, followed by the smallest-numbered node in a's component.
Queries: Queries:
[[3, 13], [0, 4], [8, 19], [24, 14], [11, 14], [3, 1], [17, 25], [21, 12], [15, 22], [24, 12], [15, 2], [19, 21], [12, 10]].
Give only the answer: concatenated pairs of (connect bit, canonical node of a), separated by spaces, in query order.

### Answer

1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 1 0 0 0 1 0 1 0 0 12

