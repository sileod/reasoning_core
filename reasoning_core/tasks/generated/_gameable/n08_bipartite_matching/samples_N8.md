# samples_N8

## Level 0 example 1

### Prompt

Instruction:
Find the largest matching in the bipartite graph. A matching is a set of edges, no two of which share an endpoint. Every edge connects a left vertex to a right vertex. Report the maximum number of edges in a matching as an integer.

Graph:
{'left': [0, 1, 2], 'right': [3, 4, 5], 'edges': [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 2)]}

The answer is an integer.

### Answer

3

## Level 0 example 2

### Prompt

Instruction:
Find the largest matching in the bipartite graph. A matching is a set of edges, no two of which share an endpoint. Every edge connects a left vertex to a right vertex. Report the maximum number of edges in a matching as an integer.

Graph:
{'left': [0, 1, 2], 'right': [3, 4, 5], 'edges': [(0, 0), (1, 0), (1, 1), (2, 0), (2, 2)]}

The answer is an integer.

### Answer

3

## Level 2 example 1

### Prompt

Instruction:
Find the largest matching in the bipartite graph. A matching is a set of edges, no two of which share an endpoint. Every edge connects a left vertex to a right vertex. Report the maximum number of edges in a matching as an integer.

Graph:
{'left': [0, 1, 2, 3, 4, 5, 6], 'right': [7, 8, 9, 10, 11, 12, 13], 'edges': [(0, 0), (0, 3), (0, 5), (0, 6), (1, 0), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 2), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 1), (4, 3), (4, 4), (4, 5), (4, 6), (5, 0), (5, 3), (5, 4), (5, 6), (6, 0), (6, 1), (6, 3), (6, 5), (6, 6)]}

The answer is an integer.

### Answer

7

## Level 2 example 2

### Prompt

Instruction:
Find the largest matching in the bipartite graph. A matching is a set of edges, no two of which share an endpoint. Every edge connects a left vertex to a right vertex. Report the maximum number of edges in a matching as an integer.

Graph:
{'left': [0, 1, 2, 3, 4, 5, 6], 'right': [7, 8, 9, 10, 11, 12, 13], 'edges': [(0, 0), (0, 2), (0, 3), (0, 6), (1, 0), (1, 1), (1, 2), (1, 4), (1, 5), (2, 2), (3, 1), (3, 2), (3, 4), (3, 5), (4, 0), (4, 2), (4, 3), (4, 4), (4, 5), (5, 0), (5, 1), (5, 2), (6, 0), (6, 1), (6, 2)]}

The answer is an integer.

### Answer

7

## Level 5 example 1

### Prompt

Instruction:
Find the largest matching in the bipartite graph. A matching is a set of edges, no two of which share an endpoint. Every edge connects a left vertex to a right vertex. Report the maximum number of edges in a matching as an integer.

Graph:
{'left': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'right': [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], 'edges': [(0, 1), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 10), (0, 11), (1, 0), (1, 1), (1, 3), (1, 4), (1, 7), (1, 8), (1, 9), (1, 10), (2, 0), (2, 2), (2, 5), (2, 6), (2, 9), (2, 10), (2, 12), (3, 3), (3, 6), (3, 7), (3, 8), (3, 11), (3, 12), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (5, 1), (5, 2), (5, 5), (5, 9), (5, 10), (5, 12), (6, 0), (6, 3), (6, 6), (7, 4), (7, 6), (7, 7), (7, 10), (8, 3), (8, 5), (8, 7), (8, 8), (8, 12), (9, 0), (9, 2), (9, 7), (9, 9), (9, 11), (10, 0), (10, 3), (10, 5), (10, 7), (10, 9), (11, 0), (11, 4), (11, 6), (11, 7), (11, 9), (12, 0), (12, 1), (12, 3), (12, 5), (12, 6), (12, 8), (12, 9), (12, 12)]}

The answer is an integer.

### Answer

13

## Level 5 example 2

### Prompt

Instruction:
Find the largest matching in the bipartite graph. A matching is a set of edges, no two of which share an endpoint. Every edge connects a left vertex to a right vertex. Report the maximum number of edges in a matching as an integer.

Graph:
{'left': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'right': [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], 'edges': [(0, 1), (0, 3), (0, 7), (0, 8), (0, 10), (1, 2), (1, 3), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 0), (2, 6), (2, 7), (2, 10), (3, 0), (3, 6), (3, 8), (3, 9), (3, 10), (3, 11), (4, 0), (4, 4), (4, 5), (4, 7), (4, 8), (4, 11), (5, 4), (5, 7), (5, 11), (6, 0), (6, 1), (6, 2), (6, 3), (6, 5), (6, 9), (6, 12), (7, 1), (7, 5), (7, 10), (7, 11), (7, 12), (8, 0), (8, 3), (8, 7), (8, 11), (9, 2), (9, 3), (9, 5), (9, 7), (9, 10), (9, 11), (10, 0), (10, 1), (10, 4), (10, 5), (10, 6), (11, 2), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (11, 11), (12, 7), (12, 9), (12, 10), (12, 11)]}

The answer is an integer.

### Answer

13
