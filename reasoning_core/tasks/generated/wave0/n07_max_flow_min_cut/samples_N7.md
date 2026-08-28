# Level 0
## Prompt
Nodes:
[0, 1, 2, 3, 4]

Source:
0

Sink:
4

Edges:
[(0, 3, 4), (1, 0, 3), (1, 4, 5), (2, 0, 2), (2, 3, 3), (2, 4, 6), (3, 1, 5), (3, 4, 5), (4, 2, 1)]

Consider a directed graph with capacities on every edge. Source is node 0 and sink is node 4. Compute the maximum flow value from source to sink, then list the nodes on the source side of the minimum cut in ascending order. The answer is the flow value followed by a space and the comma-separated nodes of the source side of the cut, e.g. "5 0,1,3".
## Answer
4 0

## Prompt
Nodes:
[0, 1, 2, 3, 4]

Source:
0

Sink:
4

Edges:
[(0, 3, 3), (1, 2, 5), (1, 4, 1), (2, 3, 3), (3, 1, 3), (3, 2, 3), (3, 4, 2), (4, 1, 2), (4, 2, 5), (4, 3, 5)]

Consider a directed graph with capacities on every edge. Source is node 0 and sink is node 4. Compute the maximum flow value from source to sink, then list the nodes on the source side of the minimum cut in ascending order. The answer is the flow value followed by a space and the comma-separated nodes of the source side of the cut, e.g. "5 0,1,3".
## Answer
3 0,1,2,3

# Level 2
## Prompt
Nodes:
[0, 1, 2, 3, 4, 5, 6, 7, 8]

Source:
0

Sink:
8

Edges:
[(0, 2, 8), (0, 4, 7), (0, 5, 6), (0, 6, 5), (0, 7, 8), (1, 2, 1), (1, 3, 1), (1, 6, 2), (1, 7, 1), (2, 1, 2), (2, 4, 3), (2, 5, 6), (2, 6, 4), (2, 7, 5), (2, 8, 7), (3, 0, 2), (3, 1, 5), (3, 4, 2), (3, 5, 8), (3, 6, 5), (4, 1, 1), (4, 2, 3), (4, 3, 3), (4, 5, 2), (4, 6, 3), (4, 7, 3), (5, 0, 7), (5, 2, 7), (5, 4, 8), (5, 7, 1), (5, 8, 8), (6, 1, 8), (6, 4, 4), (6, 7, 3), (7, 0, 5), (7, 1, 5), (7, 2, 2), (7, 4, 1), (7, 5, 5), (7, 6, 7), (8, 0, 8), (8, 2, 8), (8, 5, 7), (8, 6, 4)]

Consider a directed graph with capacities on every edge. Source is node 0 and sink is node 8. Compute the maximum flow value from source to sink, then list the nodes on the source side of the minimum cut in ascending order. The answer is the flow value followed by a space and the comma-separated nodes of the source side of the cut, e.g. "5 0,1,3".
## Answer
15 0,1,2,3,4,5,6,7

## Prompt
Nodes:
[0, 1, 2, 3, 4, 5, 6, 7, 8]

Source:
0

Sink:
8

Edges:
[(0, 2, 8), (0, 3, 6), (0, 6, 7), (1, 2, 7), (1, 3, 1), (1, 4, 5), (1, 5, 4), (1, 6, 3), (1, 7, 4), (2, 0, 8), (2, 1, 6), (2, 3, 7), (2, 4, 8), (2, 5, 1), (2, 6, 3), (2, 8, 4), (3, 1, 2), (3, 2, 7), (3, 7, 4), (4, 0, 5), (4, 2, 6), (4, 3, 1), (4, 5, 6), (4, 6, 4), (4, 7, 7), (5, 0, 3), (5, 2, 8), (5, 3, 6), (5, 7, 2), (6, 1, 2), (6, 3, 7), (6, 5, 8), (6, 7, 2), (7, 1, 3), (7, 2, 1), (7, 6, 6), (7, 8, 6), (8, 0, 4), (8, 1, 3), (8, 2, 8), (8, 4, 2), (8, 5, 5), (8, 6, 1), (8, 7, 7)]

Consider a directed graph with capacities on every edge. Source is node 0 and sink is node 8. Compute the maximum flow value from source to sink, then list the nodes on the source side of the minimum cut in ascending order. The answer is the flow value followed by a space and the comma-separated nodes of the source side of the cut, e.g. "5 0,1,3".
## Answer
10 0,1,2,3,4,5,6,7

# Level 5
## Prompt
Nodes:
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

Source:
0

Sink:
14

Edges:
[(0, 1, 7), (0, 2, 10), (0, 4, 8), (0, 5, 1), (0, 6, 3), (0, 9, 2), (0, 11, 9), (0, 12, 1), (1, 0, 11), (1, 2, 8), (1, 3, 6), (1, 4, 9), (1, 6, 8), (1, 7, 3), (1, 9, 6), (1, 10, 9), (1, 11, 7), (1, 12, 3), (1, 13, 6), (2, 0, 11), (2, 3, 9), (2, 4, 9), (2, 6, 2), (2, 7, 11), (2, 8, 10), (2, 10, 9), (2, 11, 5), (2, 13, 11), (2, 14, 8), (3, 0, 3), (3, 1, 3), (3, 2, 7), (3, 4, 4), (3, 6, 9), (3, 7, 5), (3, 8, 7), (3, 9, 2), (3, 10, 1), (3, 11, 1), (3, 12, 6), (3, 13, 6), (3, 14, 7), (4, 0, 8), (4, 1, 9), (4, 2, 11), (4, 3, 1), (4, 5, 11), (4, 6, 6), (4, 8, 8), (4, 10, 1), (4, 11, 3), (4, 12, 2), (4, 13, 2), (4, 14, 3), (5, 0, 6), (5, 1, 11), (5, 2, 5), (5, 3, 11), (5, 4, 1), (5, 6, 7), (5, 7, 11), (5, 8, 4), (5, 10, 7), (5, 12, 9), (5, 13, 2), (6, 0, 10), (6, 1, 7), (6, 2, 6), (6, 4, 9), (6, 5, 8), (6, 7, 8), (6, 8, 6), (6, 9, 4), (6, 11, 4), (6, 12, 5), (6, 13, 6), (7, 0, 4), (7, 1, 9), (7, 2, 9), (7, 3, 3), (7, 5, 4), (7, 6, 7), (7, 8, 4), (7, 9, 1), (7, 11, 8), (7, 13, 10), (7, 14, 6), (8, 0, 4), (8, 1, 10), (8, 2, 5), (8, 3, 8), (8, 6, 5), (8, 7, 7), (8, 10, 1), (8, 12, 10), (8, 14, 9), (9, 0, 7), (9, 2, 8), (9, 3, 1), (9, 5, 8), (9, 6, 7), (9, 8, 8), (9, 13, 3), (9, 14, 5), (10, 0, 10), (10, 2, 2), (10, 3, 1), (10, 5, 11), (10, 7, 10), (10, 8, 8), (10, 11, 7), (10, 13, 2), (10, 14, 9), (11, 0, 6), (11, 1, 8), (11, 3, 3), (11, 4, 11), (11, 5, 11), (11, 6, 9), (11, 7, 9), (11, 8, 4), (11, 10, 7), (11, 12, 7), (11, 13, 4), (11, 14, 9), (12, 0, 3), (12, 1, 5), (12, 2, 4), (12, 3, 5), (12, 4, 2), (12, 5, 6), (12, 6, 1), (12, 8, 11), (12, 9, 4), (12, 10, 11), (12, 11, 2), (12, 13, 3), (12, 14, 3), (13, 0, 8), (13, 3, 8), (13, 4, 9), (13, 5, 11), (13, 6, 5), (13, 7, 2), (13, 8, 2), (13, 9, 7), (13, 11, 9), (13, 12, 1), (13, 14, 10), (14, 1, 1), (14, 4, 7), (14, 5, 9), (14, 6, 7), (14, 7, 1), (14, 8, 3), (14, 9, 8), (14, 11, 1)]

Consider a directed graph with capacities on every edge. Source is node 0 and sink is node 14. Compute the maximum flow value from source to sink, then list the nodes on the source side of the minimum cut in ascending order. The answer is the flow value followed by a space and the comma-separated nodes of the source side of the cut, e.g. "5 0,1,3".
## Answer
41 0

## Prompt
Nodes:
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

Source:
0

Sink:
14

Edges:
[(0, 1, 10), (0, 2, 7), (0, 3, 10), (0, 4, 9), (0, 5, 2), (0, 6, 7), (0, 7, 5), (0, 9, 8), (0, 10, 1), (0, 11, 7), (0, 12, 10), (0, 13, 10), (1, 0, 7), (1, 2, 2), (1, 3, 2), (1, 4, 9), (1, 5, 9), (1, 6, 2), (1, 7, 4), (1, 8, 2), (1, 9, 9), (1, 10, 4), (1, 11, 9), (1, 12, 2), (1, 13, 9), (1, 14, 10), (2, 0, 4), (2, 3, 2), (2, 4, 3), (2, 5, 6), (2, 7, 11), (2, 10, 10), (2, 11, 6), (2, 12, 9), (2, 13, 4), (2, 14, 10), (3, 0, 9), (3, 1, 3), (3, 2, 4), (3, 4, 9), (3, 5, 3), (3, 7, 1), (3, 8, 3), (3, 9, 6), (3, 11, 9), (3, 12, 10), (3, 13, 5), (4, 0, 9), (4, 1, 7), (4, 2, 10), (4, 3, 5), (4, 5, 9), (4, 6, 4), (4, 7, 11), (4, 8, 4), (4, 9, 4), (4, 10, 2), (4, 11, 3), (4, 12, 9), (4, 13, 8), (4, 14, 3), (5, 1, 10), (5, 2, 1), (5, 4, 10), (5, 8, 4), (5, 10, 3), (5, 11, 1), (5, 12, 10), (5, 14, 10), (6, 0, 7), (6, 1, 9), (6, 2, 11), (6, 3, 8), (6, 4, 2), (6, 5, 7), (6, 7, 6), (6, 8, 4), (6, 10, 1), (6, 12, 5), (6, 13, 1), (6, 14, 7), (7, 0, 4), (7, 2, 11), (7, 3, 2), (7, 5, 2), (7, 6, 7), (7, 9, 3), (7, 10, 10), (7, 11, 3), (7, 12, 1), (7, 13, 9), (7, 14, 9), (8, 0, 10), (8, 1, 2), (8, 2, 9), (8, 3, 11), (8, 4, 6), (8, 6, 6), (8, 7, 10), (8, 9, 6), (8, 10, 10), (8, 11, 5), (8, 12, 9), (8, 13, 1), (8, 14, 11), (9, 0, 7), (9, 1, 3), (9, 2, 1), (9, 3, 4), (9, 5, 5), (9, 7, 2), (9, 10, 11), (9, 11, 11), (9, 12, 9), (9, 14, 11), (10, 0, 10), (10, 1, 3), (10, 2, 4), (10, 3, 6), (10, 4, 2), (10, 5, 3), (10, 6, 4), (10, 8, 10), (10, 9, 2), (10, 11, 3), (10, 12, 3), (10, 13, 1), (10, 14, 10), (11, 1, 3), (11, 2, 11), (11, 5, 9), (11, 6, 1), (11, 8, 11), (11, 9, 7), (11, 10, 3), (11, 12, 11), (11, 14, 11), (12, 1, 2), (12, 4, 4), (12, 5, 6), (12, 6, 2), (12, 8, 1), (12, 9, 11), (12, 11, 7), (12, 13, 8), (12, 14, 3), (13, 0, 6), (13, 1, 5), (13, 2, 6), (13, 3, 6), (13, 4, 1), (13, 5, 10), (13, 8, 6), (13, 9, 9), (13, 10, 7), (13, 14, 1), (14, 0, 8), (14, 4, 5), (14, 6, 7), (14, 7, 7), (14, 8, 1), (14, 9, 9), (14, 10, 5), (14, 11, 1), (14, 12, 8), (14, 13, 5)]

Consider a directed graph with capacities on every edge. Source is node 0 and sink is node 14. Compute the maximum flow value from source to sink, then list the nodes on the source side of the minimum cut in ascending order. The answer is the flow value followed by a space and the comma-separated nodes of the source side of the cut, e.g. "5 0,1,3".
## Answer
86 0
