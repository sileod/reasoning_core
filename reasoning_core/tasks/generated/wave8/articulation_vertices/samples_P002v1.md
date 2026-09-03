# ArticulationVertices samples (P002v1)

## Level 0

We have an undirected graph with n = 7 vertices numbered 0 through 6. Its edges are:
[(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 2)]

A vertex is an articulation vertex if removing it (and all edges incident to it) increases the number of connected components of the graph.

Output all articulation vertices of this graph, in ascending order of their labels, as the answer. If the graph has no articulation vertices, the answer is an empty list.

Use standard depth-first-search articulation-point finding to determine the answer.

Answer: [1, 2, 3, 4]

We have an undirected graph with n = 7 vertices numbered 0 through 6. Its edges are:
[(1, 0), (2, 1), (3, 0), (4, 3), (5, 1), (6, 3)]

A vertex is an articulation vertex if removing it (and all edges incident to it) increases the number of connected components of the graph.

Output all articulation vertices of this graph, in ascending order of their labels, as the answer. If the graph has no articulation vertices, the answer is an empty list.

Use standard depth-first-search articulation-point finding to determine the answer.

Answer: [0, 1, 3]

## Level 2

We have an undirected graph with n = 13 vertices numbered 0 through 12. Its edges are:
[(0, 3), (1, 0), (1, 11), (2, 1), (2, 6), (3, 1), (4, 3), (4, 8), (5, 3), (6, 3), (7, 0), (7, 12), (8, 2), (9, 8), (10, 6), (10, 11), (11, 5), (12, 11)]

A vertex is an articulation vertex if removing it (and all edges incident to it) increases the number of connected components of the graph.

Output all articulation vertices of this graph, in ascending order of their labels, as the answer. If the graph has no articulation vertices, the answer is an empty list.

Use standard depth-first-search articulation-point finding to determine the answer.

Answer: [8]

We have an undirected graph with n = 13 vertices numbered 0 through 12. Its edges are:
[(1, 0), (1, 2), (1, 11), (2, 0), (3, 2), (4, 0), (4, 5), (4, 10), (4, 11), (5, 1), (5, 10), (6, 3), (6, 12), (7, 3), (8, 5), (9, 4), (10, 9), (11, 7), (11, 12), (12, 3)]

A vertex is an articulation vertex if removing it (and all edges incident to it) increases the number of connected components of the graph.

Output all articulation vertices of this graph, in ascending order of their labels, as the answer. If the graph has no articulation vertices, the answer is an empty list.

Use standard depth-first-search articulation-point finding to determine the answer.

Answer: [5]

## Level 5

We have an undirected graph with n = 22 vertices numbered 0 through 21. Its edges are:
[(1, 0), (2, 0), (3, 0), (4, 3), (5, 1), (6, 5), (7, 3), (8, 6), (9, 3), (10, 8), (11, 2), (12, 9), (13, 7), (14, 11), (15, 14), (16, 8), (17, 10), (18, 7), (19, 12), (19, 21), (20, 12), (21, 3)]

A vertex is an articulation vertex if removing it (and all edges incident to it) increases the number of connected components of the graph.

Output all articulation vertices of this graph, in ascending order of their labels, as the answer. If the graph has no articulation vertices, the answer is an empty list.

Use standard depth-first-search articulation-point finding to determine the answer.

Answer: [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 14]

We have an undirected graph with n = 22 vertices numbered 0 through 21. Its edges are:
[(0, 10), (0, 14), (1, 0), (2, 1), (3, 0), (3, 6), (4, 2), (4, 9), (5, 1), (6, 1), (6, 15), (7, 5), (8, 1), (9, 2), (9, 19), (10, 5), (11, 2), (12, 4), (13, 5), (14, 6), (15, 8), (16, 0), (17, 13), (18, 15), (19, 12), (20, 2), (21, 13)]

A vertex is an articulation vertex if removing it (and all edges incident to it) increases the number of connected components of the graph.

Output all articulation vertices of this graph, in ascending order of their labels, as the answer. If the graph has no articulation vertices, the answer is an empty list.

Use standard depth-first-search articulation-point finding to determine the answer.

Answer: [0, 1, 2, 5, 13, 15]
