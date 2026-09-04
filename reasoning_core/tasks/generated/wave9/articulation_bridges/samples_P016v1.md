# Level 0

Prompt:
Vertices:
[0, 1, 2, 3, 4, 5, 6, 7, 8]

Edges:
[(0, 2), (0, 3), (1, 4), (1, 7), (2, 3), (2, 7), (4, 7), (5, 6), (5, 8), (6, 7), (7, 8)]

Find all articulation vertices (vertices whose removal increases the number of connected components) using the DFS low-link algorithm. List them as a comma-separated sequence in increasing order. The answer is that comma-separated list, or the single word none if there are none.

Answer:
2,7

Prompt:
Vertices:
[0, 1, 2, 3, 4, 5, 6, 7, 8]

Edges:
[(0, 3), (1, 2), (1, 3), (1, 8), (2, 5), (2, 7), (2, 8), (3, 8), (4, 6), (4, 7), (5, 8)]

Find all articulation vertices (vertices whose removal increases the number of connected components) using the DFS low-link algorithm. List them as a comma-separated sequence in increasing order. The answer is that comma-separated list, or the single word none if there are none.

Answer:
2,3,4,7

# Level 2

Prompt:
Vertices:
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

Edges:
[(0, 1), (0, 8), (0, 11), (0, 12), (1, 10), (1, 11), (1, 12), (2, 4), (2, 7), (2, 8), (2, 10), (2, 11), (3, 7), (3, 11), (4, 5), (4, 8), (5, 8), (5, 11), (5, 12), (6, 7), (6, 8), (6, 9), (6, 11), (7, 11), (8, 10), (8, 12), (10, 12)]

Find all bridge edges (edges whose removal increases the number of connected components) using the DFS low-link algorithm. List each as u-v with the smaller endpoint first, and the edges separated by semicolons in increasing lexicographic order. The answer is that semicolon-separated list, or the single word none if there are none.

Answer:
6-9

Prompt:
Vertices:
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

Edges:
[(0, 10), (1, 3), (1, 4), (1, 7), (1, 9), (1, 10), (1, 11), (2, 7), (2, 8), (2, 11), (3, 7), (3, 10), (3, 12), (4, 8), (4, 9), (4, 10), (5, 7), (5, 9), (6, 8), (6, 9), (6, 12), (7, 8), (8, 10), (8, 12), (10, 11), (11, 12)]

Find all articulation vertices (vertices whose removal increases the number of connected components) using the DFS low-link algorithm. List them as a comma-separated sequence in increasing order. The answer is that comma-separated list, or the single word none if there are none.

Answer:
10

# Level 5

Prompt:
Vertices:
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

Edges:
[(0, 1), (0, 3), (0, 4), (0, 5), (0, 13), (0, 17), (1, 2), (1, 3), (1, 5), (1, 6), (1, 8), (1, 11), (1, 13), (1, 15), (1, 17), (1, 18), (2, 6), (2, 7), (2, 10), (2, 11), (2, 15), (2, 16), (3, 4), (3, 7), (3, 12), (3, 15), (3, 18), (4, 8), (4, 9), (4, 10), (4, 11), (4, 14), (5, 10), (5, 18), (6, 8), (6, 13), (6, 14), (6, 18), (7, 9), (7, 14), (8, 9), (8, 10), (8, 14), (8, 15), (8, 17), (8, 18), (9, 14), (9, 16), (9, 17), (10, 11), (10, 13), (10, 14), (10, 17), (11, 18), (13, 15), (13, 16), (13, 17), (15, 18)]

Find all articulation vertices (vertices whose removal increases the number of connected components) using the DFS low-link algorithm. List them as a comma-separated sequence in increasing order. The answer is that comma-separated list, or the single word none if there are none.

Answer:
3

Prompt:
Vertices:
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

Edges:
[(0, 2), (0, 9), (0, 12), (0, 17), (1, 2), (1, 7), (1, 10), (1, 18), (2, 4), (2, 6), (2, 8), (2, 13), (2, 16), (2, 17), (2, 18), (3, 4), (3, 10), (3, 12), (3, 17), (4, 5), (4, 6), (4, 9), (4, 10), (4, 14), (4, 16), (5, 6), (5, 8), (5, 9), (5, 12), (5, 15), (5, 18), (6, 9), (6, 11), (6, 12), (6, 16), (6, 18), (7, 8), (7, 10), (7, 12), (7, 14), (7, 16), (7, 17), (8, 9), (8, 11), (8, 12), (8, 18), (9, 11), (11, 12), (11, 14), (11, 15), (11, 16), (11, 18), (12, 14), (12, 17), (14, 15), (14, 16), (14, 17), (15, 16), (15, 18), (17, 18)]

After removing vertex 2 (and all its incident edges), the remaining graph splits into some number of connected components. What is that number?
The answer is a single non-negative integer.

Answer:
2
