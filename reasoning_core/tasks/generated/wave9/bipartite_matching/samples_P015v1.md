# Samples for P015v1 (bipartite_matching)

## Level 0

### Example 1

Prompt:
Bipartite Graph:
6 left vertices (L0..L5) and 7 right vertices (R0..R6).
Edges (left vertex: adjacent right vertices):
L0: R2, R5, R6
L1: R1, R2, R3
L2: R1, R3, R5
L3: R0, R2, R4
L4: R0, R2, R4
L5: R0, R1, R2, R3, R6

Canonical Maximum Matching:
augment along shortest augmented paths, visiting unmatched left vertices and neighbors in increasing label order, until none remain

Question:
What is the partner of L0 in the canonical maximum matching described above?

Answer Format:
write only the partner's right index (the number k for Rk), or None if the queried left vertex is unmatched.

Answer:
2

### Example 2

Prompt:
Bipartite Graph:
4 left vertices (L0..L3) and 7 right vertices (R0..R6).
Edges (left vertex: adjacent right vertices):
L0: R1, R3, R6
L1: R0, R1, R2, R4, R5
L2: R0, R2, R4, R6
L3: R0, R4, R5

Canonical Maximum Matching:
augment along shortest augmented paths, visiting unmatched left vertices and neighbors in increasing label order, until none remain

Question:
What is the partner of L0 in the canonical maximum matching described above?

Answer Format:
write only the partner's right index (the number k for Rk), or None if the queried left vertex is unmatched.

Answer:
1

## Level 2

### Example 1

Prompt:
Bipartite Graph:
9 left vertices (L0..L8) and 7 right vertices (R0..R6).
Edges (left vertex: adjacent right vertices):
L0: R0, R1, R4, R5
L1: R0, R1, R4, R6
L2: R1, R3, R6
L3: R0, R1, R2, R5, R6
L4: R0, R1, R2, R4, R5, R6
L5: R0, R1, R2, R3, R4, R5
L6: R1, R2, R5
L7: R0, R1, R4, R5, R6
L8: R4, R5, R6

Canonical Maximum Matching:
augment along shortest augmented paths, visiting unmatched left vertices and neighbors in increasing label order, until none remain

Question:
What is the partner of L8 in the canonical maximum matching described above?

Answer Format:
write only the partner's right index (the number k for Rk), or None if the queried left vertex is unmatched.

Answer:
None

### Example 2

Prompt:
Bipartite Graph:
7 left vertices (L0..L6) and 6 right vertices (R0..R5).
Edges (left vertex: adjacent right vertices):
L0: R3, R4, R5
L1: R0, R1, R2, R3, R5
L2: R0, R3
L3: R0, R1, R2, R3, R4
L4: R1, R3
L5: R0, R2, R3, R4
L6: R0, R4

Canonical Maximum Matching:
augment along shortest augmented paths, visiting unmatched left vertices and neighbors in increasing label order, until none remain

Question:
What is the partner of L5 in the canonical maximum matching described above?

Answer Format:
write only the partner's right index (the number k for Rk), or None if the queried left vertex is unmatched.

Answer:
2

## Level 5

### Example 1

Prompt:
Bipartite Graph:
12 left vertices (L0..L11) and 12 right vertices (R0..R11).
Edges (left vertex: adjacent right vertices):
L0: R3, R4, R6, R7, R10, R11
L1: R0, R1, R7, R9, R11
L2: R0, R4, R6, R7, R8
L3: R0, R1, R2, R3, R4, R6, R7, R8, R10
L4: R0, R2, R3, R5, R7, R8, R9, R11
L5: R0, R1, R2, R5, R6, R7, R10
L6: R0, R4, R5, R6, R7, R9, R10, R11
L7: R2, R3, R4, R5, R6, R7, R8, R9, R10
L8: R1, R2, R3, R6, R10, R11
L9: R0, R3, R5, R6, R8, R10
L10: R4, R6, R8, R9, R11
L11: R1, R2, R4, R6, R7, R8, R9, R10, R11

Canonical Maximum Matching:
augment along shortest augmented paths, visiting unmatched left vertices and neighbors in increasing label order, until none remain

Question:
What is the partner of L4 in the canonical maximum matching described above?

Answer Format:
write only the partner's right index (the number k for Rk), or None if the queried left vertex is unmatched.

Answer:
2

### Example 2

Prompt:
Bipartite Graph:
9 left vertices (L0..L8) and 11 right vertices (R0..R10).
Edges (left vertex: adjacent right vertices):
L0: R0, R3, R4, R6, R8, R9
L1: R2, R4, R6, R7, R8, R9, R10
L2: R0, R1, R4, R6, R8, R9
L3: R0, R3, R6, R8, R9, R10
L4: R0, R1, R2, R4, R6, R7, R8
L5: R1, R3, R4, R6, R9, R10
L6: R0, R1, R8, R9, R10
L7: R0, R2, R3, R4, R6, R8, R9, R10
L8: R0, R2, R3, R4, R5, R7, R8, R9, R10

Canonical Maximum Matching:
augment along shortest augmented paths, visiting unmatched left vertices and neighbors in increasing label order, until none remain

Question:
What is the partner of L5 in the canonical maximum matching described above?

Answer Format:
write only the partner's right index (the number k for Rk), or None if the queried left vertex is unmatched.

Answer:
6
