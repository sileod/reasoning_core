# Level 0

Targets (Dependency List):
A
B
C: A, B
D: A, B
E: B, C

Changed Sources:
B, C

List every target that must be rebuilt: each changed source itself, plus every target that directly or transitively depends on a changed source. Answer as a single comma-separated list sorted in alphabetical order.

Answer: B, C, D, E

Targets (Dependency List):
A
B: A
C
D: A, B
E: B, D

Changed Sources:
B, D

List every target that must be rebuilt: each changed source itself, plus every target that directly or transitively depends on a changed source. Answer as a single comma-separated list sorted in alphabetical order.

Answer: B, D, E

# Level 2

Targets (Dependency List):
A
B: A
C: A, B
D: A, C
E: C
F: B, D, E
G: C, E
H: B, D, G
I: C, E, F

Changed Sources:
G, H, I

List every target that must be rebuilt: each changed source itself, plus every target that directly or transitively depends on a changed source. Answer as a single comma-separated list sorted in alphabetical order.

Answer: G, H, I

Targets (Dependency List):
A
B
C: A
D: A, B
E: A, B, C
F: B, C, D
G: C, F
H: A, B, C, D, E, G
I: A, B, C, D, F, G, H

Changed Sources:
B, F, I

List every target that must be rebuilt: each changed source itself, plus every target that directly or transitively depends on a changed source. Answer as a single comma-separated list sorted in alphabetical order.

Answer: B, D, E, F, G, H, I

# Level 5

Targets (Dependency List):
A
B: A
C
D: A
E: A, B, D
F: C, D, E
G: A, B, D, E, F
H: A, B, C, D, E, F, G
I: C, D, F, G
J: A, B, D, E, F, G, H
K: A, C, E, G, H, I
L: B, E, F, G, I, J, K
M: B, D, I, J, L
N: B, I, K
O: B, L

Changed Sources:
C, J, K, O

List every target that must be rebuilt: each changed source itself, plus every target that directly or transitively depends on a changed source. Answer as a single comma-separated list sorted in alphabetical order.

Answer: C, F, G, H, I, J, K, L, M, N, O

Targets (Dependency List):
A
B: A
C: A, B
D: B
E: A, B, C, D
F: A
G: A, B, F
H: A, B, E, F, G
I: C, F, G, H
J: A, C, E, G, I
K: B
L: C, D, G, I, J
M: A, C, D, E, F, I, K
N: E, G, H, I, J
O: A, B, C, D, E, G, J, L, M, N

Changed Sources:
E, J, N, O

List every target that must be rebuilt: each changed source itself, plus every target that directly or transitively depends on a changed source. Answer as a single comma-separated list sorted in alphabetical order.

Answer: E, H, I, J, L, M, N, O
