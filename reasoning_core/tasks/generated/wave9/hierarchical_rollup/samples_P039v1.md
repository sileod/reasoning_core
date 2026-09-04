# HierarchicalRollup samples

## Level 0

**Example 1**

Lines:
['Hierarchy: A -> B, C; B -> D, E', 'Per-node values, listed for every node: A=2, B=5, C=5, D=7, E=4.', "An override forces the computed subtotal of 'B' to equal 4."]

Question:
What is the rolled-up subtotal (sum of all values in its subtree, with the override applied) of the subtree rooted at 'E'? The answer is an integer.

Answer: 4

**Example 2**

Lines:
['Hierarchy: A -> B; B -> C; C -> D; D -> E', 'Per-node values, listed for every node: A=0, B=7, C=9, D=8, E=6.', "The requested subtotal excludes the entire subtree under child 'E' of 'D'."]

Question:
What is the rolled-up subtotal of the subtree rooted at 'D', excluding the entire subtree under its child 'E'? The answer is an integer.

Answer: 8

## Level 2

**Example 1**

Lines:
['Hierarchy: A -> B, C; B -> D; D -> E, F; E -> G; G -> H; H -> I', 'Per-node values, listed for every node: A=3, B=2, C=16, D=16, E=9, F=6, G=18, H=5, I=11.', "An override forces the computed subtotal of 'D' to equal 14."]

Question:
What is the rolled-up subtotal (sum of all values in its subtree, with the override applied) of the subtree rooted at 'I'? The answer is an integer.

Answer: 11

**Example 2**

Lines:
['Hierarchy: A -> B; B -> C, D; C -> E, F; D -> G; F -> H, I', 'Per-node values, listed for every node: A=0, B=18, C=14, D=2, E=16, F=1, G=13, H=13, I=3.', "An override forces the computed subtotal of 'E' to equal 16."]

Question:
What is the rolled-up subtotal (sum of all values in its subtree, with the override applied) of the subtree rooted at 'E'? The answer is an integer.

Answer: 16

## Level 5

**Example 1**

Lines:
['Hierarchy: A -> B; B -> C; C -> D; D -> E; E -> F; F -> G; G -> H; H -> I; I -> J; J -> K; K -> L; L -> M; M -> N; N -> O', 'Per-node values, listed for every node: A=29, B=4, C=24, D=14, E=8, F=7, G=17, H=21, I=4, J=18, K=6, L=15, M=5, N=17, O=4.', "The requested subtotal excludes the entire subtree under child 'B' of 'A'."]

Question:
What is the rolled-up subtotal of the subtree rooted at 'A', excluding the entire subtree under its child 'B'? The answer is an integer.

Answer: 29

**Example 2**

Lines:
['Hierarchy: A -> B; B -> C; C -> D; D -> E; E -> F; F -> G; G -> H; H -> I; I -> J; J -> K; K -> L; L -> M; M -> N; N -> O', 'Per-node values, listed for every node: A=14, B=24, C=7, D=14, E=27, F=6, G=13, H=1, I=28, J=0, K=15, L=24, M=6, N=17, O=0.', "An override forces the computed subtotal of 'E' to equal 4."]

Question:
What is the rolled-up subtotal (sum of all values in its subtree, with the override applied) of the subtree rooted at 'J'? The answer is an integer.

Answer: 62
