### Level 0



Prompt:
We maintain a min-heap as an array. Operations use sift-down for the smallest-child convention (the root is the minimum, children at 2i+1 and 2i+2), and sift-up for the parent at (i-1)//2. Start from the array [11, 4, 28, 5] (heapified). Then apply, in order: replace-root 22; replace-root 10; pop-min. Give the resulting array.
The answer is a comma-separated list of integers.

Answer:
11,22,28



Prompt:
We maintain a min-heap as an array. Operations use sift-down for the smallest-child convention (the root is the minimum, children at 2i+1 and 2i+2), and sift-up for the parent at (i-1)//2. Start from the array [20, 7, 25, 29] (heapified). Then apply, in order: pop-min; push 10; pop-min. Give the resulting array.
The answer is a comma-separated list of integers.

Answer:
20,29,25
### Level 2



Prompt:
We maintain a min-heap as an array. Operations use sift-down for the smallest-child convention (the root is the minimum, children at 2i+1 and 2i+2), and sift-up for the parent at (i-1)//2. Start from the array [6, 64, 18, 58, 31, 7] (heapified). Then apply, in order: pop-min; push 3; push 29; pop-min; replace-root 9; pop-min; replace-root 65. Give the resulting array.
The answer is a comma-separated list of integers.

Answer:
29,31,65,58,64



Prompt:
We maintain a min-heap as an array. Operations use sift-down for the smallest-child convention (the root is the minimum, children at 2i+1 and 2i+2), and sift-up for the parent at (i-1)//2. Start from the array [67, 64, 1, 31, 1, 3] (heapified). Then apply, in order: pop-min; replace-root 3; push 53; pop-min; push 51; push 3; push 27. Give the resulting array.
The answer is a comma-separated list of integers.

Answer:
3,27,3,31,64,53,51,67
### Level 5


Prompt:
We maintain a min-heap as an array. Operations use sift-down for the smallest-child convention (the root is the minimum, children at 2i+1 and 2i+2), and sift-up for the parent at (i-1)//2. Start from the array [67, 54, 0, 68, 96, 50, 74, 85, 98] (heapified). Then apply, in order: replace-root 79; push 31; push 91; replace-root 111; push 111; pop-min; replace-root 101; pop-min; pop-min; pop-min; push 65; pop-min; pop-min. Give the value at index 0 of the resulting array.
The answer is a single integer.

Answer:
85


Prompt:
We maintain a min-heap as an array. Operations use sift-down for the smallest-child convention (the root is the minimum, children at 2i+1 and 2i+2), and sift-up for the parent at (i-1)//2. Start from the array [35, 117, 121, 75, 63, 42, 82, 129, 42] (heapified). Then apply, in order: push 98; push 80; push 15; push 91; push 94; push 15; push 120; replace-root 110; push 111; pop-min; pop-min; replace-root 69; push 112. Give the value at index 10 of the resulting array.
The answer is a single integer.

Answer:
129
