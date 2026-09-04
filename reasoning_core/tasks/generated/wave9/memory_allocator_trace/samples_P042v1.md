Level 0

Prompt:
Strategy:
first_fit

Blocks:
[[0, 7], [0, 7], [0, 6]]

Ops:
[['alloc', 5], ['free', 0], ['alloc', 7], ['alloc', 1]]

The answer is the final free-memory layout: each free span in [lo,hi) form, block by block in the given order, joined by spaces.

Answer:
[1,7) [0,6)

Prompt:
Strategy:
first_fit

Blocks:
[[0, 3], [0, 9], [0, 9]]

Ops:
[['alloc', 4], ['free', 0], ['alloc', 5], ['free', 1]]

The answer is the final free-memory layout: each free span in [lo,hi) form, block by block in the given order, joined by spaces.

Answer:
[0,3) [0,9) [0,9)

Level 2

Prompt:
Strategy:
first_fit

Blocks:
[[0, 8], [0, 6], [0, 6], [0, 9]]

Ops:
[['alloc', 6], ['free', 0], ['alloc', 6], ['free', 1], ['alloc', 7], ['free', 2]]

The answer is the final free-memory layout: each free span in [lo,hi) form, block by block in the given order, joined by spaces.

Answer:
[0,8) [0,6) [0,6) [0,9)

Prompt:
Strategy:
first_fit

Blocks:
[[0, 4], [0, 4], [0, 11], [0, 3]]

Ops:
[['alloc', 10], ['free', 0], ['alloc', 3], ['alloc', 2], ['alloc', 8], ['free', 2]]

The answer is the final free-memory layout: each free span in [lo,hi) form, block by block in the given order, joined by spaces.

Answer:
[3,4) [0,4) [8,11) [0,3)

Level 5

Prompt:
Strategy:
first_fit

Blocks:
[[0, 6], [0, 8], [0, 3], [0, 7], [0, 14]]

Ops:
[['alloc', 12], ['alloc', 2], ['free', 1], ['free', 0], ['alloc', 4], ['alloc', 12], ['alloc', 4], ['alloc', 1], ['alloc', 2]]

The answer is the final free-memory layout: each free span in [lo,hi) form, block by block in the given order, joined by spaces.

Answer:
[5,6) [6,8) [0,3) [0,7) [12,14)

Prompt:
Strategy:
first_fit

Blocks:
[[0, 8], [0, 13], [0, 9], [0, 13], [0, 14]]

Ops:
[['alloc', 10], ['alloc', 7], ['alloc', 4], ['free', 1], ['alloc', 10], ['free', 3], ['alloc', 4], ['free', 2], ['free', 0]]

The answer is the final free-memory layout: each free span in [lo,hi) form, block by block in the given order, joined by spaces.

Answer:
[4,8) [0,13) [0,9) [0,13) [0,14)
