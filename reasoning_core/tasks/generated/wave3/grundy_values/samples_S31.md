## Level 0

Prompt:
```
Consider the following impartial game made of independent heaps. A player on their turn moves in exactly one heap. The player unable to move loses.
Heap A has 3 tokens; its only legal moves split the heap into two non-empty heaps of unequal sizes, replacing it by those two.
Heap B has 3 tokens; its only legal moves split the heap into two non-empty heaps of unequal sizes, replacing it by those two.
Give each heap's Grundy value, in the heap order A B, followed by the Grundy value of the whole position. The answer is 3 space-separated non-negative integers.
```

Answer: 1 1 0

Prompt:
```
Consider the following impartial game made of independent heaps. A player on their turn moves in exactly one heap. The player unable to move loses.
Heap A has 4 tokens; its only legal moves split the heap into two non-empty heaps of unequal sizes, replacing it by those two.
Heap B has 3 tokens; its only legal moves split the heap into two non-empty heaps of unequal sizes, replacing it by those two.
Give each heap's Grundy value, in the heap order A B, followed by the Grundy value of the whole position. The answer is 3 space-separated non-negative integers.
```

Answer: 0 1 1

## Level 2

Prompt:
```
Consider the following impartial game made of independent heaps. A player on their turn moves in exactly one heap. The player unable to move loses.
Heap A has 6 tokens; its only legal moves remove any one, two, or three tokens.
Heap B has 3 tokens; its only legal moves remove any square number of tokens (1, 4, 9, ...).
Heap C has 3 tokens; its only legal moves remove any one, two, or three tokens.
Give each heap's Grundy value, in the heap order A B C, followed by the Grundy value of the whole position. The answer is 4 space-separated non-negative integers.
```

Answer: 2 1 3 0

Prompt:
```
Consider the following impartial game made of independent heaps. A player on their turn moves in exactly one heap. The player unable to move loses.
Heap A has 7 tokens; its only legal moves remove any square number of tokens (1, 4, 9, ...).
Heap B has 5 tokens; its only legal moves remove any one, two, or three tokens.
Heap C has 3 tokens; its only legal moves remove any square number of tokens (1, 4, 9, ...).
Give each heap's Grundy value, in the heap order A B C, followed by the Grundy value of the whole position. The answer is 4 space-separated non-negative integers.
```

Answer: 0 1 1 0

## Level 5

Prompt:
```
Consider the following impartial game made of independent heaps. A player on their turn moves in exactly one heap. The player unable to move loses.
Heap A has 6 tokens; its only legal moves split the heap into two non-empty heaps of unequal sizes, replacing it by those two.
Heap B has 11 tokens; its only legal moves remove any square number of tokens (1, 4, 9, ...).
Heap C has 9 tokens; its only legal moves remove any one, two, or three tokens.
Heap D has 6 tokens; its only legal moves split the heap into two non-empty heaps of unequal sizes, replacing it by those two.
Give each heap's Grundy value, in the heap order A B C D, followed by the Grundy value of the whole position. The answer is 5 space-separated non-negative integers.
```

Answer: 1 1 1 1 0

Prompt:
```
Consider the following impartial game made of independent heaps. A player on their turn moves in exactly one heap. The player unable to move loses.
Heap A has 2 tokens; its only legal moves remove any square number of tokens (1, 4, 9, ...).
Heap B has 6 tokens; its only legal moves remove any one, two, or three tokens.
Heap C has 10 tokens; its only legal moves split the heap into two non-empty heaps of unequal sizes, replacing it by those two.
Heap D has 3 tokens; its only legal moves remove any square number of tokens (1, 4, 9, ...).
Give each heap's Grundy value, in the heap order A B C D, followed by the Grundy value of the whole position. The answer is 5 space-separated non-negative integers.
```

Answer: 0 2 0 1 3
