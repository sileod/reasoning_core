# Level 0

## Example 1 (level 0)

**Prompt**
```
Time positions are the integers 0, 1, ..., 5.
Each row shows, per position, whether that proposition holds (1) or not (0):
  a: 100111
  b: 010010
Evaluate the temporal formula
  (X (b)) /\(X (b))
over the trace above. Next (X) moves one step forward within the trace; eventually (F) requires holding at some current-or-later position; always (G) requires holding at every current-or-later position; until (U) requires the right formula to hold at some later-or-equal position with the left formula holding at every position strictly before it.
The answer is the sorted, comma-separated list of all positions where the formula is true; if it is true nowhere, the answer is the single word "none".
```

**Answer**
```
0,3
```

## Example 2 (level 0)

**Prompt**
```
Time positions are the integers 0, 1, ..., 5.
Each row shows, per position, whether that proposition holds (1) or not (0):
  a: 011000
  b: 011100
Evaluate the temporal formula
  X (! (a))
over the trace above. Next (X) moves one step forward within the trace; eventually (F) requires holding at some current-or-later position; always (G) requires holding at every current-or-later position; until (U) requires the right formula to hold at some later-or-equal position with the left formula holding at every position strictly before it.
The answer is the sorted, comma-separated list of all positions where the formula is true; if it is true nowhere, the answer is the single word "none".
```

**Answer**
```
2,3,4
```

# Level 2

## Example 1 (level 2)

**Prompt**
```
Time positions are the integers 0, 1, ..., 9.
Each row shows, per position, whether that proposition holds (1) or not (0):
  a: 1111010100
  b: 1100010000
Evaluate the temporal formula
  X ((! (G (b))) /\(X (! (a))))
over the trace above. Next (X) moves one step forward within the trace; eventually (F) requires holding at some current-or-later position; always (G) requires holding at every current-or-later position; until (U) requires the right formula to hold at some later-or-equal position with the left formula holding at every position strictly before it.
The answer is the sorted, comma-separated list of all positions where the formula is true; if it is true nowhere, the answer is the single word "none".
```

**Answer**
```
2,4,6,7
```

## Example 2 (level 2)

**Prompt**
```
Time positions are the integers 0, 1, ..., 9.
Each row shows, per position, whether that proposition holds (1) or not (0):
  a: 0101101000
  b: 1101010001
Evaluate the temporal formula
  ! (X (X (G (b))))
over the trace above. Next (X) moves one step forward within the trace; eventually (F) requires holding at some current-or-later position; always (G) requires holding at every current-or-later position; until (U) requires the right formula to hold at some later-or-equal position with the left formula holding at every position strictly before it.
The answer is the sorted, comma-separated list of all positions where the formula is true; if it is true nowhere, the answer is the single word "none".
```

**Answer**
```
0,1,2,3,4,5,6,8,9
```

# Level 5

## Example 1 (level 5)

**Prompt**
```
Time positions are the integers 0, 1, ..., 15.
Each row shows, per position, whether that proposition holds (1) or not (0):
  a: 1110101000010111
  b: 1000010110010001
  c: 0000011100110001
Evaluate the temporal formula
  F (F ((! (! (X (G (a))))) /\(F (F (X (F (a)))))))
over the trace above. Next (X) moves one step forward within the trace; eventually (F) requires holding at some current-or-later position; always (G) requires holding at every current-or-later position; until (U) requires the right formula to hold at some later-or-equal position with the left formula holding at every position strictly before it.
The answer is the sorted, comma-separated list of all positions where the formula is true; if it is true nowhere, the answer is the single word "none".
```

**Answer**
```
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14
```

## Example 2 (level 5)

**Prompt**
```
Time positions are the integers 0, 1, ..., 15.
Each row shows, per position, whether that proposition holds (1) or not (0):
  a: 0011100011011101
  b: 0000110011001110
  c: 0111111011011101
Evaluate the temporal formula
  F (X (X (X (F (X (! (a)))))))
over the trace above. Next (X) moves one step forward within the trace; eventually (F) requires holding at some current-or-later position; always (G) requires holding at every current-or-later position; until (U) requires the right formula to hold at some later-or-equal position with the left formula holding at every position strictly before it.
The answer is the sorted, comma-separated list of all positions where the formula is true; if it is true nowhere, the answer is the single word "none".
```

**Answer**
```
0,1,2,3,4,5,6,7,8,9,10
```
