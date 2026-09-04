## Level 0
### Example
#### Prompt
A Keys:
1 1 0 0

A Vals:
2 7 5 6

B Keys:
0 2 1 2

B Vals:
2 6 7 3

Join:
inner

How:
sum over matches of (K * A.val + B.val)

K:
3

Compute the exact result of the described join. The answer is a single non-negative integer.
#### Answer
78

### Example
#### Prompt
A Keys:
0 0 2 0

A Vals:
1 3 4 6

B Keys:
1 1 0 1

B Vals:
3 2 7 5

Join:
inner

How:
sum over matches of (K * A.val + B.val)

K:
4

Compute the exact result of the described join. The answer is a single non-negative integer.
#### Answer
61

## Level 2
### Example
#### Prompt
A Keys:
0 2 3 4 4 4 2 3

A Vals:
6 8 10 10 8 4 8 5

B Keys:
3 3 3 0 0 3 1 3

B Vals:
8 11 1 8 9 1 9 9

Join:
inner

How:
sum over matches of (K * A.val + B.val)

K:
1

Compute the exact result of the described join. The answer is a single non-negative integer.
#### Answer
164

### Example
#### Prompt
A Keys:
2 2 4 0 3 0 4 2

A Vals:
5 6 3 7 11 9 7 7

B Keys:
4 4 3 3 4 1 2 2

B Vals:
2 4 3 5 1 3 9 9

Join:
left

How:
sum over matches of (K * A.val + B.val)

K:
3

Compute the exact result of the described join. The answer is a single non-negative integer.
#### Answer
388

## Level 5
### Example
#### Prompt
A Keys:
7 2 0 0 4 0 2 0 0 4 0 0 6 1

A Vals:
8 14 9 12 1 11 9 12 13 6 1 5 13 15

B Keys:
4 3 4 6 4 5 2 7 5 2 1 7 5 1

B Vals:
7 2 5 7 13 10 9 3 5 2 5 17 3 2

Join:
anti

How:
sum of K * A.val over left rows without a match

K:
4

Compute the exact result of the described join. The answer is a single non-negative integer.
#### Answer
252

### Example
#### Prompt
A Keys:
7 3 0 7 5 5 7 0 6 5 0 4 1 0

A Vals:
14 6 11 15 13 2 17 14 8 4 8 16 1 16

B Keys:
3 1 6 5 2 3 4 3 2 1 7 5 2 2

B Vals:
16 6 4 6 5 10 4 9 5 14 1 12 12 2

Join:
anti

How:
sum of K * A.val over left rows without a match

K:
3

Compute the exact result of the described join. The answer is a single non-negative integer.
#### Answer
147
