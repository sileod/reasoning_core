## Level 0
### Example 1
A 3-set 3-way set-associative cache with LRU replacement holds blocks numbered 0..26. Block B maps to set B mod 3. When a block is loaded into a full set, the least-recently-used way is evicted, unless the block is already in the set, which refreshes it instead. The cache starts empty. The access sequence is

20, 18, 0, 13, 9

Process every access except the last to settle the cache state, then read the final access against that state. The answer is 'HIT set S way W' if the final access's block is cached (S is its set, W the way holding it), or 'MISS set S way W' if not (S is its set, W the way it overwrites).

Answer:
MISS set 0 way 0

### Example 2
A 3-set 3-way set-associative cache with FIFO replacement holds blocks numbered 0..17. Block B maps to set B mod 3. When a block is loaded into a full set, the oldest-loaded way is evicted, unless the block is already in the set, which refreshes it instead. The cache starts empty. The access sequence is

16, 5, 17, 16, 17

Process every access except the last to settle the cache state, then read the final access against that state. The answer is 'HIT set S way W' if the final access's block is cached (S is its set, W the way holding it), or 'MISS set S way W' if not (S is its set, W the way it overwrites).

Answer:
HIT set 2 way 2

## Level 2
### Example 1
A 5-set 4-way set-associative cache with FIFO replacement holds blocks numbered 0..59. Block B maps to set B mod 5. When a block is loaded into a full set, the oldest-loaded way is evicted, unless the block is already in the set, which refreshes it instead. The cache starts empty. The access sequence is

57, 40, 50, 13, 3, 10, 7, 55, 47

Process every access except the last to settle the cache state, then read the final access against that state. The answer is 'HIT set S way W' if the final access's block is cached (S is its set, W the way holding it), or 'MISS set S way W' if not (S is its set, W the way it overwrites).

Answer:
MISS set 2 way 3

### Example 2
A 5-set 4-way set-associative cache with FIFO replacement holds blocks numbered 0..39. Block B maps to set B mod 5. When a block is loaded into a full set, the oldest-loaded way is evicted, unless the block is already in the set, which refreshes it instead. The cache starts empty. The access sequence is

35, 19, 39, 14, 35, 21, 4, 39, 15

Process every access except the last to settle the cache state, then read the final access against that state. The answer is 'HIT set S way W' if the final access's block is cached (S is its set, W the way holding it), or 'MISS set S way W' if not (S is its set, W the way it overwrites).

Answer:
MISS set 0 way 3

## Level 5
### Example 1
A 8-set 5-way set-associative cache with LRU replacement holds blocks numbered 0..119. Block B maps to set B mod 8. When a block is loaded into a full set, the least-recently-used way is evicted, unless the block is already in the set, which refreshes it instead. The cache starts empty. The access sequence is

66, 39, 21, 57, 69, 44, 97, 117, 79, 84, 11, 39, 84, 71, 107

Process every access except the last to settle the cache state, then read the final access against that state. The answer is 'HIT set S way W' if the final access's block is cached (S is its set, W the way holding it), or 'MISS set S way W' if not (S is its set, W the way it overwrites).

Answer:
MISS set 3 way 2

### Example 2
A 8-set 5-way set-associative cache with LRU replacement holds blocks numbered 0..39. Block B maps to set B mod 8. When a block is loaded into a full set, the least-recently-used way is evicted, unless the block is already in the set, which refreshes it instead. The cache starts empty. The access sequence is

39, 1, 38, 27, 20, 11, 28, 17, 26, 29, 19, 12, 2, 26, 20

Process every access except the last to settle the cache state, then read the final access against that state. The answer is 'HIT set S way W' if the final access's block is cached (S is its set, W the way holding it), or 'MISS set S way W' if not (S is its set, W the way it overwrites).

Answer:
HIT set 4 way 0
