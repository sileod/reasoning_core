# Sample cache_replacement_trace (P041v1)

Task: execute LRU, LFU, or FIFO cache accesses with inserts and evictions under explicit tie rules, returning hits, misses, or final cache state.

## Level 0

**Prompt:**

A cache holds at most 3 distinct keys, drawn from the alphabet {0, 1, ..., 5}. It uses FIFO replacement. The evicted slot is the key that has been in the cache the longest, i.e. the earliest-inserted resident key. The cache starts empty. Process this sequence of key accesses in order: 1 2 0 0 3 0 2 4 How many of the accesses are misses?

**Answer:**

5

**Prompt:**

A cache holds at most 3 distinct keys, drawn from the alphabet {0, 1, ..., 5}. It uses LRU replacement. The evicted slot is the key that has not been accessed for the longest time, i.e. the least-recently-used resident key. The cache starts empty. Process this sequence of key accesses in order: 0 0 1 2 0 4 4 4 How many of the accesses are misses?

**Answer:**

4

## Level 2

**Prompt:**

A cache holds at most 5 distinct keys, drawn from the alphabet {0, 1, ..., 9}. It uses LFU replacement. The evicted slot is the resident key with the fewest accesses since it was inserted; ties are broken by the smallest key. The cache starts empty. Process this sequence of key accesses in order: 6 2 7 5 3 5 9 0 4 3 6 5 0 0 How many of the accesses are misses?

**Answer:**

10

**Prompt:**

A cache holds at most 5 distinct keys, drawn from the alphabet {0, 1, ..., 9}. It uses FIFO replacement. The evicted slot is the key that has been in the cache the longest, i.e. the earliest-inserted resident key. The cache starts empty. Process this sequence of key accesses in order: 9 2 6 8 4 7 5 1 2 9 0 0 7 5 How many of the accesses are misses?

**Answer:**

13

## Level 5

**Prompt:**

A cache holds at most 8 distinct keys, drawn from the alphabet {0, 1, ..., 15}. It uses LFU replacement. The evicted slot is the resident key with the fewest accesses since it was inserted; ties are broken by the earliest insertion. The cache starts empty. Process this sequence of key accesses in order: 11 11 2 4 13 13 3 6 0 14 9 2 7 15 1 1 2 3 7 13 15 4 2 How many of the accesses are misses?

**Answer:**

15

**Prompt:**

A cache holds at most 8 distinct keys, drawn from the alphabet {0, 1, ..., 15}. It uses FIFO replacement. The evicted slot is the key that has been in the cache the longest, i.e. the earliest-inserted resident key. The cache starts empty. Process this sequence of key accesses in order: 12 14 3 11 12 6 5 6 11 13 6 9 8 5 3 5 4 12 1 0 15 9 4 How many of the accesses are hits?

**Answer:**

9
