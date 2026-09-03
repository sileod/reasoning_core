## Level 0

### Example 1

This is a power-of-two buddy allocator of maximum order 2; every block size is a power of two up to 2^2, and a block's buddy is its sibling from the split that created it. When a block is freed, it merges with its buddy whenever the buddy is free, repeatedly. Currently the free blocks are: 4@start0 (all other blocks are occupied).

Allocate a block of size 2. Afterwards, report the free blocks as a multiset of sizes: for each distinct free size list 'count x size', sizes in decreasing order, separated by '; ', or write 'empty' if none remain. The answer is that list.


**Answer:**
1x2


### Example 2

This is a power-of-two buddy allocator of maximum order 6; every block size is a power of two up to 2^6, and a block's buddy is its sibling from the split that created it. When a block is freed, it merges with its buddy whenever the buddy is free, repeatedly. Currently the free blocks are: 32@start32 (all other blocks are occupied).

Free the currently-allocated block of size 32 starting at 0. Afterwards, report the free blocks as a multiset of sizes: for each distinct free size list 'count x size', sizes in decreasing order, separated by '; ', or write 'empty' if none remain. The answer is that list.


**Answer:**
1x64




## Level 2

### Example 1

This is a power-of-two buddy allocator of maximum order 5; every block size is a power of two up to 2^5, and a block's buddy is its sibling from the split that created it. When a block is freed, it merges with its buddy whenever the buddy is free, repeatedly. Currently the free blocks are: 8@start8, 16@start16 (all other blocks are occupied).

Free the currently-allocated block of size 8 starting at 0. Afterwards, report the free blocks as a multiset of sizes: for each distinct free size list 'count x size', sizes in decreasing order, separated by '; ', or write 'empty' if none remain. The answer is that list.


**Answer:**
1x32


### Example 2

This is a power-of-two buddy allocator of maximum order 6; every block size is a power of two up to 2^6, and a block's buddy is its sibling from the split that created it. When a block is freed, it merges with its buddy whenever the buddy is free, repeatedly. Currently the free blocks are: 16@start16 (all other blocks are occupied).

Allocate a block of size 8. Afterwards, report the free blocks as a multiset of sizes: for each distinct free size list 'count x size', sizes in decreasing order, separated by '; ', or write 'empty' if none remain. The answer is that list.


**Answer:**
1x8




## Level 5

### Example 1

This is a power-of-two buddy allocator of maximum order 6; every block size is a power of two up to 2^6, and a block's buddy is its sibling from the split that created it. When a block is freed, it merges with its buddy whenever the buddy is free, repeatedly. Currently the free blocks are: 8@start56 (all other blocks are occupied).

Free the currently-allocated block of size 8 starting at 48. Afterwards, report the free blocks as a multiset of sizes: for each distinct free size list 'count x size', sizes in decreasing order, separated by '; ', or write 'empty' if none remain. The answer is that list.


**Answer:**
1x16


### Example 2

This is a power-of-two buddy allocator of maximum order 3; every block size is a power of two up to 2^3, and a block's buddy is its sibling from the split that created it. When a block is freed, it merges with its buddy whenever the buddy is free, repeatedly. Currently the free blocks are: 4@start0, 2@start4 (all other blocks are occupied).

Allocate a block of size 4. Afterwards, report the free blocks as a multiset of sizes: for each distinct free size list 'count x size', sizes in decreasing order, separated by '; ', or write 'empty' if none remain. The answer is that list.


**Answer:**
1x2



