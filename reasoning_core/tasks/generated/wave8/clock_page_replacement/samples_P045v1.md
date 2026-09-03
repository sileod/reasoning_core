# Sample gallery: clock_page_replacement (P045v1)

Given a Clock (second-chance) page-replacement state and a page fault, output the frame index chosen for eviction.

## Level 0

### Prompt

A system uses Clock (second-chance) page replacement. The current state:
frame 0: page 3 (refbit 0); frame 1: page 5 (refbit 1); frame 2: page 8 (refbit 0); frame 3: page 12 (refbit 1); frame 4: page 4 (refbit 1)
The clock hand is at frame 2.
A page fault occurs for page 10.
Starting from the hand, scan frames in circular order, skipping frames whose reference bit is 1 (clearing that bit to 0) and evicting the first frame whose reference bit is 0.
The answer is the index of the frame evicted, as a single integer.

### Answer

2

## Level 0

### Prompt

A system uses Clock (second-chance) page replacement. The current state:
frame 0: page 1 (refbit 0); frame 1: page 5 (refbit 0); frame 2: page 12 (refbit 1); frame 3: page 15 (refbit 1); frame 4: page 18 (refbit 0)
The clock hand is at frame 2.
A page fault occurs for page 22.
Starting from the hand, scan frames in circular order, skipping frames whose reference bit is 1 (clearing that bit to 0) and evicting the first frame whose reference bit is 0.
The answer is the index of the frame evicted, as a single integer.

### Answer

4

## Level 2

### Prompt

A system uses Clock (second-chance) page replacement. The current state:
frame 0: page 25 (refbit 1); frame 1: page 29 (refbit 0); frame 2: page 11 (refbit 0); frame 3: page 23 (refbit 1); frame 4: page 10 (refbit 1); frame 5: page 28 (refbit 1); frame 6: page 16 (refbit 1)
The clock hand is at frame 0.
A page fault occurs for page 27.
Starting from the hand, scan frames in circular order, skipping frames whose reference bit is 1 (clearing that bit to 0) and evicting the first frame whose reference bit is 0.
The answer is the index of the frame evicted, as a single integer.

### Answer

1

## Level 2

### Prompt

A system uses Clock (second-chance) page replacement. The current state:
frame 0: page 14 (refbit 1); frame 1: page 16 (refbit 0); frame 2: page 28 (refbit 0); frame 3: page 27 (refbit 1); frame 4: page 10 (refbit 0); frame 5: page 22 (refbit 1); frame 6: page 4 (refbit 1)
The clock hand is at frame 4.
A page fault occurs for page 38.
Starting from the hand, scan frames in circular order, skipping frames whose reference bit is 1 (clearing that bit to 0) and evicting the first frame whose reference bit is 0.
The answer is the index of the frame evicted, as a single integer.

### Answer

4

## Level 5

### Prompt

A system uses Clock (second-chance) page replacement. The current state:
frame 0: page 7 (refbit 0); frame 1: page 5 (refbit 1); frame 2: page 46 (refbit 1); frame 3: page 14 (refbit 1); frame 4: page 31 (refbit 1); frame 5: page 45 (refbit 1); frame 6: page 47 (refbit 0); frame 7: page 48 (refbit 1); frame 8: page 13 (refbit 0); frame 9: page 10 (refbit 1)
The clock hand is at frame 5.
A page fault occurs for page 29.
Starting from the hand, scan frames in circular order, skipping frames whose reference bit is 1 (clearing that bit to 0) and evicting the first frame whose reference bit is 0.
The answer is the index of the frame evicted, as a single integer.

### Answer

6

## Level 5

### Prompt

A system uses Clock (second-chance) page replacement. The current state:
frame 0: page 11 (refbit 1); frame 1: page 5 (refbit 1); frame 2: page 18 (refbit 1); frame 3: page 33 (refbit 1); frame 4: page 2 (refbit 0); frame 5: page 21 (refbit 0); frame 6: page 27 (refbit 0); frame 7: page 9 (refbit 0); frame 8: page 37 (refbit 1); frame 9: page 19 (refbit 0)
The clock hand is at frame 1.
A page fault occurs for page 14.
Starting from the hand, scan frames in circular order, skipping frames whose reference bit is 1 (clearing that bit to 0) and evicting the first frame whose reference bit is 0.
The answer is the index of the frame evicted, as a single integer.

### Answer

4
