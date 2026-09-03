# Level 0

## Example 1

### Prompt

Query:
A skip list holds integer keys in sorted order and has levels numbered 1 (bottom) up to M (top). Each node lists its key followed by its height. At level L, a node's forward pointer reaches the next node whose height is at least L.

Levels:
3

Nodes:
[[10, 3], [21, 1], [23, 3], [25, 2], [34, 1], [45, 2], [55, 2], [57, 1], [58, 3]]

Target:
56

The keys with heights are: (10: 3), (21: 1), (23: 3), (25: 2), (34: 1), (45: 2), (55: 2), (57: 1), (58: 3).

Run the standard skip-list predecessor search for target 56. Start just before the first node at the top level M; at each level from M down to 1, while the forward pointer at that level reaches a node with key strictly less than 56, move to that node and record its key. Report the recorded keys in the order they are visited as a bracket list, e.g. [3, 7]. If no node is ever visited, answer [].

The answer is a bracket list of integers.

### Answer

[10, 23, 25, 45, 55]

## Example 2

### Prompt

Query:
A skip list holds integer keys in sorted order and has levels numbered 1 (bottom) up to M (top). Each node lists its key followed by its height. At level L, a node's forward pointer reaches the next node whose height is at least L.

Levels:
3

Nodes:
[[14, 3], [17, 1], [36, 1], [46, 3], [48, 1], [51, 3], [54, 1], [58, 1], [61, 3]]

Target:
51

The keys with heights are: (14: 3), (17: 1), (36: 1), (46: 3), (48: 1), (51: 3), (54: 1), (58: 1), (61: 3).

Run the standard skip-list predecessor search for target 51. Start just before the first node at the top level M; at each level from M down to 1, while the forward pointer at that level reaches a node with key strictly less than 51, move to that node and record its key. Report the recorded keys in the order they are visited as a bracket list, e.g. [3, 7]. If no node is ever visited, answer [].

The answer is a bracket list of integers.

### Answer

[14, 46, 48]



# Level 2

## Example 1

### Prompt

Query:
A skip list holds integer keys in sorted order and has levels numbered 1 (bottom) up to M (top). Each node lists its key followed by its height. At level L, a node's forward pointer reaches the next node whose height is at least L.

Levels:
5

Nodes:
[[11, 5], [13, 1], [19, 5], [22, 5], [34, 4], [40, 5], [53, 5], [59, 5], [68, 4], [71, 2], [73, 4], [74, 1], [75, 5], [77, 3], [81, 5]]

Target:
83

The keys with heights are: (11: 5), (13: 1), (19: 5), (22: 5), (34: 4), (40: 5), (53: 5), (59: 5), (68: 4), (71: 2), (73: 4), (74: 1), (75: 5), (77: 3), (81: 5).

Run the standard skip-list predecessor search for target 83. Start just before the first node at the top level M; at each level from M down to 1, while the forward pointer at that level reaches a node with key strictly less than 83, move to that node and record its key. Report the recorded keys in the order they are visited as a bracket list, e.g. [3, 7]. If no node is ever visited, answer [].

The answer is a bracket list of integers.

### Answer

[11, 19, 22, 40, 53, 59, 75, 81]

## Example 2

### Prompt

Query:
A skip list holds integer keys in sorted order and has levels numbered 1 (bottom) up to M (top). Each node lists its key followed by its height. At level L, a node's forward pointer reaches the next node whose height is at least L.

Levels:
5

Nodes:
[[13, 5], [19, 4], [20, 3], [23, 3], [26, 3], [27, 5], [39, 1], [47, 5], [52, 4], [54, 2], [57, 2], [58, 5], [64, 3], [79, 1], [82, 5]]

Target:
43

The keys with heights are: (13: 5), (19: 4), (20: 3), (23: 3), (26: 3), (27: 5), (39: 1), (47: 5), (52: 4), (54: 2), (57: 2), (58: 5), (64: 3), (79: 1), (82: 5).

Run the standard skip-list predecessor search for target 43. Start just before the first node at the top level M; at each level from M down to 1, while the forward pointer at that level reaches a node with key strictly less than 43, move to that node and record its key. Report the recorded keys in the order they are visited as a bracket list, e.g. [3, 7]. If no node is ever visited, answer [].

The answer is a bracket list of integers.

### Answer

[13, 27, 39]



# Level 5

## Example 1

### Prompt

Query:
A skip list holds integer keys in sorted order and has levels numbered 1 (bottom) up to M (top). Each node lists its key followed by its height. At level L, a node's forward pointer reaches the next node whose height is at least L.

Levels:
8

Nodes:
[[10, 1], [11, 1], [16, 4], [17, 3], [20, 4], [25, 1], [28, 3], [38, 7], [50, 7], [54, 7], [55, 5], [56, 8], [62, 1], [63, 2], [64, 6], [72, 3], [88, 8], [89, 8], [90, 8], [91, 1], [92, 7], [98, 1], [103, 7], [111, 8]]

Target:
56

The keys with heights are: (10: 1), (11: 1), (16: 4), (17: 3), (20: 4), (25: 1), (28: 3), (38: 7), (50: 7), (54: 7), (55: 5), (56: 8), (62: 1), (63: 2), (64: 6), (72: 3), (88: 8), (89: 8), (90: 8), (91: 1), (92: 7), (98: 1), (103: 7), (111: 8).

Run the standard skip-list predecessor search for target 56. Start just before the first node at the top level M; at each level from M down to 1, while the forward pointer at that level reaches a node with key strictly less than 56, move to that node and record its key. Report the recorded keys in the order they are visited as a bracket list, e.g. [3, 7]. If no node is ever visited, answer [].

The answer is a bracket list of integers.

### Answer

[38, 50, 54, 55]

## Example 2

### Prompt

Query:
A skip list holds integer keys in sorted order and has levels numbered 1 (bottom) up to M (top). Each node lists its key followed by its height. At level L, a node's forward pointer reaches the next node whose height is at least L.

Levels:
8

Nodes:
[[11, 2], [14, 4], [19, 2], [27, 8], [29, 2], [40, 8], [45, 7], [46, 4], [53, 8], [62, 7], [63, 7], [67, 4], [72, 3], [74, 2], [79, 7], [81, 7], [82, 2], [85, 7], [87, 5], [95, 3], [96, 3], [99, 1], [101, 6], [106, 8]]

Target:
110

The keys with heights are: (11: 2), (14: 4), (19: 2), (27: 8), (29: 2), (40: 8), (45: 7), (46: 4), (53: 8), (62: 7), (63: 7), (67: 4), (72: 3), (74: 2), (79: 7), (81: 7), (82: 2), (85: 7), (87: 5), (95: 3), (96: 3), (99: 1), (101: 6), (106: 8).

Run the standard skip-list predecessor search for target 110. Start just before the first node at the top level M; at each level from M down to 1, while the forward pointer at that level reaches a node with key strictly less than 110, move to that node and record its key. Report the recorded keys in the order they are visited as a bracket list, e.g. [3, 7]. If no node is ever visited, answer [].

The answer is a bracket list of integers.

### Answer

[27, 40, 53, 106]


