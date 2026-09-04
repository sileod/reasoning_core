# P034v1 samples: pivot_unpivot_transform

## Level 0

### Example 1

Prompt:

The table below is in long form. Reshape it into wide form as described.

Reshape this table from long to wide form: group the rows by the id1,id2 pair so each distinct pair becomes one output cell holding the the count of rows in that cell. Report, for each distinct id pair, the resulting value. Write the answer as lines "(id1, id2) -> value" separated by spaces, in ascending id1 then id2 order.

Table:
['id1,id2,value1,value2', '0,1,8,1', '2,2,6,8', '0,2,5,2']

The answer is a list of lines separated by spaces.

Answer:

(0, 1) -> 2 (0, 2) -> 2 (2, 2) -> 2

### Example 2

Prompt:

The table below is in wide form. Reshape it into long form as described.

Reshape this table from wide to long form: one long-form record per cell. Any missing cell shown as NA is treated as the value 4. List every resulting record as "(id, column, value)", columns in their table order and records grouped and ordered by ascending id.

Table:
['id,c0,c1,c2', '3,4,NA,0', '3,3,9,9', '0,6,NA,6']

The answer is a list of records separated by spaces.

Answer:

(id0, c0, 6) (id0, c1, 4) (id0, c2, 6) (id3, c0, 4) (id3, c1, 4) (id3, c2, 0) (id3, c0, 3) (id3, c1, 9) (id3, c2, 9)

## Level 2

### Example 1

Prompt:

The table below is in wide form. Reshape it into long form as described.

Reshape this table from wide to long form: one long-form record per cell. Any missing cell shown as NA is treated as the value 8. List every resulting record as "(id, column, value)", columns in their table order and records grouped and ordered by ascending id.

Table:
['id,c0,c1,c2,c3,c4', '2,NA,13,21,18,NA', '5,6,28,11,18,1', '2,18,27,5,3,5', '1,7,8,6,5,24', '4,12,29,16,6,NA']

The answer is a list of records separated by spaces.

Answer:

(id1, c0, 7) (id1, c1, 8) (id1, c2, 6) (id1, c3, 5) (id1, c4, 24) (id2, c0, 8) (id2, c1, 13) (id2, c2, 21) (id2, c3, 18) (id2, c4, 8) (id2, c0, 18) (id2, c1, 27) (id2, c2, 5) (id2, c3, 3) (id2, c4, 5) (id4, c0, 12) (id4, c1, 29) (id4, c2, 16) (id4, c3, 6) (id4, c4, 8) (id5, c0, 6) (id5, c1, 28) (id5, c2, 11) (id5, c3, 18) (id5, c4, 1)

### Example 2

Prompt:

The table below is in wide form. Reshape it into long form as described.

Reshape this table from wide to long form: one long-form record per cell. Any missing cell shown as NA is treated as missing and must be written as NA in the output. List every resulting record as "(id, column, value)", columns in their table order and records grouped and ordered by ascending id.

Table:
['id,c0,c1,c2,c3,c4', '5,14,13,6,17,23', '0,NA,17,14,22,NA', '1,0,NA,23,NA,NA', '5,11,NA,13,10,NA', '1,23,27,12,7,29']

The answer is a list of records separated by spaces.

Answer:

(id0, c0, NA) (id0, c1, 17) (id0, c2, 14) (id0, c3, 22) (id0, c4, NA) (id1, c0, 0) (id1, c1, NA) (id1, c2, 23) (id1, c3, NA) (id1, c4, NA) (id1, c0, 23) (id1, c1, 27) (id1, c2, 12) (id1, c3, 7) (id1, c4, 29) (id5, c0, 14) (id5, c1, 13) (id5, c2, 6) (id5, c3, 17) (id5, c4, 23) (id5, c0, 11) (id5, c1, NA) (id5, c2, 13) (id5, c3, 10) (id5, c4, NA)

## Level 5

### Example 1

Prompt:

The table below is in wide form. Reshape it into long form as described.

Reshape this table from wide to long form: one long-form record per cell. Any missing cell shown as NA is treated as missing and must be written as NA in the output. List every resulting record as "(id, column, value)", columns in their table order and records grouped and ordered by ascending id.

Table:
['id,c0,c1,c2,c3,c4,c5,c6,c7', '5,38,32,40,28,45,40,NA,58', '7,NA,2,1,NA,56,5,22,40', '8,NA,0,45,23,38,NA,18,NA', '2,55,20,23,48,28,2,48,34', '3,19,57,40,22,38,5,14,30', '5,15,52,NA,17,8,NA,36,NA', '2,47,40,NA,52,11,59,20,8', '0,26,34,16,31,43,6,8,55']

The answer is a list of records separated by spaces.

Answer:

(id0, c0, 26) (id0, c1, 34) (id0, c2, 16) (id0, c3, 31) (id0, c4, 43) (id0, c5, 6) (id0, c6, 8) (id0, c7, 55) (id2, c0, 55) (id2, c1, 20) (id2, c2, 23) (id2, c3, 48) (id2, c4, 28) (id2, c5, 2) (id2, c6, 48) (id2, c7, 34) (id2, c0, 47) (id2, c1, 40) (id2, c2, NA) (id2, c3, 52) (id2, c4, 11) (id2, c5, 59) (id2, c6, 20) (id2, c7, 8) (id3, c0, 19) (id3, c1, 57) (id3, c2, 40) (id3, c3, 22) (id3, c4, 38) (id3, c5, 5) (id3, c6, 14) (id3, c7, 30) (id5, c0, 38) (id5, c1, 32) (id5, c2, 40) (id5, c3, 28) (id5, c4, 45) (id5, c5, 40) (id5, c6, NA) (id5, c7, 58) (id5, c0, 15) (id5, c1, 52) (id5, c2, NA) (id5, c3, 17) (id5, c4, 8) (id5, c5, NA) (id5, c6, 36) (id5, c7, NA) (id7, c0, NA) (id7, c1, 2) (id7, c2, 1) (id7, c3, NA) (id7, c4, 56) (id7, c5, 5) (id7, c6, 22) (id7, c7, 40) (id8, c0, NA) (id8, c1, 0) (id8, c2, 45) (id8, c3, 23) (id8, c4, 38) (id8, c5, NA) (id8, c6, 18) (id8, c7, NA)

### Example 2

Prompt:

The table below is in long form. Reshape it into wide form as described.

Reshape this table from long to wide form: group the rows by the id1,id2 pair so each distinct pair becomes one output cell holding the the sum of all values in that cell. Report, for each distinct id pair, the resulting value. Write the answer as lines "(id1, id2) -> value" separated by spaces, in ascending id1 then id2 order.

Table:
['id1,id2,value1,value2', '2,5,19,7', '0,2,14,23', '7,7,49,33', '1,7,18,35', '4,6,48,24', '2,6,31,47', '3,6,14,17', '5,7,50,42']

The answer is a list of lines separated by spaces.

Answer:

(0, 2) -> 37 (1, 7) -> 53 (2, 5) -> 26 (2, 6) -> 78 (3, 6) -> 31 (4, 6) -> 72 (5, 7) -> 92 (7, 7) -> 82

