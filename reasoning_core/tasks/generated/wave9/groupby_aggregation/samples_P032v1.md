# Samples P032v1 - groupby_aggregation

## Level 0

### Example 1 (level 0)

**Prompt:**

We have a table with rows [group, value]. Rows: G2:16 G2:20 G1:0 G0:15 G0:4 G2:14.
Partition the rows by the group letter and for each group compute the minimum of that group's values (for 'row count', the number of rows). Keep only the groups whose minimum is at least the threshold 0. List the kept group letters in decreasing order of their minimum; when two groups tie on the minimum, order them alphabetically. Answer with the comma-separated list of kept group letters, e.g. G0,G2. If no group is kept, answer the single word none.

**Answer:** `G2,G0,G1`

### Example 2 (level 0)

**Prompt:**

We have a table with rows [group, value]. Rows: G2:0 G1:4 G1:2 G1:19 G1:15 G0:2.
Partition the rows by the group letter and for each group compute the row count of that group's values (for 'row count', the number of rows). Keep only the groups whose row count is at least the threshold 2. List the kept group letters in decreasing order of their row count; when two groups tie on the row count, order them alphabetically. Answer with the comma-separated list of kept group letters, e.g. G0,G2. If no group is kept, answer the single word none.

**Answer:** `G1`

## Level 2

### Example 1 (level 2)

**Prompt:**

We have a table with rows [group, value]. Rows: G2:4 G0:14 G3:10 G1:1 G3:9 G4:9 G1:6 G3:17 G2:5 G0:15.
Partition the rows by the group letter and for each group compute the maximum of that group's values (for 'row count', the number of rows). Keep only the groups whose maximum is at least the threshold 5. List the kept group letters in decreasing order of their maximum; when two groups tie on the maximum, order them alphabetically. Answer with the comma-separated list of kept group letters, e.g. G0,G2. If no group is kept, answer the single word none.

**Answer:** `G3,G0,G4,G1,G2`

### Example 2 (level 2)

**Prompt:**

We have a table with rows [group, value]. Rows: G3:4 G1:15 G1:16 G0:2 G1:20 G2:8 G1:10 G0:10 G2:5 G4:3.
Partition the rows by the group letter and for each group compute the sum of that group's values (for 'row count', the number of rows). Keep only the groups whose sum is at least the threshold 3. List the kept group letters in decreasing order of their sum; when two groups tie on the sum, order them alphabetically. Answer with the comma-separated list of kept group letters, e.g. G0,G2. If no group is kept, answer the single word none.

**Answer:** `G1,G2,G0,G3,G4`

## Level 5

### Example 1 (level 5)

**Prompt:**

We have a table with rows [group, value]. Rows: G5:19 G4:20 G2:4 G3:14 G7:11 G2:11 G7:5 G6:12 G5:2 G1:5 G0:19 G1:4 G7:7 G6:9 G2:10 G1:6.
Partition the rows by the group letter and for each group compute the minimum of that group's values (for 'row count', the number of rows). Keep only the groups whose minimum is at least the threshold 4. List the kept group letters in decreasing order of their minimum; when two groups tie on the minimum, order them alphabetically. Answer with the comma-separated list of kept group letters, e.g. G0,G2. If no group is kept, answer the single word none.

**Answer:** `G4,G0,G3,G6,G7,G1,G2`

### Example 2 (level 5)

**Prompt:**

We have a table with rows [group, value]. Rows: G3:4 G1:4 G7:1 G1:13 G2:13 G0:6 G6:0 G6:2 G7:3 G4:14 G7:16 G3:8 G6:13 G5:17 G5:17 G6:13.
Partition the rows by the group letter and for each group compute the sum of that group's values (for 'row count', the number of rows). Keep only the groups whose sum is at least the threshold 7. List the kept group letters in decreasing order of their sum; when two groups tie on the sum, order them alphabetically. Answer with the comma-separated list of kept group letters, e.g. G0,G2. If no group is kept, answer the single word none.

**Answer:** `G5,G6,G7,G1,G4,G2,G3`
