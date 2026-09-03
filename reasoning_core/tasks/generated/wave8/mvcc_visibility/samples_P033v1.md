## Level 0
### Example 1
**Prompt:**
An MVCC store keeps row versions as (begin,end) timestamp pairs; a version is visible to snapshot T exactly when begin <= T < end. Only the newest such version is visible.
Row versions: (4,5) (8,12)
Snapshot time: 13
Which version is visible to this snapshot? Give it as (begin,end), or the exact word 'none' if no version is visible.

**Answer:**
(8,12)

### Example 2
**Prompt:**
An MVCC store keeps row versions as (begin,end) timestamp pairs; a version is visible to snapshot T exactly when begin <= T < end. Only the newest such version is visible.
Row versions: (1,6) (10,14)
Snapshot time: 7
Which version is visible to this snapshot? Give it as (begin,end), or the exact word 'none' if no version is visible.

**Answer:**
(1,6)

## Level 2
### Example 1
**Prompt:**
An MVCC store keeps row versions as (begin,end) timestamp pairs; a version is visible to snapshot T exactly when begin <= T < end. Only the newest such version is visible.
Row versions: (2,5) (10,13) (14,15) (16,19) (24,30) (32,37)
Snapshot time: 9
Which version is visible to this snapshot? Give it as (begin,end), or the exact word 'none' if no version is visible.

**Answer:**
(2,5)

### Example 2
**Prompt:**
An MVCC store keeps row versions as (begin,end) timestamp pairs; a version is visible to snapshot T exactly when begin <= T < end. Only the newest such version is visible.
Row versions: (1,3) (4,8) (10,11) (12,15) (20,26) (31,37)
Snapshot time: 2
Which version is visible to this snapshot? Give it as (begin,end), or the exact word 'none' if no version is visible.

**Answer:**
(1,3)

## Level 5
### Example 1
**Prompt:**
An MVCC store keeps row versions as (begin,end) timestamp pairs; a version is visible to snapshot T exactly when begin <= T < end. Only the newest such version is visible.
Rows: row0: (5,8) (9,11) (15,16) (20,21) (22,25) (29,31) (32,33) (37,41) (44,47) (50,54) (58,63) (64,69) ; row1: (2,6) (10,13) (15,21) (25,30) (34,35) (38,43) (44,49) (54,58) (60,66) (68,70) (73,78) (79,80)
Snapshot time: 35
For every row, give the (begin,end) version visible to this snapshot, using the exact word 'none' for any row with no visible version. Answer as a list in row order, e.g. [(1,4),none,(2,6)].

**Answer:**
[(32,33),(34,35)]

### Example 2
**Prompt:**
An MVCC store keeps row versions as (begin,end) timestamp pairs; a version is visible to snapshot T exactly when begin <= T < end. Only the newest such version is visible.
Rows: row0: (5,6) (8,13) (15,18) (21,25) (30,31) (33,35) (37,39) (40,42) (47,48) (51,55) (58,64) (69,74) ; row1: (3,5) (10,11) (16,22) (25,28) (31,32) (36,42) (47,50) (51,52) (55,56) (57,59) (62,67) (68,70)
Snapshot time: 74
For every row, give the (begin,end) version visible to this snapshot, using the exact word 'none' for any row with no visible version. Answer as a list in row order, e.g. [(1,4),none,(2,6)].

**Answer:**
[(69,74),(68,70)]

