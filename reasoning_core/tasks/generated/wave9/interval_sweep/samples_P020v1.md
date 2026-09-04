## Level 0
### Example 1
**Prompt:**

Consider the following set of intervals on the number line, each given as (start, end) inclusive of both endpoints: Intervals:
[(11, 15), (4, 8), (3, 22), (5, 6), (9, 22)]. Perform a sweep over the endpoints, counting how many intervals are active as you move left to right, and report the maximum number of intervals that overlap at any single point. The answer is a single integer (the peak overlap).

**Answer:**

3

### Example 2
**Prompt:**

Consider the following set of intervals on the number line, each given as (start, end) inclusive of both endpoints: Intervals:
[(6, 23), (0, 13), (15, 20), (19, 24), (1, 5)]. Perform a sweep over the endpoints, counting how many intervals are active as you move left to right, and report the maximum number of intervals that overlap at any single point. The answer is a single integer (the peak overlap).

**Answer:**

3

## Level 2
### Example 1
**Prompt:**

Consider the following set of intervals on the number line, each given as (start, end) inclusive of both endpoints: Intervals:
[(15, 24), (22, 30), (8, 11), (5, 10), (13, 37), (25, 28), (23, 35), (25, 34), (2, 7)]. Merge all overlapping or touching intervals into disjoint canonical intervals. Report the merged intervals, each as start-end, separated by semicolons, in increasing order of start. Example of the answer format: 2-5; 8-10.

**Answer:**

2-11; 13-37

### Example 2
**Prompt:**

Consider the following set of intervals on the number line, each given as (start, end) inclusive of both endpoints: Intervals:
[(13, 33), (21, 28), (5, 30), (5, 39), (1, 24), (4, 12), (19, 33), (40, 43), (19, 37)]. Merge all overlapping or touching intervals into disjoint canonical intervals. Report the merged intervals, each as start-end, separated by semicolons, in increasing order of start. Example of the answer format: 2-5; 8-10.

**Answer:**

1-39; 40-43

## Level 5
### Example 1
**Prompt:**

Consider the following set of intervals on the number line, each given as (start, end) inclusive of both endpoints: Intervals:
[(3, 53), (17, 22), (35, 72), (21, 27), (36, 43), (24, 50), (8, 50), (15, 29), (43, 57), (21, 22), (39, 64), (12, 49), (5, 28), (3, 71), (4, 10)]. The query point is 35. Sweep to find every start value of the intervals that contain the query point. Report those start values in increasing order, separated by semicolons. Example of the answer format: 2; 8; 9

**Answer:**

3; 3; 8; 12; 24; 35

### Example 2
**Prompt:**

Consider the following set of intervals on the number line, each given as (start, end) inclusive of both endpoints: Intervals:
[(20, 41), (23, 26), (53, 64), (4, 39), (14, 49), (32, 54), (20, 65), (58, 71), (18, 28), (30, 32), (6, 10), (0, 70), (42, 43), (28, 66), (50, 63)]. The query point is 20. Sweep to find every start value of the intervals that contain the query point. Report those start values in increasing order, separated by semicolons. Example of the answer format: 2; 8; 9

**Answer:**

0; 4; 14; 18; 20; 20

