## Level 0
### Example 1
There are 3 numbered processors. Jobs arrive at an event queue over time; each arrival states a time, a target processor, a work amount, and a priority where a lower number means higher priority. Ties in priority are broken by order of arrival into the queue.
At time 4, a job for processor 0 arrives needing 6 units of work with priority 4.
At time 6, a job for processor 1 arrives needing 1 units of work with priority 1.
At time 2, a job for processor 2 arrives needing 2 units of work with priority 5.
At time 10, a job for processor 1 arrives needing 2 units of work with priority 5.
At time 13, a job for processor 0 arrives needing 4 units of work with priority 2.
At time 23, a job for processor 1 arrives needing 5 units of work with priority 3.
Each processor processes one job at a time. When it becomes free, it starts the pending job of highest priority (ties by earlier arrival). A job starts no earlier than its arrival time. When there are no jobs left for a processor, that processor is done.
What is the total completion time, defined as the maximum over all processors of the time when a processor finishes all of its jobs?
The answer is a non-negative integer.
Answer:
30

### Example 2
There are 3 numbered processors. Jobs arrive at an event queue over time; each arrival states a time, a target processor, a work amount, and a priority where a lower number means higher priority. Ties in priority are broken by order of arrival into the queue.
At time 15, a job for processor 0 arrives needing 5 units of work with priority 4.
At time 24, a job for processor 2 arrives needing 6 units of work with priority 3.
At time 4, a job for processor 0 arrives needing 1 units of work with priority 3.
At time 2, a job for processor 0 arrives needing 6 units of work with priority 5.
At time 25, a job for processor 0 arrives needing 6 units of work with priority 1.
At time 15, a job for processor 2 arrives needing 4 units of work with priority 5.
Each processor processes one job at a time. When it becomes free, it starts the pending job of highest priority (ties by earlier arrival). A job starts no earlier than its arrival time. When there are no jobs left for a processor, that processor is done.
What is the total completion time, defined as the maximum over all processors of the time when a processor finishes all of its jobs?
The answer is a non-negative integer.
Answer:
43

## Level 2
### Example 1
There are 5 numbered processors. Jobs arrive at an event queue over time; each arrival states a time, a target processor, a work amount, and a priority where a lower number means higher priority. Ties in priority are broken by order of arrival into the queue.
At time 39, a job for processor 0 arrives needing 4 units of work with priority 2.
At time 44, a job for processor 3 arrives needing 6 units of work with priority 2.
At time 35, a job for processor 2 arrives needing 4 units of work with priority 1.
At time 25, a job for processor 3 arrives needing 6 units of work with priority 4.
At time 34, a job for processor 0 arrives needing 4 units of work with priority 5.
At time 42, a job for processor 2 arrives needing 2 units of work with priority 2.
At time 2, a job for processor 2 arrives needing 2 units of work with priority 4.
At time 23, a job for processor 2 arrives needing 2 units of work with priority 1.
At time 27, a job for processor 0 arrives needing 1 units of work with priority 3.
At time 15, a job for processor 4 arrives needing 6 units of work with priority 4.
Each processor processes one job at a time. When it becomes free, it starts the pending job of highest priority (ties by earlier arrival). A job starts no earlier than its arrival time. When there are no jobs left for a processor, that processor is done.
What is the total completion time, defined as the maximum over all processors of the time when a processor finishes all of its jobs?
The answer is a non-negative integer.
Answer:
56

### Example 2
There are 5 numbered processors. Jobs arrive at an event queue over time; each arrival states a time, a target processor, a work amount, and a priority where a lower number means higher priority. Ties in priority are broken by order of arrival into the queue.
At time 36, a job for processor 4 arrives needing 2 units of work with priority 5.
At time 39, a job for processor 0 arrives needing 2 units of work with priority 1.
At time 46, a job for processor 3 arrives needing 4 units of work with priority 2.
At time 13, a job for processor 4 arrives needing 1 units of work with priority 2.
At time 38, a job for processor 4 arrives needing 3 units of work with priority 4.
At time 29, a job for processor 4 arrives needing 3 units of work with priority 1.
At time 43, a job for processor 3 arrives needing 2 units of work with priority 4.
At time 8, a job for processor 3 arrives needing 2 units of work with priority 3.
At time 31, a job for processor 1 arrives needing 3 units of work with priority 1.
At time 27, a job for processor 2 arrives needing 6 units of work with priority 2.
Each processor processes one job at a time. When it becomes free, it starts the pending job of highest priority (ties by earlier arrival). A job starts no earlier than its arrival time. When there are no jobs left for a processor, that processor is done.
What is the total completion time, defined as the maximum over all processors of the time when a processor finishes all of its jobs?
The answer is a non-negative integer.
Answer:
54

## Level 5
### Example 1
There are 8 numbered processors. Jobs arrive at an event queue over time; each arrival states a time, a target processor, a work amount, and a priority where a lower number means higher priority. Ties in priority are broken by order of arrival into the queue.
At time 68, a job for processor 1 arrives needing 3 units of work with priority 5.
At time 14, a job for processor 7 arrives needing 1 units of work with priority 1.
At time 21, a job for processor 2 arrives needing 5 units of work with priority 5.
At time 16, a job for processor 0 arrives needing 5 units of work with priority 5.
At time 49, a job for processor 7 arrives needing 4 units of work with priority 3.
At time 15, a job for processor 6 arrives needing 1 units of work with priority 4.
At time 10, a job for processor 0 arrives needing 1 units of work with priority 3.
At time 18, a job for processor 5 arrives needing 2 units of work with priority 1.
At time 41, a job for processor 3 arrives needing 4 units of work with priority 5.
At time 48, a job for processor 7 arrives needing 4 units of work with priority 5.
At time 42, a job for processor 3 arrives needing 3 units of work with priority 1.
At time 62, a job for processor 6 arrives needing 5 units of work with priority 1.
At time 7, a job for processor 1 arrives needing 4 units of work with priority 2.
At time 21, a job for processor 6 arrives needing 6 units of work with priority 4.
At time 6, a job for processor 0 arrives needing 2 units of work with priority 2.
At time 21, a job for processor 3 arrives needing 4 units of work with priority 5.
Each processor processes one job at a time. When it becomes free, it starts the pending job of highest priority (ties by earlier arrival). A job starts no earlier than its arrival time. When there are no jobs left for a processor, that processor is done.
What is the total completion time, defined as the maximum over all processors of the time when a processor finishes all of its jobs?
The answer is a non-negative integer.
Answer:
74

### Example 2
There are 8 numbered processors. Jobs arrive at an event queue over time; each arrival states a time, a target processor, a work amount, and a priority where a lower number means higher priority. Ties in priority are broken by order of arrival into the queue.
At time 34, a job for processor 1 arrives needing 5 units of work with priority 2.
At time 42, a job for processor 4 arrives needing 3 units of work with priority 3.
At time 66, a job for processor 3 arrives needing 2 units of work with priority 5.
At time 16, a job for processor 6 arrives needing 3 units of work with priority 1.
At time 62, a job for processor 4 arrives needing 2 units of work with priority 1.
At time 3, a job for processor 6 arrives needing 2 units of work with priority 5.
At time 14, a job for processor 5 arrives needing 4 units of work with priority 4.
At time 20, a job for processor 3 arrives needing 2 units of work with priority 3.
At time 25, a job for processor 6 arrives needing 2 units of work with priority 3.
At time 13, a job for processor 0 arrives needing 4 units of work with priority 1.
At time 53, a job for processor 5 arrives needing 1 units of work with priority 4.
At time 1, a job for processor 5 arrives needing 1 units of work with priority 5.
At time 36, a job for processor 0 arrives needing 5 units of work with priority 5.
At time 54, a job for processor 7 arrives needing 3 units of work with priority 5.
At time 61, a job for processor 5 arrives needing 3 units of work with priority 4.
At time 61, a job for processor 5 arrives needing 3 units of work with priority 2.
Each processor processes one job at a time. When it becomes free, it starts the pending job of highest priority (ties by earlier arrival). A job starts no earlier than its arrival time. When there are no jobs left for a processor, that processor is done.
What is the total completion time, defined as the maximum over all processors of the time when a processor finishes all of its jobs?
The answer is a non-negative integer.
Answer:
73

