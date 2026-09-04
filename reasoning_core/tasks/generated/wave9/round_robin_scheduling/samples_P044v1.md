# round_robin_scheduling samples
seed 2693117635

## Level 0
**Prompt:**
```
We run fixed-quantum round-robin scheduling with quantum 5 time units on 3 single-CPU processes.
Arrival time, burst time:
(P0: arrives at 0, burst 5), (P1: arrives at 1, burst 6), (P2: arrives at 4, burst 1)
The CPU serves the ready queue in FIFO order, each ready process getting at most one quantum of CPU before returning to the tail of the queue; a newly arrived process joins at the tail. A process that finishes its remaining burst before exhausting a quantum leaves the schedule.
Report completions: completions for process P2.

The answer is a non-negative integer.
```
**Answer:** `11`

**Prompt:**
```
We run fixed-quantum round-robin scheduling with quantum 5 time units on 3 single-CPU processes.
Arrival time, burst time:
(P0: arrives at 0, burst 4), (P1: arrives at 0, burst 5), (P2: arrives at 4, burst 7)
The CPU serves the ready queue in FIFO order, each ready process getting at most one quantum of CPU before returning to the tail of the queue; a newly arrived process joins at the tail. A process that finishes its remaining burst before exhausting a quantum leaves the schedule.
Report executions: executions for process P2.

The answer is a non-negative integer.
```
**Answer:** `2`


## Level 2
**Prompt:**
```
We run fixed-quantum round-robin scheduling with quantum 6 time units on 5 single-CPU processes.
Arrival time, burst time:
(P0: arrives at 0, burst 3), (P1: arrives at 4, burst 3), (P2: arrives at 5, burst 8), (P3: arrives at 6, burst 8), (P4: arrives at 6, burst 7)
The CPU serves the ready queue in FIFO order, each ready process getting at most one quantum of CPU before returning to the tail of the queue; a newly arrived process joins at the tail. A process that finishes its remaining burst before exhausting a quantum leaves the schedule.
Report completions: completions for process P1.

The answer is a non-negative integer.
```
**Answer:** `7`

**Prompt:**
```
We run fixed-quantum round-robin scheduling with quantum 6 time units on 5 single-CPU processes.
Arrival time, burst time:
(P0: arrives at 0, burst 2), (P1: arrives at 0, burst 11), (P2: arrives at 3, burst 8), (P3: arrives at 4, burst 11), (P4: arrives at 4, burst 4)
The CPU serves the ready queue in FIFO order, each ready process getting at most one quantum of CPU before returning to the tail of the queue; a newly arrived process joins at the tail. A process that finishes its remaining burst before exhausting a quantum leaves the schedule.
Report waiting: waiting for process P2.

The answer is a non-negative integer.
```
**Answer:** `20`


## Level 5
**Prompt:**
```
We run fixed-quantum round-robin scheduling with quantum 8 time units on 7 single-CPU processes.
Arrival time, burst time:
(P0: arrives at 0, burst 5), (P1: arrives at 0, burst 15), (P2: arrives at 1, burst 11), (P3: arrives at 2, burst 16), (P4: arrives at 3, burst 4), (P5: arrives at 5, burst 5), (P6: arrives at 5, burst 10)
The CPU serves the ready queue in FIFO order, each ready process getting at most one quantum of CPU before returning to the tail of the queue; a newly arrived process joins at the tail. A process that finishes its remaining burst before exhausting a quantum leaves the schedule.
Report waiting: waiting for process P0.

The answer is a non-negative integer.
```
**Answer:** `0`

**Prompt:**
```
We run fixed-quantum round-robin scheduling with quantum 8 time units on 7 single-CPU processes.
Arrival time, burst time:
(P0: arrives at 0, burst 13), (P1: arrives at 0, burst 10), (P2: arrives at 2, burst 7), (P3: arrives at 2, burst 5), (P4: arrives at 6, burst 4), (P5: arrives at 6, burst 10), (P6: arrives at 9, burst 14)
The CPU serves the ready queue in FIFO order, each ready process getting at most one quantum of CPU before returning to the tail of the queue; a newly arrived process joins at the tail. A process that finishes its remaining burst before exhausting a quantum leaves the schedule.
Report waiting: waiting for process P2.

The answer is a non-negative integer.
```
**Answer:** `14`

