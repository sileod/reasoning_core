# Round Robin Completion samples (P050v1)

## Level 0
**Prompt:**
Scheduling 4 processes with a Round Robin CPU scheduler with time quantum 2.
P1: arrive t=0, burst=5, P2: arrive t=1, burst=2, P3: arrive t=2, burst=6, P4: arrive t=4, burst=4.
Processes are scheduled in increasing index order at each round. A process runs for up to one quantum when it arrives (or is ready), then yields; if it finishes within its quantum, it completes at that moment and is output. Idle time passes when no process is ready.
Output the process completion order as process IDs separated by commas (e.g. '2,1,3').

**Answer:**
2,4,1,3

**Prompt:**
Scheduling 4 processes with a Round Robin CPU scheduler with time quantum 2.
P1: arrive t=1, burst=4, P2: arrive t=4, burst=3, P3: arrive t=2, burst=2, P4: arrive t=4, burst=2.
Processes are scheduled in increasing index order at each round. A process runs for up to one quantum when it arrives (or is ready), then yields; if it finishes within its quantum, it completes at that moment and is output. Idle time passes when no process is ready.
Output the process completion order as process IDs separated by commas (e.g. '2,1,3').

**Answer:**
3,4,1,2

## Level 2
**Prompt:**
Scheduling 6 processes with a Round Robin CPU scheduler with time quantum 2.
P1: arrive t=0, burst=1, P2: arrive t=2, burst=3, P3: arrive t=0, burst=4, P4: arrive t=3, burst=6, P5: arrive t=1, burst=4, P6: arrive t=2, burst=2.
Processes are scheduled in increasing index order at each round. A process runs for up to one quantum when it arrives (or is ready), then yields; if it finishes within its quantum, it completes at that moment and is output. Idle time passes when no process is ready.
Output the process completion order as process IDs separated by commas (e.g. '2,1,3').

**Answer:**
1,6,2,3,5,4

**Prompt:**
Scheduling 6 processes with a Round Robin CPU scheduler with time quantum 2.
P1: arrive t=1, burst=3, P2: arrive t=3, burst=3, P3: arrive t=6, burst=7, P4: arrive t=3, burst=1, P5: arrive t=3, burst=7, P6: arrive t=2, burst=5.
Processes are scheduled in increasing index order at each round. A process runs for up to one quantum when it arrives (or is ready), then yields; if it finishes within its quantum, it completes at that moment and is output. Idle time passes when no process is ready.
Output the process completion order as process IDs separated by commas (e.g. '2,1,3').

**Answer:**
4,1,2,6,3,5

## Level 5
**Prompt:**
Scheduling 9 processes with a Round Robin CPU scheduler with time quantum 2.
P1: arrive t=4, burst=3, P2: arrive t=1, burst=4, P3: arrive t=2, burst=8, P4: arrive t=6, burst=4, P5: arrive t=2, burst=6, P6: arrive t=7, burst=4, P7: arrive t=4, burst=11, P8: arrive t=1, burst=7, P9: arrive t=4, burst=6.
Processes are scheduled in increasing index order at each round. A process runs for up to one quantum when it arrives (or is ready), then yields; if it finishes within its quantum, it completes at that moment and is output. Idle time passes when no process is ready.
Output the process completion order as process IDs separated by commas (e.g. '2,1,3').

**Answer:**
1,2,4,6,5,9,3,8,7

**Prompt:**
Scheduling 9 processes with a Round Robin CPU scheduler with time quantum 2.
P1: arrive t=1, burst=10, P2: arrive t=1, burst=7, P3: arrive t=0, burst=4, P4: arrive t=8, burst=5, P5: arrive t=3, burst=2, P6: arrive t=3, burst=3, P7: arrive t=4, burst=9, P8: arrive t=7, burst=2, P9: arrive t=1, burst=4.
Processes are scheduled in increasing index order at each round. A process runs for up to one quantum when it arrives (or is ready), then yields; if it finishes within its quantum, it completes at that moment and is output. Idle time passes when no process is ready.
Output the process completion order as process IDs separated by commas (e.g. '2,1,3').

**Answer:**
5,8,3,6,9,4,2,1,7

