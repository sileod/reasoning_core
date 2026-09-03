# samples_P037v1

## Level 0

### Example 1

**Prompt:**

In a message-passing system, process P0..P1 each execute an ordered sequence of events; process sizes (number of events): P0:3 P1:5.
A global cut selects a prefix of each process: P0:2 P1:5 (cut point = number of included events, can be 0 up to the process size).
A message is sent by one process (at its send position, 1=first event) and received by another (at its receive position). Message edges: m0 P0@3->P1@2; m1 P0@3->P1@5; m2 P1@5->P0@2; m3 P0@3->P1@4; m4 P1@5->P0@3; m5 P0@3->P1@5; m6 P0@1->P1@3; m7 P0@2->P1@5.
A message is orphaned (the cut is inconsistent at that message) when its send position lies at or before the sender's cut point but its receive position lies strictly after the receiver's cut point.
How many messages are orphaned by this cut? The answer is a non-negative integer: the number of orphaned messages.

**Answer:**

1


### Example 2

**Prompt:**

In a message-passing system, process P0..P1 each execute an ordered sequence of events; process sizes (number of events): P0:4 P1:4.
A global cut selects a prefix of each process: P0:1 P1:4 (cut point = number of included events, can be 0 up to the process size).
A message is sent by one process (at its send position, 1=first event) and received by another (at its receive position). Message edges: m0 P0@2->P1@3; m1 P0@4->P1@1; m2 P1@3->P0@1; m3 P1@2->P0@1; m4 P1@4->P0@1; m5 P1@3->P0@3; m6 P0@1->P1@3; m7 P1@2->P0@3.
A message is orphaned (the cut is inconsistent at that message) when its send position lies at or before the sender's cut point but its receive position lies strictly after the receiver's cut point.
How many messages are orphaned by this cut? The answer is a non-negative integer: the number of orphaned messages.

**Answer:**

2

## Level 2

### Example 1

**Prompt:**

In a message-passing system, process P0..P3 each execute an ordered sequence of events; process sizes (number of events): P0:7 P1:6 P2:5 P3:5.
A global cut selects a prefix of each process: P0:4 P1:3 P2:3 P3:1 (cut point = number of included events, can be 0 up to the process size).
A message is sent by one process (at its send position, 1=first event) and received by another (at its receive position). Message edges: m0 P3@2->P2@3; m1 P3@2->P0@1; m2 P2@1->P3@2; m3 P1@4->P2@3; m4 P2@1->P3@3; m5 P2@1->P3@3; m6 P2@1->P1@5; m7 P2@2->P3@2; m8 P0@2->P1@2; m9 P0@2->P3@1; m10 P3@1->P2@5; m11 P1@1->P0@1; m12 P2@3->P0@5; m13 P0@2->P3@2.
A message is orphaned (the cut is inconsistent at that message) when its send position lies at or before the sender's cut point but its receive position lies strictly after the receiver's cut point.
How many messages are orphaned by this cut? The answer is a non-negative integer: the number of orphaned messages.

**Answer:**

8


### Example 2

**Prompt:**

In a message-passing system, process P0..P3 each execute an ordered sequence of events; process sizes (number of events): P0:5 P1:5 P2:7 P3:5.
A global cut selects a prefix of each process: P0:0 P1:2 P2:7 P3:2 (cut point = number of included events, can be 0 up to the process size).
A message is sent by one process (at its send position, 1=first event) and received by another (at its receive position). Message edges: m0 P2@5->P1@2; m1 P2@4->P3@2; m2 P0@3->P3@2; m3 P0@4->P3@5; m4 P3@1->P1@1; m5 P1@3->P0@4; m6 P3@2->P0@4; m7 P2@1->P0@5; m8 P2@2->P0@3; m9 P1@1->P3@3; m10 P0@2->P1@3; m11 P1@2->P3@1; m12 P0@5->P1@1; m13 P0@5->P1@1.
A message is orphaned (the cut is inconsistent at that message) when its send position lies at or before the sender's cut point but its receive position lies strictly after the receiver's cut point.
How many messages are orphaned by this cut? The answer is a non-negative integer: the number of orphaned messages.

**Answer:**

4

## Level 5

### Example 1

**Prompt:**

In a message-passing system, process P0..P6 each execute an ordered sequence of events; process sizes (number of events): P0:8 P1:8 P2:9 P3:8 P4:8 P5:10 P6:8.
A global cut selects a prefix of each process: P0:0 P1:3 P2:0 P3:5 P4:2 P5:3 P6:0 (cut point = number of included events, can be 0 up to the process size).
A message is sent by one process (at its send position, 1=first event) and received by another (at its receive position). Message edges: m0 P0@6->P3@5; m1 P4@8->P2@9; m2 P0@7->P1@8; m3 P2@6->P5@2; m4 P6@4->P1@2; m5 P2@2->P1@2; m6 P3@3->P5@3; m7 P2@4->P5@10; m8 P6@7->P5@3; m9 P5@1->P1@6; m10 P4@3->P5@1; m11 P2@2->P3@2; m12 P3@7->P1@7; m13 P4@1->P6@2; m14 P6@5->P3@3; m15 P1@6->P2@6; m16 P3@3->P2@3; m17 P4@6->P5@2; m18 P4@2->P0@3; m19 P6@3->P1@4; m20 P4@2->P0@6; m21 P6@6->P1@2; m22 P6@2->P3@2.
A message is orphaned (the cut is inconsistent at that message) when its send position lies at or before the sender's cut point but its receive position lies strictly after the receiver's cut point.
How many messages are orphaned by this cut? The answer is a non-negative integer: the number of orphaned messages.

**Answer:**

5


### Example 2

**Prompt:**

In a message-passing system, process P0..P6 each execute an ordered sequence of events; process sizes (number of events): P0:9 P1:10 P2:9 P3:9 P4:9 P5:10 P6:9.
A global cut selects a prefix of each process: P0:7 P1:9 P2:4 P3:8 P4:7 P5:1 P6:6 (cut point = number of included events, can be 0 up to the process size).
A message is sent by one process (at its send position, 1=first event) and received by another (at its receive position). Message edges: m0 P0@7->P1@10; m1 P0@9->P6@2; m2 P2@3->P1@10; m3 P2@1->P3@9; m4 P4@3->P6@8; m5 P3@4->P2@4; m6 P1@5->P0@9; m7 P2@4->P6@8; m8 P1@10->P0@2; m9 P1@1->P5@1; m10 P3@9->P0@3; m11 P0@2->P2@5; m12 P2@8->P4@6; m13 P4@9->P1@3; m14 P0@6->P1@8; m15 P2@6->P1@5; m16 P0@8->P3@8; m17 P6@4->P5@1; m18 P5@2->P3@9; m19 P0@7->P1@10; m20 P4@1->P3@7; m21 P1@10->P0@6; m22 P0@9->P3@9.
A message is orphaned (the cut is inconsistent at that message) when its send position lies at or before the sender's cut point but its receive position lies strictly after the receiver's cut point.
How many messages are orphaned by this cut? The answer is a non-negative integer: the number of orphaned messages.

**Answer:**

8

