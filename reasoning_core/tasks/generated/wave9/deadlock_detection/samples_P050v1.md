## Level 0

### Example 1

Query:
There are 4 processes, each running on a single-instance resource system with a fixed set of resources. Every resource can be held by at most one process at a time. Each process currently holds some resources and is simultaneously requesting one or more additional resources. A process blocks until it obtains every resource it requests at the same time; once it does, it finishes and releases all the resources it held. A process that requests a resource already held by another process waits for that other process to finish and release it.

Holders:
Resource ownership (each row is 'resource: current holder process'):
R0: P222
R1: P149
R2: P416
R3: P222
R4: P732
R5: P416

Requests:
Outstanding requests (each row is 'process: resources it is requesting'):
P149: [416, 222, 416]
P222: [149]
P416: [222, 732]
P732: none

Build the wait-for relation from who holds each requested resource: process A waits on process B when A requests a resource B currently holds. A deadlock exists when some processes can never obtain their requested resources because they are waiting (directly or through a chain) on resources held forever by processes waiting on each other; those processes can never complete.

If a deadlock exists, the answer is the deadlocked set: every process that can never complete, written as their labels in ascending order, separated by commas (for example "1,3,5").

If there is no deadlock, all processes can complete in some order. Produce the canonical safe sequence by this rule: repeatedly, among processes that have not yet completed and whose every requested resource is currently free (not held by any unfinished process), complete the one with the smallest label and release the resources it held. The answer is that completion order, labels separated by commas (for example "0,2,1").

The answer is a single comma-separated list of process labels: the ascending deadlocked set, or the safe completion order.

**Answer:** 149,222,416

### Example 2

Query:
There are 4 processes, each running on a single-instance resource system with a fixed set of resources. Every resource can be held by at most one process at a time. Each process currently holds some resources and is simultaneously requesting one or more additional resources. A process blocks until it obtains every resource it requests at the same time; once it does, it finishes and releases all the resources it held. A process that requests a resource already held by another process waits for that other process to finish and release it.

Holders:
Resource ownership (each row is 'resource: current holder process'):
R0: P67
R1: P67
R2: P85
R3: P501
R4: P696
R5: P85

Requests:
Outstanding requests (each row is 'process: resources it is requesting'):
P67: [85]
P85: none
P501: none
P696: none

Build the wait-for relation from who holds each requested resource: process A waits on process B when A requests a resource B currently holds. A deadlock exists when some processes can never obtain their requested resources because they are waiting (directly or through a chain) on resources held forever by processes waiting on each other; those processes can never complete.

If a deadlock exists, the answer is the deadlocked set: every process that can never complete, written as their labels in ascending order, separated by commas (for example "1,3,5").

If there is no deadlock, all processes can complete in some order. Produce the canonical safe sequence by this rule: repeatedly, among processes that have not yet completed and whose every requested resource is currently free (not held by any unfinished process), complete the one with the smallest label and release the resources it held. The answer is that completion order, labels separated by commas (for example "0,2,1").

The answer is a single comma-separated list of process labels: the ascending deadlocked set, or the safe completion order.

**Answer:** 85,67,501,696

## Level 2

### Example 1

Query:
There are 8 processes, each running on a single-instance resource system with a fixed set of resources. Every resource can be held by at most one process at a time. Each process currently holds some resources and is simultaneously requesting one or more additional resources. A process blocks until it obtains every resource it requests at the same time; once it does, it finishes and releases all the resources it held. A process that requests a resource already held by another process waits for that other process to finish and release it.

Holders:
Resource ownership (each row is 'resource: current holder process'):
R0: P857
R1: P857
R2: P825
R3: P950
R4: P950
R5: P380
R6: P525
R7: P255
R8: P857
R9: P380
R10: P525
R11: P477

Requests:
Outstanding requests (each row is 'process: resources it is requesting'):
P255: [857]
P380: [857, 857, 950]
P477: none
P525: [950]
P603: none
P825: [380, 380]
P857: none
P950: [857, 857, 857]

Build the wait-for relation from who holds each requested resource: process A waits on process B when A requests a resource B currently holds. A deadlock exists when some processes can never obtain their requested resources because they are waiting (directly or through a chain) on resources held forever by processes waiting on each other; those processes can never complete.

If a deadlock exists, the answer is the deadlocked set: every process that can never complete, written as their labels in ascending order, separated by commas (for example "1,3,5").

If there is no deadlock, all processes can complete in some order. Produce the canonical safe sequence by this rule: repeatedly, among processes that have not yet completed and whose every requested resource is currently free (not held by any unfinished process), complete the one with the smallest label and release the resources it held. The answer is that completion order, labels separated by commas (for example "0,2,1").

The answer is a single comma-separated list of process labels: the ascending deadlocked set, or the safe completion order.

**Answer:** 477,603,857,255,950,380,525,825

### Example 2

Query:
There are 8 processes, each running on a single-instance resource system with a fixed set of resources. Every resource can be held by at most one process at a time. Each process currently holds some resources and is simultaneously requesting one or more additional resources. A process blocks until it obtains every resource it requests at the same time; once it does, it finishes and releases all the resources it held. A process that requests a resource already held by another process waits for that other process to finish and release it.

Holders:
Resource ownership (each row is 'resource: current holder process'):
R0: P887
R1: P541
R2: P541
R3: P480
R4: P272
R5: P272
R6: P480
R7: P887
R8: P272
R9: P596
R10: P887
R11: P596

Requests:
Outstanding requests (each row is 'process: resources it is requesting'):
P272: [887, 480, 596]
P458: none
P480: [887, 887]
P541: [272]
P596: [480]
P793: [272, 272, 596]
P887: none
P966: [887, 480, 887, 887]

Build the wait-for relation from who holds each requested resource: process A waits on process B when A requests a resource B currently holds. A deadlock exists when some processes can never obtain their requested resources because they are waiting (directly or through a chain) on resources held forever by processes waiting on each other; those processes can never complete.

If a deadlock exists, the answer is the deadlocked set: every process that can never complete, written as their labels in ascending order, separated by commas (for example "1,3,5").

If there is no deadlock, all processes can complete in some order. Produce the canonical safe sequence by this rule: repeatedly, among processes that have not yet completed and whose every requested resource is currently free (not held by any unfinished process), complete the one with the smallest label and release the resources it held. The answer is that completion order, labels separated by commas (for example "0,2,1").

The answer is a single comma-separated list of process labels: the ascending deadlocked set, or the safe completion order.

**Answer:** 458,887,480,596,272,541,793,966

## Level 5

### Example 1

Query:
There are 14 processes, each running on a single-instance resource system with a fixed set of resources. Every resource can be held by at most one process at a time. Each process currently holds some resources and is simultaneously requesting one or more additional resources. A process blocks until it obtains every resource it requests at the same time; once it does, it finishes and releases all the resources it held. A process that requests a resource already held by another process waits for that other process to finish and release it.

Holders:
Resource ownership (each row is 'resource: current holder process'):
R0: P399
R1: P882
R2: P399
R3: P842
R4: P835
R5: P737
R6: P525
R7: P525
R8: P525
R9: P737
R10: P399
R11: P882
R12: P737
R13: P882
R14: P737
R15: P68
R16: P622
R17: P645
R18: P934
R19: P645
R20: P510

Requests:
Outstanding requests (each row is 'process: resources it is requesting'):
P68: [835, 737, 525, 525, 525, 737, 645]
P198: [399, 835, 525, 737, 68, 645, 510]
P399: none
P409: [842, 835, 737, 737, 645, 645, 510]
P510: [737, 737]
P525: [737, 510]
P622: [525, 737, 399, 68]
P645: [737, 510]
P737: none
P835: [842, 737, 737, 645]
P842: [737, 737, 737, 737]
P882: [737, 525, 737]
P934: [399, 835, 737, 399, 882, 882, 645]
P968: [842, 737, 737, 645]

Build the wait-for relation from who holds each requested resource: process A waits on process B when A requests a resource B currently holds. A deadlock exists when some processes can never obtain their requested resources because they are waiting (directly or through a chain) on resources held forever by processes waiting on each other; those processes can never complete.

If a deadlock exists, the answer is the deadlocked set: every process that can never complete, written as their labels in ascending order, separated by commas (for example "1,3,5").

If there is no deadlock, all processes can complete in some order. Produce the canonical safe sequence by this rule: repeatedly, among processes that have not yet completed and whose every requested resource is currently free (not held by any unfinished process), complete the one with the smallest label and release the resources it held. The answer is that completion order, labels separated by commas (for example "0,2,1").

The answer is a single comma-separated list of process labels: the ascending deadlocked set, or the safe completion order.

**Answer:** 399,737,510,525,645,842,835,68,198,409,622,882,934,968

### Example 2

Query:
There are 14 processes, each running on a single-instance resource system with a fixed set of resources. Every resource can be held by at most one process at a time. Each process currently holds some resources and is simultaneously requesting one or more additional resources. A process blocks until it obtains every resource it requests at the same time; once it does, it finishes and releases all the resources it held. A process that requests a resource already held by another process waits for that other process to finish and release it.

Holders:
Resource ownership (each row is 'resource: current holder process'):
R0: P312
R1: P22
R2: P879
R3: P590
R4: P892
R5: P397
R6: P457
R7: P22
R8: P312
R9: P650
R10: P892
R11: P763
R12: P650
R13: P22
R14: P397
R15: P650
R16: P457
R17: P5
R18: P879
R19: P520
R20: P312

Requests:
Outstanding requests (each row is 'process: resources it is requesting'):
P5: [650]
P22: none
P96: [590, 22, 650, 22, 397, 457, 879]
P312: [879, 590, 397, 650, 650, 22, 650]
P397: [22, 763, 457]
P406: [22, 650]
P457: [22, 22]
P520: [22, 590, 457, 650]
P590: [22, 397, 457, 763, 22, 397, 457]
P650: [22, 22]
P763: [22, 22]
P829: [22, 457]
P879: [22, 397, 763, 457]
P892: [312, 397, 650, 397, 5, 312]

Build the wait-for relation from who holds each requested resource: process A waits on process B when A requests a resource B currently holds. A deadlock exists when some processes can never obtain their requested resources because they are waiting (directly or through a chain) on resources held forever by processes waiting on each other; those processes can never complete.

If a deadlock exists, the answer is the deadlocked set: every process that can never complete, written as their labels in ascending order, separated by commas (for example "1,3,5").

If there is no deadlock, all processes can complete in some order. Produce the canonical safe sequence by this rule: repeatedly, among processes that have not yet completed and whose every requested resource is currently free (not held by any unfinished process), complete the one with the smallest label and release the resources it held. The answer is that completion order, labels separated by commas (for example "0,2,1").

The answer is a single comma-separated list of process labels: the ascending deadlocked set, or the safe completion order.

**Answer:** 22,457,650,5,406,763,397,590,520,829,879,96,312,892

