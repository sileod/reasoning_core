# Samples P041v1: paxos_chosen_value

## Level 0

### Example

Prompt:

In a Paxos round, 2 acceptors form a quorum. A value is considered chosen if some proposal received accepted votes from at least a quorum of acceptors. If several proposals have a quorum, the one with the highest proposal number decides. Votes:
Acceptor 0 voted for proposal 1 with value 64.
Acceptor 1 voted for proposal 1 with value 64.
Acceptor 2 voted for proposal 2 with value 110.

What value is already chosen? Answer with the chosen value (an integer), or the word None if no proposal has a quorum.

Answer:

64

### Example

Prompt:

In a Paxos round, 2 acceptors form a quorum. A value is considered chosen if some proposal received accepted votes from at least a quorum of acceptors. If several proposals have a quorum, the one with the highest proposal number decides. Votes:
Acceptor 0 voted for proposal 2 with value 33.
Acceptor 1 voted for proposal 2 with value 33.
Acceptor 2 voted for proposal 3 with value 75.

What value is already chosen? Answer with the chosen value (an integer), or the word None if no proposal has a quorum.

Answer:

33

## Level 2

### Example

Prompt:

In a Paxos round, 3 acceptors form a quorum. A value is considered chosen if some proposal received accepted votes from at least a quorum of acceptors. If several proposals have a quorum, the one with the highest proposal number decides. Votes:
Acceptor 0 voted for proposal 5 with value 50.

What value is already chosen? Answer with the chosen value (an integer), or the word None if no proposal has a quorum.

Answer:

None

### Example

Prompt:

In a Paxos round, 3 acceptors form a quorum. A value is considered chosen if some proposal received accepted votes from at least a quorum of acceptors. If several proposals have a quorum, the one with the highest proposal number decides. Votes:
Acceptor 2 voted for proposal 4 with value 11.
Acceptor 3 voted for proposal 4 with value 11.
Acceptor 4 voted for proposal 5 with value 3.
Acceptor 1 voted for proposal 4 with value 11.
Acceptor 0 voted for proposal 3 with value 49.

What value is already chosen? Answer with the chosen value (an integer), or the word None if no proposal has a quorum.

Answer:

11

## Level 5

### Example

Prompt:

In a Paxos round, 5 acceptors form a quorum. A value is considered chosen if some proposal received accepted votes from at least a quorum of acceptors. If several proposals have a quorum, the one with the highest proposal number decides. Votes:
Acceptor 5 voted for proposal 9 with value 30.
Acceptor 7 voted for proposal 11 with value 61.
Acceptor 1 voted for proposal 9 with value 30.
Acceptor 3 voted for proposal 9 with value 30.
Acceptor 2 voted for proposal 9 with value 30.
Acceptor 4 voted for proposal 9 with value 30.
Acceptor 0 voted for proposal 6 with value 103.
Acceptor 6 voted for proposal 11 with value 12.

What value is already chosen? Answer with the chosen value (an integer), or the word None if no proposal has a quorum.

Answer:

30

### Example

Prompt:

In a Paxos round, 5 acceptors form a quorum. A value is considered chosen if some proposal received accepted votes from at least a quorum of acceptors. If several proposals have a quorum, the one with the highest proposal number decides. Votes:
Acceptor 2 voted for proposal 1 with value 3.
Acceptor 6 voted for proposal 7 with value 1.
Acceptor 0 voted for proposal 1 with value 3.
Acceptor 3 voted for proposal 1 with value 3.
Acceptor 1 voted for proposal 1 with value 3.
Acceptor 4 voted for proposal 1 with value 3.
Acceptor 7 voted for proposal 7 with value 87.
Acceptor 5 voted for proposal 2 with value 88.

What value is already chosen? Answer with the chosen value (an integer), or the word None if no proposal has a quorum.

Answer:

3
