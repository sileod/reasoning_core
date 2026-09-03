# Level 0

## Prompt

As a voting official, consider an election over candidates 0, 1, 2. Each ballot is a strict ranking of every candidate from most preferred to least preferred. A candidate is the Condorcet winner if, in a pairwise contest against every other candidate, they are ranked above that candidate on a strict majority of the ballots.

Ballot 1: 1, 0, 2
Ballot 2: 1, 2, 0
Ballot 3: 2, 0, 1
Ballot 4: 2, 1, 0
Ballot 5: 0, 1, 2

Output the Condorcet winner as a single number (0-based candidate index), or output None if no Condorcet winner exists.

## Answer

1


## Prompt

As a voting official, consider an election over candidates 0, 1, 2. Each ballot is a strict ranking of every candidate from most preferred to least preferred. A candidate is the Condorcet winner if, in a pairwise contest against every other candidate, they are ranked above that candidate on a strict majority of the ballots.

Ballot 1: 1, 2, 0
Ballot 2: 2, 1, 0
Ballot 3: 0, 1, 2
Ballot 4: 0, 2, 1
Ballot 5: 1, 0, 2

Output the Condorcet winner as a single number (0-based candidate index), or output None if no Condorcet winner exists.

## Answer

1


# Level 2

## Prompt

As a voting official, consider an election over candidates 0, 1, 2, 3, 4. Each ballot is a strict ranking of every candidate from most preferred to least preferred. A candidate is the Condorcet winner if, in a pairwise contest against every other candidate, they are ranked above that candidate on a strict majority of the ballots.

Ballot 1: 4, 0, 1, 2, 3
Ballot 2: 4, 1, 2, 0, 3
Ballot 3: 4, 1, 0, 3, 2
Ballot 4: 4, 2, 1, 3, 0
Ballot 5: 4, 3, 1, 2, 0
Ballot 6: 0, 1, 4, 2, 3
Ballot 7: 1, 0, 4, 3, 2
Ballot 8: 2, 0, 3, 1, 4
Ballot 9: 1, 2, 4, 0, 3

Output the Condorcet winner as a single number (0-based candidate index), or output None if no Condorcet winner exists.

## Answer

4


## Prompt

As a voting official, consider an election over candidates 0, 1, 2, 3, 4. Each ballot is a strict ranking of every candidate from most preferred to least preferred. A candidate is the Condorcet winner if, in a pairwise contest against every other candidate, they are ranked above that candidate on a strict majority of the ballots.

Ballot 1: 1, 3, 0, 4, 2
Ballot 2: 1, 2, 4, 3, 0
Ballot 3: 1, 4, 3, 2, 0
Ballot 4: 1, 3, 0, 2, 4
Ballot 5: 1, 0, 2, 3, 4
Ballot 6: 2, 4, 3, 0, 1
Ballot 7: 0, 4, 2, 3, 1
Ballot 8: 2, 3, 4, 0, 1
Ballot 9: 2, 3, 1, 0, 4

Output the Condorcet winner as a single number (0-based candidate index), or output None if no Condorcet winner exists.

## Answer

1


# Level 5

## Prompt

As a voting official, consider an election over candidates 0, 1, 2, 3, 4, 5, 6, 7. Each ballot is a strict ranking of every candidate from most preferred to least preferred. A candidate is the Condorcet winner if, in a pairwise contest against every other candidate, they are ranked above that candidate on a strict majority of the ballots.

Ballot 1: 5, 2, 0, 3, 1, 7, 6, 4
Ballot 2: 5, 0, 3, 1, 6, 4, 2, 7
Ballot 3: 5, 3, 0, 6, 2, 7, 4, 1
Ballot 4: 5, 6, 1, 0, 4, 7, 3, 2
Ballot 5: 5, 4, 2, 1, 3, 0, 7, 6
Ballot 6: 5, 3, 2, 7, 6, 0, 4, 1
Ballot 7: 5, 6, 2, 0, 4, 7, 3, 1
Ballot 8: 5, 7, 4, 6, 2, 1, 3, 0
Ballot 9: 3, 5, 6, 4, 2, 7, 0, 1
Ballot 10: 3, 2, 5, 7, 0, 6, 1, 4
Ballot 11: 1, 3, 4, 7, 6, 5, 2, 0
Ballot 12: 3, 0, 7, 1, 2, 6, 5, 4
Ballot 13: 7, 0, 2, 4, 6, 5, 3, 1
Ballot 14: 4, 2, 5, 3, 7, 6, 1, 0
Ballot 15: 1, 5, 4, 3, 6, 2, 7, 0

Output the Condorcet winner as a single number (0-based candidate index), or output None if no Condorcet winner exists.

## Answer

5


## Prompt

As a voting official, consider an election over candidates 0, 1, 2, 3, 4, 5, 6, 7. Each ballot is a strict ranking of every candidate from most preferred to least preferred. A candidate is the Condorcet winner if, in a pairwise contest against every other candidate, they are ranked above that candidate on a strict majority of the ballots.

Ballot 1: 1, 4, 6, 0, 3, 7, 5, 2
Ballot 2: 1, 2, 7, 5, 3, 0, 4, 6
Ballot 3: 1, 4, 6, 7, 2, 0, 3, 5
Ballot 4: 1, 4, 7, 3, 6, 2, 0, 5
Ballot 5: 1, 0, 5, 2, 4, 6, 7, 3
Ballot 6: 1, 2, 6, 4, 0, 7, 5, 3
Ballot 7: 1, 4, 3, 7, 0, 5, 2, 6
Ballot 8: 1, 5, 2, 3, 6, 4, 7, 0
Ballot 9: 0, 6, 2, 4, 1, 5, 3, 7
Ballot 10: 3, 7, 4, 0, 5, 2, 1, 6
Ballot 11: 0, 7, 5, 4, 3, 2, 6, 1
Ballot 12: 3, 7, 0, 6, 1, 5, 4, 2
Ballot 13: 6, 2, 1, 4, 0, 3, 5, 7
Ballot 14: 4, 2, 6, 5, 0, 3, 7, 1
Ballot 15: 5, 2, 1, 0, 3, 6, 4, 7

Output the Condorcet winner as a single number (0-based candidate index), or output None if no Condorcet winner exists.

## Answer

1

