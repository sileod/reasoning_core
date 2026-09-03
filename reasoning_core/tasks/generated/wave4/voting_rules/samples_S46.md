## Level 0

### Example 1

**Prompt:**

Candidates:
['A', 'B', 'C', 'D']

Ballot Lines:
['5 voters: B > C > D > A', '5 voters: D > B > A > C', '1 voters: D > C > A > B', '3 voters: D > A > C > B', '4 voters: B > C > A > D']

Each ballot lists candidates from most preferred to least preferred; the quoted number is how many voters submitted that ranking. Compute the Borda score of every candidate: 3 points per first-place vote, 2 per second-place vote, down to 0 for last place. Give every candidate in alphabetical order as exact strings in the format:
Borda: A=s, B=s, ...
(one per candidate, in that order).

**Answer:**

`A=16,C=23,D=32,B=37`

### Example 2

**Prompt:**

Candidates:
['A', 'B', 'C', 'D']

Ballot Lines:
['4 voters: B > C > A > D', '1 voters: D > C > B > A', '5 voters: C > D > B > A', '3 voters: D > B > A > C', '3 voters: C > A > B > D']

Each ballot lists candidates from most preferred to least preferred; the quoted number is how many voters submitted that ranking. In instant-runoff voting, each round counts only each voter's highest-ranked remaining candidate. Eliminate the candidate with the fewest first-place votes; ties break by eliminating the alphabetically first. Repeat until one remains. Give the elimination order, first eliminated to winner, as a comma-separated list of candidates:
Order: X, Y, Z, ...


**Answer:**

`A,B,D,C`

## Level 2

### Example 1

**Prompt:**

Candidates:
['A', 'B', 'C', 'D']

Ballot Lines:
['8 voters: B > C > A > D', '6 voters: A > B > D > C', '1 voters: A > B > C > D', '2 voters: C > A > D > B', '3 voters: C > B > D > A']

Each ballot lists candidates from most preferred to least preferred; the quoted number is how many voters submitted that ranking. Compute the Borda score of every candidate: 3 points per first-place vote, 2 per second-place vote, down to 0 for last place. Give every candidate in alphabetical order as exact strings in the format:
Borda: A=s, B=s, ...
(one per candidate, in that order).

**Answer:**

`D=11,C=32,A=33,B=44`

### Example 2

**Prompt:**

Candidates:
['A', 'B', 'C', 'D']

Ballot Lines:
['3 voters: B > C > D > A', '2 voters: C > D > B > A', '5 voters: A > B > D > C', '6 voters: C > D > B > A', '4 voters: A > B > D > C']

Each ballot lists candidates from most preferred to least preferred; the quoted number is how many voters submitted that ranking. In instant-runoff voting, each round counts only each voter's highest-ranked remaining candidate. Eliminate the candidate with the fewest first-place votes; ties break by eliminating the alphabetically first. Repeat until one remains. Give the elimination order, first eliminated to winner, as a comma-separated list of candidates:
Order: X, Y, Z, ...


**Answer:**

`D,B,A,C`

## Level 5

### Example 1

**Prompt:**

Candidates:
['A', 'B', 'C', 'D', 'E', 'F', 'G']

Ballot Lines:
['4 voters: G > D > C > A > B > F > E', '1 voters: C > A > G > D > F > E > B', '4 voters: E > D > C > A > G > F > B', '7 voters: E > G > C > B > D > A > F', '6 voters: E > A > C > D > B > F > G', '2 voters: E > D > F > A > C > G > B', '3 voters: A > F > G > D > C > B > E', '5 voters: C > G > F > E > B > A > D', '7 voters: G > A > C > F > E > D > B']

Each ballot lists candidates from most preferred to least preferred; the quoted number is how many voters submitted that ranking. In instant-runoff voting, each round counts only each voter's highest-ranked remaining candidate. Eliminate the candidate with the fewest first-place votes; ties break by eliminating the alphabetically first. Repeat until one remains. Give the elimination order, first eliminated to winner, as a comma-separated list of candidates:
Order: X, Y, Z, ...


**Answer:**

`B,D,F,A,C,E,G`

### Example 2

**Prompt:**

Candidates:
['A', 'B', 'C', 'D', 'E', 'F', 'G']

Ballot Lines:
['3 voters: F > G > E > D > A > C > B', '11 voters: E > A > C > B > D > G > F', '1 voters: C > D > G > A > E > B > F', '8 voters: B > C > D > G > A > E > F', '8 voters: F > A > D > B > G > E > C', '1 voters: E > G > A > D > B > F > C', '3 voters: D > C > E > G > A > F > B', '9 voters: B > D > G > C > E > F > A', '6 voters: E > G > B > C > F > D > A']

Each ballot lists candidates from most preferred to least preferred; the quoted number is how many voters submitted that ranking. Compute the Borda score of every candidate: 6 points per first-place vote, 5 per second-place vote, down to 0 for last place. Give every candidate in alphabetical order as exact strings in the format:
Borda: A=s, B=s, ...
(one per candidate, in that order).

**Answer:**

`F=91,A=130,G=150,C=153,E=168,D=172,B=186`
