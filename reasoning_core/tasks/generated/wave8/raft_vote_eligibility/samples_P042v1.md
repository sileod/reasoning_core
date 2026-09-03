
## Level 0

### Prompt

In a Raft cluster, the candidates are [52, 78, 99]; candidate 52 runs at term 1 ; candidate 78 runs at term 4 ; candidate 99 runs at term 1. The current leader is 99 with candidate term 4 and last log entry (term 1, index 5). Voter 15 was last recorded voting for term 0 and has last log entry (term 3, index 13). Per the Raft log-matching safety rule, this voter may grant the vote to the leader exactly when the leader's candidate term is strictly greater than the term the voter last voted for (the request is not stale) and the leader's log is at least as up-to-date as the voter's: leader log term greater than the voter's, or equal log term with leader index at least the voter's index. Otherwise the voter refuses, either because the vote is stale or because the leader's log is not up-to-date.

Decide whether voter 15 may grant the vote to 99. Answer exactly one of three forms: "grant=<candidate term>", "stale=<term the voter last voted for>", or "log=<voter's last log index>", quoting the witness value after the '='.

### Answer

log=13


### Prompt

In a Raft cluster, the candidates are [39, 64, 89]; candidate 39 runs at term 2 ; candidate 64 runs at term 3 ; candidate 89 runs at term 3. The current leader is 89 with candidate term 1 and last log entry (term 2, index 12). Voter 16 was last recorded voting for term 0 and has last log entry (term 1, index 3). Per the Raft log-matching safety rule, this voter may grant the vote to the leader exactly when the leader's candidate term is strictly greater than the term the voter last voted for (the request is not stale) and the leader's log is at least as up-to-date as the voter's: leader log term greater than the voter's, or equal log term with leader index at least the voter's index. Otherwise the voter refuses, either because the vote is stale or because the leader's log is not up-to-date.

Decide whether voter 16 may grant the vote to 89. Answer exactly one of three forms: "grant=<candidate term>", "stale=<term the voter last voted for>", or "log=<voter's last log index>", quoting the witness value after the '='.

### Answer

grant=1



## Level 2

### Prompt

In a Raft cluster, the candidates are [63, 76, 78, 83, 96]; candidate 63 runs at term 3 ; candidate 76 runs at term 5 ; candidate 78 runs at term 1 ; candidate 83 runs at term 1 ; candidate 96 runs at term 3. The current leader is 78 with candidate term 1 and last log entry (term 1, index 0). Voter 20 was last recorded voting for term 1 and has last log entry (term 4, index 3). Per the Raft log-matching safety rule, this voter may grant the vote to the leader exactly when the leader's candidate term is strictly greater than the term the voter last voted for (the request is not stale) and the leader's log is at least as up-to-date as the voter's: leader log term greater than the voter's, or equal log term with leader index at least the voter's index. Otherwise the voter refuses, either because the vote is stale or because the leader's log is not up-to-date.

Decide whether voter 20 may grant the vote to 78. Answer exactly one of three forms: "grant=<candidate term>", "stale=<term the voter last voted for>", or "log=<voter's last log index>", quoting the witness value after the '='.

### Answer

stale=1


### Prompt

In a Raft cluster, the candidates are [31, 34, 55, 75, 95]; candidate 31 runs at term 5 ; candidate 34 runs at term 5 ; candidate 55 runs at term 1 ; candidate 75 runs at term 4 ; candidate 95 runs at term 1. The current leader is 34 with candidate term 1 and last log entry (term 1, index 13). Voter 77 was last recorded voting for term 0 and has last log entry (term 1, index 1). Per the Raft log-matching safety rule, this voter may grant the vote to the leader exactly when the leader's candidate term is strictly greater than the term the voter last voted for (the request is not stale) and the leader's log is at least as up-to-date as the voter's: leader log term greater than the voter's, or equal log term with leader index at least the voter's index. Otherwise the voter refuses, either because the vote is stale or because the leader's log is not up-to-date.

Decide whether voter 77 may grant the vote to 34. Answer exactly one of three forms: "grant=<candidate term>", "stale=<term the voter last voted for>", or "log=<voter's last log index>", quoting the witness value after the '='.

### Answer

grant=1



## Level 5

### Prompt

In a Raft cluster, the candidates are [16, 17, 21, 26, 36, 54, 78, 97]; candidate 16 runs at term 6 ; candidate 17 runs at term 6 ; candidate 21 runs at term 3 ; candidate 26 runs at term 4 ; candidate 36 runs at term 8 ; candidate 54 runs at term 3 ; candidate 78 runs at term 8 ; candidate 97 runs at term 8. The current leader is 36 with candidate term 2 and last log entry (term 4, index 30). Voter 93 was last recorded voting for term 0 and has last log entry (term 4, index 4). Per the Raft log-matching safety rule, this voter may grant the vote to the leader exactly when the leader's candidate term is strictly greater than the term the voter last voted for (the request is not stale) and the leader's log is at least as up-to-date as the voter's: leader log term greater than the voter's, or equal log term with leader index at least the voter's index. Otherwise the voter refuses, either because the vote is stale or because the leader's log is not up-to-date.

Decide whether voter 93 may grant the vote to 36. Answer exactly one of three forms: "grant=<candidate term>", "stale=<term the voter last voted for>", or "log=<voter's last log index>", quoting the witness value after the '='.

### Answer

grant=2


### Prompt

In a Raft cluster, the candidates are [18, 29, 40, 45, 46, 59, 68, 94]; candidate 18 runs at term 6 ; candidate 29 runs at term 5 ; candidate 40 runs at term 6 ; candidate 45 runs at term 3 ; candidate 46 runs at term 7 ; candidate 59 runs at term 9 ; candidate 68 runs at term 3 ; candidate 94 runs at term 9. The current leader is 94 with candidate term 7 and last log entry (term 5, index 1). Voter 88 was last recorded voting for term 0 and has last log entry (term 5, index 26). Per the Raft log-matching safety rule, this voter may grant the vote to the leader exactly when the leader's candidate term is strictly greater than the term the voter last voted for (the request is not stale) and the leader's log is at least as up-to-date as the voter's: leader log term greater than the voter's, or equal log term with leader index at least the voter's index. Otherwise the voter refuses, either because the vote is stale or because the leader's log is not up-to-date.

Decide whether voter 88 may grant the vote to 94. Answer exactly one of three forms: "grant=<candidate term>", "stale=<term the voter last voted for>", or "log=<voter's last log index>", quoting the witness value after the '='.

### Answer

log=26

