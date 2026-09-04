## Level 0
### Prompt
A small discrete Bayesian network has binary nodes A, B, C. The conditional probability tables are:
A (parents -):
  P(A=0|-)=4/5, P(A=1|-)=1/5
B (parents A):
  P(B=0|A=0)=2/5, P(B=1|A=0)=3/5
  P(B=0|A=1)=3/5, P(B=1|A=1)=2/5
C (parents A):
  P(C=0|A=0)=2/5, P(C=1|A=0)=3/5
  P(C=0|A=1)=2/5, P(C=1|A=1)=3/5

We observe the evidence C=1.
What is the posterior probability P(B=1 | C=1)?
Give the answer as a reduced fraction of the form a/b (e.g. 3/7).

### Answer
14/25

### Prompt
A small discrete Bayesian network has binary nodes A, B, C. The conditional probability tables are:
A (parents -):
  P(A=0|-)=4/5, P(A=1|-)=1/5
B (parents A):
  P(B=0|A=0)=4/5, P(B=1|A=0)=1/5
  P(B=0|A=1)=1/5, P(B=1|A=1)=4/5
C (parents BA):
  P(C=0|B=0, A=0)=1/5, P(C=1|B=0, A=0)=4/5
  P(C=0|B=1, A=0)=4/5, P(C=1|B=1, A=0)=1/5
  P(C=0|B=0, A=1)=4/5, P(C=1|B=0, A=1)=1/5
  P(C=0|B=1, A=1)=1/5, P(C=1|B=1, A=1)=4/5

We observe the evidence B=1.
What is the posterior probability P(C=1 | B=1)?
Give the answer as a reduced fraction of the form a/b (e.g. 3/7).

### Answer
1/2

## Level 2
### Prompt
A small discrete Bayesian network has binary nodes A, B, C, D, E. The conditional probability tables are:
A (parents -):
  P(A=0|-)=2/13, P(A=1|-)=11/13
B (parents A):
  P(B=0|A=0)=9/13, P(B=1|A=0)=4/13
  P(B=0|A=1)=7/13, P(B=1|A=1)=6/13
C (parents A):
  P(C=0|A=0)=3/13, P(C=1|A=0)=10/13
  P(C=0|A=1)=5/13, P(C=1|A=1)=8/13
D (parents AB):
  P(D=0|A=0, B=0)=8/13, P(D=1|A=0, B=0)=5/13
  P(D=0|A=1, B=0)=12/13, P(D=1|A=1, B=0)=1/13
  P(D=0|A=0, B=1)=1/13, P(D=1|A=0, B=1)=12/13
  P(D=0|A=1, B=1)=12/13, P(D=1|A=1, B=1)=1/13
E (parents D):
  P(E=0|D=0)=6/13, P(E=1|D=0)=7/13
  P(E=0|D=1)=5/13, P(E=1|D=1)=8/13

We observe the evidence E=0.
What is the posterior probability P(C=1 | E=0)?
Give the answer as a reduced fraction of the form a/b (e.g. 3/7).

### Answer
106508/167089

### Prompt
A small discrete Bayesian network has binary nodes A, B, C, D, E. The conditional probability tables are:
A (parents -):
  P(A=0|-)=12/13, P(A=1|-)=1/13
B (parents A):
  P(B=0|A=0)=1/13, P(B=1|A=0)=12/13
  P(B=0|A=1)=3/13, P(B=1|A=1)=10/13
C (parents AB):
  P(C=0|A=0, B=0)=2/13, P(C=1|A=0, B=0)=11/13
  P(C=0|A=1, B=0)=8/13, P(C=1|A=1, B=0)=5/13
  P(C=0|A=0, B=1)=12/13, P(C=1|A=0, B=1)=1/13
  P(C=0|A=1, B=1)=9/13, P(C=1|A=1, B=1)=4/13
D (parents AC):
  P(D=0|A=0, C=0)=9/13, P(D=1|A=0, C=0)=4/13
  P(D=0|A=1, C=0)=12/13, P(D=1|A=1, C=0)=1/13
  P(D=0|A=0, C=1)=5/13, P(D=1|A=0, C=1)=8/13
  P(D=0|A=1, C=1)=2/13, P(D=1|A=1, C=1)=11/13
E (parents DC):
  P(E=0|D=0, C=0)=1/13, P(E=1|D=0, C=0)=12/13
  P(E=0|D=1, C=0)=8/13, P(E=1|D=1, C=0)=5/13
  P(E=0|D=0, C=1)=5/13, P(E=1|D=0, C=1)=8/13
  P(E=0|D=1, C=1)=12/13, P(E=1|D=1, C=1)=1/13

We observe the evidence A=1.
What is the posterior probability P(B=1 | A=1)?
Give the answer as a reduced fraction of the form a/b (e.g. 3/7).

### Answer
10/13

## Level 5
### Prompt
A small discrete Bayesian network has binary nodes A, B, C, D, E, F. The conditional probability tables are:
A (parents -):
  P(A=0|-)=2/25, P(A=1|-)=23/25
B (parents A):
  P(B=0|A=0)=15/25, P(B=1|A=0)=10/25
  P(B=0|A=1)=4/25, P(B=1|A=1)=21/25
C (parents AB):
  P(C=0|A=0, B=0)=11/25, P(C=1|A=0, B=0)=14/25
  P(C=0|A=1, B=0)=3/25, P(C=1|A=1, B=0)=22/25
  P(C=0|A=0, B=1)=6/25, P(C=1|A=0, B=1)=19/25
  P(C=0|A=1, B=1)=2/25, P(C=1|A=1, B=1)=23/25
D (parents CA):
  P(D=0|C=0, A=0)=2/25, P(D=1|C=0, A=0)=23/25
  P(D=0|C=1, A=0)=4/25, P(D=1|C=1, A=0)=21/25
  P(D=0|C=0, A=1)=12/25, P(D=1|C=0, A=1)=13/25
  P(D=0|C=1, A=1)=13/25, P(D=1|C=1, A=1)=12/25
E (parents D):
  P(E=0|D=0)=2/25, P(E=1|D=0)=23/25
  P(E=0|D=1)=13/25, P(E=1|D=1)=12/25
F (parents B):
  P(F=0|B=0)=2/25, P(F=1|B=0)=23/25
  P(F=0|B=1)=9/25, P(F=1|B=1)=16/25

We observe the evidence C=0.
What is the posterior probability P(F=1 | C=0)?
Give the answer as a reduced fraction of the form a/b (e.g. 3/7).

### Answer
5219/7050

### Prompt
A small discrete Bayesian network has binary nodes A, B, C, D, E, F. The conditional probability tables are:
A (parents -):
  P(A=0|-)=8/25, P(A=1|-)=17/25
B (parents A):
  P(B=0|A=0)=16/25, P(B=1|A=0)=9/25
  P(B=0|A=1)=12/25, P(B=1|A=1)=13/25
C (parents B):
  P(C=0|B=0)=23/25, P(C=1|B=0)=2/25
  P(C=0|B=1)=6/25, P(C=1|B=1)=19/25
D (parents A):
  P(D=0|A=0)=10/25, P(D=1|A=0)=15/25
  P(D=0|A=1)=1/25, P(D=1|A=1)=24/25
E (parents BA):
  P(E=0|B=0, A=0)=15/25, P(E=1|B=0, A=0)=10/25
  P(E=0|B=1, A=0)=7/25, P(E=1|B=1, A=0)=18/25
  P(E=0|B=0, A=1)=6/25, P(E=1|B=0, A=1)=19/25
  P(E=0|B=1, A=1)=10/25, P(E=1|B=1, A=1)=15/25
F (parents B):
  P(F=0|B=0)=7/25, P(F=1|B=0)=18/25
  P(F=0|B=1)=11/25, P(F=1|B=1)=14/25

We observe the evidence A=1.
What is the posterior probability P(E=1 | A=1)?
Give the answer as a reduced fraction of the form a/b (e.g. 3/7).

### Answer
423/625
