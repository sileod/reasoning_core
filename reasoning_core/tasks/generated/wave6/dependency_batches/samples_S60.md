## Level 0

Prompt:
```
Consider these jobs: e, c, d, b, a.
Job e has no prerequisites and can start at any time.
Job c requires d before it can start.
Job d requires a before it can start.
Job b requires d before it can start.
Job a has no prerequisites and can start at any time.
Each job runs as early as its prerequisites allow, and any number of jobs run in parallel each round. Layer the jobs into rounds: in a single round put all jobs whose prerequisites are all finished by the start of that round, listing names in alphabetical order. Join the rounds with semicolons, joining the names in a round with commas, e.g. "a, c; b; d, e". The answer is this round listing.
```
Answer:
```
a, e; d; b, c
```

Prompt:
```
Consider these jobs: b, e, d, c, a.
Job b has no prerequisites and can start at any time.
Job e has no prerequisites and can start at any time.
Job d requires c, e before it can start.
Job c requires a before it can start.
Job a requires e before it can start.
Each job runs as early as its prerequisites allow, and any number of jobs run in parallel each round. Layer the jobs into rounds: in a single round put all jobs whose prerequisites are all finished by the start of that round, listing names in alphabetical order. Join the rounds with semicolons, joining the names in a round with commas, e.g. "a, c; b; d, e". The answer is this round listing.
```
Answer:
```
b, e; a; c; d
```

## Level 2

Prompt:
```
Consider these jobs: f, c, d, g, a, i, b, h, e.
Job f has no prerequisites and can start at any time.
Job c requires f before it can start.
Job d requires c, f, g, h before it can start.
Job g requires a, c before it can start.
Job a requires e, h before it can start.
Job i requires f, h before it can start.
Job b requires a, h before it can start.
Job h requires c before it can start.
Job e requires c, f, h, i before it can start.
Each job runs as early as its prerequisites allow, and any number of jobs run in parallel each round. Layer the jobs into rounds: in a single round put all jobs whose prerequisites are all finished by the start of that round, listing names in alphabetical order. Join the rounds with semicolons, joining the names in a round with commas, e.g. "a, c; b; d, e". The answer is this round listing.
```
Answer:
```
f; c; h; i; e; a; b, g; d
```

Prompt:
```
Consider these jobs: e, f, h, c, g, d, b, i, a.
Job e requires c, f, h before it can start.
Job f requires g before it can start.
Job h requires a, d before it can start.
Job c requires d, h, i before it can start.
Job g requires d before it can start.
Job d requires a before it can start.
Job b requires a, g before it can start.
Job i has no prerequisites and can start at any time.
Job a requires i before it can start.
Each job runs as early as its prerequisites allow, and any number of jobs run in parallel each round. Layer the jobs into rounds: in a single round put all jobs whose prerequisites are all finished by the start of that round, listing names in alphabetical order. Join the rounds with semicolons, joining the names in a round with commas, e.g. "a, c; b; d, e". The answer is this round listing.
```
Answer:
```
i; a; d; g, h; b, c, f; e
```

## Level 5

Prompt:
```
Consider these jobs: b, m, d, c, o, h, g, n, e, j, k, a, i, f, l.
Job b requires d before it can start.
Job m requires b, e, h before it can start.
Job d requires n before it can start.
Job c requires e, f, g, l before it can start.
Job o requires d, f, g, h, m, n before it can start.
Job h requires b, e, f, i before it can start.
Job g requires f, i, l, m before it can start.
Job n has no prerequisites and can start at any time.
Job e requires b, i, n before it can start.
Job j requires d, g, l, m before it can start.
Job k requires a, c, d before it can start.
Job a requires b, e, h, o before it can start.
Job i requires b before it can start.
Job f requires b, d, e, n before it can start.
Job l requires d, e, h before it can start.
Each job runs as early as its prerequisites allow, and any number of jobs run in parallel each round. Layer the jobs into rounds: in a single round put all jobs whose prerequisites are all finished by the start of that round, listing names in alphabetical order. Join the rounds with semicolons, joining the names in a round with commas, e.g. "a, c; b; d, e". The answer is this round listing.
```
Answer:
```
n; d; b; i; e; f; h; l, m; g; c, j, o; a; k
```

Prompt:
```
Consider these jobs: b, a, g, o, k, f, e, l, h, i, n, j, m, c, d.
Job b requires a, c, d before it can start.
Job a requires m before it can start.
Job g requires a, e, i, k, n before it can start.
Job o has no prerequisites and can start at any time.
Job k requires m, n before it can start.
Job f requires b, c, d, e, g, i, k, m before it can start.
Job e requires b, d before it can start.
Job l requires b, d, i, m, o before it can start.
Job h requires b, c, k, o before it can start.
Job i requires c, d, k, n before it can start.
Job n requires a before it can start.
Job j requires a, d, e, f, i, l, m before it can start.
Job m requires o before it can start.
Job c requires d, k, m, o before it can start.
Job d requires a, k, m, o before it can start.
Each job runs as early as its prerequisites allow, and any number of jobs run in parallel each round. Layer the jobs into rounds: in a single round put all jobs whose prerequisites are all finished by the start of that round, listing names in alphabetical order. Join the rounds with semicolons, joining the names in a round with commas, e.g. "a, c; b; d, e". The answer is this round listing.
```
Answer:
```
o; m; a; n; k; d; c; b, i; e, h, l; g; f; j
```
