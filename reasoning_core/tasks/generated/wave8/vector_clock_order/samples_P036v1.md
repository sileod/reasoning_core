## Level 0

Prompt:

```
A vector clock timestamp is a tuple of counters, one per process; process i's vec[i] is the number of events process i has sent before that point in time. Two timestamps u = (3, 1) and v = (2, 2) fix two moments. The order relation between u and v is determined componentwise: u is 'before' v when every u[i] <= v[i] and at least one is strict; u is 'after' v when every u[i] >= v[i] and at least one is strict; u 'equals' v when all components match; otherwise the two are 'concurrent' (one component is strictly smaller and another strictly greater). What is the order relation between u and v? The answer is one of the exact words: equal, before, after, or concurrent.
```

Answer:

```
concurrent
```

Prompt:

```
A vector clock timestamp is a tuple of counters, one per process; process i's vec[i] is the number of events process i has sent before that point in time. Two timestamps u = (1, 2) and v = (2, 2) fix two moments. The order relation between u and v is determined componentwise: u is 'before' v when every u[i] <= v[i] and at least one is strict; u is 'after' v when every u[i] >= v[i] and at least one is strict; u 'equals' v when all components match; otherwise the two are 'concurrent' (one component is strictly smaller and another strictly greater). What is the order relation between u and v? The answer is one of the exact words: equal, before, after, or concurrent.
```

Answer:

```
before
```

## Level 2

Prompt:

```
A vector clock timestamp is a tuple of counters, one per process; process i's vec[i] is the number of events process i has sent before that point in time. Two timestamps u = (4, 0, 1, 4) and v = (4, 0, 0, 4) fix two moments. The order relation between u and v is determined componentwise: u is 'before' v when every u[i] <= v[i] and at least one is strict; u is 'after' v when every u[i] >= v[i] and at least one is strict; u 'equals' v when all components match; otherwise the two are 'concurrent' (one component is strictly smaller and another strictly greater). What is the order relation between u and v? The answer is one of the exact words: equal, before, after, or concurrent.
```

Answer:

```
after
```

Prompt:

```
A vector clock timestamp is a tuple of counters, one per process; process i's vec[i] is the number of events process i has sent before that point in time. Two timestamps u = (1, 0, 3, 1) and v = (1, 0, 4, 1) fix two moments. The order relation between u and v is determined componentwise: u is 'before' v when every u[i] <= v[i] and at least one is strict; u is 'after' v when every u[i] >= v[i] and at least one is strict; u 'equals' v when all components match; otherwise the two are 'concurrent' (one component is strictly smaller and another strictly greater). What is the order relation between u and v? The answer is one of the exact words: equal, before, after, or concurrent.
```

Answer:

```
before
```

## Level 5

Prompt:

```
A vector clock timestamp is a tuple of counters, one per process; process i's vec[i] is the number of events process i has sent before that point in time. Two timestamps u = (5, 5, 2, 4, 2, 2, 5) and v = (5, 5, 2, 4, 3, 1, 5) fix two moments. The order relation between u and v is determined componentwise: u is 'before' v when every u[i] <= v[i] and at least one is strict; u is 'after' v when every u[i] >= v[i] and at least one is strict; u 'equals' v when all components match; otherwise the two are 'concurrent' (one component is strictly smaller and another strictly greater). What is the order relation between u and v? The answer is one of the exact words: equal, before, after, or concurrent.
```

Answer:

```
concurrent
```

Prompt:

```
A vector clock timestamp is a tuple of counters, one per process; process i's vec[i] is the number of events process i has sent before that point in time. Two timestamps u = (6, 0, 1, 7, 0, 5, 3) and v = (6, 0, 1, 7, 0, 5, 3) fix two moments. The order relation between u and v is determined componentwise: u is 'before' v when every u[i] <= v[i] and at least one is strict; u is 'after' v when every u[i] >= v[i] and at least one is strict; u 'equals' v when all components match; otherwise the two are 'concurrent' (one component is strictly smaller and another strictly greater). What is the order relation between u and v? The answer is one of the exact words: equal, before, after, or concurrent.
```

Answer:

```
equal
```
