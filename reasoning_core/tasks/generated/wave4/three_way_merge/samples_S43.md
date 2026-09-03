# Samples for s43_three_way_merge

## Level 0

### Prompt

```
Base:
1: line 1
2: line 2
3: line 3
4: line 4
5: line 5
6: line 6
7: line 7
8: line 8

Version A:
1: A0
2: line 2
3: line 3
4: line 4
5: line 5
6: line 6
7: line 7

Version B:
1: line 2
2: AB2
3: line 3
4: line 4
5: line 5
6: line 6
7: line 7
8: line 8

An edit is a changed, inserted or deleted line. Two edits conflict when they touch the same base line. A clean merge applies both edits to the base. If the two versions conflict, answer with 'conflict <N>', where N is the number of the first base line at which they conflict. If they merge cleanly, answer with the full merged file as numbered lines, one per line, in order of line number.
```

**Answer**

```
conflict 1
```

### Prompt

```
Base:
1: line 1
2: line 2
3: line 3
4: line 4
5: line 5
6: line 6
7: line 7
8: line 8

Version A:
1: line 1
2: line 2
3: A2
4: line 4
5: line 5
6: line 6
7: line 8

Version B:
1: line 1
2: AB1
3: line 2
4: line 3
5: line 5
6: line 6
7: line 7
8: line 8

An edit is a changed, inserted or deleted line. Two edits conflict when they touch the same base line. A clean merge applies both edits to the base. If the two versions conflict, answer with 'conflict <N>', where N is the number of the first base line at which they conflict. If they merge cleanly, answer with the full merged file as numbered lines, one per line, in order of line number.
```

**Answer**

```
1: line 1
2: AB1
3: line 2
4: A2
5: line 5
6: line 6
7: line 8
```

## Level 2

### Prompt

```
Base:
1: line 1
2: line 2
3: line 3
4: line 4
5: line 5
6: line 6
7: line 7
8: line 8
9: line 9
10: line 10
11: line 11
12: line 12

Version A:
1: line 1
2: line 2
3: line 3
4: line 4
5: line 5
6: line 6
7: AB6
8: line 7
9: line 9
10: AB9
11: line 10
12: line 11
13: line 12

Version B:
1: line 1
2: A1
3: line 3
4: line 4
5: line 5
6: line 6
7: AB6
8: line 7
9: line 8
10: AB8
11: line 9
12: line 10
13: line 11
14: line 12

An edit is a changed, inserted or deleted line. Two edits conflict when they touch the same base line. A clean merge applies both edits to the base. If the two versions conflict, answer with 'conflict <N>', where N is the number of the first base line at which they conflict. If they merge cleanly, answer with the full merged file as numbered lines, one per line, in order of line number.
```

**Answer**

```
conflict 7
```

### Prompt

```
Base:
1: line 1
2: line 2
3: line 3
4: line 4
5: line 5
6: line 6
7: line 7
8: line 8
9: line 9
10: line 10
11: line 11
12: line 12

Version A:
1: line 1
2: AB1
3: line 2
4: line 3
5: line 4
6: line 5
7: line 6
8: AB6
9: line 7
10: line 9
11: line 10
12: line 11
13: line 12

Version B:
1: line 1
2: line 2
3: line 3
4: line 5
5: line 6
6: AB6
7: line 7
8: AB7
9: line 8
10: line 9
11: line 10
12: line 11
13: line 12

An edit is a changed, inserted or deleted line. Two edits conflict when they touch the same base line. A clean merge applies both edits to the base. If the two versions conflict, answer with 'conflict <N>', where N is the number of the first base line at which they conflict. If they merge cleanly, answer with the full merged file as numbered lines, one per line, in order of line number.
```

**Answer**

```
conflict 7
```

## Level 5

### Prompt

```
Base:
1: line 1
2: line 2
3: line 3
4: line 4
5: line 5
6: line 6
7: line 7
8: line 8
9: line 9
10: line 10
11: line 11
12: line 12
13: line 13
14: line 14
15: line 15
16: line 16
17: line 17
18: line 18

Version A:
1: line 1
2: A2
3: line 4
4: line 5
5: line 6
6: A6
7: line 8
8: line 9
9: AB9
10: line 10
11: line 11
12: line 12
13: line 13
14: line 14
15: line 15
16: line 16
17: line 17
18: line 18

Version B:
1: line 1
2: A1
3: AB2
4: line 3
5: line 4
6: line 5
7: line 6
8: line 7
9: line 10
10: line 11
11: line 12
12: line 13
13: line 14
14: line 15
15: line 16
16: line 17
17: line 18

An edit is a changed, inserted or deleted line. Two edits conflict when they touch the same base line. A clean merge applies both edits to the base. If the two versions conflict, answer with 'conflict <N>', where N is the number of the first base line at which they conflict. If they merge cleanly, answer with the full merged file as numbered lines, one per line, in order of line number.
```

**Answer**

```
conflict 2
```

### Prompt

```
Base:
1: line 1
2: line 2
3: line 3
4: line 4
5: line 5
6: line 6
7: line 7
8: line 8
9: line 9
10: line 10
11: line 11
12: line 12
13: line 13
14: line 14
15: line 15
16: line 16
17: line 17
18: line 18

Version A:
1: line 1
2: line 2
3: line 3
4: A3
5: line 5
6: line 6
7: line 7
8: line 8
9: line 9
10: A9
11: line 11
12: line 12
13: AB12
14: line 13
15: line 14
16: line 15
17: line 16
18: line 17

Version B:
1: line 1
2: line 2
3: line 3
4: AB3
5: line 4
6: line 5
7: line 6
8: line 7
9: line 8
10: AB8
11: line 9
12: line 10
13: line 11
14: A11
15: line 13
16: line 14
17: line 15
18: line 16
19: line 17
20: AB17
21: line 18

An edit is a changed, inserted or deleted line. Two edits conflict when they touch the same base line. A clean merge applies both edits to the base. If the two versions conflict, answer with 'conflict <N>', where N is the number of the first base line at which they conflict. If they merge cleanly, answer with the full merged file as numbered lines, one per line, in order of line number.
```

**Answer**

```
conflict 4
```
