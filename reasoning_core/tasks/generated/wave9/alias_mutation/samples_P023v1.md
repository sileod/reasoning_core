## Level 0

### Example 1

Prompt:

Variables begin as:
a = [7]
b = [0, 9, 4, 9]
c = [9, 1, 3]
d = [9, 8, 8, 8]

Perform these operations in order:
1. c = a
2. c[0] = 3
3. c = [1]
4. d = c
5. c.append(7)
6. a = [2, 4, 8]

After all operations, what is the final value of variable a?

The answer is a list literal, e.g. [1, 2, 3].

Answer:

```
[2, 4, 8]
```

### Example 2

Prompt:

Variables begin as:
a = [1, 5]
b = [7]
c = [8, 5]
d = [4, 8]

Perform these operations in order:
1. b = [6]
2. a[1] = 9
3. d = a
4. b[0] = 1
5. c = d
6. b[0] = 3

After all operations, what is the final value of variable a?

The answer is a list literal, e.g. [1, 2, 3].

Answer:

```
[1, 9]
```

## Level 2

### Example 1

Prompt:

Variables begin as:
a = [3, 1, 0]
b = [1, 12]
c = [3, 11, 9, 5, 0]
d = [13]
e = [6, 9, 11]
f = [8, 11, 6, 3, 1]

Perform these operations in order:
1. a = [1, 9, 11, 1]
2. a = e
3. b = e
4. a = [2, 5, 11, 6]
5. d[0] = 11
6. d[0] = 3
7. c[2] = 9
8. b = e
9. d = [13, 0, 9, 1, 9]
10. a.append(11)

After all operations, what is the final value of variable a?

The answer is a list literal, e.g. [1, 2, 3].

Answer:

```
[2, 5, 11, 6, 11]
```

### Example 2

Prompt:

Variables begin as:
a = [6, 10]
b = [12, 8]
c = [10, 2]
d = [1, 9, 3]
e = [11, 1, 12, 13, 6, 5]
f = [9, 11, 9, 7, 1, 3]

Perform these operations in order:
1. f = e
2. e = b
3. c[0] = 6
4. c[1] = 8
5. c[1] = 7
6. a.append(11)
7. a.append(7)
8. c = [3, 3, 3, 2, 12]
9. a = [7, 7]
10. b = f

After all operations, what is the final value of variable a?

The answer is a list literal, e.g. [1, 2, 3].

Answer:

```
[7, 7]
```

## Level 5

### Example 1

Prompt:

Variables begin as:
a = [11]
b = [18, 15, 10, 13]
c = [4, 19, 13, 4]
d = [1, 16, 15, 18]
e = [17, 14, 16, 0, 11, 12, 2]
f = [12]
g = [18]
h = [5, 2, 11, 4, 18, 0, 6]
i = [16, 17, 10, 0, 12, 18, 17]

Perform these operations in order:
1. a.append(13)
2. c.append(6)
3. a = d
4. i.append(0)
5. g = [11, 1]
6. f = [5, 1, 0, 19, 6]
7. a[2] = 2
8. i.append(13)
9. d.append(7)
10. f = [2, 17, 8, 4]
11. a = g
12. a = [10, 9, 5, 5, 19, 6, 14]
13. c = [14, 13, 8, 9, 14, 1, 13, 18]
14. g[1] = 8
15. g = [10, 0]
16. c[7] = 7

After all operations, what is the final value of variable a?

The answer is a list literal, e.g. [1, 2, 3].

Answer:

```
[10, 9, 5, 5, 19, 6, 14]
```

### Example 2

Prompt:

Variables begin as:
a = [13, 10, 15, 1, 16]
b = [19]
c = [16, 15, 16, 0, 14, 12, 10, 5, 10]
d = [1, 15, 1, 11, 2, 15, 10, 6, 16]
e = [5, 2, 9, 11, 2, 13]
f = [10, 2, 12]
g = [11, 4, 1, 16, 16, 15]
h = [19, 14, 1, 16, 3, 7]
i = [11, 2, 18, 3, 12, 0, 12]

Perform these operations in order:
1. g[2] = 12
2. e = [1, 7, 13, 9, 5, 17]
3. d = [16, 5, 4, 11]
4. g[1] = 14
5. i[6] = 6
6. h.append(4)
7. i = c
8. a[3] = 13
9. d = a
10. i = f
11. d[2] = 12
12. c[5] = 17
13. i.append(18)
14. d = a
15. f.append(3)
16. i = [7, 19, 3, 5, 5, 3]

After all operations, what is the final value of variable a?

The answer is a list literal, e.g. [1, 2, 3].

Answer:

```
[13, 10, 12, 13, 16]
```
