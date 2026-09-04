## Level 0

### Example

**Prompt:**

N Replicas:
5

Ops:
['write(0,2,4) = 2', 'write(0,4) = 4', 'write(0,1,3) = 2', 'read(0,1,2,3,4)', 'read(0,1,3,4)', 'read(0,1,2,3,4)']

Queried Read:
5
There are 5 replicas, each storing a value with a version. Values are non-negative integers; a higher version is newer. Initially every replica is unset (holds no value).
Operations happen in order. A write(V) = x writes value x at a new version (each write's version is one higher than the previous write) to every replica in the listed set. A read(S) samples the replicas in set S and returns the value of the newest version among them; if several replicas hold that same newest version they all agree on its value by construction.
Schedule:
0: write(0,2,4) = 2
1: write(0,4) = 4
2: write(0,1,3) = 2
3: read(0,1,2,3,4)
4: read(0,1,3,4)
5: read(0,1,2,3,4)

Question: list the values visible to every read operation, then give the value seen by the read at index 5 as the answer.
The answer is a single non-negative integer.

**Answer:**

2

### Example

**Prompt:**

N Replicas:
5

Ops:
['write(0,4) = 2', 'write(1,2,3,4) = 0', 'write(0,1,2,3,4) = 4', 'read(1,3,4)', 'read(0,1,2,3,4)', 'read(2,4)']

Queried Read:
3
There are 5 replicas, each storing a value with a version. Values are non-negative integers; a higher version is newer. Initially every replica is unset (holds no value).
Operations happen in order. A write(V) = x writes value x at a new version (each write's version is one higher than the previous write) to every replica in the listed set. A read(S) samples the replicas in set S and returns the value of the newest version among them; if several replicas hold that same newest version they all agree on its value by construction.
Schedule:
0: write(0,4) = 2
1: write(1,2,3,4) = 0
2: write(0,1,2,3,4) = 4
3: read(1,3,4)
4: read(0,1,2,3,4)
5: read(2,4)

Question: list the values visible to every read operation, then give the value seen by the read at index 3 as the answer.
The answer is a single non-negative integer.

**Answer:**

4

## Level 2

### Example

**Prompt:**

N Replicas:
7

Ops:
['write(0,1,2,4,5,6) = 5', 'write(1,2,3,4,5,6) = 4', 'write(0,1,3,5,6) = 1', 'write(0,1,4,5) = 4', 'write(0,1,2,3,4,6) = 1', 'write(0,1,2,3,4,5,6) = 3', 'write(0,3,4,5,6) = 1', 'write(0,2,3,4,5,6) = 2', 'write(0,1,2,3,4,5,6) = 2', 'read(1,2,3,4,5,6)']

Queried Read:
9
There are 7 replicas, each storing a value with a version. Values are non-negative integers; a higher version is newer. Initially every replica is unset (holds no value).
Operations happen in order. A write(V) = x writes value x at a new version (each write's version is one higher than the previous write) to every replica in the listed set. A read(S) samples the replicas in set S and returns the value of the newest version among them; if several replicas hold that same newest version they all agree on its value by construction.
Schedule:
0: write(0,1,2,4,5,6) = 5
1: write(1,2,3,4,5,6) = 4
2: write(0,1,3,5,6) = 1
3: write(0,1,4,5) = 4
4: write(0,1,2,3,4,6) = 1
5: write(0,1,2,3,4,5,6) = 3
6: write(0,3,4,5,6) = 1
7: write(0,2,3,4,5,6) = 2
8: write(0,1,2,3,4,5,6) = 2
9: read(1,2,3,4,5,6)

Question: list the values visible to every read operation, then give the value seen by the read at index 9 as the answer.
The answer is a single non-negative integer.

**Answer:**

2

### Example

**Prompt:**

N Replicas:
7

Ops:
['write(0,1,2,3,4) = 6', 'write(0,1,2,3,4,5,6) = 1', 'write(0,1,2,3,4,5,6) = 2', 'write(0,1,2,3,4,5) = 3', 'write(0,1,3,6) = 5', 'write(0,1,3,6) = 6', 'read(0,1,2,3,4,5,6)', 'read(0,1,2,3,4,5,6)', 'read(2,4,5,6)', 'read(2,3,4,5)']

Queried Read:
9
There are 7 replicas, each storing a value with a version. Values are non-negative integers; a higher version is newer. Initially every replica is unset (holds no value).
Operations happen in order. A write(V) = x writes value x at a new version (each write's version is one higher than the previous write) to every replica in the listed set. A read(S) samples the replicas in set S and returns the value of the newest version among them; if several replicas hold that same newest version they all agree on its value by construction.
Schedule:
0: write(0,1,2,3,4) = 6
1: write(0,1,2,3,4,5,6) = 1
2: write(0,1,2,3,4,5,6) = 2
3: write(0,1,2,3,4,5) = 3
4: write(0,1,3,6) = 5
5: write(0,1,3,6) = 6
6: read(0,1,2,3,4,5,6)
7: read(0,1,2,3,4,5,6)
8: read(2,4,5,6)
9: read(2,3,4,5)

Question: list the values visible to every read operation, then give the value seen by the read at index 9 as the answer.
The answer is a single non-negative integer.

**Answer:**

6

## Level 5

### Example

**Prompt:**

N Replicas:
10

Ops:
['write(0,1,2,4,5,6,8,9) = 2', 'write(0,1,2,4,5,6,8,9) = 2', 'write(0,1,2,3,4,5,6,7,8,9) = 8', 'write(0,1,2,3,4,5,6,7,8) = 3', 'write(0,1,2,3,5,6,7,9) = 4', 'write(0,2,4,5,6,7,8,9) = 3', 'write(0,1,2,3,4,5,6,7,8,9) = 8', 'write(0,1,2,3,4,5,6,7,9) = 9', 'write(0,1,2,3,4,5,6,7,8,9) = 2', 'write(0,1,2,3,4,5,6,7,8,9) = 5', 'write(0,1,2,3,4,7,8,9) = 0', 'read(0,1,2,3,4,5,6,7,8,9)', 'read(0,1,2,3,4,5,6,7,8,9)', 'read(0,1,2,3,4,5,6,7,8,9)', 'read(0,1,2,3,4,5,6,7,8)', 'read(0,1,2,3,5,6,7,8,9)', 'read(0,2,4,5,6,7,8,9)', 'read(0,1,3,4,5,6,7,8,9)', 'read(1,2,3,4,5,6,7,8,9)', 'read(0,1,2,3,4,5,6,7,8)']

Queried Read:
16
There are 10 replicas, each storing a value with a version. Values are non-negative integers; a higher version is newer. Initially every replica is unset (holds no value).
Operations happen in order. A write(V) = x writes value x at a new version (each write's version is one higher than the previous write) to every replica in the listed set. A read(S) samples the replicas in set S and returns the value of the newest version among them; if several replicas hold that same newest version they all agree on its value by construction.
Schedule:
0: write(0,1,2,4,5,6,8,9) = 2
1: write(0,1,2,4,5,6,8,9) = 2
2: write(0,1,2,3,4,5,6,7,8,9) = 8
3: write(0,1,2,3,4,5,6,7,8) = 3
4: write(0,1,2,3,5,6,7,9) = 4
5: write(0,2,4,5,6,7,8,9) = 3
6: write(0,1,2,3,4,5,6,7,8,9) = 8
7: write(0,1,2,3,4,5,6,7,9) = 9
8: write(0,1,2,3,4,5,6,7,8,9) = 2
9: write(0,1,2,3,4,5,6,7,8,9) = 5
10: write(0,1,2,3,4,7,8,9) = 0
11: read(0,1,2,3,4,5,6,7,8,9)
12: read(0,1,2,3,4,5,6,7,8,9)
13: read(0,1,2,3,4,5,6,7,8,9)
14: read(0,1,2,3,4,5,6,7,8)
15: read(0,1,2,3,5,6,7,8,9)
16: read(0,2,4,5,6,7,8,9)
17: read(0,1,3,4,5,6,7,8,9)
18: read(1,2,3,4,5,6,7,8,9)
19: read(0,1,2,3,4,5,6,7,8)

Question: list the values visible to every read operation, then give the value seen by the read at index 16 as the answer.
The answer is a single non-negative integer.

**Answer:**

0

### Example

**Prompt:**

N Replicas:
10

Ops:
['write(0,1,2,3,4,5,6,7,8,9) = 7', 'write(0,1,2,3,5,6,7,8,9) = 1', 'write(0,1,2,3,4,5,6,7,9) = 1', 'write(0,1,2,4,5,6,7,8,9) = 7', 'write(0,1,3,4,5,6,8,9) = 11', 'write(0,1,2,3,4,5,6,7,8,9) = 4', 'write(0,1,2,3,4,5,6,7,8) = 4', 'write(0,1,2,3,4,5,6,7,8,9) = 0', 'write(0,1,2,3,4,5,6,7,9) = 7', 'write(1,2,3,4,5,6,7,8,9) = 10', 'write(0,1,2,3,5,6,7,8) = 8', 'read(0,1,2,3,4,5,6,7,8,9)', 'read(0,2,3,4,5,6,7,8,9)', 'read(0,1,2,4,5,6,7,9)', 'read(0,1,2,4,5,6,8,9)', 'read(0,1,2,3,5,7,8,9)', 'read(0,1,2,3,4,5,7,8)', 'read(0,1,2,3,4,5,6,7,8,9)', 'read(0,2,3,4,5,6,7,8,9)', 'read(0,1,2,3,4,5,6,7,8,9)']

Queried Read:
14
There are 10 replicas, each storing a value with a version. Values are non-negative integers; a higher version is newer. Initially every replica is unset (holds no value).
Operations happen in order. A write(V) = x writes value x at a new version (each write's version is one higher than the previous write) to every replica in the listed set. A read(S) samples the replicas in set S and returns the value of the newest version among them; if several replicas hold that same newest version they all agree on its value by construction.
Schedule:
0: write(0,1,2,3,4,5,6,7,8,9) = 7
1: write(0,1,2,3,5,6,7,8,9) = 1
2: write(0,1,2,3,4,5,6,7,9) = 1
3: write(0,1,2,4,5,6,7,8,9) = 7
4: write(0,1,3,4,5,6,8,9) = 11
5: write(0,1,2,3,4,5,6,7,8,9) = 4
6: write(0,1,2,3,4,5,6,7,8) = 4
7: write(0,1,2,3,4,5,6,7,8,9) = 0
8: write(0,1,2,3,4,5,6,7,9) = 7
9: write(1,2,3,4,5,6,7,8,9) = 10
10: write(0,1,2,3,5,6,7,8) = 8
11: read(0,1,2,3,4,5,6,7,8,9)
12: read(0,2,3,4,5,6,7,8,9)
13: read(0,1,2,4,5,6,7,9)
14: read(0,1,2,4,5,6,8,9)
15: read(0,1,2,3,5,7,8,9)
16: read(0,1,2,3,4,5,7,8)
17: read(0,1,2,3,4,5,6,7,8,9)
18: read(0,2,3,4,5,6,7,8,9)
19: read(0,1,2,3,4,5,6,7,8,9)

Question: list the values visible to every read operation, then give the value seen by the read at index 14 as the answer.
The answer is a single non-negative integer.

**Answer:**

8

