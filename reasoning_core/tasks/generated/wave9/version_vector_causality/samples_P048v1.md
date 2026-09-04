# Level 0

## Example 1

**Prompt:**

```
There are 4 processes P0..P3, each tracking a vector clock of size 4.

P3 executes a local operation (event E0).
P0 sends a message (event E1).
P0 executes a local operation (event E2).
P3 executes a local operation (event E3).
P0 executes a local operation (event E4).
P3 receives a message (event E5).
P2 executes a local operation (event E6).
P3 executes a local operation (event E7).

Using Lamport-style vector clocks, state the vector clock of event E7 at the moment it executes. Give the value as a comma-separated list of integers, one per process, in order P0, P1, ... .

The answer is exactly that comma-separated list.
```

**Answer:**

```
1,0,0,4
```

## Example 2

**Prompt:**

```
There are 4 processes P0..P3, each tracking a vector clock of size 4.

P0 executes a local operation (event E0).
P0 executes a local operation (event E1).
P1 executes a local operation (event E2).
P3 sends a message (event E3).
P2 sends a message (event E4).
P1 receives a message (event E5).
P1 executes a local operation (event E6).
P2 receives a message (event E7).

Using Lamport-style vector clocks, state the vector clock of event E6 at the moment it executes. Give the value as a comma-separated list of integers, one per process, in order P0, P1, ... .

The answer is exactly that comma-separated list.
```

**Answer:**

```
0,3,1,0
```



# Level 2

## Example 1

**Prompt:**

```
There are 6 processes P0..P5, each tracking a vector clock of size 6.

P4 executes a local operation (event E0).
P4 executes a local operation (event E1).
P4 executes a local operation (event E2).
P1 executes a local operation (event E3).
P5 sends a message (event E4).
P1 executes a local operation (event E5).
P0 executes a local operation (event E6).
P1 sends a message (event E7).
P3 receives a message (event E8).
P5 executes a local operation (event E9).
P5 executes a local operation (event E10).
P4 receives a message (event E11).

Using Lamport-style vector clocks, state the vector clock of event E3 at the moment it executes. Give the value as a comma-separated list of integers, one per process, in order P0, P1, ... .

The answer is exactly that comma-separated list.
```

**Answer:**

```
0,1,0,0,0,0
```

## Example 2

**Prompt:**

```
There are 6 processes P0..P5, each tracking a vector clock of size 6.

P2 executes a local operation (event E0).
P1 sends a message (event E1).
P2 executes a local operation (event E2).
P3 executes a local operation (event E3).
P1 executes a local operation (event E4).
P5 sends a message (event E5).
P5 executes a local operation (event E6).
P1 executes a local operation (event E7).
P0 executes a local operation (event E8).
P1 executes a local operation (event E9).
P1 sends a message (event E10).
P2 receives a message (event E11).

Using Lamport-style vector clocks, state the vector clock of event E7 at the moment it executes. Give the value as a comma-separated list of integers, one per process, in order P0, P1, ... .

The answer is exactly that comma-separated list.
```

**Answer:**

```
0,3,0,0,0,0
```



# Level 5

## Example 1

**Prompt:**

```
There are 9 processes P0..P8, each tracking a vector clock of size 9.

P1 executes a local operation (event E0).
P4 sends a message (event E1).
P4 executes a local operation (event E2).
P7 sends a message (event E3).
P7 executes a local operation (event E4).
P1 executes a local operation (event E5).
P7 executes a local operation (event E6).
P1 executes a local operation (event E7).
P5 executes a local operation (event E8).
P6 executes a local operation (event E9).
P3 sends a message (event E10).
P8 executes a local operation (event E11).
P4 sends a message (event E12).
P6 executes a local operation (event E13).
P6 sends a message (event E14).
P2 receives a message (event E15).
P1 receives a message (event E16).
P4 receives a message (event E17).

Using Lamport-style vector clocks, state the vector clock of event E16 at the moment it executes. Give the value as a comma-separated list of integers, one per process, in order P0, P1, ... .

The answer is exactly that comma-separated list.
```

**Answer:**

```
0,4,0,1,0,0,0,0,0
```

## Example 2

**Prompt:**

```
There are 9 processes P0..P8, each tracking a vector clock of size 9.

P1 sends a message (event E0).
P4 executes a local operation (event E1).
P0 executes a local operation (event E2).
P8 executes a local operation (event E3).
P1 executes a local operation (event E4).
P7 executes a local operation (event E5).
P2 sends a message (event E6).
P4 executes a local operation (event E7).
P1 executes a local operation (event E8).
P8 sends a message (event E9).
P6 sends a message (event E10).
P2 executes a local operation (event E11).
P6 executes a local operation (event E12).
P0 receives a message (event E13).
P8 receives a message (event E14).
P0 executes a local operation (event E15).
P3 receives a message (event E16).
P6 executes a local operation (event E17).

Using Lamport-style vector clocks, state the vector clock of event E15 at the moment it executes. Give the value as a comma-separated list of integers, one per process, in order P0, P1, ... .

The answer is exactly that comma-separated list.
```

**Answer:**

```
3,0,1,0,0,0,0,0,0
```


