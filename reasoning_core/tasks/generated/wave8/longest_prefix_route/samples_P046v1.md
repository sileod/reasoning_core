# Samples for longest_prefix_route (P046v1)

## Level 0

### Example 1

Prompt:

```
Destination:
0101101100

Routes:
[{'length': 0, 'prefix': '(default)', 'hop': 686}, {'length': 2, 'prefix': '00', 'hop': 686}, {'length': 6, 'prefix': '100110', 'hop': 158}, {'length': 8, 'prefix': '01100010', 'hop': 86}, {'length': 10, 'prefix': '0010011001', 'hop': 686}]

A route (length, prefix, hop) forwards a destination to hop whenever the first 'length' bits of the destination bitstring equal 'prefix'. A route with length 0 has prefix "(default)" and matches every destination. Apply longest-prefix matching: among all matching routes, the one with the greatest length wins; if multiple have the same greatest length the tie is not reached because prefixes of one length are distinct. The answer is the winning hop, as an integer.
```

Answer:

```
686
```

### Example 2

Prompt:

```
Destination:
1000111111

Routes:
[{'length': 0, 'prefix': '(default)', 'hop': 386}, {'length': 1, 'prefix': '1', 'hop': 840}, {'length': 8, 'prefix': '01110000', 'hop': 54}, {'length': 10, 'prefix': '0110111101', 'hop': 54}]

A route (length, prefix, hop) forwards a destination to hop whenever the first 'length' bits of the destination bitstring equal 'prefix'. A route with length 0 has prefix "(default)" and matches every destination. Apply longest-prefix matching: among all matching routes, the one with the greatest length wins; if multiple have the same greatest length the tie is not reached because prefixes of one length are distinct. The answer is the winning hop, as an integer.
```

Answer:

```
840
```

## Level 2

### Example 1

Prompt:

```
Destination:
111011100011

Routes:
[{'length': 0, 'prefix': '(default)', 'hop': 114}, {'length': 4, 'prefix': '0110', 'hop': 175}, {'length': 5, 'prefix': '10011', 'hop': 203}, {'length': 6, 'prefix': '100011', 'hop': 959}, {'length': 6, 'prefix': '100110', 'hop': 494}, {'length': 6, 'prefix': '110001', 'hop': 114}, {'length': 8, 'prefix': '00011011', 'hop': 494}, {'length': 10, 'prefix': '0010101111', 'hop': 494}, {'length': 10, 'prefix': '1011100001', 'hop': 203}]

A route (length, prefix, hop) forwards a destination to hop whenever the first 'length' bits of the destination bitstring equal 'prefix'. A route with length 0 has prefix "(default)" and matches every destination. Apply longest-prefix matching: among all matching routes, the one with the greatest length wins; if multiple have the same greatest length the tie is not reached because prefixes of one length are distinct. The answer is the winning hop, as an integer.
```

Answer:

```
114
```

### Example 2

Prompt:

```
Destination:
011011111101

Routes:
[{'length': 0, 'prefix': '(default)', 'hop': 543}, {'length': 2, 'prefix': '01', 'hop': 173}, {'length': 3, 'prefix': '001', 'hop': 15}, {'length': 3, 'prefix': '011', 'hop': 105}, {'length': 5, 'prefix': '10111', 'hop': 473}, {'length': 8, 'prefix': '00100101', 'hop': 473}, {'length': 8, 'prefix': '10110100', 'hop': 367}, {'length': 9, 'prefix': '110000111', 'hop': 978}, {'length': 12, 'prefix': '000100001000', 'hop': 277}]

A route (length, prefix, hop) forwards a destination to hop whenever the first 'length' bits of the destination bitstring equal 'prefix'. A route with length 0 has prefix "(default)" and matches every destination. Apply longest-prefix matching: among all matching routes, the one with the greatest length wins; if multiple have the same greatest length the tie is not reached because prefixes of one length are distinct. The answer is the winning hop, as an integer.
```

Answer:

```
105
```

## Level 5

### Example 1

Prompt:

```
Destination:
100000010111010

Routes:
[{'length': 0, 'prefix': '(default)', 'hop': 219}, {'length': 2, 'prefix': '11', 'hop': 738}, {'length': 3, 'prefix': '011', 'hop': 463}, {'length': 3, 'prefix': '110', 'hop': 596}, {'length': 4, 'prefix': '0100', 'hop': 219}, {'length': 5, 'prefix': '10000', 'hop': 313}, {'length': 6, 'prefix': '001100', 'hop': 73}, {'length': 6, 'prefix': '010110', 'hop': 248}, {'length': 7, 'prefix': '1000100', 'hop': 28}, {'length': 7, 'prefix': '1011101', 'hop': 607}, {'length': 11, 'prefix': '00011010101', 'hop': 28}, {'length': 14, 'prefix': '10101111000011', 'hop': 313}, {'length': 15, 'prefix': '100011110010100', 'hop': 738}]

A route (length, prefix, hop) forwards a destination to hop whenever the first 'length' bits of the destination bitstring equal 'prefix'. A route with length 0 has prefix "(default)" and matches every destination. Apply longest-prefix matching: among all matching routes, the one with the greatest length wins; if multiple have the same greatest length the tie is not reached because prefixes of one length are distinct. The answer is the winning hop, as an integer.
```

Answer:

```
313
```

### Example 2

Prompt:

```
Destination:
100111001101011

Routes:
[{'length': 0, 'prefix': '(default)', 'hop': 220}, {'length': 3, 'prefix': '011', 'hop': 747}, {'length': 4, 'prefix': '1111', 'hop': 940}, {'length': 7, 'prefix': '0001000', 'hop': 141}, {'length': 9, 'prefix': '001000010', 'hop': 328}, {'length': 9, 'prefix': '010100011', 'hop': 328}, {'length': 9, 'prefix': '100110111', 'hop': 747}, {'length': 9, 'prefix': '101110110', 'hop': 510}, {'length': 10, 'prefix': '1001100100', 'hop': 468}, {'length': 10, 'prefix': '1011010101', 'hop': 467}, {'length': 12, 'prefix': '010000011010', 'hop': 940}, {'length': 14, 'prefix': '10001000010000', 'hop': 426}]

A route (length, prefix, hop) forwards a destination to hop whenever the first 'length' bits of the destination bitstring equal 'prefix'. A route with length 0 has prefix "(default)" and matches every destination. Apply longest-prefix matching: among all matching routes, the one with the greatest length wins; if multiple have the same greatest length the tie is not reached because prefixes of one length are distinct. The answer is the winning hop, as an integer.
```

Answer:

```
220
```

