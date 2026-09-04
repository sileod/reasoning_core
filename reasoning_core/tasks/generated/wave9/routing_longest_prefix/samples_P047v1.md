# samples_P047v1 -- routing_longest_prefix

Each entry shows the exact prompt the task emits and the gold next-hop answer underneath.

## Level 0

**Prompt:**
```
A router picks the next hop for a destination address from its routing table.
Every destination and prefix below is a 9-bit binary string.

Routing table (prefix, length -> next hop (metric)):
  0010, /4 -> R5 (metric 13)
  00101111, /8 -> R4 (metric 10)
  001011, /6 -> R6 (metric 3)
  10101111, /8 -> R1 (metric 2)
  1010111, /7 -> R1 (metric 2)
  10101, /5 -> R7 (metric 11)
  1010, /4 -> R7 (metric 7)
  101011110, /9 -> R3 (metric 2)

Destination to route: 001011110

Selection rules, applied in order:
  1. An entry matches if its /P prefix equals the destination's first P bits.
  2. Among matching entries, keep the longest prefix (largest P).
  3. Ties on prefix length are broken by the lowest metric.
  4. Remaining ties on length and metric are broken by the lexicographically smallest next hop.

The answer is the selected next hop, written exactly as a hop name like R3 from the table.
```

**Answer:** R4

**Prompt:**
```
A router picks the next hop for a destination address from its routing table.
Every destination and prefix below is a 9-bit binary string.

Routing table (prefix, length -> next hop (metric)):
  0000, /4 -> R6 (metric 8)
  00, /2 -> R8 (metric 14)
  000, /3 -> R7 (metric 10)
  000011, /6 -> R5 (metric 0)
  0000, /4 -> R2 (metric 13)
  10001, /5 -> R5 (metric 14)
  10001111, /8 -> R4 (metric 16)
  10001, /5 -> R1 (metric 7)

Destination to route: 000011111

Selection rules, applied in order:
  1. An entry matches if its /P prefix equals the destination's first P bits.
  2. Among matching entries, keep the longest prefix (largest P).
  3. Ties on prefix length are broken by the lowest metric.
  4. Remaining ties on length and metric are broken by the lexicographically smallest next hop.

The answer is the selected next hop, written exactly as a hop name like R3 from the table.
```

**Answer:** R5

## Level 2

**Prompt:**
```
A router picks the next hop for a destination address from its routing table.
Every destination and prefix below is a 10-bit binary string.

Routing table (prefix, length -> next hop (metric)):
  0, /1 -> R10 (metric 0)
  0001000, /7 -> R9 (metric 16)
  000, /3 -> R5 (metric 20)
  0001000100, /10 -> R7 (metric 4)
  00010, /5 -> R6 (metric 16)
  0001000100, /10 -> R2 (metric 6)
  1001, /4 -> R1 (metric 2)
  100100010, /9 -> R5 (metric 13)
  100, /3 -> R5 (metric 19)
  10010, /5 -> R6 (metric 6)

Destination to route: 0001000100

Selection rules, applied in order:
  1. An entry matches if its /P prefix equals the destination's first P bits.
  2. Among matching entries, keep the longest prefix (largest P).
  3. Ties on prefix length are broken by the lowest metric.
  4. Remaining ties on length and metric are broken by the lexicographically smallest next hop.

The answer is the selected next hop, written exactly as a hop name like R3 from the table.
```

**Answer:** R7

**Prompt:**
```
A router picks the next hop for a destination address from its routing table.
Every destination and prefix below is a 10-bit binary string.

Routing table (prefix, length -> next hop (metric)):
  0101010, /7 -> R3 (metric 15)
  0101010111, /10 -> R9 (metric 3)
  010101, /6 -> R2 (metric 19)
  0101, /4 -> R1 (metric 19)
  0101010, /7 -> R5 (metric 3)
  1101, /4 -> R10 (metric 20)
  1101010, /7 -> R1 (metric 16)
  1101, /4 -> R10 (metric 1)
  110101011, /9 -> R9 (metric 2)
  110101011, /9 -> R6 (metric 18)

Destination to route: 0101010111

Selection rules, applied in order:
  1. An entry matches if its /P prefix equals the destination's first P bits.
  2. Among matching entries, keep the longest prefix (largest P).
  3. Ties on prefix length are broken by the lowest metric.
  4. Remaining ties on length and metric are broken by the lexicographically smallest next hop.

The answer is the selected next hop, written exactly as a hop name like R3 from the table.
```

**Answer:** R9

## Level 5

**Prompt:**
```
A router picks the next hop for a destination address from its routing table.
Every destination and prefix below is a 12-bit binary string.

Routing table (prefix, length -> next hop (metric)):
  1100111, /7 -> R4 (metric 18)
  1100, /4 -> R8 (metric 2)
  110011111, /9 -> R9 (metric 12)
  11001111, /8 -> R1 (metric 20)
  11001, /5 -> R3 (metric 11)
  1100, /4 -> R7 (metric 10)
  1100111, /7 -> R5 (metric 9)
  1100, /4 -> R11 (metric 2)
  1100111, /7 -> R10 (metric 22)
  1100, /4 -> R6 (metric 8)
  0100111, /7 -> R6 (metric 16)
  0, /1 -> R3 (metric 4)
  010, /3 -> R7 (metric 4)

Destination to route: 110011111011

Selection rules, applied in order:
  1. An entry matches if its /P prefix equals the destination's first P bits.
  2. Among matching entries, keep the longest prefix (largest P).
  3. Ties on prefix length are broken by the lowest metric.
  4. Remaining ties on length and metric are broken by the lexicographically smallest next hop.

The answer is the selected next hop, written exactly as a hop name like R3 from the table.
```

**Answer:** R9

**Prompt:**
```
A router picks the next hop for a destination address from its routing table.
Every destination and prefix below is a 12-bit binary string.

Routing table (prefix, length -> next hop (metric)):
  10101101, /8 -> R8 (metric 14)
  10101101001, /11 -> R5 (metric 22)
  10101101, /8 -> R13 (metric 24)
  101011010, /9 -> R3 (metric 7)
  1010, /4 -> R2 (metric 5)
  101, /3 -> R10 (metric 5)
  10101101001, /11 -> R1 (metric 5)
  101011010, /9 -> R11 (metric 11)
  001011010010, /12 -> R11 (metric 20)
  001011010010, /12 -> R5 (metric 3)
  00101101, /8 -> R3 (metric 9)
  001011, /6 -> R10 (metric 17)
  001011010, /9 -> R1 (metric 20)

Destination to route: 101011010010

Selection rules, applied in order:
  1. An entry matches if its /P prefix equals the destination's first P bits.
  2. Among matching entries, keep the longest prefix (largest P).
  3. Ties on prefix length are broken by the lowest metric.
  4. Remaining ties on length and metric are broken by the lexicographically smallest next hop.

The answer is the selected next hop, written exactly as a hop name like R3 from the table.
```

**Answer:** R1
