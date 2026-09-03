
## Level 0

### Example

A hash table of 11 slots indexed 0 through 10 uses open addressing with linear probing: h(key) = (key * 3) mod 11, and to insert key k the table checks slots h(k), h(k)+3, h(k)+2*3, ... wrapping modulo 11, placing k in the first slot not yet occupied.

Each of these keys is inserted in the order given, each into the first unoccupied slot its probe sequence reaches:
165, 73, 120, 191

Now the key 262 is probed with the same rule. Considering the slots it would find occupied by the keys above, the first slot on 262's probe sequence that is empty is where it would be inserted.

The answer is the index of that slot.

Answer: 5

### Example

A hash table of 11 slots indexed 0 through 10 uses open addressing with linear probing: h(key) = (key * 2) mod 11, and to insert key k the table checks slots h(k), h(k)+2, h(k)+2*2, ... wrapping modulo 11, placing k in the first slot not yet occupied.

Each of these keys is inserted in the order given, each into the first unoccupied slot its probe sequence reaches:
131, 53, 15, 51

Now the key 207 is probed with the same rule. Considering the slots it would find occupied by the keys above, the first slot on 207's probe sequence that is empty is where it would be inserted.

The answer is the index of that slot.

Answer: 0

## Level 2

### Example

A hash table of 19 slots indexed 0 through 18 uses open addressing with linear probing: h(key) = (key * 1) mod 19, and to insert key k the table checks slots h(k), h(k)+1, h(k)+2*1, ... wrapping modulo 19, placing k in the first slot not yet occupied.

Each of these keys is inserted in the order given, each into the first unoccupied slot its probe sequence reaches:
137, 37, 35, 43, 61, 65

Now the key 247 is probed with the same rule. Considering the slots it would find occupied by the keys above, the first slot on 247's probe sequence that is empty is where it would be inserted.

The answer is the index of that slot.

Answer: 0

### Example

A hash table of 19 slots indexed 0 through 18 uses open addressing with linear probing: h(key) = (key * 3) mod 19, and to insert key k the table checks slots h(k), h(k)+3, h(k)+2*3, ... wrapping modulo 19, placing k in the first slot not yet occupied.

Each of these keys is inserted in the order given, each into the first unoccupied slot its probe sequence reaches:
105, 36, 156, 71, 37, 41

Now the key 337 is probed with the same rule. Considering the slots it would find occupied by the keys above, the first slot on 337's probe sequence that is empty is where it would be inserted.

The answer is the index of that slot.

Answer: 7

## Level 5

### Example

A hash table of 46 slots indexed 0 through 45 uses open addressing with linear probing: h(key) = (key * 1) mod 46, and to insert key k the table checks slots h(k), h(k)+1, h(k)+2*1, ... wrapping modulo 46, placing k in the first slot not yet occupied.

Each of these keys is inserted in the order given, each into the first unoccupied slot its probe sequence reaches:
69, 24, 170, 132, 110, 159, 145, 99, 136

Now the key 290 is probed with the same rule. Considering the slots it would find occupied by the keys above, the first slot on 290's probe sequence that is empty is where it would be inserted.

The answer is the index of that slot.

Answer: 14

### Example

A hash table of 46 slots indexed 0 through 45 uses open addressing with linear probing: h(key) = (key * 3) mod 46, and to insert key k the table checks slots h(k), h(k)+3, h(k)+2*3, ... wrapping modulo 46, placing k in the first slot not yet occupied.

Each of these keys is inserted in the order given, each into the first unoccupied slot its probe sequence reaches:
70, 98, 76, 30, 182, 154, 69, 127, 180

Now the key 244 is probed with the same rule. Considering the slots it would find occupied by the keys above, the first slot on 244's probe sequence that is empty is where it would be inserted.

The answer is the index of that slot.

Answer: 42
