# Level 0

## Example 1

### Prompt

States:
states 0..4, start state 3, accepting states {0, 2, 3, 4}

Alphabet:
{a, b}

Transitions:
state 0: on a -> {3}; on b -> {2}
state 1: on a -> {2}; on b -> {0, 2}
state 2: on a -> {0, 4}; on b -> {3}
state 3: on a -> {1, 2}; on b -> {0}
state 4: on a -> {0, 1}; on b -> {1, 4}

Word:
the input word is 'abbba' (length 5)

Compute the number of accepting states that are active (reachable) after reading the entire input word 'abbba'.

The answer is a nonnegative integer.

### Answer

4

## Example 2

### Prompt

States:
states 0..4, start state 3, accepting states {0, 1, 3, 4}

Alphabet:
{a, b}

Transitions:
state 0: on a -> {1, 4}; on b -> {0}
state 1: on a -> {1, 3}; on b -> {2}
state 2: on a -> {1, 4}; on b -> {2, 3}
state 3: on a -> {2, 4}; on b -> {4}
state 4: on a -> {3, 4}; on b -> {0, 3}

Word:
the input word is 'aaaaa' (length 5)

Compute the number of accepting states that are active (reachable) after reading the entire input word 'aaaaa'.

The answer is a nonnegative integer.

### Answer

3



# Level 2

## Example 1

### Prompt

States:
states 0..8, start state 1, accepting states {7}

Alphabet:
{a, b}

Transitions:
state 0: on a -> {3}; on b -> {0, 4}
state 1: on a -> {5, 8}; on b -> {0}
state 2: on a -> {2, 3}; on b -> {7}
state 3: on a -> {5, 8}; on b -> {4}
state 4: on a -> {2, 8}; on b -> {2, 5}
state 5: on a -> {3, 4}; on b -> {1, 8}
state 6: on a -> {8}; on b -> {2, 6}
state 7: on a -> {1, 5}; on b -> {2}
state 8: on a -> {1}; on b -> {0, 2}

Word:
the input word is 'bbabbabaa' (length 9)

Compute the number of accepting states that are active (reachable) after reading the entire input word 'bbabbabaa'.

The answer is a nonnegative integer.

### Answer

0

## Example 2

### Prompt

States:
states 0..8, start state 0, accepting states {0, 1, 2, 3, 4, 5, 6, 7, 8}

Alphabet:
{a, b}

Transitions:
state 0: on a -> {2}; on b -> {4, 6}
state 1: on a -> {1, 5}; on b -> {1, 5}
state 2: on a -> {7}; on b -> {1, 2}
state 3: on a -> {7, 8}; on b -> {0, 3}
state 4: on a -> {5}; on b -> {0, 1}
state 5: on a -> {1, 6}; on b -> {3}
state 6: on a -> {2}; on b -> {2, 3}
state 7: on a -> {5, 8}; on b -> {3, 5}
state 8: on a -> {2}; on b -> {0, 4}

Word:
the input word is 'bbababbba' (length 9)

Compute the number of accepting states that are active (reachable) after reading the entire input word 'bbababbba'.

The answer is a nonnegative integer.

### Answer

6



# Level 5

## Example 1

### Prompt

States:
states 0..14, start state 8, accepting states {9}

Alphabet:
{a, b, c, d, e}

Transitions:
state 0: on a -> {3, 10, 11}; on b -> {0, 4}; on c -> {14}; on d -> {1, 4}; on e -> {0, 2, 14}
state 1: on a -> {2, 6, 10}; on b -> {1, 5}; on c -> {2}; on d -> {5, 10}; on e -> {12, 13}
state 2: on a -> {8}; on b -> {1, 5, 13}; on c -> {7, 12, 13}; on d -> {4, 10}; on e -> {0, 11}
state 3: on a -> {3, 7, 8}; on b -> {2}; on c -> {3, 6, 8}; on d -> {3, 11, 14}; on e -> {3, 5, 10}
state 4: on a -> {3}; on b -> {2, 7, 8}; on c -> {8, 12}; on d -> {0, 9}; on e -> {14}
state 5: on a -> {9}; on b -> {1, 8, 10}; on c -> {1, 5, 7}; on d -> {1, 4, 9}; on e -> {4, 6, 14}
state 6: on a -> {8}; on b -> {11, 13, 14}; on c -> {2, 5, 11}; on d -> {0, 12, 13}; on e -> {4, 12}
state 7: on a -> {1, 8}; on b -> {6, 8, 11}; on c -> {5, 6}; on d -> {3, 5}; on e -> {3, 14}
state 8: on a -> {6, 13, 14}; on b -> {1}; on c -> {7, 14}; on d -> {0, 8}; on e -> {6, 14}
state 9: on a -> {9, 13}; on b -> {7}; on c -> {10}; on d -> {4, 8}; on e -> {1}
state 10: on a -> {8}; on b -> {2, 8}; on c -> {8, 9, 14}; on d -> {0}; on e -> {2, 9}
state 11: on a -> {8}; on b -> {11}; on c -> {6, 12}; on d -> {14}; on e -> {14}
state 12: on a -> {3, 7, 14}; on b -> {3, 6}; on c -> {1}; on d -> {0, 13}; on e -> {2}
state 13: on a -> {4, 5}; on b -> {14}; on c -> {7}; on d -> {5}; on e -> {10}
state 14: on a -> {7}; on b -> {4, 14}; on c -> {0, 8}; on d -> {3, 11}; on e -> {8, 14}

Word:
the input word is 'dbedcecaaccbbca' (length 15)

Compute the largest accepting state that is active after reading the entire input word 'dbedcecaaccbbca'.

The answer is a single state number (integer).

### Answer

9

## Example 2

### Prompt

States:
states 0..14, start state 9, accepting states {0, 5, 6, 7, 9, 10, 12, 13, 14}

Alphabet:
{a, b, c, d, e}

Transitions:
state 0: on a -> {3, 9}; on b -> {1}; on c -> {4, 7}; on d -> {9}; on e -> {0, 8, 14}
state 1: on a -> {8}; on b -> {4}; on c -> {8, 11}; on d -> {13}; on e -> {4, 9}
state 2: on a -> {10, 11, 12}; on b -> {9}; on c -> {5, 12}; on d -> {3, 9}; on e -> {9}
state 3: on a -> {6, 7}; on b -> {2, 12, 13}; on c -> {6}; on d -> {4}; on e -> {6, 8, 13}
state 4: on a -> {5, 9, 14}; on b -> {0, 8, 13}; on c -> {2, 7}; on d -> {0}; on e -> {2}
state 5: on a -> {3, 6, 12}; on b -> {3, 10, 13}; on c -> {5}; on d -> {4}; on e -> {5, 7, 8}
state 6: on a -> {6, 8, 11}; on b -> {6, 12}; on c -> {2, 7}; on d -> {2, 7, 8}; on e -> {0, 2, 13}
state 7: on a -> {4}; on b -> {2}; on c -> {1, 2, 9}; on d -> {0, 12, 14}; on e -> {14}
state 8: on a -> {6, 9, 14}; on b -> {0}; on c -> {3, 11}; on d -> {5, 8, 12}; on e -> {10}
state 9: on a -> {6, 8}; on b -> {0, 14}; on c -> {1, 6, 8}; on d -> {4, 6, 13}; on e -> {9}
state 10: on a -> {1, 3}; on b -> {2, 7, 11}; on c -> {9}; on d -> {0, 11, 12}; on e -> {12, 13, 14}
state 11: on a -> {1, 10}; on b -> {8, 10, 11}; on c -> {1, 4, 10}; on d -> {8, 9, 14}; on e -> {3, 9}
state 12: on a -> {0, 6, 14}; on b -> {6, 10, 12}; on c -> {14}; on d -> {3, 4, 12}; on e -> {3}
state 13: on a -> {5}; on b -> {5, 8}; on c -> {1}; on d -> {2, 6, 7}; on e -> {11, 12, 13}
state 14: on a -> {3, 4, 7}; on b -> {1, 2}; on c -> {10}; on d -> {1, 10}; on e -> {11, 13}

Word:
the input word is 'cabeabbeceeccac' (length 15)

Compute the number of distinct states that are active (reachable) after reading the entire input word 'cabeabbeceeccac'.

The answer is a nonnegative integer.

### Answer

12


