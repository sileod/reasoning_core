## Level 0

### Example 1

Query:
A binary linear code over GF(2) uses 1-based positions. It is defined by these parity-check rules: for each rule, the XOR of the bits at the listed positions must equal 0.

Rules:
['positions [1, 2, 3, 5]', 'positions [2, 3, 4, 6]', 'positions [3, 5, 6, 7]']

Received Word:
1101010

Instruction:
Apply each rule, locate the single corrupted bit if there is one (at most one bit is wrong), and give the corrected codeword.

The answer is the corrected codeword as a binary string of 7 bits (0s and 1s, no spaces).

**Answer:** 1101000

### Example 2

Query:
A binary linear code over GF(2) uses 1-based positions. It is defined by these parity-check rules: for each rule, the XOR of the bits at the listed positions must equal 0.

Rules:
['positions [2, 3, 5, 7]', 'positions [2, 3, 4, 6]', 'positions [1, 2, 4, 7]']

Received Word:
0010111

Instruction:
Apply each rule, locate the single corrupted bit if there is one (at most one bit is wrong), and give the corrected codeword.

The answer is the corrected codeword as a binary string of 7 bits (0s and 1s, no spaces).

**Answer:** 0010110

## Level 2

### Example 1

Query:
A binary linear code over GF(2) uses 1-based positions. It is defined by these parity-check rules: for each rule, the XOR of the bits at the listed positions must equal 0.

Rules:
['positions [1, 2, 3, 10]', 'positions [1, 2, 3, 4, 5, 6, 8, 9, 10]', 'positions [1, 2, 4, 7, 8, 9]', 'positions [2, 3, 5, 6, 7, 8, 11]', 'positions [2, 3, 5, 9, 10]']

Received Word:
01000011010

Instruction:
Apply each rule, locate the single corrupted bit if there is one (at most one bit is wrong), and give the corrected codeword.

The answer is the corrected codeword as a binary string of 11 bits (0s and 1s, no spaces).

**Answer:** 01000010010

### Example 2

Query:
A binary linear code over GF(2) uses 1-based positions. It is defined by these parity-check rules: for each rule, the XOR of the bits at the listed positions must equal 0.

Rules:
['positions [1, 3, 4, 6, 7, 8, 9, 11]', 'positions [2, 3, 5, 6, 7, 8, 9, 11]', 'positions [1, 3, 4, 8, 11]', 'positions [3, 4, 7, 10, 11]', 'positions [1, 4, 5, 7, 9, 11]']

Received Word:
00010110011

Instruction:
Apply each rule, locate the single corrupted bit if there is one (at most one bit is wrong), and give the corrected codeword.

The answer is the corrected codeword as a binary string of 11 bits (0s and 1s, no spaces).

**Answer:** 00011110011

## Level 5

### Example 1

Query:
A binary linear code over GF(2) uses 1-based positions. It is defined by these parity-check rules: for each rule, the XOR of the bits at the listed positions must equal 0.

Rules:
['positions [2, 3, 4, 8, 10, 11, 15, 16]', 'positions [1, 2, 3, 6, 9, 10, 12, 15, 17]', 'positions [1, 11, 13, 14, 15, 17]', 'positions [3, 6, 7, 9, 16, 17]', 'positions [1, 2, 3, 5, 6, 8, 11, 12, 13, 17]', 'positions [4, 6, 7, 8, 10, 14, 16, 17]', 'positions [1, 3, 4, 5, 6, 8, 9, 14, 16, 17]', 'positions [1, 2, 3, 7, 8, 10, 11, 14, 15, 17]']

Received Word:
10010100110111111

Instruction:
Apply each rule, locate the single corrupted bit if there is one (at most one bit is wrong), and give the corrected codeword.

The answer is the corrected codeword as a binary string of 17 bits (0s and 1s, no spaces).

**Answer:** 00010100110111111

### Example 2

Query:
A binary linear code over GF(2) uses 1-based positions. It is defined by these parity-check rules: for each rule, the XOR of the bits at the listed positions must equal 0.

Rules:
['positions [1, 2, 3, 4, 5, 9, 10, 14, 15]', 'positions [3, 4, 6, 7, 8, 9, 10, 11, 12, 15]', 'positions [1, 3, 7, 9, 10, 11, 15]', 'positions [1, 2, 9, 11, 16]', 'positions [2, 3, 5, 6, 8, 9, 11, 14, 15, 16, 17]', 'positions [1, 2, 6, 9, 10, 12, 13, 14, 15, 16]', 'positions [2, 4, 5, 6, 7, 10, 15, 16]', 'positions [2, 3, 4, 7, 8, 10, 11, 16]']

Received Word:
01001011100111101

Instruction:
Apply each rule, locate the single corrupted bit if there is one (at most one bit is wrong), and give the corrected codeword.

The answer is the corrected codeword as a binary string of 17 bits (0s and 1s, no spaces).

**Answer:** 01101011100111101

