# Prefix Code Decode v1 samples

## Level 0

Below is a prefix-free binary codebook and a concatenated bitstream produced by encoding a sequence of symbols with that codebook.
Symbols:
['s0', 's1', 's2', 's3']

Codebook:
0011010 0
01 1
1010111 2
111 3

Bitstream:
01101011101101011101011010111001101010101111010111101011101
Decode the bitstream into the symbol sequence. The answer is a space-separated list of symbols in order, for example: s0 s2 s0.

Answer: s1 s2 s1 s2 s1 s1 s2 s0 s2 s2 s2 s1

Below is a prefix-free binary codebook and a concatenated bitstream produced by encoding a sequence of symbols with that codebook.
Symbols:
['s0', 's1', 's2', 's3']

Codebook:
00101 3
0100 1
1001 0
1100011 2

Bitstream:
0100110001110010010101000010110010100010010010010100101
Decode the bitstream into the symbol sequence. The answer is a space-separated list of symbols in order, for example: s0 s2 s0.

Answer: s1 s2 s0 s3 s1 s3 s0 s1 s1 s0 s3 s3

## Level 2

Below is a prefix-free binary codebook and a concatenated bitstream produced by encoding a sequence of symbols with that codebook.
Symbols:
['s0', 's1', 's2', 's3', 's4', 's5']

Codebook:
0010 4
00110 1
01100 3
01111 2
1001101 0
1111 5

Bitstream:
001011110011011110011001111111100101111011000010100110101100011110011000110001000110
Decode the bitstream into the symbol sequence. The answer is a space-separated list of symbols in order, for example: s0 s2 s0.

Answer: s4 s5 s1 s5 s1 s2 s5 s4 s5 s3 s4 s0 s3 s2 s1 s1 s4 s1

Below is a prefix-free binary codebook and a concatenated bitstream produced by encoding a sequence of symbols with that codebook.
Symbols:
['s0', 's1', 's2', 's3', 's4', 's5']

Codebook:
000011 3
0100000000 5
0101101 0
011 1
11000100 2
110010 4

Bitstream:
010110111000100110010110001000110110100000000110001001100010001101011010111100100100000000110010010000000001000000000101101
Decode the bitstream into the symbol sequence. The answer is a space-separated list of symbols in order, for example: s0 s2 s0.

Answer: s0 s2 s4 s2 s1 s1 s5 s2 s2 s1 s0 s1 s4 s5 s4 s5 s5 s0

## Level 5

Below is a prefix-free binary codebook and a concatenated bitstream produced by encoding a sequence of symbols with that codebook.
Symbols:
['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']

Codebook:
000110 5
000111 3
001 4
01001 1
011 0
1000100 7
1001011 2
110 6
111110100 8

Bitstream:
11111010000011111111010000011101001001000110110000110111110100010010011101000100100010000011101101001110000111011110100101100011010001001001011110
Decode the bitstream into the symbol sequence. The answer is a space-separated list of symbols in order, for example: s0 s2 s0.

Answer: s8 s3 s8 s3 s1 s4 s5 s6 s5 s8 s1 s4 s6 s7 s7 s3 s0 s1 s6 s3 s0 s6 s2 s5 s7 s2 s6

Below is a prefix-free binary codebook and a concatenated bitstream produced by encoding a sequence of symbols with that codebook.
Symbols:
['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']

Codebook:
0000000 4
000110 8
0010101110 7
001111 5
01 0
100111001 2
10110 6
1101010 3
11011 1

Bitstream:
00011001010011110001101101100011011010101001110010000000010111011001010111010011100110110000000001010000000001111011101101101100010101110100111001
Decode the bitstream into the symbol sequence. The answer is a space-separated list of symbols in order, for example: s0 s2 s0.

Answer: s8 s0 s0 s5 s8 s1 s8 s3 s2 s4 s0 s0 s1 s7 s2 s6 s4 s0 s0 s4 s5 s0 s1 s0 s6 s7 s2
