# Level 0
## Example
Prompt:
```
An m-bit Bloom filter has all m bit positions 0..m-1 initialized to 0. Its k hash functions map an item x to a bit index by h_j(x) = (a_j*x + b_j) mod m. The parameters are: M:
24

K:
2

A:
[7, 20]

B:
[22, 1]

Bits:
010111000000010000000010

Queries:
[943, 805, 39, 446, 504] (in the 'bits' string, position 0 is the leftmost bit). A set of items was inserted by setting, for every item, each of its k hash positions to 1. Query the items listed under 'queries'. An item is DEFINITELY ABSENT if at least one of its k hash positions is still 0; otherwise it is only POSSIBLY PRESENT. Give your answer as the 0-based indices (in the order the items appear under 'queries') of the items that are DEFINITELY ABSENT, formatted as a space-separated list sorted in ascending order; if queries 1 and 3 were absent, answer: 1 3. Write only the space-separated indices.
```
Answer:
```
0 1 2 3
```
## Example
Prompt:
```
An m-bit Bloom filter has all m bit positions 0..m-1 initialized to 0. Its k hash functions map an item x to a bit index by h_j(x) = (a_j*x + b_j) mod m. The parameters are: M:
24

K:
2

A:
[16, 22]

B:
[1, 4]

Bits:
010000000100100011101000

Queries:
[660, 164, 45, 922, 270] (in the 'bits' string, position 0 is the leftmost bit). A set of items was inserted by setting, for every item, each of its k hash positions to 1. Query the items listed under 'queries'. An item is DEFINITELY ABSENT if at least one of its k hash positions is still 0; otherwise it is only POSSIBLY PRESENT. Give your answer as the 0-based indices (in the order the items appear under 'queries') of the items that are DEFINITELY ABSENT, formatted as a space-separated list sorted in ascending order; if queries 1 and 3 were absent, answer: 1 3. Write only the space-separated indices.
```
Answer:
```
0 2 3
```
# Level 2
## Example
Prompt:
```
An m-bit Bloom filter has all m bit positions 0..m-1 initialized to 0. Its k hash functions map an item x to a bit index by h_j(x) = (a_j*x + b_j) mod m. The parameters are: M:
40

K:
4

A:
[22, 14, 5, 31]

B:
[32, 6, 33, 20]

Bits:
0011110011101100101000111110101010101101

Queries:
[41, 113, 927, 130, 43, 886, 488] (in the 'bits' string, position 0 is the leftmost bit). A set of items was inserted by setting, for every item, each of its k hash positions to 1. Query the items listed under 'queries'. An item is DEFINITELY ABSENT if at least one of its k hash positions is still 0; otherwise it is only POSSIBLY PRESENT. Give your answer as the 0-based indices (in the order the items appear under 'queries') of the items that are DEFINITELY ABSENT, formatted as a space-separated list sorted in ascending order; if queries 1 and 3 were absent, answer: 1 3. Write only the space-separated indices.
```
Answer:
```
0 1 4 5 6
```
## Example
Prompt:
```
An m-bit Bloom filter has all m bit positions 0..m-1 initialized to 0. Its k hash functions map an item x to a bit index by h_j(x) = (a_j*x + b_j) mod m. The parameters are: M:
40

K:
4

A:
[14, 12, 10, 37]

B:
[31, 19, 15, 6]

Bits:
0101010100010101100111000101000111110011

Queries:
[976, 213, 308, 698, 339, 662, 889] (in the 'bits' string, position 0 is the leftmost bit). A set of items was inserted by setting, for every item, each of its k hash positions to 1. Query the items listed under 'queries'. An item is DEFINITELY ABSENT if at least one of its k hash positions is still 0; otherwise it is only POSSIBLY PRESENT. Give your answer as the 0-based indices (in the order the items appear under 'queries') of the items that are DEFINITELY ABSENT, formatted as a space-separated list sorted in ascending order; if queries 1 and 3 were absent, answer: 1 3. Write only the space-separated indices.
```
Answer:
```
2 4 6
```
# Level 5
## Example
Prompt:
```
An m-bit Bloom filter has all m bit positions 0..m-1 initialized to 0. Its k hash functions map an item x to a bit index by h_j(x) = (a_j*x + b_j) mod m. The parameters are: M:
64

K:
7

A:
[38, 44, 32, 25, 9, 28, 55]

B:
[11, 45, 43, 46, 53, 39, 55]

Bits:
1101110111010111101001110111111101000111110101011101110011011111

Queries:
[413, 122, 338, 246, 936, 738, 431, 563, 145, 432] (in the 'bits' string, position 0 is the leftmost bit). A set of items was inserted by setting, for every item, each of its k hash positions to 1. Query the items listed under 'queries'. An item is DEFINITELY ABSENT if at least one of its k hash positions is still 0; otherwise it is only POSSIBLY PRESENT. Give your answer as the 0-based indices (in the order the items appear under 'queries') of the items that are DEFINITELY ABSENT, formatted as a space-separated list sorted in ascending order; if queries 1 and 3 were absent, answer: 1 3. Write only the space-separated indices.
```
Answer:
```
0 1 2 3 7 8
```
## Example
Prompt:
```
An m-bit Bloom filter has all m bit positions 0..m-1 initialized to 0. Its k hash functions map an item x to a bit index by h_j(x) = (a_j*x + b_j) mod m. The parameters are: M:
64

K:
7

A:
[26, 6, 39, 22, 55, 49, 16]

B:
[15, 48, 31, 2, 46, 24, 11]

Bits:
0110101111111010000111100101111011110110101111100111111011110110

Queries:
[307, 710, 342, 893, 910, 195, 837, 784, 187, 97] (in the 'bits' string, position 0 is the leftmost bit). A set of items was inserted by setting, for every item, each of its k hash positions to 1. Query the items listed under 'queries'. An item is DEFINITELY ABSENT if at least one of its k hash positions is still 0; otherwise it is only POSSIBLY PRESENT. Give your answer as the 0-based indices (in the order the items appear under 'queries') of the items that are DEFINITELY ABSENT, formatted as a space-separated list sorted in ascending order; if queries 1 and 3 were absent, answer: 1 3. Write only the space-separated indices.
```
Answer:
```
0 3 4 6 7 8 9
```
