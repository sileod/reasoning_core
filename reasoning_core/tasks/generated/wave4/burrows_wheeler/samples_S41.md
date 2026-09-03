# Level 0

## Example 1

### Prompt

String:
bbbaac$

Alphabet:
abc

This is the Burrows-Wheeler transform of some original string, produced by appending the terminator character '$' (smallest symbol) to that string, sorting all its rotations, and taking the last character of each rotation in sorted order. Invert the transform (using the LF-mapping over the sorted first column) to recover the original string and report it as the length-6 string of letters (without the '$' terminator).

The answer is a string whose characters come from the alphabet plus '$'.

### Answer

cbabab

## Example 2

### Prompt

String:
accaa

Alphabet:
abc

Calculate the Burrows-Wheeler transform of the string. Append the terminator character '$' (smallest symbol) to the end of the string, take all its rotations, sort them lexicographically, and read the last character of each rotation in sorted order. Report the full length-6 transform string (which contains exactly one '$').

The answer is a string whose characters come from the alphabet plus '$'.

### Answer

aac$ca



# Level 2

## Example 1

### Prompt

String:
bcabbc$ca

Alphabet:
abc

This is the Burrows-Wheeler transform of some original string, produced by appending the terminator character '$' (smallest symbol) to that string, sorting all its rotations, and taking the last character of each rotation in sorted order. Invert the transform (using the LF-mapping over the sorted first column) to recover the original string and report it as the length-8 string of letters (without the '$' terminator).

The answer is a string whose characters come from the alphabet plus '$'.

### Answer

caaccbbb

## Example 2

### Prompt

String:
acbabaa

Alphabet:
abc

Calculate the Burrows-Wheeler transform of the string. Append the terminator character '$' (smallest symbol) to the end of the string, take all its rotations, sort them lexicographically, and read the last character of each rotation in sorted order. Report the full length-8 transform string (which contains exactly one '$').

The answer is a string whose characters come from the alphabet plus '$'.

### Answer

aabb$aca



# Level 5

## Example 1

### Prompt

String:
becd

Alphabet:
abcdef

Calculate the Burrows-Wheeler transform of the string. Append the terminator character '$' (smallest symbol) to the end of the string, take all its rotations, sort them lexicographically, and read the last character of each rotation in sorted order. Report the full length-5 transform string (which contains exactly one '$').

The answer is a string whose characters come from the alphabet plus '$'.

### Answer

d$ecb

## Example 2

### Prompt

String:
defeebeafd$d

Alphabet:
abcdef

This is the Burrows-Wheeler transform of some original string, produced by appending the terminator character '$' (smallest symbol) to that string, sorting all its rotations, and taking the last character of each rotation in sorted order. Invert the transform (using the LF-mapping over the sorted first column) to recover the original string and report it as the length-11 string of letters (without the '$' terminator).

The answer is a string whose characters come from the alphabet plus '$'.

### Answer

fbdfedeeaed


