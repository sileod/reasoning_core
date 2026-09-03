# semantic_version_precedence samples

## Level 0

### Example

Prompt:

Under Semantic Versioning, precedence is determined by comparing major, minor, and patch as numbers, and when those are equal, by pre-release identifiers: a version without a pre-release is higher than one with a pre-release, and pre-release identifiers compare left to right with numeric identifiers ordered numerically (and lower than non-numeric ones) while non-numeric identifiers compare by ASCII order, and a longer pre-release whose prefix matches is higher.

Version A:
2.2.3

Version B:
1.0.2

Which of the two versions has higher precedence? The answer is that version string verbatim, or 'equal' if they have equal precedence.

Answer:

2.2.3

### Example

Prompt:

Under Semantic Versioning, precedence is determined by comparing major, minor, and patch as numbers, and when those are equal, by pre-release identifiers: a version without a pre-release is higher than one with a pre-release, and pre-release identifiers compare left to right with numeric identifiers ordered numerically (and lower than non-numeric ones) while non-numeric identifiers compare by ASCII order, and a longer pre-release whose prefix matches is higher.

Version A:
1.0.2

Version B:
1.1.1

Which of the two versions has higher precedence? The answer is that version string verbatim, or 'equal' if they have equal precedence.

Answer:

1.1.1


## Level 2

### Example

Prompt:

Under Semantic Versioning, precedence is determined by comparing major, minor, and patch as numbers, and when those are equal, by pre-release identifiers: a version without a pre-release is higher than one with a pre-release, and pre-release identifiers compare left to right with numeric identifiers ordered numerically (and lower than non-numeric ones) while non-numeric identifiers compare by ASCII order, and a longer pre-release whose prefix matches is higher.

Version A:
0.7.3-dev.gamma

Version B:
0.7.3-zeta.dev

Which of the two versions has higher precedence? The answer is that version string verbatim, or 'equal' if they have equal precedence.

Answer:

0.7.3-zeta.dev

### Example

Prompt:

Under Semantic Versioning, precedence is determined by comparing major, minor, and patch as numbers, and when those are equal, by pre-release identifiers: a version without a pre-release is higher than one with a pre-release, and pre-release identifiers compare left to right with numeric identifiers ordered numerically (and lower than non-numeric ones) while non-numeric identifiers compare by ASCII order, and a longer pre-release whose prefix matches is higher.

Version A:
6.4.6

Version B:
0.1.1

Which of the two versions has higher precedence? The answer is that version string verbatim, or 'equal' if they have equal precedence.

Answer:

6.4.6


## Level 5

### Example

Prompt:

Under Semantic Versioning, precedence is determined by comparing major, minor, and patch as numbers, and when those are equal, by pre-release identifiers: a version without a pre-release is higher than one with a pre-release, and pre-release identifiers compare left to right with numeric identifiers ordered numerically (and lower than non-numeric ones) while non-numeric identifiers compare by ASCII order, and a longer pre-release whose prefix matches is higher.

Version A:
5.13.6-31.rc.37

Version B:
5.13.6-16

Which of the two versions has higher precedence? The answer is that version string verbatim, or 'equal' if they have equal precedence.

Answer:

5.13.6-31.rc.37

### Example

Prompt:

Under Semantic Versioning, precedence is determined by comparing major, minor, and patch as numbers, and when those are equal, by pre-release identifiers: a version without a pre-release is higher than one with a pre-release, and pre-release identifiers compare left to right with numeric identifiers ordered numerically (and lower than non-numeric ones) while non-numeric identifiers compare by ASCII order, and a longer pre-release whose prefix matches is higher.

Version A:
2.7.11-zeta.9

Version B:
2.7.11-zeta.9

Which of the two versions has higher precedence? The answer is that version string verbatim, or 'equal' if they have equal precedence.

Answer:

equal

