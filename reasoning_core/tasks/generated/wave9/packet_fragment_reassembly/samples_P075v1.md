## Level 0

### Example 1

Prompt:

```
A payload message has been split into overlapping fragments. Each fragment gives its ORIGINAL offset (position where its first character belongs in the payload) and its content. Some fragments were lost in transit and are marked MISSING; a missing fragment contributes no characters and provides no information.

Rules:
- Fragments may overlap: where two fragments cover the same payload position they must agree on the character there. Reconcile overlaps by alignment of offsets; the payload is the longest consistent concatenation of the fragments by their offsets.
- Any payload position covered by no non-missing fragment is a GAP that cannot be reconstructed from the available data.
- The payload has a known total length of 26 characters (positions 0 .. 25).
- Overlap tolerance between adjacent fragments is at most 3 characters.

Fragments:
  - index 0, offset 0: CFHH
  - index 1, offset 4: GDFHFFFGBGE
  - index 2, offset 17: HHGHDHCAH
  - index 3, offset 18: HGHDHCAH
  - index 4, offset 23: CAH

Task: reconstruct the full payload by merging the fragments at their stated offsets. If the non-missing fragments cover every position of the payload consistently, the answer is that fully reconstructed payload string. If any position cannot be determined (because no fragment covers it), the answer is the error state UNRECOVERABLE: followed by the sorted list of uncovered intervals, using a-b for an interval from a to b inclusive and a single number for a lone position, intervals separated by commas. Give only the answer.
```

Answer:

```
UNRECOVERABLE:15-16
```

### Example 2

Prompt:

```
A payload message has been split into overlapping fragments. Each fragment gives its ORIGINAL offset (position where its first character belongs in the payload) and its content. Some fragments were lost in transit and are marked MISSING; a missing fragment contributes no characters and provides no information.

Rules:
- Fragments may overlap: where two fragments cover the same payload position they must agree on the character there. Reconcile overlaps by alignment of offsets; the payload is the longest consistent concatenation of the fragments by their offsets.
- Any payload position covered by no non-missing fragment is a GAP that cannot be reconstructed from the available data.
- The payload has a known total length of 15 characters (positions 0 .. 14).
- Overlap tolerance between adjacent fragments is at most 3 characters.

Fragments:
  - index 0, offset 0: ECGE
  - index 1, offset 6: HAA
  - index 2, offset 12: EFF
  - index 3, offset 13: FF
  - index 4, offset 14: F

Task: reconstruct the full payload by merging the fragments at their stated offsets. If the non-missing fragments cover every position of the payload consistently, the answer is that fully reconstructed payload string. If any position cannot be determined (because no fragment covers it), the answer is the error state UNRECOVERABLE: followed by the sorted list of uncovered intervals, using a-b for an interval from a to b inclusive and a single number for a lone position, intervals separated by commas. Give only the answer.
```

Answer:

```
UNRECOVERABLE:4-5,9-11
```

## Level 2

### Example 1

Prompt:

```
A payload message has been split into overlapping fragments. Each fragment gives its ORIGINAL offset (position where its first character belongs in the payload) and its content. Some fragments were lost in transit and are marked MISSING; a missing fragment contributes no characters and provides no information.

Rules:
- Fragments may overlap: where two fragments cover the same payload position they must agree on the character there. Reconcile overlaps by alignment of offsets; the payload is the longest consistent concatenation of the fragments by their offsets.
- Any payload position covered by no non-missing fragment is a GAP that cannot be reconstructed from the available data.
- The payload has a known total length of 29 characters (positions 0 .. 28).
- Overlap tolerance between adjacent fragments is at most 3 characters.

Fragments:
  - index 0, offset 0: EGCEGFBHB
  - index 1, offset 13: CAAAHD
  - index 2, offset 22: MISSING
  - index 3, offset 27: EB
  - index 4, offset 25: MISSING
  - index 5, offset 25: CFEB
  - index 6, offset 28: B

Task: reconstruct the full payload by merging the fragments at their stated offsets. If the non-missing fragments cover every position of the payload consistently, the answer is that fully reconstructed payload string. If any position cannot be determined (because no fragment covers it), the answer is the error state UNRECOVERABLE: followed by the sorted list of uncovered intervals, using a-b for an interval from a to b inclusive and a single number for a lone position, intervals separated by commas. Give only the answer.
```

Answer:

```
UNRECOVERABLE:9-12,19-24
```

### Example 2

Prompt:

```
A payload message has been split into overlapping fragments. Each fragment gives its ORIGINAL offset (position where its first character belongs in the payload) and its content. Some fragments were lost in transit and are marked MISSING; a missing fragment contributes no characters and provides no information.

Rules:
- Fragments may overlap: where two fragments cover the same payload position they must agree on the character there. Reconcile overlaps by alignment of offsets; the payload is the longest consistent concatenation of the fragments by their offsets.
- Any payload position covered by no non-missing fragment is a GAP that cannot be reconstructed from the available data.
- The payload has a known total length of 31 characters (positions 0 .. 30).
- Overlap tolerance between adjacent fragments is at most 3 characters.

Fragments:
  - index 0, offset 0: MISSING
  - index 1, offset 7: DGAAGCHAB
  - index 2, offset 18: AGDAEA
  - index 3, offset 24: HGEDHEH
  - index 4, offset 29: EH
  - index 5, offset 30: H
  - index 6, offset 25: GEDHEH

Task: reconstruct the full payload by merging the fragments at their stated offsets. If the non-missing fragments cover every position of the payload consistently, the answer is that fully reconstructed payload string. If any position cannot be determined (because no fragment covers it), the answer is the error state UNRECOVERABLE: followed by the sorted list of uncovered intervals, using a-b for an interval from a to b inclusive and a single number for a lone position, intervals separated by commas. Give only the answer.
```

Answer:

```
UNRECOVERABLE:0-6,16-17
```

## Level 5

### Example 1

Prompt:

```
A payload message has been split into overlapping fragments. Each fragment gives its ORIGINAL offset (position where its first character belongs in the payload) and its content. Some fragments were lost in transit and are marked MISSING; a missing fragment contributes no characters and provides no information.

Rules:
- Fragments may overlap: where two fragments cover the same payload position they must agree on the character there. Reconcile overlaps by alignment of offsets; the payload is the longest consistent concatenation of the fragments by their offsets.
- Any payload position covered by no non-missing fragment is a GAP that cannot be reconstructed from the available data.
- The payload has a known total length of 30 characters (positions 0 .. 29).
- Overlap tolerance between adjacent fragments is at most 3 characters.

Fragments:
  - index 0, offset 0: DGFBA
  - index 1, offset 8: MISSING
  - index 2, offset 13: MISSING
  - index 3, offset 18: DFGBG
  - index 4, offset 23: CAADDB
  - index 5, offset 29: E
  - index 6, offset 28: BE
  - index 7, offset 27: MISSING
  - index 8, offset 28: BE
  - index 9, offset 27: DBE
  - index 10, offset 29: E
  - index 11, offset 28: BE

Task: reconstruct the full payload by merging the fragments at their stated offsets. If the non-missing fragments cover every position of the payload consistently, the answer is that fully reconstructed payload string. If any position cannot be determined (because no fragment covers it), the answer is the error state UNRECOVERABLE: followed by the sorted list of uncovered intervals, using a-b for an interval from a to b inclusive and a single number for a lone position, intervals separated by commas. Give only the answer.
```

Answer:

```
UNRECOVERABLE:5-17
```

### Example 2

Prompt:

```
A payload message has been split into overlapping fragments. Each fragment gives its ORIGINAL offset (position where its first character belongs in the payload) and its content. Some fragments were lost in transit and are marked MISSING; a missing fragment contributes no characters and provides no information.

Rules:
- Fragments may overlap: where two fragments cover the same payload position they must agree on the character there. Reconcile overlaps by alignment of offsets; the payload is the longest consistent concatenation of the fragments by their offsets.
- Any payload position covered by no non-missing fragment is a GAP that cannot be reconstructed from the available data.
- The payload has a known total length of 32 characters (positions 0 .. 31).
- Overlap tolerance between adjacent fragments is at most 3 characters.

Fragments:
  - index 0, offset 0: EAABD
  - index 1, offset 7: GCC
  - index 2, offset 11: MISSING
  - index 3, offset 15: MISSING
  - index 4, offset 21: ECD
  - index 5, offset 26: ABF
  - index 6, offset 29: CDB
  - index 7, offset 27: BFCDB
  - index 8, offset 29: MISSING
  - index 9, offset 31: B
  - index 10, offset 29: MISSING
  - index 11, offset 28: FCDB

Task: reconstruct the full payload by merging the fragments at their stated offsets. If the non-missing fragments cover every position of the payload consistently, the answer is that fully reconstructed payload string. If any position cannot be determined (because no fragment covers it), the answer is the error state UNRECOVERABLE: followed by the sorted list of uncovered intervals, using a-b for an interval from a to b inclusive and a single number for a lone position, intervals separated by commas. Give only the answer.
```

Answer:

```
UNRECOVERABLE:5-6,10-20,24-25
```
