# Spreadsheet Formula Dependency - samples

## Level 0

**Prompt:**

```
Below is a spreadsheet. Each line gives a cell name and either a literal number or a formula. A formula's value is computed from the current values of the cells it references; references always point at cells listed above them.

Cells:
A1: 14
A2: max(A1, 6)
A3: max(A2, A1)
A4: max(A3)
A5: A4 + 8
A6: max(A4, A3, A5, A2)
A7: 4
A8: 14
A9: min(A6, 15)
A10: max(A9, A2, A4, A1)
Afterward the following cells are changed: A8 set to 14, A7 set to 4.
Queries: the value of cell A4.

The answer is a single integer.
```

**Answer:** 14

**Prompt:**

```
Below is a spreadsheet. Each line gives a cell name and either a literal number or a formula. A formula's value is computed from the current values of the cells it references; references always point at cells listed above them.

Cells:
A1: 15
A2: 8
A3: max(A1)
A4: 13
A5: 8
A6: 10
A7: 4
A8: min(A2, A1)
A9: A3 + A6
A10: A1 + A9 + 15
Afterward the following cells are changed: A6 set to 10, A7 set to 4.
Queries: the value of cell A10.

The answer is a single integer.
```

**Answer:** 55

## Level 2

**Prompt:**

```
Below is a spreadsheet. Each line gives a cell name and either a literal number or a formula. A formula's value is computed from the current values of the cells it references; references always point at cells listed above them.

Cells:
A1: 22
A2: 17
A3: A1 + A2
A4: max(A1, A2, A3)
A5: A3 + A1 + A2
A6: min(A4, A1, A5, A3)
A7: min(A1, A3, A4, A2, 20)
A8: 23
A9: min(A1, A3)
A10: max(A7:A9)
A11: min(A6, A3, A10)
A12: min(A1:A2)
A13: max(A3)
A14: A12 + A11 + A6 + A13 + 19
A15: 6
A16: 16
Afterward the following cells are changed: A16 set to 16, A2 set to 17, A8 set to 23.
Queries: the value of cell A6.

The answer is a single integer.
```

**Answer:** 22

**Prompt:**

```
Below is a spreadsheet. Each line gives a cell name and either a literal number or a formula. A formula's value is computed from the current values of the cells it references; references always point at cells listed above them.

Cells:
A1: 17
A2: A1
A3: 4
A4: 14
A5: max(A1, A3, A4, A2)
A6: 12
A7: min(A2, A5, A6, A1)
A8: min(A5, 20)
A9: sum(A5:A7)
A10: 6
A11: 15
A12: min(A1, A7, A9, A3, 18)
A13: 8
A14: 22
A15: 23
A16: 10
Afterward the following cells are changed: A3 set to 4, A13 set to 8, A14 set to 22.
Queries: the value of cell A7.

The answer is a single integer.
```

**Answer:** 12

## Level 5

**Prompt:**

```
Below is a spreadsheet. Each line gives a cell name and either a literal number or a formula. A formula's value is computed from the current values of the cells it references; references always point at cells listed above them.

Cells:
A1: 16
A2: 21
A3: A1
A4: min(A2, A1)
A5: A3
A6: min(A1:A4)
A7: max(A1:A4)
A8: max(A3:A4)
A9: max(A1, A7, A5, A6)
A10: A9 + A5 + A7 + A1 + A3 + 17
A11: 31
A12: max(A11, A9, A2, 17)
A13: max(A7)
A14: 33
A15: 5
A16: 8
A17: A1
A18: 6
A19: min(A14:A16)
A20: sum(A17:A18)
A21: 31
A22: A4 + A9 + A5 + A11 + 3
A23: 9
A24: 11
A25: 16
Afterward the following cells are changed: A23 set to 9, A2 set to 21, A16 set to 8.
Queries: the value of cell A6.

The answer is a single integer.
```

**Answer:** 16

**Prompt:**

```
Below is a spreadsheet. Each line gives a cell name and either a literal number or a formula. A formula's value is computed from the current values of the cells it references; references always point at cells listed above them.

Cells:
A1: 33
A2: 11
A3: max(A2)
A4: min(A3, 14)
A5: sum(A2:A3)
A6: max(A4, 11)
A7: min(A4, A5, A1, A2, A6)
A8: min(A4, A1, A7, A5)
A9: 29
A10: max(A7:A8)
A11: min(A2, A4, 6)
A12: A4
A13: A7 + A1
A14: sum(A1:A2)
A15: 32
A16: A11
A17: max(A11, A7, A14, A10, A13, 32)
A18: min(A11)
A19: sum(A3:A6)
A20: 3
A21: max(A11, A13, A14, A1)
A22: max(A13, A12, A7, A17, 18)
A23: min(A8, 33)
A24: max(A3, A17)
A25: min(A12, A21)
Afterward the following cells are changed: A20 set to 3, A1 set to 33, A9 set to 29.
Queries: the value of cell A4.

The answer is a single integer.
```

**Answer:** 11
