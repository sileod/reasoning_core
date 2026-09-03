## Level 0

### Example 1

Prompt:
```
A spreadsheet has 2 rows and 2 columns.
A1 = 14
A2 = 19 * B1
B1 = A1 - 5
B2 = B1 * 4

What is the value of B2?
```

Answer: 36

### Example 2

Prompt:
```
A spreadsheet has 2 rows and 2 columns.
A1 = 8
A2 = A1 - 4
B1 = A1 * 11
B2 = SUM(B1)

What is the value of B2?
```

Answer: 88

## Level 2

### Example 1

Prompt:
```
A spreadsheet has 4 rows and 4 columns.
A1 = 35
A2 = SUM(C1)
A3 = A1 - B1
A4 = B3 + D3
B1 = SUM(A1)
B2 = A2 - 4
B3 = SUM(B2)
B4 = SUM(A3)
C1 = A1 * A1
C2 = SUM(A1,B1,C1,D1)
C3 = A3 - B2
C4 = SUM(D1)
D1 = A1 - 26
D2 = C2 - 25
D3 = SUM(D2)
D4 = C3 - D2

What is the value of D4?
```

Answer: -2500

### Example 2

Prompt:
```
A spreadsheet has 4 rows and 4 columns.
A1 = 36
A2 = C1 + 28
A3 = A2 + B1
A4 = A3 + 11
B1 = SUM(A1)
B2 = A2 + 45
B3 = SUM(A2,B1,C1,D1)
B4 = 40 - A4
C1 = 34 + B1
C2 = SUM(B2)
C3 = D2 + A1
C4 = SUM(B3)
D1 = SUM(A1)
D2 = SUM(D1)
D3 = C3 - C2
D4 = SUM(B1,B2,D2,D3)

What is the value of D4?
```

Answer: 144

## Level 5

### Example 1

Prompt:
```
A spreadsheet has 7 rows and 7 columns.
A1 = 82
A2 = G1 + E1
A3 = SUM(E1,E2)
A4 = E1 - C3
A5 = E2 * D2
A6 = E5 - 89
A7 = SUM(B6)
B1 = SUM(A1)
B2 = SUM(B1)
B3 = F1 + 87
B4 = G1 - C2
B5 = SUM(C1)
B6 = D2 + G5
B7 = D6 * D6
C1 = A1 + 16
C2 = SUM(A2,B2,C1,D1,E1,G1)
C3 = E2 * B3
C4 = SUM(B1,B2)
C5 = F4 - 89
C6 = F2 + 66
C7 = F1 * 14
D1 = B1 - B1
D2 = C2 * F1
D3 = B3 + G1
D4 = 81 - E2
D5 = 9 - A2
D6 = SUM(C4)
D7 = SUM(C6,D1,G1)
E1 = SUM(A1,B1,C1,D1)
E2 = A2 + C2
E3 = SUM(C1,F2)
E4 = B1 - 92
E5 = A3 - C2
E6 = A3 * F5
E7 = G5 - D4
F1 = B1 - B1
F2 = 50 * G1
F3 = F2 * C3
F4 = A2 - G3
F5 = D2 * B4
F6 = E1 + C5
F7 = 12 - D4
G1 = C1 - D1
G2 = D1 * 55
G3 = SUM(A2,B3,F2,F3)
G4 = 17 + B2
G5 = SUM(C5)
G6 = D5 + A5
G7 = SUM(C3)

What is the value of G7?
```

Answer: 109620

### Example 2

Prompt:
```
A spreadsheet has 7 rows and 7 columns.
A1 = 92
A2 = SUM(E1,F1)
A3 = SUM(A1,B2,C2,E1,F1,G2)
A4 = D2 * 82
A5 = SUM(B2,F1)
A6 = 78 + F1
A7 = 81 - D2
B1 = A1 + A1
B2 = SUM(A1,D1)
B3 = 19 - G2
B4 = SUM(A1,F3)
B5 = SUM(A2,B1,B4,C2,D1,F1)
B6 = SUM(F1)
B7 = C2 * A5
C1 = SUM(A1,B1)
C2 = E1 - C1
C3 = B2 - 14
C4 = SUM(G1)
C5 = A4 - 44
C6 = SUM(B1,B4,C3,D1,D5,E2,E4,F5,G2)
C7 = SUM(F1)
D1 = C1 * B1
D2 = SUM(A2,B2)
D3 = SUM(A1,B1,E1,F2)
D4 = F1 + 28
D5 = B3 + A2
D6 = G4 + D5
D7 = 77 - A3
E1 = C1 - A1
E2 = SUM(B1)
E3 = F2 - 23
E4 = SUM(A4,F3)
E5 = F3 * 67
E6 = 89 + E1
E7 = A1 * D1
F1 = SUM(B1)
F2 = SUM(A1,G1)
F3 = 28 * B3
F4 = E3 - G3
F5 = SUM(A4,C3,C5)
F6 = A2 + 77
F7 = C7 + C5
G1 = F1 + 63
G2 = SUM(G1)
G3 = F3 * 5
G4 = G1 + 80
G5 = SUM(G3)
G6 = B6 - F3
G7 = 18 - B7

What is the value of G7?
```

Answer: 4697538
