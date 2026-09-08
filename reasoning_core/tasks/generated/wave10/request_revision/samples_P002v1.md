# samples_P002v1

## Level 0
**Prompt:**
```
Lines 1, 6, 9 are in force initially.
Instruction 1: add line 11 (requirement R0).
Instruction 2: cancel requirement R0.
Instruction 3: reinstate requirement R0.
Instruction 4: remove line 9 (requirement R1).
After all instructions, which lines remain in force? Give the answer as a comma-separated list, or 'none' if no line remains.
```
**Answer:**
```
1,6,11
```

**Prompt:**
```
Lines 12 are in force initially.
Instruction 1: add line 3 (requirement R0).
Instruction 2: cancel requirement R0.
Instruction 3: add line 1 (requirement R1).
Instruction 4: cancel requirement R1.
After all instructions, which lines remain in force? Give the answer as a comma-separated list, or 'none' if no line remains.
```
**Answer:**
```
12
```

## Level 2
**Prompt:**
```
Lines 3 are in force initially.
Instruction 1: add line 1 (requirement R0).
Instruction 2: cancel requirement R0.
Instruction 3: add line 6 (requirement R1).
Instruction 4: remove line 3 (requirement R2).
Instruction 5: reinstate requirement R0.
Instruction 6: cancel requirement R0.
Instruction 7: cancel requirement R1.
Instruction 8: reinstate requirement R0.
After all instructions, which lines remain in force? Give the answer as a comma-separated list, or 'none' if no line remains.
```
**Answer:**
```
1
```

**Prompt:**
```
Lines 2, 9 are in force initially.
Instruction 1: add line 8 (requirement R0).
Instruction 2: remove line 8 (requirement R1).
Instruction 3: cancel requirement R1.
Instruction 4: remove line 8 (requirement R2).
Instruction 5: remove line 8 (requirement R3).
Instruction 6: reinstate requirement R1.
Instruction 7: remove line 8 (requirement R4).
Instruction 8: add line 3 (requirement R5).
After all instructions, which lines remain in force? Give the answer as a comma-separated list, or 'none' if no line remains.
```
**Answer:**
```
2,3,9
```

## Level 5
**Prompt:**
```
Lines 2, 4, 7 are in force initially.
Instruction 1: remove line 7 (requirement R0).
Instruction 2: add line 1 (requirement R1).
Instruction 3: add line 10 (requirement R2).
Instruction 4: remove line 2 (requirement R3).
Instruction 5: remove line 1 (requirement R4).
Instruction 6: add line 8 (requirement R5).
Instruction 7: cancel requirement R5.
Instruction 8: cancel requirement R3.
Instruction 9: cancel requirement R0.
Instruction 10: reinstate requirement R0.
Instruction 11: remove line 10 (requirement R6).
Instruction 12: add line 6 (requirement R7).
Instruction 13: remove line 1 (requirement R8).
Instruction 14: reinstate requirement R3.
After all instructions, which lines remain in force? Give the answer as a comma-separated list, or 'none' if no line remains.
```
**Answer:**
```
4,6
```

**Prompt:**
```
Lines 4 are in force initially.
Instruction 1: remove line 4 (requirement R0).
Instruction 2: cancel requirement R0.
Instruction 3: add line 2 (requirement R1).
Instruction 4: cancel requirement R1.
Instruction 5: reinstate requirement R0.
Instruction 6: add line 8 (requirement R2).
Instruction 7: reinstate requirement R1.
Instruction 8: cancel requirement R2.
Instruction 9: cancel requirement R0.
Instruction 10: reinstate requirement R2.
Instruction 11: reinstate requirement R0.
Instruction 12: remove line 4 (requirement R3).
Instruction 13: add line 5 (requirement R4).
Instruction 14: add line 1 (requirement R5).
After all instructions, which lines remain in force? Give the answer as a comma-separated list, or 'none' if no line remains.
```
**Answer:**
```
1,2,5,8
```
