# Samples for counterfactual_replay (P016v1)

## Level 0

**Prompt:**

Consider these quantities, computed in order so that each later quantity uses only earlier ones:
q0 = 9
q1 = -5
q2 = 2*q0 - q1 + 5
q3 = -2*q1 - 8
In the actual scenario q0 equals 9.
Now suppose that instead q0 had been 6 (instead of 9). Recompute every quantity that depends, directly or indirectly, on q0 using this new value, and keep every quantity that does not depend on q0 exactly as it originally was. What is the resulting value of q2?
The answer is the single integer value of q2 (it may be negative).

**Answer:**

22

**Prompt:**

Consider these quantities, computed in order so that each later quantity uses only earlier ones:
q0 = -8
q1 = 3
q2 = -q0 - q1 - 7
q3 = -2*q1 + 2
In the actual scenario q0 equals -8.
Now suppose that instead q0 had been 2 (instead of -8). Recompute every quantity that depends, directly or indirectly, on q0 using this new value, and keep every quantity that does not depend on q0 exactly as it originally was. What is the resulting value of q2?
The answer is the single integer value of q2 (it may be negative).

**Answer:**

-12

## Level 2

**Prompt:**

Consider these quantities, computed in order so that each later quantity uses only earlier ones:
q0 = -18
q1 = -1
q2 = -16
q3 = 3*q0 - 2
q4 = -3*q1 - q2 + 12
q5 = 2*q1 - 2*q2 - 9
q6 = 2*q3 - 17
q7 = -3*q5 - 8
q8 = -2*q4 + 3*q5 - 8
q9 = q6 - 3*q8 - 13
q10 = 3*q7 + q8 + 5
q11 = -3*q8 - 1
In the actual scenario q0 equals -18.
Now suppose that instead q0 had been -5 (instead of -18). Recompute every quantity that depends, directly or indirectly, on q0 using this new value, and keep every quantity that does not depend on q0 exactly as it originally was. What is the resulting value of q9?
The answer is the single integer value of q9 (it may be negative).

**Answer:**

-43

**Prompt:**

Consider these quantities, computed in order so that each later quantity uses only earlier ones:
q0 = -20
q1 = -8
q2 = -18
q3 = -2*q0
q4 = -2*q1 - 2*q2 + 13
q5 = -3*q1 + 3*q2 + 12
q6 = -2*q3 - 1
q7 = 2*q4 - 3*q5 - 16
q8 = q4 - q5 - 6
q9 = 2*q6 + 2*q8 - 6
q10 = -q7
q11 = -q7 - 19
In the actual scenario q0 equals -20.
Now suppose that instead q0 had been 1 (instead of -20). Recompute every quantity that depends, directly or indirectly, on q0 using this new value, and keep every quantity that does not depend on q0 exactly as it originally was. What is the resulting value of q9?
The answer is the single integer value of q9 (it may be negative).

**Answer:**

154

## Level 5

**Prompt:**

Consider these quantities, computed in order so that each later quantity uses only earlier ones:
q0 = 14
q1 = 10
q2 = -19
q3 = -39
q4 = -15
q5 = 4*q0 + 1
q6 = -3*q1 - 3*q4 - 20
q7 = -4*q4 + 19
q8 = 2*q4 + 25
q9 = 3*q1 - 2*q2 + 4*q3 - 5
q10 = -4*q5 - 32
q11 = 2*q8 - 3*q9 - 25
q12 = -q8 - 11
q13 = -4*q6 + 3*q8 - 1
q14 = 4*q6 - 20
q15 = q10 - 33
q16 = 3*q11 + 3*q12 + 33
q17 = 2*q11 + q13 - 3*q14 + 18
q18 = -2*q11 - 2*q13 + q14 - 20
q19 = 2*q12 - q13 - 3*q14 + 36
q20 = -3*q15 - 24
q21 = -2*q16 + 2*q17 - 4*q19 - 33
q22 = 2*q16 + 2*q18 - 39
q23 = 4*q16 + 2*q18 - 2*q19 + 4
q24 = -2*q18 + 4*q19 + 3
q25 = -4*q20 + 18
q26 = 3*q21 - 2*q22 - 2*q23 + 12
q27 = 3*q21 + 3*q24 + 17
q28 = -2*q22 + 3*q24 - 15
q29 = -3*q21 + 2*q23 + q24 - 18
q30 = 4*q25 - 2*q26
q31 = 2*q26 + 37
q32 = 4*q27 - 2*q28 - 26
q33 = 2*q27 + 2*q28 - 4*q29 + 39
q34 = -3*q28 + q29 + 1
In the actual scenario q0 equals 14.
Now suppose that instead q0 had been 36 (instead of 14). Recompute every quantity that depends, directly or indirectly, on q0 using this new value, and keep every quantity that does not depend on q0 exactly as it originally was. What is the resulting value of q30?
The answer is the single integer value of q30 (it may be negative).

**Answer:**

-17794

**Prompt:**

Consider these quantities, computed in order so that each later quantity uses only earlier ones:
q0 = -15
q1 = -34
q2 = -21
q3 = 17
q4 = -10
q5 = -q0 - 36
q6 = -3*q3 + 24
q7 = -q4 - 4
q8 = -q1 - 2*q2 + q4 + 32
q9 = -q1 - 4*q3 - 39
q10 = 4*q5 - 9
q11 = -4*q8 - 4
q12 = 4*q6 + 3*q7 + 4*q9 - 16
q13 = 2*q6 - 2*q9 - 17
q14 = 4*q8 + 31
q15 = 3*q10 + 36
q16 = -2*q11 + 4*q13 - 4*q14 - 5
q17 = -2*q11 + 4*q14 - 39
q18 = -4*q14 - 29
q19 = -3*q11 + 2*q12 + q13 + 37
q20 = 3*q15 + 26
q21 = 3*q17
q22 = 3*q16 - q17 - 29
q23 = q17 - 33
q24 = 3*q16 - 2*q18 + 34
q25 = 2*q20 + 37
q26 = 4*q21 - 3*q22 - 4*q23 + 29
q27 = -q23 - 2*q24 - 10
q28 = 3*q21 + q23 + 25
q29 = -q23 + 24
q30 = -4*q25 + q27 + 23
q31 = 3*q28 + 5
q32 = -2*q29 + 38
q33 = 2*q26 - q27 + 3*q29 + 6
q34 = -3*q27 + 2*q28 - 4*q29 - 9
In the actual scenario q0 equals -15.
Now suppose that instead q0 had been -8 (instead of -15). Recompute every quantity that depends, directly or indirectly, on q0 using this new value, and keep every quantity that does not depend on q0 exactly as it originally was. What is the resulting value of q30?
The answer is the single integer value of q30 (it may be negative).

**Answer:**

1771
