# samples_P055v1

## Level 0

### Example 1

**Prompt:**

Under three-valued logic with True, False, and Unknown, a network of gates holds a truth value per named input and gate. Evaluate bottom-up using the rules below.

Rules: Each of True, False, and Unknown is a valid truth value. not True = False, not False = True, not Unknown = Unknown; and returns False if either operand is False, True only if both are True, otherwise Unknown; or returns True if either operand is True, False only if both are False, otherwise Unknown.

in0 = False
in1 = False
in2 = Unknown
g0 = not in2
g1 = in1 and in1
g2 = g0 or in2
g3 = g1 or in0
g4 = g3 and (False)
g5 = g3 and (False)
g6 = g3 and (False)

Report the propagated value of every gate g0, g1, ... in order, as a semicolon-separated list e.g. True; Unknown; False. The answer is exactly that list.

**Answer:**

Unknown; False; Unknown; False; False; False; False

### Example 2

**Prompt:**

Under three-valued logic with True, False, and Unknown, a network of gates holds a truth value per named input and gate. Evaluate bottom-up using the rules below.

Rules: Each of True, False, and Unknown is a valid truth value. not True = False, not False = True, not Unknown = Unknown; and returns False if either operand is False, True only if both are True, otherwise Unknown; or returns True if either operand is True, False only if both are False, otherwise Unknown.

in0 = Unknown
in1 = Unknown
in2 = True
g0 = in2 and in2
g1 = not in2
g2 = not in1
g3 = g1 or in0

Report the propagated value of every gate g0, g1, ... in order, as a semicolon-separated list e.g. True; Unknown; False. The answer is exactly that list.

**Answer:**

True; False; Unknown; Unknown

## Level 2

### Example 1

**Prompt:**

Under three-valued logic with True, False, and Unknown, a network of gates holds a truth value per named input and gate. Evaluate bottom-up using the rules below.

Rules: Each of True, False, and Unknown is a valid truth value. not True = False, not False = True, not Unknown = Unknown; and returns False if either operand is False, True only if both are True, otherwise Unknown; or returns True if either operand is True, False only if both are False, otherwise Unknown.

in0 = False
in1 = False
in2 = True
in3 = Unknown
in4 = False
g0 = in4 or in0
g1 = in4 or in2
g2 = in1 or in1
g3 = not in2
g4 = in2 and in0
g5 = not in0
g6 = in3 and in1
g7 = not g0

Report the propagated value of every gate g0, g1, ... in order, as a semicolon-separated list e.g. True; Unknown; False. The answer is exactly that list.

**Answer:**

False; True; False; False; False; True; False; True

### Example 2

**Prompt:**

Under three-valued logic with True, False, and Unknown, a network of gates holds a truth value per named input and gate. Evaluate bottom-up using the rules below.

Rules: Each of True, False, and Unknown is a valid truth value. not True = False, not False = True, not Unknown = Unknown; and returns False if either operand is False, True only if both are True, otherwise Unknown; or returns True if either operand is True, False only if both are False, otherwise Unknown.

in0 = False
in1 = True
in2 = Unknown
in3 = Unknown
in4 = False
g0 = in4 or in4
g1 = not g0
g2 = in2 or in2
g3 = in0 or in2
g4 = g1 or g1
g5 = in4 and g3
g6 = in1 or in0
g7 = g6 and g6

Report the propagated value of every gate g0, g1, ... in order, as a semicolon-separated list e.g. True; Unknown; False. The answer is exactly that list.

**Answer:**

False; True; Unknown; Unknown; True; False; True; True

## Level 5

### Example 1

**Prompt:**

Under three-valued logic with True, False, and Unknown, a network of gates holds a truth value per named input and gate. Evaluate bottom-up using the rules below.

Rules: Each of True, False, and Unknown is a valid truth value. not True = False, not False = True, not Unknown = Unknown; and returns False if either operand is False, True only if both are True, otherwise Unknown; or returns True if either operand is True, False only if both are False, otherwise Unknown.

in0 = False
in1 = Unknown
in2 = False
in3 = False
in4 = Unknown
in5 = True
in6 = False
in7 = False
g0 = in0 and in5
g1 = in0 or g0
g2 = not in0
g3 = in2 and in2
g4 = g3 and in3
g5 = in7 and g1
g6 = not in3
g7 = in0 and g1
g8 = not in7
g9 = in7 and g7
g10 = g3 and g8
g11 = not in4
g12 = not g4
g13 = in6 or g6
g14 = g13 and (True)
g15 = g13 and (True)
g16 = g13 and (True)
g17 = g13 and (True)
g18 = g13 and (True)
g19 = g13 and (True)
g20 = g13 and (True)
g21 = g13 and (True)

Report the propagated value of every gate g0, g1, ... in order, as a semicolon-separated list e.g. True; Unknown; False. The answer is exactly that list.

**Answer:**

False; False; True; False; False; False; True; False; True; False; False; Unknown; True; True; True; True; True; True; True; True; True; True

### Example 2

**Prompt:**

Under three-valued logic with True, False, and Unknown, a network of gates holds a truth value per named input and gate. Evaluate bottom-up using the rules below.

Rules: Each of True, False, and Unknown is a valid truth value. not True = False, not False = True, not Unknown = Unknown; and returns False if either operand is False, True only if both are True, otherwise Unknown; or returns True if either operand is True, False only if both are False, otherwise Unknown.

in0 = True
in1 = False
in2 = Unknown
in3 = True
in4 = Unknown
in5 = True
in6 = True
in7 = False
g0 = in3 and in3
g1 = not in4
g2 = in3 or in6
g3 = g0 or in0
g4 = not in3
g5 = not in6
g6 = in1 or g1
g7 = in2 or g6
g8 = in1 or g0
g9 = in3 and in2
g10 = not in3
g11 = not g6
g12 = g6 or g8
g13 = g3 or in1
g14 = g13 and (not (False))
g15 = g13 and (not (False))
g16 = g13 and (not (False))
g17 = g13 and (not (False))
g18 = g13 and (not (False))
g19 = g13 and (not (False))
g20 = g13 and (not (False))
g21 = g13 and (not (False))

Report the propagated value of every gate g0, g1, ... in order, as a semicolon-separated list e.g. True; Unknown; False. The answer is exactly that list.

**Answer:**

True; Unknown; True; True; False; False; Unknown; Unknown; True; Unknown; False; Unknown; True; True; True; True; True; True; True; True; True; True
