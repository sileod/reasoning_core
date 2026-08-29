# Level 0

## Example 1

### Prompt

Factor c is independently true with probability 0.6.
Factor e is independently true with probability 0.4.
The observation holds exactly when not (((factor c is false and factor c is false) and not (factor e is false))). We observe it.
Hidden factor values:
1. c
0. not c
3. e
2. not e
Two complete explanations are under consideration:
Explanation A: c=true, e=false
Explanation B: c=false, e=false
Which of these two complete explanations is more probable?
Answer with the space-separated indexes (from the options above) of the winning explanation's chosen values.

### Answer

1 2

## Example 2

### Prompt

Factor a is independently true with probability 0.4.
Factor b is independently true with probability 0.7.
Factor d is independently true with probability 0.6.
The observation holds exactly when not (((factor b or factor a) or (factor d and factor b))). We observe it.
Hidden factor values:
0. a
1. not a
2. b
3. not b
4. d
5. not d
Two complete explanations are under consideration:
Explanation A: a=false, b=false, d=true
Explanation B: a=false, b=false, d=false
Which of these two complete explanations is more probable?
Answer with the space-separated indexes (from the options above) of the winning explanation's chosen values.

### Answer

1 3 4

# Level 2

## Example 1

### Prompt

Factor a is independently true with probability 0.2.
Factor b is independently true with probability 0.9.
Factor d is independently true with probability 0.7.
Factor e is independently true with probability 0.4.
The observation holds exactly when not ((not ((if factor d, then (if factor a, then factor d; otherwise factor a); otherwise (factor a or factor e))) or factor b)). We observe it.
Hidden factor values:
1. a
0. not a
3. b
2. not b
5. d
4. not d
7. e
6. not e
Two complete explanations are under consideration:
Explanation A: a=false, b=false, d=false, e=true
Explanation B: a=true, b=false, d=true, e=false
Which of these two complete explanations is more probable?
Answer with the space-separated indexes (from the options above) of the winning explanation's chosen values.

### Answer

0 2 4 7

## Example 2

### Prompt

Factor b is independently true with probability 0.6.
Factor c is independently true with probability 0.8.
Factor d is independently true with probability 0.4.
Factor e is independently true with probability 0.7.
Factor f is independently true with probability 0.6.
The observation holds exactly when ((factor c and (if factor d, then factor f is false; otherwise (factor c or (factor b or factor d)))) and (factor e is false or factor d)). We observe it.
Hidden factor values:
0. b
1. not b
2. c
3. not c
5. d
4. not d
6. e
7. not e
9. f
8. not f
Two complete explanations are under consideration:
Explanation A: b=true, c=true, d=true, e=true, f=false
Explanation B: b=true, c=true, d=false, e=false, f=true
Which of these two complete explanations is more probable?
Answer with the space-separated indexes (from the options above) of the winning explanation's chosen values.

### Answer

0 2 5 6 8

# Level 5

## Example 1

### Prompt

Factor a is independently true with probability 0.8.
Factor b is independently true with probability 0.4.
Factor c is independently true with probability 0.3.
Factor d is independently true with probability 0.8.
Factor e is independently true with probability 0.9.
Factor f is independently true with probability 0.9.
The observation holds exactly when ((if factor d, then (not (not (factor e is false)) and factor a); otherwise factor c) and ((factor b or factor f) and (if factor c, then ((factor a and factor a) and (if factor e, then (factor f or (factor c or factor f)); otherwise factor c is false)); otherwise ((factor f is false or factor a) and factor d)))). We observe it.
Hidden factor values:
1. a
0. not a
2. b
3. not b
5. c
4. not c
7. d
6. not d
9. e
8. not e
11. f
10. not f
Two complete explanations are under consideration:
Explanation A: a=true, b=false, c=false, d=true, e=false, f=true
Explanation B: a=true, b=false, c=true, d=false, e=true, f=true
Which of these two complete explanations is more probable?
Answer with the space-separated indexes (from the options above) of the winning explanation's chosen values.

### Answer

1 3 4 7 8 11

## Example 2

### Prompt

Factor b is independently true with probability 0.8.
Factor c is independently true with probability 0.4.
Factor d is independently true with probability 0.8.
Factor e is independently true with probability 0.4.
Factor f is independently true with probability 0.3.
The observation holds exactly when ((factor f or (factor c and ((factor d and factor e) and (if (factor d is false and factor c), then factor c; otherwise (factor e or factor f))))) and (if ((factor e and factor c is false) and factor d is false), then factor f is false; otherwise factor b)). We observe it.
Hidden factor values:
1. b
0. not b
2. c
3. not c
5. d
4. not d
7. e
6. not e
9. f
8. not f
Two complete explanations are under consideration:
Explanation A: b=true, c=true, d=true, e=true, f=false
Explanation B: b=true, c=false, d=true, e=false, f=true
Which of these two complete explanations is more probable?
Answer with the space-separated indexes (from the options above) of the winning explanation's chosen values.

### Answer

1 2 5 7 8
