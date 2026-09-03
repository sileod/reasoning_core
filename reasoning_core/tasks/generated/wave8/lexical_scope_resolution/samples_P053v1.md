## Level 0
### Example 1
**Prompt:**
Consider the following program with nested scopes. Inner scopes see declarations from their enclosing scopes, and a name binds to the nearest declaration in the lexical scope chain (lexical scoping).

Program:
scope(root):
  m = 100
  i = 101
scope(inside scope 1):
  d = 102
  h = 103
  v = 104
  b = 105
scope(inside scope 2):
  n = 106
  x = 107
  o = 108

What does x resolve to inside scope 2?

Give the answer as the binding in the form 'name = value'.

**Answer:**
x = 107

### Example 2
**Prompt:**
Consider the following program with nested scopes. Inner scopes see declarations from their enclosing scopes, and a name binds to the nearest declaration in the lexical scope chain (lexical scoping).

Program:
scope(root):
  y = 100
scope(inside scope 1):
  m = 101
  l = 102
  q = 103
scope(inside scope 2):
  l = 104

What does l resolve to inside scope 2?

Give the answer as the binding in the form 'name = value'.

**Answer:**
l = 104

## Level 2
### Example 1
**Prompt:**
Consider the following program with nested scopes. Inner scopes see declarations from their enclosing scopes, and a name binds to the nearest declaration in the lexical scope chain (lexical scoping).

Program:
scope(root):
  t = 100
  c = 101
  e = 102
  k = 103
  g = 104
scope(inside scope 1):
  u = 105
  e = 106
  f = 107
  g = 108
  v = 109
  q = 110
scope(inside scope 2):
  g = 111
  c = 112
  t = 113
  p = 114
  o = 115
scope(inside scope 3):
  d = 116
scope(inside scope 4):
  d = 117
  n = 118
  s = 119
  k = 120

What does d resolve to inside scope 4?

Give the answer as the binding in the form 'name = value'.

**Answer:**
d = 117

### Example 2
**Prompt:**
Consider the following program with nested scopes. Inner scopes see declarations from their enclosing scopes, and a name binds to the nearest declaration in the lexical scope chain (lexical scoping).

Program:
scope(root):
  x = 100
scope(inside scope 1):
  s = 101
scope(inside scope 2):
  v = 102
  h = 103
  q = 104
scope(inside scope 3):
  w = 105
scope(inside scope 4):
  n = 106
  l = 107
  y = 108
  i = 109

What does y resolve to inside scope 4?

Give the answer as the binding in the form 'name = value'.

**Answer:**
y = 108

## Level 5
### Example 1
**Prompt:**
Consider the following program with nested scopes. Inner scopes see declarations from their enclosing scopes, and a name binds to the nearest declaration in the lexical scope chain (lexical scoping).

Program:
scope(root):
  c = 100
  r = 101
  d = 102
  s = 103
scope(inside scope 1):
  k = 104
  h = 105
scope(inside scope 2):
  c = 106
  l = 107
  r = 108
scope(inside scope 3):
  d = 109
  i = 110
  g = 111
  b = 112
  f = 113
  v = 114
scope(inside scope 4):
  f = 115
  l = 116
  g = 117
scope(inside scope 5):
  q = 118
  n = 119
  w = 120
  j = 121
  z = 122
  i = 123
scope(inside scope 6):
  e = 124
  q = 125
  v = 126
scope(inside scope 7):
  h = 127
  j = 128
  v = 129

What does v resolve to inside scope 7?

Give the answer as the binding in the form 'name = value'.

**Answer:**
v = 129

### Example 2
**Prompt:**
Consider the following program with nested scopes. Inner scopes see declarations from their enclosing scopes, and a name binds to the nearest declaration in the lexical scope chain (lexical scoping).

Program:
scope(root):
  x = 100
scope(inside scope 1):
  t = 101
  e = 102
  r = 103
  z = 104
  o = 105
  n = 106
scope(inside scope 2):
  o = 107
  t = 108
  e = 109
  y = 110
  u = 111
  c = 112
  j = 113
  p = 114
scope(inside scope 3):
  k = 115
scope(inside scope 4):
  k = 116
  m = 117
  o = 118
  f = 119
scope(inside scope 5):
  y = 120
scope(inside scope 6):
  a = 121
  z = 122
  y = 123
  s = 124
scope(inside scope 7):
  b = 125
  a = 126

What does a resolve to inside scope 7?

Give the answer as the binding in the form 'name = value'.

**Answer:**
a = 126

