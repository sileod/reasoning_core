# Level 0

## Example 1

**Prompt:**

Each lambda (lambda: X) captures the CURRENT value of X at the moment the lambda is created; later assignments to X do not change what an already-created lambda sees.

Program:
u = 2
c = 1
h = (lambda: u)
h = (lambda: u)
h = (lambda: u)
h = (lambda: c)
h = (lambda: c)
h = (lambda: c)
h = (lambda: c)
g = (lambda: u)
u = -3
f = (lambda: c)

Call:
f()

What is the value of f()?

**Answer:** 1

## Example 2

**Prompt:**

Each lambda (lambda: X) captures the CURRENT value of X at the moment the lambda is created; later assignments to X do not change what an already-created lambda sees.

Program:
c = -9
j = -8
h = (lambda: c)
h = (lambda: c)
h = (lambda: c)
h = (lambda: c)
g = (lambda: c)
c = -5
f = (lambda: j)

Call:
f()

What is the value of f()?

**Answer:** -8

# Level 2

## Example 1

**Prompt:**

Each lambda (lambda: X) captures the CURRENT value of X at the moment the lambda is created; later assignments to X do not change what an already-created lambda sees.

Program:
r = -2
z = 1
m = 3
x = -9
g = (lambda: x)
x = 3
f = (lambda: r)
g = (lambda: x)
x = -3
f = (lambda: z)
g = (lambda: x)
x = 5
g = (lambda: m)
m = -4

Call:
g()

What is the value of g()?

**Answer:** 3

## Example 2

**Prompt:**

Each lambda (lambda: X) captures the CURRENT value of X at the moment the lambda is created; later assignments to X do not change what an already-created lambda sees.

Program:
b = 0
g = -1
e = 6
k = -3
h = (lambda: e)
h = (lambda: e)
h = (lambda: e)
h = (lambda: e)
f = (lambda: e)
h = (lambda: e)
h = (lambda: e)
h = (lambda: e)
h = (lambda: e)
f = (lambda: g)
f = (lambda: b)
f = (lambda: g)

Call:
f()

What is the value of f()?

**Answer:** -1

# Level 5

## Example 1

**Prompt:**

Each lambda (lambda: X) captures the CURRENT value of X at the moment the lambda is created; later assignments to X do not change what an already-created lambda sees.

Program:
h = 1
w = -1
i = -8
d = 4
o = 8
c = -2
v = -9
g = (lambda: w)
w = 3
f = (lambda: v)
g = (lambda: o)
o = 2
f = (lambda: i)
h = (lambda: v)
h = (lambda: v)
h = (lambda: v)
h = (lambda: v)
f = (lambda: w)
g = (lambda: h)
h = 9
g = (lambda: d)
d = -8
g = (lambda: i)
i = 9

Call:
g()

What is the value of g()?

**Answer:** -8

## Example 2

**Prompt:**

Each lambda (lambda: X) captures the CURRENT value of X at the moment the lambda is created; later assignments to X do not change what an already-created lambda sees.

Program:
f = -2
j = 7
y = 0
o = -3
g = -4
h = -9
m = -7
h = (lambda: g)
h = (lambda: g)
h = (lambda: g)
h = (lambda: g)
h = (lambda: h)
h = (lambda: h)
h = (lambda: m)
h = (lambda: m)
h = (lambda: m)
h = (lambda: m)
g = (lambda: y)
y = -8
f = (lambda: o)
f = (lambda: g)
f = (lambda: o)
h = (lambda: h)
h = (lambda: h)
h = (lambda: h)
f = (lambda: m)

Call:
f()

What is the value of f()?

**Answer:** -7

