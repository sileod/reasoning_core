# Level 0

## Example 1

### Prompt

Program:
Variables a, b and c are initialized to 0. The program below is analyzed with interval abstract interpretation: every expression yields an integer interval [lo, hi].
An assignment x := <expr> replaces x's interval by the interval of <expr>.
A join 'x := any{ p1 , p2 , ... }' puts x into the least upper bound (the interval spanning all of those options together with x's previous interval).
A loop that repeatedly does x := x + d is resolved by standard widening: a positive increment raises x's upper bound to +infinity and a negative increment lowers its lower bound to -infinity.

b := any{ a , b , b }
c := b + 4
c := any{ 4 , b , c }
a := any{ a , -1 , b--5 , a }

Compute the abstract interval of variable 'a' after the whole program has been analyzed.

The answer is the interval in the form (lo, hi), using -inf or +inf for unbounded ends, for example (-inf, 5) or (3, +inf).

### Answer

(-1, 5)

## Example 2

### Prompt

Program:
Variables a, b and c are initialized to 0. The program below is analyzed with interval abstract interpretation: every expression yields an integer interval [lo, hi].
An assignment x := <expr> replaces x's interval by the interval of <expr>.
A join 'x := any{ p1 , p2 , ... }' puts x into the least upper bound (the interval spanning all of those options together with x's previous interval).
A loop that repeatedly does x := x + d is resolved by standard widening: a positive increment raises x's upper bound to +infinity and a negative increment lowers its lower bound to -infinity.

b := a + 1
b := any{ 5 , b , b }
a := c + 4
loop: c := c - 1   (widened)

Compute the abstract interval of variable 'a' after the whole program has been analyzed.

The answer is the interval in the form (lo, hi), using -inf or +inf for unbounded ends, for example (-inf, 5) or (3, +inf).

### Answer

(4, 4)



# Level 2

## Example 1

### Prompt

Program:
Variables a, b and c are initialized to 0. The program below is analyzed with interval abstract interpretation: every expression yields an integer interval [lo, hi].
An assignment x := <expr> replaces x's interval by the interval of <expr>.
A join 'x := any{ p1 , p2 , ... }' puts x into the least upper bound (the interval spanning all of those options together with x's previous interval).
A loop that repeatedly does x := x + d is resolved by standard widening: a positive increment raises x's upper bound to +infinity and a negative increment lowers its lower bound to -infinity.

loop: b := b + 3   (widened)
b := b + -7
c := a + -8
loop: b := b - 6   (widened)
b := any{ 9 , -1 , 6 , b }
a := any{ b+5 , 3 , a }
b := any{ 4 , 5 , b }
loop: c := c + 2   (widened)

Compute the abstract interval of variable 'c' after the whole program has been analyzed.

The answer is the interval in the form (lo, hi), using -inf or +inf for unbounded ends, for example (-inf, 5) or (3, +inf).

### Answer

(-8, +inf)

## Example 2

### Prompt

Program:
Variables a, b and c are initialized to 0. The program below is analyzed with interval abstract interpretation: every expression yields an integer interval [lo, hi].
An assignment x := <expr> replaces x's interval by the interval of <expr>.
A join 'x := any{ p1 , p2 , ... }' puts x into the least upper bound (the interval spanning all of those options together with x's previous interval).
A loop that repeatedly does x := x + d is resolved by standard widening: a positive increment raises x's upper bound to +infinity and a negative increment lowers its lower bound to -infinity.

c := a - -2
a := b + -8
c := c + -2
a := c + -4
b := c - 7
b := any{ b , c-6 , b }
a := any{ b-1 , a--5 , 9 , a }
b := any{ 4 , c--9 , b }

Compute the abstract interval of variable 'a' after the whole program has been analyzed.

The answer is the interval in the form (lo, hi), using -inf or +inf for unbounded ends, for example (-inf, 5) or (3, +inf).

### Answer

(-8, 9)



# Level 5

## Example 1

### Prompt

Program:
Variables a, b and c are initialized to 0. The program below is analyzed with interval abstract interpretation: every expression yields an integer interval [lo, hi].
An assignment x := <expr> replaces x's interval by the interval of <expr>.
A join 'x := any{ p1 , p2 , ... }' puts x into the least upper bound (the interval spanning all of those options together with x's previous interval).
A loop that repeatedly does x := x + d is resolved by standard widening: a positive increment raises x's upper bound to +infinity and a negative increment lowers its lower bound to -infinity.

b := any{ -16 , 3 , a , b }
b := b - -9
b := c - -1
b := any{ 10 , b-1 , a+-11 , b }
b := a - -9
b := any{ -1 , 9 , -1 , b }
c := c + -15
c := a + 13
c := c + -11
b := b + -4
a := any{ b--14 , b-8 , c , a }
b := any{ b , b , a , b+-13 , b }
c := c - 15
c := a - 15

Compute the abstract interval of variable 'a' after the whole program has been analyzed.

The answer is the interval in the form (lo, hi), using -inf or +inf for unbounded ends, for example (-inf, 5) or (3, +inf).

### Answer

(-13, 19)

## Example 2

### Prompt

Program:
Variables a, b and c are initialized to 0. The program below is analyzed with interval abstract interpretation: every expression yields an integer interval [lo, hi].
An assignment x := <expr> replaces x's interval by the interval of <expr>.
A join 'x := any{ p1 , p2 , ... }' puts x into the least upper bound (the interval spanning all of those options together with x's previous interval).
A loop that repeatedly does x := x + d is resolved by standard widening: a positive increment raises x's upper bound to +infinity and a negative increment lowers its lower bound to -infinity.

c := c + -1
loop: c := c + 10   (widened)
c := any{ c , a , c }
c := c - 3
a := b + -14
a := b + 5
c := any{ 6 , a , c }
loop: c := c + 16   (widened)
b := a + -8
b := a + 3
b := c + -13
a := c + -11
b := a - -11
b := b + 5

Compute the abstract interval of variable 'c' after the whole program has been analyzed.

The answer is the interval in the form (lo, hi), using -inf or +inf for unbounded ends, for example (-inf, 5) or (3, +inf).

### Answer

(-4, +inf)


