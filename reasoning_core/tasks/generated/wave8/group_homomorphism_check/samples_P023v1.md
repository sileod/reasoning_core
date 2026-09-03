# Samples for group_homomorphism_check

## Level 0

The domain is the cyclic group of order 6 under addition mod 6; its Cayley table is:
0 1 2 3 4 5
1 2 3 4 5 0
2 3 4 5 0 1
3 4 5 0 1 2
4 5 0 1 2 3
5 0 1 2 3 4
The codomain is the cyclic group of order 6 under addition mod 6; its Cayley table is:
0 1 2 3 4 5
1 2 3 4 5 0
2 3 4 5 0 1
3 4 5 0 1 2
4 5 0 1 2 3
5 0 1 2 3 4
The candidate map f sends each element of the domain to an element of the codomain, listed as f(0) f(1) ... f(n-1):
f = [0 4 2 0 4 2]
For every ordered pair (x, y) of domain elements the map is a homomorphism exactly when f(x·y) = f(x)·f(y). Count the number of ordered pairs (x, y) for which this condition fails (this count is 0 if and only if f is a homomorphism).
The answer is the integer count.

Answer:
0

The domain is the cyclic group of order 6 under addition mod 6; its Cayley table is:
0 1 2 3 4 5
1 2 3 4 5 0
2 3 4 5 0 1
3 4 5 0 1 2
4 5 0 1 2 3
5 0 1 2 3 4
The codomain is the cyclic group of order 6 under addition mod 6; its Cayley table is:
0 1 2 3 4 5
1 2 3 4 5 0
2 3 4 5 0 1
3 4 5 0 1 2
4 5 0 1 2 3
5 0 1 2 3 4
The candidate map f sends each element of the domain to an element of the codomain, listed as f(0) f(1) ... f(n-1):
f = [0 2 4 0 3 4]
For every ordered pair (x, y) of domain elements the map is a homomorphism exactly when f(x·y) = f(x)·f(y). Count the number of ordered pairs (x, y) for which this condition fails (this count is 0 if and only if f is a homomorphism).
The answer is the integer count.

Answer:
13

## Level 2

The domain is the cyclic group of order 12 under addition mod 12; its Cayley table is:
0 1 2 3 4 5 6 7 8 9 10 11
1 2 3 4 5 6 7 8 9 10 11 0
2 3 4 5 6 7 8 9 10 11 0 1
3 4 5 6 7 8 9 10 11 0 1 2
4 5 6 7 8 9 10 11 0 1 2 3
5 6 7 8 9 10 11 0 1 2 3 4
6 7 8 9 10 11 0 1 2 3 4 5
7 8 9 10 11 0 1 2 3 4 5 6
8 9 10 11 0 1 2 3 4 5 6 7
9 10 11 0 1 2 3 4 5 6 7 8
10 11 0 1 2 3 4 5 6 7 8 9
11 0 1 2 3 4 5 6 7 8 9 10
The codomain is the cyclic group of order 12 under addition mod 12; its Cayley table is:
0 1 2 3 4 5 6 7 8 9 10 11
1 2 3 4 5 6 7 8 9 10 11 0
2 3 4 5 6 7 8 9 10 11 0 1
3 4 5 6 7 8 9 10 11 0 1 2
4 5 6 7 8 9 10 11 0 1 2 3
5 6 7 8 9 10 11 0 1 2 3 4
6 7 8 9 10 11 0 1 2 3 4 5
7 8 9 10 11 0 1 2 3 4 5 6
8 9 10 11 0 1 2 3 4 5 6 7
9 10 11 0 1 2 3 4 5 6 7 8
10 11 0 1 2 3 4 5 6 7 8 9
11 0 1 2 3 4 5 6 7 8 9 10
The candidate map f sends each element of the domain to an element of the codomain, listed as f(0) f(1) ... f(n-1):
f = [0 4 10 1 11 1 6 4 4 6 9 7]
For every ordered pair (x, y) of domain elements the map is a homomorphism exactly when f(x·y) = f(x)·f(y). Count the number of ordered pairs (x, y) for which this condition fails (this count is 0 if and only if f is a homomorphism).
The answer is the integer count.

Answer:
110

The domain is the cyclic group of order 12 under addition mod 12; its Cayley table is:
0 1 2 3 4 5 6 7 8 9 10 11
1 2 3 4 5 6 7 8 9 10 11 0
2 3 4 5 6 7 8 9 10 11 0 1
3 4 5 6 7 8 9 10 11 0 1 2
4 5 6 7 8 9 10 11 0 1 2 3
5 6 7 8 9 10 11 0 1 2 3 4
6 7 8 9 10 11 0 1 2 3 4 5
7 8 9 10 11 0 1 2 3 4 5 6
8 9 10 11 0 1 2 3 4 5 6 7
9 10 11 0 1 2 3 4 5 6 7 8
10 11 0 1 2 3 4 5 6 7 8 9
11 0 1 2 3 4 5 6 7 8 9 10
The codomain is the cyclic group of order 12 under addition mod 12; its Cayley table is:
0 1 2 3 4 5 6 7 8 9 10 11
1 2 3 4 5 6 7 8 9 10 11 0
2 3 4 5 6 7 8 9 10 11 0 1
3 4 5 6 7 8 9 10 11 0 1 2
4 5 6 7 8 9 10 11 0 1 2 3
5 6 7 8 9 10 11 0 1 2 3 4
6 7 8 9 10 11 0 1 2 3 4 5
7 8 9 10 11 0 1 2 3 4 5 6
8 9 10 11 0 1 2 3 4 5 6 7
9 10 11 0 1 2 3 4 5 6 7 8
10 11 0 1 2 3 4 5 6 7 8 9
11 0 1 2 3 4 5 6 7 8 9 10
The candidate map f sends each element of the domain to an element of the codomain, listed as f(0) f(1) ... f(n-1):
f = [0 1 3 3 10 10 0 2 4 8 6 10]
For every ordered pair (x, y) of domain elements the map is a homomorphism exactly when f(x·y) = f(x)·f(y). Count the number of ordered pairs (x, y) for which this condition fails (this count is 0 if and only if f is a homomorphism).
The answer is the integer count.

Answer:
102

## Level 5

The domain is the cyclic group of order 21 under addition mod 21; its Cayley table is:
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0
2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1
3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2
4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3
5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4
6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5
7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6
8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7
9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8
10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9
11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10
12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11
13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12
14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13
15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
The codomain is the cyclic group of order 21 under addition mod 21; its Cayley table is:
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0
2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1
3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2
4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3
5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4
6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5
7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6
8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7
9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8
10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9
11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10
12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11
13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12
14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13
15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
The candidate map f sends each element of the domain to an element of the codomain, listed as f(0) f(1) ... f(n-1):
f = [7 5 6 11 0 16 1 5 3 14 11 1 18 12 0 2 2 12 6 15 9]
For every ordered pair (x, y) of domain elements the map is a homomorphism exactly when f(x·y) = f(x)·f(y). Count the number of ordered pairs (x, y) for which this condition fails (this count is 0 if and only if f is a homomorphism).
The answer is the integer count.

Answer:
425

The domain is the cyclic group of order 21 under addition mod 21; its Cayley table is:
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0
2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1
3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2
4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3
5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4
6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5
7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6
8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7
9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8
10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9
11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10
12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11
13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12
14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13
15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
The codomain is the cyclic group of order 21 under addition mod 21; its Cayley table is:
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0
2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1
3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2
4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3
5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4
6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5
7 8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6
8 9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7
9 10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8
10 11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9
11 12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10
12 13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11
13 14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12
14 15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13
15 16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
16 17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
17 18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
18 19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
19 20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
20 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
The candidate map f sends each element of the domain to an element of the codomain, listed as f(0) f(1) ... f(n-1):
f = [0 16 0 20 19 3 9 14 8 18 14 18 14 7 12 15 5 18 13 14 10]
For every ordered pair (x, y) of domain elements the map is a homomorphism exactly when f(x·y) = f(x)·f(y). Count the number of ordered pairs (x, y) for which this condition fails (this count is 0 if and only if f is a homomorphism).
The answer is the integer count.

Answer:
388
