# C3 Linearization samples (P054v2)

## Level 0

Prompt:
Classes:
class B(A)
class C(B)
class D(C)
class E(D)
class F(E)
class G(F, A)

Query:
class D

Each line declares a class and its direct base classes, in C3 (Dylan/C3) linearization semantics. Compute the C3 method-resolution order (MRO) of the queried class: it starts with that class itself, then merges the parent linearizations, most-derived class first, ending at the least-derived ancestor.

Output only the MRO as a single line: the class names in linearization order, separated by commas. For example, if the MRO is X then Y then Z, the answer is 'X, Y, Z'.

The answer is the comma-separated MRO of the queried class.

Answer:
D, C, B, A

---

Prompt:
Classes:
class B(A)
class C(B)
class D(C, B)
class E(D, A)
class F(E)
class G(F)

Query:
class D

Each line declares a class and its direct base classes, in C3 (Dylan/C3) linearization semantics. Compute the C3 method-resolution order (MRO) of the queried class: it starts with that class itself, then merges the parent linearizations, most-derived class first, ending at the least-derived ancestor.

Output only the MRO as a single line: the class names in linearization order, separated by commas. For example, if the MRO is X then Y then Z, the answer is 'X, Y, Z'.

The answer is the comma-separated MRO of the queried class.

Answer:
D, C, B, A

---

## Level 2

Prompt:
Classes:
class B(A)
class C(B, A)
class D(C, A)
class E(D)
class F(E, C)
class G(F)
class H(G, B, A)
class I(H)
class J(I, G)
class K(J)

Query:
class G

Each line declares a class and its direct base classes, in C3 (Dylan/C3) linearization semantics. Compute the C3 method-resolution order (MRO) of the queried class: it starts with that class itself, then merges the parent linearizations, most-derived class first, ending at the least-derived ancestor.

Output only the MRO as a single line: the class names in linearization order, separated by commas. For example, if the MRO is X then Y then Z, the answer is 'X, Y, Z'.

The answer is the comma-separated MRO of the queried class.

Answer:
G, F, E, D, C, B, A

---

Prompt:
Classes:
class B(A)
class C(B)
class D(C, A)
class E(D)
class F(E, B, A)
class G(F)
class H(G, D)
class I(H)
class J(I)
class K(J)

Query:
class J

Each line declares a class and its direct base classes, in C3 (Dylan/C3) linearization semantics. Compute the C3 method-resolution order (MRO) of the queried class: it starts with that class itself, then merges the parent linearizations, most-derived class first, ending at the least-derived ancestor.

Output only the MRO as a single line: the class names in linearization order, separated by commas. For example, if the MRO is X then Y then Z, the answer is 'X, Y, Z'.

The answer is the comma-separated MRO of the queried class.

Answer:
J, I, H, G, F, E, D, C, B, A

---

## Level 5

Prompt:
Classes:
class B(A)
class C(B)
class D(C, B)
class E(D)
class F(E)
class G(F)
class H(G, C, B, A)
class I(H)
class J(I, B, A)
class K(J, C, B, A)
class L(K)
class M(L, G)
class N(M)
class O(N)
class P(O, I)
class Q(P)

Query:
class P

Each line declares a class and its direct base classes, in C3 (Dylan/C3) linearization semantics. Compute the C3 method-resolution order (MRO) of the queried class: it starts with that class itself, then merges the parent linearizations, most-derived class first, ending at the least-derived ancestor.

Output only the MRO as a single line: the class names in linearization order, separated by commas. For example, if the MRO is X then Y then Z, the answer is 'X, Y, Z'.

The answer is the comma-separated MRO of the queried class.

Answer:
P, O, N, M, L, K, J, I, H, G, F, E, D, C, B, A

---

Prompt:
Classes:
class B(A)
class C(B, A)
class D(C, B, A)
class E(D, A)
class F(E, A)
class G(F, A)
class H(G, E)
class I(H)
class J(I)
class K(J)
class L(K, C, A)
class M(L)
class N(M, D, B)
class O(N, K)
class P(O, B)
class Q(P, B, A)

Query:
class N

Each line declares a class and its direct base classes, in C3 (Dylan/C3) linearization semantics. Compute the C3 method-resolution order (MRO) of the queried class: it starts with that class itself, then merges the parent linearizations, most-derived class first, ending at the least-derived ancestor.

Output only the MRO as a single line: the class names in linearization order, separated by commas. For example, if the MRO is X then Y then Z, the answer is 'X, Y, Z'.

The answer is the comma-separated MRO of the queried class.

Answer:
N, M, L, K, J, I, H, G, F, E, D, C, B, A

---
