# Level 0

Prompt:


Expr:
(+ (+ [-7, -1] [-3, -1]) [2, 4])

Compute the exact result of this interval-arithmetic expression on closed intervals with rational endpoints. Operations: + (sum), - (difference), * (product), / (quotient; undefined when the divisor contains zero), intersection (the overlap of the two intervals, empty when they do not overlap). Propagate values exactly. Give the tight resulting interval in the form [lo, hi] with lo and hi as reduced fractions or integers, or the single word empty when the expression has no value. For example an intersection of [0, 1] and [2, 3] is empty.

Answer:


[-8, 2]

---


Prompt:


Expr:
(- (* [-2, 4] [-3, 1]) [-4, 0])

Compute the exact result of this interval-arithmetic expression on closed intervals with rational endpoints. Operations: + (sum), - (difference), * (product), / (quotient; undefined when the divisor contains zero), intersection (the overlap of the two intervals, empty when they do not overlap). Propagate values exactly. Give the tight resulting interval in the form [lo, hi] with lo and hi as reduced fractions or integers, or the single word empty when the expression has no value. For example an intersection of [0, 1] and [2, 3] is empty.

Answer:


[-12, 10]

---


# Level 2

Prompt:


Expr:
(+ (+ (/ [3, 5] [3, 6]) [-5, 7]) [-9, 7])

Compute the exact result of this interval-arithmetic expression on closed intervals with rational endpoints. Operations: + (sum), - (difference), * (product), / (quotient; undefined when the divisor contains zero), intersection (the overlap of the two intervals, empty when they do not overlap). Propagate values exactly. Give the tight resulting interval in the form [lo, hi] with lo and hi as reduced fractions or integers, or the single word empty when the expression has no value. For example an intersection of [0, 1] and [2, 3] is empty.

Answer:


[-27/2, 47/3]

---


Prompt:


Expr:
(intersection [-8, 0] [3, 7])

Compute the exact result of this interval-arithmetic expression on closed intervals with rational endpoints. Operations: + (sum), - (difference), * (product), / (quotient; undefined when the divisor contains zero), intersection (the overlap of the two intervals, empty when they do not overlap). Propagate values exactly. Give the tight resulting interval in the form [lo, hi] with lo and hi as reduced fractions or integers, or the single word empty when the expression has no value. For example an intersection of [0, 1] and [2, 3] is empty.

Answer:


empty

---


# Level 5

Prompt:


Expr:
(+ (+ (/ [-10, 2] [4, 16]) [-24, 12]) [-8, 2])

Compute the exact result of this interval-arithmetic expression on closed intervals with rational endpoints. Operations: + (sum), - (difference), * (product), / (quotient; undefined when the divisor contains zero), intersection (the overlap of the two intervals, empty when they do not overlap). Propagate values exactly. Give the tight resulting interval in the form [lo, hi] with lo and hi as reduced fractions or integers, or the single word empty when the expression has no value. For example an intersection of [0, 1] and [2, 3] is empty.

Answer:


[-69/2, 29/2]

---


Prompt:


Expr:
(/ [-8, 4] [4, 11])

Compute the exact result of this interval-arithmetic expression on closed intervals with rational endpoints. Operations: + (sum), - (difference), * (product), / (quotient; undefined when the divisor contains zero), intersection (the overlap of the two intervals, empty when they do not overlap). Propagate values exactly. Give the tight resulting interval in the form [lo, hi] with lo and hi as reduced fractions or integers, or the single word empty when the expression has no value. For example an intersection of [0, 1] and [2, 3] is empty.

Answer:


[-2, 1]

---

