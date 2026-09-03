## Level 0

Prompt:
A schedule interleaves the read, write and commit operations of database transactions T0,T1,... on data items X0,X1,.... Each line is one operation "Tn reads Xk", "Tn writes Xk", or "Tn commits". A write stays uncommitted until the writing transaction commits. Reading an item written by an uncommitted transaction is a dirty read; a transaction overwriting an item that an uncommitted transaction wrote is a dirty write.

Classify by the strictest property the schedule violates:
- unrecoverable: some transaction commits after reading data whose writing transaction has not yet committed.
- recoverable: dirty reads occur, but no transaction commits before every writer of a value it read has committed.
- cascadeless: no dirty reads occur, but some dirty write happens.
- strict: no dirty writes and no dirty reads occur.
The four classes are exclusive and exhaustive; report the one that applies.

Schedule:
1. T0 reads X0
2. T0 writes X0
3. T0 commits
4. T1 reads X0
5. T1 reads X2
6. T1 reads X0
7. T1 commits
8. T2 reads X2
9. T2 reads X1
10. T2 commits
11. T3 writes X2
12. T3 writes X1
13. T3 reads X2
14. T3 commits

Answer with the class name, a space, then the 1-based index of the first operation that justifies it: the first dirty read whose reader commits before its writer for unrecoverable, the first dirty read for recoverable, the first dirty write for cascadeless, and 0 for strict.

Example: for a schedule whose first bad operation is operation 3, answer "recoverable 3".

Answer:
strict 0

Prompt:
A schedule interleaves the read, write and commit operations of database transactions T0,T1,... on data items X0,X1,.... Each line is one operation "Tn reads Xk", "Tn writes Xk", or "Tn commits". A write stays uncommitted until the writing transaction commits. Reading an item written by an uncommitted transaction is a dirty read; a transaction overwriting an item that an uncommitted transaction wrote is a dirty write.

Classify by the strictest property the schedule violates:
- unrecoverable: some transaction commits after reading data whose writing transaction has not yet committed.
- recoverable: dirty reads occur, but no transaction commits before every writer of a value it read has committed.
- cascadeless: no dirty reads occur, but some dirty write happens.
- strict: no dirty writes and no dirty reads occur.
The four classes are exclusive and exhaustive; report the one that applies.

Schedule:
1. T0 reads X1
2. T0 commits
3. T1 reads X0
4. T1 writes X1
5. T1 writes X0
6. T1 commits
7. T2 writes X1
8. T3 reads X1
9. T3 commits
10. T2 commits

Answer with the class name, a space, then the 1-based index of the first operation that justifies it: the first dirty read whose reader commits before its writer for unrecoverable, the first dirty read for recoverable, the first dirty write for cascadeless, and 0 for strict.

Example: for a schedule whose first bad operation is operation 3, answer "recoverable 3".

Answer:
unrecoverable 8


## Level 2

Prompt:
A schedule interleaves the read, write and commit operations of database transactions T0,T1,... on data items X0,X1,.... Each line is one operation "Tn reads Xk", "Tn writes Xk", or "Tn commits". A write stays uncommitted until the writing transaction commits. Reading an item written by an uncommitted transaction is a dirty read; a transaction overwriting an item that an uncommitted transaction wrote is a dirty write.

Classify by the strictest property the schedule violates:
- unrecoverable: some transaction commits after reading data whose writing transaction has not yet committed.
- recoverable: dirty reads occur, but no transaction commits before every writer of a value it read has committed.
- cascadeless: no dirty reads occur, but some dirty write happens.
- strict: no dirty writes and no dirty reads occur.
The four classes are exclusive and exhaustive; report the one that applies.

Schedule:
1. T0 writes X2
2. T0 reads X2
3. T0 reads X2
4. T0 writes X1
5. T0 commits
6. T1 writes X0
7. T2 writes X0
8. T1 commits
9. T2 commits
10. T3 reads X1
11. T3 writes X1
12. T3 reads X2
13. T3 commits

Answer with the class name, a space, then the 1-based index of the first operation that justifies it: the first dirty read whose reader commits before its writer for unrecoverable, the first dirty read for recoverable, the first dirty write for cascadeless, and 0 for strict.

Example: for a schedule whose first bad operation is operation 3, answer "recoverable 3".

Answer:
cascadeless 7

Prompt:
A schedule interleaves the read, write and commit operations of database transactions T0,T1,... on data items X0,X1,.... Each line is one operation "Tn reads Xk", "Tn writes Xk", or "Tn commits". A write stays uncommitted until the writing transaction commits. Reading an item written by an uncommitted transaction is a dirty read; a transaction overwriting an item that an uncommitted transaction wrote is a dirty write.

Classify by the strictest property the schedule violates:
- unrecoverable: some transaction commits after reading data whose writing transaction has not yet committed.
- recoverable: dirty reads occur, but no transaction commits before every writer of a value it read has committed.
- cascadeless: no dirty reads occur, but some dirty write happens.
- strict: no dirty writes and no dirty reads occur.
The four classes are exclusive and exhaustive; report the one that applies.

Schedule:
1. T0 writes X4
2. T0 reads X2
3. T0 writes X2
4. T0 reads X3
5. T0 commits
6. T1 writes X0
7. T1 writes X3
8. T1 reads X3
9. T1 writes X0
10. T1 commits
11. T2 writes X4
12. T3 writes X4
13. T2 commits
14. T3 commits

Answer with the class name, a space, then the 1-based index of the first operation that justifies it: the first dirty read whose reader commits before its writer for unrecoverable, the first dirty read for recoverable, the first dirty write for cascadeless, and 0 for strict.

Example: for a schedule whose first bad operation is operation 3, answer "recoverable 3".

Answer:
cascadeless 12


## Level 5

Prompt:
A schedule interleaves the read, write and commit operations of database transactions T0,T1,... on data items X0,X1,.... Each line is one operation "Tn reads Xk", "Tn writes Xk", or "Tn commits". A write stays uncommitted until the writing transaction commits. Reading an item written by an uncommitted transaction is a dirty read; a transaction overwriting an item that an uncommitted transaction wrote is a dirty write.

Classify by the strictest property the schedule violates:
- unrecoverable: some transaction commits after reading data whose writing transaction has not yet committed.
- recoverable: dirty reads occur, but no transaction commits before every writer of a value it read has committed.
- cascadeless: no dirty reads occur, but some dirty write happens.
- strict: no dirty writes and no dirty reads occur.
The four classes are exclusive and exhaustive; report the one that applies.

Schedule:
1. T0 writes X5
2. T0 reads X3
3. T0 reads X7
4. T0 reads X3
5. T0 writes X2
6. T0 reads X7
7. T0 commits
8. T1 reads X1
9. T1 reads X1
10. T1 writes X5
11. T1 reads X6
12. T1 reads X3
13. T1 writes X6
14. T1 commits
15. T2 writes X6
16. T3 reads X6
17. T2 commits
18. T3 commits
19. T4 writes X5
20. T4 writes X4
21. T4 writes X4
22. T4 reads X5
23. T4 reads X4
24. T4 writes X3
25. T4 commits
26. T5 writes X6
27. T5 reads X4
28. T5 reads X1
29. T5 reads X3
30. T5 reads X0
31. T5 reads X3
32. T5 reads X1
33. T5 commits
34. T6 reads X6
35. T6 writes X2
36. T6 reads X6
37. T6 reads X6
38. T6 reads X3
39. T6 reads X6
40. T6 reads X1
41. T6 commits

Answer with the class name, a space, then the 1-based index of the first operation that justifies it: the first dirty read whose reader commits before its writer for unrecoverable, the first dirty read for recoverable, the first dirty write for cascadeless, and 0 for strict.

Example: for a schedule whose first bad operation is operation 3, answer "recoverable 3".

Answer:
recoverable 16

Prompt:
A schedule interleaves the read, write and commit operations of database transactions T0,T1,... on data items X0,X1,.... Each line is one operation "Tn reads Xk", "Tn writes Xk", or "Tn commits". A write stays uncommitted until the writing transaction commits. Reading an item written by an uncommitted transaction is a dirty read; a transaction overwriting an item that an uncommitted transaction wrote is a dirty write.

Classify by the strictest property the schedule violates:
- unrecoverable: some transaction commits after reading data whose writing transaction has not yet committed.
- recoverable: dirty reads occur, but no transaction commits before every writer of a value it read has committed.
- cascadeless: no dirty reads occur, but some dirty write happens.
- strict: no dirty writes and no dirty reads occur.
The four classes are exclusive and exhaustive; report the one that applies.

Schedule:
1. T0 writes X6
2. T1 reads X6
3. T0 commits
4. T1 commits
5. T2 reads X0
6. T2 writes X7
7. T2 reads X5
8. T2 writes X3
9. T2 writes X6
10. T2 writes X1
11. T2 commits
12. T3 writes X0
13. T3 reads X7
14. T3 reads X6
15. T3 reads X2
16. T3 reads X7
17. T3 reads X6
18. T3 reads X3
19. T3 writes X0
20. T3 commits
21. T4 writes X3
22. T4 reads X6
23. T4 reads X0
24. T4 reads X2
25. T4 reads X7
26. T4 writes X1
27. T4 reads X2
28. T4 reads X6
29. T4 commits
30. T5 reads X0
31. T5 writes X5
32. T5 reads X4
33. T5 writes X7
34. T5 reads X7
35. T5 reads X7
36. T5 writes X6
37. T5 writes X0
38. T5 commits
39. T6 reads X2
40. T6 writes X3
41. T6 reads X1
42. T6 writes X0
43. T6 reads X1
44. T6 writes X5
45. T6 writes X1
46. T6 reads X5
47. T6 commits

Answer with the class name, a space, then the 1-based index of the first operation that justifies it: the first dirty read whose reader commits before its writer for unrecoverable, the first dirty read for recoverable, the first dirty write for cascadeless, and 0 for strict.

Example: for a schedule whose first bad operation is operation 3, answer "recoverable 3".

Answer:
recoverable 2

