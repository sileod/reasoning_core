# Level 0
**Prompt:**
Variables: x0 in {0,1}, x1 in {0,1}
Constraints:
  0: (x1 or x0)
  1: (~x1)
  2: (~x0 or x1)
  3: (~x0 or ~x1)
  4: (~x1 or x0)

The whole set of constraints is unsatisfiable. Find a lexicographically canonical subset-minimal inconsistent subset. A subset is subset-minimal inconsistent if it is unsatisfiable and every strictly smaller subset of it is satisfiable. Among all such subsets, report one with the minimum number of constraints; if several tie, report the one whose sorted constraint indices form the lexicographically smallest tuple.
Answer with the sorted constraint indices as a comma-separated list, e.g. "0,2" or "1".
**Answer:**
0,1,2

**Prompt:**
Variables: x0 in {0,1}, x1 in {0,1}
Constraints:
  0: (~x0 or ~x1)
  1: (x0 or ~x1)
  2: (~x1 or ~x0)
  3: (x1)
  4: (~x0 or ~x1)

The whole set of constraints is unsatisfiable. Find a lexicographically canonical subset-minimal inconsistent subset. A subset is subset-minimal inconsistent if it is unsatisfiable and every strictly smaller subset of it is satisfiable. Among all such subsets, report one with the minimum number of constraints; if several tie, report the one whose sorted constraint indices form the lexicographically smallest tuple.
Answer with the sorted constraint indices as a comma-separated list, e.g. "0,2" or "1".
**Answer:**
0,1,3

# Level 2
**Prompt:**
Variables: x0 in {0,1}, x1 in {0,1}, x2 in {0,1}, x3 in {0,1}
Constraints:
  0: (x1 or x0)
  1: (x2 or ~x1)
  2: (x3 or ~x1)
  3: (~x0 or x2)
  4: (T)
  5: (~x3 or x2 or ~x0)
  6: (~x2)

The whole set of constraints is unsatisfiable. Find a lexicographically canonical subset-minimal inconsistent subset. A subset is subset-minimal inconsistent if it is unsatisfiable and every strictly smaller subset of it is satisfiable. Among all such subsets, report one with the minimum number of constraints; if several tie, report the one whose sorted constraint indices form the lexicographically smallest tuple.
Answer with the sorted constraint indices as a comma-separated list, e.g. "0,2" or "1".
**Answer:**
0,1,3,6

**Prompt:**
Variables: x0 in {0,1}, x1 in {0,1}, x2 in {0,1}, x3 in {0,1}
Constraints:
  0: (x1 or x3)
  1: (x1 or x2)
  2: (x3)
  3: (~x3)
  4: (~x1)
  5: (~x0)
  6: (F)

The whole set of constraints is unsatisfiable. Find a lexicographically canonical subset-minimal inconsistent subset. A subset is subset-minimal inconsistent if it is unsatisfiable and every strictly smaller subset of it is satisfiable. Among all such subsets, report one with the minimum number of constraints; if several tie, report the one whose sorted constraint indices form the lexicographically smallest tuple.
Answer with the sorted constraint indices as a comma-separated list, e.g. "0,2" or "1".
**Answer:**
2,3

# Level 5
**Prompt:**
Variables: x0 in {0,1}, x1 in {0,1}, x2 in {0,1}, x3 in {0,1}, x4 in {0,1}, x5 in {0,1}, x6 in {0,1}
Constraints:
  0: (x5 or x1)
  1: (x2)
  2: (x4 or x0)
  3: (T)
  4: (F)
  5: (T)
  6: (F)
  7: (~x2)
  8: (x4 or ~x6)
  9: (~x1 or x2)
  10: (T)

The whole set of constraints is unsatisfiable. Find a lexicographically canonical subset-minimal inconsistent subset. A subset is subset-minimal inconsistent if it is unsatisfiable and every strictly smaller subset of it is satisfiable. Among all such subsets, report one with the minimum number of constraints; if several tie, report the one whose sorted constraint indices form the lexicographically smallest tuple.
Answer with the sorted constraint indices as a comma-separated list, e.g. "0,2" or "1".
**Answer:**
1,7

**Prompt:**
Variables: x0 in {0,1}, x1 in {0,1}, x2 in {0,1}, x3 in {0,1}, x4 in {0,1}, x5 in {0,1}, x6 in {0,1}
Constraints:
  0: (T)
  1: (x5 or ~x2)
  2: (T)
  3: (x4 or ~x6)
  4: (~x0 or ~x6)
  5: (~x2)
  6: (~x6 or ~x2)
  7: (x2)
  8: (~x3)
  9: (T)
  10: (x4)

The whole set of constraints is unsatisfiable. Find a lexicographically canonical subset-minimal inconsistent subset. A subset is subset-minimal inconsistent if it is unsatisfiable and every strictly smaller subset of it is satisfiable. Among all such subsets, report one with the minimum number of constraints; if several tie, report the one whose sorted constraint indices form the lexicographically smallest tuple.
Answer with the sorted constraint indices as a comma-separated list, e.g. "0,2" or "1".
**Answer:**
5,7
