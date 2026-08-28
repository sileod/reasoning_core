# Level 0
## Example 1
**Prompt:**

Program:
if (x0 > (x1 * 2)):
x1 = (0 * (6 * 2))
else:
x1 = ((1 - x1) + 1)
x0 = ((5 + x1) - (6 * 2))
x0 = 4
x0 = x0

Before the program: x0 in [1,2]; x1 in [0,2].

An interval abstract interpretation computes the abstract range [0,2] for variable x1 at the program end, and exhaustive concrete execution finds the true reachable range [0,1].

Compare the abstract range to the true concrete range. Answer with exactly 'exact' if the abstract range equals the true reachable range (sound and complete), or 'sound' if the abstract range is a strict over-approximation (sound but not complete).

**Answer:** `sound`

## Example 2
**Prompt:**

Program:
x1 = ((6 - 5) - (2 * 0))
x0 = x0
x0 = 2
if (x1 < (6 - 5)):
x0 = (x0 - x1)
else:
x1 = (x0 + x0)

Before the program: x0 in [0,2]; x1 in [2,5].

An interval abstract interpretation computes the abstract range [1,2] for variable x0 at the program end, and exhaustive concrete execution finds the true reachable range [2,2].

Compare the abstract range to the true concrete range. Answer with exactly 'exact' if the abstract range equals the true reachable range (sound and complete), or 'sound' if the abstract range is a strict over-approximation (sound but not complete).

**Answer:** `sound`

# Level 2
## Example 1
**Prompt:**

Program:
if ((x0 + 8) > (10 - 8)):
x0 = x3
else:
x2 = x2
x1 = 3
x3 = (3 * (2 + 3))
x3 = x3
x0 = (x0 * (x0 - x0))
x1 = x0

Before the program: x0 in [4,8]; x1 in [1,4]; x2 in [3,8]; x3 in [3,6].

An interval abstract interpretation computes the abstract range [15,15] for variable x3 at the program end, and exhaustive concrete execution finds the true reachable range [15,15].

Compare the abstract range to the true concrete range. Answer with exactly 'exact' if the abstract range equals the true reachable range (sound and complete), or 'sound' if the abstract range is a strict over-approximation (sound but not complete).

**Answer:** `exact`

## Example 2
**Prompt:**

Program:
x0 = ((5 * x2) + (x1 * 3))
x2 = x3
x2 = ((x0 * 5) + x0)
x3 = (10 - (x1 + x0))
x0 = 10

Before the program: x0 in [0,5]; x1 in [2,7]; x2 in [2,4]; x3 in [3,5].

An interval abstract interpretation computes the abstract range [96,246] for variable x2 at the program end, and exhaustive concrete execution finds the true reachable range [96,246].

Compare the abstract range to the true concrete range. Answer with exactly 'exact' if the abstract range equals the true reachable range (sound and complete), or 'sound' if the abstract range is a strict over-approximation (sound but not complete).

**Answer:** `exact`

# Level 5
## Example 1
**Prompt:**

Program:
x6 = (x5 * 14)
x0 = ((10 * 14) - x2)
x5 = ((x1 * x1) * 6)
x4 = ((3 - x5) * 14)
x1 = ((7 * x6) * (x2 * 14))
x2 = x1
x0 = x1
x3 = 0
x6 = ((12 + 13) + 6)

Before the program: x0 in [1,6]; x1 in [7,15]; x2 in [1,3]; x3 in [6,13]; x4 in [1,9]; x5 in [7,15]; x6 in [1,6].

An interval abstract interpretation computes the abstract range [294,1350] for variable x5 at the program end, and exhaustive concrete execution finds the true reachable range [294,1350].

Compare the abstract range to the true concrete range. Answer with exactly 'exact' if the abstract range equals the true reachable range (sound and complete), or 'sound' if the abstract range is a strict over-approximation (sound but not complete).

**Answer:** `exact`

## Example 2
**Prompt:**

Program:
x0 = ((10 + 14) * (3 - 10))
x0 = ((8 - x4) * 10)
x4 = 11
if ((x3 + x6) < (x0 * x0)):
x1 = (16 - (16 - 11))
else:
x5 = ((7 - 6) + (x2 * x3))
x6 = ((1 - 10) - x0)
x4 = (x2 + (2 * x2))
x1 = ((x0 - 5) + 7)
x3 = ((1 + x6) - (7 - 0))
x0 = x2

Before the program: x0 in [4,11]; x1 in [4,7]; x2 in [2,3]; x3 in [8,13]; x4 in [8,16]; x5 in [8,10]; x6 in [4,9].

An interval abstract interpretation computes the abstract range [6,9] for variable x4 at the program end, and exhaustive concrete execution finds the true reachable range [6,9].

Compare the abstract range to the true concrete range. Answer with exactly 'exact' if the abstract range equals the true reachable range (sound and complete), or 'sound' if the abstract range is a strict over-approximation (sound but not complete).

**Answer:** `exact`
