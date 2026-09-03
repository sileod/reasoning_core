## Level 0

A Petri net has places with the given token counts and weighted pre-arcs from each transition to its input places. A transition is enabled when every input place holds at least the arc weight of tokens. Given the marking and arcs, list all enabled transitions in the answer, in lexicographic (sorted) order, separated by commas. If no transition is enabled, the answer is exactly 'none'.

Places:
{'p0': 4, 'p1': 5, 'p2': 2, 'p3': 5}

Transitions:
['t0', 't1', 't2', 't3', 't4']

Pre Arcs:
[('t0', 'p0', 2), ('t0', 'p1', 3), ('t0', 'p3', 2), ('t1', 'p1', 3), ('t1', 'p2', 3), ('t2', 'p1', 1), ('t2', 'p2', 1), ('t3', 'p0', 1), ('t3', 'p1', 3), ('t4', 'p0', 2), ('t4', 'p1', 1)]

The answer is:

Answer: t0, t2, t3, t4

A Petri net has places with the given token counts and weighted pre-arcs from each transition to its input places. A transition is enabled when every input place holds at least the arc weight of tokens. Given the marking and arcs, list all enabled transitions in the answer, in lexicographic (sorted) order, separated by commas. If no transition is enabled, the answer is exactly 'none'.

Places:
{'p0': 5, 'p1': 6, 'p2': 1, 'p3': 0}

Transitions:
['t0', 't1', 't2', 't3', 't4']

Pre Arcs:
[('t0', 'p0', 2), ('t0', 'p1', 1), ('t1', 'p1', 1), ('t2', 'p0', 1), ('t2', 'p2', 3), ('t2', 'p3', 1), ('t3', 'p1', 1), ('t3', 'p2', 2), ('t3', 'p3', 2), ('t4', 'p1', 3)]

The answer is:

Answer: t0, t1, t4

## Level 2

A Petri net has places with the given token counts and weighted pre-arcs from each transition to its input places. A transition is enabled when every input place holds at least the arc weight of tokens. Given the marking and arcs, list all enabled transitions in the answer, in lexicographic (sorted) order, separated by commas. If no transition is enabled, the answer is exactly 'none'.

Places:
{'p0': 2, 'p1': 1, 'p2': 7, 'p3': 5, 'p4': 5, 'p5': 4}

Transitions:
['t0', 't1', 't2', 't3', 't4', 't5', 't6']

Pre Arcs:
[('t0', 'p1', 4), ('t0', 'p2', 2), ('t1', 'p3', 3), ('t1', 'p4', 2), ('t2', 'p1', 4), ('t2', 'p3', 1), ('t3', 'p4', 5), ('t4', 'p2', 2), ('t4', 'p4', 4), ('t4', 'p5', 5), ('t5', 'p0', 4), ('t5', 'p3', 5), ('t6', 'p1', 5), ('t6', 'p5', 4)]

The answer is:

Answer: t1, t3

A Petri net has places with the given token counts and weighted pre-arcs from each transition to its input places. A transition is enabled when every input place holds at least the arc weight of tokens. Given the marking and arcs, list all enabled transitions in the answer, in lexicographic (sorted) order, separated by commas. If no transition is enabled, the answer is exactly 'none'.

Places:
{'p0': 0, 'p1': 2, 'p2': 6, 'p3': 6, 'p4': 4, 'p5': 2}

Transitions:
['t0', 't1', 't2', 't3', 't4', 't5', 't6']

Pre Arcs:
[('t0', 'p0', 5), ('t0', 'p1', 2), ('t0', 'p5', 4), ('t1', 'p0', 3), ('t1', 'p2', 4), ('t1', 'p4', 4), ('t2', 'p2', 1), ('t3', 'p2', 2), ('t3', 'p3', 2), ('t3', 'p5', 5), ('t4', 'p4', 1), ('t5', 'p3', 5), ('t5', 'p4', 1), ('t5', 'p5', 4), ('t6', 'p1', 1), ('t6', 'p2', 5)]

The answer is:

Answer: t2, t4, t6

## Level 5

A Petri net has places with the given token counts and weighted pre-arcs from each transition to its input places. A transition is enabled when every input place holds at least the arc weight of tokens. Given the marking and arcs, list all enabled transitions in the answer, in lexicographic (sorted) order, separated by commas. If no transition is enabled, the answer is exactly 'none'.

Places:
{'p0': 5, 'p1': 3, 'p2': 6, 'p3': 6, 'p4': 11, 'p5': 3, 'p6': 11, 'p7': 1, 'p8': 2}

Transitions:
['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9']

Pre Arcs:
[('t0', 'p6', 2), ('t1', 'p1', 4), ('t2', 'p1', 7), ('t2', 'p2', 2), ('t2', 'p4', 4), ('t3', 'p4', 4), ('t4', 'p7', 2), ('t5', 'p2', 3), ('t5', 'p4', 2), ('t5', 'p6', 5), ('t6', 'p0', 5), ('t6', 'p6', 5), ('t6', 'p8', 8), ('t7', 'p0', 6), ('t7', 'p5', 3), ('t7', 'p6', 2), ('t8', 'p2', 1), ('t8', 'p3', 5), ('t8', 'p6', 8), ('t9', 'p4', 3)]

The answer is:

Answer: t0, t3, t5, t8, t9

A Petri net has places with the given token counts and weighted pre-arcs from each transition to its input places. A transition is enabled when every input place holds at least the arc weight of tokens. Given the marking and arcs, list all enabled transitions in the answer, in lexicographic (sorted) order, separated by commas. If no transition is enabled, the answer is exactly 'none'.

Places:
{'p0': 11, 'p1': 3, 'p2': 9, 'p3': 2, 'p4': 11, 'p5': 2, 'p6': 3, 'p7': 10, 'p8': 3}

Transitions:
['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9']

Pre Arcs:
[('t0', 'p7', 8), ('t1', 'p3', 3), ('t1', 'p8', 7), ('t2', 'p0', 2), ('t2', 'p3', 4), ('t2', 'p5', 7), ('t3', 'p8', 3), ('t4', 'p6', 1), ('t5', 'p4', 8), ('t5', 'p8', 1), ('t6', 'p1', 8), ('t6', 'p3', 1), ('t7', 'p1', 8), ('t8', 'p5', 6), ('t8', 'p8', 5), ('t9', 'p5', 4), ('t9', 'p6', 1)]

The answer is:

Answer: t0, t3, t4, t5

