## Level 0

A small Petri net has the following places with their token counts:
- p1: 2 tokens
- p2: 3 tokens
- p3: 3 tokens
- p4: 2 tokens

Each transition lists its input places as pairs (place, weight):
- t1: consumes p3/3
- t2: consumes p3/2
- t3: consumes p4/2, p3/2

A transition is enabled if every input place holds at least its weight. Two enabled transitions with distinct names are a conflicting pair if they share some input place whose token count is less than the sum of the two weights on it (so firing both is impossible). The deficit of that shared place is the sum of the two weights minus its token count.

Among all conflicting pairs, take the one whose two transition names are lexicographically smallest (the pair with the smallest first name, then the smallest second name; names sort as t1 < t10 < t2). If that pair conflicts over several shared places, use the place with the largest deficit (ties to the place with the smallest name).

Answer as None if there is no conflicting pair, otherwise as firstTransition,secondTransition,sharedPlace,deficit (for example "t2,t5,p3,2" would mean t2 and t5 conflict over p3 which is short by 2 tokens).

Answer: t1,t2,p3,2

A small Petri net has the following places with their token counts:
- p1: 5 tokens
- p2: 1 tokens
- p3: 4 tokens
- p4: 2 tokens

Each transition lists its input places as pairs (place, weight):
- t1: consumes p1/4
- t2: consumes p2/1
- t3: consumes p3/3

A transition is enabled if every input place holds at least its weight. Two enabled transitions with distinct names are a conflicting pair if they share some input place whose token count is less than the sum of the two weights on it (so firing both is impossible). The deficit of that shared place is the sum of the two weights minus its token count.

Among all conflicting pairs, take the one whose two transition names are lexicographically smallest (the pair with the smallest first name, then the smallest second name; names sort as t1 < t10 < t2). If that pair conflicts over several shared places, use the place with the largest deficit (ties to the place with the smallest name).

Answer as None if there is no conflicting pair, otherwise as firstTransition,secondTransition,sharedPlace,deficit (for example "t2,t5,p3,2" would mean t2 and t5 conflict over p3 which is short by 2 tokens).

Answer: None

## Level 2

A small Petri net has the following places with their token counts:
- p1: 1 tokens
- p2: 1 tokens
- p3: 5 tokens
- p4: 2 tokens
- p5: 5 tokens
- p6: 5 tokens

Each transition lists its input places as pairs (place, weight):
- t1: consumes p1/1
- t2: consumes p2/1
- t3: consumes p3/1
- t4: consumes p4/1
- t5: consumes p5/3

A transition is enabled if every input place holds at least its weight. Two enabled transitions with distinct names are a conflicting pair if they share some input place whose token count is less than the sum of the two weights on it (so firing both is impossible). The deficit of that shared place is the sum of the two weights minus its token count.

Among all conflicting pairs, take the one whose two transition names are lexicographically smallest (the pair with the smallest first name, then the smallest second name; names sort as t1 < t10 < t2). If that pair conflicts over several shared places, use the place with the largest deficit (ties to the place with the smallest name).

Answer as None if there is no conflicting pair, otherwise as firstTransition,secondTransition,sharedPlace,deficit (for example "t2,t5,p3,2" would mean t2 and t5 conflict over p3 which is short by 2 tokens).

Answer: None

A small Petri net has the following places with their token counts:
- p1: 4 tokens
- p2: 3 tokens
- p3: 8 tokens
- p4: 8 tokens
- p5: 8 tokens
- p6: 2 tokens
- p7: 7 tokens
- p8: 2 tokens

Each transition lists its input places as pairs (place, weight):
- t1: consumes p1/4
- t2: consumes p1/4
- t3: consumes p2/1, p6/2, p4/1
- t4: consumes p6/1, p8/2, p2/2
- t5: consumes p2/3, p5/2
- t6: consumes p5/8, p8/2
- t7: consumes p4/1, p1/1

A transition is enabled if every input place holds at least its weight. Two enabled transitions with distinct names are a conflicting pair if they share some input place whose token count is less than the sum of the two weights on it (so firing both is impossible). The deficit of that shared place is the sum of the two weights minus its token count.

Among all conflicting pairs, take the one whose two transition names are lexicographically smallest (the pair with the smallest first name, then the smallest second name; names sort as t1 < t10 < t2). If that pair conflicts over several shared places, use the place with the largest deficit (ties to the place with the smallest name).

Answer as None if there is no conflicting pair, otherwise as firstTransition,secondTransition,sharedPlace,deficit (for example "t2,t5,p3,2" would mean t2 and t5 conflict over p3 which is short by 2 tokens).

Answer: t1,t2,p1,4

## Level 5

A small Petri net has the following places with their token counts:
- p1: 2 tokens
- p2: 2 tokens
- p3: 11 tokens
- p4: 1 tokens
- p5: 8 tokens
- p6: 2 tokens
- p7: 7 tokens
- p8: 3 tokens
- p9: 7 tokens
- p10: 10 tokens
- p11: 13 tokens
- p12: 2 tokens
- p13: 7 tokens

Each transition lists its input places as pairs (place, weight):
- t1: consumes p10/3, p7/1
- t2: consumes p10/8
- t3: consumes p12/1, p1/2, p11/9
- t4: consumes p5/3, p8/2, p2/2
- t5: consumes p3/9, p8/3, p5/3
- t6: consumes p6/1, p13/1
- t7: consumes p6/2, p8/2, p2/2
- t8: consumes p5/1, p2/2
- t9: consumes p5/7, p6/2
- t10: consumes p7/3, p3/5, p9/1
- t11: consumes p13/5, p2/1
- t12: consumes p3/10, p5/3, p12/1

A transition is enabled if every input place holds at least its weight. Two enabled transitions with distinct names are a conflicting pair if they share some input place whose token count is less than the sum of the two weights on it (so firing both is impossible). The deficit of that shared place is the sum of the two weights minus its token count.

Among all conflicting pairs, take the one whose two transition names are lexicographically smallest (the pair with the smallest first name, then the smallest second name; names sort as t1 < t10 < t2). If that pair conflicts over several shared places, use the place with the largest deficit (ties to the place with the smallest name).

Answer as None if there is no conflicting pair, otherwise as firstTransition,secondTransition,sharedPlace,deficit (for example "t2,t5,p3,2" would mean t2 and t5 conflict over p3 which is short by 2 tokens).

Answer: t1,t2,p10,1

A small Petri net has the following places with their token counts:
- p1: 16 tokens
- p2: 6 tokens
- p3: 14 tokens
- p4: 13 tokens
- p5: 10 tokens
- p6: 8 tokens
- p7: 15 tokens
- p8: 17 tokens
- p9: 9 tokens
- p10: 17 tokens
- p11: 2 tokens
- p12: 5 tokens
- p13: 3 tokens
- p14: 13 tokens
- p15: 7 tokens
- p16: 4 tokens
- p17: 18 tokens
- p18: 5 tokens

Each transition lists its input places as pairs (place, weight):
- t1: consumes p12/4
- t2: consumes p12/5
- t3: consumes p16/1, p12/5, p11/2
- t4: consumes p7/7, p2/6, p6/8
- t5: consumes p18/3, p3/5
- t6: consumes p4/4, p12/3, p7/9
- t7: consumes p18/1, p4/9
- t8: consumes p17/3, p4/8
- t9: consumes p14/8, p18/5, p10/10
- t10: consumes p17/8, p13/1, p11/2
- t11: consumes p11/1, p7/8
- t12: consumes p3/1, p1/11
- t13: consumes p6/3, p4/12
- t14: consumes p18/5, p4/10
- t15: consumes p2/5, p13/2, p15/1
- t16: consumes p11/2, p2/6
- t17: consumes p6/8, p9/7, p13/3

A transition is enabled if every input place holds at least its weight. Two enabled transitions with distinct names are a conflicting pair if they share some input place whose token count is less than the sum of the two weights on it (so firing both is impossible). The deficit of that shared place is the sum of the two weights minus its token count.

Among all conflicting pairs, take the one whose two transition names are lexicographically smallest (the pair with the smallest first name, then the smallest second name; names sort as t1 < t10 < t2). If that pair conflicts over several shared places, use the place with the largest deficit (ties to the place with the smallest name).

Answer as None if there is no conflicting pair, otherwise as firstTransition,secondTransition,sharedPlace,deficit (for example "t2,t5,p3,2" would mean t2 and t5 conflict over p3 which is short by 2 tokens).

Answer: t1,t2,p12,4
