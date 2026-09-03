Level 0
Example 1
Prompt:
A router must select the best path to a destination from the candidate routes below. The best-path rule sequence, applied in order, is:
1. Highest local-preference wins.
2. If tied, the shortest AS-path length wins.
3. If still tied, the lowest origin value wins (IGP=0, EGP=1, incomplete=2).
4. If still tied, the lowest MED wins.
5. If still tied, the lowest neighbor IP address wins.

Routes:
C local-pref 190 as-path-length 6 origin IGP med 23 neighbor 10.54.144.1
B local-pref 188 as-path-length 1 origin IGP med 54 neighbor 10.58.198.1
D local-pref 114 as-path-length 1 origin EGP med 61 neighbor 10.37.51.1
A local-pref 184 as-path-length 5 origin incomplete med 51 neighbor 10.151.157.1

The champion route is the single route selected by that rule sequence. Print its identifier letter, e.g. A.
Answer: C

Level 0
Example 2
Prompt:
A router must select the best path to a destination from the candidate routes below. The best-path rule sequence, applied in order, is:
1. Highest local-preference wins.
2. If tied, the shortest AS-path length wins.
3. If still tied, the lowest origin value wins (IGP=0, EGP=1, incomplete=2).
4. If still tied, the lowest MED wins.
5. If still tied, the lowest neighbor IP address wins.

Routes:
B local-pref 190 as-path-length 1 origin EGP med 75 neighbor 10.110.33.1
A local-pref 176 as-path-length 3 origin incomplete med 7 neighbor 10.74.79.1
D local-pref 135 as-path-length 4 origin incomplete med 64 neighbor 10.48.180.1
C local-pref 109 as-path-length 6 origin incomplete med 77 neighbor 10.188.197.1

The champion route is the single route selected by that rule sequence. Print its identifier letter, e.g. A.
Answer: B

Level 2
Example 1
Prompt:
A router must select the best path to a destination from the candidate routes below. The best-path rule sequence, applied in order, is:
1. Highest local-preference wins.
2. If tied, the shortest AS-path length wins.
3. If still tied, the lowest origin value wins (IGP=0, EGP=1, incomplete=2).
4. If still tied, the lowest MED wins.
5. If still tied, the lowest neighbor IP address wins.

Routes:
F local-pref 95 as-path-length 5 origin IGP med 25 neighbor 10.87.154.1
A local-pref 216 as-path-length 4 origin EGP med 33 neighbor 10.176.86.1
E local-pref 118 as-path-length 3 origin incomplete med 41 neighbor 10.142.59.1
D local-pref 61 as-path-length 3 origin incomplete med 64 neighbor 10.95.92.1
C local-pref 190 as-path-length 5 origin EGP med 32 neighbor 10.62.174.1
B local-pref 114 as-path-length 2 origin IGP med 60 neighbor 10.80.29.1

The champion route is the single route selected by that rule sequence. Print its identifier letter, e.g. A.
Answer: A

Level 2
Example 2
Prompt:
A router must select the best path to a destination from the candidate routes below. The best-path rule sequence, applied in order, is:
1. Highest local-preference wins.
2. If tied, the shortest AS-path length wins.
3. If still tied, the lowest origin value wins (IGP=0, EGP=1, incomplete=2).
4. If still tied, the lowest MED wins.
5. If still tied, the lowest neighbor IP address wins.

Routes:
B local-pref 188 as-path-length 1 origin IGP med 41 neighbor 10.28.130.1
A local-pref 199 as-path-length 5 origin EGP med 21 neighbor 10.22.20.1
C local-pref 210 as-path-length 1 origin EGP med 31 neighbor 10.60.68.1
F local-pref 110 as-path-length 3 origin EGP med 16 neighbor 10.154.39.1
E local-pref 159 as-path-length 5 origin IGP med 0 neighbor 10.160.83.1
D local-pref 146 as-path-length 6 origin incomplete med 22 neighbor 10.105.137.1

The champion route is the single route selected by that rule sequence. Print its identifier letter, e.g. A.
Answer: C

Level 5
Example 1
Prompt:
A router must select the best path to a destination from the candidate routes below. The best-path rule sequence, applied in order, is:
1. Highest local-preference wins.
2. If tied, the shortest AS-path length wins.
3. If still tied, the lowest origin value wins (IGP=0, EGP=1, incomplete=2).
4. If still tied, the lowest MED wins.
5. If still tied, the lowest neighbor IP address wins.

Routes:
B local-pref 147 as-path-length 4 origin incomplete med 32 neighbor 10.72.67.1
D local-pref 162 as-path-length 6 origin IGP med 46 neighbor 10.35.59.1
K local-pref 171 as-path-length 2 origin IGP med 22 neighbor 10.106.86.1
A local-pref 211 as-path-length 6 origin IGP med 13 neighbor 10.42.194.1
C local-pref 203 as-path-length 5 origin IGP med 3 neighbor 10.137.131.1
H local-pref 172 as-path-length 3 origin EGP med 50 neighbor 10.95.193.1
J local-pref 198 as-path-length 6 origin IGP med 22 neighbor 10.152.200.1
I local-pref 173 as-path-length 5 origin incomplete med 73 neighbor 10.69.118.1
E local-pref 198 as-path-length 4 origin incomplete med 17 neighbor 10.48.73.1
G local-pref 102 as-path-length 1 origin EGP med 14 neighbor 10.104.55.1
F local-pref 158 as-path-length 3 origin EGP med 57 neighbor 10.35.17.1

The champion route is the single route selected by that rule sequence. Print its identifier letter, e.g. A.
Answer: A

Level 5
Example 2
Prompt:
A router must select the best path to a destination from the candidate routes below. The best-path rule sequence, applied in order, is:
1. Highest local-preference wins.
2. If tied, the shortest AS-path length wins.
3. If still tied, the lowest origin value wins (IGP=0, EGP=1, incomplete=2).
4. If still tied, the lowest MED wins.
5. If still tied, the lowest neighbor IP address wins.

Routes:
A local-pref 191 as-path-length 3 origin EGP med 18 neighbor 10.25.74.1
G local-pref 169 as-path-length 4 origin EGP med 2 neighbor 10.105.91.1
J local-pref 171 as-path-length 5 origin EGP med 70 neighbor 10.36.152.1
D local-pref 187 as-path-length 5 origin EGP med 24 neighbor 10.189.56.1
H local-pref 109 as-path-length 6 origin incomplete med 79 neighbor 10.149.105.1
I local-pref 141 as-path-length 2 origin incomplete med 65 neighbor 10.111.37.1
F local-pref 105 as-path-length 6 origin EGP med 78 neighbor 10.12.56.1
K local-pref 86 as-path-length 2 origin EGP med 72 neighbor 10.27.37.1
E local-pref 111 as-path-length 1 origin incomplete med 57 neighbor 10.23.104.1
C local-pref 135 as-path-length 1 origin incomplete med 34 neighbor 10.154.20.1
B local-pref 181 as-path-length 6 origin EGP med 13 neighbor 10.172.100.1

The champion route is the single route selected by that rule sequence. Print its identifier letter, e.g. A.
Answer: A

