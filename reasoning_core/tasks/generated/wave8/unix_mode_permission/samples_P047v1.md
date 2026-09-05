# UnixModePermission - samples P047v1

## Level 0

A Unix file has the following octal mode, owner, and set of user accounts. Access is decided by the standard permission rules: the file owner uses the owner ('owner=') bits; otherwise a user whose groups contain the file's gid uses the group ('group=') bits; and every other requester uses the 'other=' bits. A requested access (read/write/execute) maps to the r/w/x bit of the applicable class.

File Mode:
0551 (owner=rx group=rx other=x)

File Owner:
uid=2810, gid=4919

Requested Access:
read

Users:
1. uid=2164; groups=[1558, 2298, 2303, 2946, 4919]
2. uid=1190; groups=[2303]
3. uid=166; groups=[2298, 4163, 4919]
4. uid=3086; groups=[1558, 2298]
5. uid=691; groups=[1558, 2303, 4163]
6. uid=1480; groups=[2303]
7. uid=755; groups=[1838, 2946, 4919]
8. uid=2603; groups=[1558, 1838, 2298]

Which users (by 1-based index in the Users list) are granted the requested access? Give their indices as space-separated integers in ascending order, or the single word 'none' if not one is granted.

Answer: 1 3 7

A Unix file has the following octal mode, owner, and set of user accounts. Access is decided by the standard permission rules: the file owner uses the owner ('owner=') bits; otherwise a user whose groups contain the file's gid uses the group ('group=') bits; and every other requester uses the 'other=' bits. A requested access (read/write/execute) maps to the r/w/x bit of the applicable class.

File Mode:
04644 (owner=rw group=r other=r)

File Owner:
uid=3380, gid=4998

Requested Access:
read

Users:
1. uid=2444; groups=[2621, 3022, 3881]
2. uid=2319; groups=[2621, 3022, 3881, 3916, 4896, 4998]
3. uid=3467; groups=[2535, 2671, 3916, 4998]
4. uid=3579; groups=[4896]
5. uid=251; groups=[2535, 2621, 2671, 3022, 3881, 4998]
6. uid=3544; groups=[2671, 3916, 4998]
7. uid=2049; groups=[2535]
8. uid=2510; groups=[2621, 2671, 3916]

Which users (by 1-based index in the Users list) are granted the requested access? Give their indices as space-separated integers in ascending order, or the single word 'none' if not one is granted.

Answer: 1 2 3 4 5 6 7 8

## Level 2

A Unix file has the following octal mode, owner, and set of user accounts. Access is decided by the standard permission rules: the file owner uses the owner ('owner=') bits; otherwise a user whose groups contain the file's gid uses the group ('group=') bits; and every other requester uses the 'other=' bits. A requested access (read/write/execute) maps to the r/w/x bit of the applicable class.

File Mode:
06470 (owner=r group=rwx other=-)

File Owner:
uid=2243, gid=4395

Requested Access:
write

Users:
1. uid=688; groups=[1726, 2653, 4397]
2. uid=886; groups=[1670, 1726, 2232, 2653, 3058, 3106, 4395, 4397]
3. uid=115; groups=[1670, 1726, 2232, 2451, 2653, 3393]
4. uid=3130; groups=[1670, 1726, 2232, 2653, 4395, 4397]
5. uid=2260; groups=[2653]
6. uid=3955; groups=[2232, 2653, 3058, 3106]
7. uid=3986; groups=[2232, 3058, 3106, 3393, 4397]
8. uid=2722; groups=[1670, 1726, 2451, 2653]
9. uid=1811; groups=[1726, 2232, 2451, 3058]
10. uid=3752; groups=[1726, 3393]
11. uid=1421; groups=[2653]
12. uid=1971; groups=[3106]

Which users (by 1-based index in the Users list) are granted the requested access? Give their indices as space-separated integers in ascending order, or the single word 'none' if not one is granted.

Answer: 2 4

A Unix file has the following octal mode, owner, and set of user accounts. Access is decided by the standard permission rules: the file owner uses the owner ('owner=') bits; otherwise a user whose groups contain the file's gid uses the group ('group=') bits; and every other requester uses the 'other=' bits. A requested access (read/write/execute) maps to the r/w/x bit of the applicable class.

File Mode:
06341 (owner=wx group=r other=x)

File Owner:
uid=3850, gid=3413

Requested Access:
read

Users:
1. uid=3614; groups=[1733, 2380, 3601]
2. uid=2005; groups=[1074, 3597]
3. uid=2718; groups=[1766, 3601, 4793]
4. uid=2493; groups=[1766, 2380, 2474, 3601, 4793]
5. uid=2061; groups=[2380, 2474, 3413]
6. uid=2675; groups=[1763]
7. uid=2186; groups=[1733, 3597]
8. uid=1624; groups=[3601]
9. uid=385; groups=[1074, 1733, 1763, 1766, 2380, 3413, 3597]
10. uid=1357; groups=[2380, 3413]
11. uid=3068; groups=[1766, 3413]
12. uid=331; groups=[1733, 1763, 1766, 2380, 3597, 3601, 4793]

Which users (by 1-based index in the Users list) are granted the requested access? Give their indices as space-separated integers in ascending order, or the single word 'none' if not one is granted.

Answer: 5 9 10 11

## Level 5

A Unix file has the following octal mode, owner, and set of user accounts. Access is decided by the standard permission rules: the file owner uses the owner ('owner=') bits; otherwise a user whose groups contain the file's gid uses the group ('group=') bits; and every other requester uses the 'other=' bits. A requested access (read/write/execute) maps to the r/w/x bit of the applicable class.

File Mode:
06555 (owner=rx group=rx other=rx)

File Owner:
uid=3041, gid=1496

Requested Access:
read

Users:
1. uid=2229; groups=[1133, 2552, 2592, 3478, 3803, 4138, 4300, 4607]
2. uid=3691; groups=[1133, 1588, 2552, 4300]
3. uid=1800; groups=[1133, 1496, 1588, 2072, 2552, 2592, 3478, 4241, 4607]
4. uid=3084; groups=[1133, 1496, 1588, 2072, 2592, 3478, 3803, 4138, 4241, 4300, 4607]
5. uid=3028; groups=[1133, 1496, 1575, 2072, 2552, 2592, 3478, 4138, 4241, 4300, 4607]
6. uid=3158; groups=[1133, 1575, 2072, 3803]
7. uid=3433; groups=[1133, 1496, 1575, 2072, 2592, 3478, 3803, 4138, 4241, 4300, 4607]
8. uid=451; groups=[1133, 1575, 2072, 2552, 3478, 3803, 4138, 4300, 4607]
9. uid=3811; groups=[1575, 1588, 2072, 2592, 3478, 3803, 4241, 4300]
10. uid=2804; groups=[1133, 1575, 2072, 2552, 3478, 3803, 4300, 4607]
11. uid=2966; groups=[1133, 1496, 1575, 1588, 2072, 2592, 3478, 3803, 4241, 4607]
12. uid=855; groups=[1133, 3478, 4138, 4241, 4300]
13. uid=2806; groups=[1133, 1496, 1575, 2552, 2592, 3478, 3803, 4241]
14. uid=622; groups=[1133, 1496, 1575, 1588, 2072, 2552, 2592, 4138, 4300, 4607]
15. uid=2595; groups=[1133, 1496, 1588, 2552, 3478, 3803, 4138, 4607]
16. uid=669; groups=[1588, 2552, 3478, 4138]
17. uid=2779; groups=[1133, 1496, 2072, 3478, 3803, 4138, 4241, 4300]
18. uid=767; groups=[1133, 1575, 1588, 2552, 2592, 4241]

Which users (by 1-based index in the Users list) are granted the requested access? Give their indices as space-separated integers in ascending order, or the single word 'none' if not one is granted.

Answer: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18

A Unix file has the following octal mode, owner, and set of user accounts. Access is decided by the standard permission rules: the file owner uses the owner ('owner=') bits; otherwise a user whose groups contain the file's gid uses the group ('group=') bits; and every other requester uses the 'other=' bits. A requested access (read/write/execute) maps to the r/w/x bit of the applicable class.

File Mode:
06217 (owner=w group=x other=rwx)

File Owner:
uid=2357, gid=3415

Requested Access:
read

Users:
1. uid=2359; groups=[1221, 3415, 4550]
2. uid=3874; groups=[2285, 2687, 4738]
3. uid=1579; groups=[1221, 2285, 4185]
4. uid=1213; groups=[1221, 2285, 2687, 2761, 4550]
5. uid=392; groups=[1221, 1738, 2285, 2687, 2761, 3415, 4185, 4500, 4738, 4831]
6. uid=2967; groups=[2761]
7. uid=3976; groups=[4831]
8. uid=819; groups=[1108, 1221, 1738, 2285, 4185, 4500, 4550, 4738, 4831, 4837]
9. uid=971; groups=[1108]
10. uid=2453; groups=[1108, 1738, 2285, 2687, 3415, 4185, 4550, 4738, 4831]
11. uid=949; groups=[1108, 1221, 1738, 2285, 2687, 2761, 3415, 4185, 4500, 4831, 4837]
12. uid=3069; groups=[4185]
13. uid=3662; groups=[1738, 4500, 4738, 4831]
14. uid=2477; groups=[1108, 1221, 2761, 4185, 4500, 4550, 4738, 4831, 4837]
15. uid=509; groups=[2687, 2761, 3415, 4500, 4550, 4837]
16. uid=2301; groups=[1108, 1221, 1738, 2285, 4837]
17. uid=2224; groups=[1108, 2285, 2687, 2761, 4550]
18. uid=1459; groups=[4185]

Which users (by 1-based index in the Users list) are granted the requested access? Give their indices as space-separated integers in ascending order, or the single word 'none' if not one is granted.

Answer: 2 3 4 6 7 8 9 12 13 14 16 17 18
