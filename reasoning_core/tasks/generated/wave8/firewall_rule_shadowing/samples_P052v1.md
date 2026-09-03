## Level 0
### Example 1
**Prompt:**
An ordered firewall processes packets top to bottom: the first matching rule decides the action (allow or deny), later rules are ignored for that packet. A rule is fully shadowed when some strictly earlier rule matches every packet that it would match and has the same action, so the later rule can never affect any packet.
Each rule has the form: INDEX PROTOCOL SRC-LO-SRC-HI DST-LO-DST-HI ACTION, where the protocol is tcp or udp and SRC/DST are inclusive address ranges from 0 upward. Two rules match the same packet only if their protocols are equal and both the source ranges and the destination ranges overlap.
Rules:
0 udp 39-55 36-61 allow
1 udp 33-46 54-54 allow
2 tcp 55-58 51-56 allow
3 tcp 46-50 4-33 allow
4 udp 12-23 12-44 deny
5 udp 53-53 55-59 allow

Which is the FIRST rule (smallest index) that is fully shadowed by an earlier rule? Output its rule verbatim in the exact format INDEX PROTOCOL SRC-LO-SRC-HI DST-LO-DST-HI ACTION, or output None if no rule is fully shadowed.
The answer is the rule itself.

**Answer:**
5 udp 53-53 55-59 allow

### Example 2
**Prompt:**
An ordered firewall processes packets top to bottom: the first matching rule decides the action (allow or deny), later rules are ignored for that packet. A rule is fully shadowed when some strictly earlier rule matches every packet that it would match and has the same action, so the later rule can never affect any packet.
Each rule has the form: INDEX PROTOCOL SRC-LO-SRC-HI DST-LO-DST-HI ACTION, where the protocol is tcp or udp and SRC/DST are inclusive address ranges from 0 upward. Two rules match the same packet only if their protocols are equal and both the source ranges and the destination ranges overlap.
Rules:
0 udp 50-50 12-23 allow
1 udp 9-55 15-60 allow
2 tcp 57-58 9-45 deny
3 udp 30-52 11-15 allow
4 udp 12-48 32-59 allow
5 tcp 19-19 55-55 deny

Which is the FIRST rule (smallest index) that is fully shadowed by an earlier rule? Output its rule verbatim in the exact format INDEX PROTOCOL SRC-LO-SRC-HI DST-LO-DST-HI ACTION, or output None if no rule is fully shadowed.
The answer is the rule itself.

**Answer:**
4 udp 12-48 32-59 allow

## Level 2
### Example 1
**Prompt:**
An ordered firewall processes packets top to bottom: the first matching rule decides the action (allow or deny), later rules are ignored for that packet. A rule is fully shadowed when some strictly earlier rule matches every packet that it would match and has the same action, so the later rule can never affect any packet.
Each rule has the form: INDEX PROTOCOL SRC-LO-SRC-HI DST-LO-DST-HI ACTION, where the protocol is tcp or udp and SRC/DST are inclusive address ranges from 0 upward. Two rules match the same packet only if their protocols are equal and both the source ranges and the destination ranges overlap.
Rules:
0 tcp 209-244 54-152 deny
1 tcp 13-181 114-153 deny
2 udp 85-123 244-254 allow
3 udp 98-212 191-234 deny
4 tcp 116-156 250-252 deny
5 udp 194-230 110-146 allow
6 tcp 35-157 98-218 deny
7 tcp 140-220 57-216 deny
8 udp 149-240 26-215 allow
9 tcp 231-239 111-119 deny
10 tcp 134-238 12-193 deny
11 udp 251-254 229-236 deny

Which is the FIRST rule (smallest index) that is fully shadowed by an earlier rule? Output its rule verbatim in the exact format INDEX PROTOCOL SRC-LO-SRC-HI DST-LO-DST-HI ACTION, or output None if no rule is fully shadowed.
The answer is the rule itself.

**Answer:**
9 tcp 231-239 111-119 deny

### Example 2
**Prompt:**
An ordered firewall processes packets top to bottom: the first matching rule decides the action (allow or deny), later rules are ignored for that packet. A rule is fully shadowed when some strictly earlier rule matches every packet that it would match and has the same action, so the later rule can never affect any packet.
Each rule has the form: INDEX PROTOCOL SRC-LO-SRC-HI DST-LO-DST-HI ACTION, where the protocol is tcp or udp and SRC/DST are inclusive address ranges from 0 upward. Two rules match the same packet only if their protocols are equal and both the source ranges and the destination ranges overlap.
Rules:
0 udp 236-245 199-242 allow
1 tcp 126-191 79-213 deny
2 tcp 179-188 90-250 deny
3 udp 160-213 169-243 deny
4 udp 232-234 7-126 deny
5 udp 19-121 160-229 deny
6 udp 116-192 54-111 allow
7 tcp 18-95 203-240 allow
8 tcp 74-154 55-181 allow
9 tcp 156-164 70-118 deny
10 udp 79-227 137-175 deny
11 udp 236-238 207-220 allow

Which is the FIRST rule (smallest index) that is fully shadowed by an earlier rule? Output its rule verbatim in the exact format INDEX PROTOCOL SRC-LO-SRC-HI DST-LO-DST-HI ACTION, or output None if no rule is fully shadowed.
The answer is the rule itself.

**Answer:**
11 udp 236-238 207-220 allow

## Level 5
### Example 1
**Prompt:**
An ordered firewall processes packets top to bottom: the first matching rule decides the action (allow or deny), later rules are ignored for that packet. A rule is fully shadowed when some strictly earlier rule matches every packet that it would match and has the same action, so the later rule can never affect any packet.
Each rule has the form: INDEX PROTOCOL SRC-LO-SRC-HI DST-LO-DST-HI ACTION, where the protocol is tcp or udp and SRC/DST are inclusive address ranges from 0 upward. Two rules match the same packet only if their protocols are equal and both the source ranges and the destination ranges overlap.
Rules:
0 udp 194-337 198-383 deny
1 udp 410-491 78-283 deny
2 tcp 502-514 353-489 deny
3 tcp 186-302 404-421 deny
4 tcp 408-499 426-504 deny
5 udp 533-540 177-282 allow
6 udp 134-214 341-466 deny
7 udp 514-542 379-511 deny
8 udp 143-371 496-532 deny
9 udp 256-524 533-535 allow
10 udp 348-389 281-471 deny
11 udp 206-330 119-202 allow
12 udp 166-264 109-205 deny
13 tcp 478-480 445-481 deny
14 udp 3-263 502-539 deny
15 tcp 536-542 177-375 deny
16 tcp 101-138 261-425 deny
17 udp 489-524 76-196 allow
18 udp 275-509 36-144 deny
19 tcp 172-218 539-543 allow
20 udp 320-375 296-314 allow

Which is the FIRST rule (smallest index) that is fully shadowed by an earlier rule? Output its rule verbatim in the exact format INDEX PROTOCOL SRC-LO-SRC-HI DST-LO-DST-HI ACTION, or output None if no rule is fully shadowed.
The answer is the rule itself.

**Answer:**
13 tcp 478-480 445-481 deny

### Example 2
**Prompt:**
An ordered firewall processes packets top to bottom: the first matching rule decides the action (allow or deny), later rules are ignored for that packet. A rule is fully shadowed when some strictly earlier rule matches every packet that it would match and has the same action, so the later rule can never affect any packet.
Each rule has the form: INDEX PROTOCOL SRC-LO-SRC-HI DST-LO-DST-HI ACTION, where the protocol is tcp or udp and SRC/DST are inclusive address ranges from 0 upward. Two rules match the same packet only if their protocols are equal and both the source ranges and the destination ranges overlap.
Rules:
0 udp 294-309 44-162 deny
1 udp 12-24 143-407 deny
2 udp 16-291 511-517 allow
3 udp 67-268 384-459 deny
4 tcp 45-416 163-381 deny
5 tcp 264-281 125-454 deny
6 tcp 313-337 397-542 deny
7 udp 483-515 193-235 deny
8 udp 59-106 254-329 allow
9 tcp 130-401 440-531 deny
10 tcp 233-509 99-151 allow
11 tcp 171-288 464-536 allow
12 tcp 504-524 396-525 deny
13 tcp 213-409 149-158 allow
14 udp 228-327 2-256 deny
15 tcp 52-343 381-508 allow
16 udp 67-293 95-275 allow
17 udp 155-207 417-508 deny
18 udp 512-517 516-528 deny
19 tcp 280-394 507-530 deny
20 tcp 460-537 62-91 deny

Which is the FIRST rule (smallest index) that is fully shadowed by an earlier rule? Output its rule verbatim in the exact format INDEX PROTOCOL SRC-LO-SRC-HI DST-LO-DST-HI ACTION, or output None if no rule is fully shadowed.
The answer is the rule itself.

**Answer:**
19 tcp 280-394 507-530 deny
