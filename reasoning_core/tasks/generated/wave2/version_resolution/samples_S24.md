# Level 0

## Example 1

### Prompt

Packages:
Available versions:
http: 2.0.0, 2.0.3, 2.0.9
json: 5.0.0, 5.1.2, 5.1.3, 5.1.5
xml: 3.0.0, 3.0.5, 3.0.9

Requires:
Each package version declares ranges on packages listed earlier (resolution order is the package order):
json 5.0.0: requires http == 2.0.9
json 5.1.2: requires http > 2.0.9
json 5.1.3: requires http > 2.0.9
json 5.1.5: requires http > 2.0.9
xml 3.0.0: requires json >= 5.0.0, < 5.1.3
xml 3.0.5: requires json >= 5.0.0, <= 5.5.9
xml 3.0.9: requires json > 5.0.0

Notation:
Semantic ranges: >=, >, <=, <, ==; ^a.b.c means at least a.b.c and below the next major (e.g. ^1.2.0 -> >=1.2.0 and <2.0.0); ~a.b.c means at least a.b.c and below the next minor. A comma lists both bounds.

A resolver picks versions by processing packages one at a time in resolution order (http, json, xml); for each package it chooses the highest available version whose declared requirement ranges are all satisfied by the versions already picked for the packages resolved before it. Package xml is resolved last.

Which version of xml is picked?

The answer is a version string.

### Answer

3.0.5

## Example 2

### Prompt

Packages:
Available versions:
http: 2.0.0, 2.1.1, 2.1.3, 4.1.4
json: 4.0.0, 5.1.1, 6.0.1
xml: 4.0.0, 4.0.1, 5.2.3

Requires:
Each package version declares ranges on packages listed earlier (resolution order is the package order):
json 4.0.0: requires http >= 2.1.3
json 5.1.1: requires http >= 2.1.3
json 6.0.1: requires http ^4.1.4
xml 4.0.0: requires http >= 2.0.0
xml 4.0.1: requires http >= 4.1.4, <= 4.6.9
xml 5.2.3: requires http < 4.1.4

Notation:
Semantic ranges: >=, >, <=, <, ==; ^a.b.c means at least a.b.c and below the next major (e.g. ^1.2.0 -> >=1.2.0 and <2.0.0); ~a.b.c means at least a.b.c and below the next minor. A comma lists both bounds.

A resolver picks versions by processing packages one at a time in resolution order (http, json, xml); for each package it chooses the highest available version whose declared requirement ranges are all satisfied by the versions already picked for the packages resolved before it. Package xml is resolved last.

Which version of xml is picked?

The answer is a version string.

### Answer

4.0.1



# Level 2

## Example 1

### Prompt

Packages:
Available versions:
http: 2.0.0, 2.0.5, 4.1.4
json: 2.0.0, 4.0.2, 4.0.5, 4.0.7
xml: 1.0.0, 1.0.1, 1.0.6, 1.1.3, 1.1.6
sql: 3.0.0, 3.0.3, 3.3.0
core: 2.0.0, 4.0.2, 5.1.1, 5.1.6, 7.2.3

Requires:
Each package version declares ranges on packages listed earlier (resolution order is the package order):
json 2.0.0: requires http >= 2.0.5
json 4.0.2: requires http >= 4.1.4, <= 4.6.9
json 4.0.5: requires http < 4.1.4
json 4.0.7: requires http < 4.1.4
xml 1.0.0: requires json == 4.0.2
xml 1.0.1: requires json > 4.0.2
xml 1.0.6: requires json > 4.0.2
xml 1.1.3: requires json > 4.0.2
xml 1.1.6: requires json > 4.0.2
sql 3.0.0: requires json >= 4.0.2
sql 3.0.3: requires json >= 2.0.0
sql 3.3.0: requires json >= 4.0.2
core 2.0.0: requires http >= 2.0.5
core 4.0.2: requires http >= 2.0.5
core 5.1.1: requires http >= 2.0.0, < 2.0.5
core 5.1.6: requires http == 4.1.4
core 7.2.3: requires http < 4.1.4

Notation:
Semantic ranges: >=, >, <=, <, ==; ^a.b.c means at least a.b.c and below the next major (e.g. ^1.2.0 -> >=1.2.0 and <2.0.0); ~a.b.c means at least a.b.c and below the next minor. A comma lists both bounds.

A resolver picks versions by processing packages one at a time in resolution order (http, json, xml, sql, core); for each package it chooses the highest available version whose declared requirement ranges are all satisfied by the versions already picked for the packages resolved before it. Package core is resolved last.

Which version of core is picked?

The answer is a version string.

### Answer

5.1.6

## Example 2

### Prompt

Packages:
Available versions:
http: 3.0.0, 3.3.2, 3.4.0, 5.0.3
json: 2.0.0, 3.0.2, 3.2.1
xml: 4.0.0, 6.0.3, 6.2.4
sql: 5.0.0, 6.0.0, 6.0.2, 6.0.5, 6.2.2
core: 3.0.0, 3.2.0, 3.5.2, 3.5.6

Requires:
Each package version declares ranges on packages listed earlier (resolution order is the package order):
json 2.0.0: requires http == 3.3.2
json 3.0.2: requires http >= 3.0.0
json 3.2.1: requires http ^5.0.3
xml 4.0.0: requires http == 5.0.3
xml 6.0.3: requires http == 5.0.3
xml 6.2.4: requires http ~5.0.3
sql 5.0.0: requires xml == 6.2.4
sql 6.0.0: requires xml < 6.2.4
sql 6.0.2: requires xml < 6.2.4
sql 6.0.5: requires xml < 6.2.4
sql 6.2.2: requires xml > 6.2.4
core 3.0.0: requires xml >= 6.0.3
core 3.2.0: requires xml >= 6.2.4
core 3.5.2: requires xml < 6.2.4
core 3.5.6: requires xml < 6.2.4

Notation:
Semantic ranges: >=, >, <=, <, ==; ^a.b.c means at least a.b.c and below the next major (e.g. ^1.2.0 -> >=1.2.0 and <2.0.0); ~a.b.c means at least a.b.c and below the next minor. A comma lists both bounds.

A resolver picks versions by processing packages one at a time in resolution order (http, json, xml, sql, core); for each package it chooses the highest available version whose declared requirement ranges are all satisfied by the versions already picked for the packages resolved before it. Package core is resolved last.

Which version of core is picked?

The answer is a version string.

### Answer

3.2.0



# Level 5

## Example 1

### Prompt

Packages:
Available versions:
http: 4.0.0, 4.0.3, 6.0.4, 7.2.1
json: 3.0.0, 3.3.3, 4.0.4
xml: 1.0.0, 2.2.2, 3.2.1, 4.1.4
sql: 1.0.0, 1.0.2, 2.1.2
core: 2.0.0, 2.0.3, 3.0.1, 3.0.2, 4.1.3, 5.2.4
util: 1.0.0, 1.2.2, 1.3.0, 1.3.6, 1.4.1, 1.7.1
io: 2.0.0, 3.1.3, 3.4.4, 3.6.3, 3.8.0, 4.2.2
log: 4.0.0, 6.0.0, 6.0.1

Requires:
Each package version declares ranges on packages listed earlier (resolution order is the package order):
json 3.0.0: requires http >= 7.2.1
json 3.3.3: requires http > 7.2.1
json 4.0.4: requires http > 7.2.1
xml 1.0.0: requires json ~3.0.0
xml 2.2.2: requires json > 3.0.0
xml 3.2.1: requires json > 3.0.0
xml 4.1.4: requires json > 3.0.0
sql 1.0.0: requires http >= 6.0.4, < 7.2.1; requires xml >= 3.2.1
sql 1.0.2: requires http ~7.2.1; requires xml == 1.0.0
sql 2.1.2: requires http > 7.2.1; requires xml >= 3.2.1
core 2.0.0: requires http == 4.0.0; requires xml >= 2.2.2, < 4.1.4
core 2.0.3: requires http == 6.0.4; requires xml == 3.2.1
core 3.0.1: requires http >= 4.0.0, < 4.0.3; requires xml >= 2.2.2, < 3.2.1
core 3.0.2: requires http >= 4.0.3; requires xml >= 2.2.2, < 3.2.1
core 4.1.3: requires http == 6.0.4; requires xml == 1.0.0
core 5.2.4: requires http ~7.2.1; requires xml ^1.0.0
util 1.0.0: requires xml >= 1.0.0
util 1.2.2: requires xml >= 1.0.0
util 1.3.0: requires xml > 1.0.0
util 1.3.6: requires xml > 1.0.0
util 1.4.1: requires xml > 1.0.0
util 1.7.1: requires xml > 1.0.0
io 2.0.0: requires http >= 4.0.3; requires xml == 1.0.0; requires core >= 2.0.0, < 5.2.4
io 3.1.3: requires http >= 4.0.0; requires xml >= 1.0.0; requires core >= 2.0.3
io 3.4.4: requires http >= 4.0.3, < 7.2.1; requires xml >= 2.2.2; requires core >= 4.1.3, < 5.2.4
io 3.6.3: requires http >= 6.0.4; requires xml >= 1.0.0, < 4.1.4; requires core >= 5.2.4
io 3.8.0: requires http == 7.2.1; requires xml >= 3.2.1, < 4.1.4; requires core >= 3.0.2
io 4.2.2: requires http ~7.2.1; requires xml >= 1.0.0, <= 1.5.9; requires core ^5.2.4
log 4.0.0: requires util >= 1.3.6, < 1.7.1
log 6.0.0: requires util >= 1.2.2, <= 1.7.9
log 6.0.1: requires util < 1.2.2

Notation:
Semantic ranges: >=, >, <=, <, ==; ^a.b.c means at least a.b.c and below the next major (e.g. ^1.2.0 -> >=1.2.0 and <2.0.0); ~a.b.c means at least a.b.c and below the next minor. A comma lists both bounds.

A resolver picks versions by processing packages one at a time in resolution order (http, json, xml, sql, core, util, io, log); for each package it chooses the highest available version whose declared requirement ranges are all satisfied by the versions already picked for the packages resolved before it. Package log is resolved last.

Which version of log is picked?

The answer is a version string.

### Answer

6.0.0

## Example 2

### Prompt

Packages:
Available versions:
http: 1.0.0, 2.1.0, 2.1.5, 2.1.8, 2.2.0, 2.3.3
json: 2.0.0, 2.0.4, 4.0.0, 4.3.0, 4.3.5, 4.5.3
xml: 1.0.0, 1.3.0, 1.5.1
sql: 4.0.0, 4.0.2, 6.0.2, 8.1.3
core: 2.0.0, 2.0.4, 2.2.1
util: 2.0.0, 2.2.2, 2.2.4
io: 3.0.0, 3.2.0, 4.2.1
log: 4.0.0, 4.0.4, 4.3.2, 4.3.6

Requires:
Each package version declares ranges on packages listed earlier (resolution order is the package order):
json 2.0.0: requires http == 2.1.8
json 2.0.4: requires http >= 2.1.0, < 2.3.3
json 4.0.0: requires http >= 2.3.3, <= 2.8.9
json 4.3.0: requires http > 2.3.3
json 4.3.5: requires http > 2.3.3
json 4.5.3: requires http > 2.3.3
xml 1.0.0: requires http >= 2.1.8, < 2.3.3; requires json >= 4.0.0
xml 1.3.0: requires http >= 2.1.0, < 2.2.0; requires json >= 2.0.0
xml 1.5.1: requires http ^2.3.3; requires json >= 4.0.0
sql 4.0.0: requires http >= 2.1.8; requires json == 4.3.0; requires xml >= 1.0.0
sql 4.0.2: requires http >= 2.3.3; requires json ~4.0.0; requires xml ^1.5.1
sql 6.0.2: requires http < 2.3.3; requires json >= 4.0.0; requires xml == 1.5.1
sql 8.1.3: requires http < 2.3.3; requires json >= 4.3.5; requires xml == 1.0.0
core 2.0.0: requires http ^2.3.3; requires json >= 4.0.0; requires sql >= 4.0.2, <= 4.5.9
core 2.0.4: requires http > 2.3.3; requires json >= 4.3.0, < 4.5.3; requires sql >= 4.0.0, < 4.0.2
core 2.2.1: requires http > 2.3.3; requires json >= 2.0.0; requires sql >= 4.0.0, < 6.0.2
util 2.0.0: requires json == 4.3.0
util 2.2.2: requires json == 4.3.5
util 2.2.4: requires json ~4.0.0
io 3.0.0: requires http >= 2.1.8; requires core >= 2.0.0, < 2.0.4
io 3.2.0: requires http >= 2.1.0, < 2.1.5; requires core >= 2.0.4
io 4.2.1: requires http ^2.3.3; requires core ~2.0.0
log 4.0.0: requires xml >= 1.0.0, < 1.3.0; requires util >= 2.2.2
log 4.0.4: requires xml >= 1.0.0, < 1.3.0; requires util == 2.2.4
log 4.3.2: requires xml ^1.5.1; requires util ~2.2.4
log 4.3.6: requires xml < 1.5.1; requires util >= 2.2.4

Notation:
Semantic ranges: >=, >, <=, <, ==; ^a.b.c means at least a.b.c and below the next major (e.g. ^1.2.0 -> >=1.2.0 and <2.0.0); ~a.b.c means at least a.b.c and below the next minor. A comma lists both bounds.

A resolver picks versions by processing packages one at a time in resolution order (http, json, xml, sql, core, util, io, log); for each package it chooses the highest available version whose declared requirement ranges are all satisfied by the versions already picked for the packages resolved before it. Package log is resolved last.

Which version of log is picked?

The answer is a version string.

### Answer

4.3.2


