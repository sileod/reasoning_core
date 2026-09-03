## Level 0

### Prompt

POSIX ACL permission check for a single file.

Owner: dave  (owner entry: r)
Named user entries: ivan:w; judy:wx
Owning group: test  (owning-group entry: wx)
Named group entries: dev:wx; infra:rw
Mask: r
Other: rw

Subjects and the groups each belongs to:
  alice: {core}
  erin: {dev,sec}
  frank: {infra}

The check, in priority order: if the subject is the owner, the effective set is the owner entry, subject to no mask. Otherwise, if a named user entry matches the subject, the effective set is that entry intersected with the mask. Else if the subject is a member of the owning group, it is the owning-group entry intersected with the mask. Else if a named group entry matches (first in listed order), it is that entry intersected with the mask. Otherwise the effective set is the other entry.

For each subject in the order listed above, give its effective permission set, joining the results with ';' (use '-' for no permission bits). For example the answer format looks like r-x;r;rwx.
The answer is that single joined string.

### Answer

rw;-;r

### Prompt

POSIX ACL permission check for a single file.

Owner: grace  (owner entry: rx)
Named user entries: erin:rwx; ivan:x
Owning group: lab  (owning-group entry: x)
Named group entries: infra:-; test:x
Mask: rwx
Other: w

Subjects and the groups each belongs to:
  carol: {core,infra,sec}
  frank: {sec}
  heidi: {ops,test}

The check, in priority order: if the subject is the owner, the effective set is the owner entry, subject to no mask. Otherwise, if a named user entry matches the subject, the effective set is that entry intersected with the mask. Else if the subject is a member of the owning group, it is the owning-group entry intersected with the mask. Else if a named group entry matches (first in listed order), it is that entry intersected with the mask. Otherwise the effective set is the other entry.

For each subject in the order listed above, give its effective permission set, joining the results with ';' (use '-' for no permission bits). For example the answer format looks like r-x;r;rwx.
The answer is that single joined string.

### Answer

-;w;x

## Level 2

### Prompt

POSIX ACL permission check for a single file.

Owner: heidi  (owner entry: x)
Named user entries: frank:x; carol:x; grace:rwx
Owning group: core  (owning-group entry: rwx)
Named group entries: infra:-; staff:wx; dev:w
Mask: rw
Other: x

Subjects and the groups each belongs to:
  bob: none
  dave: {staff,test}
  ivan: none
  judy: {core,sec,staff}

The check, in priority order: if the subject is the owner, the effective set is the owner entry, subject to no mask. Otherwise, if a named user entry matches the subject, the effective set is that entry intersected with the mask. Else if the subject is a member of the owning group, it is the owning-group entry intersected with the mask. Else if a named group entry matches (first in listed order), it is that entry intersected with the mask. Otherwise the effective set is the other entry.

For each subject in the order listed above, give its effective permission set, joining the results with ';' (use '-' for no permission bits). For example the answer format looks like r-x;r;rwx.
The answer is that single joined string.

### Answer

x;w;x;rw

### Prompt

POSIX ACL permission check for a single file.

Owner: dave  (owner entry: r)
Named user entries: heidi:x; bob:r; erin:rwx
Owning group: staff  (owning-group entry: rwx)
Named group entries: sec:x; core:w; infra:wx
Mask: rw
Other: -

Subjects and the groups each belongs to:
  carol: {infra,sec,staff}
  grace: {infra,lab}
  ivan: none
  judy: none

The check, in priority order: if the subject is the owner, the effective set is the owner entry, subject to no mask. Otherwise, if a named user entry matches the subject, the effective set is that entry intersected with the mask. Else if the subject is a member of the owning group, it is the owning-group entry intersected with the mask. Else if a named group entry matches (first in listed order), it is that entry intersected with the mask. Otherwise the effective set is the other entry.

For each subject in the order listed above, give its effective permission set, joining the results with ';' (use '-' for no permission bits). For example the answer format looks like r-x;r;rwx.
The answer is that single joined string.

### Answer

rw;w;-;-

## Level 5

### Prompt

POSIX ACL permission check for a single file.

Owner: alice  (owner entry: r)
Named user entries: carol:x; erin:w; ivan:rw
Owning group: core  (owning-group entry: rx)
Named group entries: dev:wx; ops:rw; staff:rwx; infra:rw
Mask: rx
Other: x

Subjects and the groups each belongs to:
  bob: {infra}
  dave: {dev}
  frank: none
  grace: {dev}
  heidi: {lab,staff,test}
  judy: {dev,sec}

The check, in priority order: if the subject is the owner, the effective set is the owner entry, subject to no mask. Otherwise, if a named user entry matches the subject, the effective set is that entry intersected with the mask. Else if the subject is a member of the owning group, it is the owning-group entry intersected with the mask. Else if a named group entry matches (first in listed order), it is that entry intersected with the mask. Otherwise the effective set is the other entry.

For each subject in the order listed above, give its effective permission set, joining the results with ';' (use '-' for no permission bits). For example the answer format looks like r-x;r;rwx.
The answer is that single joined string.

### Answer

r;x;x;x;rx;x

### Prompt

POSIX ACL permission check for a single file.

Owner: heidi  (owner entry: rwx)
Named user entries: carol:x; bob:rwx; erin:rw
Owning group: ops  (owning-group entry: rwx)
Named group entries: infra:w; sec:rx; test:x; staff:r
Mask: w
Other: rwx

Subjects and the groups each belongs to:
  alice: {ops,test}
  dave: {core,test}
  frank: {lab,sec,test}
  grace: {core}
  ivan: {ops}
  judy: {infra,ops}

The check, in priority order: if the subject is the owner, the effective set is the owner entry, subject to no mask. Otherwise, if a named user entry matches the subject, the effective set is that entry intersected with the mask. Else if the subject is a member of the owning group, it is the owning-group entry intersected with the mask. Else if a named group entry matches (first in listed order), it is that entry intersected with the mask. Otherwise the effective set is the other entry.

For each subject in the order listed above, give its effective permission set, joining the results with ';' (use '-' for no permission bits). For example the answer format looks like r-x;r;rwx.
The answer is that single joined string.

### Answer

w;-;-;rwx;w;w

