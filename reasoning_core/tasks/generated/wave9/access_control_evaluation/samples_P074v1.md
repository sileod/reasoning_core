# Level 0

```
A system evaluates access under a role-based access control policy.
Groups and their explicit permissions (allow or deny) per action:
  - g0: write: allow, read: deny
  - g1: execute: deny
  - g2: (no explicit permissions)
Group membership edges (a member of the left side is also a member of the right side): g1->g0; g1->g2; g1->g2; g2->g1; u0->g0; u0->g2
Permissions are inherited transitively: any group the user reaches through membership edges confers its explicit permissions on the user.
The precedence policy is deny-overrides: under deny-overrides any explicit deny on a reached group blocks the action, and otherwise an allow grants it; under allow-overrides any allow grants it, and otherwise a deny blocks it. If no reached group states the action, the outcome is undefined.
The user is 'u0', and its direct memberships are given by the edges from that user.
For each action in the fixed order read, write, execute, state the effective outcome.
The answer is a single comma-separated list, one of allow/deny/undefined per action in the same order, e.g. 'undefined,allow,deny'. Nothing else.
```
Answer: deny,allow,deny

```
A system evaluates access under a role-based access control policy.
Groups and their explicit permissions (allow or deny) per action:
  - g0: (no explicit permissions)
  - g1: execute: allow, read: deny
  - g2: execute: deny, read: allow
Group membership edges (a member of the left side is also a member of the right side): g1->g0; g2->g0; g2->g0; g2->g1; g2->g1; u1->g2
Permissions are inherited transitively: any group the user reaches through membership edges confers its explicit permissions on the user.
The precedence policy is allow-overrides: under deny-overrides any explicit deny on a reached group blocks the action, and otherwise an allow grants it; under allow-overrides any allow grants it, and otherwise a deny blocks it. If no reached group states the action, the outcome is undefined.
The user is 'u1', and its direct memberships are given by the edges from that user.
For each action in the fixed order read, write, execute, state the effective outcome.
The answer is a single comma-separated list, one of allow/deny/undefined per action in the same order, e.g. 'undefined,allow,deny'. Nothing else.
```
Answer: allow,undefined,allow

# Level 2

```
A system evaluates access under a role-based access control policy.
Groups and their explicit permissions (allow or deny) per action:
  - g0: execute: deny
  - g1: delete: allow, write: deny, execute: allow
  - g2: read: deny, execute: deny, write: deny
  - g3: (no explicit permissions)
  - g4: read: allow, execute: deny
Group membership edges (a member of the left side is also a member of the right side): g0->g1; g0->g4; g1->g0; g1->g3; g3->g1; g3->g1; g3->g4; g4->g0; g4->g2; u0->g3
Permissions are inherited transitively: any group the user reaches through membership edges confers its explicit permissions on the user.
The precedence policy is allow-overrides: under deny-overrides any explicit deny on a reached group blocks the action, and otherwise an allow grants it; under allow-overrides any allow grants it, and otherwise a deny blocks it. If no reached group states the action, the outcome is undefined.
The user is 'u0', and its direct memberships are given by the edges from that user.
For each action in the fixed order read, write, execute, delete, state the effective outcome.
The answer is a single comma-separated list, one of allow/deny/undefined per action in the same order, e.g. 'undefined,allow,deny'. Nothing else.
```
Answer: allow,deny,allow,allow

```
A system evaluates access under a role-based access control policy.
Groups and their explicit permissions (allow or deny) per action:
  - g0: read: deny
  - g1: execute: allow, write: allow, read: allow
  - g2: execute: deny, write: allow, read: allow
  - g3: (no explicit permissions)
  - g4: write: allow, read: allow
Group membership edges (a member of the left side is also a member of the right side): g0->g2; g0->g2; g0->g3; g1->g2; g2->g3; g3->g1; g3->g2; u1->g0; u1->g2
Permissions are inherited transitively: any group the user reaches through membership edges confers its explicit permissions on the user.
The precedence policy is deny-overrides: under deny-overrides any explicit deny on a reached group blocks the action, and otherwise an allow grants it; under allow-overrides any allow grants it, and otherwise a deny blocks it. If no reached group states the action, the outcome is undefined.
The user is 'u1', and its direct memberships are given by the edges from that user.
For each action in the fixed order read, write, execute, delete, state the effective outcome.
The answer is a single comma-separated list, one of allow/deny/undefined per action in the same order, e.g. 'undefined,allow,deny'. Nothing else.
```
Answer: deny,allow,deny,undefined

# Level 5

```
A system evaluates access under a role-based access control policy.
Groups and their explicit permissions (allow or deny) per action:
  - g0: read: deny, write: allow
  - g1: write: allow, delete: allow, execute: allow
  - g2: execute: allow, read: deny, write: deny
  - g3: edit: allow
  - g4: execute: deny
  - g5: (no explicit permissions)
  - g6: (no explicit permissions)
  - g7: edit: allow, delete: allow
Group membership edges (a member of the left side is also a member of the right side): g0->g2; g1->g0; g1->g2; g1->g2; g1->g3; g2->g0; g3->g2; g3->g4; g3->g4; g3->g6; g5->g0; g6->g1; g6->g1; u0->g1; u0->g4
Permissions are inherited transitively: any group the user reaches through membership edges confers its explicit permissions on the user.
The precedence policy is allow-overrides: under deny-overrides any explicit deny on a reached group blocks the action, and otherwise an allow grants it; under allow-overrides any allow grants it, and otherwise a deny blocks it. If no reached group states the action, the outcome is undefined.
The user is 'u0', and its direct memberships are given by the edges from that user.
For each action in the fixed order read, write, execute, delete, edit, state the effective outcome.
The answer is a single comma-separated list, one of allow/deny/undefined per action in the same order, e.g. 'undefined,allow,deny'. Nothing else.
```
Answer: deny,allow,allow,allow,allow

```
A system evaluates access under a role-based access control policy.
Groups and their explicit permissions (allow or deny) per action:
  - g0: write: deny, execute: deny, read: deny, delete: deny
  - g1: (no explicit permissions)
  - g2: delete: allow
  - g3: read: deny
  - g4: read: allow
  - g5: (no explicit permissions)
  - g6: write: deny, read: deny, edit: allow
  - g7: execute: allow, delete: allow
Group membership edges (a member of the left side is also a member of the right side): g0->g3; g0->g7; g1->g3; g2->g3; g4->g0; g4->g2; g4->g6; g5->g7; g6->g3; g6->g3; g6->g7; g7->g1; g7->g4; u0->g5
Permissions are inherited transitively: any group the user reaches through membership edges confers its explicit permissions on the user.
The precedence policy is allow-overrides: under deny-overrides any explicit deny on a reached group blocks the action, and otherwise an allow grants it; under allow-overrides any allow grants it, and otherwise a deny blocks it. If no reached group states the action, the outcome is undefined.
The user is 'u0', and its direct memberships are given by the edges from that user.
For each action in the fixed order read, write, execute, delete, edit, state the effective outcome.
The answer is a single comma-separated list, one of allow/deny/undefined per action in the same order, e.g. 'undefined,allow,deny'. Nothing else.
```
Answer: allow,deny,allow,allow,allow
