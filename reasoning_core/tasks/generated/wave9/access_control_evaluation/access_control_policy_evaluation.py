from dataclasses import dataclass
import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'access_control_policy_evaluation (draw 1 of 1)',
 'hypothesis': 'HV-074',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/access_control_policy_evaluation',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3220316931,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class AccessControlConfig(Config):
    n_groups: int = 3
    n_perms: int = 3
    memberships: int = 5
    rules: int = 5
    n_users: int = 2

    def apply_difficulty(self, level):
        self.n_groups = sround(self.n_groups + level)
        self.n_perms = sround(self.n_perms + level // 2)
        self.memberships = sround(self.memberships + level * 2)
        self.rules = sround(self.rules + level * 2)


ACTS = ["read", "write", "execute", "delete", "edit", "list", "append", "share"]


def _traverse(user, memberships, groups):
    direct = set()
    for (u, g) in memberships:
        if u == user and g in groups:
            direct.add(g)
    reached = set(direct)
    stack = list(direct)
    while stack:
        cur = stack.pop()
        for (a, b) in memberships:
            if a == cur and b in groups and b not in reached:
                reached.add(b)
                stack.append(b)
    return reached


def _decision(groups, user, perm, memberships, precedence):
    reached = _traverse(user, memberships, groups)
    allow = False
    deny = False
    for g in reached:
        r = groups[g].get(perm)
        if r == "allow":
            allow = True
        elif r == "deny":
            deny = True
    if precedence == "deny-overrides":
        if deny:
            return "deny"
        return "allow" if allow else "undefined"
    else:
        if allow:
            return "allow"
        if deny:
            return "deny"
        return "undefined"


class AccessControlEvaluation(Task):
    summary = "Resolve inherited allow and deny permissions, group membership, and explicit exceptions under a stated precedence policy, returning the effective outcome for every action as a sorted list."
    config_cls = AccessControlConfig

    def generate_entry(self):
        cfg = self.config
        n_groups = max(2, int(cfg.n_groups))
        n_perms = int(cfg.n_perms)
        acts = ACTS[:max(3, min(n_perms, len(ACTS)))]
        gnames = [f"g{i}" for i in range(n_groups)]
        unames = [f"u{i}" for i in range(int(cfg.n_users))]

        while True:
            groups = {}
            for g in gnames:
                entry = {}
                k = random.randint(0, len(acts) - 1)
                for p in random.sample(acts, k=k):
                    entry[p] = random.choice(["allow", "deny"])
                groups[g] = entry

            memberships = []
            for _ in range(int(cfg.memberships)):
                a = random.choice(gnames)
                b = random.choice(gnames)
                if a != b:
                    memberships.append((a, b))

            user = random.choice(unames)
            for (u, g) in memberships:
                pass
            own_edges = [(u, g) for (u, g) in memberships if u == user and g in gnames]
            for g in random.sample(gnames, k=random.randint(1, 2)):
                if g not in [gg for (_, gg) in own_edges]:
                    memberships.append((user, g))

            precedence = random.choice(["deny-overrides", "allow-overrides"])

            eff = [_decision(groups, user, p, memberships, precedence) for p in acts]
            if "allow" in eff or "deny" in eff:
                break

        answer = ",".join(eff)

        metadata = edict({
            "groups": {k: dict(v) for k, v in sorted(groups.items())},
            "memberships": sorted(memberships),
            "user": user,
            "acts": list(acts),
            "precedence": precedence,
            "eff": eff,
        })
        metadata.payload = {
            "groups": {k: dict(v) for k, v in sorted(groups.items())},
            "memberships": sorted(memberships),
            "user": user,
            "actions": list(acts),
            "precedence": precedence,
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        lines = []
        lines.append("A system evaluates access under a role-based access control policy.")
        lines.append("Groups and their explicit permissions (allow or deny) per action:")
        for g, d in metadata.groups.items():
            if not d:
                lines.append(f"  - {g}: (no explicit permissions)")
            else:
                parts = ", ".join(f"{p}: {v}" for p, v in d.items())
                lines.append(f"  - {g}: {parts}")
        if metadata.memberships:
            mb = "; ".join(f"{a}->{b}" for a, b in metadata.memberships)
            lines.append(f"Group membership edges (a member of the left side is also a member of the right side): {mb}")
        else:
            lines.append("Group membership edges: (none)")
        lines.append(f"Permissions are inherited transitively: any group the user reaches through membership edges "
                     f"confers its explicit permissions on the user.")
        lines.append(f"The precedence policy is {metadata.precedence}: under deny-overrides any explicit deny on a reached "
                     f"group blocks the action, and otherwise an allow grants it; under allow-overrides any allow grants it, "
                     f"and otherwise a deny blocks it. If no reached group states the action, the outcome is undefined.")
        lines.append(f"The user is '{metadata.user}', and its direct memberships are given by the edges from that user.")
        lines.append(f"For each action in the fixed order {', '.join(metadata.acts)}, state the effective outcome.")
        lines.append("The answer is a single comma-separated list, one of allow/deny/undefined per action in the same "
                     "order, e.g. 'undefined,allow,deny'. Nothing else.")
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        a = answer.strip()
        parts = [p.strip().lower() for p in a.split(",")]
        gold = [p for p in entry.answer.split(",")]
        if parts == gold:
            return 1.0
        return 0.0
