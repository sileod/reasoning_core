import random
from dataclasses import dataclass

from reasoning_core.template import Config, Entry, Task, edict, stochastic_rounding as sround

PERM_CHARS = "rwx"

TASK_META = {'parent_source_id': None,
 'idea': 'posix_acl_permission (draw 2 of 2)',
 'hypothesis': 'W1-048',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/posix_acl_permission',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 737652980,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _fmt(s):
    return "".join(c for c in PERM_CHARS if c in s) or "-"


def _interp(perm):
    if perm in ("", "-"):
        return set()
    return set(perm)


def _rand_bit():
    bits = {c for c in PERM_CHARS if random.random() < 0.5}
    return "".join(sorted(bits)) or "-"


def effective_perm(owner, owner_perm, named_users, owning_group, owngroup_perm,
                   named_groups, group_order, mask, other, subject, subject_groups):
    """POSIX ACL access check for one subject, returns the effective rwx string.

    Priority order: owner -> named user -> owning-group membership -> first named
    group in listed order -> other. The mask caps named-user, owning-group and
    named-group entries but never the owner or other entries.
    """
    if subject == owner:
        perms = _interp(owner_perm)
    elif subject in named_users:
        perms = _interp(named_users[subject]) & _interp(mask)
    elif owning_group in subject_groups:
        perms = _interp(owngroup_perm) & _interp(mask)
    else:
        matched = None
        for g in group_order:
            if g in subject_groups and g in named_groups:
                matched = g
                break
        if matched is not None:
            perms = _interp(named_groups[matched]) & _interp(mask)
        else:
            perms = _interp(other)
    return _fmt(perms)


@dataclass
class PosixAclPermissionV2Config(Config):
    n_subjects: int = 3
    n_named_users: int = 2
    n_named_groups: int = 2
    max_attempts: int = 200

    def apply_difficulty(self, level):
        self.n_subjects = sround(self.n_subjects + 0.6 * level)
        self.n_named_users = sround(self.n_named_users + 0.5 * level)
        self.n_named_groups = sround(self.n_named_groups + 0.5 * level)
        self.max_attempts = sround(self.max_attempts + 40 * level)


class PosixAclPermission(Task):
    summary = ("Given POSIX ACL owner/named-user/owning-group/named-group/mask/other "
               "entries and each subject's groups, output each subject's effective "
               "permission set in a fixed order.")
    config_cls = PosixAclPermissionV2Config
    task_version = 2

    def generate_entry(self):
        cfg = self.config
        users = ["alice", "bob", "carol", "dave", "erin", "frank",
                 "grace", "heidi", "ivan", "judy"]
        group_pool = ["staff", "dev", "ops", "test", "core", "infra", "sec", "lab"]

        for _ in range(cfg.max_attempts):
            owner = random.choice(users)
            candidates = [u for u in users if u != owner]
            n_subj = max(2, cfg.n_subjects)
            subjects = sorted(random.sample(candidates, n_subj))

            owner_perm = _rand_bit()
            other = _rand_bit()
            # mask must keep at least one bit or every masked entry collapses.
            mask_bits = {c for c in PERM_CHARS if random.random() < 0.6}
            if not mask_bits:
                mask_bits.add(random.choice(PERM_CHARS))
            mask = "".join(sorted(mask_bits))

            named_users = {}
            user_names = [u for u in candidates if u not in subjects]
            for u in random.sample(user_names, min(cfg.n_named_users, len(user_names))):
                named_users[u] = _rand_bit()

            owning_group = random.choice(group_pool)
            owngroup_perm = _rand_bit()
            named_groups = {}
            ng_pool = [g for g in group_pool if g != owning_group]
            for g in random.sample(ng_pool, min(cfg.n_named_groups, len(ng_pool))):
                named_groups[g] = _rand_bit()
            group_order = sorted(named_groups.keys())

            subject_groups = {}
            for s in subjects:
                picks = random.sample(group_pool, random.randint(0, 3))
                subject_groups[s] = sorted(picks)

            perms = []
            for s in subjects:
                perms.append(effective_perm(
                    owner, owner_perm, named_users, owning_group, owngroup_perm,
                    named_groups, group_order, mask, other, s,
                    set(subject_groups[s])))

            distinct = len(set(perms))
            need = min(3, n_subj)
            if distinct < need:
                continue

            metadata = edict({
                "owner": owner,
                "owner_perm": owner_perm,
                "named_users": named_users,
                "owning_group": owning_group,
                "owngroup_perm": owngroup_perm,
                "named_groups": named_groups,
                "mask": mask,
                "other": other,
                "subjects": [(s, tuple(subject_groups[s])) for s in subjects],
                "perms": perms,
            })
            metadata.payload = {
                "owner": owner,
                "named users": {k: named_users[k] for k in sorted(named_users)},
                "owning group": owning_group,
                "named groups": {k: named_groups[k] for k in sorted(named_groups)},
                "mask": mask,
                "other": other,
                "subjects": {
                    s: "{" + ",".join(subject_groups[s]) + "}"
                    for s in subjects
                },
            }
            answer = ";".join(perms)
            return Entry(metadata=metadata, answer=answer)
        raise RuntimeError("Failed to generate a valid ACL instance")

    def render_prompt(self, metadata):
        lines = []
        lines.append("POSIX ACL permission check for a single file.")
        lines.append("")
        lines.append(f"Owner: {metadata.owner}  (owner entry: {metadata.owner_perm})")
        nu = "; ".join(f"{u}:{p}" for u, p in metadata.named_users.items())
        lines.append(f"Named user entries: {nu or '(none)'}")
        lines.append(f"Owning group: {metadata.owning_group}  "
                     f"(owning-group entry: {metadata.owngroup_perm})")
        ng = "; ".join(f"{g}:{p}" for g, p in metadata.named_groups.items())
        lines.append(f"Named group entries: {ng or '(none)'}")
        lines.append(f"Mask: {metadata.mask}")
        lines.append(f"Other: {metadata.other}")
        lines.append("")
        lines.append("Subjects and the groups each belongs to:")
        for s, groups in metadata.subjects:
            lines.append(f"  {s}: {{{','.join(groups)}}}" if groups else f"  {s}: none")
        lines.append("")
        lines.append("The check, in priority order: if the subject is the owner, the "
                     "effective set is the owner entry, subject to no mask. Otherwise, "
                     "if a named user entry matches the subject, the effective set is "
                     "that entry intersected with the mask. Else if the subject is a "
                     "member of the owning group, it is the owning-group entry "
                     "intersected with the mask. Else if a named group entry matches "
                     "(first in listed order), it is that entry intersected with the "
                     "mask. Otherwise the effective set is the other entry.")
        lines.append("")
        lines.append("For each subject in the order listed above, give its effective "
                     "permission set, joining the results with ';' (use '-' for no "
                     "permission bits). For example the answer format looks like "
                     "r-x;r;rwx.")
        lines.append("The answer is that single joined string.")
        return "\n".join(lines)

    def score_answer(self, answer, entry):
        try:
            return float(str(answer).strip() == str(entry.answer).strip())
        except Exception:
            return 0.0
