"""Unix mode permission: decide which users are granted a requested access.

Given a file's octal mode, owner (uid/gid) and a list of users (each with a uid and
a set of group memberships), the task asks which users can perform a requested
access (read/write/execute). The classic Unix precedence rule decides: effective
owner applies to the file's owner; otherwise a user whose groups contain the file's
gid gets the group bits; otherwise everybody gets the "other" bits.

The owner here is a distinct system identity (no generated user equals it), so the
interesting branch is group membership versus other, which is exactly the reasoning
the data should exercise. The answer is the sorted list of granted user indices (or
"none"), which varies widely across examples and carries the witness.
"""

import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload

ACCESS_VAL = {"read": 4, "write": 2, "execute": 1}


@dataclass
class UnixModePermissionConfig(Config):
    n_users: int = 4
    n_groups: int = 5

    def apply_difficulty(self, level):
        self.n_users = 8 + 2 * level
        self.n_groups = 8 + level


def user_granted(mode, owner_gid, user_uid, user_gids, access):
    """Return True if ``user`` (as a requester process) is granted ``access``.

    mode is an octal int including special bits; only the rwx class bits that apply
    are consulted. uid 0 (root) is never generated, so no special casing is needed.
    """
    val = ACCESS_VAL[access]
    if owner_gid in user_gids:
        shift = 3
    else:
        shift = 0
    return (mode >> shift) & val != 0


def parse_granted(text):
    """Parse a canonical answer. Raises ValueError on empty/non-numeric input."""
    if text is None:
        raise ValueError
    text = str(text).strip()
    if text == "":
        raise ValueError
    if text.lower() == "none":
        return ()
    try:
        return tuple(int(p) for p in text.split())
    except ValueError:
        raise ValueError


def render_users(users):
    return "\n".join(
        "%d. uid=%d; groups=%s" % (i, u["uid"], sorted(u["groups"]))
        for i, u in enumerate(users, start=1)
    )


def format_mode(mode):
    return "0%03o (owner=%s group=%s other=%s)" % (
        mode,
        "rwx" if mode & 0o700 == 0o700 else chars(mode, 6),
        chars(mode, 3),
        chars(mode, 0),
    )


def chars(mode, shift):
    v = (mode >> shift) & 0o7
    return "".join(c for c, bit in (("r", 4), ("w", 2), ("x", 1)) if v & bit != 0) or "-"


class UnixModePermission(Task):
    summary = ("Given Unix mode metadata and user identity, decide which users are "
               "granted read/write/execute access; output the sorted granted user "
               "indices or 'none'.")
    config_cls = UnixModePermissionConfig
    task_version = 2

    def generate_entry(self):
        n = self.config.n_users
        access = random.choice(["read", "write", "execute"])

        owner_uid = random.randrange(2000, 4000)
        group_pool = random.sample(range(1000, 5000), self.config.n_groups)
        owner_gid = random.choice(group_pool)

        users = []
        for _ in range(n):
            uid = random.randrange(100, 4000)
            primary = random.choice(group_pool)
            supplement = random.sample([g for g in group_pool if g != primary],
                                       random.randrange(0, max(1, self.config.n_groups - 2)))
            users.append({"uid": uid, "groups": sorted(set([primary] + supplement))})

        mode = 0
        for shift in (0, 3, 6):
            mode |= random.randrange(8) << shift
        special = random.choice([0, 0o4000, 0o2000, 0o1000, 0o6000])
        mode |= special

        granted = []
        for i, u in enumerate(users, start=1):
            if user_granted(mode, owner_gid, u["uid"], u["groups"], access):
                granted.append(i)

        answer = " ".join(str(i) for i in granted) if granted else "none"
        assert parse_granted(answer) == tuple(granted)

        payload = edict({
            "File Mode": format_mode(mode),
            "File Owner": "uid=%d, gid=%d" % (owner_uid, owner_gid),
            "Requested Access": access,
            "Users": render_users(users),
        })
        metadata = edict({
            "payload": payload,
            "mode": mode,
            "owner_uid": owner_uid,
            "owner_gid": owner_gid,
            "access": access,
            "users": users,
        })
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        preamble = (
            "A Unix file has the following octal mode, owner, and set of user "
            "accounts. Access is decided by the standard permission rules: the file "
            "owner uses the owner ('owner=') bits; otherwise a user whose groups "
            "contain the file's gid uses the group ('group=') bits; and every other "
            "requester uses the 'other=' bits. A requested access (read/write/"
            "execute) maps to the r/w/x bit of the applicable class."
        )
        body = render_payload(metadata.payload)
        fmt = ("Which users (by 1-based index in the Users list) are granted the "
               "requested access? Give their indices as space-separated integers in "
               "ascending order, or the single word 'none' if not one is granted.")
        return "%s\n\n%s\n\n%s" % (preamble, body, fmt)

    def score_answer(self, answer, entry):
        gold = parse_granted(entry.answer)
        try:
            got = parse_granted(answer)
        except Exception:
            return 0.0
        if got == gold:
            return 1.0
        if got and gold and got != gold and all(g in gold for g in got):
            return 0.5
        return 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'unix_mode_permission (draw 1 of 2)',
 'hypothesis': 'W1-047',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/unix_mode_permission',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 812548150,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
