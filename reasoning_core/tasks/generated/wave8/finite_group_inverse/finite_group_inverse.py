import random
from dataclasses import dataclass
from itertools import permutations

from reasoning_core.template import Config, Entry, Task, edict, render_payload, stochastic_rounding as sround


@dataclass
class FiniteGroupInverseConfig(Config):
    min_order: int = 4
    max_order: int = 7
    max_attempts: int = 80
    seed: int = 24601

    def apply_difficulty(self, level):
        self.min_order = sround(self.min_order + 0.4 * level)
        self.max_order = sround(self.max_order + 1.2 * level)
        self.max_attempts = sround(self.max_attempts + 15 * level)


def _labels(n):
    return [chr(65 + i) for i in range(n)]


def _compose(p, q):
    return tuple(p[x] for x in q)


def _cyclic_table(n):
    return [[(a + b) % n for b in range(n)] for a in range(n)]


def _dihedral_table(n):
    m = n // 2

    def idx(b, i):
        return b * m + i

    table = [[0] * n for _ in range(n)]
    for b1 in range(2):
        for i1 in range(m):
            for b2 in range(2):
                for i2 in range(m):
                    rot = (i1 + ((-1) ** b1) * i2) % m
                    fl = (b1 + b2) % 2
                    table[idx(b1, i1)][idx(b2, i2)] = idx(fl, rot)
    return table


_Q8_INDEX = {"1": 0, "-1": 1, "i": 2, "-i": 3, "j": 4, "-j": 5, "k": 6, "-k": 7}
_Q8_UNIT = {
    ("1", "1"): ("1", 1), ("1", "i"): ("i", 1), ("1", "j"): ("j", 1), ("1", "k"): ("k", 1),
    ("i", "1"): ("i", 1), ("i", "i"): ("1", -1), ("i", "j"): ("k", 1), ("i", "k"): ("j", -1),
    ("j", "1"): ("j", 1), ("j", "i"): ("k", -1), ("j", "j"): ("1", -1), ("j", "k"): ("i", 1),
    ("k", "1"): ("k", 1), ("k", "i"): ("j", 1), ("k", "j"): ("i", -1), ("k", "k"): ("1", -1),
}


def _q8_table(n):
    def qmul(a, b):
        sa = -1 if a.startswith("-") else 1
        sb = -1 if b.startswith("-") else 1
        ua = a.lstrip("-")
        ub = b.lstrip("-")
        unit, su = _Q8_UNIT[(ua, ub)]
        sign = sa * sb * su
        if sign < 0:
            return "-" + unit
        return unit

    elems = ["1", "-1", "i", "-i", "j", "-j", "k", "-k"]
    table = [[_Q8_INDEX[qmul(a, b)] for b in elems] for a in elems]
    return table


def _perm_group_table(elems):
    index = {p: i for i, p in enumerate(elems)}
    n = len(elems)
    table = [[index[_compose(a, b)] for b in elems] for a in elems]
    return table


def _symmetric_table(n):
    elems = list(permutations(range(4))) if n == 24 else list(permutations(range(3)))
    return _perm_group_table(elems)


def _alternating_table(n):
    elems = [p for p in permutations(range(4)) if _sign_parity(p) == 0]
    return _perm_group_table(elems)


def _sign_parity(p):
    inv = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                inv += 1
    return inv % 2


def _build_group(n, kind):
    if kind == "cyclic":
        return _cyclic_table(n)
    if kind == "dihedral" and n >= 6 and n % 2 == 0:
        return _dihedral_table(n)
    if kind == "q8" and n == 8:
        return _q8_table(n)
    if kind == "s3" and n == 6:
        return _symmetric_table(n)
    if kind == "a4" and n == 12:
        return _alternating_table(n)
    if kind == "s4" and n == 24:
        return _symmetric_table(n)
    return None


def _available_kinds(n):
    kinds = ["cyclic"]
    if 6 <= n and n % 2 == 0:
        kinds.append("dihedral")
    if n == 8:
        kinds.append("q8")
    if n == 6:
        kinds.append("s3")
    if n == 12:
        kinds.append("a4")
    if n == 24:
        kinds.append("s4")
    return kinds


def _render_table(table, labels):
    width = max(len(str(labels[0])), 2)
    head = "   " + " ".join(labels[i].rjust(width) for i in range(len(labels)))
    rows = [head]
    for a in range(len(labels)):
        cells = " ".join(labels[table[a][b]].rjust(width) for b in range(len(labels)))
        rows.append(labels[a].rjust(width) + "  " + cells)
    return "\n".join(rows)


class FiniteGroupInverse(Task):
    summary = ("Given a labeled Cayley table of a finite group and an element name, "
               "output the label of its inverse across cyclic, dihedral, quaternion, "
               "symmetric, and alternating group instances.")
    config_cls = FiniteGroupInverseConfig
    task_version = 2

    def generate_entry(self):
        cfg = self.config
        for _ in range(cfg.max_attempts):
            n = random.randint(cfg.min_order, cfg.max_order)
            kind = random.choice(_available_kinds(n))
            table = _build_group(n, kind)
            if table is None:
                continue
            ren = list(range(n))
            random.shuffle(ren)
            disp = [[0] * n for _ in range(n)]
            for a in range(n):
                for b in range(n):
                    disp[ren[a]][ren[b]] = ren[table[a][b]]
            identity = ren[0]
            x_old = random.randrange(n)
            y_old = next(c for c in range(n) if table[x_old][c] == 0)
            answer_label = _labels(n)[ren[y_old]]
            target_label = _labels(n)[ren[x_old]]
            labels = _labels(n)
            for c in range(n):
                if disp[identity][c] != c or disp[c][identity] != c:
                    break
            else:
                metadata = edict({
                    "group": kind,
                    "order": int(n),
                    "identity": labels[identity],
                    "element": target_label,
                    "inverse": answer_label,
                })
                metadata.payload = {
                    "Cayley table": _render_table(disp, labels),
                    "Element": target_label,
                }
                return Entry(metadata=metadata, answer=answer_label)
        raise RuntimeError(f"could not build a group of order in [{cfg.min_order},{cfg.max_order}]")

    def render_prompt(self, metadata):
        return (
            f"{render_payload(metadata.payload)}\n\n"
            "The table above is the multiplication (Cayley) table of a finite group: "
            "the cell at row r and column c gives the product r * c of the two row/column "
            "elements. Determine the inverse of the element labeled "
            f"{metadata.element} -- the unique element g for which {metadata.element} * g equals the "
            "identity element. Give only the label of that inverse element, for example 'C'."
        )

    def score_answer(self, answer, entry):
        return 1.0 if str(answer).strip() == str(entry.answer).strip() else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'finite_group_inverse (draw 1 of 2)',
 'hypothesis': 'W1-019',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/finite_group_inverse',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 4262150195,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
