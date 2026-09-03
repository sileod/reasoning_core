from dataclasses import dataclass
import random

from reasoning_core.template import Task, Entry, Config, edict

TASK_META = {'parent_source_id': None,
 'idea': 'Add three-way merge reasoning over short line-based files.',
 'hypothesis': 'S43',
 'changes': 'Ask for the merged file, or for the line at which two edits '
            'conflict.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3493938574,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def apply_ops(base, ops):
    opmap = {}
    for (i, kind, val) in ops:
        opmap.setdefault(i, []).append((kind, val))
    out = []
    for i in range(len(base)):
        adds_before = [v for (k, v) in opmap.get(i, []) if k == "ins"]
        out.extend(adds_before)
        present = True
        replacement = None
        for (k, v) in opmap.get(i, []):
            if k == "del":
                present = False
            elif k == "chg":
                replacement = v
        if not present:
            continue
        out.append(base[i] if replacement is None else replacement)
    return out


def build_merged(base, a_ops, b_ops):
    a_idx = {o[0] for o in a_ops}
    b_idx = {o[0] for o in b_ops}
    inter = a_idx & b_idx
    if inter:
        return None, sorted(inter)[0]
    merged = apply_ops(base, sorted(a_ops + b_ops, key=lambda o: o[0]))
    return merged, None


@dataclass
class ThreeWayMergeConfig(Config):
    n_lines: int = 8
    n_edits: int = 2

    def apply_difficulty(self, level):
        self.n_lines = 8 + level * 2
        self.n_edits = 2 + (level // 2)


class ThreeWayMerge(Task):
    config_cls = ThreeWayMergeConfig

    def generate_entry(self):
        n_lines = int(self.config.n_lines)
        n_edits = int(self.config.n_edits)

        base = ["line %d" % i for i in range(1, n_lines + 1)]

        def diff_one():
            idx_set = tuple(sorted(random.sample(range(n_lines), n_edits)))
            ops = []
            for i in idx_set:
                choice = random.random()
                if choice < 0.4:
                    kind, val = "del", None
                elif choice < 0.7:
                    kind, val = "chg", "A%d" % i
                else:
                    kind, val = "ins", "AB%d" % i
                ops.append((i, kind, val))
            return ops

        a_ops = diff_one()
        b_ops = diff_one()
        conflicting = random.random() < 1.0 / 3.0
        if conflicting:
            common = tuple(sorted(random.sample(range(n_lines),
                                                random.randrange(1, n_edits + 1))))
            a_idx = {o[0] for o in a_ops}
            b_idx = {o[0] for o in b_ops}
            for ci in common:
                if ci not in a_idx:
                    a_ops = list(a_ops) + [(ci, random.choice(["del", "chg", "ins"]), "CA%d" % ci)]
                if ci not in b_idx:
                    b_ops = list(b_ops) + [(ci, random.choice(["del", "chg", "ins"]), "CB%d" % ci)]
        a_ops = sorted(a_ops, key=lambda o: o[0])
        b_ops = sorted(b_ops, key=lambda o: o[0])

        merged, conflict_line = build_merged(base, a_ops, b_ops)

        if conflict_line is not None:
            answer = "conflict %d" % (conflict_line + 1)
        else:
            answer = "\n".join("%d: %s" % (i + 1, ln) for i, ln in enumerate(merged))

        metadata = edict({
            "base": base,
            "version_a": apply_ops(base, a_ops),
            "version_b": apply_ops(base, b_ops),
            "merged": merged,
            "conflict_line": conflict_line,
        })
        metadata.payload = {
            "base": base,
            "version_a": apply_ops(base, a_ops),
            "version_b": apply_ops(base, b_ops),
        }
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        fmts = []
        for key, label in (("base", "Base"), ("version_a", "Version A"), ("version_b", "Version B")):
            fmts.append("%s:\n%s" % (label, numbered(metadata.payload[key])))
        rule = ("An edit is a changed, inserted or deleted line. Two edits "
                "conflict when they touch the same base line. A clean merge "
                "applies both edits to the base. If the two versions conflict, "
                "answer with 'conflict <N>', where N is the number of the first "
                "base line at which they conflict. If they merge cleanly, answer "
                "with the full merged file as numbered lines, one per line, in "
                "order of line number.")
        return "\n\n".join(fmts) + "\n\n" + rule


def numbered(lines):
    return "\n".join("%d: %s" % (i + 1, ln) for i, ln in enumerate(lines))


def _parse_entry(ans):
    s = ans.strip()
    if ":" not in s:
        return None
    num, _, rest = s.partition(":")
    num = num.strip()
    if not num.isdigit():
        return None
    return int(num), rest.strip()


def score_answer(answer, entry):
    gold = entry.answer.strip()
    ans = answer.strip()
    if entry.metadata.get("conflict_line") is not None:
        if ans == gold:
            return 1.0
        try:
            parts = ans.split()
            if len(parts) == 2 and parts[0] == "conflict":
                return 1.0 if int(parts[1]) == int(gold.split()[1]) else 0.0
        except Exception:
            return 0.0
        return 0.0
    exp = [l for l in gold.split("\n") if l.strip()]
    act = [l for l in ans.split("\n") if l.strip()]
    if len(exp) != len(act):
        return 0.0
    for e, a in zip(exp, act):
        eo = _parse_entry(e)
        ao = _parse_entry(a)
        if eo is None or ao is None:
            return 0.0
        if eo[0] != ao[0] or eo[1] != ao[1]:
            return 0.0
    return 1.0
