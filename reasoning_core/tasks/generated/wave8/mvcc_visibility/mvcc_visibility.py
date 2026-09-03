import random

from reasoning_core.template import Task, Entry, Config, edict


class MVCCConfig(Config):
    n_versions: int = 3
    n_rows: int = 1

    def apply_difficulty(self, level):
        self.n_versions = 2 + level * 2
        self.n_rows = 1 + (level // 3)


def _visible(versions, snap_ts):
    vis = [v for v in versions if v[0] <= snap_ts]
    if not vis:
        return None
    return max(vis, key=lambda v: v[0])


def _fmt(v):
    if v is None:
        return "none"
    return f"({v[0]},{v[1]})"


def _fmt_row(versions):
    return " ".join(f"({lo},{hi})" for lo, hi in versions)


def _parse_answer(s):
    s = s.strip().lower().replace(" ", "")
    if s == "none":
        return ("scalar", None)
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        if not inner:
            return ("list", [])
        pieces = inner.split("),(")
        out = []
        for piece in pieces:
            piece = piece.strip("()")
            if piece == "none":
                out.append(None)
            else:
                a, b = piece.split(",")
                out.append((int(a), int(b)))
        return ("list", out)
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].split(",")
        return ("tuple", (int(inner[0]), int(inner[1])))
    raise ValueError("bad answer: " + repr(s))


def score_answer(answer, entry):
    try:
        kind, parsed = _parse_answer(answer)
    except Exception:
        return 0.0
    try:
        tkind, tparsed = _parse_answer(entry.answer)
    except Exception:
        return 0.0
    if kind != tkind:
        return 0.0
    if kind == "scalar":
        return 1.0 if parsed == tparsed else 0.0
    if kind == "tuple":
        return 1.0 if parsed == tparsed else 0.0
    lp = [(None if x is None else (int(x[0]), int(x[1]))) for x in parsed]
    tp = [(None if x is None else (int(x[0]), int(x[1]))) for x in tparsed]
    return 1.0 if lp == tp else 0.0


class MVCCVisibility(Task):
    summary = ("Given transaction timestamps and row versions, output the version "
               "visible to a queried snapshot. For one queried snapshot return the "
               "visible (begin,end) version of each row as a list, 'none' when a row "
               "has no visible version; timestamps spread over a wide range so "
               "answers vary across every level.")

    config_cls = MVCCConfig

    def generate_entry(self):
        cfg = self.config
        n_versions = int(cfg.n_versions)
        n_rows = int(cfg.n_rows)
        while True:
            rows = []
            for _ in range(n_rows):
                versions = []
                ts = 0
                for _ in range(n_versions):
                    ts += random.randint(1, 5)
                    lo = ts
                    ts += random.randint(1, 6)
                    hi = ts
                    versions.append((lo, hi))
                rows.append(versions)

            snap_ts = random.randint(1, 8 + 6 * n_versions)

            visible = [_visible(row, snap_ts) for row in rows]
            if all(v is None for v in visible):
                continue
            if n_rows == 1 and visible[0] is None:
                if random.random() < 0.6:
                    continue

            if n_rows == 1:
                answer = _fmt(visible[0])
                payload = {
                    "row": _fmt_row(rows[0]),
                    "mode": "row",
                    "snapshot": snap_ts,
                }
            else:
                answer = "[" + ",".join(_fmt(v) for v in visible) + "]"
                payload = {
                    "rows": [_fmt_row(r) for r in rows],
                    "mode": "rows",
                    "snapshot": snap_ts,
                }
            metadata = edict({"payload": payload})
            entry = Entry(metadata=metadata, answer=answer)
            if score_answer(answer, entry) == 1.0:
                return entry

    def render_prompt(self, metadata):
        p = metadata.payload
        snap = p["snapshot"]
        if p["mode"] == "row":
            return (f"An MVCC store keeps row versions as (begin,end) timestamp pairs; "
                    f"a version is visible to snapshot T exactly when begin <= T < end. "
                    f"Only the newest such version is visible.\n"
                    f"Row versions: {p['row']}\n"
                    f"Snapshot time: {snap}\n"
                    f"Which version is visible to this snapshot? Give it as (begin,end), "
                    f"or the exact word 'none' if no version is visible.")
        rows = " ; ".join(f"row{i}: {r}" for i, r in enumerate(p["rows"]))
        return (f"An MVCC store keeps row versions as (begin,end) timestamp pairs; a "
                f"version is visible to snapshot T exactly when begin <= T < end. Only "
                f"the newest such version is visible.\n"
                f"Rows: {rows}\n"
                f"Snapshot time: {snap}\n"
                f"For every row, give the (begin,end) version visible to this snapshot, "
                f"using the exact word 'none' for any row with no visible version. "
                f"Answer as a list in row order, e.g. [(1,4),none,(2,6)].")

    def score_answer(self, answer, entry):
        return score_answer(answer, entry)


TASK_META = {'parent_source_id': None,
 'idea': 'mvcc_visibility (draw 1 of 2)',
 'hypothesis': 'W1-033',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/mvcc_visibility',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1305167045,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
