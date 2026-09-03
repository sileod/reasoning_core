import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'Add dependency resolution under version constraints.',
 'hypothesis': 'S24',
 'changes': 'Ask which version a resolver picks given stated ranges and an '
            'available version list.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1549347574,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


_PKG_NAMES = ["http", "json", "xml", "sql", "core", "util", "io",
              "log", "db", "auth", "cache", "cli"]


@dataclass
class VersionResolutionConfig(Config):
    n_packages: int = 3
    max_versions: int = 4
    max_deps: int = 1

    def apply_difficulty(self, level):
        self.n_packages = sround(self.n_packages + level)
        self.max_versions = sround(self.max_versions + (level > 1) + (level > 3))
        self.max_deps = sround(self.max_deps + (level > 2) + (level > 4))


def _vtok(s):
    return tuple(int(x) for x in s.split("."))


def _vstr(t):
    return ".".join(str(x) for x in t)


def _range_minmax(rs):
    lo, hi = (0, 0, 0), None
    lo_excl = hi_excl = False
    for clause in rs.split(","):
        clause = clause.strip()
        if clause.startswith(">="):
            v = _vtok(clause[2:].strip())
            if v > lo or (v == lo and not lo_excl):
                lo, lo_excl = v, False
        elif clause.startswith(">"):
            v = _vtok(clause[1:].strip())
            if v > lo or (v == lo and not lo_excl):
                lo, lo_excl = v, True
        elif clause.startswith("<="):
            v = _vtok(clause[2:].strip())
            if hi is None or v < hi or (v == hi and not hi_excl):
                hi, hi_excl = v, False
        elif clause.startswith("<"):
            v = _vtok(clause[1:].strip())
            if hi is None or v < hi or (v == hi and not hi_excl):
                hi, hi_excl = v, True
        elif clause.startswith("=="):
            v = _vtok(clause[2:].strip())
            lo, lo_excl = max(lo, v), False
            hi, hi_excl = v, False
        elif clause.startswith("^"):
            v = _vtok(clause[1:].strip())
            if v > lo or (v == lo and not lo_excl):
                lo, lo_excl = v, False
            cap = (v[0] + 1, 0, 0)
            if hi is None or cap < hi or (cap == hi and not hi_excl):
                hi, hi_excl = cap, True
        elif clause.startswith("~"):
            v = _vtok(clause[1:].strip())
            if v > lo or (v == lo and not lo_excl):
                lo, lo_excl = v, False
            cap = (v[0], v[1] + 1, 0)
            if hi is None or cap < hi or (cap == hi and not hi_excl):
                hi, hi_excl = cap, True
        else:
            raise ValueError(clause)
    return lo, lo_excl, hi, hi_excl


def _in_range(t, rs):
    lo, lo_excl, hi, hi_excl = _range_minmax(rs)
    if t < lo:
        return False
    if lo_excl and t == lo:
        return False
    if hi is None:
        return True
    return t < hi if hi_excl else t <= hi


def _resolve(m, versions, reqs):
    """Greedy highest-satisfying resolution in stated order.

    reqs[i] is a list of (version_index, [(dep_index, range_str), ...]).
    Returns chosen version index per package.
    """
    chosen = [None] * m
    for i in range(m):
        cand = None
        for vi in range(len(versions[i]) - 1, -1, -1):
            ok = True
            for ver, req_list in reqs[i]:
                if ver == vi:
                    for dep, rs in req_list:
                        if not _in_range(versions[dep][chosen[dep]], rs):
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                cand = vi
                break
        chosen[i] = cand
    return chosen


def _gen_versions(n_versions):
    floor = (random.randint(1, 5), 0, 0)
    versions = [floor]
    while len(versions) < n_versions:
        mv, mn, mp = versions[-1]
        step = random.choice(["major", "minor", "patch"])
        if step == "major":
            nv = (mv + random.randint(1, 2), random.randint(0, 2), random.randint(0, 4))
        elif step == "minor":
            nv = (mv, mn + random.randint(1, 3), random.randint(0, 4))
        else:
            nv = (mv, mn, mp + random.randint(1, 6))
        if nv > versions[-1]:
            versions.append(nv)
    return versions


def _gen_range_including(v):
    style = random.choice(["ge", "eq", "wide", "caret", "tilde"])
    if style == "ge":
        return f">= {_vstr(v)}"
    if style == "eq":
        return f"== {_vstr(v)}"
    if style == "wide":
        return f">= {_vstr(v)}, <= {_vstr((v[0], v[1] + 5, 9))}"
    if style == "caret":
        return f"^{_vstr(v)}"
    return f"~{_vstr(v)}"


def _gen_range_excluding(v):
    if random.random() < 0.5 and v[1] > 0:
        return f"< {_vstr(v)}"
    return f"> {_vstr(v)}"


def _gen_range_any(vs):
    lo_idx = random.randrange(len(vs))
    hi_idx = random.randrange(lo_idx, len(vs))
    lo = vs[lo_idx]
    hi = vs[hi_idx]
    if random.random() < 0.5:
        return f">= {_vstr(lo)}"
    if hi > lo:
        return f">= {_vstr(lo)}, < {_vstr(hi)}"
    return f"== {_vstr(lo)}"


class VersionResolution(Task):
    config_cls = VersionResolutionConfig

    def generate_entry(self):
        cfg = self.config
        m = max(2, min(cfg.n_packages, len(_PKG_NAMES)))
        names = _PKG_NAMES[:m]
        versions = [_gen_versions(random.randint(3, max(3, cfg.max_versions))) for _ in range(m)]

        deps = [[] for _ in range(m)]
        for i in range(1, m):
            n_deps = random.randint(1, min(max(1, cfg.max_deps), i))
            deps[i] = sorted(random.sample(range(i), n_deps))

        k = [len(versions[0]) - 1] + [0] * (m - 1)
        for i in range(1, m - 1):
            k[i] = random.randrange(len(versions[i]))
        k[m - 1] = random.randrange(1, len(versions[m - 1]) - 1)

        chosen = [None] * m
        chosen[0] = k[0]
        reqs = [[] for _ in range(m)]
        for i in range(1, m):
            dep_chosen = {d: chosen[d] for d in deps[i]}
            excl_dep = deps[i][0]
            per_ver = []
            for vi in range(len(versions[i])):
                out = []
                for d in deps[i]:
                    chosen_tuple = versions[d][dep_chosen[d]]
                    if vi > k[i] and d == excl_dep:
                        out.append((d, _gen_range_excluding(chosen_tuple)))
                    elif vi == k[i]:
                        out.append((d, _gen_range_including(chosen_tuple)))
                    else:
                        out.append((d, _gen_range_any(versions[d])))
                per_ver.append((vi, out))
            reqs[i] = per_ver
            chosen[i] = k[i]

        resolved = _resolve(m, versions, reqs)
        if resolved != chosen or any(c is None for c in resolved):
            raise RuntimeError("resolution mismatch")

        target_idx = m - 1
        answer = versions[target_idx][k[target_idx]]

        pkg_lines = [f"{names[i]}: " + ", ".join(_vstr(v) for v in versions[i]) for i in range(m)]
        dep_lines = []
        for i in range(1, m):
            for vi, out in reqs[i]:
                req_txt = "; ".join(f"requires {names[d]} {rs}" for d, rs in out)
                dep_lines.append(f"{names[i]} {_vstr(versions[i][vi])}: {req_txt}")

        metadata = edict({
            "payload": {
                "packages": "Available versions:\n" + "\n".join(pkg_lines),
                "requires": "Each package version declares ranges on packages listed "
                            "earlier (resolution order is the package order):\n"
                            + "\n".join(dep_lines),
                "notation": "Semantic ranges: >=, >, <=, <, ==; ^a.b.c means at least "
                            "a.b.c and below the next major (e.g. ^1.2.0 -> >=1.2.0 and "
                            "<2.0.0); ~a.b.c means at least a.b.c and below the next "
                            "minor. A comma lists both bounds.",
            },
            "target": names[target_idx],
            "resolution_order": ", ".join(names),
        })
        return Entry(metadata=metadata, answer=_vstr(answer))

    def render_prompt(self, metadata):
        payload = render_payload(metadata.payload)
        return (f"{payload}\n\n"
                f"A resolver picks versions by processing packages one at a time in "
                f"resolution order ({metadata.resolution_order}); for each package it "
                f"chooses the highest available version whose declared requirement "
                f"ranges are all satisfied by the versions already picked for the "
                f"packages resolved before it. Package {metadata.target} is resolved "
                f"last.\n\n"
                f"Which version of {metadata.target} is picked?\n\n"
                f"The answer is a version string.")

    def score_answer(self, answer, entry):
        a = str(answer).strip().strip('"').strip("'").strip()
        if a.lower().startswith("version "):
            a = a[len("version "):].strip()
        if a == entry.answer:
            return 1.0
        try:
            return 1.0 if _vtok(a) == _vtok(entry.answer) else 0.0
        except (ValueError, IndexError):
            return 0.0
