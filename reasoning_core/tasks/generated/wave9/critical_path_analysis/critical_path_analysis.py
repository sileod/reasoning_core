import random
from dataclasses import dataclass

from reasoning_core.template import Entry, Config, Task, edict, stochastic_rounding as sround


@dataclass
class CriticalPathAnalysisConfig(Config):
    n_tasks: int = 6
    max_out: int = 2
    dur_hi: int = 9

    def apply_difficulty(self, level):
        self.n_tasks = sround(4 + 3 * level)
        self.max_out = sround(1 + level // 2 + 1)
        self.dur_hi = sround(5 + 3 * level)


def _topo(g, n):
    indeg = {i: 0 for i in range(1, n + 1)}
    for u in range(1, n + 1):
        for v in g[u]:
            indeg[v] += 1
    stack = [i for i in range(1, n + 1) if indeg[i] == 0]
    topo = []
    while stack:
        u = stack.pop()
        topo.append(u)
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    return topo


def _analyze(g, n, durs):
    topo = _topo(g, n)
    earliest = {}
    for u in topo:
        best = 0
        for p in range(1, n + 1):
            if u in g[p]:
                best = max(best, earliest[p] + durs[p])
        earliest[u] = best
    total = max(earliest[i] + durs[i] for i in range(1, n + 1))
    latest = {}
    for u in reversed(topo):
        if not g[u]:
            latest[u] = total - durs[u]
        else:
            m = min(latest[v] for v in g[u])
            latest[u] = m - durs[u]
    return total, earliest, latest


def _critical(g, n, durs, total, earliest, latest):
    crit = []
    for i in range(1, n + 1):
        if (latest[i] - earliest[i]) == 0:
            crit.append(i)
    return sorted(crit)


class CriticalPathAnalysis(Task):
    summary = ("Compute earliest and latest start times, slack, and critical activities in precedence networks "
               "with durations, returning the set of critical activities as sorted labels.")
    config_cls = CriticalPathAnalysisConfig

    def generate_entry(self):
        cfg = self.config
        n = cfg.n_tasks
        while True:
            g = {i: [] for i in range(1, n + 1)}
            for i in range(1, n + 1):
                n_out = random.randint(0, cfg.max_out)
                candidates = [j for j in range(i + 1, n + 1)]
                random.shuffle(candidates)
                for j in candidates[:n_out]:
                    if j not in g[i]:
                        g[i].append(j)
            if len(_topo(g, n)) != n:
                continue
            durs = {i: random.randint(1, cfg.dur_hi) for i in range(1, n + 1)}
            total, earliest, latest = _analyze(g, n, durs)
            if _topo(g, n) == [i for i in range(1, n + 1)] and _critical(g, n, durs, total, earliest, latest) == list(range(1, n + 1)):
                continue
            break
        total, earliest, latest = _analyze(g, n, durs)
        crit = _critical(g, n, durs, total, earliest, latest)
        answer = " ".join(chr(64 + i) for i in crit)
        labels = [chr(64 + i) for i in range(1, n + 1)]
        payload = {
            "tasks": " ".join(labels),
            "durations": {l: int(durs[i]) for i, l in enumerate(labels, start=1)},
            "precedence": {labels[i - 1]: [labels[j - 1] for j in g[i]] for i in range(1, n + 1)},
        }
        metadata = edict(
            n=n,
            g={str(k): v for k, v in g.items()},
            durs={str(k): int(v) for k, v in durs.items()},
            payload=payload,
            makespan=int(total),
            earliest={str(k): int(v) for k, v in earliest.items()},
            latest={str(k): int(v) for k, v in latest.items()},
        )
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, m):
        prec = "\n".join(f"  {a} -> {(' '.join(b) if b else '(none)')}"
                         for a, b in m.payload["precedence"].items())
        durs = "  ".join(f"{a}: {d}" for a, d in m.payload["durations"].items())
        return (
            "Consider a project made of precedence-constrained tasks, each with a duration. "
            "A task can start only once all its predecessors have finished. Compute the earliest and latest "
            "start times for every task, then the slack of each task (latest minus earliest start). A task is "
            "critical when its slack is exactly zero. Using the critical-path method (CPM), find every critical "
            "activity.\n\n"
            f"Tasks: {m.payload['tasks']}\n"
            f"Durations:\n  {durs}\n"
            f"Precedence (a -> b means a must finish before b starts):\n{prec}\n\n"
            "The answer is the space-separated, alphabetically sorted list of critical task labels (for example "
            "\"A C E\")."
        )

    def score_answer(self, answer, entry):
        norm = lambda x: " ".join(str(x).upper().replace(",", " ").split())
        return float(norm(answer) == norm(entry.answer))


TASK_META = {'parent_source_id': None,
 'idea': 'critical_path_analysis (draw 1 of 1)',
 'hypothesis': 'HV-068',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/critical_path_analysis',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1776440515,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
