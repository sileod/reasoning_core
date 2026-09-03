import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


@dataclass
class DependencyBatchesConfig(Config):
    n_jobs: int = 5
    depth: int = 3

    def apply_difficulty(self, level):
        self.n_jobs = sround(self.n_jobs + 2 * level)
        self.depth = sround(self.depth + level)


def _parse_answer(answer):
    rounds = []
    for chunk in answer.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        names = tuple(sorted(name.strip() for name in chunk.split(",") if name.strip()))
        rounds.append(names)
    return tuple(rounds)


def _score(answer, entry):
    try:
        got = _parse_answer(answer)
        gold = _parse_answer(entry.answer)
    except Exception:
        return 0.0
    if not got:
        return 0.0
    if got == gold:
        return 1.0
    return 0.0


class DependencyBatches(Task):
    config_cls = DependencyBatchesConfig

    def generate_entry(self):
        cfg = self.config
        n = int(cfg.n_jobs)
        target_depth = max(2, int(cfg.depth))
        target_depth = min(target_depth, n)

        while True:
            names = [chr(ord("a") + i) for i in range(n)]
            topo = list(names)
            random.shuffle(topo)
            rank = {node: i for i, node in enumerate(topo)}

            pre = {node: set() for node in names}

            chain = topo[:target_depth]
            for i in range(1, len(chain)):
                pre[chain[i]].add(chain[i - 1])

            for b in names:
                earlier = [a for a in names if rank[a] < rank[b] and a != b]
                random.shuffle(earlier)
                for a in earlier:
                    if random.random() < 0.35:
                        pre[b].add(a)

            rounds = []
            remaining = set(names)
            while remaining:
                frontier = sorted(nn for nn in remaining if not (pre[nn] & remaining))
                rounds.append(tuple(frontier))
                remaining -= set(frontier)

            dist = {node: 1 for node in names}
            for node in topo:
                dist[node] = max(dist[node], 1 + max((dist[a] for a in pre[node]), default=0))
            depth = max(dist.values())
            if depth < 2:
                continue
            rounds = tuple(rounds)
            flat = {nn for r in rounds for nn in r}
            if flat != set(names):
                continue

            jobs = list(names)
            random.shuffle(jobs)
            sentences = []
            for b in jobs:
                deps = sorted(pre[b])
                if deps:
                    sentences.append(f"Job {b} requires " + ", ".join(deps) + " before it can start.")
                else:
                    sentences.append(f"Job {b} has no prerequisites and can start at any time.")

            metadata = edict(
                {
                    "n_jobs": int(n),
                    "depth": int(depth),
                    "prereqs": {b: sorted(pre[b]) for b in names},
                    "rounds": [list(r) for r in rounds],
                }
            )
            metadata.payload = {"jobs": ", ".join(jobs), "statements": sentences}
            answer = "; ".join(", ".join(r) for r in rounds)
            return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        body = "\n".join(metadata.payload["statements"])
        return (
            f"Consider these jobs: {metadata.payload['jobs']}.\n{body}\n"
            f"Each job runs as early as its prerequisites allow, and any number of "
            f"jobs run in parallel each round. Layer the jobs into rounds: in a single "
            f"round put all jobs whose prerequisites are all finished by the start of "
            f"that round, listing names in alphabetical order. Join the rounds with "
            f"semicolons, joining the names in a round with commas, e.g. "
            f"\"a, c; b; d, e\". The answer is this round listing."
        )

    def score_answer(self, answer, entry):
        return _score(answer, entry)


TASK_META = {'parent_source_id': None,
 'idea': 'Layer a dependency graph into the batches a scheduler would run.',
 'hypothesis': 'S60',
 'changes': 'New task; the answer is a list of groups, not a single order.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1613754928,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
