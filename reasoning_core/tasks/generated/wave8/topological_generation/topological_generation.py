from dataclasses import dataclass
import random

from reasoning_core.template import Task, Entry, Config, edict

TASK_META = {'parent_source_id': None,
 'idea': 'topological_generation (draw 1 of 2)',
 'hypothesis': 'W1-006',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/topological_generation',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3867398156,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _rounds_removal(n_nodes, edges):
    removed = [False] * n_nodes
    indeg = [0] * n_nodes
    adj = [[] for _ in range(n_nodes)]
    rounds_of = [0] * n_nodes
    for a, b in edges:
        adj[a].append(b)
        indeg[b] += 1
    round_num = 0
    total = 0
    while total < n_nodes:
        current = [v for v in range(n_nodes) if not removed[v] and indeg[v] == 0]
        if not current:
            break
        round_num += 1
        for v in current:
            removed[v] = True
            total += 1
            rounds_of[v] = round_num
            for w in adj[v]:
                indeg[w] -= 1
    return round_num, total, rounds_of


@dataclass
class TopologicalGenerationConfig(Config):
    n_nodes: int = 8
    depth: int = 5
    p: float = 0.35

    def apply_difficulty(self, level):
        self.n_nodes = self.n_nodes + 2 * level
        self.depth = self.depth + 1 * level
        self.p = min(0.6, 0.35 + 0.05 * level)


class TopologicalGeneration(Task):
    summary = "Given a DAG and node, output its round under repeated zero-indegree removal; vary node count, edge density, and target node rank, spreading targets uniformly across early, middle, and late removal rounds."
    config_cls = TopologicalGenerationConfig

    def generate_entry(self):
        cfg = self.config
        rng = random
        while True:
            n = cfg.n_nodes
            D = cfg.depth
            if D > n:
                D = n
            target = rng.randrange(n)
            pool = list(range(n))
            rng.shuffle(pool)
            counts = [n // D + (1 if i < n % D else 0) for i in range(D)]
            nodes_by_round = []
            idx = 0
            for c in counts:
                nodes_by_round.append(sorted(pool[idx:idx + c]))
                idx += c
            target_round = None
            for r, grp in enumerate(nodes_by_round, start=1):
                if target in grp:
                    target_round = r
                    break
            edges = set()
            for r in range(2, D + 1):
                prev = nodes_by_round[r - 2]
                for v in nodes_by_round[r - 1]:
                    edges.add((rng.choice(prev), v))
            for i in range(D):
                left = nodes_by_round[i]
                for j in range(i + 1, D):
                    right = nodes_by_round[j]
                    k = 0
                    while rng.random() < cfg.p and k < 3:
                        edges.add((rng.choice(left), rng.choice(right)))
                        k += 1
            edges = sorted(edges)
            rounds, total, rounds_of = _rounds_removal(n, edges)
            if total < n or target_round is None:
                continue
            if rounds_of[target] != target_round:
                continue
            metadata = edict({
                "n_nodes": int(n),
                "depth": int(D),
                "p": float(cfg.p),
                "target": int(target),
                "edges": [[int(a), int(b)] for a, b in edges],
                "round": int(target_round),
            })
            metadata.payload = {
                "n_nodes": metadata.n_nodes,
                "target": metadata.target,
                "edges": metadata.edges,
            }
            return Entry(metadata=metadata, answer=str(target_round))

    def render_prompt(self, metadata):
        edges = ", ".join(f"({a}, {b})" for a, b in metadata.edges)
        return (
            f"Consider a directed acyclic graph on nodes 0..{metadata.n_nodes - 1} with edges "
            f"{{{edges}}}. Repeatedly remove, in each round, all nodes that currently have "
            f"zero in-degree, all at once; removing them may unlock others which are removed in "
            f"the next round. Node {metadata.target} is removed in round ___ (rounds are counted "
            f"starting from 1 for the first batch of removed nodes).\n\n"
            f"Your answer is that round number, an integer."
        )

    def score_answer(self, answer, entry):
        try:
            val = int(str(answer).strip())
        except (ValueError, TypeError):
            return 0.0
        return 1.0 if val == int(entry.answer) else 0.0
