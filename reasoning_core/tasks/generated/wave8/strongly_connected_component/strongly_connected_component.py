import random
from dataclasses import dataclass

import networkx as nx

from reasoning_core.template import Task, Entry, Config, edict, render_payload

TASK_META = {'parent_source_id': None,
 'idea': 'strongly_connected_component (draw 1 of 2)',
 'hypothesis': 'W1-001',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/strongly_connected_component',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3057161243,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class SCCConfig(Config):
    n_min: int = 5
    n_max: int = 9
    p: float = 0.35

    def apply_difficulty(self, level):
        self.n_min = int(self.n_min + level)
        self.n_max = int(self.n_max + level)
        self.p = min(0.65, self.p + 0.05 * level)


def _parse_answer(answer):
    try:
        return tuple(sorted(int(x) for x in answer.strip("[](){} ,").split(",") if x.strip() != ""))
    except (ValueError, TypeError):
        return None


class StronglyConnectedComponent(Task):
    summary = "Given a directed random graph and a designated node, output the sorted integer member list of the node's strongly connected component across varied sizes and densities."
    config_cls = SCCConfig

    def generate_entry(self):
        c = self.config
        while True:
            n = random.randint(c.n_min, c.n_max)
            g = nx.DiGraph()
            g.add_nodes_from(range(n))
            for u in range(n):
                for v in range(n):
                    if u != v and random.random() < c.p:
                        g.add_edge(u, v)
            target = random.randrange(n)
            scc = nx.strongly_connected_components(g)
            comp = next(s for s in scc if target in s)
            members = sorted(int(x) for x in comp)
            if len(members) < 1:
                continue
            metadata = edict({
                "n": n,
                "target": target,
                "edges": sorted((int(u), int(v)) for u, v in g.edges),
                "members": members,
            })
            metadata.payload = {"n": n, "target": target, "edges": metadata.edges}
            answer = "[" + ", ".join(str(m) for m in members) + "]"
            return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        edge_lines = "\n".join(f"{u} -> {v}" for u, v in metadata.edges)
        return (f"A directed graph has nodes numbered 0 to {metadata.n - 1}. "
                f"Its edges are:\n{edge_lines}\n"
                f"List the members of the strongly connected component containing node {metadata.target}, "
                f"as a single sorted list of node numbers. The answer is a list.")

    def score_answer(self, answer, entry):
        try:
            got = set(int(x) for x in answer.strip("[](){} ,").split(",") if x.strip() != "")
        except (ValueError, TypeError):
            return 0.0
        gold = set(entry.metadata["members"])
        if not got or got != gold:
            return 0.0
        return 1.0
