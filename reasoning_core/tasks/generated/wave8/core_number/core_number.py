import random
from dataclasses import dataclass

import networkx as nx

from reasoning_core.template import Task, Entry, Config, edict, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'core_number (draw 1 of 2)',
 'hypothesis': 'W1-004',
 'changes': 'new task in reasoning_core/tasks/generated/wave8/core_number',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2202313084,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class CoreNumberConfig(Config):
    n_nodes: int = 6
    p: float = 0.5

    def apply_difficulty(self, level):
        self.n_nodes = sround(self.n_nodes + 2 * level)
        self.p = 0.4 + 0.06 * level


class CoreNumber(Task):
    summary = "Given an undirected graph and a query node, output that node's k-core number (0 for isolated/1 for those outside any 2-core); connected graphs with balanced answer spread."
    config_cls = CoreNumberConfig

    def generate_entry(self):
        cfg = self.config
        while True:
            n = cfg.n_nodes
            G = nx.gnp_random_graph(n, cfg.p, seed=random.randrange(2 ** 32))
            if n == 1 or nx.number_connected_components(G) == 1:
                break
        cores = nx.core_number(G)
        query = random.randrange(n)
        knum = int(cores[query])
        metadata = edict({
            "nodes": int(n),
            "edges": sorted((int(u), int(v)) for u, v in G.edges()),
            "query": int(query),
            "degree": int(G.degree(query)),
        })
        metadata.payload = {
            "nodes": metadata.nodes,
            "edges": metadata.edges,
            "query": metadata.query,
        }
        assert 0 <= knum <= n - 1
        return Entry(metadata=metadata, answer=str(knum))

    def render_prompt(self, metadata):
        payload = f"The graph has {metadata.nodes} nodes numbered 0 to {metadata.nodes-1} with edges {metadata.edges}."
        payload += f"\nWhat is the k-core number of node {metadata.query}?"
        return payload + "\n\nThe answer is an integer."

    def score_answer(self, answer, entry):
        try:
            return 1.0 if int(entry.answer) == int(answer) else 0.0
        except (ValueError, TypeError):
            return 0.0
