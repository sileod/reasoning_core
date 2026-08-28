import random
from dataclasses import dataclass

import networkx as nx

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'Add bipartite matching as a direct reasoning primitive.',
 'hypothesis': 'N8',
 'changes': 'Implement maximum-cardinality or maximum-weight matching with '
            'deterministic tie handling.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2431745573,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 20,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class BipartiteMatchingConfig(Config):
    left: int = 3
    right: int = 3
    p: float = 0.5
    use_weights: bool = False
    max_w: int = 9

    def apply_difficulty(self, level):
        self.left = sround(self.left + 2 * level)
        self.right = sround(self.right + 2 * level)
        self.p = max(0.3, min(0.8, 0.7 - 0.06 * level))
        self.max_w = sround(self.max_w + 6 * level)


class BipartiteMatching(Task):
    config_cls = BipartiteMatchingConfig

    def generate_entry(self):
        cfg = self.config
        edges = set()
        g = nx.Graph()
        g.add_nodes_from(range(cfg.left + cfg.right))
        for l in range(cfg.left):
            for r in range(cfg.right):
                if random.random() < cfg.p:
                    edges.add((l, r))
        if not edges:
            edges.add((0, 0))
            g.add_edge(0, 0)
        for (l, r) in edges:
            g.add_edge(l, cfg.left + r)

        if cfg.use_weights:
            weights = {e: random.randint(1, cfg.max_w) for e in edges}
            nxa = nx.Graph()
            for (l, r) in edges:
                w = weights[(l, r)]
                nxa.add_edge(l, cfg.left + r, weight=w)
            matching = nx.max_weight_matching(nxa, maxcardinality=True)
            pairs = sorted((min(u, v), max(u, v)) for u, v in matching)
            pairs = sorted(pairs)
            total = sum(weights[(u, v - cfg.left)] if u < cfg.left else weights[(v, u - cfg.left)] for u, v in pairs)
            total = 0
            for u, v in pairs:
                if u < cfg.left:
                    total += weights[(u, v - cfg.left)]
                else:
                    total += weights[(v, u - cfg.left)]
            answer = str(total)
            payload_edges = sorted((l, r, weights[(l, r)]) for (l, r) in edges)
            instruction = (
                "Find a matching in the bipartite graph. A matching is a set of edges, no two of "
                "which share an endpoint. Maximize the total weight of the matching (you may leave "
                "vertices unmatched if it helps). Every edge is written as (left, right, weight). "
                "Report the maximum possible total weight as an integer."
            )
            payload = {"left": list(range(cfg.left)), "right": list(range(cfg.left, cfg.left + cfg.right)),
                       "edges": payload_edges}
        else:
            g2 = nx.Graph()
            for (l, r) in edges:
                g2.add_edge(l, cfg.left + r)
            matching = nx.max_weight_matching(g2, maxcardinality=True)
            pairs = sorted(tuple(sorted(e)) for e in matching)
            pairs = sorted(pairs)
            answer = str(len(pairs))
            instruction = (
                "Find the largest matching in the bipartite graph. A matching is a set of edges, "
                "no two of which share an endpoint. Every edge connects a left vertex to a right "
                "vertex. Report the maximum number of edges in a matching as an integer."
            )
            payload = {"left": list(range(cfg.left)), "right": list(range(cfg.left, cfg.left + cfg.right)),
                       "edges": sorted((l, r) for (l, r) in edges)}

        metadata = edict({
            "answer": answer,
            "matching_pairs": pairs,
            "use_weights": cfg.use_weights,
        })
        metadata.payload = {"instruction": instruction, "graph": payload}
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return f"{render_payload(metadata.payload)}\n\nThe answer is an integer."

    def score_answer(self, answer, entry):
        try:
            return 1.0 if int(str(answer).strip()) == int(entry.answer) else 0.0
        except Exception:
            return 0.0
