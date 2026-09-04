import random
from dataclasses import dataclass

import networkx as nx

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'maximum_flow (draw 1 of 1)',
 'hypothesis': 'HV-014',
 'changes': 'new task in reasoning_core/tasks/generated/wave9/maximum_flow',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3344734409,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class MaximumFlowConfig(Config):
    n_nodes: int = 5
    p: float = 0.5
    max_cap: int = 6

    def apply_difficulty(self, level):
        self.n_nodes = sround(self.n_nodes + 2 * level)
        self.p = min(0.85, 0.5 + 0.06 * level)
        self.max_cap = sround(self.max_cap + 5 * level)


class MaximumFlow(Task):
    config_cls = MaximumFlowConfig
    summary = ("Execute a canonical augmenting-path (Edmonds-Karp) maximum-flow procedure on "
               "randomly wired integer-capacity directed DAG flow networks, returning the final "
               "maximum flow value as a non-negative integer.")

    def generate_entry(self):
        cfg = self.config
        n = cfg.n_nodes
        source = 0
        sink = n - 1

        while True:
            edges = []
            for u in range(n - 1):
                for v in range(u + 1, n):
                    if random.random() < cfg.p:
                        cap = random.randint(1, cfg.max_cap)
                        edges.append((u, v, cap))
            if not edges:
                continue
            g = nx.DiGraph()
            g.add_nodes_from(range(n))
            for (u, v, c) in edges:
                g.add_edge(u, v, capacity=c)
            try:
                if not nx.has_path(g, source, sink):
                    continue
            except nx.NetworkXException:
                continue
            flow_value, flow_dict = nx.maximum_flow(g, source, sink)
            break

        if flow_value < 0:
            raise RuntimeError("negative max flow")

        payload_edges = sorted(edges)
        payload = {"source": source, "sink": sink, "edges": payload_edges}
        instruction = (
            "Find the maximum flow from the source vertex to the sink vertex in the "
            "integer-capacity directed network using the canonical augmenting-path (Edmonds-Karp) "
            "procedure. Each edge is written as (from, to, capacity). Report the value of a "
            "maximum flow as an integer."
        )
        metadata = edict({
            "flow_value": int(flow_value),
            "edges": payload_edges,
        })
        metadata.payload = {"instruction": instruction, "graph": payload}
        answer = str(int(flow_value))
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return f"{render_payload(metadata.payload)}\n\nThe answer is an integer."

    def score_answer(self, answer, entry):
        try:
            return 1.0 if int(str(answer).strip()) == int(entry.answer) else 0.0
        except Exception:
            return 0.0
