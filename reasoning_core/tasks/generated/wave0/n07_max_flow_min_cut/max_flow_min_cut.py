import ast
import random

import networkx as nx

from reasoning_core.template import Config, Entry, Task, edict, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'Add global capacitated-flow reasoning.',
 'hypothesis': 'N7',
 'changes': 'Implement maximum-flow values and canonical minimum-cut-side '
            'queries.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3487141474,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 20,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


class MaxFlowMinCutConfig(Config):
    n_nodes: int = 5
    p_edge: float = 0.5
    max_cap: int = 6

    def apply_difficulty(self, level):
        self.n_nodes = sround(self.n_nodes + 2 * level)
        self.p_edge = min(0.9, self.p_edge + 0.05 * level)
        self.max_cap = sround(self.max_cap + level)


class MaxFlowMinCut(Task):
    summary = "Compute maximum flow values and identify valid source-side minimum cuts in generated capacitated directed graphs."

    config_cls = MaxFlowMinCutConfig

    def _build(self):
        n = self.config.n_nodes
        max_cap = self.config.max_cap
        while True:
            G = nx.DiGraph()
            G.add_nodes_from(range(n))
            source = 0
            sink = n - 1
            for u in range(n):
                for v in range(n):
                    if u == v:
                        continue
                    if u == 0 and v == n - 1:
                        continue
                    if random.random() < self.config.p_edge:
                        cap = random.randint(1, max_cap)
                        if G.has_edge(u, v):
                            G[u][v]['capacity'] += cap
                        else:
                            G.add_edge(u, v, capacity=cap)
            if source == sink:
                continue
            if not nx.has_path(G, source, sink):
                continue
            try:
                flow_value, flow_dict = nx.maximum_flow(G, source, sink)
            except nx.NetworkXUnbounded:
                continue
            if flow_value <= 0:
                continue
            return G, source, sink, flow_dict

    def generate_entry(self):
        G, source, sink, flow_dict = self._build()
        edges = []
        for u, v, d in G.edges(data=True):
            edges.append((int(u), int(v), int(d['capacity'])))

        value, cut = nx.minimum_cut(G, source, sink)
        reachable, _ = cut
        reachable = sorted(int(x) for x in reachable)
        canonical = reachable[0] == 0

        flow_edges = []
        for u in flow_dict:
            for v, f in flow_dict[u].items():
                if f > 0:
                    flow_edges.append((int(u), int(v), int(f)))

        metadata = edict({
            'n_nodes': int(G.number_of_nodes()),
            'source': source,
            'sink': sink,
            'max_capacity': int(G.number_of_edges()),
            'edges': edges,
            'flow_value': int(value),
            'mincut_side': reachable,
            'canonical': canonical,
        })
        metadata.payload = {
            'nodes': list(range(n_nodes := int(G.number_of_nodes()))),
            'source': source,
            'sink': sink,
            'edges': edges,
        }
        answer = f"{int(value)} {','.join(str(x) for x in reachable)}"
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (
            f"{render_payload(metadata.payload)}\n\n"
            f"Consider a directed graph with capacities on every edge. "
            f"Source is node {metadata.payload['source']} and sink is node {metadata.payload['sink']}. "
            f"Compute the maximum flow value from source to sink, then list the nodes on the "
            f"source side of the minimum cut in ascending order. "
            f"The answer is the flow value followed by a space and the comma-separated nodes "
            f"of the source side of the cut, e.g. \"5 0,1,3\"."
        )

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        text = str(answer).strip()
        try:
            parts = text.split()
            if len(parts) != 2:
                return 0.0
            val = int(parts[0])
            nodes = [int(x) for x in parts[1].split(',')]
        except ValueError:
            return 0.0
        gold_val = int(entry.metadata.flow_value)
        if val != gold_val:
            return 0.0
        # A minimum cut is not unique, so score the cut the answer describes rather
        # than the one the generator happened to find: any source side whose outgoing
        # capacity equals the max flow is a minimum cut (max-flow min-cut theorem).
        source = int(entry.metadata.source)
        sink = int(entry.metadata.sink)
        side = set(nodes)
        if source not in side or sink in side:
            return 0.0
        if not side <= {int(u) for u, _, _ in entry.metadata.edges} | {
                int(v) for _, v, _ in entry.metadata.edges}:
            return 0.0
        cut = sum(int(c) for u, v, c in entry.metadata.edges
                  if int(u) in side and int(v) not in side)
        return 1.0 if cut == gold_val else 0.0


from reasoning_core.template import render_payload
