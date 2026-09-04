import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict


@dataclass
class TopologicalLayeringConfig(Config):
    n_nodes: int = 4
    n_edges: int = 5

    def apply_difficulty(self, level):
        self.n_nodes = max(3, 4 + level)
        self.n_edges = int((self.n_nodes * (self.n_nodes - 1)) // 3)


def _is_dag(n, edges):
    indeg = [0] * n
    succ = [[] for _ in range(n)]
    for u, v in edges:
        succ[u].append(v)
        indeg[v] += 1
    import heapq
    queue = sorted((x for x in range(n) if indeg[x] == 0))
    processed = 0
    while queue:
        u = heapq.heappop(queue)
        processed += 1
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(queue, v)
    return processed == n


def _removal_order(n, edges, tie):
    indeg = [0] * n
    succ = [[] for _ in range(n)]
    for u, v in edges:
        succ[u].append(v)
        indeg[v] += 1
    remaining = set(range(n))
    order = []
    for _ in range(n):
        cand = sorted(x for x in remaining if indeg[x] == 0)
        if not cand:
            return None
        pick = cand[-1] if tie == "max" else cand[0]
        order.append(pick)
        remaining.discard(pick)
        for v in succ[pick]:
            indeg[v] -= 1
    return order


def _stringify_graph(n, edges):
    return "\n".join(f"{u} -> {v}" for u, v in sorted(edges))


class TopologicalLayering(Task):
    summary = "Remove current zero-indegree nodes from a DAG one round at a time under a stated tie rule (smallest or largest id), returning the queried round's removed nodes or the full canonical removal order."

    config_cls = TopologicalLayeringConfig

    def generate_entry(self):
        n = self.config.n_nodes
        n_edges_goal = self.config.n_edges
        mode = random.choice(["round", "order"])
        tie = random.choice(["min", "max"])

        while True:
            perm = list(range(n))
            random.shuffle(perm)
            pos = {node: i for i, node in enumerate(perm)}
            candidates = [
                (u, v)
                for u in range(n)
                for v in range(n)
                if u != v and pos[u] < pos[v]
            ]
            random.shuffle(candidates)
            edges = candidates[:n_edges_goal]
            order = _removal_order(n, edges, tie)
            if order is None:
                continue
            break

        edges_str = _stringify_graph(n, edges)
        tie_txt = (
            "remove the smallest-indexed zero-indegree node first"
            if tie == "min"
            else "remove the largest-indexed zero-indegree node first"
        )

        if mode == "round":
            nodes_per_round = 2
            n_rounds = (n + nodes_per_round - 1) // nodes_per_round
            picked_round = random.randint(1, n_rounds)
            low = (picked_round - 1) * nodes_per_round
            high = min(n, low + nodes_per_round)
            removed = order[low:high]
            answer = ",".join(str(x) for x in sorted(removed))
            payload = {
                "nodes": " ".join(str(i) for i in range(n)),
                "edges": edges_str,
                "tie": tie_txt,
                "round": picked_round,
                "nodes_per_round": nodes_per_round,
            }
            query = (
                f"Which nodes are removed in round {picked_round}? Give them as a "
                f"comma-separated list in ascending order."
            )
        else:
            answer = ",".join(str(x) for x in order)
            payload = {
                "nodes": " ".join(str(i) for i in range(n)),
                "edges": edges_str,
                "tie": tie_txt,
            }
            query = (
                "List every node in the exact order it is removed, as a "
                "comma-separated list of node numbers."
            )

        metadata = edict(payload)
        metadata.payload = payload
        metadata.query = query
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        payload = metadata.payload
        base = (
            f"Consider the directed acyclic graph on nodes {{{payload['nodes']}}}.\n"
            f"Edges:\n{payload['edges']}\n"
            f"Rule: {payload['tie']}.\n"
        )
        if "round" in payload:
            return (
                base
                + f"The nodes are removed in rounds, {payload['nodes_per_round']} per "
                f"round, always removing zero-indegree nodes in the order the rule "
                f"specifies, until the graph is empty.\n"
                f"Which nodes are removed in round {payload['round']}?\n"
                f"The answer is the comma-separated list of those node numbers in "
                f"ascending order."
            )
        return (
            base
            + "The nodes are removed one at a time, always removing zero-indegree "
            f"nodes in the order the rule specifies, until the graph is empty.\n"
            "List every node in the exact order it is removed, as a comma-separated "
            "list of node numbers."
        )

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        norm = "".join(answer.split())
        expected = "".join(entry.answer.split())
        return 1.0 if norm == expected else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'topological_layering (draw 1 of 1)',
 'hypothesis': 'HV-012',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/topological_layering',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1277236794,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
