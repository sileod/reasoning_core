from dataclasses import dataclass
import random

import networkx as nx

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'eulerian_status (draw 1 of 2)',
 'hypothesis': 'W1-005',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/eulerian_status',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3825253341,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class EulerianStatusConfig(Config):
    n_verts: int = 8
    switches: int = 6

    def apply_difficulty(self, level):
        self.n_verts = sround(self.n_verts + level * 2)
        self.switches = sround(self.switches + level)


def _build_graph(n, switches, status):
    for _ in range(200):
        vs = list(range(n))
        random.shuffle(vs)
        edges = set()
        for i in range(n):
            a, b = vs[i], vs[(i + 1) % n]
            edges.add((min(a, b), max(a, b)))
        for _s in range(switches):
            el = random.sample(sorted(edges), min(2, len(edges))) if len(edges) >= 2 else []
            if len(el) != 2:
                break
            e1, e2 = el
            a, b = e1
            c, d = e2
            if len({a, b, c, d}) < 4:
                continue
            done = False
            for (x1, y1), (x2, y2) in (((a, c), (b, d)), ((a, d), (b, c))):
                e3 = (min(x1, y1), max(x1, y1))
                e4 = (min(x2, y2), max(x2, y2))
                if e3 in edges or e4 in edges or e3 == e4:
                    continue
                edges.discard(e1)
                edges.discard(e2)
                edges.add(e3)
                edges.add(e4)
                done = True
                break
            if not done:
                continue
        if status == "circuit":
            pass
        elif status == "open":
            pairs = list(edges)
            added = False
            for _a in range(50):
                u = random.randrange(n)
                v = random.randrange(n)
                if u == v:
                    continue
                e = (min(u, v), max(u, v))
                if e in edges:
                    continue
                edges.add(e)
                added = True
                break
            if not added:
                continue
        else:
            k = random.randint(2, max(2, min(4, n // 3)))
            verts = list(range(n))
            random.shuffle(verts)
            targets = verts[:2 * k]
            ok = True
            for i in range(k):
                e = (min(targets[2 * i], targets[2 * i + 1]),
                     max(targets[2 * i], targets[2 * i + 1]))
                if e in edges:
                    ok = False
                    break
            if not ok:
                continue
            for i in range(k):
                e = (min(targets[2 * i], targets[2 * i + 1]),
                     max(targets[2 * i], targets[2 * i + 1]))
                edges.add(e)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(sorted(edges))
        if not nx.is_connected(G):
            continue
        perm = list(range(n))
        random.shuffle(perm)
        relab = {(min(perm[u], perm[v]), max(perm[u], perm[v])) for (u, v) in edges}
        relab = sorted(relab)
        G2 = nx.Graph()
        G2.add_nodes_from(range(n))
        G2.add_edges_from(relab)
        deg = dict(G2.degree())
        odd = sorted([v for v in range(n) if deg[v] % 2 == 1])
        n_odd = len(odd)
        if status == "circuit" and n_odd == 0:
            ans = "circuit %d" % min(range(n))
            return relab, ans
        if status == "open" and n_odd == 2:
            ans = "open %d" % odd[0]
            return relab, ans
        if status == "none" and n_odd >= 4:
            ans = "none %d" % n_odd
            return relab, ans
    raise RuntimeError("could not build graph")


class EulerianStatus(Task):
    summary = ("Classify an undirected connected graph as none (reporting the count of odd "
               "vertices), an open Euler trail (reporting the smaller odd-degree start), or an "
               "Euler circuit (reporting the smallest start) over cycle-with-chords instances.")
    config_cls = EulerianStatusConfig

    def generate_entry(self):
        cfg = self.config
        status = random.choice(["circuit", "open", "none"])
        edges, ans = _build_graph(cfg.n_verts, cfg.switches, status)
        metadata = edict({
            "status": status,
            "n_verts": cfg.n_verts,
            "edges": [[int(u), int(v)] for (u, v) in edges],
        })
        metadata.payload = {"edges": metadata.edges, "n_verts": metadata.n_verts}
        return Entry(metadata=metadata, answer=ans)

    def render_prompt(self, metadata):
        n = metadata.n_verts
        edge_lines = ", ".join("(%d,%d)" % (u, v) for (u, v) in metadata.edges)
        body = (
            "Consider the undirected graph G whose vertices are numbered 0 through %d inclusive "
            "and whose edges are: %s. "
            "G is connected. An Euler circuit exists when every vertex has even degree; an open "
            "Euler trail exists when exactly two vertices have odd degree; otherwise G has no "
            "Euler trail. "
            "Classify G and give its canonical start. "
            "Write the answer in exactly one of these three forms: "
            "'none K' where K is the number of odd-degree vertices; "
            "'open V' where V is the smaller of the two odd-degree vertices; "
            "'circuit V' where V is the smallest-numbered vertex."
        ) % (n - 1, edge_lines)
        payload = {"edges": metadata.edges, "var": body}
        return render_payload(payload)

    def score_answer(self, answer, entry):
        return 1.0 if str(answer).strip() == entry.answer else 0.0
