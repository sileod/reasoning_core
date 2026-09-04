import random

from reasoning_core.template import Task, Entry, Config, edict

TASK_META = {'parent_source_id': None,
 'idea': 'transitive_reduction (draw 1 of 1)',
 'hypothesis': 'HV-017',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/transitive_reduction',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1958495724,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                              'sandbox': {'name': 'bubblewrap',
                                          'version': 'bubblewrap 0.8.0'}}}}


def _label(i):
    return f"n{i}"


def _reachable(adj, s):
    seen = {s}
    stack = list(adj[s])
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for w in adj[cur]:
            if w not in seen:
                stack.append(w)
    return seen


def reachability_edges(edges, n):
    adj = {u: set() for u in range(n)}
    for u, v in edges:
        adj[u].add(v)
    out = set()
    for s in range(n):
        for t in _reachable(adj, s):
            if t != s:
                out.add((s, t))
    return out


def transitive_reduction(edges, n):
    adj = {u: set() for u in range(n)}
    for u, v in edges:
        adj[u].add(v)
    reduced = set()
    for u, v in edges:
        alt = adj[u] - {v}
        saw = set(alt)
        stack = list(alt)
        found = False
        while stack:
            cur = stack.pop()
            if cur == v:
                found = True
                break
            for w in adj[cur]:
                if w not in saw:
                    saw.add(w)
                    stack.append(w)
        if not found:
            reduced.add((u, v))
    return reduced


def _random_dag(n, m):
    order = list(range(n))
    random.shuffle(order)
    pos = {v: i for i, v in enumerate(order)}
    edges = set()
    tries = 0
    while len(edges) < m and tries < 10000:
        u = random.randrange(n)
        v = random.randrange(n)
        if pos[u] < pos[v]:
            edges.add((u, v))
        tries += 1
    return list(edges)


def _desc_payload(edges, n):
    pairs = sorted((_label(u), _label(v)) for u, v in edges)
    lines = " ".join(f"{a} -> {b}" for a, b in pairs)
    return {"n": int(n), "edges": lines}


def edges_to_answer(red):
    return "; ".join(f"{_label(u)} -> {_label(v)}" for u, v in sorted(red))


class TransitiveReductionConfig(Config):
    n_nodes: int = 7
    density: float = 0.35

    def apply_difficulty(self, level):
        self.n_nodes = int(7 + round(1.7 * level))
        self.density = min(0.55, 0.30 + 0.03 * level)


class TransitiveReduction(Task):
    summary = ("Generate random labeled DAGs and either enumerate the edges of "
               "the unique transitive reduction (removing edges whose endpoints "
               "already connect via an alternative path) as a sorted edge list, "
               "or answer whether a queried edge belongs to the reduction; "
               "output regimes are multi-edge sortable lists and binary status.")

    config_cls = TransitiveReductionConfig

    def generate_entry(self):
        n = self.config.n_nodes
        for _ in range(300):
            max_edges = n * (n - 1) // 2
            lo = max(n - 1, 2)
            hi = min(n * 3, max_edges)
            if hi < lo:
                hi = lo
            m = random.randint(lo, hi)
            edges = _random_dag(n, m)
            red = transitive_reduction(edges, n)
            if reachability_edges(edges, n) == reachability_edges(list(red), n):
                break
        else:
            return self.generate_entry()

        if random.random() < 0.85:
            answer = edges_to_answer(list(red))
            payload = _desc_payload(edges, n)
        else:
            red_set = set(red)
            if random.random() < 0.5 and edges:
                qe = random.choice(edges)
            else:
                candidates = []
                for _ in range(200):
                    u = random.randrange(n)
                    v = random.randrange(n)
                    if u != v:
                        candidates.append((u, v))
                qe = random.choice(candidates)
            payload = _desc_payload(edges, n)
            payload["query"] = f"{_label(qe[0])} -> {_label(qe[1])}"
            answer = "yes" if qe in red_set else "no"

        metadata = edict(payload)
        metadata.payload = payload
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        body = (f"A directed acyclic graph has nodes n0..n{metadata['n'] - 1} "
                f"and directed edges: {metadata['edges']}.")
        if "query" in metadata:
            return (body + " The transitive reduction of a DAG is the unique "
                    "minimal subgraph with the same reachability, obtained by "
                    "removing every edge whose endpoints are already connected "
                    "by an alternative directed path. Is the edge "
                    f"{metadata['query']} present in the resulting transitive "
                    "reduction? The answer is 'yes' or 'no'.")
        return (body + " The transitive reduction of a DAG is the unique "
                "minimal subgraph with the same reachability, obtained by "
                "removing every edge whose endpoints are already connected by "
                "an alternative directed path. List the edges of this DAG's "
                "transitive reduction, each in the form 'a -> b', separated by "
                "a semicolon and sorted lexicographically by source then "
                "target, for example 'n0 -> n1; n0 -> n3'.")

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        if "query" in entry.metadata:
            return 1.0 if answer.strip().lower() == entry.answer else 0.0
        expected = set(x.strip() for x in entry.answer.split(";"))
        got = set(x.strip() for x in answer.split(";"))
        return 1.0 if got == expected else 0.0
