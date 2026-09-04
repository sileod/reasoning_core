from dataclasses import dataclass
from collections import deque

import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


@dataclass
class BipartiteMatchingConfig(Config):
    min_left: int = 4
    max_left: int = 7
    min_right: int = 4
    max_right: int = 7
    edge_prob: float = 0.5

    def apply_difficulty(self, level):
        self.min_left = sround(self.min_left + level)
        self.max_left = sround(self.max_left + level)
        self.min_right = sround(self.min_right + level)
        self.max_right = sround(self.max_right + level)
        self.edge_prob = self.edge_prob + 0.02 * level


def _hopcroft_karp(nL, nR, adj):
    pairL = [-1] * nL
    pairR = [-1] * nR
    dist = [0] * nL
    INF = float("inf")

    def bfs():
        q = deque()
        for l in range(nL):
            if pairL[l] == -1:
                dist[l] = 0
                q.append(l)
            else:
                dist[l] = INF
        found = False
        while q:
            l = q.popleft()
            for r in adj[l]:
                l2 = pairR[r]
                if l2 == -1:
                    found = True
                elif dist[l2] == INF:
                    dist[l2] = dist[l] + 1
                    q.append(l2)
        return found

    def dfs(l):
        for r in adj[l]:
            l2 = pairR[r]
            if l2 == -1 or (dist[l2] == dist[l] + 1 and dfs(l2)):
                pairL[l] = r
                pairR[r] = l
                return True
        dist[l] = INF
        return False

    while bfs():
        for l in range(nL):
            if pairL[l] == -1:
                dfs(l)
    return pairL, pairR


def _has_augmenting_path(nL, nR, adj, pairL, pairR):
    visited = [False] * nL
    q = deque()
    for l in range(nL):
        if pairL[l] == -1:
            visited[l] = True
            q.append(l)
    while q:
        l = q.popleft()
        for r in adj[l]:
            if r == pairL[l]:
                continue
            l2 = pairR[r]
            if l2 == -1:
                return True
            if not visited[l2]:
                visited[l2] = True
                q.append(l2)
    return False


def _render_edges(nL, nR, adj):
    lines = []
    for l in range(nL):
        rs = [r for r in adj[l]]
        lines.append("L%d: %s" % (l, ", ".join("R%d" % r for r in rs) if rs else "-"))
    return "\n".join(lines)


class BipartiteMatching(Task):
    summary = ("Construct a canonical maximum bipartite matching via ordered augmenting paths and "
               "report the partner of a queried left vertex as an index or None.")
    config_cls = BipartiteMatchingConfig
    task_version = 2

    def generate_entry(self):
        cfg = self.config
        nL = random.randint(cfg.min_left, cfg.max_left)
        nR = random.randint(cfg.min_right, cfg.max_right)
        p = cfg.edge_prob

        for _ in range(400):
            adj = []
            for l in range(nL):
                cand = [r for r in range(nR) if random.random() < p]
                if not cand:
                    cand = [random.randrange(nR)]
                adj.append(sorted(set(cand)))

            pairL, pairR = _hopcroft_karp(nL, nR, adj)

            if _has_augmenting_path(nL, nR, adj, pairL, pairR):
                continue

            if sum(1 for x in pairL if x != -1) == 0:
                continue

            q = random.randrange(nL)
            partner = pairL[q]
            answer = str(partner) if partner != -1 else "None"

            metadata = edict({
                "left": nL,
                "right": nR,
                "edges": _render_edges(nL, nR, adj),
                "query": q,
                "algorithm": "augment along shortest augmented paths, "
                             "visiting unmatched left vertices and neighbors in increasing label order, "
                             "until none remain",
            })
            metadata.payload = {
                "Bipartite graph": "%d left vertices (L0..L%d) and %d right vertices (R0..R%d).\n"
                                  "Edges (left vertex: adjacent right vertices):\n%s"
                                  % (nL, nL - 1, nR, nR - 1, metadata.edges),
                "Canonical maximum matching": metadata.algorithm,
                "Question": "What is the partner of %s in the canonical maximum matching described above?"
                            % ("L%d" % q),
                "Answer format": "write only the partner's right index (the number k for Rk), or None if "
                                 "the queried left vertex is unmatched.",
            }
            return Entry(metadata=metadata, answer=answer)
        raise RuntimeError("bipartite_matching: could not build a valid instance")

    def render_prompt(self, metadata):
        return render_payload(metadata.payload)

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        return 1.0 if str(answer).strip() == str(entry.answer).strip() else 0.0

    def distractor_candidates(self, entry):
        adj_lines = entry.metadata.edges.splitlines()
        q = entry.metadata.query
        if q >= len(adj_lines):
            return
        tokens = adj_lines[q].split(":", 1)[1].replace("R", " ").replace(",", " ").replace("-", "")
        cands = [t for t in tokens.split() if t.isdigit()]
        for c in cands:
            yield c
        yield "None"
        yield str(q)


TASK_META = {'parent_source_id': None,
 'idea': 'bipartite_matching (draw 1 of 1)',
 'hypothesis': 'HV-015',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/bipartite_matching',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3986913703,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
