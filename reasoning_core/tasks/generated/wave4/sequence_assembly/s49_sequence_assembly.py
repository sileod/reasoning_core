import random
from dataclasses import dataclass, field
from collections import Counter, deque

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround
from reasoning_core.utils import score_scalar

TASK_META = {'parent_source_id': None,
 'idea': 'Add sequence assembly from overlapping fragments.',
 'hypothesis': 'S49',
 'changes': 'Ask for the sequence that a set of k-mers or fragments '
            'reconstructs.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3175732428,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def build_graph(frags):
    in_deg = Counter()
    out_deg = Counter()
    edges = Counter()
    src_nodes = set()
    dst_nodes = set()
    for f in frags:
        s = f[:-1]
        d = f[1:]
        edges[(s, d)] += 1
        out_deg[s] += 1
        in_deg[d] += 1
        src_nodes.add(s)
        dst_nodes.add(d)
    return edges, in_deg, out_deg, src_nodes, dst_nodes


def traverse_all(edges, src_nodes, dst_nodes):
    all_nodes = src_nodes | dst_nodes
    # vertices with a nonzero incident edge count are those with in or out > 0
    active = {n for n in all_nodes if edges.get((n, n), 0) > 0 or
              sum(edges[(n, d)] for d, _ in [(n, x) for x in all_nodes]) > 0 or
              sum(edges[(s, n)] for s in all_nodes) > 0}
    # simpler: build adjacency
    adj = {}
    for (s, d), c in edges.items():
        adj.setdefault(s, []).extend([d] * c)
    # start from nodes with out=1 and in=0 (true starts), or any active
    starts = [n for n in all_nodes if in_deg_from(edges, n, all_nodes) == 0 and out_deg_from(edges, n, all_nodes) > 0]
    if not starts:
        starts = list(active)
    if not starts:
        return True
    # Do a DFS counting distinct Eulerian trails. Cap enumeration.
    LIMIT = 3
    count = 0

    todo = []

    def rec(node, rem):
        nonlocal count
        if count >= LIMIT:
            return
        outs = adj.get(node, [])
        moved = False
        for d in list(outs):
            outs.remove(d)
            rec(d, rem - 1)
            outs.insert(0, d)
            moved = True
        if rem == 0:
            count += 1
        del moved

    # enumerate only over edges actually present
    total_edges = sum(edges.values())
    for start in starts:
        rec(start, total_edges)
        if count >= LIMIT:
            return True
    return count > 1


def in_deg_from(edges, n, all_nodes):
    return sum(c for (s, d), c in edges.items() if d == n)


def out_deg_from(edges, n, all_nodes):
    return sum(c for (s, d), c in edges.items() if s == n)


def count_eulerian_trails(edges, all_nodes):
    adj = {}
    for (s, d), c in edges.items():
        adj.setdefault(s, []).extend([d] * c)
    total_edges = sum(edges.values())
    starts = [n for n in all_nodes if in_deg_from(edges, n, all_nodes) == 0 and out_deg_from(edges, n, all_nodes) > 0]
    if not starts:
        starts = [n for n in all_nodes if (in_deg_from(edges, n, all_nodes) + out_deg_from(edges, n, all_nodes)) > 0]
    LIMIT = 3
    count = 0

    for start in starts:
        count += _dfs_count(start, adj, total_edges, LIMIT - count)
        if count >= LIMIT:
            return LIMIT
    return count


def _dfs_count(node, adj, rem, limit):
    count = 0
    outs = list(adj.get(node, []))
    # use pointer-based edges
    def rec(n, r):
        nonlocal count
        if count >= limit:
            return
        a = adj.get(n, [])
        if not a:
            if r == 0:
                count += 1
            return
        i = 0
        while i < len(a):
            d = a.pop(i)
            rec(d, r - 1)
            a.insert(i, d)
            i += 1
    rec(node, rem)
    return count


@dataclass
class ArrayConfig(Config):
    length: int = 8
    alphabet_size: int = 4
    k: int = 3
    ambiguous_fraction: float = 0.17

    def apply_difficulty(self, level):
        self.length = sround(self.length + 2 * level)
        self.alphabet_size = sround(self.alphabet_size - level)
        self.alphabet_size = max(2, self.alphabet_size)
        self.k = sround(self.k + level)


class SequenceAssembly(Task):
    config_cls = ArrayConfig

    def generate_entry(self):
        c = self.config
        string_length = int(c.length)
        alpha_size = int(c.alphabet_size)
        k = int(c.k)

        alphabet = 'ACGT'[:alpha_size]
        if len(alphabet) < alpha_size:
            alphabet = ''.join(chr(ord('a') + i) for i in range(alpha_size))

        # Decide ambiguity first
        if random.random() < float(c.ambiguous_fraction):
            string, frags, unique = generate_ambiguous(alpha_size, k, string_length)
        else:
            string, frags, unique = generate_unique(alpha_size, k, string_length)

        # Verify
        ct = Counter(frags)
        if unique:
            assert Counter(string[i:i + k] for i in range(len(string) - k + 1)) == ct
        else:
            # ambiguous: at least two distinct strings produce same multiset
            pass

        payload = {"fragments": sorted(frags)}
        metadata = edict({
            "string_length": string_length,
            "alphabet_size": alpha_size,
            "k": k,
            "ambiguous": unique,
            "answer": string if unique else "ambiguous",
            "payload": payload,
        })
        answer = string if unique else "ambiguous"
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (
            f"The following is a multiset of length-{metadata.k} overlapping "
            f"fragments of an unknown string over an alphabet of size "
            f"{metadata.alphabet_size}:\n\n"
            f"{render_payload(metadata.payload)}\n\n"
            f"If exactly one string over the alphabet can explain these fragments, "
            f"give that string as the answer. If more than one string explains them, "
            f"give the single word \"ambiguous\" as the answer."
        )

    def score_answer(self, answer, entry):
        gold = entry.answer
        if answer is None:
            return 0.0
        a = str(answer).strip().lower()
        g = gold.strip().lower()
        if g == "ambiguous":
            return 1.0 if a == "ambiguous" else 0.0
        return 1.0 if a == g else 0.0


def generate_unique(alpha_size, k, length):
    alphabet = 'ACGT'[:alpha_size]
    if len(alphabet) < alpha_size:
        alphabet = ''.join(chr(ord('a') + i) for i in range(alpha_size))
    for _ in range(2000):
        s = ''.join(random.choice(alphabet) for _ in range(length))
        frags = [s[i:i + k] for i in range(length - k + 1)]
        edges, in_deg, out_deg, src, dst = build_graph(frags)
        all_nodes = src | dst
        n_trails = count_eulerian_trails(edges, all_nodes)
        if n_trails == 1:
            return s, frags, True
    raise RuntimeError("failed to generate unique instance")


def generate_ambiguous(alpha_size, k, length):
    alphabet = 'ACGT'[:alpha_size]
    if len(alphabet) < alpha_size:
        alphabet = ''.join(chr(ord('a') + i) for i in range(alpha_size))
    for _ in range(2000):
        s = ''.join(random.choice(alphabet) for _ in range(length))
        frags = [s[i:i + k] for i in range(length - k + 1)]
        # build graph and check if more than one distinct trail exists
        edges, in_deg, out_deg, src, dst = build_graph(frags)
        all_nodes = src | dst
        n_trails = count_eulerian_trails(edges, all_nodes)
        if n_trails >= 2:
            return s, frags, False
    raise RuntimeError("failed to generate ambiguous instance")
