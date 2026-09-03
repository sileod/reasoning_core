"""Given a build dependency DAG and changed sources, output all targets that must rebuild."""

import random
import string
from dataclasses import dataclass
from itertools import product

import networkx as nx

from reasoning_core.template import Config, Entry, Task, edict, render_payload, stochastic_rounding as sround


def _names(n_nodes):
    names = []
    for k in range(1, 4):
        for tup in product(string.ascii_uppercase, repeat=k):
            names.append("".join(tup))
            if len(names) >= n_nodes:
                return names[:n_nodes]
    return names[:n_nodes]


def _parse_answer(answer):
    if answer is None:
        return None
    names = [p.strip() for p in str(answer).replace(",", " ").split() if p.strip()]
    return sorted(set(names)) if names else None


def _compute_closure(edges_by_index, n_nodes, changed_idx):
    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        for j in edges_by_index[i]:
            g.add_edge(j, i)
    rebuild_idx = set()
    for c in changed_idx:
        rebuild_idx.add(c)
        rebuild_idx.update(nx.descendants(g, c))
    return sorted(rebuild_idx)


@dataclass
class RebuildSetConfig(Config):
    n_nodes: int = 5
    n_changed: int = 1
    edge_prob: float = 0.5

    def apply_difficulty(self, level):
        self.n_nodes = sround(5 + 2 * level)
        self.n_changed = sround(2 + level // 2)
        self.edge_prob = 0.5


class RebuildSet(Task):
    task_name = "incremental_build_rebuild_set"
    summary = (
        "Given a build dependency DAG and changed sources, output all targets that must "
        "rebuild: the changed sources themselves plus every target that directly or "
        "transitively depends on them, read off the transitive closure."
    )
    config_cls = RebuildSetConfig

    def generate_entry(self):
        cfg = self.config
        n = cfg.n_nodes
        names = _names(n)

        edges_by_index = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i):
                if random.random() < cfg.edge_prob:
                    edges_by_index[i].append(j)

        n_ch = min(cfg.n_changed, n)
        changed_idx = sorted(random.sample(range(n), n_ch))
        rebuild_idx = _compute_closure(edges_by_index, n, changed_idx)

        rebuild_names = [names[i] for i in rebuild_idx]
        changed_names = [names[i] for i in changed_idx]

        dep_text = []
        for i in range(n):
            deps = [names[j] for j in sorted(edges_by_index[i])]
            dep_text.append(names[i] + (": " + ", ".join(deps) if deps else ""))

        payload = {
            "Targets (dependency list)": "\n".join(dep_text),
            "Changed sources": ", ".join(changed_names),
        }

        gold = ", ".join(rebuild_names)

        assert len(rebuild_names) >= 1, "closure never empty (changed nodes rebuild)"
        assert set(changed_names).issubset(set(rebuild_names))
        assert _parse_answer(gold) == sorted(set(rebuild_names))

        metadata = edict({"payload": payload, "rebuild": list(rebuild_names)})
        return Entry(metadata=metadata, answer=gold)

    def render_prompt(self, metadata):
        return (
            render_payload(metadata.payload)
            + "\n\nList every target that must be rebuilt: each changed source itself, plus "
            "every target that directly or transitively depends on a changed source. "
            "Answer as a single comma-separated list sorted in alphabetical order."
        )

    def score_answer(self, answer, entry):
        parsed = _parse_answer(answer)
        if parsed is None:
            return 0.0
        gold = sorted(set(entry.metadata["rebuild"]))
        return 1.0 if parsed == gold else 0.0


Task.register(RebuildSet)


TASK_META = {'parent_source_id': None,
 'idea': 'incremental_build_rebuild_set (draw 1 of 2)',
 'hypothesis': 'W1-079',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/incremental_build_rebuild_set',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 4082635274,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
