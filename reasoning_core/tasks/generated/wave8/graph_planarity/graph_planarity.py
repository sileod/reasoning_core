import random
import math
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'graph_planarity (draw 2 of 2)',
 'hypothesis': 'W1-007',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/graph_planarity',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3982003255,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class GraphPlanarityConfig(Config):
    v_lo: int = 3
    v_hi: int = 12

    def apply_difficulty(self, level):
        self.v_lo = sround(self.v_lo + level)
        self.v_hi = sround(self.v_hi + 2 * level)


def _genus_complete(n):
    if n < 3:
        return 0
    return math.ceil((n - 3) * (n - 4) / 12.0)


def _genus_complete_bipartite(a, b):
    if a < 2 or b < 2:
        return 0
    return math.ceil((a - 2) * (b - 2) / 4.0)


class GraphPlanarity(Task):
    summary = "Answer whether undirected complete graphs K_n and complete bipartite graphs K_{a,b} are planar by reporting their genus: a nonnegative integer, 0 iff planar, from the Ringel-Youngs formulas."

    config_cls = GraphPlanarityConfig

    def generate_entry(self):
        cfg = self.config
        if random.random() < 0.5:
            n = random.randint(cfg.v_lo, cfg.v_hi)
            genus = _genus_complete(n)
            label = "complete graph"
            spec = f"the complete graph K_{n} on {n} vertices, in which every pair of vertices is joined by an edge"
        else:
            a = random.randint(cfg.v_lo, cfg.v_hi)
            b = random.randint(cfg.v_lo, cfg.v_hi)
            if a > b:
                a, b = b, a
            genus = _genus_complete_bipartite(a, b)
            label = "complete bipartite graph"
            spec = f"the complete bipartite graph K_{{{a},{b}}}, whose vertices split into two groups of sizes {a} and {b} and every vertex of one group is adjacent to every vertex of the other (no edges within a group)"

        assert genus >= 0, genus
        expected = _genus_complete(n) if label == "complete graph" else _genus_complete_bipartite(a, b)
        assert genus == expected

        planar = "Planar" if genus == 0 else "Not planar"
        metadata = edict({
            "payload": {
                "graph": f"Consider {spec}.",
                "genus_note": ("The genus of a graph is the minimum number of holes of a surface on "
                               "which it can be drawn with no edge crossings; a graph is planar iff "
                               "its genus is 0."),
            },
            "label": label,
            "planar": planar,
            "size_info": (f"{n}" if label == "complete graph" else f"{a},{b}"),
        })
        return Entry(metadata=metadata, answer=str(genus))

    def render_prompt(self, metadata):
        payload = render_payload(metadata.payload)
        return (f"{payload}\n\nReport the genus of {metadata.label}.\n\n"
                f"The answer is a single nonnegative integer (0 means the graph is planar).")

    def score_answer(self, answer, entry):
        try:
            return 1.0 if float(answer) == float(entry.answer) else 0.0
        except (TypeError, ValueError):
            return 0.0
