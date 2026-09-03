import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'poset_cover_query (draw 1 of 2)',
 'hypothesis': 'W1-025',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/poset_cover_query',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2960624742,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


@dataclass
class PosetCoverQueryConfig(Config):
    n_elements: int = 8
    edge_density: float = 0.5
    n_extra: int = 2

    def apply_difficulty(self, level):
        self.n_elements = 6 + sround(2.0 * level)
        self.edge_density = min(0.9, 0.4 + 0.08 * level)
        self.n_extra = 1 + sround(level)


def _build_poset(names, n_extra):
    """Build a poset (as transitive reduction edge set + cover map) on names.

    Returns (edge_set, covers) where edge_set is the set of comparability
    relations and covers maps each element to its sorted list of covering
    elements.
    """
    order = list(names)
    random.shuffle(order)
    edge_set = set()
    # guarantee a chain backbone so the poset is non-trivial
    edge_set.add((order[0], order[1]))
    for i in range(1, len(order) - 1):
        if random.random() < 0.6:
            edge_set.add((order[i], order[i + 1]))
    # add extra independences edges
    for _ in range(n_extra):
        u = random.choice(names)
        v = random.choice(names)
        if u != v and (u, v) not in edge_set and (v, u) not in edge_set:
            edge_set.add((u, v))
    # transitive closure
    adj = {x: [] for x in names}
    for (u, v) in edge_set:
        adj[u].append(v)
    reach = {}
    for x in names:
        seen = set()
        stack = [x]
        while stack:
            node = stack.pop()
            for nxt in adj[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        reach[x] = seen
    # covers
    covers = {}
    for x in names:
        cv = []
        for y in reach[x]:
            mids = [z for z in reach[x] if z != y and y in reach[z]]
            if not mids:
                cv.append(y)
        covers[x] = sorted(cv)
    return edge_set, covers


class PosetCoverQuery(Task):
    summary = ("Given a finite partial order as comparability relations, output the "
               "elements that directly cover a queried element (its lower cover set), "
               "last value NONE when empty, across varied poset families under "
               "different element counts and edge densities.")
    config_cls = PosetCoverQueryConfig
    task_version = 2

    def generate_entry(self):
        c = self.config
        n = int(c.n_elements)
        names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"][:n]
        n_extra = int(c.n_extra)

        # resample until we get an informative query (both empty and non-empty
        # cover sets appear across instances to balance the label space)
        for _ in range(200):
            edge_set, covers = _build_poset(names, n_extra)
            query = random.choice(names)
            cv = covers[query]
            if cv or random.random() < 0.5:
                break

        answer_set = sorted(cv)
        answer = ", ".join(answer_set) if answer_set else "NONE"

        rel = "; ".join(f"{u} < {v}" for (u, v) in sorted(edge_set))

        payload = {
            "elements": names,
            "relations": rel,
            "query": query,
        }
        metadata = edict({
            "elements": names,
            "relations": sorted(edge_set),
            "query": query,
            "cover": answer_set,
        })
        metadata.payload = payload
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        p = metadata.payload
        els = ", ".join(p["elements"])
        return (
            f"Consider the finite partial order on the elements {{{els}}}. "
            f"The order is specified by comparability relations, where u < v means "
            f"u is strictly below v: {p['relations']}. "
            f"Element x is said to be covered by element y when x < y and there is "
            f"no third element z with x < z < y. "
            f"List all elements that cover {p['query']}, that is every y such that "
            f"{p['query']} < y and no element lies strictly between {p['query']} "
            f"and y. Give them in increasing alphabetical order separated by commas, "
            f"or give the single word NONE if there are none.\n\n"
            f"The answer is:"
        )

    def score_answer(self, answer, entry):
        s = answer.strip()
        if s == "":
            return 0.0
        return 1.0 if s == entry.answer else 0.0
