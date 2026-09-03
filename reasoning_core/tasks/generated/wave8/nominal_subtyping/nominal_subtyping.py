import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict
from reasoning_core.template import stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'nominal_subtyping (draw 2 of 2)',
 'hypothesis': 'W1-055',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/nominal_subtyping',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 823763445,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


class Node(object):
    __slots__ = ('name', 'parents')

    def __init__(self, name):
        self.name = name
        self.parents = []


def build_hierarchy(n_nodes, n_sources):
    names = ['T%d' % i for i in range(n_nodes)]
    nodes = [Node(n) for n in names]
    sources = random.sample(nodes, n_sources)
    others = [n for n in nodes if n not in sources]
    random.shuffle(others)
    for n in others:
        candidates = [x for x in nodes if x is not n]
        k = random.randint(1, min(4, len(candidates)))
        n.parents = random.sample(candidates, k)
    return nodes


def all_supers(node, nodes):
    seen = set()
    stack = list(node.parents)
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack.extend(p.parents)
    return seen


def subtype_exists(a, b, nodes):
    return a is b or b in all_supers(a, nodes)


def subtype_witness(a, b, nodes):
    if a is b:
        return [a.name]
    parent = {a: None}
    order = [a]
    seen = {a}
    found = False
    idx = 0
    while idx < len(order) and not found:
        x = order[idx]
        idx += 1
        for p in x.parents:
            if p in seen:
                continue
            seen.add(p)
            parent[p] = x
            order.append(p)
            if p is b:
                found = True
                break
    if not found:
        return None
    chain = []
    cur = b
    while cur is not None:
        chain.append(cur.name)
        cur = parent[cur]
    chain.reverse()
    return chain


def sorted_supers(a, nodes):
    return sorted(s.name for s in all_supers(a, nodes))


@dataclass
class NominalSubtypingConfig(Config):
    n_nodes: int = 6
    n_sources: int = 2

    def apply_difficulty(self, level):
        self.n_nodes = sround(self.n_nodes + level * 2)
        self.n_sources = sround(max(1, self.n_sources - level // 4))


class NominalSubtyping(Task):
    summary = ("Given a nominal type hierarchy, answer whether one type is a "
               "subtype of another; on subtype give the transitive inheritance "
               "chain as witness, on non-subtype give NO followed by the sorted "
               "list of that type's inherited supertypes. Modes: positive inherited "
               "chain, negative disjoint types, direct self.")
    config_cls = NominalSubtypingConfig

    def generate_entry(self):
        cfg = self.config
        n_nodes = max(4, int(cfg.n_nodes))
        n_sources = max(1, min(n_nodes - 1, int(cfg.n_sources)))

        nodes = build_hierarchy(n_nodes, n_sources)

        for _ in range(500):
            b = random.choice(nodes)
            supers = [a for a in nodes if subtype_exists(a, b, nodes)]
            non = [a for a in nodes
                   if not subtype_exists(a, b, nodes) and sorted_supers(a, nodes)]
            if not supers:
                continue
            if non and random.random() < 0.5:
                a = random.choice(non)
                positive = False
            else:
                a = random.choice(supers)
                positive = True
            break
        else:
            raise RuntimeError("could not build a valid subtype instance")

        assert subtype_exists(a, b, nodes) == positive
        if positive:
            chain = subtype_witness(a, b, nodes)
            assert chain is not None
            assert chain[0] == a.name and chain[-1] == b.name
            byname = {n.name: n for n in nodes}
            for i in range(len(chain) - 1):
                assert byname[chain[i + 1]] in byname[chain[i]].parents
            answer = "YES: " + ", ".join(chain)
            witness = list(chain)
        else:
            assert not subtype_exists(a, b, nodes)
            stray = sorted_supers(a, nodes)
            assert b.name not in stray and stray
            for sname in stray:
                assert sname in [n.name for n in nodes]
            answer = "NO: " + ", ".join(stray)
            witness = list(stray)

        hierarchy = {}
        for n in nodes:
            hierarchy[n.name] = sorted(p.name for p in n.parents)

        metadata = edict({
            "hierarchy": hierarchy,
            "a": a.name,
            "b": b.name,
            "positive": positive,
            "witness": witness,
            "mode": "positive" if positive else ("self" if a is b else "negative"),
        })
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        h = metadata.hierarchy
        lines = []
        for name in sorted(h):
            parents = h[name]
            if parents:
                lines.append(" - %s inherits from: %s" % (name, ", ".join(parents)))
            else:
                lines.append(" - %s inherits from: (none)" % name)
        head = (
            "Here is a nominal type hierarchy. Each line lists a type and the types "
            "it directly inherits from. A type X is a subtype of type Y exactly when "
            "X is Y itself or X inherits, possibly over several steps, from Y.\n"
            "The hierarchy:\n" + "\n".join(lines)
        )
        q = (
            "\nIs %s a subtype of %s?\n"
            "If yes, answer YES followed by a colon and the chain of type names from "
            "%s down to %s, separated by commas.\n"
            "If no, answer NO followed by a colon and the full list of every type %s "
            "inherits from, in sorted order, separated by commas."
            % (metadata.a, metadata.b, metadata.a, metadata.b, metadata.a)
        )
        return head + "\n" + q

    def score_answer(self, answer, entry):
        if answer is None:
            return 0.0
        gold = entry.answer
        a = str(answer).strip()
        if not a:
            return 0.0
        if gold.startswith("YES"):
            if not a.upper().startswith("YES"):
                return 0.0
            rest = a[len("YES"):].lstrip(": ").strip()
            gold_rest = gold[len("YES"):].lstrip(": ").strip()
            g = [t.strip() for t in gold_rest.split(",") if t.strip()]
            r = [t.strip() for t in rest.split(",") if t.strip()]
            if r == g:
                return 1.0
            if r and r[0] == g[0] and r[-1] == g[-1]:
                return 0.5
            return 0.25
        else:
            if not a.upper().startswith("NO"):
                return 0.0
            rest = a[len("NO"):].lstrip(": ").strip()
            gold_rest = gold[len("NO"):].lstrip(": ").strip()
            g = [t.strip() for t in gold_rest.split(",") if t.strip()]
            r = [t.strip() for t in rest.split(",") if t.strip()]
            if r == g:
                return 1.0
            if set(r) == set(g):
                return 0.5
            return 0.0
