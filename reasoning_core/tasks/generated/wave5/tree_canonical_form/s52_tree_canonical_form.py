import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround

TASK_META = {'parent_source_id': None,
 'idea': 'Add canonical naming of a rooted tree, so two descriptions can be '
         'compared.',
 'hypothesis': 'S52',
 'changes': "Ask for a tree's canonical form, or whether two trees are the "
            'same shape.',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'adapter_name': 'harness-link',
                'adapter_version': 'harness-link albert 0.3.0',
                'harness_name': 'opencode',
                'harness_version': '1.18.20',
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1691368962,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 50,
                             'timeout_seconds': 1200,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def canonical(children, node):
    if not children[node]:
        return "()"
    subs = sorted(canonical(children, c) for c in children[node])
    return "(" + "".join(subs) + ")"


def parse_parent_list(s):
    items = [x.strip() for x in s.strip()[1:-1].split(",")]
    return [int(x) for x in items if x != ""]


MAX_LEVEL = 6


def build_tree(parents):
    n = len(parents) + 1
    children = [[] for _ in range(n)]
    for i, p in enumerate(parents):
        children[p].append(i + 1)
    return children


def generate_parents(n, depth, rng):
    while True:
        parents = [0] * (n - 1)
        depths = [0] * n
        for i in range(1, n):
            p = rng.randrange(i)
            parents[i - 1] = p
            depths[i] = depths[p] + 1
        if max(depths) >= depth:
            return parents


def relabeled_parents(parents, rng):
    n = len(parents) + 1
    par = [-1] + list(parents)
    head = list(range(1, n))
    rng.shuffle(head)
    label = [0] + head
    inverse = [0] * n
    for v in range(n):
        inverse[label[v]] = v
    par2 = [-1] * n
    for v in range(1, n):
        par2[label[v]] = label[par[v]]
    return par2[1:]


class TreeCanonicalFormConfig(Config):
    n_nodes: int = 7
    depth: int = 3
    two_trees_prob: float = 0.2

    def apply_difficulty(self, level):
        self.n_nodes = sround(7 + 1.6 * level)
        self.depth = sround(3 + 0.9 * level)
        self.two_trees_prob = 0.2


class TreeCanonicalForm(Task):
    config_cls = TreeCanonicalFormConfig

    def generate_entry(self):
        n = int(self.config.n_nodes)
        depth = int(self.config.depth)
        parents = generate_parents(n, depth, random)
        children = build_tree(parents)

        parents2 = None
        if random.random() < self.config.two_trees_prob and n >= 4:
            if random.random() < 0.5:
                parents2 = relabeled_parents(parents, random)
            else:
                while True:
                    cand = generate_parents(n, depth, random)
                    if canonical(build_tree(cand), 0) != canonical(children, 0):
                        parents2 = cand
                        break

        can = canonical(children, 0)
        two_trees = parents2 is not None

        if two_trees:
            can2 = canonical(build_tree(parents2), 0)
            same = (can == can2)
            answer = "same shape" if same else "different shapes"
            payload = {
                "tree1_parent_list": "[" + ",".join(str(x) for x in parents) + "]",
                "tree2_parent_list": "[" + ",".join(str(x) for x in parents2) + "]",
            }
            metadata = edict({
                "n_nodes": n,
                "depth": depth,
                "same": same,
                "two_trees": True,
                "prompt_kind": answer,
                "can1": can,
                "can2": can2,
            })
            metadata.payload = payload
        else:
            payload = {"parent_list": "[" + ",".join(str(x) for x in parents) + "]"}
            metadata = edict({
                "n_nodes": n,
                "depth": depth,
                "two_trees": False,
                "canonical": can,
            })
            metadata.payload = payload
            answer = can

        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        if metadata.two_trees:
            payload = render_payload(metadata.payload)
            return (
                payload
                + "\n\nThe parent list of a rooted tree gives, for each node "
                "indexed from 1 to n in order, the index of its parent (the "
                "root has parent 0). The canonical form writes a leaf as () "
                "and an internal node as (children sorted lexicographically "
                "by their own canonical strings).\n"
                "Are these two trees the same shape (i.e. identical canonical "
                "forms)? The answer is exactly 'same shape' or 'different "
                "shapes'."
            )
        payload = render_payload(metadata.payload)
        return (
            payload
            + "\n\nThe parent list of a rooted tree gives, for each node "
            "indexed from 1 to n in order, the index of its parent (the "
            "root has parent 0). Write the canonical form of this tree: a "
            "leaf is (), and an internal node is (its children sorted "
            "lexicographically by their own canonical strings and "
            "concatenated). The answer is that exact canonical string."
        )

    def score_answer(self, answer, entry):
        if entry.metadata.two_trees:
            target = entry.metadata.prompt_kind
            return 1.0 if answer == target else 0.0
        return 1.0 if answer == entry.metadata.canonical else 0.0

    def distractor_candidates(self, entry):
        return []
