"""Trace nested recursive calls with local parameters and return values."""

import random

from reasoning_core.template import Task, Entry, Config, edict

TASK_META = {'parent_source_id': None,
 'idea': 'call_stack_trace (draw 1 of 1)',
 'hypothesis': 'HV-025',
 'changes': 'new task in reasoning_core/tasks/generated/wave9/call_stack_trace',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3124932900,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _make_tree(rng, size):
    nodes = [{'value': rng.randrange(-9, 10), 'children': []} for _ in range(size)]
    for i in range(1, size):
        parent = rng.randrange(i)
        nodes[parent]['children'].append(i)
    return nodes


def _sum_truth(nodes, idx):
    total = nodes[idx]['value']
    for c in nodes[idx]['children']:
        total += _sum_truth(nodes, c)
    return total


def _render(nodes, idx):
    inner = ",".join(_render(nodes, c) for c in nodes[idx]['children'])
    if inner:
        return "%d(%s)" % (nodes[idx]['value'], inner)
    return "%d" % nodes[idx]['value']


class CallStackConfig(Config):
    n_nodes: int = 6
    max_depth: int = 3

    def apply_difficulty(self, level):
        self.n_nodes = 4 + level + (level // 2)
        self.max_depth = 2 + level


class CallStackTrace(Task):
    summary = ("Trace nested recursive and mutually-recursive calls tracking local "
               "parameters and return values, reporting the value returned by the "
               "frame of a queried tree node with varied subtree shapes and node "
               "scalar values.")
    config_cls = CallStackConfig
    task_version = 2

    def generate_entry(self):
        size = int(self.config.n_nodes)
        nodes = _make_tree(random, size)
        preorder = []
        def _walk(idx):
            preorder.append(idx)
            for c in nodes[idx]['children']:
                _walk(c)
        _walk(0)
        target_pos = random.randrange(size)
        target_node = preorder[target_pos]
        truth = _sum_truth(nodes, target_node)
        tree_repr = _render(nodes, 0)

        metadata = edict({
            'tree': tree_repr,
            'target': target_pos,
            'truth': int(truth),
            'size': int(size),
        })
        metadata.payload = {
            'tree': tree_repr,
            'target': target_pos,
        }
        return Entry(metadata=metadata, answer=str(int(truth)))

    def render_prompt(self, metadata):
        return ("A recursive function sum_subtree(n) adds a node's value to the "
                "sums returned by its children: it returns n.value + "
                "sum(sum_subtree(c) for c in n.children). A tree is written in "
                "prefix notation as value(child1,...,childk); leaves are bare "
                "values.\n\n"
                "Tree: %s\n\n"
                "Count the nodes of the tree in the order they are written "
                "(left to right, parent before its children); the first node "
                "written is position 0. What does sum_subtree return for the "
                "node at position %d?\n\n"
                "The answer is a single integer." %
                (metadata.payload['tree'], metadata.payload['target']))

    def score_answer(self, answer, entry):
        try:
            a = int(str(answer).strip())
        except (ValueError, TypeError):
            return 0.0
        return 1.0 if a == int(entry.answer) else 0.0
