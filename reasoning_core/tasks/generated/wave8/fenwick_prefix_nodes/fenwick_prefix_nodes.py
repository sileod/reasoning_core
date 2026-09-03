import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


TASK_META = {'parent_source_id': None,
 'idea': 'fenwick_prefix_nodes (draw 1 of 2)',
 'hypothesis': 'W1-013',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/fenwick_prefix_nodes',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2872449786,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _fenwick_path(i):
    path = []
    while i > 0:
        path.append(i)
        i -= i & -i
    return path


@dataclass
class FenwickPrefixNodesConfig(Config):
    max_i: int = 64
    min_len: int = 2

    def apply_difficulty(self, level):
        self.max_i = sround(self.max_i * (2 ** level))
        self.min_len = sround(self.min_len + level)


class FenwickPrefixNodes(Task):
    summary = "Given a 1-based index i in a Fenwick tree, list the indices visited by the prefix sum query starting at i and repeatedly dropping the lowest set bit, descending to 0."

    config_cls = FenwickPrefixNodesConfig

    def generate_entry(self):
        cfg = self.config
        max_i = cfg.max_i
        while True:
            i = random.randint(1, max_i)
            path = _fenwick_path(i)
            if len(path) >= cfg.min_len:
                break
        answer = ",".join(map(str, path))
        metadata = edict({
            "payload": {
                "index": i,
                "size": max_i,
            },
        })
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        i = metadata.payload["index"]
        size = metadata.payload["size"]
        return (f"A Fenwick tree (binary indexed tree) is indexed 1..{size}. Its prefix sum query for "
                f"prefix {i} visits node {i}, adds its value, then subtracts the lowest set bit of the "
                f"current node ({'i & -i'}) to move to the next node, repeating until the node becomes 0.\n\n"
                f"List the nodes visited, starting at {i} and going down to (but not including) 0, in "
                f"decreasing order.\n\nThe answer is a comma-separated list of integers, e.g. \"8,0\" "
                f"would be wrong because it includes 0; give just the visited nonzero nodes, like \"12,8\".")

    def score_answer(self, answer, entry):
        try:
            given = [int(x.strip()) for x in answer.split(",") if x.strip() != ""]
        except (TypeError, ValueError):
            return 0.0
        gold = [int(x) for x in entry.answer.split(",")]
        return 1.0 if given == gold else 0.0
