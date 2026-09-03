from dataclasses import dataclass
import ast
import random

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


@dataclass
class SkipListSearchPathConfig(Config):
    n_nodes: int = 9
    max_level: int = 3

    def apply_difficulty(self, level):
        self.n_nodes = sround(self.n_nodes + 3 * level)
        self.max_level = sround(self.max_level + level)


def _forward_index(keys, heights, i, level, n):
    for j in range(i + 1, n):
        if heights[j] >= level:
            return j
    return -1


def _search_visited(keys, heights, max_level, target, n):
    result = []
    i = -1
    for level in range(max_level, 0, -1):
        nxt = _forward_index(keys, heights, i, level, n)
        while nxt != -1 and keys[nxt] < target:
            result.append(keys[nxt])
            i = nxt
            nxt = _forward_index(keys, heights, i, level, n)
    return result


class SkipListSearchPath(Task):
    summary = "Given explicit skip-list keys with tower heights and a target, output the visited keys in search order."

    config_cls = SkipListSearchPathConfig

    def generate_entry(self):
        cfg = self.config
        n = cfg.n_nodes
        max_level = cfg.max_level

        keys = sorted(random.sample(range(10, 10 + n * 3 + 30), n))
        heights = [random.randint(1, max_level) for _ in range(n)]
        heights[-1] = max_level

        lo = keys[0] - 1
        hi = keys[-1] + 6
        target = random.randint(lo, hi)

        visited = _search_visited(keys, heights, max_level, target, n)

        check = _search_visited(keys, heights, max_level, target, n)
        assert check == visited
        assert all(1 <= h <= max_level for h in heights)
        assert heights[-1] == max_level
        assert len(heights) == n and len(keys) == n
        assert all(keys[i] < keys[i + 1] for i in range(n - 1))
        for v in visited:
            assert v < target

        pairs = [[int(k), int(h)] for k, h in zip(keys, heights)]
        metadata = edict({
            "max_level": int(max_level),
            "nodes": pairs,
            "target": int(target),
            "visited": [int(v) for v in visited],
        })
        metadata.payload = {
            "query": (
                "A skip list holds integer keys in sorted order and has levels numbered "
                "1 (bottom) up to M (top). Each node lists its key followed by its height. "
                "At level L, a node's forward pointer reaches the next node whose height is "
                "at least L."
            ),
            "levels": int(max_level),
            "nodes": pairs,
            "target": int(target),
        }
        answer = "[" + ", ".join(str(int(v)) for v in visited) + "]"
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        node_txt = ", ".join(f"({k}: {h})" for k, h in metadata.nodes)
        return (
            f"{render_payload(metadata.payload)}\n\n"
            f"The keys with heights are: {node_txt}.\n\n"
            f"Run the standard skip-list predecessor search for target {metadata.target}. "
            f"Start just before the first node at the top level M; at each level from M down "
            f"to 1, while the forward pointer at that level reaches a node with key strictly "
            f"less than {metadata.target}, move to that node and record its key. "
            f"Report the recorded keys in the order they are visited as a bracket list, "
            f"e.g. [3, 7]. If no node is ever visited, answer [].\n\n"
            f"The answer is a bracket list of integers."
        )

    def score_answer(self, answer, entry):
        try:
            a = ast.literal_eval(str(answer).strip())
        except (SyntaxError, ValueError, TypeError):
            return 0.0
        if not isinstance(a, list):
            return 0.0
        gold = [int(v) for v in entry.metadata.visited]
        if len(a) != len(gold):
            return 0.0
        for x, g in zip(a, gold):
            try:
                if int(x) != g:
                    return 0.0
            except (TypeError, ValueError):
                return 0.0
        return 1.0


TASK_META = {'parent_source_id': None,
 'idea': 'skip_list_search_path (draw 1 of 2)',
 'hypothesis': 'W1-017',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/skip_list_search_path',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 1243922422,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
