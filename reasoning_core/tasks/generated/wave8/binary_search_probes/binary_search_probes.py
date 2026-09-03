import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict, render_payload, stochastic_rounding as sround


@dataclass
class BinarySearchProbesConfig(Config):
    n: int = 16
    base_value: int = 8
    spread: int = 20

    def apply_difficulty(self, level):
        self.n = sround(self.n + 4 * level)
        self.base_value = sround(self.base_value + 2 * level)
        self.spread = sround(self.spread + 5 * level)


def _parse_answer(answer):
    cleaned = answer.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1].strip()
    if not cleaned:
        return []
    parts = cleaned.split(",")
    return [int(p.strip()) for p in parts]


def _binary_search_probes(arr, target):
    lo, hi = 0, len(arr) - 1
    probes = []
    while lo <= hi:
        mid = (lo + hi) // 2
        probes.append(mid)
        if arr[mid] == target:
            break
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return probes


class BinarySearchProbes(Task):
    summary = ("Given a sorted array and a target, output the indices probed by deterministic "
               "binary search: present-target exact hits, absent targets on either side, sizes "
               "spanning odd/even and small-to-large arrays.")
    config_cls = BinarySearchProbesConfig

    def generate_entry(self):
        n = self.config.n
        arr = sorted(random.sample(range(0, self.config.base_value + self.config.spread), n))
        if random.random() < 0.5:
            target = random.choice(arr)
        else:
            target = random.randint(0, self.config.base_value + self.config.spread - 1)
        probes = _binary_search_probes(arr, target)
        assert all(0 <= p < n for p in probes)
        target_found = target in arr
        if target_found:
            assert arr[probes[-1]] == target
        else:
            assert arr[probes[-1]] != target
        metadata = edict({
            "arr": arr,
            "target": target,
            "payload": {"array": arr, "target": int(target)},
        })
        answer = str(probes)
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        return (
            f"{render_payload(metadata.payload)}\n\n"
            "List the indices probed by deterministic binary search on this sorted array, "
            "probing the middle index (with the lower middle on even-length intervals), "
            "stopping when the target is found or the search space is exhausted. "
            "Give the answer as a list of the probed indices, e.g. [4, 1, 2]."
        )

    def score_answer(self, answer, entry):
        try:
            got = _parse_answer(answer)
        except (ValueError, TypeError):
            return 0.0
        gold = _parse_answer(entry.answer)
        return 1.0 if got == gold else 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'binary_search_probes (draw 1 of 2)',
 'hypothesis': 'W1-016',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/binary_search_probes',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 3549874681,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1500,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}
