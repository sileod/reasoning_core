"""Apply one key increase or decrease in a binary heap and output the key's final index."""

import random
from dataclasses import dataclass

from reasoning_core.template import Task, Entry, Config, edict


TASK_META = {'parent_source_id': None,
 'idea': 'heap_key_update (draw 1 of 2)',
 'hypothesis': 'W1-009',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave8/heap_key_update',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2479838412,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


def _is_valid_heap(arr):
    n = len(arr)
    for i in range(n):
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[i] > arr[l]:
            return False
        if r < n and arr[i] > arr[r]:
            return False
    return True


def _sift_down(arr, i):
    n = len(arr)
    while True:
        l = 2 * i + 1
        r = 2 * i + 2
        smallest = i
        if l < n and arr[l] < arr[smallest]:
            smallest = l
        if r < n and arr[r] < arr[smallest]:
            smallest = r
        if smallest == i:
            return
        arr[i], arr[smallest] = arr[smallest], arr[i]
        i = smallest


def _build_heap(n, base, stride):
    vals = list(range(base, base + n * stride, stride))
    random.shuffle(vals)
    arr = list(vals)
    for i in range(n // 2 - 1, -1, -1):
        _sift_down(arr, i)
    return arr


def _sift_up(arr, i):
    while i > 0:
        parent = (i - 1) // 2
        if arr[parent] <= arr[i]:
            return
        arr[i], arr[parent] = arr[parent], arr[i]
        i = parent


@dataclass
class HeapConfig(Config):
    n: int = 8

    def apply_difficulty(self, level):
        self.n = 4 + level * 3


class HeapKeyUpdate(Task):
    summary = "Apply one key increase or decrease in a binary min-heap and output the key's final index."
    config_cls = HeapConfig

    def generate_entry(self):
        n = self.config.n
        base = random.randrange(1, 100)
        stride = random.randrange(2, 6)
        for _ in range(400):
            arr = _build_heap(n, base, stride)
            assert _is_valid_heap(arr), "heap invariant violated"
            idx = random.randrange(n)
            original = arr[idx]
            ref = random.randrange(n)
            if idx == ref:
                continue
            # new value placed just above the reference node's value so the key
            # sifts toward that reference position; off-lattice keeps it distinct.
            offset = random.randrange(1, stride)
            new_val = arr[ref] + offset
            if new_val < 0:
                continue
            decrease = new_val < original
            break
        else:
            raise RuntimeError("could not produce a valid update")

        arr2 = list(arr)
        arr2[idx] = new_val
        if decrease:
            _sift_up(arr2, idx)
        else:
            _sift_down(arr2, idx)
        assert _is_valid_heap(arr2), "post-update heap invalid"

        final_pos = arr2.index(new_val)
        assert 0 <= final_pos < n, "final index out of range"
        assert int(new_val) >= 0, "new value must be non-negative"

        metadata = edict({
            "n": int(n),
            "heap": [int(x) for x in arr],
            "key_index": int(idx),
            "updated_value": int(new_val),
            "final_index": int(final_pos),
        })
        metadata.payload = {
            "heap": metadata.heap,
            "key_index": metadata.key_index,
            "updated_value": metadata.updated_value,
        }

        return Entry(metadata=metadata, answer=str(int(final_pos)))

    def render_prompt(self, metadata):
        heap = ", ".join(str(x) for x in metadata.payload["heap"])
        return (
            f"The list [{heap}] is the array representation of a binary min-heap "
            f"(each node is at most its children). The key at index "
            f"{metadata.payload['key_index']} (0-based) is changed to value "
            f"{metadata.payload['updated_value']}, and the heap property is restored by "
            f"sifting this key either up or down as needed. What is the final 0-based "
            f"index of this key after the update? The answer is an integer."
        )

    def score_answer(self, answer, entry):
        import ast
        try:
            parsed = ast.literal_eval(answer.strip())
        except Exception:
            return 0.0
        if isinstance(parsed, bool):
            return 0.0
        if isinstance(parsed, int):
            return 1.0 if parsed == int(entry.answer) else 0.0
        return 0.0
