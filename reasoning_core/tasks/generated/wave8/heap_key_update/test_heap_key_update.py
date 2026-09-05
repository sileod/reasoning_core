import random

from reasoning_core.tasks.generated.wave8.heap_key_update.heap_key_update import (
    HeapKeyUpdate, _is_valid_heap
)


def _simulate_answer(metadata):
    arr = list(metadata["heap"])
    idx = metadata["key_index"]
    new_val = metadata["updated_value"]
    old = arr[idx]
    arr[idx] = new_val
    n = len(arr)
    if new_val < old:
        i = idx
        while i > 0:
            parent = (i - 1) // 2
            if arr[parent] <= arr[i]:
                break
            arr[i], arr[parent] = arr[parent], arr[i]
            i = parent
    else:
        i = idx
        while True:
            l = 2 * i + 1
            r = 2 * i + 2
            smallest = i
            if l < n and arr[l] < arr[smallest]:
                smallest = l
            if r < n and arr[r] < arr[smallest]:
                smallest = r
            if smallest == i:
                break
            arr[i], arr[smallest] = arr[smallest], arr[i]
            i = smallest
    return arr.index(new_val)


def test_generates_and_scores():
    random.seed(1)
    task = HeapKeyUpdate()
    for level in range(7):
        task.config.set_level(level)
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0
        assert int(ex.answer) == _simulate_answer(ex.metadata)


def test_junk_scores_zero():
    random.seed(2)
    task = HeapKeyUpdate()
    ex = task.generate_example()
    for junk in ["", "abc", "1.5", "-3", "None"]:
        assert task.score_answer(junk, ex) == 0.0
