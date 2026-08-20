from reasoning_core import get_task, list_tasks
import pytest


TASKS = [
    "controlled_code_execution",
    "boolean_propagation_search",
    "backtracking_search",
    "dynamic_programming",
    "fixpoint_iteration",
    "variable_elimination",
    "shift_reduce_parsing",
]


def test_generated_tasks_are_discovered():
    assert set(TASKS) <= set(list_tasks())


def test_generated_tasks_smoke():
    for name in TASKS:
        task = get_task(name)
        example = task.generate_example(level=0)
        assert example.prompt
        assert task.score_answer(example.answer, example) == 1
        assert task.score_answer("definitely wrong", example) != 1


@pytest.mark.parametrize(("name", "level"), [
    ("fixpoint_iteration", 6),
    ("spatial_folding", 0),
    ("spatial_folding", 5),
    ("spatial_folding", 6),
    ("variable_elimination", 0),
])
def test_balanced_batch_completes_at_previously_missing_levels(name, level):
    assert len(get_task(name).generate_balanced_batch(16, level=level, max_tokens=0)) == 16
