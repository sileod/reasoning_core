from reasoning_core import get_task, list_tasks


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
