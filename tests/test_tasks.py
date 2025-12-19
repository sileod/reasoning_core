
import pytest
from reasoning_core.template import Task, DATASETS
from reasoning_core.tasks import arithmetics, causal_reasoning, formal_maths, grammar, logic, planning, regex_following, sequential_induction, set_operations


@pytest.mark.parametrize("task_name, task_cls", DATASETS.items())
def test_task_consistency(task_name, task_cls):
    """
    Tests that for any given task, the score of the correct answer is 1.
    """
    if task_name in failing_tasks:
        pytest.xfail(f"Task {task_name} is known to fail due to environment issues (eprover).")
    
    task = task_cls()
    problem = task.generate_example()
    score = task.score_answer(problem.answer, problem)
    assert score == 1, f"Task {task_name} failed consistency check. For a generated problem, the score of the correct answer should be 1, but it was {score}."

@pytest.mark.parametrize("task_name, task_cls", DATASETS.items())
def test_set_level_over_c_and_seed_invariance(task_name, task_cls):
    task = task_cls()
    config = task.config

    # Ensure the attributes exist (fail loudly if not)
    assert hasattr(config, "c"), f"Task {task_name} has no config.c"
    assert hasattr(config, "seed"), f"Task {task_name} has no config.seed"

    # Fix known values
    config.c = 0.5
    config.seed = 42

    # Save original values
    c_before = config.c
    seed_before = config.seed

    # Apply set_level
    config.set_level(1)

    # Check invariants
    assert config.c == c_before, (
        f"Task {task_name}: config.c was modified by set_level "
        f"(before={c_before}, after={config.c})"
    )
    assert config.seed == seed_before, (
        f"Task {task_name}: config.seed was modified by set_level "
        f"(before={seed_before}, after={config.seed})"
    )