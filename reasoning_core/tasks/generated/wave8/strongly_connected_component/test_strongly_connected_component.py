import random

from reasoning_core.tasks.generated.wave8.strongly_connected_component.strongly_connected_component import (
    StronglyConnectedComponent,
)


def test_gold_scoring():
    random.seed(1)
    task = StronglyConnectedComponent()
    for _ in range(50):
        ex = task.generate_example()
        assert task.score_answer(ex.answer, ex) == 1.0


def test_bad_answers():
    random.seed(2)
    task = StronglyConnectedComponent()
    for _ in range(50):
        ex = task.generate_example()
        assert task.score_answer("", ex) == 0.0
        assert task.score_answer("garbage", ex) == 0.0
        wrong = [x for x in range(ex.metadata["n"]) if x not in ex.metadata["members"]]
        assert task.score_answer(str(wrong), ex) == 0.0


def test_all_levels():
    for level in range(7):
        task = StronglyConnectedComponent()
        for _ in range(30):
            ex = task.generate_example(level=level)
            assert task.score_answer(ex.answer, ex) == 1.0
            members = ex.metadata["members"]
            assert len(members) >= 1
            assert ex.metadata["target"] in members


def test_answer_in_prompt_domain():
    random.seed(3)
    task = StronglyConnectedComponent()
    for _ in range(20):
        ex = task.generate_example()
        assert all(0 <= m < ex.metadata["n"] for m in ex.metadata["members"])
