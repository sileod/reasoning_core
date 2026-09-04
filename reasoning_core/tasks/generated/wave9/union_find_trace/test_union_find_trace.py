import random

from reasoning_core.tasks.generated.wave9.union_find_trace.union_find_trace import UnionFindTrace, UFConfig


def test_union_find_trace_validate():
    random.seed(1602037825)
    task = UnionFindTrace()
    x = task.generate_example()
    assert task.score_answer(x.answer, x) == 1.0


def test_score_rejects_junk_and_empty():
    random.seed(12345)
    task = UnionFindTrace()
    x = task.generate_example()
    assert task.score_answer("", x) < 1.0
    assert task.score_answer("not a number here", x) < 1.0


def test_score_rejects_wrong_length_and_wrong_value():
    random.seed(999)
    task = UnionFindTrace()
    x = task.generate_example()
    ref = x.answer.split()
    wrong = list(ref)
    wrong[0] = str(1 - int(wrong[0]))
    assert task.score_answer(" ".join(wrong), x) < 1.0
    assert task.score_answer("1", x) < 1.0


def test_difficulty_changes():
    c = UFConfig()
    base_edges = c.n_edges
    c.set_level(6)
    assert c.n_edges > base_edges


def test_answers_vary_across_examples():
    random.seed(7)
    task = UnionFindTrace()
    answers = {task.generate_example().answer for _ in range(30)}
    assert len(answers) >= 5
